"""Typed error detection shared by trajectory-to-training converters.

CodeAct records execution failures directly on a trajectory step.  OpenHands
instead records them as typed events, and one LLM completion may be represented
by several steps while parallel tool results arrive.  This module keeps those
serialization details out of the AReaL and Tinker converters and makes error
filtering operate on the sampled completion rather than on one event record.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Private side channel carried only from trajectory exporters to the
# group-centering workflow. It must be removed before actor/server dispatch.
ERROR_ACTION_MASK_KEY = "_platoon_error_action_mask"


def completion_id_for_step(step: Mapping[str, Any]) -> str | None:
    """Return the exported LLM completion ID attached to ``step``."""

    misc = step.get("misc")
    if not isinstance(misc, Mapping):
        return None
    action_misc = misc.get("action_misc")
    if not isinstance(action_misc, Mapping):
        return None
    completion_id = action_misc.get("completion_id")
    return completion_id if isinstance(completion_id, str) and completion_id else None


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _event_kind(event: object) -> str:
    kind = _field(event, "kind")
    if kind is None:
        kind = _field(event, "type")
    if kind is None:
        # Supporting model instances as well as their serialized dictionaries
        # makes the detector useful at both rollout and replay boundaries.
        kind = type(event).__name__
    if hasattr(kind, "value"):
        kind = kind.value
    return str(kind)


def _normalized_kind(event: object) -> str:
    return "".join(character for character in _event_kind(event).lower() if character.isalnum())


def _iter_event_records(value: object, wrapper_name: str) -> Iterable[object]:
    """Yield events from list, RootModel, and named-wrapper serializations."""

    if value is None or isinstance(value, (str, bytes, bytearray)):
        return
    if isinstance(value, Sequence):
        for item in value:
            yield from _iter_event_records(item, wrapper_name)
        return

    kind = _field(value, "kind")
    event_type = _field(value, "type")
    if kind is not None or event_type is not None:
        yield value
        return

    # Platoon has persisted both bare lists and small Pydantic/dataclass
    # wrappers over time.  Prefer known wrapper fields, then conservatively
    # descend through an untyped mapping/RootModel.
    for name in (wrapper_name, "events", "root"):
        nested = _field(value, name)
        if nested is not None and nested is not value:
            yield from _iter_event_records(nested, wrapper_name)
            return

    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _iter_event_records(nested, wrapper_name)


def _observation_reports_error(observation: object) -> bool:
    if observation is None:
        return False
    is_error = _field(observation, "is_error")
    if isinstance(is_error, bool):
        return is_error
    # Observation payloads can themselves be RootModels/wrappers.  Restrict
    # recursion to containers so arbitrary strings cannot look like errors.
    if isinstance(observation, Mapping):
        return any(_observation_reports_error(value) for value in observation.values())
    if isinstance(observation, Sequence) and not isinstance(observation, (str, bytes, bytearray)):
        return any(_observation_reports_error(value) for value in observation)
    root = _field(observation, "root")
    return root is not None and root is not observation and _observation_reports_error(root)


def openhands_event_reports_error(event: object) -> bool:
    """Return whether a serialized or live OpenHands event is a tool error."""

    kind = _normalized_kind(event)
    if kind == "agenterrorevent":
        # Invalid tool names/arguments and action-conversion failures use this
        # event rather than an ObservationEvent.
        return True
    if (
        kind == "userrejectobservation"
        and _field(event, "rejection_source") == "hook"
    ):
        # A hook rejection is a scaffold/policy error, unlike an interactive
        # user declining a confirmation request.
        return True

    observation = _field(event, "observation")
    if observation is not None and _observation_reports_error(observation):
        return True

    # Accept a bare Observation model/event too.  Normal OpenHands persistence
    # wraps it in ObservationEvent, but this form is useful for old replays and
    # lightweight harnesses.
    return "observation" in kind and _field(event, "is_error") is True


def step_reports_error(step: Mapping[str, Any]) -> bool:
    """Detect legacy CodeAct and typed OpenHands execution errors."""

    # Preserve CodeAct's explicit error convention.
    if step.get("error"):
        return True
    output = step.get("output")
    if isinstance(output, str) and "traceback" in output.lower():
        return True

    observation_events = step.get("observation_events")
    return any(
        openhands_event_reports_error(event)
        for event in _iter_event_records(observation_events, "observation_events")
    )


def trajectory_has_positive_error_credit(
    trajectory_reward: float,
    reward_metrics: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether erroneous actions would receive positive policy credit.

    ``reward/success`` is deliberately preferred to the shaped trajectory
    reward.  Efficiency penalties and auxiliary rewards can move the shaped
    value below or above a historical threshold without changing whether the
    task itself succeeded.  Older CodeAct reward processors may not provide
    components, so positive raw reward remains the backwards-compatible
    fallback.  Zero/negative trajectories retain their erroneous completions
    as useful negative signal.
    """

    if reward_metrics is not None:
        for key in ("reward/success", "success"):
            if key not in reward_metrics:
                continue
            try:
                return float(reward_metrics[key]) > 0.0
            except (TypeError, ValueError):
                break
    return float(trajectory_reward) > 0.0


def detected_error_completion_ids(steps: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return erroneous completion IDs without making a reward decision.

    Exporters use this to build a token-aligned side channel.  Whether those
    tokens are suppressed is intentionally deferred until group-centered
    policy credit is known.
    """

    return {
        completion_id
        for step in steps
        if step_reports_error(step)
        and (completion_id := completion_id_for_step(step)) is not None
    }
