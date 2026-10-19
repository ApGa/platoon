"""Error and message detection for trajectory-to-training conversion.

Detection operates on the sampled completion rather than on one step record,
because one completion may be represented by several steps while parallel tool
results arrive.  Core detects the harness-agnostic conventions: CodeAct records
execution failures on a step (``error``, or an ``output`` with a traceback),
and tool observations (``observation_events``) report ``is_error``.

A step keeps these fields where its harness put them.  Legacy steps carried
them as step fields; a gateway step (:mod:`platoon.gateway.trace`, every
rollout a harness runs through the gateway) keeps the harness's annotations of
its model call in the step's ``misc`` (``ctx.annotate_step``): CodeAct
annotates ``error`` and ``output`` there, and an OpenHands harness its
``observation_events``.  The detector reads both places.

Harness-specific detectors extend both sets through the
``trajectory.error_completion_ids`` and ``trajectory.message_completion_ids``
hooks (see :mod:`platoon.hooks`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from platoon.hooks import hooks

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


def _event_reports_error(event: object) -> bool:
    """Return whether a serialized or live tool observation reports an error."""

    observation = _field(event, "observation")
    if observation is not None and _observation_reports_error(observation):
        return True
    # A bare observation record, not wrapped in an observation event.
    return "observation" in _normalized_kind(event) and _field(event, "is_error") is True


def _step_records(step: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Where a step's fields live: the step itself (legacy steps), then its ``misc`` (a gateway step's
    annotations of its model call)."""

    yield step
    misc = step.get("misc")
    if isinstance(misc, Mapping):
        yield misc


def step_reports_error(step: Mapping[str, Any]) -> bool:
    """Detect CodeAct execution errors and tool observations that report errors, on the step or in its misc."""

    for record in _step_records(step):
        # CodeAct's explicit error convention: the cell's error, or an output with a traceback in it.
        if record.get("error"):
            return True
        output = record.get("output")
        if isinstance(output, str) and "traceback" in output.lower():
            return True
        observation_events = record.get("observation_events")
        if any(_event_reports_error(event) for event in _iter_event_records(observation_events, "observation_events")):
            return True
    return False


def detected_error_completion_ids(steps: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return erroneous completion IDs without making a reward decision.

    Exporters use this to build a token-aligned side channel.  Whether those
    tokens are suppressed is intentionally deferred until group-centered
    policy credit is known.
    """

    steps = list(steps)
    detected = {
        completion_id
        for step in steps
        if step_reports_error(step) and (completion_id := completion_id_for_step(step)) is not None
    }
    for detector in hooks("trajectory.error_completion_ids"):
        detected |= detector(steps)
    return detected


def detected_message_completion_ids(steps: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return completions that only sent a message and executed no action.

    Telling a message from an action needs the harness's own record of the
    completion, so detection comes entirely from the
    ``trajectory.message_completion_ids`` hooks.
    """

    steps = list(steps)
    detected: set[str] = set()
    for detector in hooks("trajectory.message_completion_ids"):
        detected |= detector(steps)
    return detected
