"""Judges: the model calls that judge agents rather than solve the task, and what a rollout records about them.

A rollout runs its judges unless its ``judges`` setting is off (``RolloutConfig.judges``, which ``platoon eval``
sets from ``inference.judges``).  The judges are verifier agents (``agents.child_grading: verifier``) with the
behavior judges that run beside them, parent-usage judges, ``judge``-role agents
(``EpisodeContext.judge_endpoint``) and ``judge`` calls.  ``policy`` and ``aux`` calls are the agents solving the
task, and environment graders are the measurement: they always run.

What the trajectories record, under these misc keys:

* :data:`JUDGES_MISC_KEY` on the root trajectory of a rollout whose judges are off: ``{"enabled": false}``.
* :data:`CHILD_GRADING_MISC_KEY` on a child that shares the episode's outcome although its environment grades
  children by verifier: ``{"configured": "verifier", "applied": "none", "reason": ...}``, the reason being one of
  :data:`GRADING_FALLBACK_REASONS`.
* :data:`JUDGES_REFUSED_MISC_KEY` on an agent that asked for a judge while they were off: ``{"count", "first"}``.

Standard library only, so the gateway, the trace and the evaluation report all read these without importing
each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from platoon.utils.trajectory_status import SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY

# The role of a judge agent (``EpisodeContext.judge_endpoint``); its calls are never trained.
JUDGE_ROLE = "judge"
# The role of a verifier agent (``platoon.harness.verify.VERIFIER_ROLE``).
VERIFIER_ROLE = "verifier"

JUDGES_MISC_KEY = "judges"
CHILD_GRADING_MISC_KEY = "child_grading"
JUDGES_REFUSED_MISC_KEY = "judges_refused"
# The trace's per-agent record in trajectory misc (``platoon.gateway.trace.AGENT_MISC_KEY``).
_AGENT_MISC_KEY = "agent"

# Why a child of an environment that grades children by verifier was not judged, and so shares the episode's
# outcome: the run's judges are off; ``enable_subagent_verifiers`` is false in the rollout config;
# ``agents.verifier.enabled`` is false; or ``agents.verifier.max_steps`` is 0.
JUDGES_OFF = "judges_off"
GRADING_FALLBACK_REASONS = (JUDGES_OFF, "verifiers_off", "verifier_disabled", "verifier_max_steps_0")


def grading_fallback(reason: str, *, configured: str = "verifier") -> dict[str, str]:
    """The :data:`CHILD_GRADING_MISC_KEY` record of a child graded ``none`` instead of ``configured``."""

    if reason not in GRADING_FALLBACK_REASONS:
        raise ValueError(f"Unknown grading fallback {reason!r}; expected one of {GRADING_FALLBACK_REASONS}")
    return {"configured": configured, "applied": "none", "reason": reason}


def _misc(trajectory: Any) -> Mapping[str, Any]:
    misc = trajectory.get("misc") if isinstance(trajectory, Mapping) else getattr(trajectory, "misc", None)
    return misc if isinstance(misc, Mapping) else {}


def _task_misc(trajectory: Any) -> Mapping[str, Any]:
    task = trajectory.get("task") if isinstance(trajectory, Mapping) else getattr(trajectory, "task", None)
    misc = task.get("misc") if isinstance(task, Mapping) else getattr(task, "misc", None)
    return misc if isinstance(misc, Mapping) else {}


def _trajectories(collection: Any) -> list[Any]:
    trajectories = (
        collection.get("trajectories") if isinstance(collection, Mapping) else getattr(collection, "trajectories", None)
    )
    return list(trajectories.values()) if isinstance(trajectories, Mapping) else []


def judges_enabled(collection: Any) -> bool:
    """Whether a rollout's judges ran: false when its root trajectory records them off (a
    ``TrajectoryCollection`` or its dump)."""

    trajectories = _trajectories(collection)
    record = _misc(trajectories[0]).get(JUDGES_MISC_KEY) if trajectories else None
    return not (isinstance(record, Mapping) and record.get("enabled") is False)


def is_judge_trajectory(trajectory: Any) -> bool:
    """Whether a trajectory (or its dump) is a verifier's or a judge's: a verifier agent, one of its helpers, or a
    ``judge``-role agent.  Such agents judge the policy's agents; they never train, and evaluation reports count
    them apart from the policy's."""

    if _task_misc(trajectory).get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
        return True
    misc = _misc(trajectory)
    if misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
        return True
    agent = misc.get(_AGENT_MISC_KEY)
    return isinstance(agent, Mapping) and agent.get("role") in (VERIFIER_ROLE, JUDGE_ROLE)


__all__ = [
    "CHILD_GRADING_MISC_KEY",
    "GRADING_FALLBACK_REASONS",
    "JUDGES_MISC_KEY",
    "JUDGES_OFF",
    "JUDGES_REFUSED_MISC_KEY",
    "JUDGE_ROLE",
    "VERIFIER_ROLE",
    "grading_fallback",
    "is_judge_trajectory",
    "judges_enabled",
]
