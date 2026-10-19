"""Backend-neutral trajectory terminal-status helpers."""

from __future__ import annotations

from typing import Any

TRAJECTORY_CANCELLED_MISC_KEY = "trajectory_cancelled"
TRAJECTORY_TIMED_OUT_MISC_KEY = "trajectory_timed_out"
TRAJECTORY_INVALID_MISC_KEY = "trajectory_invalid"
# A valid failed outcome, distinct from infrastructure errors/cancellation.
# Recursive graders must not replace its zero reward with another score.
TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY = "trajectory_forced_zero_reward"
# Training-exclusion markers the datum builder honors.  ``exclude_from_training``
# drops a trajectory entirely (synthetic verifier and judge trajectories);
# ``exclude_from_policy_training`` drops only its policy datums, keeping its
# reward and statistics.  A task whose misc carries
# ``subagent_reward_verifier_task`` is a verifier's and is never trained either.
EXCLUDE_FROM_TRAINING_MISC_KEY = "exclude_from_training"
EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY = "exclude_from_policy_training"
SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY = "subagent_reward_verifier_task"
# Where a trajectory whose agent has an outcome of its own records it (``platoon.gateway.trace``): its status,
# rewards and reason.
OUTCOME_MISC_KEY = "outcome"
# The key of a scored outcome's record (``misc["outcome"]``) that names the reward key it trained: the name its grader
# reported the reward under (``Outcome.primary``), or null for a score platoon gave itself, which no grader reported
# (``platoon.envs.service.PLATOON_ZERO_KEY``).
OUTCOME_REWARD_KEY = "reward_key"

# The untrainable-outcome markers on a trajectory's misc, for trajectories that record no outcome status.
_STATUS_MARKERS = (
    (TRAJECTORY_INVALID_MISC_KEY, "invalid"),
    (TRAJECTORY_TIMED_OUT_MISC_KEY, "timeout"),
    (TRAJECTORY_CANCELLED_MISC_KEY, "cancelled"),
)


def trajectory_has_forced_zero_reward(trajectory: Any) -> bool:
    misc = trajectory.get("misc", {}) if isinstance(trajectory, dict) else getattr(trajectory, "misc", {})
    return isinstance(misc, dict) and bool(misc.get(TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY))


def trajectory_was_cancelled(trajectory: Any) -> bool:
    """Return whether a trajectory was cancelled before it completed.

    New rollouts carry an explicit misc marker.  The error-message fallback
    keeps older serialized/event-replayed trajectories safe as well.
    """

    if isinstance(trajectory, dict):
        misc = trajectory.get("misc", {})
        error_message = trajectory.get("error_message")
    else:
        misc = getattr(trajectory, "misc", {})
        error_message = getattr(trajectory, "error_message", None)

    if isinstance(misc, dict) and bool(misc.get(TRAJECTORY_CANCELLED_MISC_KEY)):
        return True
    return isinstance(error_message, str) and (
        "CancelledError" in error_message or "Episode cancelled" in error_message
    )


def trajectory_was_timed_out(trajectory: Any) -> bool:
    """Return whether a step-level deadline interrupted the trajectory."""

    if isinstance(trajectory, dict):
        misc = trajectory.get("misc", {})
        error_message = trajectory.get("error_message")
    else:
        misc = getattr(trajectory, "misc", {})
        error_message = getattr(trajectory, "error_message", None)

    if isinstance(misc, dict) and bool(misc.get(TRAJECTORY_TIMED_OUT_MISC_KEY)):
        return True
    return isinstance(error_message, str) and (
        "Episode timed out" in error_message or "\nTimeoutError:" in error_message
    )


def trajectory_was_invalid(trajectory: Any) -> bool:
    """Return whether an environment marked the completed result invalid."""

    misc = trajectory.get("misc", {}) if isinstance(trajectory, dict) else getattr(trajectory, "misc", {})
    return isinstance(misc, dict) and bool(misc.get(TRAJECTORY_INVALID_MISC_KEY))


def trajectory_was_interrupted(trajectory: Any) -> bool:
    """Return whether the trajectory's policy tokens are ineligible."""

    return (
        trajectory_was_cancelled(trajectory)
        or trajectory_was_timed_out(trajectory)
        or trajectory_was_invalid(trajectory)
    )


def trained_reward_key(trajectory: Any) -> str | None:
    """The reward key a trajectory's scored outcome trained (``misc["outcome"]["reward_key"]``); None for an outcome
    that is not scored, a score platoon gave itself, and a trajectory that records no key."""

    misc = trajectory.get("misc") if isinstance(trajectory, dict) else getattr(trajectory, "misc", None)
    outcome = misc.get(OUTCOME_MISC_KEY) if isinstance(misc, dict) else None
    if not isinstance(outcome, dict) or outcome.get("status") != "scored":
        return None
    key = outcome.get(OUTCOME_REWARD_KEY)
    return key if isinstance(key, str) and key else None


def outcome_status(trajectory: Any) -> str | None:
    """The status of the outcome a trajectory recorded (``misc["outcome"]["status"]``), else the one its untrainable
    marker implies, else None.  Never read from the error message."""

    misc = trajectory.get("misc") if isinstance(trajectory, dict) else getattr(trajectory, "misc", None)
    if not isinstance(misc, dict):
        return None
    outcome = misc.get(OUTCOME_MISC_KEY)
    if isinstance(outcome, dict) and isinstance(outcome.get("status"), str):
        return outcome["status"]
    return next((status for key, status in _STATUS_MARKERS if misc.get(key)), None)
