"""Backend-neutral trajectory terminal-status helpers."""

from __future__ import annotations

from typing import Any

TRAJECTORY_CANCELLED_MISC_KEY = "trajectory_cancelled"
TRAJECTORY_TIMED_OUT_MISC_KEY = "trajectory_timed_out"
TRAJECTORY_INVALID_MISC_KEY = "trajectory_invalid"


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

    misc = (
        trajectory.get("misc", {})
        if isinstance(trajectory, dict)
        else getattr(trajectory, "misc", {})
    )
    return isinstance(misc, dict) and bool(
        misc.get(TRAJECTORY_INVALID_MISC_KEY)
    )


def trajectory_was_interrupted(trajectory: Any) -> bool:
    """Return whether the trajectory's policy tokens are ineligible."""

    return (
        trajectory_was_cancelled(trajectory)
        or trajectory_was_timed_out(trajectory)
        or trajectory_was_invalid(trajectory)
    )
