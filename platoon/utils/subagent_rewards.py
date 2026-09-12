from __future__ import annotations

from typing import Any

from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.trajectory_status import trajectory_has_forced_zero_reward

SUBAGENT_DELEGATION_REWARD_MISC_KEY = "subagent_delegation_reward"


def _get_trajectories(trajectory_collection: dict[str, Any] | TrajectoryCollection) -> dict[str, Any]:
    return (
        trajectory_collection["trajectories"]
        if isinstance(trajectory_collection, dict)
        else trajectory_collection.trajectories
    )


def _get_trajectory_reward(trajectory: Any) -> float:
    return float(trajectory["reward"] if isinstance(trajectory, dict) else trajectory.reward)


def _set_trajectory_reward(trajectory: Any, reward: float) -> None:
    if isinstance(trajectory, dict):
        trajectory["reward"] = reward
    else:
        trajectory.reward = reward


def _get_steps(trajectory: Any) -> list[Any]:
    return trajectory.get("steps", []) if isinstance(trajectory, dict) else trajectory.steps


def _get_step_reward_misc(step: Any) -> dict[str, Any]:
    if isinstance(step, dict):
        return step.setdefault("misc", {}).setdefault("reward_misc", {})
    if step.misc is None:
        step.misc = {}
    return step.misc.setdefault("reward_misc", {})


def _get_trajectory_misc(trajectory: Any) -> dict[str, Any]:
    if isinstance(trajectory, dict):
        return trajectory.setdefault("misc", {})
    return trajectory.misc


def _get_parent_id(trajectory: Any) -> str | None:
    parent_info = trajectory.get("parent_info") if isinstance(trajectory, dict) else trajectory.parent_info
    if isinstance(parent_info, dict):
        parent_id = parent_info.get("id")
    else:
        parent_id = getattr(parent_info, "id", None)
    return str(parent_id) if parent_id is not None else None


def _is_excluded_from_training(trajectory: Any) -> bool:
    return bool(_get_trajectory_misc(trajectory).get("exclude_from_training"))


def _get_base_success(trajectory: Any) -> float:
    """Return success before any delegation bonus is applied."""

    if trajectory_has_forced_zero_reward(trajectory):
        return 0.0

    steps = _get_steps(trajectory)
    if steps:
        reward_misc = _get_step_reward_misc(steps[-1])
        success = reward_misc.get("reward/success")
        if isinstance(success, (int, float)):
            return float(success)
    return _get_trajectory_reward(trajectory)


def add_direct_subagent_delegation_rewards(
    trajectory_collection: dict[str, Any] | TrajectoryCollection,
    coefficient: float,
) -> dict[str, Any] | TrajectoryCollection:
    """Attach an Oolong-style direct-child delegation bonus to every trajectory.

    Only direct, trainable child trajectories contribute.  In particular,
    verifier trajectories are excluded because they are marked
    ``exclude_from_training``.  Each child contributes its base
    ``reward/success`` before its own delegation bonus, so nested delegation is
    rewarded independently at every depth without recursively compounding the
    same success up the tree.
    """

    coefficient = float(coefficient)
    if coefficient < 0:
        raise ValueError("subagent delegation reward coefficient must be non-negative")

    trajectories = _get_trajectories(trajectory_collection)
    trainable = {
        str(trajectory_id): trajectory
        for trajectory_id, trajectory in trajectories.items()
        if not _is_excluded_from_training(trajectory)
    }
    child_scores: dict[str, list[float]] = {trajectory_id: [] for trajectory_id in trainable}

    for child_id, child in trainable.items():
        parent_id = _get_parent_id(child)
        if parent_id is None or parent_id not in trainable:
            continue
        # Iterating the trajectory mapping counts each completed child ID once,
        # including when several children were launched concurrently.
        child_scores[parent_id].append(_get_base_success(child))

    for trajectory_id, trajectory in trainable.items():
        scores = child_scores[trajectory_id]
        launched = len(scores)
        succeeded = sum(scores)
        success_rate = succeeded / launched if launched else 0.0
        _get_trajectory_misc(trajectory)[SUBAGENT_DELEGATION_REWARD_MISC_KEY] = {
            "coefficient": coefficient,
            "launched": float(launched),
            "succeeded": float(succeeded),
            "success_rate": float(success_rate),
            "bonus": 0.0 if trajectory_has_forced_zero_reward(trajectory) else float(coefficient * success_rate),
        }

    return trajectory_collection


def propagate_root_success(
    trajectory_collection: dict[str, Any] | TrajectoryCollection,
) -> dict[str, Any] | TrajectoryCollection:
    """Propagate root success, preserving explicit terminal zero outcomes."""
    trajectories = _get_trajectories(trajectory_collection)
    if not trajectories:
        return trajectory_collection

    _, root_trajectory = next(iter(trajectories.items()))
    root_success = _get_base_success(root_trajectory)

    for trajectory in trajectories.values():
        success = 0.0 if trajectory_has_forced_zero_reward(trajectory) else root_success
        _set_trajectory_reward(trajectory, success)
        steps = _get_steps(trajectory)
        if steps:
            _get_step_reward_misc(steps[-1])["reward/success"] = success
        for step in steps:
            reward_misc = _get_step_reward_misc(step)
            launched = float(reward_misc.get("reward/subagent_launched", 0.0))
            if launched > 0:
                reward_misc["reward/subagent_succeeded"] = launched * success

    return trajectory_collection


# Backward compatibility for external plugins importing the historical typo.
# In-repository code and configuration use ``propagate_root_success``.
propogate_root_success = propagate_root_success
