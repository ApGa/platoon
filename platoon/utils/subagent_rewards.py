from __future__ import annotations

from typing import Any

from platoon.episode.trajectory import TrajectoryCollection


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


def propogate_root_success(
    trajectory_collection: dict[str, Any] | TrajectoryCollection,
) -> dict[str, Any] | TrajectoryCollection:
    """Rewrite recursive rollout rewards so all trajectories use root success."""
    trajectories = _get_trajectories(trajectory_collection)
    if not trajectories:
        return trajectory_collection

    _, root_trajectory = next(iter(trajectories.items()))
    root_steps = _get_steps(root_trajectory)
    root_success = _get_trajectory_reward(root_trajectory)
    if root_steps:
        root_success = float(_get_step_reward_misc(root_steps[-1]).get("reward/success", root_success))

    for trajectory in trajectories.values():
        _set_trajectory_reward(trajectory, root_success)
        steps = _get_steps(trajectory)
        if steps:
            _get_step_reward_misc(steps[-1])["reward/success"] = root_success
        for step in steps:
            reward_misc = _get_step_reward_misc(step)
            launched = float(reward_misc.get("reward/subagent_launched", 0.0))
            if launched > 0:
                reward_misc["reward/subagent_succeeded"] = launched * root_success

    return trajectory_collection
