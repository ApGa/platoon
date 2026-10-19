"""Core's reward processor for recursive agents, and the subagent reward finalizer that ``single`` applies.

``delegation_capped`` is the reward processor ``platoon-delegation`` (``reward_processor: platoon-delegation``, which
``textcraft-synth@1`` sets): a trajectory's score is its ``reward/success`` plus a delegation bonus capped at
:data:`DELEGATION_REWARD_CAP` (0, so the score is the success alone, as before). The ``reward/*`` components are
summed over the trajectory's steps: the trace puts the agent's grade on its last step as ``reward/success``, and
``ctx.agents.spawn`` puts ``reward/subagent_launched`` / ``reward/subagent_succeeded`` on each step that launched
children. Platoon's earlier recursive environments wrote all three on every step, zeros included, so each component
defaults to 0 here (``reward/success`` to the trajectory's reward): every trajectory reports every component and the
rollout averages match theirs.

``zero_subagent_rewards`` is the finalizer of ``skip_subagent_reward_computation`` (rollout config): the ``single``
program adds it before ``propagate_root_success``, the order of platoon's earlier recursive rollouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platoon.trajectory import TrajectoryCollection

DELEGATION_REWARD_CAP = 0.0
COMPONENTS = ("reward/success", "reward/subagent_launched", "reward/subagent_succeeded")


def delegation_capped(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """``platoon-delegation``: the trajectory's success plus the capped delegation bonus, and every component."""

    rewards: dict[str, float] = {}
    for step in trajectory.get("steps", ()):
        reward_misc = (step.get("misc") or {}).get("reward_misc") or {}
        for key, value in reward_misc.items():
            if key.startswith("reward/") and isinstance(value, (int, float)):
                rewards[key] = rewards.get(key, 0.0) + float(value)
    rewards.setdefault("reward/success", float(trajectory.get("reward", 0.0)))
    for key in COMPONENTS[1:]:
        rewards.setdefault(key, 0.0)
    score = rewards["reward/success"]
    launched = rewards["reward/subagent_launched"]
    if launched > 0:
        score += DELEGATION_REWARD_CAP * rewards["reward/subagent_succeeded"] / launched
    return score, rewards


def zero_subagent_rewards(collection: TrajectoryCollection) -> None:
    """Give every subagent trajectory reward 0 and count no launched child as a success
    (``skip_subagent_reward_computation``), as platoon's earlier environments scored subagent tasks."""

    for trajectory in collection.trajectories.values():
        if trajectory.parent_info is not None:
            trajectory.reward = 0.0
            if trajectory.steps:
                trajectory.steps[-1].misc.setdefault("reward_misc", {})["reward/success"] = 0.0
        for step in trajectory.steps:
            reward_misc = step.misc.get("reward_misc") or {}
            if "reward/subagent_succeeded" in reward_misc:
                reward_misc["reward/subagent_succeeded"] = 0.0


__all__ = ["COMPONENTS", "DELEGATION_REWARD_CAP", "delegation_capped", "zero_subagent_rewards"]
