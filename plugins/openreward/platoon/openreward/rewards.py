from __future__ import annotations

from platoon.agents.actions.subagent import SUBAGENT_REWARD_JUDGMENT_MISC_KEY
from platoon.utils.subagent_rewards import SUBAGENT_DELEGATION_REWARD_MISC_KEY

from platoon.openreward.constants import OPENREWARD_ENVIRONMENT_LABEL_KEY


def _judgment_score(traj: dict) -> float | None:
    misc = traj.get("misc", {})
    judgment = misc.get(SUBAGENT_REWARD_JUDGMENT_MISC_KEY) if isinstance(misc, dict) else None
    if not isinstance(judgment, dict):
        return None
    score = judgment.get("score")
    if not isinstance(score, (int, float)):
        return None
    return min(1.0, max(0.0, float(score)))


def _environment_reward_key(traj: dict) -> str | None:
    task = traj.get("task")
    if not isinstance(task, dict):
        return None
    misc = task.get("misc")
    if not isinstance(misc, dict):
        return None
    label = misc.get(OPENREWARD_ENVIRONMENT_LABEL_KEY)
    if not isinstance(label, str) or not label:
        return None
    metric_label = "".join(char if char.isalnum() or char in "._-" else "_" for char in label)
    return f"reward/openreward_env/{metric_label}"


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    """Return base task success plus an Oolong-style delegation bonus."""

    openreward_score = float(traj.get("reward", 0.0))
    judgment_score = _judgment_score(traj)
    base_reward = judgment_score if judgment_score is not None else openreward_score
    rewards_dict: dict[str, float] = {
        "reward/success": base_reward,
        "reward/openreward": openreward_score,
    }
    environment_reward_key = _environment_reward_key(traj)
    if environment_reward_key is not None:
        rewards_dict[environment_reward_key] = openreward_score
    if judgment_score is not None:
        rewards_dict["reward/subagent_judgment"] = judgment_score
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for key, value in reward_misc.items():
            if key.startswith("reward/") and isinstance(value, (int, float)):
                rewards_dict[key] = float(value)

    delegation = traj.get("misc", {}).get(SUBAGENT_DELEGATION_REWARD_MISC_KEY, {})
    if not isinstance(delegation, dict):
        delegation = {}
    launched = float(delegation.get("launched", 0.0))
    succeeded = float(delegation.get("succeeded", 0.0))
    delegation_bonus = float(delegation.get("bonus", 0.0))
    reward = base_reward + delegation_bonus

    # Match Oolong's recursive reward contract. These are semantic zeros for
    # trajectories that did not delegate, not missing observations.
    rewards_dict.update(
        {
            "reward/subagent_launched": launched,
            "reward/subagent_succeeded": succeeded,
            "reward/delegation_bonus": delegation_bonus,
            "reward/total": reward,
        }
    )
    return reward, rewards_dict
