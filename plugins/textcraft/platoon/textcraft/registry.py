"""Registered TextCraft components for shared Platoon trainers."""

from __future__ import annotations

from typing import Any

from platoon.registry import (
    register_dataset_loader,
    register_reward_processor,
    register_rollout,
    register_task_loader,
    register_trainer_config,
)
from platoon.textcraft.synth_rollout import (
    run_synth_depth_aware_rollout,
    run_synth_recursive_rollout,
    run_synth_rollout,
)
from platoon.textcraft.synth_tasks import (
    Difficulty,
    get_synth_task,
    get_synth_task_ids,
    get_synth_task_ids_by_difficulty,
)

_TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP = 0.0


@register_task_loader("textcraft/synth")
def load_synth_task(task_id: str):
    return get_synth_task(task_id)


def _get_filtered_synth_task_ids(
    split: str,
    difficulties: list[str] | None,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> list[str]:
    split_name = "val" if split == "eval" else split
    if not difficulties:
        return get_synth_task_ids(split_name, num_samples_train, num_samples_val)

    task_ids: list[str] = []
    for difficulty_name in difficulties:
        try:
            difficulty = Difficulty(difficulty_name.lower())
        except ValueError:
            valid = [difficulty.value for difficulty in Difficulty]
            raise ValueError(f"Invalid difficulty {difficulty_name!r}. Valid options: {valid}") from None
        task_ids.extend(get_synth_task_ids_by_difficulty(split_name, difficulty, num_samples_train, num_samples_val))
    return task_ids


@register_dataset_loader("textcraft/synth")
def load_synth_dataset(
    config: Any,
    split: str,
    difficulties: list[str] | None = None,
    limit: int | None = None,
    num_samples_train: int = 2522,
    num_samples_val: int = 632,
):
    task_ids = _get_filtered_synth_task_ids(
        split,
        difficulties=difficulties,
        num_samples_train=num_samples_train,
        num_samples_val=num_samples_val,
    )
    if limit is not None:
        task_ids = task_ids[:limit]
    return task_ids


register_rollout("textcraft/synth/linear", run_synth_rollout)
register_rollout("textcraft/synth/recursive", run_synth_recursive_rollout)
register_rollout("textcraft/synth/depth_aware", run_synth_depth_aware_rollout)


@register_reward_processor("textcraft/synth/delegation_capped")
def synth_reward_processor(traj: dict[str, Any]) -> tuple[float, dict[str, float]]:
    rewards_dict: dict[str, float] = {}
    for step in traj["steps"]:
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    success_reward = rewards_dict.get("reward/success", 0.0)
    score = success_reward
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += _TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP * subagent_success_rate
    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


try:
    from platoon.textcraft.areal_config import TextCraftSynthArealTrainerConfig

    register_trainer_config("textcraft/synth/areal", TextCraftSynthArealTrainerConfig)
except Exception:
    pass

try:
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig

    register_trainer_config("textcraft/synth/tinker", PlatoonTinkerRLTrainerConfig)
except Exception:
    pass
