"""Oolong training script using the Tinker backend.

Usage:
    python -m platoon.oolong.train_scripts.tinker.train_tinker
    python -m platoon.oolong.train_scripts.tinker.train_tinker --config oolong_linear_tinker.yaml
"""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from platoon.oolong.rollout import run_recursive_rollout, run_rollout
from platoon.oolong.tasks import AnswerType, TaskGroup, get_task, get_task_ids
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("platoon").setLevel(logging.DEBUG)

_OOLONG_DELEGATION_REWARD_CAP = 0.4


@dataclass
class OolongTinkerTrainerConfig(PlatoonTinkerRLTrainerConfig):
    recursive: bool = False
    seed: int = 42
    oolong_dataset: str = "synth"
    task_group: str | None = None
    answer_type: str | None = None
    min_context_len: int | None = None
    max_context_len: int | None = None


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    """Extract Oolong reward components from trajectory steps."""
    rewards_dict: dict[str, float] = {}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if not reward_key.startswith("reward/"):
                continue
            rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    score = rewards_dict.get("reward/success", 0.0)
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += _OOLONG_DELEGATION_REWARD_CAP * subagent_success_rate

    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_filtered_task_ids(config: OolongTinkerTrainerConfig, split: str) -> list[str]:
    """Get task IDs for a split using the configured Oolong dataset filters."""
    filter_kwargs = {}
    if config.task_group is not None:
        filter_kwargs["task_group"] = TaskGroup(config.task_group)
    if config.answer_type is not None:
        filter_kwargs["answer_type"] = AnswerType(config.answer_type)
    if config.min_context_len is not None:
        filter_kwargs["min_context_len"] = config.min_context_len
    if config.max_context_len is not None:
        filter_kwargs["max_context_len"] = config.max_context_len

    task_ids = get_task_ids(config.oolong_dataset, split, **filter_kwargs)
    if not task_ids:
        raise ValueError(
            "No Oolong tasks matched the requested filters: "
            f"dataset={config.oolong_dataset}, split={split}, "
            f"task_group={config.task_group}, answer_type={config.answer_type}, "
            f"min_context_len={config.min_context_len}, max_context_len={config.max_context_len}."
        )
    return task_ids


async def main(args: list[str]) -> None:
    default_config = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "train"
        / "tinker"
        / "oolong_recursive_tinker.yaml"
    )
    config, _ = load_config(
        args=args,
        config_class=OolongTinkerTrainerConfig,
        default_config_path=str(default_config),
    )

    train_task_ids = get_filtered_task_ids(config, "validation")
    eval_task_ids = get_filtered_task_ids(config, "test")

    random.seed(config.seed)
    random.shuffle(train_task_ids)
    random.shuffle(eval_task_ids)

    rollout_fn = run_recursive_rollout if config.recursive else run_rollout

    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_task_ids])
    eval_dataset = Dataset.from_list([{"task_id": task_id} for task_id in eval_task_ids[:100]])

    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    async with trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=rollout_fn,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=True,
            reward_processor=reward_processor,
        )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=rollout_fn,
            get_task_fn=get_task,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="eval",
            filter_errors=False,
            reward_processor=reward_processor,
        )

        await trainer.train(
            train_workflow=train_workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
