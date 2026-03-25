"""DeepDive training script using the Tinker backend."""

from __future__ import annotations

import asyncio
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from platoon.deepdive.rollout import run_recursive_rollout, run_rollout
from platoon.deepdive.tasks import get_task, get_task_ids
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

_DEEPDIVE_DELEGATION_REWARD_CAP = 0.


@dataclass
class DeepDiveTinkerTrainerConfig(PlatoonTinkerRLTrainerConfig):
    recursive: bool = True
    train_split: str = "qa_rl"
    eval_split: str = "qa_sft"
    train_num_tasks: int | None = None
    eval_num_tasks: int = 100
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    rewards_dict: dict[str, float] = {}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    score = rewards_dict.get("reward/success", 0.0)
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += _DEEPDIVE_DELEGATION_REWARD_CAP * subagent_success_rate

    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def _select_task_ids(split: str, limit: int | None, seed: int) -> list[str]:
    task_ids = list(get_task_ids(split))
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    if limit is not None and limit > 0:
        task_ids = task_ids[:limit]
    return task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parent.parent.parent / "configs" / "tinker" / "deepdive_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=DeepDiveTinkerTrainerConfig,
        default_config_path=str(default_config),
    )

    rollout_fn = run_recursive_rollout if config.recursive else run_rollout
    train_task_ids = _select_task_ids(config.train_split, config.train_num_tasks, config.seed)
    eval_task_ids = _select_task_ids(config.eval_split, config.eval_num_tasks, config.seed + 1)

    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_task_ids])
    eval_dataset = Dataset.from_list([{"task_id": task_id} for task_id in eval_task_ids])

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
