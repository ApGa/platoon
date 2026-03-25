"""DeepDive training script using the AReaL backend."""

from __future__ import annotations

import random
import sys
from copy import deepcopy
from dataclasses import dataclass

from areal.api.cli_args import load_expr_config
from datasets import Dataset

from platoon.deepdive.rollout import run_recursive_rollout, run_rollout
from platoon.deepdive.tasks import get_task, get_task_ids
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow

_DEEPDIVE_DELEGATION_REWARD_CAP = 0.0


@dataclass
class DeepDiveArealTrainerConfig(PlatoonArealRLTrainerConfig):
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


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, DeepDiveArealTrainerConfig)
    config: DeepDiveArealTrainerConfig = config

    rollout_fn = run_recursive_rollout if config.recursive else run_rollout
    train_task_ids = _select_task_ids(config.train_split, config.train_num_tasks, config.seed)
    eval_task_ids = _select_task_ids(config.eval_split, config.eval_num_tasks, config.seed + 1)

    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_task_ids])
    val_dataset = Dataset.from_list([{"task_id": task_id} for task_id in eval_task_ids])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            config.workflow_config,
            trainer.proxy_server,
            "train_rollout",
            trainer.actor.device,
            filter_errors=True,
            reward_processor=reward_processor,
        )

        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1
        eval_workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            eval_workflow_config,
            trainer.eval_proxy_server,
            "eval_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
