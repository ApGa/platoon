"""Email-search training script using the AReaL backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
import sys

from areal.api.cli_args import load_expr_config
from datasets import Dataset

from platoon.email_search.rollout import run_recursive_rollout, run_rollout
from platoon.email_search.tasks import get_task, get_task_ids
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow

_EMAIL_SEARCH_DELEGATION_REWARD_CAP = 0.1
_DEFAULT_REWARD_KEYS = {
    "reward/success": 0.0,
    "reward/subagent_launched": 0.0,
    "reward/subagent_succeeded": 0.0,
}


@dataclass
class EmailSearchArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = True
    train_split: str = "train"
    eval_split: str = "test"
    train_num_tasks: int | None = None
    eval_num_tasks: int | None = 100
    max_messages: int | None = 1
    exclude_known_bad_queries: bool = True
    seed: int = 42


def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    rewards_dict: dict[str, float] = dict(_DEFAULT_REWARD_KEYS)
    found_reward_metric = False
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                found_reward_metric = True
                rewards_dict[reward_key] = rewards_dict.get(reward_key, 0.0) + float(reward_value)

    score = rewards_dict["reward/success"]
    launched = rewards_dict["reward/subagent_launched"]
    if launched > 0:
        subagent_success_rate = rewards_dict["reward/subagent_succeeded"] / launched
        score += _EMAIL_SEARCH_DELEGATION_REWARD_CAP * subagent_success_rate

    if not found_reward_metric:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def _select_task_ids(
    split: str,
    limit: int | None,
    seed: int,
    max_messages: int | None,
    exclude_known_bad_queries: bool,
) -> list[str]:
    task_ids = list(
        get_task_ids(
            split=split,
            max_messages=max_messages,
            exclude_known_bad_queries=exclude_known_bad_queries,
        )
    )
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    if limit is not None and limit > 0:
        task_ids = task_ids[:limit]
    return task_ids


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, EmailSearchArealTrainerConfig)
    config: EmailSearchArealTrainerConfig = config

    rollout_fn = run_recursive_rollout if config.recursive else run_rollout
    train_task_ids = _select_task_ids(
        config.train_split,
        config.train_num_tasks,
        config.seed,
        config.max_messages,
        config.exclude_known_bad_queries,
    )
    eval_task_ids = _select_task_ids(
        config.eval_split,
        config.eval_num_tasks,
        config.seed + 1,
        config.max_messages,
        config.exclude_known_bad_queries,
    )

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
