"""Oolong training script using AReaL backend.

Usage:
    python -m platoon.oolong.train_areal --config oolong_areal.yaml
"""

import random
import sys
from copy import deepcopy
from dataclasses import dataclass

from datasets import Dataset
from areal.api.cli_args import load_expr_config

from platoon.oolong.tasks import AnswerType, TaskGroup, get_task, get_task_ids
from platoon.oolong.rollout import run_rollout, run_recursive_rollout
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow


@dataclass
class OolongArealTrainerConfig(PlatoonArealRLTrainerConfig):
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
        subagent_success_rate = rewards_dict["reward/subagent_succeeded"] / launched
        score += 0.4 * subagent_success_rate
    
    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_filtered_task_ids(config: OolongArealTrainerConfig, split: str) -> list[str]:
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


def main(args):
    config, _ = load_expr_config(args, OolongArealTrainerConfig)
    config: OolongArealTrainerConfig = config

    # Create datasets from Oolong. There is no train split, so validation is used
    # for training and test is used for evaluation.
    # Note: oolong has validation and test splits, no train split
    train_task_ids = get_filtered_task_ids(config, "validation")
    eval_task_ids = get_filtered_task_ids(config, "test")

    # Shuffle before truncating to get diverse samples (use seed from config)
    random.seed(config.seed)
    random.shuffle(train_task_ids)
    random.shuffle(eval_task_ids)

    rollout_fn = run_rollout
    if config.recursive:
        rollout_fn = run_recursive_rollout

    train_dataset = Dataset.from_list([
        {"task_id": x} for x in train_task_ids
    ])
    val_dataset = Dataset.from_list([
        {"task_id": x} for x in eval_task_ids[:100]
    ])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:

        proxy_server = trainer.proxy_server
        eval_proxy_server = trainer.eval_proxy_server

        # Use recursive rollout for long-context tasks
        workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            config.workflow_config,
            proxy_server,
            'train_rollout',
            trainer.actor.device,
            filter_errors=True,
            reward_processor=reward_processor
        )

        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1

        eval_workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            eval_workflow_config,
            eval_proxy_server,
            'eval_rollout',
            trainer.actor.device,
            reward_processor=reward_processor
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
