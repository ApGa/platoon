"""TextCraft-Synth training script using Areal backend.

Uses the synthetic dataset with deeper crafting hierarchies and difficulty tagging.

Usage:
    python -m areal.launcher.local train_areal_synth_recursive.py --config textcraft_synth_recursive_areal.yaml
    python -m areal.launcher.local train_areal_synth_recursive.py --config textcraft_synth_recursive_areal.yaml train.batch_size=64
"""

import logging
import sys

from areal.api.cli_args import load_expr_config
from datasets import Dataset

# Enable debug logging for platoon workflows
logging.basicConfig(level=logging.WARNING)  # Quiet by default
logging.getLogger("platoon.train.areal.workflows").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence httpx spam

from platoon.textcraft.synth_rollout import run_synth_recursive_rollout  # noqa: E402
from platoon.textcraft.synth_tasks import (  # noqa: E402
    Difficulty,
    get_synth_task,
    get_synth_task_ids,
    get_synth_task_ids_by_difficulty,
)
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig  # noqa: E402
from platoon.train.areal.workflows import StepWiseArealWorkflow  # noqa: E402

logger = logging.getLogger("platoon.textcraft.train_areal_synth")


def reward_processor(traj: dict) -> tuple[float, dict]:
    """Process trajectory rewards, extracting individual reward components."""
    rewards_dict = {}
    for step in traj["steps"]:
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                if reward_key not in rewards_dict:
                    rewards_dict[reward_key] = 0.0
                rewards_dict[reward_key] += reward_value

    success_reward = rewards_dict.get("reward/success", 0.0)
    other_rewards = min(sum(rewards_dict.values()) - success_reward, 0.4)
    score = success_reward + other_rewards
    return score, rewards_dict


def get_filtered_task_ids(
    split: str,
    difficulties: list[str] | None,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> list[str]:
    """Get task IDs, optionally filtered by difficulty levels.

    Args:
        split: "train" or "val"
        difficulties: List of difficulty names to include (e.g., ["easy", "medium"]).
                     If None or empty, returns all tasks.
        num_samples_train: Total number of training samples in the dataset
        num_samples_val: Total number of validation samples in the dataset

    Returns:
        List of task IDs matching the specified difficulties
    """
    if not difficulties:
        return get_synth_task_ids(split, num_samples_train, num_samples_val)

    # Collect task IDs for each requested difficulty
    all_ids = []
    for diff_name in difficulties:
        try:
            diff = Difficulty(diff_name.lower())
        except ValueError:
            valid = [d.value for d in Difficulty]
            raise ValueError(f"Invalid difficulty '{diff_name}'. Valid options: {valid}")

        ids = get_synth_task_ids_by_difficulty(split, diff, num_samples_train, num_samples_val)
        all_ids.extend(ids)

    return all_ids


def main(args):
    config, raw_config = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config

    # Get difficulty filter from config (if specified)
    # Can be set via CLI: train_difficulties='["easy","medium"]'
    train_difficulties = ["medium"]  # raw_config.get("train_difficulties", None)
    eval_difficulties = None  # raw_config.get("eval_difficulties", None)

    # Create datasets with optional difficulty filtering
    train_task_ids = get_filtered_task_ids("train", train_difficulties, num_samples_train=2522)
    eval_task_ids = get_filtered_task_ids("val", eval_difficulties, num_samples_val=632)[:100]

    if train_difficulties:
        logger.info(f"Filtering train tasks to difficulties: {train_difficulties}")
    if eval_difficulties:
        logger.info(f"Filtering eval tasks to difficulties: {eval_difficulties}")

    train_dataset = Dataset.from_list([{"task_id": x} for x in train_task_ids])
    val_dataset = Dataset.from_list([{"task_id": x} for x in eval_task_ids])

    logger.info(f"Train dataset: {len(train_dataset)} tasks")
    logger.info(f"Eval dataset: {len(val_dataset)} tasks")

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        proxy_server = trainer.proxy_server
        workflow = StepWiseArealWorkflow(
            run_synth_recursive_rollout,
            get_synth_task,
            config.workflow_config,
            proxy_server,
            "train_rollout",
            trainer.actor.device,
            filter_errors=True,
            reward_processor=reward_processor,
        )
        eval_workflow = StepWiseArealWorkflow(
            run_synth_recursive_rollout,
            get_synth_task,
            config.workflow_config,
            proxy_server,
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
