"""TextCraft-Synth depth-aware training script using Tinker backend.

Uses DepthAwareStepBudgetTracker: each agent (root + subagents) gets 25
independent steps, and the delegation tree depth is capped at 12.

Usage:
    python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_depth_aware
    python -m platoon.textcraft.train_scripts.tinker.train_tinker_synth_depth_aware \\
        --config textcraft_synth_depth_aware_tinker.yaml
"""

import asyncio
import logging
import sys
from pathlib import Path

from datasets import Dataset

from platoon.textcraft.synth_rollout import run_synth_depth_aware_rollout
from platoon.textcraft.synth_tasks import (
    Difficulty,
    get_synth_task,
    get_synth_task_ids,
    get_synth_task_ids_by_difficulty,
)
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
logger = logging.getLogger("platoon.textcraft.train_synth_depth_aware")
_TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP = 0.0


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
    score = success_reward
    launched = rewards_dict.get("reward/subagent_launched", 0.0)
    if launched > 0:
        subagent_success_rate = rewards_dict.get("reward/subagent_succeeded", 0.0) / launched
        score += _TEXTCRAFT_SYNTH_DELEGATION_REWARD_CAP * subagent_success_rate
    if not rewards_dict:
        score = float(traj.get("reward", 0.0))
    return score, rewards_dict


def get_filtered_task_ids(
    split: str,
    difficulties: list[str] | None,
    num_samples_train: int = 10000,
    num_samples_val: int = 1000,
) -> list[str]:
    """Get task IDs, optionally filtered by difficulty levels."""
    if not difficulties:
        return get_synth_task_ids(split, num_samples_train, num_samples_val)

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


async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "../../configs/tinker/textcraft_synth_depth_aware_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )

    # Get difficulty filter from config (if specified)
    train_difficulties = ["medium"]
    eval_difficulties = None

    # Create datasets with optional difficulty filtering
    train_task_ids = get_filtered_task_ids("train", train_difficulties, num_samples_train=2522)
    eval_task_ids = get_filtered_task_ids("val", eval_difficulties, num_samples_val=632)[:100]

    if train_difficulties:
        logger.info(f"Filtering train tasks to difficulties: {train_difficulties}")
    if eval_difficulties:
        logger.info(f"Filtering eval tasks to difficulties: {eval_difficulties}")

    train_dataset = Dataset.from_list([{"task_id": x} for x in train_task_ids])
    eval_dataset = Dataset.from_list([{"task_id": x} for x in eval_task_ids])

    logger.info(f"Train dataset: {len(train_dataset)} tasks")
    logger.info(f"Eval dataset: {len(eval_dataset)} tasks")

    # Create trainer and run with context manager for proper cleanup
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    async with trainer:
        # Create workflows using the depth-aware rollout function
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_synth_depth_aware_rollout,
            get_task_fn=get_synth_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=True,
            reward_processor=reward_processor,
        )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_synth_depth_aware_rollout,
            get_task_fn=get_synth_task,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="eval",
            filter_errors=False,
            reward_processor=reward_processor,
        )

        # Run training
        await trainer.train(
            train_workflow=train_workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
