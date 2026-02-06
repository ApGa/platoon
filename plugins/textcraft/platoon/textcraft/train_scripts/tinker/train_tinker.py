"""TextCraft training script using Tinker backend.

Usage:
    python -m platoon.textcraft.train_tinker --config textcraft_tinker.yaml
    python -m platoon.textcraft.train_tinker --config textcraft_tinker.yaml --train.batch_size 64
"""

import asyncio
import logging
import sys
from pathlib import Path

from datasets import Dataset

from platoon.textcraft.rollout import run_recursive_rollout
from platoon.textcraft.tasks import get_task, get_task_ids
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
    score = sum(rewards_dict.values())
    return score, rewards_dict


async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "textcraft_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )

    # Create datasets
    train_dataset = Dataset.from_list([{"task_id": x} for x in get_task_ids("train", 1000)])
    eval_dataset = Dataset.from_list([{"task_id": x} for x in get_task_ids("val", 100)])

    # Create trainer and run with context manager for proper cleanup
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    async with trainer:
        # Create workflows - use trainer.run_log_path for run-specific output
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_recursive_rollout,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=True,
            reward_processor=reward_processor,
        )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_recursive_rollout,
            get_task_fn=get_task,
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
