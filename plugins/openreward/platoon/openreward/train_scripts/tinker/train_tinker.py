"""OpenReward training script using the Tinker backend."""

from __future__ import annotations

import asyncio
import random
import sys
from dataclasses import asdict
from pathlib import Path

from datasets import Dataset
from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config

from platoon.openreward.rewards import reward_processor
from platoon.openreward.rollout import run_rollout
from platoon.openreward.tasks import get_task, get_task_ids
from platoon.openreward.tinker_config import OpenRewardTinkerTrainerConfig


def _attach_openreward_config(config: OpenRewardTinkerTrainerConfig) -> None:
    payload = asdict(config.openreward)
    for workflow_config in (config.train.workflow_config, config.eval.workflow_config):
        rollout_extra = dict(workflow_config.rollout_config.extra or {})
        rollout_extra["openreward"] = payload
        workflow_config.rollout_config.extra = rollout_extra


def _select_task_ids(config: OpenRewardTinkerTrainerConfig, *, split: str, limit: int | None, seed: int) -> list[str]:
    task_ids = get_task_ids(config.openreward, split=split, limit=limit)
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    return task_ids


async def main(args: list[str]) -> None:
    default_config = Path(__file__).parents[2] / "configs" / "tinker" / "toolathlon_openhands_tinker.yaml"
    config, _ = load_config(
        args=args,
        config_class=OpenRewardTinkerTrainerConfig,
        default_config_path=str(default_config),
    )
    config: OpenRewardTinkerTrainerConfig = config
    _attach_openreward_config(config)

    train_task_ids = _select_task_ids(
        config,
        split=config.openreward.split,
        limit=config.openreward.train_task_limit,
        seed=config.seed,
    )
    eval_task_ids = _select_task_ids(
        config,
        split=config.openreward.eval_split or config.openreward.split,
        limit=config.openreward.eval_task_limit,
        seed=config.seed + 1,
    )

    train_dataset = Dataset.from_list([{"task_id": task_id} for task_id in train_task_ids])
    eval_dataset = Dataset.from_list([{"task_id": task_id} for task_id in eval_task_ids])

    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    async with trainer:
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=False,
            reward_processor=reward_processor,
        )

        eval_workflow_config = config.eval.workflow_config
        # Evaluation reports rollout metrics but never trains, so datum-level
        # compute filtering is deliberately a train-only policy.
        eval_workflow_config.filter_zero_advantage_datums = False
        if eval_workflow_config.group_size != 1:
            eval_workflow_config = WorkflowConfig(
                group_size=1,
                rollout_config=eval_workflow_config.rollout_config,
                leave_one_out_baseline=eval_workflow_config.leave_one_out_baseline,
                depth_level_weighting=eval_workflow_config.depth_level_weighting,
                subagent_datum_keep_probability=eval_workflow_config.subagent_datum_keep_probability,
                subagent_datum_sampling_seed=eval_workflow_config.subagent_datum_sampling_seed,
                filter_zero_advantage_datums=False,
            )

        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_rollout,
            get_task_fn=get_task,
            config=eval_workflow_config,
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
