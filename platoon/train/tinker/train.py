"""Tinker training entrypoint for Auto-selected Platoon components."""

from __future__ import annotations

import asyncio
import sys

from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    AutoWorkflow,
)
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config


async def arun_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    """Run Tinker training from an environment-backed Platoon config."""

    config, _ = load_config(
        args=args or sys.argv[1:],
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=default_config_path,
    )
    config: PlatoonTinkerRLTrainerConfig = config
    AutoEnvironment.load(config)
    environment = AutoEnvironment.from_config(config)

    train_dataset = AutoDataset.from_config(config, "train")
    eval_dataset = AutoDataset.from_config(config, "eval")
    rollout_fn = AutoRollout.from_config(config, "train")
    eval_rollout_fn = AutoRollout.from_config(config, "eval")
    get_task_fn = AutoTaskLoader.from_config(config)
    reward_processor = AutoRewardProcessor.from_config(config)
    workflow_cls = AutoWorkflow.from_config(config, default=GroupRolloutWorkflow)

    trainer = PlatoonTinkerRLTrainer(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
    async with trainer:
        workflow_kwargs = dict(environment.workflow_kwargs)
        train_workflow = workflow_cls(
            rollout_fn=rollout_fn,
            get_task_fn=get_task_fn,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=workflow_kwargs.pop("stats_scope", "train"),
            filter_errors=workflow_kwargs.pop("filter_errors", True),
            reward_processor=reward_processor,
            **workflow_kwargs,
        )

        eval_workflow_kwargs = dict(environment.eval_workflow_kwargs)
        eval_workflow = workflow_cls(
            rollout_fn=eval_rollout_fn,
            get_task_fn=get_task_fn,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=eval_workflow_kwargs.pop("stats_scope", "eval"),
            filter_errors=eval_workflow_kwargs.pop("filter_errors", False),
            reward_processor=reward_processor,
            **eval_workflow_kwargs,
        )

        await trainer.train(train_workflow=train_workflow, eval_workflow=eval_workflow)


def run_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    asyncio.run(arun_tinker_training(args=args, default_config_path=default_config_path))


if __name__ == "__main__":
    run_tinker_training()
