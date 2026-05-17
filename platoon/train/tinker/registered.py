"""Registry-driven Tinker training entrypoint."""

from __future__ import annotations

import asyncio
import sys

from platoon.train.registered import (
    build_registered_dataset,
    load_plugin_components,
    resolve_registered_reward_processor,
    resolve_registered_rollout,
    resolve_registered_task_loader,
    resolve_registered_workflow,
)
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config


async def arun_registered_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    """Run Tinker training from registry-backed plugin config."""

    config, _ = load_config(
        args=args or sys.argv[1:],
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=default_config_path,
    )
    config: PlatoonTinkerRLTrainerConfig = config
    load_plugin_components(config.plugin)

    train_dataset = build_registered_dataset(config, "train")
    eval_dataset = build_registered_dataset(config, "eval")
    rollout_fn = resolve_registered_rollout(config, "train")
    eval_rollout_fn = resolve_registered_rollout(config, "eval")
    get_task_fn = resolve_registered_task_loader(config)
    reward_processor = resolve_registered_reward_processor(config)
    workflow_cls = resolve_registered_workflow(config, GroupRolloutWorkflow)

    trainer = PlatoonTinkerRLTrainer(config=config, train_dataset=train_dataset, eval_dataset=eval_dataset)
    async with trainer:
        workflow_kwargs = dict(config.plugin.workflow_kwargs)
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

        eval_workflow_kwargs = dict(config.plugin.eval_workflow_kwargs)
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


def run_registered_tinker_training(args: list[str] | None = None, default_config_path: str | None = None) -> None:
    asyncio.run(arun_registered_tinker_training(args=args, default_config_path=default_config_path))


if __name__ == "__main__":
    run_registered_tinker_training()
