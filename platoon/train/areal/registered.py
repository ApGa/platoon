"""Registry-driven AReaL training entrypoint."""

from __future__ import annotations

import sys
from copy import deepcopy

from areal.api.cli_args import load_expr_config

from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import GroupRolloutWorkflow
from platoon.train.registered import (
    build_registered_dataset,
    load_plugin_components,
    resolve_registered_reward_processor,
    resolve_registered_rollout,
    resolve_registered_task_loader,
    resolve_registered_workflow,
)


def run_registered_areal_training(args: list[str] | None = None) -> None:
    """Run AReaL training from registry-backed plugin config."""

    config, _ = load_expr_config(args or sys.argv[1:], PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config
    load_plugin_components(config.plugin)

    train_dataset = build_registered_dataset(config, "train")
    val_dataset = build_registered_dataset(config, "eval")
    rollout_fn = resolve_registered_rollout(config, "train")
    eval_rollout_fn = resolve_registered_rollout(config, "eval")
    get_task_fn = resolve_registered_task_loader(config)
    reward_processor = resolve_registered_reward_processor(config)
    workflow_cls = resolve_registered_workflow(config, GroupRolloutWorkflow)

    with PlatoonArealRLTrainer(config=config, train_dataset=train_dataset, val_dataset=val_dataset) as trainer:
        workflow_kwargs = dict(config.plugin.workflow_kwargs)
        workflow = workflow_cls(
            rollout_fn,
            get_task_fn,
            config.workflow_config,
            trainer.proxy_base_url,
            trainer.proxy_admin_api_key,
            output_subdir=workflow_kwargs.pop("output_subdir", "train_rollout"),
            filter_errors=workflow_kwargs.pop("filter_errors", True),
            reward_processor=reward_processor,
            **workflow_kwargs,
        )

        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1
        eval_workflow_kwargs = dict(config.plugin.eval_workflow_kwargs)
        eval_workflow = workflow_cls(
            eval_rollout_fn,
            get_task_fn,
            eval_workflow_config,
            trainer.eval_proxy_base_url or trainer.proxy_base_url,
            trainer.proxy_admin_api_key,
            output_subdir=eval_workflow_kwargs.pop("output_subdir", "eval_rollout"),
            filter_errors=eval_workflow_kwargs.pop("filter_errors", False),
            reward_processor=reward_processor,
            **eval_workflow_kwargs,
        )

        trainer.train(workflow=workflow, eval_workflow=eval_workflow)


if __name__ == "__main__":
    run_registered_areal_training()
