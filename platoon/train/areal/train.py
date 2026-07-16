"""AReaL training entrypoint for Auto-selected Platoon components."""

from __future__ import annotations

import sys
from copy import deepcopy

from areal.api.cli_args import load_expr_config

from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import GroupRolloutWorkflow
from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    AutoWorkflow,
)


def run_areal_training(args: list[str] | None = None) -> None:
    """Run AReaL training from an environment-backed Platoon config."""

    config, _ = load_expr_config(args or sys.argv[1:], PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config
    AutoEnvironment.load(config)
    environment = AutoEnvironment.from_config(config)

    train_dataset = AutoDataset.from_config(config, "train")
    val_dataset = AutoDataset.from_config(config, "eval")
    rollout_fn = AutoRollout.from_config(config, "train")
    eval_rollout_fn = AutoRollout.from_config(config, "eval")
    get_task_fn = AutoTaskLoader.from_config(config)
    reward_processor = AutoRewardProcessor.from_config(config)
    workflow_cls = AutoWorkflow.from_config(config, default=GroupRolloutWorkflow)

    with PlatoonArealRLTrainer(config=config, train_dataset=train_dataset, val_dataset=val_dataset) as trainer:
        workflow_kwargs = dict(environment.workflow_kwargs)
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
        # Datum sampling is a training-throughput policy.  Evaluation should
        # always retain the complete trajectory tree.
        eval_workflow_config.subagent_datum_keep_probability = 1.0
        eval_workflow_config.filter_zero_advantage_datums = False
        eval_workflow_kwargs = dict(environment.eval_workflow_kwargs)
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
    run_areal_training()
