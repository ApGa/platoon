from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import asdict

from areal.api.cli_args import load_expr_config
from datasets import Dataset
from platoon.train.areal.workflows import GroupRolloutWorkflow

from platoon.openreward.areal_config import OpenRewardArealTrainerConfig
from platoon.openreward.areal_trainer import OpenRewardArealRLTrainer
from platoon.openreward.rewards import reward_processor
from platoon.openreward.rollout import run_rollout
from platoon.openreward.tasks import get_task, get_task_records


def _attach_openreward_config(config: OpenRewardArealTrainerConfig) -> None:
    rollout_extra = dict(config.workflow_config.rollout_config.extra or {})
    rollout_extra["openreward"] = asdict(config.openreward)
    config.workflow_config.rollout_config.extra = rollout_extra


def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, OpenRewardArealTrainerConfig)
    config: OpenRewardArealTrainerConfig = config
    _attach_openreward_config(config)

    train_dataset = Dataset.from_list(get_task_records(config.openreward))
    val_dataset = Dataset.from_list(get_task_records(config.openreward, evaluation=True))

    with OpenRewardArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        workflow = GroupRolloutWorkflow(
            run_rollout,
            get_task,
            config.workflow_config,
            trainer.proxy_base_url,
            trainer.proxy_admin_api_key,
            output_subdir="train_rollout",
            reward_processor=reward_processor,
            filter_errors=False,
        )

        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1
        # Datum sampling is a training-throughput policy.  Evaluation should
        # always retain the complete trajectory tree.
        eval_workflow_config.subagent_datum_keep_probability = 1.0
        eval_workflow_config.filter_zero_advantage_datums = False

        eval_workflow = GroupRolloutWorkflow(
            run_rollout,
            get_task,
            eval_workflow_config,
            trainer.eval_proxy_base_url or trainer.proxy_base_url,
            trainer.proxy_admin_api_key,
            output_subdir="eval_rollout",
            reward_processor=reward_processor,
            filter_errors=False,
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
