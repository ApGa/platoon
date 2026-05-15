import sys

from copy import deepcopy
from areal.api.cli_args import load_expr_config
from datasets import Dataset

from platoon.codegrep.rollout import run_rollout
from platoon.codegrep.tasks import get_task, get_task_ids
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import GroupRolloutWorkflow


def main(args):
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config

    # TODO: Design a TaskLoader protocol and add configs + factory for this.
    train_dataset = Dataset.from_list([{"task_id": x} for x in get_task_ids("train")])
    val_dataset = Dataset.from_list([{"task_id": x} for x in get_task_ids("val")])

    with PlatoonArealRLTrainer(
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
        )
        
        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1
        
        eval_workflow = GroupRolloutWorkflow(
            run_rollout,
            get_task,
            eval_workflow_config,
            trainer.eval_proxy_base_url or trainer.proxy_base_url,
            trainer.proxy_admin_api_key,
            output_subdir="eval_rollout",
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
