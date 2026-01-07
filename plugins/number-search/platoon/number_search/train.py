import sys
from datasets import Dataset
from areal.api.cli_args import load_expr_config

from platoon.number_search.tasks import get_task_ids, get_task
from platoon.number_search.rollout import run_rollout
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow


def main(args):
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config
    
    # TODO: Design a TaskLoader protocol and add configs + factory for this.
    train_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("train", 1000)])
    val_dataset = Dataset.from_list([{ "task_id": x } for x in get_task_ids("val", 100)])
    
    
    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
    
        proxy_server = trainer.proxy_server
        workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device)
        eval_workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device)
        
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
