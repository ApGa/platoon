import sys
import logging
from datasets import Dataset
from areal.api.cli_args import load_expr_config
logging.basicConfig(level=logging.INFO)  # Quiet by default
logging.getLogger("platoon.train.areal.workflows").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence httpx spam

from platoon.codescout.tasks import get_task, load_data
from platoon.codescout.rollout import run_rollout
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow

def main(args):
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config
    
    train_datamap, val_datamap = load_data()
    train_dataset = Dataset.from_list([{ "task_id": x } for x in train_datamap.keys()])
    val_dataset = Dataset.from_list([{ "task_id": x } for x in val_datamap.keys()])

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