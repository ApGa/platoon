import sys
import logging
from datasets import Dataset
from areal.api.cli_args import load_expr_config
# Enable debug logging for platoon workflows
logging.basicConfig(level=logging.WARNING)  # Quiet by default
logging.getLogger("platoon.train.areal.workflows").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence httpx spam

from platoon.openhands_rl.tasks import get_task, load_data
from platoon.openhands_rl.rollout import run_rollout
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow

def main(args):
    config, _ = load_expr_config(args, PlatoonArealRLTrainerConfig)
    config: PlatoonArealRLTrainerConfig = config
    
    train_datamap, val_datamap = load_data()
    train_dataset = Dataset.from_list([{ "task_id": x } for x in train_datamap.keys()][:1000])
    val_dataset = Dataset.from_list([{ "task_id": x } for x in val_datamap.keys()][:100])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
    
        proxy_server = trainer.proxy_server
        # TODO: do we need custom reward processor here?
        workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device, filter_errors=True)
        eval_workflow = StepWiseArealWorkflow(run_rollout, get_task, config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device)
        
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )

