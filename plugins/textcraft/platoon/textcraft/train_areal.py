import sys
import logging
from datasets import Dataset
from areal.api.cli_args import load_expr_config

# Enable debug logging for platoon workflows
logging.basicConfig(level=logging.WARNING)  # Quiet by default
logging.getLogger("platoon.train.areal.workflows").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Silence httpx spam

from platoon.textcraft.tasks import get_task_ids, get_task
from platoon.textcraft.rollout import run_rollout, run_recursive_rollout
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow

def reward_processor(traj: dict) -> tuple[float, dict]:
    rewards_dict = dict()
    for step in traj['steps']:
        for reward_key, reward_value in step['misc']['reward_misc'].items():
            if reward_key.startswith('reward/'):
                if reward_key not in rewards_dict:
                    rewards_dict[reward_key] = 0.0
                rewards_dict[reward_key] += reward_value
    score = sum(rewards_dict.values())
    return score, rewards_dict

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
        workflow = StepWiseArealWorkflow(run_recursive_rollout, get_task, config.workflow_config, proxy_server, 'train_rollout', trainer.actor.device, filter_errors=True, reward_processor=reward_processor)
        eval_workflow = StepWiseArealWorkflow(run_recursive_rollout, get_task, config.workflow_config, proxy_server, 'eval_rollout', trainer.actor.device, reward_processor=reward_processor)
        
        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
