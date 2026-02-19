import sys

from copy import deepcopy
from areal.api.cli_args import load_expr_config
from appworld import load_task_ids
from datasets import Dataset
from dataclasses import dataclass

from platoon.appworld.rollout import run_rollout, run_recursive_rollout
from platoon.appworld.tasks import get_task
from platoon.train.areal import PlatoonArealRLTrainer, PlatoonArealRLTrainerConfig
from platoon.train.areal.workflows import StepWiseArealWorkflow


@dataclass
class AppWorldArealTrainerConfig(PlatoonArealRLTrainerConfig):
    recursive: bool = False

def reward_processor(traj: dict) -> tuple[float, dict]:
    """Process trajectory rewards, extracting individual reward components."""
    rewards_dict = {}
    for step in traj["steps"]:
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith("reward/"):
                if reward_key not in rewards_dict:
                    rewards_dict[reward_key] = 0.0
                rewards_dict[reward_key] += reward_value

    success_reward = rewards_dict.get("reward/success", 0.0)
    other_rewards = min(sum(rewards_dict.values()) - success_reward, 0.4)
    score = success_reward + other_rewards
    return score, rewards_dict

def main(args):
    config, _ = load_expr_config(args, AppWorldArealTrainerConfig)
    config: AppWorldArealTrainerConfig = config
    
    if config.recursive:
        rollout_fn = run_recursive_rollout
    else:
        rollout_fn = run_rollout

    train_dataset = Dataset.from_list([{"task_id": x} for x in load_task_ids(dataset_name="train")])
    val_dataset = Dataset.from_list([{"task_id": x} for x in load_task_ids(dataset_name="dev")])

    with PlatoonArealRLTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    ) as trainer:
        proxy_server = trainer.proxy_server
        eval_proxy_server = trainer.eval_proxy_server
        workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            config.workflow_config,
            proxy_server,
            "train_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
            filter_errors=False,
        )
        
        eval_workflow_config = deepcopy(config.workflow_config)
        eval_workflow_config.group_size = 1
        
        eval_workflow = StepWiseArealWorkflow(
            rollout_fn,
            get_task,
            eval_workflow_config,
            eval_proxy_server,
            "eval_rollout",
            trainer.actor.device,
            reward_processor=reward_processor,
            filter_errors=False,
        )

        trainer.train(
            workflow=workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
