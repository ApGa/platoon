"""TextCraft-Synth training script using Tinker backend.

Uses the synthetic dataset with deeper crafting hierarchies and difficulty tagging.

Usage:
    python -m platoon.textcraft.train_tinker_synth --config textcraft_synth_tinker.yaml
    python -m platoon.textcraft.train_tinker_synth --config textcraft_synth_tinker.yaml train.batch_size=64
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from datasets import Dataset

from platoon.textcraft.synth_tasks import get_synth_task_ids, get_synth_task
from platoon.textcraft.env import create_synth_env
from platoon.textcraft.agent import TextCraftAgent
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import load_config
from platoon.utils.llm_client import LLMClient
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.envs.base import Task
from platoon.config_defs import RolloutConfig

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("platoon").setLevel(logging.DEBUG)
logger = logging.getLogger("platoon.textcraft.train_synth")


async def run_synth_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a rollout for a TextCraft-Synth task."""
    agent = env = None
    try:
        llm_client = LLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            default_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        env = create_synth_env(task)
        agent = TextCraftAgent(llm_client=llm_client)
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
       
        events_path = os.path.join(
            config.output_dir,
            "events",
            f"events_{task.id}_{traj_collection.id}.jsonl"
        )
       
        traj_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=traj_collection.id,
                process_id=os.getpid()
            )
        )
        
        if config.verbose:
            logger.info(f"Process {os.getpid()}: Starting rollout for task {task.id}")
        
        rollout_task = asyncio.create_task(run_episode(agent, env))
        
        try:
            final_obs = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(f"Process {os.getpid()}: Task cancellation did not complete in 5s for {task.id}, abandoning")
            raise
        
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get()
        
    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}")
        raise
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()


def reward_processor(traj: dict) -> tuple[float, dict]:
    """Process trajectory rewards, extracting individual reward components."""
    rewards_dict = {}
    for step in traj['steps']:
        reward_misc = step.get('misc', {}).get('reward_misc', {})
        for reward_key, reward_value in reward_misc.items():
            if reward_key.startswith('reward/'):
                if reward_key not in rewards_dict:
                    rewards_dict[reward_key] = 0.0
                rewards_dict[reward_key] += reward_value
    score = sum(rewards_dict.values())
    return score, rewards_dict


async def main(args: list[str]):
    # Load config from YAML and CLI overrides
    default_config = Path(__file__).parent / "textcraft_synth_tinker.yaml"
    config, raw_config = load_config(
        args=args,
        config_class=PlatoonTinkerRLTrainerConfig,
        default_config_path=str(default_config),
    )
    
    # Create datasets
    # Train: use all 10000 samples
    # Eval: use first 100 of 1000 for faster validation
    train_task_ids = get_synth_task_ids("train", num_samples_train=10000, num_samples_val=1000)
    eval_task_ids = get_synth_task_ids("val", num_samples_train=10000, num_samples_val=1000)[:100]
    
    train_dataset = Dataset.from_list([{"task_id": x} for x in train_task_ids])
    eval_dataset = Dataset.from_list([{"task_id": x} for x in eval_task_ids])
    
    logger.info(f"Train dataset: {len(train_dataset)} tasks")
    logger.info(f"Eval dataset: {len(eval_dataset)} tasks")
    
    # Create trainer and run with context manager for proper cleanup
    trainer = PlatoonTinkerRLTrainer(
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    async with trainer:
        # Create workflows
        train_workflow = GroupRolloutWorkflow(
            rollout_fn=run_synth_rollout,
            get_task_fn=get_synth_task,
            config=config.train.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="train",
            filter_errors=True,
            reward_processor=reward_processor,
        )
        
        eval_workflow = GroupRolloutWorkflow(
            rollout_fn=run_synth_rollout,
            get_task_fn=get_synth_task,
            config=config.eval.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope="eval",
            filter_errors=False,
            reward_processor=reward_processor,
        )
        
        # Run training
        await trainer.train(
            train_workflow=train_workflow,
            eval_workflow=eval_workflow,
        )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
