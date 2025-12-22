from .env import TextCraftEnv, TextCraftRecursiveEnv
from .agent import TextCraftAgent, TextCraftRecursiveAgent
from platoon.train.rl import RolloutConfig
from platoon.utils.llm_client import LLMClient
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.visualization.event_sinks import JsonlFileSink
import os
import asyncio
from contextlib import suppress
from platoon.envs.base import Task
from logging import getLogger


logger = getLogger("platoon.textcraft.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key
        )
        env = TextCraftEnv(task)
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
            with suppress(asyncio.CancelledError):
                await rollout_task
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


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key
        )
        env = TextCraftRecursiveEnv(task, subagent_launch_reward=0.2)
        agent = TextCraftRecursiveAgent(llm_client=llm_client)
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
            with suppress(asyncio.CancelledError):
                await rollout_task
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