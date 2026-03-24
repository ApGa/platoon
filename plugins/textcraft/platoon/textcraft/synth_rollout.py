"""Rollout functions for TextCraft-Synth environment."""

import asyncio
import os
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.utils.subagent_rewards import propogate_root_success
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import TextCraftAgent, TextCraftDepthAwareAgent, TextCraftRecursiveAgent
from .env import create_synth_depth_aware_env, create_synth_env, create_synth_recursive_env

logger = getLogger("platoon.textcraft.synth_rollout")


async def run_synth_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a rollout for a TextCraft-Synth task."""
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            # Disable Qwen3 reasoning/thinking mode for faster inference
            # default_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        env = create_synth_env(task)
        agent = TextCraftAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")

        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        if config.verbose:
            logger.info(f"Process {os.getpid()}: Starting rollout for task {task.id}")

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))

        try:
            _ = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
            rollout_task.cancel()
            # Don't wait indefinitely - tinker's sample_async may not be cancellable
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    f"Process {os.getpid()}: Task cancellation did not complete in 5s for {task.id}, abandoning"
                )
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


_TEXTCRAFT_SYNTH_MAX_DEPTH = 6


async def run_synth_depth_aware_rollout(
    task: Task,
    config: RolloutConfig,
    per_agent_max_steps: int = 25,
    max_depth: int = _TEXTCRAFT_SYNTH_MAX_DEPTH,
) -> dict | TrajectoryCollection:
    """Run a depth-aware recursive rollout for a TextCraft-Synth task.

    Uses ``DepthAwareStepBudgetTracker``: each agent (root and every
    subagent) gets an independent budget of *per_agent_max_steps* steps,
    and the delegation tree depth is capped at *max_depth*.
    """
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )

        # Override the task's max_steps so the root agent also uses per_agent_max_steps
        task.max_steps = per_agent_max_steps

        env = create_synth_depth_aware_env(
            task,
            subagent_max_steps=per_agent_max_steps,
            skip_subagent_reward_computation=config.skip_subagent_reward_computation,
        )
        agent = TextCraftDepthAwareAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

        # Install the depth-aware budget tracker BEFORE run_episode so it
        # is picked up instead of the default StepBudgetTracker.
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=max_depth))

        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")
        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        if config.verbose:
            logger.info(f"Process {os.getpid()}: Starting depth-aware rollout for task {task.id}")

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))

        try:
            _ = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    f"Process {os.getpid()}: Task cancellation did not complete in 5s for {task.id}, abandoning"
                )
            raise

        result: dict | TrajectoryCollection
        if config.return_dict:
            result = current_trajectory_collection.get().to_dict()
        else:
            result = current_trajectory_collection.get()
        if config.propogate_root_success:
            result = propogate_root_success(result)
        return result

    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}")
        raise
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()

async def run_synth_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a recursive rollout for a TextCraft-Synth task."""
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            # Disable Qwen3 reasoning/thinking mode for faster inference
            # default_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        env = create_synth_recursive_env(
            task,
            skip_subagent_reward_computation=config.skip_subagent_reward_computation,
        )
        agent = TextCraftRecursiveAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")

        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        if config.verbose:
            logger.info(f"Process {os.getpid()}: Starting rollout for task {task.id}")

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))

        try:
            _ = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
            rollout_task.cancel()
            # Don't wait indefinitely - tinker's sample_async may not be cancellable
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    f"Process {os.getpid()}: Task cancellation did not complete in 5s for {task.id}, abandoning"
                )
            raise

        result: dict | TrajectoryCollection
        if config.return_dict:
            result = current_trajectory_collection.get().to_dict()
        else:
            result = current_trajectory_collection.get()
        if config.propogate_root_success:
            result = propogate_root_success(result)
        return result

    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}")
        raise
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
