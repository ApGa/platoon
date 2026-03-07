import asyncio
import os
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import AppWorldAgent, AppWorldDepthAwareAgent, AppWorldRecursiveAgent
from .env import AppWorldDepthAwareEnv, AppWorldEnv, AppWorldRecursiveEnv

logger = getLogger("platoon.textcraft.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    episode_started = False
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = AppWorldEnv(task, timeout_seconds=config.step_timeout)
        agent = AppWorldAgent(
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
        episode_started = True

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
        # run_episode() owns agent/env shutdown once started.
        # We only clean up here if startup failed before run_episode was launched.
        if not episode_started:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    episode_started = False
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = AppWorldRecursiveEnv(
            task,
            per_step_subagent_success_reward=0.2,
            per_step_subagent_reward_ceiling=0.4,
            timeout_seconds=config.step_timeout,
        )
        agent = AppWorldRecursiveAgent(
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
        episode_started = True

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
        # run_episode() owns agent/env shutdown once started.
        # We only clean up here if startup failed before run_episode was launched.
        if not episode_started:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()


_APPWORLD_MAX_DEPTH = 6


async def run_depth_aware_rollout(
    task: Task,
    config: RolloutConfig,
    per_agent_max_steps: int = 40,
    max_depth: int = _APPWORLD_MAX_DEPTH,
) -> dict | TrajectoryCollection:
    """Run a depth-aware recursive rollout for an AppWorld task.

    Uses ``DepthAwareStepBudgetTracker``: each agent (root and every
    subagent) gets an independent budget of *per_agent_max_steps* steps,
    and the delegation tree depth is capped at *max_depth*.
    """
    agent = env = None
    episode_started = False
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )

        # Override the task's max_steps so the root agent also uses per_agent_max_steps
        task.max_steps = per_agent_max_steps

        env = AppWorldDepthAwareEnv(
            task,
            subagent_max_steps=per_agent_max_steps,
            per_step_subagent_success_reward=0.2,
            per_step_subagent_reward_ceiling=0.4,
            timeout_seconds=config.step_timeout,
        )
        agent = AppWorldDepthAwareAgent(
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
        episode_started = True

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

        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get()

    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}")
        raise
    finally:
        if not episode_started:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()
