import asyncio
import os
from logging import getLogger

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import AppWorldAgent, AppWorldRecursiveAgent
from .env import AppWorldEnv, AppWorldRecursiveEnv

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
        # AppWorldEnv.__init__ calls AppWorld() synchronously, which starts REST API
        # servers and databases. Running it in an executor keeps the event loop
        # responsive so asyncio timeouts can still fire during initialization.
        loop = asyncio.get_running_loop()
        try:
            env = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: AppWorldEnv(task)),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            print(f"[AppWorldRollout] AppWorld init timeout (120s) for task {task.id} — aborting rollout")
            raise
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
        # AppWorldRecursiveEnv.__init__ calls AppWorld() synchronously, which starts REST
        # API servers and databases. Running it in an executor keeps the event loop
        # responsive so asyncio timeouts can still fire during initialization.
        loop = asyncio.get_running_loop()
        try:
            env = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: AppWorldRecursiveEnv(
                        task,
                        per_step_subagent_success_reward=0.2,
                        per_step_subagent_reward_ceiling=0.4,
                    ),
                ),
                timeout=120.0,
            )
        except asyncio.TimeoutError:
            print(f"[AppWorldRollout] AppWorld init timeout (120s) for task {task.id} — aborting rollout")
            raise
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
