"""Rollout execution for Oolong benchmark."""
from .env import OolongEnv, OolongRecursiveEnv
from .agent import OolongAgent, OolongRecursiveAgent
from .chain_of_agents import OolongChainOfAgentsAgent
from platoon.config_defs import RolloutConfig
from platoon.utils.llm_client import LiteLLMClient
from platoon.utils.subagent_rewards import propogate_root_success
from platoon.episode.context import current_trajectory_collection, budget_tracker
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection, DepthAwareStepBudgetTracker
from platoon.visualization.event_sinks import JsonlFileSink
import os
import asyncio
from platoon.envs.base import Task
from logging import getLogger


logger = getLogger("platoon.oolong.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a single rollout for an Oolong task.

    Args:
        task: The Oolong task to run
        config: Rollout configuration

    Returns:
        TrajectoryCollection or dict depending on config.return_dict
    """
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            # Disable Qwen3 reasoning/thinking mode for faster inference
            # default_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        env = OolongEnv(task)
        agent = OolongAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
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

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))

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


async def run_chain_of_agents_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run the Chain-of-Agents baseline for a single Oolong task."""
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = OolongRecursiveEnv(task, skip_subagent_reward_computation=True)
        agent = OolongChainOfAgentsAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
            chunk_chars=config.chain_chunk_chars,
            max_communication_chars=config.chain_max_communication_chars,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=1))

        events_path = os.path.join(
            config.output_dir,
            "events",
            f"events_{task.id}_{traj_collection.id}.jsonl",
        )
        traj_collection.register_event_handlers(
            JsonlFileSink(
                events_path,
                collection_id=traj_collection.id,
                process_id=os.getpid(),
            )
        )

        if config.verbose:
            logger.info(f"Process {os.getpid()}: Starting Chain-of-Agents rollout for task {task.id}")

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error(f"Process {os.getpid()}: Chain-of-Agents rollout timed out for task {task.id}")
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
            print(f"Error running Chain-of-Agents rollout for task {task.id}: {e}")
        raise
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    """Run a recursive rollout for an Oolong task with subagent support.

    Args:
        task: The Oolong task to run
        config: Rollout configuration

    Returns:
        TrajectoryCollection or dict depending on config.return_dict
    """
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            # Disable Qwen3 reasoning/thinking mode for faster inference
            # default_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        env = OolongRecursiveEnv(
            task,
            skip_subagent_reward_computation=config.skip_subagent_reward_computation,
        )
        agent = OolongRecursiveAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=2))

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
            logger.info(f"Process {os.getpid()}: Starting recursive rollout for task {task.id}")

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))

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
