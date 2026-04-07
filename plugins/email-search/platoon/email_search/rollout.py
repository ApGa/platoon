"""Rollout execution for email-search tasks."""

from __future__ import annotations

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

from .agent import EmailSearchAgent, EmailSearchRecursiveAgent
from .env import EmailSearchEnv, EmailSearchRecursiveEnv

logger = getLogger("platoon.email_search.rollout")


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = EmailSearchEnv(task)
        agent = EmailSearchAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)

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
            logger.info("Process %s: Starting rollout for task %s", os.getpid(), task.id)

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error("Process %s: Rollout timed out for task %s", os.getpid(), task.id)
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "Process %s: Task cancellation did not complete in 5s for %s, abandoning",
                    os.getpid(),
                    task.id,
                )
            raise

        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        return current_trajectory_collection.get()
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()


async def run_recursive_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = EmailSearchRecursiveEnv(task)
        agent = EmailSearchRecursiveAgent(
            llm_client=llm_client,
            inference_params=config.inference_params,
        )
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=3))

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
            logger.info("Process %s: Starting recursive rollout for task %s", os.getpid(), task.id)

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                logger.error("Process %s: Rollout timed out for task %s", os.getpid(), task.id)
            rollout_task.cancel()
            try:
                await asyncio.wait_for(rollout_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "Process %s: Task cancellation did not complete in 5s for %s, abandoning",
                    os.getpid(),
                    task.id,
                )
            raise

        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        return current_trajectory_collection.get()
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
