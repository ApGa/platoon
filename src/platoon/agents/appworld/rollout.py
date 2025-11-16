import os
import asyncio
from contextlib import suppress

from platoon.agents.appworld.codeact import AppWorldCodeActAgent
from platoon.agents.appworld.recursive_agent import (
    AppWorldRecursiveCodeActAgent,
)
from platoon.envs.base import Task
from platoon.envs.appworld.env import (
    AppWorldCodeExecutor,
    AppWorldEnv,
    AppWorldRecursiveCodeExecutor,
)
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LLMClient
from platoon.generators.types import RolloutGeneratorConfig
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.train.areal_integration import ArealEventSink, ArealLLMClient

def run_single_rollout_process(args: tuple[str, dict]) -> dict:
    task_id, config_dict = args
    
    config = RolloutGeneratorConfig(**config_dict)
    
    if "APPWORLD_ROOT" not in os.environ and hasattr(config, 'appworld_root'):
        os.environ["APPWORLD_ROOT"] = config.appworld_root
    
    async def run_rollout() -> dict:
        agent = env = None
        try:
            task = Task(id=task_id)
            if config.max_steps_per_rollout is not None:
                task.max_steps = config.max_steps_per_rollout
                
            executor = AppWorldCodeExecutor(task, local_mode=True)
            env = AppWorldEnv(task, code_executor=executor)
            
            if hasattr(config, 'llm_client') and config.llm_client is not None: # TODO: Consider a cleaner way to pass training args.
                llm_client = config.llm_client  
            else:
                llm_client = LLMClient(model=config.model_name, base_url=config.model_endpoint)
            
            agent = AppWorldCodeActAgent(llm_client=llm_client)
            
            traj_collection = TrajectoryCollection()
            current_trajectory_collection.set(traj_collection)
            # Stream events to a JSONL file under a common directory for live TUI consumption
            events_dir = os.path.join(config.output_dir, "events")
            
            if isinstance(llm_client, ArealLLMClient):
                engine = llm_client.proxy_engine
                events_dir = os.path.join(config.output_dir, "events", engine.get_version())
            
            events_path = os.path.join(
                events_dir, f"events_{task_id}_{traj_collection.id}.jsonl"
            )
            traj_collection.register_event_handlers(
                JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
            )
            
            if isinstance(llm_client, ArealLLMClient):
                traj_collection.register_event_handlers(ArealEventSink())
            
            if config.verbose:
                print(f"Process {os.getpid()}: Starting rollout for task {task_id}")

            # Enforce per-rollout timeout so that a hanging rollout is actually cancelled
            rollout_task = asyncio.create_task(run_episode(agent, env))
            try:
                # if config.per_rollout_timeout_seconds:
                #     final_obs = await asyncio.wait_for(
                #         rollout_task, timeout=config.per_rollout_timeout_seconds
                #     )
                # else:
                final_obs = await rollout_task
            except asyncio.TimeoutError:
                if config.verbose:
                    print(
                        f"Process {os.getpid()}: Rollout for task {task_id} timed out after "
                        f"{config.per_rollout_timeout_seconds}s; cancelling."
                    )
                rollout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rollout_task
                raise
            
            if config.verbose:
                print(f"Process {os.getpid()}: Completed rollout for task {task_id} - "
                     f"Reward: {final_obs.reward}, Steps: {len(final_obs.steps)}")
            
            return current_trajectory_collection.get().to_dict()
            
        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Failed rollout for task {task_id}: {e}")
            raise
        finally:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()
            
    return asyncio.run(run_rollout())

def run_single_recursive_rollout_process(args: tuple[str, dict]) -> dict:
    task_id, config_dict = args

    config = RolloutGeneratorConfig(**config_dict)

    if "APPWORLD_ROOT" not in os.environ and hasattr(config, 'appworld_root'):
        os.environ["APPWORLD_ROOT"] = config.appworld_root

    async def run_rollout() -> dict:
        agent = env = None
        try:
            task = Task(id=task_id)
            if config.max_steps_per_rollout is not None:
                task.max_steps = config.max_steps_per_rollout

            executor = AppWorldRecursiveCodeExecutor(task, local_mode=True)
            env = AppWorldEnv(task, code_executor=executor)

            if hasattr(config, 'llm_client') and config.llm_client is not None: # TODO: Consider a cleaner way to pass training args.
                llm_client = config.llm_client
            else:
                llm_client = LLMClient(model=config.model_name, base_url=config.model_endpoint)
            agent = AppWorldRecursiveCodeActAgent(llm_client=llm_client, use_parent_state=False)

            traj_collection = TrajectoryCollection()
            current_trajectory_collection.set(traj_collection)
            # Stream events to a JSONL file under a common directory for live TUI consumption
            events_dir = os.path.join(config.output_dir, "events")
            events_path = os.path.join(
                events_dir, f"events_{task_id}_{traj_collection.id}.jsonl"
            )
            traj_collection.register_event_handlers(
                JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
            )

            if isinstance(llm_client, ArealLLMClient):
                print("Registering ArealEventSink for recursive rollout")
                traj_collection.register_event_handlers(ArealEventSink())

            if config.verbose:
                print(f"Process {os.getpid()}: Starting RECURSIVE rollout for task {task_id}")

            # Enforce per-rollout timeout so that a hanging rollout is actually cancelled
            rollout_task = asyncio.create_task(
                run_episode(agent, env, verbose=config.verbose)
            )
            try:
                # if config.per_rollout_timeout_seconds:
                #     final_obs = await asyncio.wait_for(
                #         rollout_task, timeout=config.per_rollout_timeout_seconds
                #     )
                # else:
                final_obs = await rollout_task
            except asyncio.TimeoutError:
                if config.verbose:
                    print(
                        f"Process {os.getpid()}: RECURSIVE rollout for task {task_id} timed out after "
                        f"{config.per_rollout_timeout_seconds}s; cancelling."
                    )
                rollout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rollout_task
                raise

            if config.verbose:
                print(
                    f"Process {os.getpid()}: Completed RECURSIVE rollout for task {task_id} - "
                    f"Reward: {final_obs.reward}, Steps: {len(final_obs.steps)}"
                )

            return current_trajectory_collection.get().to_dict()

        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Failed RECURSIVE rollout for task {task_id}: {e}")
            raise
        finally:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()

    return asyncio.run(run_rollout())