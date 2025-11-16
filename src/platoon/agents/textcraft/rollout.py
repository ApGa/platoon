from platoon.envs.textcraft.tasks import get_task
from platoon.generators.types import RolloutGeneratorConfig
from platoon.envs.textcraft.env import TextCraftEnv
from platoon.agents.textcraft.agent import TextCraftAgent
from platoon.utils.llm_client import LLMClient
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.train.areal_integration import ArealLLMClient
from platoon.train.areal_integration import ArealEventSink
from platoon.visualization.event_sinks import JsonlFileSink
import os
import asyncio
from contextlib import suppress


def run_single_rollout_process(args: tuple[str, dict]) -> dict:
    task_id, config_dict = args
    config = RolloutGeneratorConfig(**config_dict)
    
    async def run_rollout() -> dict:
        agent = env = None
        try:
            task = get_task(task_id)
            env = TextCraftEnv(task)
            if config.max_steps_per_rollout is not None:
                task.max_steps = config.max_steps_per_rollout
            
            if hasattr(config, 'llm_client') and config.llm_client is not None:
                llm_client = config.llm_client  
            else:
                llm_client = LLMClient(model=config.model_name, base_url=config.model_endpoint)
            
            agent = TextCraftAgent(llm_client=llm_client)
            
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
                final_obs = await rollout_task
            except asyncio.TimeoutError:
                if config.verbose:
                    print(f"Process {os.getpid()}: Rollout timed out for task {task_id}")
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
            
            return current_trajectory_collection.get().to_dict()
    
    return asyncio.run(run_rollout())


