"""Test script for TextCraft rollout"""
import asyncio
from platoon.envs.textcraft.tasks import get_task
from platoon.envs.textcraft.env import TextCraftEnv
from platoon.agents.textcraft.agent import TextCraftAgent
from platoon.utils.llm_client import LLMClient
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.episode.context import current_trajectory_collection
from platoon.visualization.event_sinks import JsonlFileSink


async def test_single_rollout(task_id: str = "textcraft_train_0"):
    """Test a single rollout with a specific task"""
    print(f"Testing rollout for task: {task_id}")
    
    # Load task
    task = get_task(task_id)
    print(f"\nTask Goal: {task.goal}")
    print(f"Target Items: {task.misc.get('target_items', {})}")
    print(f"Initial Inventory: {task.misc.get('initial_inventory', {})}")
    print(f"Max Steps: {task.max_steps}")
    
    # Create environment
    env = TextCraftEnv(task)
    
    # Create agent with local LLM (update endpoint as needed)
    llm_client = LLMClient(
        model="neulab/claude-sonnet-4-5-20250929",
    )
    agent = TextCraftAgent(llm_client=llm_client)
    
    # Setup trajectory collection
    traj_collection = TrajectoryCollection()
    current_trajectory_collection.set(traj_collection)

    # Setup event sink
    event_sink = JsonlFileSink("events.jsonl")
    traj_collection.register_event_handlers([event_sink])
    
    try:
        # Run episode
        print("\nStarting episode...")
        final_obs = await run_episode(agent, env)
        
        print(f"\n{'='*60}")
        print("Episode Complete!")
        print(f"{'='*60}")
        print(f"Reward: {final_obs.reward}")
        print(f"Total Steps: {len(final_obs.steps)}")
        print(f"Success: {final_obs.reward > 0}")
        
        # Show trajectory
        print(f"\n{'='*60}")
        print("Trajectory:")
        print(f"{'='*60}")
        for i, step in enumerate(final_obs.steps, 1):
            print(f"\nStep {i}:")
            print(f"  Action: {step.action[:200]}...")  # Truncate long actions
            print(f"  Reward: {step.reward}")
            if step.observation:
                print(f"  Observation: {step.observation[:200]}...")
        
    finally:
        await agent.close()
        await env.close()


if __name__ == "__main__":
    import sys
    task_id = sys.argv[1] if len(sys.argv) > 1 else "textcraft.train.0"
    asyncio.run(test_single_rollout(task_id))

