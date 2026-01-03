import os
from platoon.envs.base import Task
from .env import CodeGreEnv
from platoon.utils.llm_client import LLMClient
import subprocess
from pathlib import Path
from openhands.sdk import LLM
from pydantic import SecretStr
from platoon.episode.trajectory import TrajectoryCollection
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.episode.context import current_trajectory_collection
from platoon.openhands.agent import OpenHandsAgent
from openhands.tools.preset.default import get_default_agent
from platoon.config_defs import RolloutConfig
import asyncio
from contextlib import suppress
from platoon.episode.loop import run_episode


def clone_instance(
    repo_name: str, commit_id: str, instance_id: str, output_dir: Path
) -> bool:
    """
    Clone a repository at a specific commit into a separate directory.

    Args:
        repo_name: Repository name in format 'owner/repo'
        commit_id: Commit hash to checkout
        instance_id: Instance ID for directory naming
        output_dir: Base output directory

    Returns:
        True if successful, False otherwise
    """
    # Create instance directory name: repo_instance-id
    # E.g., astropy_astropy-12907
    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = output_dir / instance_dir_name

    # Skip if already exists
    if instance_path.exists():
        print(f"  ✓ Instance {instance_id} already exists")
        return True, instance_path

    try:
        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                f"https://github.com/{repo_name}.git",
                str(instance_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Checkout the specific commit
        subprocess.run(
            ["git", "-C", str(instance_path), "checkout", commit_id],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"  ✓ Cloned {instance_id} at commit {commit_id[:8]}")
        return True, instance_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error cloning {instance_id}: {e.stderr}")
        return False, None
    
    
async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        workspace = Path(config.output_dir) / "testbed"
        instance_id = task.misc['instance_id']
        repo_name = task.misc['repo']
        commit_id = task.misc['base_commit']
        problem_statement = task.misc['problem_statement']
        status, working_dir = clone_instance(repo_name, commit_id, instance_id, workspace)
        task.goal = f"""I have access to a python code repository in the directory {working_dir} . 

Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Act as a code search agent to find the relevant files where we need to make changes so that <issue_description> is met?
NOTE: You do not need to solve the issue, all you need to do is find the relevant files. Your output will be used to guide another agent to solve the issue.
You can use tools such as grep to help with this. Please output your answer as just a list of the relevant files that are directly related to the <issue_description> in the format <file-list>['file1', 'file2', ...]</file-list>."""

        if not status:
            raise ValueError(f"Failed to clone instance {instance_id} for task {task['id']}")
        
        
        llm = LLM(model="openai/" + config.model_name, api_key=SecretStr(config.model_api_key), base_url=config.model_endpoint, usage_id="agent",  native_tool_calling=True)

        agent = OpenHandsAgent()
        env = CodeGreEnv(task=task, agent=get_default_agent(llm=llm, cli_mode=True), workspace=workspace)
        
        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}.jsonl")
        
        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )
        
        if config.verbose:
            print(f"Process {os.getpid()}: Starting rollout for task {task.id}")
        
        rollout_task = asyncio.create_task(run_episode(agent, env))
        try:
            final_obs = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
            rollout_task.cancel()
            with suppress(asyncio.CancelledError):
                await rollout_task
                
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
