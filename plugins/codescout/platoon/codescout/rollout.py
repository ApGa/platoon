import os
from jinja2 import Environment, FileSystemLoader
import asyncio
from platoon.envs.base import Task
from platoon.codescout.env import CodeScoutEnv
from platoon.utils.llm_client import LLMClient
import subprocess
from pathlib import Path
from openhands.sdk import LLM, get_logger, Agent, AgentBase, Tool
# from openhands.workspace import DockerWorkspace, APIRemoteWorkspace, ApptainerWorkspace
from platoon.episode.trajectory import TrajectoryCollection
from platoon.config_defs import RolloutConfig
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.preset import get_default_agent
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection
from pydantic import SecretStr
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.codescout.tasks import EVAL_AGENT_SERVER_IMAGE, SDK_SHORT_SHA, ENV_SETUP_COMMANDS, SYSTEM_PROMPT_FILENAME, USER_PROMPT_FILENAME, APPTAINER_CACHE_DIR
from platoon.openhands.agent import OpenHandsAgent
import platform
import uuid
from platoon.codescout.custom_agent import CustomAgent
from platoon.codescout.localization_finish import LocalizationFinishTool
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.terminal import TerminalTool
logger = get_logger(__name__)

def prepare_workspace(instance: dict):
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    instance_id: str = instance["instance_id"]
    repo_name: str = instance["repo"]
    patch: str = instance["patch"]

    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = workspace / instance_dir_name

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
        subprocess.run(
            ["git", "-C", str(instance_path), "apply"],
            input=patch,
            check=True,
            capture_output=True,
            text=True,
        )
        return True, instance_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error cloning {instance_id}: {e.stderr}")
        return False, None

def get_instruction(
    instance: dict,
    prompt_path: str,
    workspace_path: str,
) -> str:
    """Generate instruction for the agent."""
    # Set up Jinja2 environment
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
        "working_dir": workspace_path,
    }

    # Render the instruction
    instruction = template.render(context)
    return instruction


def prepare_llm(config: RolloutConfig) -> LLM:
    model_name = config.model_name
    temperature = 1.0
    if not model_name.startswith("openai/") and not model_name.startswith("litellm_proxy/"):
        model_name = "openai/" + model_name

    llm=LLM(
            usage_id="agent",
            model=model_name,
            base_url=config.model_endpoint,
            api_key="sk-xxx",
            temperature=temperature,
            litellm_extra_body={
                # "return_token_ids": True,
                "include_stop_str_in_output": False,
                "chat_template_kwargs": {
                    # "add_generation_prompt": True,
                    "enable_thinking": False
                }
            }
        )
    return llm

def prepare_agent(llm: LLM, system_prompt_path: str) -> AgentBase:
    # TODO: make tools configurable via instance/env vars or config
    # current behaviour: uses default tools without browser
    register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
    tools = [
        Tool(name=TerminalTool.name),
        Tool(name="localization_finish"),
    ]
    agent = CustomAgent(
        llm=llm,
        tools=tools,
        system_prompt_filename=system_prompt_path
    )
    return agent

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = agent_wrapper_platoon = None
    WORKSPACE_SETUP_TIMEOUT = 1200  # 20 minutes max for workspace setup
    try:
        """
        Steps:
            1. Create a new workspace (apptainer/remote/local), openhands agent, and initialize env
            2. Create trajectory collection and register event handlers
        """
        if config.verbose:
            print(f"[run_rollout] Process {os.getpid()}: Starting rollout for task {task.id}", flush=True)
        instance: dict = task.misc
        loop = asyncio.get_event_loop()
        try:
            status, working_dir = await asyncio.wait_for(
                loop.run_in_executor(None, prepare_workspace, instance),
                timeout=WORKSPACE_SETUP_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Workspace setup timed out after {WORKSPACE_SETUP_TIMEOUT}s for task {task.id}"
            )
        if not status or working_dir is None:
            raise RuntimeError(f"Workspace setup failed for task {task.id}")
        
        user_prompt_filename = USER_PROMPT_FILENAME
        system_prompt_filename = SYSTEM_PROMPT_FILENAME
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        user_prompt_path = prompt_dir / user_prompt_filename
        system_prompt_path = prompt_dir / system_prompt_filename
        assert user_prompt_path.exists(), f"User prompt path {user_prompt_path} not found"
        assert system_prompt_path.exists(), f"System prompt path {system_prompt_path} not found"
        input_message = get_instruction(instance, str(user_prompt_path), str(working_dir))

        task.goal = input_message
        task.max_steps = config.max_steps if config.max_steps is not None else 6

        llm: LLM = prepare_llm(config)
        agent: AgentBase = prepare_agent(llm, str(system_prompt_path))
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: CodeScoutEnv = CodeScoutEnv(task=task, agent=agent, workspace=str(working_dir))

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

        rollout_task = asyncio.create_task(run_episode(agent_wrapper_platoon, env, timeout=300))
        try:
            # Apply a hard timeout to the entire rollout, not just individual steps
            _ = await asyncio.wait_for(rollout_task, timeout=330)
        except asyncio.TimeoutError:
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout timed out for task {task.id}", flush=True)
            # The task should already be cancelled by wait_for, but let's be explicit
            raise
        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout failed for task {task.id}: {e}", flush=True)
            raise

        try:
            if working_dir.exists():
                os.system(f"rm -rf {str(working_dir)}")
                logger.info(f"Removed workspace {str(working_dir)}")
        except Exception as _:
            pass
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get() 
    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
        raise
    # finally:
    #     pass
        # Safety-net cleanup: ensure env is closed even if run_episode never ran
        # (e.g. error during workspace setup after env was created).
        # env.close() is idempotent — if run_episode already called it,
        # self._conversation will be None and this is a no-op.
        # if env is not None:
        #     try:
        #         await asyncio.wait_for(env.close(), timeout=30)
        #     except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
        #         print(f"Warning: safety-net env.close() in run_rollout: {type(e).__name__}: {e}", flush=True)
        # if agent_wrapper_platoon is not None:
        #     try:
        #         await asyncio.wait_for(agent_wrapper_platoon.close(), timeout=10)
        #     except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
        #         print(f"Warning: safety-net agent.close() in run_rollout: {type(e).__name__}: {e}", flush=True)
