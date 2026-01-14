import os
from jinja2 import Environment, FileSystemLoader
import asyncio
from platoon.envs.base import Task
from .env import OpenHandsRLEnv
from platoon.utils.llm_client import LLMClient
import subprocess
from pathlib import Path
from openhands.sdk import LLM, get_logger, Agent, AgentBase, Tool
from openhands.workspace import DockerWorkspace, APIRemoteWorkspace, ApptainerWorkspace
from platoon.episode.trajectory import TrajectoryCollection
from platoon.config_defs import RolloutConfig
from openhands.sdk.workspace import BaseWorkspace
from openhands.tools.preset import get_default_agent
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection
from pydantic import SecretStr
from platoon.visualization.event_sinks import JsonlFileSink
from .tasks import EVAL_AGENT_SERVER_IMAGE, SDK_SHORT_SHA, ENV_SETUP_COMMANDS, PROMPT_FILENAME
from platoon.openhands.agent import OpenHandsAgent
import platform
logger = get_logger(__name__)

# TODO: consider pre-building all docker images, and adding their names in instance on Huggingface dataset for simpler code here
def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/xingyaoww/", #NOTE: default changed to match SWE-Gym
    # dataset: str = "swe-gym" #TODO: add dataset parameter in future
) -> str:
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    # SWE-Gym image: docker.io/xingyaoww/sweb.eval.x86_64.project-monai_s_monai-6969
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace('__', '_s_')  # to comply with docker image naming convention
    official_image_name = (docker_image_prefix.rstrip('/') + '/' + image_name).lower()
    logger.info(f"Official {docker_image_prefix} image: {official_image_name}")
    return official_image_name

# NOTE: the below function is for SWE-Bench.
# def get_official_docker_image(
#     instance_id: str,
#     docker_image_prefix="docker.io/swebench/",
# ) -> str:
#     # Official SWE-Bench image
#     # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
#     repo, name = instance_id.split("__")
#     official_image_name = docker_image_prefix.rstrip("/")
#     official_image_name += f"/sweb.eval.x86_64.{repo}_1776_{name}:latest".lower()
#     logger.debug(f"Official SWE-Bench image: {official_image_name}")
#     return official_image_name

def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Bench instance ID from official SWE-Bench image name.

    Example:
        docker.io/swebench/sweb.eval.x86_64.django_1776_django-12155:latest
        -> sweb.eval.x86_64.django_1776_django-12155
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name

def detect_platform():
    """Detects the correct platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

def get_instruction(
    instance: dict,
    workspace_path: str,
    prompt_path: str
) -> str:
    """Generate user instruction for the agent for SWE-Bench-style tasks."""
    # Set up Jinja2 environment
    # NOTE: Template will not work for SWE-Smith as its base commit is None
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    instance["repo_path"] = workspace_path
    # Prepare context for rendering
    context = {
        "instance": instance,
        "actual_workspace_path": workspace_path,
    }
    context["test_instructions"] = ""

    # Render the instruction
    instruction = template.render(context)
    return instruction

def prepare_workspace(instance: dict, task: Task) -> BaseWorkspace:
    official_docker_image: str = get_official_docker_image(instance["instance_id"])
    build_target: str = "source-minimal" #NOTE: no other targets work, so this is hard-coded for the time being
    custom_tag: str = extract_custom_tag(official_docker_image)
    suffix: str = f"-{build_target}" if build_target != "binary" else ""
    agent_server_image: str = f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"

    workspace_type: str = instance.get("workspace_type", "apptainer") #TODO: make sure the instance dict has this key
    env_setup_commands =  instance.get("env_setup_commands", ENV_SETUP_COMMANDS) #TODO: make sure the instance dict has this key
    if workspace_type == "apptainer":
        workspace = ApptainerWorkspace(
            server_image=agent_server_image,
            working_dir="/workspace",
            platform=detect_platform(),
        )
    elif workspace_type == "remote":
        # TODO: check if the environment variables are passed till this point by AReaL
        runtime_api_key = os.getenv("RUNTIME_API_KEY")
        runtime_api_url = os.getenv("RUNTIME_API_URL", "https://runtime.eval.all-hands.dev")
        workspace = APIRemoteWorkspace(
            runtime_api_url=runtime_api_url,
            runtime_api_key=runtime_api_key,
            server_image=agent_server_image,
            target_type="source" if "source" in build_target else "binary",
        )
    else: #NOTE: Docker workspace not supported yet since Babel doesn't allow docker access
        raise NotImplementedError(f"Workspace type {workspace_type} not implemented yet.")
    for cmd in env_setup_commands:
        res = workspace.execute_command(cmd)
        if res.exit_code != 0:
            raise RuntimeError(
                f"Failed to run env setup command '{cmd}': {res.stderr}"
            )
        logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
    # NOTE: Setup repository in workspace (note that we assume the workspace is remote and has the repo pre-configured from SWE-{Bench, Gym, Smith}'s docker containers)
    repo_path = f"/workspace/{instance['repo'].split('/')[-1]}/"
    logger.info(f"Repo path in Remote workspace: {repo_path}")
    instance["repo_path"] = repo_path
    
    cp_testbed_repo = workspace.execute_command(
        (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}")
    )
    assert cp_testbed_repo.exit_code == 0, (
        f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
    )
    git_reset = workspace.execute_command(f"cd {repo_path} ; git reset --hard")
    assert git_reset.exit_code == 0, f"git reset failed: {git_reset.stderr}"
    return workspace

def prepare_llm(config: RolloutConfig) -> LLM:
    is_train: bool = config.train
    # TODO: make more adjustments based on training phase
    if is_train:
        temperature = 1.0
    else:
        temperature = 0.6
    
    return LLM(
            usage_id="agent",
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=SecretStr(config.model_api_key) if config.model_api_key is not None else None,
            temperature=temperature,
            litellm_extra_body={
                # "return_token_ids": True,
                "include_stop_str_in_output": False,
                "add_generation_prompt": True,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        )

def prepare_agent(llm: LLM) -> AgentBase:
    # TODO: make tools configurable via instance/env vars or config
    # current behaviour: uses default tools without browser
    return get_default_agent(llm=llm, cli_mode=True) # browser is added iff cli_mode is False

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        """
        Steps:
            1. Create a new workspace (apptainer/remote/docker), openhands agent, and initialize env
            2. Create trajectory collection and register event handlers
        """
        instance: dict = task.misc # SWE-Bench styled instance, with extra keys: "workspace_type", "docker_image_prefix", "dataset_type", etc.
        workspace: BaseWorkspace = prepare_workspace(instance)
        
        # Get task-specific instruction and configure task parameters
        prompt_filename = instance.get("prompt_filename", PROMPT_FILENAME) #NOTE: make sure the instance dict has this key if customized prompt is desired
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        prompt_path = prompt_dir / prompt_filename
        assert prompt_path.exists(), f"Prompt path {prompt_path} not found"
        prompt_path = str(prompt_path)
        repo_path = f"/workspace/{instance['repo'].split('/')[-1]}/"
        instruction = get_instruction(instance, repo_path, prompt_path)
        task.goal = instruction
        task.max_steps = config.max_steps if config.max_steps is not None else 100

        llm: LLM = prepare_llm(config)
        agent: AgentBase = prepare_agent(llm)
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: OpenHandsRLEnv = OpenHandsRLEnv(task=task, agent=agent, workspace=workspace)

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

        rollout_task = asyncio.create_task(run_episode(agent_wrapper_platoon, env)) #NOTE: run_episode only calls agent_act which will check the event stream for new actions/observations from the agent-sdk's conversation state
        
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
        if agent_wrapper_platoon is not None:
            await agent_wrapper_platoon.close()
        if env is not None:
            await env.close()