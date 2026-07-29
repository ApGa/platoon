import os
import json
from jinja2 import Environment, FileSystemLoader
import asyncio
import time
from platoon.envs.base import Task
from platoon.issue_resolution.env import SWEBenchEnv
from pathlib import Path
from openhands.sdk import LLM, get_logger, Agent, Tool, AgentBase
from openhands.tools import get_default_tools
from openhands.workspace import ApptainerWorkspace
from platoon.episode.trajectory import TrajectoryCollection
from platoon.config_defs import RolloutConfig
from openhands.sdk.workspace import BaseWorkspace
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection, finish_message, error_message
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.issue_resolution.tasks import EVAL_AGENT_SERVER_IMAGE, SDK_SHORT_SHA, ENV_SETUP_COMMANDS, USER_PROMPT_FILENAME, APPTAINER_CACHE_DIR
from platoon.openhands.agent import OpenHandsAgent
import platform
import uuid
from openhands.tools.terminal import TerminalTool
from platoon.train.tinker.fastapi_litellm_proxy import SESSION_HEADER, get_active_tinker_http_proxy

logger = get_logger(__name__)

# NOTE: ApptainerWorkspace._wait_for_health has a hard-coded default of 120s.
# If that is too short when the SIF cache is cold or the agent server is slow to start. Patch it to default to 600s instead using below monkey patch.
# _orig_wait_for_health = ApptainerWorkspace._wait_for_health
# def _patched_wait_for_health(self, timeout: float = 600.0) -> None:
#     return _orig_wait_for_health(self, timeout=timeout)
# ApptainerWorkspace._wait_for_health = _patched_wait_for_health  # type: ignore[method-assign]

def detect_platform():
    """Detects the correct platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

# NOTE: the below function is for SWE-Smith.
def get_official_docker_image(
    instance: dict,
) -> str:
    # Official SWE-Smith image
    # swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
    image_name: str = instance["image_name"]
    official_image_name: str = image_name.lower().strip()
    if not official_image_name.startswith("docker.io"):
        official_image_name = f"docker.io/{official_image_name}"
    logger.debug(f"Official SWE-Smith image: {official_image_name}")
    return official_image_name

def extract_custom_tag(base_image: str) -> str:
    """
    Extract SWE-Bench instance ID from official SWE-Smith image name.

    Example:
        docker.io/swebench/swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
        -> swesmith.x86_64.oauthlib_1776_oauthlib.1fd52536
    """
    name_tag = base_image.split("/")[-1]
    name = name_tag.split(":")[0]
    return name


def agent_server_image_for_instance(instance: dict, build_target: str = "source-minimal") -> str:
    official_docker_image = get_official_docker_image(instance)
    custom_tag = extract_custom_tag(official_docker_image)
    suffix = f"-{build_target}" if build_target != "binary" else ""
    return f"{EVAL_AGENT_SERVER_IMAGE}:{SDK_SHORT_SHA}-{custom_tag}{suffix}"


def prepare_workspace(instance: dict) -> BaseWorkspace:
    # workspace_type: str = instance.get("workspace_type", "apptainer") #TODO: make sure the instance dict has this key
    # env_setup_commands =  instance.get("env_setup_commands", ENV_SETUP_COMMANDS) #TODO: make sure the instance dict has this key
    workspace_kwargs = {
        "working_dir": "/workspace",
        "platform": detect_platform(),
        "cache_dir": os.environ.get("APPTAINER_CACHEDIR", APPTAINER_CACHE_DIR),
        "detach_logs": True,
        "health_check_timeout": 600,
    }
    image_name = instance["image_name"].split("/")[-1]
    sif_path = f"{APPTAINER_CACHE_DIR}/43376f1-93c33d0-{image_name}-source-minimal.sif"
    if sif_path is None:
        workspace_kwargs["server_image"] = agent_server_image_for_instance(instance)
    else:
        workspace_kwargs["sif_file"] = sif_path
    workspace = ApptainerWorkspace(**workspace_kwargs)
    for cmd in ENV_SETUP_COMMANDS:
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
        (f"mkdir -p {repo_path} ; cp -r /testbed/. {repo_path}"), timeout=900
    )
    assert cp_testbed_repo.exit_code == 0, (
        f"cp_testbed_repo failed: {cp_testbed_repo.stderr}"
    )
    # patch_str = instance["patch"]
    # # apply this patch to the repository in the remote workspace.
    # apply_patch = workspace.execute_command(f"cd {repo_path} && git apply <<'EOF'\n{patch_str}\nEOF", timeout=900)
    # assert apply_patch.exit_code == 0, f"apply_patch failed: {apply_patch.stderr}"

    commit_id = instance["instance_id"]
    git_fetch = workspace.execute_command(f"cd {repo_path} && git fetch", timeout=300)
    assert git_fetch.exit_code == 0, f"git fetch failed: {git_fetch.stderr}"

    checkout_commit = workspace.execute_command(f"cd {repo_path} && git checkout {commit_id}", timeout=300)
    assert checkout_commit.exit_code == 0, f"git checkout failed: {checkout_commit.stderr}"

    # Extract the ground truth patch (the reverse of the bug-introducing patch).
    # `git apply` only touched the working tree; HEAD/index still hold the clean
    # code, so `git diff -R` outputs the fix without modifying any state.
    # gt_patch = workspace.execute_command(
    #     f"cd {repo_path} && git diff -R --no-color", timeout=300
    # )
    # assert gt_patch.exit_code == 0, f"ground truth patch extraction failed: {gt_patch.stderr}"
    # instance["ground_truth_patch"] = gt_patch.stdout

    return workspace

def get_instruction(
    instance: dict,
    prompt_path: str,
) -> str:
    """Generate user instruction for the agent for SWE-Bench-style tasks."""
    # Set up Jinja2 environment
    # NOTE: Jinja template will not work for SWE-Smith as its base commit is None
    prompts_dir = os.path.dirname(prompt_path)
    template_name = os.path.basename(prompt_path)
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering
    context = {
        "instance": instance,
    }

    # Render the instruction
    instruction = template.render(context)
    return instruction

def prepare_llm(config: RolloutConfig, tinker_proxy_session_id: str | None = None) -> LLM:
    model_name = config.model_name
    temperature = config.inference_params.temperature
    if not model_name.startswith("openai/") and not model_name.startswith("litellm_proxy/"):
        model_name = "openai/" + model_name

    active_proxy = get_active_tinker_http_proxy()
    extra_headers = None
    if tinker_proxy_session_id is not None:
        extra_headers = {SESSION_HEADER: tinker_proxy_session_id}

    llm=LLM(
            usage_id="agent",
            model=model_name,
            num_retries=2,
            base_url=config.model_endpoint,
            api_key=config.model_api_key or "sk-xxx",
            temperature=temperature,
            max_input_tokens=active_proxy.context_window_length if active_proxy is not None else None,
            max_output_tokens=config.inference_params.max_completion_tokens,
            extra_headers=extra_headers,
            litellm_extra_body={
                "include_stop_str_in_output": False,
                "chat_template_kwargs": {
                    # "add_generation_prompt": True, #NOTE: setting this to true raises errors
                    "enable_thinking": False
                }
            }
        )
    return llm

def prepare_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=get_default_tools(enable_browser=False),
        system_prompt_kwargs={"cli_mode": True},
        condenser=None,
    )

async def cleanup_resources(agent, env):
    if env is not None:
        await env.close()
        env = None

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = agent_wrapper_platoon = None
    tinker_proxy = get_active_tinker_http_proxy()
    tinker_proxy_session_id = str(uuid.uuid4()) if tinker_proxy is not None else None
    rollout_start = time.perf_counter()
    prepare_workspace_s: float | None = None
    prompt_build_s: float | None = None
    agent_init_s: float | None = None
    agent_loop_s: float | None = None
    cleanup_total_s = 0.0
    status = "started"
    error_detail: str | None = None
    events_path: str | None = None
    collection_id: str | None = None

    async def run_cleanup() -> None:
        nonlocal cleanup_total_s
        cleanup_start = time.perf_counter()
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        finally:
            cleanup_total_s += time.perf_counter() - cleanup_start

    try:
        if config.verbose:
            print(f"[run_rollout] Process {os.getpid()}: Starting rollout for task {task.id}", flush=True)
        instance: dict = task.misc
        try:
            loop = asyncio.get_event_loop()
            # Run in a separate thread to avoid blocking the event loop.
            prepare_workspace_start = time.perf_counter()
            workspace = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                prepare_workspace,
                instance
            )
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
        except Exception as e:
            prepare_workspace_s = time.perf_counter() - prepare_workspace_start
            status = "workspace_setup_failed"
            error_detail = str(e)
            print(f"[run_rollout] Workspace setup failed for task {task.id}: {e}", flush=True)
            raise RuntimeError(
                f"Workspace setup failed for task {task.id}: {e}"
            )
        working_dir = "/workspace"
        # if not status or working_dir is None:
        #     raise RuntimeError(f"Workspace setup failed for task {task.id}")
        
        user_prompt_filename = USER_PROMPT_FILENAME
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        user_prompt_path = prompt_dir / user_prompt_filename
        assert user_prompt_path.exists(), f"User prompt path {user_prompt_path} not found"
        prompt_build_start = time.perf_counter()
        input_message = get_instruction(instance, str(user_prompt_path))
        prompt_build_s = time.perf_counter() - prompt_build_start

        task.goal = input_message
        task.max_steps = config.max_steps if config.max_steps is not None else 100

        agent_init_start = time.perf_counter()
        llm: LLM = prepare_llm(config, tinker_proxy_session_id=tinker_proxy_session_id)
        agent: Agent = prepare_agent(llm)
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: SWEBenchEnv = SWEBenchEnv(task=task, agent=agent, workspace=workspace)
        agent_init_s = time.perf_counter() - agent_init_start

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        collection_id = traj_collection.id

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

        rollout_task = asyncio.create_task(run_episode(agent_wrapper_platoon, env, timeout=600))
        try:
            # Apply a hard timeout to the entire rollout, not just individual steps
            agent_loop_start = time.perf_counter()
            traj = await asyncio.wait_for(rollout_task, timeout=1800)
            agent_loop_s = time.perf_counter() - agent_loop_start
        except asyncio.TimeoutError:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "timeout"
            error_detail = "rollout_timed_out"
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout timed out for task {task.id}", flush=True)
            raise
        except Exception as e:
            agent_loop_s = time.perf_counter() - agent_loop_start
            status = "run_episode_failed"
            error_detail = str(e)
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout failed for task {task.id}: {str(e)}", flush=True)
            raise

        await run_cleanup()
        # ignore_rollout: bool = False
        # if env.ignore_rollout:
        #     ignore_rollout = True
        # finish_msg = traj.finish_message
        # error_msg = traj.error_message
        # if finish_msg is None or "Error in episode loop at step" in (error_msg or ""):
        #     ignore_rollout = True
        # if ignore_rollout:
        #     status = "ignored_rollout"
        #     error_detail = error_msg or "internal_errors"
        #     raise RuntimeError(f"Rollout ignored for task {task.id} due to internal errors")
        status = "success"
        if config.return_dict:
            result = current_trajectory_collection.get().to_dict()
            if tinker_proxy is not None and tinker_proxy_session_id is not None:
                result["_tinker_interactions"] = tinker_proxy.pop_interactions(tinker_proxy_session_id)
            return result
        else:
            return current_trajectory_collection.get() 
    except Exception as e:
        if status == "started":
            status = "error"
            error_detail = str(e)
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
        if tinker_proxy is not None and tinker_proxy_session_id is not None:
            tinker_proxy.discard_session(tinker_proxy_session_id)
        await run_cleanup()
        raise
    finally:
        await run_cleanup()
        print(
            "ROLLOUT_TIMING "
            + json.dumps(
                {
                    "task_id": task.id,
                    "collection_id": collection_id,
                    "pid": os.getpid(),
                    "status": status,
                    "error": error_detail,
                    "prepare_workspace_s": prepare_workspace_s,
                    "prompt_build_s": prompt_build_s,
                    "agent_init_s": agent_init_s,
                    "agent_loop_s": agent_loop_s,
                    "cleanup_total_s": cleanup_total_s,
                    "total_rollout_s": time.perf_counter() - rollout_start,
                    "events_path": events_path,
                },
                sort_keys=True,
            ),
            flush=True,
        )