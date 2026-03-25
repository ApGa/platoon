import os
from jinja2 import Environment, FileSystemLoader
import asyncio
from platoon.envs.base import Task
from platoon.codescout.env import CodeScoutEnv
from pathlib import Path
from openhands.sdk import LLM, get_logger, Agent, Tool
from openhands.workspace import ApptainerWorkspace
from platoon.episode.trajectory import TrajectoryCollection
from platoon.config_defs import RolloutConfig
from platoon.episode.loop import run_episode
from platoon.episode.context import current_trajectory_collection
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.codescout.tasks import EVAL_AGENT_SERVER_IMAGE, USER_PROMPT_FILENAME, APPTAINER_CACHE_DIR
from platoon.openhands.agent import OpenHandsAgent
import platform
import uuid
from openhands.tools.terminal import TerminalTool
from platoon.codescout.custom_tools.localization_finish import LocalizationFinishTool  # noqa: F401 - registers the tool with the correct module qualname

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

def prepare_workspace(instance: dict):
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    instance_id: str = instance["instance_id"]
    repo_name: str = instance["repo"]
    patch: str = instance["patch"]

    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = workspace / instance_dir_name

    os.makedirs(APPTAINER_CACHE_DIR, exist_ok=True)

    # use the openhands agent server image and then setup env manually
    workspace = ApptainerWorkspace(
        server_image=EVAL_AGENT_SERVER_IMAGE,
        working_dir=str(instance_path),
        platform=detect_platform(),
        cache_dir=os.environ.get("APPTAINER_CACHEDIR", APPTAINER_CACHE_DIR),
        detach_logs=True
    )

    def _run(cmd: str, timeout: float = 120.0) -> None:
        """Run a command inside the workspace, raising on failure or timeout.

        httpx.ReadTimeout bubbles out of execute_command with the unhelpful
        message 'timed out'.  This wrapper catches *any* exception and
        re-raises with the command text so we can identify the culprit.
        """
        try:
            result = workspace.execute_command(cmd, timeout=timeout)
        except Exception as exc:
            raise RuntimeError(f"Command raised {type(exc).__name__}: {exc}\n  cmd: {cmd}") from exc
        if result.exit_code != 0:
            raise RuntimeError(
                f"Command failed (exit {result.exit_code}): {result.stderr}\n  cmd: {cmd}"
            )

    try:
        _run(f"git clone https://github.com/{repo_name}.git {str(instance_path)}", timeout=120.0)
        _run(f"cd {str(instance_path)} && git apply <<'EOF'\n{patch}\nEOF", timeout=120.0)
    except Exception as e:
        raise RuntimeError(f"Error preparing workspace for instance {instance_id}: {e}")
    return True, instance_path, workspace

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
                "include_stop_str_in_output": False,
                "chat_template_kwargs": {
                    # "add_generation_prompt": True, #NOTE: setting this to true raises errors
                    "enable_thinking": False
                }
            }
        )
    return llm

async def cleanup_resources(agent, env):
    if env is not None:
        await env.close()
        env = None

async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = agent_wrapper_platoon = None
    try:
        if config.verbose:
            print(f"[run_rollout] Process {os.getpid()}: Starting rollout for task {task.id}", flush=True)
        instance: dict = task.misc
        try:
            loop = asyncio.get_event_loop()
            # Run in a separate thread to avoid blocking the event loop.
            status, working_dir, workspace = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                prepare_workspace,
                instance
            )
        except Exception as e:
            raise RuntimeError(
                f"Workspace setup failed for task {task.id}: {e}"
            )
        if not status or working_dir is None:
            raise RuntimeError(f"Workspace setup failed for task {task.id}")
        
        user_prompt_filename = USER_PROMPT_FILENAME
        prompt_dir = (Path(__file__).parent / "prompts").resolve()
        user_prompt_path = prompt_dir / user_prompt_filename
        assert user_prompt_path.exists(), f"User prompt path {user_prompt_path} not found"
        input_message = get_instruction(instance, str(user_prompt_path), str(working_dir))

        task.goal = input_message
        task.max_steps = config.max_steps if config.max_steps is not None else 6

        llm: LLM = prepare_llm(config)
        agent: Agent = Agent(
            llm=llm,
            tools=[Tool(name=TerminalTool.name), Tool(name="LocalizationFinishTool")],
            system_prompt_filename="/app/prompts_codescout/system_prompt.j2",
            include_default_tools=[]
        )
        agent_wrapper_platoon: OpenHandsAgent = OpenHandsAgent()
        env: CodeScoutEnv = CodeScoutEnv(task=task, agent=agent, workspace=workspace)

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
            raise
        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout failed for task {task.id}: {str(e)}", flush=True)
            raise

        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as _:
            pass
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get() 
    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as _:
            pass
        raise
    finally:
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as _:
            pass