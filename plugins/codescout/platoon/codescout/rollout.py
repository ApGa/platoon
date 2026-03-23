import os
from jinja2 import Environment, FileSystemLoader
import asyncio
from platoon.envs.base import Task
from platoon.codescout.env import CodeScoutEnv
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
from platoon.codescout.tasks import EVAL_AGENT_SERVER_IMAGE, SDK_SHORT_SHA, ENV_SETUP_COMMANDS, SYSTEM_PROMPT_FILENAME, USER_PROMPT_FILENAME, APPTAINER_CACHE_DIR
from platoon.openhands.agent import OpenHandsAgent
import platform
import uuid
from openhands.sdk.tool import Tool, register_tool
from openhands.tools.terminal import TerminalTool
from platoon.codescout.custom_tools.localization_finish import LocalizationFinishTool  # noqa: F401 - registers the tool with the correct module qualname

logger = get_logger(__name__)

import signal
import subprocess
import threading
from typing import Callable
def _patched_start_container(self: ApptainerWorkspace) -> None:
    """Patched start: ensure the child runs in a new session/process group."""
    # Prepare environment variables
    env_args: list[str] = []
    for key in self.forward_env:
        if key in os.environ:
            env_args += ["--env", f"{key}={os.environ[key]}"]

    # Prepare bind mounts
    bind_args: list[str] = []
    if self.mount_dir:
        mount_path = "/workspace"
        bind_args += ["--bind", f"{self.mount_dir}:{mount_path}"]

    # Build container options
    container_opts: list[str] = []
    if self.use_fakeroot:
        container_opts.append("--fakeroot")
    if self.enable_docker_compat:
        container_opts.append("--compat")
    if self.disable_mount_locations:
        for loc in self.disable_mount_locations:
            container_opts += ["--no-mount", loc]

    server_cmd = [
        "apptainer",
        "run",
        *container_opts,
        *env_args,
        *bind_args,
        self._sif_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(self.host_port),
    ]

    self._process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    if self.detach_logs:
        self._logs_thread = threading.Thread(target=self._stream_logs, daemon=True)
        self._logs_thread.start()


def _patched_cleanup(self: ApptainerWorkspace) -> None:
    """Patched cleanup: terminate the full process group to avoid orphans."""
    if getattr(self, "_instance_name", None):
        self._stop_logs.set()
        if self._logs_thread and self._logs_thread.is_alive():
            self._logs_thread.join(timeout=2)

        if self._process:
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGTERM)
                self._process.wait(timeout=5)
            except Exception:
                try:
                    pgid = os.getpgid(self._process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                    self._process.wait(timeout=2)
                except Exception:
                    pass

        self._process = None
        self._instance_name = None



# ApptainerWorkspace._wait_for_health has a hard-coded default of 120s.
# That is too short when the SIF cache is cold or the agent server is slow
# to start. Patch it to default to 600s instead.
_orig_wait_for_health = ApptainerWorkspace._wait_for_health
def _patched_wait_for_health(self, timeout: float = 600.0) -> None:
    return _orig_wait_for_health(self, timeout=timeout)
ApptainerWorkspace._wait_for_health = _patched_wait_for_health  # type: ignore[method-assign]
ApptainerWorkspace._start_container = _patched_start_container  # type: ignore[method-assign]
ApptainerWorkspace.cleanup = _patched_cleanup  # type: ignore[method-assign]


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
def detect_platform():
    """Detects the correct platform string."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return "linux/arm64"
    return "linux/amd64"

def prepare_workspace2(instance: dict):
    uuid_str = str(uuid.uuid4())[:8]
    workspace = Path(f"/tmp/testbed/{uuid_str}/")
    instance_id: str = instance["instance_id"]
    repo_name: str = instance["repo"]
    patch: str = instance["patch"]

    instance_dir_name = f"{repo_name.replace('/', '_')}_{instance_id}"
    instance_path = workspace / instance_dir_name

    # use the openhands agent server image and then setup env manually
    workspace = ApptainerWorkspace(
        server_image="docker.io/adityasoni8/eval-agent-server:de65ac5-custom-base-image_tag_latest-source", #TODO
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

    # _run("curl -LO https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/ripgrep_14.1.1-1_amd64.deb", timeout=120.0)
    # try:
    #     _run("sudo dpkg -i ripgrep_14.1.1-1_amd64.deb", timeout=180.0)
    # except Exception as e:
    #     pass

    # clone repository inside workspace
    try:
        _run(f"git clone https://<PAT_key>@github.com/{repo_name}.git {str(instance_path)}", timeout=120.0)
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
    
    # register_tool(LocalizationFinishTool.name, LocalizationFinishTool)
    tools = [
        Tool(name=TerminalTool.name),
        # Tool(name="localization_finish"),
    ]
    tools.append(Tool(name="LocalizationFinishTool"))
    from openhands.sdk.agent import Agent
    import types
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_filename="/app/prompts_codescout/system_prompt.j2",
        # system_prompt_filename=system_prompt_path,
        include_default_tools=[]
    )
    # def _initialize(self, state: "ConversationState"):
    #     """Create an AgentBase instance from an AgentSpec."""

    #     if self._tools:
    #         logger.warning("Agent already initialized; skipping re-initialization.")
    #         return

    #     tools: list[ToolDefinition] = []

    #     # Use ThreadPoolExecutor to parallelize tool resolution
    #     with ThreadPoolExecutor(max_workers=4) as executor:
    #         futures = []

    #         # Submit tool resolution tasks
    #         for tool_spec in self.tools:
    #             future = executor.submit(resolve_tool, tool_spec, state)
    #             futures.append(future)

    #         # Submit MCP tools creation if configured
    #         if self.mcp_config:
    #             future = executor.submit(create_mcp_tools, self.mcp_config, 30)
    #             futures.append(future)

    #         # Collect results as they complete
    #         for future in futures:
    #             result = future.result()
    #             tools.extend(result)

    #     logger.info(
    #         f"Loaded {len(tools)} tools from spec: {[tool.name for tool in tools]}"
    #     )
    #     if self.filter_tools_regex:
    #         pattern = re.compile(self.filter_tools_regex)
    #         tools = [tool for tool in tools if pattern.match(tool.name)]
    #         logger.info(
    #             f"Filtered to {len(tools)} tools after applying regex filter: "
    #             f"{[tool.name for tool in tools]}",
    #         )

    #     # Do not include built-in tools; not subject to filtering
    #     # Instantiate built-in tools using their .create() method
    #     # for tool_class in BUILT_IN_TOOLS:
    #     #     tools.extend(tool_class.create(state))

    #     # Check tool types
    #     for tool in tools:
    #         if not isinstance(tool, ToolDefinition):
    #             raise ValueError(
    #                 f"Tool {tool} is not an instance of 'ToolDefinition'. "
    #                 f"Got type: {type(tool)}"
    #             )

    #     # Check name duplicates
    #     tool_names = [tool.name for tool in tools]
    #     if len(tool_names) != len(set(tool_names)):
    #         duplicates = set(name for name in tool_names if tool_names.count(name) > 1)
    #         raise ValueError(f"Duplicate tool names found: {duplicates}")

    #     # Store tools in a dict for easy access
    #     self._tools = {tool.name: tool for tool in tools}

    # agent._initialize = types.MethodType(_initialize, agent)

    return agent

async def cleanup_resources(agent, env):
    if env is not None:
        await env.close()
        env = None

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
        try:
            loop = asyncio.get_event_loop()
            status, working_dir, workspace = await loop.run_in_executor(
                None,  # Uses default ThreadPoolExecutor
                prepare_workspace2,
                instance
            )
        except Exception as e:
            raise RuntimeError(
                f"Workspace setup failed for task {task.id}: {e}"
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
        # env: CodeScoutEnv = CodeScoutEnv(task=task, agent=agent, workspace=str(working_dir))
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
            # The task should already be cancelled by wait_for, but let's be explicit
            raise
        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Rollout failed for task {task.id}: {e}", flush=True)
            raise

        try:
            # if working_dir.exists():
            #     os.system(f"rm -rf {str(working_dir)}")
            #     logger.info(f"Removed workspace {str(working_dir)}")
            print("cleaning up resources", flush=True)
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
            print("cleanup complete", flush=True)
            # asyncio.create_task(cleanup_resources(agent_wrapper_platoon, env), timeout=)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as _:
            pass
        if config.return_dict:
            return current_trajectory_collection.get().to_dict()
        else:
            return current_trajectory_collection.get() 
    except Exception as e:
        if config.verbose:
            print(f"Error running rollout for task {task.id}: {e}", flush=True)
        print("cleaning up resources", flush=True)
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as cleanup_exc:
            pass
        print("cleanup complete", flush=True)
        raise
    finally:
        # pass
        # Safety-net cleanup: ensure env is closed even if run_episode never ran
        # (e.g. error during workspace setup after env was created).
        # env.close() is idempotent — if run_episode already called it,
        # self._conversation will be None and this is a no-op.
        print("cleaning up resources", flush=True)
        try:
            await asyncio.wait_for(cleanup_resources(agent_wrapper_platoon, env), timeout=60)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as cleanup_exc:
            pass
        print("cleanup complete", flush=True)

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
