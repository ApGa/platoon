import os
from platoon.generators.types import RolloutGeneratorConfig
from platoon.envs.base import Task
from platoon.envs.agent_sdk_swe.env import AgentSDKSWEEnv
from platoon.utils.llm_client import LLMClient
import subprocess
from pathlib import Path
from openhands.sdk import LLM
from platoon.train.areal_integration import ArealLLMClient
from platoon.train.areal_integration import get_or_start_openai_compat_server
from pydantic import SecretStr
from platoon.episode.trajectory import TrajectoryCollection
from platoon.visualization.event_sinks import JsonlFileSink
from platoon.episode.context import current_trajectory_collection
from platoon.agents.agent_sdk_swe.agent import AgentSDKSWEAgent
from openhands.tools.preset.default import get_default_agent
from openhands.sdk import Agent # TODO: the default agent has file tracker tool, do we need it??
import asyncio
from contextlib import suppress
from platoon.episode.loop import run_episode
from platoon.envs.agent_sdk_swe.tasks import get_task
from openhands.workspace import DockerWorkspace, APIRemoteWorkspace
from openhands.sdk.workspace import RemoteWorkspace
from openhands.agent_server.docker.build import SDK_VERSION, _base_slug
from openhands.sdk import get_logger

logger = get_logger(__name__)

def get_official_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/xingyaoww/", #NOTE: default changed to match SWE-Gym
) -> str:
    # TODO: this is mostly correct for SWE-Gym but probably not for other datasets
    # Official SWE-Bench image
    # swebench/sweb.eval.x86_64.django_1776_django-11333:v1
    # SWE-Gym image: docker.io/xingyaoww/sweb.eval.x86_64.project-monai_s_monai-6969
    image_name = 'sweb.eval.x86_64.' + instance_id
    image_name = image_name.replace('__', '_s_')  # to comply with docker image naming convention
    official_image_name = (docker_image_prefix.rstrip('/') + '/' + image_name).lower()
    logger.info(f"Official SWE-Bench image: {official_image_name}")
    return official_image_name


def get_agent_server_docker_image(
    instance_id: str,
    docker_image_prefix="docker.io/xingyaoww/", #NOTE: default changed to match SWE-Gym
    target: str = "source-minimal",
) -> str:
    official_image_name = get_official_docker_image(instance_id, docker_image_prefix)
    # TODO: check correctness for SWE-Gym, and it depends on how we name the pre-built agent-server images with SWE-Gym images as the base image.
    return (
        "ghcr.io/all-hands-ai/agent-server"
        + f":v{SDK_VERSION}_{_base_slug(official_image_name)}_{target}-dev"
    )

# TODO: the below function is borrowed from main branch of benchmarks repo. Is it correct for SWE-Gym as well?
ENV_SETUP_COMMANDS = ["export PIP_CACHE_DIR=~/.cache/pip"]
def prepare_workspace(task: Task) -> RemoteWorkspace:
        """
        # TODO: which workspace to use --> DockerWorkspace or APIRemoteWorkspace??
        """
        SKIP_BUILD = os.getenv("SKIP_BUILD", "1").lower() in ("1", "true", "yes")
        logger.info(f"SKIP_BUILD={SKIP_BUILD}")
        instance_id = task.misc["instance_id"]
        assert instance_id is not None, "Task must have 'instance_id' in misc field."
        if SKIP_BUILD:
            agent_server_image = get_agent_server_docker_image(instance_id, "docker.io/xingyaoww/")
            workspace = DockerWorkspace(
                server_image=agent_server_image,
                working_dir="/workspace",
            )
        else:
            official_docker_image = get_official_docker_image(instance_id)
            workspace = DockerWorkspace(
                base_image=official_docker_image,
                working_dir="/workspace",
                target="source-minimal",
            )
            logger.info(
                f"Building workspace from {official_docker_image}. "
                "This may take a while...\n"
                "You can run benchmarks/swe_bench/build_images.py and set "
                "SWE_BENCH_SKIP_BUILD=1 to skip building and use pre-built "
                "agent-server image."
            )
        
        for cmd in ENV_SETUP_COMMANDS:
            res = workspace.execute_command(cmd)
            if res.exit_code != 0:
                raise RuntimeError(
                    f"Failed to run env setup command '{cmd}': {res.stderr}"
                )
            logger.debug(f"Ran env setup command '{cmd}': {res.stdout}")
        return workspace

def run_single_rollout_process(args: tuple[str, dict]) -> dict:
    instance_id, config_dict = args
    config = RolloutGeneratorConfig(**config_dict)

    async def run_rollout() -> dict:
        agent = env = None
        try:
            task = get_task(instance_id)
            # NOTE: task.misc should be a dictionary containing all the key, value pairs from the original huggingface dataset's instance
            workspace = prepare_workspace(task)

            if not isinstance(workspace, RemoteWorkspace):
                raise ValueError(f"Failed to setup remote workspace for instance {task.misc['instance_id']}")


            if hasattr(config, 'llm_client') and config.llm_client is not None: 
                llm_client = config.llm_client  
            else:
                llm_client = LLMClient(model=config.model_name, base_url=config.model_endpoint)

            base_url = config.model_endpoint
            if isinstance(llm_client, ArealLLMClient):
                engine = llm_client.proxy_engine
                # TODO: how to make the LLM endpoint publicly accessible? Is setting host to 0.0.0.0 sufficient assuming the machine has a public IP?
                base_url = get_or_start_openai_compat_server(engine, config.model_name, "0.0.0.0")
            
            api_key = llm_client.api_key if isinstance(llm_client, LLMClient) else 'NONE'
            # TODO: how to configure LLM's generation config here?
            llm = LLM(model=config.model_name, api_key=SecretStr(api_key), base_url=base_url, usage_id="agent")

            # TODO: configure agent's tools here instead of using the default agent?
            env = AgentSDKSWEEnv(task=task, agent=get_default_agent(llm=llm, cli_mode=False), workspace=workspace)
            agent = AgentSDKSWEAgent()


            traj_collection = TrajectoryCollection()
            current_trajectory_collection.set(traj_collection)
            # Stream events to a JSONL file under a common directory for live TUI consumption
            events_dir = os.path.join(config.output_dir, "events")
            events_path = os.path.join(
                events_dir, f"events_{task.id}_{traj_collection.id}.jsonl"
            )
            traj_collection.register_event_handlers(
                JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
            )

            if config.verbose:
                print(f"Process {os.getpid()}: Starting rollout for task {task.id}")

            # Enforce per-rollout timeout so that a hanging rollout is actually cancelled
            rollout_task = asyncio.create_task(run_episode(agent, env))
            try:
                final_obs = await rollout_task
            except asyncio.TimeoutError:
                if config.verbose:
                    print(f"Process {os.getpid()}: Rollout timed out for task {task.id}")
                rollout_task.cancel()
                with suppress(asyncio.CancelledError):
                    await rollout_task
                raise

        except Exception as e:
            if config.verbose:
                print(f"Process {os.getpid()}: Failed rollout for task {task.id}: {e}")
            raise
        finally:
            if agent is not None:
                await agent.close()
            if env is not None:
                await env.close()
            
            return current_trajectory_collection.get().to_dict()

    return asyncio.run(run_rollout())