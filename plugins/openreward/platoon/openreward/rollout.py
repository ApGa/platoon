from __future__ import annotations

import asyncio
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from openhands.sdk import LLM
from openhands.sdk import Agent as OpenHandsSDKAgent
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.openhands.agent import OpenHandsAgent
from platoon.visualization.event_sinks import JsonlFileSink
from pydantic import SecretStr

from platoon.openreward.config_defs import OpenRewardConfig
from platoon.openreward.env import OpenRewardOpenHandsEnv


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "task"


def _openreward_config(config: RolloutConfig) -> OpenRewardConfig:
    extra = config.extra or {}
    return OpenRewardConfig.from_mapping(extra.get("openreward"))


def _build_mcp_config(task: Task, config: RolloutConfig, openreward_config: OpenRewardConfig, output_dir: str) -> dict:
    bridge_args = [
        "-m",
        "platoon.openreward.mcp_bridge",
        "--env-name",
        openreward_config.env_name,
        "--split",
        openreward_config.split,
        "--task-name",
        str(task.id),
        "--session-url",
        openreward_config.session_url,
        "--api-key",
        openreward_config.api_key,
        "--output-dir",
        output_dir,
        "--max-tool-calls",
        str(openreward_config.max_tool_calls),
    ]
    if openreward_config.api_url:
        bridge_args.extend(["--api-url", openreward_config.api_url])

    return {
        "mcpServers": {
            "openreward": {
                "command": sys.executable,
                "args": bridge_args,
                "env": {
                    "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                    "OPENHANDS_SUPPRESS_BANNER": "1",
                },
            }
        }
    }


def _build_llm(config: RolloutConfig) -> LLM:
    inference_params = config.inference_params
    api_key = config.model_api_key
    return LLM(
        usage_id="platoon-openreward-openhands",
        model=config.model_name or "openai/gpt-4o-mini",
        base_url=config.model_endpoint,
        api_key=SecretStr(api_key) if api_key else None,
        temperature=inference_params.temperature,
        top_p=inference_params.top_p,
        max_output_tokens=inference_params.max_completion_tokens,
        timeout=config.step_timeout,
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    openreward_config = _openreward_config(config)
    task_id = str(task.id)
    rollout_id = uuid.uuid4().hex[:8]
    rollout_output_dir = os.path.join(config.output_dir, "openreward", _slug(task_id), rollout_id)
    bridge_output_dir = os.path.join(rollout_output_dir, "bridge")
    workspace_dir = os.path.join(rollout_output_dir, "workspace")
    Path(bridge_output_dir).mkdir(parents=True, exist_ok=True)
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    llm = _build_llm(config)
    oh_agent = OpenHandsSDKAgent(
        llm=llm,
        tools=[],
        mcp_config=_build_mcp_config(task, config, openreward_config, bridge_output_dir),
        include_default_tools=[],
    )
    env = OpenRewardOpenHandsEnv(task=task, agent=oh_agent, workspace=workspace_dir)
    agent = OpenHandsAgent()

    traj_collection = TrajectoryCollection()
    current_trajectory_collection.set(traj_collection)
    events_path = os.path.join(config.output_dir, "events", f"events_{_slug(task_id)}_{traj_collection.id}.jsonl")
    traj_collection.register_event_handlers(
        JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
    )

    try:
        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(rollout_task, timeout=config.timeout)
    except asyncio.TimeoutError:
        rollout_task.cancel()
        try:
            await asyncio.wait_for(rollout_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        raise

    if config.return_dict:
        result = current_trajectory_collection.get().to_dict()
        result["misc"] = {
            "openreward": asdict(openreward_config),
            "rollout_output_dir": rollout_output_dir,
            "bridge_output_dir": bridge_output_dir,
        }
        return result
    return current_trajectory_collection.get()


def reward_processor(traj: dict) -> tuple[float, dict]:
    reward = float(traj.get("reward", 0.0))
    rewards_dict: dict[str, float] = {"reward/success": reward, "reward/openreward": reward}
    for step in traj.get("steps", []):
        reward_misc = step.get("misc", {}).get("reward_misc", {})
        for key, value in reward_misc.items():
            if key.startswith("reward/") and isinstance(value, (int, float)):
                rewards_dict[key] = float(value)
    return reward, rewards_dict
