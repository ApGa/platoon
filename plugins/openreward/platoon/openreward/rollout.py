from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import cast

from openhands.sdk import LLM
from openhands.sdk import Agent as OpenHandsSDKAgent
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from platoon.agents.actions.subagent import (
    SubagentRewardJudgeConfig,
)
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import (
    budget_tracker,
    current_trajectory_collection,
    subagent_reward_judge_config,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, TrajectoryCollection
from platoon.openhands.agent import OpenHandsAgent
from platoon.openhands.recursive import (
    PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX,
    RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX,
    RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX,
    RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX,
    append_system_message_suffix,
    append_user_message_suffix,
    with_programmatic_tool_calling,
    with_task_tracker_tool,
)
from platoon.utils.subagent_rewards import (
    add_direct_subagent_delegation_rewards,
    propogate_root_success,
)
from platoon.visualization.event_sinks import JsonlFileSink

from platoon.openreward.config_defs import OpenRewardConfig
from platoon.openreward.env import OpenRewardOpenHandsEnv

OPENREWARD_CONDENSER_KEEP_FIRST = 2


def _patch_mcp_boot_timeout() -> None:
    """Raise OpenHands' hardcoded 30s MCP tool-listing timeout.

    At episode start OpenHands spawns the openreward mcp_bridge, which boots a
    per-session env (clone Postgres DB + spawn several node/python MCP servers).
    Under high ``num_concurrent_workers`` these boots all land on the gym server
    at once and routinely exceed 30s, so ``Agent._initialize`` raises
    ``MCPTimeoutError`` and the episode dies at step 0. The 30s is a hardcoded
    literal in ``openhands.sdk.agent.base`` (no config hook), so we wrap the
    ``create_mcp_tools`` symbol it calls and force a larger timeout. Override via
    ``OPENREWARD_MCP_TIMEOUT`` (seconds).
    """
    try:
        timeout = float(os.environ.get("OPENREWARD_MCP_TIMEOUT", "120"))
    except (TypeError, ValueError):
        timeout = 120.0
    try:
        from openhands.sdk.agent import base as _oh_agent_base
    except Exception:
        return
    original = getattr(_oh_agent_base, "create_mcp_tools", None)
    if original is None or getattr(original, "_openreward_patched", False):
        return

    def _create_mcp_tools(config, _timeout=30, *args, **kwargs):
        return original(config, timeout, *args, **kwargs)

    setattr(_create_mcp_tools, "_openreward_patched", True)
    _oh_agent_base.create_mcp_tools = _create_mcp_tools


_patch_mcp_boot_timeout()


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in value).strip("-") or "task"


def _openreward_config(config: RolloutConfig) -> OpenRewardConfig:
    extra = config.extra or {}
    return OpenRewardConfig.from_mapping(extra.get("openreward"))


def _select_session_url(default_url: str, shard_key: str) -> str:
    """Pick one env-server backend for this rollout when sharding is enabled.

    Multinode training runs the rollout workflow on a single controller node, so
    every session would otherwise hit one env server and bottleneck. When
    ``OPENREWARD_SESSION_URLS`` (comma-separated) is set, we spread rollouts over
    those backends by hashing a per-rollout key. Each rollout is exactly one ORS
    session, so binding the whole session to one backend keeps the gym's
    in-container session affinity (consistent hash on X-Session-ID) intact end to
    end -- no extra load balancer required. Unset -> use the config's session_url.
    """
    urls = os.environ.get("OPENREWARD_SESSION_URLS", "").strip()
    if not urls:
        return default_url
    candidates = [u.strip() for u in urls.split(",") if u.strip()]
    if not candidates:
        return default_url
    digest = hashlib.sha1(shard_key.encode("utf-8")).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]


def _build_mcp_config(task: Task, config: RolloutConfig, openreward_config: OpenRewardConfig, output_dir: str) -> dict:
    session_url = _select_session_url(openreward_config.session_url, output_dir)
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
        session_url,
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
                    # The MCP SDK launches stdio servers in a detached process
                    # session.  Give the bridge the expected direct parent so
                    # it can detect a rollout worker that died before Linux's
                    # parent-death signal was installed.
                    "PLATOON_ROLLOUT_PARENT_PID": str(os.getpid()),
                },
            }
        }
    }


def _build_llm(config: RolloutConfig) -> LLM:
    inference_params = config.inference_params
    api_key = config.model_api_key
    model_name = config.model_name or "openai/gpt-4o-mini"
    # Keep OpenHands' local token counter on the exact tokenizer/template used
    # by AReaL and SGLang.  This is especially important for local Qwen model
    # overlays: using the old hard-coded Qwen3.5 tokenizer would make condenser
    # thresholds disagree with the prompts that are actually sent for rollout.
    custom_tokenizer = model_name.removeprefix("openai/")
    return LLM(
        usage_id="platoon-openreward-openhands",
        model=model_name,
        base_url=config.model_endpoint,
        api_key=api_key or None,
        temperature=inference_params.temperature,
        top_p=inference_params.top_p,
        max_output_tokens=inference_params.max_completion_tokens,
        timeout=config.step_timeout,
        custom_tokenizer=custom_tokenizer,
    )


def _configure_openhands_agent(
    agent: OpenHandsSDKAgent,
    openreward_config: OpenRewardConfig,
) -> OpenHandsSDKAgent:
    suffix_parts: list[str] = []
    user_suffix_parts: list[str] = []
    configured_agent = agent
    if openreward_config.enable_programmatic_tool_calling:
        configured_agent = cast(
            OpenHandsSDKAgent,
            with_programmatic_tool_calling(configured_agent),
        )
        suffix_parts.append(PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX)
    if openreward_config.enable_recursive_subagents:
        configured_agent = cast(
            OpenHandsSDKAgent,
            with_task_tracker_tool(configured_agent),
        )
        suffix_parts.append(RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX)
        user_suffix_parts.append(RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX)
        if openreward_config.subagent_max_depth is not None:
            suffix_parts.append(
                "Recursive subagents are limited to maximum depth "
                f"{openreward_config.subagent_max_depth}; the root agent is depth 0."
            )
    if openreward_config.openhands_system_prompt_suffix:
        suffix_parts.append(openreward_config.openhands_system_prompt_suffix)

    suffix = "\n\n".join(part.strip() for part in suffix_parts if part.strip())
    user_suffix = "\n\n".join(part.strip() for part in user_suffix_parts if part.strip())
    configured_agent = cast(
        OpenHandsSDKAgent,
        append_system_message_suffix(configured_agent, suffix),
    )
    return cast(
        OpenHandsSDKAgent,
        append_user_message_suffix(configured_agent, user_suffix),
    )


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    openreward_config = _openreward_config(config)
    task_id = str(task.id)
    initial_goal_suffix = (
        RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX if openreward_config.enable_recursive_subagents else None
    )

    rollout_id = uuid.uuid4().hex[:8]
    openhands_conversation_id = uuid.uuid4()
    rollout_output_dir = os.path.join(config.output_dir, "openreward", _slug(task_id), rollout_id)
    bridge_output_dir = os.path.join(rollout_output_dir, "bridge")
    openhands_persistence_dir = os.path.join(rollout_output_dir, "openhands")
    workspace_dir = os.path.join(rollout_output_dir, "workspace")
    Path(bridge_output_dir).mkdir(parents=True, exist_ok=True)
    Path(openhands_persistence_dir).mkdir(parents=True, exist_ok=True)
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

    llm = _build_llm(config)
    oh_agent = _configure_openhands_agent(
        OpenHandsSDKAgent(
            llm=llm,
            tools=[],
            mcp_config=_build_mcp_config(task, config, openreward_config, bridge_output_dir),
            include_default_tools=[],
            condenser=LLMSummarizingCondenser(
                llm=llm,
                keep_first=OPENREWARD_CONDENSER_KEEP_FIRST,
                max_tokens=int(0.8 * 32768),  # TODO: Make this configurable
            ),
        ),
        openreward_config,
    )
    env = OpenRewardOpenHandsEnv(
        task=task,
        agent=oh_agent,
        workspace=workspace_dir,
        persistence_dir=openhands_persistence_dir,
        conversation_id=openhands_conversation_id,
        enable_recursive_subagents=openreward_config.enable_recursive_subagents,
        subagent_default_max_steps=openreward_config.subagent_default_max_steps,
        initial_goal_suffix=initial_goal_suffix,
    )
    agent = OpenHandsAgent()

    traj_collection = TrajectoryCollection()
    events_path = os.path.join(config.output_dir, "events", f"events_{_slug(task_id)}_{traj_collection.id}.jsonl")
    traj_collection.register_event_handlers(
        JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
    )
    tokens = [current_trajectory_collection.set(traj_collection)]
    if openreward_config.enable_recursive_subagents:
        tokens.append(budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=openreward_config.subagent_max_depth)))
    tokens.append(
        subagent_reward_judge_config.set(
            SubagentRewardJudgeConfig(max_steps=openreward_config.subagent_reward_judge_max_steps)
            if openreward_config.enable_subagent_reward_judging
            else None
        )
    )

    rollout_timed_out = False
    try:
        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(rollout_task, timeout=config.timeout)
    except asyncio.TimeoutError:
        # ``asyncio.wait_for`` has already cancelled ``run_episode`` and waited
        # for its bounded cleanup before raising.  The episode loop finalizes
        # and marks every trajectory on the active cancellation stack, so the
        # collection is now a coherent partial result: interrupted policy data
        # is filtered downstream while completed siblings/descendants remain
        # usable.  Root-success propagation cannot safely use a partial root,
        # so preserve the historical discard behavior for that legacy mode.
        if config.propogate_root_success:
            raise
        rollout_timed_out = True
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    result = traj_collection
    delegation_coefficient = openreward_config.subagent_delegation_reward_coefficient
    if delegation_coefficient > 0:
        if config.propogate_root_success:
            raise ValueError(
                "OpenReward delegation rewards require propogate_root_success=false "
                "so direct child verifier scores remain intact"
            )
        add_direct_subagent_delegation_rewards(result, delegation_coefficient)
    if config.propogate_root_success:
        propogate_root_success(result)

    if config.return_dict:
        result = result.to_dict()
        result["misc"] = {
            "openreward": asdict(openreward_config),
            "rollout_output_dir": rollout_output_dir,
            "bridge_output_dir": bridge_output_dir,
            "openhands_persistence_dir": openhands_persistence_dir,
            "openhands_conversation_id": str(openhands_conversation_id),
            "rollout_timed_out": rollout_timed_out,
        }
        return result
    return result
