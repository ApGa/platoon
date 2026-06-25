from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event.base import Event

from platoon.envs.base import SubTask, Task
from platoon.openhands.env import OpenHandsEnv

_CURRENT_AGENT_TASK_GOAL_KEY = "openreward_current_agent_task_goal"
_ROOT_AGENT_TASK_GOAL_KEY = "openreward_root_agent_task_goal"


def _claim_done_payload(event: Event) -> dict[str, Any] | None:
    observation = getattr(event, "observation", None)
    if getattr(observation, "tool_name", None) != "claim_done":
        return None

    for block in getattr(observation, "content", []) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("finished") is True:
            return payload
    return None


class _NonOwningToolExecutor:
    def __init__(self, executor: Any) -> None:
        self._executor = executor

    def __call__(self, action: Any, conversation: Any | None = None) -> Any:
        return self._executor(action, conversation)

    def interrupt(self) -> None:
        interrupt = getattr(self._executor, "interrupt", None)
        if interrupt is not None:
            interrupt()

    def close(self) -> None:
        return None


def _non_owning_tool(tool: Any) -> Any:
    executable_tool = tool.as_executable()
    return tool.set_executor(_NonOwningToolExecutor(executable_tool.executor))


def _openreward_mcp_tools(agent: Any) -> dict[str, Any]:
    return {
        name: _non_owning_tool(tool)
        for name, tool in agent.tools_map.items()
        if getattr(tool, "mcp_server_name", None) == "openreward"
    }


def _inject_shared_openreward_tools(agent: Any, tools: dict[str, Any]) -> None:
    if not tools:
        return
    agent._tools = {**agent.tools_map, **tools}


def _observation_json_payload(observation: Any) -> dict[str, Any]:
    for block in getattr(observation, "content", []) or []:
        text = block if isinstance(block, str) else getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "prompt" in payload:
            return payload
    raise RuntimeError("OpenReward get_task returned no JSON prompt payload")


def _task_context_goal(task: Task) -> str:
    value = task.misc.get(_CURRENT_AGENT_TASK_GOAL_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return (task.goal or "").strip()


def _root_task_context_goal(task: Task, fallback: str) -> str:
    value = task.misc.get(_ROOT_AGENT_TASK_GOAL_KEY)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _append_suffix(message: str, suffix: str | None) -> str:
    if suffix is not None and suffix.strip():
        return "\n\n".join([message.strip(), suffix.strip()])
    return message.strip()


def _format_subagent_task_goal(
    *,
    task: SubTask,
    root_task_goal: str,
    initial_goal_suffix: str | None,
) -> str:
    parts = [
        "You are a sub-agent provided a task by a parent in a recursive tree of agents.",
        f"Your Task:\n{(task.goal or '').strip()}",
    ]

    context_parts: list[str] = []
    if task.parent_tasks:
        parent_task = task.parent_tasks[-1]
        context_parts.append(f"Parent Agent Task:\n{_task_context_goal(parent_task)}")
        root_task = task.parent_tasks[0]
        if root_task is not parent_task:
            context_parts.append(f"Root Agent Task:\n{_task_context_goal(root_task) or root_task_goal}")

    if context_parts:
        parts.append("For additional context:\n\n" + "\n\n".join(context_parts))

    return _append_suffix("\n\n".join(parts), initial_goal_suffix)


def _task_misc_with_prompt_context(
    *,
    task: Task,
    root_task_goal: str,
    current_task_goal: str,
) -> dict[str, Any]:
    return {
        **task.misc,
        _CURRENT_AGENT_TASK_GOAL_KEY: current_task_goal,
        _ROOT_AGENT_TASK_GOAL_KEY: _root_task_context_goal(task, root_task_goal),
    }


def _format_openreward_task_goal(
    *,
    task: Task,
    payload: dict[str, Any],
    initial_goal_suffix: str | None = None,
) -> str:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise RuntimeError("OpenReward get_task payload did not include a prompt")

    if isinstance(task, SubTask):
        return _format_subagent_task_goal(
            task=task,
            root_task_goal=prompt,
            initial_goal_suffix=initial_goal_suffix,
        )

    return _append_suffix(prompt, initial_goal_suffix)


class OpenRewardOpenHandsEnv(OpenHandsEnv):
    def __init__(
        self,
        *args,
        initial_goal_suffix: str | None = None,
        shared_openreward_tools: dict[str, Any] | None = None,
        **kwargs,
    ):
        self._final_payload: dict[str, Any] | None = None
        self._final_reward_consumed = False
        self._initial_goal_suffix = initial_goal_suffix
        self._shared_openreward_tools = shared_openreward_tools or {}
        self._external_callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks = [*self._external_callbacks, self._stop_on_claim_done]
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def _shared_tools_for_fork(self) -> dict[str, Any]:
        if self._shared_openreward_tools:
            return self._shared_openreward_tools
        if self._conversation is None:
            return {}
        conversation = cast(Any, self._conversation)
        conversation._ensure_agent_ready()
        return _openreward_mcp_tools(conversation.agent)

    def _fork_agent_with_shared_openreward_session(self) -> Any:
        from platoon.openhands.recursive import copy_agent_config_for_fork

        agent = copy_agent_config_for_fork(self._agent)
        return agent.model_copy(update={"mcp_config": {}})

    async def fork(self, task: Task) -> "OpenRewardOpenHandsEnv":
        return type(self)(
            task=task,
            agent=self._fork_agent_with_shared_openreward_session(),
            workspace=self._workspace,
            callbacks=self._external_callbacks,
            persistence_dir=self._persistence_dir,
            conversation_id=uuid4() if self._persistence_dir is not None else None,
            enable_recursive_subagents=self._enable_recursive_subagents,
            subagent_default_max_steps=self._subagent_default_max_steps,
            initial_goal_suffix=self._initial_goal_suffix,
            shared_openreward_tools=self._shared_tools_for_fork(),
        )

    async def _initial_user_message(self) -> str:
        payload = await asyncio.to_thread(self._get_openreward_task_payload)
        initial_user_message = _format_openreward_task_goal(
            task=self._task,
            payload=payload,
            initial_goal_suffix=self._initial_goal_suffix,
        )
        current_task_goal = (
            (self._task.goal or "").strip()
            if isinstance(self._task, SubTask)
            else str(payload.get("prompt") or "").strip()
        )
        self._task = replace(
            self._task,
            goal=initial_user_message,
            misc=_task_misc_with_prompt_context(
                task=self._task,
                root_task_goal=str(payload.get("prompt") or "").strip(),
                current_task_goal=current_task_goal,
            ),
        )
        return initial_user_message

    def _get_openreward_task_payload(self) -> dict[str, Any]:
        if self._conversation is None:
            raise RuntimeError("OpenReward conversation has not been initialized")

        conversation = cast(Any, self._conversation)
        conversation._ensure_agent_ready()
        _inject_shared_openreward_tools(
            conversation.agent,
            self._shared_openreward_tools,
        )
        tool = conversation.agent.tools_map.get("get_task")
        if tool is None:
            raise RuntimeError("OpenReward MCP bridge did not expose get_task")

        action = tool.action_from_arguments({})
        return _observation_json_payload(tool(action, conversation))

    def _stop_on_claim_done(self, event: Event) -> None:
        payload = _claim_done_payload(event)
        if payload is None:
            return
        self._final_payload = payload
        if self._conversation is not None:
            self._conversation.state.execution_status = ConversationExecutionStatus.FINISHED

    async def evaluate(self) -> tuple[float, dict]:
        if self._final_payload is None or self._final_reward_consumed:
            return 0.0, {}

        self._final_reward_consumed = True
        reward = self._final_payload.get("reward", 0.0)
        if not isinstance(reward, (int, float)):
            reward = 0.0
        return float(reward), {
            "reward/success": float(reward),
            "reward/openreward": float(reward),
            "openreward/final_payload": self._final_payload,
        }
