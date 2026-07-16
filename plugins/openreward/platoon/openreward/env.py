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

_READ_ONLY_OPENREWARD_TOOL_NAMES = frozenset(
    {"get_task", "get_status", "get_tool_details", "view"}
)
_READ_ONLY_SUBAGENT_GOAL_SUFFIX = (
    "Environment access for this child is read-only. You inspect the parent's "
    "live OpenReward workspace; this is not an independent worktree. Use the "
    "available inspection tools and return concrete evidence, file/line "
    "references, and proposed replacements or patch text to the parent with "
    "`finish`. You cannot edit files or submit the environment result. The "
    "parent alone is responsible for applying changes and submitting."
)


def _finished_reward_payload(event: Event) -> dict[str, Any] | None:
    observation = getattr(event, "observation", None)
    tool_name = getattr(observation, "tool_name", None)
    if not isinstance(tool_name, str):
        return None

    for block in getattr(observation, "content", []) or []:
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("finished") is not True:
            continue

        reward = payload.get("reward")
        has_numeric_reward = isinstance(reward, (int, float)) and not isinstance(reward, bool)
        # Preserve the bridge's explicit claim_done completion signal, while
        # also accepting environments whose own terminal tool (for example
        # submit_answer) returns the final reward directly.
        if tool_name == "claim_done" or has_numeric_reward:
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


def _filter_openreward_tools(
    tools: dict[str, Any],
    subagent_environment_access: str,
) -> dict[str, Any]:
    if subagent_environment_access == "shared":
        if "claim_done" not in tools:
            return tools
        return {name: tool for name, tool in tools.items() if name != "claim_done"}
    if subagent_environment_access == "read_only":
        # This is deliberately an allowlist, not a mutator denylist. Generic
        # dispatch/code tools (call_tool and python_execute) and unknown future
        # tools could write or submit indirectly, so they are not safe here.
        return {
            name: tool
            for name, tool in tools.items()
            if name in _READ_ONLY_OPENREWARD_TOOL_NAMES
        }
    raise ValueError(
        "subagent_environment_access must be 'shared' or 'read_only', got "
        f"{subagent_environment_access!r}"
    )


def _openreward_mcp_tools(
    agent: Any,
    subagent_environment_access: str = "shared",
) -> dict[str, Any]:
    tools = {
        name: tool
        for name, tool in agent.tools_map.items()
        if getattr(tool, "mcp_server_name", None) == "openreward"
    }
    return {
        name: _non_owning_tool(tool)
        for name, tool in _filter_openreward_tools(
            tools,
            subagent_environment_access,
        ).items()
    }


def _inject_shared_openreward_tools(
    agent: Any,
    tools: dict[str, Any],
    subagent_environment_access: str,
) -> None:
    existing_tools = agent.tools_map
    if subagent_environment_access == "read_only":
        # A copied SDK agent should not retain initialized MCP tools after its
        # mcp_config is cleared. Remove any such stale tools defensively before
        # installing the strict child allowlist.
        existing_tools = {
            name: tool
            for name, tool in existing_tools.items()
            if getattr(tool, "mcp_server_name", None) != "openreward"
        }
    elif not tools:
        return

    agent._tools = {**existing_tools, **tools}


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
        (
            "When you are done, call `finish`. Its message is returned verbatim "
            "to the parent, so include only the requested answer, data, and "
            "essential evidence. If the parent asks for a structured format, "
            "return exactly that format. Avoid process notes, budget notes, and "
            "internal event bookkeeping unless they are necessary for the task."
        ),
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
        subagent_environment_access: str = "shared",
        **kwargs,
    ):
        self._final_payload: dict[str, Any] | None = None
        self._final_reward_consumed = False
        self._initial_goal_suffix = initial_goal_suffix
        self._shared_openreward_tools = shared_openreward_tools or {}
        if subagent_environment_access not in {"shared", "read_only"}:
            raise ValueError(
                "subagent_environment_access must be 'shared' or 'read_only'"
            )
        self._subagent_environment_access = subagent_environment_access
        self._external_callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks = [*self._external_callbacks, self._stop_on_openreward_finished]
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def _shared_tools_for_fork(self) -> dict[str, Any]:
        if self._shared_openreward_tools:
            return _filter_openreward_tools(
                self._shared_openreward_tools,
                self._subagent_environment_access,
            )
        if self._conversation is None:
            return {}
        conversation = cast(Any, self._conversation)
        conversation._ensure_agent_ready()
        return _openreward_mcp_tools(
            conversation.agent,
            self._subagent_environment_access,
        )

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
            subagent_environment_access=self._subagent_environment_access,
        )

    async def _initial_user_message(self) -> str:
        payload = await asyncio.to_thread(self._get_openreward_task_payload)
        initial_user_message = _format_openreward_task_goal(
            task=self._task,
            payload=payload,
            initial_goal_suffix=self._initial_goal_suffix,
        )
        if (
            isinstance(self._task, SubTask)
            and self._subagent_environment_access == "read_only"
        ):
            initial_user_message = _append_suffix(
                initial_user_message,
                _READ_ONLY_SUBAGENT_GOAL_SUFFIX,
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
            self._subagent_environment_access,
        )
        tool = conversation.agent.tools_map.get("get_task")
        if tool is None:
            raise RuntimeError("OpenReward MCP bridge did not expose get_task")

        action = tool.action_from_arguments({})
        return _observation_json_payload(tool(action, conversation))

    def _stop_on_openreward_finished(self, event: Event) -> None:
        payload = _finished_reward_payload(event)
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
