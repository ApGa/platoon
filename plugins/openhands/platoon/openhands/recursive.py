from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import Future
from contextvars import Context
from copy import deepcopy
from typing import Any, cast
from uuid import uuid4

from platoon.agents.actions.subagent import launch_subagent
from pydantic import Field

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.context import AgentContext
from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    Tool,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.task_tracker import TaskTrackerTool

PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX = (
    "When programmatic_tool_calling (PTC) is available, use it for multi-step tool "
    "orchestration, persistent Python state, and concurrent independent work.\n\n"
    "Inside programmatic_tool_calling, OpenHands tools are available as Python "
    "callables. Use `await asyncio.gather(...)` when launching independent async "
    "tool calls concurrently. Store observations in variables, inspect their "
    "`.text` when needed, combine the results in Python, and then continue with "
    "the task."
)

PROGRAMMATIC_TOOL_CALLING_ORCHESTRATION_ONLY_SYSTEM_PROMPT_SUFFIX = (
    "PTC is in orchestration-only mode. Its Python runtime is outside the task "
    "environment, so do not use Python file, operating-system, process, shell, "
    "or network APIs to interact with the task. Call the environment tools "
    "available through `tools` or `atools` instead. Catalog-only tools must be "
    "invoked through the environment's advertised dispatcher/meta-tool."
)

TASK_TRACKER_SYSTEM_PROMPT_SUFFIX = (
    "For every nontrivial multi-step task, use `task_tracker` to maintain a "
    "short plan. Create the plan before beginning substantive environment work, "
    "and update it as work completes or the plan changes. Skip the tracker only "
    "for a genuinely atomic task that can be completed in one or two tool calls."
)

TASK_TRACKER_INITIAL_TASK_SUFFIX = (
    "Task-tracking guidance: for nontrivial multi-step work, use `task_tracker` "
    "to maintain a short plan. Create it before beginning substantive environment "
    "work and update it as work completes or the plan changes."
)

SHARED_WORKSPACE_SUBAGENT_SYSTEM_PROMPT_SUFFIX = (
    "Subagent workspace coordination: you start in the same live workspace as "
    "your parent, and sibling subagents may use that workspace concurrently. "
    "Changes are visible across agents immediately, so inspect the current state "
    "before editing and preserve unrelated concurrent work. Use the shared "
    "workspace directly when that is safe. If your assigned work needs isolation "
    "to avoid conflicts, create an isolated directory or Git worktree only when "
    "the available tools and repository support it. Before calling `finish`, "
    "either integrate the intended changes into the parent's shared workspace or "
    "clearly tell the parent where the isolated changes, commit, or patch are and "
    "what remains to integrate."
)

RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX = (
    "For multi-part tasks, actively look for independent work that can run in "
    "parallel or make progress while you continue planning. Prefer delegating "
    "self-contained investigation, verification, summarization, or data-gathering "
    "subtasks to launch_subagent instead of doing all work in the root agent. "
    "Give each child a clear, self-contained goal.\n\n"
    "If programmatic_tool_calling is also available, use "
    "`await atools.launch_subagent(goal=...)` from Python, and "
    "`await asyncio.gather(...)` to run independent child agents concurrently.\n\n"
    "At the start of a nontrivial task, consider whether at least one subagent "
    "can help decompose the work before you perform all tool calls yourself. "
    "Launch subagents only for work that can make progress independently."
)

RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX = (
    "For this task, strongly prefer using recursive subagents for independent "
    "subtasks. After reading the task, create or update a task_tracker plan. "
    "When the task has self-contained data-gathering, verification, analysis, "
    "or artifact-building subtasks, the first action after creating the plan "
    "should launch at least one subagent for a plan item unless there is a "
    "clear reason it would not help. When programmatic_tool_calling is "
    "available, use `await atools.launch_subagent(...)` and "
    "`await asyncio.gather(...)` for independent child work."
)

RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX = (
    "Recursive coordination guidance: for nontrivial multi-step work, use "
    "`task_tracker` to maintain a plan. When a plan item is self-contained, "
    "delegate it with `launch_subagent` instead of doing all work yourself. "
    "When programmatic_tool_calling is available, prefer code shaped like "
    '`child = await atools.launch_subagent(goal="...")`. If '
    "there are multiple independent child tasks, use `await asyncio.gather(...)` "
    "to run them concurrently."
)

LAUNCH_SUBAGENT_TOOL_NAME = "launch_subagent"
PARALLEL_TASK_TRACKER_TOOL_SPEC_NAME = "platoon_parallel_task_tracker"
DEFAULT_SUBAGENT_MAX_STEPS = 50
FINISH_TOOL_CLASS_NAME = "FinishTool"

_RUNTIMES: dict[str, "LaunchSubagentRuntime"] = {}
_RUNTIMES_LOCK = threading.Lock()


class LaunchSubagentAction(Action):
    goal: str = Field(description="Task goal for the child agent.")


class LaunchSubagentObservation(Observation):
    pass


def _parallel_task_tracker_description(description: str) -> str:
    description = description.replace(" (maintain single focus)", "")
    description = description.replace(
        "   - Limit active work to ONE task at any given time\n",
        "   - Multiple delegated items may be in progress concurrently\n",
    )
    return description.replace(
        "   - Complete current activities before initiating new ones\n",
        "",
    )


class ParallelTaskTrackerTool(TaskTrackerTool):
    name = TaskTrackerTool.name

    @classmethod
    def create(cls, conv_state: Any) -> Sequence["ParallelTaskTrackerTool"]:
        return [
            tool.model_copy(
                update={
                    "description": _parallel_task_tracker_description(
                        tool.description
                    )
                }
            )
            for tool in super().create(conv_state)
        ]


register_tool(PARALLEL_TASK_TRACKER_TOOL_SPEC_NAME, ParallelTaskTrackerTool)


class LaunchSubagentRuntime:
    def __init__(self) -> None:
        self.id = str(uuid4())
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._context: Context | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._waiters: dict[asyncio.Task[Any], Future[Any]] = {}
        self._closed = False
        with _RUNTIMES_LOCK:
            _RUNTIMES[self.id] = self

    def bind(self, loop: asyncio.AbstractEventLoop, context: Context) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot bind a closed launch_subagent runtime")
            self._loop = loop
            self._context = context

    def close(self) -> None:
        with _RUNTIMES_LOCK:
            _RUNTIMES.pop(self.id, None)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop

        def _cancel_all() -> None:
            with self._lock:
                tasks = list(self._tasks)
                waiters = list(self._waiters.values())
            # Cancel the thread-facing Futures as well as their asyncio Tasks so
            # a synchronous OpenHands tool executor unblocks immediately while
            # cancellation cascades through child/grandchild episode cleanup.
            for waiter in waiters:
                waiter.cancel()
            for task in tasks:
                task.cancel()

        if loop is None or loop.is_closed():
            # asyncio Tasks cannot be safely touched from this thread once the
            # loop is gone, but concurrent.futures.Future is thread-safe.  At
            # least unblock every synchronous tool-executor caller.
            with self._lock:
                waiters = list(self._waiters.values())
            for waiter in waiters:
                waiter.cancel()
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            _cancel_all()
        else:
            try:
                loop.call_soon_threadsafe(_cancel_all)
            except RuntimeError:
                # The loop can close after is_closed() above.  Preserve the
                # same no-hanging-waiter guarantee in that race.
                with self._lock:
                    waiters = list(self._waiters.values())
                for waiter in waiters:
                    waiter.cancel()

    async def aclose(self, timeout: float = 10.0) -> None:
        """Cancel and briefly await every recursive child owned by this runtime."""

        self.close()
        with self._lock:
            tasks = list(self._tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        # Never await cancellation without a bound here: a third-party SDK
        # coroutine may suppress CancelledError.  The rollout subprocess's
        # process-tree timeout remains the final backstop for such a task.
        # Retrieve exceptions from completed tasks so teardown does not emit
        # "Task exception was never retrieved" warnings.
        for task in done:
            if not task.cancelled():
                task.exception()

    def run(
        self,
        *,
        goal: str,
        max_steps: int,
        task_misc: dict[str, Any] | None,
        verbose: bool,
    ) -> Any:
        with self._lock:
            loop = self._loop
            context = self._context
            closed = self._closed
        if closed:
            raise RuntimeError("launch_subagent runtime is closed")
        if loop is None or context is None:
            raise RuntimeError("launch_subagent runtime is not bound to an episode")
        if loop.is_closed():
            raise RuntimeError("launch_subagent runtime loop is closed")

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            raise RuntimeError("launch_subagent cannot block from the episode loop")

        future: Future[Any] = Future()

        async def _run_subagent() -> Any:
            # Stream redirection mutates process-global state and cannot be
            # held safely across an await: concurrent child agents can restore
            # each other's streams out of order. Child logging already follows
            # the rollout process's configured sinks, so schedule it directly.
            try:
                result = await launch_subagent(
                    goal=goal,
                    max_steps=max_steps,
                    task_misc=task_misc,
                    verbose=verbose,
                )
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except BaseException as exc:
                if not future.done():
                    future.set_exception(exc)
                raise
            else:
                # Resolve the thread-facing future inside the coroutine rather
                # than waiting for a Task done callback on a later loop turn.
                # This also wakes asyncio.to_thread's completion path promptly
                # when the child finishes without ever suspending.
                if not future.done():
                    future.set_result(result)
                return result

        def _start() -> None:
            with self._lock:
                if self._closed:
                    future.cancel()
                    return
            try:
                task = asyncio.create_task(_run_subagent())
            except BaseException as exc:
                future.set_exception(exc)
                return
            with self._lock:
                self._tasks.add(task)
                self._waiters[task] = future

            def _finish(done_task: asyncio.Task[Any]) -> None:
                with self._lock:
                    self._tasks.discard(done_task)
                    self._waiters.pop(done_task, None)
                if done_task.cancelled():
                    if not future.done():
                        future.cancel()
                    return
                try:
                    done_task.result()
                except BaseException as exc:
                    # Retrieve the task exception even when ``close()`` already
                    # cancelled the thread-facing Future.  Returning before
                    # ``result()`` would trigger asyncio's un-retrieved-task
                    # warning during teardown.
                    if not future.done():
                        future.set_exception(exc)
                    return

            task.add_done_callback(_finish)

        try:
            loop.call_soon_threadsafe(_start, context=context)
        except RuntimeError as exc:
            # The loop may close between the explicit is_closed() check and
            # scheduling.  Do not block forever on a Future that no loop can
            # complete.
            future.cancel()
            raise RuntimeError("launch_subagent runtime loop closed while scheduling") from exc
        return future.result()


class LaunchSubagentExecutor(ToolExecutor[LaunchSubagentAction, LaunchSubagentObservation]):
    def __init__(self, runtime_id: str, default_max_steps: int) -> None:
        self._runtime_id = runtime_id
        self._default_max_steps = default_max_steps

    def __call__(
        self,
        action: LaunchSubagentAction,
        conversation: Any | None = None,
    ) -> LaunchSubagentObservation:
        _ = conversation
        with _RUNTIMES_LOCK:
            runtime = _RUNTIMES.get(self._runtime_id)
        if runtime is None:
            return LaunchSubagentObservation.from_text(
                "launch_subagent runtime is no longer available.",
                is_error=True,
            )

        try:
            result = runtime.run(
                goal=action.goal,
                max_steps=self._default_max_steps,
                task_misc=None,
                verbose=True,
            )
        except BaseException as exc:
            return LaunchSubagentObservation.from_text(
                f"{exc.__class__.__name__}: {exc}",
                is_error=True,
            )
        return LaunchSubagentObservation.from_text(str(result))


class LaunchSubagentTool(ToolDefinition[LaunchSubagentAction, LaunchSubagentObservation]):
    name = LAUNCH_SUBAGENT_TOOL_NAME

    @classmethod
    def create(
        cls,
        conv_state: Any,
        runtime_id: str,
        default_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS,
    ) -> Sequence["LaunchSubagentTool"]:
        _ = conv_state
        return [
            cls(
                description=(
                    "Launch a recursive Platoon subagent on a child task. The "
                    "child runs with the same forkable agent and environment "
                    "configuration, including programmatic tool calling when enabled. "
                    "Child step budget is configured by the rollout."
                ),
                action_type=LaunchSubagentAction,
                observation_type=LaunchSubagentObservation,
                annotations=ToolAnnotations(
                    title=LAUNCH_SUBAGENT_TOOL_NAME,
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=LaunchSubagentExecutor(
                    runtime_id=runtime_id,
                    default_max_steps=default_max_steps,
                ),
            )
        ]

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)


register_tool(LaunchSubagentTool.name, LaunchSubagentTool)


def _replace_tool(agent: AgentBase, tool: Tool) -> AgentBase:
    tools = [existing for existing in agent.tools if existing.name != tool.name]
    tools.append(tool)
    return cast(AgentBase, agent.model_copy(update={"tools": tools}))


def with_launch_subagent_tool(
    agent: AgentBase,
    *,
    runtime: LaunchSubagentRuntime,
    default_max_steps: int,
) -> AgentBase:
    return _replace_tool(
        agent,
        Tool(
            name=LaunchSubagentTool.name,
            params={
                "runtime_id": runtime.id,
                "default_max_steps": default_max_steps,
            },
        ),
    )


def with_programmatic_tool_calling(
    agent: AgentBase,
    *,
    mode: str = "unrestricted",
    max_tool_calls_per_execution: int = 1024,
) -> AgentBase:
    from openhands.tools.programmatic_tool_calling import ProgrammaticToolCallingTool

    return _replace_tool(
        agent,
        Tool(
            name=ProgrammaticToolCallingTool.name,
            params={
                "mode": mode,
                "max_tool_calls_per_execution": max_tool_calls_per_execution,
            },
        ),
    )


def with_task_tracker_tool(agent: AgentBase) -> AgentBase:
    tools = [
        existing
        for existing in agent.tools
        if existing.name
        not in {TaskTrackerTool.name, PARALLEL_TASK_TRACKER_TOOL_SPEC_NAME}
    ]
    tools.append(Tool(name=PARALLEL_TASK_TRACKER_TOOL_SPEC_NAME))
    return cast(AgentBase, agent.model_copy(update={"tools": tools}))


def with_finish_tool(agent: AgentBase) -> AgentBase:
    if FINISH_TOOL_CLASS_NAME in agent.include_default_tools:
        return agent
    return cast(
        AgentBase,
        agent.model_copy(update={"include_default_tools": [*agent.include_default_tools, FINISH_TOOL_CLASS_NAME]}),
    )


def append_system_message_suffix(agent: AgentBase, suffix: str | None) -> AgentBase:
    if suffix is None or not suffix.strip():
        return agent

    context = agent.agent_context or AgentContext()
    if suffix.strip() in (context.system_message_suffix or ""):
        return agent
    parts = [value.strip() for value in (context.system_message_suffix, suffix) if value is not None and value.strip()]
    merged_context = context.model_copy(update={"system_message_suffix": "\n\n".join(parts)})
    return cast(AgentBase, agent.model_copy(update={"agent_context": merged_context}))


def with_shared_workspace_subagent_prompt(agent: AgentBase) -> AgentBase:
    return append_system_message_suffix(
        agent,
        SHARED_WORKSPACE_SUBAGENT_SYSTEM_PROMPT_SUFFIX,
    )


def append_user_message_suffix(agent: AgentBase, suffix: str | None) -> AgentBase:
    if suffix is None or not suffix.strip():
        return agent

    context = agent.agent_context or AgentContext()
    parts = [value.strip() for value in (context.user_message_suffix, suffix) if value is not None and value.strip()]
    merged_context = context.model_copy(update={"user_message_suffix": "\n\n".join(parts)})
    return cast(AgentBase, agent.model_copy(update={"agent_context": merged_context}))


def copy_agent_config_for_fork(agent: AgentBase) -> AgentBase:
    field_values = {field_name: getattr(agent, field_name) for field_name in type(agent).model_fields}
    if "tools" in field_values:
        field_values["tools"] = list(agent.tools)
    if "include_default_tools" in field_values:
        field_values["include_default_tools"] = list(agent.include_default_tools)
    if "mcp_config" in field_values:
        field_values["mcp_config"] = deepcopy(agent.mcp_config)
    return cast(AgentBase, type(agent).model_validate(field_values))
