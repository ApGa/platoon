from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import io
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENHANDS_PLUGIN_ROOT = REPO_ROOT / "plugins/openhands"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_openhands_stubs(monkeypatch) -> None:
    class AgentBase(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        tools: list[Any] = Field(default_factory=list)
        agent_context: Any | None = None
        include_default_tools: list[str] = Field(default_factory=list)
        mcp_config: dict[str, Any] = Field(default_factory=dict)
        _initialized: bool = PrivateAttr(default=False)
        _runtime_lock: Any | None = PrivateAttr(default=None)

    class AgentContext(BaseModel):
        system_message_suffix: str | None = None
        user_message_suffix: str | None = None

    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        content: list[str] = Field(default_factory=list)
        is_error: bool = False

        @classmethod
        def from_text(cls, text: str, is_error: bool = False, **kwargs):
            return cls(content=[text], is_error=is_error, **kwargs)

        @property
        def text(self) -> str:
            return "\n".join(self.content)

    @dataclass(frozen=True)
    class DeclaredResources:
        keys: tuple[str, ...]
        declared: bool

    class Tool(BaseModel):
        name: str
        params: dict[str, Any] = Field(default_factory=dict)

    class ToolAnnotations(BaseModel):
        title: str | None = None
        readOnlyHint: bool = False
        destructiveHint: bool = True
        idempotentHint: bool = False
        openWorldHint: bool = True

    class ToolExecutor:
        def __class_getitem__(cls, typevar_values):
            return cls

    class ToolDefinition(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        name: ClassVar[str] = ""
        description: str
        action_type: type[Any]
        observation_type: type[Any] | None = None
        annotations: Any | None = None
        executor: Any | None = None

        def __class_getitem__(cls, typevar_values):
            return cls

        def declared_resources(self, action):
            return DeclaredResources(keys=(), declared=False)

    registered_tools: dict[str, Any] = {}

    def register_tool(name: str, factory: Any) -> None:
        registered_tools[name] = factory

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk")
    _module(monkeypatch, "openhands.sdk.agent")
    _module(monkeypatch, "openhands.sdk.agent.base", AgentBase=AgentBase)
    _module(monkeypatch, "openhands.sdk.context", AgentContext=AgentContext)
    _module(
        monkeypatch,
        "openhands.sdk.tool",
        Action=Action,
        DeclaredResources=DeclaredResources,
        Observation=Observation,
        Tool=Tool,
        ToolAnnotations=ToolAnnotations,
        ToolDefinition=ToolDefinition,
        ToolExecutor=ToolExecutor,
        register_tool=register_tool,
        registered_tools=registered_tools,
    )

    class ProgrammaticToolCallingTool:
        name = "programmatic_tool_calling"

    class TaskTrackerTool:
        name = "task_tracker"

    _module(monkeypatch, "openhands.tools")
    _module(
        monkeypatch,
        "openhands.tools.programmatic_tool_calling",
        ProgrammaticToolCallingTool=ProgrammaticToolCallingTool,
    )
    _module(
        monkeypatch,
        "openhands.tools.task_tracker",
        TaskTrackerTool=TaskTrackerTool,
    )


def _load_recursive_module(monkeypatch):
    _install_openhands_stubs(monkeypatch)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    monkeypatch.delitem(sys.modules, "platoon.openhands.recursive", raising=False)

    package = types.ModuleType("platoon.openhands")
    package.__path__ = [str(OPENHANDS_PLUGIN_ROOT / "platoon/openhands")]
    monkeypatch.setitem(sys.modules, "platoon.openhands", package)

    spec = importlib.util.spec_from_file_location(
        "platoon.openhands.recursive",
        OPENHANDS_PLUGIN_ROOT / "platoon/openhands/recursive.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openhands.recursive", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_launch_subagent_runtime_preserves_bound_context(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    marker = contextvars.ContextVar("marker")
    marker.set("parent-context")

    async def fake_launch_subagent(goal, max_steps, task_misc, verbose):
        return {
            "goal": goal,
            "max_steps": max_steps,
            "task_misc": task_misc,
            "verbose": verbose,
            "marker": marker.get(),
        }

    monkeypatch.setattr(recursive, "launch_subagent", fake_launch_subagent)
    runtime = recursive.LaunchSubagentRuntime()
    runtime.bind(asyncio.get_running_loop(), contextvars.copy_context())

    try:
        result = await asyncio.to_thread(
            runtime.run,
            goal="child",
            max_steps=3,
            task_misc={"shard": "left"},
            verbose=False,
        )
    finally:
        runtime.close()

    assert result == {
        "goal": "child",
        "max_steps": 3,
        "task_misc": {"shard": "left"},
        "verbose": False,
        "marker": "parent-context",
    }


@pytest.mark.asyncio
async def test_launch_subagent_runtime_allows_concurrent_children(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def fake_launch_subagent(goal, max_steps, task_misc, verbose):
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        async with lock:
            active -= 1
        return goal

    monkeypatch.setattr(recursive, "launch_subagent", fake_launch_subagent)
    runtime = recursive.LaunchSubagentRuntime()
    runtime.bind(asyncio.get_running_loop(), contextvars.copy_context())

    try:
        results = await asyncio.gather(
            asyncio.to_thread(
                runtime.run,
                goal="child-a",
                max_steps=2,
                task_misc=None,
                verbose=True,
            ),
            asyncio.to_thread(
                runtime.run,
                goal="child-b",
                max_steps=2,
                task_misc=None,
                verbose=True,
            ),
        )
    finally:
        runtime.close()

    assert sorted(results) == ["child-a", "child-b"]
    assert max_active == 2


@pytest.mark.asyncio
async def test_launch_subagent_runtime_uses_real_stdio(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)

    async def fake_launch_subagent(goal, max_steps, task_misc, verbose):
        _ = goal, max_steps, task_misc, verbose
        return sys.stderr is sys.__stderr__ and sys.stdout is sys.__stdout__

    monkeypatch.setattr(recursive, "launch_subagent", fake_launch_subagent)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    runtime = recursive.LaunchSubagentRuntime()
    runtime.bind(asyncio.get_running_loop(), contextvars.copy_context())

    try:
        result = await asyncio.to_thread(
            runtime.run,
            goal="child",
            max_steps=2,
            task_misc=None,
            verbose=True,
        )
    finally:
        runtime.close()

    assert result is True


def test_recursive_helpers_install_tools_and_prompt_suffix(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    from openhands.sdk.agent.base import AgentBase
    from openhands.sdk.context import AgentContext
    from openhands.sdk.tool import Tool

    runtime = recursive.LaunchSubagentRuntime()
    agent = AgentBase(
        tools=[Tool(name="existing")],
        agent_context=AgentContext(
            system_message_suffix="base",
            user_message_suffix="user-base",
        ),
    )

    try:
        agent = recursive.with_task_tracker_tool(agent)
        agent = recursive.with_programmatic_tool_calling(agent)
        agent = recursive.with_launch_subagent_tool(
            agent,
            runtime=runtime,
            default_max_steps=7,
        )
        agent = recursive.with_finish_tool(agent)
        agent = recursive.append_system_message_suffix(agent, "extra")
        agent = recursive.append_user_message_suffix(agent, "user-extra")
    finally:
        runtime.close()

    assert [tool.name for tool in agent.tools] == [
        "existing",
        "task_tracker",
        "programmatic_tool_calling",
        "launch_subagent",
    ]
    assert agent.tools[-1].params == {
        "runtime_id": runtime.id,
        "default_max_steps": 7,
    }
    assert agent.include_default_tools == ["FinishTool"]
    assert agent.agent_context.system_message_suffix == "base\n\nextra"
    assert agent.agent_context.user_message_suffix == "user-base\n\nuser-extra"
    assert "task_tracker tool" in recursive.RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX
    assert "launch at least one subagent" in recursive.RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX
    assert "Recursive coordination guidance" in recursive.RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX
    assert "max_steps" not in recursive.RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX
    assert "max_steps" not in recursive.RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX
    assert "OpenReward" not in recursive.RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX


def test_launch_subagent_tool_declares_no_shared_resources(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    action = recursive.LaunchSubagentAction(goal="child")
    tool = recursive.LaunchSubagentTool.create(
        conv_state=None,
        runtime_id="runtime",
    )[0]

    resources = tool.declared_resources(action)

    assert set(recursive.LaunchSubagentAction.model_fields) == {"goal"}
    assert resources.declared is True
    assert resources.keys == ()


def test_launch_subagent_tool_defaults_to_50_steps(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    tool = recursive.LaunchSubagentTool.create(
        conv_state=None,
        runtime_id="runtime",
    )[0]

    assert recursive.DEFAULT_SUBAGENT_MAX_STEPS == 50
    assert tool.executor._default_max_steps == 50


def test_copy_agent_config_for_fork_drops_private_runtime_state(monkeypatch):
    recursive = _load_recursive_module(monkeypatch)
    from openhands.sdk.agent.base import AgentBase
    from openhands.sdk.tool import Tool

    agent = AgentBase(
        tools=[Tool(name="programmatic_tool_calling")],
        include_default_tools=["FinishTool"],
        mcp_config={"mcpServers": {"openreward": {"command": "python"}}},
    )
    agent._initialized = True
    agent._runtime_lock = threading.Lock()

    copied = recursive.copy_agent_config_for_fork(agent)

    assert copied is not agent
    assert copied.tools == agent.tools
    assert copied.include_default_tools == agent.include_default_tools
    assert copied.mcp_config == agent.mcp_config
    assert copied.mcp_config is not agent.mcp_config
    assert copied._initialized is False
    assert copied._runtime_lock is None


@pytest.mark.asyncio
async def test_programmatic_tool_calling_launches_subagents_mechanically(monkeypatch):
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")
    ptc = pytest.importorskip("openhands.tools.programmatic_tool_calling")
    recursive = pytest.importorskip("platoon.openhands.recursive")

    from platoon.agents.base import ForkableAgent
    from platoon.envs.base import ForkableEnv, Observation, Task
    from platoon.episode.context import (
        budget_tracker,
        current_trajectory,
        current_trajectory_collection,
        finish_message,
    )
    from platoon.episode.loop import run_episode
    from platoon.episode.trajectory import (
        DepthAwareStepBudgetTracker,
        TrajectoryCollection,
        TrajectoryStep,
    )
    from platoon.visualization.event_sinks import JsonlFileSink

    @dataclass
    class ChildRunState:
        active: int = 0
        max_active: int = 0
        grandchild_active: int = 0
        max_grandchild_active: int = 0
        goals: list[str] = field(default_factory=list)
        root_outputs: list[str] = field(default_factory=list)
        root_errors: list[bool] = field(default_factory=list)

    @dataclass
    class DeterministicAgent(ForkableAgent):
        state: ChildRunState
        root_actions: list[Any] = field(default_factory=list)
        action_index: int = 0

        async def act(self, obs: Observation) -> Any:
            assert obs.task is not None
            if obs.task.goal == "root":
                action = self.root_actions[self.action_index]
                self.action_index += 1
                return action
            return {"observed_goal": obs.task.goal or ""}

        async def reset(self) -> None:
            self.action_index = 0

        async def close(self) -> None:
            return None

        async def fork(self, task: Task) -> "DeterministicAgent":
            return DeterministicAgent(state=self.state)

    @dataclass
    class DeterministicEnv(ForkableEnv):
        _task: Task
        state: ChildRunState
        programmatic_tool: Any | None = None
        launch_runtime: Any | None = None
        conversation: Any | None = None

        async def reset(self) -> Observation:
            traj_collection = current_trajectory_collection.get()
            traj_collection.set_trajectory_task(current_trajectory.get().id, self._task)
            if self._task.goal == "root":
                self.launch_runtime = recursive.LaunchSubagentRuntime()
                self.launch_runtime.bind(
                    asyncio.get_running_loop(),
                    contextvars.copy_context(),
                )
                self.programmatic_tool = ptc.ProgrammaticToolCallingTool.create(conv_state=None)[0]
                launch_tool = recursive.LaunchSubagentTool.create(
                    conv_state=None,
                    runtime_id=self.launch_runtime.id,
                    default_max_steps=3,
                )[0]
                self.conversation = types.SimpleNamespace(
                    agent=types.SimpleNamespace(
                        tools_map={
                            self.programmatic_tool.name: self.programmatic_tool,
                            launch_tool.name: launch_tool,
                        }
                    )
                )
            return Observation(task=self._task, finished=False)

        async def step(self, action: Any) -> Observation:
            if self._task.goal == "root":
                assert self.programmatic_tool is not None
                assert self.conversation is not None
                obs = await asyncio.to_thread(
                    self.programmatic_tool.executor,
                    action,
                    self.conversation,
                )
                self.state.root_outputs.append(obs.text)
                self.state.root_errors.append(obs.is_error)
                traj_collection = current_trajectory_collection.get()
                traj_collection.add_trajectory_step(
                    current_trajectory.get().id,
                    TrajectoryStep(
                        misc={
                            "root_ptc_code": action.code,
                            "root_ptc_output": obs.text,
                            "root_ptc_is_error": obs.is_error,
                        }
                    ),
                )
                if len(self.state.root_outputs) >= 2:
                    finish_message.set("root completed")
                    return Observation(task=self._task, finished=True)
                return Observation(task=self._task, finished=False)

            self.state.active += 1
            self.state.max_active = max(self.state.max_active, self.state.active)
            self.state.goals.append(self._task.goal or "")
            is_grandchild = (self._task.goal or "").startswith("revenue ")
            if is_grandchild:
                self.state.grandchild_active += 1
                self.state.max_grandchild_active = max(
                    self.state.max_grandchild_active,
                    self.state.grandchild_active,
                )
            try:
                if self._task.goal == "collect revenue":
                    nested_obs = await self._launch_revenue_grandchildren()
                    step_misc = {
                        "child_action": action,
                        "nested_ptc_output": nested_obs.text,
                    }
                    finish_text = f"completed collect revenue with {nested_obs.text}"
                else:
                    await asyncio.sleep(0.05)
                    step_misc = {"child_action": action}
                    finish_text = f"completed {self._task.goal}"
            finally:
                if is_grandchild:
                    self.state.grandchild_active -= 1
                self.state.active -= 1

            traj_collection = current_trajectory_collection.get()
            traj_collection.add_trajectory_step(
                current_trajectory.get().id,
                TrajectoryStep(misc=step_misc),
            )
            finish_message.set(finish_text)
            return Observation(task=self._task, finished=True)

        async def _launch_revenue_grandchildren(self):
            runtime = recursive.LaunchSubagentRuntime()
            runtime.bind(asyncio.get_running_loop(), contextvars.copy_context())
            programmatic_tool = ptc.ProgrammaticToolCallingTool.create(conv_state=None)[0]
            launch_tool = recursive.LaunchSubagentTool.create(
                conv_state=None,
                runtime_id=runtime.id,
                default_max_steps=2,
            )[0]
            conversation = types.SimpleNamespace(
                agent=types.SimpleNamespace(
                    tools_map={
                        programmatic_tool.name: programmatic_tool,
                        launch_tool.name: launch_tool,
                    }
                )
            )
            action = ptc.ProgrammaticToolCallingAction(
                code=(
                    "north, europe = await asyncio.gather(\n"
                    '    atools.launch_subagent(goal="revenue north"),\n'
                    '    atools.launch_subagent(goal="revenue europe"),\n'
                    ")\n"
                    "grandchild_outputs = [north.text, europe.text]\n"
                    "grandchild_outputs"
                )
            )
            try:
                return await asyncio.to_thread(
                    programmatic_tool.executor,
                    action,
                    conversation,
                )
            finally:
                runtime.close()
                programmatic_tool.executor.close()

        async def close(self) -> None:
            if self.launch_runtime is not None:
                self.launch_runtime.close()
            if self.programmatic_tool is not None:
                self.programmatic_tool.executor.close()
            return None

        async def observe(self) -> Observation:
            return Observation(task=self._task, finished=False)

        @property
        def task(self) -> Task:
            return self._task

        async def fork(self, task: Task) -> "DeterministicEnv":
            return DeterministicEnv(_task=task, state=self.state)

    state = ChildRunState()
    root_task = Task(goal="root", id="root", max_steps=20)
    first_action = ptc.ProgrammaticToolCallingAction(
        code=(
            "child_a, child_b = await asyncio.gather(\n"
            '    atools.launch_subagent(goal="collect revenue"),\n'
            '    atools.launch_subagent(goal="collect support"),\n'
            ")\n"
            "child_outputs = [child_a.text, child_b.text]\n"
            "child_outputs"
        )
    )
    second_action = ptc.ProgrammaticToolCallingAction(code="summary = ' | '.join(sorted(child_outputs))\nsummary")
    root_agent = DeterministicAgent(
        state=state,
        root_actions=[first_action, second_action],
    )
    root_env = DeterministicEnv(_task=root_task, state=state)
    events_path = Path("/private/tmp/openreward-recursive-ptc-recursive-mock-events/events.jsonl")
    traj_collection = TrajectoryCollection()
    traj_collection.register_event_handlers(
        JsonlFileSink(
            events_path,
            collection_id=traj_collection.id,
            process_id="recursive-ptc-smoke",
        )
    )
    tokens = [
        current_trajectory_collection.set(traj_collection),
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=2)),
    ]

    try:
        root_traj = await run_episode(root_agent, root_env, timeout=5)
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    assert len(root_traj.steps) == 2
    assert state.root_errors == [False, False]
    assert "completed collect revenue" in state.root_outputs[0]
    assert "completed collect support" in state.root_outputs[0]
    assert "completed revenue north" in state.root_outputs[0]
    assert "completed revenue europe" in state.root_outputs[0]
    assert "completed collect revenue" in state.root_outputs[1]
    assert "completed collect support" in state.root_outputs[1]
    assert state.max_active >= 2
    assert state.max_grandchild_active == 2
    assert sorted(state.goals) == [
        "collect revenue",
        "collect support",
        "revenue europe",
        "revenue north",
    ]

    trajectories_by_goal = {
        traj.task.goal: traj for traj in traj_collection.trajectories.values() if traj.task is not None
    }
    assert set(trajectories_by_goal) == {
        "root",
        "collect revenue",
        "collect support",
        "revenue north",
        "revenue europe",
    }
    revenue_parent_id = trajectories_by_goal["collect revenue"].id
    assert trajectories_by_goal["revenue north"].parent_info.id == revenue_parent_id
    assert trajectories_by_goal["revenue europe"].parent_info.id == revenue_parent_id
    assert events_path.exists()
    events_text = events_path.read_text()
    assert "revenue north" in events_text
    assert "revenue europe" in events_text
