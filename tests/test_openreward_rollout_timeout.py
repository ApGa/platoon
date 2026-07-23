from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from platoon.agents.actions.subagent import launch_subagent
from platoon.config_defs import RolloutConfig
from platoon.envs.base import Observation, SubTask, Task
from platoon.episode.context import current_trajectory, current_trajectory_collection
from platoon.episode.trajectory import TrajectoryStep
from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_rollout_module(
    monkeypatch,
    *,
    agent_type=None,
    env_type=None,
    recursive_overrides=None,
):
    class _AcceptsKeywords:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Agent:
        async def reset(self) -> None:
            return None

        async def act(self, observation):
            raise AssertionError("the environment reset should remain blocked")

        async def close(self) -> None:
            return None

    class _HangingEnv:
        def __init__(self, *, task, **kwargs):
            self._task = task

        @property
        def task(self):
            return self._task

        async def reset(self):
            collection = current_trajectory_collection.get()
            root = current_trajectory.get()
            collection.set_trajectory_task(root.id, self._task)
            # Model a child which finished before the root became a straggler.
            # The timeout should finalize only the active root and preserve
            # this completed sibling/descendant in the partial collection.
            child = collection.create_trajectory(parent_traj=root)
            collection.set_trajectory_task(
                child.id,
                Task(id="completed-child", goal="already done", max_steps=1),
            )
            child.reward = 1.0
            child.finish_message = "child done"
            collection.finish_trajectory(child.id)
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def step(self, action):
            raise AssertionError("reset should remain blocked")

        async def observe(self):
            raise AssertionError("observe should not be called")

        async def close(self) -> None:
            return None

    def _identity(agent, *args, **kwargs):
        return agent

    agent_type = agent_type or _Agent
    env_type = env_type or _HangingEnv

    _module(monkeypatch, "openhands")
    _module(monkeypatch, "openhands.sdk", LLM=_AcceptsKeywords, Agent=_AcceptsKeywords)
    _module(monkeypatch, "openhands.sdk.context")
    _module(
        monkeypatch,
        "openhands.sdk.context.condenser",
        LLMSummarizingCondenser=_AcceptsKeywords,
    )
    _module(
        monkeypatch,
        "platoon.agents.actions.subagent",
        SubagentRewardJudgeConfig=_AcceptsKeywords,
    )
    _module(monkeypatch, "platoon.openhands")
    _module(monkeypatch, "platoon.openhands.agent", OpenHandsAgent=agent_type)
    recursive_attrs = {
        "PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX": "",
        "RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX": "",
        "RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX": "",
        "RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX": "",
        "append_system_message_suffix": _identity,
        "append_user_message_suffix": _identity,
        "with_programmatic_tool_calling": _identity,
        "with_task_tracker_tool": _identity,
    }
    recursive_attrs.update(recursive_overrides or {})
    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        **recursive_attrs,
    )
    _module(monkeypatch, "platoon.openreward.env", OpenRewardOpenHandsEnv=env_type)

    package = types.ModuleType("platoon.openreward")
    package.__path__ = [str(OPENREWARD_ROOT)]
    monkeypatch.setitem(sys.modules, "platoon.openreward", package)
    monkeypatch.delitem(sys.modules, "platoon.openreward.config_defs", raising=False)
    monkeypatch.delitem(sys.modules, "platoon.openreward.rollout", raising=False)

    spec = importlib.util.spec_from_file_location(
        "platoon.openreward.rollout",
        OPENREWARD_ROOT / "rollout.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "platoon.openreward.rollout", module)
    spec.loader.exec_module(module)
    return module


def _agent_configuration_spies():
    prompts = {"system": [], "user": []}

    def _with_tool(name):
        def _configure(agent):
            agent.tools.append(name)
            return agent

        return _configure

    def _append_prompt(kind):
        def _append(agent, suffix):
            prompts[kind].append(suffix)
            return agent

        return _append

    return prompts, {
        "PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX": "ptc-guidance",
        "RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX": "recursive-initial-guidance",
        "RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX": "recursive-system-guidance",
        "RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX": "recursive-user-guidance",
        "append_system_message_suffix": _append_prompt("system"),
        "append_user_message_suffix": _append_prompt("user"),
        "with_programmatic_tool_calling": _with_tool("programmatic_tool_calling"),
        "with_task_tracker_tool": _with_tool("task_tracker"),
    }


def test_nonrecursive_ptc_and_task_tracker_do_not_add_delegation(monkeypatch) -> None:
    prompts, recursive_overrides = _agent_configuration_spies()
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    agent = types.SimpleNamespace(tools=[])
    config = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_programmatic_tool_calling": True,
            "enable_task_tracker": True,
            "enable_recursive_subagents": False,
        }
    )

    configured = rollout._configure_openhands_agent(agent, config)

    assert configured.tools == ["programmatic_tool_calling", "task_tracker"]
    assert "launch_subagent" not in configured.tools
    assert prompts["system"] == ["ptc-guidance"]
    assert prompts["user"] == [""]
    assert all("recursive" not in prompt for values in prompts.values() for prompt in values)


def test_recursive_mode_still_installs_task_tracker_and_delegation_prompts(monkeypatch) -> None:
    prompts, recursive_overrides = _agent_configuration_spies()
    rollout = _load_rollout_module(
        monkeypatch,
        recursive_overrides=recursive_overrides,
    )
    agent = types.SimpleNamespace(tools=[])
    config = rollout.OpenRewardConfig.from_mapping(
        {
            "enable_recursive_subagents": True,
            "subagent_max_depth": 2,
        }
    )

    configured = rollout._configure_openhands_agent(agent, config)

    assert configured.tools == ["task_tracker"]
    assert prompts["system"] == [
        "recursive-system-guidance\n\nRecursive subagents are limited to maximum depth 2; the root agent is depth 0."
    ]
    assert prompts["user"] == ["recursive-user-guidance"]


def _counting_rollout_types(*, child_max_steps: int | None = None):
    class CountingAgent:
        async def reset(self) -> None:
            return None

        async def act(self, observation):
            return "continue"

        async def close(self) -> None:
            return None

        async def fork(self, task: Task):
            return type(self)()

    class CountingEnv:
        instances = []

        def __init__(self, *, task, **kwargs):
            self._task = task
            self._launched_child = False
            self.step_calls = 0
            self.instances.append(self)

        @property
        def task(self):
            return self._task

        async def reset(self):
            collection = current_trajectory_collection.get()
            trajectory = current_trajectory.get()
            collection.set_trajectory_task(trajectory.id, self._task)
            # OpenHands records its initial observation as a trajectory step.
            collection.add_trajectory_step(trajectory.id, TrajectoryStep())
            return Observation(task=self._task)

        async def step(self, action):
            if (
                child_max_steps is not None
                and not isinstance(self._task, SubTask)
                and not self._launched_child
            ):
                self._launched_child = True
                await launch_subagent(
                    goal="bounded child",
                    max_steps=child_max_steps,
                )
            self.step_calls += 1
            collection = current_trajectory_collection.get()
            trajectory = current_trajectory.get()
            collection.add_trajectory_step(trajectory.id, TrajectoryStep())
            return Observation(task=self._task)

        async def observe(self):
            return Observation(task=self._task)

        async def close(self) -> None:
            return None

        async def fork(self, task: Task):
            return type(self)(task=task)

    return CountingAgent, CountingEnv


@pytest.mark.asyncio
async def test_openreward_rollout_timeout_returns_marked_partial_collection(monkeypatch, tmp_path) -> None:
    rollout = _load_rollout_module(monkeypatch)
    config = RolloutConfig(
        output_dir=str(tmp_path),
        timeout=0.01,
        step_timeout=60,
        return_dict=True,
        propogate_root_success=False,
    )

    result = await rollout.run_rollout(
        Task(id="partial-timeout", goal="wait forever", max_steps=1),
        config,
    )

    assert result["misc"]["rollout_timed_out"] is True
    assert len(result["trajectories"]) == 2
    root, child = result["trajectories"].values()
    assert root["misc"][TRAJECTORY_CANCELLED_MISC_KEY] is True
    assert "Episode cancelled" in root["error_message"]
    assert child["finish_message"] == "child done"
    assert TRAJECTORY_CANCELLED_MISC_KEY not in child["misc"]


@pytest.mark.asyncio
async def test_openreward_rollout_applies_configured_root_step_cap(monkeypatch, tmp_path) -> None:
    agent_type, env_type = _counting_rollout_types()
    rollout = _load_rollout_module(
        monkeypatch,
        agent_type=agent_type,
        env_type=env_type,
    )
    source_task = Task(id="bounded-root", goal="keep going")
    config = RolloutConfig(
        output_dir=str(tmp_path),
        max_steps=3,
        timeout=5,
        step_timeout=1,
        return_dict=True,
    )

    result = await rollout.run_rollout(source_task, config)

    root = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is None
    )
    assert len(root["steps"]) == 3
    assert env_type.instances[0]._task.max_steps == 3
    assert env_type.instances[0].step_calls == 2
    assert result["misc"]["rollout_timed_out"] is False
    # Applying a rollout-local override should not mutate reusable task data.
    assert source_task.max_steps is None


@pytest.mark.asyncio
async def test_openreward_root_cap_preserves_recursive_child_budget(monkeypatch, tmp_path) -> None:
    agent_type, env_type = _counting_rollout_types(child_max_steps=2)
    rollout = _load_rollout_module(
        monkeypatch,
        agent_type=agent_type,
        env_type=env_type,
    )
    config = RolloutConfig(
        output_dir=str(tmp_path),
        max_steps=4,
        timeout=5,
        step_timeout=1,
        return_dict=True,
        extra={
            "openreward": {
                "enable_recursive_subagents": True,
            }
        },
    )

    result = await rollout.run_rollout(
        Task(id="recursive-root", goal="delegate then continue"),
        config,
    )

    root = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is None
    )
    child = next(
        trajectory
        for trajectory in result["trajectories"].values()
        if trajectory["parent_info"] is not None
    )
    assert root["task"]["max_steps"] == 4
    assert len(root["steps"]) == 4
    assert child["task"]["max_steps"] == 2
    assert len(child["steps"]) == 2
    assert result["misc"]["rollout_timed_out"] is False
