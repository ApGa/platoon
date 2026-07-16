from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory, current_trajectory_collection
from platoon.utils.trajectory_status import TRAJECTORY_CANCELLED_MISC_KEY

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENREWARD_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward"


def _module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_rollout_module(monkeypatch):
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
    _module(monkeypatch, "platoon.openhands.agent", OpenHandsAgent=_Agent)
    _module(
        monkeypatch,
        "platoon.openhands.recursive",
        PROGRAMMATIC_TOOL_CALLING_SYSTEM_PROMPT_SUFFIX="",
        RECURSIVE_SUBAGENT_INITIAL_TASK_SUFFIX="",
        RECURSIVE_SUBAGENT_SYSTEM_PROMPT_SUFFIX="",
        RECURSIVE_SUBAGENT_USER_MESSAGE_SUFFIX="",
        append_system_message_suffix=_identity,
        append_user_message_suffix=_identity,
        with_programmatic_tool_calling=_identity,
        with_task_tracker_tool=_identity,
    )
    _module(monkeypatch, "platoon.openreward.env", OpenRewardOpenHandsEnv=_HangingEnv)

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
