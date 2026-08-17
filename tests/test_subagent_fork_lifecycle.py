from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import pytest

from platoon.agents.actions.subagent import launch_subagent
from platoon.envs.base import Observation, Task
from platoon.episode.context import (
    budget_tracker,
    current_agent,
    current_env,
    current_trajectory,
    current_trajectory_collection,
    episode_step_timeout,
)
from platoon.episode.trajectory import StepBudgetTracker, TrajectoryCollection


@dataclass
class _LifecycleState:
    events: list[str] = field(default_factory=list)
    agent_fork_error: BaseException | None = None
    env_fork_error: BaseException | None = None
    child_agent_close_error: BaseException | None = None
    block_env_fork: bool = False
    env_fork_started: asyncio.Event = field(default_factory=asyncio.Event)
    child_agent: _TrackingAgent | None = None
    child_env: _TrackingEnv | None = None


class _RecordingStepBudgetTracker(StepBudgetTracker):
    def __init__(self, state: _LifecycleState) -> None:
        super().__init__()
        self._state = state

    def reserve_budget(
        self,
        requested_budget: float,
        raise_on_failure: bool = False,
        *,
        child_depth_scope="policy",
    ) -> bool:
        self._state.events.append("budget.reserve")
        return super().reserve_budget(
            requested_budget,
            raise_on_failure,
            child_depth_scope=child_depth_scope,
        )

    def release_budget(self, amount_to_release: float) -> None:
        self._state.events.append("budget.release")
        super().release_budget(amount_to_release)


class _TrackingAgent:
    def __init__(self, state: _LifecycleState, *, child: bool = False) -> None:
        self._state = state
        self._child = child
        self.close_calls = 0

    async def act(self, obs: Observation):
        raise AssertionError("finished-on-reset child must not act")

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        self.close_calls += 1
        self._state.events.append("child_agent.close" if self._child else "parent_agent.close")
        if self._child and self._state.child_agent_close_error is not None:
            raise self._state.child_agent_close_error

    async def fork(self, task: Task) -> _TrackingAgent:
        self._state.events.append("agent.fork")
        if self._state.agent_fork_error is not None:
            raise self._state.agent_fork_error
        child = type(self)(self._state, child=True)
        self._state.child_agent = child
        return child


class _TrackingEnv:
    def __init__(self, task: Task, state: _LifecycleState, *, child: bool = False) -> None:
        self._task = task
        self._state = state
        self._child = child
        self.close_calls = 0

    @property
    def task(self) -> Task:
        return self._task

    async def reset(self) -> Observation:
        current_trajectory_collection.get().set_trajectory_task(current_trajectory.get().id, self._task)
        return Observation(task=self._task, finished=True)

    async def step(self, action) -> Observation:
        raise AssertionError("finished-on-reset child must not step")

    async def observe(self) -> Observation:
        return Observation(task=self._task)

    async def close(self) -> None:
        self.close_calls += 1
        self._state.events.append("child_env.close" if self._child else "parent_env.close")

    async def fork(self, task: Task) -> _TrackingEnv:
        self._state.events.append("env.fork")
        self._state.env_fork_started.set()
        if self._state.env_fork_error is not None:
            raise self._state.env_fork_error
        if self._state.block_env_fork:
            await asyncio.Future()
        child = type(self)(task, self._state, child=True)
        self._state.child_env = child
        return child


@contextmanager
def _launch_context(
    state: _LifecycleState,
    *,
    parent_max_steps: int = 100,
) -> Iterator[_RecordingStepBudgetTracker]:
    collection = TrajectoryCollection()
    root = collection.create_trajectory()
    root_task = Task(id="root", goal="root", max_steps=parent_max_steps)
    collection.set_trajectory_task(root.id, root_task)
    agent = _TrackingAgent(state)
    env = _TrackingEnv(root_task, state)
    tracker = _RecordingStepBudgetTracker(state)
    tokens = [
        current_trajectory_collection.set(collection),
        current_trajectory.set(root),
        current_agent.set(agent),
        current_env.set(env),
        budget_tracker.set(tracker),
        episode_step_timeout.set(1),
    ]
    try:
        yield tracker
    finally:
        for token in reversed(tokens):
            token.var.reset(token)


@pytest.mark.asyncio
async def test_budget_admission_precedes_resource_forks_and_denial_allocates_nothing() -> None:
    state = _LifecycleState()
    with _launch_context(state, parent_max_steps=2) as tracker:
        message = await launch_subagent(goal="too large", max_steps=10)

        assert message.startswith("Not enough budget to launch subagent for goal too large.")
        assert state.events == ["budget.reserve"]
        assert state.child_agent is None
        assert state.child_env is None
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0


@pytest.mark.asyncio
async def test_env_fork_failure_closes_returned_agent_and_releases_budget() -> None:
    state = _LifecycleState(
        env_fork_error=RuntimeError("environment fork failed"),
        child_agent_close_error=RuntimeError("agent close also failed"),
    )
    with _launch_context(state) as tracker:
        with pytest.raises(RuntimeError, match="environment fork failed"):
            await launch_subagent(goal="child", max_steps=10)

        assert state.child_agent is not None
        assert state.child_agent.close_calls == 1
        assert state.child_env is None
        assert state.events == [
            "budget.reserve",
            "agent.fork",
            "env.fork",
            "budget.release",
            "child_agent.close",
        ]
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0


@pytest.mark.asyncio
async def test_agent_fork_failure_releases_budget_without_forking_environment() -> None:
    state = _LifecycleState(agent_fork_error=RuntimeError("agent fork failed"))
    with _launch_context(state) as tracker:
        with pytest.raises(RuntimeError, match="agent fork failed"):
            await launch_subagent(goal="child", max_steps=10)

        assert state.events == ["budget.reserve", "agent.fork", "budget.release"]
        assert state.child_agent is None
        assert state.child_env is None
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0


@pytest.mark.asyncio
async def test_cancellation_during_env_fork_closes_agent_and_releases_budget() -> None:
    state = _LifecycleState(block_env_fork=True)
    with _launch_context(state) as tracker:
        launch_task = asyncio.create_task(launch_subagent(goal="child", max_steps=10))
        await asyncio.wait_for(state.env_fork_started.wait(), timeout=1)
        launch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch_task

        assert state.child_agent is not None
        assert state.child_agent.close_calls == 1
        assert state.child_env is None
        assert state.events == [
            "budget.reserve",
            "agent.fork",
            "env.fork",
            "budget.release",
            "child_agent.close",
        ]
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0


@pytest.mark.asyncio
async def test_success_hands_ownership_to_episode_without_double_close() -> None:
    state = _LifecycleState()
    with _launch_context(state) as tracker:
        message = await launch_subagent(goal="child", max_steps=10)

        assert message == ""
        assert state.child_agent is not None
        assert state.child_env is not None
        assert state.child_agent.close_calls == 1
        assert state.child_env.close_calls == 1
        assert state.events == [
            "budget.reserve",
            "agent.fork",
            "env.fork",
            "child_agent.close",
            "child_env.close",
            "budget.release",
        ]
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0


@pytest.mark.asyncio
async def test_cancellation_before_episode_handoff_closes_both_returned_forks(monkeypatch) -> None:
    state = _LifecycleState()
    real_create_task = asyncio.create_task

    async def cancelled_before_start() -> None:
        raise asyncio.CancelledError

    def create_cancelled_task(coroutine):
        coroutine.close()
        return real_create_task(cancelled_before_start())

    with _launch_context(state) as tracker:
        monkeypatch.setattr(asyncio, "create_task", create_cancelled_task)
        with pytest.raises(asyncio.CancelledError):
            await launch_subagent(goal="child", max_steps=10)

        assert state.child_agent is not None
        assert state.child_env is not None
        assert state.child_agent.close_calls == 1
        assert state.child_env.close_calls == 1
        assert state.events == [
            "budget.reserve",
            "agent.fork",
            "env.fork",
            "budget.release",
            "child_agent.close",
            "child_env.close",
        ]
        assert tracker.reserved_trajectory_budgets[current_trajectory.get().id] == 0
