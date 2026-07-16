from __future__ import annotations

import asyncio

import pytest

from platoon.envs.base import Observation, Task
from platoon.episode.context import budget_tracker, current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import StepBudgetTracker, TrajectoryCollection
from platoon.utils.trajectory_status import (
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_TIMED_OUT_MISC_KEY,
    trajectory_was_interrupted,
)


class _IdleAgent:
    def __init__(self) -> None:
        self.closed = False

    async def act(self, obs: Observation):
        raise AssertionError("reset never completes")

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


class _HangingResetEnv:
    def __init__(self) -> None:
        self._task = Task(id="timeout-test", goal="wait forever", max_steps=1)
        self.closed = False

    @property
    def task(self) -> Task:
        return self._task

    async def reset(self) -> Observation:
        collection = current_trajectory_collection.get()
        from platoon.episode.context import current_trajectory

        collection.set_trajectory_task(current_trajectory.get().id, self._task)
        await asyncio.Future()
        raise AssertionError("unreachable")

    async def step(self, action) -> Observation:
        raise AssertionError("reset never completes")

    async def observe(self) -> Observation:
        return Observation(task=self._task)

    async def close(self) -> None:
        self.closed = True


class _HangingActAgent(_IdleAgent):
    async def act(self, obs: Observation):
        await asyncio.Future()
        raise AssertionError("unreachable")


class _ReadyEnv(_HangingResetEnv):
    async def reset(self) -> Observation:
        collection = current_trajectory_collection.get()
        from platoon.episode.context import current_trajectory

        collection.set_trajectory_task(current_trajectory.get().id, self._task)
        return Observation(task=self._task, finished=False)


@pytest.mark.asyncio
async def test_run_episode_propagates_outer_timeout_after_cleanup() -> None:
    """A rollout deadline must not be swallowed by run_episode's finally block."""

    collection = TrajectoryCollection()
    collection_token = current_trajectory_collection.set(collection)
    budget_token = budget_tracker.set(StepBudgetTracker())
    agent = _IdleAgent()
    env = _HangingResetEnv()
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(run_episode(agent, env, timeout=60), timeout=0.02)
    finally:
        budget_tracker.reset(budget_token)
        current_trajectory_collection.reset(collection_token)

    assert agent.closed
    assert env.closed
    assert len(collection.trajectories) == 1
    trajectory = next(iter(collection.trajectories.values()))
    assert trajectory.error_message is not None
    assert "Episode cancelled" in trajectory.error_message
    assert trajectory.misc[TRAJECTORY_CANCELLED_MISC_KEY] is True


@pytest.mark.asyncio
async def test_run_episode_marks_internal_step_timeout_ineligible() -> None:
    collection = TrajectoryCollection()
    collection_token = current_trajectory_collection.set(collection)
    budget_token = budget_tracker.set(StepBudgetTracker())
    agent = _HangingActAgent()
    env = _ReadyEnv()
    try:
        trajectory = await run_episode(agent, env, timeout=0.01)
    finally:
        budget_tracker.reset(budget_token)
        current_trajectory_collection.reset(collection_token)

    assert trajectory.misc[TRAJECTORY_TIMED_OUT_MISC_KEY] is True
    assert trajectory.error_message is not None
    assert "Episode timed out" in trajectory.error_message
    assert trajectory_was_interrupted(trajectory)
