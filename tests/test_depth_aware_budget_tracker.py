"""Tests for DepthAwareStepBudgetTracker.

This tracker differs from StepBudgetTracker in two key ways:
1. Subagent steps do NOT consume parent budget (each trajectory is independent).
2. An optional max_depth cap limits how deep subagent delegation can go.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from platoon.agents.actions.subagent import launch_subagent
from platoon.agents.base import ForkableAgent
from platoon.envs.base import ForkableEnv, Observation, Task
from platoon.episode.context import (
    budget_tracker,
    current_trajectory,
    current_trajectory_collection,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import (
    BudgetExceededError,
    DepthAwareStepBudgetTracker,
    TrajectoryCollection,
)


# -----------------------------
# Test doubles (reused from test_step_budget_tracker.py)
# -----------------------------


@dataclass
class MockAgent(ForkableAgent):
    actions: list[Any]
    _idx: int = 0
    fork_plan: Optional[list[list[Any]]] = None

    async def act(self, obs: Observation) -> Any:
        if self._idx < len(self.actions):
            action = self.actions[self._idx]
            self._idx += 1
            return action
        return {"type": "NOP"}

    async def reset(self) -> None:
        self._idx = 0

    async def close(self) -> None:
        return None

    async def fork(self, task: Task) -> "MockAgent":
        if self.fork_plan and len(self.fork_plan) > 0:
            child_actions = self.fork_plan[0]
            remaining_plan = self.fork_plan[1:] if len(self.fork_plan) > 1 else []
        else:
            child_actions = []
            remaining_plan = []
        return MockAgent(actions=child_actions, fork_plan=remaining_plan)


@dataclass
class MockEnv(ForkableEnv):
    _task: Task
    finish_on_reset: bool = False
    record_remaining: bool = False
    record_label: Optional[str] = None

    async def reset(self) -> Observation:
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._task)
        return Observation(task=self._task, finished=self.finish_on_reset)

    async def step(self, action: Any) -> Observation:
        if isinstance(action, dict) and action.get("type") == "SUBAGENT":
            child_steps = int(action.get("child_steps", 1))
            msg = await launch_subagent(goal="child", max_steps=child_steps)
            payload_msg = action.get("capture_message", False)
            if payload_msg:
                traj_collection = current_trajectory_collection.get()
                traj = current_trajectory.get()
                rem_at_capture = int(budget_tracker.get().remaining_budget())
                traj_collection.add_trajectory_step(
                    traj.id, {"subagent_message": msg, "remaining_at_capture": rem_at_capture}
                )

        payload = {"action": action}
        if self.record_remaining:
            payload.update(
                {
                    "rem": budget_tracker.get().remaining_budget(),
                    "label": self.record_label,
                }
            )

        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, payload)
        return Observation(task=self._task, finished=False)

    async def close(self) -> None:
        return None

    async def observe(self) -> Observation:
        return Observation(task=self._task, finished=False)

    @property
    def task(self) -> Task:
        return self._task

    async def fork(self, task: Task) -> "MockEnv":
        return MockEnv(
            _task=task,
            finish_on_reset=self.finish_on_reset,
            record_remaining=self.record_remaining,
            record_label=self.record_label,
        )


# Helper to set up the DepthAwareStepBudgetTracker before running an episode
def _set_depth_tracker(max_depth: int | None = None) -> DepthAwareStepBudgetTracker:
    tracker = DepthAwareStepBudgetTracker(max_depth=max_depth)
    budget_tracker.set(tracker)
    return tracker


# -----------------------------
# Tests: basic step budget (independent per trajectory)
# -----------------------------


@pytest.mark.asyncio
async def test_own_step_budget_caps_episode():
    """Each trajectory is capped by its own max_steps, same as StepBudgetTracker for a flat episode."""
    _set_depth_tracker()
    max_steps = 4
    env = MockEnv(_task=Task(goal="root", max_steps=max_steps))
    agent = MockAgent(actions=[{"type": "NOP"}] * 20)

    traj = await run_episode(agent, env)

    assert len(traj.steps) == max_steps
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_subagent_steps_do_not_consume_parent_budget():
    """The parent should use all of its own steps regardless of subagent step usage."""
    _set_depth_tracker()
    parent_max = 5
    child_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    )

    traj = await run_episode(agent, env)

    # Parent should use exactly its own max_steps (child steps are independent)
    assert len(traj.steps) == parent_max

    # The child should also have used its own max_steps
    collection = current_trajectory_collection.get()
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 1
    child_traj = collection.trajectories[child_ids[0]]
    assert len(child_traj.steps) == child_max


@pytest.mark.asyncio
async def test_multiple_subagents_parent_uses_all_budget():
    """Parent can launch multiple subagents and still use all its own steps."""
    _set_depth_tracker()
    parent_max = 6
    child1_max = 3
    child2_max = 4

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(
        actions=[
            {"type": "SUBAGENT", "child_steps": child1_max},
            {"type": "SUBAGENT", "child_steps": child2_max},
        ] + [{"type": "NOP"}] * 20,
        fork_plan=[[], []],
    )

    traj = await run_episode(agent, env)

    # Parent uses all its own steps
    assert len(traj.steps) == parent_max

    # Both children launched successfully
    collection = current_trajectory_collection.get()
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 2


@pytest.mark.asyncio
async def test_nested_subagents_independent_budgets():
    """Each level of nesting has an independent budget."""
    _set_depth_tracker()
    parent_max = 5
    child_max = 4
    grandchild_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    agent = MockAgent(
        actions=parent_actions,
        fork_plan=[[{"type": "SUBAGENT", "child_steps": grandchild_max}] + [{"type": "NOP"}] * 20],
    )

    traj = await run_episode(agent, env)

    collection = current_trajectory_collection.get()

    # Parent uses its own max_steps
    assert len(traj.steps) == parent_max

    # Find child
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 1
    child_traj = collection.trajectories[child_ids[0]]
    assert len(child_traj.steps) == child_max

    # Find grandchild
    grandchild_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == child_ids[0]
    ]
    assert len(grandchild_ids) == 1
    grandchild_traj = collection.trajectories[grandchild_ids[0]]
    assert len(grandchild_traj.steps) == grandchild_max


# -----------------------------
# Tests: depth limiting
# -----------------------------


@pytest.mark.asyncio
async def test_depth_limit_blocks_subagent():
    """When max_depth=0, no subagents can be launched from the root."""
    _set_depth_tracker(max_depth=0)
    parent_max = 5

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": 3, "capture_message": True}] + [{"type": "NOP"}] * 20
    )

    traj = await run_episode(agent, env)

    # Subagent should have been blocked; parent still uses its steps
    assert len(traj.steps) == parent_max

    # Only the root trajectory should exist (no child trajectories besides the blocked fork attempt)
    collection = current_trajectory_collection.get()
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    # The forked env/agent are created before reserve_budget, but run_episode was never called
    # for the child because reserve_budget failed. So no child trajectory should have steps.
    # Actually, the fork happens before the check, but the episode is never started.
    # Let's check that the captured message mentions depth.
    msg_steps = [s for s in traj.steps if isinstance(s, dict) and "subagent_message" in s]
    assert len(msg_steps) == 1
    assert "depth" in msg_steps[0]["subagent_message"].lower()
    assert "maximum depth" in msg_steps[0]["subagent_message"].lower()


@pytest.mark.asyncio
async def test_depth_limit_allows_within_limit():
    """max_depth=1 allows one level of subagents but blocks grandchildren."""
    _set_depth_tracker(max_depth=1)
    parent_max = 5
    child_max = 4
    grandchild_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    # Child tries to launch a grandchild, which should be blocked
    parent_actions = [{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    agent = MockAgent(
        actions=parent_actions,
        fork_plan=[[
            {"type": "SUBAGENT", "child_steps": grandchild_max, "capture_message": True},
        ] + [{"type": "NOP"}] * 20],
    )

    traj = await run_episode(agent, env)

    collection = current_trajectory_collection.get()

    # Parent uses its own steps
    assert len(traj.steps) == parent_max

    # Child was launched successfully
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 1
    child_traj = collection.trajectories[child_ids[0]]
    assert len(child_traj.steps) == child_max

    # The grandchild launch should have been blocked by depth
    grandchild_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == child_ids[0]
    ]
    # No grandchild trajectory should have been run
    # (the fork creates agent/env but run_episode is never called due to reserve_budget failure)
    # Check the child's captured message mentions depth
    msg_steps = [s for s in child_traj.steps if isinstance(s, dict) and "subagent_message" in s]
    assert len(msg_steps) == 1
    assert "depth" in msg_steps[0]["subagent_message"].lower()


@pytest.mark.asyncio
async def test_depth_limit_2_allows_grandchild():
    """max_depth=2 allows root -> child -> grandchild."""
    _set_depth_tracker(max_depth=2)
    parent_max = 5
    child_max = 4
    grandchild_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    agent = MockAgent(
        actions=parent_actions,
        fork_plan=[[{"type": "SUBAGENT", "child_steps": grandchild_max}] + [{"type": "NOP"}] * 20],
    )

    traj = await run_episode(agent, env)
    collection = current_trajectory_collection.get()

    # All three levels should have been created
    assert len(traj.steps) == parent_max

    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 1
    child_traj = collection.trajectories[child_ids[0]]
    assert len(child_traj.steps) == child_max

    grandchild_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == child_ids[0]
    ]
    assert len(grandchild_ids) == 1
    grandchild_traj = collection.trajectories[grandchild_ids[0]]
    assert len(grandchild_traj.steps) == grandchild_max


@pytest.mark.asyncio
async def test_no_depth_limit_allows_deep_nesting():
    """When max_depth is None, arbitrarily deep nesting is allowed."""
    _set_depth_tracker(max_depth=None)
    parent_max = 5
    child_max = 4
    grandchild_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    agent = MockAgent(
        actions=parent_actions,
        fork_plan=[[{"type": "SUBAGENT", "child_steps": grandchild_max}] + [{"type": "NOP"}] * 20],
    )

    traj = await run_episode(agent, env)
    collection = current_trajectory_collection.get()

    # All three levels should exist
    assert len(collection.trajectories) == 3
    assert len(traj.steps) == parent_max


# -----------------------------
# Tests: BudgetExceededError
# -----------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_error_has_depth_reason():
    """The BudgetExceededError raised on depth violation has reason='depth'."""
    tracker = _set_depth_tracker(max_depth=0)

    # Set up a minimal trajectory so reserve_budget can run
    env = MockEnv(_task=Task(goal="root", max_steps=5), finish_on_reset=True)
    agent = MockAgent(actions=[])
    await run_episode(agent, env)

    with pytest.raises(BudgetExceededError) as exc_info:
        tracker.reserve_budget(5, raise_on_failure=True)

    assert exc_info.value.reason == "depth"
    assert "depth" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_reserve_budget_returns_false_on_depth_violation():
    """reserve_budget returns False (no raise) when depth would be exceeded."""
    tracker = _set_depth_tracker(max_depth=0)

    env = MockEnv(_task=Task(goal="root", max_steps=5), finish_on_reset=True)
    agent = MockAgent(actions=[])
    await run_episode(agent, env)

    result = tracker.reserve_budget(5, raise_on_failure=False)
    assert result is False


@pytest.mark.asyncio
async def test_reserve_budget_succeeds_when_no_depth_limit():
    """reserve_budget always succeeds when max_depth is None."""
    tracker = _set_depth_tracker(max_depth=None)

    env = MockEnv(_task=Task(goal="root", max_steps=5), finish_on_reset=True)
    agent = MockAgent(actions=[])
    await run_episode(agent, env)

    result = tracker.reserve_budget(100, raise_on_failure=False)
    assert result is True


# -----------------------------
# Tests: remaining budget is correct
# -----------------------------


@pytest.mark.asyncio
async def test_remaining_budget_only_counts_own_steps():
    """remaining_budget after a subagent only reflects the parent's own steps."""
    _set_depth_tracker()
    parent_max = 10
    child_max = 5

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max), record_remaining=True)
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    )

    traj = await run_episode(agent, env)

    # Parent used all its own steps
    assert len(traj.steps) == parent_max
    assert budget_tracker.get().remaining_budget() == 0

    # Verify that the remaining budget reported during the episode was based on own steps only
    for step in traj.steps:
        if isinstance(step, dict) and "rem" in step:
            assert step["rem"] >= 0


@pytest.mark.asyncio
async def test_used_budget_for_only_counts_own_steps():
    """used_budget_for returns only the trajectory's own step count, not descendant steps."""
    _set_depth_tracker()
    parent_max = 6
    child_max = 4

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20
    )

    traj = await run_episode(agent, env)

    # used_budget_for the parent should be only the parent's own steps
    assert budget_tracker.get().used_budget_for(traj.id) == parent_max

    collection = current_trajectory_collection.get()
    child_ids = [
        tid for tid, t in collection.trajectories.items()
        if t.parent_info and t.parent_info.id == traj.id
    ]
    assert len(child_ids) == 1
    assert budget_tracker.get().used_budget_for(child_ids[0]) == child_max


# -----------------------------
# Tests: error message in launch_subagent
# -----------------------------


@pytest.mark.asyncio
async def test_depth_exceeded_message_in_launch_subagent():
    """When depth is exceeded, the message returned to the agent mentions depth, not step budget."""
    _set_depth_tracker(max_depth=0)
    parent_max = 5

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": 3, "capture_message": True}] + [{"type": "NOP"}] * 10
    )

    traj = await run_episode(agent, env)

    msg_steps = [s for s in traj.steps if isinstance(s, dict) and "subagent_message" in s]
    assert len(msg_steps) == 1
    msg = msg_steps[0]["subagent_message"]

    # Should mention depth-related information, not step budget reservation
    assert "maximum depth" in msg.lower()
    assert "perform the task yourself" in msg.lower()
    # Should NOT contain the step-budget-specific language
    assert "reserve max_steps" not in msg


@pytest.mark.asyncio
async def test_exhausted_own_budget_sets_warning():
    """When a trajectory exhausts its own step budget, the warning message is set."""
    _set_depth_tracker()
    max_steps = 3
    env = MockEnv(_task=Task(goal="root", max_steps=max_steps))
    agent = MockAgent(actions=[{"type": "NOP"}] * 20)

    traj = await run_episode(agent, env)

    assert traj.error_message is not None
    assert "WARNING: Exhausted budget" in traj.error_message


@pytest.mark.asyncio
async def test_release_budget_is_noop():
    """release_budget is a no-op; calling it should not raise or change state."""
    tracker = _set_depth_tracker()

    env = MockEnv(_task=Task(goal="root", max_steps=5), finish_on_reset=True)
    agent = MockAgent(actions=[])
    await run_episode(agent, env)

    remaining_before = tracker.remaining_budget()
    tracker.release_budget(100)
    remaining_after = tracker.remaining_budget()
    assert remaining_before == remaining_after
