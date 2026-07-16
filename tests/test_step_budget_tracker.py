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
from platoon.episode.trajectory import TrajectoryCollection

# -----------------------------
# Test doubles for Agent and Env
# -----------------------------


@dataclass
class MockAgent(ForkableAgent):
    actions: list[Any]
    _idx: int = 0
    # Optional plan for successive forks; each fork consumes one entry as that fork's actions
    fork_plan: Optional[list[list[Any]]] = None

    async def act(self, obs: Observation) -> Any:
        if self._idx < len(self.actions):
            action = self.actions[self._idx]
            self._idx += 1
            return action
        # Default to NOP once actions are exhausted
        return {"type": "NOP"}

    async def reset(self) -> None:
        self._idx = 0

    async def close(self) -> None:
        return None

    async def fork(self, task: Task) -> "MockAgent":
        # If a fork plan is provided, pop one sequence for this child; deeper forks get no special actions
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
    # If True, each step records remaining budgets into the step payload
    record_remaining: bool = False
    record_label: Optional[str] = None

    async def reset(self) -> Observation:
        # Register task on the current trajectory
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.set_trajectory_task(traj.id, self._task)
        return Observation(task=self._task, finished=self.finish_on_reset)

    async def step(self, action: Any) -> Observation:
        # Optionally spawn a subagent when instructed
        if isinstance(action, dict) and action.get("type") == "SUBAGENT":
            child_steps = int(action.get("child_steps", 1))
            # Reserve, run, and release occurs inside launch_subagent
            msg = await launch_subagent(goal="child", max_steps=child_steps)
            # Record message for validation in tests when requested
            payload_msg = action.get("capture_message", False)
            if payload_msg:
                # Stash the message for the current trajectory as a step
                traj_collection = current_trajectory_collection.get()
                traj = current_trajectory.get()
                rem_at_capture = int(budget_tracker.get().remaining_budget())
                traj_collection.add_trajectory_step(
                    traj.id, {"subagent_message": msg, "remaining_at_capture": rem_at_capture}
                )

        # Prepare step payload, optionally embedding budget snapshots
        payload = {"action": action}
        if self.record_remaining:
            payload.update(
                {
                    "rem_nr": budget_tracker.get().remaining_budget(),
                    "rem_r": budget_tracker.get().remaining_budget(),
                    "label": self.record_label,
                }
            )

        # Record a step for the current (parent) trajectory
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        traj_collection.add_trajectory_step(traj.id, payload)

        # Never mark finished here; the episode halts via budget exhaustion
        return Observation(task=self._task, finished=False)

    async def close(self) -> None:
        return None

    async def observe(self) -> Observation:
        # Minimal implementation; not used by run_episode
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


# -----------------------------
# Helper utilities
# -----------------------------


def compute_recursive_used_steps(collection: TrajectoryCollection, root_traj_id: str) -> int:
    used = 0
    stack = [root_traj_id]
    seen = set([root_traj_id])
    while stack:
        curr = stack.pop()
        used += len(collection.trajectories[curr].steps)
        for child_id, child in collection.trajectories.items():
            p = child.parent_info
            if p is not None and p.id == curr and child_id not in seen:
                seen.add(child_id)
                stack.append(child_id)
    return used


# -----------------------------
# Tests
# -----------------------------


@pytest.mark.asyncio
async def test_budget_caps_episode_steps():
    max_steps = 3
    env = MockEnv(_task=Task(goal="root", max_steps=max_steps))
    agent = MockAgent(actions=[{"type": "NOP"}] * 10)

    traj = await run_episode(agent, env)

    # Exactly capped by the budget
    assert len(traj.steps) == max_steps

    # Remaining budget should be zero, never negative
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_parent_budget_caps_with_subagent():
    parent_max = 7
    child_max = 4
    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    # First action triggers a subagent; subsequent actions are NOPs
    agent = MockAgent(actions=[{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 20)

    traj = await run_episode(agent, env)

    # Sum used steps across the whole tree and ensure we don't exceed budget
    collection = current_trajectory_collection.get()
    total_used = compute_recursive_used_steps(collection, traj.id)
    assert total_used == parent_max

    # Remaining budget should be zero, never negative
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_reserve_and_release_adjusts_remaining():
    # Make a trajectory that halts immediately so we can test reservation math
    env = MockEnv(_task=Task(goal="root", max_steps=5), finish_on_reset=True)
    agent = MockAgent(actions=[])
    await run_episode(agent, env)

    # Reserve within budget
    assert budget_tracker.get().reserve_budget(3) is True
    # Remaining subtracts reserved
    assert budget_tracker.get().remaining_budget() == 2

    # Releasing more than reserved must not make reserved negative
    budget_tracker.get().release_budget(5)
    assert budget_tracker.get().remaining_budget() == 5


@pytest.mark.asyncio
async def test_subagent_reservation_failure_no_budget_consumption():
    # Parent only has 2 steps. Try to launch a child that requests 10 steps.
    parent_max = 2
    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    agent = MockAgent(actions=[{"type": "SUBAGENT", "child_steps": 10}])

    traj = await run_episode(agent, env)

    # Parent should have taken exactly 2 steps due to budget cap
    assert len(traj.steps) == parent_max
    # Remaining budget never negative
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_nested_subagents_budget_caps():
    parent_max = 9
    child1_max = 5
    child2_max = 4

    # Parent triggers a child, which triggers a grandchild once
    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [{"type": "SUBAGENT", "child_steps": child1_max}] + [{"type": "NOP"}] * 50
    agent = MockAgent(actions=parent_actions, fork_plan=[[{"type": "SUBAGENT", "child_steps": child2_max}]])

    traj = await run_episode(agent, env)

    collection = current_trajectory_collection.get()
    total_used = compute_recursive_used_steps(collection, traj.id)
    assert total_used == parent_max
    assert budget_tracker.get().remaining_budget() == 0

    # Verify child and grandchild each constrained by their own caps
    # Find direct child of parent
    child_ids = [tid for tid, t in collection.trajectories.items() if t.parent_info and t.parent_info.id == traj.id]
    assert len(child_ids) == 1
    # Child's own steps plus its descendants should equal child1_max
    child_traj = collection.trajectories[child_ids[0]]
    grandchild_ids = [
        tid for tid, t in collection.trajectories.items() if t.parent_info and t.parent_info.id == child_ids[0]
    ]
    assert len(grandchild_ids) == 1
    grandchild_traj = collection.trajectories[grandchild_ids[0]]
    assert len(child_traj.steps) + len(grandchild_traj.steps) == child1_max
    # Find grandchild of parent (child of child)
    assert len(grandchild_traj.steps) == child2_max


@pytest.mark.asyncio
async def test_two_children_parent_budget_filled():
    parent_max = 9
    child1_max = 4
    child2_max = 3

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [
        {"type": "SUBAGENT", "child_steps": child1_max},
        {"type": "SUBAGENT", "child_steps": child2_max},
    ] + [{"type": "NOP"}] * 50
    agent = MockAgent(actions=parent_actions, fork_plan=[[], []])

    traj = await run_episode(agent, env)

    collection = current_trajectory_collection.get()
    total_used = compute_recursive_used_steps(collection, traj.id)
    assert total_used == parent_max
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_oversubscribed_children_blocked_by_reservation():
    # Two children can collectively exceed the parent's budget, but remaining must never be negative
    parent_max = 6
    child1_max = 4
    child2_max = 4

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [
        {"type": "SUBAGENT", "child_steps": child1_max},
        {"type": "SUBAGENT", "child_steps": child2_max},
    ]
    agent = MockAgent(actions=parent_actions, fork_plan=[[], []])

    traj = await run_episode(agent, env)

    collection = current_trajectory_collection.get()
    # With reservation-aware remaining, second child will fail to reserve; total used equals parent budget.
    total_used = compute_recursive_used_steps(collection, traj.id)
    assert total_used == parent_max
    assert budget_tracker.get().remaining_budget() == 0


@pytest.mark.asyncio
async def test_subagent_return_omits_budget_metadata_simple():
    parent_max = 6
    child_max = 4

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    # Capture the subagent message in a step for assertion
    agent = MockAgent(
        actions=[{"type": "SUBAGENT", "child_steps": child_max, "capture_message": True}] + [{"type": "NOP"}] * 10
    )

    traj = await run_episode(agent, env)

    # Find the captured message
    msg_steps = [s for s in traj.steps if isinstance(s, dict) and "subagent_message" in s]
    assert len(msg_steps) == 1
    msg_step = msg_steps[0]
    msg: str = msg_step["subagent_message"]
    assert msg == "Subagent did not finish before its step budget was exhausted."
    assert "Budget used by subagent" not in msg
    assert "Total remaining budget" not in msg
    assert "WARNING: Exhausted budget" not in msg
    assert "Traceback" not in msg


@pytest.mark.asyncio
async def test_subagent_return_omits_budget_metadata_nested():
    parent_max = 10
    child_max = 6
    grandchild_max = 4

    env = MockEnv(_task=Task(goal="root", max_steps=parent_max))
    parent_actions = [{"type": "SUBAGENT", "child_steps": child_max, "capture_message": True}] + [{"type": "NOP"}] * 20
    agent = MockAgent(actions=parent_actions, fork_plan=[[{"type": "SUBAGENT", "child_steps": grandchild_max}]])

    traj = await run_episode(agent, env)

    # Find the captured message
    msg_steps = [s for s in traj.steps if isinstance(s, dict) and "subagent_message" in s]
    assert len(msg_steps) == 1
    msg_step = msg_steps[0]
    msg: str = msg_step["subagent_message"]

    assert msg == "Subagent did not finish before its step budget was exhausted."
    assert "Budget used by subagent" not in msg
    assert "Total remaining budget" not in msg
    assert "WARNING: Exhausted budget" not in msg
    assert "Traceback" not in msg


@pytest.mark.asyncio
async def test_exhausted_budget_sets_warning_message():
    max_steps = 3
    env = MockEnv(_task=Task(goal="root", max_steps=max_steps))
    agent = MockAgent(actions=[{"type": "NOP"}] * 20)

    traj = await run_episode(agent, env)

    assert traj.error_message is not None
    assert "WARNING: Exhausted budget" in traj.error_message


@pytest.mark.asyncio
async def test_child_remaining_snapshots_non_negative():
    parent_max = 7
    child_max = 4
    env = MockEnv(_task=Task(goal="root", max_steps=parent_max), record_remaining=True, record_label="parent")
    agent = MockAgent(actions=[{"type": "SUBAGENT", "child_steps": child_max}] + [{"type": "NOP"}] * 10)

    traj = await run_episode(agent, env)

    # Find child trajectory and ensure recorded remaining budgets never negative
    collection = current_trajectory_collection.get()
    child_ids = [tid for tid, t in collection.trajectories.items() if t.parent_info and t.parent_info.id == traj.id]
    assert len(child_ids) == 1
    child_traj = collection.trajectories[child_ids[0]]
    assert len(child_traj.steps) == child_max
    for step in child_traj.steps:
        # Steps are dicts with rem_nr/rem_r from MockEnv when record_remaining=True
        assert step.get("rem_nr", 0) >= 0
        assert step.get("rem_r", 0) >= 0
