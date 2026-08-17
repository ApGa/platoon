from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from platoon.agents.actions.subagent import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
    SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY,
    SubagentRewardJudgeConfig,
    _normalize_judgment,
    _record_judgment_reward,
    launch_subagent,
)
from platoon.agents.base import ForkableAgent
from platoon.envs.base import ForkableEnv, Observation, Task
from platoon.episode.context import (
    budget_tracker,
    current_trajectory,
    current_trajectory_collection,
    finish_message,
    subagent_reward_judge_config,
)
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import DepthAwareStepBudgetTracker, Trajectory, TrajectoryCollection


@dataclass
class JudgeRunState:
    parent_messages: list[str] = field(default_factory=list)


@dataclass
class FinishEventRecorder:
    finished: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    def on_trajectory_created(self, trajectory: Trajectory) -> None:
        pass

    def on_trajectory_step_added(self, trajectory: Trajectory, step: Any) -> None:
        pass

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:
        pass

    def on_trajectory_finished(self, trajectory: Trajectory) -> None:
        self.finished.append((trajectory.id, trajectory.reward, dict(trajectory.misc)))


@dataclass
class DeterministicJudgeAgent(ForkableAgent):
    async def act(self, obs: Observation) -> Any:
        assert obs.task is not None
        if obs.task.goal == "root":
            return {"type": "launch_child"}
        if obs.task.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
            return {"type": "verify"}
        return {"type": "child_work"}

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def fork(self, task: Task) -> "DeterministicJudgeAgent":
        return DeterministicJudgeAgent()


@dataclass
class DeterministicJudgeEnv(ForkableEnv):
    _task: Task
    state: JudgeRunState
    verifier_message: str

    async def reset(self) -> Observation:
        current_trajectory_collection.get().set_trajectory_task(
            current_trajectory.get().id,
            self._task,
        )
        return Observation(task=self._task, finished=False)

    async def step(self, action: Any) -> Observation:
        traj_collection = current_trajectory_collection.get()
        traj = current_trajectory.get()
        if self._task.goal == "root":
            child_message = await launch_subagent(
                goal="Create report.txt and summarize it.",
                max_steps=3,
            )
            self.state.parent_messages.append(child_message)
            traj_collection.add_trajectory_step(
                traj.id,
                {"action": action, "child_message": child_message},
            )
            finish_message.set("root done")
            return Observation(task=self._task, finished=True)

        if self._task.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
            traj_collection.add_trajectory_step(
                traj.id,
                {"action": action, "verifier_goal": self._task.goal},
            )
            finish_message.set(self.verifier_message)
            return Observation(task=self._task, finished=True)

        traj_collection.add_trajectory_step(traj.id, {"action": action})
        finish_message.set("Created report.txt with the requested summary.")
        return Observation(task=self._task, finished=True)

    async def close(self) -> None:
        return None

    async def observe(self) -> Observation:
        return Observation(task=self._task, finished=False)

    @property
    def task(self) -> Task:
        return self._task

    async def fork(self, task: Task) -> "DeterministicJudgeEnv":
        return DeterministicJudgeEnv(
            _task=task,
            state=self.state,
            verifier_message=self.verifier_message,
        )


@dataclass
class NestedVerifierEnv(ForkableEnv):
    _task: Task
    helper_messages: list[str]
    blocked_grandhelper_messages: list[str]

    async def reset(self) -> Observation:
        current_trajectory_collection.get().set_trajectory_task(
            current_trajectory.get().id,
            self._task,
        )
        return Observation(task=self._task, finished=False)

    async def step(self, action: Any) -> Observation:
        _ = action
        if self._task.goal == "root":
            await launch_subagent(goal="solver child", max_steps=3)
            finish_message.set("root done")
            return Observation(task=self._task, finished=True)

        if not self._task.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
            finish_message.set("solver completed")
            return Observation(task=self._task, finished=True)

        if self._task.misc.get("verifier_helper"):
            blocked_message = await launch_subagent(
                goal="nested verifier helper",
                max_steps=1,
            )
            self.blocked_grandhelper_messages.append(blocked_message)
            finish_message.set("inspected environment evidence")
            return Observation(task=self._task, finished=True)

        helper_message = await launch_subagent(
            goal="inspect verifier evidence",
            max_steps=2,
            # Even explicit child metadata must not escape a verifier tree.
            task_misc={
                "verifier_helper": True,
                SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY: False,
                SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY: "incorrect",
            },
        )
        self.helper_messages.append(helper_message)
        finish_message.set(
            json.dumps(
                {
                    "status": "verified",
                    "score": 1.0,
                    "summary": "helper inspected the environment",
                    "evidence": [helper_message],
                }
            )
        )
        return Observation(task=self._task, finished=True)

    async def close(self) -> None:
        return None

    async def observe(self) -> Observation:
        return Observation(task=self._task, finished=False)

    @property
    def task(self) -> Task:
        return self._task

    async def fork(self, task: Task) -> "NestedVerifierEnv":
        return NestedVerifierEnv(
            task,
            self.helper_messages,
            self.blocked_grandhelper_messages,
        )


def _trajectory_by_goal(collection: TrajectoryCollection, goal: str) -> Trajectory:
    for trajectory in collection.trajectories.values():
        if trajectory.task is not None and trajectory.task.goal == goal:
            return trajectory
    raise AssertionError(f"Missing trajectory for goal {goal!r}")


@pytest.mark.asyncio
async def test_subagent_judging_records_verifier_result():
    state = JudgeRunState()
    verifier_message = json.dumps(
        {
            "status": "verified",
            "score": 0.8,
            "summary": "report.txt was present and matched the request",
            "passed_claims": ["report.txt exists"],
            "failed_claims": [],
            "evidence": ["read report.txt"],
        }
    )
    collection = TrajectoryCollection()
    recorder = FinishEventRecorder()
    collection.register_event_handlers(recorder)
    tokens = [
        current_trajectory_collection.set(collection),
        budget_tracker.set(DepthAwareStepBudgetTracker()),
        subagent_reward_judge_config.set(SubagentRewardJudgeConfig(max_steps=3)),
    ]

    try:
        await run_episode(
            DeterministicJudgeAgent(),
            DeterministicJudgeEnv(
                _task=Task(goal="root", id="root", max_steps=10),
                state=state,
                verifier_message=verifier_message,
            ),
        )
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    child = _trajectory_by_goal(collection, "Create report.txt and summarize it.")
    assert state.parent_messages == ["Created report.txt with the requested summary."]
    judgment = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert judgment["status"] == "verified"
    assert judgment["score"] == 0.8
    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is True
    assert judgment["passed_claims"] == ["report.txt exists"]
    assert EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in child.misc
    assert child.reward == 0.8
    assert child.steps[-1]["misc"]["reward_misc"] == {
        "reward/success": 0.8,
        "reward/subagent_judgment": 0.8,
    }
    child_finish_events = [(reward, misc) for traj_id, reward, misc in recorder.finished if traj_id == child.id]
    assert child_finish_events[0] == (0.0, {})
    assert any(misc.get(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY) is True for _, misc in child_finish_events[1:-1])
    pending_events = [
        (reward, misc)
        for reward, misc in child_finish_events[1:-1]
        if misc.get(SUBAGENT_REWARD_JUDGMENT_MISC_KEY, {}).get("status") == "pending"
    ]
    assert pending_events
    assert all(reward == 0.0 for reward, _ in pending_events)
    assert child_finish_events[-1][0] == 0.8
    assert child_finish_events[-1][1][SUBAGENT_REWARD_JUDGMENT_MISC_KEY]["score"] == 0.8

    verifier = collection.trajectories[judgment["verifier_trajectory_id"]]
    assert verifier.parent_info is not None
    assert verifier.parent_info.id == child.id
    assert verifier.misc[SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] == child.id
    assert verifier.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] is True
    assert verifier.task is not None
    assert verifier.task.misc[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] is True
    assert child.task is not None
    assert verifier.task.parent_tasks[-1].id == child.task.id
    assert SUBAGENT_REWARD_JUDGMENT_MISC_KEY not in verifier.misc
    verifier_finish_events = [(reward, misc) for traj_id, reward, misc in recorder.finished if traj_id == verifier.id]
    assert verifier_finish_events[-1][1][SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] == child.id
    assert verifier_finish_events[-1][1][EXCLUDE_FROM_TRAINING_MISC_KEY] is True


@pytest.mark.asyncio
async def test_verifier_descendant_stays_in_verifier_tree_and_is_not_reverified():
    collection = TrajectoryCollection()
    helper_messages: list[str] = []
    blocked_grandhelper_messages: list[str] = []
    tokens = [
        current_trajectory_collection.set(collection),
        # Policy recursion permits only root -> solver. The verifier root and
        # its one helper are synthetic depth exceptions.
        budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=1)),
        subagent_reward_judge_config.set(SubagentRewardJudgeConfig(max_steps=4)),
    ]

    try:
        await run_episode(
            DeterministicJudgeAgent(),
            NestedVerifierEnv(
                _task=Task(goal="root", id="root", max_steps=20),
                helper_messages=helper_messages,
                blocked_grandhelper_messages=blocked_grandhelper_messages,
            ),
        )
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    assert helper_messages == ["inspected environment evidence"]
    assert len(blocked_grandhelper_messages) == 1
    assert "one helper level" in blocked_grandhelper_messages[0]
    assert len(collection.trajectories) == 4

    solver = _trajectory_by_goal(collection, "solver child")
    verifier = next(
        trajectory
        for trajectory in collection.trajectories.values()
        if trajectory.task is not None
        and trajectory.task.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY)
        and not trajectory.task.misc.get("verifier_helper")
    )
    helper = _trajectory_by_goal(collection, "inspect verifier evidence")
    assert helper.task is not None
    assert helper.task.misc[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] is True
    assert SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY not in helper.task.misc
    assert helper.parent_info is not None
    assert helper.parent_info.id == verifier.id
    assert SUBAGENT_REWARD_JUDGMENT_MISC_KEY not in helper.misc
    assert solver.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]["verifier_trajectory_id"] == verifier.id


@pytest.mark.asyncio
async def test_subagent_judging_records_unparseable_verdict():
    state = JudgeRunState()
    collection = TrajectoryCollection()
    tokens = [
        current_trajectory_collection.set(collection),
        budget_tracker.set(DepthAwareStepBudgetTracker()),
        subagent_reward_judge_config.set(SubagentRewardJudgeConfig(max_steps=3)),
    ]

    try:
        await run_episode(
            DeterministicJudgeAgent(),
            DeterministicJudgeEnv(
                _task=Task(goal="root", id="root", max_steps=10),
                state=state,
                verifier_message="looks good to me",
            ),
        )
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    child = _trajectory_by_goal(collection, "Create report.txt and summarize it.")
    judgment = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert judgment["status"] == "unparseable"
    assert judgment["score"] == 0.0
    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False
    assert judgment["summary"] == "looks good to me"
    assert child.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True
    assert child.reward == 0.0
    assert child.steps[-1]["misc"]["reward_misc"] == {
        "reward/success": 0.0,
        "reward/subagent_judgment": 0.0,
    }


def test_valid_failed_judgment_remains_trainable_at_budget_boundary():
    raw_message = json.dumps(
        {
            "status": "failed",
            "score": 0.0,
            "summary": "The requested artifact was not present.",
        }
    )
    verifier = Trajectory(
        id="verifier",
        finish_message=raw_message,
        # The loop may set this warning on the same step that records finish.
        # A real parsed verdict must still be usable in that case.
        error_message="WARNING: Exhausted budget when running episode.",
    )
    child = Trajectory(id="child", steps=[{"misc": {}}])

    judgment = _normalize_judgment(raw_message, verifier)
    child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    _record_judgment_reward(child, judgment)

    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is True
    assert child.reward == 0.0
    assert EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in child.misc


def test_parseable_verdict_without_finish_is_not_trainable():
    raw_message = json.dumps({"status": "verified", "score": 1.0})
    verifier = Trajectory(id="verifier", finish_message=None, error_message="timed out")
    child = Trajectory(id="child", steps=[{"misc": {}}])

    judgment = _normalize_judgment(raw_message, verifier)
    child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    _record_judgment_reward(child, judgment)

    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False
    assert child.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True
