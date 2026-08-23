from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

import platoon.agents.actions.subagent as subagent_actions
from platoon.agents.actions.subagent import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY,
    SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
    SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY,
    SubagentRewardJudgeConfig,
    _combine_outcome_and_behavior_judgments,
    _normalize_behavior_judgment,
    _normalize_judgment,
    _record_judgment_reward,
    _run_behavior_judge,
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
class StaticBehaviorJudge:
    verdict: Any = None
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def judge(self, *, goal: str, trajectory: Trajectory) -> dict[str, Any]:
        self.calls.append((goal, trajectory.id))
        if self.error is not None:
            raise self.error
        return self.verdict


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
    assert SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY not in child.misc
    assert SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY not in child.misc
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
@pytest.mark.parametrize(
    ("verdict", "expected_status", "expected_gate", "expected_score", "eligible"),
    [
        (
            {
                "status": "pass",
                "passed": True,
                "reason": "The child directly created and checked the report.",
                "evidence": ["write report.txt", "read report.txt"],
            },
            "verified",
            1.0,
            0.8,
            True,
        ),
        (
            {
                "status": "fail",
                "passed": False,
                "reason": "The child only forwarded the task to a sibling.",
                "violations": ["credit assignment"],
            },
            "behavior_rejected",
            0.0,
            0.0,
            True,
        ),
        (
            {
                "status": "insufficient_evidence",
                "passed": None,
                "reason": "The retained history is not sufficient to judge behavior.",
            },
            "behavior_judge_invalid",
            0.0,
            0.0,
            False,
        ),
        (
            {
                "status": "pass",
                "passed": False,
                "reason": "Contradictory response.",
            },
            "behavior_judge_invalid",
            0.0,
            0.0,
            False,
        ),
    ],
)
async def test_behavior_judgment_gates_outcome_reward(
    verdict: dict[str, Any],
    expected_status: str,
    expected_gate: float,
    expected_score: float,
    eligible: bool,
):
    state = JudgeRunState()
    behavior_judge = StaticBehaviorJudge(verdict=verdict)
    collection = TrajectoryCollection()
    tokens = [
        current_trajectory_collection.set(collection),
        budget_tracker.set(DepthAwareStepBudgetTracker()),
        subagent_reward_judge_config.set(SubagentRewardJudgeConfig(max_steps=3, behavior_judge=behavior_judge)),
    ]
    verifier_message = json.dumps(
        {
            "status": "verified",
            "score": 0.8,
            "summary": "report.txt matched the request",
            "evidence": ["read report.txt"],
        }
    )

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
    outcome = child.misc[SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY]
    behavior = child.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY]
    combined = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert behavior_judge.calls == [("Create report.txt and summarize it.", child.id)]
    assert outcome["status"] == "verified"
    assert outcome["score"] == 0.8
    assert combined["verifier_trajectory_id"] == outcome["verifier_trajectory_id"]
    assert combined["outcome_judgment"] == outcome
    assert combined["behavior_judgment"] == behavior
    assert combined["outcome_status"] == "verified"
    assert combined["outcome_score"] == 0.8
    assert combined["behavior_gate"] == expected_gate
    assert combined["status"] == expected_status
    assert combined["score"] == expected_score
    assert combined[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is eligible
    assert child.reward == expected_score
    assert (EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in child.misc) is eligible
    assert child.steps[-1]["misc"]["reward_misc"] == {
        "reward/success": expected_score,
        "reward/subagent_judgment": expected_score,
        "reward/subagent_outcome_judgment": 0.8,
        "reward/subagent_behavior_gate": expected_gate,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "verifier_message",
        "expected_outcome_status",
        "expected_behavior_status",
        "eligible",
    ),
    [
        (
            json.dumps(
                {
                    "status": "failed",
                    "score": 0.0,
                    "summary": "The requested result was not present.",
                }
            ),
            "failed",
            "not_run_zero_outcome",
            True,
        ),
        (
            "not a structured outcome verdict",
            "unparseable",
            "not_run_ineligible_outcome",
            False,
        ),
    ],
)
async def test_behavior_judge_skips_zero_and_ineligible_outcomes(
    verifier_message: str,
    expected_outcome_status: str,
    expected_behavior_status: str,
    eligible: bool,
):
    state = JudgeRunState()
    behavior_judge = StaticBehaviorJudge(
        verdict={"status": "pass", "passed": True, "reason": "Should not run."}
    )
    collection = TrajectoryCollection()
    tokens = [
        current_trajectory_collection.set(collection),
        budget_tracker.set(DepthAwareStepBudgetTracker()),
        subagent_reward_judge_config.set(
            SubagentRewardJudgeConfig(max_steps=3, behavior_judge=behavior_judge)
        ),
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
    outcome = child.misc[SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY]
    behavior = child.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY]
    effective = child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]
    assert behavior_judge.calls == []
    assert outcome["status"] == expected_outcome_status
    assert effective == outcome
    assert effective["score"] == 0.0
    assert effective[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is eligible
    assert behavior["status"] == expected_behavior_status
    assert behavior["judged"] is False
    assert child.reward == 0.0
    assert (EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in child.misc) is eligible
    assert child.steps[-1]["misc"]["reward_misc"] == {
        "reward/success": 0.0,
        "reward/subagent_judgment": 0.0,
        "reward/subagent_outcome_judgment": 0.0,
    }


def test_behavior_judgment_requires_strict_binary_schema_and_preserves_details():
    valid_fail = _normalize_behavior_judgment(
        {
            "status": "FAIL",
            "passed": False,
            "reason": "  It repeatedly issued invalid calls.  ",
            "violations": ["invalid tool loop"],
        }
    )
    inconsistent = _normalize_behavior_judgment(
        {"status": "pass", "passed": 1, "reason": "Integer is not a JSON boolean."}
    )
    missing_passed = _normalize_behavior_judgment(
        {"status": "insufficient_evidence", "reason": "Missing explicit null."}
    )

    assert valid_fail == {
        "status": "fail",
        "passed": False,
        "reason": "It repeatedly issued invalid calls.",
        "violations": ["invalid tool loop"],
        "gate": 0.0,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: True,
    }
    assert inconsistent["status"] == "unparseable"
    assert inconsistent["gate"] == 0.0
    assert inconsistent[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False
    assert missing_passed["status"] == "unparseable"
    assert missing_passed[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False


def test_valid_behavior_fail_is_a_trainable_zero_gate():
    outcome = {
        "status": "verified",
        "score": 1.0,
        "verifier_trajectory_id": "verifier",
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: True,
    }
    behavior = _normalize_behavior_judgment({"status": "fail", "passed": False, "reason": "Forwarded all work."})

    combined = _combine_outcome_and_behavior_judgments(outcome, behavior)

    assert combined["status"] == "behavior_rejected"
    assert combined["score"] == 0.0
    assert combined["verifier_trajectory_id"] == "verifier"
    assert combined[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is True


@pytest.mark.asyncio
async def test_behavior_judge_exception_is_ineligible():
    judge = StaticBehaviorJudge(error=RuntimeError("model endpoint unavailable"))

    judgment = await _run_behavior_judge(
        judge,
        goal="delegated goal",
        trajectory=Trajectory(id="child"),
    )

    assert judgment["status"] == "judge_error"
    assert judgment["passed"] is None
    assert judgment["gate"] == 0.0
    assert judgment["error_type"] == "RuntimeError"
    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False


@pytest.mark.asyncio
async def test_cancelling_behavior_judgment_after_positive_outcome_cleans_up_task(monkeypatch):
    outcome_completed = asyncio.Event()
    behavior_started = asyncio.Event()
    behavior_cancelled = asyncio.Event()

    async def completed_outcome_judge(**kwargs: Any) -> Trajectory:
        _ = kwargs
        outcome_completed.set()
        return Trajectory(
            id="verifier",
            finish_message=json.dumps(
                {
                    "status": "verified",
                    "score": 1.0,
                    "summary": "Outcome verified.",
                }
            ),
        )

    @dataclass
    class BlockedBehaviorJudge:
        async def judge(self, *, goal: str, trajectory: Trajectory) -> dict[str, Any]:
            _ = goal, trajectory
            assert outcome_completed.is_set()
            behavior_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                behavior_cancelled.set()
                raise

    monkeypatch.setattr(
        subagent_actions,
        "_run_subagent_trajectory",
        completed_outcome_judge,
    )
    collection = TrajectoryCollection()
    child = Trajectory(
        id="child",
        task=Task(goal="child goal", id="child-task", max_steps=3),
        steps=[{"misc": {}}],
    )
    collection.trajectories[child.id] = child
    tokens = [
        current_trajectory_collection.set(collection),
        subagent_reward_judge_config.set(
            SubagentRewardJudgeConfig(
                max_steps=3,
                behavior_judge=BlockedBehaviorJudge(),
            )
        ),
    ]

    try:
        judging = asyncio.create_task(subagent_actions._maybe_judge_subagent(goal="child goal", traj=child))
        await asyncio.wait_for(behavior_started.wait(), timeout=1)
        judging.cancel()
        with pytest.raises(asyncio.CancelledError):
            await judging
    finally:
        for token in reversed(tokens):
            token.var.reset(token)

    assert outcome_completed.is_set()
    assert behavior_cancelled.is_set()


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


@pytest.mark.parametrize(
    ("status", "score"),
    [
        ("failed", 1.0),
        ("insufficient_evidence", 0.8),
        ("verified", 0.0),
        ("partial", 1.0),
        ("partial", float("nan")),
        ("verified", True),
    ],
)
def test_contradictory_outcome_verdict_is_fail_closed_and_ineligible(
    status,
    score,
):
    raw_message = json.dumps({"status": status, "score": score})
    verifier = Trajectory(
        id="verifier",
        finish_message=raw_message,
    )
    child = Trajectory(id="child", steps=[{"misc": {}}])

    judgment = _normalize_judgment(raw_message, verifier)
    child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    _record_judgment_reward(child, judgment)

    assert judgment["score"] == 0.0
    assert judgment["schema_error"] == (
        "Outcome verifier status and score are inconsistent."
    )
    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False
    assert child.reward == 0.0
    assert child.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True


def test_parseable_verdict_without_finish_is_not_trainable():
    raw_message = json.dumps({"status": "verified", "score": 1.0})
    verifier = Trajectory(id="verifier", finish_message=None, error_message="timed out")
    child = Trajectory(id="child", steps=[{"misc": {}}])

    judgment = _normalize_judgment(raw_message, verifier)
    child.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    _record_judgment_reward(child, judgment)

    assert judgment[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] is False
    assert child.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True
