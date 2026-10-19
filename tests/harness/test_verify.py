"""Verifier verdicts and judge protocols (platoon.harness.verify), ported from the legacy judging tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from platoon.harness.agent import AgentResult
from platoon.harness.verify import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY,
    SUBAGENT_REWARD_JUDGMENT_MISC_KEY,
    apply_subagent_parent_usage_gates,
    behavior_judge_error,
    combine_outcome_and_behavior_judgments,
    error_message,
    normalize_behavior_judgment,
    normalize_judgment,
    normalize_parent_usage_judgment,
    record_judgment,
    return_message,
    verifier_judgment,
)
from platoon.task import Task
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.utils.trajectory_status import TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY


def verdict(status: str, score: Any = None, **extra: Any) -> str:
    body: dict[str, Any] = {"status": status, **extra}
    if score is not None:
        body["score"] = score
    return json.dumps(body)


def judge(message: str, *, finished: bool = True, forced_zero: bool = False) -> dict[str, Any]:
    return normalize_judgment(
        message,
        verifier_trajectory_id="verifier-1",
        verifier_error_message=None,
        verifier_finished=finished,
        verifier_forced_zero=forced_zero,
    )


@pytest.mark.parametrize(
    ("message", "score", "eligible"),
    [
        (verdict("verified", 1.0), 1.0, True),
        (verdict("partial", 0.5), 0.5, True),
        (verdict("failed", 0.0), 0.0, True),
        (verdict("insufficient_evidence"), 0.0, True),
        ("```json\n" + verdict("verified", 1.0) + "\n```", 1.0, True),
        ("I checked it. " + verdict("verified", 0.8) + " Done.", 0.8, True),
        (verdict("verified", 0.0), 0.0, False),  # contradictory: fail closed
        (verdict("failed", 1.0), 0.0, False),
        (verdict("verified", "high"), 0.0, False),
        (verdict("verified", True), 0.0, False),
        (verdict("great", 1.0), 0.0, False),
        ("not json at all", 0.0, False),
    ],
)
def test_outcome_verdicts_normalize_fail_closed(message, score, eligible):
    judgment = judge(message)
    assert judgment["score"] == score and judgment["training_eligible"] is eligible
    assert judgment["verifier_trajectory_id"] == "verifier-1"


def test_a_parseable_verdict_without_finish_or_with_a_forced_zero_is_not_trainable():
    assert judge(verdict("verified", 1.0), finished=False)["training_eligible"] is False
    assert judge(verdict("verified", 1.0), forced_zero=True)["training_eligible"] is False


def test_verifier_judgment_reads_the_agent_result():
    @dataclass
    class Verifier:
        trajectory_id: str = "v"
        misc: dict = field(default_factory=dict)

    done = verifier_judgment(Verifier(), AgentResult("completed", final_message=verdict("verified", 1.0)))
    assert done["training_eligible"] is True
    limited = verifier_judgment(Verifier(), AgentResult("limit"))
    assert limited["status"] == "unparseable" and limited["verifier_error_message"].startswith("WARNING")


def test_return_and_error_messages_keep_the_legacy_wording():
    assert return_message(AgentResult("completed", final_message="hi")) == "hi"
    assert return_message(AgentResult("limit")) == "Subagent did not finish before its step budget was exhausted."
    assert return_message(AgentResult("error", error="boom")) == "Subagent failed before finishing."
    assert error_message(AgentResult("completed")) is None


def test_behavior_judgments_need_the_strict_binary_schema_and_keep_details():
    passed = normalize_behavior_judgment({"status": "Pass", "passed": True, "reason": " fine ", "evidence": ["e"]})
    assert passed["gate"] == 1.0 and passed["training_eligible"] and passed["evidence"] == ["e"]
    failed = normalize_behavior_judgment({"status": "fail", "passed": False, "reason": "looped"})
    assert failed["gate"] == 0.0 and failed["training_eligible"]
    unsure = normalize_behavior_judgment({"status": "insufficient_evidence", "passed": None, "reason": "?"})
    assert unsure["gate"] == 0.0 and not unsure["training_eligible"]
    for bad in ({"status": "pass", "passed": False, "reason": "x"}, {"status": "pass", "reason": "x"}, "pass", {}):
        assert normalize_behavior_judgment(bad)["status"] == "unparseable"
    assert behavior_judge_error(RuntimeError("down"))["training_eligible"] is False


def test_behavior_verdicts_gate_the_outcome_score():
    outcome = judge(verdict("verified", 0.8))
    rejected = combine_outcome_and_behavior_judgments(
        outcome, normalize_behavior_judgment({"status": "fail", "passed": False, "reason": "looped"})
    )
    assert rejected["status"] == "behavior_rejected" and rejected["score"] == 0.0 and rejected["training_eligible"]
    accepted = combine_outcome_and_behavior_judgments(
        outcome, normalize_behavior_judgment({"status": "pass", "passed": True, "reason": "ok"})
    )
    assert accepted["score"] == 0.8 and accepted["outcome_score"] == 0.8 and accepted["behavior_gate"] == 1.0
    invalid = combine_outcome_and_behavior_judgments(outcome, behavior_judge_error(RuntimeError("x")))
    assert invalid["status"] == "behavior_judge_invalid" and not invalid["training_eligible"]


@dataclass
class Call:
    call_id: str


@dataclass
class JudgedAgent:
    calls: list[Call] = field(default_factory=lambda: [Call("c1"), Call("c2")])
    misc: dict[str, Any] = field(default_factory=dict)
    step_misc: dict[str, dict[str, Any]] = field(default_factory=dict)


def test_record_judgment_writes_the_trainer_keys_and_the_child_outcome():
    agent = JudgedAgent()
    outcome = record_judgment(agent, judge(verdict("partial", 0.5)))
    assert outcome.status == "scored" and outcome.reward == 0.5
    assert agent.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY]["status"] == "partial"
    assert EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in agent.misc
    assert agent.step_misc["c2"]["reward_misc"] == {"reward/success": 0.5, "reward/subagent_judgment": 0.5}

    ineligible = record_judgment(agent, judge("garbage"))
    assert ineligible.reward == 0.0 and agent.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True


def test_a_judgment_cannot_replace_an_explicit_terminal_zero():
    agent = JudgedAgent(misc={TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY: True})
    assert record_judgment(agent, judge(verdict("verified", 1.0))).reward == 0.0


def test_parent_usage_verdicts_normalize_fail_closed():
    assert normalize_parent_usage_judgment({"status": "used", "used": True, "reason": "r"})["gate"] == 1.0
    assert normalize_parent_usage_judgment({"status": "not_used", "used": False, "reason": "r"})["training_eligible"]
    assert normalize_parent_usage_judgment({"status": "used", "used": False, "reason": "r"})["status"] == "unparseable"


def _trajectory(trajectory_id: str, *, parent: str | None, score: float | None, goal: str) -> Trajectory:
    trajectory = Trajectory(
        id=trajectory_id,
        task=Task(goal=goal),
        parent_info=ParentInfo(id=parent, fork_step=0) if parent else None,
        steps=[TrajectoryStep(misc={})],
    )
    if score is not None:
        trajectory.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = {
            "status": "verified",
            "score": score,
            "training_eligible": True,
        }
    return trajectory


class UsageJudge:
    def __init__(self, used: dict[str, bool | None]) -> None:
        self.used = used
        self.calls: list[str] = []

    async def judge_parent_usage(self, *, parent_goal, parent_trajectory, child_goal, child_trajectory):
        self.calls.append(child_goal)
        used = self.used[child_goal]
        status = {True: "used", False: "not_used", None: "insufficient_evidence"}[used]
        return {"status": status, "used": used, "reason": "audited"}


def test_parent_usage_gates_suppress_only_decisively_unused_positive_children():
    trajectories = [
        _trajectory("root", parent=None, score=None, goal="root"),
        _trajectory("used", parent="root", score=1.0, goal="used"),
        _trajectory("unused", parent="root", score=1.0, goal="unused"),
        _trajectory("unsure", parent="root", score=1.0, goal="unsure"),
        _trajectory("zero", parent="root", score=0.0, goal="zero"),
    ]
    collection = TrajectoryCollection(id="r", trajectories={t.id: t for t in trajectories})
    usage = UsageJudge({"used": True, "unused": False, "unsure": None})
    asyncio.run(apply_subagent_parent_usage_gates(collection, usage))

    assert sorted(usage.calls) == ["unsure", "unused", "used"]  # the zero-score child is not audited
    misc = {t.id: t.misc for t in trajectories}
    assert misc["unused"][EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] is True
    assert EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in misc["used"]
    assert EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY not in misc["unsure"]
    assert misc["unused"][SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY]["status"] == "not_used"
    reward_misc = trajectories[2].steps[-1].misc["reward_misc"]
    assert reward_misc == {"reward/subagent_parent_usage_gate": 0.0, "reward/subagent_parent_usage_judgment_valid": 1.0}
    with pytest.raises(ValueError):
        asyncio.run(apply_subagent_parent_usage_gates(collection, usage, max_concurrency=0))
