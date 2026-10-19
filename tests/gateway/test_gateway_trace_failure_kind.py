"""An untrainable outcome's failure kind reaches the trajectory with its status, so reports can count failures by
kind (``harness_crash``, ``env_busy``, ...) instead of parsing reason text; scored outcomes record what they did."""

from __future__ import annotations

from platoon.envs.service import Outcome
from platoon.gateway.core import AgentSession
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import Task


def _agent(agent_id: str, trajectory_id: str, parent: str | None = None) -> AgentSession:
    return AgentSession(
        agent_id=agent_id,
        rollout_id="r",
        trajectory_id=trajectory_id,
        secret="k",
        interactions={},
        parent_agent_id=parent,
    )


def test_an_infrastructure_outcome_records_its_failure_kind():
    outcome = Outcome.infra("harness_crash", reason="custom@1 exited with code 2")

    (root,) = build_trajectory_collection(
        [_agent("a0", "root")], task=Task(id="t"), outcome=outcome
    ).trajectories.values()

    assert root.misc["outcome"] == {
        "status": "error",
        "rewards": {},
        "reason": "custom@1 exited with code 2",
        "failure_kind": "harness_crash",
    }


def test_a_childs_own_invalid_outcome_records_its_failure_kind():
    root, child = _agent("a0", "root"), _agent("a1", "child", parent="a0")
    child.outcome = Outcome.infra("grader_invalid", status="invalid", reason="the env declared it invalid")

    collection = build_trajectory_collection(
        [root, child], task=Task(id="t"), outcome=Outcome(status="scored", rewards={"reward": 1.0})
    )

    assert collection.trajectories["child"].misc["outcome"]["failure_kind"] == "grader_invalid"
    assert "failure_kind" not in collection.trajectories["root"].misc["outcome"]


def test_a_scored_outcome_records_no_failure_kind():
    outcome = Outcome(status="scored", rewards={"reward": 1.0})

    (root,) = build_trajectory_collection(
        [_agent("a0", "root")], task=Task(id="t"), outcome=outcome
    ).trajectories.values()

    assert root.misc["outcome"] == {
        "status": "scored",
        "rewards": {"reward": 1.0},
        "reason": None,
        "reward_key": "reward",
    }
