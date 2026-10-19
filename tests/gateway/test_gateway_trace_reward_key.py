"""A scored outcome's record names the reward key it trained (``misc["outcome"]["reward_key"]``): the name its grader
reported the reward under (``Outcome.primary``), null for a score platoon gave itself, which no grader reported; an
outcome that is not scored names none.  ``trained_reward_key`` reads it back."""

from __future__ import annotations

from platoon.envs.harbor.grading import RewardContract
from platoon.envs.service import PLATOON_ZERO_KEY, Outcome
from platoon.gateway.core import AgentSession
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import Task
from platoon.utils.trajectory_status import trained_reward_key


def _agent(agent_id: str, trajectory_id: str, parent: str | None = None) -> AgentSession:
    return AgentSession(
        agent_id=agent_id,
        rollout_id="r",
        trajectory_id=trajectory_id,
        secret="k",
        interactions={},
        parent_agent_id=parent,
    )


def _root(outcome: Outcome):
    """The root trajectory, and the record a rollout keeps of it (``TrajectoryCollection.to_dict``)."""

    collection = build_trajectory_collection([_agent("a0", "root")], task=Task(id="t"), outcome=outcome)
    return collection.trajectories["root"], collection.to_dict()["trajectories"]["root"]


def test_a_scored_outcome_names_the_reward_key_it_trained():
    root, recorded = _root(
        Outcome(status="scored", rewards={"correctness": 1.0, "submission": 1.0}, primary="correctness")
    )

    assert root.misc["outcome"] == {
        "status": "scored",
        "rewards": {"correctness": 1.0, "submission": 1.0},
        "reason": None,
        "reward_key": "correctness",
    }
    assert trained_reward_key(root) == trained_reward_key(recorded) == "correctness"  # also as a rollout records it


def test_platoons_own_zero_trained_no_reward_key():
    zero = RewardContract.coerce(["reward", "correctness"]).zero("workdir_missing", {"workdir": "/app"})
    assert zero.metadata == {"workdir": "/app", PLATOON_ZERO_KEY: True}

    root, _ = _root(zero)

    assert root.misc["outcome"]["reward_key"] is None and root.reward == 0.0
    assert trained_reward_key(root) is None


def test_an_outcome_that_is_not_scored_names_no_reward_key():
    root, _ = _root(Outcome.infra("verifier_ambiguous_primary", status="invalid"))

    assert "reward_key" not in root.misc["outcome"]
    assert trained_reward_key(root) is None
    assert trained_reward_key({"misc": {"outcome": {"status": "scored", "rewards": {"reward": 1.0}}}}) is None  # older


def test_a_childs_own_outcome_names_its_own_key():
    root, child = _agent("a0", "root"), _agent("a1", "child", parent="a0")
    child.outcome = Outcome(status="scored", rewards={"score": 0.5}, primary="score")

    collection = build_trajectory_collection(
        [root, child], task=Task(id="t"), outcome=Outcome(status="scored", rewards={"reward": 1.0})
    )

    assert trained_reward_key(collection.trajectories["root"]) == "reward"
    assert trained_reward_key(collection.trajectories["child"]) == "score"
