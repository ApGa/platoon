"""The judge seam, on a scripted policy as the judge: a producer the entry names is given the rollout's judge as model
access, opened once a rollout and only when a producer could use it, and a role trains on the rows the judge's answers
became.  Platoon records each call as the judge-role agent's, which never trains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from platoon_harness.testing import AskModel, Call, Say
from teams_toys import JUDGING, RATING, WITNESSED, WITNESSING, episode, named, prove, scripted, team, unlocking

from platoon.testing import last_user_message
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection

GOAL = "Open the box. Its code is 1."
RECRUITS = (("w1", "worker"), ("s1", "scout"))
WORKER = scripted([AskModel(), Call("agent_finish", {"gist": "unlocked"})])
SPAWNING = [Call("agent_spawn", {"name": name, "child_role": role, "goal": GOAL}) for name, role in RECRUITS]
# A lead that spawns a worker, w1, and a scout, s1, which work alike.
TREE = named(
    team(
        lead={"harness": scripted(SPAWNING, [Say("done")]), "may_spawn": ["worker", "scout"]},
        worker={"harness": WORKER},
        scout={"harness": WORKER},
    )
)


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def judged(prompt: str) -> str:
    """The lockbox's model, which, asked as the judge to rate an agent, rates it 0.75."""

    return "0.75" if last_user_message(prompt).startswith("Rate @") else unlocking(prompt)


def judges(opened: list[Any]) -> list[Any]:
    """The judge-role agents among the platoon agents a gateway opened."""

    return [agent for agent in opened if agent.role == "judge"]


def test_a_judge_producer_writes_local_rewards_a_role_trains_on(proven, gateway_agents):
    """The judge rates each agent that started, and the worker trains on its rating; platoon keeps each rating's call
    as the judge's, under the root's ``judge_calls``, which no row trains."""

    run = episode(TREE, judged, harness_options={"rewards": {"worker": {"producer": JUDGING}}})

    ours = {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}
    (judge,) = judges(gateway_agents)
    rows = get_train_data_for_trajectory_collection(run.collection.to_dict(), run.interactions, "test:1", 0)
    trained = {call for calls in rows.datum_calls for call, _, _ in calls}
    audited = run.root.misc["judge_calls"]
    assert (ours["lead"].reward, ours["w1"].reward, ours["s1"].reward) == (1.0, 0.75, 1.0)
    assert [(call["agent_id"], call["role"]) for call in audited] == [(judge.agent_id, "judge")] * 3
    assert {call["call_id"] for call in audited} == set(judge.call_ids) and not trained & set(judge.call_ids)
    assert len(run.trajectories) == 3  # the judge has none


@pytest.mark.parametrize(
    ("rewards", "rollout", "opened"),
    [
        pytest.param({"worker": {"producer": RATING}, "scout": {"producer": WITNESSING}}, None, 1, id="two-producers"),
        pytest.param({}, None, 0, id="no-producer"),
        pytest.param({"scout": {"producer": WITNESSING}}, {"judges": False}, 0, id="judges-off"),
    ],
)
def test_the_judge_endpoint_is_opened_once_and_only_when_a_producer_could_use_it(
    proven, gateway_agents, rewards, rollout, opened
):
    WITNESSED.clear()

    episode(TREE, harness_options={"rewards": rewards}, rollout=rollout)

    assert len(judges(gateway_agents)) == opened
    assert [seen["judge"] is not None for seen in WITNESSED] == ([bool(opened)] if "scout" in rewards else [])
