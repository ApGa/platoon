"""The ``platoon-harness`` program's finalizer, on rollouts of the lockbox through platoon's gateway: each trajectory
joined to its agent, each step to a call its agent reported, compactions as ``aux``, eligibility as markers, a
producer's row as a role's reward, ``misc["team"]`` and the root's policy; and the rows that then train, as platoon's
converter makes them."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

import pytest
from platoon_harness import Session, Team
from platoon_harness.testing import AskModel, Call, Compact, Say
from teams_toys import RATING, WITNESSING, episode, named, prove, scripted, session_name, team, unlocking, wayward

import platoon as pt
from platoon.harness.builtin.platoon_harness import bridge
from platoon.harness.builtin.platoon_harness.finalize import EPISODE, Mismatch, finalize
from platoon.rewards import delegation_capped
from platoon.testing import ScriptedPolicy, make_gateway
from platoon.train.workers.run import DEFAULT_REWARD_PROCESSOR
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_INVALID_MISC_KEY,
    TRAJECTORY_TIMED_OUT_MISC_KEY,
)

GOAL = "Open the box. Its code is 1."


def tree(worker: str, *, lead: str | None = None, goal: str = GOAL) -> str:
    """A team whose lead spawns one worker, ``w1``, told ``goal``, and says it is done once the worker reports."""

    spawn = Call("agent_spawn", {"name": "w1", "child_role": "worker", "goal": goal})
    lead = lead or scripted([spawn], [Say("the worker has it")])
    return named(team(lead={"harness": lead, "may_spawn": ["worker"]}, worker={"harness": worker}))


FINISHING = [AskModel(), Call("agent_finish", {"gist": "unlocked"})]
TREE = tree(scripted(FINISHING))


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


@pytest.fixture
def given(monkeypatch: pytest.MonkeyPatch) -> list[pt.TrajectoryCollection]:
    """Each rollout's collection as the program gave the finalizer it, before it marked anything."""

    collections: list[pt.TrajectoryCollection] = []

    async def keeping(collection: pt.TrajectoryCollection, name: str, producers: Any) -> Any:
        collections.append(copy.deepcopy(collection))
        return await finalize(collection, name, producers)

    monkeypatch.setattr(bridge, "finalize", keeping)
    return collections


def finalized_again(run: Any, collection: pt.TrajectoryCollection, producers: Any = None) -> tuple[Mismatch, ...]:
    """The finalizer, run again on ``collection`` against the rollout's ended session."""

    return asyncio.run(finalize(collection, session_name(run), producers or {}))


def reported(run: Any) -> dict[str, list[str | None]]:
    """The ids of the calls each agent reported, in order."""

    async def calls() -> dict[str, list[str | None]]:
        async with Session.attach(session_name(run)) as ended:
            found = {}
            for row in await ended.agents():
                answer = await ended.eligibility(f"@{row['agent']}", EPISODE)
                found[row["agent"]] = [call.call_id for call in answer.calls]
            return found

    return asyncio.run(calls())


def converted(run: Any, collection: pt.TrajectoryCollection | None = None, **settings: Any) -> Any:
    """Platoon's converter on a rollout's collection: its rows, each with the calls it merged."""

    collection = collection or run.collection
    return get_train_data_for_trajectory_collection(collection.to_dict(), run.interactions, "test:1", 0, **settings)


def trained(run: Any) -> set[str]:
    """The calls whose tokens train in a rollout's rows."""

    return {call for calls in converted(run).datum_calls for call, _, _ in calls}


def completions(trajectory: pt.Trajectory) -> list[str]:
    return [step.misc["action_misc"]["completion_id"] for step in trajectory.steps]


def excluded(trajectory: pt.Trajectory) -> list[bool]:
    return [bool(step.misc.get(EXCLUDE_FROM_TRAINING_MISC_KEY)) for step in trajectory.steps]


def by_agent(run: Any) -> dict[str | None, pt.Trajectory]:
    return {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}


def test_every_reported_call_is_a_step_and_every_step_a_reported_call(proven, given):
    run = episode(TREE)

    assert finalized_again(run, given[0]) == ()
    assert not any(any(excluded(trajectory)) for trajectory in run.trajectories)
    assert trained(run) == {call for calls in reported(run).values() for call in calls}


def test_every_compaction_is_aux_joined_and_outside_the_step_budget(proven, given):
    run = episode(named(team(solver={"harness": scripted([AskModel(), Compact()])})), max_steps=2)

    assert [step.misc["action_misc"].get("kind", "policy") for step in run.root.steps] == ["policy", "policy", "aux"]
    assert run.root.misc["budget"]["stops"] == 0  # two policy calls, within its two steps: the compaction was not one
    assert finalized_again(run, given[0]) == () and excluded(run.root) == [False, False, False]
    assert completions(run.root)[-1] in trained(run)


def test_a_compaction_sent_without_the_headers_is_counted_and_named_a_mismatch(proven, given):
    run = episode(named(team(solver={"harness": wayward([AskModel(), Compact()], headers=False)})), max_steps=3)

    kinds = [step.misc["action_misc"].get("kind", "policy") for step in run.root.steps]
    (mismatch,) = finalized_again(run, given[0])
    assert kinds == ["policy", "policy", "policy"]  # the gateway counted it against the step budget
    assert mismatch == Mismatch("solver", (), (), (completions(run.root)[-1],), ())
    assert completions(run.root)[-1] in trained(run)  # a compaction the harness reported trains all the same


def test_an_aux_step_no_reported_compaction_names_is_excluded(proven, given):
    def as_its_turns(call: Any) -> Any:
        return call.model_copy(update={"purpose": None})

    run = episode(named(team(solver={"harness": wayward([AskModel(), Compact()], rewrite=as_its_turns)})))

    (mismatch,) = finalized_again(run, given[0])
    assert run.root.steps[-1].misc["action_misc"]["kind"] == "aux"
    assert excluded(run.root) == [False, False, True]
    assert mismatch == Mismatch("solver", (), (completions(run.root)[-1],), (), ())
    assert completions(run.root)[-1] not in trained(run)


def test_a_step_no_reported_call_names_is_excluded(proven, given):
    hidden: list[str] = []

    def hiding_the_first(call: Any) -> Any:
        if hidden:
            return call
        hidden.append(call.call_id)
        return None

    run = episode(named(team(solver={"harness": wayward(rewrite=hiding_the_first)})))

    (mismatch,) = finalized_again(run, given[0])
    assert excluded(run.root) == [True, False]
    assert mismatch == Mismatch("solver", (), tuple(hidden), (), ())
    assert trained(run) == {completions(run.root)[1]}


@pytest.mark.parametrize("barred", ["first", "last"])
def test_a_call_reported_twice_never_trains(proven, given, barred):
    """A harness that reports each call twice, once for a purpose the record bars and once for its turn, leaves which
    report holds unknown, whichever comes first: no step of its trains, and the mismatch names each call."""

    def twice(call: Any) -> list[Any]:
        titled = call.model_copy(update={"purpose": "title"})
        return [titled, call] if barred == "first" else [call, titled]

    run = episode(named(team(solver={"harness": wayward(rewrite=twice)})))

    (mismatch,) = finalized_again(run, given[0])
    assert excluded(run.root) == [True, True]
    assert mismatch == Mismatch("solver", (), (), (), tuple(completions(run.root)))
    assert trained(run) == set()


def test_an_ineligible_call_trains_no_tokens(proven):
    titled: list[str] = []

    def titling_the_first(call: Any) -> Any:
        if titled:
            return call
        titled.append(call.call_id)
        return call.model_copy(update={"purpose": "title"})

    run = episode(named(team(solver={"harness": wayward(rewrite=titling_the_first)})))

    assert excluded(run.root) == [True, False]
    assert trained(run) == {completions(run.root)[1]}
    assert run.root.misc.get(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY) is None


def test_an_ineligible_child_keeps_its_reward_and_trains_no_rows(proven):
    run = episode(tree(wayward(FINISHING, forks=True)), agents={"child_grading": "env"})

    worker = by_agent(run)["w1"]
    rows = converted(run, tag_datums=True)
    assert worker.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] and worker.reward == 1.0
    assert worker.id not in rows.datum_trajectories
    assert [stats.reward for stats in rows.trajectory_stats if stats.trajectory_id == worker.id] == [1.0]


def test_an_ineligible_root_keeps_the_task_reward_and_trains_no_rows(proven):
    run = episode(named(team(solver={"harness": wayward(forks=True)})))

    rows = converted(run)
    assert excluded(run.root) == [True, True]
    assert (rows.task_reward, rows.task_reward_valid, rows.datums) == (1.0, True, [])


def test_a_non_root_trajectory_with_no_turn_of_ours_is_excluded_whole(proven):
    run = episode(tree(scripted(FINISHING), goal=""))  # spawned with no goal, and never woken

    worker = by_agent(run)["w1"]
    assert worker.steps == [] and worker.misc[EXCLUDE_FROM_TRAINING_MISC_KEY]
    assert run.root.misc.get(EXCLUDE_FROM_TRAINING_MISC_KEY) is None


@pytest.mark.parametrize("processor", [DEFAULT_REWARD_PROCESSOR, delegation_capped], ids=["default", "delegation"])
def test_a_role_trains_on_the_row_its_entry_names(proven, processor):
    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": RATING}}})

    worker = by_agent(run)["w1"]
    stats = converted(run, reward_processor=processor).trajectory_stats
    scored = {trajectory.trajectory_id: trajectory.reward for trajectory in stats}
    assert scored == {run.root.id: 1.0, worker.id: 0.25}


@pytest.mark.parametrize(
    ("status", "marker"),
    [
        ("invalid", TRAJECTORY_INVALID_MISC_KEY),
        ("error", TRAJECTORY_INVALID_MISC_KEY),
        ("timeout", TRAJECTORY_TIMED_OUT_MISC_KEY),
        ("cancelled", TRAJECTORY_CANCELLED_MISC_KEY),
    ],
)
def test_a_non_scored_row_becomes_its_marker(proven, status, marker):
    rewards = {"worker": {"producer": RATING, "status": status}}
    run = episode(TREE, harness_options={"rewards": rewards})

    worker = by_agent(run)["w1"]
    assert worker.misc[marker] and worker.misc["team"]["row"] is not None
    assert worker.id not in converted(run, tag_datums=True).datum_trajectories


def test_a_missing_row_becomes_an_invalid_marker(proven):
    """A producer that writes no row for an agent leaves its role's agent with nothing to train on: it is marked
    ``trajectory_invalid``, and its ``misc["team"]`` names no row beside the marker and no grade, where a
    row the producer wrote invalid is named, and none for its one span."""

    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": WITNESSING}}})

    worker = by_agent(run)["w1"]
    assert worker.misc[TRAJECTORY_INVALID_MISC_KEY]
    session = run.root.misc["team"]["session"]
    assert worker.misc["team"] == {"session": session, "agent": "w1", "row": None, "spans": [None]}


def test_the_finalizer_never_removes_a_platoon_marker(proven, given):
    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": RATING}}})
    collection = given[0]
    worker = list(collection.trajectories.values())[1]
    worker.misc[TRAJECTORY_CANCELLED_MISC_KEY] = True  # as platoon marks a child cut off while it is graded

    finalized_again(run, collection, {"worker": RATING})

    assert worker.misc[TRAJECTORY_CANCELLED_MISC_KEY] and worker.reward == 0.25


def test_the_finalizer_is_a_pure_function_of_the_record(proven, given):
    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": RATING}}})
    first, second = copy.deepcopy(given[0]), copy.deepcopy(given[0])

    finalized_again(run, first, {"worker": RATING})
    finalized_again(run, second, {"worker": RATING})

    assert first.to_dict() == second.to_dict() != given[0].to_dict()
    assert [trajectory.misc["team"] for trajectory in first.trajectories.values()] == [
        trajectory.misc["team"] for trajectory in run.trajectories
    ]


def test_each_trajectory_names_its_session_agent_and_reward_row(proven):
    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": RATING}}})

    async def rows() -> dict[str, int]:
        async with Session.attach(session_name(run)) as ended:
            return {row.producer: row.id for row in await ended.annotations()}

    ids, session = asyncio.run(rows()), run.root.misc["team"]["session"]
    named = [trajectory.misc["team"] for trajectory in run.trajectories]
    assert named == [
        {"session": session, "agent": "lead", "row": ids["platoon"]},
        {"session": session, "agent": "w1", "row": ids[RATING], "spans": [None]},  # a row on its start alone
    ]
    assert set(run.root.misc) >= {"team", "outcome"} and "reason" not in run.root.misc["team"]


def test_a_root_whose_policy_differs_from_its_roles_fails_the_rollout(proven):
    """The root samples the default policy, and an agent spawned in the entry agent's role samples its role's, so a
    config that binds that role to another policy fails each rollout of a team that spawns one."""

    spawn = Call("agent_spawn", {"name": "lead-2", "child_role": "lead", "goal": GOAL})
    lead = {"harness": scripted([spawn], [Say("done")]), "may_spawn": ["lead"], "approvals": "allow"}
    looping = named(Team.model_validate({"roles": {"lead": lead}, "topology": {"limits": {"max_depth": 1}}}))
    gateway, other = make_gateway(ScriptedPolicy(unlocking)), make_gateway(ScriptedPolicy(unlocking))
    gateway.serve_policies("main", runtimes={"other": other}, roles={"lead": "other"})

    run = episode(looping, gateway, agents={"max_depth": 1})

    assert run.outcome["status"] == "error" and run.outcome["failure_kind"] == "finalizer:ValueError"
    said = "the root, lead, sampled policy 'main', and lead-2 of its role `lead` sampled 'other'"
    assert said in run.outcome["reason"]
