"""A function harness's root agent opens with its entry's root role (``environments[].agents.root_role``, else the one
its episode program declares, ``@pt.program(root_role=...)``, else ``policy``): it samples that role's policy for its
whole life, takes that role's tools, and is recorded under both, so that a root bound to a frozen policy makes no
datums while the subagents it spawns train; what the decorator refuses; and what ``platoon program show`` prints of
a program's root role."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inline_fakes import ToyService, routed_task

import platoon as pt
from platoon import cli
from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.programs import resolve_program
from platoon.utils.tinker_data_processing import RolloutPolicies, get_train_data_for_trajectory_collection


@pt.program("roottest-solving", root_role="solver")
async def solving(ep: pt.EpisodeContext) -> pt.Outcome:
    """Runs the root and grades it, recording the role the root ran with as the program saw it."""

    ep.root.misc["program_saw_role"] = ep.root.role
    await ep.run()
    return await ep.grade()


@pytest.fixture(autouse=True)
def _reset_service():
    ToyService.sessions.clear()
    yield
    ToyService.sessions.clear()


# Each policy's reply to the last user message, by its first matching prefix: the root delegates (goal:root) or submits
# (goal:direct), a child finishes, and a root that submitted, or was refused the tool, finishes.
REPLIES = {
    "goal:root": "SPAWN goal:child",
    "goal:child": "FINISH child",
    "goal:direct": "SUBMIT ok",
    "obs: submitted": "FINISH done",
    "obs: Tool": "FINISH refused",
    "obs:": "SUBMIT ok",
}


def reply(message: str) -> str:
    return next((text for prefix, text in REPLIES.items() if message.startswith(prefix)), "FINISH no rule")


@pytest.fixture
def served(make_gateway):
    """A gateway of policy main that serves helper to the agents of role solver; each with a sampler of its own."""

    main, main_sampler = make_gateway(reply)
    helper, helper_sampler = make_gateway(reply)
    main.serve_policies("main", runtimes={"helper": helper}, roles={"solver": "helper"})
    return main, main_sampler, helper_sampler


def run(gateway, task) -> tuple[dict[str, Any], dict]:
    from platoon.train.tinker.proxy import proxy_interactions

    interactions: dict = {}
    token = proxy_interactions.set(interactions)
    try:
        config = RolloutConfig(verbose=False, return_dict=True)
        collection = asyncio.run(run_harness_rollout(task, config, gateway=EmbeddedGateway(gateway)))
    finally:
        proxy_interactions.reset(token)
    assert isinstance(collection, dict)  # config.return_dict
    return collection, interactions


def agents_of(collection: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Each trajectory's (role, policy), the root's first."""

    return [
        (trajectory["misc"]["agent"]["role"], trajectory["misc"]["agent"].get("policy"))
        for trajectory in collection["trajectories"].values()
    ]


def test_a_root_of_a_bound_role_samples_its_policy_and_its_subagents_their_own(served):
    gateway, main_sampler, helper_sampler = served

    collection, interactions = run(gateway, routed_task(agents={"root_role": "solver", "max_depth": 1}))

    assert agents_of(collection) == [("solver", "helper"), ("subagent", "main")]
    assert helper_sampler.calls == ["goal:root", "obs: [completed] child", "obs: submitted"]
    assert main_sampler.calls == ["goal:child"]
    assert sorted(interaction.policy for interaction in interactions.values()) == ["helper"] * 3 + ["main"]
    root_id, child_id = collection["trajectories"]
    assert collection["trajectories"][root_id]["reward"] == 1.0  # its own submission, graded as any root's

    # Converted as in a run that trains main alone (helper frozen): the root's rows are left out, the subagent's train,
    # and both agents keep their tags.
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection,
        interactions=interactions,
        task_id="t0",
        checkpoint_version=0,
        tag_datums=True,
        policies=RolloutPolicies(default="main", versions={"main": 2, "helper": 0}, trained=frozenset({"main"})),
    )
    assert result.datums and set(result.datum_trajectories or ()) == {child_id}
    assert result.num_policy_excluded_datums > 0
    tags = {tag.trajectory_id: (tag.role, tag.policy) for tag in result.agents or ()}
    assert tags == {root_id: ("solver", "helper"), child_id: ("subagent", "main")}


def test_a_program_declares_its_root_s_role_and_an_entry_s_own_wins(served):
    gateway, _, helper_sampler = served

    declared, _ = run(gateway, routed_task(goal="goal:direct", program="roottest-solving"))
    overridden, _ = run(
        gateway, routed_task(goal="goal:direct", program="roottest-solving", agents={"root_role": "policy"})
    )

    (root,) = declared["trajectories"].values()
    assert agents_of(declared) == [("solver", "helper")] and root["misc"]["program_saw_role"] == "solver"
    assert helper_sampler.calls == ["goal:direct", "obs: submitted"] and root["reward"] == 1.0
    assert agents_of(overridden) == [("policy", "main")]  # the entry's root role over the program's


def test_a_root_takes_its_role_s_tools(served):
    gateway, _, _ = served
    gateway.limit_tools({"solver": ["look"]})

    collection, _ = run(gateway, routed_task(goal="goal:direct", agents={"root_role": "solver"}))

    (root,) = collection["trajectories"].values()
    assert root["misc"]["tools_refused"] == {"count": 1, "first": "submit"}  # its SUBMIT, refused unrun
    assert root["reward"] == 0.0 and ToyService.sessions[0].state.tool_calls == []


def test_a_root_without_a_root_role_takes_the_role_policy_and_samples_the_default(served):
    gateway, _, helper_sampler = served

    collection, _ = run(gateway, routed_task(goal="goal:direct"))

    assert agents_of(collection) == [("policy", "main")] and helper_sampler.calls == []


def test_a_rollout_of_a_program_that_does_not_resolve_fails_where_it_runs_the_program(served):
    """Its root opens with the role policy and the rollout fails as it did before root roles, naming the program."""

    gateway, main_sampler, helper_sampler = served

    collection, _ = run(gateway, routed_task(goal="goal:direct", program="tests.inline.nowhere:program"))

    (root,) = collection["trajectories"].values()
    assert agents_of(collection) == [("policy", "main")]
    assert root["misc"]["outcome"]["failure_kind"] == "unexpected:ModuleNotFoundError"
    assert root["misc"]["outcome"]["reason"].startswith("program tests.inline.nowhere:program: No module named")
    assert main_sampler.calls == helper_sampler.calls == []


def test_the_decorator_records_a_program_s_root_role_and_refuses_one_that_is_no_role_name(capsys):
    assert resolve_program("roottest-solving").root_role == "solver"
    cli.main(["program", "show", "roottest-solving"])
    assert capsys.readouterr().out.splitlines()[3] == "root role: solver"
    cli.main(["program", "show", "single"])
    assert not any(line.startswith("root role:") for line in capsys.readouterr().out.splitlines())
    assert resolve_program("single").root_role is None
    refused: list[Any] = ["", 3]
    for value in refused:
        with pytest.raises(ValueError, match=r"^@program\('roottest-bad'\): root_role must be a role name"):
            pt.program("roottest-bad", root_role=value)
    for role in ("verifier", "judge"):  # the roles of the agents that judge others
        with pytest.raises(ValueError) as raised:
            pt.program("roottest-bad", root_role=role)
        assert str(raised.value) == (
            f"@program('roottest-bad'): root_role {role!r} is the role of agents platoon opens to judge others, which "
            "never train, so no root takes it: give the program's roots a role of their own"
        )
