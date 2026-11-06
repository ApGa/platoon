"""An entry's root role in its route (``environments[].agents.root_role``): a route travels to rollout workers, and into
every trajectory, without the key when its entry leaves it unset, so that every rollout request of a config that sets
none keeps its bytes, and with it when set; which role an entry's root takes (its own, else its program's, else
``policy``), and what a rollout does with a program that does not resolve; what loading refuses of the key; and that a
TOML harness spec's entry may name its root's role, the one ``agents`` setting that applies to a spec."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict

import pytest

import platoon as pt
from platoon.harness.registry import get_harness
from platoon.harness.rollout import root_role_of
from platoon.task import Task
from platoon.train.auto import environment_route, harness_setting_problems
from platoon.train.components import (
    ROUTE_MISC_KEY,
    AgentTreeConfig,
    EnvironmentConfig,
    EnvironmentRoute,
    VerifierConfig,
    entry_root_role,
)

NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"


@pt.program("roottest-generating", root_role="generator")
async def generating(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()
    return await ep.grade()


def _agents_before_root_roles(**settings) -> dict:
    """An entry's ``agents`` as a route carried them before routes knew of root roles, key for key, in order."""

    return {
        "max_depth": None,
        "budget": "linear",
        "child_max_steps": 15,
        "child_grading": "none",
        "child_access": "full",
        "verifier": asdict(VerifierConfig()),
        "fork": None,
        "child_timeout_s": None,
    } | settings


def test_a_route_whose_entry_names_no_root_role_travels_as_it_did_before():
    entry = EnvironmentConfig(service=NUMBER_SEARCH, harness="platoon-react", agents=AgentTreeConfig(max_depth=1))
    route = environment_route(entry)
    payload = route.attach(Task(id="val:0")).misc[ROUTE_MISC_KEY]

    assert "root_role" not in payload["agents"]
    # The same keys, values and order as before: a pickled request, and a golden's JSON, keep their bytes.
    assert json.dumps(payload["agents"]) == json.dumps(_agents_before_root_roles(max_depth=1))
    assert list(payload) == list(asdict(route))
    expected = asdict(route)
    del expected["agents"]["root_role"]
    assert pickle.dumps(payload) == pickle.dumps(expected)


def test_a_route_carries_its_entry_s_root_role_to_the_worker_and_into_trajectories():
    entry = EnvironmentConfig(
        service=NUMBER_SEARCH, harness="platoon-react", agents=AgentTreeConfig(root_role="solver")
    )
    task = environment_route(entry).attach(Task(id="val:0"))
    trajectory = {"task": {"misc": pickle.loads(pickle.dumps(task.misc))}}

    assert task.misc[ROUTE_MISC_KEY]["agents"]["root_role"] == "solver"
    for route in (EnvironmentRoute.of_task(task), EnvironmentRoute.of_trajectory(trajectory)):
        assert route is not None and route.agents.root_role == "solver"
    unnamed = EnvironmentRoute.of_task(EnvironmentRoute(name="e").attach(Task(id="t")))
    assert unnamed is not None and unnamed.agents.root_role is None


def test_an_entry_s_root_takes_its_own_role_else_the_one_its_program_declares():
    def entry(**settings) -> EnvironmentConfig:
        return EnvironmentConfig(service=NUMBER_SEARCH, harness="platoon-react", **settings)

    assert entry_root_role(entry(agents={"root_role": "solver"})) == "solver"
    assert entry_root_role(entry(program="roottest-generating")) == "generator"
    assert entry_root_role(entry(program="roottest-generating"), programs=False) is None  # no program resolved
    assert entry_root_role(entry(program="roottest-generating", agents={"root_role": "policy"})) == "policy"
    assert entry_root_role(entry()) is None  # single declares none: the root's role is policy
    route = environment_route(entry(program="roottest-generating"))
    assert route.agents.root_role is None and entry_root_role(route) == "generator"  # the route carries no default
    assert root_role_of(route) == "generator" and root_role_of(environment_route(entry())) == "policy"


def test_a_rollout_of_a_program_that_does_not_resolve_opens_its_root_as_before():
    """Resolving the program raises its error, and a rollout opens the root with the role policy, then fails where it
    runs the program, saying why, as it did before root roles."""

    route = environment_route(
        EnvironmentConfig(service=NUMBER_SEARCH, harness="platoon-react", program="tests.train.nowhere:program")
    )

    with pytest.raises(ModuleNotFoundError):
        entry_root_role(route)
    assert root_role_of(route) == "policy"


@pytest.mark.parametrize("value", ["", 3, ["solver"]])
def test_a_root_role_that_is_no_role_name_is_refused(value):
    with pytest.raises(ValueError, match=r"^agents\.root_role must be a role name, a non-empty string, or null; got "):
        AgentTreeConfig(root_role=value)


@pytest.mark.parametrize("value", ["verifier", "judge"])
def test_a_root_role_of_the_agents_that_judge_others_is_refused(value):
    with pytest.raises(ValueError) as raised:
        AgentTreeConfig(root_role=value)
    assert str(raised.value) == (
        f"agents.root_role: {value!r} is the role of agents platoon opens to judge others, which never train, so no "
        "root takes it: give the root a role of its own, and bind that role to the judge's policy "
        "(agents.roles.<role>.policy)"
    )


def test_a_toml_harness_spec_s_entry_may_name_its_root_s_role_and_no_other_agents_setting():
    def problems(**agents) -> list[str]:
        environment = EnvironmentConfig(service=NUMBER_SEARCH, harness="platoon-bash", agents=AgentTreeConfig(**agents))
        return harness_setting_problems(environment, [get_harness(ref) for ref in environment.harness_refs()])

    assert problems(root_role="solver") == []
    assert problems(root_role="solver", max_depth=0) == []
    [refused] = problems(root_role="solver", max_depth=1)
    assert refused == (
        "environments[default]: agents settings never apply: harness platoon-bash@1 is a TOML harness spec, which "
        "cannot spawn subagents; use a function harness, or leave out every agents setting but root_role"
    )
