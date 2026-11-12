"""OpenHands on platoon's gateway (slow: openhands-sdk 1.59.0, which platoon-harness's openhands extra holds and which
never enters platoon's lock): an agent of a team on OpenHands calls on a platoon key of its own, so each of its calls
lands on its trajectory and joins the response id OpenHands reported.  Once its history passes what its condenser keeps,
the condenser asks for a summary with the admission's compaction headers: that call is ``aux``, outside the step budget,
and joins the id OpenHands reported for it."""

from __future__ import annotations

import asyncio
import importlib.util

import pytest
from platoon_harness.drivers.levels import Conformance
from platoon_harness.runtime.paths import write_conformance
from platoon_harness.testing import check_harness
from teams_toys import episode, held_to_identity, named, team

from platoon.testing import ScriptedPolicy, make_gateway, tool_call

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openhands") is None,
    reason="openhands-sdk is not installed for this interpreter: install platoon-harness's openhands extra beside "
    "platoon (docs/how-to/teams.md#install)",
)


@pytest.fixture(scope="module")
def openhands_checked(tmp_path_factory: pytest.TempPathFactory) -> Conformance:
    """The conformance result that earns OpenHands' sealed launch trainable here, made once."""

    return asyncio.run(check_harness("openhands", tmp_path_factory.mktemp("openhands-checks")))


def condensing(prompt: str) -> str:
    """A model that tries one wrong code after another, until OpenHands' condenser asks it for a summary, and once the
    summary stands in for its earlier tries, unlocks the box and is done."""

    if "maintaining a context-aware state summary" in prompt:  # OpenHands' condenser
        return "the summary"
    if "unlocked" in prompt:
        return "done"
    if "the summary" in prompt:
        return tool_call("environment_unlock", code="1")
    return tool_call("environment_unlock", code=str(100 + prompt.count("wrong code")))  # never the same try twice


def test_openhands_joins_platoons_gateway_and_its_condenser_is_aux(
    _sdk_home, openhands_checked, gateway_agents, admitted
):
    """Forty wrong tries, past which the condenser summarizes, then the box unlocked and done: forty-two policy steps,
    the whole budget, and the condenser's call beside them."""

    write_conformance(_sdk_home, openhands_checked)

    run = episode(named(team(solver={"harness": "openhands"})), make_gateway(ScriptedPolicy(condensing)), max_steps=42)

    held_to_identity(run, gateway_agents, admitted)
    kinds = [step.misc["action_misc"].get("kind", "policy") for step in run.root.steps]
    assert kinds == [*["policy"] * 40, "aux", *["policy"] * 2]
    assert run.root.misc["budget"] == {"enforce": "gateway", "max_steps": 42, "stops": 0}
    assert (run.outcome["status"], run.reward) == ("scored", 1.0)
