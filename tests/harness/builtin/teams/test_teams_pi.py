"""pi on platoon's gateway (slow: pi 1.5.0 on Node, found on PATH): an agent of a team on pi calls on a platoon key of
its own, so each of its calls lands on its trajectory and joins the call id pi reported.  When the gateway refuses a
request as longer than the context window, pi compacts through platoon-harness's extension, with the admission's
compaction headers, and retries: its compaction call is ``aux``, outside the step budget, and joins the id pi reported
for it.  And a pi agent the root's close cancels trains as retired unless a turn of its was under way."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

import pytest
from platoon_harness import Session
from platoon_harness.drivers.levels import Conformance
from platoon_harness.runtime.paths import write_conformance
from platoon_harness.testing import check_harness
from teams_toys import UNLOCK, episode, held_to_identity, named, session_name, team

from platoon.testing import ScriptedPolicy, last_user_message, make_gateway, tool_call

pytestmark = pytest.mark.skipif(shutil.which("pi") is None, reason="pi is not installed: no pi is on PATH")

# Each wrong try's words, about 30,000 tokens of the toy tokenizer: the sixth request passes the window below.
LONG = " ".join(["x" * 1000] * 30)
WINDOW = 150_000


@pytest.fixture(scope="module")
def pi_checked(tmp_path_factory: pytest.TempPathFactory) -> Conformance:
    """The conformance result that earns pi's sealed launch trainable here, made once."""

    return asyncio.run(check_harness("pi", tmp_path_factory.mktemp("pi-checks")))


def overflowing(prompt: str) -> str:
    """A model that tries a wrong code at length until the gateway refuses a request as too long, summarizes what pi
    asks it to, and once the summary stands in for its earlier tries, unlocks the box and is done."""

    if "context summarization assistant" in prompt:  # pi's own summarizer
        return "the summary"
    if "unlocked" in prompt:
        return "done"
    if "the summary" in prompt:
        return tool_call("mcp__environment__unlock", code="1")
    return LONG + tool_call("mcp__environment__unlock", code="9")


def test_pi_joins_platoons_gateway_and_compacts_as_aux(_sdk_home, pi_checked, gateway_agents, admitted):
    """Five long tries, a refused sixth request, one compaction and its retry: seven policy steps, the whole budget,
    and the compaction beside them."""

    write_conformance(_sdk_home, pi_checked)
    gateway = make_gateway(ScriptedPolicy(overflowing), context_window_length=WINDOW)

    run = episode(named(team(solver={"harness": "pi"})), gateway, max_steps=7)

    held_to_identity(run, gateway_agents, admitted)
    assert [step.misc["action_misc"].get("kind", "policy") for step in run.root.steps] == [
        *["policy"] * 5,
        "aux",
        *["policy"] * 2,
    ]
    assert [agent.context_overflows for agent in gateway_agents] == [1]
    assert run.root.misc["budget"] == {"enforce": "gateway", "max_steps": 7, "stops": 0}
    assert (run.outcome["status"], run.reward) == ("scored", 1.0)


TRIES = "You try one code, and wait to hear from your lead."
WAIT = "mcp__ph__wait"


def parking(prompt: str) -> str:
    """The model of a tree on pi: the lead spawns a worker and waits for it to end, and the worker tries a wrong code
    and waits to hear from the lead, so each ends its turn parked, and the worker never finishes."""

    last = last_user_message(prompt)
    if '"parked"' in last:
        return "waiting"
    if TRIES in prompt:
        tried = last.startswith("<tool_response>")
        return tool_call(WAIT, {"from": ["@lead"]}) if tried else tool_call(UNLOCK, code="9")
    if "code is" in last:
        return tool_call("mcp__ph__agent_spawn", name="w1", child_role="worker", goal="Try a code.")
    return tool_call(WAIT, agents=["@w1"])


async def end_of(run: Any, agent: str) -> list[Any]:
    """An agent's end in a rollout's session."""

    async with Session.attach(session_name(run)) as ended:
        return [event async for event in ended.events(streams=[f"agent/{agent}"]) if event.type == "agent.ended"]


def test_a_pi_agent_the_close_cancels_trains_as_retired_unless_a_turn_of_its_was_under_way(_sdk_home, pi_checked):
    """A pi lead parked on its pi worker's end, which never comes, as the worker parks to hear from the lead and never
    finishes: the session stalls, and the root retires the worker, which breaks in on the lead's wait, so the root
    cancels the lead. The lead's run is cancelled exactly when its end names a turn of its, which pi had begun first.
    Which way that goes is pi's, so this fails on a close that cancels the lead's run only when pi is the slower; the
    deterministic guard is the ScriptedDriver's
    `test_an_agent_the_close_cancels_with_no_turn_under_way_trains_as_retired`."""

    write_conformance(_sdk_home, pi_checked)
    crew = team(
        lead={"harness": "pi", "may_spawn": ["worker"], "instructions": "You lead."},
        worker={"harness": "pi", "instructions": TRIES},
    )

    run = episode(named(crew), ScriptedPolicy(parking), agents={"budget": "depth_aware"})

    (lead,), (worker,) = asyncio.run(end_of(run, "lead")), asyncio.run(end_of(run, "w1"))
    assert (worker.data["status"], lead.data["status"]) == ("retired", "cancelled")
    assert run.root.misc["agent_result"]["status"] == ("completed" if lead.turn is None else "cancelled")
