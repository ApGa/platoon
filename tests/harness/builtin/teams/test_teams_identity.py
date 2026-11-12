"""How a model call is tied to an agent, on platoon's gateway and a scripted policy: each agent of a team calls on a
platoon key of its own, so each call lands on its own agent's trajectory and joins a call its agent reported, and no key
is written anywhere.  The ScriptedDriver reference and a wire program, shared and isolated, are held to it here
(``held_to_identity``), pi and OpenHands in the slow fixtures; a harness that cannot launch sealed is refused, a native
child is never split from its parent, and two rollouts share one loop."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from aiohttp import web
from platoon_harness import Session, Team
from platoon_harness.bwrap import Bubblewrap
from platoon_harness.runtime.paths import write_conformance
from platoon_harness.testing import AskModel, Call, Compact, Say
from teams_toys import (
    KEEPING,
    episode,
    held_to_identity,
    journal,
    leaked,
    named,
    prove,
    rollout,
    scripted,
    session_name,
    team,
    wire,
)

from platoon.gateway.server import GatewayServer
from platoon.harness.builtin.platoon_harness import bridge
from platoon.harness.builtin.platoon_harness.authority import PlatoonAuthority
from platoon.utils.trajectory_status import EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY, TRAJECTORY_INVALID_MISC_KEY

GOAL = "Open the box. Its code is 1."
SPAWNING = [Call("agent_spawn", {"name": "w1", "child_role": "worker", "goal": GOAL})]


def tree(worker: Any) -> str:
    """A team whose lead spawns one worker, ``w1``, running ``worker``, and says it is done once the worker reports."""

    lead = {"harness": scripted(SPAWNING, [Say("the worker has it")]), "may_spawn": ["worker"]}
    return named(team(lead=lead, worker={"harness": worker}))


TREE = tree(scripted([AskModel(), Compact(), Call("agent_finish", {"gist": "unlocked"})]))
PAIR = named(team(alpha={"harness": scripted()}, beta={"harness": scripted()}))


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def by_agent(run: Any) -> dict[str | None, Any]:
    return {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}


def completions(trajectory: Any) -> list[str]:
    return [step.misc["action_misc"]["completion_id"] for step in trajectory.steps]


def test_every_call_lands_on_its_own_agents_trajectory(proven, wired, gateway_agents, admitted):
    """On the ScriptedDriver reference, spawned, with a compaction, and on a wire program, minted beside it."""

    write_conformance(proven, wired)
    spawned, minted = episode(TREE), episode(named(team(alpha={"harness": scripted()}, beta={"harness": wire()})))

    for run, asking in ((spawned, ["w1"]), (minted, ["alpha", "beta"])):
        held_to_identity(run, gateway_agents, admitted)
        ours = {agent: trajectory for agent, trajectory in by_agent(run).items() if agent}
        made = {agent: completions(trajectory) for agent, trajectory in ours.items()}
        tools = {
            agent: [call["name"] for call in trajectory.misc.get("tool_calls", ())]
            for agent, trajectory in ours.items()
        }
        assert [agent for agent, calls in made.items() if calls] == asking
        assert sum(map(len, made.values())) == len(run.policy.calls)
        assert tools == {agent: ["unlock"] if agent in asking else [] for agent in ours}  # each agent's own record
    kinds = [step.misc["action_misc"].get("kind", "policy") for step in by_agent(spawned)["w1"].steps]
    assert kinds == ["policy", "policy", "aux"]


def test_an_isolated_agent_trains_on_a_key_of_its_own(proven, wired, gateway_agents, admitted):
    """A worker isolated by platoon-harness's ``bwrap``, which writes only in its own folder and sees nothing of your
    home or of any session, runs the wire program, spawned through platoon and launched sealed on its gateway by the
    bridge's authority: it reaches the gateway and its tools on loopback on a key of its own, its calls land on its own
    trajectory, exact and eligible, its start records the ``isolated`` assurance with trainable met, and no form of a
    key is written anywhere.  Where bubblewrap cannot confine, it says so and skips."""

    if (why := Bubblewrap().refusal()) is not None:
        pytest.skip(f"bubblewrap cannot confine here, so no isolated agent ran: {why}")
    write_conformance(proven, wired)
    lead = {"harness": scripted(SPAWNING, [Say("the worker has it")]), "may_spawn": ["worker"]}
    worker = {"harness": wire(), "workspace": "isolated"}
    roles = {name: {**role, "approvals": "allow"} for name, role in (("lead", lead), ("worker", worker))}

    run = episode(named(Team.model_validate({"extensions": ["bwrap"], "roles": roles})))

    held_to_identity(run, gateway_agents, admitted)
    started = {event.agent: event.data for event in asyncio.run(events(run)) if event.type == "agent.started"}
    assert {agent: (data["assurance"], data["levels"]["trainable"]["state"]) for agent, data in started.items()} == {
        "lead": ("cooperative", "met"),
        "w1": ("isolated", "met"),
    }
    isolated = by_agent(run)["w1"]
    assert len(completions(isolated)) == len(run.policy.calls) == 2
    assert [call["name"] for call in isolated.misc["tool_calls"]] == ["unlock"]
    assert run.outcome["status"] == "scored"


class Leaking(PlatoonAuthority):
    """PlatoonAuthority under a root whose endpoint holds its key in a query, which no gateway URL does."""

    def __init__(self, root: Any, bridge: Any, *, entry: str | None) -> None:
        endpoint = root.endpoint
        held = dataclasses.replace(endpoint, base_url=f"{endpoint.base_url}?key={endpoint.api_key}")
        super().__init__(SimpleNamespace(endpoint=held), bridge, entry=entry)


@pytest.mark.parametrize("case", ["rollout", "mint-failure", "refused-url"])
def test_no_gateway_key_is_written_anywhere(proven, wired, gateway_agents, monkeypatch, caplog, tmp_path, case):
    """After a rollout whose worker runs a wire program, a mint that fails on a URL holding the minted key, and a root
    whose URL holds its key, no form of any key is in the platoon-harness root (journal, annotations, logs, agent
    folders and the files drivers wrote), the rollout's working directory or files, trajectory misc and error messages,
    the logs, or a driver's command line."""

    write_conformance(proven, wired)
    caplog.set_level(logging.DEBUG)
    cwd, note = tmp_path / "cwd", tmp_path / "note.json"
    keep = {"producer": KEEPING, "into": str(cwd)}
    if case == "rollout":
        run = episode(tree(wire("note", str(note))), harness_options={"rewards": {"worker": keep}})
        assert json.loads(note.read_text())["argv"][1:] == ["note", str(note)]
    elif case == "mint-failure":
        minted = GatewayServer._create_agent

        async def under_the_key(self: GatewayServer, request: Any) -> web.Response:
            reply = json.loads((await minted(self, request)).text)
            origin = reply["base_url"].removesuffix("/v1")
            return web.json_response({**reply, "base_url": f"{origin}/s/{reply['api_key']}/v2"}, status=201)

        monkeypatch.setattr(GatewayServer, "_create_agent", under_the_key)
        run = episode(PAIR, harness_options={"rewards": {"alpha": keep}})
        failed = [event.data["error"] for event in asyncio.run(events(run)) if event.type == "effect.failed"]
        assert failed == ["ValueError: the mint's base_url has a path other than /v1 or /mcp"] * 2
    else:
        monkeypatch.setattr(bridge, "PlatoonAuthority", Leaking)
        run = episode(named(team(solver={"harness": scripted()})))
        assert "the root's endpoint holds a query or a fragment" in run.outcome["reason"]

    keys = [agent.api_key for agent in gateway_agents]
    places = [proven, run.output_dir, cwd, note]
    texts = {"the collection": json.dumps(run.collection.to_dict(), default=str), "the logs": caplog.text}
    assert keys and leaked(keys, places, texts) == []


async def events(run: Any) -> list[Any]:
    async with Session.attach(session_name(run)) as ended:
        return [event async for event in ended.events()]


@pytest.mark.parametrize(
    "harness",
    ["codex", "claude", {"command": ["/bin/true"], "mode": "oneshot"}],
    ids=["codex", "claude", "exec"],
)
def test_codex_claude_and_exec_are_refused_from_a_trainable_team(proven, monkeypatch, tmp_path, harness):
    """None of them launches sealed: codex's Responses dialect drops its tools behind a gateway, Claude Code's Messages
    dialect is refused its system role, and exec reports no calls.  So the session's start refuses the team before any
    harness launches, and the rollout is its program's error, never a score.  The harnesses' own binaries stay off
    PATH, and the test's home is its own, so that nothing of the user's runs."""

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    run = episode(named(team(solver={"harness": harness})))

    assert (run.outcome["status"], run.outcome["failure_kind"]) == ("error", "unexpected:ConfigError")
    assert "has no sealed launch, which a trainable session needs" in run.outcome["reason"]
    assert run.root.misc[TRAJECTORY_INVALID_MISC_KEY] and not (proven / "sessions").exists()


def test_a_native_child_is_never_exact_and_never_split(proven, wired):
    """A wire program that reports a native child of its own is ``forked``: its child calls on its key, so their calls
    are its own, on its one trajectory, and none of them trains."""

    write_conformance(proven, wired)

    run = episode(tree(wire("forks")), agents={"child_grading": "env"})

    held = asyncio.run(journal(session_name(run)))
    answer, worker = held.eligibility["w1"], by_agent(run)["w1"]
    assert (answer.attribution, answer.verdict, [bar.reason for bar in answer.reasons]) == (
        "forked",
        "ineligible",
        ["forked"],
    )
    assert completions(worker) == [call.call_id for call in answer.calls] != []
    assert list(by_agent(run)) == ["lead", "w1"]  # no trajectory of the native child's
    assert worker.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY]


def test_two_rollouts_share_one_worker_loop(proven):
    """Two rollouts in one event loop, as a rollout worker runs them: each starts its session while the other's runs,
    and each agent's calls land on its own rollout."""

    async def both() -> list[Any]:
        return list(await asyncio.gather(rollout(TREE), rollout(TREE)))

    runs = asyncio.run(both())

    held = [asyncio.run(events(run)) for run in runs]
    spans = [(found[0].wall_time, found[-1].wall_time) for found in held]
    assert [run.outcome["status"] for run in runs] == ["scored", "scored"]
    assert len({run.root.misc["team"]["session"] for run in runs}) == 2
    assert spans[0][0] < spans[1][1] and spans[1][0] < spans[0][1]
    for run, found in zip(runs, held, strict=True):
        reported = [event.data["call_id"] for event in found if event.type == "model.called"]
        assert reported == completions(by_agent(run)["w1"]) and len(reported) == len(run.policy.calls)
