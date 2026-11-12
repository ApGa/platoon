"""The ``platoon-harness`` program and harness, each rolled out on the lockbox through platoon's gateway and a scripted
policy: the root rule, the two admission routes, the refusal kept on the bridge, the rollout's working directory, the
closing sequence and the root's report, budget stops, ``single``'s finalizers, platoon's grades and the producers."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from platoon_harness import Session, Team
from platoon_harness.credit import PROPAGATE
from platoon_harness.runtime.paths import write_conformance
from platoon_harness.testing import AskModel, Call, Exit, Hang, Say
from teams_toys import (
    MAX_STEPS,
    RATING,
    SOLO,
    UNLOCK,
    WITNESSED,
    WITNESSING,
    episode,
    named,
    prove,
    scripted,
    session_name,
    team,
    unlocking,
    wayward,
    wire,
)
from testing_toys import LOCKBOX

import platoon as pt
from platoon.gateway.core import BUDGET_STOP_PREFIX
from platoon.gateway.server import GatewayServer
from platoon.harness.builtin.platoon_harness import bridge
from platoon.testing import ScriptedPolicy, run_episode, tool_call
from platoon.train import components
from platoon.utils.trajectory_status import (
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_INVALID_MISC_KEY,
)

GOAL = "Open the box. Its code is 1."
PROPAGATING = {"producer": PROPAGATE, "retained": 0.5}
VERDICT = '{"status": "verified", "score": 1.0}'
FINISH = Call("agent_finish", {"gist": "unlocked"})
WORKER = scripted([AskModel(), FINISH])
PAIR = named(team(alpha={"harness": scripted()}, beta={"harness": scripted()}))


def spawning(*names: str, role: str = "worker") -> list[Any]:
    """A turn that spawns ``names`` of ``role``, each told the lockbox's code."""

    return [Call("agent_spawn", {"name": name, "child_role": role, "goal": GOAL}) for name in names]


TREE = named(
    team(
        lead={"harness": scripted(spawning("w1"), [Say("the worker has it")]), "may_spawn": ["worker"]},
        worker={"harness": WORKER},
    )
)


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def wrong_code(prompt: str) -> str:
    """A model that never finds the code, and so never stops on its own."""

    return tool_call(UNLOCK, code="9")


async def read(name: str) -> dict[str, Any]:
    """What a rollout's ended session holds: its events, its agents and its annotations."""

    async with Session.attach(name) as ended:
        return {
            "events": [event async for event in ended.events()],
            "agents": {row["agent"]: row for row in await ended.agents()},
            "annotations": await ended.annotations(),
        }


def session_of(run: Any) -> dict[str, Any]:
    return asyncio.run(read(session_name(run)))


def by_agent(run: Any) -> dict[str | None, pt.Trajectory]:
    return {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}


def test_team_refuses_to_run_outside_its_program(proven):
    run = asyncio.run(
        run_episode(
            "test:1",
            service=LOCKBOX,
            harness="platoon-harness",
            program="single",
            harness_options={"team": SOLO},
            policy=ScriptedPolicy(unlocking),
            max_steps=MAX_STEPS,
        )
    )

    assert run.root.misc["agent_result"]["error"] == (
        "harness platoon-harness runs under `program: platoon-harness`, which gives it its team"
    )
    assert (run.outcome["status"], run.outcome["failure_kind"]) == ("error", "harness_crash")
    assert not (proven / "sessions").exists()


def test_team_is_a_member_exactly_when_its_options_carry_a_handoff(proven):
    run = episode(TREE)

    root, worker = run.trajectories
    assert [str(path) for path in (proven / "sessions").iterdir()] == [root.misc["team"]["session"]]
    assert worker.misc["agent_result"] == {"status": "completed", "error": None}  # the member's run
    assert worker.finish_message.startswith("unlocked")  # its report: the gist, then its last words


def test_a_team_that_starts_one_agent_gives_it_the_roots_endpoint(proven):
    run = episode(SOLO)

    (root,) = run.trajectories
    assert (run.outcome["status"], run.reward) == ("scored", 1.0)
    assert root.misc["team"]["agent"] == "solver"
    assert len(root.steps) == 2  # the agent's two calls are the root's
    admitted = {event.agent: event.data["result"] for event in session_of(run)["events"] if _admit(event)}
    assert admitted == {"solver": {"agent_id": root.misc["agent"]["agent_id"]}}


def test_a_mesh_runs_under_a_zero_call_root(proven):
    run = episode(PAIR)

    root, *peers = run.trajectories
    assert (root.steps, root.misc["team"]["agent"], root.misc["agent_result"]["status"]) == ([], None, "completed")
    assert sorted(peer.misc["team"]["agent"] for peer in peers) == ["alpha", "beta"]
    assert all(peer.parent_info.id == root.id and peer.steps for peer in peers)
    assert run.outcome["status"] == "scored"


def test_a_child_of_a_platoon_context_is_spawned_and_any_other_is_minted(proven):
    """A spawned agent runs the member harness under platoon, so its run is recorded; a minted one runs none, and is
    minted on its parent's key, so platoon records it below its parent."""

    tree = by_agent(episode(TREE))
    mesh = by_agent(
        episode(
            named(
                team(
                    alpha={"harness": scripted(spawning("w1"), [Say("done")]), "may_spawn": ["worker"]},
                    beta={"harness": scripted([Say("standing by")])},
                    worker={"harness": WORKER},
                )
            )
        )
    )

    assert tree["w1"].parent_info.id == tree["lead"].id and "agent_result" in tree["w1"].misc
    assert mesh["w1"].parent_info.id == mesh["alpha"].id and "agent_result" not in mesh["w1"].misc
    assert mesh["alpha"].parent_info.id == mesh[None].id and "agent_result" not in mesh["alpha"].misc


def test_a_refused_spawn_fails_with_platoons_words(proven):
    run = episode(TREE, agents={"max_depth": 0})

    failed = [event for event in session_of(run)["events"] if event.type == "effect.failed" and event.agent == "w1"]
    assert len(run.trajectories) == 1  # platoon opened no agent
    assert [event.data["kind"] for event in failed] == ["admit"]
    assert "Not enough budget to launch subagent" in failed[0].data["error"]
    assert "the maximum allowed depth of 0" in failed[0].data["error"]


def test_a_lost_mint_is_never_retried_and_its_orphan_never_trains(proven, monkeypatch):
    minted = GatewayServer._create_agent
    answered: list[int] = []

    async def losing_the_first(self: GatewayServer, request: Any) -> Any:
        response = await minted(self, request)
        answered.append(response.status)
        if len(answered) == 1:
            request.transport.close()  # the agent is minted, and its answer never arrives
        return response

    monkeypatch.setattr(GatewayServer, "_create_agent", losing_the_first)
    run = episode(PAIR)

    agents = session_of(run)["agents"]
    (orphan,) = [trajectory for trajectory in run.trajectories[1:] if trajectory.misc["team"]["agent"] is None]
    assert answered == [201, 201]  # one mint for each agent, none again
    assert sorted(row["status"] for row in agents.values()) == ["failed", "retired"]
    assert orphan.steps == [] and orphan.misc[EXCLUDE_FROM_TRAINING_MISC_KEY]


class Grading(components.EnvironmentRoute):
    """An entry that grades a harness that crashed or ran out of time, as the environment was left."""

    def __init__(self, **fields: Any) -> None:
        fields["on_harness_exit"] = "grade"
        super().__init__(**fields)


def unmade(*args: Any, **kwargs: Any) -> Any:
    raise ValueError("the root's endpoint has a path other than /v1 or /mcp")


@pytest.mark.parametrize(
    ("kept", "authority", "refusal"),
    [
        pytest.param(False, bridge.PlatoonAuthority, ("ConfigError", "write_conformance"), id="no-kept-result"),
        pytest.param(True, unmade, ("ValueError", "a path other than /v1 or /mcp"), id="no-authority"),
    ],
)
def test_a_refusal_at_the_sessions_start_is_an_error_outcome_never_a_score(
    _sdk_home, tmp_path, monkeypatch, kept, authority, refusal
):
    """Whatever keeps the session from starting, a team whose harness has no conformance result kept or an authority
    that cannot be made, is the program's ``error``, even under an ``on_harness_exit`` that grades a crashed harness."""

    monkeypatch.setattr(components, "EnvironmentRoute", Grading)
    monkeypatch.setattr(bridge, "PlatoonAuthority", authority)
    if kept:
        prove(_sdk_home)

    run = episode(SOLO)

    kind, said = refusal
    assert (run.outcome["status"], run.outcome["failure_kind"]) == ("error", f"unexpected:{kind}")
    assert said in run.outcome["reason"]
    assert run.root.misc[TRAJECTORY_INVALID_MISC_KEY]
    assert not list((tmp_path / "scratch").glob("platoon-harness-rollout-*"))  # its working directory, removed


def test_the_programs_cwd_exists_before_the_session_and_is_removed_after_the_producers(proven):
    WITNESSED.clear()

    run = episode(TREE, harness_options={"rewards": {"worker": {"producer": WITNESSING}}})

    (seen,) = WITNESSED
    assert run.outcome["status"] == "scored"
    assert (seen["files"], seen["commits"]) == ([".git"], ["platoon: An empty repository for one rollout"])
    assert not seen["cwd"].exists()


def test_a_rollouts_session_has_ended_before_its_producers_run(proven):
    WITNESSED.clear()

    episode(TREE, harness_options={"rewards": {"worker": {"producer": WITNESSING}}})

    (seen,) = WITNESSED
    assert seen["state"] in ("closed", "stopped")
    assert seen["agents"] == {"lead": "ended", "w1": "ended"}
    assert seen["judge"] is not None  # the rollout's judges are on


def test_every_agent_ends_before_the_root_returns(proven):
    run = episode(TREE)

    held = session_of(run)
    ended = {event.agent: event.seq for event in held["events"] if event.type == "agent.ended"}
    (stopped,) = [event.seq for event in held["events"] if event.type == "session.stopped"]
    assert set(ended) == {"lead", "w1"} and max(ended.values()) < stopped
    assert all("agent_result" in trajectory.misc for trajectory in run.trajectories)


def ends_of(run: Any) -> dict[str | None, Any]:
    """Each agent's end in a rollout's session, by agent."""

    return {event.agent: event for event in session_of(run)["events"] if event.type == "agent.ended"}


def waiting(name: str) -> Call:
    """A step that waits for ``name`` to end: its turn then ends with its agent parked on the wait."""

    return Call("wait", {"agents": [f"@{name}"]})


def hearing(name: str) -> Call:
    """A step that waits to hear from ``name``: its turn then ends with its agent parked on the wait, so that one that
    does not stay neither finishes nor reports."""

    return Call("wait", {"from": [f"@{name}"]})


def test_an_agent_the_close_cancels_with_no_turn_under_way_trains_as_retired(proven):
    """A tree whose lead waits for its child to end, as the child waits for its own, and whose leaf waits to hear from
    its parent, which never says anything, so it ends its turn and never finishes: the session stalls, still open, and
    the root ends every agent, deepest first.  Retiring the leaf breaks in on its parent's wait, so the parent is on
    its way to a turn and the root cancels it, which breaks in on the lead's, and the root cancels the lead too.  On a
    harness slow to begin that turn, each ends with no turn under way, every turn it took whole: the record keeps each
    cancel, and each trains as retired, a member and the entry agent alike, so no run is cancelled and the rollout is
    scored."""

    crew = team(
        lead={
            "harness": wayward([AskModel(), *spawning("m1", role="mid"), waiting("m1")], begins=1),
            "may_spawn": ["mid"],
        },
        mid={
            "harness": wayward([AskModel(), *spawning("l1", role="leaf"), waiting("l1")], begins=1),
            "may_spawn": ["leaf"],
        },
        leaf={"harness": scripted([AskModel(), hearing("m1")])},
    )

    run = episode(named(crew), agents={"budget": "depth_aware"})

    ended = {agent: (end.data["status"], end.turn) for agent, end in ends_of(run).items()}
    mid = by_agent(run)["m1"]
    assert ended == {"l1": ("retired", None), "m1": ("cancelled", None), "lead": ("cancelled", None)}
    assert [run.root.misc["agent_result"]["status"], mid.misc["agent_result"]["status"]] == ["completed"] * 2
    assert mid.steps and not any(step.misc.get(EXCLUDE_FROM_TRAINING_MISC_KEY) for step in mid.steps)
    assert not any(trajectory.misc.get(TRAJECTORY_CANCELLED_MISC_KEY) for trajectory in run.trajectories)
    assert run.outcome["status"] == "scored"


def test_an_agent_whose_turn_the_close_cuts_stays_cancelled(proven, monkeypatch):
    """The lead of a tree that stalls, waiting for its worker to end as the worker waits to hear from it, begins its
    turn on the worker's retirement before the root comes to it: the root's cancel cuts that turn, which its end names,
    so the lead's run stays cancelled, and so does the rollout."""

    retire = Session.retire

    async def retiring(self: Session, agent: str) -> None:
        head = await self.head()
        await retire(self, agent)
        if agent == "@w1":  # the root comes to the lead once the lead's turn on w1's end has begun
            async for event in self.events(head, streams=["turn/lead"], follow=True):
                if event.type == "turn.started":
                    break

    monkeypatch.setattr(Session, "retire", retiring)
    crew = team(
        lead={
            "harness": scripted([*spawning("w1"), waiting("w1")], [Hang()]),
            "may_spawn": ["worker"],
        },
        worker={"harness": scripted([AskModel(), hearing("lead")])},
    )

    run = episode(named(crew), agents={"budget": "depth_aware"})

    lead = ends_of(run)["lead"]
    assert (lead.data["status"], lead.turn is not None) == ("cancelled", True)
    assert run.root.misc["agent_result"]["status"] == "cancelled"
    assert run.outcome["status"] == "cancelled"


def test_a_child_that_ends_with_its_parent_with_no_turn_under_way_trains_as_retired(proven):
    """A lead hears from its middle agent, which heard its leaf's first turn end and messaged it, and cancels the
    middle agent.  The message handed the leaf a turn its harness, slow to begin it, never begins, so the leaf ends
    with its parent with no turn under way, every turn it took whole: platoon-harness ends it retired, not cancelled,
    and its run is completed, while the middle agent, cancelled by name, stays cancelled."""

    again = Call("message_send", {"to": "@l1", "gist": "again"})
    lead = [*spawning("m1", role="mid"), hearing("m1")], [Call("agent_cancel", {"agent": "@m1"})]
    roles = {
        "lead": {"harness": scripted(*lead), "may_spawn": ["mid"]},
        "mid": {"harness": scripted([*spawning("l1", role="leaf"), hearing("l1")], [again]), "may_spawn": ["leaf"]},
        "leaf": {"harness": wayward([AskModel()], begins=1), "stays": True},
    }
    crew = {"roles": {name: {**role, "approvals": "allow"} for name, role in roles.items()}}
    topology = {"edges": {"mid": ["children"]}}

    run = episode(named(Team.model_validate({**crew, "topology": topology})), agents={"budget": "depth_aware"})

    ends, record = ends_of(run), session_of(run)["events"]
    turned = max(event.seq for event in record if event.type == "turn.ended" and event.agent == "l1")
    offered = [event.seq for event in record if event.type == "context.offered" and event.stream == "context/l1"]
    assert (ends["l1"].data["status"], ends["l1"].turn, ends["m1"].data["status"]) == ("retired", None, "cancelled")
    assert any(turned < seq < ends["l1"].seq for seq in offered)  # handed a turn after its last, before its end
    assert by_agent(run)["l1"].misc["agent_result"]["status"] == "completed"


STALLING = [*spawning("w1"), waiting("w1")]


@pytest.mark.parametrize(
    "lead",
    [
        pytest.param({"harness": scripted(STALLING, [FINISH])}, id="finishes"),
        pytest.param({"harness": scripted(STALLING, [AskModel()]), "stays": False}, id="leaves"),
    ],
)
def test_an_agent_that_ends_of_itself_after_the_close_read_it_ends_as_its_record_says(proven, monkeypatch, lead):
    """The close reads the agents once, before it ends any.  The lead of a tree that stalls, waiting for its worker to
    end as the worker waits to hear from it, is woken by the worker's retirement and ends of itself before the root
    comes to it: it finishes, or, as one that does not stay, ends its turn on its own and so finishes.  The root takes
    it as ended, as its record says, so its run and the rollout's are whole."""

    retire = Session.retire

    async def retiring(self: Session, agent: str) -> None:
        head = await self.head()
        await retire(self, agent)
        if agent == "@w1":  # the root comes to the lead, which it read as live, once the lead has ended
            async for event in self.events(head, streams=["agent/lead"], follow=True):
                if event.type == "agent.ended":
                    break

    monkeypatch.setattr(Session, "retire", retiring)
    crew = team(lead={**lead, "may_spawn": ["worker"]}, worker={"harness": scripted([AskModel(), hearing("lead")])})

    run = episode(named(crew))

    assert {agent: end.data["status"] for agent, end in ends_of(run).items()} == {"w1": "retired", "lead": "finished"}
    assert run.root.misc["agent_result"]["status"] == "completed"
    assert run.outcome["status"] == "scored"


def test_a_zero_call_root_reports_error_when_an_agent_ended_lost_or_failed(proven):
    run = episode(named(team(alpha={"harness": scripted()}, beta={"harness": scripted([Exit()])})))

    statuses = {name: row["status"] for name, row in session_of(run)["agents"].items()}
    assert statuses["beta"] in ("lost", "failed") and statuses["alpha"] == "retired"
    assert run.root.misc["agent_result"] == {"status": "error", "error": "beta ended lost or failed"}
    assert run.outcome["failure_kind"] == "harness_crash"


def test_a_cut_rollout_leaves_no_process_and_trains_nothing_of_ours(proven, wired, tmp_path):
    """Platoon cuts a rollout whose agent works on past its deadline: the session stops inside the cut, so its harness
    outlives it in no process, the agent keeps no end, so none of its calls train, and the finalizer still runs."""

    write_conformance(proven, wired)
    marker = str(tmp_path)  # in the command line of this test's harness alone

    run = episode(named(team(solver={"harness": wire("hang", marker)})), rollout={"timeout": 10})

    answer = asyncio.run(eligibility(run, "solver"))
    assert run.outcome["status"] == "timeout"
    assert not running(marker)
    assert [bar.reason for bar in answer.reasons] == ["no_end"]
    assert run.root.misc["team"]["agent"] == "solver"
    assert run.root.steps and all(step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] for step in run.root.steps)


@pytest.mark.parametrize("shape", ["tree", "mesh"])
def test_a_cut_rollout_runs_no_producer_and_stays_a_timeout(proven, wired, tmp_path, shape):
    """A rollout cut before its episode closed has no team reward to split, and no agent of it ended: the program runs
    no producer, ``propagate`` as the mesh recipe names it or any other, and the rollout stays the timeout it is."""

    write_conformance(proven, wired)
    hanging = {"harness": wire("hang", str(tmp_path))}
    teams = {
        "tree": (team(lead={**hanging, "may_spawn": ["worker"]}, worker=hanging), {"worker": PROPAGATING}),
        "mesh": (team(alpha=hanging, beta=hanging), {"alpha": PROPAGATING, "beta": {"producer": WITNESSING}}),
    }
    crew, rewards = teams[shape]
    WITNESSED.clear()

    run = episode(named(crew), harness_options={"rewards": rewards}, rollout={"timeout": 10})

    assert run.outcome["status"] == "timeout"
    assert WITNESSED == [] and not running(str(tmp_path))
    assert [row.producer for row in session_of(run)["annotations"]] == []


async def eligibility(run: Any, agent: str) -> Any:
    async with Session.attach(session_name(run)) as ended:
        return await ended.eligibility(f"@{agent}", "task")


def running(marker: str) -> list[str]:
    """The processes whose command line holds ``marker``."""

    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        with contextlib.suppress(OSError):
            if marker.encode() in cmdline.read_bytes():
                found.append(cmdline.parent.name)
    return found


def test_a_crashed_entry_harness_is_platoons_harness_crash(proven):
    run = episode(named(team(solver={"harness": scripted([Exit()])})))

    assert run.root.misc["agent_result"]["status"] == "error"
    assert (run.outcome["status"], run.outcome["failure_kind"]) == ("error", "harness_crash")
    assert run.root.misc[TRAJECTORY_INVALID_MISC_KEY]


def test_an_agent_answered_by_a_budget_stop_is_retired_and_grades_limit(proven):
    """A spawned agent past its ``max_steps`` gets the budget stop, never sampled; the bridge retires it once it is at
    rest, before the session closes, and platoon grades its run ``limit``, with reason ``budget``."""

    policy = ScriptedPolicy(wrong_code)
    run = episode(TREE, policy, agents={"child_max_steps": 1, "child_grading": "env"})

    worker = by_agent(run)["w1"]
    events = session_of(run)["events"]
    stops = [event for event in events if event.type == "model.called" and _stopped(event)]
    retired = [event.seq for event in events if event.type == "agent.ended" and event.agent == "w1"]
    closed = [event.seq for event in events if event.type == "session.closed"]
    assert len(policy.calls) == 1 and len(worker.steps) == 1
    assert [event.agent for event in stops] == ["w1"]
    assert retired < closed
    assert worker.misc["agent_result"]["status"] == "limit"
    assert worker.misc["outcome"]["reason"] == "budget"


def test_a_mesh_stopped_at_its_shared_cap_grades_with_reason_budget(proven):
    """The peers of a zero-call root are minted on its key and share its cap, so once the gateway stops one of them the
    root's run, which makes no call of its own, is a ``limit`` run, graded with reason ``budget``."""

    run = episode(PAIR, ScriptedPolicy(wrong_code), max_steps=2)

    assert sum(peer.misc["budget"]["stops"] for peer in run.trajectories[1:]) >= 1
    assert run.root.misc["budget"]["stops"] == 0 and run.root.misc["agent_result"]["status"] == "limit"
    assert (run.outcome["status"], run.outcome["reason"]) == ("scored", "budget")


def test_a_verifier_never_reports_its_childs_end(proven):
    """platoon runs a child's verifier on the child's harness and options, which for a child of a team hand that child
    off again: the member refuses a handoff it has served, so the verifier's run is an error, and the child's own last
    words are never its verdict.  The load check refuses such an entry; this holds without it."""

    worker = scripted([Call("agent_finish", {"gist": VERDICT})])
    crew = named(
        team(
            lead={"harness": scripted(spawning("w1"), [Say("done")]), "may_spawn": ["worker"]},
            worker={"harness": worker},
        )
    )

    run = episode(crew, agents={"child_grading": "verifier"})

    (verifier,) = [trajectory for trajectory in run.trajectories if trajectory.misc["agent"]["role"] == "verifier"]
    judgment = by_agent(run)["w1"].misc["subagent_reward_judgment"]
    assert verifier.misc["agent_result"]["status"] == "error"
    assert "w1's handoff was served" in verifier.misc["agent_result"]["error"]
    assert judgment["training_eligible"] is False and judgment.get("raw_response") != VERDICT


def test_minted_peers_share_their_roots_cap(proven):
    policy = ScriptedPolicy(wrong_code)
    run = episode(PAIR, policy, max_steps=2)

    peers = run.trajectories[1:]
    assert len(policy.calls) == 2 and sum(len(peer.steps) for peer in peers) == 2
    assert all(peer.misc["budget"]["max_steps"] == 2 for peer in peers)


def delegated(**rollout: Any) -> dict[str | None, pt.Trajectory]:
    """A tree whose lead spawns a worker, which trains on a producer's row of 0.25, and a scout, which names none."""

    tree = named(
        team(
            lead={
                "harness": scripted(spawning("w1") + spawning("s1", role="scout"), [Say("done")]),
                "may_spawn": ["worker", "scout"],
            },
            worker={"harness": WORKER},
            scout={"harness": WORKER},
        )
    )
    return by_agent(
        episode(tree, harness_options={"rewards": {"worker": {"producer": RATING}}}, rollout=rollout or None)
    )


@pytest.mark.parametrize(
    ("flags", "scout"),
    [({"propagate_root_success": True}, 1.0), ({"skip_subagent_reward_computation": True}, 0.0)],
)
def test_singles_finalizers_run_before_ours_under_their_flags(proven, flags, scout):
    trajectories = delegated(**flags)

    assert trajectories["s1"].reward == scout  # single's finalizer ran, under its flag
    assert trajectories["w1"].reward == 0.25  # and ours after it


def test_a_role_that_names_no_producer_keeps_what_single_gives(proven):
    trajectories = delegated(propagate_root_success=True, skip_subagent_reward_computation=True)

    assert trajectories["s1"].reward == 1.0  # zeroed, then the root's success
    assert trajectories["s1"].misc["team"]["row"] is None


def test_platoons_grades_are_annotations_of_the_rollout(proven):
    run = episode(TREE, agents={"child_grading": "env"})

    held = session_of(run)
    rows = {(row.scope, row.target): row for row in held["annotations"]}
    closed = next(f"{e.stream}@{e.version}" for e in held["events"] if e.type == "episode.closed")
    started = next(f"{e.stream}@{e.version}" for e in held["events"] if e.type == "agent.started" and e.agent == "w1")
    assert set(rows) == {("team", closed), ("local", started)}
    for row in rows.values():
        assert (row.kind, row.producer, row.producer_version) == ("reward", "platoon", pt.__version__)
        assert (row.status, row.value, row.evidence) == ("scored", 1.0, ())
    assert run.root.misc["team"]["row"] == rows["team", closed].id
    assert by_agent(run)["w1"].misc["team"]["row"] == rows["local", started].id


def _admit(event: Any) -> bool:
    return event.type == "effect.finished" and event.data["kind"] == "admit"


def _stopped(event: Any) -> bool:
    return (event.data["call_id"] or "").startswith(BUDGET_STOP_PREFIX)
