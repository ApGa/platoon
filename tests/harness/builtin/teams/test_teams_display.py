"""What the ``platoon-harness`` program's finalizer shows of each call, for platoon's trajectory viewer: each tool
result the record ties to the call, in the order the results were recorded, and what its harness said it took before it,
by ref, with the text the record proves it was shown; a tool result the record ties to no shown call, and an offer no
shown call follows the take of, are left out and counted.  And platoon's viewer reads a team rollout's steps by them,
and each trajectory names the folder of its session, which opens as the session's history."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from platoon_harness import Refused, Session, Team
from platoon_harness.kernel import Event as Fact
from platoon_harness.primitives import CallEligibility
from platoon_harness.runtime.paths import root_of
from platoon_harness.testing import AskModel, Call
from teams_toys import UNLOCK, episode, named, prove, scripted, session_name, team, unlocking, wayward

from platoon.harness.builtin.platoon_harness.finalize import BODY_LIMIT, Display
from platoon.testing import last_user_message, tool_call
from platoon.visualization.event_sinks import emit_rollout_events
from platoon.visualization.tui import Event, PlayPauseFriendlyTree, TrajectoryTree

GOAL = "Open the box. Its code is 1."
SPAWN = Call("agent_spawn", {"name": "w1", "child_role": "worker", "goal": GOAL})
TREE = named(
    team(
        lead={"harness": scripted([SPAWN], [AskModel()]), "may_spawn": ["worker"]},
        worker={"harness": scripted([AskModel(), Call("agent_finish", {"gist": "unlocked"})])},
    )
)


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def trying_twice(prompt: str) -> str:
    """A lockbox's model: two codes in one call, the wrong one first, and once tools answered, done."""

    if last_user_message(prompt).startswith("<tool_response>"):
        return "done"
    return f"{tool_call(UNLOCK, code='2')}\n{tool_call(UNLOCK, code='1')}"


def results(step: Any) -> list[tuple[Any, ...]]:
    return [
        (result["name"], result["tool_call_id"], result["outcome"], result["output"])
        for result in step.misc["tool_results"]
    ]


def events_of(run: Any) -> list[Any]:
    async def read() -> list[Any]:
        async with Session.attach(session_name(run)) as ended:
            return [event async for event in ended.events()]

    return asyncio.run(read())


def ends(run: Any, agent: str) -> list[str]:
    """The refs of each tool's end ``agent`` reported, in order."""

    return [
        f"{event.stream}@{event.version}"
        for event in events_of(run)
        if event.agent == agent and event.type == "item.recorded" and event.data["phase"] == "completed"
    ]


def ref_of(run: Any, kind: str, agent: str) -> str:
    """The ref of the first event of ``kind`` by ``agent``."""

    return next(
        f"{event.stream}@{event.version}" for event in events_of(run) if (event.type, event.agent) == (kind, agent)
    )


def test_each_step_shows_the_tool_results_its_call_made_in_the_order_they_were_recorded(proven):
    run = episode(named(team(solver={"harness": wayward(results=True)})), trying_twice)

    first, last = run.root.steps
    asked = [called["id"] for called in first.misc["message"]["tool_calls"]]
    assert results(first) == [
        (UNLOCK, asked[0], "completed", "wrong code"),
        (UNLOCK, asked[1], "completed", "unlocked"),
    ]
    assert [result["ref"] for result in first.misc["tool_results"]] == ends(run, "solver")
    assert last.misc["tool_results"] == [] and run.root.misc["tool_results_left_out"] == 0
    assert not any({"action_events", "observation_events"} & step.misc.keys() for step in run.root.steps)


def test_a_tool_result_the_record_ties_to_no_shown_call_is_left_out_and_counted(proven):
    hidden: list[str] = []

    def hiding_the_first(call: Any) -> Any:
        if hidden:
            return call
        hidden.append(call.call_id)
        return None

    tree = episode(TREE)
    unreported = episode(named(team(solver={"harness": wayward(rewrite=hiding_the_first, results=True)})))

    lead, worker = tree.trajectories
    assert lead.misc["tool_results_left_out"] == 1  # its spawn, which no model call asked for
    assert [len(step.misc["tool_results"]) for step in worker.steps] == [1, 0]
    assert worker.misc["tool_results_left_out"] == 1  # its finish, as scripted
    first, second = unreported.root.steps
    assert not {"tool_results", "inputs"} & first.misc.keys()  # its call was never reported
    assert second.misc["tool_results"] == [] and unreported.root.misc["tool_results_left_out"] == 1


def reading_the_board(prompt: str) -> str:
    """A model that reads the board when its last message tells it to, and otherwise unlocks the box."""

    if "see the board" in last_user_message(prompt):
        return tool_call("board_get", board="notes", name="code")
    return unlocking(prompt)


def on_the_board(lead: str) -> Any:
    """A rollout of a manager-worker team, whose lead runs on ``lead``: its worker posts the code, tells the lead to
    see the board and finishes, and the lead's model reads the board."""

    worker = scripted(
        [
            AskModel(),
            Call("board_post", {"board": "notes", "name": "code", "gist": "the code is 1", "fields": {"tried": 1}}),
            Call("message_send", {"to": "@lead", "gist": "see the board"}),
            Call("agent_finish", {"gist": "unlocked"}),
        ]
    )
    roles = {
        "lead": {"harness": lead, "may_spawn": ["worker"], "approvals": "allow", "stays": False},
        "worker": {"harness": worker, "approvals": "allow"},
    }
    topology = {"manager-worker": {"manager": "lead", "workers": "worker"}}
    return episode(named(Team.model_validate({"roles": roles, "topology": topology})), reading_the_board)


def by_agent(run: Any) -> dict[str, Any]:
    return {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}


def test_each_step_shows_what_its_harness_said_it_took_before_its_call_by_ref(proven):
    """A worker's first call follows its goal and its instructions; its lead's first follows the task, its
    instructions and the worker's message, as its harness took them; the board entry it read, gist and fields as it was
    shown them, follows that read; and the lead's next turn's first call follows the worker's report."""

    run = on_the_board(wayward([SPAWN], [AskModel()], results=True))

    lead, worker = by_agent(run)["lead"], by_agent(run)["w1"]
    (task, instructions, message), *rest = [step.misc["inputs"] for step in lead.steps]
    assert task == {"ref": "inbox/lead@1", "from": "program:platoon", "text": GOAL}
    assert instructions["ref"] == "agent/lead@2" and instructions["text"].startswith("You are @lead.")
    assert message == {"ref": "inbox/lead@2", "from": "agent:w1", "text": "see the board"}
    entry = {"ref": "board/notes/code@1", "from": "agent:w1", "text": json.dumps(["the code is 1", {"tried": 1}])}
    report = {"ref": ref_of(run, "agent.finished", "w1"), "from": "agent:w1", "text": "unlocked"}
    assert rest == [[entry], [report], []]  # its read's result, "done", and next turn's unlock
    assert lead.misc["inputs_left_out"] == 0
    goal, instructions = worker.steps[0].misc["inputs"]
    assert goal == {"ref": "agent/w1@1", "from": "agent:lead", "text": GOAL}
    assert instructions["ref"] == "agent/w1@2" and instructions["text"].startswith("You are @w1.")


def test_a_step_shows_an_announce_by_the_words_it_showed(proven):
    """A board that wakes its reader with each post shows the lead an announce of the worker's post, by the post's
    ref, in words the post itself does not hold: the entry's name and gist."""

    worker = scripted(
        [
            AskModel(),
            Call("board_post", {"board": "news", "name": "code", "gist": "the code is 1"}),
            Call("agent_finish", {"gist": "unlocked"}),
        ]
    )
    roles = {
        "lead": {"harness": scripted([SPAWN]), "may_spawn": ["worker"], "approvals": "allow", "stays": False},
        "worker": {"harness": worker, "approvals": "allow"},
    }
    news = {"kind": "board", "writers": "role:worker", "readers": "role:lead", "delivery": "wake"}
    topology = {"manager-worker": {"manager": "lead", "workers": "worker"}, "channels": {"news": news}}

    run = episode(named(Team.model_validate({"roles": roles, "topology": topology})))

    shown = [item for step in by_agent(run)["lead"].steps for item in step.misc["inputs"]]
    assert {"ref": "board/news/code@1", "from": "agent:w1", "text": "code: the code is 1"} in shown


def test_an_offer_its_harness_never_said_it_took_is_left_out_and_counted(proven):
    """A harness that sends no tool call's id with its call leaves nothing to confirm the board entry its read was
    handed."""

    run = on_the_board(wayward([SPAWN], [AskModel()], ids=False))

    lead = by_agent(run)["lead"]
    assert [[item["ref"] for item in step.misc["inputs"]] for step in lead.steps][1] == []
    assert lead.misc["inputs_left_out"] == 1


class Journal:
    """A session's record, as the finalizer's display reads it: its events; its bodies, each refused past a limit as
    platoon-harness refuses one; and what each offer showed, of each item the gist, body or line of a body the record
    holds whose hash is the item's digest."""

    def __init__(self, *facts: Fact, bodies: Sequence[bytes] = ()) -> None:
        self.facts, self.bodies = facts, {f"sha256:{sha256(body).hexdigest()}": body for body in bodies}

    async def events(self) -> Any:
        for fact in self.facts:
            yield fact

    async def shown(self, offer: str) -> tuple[str | None, ...]:
        (made,) = [fact for fact in self.facts if f"{fact.stream}@{fact.version}" == offer]
        bodies = [body.decode() for body in self.bodies.values()]
        held = [fact.gist for fact in self.facts] + bodies + [line for body in bodies for line in body.splitlines(True)]
        texts = {sha256(text.encode()).hexdigest(): text for text in held}
        return tuple(texts.get(item["digest"]) for item in made.data["items"])

    async def get_body(self, ref: str, *, limit: int | None = None) -> bytes:
        if ref not in self.bodies:
            raise Refused("not_found", "this record holds no such body")
        if limit is not None and len(self.bodies[ref]) > limit:
            raise Refused("unsupported", f"this body is past the {limit} bytes this reader takes")
        return self.bodies[ref]


def fact(seq: int, kind: str, stream: str, data: Any, *, actor: str = "agent:a", gist: str = "") -> Fact:
    """Event ``seq`` of agent ``a``'s record, at version ``seq`` of its stream."""

    return Fact.model_validate(
        {
            "v": 3,
            "seq": seq,
            "stream": stream,
            "version": seq,
            "type": kind,
            "actor": actor,
            "agent": "a",
            "turn": None,
            "context_version": None,
            "episode": None,
            "topology_rev": 0,
            "correlation": None,
            "cause": (),
            "based_on": (),
            "native": {},
            "model": None,
            "wall_time": 0.0,
            "gist": gist,
            "data": data,
            "outcome": None,
        }
    )


def called(seq: int, call_id: str) -> tuple[Fact, CallEligibility]:
    """A model call agent ``a`` reported, and as its eligibility lists it."""

    ref = f"turn/a@{seq}"
    listed = CallEligibility(call=ref, call_id=call_id, purpose=None, attribution=None, verdict="eligible", reasons=())
    return fact(seq, "model.called", "turn/a", {"call_id": call_id}), listed


def message(seq: int, text: str) -> Fact:
    return fact(seq, "message.sent", "inbox/a", {}, actor="agent:b", gist=text)


def offered(seq: int, *items: tuple[str, str]) -> Fact:
    """An offer of messages to agent ``a``, each by its ref and the text it was shown."""

    shown = [{"ref": ref, "form": "gist", "digest": sha256(text.encode()).hexdigest()} for ref, text in items]
    return fact(seq, "context.offered", "context/a", {"items": shown, "how": "steer", "boundary": "tool"})


def took(seq: int, offer: int) -> Fact:
    return fact(seq, "context.delivered", "context/a", {"input": offer})


def tool(seq: int, phase: str, native_id: str, call_id: str | None, body: bytes | None = None) -> Fact:
    data = {"kind": "tool", "phase": phase, "native_id": native_id, "call_id": call_id, "outcome": None, "body": None}
    if phase == "completed":
        data |= {"outcome": "completed", "body": None if body is None else f"sha256:{sha256(body).hexdigest()}"}
    return fact(seq, "item.recorded", "turn/a", data)


def display(*facts: Fact | tuple[Fact, CallEligibility], bodies: Sequence[bytes] = ()) -> Display:
    calls = [each[1] for each in facts if isinstance(each, tuple)]
    record = Journal(*(each[0] if isinstance(each, tuple) else each for each in facts), bodies=bodies)
    return asyncio.run(Display.read(record, {"a": calls}))["a"]


def test_an_offer_taken_during_a_call_shows_on_the_call_after_its_take():
    """pi reports a call once it has ended, so a steer offered while it ran, and taken after, was in no request of
    it; and one taken after the agent's last call shows on none."""

    shown = display(
        message(1, "first"),
        offered(2, ("inbox/a@1", "first")),
        called(3, "A"),
        took(4, 2),
        called(5, "B"),
        message(6, "last"),
        offered(7, ("inbox/a@6", "last")),
        took(8, 7),
    )

    assert shown.inputs == {"A": [], "B": [{"ref": "inbox/a@1", "from": "agent:b", "text": "first"}]}
    assert shown.inputs_held == 2


def test_an_offer_never_taken_shows_on_no_call():
    shown = display(message(1, "unread"), offered(2, ("inbox/a@1", "unread")), called(3, "A"), called(4, "B"))

    assert shown.inputs == {"A": [], "B": []} and shown.inputs_held == 1


def test_a_tool_result_whose_native_id_two_calls_started_is_left_out():
    """A start that names no call ties nothing, so the other start's call takes a result that names only its native
    id; two starts naming two calls tie it to neither."""

    shown = display(
        called(1, "A"),
        tool(2, "started", "n", "A"),
        tool(3, "started", "n", None),
        tool(4, "completed", "n", None),
        called(5, "B"),
        tool(6, "started", "m", "B"),
        called(7, "C"),
        tool(8, "started", "m", "C"),
        tool(9, "completed", "m", None),
    )

    assert {call: [result["ref"] for result in results] for call, results in shown.results.items()} == {
        "A": ["turn/a@4"]
    }
    assert shown.results_held == 2


def test_an_input_whose_shown_text_no_hash_proves_has_none():
    """The text an offer's digest names matches neither the gist the event holds nor a body the record keeps, so
    platoon-harness shows none."""

    body = {"ref": "inbox/a@1", "form": "body", "digest": sha256(b"a body").hexdigest()}
    gist = {"ref": "inbox/a@1", "form": "gist", "digest": sha256(b"what it was shown").hexdigest()}
    shown = display(
        message(1, "its gist"),
        fact(2, "context.offered", "context/a", {"items": [body, gist], "how": "wake", "boundary": "turn"}),
        took(3, 2),
        called(4, "A"),
    )

    assert shown.inputs["A"] == [{"ref": "inbox/a@1", "from": "agent:b", "text": None}] * 2


def test_an_input_of_lines_of_a_body_shows_them_and_who_made_the_event_they_are_of():
    """An item that shows lines of a body names them by its event's ref and a fragment, so its maker is the event's."""

    lines = {"ref": "inbox/a@1#L2-2", "form": "body", "digest": sha256(b"line two\n").hexdigest()}
    shown = display(
        message(1, "its gist"),
        fact(2, "context.offered", "context/a", {"items": [lines], "how": "wake", "boundary": "turn"}),
        took(3, 2),
        called(4, "A"),
        bodies=[b"line one\nline two\n"],
    )

    assert shown.inputs["A"] == [{"ref": "inbox/a@1#L2-2", "from": "agent:b", "text": "line two\n"}]


def test_a_body_past_the_limit_is_named_by_its_ref_alone():
    short, long = b"unlocked", b"x" * (BODY_LIMIT + 1)
    shown = display(
        called(1, "A"),
        tool(2, "completed", "n", "A", short),
        tool(3, "completed", "m", "A", long),
        bodies=[short, long],
    )

    assert [(result["output"], result["ref"]) for result in shown.results["A"]] == [
        ("unlocked", "turn/a@2"),
        (None, "turn/a@3"),
    ]


def test_a_team_rollout_reads_in_platoons_viewer(proven, tmp_path):
    run = episode(named(team(solver={"harness": wayward(results=True)})), trying_twice)

    path = emit_rollout_events(run.collection.to_dict(), tmp_path / "events.jsonl", process_id=1)
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("collections")
    for line in path.read_text().splitlines():
        record = json.loads(line)
        tree.ingest(Event(type=record["type"], data=record))

    labels = [str(node.label) for node in tree.traj_steps_nodes[run.root.id].children]
    assert labels[0] == f"step 0: tools: {UNLOCK} x2; inputs: 2"  # its instructions, and the task


def test_each_trajectory_names_its_sessions_folder_which_opens_as_its_history(proven):
    run = episode(TREE)

    async def opened(folder: Path) -> tuple[str, list[str]]:
        async with Session.attach(folder.name, root=root_of(folder), view=True) as ended:
            return (await ended.status())["state"], [row["agent"] for row in await ended.agents()]

    (folder,) = {Path(trajectory.misc["team"]["session"]) for trajectory in run.trajectories}
    assert folder.is_absolute() and folder == proven / "sessions" / run.root.misc["agent"]["agent_id"]
    assert asyncio.run(opened(folder)) == ("closed", ["lead", "w1"])
