"""What of a team's rollout trains: :func:`finalize`, the ``platoon-harness`` program's finalizer, which reads the ended
session beside the trajectory collection, and is a pure function of the two.  It runs after ``single``'s finalizers.

1. **Join.**  A trajectory is the agent's of ours whose admission named its platoon ``agent_id``.
2. **Nothing of ours.**  A trajectory other than the root's with no turn of ours behind it, a mint whose answer was
   lost, an agent never woken or one of platoon's verifiers, is excluded whole.
3. **Calls and steps.**  A step trains only when its ``completion_id`` is a call its agent reported once, for its turn
   or for ``compaction``, and an ``aux`` step only when that call is a compaction: so nothing joins by order, and a call
   of unknown purpose, or reported twice, never trains.  Each agent's calls with no step, steps with no call,
   compactions the gateway counted as policy calls and calls reported twice are logged as a :class:`Mismatch`.
4. **Rewards.**  An agent whose role names a producer, the root aside, which trains on the environment's grade, takes
   the one row that producer wrote on its first start: a scored row sets its reward, where the default processor and
   ``platoon-delegation`` read it; any other, or none, sets the marker of its status, ``trajectory_invalid`` for a
   missing one.  Its trajectory names every span of the agent's, scored or not, oldest first, in ``misc["spans"]``,
   which ``credit: keyed`` reads: each the ids of its calls the agent reported once that have a step, and the value of
   the one row that producer wrote on the span's name, the ``model.called`` of its last call, when it is scored, None
   otherwise.  So a span with no step, or no scored row, trains nothing, and the agent's own row still decides whether
   any of it trains.  Any other agent keeps what ``single``'s finalizers gave it, and platoon's own markers stand.
5. **Eligibility.**  An ineligible call's step is excluded; so is every step of an ineligible root, which keeps the
   task reward; an ineligible child is excluded from policy training, and keeps its reward.
6. **Its place in the record.**  Each trajectory's ``misc["team"]`` names its session's folder, its agent and the row
   it trains on, None for none, and for an agent of a producer's role, each span's row, in ``misc["spans"]``'s order,
   None for none.  The zero-call root names no agent, as it is the bridge's and none of the team's, and a producer
   role's agent with no row names none beside its ``trajectory_invalid``.
7. **Policy.**  The rollout fails when the root samples another policy than an agent of the entry agent's role.
8. **Display.**  Each step whose call its agent reported once holds, for platoon's viewer, ``tool_results``: each tool
   result the record ties to that call, by the call id the result names or else the one call its tool call's start
   names, in the order the results were recorded; and ``inputs``: each item of each offer its harness said it took
   after its previous call was reported and before this one was, by the item's ref.  Each trajectory of an agent of
   ours counts the tool results and the offered items it shows on no step, an offer never taken among them, as
   ``tool_results_left_out`` and ``inputs_left_out``.  Nothing is presented as OpenHands events.  ``propagate@4``
   reads a span's takes in the same window, per call, bounded too by the agent's latest start or resume, and leaves
   out the agent's own events.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from platoon_harness import Refused, Session
from platoon_harness.annotations import Annotation
from platoon_harness.kernel import Event
from platoon_harness.primitives import CallEligibility, Eligibility, Span
from platoon_harness.runtime.paths import find_root, session_folder

import platoon as pt
from platoon.gateway.core import BUDGET_STOP_PREFIX
from platoon.utils.trajectory_status import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_INVALID_MISC_KEY,
    TRAJECTORY_TIMED_OUT_MISC_KEY,
)

logger = logging.getLogger(__name__)

# The episode each rollout's session opens, for its task.
EPISODE = "task"
# Who writes platoon's grades into the session, at platoon's version.
GRADER = "platoon"
# The marker of each status a row other than a scored one has.
MARKERS = {
    "invalid": TRAJECTORY_INVALID_MISC_KEY,
    "error": TRAJECTORY_INVALID_MISC_KEY,
    "timeout": TRAJECTORY_TIMED_OUT_MISC_KEY,
    "cancelled": TRAJECTORY_CANCELLED_MISC_KEY,
}
# The most bytes of a body a step holds for display: a longer one is named by its ref alone.
BODY_LIMIT = 64 * 1024


@dataclass(frozen=True)
class Record:
    """What a rollout's session says of its agents, read once: each agent's platoon ``agent_id``, as its admission
    named it, the ref of each agent's first start, the agents a turn of whose began, and the ref of the episode's
    close, None while it is open."""

    agents: Mapping[str, str]
    started: Mapping[str, str]
    turned: frozenset[str]
    closed: str | None

    @classmethod
    async def read(cls, session: Session) -> Record:
        agents: dict[str, str] = {}
        started: dict[str, str] = {}
        turned: set[str] = set()
        closed = None
        async for event in session.events():
            ref = f"{event.stream}@{event.version}"
            if event.type == "episode.closed" and event.stream == f"episode/{EPISODE}":
                closed = ref
            elif event.agent is None:
                continue
            elif event.type == "effect.finished" and event.data["kind"] == "admit" and event.data["result"]:
                agents[event.agent] = event.data["result"]["agent_id"]
            elif event.type == "agent.started":
                started.setdefault(event.agent, ref)
            elif event.type == "turn.started":
                turned.add(event.agent)
        return cls(agents=agents, started=started, turned=frozenset(turned), closed=closed)


@dataclass(frozen=True)
class Display:
    """What a rollout's session shows of one agent's calls, for platoon's viewer, read once: for each call it reported
    once, the tool results the record ties to it, in the order they were recorded, and the items of each offer its
    harness said it took after its previous call was reported and before this one was; and how many tool results and
    offered items it has in all."""

    results: Mapping[str, list[dict[str, Any]]]
    inputs: Mapping[str, list[dict[str, Any]]]
    results_held: int
    inputs_held: int

    @classmethod
    async def read(cls, session: Session, calls: Mapping[str, Sequence[CallEligibility]]) -> dict[str, Display]:
        """Each agent's, from the calls its eligibility lists."""

        events: dict[str, Event] = {}
        offered: dict[str, dict[int, Event]] = defaultdict(dict)
        taken: dict[str, dict[int, int]] = defaultdict(dict)
        tools: dict[str, list[tuple[str, Event]]] = defaultdict(list)
        async for event in session.events():
            ref = f"{event.stream}@{event.version}"
            events[ref] = event
            agent = event.agent
            if agent is None or agent not in calls:
                continue
            if event.type == "context.offered" and event.stream == f"context/{agent}":
                offered[agent][event.seq] = event
            elif event.type == "context.delivered" and event.stream == f"context/{agent}":
                taken[agent].setdefault(event.data["input"], event.seq)  # its first confirmation is the one
            elif event.type == "item.recorded" and event.data["kind"] == "tool":
                tools[agent].append((ref, event))
        return {
            agent: await cls._of(session, events, reported, offered[agent], taken[agent], tools[agent])
            for agent, reported in calls.items()
        }

    @classmethod
    async def _of(
        cls,
        session: Session,
        events: Mapping[str, Event],
        reported: Sequence[CallEligibility],
        offered: Mapping[int, Event],
        taken: Mapping[int, int],
        tools: Sequence[tuple[str, Event]],
    ) -> Display:
        counts = Counter(call.call_id for call in reported if call.call_id is not None)
        once = {call.call_id for call in reported if call.call_id is not None and counts[call.call_id] == 1}
        starts: dict[str, set[str]] = defaultdict(set)
        for _, item in tools:  # a start that names no call ties nothing
            if item.data["phase"] == "started" and item.data["native_id"] and item.data["call_id"]:
                starts[item.data["native_id"]].add(item.data["call_id"])
        ended = [(ref, item) for ref, item in tools if item.data["phase"] == "completed"]
        results: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ref, item in ended:
            native, call = item.data["native_id"], item.data["call_id"]
            if call is None and native is not None and len(starts[native]) == 1:
                (call,) = starts[native]
            if call in once:
                output = await _text(session, item.data["body"])
                results[call].append(
                    {"tool_call_id": native, "outcome": item.data["outcome"], "output": output, "ref": ref}
                )
        order = sorted(reported, key=lambda call: events[call.call].seq)
        seqs = [events[call.call].seq for call in order]
        inputs: dict[str, list[dict[str, Any]]] = {call: [] for call in once}
        for offer, at in sorted(taken.items(), key=lambda pair: pair[1]):
            following = bisect_right(seqs, at)
            call = order[following].call_id if following < len(order) else None
            if call in inputs and offer in offered:
                inputs[call] += await _inputs(session, events, offered[offer])
        held = sum(len(offer.data["items"]) for offer in offered.values())
        return cls(results=results, inputs=inputs, results_held=len(ended), inputs_held=held)


async def _text(session: Session, body: str | None) -> str | None:
    """A stored body as text, or None for none, one no longer held, one past :data:`BODY_LIMIT` bytes and one that is
    no text."""

    if body is None:
        return None
    try:
        return (await session.get_body(body, limit=BODY_LIMIT)).decode()
    except (Refused, UnicodeDecodeError):
        return None


async def _inputs(session: Session, events: Mapping[str, Event], offer: Event) -> list[dict[str, Any]]:
    """Each item of an offer: its ref, who made the event it names, lines of its body too, and the text it showed, as
    platoon-harness's ``Session.shown`` proves it by the item's digest, None where it does not."""

    inputs = []
    for item, text in zip(offer.data["items"], await session.shown(f"{offer.stream}@{offer.version}"), strict=True):
        event = events.get(item["ref"].partition("#")[0])
        inputs.append({"ref": item["ref"], "from": None if event is None else event.actor, "text": text})
    return inputs


@dataclass(frozen=True)
class Mismatch:
    """Where an agent's record and its trajectory part: the calls it reported with no step, budget stops aside, which
    nothing samples; the steps no call it reported names, an ``aux`` one too that no reported compaction names; the
    compactions it reported that the gateway counted as policy calls, as their headers were not sent; and the calls it
    reported more than once, whose purpose and verdict are then unknown."""

    agent: str | None
    calls_without_step: tuple[str, ...]
    steps_without_call: tuple[str, ...]
    compactions_counted: tuple[str, ...]
    calls_reported_twice: tuple[str, ...]


async def finalize(
    collection: pt.TrajectoryCollection, name: str, producers: Mapping[str, str]
) -> tuple[Mismatch, ...]:
    """Mark what of ``collection`` trains, from its ended session ``name`` (see the module docstring), where
    ``producers`` names the producer each role trains on; return each agent's mismatch, which it logs."""

    trajectories = list(collection.trajectories.values())
    folder = str(session_folder(find_root(), name).absolute())
    async with Session.attach(name) as session:
        record = await Record.read(session)
        roles = {row["agent"]: row["role"] for row in await session.agents()}
        rows = await session.annotations()
        by_platoon = {platoon_id: agent for agent, platoon_id in record.agents.items()}
        if len(by_platoon) != len(record.agents):
            raise ValueError("two agents of the session were admitted as one platoon agent")
        agents = [by_platoon.get(trajectory.misc["agent"]["agent_id"]) for trajectory in trajectories]
        answers: dict[str | None, Eligibility] = {
            agent: await _eligibility(session, agent) for agent in agents if agent in record.turned
        }
        calls = {agent: answers[agent].calls if agent in answers else () for agent in agents if agent is not None}
        display = await Display.read(session, calls)
        spans = {
            agent: await _spans(session, agent)
            for agent in agents[1:]
            if agent is not None and roles[agent] in producers
        }
    _check_policies(trajectories, agents, roles)
    mismatches = []
    for index, (trajectory, agent) in enumerate(zip(trajectories, agents, strict=True)):
        root, eligibility = index == 0, answers.get(agent)
        if mismatch := _join(trajectory, agent, eligibility):
            mismatches.append(mismatch)
            logger.warning("%s", mismatch)
        if not root and eligibility is None:
            trajectory.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
        elif eligibility is not None and eligibility.verdict == "ineligible":
            if root:
                for step in trajectory.steps:
                    step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
            else:
                trajectory.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] = True
        start = None if agent is None else record.started.get(agent)
        named: dict[str, list[int | None]] = {}
        if root:
            row = _row(rows, record.closed, GRADER, "team")
        elif agent is not None and roles[agent] in producers:
            producer = producers[roles[agent]]
            row = _producer_row(rows, start, producer, f"{agent}'s start")
            _reward(trajectory, row)
            named["spans"] = _name_spans(trajectory, agent, spans[agent], eligibility, rows, producer)
        else:
            row = _row(rows, start, GRADER, "local")
        trajectory.misc["team"] = {"session": folder, "agent": agent, "row": None if row is None else row.id, **named}
        if agent is not None:
            _show(trajectory, display[agent])
    return tuple(mismatches)


def _show(trajectory: pt.Trajectory, display: Display) -> None:
    """Put on each step whose call its agent reported once the tool results and inputs ``display`` holds for it, each
    result named as its step's message named that tool call, and count the results and inputs no step shows."""

    results_left_out, inputs_left_out = display.results_held, display.inputs_held
    for step in trajectory.steps:
        call_id = step.misc["action_misc"]["completion_id"]
        if call_id not in display.inputs:
            continue
        asked = (step.misc.get("message") or {}).get("tool_calls") or []
        names = {called["id"]: called["function"]["name"] for called in asked}
        step.misc["tool_results"] = [
            {"name": names.get(result["tool_call_id"]), **result} for result in display.results.get(call_id, [])
        ]
        step.misc["inputs"] = display.inputs[call_id]
        results_left_out -= len(step.misc["tool_results"])
        inputs_left_out -= len(step.misc["inputs"])
    trajectory.misc["tool_results_left_out"] = results_left_out
    trajectory.misc["inputs_left_out"] = inputs_left_out


def _check_policies(
    trajectories: Sequence[pt.Trajectory], agents: Sequence[str | None], roles: Mapping[str, str]
) -> None:
    """Refuse a rollout whose root sampled another policy than an agent of the entry agent's role did."""

    entry = agents[0]
    if entry is None:
        return
    policy = trajectories[0].misc["agent"].get("policy")
    for trajectory, agent in zip(trajectories[1:], agents[1:], strict=True):
        if agent is not None and roles[agent] == roles[entry] and trajectory.misc["agent"].get("policy") != policy:
            raise ValueError(
                f"the root, {entry}, sampled policy {policy!r}, and {agent} of its role `{roles[entry]}` sampled "
                f"{trajectory.misc['agent'].get('policy')!r}"
            )


async def _eligibility(session: Session, agent: str) -> Eligibility:
    """An agent's eligibility in the episode, with every one of its calls."""

    answer = await session.eligibility(f"@{agent}", EPISODE)
    calls = list(answer.calls)
    while answer.cursor is not None:
        answer = await session.eligibility(f"@{agent}", EPISODE, after=answer.cursor)
        calls += answer.calls
    return answer.model_copy(update={"calls": tuple(calls)})


async def _spans(session: Session, agent: str) -> tuple[Span, ...]:
    """Every span of an agent, oldest first."""

    answer = await session.spans(f"@{agent}")
    spans = list(answer.spans)
    while answer.cursor is not None:
        answer = await session.spans(f"@{agent}", after=answer.cursor)
        spans += answer.spans
    return tuple(spans)


def _join(trajectory: pt.Trajectory, agent: str | None, eligibility: Eligibility | None) -> Mismatch | None:
    """Exclude each step that does not join an eligible call its agent reported once (an ``aux`` one, a compaction),
    and say where the two part."""

    reported = eligibility.calls if eligibility is not None else ()
    counts = Counter(call.call_id for call in reported if call.call_id is not None)
    twice = [call_id for call_id, count in counts.items() if count > 1]
    calls: dict[str, CallEligibility] = {
        call.call_id: call for call in reported if call.call_id is not None and counts[call.call_id] == 1
    }
    stepped: set[str] = set()
    without_call: list[str] = []
    counted: list[str] = []
    for step in trajectory.steps:
        action = step.misc["action_misc"]
        call_id, aux = action["completion_id"], action.get("kind") == "aux"
        stepped.add(call_id)
        if call_id in twice:  # which of its reports holds is unknown
            step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
            continue
        call = calls.get(call_id)
        if call is None or (aux and call.purpose != "compaction"):
            without_call.append(call_id)
        elif call.purpose == "compaction" and not aux:
            counted.append(call_id)
        if call is None or call_id in without_call or call.verdict == "ineligible":
            step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
    without_step = [call for call in calls if call not in stepped and not call.startswith(BUDGET_STOP_PREFIX)]
    if not (without_step or without_call or counted or twice):
        return None
    return Mismatch(agent, tuple(without_step), tuple(without_call), tuple(counted), tuple(twice))


def _row(rows: Sequence[Annotation], target: str | None, producer: str, scope: str) -> Annotation | None:
    """The row ``producer`` wrote on ``target`` in ``scope``, or None."""

    return next((row for row in rows if (row.target, row.producer, row.scope) == (target, producer, scope)), None)


def _producer_row(rows: Sequence[Annotation], target: str | None, producer: str, named: str) -> Annotation | None:
    """The one row ``producer`` wrote on ``target``, ``named`` for an error, or None; several refuse the rollout."""

    found = [row for row in rows if row.target == target and row.producer == producer]
    if len(found) > 1:
        raise ValueError(f"{producer} wrote {len(found)} rows on {named}, and its role trains on one")
    return found[0] if found else None


def _name_spans(
    trajectory: pt.Trajectory,
    agent: str,
    spans: Sequence[Span],
    eligibility: Eligibility | None,
    rows: Sequence[Annotation],
    producer: str,
) -> list[int | None]:
    """Name each of an agent's ``spans`` in its trajectory's ``misc["spans"]``, in their order: the ids of the calls on
    its turn stream from the span's first call to its last that the agent reported once and that have a step, and the
    value of the one row ``producer`` wrote on the span's name when that row is scored, None otherwise; and return each
    span's row id, None for none."""

    reported = eligibility.calls if eligibility is not None else ()
    counts = Counter(call.call_id for call in reported if call.call_id is not None)
    stepped = {step.misc["action_misc"]["completion_id"] for step in trajectory.steps}
    named, ids = [], []
    for span in spans:
        first, last = _version(span.first), _version(span.ref)
        calls = [
            call.call_id
            for call in reported
            if call.call_id is not None
            and counts[call.call_id] == 1
            and call.call_id in stepped
            and first <= _version(call.call) <= last
        ]
        row = _producer_row(rows, span.ref, producer, f"{agent}'s span {span.ref}")
        reward = row.value if row is not None and row.status == "scored" else None
        named.append({"calls": calls, "reward": reward})
        ids.append(None if row is None else row.id)
    trajectory.misc["spans"] = named
    return ids


def _version(ref: str) -> int:
    """The version a ref names on its stream."""

    return int(ref.rpartition("@")[2])


def _reward(trajectory: pt.Trajectory, row: Annotation | None) -> None:
    """Put a producer's row where a reward is read: a scored one's value as the trajectory's reward and its last step's
    ``reward/success``, any other status as its marker, and no row as ``trajectory_invalid``."""

    if row is None or row.status != "scored":
        trajectory.misc[MARKERS["invalid" if row is None else row.status]] = True
        return
    assert row.value is not None
    trajectory.reward = row.value
    if trajectory.steps:
        trajectory.steps[-1].misc.setdefault("reward_misc", {})["reward/success"] = row.value
