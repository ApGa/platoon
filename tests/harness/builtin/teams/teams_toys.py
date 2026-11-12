"""Teams these tests roll out, on core's ``Lockbox`` toy (``tests/testing/testing_toys.py``), or on
:class:`Ledger`, which scores what it is told, and a scripted policy.  Their agents run on ScriptedDriver,
platoon-harness's harness that needs no binary, whose model calls go through platoon's gateway and whose tool calls are
real MCP, and which launches sealed; or on :class:`Wayward`, which breaks that harness's contract where a test says.
:func:`held_to_identity` asserts what every identity fixture asserts of a rollout.

A role's harness and a trusted team are each named by ``module:attr``, so :func:`harness` and :func:`named` file what a
test builds here, as attributes of this module.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from openreward.environments import Environment, TextBlock, ToolOutput, tool
from platoon_harness import Session, Team
from platoon_harness.annotations import Status
from platoon_harness.config import Secret, forms
from platoon_harness.drivers import (
    ENVIRONMENT,
    AgentSpec,
    ChildSeen,
    Delivery,
    Driver,
    Gateway,
    Item,
    ModelCall,
    Report,
    Reporter,
    TurnStarted,
)
from platoon_harness.primitives import Eligibility, Repo
from platoon_harness.runtime import Admission
from platoon_harness.runtime.paths import HOME_VARIABLE, write_conformance
from platoon_harness.testing import ScriptedDriver, check_driver, every_turn_asks_the_model
from pydantic import BaseModel
from testing_toys import LOCKBOX

from platoon.gateway.core import BUDGET_STOP_PREFIX, AgentSession
from platoon.harness.builtin.platoon_harness.finalize import EPISODE, Record
from platoon.testing import ScriptedPolicy, last_user_message, run_episode, tool_call

_FILED: dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    try:
        return _FILED[name]
    except KeyError:
        raise AttributeError(name) from None


def harness(make: Callable[[], Driver]) -> str:
    """The ``harness`` of a role whose driver ``make`` builds, one per agent."""

    key = f"driver_{len(_FILED)}"
    _FILED[key] = make
    return f"{__name__}:{key}"


def scripted(*turns: Sequence[Any]) -> str:
    """The ``harness`` of a role whose agents take ``turns``, and then ask the model in each turn after."""

    return harness(lambda: ScriptedDriver(itertools.chain(turns, every_turn_asks_the_model())))


def named(team: Team) -> str:
    """A trusted ``module:attr`` naming ``team``, as an entry's ``team`` option names one."""

    key = f"team_{len(_FILED)}"
    _FILED[key] = team
    return f"{__name__}:{key}"


def team(**roles: Mapping[str, Any]) -> Team:
    """A trainable team of ``roles``: each answers its approvals itself."""

    return Team.model_validate({"roles": {name: {"approvals": "allow", **role} for name, role in roles.items()}})


class Wayward(ScriptedDriver):
    """ScriptedDriver as a harness that breaks its contract where a test says: it reports each model call as
    ``rewrite`` gives it, none for None and each of a list; it sends its compactions without the gateway's headers
    unless ``headers``; with ``forks`` it reports a native child of its own once it is up; with ``results`` it reports
    each tool's end as pi and OpenHands do, with the tool's text as its body, under the tool call's native id alone;
    without ``ids`` it sends no tool call's id with the call, so no op joins the model call that chose it and nothing
    confirms what a read was handed; and with ``begins`` it begins no turn after that many: a message handed to it
    after them begins none, so its agent stays on its way to a turn, as one on a harness slow to begin it is
    meanwhile.  Its conformance result, kept for a well-behaved one, earns it trainable all the same, as a check cannot
    see what a harness does later."""

    def __init__(
        self,
        turns: Iterable[Sequence[Any]] = (),
        *,
        rewrite: Callable[[ModelCall], ModelCall | list[ModelCall] | None] = lambda call: call,
        headers: bool = True,
        forks: bool = False,
        results: bool = False,
        ids: bool = True,
        begins: int | None = None,
    ) -> None:
        super().__init__(turns, name="wayward")
        self._rewrite, self._headers, self._forks, self._results = rewrite, headers, forks, results
        self._ids = ids
        self._begins, self._begun = begins, 0

    async def start(self, spec: AgentSpec, report: Reporter) -> None:
        if not self._headers and isinstance(spec.access, Gateway):
            spec = spec.model_copy(update={"access": spec.access.model_copy(update={"compaction_headers": {}})})

        async def reported(item: Report) -> None:
            self._begun += isinstance(item, TurnStarted)
            if self._results and isinstance(item, Item) and item.kind == "tool" and item.phase == "completed":
                said = "".join(block.get("text", "") for block in self.results[-1]["content"])
                item = item.model_copy(update={"body": said, "call_id": None})
            kept = self._rewrite(item) if isinstance(item, ModelCall) else item
            for each in [] if kept is None else kept if isinstance(kept, list) else [kept]:
                await report(each)

        await super().start(spec, reported)
        if self._forks:
            await report(ChildSeen(native="its-own"))

    async def deliver(self, delivery: Delivery) -> None:
        if self._begins is None or self._begun < self._begins:
            await super().deliver(delivery)

    async def _mcp(self, server: Any, method: str, params: Mapping[str, Any]) -> Any:
        if not self._ids and method == "tools/call":
            params = {key: value for key, value in params.items() if key != "_meta"}
        return await super()._mcp(server, method, params)


def wayward(*turns: Sequence[Any], **contract: Any) -> str:
    """The ``harness`` of a role whose agents take ``turns`` on a :class:`Wayward` harness, and ask the model after."""

    return harness(lambda: Wayward(itertools.chain(turns, every_turn_asks_the_model()), **contract))


ASKING = scripted()
SOLO = named(team(solver={"harness": ASKING}))
WIRED = Path(__file__).with_name("teams_wired.py")
SERVED = f"mcp__{ENVIRONMENT}__"
"""What ScriptedDriver, and so :class:`Wayward` and the wire program, shows its model before the name of each of the
environment's tools, as pi does: the environment's server, so that none collides with a tool of the harness's own."""
UNLOCK = f"{SERVED}unlock"


def session_name(run: Any) -> str:
    """The name of a rollout's session: the last part of its folder, which its root's ``misc["team"]`` names."""

    return Path(run.root.misc["team"]["session"]).name


def wire(*args: str) -> dict[str, Any]:
    """The ``harness`` of a role whose agents run the wire program (``teams_wired.py``) with ``args``."""

    return {"command": [sys.executable, str(WIRED), *args], "mode": "wire"}


def prove(root: Path) -> None:
    """Keep under ``root`` the conformance results that earn a sealed launch of ScriptedDriver and of
    :class:`Wayward` trainable."""

    for make in (ScriptedDriver, Wayward):
        checked = check_driver(lambda make=make: make(every_turn_asks_the_model()), folder=root.parent / "checks")
        write_conformance(root, asyncio.run(checked))


def unlocking(prompt: str) -> str:
    """The model of a lockbox's agent: it tries the code it was told, and once a tool answered, says it is done."""

    if last_user_message(prompt).startswith("<tool_response>"):
        return "done"
    if code := re.search(r"code is (\d+)", prompt):
        return tool_call(UNLOCK, code=code[1])
    return "nothing to do"


# The step budget of a rollout that sets none.
MAX_STEPS = 100


def rollout(team: str, policy: Any = unlocking, *, task: str = "test:1", service: str = LOCKBOX, **settings: Any):
    """One rollout of ``task``, by default the lockbox's first, whose code is 1, by ``team`` under the
    ``platoon-harness`` program, sampling ``policy``, a script function, a ``ScriptedPolicy`` or a gateway, with
    ``settings`` of the entry and the run, and by default a step budget of :data:`MAX_STEPS`, as the gateway holds the
    harness to one: a coroutine."""

    options = {"team": team, **settings.pop("harness_options", {})}
    settings.setdefault("max_steps", MAX_STEPS)
    policy = ScriptedPolicy(policy) if callable(policy) else policy
    return run_episode(
        task,
        service=service,
        harness="platoon-harness",
        program="platoon-harness",
        policy=policy,
        harness_options=options,
        **settings,
    )


def episode(team: str, policy: Any = unlocking, **settings: Any):
    """:func:`rollout`, run."""

    return asyncio.run(rollout(team, policy, **settings))


async def rating(
    session: Session, episode: str, *, judge: Gateway | None, status: Status = "scored", value: float = 0.25
) -> None:
    """A producer that gives each agent that started the row ``status``, with ``value`` when it is scored, under its
    own name, which an entry names it by."""

    record, head = await Record.read(session), await session.head()
    for start in record.started.values():
        await session.annotate(
            start,
            kind="credit",
            producer=RATING,
            producer_version="1",
            scope="local",
            status=status,
            value=value if status == "scored" else None,
            head_seq=head,
        )


RATING = f"{__name__}:rating"


async def spanning(
    session: Session,
    episode: str,
    *,
    judge: Gateway | None,
    values: Sequence[float | None] = (),
    status: Status = "scored",
) -> None:
    """A producer that gives each span of each agent that started, oldest first, a row of the value ``values`` holds in
    its place, and none for None or past them; and the agent's first start a row of ``status``, with the sum of its
    spans' values when it is scored; under its own name, which an entry names it by."""

    record, head = await Record.read(session), await session.head()
    for agent, start in record.started.items():
        answer = await session.spans(f"@{agent}")
        assert answer.cursor is None  # a toy's spans fit one page
        given = [(span.ref, value) for span, value in zip(answer.spans, values, strict=False) if value is not None]
        rows: list[tuple[str, Status, float]] = [(target, "scored", value) for target, value in given]
        for target, said, value in [*rows, (start, status, sum(value for _, value in given))]:
            await session.annotate(
                target,
                kind="credit",
                producer=SPANNING,
                producer_version="1",
                scope="local",
                status=said,
                value=value if said == "scored" else None,
                head_seq=head,
            )


SPANNING = f"{__name__}:spanning"
WITNESSED: list[dict[str, Any]] = []


async def witnessing(session: Session, episode: str, *, judge: Gateway | None) -> None:
    """A producer that writes nothing, and notes what it found: the session's state, each agent's, its working
    directory, as the session opened on it, what that holds and its commits, and the judge it was given."""

    opened = await anext(session.events(streams=["session"]))
    cwd = Path(Repo.model_validate(opened.data["repo"]).top)
    WITNESSED.append(
        {
            "state": (await session.status())["state"],
            "agents": {row["agent"]: row["state"] for row in await session.agents()},
            "cwd": cwd,
            "files": sorted(path.name for path in cwd.iterdir()),
            "commits": subprocess.run(
                ["git", "log", "--format=%an: %s"], cwd=cwd, capture_output=True, text=True, check=True
            ).stdout.splitlines(),
            "judge": judge,
        }
    )


WITNESSING = f"{__name__}:witnessing"


async def keeping(session: Session, episode: str, *, judge: Gateway | None, into: str) -> None:
    """A producer that writes nothing, and copies the rollout's working directory, as the session opened on it, to
    ``into``."""

    opened = await anext(session.events(streams=["session"]))
    shutil.copytree(Repo.model_validate(opened.data["repo"]).top, into)


KEEPING = f"{__name__}:keeping"


async def judging(session: Session, episode: str, *, judge: Gateway | None) -> None:
    """A producer that asks its judge, over the judge's HTTP endpoint, to rate each agent that started from 0 to 1, and
    gives each the rating as its row."""

    assert judge is not None
    record, head = await Record.read(session), await session.head()
    headers = {"Authorization": f"Bearer {judge.bearer.reveal()}"}
    async with aiohttp.ClientSession() as http:
        for agent, start in record.started.items():
            body = {"model": judge.model, "messages": [{"role": "user", "content": f"Rate @{agent} from 0 to 1."}]}
            async with http.post(f"{judge.origin}/v1/chat/completions", json=body, headers=headers) as answer:
                rating = float((await answer.json())["choices"][0]["message"]["content"])
            await session.annotate(
                start,
                kind="credit",
                producer=JUDGING,
                producer_version="1",
                scope="local",
                status="scored",
                value=rating,
                head_seq=head,
            )


JUDGING = f"{__name__}:judging"
LEDGER = f"python://{__name__}:Ledger"


class Score(BaseModel):
    value: float
    invalid: bool = False


class Ledger(Environment):
    """Score what an agent says: ``score`` ends the episode, or a child's part of it, with the reward ``value``, or,
    when ``invalid``, with a grade it declares invalid.  A child works on a ledger of its own."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"id": "ledger-0"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Score the work.")]

    @tool
    async def score(self, args: Score) -> ToolOutput:
        """Score the work."""

        metadata = {"invalid": True} if args.invalid else {}
        return ToolOutput(blocks=[TextBlock(text="scored")], reward=args.value, finished=True, metadata=metadata)

    def platoon_fork(self, task_spec: dict) -> Ledger:
        return Ledger(task_spec)


@dataclass(frozen=True)
class Journal:
    """What a rollout's session says of each of its agents, for :func:`held_to_identity`: its platoon ``agent_id``, its
    eligibility with every call, the result its start recorded, and each tool it called with a native id, with the call
    that asked for it."""

    agents: Mapping[str, str]
    eligibility: Mapping[str, Eligibility]
    conformance: Mapping[str, str | None]
    tools: Mapping[str, set[tuple[str, str]]]


async def journal(name: str) -> Journal:
    async with Session.attach(name) as session:
        record = await Record.read(session)
        conformance: dict[str, str | None] = {}
        tools: dict[str, set[tuple[str, str]]] = {agent: set() for agent in record.agents}
        async for event in session.events():
            if event.agent is None:
                continue
            if event.type == "agent.started":
                conformance.setdefault(event.agent, event.data["conformance"])
            elif event.type == "item.recorded" and event.data["kind"] == "tool" and event.data["call_id"]:
                if event.data["native_id"]:
                    tools[event.agent].add((event.data["native_id"], event.data["call_id"]))
        eligibility = {}
        for agent in record.agents:
            answer = await session.eligibility(f"@{agent}", EPISODE)
            calls = list(answer.calls)
            while answer.cursor is not None:
                answer = await session.eligibility(f"@{agent}", EPISODE, after=answer.cursor)
                calls += answer.calls
            eligibility[agent] = answer.model_copy(update={"calls": tuple(calls)})
    return Journal(record.agents, eligibility, conformance, tools)


def leaked(keys: Iterable[str], folders: Iterable[Path], texts: Mapping[str, str]) -> list[str]:
    """Where a form of one of ``keys`` (``platoon_harness.config.forms``) is written: each file under ``folders``, or
    each text of ``texts`` by its name; never the key."""

    held = {form for key in keys for form in forms(Secret(key))}
    found = []
    for folder in folders:
        files = folder.rglob("*") if folder.is_dir() else [folder]
        found += [str(path) for path in files if path.is_file() and any(form in path.read_bytes() for form in held)]
    return found + [name for name, text in texts.items() if any(form in text.encode() for form in held)]


def held_to_identity(
    run: Any,
    opened: Sequence[AgentSession],
    admitted: Sequence[tuple[str, Admission]],
    *,
    folders: Iterable[Path] = (),
    texts: Mapping[str, str] | None = None,
) -> None:
    """What every identity fixture asserts of a rollout through the bridge, given the platoon agents the gateway
    ``opened`` and each agent of ours with what the authority ``admitted`` it on:

    * each trajectory of ours has the platoon ``agent_id`` its agent's admission named, as the authority gave it and as
      the journal recorded it, one to one;
    * each agent's reported calls, those that never answered and budget stops aside, are its trajectory's steps, in
      order;
    * a step is ``aux`` exactly when its call was reported as a compaction;
    * a tool call's native id, where it was reported, is once in its step's tool calls;
    * each agent is ``exact`` and eligible, from the conformance result its start recorded;
    * no value an agent was admitted on holds ``/s/``;
    * no form of a key is written in the platoon-harness root (journal, annotations, logs, agent folders and the files
      drivers wrote), the rollout's files, the collection (trajectory misc and error messages), ``folders`` or
      ``texts``."""

    held = asyncio.run(journal(session_name(run)))
    ours = {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}
    ours.pop(None, None)
    platoons = {agent: trajectory.misc["agent"]["agent_id"] for agent, trajectory in ours.items()}
    given = {(agent, admission.backend["agent_id"]) for agent, admission in admitted}
    assert {(agent, agent_id) for agent, agent_id in given if agent_id in platoons.values()} == set(platoons.items())
    assert platoons == held.agents
    assert len(set(held.agents.values())) == len(held.agents)
    for agent, trajectory in ours.items():
        answer = held.eligibility[agent]
        steps = {step.misc["action_misc"]["completion_id"]: step for step in trajectory.steps}
        calls = [call for call in answer.calls if call.call_id and not call.call_id.startswith(BUDGET_STOP_PREFIX)]
        assert [call.call_id for call in calls] == list(steps), agent
        for call in calls:
            aux = steps[call.call_id].misc["action_misc"].get("kind") == "aux"
            assert aux == (call.purpose == "compaction"), (agent, call)
        for native, call_id in held.tools[agent]:
            asked = steps[call_id].misc["message"].get("tool_calls") or []
            assert [called["id"] for called in asked].count(native) == 1, (agent, native)
        assert (answer.attribution, answer.verdict, held.conformance[agent] is not None) == ("exact", "eligible", True)
    values = [value for _, admission in admitted for value in _given(admission)]
    assert values and not [value for value in values if "/s/" in value]
    texts = {"the collection": json.dumps(run.collection.to_dict(), default=str), **(texts or {})}
    keys = [agent.api_key for agent in opened]
    assert keys and leaked(keys, [Path(os.environ[HOME_VARIABLE]), run.output_dir, *folders], texts) == []


def _given(admission: Admission) -> list[str]:
    access = admission.access
    assert isinstance(access, Gateway)
    return [access.origin, access.tools or "", access.model, *access.compaction_headers.values()]
