"""How the ``platoon-harness`` program and harness run a team of agents that platoon-harness runs, rolled out as one
platoon episode.

An entry runs it with ``harness: platoon-harness`` and ``program: platoon-harness``.  Its ``harness_options`` hold
``team``, a team file (``.toml``) or a trusted ``module:attr`` naming a :class:`~platoon_harness.Team`, and,
optionally, ``rewards``, the producer each role trains on, with what it runs with::

    harness: platoon-harness
    program: platoon-harness
    harness_options:
      team: crew.toml
      rewards:
        worker: {producer: "platoon_harness.credit:propagate", retained: 0.5}

* **The program** adds the finalizers ``single`` adds, under the same settings and in the same order, and then its own
  (:mod:`.finalize`).  It makes the rollout's working directory, an empty repository with one empty commit, and runs
  the root.  What kept the team's session from starting it raises, so that it is the rollout's ``error`` outcome and
  never a score.  Then it grades, writes platoon's grades into the ended session as annotations, runs each producer
  ``rewards`` names once, on an episode that closed, and removes the working directory, whatever happened.
* **The harness** is the root when its options carry no handoff.  It starts the team's session as the program
  ``platoon``, trainable, with :class:`~.authority.PlatoonAuthority`, opens the episode, sends the instruction to
  every agent the session starts, retires each agent at rest once the gateway has answered it with a budget stop, and
  waits for the session to settle.  Then it ends every agent, deepest first, one it cancels with no
  turn of its under way reading as retired and one that has ended of itself meanwhile as it ended, closes the episode
  on what each starting agent said last, closes the session if it is still open, and reports the entry agent's end,
  or under the zero-call root of a team that starts several, ``error`` exactly when an agent ended lost or failed.
  With a handoff it is a member: it hands its context to its agent's admission and reports that agent's end, once; a
  run under a handoff it served, such as a verifier platoon runs with its child's harness and options, it refuses.
* **The load check**, :func:`check_entry`, refuses what the entry and its team decide, before anything runs; what the
  machine decides, the session's start refuses.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import os
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from platoon_harness import ConfigError, Refused, Session, Team
from platoon_harness.config import load_team
from platoon_harness.drivers import DriverError
from platoon_harness.kernel import GIST_LIMIT
from platoon_harness.runtime import AgentOutcome, Gateway
from platoon_harness.runtime.git import git
from platoon_harness.team import Command, trainable_problems

import platoon as pt
from platoon.extensions import ExtensionError
from platoon.gateway.core import BUDGET_STOP_PREFIX
from platoon.rewards import zero_subagent_rewards
from platoon.train.credit.registry import resolve_credit_estimator
from platoon.utils.scratch import mkdtemp
from platoon.utils.subagent_rewards import propagate_root_success

from . import NAME, team_harness
from .authority import BRIDGE, HANDOFF, PlatoonAuthority, normalize, result_of
from .finalize import EPISODE, GRADER, Record, finalize

# Who acts in each rollout's session.
PRINCIPAL = "program:platoon"
# Role names platoon gives a meaning of its own: the root's, a verifier's and a judge's.
RESERVED_ROLES = ("policy", "verifier", "judge")
# The agents settings a mint cannot honour, so that a team that starts several agents leaves them at their defaults.
UNMINTABLE = ("fork", "child_grading", "child_access", "max_depth", "child_max_steps", "child_timeout_s", "budget")
# The reward processors that read a reward where a producer's row lands: the default and platoon-delegation.
REWARD_PROCESSORS = (None, "platoon-delegation")


@dataclass(eq=False)
class Bridge:
    """What the program hands its root harness, and the root its members: the team, the rollout's working
    directory and the name of its session, the root's platoon ``agent_id``; once the root starts the session, its
    authority and the session; and what kept the session from starting, if anything did."""

    team: Team
    cwd: Path
    name: str
    authority: PlatoonAuthority | None = None
    session: Session | None = None
    refusal: Exception | None = None


def load(team: object) -> Team:
    """The team a ``team`` option names: a ``.toml`` team file, loaded over the host's user config and the
    ``platoon-harness.toml`` beside it, with ``PLATOON_HARNESS_CONFIG`` over it, and trusted as the config that names it
    is; or a ``module:attr`` naming a ``Team``, as it is."""

    if isinstance(team, str) and team.endswith(".toml"):
        return load_team(Path(team), cwd=Path(team).parent, trust_project=True)
    if not isinstance(team, str) or ":" not in team:
        raise ValueError("name a team file (`.toml`) or a `module:attr` naming a Team")
    found = _imported(team)
    if not isinstance(found, Team):
        raise ValueError(f"`{team}` names a {type(found).__name__}, not a Team")
    return found


def entry_of(team: Team) -> str | None:
    """The agent that runs on the root's own endpoint: the one agent a team starts, or None for a team that starts
    several, which runs under a zero-call root."""

    starting = team.recipients()
    return starting[0].removeprefix("@") if len(starting) == 1 else None


def _imported(reference: str) -> Any:
    """The object a trusted ``module:attr`` names."""

    module, _, attr = reference.partition(":")
    if not module or not attr:
        raise ValueError(f"`{reference}` is no `module:attr`")
    return functools.reduce(getattr, attr.split("."), importlib.import_module(module))


def _rewards(options: Mapping[str, Any]) -> dict[str, tuple[str, dict[str, Any]]]:
    """``harness_options.rewards`` as each role's producer and what it runs with."""

    rewards = options.get("rewards", {})
    if not isinstance(rewards, Mapping):
        raise ValueError("rewards maps each role to {producer: <module:attr>, ...what it runs with}")
    found = {}
    for role, entry in rewards.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("producer"), str):
            raise ValueError(f"rewards.{role} is {{producer: <module:attr>, ...what it runs with}}")
        found[role] = (entry["producer"], {key: value for key, value in entry.items() if key != "producer"})
    return found


# --- the load check ---------------------------------------------------------------------------------------------


def check_entry(options: Mapping[str, Any], ctx: pt.CheckContext) -> None:
    """Refuse an entry the program cannot run in ``ctx.phase``, naming every problem, from the entry and its team
    alone: nothing runs, and what the machine decides is the session's start's to refuse."""

    if problems := list(_problems(ctx.environment, ctx.phase)):
        raise ValueError("; ".join(problems))


def _problems(environment: Any, phase: str) -> Iterator[str]:
    harness = environment.harness
    if not isinstance(harness, str) or not _is_ours(harness):
        yield f"harness: program {NAME} runs harness {NAME}, and this entry names {harness!r}"
    options = environment.harness_options
    for key in sorted(options.keys() - {"team", "rewards"}):
        yield f"harness_options.{key}: harness {NAME} takes `team` and `rewards` alone"
    try:
        team = load(options.get("team"))
    except ConfigError as error:
        yield from (f"harness_options.team: {problem}" for problem in error.problems)
        return
    except Exception as error:
        yield f"harness_options.team: {error}"
        return
    yield from (f"team {problem}" for problem in trainable_problems(team))
    if team.topology.until == "program":
        yield (
            "team topology.until: `program` is left to the program that owns the session, and this program closes "
            "a rollout's session once it settles, which one left to `program` does only when its budget runs out: "
            "leave `until` at `settled`, or name a role"
        )
    if not team.recipients():
        yield "harness_options.team: the team starts no agent, so the task would reach nobody"
    entry = entry_of(team)
    for name, role in team.roles.items():
        if name in RESERVED_ROLES:
            yield f"team roles.{name}: platoon gives the role `{name}` a meaning of its own"
        program = role.harness.command[0] if isinstance(role.harness, Command) else None
        if program is not None and os.sep in program and not os.path.isabs(program):
            yield (
                f"team roles.{name}.harness: `{program}` is a relative path, and a rollout runs in an empty "
                "repository: name it absolutely, or by a name on PATH"
            )
        spawned = sorted(parent for parent in team.spawners(name) if team.spawners(parent))
        if entry is not None and role.on_parent_exit == "adopt" and spawned:  # under several, every agent is minted
            yield (
                f"team roles.{name}.on_parent_exit: platoon's spawn of an agent of "
                f"{' or '.join(f'`{parent}`' for parent in spawned)} returns only once the agent's children have "
                "ended, and the bridge holds the agent's end until then, so an `adopt` child that outlives it would "
                "hold its end, and one that stays or waits on the tree would hold the rollout until it times out: "
                f"leave `{name}` `cancel-subtree`"
            )
    agents = environment.agents
    if agents.child_grading == "verifier":
        yield (
            "agents.child_grading: platoon runs a child's verifier on the child's harness and options, and a team's "
            "child's are that child, so its own end would be its verdict"
        )
    if entry is None:
        defaults = type(agents)()
        for setting in UNMINTABLE:
            if getattr(agents, setting) != getattr(defaults, setting):
                yield f"agents.{setting}: a team that starts several agents mints them, and a mint cannot honour it"
    yield from _reward_problems(environment, team, phase)


def _is_ours(harness: str) -> bool:
    try:
        resolved = pt.get_harness(harness)
    except ExtensionError:  # unknown or ambiguous
        return False
    return isinstance(resolved, pt.FunctionHarness) and resolved.fn is team_harness


def _reward_problems(environment: Any, team: Team, phase: str) -> Iterator[str]:
    try:
        rewards = _rewards(environment.harness_options)
    except ValueError as error:
        yield f"harness_options.{error}"
        return
    # An entry without `credit` is credited as the run chooses, which a check does not see: with keyed, which reads
    # spans, in a run that trains several policies, else with the workflow's own, which the run refuses on the first
    # group whose agents name spans.
    credit = environment.credit
    if rewards and phase == "train" and credit is not None and not _reads_spans(credit):
        yield (
            f"credit: each span of a role `rewards` names trains on its own row only under a credit that reads spans, "
            f"and {credit!r} reads none: write `credit: keyed`"
        )
    if rewards and environment.reward_processor not in REWARD_PROCESSORS:
        yield (
            f"reward_processor: a producer's row lands where the default processor and platoon-delegation read a "
            f"reward, and {environment.reward_processor!r} reads its own"
        )
    entry = entry_of(team)
    first: dict[str, tuple[str, dict[str, Any]]] = {}  # the first role to name each producer, and its settings
    for role, (producer, params) in rewards.items():
        where = f"harness_options.rewards.{role}"
        if role not in team.roles:
            yield f"{where}: the team has no role `{role}`"
        elif role == entry and not team.spawners(role):
            yield f"{where}: only the root plays `{role}`, and the root trains on the environment's grade"
        earlier, settings = first.setdefault(producer, (role, params))
        if settings != params:
            yield f"{where}: `{producer}` runs once a rollout, and rewards.{earlier} runs it with other settings"
        try:
            _imported(producer)
        except Exception as error:
            yield f"{where}.producer: `{producer}` does not resolve: {type(error).__name__}: {error}"


def _reads_spans(credit: str) -> bool:
    """Whether the credit estimator ``credit`` names reads the spans its agents name; an estimator that does not
    resolve does, as loading the config refuses it."""

    try:
        return bool(getattr(resolve_credit_estimator(credit), "reads_spans", False))
    except ExtensionError:
        return True


# --- the harness ------------------------------------------------------------------------------------------------


async def run_agent(ctx: pt.AgentContext) -> pt.AgentResult:
    """The harness: as the root, the whole session; as a member, under a handoff, the one agent it was spawned for."""

    bridge = ctx.options.get(BRIDGE)
    if not isinstance(bridge, Bridge):
        return pt.AgentResult("error", error=f"harness {NAME} runs under `program: {NAME}`, which gives it its team")
    agent = ctx.options.get(HANDOFF)
    if agent is None:
        return await _root(ctx, bridge)
    return await _member(ctx, bridge, agent)


async def _root(ctx: pt.AgentContext, bridge: Bridge) -> pt.AgentResult:
    """Run the team's session, whose entry agent, if it starts one, runs on the root's own endpoint."""

    entry, starting = entry_of(bridge.team), bridge.team.recipients()
    try:
        authority = bridge.authority = PlatoonAuthority(ctx, bridge, entry=entry)
        session = await Session.start(
            bridge.team,
            name=bridge.name,
            principal=PRINCIPAL,
            authority=authority,
            cwd=bridge.cwd,
            trainable=True,
        )
    except Exception as refusal:
        bridge.refusal = refusal
        return pt.AgentResult("error", error=f"the session refused to start: {refusal}")
    bridge.session = session
    async with session:
        await session.open_episode(EPISODE)
        text = ctx.instruction
        gist, body = (text, None) if len(text) <= GIST_LIMIT else (f"{text[: GIST_LIMIT - 1]}…", text)
        for agent in starting:
            await session.send(agent, gist, body=body)
        async with asyncio.TaskGroup() as tasks:
            retiring = tasks.create_task(_retire_once_stopped(session))
            await session.settled()
            retiring.cancel()
        await _end_every_agent(session, authority)
        rows = await session.agents()
        said = [row["said"] for row in rows if f"@{row['agent']}" in starting and row["said"] is not None]
        await session.close_episode(based_on=said)
        if (await session.status())["state"] == "open":  # `settled` closed it at its `until`, unless it needs a person
            await session.close()
        ends = await authority.ended(row["agent"] for row in rows)
    return _report(ends, entry)


async def _retire_once_stopped(session: Session) -> None:
    """Retire each agent the gateway has answered with a budget stop as soon as it is at rest, so that no message wakes
    it into more stops."""

    stopped: set[str | None] = set()
    async for event in session.events(match="turn/*", follow=True):
        if event.type == "model.called" and (event.data["call_id"] or "").startswith(BUDGET_STOP_PREFIX):
            stopped.add(event.agent)
        elif event.type == "turn.ended" and event.agent in stopped:
            try:
                await session.retire(f"@{event.agent}")
            except Refused as refused:  # `conflict`: woken again, or ended; its next turn's end tries again
                if refused.refusal.code != "conflict":
                    raise


async def _end_every_agent(session: Session, authority: PlatoonAuthority) -> None:
    """Retire each agent at rest and cancel each at work, deepest first, as one read of the agents found them.  A
    child's end may wake its parent, which is then on its way to a turn, and so cancelled: the authority tells one
    cancelled with no turn of its under way as retired (:meth:`PlatoonAuthority.closing`).  Such a parent may also end
    of itself before the close comes to it, and is then told as it ended."""

    for row in sorted(await session.agents(), key=lambda row: row["depth"], reverse=True):
        if row["state"] == "ended":
            continue
        agent = row["agent"]
        try:
            await session.retire(f"@{agent}")
        except Refused as refused:  # `conflict`: it is at work, which only a cancel ends, or it has ended since
            if refused.refusal.code != "conflict":
                raise
            close, under_way = authority.closing(agent), True  # until its end says otherwise, it is told as it ended
            try:
                await session.cancel_agent(f"@{agent}")
                under_way = await _turn_under_way(session, agent)
            except Refused as ended:  # `conflict`: it has ended since the read, and not by the close's hand
                if ended.refusal.code != "conflict":
                    raise
            finally:
                close.set_result(under_way)


async def _turn_under_way(session: Session, agent: str) -> bool:
    """Whether a turn of an ended agent's was under way when it ended: its end names the turn then open, if any."""

    (end,) = [event async for event in session.events(streams=[f"agent/{agent}"]) if event.type == "agent.ended"]
    return end.turn is not None


def _report(ends: Mapping[str, AgentOutcome], entry: str | None) -> pt.AgentResult:
    """The root's run: the entry agent's end, or under a zero-call root, ``error`` naming the agents that ended lost or
    failed, and ``completed`` when none did."""

    if entry is not None:
        return result_of(entry, ends[entry])
    failed = sorted(agent for agent, outcome in ends.items() if outcome.status in ("lost", "failed"))
    if failed:
        return pt.AgentResult("error", error=f"{', '.join(failed)} ended lost or failed")
    return pt.AgentResult("completed")


async def _member(ctx: pt.AgentContext, bridge: Bridge, agent: str) -> pt.AgentResult:
    """Run one spawned agent of the root's session on this context, and report its end."""

    assert bridge.authority is not None and bridge.session is not None
    try:
        outcome = await bridge.authority.serve(agent, ctx)
    except DriverError as refused:  # its handoff was served: this run's end would not be its own
        return pt.AgentResult("error", error=str(refused))
    except asyncio.CancelledError:  # platoon cancelled the run: so its agent ends, unless it has, or its session has
        with contextlib.suppress(Refused):
            await bridge.session.cancel_agent(f"@{agent}")
        raise
    return result_of(agent, outcome)


# --- the program ------------------------------------------------------------------------------------------------


async def run_episode(ep: pt.EpisodeContext) -> pt.Outcome:
    """The program: ``single``'s finalizers and the team's, the root in an empty repository, platoon's grades and the
    producers ``rewards`` names, each written into the ended session."""

    options = ep.route.harness_options
    rewards = _rewards(options)
    if ep.config.skip_subagent_reward_computation:
        ep.add_finalizer(zero_subagent_rewards)
    if ep.config.propagate_root_success:
        ep.add_finalizer(propagate_root_success)  # ty: ignore[invalid-argument-type]
    producers = {role: producer for role, (producer, _) in rewards.items()}
    bridge = Bridge(team=load(options["team"]), cwd=Path(mkdtemp("platoon-harness-rollout-")), name=ep.root.agent_id)

    async def finalizing(collection: pt.TrajectoryCollection) -> None:
        if bridge.session is not None:  # else nothing of ours ran
            await finalize(collection, bridge.name, producers)

    ep.add_finalizer(finalizing)
    try:
        await asyncio.to_thread(_commit_nothing, bridge.cwd)
        await ep.run(options={BRIDGE: bridge})
        if bridge.refusal is not None:
            raise bridge.refusal
        outcome = await ep.grade()
        if bridge.session is None:  # the root was cut before its session started
            return outcome
        assert bridge.authority is not None
        async with Session.attach(bridge.name) as ended:
            record = await Record.read(ended)
            await _write_grades(ended, record, outcome, await bridge.authority.grades())
            if record.closed is not None:  # else the root was cut first: no team reward, and no agent ended
                judge = _judge(ep) if rewards and ep.judges else None
                for producer, params in {producer: params for producer, params in rewards.values()}.items():
                    await _imported(producer)(ended, EPISODE, judge=judge, **params)
        return outcome
    finally:
        await asyncio.to_thread(shutil.rmtree, bridge.cwd, ignore_errors=True)


def _commit_nothing(cwd: Path) -> None:
    """Make ``cwd`` a repository with one empty commit, platoon's: a rollout's working directory."""

    git("init", "-q", cwd=cwd)
    who = ("-c", "user.name=platoon", "-c", "user.email=platoon@localhost", "-c", "commit.gpgsign=false")
    git(*who, "commit", "-q", "--allow-empty", "-m", "An empty repository for one rollout", cwd=cwd)


async def _write_grades(
    session: Session, record: Record, outcome: pt.Outcome, children: Mapping[str, pt.Outcome]
) -> None:
    """Platoon's grades as rewards of the producer ``platoon`` at its version: the episode's outcome as the team's on
    the episode's close, and each graded spawned agent's on its first start."""

    head = await session.head()
    graded: list[tuple[str, Literal["team", "local"], pt.Outcome]] = []
    graded += [(record.closed, "team", outcome)] if record.closed is not None else []
    graded += [(record.started[agent], "local", grade) for agent, grade in children.items() if agent in record.started]
    for target, scope, grade in graded:
        await session.annotate(
            target,
            kind="reward",
            producer=GRADER,
            producer_version=pt.__version__,
            scope=scope,
            status=grade.status,
            value=grade.reward if grade.status == "scored" else None,
            head_seq=head,
        )


def _judge(ep: pt.EpisodeContext) -> Gateway:
    """The rollout's judge, as model access for a producer that asks a model."""

    endpoint = ep.judge_endpoint()
    origin, bearer = normalize(endpoint.base_url, endpoint.api_key, "the judge's endpoint")
    return Gateway(origin=origin, bearer=bearer, model=endpoint.model)
