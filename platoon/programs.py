"""Episode programs: how one task's rollout composes harness runs, grading and post-processing.

A program is ``async def program(ep: EpisodeContext) -> Outcome`` registered with ``@pt.program(name)`` (as
``name@version``, version ``1`` unless it says otherwise); an environment picks one with ``program:`` (default
``single``).  It runs in the rollout worker as trusted code and fails closed: an exception is an ``error``
outcome, never a score.  An extension provides programs through the ``platoon.programs`` entry-point group, one
entry point per program, named as the program is (:mod:`platoon.extensions`).

```python
@pt.program("solve-then-check")
async def solve_then_check(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()                                   # the root agent, with the route's harness
    ep.add_finalizer(my_reward_shaping)              # runs on the built TrajectoryCollection
    return await ep.grade()
```
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import threading
import types
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from platoon import extensions

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from platoon.config_defs import RolloutConfig
    from platoon.envs.service import EnvSession, Outcome
    from platoon.gateway.core import AgentSession
    from platoon.harness.agent import AgentResult, ModelEndpoint
    from platoon.harness.inline import InlineRuntime
    from platoon.harness.verify import SubagentBehaviorJudge
    from platoon.task import Task
    from platoon.train.components import EnvironmentRoute
    from platoon.trajectory import TrajectoryCollection

logger = logging.getLogger(__name__)

DEFAULT_PROGRAM = "single"
# The reason of the placeholder outcome an episode snapshot carries (``EpisodeContext.snapshot``).
SNAPSHOT_REASON = "snapshot"
# The key under which a training episode's draw travels (:class:`DrawInfo`): in its task row, and in its rollout
# config's ``extra``, when its trainer asks for draw provenance.  It is platoon's: a config's own ``extra`` leaves it
# out.
DRAW_KEY = "platoon_draw"


@dataclass(frozen=True)
class DrawInfo:
    """The run and the training draw an episode belongs to (``ctx.draw``), when its trainer asks for them (RFC 0021's
    draw provenance, ``TrainerNeeds.draw_provenance``); never in an evaluation.  A draw's key ``(start, seq)`` is
    never reused in its run, not even by a draw a resume makes again, and a step's end names the draws it committed by
    it (``StepEnd.consumed``)."""

    # The run directory, where a program may write the files of its draw that its trainer reads.
    run_dir: str
    # The start of the run that drew it, and its number among that start's draws.
    start: int
    seq: int
    # Its position in the run's order of training tasks (None: a draw that a task sampler decided by itself).
    position: int | None
    # The training step that was current when its group started.
    step: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.start, self.seq)


Finalizer = Callable[["TrajectoryCollection"], "Awaitable[None] | None"]


class EpisodeContext(Protocol):
    """One rollout of one task, as a program sees it.

    ``behavior_judge`` (verifier grading of children) and ``tool_transform`` (rewrites the MCP tool listing of
    every agent, ``platoon.gateway.mcp.bind_agent_tools``) may be set before :meth:`run`.  ``root`` is the root
    agent's record (``platoon.gateway.AgentSession``: its misc, result and calls), e.g. for self-graded harnesses.
    """

    # The task the rollout works on.
    task: Task
    # The rollout's environment session.
    session: EnvSession
    # The rollout config.
    config: RolloutConfig
    # The environment entry the task came from: its harness, program, options and subagent settings.
    route: EnvironmentRoute
    # The root agent's record: its misc, result and calls.
    root: AgentSession
    # Judges children for verifier grading; set it before ``run``.
    behavior_judge: SubagentBehaviorJudge | None
    # Rewrites the MCP tool listing of every agent; set it before ``run``.
    tool_transform: Callable[..., Any] | None
    # Whether this rollout's judges run (the rollout config's ``judges``, which ``platoon eval`` sets from
    # ``inference.judges``).  While they are off, ``judge_endpoint`` and ``judge`` calls are refused
    # (``platoon.gateway.JudgesOffError``) and verifiers do not run: a program that judges reads it and does without.
    judges: bool
    # The run and the training draw this episode belongs to, when its trainer asks for them (experimental); None in an
    # evaluation and otherwise.
    draw: DrawInfo | None

    async def run(
        self,
        harness: str | None = None,
        *,
        goal: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        """Run the root agent: ``harness`` (default: the route's) on ``goal`` (default: the instruction)."""
        ...

    async def grade(self) -> Outcome:
        """Grade the session, failing closed (a grader that raises is an ``error`` outcome)."""
        ...

    def judge_endpoint(self) -> ModelEndpoint:
        """A model endpoint on a ``judge``-role agent under the root: recorded for audit, never trained.  Refused
        with ``platoon.gateway.JudgesOffError`` while :attr:`judges` is false."""
        ...

    def add_finalizer(self, fn: Finalizer) -> None:
        """Run ``fn`` on the rollout's ``TrajectoryCollection`` once it is built, before it is returned.

        Finalizers run in the order they were added; adding the same finalizer again does nothing.
        """
        ...

    def snapshot(self) -> TrajectoryCollection:
        """The agent tree recorded so far, as a ``TrajectoryCollection`` (the root's trajectory first).

        It is built by the trace that builds the rollout's result (``platoon.gateway.trace``), with a placeholder
        episode outcome (``scored``, reward 0, reason ``snapshot``) for the root and every agent without an outcome
        of its own, e.g. for a judge that reads a child's lineage mid-episode.  Nothing is closed or changed, and
        the result is unaffected; the snapshot shares annotation values with the live records, so treat it as
        read-only.
        """
        ...


# An episode program: ``async def program(ep: EpisodeContext) -> Outcome``.
EpisodeProgram = Callable[[EpisodeContext], Awaitable["Outcome"]]
# The attribute the decorator sets on the function it registers.
PROGRAM_ATTRIBUTE = "__platoon_program__"


@dataclass(frozen=True)
class RegisteredProgram:
    """An episode program as configs name it: ``name@version``, and its function.

    A ``module:attr`` reference to a function that ``@pt.program`` did not register resolves to a record named
    by that reference, with no version.
    """

    name: str
    version: str
    fn: EpisodeProgram = field(repr=False, compare=False)

    @property
    def ref(self) -> str:
        """``name@version``, the canonical identity of the program."""
        return f"{self.name}@{self.version}" if self.version else self.name

    @property
    def source(self) -> str:
        """``module:qualname`` of the function."""
        return f"{self.fn.__module__}:{getattr(self.fn, '__qualname__', type(self.fn).__qualname__)}"


_lock = threading.Lock()
_programs: dict[str, RegisteredProgram] = {}


def program(name: str, *, version: str = "1") -> Callable[[EpisodeProgram], EpisodeProgram]:
    """Register the decorated ``async def fn(ep: EpisodeContext) -> Outcome`` as episode program ``name@version``.

    An environment config runs it with ``program: name`` (or ``name@version`` when several versions are
    registered): through an installed ``platoon.programs`` entry point named ``name``, or after the entry's
    ``package`` imports its module.  The name follows platoon's naming rule (``<owner>-<name>``, lower case;
    ``platoon-*``, ``single`` and ``gym`` are core's).  Registering the same function again (a re-imported module)
    is allowed; another function under a registered ``name@version`` is an error.
    """

    extensions.check_name(name, what="program")
    version = extensions.check_version(version, what=f"program {name!r}")

    def decorate(fn: EpisodeProgram) -> EpisodeProgram:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@program({name!r}) needs an async function, got {fn!r}")
        extensions.check_registration("program", name, module=getattr(fn, "__module__", None))
        record = RegisteredProgram(name=name, version=version, fn=fn)
        register_program(record)
        setattr(fn, PROGRAM_ATTRIBUTE, record)
        return fn

    return decorate


def register_program(record: RegisteredProgram) -> None:
    """Add ``record`` to the process registry (see :func:`program`)."""

    with _lock:
        existing = _programs.get(record.ref)
        if existing is not None and existing.source != record.source:
            raise ValueError(f"Program {record.ref} is defined twice: {existing.source} and {record.source}")
        _programs[record.ref] = record


def program_of(fn: object) -> RegisteredProgram | None:
    """The :class:`RegisteredProgram` a decorated function was registered as, or None."""

    record = getattr(fn, PROGRAM_ATTRIBUTE, None)
    return record if isinstance(record, RegisteredProgram) else None


def registered_programs() -> list[RegisteredProgram]:
    """Every program registered in this process, by reference."""

    with _lock:
        return [_programs[ref] for ref in sorted(_programs)]


# Modules whose import registers built-in programs besides ``single``.
BUILTIN_PROGRAM_MODULES = ("platoon.gym",)


@functools.cache
def _import_builtin_programs() -> None:
    for module in BUILTIN_PROGRAM_MODULES:
        importlib.import_module(module)


def builtin_programs() -> list[RegisteredProgram]:
    """Core's programs (``single``, ``gym``)."""

    _import_builtin_programs()
    return [record for record in registered_programs() if extensions.is_core_module(record.fn.__module__)]


def resolve_program(ref: str) -> RegisteredProgram:
    """The program ``ref`` names: ``name[@version]``, or ``module:attr`` (imported, which also registers the
    module's decorators).

    A name is core's (``single``, ``gym``), an installed ``platoon.programs`` entry point's (loaded the first time
    it is named, after its compatibility check), or one a ``package`` module registered.  A bare name must have a
    single registered version.  Raises :class:`~platoon.extensions.ExtensionError` (a ``ValueError``) when
    nothing, or more than one program, matches.
    """

    if ":" in ref:
        module, _, attr = ref.partition(":")
        value: Any = functools.reduce(getattr, attr.split("."), importlib.import_module(module))
        if not callable(value):
            raise TypeError(f"Program {ref!r} is not callable")
        return program_of(value) or RegisteredProgram(name=ref, version="", fn=value)
    _import_builtin_programs()
    name, _ = extensions.split_ref(ref)
    extensions.load_named("program", name)
    candidates = registered_programs()
    matches = extensions.matching(ref, candidates)
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise extensions.ambiguous("program", ref, [record.ref for record in matches])
    raise extensions.unknown("program", ref, [record.ref for record in candidates])


def get_program(name: str) -> EpisodeProgram:
    """The function of the program ``name`` names (see :func:`resolve_program`)."""

    return resolve_program(name).fn


def available_programs() -> list[str]:
    """Every program registered in this process: core's, and those loaded or imported so far."""

    _import_builtin_programs()
    return [record.ref for record in registered_programs()]


def accept_entry_point(value: Any, origin: extensions.Origin, entry_point: EntryPoint) -> None:
    """Register what a ``platoon.programs`` entry point loads to: an ``@pt.program`` function named as the entry
    point is, a list of them (several versions), or a module whose import registered one."""

    if isinstance(value, types.ModuleType):
        records = [
            record
            for record in registered_programs()
            if record.name == entry_point.name
            and (record.fn.__module__ == value.__name__ or record.fn.__module__.startswith(value.__name__ + "."))
        ]
        if not records:
            raise ValueError(f"module {value.__name__} registers no program named {entry_point.name!r}")
    else:
        items = list(value) if isinstance(value, (list, tuple)) else [value]
        records = []
        for item in items:
            record = program_of(item)
            if record is None:
                raise TypeError(f"a platoon.programs entry point must load to an @pt.program function, got {item!r}")
            if record.name != entry_point.name:
                raise ValueError(
                    f"it loads to program {record.ref}, whose name is not the entry point's {entry_point.name!r}; "
                    "name the entry point after the program"
                )
            records.append(record)
    for record in records:
        extensions.record_origin("program", record.ref, origin)


@program(DEFAULT_PROGRAM)
async def single(ep: EpisodeContext) -> Outcome:
    """Run the route's harness on the root agent, then grade the session.

    It adds the standard finalizers the rollout config asks for, in the order of the original recursive rollouts:
    with ``skip_subagent_reward_computation``, :func:`platoon.rewards.zero_subagent_rewards`, so every subagent
    scores 0; then, with ``propagate_root_success``,
    :func:`platoon.utils.subagent_rewards.propagate_root_success`, so every trajectory but a forced-zero one takes
    the root's success.  Custom programs own their finalizers and their order: one that wants either adds it
    itself, where its other reward finalizers need it.
    """

    from platoon.rewards import zero_subagent_rewards
    from platoon.utils.subagent_rewards import propagate_root_success

    await ep.run()
    if ep.config.skip_subagent_reward_computation:
        ep.add_finalizer(zero_subagent_rewards)
    if ep.config.propagate_root_success:
        ep.add_finalizer(propagate_root_success)
    return await ep.grade()


class Episode:
    """The :class:`EpisodeContext` ``harness_rollout`` gives a program, over an inline runtime."""

    def __init__(
        self,
        *,
        task: Task,
        session: EnvSession,
        config: RolloutConfig,
        route: EnvironmentRoute,
        runtime: InlineRuntime,
        root: AgentSession,
    ) -> None:
        self.task = task
        self.session = session
        self.config = config
        self.route = route
        self.runtime = runtime
        self.root = root
        self.behavior_judge: SubagentBehaviorJudge | None = None
        self.tool_transform: Callable[..., Any] | None = None
        self.root_result: AgentResult | None = None
        self.harness_ref: str | None = None
        self.finalizers: list[Finalizer] = []
        self._judge: AgentSession | None = None

    async def run(
        self,
        harness: str | None = None,
        *,
        goal: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AgentResult:
        from platoon.harness.function import FunctionHarness
        from platoon.harness.inline import InlineAgentContext, agent_timeout, run_agent
        from platoon.harness.registry import get_harness

        if self.root_result is not None or self.harness_ref is not None:
            raise RuntimeError("The root agent runs once per episode")
        ref = harness or self.route.harness
        if ref is None:
            raise ValueError("No harness to run: the route names none and run() got none")
        resolved = get_harness(ref)
        if not isinstance(resolved, FunctionHarness):
            raise TypeError(f"EpisodeContext.run runs function harnesses; {resolved.ref} is a TOML spec")
        self.harness_ref = resolved.ref
        self.runtime.behavior_judge = self.behavior_judge
        self.runtime.tool_transform = self.tool_transform
        ctx = InlineAgentContext(
            self.runtime,
            self.root,
            task=self.task,
            session=self.session,
            harness=resolved,
            instruction=goal or self.session.instruction or self.task.goal or "",
            options=self.route.harness_options if options is None else options,
        )
        self.root_result = await run_agent(ctx, timeout=agent_timeout(self.config, self.session))
        return self.root_result

    async def grade(self) -> Outcome:
        """Grade the session with the root run's failure taxonomy (``platoon.harness.inline.failure_outcome``)."""
        from platoon.harness.inline import failure_outcome, grade_session

        reason = None
        if self.root_result is not None:
            failed, reason = failure_outcome(
                self.root_result,
                context_overflows=self.root.context_overflows,
                on_context_overflow=self.route.on_context_overflow,
                harness_ref=self.harness_ref or "harness",
                on_harness_exit=self.route.on_harness_exit,
                gateway_errors=self.root.gateway_errors,
            )
            if failed is not None:
                return failed
        return await grade_session(self.session, reason=reason)

    @property
    def judges(self) -> bool:
        return bool(getattr(self.root, "judges", True))

    @property
    def draw(self) -> DrawInfo | None:
        fields = (getattr(self.config, "extra", None) or {}).get(DRAW_KEY)
        return DrawInfo(**fields) if fields else None

    def judge_endpoint(self) -> ModelEndpoint:
        if self._judge is None:
            self._judge = self.runtime.gateway.open_subagent(self.root, role="judge", goal="judge")
        return self.runtime.endpoint(self._judge)

    def add_finalizer(self, fn: Finalizer) -> None:
        if fn not in self.finalizers:  # a finalizer registered twice (say, by a wrapped program) runs once
            self.finalizers.append(fn)

    def snapshot(self) -> TrajectoryCollection:
        from platoon.envs.service import Outcome
        from platoon.gateway.trace import build_trajectory_collection
        from platoon.harness.inline import recorded_task

        # The rollout's agents stay open (and listed) until harness_rollout closes them after the finalizers.
        agents = self.runtime.gateway.agents(self.root.rollout_id)
        placeholder = Outcome(status="scored", rewards={"reward": 0.0}, reason=SNAPSHOT_REASON)
        return build_trajectory_collection(agents, task=recorded_task(self.task, self.session), outcome=placeholder)

    async def finalize(self, collection: TrajectoryCollection) -> None:
        """Run the finalizers in registration order."""
        for fn in self.finalizers:
            result = fn(collection)
            if inspect.isawaitable(result):
                await result


__all__ = [
    "BUILTIN_PROGRAM_MODULES",
    "DEFAULT_PROGRAM",
    "Episode",
    "EpisodeContext",
    "EpisodeProgram",
    "Finalizer",
    "PROGRAM_ATTRIBUTE",
    "RegisteredProgram",
    "SNAPSHOT_REASON",
    "accept_entry_point",
    "available_programs",
    "builtin_programs",
    "get_program",
    "program",
    "program_of",
    "register_program",
    "registered_programs",
    "resolve_program",
    "single",
]
