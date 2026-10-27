"""Task samplers: how a training run chooses the task each of its rollout groups runs (RFC 0019, section 5).

A task sampler is a small class: :meth:`Sampler.next` answers one decision with a task of its pool (a :class:`Draw`,
or a bare task id), or, from a run's sampler that hands its slots to the entries, a :class:`Slot` that gives the draw
to an entry's own sampler.  It is built once per start of the process, ``Class(setup, options)``
(:class:`SamplerSetup`), and each decision gets a :class:`SamplerContext`: its pool, its decision index, the key its
randomness follows from, and which tasks it may choose.  ``state_dict`` and ``load_state_dict`` save and restore what
it keeps between decisions, with the dispatcher's state at each full checkpoint.

A **static** sampler (``adaptive = False``) is a function of its decision index alone, and is asked when the
dispatcher draws a new position of the run's order: ``weighted`` hands slot ``n`` to the entry its credit rule picks,
and an entry's ``uniform`` takes the next task of the entry's seeded passes.  An **adaptive** sampler
(``adaptive = True``) reads the run's per-task statistics (:class:`TaskStatsView`) and is asked only when a group can
start: within the horizon, with its group's rollout permits held, choosing among the tasks whose label has room.  The
dispatcher keeps each sampler's decision count, which advances whenever a decision binds a draw or a slot; a
decision's randomness is keyed by the sampler's seed, its name and that count (:func:`decision_key`), so a run, and a
resume of it, makes the same decisions.

The dispatcher refuses an answer outside the contract (a task outside the pool, a task whose label is full from an
adaptive sampler, a slot a sampler may not give, a reason or score that cannot be recorded, nothing while a task is
eligible), and a sampler that raises: the decision then gets a fallback draw, chosen by the decision's key, which uses
the decision up, and the failure is logged once per sampler and counted (:mod:`platoon.train.sampler_runtime`).

Samplers are registered by name with :func:`task_sampler` from a module an environment entry's ``package`` names, or
named as ``module:Class``; core's names (:data:`BUILTIN_TASK_SAMPLERS`, and ``auto``) are reserved
(:func:`resolve_task_sampler`).  The API is experimental (``platoon.experimental``): a minor release may change it.
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

from platoon.train.task_stats import (
    EntryInfo,
    GroupRecord,
    GroupTotals,
    TaskCounts,
    TaskStats,
    TaskStatsStore,
)

# A sampler's level: the run's sampler (train.task_sampler), or an entry's (environments[].task_sampler).
Level = Literal["run", "entry"]
S = TypeVar("S", bound=type)

# The most characters a draw's reason may have: it goes into the group's record and the draw's event.
MAX_REASON_LENGTH = 64
# The registry kind of task samplers (platoon.registry), which @task_sampler registers in.
REGISTRY_KIND = "task_sampler"
# Core's samplers, by name: resolved from here before the registry is read, so that nothing installed or registered can
# shadow them, whatever the import order.  ``auto`` is reserved too: the config resolves it (weighted for a mixture,
# the run's uniform passes for a single environment).
BUILTIN_TASK_SAMPLERS: dict[str, str] = {
    "weighted": "platoon.train.samplers:WeightedSampler",
    "uniform": "platoon.train.samplers:UniformSampler",
    "pass_rate": "platoon.train.samplers:PassRateSampler",
}
RESERVED_TASK_SAMPLER_NAMES = frozenset({"auto", *BUILTIN_TASK_SAMPLERS})
# What a registered name may be: no spaces, and neither ``:`` (module:Class) nor ``@`` (an entry's qualified name).
_NAME = re.compile(r"[^\s:@]+")
# The metric keys the dispatcher and the trainer write themselves under a sampler's prefix (``train/sampler/``, or
# ``train/sampler/environment/<entry>/``), which a sampler's ``metrics()`` may not take, and neither may ``draws/...``:
# its draws by reason, its decision time, an entry's waiting slots, the slots skipped, and what it predicted of the
# groups a step consumed against what they did.
RESERVED_METRICS = frozenset(
    {"draws", "decision_ms", "decision_ms_max", "tokens", "skipped_slots", "predicted_informative", "informative_share"}
)


def reserved_metric(key: str) -> bool:
    """Whether a sampler's metric key is one the dispatcher writes itself (:data:`RESERVED_METRICS`, ``draws/...``)."""

    return key in RESERVED_METRICS or key.startswith("draws/")


def _plain(name: str) -> str:
    """A sampler name as it reads, whatever its case and its ``-`` or ``_``: ``Pass-Rate`` reads as ``pass_rate``."""

    return name.casefold().replace("-", "_")


@dataclass(frozen=True)
class Draw:
    """A sampler's answer: one task of its pool.  ``reason`` (a non-empty string of at most 64 characters) and
    ``score`` (None or a finite number) go into the group's record (``GroupRecord.reason``, ``.score``), its draw
    event and the step's metrics.  A pool whose listing repeats a task id draws the first row with that id."""

    # The task drawn: one of the decision's ``ctx.task_ids``.
    task_id: str
    # Why it was drawn, as the group's record, its draw event and the step's draws by reason (draws/<reason>) keep it.
    reason: str = "order"
    # What the sampler made of the task, None for nothing: the group's record keeps it, and pass_rate's is the chance
    # it predicted that the group trains (predicted_informative).
    score: float | None = None


@dataclass(frozen=True)
class Slot:
    """A run sampler's answer that hands this draw to an entry's own sampler, as ``weighted`` does.  Only a static
    run sampler whose class sets ``entry_slots = True`` may return one."""

    # The label of the entry whose sampler draws the task.
    entry: str


@dataclass(frozen=True)
class SamplerSetup:
    """What a sampler is built with, once per start of the process: ``Class(setup, options)``."""

    # Its qualified name: the run's sampler as the config names it ("auto", "weighted", "pass_rate", "acme.s:Mine"),
    # an entry's with the entry's label ("uniform@quick", "pass_rate@hard").
    name: str
    # Which sampler it is: the run's ("run", train.task_sampler) or an entry's ("entry").
    level: Level
    # The entry, at the entry level.
    entry: str | None
    # Its pool, in listing order: the run's every training task, or one entry's.
    task_ids: tuple[str, ...]
    # Every entry with training tasks, in config order.
    entries: Mapping[str, EntryInfo]
    # The run's train.batch_size.
    batch_size: int
    # The column the run's labels come from (train.label_by), whose label limits eligibility follows.
    label_column: str
    # The seed its decisions' keys follow from (SamplerContext.key): the entry's seed at the entry level, 0 at the run
    # level, plus its parsed options' seed when they have an integer one (as pass_rate's do), so a sampler with a seed
    # option reads it here and need not add it again.
    seed: int


def decision_key(seed: int, name: str, decision: int) -> int:
    """The key of decision ``decision`` of the sampler named ``name`` with seed ``seed``: the first 128 bits of
    ``sha256(f"{seed}:{name}:{decision}")``, the same in every process, whatever its hash seed."""

    return int.from_bytes(hashlib.sha256(f"{seed}:{name}:{decision}".encode()).digest()[:16], "big")


class TaskStatsView:
    """The run's per-task statistics as an adaptive sampler reads them, at policy version ``now``: read-only, every
    result a value of its own.  ``live`` is False when the statistics stopped in this start (the view reads what they
    held) or never started (it reads an empty store over the pool).

    :meth:`counts` is the cheap read (a task's counts and streaks, read from its record); :meth:`task` builds the
    task's whole record, so a decision that reads it for every task of a large pool spends time in proportion to the
    pool: read ``counts`` for every task, and ``task`` for a few."""

    def __init__(self, store: TaskStatsStore, *, now: int, live: bool = True) -> None:
        self._store = store
        self._now = now
        self._live = live
        # The read a decision makes for every task of its pool, bound to the store's own: one call a task, not two.
        self.counts = store.counts  # type: ignore[method-assign]

    @property
    def now(self) -> int:
        """The policy version the view reads at."""

        return self._now

    @property
    def live(self) -> bool:
        """False when the statistics stopped in this start, or never started: then the view reads what they held, or
        nothing."""

        return self._live

    def task_ids(self) -> tuple[str, ...]:
        """Every training task of the run, in pool order."""

        return tuple(self._store.task_ids())

    def counts(self, task_id: str) -> TaskCounts:
        """The task's counts and streaks (:class:`~platoon.train.task_stats.TaskCounts`), read without building
        anything: the read to make for every task of a pool.  ``KeyError`` for a task the statistics do not hold."""

        return self._store.counts(task_id)

    def task(self, task_id: str) -> TaskStats:
        """The task's whole record (:class:`~platoon.train.task_stats.TaskStats`): attempts, passes, kinds, fates, the
        newest groups and decayed means, built afresh, so read it for a few tasks only.  ``KeyError`` for a task the
        statistics do not hold."""

        return self._store.task(task_id)

    def posterior(self, task_id: str) -> tuple[float, float]:
        """``(alpha, beta)`` of the task's posterior pass rate at :attr:`now`: its decayed passes and failures over a
        prior of weight 2 at its entry's decayed pass rate."""

        return self._store.posterior(task_id, now=self._now)

    def entry(self, name: str) -> GroupTotals:
        """What the groups of entry ``name`` add up to."""

        return self._store.entry(name)

    def label(self, label: str) -> GroupTotals:
        """What the groups of the tasks labelled ``label`` add up to."""

        return self._store.label(label)

    def labels(self) -> tuple[str, ...]:
        """The labels of the run's training tasks."""

        return tuple(self._store.labels())

    def group_size(self, entry: str | None) -> int:
        """Entry ``entry``'s training group size (None: the run's)."""

        return self._store.group_size(entry)

    def threshold(self, entry: str | None) -> float:
        """The reward at or above which a rollout of entry ``entry`` passes (its ``success_threshold``)."""

        return self._store.threshold(entry)


@dataclass(frozen=True)
class SamplerContext:
    """What one decision sees.  A decision is one synchronous call; it must not keep the context."""

    # The candidates: the sampler's whole pool (SamplerSetup.task_ids), in listing order.  Eligibility is the
    # predicate below, never a filtered list, so that an index means the same in every decision.
    task_ids: Sequence[str]
    # The entry whose tasks it chooses among, at the entry level; None at the run level.
    entry: str | None
    # This sampler's decision index, 0-based, saved and restored: it advances whenever a draw or a slot binds for it,
    # its own or the fallback's, and never for a None while nothing is eligible.
    decision: int
    # decision_key(seed, name, decision): what the decision's randomness follows from (a sampler that brings numpy
    # seeds numpy.random.Generator(numpy.random.Philox(key)) with it).
    key: int
    # Whether a task may be chosen now: its label is below its limit; always true for a static sampler, whose draws may
    # wait for their label.
    eligible: Callable[[str], bool]
    # The run's per-task statistics, which only an adaptive sampler reads (TaskStatsView); None for a static one.
    stats: TaskStatsView | None
    # The training steps committed.
    step: int
    # The policy version now, by policy: {"main": version}.
    versions: Mapping[str, int]

    @cached_property
    def rng(self) -> random.Random:
        """``random.Random(key)``: the decision's random numbers, built when first read."""

        return random.Random(self.key)


@dataclass(frozen=True)
class SamplerCheckContext:
    """What a sampler's load-time ``check`` sees: the parts of the config a sampler's rules need, as plain data."""

    # The level it is checked at: the run's sampler ("run") or an entry's ("entry").
    level: Level
    # The entry, at the entry level.
    entry: str | None
    # train.task_sampler as written ("auto" when unset).
    run_sampler: str
    # Every entry by label, in config order, each exactly {"weight": a number or "tasks", "seed": int, "group_size":
    # int (the entry's training group size), "success_threshold": float, "task_sampler": str | None,
    # "task_sampler_options": dict}.  A check runs before the tasks are listed, so a weight is as written.
    entries: Mapping[str, Mapping[str, Any]]
    # The run's train.batch_size.
    batch_size: int
    # The column the run's labels come from (train.label_by).
    label_column: str


@runtime_checkable
class TaskSampler(Protocol):
    """What the dispatcher calls of a sampler."""

    # Whether it reads the run's statistics, and so decides when a group can start (Sampler.adaptive).
    adaptive: ClassVar[bool]

    def next(self, ctx: SamplerContext) -> Draw | Slot | str | None:
        """Answer one decision (:meth:`Sampler.next`)."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """What it keeps between decisions, as JSON data with string keys (:meth:`Sampler.state_dict`)."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Take up what :meth:`state_dict` saved, or ``ValueError`` (:meth:`Sampler.load_state_dict`)."""
        ...


class Sampler:
    """An optional base class of task samplers: the protocol's defaults, and the optional hooks.  Built once per start
    of the process as ``Class(setup, options)``, it keeps the :class:`SamplerSetup` as ``self.setup`` and the options,
    parsed against :attr:`options_schema` (None without one), as ``self.options``."""

    # Whether it reads the run's statistics, and so decides when a group can start; False: a function of its decision
    # index, asked when the dispatcher draws a new position.
    adaptive: ClassVar[bool] = False
    # Whether next() may return Slot(entry): a static run sampler only.
    entry_slots: ClassVar[bool] = False
    # Its options' schema: a dataclass, a pydantic model, or a function of the options' mapping that returns them
    # parsed and raises ValueError; None: it takes no options.  ``self.options`` holds them parsed.
    options_schema: ClassVar[Any] = None

    def __init__(self, setup: SamplerSetup, options: Any = None) -> None:
        """Built once per start of the process: ``setup`` (:class:`SamplerSetup`) is kept as ``self.setup``, and its
        options, parsed against :attr:`options_schema` (None without one), as ``self.options``."""

        self.setup = setup
        self.options = options

    def next(self, ctx: SamplerContext) -> Draw | Slot | str | None:
        """Answer one decision: a :class:`Draw` of a task of ``ctx.task_ids`` (for an adaptive sampler, one that
        ``ctx.eligible`` accepts), or that task's id; from a static run sampler whose class sets :attr:`entry_slots`,
        a :class:`Slot` that hands the draw to an entry's sampler; None only when nothing is eligible.  One synchronous
        call, its randomness taken from ``ctx.rng`` or ``ctx.key``; an answer outside these, or an exception, gets the
        dispatcher's fallback draw."""

        raise NotImplementedError(f"{type(self).__name__} does not implement next(ctx)")

    def observe(self, group: GroupRecord) -> None:
        """A settled group of its pool, in draw order (adaptive samplers only), before its next decision and before
        every save; it must also work without them."""

    def state_dict(self) -> dict[str, Any]:
        """What it keeps between decisions, as JSON data: nothing."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Take up what :meth:`state_dict` saved: nothing; a ``ValueError`` for any other state."""

        if state:
            raise ValueError(f"{type(self).__name__} keeps no state, got {dict(state)!r}")

    def metrics(self, stats: TaskStatsView | None) -> Mapping[str, float]:
        """Its metrics of the step, each a finite number, which every step logs under its prefix
        (``train/sampler/<key>`` for the run's sampler, ``train/sampler/environment/<entry>/<key>`` for an entry's):
        called once a step with the statistics' view for an adaptive sampler, with None for a static one.  The keys
        the dispatcher writes itself there (:data:`RESERVED_METRICS`, and ``draws/...``) are left out, with an error
        logged once.  None by default."""

        return {}

    @staticmethod
    def check(options: Any, ctx: SamplerCheckContext) -> None:
        """Refuse a config it cannot serve, with a ``ValueError`` that says why: it accepts every config."""


class SamplerError(RuntimeError):
    """A sampler's answer the dispatcher refused, or a decision that has no task to fall back to: it names the sampler
    and what it returned."""

    def __init__(self, sampler: str, message: str) -> None:
        super().__init__(f"task sampler {sampler}: {message}")
        self.sampler = sampler


# --- registration and resolution -------------------------------------------------------------------------------------


def task_sampler(
    name: str, *, options: Any = None, check: Callable[[Any, SamplerCheckContext], None] | None = None
) -> Callable[[S], S]:
    """Register a sampler class under ``name``, from a module that an environment entry's ``package`` names, so that
    configs can name it (``task_sampler: <name>``); ``options`` sets its ``options_schema`` and ``check`` its load-time
    check.  Core's names (``auto``, ``weighted``, ``uniform``, ``pass_rate``) are reserved whatever was imported first,
    and so is any name that reads as one of them whatever its case and its ``-`` or ``_`` (``Pass-Rate``, which a
    config that misspells core's name would otherwise pick up); a name registered twice is an error."""

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError(f"a task sampler's name is a word without spaces, ':' or '@', got {name!r}")
    if _plain(name) in RESERVED_TASK_SAMPLER_NAMES:
        reads = "is" if name in RESERVED_TASK_SAMPLER_NAMES else f"reads as {_plain(name)!r}, which is"
        raise ValueError(
            f"task sampler name {name!r} {reads} core's ({', '.join(sorted(RESERVED_TASK_SAMPLER_NAMES))} are "
            "reserved): register yours under another name"
        )

    def decorator(cls: S) -> S:
        from platoon.registry import register_component

        if not isinstance(cls, type):
            raise TypeError(f"@task_sampler({name!r}) registers a class, got {cls!r}")
        if options is not None:
            cls.options_schema = options  # type: ignore[attr-defined]
        if check is not None:
            cls.check = staticmethod(check)  # type: ignore[attr-defined]
        register_component(REGISTRY_KIND, name, cls)
        return cls

    return decorator


def resolve_task_sampler(ref: str) -> type:
    """The sampler class ``ref`` names: core's (:data:`BUILTIN_TASK_SAMPLERS`), then a name a package registered
    (:func:`task_sampler`), then ``module:Class``.  An :class:`~platoon.extensions.ExtensionError` that says why for
    anything else (with the nearest name), for a class that is not a sampler (``next``, ``state_dict``,
    ``load_state_dict``, and ``adaptive`` set to True or False), and for one that both hands slots to entries
    (``entry_slots``) and is adaptive.  ``auto`` names no class: the config resolves it."""

    from platoon.extensions import ExtensionError
    from platoon.registry import get_registry, import_from_string

    if not isinstance(ref, str) or not ref:
        raise ExtensionError(f"a task sampler is named by a non-empty string, got {ref!r}")
    if ref in BUILTIN_TASK_SAMPLERS:
        return import_from_string(BUILTIN_TASK_SAMPLERS[ref])
    if ref == "auto":
        raise ExtensionError(
            "auto names no class: the run's sampler resolves it (weighted for a mixture, the run's uniform passes for "
            "a single environment), and an entry's sampler cannot be auto"
        )
    registry = get_registry(REGISTRY_KIND)
    if ":" not in ref:
        if ref not in registry.names():
            known = [*BUILTIN_TASK_SAMPLERS, *registry.names()]
            close = difflib.get_close_matches(ref, known, n=1, cutoff=0.6)
            raise ExtensionError(
                f"unknown task sampler {ref!r}"
                + (f"; did you mean {close[0]}?" if close else ";")
                + " Core's: auto and weighted (run level), uniform and pass_rate (either level); a package an entry "
                "names registers others with @task_sampler, or name one as module:Class"
            )
        cls = registry.get(ref)
    else:
        try:
            cls = import_from_string(ref)
        except Exception as exc:
            raise ExtensionError(f"{ref} does not import ({type(exc).__name__}: {exc})") from exc
    sampler_problem(cls, ref)
    return cls


def sampler_problem(cls: Any, ref: str) -> None:
    """Raise :class:`~platoon.extensions.ExtensionError` when ``cls`` (what ``ref`` resolved to) is no task sampler
    class, or hands slots and is adaptive."""

    from platoon.extensions import ExtensionError

    methods = ("next", "state_dict", "load_state_dict")
    if (
        not isinstance(cls, type)
        or not all(callable(getattr(cls, method, None)) for method in methods)
        or not isinstance(getattr(cls, "adaptive", None), bool)
    ):
        raise ExtensionError(
            f"{ref} is not a task sampler: a class with next(ctx), state_dict() and load_state_dict(state), and "
            "adaptive set to True or False"
        )
    if getattr(cls, "entry_slots", False) is True and cls.adaptive:
        raise ExtensionError(
            f"{ref} hands slots to entries and is adaptive; in this release only a static run sampler (adaptive = "
            "False) hands slots, so set one of them to False"
        )
    if inspect.iscoroutinefunction(getattr(cls, "next")):
        raise ExtensionError(
            f"{ref} is not a task sampler: its next(ctx) is a coroutine, and a decision is one synchronous call"
        )


def hands_slots(cls: type) -> bool:
    """Whether a sampler class hands slots to the entries' samplers (``entry_slots``)."""

    return getattr(cls, "entry_slots", False) is True


def options_schema(cls: type) -> Any:
    """A sampler class's options schema (``options_schema``), None for one that takes no options."""

    return getattr(cls, "options_schema", None)
