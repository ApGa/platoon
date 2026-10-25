"""The built-in task samplers (RFC 0019, section 5), and what a config chooses at each level.

- :class:`WeightedSampler`, ``weighted`` (and ``auto`` for a mixture): it hands slot ``n`` of the run's order to the
  entry with the most credit, each entry with tasks gaining its share of the weights per slot and the chosen one paying
  1, ties rotating from the entry after the last one chosen; the entry's own sampler fills the slot.
- :class:`UniformSampler`, ``uniform``.  At the entry level, under ``weighted``: the entry's tasks in seeded shuffled
  passes, pass ``p`` of the ``i``-th entry with tasks being ``random.Random(seed + i * 10007 + p * 101).shuffle``, and
  decision ``n`` (the ``n``-th slot the entry received) the ``n``-th task of the passes.  At the run level (and
  ``auto`` for a single environment): every task of the pool in the passes ``Dataset.shuffle(42 + 101 p)`` gives.
- :class:`_ListingOrder` (internal): ``auto`` over a listing whose rows are all marked to keep their order, drawn as
  it is, the whole batches its rows fill each epoch.
- :class:`_PositionOrder` (internal): an order handed to the dispatcher, or the trainer's dataloader order when the
  trainer was given no sampling: decision ``n`` is the order's row ``n``.
- :class:`PassRateSampler`, ``pass_rate`` (experimental), at either level: an adaptive sampler that chooses, when a
  group can start, the task whose next group is most likely to mix passes and failures, by Thompson sampling on each
  task's posterior pass rate in the run's per-task statistics (:class:`PassRateOptions`).

Each static one is a function of its decision index and its setup, with caches only: ``state_dict()`` is ``{}``,
and a fresh instance asked at decision ``n`` answers what the first one answered.  They draw the rows platoon 0.3.0
drew: they repeat the arithmetic of its order classes (:class:`~platoon.train.components.WeightedOrder` with its
cycling pools, and the shuffled passes, :class:`~platoon.train.tinker.dataset_order.ShuffledPasses` and
``PoolPasses``), which stay as they are, the trainer still counting epochs with them, and which the tests hold the
samplers to.  The built-ins answer with a pool row's index (``_Row``), since a listing that repeats a task id is drawn
row by row.

What a config chooses (:class:`SamplerChoice`, :func:`choose_samplers`): ``train.task_sampler`` at the run level and,
under a run sampler that hands slots, each entry's ``task_sampler``, with their options parsed and each sampler's
``check`` run.  The checks a config's samplers get when it loads (the functions its classes' ``VALUE_PROBLEMS`` call,
and :func:`config_sampler_problems`) are here too.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from platoon.train.sampler_api import (
    Draw,
    Sampler,
    SamplerCheckContext,
    SamplerContext,
    SamplerSetup,
    Slot,
    TaskStatsView,
    hands_slots,
    options_schema,
    resolve_task_sampler,
)
from platoon.train.sampler_runtime import _Row
from platoon.train.task_stats import NO_EVIDENCE_LIMIT, informative_at

logger = logging.getLogger(__name__)

# platoon 0.3.0's seeds, which the built-ins keep so that they draw its rows: a single environment's shuffled passes
# and uniform's are Dataset.shuffle(42 + 101 p) (SHUFFLE_SEED and PASS_SEED_STRIDE of
# platoon.train.tinker.dataset_order), and an entry's passes under weighted random.Random(seed + 10007 i + 101 p)
# (platoon.train.components.WeightedOrder).
SHUFFLE_SEED = 42
PASS_SEED_STRIDE = 101
ENTRY_SEED_STRIDE = 10_007


# Core's samplers at each level, as a config names them: the run's (train.task_sampler) and an entry's.
RUN_SAMPLERS = ("auto", "weighted", "uniform", "pass_rate")
ENTRY_SAMPLERS = ("uniform", "pass_rate")
# Core's static run samplers: they take no options, and draw what platoon 0.3.0's orders drew (a dataloader's order).
STATIC_RUN_SAMPLERS = ("auto", "weighted", "uniform")


@dataclass(frozen=True)
class SamplerChoice:
    """What a config chooses at one level, resolved: the sampler's name as the config gives it (``auto``,
    ``weighted``, ``uniform``, ``pass_rate``, a registered name, ``module:Class``), its class, its options parsed
    (None: it takes none), and the options as the config wrote them."""

    name: str
    cls: type
    options: Any = None
    written: Mapping[str, Any] = field(default_factory=dict)

    @property
    def adaptive(self) -> bool:
        return getattr(self.cls, "adaptive", False) is True

    @property
    def hands_slots(self) -> bool:
        return hands_slots(self.cls)

    @property
    def core(self) -> bool:
        """Whether it is one of core's samplers."""

        return self.cls in _core_sampler_classes()

    def options_json(self) -> dict[str, Any]:
        """Its options as ``lock.json`` and the sampling fingerprint record them: parsed, defaults included."""

        from platoon.options import schema_options

        return schema_options(options_schema(self.cls), name=self.name).as_json(self.options, self.written)


@dataclass(frozen=True)
class SamplingChoices:
    """The samplers a config chooses (:func:`choose_samplers`): the run's, and each entry's that the run's sampler
    hands slots to, by label in config order (none for a run sampler that hands none)."""

    run: SamplerChoice
    entries: Mapping[str, SamplerChoice]


class WeightedSampler(Sampler):
    """``weighted``, and ``auto`` for a mixture: decision ``n`` is ``Slot(entry)`` for the entry 0.3.0's credit rule
    picks at slot ``n`` (see the module docstring), by the same float operations in the same order.  It keeps the
    credits after its last decision and goes on from them; a decision before it replays the slots from the first."""

    entry_slots = True

    def __init__(self, setup: SamplerSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self._labels = list(setup.entries)
        if not self._labels:
            raise ValueError("Cannot interleave a mixture whose environments have no tasks")
        weights = [
            float(info.weight) if info.weight is not None else float(len(info.task_ids))
            for info in setup.entries.values()
        ]
        total = sum(weights)
        self._shares = [weight / total for weight in weights]
        # The decision the credits are ready for, the credits and the cursor; and the last answer.
        self._next = 0
        self._credits = [0.0] * len(self._labels)
        self._cursor = 0
        self._last: tuple[int, int] | None = None

    def next(self, ctx: SamplerContext) -> Slot:
        return Slot(self._labels[self._chosen(ctx.decision)])

    def _chosen(self, decision: int) -> int:
        if self._last is not None and self._last[0] == decision:
            return self._last[1]
        if decision < self._next:
            self._next, self._credits, self._cursor = 0, [0.0] * len(self._labels), 0
        count = len(self._labels)
        credits, cursor, chosen = self._credits, self._cursor, 0
        while self._next <= decision:
            credits = [credit + share for credit, share in zip(credits, self._shares)]
            best = max(credits)
            chosen = next(
                index
                for offset in range(count)
                if math.isclose(credits[index := (cursor + offset) % count], best, rel_tol=0.0, abs_tol=1e-12)
            )
            credits[chosen] -= 1.0
            cursor = (chosen + 1) % count
            self._next += 1
        self._credits, self._cursor = credits, cursor
        self._last = (decision, chosen)
        return chosen


class UniformSampler(Sampler):
    """``uniform``: its pool in seeded shuffled passes, decision ``n`` being the task at offset ``n mod N`` of pass
    ``n // N`` (``N`` tasks).  An entry's passes are ``random.Random(seed + i * 10007 + p * 101)``'s shuffles of its
    tasks (``i`` its index among the entries with tasks); the run's are ``Dataset.shuffle(42 + p * 101)``'s order of
    the pool (``datasets`` is imported when a pass is shuffled)."""

    def __init__(self, setup: SamplerSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self._count = len(setup.task_ids)
        if not self._count:
            raise ValueError(f"task sampler {setup.name} has no task to draw")
        self._passes: dict[int, list[int]] = {}
        if setup.level == "entry":
            info = setup.entries.get(setup.entry) if setup.entry is not None else None
            index = info.index if info is not None and info.index is not None else 0
            self._seed = setup.seed + index * ENTRY_SEED_STRIDE
            self._shuffle: Callable[[int], list[int]] = self._entry_pass
        else:
            self._seed = SHUFFLE_SEED
            self._shuffle = self._run_pass

    def next(self, ctx: SamplerContext) -> _Row:
        number, offset = divmod(ctx.decision, self._count)
        return _Row(self._pass(number)[offset])

    def _pass(self, number: int) -> list[int]:
        if number not in self._passes:
            if len(self._passes) >= 2:  # the dispatcher asks forward: the pass it is in, and the one before
                self._passes.pop(min(self._passes))
            self._passes[number] = self._shuffle(number)
        return self._passes[number]

    def _entry_pass(self, number: int) -> list[int]:
        order = list(range(self._count))
        random.Random(self._seed + number * PASS_SEED_STRIDE).shuffle(order)
        return order

    def _run_pass(self, number: int) -> list[int]:
        from datasets import Dataset

        shuffled = Dataset.from_dict({"index": list(range(self._count))}).shuffle(
            seed=self._seed + number * PASS_SEED_STRIDE
        )
        return [int(index) for index in shuffled[0 : self._count]["index"]]


class _ListingOrder(Sampler):
    """``auto`` over a listing whose rows are all marked to keep their order: the listing as it is, its first ``E``
    rows every epoch, ``E`` the whole batches they fill (0.3.0's ``FixedOrder``)."""

    def __init__(self, setup: SamplerSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self._epoch = len(setup.task_ids) // setup.batch_size * setup.batch_size
        if self._epoch < 1:
            raise ValueError(
                f"batch_size {setup.batch_size} is more than the listing's {len(setup.task_ids)} rows, so no full "
                "batch exists"
            )

    def next(self, ctx: SamplerContext) -> _Row:
        return _Row(ctx.decision % self._epoch)


class _PositionOrder(Sampler):
    """An order handed to the dispatcher directly, or the trainer's dataloader order when it was given no sampling:
    decision ``n`` is ``order(n)``'s row, which 0.3.0's dispatcher drew at position ``n``."""

    def __init__(self, setup: SamplerSetup, options: Any = None, *, order: Callable[[int], Mapping[str, Any]]) -> None:
        super().__init__(setup, options)
        self._order = order

    def next(self, ctx: SamplerContext) -> _Row:
        return _Row(None, row=self._order(ctx.decision))


# --- pass_rate ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PassRateOptions:
    """``pass_rate``'s options (``task_sampler_options``).  The rest is the per-task statistics' (RFC 0019, section
    3.3): a half-life of 32 policy versions, a prior of weight 2 at the entry's decayed pass rate, members without a
    valid result censored, and a task whose last 3 groups had no valid member drawn by exploration only."""

    # The share of decisions that go to the eligible task drawn least recently (never drawn first), from 0 to 1.
    explore: float = 0.2
    # The most draws of a task in a window of as many decisions of the sampler as its pool has tasks, from 1; lifted
    # for a decision where every candidate is at it; null: no cap.
    max_draws_per_epoch: int | None = 3
    # A task with this many draws that no committed step has consumed yet is not drawn, from 1; when every candidate
    # is at it, those with the fewest stay; null: no limit.
    max_outstanding_per_task: int | None = None
    # At most this many eligible tasks are scored by a decision that exploits, from 1; a seeded sample of them beyond.
    candidates: int = 4096
    # Added to the seed its decisions' randomness follows from (at the entry level, the entry's seed).
    seed: int = 0

    def __post_init__(self) -> None:
        explore = self.explore
        if (
            isinstance(explore, bool)
            or not isinstance(explore, (int, float))
            or not math.isfinite(explore)
            or not 0 <= explore <= 1
        ):
            raise ValueError(f"explore: must be between 0 and 1, got {explore!r}")
        for name in ("max_draws_per_epoch", "max_outstanding_per_task"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name}: must be an integer from 1, or null for no limit, got {value!r}")
        if isinstance(self.candidates, bool) or not isinstance(self.candidates, int) or self.candidates < 1:
            raise ValueError(f"candidates: must be an integer from 1, got {self.candidates!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed: must be an integer, got {self.seed!r}")


def _check_pass_rate(options: Any, ctx: SamplerCheckContext) -> None:
    """``pass_rate`` ranks tasks by the chance that a group mixes passes and failures, which a group of one never
    does: refuse a pool with an entry of group size 1."""

    entries = ctx.entries if ctx.level == "run" else {ctx.entry: ctx.entries.get(ctx.entry, {})}
    for label, entry in entries.items():
        if entry.get("group_size") == 1:
            whose = "its group size is 1" if ctx.level == "entry" else f"environments[{label}]'s group size is 1"
            raise ValueError(
                "pass_rate chooses the tasks whose next group is likely to mix passes and failures, which a group "
                f"of one never does ({whose}); use uniform, or a group size of 2 or more"
            )


class PassRateSampler(Sampler):
    """``pass_rate`` (experimental): the task whose next group of ``G`` is most likely to mix passes and failures,
    where it trains.  Each decision, among the eligible tasks of its pool (their label has room):

    1. an unseen task (no settled group) with a draw outstanding waits for its first result; a task with
       ``max_outstanding_per_task`` draws outstanding is left out; a task drawn ``max_draws_per_epoch`` times in the
       current window (as many of this sampler's decisions as its pool has tasks) is left out.  Each rule that would
       leave nothing is lifted for the decision (the first two keep the tasks with the fewest draws outstanding).
    2. With probability ``explore``, the task drawn least recently (reason ``explore``); otherwise Thompson sampling
       (reason ``exploit``): for each task with evidence (at most ``candidates`` of them, a seeded sample beyond), a
       pass rate ``p`` drawn from its posterior and scored ``1 - p^G - (1 - p)^G``, the best one drawn.

    The draw's score is the chosen task's ``E[informative_G]`` under its posterior.  It reads each task's counts
    (:meth:`TaskStatsView.counts`) and the posteriors of the tasks it scores, never a whole :class:`TaskStats`, so a
    decision costs one pass over the pool.  Without statistics (they stopped, or never started) it explores only, and
    warns once."""

    adaptive = True
    options_schema = PassRateOptions
    check = staticmethod(_check_pass_rate)

    def __init__(self, setup: SamplerSetup, options: Any = None) -> None:
        if options is None:
            options = PassRateOptions()
        elif isinstance(options, Mapping):
            options = PassRateOptions(**dict(options))
        super().__init__(setup, options)
        # Its tasks, each once, in listing order: a listing that repeats a task id draws its first row.
        self._ids = tuple(dict.fromkeys(setup.task_ids))
        self._count = len(self._ids)
        if not self._count:
            raise ValueError(f"task sampler {setup.name} has no task to draw")
        own = setup.entries.get(setup.entry) if setup.entry is not None else None
        self._entry_size = own.group_size if own is not None else None
        self._sizes: dict[str, int] = {}
        for info in setup.entries.values():
            for task in info.task_ids:
                self._sizes.setdefault(task, info.group_size)
        # The decision of each task's last draw; the window, its draws by task, and the window before's.
        self._last_drawn: dict[str, int] = {}
        self._window = 0
        self._window_draws: dict[str, int] = {}
        self._previous: dict[str, int] = {}
        self._warned = False
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self._decisions = 0
        self._eligible = 0.0
        self._lifted = 0
        self._prior = [0.0, 0]
        self._age = [0.0, 0]

    def _size(self, task: str, stats: TaskStatsView | None) -> int:
        if self._entry_size is not None:
            return self._entry_size
        size = self._sizes.get(task)
        if size is None:
            size = stats.group_size(None) if stats is not None else 1
        return size

    def next(self, ctx: SamplerContext) -> Draw | None:
        options: PassRateOptions = self.options
        window = ctx.decision // self._count
        if window != self._window:
            self._previous = self._window_draws if window == self._window + 1 else {}
            self._window_draws, self._window = {}, window
        eligible = ctx.eligible
        stats = ctx.stats
        live = stats is not None and stats.live
        pool = [task for task in self._ids if eligible(task)]
        if not pool:
            return None
        # Of each eligible task, what the rules read of its counts (the cheap read), kept as plain numbers: its draws
        # outstanding, whether it is unseen with a draw outstanding (it waits for its first result), and whether its
        # last groups had no valid member.  Keeping the counts themselves would keep an object per task alive through
        # the decision, which the garbage collector would trace and promote on a large pool.
        outstanding: dict[str, int] | None = None
        waits: set[str] = set()
        no_evidence: set[str] = set()
        if stats is not None:
            outstanding, read = {}, stats.counts
            for task in pool:
                try:
                    counts = read(task)
                except KeyError:  # a task the statistics do not hold reads as never drawn
                    outstanding[task] = 0
                    continue
                busy = outstanding[task] = counts[1]
                if busy and not counts[0]:
                    waits.add(task)
                if counts[4] >= NO_EVIDENCE_LIMIT:
                    no_evidence.add(task)
        if not live and not self._warned:
            self._warned = True
            logger.warning(
                "Task sampler %s: the run's per-task statistics are not available (they stopped, or never started), "
                "so it draws by exploration alone, the task drawn least recently first",
                self.setup.name,
            )
        if outstanding is not None:
            # An unseen task (no settled group) with a draw outstanding waits for its first result.
            if waits:
                pool = _unless_empty(pool, [task for task in pool if task not in waits], outstanding)
            limit = options.max_outstanding_per_task
            if limit is not None:
                pool = _unless_empty(pool, [task for task in pool if outstanding[task] < limit], outstanding)
        lifted = False
        cap = options.max_draws_per_epoch
        if cap is not None:
            draws = self._window_draws
            capped = [task for task in pool if draws.get(task, 0) < cap]
            if capped:
                pool = capped
            else:
                lifted = True
        rng = ctx.rng
        chosen: str | None = None
        reason = "explore"
        if live and rng.random() >= options.explore:
            scored = [task for task in pool if task not in no_evidence] if no_evidence else pool
            if scored:
                reason = "exploit"
                if len(scored) > options.candidates:
                    scored = rng.sample(scored, options.candidates)
                chosen = self._thompson(scored, stats, rng)
        if chosen is None:
            last = self._last_drawn
            oldest = min(last.get(task, -1) for task in pool)
            ties = [task for task in pool if last.get(task, -1) == oldest]
            chosen = ties[0] if len(ties) == 1 else ties[rng.randrange(len(ties))]
        score = None
        if stats is not None:
            alpha, beta = stats.posterior(chosen)
            score = informative_at(alpha, beta, self._size(chosen, stats))
            self._prior[0] += 2.0 / (alpha + beta)
            self._prior[1] += 1
            try:
                version = stats.counts(chosen)[5]  # the version of its newest evidence, None before any
            except KeyError:
                version = None
            if version is not None:
                self._age[0] += max(0, stats.now - version)
                self._age[1] += 1
        self._decisions += 1
        self._eligible += len(pool) / self._count
        self._lifted += lifted
        self._last_drawn[chosen] = ctx.decision
        self._window_draws[chosen] = self._window_draws.get(chosen, 0) + 1
        return Draw(chosen, reason, score)

    def _thompson(self, tasks: Sequence[str], stats: Any, rng: random.Random) -> str:
        """The task whose pass rate, drawn from its posterior, scores best: ``1 - p^G - (1 - p)^G``; ties by ``rng``."""

        posterior, beta_draw = stats.posterior, rng.betavariate
        fixed = self._entry_size  # an entry's tasks share its group size; over every task, each its entry's
        sizes, default = self._sizes, stats.group_size(None)
        best, ties = -1.0, []
        for task in tasks:
            alpha, beta = posterior(task)
            p = beta_draw(alpha, beta)
            size = fixed if fixed is not None else sizes.get(task, default)
            score = 1.0 - p**size - (1.0 - p) ** size
            if score > best:
                best, ties = score, [task]
            elif score == best:
                ties.append(task)
        return ties[0] if len(ties) == 1 else ties[rng.randrange(len(ties))]

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_drawn": dict(self._last_drawn),
            "window": self._window,
            "window_draws": dict(self._window_draws),
            "previous_window_draws": dict(self._previous),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Take up a saved state: tasks that left the pool are dropped, and new ones start as never drawn.  ``{}``
        starts afresh; anything else it cannot read is a ``ValueError``."""

        if not state:
            return
        if not isinstance(state, Mapping):
            raise ValueError(f"pass_rate's state is a mapping, got {type(state).__name__}")
        try:
            last = _counts_of(state["last_drawn"], "last_drawn")
            window = state["window"]
            draws = _counts_of(state["window_draws"], "window_draws")
            previous = _counts_of(state["previous_window_draws"], "previous_window_draws")
        except KeyError as exc:
            raise ValueError(f"pass_rate's state has no {exc.args[0]!r}") from None
        if isinstance(window, bool) or not isinstance(window, int) or window < 0:
            raise ValueError(f"pass_rate's state: window is a decision window, 0 or more, got {window!r}")
        known = set(self._ids)
        self._last_drawn = {task: value for task, value in last.items() if task in known}
        self._window = window
        self._window_draws = {task: value for task, value in draws.items() if task in known}
        self._previous = {task: value for task, value in previous.items() if task in known}

    def metrics(self, stats: TaskStatsView | None) -> Mapping[str, float]:
        """Over its decisions since the last step: ``eligible_share`` (the share of its pool a decision could choose
        from, after the rules), ``cap_lifted`` (the share of decisions where every candidate was at the window's cap),
        ``prior_share`` (``2 / (alpha + beta)`` of the drawn tasks' posteriors, how much the prior weighs) and
        ``evidence_age`` (the policy versions since the drawn tasks' newest evidence, those with evidence); and
        ``top10_share``, the share of the window's draws that went to the most-drawn tenth of its pool (at least one
        task), the window before's while the current one holds fewer than half a pool of draws."""

        found: dict[str, float] = {}
        current = self._window_draws
        draws = current if sum(current.values()) * 2 >= self._count or not self._previous else self._previous
        total = sum(draws.values())
        if total:
            top = sorted(draws.values(), reverse=True)[: max(1, self._count // 10)]
            found["top10_share"] = sum(top) / total
        if self._decisions:
            found["eligible_share"] = self._eligible / self._decisions
            found["cap_lifted"] = self._lifted / self._decisions
            if self._prior[1]:
                found["prior_share"] = self._prior[0] / self._prior[1]
            if self._age[1]:
                found["evidence_age"] = self._age[0] / self._age[1]
        self._reset_metrics()
        return found


def _unless_empty(pool: list[str], kept: list[str], outstanding: Mapping[str, int]) -> list[str]:
    """``kept``, or, when a rule would leave nothing, the tasks of ``pool`` with the fewest draws outstanding."""

    if kept:
        return kept
    fewest = min(outstanding[task] for task in pool)
    return [task for task in pool if outstanding[task] == fewest]


def _counts_of(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not all(
        isinstance(task, str) and isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for task, count in value.items()
    ):
        raise ValueError(f"pass_rate's state: {name} maps task ids to counts, got {value!r}")
    return dict(value)


# --- what a config chooses -------------------------------------------------------------------------------------------


def _core_sampler_classes() -> tuple[type, ...]:
    return (WeightedSampler, UniformSampler, PassRateSampler, _ListingOrder, _PositionOrder)


def _packages(root: Any) -> bool:
    """Whether some environment entry of the config as given names a ``package``, which may register samplers."""

    environments = root.get("environments") if isinstance(root, Mapping) else None
    return isinstance(environments, list) and any(
        isinstance(entry, Mapping) and entry.get("package") for entry in environments
    )


def _unknown_name(name: str, core: Sequence[str], packages: bool) -> bool:
    """Whether a sampler name names nothing that could resolve: empty, or not core's, not ``module:Class``, while no
    entry names a package."""

    return not name or (name not in core and ":" not in name and not packages)


def _not_one_of(path: str, name: str, core: Sequence[str]) -> str:
    return (
        f"{path}: {name!r} is not one of {', '.join(map(repr, core))}, a name an environment's package registers, "
        "or module:Class"
    )


def _entry_names_a_run_sampler(path: str, name: str) -> str:
    return (
        f"{path}: {name} shares a run's slots among its entries; an entry's sampler chooses among the entry's own "
        "tasks: uniform, pass_rate, a name a package registers, or module:Class"
    )


def _options_problems(name: str | None, values: Any, path: str) -> list[str]:
    """The problems of the options ``values`` of core's sampler ``name`` (none for another sampler's, which are checked
    where it resolves)."""

    from platoon.options import schema_options

    if name in STATIC_RUN_SAMPLERS:
        return schema_options(None, name=name).option_problems(values, path)
    if name == "pass_rate":
        return schema_options(PassRateOptions, name=name).option_problems(values, path)
    return []


def run_sampler_problems(value: Any, train: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """``train.task_sampler``'s problems when the config loads (``TrainConfig.VALUE_PROBLEMS``): a name that nothing
    could resolve, while no entry names a package that could register it."""

    if not isinstance(value, str) or not _unknown_name(value, RUN_SAMPLERS, _packages(root)):
        return []
    return [_not_one_of(path, value, RUN_SAMPLERS)]


def run_options_problems(value: Any, train: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """``train.task_sampler_options``' problems when the config loads: options for a built-in sampler that takes none,
    and pass_rate's checked against :class:`PassRateOptions`."""

    if not isinstance(value, Mapping):
        return []
    name = train.get("task_sampler") if isinstance(train, Mapping) else None
    if name is None:
        name = "auto"
    return _options_problems(name, value, path) if isinstance(name, str) else []


def entry_sampler_problems(value: Any, entry: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """An entry's ``task_sampler``'s problems when the config loads (``EnvironmentConfig.VALUE_PROBLEMS``): a run
    sampler's name (``auto``, ``weighted``), and a name that nothing could resolve, while no entry names a package."""

    if not isinstance(value, str):
        return []
    if value in ("auto", "weighted"):
        return [_entry_names_a_run_sampler(path, value)]
    if _unknown_name(value, ENTRY_SAMPLERS, _packages(root)):
        return [_not_one_of(path, value, ENTRY_SAMPLERS)]
    return []


def entry_options_problems(value: Any, entry: Mapping[str, Any], root: Any, path: str) -> list[str]:
    """An entry's ``task_sampler_options``' problems when the config loads: options for ``uniform`` (its default),
    which takes none, and pass_rate's checked against :class:`PassRateOptions`."""

    if not isinstance(value, Mapping):
        return []
    name = entry.get("task_sampler") if isinstance(entry, Mapping) else None
    name = "uniform" if name is None else name
    return _options_problems(name, value, path) if isinstance(name, str) and name in ENTRY_SAMPLERS else []


def config_sampler_problems(config: Any) -> list[str]:
    """What a loaded (or built) config's task samplers refuse, one line each: under ``train.task_sampler: uniform`` or
    ``pass_rate``, which draw among every training task of the run with no entry's share and hand no slots to the
    entries, an entry's numeric ``weight`` and an entry's own ``task_sampler`` or ``task_sampler_options``; and the
    checks the loader makes of the names and the built-ins' options, again for a config built in code, which never
    passes the loader."""

    train = getattr(config, "train", None)
    environments = list(getattr(config, "environments", None) or [])
    run = str(getattr(train, "task_sampler", None) or "auto")
    packages = any(getattr(environment, "package", None) for environment in environments)
    problems: list[str] = []
    if train is not None:
        if _unknown_name(run, RUN_SAMPLERS, packages):
            problems.append(_not_one_of("train.task_sampler", run, RUN_SAMPLERS))
        else:
            problems += _options_problems(
                run, getattr(train, "task_sampler_options", None) or {}, "train.task_sampler_options"
            )
    for index, environment in enumerate(environments):
        name = getattr(environment, "task_sampler", None)
        options = getattr(environment, "task_sampler_options", None) or {}
        where = f"environments.{index}"
        if isinstance(name, str) and name in ("auto", "weighted"):
            problems.append(_entry_names_a_run_sampler(f"{where}.task_sampler", name))
        elif isinstance(name, str) and _unknown_name(name, ENTRY_SAMPLERS, packages):
            problems.append(_not_one_of(f"{where}.task_sampler", name, ENTRY_SAMPLERS))
        elif name is None or name in ENTRY_SAMPLERS:
            problems += _options_problems(name or "uniform", options, f"{where}.task_sampler_options")
    if run not in ("uniform", "pass_rate"):
        return problems
    for environment in environments:
        label = environment.label
        if environment.weight != "tasks":
            if run == "uniform":
                problems.append(
                    f"environments[{label}].weight: {environment.weight}: train.task_sampler: uniform draws every "
                    "training task of every entry equally often, with no entry's share, so it takes no weights; leave "
                    "weight out (weight: tasks, the default, is uniform's own share), or set train.task_sampler: "
                    "weighted to share the slots by weight"
                )
            else:
                problems.append(
                    f"environments[{label}].weight: {environment.weight}: train.task_sampler: pass_rate chooses among "
                    "every training task of the run by its statistics, with no entry's share, so it takes no weights; "
                    "leave weight out, or keep train.task_sampler: weighted and set task_sampler: pass_rate on the "
                    "entries, which keeps each entry's share"
                )
        for key, value in (
            ("task_sampler", getattr(environment, "task_sampler", None)),
            ("task_sampler_options", getattr(environment, "task_sampler_options", None)),
        ):
            if value:
                given = f" {value}:" if key == "task_sampler" else ""
                problems.append(
                    f"environments[{label}].{key}:{given} train.task_sampler: {run} draws every training task of the "
                    "run itself, with no entry slots, so it asks no entry's sampler; leave it out, or set "
                    "train.task_sampler: weighted to give each entry its share of the slots and let its own sampler "
                    "fill them"
                )
    return problems


def single_environment_order(config: Any) -> bool:
    """Whether ``auto`` draws a single environment's listing itself (the run's ``uniform`` passes, or the listing as
    it is), rather than as a mixture of one: one entry, whose own ``task_sampler`` is unset or ``uniform``."""

    environments = list(getattr(config, "environments", None) or [])
    return len(environments) == 1 and getattr(environments[0], "task_sampler", None) in (None, "uniform")


def choose_samplers(config: Any) -> SamplingChoices:
    """The task samplers ``config`` chooses, resolved (core's names first, then names the entries' packages register,
    then ``module:Class``), each class checked, its options parsed against its schema and its ``check`` run, as
    ``platoon train`` and ``platoon config validate`` do before anything runs: the run's (``auto`` is ``weighted`` for
    a mixture and the run's ``uniform`` for a single environment's listing), and, when it hands slots, each entry's
    (``uniform`` by default).  A :class:`~platoon.utils.config.ConfigError` with every problem, each with its path.
    The entries' packages must be imported first, as a run imports them."""

    from platoon.extensions import ExtensionError
    from platoon.options import schema_options
    from platoon.utils.config import ConfigError

    train = getattr(config, "train", None)
    environments = list(getattr(config, "environments", None) or [])
    run_name = str(getattr(train, "task_sampler", None) or "auto")
    problems: list[str] = []

    def resolve(name: str, where: str) -> type | None:
        if name == "auto":
            return UniformSampler if single_environment_order(config) else WeightedSampler
        try:
            return resolve_task_sampler(name)
        except ExtensionError as exc:
            problems.append(f"{where}: {exc}")
            return None

    def parse(name: str, cls: type, values: Any, where: str) -> Any:
        parsed, found = schema_options(options_schema(cls), name=name).check(values, where)
        problems.extend(found)
        return parsed

    contexts = _check_entries(config, environments)
    batch_size = int(getattr(train, "batch_size", 0) or 0)
    column = _label_column(config)

    def check(choice: SamplerChoice, level: str, entry: str | None, where: str) -> None:
        method = getattr(choice.cls, "check", None)
        if not callable(method):
            return
        ctx = SamplerCheckContext(
            level=level,  # type: ignore[arg-type]
            entry=entry,
            run_sampler=run_name,
            entries=contexts,
            batch_size=batch_size,
            label_column=column,
        )
        try:
            method(choice.options, ctx)
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            problems.append(f"{where}: {message if message.startswith(choice.name) else f'{choice.name}: {message}'}")

    run_written = dict(getattr(train, "task_sampler_options", None) or {})
    run_cls = resolve(run_name, "train.task_sampler")
    run: SamplerChoice | None = None
    if run_cls is not None:
        options = parse(run_name, run_cls, run_written, "train.task_sampler_options")
        run = SamplerChoice(run_name, run_cls, options, run_written)
        if not problems:
            check(run, "run", None, "train.task_sampler")
    entries: dict[str, SamplerChoice] = {}
    for index, environment in enumerate(environments):
        label = environment.label
        name = getattr(environment, "task_sampler", None)
        written = dict(getattr(environment, "task_sampler_options", None) or {})
        where = f"environments[{label}].task_sampler"
        if run is None:
            continue
        if not run.hands_slots:
            if (name or written) and not run.core:
                given = f" {name}:" if name else ""
                problems.append(
                    f"{where}:{given} train.task_sampler: {run_name} hands no slots to entries (its entry_slots is "
                    "False), so it asks no entry's sampler; leave it out"
                )
            continue
        name = name or "uniform"
        cls = resolve(name, where)
        if cls is None:
            continue
        if hands_slots(cls):
            problems.append(f"{where}: {name} hands slots to entries, which only the run's sampler does")
            continue
        before = len(problems)
        options = parse(name, cls, written, f"environments.{index}.task_sampler_options")
        choice = SamplerChoice(name, cls, options, written)
        if len(problems) == before:
            check(choice, "entry", label, where)
        entries[label] = choice
    if problems or run is None:
        raise ConfigError(problems)
    return SamplingChoices(run=run, entries=entries)


def _check_entries(config: Any, environments: Sequence[Any]) -> dict[str, dict[str, Any]]:
    """What a sampler's ``check`` sees of each entry, by label in config order (:class:`SamplerCheckContext`)."""

    from platoon.train import entry_settings

    try:
        settings = entry_settings.run_settings(config)
    except Exception:  # a stand-in config: the run's group size
        settings = {}
    run_size = getattr(getattr(getattr(config, "train", None), "workflow_config", None), "group_size", None)
    found: dict[str, dict[str, Any]] = {}
    for environment in environments:
        own = getattr(settings.get(environment.label), "train", None)
        size = getattr(own, "group_size", None) or run_size
        found[environment.label] = {
            "weight": environment.weight,
            "seed": environment.seed,
            "group_size": None if size is None else int(size),
            "success_threshold": float(getattr(environment, "success_threshold", 1.0)),
            "task_sampler": getattr(environment, "task_sampler", None),
            "task_sampler_options": dict(getattr(environment, "task_sampler_options", None) or {}),
        }
    return found


def _label_column(config: Any) -> str:
    from platoon.train.labels import label_column

    try:
        return label_column(config)
    except Exception:
        from platoon.train.components import ENVIRONMENT_COLUMN

        return ENVIRONMENT_COLUMN


def sampler_lock_records(config: Any, choices: SamplingChoices) -> list[dict[str, Any]]:
    """What ``lock.json`` records of the run's task samplers, one record per sampler (``kind: task_sampler``): the
    reference as the config gives it, what it resolved to, the distribution and version that provide it, its level,
    the entries it chooses for, and its options with their defaults (the run's under ``train``, an entry's under its
    label)."""

    from platoon.extensions import origin_of_module

    def describe(choice: SamplerChoice, level: str) -> dict[str, Any]:
        cls = choice.cls
        source = f"{cls.__module__}:{cls.__qualname__}"
        origin = origin_of_module(cls.__module__)
        return {
            "kind": "task_sampler",
            "ref": choice.name,
            "resolved": source,
            "source": source,
            "distribution": origin.distribution,
            "distribution_version": origin.version,
            "entry_point": None,
            "level": level,
            "environments": [],
            "options": {},
        }

    environments = [environment.label for environment in getattr(config, "environments", None) or []]
    run = describe(choices.run, "run")
    run["environments"] = list(environments)
    run["options"] = {"train": choices.run.options_json()}
    records = [run]
    by_ref: dict[tuple[str, str], dict[str, Any]] = {}
    for label, choice in choices.entries.items():
        key = (choice.name, f"{choice.cls.__module__}:{choice.cls.__qualname__}")
        if key not in by_ref:
            by_ref[key] = describe(choice, "entry")
            records.append(by_ref[key])
        by_ref[key]["environments"].append(label)
        by_ref[key]["options"][label] = choice.options_json()
    return records
