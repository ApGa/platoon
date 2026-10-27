"""What the dispatcher asks when it draws a training task: the run's task samplers over the run's task pool (RFC 0019,
section 5).

- :class:`TaskPool` holds the rows a run draws, by index in listing order, with each index's task id, entry and label.
  ``row(index)`` returns that row as platoon 0.3.0 drew it: for a pool backed by a dataset, a fresh conversion of the
  dataset's row each time, as 0.3.0's shuffled passes converted every epoch afresh; for a pool of rows, a shallow copy,
  as the dataloader's batches were.
- :class:`TaskSampling` is how a run draws, as :meth:`platoon.train.auto.AutoDataset.training` resolves it for
  ``platoon train``: the pool, the run's sampler, the sampler of each entry the run's sampler hands slots to, and each
  entry with training tasks.
- :class:`SamplerSet` holds the sampler instances, one for the run and one for each entry it hands slots to, with
  each one's decision count; it builds each decision's context, checks each answer, and gives a decision that fails a
  fallback draw.  It saves the counts and the samplers' states with the dispatcher's state, takes them up at a resume,
  and recomputes them from a position alone when a checkpoint saved none (a static sampler is a function of its
  decision count).

A static run sampler's decision ``n`` is the draw at position ``n`` of the run's order, and an entry's static sampler
under ``weighted`` counts the slots its entry received: so a resume from a position alone continues every static
sampler where an uninterrupted run would stand.  An adaptive sampler is asked when a group can start
(:meth:`SamplerSet.decide`): it sees the run's per-task statistics through a read-only view, built after the recorder
has taken in every settled group (which the set's listener hands, in draw order, to the adaptive samplers'
``observe``), and only the tasks whose label has room are eligible; its decision count advances with each draw it
binds, and a resume takes it up from the dispatcher's state, or starts it afresh.

**When a decision fails.**  A sampler that raises, returns nothing, or returns something outside the contract (a task
outside its pool, a slot it may not give, a reason or a score that cannot be recorded) gets a fallback draw: of its
pool in listing order (an adaptive sampler's eligible tasks), the task at offset ``key mod`` their number, or, for a
sampler that hands slots, a slot to an entry chosen by the decision's key with probability proportional to the
entry's weight.  The fallback keeps no state, so a resumed run makes the fallbacks an uninterrupted one makes; it uses
the decision up, so a static sampler that fails at one decision loses that one draw.  Each kind of failure is logged
once per sampler, and every fallback is counted.  Core's static samplers are the exception: they cannot fail on a valid
setup, and what they raise (an order handed to the trainer that fails, say) stops the run, as platoon 0.3.0's orders
did.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

from platoon.train.components import ENVIRONMENT_COLUMN, row_entry
from platoon.train.labels import label_text, metric_prefix, task_label
from platoon.train.sampler_api import (
    MAX_REASON_LENGTH,
    RESERVED_METRICS,
    Draw,
    Sampler,
    SamplerContext,
    SamplerError,
    SamplerSetup,
    Slot,
    TaskStatsView,
    decision_key,
    reserved_metric,
)
from platoon.train.task_stats import MAIN, EntryInfo, TaskStatsStore

if TYPE_CHECKING:
    from platoon.train.samplers import SamplerChoice

logger = logging.getLogger(__name__)


# --- the pool -------------------------------------------------------------------------------------------------------


class TaskPool:
    """The rows a run draws, by index in listing order (see the module docstring): ``task_ids``, ``entries`` and
    ``labels`` give each row's task id, environment entry and label, and the rows are ``rows`` or ``dataset``'s.  Build
    one with :meth:`from_rows` or :meth:`from_dataset`."""

    def __init__(
        self,
        task_ids: Iterable[str],
        entries: Iterable[str | None],
        labels: Iterable[str | None],
        *,
        rows: Sequence[Mapping[str, Any]] | None = None,
        dataset: Any = None,
        drop_columns: Iterable[str] = (),
    ) -> None:
        self._task_ids = tuple(task_ids)
        self._entries = tuple(entries)
        self._labels = tuple(labels)
        if not len(self._task_ids) == len(self._entries) == len(self._labels):
            raise ValueError("a task pool needs a task id, an entry and a label for each row")
        self._rows = rows
        self._dataset = dataset
        self._drop = frozenset(drop_columns)
        by_entry: dict[str, list[int]] = {}
        for index, entry in enumerate(self._entries):
            if entry is not None:
                by_entry.setdefault(entry, []).append(index)
        self._by_entry = {entry: tuple(indices) for entry, indices in by_entry.items()}
        self._first: dict[str, int] | None = None

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        entries: Sequence[str],
        column: str,
        drop_columns: Iterable[str] = (),
        row_entries: Iterable[str | None] | None = None,
    ) -> TaskPool:
        """A pool of ``rows`` as they are (each drawn as a shallow copy, without ``drop_columns``), of a run whose
        entries are ``entries`` (their labels, in config order) and whose label column is ``column``.  Each row's
        entry is the one ``row_entries`` gives, as the listing says (the entry that listed it), else the one its
        environment column names (:func:`~platoon.train.components.row_entry`)."""

        rows = list(rows)
        labels = tuple(entries)
        return cls(
            (str(row.get("task_id")) for row in rows),
            (row_entry(row, labels) for row in rows) if row_entries is None else row_entries,
            (task_label(row, column) for row in rows),
            rows=rows,
            drop_columns=drop_columns,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        entries: Sequence[str],
        column: str,
        drop_columns: Iterable[str] = (),
        row_entries: Iterable[str | None] | None = None,
    ) -> TaskPool:
        """A pool of a Hugging Face dataset's rows, each drawn as a fresh conversion of the row (without
        ``drop_columns``), of a run whose entries are ``entries`` and whose label column is ``column``; each row's
        entry as :meth:`from_rows` reads it."""

        count = len(dataset)
        names = set(dataset.column_names)
        labels = tuple(entries)

        def column_values(name: str) -> list[Any]:
            return list(dataset[name]) if name in names else [None] * count

        if row_entries is None:
            row_entries = (
                row_entry({ENVIRONMENT_COLUMN: value}, labels) for value in column_values(ENVIRONMENT_COLUMN)
            )
        return cls(
            (str(value) for value in column_values("task_id")),
            row_entries,
            (label_text(value) for value in column_values(column)),
            dataset=dataset,
            drop_columns=drop_columns,
        )

    def __len__(self) -> int:
        return len(self._task_ids)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._first_indices()

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Each row's task id, in listing order."""

        return self._task_ids

    def task_id(self, index: int) -> str:
        return self._task_ids[index]

    def entry(self, index: int) -> str | None:
        return self._entries[index]

    def label(self, index: int) -> str | None:
        return self._labels[index]

    def entry_indices(self, entry: str) -> tuple[int, ...]:
        """The indices of an entry's rows, in listing order."""

        return self._by_entry.get(entry, ())

    def first_index(self, task_id: str) -> int | None:
        """The index of the first row with ``task_id``, or None."""

        return self._first_indices().get(task_id)

    def row(self, index: int) -> dict[str, Any]:
        """The row at ``index``, as 0.3.0 drew it: a fresh conversion of a dataset's row, or a shallow copy of a row."""

        if self._dataset is not None:
            row = dict(self._dataset[index])
        elif self._rows is not None:
            row = dict(self._rows[index])
        else:
            raise IndexError(f"the pool holds no rows, so it has no row {index}")
        for column in self._drop:
            row.pop(column, None)
        return row

    def reissue_row(self, task_id: str, index: int | None) -> tuple[dict[str, Any], int] | None:
        """The row a resume draws again for an outstanding draw of ``task_id`` saved at pool index ``index``, with its
        index: that row while it still holds the task, else the first row with the task id (the pool changed); None
        when no row holds the task any more."""

        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(self._task_ids)
            and self._task_ids[index] == task_id
        ):
            return self.row(index), index
        first = self.first_index(task_id)
        return None if first is None else (self.row(first), first)

    def _first_indices(self) -> dict[str, int]:
        if self._first is None:
            first: dict[str, int] = {}
            for index, task_id in enumerate(self._task_ids):
                first.setdefault(task_id, index)
            self._first = first
        return self._first


@dataclass(frozen=True)
class _Row:
    """The built-in samplers' answer: the row of the sampler's pool at ``index`` (into ``ctx.task_ids``, the pool in
    listing order), which only an index can name in a listing that repeats a task id; or, with ``index`` None, ``row``
    itself, from an order handed to the dispatcher directly.  Only core's classes may answer with it."""

    index: int | None
    reason: str = "order"
    row: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class Decided:
    """A decision that binds: the row to draw (a copy of its own), its pool index (None for a row an order handed in),
    the qualified name of the sampler whose decision it is, and the draw's reason and score."""

    row: dict[str, Any]
    index: int | None
    sampler: str
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class Owed:
    """A static run sampler's slot owed to ``entry``, whose sampler is adaptive: the entry's sampler decides its task
    when the entry can start a group (:meth:`SamplerSet.decide`).  ``fallback``: the run sampler's fallback handed the
    slot (its decision failed), so the draw that fills it is the run's fallback, as a fallback's slot to an entry with
    a static sampler is."""

    entry: str
    fallback: bool = False


# A reason becomes a metric key only when it is a short word: the first REASON_KEYS of them a sampler gives; others
# count under ``draws/other``, while the group's record and its draw event keep the reason as given.
REASON_KEY = re.compile(r"[a-z][a-z0-9_]{0,31}")
REASON_KEYS = 16
# The reasons the dispatcher gives itself, always metric keys.
DISPATCH_REASONS = ("fallback", "reissue")


@dataclass(frozen=True)
class TaskSampling:
    """How a run draws its training tasks, as :meth:`~platoon.train.auto.AutoDataset.training` resolves it: the pool,
    the run's sampler, the sampler of each entry that the run's sampler hands slots to (in config order), each entry
    with training tasks (in config order), and an epoch's size, which the trainer counts epochs by."""

    pool: TaskPool
    run: SamplerChoice
    entries: Mapping[str, SamplerChoice]
    infos: Mapping[str, EntryInfo]
    epoch_size: int


# --- the samplers -----------------------------------------------------------------------------------------------------


def _always(task_id: str) -> bool:
    return True


def _never(task_id: str) -> bool:
    return False


@cache
def _core_classes() -> frozenset[type]:
    """Core's static samplers: the only ones whose answers may name a row by its index (:class:`_Row`), and the ones
    that fail only as platoon 0.3.0's orders did, which a valid setup never makes them do (an order handed in, which
    ``_PositionOrder`` draws as it is, raising), so their exceptions stop the run as 0.3.0's did, with no fallback."""

    from platoon.train import samplers

    return frozenset(
        {samplers.WeightedSampler, samplers.UniformSampler, samplers._ListingOrder, samplers._PositionOrder}
    )


class _Member:
    """One sampler of the set: its instance, its setup and options, its candidates' pool indices (in listing order),
    the seed its keys follow from, its decision count, and its failures."""

    def __init__(
        self,
        *,
        sampler: Any,
        setup: SamplerSetup,
        options: Any,
        indices: Sequence[int],
        seed: int,
        rebuild: Callable[[], Any],
    ) -> None:
        self.sampler = sampler
        self.setup = setup
        self.options = options
        self.indices = tuple(indices)
        self.seed = seed
        # Builds the instance afresh, ``rebuild()``: a state that does not load leaves the sampler as it was built.
        self.rebuild = rebuild
        self.decisions = 0
        self.fallbacks = 0
        self._logged: set[str] = set()
        self._local: dict[str, int] | None = None
        self._tasks: frozenset[str] | None = None
        # Since the last step's metrics: the draws by reason (as metric keys), and the time its decisions took.
        self.draws: Counter[str] = Counter()
        self.reason_keys: set[str] = set()
        self.seconds = 0.0
        self.slowest = 0.0
        self.timed = 0

    @property
    def adaptive(self) -> bool:
        return getattr(type(self.sampler), "adaptive", False) is True

    @property
    def observes(self) -> bool:
        """Whether it is handed settled groups: an adaptive sampler whose class defines ``observe`` (not the base
        class's)."""

        observe = getattr(type(self.sampler), "observe", None)
        return self.adaptive and callable(observe) and observe is not Sampler.observe

    def holds(self, task_id: str) -> bool:
        """Whether ``task_id`` is in its pool."""

        return task_id in self.task_set()

    def task_set(self) -> frozenset[str]:
        """The task ids of its pool."""

        if self._tasks is None:
            self._tasks = frozenset(self.setup.task_ids)
        return self._tasks

    @property
    def prefix(self) -> str:
        """Where its metrics go, below ``train/``: ``sampler/`` for the run's, ``sampler/environment/<entry>/`` for an
        entry's."""

        return "sampler/" if self.entry is None else f"sampler/{metric_prefix(ENVIRONMENT_COLUMN, self.entry)}/"

    def count(self, reason: str) -> None:
        """Count a draw of its decisions (or a draw of its that a resume drew again) by its reason, as a metric key."""

        if reason in DISPATCH_REASONS or reason in self.reason_keys:
            key = reason
        elif REASON_KEY.fullmatch(reason) and len(self.reason_keys) < REASON_KEYS:
            self.reason_keys.add(reason)
            key = reason
        else:
            key = "other"
        self.draws[key] += 1

    def timing(self, seconds: float) -> None:
        self.seconds += seconds
        self.slowest = max(self.slowest, seconds)
        self.timed += 1

    @property
    def name(self) -> str:
        return self.setup.name

    @property
    def level(self) -> str:
        return self.setup.level

    @property
    def entry(self) -> str | None:
        return self.setup.entry

    @property
    def class_ref(self) -> str:
        cls = type(self.sampler)
        return f"{cls.__module__}:{cls.__qualname__}"

    @property
    def slots(self) -> bool:
        """Whether it hands slots to the entries' samplers: a run sampler whose class sets ``entry_slots``."""

        return self.level == "run" and getattr(type(self.sampler), "entry_slots", False) is True

    def local_index(self, task_id: Any) -> int | None:
        """The index into its candidates of the first row of ``task_id``, or None."""

        if self._local is None:
            local: dict[str, int] = {}
            for index, candidate in enumerate(self.setup.task_ids):
                local.setdefault(candidate, index)
            self._local = local
        return self._local.get(task_id) if isinstance(task_id, str) else None

    def log_once(self, kind: str, message: str, *args: Any, exc_info: bool = False) -> None:
        if kind in self._logged:
            logger.debug(message, *args, exc_info=exc_info)
            return
        self._logged.add(kind)
        logger.error(message, *args, exc_info=exc_info)


class SamplerSet:
    """The run's task samplers (see the module docstring): :meth:`build` makes them from a :class:`TaskSampling`, and
    :meth:`of_order` from an order handed in.  The dispatcher asks :meth:`static` for the draw at each new position,
    and :meth:`decide` for an adaptive sampler's draw when a group can start; it saves :meth:`state` with each full
    checkpoint, and takes it up with :meth:`restore`.  :meth:`attach` hands it the run's per-task statistics, which
    the adaptive samplers read and observe."""

    def __init__(
        self,
        pool: TaskPool,
        run: _Member,
        entries: Mapping[str, _Member],
        infos: Mapping[str, EntryInfo],
        *,
        label_column: str = ENVIRONMENT_COLUMN,
    ):
        self.pool = pool
        self.run = run
        self.entries = dict(entries)
        self.infos = dict(infos)
        self.label_column = label_column
        # The run's per-task statistics (a TaskRecorder), which adaptive decisions read; and an empty store over the
        # pool, which they read when the statistics never started.
        self._recorder: Any = None
        self._empty: TaskStatsStore | None = None
        # Each task's label and group size, by its first row in the pool, which eligibility reads; each entry's labels,
        # the labels of each sampler's pool, and the largest group of each sampler's pool.
        self._labels: dict[str, str | None] | None = None
        self._sizes: dict[str, int] | None = None
        self._entry_labels: dict[str, tuple[str | None, ...]] = {}
        self._labels_of: dict[str, tuple[str | None, ...]] = {}
        self._largest: dict[str, int] = {}

    @classmethod
    def build(cls, sampling: TaskSampling, *, batch_size: int, label_column: str) -> SamplerSet:
        """The samplers ``sampling`` names: the run's over the whole pool, and, when it hands slots, each asked
        entry's over that entry's tasks, named ``<sampler>@<entry>``.  A sampler's decisions are keyed by the entry's
        seed (0 at the run level) plus its options' ``seed``, when they have one."""

        pool, infos = sampling.pool, dict(sampling.infos)

        def member(choice: SamplerChoice, entry: str | None, indices: Sequence[int]) -> _Member:
            info = infos.get(entry) if entry is not None else None
            seed = info.seed if info is not None else 0
            extra = getattr(choice.options, "seed", None)
            if isinstance(extra, int) and not isinstance(extra, bool):
                seed += extra
            setup = SamplerSetup(
                name=choice.name if entry is None else f"{choice.name}@{entry}",
                level="run" if entry is None else "entry",
                entry=entry,
                task_ids=tuple(pool.task_id(index) for index in indices),
                entries=infos,
                batch_size=batch_size,
                label_column=label_column,
                seed=seed,
            )
            return _Member(
                sampler=choice.cls(setup, choice.options),
                setup=setup,
                options=choice.options,
                indices=indices,
                seed=seed,
                rebuild=lambda: choice.cls(setup, choice.options),
            )

        run = member(sampling.run, None, range(len(pool)))
        entries = {}
        if run.slots:
            for label, choice in sampling.entries.items():
                if label in infos and pool.entry_indices(label):
                    entries[label] = member(choice, label, pool.entry_indices(label))
        return cls(pool, run, entries, infos, label_column=label_column)

    @classmethod
    def of_order(
        cls,
        order: Callable[[int], Mapping[str, Any]],
        *,
        pool: TaskPool | None = None,
        name: str = "auto",
        batch_size: int = 1,
        label_column: str = ENVIRONMENT_COLUMN,
    ) -> SamplerSet:
        """A static run sampler whose decision ``n`` is ``order(n)``'s row (an order handed to the dispatcher
        directly, or a trainer's dataloader order), over ``pool`` (the rows a resume and a fallback take; none by
        default)."""

        from platoon.train.samplers import _PositionOrder

        pool = pool if pool is not None else TaskPool((), (), ())
        setup = SamplerSetup(
            name=name,
            level="run",
            entry=None,
            task_ids=pool.task_ids,
            entries={},
            batch_size=batch_size,
            label_column=label_column,
            seed=0,
        )
        run = _Member(
            sampler=_PositionOrder(setup, order=order),
            setup=setup,
            options=None,
            indices=range(len(pool)),
            seed=0,
            rebuild=lambda: _PositionOrder(setup, order=order),
        )
        return cls(pool, run, {}, {}, label_column=label_column)

    def members(self) -> list[_Member]:
        return [self.run, *self.entries.values()]

    def adaptive(self, entry: str | None = None) -> bool:
        """Whether the run's sampler (``entry`` None) or an entry's is adaptive."""

        member = self.run if entry is None else self.entries.get(entry)
        return member is not None and member.adaptive

    def group_size(self, entry: str | None) -> int | None:
        """The training group size of ``entry``'s groups, or for None the largest of the run's entries (what an
        adaptive run sampler's decision holds permits for); None when the set does not know."""

        if entry is not None:
            info = self.infos.get(entry)
            return info.group_size if info is not None else None
        sizes = [info.group_size for info in self.infos.values()]
        return max(sizes) if sizes else None

    def entry_labels(self, entry: str) -> tuple[str | None, ...]:
        """The labels an entry's tasks carry, each once, in listing order."""

        found = self._entry_labels.get(entry)
        if found is None:
            found = tuple(dict.fromkeys(self.pool.label(index) for index in self.pool.entry_indices(entry)))
            self._entry_labels[entry] = found
        return found

    # --- the statistics -------------------------------------------------------------------------------------------

    def attach(self, recorder: Any) -> None:
        """Read the run's per-task statistics from ``recorder`` (a TaskRecorder): an adaptive decision's view takes
        every settled group in first, and each group taken in is handed, in draw order, to the ``observe`` of each
        adaptive sampler whose pool holds its task."""

        self._recorder = recorder
        if any(member.observes for member in self.members()) and callable(getattr(recorder, "listen", None)):
            recorder.listen(self._observe)

    def _observe(self, record: Any) -> None:
        for member in self.members():
            if not member.observes or not member.holds(record.task_id):
                continue
            try:
                member.sampler.observe(record)
            except Exception as exc:
                member.log_once(
                    "observe",
                    "Task sampler %s raised in observe (%s: %s): the group is dropped for it, and later ones are "
                    "handed to it as before; this is logged once",
                    member.name,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )

    def view(self, now: int, *, flush: bool = True) -> TaskStatsView:
        """The run's per-task statistics at policy version ``now``, every settled group taken in: those the run keeps
        (``live`` False once they stopped), or, when they never started, an empty store over the pool (``live``
        False).  With ``flush`` False, as a group admission rule reads them when a group settles, the settled groups
        held since the last flush stay held (:meth:`~platoon.train.task_records.TaskRecorder.peek_store`)."""

        recorder = self._recorder
        store = None
        if recorder is not None and not flush and callable(peek := getattr(recorder, "peek_store", None)):
            store = peek()
        elif recorder is not None:
            store = getattr(recorder, "store", None)  # a recorder's store property flushes
        if isinstance(store, TaskStatsStore):
            return TaskStatsView(store, now=now, live=not getattr(recorder, "stopped", False))
        if self._empty is None:
            pool = self.pool
            self._empty = TaskStatsStore(
                [(pool.task_id(index), pool.entry(index), pool.label(index)) for index in range(len(pool))],
                self.infos,
                column=self.label_column,
                group_size=self.group_size(None) or 1,
            )
        return TaskStatsView(self._empty, now=now, live=False)

    # --- static decisions -----------------------------------------------------------------------------------------

    def static(self, position: int, *, step: int = 0, versions: Mapping[str, int] | None = None) -> Decided | Owed:
        """The draw at ``position`` of the run's order, from its static run sampler: the run's sampler's decision
        ``position``, and, for a slot to an entry with a static sampler, that sampler's decision at the entry's count;
        a slot to an entry with an adaptive sampler is :class:`Owed`, decided when the entry can start a group.
        Every sampler's count advances by the decision it made.  A draw whose slot the run sampler's fallback handed
        names the run's sampler, with reason ``fallback``: the decision it replaced."""

        if self.run.decisions != position:
            self._sync(position)
        versions = versions if versions is not None else {MAIN: 0}
        fallbacks = self.run.fallbacks
        answer = self._decide(self.run, step=step, versions=versions)
        if isinstance(answer, Slot):
            member = self.entries[answer.entry]
            if member.adaptive:
                return Owed(answer.entry, fallback=self.run.fallbacks > fallbacks)
            answer = self._decide(member, step=step, versions=versions)
            if self.run.fallbacks > fallbacks:
                answer = self._as_fallback(answer)
        # An entry's sampler hands no slot: _accept refuses one, and its fallback is a task.
        assert isinstance(answer, Decided)
        return answer

    def entry_static(
        self, entry: str, *, step: int = 0, versions: Mapping[str, int] | None = None, fallback: bool = False
    ) -> Decided:
        """The next decision of ``entry``'s static sampler (a resume binds the slots an entry's adaptive sampler owed,
        once its sampler is static); with ``fallback``, the slot was the run sampler's fallback's, and so is the
        draw."""

        answer = self._decide(self.entries[entry], step=step, versions=versions or {MAIN: 0})
        assert isinstance(answer, Decided)
        return self._as_fallback(answer) if fallback else answer

    def _as_fallback(self, decided: Decided) -> Decided:
        """``decided``, the task an entry's sampler chose for a slot the run sampler's fallback handed, as the draw of
        the run's sampler whose decision it replaced: reason ``fallback``, no score."""

        return Decided(decided.row, decided.index, self.run.name, "fallback")

    def decide(
        self,
        entry: str | None,
        *,
        room: Callable[[str | None], bool],
        step: int = 0,
        versions: Mapping[str, int] | None = None,
        fallback: bool = False,
        most: int | None = None,
    ) -> Decided | None:
        """An adaptive sampler's decision, made when a group can start: the run's sampler over the whole pool
        (``entry`` None) or ``entry``'s over its tasks, with the run's statistics in view and a task eligible when its
        label has room (``room(label)``) and, with ``most``, its group takes at most that many rollout permits (those
        the dispatcher holds for the decision).  A failed decision (an exception, an answer outside the contract,
        nothing while a task is eligible) gets the fallback: the eligible task at offset ``key mod`` their number.
        None, with no count advanced, when no task is eligible.  With ``fallback`` the slot it fills was the run
        sampler's fallback's, and the draw is the run's fallback (:meth:`static`)."""

        member = self.run if entry is None else self.entries[entry]
        versions = versions if versions is not None else {MAIN: 0}
        decision = member.decisions
        ctx = self.context(member, room=room, step=step, versions=versions, most=most)
        eligible = ctx.eligible
        started = time.perf_counter()
        try:
            answer = member.sampler.next(ctx)
        except Exception as exc:
            member.timing(time.perf_counter() - started)
            member.log_once(
                "raised",
                "Task sampler %s raised at decision %d (%s: %s): the run draws a fallback task instead, chosen by the "
                "decision's key among the eligible tasks, and counts every fallback; the error is logged once",
                member.name,
                decision,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            result = None
        else:
            member.timing(time.perf_counter() - started)
            if answer is None and not any(eligible(task) for task in member.setup.task_ids):
                return None  # nothing eligible: nothing decided
            result = self._accept(member, answer, eligible=eligible)
            if result is not None and not isinstance(result, Decided):
                result = None  # an adaptive sampler hands no slot
            if result is None:
                member.log_once(
                    "refused" if answer is not None else "none",
                    "%s: the run draws a fallback task instead, chosen by the decision's key among the eligible tasks, "
                    "and counts every fallback; this is logged once",
                    SamplerError(member.name, f"decision {decision} answered {_brief(answer)}, which it may not"),
                )
        if result is None:
            replaced = self._fallback(member, ctx, eligible=eligible)
            if replaced is None:
                return None  # no eligible task to fall back to: nothing decided
            member.fallbacks += 1
            result = replaced
        member.decisions = decision + 1
        assert isinstance(result, Decided)
        return self._as_fallback(result) if fallback else result

    def context(
        self,
        member: _Member,
        *,
        room: Callable[[str | None], bool] | None = None,
        step: int = 0,
        versions: Mapping[str, int] | None = None,
        most: int | None = None,
    ) -> SamplerContext:
        """The context of ``member``'s next decision, as the dispatcher builds it: a static sampler's, without
        statistics, every task eligible; with ``room`` (whether a label has room), an adaptive sampler's, with the
        run's statistics in view (:meth:`view`) and a task of its pool eligible when its label has room and, with
        ``most``, its entry's group takes at most ``most`` rollout permits."""

        versions = versions if versions is not None else {MAIN: 0}
        if room is None:
            return self._context(member, member.decisions, step=step, versions=versions)
        sizes = self._task_sizes() if most is not None and self._sizes_differ(member, most) else None
        own = self._member_labels(member)
        if sizes is None and len(own) == 1:
            # Every task of its pool carries one label (an entry's, by default): its room is read once, and a task is
            # eligible when it is the sampler's, by a lookup that costs next to nothing on a large pool.
            eligible: Callable[[str], bool] = member.task_set().__contains__ if room(own[0]) else _never
        else:
            labels = self._task_labels()
            missing = object()
            # Each label's room, read once in a decision: a label fills only between decisions.
            rooms: dict[Any, bool] = {}

            def by_label(task_id: str) -> bool:
                label = labels.get(task_id, missing) if member.holds(task_id) else missing
                if label is missing or (sizes is not None and sizes.get(task_id, 0) > most):  # type: ignore[operator]
                    return False
                has = rooms.get(label)
                if has is None:
                    has = rooms[label] = room(label)  # type: ignore[arg-type]
                return has

            eligible = by_label

        decision = member.decisions
        return SamplerContext(
            task_ids=member.setup.task_ids,
            entry=member.entry,
            decision=decision,
            key=decision_key(member.seed, member.name, decision),
            eligible=eligible,
            stats=self.view(versions.get(MAIN, 0)),
            step=step,
            versions=versions,
        )

    def accepts(self, member: _Member, answer: Any, ctx: SamplerContext) -> bool:
        """Whether the dispatcher takes ``answer`` to ``member``'s decision of ``ctx``, by the rules :meth:`decide`
        and :meth:`static` apply: a task of its pool (eligible, for an adaptive sampler), a slot it may give, a
        recordable reason and score."""

        result = self._accept(member, answer, eligible=ctx.eligible if member.adaptive else None)
        return result is not None and (not member.adaptive or isinstance(result, Decided))

    def _task_labels(self) -> dict[str, str | None]:
        """Each task's label, by its first row in the pool."""

        if self._labels is None:
            pool, labels = self.pool, {}
            for index in range(len(pool)):
                labels.setdefault(pool.task_id(index), pool.label(index))
            self._labels = labels
        return self._labels

    def _member_labels(self, member: _Member) -> tuple[str | None, ...]:
        """The labels the tasks of ``member``'s pool carry, each once, in listing order."""

        found = self._labels_of.get(member.name)
        if found is None:
            labels = self._task_labels()
            found = self._labels_of[member.name] = tuple(dict.fromkeys(labels[task] for task in member.setup.task_ids))
        return found

    def _task_sizes(self) -> dict[str, int]:
        """Each task's group size, by the entry of its first row in the pool (a task of no entry: the run's largest)."""

        if self._sizes is None:
            pool, sizes = self.pool, {}
            largest = self.group_size(None) or 1
            for index in range(len(pool)):
                info = self.infos.get(pool.entry(index))  # type: ignore[arg-type]
                sizes.setdefault(pool.task_id(index), info.group_size if info is not None else largest)
            self._sizes = sizes
        return self._sizes

    def _sizes_differ(self, member: _Member, most: int) -> bool:
        """Whether some task of ``member``'s pool takes more than ``most`` rollout permits."""

        largest = self._largest.get(member.name)
        if largest is None:
            sizes = self._task_sizes()
            largest = self._largest[member.name] = max((sizes[task] for task in member.setup.task_ids), default=0)
        return largest > most

    def _context(self, member: _Member, decision: int, *, step: int, versions: Mapping[str, int]) -> SamplerContext:
        return SamplerContext(
            task_ids=member.setup.task_ids,
            entry=member.entry,
            decision=decision,
            key=decision_key(member.seed, member.name, decision),
            eligible=_always,
            stats=None,
            step=step,
            versions=versions,
        )

    def _decide(self, member: _Member, *, step: int, versions: Mapping[str, int]) -> Decided | Slot:
        """``member``'s next decision, checked, or its fallback; its count advances either way.  Core's static
        samplers get no fallback: what they raise, or an answer they may not give, stops the run (:func:`_core`)."""

        decision = member.decisions
        ctx = self._context(member, decision, step=step, versions=versions)
        started = time.perf_counter()
        try:
            answer = member.sampler.next(ctx)
        except Exception as exc:
            member.timing(time.perf_counter() - started)
            if _core(member):
                raise
            member.log_once(
                "raised",
                "Task sampler %s raised at decision %d (%s: %s): the run draws a fallback task instead, chosen by the "
                "decision's key, and counts every fallback; the error is logged once",
                member.name,
                decision,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            result = None
        else:
            member.timing(time.perf_counter() - started)
            result = self._accept(member, answer)
            if result is None:
                refused = SamplerError(member.name, f"decision {decision} answered {_brief(answer)}, which it may not")
                if _core(member):
                    raise refused
                member.log_once(
                    "refused" if answer is not None else "none",
                    "%s: the run draws a fallback task instead, chosen by the decision's key, and counts every "
                    "fallback; this is logged once",
                    refused,
                )
        if result is None:
            member.fallbacks += 1
            result = self._fallback(member, ctx)
            assert result is not None  # every task of a static sampler's pool is eligible
        member.decisions = decision + 1
        return result

    def _accept(
        self, member: _Member, answer: Any, *, eligible: Callable[[str], bool] | None = None
    ) -> Decided | Slot | None:
        """What ``answer`` binds, or None when the dispatcher refuses it: a task outside the sampler's pool (or, for an
        adaptive sampler, not ``eligible``), a slot it may not give, a reason or a score that cannot be recorded,
        core's row answer from another class, or anything else.  A static sampler's None is refused too: every task
        of its pool is eligible."""

        if isinstance(answer, _Row):
            if type(member.sampler) not in _core_classes() or not _valid_reason(answer.reason):
                return None
            if answer.index is None:
                if not isinstance(answer.row, Mapping):
                    return None
                return Decided(dict(answer.row), None, member.name, answer.reason)
            local = answer.index
            if isinstance(local, bool) or not isinstance(local, int) or not 0 <= local < len(member.indices):
                return None
            if eligible is not None and not eligible(member.setup.task_ids[local]):
                return None
            index = member.indices[local]
            return Decided(self.pool.row(index), index, member.name, answer.reason)
        if isinstance(answer, str):
            answer = Draw(answer)
        if isinstance(answer, Draw):
            if not _valid_reason(answer.reason) or not _valid_score(answer.score):
                return None
            local = member.local_index(answer.task_id)
            if local is None or (eligible is not None and not eligible(answer.task_id)):
                return None
            index = member.indices[local]
            score = None if answer.score is None else float(answer.score)
            return Decided(self.pool.row(index), index, member.name, answer.reason, score)
        if isinstance(answer, Slot):
            return answer if member.slots and answer.entry in self.entries else None
        return None

    def _fallback(
        self, member: _Member, ctx: SamplerContext, *, eligible: Callable[[str], bool] | None = None
    ) -> Decided | Slot | None:
        """The draw that replaces a decision that failed: a slot to an entry chosen by weight, for a sampler that hands
        slots, else of its pool's tasks in listing order (an adaptive sampler's ``eligible`` ones, the first row of
        each), the one at offset ``key mod`` their number; None when no task is eligible.  It keeps no state."""

        if member.slots and self.entries:
            return Slot(self._weighted_entry(ctx.key))
        if not member.indices:
            raise SamplerError(member.name, "its decision failed, and its pool holds no task to fall back to")
        if eligible is None:
            index = member.indices[ctx.key % len(member.indices)]
            return Decided(self.pool.row(index), index, member.name, "fallback")
        ids = member.setup.task_ids
        candidates = [local for local, task in enumerate(ids) if member.local_index(task) == local and eligible(task)]
        if not candidates:
            return None
        index = member.indices[candidates[ctx.key % len(candidates)]]
        return Decided(self.pool.row(index), index, member.name, "fallback")

    def _weighted_entry(self, key: int) -> str:
        """An entry the run's sampler hands slots to, chosen by ``key`` with probability proportional to its weight."""

        labels = list(self.entries)
        weights = [_weight(self.infos.get(label)) for label in labels]
        point = random.Random(key).random() * sum(weights)
        for label, weight in zip(labels, weights):
            point -= weight
            if point < 0:
                return label
        return labels[-1]

    # --- where the counts stand -------------------------------------------------------------------------------------

    def _sync(self, position: int) -> None:
        """Bring every count to where an uninterrupted run stands at ``position``: the run's sampler at decision
        ``position``, and each entry's sampler at the slots its entry received before it, counted by asking the run's
        sampler its decisions again (static samplers are functions of their decision index)."""

        static = {label: member for label, member in self.entries.items() if not member.adaptive}
        if not static:
            self.run.decisions = position
            return
        counts = self._slot_counts(position)
        self.run.decisions = position
        for label, member in static.items():
            member.decisions = counts[label]

    def _slot_counts(self, position: int) -> dict[str, int]:
        """How many of the run sampler's first ``position`` decisions handed a slot to each entry: its answer, or, where
        it fails, the fallback's slot."""

        run = self.run
        counts = dict.fromkeys(self.entries, 0)
        versions = {MAIN: 0}
        core = _core(run)
        for decision in range(position):
            ctx = self._context(run, decision, step=0, versions=versions)
            try:
                answer = self._accept(run, run.sampler.next(ctx))
            except Exception:
                if core:
                    raise
                answer = None
            if answer is None:
                if core:
                    raise SamplerError(run.name, f"decision {decision} answered nothing it may give")
                answer = self._fallback(run, ctx)
            if isinstance(answer, Slot):
                counts[answer.entry] += 1
        return counts

    # --- checkpoints ----------------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What a full checkpoint saves of the samplers, with the dispatcher's state: each one's name, class, decision
        count and ``state_dict()`` (the run's under ``run``, each entry's by label under ``entries``).  A
        ``state_dict()`` that raises, or that is not JSON, is left out with a warning: a resume starts that sampler
        afresh."""

        return {
            "run": self._record(self.run),
            "entries": {label: self._record(member) for label, member in self.entries.items()},
        }

    def _record(self, member: _Member) -> dict[str, Any]:
        record: dict[str, Any] = {"name": member.name, "class": member.class_ref, "decisions": member.decisions}
        try:
            state = member.sampler.state_dict()
            text = json.dumps(state, allow_nan=False)
            if not isinstance(state, Mapping):
                raise TypeError(f"state_dict() returned a {type(state).__name__}, not a mapping")
            # A copy of its own, whole: a checkpoint writes the state in a thread while the dispatcher goes on deciding,
            # and a state_dict() may return the containers the sampler keeps changing.
            state = json.loads(text)
        except Exception as exc:
            member.log_once(
                "state",
                "Could not save the state of task sampler %s (%s: %s): a resume from this checkpoint starts it afresh",
                member.name,
                type(exc).__name__,
                exc,
            )
        else:
            record["state"] = state
        return record

    def restore(self, saved: Any, *, next_position: int) -> None:
        """Take up what a checkpoint saved of the samplers (``saved``, :meth:`state`'s; None for a checkpoint that
        saved none, as platoon 0.3.0's did), the order going on at ``next_position``.  A sampler saved under the same
        name and class loads its state and its count; a static one starts at the count its level implies (the run's at
        ``next_position``, an entry's at the slots its entry received before it), and an adaptive one that was not
        saved, or whose state does not load (which warns), starts afresh, at decision 0."""

        saved_entries = saved.get("entries") if isinstance(saved, Mapping) else None
        decisions = self._load(self.run, saved.get("run") if isinstance(saved, Mapping) else None)
        self.run.decisions = (decisions or 0) if self.run.adaptive else next_position
        counts: dict[str, int] | None = None
        for label, member in self.entries.items():
            record = saved_entries.get(label) if isinstance(saved_entries, Mapping) else None
            decisions = self._load(member, record)
            if decisions is None and not member.adaptive:
                if counts is None:
                    counts = self._slot_counts(next_position)
                decisions = counts[label]
            member.decisions = decisions or 0

    def _load(self, member: _Member, record: Any) -> int | None:
        """Load ``member``'s saved state when ``record`` names the same sampler and class, and return its saved count;
        None when it names another (or none) or its state does not load, which leaves the sampler as built."""

        if not isinstance(record, Mapping):
            return None
        if (record.get("name"), record.get("class")) != (member.name, member.class_ref):
            return None
        decisions = record.get("decisions")
        if isinstance(decisions, bool) or not isinstance(decisions, int) or decisions < 0:
            return None
        if "state" not in record:  # its state could not be saved
            return None
        try:
            state = record["state"]
            if not isinstance(state, Mapping):
                raise ValueError(f"a sampler's state is a mapping, got {type(state).__name__}")
            member.sampler.load_state_dict(state)
        except Exception as exc:
            logger.warning(
                "The state that task sampler %s saved does not load (%s: %s): it starts afresh",
                member.name,
                type(exc).__name__,
                exc,
            )
            member.sampler = member.rebuild()
            return None
        return decisions

    # --- what the metrics read --------------------------------------------------------------------------------------

    def fallbacks(self) -> dict[str, int]:
        """The fallback draws so far, by sampler."""

        return {member.name: member.fallbacks for member in self.members()}

    def member(self, name: str | None) -> _Member | None:
        """The sampler whose qualified name is ``name``, or None."""

        return next((member for member in self.members() if member.name == name), None)

    def prefix(self, name: str | None) -> str | None:
        """Where the metrics of the sampler named ``name`` go (``sampler/`` or ``sampler/environment/<entry>/``), or
        None for a sampler the set does not hold."""

        member = self.member(name)
        return member.prefix if member is not None else None

    def count(self, sampler: str | None, reason: str) -> None:
        """Count a draw: by the run's counts, and by its sampler's when that is an entry's."""

        self.run.count(reason)
        member = self.member(sampler)
        if member is not None and member is not self.run:
            member.count(reason)

    def step_metrics(self, now: int) -> dict[str, float]:
        """The samplers' metrics of the step, from their draws and decisions since the last step (keys below
        ``train/``): under each sampler's prefix (``sampler/``, ``sampler/environment/<entry>/``), ``draws`` and
        ``draws/<reason>``, the run's counting every draw; ``decision_ms`` and ``decision_ms_max``, the mean and the
        longest time its decisions took; and what an adaptive sampler's ``metrics(stats)`` returns (a static one's
        too, without statistics), each a finite number.  A sampler whose ``metrics`` raises is logged once."""

        found: dict[str, float] = {}
        view: TaskStatsView | None = None
        for member in self.members():
            prefix = member.prefix
            if member.draws or member is self.run:
                found[f"{prefix}draws"] = float(sum(member.draws.values()))
                for reason, count in member.draws.items():
                    found[f"{prefix}draws/{reason}"] = float(count)
            if member.timed:
                found[f"{prefix}decision_ms"] = 1000.0 * member.seconds / member.timed
                found[f"{prefix}decision_ms_max"] = 1000.0 * member.slowest
            member.draws.clear()
            member.seconds, member.slowest, member.timed = 0.0, 0.0, 0
            if member.adaptive and view is None:
                view = self.view(now)
            try:
                values = member.sampler.metrics(view if member.adaptive else None)
                if not isinstance(values, Mapping):
                    raise TypeError(f"metrics() returned a {type(values).__name__}, not a mapping")
                taken: dict[str, float] = {}
                for key, value in values.items():
                    if not isinstance(key, str) or not key:
                        raise ValueError(f"metric key {key!r} is not a non-empty string")
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"metric {key!r} is {value!r}, not a finite number")
                    taken[key] = float(value)
            except Exception as exc:
                member.log_once(
                    "metrics",
                    "Task sampler %s: its metrics could not be read (%s: %s); this is logged once",
                    member.name,
                    type(exc).__name__,
                    exc,
                )
                continue
            for key, value in taken.items():
                if reserved_metric(key):  # the dispatcher's own count under the same prefix stays as it counted
                    member.log_once(
                        "reserved",
                        "Task sampler %s: its metric %r is one the dispatcher writes itself under the sampler's prefix "
                        "(%s, draws/<reason>), so it is left out; this is logged once",
                        member.name,
                        key,
                        ", ".join(sorted(RESERVED_METRICS)),
                    )
                    continue
                found[f"{prefix}{key}"] = value
        return found


def _core(member: _Member) -> bool:
    """Whether ``member`` is one of core's static samplers (:func:`_core_classes`), whose failures stop the run."""

    return type(member.sampler) in _core_classes()


def _valid_reason(reason: Any) -> bool:
    return isinstance(reason, str) and 0 < len(reason) <= MAX_REASON_LENGTH


def _valid_score(score: Any) -> bool:
    return score is None or (
        isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score))
    )


def _weight(info: EntryInfo | None) -> float:
    if info is None:
        return 1.0
    if info.weight is not None:
        return float(info.weight)
    return float(len(info.task_ids)) or 1.0


def _brief(answer: Any) -> str:
    text = repr(answer)
    return text if len(text) <= 200 else text[:197] + "..."
