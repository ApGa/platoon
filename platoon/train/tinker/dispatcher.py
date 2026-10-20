"""The dispatcher of a training run: it starts each rollout group when the group can start (RFC 0019, section 4).

It replaces the producer that drew a batch's rows ahead of the rollout workers, the pending queue they took them
from, and the pool of long-lived workers: :meth:`Dispatcher.run` draws a task only when its group can start, and starts
each group as a task of its own.  A group starts when three things allow it:

- **The horizon.**  With ``k = train.max_staleness``, the groups dispatched and not yet dequeued by a step (the
  pipeline: running, or settled and waiting for a step), plus those the current step has dequeued, stay below
  ``(k + 1) × batch_size``: the lookahead the producer kept, which lets the rollouts of a step run at most ``k`` steps
  ahead of training.  The count drops back when a step starts, after its policy is published.  Without a bound there
  is no horizon.  Either way the dispatcher never starts more groups than the run's remaining steps consume, so no
  group runs past the plan.  With ``k = 0`` a step trains exactly the groups drawn for it, as the producer had it,
  unless a label's limit holds one of them back.
- **Rollout permits** (``train.max_concurrent_rollouts``): a group takes its entry's group size of them at once,
  first come first served, and returns them when its workflow returns.
- **Label capacity** (``train.rollout_group_limits``): at most that many groups of a label run at once.

A draw of the order whose label is at its limit waits in the label's first-in, first-out backlog, which does not count
toward the horizon, so the dispatcher draws on for the other labels.  Once :data:`BACKLOG_BATCHES` batches of a label's
draws wait, a new draw of that label is skipped: its position is used up, its fate is ``skipped``, and a warning says so
once.  The dispatcher draws only while some label of the run has room, so it never draws only to skip, and a skip
always leaves the slot to a label that can use it.  Each draw it starts is, in this order: the oldest waiting draw whose
label has room (the draws a resume reissued come first), then the next position of the order.  It wakes when something
it waits for changes, never by polling: a group's release of its label and permits, a step that dequeues a group, a step
that starts.

**Slots held for a capped label.**  The horizon's slots free when a step starts, and the draws that start then take
them, so a capped label whose group ends in the middle of a step would wait for the next step to start its next group:
a limit of ``n`` would let the label start at most ``n`` groups a step, however short its groups.  So while a capped
label has draws waiting, each of its running groups holds one more slot of the horizon, which the other labels' draws
leave free, for the draw that follows it: when the group ends, that draw starts at once.  It holds them only as far as
the capped labels' running groups and the slots they hold fit in ``k`` batches, so that the other labels always have a
batch of the horizon; with ``k = 0`` there is no slot to hold, and a capped label starts at most its limit of groups
each step.

**Exact resume.**  Every draw is numbered ``(start, seq)`` and stamped on its row (``_platoon_seq``,
``_platoon_position``), and the dispatcher keeps every draw no committed step consumed (outstanding).  What a full
checkpoint saves of it (:func:`save_state`) is a file of its own, ``tasks/dispatch-s<start>-<checkpoint>.json``, which
the checkpoint's row names (``dispatch``), so that it does not depend on the run's per-task statistics: the next
position of the order, the outstanding draws, each label's waiting draws and a fingerprint of the sampling inputs
(:meth:`Dispatcher.state`), and, in ``tasks/kept-s<start>-<checkpoint>.json.gz``
(:mod:`platoon.train.tinker.kept_groups`), the groups that settled and no committed step consumed
(:meth:`Dispatcher.kept`).  A start that resumes reads it back
(:func:`read_state`) and rebuilds the dispatcher from it (:func:`plan_resume`, :meth:`Dispatcher.restore`): the kept
groups go to the result queue first, so the steps after the resume train them without running them again; every other
outstanding draw (one that was running, waiting for its label, or whose group could not be kept) is drawn again first,
as a new draw of this start (reason ``reissue``) at its old place in the order; and the order goes on from where the
checkpoint left it.  So every position of the order is consumed exactly once: trained, dropped as stale, failed or
skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Container, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from platoon.train.task_stats import read_tasks_file, write_tasks_file

logger = logging.getLogger(__name__)

# How many draws of a label at its limit may wait, as a multiple of the batch size, before its new draws are skipped.
BACKLOG_BATCHES = 4
# The columns the dispatcher stamps on each training row it draws: the start's draw number and the row's place in the
# order (its position; a reissued draw keeps the position of the draw it replaces).
SEQ_COLUMN = "_platoon_seq"
POSITION_COLUMN = "_platoon_position"
# The format of the file of the dispatcher's state that a full checkpoint saves, which its row names.
STATE_FORMAT = 1

Key = tuple[int, int]


class Permits(Protocol):
    """Rollout permits that a group takes whole before it starts and returns when its workflow returns."""

    async def acquire(self, count: int) -> None: ...

    def release(self, count: int) -> None: ...


class RowReader(Protocol):
    """What the dispatcher reads of each row it draws (``of``: its entry, its label and its group size), and the run's
    rollout permits."""

    @property
    def permits(self) -> Permits: ...

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]: ...


class DrawRecorder(Protocol):
    """The run's per-task statistics, as the dispatcher tells them of each draw and skip
    (:class:`~platoon.train.tinker.task_records.TaskRecorder`)."""

    def draw(
        self,
        row: Mapping[str, Any],
        *,
        seq: int,
        position: int | None,
        version: int,
        reissues: tuple[int, int] | None = None,
    ) -> None: ...

    def skipped(self, row: Mapping[str, Any]) -> None: ...


@dataclass(eq=False)
class Draw:
    """One drawn task: its row, stamped with its draw number and position, its key in the run's statistics
    (``(start, seq)``), and what the dispatcher reads of it: its entry, its label and its group size."""

    row: dict[str, Any]
    start: int
    seq: int
    position: int | None
    task_id: str
    entry: str | None
    label: str | None
    group_size: int
    # Taken in the order the draws were made (or restored): of the waiting draws whose label has room, the one with the
    # smallest ticket starts first.
    ticket: int

    @property
    def key(self) -> Key:
        return (self.start, self.seq)


# A training group's coroutine: it runs the group of ``draw``, calls ``release`` once its workflow has returned (so that
# the group's label slot and rollout permits are free before anything else), then hands its output to
# :meth:`Dispatcher.settle`.
GroupRunner = Callable[[Draw, Callable[[], None]], Awaitable[None]]


class Dispatcher:
    """Starts a training run's rollout groups as the horizon, the rollout permits and the label limits allow (see the
    module docstring).

    ``rows`` reads each row's entry, label and group size and holds the run's rollout permits; ``order(position)`` is
    the row at a position of the run's order; ``steps`` the training steps this start runs; ``labels`` every label of
    the run's training tasks, which must be complete: the dispatcher draws only while one of them has room, and learns
    another one only by drawing it, so a label it is not given never counts as having room before its first draw;
    ``recorder`` the run's per-task statistics, told of each draw and skip; ``version()`` the policy version now.
    ``start`` numbers this start's draws, and the order starts at ``next_position``."""

    def __init__(
        self,
        *,
        rows: RowReader,
        order: Callable[[int], Mapping[str, Any]],
        batch_size: int,
        max_staleness: int | None,
        steps: int,
        limits: Mapping[str, int] | None = None,
        labels: Iterable[str | None] = (),
        start: int = 0,
        next_position: int = 0,
        backlog_limit: int | None = None,
        recorder: DrawRecorder | None = None,
        version: Callable[[], int] = lambda: 0,
        finish_batches: bool = False,
        sampler: str = "auto",
        fingerprint: Mapping[str, Any] | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if backlog_limit is not None and (
            isinstance(backlog_limit, bool) or not isinstance(backlog_limit, int) or backlog_limit < 1
        ):
            raise ValueError("backlog_limit must be a positive integer")
        self._rows = rows
        self._order = order
        self.batch_size = batch_size
        self.max_staleness = max_staleness
        self._limits: dict[str | None, int] = {label: limit for label, limit in (limits or {}).items()}
        self._labels: set[str | None] = set(labels)
        self.start = start
        self.next_position = next_position
        self.backlog_limit = backlog_limit if backlog_limit is not None else BACKLOG_BATCHES * batch_size
        self._recorder = recorder
        self._version = version
        # With several updates per batch, a step that applied one finishes after a stop request, so the groups it still
        # needs must start: the dispatcher then goes on until the train loop's task group cancels it.
        self.finish_batches = finish_batches
        self.sampler = sampler
        self.fingerprint = dict(fingerprint or {})
        # The groups dispatched and not yet dequeued by a step (running, or settled and waiting in the result queue),
        # and those the current step has dequeued: together, the horizon's count.
        self.pipeline = 0
        self.admitted = 0
        # The most the horizon's count reached, which the tests check.
        self.peak = 0
        self._steps_left = steps
        self._active: Counter[str | None] = Counter()
        # The draws that wait, by label, each in the order it was drawn: those a resume reissued, and those whose label
        # was at its limit (the label's backlog).
        self._waiting: dict[str | None, deque[Draw]] = {}
        # The groups that settled and no committed step consumed yet, by key, in the order they settled; a full
        # checkpoint keeps them (kept).
        self._settled: dict[Key, Any] = {}
        # Every draw no committed step consumed, by key, in the order it was drawn: its task and its position.
        self._outstanding: dict[Key, tuple[str, int | None]] = {}
        # Draws of each label skipped because its backlog was full.
        self.skipped: Counter[str | None] = Counter()
        self._next_seq = 0
        self._next_ticket = 0
        self._changed = asyncio.Event()
        self.results: asyncio.Queue[Any] | None = None

    # --- what the train loop tells it -------------------------------------------------------------------------------

    def dequeued(self) -> None:
        """A step dequeued a group: it leaves the pipeline and counts toward the step."""

        self.pipeline = max(0, self.pipeline - 1)
        self.admitted += 1
        self._changed.set()

    def committed(self, keys: Iterable[Key | None]) -> None:
        """The step that dequeued the groups of ``keys`` committed: they were consumed, so no checkpoint keeps them."""

        for key in keys:
            if key is not None:
                self._settled.pop(key, None)
                self._outstanding.pop(key, None)

    def step_started(self) -> None:
        """A new step starts, its policy published: the groups the last one dequeued no longer count."""

        self.admitted = 0
        self._steps_left = max(0, self._steps_left - 1)
        self._changed.set()

    # --- what a group tells it --------------------------------------------------------------------------------------

    def settle(self, draw: Draw, item: Any) -> None:
        """The group of ``draw`` settled: ``item`` (what a step dequeues) goes to the result queue, and is kept until a
        committed step consumes it."""

        self._settled[draw.key] = item
        if self.results is not None:
            self.results.put_nowait(item)
        self._changed.set()

    # --- the loop ---------------------------------------------------------------------------------------------------

    def bind(self, results: asyncio.Queue[Any]) -> None:
        """Hand the dispatcher the queue its groups settle into, the groups a resume kept first, in the order they
        settled."""

        self.results = results
        for item in self._settled.values():
            results.put_nowait(item)

    async def run(
        self,
        *,
        spawn: Callable[[Coroutine[Any, Any, None]], Any],
        run_group: GroupRunner,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Start groups for as long as the run needs them: ``spawn`` runs a coroutine as a task (the train loop's task
        group), ``run_group`` runs one group, and ``stop`` asks to stop starting groups, unless a step that applied an
        update must still finish (``finish_batches``)."""

        while not self._stopping(stop):
            draw = self._choose()
            if draw is None:
                self._changed.clear()
                await self._changed.wait()
                continue
            await self._start(draw, spawn, run_group, stop)

    def _stopping(self, stop: asyncio.Event | None) -> bool:
        return stop is not None and stop.is_set() and not self.finish_batches

    def _horizon_has_room(self, label: str | None = None, *, waiting: bool = False) -> bool:
        """Whether another group may be dispatched: the run's remaining steps consume more than the groups in the
        pipeline and those the current step dequeued, and so does the horizon, ``(k + 1) × batch_size``, less the slots
        held for the capped labels' next groups (:meth:`_held`), those held for ``label`` excepted for one of its
        ``waiting`` draws, which they are held for."""

        in_use = self.pipeline + self.admitted
        if in_use >= self._steps_left * self.batch_size:
            return False
        if self.max_staleness is None:
            return True
        held = self._held()
        if waiting and held and self._waiting.get(label) and label in self._limits:
            held -= min(self._active[label], held)
        return in_use + held < (self.max_staleness + 1) * self.batch_size

    def _held(self) -> int:
        """The slots of the horizon held for the groups that follow the capped labels' running ones: one for each
        running group of a capped label that has draws waiting, as far as those groups and the slots held for them fit
        in ``k`` batches (none with ``k = 0``).  The label's oldest waiting draw takes the slot when its label has
        room."""

        if not self.max_staleness or not self._limits:
            return 0
        running = sum(
            self._active[label] for label, waiting in self._waiting.items() if waiting and label in self._limits
        )
        return min(running, max(0, self.max_staleness * self.batch_size - running))

    def _has_room(self, label: str | None) -> bool:
        return label not in self._limits or self._active[label] < self._limits[label]

    def _some_label_has_room(self) -> bool:
        """Whether a group of some label of the run could start: one without a limit, or below it.  With no limits,
        or no label known, nothing is known to be full."""

        if not self._limits or not self._labels:
            return True
        return any(self._has_room(label) for label in self._labels)

    def _choose(self) -> Draw | None:
        """The next draw to start, its label's slot free and the horizon's room for it: the oldest waiting draw whose
        label has room, else a new draw of the order, while some label has room (a new draw whose label is at its limit
        waits, or past its backlog is skipped).  None when nothing can start now."""

        best: deque[Draw] | None = None
        for label, waiting in self._waiting.items():
            if (
                waiting
                and self._has_room(label)
                and self._horizon_has_room(label, waiting=True)
                and (best is None or waiting[0].ticket < best[0].ticket)
            ):
                best = waiting
        if best is not None:
            return best.popleft()
        if not self._horizon_has_room():
            return None
        while self._some_label_has_room():
            draw = self._new_draw()
            if self._has_room(draw.label):
                return draw
            waiting = self._waiting.setdefault(draw.label, deque())
            if len(waiting) < self.backlog_limit:
                waiting.append(draw)  # waits for its label's limit
                if not self._horizon_has_room():  # its label's running groups now hold a slot for it
                    return None
                continue
            self._skip(draw)
        return None

    async def _start(
        self,
        draw: Draw,
        spawn: Callable[[Coroutine[Any, Any, None]], Any],
        run_group: GroupRunner,
        stop: asyncio.Event | None = None,
    ) -> None:
        """Take the draw's label slot and its group's permits (whole, first come first served), and start its group;
        a stop request that came while it waited for permits puts the draw back, first in line."""

        if draw.label in self._limits:
            self._active[draw.label] += 1
        self.pipeline += 1
        self.peak = max(self.peak, self.pipeline + self.admitted)
        try:
            await self._rows.permits.acquire(draw.group_size)
        except BaseException:
            self._unstart(draw)
            raise
        if self._stopping(stop):
            self._rows.permits.release(draw.group_size)
            self._unstart(draw)
            self._waiting.setdefault(draw.label, deque()).appendleft(draw)
            return
        spawn(self._group(draw, run_group))

    def _unstart(self, draw: Draw) -> None:
        if draw.label in self._limits:
            self._active[draw.label] -= 1
        self.pipeline -= 1

    async def _group(self, draw: Draw, run_group: GroupRunner) -> None:
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                self._rows.permits.release(draw.group_size)
                if draw.label in self._limits:
                    self._active[draw.label] -= 1
                self._changed.set()

        try:
            await run_group(draw, release)
        finally:
            release()

    # --- draws ------------------------------------------------------------------------------------------------------

    def _new_draw(self) -> Draw:
        position = self.next_position
        self.next_position += 1
        draw = self._draw(dict(self._order(position)), position)
        self._record(draw)
        return draw

    def _draw(self, row: dict[str, Any], position: int | None) -> Draw:
        """Number ``row`` as this start's next draw, at ``position`` in the order: it is outstanding until a committed
        step consumes it, or it is skipped."""

        seq = self._next_seq
        self._next_seq += 1
        row[SEQ_COLUMN] = seq
        row[POSITION_COLUMN] = position
        entry, label, group_size = self._rows.of(row)
        self._labels.add(label)
        ticket = self._next_ticket
        self._next_ticket += 1
        draw = Draw(row, self.start, seq, position, str(row.get("task_id")), entry, label, group_size, ticket)
        self._outstanding[draw.key] = (draw.task_id, position)
        return draw

    def _record(self, draw: Draw, *, reissues: Key | None = None, version: int | None = None) -> None:
        if self._recorder is not None:
            self._recorder.draw(
                draw.row,
                seq=draw.seq,
                position=draw.position,
                version=self._version() if version is None else version,
                reissues=reissues,
            )

    def _skip(self, draw: Draw) -> None:
        self._outstanding.pop(draw.key, None)
        self.skipped[draw.label] += 1
        if self._recorder is not None:
            self._recorder.skipped(draw.row)
        if self.skipped[draw.label] == 1:
            logger.warning(
                "Label %r has %d rows waiting for its limit (train.rollout_group_limits: %d at once): its new rows are "
                "skipped, so that the other labels' rows keep the batches full; its share of the order asks for more "
                "groups than its limit runs",
                draw.label,
                self.backlog_limit,
                self._limits.get(draw.label, 0),
            )

    # --- what the run's metrics read ----------------------------------------------------------------------------------

    def parked_by_label(self) -> Counter[str | None]:
        """The draws that wait for their label's limit, by label."""

        return Counter(
            {label: len(waiting) for label, waiting in self._waiting.items() if waiting and label in self._limits}
        )

    # --- checkpoints and resume ---------------------------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """What a full checkpoint saves of the dispatcher, beside the groups it keeps (:meth:`kept`): this start's
        number, the sampler and the fingerprint of its inputs, the next position of the order, the outstanding draws
        (``[start, seq, task id, position]`` each, in the order they were drawn), and each label's waiting draws
        (``[start, seq]`` each, oldest first).  A resume draws again the outstanding draws that neither wait nor are
        kept."""

        return {
            "start": self.start,
            "sampler": self.sampler,
            "fingerprint": self.fingerprint,
            "next_position": self.next_position,
            "outstanding": [
                [start, seq, task, position] for (start, seq), (task, position) in self._outstanding.items()
            ],
            "waiting": {
                "" if label is None else label: [list(draw.key) for draw in waiting]
                for label, waiting in self._waiting.items()
                if waiting
            },
        }

    def kept(self) -> list[tuple[Key, Any]]:
        """The groups that settled and no committed step consumed, ``(key, item)`` each, in the order they settled."""

        return list(self._settled.items())

    def restore(
        self,
        *,
        kept: Sequence[tuple[Key, str, int | None, Any]],
        reissue: Sequence[tuple[Key, Mapping[str, Any], int | None]],
        waiting: Sequence[tuple[Key, Mapping[str, Any], int | None]],
        version: int,
    ) -> None:
        """Take up what a checkpoint saved (:func:`plan_resume`): ``kept`` groups, ``(key, task id, position, item)``
        each, which the steps dequeue first and which stay outstanding until a committed step consumes them; the draws
        to draw again, ``(old key, row, position)`` each, oldest first, before any new draw (``reissue``); and the draws
        that waited for their label, in their labels' order (``waiting``).  Each draw drawn again is a new draw of this
        start, at its old position, at policy ``version``, and replaces the old one among the outstanding draws."""

        for key, task_id, position, item in kept:
            self._settled[key] = item
            self._outstanding[key] = (task_id, position)
        self.pipeline += len(kept)
        for old, row, position in [*reissue, *waiting]:
            draw = self._draw(dict(row), position)
            self._record(draw, reissues=old, version=version)
            self._waiting.setdefault(draw.label, deque()).append(draw)


# --- what a full checkpoint saves of it, and what a resume takes up --------------------------------------------------


def state_name(start: int, checkpoint: str) -> str:
    """The file of the dispatcher's state that a start's checkpoint saves: ``dispatch-s<start>-<checkpoint>.json``."""

    return f"dispatch-s{start}-{checkpoint}.json"


def save_state(
    run_dir: str | os.PathLike[str], checkpoint: str, state: Mapping[str, Any], kept: Sequence[tuple[Key, Any]]
) -> dict[str, Any] | None:
    """Write what checkpoint ``checkpoint`` saves of the dispatcher: the groups it keeps (``kept``, as
    :meth:`Dispatcher.kept` gives them; those that can be kept) to ``tasks/kept-s<start>-<checkpoint>.json.gz``, then
    its state (``state``, :meth:`Dispatcher.state`'s, which names that file with its SHA-256) to
    ``tasks/dispatch-s<start>-<checkpoint>.json``, and return the record the checkpoint's row keeps in ``loop_state``
    (``dispatch``): ``{file, sha256, format, start}``, and the kept groups' file (``kept``) when there is one.  Kept
    groups that cannot be written leave the state without them, with a warning, so that a resume draws their tasks
    again; a state that cannot be written leaves the checkpoint without one, with a warning, so that a resume from it
    draws from its batch.  Never raises."""

    from platoon.train.tinker.kept_groups import write_kept_groups

    try:
        start = int(state["start"])
        kept_record = write_kept_groups(run_dir, start, checkpoint, kept)
        content = json.dumps(
            {"format": STATE_FORMAT, **state, "kept": kept_record}, separators=(",", ":"), allow_nan=False
        )
        record = {
            **write_tasks_file(run_dir, state_name(start, checkpoint), (content + "\n").encode()),
            "format": STATE_FORMAT,
            "start": start,
        }
    except OSError as exc:  # the disk, not the state
        logger.warning(
            "Could not write where the run's draws stand with checkpoint %s (%s): a resume from it draws from its "
            "batch, and rolls out again what had not trained",
            checkpoint,
            exc,
        )
        return None
    except Exception:  # what a resume takes up never stops training
        logger.warning("Could not save where the run's draws stand with checkpoint %s", checkpoint, exc_info=True)
        return None
    if kept_record is not None:
        record["kept"] = kept_record["file"]
    return record


def read_state(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> dict[str, Any]:
    """The dispatcher's state that a checkpoint row's ``dispatch`` record names, checked against its SHA-256; a
    ``ValueError`` that says why when it is missing, changed or unreadable."""

    content = read_tasks_file(run_dir, record)
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ValueError(f"{record.get('file')} is not JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != STATE_FORMAT:
        raise ValueError(f"{record.get('file')} holds no dispatcher's state of format {STATE_FORMAT}")
    return data


@dataclass(frozen=True)
class ResumePlan:
    """What a resume does with each draw a checkpoint left outstanding (:func:`plan_resume`): ``(key, task id,
    position)`` each."""

    # The groups the checkpoint kept, which the steps train first, in the order they settled.
    kept: list[tuple[Key, str, int | None]]
    # The draws to draw again first, in (start, seq) order: their groups were running, or could not be kept.
    reissue: list[tuple[Key, str, int | None]]
    # The draws that waited for their label's limit, back to their labels' backlogs in order.
    waiting: list[tuple[Key, str, int | None]]
    # The draws whose task left the run's training tasks.
    lost: list[tuple[Key, str, int | None]]

    def keys(self) -> list[Key]:
        """Every outstanding draw the plan takes up or loses."""

        return [key for part in (self.kept, self.reissue, self.waiting, self.lost) for key, _task, _position in part]


def plan_resume(
    *,
    outstanding: Iterable[Sequence[Any]],
    waiting: Mapping[str, Iterable[Sequence[Any]]],
    kept: Iterable[Key],
    pool: Container[str],
) -> ResumePlan:
    """Split the draws a checkpoint left outstanding (``outstanding``: ``[start, seq, task id, position]`` each) by
    what a resume does with them, given the keys of the groups the checkpoint kept that read back (``kept``, in the
    order they settled), each label's waiting draws (``waiting``, ``[start, seq]`` each, oldest first) and the run's
    training tasks now (``pool``): a kept group trains as it is; a draw that waited for its label goes back to its
    backlog; every other one is drawn again first; and a draw whose task left ``pool`` is lost.  A kept group or a
    waiting draw that is not outstanding is passed over.  A ``ValueError`` for data it cannot read."""

    try:
        drawn: dict[Key, tuple[str, int | None]] = {
            (int(start), int(seq)): (str(task), None if position is None else int(position))
            for start, seq, task, position in outstanding
        }
        waits = [(int(start), int(seq)) for keys in waiting.values() for start, seq in keys]
        kept_keys = [(int(start), int(seq)) for start, seq in kept]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot read the outstanding draws: {type(exc).__name__}: {exc}") from exc
    plan = ResumePlan(kept=[], reissue=[], waiting=[], lost=[])
    taken: set[Key] = set()
    others = sorted(set(drawn) - set(kept_keys) - set(waits))
    for name, keys in (("kept", kept_keys), ("waiting", waits), ("reissue", others)):
        for key in keys:
            if key not in drawn or key in taken:
                continue
            taken.add(key)
            task, position = drawn[key]
            getattr(plan, name if task in pool else "lost").append((key, task, position))
    return plan
