"""The trainer's records of its groups: each draw, settle and fate into the run's per-task statistics.

:class:`TaskRecorder` is how the trainer keeps a :class:`~platoon.train.task_stats.TaskStatsStore`: the dispatcher
(:mod:`platoon.train.dispatch.dispatcher`) records each draw it numbers (:meth:`TaskRecorder.draw`), with the sampler
that drew it, why and its score, and each draw it skips; each group records itself when it settles
(:meth:`TaskRecorder.settle`, from the row it ran and its workflow's output); and the train loop gives each group its
fate once the step that dequeued it commits (:meth:`TaskRecorder.fate`).  Each full checkpoint saves the store as a
snapshot (:meth:`TaskRecorder.snapshot`, whose record goes into the checkpoint's ``loop_state``).  A start that resumes
loads the snapshot its checkpoint names before its launch save (:meth:`TaskRecorder.begin`).  The dispatcher saves where
the draws stand in a file of its own (:mod:`platoon.train.dispatch.dispatcher`), and takes up the outstanding draws from
it: the groups the checkpoint kept train as they are, and the others are drawn again, each still one draw of its task
(:meth:`TaskStatsStore.reissue`).  When the snapshot cannot be read, the start goes on from the newest older one that
can, else from empty, and the run's ``lock.json`` records why (:func:`record_lost_state`); the draws that older snapshot
holds as outstanding and the checkpoint no longer does count as lost.  So do all of them when the checkpoint saved no
state of the dispatcher's draws (one saved before platoon 0.3.0, or whose write failed), and the run then draws from the
checkpoint's batch.  ``tasks/events.jsonl`` logs every step of it, a start's own events once its launch save has
succeeded (:meth:`TaskRecorder.started`), so that a start that dies before it leaves none.

**Draw order.**  A settled group's record is made when it settles, but the store takes it in later, with every other
record held then, in the order the groups were drawn (``(start, seq)``): before each decision of an adaptive task
sampler, each full checkpoint's snapshot and each step's metrics (:meth:`TaskRecorder.flush`, which every reader of the
store runs first, the :attr:`TaskRecorder.store` property included).  So the statistics do not depend on the order in
which groups completed between two of those points: their decayed sums are float additions, whose order shows in their
last bits, and a task's streaks and newest groups follow its groups' order.  The flush also hands each record, in the
same order, to the adaptive samplers that observe their pool's groups (:meth:`TaskRecorder.listen`).  A group's fate
does not wait for it: what became of a draw is counted from the outstanding draws alone.

The statistics never fail training: a snapshot that cannot be written leaves its checkpoint without one, and the next
full save writes them again; any other unexpected error stops the recorder for the rest of the start, with a warning,
and the checkpoints it saves then name no snapshot: a resume from one of them goes on from an older snapshot's
statistics, and takes up the draws all the same, from the dispatcher's own file.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from platoon.train import entry_settings
from platoon.train.components import row_entry
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, SEQ_COLUMN
from platoon.train.labels import label_column, task_label
from platoon.train.task_stats import (
    LOOP_STATE_KEY,
    MAIN,
    EntryInfo,
    EventLog,
    GroupRecord,
    TaskStatsStore,
    next_start,
    read_snapshot,
    readable_snapshot,
    snapshot_name,
    write_snapshot,
)
from platoon.utils.config import LOCK_NAME, update_run_lock

logger = logging.getLogger(__name__)

# The lock.json keys that list the starts that could not load the statistics their checkpoint named
# (record_lost_state), and those whose sampling inputs changed since it (record_sampling_changed).
LOST_KEY = "tasks_state_lost"
CHANGED_KEY = "sampling_changed"


class TaskRecorder:
    """Records a training run's groups into its per-task statistics (see the module docstring)."""

    def __init__(self, *, run_dir: str, config: Any, tasks: Iterable[Mapping[str, Any]], start: int) -> None:
        self.run_dir = run_dir
        self.start = start
        self.column = label_column(config)
        settings = entry_settings.run_settings(config)
        thresholds = {
            entry.label: float(getattr(entry, "success_threshold", 1.0))
            for entry in getattr(config, "environments", None) or []
        }
        self.entries = {
            label: EntryInfo(group_size=int(own.train.group_size), success_threshold=thresholds.get(label, 1.0))
            for label, own in settings.items()
        }
        run = getattr(getattr(config, "train", None), "workflow_config", None)
        self.group_size = int(getattr(run, "group_size", 1) or 1)
        labels = tuple(self.entries)
        # The run's training tasks: (task id, entry, label) each, in pool order.
        self._pool = [(str(row["task_id"]), row_entry(row, labels), task_label(row, self.column)) for row in tasks]
        self._store = self._new_store()
        # The settled groups' records the store has not taken in yet (flush), and who is handed each one it takes in.
        self._held: list[GroupRecord] = []
        self._listeners: list[Callable[[GroupRecord], None]] = []
        self.sampler = str(getattr(getattr(config, "train", None), "task_sampler", None) or "auto")
        self.events = EventLog(run_dir)
        # Set by an unexpected error: the recorder records nothing more in this start.
        self.stopped = False
        self._next_seq = 0
        # The policy version each outstanding draw of this start was drawn at, by (start, seq), and what drew it: the
        # sampler, the reason and the score.
        self._drawn_at: dict[tuple[int, int], int] = {}
        self._drawn_by: dict[tuple[int, int], tuple[str, str, float | None]] = {}
        # The events of this start's beginning, written once its launch save has succeeded (started); None after.
        self._beginning: list[tuple[str, dict[str, Any]]] | None = []

    @classmethod
    def begin(
        cls,
        *,
        run_dir: str,
        config: Any,
        tasks: Iterable[Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
        checkpoint: str | None,
        resume: Mapping[str, Any] | None,
        batch: int,
    ) -> tuple[TaskRecorder, dict[str, Any] | None]:
        """The recorder of a new start of a run whose ``checkpoints.jsonl`` holds ``records``, numbered after the
        starts they name, with the statistics of the checkpoint it resumes from (``checkpoint``, whose row's
        ``loop_state`` names its snapshot in ``resume``) loaded; the dispatcher then takes up its outstanding draws
        (:meth:`lose_outstanding` counts as lost those it cannot draw again).  When that checkpoint names no snapshot
        (one saved before platoon 0.3.0, or when its write failed), or its file is missing, changed or unreadable, the
        start goes on from the newest older checkpoint whose snapshot reads back, else from empty, and the second
        value says so: ``{start, checkpoint, problem, loaded}`` (the checkpoint whose snapshot it loaded, or None), for
        ``lock.json`` (:func:`record_lost_state`).  The start's events wait for :meth:`started`."""

        recorder = cls(run_dir=run_dir, config=config, tasks=tasks, start=next_start(records))
        lost = None
        if checkpoint is not None:
            problem, loaded = None, checkpoint
            if resume is None:
                problem = f"checkpoint {checkpoint} names no snapshot of the per-task statistics"
            else:
                try:
                    recorder._store.load(read_snapshot(run_dir, resume))
                except ValueError as exc:
                    problem = str(exc)
                    recorder._store = recorder._new_store()  # a snapshot read in part is no state to go on from
            if problem is not None:
                loaded = recorder._load_older(run_dir, records, checkpoint)
                lost = {"start": recorder.start, "checkpoint": checkpoint, "problem": problem, "loaded": loaded}
                if loaded is not None:
                    logger.warning(
                        "The run's per-task statistics go on from those checkpoint %s saved, older than checkpoint %s "
                        "this start resumes from, so the groups between the two are not counted: %s",
                        loaded,
                        checkpoint,
                        problem,
                    )
                else:
                    logger.warning("The run's per-task statistics start from empty: %s", problem)
        recorder._event(
            "start",
            start=recorder.start,
            batch=batch,
            checkpoint=checkpoint,
            state_lost=lost["problem"] if lost is not None else None,
            loaded=lost["loaded"] if lost is not None else checkpoint,
        )
        return recorder, lost

    def lose_outstanding(self, *, keep: Iterable[tuple[int, int]] = ()) -> int:
        """Count every outstanding draw but those of ``keep`` as lost, as a resume does with those it cannot draw
        again (all of them, when its checkpoint saved no state of the dispatcher's draws; else those of an older
        snapshot that the checkpoint no longer holds); returns how many.  Never raises."""

        if self.stopped:
            return 0
        try:
            keep = set(keep)
            lost = [
                (start, seq) for start, seq, _task, _position in self._store.outstanding if (start, seq) not in keep
            ]
            for start, seq in lost:
                self._store.fate(start, seq, "lost")
                self._event("fate", start=start, seq=seq, fate="lost")
            return len(lost)
        except Exception:
            self._stop("counting the outstanding draws as lost")
            return 0

    def _load_older(self, run_dir: str, records: Sequence[Mapping[str, Any]], checkpoint: str) -> str | None:
        """Load the newest snapshot that reads back among the rows before the last one of ``checkpoint`` (the
        checkpoint a start resumes from, whose own snapshot did not), and return its checkpoint's name; None, with the
        store empty, when none does."""

        names = [row.get("name") for row in records]
        end = len(names) - 1 - names[::-1].index(checkpoint) if checkpoint in names else len(names)
        older = [row for row in records[:end] if isinstance(row.get(LOOP_STATE_KEY), Mapping)]
        found = readable_snapshot(run_dir, older)
        while found is not None:
            try:
                self._store.load(found.data)
                return str(found.row.get("name"))
            except ValueError:  # it reads, but holds no state this store can load: try the next older one
                self._store = self._new_store()
                found = readable_snapshot(run_dir, older[: older.index(found.row)])
        return None

    def started(self) -> None:
        """This start's launch save succeeded: write the events of its beginning to the event log (its ``start``
        event, a ``lost`` fate for each draw it could not draw again, a ``draw`` for each one it draws again), and
        each later event as it happens.  A start that dies before then leaves none, so that the next start, which takes
        its number, writes them once."""

        beginning, self._beginning = self._beginning or [], None
        for event, fields in beginning:
            self.events.write(event, **fields)

    def _event(self, event: str, **fields: Any) -> None:
        if self._beginning is not None:
            self._beginning.append((event, fields))
        else:
            self.events.write(event, **fields)

    def _new_store(self) -> TaskStatsStore:
        return TaskStatsStore(self._pool, self.entries, column=self.column, group_size=self.group_size)

    @property
    def store(self) -> TaskStatsStore:
        """The run's statistics, every settled group taken in (:meth:`flush`)."""

        self.flush()
        return self._store

    @store.setter
    def store(self, store: TaskStatsStore) -> None:
        self._store = store

    def peek_store(self) -> TaskStatsStore:
        """The run's statistics without taking in the settled groups held since the last flush: what a group admission
        rule reads when a group settles (a flush at every settle would take records in completion order)."""

        return self._store

    def listen(self, listener: Callable[[GroupRecord], None]) -> None:
        """Hand ``listener`` each settled group's record as the store takes it in, in draw order (:meth:`flush`)."""

        self._listeners.append(listener)

    def flush(self) -> None:
        """Take in the records of the groups that settled since the last flush, in the order they were drawn
        (``(start, seq)``), and hand each to the listeners (:meth:`listen`) as it is taken in.  Runs before each
        adaptive decision, snapshot and step's metrics; nothing once the recorder has stopped."""

        if self.stopped or not self._held:
            return
        held, self._held = sorted(self._held, key=lambda record: (record.start, record.seq)), []
        for record in held:
            try:
                self._store.settle(record)
            except Exception:
                self._stop("recording a settled group")
                return
            for listener in self._listeners:
                try:
                    listener(record)
                except Exception:  # a listener's own failure: the statistics never fail training
                    logger.warning("Could not hand a settled group to %r", listener, exc_info=True)

    # --- draws, settles and fates ----------------------------------------------------------------------------------

    def _stop(self, what: str) -> None:
        self.stopped = True
        self._held = []  # what the store had not taken in is dropped, as later settles are
        logger.warning("The run's per-task statistics stop for the rest of this start: %s failed", what, exc_info=True)

    def draw(
        self,
        row: Mapping[str, Any],
        *,
        seq: int,
        position: int | None,
        version: int,
        reissues: tuple[int, int] | None = None,
        sampler: str | None = None,
        reason: str | None = None,
        score: float | None = None,
    ) -> None:
        """Record ``row``, which the dispatcher numbered as this start's draw ``seq``, at ``position`` in the order and
        policy ``version``; ``reissues`` names the outstanding draw of an earlier start that it draws again, which it
        replaces among the outstanding draws.  ``sampler`` (its qualified name; the run's ``train.task_sampler``
        when not given), ``reason`` (``reissue`` for a draw drawn again, else ``order`` when not given) and ``score``
        say what drew it and why: the group's record and the draw's event keep them."""

        self._next_seq = max(self._next_seq, seq + 1)
        if self.stopped:
            return
        try:
            task_id = str(row.get("task_id"))
            key = (self.start, seq)
            if reissues is None or not self._store.reissue(reissues, key):
                self._store.draw(self.start, seq, task_id, position)
            self._drawn_at[key] = version
            sampler = sampler or self.sampler
            reason = reason or ("reissue" if reissues is not None else "order")
            self._drawn_by[key] = (sampler, reason, score)
            fields: dict[str, Any] = {
                "start": self.start,
                "seq": seq,
                "task_id": task_id,
                "position": position,
                "version": version,
                "sampler": sampler,
                "reason": reason,
            }
            if score is not None:
                fields["score"] = score
            if reissues is not None:
                fields["reissues"] = list(reissues)
            self._event("draw", **fields)
        except Exception:
            self._stop("recording a draw")

    def skipped(self, row: Mapping[str, Any]) -> None:
        """A drawn row that a capped label's full backlog skipped."""

        seq = row.get(SEQ_COLUMN)
        self._fate((self.start, seq) if isinstance(seq, int) else None, "skipped")

    def slot_skipped(self, *, position: int, entry: str, why: str) -> None:
        """A slot of the order owed to ``entry``'s adaptive sampler whose position was used up without a draw: its
        entry's queue was full (``backlog``), or a resume found that the entry no longer gets slots (``no_slots``) or
        left the run (``entry_left``).  Its event accounts for the position in the event log."""

        if self.stopped:
            return
        try:
            self._event("slot_skipped", start=self.start, position=position, entry=entry, why=why)
        except Exception:
            self._stop("recording a skipped slot")

    def lose(self, key: tuple[int, int], task_id: str | None = None) -> None:
        """An outstanding draw of ``task_id`` that a resume cannot draw again: its task left the pool."""

        self._fate(key, "lost", task_id)

    def settle(
        self, row: Mapping[str, Any], *, entry: str | None, label: str | None, output: Any
    ) -> GroupRecord | None:
        """Record the group of ``row`` that settled with ``output`` (None for a workflow that raised): its record, which
        the store takes in at the next :meth:`flush`, in draw order; None for a row this recorder did not draw (an
        evaluation's)."""

        seq = row.get(SEQ_COLUMN)
        if self.stopped or not isinstance(seq, int):
            return None
        try:
            return self._settle(row, seq, entry=entry, label=label, output=output)
        except Exception:
            self._stop("recording a settled group")
            return None

    def _settle(
        self, row: Mapping[str, Any], seq: int, *, entry: str | None, label: str | None, output: Any
    ) -> GroupRecord:
        info = self.entries.get(entry) if entry is not None else None
        sampler, reason, score = self._drawn_by.get((self.start, seq), (self.sampler, "order", None))
        record = group_record(
            start=self.start,
            seq=seq,
            task_id=str(row.get("task_id")),
            entry=entry,
            labels={self.column: label} if label is not None else {},
            group_size=info.group_size if info is not None else None,
            sampler=sampler,
            reason=reason,
            score=score,
            position=row.get(POSITION_COLUMN),
            drawn_at={MAIN: self._drawn_at.get((self.start, seq), 0)},
            output=output,
        )
        self._held.append(record)
        self._event(
            "settle",
            start=self.start,
            seq=seq,
            task_id=record.task_id,
            verdict=record.verdict,
            valid=record.valid,
            passes=record.passes(self._store.threshold(entry)),
            version=record.oldest_version(),
        )
        return record

    def fate(self, record: GroupRecord, fate: str, *, rule: str | None = None, admission: str | None = None) -> None:
        """What became of a settled group once the step that dequeued it committed: consumed, stale or failed; or, under
        a group admission rule, when it settled: rejected, or failed, with the ``rule`` that decided it and its reason
        (``admission``), which the fate's event names.  A group an earlier start's checkpoint kept has that start's
        key."""

        self._fate((record.start, record.seq), fate, record.task_id, rule=rule, admission=admission)

    def fate_of(
        self, key: tuple[int, int], fate: str, *, rule: str | None = None, admission: str | None = None
    ) -> None:
        """The fate of draw ``key`` whose group has no record: a start whose statistics had stopped kept it."""

        self._fate(key, fate, rule=rule, admission=admission)

    def _fate(
        self,
        key: tuple[int, int] | None,
        fate: str,
        task_id: str | None = None,
        *,
        rule: str | None = None,
        admission: str | None = None,
    ) -> None:
        if self.stopped or key is None:
            return
        try:
            self._drawn_at.pop(key, None)
            self._drawn_by.pop(key, None)
            self._store.fate(key[0], key[1], fate, task_id)
            decided = {"rule": rule, "admission": admission} if rule is not None or admission is not None else {}
            self._event("fate", start=key[0], seq=key[1], fate=fate, **decided)
        except Exception:
            self._stop("recording a fate")

    # --- checkpoints and metrics ---------------------------------------------------------------------------------

    def snapshot(self, checkpoint: str) -> dict[str, Any] | None:
        """Write the store as the snapshot of ``checkpoint``, which this start is saving, and return the record the
        checkpoint's row keeps in ``loop_state`` (``tasks``).  None once the recorder has stopped, and None, with a
        warning, when the snapshot cannot be written: the statistics go on, and the next full save writes them
        again."""

        self.flush()
        if self.stopped:
            return None
        try:
            data = self._store.snapshot(start=self.start, next_seq=self._next_seq)
            return write_snapshot(self.run_dir, snapshot_name(self.start, checkpoint), data)
        except OSError as exc:  # the disk, not the statistics
            logger.warning(
                "Could not write the per-task statistics with checkpoint %s (%s): it names no snapshot, the statistics "
                "go on, and the next full checkpoint saves them",
                checkpoint,
                exc,
            )
            return None
        except Exception:
            self._stop(f"saving them with checkpoint {checkpoint}")
            return None

    def step_metrics(self, *, now: int, prefixes: Mapping[str, str]) -> dict[str, float]:
        self.flush()
        if self.stopped:
            return {}
        try:
            return self._store.step_metrics(now=now, prefixes=prefixes)
        except Exception:
            self._stop("computing the step metrics")
            return {}

    def threshold(self, entry: str | None) -> float:
        return self._store.threshold(entry)

    def close(self) -> None:
        self.events.close()


def group_record(
    *,
    start: int,
    seq: int,
    task_id: str,
    entry: str | None,
    labels: Mapping[str, str],
    group_size: int | None,
    sampler: str,
    reason: str,
    score: float | None,
    position: int | None,
    drawn_at: Mapping[str, int],
    output: Any,
) -> GroupRecord:
    """The record of a group of draw ``(start, seq)`` of ``task_id`` that settled with ``output`` (None for a workflow
    that raised or returned nothing: verdict ``failed``): its members and verdict as the output gives them (an output
    without a verdict, such as a plain list of datums, has verdict ``unknown``), its group size ``group_size`` (its
    members' number when None).  What the per-task statistics take in (:meth:`TaskRecorder.settle`), and what a group
    admission rule decides on when the statistics stopped or never started."""

    verdict = getattr(output, "verdict", None) if output is not None else "failed"
    members = tuple(getattr(output, "members", ()) or ())
    return GroupRecord(
        start=start,
        seq=seq,
        task_id=task_id,
        entry=entry,
        labels=dict(labels),
        harness=None,
        group_size=group_size if group_size is not None else len(members),
        sampler=sampler,
        reason=reason,
        score=score,
        position=position,
        drawn_at=dict(drawn_at),
        members=members,
        verdict=verdict if isinstance(verdict, str) else "unknown",
    )


def record_lost_state(run_dir: str | Path, lost: Mapping[str, Any]) -> None:
    """Add ``lost``, a start that could not load the statistics its checkpoint named (:meth:`TaskRecorder.begin`), to
    the list ``lock.json`` keeps of them (``tasks_state_lost``): each supervised start of one ``platoon train``
    writes into the same ``lock.json``, so the list says which starts lost what.  Never raises."""

    _append_to_lock(run_dir, LOST_KEY, lost, "that this start's per-task statistics were lost")


def _append_to_lock(run_dir: str | Path, key: str, value: Mapping[str, Any], what: str) -> None:
    try:
        path = Path(run_dir) / LOCK_NAME
        lock = json.loads(path.read_text()) if path.is_file() else None
        earlier = lock.get(key) if isinstance(lock, dict) else None
        update_run_lock(run_dir, {key: [*(earlier if isinstance(earlier, list) else []), dict(value)]})
    except Exception:  # the statistics never fail training
        logger.warning("Could not record in lock.json %s", what, exc_info=True)


def record_sampling_changed(run_dir: str | Path, change: Mapping[str, Any]) -> None:
    """Add ``change``, a start whose sampling inputs differ from those of the checkpoint it resumed from
    (``{start, checkpoint, changes}``), to the list ``lock.json`` keeps of them (``sampling_changed``).  Never
    raises."""

    _append_to_lock(run_dir, CHANGED_KEY, change, "that the run's sampling inputs changed")


def resume_record(resume_info: Any) -> Mapping[str, Any] | None:
    """The record of the snapshot a checkpoint row names (``loop_state``'s ``tasks``), None when it names none."""

    extra = getattr(resume_info, "extra", None) or {}
    record = extra.get(LOOP_STATE_KEY)
    return record if isinstance(record, Mapping) else None
