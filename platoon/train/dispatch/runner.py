"""One training group's run: its workflow awaited with watchdog heartbeats, what it adds to the statistics, and its
settling, or its decision first under the run's group admission rule.

The functions here take the start's rollout source (:class:`~platoon.train.dispatch.source.RolloutSource`), whose
dispatcher started the group, and read its parts: its workflow, per-task statistics, admission rules, watchdog and
tracker.  ``_run_group`` runs any group of a workflow, an evaluation's too.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from platoon.train.admissions import AdmissionSet
from platoon.train.credit.runtime import CreditEstimatorError, settle_credit
from platoon.train.dispatch.dispatcher import BoundDraw
from platoon.train.dispatch.permits import DrawLease
from platoon.train.dispatch.settled import _Settled, carries_datums, emptied
from platoon.train.leases import holding
from platoon.train.trainer.waiting import _await_rollout_with_heartbeat
from platoon.train.workflow.protocol import RolloutWorkflow, report_of
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.dispatch.source import RolloutSource
    from platoon.train.trainer.signals import Watchdog

logger = logging.getLogger(__name__)


def _group_done(rollout: Any, elapsed: float, task_id: Any, tracker: StatsTracker, watchdog: "Watchdog | None") -> None:
    """What a group that ended adds to the statistics: its time, with the group's own statistics when its report gives
    them (the step that consumes it records them, :func:`~platoon.train.dispatch.settled._record_group_stats`), else in
    ``tracker`` at once (a failed group, or a workflow's plain list); a log line for a slow one; and a watchdog
    heartbeat."""

    report = report_of(rollout)
    if report.group_stats is not None:
        report.group_stats.scalar(**{"rollout_completion/rollout_time": elapsed})
    else:
        tracker.scalar(**{"rollout_completion/rollout_time": elapsed})
    if report.rollout_stats is not None:
        report.rollout_stats.add("rollout_time", [elapsed])
    if elapsed > 120.0:
        logger.warning(f"Rollout for task {task_id} took {elapsed:.1f}s (very slow)")
    elif elapsed > 60.0:
        logger.info(f"Rollout for task {task_id} took {elapsed:.1f}s (slow)")
    if watchdog:
        watchdog.heartbeat()


async def _run_group(
    workflow: RolloutWorkflow, row: dict[str, Any], *, watchdog: "Watchdog | None", tracker: StatsTracker
) -> Any:
    """Run the group of ``row`` and return its workflow's output; None for a workflow that raised, or whose own
    await was cancelled while this group was not (an inner task it awaited was cancelled): the group failed,
    which ``tracker`` records at once (``rollout_completion/failed_rollouts``), no group output carrying it to the
    step that consumes the group.  A cancellation of the group itself goes on, and so does the error of a credit
    estimator (:class:`~platoon.train.credit.runtime.CreditEstimatorError`), which stops the run."""

    task_id = row.get("task_id", "unknown")
    try:
        return await _await_rollout_with_heartbeat(workflow.arun_episode(row), watchdog=watchdog)
    except CreditEstimatorError:
        raise
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is None or current.cancelling():
            raise
        logger.exception(f"The rollout group for task {task_id} was cancelled from within: it counts as failed")
    except Exception as e:
        logger.exception(f"Exception in rollout group for task {task_id}: {e}")
    tracker.scalar(**{"rollout_completion/failed_rollouts": 1.0})
    return None


async def _run_training_group(source: "RolloutSource", draw: BoundDraw, lease: DrawLease) -> None:
    """Run the group of one of the source's draws, whose label slot and rollout permits are taken, with the draw's
    ``lease`` as its workflow's :func:`~platoon.train.leases.rollout_lease`: release them once its workflow returns
    (calling ``lease``), settle its credit estimator's state (:func:`~platoon.train.credit.runtime.settle_credit`),
    record the group in the run's per-task statistics, and hand its output to the step that dequeues it
    (:meth:`Dispatcher.settle`); a group that failed settles with no output.  Under a group admission rule, the group
    is decided first (:func:`_settle_under_rule`).  Nothing is awaited from the settling of its credit to its hand-over,
    so a full checkpoint, which takes the estimators' state where it takes the groups it keeps, holds both or
    neither."""

    dispatcher, tracker, watchdog = source.dispatcher, source.tracker, source.watchdog
    assert dispatcher is not None and tracker is not None  # the source started it
    start_time = time.perf_counter()
    try:
        with holding(lease):
            rollout = await _run_group(source.workflow, source.episode_row(draw), watchdog=watchdog, tracker=tracker)
    finally:
        lease()
    _group_done(rollout, time.perf_counter() - start_time, draw.task_id, tracker, watchdog)
    settle_credit(rollout)
    recorder = source.recorder
    record = (
        recorder.settle(draw.row, entry=draw.entry, label=draw.label, output=rollout) if recorder is not None else None
    )
    if (admissions := source.admissions) is not None:
        _settle_under_rule(source, admissions, draw, rollout, record)
        return
    dispatcher.settle(draw, _Settled(draw.entry, draw.label, rollout, record, draw.key))


def _settle_under_rule(
    source: "RolloutSource", admissions: AdmissionSet, draw: BoundDraw, rollout: Any, record: Any
) -> None:
    """Decide a settled group by the run's group admission rule (RFC 0019, section 6): an admitted group settles
    as any group does, with its reason; a rejected or failed one gets its fate at once (with the rule and its
    reason), leaves the outstanding draws, and only its marker goes to the result queue (:meth:`Dispatcher.reject`).
    A rejected group with datums frees its slot at once; one that failed or carried no datum, which can have ended
    at once, keeps it until a step dequeues its marker.  Synchronous, and never raises: nothing runs between the
    fate and the reject, so a checkpoint finds the statistics' outstanding draws and the dispatcher's alike."""

    dispatcher = source.dispatcher
    assert dispatcher is not None  # the source started it
    decision, rule = admissions.at_settle(draw, rollout, record, now=source.version())
    if decision.admit:
        dispatcher.settle(draw, _Settled(draw.entry, draw.label, rollout, record, draw.key, admitted=decision.reason))
        return
    held = not carries_datums(rollout)  # before its datums are dropped
    marker = _Settled(
        draw.entry,
        draw.label,
        emptied(rollout) if rollout is not None else None,
        record,
        None,
        rejected=decision.reason,
        window=dispatcher.step,
        held=held,
    )
    fate = "failed" if rollout is None else "rejected"
    recorder = source.recorder
    if recorder is not None and record is not None:
        recorder.fate(record, fate, rule=rule, admission=decision.reason)
    elif recorder is not None:
        recorder.fate_of(draw.key, fate, rule=rule, admission=decision.reason)
    dispatcher.reject(draw, marker, hold=held)
