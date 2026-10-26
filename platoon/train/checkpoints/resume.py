"""Resuming from a checkpoint: the policy counters it saved, and the draws it left outstanding, the groups it kept
and its samplers' states, which the dispatcher takes up.

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from platoon.train.checkpoints.dispatch_state import (
    plan_resume,
    read_state,
    saved_draws,
    saved_fallback_slots,
    saved_token_positions,
)
from platoon.train.checkpoints.kept_groups import decode_group, read_kept
from platoon.train.checkpoints.saving import ORDER_POSITION_KEY
from platoon.train.dispatch.dispatcher import Dispatcher
from platoon.train.dispatch.settled import _Settled
from platoon.train.sampling import fingerprint_changes
from platoon.train.task_records import record_sampling_changed
from platoon.train.task_stats import DISPATCH_KEY

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


def _resume_policy_counters(resume_info, num_minibatches: int) -> tuple[int, int]:
    """The ``policy_version`` and ``optimizer_steps`` a checkpoint saved.  A checkpoint without them comes from a
    trainer that published once per consumed batch: one policy version and ``num_minibatches`` optimizer steps."""

    batch = int(resume_info.batch or 0) if resume_info is not None else 0
    extra = resume_info.extra if resume_info is not None else {}
    policy_version = extra.get("policy_version", batch)
    optimizer_steps = extra.get("optimizer_steps", batch * num_minibatches)
    for name, value in (("policy_version", policy_version), ("optimizer_steps", optimizer_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint {name} must be a non-negative integer")
    return policy_version, optimizer_steps


def _resume_position(dispatcher: Dispatcher, extra: Mapping[str, Any]) -> bool:
    """For a resume that cannot read the dispatcher's state: move the order to where the checkpoint's row says it
    stood (``next_position``, which a checkpoint saved under a group admission rule records), so that the resume
    draws no position again that the run had consumed (a step under a rule draws up to its cap's worth of
    positions, so the checkpoint's batch lags behind).  Whether the row said, and the order moved."""

    position = extra.get(ORDER_POSITION_KEY)
    if isinstance(position, bool) or not isinstance(position, int) or position < dispatcher.next_position:
        return False
    dispatcher.next_position = position
    return True


def _restore_dispatch(
    trainer: "PlatoonTinkerRLTrainer", dispatcher: Dispatcher, resume_info: Any, policy_version: int
) -> None:
    """Take up the draws that the checkpoint this start resumes from (``resume_info``) left outstanding, from the
    dispatcher's state its row names (``dispatch``): the groups it kept go to the steps first, the draws that waited
    for their label back to their backlogs, and the rest are drawn again, each from the row it drew (the pool's
    first row of its task when that row no longer holds it, or a state of platoon 0.3.0 named none); the order goes
    on where it stopped, each task sampler at its saved count, else at the count the position implies.  A
    change of the sampling inputs is warned of and recorded in ``lock.json`` (``sampling_changed``), and the run
    goes on with it: the order goes on at the same position, and a draw whose task left the pool counts as lost.
    A checkpoint without that state (one saved before platoon 0.3.0, or one whose write failed), or a state that
    cannot be read, has the outstanding draws the per-task statistics hold counted as lost, and the order goes on
    from the checkpoint's batch, as the producer had it; or, from a checkpoint saved under a group admission rule,
    from where its row says the order stood (:meth:`_resume_position`)."""

    checkpoint = str(resume_info.name)
    recorder = getattr(trainer, "_recorder", None)
    extra = getattr(resume_info, "extra", None) or {}
    record = extra.get(DISPATCH_KEY)
    if not isinstance(record, Mapping):
        lost = recorder.lose_outstanding() if recorder is not None else 0
        if trainer._resume_position(dispatcher, extra):
            logger.info(
                "Checkpoint %s saved no state of the run's draws (its write failed): the run draws from where its "
                "order stood, position %d, which its row records, and %d draws that no committed step had trained "
                "count as lost",
                checkpoint,
                dispatcher.next_position,
                lost,
            )
            return
        logger.info(
            "Checkpoint %s saved no state of the run's draws (a platoon before 0.3.0 saved none, or its write "
            "failed): the run draws from the order where the checkpoint's batch begins, and %d draws that no "
            "committed step had trained count as lost",
            checkpoint,
            lost,
        )
        return
    pool = dispatcher.samplers.pool
    try:
        state = read_state(trainer.run_log_path, record)
        position = state.get("next_position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ValueError(f"next_position {position!r} is no position")
        kept = trainer._read_kept_groups(state.get("kept"))
        drawn, saved_tokens, fallback_slots = trainer._saved_samplings(state, checkpoint)
        plan = plan_resume(
            outstanding=state.get("outstanding") or [],
            waiting=state.get("waiting") or {},
            kept=kept,
            pool=pool,
        )
        # Each draw drawn again takes the row it drew: its pool row while that row still holds its task, else the
        # pool's first row of the task (the pool changed, or a state of format 1 saved no row).
        rows: dict[tuple[int, int], dict[str, Any]] = {}
        taken: dict[tuple[int, int], tuple[int | None, str | None]] = {}
        for key, task_id, _place in [*plan.reissue, *plan.waiting]:
            index, by = drawn.get(key, (None, None))
            found = pool.reissue_row(task_id, index)
            if found is None:
                raise ValueError(f"no row of the training tasks holds task {task_id!r}")
            rows[key], taken[key] = found[0], (found[1], by)
        for key, _task_id, _place in plan.kept:
            taken[key] = drawn.get(key, (None, None))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        if recorder is not None:
            recorder.lose_outstanding()
        moved = trainer._resume_position(dispatcher, extra)
        logger.warning(
            "The state of the run's draws that checkpoint %s saved cannot be read (%s): the draws it left "
            "outstanding are lost, and the order goes on from %s",
            checkpoint,
            exc,
            f"position {dispatcher.next_position}, where its row says it stood" if moved else "the checkpoint's batch",
        )
        return
    if recorder is not None:
        # The draws an older snapshot of the statistics holds as outstanding that the checkpoint no longer does: the
        # statistics lost what became of them, when the checkpoint's own snapshot did not read back.
        recorder.lose_outstanding(keep=plan.keys())
        for key, task_id, _position in plan.lost:
            recorder.lose(key, task_id)
    if changes := fingerprint_changes(state.get("fingerprint"), dispatcher.fingerprint):
        logger.warning(
            "The run's sampling inputs changed since checkpoint %s: %s. The run goes on with them, from where the "
            "order stopped, and the draws of tasks that left the run's training tasks are lost",
            checkpoint,
            "; ".join(changes),
        )
        record_sampling_changed(
            trainer.run_log_path, {"start": dispatcher.start, "checkpoint": checkpoint, "changes": changes}
        )
    dispatcher.next_position = position
    dispatcher.samplers.restore(state.get("samplers"), next_position=position)
    dispatcher.restore(
        kept=[(key, task_id, place, kept[key]) for key, task_id, place in plan.kept],
        reissue=[(key, rows[key], place) for key, _task_id, place in plan.reissue],
        waiting=[(key, rows[key], place) for key, _task_id, place in plan.waiting],
        version=policy_version,
        drawn=taken,
    )
    tokens = dispatcher.restore_tokens(saved_tokens, version=policy_version, fallbacks=fallback_slots)
    if tokens["dropped"]:
        logger.warning(
            "Resuming from checkpoint %s: %d slots that waited for an entry's task sampler are dropped, since "
            "their entry no longer gets slots (each logged as slot_skipped in tasks/events.jsonl)",
            checkpoint,
            tokens["dropped"],
        )
    logger.info(
        "Resuming from checkpoint %s: %d groups that had settled and no step had trained train first, %d draws "
        "that had not trained are drawn again first%s, and the order goes on at position %d",
        checkpoint,
        len(plan.kept),
        len(plan.reissue) + len(plan.waiting),
        f" ({len(plan.lost)} lost: their tasks left the run)" if plan.lost else "",
        position,
    )


def _saved_samplings(
    state: Mapping[str, Any], checkpoint: str
) -> tuple[dict[tuple[int, int], tuple[int | None, str | None]], dict[str, list[int]], frozenset[int]]:
    """What a dispatcher's state of format 2 adds to platoon 0.3.0's: the row each outstanding draw drew and the
    sampler that drew it, each entry's tokens, and those whose slot the run sampler's fallback handed.  A part that
    cannot be read is left out with a warning, the rest of the state taken up: without the rows, each draw drawn
    again takes the first row of its task, as from a state of format 1; without the tokens, their slots are
    lost."""

    try:
        drawn = saved_draws(state)
    except ValueError as exc:
        logger.warning(
            "Resuming from checkpoint %s: %s; each draw drawn again takes the first row of its task instead",
            checkpoint,
            exc,
        )
        drawn = {}
    try:
        tokens, fallbacks = saved_token_positions(state), saved_fallback_slots(state)
    except ValueError as exc:
        logger.warning(
            "Resuming from checkpoint %s: %s; the slots that waited for an entry's task sampler are lost",
            checkpoint,
            exc,
        )
        tokens, fallbacks = {}, frozenset()
    return drawn, tokens, fallbacks


def _read_kept_groups(trainer: "PlatoonTinkerRLTrainer", record: Any) -> dict[tuple[int, int], "_Settled"]:
    """The groups the checkpoint a start resumes from kept, as the steps dequeue them, by draw, in the order they
    settled; none, with a warning, when their file cannot be read, so that their tasks are rolled out again."""

    if not isinstance(record, Mapping):
        return {}
    try:
        groups = [decode_group(group) for group in read_kept(trainer.run_log_path, record)]
    except ValueError as exc:
        logger.warning(
            "The groups the checkpoint kept, which had settled and no step had trained yet, cannot be read, so "
            "their tasks are rolled out again: %s",
            exc,
        )
        return {}
    return {
        key: _Settled(entry, label, output, group_record, key) for key, entry, label, group_record, output in groups
    }
