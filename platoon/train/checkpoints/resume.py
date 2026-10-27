"""Resuming from a checkpoint: where a start begins (:class:`StartPoint`: the checkpoint it resumes from, and for each
policy, :class:`PolicyStart`, the counters that checkpoint saved), and the draws it left outstanding, the groups it kept
and its samplers' states, which the start's rollout source takes up (``_restore_dispatch``, in the steps below it).

``_restore_dispatch`` and ``_read_kept_groups`` take the start's rollout source
(:class:`~platoon.train.dispatch.source.RolloutSource`), whose per-task statistics and run directory they read.  The
source calls ``_restore_dispatch`` as it resumes, as does the rl trainer's method of its name; the steps below it call
the others here as module functions.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tinker_cookbook import checkpoint_utils

from platoon.train.checkpoints.barrier import newest_barrier
from platoon.train.checkpoints.dispatch_state import (
    ResumePlan,
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
from platoon.train.policy.spec import MAIN, PolicyId
from platoon.train.sampling import fingerprint_changes
from platoon.train.task_records import TaskRecorder, record_sampling_changed
from platoon.train.task_stats import DISPATCH_KEY

if TYPE_CHECKING:
    from platoon.train.dispatch.source import RolloutSource

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


@dataclass(frozen=True)
class PolicyStart:
    """Where one policy begins a start: the checkpoint row it resumes from (``record``: its weights and its optimizer's
    state; None for a fresh run), and the policy version and optimizer steps that row saved."""

    policy: PolicyId
    record: checkpoint_utils.CheckpointRecord | None
    version: int
    optimizer_steps: int


@dataclass(frozen=True)
class StartPoint:
    """Where a start of training begins: the checkpoint row it resumes from (``record``, the run's last one in
    ``checkpoints.jsonl``; None for a fresh run), the step it starts at (``step``, the row's ``batch``), the policy
    version and optimizer steps that row saved (:func:`_resume_policy_counters`), the whole ``loop_state`` it holds
    (empty for a fresh run), where each part of the run's state finds the keys it saved, and where each trained policy
    begins (``policies``: the leader from that row, each other one from the row it names)."""

    record: checkpoint_utils.CheckpointRecord | None
    step: int
    policy_version: int
    optimizer_steps: int
    loop_state: Mapping[str, Any]
    policies: Mapping[PolicyId, PolicyStart] = field(default_factory=dict)

    # The row and the step under the names the rl trainer reads them by: its tests give its methods stand-in starts
    # that have only those (0.5.0 development names, which go once the tests give start points).

    @property
    def resume_info(self) -> checkpoint_utils.CheckpointRecord | None:
        return self.record

    @property
    def batch(self) -> int:
        return self.step

    @classmethod
    def read(
        cls, run_dir: str, num_minibatches: int, *, leader: PolicyId = MAIN, others: Sequence[PolicyId] = ()
    ) -> "StartPoint":
        """The start point of a start of the run in ``run_dir``, whose steps take ``num_minibatches`` optimizer steps
        each (what a checkpoint without the counters implies): ``leader``'s newest row in the run's
        ``checkpoints.jsonl``, for a run of one trained policy; with the run's other trained policies (``others``),
        the newest whose rows of them each one still has, and each one's start from the row it names
        (:func:`~platoon.train.checkpoints.barrier.newest_barrier`)."""

        if others:
            return cls._read_barrier(run_dir, num_minibatches, leader, others)
        resume_info = checkpoint_utils.get_last_checkpoint(run_dir)
        start_batch = (resume_info.batch or 0) if resume_info else 0
        policy_version, optimizer_steps = _resume_policy_counters(resume_info, num_minibatches)
        policies = {leader: PolicyStart(leader, resume_info, policy_version, optimizer_steps)}
        return cls(resume_info, start_batch, policy_version, optimizer_steps, _row_loop_state(resume_info), policies)

    @classmethod
    def _read_barrier(
        cls, run_dir: str, num_minibatches: int, leader: PolicyId, others: Sequence[PolicyId]
    ) -> "StartPoint":
        """The start point of a run of several trained policies: the leader's row of the newest checkpoint whose rows
        of the other policies hold, and each other policy's start from its row (a fresh run's at 0 without one)."""

        barrier = newest_barrier(run_dir, others)
        if barrier is None:  # a fresh run: every policy starts at 0
            fresh = {policy: PolicyStart(policy, None, 0, 0) for policy in (leader, *others)}
            return cls(None, 0, 0, 0, {}, fresh)
        record = barrier.record
        policy_version, optimizer_steps = _resume_policy_counters(record, num_minibatches)
        policies = {leader: PolicyStart(leader, record, policy_version, optimizer_steps)}
        for policy, row in barrier.rows.items():
            version, steps = _resume_policy_counters(row, num_minibatches)
            policies[policy] = PolicyStart(policy, row, version, steps)
        return cls(record, record.batch or 0, policy_version, optimizer_steps, _row_loop_state(record), policies)


def _row_loop_state(resume_info: checkpoint_utils.CheckpointRecord | None) -> dict[str, Any]:
    """The ``loop_state`` a checkpoint's row was saved with: its ``batch``, when it has one, then the other keys, which
    the row keeps as they were saved (``extra``).  Empty without a checkpoint."""

    if resume_info is None:
        return {}
    batch = {"batch": resume_info.batch} if resume_info.batch is not None else {}
    return {**batch, **resume_info.extra}


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


def _restore_dispatch(source: "RolloutSource", dispatcher: Dispatcher, resume_info: Any, policy_version: int) -> None:
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
    from where its row says the order stood (:func:`_resume_position`).

    The steps, in order: :func:`resume_without_state`, or :func:`read_resume_plan` (:func:`resume_unreadable_state`
    when it cannot), then :func:`account_and_warn` and :func:`apply_resume_plan`."""

    checkpoint = str(resume_info.name)
    recorder = source.recorder
    extra = getattr(resume_info, "extra", None) or {}
    record = extra.get(DISPATCH_KEY)
    if not isinstance(record, Mapping):
        resume_without_state(dispatcher, recorder, extra, checkpoint)
        return
    pool = dispatcher.samplers.pool
    try:
        saved = read_resume_plan(source, record, checkpoint, pool)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        resume_unreadable_state(dispatcher, recorder, extra, checkpoint, exc)
        return
    account_and_warn(source.run_dir, dispatcher, recorder, saved, checkpoint)
    apply_resume_plan(dispatcher, saved, checkpoint, policy_version)


def resume_without_state(
    dispatcher: Dispatcher, recorder: TaskRecorder | None, extra: Mapping[str, Any], checkpoint: str
) -> None:
    """A checkpoint whose row names no state of the dispatcher (``extra``, its ``loop_state``): the outstanding draws
    the per-task statistics hold count as lost, and the order goes on from where the row says it stood, else from the
    checkpoint's batch."""

    lost = recorder.lose_outstanding() if recorder is not None else 0
    if _resume_position(dispatcher, extra):
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


@dataclass(frozen=True)
class SavedDispatch:
    """The dispatcher's state that a checkpoint saved, read back and planned for the resume (:func:`read_resume_plan`):
    the state as read, the order's position, the groups it kept, what becomes of each outstanding draw (``plan``), the
    row each draw drawn again takes, each draw's pool row and the sampler that drew it (``taken``), each entry's tokens
    and those whose slot the run sampler's fallback handed."""

    state: Mapping[str, Any]
    position: int
    kept: dict[tuple[int, int], _Settled]
    plan: ResumePlan
    rows: dict[tuple[int, int], dict[str, Any]]
    taken: dict[tuple[int, int], tuple[int | None, str | None]]
    saved_tokens: dict[str, list[int]]
    fallback_slots: frozenset[int]


def read_resume_plan(source: "RolloutSource", record: Mapping[str, Any], checkpoint: str, pool: Any) -> SavedDispatch:
    """Read the dispatcher's state that the checkpoint's row names (``record``), the groups it kept and the samplings
    it saved, and plan the resume over the run's task ``pool``: an ``AttributeError``, ``KeyError``, ``TypeError`` or
    ``ValueError`` for a state that cannot be read or taken up, such as a draw whose task no row holds."""

    state = read_state(source.run_dir, record)
    position = state.get("next_position")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise ValueError(f"next_position {position!r} is no position")
    kept = _read_kept_groups(source, state.get("kept"))
    drawn, saved_tokens, fallback_slots = _saved_samplings(state, checkpoint)
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
    return SavedDispatch(state, position, kept, plan, rows, taken, saved_tokens, fallback_slots)


def resume_unreadable_state(
    dispatcher: Dispatcher, recorder: TaskRecorder | None, extra: Mapping[str, Any], checkpoint: str, exc: Exception
) -> None:
    """A state of the dispatcher that cannot be read (``exc``): the draws it left outstanding are lost, and the order
    goes on from where the checkpoint's row (``extra``) says it stood, else from the checkpoint's batch."""

    if recorder is not None:
        recorder.lose_outstanding()
    moved = _resume_position(dispatcher, extra)
    logger.warning(
        "The state of the run's draws that checkpoint %s saved cannot be read (%s): the draws it left "
        "outstanding are lost, and the order goes on from %s",
        checkpoint,
        exc,
        f"position {dispatcher.next_position}, where its row says it stood" if moved else "the checkpoint's batch",
    )


def account_and_warn(
    run_dir: str,
    dispatcher: Dispatcher,
    recorder: TaskRecorder | None,
    saved: SavedDispatch,
    checkpoint: str,
) -> None:
    """Before the dispatcher takes the state up: the per-task statistics lose the draws the resume does not take up,
    and a change of the run's sampling inputs since the checkpoint is warned of and recorded in the ``lock.json`` of
    ``run_dir``."""

    if recorder is not None:
        # The draws an older snapshot of the statistics holds as outstanding that the checkpoint no longer does: the
        # statistics lost what became of them, when the checkpoint's own snapshot did not read back.
        recorder.lose_outstanding(keep=saved.plan.keys())
        for key, task_id, _position in saved.plan.lost:
            recorder.lose(key, task_id)
    if changes := fingerprint_changes(saved.state.get("fingerprint"), dispatcher.fingerprint):
        logger.warning(
            "The run's sampling inputs changed since checkpoint %s: %s. The run goes on with them, from where the "
            "order stopped, and the draws of tasks that left the run's training tasks are lost",
            checkpoint,
            "; ".join(changes),
        )
        record_sampling_changed(run_dir, {"start": dispatcher.start, "checkpoint": checkpoint, "changes": changes})


def apply_resume_plan(dispatcher: Dispatcher, saved: SavedDispatch, checkpoint: str, policy_version: int) -> None:
    """The dispatcher takes the state up: the order's position, the task samplers' states, the kept groups, the draws
    drawn again and those back in their labels' backlogs, at ``policy_version``, and each entry's tokens."""

    plan, position = saved.plan, saved.position
    dispatcher.next_position = position
    dispatcher.samplers.restore(saved.state.get("samplers"), next_position=position)
    dispatcher.restore(
        kept=[(key, task_id, place, saved.kept[key]) for key, task_id, place in plan.kept],
        reissue=[(key, saved.rows[key], place) for key, _task_id, place in plan.reissue],
        waiting=[(key, saved.rows[key], place) for key, _task_id, place in plan.waiting],
        version=policy_version,
        drawn=saved.taken,
    )
    tokens = dispatcher.restore_tokens(saved.saved_tokens, version=policy_version, fallbacks=saved.fallback_slots)
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


def _read_kept_groups(source: "RolloutSource", record: Any) -> dict[tuple[int, int], "_Settled"]:
    """The groups the checkpoint a start resumes from kept, as the steps dequeue them, by draw, in the order they
    settled; none, with a warning, when their file cannot be read, so that their tasks are rolled out again."""

    if not isinstance(record, Mapping):
        return {}
    try:
        groups = [decode_group(group) for group in read_kept(source.run_dir, record)]
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
