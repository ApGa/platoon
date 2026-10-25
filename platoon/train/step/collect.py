"""Collecting a step's groups, the first stage of a step: what the step tallies of the groups it dequeues, for its
metrics and its commit (:class:`StepTally`), and the dispatcher as a step tells it what it dequeued, took and committed
(:class:`StepDispatch`).  And what a step reads of each group beside its datums: the workload its output carries,
which a custom workflow need not give.

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, dequeues the groups and counts each here.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from platoon.train.step.metrics import LABEL_STATS, _no_groups_yet
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.rollout_workload import RolloutWorkload

if TYPE_CHECKING:
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.dispatch.settled import _Settled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TaskWorkloadPayload:
    """Validated workload sidechannel attached to a list-compatible result."""

    workload: RolloutWorkload
    requested_rollouts: int
    observed_rollouts: int
    trainable_rollouts: int
    task_retained_datums: int


def _get_task_workload_payload(rollout: Any) -> _TaskWorkloadPayload | None:
    """Read optional workload metadata without constraining custom workflows."""

    workload = getattr(rollout, "workload", None)
    if not isinstance(workload, RolloutWorkload):
        return None
    count_names = (
        "requested_rollouts",
        "observed_rollouts",
        "trainable_rollouts",
        "task_retained_datums",
    )
    counts = [getattr(rollout, name, None) for name in count_names]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        logger.warning("Ignoring malformed task workload rollout counts: %s", dict(zip(count_names, counts)))
        return None
    return _TaskWorkloadPayload(workload, *counts)


@dataclass(kw_only=True)
class StepTally:
    """What a step counts of the groups it dequeues and of the datums it sends, for its metrics, its log lines and its
    commit, by the stage that counts it.  ``label_stats`` and ``kinds`` hold each label's share, of the labels whose
    metrics the step logs (``kinds`` the whole batch's too, under ``""``); ``label_stale`` counts every label's stale
    groups.  :meth:`empty` builds the four without a default."""

    # --- collect: each group the step dequeued, stale or not ---------------------------------------------------------
    total_tasks: int = 0
    # The draws of the groups it dequeued: a checkpoint keeps them until the step commits.
    keys: list[tuple[int, int] | None] = field(default_factory=list)
    groups_without_stats: int = 0
    # What generating them took, of those whose output carries its workload.
    total_tasks_with_workload_metadata: int = 0
    workload: RolloutWorkload
    total_requested_rollouts: int = 0
    total_observed_rollouts: int = 0
    total_workflow_trainable_rollouts: int = 0
    total_task_retained_datums: int = 0
    stale_groups: int = 0
    label_stale: Counter[str | None] = field(default_factory=Counter)
    # The step's groups in the per-task statistics, each with its record and its draw, and its fate once the step
    # commits.
    fates: list[tuple[Any, tuple[int, int] | None, str]] = field(default_factory=list)
    # The kinds of the groups the step trained on, by label ("" for the whole batch).
    kinds: dict[str, Counter[str]]
    # Of the consumed groups a task sampler drew with a score, by the sampler's metric prefix: the sum of the scores
    # (each the chance it predicted that the group trains), the groups, and those that trained.
    predictions: dict[str, list[float]] = field(default_factory=dict)
    # Of the groups neither stale nor failed (a rejected group's marker too): their rollout statistics, each logged
    # label's share of them (its groups, rewards and outcome counts), and their datums.
    rollout_stats: RolloutStats
    label_stats: dict[str, RolloutStats]
    total_datums: int = 0
    # --- prepare: the datums the step sent to forward_backward, and the microbatches that took no forward/backward
    # (their objective was zero, or they had no datums) --------------------------------------------------------------
    submitted_training_datums: int = 0
    submitted_training_attention_tokens: int = 0
    submitted_training_loss_tokens: int = 0
    zero_objective_microbatches: int = 0
    empty_microbatches: int = 0
    # --- update ------------------------------------------------------------------------------------------------------
    optimizer_steps: int = 0

    @classmethod
    def empty(cls, labels: Sequence[str]) -> "StepTally":
        """A step's tally before its first group: no workload, no group's statistics, and each label of ``labels`` at
        zero."""

        return cls(
            workload=RolloutWorkload(),
            rollout_stats=_no_groups_yet(),
            label_stats={label: _no_groups_yet() for label in labels},
            kinds={"": Counter(), **{label: Counter() for label in labels}},
        )

    def count_dequeued(self, settled: "_Settled") -> RolloutStats | None:
        """Count a group the step dequeued, stale or not: its draw, and the workload its output carries; the group's
        rollout statistics, when its output has them."""

        self.keys.append(settled.key)
        rollout = settled.rollout
        self.total_tasks += 1
        rollout_stats = getattr(rollout, "rollout_stats", None)
        if not isinstance(rollout_stats, RolloutStats):
            self.groups_without_stats += 1
            rollout_stats = None
        workload_payload = _get_task_workload_payload(rollout)
        if workload_payload is not None:
            # Attribute generation work when this outer batch
            # actually consumes the queue item.  Producer-side
            # recording would leak prefetched future tasks into
            # the current step.  Stale work is deliberately
            # included so wasted inference remains visible.
            self.total_tasks_with_workload_metadata += 1
            self.workload += workload_payload.workload
            self.total_requested_rollouts += workload_payload.requested_rollouts
            self.total_observed_rollouts += workload_payload.observed_rollouts
            self.total_workflow_trainable_rollouts += workload_payload.trainable_rollouts
            self.total_task_retained_datums += workload_payload.task_retained_datums
        return rollout_stats

    def record_fate(self, record: Any, key: tuple[int, int] | None, fate: str) -> None:
        """A group in the per-task statistics, with its record and its draw: its fate once the step commits."""

        self.fates.append((record, key, fate))

    def commit_fates(self, recorder: Any) -> None:
        """Tell the per-task statistics (``recorder``) what became of each of the step's groups, as the step commits."""

        for record, key, fate in self.fates:
            if record is not None:
                recorder.fate(record, fate)
            elif key is not None:  # a group a start kept after its statistics stopped: no record of its own
                recorder.fate_of(key, fate)

    def count_stale(self, label: str | None) -> None:
        """A group of ``label`` that the stale filter dropped."""

        self.stale_groups += 1
        self.label_stale[label] += 1

    def count_kind(self, kind: str, label: str | None) -> None:
        """A consumed group of kind ``kind`` (``informative``, ``all_pass``, ...), of its row's ``label``."""

        self.kinds[""][kind] += 1
        if label in self.kinds:
            self.kinds[label][kind] += 1

    def count_prediction(self, where: str, record: Any) -> None:
        """A group that a task sampler, whose metrics go to ``where``, drew with a score: its record's score, the
        group, and whether it trained."""

        predicted = self.predictions.setdefault(where, [0.0, 0.0, 0.0])
        predicted[0] += record.score
        predicted[1] += 1
        predicted[2] += record.verdict == "trained"

    def merge_stats(self, rollout_stats: RolloutStats, label: str | None) -> None:
        """Merge the rollout statistics of a group of ``label`` that is neither stale nor failed (a rejected group's
        marker too) into the batch's and the label's."""

        self.rollout_stats.merge(rollout_stats)
        if label in self.label_stats:  # a label whose metrics are logged
            self.label_stats[label].merge(rollout_stats.subset(LABEL_STATS))

    def count_submitted(self, datums: int, attention_tokens: int, loss_tokens: int) -> None:
        """A microbatch the step sent to forward_backward: its datums, their attention tokens and their loss tokens."""

        self.submitted_training_datums += datums
        self.submitted_training_attention_tokens += attention_tokens
        self.submitted_training_loss_tokens += loss_tokens


class StepDispatch:
    """The dispatcher (:mod:`platoon.train.dispatch.dispatcher`) as a step tells it what it dequeued, took and
    committed, and that the next step started; or none, when a test replaces the trainer's ``_dispatch`` and queues the
    groups itself: then nothing is told, a group admission rule counts every group a step takes, and no step logs the
    task samplers' metrics."""

    def __init__(self, dispatcher: "Dispatcher | None") -> None:
        self.dispatcher = dispatcher

    def dequeued(self) -> None:
        """A step that counts every group dequeued one."""

        if self.dispatcher is not None:
            self.dispatcher.dequeued()

    def took(self, taken: str, settled: "_Settled") -> bool:
        """A step under a group admission rule took ``settled`` as ``taken``: whether that counts toward its cap."""

        return self.dispatcher.took(taken, settled.window, held=settled.held) if self.dispatcher is not None else True

    def prefix(self, sampler: str | None) -> str | None:
        """Where the metrics of the task sampler named ``sampler`` go; None without a dispatcher or such a sampler."""

        return self.dispatcher.samplers.prefix(sampler) if self.dispatcher is not None else None

    def committed(self, keys: list[tuple[int, int] | None]) -> None:
        """The step committed: the groups it dequeued are consumed, and a checkpoint no longer keeps them."""

        if self.dispatcher is not None:
            self.dispatcher.committed(keys)

    def step_started(self) -> None:
        """The step published its policy: the horizon moves one batch on."""

        if self.dispatcher is not None:
            self.dispatcher.step_started()

    def blocked_shares(self) -> dict[str, float]:
        """What held the dispatcher back from starting groups since the last step (its ``dispatch/blocked_share/*``
        metrics); none without a dispatcher."""

        return self.dispatcher.blocked_shares() if self.dispatcher is not None else {}
