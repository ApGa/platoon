"""Collecting a step's groups, the first stage of a step: what the step tallies of the groups it dequeues, for its
metrics and its commit (:class:`StepTally`), and the dispatcher as a step tells it what it dequeued, took and committed
(:class:`StepDispatch`).  And what a step reads of each group beside its datums: the workload its output's report
gives, which a custom workflow need not give.

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, dequeues the groups and counts each here.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from platoon.train.dispatch.settled import LABEL_STATS, _no_groups_yet
from platoon.train.workflow.protocol import malformed_counts, report_of
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.rollout_workload import RolloutWorkload

if TYPE_CHECKING:
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.dispatch.settled import _Settled
    from platoon.train.dispatch.source import RolloutSource

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
    """The workload a group's output gives in its report (:func:`~platoon.train.workflow.protocol.report_of`), with
    its rollout counts; None when it gives none, as a custom workflow's need not.  Malformed counts it carries are
    logged (:func:`~platoon.train.workflow.protocol.malformed_counts`), and left out with its workload."""

    report = report_of(rollout)
    # Each of the five is checked, so that each is known to be set; a workload comes with its four counts.
    workload, requested, observed = report.workload, report.requested_rollouts, report.observed_rollouts
    trainable, retained = report.trainable_rollouts, report.task_retained_datums
    if workload is None or requested is None or observed is None or trainable is None or retained is None:
        if (malformed := malformed_counts(rollout)) is not None:
            logger.warning("Ignoring malformed task workload rollout counts: %s", malformed)
        return None
    return _TaskWorkloadPayload(workload, requested, observed, trainable, retained)


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
    # The fate of each of the step's groups in the per-task statistics, by draw, which the step commits.
    fates: dict[tuple[int, int], str] = field(default_factory=dict)
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
        """Count a group the step dequeued, stale or not: its draw, and the workload its output gives; the group's
        rollout statistics, when its output's report gives them."""

        self.keys.append(settled.key)
        rollout = settled.rollout
        self.total_tasks += 1
        rollout_stats = report_of(rollout).rollout_stats
        if rollout_stats is None:
            self.groups_without_stats += 1
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

    def record_fate(self, key: tuple[int, int] | None, fate: str) -> None:
        """The fate of the group of draw ``key`` in the per-task statistics, once the step commits (a group without a
        draw, which a test queued, has none)."""

        if key is not None:
            self.fates[key] = fate

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
    """The start's draws (:class:`~platoon.train.dispatch.source.RolloutSource`) and their dispatcher
    (:mod:`platoon.train.dispatch.dispatcher`) as a step tells them what it dequeued, took and committed, and that the
    next step started, and reads where the task samplers' metrics go and what held the dispatcher back.  When a test
    replaces the trainer's ``_dispatch`` and queues the groups itself, the loop has no dispatcher (None): then it is
    told nothing (the draws' per-task statistics, when the start keeps them, still get each group's fate), a group
    admission rule counts every group a step takes, and no step logs the task samplers' metrics.  A trainer that a test
    drives outside a start has no draws either (None)."""

    def __init__(self, draws: "RolloutSource | None", dispatcher: "Dispatcher | None") -> None:
        self.draws = draws
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

    def committed(self, keys: list[tuple[int, int] | None], fates: dict[tuple[int, int], str]) -> None:
        """The step committed: each of its groups' fate (``fates``, by draw) goes to the draws' statistics, and the
        dispatcher no longer keeps the groups it dequeued (:meth:`RolloutSource.committed`); without the loop's
        dispatcher, the fates alone (:meth:`RolloutSource.record_fates`)."""

        if self.draws is None:
            return
        if self.dispatcher is not None:
            self.draws.committed(keys, fates=fates)
        else:
            self.draws.record_fates(keys, fates=fates)

    def step_started(self) -> None:
        """The step published its policy: the draws' horizon moves one batch on (:meth:`RolloutSource.step_started`)."""

        if self.draws is not None and self.dispatcher is not None:
            self.draws.step_started()

    def blocked_shares(self) -> dict[str, float]:
        """What held the dispatcher back from starting groups since the last step (its ``dispatch/blocked_share/*``
        metrics); none without a dispatcher."""

        return self.dispatcher.blocked_shares() if self.dispatcher is not None else {}
