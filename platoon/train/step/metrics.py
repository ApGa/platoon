"""The metrics a training step logs, its last stage (the seventh), some of which an evaluation logs too: the
statistics before any group, the reward keys trained so far, each group's own statistics and the
programmatic-tool-calling counters; and the step's metrics in the order the step writes them
(:func:`record_batch_stats`, :func:`record_task_metrics`, :func:`record_sampler_metrics`, :func:`record_label_limits`,
:func:`record_batch_counts`), with what they carry from step to step (:class:`RunTotals`).

And what a straggler policy censors (RFC 0020, :class:`Censoring`): each step's ``batch/censoring/*`` metrics, and each
label's.

Over the groups a step dequeues that a straggler policy applies to (stale and rejected ones included), from the members
their records in the per-task statistics carry: ``censored_share``, the share of rollouts the policy cancelled; and
over the groups whose valid rollouts both passed and failed (``mixed_groups``), the pass rate of each group's slowest
valid rollout (``slowest_pass_rate``) against that of its others (``others_pass_rate``).  Slow rollouts fail more
often, so a cut of slow rollouts keeps a sample that leans the way that gap does.  A run without a straggler policy logs
none of them.

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, writes its metrics through these, then exports
them with the run's counters and logs them.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from platoon.train.labels import metric_prefix
from platoon.train.task_stats import MemberResult
from platoon.train.workflow.protocol import report_of
from platoon.utils.rollout_stats import REWARD_KEY_MISMATCH, REWARD_KEY_PREFIX, RolloutStats, outcome_counts
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.dispatch.dispatcher import Dispatcher
    from platoon.train.step.collect import StepTally
    from platoon.train.task_records import TaskRecorder
    from platoon.train.trainer.state import TrainLoopSharedState

# What a label's own metrics (a step's ``batch/<column>/<label>/...``, an evaluation's ``<column>/<label>/...``) keep of
# its groups' rollout statistics, beside their group and outcome counts (platoon.train.labels): the reward, and the
# steps of each trajectory, which an entry's own step limit shapes.
LABEL_STATS = ("task_reward", "num_steps")


def _no_groups_yet() -> RolloutStats:
    """A step's statistics, or a label's, before any group: no groups, no rollout of any outcome status and no group
    whose rollouts trained different reward keys, so that each count is logged every step, also at zero."""

    return RolloutStats(counts={**outcome_counts(()), REWARD_KEY_MISMATCH: 0})


def _count_every_reward_key(seen: set[str], stats: Iterable[RolloutStats]) -> None:
    """Add the ``reward_key/<key>`` counts of ``stats`` to ``seen``, the reward keys trained so far, and give each of
    ``stats`` every key seen, at zero when it trained none: each key is logged every step once it has trained."""

    stats = list(stats)
    seen.update(name for item in stats for name in item.counts if name.startswith(REWARD_KEY_PREFIX))
    for item in stats:
        for name in seen:
            item.counts.setdefault(name, 0)


def _record_group_stats(rollout: Any, tracker: StatsTracker) -> None:
    """Record what a group's workflow measured about the group (its report's ``group_stats``) in the consuming step.

    ``GroupRolloutWorkflow`` keeps a group's completion statistics, credit and workload distributions with the
    group instead of recording them when it completes: with prefetching, groups of the next step complete during
    this one, and their statistics would land in this step (a step counting 20 rollouts against the 16 it trained).
    """

    group_stats = report_of(rollout).group_stats
    if group_stats is not None:
        group_stats.record(tracker)


def _ptc_cell_guard_stats(stats_scope: str = "train") -> dict[str, float]:
    """Counters for programmatic-tool-calling cells that have not returned.

    A PTC cell runs in-process on a rollout worker thread with no timeout, and a
    Python thread cannot be killed, so a never-terminating cell consumes a
    worker thread for the rest of the run without raising anything. The guard
    that counts them ships with the extension that owns programmatic tool
    calling and reports through the ``rollout_worker.telemetry`` hook; core does
    not depend on it, so a run without that extension -- or any failure inside
    the guard -- contributes no scalars instead of failing. Never raises: this
    is called inline in the train loop's stats path.

    With ``rollout_isolation: process`` the guard's registry lives in the
    *workers*, so this process's own snapshot is permanently empty.  Because
    this function swallows everything and returns ``{}``, that regression would
    be completely silent under unchanged key names -- so worker-reported
    counters take precedence when there are any.  They are deliberately the
    cumulative ones plus ``*_at_exit`` gauges: summing the instantaneous
    ``ptc_cells_stuck`` across short-lived workers would read zero precisely
    when cells are being killed for hanging.  The instantaneous names keep
    their original meaning in ``inprocess`` mode.
    """

    try:
        from platoon.train.workers.telemetry import worker_cell_guard_stats

        forwarded = worker_cell_guard_stats(stats_scope)
    except Exception:
        forwarded = {}
    if forwarded:
        return forwarded
    from platoon.train.workers.telemetry import local_telemetry

    return local_telemetry()


@dataclass
class RunTotals:
    """What a start's step metrics carry from step to step: the reward keys trained so far, each logged every step once
    it has trained (:func:`_count_every_reward_key`), and what the dispatcher had skipped by the last step."""

    reward_keys: set[str] = field(default_factory=set)
    # Draws a capped label's full backlog skipped, logged so far.
    skipped_seen: Counter[str | None] = field(default_factory=Counter)
    # Slots owed to adaptive entries that were skipped, logged so far.
    skipped_slots_seen: int = 0


def record_batch_stats(tracker: StatsTracker, tally: "StepTally", totals: RunTotals, column: str) -> None:
    """The rollout statistics of the groups the step dequeued that are neither stale nor failed (under a group
    admission rule, the rejected groups' markers too), the batch's and each logged label's, beside each label's stale
    groups; each reward key trained so far among them, at zero when the step trained none."""

    _count_every_reward_key(totals.reward_keys, [tally.rollout_stats, *tally.label_stats.values()])
    tally.rollout_stats.record(tracker, prefix="batch")
    for label, stats in tally.label_stats.items():
        stats.record(tracker, prefix=f"batch/{metric_prefix(column, label)}")
        tracker.scalar(**{f"batch/{metric_prefix(column, label)}/stale_groups": float(tally.label_stale[label])})


def record_task_metrics(
    tracker: StatsTracker, recorder: "TaskRecorder", tally: "StepTally", prefixes: Mapping[str, str], version: int
) -> None:
    """The per-task statistics' metrics of the step, at policy ``version``, once every settled group is taken in."""

    recorder.flush()  # every settled group taken in first: a store that fails to stops the statistics
    if not recorder.stopped:
        # The kinds of the groups the step trained on, and what each label's tasks' groups say so far; nothing once
        # the statistics stopped, rather than zeros that read as no informative groups.
        for label, kinds in tally.kinds.items():
            where = f"batch/{prefixes[label]}/" if label else "batch/"
            for kind in ("informative", "all_pass", "all_fail", "uniform_mixed"):
                tracker.scalar(**{f"{where}{kind}_groups": float(kinds[kind])})
        tracker.scalar(**recorder.step_metrics(now=version, prefixes=prefixes))


def record_sampler_metrics(
    tracker: StatsTracker, dispatcher: "Dispatcher", totals: RunTotals, tally: "StepTally", version: int
) -> dict[str, float]:
    """What the task samplers drew since the last step and why, how long their decisions took, their own metrics, the
    slots waiting for an adaptive entry's sampler and those skipped; and, of the groups the step consumed, the share
    each sampler predicted would train against the share that did.  The samplers' own metrics, at policy ``version``,
    which it returns."""

    sampler_metrics = dispatcher.samplers.step_metrics(version)
    tracker.scalar(**sampler_metrics)
    for entry, count in dispatcher.tokens_by_entry().items():
        where = dispatcher.samplers.prefix(dispatcher.samplers.entries[entry].name)
        tracker.scalar(**{f"{where}tokens": float(count)})
    skipped_slots = dispatcher.skipped_slots - totals.skipped_slots_seen
    totals.skipped_slots_seen = dispatcher.skipped_slots
    tracker.scalar(**{"sampler/skipped_slots": float(skipped_slots)})
    for where, (scores, groups, trained) in tally.predictions.items():
        tracker.scalar(
            **{
                f"{where}predicted_informative": scores / groups,
                f"{where}informative_share": trained / groups,
            }
        )
    return sampler_metrics


def record_label_limits(
    tracker: StatsTracker, dispatcher: "Dispatcher", totals: RunTotals, labels: Sequence[str], column: str
) -> None:
    """Draws a capped label's full backlog skipped since the last step: its share of the order asks for more groups
    than its limit runs.  And the draws that wait for their label's limit now (parked)."""

    skipped = dispatcher.skipped - totals.skipped_seen
    totals.skipped_seen.update(skipped)
    parked = dispatcher.parked_by_label()
    tracker.scalar(**{"batch/skipped_rows": float(skipped.total()), "batch/parked_rows": float(parked.total())})
    for label in labels:
        where = f"batch/{metric_prefix(column, label)}"
        tracker.scalar(**{f"{where}/skipped_rows": float(skipped[label]), f"{where}/parked_rows": float(parked[label])})


def record_batch_counts(tracker: StatsTracker, tally: "StepTally", state: "TrainLoopSharedState") -> None:
    """What the step dequeued and how often it stepped the optimizer, the policy's version and the run's updates."""

    tracker.scalar(
        **{
            "batch/groups_dequeued": float(tally.total_tasks),
            "batch/stale_groups": float(tally.stale_groups),
            "batch/groups_without_stats": float(tally.groups_without_stats),
            "batch/optimizer_updates": float(tally.optimizer_steps),
            "batch/skipped_optimizer": float(tally.optimizer_steps == 0),
            "policy/version": float(state.model_info.llm.version),
            "optim/updates_total": float(getattr(state, "optimizer_steps", 0)),
        }
    )


class _Counts:
    """The batch's or one label's counts."""

    __slots__ = ("censored", "members", "mixed", "others", "others_passed", "slowest_passed")

    def __init__(self) -> None:
        self.members = self.censored = 0
        self.mixed = self.slowest_passed = self.others = self.others_passed = 0

    def add(self, members: Sequence[MemberResult], threshold: float) -> None:
        self.members += len(members)
        self.censored += sum(member.disposition == "straggler_killed" for member in members)
        valid = [member for member in members if member.valid]
        passed = [member.passes(threshold) for member in valid]
        timed = [index for index, member in enumerate(valid) if member.seconds is not None]
        if not timed or all(passed) or not any(passed):
            return  # not mixed, or no valid member says how long it took
        slowest = max(timed, key=lambda index: valid[index].seconds or 0.0)
        self.mixed += 1
        self.slowest_passed += passed[slowest]
        self.others += len(valid) - 1
        self.others_passed += sum(passed) - passed[slowest]

    def metrics(self, where: str) -> dict[str, float]:
        found = {f"{where}censoring/mixed_groups": float(self.mixed)}
        if self.members:
            found[f"{where}censoring/censored_share"] = self.censored / self.members
        if self.mixed:
            found[f"{where}censoring/slowest_pass_rate"] = self.slowest_passed / self.mixed
        if self.others:
            found[f"{where}censoring/others_pass_rate"] = self.others_passed / self.others
        return found


class Censoring:
    """Each step's censoring metrics (module docstring), from the groups the step dequeues (:meth:`add`).  ``settings``
    are the entries' settings (``platoon.train.entry_settings.run_settings``) and ``run`` the run's workflow config,
    which say where a straggler policy applies; ``threshold(entry)`` is the success threshold of an entry's groups."""

    def __init__(
        self, settings: Mapping[str, Any], run: Any, *, threshold: Callable[[str | None], float] | None = None
    ) -> None:
        self._settings = settings
        self._run = run
        self._threshold = threshold
        self._counts: dict[str | None, _Counts] = {}

    def add(self, label: str | None, record: Any) -> None:
        """A step dequeued a group of ``label``, with its record in the per-task statistics (None without one)."""

        if record is None or not record.members or not self._policy_applies(record.entry, label):
            return
        threshold = self._threshold(record.entry) if self._threshold is not None else 1.0
        for key in {None, label}:  # the batch's, and the label's
            self._counts.setdefault(key, _Counts()).add(record.members, threshold)

    def step_metrics(self, prefixes: Mapping[str, str]) -> dict[str, float]:
        """The step's metrics, each logged as ``train/<key>``: the batch's and each logged label's (``prefixes``: label
        -> its metric prefix) that a straggler policy applied to; then the next step starts."""

        counts, self._counts = self._counts, {}
        found = counts[None].metrics("batch/") if None in counts else {}
        for label, prefix in prefixes.items():
            if label in counts:
                found.update(counts[label].metrics(f"batch/{prefix}/"))
        return found

    def _policy_applies(self, entry: str | None, label: str | None) -> bool:
        """Whether a straggler policy cuts the groups of ``entry`` and ``label``, as ``GroupRolloutWorkflow`` reads
        it: the label's own policy in ``straggler_policies``, else the entry's ``straggler_timeout_seconds``."""

        own = self._settings.get(entry) if entry is not None else None
        workflow = own.train if own is not None else self._run
        policy = workflow.straggler_policies.get(label)
        timeout = policy.timeout_seconds if policy is not None else workflow.straggler_timeout_seconds
        return timeout is not None
