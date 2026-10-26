"""The metrics a training step logs of the groups it dequeued, some of which an evaluation logs too: the statistics
before any group, the reward keys trained so far, each group's own statistics and the programmatic-tool-calling
counters.

And what a straggler policy censors (RFC 0020, :class:`Censoring`): each step's ``batch/censoring/*`` metrics, and each
label's.

Over the groups a step dequeues that a straggler policy applies to (stale and rejected ones included), from the members
their records in the per-task statistics carry: ``censored_share``, the share of rollouts the policy cancelled; and
over the groups whose valid rollouts both passed and failed (``mixed_groups``), the pass rate of each group's slowest
valid rollout (``slowest_pass_rate``) against that of its others (``others_pass_rate``).  Slow rollouts fail more
often, so a cut of slow rollouts keeps a sample that leans the way that gap does.  A run without a straggler policy logs
none of them.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from platoon.train.task_stats import MemberResult
from platoon.utils.rollout_stats import REWARD_KEY_MISMATCH, REWARD_KEY_PREFIX, RolloutStats, outcome_counts
from platoon.utils.stats_tracker import DeferredStats, StatsTracker

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
    """Record what a group's workflow measured about the group (its ``group_stats``) in the consuming step.

    ``GroupRolloutWorkflow`` keeps a group's completion statistics, credit and workload distributions with the
    group instead of recording them when it completes: with prefetching, groups of the next step complete during
    this one, and their statistics would land in this step (a step counting 20 rollouts against the 16 it trained).
    """

    group_stats = getattr(rollout, "group_stats", None)
    if isinstance(group_stats, DeferredStats):
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
