"""A settled group as a step dequeues it (``_Settled``): its output with the entry and label of its row, its record
and its draw, and the markers of the groups an admission rule decided when they settled.  And what the settled groups
add to the statistics of the training step or the evaluation that consumes them: their statistics before any group,
the reward keys trained so far and each group's own statistics.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from platoon.train.workflow.protocol import OutputReport, report_of
from platoon.utils.rollout_stats import REWARD_KEY_MISMATCH, REWARD_KEY_PREFIX, RolloutStats, outcome_counts
from platoon.utils.stats_tracker import StatsTracker

# What a label's own metrics (a step's ``batch/<column>/<label>/...``, an evaluation's ``<column>/<label>/...``) keep of
# its groups' rollout statistics, beside their group and outcome counts (platoon.train.labels): the reward, and the
# steps of each trajectory, which an entry's own step limit shapes.
LABEL_STATS = ("task_reward", "num_steps")


@dataclass(frozen=True)
class _Settled:
    """A group's output as a rollout group hands it to the step that consumes it, with the environment entry and the
    label of the row it ran.  The stale filter and the label metrics read those from the row, which the group holds,
    not from the output: a custom workflow's plain list of datums cannot carry them.

    - ``record``: the group's record in the run's per-task statistics, when the trainer keeps them;
    - ``key``: its draw's ``(start, seq)``, by which the dispatcher keeps it until a committed step consumes it.

    Under a group admission rule (RFC 0019, section 6), ``admitted`` is an admitted group's reason.  A group rejected
    or failed when it settled goes to the step as a marker:

    - ``rejected``: its reason;
    - ``window``: the step current when it settled;
    - no ``key``, and its output without its datums (None for a failed group's);
    - ``held``: whether its group kept its slot of the horizon until a step dequeues it (a group that failed or
      carried no datum, :meth:`~platoon.train.dispatch.dispatcher.Dispatcher.reject`)."""

    entry: str | None
    label: str | None
    rollout: Any
    record: Any = None
    key: tuple[int, int] | None = None
    admitted: str | None = None
    rejected: str | None = None
    window: int | None = None
    held: bool = False


def carries_datums(output: Any) -> bool:
    """Whether a settled group's output carries a datum to train: a list (or other sized output) with one, or an output
    whose size is unknown.  A group whose output carries none (it failed, or its verdict is ``no_results``,
    ``no_completed_root`` and the like) can have ended at once.  Never raises."""

    if output is None:
        return False
    try:
        return len(output) > 0
    except Exception:
        return True


def emptied(output: Any) -> Any:
    """A rejected group's output as its marker carries it, so that the step that dequeues the marker reads of it what
    it reads of a list's.  Never raises.

    - A list (a ``GroupRolloutWorkflow`` output, any list subclass) is emptied of its datums in place, its other
      attributes kept.
    - Any other output (a ``RolloutOutput``) is replaced by an empty list that keeps its report (:class:`_Emptied`).
    - A list that refuses to be emptied is replaced by an empty list."""

    if isinstance(output, list):
        try:
            del output[:]
            return output
        except Exception:
            return []
    return _Emptied(report_of(output))


class _Emptied(list):
    """A rejected group's output that is no list, as its marker carries it: no datums, and the output's report."""

    def __init__(self, report: OutputReport) -> None:
        super().__init__()
        self.report = report


def _output_label(rollout: Any) -> str | None:
    """The label a group's output gives in its report (``GroupRolloutWorkflow``'s ``label``), if any."""

    return report_of(rollout).label


def _settled(item: Any) -> _Settled:
    """A result queue item as a worker's :class:`_Settled`: itself, or a bare output (a test's or a custom loop's),
    whose entry and label are its report's (``GroupRolloutWorkflow``'s ``entry`` and ``label``), when it gives them."""

    if isinstance(item, _Settled):
        return item
    report = report_of(item)
    return _Settled(report.entry, report.label, item)


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
