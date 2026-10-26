"""A settled group as a step dequeues it (``_Settled``): its output with the entry and label of its row, its record
and its draw, and the markers of the groups an admission rule decided when they settled.
"""

from dataclasses import dataclass
from typing import Any

from platoon.train.workflow.protocol import OutputReport, report_of


@dataclass(frozen=True)
class _Settled:
    """A group's output as a rollout group hands it to the step that consumes it, with the environment entry and the
    label of the row it ran: the stale filter and the label metrics read them from the row, which the group holds,
    and not from the output, which a custom workflow's plain list of datums cannot carry.  ``record`` is the group's
    record in the run's per-task statistics, when the trainer keeps them, and ``key`` its draw's ``(start, seq)``,
    which the dispatcher keeps it by until a committed step consumes it.

    Under a group admission rule (RFC 0019, section 6), ``admitted`` is an admitted group's reason, and a group
    rejected or failed when it settled goes to the step as a marker: ``rejected`` its reason, ``window`` the step
    current when it settled, no ``key``, its output without its datums (None for a failed group's), and ``held``
    whether its group kept its slot of the horizon until a step dequeues it (a group that failed or carried no datum,
    :meth:`~platoon.train.dispatch.dispatcher.Dispatcher.reject`)."""

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
    """A rejected group's output as its marker carries it: a list (a ``GroupRolloutWorkflow`` output, any list
    subclass) emptied of its datums in place, its other attributes kept; any other output (a ``RolloutOutput``)
    replaced by an empty list that keeps its report (:class:`_Emptied`), so that the step that dequeues the marker
    reads of it what it reads of a list's; a list that refuses to be emptied replaced by an empty list.  Never
    raises."""

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
