"""A step's metrics of what the dispatcher did since the step before (``platoon.train.step.metrics``): the draws a
capped label's backlog skipped and the slots owed to adaptive entries that were skipped, each a step's own count, from
the dispatcher's running totals (``RunTotals`` holds what earlier steps logged)."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from platoon.train.step.collect import StepTally
from platoon.train.step.metrics import RunTotals, record_label_limits, record_sampler_metrics
from platoon.utils.stats_tracker import StatsTracker


class _Dispatcher:
    """What the metrics read of a dispatcher: its running totals, which a test moves on between steps."""

    def __init__(self) -> None:
        self.skipped_slots = 0
        self.skipped: Counter[str | None] = Counter()
        self.samplers = SimpleNamespace(step_metrics=lambda version: {}, prefix=lambda name: "sampler/")

    def tokens_by_entry(self) -> dict:
        return {}

    def parked_by_label(self) -> Counter[str | None]:
        return Counter()


def test_each_step_logs_the_sampler_slots_skipped_since_the_step_before():
    dispatcher, totals = _Dispatcher(), RunTotals()
    logged = []
    for total in (3, 5, 5):
        dispatcher.skipped_slots = total
        tracker = StatsTracker("train")
        record_sampler_metrics(tracker, dispatcher, totals, StepTally.empty([]), 0)
        logged.append(tracker.export()["train/sampler/skipped_slots"])

    assert logged == [3.0, 2.0, 0.0]


def test_each_step_logs_the_rows_a_capped_label_s_backlog_skipped_since_the_step_before():
    dispatcher, totals = _Dispatcher(), RunTotals()
    logged = []
    for skipped in (Counter({"slow": 2}), Counter({"slow": 3}), Counter({"slow": 3})):
        dispatcher.skipped = skipped
        tracker = StatsTracker("train")
        record_label_limits(tracker, dispatcher, totals, ["slow"], "environment")
        stats = tracker.export()
        logged.append((stats["train/batch/skipped_rows"], stats["train/batch/environment/slow/skipped_rows"]))

    assert logged == [(2.0, 2.0), (1.0, 1.0), (0.0, 0.0)]
