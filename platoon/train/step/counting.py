"""Which groups a step counts as it collects them, in the first stage of a step (:class:`GroupCounting`).  Without a
group admission rule, each group the step dequeues takes a place of its microbatch.  Under a rule (RFC 0019, section
6), each step trains the first ``batch_size`` admitted groups it dequeues that are not stale, and closes once it has
counted its cap (:mod:`platoon.train.admissions`).

The step itself, :class:`~platoon.train.step.step.TrainStep`, counts each group it takes here.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from platoon.train.admissions import AdmissionSet
from platoon.train.dispatch.settled import SettledGroup
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.step.collect import StepDispatch, StepTally


class GroupCounting:
    """How step ``step`` counts the groups it dequeues: each one (``admissions`` None), or by the run's group admission
    rules, under which ``counted`` is what the step counted toward its cap, ``admitted`` toward its batch, and
    ``closed`` whether it closed on its cap (only a step under a rule closes)."""

    def __init__(
        self, admissions: AdmissionSet | None, dispatch: "StepDispatch", tally: "StepTally", step: int
    ) -> None:
        self.admissions = admissions
        self.dispatch = dispatch
        self.tally = tally
        self.step = step
        self.counted = self.admitted = 0
        self.closed = False

    @property
    def counts_every_group(self) -> bool:
        """No rule: each item a step dequeues takes a place of its microbatch, counted before the step reads it, so
        that nothing it finds can skip it.  Under a rule only an admitted group that is not stale does, once taken."""

        return self.admissions is None

    def take(self, settled: SettledGroup, taken: str, label: str | None) -> None:
        """Under a rule, count a group of ``label`` that the step took as ``taken`` (``marker``: one the rule decided
        when it settled; else ``stale``, ``failed`` or ``admitted``), and of a marker of a rejected group, its task
        sampler's prediction (in the step's tally, as a consumed group's).  Nothing without a rule."""

        admissions = self.admissions
        if admissions is None:
            return
        counts = self.dispatch.took(taken, settled)
        self.counted += counts
        self.admitted += taken == "admitted"
        admissions.dequeued(label, taken, settled)
        record = settled.record
        if (
            taken == "marker"
            and settled.output is not None
            and record is not None
            and record.score is not None
            and (where := self.dispatch.prefix(record.sampler)) is not None
        ):  # a rejected group's prediction counts, as a consumed group's does (stale, failed not)
            self.tally.count_prediction(where, record)
        if self.counted >= admissions.cap:
            self.closed = True
        elif counts:
            admissions.waiting(self.step, counted=self.counted, admitted=self.admitted)

    def short_microbatch_weight(self, microbatch: int, groups: int, groups_per_microbatch: int) -> float | None:
        """Under a rule, the weight of a short step's partial last microbatch, after full ones, of ``groups`` groups:
        what makes its groups weigh as those of the full ones do.  None for any other microbatch."""

        if self.closed and microbatch > 0 and groups < groups_per_microbatch:
            return groups_per_microbatch / max(1, groups)
        return None

    def step_closed(self) -> None:
        """The step dequeues no more: under a rule, a short step, or one that admitted nothing, is warned of once
        each."""

        if self.admissions is not None:
            self.admissions.step_closed(self.step, counted=self.counted, admitted=self.admitted)

    def counted_without_admission(self) -> int | None:
        """Under a rule, of a step that admitted no group, the groups it counted; None otherwise."""

        return self.counted if self.admissions is not None and not self.admitted else None

    def record_step_metrics(
        self, tracker: StatsTracker, prefixes: Mapping[str, str], sampler_metrics: Mapping[str, Any] | None
    ) -> None:
        """Under a rule: what the step admitted, rejected and dropped, by label and reason, and what it counted, beside
        the task samplers' draws (``sampler_metrics``: their metrics, None without a dispatcher)."""

        if self.admissions is None:
            return
        draws = sampler_metrics.get("sampler/draws") if sampler_metrics is not None else None
        tracker.scalar(
            **self.admissions.step_metrics(prefixes=prefixes, counted=self.counted, admitted=self.admitted, draws=draws)
        )
