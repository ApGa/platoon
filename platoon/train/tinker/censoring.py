"""What a straggler policy censors (RFC 0020): each step's ``batch/censoring/*`` metrics, and each label's.

Over the groups a step dequeues that a straggler policy applies to (stale and rejected ones included), from the members
their records in the per-task statistics carry: ``censored_share``, the share of rollouts the policy cancelled; and
over the groups whose valid rollouts both passed and failed (``mixed_groups``), the pass rate of each group's slowest
valid rollout (``slowest_pass_rate``) against that of its others (``others_pass_rate``).  Slow rollouts fail more
often, so a cut of slow rollouts keeps a sample that leans the way that gap does.  A run without a straggler policy logs
none of them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from platoon.train.task_stats import MemberResult


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
