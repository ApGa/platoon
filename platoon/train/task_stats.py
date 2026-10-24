"""Per-task statistics of a training run: what happened to every group, and what each task's groups say so far.

Each group's output says what happened: its task, its members' outcomes (a :class:`MemberResult` each) and the
workflow's verdict on what the group trains.  When a group settles, the trainer records it as a :class:`GroupRecord`
in the run's :class:`TaskStatsStore`, which keeps one record per training task (:class:`TaskStats`) and totals per
environment entry and per label (:class:`GroupTotals`): groups, attempts, passes, informative and uniform groups,
the current runs of uniform groups and of groups without a valid member, decayed evidence, decayed means of steps,
tokens, time and reward, the newest groups' summaries, and what became of each draw.  Every settled group counts once,
stale groups and failed ones included: the work happened, and a uniform group is the most telling evidence about a
task's difficulty.

**Passes.**  A member passes when it is valid (its root completed with a scored outcome, so it enters the baseline)
and the reward its environment scored, before any reward processor, reaches its entry's ``success_threshold`` (1.0 by
default), as ``platoon eval`` judges success: a shaping reward processor changes what trains, not what passes.  As in
``platoon eval``, a rollout whose trace records no outcome passes on its own success flag at 1, whatever the
threshold, and one without a flag on its trained reward.  Members without a valid result (errors, timeouts, cancelled
stragglers, dead workers) are censored, neither passes nor failures, as the baseline leaves them out.

**Verdicts** (:data:`VERDICTS`).  ``trained``: the group carries signal, an informative group.  ``uniform``: every
retained advantage is zero; an all-pass group when every valid member passed, an all-fail group when every one failed,
and a mixed one otherwise (no datum kept the signal, or the trained reward and the scored success disagree).  The
rest say why a group trains nothing: ``reward_key_mismatch``, ``no_results``, ``below_min``, ``no_completed_root``,
``no_datums``, ``failed`` for a workflow that raised or returned nothing, and ``unknown`` for a workflow that gives no
verdict.  A custom workflow's plain list of datums says nothing of its members either: its groups count, but they move
no streak and give no evidence, so its tasks are not counted as seen.

**Decay.**  Evidence decays by policy version: a group sampled at version ``v`` weighs ``2 ** (-(V - v) / H)`` at
version ``V``, with ``H`` the half-life (:data:`HALF_LIFE_VERSIONS`).  A task's posterior pass rate is
``Beta(a0 + s, b0 + f)`` of its decayed passes and failures, with a prior centered on its entry's decayed pass rate,
``m = (1 + P) / (2 + N)``, of strength :data:`PRIOR_STRENGTH` (``a0 = 2m``, ``b0 = 2(1 - m)``); :func:`pass_at` gives
its expected pass@k.  The ``tasks/`` metrics of each step average these over the tasks each label has seen.

**Persistence.**  At each full checkpoint the trainer writes the store as a snapshot, ``tasks/state-s<start>-
<checkpoint>.json`` in the run directory (:func:`write_snapshot`), and the checkpoint's row names it, with its SHA-256,
in its ``loop_state`` (``tasks``); a start that resumes from the row loads it (:func:`read_snapshot`), or, when it
cannot, the newest older one that reads back (:func:`readable_snapshot`).  A snapshot holds the draws no committed
step had consumed (outstanding).  The checkpoint saves where those draws stand in a file of the dispatcher's own
(:mod:`platoon.train.tinker.dispatcher`, which its row names as ``dispatch``), and a resume takes them up from it: the
steps after it train the groups the checkpoint kept as they are, and it draws the others again
(:meth:`TaskStatsStore.reissue` moves each to its new draw).  A resume from a checkpoint without that file (one saved
before platoon 0.3.0, or whose write failed) counts them as lost, as one does the draws of an older snapshot loaded in
place of the checkpoint's own that the checkpoint no longer holds.  A group that had settled and that a resume rolls
out again (its checkpoint could not keep it) counts as two groups of one draw: both rollouts happened.
``tasks/events.jsonl`` logs each draw, settle and fate, one JSON line each (:class:`EventLog`), for audit and
``platoon tasks``; a resume never reads it.

The record types are experimental: a later release may add fields to them.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# The workflow's verdicts on a group (GroupRolloutWorkflow): what it trains, or why it trains nothing.
VERDICTS = (
    "trained",
    "uniform",
    "reward_key_mismatch",
    "no_results",
    "below_min",
    "no_completed_root",
    "no_datums",
    "failed",
    "unknown",
)
# What becomes of a draw: a committed step consumed its group, dropped it as stale, or found that its workflow failed;
# a capped label's full backlog skipped its row; a resume found it outstanding and could not draw it again (its task
# left the pool, or the checkpoint saved no state of the dispatcher's draws); or a group admission rule rejected its
# group when it settled (platoon.train.admissions), the last so that the other fates keep their places.
FATES = ("consumed", "stale", "failed", "skipped", "lost", "rejected")
# Policy versions after which a group's evidence weighs half.
HALF_LIFE_VERSIONS = 32.0
# The total weight of a task's prior, centered on its entry's pass rate.
PRIOR_STRENGTH = 2.0
# Groups in a row without a valid member that count a task as without evidence (a broken grader or setup).
NO_EVIDENCE_LIMIT = 3
# The newest groups each task keeps a summary of.
HISTORY = 8
# The bands a step counts the seen tasks in, by posterior pass rate (band).
BANDS = ("zero", "low", "mid", "high", "solved")
# The snapshot's format, which its row names.
SNAPSHOT_FORMAT = 1
# The directory of a run's snapshots and event log, and the event log's name.
TASKS_DIR = "tasks"
EVENTS_NAME = "events.jsonl"
# The key of a checkpoint row's loop_state that names its snapshot, and the one that names the state of the
# dispatcher's draws (platoon.train.tinker.dispatcher): each record names the start that wrote it.
LOOP_STATE_KEY = "tasks"
DISPATCH_KEY = "dispatch"
# The main policy, by whose versions evidence ages until a run trains several policies.
MAIN = "main"


# --- what happened to a group -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberResult:
    """One member of a group: what its rollout returned."""

    # The root's reward as trained, after the reward processor; None: no result (a setup failure, a dead worker, a
    # cancelled straggler).
    reward: float | None
    # The reward its environment scored, before any reward processor, as platoon eval judges success; None when its
    # outcome was not scored.
    scored: float | None
    # Whether its root completed with a scored outcome, so that it entered the baseline.
    valid: bool
    # The status of its root's outcome: scored, invalid, error, timeout, cancelled, or none without one.
    outcome: str
    # How it ended: completed, straggler_killed, worker_crashed, and the rest of the worker dispositions.
    disposition: str
    # The reward key its root trained, if any.
    reward_key: str | None
    # Its root's steps, when known.
    steps: int | None
    # Its root's output tokens, when known.
    output_tokens: int | None
    # Its rollout's seconds, when known.
    seconds: float | None
    # The oldest and newest policy version its datums sampled, by policy.
    versions: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    # Whether ``scored`` is the rollout's own success flag, of a rollout whose trace records no outcome: platoon eval
    # counts such a flag as a pass at 1, whatever the threshold, and so does ``passes``.
    flag: bool = False

    def passes(self, threshold: float) -> bool:
        """Whether it passed: valid, with a scored reward that reaches ``threshold`` (1 for a rollout's own success
        flag, :attr:`flag`), as platoon eval judges success."""

        return self.valid and self.scored is not None and self.scored >= (1.0 if self.flag else threshold)

    def to_json(self) -> dict[str, Any]:
        """The member as JSON data, as its group's record keeps it (:meth:`from_json` reads it back)."""

        return {
            "reward": self.reward,
            "scored": self.scored,
            "valid": self.valid,
            "outcome": self.outcome,
            "disposition": self.disposition,
            "reward_key": self.reward_key,
            "steps": self.steps,
            "output_tokens": self.output_tokens,
            "seconds": self.seconds,
            "versions": {policy: list(span) for policy, span in self.versions.items()},
            "flag": self.flag,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> MemberResult:
        """The member :meth:`to_json` wrote."""

        return cls(
            reward=_number(data["reward"]),
            scored=_number(data["scored"]),
            valid=bool(data["valid"]),
            outcome=str(data["outcome"]),
            disposition=str(data["disposition"]),
            reward_key=None if data["reward_key"] is None else str(data["reward_key"]),
            steps=None if data["steps"] is None else int(data["steps"]),
            output_tokens=None if data["output_tokens"] is None else int(data["output_tokens"]),
            seconds=_number(data["seconds"]),
            versions={str(policy): (int(span[0]), int(span[1])) for policy, span in data["versions"].items()},
            flag=bool(data.get("flag", False)),
        )


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class GroupRecord:
    """One settled group: which draw it was, and what its members did."""

    # Which start of the run drew it: (start, seq) is unique in a run.
    start: int
    # Its draw's number within that start.
    seq: int
    # The task it ran.
    task_id: str
    # The environment entry it ran (its label).
    entry: str | None
    # Its labels: {label column: label}.
    labels: Mapping[str, str]
    # The harness it ran, when known.
    harness: str | None
    # How many rollouts the group ran.
    group_size: int
    # The task sampler that drew it: the run's sampler as the config names it (train.task_sampler, "auto" by
    # default), or an entry's, "<sampler>@<entry>" ("uniform@quick", "pass_rate@hard").
    sampler: str
    # Why it was drawn: "order" for a static order's draw, "explore" or "exploit" for pass_rate's, "fallback" for one
    # the dispatcher drew after the sampler's decision failed, "reissue" for a draw a resume drew again, or a custom
    # sampler's own.
    reason: str
    # The sampler's score of the draw, None without one (pass_rate's: the chance it predicted that the group trains).
    score: float | None
    # The draw's place in the run's order; a reissued draw keeps its old one.
    position: int | None
    # The policy versions when it was drawn, by policy.
    drawn_at: Mapping[str, int]
    # What each of its rollouts did.
    members: tuple[MemberResult, ...]
    # The workflow's verdict (VERDICTS).
    verdict: str

    @property
    def valid(self) -> int:
        """Its valid members."""

        return sum(member.valid for member in self.members)

    def passes(self, threshold: float) -> int:
        """Its valid members whose scored reward reaches ``threshold``."""

        return sum(member.passes(threshold) for member in self.members)

    def oldest_version(self, policy: str = MAIN) -> int | None:
        """The oldest policy version its datums sampled (the stale filter's reference), else the one it was drawn at."""

        versions = [member.versions[policy][0] for member in self.members if policy in member.versions]
        return min(versions) if versions else self.drawn_at.get(policy)

    def kind(self, threshold: float) -> str | None:
        """What the group was, by the step metrics' partition of the groups that trained or were uniform:
        ``informative``, ``all_pass``, ``all_fail`` or ``uniform_mixed``; None for another verdict."""

        if self.verdict == "trained":
            return "informative"
        if self.verdict != "uniform":
            return None
        valid = self.valid
        passes = self.passes(threshold)
        if valid and passes == valid:
            return "all_pass"
        if valid and passes == 0:
            return "all_fail"
        return "uniform_mixed"

    def to_json(self) -> dict[str, Any]:
        """The record as JSON data, as a checkpoint's kept groups keep it (:meth:`from_json` reads it back)."""

        return {
            "start": self.start,
            "seq": self.seq,
            "task_id": self.task_id,
            "entry": self.entry,
            "labels": dict(self.labels),
            "harness": self.harness,
            "group_size": self.group_size,
            "sampler": self.sampler,
            "reason": self.reason,
            "score": self.score,
            "position": self.position,
            "drawn_at": dict(self.drawn_at),
            "members": [member.to_json() for member in self.members],
            "verdict": self.verdict,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> GroupRecord:
        """The record :meth:`to_json` wrote."""

        return cls(
            start=int(data["start"]),
            seq=int(data["seq"]),
            task_id=str(data["task_id"]),
            entry=None if data["entry"] is None else str(data["entry"]),
            labels={str(column): str(label) for column, label in data["labels"].items()},
            harness=None if data["harness"] is None else str(data["harness"]),
            group_size=int(data["group_size"]),
            sampler=str(data["sampler"]),
            reason=str(data["reason"]),
            score=_number(data["score"]),
            position=None if data["position"] is None else int(data["position"]),
            drawn_at={str(policy): int(version) for policy, version in data["drawn_at"].items()},
            members=tuple(MemberResult.from_json(member) for member in data["members"]),
            verdict=str(data["verdict"]),
        )


@dataclass(frozen=True)
class GroupSummary:
    """One of a task's newest groups, as its history keeps it."""

    # Which start of the run drew it.
    start: int
    # Its draw's number within that start.
    seq: int
    # The policy version of its oldest datum, None without one.
    version: int | None
    # Its valid members.
    valid: int
    # Its members that passed.
    passes: int
    # The mean of its valid members' trained rewards, None without one.
    reward_mean: float | None
    # The workflow's verdict (VERDICTS).
    verdict: str
    # The mean of its members' steps, when known.
    steps: float | None
    # The mean of its members' seconds, when known.
    seconds: float | None


# --- decay ------------------------------------------------------------------------------------------------------------


def decay(versions: float, half_life: float | None) -> float:
    """The weight of evidence ``versions`` policy versions old: halved every ``half_life`` versions (None: no decay)."""

    if half_life is None or versions <= 0:
        return 1.0
    return 2.0 ** (-versions / half_life)


class DecayedCounts:
    """Passes and failures that decay by policy version: ``(s, f)`` as they weigh at ``version``.  Newer evidence decays
    the counts to its own version and then adds; older evidence, such as a long group that settles late, adds at a
    discount."""

    def __init__(self, half_life: float | None, passes: float = 0.0, fails: float = 0.0, version: int | None = None):
        self.half_life = half_life
        self.passes = passes
        self.fails = fails
        self.version = version

    def add(self, passes: float, fails: float, version: int) -> None:
        """Count ``passes`` and ``fails`` of policy version ``version``."""

        if self.version is None or version >= self.version:
            weight = decay(version - self.version, self.half_life) if self.version is not None else 1.0
            self.passes, self.fails = self.passes * weight + passes, self.fails * weight + fails
            self.version = version
        else:
            weight = decay(self.version - version, self.half_life)
            self.passes += passes * weight
            self.fails += fails * weight

    def at(self, now: int) -> tuple[float, float]:
        """``(passes, fails)`` as they weigh at policy version ``now``."""

        if self.version is None:
            return 0.0, 0.0
        weight = decay(now - self.version, self.half_life)
        return self.passes * weight, self.fails * weight

    def to_json(self) -> list[Any]:
        """``[passes, fails, version]``, as a snapshot keeps them."""

        return [self.passes, self.fails, self.version]

    @classmethod
    def from_json(cls, data: Sequence[Any], half_life: float | None) -> DecayedCounts:
        """The counts :meth:`to_json` wrote, decaying with ``half_life``."""

        passes, fails, version = data
        return cls(half_life, float(passes), float(fails), None if version is None else int(version))


class DecayedMean:
    """A mean whose terms decay by policy version, as :class:`DecayedCounts` does."""

    def __init__(self, half_life: float | None, total: float = 0.0, weight: float = 0.0, version: int | None = None):
        self._counts = DecayedCounts(half_life, total, weight, version)

    def add(self, value: float, version: int) -> None:
        """Count a term ``value`` of policy version ``version``."""

        self._counts.add(value, 1.0, version)

    def at(self, now: int) -> float | None:
        """The mean as its terms weigh at policy version ``now``; None before any."""

        total, weight = self._counts.at(now)
        return total / weight if weight > 0 else None

    def to_json(self) -> list[Any]:
        """``[total, weight, version]``, as a snapshot keeps them."""

        return self._counts.to_json()

    def copy(self) -> DecayedMean:
        """A mean of its own with the same terms."""

        counts = self._counts
        return DecayedMean(counts.half_life, counts.passes, counts.fails, counts.version)

    @classmethod
    def from_json(cls, data: Sequence[Any], half_life: float | None) -> DecayedMean:
        """The mean :meth:`to_json` wrote, decaying with ``half_life``."""

        counts = DecayedCounts.from_json(data, half_life)
        return cls(half_life, counts.passes, counts.fails, counts.version)


def band(rate: float) -> str:
    """The band of a posterior pass rate: ``zero`` at most 0.05, ``low`` at most 0.2, ``mid`` below 0.8, ``high``
    below 0.95, ``solved`` from 0.95."""

    if rate <= 0.05:
        return "zero"
    if rate <= 0.2:
        return "low"
    if rate < 0.8:
        return "mid"
    return "high" if rate < 0.95 else "solved"


def pass_at(alpha: float, beta: float, k: int) -> float:
    """The expected pass@k of a task whose pass rate is ``Beta(alpha, beta)``: ``1 - prod_{i<k} (beta + i) / (alpha +
    beta + i)``, for fractional, decayed counts too; pass@1 is the posterior mean."""

    product = 1.0
    for index in range(max(k, 0)):
        product *= (beta + index) / (alpha + beta + index)
    return 1.0 - product


def informative_at(alpha: float, beta: float, k: int) -> float:
    """The expected chance that a group of ``k`` mixes passes and failures: ``E[pass@k] - E[p^k]``, never below 0."""

    product = 1.0
    for index in range(max(k, 0)):
        product *= (alpha + index) / (alpha + beta + index)
    return max(0.0, pass_at(alpha, beta, k) - product)  # exactly 0 for k = 1, but for the rounding


# --- the records ------------------------------------------------------------------------------------------------------


_METRICS = ("steps", "output_tokens", "seconds", "reward")


@dataclass(frozen=True)
class EntryInfo:
    """What platoon knows of an environment entry: the store reads its group size and the reward that counts as a
    pass, and task samplers the rest (``SamplerSetup.entries``)."""

    # Its training groups' size.
    group_size: int
    # The reward at or above which a rollout passes.
    success_threshold: float = 1.0
    # Its label (the entry's name).
    name: str | None = None
    # Its weight as a number: weight: tasks reads as its training tasks' number.
    weight: float | None = None
    # Its seed.
    seed: int = 0
    # Its index among the run's entries with training tasks.
    index: int | None = None
    # Its training tasks' ids, in listing order.
    task_ids: tuple[str, ...] = ()


class _Totals:
    """Counts over groups (a task's, an entry's, a label's)."""

    def __init__(self, half_life: float | None) -> None:
        self.groups = 0
        # The groups whose workflow said what each member did (members): not a custom workflow's plain list of datums.
        self.reported = 0
        self.attempts = 0
        self.passes = 0
        self.reward_sum = 0.0
        self.kinds: Counter[str] = Counter()
        self.evidence = DecayedCounts(half_life)

    def add(self, record: GroupRecord, threshold: float, version: int | None) -> None:
        valid = record.valid
        passes = record.passes(threshold)
        self.groups += 1
        self.reported += bool(record.members)
        self.attempts += valid
        self.passes += passes
        self.reward_sum += sum(member.reward for member in record.members if member.valid and member.reward is not None)
        kind = record.kind(threshold)
        if kind is not None:
            self.kinds[kind] += 1
        if valid and version is not None:
            self.evidence.add(passes, valid - passes, version)

    def to_json(self) -> dict[str, Any]:
        return {
            "groups": self.groups,
            "reported": self.reported,
            "attempts": self.attempts,
            "passes": self.passes,
            "reward_sum": self.reward_sum,
            "kinds": {kind: self.kinds[kind] for kind in sorted(self.kinds)},
            "evidence": self.evidence.to_json(),
        }

    def load(self, data: Mapping[str, Any], half_life: float | None) -> None:
        self.groups = int(data["groups"])
        self.reported = int(data["reported"])
        self.attempts = int(data["attempts"])
        self.passes = int(data["passes"])
        self.reward_sum = float(data["reward_sum"])
        self.kinds = Counter({str(kind): int(count) for kind, count in data["kinds"].items()})
        self.evidence = DecayedCounts.from_json(data["evidence"], half_life)


class _Task(_Totals):
    """One training task's record."""

    def __init__(self, entry: str | None, label: str | None, half_life: float | None) -> None:
        super().__init__(half_life)
        self.entry = entry
        self.label = label
        self.uniform_streak = 0
        self.no_evidence_streak = 0
        self.metrics = {name: DecayedMean(half_life) for name in _METRICS}
        self.history: deque[GroupSummary] = deque(maxlen=HISTORY)
        self.fates: Counter[str] = Counter()
        self.drawn = 0
        self.outstanding = 0

    def settle(self, record: GroupRecord, threshold: float) -> None:
        version = record.oldest_version()
        self.add(record, threshold, version)
        if record.members or record.verdict != "unknown":  # a workflow that says nothing of a group moves no streak
            self.uniform_streak = self.uniform_streak + 1 if record.verdict == "uniform" else 0
            self.no_evidence_streak = 0 if record.valid else self.no_evidence_streak + 1
        if version is not None:
            for member in record.members:
                for name in ("steps", "output_tokens", "seconds"):
                    value = getattr(member, name)
                    if value is not None:
                        self.metrics[name].add(float(value), version)
                if member.valid and member.reward is not None:
                    self.metrics["reward"].add(float(member.reward), version)
        rewards = [member.reward for member in record.members if member.valid and member.reward is not None]
        steps = [member.steps for member in record.members if member.steps is not None]
        seconds = [member.seconds for member in record.members if member.seconds is not None]
        self.history.append(
            GroupSummary(
                start=record.start,
                seq=record.seq,
                version=version,
                valid=record.valid,
                passes=record.passes(threshold),
                reward_mean=sum(rewards) / len(rewards) if rewards else None,
                verdict=record.verdict,
                steps=sum(steps) / len(steps) if steps else None,
                seconds=max(seconds) if seconds else None,
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "entry": self.entry,
            "label": self.label,
            **super().to_json(),
            "uniform_streak": self.uniform_streak,
            "no_evidence_streak": self.no_evidence_streak,
            "metrics": {name: mean.to_json() for name, mean in self.metrics.items()},
            "history": [[getattr(summary, name) for name in _SUMMARY_FIELDS] for summary in self.history],
            "fates": {fate: self.fates[fate] for fate in sorted(self.fates)},
            "drawn": self.drawn,
            "outstanding": self.outstanding,
        }

    def load(self, data: Mapping[str, Any], half_life: float | None) -> None:
        super().load(data, half_life)
        self.uniform_streak = int(data["uniform_streak"])
        self.no_evidence_streak = int(data["no_evidence_streak"])
        self.metrics = {name: DecayedMean.from_json(data["metrics"][name], half_life) for name in _METRICS}
        self.history = deque(
            (GroupSummary(**dict(zip(_SUMMARY_FIELDS, summary))) for summary in data["history"]), maxlen=HISTORY
        )
        self.fates = Counter({str(fate): int(count) for fate, count in data["fates"].items()})
        self.drawn = int(data["drawn"])
        self.outstanding = int(data["outstanding"])


_SUMMARY_FIELDS = tuple(GroupSummary.__dataclass_fields__)


_new_tuple = tuple.__new__


class TaskCounts(NamedTuple):
    """A training task's counts, read without building its :class:`TaskStats` (:meth:`TaskStatsStore.counts`): a
    tuple, whose fields a decision over every task of a large pool reads and lets go, rather than keeping one per
    task."""

    # Its settled groups.
    groups: int
    # Its draws not yet consumed by a committed step.
    outstanding: int
    # All its draws.
    drawn: int
    # Groups in a row, up to the newest, whose verdict was uniform.
    uniform_streak: int
    # Groups in a row, up to the newest, that had no valid member.
    no_evidence_streak: int
    # The policy version its decayed evidence stands at (that of its newest evidence); None before any.
    evidence_version: int | None


@dataclass(frozen=True)
class GroupTotals:
    """What an entry's or a label's groups add up to."""

    # The settled groups.
    groups: int
    # The groups whose workflow said what each member did: a custom workflow's plain list of datums says nothing.
    reported_groups: int
    # Their valid members.
    attempts: int
    # Their members that passed.
    passes: int
    # The sum of their valid members' trained rewards.
    reward_sum: float
    # The groups that trained (their verdict was trained).
    informative_groups: int
    # The groups whose rewards were all alike (their verdict was uniform), every valid member of which passed.
    all_pass_groups: int
    # The uniform groups no valid member of which passed.
    all_fail_groups: int
    # The other uniform groups: their members mixed passes and failures, yet their rewards were all alike.
    uniform_mixed_groups: int
    # Decayed (passes, fails) and their version.
    decayed: DecayedCounts

    def evidence(self, *, now: int) -> tuple[float, float]:
        """Decayed ``(passes, fails)`` at policy version ``now``."""

        return self.decayed.at(now)


@dataclass(frozen=True)
class TaskStats(GroupTotals):
    """One training task's statistics, read-only: what its groups add up to (the fields of :class:`GroupTotals`), and
    the rest below."""

    # The task.
    task_id: str
    # Its environment entry's label.
    entry: str | None
    # Its label, its value of the run's label column.
    label: str | None
    # Groups in a row, up to the newest, whose verdict was uniform.
    uniform_streak: int
    # Groups in a row, up to the newest, that had no valid member.
    no_evidence_streak: int
    # Its draws not yet consumed by a committed step.
    outstanding: int
    # All its draws.
    drawn: int
    # What became of its draws, by fate (FATES).
    fates: Mapping[str, int]
    # Its newest groups, oldest first.
    history: tuple[GroupSummary, ...]
    # The decayed means of its members' steps, output_tokens, seconds and trained reward, by name (metric).
    means: Mapping[str, DecayedMean]

    def metric(self, name: str, *, now: int) -> float | None:
        """The decayed mean of ``steps``, ``output_tokens``, ``seconds`` or the trained ``reward`` of its members at
        policy version ``now``; None before any."""

        return self.means[name].at(now)


def _totals(totals: _Totals) -> dict[str, Any]:
    evidence = totals.evidence
    return {
        "groups": totals.groups,
        "reported_groups": totals.reported,
        "attempts": totals.attempts,
        "passes": totals.passes,
        "reward_sum": totals.reward_sum,
        "informative_groups": totals.kinds["informative"],
        "all_pass_groups": totals.kinds["all_pass"],
        "all_fail_groups": totals.kinds["all_fail"],
        "uniform_mixed_groups": totals.kinds["uniform_mixed"],
        "decayed": DecayedCounts(evidence.half_life, evidence.passes, evidence.fails, evidence.version),
    }


# --- the store --------------------------------------------------------------------------------------------------------


class TaskStatsStore:
    """The per-task statistics of a run's training tasks: what each task's groups add up to, its streaks and newest
    groups, its evidence decayed by policy version, and what became of each of its draws (RFC 0019, section 3.3).

    ``tasks`` are the run's training tasks, in pool order: ``(task id, entry, label)`` each, the entry its environment
    entry's label and the label its value of the run's label column.  ``entries`` say each entry's group size and
    success threshold; ``group_size`` is the run's, for a task of no entry.  The trainer :meth:`draw`\\s each group's
    task, :meth:`settle`\\s each group's record, and gives each draw its fate (:meth:`fate`); :meth:`snapshot` and
    :meth:`load` save and restore it, which the trainer does at each full checkpoint."""

    def __init__(
        self,
        tasks: Iterable[tuple[str, str | None, str | None]],
        entries: Mapping[str, EntryInfo],
        *,
        column: str,
        group_size: int = 1,
        half_life: float | None = HALF_LIFE_VERSIONS,
        prior_strength: float = PRIOR_STRENGTH,
        no_evidence_limit: int = NO_EVIDENCE_LIMIT,
    ) -> None:
        self.column = column
        self.default_group_size = group_size
        self.half_life = half_life
        self.prior_strength = prior_strength
        self.no_evidence_limit = no_evidence_limit
        self.entries = dict(entries)
        self._tasks: dict[str, _Task] = {}
        for task_id, entry, label in tasks:
            self._tasks.setdefault(task_id, _Task(entry, label, half_life))
        self._entry_totals = {entry: _Totals(half_life) for entry in self.entries}
        self._label_totals: dict[str, _Totals] = {}
        for task in self._tasks.values():
            if task.label is not None:
                self._label_totals.setdefault(task.label, _Totals(half_life))
        self._run_totals = _Totals(half_life)
        # Each entry's prior pass rate at a policy version, (version, rate), until a group settles (posterior).
        self._rates: dict[str | None, tuple[int, float]] = {}
        # Draws no committed step has consumed: (start, seq) -> (task id, position).
        self._outstanding: dict[tuple[int, int], tuple[str, int | None]] = {}

    # --- what happens ------------------------------------------------------------------------------------------------

    def draw(self, start: int, seq: int, task_id: str, position: int | None = None) -> None:
        """A group of ``task_id`` was drawn, as draw ``seq`` of start ``start``."""

        self._outstanding[(start, seq)] = (task_id, position)
        task = self._tasks.get(task_id)
        if task is not None:
            task.drawn += 1
            task.outstanding += 1

    def settle(self, record: GroupRecord) -> None:
        """Count a settled group: its task's record, its entry's totals and its label's."""

        self._rates.clear()
        task = self._tasks.get(record.task_id)
        threshold = self.threshold(task.entry if task is not None else record.entry)
        version = record.oldest_version()
        if task is None:  # a task outside the pool: the run's totals alone
            self._run_totals.add(record, threshold, version)
            return
        task.settle(record, threshold)
        self._run_totals.add(record, threshold, version)
        if task.entry in self._entry_totals:
            self._entry_totals[task.entry].add(record, threshold, version)
        if task.label is not None:
            self._label_totals.setdefault(task.label, _Totals(self.half_life)).add(record, threshold, version)

    def fate(self, start: int, seq: int, fate: str, task_id: str | None = None) -> None:
        """What became of draw ``seq`` of start ``start`` (FATES).  A skipped row was never outstanding: name its
        ``task_id``."""

        if fate not in FATES:
            raise ValueError(f"a draw's fate is one of {FATES}, got {fate!r}")
        drawn = self._outstanding.pop((start, seq), None)
        name = drawn[0] if drawn is not None else task_id
        task = self._tasks.get(name) if name is not None else None
        if task is None:
            return
        if drawn is not None:
            task.outstanding = max(0, task.outstanding - 1)
        task.fates[fate] += 1

    def reissue(self, old: Sequence[int], new: tuple[int, int]) -> bool:
        """A resume draws outstanding draw ``old`` (``(start, seq)``) again as draw ``new``, of the same task at the
        same position: the same draw of its task, under a new number, so it is neither counted as drawn twice nor given
        a fate.  False, changing nothing, when ``old`` is not outstanding."""

        drawn = self._outstanding.pop((int(old[0]), int(old[1])), None)
        if drawn is None:
            return False
        self._outstanding[new] = drawn
        return True

    def lose_outstanding(self) -> list[tuple[int, int]]:
        """Count every outstanding draw as lost (a resume that cannot draw them again: its checkpoint saved no
        dispatcher's state); returns them, ``(start, seq)`` each."""

        lost = list(self._outstanding)
        for start, seq in lost:
            self.fate(start, seq, "lost")
        return lost

    def threshold(self, entry: str | None) -> float:
        """The reward at or above which a rollout of entry ``entry`` passes (1.0 for a task of no entry)."""

        info = self.entries.get(entry) if entry is not None else None
        return info.success_threshold if info is not None else 1.0

    def group_size(self, entry: str | None) -> int:
        """Entry ``entry``'s training group size (the run's for a task of no entry)."""

        info = self.entries.get(entry) if entry is not None else None
        return info.group_size if info is not None else self.default_group_size

    # --- reading ---------------------------------------------------------------------------------------------------

    @property
    def outstanding(self) -> list[tuple[int, int, str, int | None]]:
        """The draws no committed step has consumed: ``(start, seq, task id, position)`` each, in draw order."""

        return [(start, seq, task_id, position) for (start, seq), (task_id, position) in self._outstanding.items()]

    def task_ids(self) -> list[str]:
        """Every training task, in pool order."""

        return list(self._tasks)

    def task(self, task_id: str) -> TaskStats:
        """The task's whole record, built afresh (:meth:`counts` is the cheap read).  ``KeyError`` for a task the
        store does not hold."""

        task = self._tasks[task_id]
        return TaskStats(
            **_totals(task),
            task_id=task_id,
            entry=task.entry,
            label=task.label,
            uniform_streak=task.uniform_streak,
            no_evidence_streak=task.no_evidence_streak,
            outstanding=task.outstanding,
            drawn=task.drawn,
            fates=dict(task.fates),
            history=tuple(task.history),
            means={name: mean.copy() for name, mean in task.metrics.items()},
        )

    def counts(self, task_id: str) -> TaskCounts:
        """The task's counts and streaks, read from its record without building anything: the cheap read of a task,
        which a task sampler makes for every task of its pool (:meth:`task` builds the whole record)."""

        task = self._tasks[task_id]
        # Built as TaskCounts._make builds it, without the call of its __new__: a decision reads every task's.
        return _new_tuple(
            TaskCounts,
            (
                task.groups,
                task.outstanding,
                task.drawn,
                task.uniform_streak,
                task.no_evidence_streak,
                task.evidence.version,
            ),
        )

    def entry(self, name: str) -> GroupTotals:
        """What the groups of entry ``name`` add up to."""

        return GroupTotals(**_totals(self._entry_totals[name]))

    def label(self, label: str) -> GroupTotals:
        """What the groups of the tasks labelled ``label`` add up to."""

        return GroupTotals(**_totals(self._label_totals[label]))

    def labels(self) -> list[str]:
        """The labels of the training tasks."""

        return list(self._label_totals)

    def posterior(self, task_id: str, *, now: int) -> tuple[float, float]:
        """``(alpha, beta)`` of the task's posterior pass rate at policy version ``now``: its decayed passes and
        failures over a prior centered on its entry's decayed pass rate (the run's, for a task of no entry)."""

        task = self._tasks[task_id]
        cached = self._rates.get(task.entry)
        if cached is not None and cached[0] == now:
            rate = cached[1]
        else:  # the entry's rate, read once per version and settle: a decision reads every task's posterior
            prior = self._entry_totals.get(task.entry) if task.entry is not None else None
            passes, fails = (prior if prior is not None else self._run_totals).evidence.at(now)
            rate = (1.0 + passes) / (2.0 + passes + fails)
            self._rates[task.entry] = (now, rate)
        own_passes, own_fails = task.evidence.at(now)
        return self.prior_strength * rate + own_passes, self.prior_strength * (1.0 - rate) + own_fails

    # --- metrics ---------------------------------------------------------------------------------------------------

    def step_metrics(self, *, now: int, prefixes: Mapping[str, str]) -> dict[str, float]:
        """The ``tasks/`` metrics of a step at policy version ``now``: over every task (``tasks/...``) and over each
        label's (``tasks/<prefix>/...``, ``prefixes`` mapping each logged label to its metric prefix): the mean expected
        pass@1 and pass@G (G each task's entry's group size) over the seen tasks, those with a group whose members were
        recorded (a custom workflow's plain list of datums records none), the share of tasks seen (``coverage``), the
        seen tasks by posterior pass rate (``bands/<band>``, BANDS), and the tasks without evidence
        (``no_evidence``)."""

        groups: dict[str, list[str]] = {"": list(self._tasks)}
        for task_id, task in self._tasks.items():
            if task.label in prefixes:
                groups.setdefault(prefixes[task.label], []).append(task_id)
        for prefix in prefixes.values():
            groups.setdefault(prefix, [])
        metrics: dict[str, float] = {}
        for prefix, task_ids in groups.items():
            name = f"tasks/{prefix}/" if prefix else "tasks/"
            seen = [task_id for task_id in task_ids if self._tasks[task_id].reported]
            bands: Counter[str] = Counter()
            pass_1, pass_g = [], []
            for task_id in seen:
                alpha, beta = self.posterior(task_id, now=now)
                pass_1.append(alpha / (alpha + beta))
                pass_g.append(pass_at(alpha, beta, self.group_size(self._tasks[task_id].entry)))
                bands[band(pass_1[-1])] += 1
            if pass_1:
                metrics[f"{name}pass_at_1"] = sum(pass_1) / len(pass_1)
                metrics[f"{name}pass_at_G"] = sum(pass_g) / len(pass_g)
            metrics[f"{name}coverage"] = len(seen) / len(task_ids) if task_ids else 0.0
            for name_of_band in BANDS:
                metrics[f"{name}bands/{name_of_band}"] = float(bands[name_of_band])
            metrics[f"{name}no_evidence"] = float(
                sum(self._tasks[task_id].no_evidence_streak >= self.no_evidence_limit for task_id in task_ids)
            )
        return metrics

    # --- persistence ---------------------------------------------------------------------------------------------

    def snapshot(self, *, start: int, next_seq: int) -> dict[str, Any]:
        """The store as JSON data, with the start that writes it and its next draw's number."""

        return {
            "format": SNAPSHOT_FORMAT,
            "start": start,
            "next_seq": next_seq,
            "column": self.column,
            "group_size": self.default_group_size,
            "half_life_versions": self.half_life,
            "prior_strength": self.prior_strength,
            "no_evidence_limit": self.no_evidence_limit,
            "entries": {
                name: {
                    "group_size": info.group_size,
                    "success_threshold": info.success_threshold,
                    "totals": self._entry_totals[name].to_json(),
                }
                for name, info in self.entries.items()
            },
            "labels": {label: totals.to_json() for label, totals in self._label_totals.items()},
            "run": self._run_totals.to_json(),
            "tasks": {task_id: task.to_json() for task_id, task in self._tasks.items()},
            "outstanding": [list(draw) for draw in self.outstanding],
        }

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any]) -> TaskStatsStore:
        """The store a :meth:`snapshot` holds, its pool the tasks it holds (``platoon tasks``); a ``ValueError`` for
        data it cannot read."""

        try:
            half_life = data["half_life_versions"]
            store = cls(
                ((str(task_id), task["entry"], task["label"]) for task_id, task in data["tasks"].items()),
                {
                    str(name): EntryInfo(int(entry["group_size"]), float(entry["success_threshold"]))
                    for name, entry in data["entries"].items()
                },
                column=str(data["column"]),
                group_size=int(data["group_size"]),
                half_life=None if half_life is None else float(half_life),
                prior_strength=float(data["prior_strength"]),
                no_evidence_limit=int(data["no_evidence_limit"]),
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot read the task statistics snapshot: {type(exc).__name__}: {exc}") from exc
        store.load(data)
        return store

    def load(self, data: Mapping[str, Any]) -> None:
        """Restore what a :meth:`snapshot` holds: the records of the tasks still in the pool, every total and the
        outstanding draws.  A task the snapshot does not hold starts empty; one that left the pool is dropped.  A
        ``ValueError`` for data it cannot read."""

        self._rates.clear()
        try:
            if data.get("format") != SNAPSHOT_FORMAT:
                raise ValueError(f"format {data.get('format')!r}, not {SNAPSHOT_FORMAT}")
            for name, entry in data["entries"].items():
                if name in self._entry_totals:
                    self._entry_totals[name].load(entry["totals"], self.half_life)
            for label, totals in data["labels"].items():
                self._label_totals.setdefault(label, _Totals(self.half_life)).load(totals, self.half_life)
            self._run_totals.load(data["run"], self.half_life)
            for task_id, task in data["tasks"].items():
                if task_id in self._tasks:
                    self._tasks[task_id].load(task, self.half_life)
            self._outstanding = {
                (int(start), int(seq)): (str(task_id), None if position is None else int(position))
                for start, seq, task_id, position in data["outstanding"]
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot read the task statistics snapshot: {type(exc).__name__}: {exc}") from exc


def snapshot_bytes(data: Mapping[str, Any]) -> bytes:
    """A snapshot as written: compact JSON, its keys in the store's order, which a load and a snapshot keep."""

    return (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_tasks_file(run_dir: str | os.PathLike[str], name: str, content: bytes) -> dict[str, Any]:
    """Write ``content`` as ``tasks/<name>`` in ``run_dir``, durably (a file synced, then renamed into place, so that
    no half-written file is left behind), and return what a checkpoint's row records of it: ``{file, sha256}``."""

    directory = Path(run_dir, TASKS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    partial = directory / f".{name}.partial"
    try:
        with open(partial, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    return {"file": f"{TASKS_DIR}/{name}", "sha256": hashlib.sha256(content).hexdigest()}


def read_tasks_file(run_dir: str | os.PathLike[str], record: Mapping[str, Any], *, by: str = "checkpoint") -> bytes:
    """The content of the file a record of :func:`write_tasks_file` names (``file``), checked against its SHA-256; a
    ``ValueError`` that says why when it names none, or the file is missing, changed or unreadable (``by`` names what
    recorded it, for the message)."""

    name = record.get("file")
    if not isinstance(name, str) or not name:
        raise ValueError(f"the {by} names no file: {dict(record)!r}")
    path = Path(run_dir, name)
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path} cannot be read: {exc}") from exc
    if hashlib.sha256(content).hexdigest() != record.get("sha256"):
        raise ValueError(f"{path} does not have the SHA-256 its {by} recorded")
    return content


def write_snapshot(run_dir: str | os.PathLike[str], name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Write a snapshot as ``tasks/<name>`` in ``run_dir``, durably, and return what the checkpoint's row records of
    it: ``{file, sha256, format, start}``."""

    return {
        **write_tasks_file(run_dir, name, snapshot_bytes(data)),
        "format": SNAPSHOT_FORMAT,
        "start": int(data["start"]),
    }


def snapshot_name(start: int, checkpoint: str) -> str:
    """The snapshot of a checkpoint a start saves: ``state-s<start>-<checkpoint>.json``.  The start is in the name
    because a start that resumes from ``recovery-<n>`` saves a checkpoint of that name again, under its own training
    run."""

    return f"state-s{start}-{checkpoint}.json"


def read_snapshot(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> dict[str, Any]:
    """The snapshot a checkpoint row's ``tasks`` record names, checked against its SHA-256; a ``ValueError`` that says
    why when it is missing, changed or unreadable."""

    if not isinstance(record.get("file"), str) or not record.get("file"):
        raise ValueError(f"the checkpoint names no snapshot file: {dict(record)!r}")
    content = read_tasks_file(run_dir, record)
    path = Path(run_dir, record["file"])
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} holds no snapshot")
    return data


@dataclass(frozen=True)
class FoundSnapshot:
    """The newest snapshot of a run's per-task statistics that reads back (:func:`readable_snapshot`)."""

    # The checkpoint row that names it, and what it holds.
    row: Mapping[str, Any]
    data: dict[str, Any]
    # Why each newer row's snapshot, or lack of one, was passed over, newest first.
    passed_over: tuple[str, ...] = ()


def readable_snapshot(run_dir: str | os.PathLike[str], rows: Sequence[Mapping[str, Any]]) -> FoundSnapshot | None:
    """The newest of a run's checkpoint rows ``rows`` (in ``checkpoints.jsonl``'s order) whose snapshot of the
    per-task statistics reads back (:func:`read_snapshot`), with why each newer row was passed over: it names none
    (saved by a platoon that kept none, or when the write failed), or its file is missing, changed or unreadable.
    None when no row's does."""

    passed_over: list[str] = []
    for row in reversed(rows):
        record = row.get(LOOP_STATE_KEY)
        name = row.get("name")
        if not isinstance(record, Mapping):
            passed_over.append(f"checkpoint {name} names no snapshot of the per-task statistics")
            continue
        try:
            return FoundSnapshot(row, read_snapshot(run_dir, record), tuple(passed_over))
        except ValueError as exc:
            passed_over.append(f"checkpoint {name}: {exc}")
    return None


def checkpoint_rows(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The rows of a run directory's ``checkpoints.jsonl`` that parse, in order; none without the file."""

    path = Path(run_dir, "checkpoints.jsonl")
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def newest_snapshot(run_dir: str | os.PathLike[str]) -> FoundSnapshot | None:
    """The newest snapshot of the per-task statistics that a row of a run directory's ``checkpoints.jsonl`` names and
    that reads back (:func:`readable_snapshot`), the rows that name none passed over in silence; None when none
    does."""

    rows = [row for row in checkpoint_rows(run_dir) if isinstance(row.get(LOOP_STATE_KEY), Mapping)]
    return readable_snapshot(run_dir, rows)


# The columns of a task's row in ``platoon tasks`` (task_rows).
TASK_COLUMNS = (
    "task_id",
    "entry",
    "label",
    "groups",
    "reported_groups",
    "attempts",
    "passes",
    "informative_groups",
    "all_pass_groups",
    "all_fail_groups",
    "uniform_mixed_groups",
    "uniform_streak",
    "no_evidence_streak",
    "drawn",
    "outstanding",
    *(f"fate_{fate}" for fate in FATES),
    "group_size",
    "pass_rate",
    "pass_at_G",
    "informative_at_G",
    "newest_version",
)


def task_rows(store: TaskStatsStore, *, now: int, label: str | None = None) -> list[dict[str, Any]]:
    """One row per task of ``store`` (of ``label``'s, when given), in pool order, with its counts, what became of its
    draws, and its posterior at policy version ``now``: the expected pass rate, pass@G and chance of an informative
    group, G its entry's group size (TASK_COLUMNS)."""

    rows = []
    for task_id in store.task_ids():
        task = store.task(task_id)
        if label is not None and task.label != label:
            continue
        alpha, beta = store.posterior(task_id, now=now)
        size = store.group_size(task.entry)
        versions = [summary.version for summary in task.history if summary.version is not None]
        rows.append(
            {
                "task_id": task_id,
                "entry": task.entry,
                "label": task.label,
                "groups": task.groups,
                "reported_groups": task.reported_groups,
                "attempts": task.attempts,
                "passes": task.passes,
                "informative_groups": task.informative_groups,
                "all_pass_groups": task.all_pass_groups,
                "all_fail_groups": task.all_fail_groups,
                "uniform_mixed_groups": task.uniform_mixed_groups,
                "uniform_streak": task.uniform_streak,
                "no_evidence_streak": task.no_evidence_streak,
                "drawn": task.drawn,
                "outstanding": task.outstanding,
                **{f"fate_{fate}": task.fates.get(fate, 0) for fate in FATES},
                "group_size": size,
                "pass_rate": alpha / (alpha + beta),
                "pass_at_G": pass_at(alpha, beta, size),
                "informative_at_G": informative_at(alpha, beta, size),
                "newest_version": max(versions) if versions else None,
            }
        )
    return rows


def next_start(records: Iterable[Mapping[str, Any]]) -> int:
    """The number of a new start of a run whose ``checkpoints.jsonl`` holds ``records``: one more than the largest
    start a row's snapshot or dispatcher's state names (each start saves a checkpoint before it draws anything), else
    1."""

    starts = [
        int(record[key]["start"])
        for record in records
        for key in (LOOP_STATE_KEY, DISPATCH_KEY)
        if isinstance(record.get(key), Mapping) and isinstance(record[key].get("start"), int)
    ]
    return max(starts, default=0) + 1


# --- the event log ----------------------------------------------------------------------------------------------------


class EventLog:
    """``tasks/events.jsonl`` of a run directory: one JSON line per draw, settle, fate and start, appended as it
    happens, for audit and ``platoon tasks``.  Losing it costs history, never training state: a write that fails is
    logged once and the log stops."""

    def __init__(self, run_dir: str | os.PathLike[str] | None) -> None:
        self.path = Path(run_dir, TASKS_DIR, EVENTS_NAME) if run_dir is not None else None
        self._handle: Any = None

    def write(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        try:
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                torn = _ends_mid_line(self.path)
                self._handle = open(self.path, "a", encoding="utf-8")  # kept open for the run
                if torn:  # a start that died mid-line: the line it left stays alone, and this start's first event too
                    self._handle.write("\n")
            self._handle.write(json.dumps({"event": event, **fields}, separators=(",", ":")) + "\n")
            self._handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            logger.warning("Cannot write %s (%s); the task event log stops", self.path, exc)
            self.path = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _ends_mid_line(path: Path) -> bool:
    """Whether a file holds a last line without its newline, as a write cut short leaves it."""

    try:
        with open(path, "rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except FileNotFoundError:
        return False


def read_events(run_dir: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """The events of a run's ``tasks/events.jsonl``, in order; a line that does not parse (one being written) is
    skipped, and a missing log has none."""

    path = Path(run_dir, TASKS_DIR, EVENTS_NAME)
    if not path.is_file():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event
