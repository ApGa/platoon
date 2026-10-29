"""Staleness, as a step collects its groups (the first stage of a step): the stale filter, which drops a group whose
oldest datum is more policy versions old than its entry's bound, or under ``train.stale_filter: datum`` only the datums
that are, the whole group only when every datum is (:class:`StaleFilter`; the version a datum carries,
:func:`datum_version`); and the staleness warning (RFC 0020, :class:`StaleWatch`): when more than 2% of the groups the
last 10 steps consumed went stale.

Stale groups are the long ones, which are harder and more often mixed, so dropping them biases what trains.  The
warning names what held the dispatcher back most over those steps (its ``dispatch/blocked_share/<reason>`` metrics)
and the fixes: ``train.max_staleness + 1``, which changes only the stale filter where the horizon had room, and the
per-datum stale filter, which keeps the part of a long rollout within the bound.

The per-datum filter keeps the advantages the group's credit gave each datum it keeps: the credit ran over the whole
group as it settled, so each kept term keeps its expectation.  A datum's version is the oldest its own model calls
sampled when its workflow stamps it so (``GroupRolloutWorkflow`` does under ``datum``), else its rollout's.

In a run that trains several policies, each policy's rows of a group are judged on their own, against that policy's
version and bound (:meth:`StaleFilter.for_policy`): a policy's stale rows are dropped together, and the group's other
rows still train.

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, asks the filter of each group it dequeues, and
logs, counts and drops the stale groups and datums.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from platoon.train.config_defs import TrainConfig

if TYPE_CHECKING:
    import tinker

    from platoon.train.entry_settings import EntrySettings

logger = logging.getLogger(__name__)

# Over the groups the last STALE_WINDOW steps consumed, more than STALE_SHARE dropped as stale.
STALE_WINDOW = 10
STALE_SHARE = 0.02
# A horizon that held the dispatcher back for less than this share of those steps had room.
HORIZON_FREE = 0.05

_HELD_BACK_BY = {
    "permits": "the rollout permits (train.max_concurrent_rollouts)",
    "horizon": "the horizon (train.max_staleness)",
    "labels": "the labels' limits (train.rollout_group_limits)",
    "plan": "the end of the run's plan",
    "sampler": "a task sampler that found no eligible task",
}


@dataclass(frozen=True)
class StaleVerdict:
    """What the stale filter found of a group: ``stale``, and of a stale one, the version of its oldest datum
    (``version``), the policy's (``now``) and the bound it is beyond (``bound``).  Under the per-datum filter, a group
    that is not stale may have datums beyond the bound, which the step leaves out: their indices in its rows
    (``masked``), with the version of the oldest, the policy's and the bound."""

    stale: bool
    version: int | None = None
    now: int | None = None
    bound: int | None = None
    masked: tuple[int, ...] = ()


_NOT_STALE = StaleVerdict(stale=False)


def datum_version(datum: tinker.Datum) -> int | None:
    """The policy version a datum's tokens were sampled at (its ``checkpoint_version`` loss input), if it has one."""

    version = datum.loss_fn_inputs.get("checkpoint_version")
    return int(version.to_torch().item()) if version is not None else None


class StaleFilter:
    """The stale filter's bound for each entry's groups: its own ``max_staleness`` (``bounds``), else the run's
    (``run_bound``; None: no bound).  ``per_datum`` (``train.stale_filter: datum``): it judges each datum against the
    bound, not each group."""

    def __init__(self, run_bound: int | None, bounds: Mapping[str, int | None], *, per_datum: bool = False) -> None:
        self.run_bound = run_bound
        self.bounds = bounds
        self.per_datum = per_datum

    @classmethod
    def of_run(cls, train: TrainConfig, settings: Mapping[str, EntrySettings]) -> StaleFilter:
        """The stale filter of a run's steps: the run's bound (``train.max_staleness``) and each entry's own (the
        ``max_staleness`` of its ``settings``, by label), per datum under ``train.stale_filter: datum``.  A partial
        config without ``stale_filter`` (a test's ``SimpleNamespace``) reads as ``TrainConfig``'s default."""

        bounds = {label: own.max_staleness for label, own in settings.items()}
        stale_filter = getattr(train, "stale_filter", TrainConfig.stale_filter)
        return cls(train.max_staleness, bounds, per_datum=stale_filter == "datum")

    def for_policy(self, bound: int | None) -> StaleFilter:
        """The filter of a trained policy's rows: this one, the run's (each entry's bound, else the run's), for a
        policy without a bound of its own; else one of the policy's ``bound`` for every entry's rows, judged as this one
        judges (per datum or per group)."""

        return self if bound is None else StaleFilter(bound, {}, per_datum=self.per_datum)

    def check(self, entry: str | None, output: Sequence[Any] | None, now: int) -> StaleVerdict:
        """Whether a group of ``entry`` whose output is ``output`` (None: it failed) is stale at policy version
        ``now``: more versions old than its bound.  Group members can begin sampling at different versions after
        environment setup, so a group is as old as its oldest datum, and is kept or dropped whole, its advantages
        intact; a datum without a version is as new as the policy.  Under the per-datum filter, each datum more
        versions old than the bound is masked, and the group is stale only when all of them are
        (:meth:`_check_datums`)."""

        max_staleness = self.bounds.get(entry, self.run_bound) if entry is not None else self.run_bound
        if max_staleness is None or output is None or len(output) == 0:
            return _NOT_STALE
        if self.per_datum:
            return self._check_datums(output, now, max_staleness)
        versions = [version for datum in output if (version := datum_version(datum)) is not None]
        rollout_checkpoint_version = min(versions, default=now)
        if now - rollout_checkpoint_version > max_staleness:
            return StaleVerdict(stale=True, version=rollout_checkpoint_version, now=now, bound=max_staleness)
        return _NOT_STALE

    @staticmethod
    def _check_datums(output: Sequence[Any], now: int, max_staleness: int) -> StaleVerdict:
        """The per-datum verdict on a group's rows ``output`` at policy version ``now``: the rows more than
        ``max_staleness`` versions old, masked, or the whole group stale when every row is."""

        versions = [datum_version(datum) for datum in output]
        masked = tuple(
            index for index, version in enumerate(versions) if version is not None and now - version > max_staleness
        )
        if not masked:
            return _NOT_STALE
        oldest = min(version for version in versions if version is not None)
        if len(masked) == len(versions):
            return StaleVerdict(stale=True, version=oldest, now=now, bound=max_staleness)
        return StaleVerdict(stale=False, version=oldest, now=now, bound=max_staleness, masked=masked)


class StaleWatch:
    """Counts the groups each step consumes and those the stale filter drops (:meth:`dequeued`), and warns once per
    start of a run (:meth:`step_ended`).  ``max_staleness`` is the run's bound (None: no stale filter, no warning).
    Under the per-datum filter it counts the groups dropped whole only: a group that keeps any datum counts as
    consumed, however many it lost (``train/batch/stale_datums`` counts those)."""

    def __init__(self, max_staleness: int | None) -> None:
        self.max_staleness = max_staleness
        self.warned = False
        self._groups = self._stale = 0
        # The last STALE_WINDOW steps: the groups each consumed, those dropped as stale, and its blocked shares.
        self._window: deque[tuple[int, int, Mapping[str, float]]] = deque(maxlen=STALE_WINDOW)

    def dequeued(self, stale: bool) -> None:
        """The step consumed a group (not a marker of one decided when it settled); ``stale``: the filter dropped it."""

        self._groups += 1
        self._stale += stale

    def step_ended(self, blocked: Mapping[str, float]) -> None:
        """The step ended, with the dispatcher's ``dispatch/blocked_share/<reason>`` metrics (none without one)."""

        self._window.append((self._groups, self._stale, blocked))
        self._groups = self._stale = 0
        if not self.warned and (message := self._message()) is not None:
            self.warned = True
            logger.warning(message)

    def _message(self) -> str | None:
        if len(self._window) < STALE_WINDOW or self.max_staleness is None:
            return None
        groups = sum(count for count, _stale, _blocked in self._window)
        stale = sum(count for _groups, count, _blocked in self._window)
        if not groups or stale / groups <= STALE_SHARE:
            return None
        shares: Counter[str] = Counter()
        for _groups, _stale, blocked in self._window:
            shares.update({key.rsplit("/", 1)[-1]: share / STALE_WINDOW for key, share in blocked.items()})
        horizon = shares["horizon"]
        binding = max(shares, key=lambda reason: shares[reason], default=None)
        if binding is not None and shares[binding] > 0:
            held = f"what held the dispatcher back most was {_HELD_BACK_BY[binding]} ({shares[binding]:.0%})"
        else:
            held = "nothing held the dispatcher back"
        k = self.max_staleness
        if horizon < HORIZON_FREE:
            raise_bound = (
                f"The horizon had room (it held the dispatcher back {horizon:.0%} of the time), so "
                f"train.max_staleness: {k + 1} changes only the stale filter: a group may then be one more policy "
                "version old and train."
            )
        else:
            raise_bound = (
                f"The horizon held the dispatcher back {horizon:.0%} of the time, so train.max_staleness: {k + 1} also "
                "lets rollouts run one more step ahead, which ages the data further."
            )
        return (
            f"{stale / groups:.1%} of the groups the last {STALE_WINDOW} steps consumed ({stale} of {groups}) were "
            f"dropped as stale, older than their bound (train.max_staleness: {k}, or their entry's own); {held}. "
            f"{raise_bound} Under train.stale_filter: datum (experimental), a group loses only its datums beyond the "
            "bound, and is dropped whole only when none is within it. Stale groups are the long ones, which are harder "
            "and more often mixed, so dropping them biases what trains (docs/how-to/train-and-eval.md; warned once per "
            "start)"
        )
