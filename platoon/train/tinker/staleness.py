"""The staleness warning (RFC 0020): when more than 2% of the groups the last 10 steps consumed went stale.

Stale groups are the long ones, which are harder and more often mixed, so dropping them biases what trains.  The
warning names what held the dispatcher back most over those steps (its ``dispatch/blocked_share/<reason>`` metrics)
and the fixes: ``train.max_staleness + 1``, which changes only the stale filter where the horizon had room, and the
per-datum stale filter that RFC 0020 plans.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from collections.abc import Mapping

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


class StaleWatch:
    """Counts the groups each step consumes and those the stale filter drops (:meth:`dequeued`), and warns once per
    start of a run (:meth:`step_ended`).  ``max_staleness`` is the run's bound (None: no stale filter, no warning)."""

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
            f"{raise_bound} A per-datum stale filter, which RFC 0020 plans for a later release, would keep the part of "
            "a long rollout within the bound instead of dropping its whole group. Stale groups are the long ones, "
            "which are harder and more often mixed, so dropping them biases what trains "
            "(docs/how-to/train-and-eval.md; warned once per start)"
        )
