"""``example-partial-credit``: a reward processor that gives near misses part of the reward.

A reward processor turns each finished trajectory, as a dict, into the reward it trains on and the components that
training reports as metrics. This one keeps an exact answer at 1.0 and gives a wrong numeric answer half of the
``close`` reward that ``example-exact-answer`` graded, so near misses train better than far ones. Evaluation still
counts only exact answers as successes: its success rate reads the graded reward, not this one.
"""

from __future__ import annotations

from typing import Any


def partial_credit(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """The training reward of one trajectory, and its components (keys that start with ``reward/``)."""

    graded = ((trajectory.get("misc") or {}).get("outcome") or {}).get("rewards") or {}
    exact = float(trajectory.get("reward") or 0.0)
    close = float(graded.get("close", 0.0))
    reward = exact if exact >= 1.0 else 0.5 * close
    return reward, {"reward/exact": exact, "reward/close": close}
