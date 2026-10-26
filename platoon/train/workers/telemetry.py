"""Cross-process aggregation of the programmatic-tool-calling cell guard.

The guard ships with the extension that owns programmatic tool calling and
reports its counters through the ``rollout_worker.telemetry`` hook (see
:mod:`platoon.hooks`); :func:`local_telemetry` reads them for this process.

The guard's registry is **process-global**, so once rollouts run elsewhere the
trainer's own snapshot is permanently empty -- and ``_ptc_cell_guard_stats``
swallows every exception and returns ``{}``, so that regression would be
completely silent under unchanged key names.

Summing the guard's *gauges* across workers would be worse than useless.
``ptc_cells_stuck``, ``ptc_threads_leaked`` and ``ptc_cells_running`` are
recomputed from the live set on every call and are documented as *current, not
cumulative*: every healthy short-lived worker reports 0 at exit, and the one
worker that had a stuck cell is exactly the one that gets SIGKILLed and writes
no response.  The metric would read zero precisely when cells are being killed
for hanging.

So workers forward only the genuinely cumulative counters plus the gauges
sampled **once at exit** under distinct ``*_at_exit`` names, and a worker that
is killed contributes the same counters through its ``.diag`` marker.  The
instantaneous gauge names stay reserved for ``inprocess`` mode, where they keep
their original meaning.
"""

from __future__ import annotations

import logging
import threading

from platoon.hooks import hooks

__all__ = [
    "CUMULATIVE_KEYS",
    "local_telemetry",
    "record_worker_cell_guard",
    "reset_for_tests",
    "worker_cell_guard_stats",
]

#: Only these cross a process boundary.  ``ptc_cells_running`` deliberately
#: does not: an exited worker is running nothing, so summing it is meaningless.
CUMULATIVE_KEYS = (
    "ptc_cells_started",
    "ptc_cells_finished",
    "ptc_cells_stuck_at_exit",
    "ptc_threads_leaked_at_exit",
)

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_TOTALS: dict[str, dict[str, float]] = {}
_REPORTS: dict[str, int] = {}


def local_telemetry() -> dict[str, float]:
    """This process's counters from every ``rollout_worker.telemetry`` hook.

    Empty when no extension registered one.  Never raises: a failing hook
    contributes nothing, because this runs inline in the train loop's stats
    path and in a dying worker's diagnostics.
    """

    counters: dict[str, float] = {}
    for snapshot in hooks("rollout_worker.telemetry"):
        try:
            counters.update({key: float(value) for key, value in snapshot().items()})
        except Exception:
            logger.debug("rollout_worker.telemetry hook %r failed", snapshot, exc_info=True)
    return counters


def record_worker_cell_guard(scope: str, counters) -> None:
    """Accumulate one worker's counters into ``scope``.  Never raises.

    Accepts both a response's ``cell_guard`` dict and a ``.diag`` marker's
    string-valued fields, so a SIGKILLed worker still contributes.
    """

    if not counters:
        return
    try:
        contributed = {}
        for key in CUMULATIVE_KEYS:
            if key not in counters:
                continue
            try:
                contributed[key] = float(counters[key])
            except (TypeError, ValueError):
                continue
        if not contributed:
            return
        with _LOCK:
            totals = _TOTALS.setdefault(scope, {})
            for key, value in contributed.items():
                totals[key] = totals.get(key, 0.0) + value
            _REPORTS[scope] = _REPORTS.get(scope, 0) + 1
    except Exception:  # pragma: no cover - instrumentation must not raise
        return


def worker_cell_guard_stats(scope: str, *, reset: bool = True) -> dict[str, float]:
    """Aggregated worker counters for one training step.  Never raises."""

    try:
        with _LOCK:
            totals = dict(_TOTALS.get(scope, {}))
            reports = _REPORTS.get(scope, 0)
            if reset:
                _TOTALS.pop(scope, None)
                _REPORTS.pop(scope, None)
        if not reports:
            return {}
        stats = {key: totals.get(key, 0.0) for key in CUMULATIVE_KEYS}
        stats["ptc_worker_reports"] = float(reports)
        return stats
    except Exception:  # pragma: no cover
        return {}


def reset_for_tests() -> None:
    with _LOCK:
        _TOTALS.clear()
        _REPORTS.clear()
