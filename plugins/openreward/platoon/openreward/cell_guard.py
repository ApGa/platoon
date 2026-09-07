"""Report programmatic-tool-calling cells that never come back.

This module **only measures and reports**.  It never interrupts, cancels,
truncates or kills a cell, and it never exits the process.  A legitimate cell
that runs for an hour is reported and then left alone; the guard's whole job is
to turn an invisible failure into a loud, countable one.

The failure it makes visible
============================

``openhands.tools.programmatic_tool_calling.impl`` executes each PTC cell
*in-process* on an ``InteractiveShellEmbed``
(``ProgrammaticToolCallingExecutor.__call__`` -> ``_run_cell`` ->
``shell.run_cell_async``).  There is no timeout anywhere on that path, and the
call runs on a worker thread taken from the event loop's default executor
(``Tool.acall`` -> ``loop.run_in_executor(None, ...)``, via ``Agent.astep`` ->
``_aexecute_actions``).

A Python thread cannot be killed.  ``asyncio`` cancellation, ``wait_for`` and
platoon's straggler timeout (``WorkflowConfig.straggler_timeout_seconds``) all
abandon the *awaiting coroutine*; the thread underneath keeps running the
model-authored code forever.  So a model that writes ``while True: pass`` costs
one executor thread and one core permanently, and nothing in the stack says so:
the trajectory is marked interrupted, the training loop keeps turning, and the
only symptom is that throughput quietly decays as threads are consumed.

Because the Tinker path hosts *every* rollout in one asyncio process on the head
node, those threads come from a pool that is shared by every concurrent rollout.

What this module installs
=========================

``install()`` wraps ``ProgrammaticToolCallingExecutor.__call__`` (idempotently,
and degrading to a no-op when OpenHands is absent) so that every cell is
registered with its start time, its worker thread and a one-line prefix of the
source the model wrote, and de-registered when it returns.  A daemon monitor
thread then reports, once per cell:

* a WARNING when a cell passes ``OPENREWARD_PTC_CELL_WARN_SECONDS`` (*stuck*),
* an ERROR when it passes ``OPENREWARD_PTC_CELL_LEAK_SECONDS`` (*presumed
  leaked*: the thread is treated as permanently lost),

plus a periodic summary while either count is non-zero.  :func:`snapshot`
exposes ``ptc_cells_stuck`` and ``ptc_threads_leaked`` so the trainer can log
them as scalars.  Both counts are *current*, not cumulative: a slow-but-finite
cell decrements them when it finishes, so a non-zero steady state means threads
that are genuinely gone.

The wrapper adds no timeout and no control flow: the cell's return value and any
exception it raises propagate exactly as before.

This is instrumentation, not a fix
==================================

The durable fix is to run each rollout in its own **process** (the AReaL path
did this; the Tinker path does not).  A process is the only unit of execution
Python can actually kill, so only process-per-rollout can bound a runaway cell
rather than merely counting it -- and some environments cannot share a process
at all regardless of runaway risk (AppWorld patches ``time`` with ``freezegun``
process-wide, so two concurrent AppWorld rollouts in one interpreter are already
incorrect).  Until then this module buys visibility, nothing more.

``OPENREWARD_PTC_CELL_GUARD=0`` makes the whole module a no-op.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CellGuardRegistry",
    "enabled",
    "install",
    "install_programmatic_tool_calling_cell_guard",
    "leak_seconds",
    "registry",
    "report_seconds",
    "snapshot",
    "source_prefix_chars",
    "start_monitor",
    "warn_seconds",
]

logger = logging.getLogger(__name__)

_FALSEY = {"0", "false", "no", "off"}

# Cell #1 is the first cell of the process; the counter is global (not
# per-executor) so a warning identifies one cell across every rollout.
_DEFAULT_WARN_SECONDS = 300.0
_DEFAULT_LEAK_SECONDS = 1800.0
_DEFAULT_REPORT_SECONDS = 60.0
_DEFAULT_SOURCE_PREFIX_CHARS = 240

# Upper bound on the monitor's sleep: a long warn threshold must not delay the
# periodic summary, and a tiny one (tests) must not spin.
_MAX_POLL_SECONDS = 15.0
_MIN_POLL_SECONDS = 0.02


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (AttributeError, ValueError):
        logger.warning("PTC cell guard: ignoring unparseable %s=%r; using %s", name, raw, default)
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError):
        logger.warning("PTC cell guard: ignoring unparseable %s=%r; using %s", name, raw, default)
        return default
    return value if value > 0 else default


def enabled() -> bool:
    """Whether :func:`install` should do anything.

    ``OPENREWARD_PTC_CELL_GUARD=0`` restores the unobserved upstream behaviour.
    The guard is passive, so there is no production reason to turn it off; the
    switch exists because the module monkeypatches a third-party class and an
    operator needs a documented way to take that out of the loop.
    """

    return os.environ.get("OPENREWARD_PTC_CELL_GUARD", "1").strip().lower() not in _FALSEY


def warn_seconds() -> float:
    """Age at which a still-running cell is reported as *stuck*."""

    return _env_float("OPENREWARD_PTC_CELL_WARN_SECONDS", _DEFAULT_WARN_SECONDS)


def leak_seconds() -> float:
    """Age at which a still-running cell's thread is *presumed leaked*.

    Never below :func:`warn_seconds`: a cell is always reported as stuck before
    it is written off.
    """

    return max(_env_float("OPENREWARD_PTC_CELL_LEAK_SECONDS", _DEFAULT_LEAK_SECONDS), warn_seconds())


def report_seconds() -> float:
    """Cadence of the periodic summary line (emitted only when counts are non-zero)."""

    return _env_float("OPENREWARD_PTC_CELL_GUARD_REPORT_SECONDS", _DEFAULT_REPORT_SECONDS)


def source_prefix_chars() -> int:
    """How much of the model-authored cell source to quote in a report."""

    return _env_int("OPENREWARD_PTC_CELL_SOURCE_CHARS", _DEFAULT_SOURCE_PREFIX_CHARS)


@dataclass
class _CellRecord:
    """One in-flight PTC cell."""

    cell_id: int
    started_at: float
    thread_name: str
    thread_ident: int
    source_prefix: str
    warned: bool = False
    leaked: bool = False

    def elapsed(self, now: float) -> float:
        return now - self.started_at


@dataclass
class CellGuardRegistry:
    """Process-wide book of in-flight cells, and the counts derived from it.

    Every method is safe to call from any thread and never raises for the
    caller: the wrapper around a cell must not be able to break the cell.
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _live: dict[int, _CellRecord] = field(default_factory=dict, repr=False)
    _next_cell_id: int = 1
    _started: int = 0
    _finished: int = 0
    _warnings_emitted: int = 0
    _leaks_reported: int = 0
    _last_report_at: float = 0.0

    # -- lifecycle -------------------------------------------------------
    def begin(self, code: Any, now: float | None = None) -> int:
        """Register a cell that is about to run on the calling thread."""

        thread = threading.current_thread()
        with self._lock:
            cell_id = self._next_cell_id
            self._next_cell_id += 1
            self._started += 1
            self._live[cell_id] = _CellRecord(
                cell_id=cell_id,
                started_at=time.monotonic() if now is None else now,
                thread_name=thread.name,
                thread_ident=thread.ident or 0,
                source_prefix=_source_prefix(code),
            )
        return cell_id

    def end(self, cell_id: int, now: float | None = None) -> None:
        """De-register a cell that returned (normally or by raising)."""

        with self._lock:
            record = self._live.pop(cell_id, None)
            if record is None:
                return
            self._finished += 1
            if not record.warned:
                return
        # Logged outside the lock: a cell that was already reported has to be
        # reported as recovered, or the earlier WARNING reads as permanent.
        elapsed = record.elapsed(time.monotonic() if now is None else now)
        logger.warning(
            "PTC cell guard: cell #%d finished after %.0fs on thread %s (id=%s) "
            "-- slow, not leaked%s",
            record.cell_id,
            elapsed,
            record.thread_name,
            record.thread_ident,
            " (was presumed leaked)" if record.leaked else "",
        )

    # -- counts ----------------------------------------------------------
    def counts(self, now: float | None = None) -> dict[str, int]:
        """Current stuck/leaked/running counts, recomputed from the live set."""

        moment = time.monotonic() if now is None else now
        warn = warn_seconds()
        leak = leak_seconds()
        stuck = 0
        leaked = 0
        with self._lock:
            for record in self._live.values():
                elapsed = record.elapsed(moment)
                if elapsed >= warn:
                    stuck += 1
                if elapsed >= leak:
                    leaked += 1
            running = len(self._live)
            started = self._started
            finished = self._finished
        return {
            "ptc_cells_stuck": stuck,
            "ptc_threads_leaked": leaked,
            "ptc_cells_running": running,
            "ptc_cells_started": started,
            "ptc_cells_finished": finished,
        }

    # -- reporting -------------------------------------------------------
    def poll(self, now: float | None = None) -> dict[str, int]:
        """Emit any new per-cell reports plus the periodic summary.

        Separated from the monitor thread so the reporting logic can be driven
        deterministically (by tests, and by anything that wants a report on
        demand).  Returns the same mapping as :meth:`counts`.
        """

        moment = time.monotonic() if now is None else now
        warn = warn_seconds()
        leak = leak_seconds()
        newly_stuck: list[tuple[_CellRecord, float]] = []
        newly_leaked: list[tuple[_CellRecord, float]] = []
        with self._lock:
            for record in self._live.values():
                elapsed = record.elapsed(moment)
                if not record.warned and elapsed >= warn:
                    record.warned = True
                    self._warnings_emitted += 1
                    newly_stuck.append((record, elapsed))
                if not record.leaked and elapsed >= leak:
                    record.leaked = True
                    self._leaks_reported += 1
                    newly_leaked.append((record, elapsed))

        for record, elapsed in newly_stuck:
            logger.warning(
                "PTC cell guard: cell #%d still running after %.0fs (warn threshold %.0fs) "
                "on thread %s (id=%s). The cell runs in-process on an IPython shell with no "
                "timeout and cannot be interrupted; this thread is occupied until the cell "
                "returns on its own. Cell source: %s",
                record.cell_id,
                elapsed,
                warn,
                record.thread_name,
                record.thread_ident,
                record.source_prefix,
            )
        for record, elapsed in newly_leaked:
            logger.error(
                "PTC cell guard: cell #%d PRESUMED LEAKED after %.0fs (leak threshold %.0fs) "
                "on thread %s (id=%s). Treat this rollout worker thread as permanently lost: "
                "Python cannot kill a running thread, so it is counted, not recovered. "
                "Cell source: %s",
                record.cell_id,
                elapsed,
                leak,
                record.thread_name,
                record.thread_ident,
                record.source_prefix,
            )

        counts = self.counts(moment)
        self._maybe_report_summary(counts, moment)
        return counts

    def _maybe_report_summary(self, counts: dict[str, int], moment: float) -> None:
        if not (counts["ptc_cells_stuck"] or counts["ptc_threads_leaked"]):
            return
        with self._lock:
            if moment - self._last_report_at < report_seconds():
                return
            self._last_report_at = moment
        logger.warning(
            "PTC cell guard: %d cell(s) stuck >%.0fs, %d thread(s) presumed leaked >%.0fs "
            "(%d cell(s) in flight; %d started / %d finished this process). "
            "Leaked threads are never returned to the rollout executor pool.",
            counts["ptc_cells_stuck"],
            warn_seconds(),
            counts["ptc_threads_leaked"],
            leak_seconds(),
            counts["ptc_cells_running"],
            counts["ptc_cells_started"],
            counts["ptc_cells_finished"],
        )

    def poll_interval(self) -> float:
        """How long the monitor should sleep between :meth:`poll` calls."""

        return max(_MIN_POLL_SECONDS, min(_MAX_POLL_SECONDS, warn_seconds() / 4.0))


def _source_prefix(code: Any) -> str:
    """A single-line, bounded excerpt of the model-authored cell source."""

    try:
        text = code if isinstance(code, str) else str(code)
    except Exception:  # pragma: no cover - defensive: __str__ of a hostile object
        return "<unavailable>"
    collapsed = " ".join(text.split())
    if not collapsed:
        return "<empty>"
    limit = source_prefix_chars()
    if len(collapsed) <= limit:
        return repr(collapsed)
    return repr(collapsed[:limit]) + f" ... (+{len(collapsed) - limit} chars)"


_REGISTRY = CellGuardRegistry()

_install_lock = threading.RLock()
_monitor_thread: threading.Thread | None = None


def registry() -> CellGuardRegistry:
    """The process-wide registry the installed wrapper writes to."""

    return _REGISTRY


def snapshot() -> dict[str, int]:
    """Current guard counters, for the trainer's scalar logging.

    Always safe to call, including when the guard was never installed (every
    count is then zero).
    """

    try:
        return _REGISTRY.counts()
    except Exception:  # pragma: no cover - a metric must never break a train step
        return {
            "ptc_cells_stuck": 0,
            "ptc_threads_leaked": 0,
            "ptc_cells_running": 0,
            "ptc_cells_started": 0,
            "ptc_cells_finished": 0,
        }


def _monitor_loop() -> None:
    while True:
        try:
            _REGISTRY.poll()
        except Exception:  # pragma: no cover - the monitor must outlive its own bugs
            logger.exception("PTC cell guard: monitor iteration failed")
        time.sleep(_REGISTRY.poll_interval())


def start_monitor() -> bool:
    """Start the daemon monitor thread once."""

    global _monitor_thread
    with _install_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False
        thread = threading.Thread(target=_monitor_loop, name="ptc-cell-guard", daemon=True)
        thread.start()
        _monitor_thread = thread
        return True


def install_programmatic_tool_calling_cell_guard() -> bool:
    """Wrap the PTC executor's cell entry point so every cell is accounted for.

    Returns whether this call installed the wrapper (``False`` when it was
    already installed or when OpenHands is not importable).
    """

    try:
        from openhands.tools.programmatic_tool_calling import impl as ptc_impl
    except Exception:
        return False
    executor_cls = getattr(ptc_impl, "ProgrammaticToolCallingExecutor", None)
    if executor_cls is None:
        return False

    with _install_lock:
        original = executor_cls.__call__
        if getattr(original, "_platoon_cell_guard", False):
            return False

        @functools.wraps(original)
        def guarded_call(self, action, *args, **kwargs):
            # Bookkeeping is wrapped defensively at both ends: a bug in the
            # guard must never change whether a cell runs, what it returns, or
            # what it raises.
            try:
                cell_id: int | None = _REGISTRY.begin(getattr(action, "code", ""))
            except Exception:
                logger.exception("PTC cell guard: failed to register a cell; running it unobserved")
                cell_id = None
            try:
                return original(self, action, *args, **kwargs)
            finally:
                if cell_id is not None:
                    try:
                        _REGISTRY.end(cell_id)
                    except Exception:
                        logger.exception("PTC cell guard: failed to de-register cell #%s", cell_id)

        guarded_call._platoon_cell_guard = True  # type: ignore[attr-defined]
        executor_cls.__call__ = guarded_call  # type: ignore[method-assign]
        return True


def install() -> dict[str, bool]:
    """Install the guard; safe to call repeatedly.

    A no-op under ``OPENREWARD_PTC_CELL_GUARD=0``.
    """

    if not enabled():
        return {"cell_guard": False, "monitor": False}
    installed = install_programmatic_tool_calling_cell_guard()
    monitor = start_monitor()
    if installed:
        logger.info(
            "PTC cell guard installed: reporting cells still running after %.0fs, "
            "presuming their threads leaked after %.0fs. The guard never interrupts a cell.",
            warn_seconds(),
            leak_seconds(),
        )
    return {"cell_guard": installed, "monitor": monitor}
