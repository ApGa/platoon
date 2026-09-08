"""Client-side throughput instrumentation for the Tinker rollout path.

Why this module exists
======================

The Tinker path is rollout-bound: a training step spends the overwhelming
majority of its wall clock collecting rollouts, and the SGLang engines behind
the gateway are nearly idle while it does.  Nothing in the trainer measured the
*client*, so "the client is the bottleneck" could only be inferred.  These
metrics measure it directly.  Every key answers one question:

Concurrency and queueing -- *are we running as many rollouts at once as we
asked for, and is the trainer waiting on them?*

``rollout_client/rollouts_in_flight_mean``
    Time-weighted mean number of rollouts running concurrently over the step.
    Compare with ``rollouts_in_flight_configured``: a large shortfall means
    rollouts are not actually overlapping.
``rollout_client/rollouts_in_flight_max``
    Peak concurrency actually reached in the step.
``rollout_client/rollouts_in_flight_configured``
    The ceiling implied by config (workflow workers x group size).
``rollout_client/groups_in_flight_mean`` / ``_max``
    The same at task granularity, so a group-atomic admission gate is visible.
``rollout_client/trainer_queue_wait_seconds_total`` / ``_mean`` / ``_max``
    How long the training loop blocked on the rollout result queue this step.
    This is the trainer's own view of "rollout-bound": if it is close to the
    step wall clock, training is starved; if it is near zero, rollouts are
    keeping up and the step is training-bound.
``rollout_client/trainer_queue_ready_fraction``
    Fraction of dequeues that were satisfied immediately (a result was already
    waiting).  1.0 means the producer is comfortably ahead.

Where an episode's wall clock goes -- *model call, client CPU, or the
environment?*  Recorded per model call by the litellm provider and pooled
across every rollout consumed in the step.

``rollout_client/agent_step_seconds_p50`` / ``_p90``
    One agent step, measured end-of-previous-model-call to
    end-of-this-model-call: environment/tool work plus prompt rendering plus
    the sampling round trip plus response parsing.
``rollout_client/agent_step_model_seconds_p50`` / ``_p90``
    Of that step, the time awaiting ``sample_async`` (the gateway round trip).
``rollout_client/agent_step_client_seconds_p50`` / ``_p90``
    Of that step, the time spent *inside the client* rendering and tokenizing
    the prompt and parsing/detokenizing the response.  This is the work that
    serialises on the GIL when many rollouts share one interpreter, so a large
    value here -- or a value that grows with concurrency -- is the direct
    signature of a client-side bottleneck.
``rollout_client/agent_step_tool_seconds_p50`` / ``_p90``
    Of that step, everything between the end of one model call and the start of
    the next: tool execution, environment round trips, agent scaffolding.
``rollout_client/agent_step_model_fraction`` / ``_client_fraction`` /
``_tool_fraction``
    The same split as shares of total recorded agent-step time, so one number
    says where the episode went.
``rollout_client/model_calls_per_rollout_mean``
    Model calls in one rollout tree; the multiplier on every per-call cost.

Client CPU -- *is the client saturated?*

``rollout_client/cpu_percent``
    Trainer-process CPU over the step (100 = one core busy).
``rollout_client/worker_cpu_percent``
    CPU of reaped child processes over the step, from ``RUSAGE_CHILDREN``, so
    with process-per-rollout the rollout CPU shows up here instead.
``rollout_client/total_cpu_percent`` / ``cpu_cores_busy`` / ``cpu_saturation``
    Trainer plus children; cores busy; and that as a fraction of the node's
    cores.  ``cpu_saturation`` near 1.0 means the client is out of CPU.
``rollout_client/rss_bytes`` / ``worker_rss_bytes`` /
``worker_peak_rss_bytes_sum`` / ``worker_peak_rss_bytes_max``
    Trainer RSS, live worker RSS at sample time, and the workers' own
    self-sampled peaks for the rollouts consumed this step.
``rollout_client/threads``
    Trainer OS thread count -- the pool that a runaway PTC cell consumes.

Cost of isolation -- *what did process-per-rollout cost us?*

``rollout_client/dequeue_to_first_model_call_seconds_p50`` / ``_p90``
    From a task leaving the trainer's queue to that rollout's first model call.
    In ``process`` mode this includes worker launch and imports, so it is the
    end-to-end startup cost as the schedule feels it.
``rollout_worker_launch_seconds`` / ``rollout_worker_startup_seconds``
    (emitted by the workflow) exec-to-ready, and the child's own import and
    registration time.

All of this is best-effort: nothing here may raise into a rollout or a train
step.  Every public entry point swallows its own errors.
"""

from __future__ import annotations

import logging
import os
import resource
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_TIMING_SAMPLES",
    "ClientThroughputMonitor",
    "RolloutTimeline",
    "RolloutTimingSummary",
    "current_timeline",
    "monitor",
    "record_model_call",
    "step_metrics",
    "timeline_scope",
]

# Per-call timings shipped back from a worker are strided down to this many
# entries: 512 four-float tuples is ~16 KB against a 1-6 MB response, and is
# plenty for a p50/p90 pooled over dozens of rollouts.
MAX_TIMING_SAMPLES = 512

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
_CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


# --------------------------------------------------------------------------
# Per-rollout timeline (runs in whichever process hosts the rollout)
# --------------------------------------------------------------------------


@dataclass
class RolloutTimeline:
    """Mutable per-rollout-tree record of model-call phase timings.

    Lives in a :class:`~contextvars.ContextVar` set by
    ``TinkerLLMProxySession``, exactly like the interaction map, so subagents,
    judges and the reward verifier all write into their root rollout's
    timeline.
    """

    started_at: float = field(default_factory=time.monotonic)
    first_call_started_at: float | None = None
    last_call_ended_at: float | None = None
    calls: list[tuple[float, float, float, float]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        *,
        started_at: float,
        render_seconds: float,
        sample_seconds: float,
        parse_seconds: float,
        ended_at: float,
    ) -> None:
        """Record one completed model call.  Never raises."""

        try:
            with self.lock:
                if self.first_call_started_at is None:
                    self.first_call_started_at = started_at
                previous_end = self.last_call_ended_at
                # Time since the previous model call in this rollout tree
                # finished: tool execution, environment round trips and agent
                # scaffolding.  For the first call there is no previous end, so
                # the gap is attributed to startup instead (see
                # ``first_model_call_seconds``) and recorded as zero here.
                gap = max(0.0, started_at - previous_end) if previous_end is not None else 0.0
                self.last_call_ended_at = ended_at
                self.calls.append(
                    (
                        float(gap),
                        max(0.0, float(render_seconds)),
                        max(0.0, float(sample_seconds)),
                        max(0.0, float(parse_seconds)),
                    )
                )
        except Exception:  # pragma: no cover - instrumentation must not break a rollout
            logger.debug("rollout timeline: failed to record a model call", exc_info=True)

    def summary(self, *, now: float | None = None) -> "RolloutTimingSummary":
        """Freeze this timeline into a picklable summary.  Never raises."""

        try:
            moment = time.monotonic() if now is None else now
            with self.lock:
                calls = list(self.calls)
                started_at = self.started_at
                first = self.first_call_started_at
            return RolloutTimingSummary.from_calls(
                calls,
                rollout_seconds=max(0.0, moment - started_at),
                first_model_call_seconds=(max(0.0, first - started_at) if first is not None else -1.0),
            )
        except Exception:  # pragma: no cover
            logger.debug("rollout timeline: failed to summarise", exc_info=True)
            return RolloutTimingSummary()


def _stride(values: list[tuple[float, float, float, float]], limit: int) -> tuple:
    if len(values) <= limit:
        return tuple(values)
    step = len(values) / float(limit)
    return tuple(values[min(len(values) - 1, int(index * step))] for index in range(limit))


@dataclass(frozen=True)
class RolloutTimingSummary:
    """Picklable per-rollout timing, exact in the sums and strided in samples.

    ``first_model_call_seconds`` is ``-1.0`` when the rollout never reached a
    model call, so "no data" is never confused with "zero seconds".
    """

    model_calls: int = 0
    rollout_seconds: float = 0.0
    first_model_call_seconds: float = -1.0
    tool_seconds_total: float = 0.0
    render_seconds_total: float = 0.0
    sample_seconds_total: float = 0.0
    parse_seconds_total: float = 0.0
    # (gap, render, sample, parse) per call, strided to MAX_TIMING_SAMPLES.
    samples: tuple[tuple[float, float, float, float], ...] = ()

    @classmethod
    def from_calls(
        cls,
        calls: list[tuple[float, float, float, float]],
        *,
        rollout_seconds: float,
        first_model_call_seconds: float,
    ) -> "RolloutTimingSummary":
        return cls(
            model_calls=len(calls),
            rollout_seconds=float(rollout_seconds),
            first_model_call_seconds=float(first_model_call_seconds),
            tool_seconds_total=float(sum(call[0] for call in calls)),
            render_seconds_total=float(sum(call[1] for call in calls)),
            sample_seconds_total=float(sum(call[2] for call in calls)),
            parse_seconds_total=float(sum(call[3] for call in calls)),
            samples=_stride(calls, MAX_TIMING_SAMPLES),
        )


_timeline: ContextVar[RolloutTimeline | None] = ContextVar("platoon_rollout_timeline", default=None)


def current_timeline() -> RolloutTimeline | None:
    """The timeline for the rollout tree running on this task, if any."""

    try:
        return _timeline.get()
    except Exception:  # pragma: no cover
        return None


@contextmanager
def timeline_scope(timeline: RolloutTimeline | None = None) -> Iterator[RolloutTimeline]:
    """Bind a fresh timeline for the duration of one rollout tree."""

    active = timeline if timeline is not None else RolloutTimeline()
    token = _timeline.set(active)
    try:
        yield active
    finally:
        try:
            _timeline.reset(token)
        except ValueError:  # pragma: no cover - reset in a different context
            _timeline.set(None)


def record_model_call(
    *,
    started_at: float,
    render_seconds: float,
    sample_seconds: float,
    parse_seconds: float,
    ended_at: float,
) -> None:
    """Record one model call against the active rollout timeline.  Never raises."""

    active = current_timeline()
    if active is None:
        return
    active.record(
        started_at=started_at,
        render_seconds=render_seconds,
        sample_seconds=sample_seconds,
        parse_seconds=parse_seconds,
        ended_at=ended_at,
    )


# --------------------------------------------------------------------------
# Process resource sampling (no third-party dependency: /proc + getrusage)
# --------------------------------------------------------------------------


def read_rss_bytes(pid: int | None = None) -> int:
    """Resident set size in bytes, or 0 when it cannot be read."""

    target = "self" if pid is None else str(pid)
    try:
        with open(f"/proc/{target}/statm", "rb") as handle:
            fields = handle.read().split()
        return int(fields[1]) * _PAGE_SIZE
    except Exception:
        return 0


def read_thread_count(pid: int | None = None) -> int:
    target = "self" if pid is None else str(pid)
    try:
        with open(f"/proc/{target}/status", "rb") as handle:
            for line in handle:
                if line.startswith(b"Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def read_process_cpu_seconds(pid: int | None = None) -> float:
    """User+system CPU seconds consumed by a process, or 0.0."""

    target = "self" if pid is None else str(pid)
    try:
        with open(f"/proc/{target}/stat", "rb") as handle:
            raw = handle.read()
        # comm may contain spaces and parentheses; fields are counted after it.
        tail = raw[raw.rindex(b")") + 2 :].split()
        # /proc/pid/stat fields 14 and 15 (1-based) are utime and stime; after
        # dropping pid and comm they are offsets 11 and 12.
        return (int(tail[11]) + int(tail[12])) / float(_CLOCK_TICKS)
    except Exception:
        return 0.0


def _self_cpu_seconds() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    except Exception:  # pragma: no cover
        return 0.0


def _children_cpu_seconds() -> float:
    """CPU seconds of *reaped* children (and their reaped descendants).

    The supervisor waits on every worker it launches, so with
    process-per-rollout this is an exact per-step accounting of rollout CPU
    with no sampling error and no extra thread.
    """

    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime
    except Exception:  # pragma: no cover
        return 0.0


# --------------------------------------------------------------------------
# Parent-side aggregation, one monitor per stats scope
# --------------------------------------------------------------------------


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


class _InFlightGauge:
    """Time-weighted concurrency gauge over an export window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current = 0
        self._max = 0
        self._integral = 0.0
        self._last_change = time.monotonic()
        self._window_start = self._last_change

    def _accumulate(self, now: float) -> None:
        self._integral += self._current * (now - self._last_change)
        self._last_change = now

    def enter(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._accumulate(now)
            self._current += 1
            self._max = max(self._max, self._current)

    def exit(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._accumulate(now)
            self._current = max(0, self._current - 1)

    def sample(self, *, reset: bool) -> tuple[float, int, int]:
        with self._lock:
            now = time.monotonic()
            self._accumulate(now)
            elapsed = max(1e-9, now - self._window_start)
            mean = self._integral / elapsed
            peak = self._max
            current = self._current
            if reset:
                self._integral = 0.0
                self._window_start = now
                self._max = current
            return mean, peak, current


class ClientThroughputMonitor:
    """Per-stats-scope accumulator for the client-side throughput metrics.

    Every method is safe to call from any thread and never raises for its
    caller: instrumentation must not be able to break a rollout or a step.
    """

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self._lock = threading.Lock()
        self._rollouts = _InFlightGauge()
        self._groups = _InFlightGauge()
        self._queue_waits: list[float] = []
        self._queue_immediate = 0
        self._queue_dequeues = 0
        self._first_call_seconds: list[float] = []
        self._call_samples: list[tuple[float, float, float, float]] = []
        self._model_calls_per_rollout: list[float] = []
        self._tool_total = 0.0
        self._render_total = 0.0
        self._sample_total = 0.0
        self._parse_total = 0.0
        self._worker_peak_rss: list[int] = []
        self._worker_pids: dict[int, int] = {}
        self._configured_concurrency = 0.0
        self._window_start = time.monotonic()
        self._self_cpu = _self_cpu_seconds()
        self._children_cpu = _children_cpu_seconds()

    # -- concurrency -----------------------------------------------------
    def set_configured_concurrency(self, value: float) -> None:
        try:
            with self._lock:
                self._configured_concurrency = float(value)
        except Exception:  # pragma: no cover
            pass

    @contextmanager
    def rollout_in_flight(self) -> Iterator[None]:
        self._rollouts.enter()
        try:
            yield
        finally:
            self._rollouts.exit()

    @contextmanager
    def group_in_flight(self) -> Iterator[None]:
        self._groups.enter()
        try:
            yield
        finally:
            self._groups.exit()

    # -- trainer queue ---------------------------------------------------
    def record_queue_wait(self, seconds: float, *, immediate: bool) -> None:
        try:
            with self._lock:
                self._queue_waits.append(max(0.0, float(seconds)))
                self._queue_dequeues += 1
                if immediate:
                    self._queue_immediate += 1
        except Exception:  # pragma: no cover
            pass

    # -- per-rollout timing ---------------------------------------------
    def record_rollout_timing(
        self, summary: RolloutTimingSummary | None, *, startup_offset: float = 0.0
    ) -> None:
        """Fold one rollout's timing in.

        ``startup_offset`` is the parent-side cost the rollout's own timeline
        cannot see -- with process isolation, the worker's exec-to-ready time.
        Adding it here keeps ``dequeue_to_first_model_call`` comparable between
        ``inprocess`` and ``process`` modes, which is the whole point of
        measuring it.
        """

        if summary is None:
            return
        try:
            with self._lock:
                self._call_samples.extend(summary.samples)
                self._model_calls_per_rollout.append(float(summary.model_calls))
                self._tool_total += summary.tool_seconds_total
                self._render_total += summary.render_seconds_total
                self._sample_total += summary.sample_seconds_total
                self._parse_total += summary.parse_seconds_total
                if summary.first_model_call_seconds >= 0.0:
                    self._first_call_seconds.append(
                        summary.first_model_call_seconds + max(0.0, float(startup_offset))
                    )
        except Exception:  # pragma: no cover
            pass

    def record_dequeue_to_first_model_call(self, seconds: float) -> None:
        """Startup latency measured by the parent (covers worker launch)."""

        try:
            with self._lock:
                self._first_call_seconds.append(max(0.0, float(seconds)))
        except Exception:  # pragma: no cover
            pass

    def record_worker_peak_rss(self, peak_rss_bytes: int) -> None:
        try:
            if peak_rss_bytes > 0:
                with self._lock:
                    self._worker_peak_rss.append(int(peak_rss_bytes))
        except Exception:  # pragma: no cover
            pass

    # -- live worker processes ------------------------------------------
    def register_worker(self, pid: int) -> None:
        try:
            with self._lock:
                self._worker_pids[pid] = pid
        except Exception:  # pragma: no cover
            pass

    def unregister_worker(self, pid: int) -> None:
        try:
            with self._lock:
                self._worker_pids.pop(pid, None)
        except Exception:  # pragma: no cover
            pass

    # -- export ----------------------------------------------------------
    def export(self, *, reset: bool = True, prefix: str = "rollout_client") -> dict[str, float]:
        """Aggregate the window into scalars.  Never raises."""

        try:
            return self._export(reset=reset, prefix=prefix)
        except Exception:  # pragma: no cover - a metric must not break a step
            logger.debug("client throughput monitor: export failed", exc_info=True)
            return {}

    def _export(self, *, reset: bool, prefix: str) -> dict[str, float]:
        prefix = prefix.rstrip("/")
        now = time.monotonic()
        rollout_mean, rollout_max, _rollout_now = self._rollouts.sample(reset=reset)
        group_mean, group_max, _group_now = self._groups.sample(reset=reset)

        with self._lock:
            window = max(1e-9, now - self._window_start)
            queue_waits = list(self._queue_waits)
            queue_immediate = self._queue_immediate
            queue_dequeues = self._queue_dequeues
            first_call = sorted(self._first_call_seconds)
            samples = list(self._call_samples)
            calls_per_rollout = list(self._model_calls_per_rollout)
            tool_total = self._tool_total
            render_total = self._render_total
            sample_total = self._sample_total
            parse_total = self._parse_total
            worker_peaks = list(self._worker_peak_rss)
            worker_pids = list(self._worker_pids)
            configured = self._configured_concurrency
            self_cpu_before = self._self_cpu
            children_cpu_before = self._children_cpu

            self_cpu_now = _self_cpu_seconds()
            children_cpu_now = _children_cpu_seconds()
            if reset:
                self._queue_waits.clear()
                self._queue_immediate = 0
                self._queue_dequeues = 0
                self._first_call_seconds.clear()
                self._call_samples.clear()
                self._model_calls_per_rollout.clear()
                self._tool_total = 0.0
                self._render_total = 0.0
                self._sample_total = 0.0
                self._parse_total = 0.0
                self._worker_peak_rss.clear()
                self._window_start = now
                self._self_cpu = self_cpu_now
                self._children_cpu = children_cpu_now

        metrics: dict[str, float] = {
            f"{prefix}/rollouts_in_flight_mean": rollout_mean,
            f"{prefix}/rollouts_in_flight_max": float(rollout_max),
            f"{prefix}/rollouts_in_flight_configured": configured,
            f"{prefix}/groups_in_flight_mean": group_mean,
            f"{prefix}/groups_in_flight_max": float(group_max),
            f"{prefix}/window_seconds": window,
        }
        if configured > 0:
            metrics[f"{prefix}/concurrency_achieved_fraction"] = rollout_mean / configured

        if queue_dequeues:
            metrics[f"{prefix}/trainer_queue_wait_seconds_total"] = float(sum(queue_waits))
            metrics[f"{prefix}/trainer_queue_wait_seconds_mean"] = float(sum(queue_waits) / len(queue_waits))
            metrics[f"{prefix}/trainer_queue_wait_seconds_max"] = float(max(queue_waits))
            metrics[f"{prefix}/trainer_queue_ready_fraction"] = queue_immediate / float(queue_dequeues)
            metrics[f"{prefix}/trainer_queue_dequeues"] = float(queue_dequeues)

        if samples:
            gaps = sorted(call[0] for call in samples)
            clients = sorted(call[1] + call[3] for call in samples)
            models = sorted(call[2] for call in samples)
            steps = sorted(call[0] + call[1] + call[2] + call[3] for call in samples)
            for name, values in (
                ("agent_step_seconds", steps),
                ("agent_step_tool_seconds", gaps),
                ("agent_step_model_seconds", models),
                ("agent_step_client_seconds", clients),
            ):
                metrics[f"{prefix}/{name}_p50"] = _percentile(values, 0.5)
                metrics[f"{prefix}/{name}_p90"] = _percentile(values, 0.9)
            metrics[f"{prefix}/agent_step_samples"] = float(len(samples))

        client_total = render_total + parse_total
        accounted = tool_total + client_total + sample_total
        if accounted > 0:
            metrics[f"{prefix}/agent_step_model_fraction"] = sample_total / accounted
            metrics[f"{prefix}/agent_step_client_fraction"] = client_total / accounted
            metrics[f"{prefix}/agent_step_tool_fraction"] = tool_total / accounted
            metrics[f"{prefix}/agent_step_seconds_accounted_total"] = accounted
            metrics[f"{prefix}/agent_step_model_seconds_total"] = sample_total
            metrics[f"{prefix}/agent_step_client_seconds_total"] = client_total
            metrics[f"{prefix}/agent_step_tool_seconds_total"] = tool_total

        if calls_per_rollout:
            metrics[f"{prefix}/model_calls_per_rollout_mean"] = float(
                sum(calls_per_rollout) / len(calls_per_rollout)
            )

        if first_call:
            metrics[f"{prefix}/dequeue_to_first_model_call_seconds_p50"] = _percentile(first_call, 0.5)
            metrics[f"{prefix}/dequeue_to_first_model_call_seconds_p90"] = _percentile(first_call, 0.9)

        self_percent = max(0.0, (self_cpu_now - self_cpu_before)) / window * 100.0
        children_percent = max(0.0, (children_cpu_now - children_cpu_before)) / window * 100.0
        cores = float(os.cpu_count() or 1)
        metrics[f"{prefix}/cpu_percent"] = self_percent
        metrics[f"{prefix}/worker_cpu_percent"] = children_percent
        metrics[f"{prefix}/total_cpu_percent"] = self_percent + children_percent
        metrics[f"{prefix}/cpu_cores_busy"] = (self_percent + children_percent) / 100.0
        metrics[f"{prefix}/cpu_saturation"] = (self_percent + children_percent) / (100.0 * cores)
        metrics[f"{prefix}/rss_bytes"] = float(read_rss_bytes())
        metrics[f"{prefix}/threads"] = float(read_thread_count())
        if worker_pids:
            metrics[f"{prefix}/worker_rss_bytes"] = float(sum(read_rss_bytes(pid) for pid in worker_pids))
            metrics[f"{prefix}/worker_processes_live"] = float(len(worker_pids))
        if worker_peaks:
            metrics[f"{prefix}/worker_peak_rss_bytes_sum"] = float(sum(worker_peaks))
            metrics[f"{prefix}/worker_peak_rss_bytes_max"] = float(max(worker_peaks))
        return metrics


_MONITORS: dict[str, ClientThroughputMonitor] = {}
_MONITOR_LOCK = threading.Lock()


def monitor(scope: str = "train") -> ClientThroughputMonitor:
    """The process-wide monitor for a stats scope (``train`` / ``eval``)."""

    with _MONITOR_LOCK:
        existing = _MONITORS.get(scope)
        if existing is None:
            existing = ClientThroughputMonitor(scope)
            _MONITORS[scope] = existing
        return existing


def step_metrics(scope: str = "train", *, reset: bool = True) -> dict[str, float]:
    """Scalars for one training step, ready to merge into ``step_stats``.

    Never raises: this is called inline in the train loop's stats path.
    """

    try:
        return monitor(scope).export(reset=reset)
    except Exception:  # pragma: no cover
        return {}


def reset_monitors_for_tests() -> None:
    """Drop every monitor; test-only."""

    with _MONITOR_LOCK:
        _MONITORS.clear()
