from __future__ import annotations

import asyncio
import linecache
import logging
import os
import sys
import threading
import time
import traceback
import weakref
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.getenv("PLATOON_DEBUG_HANGS", "").lower() in {"1", "true", "yes", "on"}


def _threshold_sec() -> float:
    value = os.getenv("PLATOON_DEBUG_HANG_THRESHOLD_SEC", "60")
    try:
        return max(float(value), 1.0)
    except ValueError:
        return 60.0


def _interval_sec() -> float:
    value = os.getenv("PLATOON_DEBUG_HANG_INTERVAL_SEC", "15")
    try:
        return max(float(value), 1.0)
    except ValueError:
        return 15.0


def _max_tasks() -> int:
    value = os.getenv("PLATOON_DEBUG_HANG_MAX_TASKS", "3")
    try:
        return max(int(value), 1)
    except ValueError:
        return 3


def _max_frames() -> int:
    value = os.getenv("PLATOON_DEBUG_HANG_MAX_FRAMES", "8")
    try:
        return max(int(value), 1)
    except ValueError:
        return 8


def _source_context_lines() -> int:
    value = os.getenv("PLATOON_DEBUG_HANG_SOURCE_CONTEXT", "12")
    try:
        return max(int(value), 0)
    except ValueError:
        return 12


@dataclass
class _TrackedTask:
    request_id: str
    kind: str
    task_ref: weakref.ReferenceType[asyncio.Task[Any]]
    thread_ident: int | None
    started_at: float
    metadata: dict[str, Any] = field(default_factory=dict)
    last_dump_at: float = 0.0


_TRACKED: dict[str, _TrackedTask] = {}
_TRACKED_LOCK = threading.Lock()
_WATCHDOG_THREAD: threading.Thread | None = None
_STOP_EVENT = threading.Event()


def track_current_task(request_id: str, kind: str, metadata: dict[str, Any] | None = None) -> None:
    if not enabled():
        return

    task = asyncio.current_task()
    if task is None:
        return

    with _TRACKED_LOCK:
        _TRACKED[request_id] = _TrackedTask(
            request_id=request_id,
            kind=kind,
            task_ref=weakref.ref(task),
            thread_ident=threading.get_ident(),
            started_at=time.monotonic(),
            metadata=dict(metadata or {}),
        )
        _ensure_watchdog_thread()


def untrack(request_id: str) -> None:
    if not enabled():
        return

    with _TRACKED_LOCK:
        _TRACKED.pop(request_id, None)


def _ensure_watchdog_thread() -> None:
    global _WATCHDOG_THREAD
    if _WATCHDOG_THREAD is not None and _WATCHDOG_THREAD.is_alive():
        return
    _STOP_EVENT.clear()
    _WATCHDOG_THREAD = threading.Thread(
        target=_watchdog_loop,
        name="platoon-hang-watchdog",
        daemon=True,
    )
    _WATCHDOG_THREAD.start()


def _watchdog_loop() -> None:
    while True:
        if _STOP_EVENT.wait(_interval_sec()):
            return

        with _TRACKED_LOCK:
            tracked = dict(_TRACKED)
        if not tracked:
            _STOP_EVENT.set()
            return

        now = time.monotonic()
        stale: list[_TrackedTask] = []
        expired_ids: list[str] = []
        for request_id, record in tracked.items():
            task = record.task_ref()
            if task is None or task.done():
                expired_ids.append(request_id)
                continue
            if now - record.started_at >= _threshold_sec():
                stale.append(record)

        if expired_ids:
            with _TRACKED_LOCK:
                for request_id in expired_ids:
                    current = _TRACKED.get(request_id)
                    if current is not None and (current.task_ref() is None or current.task_ref().done()):
                        _TRACKED.pop(request_id, None)

        stale.sort(key=lambda record: record.started_at)
        for record in stale[: _max_tasks()]:
            if now - record.last_dump_at < _interval_sec() * 0.9:
                continue
            _log_task_stack(record, now)
            with _TRACKED_LOCK:
                current = _TRACKED.get(record.request_id)
                if current is not None:
                    current.last_dump_at = now


def _log_task_stack(record: _TrackedTask, now: float) -> None:
    task = record.task_ref()
    if task is None:
        return

    formatted_stack: list[str] = []
    source_context = ""
    if record.thread_ident is not None:
        frame = sys._current_frames().get(record.thread_ident)
        if frame is not None:
            formatted_stack = traceback.format_stack(frame, limit=_max_frames())
            source_context = _format_frame_source_context(frame)

    waiter = getattr(task, "_fut_waiter", None)
    logger.warning(
        "hang_watchdog kind=%s request_id=%s age_s=%.1f thread_ident=%s task_name=%s task_coro=%r waiter=%r metadata=%s\n%s%s",
        record.kind,
        record.request_id,
        now - record.started_at,
        record.thread_ident,
        task.get_name(),
        task.get_coro(),
        waiter,
        record.metadata,
        "".join(formatted_stack).rstrip() or "<no python frames available>",
        source_context,
    )


def _format_frame_source_context(frame: Any) -> str:
    """Format a small source window around the currently executing frame."""
    try:
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
    except Exception:
        return ""

    context = _source_context_lines()
    if context <= 0 or not filename:
        return ""

    lines = linecache.getlines(filename)
    if not lines:
        return ""

    start = max(1, lineno - context // 2)
    end = min(len(lines), lineno + context // 2)
    rendered: list[str] = ["\n--- source context ---\n"]
    for line_no in range(start, end + 1):
        marker = ">" if line_no == lineno else " "
        rendered.append(f"{marker} {line_no:04d}: {lines[line_no - 1].rstrip()}\n")
    return "".join(rendered)
