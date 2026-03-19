from __future__ import annotations

import asyncio
import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

_WRITE_LOCK = threading.Lock()
_SPAN_STACK: contextvars.ContextVar[tuple["_SpanState", ...]] = contextvars.ContextVar(
    "platoon_span_profile_stack",
    default=(),
)


def enabled() -> bool:
    return os.getenv("PLATOON_PROFILE_SPANS", "").lower() in {"1", "true", "yes", "on"}


def output_path() -> str:
    return os.getenv("PLATOON_PROFILE_SPANS_PATH", "/tmp/platoon_span_profile.jsonl")


@dataclass
class _SpanState:
    span_id: str
    name: str
    started_at: float
    parent_span_id: str | None
    metadata: dict[str, Any]
    child_time_ms: float = 0.0


def _append_record(record: dict[str, Any]) -> None:
    path = output_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _WRITE_LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


@asynccontextmanager
async def profile_span(name: str, metadata: dict[str, Any] | None = None) -> AsyncIterator[str | None]:
    """Record a hierarchical async wall-time span when profiling is enabled."""
    if not enabled():
        yield None
        return

    stack = _SPAN_STACK.get()
    parent = stack[-1] if stack else None
    state = _SpanState(
        span_id=str(uuid.uuid4()),
        name=name,
        started_at=time.perf_counter(),
        parent_span_id=parent.span_id if parent is not None else None,
        metadata=dict(metadata or {}),
    )
    token = _SPAN_STACK.set((*stack, state))
    try:
        yield state.span_id
    finally:
        ended_at = time.perf_counter()
        inclusive_ms = (ended_at - state.started_at) * 1000.0
        self_ms = max(0.0, inclusive_ms - state.child_time_ms)
        _SPAN_STACK.reset(token)
        parent_stack = _SPAN_STACK.get()
        if parent_stack:
            parent_stack[-1].child_time_ms += inclusive_ms

        task = asyncio.current_task()
        _append_record(
            {
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "inclusive_ms": round(inclusive_ms, 3),
                "metadata": state.metadata,
                "name": state.name,
                "parent_span_id": state.parent_span_id,
                "pid": os.getpid(),
                "self_ms": round(self_ms, 3),
                "span_id": state.span_id,
                "task_name": task.get_name() if task is not None else None,
            }
        )
