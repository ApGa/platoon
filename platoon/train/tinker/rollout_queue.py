"""Bounded rollout admission with optional capacity limits on dataset groups."""

from __future__ import annotations

import asyncio
from collections import Counter, deque


class RolloutTaskQueue:
    """Reserve capacity when a worker takes a task, before its model is bound.

    A blocked environment does not occupy a worker or block other pending
    environments. Rows retain FIFO order within each environment, and the
    bounded pending queue applies backpressure without dropping dataset rows.
    Capacity covers the whole workflow, including environment teardown.
    """

    def __init__(self, maxsize: int, *, key: str, limits: dict[str, int]):
        if maxsize < 1:
            raise ValueError("rollout queue maxsize must be positive")
        if not key or any(
            not isinstance(name, str) or not name or isinstance(n, bool) or not isinstance(n, int) or n < 1
            for name, n in limits.items()
        ):
            raise ValueError("rollout group limits require nonempty keys and positive integer capacities")
        self._maxsize = maxsize
        self._key = key
        self._limits = dict(limits)
        self._pending: deque[dict] = deque()
        self._active: Counter[str] = Counter()
        self._changed = asyncio.Condition()

    async def put(self, item: dict) -> None:
        async with self._changed:
            await self._changed.wait_for(lambda: len(self._pending) < self._maxsize)
            self._pending.append(item)
            self._changed.notify_all()

    async def get(self) -> dict:
        async with self._changed:
            while True:
                for index, item in enumerate(self._pending):
                    group = item.get(self._key)
                    if group not in self._limits or self._active[group] < self._limits[group]:
                        del self._pending[index]
                        if group in self._limits:
                            self._active[group] += 1
                        self._changed.notify_all()
                        return item
                await self._changed.wait()

    async def release(self, item: dict) -> None:
        group = item.get(self._key)
        if group not in self._limits:
            return
        async with self._changed:
            if self._active[group] <= 0:
                raise RuntimeError(f"rollout group capacity released without admission: {group}")
            self._active[group] -= 1
            self._changed.notify_all()
