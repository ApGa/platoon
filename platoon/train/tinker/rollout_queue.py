"""Bounded rollout admission with optional capacity limits on dataset groups."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, deque
from collections.abc import Iterable

from platoon.train.labels import task_label

logger = logging.getLogger(__name__)

# How many rows of a label at its limit may wait, as a multiple of the queue's bound (a batch).
BACKLOG_BATCHES = 4


class RolloutTaskQueue:
    """Reserve capacity when a worker takes a task, before its model is bound.

    A task's group is its label: its value of the ``key`` column as text (:func:`platoon.train.labels.task_label`),
    so that ``limits`` keyed on labels apply to any column's values.

    A blocked environment does not occupy a worker or block other pending
    environments. Rows retain FIFO order within each environment, and the
    bounded pending queue applies backpressure without dropping the rows a worker
    can take. Capacity covers the whole workflow, including environment teardown.

    A label at its limit never stalls the others:

    - **Parked rows do not count.** The bound (``maxsize``) counts the pending rows that a worker could take: a row of
      a label at its limit waits, parked, beside it (:meth:`parked`).
    - **A full backlog skips.** At most ``backlog_limit`` rows (by default :data:`BACKLOG_BATCHES` times ``maxsize``)
      of a label with a limit wait at once; ``put`` skips a new row of a label whose backlog is full and returns
      False, counts it in ``skipped`` and warns once per label. The other labels' rows keep the batches full.
    - **Never while nothing can start.** ``put`` waits, rather than skip, while no label of the run has room: every
      label it was given (``labels``, the labels of the run's tasks) and every label it has queued is at its limit.
      A label without a limit always has room, so a skip always leaves a slot that a row of a label with room can
      use, and a row is never skipped only to wait.
    """

    def __init__(
        self,
        maxsize: int,
        *,
        key: str,
        limits: dict[str, int],
        backlog_limit: int | None = None,
        labels: Iterable[str | None] = (),
    ):
        if maxsize < 1:
            raise ValueError("rollout queue maxsize must be positive")
        if not key or any(
            not isinstance(name, str) or not name or isinstance(n, bool) or not isinstance(n, int) or n < 1
            for name, n in limits.items()
        ):
            raise ValueError("rollout group limits require nonempty keys and positive integer capacities")
        if backlog_limit is not None and (
            isinstance(backlog_limit, bool) or not isinstance(backlog_limit, int) or backlog_limit < 1
        ):
            raise ValueError("rollout queue backlog_limit must be a positive integer")
        self._maxsize = maxsize
        self._key = key
        self._limits = dict(limits)
        self._backlog_limit = backlog_limit if backlog_limit is not None else BACKLOG_BATCHES * maxsize
        self._pending: deque[dict] = deque()
        self._active: Counter[str] = Counter()
        # The labels of the run's tasks, and of every row queued since (None for a row without one).
        self._labels: set[str | None] = set(labels)
        # Rows of each label left out because its backlog was full.
        self.skipped: Counter[str] = Counter()
        self._changed = asyncio.Condition()

    def _label(self, item: dict) -> str | None:
        return task_label(item, self._key)

    def _blocked(self, label: str | None) -> bool:
        """Whether a row of ``label`` must wait: its label has a limit, and as many groups of it run as it allows."""

        return label in self._limits and self._active[label] >= self._limits[label]

    def _takeable(self) -> int:
        """The pending rows a worker could take now: those of labels below their limits."""

        return sum(1 for item in self._pending if not self._blocked(self._label(item)))

    def _waiting(self, label: str | None) -> int:
        return sum(1 for item in self._pending if self._label(item) == label)

    def _some_label_has_room(self) -> bool:
        """Whether a row of some label of the run could start now: one without a limit, or below its limit."""

        return any(not self._blocked(label) for label in self._labels)

    def parked(self) -> int:
        """The pending rows that wait for their label's limit."""

        return len(self._pending) - self._takeable()

    def parked_by_label(self) -> Counter[str | None]:
        """:meth:`parked`, by label."""

        return Counter(label for item in self._pending if self._blocked(label := self._label(item)))

    async def changed(self) -> None:
        """Wait until a row is queued, taken or released."""

        async with self._changed:
            await self._changed.wait()

    async def put(self, item: dict) -> bool:
        """Queue ``item``, waiting while the rows a worker could take fill the bound; False when its label's backlog
        is full and the row is skipped (:attr:`skipped`)."""

        label = self._label(item)
        async with self._changed:
            self._labels.add(label)
            while True:
                if self._blocked(label):
                    if self._waiting(label) < self._backlog_limit:
                        self._pending.append(item)  # parked beside the bound until its label has room
                        self._changed.notify_all()
                        return True
                    if not self._some_label_has_room():
                        await self._changed.wait()  # nothing can start: wait for a release rather than skip
                        continue
                    self.skipped[label] += 1
                    if self.skipped[label] == 1:
                        logger.warning(
                            "Label %r has %d rows waiting for its limit (train.rollout_group_limits: %d at once): its "
                            "new rows are skipped, so that the other labels' rows keep the batches full; its share of "
                            "the order asks for more groups than its limit runs",
                            label,
                            self._backlog_limit,
                            self._limits[label],
                        )
                    return False
                if self._takeable() < self._maxsize:
                    self._pending.append(item)
                    self._changed.notify_all()
                    return True
                await self._changed.wait()

    async def get(self) -> dict:
        async with self._changed:
            while True:
                for index, item in enumerate(self._pending):
                    group = self._label(item)
                    if group not in self._limits or self._active[group] < self._limits[group]:
                        del self._pending[index]
                        if group in self._limits:
                            self._active[group] += 1
                        self._changed.notify_all()
                        return item
                await self._changed.wait()

    async def release(self, item: dict) -> None:
        group = self._label(item)
        if group not in self._limits:
            return
        async with self._changed:
            if self._active[group] <= 0:
                raise RuntimeError(f"rollout group capacity released without admission: {group}")
            self._active[group] -= 1
            self._changed.notify_all()
