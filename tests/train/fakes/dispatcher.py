"""The dispatcher on its own (``platoon.train.dispatch``), as its tests drive it: the rows it reads, what it tells the
run's statistics, a run's samplers over a mixture of two entries, and a way to let its tasks run until nothing changes.

- :class:`LabelledRows`: rows without an entry, labelled by a column, each its group's ``size``; :class:`EntryRows`:
  rows by entry, each entry's groups of a size of its own;
- :class:`Recorder`: what the dispatcher tells the run's statistics of its draws and skips;
- :func:`two_entries`: the task sampling of entries ``a`` and ``b`` of four tasks each, ``a``'s sampler adaptive
  (:class:`Chooser`) unless the test names another;
- :func:`settle`: let every task run until nothing changes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.dispatch.permits import PermitPool
from platoon.train.labels import task_label
from platoon.train.sampler_api import Draw, Sampler
from platoon.train.sampler_runtime import TaskPool, TaskSampling
from platoon.train.samplers import SamplerChoice, UniformSampler, WeightedSampler
from platoon.train.task_stats import EntryInfo


class LabelledRows:
    """What the dispatcher reads of a row: no entry, its ``env`` label, and its ``size`` (default 1), with ``permits``
    rollout permits to share."""

    def __init__(self, permits: int = 1000, column: str = "env") -> None:
        self.permits = PermitPool(permits)
        self.column = column

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        return None, task_label(data, self.column), int(data.get("size", 1))


class EntryRows:
    """Rows by entry (their ENVIRONMENT_COLUMN), labelled by ``column``, each entry's group ``sizes``."""

    def __init__(self, permits: int, sizes: Mapping[str, int], column: str = ENVIRONMENT_COLUMN) -> None:
        self.permits = PermitPool(permits)
        self.sizes = dict(sizes)
        self.column = column

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        entry = data.get(ENVIRONMENT_COLUMN)
        return entry, task_label(data, self.column), self.sizes.get(entry, 1)


class Recorder:
    """Records what the dispatcher tells the run's statistics."""

    def __init__(self) -> None:
        self.draws: list[dict[str, Any]] = []
        self.skips: list[int] = []

    def draw(self, row, *, seq, position, version, reissues=None, sampler=None, reason=None, score=None) -> None:
        self.draws.append({"seq": seq, "position": position, "version": version, "reissues": reissues})

    def skipped(self, row) -> None:
        self.skips.append(row[POSITION_COLUMN])


class Chooser(Sampler):
    """An adaptive sampler: of its pool's eligible tasks, the one at its decision index, in turn; it counts the
    decisions it was asked and records whether its context's statistics were in view."""

    adaptive = True

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.asked: list[int] = []

    def next(self, ctx):
        assert ctx.stats is not None
        self.asked.append(ctx.decision)
        eligible = [task for task in ctx.task_ids if ctx.eligible(task)]
        if not eligible:
            return None
        return Draw(eligible[ctx.decision % len(eligible)], "chosen", 0.5)


def two_entries(
    *,
    a: Any = Chooser,
    b: Any = UniformSampler,
    run: Any = WeightedSampler,
    sizes: Mapping[str, int] | None = None,
    kinds: bool = False,
    weights: Mapping[str, float] | None = None,
) -> TaskSampling:
    """Entries a and b, four tasks each, weighted one to one (or by ``weights``); a's sampler adaptive by default.  With
    ``kinds``, a's tasks carry two values of the column ``kind``."""

    sizes = dict(sizes or {"a": 1, "b": 1})
    weights = dict(weights or {"a": 1.0, "b": 1.0})
    rows = []
    for entry in ("a", "b"):
        for index in range(4):
            row = {"task_id": f"{entry}::{index}", ENVIRONMENT_COLUMN: entry}
            if kinds:
                row["kind"] = ("x" if index % 2 else "y") if entry == "a" else "z"
            rows.append(row)
    column = "kind" if kinds else ENVIRONMENT_COLUMN
    pool = TaskPool.from_rows(rows, entries=["a", "b"], column=column)
    infos = {
        entry: EntryInfo(
            group_size=sizes[entry],
            name=entry,
            weight=weights[entry],
            seed=7,
            index=index,
            task_ids=pool.task_ids[4 * index : 4 * index + 4],
        )
        for index, entry in enumerate(("a", "b"))
    }

    def choice(cls: Any) -> SamplerChoice:
        name = {UniformSampler: "uniform", WeightedSampler: "weighted"}.get(cls, cls.__name__.lower())
        return SamplerChoice(name, cls)

    return TaskSampling(
        pool=pool,
        run=choice(run),
        entries={"a": choice(a), "b": choice(b)} if getattr(run, "entry_slots", False) else {},
        infos=infos,
        epoch_size=8,
    )


async def settle() -> None:
    """Let every task run until nothing changes."""

    for _ in range(50):
        await asyncio.sleep(0)
