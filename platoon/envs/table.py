"""Task tables: tasks with no tools and no grader, for harnesses that grade themselves.

``task_table(rows, splits=...)`` serves plain rows as an environment service:
a session's instruction is the row's goal, it has no env tools, and its
``grade()`` is ``invalid`` (``reason="self_grade"``), so nothing trains unless
the episode program sets the outcome itself. In config, ``service: table`` with
``service_kwargs: {rows: [...]}`` or ``{path: tasks.jsonl}`` builds the same.

A row is a goal string or a mapping: ``id`` (or ``task_id``; default
``task-<n>``), ``goal`` (or ``instruction``/``prompt``), optional ``split``
and ``max_steps``; every other key goes to ``Task.misc``. Splits come from
``splits={split: [ids]}``, else from the rows' ``split``, else every row is in
``train``.

Use ``path`` for anything beyond a handful of rows: a config entry's
``service_kwargs`` travel with every one of its tasks (in the task's route), so
inline ``rows`` are copied into every task, and each rollout builds its own
service. A ``path`` table is read once per process, off the event loop, and
shared by every service that names the same unchanged file.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import dataclasses
import json
import logging
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from platoon.envs.service import ForkMode, Outcome, ToolResult, ToolSpec
from platoon.task import Task

logger = logging.getLogger(__name__)

_GOAL_KEYS = ("goal", "instruction", "prompt")
_RESERVED_KEYS = {"id", "task_id", "split", "max_steps", *_GOAL_KEYS}
# More inline rows than this get a warning (once per process): they belong in a JSONL file.
INLINE_ROWS_WARNING = 20


@dataclass(frozen=True)
class _Rows:
    """A parsed table: each task's row by id, and the splits its rows name.

    A file's rows stay JSON text until a task is asked for: kept as parsed objects, a large table would add
    hundreds of thousands of objects for every garbage collection to walk while holding the GIL.
    """

    entries: dict[str, tuple[int, Any]]  # task id -> (row number, row or its JSON text)
    splits: dict[str, list[str]]
    encoded: bool  # the rows are JSON text

    def task(self, task_id: str) -> Task:
        """A new ``Task`` for ``task_id``, the caller's own: nothing in it is shared with the table."""

        number, row = self.entries[task_id]
        return _task(json.loads(row) if self.encoded else copy.deepcopy(row), number)[0]


class TaskTableService:
    """An environment service over a fixed table of tasks (see the module docstring)."""

    # The JSONL file a config file names relative to itself (RFC 0018, section 5).
    path_options: ClassVar[tuple[str, ...]] = ("path",)
    # The tools its sessions serve: none (platoon.envs.service.service_declared_tools).
    declared_tools: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any] | str] | None = None,
        *,
        splits: Mapping[str, Sequence[str]] | None = None,
        path: str | Path | None = None,
        name: str = "table",
    ) -> None:
        if (rows is None) == (path is None):
            raise ValueError("A task table needs rows or a JSONL path, not both")
        self.name = name
        self._path = Path(path) if path is not None else None
        self._explicit_splits = {split: list(ids) for split, ids in splits.items()} if splits is not None else None
        self._rows: _Rows | None = None
        self._splits: dict[str, list[str]] = {}
        if rows is not None:
            rows = list(rows)
            if len(rows) > INLINE_ROWS_WARNING:
                _warn_inline_rows(name, len(rows))
            self._use(_parse_rows(rows, f"Task table {name!r}"))

    async def list_splits(self) -> list[str]:
        await self._load()
        return list(self._splits)

    async def list_tasks(self, split: str) -> list[str]:
        await self._load()
        if split not in self._splits:
            raise ValueError(f"Task table {self.name!r} has no split {split!r}; its splits are {list(self._splits)}")
        return list(self._splits[split])

    async def get_task(self, task_id: str) -> Task:
        rows = await self._load()
        if task_id not in rows.entries:
            raise ValueError(f"Task table {self.name!r} has no task {task_id!r}")
        return rows.task(task_id)

    async def open(self, task: Task, *, rollout_id: str) -> TableSession:
        rows = await self._load()
        if task.goal is None and task.id in rows.entries:
            task = dataclasses.replace(task, goal=rows.task(task.id).goal)
        return TableSession(self, task, rollout_id=rollout_id)

    async def _load(self) -> _Rows:
        if self._rows is None:
            assert self._path is not None
            self._use(await load_table(self._path))
        assert self._rows is not None
        return self._rows

    def _use(self, rows: _Rows) -> None:
        if self._explicit_splits is not None:
            unknown = sorted({task_id for ids in self._explicit_splits.values() for task_id in ids} - set(rows.entries))
            if unknown:
                raise ValueError(f"Task table {self.name!r} splits name unknown task ids {unknown[:5]}")
            self._splits = self._explicit_splits
        else:
            self._splits = rows.splits
        self._rows = rows


class TableSession:
    """A session over one table row: its goal and nothing else."""

    def __init__(self, service: TaskTableService, task: Task, *, rollout_id: str) -> None:
        self.task = task
        self.instruction = task.goal or ""
        self.sandbox = None
        self.rollout_id = rollout_id
        self._service = service

    async def tools(self) -> list[ToolSpec]:
        return []

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        return ToolResult.text(f"Task table {self._service.name!r} has no tools (called {name!r})", is_error=True)

    async def grade(self) -> Outcome:
        return Outcome.infra("no_grader", status="invalid", reason="self_grade")

    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> TableSession:
        if mode not in ("shared", "copy", "fresh"):
            raise ValueError(f"Unknown fork mode {mode!r}")
        return TableSession(self._service, task or self.task, rollout_id=self.rollout_id)

    async def close(self) -> None:
        return None


def task_table(
    rows: Iterable[Mapping[str, Any] | str], *, splits: Mapping[str, Sequence[str]] | None = None, name: str = "table"
) -> TaskTableService:
    """An :class:`~platoon.envs.service.EnvironmentService` with tasks and no tools (see the module docstring)."""

    return TaskTableService(rows, splits=splits, name=name)


# The tables read from files, by (real path, mtime in ns, size): each file is parsed once per process and every
# service that names it shares the parse, which runs on its own thread while callers (on any loop) await it.
_tables: dict[tuple[str, int, int], concurrent.futures.Future[_Rows]] = {}
# The parses running, by the path callers named: a caller joins one without a syscall, since a syscall's return
# waits for the GIL that the parsing thread holds.
_parsing: dict[str, concurrent.futures.Future[_Rows]] = {}
_tables_lock = threading.Lock()


async def load_table(path: str | Path) -> _Rows:
    """The parsed table in JSONL file ``path``, shared by every caller in this process until the file changes."""

    with _tables_lock:
        parsed = _parsing.get(str(path))
    if parsed is None:
        real = os.path.realpath(path)
        stat = os.stat(real)
        key = (real, stat.st_mtime_ns, stat.st_size)
        with _tables_lock:
            parsed = _tables.get(key)
            if parsed is None:
                for stale in [other for other in _tables if other[0] == real]:
                    del _tables[stale]  # an earlier version of the file
                parsed = concurrent.futures.Future()
                parsed.set_running_or_notify_cancel()  # nobody can cancel it: every caller shares it
                _tables[key] = _parsing[str(path)] = parsed
                thread = threading.Thread(
                    target=_parse_into, args=(parsed, key, str(path)), name="platoon-task-table", daemon=True
                )
                thread.start()
    return await asyncio.wrap_future(parsed)


def _parse_into(parsed: concurrent.futures.Future[_Rows], key: tuple[str, int, int], named: str) -> None:
    try:
        rows = _parse_file(key[0])
    except BaseException as exc:  # noqa: BLE001 - delivered to every caller
        with _tables_lock:
            del _parsing[named]
            if _tables.get(key) is parsed:
                del _tables[key]  # the next caller tries again
        parsed.set_exception(exc)
    else:
        with _tables_lock:
            del _parsing[named]
        parsed.set_result(rows)


def _parse_file(path: str) -> _Rows:
    # One read, then parse from memory. Read line by line, the file releases the GIL at every buffered chunk, and
    # each release wakes the event loop's thread too early for it to force a switch, so it waited out the whole
    # parse. Split on "\n" only: splitlines() also splits on characters such as U+2028 inside JSON strings.
    text = Path(path).read_text(encoding="utf-8")
    return _index_rows(
        ((line, json.loads(line)) for line in text.split("\n") if line.strip()), f"Task table {path}", True
    )


def _parse_rows(rows: Iterable[Mapping[str, Any] | str], what: str) -> _Rows:
    return _index_rows(((row, row) for row in rows), what, False)


def _index_rows(rows: Iterable[tuple[Any, Any]], what: str, encoded: bool) -> _Rows:
    """Index ``(stored, parsed)`` rows by task id, checking every row once; only ``stored`` is kept."""

    entries: dict[str, tuple[int, Any]] = {}
    splits: dict[str, list[str]] = {}
    for number, (stored, row) in enumerate(rows):
        task, split = _task(row, number)
        task_id = str(task.id)
        if task_id in entries:
            raise ValueError(f"{what} has two rows with id {task_id!r}")
        entries[task_id] = (number, stored)
        splits.setdefault(split or "train", []).append(task_id)
    return _Rows(entries=entries, splits=splits, encoded=encoded)


_warned_inline_rows = False


def _warn_inline_rows(name: str, count: int) -> None:
    global _warned_inline_rows
    if _warned_inline_rows:
        return
    _warned_inline_rows = True
    logger.warning(
        "Task table %r has %d inline rows; inline rows are copied into every task of a config entry, so put them "
        "in a JSONL file and name it with path instead",
        name,
        count,
    )


def _task(row: Mapping[str, Any] | str, number: int) -> tuple[Task, str | None]:
    if isinstance(row, str):
        return Task(id=f"task-{number}", goal=row), None
    if not isinstance(row, Mapping):
        raise TypeError(f"A task table row is a goal string or a mapping, got {type(row).__name__}")
    task_id = row.get("id", row.get("task_id", f"task-{number}"))
    goal = next((row[key] for key in _GOAL_KEYS if row.get(key) is not None), None)
    split = row.get("split")
    misc = {key: value for key, value in row.items() if key not in _RESERVED_KEYS}
    return Task(id=str(task_id), goal=goal, max_steps=row.get("max_steps"), misc=misc), split
