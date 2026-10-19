"""Leave tasks out of an environment entry's task lists: ``environments[].exclude_tasks``.

An item names a task by its id, the form the entry's task list holds (an ORS environment's ``<split>:<index>``, a
Harbor task's directory name, a task table's row id), or by its name when the service names its tasks
(``list_task_names``, which ORS environments have).  :func:`excluded_service_tasks` matches the items against the
tasks a service listed and refuses the items that name no task of the service: none of the listed splits, nor of
its other splits (``list_splits``), so that a misspelled name cannot leave a task in the run.
:func:`exclude_loaded_tasks` drops them from what a dataset loader returned.  That list is all platoon sees of a
loader's tasks (the loader applies its own splits and ``limit``), so there an item that matches none of them is
only logged.

What each list lost is a :class:`TaskExclusion`; the commands add them to the run's ``lock.json`` with
:func:`record_exclusions`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from platoon.train.components import TASK_ID_SEPARATOR
from platoon.utils.config import ConfigError, update_run_lock

logger = logging.getLogger(__name__)

# The ``lock.json`` key under which a run records what ``exclude_tasks`` left out of its task lists.
LOCK_KEY = "excluded_tasks"


@dataclass(frozen=True)
class TaskExclusion:
    """What an entry's ``exclude_tasks`` left out of one of its task lists."""

    # The entry's label (its name, or ``default``).
    environment: str
    # The split the run listed: ``train``, ``eval``, or an evaluation config's ``split``.
    split: str
    # How many tasks the list held before the exclusion (before any ``limit``).
    listed: int
    # The ids of the tasks left out, in listed order.
    excluded: tuple[str, ...]


class UnknownTasks(ValueError):
    """``exclude_tasks`` items that name no task of the environment."""

    def __init__(self, items: Sequence[str], *, examples: Sequence[str] = ()) -> None:
        self.items = list(items)
        message = f"{self.items} match no task id or name of the environment"
        if examples:
            message += f" (its tasks have ids and names like {' and '.join(map(repr, examples))})"
        if any(TASK_ID_SEPARATOR in item for item in self.items):
            message += f"; an entry names its own tasks, without the mixture's '<name>{TASK_ID_SEPARATOR}' prefix"
        super().__init__(message)


async def excluded_service_tasks(
    service: Any, splits: Sequence[str], task_ids: Sequence[str], items: Sequence[str]
) -> set[str]:
    """The ids among ``task_ids``, which ``service`` listed for ``splits``, that ``items`` name by id or by name.

    Raises :class:`UnknownTasks` for the items that name no task of the service: an item may name a task the
    listing left out (a task of another split, or one that the service's own task selection drops), but not one the
    service does not have.  The service's other splits are listed only when an item matches none of ``splits``.
    """

    wanted = set(items)
    names = await _task_names(service, splits)
    excluded = {task_id for task_id in task_ids if task_id in wanted or names.get(task_id) in wanted}
    unknown = wanted - set(task_ids) - names.keys() - set(names.values())
    if unknown:
        for split in await _other_splits(service, splits):
            unknown -= await _split_identifiers(service, split)
            if not unknown:
                break
    if unknown:
        examples = [task_ids[0], names.get(task_ids[0])] if task_ids else []
        raise UnknownTasks(
            [item for item in items if item in unknown], examples=list(dict.fromkeys(e for e in examples if e))
        )
    return excluded


async def _task_names(service: Any, splits: Sequence[str]) -> dict[str, str]:
    """``{task id: name}`` of every task of ``splits``, from the service's ``list_task_names``; empty for a service
    that does not name its tasks apart from their ids."""

    list_task_names = getattr(service, "list_task_names", None)
    if not callable(list_task_names):
        return {}
    names: dict[str, str] = {}
    for split in splits:
        names.update(await list_task_names(split))
    return names


async def _other_splits(service: Any, splits: Sequence[str]) -> list[str]:
    """The service's splits other than ``splits``, from its ``list_splits``; none for a service without one."""

    list_splits = getattr(service, "list_splits", None)
    if not callable(list_splits):
        return []
    names = [str(getattr(split, "name", split)) for split in await list_splits()]
    return [name for name in names if name not in splits]


async def _split_identifiers(service: Any, split: str) -> set[str]:
    """Every id and name of the tasks of ``split``."""

    names = await _task_names(service, [split])
    if names:
        return set(names) | set(names.values())
    return set(await service.list_tasks(split))


def exclude_loaded_tasks(result: Any, items: Sequence[str], *, where: str) -> tuple[Any, int, list[str]]:
    """``result``, what a dataset loader returned, without the tasks whose ids ``items`` name; also how many tasks
    it held and the ids left out.

    ``result`` is a list of task ids or of rows with a ``task_id``, or a ``datasets.Dataset`` with a ``task_id``
    column.  A loader's tasks are known only by the ids it returns, so an item that matches none of them is logged,
    not refused.
    """

    wanted = set(items)
    if isinstance(result, list):
        task_ids = [str(row["task_id"]) if isinstance(row, Mapping) else str(row) for row in result]
        kept: Any = [row for row, task_id in zip(result, task_ids) if task_id not in wanted]
    elif callable(getattr(result, "select", None)) and "task_id" in (getattr(result, "column_names", None) or ()):
        task_ids = [str(task_id) for task_id in result["task_id"]]
        kept = result.select(
            [index for index, task_id in enumerate(task_ids) if task_id not in wanted], keep_in_memory=True
        )
    else:
        raise ConfigError(
            f"{where}.exclude_tasks needs the task ids its dataset loader returns: a list of task ids or of rows "
            f"with a task_id, or a datasets.Dataset with a task_id column; the loader returned a "
            f"{type(result).__name__}"
        )
    excluded = [task_id for task_id in task_ids if task_id in wanted]
    listed = set(task_ids)
    if unmatched := [item for item in items if item not in listed]:
        logger.warning(
            "%s.exclude_tasks: %s match none of the %d task ids the dataset loader returned (such as %r); a "
            "loader's tasks are matched by their ids",
            where,
            unmatched,
            len(task_ids),
            task_ids[0] if task_ids else None,
        )
    return kept, len(task_ids), excluded


def record_exclusions(run_dir: str | os.PathLike[str], exclusions: Sequence[TaskExclusion]) -> None:
    """Add what ``exclude_tasks`` left out to the ``lock.json`` in ``run_dir``, under ``excluded_tasks``:
    ``{environment: {split: {"listed": n, "excluded": [task id, ...]}}}``.  Nothing to record, or no ``lock.json``
    (a run that no command started), writes nothing."""

    if not exclusions:
        return
    record: dict[str, dict[str, Any]] = {}
    for exclusion in exclusions:
        record.setdefault(exclusion.environment, {})[exclusion.split] = {
            "listed": exclusion.listed,
            "excluded": list(exclusion.excluded),
        }
    if not update_run_lock(run_dir, {LOCK_KEY: record}):
        logger.debug("No lock.json in %s; the excluded tasks are not recorded", run_dir)
