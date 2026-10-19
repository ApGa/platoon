"""Rollout event files: the JSONL records the trajectory viewer tails and replays.

A rollout's trajectories are built from its gateway records once it finishes, so
rollouts write their events afterwards: :func:`emit_rollout_events` turns a
finished ``TrajectoryCollection`` (or its ``to_dict`` dump) into one file under
``<output_dir>/events/`` (:func:`rollout_events_path`). Each line is one event,
``{"type", "collection_id", "process_id", "ts", ...}``, of the types
``trajectory_created``, ``trajectory_task_set``, ``trajectory_step_added`` and
``trajectory_finished``, in that order per trajectory, root first.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from pydantic import BaseModel

if TYPE_CHECKING:
    from platoon.trajectory import TrajectoryCollection

# Rollout ids and mixture task ids may hold path separators and colons.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return str(obj)
    # Walk dataclass fields WITHOUT dataclasses.asdict(): asdict() deep-copies
    # every leaf value, and trajectory steps embed live SDK objects (e.g.
    # OpenHands events holding a threading.Lock / asyncio.Future) that cannot be
    # copied/pickled. Recursing via _to_jsonable routes those through
    # model_dump()/str() instead of copy.deepcopy.
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    # Fallback to string
    return str(obj)


# The misc keys the viewer reads when a trajectory is created, to place verifier trajectories under the
# trajectory they verify; the finished record carries the whole misc.
_CREATED_MISC_KEYS = (
    "subagent_reward_verifier_task",
    "subagent_reward_verifies_trajectory_id",
    "platoon_subagent_verifies_trajectory_id",
)


def _created_trajectory(traj: Dict[str, Any]) -> Dict[str, Any]:
    """A trajectory as its ``trajectory_created`` record carries it: no steps, task or misc beyond the viewer's
    keys, which the task, step and finished records that follow carry (writing them twice doubled every file)."""

    created = {**traj, "steps": [], "task": None}
    misc = traj.get("misc")
    if isinstance(misc, dict):
        created["misc"] = {key: misc[key] for key in _CREATED_MISC_KEYS if key in misc}
    return created


def trajectory_collection_dump_to_events(traj_collection_dump: Dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a serialized TrajectoryCollection dump (as produced by asdict) to event records.

    This enables offline visualization of previously saved rollouts.  Each part of a trajectory is written
    once: its ``trajectory_created`` record holds no steps or task (see :func:`_created_trajectory`).
    """
    events: list[dict[str, Any]] = []
    now = time.time()
    seq = 0
    collection_id = traj_collection_dump.get("id")
    trajectories = traj_collection_dump.get("trajectories", {})
    for traj in trajectories.values():
        events.append(
            {
                "type": "trajectory_created",
                "trajectory": _created_trajectory(traj),
                "collection_id": collection_id,
                "ts": now + (seq * 1e-6),
            }
        )
        seq += 1
        events.append(
            {
                "type": "trajectory_task_set",
                "trajectory_id": traj.get("id"),
                "task": traj.get("task"),
                "collection_id": collection_id,
                "ts": now + (seq * 1e-6),
            }
        )
        seq += 1
        # Emit zero-based step indices for consistency with live UI
        for idx, step in enumerate(traj.get("steps", []), start=0):
            events.append(
                {
                    "type": "trajectory_step_added",
                    "trajectory_id": traj.get("id"),
                    "step_index": idx,
                    "step": step,
                    "reward": traj.get("reward", 0.0),
                    # Include terminal/error messages if present in the dump
                    "finish_message": traj.get("finish_message"),
                    "error_message": traj.get("error_message"),
                    "collection_id": collection_id,
                    "ts": now + (seq * 1e-6),
                }
            )
            seq += 1
        # Emit a final finish event for each trajectory to capture terminal status
        events.append(
            {
                "type": "trajectory_finished",
                "trajectory_id": traj.get("id"),
                "reward": traj.get("reward", 0.0),
                "finish_message": traj.get("finish_message"),
                "error_message": traj.get("error_message"),
                "misc": traj.get("misc", {}),
                "collection_id": collection_id,
                "ts": now + (seq * 1e-6),
            }
        )
        seq += 1
    return events


def write_events_from_dump_to_jsonl(traj_collection_dump: Dict[str, Any], filepath: str | Path) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = trajectory_collection_dump_to_events(traj_collection_dump)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def rollout_events_path(output_dir: str | Path, task_id: str | None, collection_id: str) -> Path:
    """Where a rollout's event file goes: ``<output_dir>/events/events_<task id>_<collection id>.jsonl``."""

    task_part = _UNSAFE_FILENAME_CHARS.sub("_", str(task_id))[-120:]
    return Path(output_dir) / "events" / f"events_{task_part}_{_UNSAFE_FILENAME_CHARS.sub('_', collection_id)}.jsonl"


def emit_rollout_events(
    collection: TrajectoryCollection | Mapping[str, Any],
    path: str | Path,
    *,
    process_id: int | None = None,
) -> Path:
    """Write a finished rollout's events to ``path`` for the trajectory viewer, replacing any earlier file.

    ``collection`` is the ``TrajectoryCollection`` a rollout returned or its ``to_dict`` dump.  Every record
    also carries ``process_id`` (default: this process's), like the records live event sinks used to write.
    """

    dump = collection if isinstance(collection, Mapping) else collection.to_dict()
    process_id = os.getpid() if process_id is None else process_id
    lines = []
    for record in trajectory_collection_dump_to_events(_to_jsonable(dump)):
        record["process_id"] = process_id
        lines.append(json.dumps(record) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_events_path(path, process_id)
    try:
        with temporary.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        # Viewers tailing the directory see the whole file or none of it.
        temporary.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return path


def temporary_events_path(path: str | Path, process_id: int) -> Path:
    """Where process ``process_id`` writes the event file ``path`` before renaming it into place: a hidden
    ``.<name>.<pid>.tmp`` beside it."""

    path = Path(path)
    return path.with_name(f".{path.name}.{process_id}.tmp")


def remove_temporary_events(output_dir: str | Path, process_id: int) -> int:
    """Delete the temporary event files process ``process_id`` left under ``<output_dir>/events/`` and return
    how many there were.  A writer removes its own unless it is killed (``SIGKILL``, ``os._exit``) mid-write, so
    the rollout-worker supervisor calls this for a worker that ended without a response."""

    removed = 0
    for path in (Path(output_dir) / "events").glob(f".events_*.{process_id}.tmp"):
        with contextlib.suppress(OSError):
            path.unlink()
            removed += 1
    return removed
