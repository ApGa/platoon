from __future__ import annotations

import json
import queue
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel  # TODO: Add pydantic as explicit dependency?

from platoon.envs.base import Task
from platoon.episode.trajectory import Trajectory, TrajectoryEventHandler


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


class JsonlFileSink(TrajectoryEventHandler):
    """Writes trajectory events as JSONL records for offline or live tailing.

    Each line is a dict with at least keys: {"type": str, ...}.
    """

    def __init__(
        self,
        filepath: str | Path,
        create_parents: bool = True,
        *,
        collection_id: str | None = None,
        process_id: str | int | None = None,
    ) -> None:
        self.filepath = Path(filepath)
        if self.filepath.exists():
            self.filepath.unlink()
        if create_parents:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.collection_id = collection_id
        self.process_id = process_id

    def _write(self, record: dict[str, Any]) -> None:
        if self.collection_id is not None and "collection_id" not in record:
            record["collection_id"] = self.collection_id
        if self.process_id is not None and "process_id" not in record:
            record["process_id"] = self.process_id
        # Attach wall-clock timestamp for ordering and replay
        record.setdefault("ts", time.time())
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def on_trajectory_created(self, trajectory: Trajectory) -> None:  # type: ignore[override]
        self._write(
            {
                "type": "trajectory_created",
                "trajectory": _to_jsonable(trajectory),
            }
        )

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:  # type: ignore[override]
        self._write(
            {
                "type": "trajectory_task_set",
                "trajectory_id": trajectory.id,
                "task": _to_jsonable(task),
            }
        )

    def on_trajectory_step_added(self, trajectory: Trajectory, step: Any) -> None:  # type: ignore[override]
        self._write(
            {
                "type": "trajectory_step_added",
                "trajectory_id": trajectory.id,
                # zero-based index for steps in UI/consumers
                "step_index": len(trajectory.steps) - 1,
                "step": _to_jsonable(step),
                "reward": trajectory.reward,
                # propagate fields that may update when a step is added
                "finish_message": trajectory.finish_message,
                "error_message": trajectory.error_message,
            }
        )

    def on_trajectory_finished(self, trajectory: Trajectory) -> None:  # type: ignore[override]
        self._write(
            {
                "type": "trajectory_finished",
                "trajectory_id": trajectory.id,
                "reward": trajectory.reward,
                "finish_message": trajectory.finish_message,
                "error_message": trajectory.error_message,
                "misc": _to_jsonable(trajectory.misc),
            }
        )


class QueueSink(TrajectoryEventHandler):
    """Pushes events into a thread-safe queue for consumption by a UI thread/process."""

    def __init__(self, q: queue.Queue, *, collection_id: str | None = None, process_id: str | int | None = None):
        self.q = q
        self.collection_id = collection_id
        self.process_id = process_id

    def _put(self, record: dict[str, Any]) -> None:
        if self.collection_id is not None and "collection_id" not in record:
            record["collection_id"] = self.collection_id
        if self.process_id is not None and "process_id" not in record:
            record["process_id"] = self.process_id
        # Attach wall-clock timestamp for ordering and replay
        record.setdefault("ts", time.time())
        try:
            self.q.put_nowait(record)
        except queue.Full:
            # Drop on full to avoid blocking rollout
            pass

    def on_trajectory_created(self, trajectory: Trajectory) -> None:  # type: ignore[override]
        self._put(
            {
                "type": "trajectory_created",
                "trajectory": _to_jsonable(trajectory),
            }
        )

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:  # type: ignore[override]
        self._put(
            {
                "type": "trajectory_task_set",
                "trajectory_id": trajectory.id,
                "task": _to_jsonable(task),
            }
        )

    def on_trajectory_step_added(self, trajectory: Trajectory, step: Any) -> None:  # type: ignore[override]
        self._put(
            {
                "type": "trajectory_step_added",
                "trajectory_id": trajectory.id,
                # zero-based index for steps in UI/consumers
                "step_index": len(trajectory.steps) - 1,
                "step": _to_jsonable(step),
                "reward": trajectory.reward,
                # propagate fields that may update when a step is added
                "finish_message": trajectory.finish_message,
                "error_message": trajectory.error_message,
            }
        )

    def on_trajectory_finished(self, trajectory: Trajectory) -> None:  # type: ignore[override]
        self._put(
            {
                "type": "trajectory_finished",
                "trajectory_id": trajectory.id,
                "reward": trajectory.reward,
                "finish_message": trajectory.finish_message,
                "error_message": trajectory.error_message,
                "misc": _to_jsonable(trajectory.misc),
            }
        )


def trajectory_collection_dump_to_events(traj_collection_dump: Dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a serialized TrajectoryCollection dump (as produced by asdict) to event records.

    This enables offline visualization of previously saved rollouts.
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
                "trajectory": traj,
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


def enqueue_events_from_dump(traj_collection_dump: Dict[str, Any], q: queue.Queue) -> None:
    for r in trajectory_collection_dump_to_events(traj_collection_dump):
        try:
            q.put_nowait(r)
        except queue.Full:
            break


class MarkdownFileSink(TrajectoryEventHandler):
    """Replicates the original human-readable markdown logging behavior.

    Deprecated behavior existed on TrajectoryCollection.output_file; use this sink instead.
    """

    def __init__(self, filepath: str | Path, create_parents: bool = True) -> None:
        self.filepath = Path(filepath)
        if self.filepath.exists():
            self.filepath.unlink()
        if create_parents:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
        # initialize file
        with self.filepath.open("w", encoding="utf-8") as f:
            f.write("## Trajectory Collection\n")

    def on_trajectory_created(self, trajectory: Trajectory) -> None:  # type: ignore[override]
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(f"## Created new trajectory: {trajectory.id}\n")
            if trajectory.parent_info is not None:
                f.write(f"Forked from {trajectory.parent_info.id} at step {trajectory.parent_info.fork_step}\n")

    def on_trajectory_task_set(self, trajectory: Trajectory, task: Task | None) -> None:  # type: ignore[override]
        # no-op for markdown sink until first step; goal printed per step
        return

    def on_trajectory_step_added(self, trajectory: Trajectory, step: Any) -> None:  # type: ignore[override]
        with self.filepath.open("a", encoding="utf-8") as f:
            f.write(f"## Step {len(trajectory.steps)} for trajectory {trajectory.id}:\n")
            goal = getattr(trajectory.task, "goal", None) if trajectory.task is not None else None
            f.write(f"### Goal: {goal}\n")
            f.write(str(step) + "\n")
            f.write("--------------------------------\n")
