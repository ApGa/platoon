"""Analyze fan-out vs cancellation for a single AppWorld task from event JSONLs.

Usage:
    python -m platoon.appworld.inference_scripts.analyze_task_fanout \
        --events-root /mnt/efs/tmp/areal/experiments/<trial>/train_rollout \
        --task-id 34d9492_3
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CollectionStats:
    file: str
    collection_id: str | None
    total_trajectories: int
    cancelled_trajectories: int
    root_cancelled: bool
    max_fanout_single_step: int
    fanout_steps_ge_2: int
    fanout_steps_ge_3: int

    @property
    def cancel_rate(self) -> float:
        if self.total_trajectories == 0:
            return 0.0
        return self.cancelled_trajectories / self.total_trajectories


def _task_id_from_filename(path: Path) -> str | None:
    # Expected format: events_<task_id>_<collection_id>.jsonl
    name = path.name
    if not name.startswith("events_") or not name.endswith(".jsonl"):
        return None
    stem = name[len("events_") : -len(".jsonl")]
    idx = stem.rfind("_")
    if idx <= 0:
        return None
    return stem[:idx]


def analyze_collection(path: Path) -> CollectionStats | None:
    parent_by_traj: dict[str, str | None] = {}
    parent_step_counter: Counter[tuple[str, int]] = Counter()
    cancelled = 0
    collection_id: str | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            if collection_id is None:
                collection_id = rec.get("collection_id")

            t = rec.get("type")
            if t == "trajectory_created":
                traj = rec.get("trajectory") or {}
                traj_id = traj.get("id")
                parent_info = traj.get("parent_info") or {}
                parent_id = parent_info.get("id")
                fork_step = parent_info.get("fork_step")
                if isinstance(traj_id, str):
                    parent_by_traj[traj_id] = parent_id if isinstance(parent_id, str) else None
                    if isinstance(parent_id, str) and isinstance(fork_step, int):
                        parent_step_counter[(parent_id, fork_step)] += 1
            elif t == "trajectory_finished":
                err = rec.get("error_message") or ""
                if isinstance(err, str) and "CancelledError" in err:
                    cancelled += 1

    if not parent_by_traj:
        return None

    root_ids = {tid for tid, pid in parent_by_traj.items() if pid is None}
    root_cancelled = False

    # Second pass to detect root cancellations specifically.
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "trajectory_finished":
                continue
            traj_id = rec.get("trajectory_id")
            err = rec.get("error_message") or ""
            if traj_id in root_ids and isinstance(err, str) and "CancelledError" in err:
                root_cancelled = True
                break

    fanouts = list(parent_step_counter.values())
    max_fanout = max(fanouts) if fanouts else 0
    fanout_ge_2 = sum(1 for v in fanouts if v >= 2)
    fanout_ge_3 = sum(1 for v in fanouts if v >= 3)

    return CollectionStats(
        file=str(path),
        collection_id=collection_id,
        total_trajectories=len(parent_by_traj),
        cancelled_trajectories=cancelled,
        root_cancelled=root_cancelled,
        max_fanout_single_step=max_fanout,
        fanout_steps_ge_2=fanout_ge_2,
        fanout_steps_ge_3=fanout_ge_3,
    )


def iter_task_event_files(events_root: Path, task_id: str) -> list[Path]:
    return sorted(events_root.rglob(f"events_{task_id}_*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fan-out vs cancellation for one AppWorld task.")
    parser.add_argument("--events-root", type=Path, required=True, help="Root containing train_rollout/*/events files.")
    parser.add_argument("--task-id", type=str, required=True, help="Task ID to analyze (e.g. 34d9492_3).")
    args = parser.parse_args()

    files = iter_task_event_files(args.events_root, args.task_id)
    if not files:
        print(f"No matching event files for task_id={args.task_id} under {args.events_root}")
        return

    rows: list[CollectionStats] = []
    for f in files:
        if _task_id_from_filename(f) != args.task_id:
            continue
        stats = analyze_collection(f)
        if stats is not None:
            rows.append(stats)

    if not rows:
        print(f"No valid collections parsed for task_id={args.task_id}")
        return

    print(f"Task {args.task_id}: {len(rows)} collections analyzed")
    print("-" * 110)
    print(
        "max_fanout  cancel_rate  cancelled/total  fanout_steps>=2  root_cancelled  file"
    )
    for r in sorted(rows, key=lambda x: (x.max_fanout_single_step, x.cancel_rate), reverse=True):
        print(
            f"{r.max_fanout_single_step:10d}  "
            f"{r.cancel_rate:10.2%}  "
            f"{r.cancelled_trajectories:3d}/{r.total_trajectories:<3d}          "
            f"{r.fanout_steps_ge_2:6d}       "
            f"{str(r.root_cancelled):>5s}        "
            f"{Path(r.file).name}"
        )

    # Aggregate by max-fanout bucket.
    buckets: dict[str, list[CollectionStats]] = defaultdict(list)
    for r in rows:
        if r.max_fanout_single_step <= 1:
            key = "fanout<=1"
        elif r.max_fanout_single_step == 2:
            key = "fanout=2"
        else:
            key = "fanout>=3"
        buckets[key].append(r)

    print("\nAggregate by max fan-out bucket")
    print("-" * 110)
    for key in ("fanout<=1", "fanout=2", "fanout>=3"):
        vals = buckets.get(key, [])
        if not vals:
            continue
        total_traj = sum(v.total_trajectories for v in vals)
        total_cancel = sum(v.cancelled_trajectories for v in vals)
        root_cancel_count = sum(1 for v in vals if v.root_cancelled)
        print(
            f"{key:10s}: collections={len(vals):3d}, "
            f"cancel_rate={((total_cancel / total_traj) if total_traj else 0.0):.2%}, "
            f"root_cancelled_collections={root_cancel_count}"
        )


if __name__ == "__main__":
    main()

