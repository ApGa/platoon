"""Analyze Oolong real-task inference results stratified by episode-count buckets.

Usage:
    python -m platoon.oolong.inference_scripts.analyze_real_results_by_episode_count \
        /path/to/anchor_result_dir \
        /path/to/comparison_result_dir \
        --limit 650
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset

from platoon.oolong.inference_scripts.run_inference import reward_processor

BUCKET_ORDER = ["overall", "1_episode", "2_episodes", "3_episodes", "unknown"]
BUCKET_LABELS = {
    "overall": "overall",
    "1_episode": "55K",
    "2_episodes": "118K",
    "3_episodes": "175K",
    "unknown": "unknown",
}


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    min: float | None = None
    max: float | None = None

    def add(self, value: float | None) -> None:
        if value is None:
            return
        numeric_value = float(value)
        self.count += 1
        self.total += numeric_value
        self.min = numeric_value if self.min is None else min(self.min, numeric_value)
        self.max = numeric_value if self.max is None else max(self.max, numeric_value)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": (self.total / self.count) if self.count else 0.0,
            "min": self.min if self.min is not None else 0.0,
            "max": self.max if self.max is not None else 0.0,
        }


@dataclass
class BucketAccumulator:
    task_count: int = 0
    total_rollouts: int = 0
    valid_rollouts: int = 0
    successful_rollouts: int = 0
    errored_rollouts: int = 0
    task_success_at_k_sum: float = 0.0
    task_reward_at_k_mean_sum: float = 0.0
    task_reward_at_k_max_sum: float = 0.0
    task_reward_at_k_min_sum: float = 0.0
    reward_stats: RunningStats = field(default_factory=RunningStats)
    step_stats: RunningStats = field(default_factory=RunningStats)
    step_stats_success: RunningStats = field(default_factory=RunningStats)
    step_stats_failure: RunningStats = field(default_factory=RunningStats)
    wall_time_stats: RunningStats = field(default_factory=RunningStats)
    wall_time_stats_success: RunningStats = field(default_factory=RunningStats)
    wall_time_stats_failure: RunningStats = field(default_factory=RunningStats)

    def add_task(self, task_record: dict[str, Any]) -> None:
        self.task_count += 1
        self.task_success_at_k_sum += float(task_record.get("success_at_k", 0.0))
        self.task_reward_at_k_mean_sum += float(task_record.get("reward_at_k_mean", 0.0))
        self.task_reward_at_k_max_sum += float(task_record.get("reward_at_k_max", 0.0))
        self.task_reward_at_k_min_sum += float(task_record.get("reward_at_k_min", 0.0))

        for rollout in task_record.get("rollouts", []):
            self.total_rollouts += 1
            if rollout.get("error") is not None:
                self.errored_rollouts += 1
                continue

            self.valid_rollouts += 1
            success = bool(rollout.get("success"))
            reward = _maybe_float(rollout.get("reward"))
            num_steps = _maybe_float(rollout.get("num_steps_total"))
            wall_time = _maybe_float(rollout.get("wall_time_seconds"))

            self.reward_stats.add(reward)
            self.step_stats.add(num_steps)
            self.wall_time_stats.add(wall_time)

            if success:
                self.successful_rollouts += 1
                self.step_stats_success.add(num_steps)
                self.wall_time_stats_success.add(wall_time)
            else:
                self.step_stats_failure.add(num_steps)
                self.wall_time_stats_failure.add(wall_time)

    def to_summary(self) -> dict[str, Any]:
        failed_rollouts = self.valid_rollouts - self.successful_rollouts
        return {
            "summary": {
                "total_tasks": self.task_count,
                "total_rollouts": self.total_rollouts,
                "valid_rollouts": self.valid_rollouts,
                "successful_rollouts": self.successful_rollouts,
                "failed_rollouts": failed_rollouts,
                "errored_rollouts": self.errored_rollouts,
                "success_rate": (
                    self.successful_rollouts / self.valid_rollouts if self.valid_rollouts else 0.0
                ),
                "success_at_k": (self.task_success_at_k_sum / self.task_count) if self.task_count else 0.0,
                "reward_mean": self.reward_stats.to_dict()["mean"],
                "reward_max": self.reward_stats.to_dict()["max"],
                "reward_min": self.reward_stats.to_dict()["min"],
                "reward_at_k_mean": (
                    self.task_reward_at_k_mean_sum / self.task_count if self.task_count else 0.0
                ),
                "reward_at_k_max": (
                    self.task_reward_at_k_max_sum / self.task_count if self.task_count else 0.0
                ),
                "reward_at_k_min": (
                    self.task_reward_at_k_min_sum / self.task_count if self.task_count else 0.0
                ),
            },
            "stats": {
                "num_steps_total": {
                    "overall": self.step_stats.to_dict(),
                    "success": self.step_stats_success.to_dict(),
                    "failure": self.step_stats_failure.to_dict(),
                },
                "rollout_wall_time_seconds": {
                    "overall": self.wall_time_stats.to_dict(),
                    "success": self.wall_time_stats_success.to_dict(),
                    "failure": self.wall_time_stats_failure.to_dict(),
                },
            },
        }


@dataclass
class AnalyzedRun:
    label: str
    result_dir: str
    bucket_metrics: dict[str, Any]
    nonzero_score_task_ids: set[str]
    task_records_by_id: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "result_dir": self.result_dir,
            "bucket_metrics": self.bucket_metrics,
        }


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _count_steps(traj: dict[str, Any]) -> int:
    steps = traj.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _compute_depth_stats(collection: dict[str, Any]) -> tuple[int, int, int, dict[str, int], dict[str, int]]:
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return 0, 0, 0, {}, {}

    ids = list(trajectories.keys())
    root_id = ids[0]
    parents: dict[str, str | None] = {}
    steps_per_traj: dict[str, int] = {}
    for traj_id, traj in trajectories.items():
        if not isinstance(traj, dict):
            continue
        parent_info = traj.get("parent_info")
        parent_id = parent_info.get("id") if isinstance(parent_info, dict) else None
        parents[traj_id] = parent_id
        steps_per_traj[traj_id] = _count_steps(traj)

    depth_cache: dict[str, int] = {}

    def _depth_for(traj_id: str) -> int:
        if traj_id in depth_cache:
            return depth_cache[traj_id]
        parent_id = parents.get(traj_id)
        if traj_id == root_id or parent_id is None or parent_id not in parents:
            depth_cache[traj_id] = 0
            return 0
        depth = _depth_for(parent_id) + 1
        depth_cache[traj_id] = depth
        return depth

    depth_counts: dict[str, int] = {}
    depth_steps: dict[str, int] = {}
    num_steps_total = 0
    num_steps_root = 0
    num_steps_sub = 0
    num_subtrajectories = 0

    for traj_id, step_count in steps_per_traj.items():
        depth = _depth_for(traj_id)
        depth_key = str(depth)
        num_steps_total += step_count
        if depth == 0:
            num_steps_root += step_count
        else:
            num_subtrajectories += 1
            num_steps_sub += step_count
            depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
            depth_steps[depth_key] = depth_steps.get(depth_key, 0) + step_count

    return num_steps_total, num_steps_root, num_steps_sub, depth_counts, depth_steps


def _rollout_success(root: dict[str, Any], root_reward: float) -> bool:
    steps = root.get("steps")
    if isinstance(steps, list) and steps:
        last_step = steps[-1]
        if isinstance(last_step, dict):
            reward_misc = last_step.get("misc", {}).get("reward_misc", {})
            if isinstance(reward_misc, dict):
                for key in ("reward/success", "success"):
                    if key in reward_misc:
                        try:
                            return float(reward_misc[key]) >= 1.0
                        except Exception:
                            pass
    return float(root_reward) >= 1.0


def _load_episode_count_map() -> dict[str, int]:
    dataset = load_dataset("oolongbench/oolong-real", "dnd", split="test")
    return {
        f"oolong.real.test.{idx}": len(example.get("episodes") or [])
        for idx, example in enumerate(dataset)
    }


def _episode_bucket(task_id: str, episode_count_by_task_id: dict[str, int]) -> str:
    count = episode_count_by_task_id.get(task_id)
    if count == 1:
        return "1_episode"
    if count == 2:
        return "2_episodes"
    if count == 3:
        return "3_episodes"
    return "unknown"


def _has_nonzero_score(task_record: dict[str, Any]) -> bool:
    for key in ("reward_at_k_max", "reward_at_k_mean", "success_at_k"):
        value = _maybe_float(task_record.get(key))
        if value is not None and value > 0.0:
            return True
    return False


def _build_task_record_from_rollout_dir(rollout_dir: Path) -> tuple[datetime | None, dict[str, Any]] | None:
    metadata_path = rollout_dir / "metadata.json"
    collection_path = rollout_dir / "trajectory_collection.json"
    if not metadata_path.exists() or not collection_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return None
    root = next(iter(trajectories.values()))
    if not isinstance(root, dict):
        return None

    reward, reward_components = reward_processor(root)
    success = _rollout_success(root, reward)
    num_steps_total, num_steps_root, num_steps_sub, depth_counts, depth_steps = _compute_depth_stats(collection)
    task_id = str(metadata.get("task_id") or rollout_dir.parent.name)
    rollout_index = int(metadata.get("rollout_index", rollout_dir.name.replace("rollout_", "")))
    wall_time_seconds = metadata.get("wall_time_seconds")
    created_at = _parse_iso_datetime(metadata.get("created_at"))
    error = metadata.get("error")

    task_record = {
        "task_id": task_id,
        "success_at_k": 1.0 if success else 0.0,
        "num_rollouts": 1,
        "num_valid_rollouts": 0 if error is not None else 1,
        "num_failed_rollouts": 0 if success or error is not None else 1,
        "num_successful_rollouts": 1 if success else 0,
        "success_rate_within_task": 1.0 if success else 0.0,
        "reward_at_k_mean": float(reward),
        "reward_at_k_max": float(reward),
        "reward_at_k_min": float(reward),
        "rollouts": [
            {
                "task_id": task_id,
                "rollout_index": rollout_index,
                "success": success,
                "reward": float(reward),
                "reward_components": reward_components,
                "num_steps_total": num_steps_total,
                "num_steps_root": num_steps_root,
                "num_steps_subtrajectories": num_steps_sub,
                "num_subtrajectories": sum(depth_counts.values()),
                "subtrajectory_depth_counts": depth_counts,
                "subtrajectory_depth_steps": depth_steps,
                "wall_time_seconds": wall_time_seconds,
                "workflow_metrics": {},
                "source_path": str(collection_path),
                "error": error,
            }
        ],
    }
    return created_at, task_record


def _iter_current_task_records(result_dir: Path) -> list[tuple[datetime | None, dict[str, Any]]]:
    records: list[tuple[datetime | None, dict[str, Any]]] = []
    for metadata_path in sorted((result_dir / "rollouts").glob("*/rollout_*/metadata.json")):
        built = _build_task_record_from_rollout_dir(metadata_path.parent)
        if built is not None:
            records.append(built)
    records.sort(key=lambda item: (item[0] is None, item[0], item[1].get("task_id")))
    return records


def _iter_report_task_records(result_dir: Path) -> Iterable[dict[str, Any]]:
    task_results_path = result_dir / "reports" / "task_results.jsonl"
    if task_results_path.exists():
        with task_results_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Failed to parse {task_results_path} line {line_number}: {exc}") from exc
                if isinstance(record, dict):
                    yield record
        return

    final_report_path = result_dir / "reports" / "final_report.json"
    if not final_report_path.exists():
        raise FileNotFoundError(f"Could not find reports/task_results.jsonl or reports/final_report.json in {result_dir}")

    with final_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for record in report.get("tasks", []):
        if isinstance(record, dict):
            yield record


def _summarize_task_records(task_records: Iterable[dict[str, Any]], episode_count_by_task_id: dict[str, int]) -> dict[str, Any]:
    buckets = {bucket: BucketAccumulator() for bucket in BUCKET_ORDER}
    for task_record in task_records:
        task_id = task_record.get("task_id")
        bucket = _episode_bucket(str(task_id), episode_count_by_task_id) if isinstance(task_id, str) else "unknown"
        buckets["overall"].add_task(task_record)
        buckets[bucket].add_task(task_record)
    return {
        bucket: acc.to_summary()
        for bucket, acc in buckets.items()
        if bucket == "overall" or acc.task_count > 0
    }


def analyze_result_dirs(anchor_dir: Path, comparison_dir: Path, limit: int) -> dict[str, Any]:
    episode_count_by_task_id = _load_episode_count_map()

    anchor_records = _iter_current_task_records(anchor_dir)
    selected_anchor = [record for _, record in anchor_records[:limit]]
    selected_task_ids = [record["task_id"] for record in selected_anchor if isinstance(record.get("task_id"), str)]

    comparison_records_by_id = {
        record["task_id"]: record
        for record in _iter_report_task_records(comparison_dir)
        if isinstance(record.get("task_id"), str)
    }
    selected_comparison = [comparison_records_by_id[task_id] for task_id in selected_task_ids if task_id in comparison_records_by_id]

    anchor_nonzero = {
        record["task_id"]
        for record in selected_anchor
        if isinstance(record.get("task_id"), str) and _has_nonzero_score(record)
    }
    comparison_nonzero = {
        record["task_id"]
        for record in selected_comparison
        if isinstance(record.get("task_id"), str) and _has_nonzero_score(record)
    }
    common_nonzero_task_ids = sorted(anchor_nonzero & comparison_nonzero)

    payload = {
        "selection": {
            "anchor_dir": str(anchor_dir),
            "comparison_dir": str(comparison_dir),
            "requested_limit": limit,
            "anchor_selected_tasks": len(selected_anchor),
            "comparison_selected_tasks": len(selected_comparison),
            "matched_task_ids": len(selected_task_ids),
            "common_nonzero_task_count": len(common_nonzero_task_ids),
            "task_ids": selected_task_ids,
        },
        "results": [
            {
                "label": anchor_dir.name,
                "result_dir": str(anchor_dir),
                "bucket_metrics": _summarize_task_records(selected_anchor, episode_count_by_task_id),
            },
            {
                "label": comparison_dir.name,
                "result_dir": str(comparison_dir),
                "bucket_metrics": _summarize_task_records(selected_comparison, episode_count_by_task_id),
            },
        ],
        "common_nonzero_intersection": {
            "task_ids": common_nonzero_task_ids,
            "task_count": len(common_nonzero_task_ids),
            "results": [
                {
                    "label": anchor_dir.name,
                    "result_dir": str(anchor_dir),
                    "bucket_metrics": _summarize_task_records(
                        [record for record in selected_anchor if record.get("task_id") in common_nonzero_task_ids],
                        episode_count_by_task_id,
                    ),
                },
                {
                    "label": comparison_dir.name,
                    "result_dir": str(comparison_dir),
                    "bucket_metrics": _summarize_task_records(
                        [record for record in selected_comparison if record.get("task_id") in common_nonzero_task_ids],
                        episode_count_by_task_id,
                    ),
                },
            ],
        },
    }
    return payload


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_float(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_lines = [" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _summary_rows(result: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for bucket in BUCKET_ORDER:
        metrics = result["bucket_metrics"].get(bucket)
        if not metrics:
            continue
        summary = metrics["summary"]
        stats = metrics["stats"]
        rows.append(
            [
                BUCKET_LABELS[bucket],
                str(summary["total_tasks"]),
                _format_percent(summary["success_at_k"]),
                _format_percent(summary["success_rate"]),
                _format_float(summary["reward_at_k_mean"]),
                _format_float(summary["reward_mean"]),
                _format_float(stats["num_steps_total"]["overall"]["mean"]),
                _format_float(stats["rollout_wall_time_seconds"]["overall"]["mean"]),
            ]
        )
    return rows


def print_text_report(payload: dict[str, Any]) -> None:
    headers = ["bucket", "tasks", "success@k", "success", "reward@k", "reward", "avg_steps", "avg_time_s"]
    for result in payload["results"]:
        print(f"\nRun: {result['label']}")
        print(result["result_dir"])
        print(_render_table(headers, _summary_rows(result)))

    print(
        f"\nSelection: {payload['selection']['anchor_selected_tasks']} anchor tasks, "
        f"{payload['selection']['comparison_selected_tasks']} comparison tasks"
    )
    print(f"Common non-zero-score task intersection: {payload['common_nonzero_intersection']['task_count']} tasks")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Oolong real-task inference results stratified by episode-count buckets."
    )
    parser.add_argument("anchor_result_dir", help="Result dir whose current rollouts determine the first-N task subset.")
    parser.add_argument("comparison_result_dir", help="Comparison result dir, typically a completed baseline run.")
    parser.add_argument("--limit", type=int, default=650, help="Number of anchor tasks to compare (default: 650).")
    parser.add_argument("--json-out", default=None, help="Optional path to write the machine-readable summary JSON.")
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout instead of a text report.")
    args = parser.parse_args()

    payload = analyze_result_dirs(
        anchor_dir=Path(args.anchor_result_dir).expanduser().resolve(),
        comparison_dir=Path(args.comparison_result_dir).expanduser().resolve(),
        limit=args.limit,
    )

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(payload)


if __name__ == "__main__":
    main()