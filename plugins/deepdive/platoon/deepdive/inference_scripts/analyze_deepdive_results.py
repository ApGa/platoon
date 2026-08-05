"""Compare DeepDive inference results across runs.

Usage:
    python platoon/deepdive/inference_scripts/analyze_deepdive_results.py \
        /path/to/result_dir_or_rollouts_a \
        /path/to/result_dir_or_rollouts_b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


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

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "min": self.min if self.min is not None else 0.0,
            "max": self.max if self.max is not None else 0.0,
        }


@dataclass
class TaskMetrics:
    task_id: str
    success: bool
    reward: float
    num_steps_total: float
    num_steps_root: float
    num_subtrajectories: float
    max_subtrajectory_depth: float
    wall_time_seconds: float
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    finish_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "num_steps_total": self.num_steps_total,
            "num_steps_root": self.num_steps_root,
            "num_subtrajectories": self.num_subtrajectories,
            "max_subtrajectory_depth": self.max_subtrajectory_depth,
            "wall_time_seconds": self.wall_time_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_message": self.finish_message,
        }


@dataclass
class RunAccumulator:
    task_count: int = 0
    total_rollouts: int = 0
    valid_rollouts: int = 0
    successful_rollouts: int = 0
    errored_rollouts: int = 0
    task_success_at_k_sum: float = 0.0
    task_reward_at_k_mean_sum: float = 0.0
    reward_stats: RunningStats = field(default_factory=RunningStats)
    step_stats: RunningStats = field(default_factory=RunningStats)
    root_step_stats: RunningStats = field(default_factory=RunningStats)
    subtrajectory_stats: RunningStats = field(default_factory=RunningStats)
    depth_stats: RunningStats = field(default_factory=RunningStats)
    wall_time_stats: RunningStats = field(default_factory=RunningStats)
    prompt_token_stats: RunningStats = field(default_factory=RunningStats)
    completion_token_stats: RunningStats = field(default_factory=RunningStats)
    total_token_stats: RunningStats = field(default_factory=RunningStats)

    def add_task(self, task_record: dict[str, Any]) -> None:
        self.task_count += 1
        self.task_success_at_k_sum += float(task_record.get("success_at_k", 0.0))
        self.task_reward_at_k_mean_sum += float(task_record.get("reward_at_k_mean", 0.0))

        for rollout in task_record.get("rollouts", []):
            if not isinstance(rollout, dict):
                continue
            self.total_rollouts += 1
            if rollout.get("error") is not None:
                self.errored_rollouts += 1
                continue

            self.valid_rollouts += 1
            if bool(rollout.get("success")):
                self.successful_rollouts += 1

            token_usage = _extract_rollout_token_usage(rollout)
            self.reward_stats.add(_maybe_float(rollout.get("reward")))
            self.step_stats.add(_maybe_float(rollout.get("num_steps_total")))
            self.root_step_stats.add(_maybe_float(rollout.get("num_steps_root")))
            self.subtrajectory_stats.add(_maybe_float(rollout.get("num_subtrajectories")))
            self.depth_stats.add(_max_subtrajectory_depth(rollout))
            self.wall_time_stats.add(_maybe_float(rollout.get("wall_time_seconds")))
            self.prompt_token_stats.add(token_usage["prompt_tokens"])
            self.completion_token_stats.add(token_usage["completion_tokens"])
            self.total_token_stats.add(token_usage["total_tokens"])

    def to_summary(self) -> dict[str, Any]:
        failed_rollouts = self.valid_rollouts - self.successful_rollouts
        return {
            "total_tasks": self.task_count,
            "total_rollouts": self.total_rollouts,
            "valid_rollouts": self.valid_rollouts,
            "successful_rollouts": self.successful_rollouts,
            "failed_rollouts": failed_rollouts,
            "errored_rollouts": self.errored_rollouts,
            "success_rate": self.successful_rollouts / self.valid_rollouts
            if self.valid_rollouts
            else 0.0,
            "success_at_k": self.task_success_at_k_sum / self.task_count
            if self.task_count
            else 0.0,
            "reward_mean": self.reward_stats.mean,
            "reward_at_k_mean": self.task_reward_at_k_mean_sum / self.task_count
            if self.task_count
            else 0.0,
            "num_steps_total": self.step_stats.to_dict(),
            "num_steps_root": self.root_step_stats.to_dict(),
            "num_subtrajectories": self.subtrajectory_stats.to_dict(),
            "max_subtrajectory_depth": self.depth_stats.to_dict(),
            "wall_time_seconds": self.wall_time_stats.to_dict(),
            "prompt_tokens": self.prompt_token_stats.to_dict(),
            "completion_tokens": self.completion_token_stats.to_dict(),
            "total_tokens": self.total_token_stats.to_dict(),
        }


@dataclass
class AnalyzedRun:
    label: str
    result_dir: str
    summary: dict[str, Any]
    task_metrics_by_id: dict[str, TaskMetrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "result_dir": self.result_dir,
            "summary": self.summary,
            "task_metrics": {
                task_id: metrics.to_dict()
                for task_id, metrics in sorted(self.task_metrics_by_id.items())
            },
        }


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _resolve_result_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name == "rollouts":
        return resolved.parent
    return resolved


def _iter_task_records(result_dir: Path) -> Iterable[dict[str, Any]]:
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
                    raise ValueError(f"Failed to parse {task_results_path} line {line_number}: {exc}")
                if isinstance(record, dict):
                    yield record
        return

    final_report_path = result_dir / "reports" / "final_report.json"
    if not final_report_path.exists():
        raise FileNotFoundError(
            f"Could not find either {task_results_path} or {final_report_path} for {result_dir}"
        )

    with final_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for record in report.get("tasks", []):
        if isinstance(record, dict):
            yield record


def _extract_rollout_token_usage(rollout: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0

    trajectory_collection = rollout.get("trajectory_collection")
    if not isinstance(trajectory_collection, dict):
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    trajectories = trajectory_collection.get("trajectories")
    if not isinstance(trajectories, dict):
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    for trajectory in trajectories.values():
        if not isinstance(trajectory, dict):
            continue
        steps = trajectory.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            usage = (
                step.get("misc", {})
                .get("action_misc", {})
                .get("usage", {})
            )
            if not isinstance(usage, dict):
                continue

            step_prompt_tokens = _maybe_float(usage.get("prompt_tokens")) or 0.0
            step_completion_tokens = _maybe_float(usage.get("completion_tokens")) or 0.0
            step_total_tokens = _maybe_float(usage.get("total_tokens"))
            prompt_tokens += step_prompt_tokens
            completion_tokens += step_completion_tokens
            total_tokens += (
                step_total_tokens
                if step_total_tokens is not None
                else step_prompt_tokens + step_completion_tokens
            )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _max_subtrajectory_depth(rollout: dict[str, Any]) -> float:
    depth_counts = rollout.get("subtrajectory_depth_counts")
    if not isinstance(depth_counts, dict) or not depth_counts:
        return 0.0
    depths = []
    for depth, count in depth_counts.items():
        if not count:
            continue
        numeric_depth = _maybe_float(depth)
        if numeric_depth is not None:
            depths.append(numeric_depth)
    return max(depths) if depths else 0.0


def _extract_task_metrics(task_record: dict[str, Any]) -> TaskMetrics | None:
    task_id = task_record.get("task_id")
    if not isinstance(task_id, str):
        return None

    rollouts = [rollout for rollout in task_record.get("rollouts", []) if isinstance(rollout, dict)]
    rollout = rollouts[0] if rollouts else {}
    token_usage = _extract_rollout_token_usage(rollout)
    return TaskMetrics(
        task_id=task_id,
        success=bool(task_record.get("success_at_k", 0.0)),
        reward=float(task_record.get("reward_at_k_mean", 0.0)),
        num_steps_total=_maybe_float(rollout.get("num_steps_total")) or 0.0,
        num_steps_root=_maybe_float(rollout.get("num_steps_root")) or 0.0,
        num_subtrajectories=_maybe_float(rollout.get("num_subtrajectories")) or 0.0,
        max_subtrajectory_depth=_max_subtrajectory_depth(rollout),
        wall_time_seconds=_maybe_float(rollout.get("wall_time_seconds")) or 0.0,
        prompt_tokens=token_usage["prompt_tokens"],
        completion_tokens=token_usage["completion_tokens"],
        total_tokens=token_usage["total_tokens"],
        finish_message=rollout.get("finish_message") if isinstance(rollout.get("finish_message"), str) else None,
    )


def analyze_result_path(path: Path, label: str | None = None) -> AnalyzedRun:
    result_dir = _resolve_result_dir(path)
    accumulator = RunAccumulator()
    task_metrics_by_id = {}

    for task_record in _iter_task_records(result_dir):
        accumulator.add_task(task_record)
        task_metrics = _extract_task_metrics(task_record)
        if task_metrics is not None:
            task_metrics_by_id[task_metrics.task_id] = task_metrics

    return AnalyzedRun(
        label=label or result_dir.name,
        result_dir=str(result_dir),
        summary=accumulator.to_summary(),
        task_metrics_by_id=task_metrics_by_id,
    )


def build_pairwise_comparison(left: AnalyzedRun, right: AnalyzedRun) -> dict[str, Any]:
    left_task_ids = set(left.task_metrics_by_id)
    right_task_ids = set(right.task_metrics_by_id)
    common_task_ids = sorted(left_task_ids & right_task_ids)

    left_only_success = []
    right_only_success = []
    both_success = []
    neither_success = []
    per_task_deltas = []

    for task_id in common_task_ids:
        left_metrics = left.task_metrics_by_id[task_id]
        right_metrics = right.task_metrics_by_id[task_id]
        if left_metrics.success and right_metrics.success:
            both_success.append(task_id)
        elif left_metrics.success:
            left_only_success.append(task_id)
        elif right_metrics.success:
            right_only_success.append(task_id)
        else:
            neither_success.append(task_id)

        per_task_deltas.append(
            {
                "task_id": task_id,
                f"{left.label}_success": left_metrics.success,
                f"{right.label}_success": right_metrics.success,
                "reward_delta": left_metrics.reward - right_metrics.reward,
                "steps_delta": left_metrics.num_steps_total - right_metrics.num_steps_total,
                "wall_time_seconds_delta": (
                    left_metrics.wall_time_seconds - right_metrics.wall_time_seconds
                ),
                "total_tokens_delta": left_metrics.total_tokens - right_metrics.total_tokens,
                f"{left.label}_finish_message": left_metrics.finish_message,
                f"{right.label}_finish_message": right_metrics.finish_message,
            }
        )

    return {
        "common_task_count": len(common_task_ids),
        "left_missing_task_ids": sorted(right_task_ids - left_task_ids),
        "right_missing_task_ids": sorted(left_task_ids - right_task_ids),
        "both_success_task_ids": both_success,
        "left_only_success_task_ids": left_only_success,
        "right_only_success_task_ids": right_only_success,
        "neither_success_task_ids": neither_success,
        "per_task_deltas": per_task_deltas,
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_float(value: float, precision: int = 2) -> str:
    return f"{value:.{precision}f}"


def _format_delta(value: float, precision: int = 2) -> str:
    return f"{value:+.{precision}f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator_line = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_lines = [" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([header_line, separator_line, *row_lines])


def _summary_rows(results: list[AnalyzedRun]) -> list[list[str]]:
    rows = []
    for result in results:
        summary = result.summary
        rows.append(
            [
                result.label,
                str(summary["total_tasks"]),
                _format_percent(summary["success_at_k"]),
                _format_float(summary["reward_at_k_mean"]),
                _format_float(summary["num_steps_total"]["mean"]),
                _format_float(summary["num_subtrajectories"]["mean"]),
                _format_float(summary["max_subtrajectory_depth"]["mean"]),
                _format_float(summary["wall_time_seconds"]["mean"]),
                _format_float(summary["total_tokens"]["mean"]),
            ]
        )
    return rows


def _delta_rows(left: AnalyzedRun, right: AnalyzedRun) -> list[list[str]]:
    rows = []
    metric_paths = [
        ("success@k", ("success_at_k",), True),
        ("reward@k", ("reward_at_k_mean",), False),
        ("avg_steps", ("num_steps_total", "mean"), False),
        ("avg_root_steps", ("num_steps_root", "mean"), False),
        ("avg_subtrajs", ("num_subtrajectories", "mean"), False),
        ("avg_depth", ("max_subtrajectory_depth", "mean"), False),
        ("avg_time_s", ("wall_time_seconds", "mean"), False),
        ("avg_total_tokens", ("total_tokens", "mean"), False),
    ]
    for metric_name, path, is_percent in metric_paths:
        left_value = _nested_get(left.summary, path)
        right_value = _nested_get(right.summary, path)
        delta = left_value - right_value
        rows.append(
            [
                metric_name,
                _format_percent(left_value) if is_percent else _format_float(left_value),
                _format_percent(right_value) if is_percent else _format_float(right_value),
                _format_delta(100.0 * delta, 1) + " pp" if is_percent else _format_delta(delta),
            ]
        )
    return rows


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def print_text_report(results: list[AnalyzedRun], comparison: dict[str, Any] | None) -> None:
    print("Run summaries:")
    print(
        _render_table(
            [
                "run",
                "tasks",
                "success@k",
                "reward@k",
                "avg_steps",
                "avg_subtrajs",
                "avg_depth",
                "avg_time_s",
                "avg_total_tokens",
            ],
            _summary_rows(results),
        )
    )

    if comparison is None:
        return

    left, right = results
    print(f"\nDeltas ({left.label} - {right.label}):")
    print(_render_table(["metric", left.label, right.label, "delta"], _delta_rows(left, right)))

    print("\nCommon-task outcomes:")
    print(
        _render_table(
            ["bucket", "count", "task_ids"],
            [
                [
                    "both_success",
                    str(len(comparison["both_success_task_ids"])),
                    _compact_ids(comparison["both_success_task_ids"]),
                ],
                [
                    f"{left.label}_only_success",
                    str(len(comparison["left_only_success_task_ids"])),
                    _compact_ids(comparison["left_only_success_task_ids"]),
                ],
                [
                    f"{right.label}_only_success",
                    str(len(comparison["right_only_success_task_ids"])),
                    _compact_ids(comparison["right_only_success_task_ids"]),
                ],
                [
                    "neither_success",
                    str(len(comparison["neither_success_task_ids"])),
                    _compact_ids(comparison["neither_success_task_ids"]),
                ],
            ],
        )
    )

    if comparison["left_missing_task_ids"] or comparison["right_missing_task_ids"]:
        print("\nMissing tasks:")
        print(f"  Missing from {left.label}: {_compact_ids(comparison['left_missing_task_ids'])}")
        print(f"  Missing from {right.label}: {_compact_ids(comparison['right_missing_task_ids'])}")


def _compact_ids(task_ids: list[str], limit: int = 12) -> str:
    if not task_ids:
        return "-"
    rendered = ", ".join(task_ids[:limit])
    if len(task_ids) > limit:
        rendered += f", ... (+{len(task_ids) - limit})"
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DeepDive inference results across runs.")
    parser.add_argument(
        "result_paths",
        nargs="+",
        help="Result directories or their rollouts subdirectories.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels matching result_paths.",
    )
    parser.add_argument("--json-out", default=None, help="Optional path to write summary JSON.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of tables.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.result_paths):
        raise ValueError("--labels must have the same number of values as result_paths")

    analyzed_results = [
        analyze_result_path(Path(path), label=args.labels[index] if args.labels else None)
        for index, path in enumerate(args.result_paths)
    ]
    comparison = (
        build_pairwise_comparison(analyzed_results[0], analyzed_results[1])
        if len(analyzed_results) == 2
        else None
    )
    payload = {
        "results": [result.to_dict() for result in analyzed_results],
        "pairwise_comparison": comparison,
    }

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(analyzed_results, comparison)


if __name__ == "__main__":
    main()
