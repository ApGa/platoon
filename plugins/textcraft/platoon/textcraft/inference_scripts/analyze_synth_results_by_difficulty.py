"""Analyze TextCraft-Synth inference results stratified by task difficulty.

Usage:
    python platoon/textcraft/inference_scripts/analyze_synth_results_by_difficulty.py \
        /path/to/result_dir_a \
        /path/to/result_dir_b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from platoon.textcraft.synth_tasks import get_synth_task


DIFFICULTY_ORDER = ["overall", "easy", "medium", "hard", "extreme", "unknown"]


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
    prompt_token_stats: RunningStats = field(default_factory=RunningStats)
    prompt_token_stats_success: RunningStats = field(default_factory=RunningStats)
    prompt_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    completion_token_stats: RunningStats = field(default_factory=RunningStats)
    completion_token_stats_success: RunningStats = field(default_factory=RunningStats)
    completion_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    total_token_stats: RunningStats = field(default_factory=RunningStats)
    total_token_stats_success: RunningStats = field(default_factory=RunningStats)
    total_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    uncached_prompt_token_stats: RunningStats = field(default_factory=RunningStats)
    uncached_prompt_token_stats_success: RunningStats = field(default_factory=RunningStats)
    uncached_prompt_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    cache_read_prompt_token_stats: RunningStats = field(default_factory=RunningStats)
    cache_read_prompt_token_stats_success: RunningStats = field(default_factory=RunningStats)
    cache_read_prompt_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    cache_write_completion_token_stats: RunningStats = field(default_factory=RunningStats)
    cache_write_completion_token_stats_success: RunningStats = field(default_factory=RunningStats)
    cache_write_completion_token_stats_failure: RunningStats = field(default_factory=RunningStats)
    max_depth_stats: RunningStats = field(default_factory=RunningStats)
    craft_step_stats: RunningStats = field(default_factory=RunningStats)

    def add_task(self, task_record: dict[str, Any], metadata: dict[str, Any]) -> None:
        self.task_count += 1
        self.task_success_at_k_sum += float(task_record.get("success_at_k", 0.0))
        self.task_reward_at_k_mean_sum += float(task_record.get("reward_at_k_mean", 0.0))
        self.task_reward_at_k_max_sum += float(task_record.get("reward_at_k_max", 0.0))
        self.task_reward_at_k_min_sum += float(task_record.get("reward_at_k_min", 0.0))
        self.max_depth_stats.add(_maybe_float(metadata.get("max_depth")))
        self.craft_step_stats.add(_maybe_float(metadata.get("num_craft_steps")))

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
            token_usage = _extract_rollout_token_usage(rollout)

            self.reward_stats.add(reward)
            self.step_stats.add(num_steps)
            self.wall_time_stats.add(wall_time)
            self.prompt_token_stats.add(token_usage["prompt_tokens"])
            self.completion_token_stats.add(token_usage["completion_tokens"])
            self.total_token_stats.add(token_usage["total_tokens"])
            self.uncached_prompt_token_stats.add(token_usage["uncached_prompt_tokens"])
            self.cache_read_prompt_token_stats.add(token_usage["cache_read_prompt_tokens"])
            self.cache_write_completion_token_stats.add(token_usage["cache_write_completion_tokens"])

            if success:
                self.successful_rollouts += 1
                self.step_stats_success.add(num_steps)
                self.wall_time_stats_success.add(wall_time)
                self.prompt_token_stats_success.add(token_usage["prompt_tokens"])
                self.completion_token_stats_success.add(token_usage["completion_tokens"])
                self.total_token_stats_success.add(token_usage["total_tokens"])
                self.uncached_prompt_token_stats_success.add(token_usage["uncached_prompt_tokens"])
                self.cache_read_prompt_token_stats_success.add(token_usage["cache_read_prompt_tokens"])
                self.cache_write_completion_token_stats_success.add(token_usage["cache_write_completion_tokens"])
            else:
                self.step_stats_failure.add(num_steps)
                self.wall_time_stats_failure.add(wall_time)
                self.prompt_token_stats_failure.add(token_usage["prompt_tokens"])
                self.completion_token_stats_failure.add(token_usage["completion_tokens"])
                self.total_token_stats_failure.add(token_usage["total_tokens"])
                self.uncached_prompt_token_stats_failure.add(token_usage["uncached_prompt_tokens"])
                self.cache_read_prompt_token_stats_failure.add(token_usage["cache_read_prompt_tokens"])
                self.cache_write_completion_token_stats_failure.add(token_usage["cache_write_completion_tokens"])

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
                "prompt_tokens": {
                    "overall": self.prompt_token_stats.to_dict(),
                    "success": self.prompt_token_stats_success.to_dict(),
                    "failure": self.prompt_token_stats_failure.to_dict(),
                },
                "completion_tokens": {
                    "overall": self.completion_token_stats.to_dict(),
                    "success": self.completion_token_stats_success.to_dict(),
                    "failure": self.completion_token_stats_failure.to_dict(),
                },
                "total_tokens": {
                    "overall": self.total_token_stats.to_dict(),
                    "success": self.total_token_stats_success.to_dict(),
                    "failure": self.total_token_stats_failure.to_dict(),
                },
                "uncached_prompt_tokens_est": {
                    "overall": self.uncached_prompt_token_stats.to_dict(),
                    "success": self.uncached_prompt_token_stats_success.to_dict(),
                    "failure": self.uncached_prompt_token_stats_failure.to_dict(),
                },
                "cache_read_prompt_tokens_est": {
                    "overall": self.cache_read_prompt_token_stats.to_dict(),
                    "success": self.cache_read_prompt_token_stats_success.to_dict(),
                    "failure": self.cache_read_prompt_token_stats_failure.to_dict(),
                },
                "cache_write_completion_tokens_est": {
                    "overall": self.cache_write_completion_token_stats.to_dict(),
                    "success": self.cache_write_completion_token_stats_success.to_dict(),
                    "failure": self.cache_write_completion_token_stats_failure.to_dict(),
                },
                "task_metadata": {
                    "max_depth": self.max_depth_stats.to_dict(),
                    "num_craft_steps": self.craft_step_stats.to_dict(),
                },
            },
        }


@dataclass
class AnalyzedRun:
    label: str
    result_dir: str
    difficulty_metrics: dict[str, Any]
    success_task_ids: set[str]
    task_records_by_id: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "result_dir": self.result_dir,
            "difficulty_metrics": self.difficulty_metrics,
        }


def _maybe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _extract_rollout_token_usage(rollout: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    uncached_prompt_tokens = 0.0
    cache_read_prompt_tokens = 0.0
    cache_write_completion_tokens = 0.0

    trajectory_collection = rollout.get("trajectory_collection")
    if not isinstance(trajectory_collection, dict):
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "cache_read_prompt_tokens": cache_read_prompt_tokens,
            "cache_write_completion_tokens": cache_write_completion_tokens,
        }

    trajectories = trajectory_collection.get("trajectories")
    if not isinstance(trajectories, dict):
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "uncached_prompt_tokens": uncached_prompt_tokens,
            "cache_read_prompt_tokens": cache_read_prompt_tokens,
            "cache_write_completion_tokens": cache_write_completion_tokens,
        }

    for trajectory in trajectories.values():
        if not isinstance(trajectory, dict):
            continue
        steps = trajectory.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step_index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            misc = step.get("misc")
            if not isinstance(misc, dict):
                continue
            action_misc = misc.get("action_misc")
            if not isinstance(action_misc, dict):
                continue
            usage = action_misc.get("usage")
            if not isinstance(usage, dict):
                continue

            step_prompt_tokens = _maybe_float(usage.get("prompt_tokens")) or 0.0
            step_completion_tokens = _maybe_float(usage.get("completion_tokens")) or 0.0
            step_total_tokens = _maybe_float(usage.get("total_tokens"))

            prompt_tokens += step_prompt_tokens
            completion_tokens += step_completion_tokens
            if step_index == 0:
                uncached_prompt_tokens += step_prompt_tokens
            else:
                cache_read_prompt_tokens += step_prompt_tokens
            cache_write_completion_tokens += step_completion_tokens
            total_tokens += (
                step_total_tokens
                if step_total_tokens is not None
                else step_prompt_tokens + step_completion_tokens
            )

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "uncached_prompt_tokens": uncached_prompt_tokens,
        "cache_read_prompt_tokens": cache_read_prompt_tokens,
        "cache_write_completion_tokens": cache_write_completion_tokens,
    }


def _extract_task_metadata(task_record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    for rollout in task_record.get("rollouts", []):
        trajectory_collection = rollout.get("trajectory_collection")
        if not isinstance(trajectory_collection, dict):
            continue
        trajectories = trajectory_collection.get("trajectories")
        if not isinstance(trajectories, dict) or not trajectories:
            continue

        first_trajectory = next(iter(trajectories.values()))
        if not isinstance(first_trajectory, dict):
            continue
        task = first_trajectory.get("task")
        if not isinstance(task, dict):
            continue
        misc = task.get("misc")
        if not isinstance(misc, dict):
            continue

        metadata["difficulty"] = misc.get("difficulty")
        metadata["max_depth"] = misc.get("max_depth")
        metadata["num_craft_steps"] = misc.get("num_craft_steps")
        break

    if metadata.get("difficulty"):
        return metadata

    task_id = task_record.get("task_id")
    if not isinstance(task_id, str):
        metadata["difficulty"] = "unknown"
        return metadata

    try:
        task = get_synth_task(task_id)
        misc = task.misc if isinstance(task.misc, dict) else {}
        metadata["difficulty"] = misc.get("difficulty", "unknown")
        metadata["max_depth"] = misc.get("max_depth")
        metadata["num_craft_steps"] = misc.get("num_craft_steps")
    except Exception:
        metadata["difficulty"] = "unknown"

    return metadata


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
                    raise ValueError(f"Failed to parse {task_results_path} line {line_number}: {exc}") from exc
                if isinstance(record, dict):
                    yield record
        return

    final_report_path = result_dir / "reports" / "final_report.json"
    if not final_report_path.exists():
        raise FileNotFoundError(
            f"Could not find either {task_results_path} or {final_report_path} for result dir {result_dir}"
        )

    with final_report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    for record in report.get("tasks", []):
        if isinstance(record, dict):
            yield record


def _summarize_task_records(task_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    buckets = {difficulty: BucketAccumulator() for difficulty in DIFFICULTY_ORDER}

    for task_record in task_records:
        metadata = _extract_task_metadata(task_record)
        difficulty = str(metadata.get("difficulty") or "unknown").lower()
        if difficulty not in buckets:
            difficulty = "unknown"

        buckets["overall"].add_task(task_record, metadata)
        buckets[difficulty].add_task(task_record, metadata)

    summaries = {
        difficulty: accumulator.to_summary()
        for difficulty, accumulator in buckets.items()
        if difficulty == "overall" or accumulator.task_count > 0
    }

    return summaries


def analyze_result_dir(result_dir: Path) -> AnalyzedRun:
    task_records = list(_iter_task_records(result_dir))
    task_records_by_id = {
        record["task_id"]: record for record in task_records if isinstance(record.get("task_id"), str)
    }
    success_task_ids = {
        task_id
        for task_id, record in task_records_by_id.items()
        if float(record.get("success_at_k", 0.0)) > 0.0
    }

    return AnalyzedRun(
        label=result_dir.name,
        result_dir=str(result_dir),
        difficulty_metrics=_summarize_task_records(task_records),
        success_task_ids=success_task_ids,
        task_records_by_id=task_records_by_id,
    )


def _build_common_success_payload(results: list[AnalyzedRun]) -> dict[str, Any] | None:
    if len(results) < 2:
        return None

    common_success_task_ids = set.intersection(*(result.success_task_ids for result in results))
    if not common_success_task_ids:
        return {
            "task_ids": [],
            "task_count": 0,
            "results": [result.to_dict() | {"difficulty_metrics": {"overall": BucketAccumulator().to_summary()}} for result in results],
        }

    restricted_results = []
    for result in results:
        restricted_task_records = [
            result.task_records_by_id[task_id] for task_id in sorted(common_success_task_ids) if task_id in result.task_records_by_id
        ]
        restricted_results.append(
            {
                "label": result.label,
                "result_dir": result.result_dir,
                "difficulty_metrics": _summarize_task_records(restricted_task_records),
            }
        )

    return {
        "task_ids": sorted(common_success_task_ids),
        "task_count": len(common_success_task_ids),
        "results": restricted_results,
    }


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


def _difficulty_rows(result: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for difficulty in DIFFICULTY_ORDER:
        metrics = result["difficulty_metrics"].get(difficulty)
        if not metrics:
            continue
        summary = metrics["summary"]
        stats = metrics["stats"]
        rows.append(
            [
                difficulty,
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


def _step_time_rows(result: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for difficulty in DIFFICULTY_ORDER:
        metrics = result["difficulty_metrics"].get(difficulty)
        if not metrics:
            continue
        stats = metrics["stats"]
        step_stats = stats["num_steps_total"]
        time_stats = stats["rollout_wall_time_seconds"]
        rows.append(
            [
                difficulty,
                str(step_stats["success"]["count"]),
                _format_float(step_stats["success"]["mean"]),
                _format_float(step_stats["failure"]["mean"]),
                _format_float(time_stats["success"]["mean"]),
                _format_float(time_stats["failure"]["mean"]),
            ]
        )
    return rows


def _token_rows(result: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for difficulty in DIFFICULTY_ORDER:
        metrics = result["difficulty_metrics"].get(difficulty)
        if not metrics:
            continue
        stats = metrics["stats"]
        prompt_stats = stats["prompt_tokens"]
        completion_stats = stats["completion_tokens"]
        total_stats = stats["total_tokens"]
        rows.append(
            [
                difficulty,
                _format_float(prompt_stats["overall"]["mean"]),
                _format_float(completion_stats["overall"]["mean"]),
                _format_float(total_stats["overall"]["mean"]),
                _format_float(prompt_stats["success"]["mean"]),
                _format_float(completion_stats["success"]["mean"]),
                _format_float(total_stats["success"]["mean"]),
                _format_float(prompt_stats["failure"]["mean"]),
                _format_float(completion_stats["failure"]["mean"]),
                _format_float(total_stats["failure"]["mean"]),
            ]
        )
    return rows


def _kv_cache_rows(result: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for difficulty in DIFFICULTY_ORDER:
        metrics = result["difficulty_metrics"].get(difficulty)
        if not metrics:
            continue
        stats = metrics["stats"]
        uncached_stats = stats["uncached_prompt_tokens_est"]
        cache_read_stats = stats["cache_read_prompt_tokens_est"]
        cache_write_stats = stats["cache_write_completion_tokens_est"]
        rows.append(
            [
                difficulty,
                _format_float(uncached_stats["overall"]["mean"]),
                _format_float(cache_read_stats["overall"]["mean"]),
                _format_float(cache_write_stats["overall"]["mean"]),
                _format_float(cache_read_stats["success"]["mean"]),
                _format_float(cache_read_stats["failure"]["mean"]),
            ]
        )
    return rows


def print_text_report(results: list[dict[str, Any]]) -> None:
    headers = ["difficulty", "tasks", "success@k", "success", "reward@k", "reward", "avg_steps", "avg_time_s"]
    step_time_headers = [
        "difficulty",
        "successes",
        "steps_success",
        "steps_failure",
        "time_success_s",
        "time_failure_s",
    ]
    token_headers = [
        "difficulty",
        "prompt_avg",
        "completion_avg",
        "total_avg",
        "prompt_success",
        "completion_success",
        "total_success",
        "prompt_failure",
        "completion_failure",
        "total_failure",
    ]
    kv_cache_headers = [
        "difficulty",
        "uncached_input_est",
        "cache_read_input_est",
        "cache_write_output_est",
        "cache_read_success",
        "cache_read_failure",
    ]

    for result in results:
        print(f"\nRun: {result['label']}")
        print(result["result_dir"])
        print(_render_table(headers, _difficulty_rows(result)))
        print()
        print(_render_table(step_time_headers, _step_time_rows(result)))
        print()
        print(_render_table(token_headers, _token_rows(result)))
        print()
        print(_render_table(kv_cache_headers, _kv_cache_rows(result)))

    comparison_rows: list[list[str]] = []
    step_time_comparison_rows: list[list[str]] = []
    token_comparison_rows: list[list[str]] = []
    kv_cache_comparison_rows: list[list[str]] = []
    for result in results:
        for difficulty in DIFFICULTY_ORDER:
            metrics = result["difficulty_metrics"].get(difficulty)
            if not metrics:
                continue
            summary = metrics["summary"]
            stats = metrics["stats"]
            comparison_rows.append(
                [
                    result["label"],
                    difficulty,
                    str(summary["total_tasks"]),
                    _format_percent(summary["success_at_k"]),
                    _format_percent(summary["success_rate"]),
                    _format_float(summary["reward_at_k_mean"]),
                    _format_float(stats["num_steps_total"]["overall"]["mean"]),
                    _format_float(stats["rollout_wall_time_seconds"]["overall"]["mean"]),
                ]
            )
            step_time_comparison_rows.append(
                [
                    result["label"],
                    difficulty,
                    _format_float(stats["num_steps_total"]["success"]["mean"]),
                    _format_float(stats["num_steps_total"]["failure"]["mean"]),
                    _format_float(stats["rollout_wall_time_seconds"]["success"]["mean"]),
                    _format_float(stats["rollout_wall_time_seconds"]["failure"]["mean"]),
                ]
            )
            token_comparison_rows.append(
                [
                    result["label"],
                    difficulty,
                    _format_float(stats["prompt_tokens"]["overall"]["mean"]),
                    _format_float(stats["completion_tokens"]["overall"]["mean"]),
                    _format_float(stats["total_tokens"]["overall"]["mean"]),
                    _format_float(stats["prompt_tokens"]["success"]["mean"]),
                    _format_float(stats["completion_tokens"]["success"]["mean"]),
                    _format_float(stats["total_tokens"]["success"]["mean"]),
                    _format_float(stats["prompt_tokens"]["failure"]["mean"]),
                    _format_float(stats["completion_tokens"]["failure"]["mean"]),
                    _format_float(stats["total_tokens"]["failure"]["mean"]),
                ]
            )
            kv_cache_comparison_rows.append(
                [
                    result["label"],
                    difficulty,
                    _format_float(stats["uncached_prompt_tokens_est"]["overall"]["mean"]),
                    _format_float(stats["cache_read_prompt_tokens_est"]["overall"]["mean"]),
                    _format_float(stats["cache_write_completion_tokens_est"]["overall"]["mean"]),
                    _format_float(stats["cache_read_prompt_tokens_est"]["success"]["mean"]),
                    _format_float(stats["cache_read_prompt_tokens_est"]["failure"]["mean"]),
                ]
            )

    if comparison_rows:
        print("\nComparison:")
        print(
            _render_table(
                ["run", "difficulty", "tasks", "success@k", "success", "reward@k", "avg_steps", "avg_time_s"],
                comparison_rows,
            )
        )
        print()
        print(
            _render_table(
                ["run", "difficulty", "steps_success", "steps_failure", "time_success_s", "time_failure_s"],
                step_time_comparison_rows,
            )
        )
        print()
        print(
            _render_table(
                [
                    "run",
                    "difficulty",
                    "prompt_avg",
                    "completion_avg",
                    "total_avg",
                    "prompt_success",
                    "completion_success",
                    "total_success",
                    "prompt_failure",
                    "completion_failure",
                    "total_failure",
                ],
                token_comparison_rows,
            )
        )
        print()
        print(
            _render_table(
                [
                    "run",
                    "difficulty",
                    "uncached_input_est",
                    "cache_read_input_est",
                    "cache_write_output_est",
                    "cache_read_success",
                    "cache_read_failure",
                ],
                kv_cache_comparison_rows,
            )
        )


def print_common_success_report(common_success_payload: dict[str, Any]) -> None:
    print(f"\nCommon-success task intersection: {common_success_payload['task_count']} tasks")
    if common_success_payload["task_count"] == 0:
        return
    print_text_report(common_success_payload["results"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze TextCraft-Synth inference results stratified by task difficulty."
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="One or more inference result directories containing reports/task_results.jsonl or reports/final_report.json",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the machine-readable summary JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON to stdout instead of human-readable tables.",
    )
    args = parser.parse_args()

    analyzed_results = [analyze_result_dir(Path(result_dir).expanduser().resolve()) for result_dir in args.result_dirs]
    serialized_results = [result.to_dict() for result in analyzed_results]
    common_success_payload = _build_common_success_payload(analyzed_results)
    payload = {
        "results": serialized_results,
        "common_success_intersection": common_success_payload,
    }

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_report(serialized_results)
        if common_success_payload is not None:
            print_common_success_report(common_success_payload)


if __name__ == "__main__":
    main()
