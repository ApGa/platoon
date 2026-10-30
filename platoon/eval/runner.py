from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from platoon.eval.accounting import report_blocks
from platoon.eval.config_defs import TinkerKeyRefusedError
from platoon.eval.workflow import EvalRolloutRecord, EvalWorkflow
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.utils.config import ConfigError

logger = logging.getLogger(__name__)

# The key of a rollout's ``metadata.json`` that records the run that made it (``DefaultEvalGroupWorkflow.provenance``).
PROVENANCE_KEY = "provenance"
# What a rollout whose metadata records no provenance was made with: the only backend and judges setting of the
# evaluations that wrote no provenance.
LEGACY_PROVENANCE = {"backend": "tinker", "judges": True}


def _safe_task_id(task_id: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in task_id)


def _stat_bundle(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {"count": len(values), "mean": mean(values), "min": min(values), "max": max(values)}


def _extract_task_id_from_collection(collection: dict[str, Any], fallback: str) -> str:
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return fallback
    root = next(iter(trajectories.values()))
    if not isinstance(root, dict):
        return fallback
    task = root.get("task")
    if isinstance(task, dict):
        task_id = task.get("id")
        if isinstance(task_id, str) and task_id:
            return task_id
    return fallback


def check_provenance(output_dir: str | os.PathLike[str], current: Mapping[str, Any]) -> None:
    """Refuse an output directory whose rollouts were made otherwise than ``current`` says (a run's provenance,
    :func:`platoon.eval.config_defs.run_provenance`): by another policy (another backend or model, a Tinker policy's
    other checkpoint or renderer, a hosted model's other base URL), or with judges on where they are now off (or the
    other way round).  Adding rollouts to it would mix two evaluations in one report, and a report of it
    (``stage: report``) would label the rollouts with this run's settings.  So an evaluation never resumes another
    policy's rollouts: ``platoon eval`` checks before it writes anything into the directory, and the runner again
    before its rollouts and its report.

    Each rollout's ``metadata.json`` records the run that made it (``provenance``); one that records none was made by
    an evaluation that knew only Tinker with judges on (:data:`LEGACY_PROVENANCE`).  One that records no backend counts
    as made without a policy (``backend: None``): since rollouts first recorded their provenance, every run that served
    a policy (:meth:`~platoon.eval.config_defs.EvalPolicy.provenance`) has recorded its backend, and only a run that
    served none left it out, which now records ``backend: None`` itself.  What else only one side records (a renderer,
    which rollouts made before it was recorded leave out) is not compared."""

    output_dir = Path(output_dir)
    rollout_dir = output_dir / "rollouts"
    if not current or not rollout_dir.is_dir():
        return
    for metadata_path in sorted(rollout_dir.glob("**/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        recorded = metadata.get(PROVENANCE_KEY) if isinstance(metadata, dict) else None
        if not isinstance(recorded, dict):
            recorded = LEGACY_PROVENANCE
        elif "backend" not in recorded:  # recorded before a run without a policy said so: it served none
            recorded = {**recorded, "backend": None}
        differences = [
            f"{key} {recorded[key]!r} (now {current[key]!r})"
            for key in sorted(set(recorded) & set(current))
            if recorded[key] != current[key]
        ]
        if differences:
            raise ConfigError(
                f"{output_dir} holds rollouts made with {', '.join(differences)}, as "
                f"{metadata_path.relative_to(output_dir)} records; one report would mix or mislabel them: set a new "
                "inference.output_dir, or leave it unset for a new directory of this evaluation's own, or report them "
                "with the settings that made them"
            )


class EvalRunner:
    """Run a two-stage evaluation: rollout collection, then reporting."""

    def __init__(self, workflow: EvalWorkflow, output_dir: str):
        self.workflow = workflow
        self.output_dir = Path(output_dir)
        self.rollout_dir = self.output_dir / "rollouts"
        self.report_dir = self.output_dir / "reports"
        self._last_rollout_stage_elapsed_seconds: float | None = None
        self._stop_reason: str | None = None  # why no further rollout starts (EvalRolloutRecord.stop_run)
        self._stopped: str | None = None  # why the run stopped its rollouts (stop)

    def stop(self, why: str) -> None:
        """Start no further rollout: those in flight finish, and the rollout stage then fails with ``why`` (a
        ``RuntimeError``), as when what every rollout needs failed (the run's node services)."""

        if self._stopped is None:
            logger.error("Stopping the evaluation after the rollouts in flight: %s", why)
            self._stopped = why

    def _write_rollout_artifacts(self, record: EvalRolloutRecord, collection_path: Path, metadata_path: Path) -> None:
        if record.trajectory_collection is not None:
            with collection_path.open("w", encoding="utf-8") as f:
                json.dump(record.trajectory_collection, f)

        metadata = {
            "task_id": record.task_id,
            "rollout_index": record.rollout_index,
            "source_path": record.source_path or str(collection_path),
            "wall_time_seconds": record.wall_time_seconds,
            "error": record.error,
            "outcome_status": record.outcome_status,
            "reward_key": record.reward_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # An errored rollout is not finished: a resumed run tries it again (see retry_errored).
            "status": "completed" if record.error is None else "errored",
            # Counted from the trajectories, which a report reads again: copies for people and tools that read only
            # this file.
            "usage": record.usage,
            "cost": record.cost,
            "backend": record.backend,
            "judges": record.judges,
            PROVENANCE_KEY: self._provenance(),
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_record_from_artifacts(
        self, task_id: str, rollout_index: int, collection_path: Path, metadata_path: Path
    ) -> EvalRolloutRecord | None:
        if not metadata_path.exists():
            return None
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            return None

        wall_time_seconds = metadata.get("wall_time_seconds")
        error = metadata.get("error")
        # The stored error is the run's own failure when it recorded no outcome_status (the rollout raised), and
        # otherwise the verdict on that outcome, which is derived again from the collection instead, so a report
        # or a resumed run applies the current count_timeout_as_failure to the rollouts on disk too.
        run_error = error if metadata.get("outcome_status") is None else None
        source_path = metadata.get("source_path") or str(collection_path)

        if collection_path.exists():
            try:
                with collection_path.open("r", encoding="utf-8") as f:
                    collection = json.load(f)
                record = self.workflow.record_from_collection(
                    task_id=task_id,
                    rollout_index=rollout_index,
                    trajectory_collection=collection,
                    wall_time_seconds=wall_time_seconds,
                    source_path=source_path,
                )
                record.error = run_error or record.error
                return record
            except Exception as e:
                return EvalRolloutRecord(
                    task_id=task_id,
                    rollout_index=rollout_index,
                    success=False,
                    reward=0.0,
                    reward_components={},
                    num_steps_total=0,
                    num_steps_root=0,
                    num_steps_subtrajectories=0,
                    num_subtrajectories=0,
                    wall_time_seconds=wall_time_seconds,
                    source_path=source_path,
                    error=f"Failed to parse existing rollout: {e}",
                )

        return EvalRolloutRecord(
            task_id=task_id,
            rollout_index=rollout_index,
            success=False,
            reward=0.0,
            reward_components={},
            num_steps_total=0,
            num_steps_root=0,
            num_steps_subtrajectories=0,
            num_subtrajectories=0,
            wall_time_seconds=wall_time_seconds,
            source_path=source_path,
            error=error or "Missing trajectory collection file",
        )

    async def _arun_single_rollout(
        self,
        data: dict[str, Any],
        rollout_index: int,
        resume: bool,
        retry_errored: bool = True,
    ) -> EvalRolloutRecord:
        task_id = str(data["task_id"])
        rollout_dir = self.rollout_dir / _safe_task_id(task_id) / f"rollout_{rollout_index}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        collection_path = rollout_dir / "trajectory_collection.json"
        metadata_path = rollout_dir / "metadata.json"

        if resume:
            resumed_record = self._load_record_from_artifacts(task_id, rollout_index, collection_path, metadata_path)
            if resumed_record is not None and (resumed_record.error is None or not retry_errored):
                return resumed_record

        record = await self.workflow.arun_rollout(
            data=data,
            rollout_index=rollout_index,
            output_dir=str(rollout_dir),
        )
        if record.source_path is None:
            record.source_path = str(collection_path)
        self._write_rollout_artifacts(record, collection_path, metadata_path)
        if record.stop_run is not None and self._stop_reason is None:
            logger.error("Stopping the evaluation after the rollouts in flight: %s", record.stop_run)
            self._stop_reason = record.stop_run
        return record

    def _provenance(self) -> dict[str, Any]:
        provenance = getattr(self.workflow, "provenance", None)
        return dict(provenance) if isinstance(provenance, dict) else {}

    def check_provenance(self) -> None:
        """Refuse an output directory whose rollouts were made otherwise than this run says
        (:func:`check_provenance`)."""

        check_provenance(self.output_dir, self._provenance())

    async def run_rollout_stage(
        self, dataset: list[dict[str, Any]], resume: bool = True, retry_errored: bool = True
    ) -> dict[str, Any]:
        """Collect every task's rollouts; with ``resume``, reuse the ones already on disk, except errored ones
        when ``retry_errored``."""

        self.check_provenance()
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self.workflow.config.num_concurrent_workers)
        self._stop_reason = None
        refused: list[TinkerKeyRefusedError] = []  # the Tinker service's refusal of the key, which the stage ends with

        async def _guarded_single(data: dict[str, Any], rollout_index: int) -> EvalRolloutRecord | None:
            async with sem:
                if self._stop_reason is not None or self._stopped is not None:
                    return None  # the evaluation is stopping: this rollout does not start
                try:
                    return await self._arun_single_rollout(
                        data=data, rollout_index=rollout_index, resume=resume, retry_errored=retry_errored
                    )
                except TinkerKeyRefusedError as exc:  # the policy's service refused the key: no rollout can run
                    if not refused:
                        refused.append(exc)
                        self._stop_reason = str(exc)
                    return None

        start = time.perf_counter()
        tasks = [
            _guarded_single(data, rollout_index)
            for data in dataset
            for rollout_index in range(self.workflow.config.num_rollouts_per_task)
        ]
        # Every sandbox the rollouts start, here or in a worker process (which carries this process's labels on), is
        # labelled with the evaluation's output directory, so that its sandboxes can be reaped by it.
        run_label = sandbox_cleanup.run_label(self.output_dir)
        logger.info("This evaluation labels its sandboxes %s", run_label)
        with sandbox_cleanup.labelled(run_label):
            results = await asyncio.gather(*tasks)
        records = [record for record in results if record is not None]
        self._last_rollout_stage_elapsed_seconds = time.perf_counter() - start

        if refused:  # every other rollout would be refused alike; its one line says why
            raise refused[0]
        if self._stopped is not None:
            raise RuntimeError(
                f"the evaluation stopped after {len(records)} rollout(s), {len(results) - len(records)} not started: "
                f"{self._stopped}"
            )
        if self._stop_reason is not None:
            raise ConfigError(
                f"the evaluation stopped after {len(records)} rollout(s), {len(results) - len(records)} not started: "
                f"{self._stop_reason}"
            )

        if self.workflow.config.fail_fast:
            first_error = next((record.error for record in records if record.error is not None), None)
            if first_error is not None:
                raise RuntimeError(f"Fail-fast enabled, aborting the evaluation: {first_error}")

        return {
            "num_rollouts_requested": len(tasks),
            "num_rollouts_completed": len(records),
            "num_rollouts_with_errors": len([r for r in records if r.error is not None]),
            "elapsed_seconds": self._last_rollout_stage_elapsed_seconds,
        }

    def _iter_dump_objects_from_file(self, path: Path) -> list[dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict) and "trajectories" in obj:
                    objects.append(obj)
            except Exception:
                return objects
        elif suffix == ".jsonl":
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict) and "trajectories" in obj:
                            objects.append(obj)
            except Exception:
                return objects
        return objects

    def _collect_records_from_disk(self) -> list[EvalRolloutRecord]:
        records: list[EvalRolloutRecord] = []
        metadata_files = sorted(self.rollout_dir.glob("**/metadata.json"))
        if metadata_files:
            for metadata_path in metadata_files:
                rollout_dir = metadata_path.parent
                collection_path = rollout_dir / "trajectory_collection.json"
                try:
                    with metadata_path.open("r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = {}
                task_dir = rollout_dir.parent.name
                try:
                    rollout_index = int(metadata.get("rollout_index", rollout_dir.name.replace("rollout_", "")))
                except Exception:
                    rollout_index = 0
                task_id = str(metadata.get("task_id") or task_dir)
                loaded = self._load_record_from_artifacts(task_id, rollout_index, collection_path, metadata_path)
                if loaded is not None:
                    records.append(loaded)
            return records

        # Fallback path for unstructured rollouts: scan for trajectory dump files
        fallback_files = sorted(
            [p for p in self.rollout_dir.glob("**/*") if p.is_file() and p.suffix in {".json", ".jsonl"}]
        )
        rollout_counter_by_task: dict[str, int] = {}
        for path in fallback_files:
            for obj in self._iter_dump_objects_from_file(path):
                task_id = _extract_task_id_from_collection(obj, fallback=path.parent.name or "unknown_task")
                rollout_idx = rollout_counter_by_task.get(task_id, 0)
                rollout_counter_by_task[task_id] = rollout_idx + 1
                try:
                    record = self.workflow.record_from_collection(
                        task_id=task_id,
                        rollout_index=rollout_idx,
                        trajectory_collection=obj,
                        source_path=str(path),
                    )
                    records.append(record)
                except Exception:
                    logger.exception("Failed to parse rollout collection from %s", path)
        return records

    def _aggregate_depth_stats(self, records: list[EvalRolloutRecord]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for split_name, subset in {
            "overall": records,
            "success": [r for r in records if r.error is None and r.success],
            "failure": [r for r in records if r.error is None and not r.success],
        }.items():
            depth_counts: dict[str, int] = {}
            depth_steps: dict[str, int] = {}
            for record in subset:
                for depth, count in record.subtrajectory_depth_counts.items():
                    depth_counts[depth] = depth_counts.get(depth, 0) + count
                for depth, step_count in record.subtrajectory_depth_steps.items():
                    depth_steps[depth] = depth_steps.get(depth, 0) + step_count
            split_stats: dict[str, Any] = {}
            for depth in sorted(set(depth_counts.keys()) | set(depth_steps.keys()), key=lambda x: int(x)):
                total_count = depth_counts.get(depth, 0)
                total_steps = depth_steps.get(depth, 0)
                split_stats[depth] = {
                    "total_subtrajectories": total_count,
                    "total_steps": total_steps,
                    "avg_subtrajectories_per_rollout": (total_count / len(subset)) if subset else 0.0,
                    "avg_steps_per_rollout": (total_steps / len(subset)) if subset else 0.0,
                    "avg_steps_per_subtrajectory": (total_steps / total_count) if total_count else 0.0,
                }
            out[split_name] = split_stats
        return out

    def _aggregate_workflow_metrics(self, records: list[EvalRolloutRecord]) -> dict[str, Any]:
        metric_keys: set[str] = set()
        for r in records:
            metric_keys.update(k for k, v in r.workflow_metrics.items() if isinstance(v, (float, int)))

        out: dict[str, Any] = {}
        for key in sorted(metric_keys):
            values_all = [
                float(r.workflow_metrics[key]) for r in records if key in r.workflow_metrics and r.error is None
            ]
            values_success = [
                float(r.workflow_metrics[key])
                for r in records
                if key in r.workflow_metrics and r.error is None and r.success
            ]
            values_failure = [
                float(r.workflow_metrics[key])
                for r in records
                if key in r.workflow_metrics and r.error is None and not r.success
            ]
            out[key] = {
                "overall": _stat_bundle(values_all),
                "success": _stat_bundle(values_success),
                "failure": _stat_bundle(values_failure),
            }
        return out

    def _aggregate_reward_components(self, task_rollout_results: dict[str, list[EvalRolloutRecord]]) -> dict[str, Any]:
        # Track only reward/* components to mirror training metrics semantics.
        component_keys: set[str] = set()
        for records in task_rollout_results.values():
            for record in records:
                for key, value in record.reward_components.items():
                    if key.startswith("reward/") and isinstance(value, (float, int)):
                        component_keys.add(key)

        out: dict[str, Any] = {}
        for key in sorted(component_keys):
            all_valid_values: list[float] = []
            success_values: list[float] = []
            failure_values: list[float] = []
            at_k_mean_values: list[float] = []
            at_k_max_values: list[float] = []
            at_k_min_values: list[float] = []

            for records in task_rollout_results.values():
                valid_records = [r for r in records if r.error is None]
                task_values = [float(r.reward_components.get(key, 0.0)) for r in valid_records]
                if task_values:
                    at_k_mean_values.append(mean(task_values))
                    at_k_max_values.append(max(task_values))
                    at_k_min_values.append(min(task_values))

                for record in valid_records:
                    v = float(record.reward_components.get(key, 0.0))
                    all_valid_values.append(v)
                    if record.success:
                        success_values.append(v)
                    else:
                        failure_values.append(v)

            out[key] = {
                "overall": _stat_bundle(all_valid_values),
                "success": _stat_bundle(success_values),
                "failure": _stat_bundle(failure_values),
                "at_k_mean": _stat_bundle(at_k_mean_values),
                "at_k_max": _stat_bundle(at_k_max_values),
                "at_k_min": _stat_bundle(at_k_min_values),
            }
        return out

    def _build_report(
        self,
        task_rollout_results: dict[str, list[EvalRolloutRecord]],
        total_elapsed_seconds: float,
    ) -> dict[str, Any]:
        all_rollouts = [r for records in task_rollout_results.values() for r in records]
        valid_rollouts = [r for r in all_rollouts if r.error is None]
        successful_rollouts = [r for r in valid_rollouts if r.success]
        failed_rollouts = [r for r in valid_rollouts if not r.success]
        errored_rollouts = [r for r in all_rollouts if r.error is not None]

        task_summaries: list[dict[str, Any]] = []
        for task_id, rollouts in task_rollout_results.items():
            task_valid_rollouts = [r for r in rollouts if r.error is None]
            task_successful_rollouts = [r for r in task_valid_rollouts if r.success]
            max_success = max([1.0 if r.success else 0.0 for r in task_valid_rollouts], default=0.0)
            reward_values = [r.reward for r in task_valid_rollouts]

            task_summaries.append(
                {
                    "task_id": task_id,
                    "has_valid_rollouts": bool(task_valid_rollouts),
                    "success_at_k": max_success,
                    "num_rollouts": len(rollouts),
                    "num_valid_rollouts": len(task_valid_rollouts),
                    "num_failed_rollouts": len(rollouts) - len(task_valid_rollouts),
                    "num_successful_rollouts": len(task_successful_rollouts),
                    "success_rate_within_task": (
                        len(task_successful_rollouts) / len(task_valid_rollouts) if task_valid_rollouts else 0.0
                    ),
                    "reward_at_k_mean": mean(reward_values) if reward_values else 0.0,
                    "reward_at_k_max": max(reward_values, default=0.0),
                    "reward_at_k_min": min(reward_values, default=0.0),
                    "rollouts": [asdict(r) for r in rollouts],
                }
            )

        # Per-task statistics average over the tasks that have at least one measured rollout.
        measured_tasks = [t for t in task_summaries if t["has_valid_rollouts"]]
        task_success_at_k_values = [t["success_at_k"] for t in measured_tasks]
        task_reward_at_k_mean = [t["reward_at_k_mean"] for t in measured_tasks]
        task_reward_at_k_max = [t["reward_at_k_max"] for t in measured_tasks]
        task_reward_at_k_min = [t["reward_at_k_min"] for t in measured_tasks]
        outcome_status_counts: dict[str, int] = {}
        reward_key_counts: dict[str, int] = {}
        for record in all_rollouts:
            key = record.outcome_status or ("none" if record.error is None else "raised")
            outcome_status_counts[key] = outcome_status_counts.get(key, 0) + 1
            if record.reward_key is not None:
                reward_key_counts[record.reward_key] = reward_key_counts.get(record.reward_key, 0) + 1

        rewards_all = [r.reward for r in valid_rollouts]
        steps_total_all = [float(r.num_steps_total) for r in valid_rollouts]
        steps_root_all = [float(r.num_steps_root) for r in valid_rollouts]
        steps_sub_all = [float(r.num_steps_subtrajectories) for r in valid_rollouts]
        wall_time_all = [float(r.wall_time_seconds) for r in valid_rollouts if r.wall_time_seconds is not None]

        steps_total_success = [float(r.num_steps_total) for r in successful_rollouts]
        steps_total_failure = [float(r.num_steps_total) for r in failed_rollouts]
        steps_root_success = [float(r.num_steps_root) for r in successful_rollouts]
        steps_root_failure = [float(r.num_steps_root) for r in failed_rollouts]
        steps_sub_success = [float(r.num_steps_subtrajectories) for r in successful_rollouts]
        steps_sub_failure = [float(r.num_steps_subtrajectories) for r in failed_rollouts]
        wall_time_success = [float(r.wall_time_seconds) for r in successful_rollouts if r.wall_time_seconds is not None]
        wall_time_failure = [float(r.wall_time_seconds) for r in failed_rollouts if r.wall_time_seconds is not None]

        total_tasks = len(task_rollout_results)
        summary = {
            "total_tasks": total_tasks,
            "tasks_without_valid_rollouts": total_tasks - len(measured_tasks),
            "total_rollouts": len(all_rollouts),
            "valid_rollouts": len(valid_rollouts),
            "successful_rollouts": len(successful_rollouts),
            "failed_rollouts": len(failed_rollouts),
            "errored_rollouts": len(errored_rollouts),
            # Each rollout's outcome status; "raised" for a rollout that raised, "none" for one that set none.
            "outcome_status_counts": dict(sorted(outcome_status_counts.items())),
            # The rollouts whose scored outcome trained each reward key: the name its grader reported the reward under.
            "reward_key_counts": dict(sorted(reward_key_counts.items())),
            # Success is averaged over the valid (measured) rollouts, not tasks.
            "success_rate": (len(successful_rollouts) / len(valid_rollouts)) if valid_rollouts else 0.0,
            # success@k is averaged over the tasks with a valid rollout, by the max success of each task's K.
            "success_at_k": mean(task_success_at_k_values) if task_success_at_k_values else 0.0,
            "reward_mean": mean(rewards_all) if rewards_all else 0.0,
            "reward_max": max(rewards_all, default=0.0),
            "reward_min": min(rewards_all, default=0.0),
            "reward_at_k_mean": mean(task_reward_at_k_mean) if task_reward_at_k_mean else 0.0,
            "reward_at_k_max": mean(task_reward_at_k_max) if task_reward_at_k_max else 0.0,
            "reward_at_k_min": mean(task_reward_at_k_min) if task_reward_at_k_min else 0.0,
            "elapsed_seconds": total_elapsed_seconds,
        }

        stats = {
            "num_steps_total": {
                "overall": _stat_bundle(steps_total_all),
                "success": _stat_bundle(steps_total_success),
                "failure": _stat_bundle(steps_total_failure),
            },
            "num_steps_root": {
                "overall": _stat_bundle(steps_root_all),
                "success": _stat_bundle(steps_root_success),
                "failure": _stat_bundle(steps_root_failure),
            },
            "num_steps_subtrajectories": {
                "overall": _stat_bundle(steps_sub_all),
                "success": _stat_bundle(steps_sub_success),
                "failure": _stat_bundle(steps_sub_failure),
            },
            "rollout_wall_time_seconds": {
                "overall": _stat_bundle(wall_time_all),
                "success": _stat_bundle(wall_time_success),
                "failure": _stat_bundle(wall_time_failure),
            },
        }

        rollout_config = getattr(self.workflow.config, "rollout_config", None)
        blocks = report_blocks(all_rollouts, judges_enabled=bool(getattr(rollout_config, "judges", True)))
        policy = getattr(self.workflow, "policy", None)
        provenance = self._provenance()
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "stats": stats,
            "reward_components": self._aggregate_reward_components(task_rollout_results),
            "subtrajectory_stats_by_depth": self._aggregate_depth_stats(valid_rollouts),
            "workflow_specific_metrics": self._aggregate_workflow_metrics(valid_rollouts),
            "workflow_config": asdict(self.workflow.config),
            "model": {
                "model_name": getattr(self.workflow, "model_name", None),
                "model_path": getattr(policy, "model_path", None),
                "renderer_name": getattr(policy, "renderer_name", None),
                "backend": provenance.get("backend"),
                "base_url_origin": provenance.get("base_url_origin"),
                # The models that answered (a dated snapshot, or a router's choice), with their calls: hosted only.
                "served_models": blocks["backend"].pop("served_models"),
            },
            # Model calls by kind (policy, aux, judge, and verifier agents' calls), as totals and per-rollout means.
            "usage": blocks["usage"],
            "cost": blocks["cost"],
            "backend": blocks["backend"],
            "judges": blocks["judges"],
            "tasks": task_summaries,
        }

    def generate_report(self, elapsed_seconds: float | None = None) -> dict[str, Any]:
        self.check_provenance()  # the report labels the rollouts with this run's backend, model and judges
        self.report_dir.mkdir(parents=True, exist_ok=True)
        records = self._collect_records_from_disk()
        by_task: dict[str, list[EvalRolloutRecord]] = {}
        for record in records:
            by_task.setdefault(record.task_id, []).append(record)
        for task_records in by_task.values():
            task_records.sort(key=lambda r: r.rollout_index)

        if elapsed_seconds is None:
            elapsed_seconds = self._last_rollout_stage_elapsed_seconds or 0.0
        report = self._build_report(by_task, elapsed_seconds)

        task_results_path = self.report_dir / "task_results.jsonl"
        with task_results_path.open("w", encoding="utf-8") as f:
            for task in report["tasks"]:
                f.write(json.dumps(task) + "\n")

        report_path = self.report_dir / "final_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report written to %s", report_path)
        return report

    async def arun(
        self,
        dataset: list[dict[str, Any]],
        resume: bool = True,
        run_rollouts: bool = True,
        generate_report: bool = True,
        retry_errored: bool = True,
    ) -> dict[str, Any]:
        try:
            if run_rollouts:
                await self.run_rollout_stage(dataset=dataset, resume=resume, retry_errored=retry_errored)
            if generate_report:
                return self.generate_report()
            return {"message": "Rollout stage completed", "output_dir": str(self.output_dir)}
        finally:
            close_fn = getattr(self.workflow, "aclose", None)
            if callable(close_fn):
                maybe_awaitable = close_fn()
                if asyncio.iscoroutine(maybe_awaitable):
                    await maybe_awaitable
