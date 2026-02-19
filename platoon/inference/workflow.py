from __future__ import annotations

import asyncio
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.inference.config_defs import InferenceWorkflowConfig

logger = logging.getLogger(__name__)

RewardProcessor = Callable[[dict[str, Any]], tuple[float, dict[str, float]]]
SuccessFn = Callable[[dict[str, Any], float, dict[str, float]], bool]
WorkflowMetricsFn = Callable[[dict[str, Any]], dict[str, float]]


@dataclass
class InferenceRolloutRecord:
    task_id: str
    rollout_index: int
    success: bool
    reward: float
    reward_components: dict[str, float]
    num_steps_total: int
    num_steps_root: int
    num_steps_subtrajectories: int
    num_subtrajectories: int
    subtrajectory_depth_counts: dict[str, int] = field(default_factory=dict)
    subtrajectory_depth_steps: dict[str, int] = field(default_factory=dict)
    wall_time_seconds: float | None = None
    workflow_metrics: dict[str, float] = field(default_factory=dict)
    source_path: str | None = None
    error: str | None = None
    trajectory_collection: dict[str, Any] | None = None


class InferenceWorkflow(Protocol):
    config: InferenceWorkflowConfig

    async def arun_rollout(self, data: dict[str, Any], rollout_index: int, output_dir: str) -> InferenceRolloutRecord: ...

    def record_from_collection(
        self,
        task_id: str,
        rollout_index: int,
        trajectory_collection: dict[str, Any],
        wall_time_seconds: float | None = None,
        source_path: str | None = None,
    ) -> InferenceRolloutRecord: ...


def _get_root_trajectory(collection: dict[str, Any]) -> dict[str, Any] | None:
    trajectories = collection.get("trajectories")
    if not isinstance(trajectories, dict) or not trajectories:
        return None
    root = next(iter(trajectories.values()))
    return root if isinstance(root, dict) else None


def _count_steps(traj: dict[str, Any]) -> int:
    steps = traj.get("steps")
    return len(steps) if isinstance(steps, list) else 0


def _default_success_fn_factory(success_threshold: float) -> SuccessFn:
    def _default_success_fn(collection: dict[str, Any], root_reward: float, reward_components: dict[str, float]) -> bool:
        del reward_components  # intentionally unused in default logic
        root = _get_root_trajectory(collection)
        if root is not None:
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
        return root_reward >= success_threshold

    return _default_success_fn


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


class DefaultInferenceGroupWorkflow:
    """Default rollout workflow for inference benchmarking."""

    def __init__(
        self,
        rollout_fn: Callable[[Task, RolloutConfig], Any],
        get_task_fn: Callable[[str], Task],
        config: InferenceWorkflowConfig,
        model_name: str,
        model_endpoint: str | None = None,
        model_api_key: str | None = None,
        reward_processor: RewardProcessor = lambda traj: (float(traj.get("reward", 0.0)), {}),
        success_fn: SuccessFn | None = None,
        workflow_metrics_fn: WorkflowMetricsFn | None = None,
    ):
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.config = config
        self.model_name = model_name
        self.model_endpoint = model_endpoint
        self.model_api_key = model_api_key
        self.reward_processor = reward_processor
        self.success_fn = success_fn or _default_success_fn_factory(config.success_threshold)
        self.workflow_metrics_fn = workflow_metrics_fn
        self._process_pool: ProcessPoolExecutor | None = None

    def _get_rollout_config(self, output_dir: str) -> RolloutConfig:
        rollout_config = deepcopy(self.config.rollout_config)
        rollout_config.model_name = self.model_name
        rollout_config.model_endpoint = self.model_endpoint
        rollout_config.model_api_key = self.model_api_key
        rollout_config.return_dict = True
        rollout_config.train = False
        rollout_config.output_dir = output_dir
        return rollout_config

    def record_from_collection(
        self,
        task_id: str,
        rollout_index: int,
        trajectory_collection: dict[str, Any],
        wall_time_seconds: float | None = None,
        source_path: str | None = None,
    ) -> InferenceRolloutRecord:
        root = _get_root_trajectory(trajectory_collection)
        if root is None:
            raise ValueError("No root trajectory found in trajectory collection")

        reward, reward_components = self.reward_processor(root)
        success = self.success_fn(trajectory_collection, reward, reward_components)
        num_steps_total, num_steps_root, num_steps_sub, depth_counts, depth_steps = _compute_depth_stats(
            trajectory_collection
        )

        workflow_metrics = {}
        if self.workflow_metrics_fn is not None:
            try:
                workflow_metrics = self.workflow_metrics_fn(trajectory_collection)
            except Exception:
                logger.exception("workflow_metrics_fn failed for task %s rollout %d", task_id, rollout_index)
                workflow_metrics = {}

        return InferenceRolloutRecord(
            task_id=task_id,
            rollout_index=rollout_index,
            success=success,
            reward=reward,
            reward_components=reward_components,
            num_steps_total=num_steps_total,
            num_steps_root=num_steps_root,
            num_steps_subtrajectories=num_steps_sub,
            num_subtrajectories=sum(depth_counts.values()),
            subtrajectory_depth_counts=depth_counts,
            subtrajectory_depth_steps=depth_steps,
            wall_time_seconds=wall_time_seconds,
            workflow_metrics=workflow_metrics,
            source_path=source_path,
            trajectory_collection=trajectory_collection,
        )

    def _get_or_create_process_pool(self) -> ProcessPoolExecutor:
        if self._process_pool is None:
            max_workers = self.config.subprocess_max_workers or self.config.num_concurrent_workers
            self._process_pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn"))
        return self._process_pool

    async def arun_rollout(self, data: dict[str, Any], rollout_index: int, output_dir: str) -> InferenceRolloutRecord:
        task_id = str(data["task_id"])
        start = time.perf_counter()
        try:
            rollout_config = self._get_rollout_config(output_dir=output_dir)
            if self.config.use_subprocesses:
                from platoon.inference.subprocess_worker import run_rollout_subprocess

                loop = asyncio.get_event_loop()
                pool = self._get_or_create_process_pool()
                result = await loop.run_in_executor(
                    pool,
                    run_rollout_subprocess,
                    self.rollout_fn.__module__,
                    self.rollout_fn.__name__,
                    self.get_task_fn.__module__,
                    self.get_task_fn.__name__,
                    task_id,
                    asdict(rollout_config),
                )
            else:
                task = self.get_task_fn(task_id)
                if rollout_config.max_steps is not None:
                    task.max_steps = rollout_config.max_steps
                result = await asyncio.create_task(self.rollout_fn(task, rollout_config))
                if hasattr(result, "to_dict"):
                    result = result.to_dict()

            if not isinstance(result, dict):
                raise TypeError(f"Expected rollout result dict, got {type(result)}")

            return self.record_from_collection(
                task_id=task_id,
                rollout_index=rollout_index,
                trajectory_collection=result,
                wall_time_seconds=time.perf_counter() - start,
            )
        except Exception as e:
            logger.exception("Inference rollout failed for task %s rollout %d: %s", task_id, rollout_index, e)
            return InferenceRolloutRecord(
                task_id=task_id,
                rollout_index=rollout_index,
                success=False,
                reward=0.0,
                reward_components={},
                num_steps_total=0,
                num_steps_root=0,
                num_steps_subtrajectories=0,
                num_subtrajectories=0,
                wall_time_seconds=time.perf_counter() - start,
                error=str(e),
            )

    async def aclose(self) -> None:
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self._process_pool = None
