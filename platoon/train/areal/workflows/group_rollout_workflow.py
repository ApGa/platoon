"""Group-wise AReaL rollout workflow for Platoon training."""

import asyncio
import logging
import multiprocessing as mp
import os
import uuid
from copy import deepcopy
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Callable

import torch
from areal.api import InferenceEngine, RolloutWorkflow
from areal.infra import workflow_context
from areal.utils.dynamic_import import import_from_string
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors

from platoon.envs.base import Task
from platoon.train.areal.config_defs import WorkflowConfig
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.workflow_serialization import RemoteWorkflowSerializable, callable_import_path
from platoon.utils.areal_data_processing import get_train_data_for_trajectory_collection

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class GroupRolloutWorkflow(RolloutWorkflow, RemoteWorkflowSerializable):
    """Workflow that preserves Platoon's recursive group-rollout processing."""

    def __init__(
        self,
        rollout_fn: Callable[[Task, dict], dict] | str,
        get_task_fn: Callable[[str], Task] | str,
        config: WorkflowConfig | dict[str, Any],
        proxy_base_url: str | None,
        proxy_admin_api_key: str,
        output_subdir: str = "rollout",
        filter_errors: bool = False,
        reward_processor: Callable[[dict], tuple[float, dict]] | str = lambda traj: (traj["reward"], {}),
        merge_prefixes: bool = True,
    ):
        if isinstance(config, dict):
            config = WorkflowConfig(**config)
        self.config = deepcopy(config)
        self.config.rollout_config.return_dict = True
        self.config.rollout_config.train = True
        self.proxy_base_url = proxy_base_url
        self.proxy_admin_api_key = proxy_admin_api_key
        self.rollout_fn = self._resolve_callable(rollout_fn)
        self.get_task_fn = self._resolve_callable(get_task_fn)
        self.filter_errors = filter_errors
        self.reward_processor = self._resolve_reward_processor(reward_processor)
        self.merge_prefixes = merge_prefixes
        self.output_subdir = output_subdir

    @staticmethod
    def _resolve_callable(fn: Callable | str) -> Callable:
        if isinstance(fn, str):
            return import_from_string(fn)
        return fn

    @staticmethod
    def _resolve_reward_processor(
        reward_processor: Callable[[dict], tuple[float, dict]] | str,
    ) -> Callable[[dict], tuple[float, dict]]:
        if isinstance(reward_processor, str):
            return import_from_string(reward_processor)
        return reward_processor

    def to_workflow_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "rollout_fn": callable_import_path(self.rollout_fn),
            "get_task_fn": callable_import_path(self.get_task_fn),
            "config": asdict(self.config),
            "proxy_base_url": None,
            "proxy_admin_api_key": self.proxy_admin_api_key,
            "output_subdir": self.output_subdir,
            "filter_errors": self.filter_errors,
            "merge_prefixes": self.merge_prefixes,
        }
        reward_processor_path = callable_import_path(self.reward_processor)
        if kwargs["rollout_fn"] is None or kwargs["get_task_fn"] is None:
            raise ValueError("GroupRolloutWorkflow requires importable rollout_fn/get_task_fn")
        if reward_processor_path is not None:
            kwargs["reward_processor"] = reward_processor_path
        return kwargs

    def to_remote_workflow(self) -> tuple[type["GroupRolloutWorkflow"], dict[str, Any]]:
        """Describe how this workflow should be reconstructed on workers."""
        return self.__class__, self.to_workflow_kwargs()

    def set_proxy_base_url(self, proxy_base_url: str) -> None:
        """Bind the worker-local proxy URL before rollout execution."""
        self.proxy_base_url = proxy_base_url

    def _require_proxy_base_url(self) -> str:
        if not self.proxy_base_url:
            raise RuntimeError(
                "GroupRolloutWorkflow.proxy_base_url is not set. "
                "Expected AReaL worker dispatch to inject a proxy endpoint."
            )
        return self.proxy_base_url

    def _session_task_id(self, task_id: str, rollout_number: int) -> str:
        return f"{task_id}-rollout-{rollout_number}-{uuid.uuid4().hex[:8]}"

    def _build_rollout_config(self, engine: InferenceEngine, session: ArealProxySession) -> WorkflowConfig:
        config = deepcopy(self.config)
        config.rollout_config.output_dir = os.path.join(
            config.rollout_config.output_dir,
            self.output_subdir,
        )
        config.rollout_config.model_endpoint = self._require_proxy_base_url()
        model_name = config.rollout_config.model_name or ""
        if not model_name.startswith("openai/"):
            config.rollout_config.model_name = f"openai/{model_name}"
        config.rollout_config.model_api_key = session.session_api_key
        config.rollout_config.output_dir = os.path.join(
            config.rollout_config.output_dir,
            str(engine.get_version()),
        )
        return config

    def _record_stats(self, train_data: dict) -> None:
        tracker = stats_tracker.get(workflow_context.stat_scope())

        def scalar_series(name: str, values: torch.Tensor) -> None:
            for value in values.detach().float().reshape(-1):
                tracker.scalar(**{name: value.item()})

        def scalar_value(name: str, value: torch.Tensor) -> None:
            tracker.scalar(**{name: value.detach().float().item()})

        num_steps = train_data["num_steps"]
        num_input_tokens = train_data["num_input_tokens"]
        num_output_tokens = train_data["num_output_tokens"]
        safe_num_steps = torch.clamp(num_steps, min=1.0)
        avg_input_tokens_per_step = num_input_tokens / safe_num_steps
        avg_output_tokens_per_step = num_output_tokens / safe_num_steps

        scalar_series("task_reward", train_data["task_reward"])
        scalar_series("num_output_tokens", num_output_tokens)
        scalar_series("num_input_tokens", num_input_tokens)
        scalar_series("num_steps", num_steps)
        scalar_series("avg_input_tokens_per_step", avg_input_tokens_per_step)
        scalar_series("avg_output_tokens_per_step", avg_output_tokens_per_step)

        task_rewards = train_data["task_reward"]
        scalar_value("task_reward_at_k_mean", torch.mean(task_rewards))
        scalar_value("task_reward_at_k_max", torch.max(task_rewards))
        scalar_value("task_reward_at_k_min", torch.min(task_rewards))

        for key, value in train_data.items():
            if key.startswith("root_"):
                scalar_series(key, value)
                scalar_value(f"{key}_at_k_mean", torch.mean(value))
                scalar_value(f"{key}_at_k_max", torch.max(value))
                scalar_value(f"{key}_at_k_min", torch.min(value))
            elif key.startswith("reward/"):
                scalar_series(key, value)

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        if self.config.use_subprocesses:
            results = await self._arun_episode_with_subprocesses(engine, data)
        else:
            results = await asyncio.gather(
                *[self._arun_episode_single(engine, data, i) for i in range(self.config.group_size)]
            )
        results = [result for result in results if result is not None]
        if not results:
            logger.warning("No rollout results found for task %s", data["task_id"])
            return None

        train_data = concat_padded_tensors(results)
        mean_unprocessed_reward = torch.mean(train_data["rewards"])

        if self.config.leave_one_out_baseline and len(results) > 1:
            task_rewards = train_data["task_reward"]
            total_reward = task_rewards.sum()
            loo_baselines = (total_reward - task_rewards) / (len(task_rewards) - 1)
            datum_counts = torch.tensor([r["rewards"].shape[0] for r in results])
            per_datum_baselines = torch.repeat_interleave(loo_baselines, datum_counts)
            train_data["rewards"] = train_data["rewards"] - per_datum_baselines
        else:
            train_data["rewards"] = train_data["rewards"] - torch.mean(train_data["task_reward"])

        self._record_stats(train_data)

        # Trainer-side full-batch transforms may still need batch metadata like
        # traj_depth, so the workflow only signals which datums are trainable.
        if not self.config.filter_zero_variance_groups:
            train_data["trainable_datums"] = torch.ones_like(train_data["rewards"], dtype=torch.bool)

        if train_data["rewards"].max() == train_data["rewards"].min() and len(results) > 1:
            stats_tracker.get(workflow_context.stat_scope()).scalar(zero_variance_reward_group=1.0)
            logger.info(
                "All rewards identical for task %s (reward=%.2f)",
                data["task_id"],
                mean_unprocessed_reward.item(),
            )
            if self.config.filter_zero_variance_groups:
                return None
            train_data["trainable_datums"] = torch.zeros_like(train_data["rewards"], dtype=torch.bool)

        return train_data

    async def _process_trajectory_result(
        self,
        trajectory_data: dict | None,
        session: ArealProxySession,
        task_id: str,
        rollout_number: int,
    ) -> dict | None:
        if trajectory_data is None:
            logger.warning("Rollout %s returned None for task %s", rollout_number, task_id)
            return None
        if not trajectory_data.get("trajectories"):
            logger.warning("No trajectories found for task %s rollout %s", task_id, rollout_number)
            return None

        completions = await session.export_interactions()
        use_depth_weighting = self.config.depth_level_weighting
        use_depth_discount = self.config.depth_level_discount_gamma is not None
        train_data = get_train_data_for_trajectory_collection(
            trajectory_data,
            completions,
            task_id,
            self.filter_errors,
            self.reward_processor,
            self.merge_prefixes,
            concat_fn=concat_padded_tensors,
            include_traj_depth=use_depth_weighting or use_depth_discount,
            include_traj_start=use_depth_weighting,
        )
        if train_data is None:
            logger.warning("No train data found for task %s rollout %s", task_id, rollout_number)
        return train_data

    async def _run_rollout_subprocess(
        self,
        executor: "ProcessPoolExecutor",
        engine: InferenceEngine,
        task_id: str,
        rollout_number: int,
        session: ArealProxySession,
    ) -> dict | None:
        from dataclasses import asdict

        from platoon.train.areal.subprocess_worker import run_rollout_subprocess

        config = self._build_rollout_config(engine, session)
        hard_timeout = (self.config.rollout_config.timeout or 900) + 120 + 60

        try:
            loop = asyncio.get_running_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    run_rollout_subprocess,
                    self.rollout_fn.__module__,
                    self.rollout_fn.__name__,
                    self.get_task_fn.__module__,
                    self.get_task_fn.__name__,
                    task_id,
                    asdict(config.rollout_config),
                ),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Subprocess hard timeout (%ss) for task %s rollout %s", hard_timeout, task_id, rollout_number)
            return None
        except Exception:
            logger.exception("Subprocess rollout failed for task %s rollout %s", task_id, rollout_number)
            return None

    async def _arun_episode_with_subprocesses(self, engine: InferenceEngine, data: dict) -> list[dict | None]:
        from concurrent.futures import ProcessPoolExecutor

        http_session = await workflow_context.get_aiohttp_session()
        sessions: list[ArealProxySession] = []
        raw_results: list[dict | None] = []
        executor = ProcessPoolExecutor(max_workers=self.config.group_size, mp_context=mp.get_context("spawn"))
        proxy_base_url = self._require_proxy_base_url()
        try:
            for rollout_number in range(self.config.group_size):
                session = ArealProxySession(
                    session=http_session,
                    base_url=proxy_base_url,
                    task_id=self._session_task_id(data["task_id"], rollout_number),
                    admin_api_key=self.proxy_admin_api_key,
                )
                await session.__aenter__()
                sessions.append(session)

            raw_results = await asyncio.gather(
                *[
                    self._run_rollout_subprocess(executor, engine, data["task_id"], rollout_number, session)
                    for rollout_number, session in enumerate(sessions)
                ]
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            for session in sessions:
                await session.__aexit__(None, None, None)

        if not raw_results:
            raw_results = [None] * len(sessions)

        return await asyncio.gather(
            *[
                self._process_trajectory_result(raw_result, session, data["task_id"], rollout_number)
                for rollout_number, (raw_result, session) in enumerate(zip(raw_results, sessions, strict=True))
            ]
        )

    async def _arun_episode_single(self, engine: InferenceEngine, data: dict, rollout_number: int) -> dict | None:
        task_id = data["task_id"]
        proxy_base_url = self._require_proxy_base_url()
        session = ArealProxySession(
            session=await workflow_context.get_aiohttp_session(),
            base_url=proxy_base_url,
            task_id=self._session_task_id(task_id, rollout_number),
            admin_api_key=self.proxy_admin_api_key,
        )
        trajectory_data = None
        try:
            await session.__aenter__()
            config = self._build_rollout_config(engine, session)
            task = self.get_task_fn(task_id)
            if config.rollout_config.max_steps is not None:
                task.max_steps = config.rollout_config.max_steps
            trajectory_data = await asyncio.create_task(self.rollout_fn(task, config.rollout_config))
        except Exception:
            logger.exception("Error in AReaL workflow for task %s rollout %s", task_id, rollout_number)
        finally:
            await session.__aexit__(None, None, None)

        return await self._process_trajectory_result(trajectory_data, session, task_id, rollout_number)
