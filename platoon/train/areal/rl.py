"""Platoon AReaL Trainer for distributed training."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from areal.api import WorkflowLike
from areal.api.cli_args import OpenAIProxyConfig
from areal.infra import RolloutController, current_platform
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.environ import is_single_controller
from areal.utils.perf_tracer import Category
from datasets import Dataset

from platoon.train.areal.actor import PlatoonPPOActor
from platoon.train.areal.batch_transforms import (
    BatchTransform,
    BatchTransformContext,
    build_default_batch_transforms,
    get_batch_size,
    index_batch,
    localize_rtensors,
    run_batch_transforms,
    split_batch_to_trajectories,
)
from platoon.train.areal.config_defs import PlatoonArealRLTrainerConfig, PlatoonPPOActorConfig
from platoon.train.areal.workflow_serialization import normalize_remote_workflow

logger = logging.getLogger("PlatoonArealRLTrainer")


class PlatoonArealRLTrainer(PPOTrainer):
    """Platoon's AReaL-based RL trainer."""

    def __init__(
        self,
        config: PlatoonArealRLTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None,
        batch_transforms: list[BatchTransform] | None = None,
    ):
        super().__init__(config=config, train_dataset=train_dataset, valid_dataset=val_dataset)
        self.proxy_admin_api_key = (self.config.rollout.openai or OpenAIProxyConfig()).admin_api_key
        self.proxy_base_url: str | None = None
        self.eval_proxy_base_url: str | None = None
        self.batch_transforms = self._build_batch_transforms(batch_transforms)
        self._start_platoon_proxies()

    def _create_train_engine(self, actor_config, alloc):
        if (
            isinstance(actor_config, PlatoonPPOActorConfig)
            and alloc.backend == "fsdp"
        ):
            if is_single_controller():
                actor = PlatoonPPOActor.as_controller(actor_config, self.scheduler)
            else:
                actor = PlatoonPPOActor(actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            return actor
        return super()._create_train_engine(actor_config, alloc)

    def _proxy_mode(self) -> str:
        return (self.config.rollout.openai or OpenAIProxyConfig()).mode

    def _resolve_proxy_base_url(self, controller: RolloutController) -> str | None:
        mode = self._proxy_mode()
        if mode == "online":
            controller.start_proxy_gateway()
            return controller.proxy_gateway_addr
        return None

    def _start_platoon_proxies(self) -> None:
        if not is_single_controller():
            raise NotImplementedError("Platoon's updated AReaL integration requires single-controller mode")
        if not isinstance(self.rollout, RolloutController):
            raise TypeError("Expected rollout to be a RolloutController in single-controller mode")

        logger.info("Starting Platoon proxy workers for mode=%s", self._proxy_mode())
        self.rollout.start_proxy()
        self.proxy_base_url = self._resolve_proxy_base_url(self.rollout)

        if isinstance(self.eval_rollout, RolloutController):
            self.eval_rollout.start_proxy()
            self.eval_proxy_base_url = self._resolve_proxy_base_url(self.eval_rollout)
        else:
            self.eval_proxy_base_url = self.proxy_base_url

    def _build_batch_transforms(
        self,
        extra_batch_transforms: list[BatchTransform] | None = None,
    ) -> list[BatchTransform]:
        """Build the ordered full-batch transform pipeline.

        Ordering matters:
        1. The workflow has already applied per-group reward centering.
        2. The trainer performs canonical batch reduction/filtering.
        3. These transforms run on the full concatenated batch.
        4. Only then do ref/prox/teacher enrichment and advantage computation run.
        """

        transforms = build_default_batch_transforms(self.config)
        if extra_batch_transforms:
            transforms.extend(extra_batch_transforms)
        return transforms

    @staticmethod
    def _controller_dispatch_group_size() -> int:
        """Platoon workflows already own rollout multiplicity internally."""
        return 1

    def _actor_dispatch_dp_size(self) -> int:
        """Return the DP size used by AReaL controller tensor dispatch."""
        parallel_strategy = getattr(self.actor, "parallel_strategy", None)
        if parallel_strategy is not None and getattr(parallel_strategy, "dp_size", None) is not None:
            return int(parallel_strategy.dp_size)
        return int(self.actor.data_parallel_world_size)

    def _maybe_shuffle_and_trim_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        batch_size = get_batch_size(batch)
        if batch_size == 0:
            return None
        dispatch_dp_size = self._actor_dispatch_dp_size()

        index_device = None
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
                index_device = value.device
                break
        if index_device is None:
            # RTensors expose shape metadata but fetch CPU tensors when localized,
            # so CPU indices are the safest default until index_batch localizes.
            index_device = torch.device("cpu")

        indices = torch.arange(batch_size, device=index_device)
        if self.config.rollout.shuffle_cross_task:
            indices = indices[torch.randperm(batch_size, device=index_device)]

        # Match the pre-migration trimming semantics: enforce divisibility by
        # lcm(ensure_batch_divisible_by, dp_size) so a single trim preserves both
        # guarantees, and skip trimming entirely when the batch is smaller than
        # one full multiple (sequential %-trims could over-trim and break the
        # ensure_batch_divisible_by contract).
        ensure = math.lcm(
            max(int(self.config.rollout.ensure_batch_divisible_by), 1),
            dispatch_dp_size,
        )
        total = int(indices.numel())
        if total < dispatch_dp_size:
            return None
        remainder = total % ensure
        if remainder != 0 and total >= ensure:
            indices = indices[: total - remainder]
        if int(indices.numel()) < dispatch_dp_size:
            return None

        return index_batch(batch, indices)

    def _reduce_rollout_batch(self, rollout_batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Reduce rollout items into the canonical trainer batch.

        This stage intentionally owns batch-cardinality changes such as consuming
        `trainable_datums`. User-extensible transforms only run after this
        reduction has produced a stable full batch.
        """
        rollout_batch = [item for item in rollout_batch if item]
        if not rollout_batch:
            return None

        # In single-controller mode prepare_batch returns remotized trajectories
        # whose values are RTensor handles, not torch.Tensors. AReaL's
        # concat_padded_tensors only concatenates tensor/list values and silently
        # keeps the *first* dict's value for anything else, which would drop every
        # rollout group but the first. Localize before concatenating.
        rollout_batch = [localize_rtensors(item) for item in rollout_batch]
        batch = concat_padded_tensors(rollout_batch)

        # Workflow-level stat tensors were already consumed by rollout-side stats
        # recording and do not share the per-datum batch dim, so they cannot be
        # filtered/split consistently with the rest of the batch. Drop them here
        # instead of broadcasting stale copies into every dispatched trajectory.
        stat_keys = ("task_reward", "num_steps", "num_input_tokens", "num_output_tokens")
        for key in list(batch.keys()):
            if key in stat_keys or key.startswith("root_") or key.startswith("reward/"):
                del batch[key]

        if "trainable_datums" in batch:
            trainable_mask = batch.pop("trainable_datums").bool()
            global_trainable = int(trainable_mask.sum().item())
            min_per_step = self._actor_dispatch_dp_size()
            if global_trainable < min_per_step:
                return None
            if not bool(trainable_mask.all()):
                indices = torch.nonzero(trainable_mask, as_tuple=False).squeeze(-1)
                batch = index_batch(batch, indices)

        return batch

    def _postprocess_rollout_batch(
        self,
        rollout_batch: list[dict[str, Any]],
        global_step: int,
        epoch: int,
        epoch_step: int,
    ) -> list[dict[str, Any]] | None:
        batch = self._reduce_rollout_batch(rollout_batch)
        if batch is None:
            return None

        context = BatchTransformContext(
            config=self.config,
            actor_dp_world_size=self._actor_dispatch_dp_size(),
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
        )
        batch = run_batch_transforms(batch, self.batch_transforms, context)
        if batch is None:
            return None
        batch = self._maybe_shuffle_and_trim_batch(batch)
        if batch is None:
            return None
        # Restore AReaL's canonical per-trajectory representation so downstream
        # controller dispatch can rebalance work across DP groups.
        return split_batch_to_trajectories(batch)

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = self.recover_info.last_step_info.next().global_step if self.recover_info is not None else 0
        workflow, workflow_kwargs = normalize_remote_workflow(
            workflow,
            workflow_kwargs,
        )
        eval_workflow, eval_workflow_kwargs = normalize_remote_workflow(
            eval_workflow,
            eval_workflow_kwargs,
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        if workflow is None:
            openai_cfg = self.config.rollout.openai
            if openai_cfg is not None and openai_cfg.mode == "online":
                self._ensure_proxy_started()
            else:
                raise ValueError(
                    "workflow must be specified for train() unless "
                    "openai.mode='online' is configured. "
                    "Pass a RolloutWorkflow, AgentWorkflow, or callable."
                )
        elif self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if config.total_train_steps is not None and global_step >= config.total_train_steps:
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            if self._should_offload_rollout:
                self._onload_rollout()
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={"global_step": global_step, "epoch_step": step},
                ),
            ):
                rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=self._controller_dispatch_group_size(),
                    dynamic_bs=self.config.dynamic_bs,
                )
            if self._should_offload_rollout:
                self._offload_rollout()

            rollout_batch = self._postprocess_rollout_batch(
                rollout_batch,
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
            )

            if rollout_batch is not None and self.critic is not None:
                if self._should_offload_critic:
                    self._onload_model(self.critic, role="critic")
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    values = self.critic.compute_values(rollout_batch)
                    for traj, v in zip(rollout_batch, values):
                        traj["values"] = v
                    self.critic.get_device_stats().log("critic values")
                if self._should_offload_critic:
                    self._offload_model(self.critic, role="critic")

            if rollout_batch is not None and self.ref is not None:
                if self._should_offload_ref:
                    self._onload_model(self.ref, role="ref")
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    ref_logps = self.ref.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, ref_logps):
                        traj["ref_logp"] = logp
                    self.ref.get_device_stats().log("ref logp")
                if self._should_offload_ref:
                    self._offload_model(self.ref, role="ref")

            if rollout_batch is not None and self.teacher is not None:
                if self._should_offload_teacher:
                    self._onload_model(self.teacher, role="teacher")
                with (
                    stats_tracker.record_timing("teacher_logp"),
                    perf_tracer.trace_scope(
                        "train.teacher_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    teacher_logps = self.teacher.compute_logp(rollout_batch)
                    for traj, logp in zip(rollout_batch, teacher_logps):
                        traj["teacher_logp"] = logp
                        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                        traj["distill_loss_weight"] = self.config.teacher.distill_loss_weight
                    self.teacher.get_device_stats().log("teacher logp")
                if self._should_offload_teacher:
                    self._offload_model(self.teacher, role="teacher")

            adv_batch = None
            if rollout_batch is not None:
                if self._should_offload_actor:
                    self._onload_model(self.actor, role="actor")
                if config.actor.should_compute_prox_logp():
                    with (
                        stats_tracker.record_timing("recompute_logp"),
                        perf_tracer.trace_scope(
                            "train.recompute_logp",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        prox_logps = self.actor.compute_logp(rollout_batch)
                        for traj, logp in zip(rollout_batch, prox_logps):
                            traj["prox_logp"] = logp
                        self.actor.get_device_stats().log("recompute logp")

                with (
                    stats_tracker.record_timing("compute_advantage"),
                    perf_tracer.trace_scope(
                        "train.compute_advantage",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    adv_batch = self.actor.compute_advantages(rollout_batch)
                    self.actor.get_device_stats().log("compute advantages")

                self.saver.maybe_wait_for_staging()

                with (
                    stats_tracker.record_timing("train_step"),
                    perf_tracer.trace_scope(
                        "train.ppo_update",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    self.actor.ppo_update(adv_batch)
                    self.actor.step_lr_scheduler()
                    self.actor.get_device_stats().log("ppo update")
                if self._should_offload_actor:
                    self._offload_model(self.actor, role="actor")

                if self.critic is not None:
                    if self._should_offload_critic:
                        self._onload_model(self.critic, role="critic")
                    with (
                        stats_tracker.record_timing("critic_train_step"),
                        perf_tracer.trace_scope(
                            "train.critic_ppo_update",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        self.critic.ppo_update(adv_batch)
                        self.critic.step_lr_scheduler()
                        self.critic.get_device_stats().log("ppo critic update")
                    if self._should_offload_critic:
                        self._offload_model(self.critic, role="critic")

                self.rollout.pause()

                with (
                    stats_tracker.record_timing("update_weights"),
                    perf_tracer.trace_scope(
                        "train.update_weights",
                        category=Category.COMM,
                        args={"global_step": global_step},
                    ),
                ):
                    new_version = global_step + 1
                    versioned_meta = self.weight_update_meta.with_version(new_version)
                    self.actor.update_weights(versioned_meta)
                    self.actor.set_version(new_version)
                    if self.critic is not None:
                        self.critic.set_version(new_version)
                    if self.ref is not None:
                        self.ref.set_version(new_version)
                    if self.teacher is not None:
                        self.teacher.set_version(new_version)
                    self.rollout.set_version(new_version)
                    if self.eval_rollout is not None:
                        self.eval_rollout.set_version(new_version)
            else:
                new_version = global_step + 1
                self.actor.set_version(new_version)
                if self.critic is not None:
                    self.critic.set_version(new_version)
                if self.ref is not None:
                    self.ref.set_version(new_version)
                if self.teacher is not None:
                    self.teacher.set_version(new_version)
                self.rollout.set_version(new_version)
                if self.eval_rollout is not None:
                    self.eval_rollout.set_version(new_version)

            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(epoch=epoch, epoch_step=step, global_step=global_step)

            if self._should_offload_rollout:
                self._onload_rollout(is_eval=True)
            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )
            if self._should_offload_rollout:
                self._offload_rollout(is_eval=True)

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                if rollout_batch is not None and adv_batch is not None:
                    self.actor.clear_batches(rollout_batch, adv_batch)
                if self.data_controller is not None:
                    self.data_controller.clear_batches()

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(epoch=epoch, epoch_step=step, global_step=global_step)

            self.rollout.resume()
            current_platform.synchronize()
            self._save_perf_tracer(step=global_step)

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self._controller_dispatch_group_size(),
                        is_eval=True,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
