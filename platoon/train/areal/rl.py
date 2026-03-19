"""AReaL RL Trainer for distributed training."""

import datetime
import os
from copy import deepcopy

import torch
import torch.distributed as dist
from areal.api.alloc_mode import AllocationMode
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.proxy import ProxyServer
from areal.platforms import current_platform
from areal.utils import logging, seeding, stats_tracker
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.network import find_free_ports
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from datasets import Dataset
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from platoon.train.areal.actor import create_actor
from platoon.train.areal.config_defs import PlatoonArealRLTrainerConfig
from platoon.utils.train import (
    bcast_and_split_from_rank0,
    post_process_and_redistribute_tensor_container,
    set_expandable_segments,
    tensor_container_to,
)

logger = logging.getLogger("Platoon AReaL RL Trainer")


def _get_local_batch_size(batch: dict[str, torch.Tensor | list]) -> int:
    if "attention_mask" in batch and torch.is_tensor(batch["attention_mask"]):
        return int(batch["attention_mask"].shape[0])
    if "input_ids" in batch and torch.is_tensor(batch["input_ids"]):
        return int(batch["input_ids"].shape[0])
    for value in batch.values():
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    raise ValueError("Unable to infer local batch size from batch contents")


def _index_local_batch(batch: dict[str, torch.Tensor | list], keep: torch.Tensor) -> dict[str, torch.Tensor | list]:
    local_batch_size = _get_local_batch_size(batch)
    filtered: dict[str, torch.Tensor | list] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == local_batch_size:
            filtered[key] = value[keep.to(value.device)]
        elif isinstance(value, list) and len(value) == local_batch_size:
            keep_list = keep.cpu().tolist()
            filtered[key] = [item for item, should_keep in zip(value, keep_list) if should_keep]
        else:
            filtered[key] = value
    return filtered


class PlatoonArealRLTrainer:
    """Trainer for RL with AReaL backend.

    This trainer uses FSDP for distributed training and SGLang for inference.
    """

    def __init__(
        self,
        config: PlatoonArealRLTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizerFast | None = None,
    ):
        set_expandable_segments(True)

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer

        rank = int(os.environ.get("RANK", 0))

        if self.tokenizer is None:
            self.tokenizer = load_hf_tokenizer(config.tokenizer_path)

        if self.tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(self.tokenizer.eos_token_id)

        seeding.set_random_seed(config.seed, key=f"trainer{rank}")
        allocation_mode = AllocationMode.from_str(config.allocation_mode)
        parallel_strategy = allocation_mode.train
        assert parallel_strategy is not None

        # LoRA-specific validation
        self.use_lora = config.actor.use_lora
        if self.use_lora:
            if parallel_strategy.data_parallel_size != parallel_strategy.world_size:
                raise ValueError("LoRA does not support parallelism other than FSDP.")

        self.actor = create_actor(
            config=config.actor,
            loss_fn_config=config.loss_fn_config,
        )
        self.actor.create_process_group(parallel_strategy=parallel_strategy)

        # Create dataloaders
        # NOTE: For LoRA, only rank 0 submits rollouts due to a known bug with
        # multi-rank LoRA inference. We use rank=0, world_size=1 for dataloaders
        # and broadcast the batch to all ranks after rollout.
        if self.use_lora:
            dataloader_num_replicas = 1
            dataloader_rank = 0
        else:
            dataloader_num_replicas = self.actor.data_parallel_world_size
            dataloader_rank = self.actor.data_parallel_rank

        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=config.train_dataset.batch_size // dataloader_num_replicas,
            sampler=DistributedSampler(
                train_dataset,
                num_replicas=dataloader_num_replicas,
                rank=dataloader_rank,
                shuffle=config.train_dataset.shuffle,
                drop_last=config.train_dataset.drop_last,
            ),
            num_workers=config.train_dataset.num_workers,
            collate_fn=lambda x: x,
        )

        self.val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=config.valid_dataset.batch_size // dataloader_num_replicas,
            sampler=DistributedSampler(
                val_dataset,
                num_replicas=dataloader_num_replicas,
                rank=dataloader_rank,
                shuffle=config.valid_dataset.shuffle,
                drop_last=config.valid_dataset.drop_last,
            ),
            num_workers=config.valid_dataset.num_workers,
            collate_fn=lambda x: x,
        )

        self.ft_spec = FinetuneSpec(
            total_train_epochs=config.total_train_epochs,
            dataset_size=len(self.train_dataloader) * config.train_dataset.batch_size,
            train_batch_size=config.train_dataset.batch_size,
        )

        # Initialize rollout engines
        # NOTE: For LoRA, use train_data_parallel_size=1 since only rank 0 submits
        self.rollout = RemoteSGLangEngine(config.rollout)
        rollout_dp_size = 1 if self.use_lora else parallel_strategy.dp_size
        self.rollout.initialize(train_data_parallel_size=rollout_dp_size)
        self.eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
        # NOTE: eval does not have any offpolicyness control
        self.eval_rollout.config.max_head_offpolicyness = int(1e12)
        self.eval_rollout.initialize()

        # Weight update meta differs for LoRA vs standard training
        if self.use_lora:
            self.weight_update_meta = WeightUpdateMeta.from_disk(
                config.saver.experiment_name,
                config.saver.trial_name,
                config.saver.fileroot,
                use_lora=True,
            )
        else:
            self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(allocation_mode)

        self.actor.initialize(None, self.ft_spec)
        self.actor.connect_engine(self.rollout, self.weight_update_meta)

        # Optional reference model for KL
        # Reference model uses standard FSDPPPOActor (no custom loss function needed)
        self.ref = None
        if config.actor.kl_ctl > 0 and config.ref is not None:
            from areal.engine.ppo.actor import FSDPPPOActor

            self.ref = FSDPPPOActor(config=config.ref)
            self.ref.create_process_group(parallel_strategy=parallel_strategy)
            self.ref.initialize(None, self.ft_spec)

        # Setup proxy servers
        self.llm_client = ArealOpenAI(engine=self.rollout, tokenizer=self.tokenizer)
        free_port = find_free_ports(1)[0]
        self.proxy_server = ProxyServer(free_port, client=self.llm_client)
        self.proxy_server.start(wait_until_ready=True)

        # Eval proxy uses the eval rollout engine; training rollout is paused during eval.
        self.eval_llm_client = ArealOpenAI(engine=self.eval_rollout, tokenizer=self.tokenizer)
        eval_free_port = find_free_ports(1)[0]
        self.eval_proxy_server = ProxyServer(eval_free_port, client=self.eval_llm_client)
        self.eval_proxy_server.start(wait_until_ready=True)

        all_addresses = [None for _ in range(self.actor.data_parallel_world_size)]
        dist.all_gather_object(all_addresses, self.proxy_server.public_addr, group=self.actor.data_parallel_group)
        logger.info(f"Found {len(all_addresses)} proxy servers: {all_addresses}")
        dist.barrier(device_ids=[self.actor.device.index])

        # Setup utilities
        self.saver = Saver(config.saver, self.ft_spec)
        self.stats_logger = StatsLogger(config, self.ft_spec)
        self.evaluator = Evaluator(config.evaluator, self.ft_spec)

        self.recover_handler = RecoverHandler(config.recover, self.ft_spec)
        self.recover_info = self.recover_handler.load(
            self.actor,
            self.saver,
            self.evaluator,
            self.stats_logger,
            self.train_dataloader,
            inference_engine=self.rollout,
            weight_update_meta=self.weight_update_meta,
        )

        # Increase timeout for long training runs
        dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=7200), self.actor.data_parallel_group)

    def train(self, workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow):
        """Run the training loop."""
        config = self.config
        start_step = self.recover_info.last_step_info.next().global_step if self.recover_info is not None else 0

        total_epochs = config.total_train_epochs
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        for global_step in range(start_step, max_steps):
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch
            step_info = StepInfo(
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
                steps_per_epoch=steps_per_epoch,
            )

            with stats_tracker.record_timing("rollout"):
                if self.use_lora:
                    # NOTE: For LoRA, only rank 0 submits rollouts due to a known bug
                    # where multi-rank LoRA inference has concurrency issues.
                    # The batch is then broadcast and split across all ranks.
                    batch = None
                    if dist.get_rank() == 0:
                        batch = self.rollout.prepare_batch(
                            self.train_dataloader,
                            workflow=workflow,
                            should_accept_fn=lambda sample: True,
                        )
                        batch = tensor_container_to(batch, self.actor.device)
                    batch = bcast_and_split_from_rank0(
                        batch,
                        granularity=1,
                        device=self.actor.device,
                    )
                else:
                    batch = self.actor.prepare_batch(
                        self.train_dataloader,
                        workflow=workflow,
                        should_accept_fn=lambda sample: True,
                    )

            if self.use_lora:
                # Synchronize after rollout
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()

            if batch is not None and "trainable_datums" in batch:
                trainable_mask = batch.pop("trainable_datums").bool()
                local_trainable = torch.tensor(
                    [int(trainable_mask.sum().item())],
                    device=self.actor.device,
                    dtype=torch.long,
                )
                global_trainable = local_trainable.clone()
                dist.all_reduce(global_trainable, op=dist.ReduceOp.SUM, group=self.actor.data_parallel_group)

                min_per_step = self.actor.data_parallel_world_size
                if global_trainable.item() < min_per_step:
                    batch = None
                elif not bool(trainable_mask.all()):
                    batch = _index_local_batch(batch, trainable_mask)
                    batch = post_process_and_redistribute_tensor_container(
                        batch,
                        shuffle=self.config.rollout.shuffle_cross_task,
                        ensure_divisible_by=self.config.rollout.ensure_batch_divisible_by,
                        group=self.actor.data_parallel_group,
                    )

            if batch is not None:

                if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
                    with stats_tracker.record_timing("recompute_logp"):
                        logp = self.actor.compute_logp(batch)
                        batch["prox_logp"] = logp
                        log_gpu_stats("recompute logp")

                    dist.barrier(device_ids=[self.actor.device.index])
                    current_platform.synchronize()

                if self.ref is not None:
                    with stats_tracker.record_timing("ref_logp"):
                        ref_logp = self.ref.compute_logp(batch)
                        batch["ref_logp"] = ref_logp
                        log_gpu_stats("ref logp")

                    dist.barrier(device_ids=[self.actor.device.index])
                    current_platform.synchronize()

                # Apply optional depth-based reward weighting if traj_depth is present.
                if "traj_depth" in batch:
                    traj_depth = batch["traj_depth"]
                    depth_indices = traj_depth.long()
                    depth_gamma = config.workflow_config.depth_level_discount_gamma

                    if depth_gamma is not None:
                        # Alternate strategy: discount rewards by depth as gamma^d,
                        # with depth starting at 0 for root trajectories.
                        # Normalize across the full data-parallel batch so total
                        # reward mass stays unchanged and only the relative depth
                        # emphasis changes.
                        if depth_gamma < 0:
                            raise ValueError("workflow_config.depth_level_discount_gamma must be non-negative")
                        dp_group = self.actor.data_parallel_group
                        gamma = torch.tensor(depth_gamma, device=traj_depth.device, dtype=batch["rewards"].dtype)
                        raw_weights = torch.pow(gamma, depth_indices.to(batch["rewards"].dtype))
                        global_weight_stats = torch.tensor(
                            [raw_weights.sum().item(), float(raw_weights.numel())],
                            device=traj_depth.device,
                            dtype=torch.float32,
                        )
                        dist.all_reduce(global_weight_stats, op=dist.ReduceOp.SUM, group=dp_group)
                        global_raw_weight_sum = global_weight_stats[0]
                        global_num_datums = global_weight_stats[1]
                        if global_raw_weight_sum <= 0:
                            raise ValueError(
                                "workflow_config.depth_level_discount_gamma produced zero total weight for this batch"
                            )
                        normalization = (global_num_datums / global_raw_weight_sum).to(raw_weights.dtype)
                        per_datum_weights = raw_weights * normalization
                    else:
                        # depth-level inverse-frequency weighting uses full-batch trajectory
                        # counts so each depth level gets equal effective trajectory
                        # representation. Counts are computed across all data parallel
                        # workers over the full batch.
                        traj_start = batch["traj_start"]
                        dp_group = self.actor.data_parallel_group

                        # Find the global max depth across all workers.
                        local_max_depth = int(traj_depth.max().item()) if traj_depth.numel() > 0 else 0
                        max_depth_t = torch.tensor([local_max_depth], device=traj_depth.device, dtype=torch.long)
                        dist.all_reduce(max_depth_t, op=dist.ReduceOp.MAX, group=dp_group)
                        global_max_depth = int(max_depth_t.item())

                        # Count datums and distinct trajectories at each depth locally.
                        # traj_start is 1.0 at the first datum of each trajectory, so
                        # summing it gives trajectory count rather than datum count.
                        num_depths = global_max_depth + 1
                        counts = torch.zeros(2, num_depths, device=traj_depth.device)
                        for d in range(num_depths):
                            mask_d = depth_indices == d
                            counts[0, d] = mask_d.sum().float()
                            counts[1, d] = traj_start[mask_d].sum()
                        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=dp_group)
                        datum_counts = counts[0]
                        traj_counts = counts[1]

                        # Inverse-frequency weights based on trajectory counts:
                        # raw_weight_d = 1 / traj_count_d for each datum at depth d.
                        # Normalize so the global total weight equals batch size.
                        total_datums = datum_counts.sum()
                        raw_weights = torch.where(
                            traj_counts > 0,
                            1.0 / traj_counts,
                            torch.zeros_like(traj_counts),
                        )
                        # unnorm_total = sum_d datum_count_d * raw_weight_d
                        unnorm_total = (datum_counts * raw_weights).sum()
                        normalization = total_datums / unnorm_total
                        per_depth_weights = normalization * raw_weights
                        per_datum_weights = per_depth_weights[depth_indices]

                    batch["rewards"] = batch["rewards"] * per_datum_weights
                    del batch["traj_depth"]
                    if "traj_start" in batch:
                        del batch["traj_start"]

                with stats_tracker.record_timing("compute_advantage"):
                    self.actor.compute_advantages(batch)
                    log_gpu_stats("compute advantages")

                torch.cuda.empty_cache()
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()

                with stats_tracker.record_timing("train_step"):
                    stats = self.actor.ppo_update(batch)
                    self.actor.step_lr_scheduler()
                    log_gpu_stats("ppo update")

                torch.cuda.empty_cache()
                dist.barrier(device_ids=[self.actor.device.index])
                current_platform.synchronize()

                # Pause inference for updating weights, save, and evaluation
                self.rollout.pause()

                with stats_tracker.record_timing("update_weights"):
                    self.actor.update_weights(self.weight_update_meta)
                    
            self.actor.set_version(global_step + 1)
            self.rollout.set_version(global_step + 1)
            self.eval_rollout.set_version(global_step + 1)

            with stats_tracker.record_timing("save"):
                self.saver.save(self.actor, epoch, step, global_step, tokenizer=self.tokenizer)

            with stats_tracker.record_timing("checkpoint_for_recover"):
                self.recover_handler.dump(
                    self.actor,
                    step_info,
                    self.saver,
                    self.evaluator,
                    self.stats_logger,
                    self.train_dataloader,
                    tokenizer=self.tokenizer,
                )

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            with stats_tracker.record_timing("eval"):
                self._run_evaluation(eval_workflow, epoch, step, global_step)

            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # Upload statistics to the logger (e.g., wandb)
            stats = stats_tracker.export_all(reduce_group=self.actor.data_parallel_group)
            self.stats_logger.commit(epoch, step, global_step, stats)

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

            # Resume rollout
            self.rollout.resume()

    def _run_evaluation(self, eval_workflow: RolloutWorkflow, epoch: int, step: int, global_step: int):
        """Run evaluation if this process is the data parallel head.

        For LoRA mode, only rank 0 submits evaluation requests due to the same
        concurrency issues as training rollout.
        """

        def evaluate_fn():
            # For LoRA, only rank 0 submits; otherwise use data parallel head
            should_submit = dist.get_rank() == 0 if self.use_lora else self.actor.is_data_parallel_head()

            if should_submit:
                print("Evaluating...")
                cnt = 0
                for data in self.val_dataloader:
                    for item in data:
                        self.eval_rollout.submit(item, eval_workflow)
                        cnt += 1
                try:
                    # TODO: Make the eval timeout configurable
                    results = self.eval_rollout.wait(cnt, timeout=3600)
                    print(f"Evaluated {cnt} tasks")
                    print(f"Task Rewards: {results['task_reward']}")

                except TimeoutError as e:
                    print(f"Evaluation timed out after 1800 seconds: {e}")

            torch.cuda.empty_cache()
            dist.barrier(device_ids=[self.actor.device.index])
            current_platform.synchronize()

        self.evaluator.evaluate(
            evaluate_fn,
            epoch,
            step,
            global_step,
        )

    def close(self):
        """Clean up resources."""
        self.stats_logger.close()
        if hasattr(self, "eval_proxy_server"):
            self.eval_proxy_server.close()
        if hasattr(self, "proxy_server"):
            self.proxy_server.close()
        self.eval_rollout.destroy()
        self.rollout.destroy()
        if self.ref is not None:
            self.ref.destroy()
        self.actor.destroy()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise exc_value
