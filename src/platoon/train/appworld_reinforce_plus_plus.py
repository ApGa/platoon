import os
import sys
from copy import deepcopy
import math
from typing import Any, Dict

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.ppo.actor import FSDPPPOActor

from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    cycle_dataloader,
    tensor_container_to,
)
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from appworld import load_task_ids
from datasets import Dataset
from platoon.agents.appworld.areal_workflow import AppWorldArealWorkflow, AppWorldArealRecursiveWorkflow
from dataclasses import field, dataclass
from areal.api.cli_args import InferenceEngineConfig
from areal.utils.data import concat_padded_tensors
from areal.api.engine_api import TrainEngine
from areal.utils.functional import gather_logprobs_entropy
from platoon.train.aent.actor import FSDPAEntPPOActor
from platoon.train.aent.aent_args import AEntGRPOConfig, AEntPPOActorConfig
from areal.utils import stats_tracker
import pandas as pd
from platoon.envs.base import Task
from random import shuffle
from dataclasses import asdict
import datetime

os.environ["APPWORLD_ROOT"] = "/mnt/efs/platoon/src/platoon/envs/appworld"

def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


@dataclass
class VariableBatchInferenceEngineConfig(InferenceEngineConfig):
    shuffle_cross_task: bool = field(default=True)
    ensure_batch_divisible_by: int = field(default=1)

@dataclass
class AppWorldReinforcePlusPlusConfig(GRPOConfig):
    workflow_config: dict = field(default_factory=dict)
    rollout: VariableBatchInferenceEngineConfig = field(default_factory=VariableBatchInferenceEngineConfig)
    actor: AEntPPOActorConfig = field(default_factory=AEntPPOActorConfig)


def _canonicalize_container_keys(x: Any) -> Any:
    """Recursively sort dict keys to enforce stable collective ordering across ranks.

    Lists are preserved as-is; tensors and other leaf types are returned unchanged.
    """
    if isinstance(x, dict):
        return {k: _canonicalize_container_keys(x[k]) for k in sorted(x.keys())}
    if isinstance(x, list):
        return [ _canonicalize_container_keys(v) for v in x ]
    return x


def post_process_and_redistribute_tensor_container(batch: Dict[str, Any], shuffle: bool = True, ensure_divisible_by: int = 1, group: dist.ProcessGroup = None, local_sample_ratio: float = 1.0) -> Dict[str, Any]:
    # 1) Drop keys not shared by all ranks (for dicts outside lists)
    def _extract_dict_schema(x: Any, prefix: tuple[str, ...] = ()) -> Dict[tuple[str, ...], set]:
        schema: Dict[tuple[str, ...], set] = {}
        if isinstance(x, dict):
            keys = set(x.keys())
            schema[prefix] = keys
            for k in keys:
                v = x[k]
                # Only traverse into nested dicts; do not attempt per-index across lists
                if isinstance(v, dict):
                    schema.update(_extract_dict_schema(v, prefix + (k,)))
        return schema

    if isinstance(batch, dict):
        local_schema = _extract_dict_schema(batch)
        world_size = dist.get_world_size(group)
        schemas = [None for _ in range(world_size)]
        dist.all_gather_object(schemas, local_schema, group=group)

        # Compute paths present on all ranks
        common_paths = set.intersection(*[set(s.keys()) for s in schemas]) if schemas else set()
        # For each common path, compute the intersection of keys
        shared_schema: Dict[tuple[str, ...], set] = {}
        for p in common_paths:
            inter_keys = set.intersection(*[s[p] for s in schemas])
            shared_schema[p] = inter_keys

        def _prune_by_schema(x: Any, prefix: tuple[str, ...] = ()) -> Any:
            if not isinstance(x, dict):
                return x
            if prefix not in shared_schema:
                return {}
            allowed = shared_schema[prefix]
            pruned: Dict[str, Any] = {}
            for k in sorted(x.keys()):
                if k in allowed:
                    v = x[k]
                    if isinstance(v, dict):
                        v = _prune_by_schema(v, prefix + (k,))
                    pruned[k] = v
            return pruned

        batch = _prune_by_schema(batch)

    # 2) Ensure deterministic dict traversal order for nested structures before collectives
    #batch = _canonicalize_container_keys(batch)
    all_data = all_gather_tensor_container(batch, group=group)
    batch_tensors = concat_padded_tensors(all_data)
    
    # Determine batch size from a reliable 2D tensor key
    if "attention_mask" in batch_tensors and torch.is_tensor(batch_tensors["attention_mask"]) and batch_tensors["attention_mask"].ndim >= 2:
        batch_size = batch_tensors["attention_mask"].shape[0]
        index_device = batch_tensors["attention_mask"].device
    elif "input_ids" in batch_tensors and torch.is_tensor(batch_tensors["input_ids"]) and batch_tensors["input_ids"].ndim >= 2:
        batch_size = batch_tensors["input_ids"].shape[0]
        index_device = batch_tensors["input_ids"].device
    else:
        # Fallback to first tensor-like entry
        try:
            first_key = next(k for k, v in batch_tensors.items() if torch.is_tensor(v) and v.ndim >= 1)
        except StopIteration:
            raise ValueError("No tensor-like entries with ndim>=1 found to infer batch size")
        batch_size = batch_tensors[first_key].shape[0]
        index_device = batch_tensors[first_key].device

    # Build indices once: optionally shuffle (synchronized), then trim for divisibility
    indices = torch.arange(batch_size, device=index_device)
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group=group)
    if shuffle:
        if rank == 0:
            print(f"Shuffling batch of size {batch_size}")
            perm = torch.randperm(batch_size, device=index_device)
        else:
            perm = torch.empty(batch_size, dtype=torch.long, device=index_device)
        # Broadcast the same permutation to all ranks
        dist.broadcast(perm, src=0, group=group)
        indices = indices[perm]

    # Ensure divisibility by both ensure_divisible_by and world_size for equal shards
    ensure = ensure_divisible_by if ensure_divisible_by > 1 else 1
    ensure = math.lcm(ensure, world_size)
    total = indices.numel()
    remainder = total % ensure
    if remainder != 0 and total >= ensure:
        if rank == 0:
            print(f"Batch size {total} is not divisible by {ensure}, trimming to {total - remainder}")
        indices = indices[: total - remainder]
        total = indices.numel()

    # Shard by rank using the updated total
    start = rank * total // world_size
    end = (rank + 1) * total // world_size
    indices = indices[start:end]
            
    
    # Apply indexing only to tensors whose first dim equals batch size
    def _maybe_index(x):
        if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == batch_size:
            return x.index_select(0, indices.to(x.device))
        return x

    batch_tensors = {k: _maybe_index(v) for k, v in batch_tensors.items()}
    return batch_tensors


def main(args):
    config, _ = load_expr_config(args, AppWorldReinforcePlusPlusConfig)
    config: AppWorldReinforcePlusPlusConfig

    rank = int(os.getenv("RANK"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine

    #actor = FSDPReinforceActor(config=config.actor)
    actor = FSDPAEntPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    dist.distributed_c10d._set_pg_timeout(datetime.timedelta(seconds=7200), actor.data_parallel_group)

    # Shard datasets by data-parallel rank to avoid duplication across DP ranks
    dp_rank = actor.data_parallel_rank
    dp_world_size = actor.data_parallel_world_size
    

    train_ids = load_task_ids(dataset_name="train") + load_task_ids(dataset_name="dev")
    train_index_remainder = len(train_ids) % dp_world_size
    if train_index_remainder != 0:
        train_ids = train_ids[:-train_index_remainder]
        #train_ids = train_ids[:4] # debugging
    train_ids = [x for i, x in enumerate(train_ids) if i % dp_world_size == dp_rank]
    train_dataset = Dataset.from_list([{ "task_id": x } for x in train_ids])

    valid_ids = load_task_ids(dataset_name="test_normal")[:32]
    valid_index_remainder = len(valid_ids) % dp_world_size
    if valid_index_remainder != 0:
        valid_ids = valid_ids[:-valid_index_remainder]
        #valid_ids = valid_ids[:12] # debugging
    valid_ids = [x for i, x in enumerate(valid_ids) if i % dp_world_size == dp_rank]
    valid_dataset = Dataset.from_list([{ "task_id": x } for x in valid_ids])

    # Create dataset and dataloaderss
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=config.train_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.train_dataset.shuffle,
        num_workers=config.train_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.train_dataset.drop_last,
    )
    valid_dataloader = StatefulDataLoader(
        valid_dataset,
        batch_size=config.valid_dataset.batch_size // actor.data_parallel_world_size,
        shuffle=config.valid_dataset.shuffle,
        num_workers=config.valid_dataset.num_workers,
        collate_fn=lambda x: x,
        drop_last=config.valid_dataset.drop_last,
    )
   
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemoteSGLangEngine(config.rollout)
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size) #1)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    # NOTE: eval does not have any offpolicyness control
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()
    
    set_expandable_segments(True)

    actor.initialize(None, ft_spec)
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # NOTE: Weight update meta only requires address and free port of rank 0,
    # but `WeightUpdateMeta.from_fsdp_xccl` has to be executed on all ranks
    # due to `engine.get_param_specs()`.
    # Therefore, we create weight update meta on all ranks, then broadcast the one on rank 0.
    weight_update_meta = [
        WeightUpdateMeta.from_fsdp_xccl(
            AllocationMode.from_str(config.allocation_mode), actor
        )
    ]
    dist.broadcast_object_list(weight_update_meta, src=0)
    weight_update_meta = weight_update_meta[0]

    # weight_update_meta = WeightUpdateMeta.from_disk(
    #     config.saver.experiment_name,
    #     config.saver.trial_name,
    #     config.saver.fileroot,
    #     use_lora=True,
    # )

    # Create rollout workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    workflow = AppWorldArealWorkflow(
        config=config.workflow_config
    )
    eval_workflow = AppWorldArealWorkflow(
        config=config.workflow_config
    )

    # Run training.
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    data_generator = cycle_dataloader(train_dataloader)
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
            batch = None
            if actor.is_data_parallel_head():
                if config.async_training:
                    batch = rollout.prepare_batch(
                        train_dataloader,
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                else:
                    batch = rollout.rollout_batch(
                        next(data_generator),
                        workflow=workflow,
                        should_accept=lambda sample: True,
                    )
                    
                batch = tensor_container_to(batch, actor.device)
            
                reward_mask = torch.ones_like(batch['task_reward'], dtype=torch.bool)
                output_token_mask = torch.ones_like(batch['num_output_tokens'], dtype=torch.bool)
                input_token_mask = torch.ones_like(batch['num_input_tokens'], dtype=torch.bool)
                num_steps_mask = torch.ones_like(batch['num_steps'], dtype=torch.bool)
                
                stats_tracker.get('reinforce_actor').denominator(task_reward_mask=reward_mask, num_output_tokens_mask=output_token_mask, num_input_tokens_mask=input_token_mask, num_steps_mask=num_steps_mask)
                stats_tracker.get('reinforce_actor').stat(task_reward=batch['task_reward'], denominator="task_reward_mask")
                stats_tracker.get('reinforce_actor').stat(num_output_tokens=batch['num_output_tokens'], denominator="num_output_tokens_mask")
                stats_tracker.get('reinforce_actor').stat(num_input_tokens=batch['num_input_tokens'], denominator="num_input_tokens_mask")
                stats_tracker.get('reinforce_actor').stat(num_steps=batch['num_steps'], denominator="num_steps_mask")
                
                torch.cuda.empty_cache()
                
                if config.rollout.shuffle_cross_task or config.rollout.ensure_batch_divisible_by > 1:
                    batch = post_process_and_redistribute_tensor_container(batch, shuffle=config.rollout.shuffle_cross_task, ensure_divisible_by=config.rollout.ensure_batch_divisible_by, group=actor.data_parallel_group)
            
            batch = broadcast_tensor_container(
                batch,
                src_rank=actor.current_data_parallel_head(),
                group=actor.context_and_model_parallel_group,
            )
            
        torch.cuda.empty_cache()
        # Create barrier to synchronize all rollout processes.
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                # Ensure prox_logp token width matches attention_mask for proper microbatch splitting
                attn_len = batch["attention_mask"].shape[1]
                if torch.is_tensor(logp) and logp.ndim == 2 and logp.shape[1] != attn_len:
                    if logp.shape[1] < attn_len:
                        pad_width = attn_len - logp.shape[1]
                        logp = torch.nn.functional.pad(logp, (0, pad_width), value=0.0)
                    else:
                        logp = logp[:, :attn_len]
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                ref_logp = ref.compute_logp(batch)
                # Align ref_logp token width with attention_mask as well
                attn_len = batch["attention_mask"].shape[1]
                if torch.is_tensor(ref_logp) and ref_logp.ndim == 2 and ref_logp.shape[1] != attn_len:
                    if ref_logp.shape[1] < attn_len:
                        pad_width = attn_len - ref_logp.shape[1]
                        ref_logp = torch.nn.functional.pad(ref_logp, (0, pad_width), value=0.0)
                    else:
                        ref_logp = ref_logp[:, :attn_len]
                batch["ref_logp"] = ref_logp
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.aent_ppo_update(batch, global_step)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        # pause inference for updating weights, save, and evaluation
        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            if dist.get_rank() == 0:
                future = rollout.update_weights(weight_update_meta)
            actor.upload_weights(weight_update_meta)
            if dist.get_rank() == 0:
                future.result()
            dist.barrier(device_ids=[actor.device.index])
            current_platform.synchronize()

            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        torch.cuda.empty_cache()
        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    print(f"Evaluating...")
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    
                    if cnt == 0:
                        print(f"Warning: No evaluation tasks found in valid_dataloader. Skipping evaluation.")
                    else:
                        try:
                            results = eval_rollout.wait(cnt, timeout=1800) # TODO: Make the eval timeout configurable.
                            print(f"Evaluated {cnt} tasks")
                            print(f"Task Rewards: {results['task_reward']}")
                            
                            tr = torch.as_tensor(results['task_reward'], dtype=torch.float32, device=actor.device)
                            mask = torch.ones_like(tr, dtype=torch.bool)

                            stats_tracker.get("eval").denominator(task_reward_mask=mask)
                            stats_tracker.get("eval").stat("task_reward_mask", task_reward=tr)  # default: AVG/MIN/MAX
                        except TimeoutError as e:
                            print(f"Evaluation timed out after 1500 seconds: {e}")
                    
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Upload statistics to the logger (e.g., wandb)
        stats[0].update(
            stats_tracker.export_all(reduce_group=actor.data_parallel_group)
        )
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        # Resume rollout
        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
