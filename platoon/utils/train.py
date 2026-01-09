import torch
import torch.distributed as dist
from typing import Any, Dict
import math
from dataclasses import dataclass, field
from areal.utils.data import (
    all_gather_tensor_container,
    broadcast_tensor_container,
    concat_padded_tensors,
    get_batch_size,
    tensor_container_to,
)
from areal.core.dist_rollout import redistribute
from areal.api.cli_args import InferenceEngineConfig


@dataclass
class VariableBatchInferenceEngineConfig(InferenceEngineConfig):
    shuffle_cross_task: bool = field(default=False)
    ensure_batch_divisible_by: int = field(default=1)

def set_expandable_segments(enable: bool) -> None:
    """Enable or disable expandable segments for cuda.
    Args:
        enable (bool): Whether to enable expandable segments. Used to avoid OOM.
    """
    torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enable}")


def bcast_and_split_from_rank0(
    batch: Dict[str, Any] | None,
    granularity: int,
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Broadcast batch from rank 0 and split across all ranks.
    
    This is used for LoRA training where only rank 0 submits rollouts
    due to a known bug with multi-rank LoRA inference.
    
    Args:
        batch: The batch dict (only valid on rank 0, None on other ranks)
        granularity: Number of samples to keep together (e.g., group_size for GRPO)
        device: Device to move tensors to after broadcast
        
    Returns:
        The local shard of the batch for this rank
    """
    batch = broadcast_tensor_container(batch, src_rank=0)
    # Ensure batch is on the target device after broadcast (for non-rank-0 receivers)
    if device is not None:
        batch = tensor_container_to(batch, device)
    bs = get_batch_size(batch)
    world_size = dist.get_world_size()
    assert bs % world_size == 0, f"Batch size {bs} must be divisible by world_size {world_size}"
    bs_per_rank = bs // world_size
    rank = dist.get_rank()
    local_batch = []
    for i in range(rank * bs_per_rank, (rank + 1) * bs_per_rank):
        local_batch.append({k: v[i : i + 1] for k, v in batch.items()})
    local_batch = concat_padded_tensors(local_batch)
    # Make the sequences on each rank more balanced.
    return redistribute(local_batch, granularity=granularity).data
    

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