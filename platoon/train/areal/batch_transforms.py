"""Batch-level transforms for Platoon's AReaL trainer.

These transforms operate on the full post-rollout training batch after all
per-group rollout math has completed but before PPO advantages are computed.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

import torch

if TYPE_CHECKING:
    from platoon.train.areal.config_defs import PlatoonArealRLTrainerConfig


BatchDict = dict[str, Any]
_ROUTED_EXPERTS_FIELD = "routed_experts"


@lru_cache(maxsize=1)
def _rtensor_type():
    try:
        from areal.infra.rpc.rtensor import RTensor  # pyright: ignore[reportMissingImports]
    except Exception:
        return None
    return RTensor


def localize_rtensors(value: Any) -> Any:
    """Convert AReaL RTensor handles to local torch tensors when present."""

    RTensor = _rtensor_type()
    if RTensor is not None:
        return RTensor.localize(value)
    if hasattr(value, "to_local") and callable(value.to_local):
        return value.to_local()
    if isinstance(value, dict):
        return {key: localize_rtensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localize_rtensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_rtensors(item) for item in value)
    return value


@dataclass(frozen=True)
class BatchTransformContext:
    """Stable trainer-side context exposed to batch transforms."""

    config: "PlatoonArealRLTrainerConfig"
    actor_dp_world_size: int
    global_step: int | None = None
    epoch: int | None = None
    epoch_step: int | None = None


class BatchTransform(Protocol):
    """Callable protocol for full-batch trainer transforms."""

    def __call__(
        self,
        batch: BatchDict,
        context: BatchTransformContext,
    ) -> BatchDict | None: ...


def _tensor_like_batch_size(value: Any) -> int | None:
    if torch.is_tensor(value) and value.ndim >= 1:
        return int(value.shape[0])
    if hasattr(value, "shape") and hasattr(value, "ndim"):
        try:
            if int(value.ndim) >= 1:
                return int(value.shape[0])
        except (TypeError, ValueError, IndexError):
            return None
    return None


def get_batch_size(batch: BatchDict) -> int:
    attention_mask_size = _tensor_like_batch_size(batch.get("attention_mask"))
    if attention_mask_size is not None:
        return attention_mask_size
    input_ids_size = _tensor_like_batch_size(batch.get("input_ids"))
    if input_ids_size is not None:
        return input_ids_size
    for value in batch.values():
        tensor_like_size = _tensor_like_batch_size(value)
        if tensor_like_size is not None:
            return tensor_like_size
        if isinstance(value, list):
            return len(value)
    raise ValueError("Unable to infer batch size from batch contents")


def index_batch(batch: BatchDict, indices: torch.Tensor) -> BatchDict:
    batch = localize_rtensors(batch)
    batch_size = get_batch_size(batch)
    filtered: BatchDict = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
            filtered[key] = value.index_select(0, indices.to(value.device))
        elif isinstance(value, list) and len(value) == batch_size:
            keep_list = indices.cpu().tolist()
            filtered[key] = [value[i] for i in keep_list]
        else:
            filtered[key] = value
    return filtered


def split_batch_to_trajectories(batch: BatchDict) -> list[BatchDict]:
    """Split a batched dict back into per-trajectory items for AReaL dispatch.

    Platoon's trainer-side transforms intentionally operate on a temporary full
    batch view. Before handing control back to AReaL we must restore the
    canonical ``list[dict]`` trajectory representation so the controller can
    repartition work across DP groups instead of treating the whole batch as one
    atomic item.
    """

    batch = localize_rtensors(batch)
    batch_size = get_batch_size(batch)
    if batch_size == 0:
        return []

    traj_seqlens: list[int] | None = None
    attention_mask = batch.get("attention_mask")
    if torch.is_tensor(attention_mask) and attention_mask.ndim >= 2:
        traj_seqlens = attention_mask.sum(dim=-1).tolist()

    split_items: list[BatchDict] = [{} for _ in range(batch_size)]
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
            splits = list(value.split(1, dim=0))
            if traj_seqlens is not None:
                for i, seq_len in enumerate(traj_seqlens):
                    if key == _ROUTED_EXPERTS_FIELD:
                        if splits[i].ndim != 4:
                            raise ValueError(
                                "routed_experts must have shape [B,S,L,K] before trajectory dispatch, "
                                f"got {tuple(splits[i].shape)}"
                            )
                        if splits[i].shape[1] < seq_len:
                            raise ValueError(
                                f"routed_experts sequence width {splits[i].shape[1]} is shorter than "
                                f"attention length {seq_len}"
                            )
                        splits[i] = splits[i][:, :seq_len, :, :].contiguous()
                    elif splits[i].ndim >= 2 and splits[i].shape[-1] > seq_len:
                        splits[i] = splits[i][..., :seq_len]
            for i, split_value in enumerate(splits):
                split_items[i][key] = split_value
        elif isinstance(value, list) and len(value) == batch_size:
            for i, item_value in enumerate(value):
                split_items[i][key] = copy.deepcopy(item_value)
        else:
            for item in split_items:
                item[key] = copy.deepcopy(value)
    return split_items


class DepthLevelWeightingTransform:
    """Apply the original Platoon full-batch depth weighting formula.

    This intentionally runs on the trainer's concatenated rollout batch, not on
    per-task rollout groups, so the normalization matches the pre-migration math.
    """

    def __call__(
        self,
        batch: BatchDict,
        context: BatchTransformContext,
    ) -> BatchDict | None:
        if "traj_depth" not in batch:
            return batch

        rewards = localize_rtensors(batch["rewards"])
        traj_depth = torch.as_tensor(localize_rtensors(batch["traj_depth"]), device=rewards.device, dtype=torch.long)
        depth_indices = traj_depth.reshape(-1)
        depth_gamma = context.config.workflow_config.depth_level_discount_gamma

        if depth_gamma is not None:
            if depth_gamma < 0:
                raise ValueError("workflow_config.depth_level_discount_gamma must be non-negative")
            gamma = torch.tensor(depth_gamma, device=traj_depth.device, dtype=rewards.dtype)
            raw_weights = torch.pow(gamma, depth_indices.to(rewards.dtype))
            raw_weight_sum = raw_weights.sum()
            if raw_weight_sum <= 0:
                raise ValueError("workflow_config.depth_level_discount_gamma produced zero total weight for this batch")
            normalization = (raw_weights.numel() / raw_weight_sum).to(raw_weights.dtype)
            per_datum_weights = raw_weights * normalization
        else:
            traj_start = torch.as_tensor(
                localize_rtensors(batch["traj_start"]),
                device=rewards.device,
                dtype=rewards.dtype,
            ).reshape(-1)
            global_max_depth = int(traj_depth.max().item()) if traj_depth.numel() > 0 else 0
            num_depths = global_max_depth + 1
            counts = torch.zeros(2, num_depths, device=traj_depth.device)
            for depth in range(num_depths):
                mask_d = depth_indices == depth
                counts[0, depth] = mask_d.sum().float()
                counts[1, depth] = traj_start[mask_d].sum()
            datum_counts = counts[0]
            traj_counts = counts[1]

            total_datums = datum_counts.sum()
            raw_weights = torch.where(
                traj_counts > 0,
                1.0 / traj_counts,
                torch.zeros_like(traj_counts),
            )
            unnorm_total = (datum_counts * raw_weights).sum()
            if unnorm_total <= 0:
                del batch["traj_depth"]
                batch.pop("traj_start", None)
                return batch
            normalization = total_datums / unnorm_total
            per_depth_weights = normalization * raw_weights
            per_datum_weights = per_depth_weights[depth_indices]

        batch["rewards"] = rewards * per_datum_weights
        del batch["traj_depth"]
        batch.pop("traj_start", None)
        return batch


def build_default_batch_transforms(
    config: "PlatoonArealRLTrainerConfig",
) -> list[BatchTransform]:
    """Build the default trainer-side transforms from the current config."""

    if config.workflow_config.depth_level_weighting or config.workflow_config.depth_level_discount_gamma is not None:
        return [DepthLevelWeightingTransform()]
    return []


def run_batch_transforms(
    batch: BatchDict,
    transforms: Sequence[BatchTransform],
    context: BatchTransformContext,
) -> BatchDict | None:
    """Run ordered trainer-side transforms on the full batch."""

    current_batch: BatchDict | None = batch
    for transform in transforms:
        if current_batch is None:
            return None
        current_batch = transform(current_batch, context)
    return current_batch
