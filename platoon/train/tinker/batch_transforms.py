"""Batch-level transforms for Platoon's Tinker trainer.

These transforms operate at the Tinker microbatch boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import tinker
import torch
from tinker import TensorData

if TYPE_CHECKING:
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig


@dataclass(frozen=True)
class BatchTransformContext:
    """Stable trainer-side context exposed to Tinker batch transforms."""

    config: "PlatoonTinkerRLTrainerConfig"
    train_step: int
    minibatch_num: int
    microbatch_num: int


class BatchTransform(Protocol):
    """Callable protocol for microbatch-scoped trainer transforms."""

    def __call__(
        self,
        datums: list[tinker.Datum],
        context: BatchTransformContext,
    ) -> list[tinker.Datum] | None: ...


class DepthLevelWeightingTransform:
    """Apply the existing Tinker depth-level weighting at the microbatch boundary."""

    def __call__(
        self,
        datums: list[tinker.Datum],
        context: BatchTransformContext,
    ) -> list[tinker.Datum] | None:
        if not datums:
            return datums

        depths: list[int] = []
        traj_starts: list[float] = []
        action_token_counts: list[float] = []
        for datum in datums:
            loss_fn_inputs = datum.loss_fn_inputs
            if "traj_depth" not in loss_fn_inputs or "traj_start" not in loss_fn_inputs:
                raise ValueError("depth_level_weighting requires traj_depth and traj_start in tinker datums")
            depths.append(int(loss_fn_inputs["traj_depth"].to_torch().item()))
            traj_starts.append(float(loss_fn_inputs["traj_start"].to_torch().item()))
            action_token_counts.append(float(loss_fn_inputs["mask"].to_torch().sum().item()))

        depth_tensor = torch.tensor(depths, dtype=torch.long)
        traj_start_tensor = torch.tensor(traj_starts, dtype=torch.float32)
        action_token_tensor = torch.tensor(action_token_counts, dtype=torch.float32)

        num_depths = int(depth_tensor.max().item()) + 1 if len(depths) > 0 else 0
        traj_counts = torch.zeros(num_depths, dtype=torch.float32)
        action_tokens_per_depth = torch.zeros(num_depths, dtype=torch.float32)

        for depth in range(num_depths):
            mask = depth_tensor == depth
            traj_counts[depth] = traj_start_tensor[mask].sum()
            action_tokens_per_depth[depth] = action_token_tensor[mask].sum()

        raw_weights = torch.where(traj_counts > 0, 1.0 / traj_counts, torch.zeros_like(traj_counts))
        total_action_tokens = action_token_tensor.sum()
        unnorm_total = (action_tokens_per_depth * raw_weights).sum()
        if unnorm_total <= 0 or total_action_tokens <= 0:
            raise ValueError("depth_level_weighting produced zero total weight for this microbatch")

        per_depth_weights = raw_weights * (total_action_tokens / unnorm_total)
        for datum, depth in zip(datums, depths):
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            datum.loss_fn_inputs["advantages"] = TensorData.from_torch(advantages * per_depth_weights[depth])
        return datums


def build_default_batch_transforms(
    config: "PlatoonTinkerRLTrainerConfig",
) -> list[BatchTransform]:
    """Build the default Tinker trainer transforms from the current config."""

    if config.train.workflow_config.depth_level_weighting:
        return [DepthLevelWeightingTransform()]
    return []


def run_batch_transforms(
    datums: list[tinker.Datum],
    transforms: Sequence[BatchTransform],
    context: BatchTransformContext,
) -> list[tinker.Datum] | None:
    """Run ordered trainer-side transforms on the current microbatch datums."""

    current_datums: list[tinker.Datum] | None = datums
    for transform in transforms:
        if current_datums is None:
            return None
        current_datums = transform(current_datums, context)
    return current_datums
