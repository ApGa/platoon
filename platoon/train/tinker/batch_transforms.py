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


# Internal per-datum metadata used to preserve Tinker's original action-token
# loss denominator after zero-gradient datums are removed from model compute.
LOSS_NORMALIZATION_TOKENS_KEY = "_loss_normalization_tokens"


def has_zero_action_advantage(datum: tinker.Datum) -> bool:
    """Return whether a datum can make no policy-gradient contribution."""

    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0
    action_advantages = advantages[action_mask]
    if action_advantages.numel() == 0:
        return True
    # This optimization is enabled by default, so do not round a small but real
    # policy-gradient signal down to zero.
    return bool(torch.all(action_advantages == 0))


def get_datum_counts(datums: list[tinker.Datum]) -> tuple[int, int, int]:
    """Return datum, attention-token, and masked action-token counts."""

    return (
        len(datums),
        sum(int(datum.model_input.length) for datum in datums),
        sum(int(datum.loss_fn_inputs["mask"].to_torch().sum().item()) for datum in datums),
    )


def filter_zero_advantage_datums(datums: list[tinker.Datum]) -> list[tinker.Datum]:
    """Remove datums whose masked action-token advantages are all zero."""

    return [datum for datum in datums if not has_zero_action_advantage(datum)]


def set_loss_normalization_token_counts(
    datums: list[tinker.Datum],
    *,
    represented_loss_tokens: int,
) -> None:
    """Carry filtered action-token mass without sending its datums to the model."""

    retained_loss_tokens = sum(int(datum.loss_fn_inputs["mask"].to_torch().sum().item()) for datum in datums)
    filtered_loss_tokens = represented_loss_tokens - retained_loss_tokens
    if filtered_loss_tokens < 0:
        raise ValueError("represented_loss_tokens cannot be smaller than retained action-token count")

    for datum_index, datum in enumerate(datums):
        normalization_tokens = float(datum.loss_fn_inputs["mask"].to_torch().sum().item())
        if datum_index == 0:
            normalization_tokens += filtered_loss_tokens
        datum.loss_fn_inputs[LOSS_NORMALIZATION_TOKENS_KEY] = TensorData.from_torch(
            torch.tensor([normalization_tokens], dtype=torch.float32)
        )


def get_loss_normalization_token_count(datums: list[tinker.Datum]) -> float:
    """Return the action-token denominator represented by a microbatch."""

    total = 0.0
    for datum in datums:
        explicit_count = datum.loss_fn_inputs.get(LOSS_NORMALIZATION_TOKENS_KEY)
        if explicit_count is not None:
            total += float(explicit_count.to_torch().sum().item())
        else:
            total += float(datum.loss_fn_inputs["mask"].to_torch().sum().item())
    return total


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
