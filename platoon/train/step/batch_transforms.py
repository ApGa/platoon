"""The batch transforms: what a training step does to each microbatch's datums before their forward-backward pass
(:func:`run_batch_transforms`, which the training loop calls, :mod:`platoon.train.trainer.loops`), such as weighting
or discounting them by their depth in the trajectory tree, and the zero-advantage and loss-normalization helpers.

These transforms operate at the Tinker microbatch boundary.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import tinker
import torch
from tinker import TensorData

if TYPE_CHECKING:
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig


# Internal per-datum metadata used to preserve Tinker's original action-token
# loss denominator after zero-gradient datums are removed from model compute.
LOSS_NORMALIZATION_TOKENS_KEY = "_loss_normalization_tokens"


def objective_is_zero_for_zero_advantages(loss_fn: str, loss_fn_config: dict) -> bool:
    """Whether zero advantages guarantee a zero objective for a known RL loss.

    Unknown options may add an advantage-independent KL/entropy/other term.
    Be conservative until their semantics are explicitly supported here.
    """

    return loss_fn in {"cispo", "ppo", "importance_sampling"} and not (
        set(loss_fn_config) - {"clip_low_threshold", "clip_high_threshold"}
    )


def has_zero_objective(datums: list[tinker.Datum], loss_fn: str, loss_fn_config: dict) -> bool:
    """Whether a microbatch's objective is exactly zero: every advantage is 0 under a loss for which that zeroes the
    objective.  The trainer then sends it for no forward/backward, whatever ``filter_zero_advantage_datums`` says,
    so a step whose microbatches all have a zero objective takes no optimizer update.

    Exact-zero check; tiny and non-finite advantages must still be processed.
    """

    return (
        bool(datums)
        and objective_is_zero_for_zero_advantages(loss_fn, loss_fn_config)
        and all(
            "advantages" in datum.loss_fn_inputs and bool(torch.all(datum.loss_fn_inputs["advantages"].to_torch() == 0))
            for datum in datums
        )
    )


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


class DepthLevelDiscountTransform:
    """Scale each datum's advantages by ``gamma ** depth`` at the microbatch boundary.

    This is the Tinker port of AReaL's ``depth_level_discount_gamma`` strategy,
    an alternative to inverse-frequency depth weighting. The weights are
    renormalized so that the microbatch's action-token-weighted mean weight is
    one, which preserves its total action-token mass exactly as
    ``DepthLevelWeightingTransform`` does.
    """

    def __init__(self, gamma: float) -> None:
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("depth_level_discount_gamma must be a positive finite number")
        self.gamma = gamma

    def __call__(
        self,
        datums: list[tinker.Datum],
        context: BatchTransformContext,
    ) -> list[tinker.Datum] | None:
        if not datums:
            return datums

        depths: list[int] = []
        action_token_counts: list[float] = []
        for datum in datums:
            if "traj_depth" not in datum.loss_fn_inputs:
                raise ValueError("depth_level_discount_gamma requires traj_depth in tinker datums")
            depths.append(int(datum.loss_fn_inputs["traj_depth"].to_torch().item()))
            action_token_counts.append(float(datum.loss_fn_inputs["mask"].to_torch().sum().item()))

        raw_weights = torch.tensor(self.gamma, dtype=torch.float32) ** torch.tensor(depths, dtype=torch.float32)
        action_token_tensor = torch.tensor(action_token_counts, dtype=torch.float32)
        total_action_tokens = action_token_tensor.sum()
        unnorm_total = (raw_weights * action_token_tensor).sum()
        if unnorm_total <= 0 or total_action_tokens <= 0:
            raise ValueError("depth_level_discount_gamma produced zero total weight for this microbatch")

        weights = raw_weights * (total_action_tokens / unnorm_total)
        for datum, weight in zip(datums, weights):
            advantages = datum.loss_fn_inputs["advantages"].to_torch()
            datum.loss_fn_inputs["advantages"] = TensorData.from_torch(advantages * weight)
        return datums


def summarize_depths(datums: Sequence[tinker.Datum]) -> dict[int, tuple[int, float, float]] | None:
    """Per trajectory depth: datums, trajectories (``traj_start``) and the action tokens' total |advantage|.

    ``None`` when the datums carry no ``traj_depth``, i.e. no depth transform is configured.
    """

    summary: dict[int, tuple[int, float, float]] = {}
    for datum in datums:
        inputs = datum.loss_fn_inputs
        if "traj_depth" not in inputs:
            return None
        depth = int(inputs["traj_depth"].to_torch().item())
        start = float(inputs["traj_start"].to_torch().item()) if "traj_start" in inputs else 0.0
        actions = inputs["mask"].to_torch() > 0
        mass = float(inputs["advantages"].to_torch()[actions].abs().sum().item())
        count, starts, total = summary.get(depth, (0, 0.0, 0.0))
        summary[depth] = (count + 1, starts + start, total + mass)
    return summary or None


def depth_credit_metrics(
    before: dict[int, tuple[int, float, float]],
    after: dict[int, tuple[int, float, float]] | None,
) -> dict[str, float]:
    """A microbatch's depth mix (``summarize_depths`` before the transforms) and per-depth weights.

    ``credit/depth{d}/weight`` is depth ``d``'s total action-token |advantage| after the batch
    transforms over before them, reported for depths that carry any advantage.
    """

    metrics = {"credit/max_depth": float(max(before))}
    for depth, (datums, trajectories, mass) in sorted(before.items()):
        metrics[f"credit/depth{depth}/datums"] = float(datums)
        metrics[f"credit/depth{depth}/trajectories"] = trajectories
        if after is not None and depth in after and mass > 0:
            metrics[f"credit/depth{depth}/weight"] = after[depth][2] / mass
    return metrics


def build_default_batch_transforms(
    config: "PlatoonTinkerRLTrainerConfig",
) -> list[BatchTransform]:
    """Build the default Tinker trainer transforms from the current config.

    ``depth_level_weighting`` and ``depth_level_discount_gamma`` are
    alternative depth strategies; the workflow config rejects setting both.
    """

    workflow_config = config.train.workflow_config
    if workflow_config.depth_level_discount_gamma is not None:
        return [DepthLevelDiscountTransform(workflow_config.depth_level_discount_gamma)]
    if workflow_config.depth_level_weighting:
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
