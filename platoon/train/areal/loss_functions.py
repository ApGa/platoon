"""Loss functions for Platoon's AReaL integration."""

import functools
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from areal.trainer.ppo.actor import grpo_loss_fn as upstream_grpo_loss_fn
from areal.trainer.ppo.stats import infer_token_denominator
from areal.utils import stats_tracker

from platoon.registry import get_registry

@dataclass(frozen=True)
class LossFnSpec:
    """Registered loss function plus loss-specific default kwargs."""

    fn: Callable
    defaults: dict[str, Any] = field(default_factory=dict)
    signature_fn: Callable | None = None


_LOSS_FN_REGISTRY = get_registry("loss")


def register_loss_fn(
    name: str,
    defaults: dict[str, Any] | None = None,
    signature_fn: Callable | None = None,
):
    """Decorator to register a loss function by name."""

    def decorator(fn: Callable) -> Callable:
        _LOSS_FN_REGISTRY.register(
            name,
            LossFnSpec(fn=fn, defaults=dict(defaults or {}), signature_fn=signature_fn),
            exist_ok=True,
        )
        return fn

    return decorator


def get_loss_fn(name: str) -> Callable:
    """Get a loss function by name."""
    return _LOSS_FN_REGISTRY.get(name).fn


def get_loss_fn_defaults(name: str) -> dict[str, Any]:
    """Get a copy of default kwargs for a registered loss function."""

    return dict(_LOSS_FN_REGISTRY.get(name).defaults)


def list_loss_fns() -> list[str]:
    """List all registered loss functions."""
    return _LOSS_FN_REGISTRY.names()


def _filter_compatible_kwargs(fn: Callable, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def build_loss_fn(
    name: str,
    loss_fn_kwargs: dict[str, Any] | None = None,
    common_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable:
    """Resolve a registered loss and bind defaults, user kwargs, then compatible common kwargs."""

    fn = get_loss_fn(name)
    spec = _LOSS_FN_REGISTRY.get(name)
    loss_specific_kwargs = {**spec.defaults, **(loss_fn_kwargs or {}), **kwargs}
    signature_fn = spec.signature_fn or fn
    filtered_common_kwargs = _filter_compatible_kwargs(signature_fn, common_kwargs or {})
    filtered_kwargs = _filter_compatible_kwargs(signature_fn, {**loss_specific_kwargs, **filtered_common_kwargs})
    return functools.partial(fn, **filtered_kwargs)


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute sequence-level geometric mean ratios and average advantages per sequence.

    This is the GSPO (Group-level Sequence Policy Optimization) variant.
    """
    if log_ratio.ndim == 1:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for 1D tensors (packed format).")

        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        sequence_idx = torch.arange(batch_size, device=log_ratio.device).repeat_interleave(seq_lengths)

        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(batch_size, device=log_ratio.device, dtype=log_ratio.dtype).scatter_add_(
            0, sequence_idx, masked_log_ratio
        )

        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(batch_size, device=advantages.device, dtype=advantages.dtype).scatter_add_(
            0, sequence_idx, masked_advantages
        )

        valid_count_per_seq = (
            torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32)
            .scatter_add_(0, sequence_idx, loss_mask.int())
            .clamp(min=1)
        )

        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(log_ratio.dtype)
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(advantages.dtype)

        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)
        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (loss_mask.sum(dim=1).clamp(min=1))
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        ratio = torch.where(loss_mask, ratio, 0.0)

        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(log_ratio)

    return ratio, advantages


@register_loss_fn(
    "cispo",
    defaults={
        "clip_low_threshold": 0.0,
        "clip_high_threshold": 5.0,
    },
)
def cispo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    clip_low_threshold: float = 0.0,
    clip_high_threshold: float = 5.0,
    importance_sampling_level: str = "token",
    **kwargs,
) -> torch.Tensor:
    """Clipped Importance Sampling Policy Optimization (CISPO) loss function.

    CISPO clips the importance sampling weights and uses them to weight the policy gradient,
    while always passing gradients through log π_θ. This helps maintain signal to all tokens
    and preserves variance.

    Loss: L = -detach(clip(ρ, low, high)) * A * log π_θ

    Where:
        ρ = π_θ / π_old = exp(logprobs - old_logprobs)
        A = advantage

    Args:
        logits: Model output logits [batch, seq, vocab] or [total_tokens, vocab]
        input_data: Dict containing:
            - "input_ids" or "rolled_input_ids": Token labels for logprob computation
            - "logprobs": Old policy log probabilities
            - "advantages": Advantage values
            - "loss_mask": Boolean mask for valid tokens
            - "cu_seqlens": (optional) Cumulative sequence lengths for packed format
        temperature: Sampling temperature (default 1.0)
        clip_low_threshold: Lower clipping bound for importance ratio (default 0)
        clip_high_threshold: Upper clipping bound for importance ratio (default 5)
        importance_sampling_level: "token" for per-token, "sequence" for sequence-level
        **kwargs: Ignored extra arguments for compatibility

    Returns:
        Scalar loss tensor
    """
    entropy = entropy.detach()

    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"].detach()
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    cu_seqlens = input_data.get("cu_seqlens")

    loss_mask_count = loss_mask.count_nonzero() or 1

    # Compute log ratio and importance weight
    log_ratio = logprobs - old_logprobs

    if importance_sampling_level == "sequence":
        # Sequence-level geometric mean
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, advantages, loss_mask, cu_seqlens)
    else:
        # Per-token ratio
        ratio = torch.exp(log_ratio)
        ratio = torch.where(loss_mask, ratio, 0.0)

    # Clip the importance ratio (but not for gradient - detach before using as coefficient)
    clipped_ratio = torch.clamp(ratio, clip_low_threshold, clip_high_threshold)

    # CISPO loss: -detach(clipped_ratio) * advantage * logprob
    # The gradient only flows through logprobs (the log π_θ term)
    cispo_coefficient = clipped_ratio.detach()
    pg_loss = -cispo_coefficient * advantages * logprobs

    # Mask and reduce
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0.0).sum() / loss_mask_count

    # Track where clipping occurred for logging
    clip_low_mask = (ratio < clip_low_threshold).logical_and(loss_mask)
    clip_high_mask = (ratio > clip_high_threshold).logical_and(loss_mask)
    clip_mask = clip_low_mask.logical_or(clip_high_mask)

    # Log training statistics (matching areal's grpo_loss_fn pattern)
    stats_tracker.denominator(
        n_tokens=infer_token_denominator(input_data, loss_mask),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=clip_mask,
        dual_clipped_tokens=torch.zeros_like(clip_mask),
    )

    stats_tracker.stat(
        importance_weight=ratio.detach().float(),
        clamped_importance_weight=cispo_coefficient.float(),
        approx_kl=log_ratio.detach().float(),
        new_logp=logprobs.detach().float(),
        old_logp=old_logprobs.float(),
        entropy=entropy.float(),
        actor_loss=logging_loss.float(),
        denominator="n_valid_tokens",
    )

    return pg_loss


@register_loss_fn("grpo", signature_fn=upstream_grpo_loss_fn)
def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Registry wrapper around upstream AReaL GRPO/PPO loss."""

    return upstream_grpo_loss_fn(logprobs, entropy, input_data, **kwargs)


@register_loss_fn("ppo", signature_fn=upstream_grpo_loss_fn)
def ppo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    """Alias of the upstream clipped PPO loss used by AReaL."""

    return upstream_grpo_loss_fn(logprobs, entropy, input_data, **kwargs)
