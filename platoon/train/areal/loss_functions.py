"""Loss functions for AReaL RL training.

This module provides a registry of loss functions that can be used with the AReaL backend.
Each loss function follows AReaL's train_batch interface:
    - Inputs: logits, input_data, **config_kwargs
    - Returns: loss Tensor (scalar)

The loss functions compute logprobs internally from logits using gather_logprobs_entropy.
Stats are logged directly via stats_tracker inside each loss function.

To add a new loss function:
    1. Define the function following the interface
    2. Register it using @register_loss_fn("name")
    3. Add any config parameters to LossFnConfig
"""

from typing import Callable

import torch

from areal.utils import stats_tracker
from areal.utils.functional import gather_logprobs_entropy

# Import LossFnConfig from config_defs to avoid duplication
from platoon.train.areal.config_defs import LossFnConfig


# Registry for loss functions
_LOSS_FN_REGISTRY: dict[str, Callable] = {}


def register_loss_fn(name: str):
    """Decorator to register a loss function by name."""
    def decorator(fn: Callable) -> Callable:
        _LOSS_FN_REGISTRY[name] = fn
        return fn
    return decorator


def get_loss_fn(name: str) -> Callable:
    """Get a loss function by name."""
    if name not in _LOSS_FN_REGISTRY:
        available = list(_LOSS_FN_REGISTRY.keys())
        raise ValueError(f"Unknown loss function: {name}. Available: {available}")
    return _LOSS_FN_REGISTRY[name]


def list_loss_fns() -> list[str]:
    """List all registered loss functions."""
    return list(_LOSS_FN_REGISTRY.keys())


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
            raise ValueError(
                "cu_seqlens is required for 1D tensors (packed format)."
            )
        
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        sequence_idx = torch.arange(
            batch_size, device=log_ratio.device
        ).repeat_interleave(seq_lengths)
        
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(
            batch_size, device=log_ratio.device, dtype=log_ratio.dtype
        ).scatter_add_(0, sequence_idx, masked_log_ratio)
        
        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(
            batch_size, device=advantages.device, dtype=advantages.dtype
        ).scatter_add_(0, sequence_idx, masked_advantages)
        
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
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / (
            loss_mask.sum(dim=1).clamp(min=1)
        )
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        ratio = torch.where(loss_mask, ratio, 0.0)
        
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(log_ratio)
    
    return ratio, advantages


@register_loss_fn("cispo")
def cispo_loss_fn(
    logits: torch.Tensor,
    input_data: dict,
    temperature: float = 1.0,
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
    # Get labels for computing logprobs (same as areal's grpo_loss_fn)
    labels = input_data.get(
        "rolled_input_ids",
        torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
    )
    
    # Compute logprobs and entropy from logits
    logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
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
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(
            log_ratio, advantages, loss_mask, cu_seqlens
        )
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
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=clip_mask,
        dual_clipped_tokens=torch.zeros_like(clip_mask),
    )
    
    stats_tracker.stat(
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        new_logp=logprobs.detach(),
        old_logp=old_logprobs,
        entropy=entropy.float(),
        actor_loss=logging_loss,
        denominator="n_valid_tokens",
    )
    
    return pg_loss


@register_loss_fn("grpo")
def grpo_loss_fn(
    logits: torch.Tensor,
    input_data: dict,
    temperature: float = 1.0,
    eps_clip: float = 0.2,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behav_imp_weight_cap: float | None = None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    **kwargs,
) -> torch.Tensor:
    """GRPO/PPO loss function with standard clipping.
    
    This matches areal's grpo_loss_fn interface (logits, input_data).
    """
    from areal.utils.functional import ppo_actor_loss_fn
    
    # Get labels for computing logprobs
    labels = input_data.get(
        "rolled_input_ids",
        torch.roll(input_data["input_ids"], shifts=-1, dims=-1),
    )
    
    # Compute logprobs and entropy from logits
    logprobs, entropy = gather_logprobs_entropy(logits, labels, temperature)
    entropy = entropy.detach()
    
    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    prox_logp = input_data.get("prox_logp", old_logprobs)
    cu_seqlens = input_data.get("cu_seqlens")
    
    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=prox_logp,
        old_logprobs=old_logprobs,
        advantages=advantages,
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        loss_mask=loss_mask,
        c_clip=c_clip,
        behav_imp_weight_cap=behav_imp_weight_cap,
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=cu_seqlens,
    )
    
    # Log training statistics (matching areal's grpo_loss_fn pattern)
    stats_tracker.denominator(
        n_tokens=torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device),
        n_valid_tokens=loss_mask.bool(),
        clipped_tokens=stat["clip_mask"],
        dual_clipped_tokens=stat["dual_clip_mask"],
    )
    
    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=old_logprobs,
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        denominator="n_valid_tokens",
    )
    
    return loss


# Alias for backwards compatibility
register_loss_fn("ppo")(grpo_loss_fn)
