"""Loss functions for AReaL RL training.

This module provides a registry of loss functions that can be used with the AReaL backend.
Each loss function follows the same interface:
    - Inputs: logprobs, entropy, input_data, **config_kwargs
    - Returns: (loss: Tensor, stat: dict)

To add a new loss function:
    1. Define the function following the interface
    2. Register it using @register_loss_fn("name")
    3. Add any config parameters to LossFnConfig
"""

from typing import Callable

import torch

from areal.utils import stats_tracker

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
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    clip_low_threshold: float = 0.0,
    clip_high_threshold: float = 5.0,
    importance_sampling_level: str = "token",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Clipped Importance Sampling Policy Optimization (CISPO) loss function.
    
    CISPO clips the importance sampling weights and uses them to weight the policy gradient,
    while always passing gradients through log π_θ. This helps maintain signal to all tokens
    and preserves variance.
    
    Loss: L = -detach(clip(ρ, low, high)) * A * log π_θ
    
    Where:
        ρ = π_θ / π_old = exp(logprobs - old_logprobs)
        A = advantage
    
    Args:
        logprobs: Current policy log probabilities [batch, seq] or [total_tokens]
        entropy: Token entropy (detached, for logging)
        input_data: Dict containing:
            - "logprobs": Old policy log probabilities
            - "advantages": Advantage values
            - "loss_mask": Boolean mask for valid tokens
            - "cu_seqlens": (optional) Cumulative sequence lengths for packed format
        clip_low_threshold: Lower clipping bound for importance ratio (default 0)
        clip_high_threshold: Upper clipping bound for importance ratio (default 5)
        importance_sampling_level: "token" for per-token, "sequence" for sequence-level
        **kwargs: Ignored extra arguments for compatibility
    
    Returns:
        Tuple of (loss, statistics dict)
    """
    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"].detach()
    loss_mask = input_data["loss_mask"].bool()
    cu_seqlens = input_data.get("cu_seqlens")
    
    loss_mask_count = loss_mask.count_nonzero() or 1
    entropy = entropy.detach()
    
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
    
    # Statistics dict compatible with PPO logging
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=clip_mask,
        dual_clip_mask=torch.zeros_like(clip_mask),  # CISPO doesn't use dual clipping
        clipped_ratio=clipped_ratio.detach(),
        clip_low_mask=clip_low_mask,
        clip_high_mask=clip_high_mask,
    )
    
    return pg_loss, stat


@register_loss_fn("grpo")
def grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float = 0.2,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behav_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """GRPO/PPO loss function with standard clipping.
    
    This is a wrapper that calls into the AReaL ppo_actor_loss_fn.
    """
    from areal.utils.functional import ppo_actor_loss_fn
    
    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
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
    
    return loss, stat


# Alias for backwards compatibility
register_loss_fn("ppo")(grpo_loss_fn)


@register_loss_fn("sapo")
def sapo_loss_fn_wrapper(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    importance_sampling_level: str = "token",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """SAPO (Soft Adaptive Policy Optimization) loss function wrapper.
    
    SAPO replaces PPO clipping with soft sigmoid gates.
    """
    from areal.utils.functional import sapo_loss_fn
    
    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    cu_seqlens = input_data.get("cu_seqlens")
    
    loss, stat = sapo_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        tau_pos=sapo_tau_pos,
        tau_neg=sapo_tau_neg,
        loss_mask=loss_mask,
        importance_sampling_level=importance_sampling_level,
        cu_seqlens=cu_seqlens,
    )
    
    return loss, stat


def compute_loss_with_stats(
    loss_fn_name: str,
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    config: LossFnConfig | dict,
) -> tuple[torch.Tensor, dict]:
    """Compute loss and log statistics using the specified loss function.
    
    This is the main entry point for computing losses. It handles:
    1. Looking up the loss function by name
    2. Extracting relevant config parameters
    3. Computing the loss
    4. Logging statistics to the stats tracker
    
    Args:
        loss_fn_name: Name of the loss function to use
        logprobs: Current policy log probabilities
        entropy: Token entropy
        input_data: Dict with training data (advantages, old_logprobs, etc.)
        config: LossFnConfig or dict with loss function parameters
    
    Returns:
        Tuple of (loss, statistics dict)
    """
    loss_fn = get_loss_fn(loss_fn_name)
    
    # Convert config to dict if needed
    if isinstance(config, LossFnConfig):
        config_dict = {
            k: getattr(config, k) 
            for k in config.__dataclass_fields__ 
            if hasattr(config, k)
        }
    else:
        config_dict = config
    
    # Call the loss function
    loss, stat = loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        **config_dict,
    )
    
    # Log statistics
    loss_mask = input_data["loss_mask"].bool()
    
    stats_tracker.denominator(
        n_tokens=torch.ones_like(loss_mask, dtype=torch.bool, device=logprobs.device),
        n_valid_tokens=loss_mask,
        clipped_tokens=stat.get("clip_mask", torch.zeros_like(loss_mask)),
    )
    
    stats_tracker.stat(
        importance_weight=stat["importance_weight"],
        approx_kl=stat["approx_kl"],
        new_logp=logprobs.detach(),
        old_logp=input_data["logprobs"],
        entropy=entropy.float(),
        actor_loss=stat["loss"],
        clip_ratio=stat.get("clip_mask", torch.zeros_like(loss_mask)).float(),
        denominator="n_valid_tokens",
    )
    
    # Log loss-function-specific stats
    if "clipped_ratio" in stat:
        stats_tracker.stat(
            clipped_ratio=stat["clipped_ratio"],
            denominator="n_valid_tokens",
        )
    
    if "behave_imp_weight" in stat:
        stats_tracker.denominator(unclipped_behave_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"],
            behave_approx_kl=stat["behave_approx_kl"],
            denominator="unclipped_behave_tokens",
        )
    
    return loss, stat

