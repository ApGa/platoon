"""The objective RL steps train under (:class:`PolicyGradient`): one of the Tinker service's policy-gradient losses
(``cispo``, ``ppo``, ``importance_sampling``) with its keyword arguments, the loss inputs that only the trainer reads,
and the training metrics of each microbatch (:func:`compute_training_metrics`: KL, entropy, importance weights and
clipping).
"""

from typing import Any

import tinker
import torch
from tinker_cookbook.rl.metrics import compute_kl_sample_train

from platoon.train.objectives.api import LOSS_NORMALIZATION_TOKENS_KEY


def compute_training_metrics(
    datums: list[tinker.Datum],
    forward_backward_result: tinker.ForwardBackwardOutput,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
) -> dict[str, float]:
    """Compute training metrics from forward_backward results.

    Extracts training logprobs from the forward pass and computes:
    - KL divergence between sampling and training policies
    - Entropy (approximated from sampling logprobs)
    - Importance weight statistics
    - Clipping statistics (for cispo/ppo loss functions)

    Args:
        datums: Original datums with mask and sampling logprobs in loss_fn_inputs
        forward_backward_result: Result from forward_backward containing training logprobs
        loss_fn: Loss function type ('cispo', 'ppo', 'importance_sampling', etc.)
        loss_fn_config: Loss function config with clip thresholds

    Returns:
        Dictionary of training metrics
    """
    # Extract training logprobs from forward_backward result
    training_logprobs_list: list[torch.Tensor] = []
    for output in forward_backward_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_list.append(training_logprobs)

    # Use tinker_cookbook's function for KL and entropy
    kl_metrics = compute_kl_sample_train(datums, training_logprobs_list)

    # Compute importance weight statistics
    all_importance_weights: list[torch.Tensor] = []
    all_clipped_low: list[torch.Tensor] = []
    all_clipped_high: list[torch.Tensor] = []

    for datum, training_logprobs in zip(datums, training_logprobs_list):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0

        # Extract action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            # Importance weight = exp(new_logprob - old_logprob) = π_new / π_old
            log_ratio = training_logprobs_actions - sampling_logprobs_actions
            importance_weights = torch.exp(log_ratio)
            all_importance_weights.append(importance_weights)

            # Compute clipping stats for cispo/ppo (both use clip_low_threshold/clip_high_threshold)
            if loss_fn in ("cispo", "ppo"):
                clip_low = loss_fn_config.get("clip_low_threshold", 0.0)
                clip_high = loss_fn_config.get("clip_high_threshold", 5.0)

                clipped_low = importance_weights < clip_low
                clipped_high = importance_weights > clip_high
                all_clipped_low.append(clipped_low)
                all_clipped_high.append(clipped_high)

    metrics = dict(kl_metrics)

    if all_importance_weights:
        flat_weights = torch.cat(all_importance_weights)
        metrics["optim/importance_weight_mean"] = flat_weights.mean().item()
        metrics["optim/importance_weight_std"] = flat_weights.std().item()
        metrics["optim/importance_weight_min"] = flat_weights.min().item()
        metrics["optim/importance_weight_max"] = flat_weights.max().item()

        # Clipping stats for cispo/ppo
        if loss_fn in ("cispo", "ppo") and all_clipped_low:
            flat_clipped_low = torch.cat(all_clipped_low)
            flat_clipped_high = torch.cat(all_clipped_high)
            total_tokens = len(flat_clipped_low)
            metrics["optim/clip_frac_low"] = flat_clipped_low.sum().item() / total_tokens
            metrics["optim/clip_frac_high"] = flat_clipped_high.sum().item() / total_tokens
            metrics["optim/clip_frac_total"] = (flat_clipped_low | flat_clipped_high).sum().item() / total_tokens

    return metrics


class PolicyGradient:
    """A policy's rows train under the policy-gradient loss ``loss_fn`` with ``loss_fn_config`` (``train.loss_fn``
    and ``train.loss_fn_config``).  Each microbatch's training metrics are :func:`compute_training_metrics`'s, looked
    up in this module at each call, so that a patch of the module's name reaches them."""

    # The loss inputs that only the trainer reads, which the datums sent to forward_backward leave out: the action
    # mask (redundant there, since the advantages of masked tokens are 0), the sampler version the datum came from,
    # its trajectory depth and start, which the batch transforms weight by, and the action tokens it represents, which
    # they keep for the loss's normalization.  The metrics read the mask from the originals.
    side_channels = frozenset({"mask", "checkpoint_version", "traj_depth", "traj_start", LOSS_NORMALIZATION_TOKENS_KEY})

    def __init__(self, loss_fn: str, loss_fn_config: dict[str, Any]) -> None:
        self.loss_fn = loss_fn
        self.loss_fn_config = loss_fn_config

    def metrics(self, originals: list[tinker.Datum], result: tinker.ForwardBackwardOutput) -> dict[str, float]:
        """A microbatch's training metrics: the KL between the sampler and the trained policy, the entropy, the
        importance weights and, for ``cispo`` and ``ppo``, how many of them were clipped."""

        return compute_training_metrics(
            datums=originals,
            forward_backward_result=result,
            loss_fn=self.loss_fn,
            loss_fn_config=self.loss_fn_config,
        )
