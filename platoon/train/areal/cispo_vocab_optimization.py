"""Megatron CISPO training path without metric-only vocabulary work."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from platoon.train.areal.fused_linear_logprobs import (
    FusedLinearLogprobsHeadState,
    vocab_parallel_fused_linear_logprobs,
)
from platoon.train.areal.loss_functions import supports_logprobs_only_training


def _validate_logprobs_only_loss(engine: Any, loss_fn: Callable[..., torch.Tensor]) -> None:
    if not supports_logprobs_only_training(loss_fn):
        raise RuntimeError(
            "Logprobs-only Megatron training was requested for a loss that does "
            "not declare entropy and vocabulary diagnostics optional."
        )
    if engine.config.is_critic:
        raise RuntimeError("Logprobs-only CISPO training is actor-only.")
    if engine.enable_tree_training:
        raise RuntimeError("Logprobs-only CISPO training does not support tree training.")


def _labels_and_megatron_module(inputs: dict[str, Any], megatron_engine_module: Any | None) -> tuple[torch.Tensor, Any]:
    if megatron_engine_module is None:
        import areal.engine.megatron_engine as megatron_engine_module  # pyright: ignore[reportMissingImports]

    cp_local_labels = inputs.get("_cp_local_labels")
    if cp_local_labels is not None:
        labels = cp_local_labels
    else:
        labels = torch.roll(inputs["input_ids"], shifts=-1, dims=-1)
    return labels, megatron_engine_module


def _finish_logprobs_only_loss(
    logprobs: torch.Tensor,
    inputs: dict[str, Any],
    loss_fn: Callable[..., torch.Tensor],
    *,
    local_weight: torch.Tensor,
    total_loss_weight: torch.Tensor,
    loss_multiplier: float,
    megatron_engine_module: Any,
) -> torch.Tensor:
    cp_padded_cu_seqlens = inputs.get("_cp_padded_cu_seqlens")
    loss_inputs = inputs
    if cp_padded_cu_seqlens is not None:
        logprobs = megatron_engine_module.reassemble_cp_packed_logprobs(
            logprobs,
            cp_padded_cu_seqlens,
        )
        logprobs = megatron_engine_module.unpad_logits(
            logprobs,
            inputs.get("_cp_padding_length", 0),
            cp_padded_cu_seqlens,
            inputs.get("_cp_old_cu_seqlens"),
        )
        loss_inputs = {key: value for key, value in inputs.items() if not key.startswith("_cp_")}

    loss = loss_fn(
        logprobs,
        None,
        loss_inputs,
        vocab_min_logits=None,
        vocab_max_logits=None,
        vocab_mean_logits=None,
        vocab_norm_logits=None,
    )
    loss_scale = local_weight / total_loss_weight * loss_multiplier
    return loss * loss_scale


def compute_logprobs_only_megatron_loss(
    engine: Any,
    output: torch.Tensor,
    inputs: dict[str, Any],
    loss_fn: Callable[..., torch.Tensor],
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    total_loss_weight: torch.Tensor,
    loss_multiplier: float = 1.0,
    *,
    megatron_engine_module: Any | None = None,
) -> torch.Tensor:
    """Compute an actor loss with selected-token log-probability autograd only.

    AReaL's default Megatron actor path always computes entropy and four
    vocabulary-wide diagnostics.  Platoon's CISPO loss detaches entropy and
    uses it only for telemetry, while ignoring the diagnostics.  This path
    therefore preserves the policy loss and all non-entropy metrics, but does
    not materialize those metric-only values.

    The loss must explicitly advertise support through
    ``supports_logprobs_only_training``.  This makes the optimization fail
    closed if a plugin replaces CISPO with a loss that actually needs entropy.
    """

    _validate_logprobs_only_loss(engine, loss_fn)

    local_weight = loss_weight_fn(inputs)
    if local_weight == 0:
        return output.mean() * 0.0

    labels, megatron_engine_module = _labels_and_megatron_module(inputs, megatron_engine_module)

    tp_group = None
    if megatron_engine_module.mpu.get_tensor_model_parallel_world_size() > 1:
        tp_group = megatron_engine_module.mpu.get_tensor_model_parallel_group()
    logprobs = megatron_engine_module.gather_logprobs(
        output,
        labels,
        temperature=engine.config.temperature,
        tp_group=tp_group,
    )
    return _finish_logprobs_only_loss(
        logprobs,
        inputs,
        loss_fn,
        local_weight=local_weight,
        total_loss_weight=total_loss_weight,
        loss_multiplier=loss_multiplier,
        megatron_engine_module=megatron_engine_module,
    )


def compute_fused_linear_megatron_cispo_loss(
    engine: Any,
    hidden_states: torch.Tensor,
    inputs: dict[str, Any],
    loss_fn: Callable[..., torch.Tensor],
    loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    total_loss_weight: torch.Tensor,
    state: FusedLinearLogprobsHeadState,
    loss_multiplier: float = 1.0,
    *,
    chunk_size: int = 512,
    megatron_engine_module: Any | None = None,
) -> torch.Tensor:
    """Compute exact CISPO log-probabilities without materializing full logits.

    ``hidden_states`` is the conditional LM-head wrapper's TP-mapped output,
    not vocabulary logits.  Projection/softmax is chunked and recomputed in
    backward, so arbitrary detached CISPO coefficients remain exact.
    """

    _validate_logprobs_only_loss(engine, loss_fn)
    labels, megatron_engine_module = _labels_and_megatron_module(inputs, megatron_engine_module)
    weight = state.active_weight
    if weight is None:
        raise RuntimeError("Fused-linear CISPO loss ran before the conditional LM-head wrapper")

    local_weight = loss_weight_fn(inputs)
    if local_weight == 0:
        # Keep both decoder and output-weight DDP hooks participating even for
        # a microbatch with no valid actor tokens.
        return (hidden_states.mean() + weight.mean()) * 0.0

    tp_group = None
    tp_world_size = megatron_engine_module.mpu.get_tensor_model_parallel_world_size()
    if tp_world_size > 1:
        # The output layer's group is authoritative for its vocab partition.
        # Some Megatron versions return equivalent ProcessGroup wrappers rather
        # than the identical Python object from the MPU getter.
        tp_group = state.tp_group
        if torch.distributed.get_world_size(tp_group) != tp_world_size:
            raise RuntimeError("Fused LM-head TP group size does not match the Megatron actor")

    logprobs = vocab_parallel_fused_linear_logprobs(
        hidden_states,
        weight,
        labels,
        temperature=engine.config.temperature,
        tp_group=tp_group,
        chunk_size=chunk_size,
    )
    return _finish_logprobs_only_loss(
        logprobs,
        inputs,
        loss_fn,
        local_weight=local_weight,
        total_loss_weight=total_loss_weight,
        loss_multiplier=loss_multiplier,
        megatron_engine_module=megatron_engine_module,
    )
