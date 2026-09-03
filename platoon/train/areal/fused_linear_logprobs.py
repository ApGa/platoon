"""Chunked fused-linear selected-token log-probabilities for Megatron CISPO.

This is a *pre-logits* optimization: the normal Megatron output projection is
replaced during actor training, and the local vocabulary projection is
recomputed in bounded token chunks inside a custom autograd function.  Unlike
``cross_entropy_loss_fusion`` (which starts after a full logits tensor exists),
this path never creates a ``[all_tokens, local_vocab]`` tensor.

The operation returns one selected-token log-probability per token.  Its
backward accepts arbitrary per-token upstream weights, which is required by
CISPO; it is therefore not equivalent to a scalar mean cross-entropy kernel.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _tp_world_size_and_rank(tp_group: Any | None) -> tuple[int, int]:
    if tp_group is None:
        return 1, 0
    return dist.get_world_size(tp_group), dist.get_rank(tp_group)


def _all_reduce_(tensor: torch.Tensor, op: dist.ReduceOp, tp_group: Any | None) -> None:
    world_size, _ = _tp_world_size_and_rank(tp_group)
    if world_size > 1:
        dist.all_reduce(tensor, op=op, group=tp_group)


def _chunk_logprob_statistics(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    tp_group: Any | None,
    vocab_start: int,
    need_target: bool,
    chunk_observer: Callable[[torch.Tensor], None] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return local-vocab probabilities and optionally global target log-probs."""

    local_logits = F.linear(hidden, weight)
    if chunk_observer is not None:
        chunk_observer(local_logits)
    scaled_logits = local_logits.float().div_(temperature)

    global_max = scaled_logits.amax(dim=-1)
    _all_reduce_(global_max, dist.ReduceOp.MAX, tp_group)
    shifted_logits = scaled_logits.sub_(global_max.unsqueeze(-1))

    logprobs = None
    if need_target:
        local_vocab = weight.shape[0]
        local_labels = labels - vocab_start
        target_is_local = (local_labels >= 0) & (local_labels < local_vocab)
        safe_labels = local_labels.masked_fill(~target_is_local, 0)
        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
        selected = shifted_logits[row_ids, safe_labels].clone()
        selected = selected.masked_fill(~target_is_local, 0.0)

    probabilities = shifted_logits.exp_()
    sum_exp = probabilities.sum(dim=-1)

    if need_target:
        # One SUM collective carries both denominator and selected numerator.
        global_stats = torch.stack((sum_exp, selected), dim=-1)
        _all_reduce_(global_stats, dist.ReduceOp.SUM, tp_group)
        sum_exp, selected = global_stats.unbind(dim=-1)
        logprobs = selected - sum_exp.log()
    else:
        _all_reduce_(sum_exp, dist.ReduceOp.SUM, tp_group)

    if need_target:
        # Forward only needs log-probabilities. Avoid an unused full-vocab
        # normalization pass; backward recomputation needs probabilities.
        probabilities = None
    else:
        probabilities.div_(sum_exp.unsqueeze(-1))
    return probabilities, logprobs


def _lm_head_compute_input(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Match Megatron/autocast's LM-head GEMM input dtype explicitly.

    ``torch.autograd.Function.forward`` does not reliably inherit the ambient
    autocast policy. Megatron models may therefore hand this function FP32
    final-norm activations and a BF16 LM-head weight even though the ordinary
    ``ColumnParallelLinear`` GEMM runs in BF16. The cast remains represented by
    this custom function's backward, which returns dHidden in the original
    activation dtype.
    """

    return hidden.to(dtype=weight.dtype)


class _VocabParallelFusedLinearLogprobs(torch.autograd.Function):
    """Recompute a TP-sharded LM head in backward instead of saving logits."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        tp_group: Any | None,
        chunk_size: int,
        chunk_observer: Callable[[torch.Tensor], None] | None,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        world_size, tp_rank = _tp_world_size_and_rank(tp_group)
        del world_size
        vocab_start = tp_rank * weight.shape[0]

        output = torch.empty(flat_labels.shape, dtype=torch.float32, device=flat_hidden.device)
        for start in range(0, flat_hidden.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_hidden.shape[0])
            hidden_chunk = _lm_head_compute_input(flat_hidden[start:stop], weight)
            _, chunk_logprobs = _chunk_logprob_statistics(
                hidden_chunk,
                weight,
                flat_labels[start:stop],
                temperature=temperature,
                tp_group=tp_group,
                vocab_start=vocab_start,
                need_target=True,
                chunk_observer=chunk_observer,
            )
            assert chunk_logprobs is not None
            output[start:stop] = chunk_logprobs

        ctx.save_for_backward(hidden_states, weight, labels)
        ctx.temperature = temperature
        ctx.tp_group = tp_group
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        return output.reshape(labels.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[Any, ...]:
        hidden_states, weight, labels = ctx.saved_tensors
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_labels = labels.reshape(-1)
        flat_grad_output = grad_output.reshape(-1).float()

        hidden_grad = torch.empty_like(flat_hidden) if ctx.needs_input_grad[0] else None
        # Match Liger/FLA's accumulation policy: sum chunk contributions in
        # FP32, then cast once at the autograd boundary. This avoids repeated
        # BF16 rounding across a long sequence.
        weight_grad = torch.zeros_like(weight, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        local_vocab = weight.shape[0]

        for start in range(0, flat_hidden.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_hidden.shape[0])
            hidden_chunk = flat_hidden[start:stop]
            compute_hidden_chunk = _lm_head_compute_input(hidden_chunk, weight)
            labels_chunk = flat_labels[start:stop]
            probabilities, _ = _chunk_logprob_statistics(
                compute_hidden_chunk,
                weight,
                labels_chunk,
                temperature=ctx.temperature,
                tp_group=ctx.tp_group,
                vocab_start=ctx.vocab_start,
                need_target=False,
            )
            assert probabilities is not None

            upstream = flat_grad_output[start:stop]
            probabilities.mul_(-upstream.unsqueeze(-1))
            local_labels = labels_chunk - ctx.vocab_start
            target_is_local = (local_labels >= 0) & (local_labels < local_vocab)
            safe_labels = local_labels.masked_fill(~target_is_local, 0)
            row_ids = torch.arange(hidden_chunk.shape[0], device=hidden_chunk.device)
            probabilities[row_ids, safe_labels] += upstream * target_is_local
            probabilities.div_(ctx.temperature)

            # The real LM-head GEMM produces model-dtype logits and casts them
            # to FP32 for softmax.  Cast the logit gradient back before the two
            # GEMMs to preserve that autograd boundary.
            logits_grad = probabilities.to(weight.dtype)
            if hidden_grad is not None:
                # The projection input cast's backward restores the original
                # hidden-state dtype after the model-dtype GEMM.
                hidden_grad[start:stop] = torch.mm(logits_grad, weight).to(hidden_chunk.dtype)
            if weight_grad is not None:
                chunk_weight_grad = torch.mm(logits_grad.transpose(0, 1), compute_hidden_chunk)
                weight_grad.add_(chunk_weight_grad.float())

        if hidden_grad is not None:
            hidden_grad = hidden_grad.reshape_as(hidden_states)
        if weight_grad is not None:
            weight_grad = weight_grad.to(weight.dtype)
        return hidden_grad, weight_grad, None, None, None, None, None


def vocab_parallel_fused_linear_logprobs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    tp_group: Any | None = None,
    chunk_size: int = 512,
    _chunk_observer: Callable[[torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Project and select log-probabilities without full-sequence logits.

    ``weight`` is the local contiguous vocabulary shard ``[V/TP, H]``.  The
    hidden states must already have Megatron's output-layer TP mapping applied:
    sequence-parallel hidden states are all-gathered (whose backward performs a
    reduce-scatter), while replicated hidden states use the identity mapping
    whose backward all-reduces dHidden.
    """

    if hidden_states.shape[:-1] != labels.shape:
        raise ValueError(
            "hidden-state leading dimensions must match labels: "
            f"got {tuple(hidden_states.shape)} and {tuple(labels.shape)}"
        )
    if hidden_states.shape[-1] != weight.shape[-1]:
        raise ValueError(f"hidden size {hidden_states.shape[-1]} does not match weight {weight.shape[-1]}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if labels.dtype != torch.long:
        raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}")
    return _VocabParallelFusedLinearLogprobs.apply(
        hidden_states,
        weight,
        labels,
        float(temperature),
        tp_group,
        int(chunk_size),
        _chunk_observer,
    )


@dataclass
class FusedLinearLogprobsHeadState:
    """State shared between the conditional LM-head wrapper and CISPO loss."""

    output_layer: Any
    original_forward: Callable[..., Any]
    tp_group: Any
    active_weight: torch.Tensor | None = None
    fused_forward_calls: int = 0


_ACTIVE_FUSED_HEAD: contextvars.ContextVar[FusedLinearLogprobsHeadState | None] = contextvars.ContextVar(
    "platoon_active_fused_linear_logprobs_head", default=None
)
_HEAD_STATE_ATTRIBUTE = "_platoon_fused_linear_logprobs_state"


def _unwrap_model_chunk(model: Any) -> Any:
    seen: set[int] = set()
    current = model
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "module"):
            current = current.module
            continue
        if hasattr(current, "language_model"):
            current = current.language_model
            continue
        break
    return current


def install_fused_linear_logprobs_head(
    model_chunks: Iterable[Any],
    *,
    copy_to_tp_region: Callable[..., torch.Tensor] | None = None,
    gather_from_sp_region: Callable[..., torch.Tensor] | None = None,
) -> FusedLinearLogprobsHeadState | None:
    """Conditionally bypass one local Megatron LM head during fused training.

    Pipeline ranks without a post-process model legitimately return ``None``.
    The initial integration intentionally supports one output head per rank
    (VPP=1), text-only GPT output, no bias, and no deferred embedding wgrad.
    Calls outside :func:`activate_fused_linear_logprobs_head` retain the exact
    original ``ColumnParallelLinear.forward`` behavior.
    """

    candidates: list[tuple[Any, Any]] = []
    missing_post_process_head = False
    for wrapped_model in model_chunks:
        model = _unwrap_model_chunk(wrapped_model)
        output_layer = getattr(model, "output_layer", None)
        if output_layer is None:
            missing_post_process_head |= bool(getattr(model, "post_process", False))
            continue
        candidates.append((model, output_layer))

    if missing_post_process_head:
        raise RuntimeError("A post-process Megatron model chunk has no output_layer")
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("Fused-linear CISPO currently requires exactly one LM head per pipeline rank (VPP=1)")

    model, output_layer = candidates[0]
    existing = getattr(output_layer, _HEAD_STATE_ATTRIBUTE, None)
    if existing is not None:
        return existing
    if getattr(output_layer, "_platoon_fp32_lm_head_hook_handle", None) is not None:
        raise RuntimeError(
            "Remove the FP32 LM-head output hook before installing fused-linear CISPO; "
            "the fused path performs the projection in model dtype and casts each logits chunk itself"
        )
    if getattr(output_layer, "bias", None) is not None:
        raise RuntimeError("Fused-linear CISPO does not support an LM-head bias")
    if bool(getattr(output_layer, "gather_output", False)):
        raise RuntimeError("Fused-linear CISPO requires vocab-parallel (ungathered) LM-head output")
    if bool(getattr(output_layer, "disable_grad_reduce", False)):
        raise RuntimeError("Fused-linear CISPO does not support disable_grad_reduce on the LM head")
    layer_config = getattr(output_layer, "config", None)
    if bool(getattr(layer_config, "defer_embedding_wgrad_compute", False)):
        raise RuntimeError("Fused-linear CISPO does not support deferred embedding wgrad")
    if bool(getattr(getattr(model, "config", None), "use_mup", False)):
        raise RuntimeError("Fused-linear CISPO does not yet implement MuP output scaling")
    if bool(getattr(model, "mtp_process", False)):
        raise RuntimeError("Fused-linear CISPO does not support MTP output processing")

    if copy_to_tp_region is None or gather_from_sp_region is None:
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            gather_from_sequence_parallel_region,
        )

        copy_to_tp_region = copy_to_tensor_model_parallel_region
        gather_from_sp_region = gather_from_sequence_parallel_region

    original_forward = output_layer.forward
    state = FusedLinearLogprobsHeadState(
        output_layer=output_layer,
        original_forward=original_forward,
        tp_group=output_layer.tp_group,
    )

    def _conditional_forward(
        layer_self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        if _ACTIVE_FUSED_HEAD.get() is not state:
            return original_forward(input_, weight=weight, runtime_gather_output=runtime_gather_output)
        if runtime_gather_output:
            raise RuntimeError("Fused-linear CISPO cannot gather full-vocabulary output")
        active_weight = weight if weight is not None else layer_self.weight
        if active_weight is None:
            raise RuntimeError("Megatron LM head did not supply its shared output weight")
        state.active_weight = active_weight
        state.fused_forward_calls += 1

        if bool(getattr(layer_self, "sequence_parallel", False)):
            mapped_hidden = gather_from_sp_region(input_, group=layer_self.tp_group)
        else:
            mapped_hidden = copy_to_tp_region(input_, group=layer_self.tp_group)
        return mapped_hidden, None

    output_layer.forward = MethodType(_conditional_forward, output_layer)
    setattr(output_layer, _HEAD_STATE_ATTRIBUTE, state)
    return state


@contextlib.contextmanager
def activate_fused_linear_logprobs_head(
    state: FusedLinearLogprobsHeadState | None,
) -> Iterator[None]:
    """Activate an installed head only for the surrounding training schedule."""

    token = _ACTIVE_FUSED_HEAD.set(state)
    try:
        yield
    finally:
        _ACTIVE_FUSED_HEAD.reset(token)


def fused_linear_logprobs_head_is_active(
    state: FusedLinearLogprobsHeadState | None,
) -> bool:
    return state is not None and _ACTIVE_FUSED_HEAD.get() is state
