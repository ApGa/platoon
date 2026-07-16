"""Opt-in FP32 LM-head output support for Platoon Megatron actors.

Pinned AReaL exposes ``megatron.enable_fp32_lm_head`` but does not apply the
setting when models are created through ``megatron-bridge``.  Megatron Core
0.17.0 also has no native flag with that name.  This adapter implements the
documented AReaL behavior at the narrowest stable boundary: the output of the
Megatron language-model head.

The projection itself still runs in the model compute dtype.  Its logits are
cast to FP32 before any log-probability or loss computation.  The cast is
autograd-preserving, so gradients flow back through the original output layer
in its normal dtype.  FP32 logits use twice the storage of BF16 logits and the
cast briefly overlaps both tensors; callers must opt in explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

_HOOK_HANDLE_ATTRIBUTE = "_platoon_fp32_lm_head_hook_handle"


def _cast_lm_head_output_to_fp32(_module, _inputs, output):
    """Cast only the LM-head logits while preserving its output container."""

    if torch.is_tensor(output):
        return output.float()
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        # Megatron ColumnParallelLinear returns ``(output, output_bias)``.  The
        # optional bias must remain untouched; adding BF16 bias to FP32 logits
        # naturally promotes the result while preserving the existing contract.
        return (output[0].float(), *output[1:])
    raise TypeError(
        "Megatron LM head returned an unsupported output; expected a Tensor or "
        "a non-empty tuple whose first item is a Tensor."
    )


def _unwrap_model_chunk(model: Any) -> Any:
    """Unwrap DDP and language-model containers without importing Megatron."""

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


def install_fp32_lm_head_output_hooks(
    model_chunks: Iterable[Any],
    *,
    enabled: bool,
    is_critic: bool,
) -> int:
    """Install one idempotent FP32-output hook per local LM-head chunk.

    Pipeline ranks without the post-process stage legitimately have no output
    layer and therefore install zero hooks.  Critic/value heads are deliberately
    excluded: ``enable_fp32_lm_head`` is a language-model-head option and the
    existing Megatron value head already controls its own output dtype.
    """

    if not enabled or is_critic:
        return 0

    installed = 0
    missing_post_process_head = False
    for model in model_chunks:
        model = _unwrap_model_chunk(model)
        output_layer = getattr(model, "output_layer", None)
        if output_layer is None:
            missing_post_process_head |= bool(getattr(model, "post_process", False))
            continue
        if getattr(output_layer, _HOOK_HANDLE_ATTRIBUTE, None) is not None:
            continue
        handle = output_layer.register_forward_hook(_cast_lm_head_output_to_fp32)
        setattr(output_layer, _HOOK_HANDLE_ATTRIBUTE, handle)
        installed += 1
    if missing_post_process_head:
        raise RuntimeError(
            "FP32 LM-head output was requested, but a post-process Megatron model chunk has no output_layer to adapt."
        )
    return installed
