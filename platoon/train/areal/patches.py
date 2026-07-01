from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from functools import lru_cache, wraps
from typing import Any


def _patch_hf_tokenizer_download_race() -> None:
    """Avoid corrupt Hugging Face tokenizer JSON during proxy worker startup.

    AReaL's helper force-downloads tokenizers. When multiple proxy workers start
    together, they can concurrently rewrite the same HF cache entry, leaving JSON
    files with duplicated content. Load from cache by default and serialize the
    one forced refresh path used to repair a bad cache entry.
    """

    import areal.utils.hf_utils as hf_utils  # pyright: ignore[reportMissingImports]
    import transformers  # pyright: ignore[reportMissingImports]

    original = hf_utils.load_hf_tokenizer
    if getattr(original, "__platoon_hf_tokenizer_patch__", False):
        return

    @lru_cache(maxsize=8)
    def _load_hf_tokenizer_without_racy_force_download(
        model_name_or_path: str,
        fast_tokenizer=True,
        padding_side: str | None = None,
    ) -> transformers.PreTrainedTokenizerFast:
        kwargs = {}
        if padding_side is not None:
            kwargs["padding_side"] = padding_side

        lock_name = hashlib.sha256(model_name_or_path.encode("utf-8")).hexdigest()
        lock_path = os.path.join(tempfile.gettempdir(), f"platoon-hf-tokenizer-{lock_name}.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    fast_tokenizer=fast_tokenizer,
                    trust_remote_code=True,
                    force_download=False,
                    **kwargs,
                )
            except json.JSONDecodeError:
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    fast_tokenizer=fast_tokenizer,
                    trust_remote_code=True,
                    force_download=True,
                    **kwargs,
                )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            return tokenizer

    _load_hf_tokenizer_without_racy_force_download.__platoon_hf_tokenizer_patch__ = True
    hf_utils.load_hf_tokenizer = _load_hf_tokenizer_without_racy_force_download

    try:
        import areal.experimental.openai.proxy.proxy_rollout_server as proxy_server  # pyright: ignore[reportMissingImports]

        proxy_server.load_hf_tokenizer = _load_hf_tokenizer_without_racy_force_download
    except Exception:
        pass


def _patch_model_response_custom_stop_sequences() -> None:
    """Allow AReaL proxy responses that stop on custom text stop sequences.

    Platoon agents use OpenAI ``stop`` sequences such as ``</python>``. SGLang
    reports these as ``stop_reason="stop"`` without necessarily appending the
    tokenizer EOS/PAD token, while this AReaL release requires EOS/PAD for every
    non-length stop. In that case the generated tokens are already the training
    target, so return them unchanged instead of failing the proxy request.
    """

    from areal.api.io_struct import ModelResponse  # pyright: ignore[reportMissingImports]

    original = ModelResponse.output_tokens_without_stop
    if getattr(original.fget, "__platoon_custom_stop_patch__", False):
        return

    def _output_tokens_without_custom_stop_error(self) -> list[int]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is None, cannot get output_tokens_without_stop")
        if self.stop_reason not in ["length", "abort"] and self.output_tokens:
            if not self.end_with_stop:
                return self.output_tokens
            pad_or_eos_len = 0
            eos_id = self.tokenizer.eos_token_id
            pad_id = self.tokenizer.pad_token_id
            stop_tokens = {eos_id, pad_id}
            stop_tokens.discard(None)
            for tok in reversed(self.output_tokens):
                if tok in stop_tokens:
                    pad_or_eos_len += 1
                else:
                    break
            if pad_or_eos_len == len(self.output_tokens):
                raise ValueError(
                    "All output_tokens are EOS or PAD tokens; cannot strip stop tokens without removing entire output."
                )
            return self.output_tokens[:-pad_or_eos_len]
        return self.output_tokens

    _output_tokens_without_custom_stop_error.__platoon_custom_stop_patch__ = True
    ModelResponse.output_tokens_without_stop = property(_output_tokens_without_custom_stop_error)


def _patch_megatron_bridge_attention_backend() -> None:
    """Allow Platoon launchers to force Megatron Bridge attention backend.

    AReaL's public config schema does not expose Megatron Core's
    ``attention_backend`` field, but Megatron Bridge providers do have the field
    before ``finalize()``. Set it from an env var at provider-construction time.
    """

    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # pyright: ignore[reportMissingImports]
        from megatron.core.transformer.enums import AttnBackend  # pyright: ignore[reportMissingImports]
    except Exception:
        return

    original = AutoBridge.to_megatron_provider
    if getattr(original, "__platoon_attention_backend_patch__", False):
        return

    def _forced_attention_backend():
        backend_name = os.environ.get("PLATOON_MEGATRON_ATTENTION_BACKEND", "").strip().lower()
        if not backend_name:
            return None
        try:
            return AttnBackend[backend_name]
        except KeyError as exc:
            allowed = ", ".join(member.name for member in AttnBackend)
            raise ValueError(
                "Invalid PLATOON_MEGATRON_ATTENTION_BACKEND="
                f"{backend_name!r}; expected one of: {allowed}"
            ) from exc

    @wraps(original)
    def _to_megatron_provider_with_forced_attention_backend(self, *args, **kwargs):
        provider = original(self, *args, **kwargs)
        attention_backend = _forced_attention_backend()
        if attention_backend is not None:
            provider.attention_backend = attention_backend
        return provider

    _to_megatron_provider_with_forced_attention_backend.__platoon_attention_backend_patch__ = True
    AutoBridge.to_megatron_provider = _to_megatron_provider_with_forced_attention_backend


def _patch_megatron_bridge_qwen35_tp_validation() -> None:
    """Relax an over-strict Megatron Bridge Qwen3.5 TP validation guard.

    megatron-bridge 0.4.0's Qwen3.5-VL providers require
    ``tensor_model_parallel_size <= num_query_groups``. Megatron Core's actual
    compatibility rule is less restrictive: the two sizes only need to be
    multiples/divisors of each other, with TP > KV groups handled by KV-head
    replication. Newer Megatron Bridge versions have removed this provider-level
    guard, so mirror that behavior here while leaving Core's validation intact.
    """

    try:
        from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (  # pyright: ignore[reportMissingImports]
            Qwen35VLModelProvider,
            Qwen35VLMoEModelProvider,
        )
    except Exception:
        return

    def _allow_megatron_core_to_validate_parallelism(self) -> None:
        return None

    for provider_cls in (Qwen35VLModelProvider, Qwen35VLMoEModelProvider):
        current = getattr(provider_cls, "validate_parallelism", None)
        if getattr(current, "__platoon_qwen35_tp_validation_patch__", False):
            continue
        _allow_megatron_core_to_validate_parallelism.__platoon_qwen35_tp_validation_patch__ = True
        provider_cls.validate_parallelism = _allow_megatron_core_to_validate_parallelism


def _patch_megatron_bridge_qwen35_drop_mtp_for_rl() -> None:
    """Drop Qwen3.5 MTP heads for the experimental GDN CP RL path.

    AReaL commit 4be0c641 made Megatron-Bridge MTP opt-in for RL because MTP is
    not used by rollout/inference and complicates RL training/export. This local
    AReaL checkout predates that ``enable_mtp`` config, so mirror the default
    behavior for the opt-in Qwen3.5 GDN CP path. Set
    ``PLATOON_QWEN35_GDN_CP_ENABLE_MTP=1`` to keep the head for debugging.
    """

    if not _qwen35_gdn_cp_enabled():
        return
    enable_mtp = os.environ.get("PLATOON_QWEN35_GDN_CP_ENABLE_MTP", "").strip().lower()
    if enable_mtp in {"1", "true", "yes", "on"}:
        return

    try:
        from megatron.bridge.models.conversion.auto_bridge import AutoBridge  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron Bridge AutoBridge.") from exc

    original = AutoBridge.to_megatron_provider
    if getattr(original, "__platoon_qwen35_gdn_cp_drop_mtp_patch__", False):
        return

    @wraps(original)
    def _to_megatron_provider_without_mtp_for_gdn_cp(self, *args, **kwargs):
        provider = original(self, *args, **kwargs)
        if (
            getattr(provider, "experimental_attention_variant", None) == "gated_delta_net"
            and getattr(provider, "mtp_num_layers", None)
        ):
            _log_qwen35_gdn_cp_once(
                _to_megatron_provider_without_mtp_for_gdn_cp,
                "drop_mtp",
                "Dropping Qwen3.5 GDN MTP head for CP RL path "
                "(set PLATOON_QWEN35_GDN_CP_ENABLE_MTP=1 to keep it).",
            )
            provider.mtp_num_layers = None
        return provider

    _to_megatron_provider_without_mtp_for_gdn_cp.__platoon_qwen35_gdn_cp_drop_mtp_patch__ = True
    AutoBridge.to_megatron_provider = _to_megatron_provider_without_mtp_for_gdn_cp


def _patch_megatron_bridge_qwen35_cp_per_token_loss() -> None:
    """Enable Qwen3.5 VL wrapper's CP-safe token loss mode for text-only GDN CP."""

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (  # pyright: ignore[reportMissingImports]
            Qwen35VLModelProvider,
            Qwen35VLMoEModelProvider,
        )
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron Bridge Qwen3.5 providers.") from exc

    for provider_cls in (Qwen35VLModelProvider, Qwen35VLMoEModelProvider):
        original = provider_cls.provide
        if getattr(original, "__platoon_qwen35_cp_per_token_loss_patch__", False):
            continue

        @wraps(original)
        def _provide_with_cp_per_token_loss(self, *args, __original=original, **kwargs):
            if (
                getattr(self, "experimental_attention_variant", None) == "gated_delta_net"
                and getattr(self, "context_parallel_size", 1) > 1
            ):
                self.calculate_per_token_loss = True
            return __original(self, *args, **kwargs)

        _provide_with_cp_per_token_loss.__platoon_qwen35_cp_per_token_loss_patch__ = True
        provider_cls.provide = _provide_with_cp_per_token_loss


def _patch_megatron_checkpoint_optimizer_metadata() -> None:
    """Save Megatron distributed optimizer state with a supported sharding mode.

    AReaL's Megatron checkpoint manager calls
    ``optimizer.sharded_state_dict(state_dict)`` without metadata. With newer
    Megatron Core releases that default can produce optimizer shards using
    ``flattened_range``, which ``ShardedTensor.validate_metadata_integrity()``
    rejects. Request the ``dp_reshardable`` strategy instead; it avoids
    ``flattened_range`` and is suitable for recovery checkpoints where DP layout
    is expected to remain stable across resume.

    Newer distributed-optimizer bucket state can also expose padded local bucket
    shards while recording the unpadded bucket length as ``global_shape``. PyTorch
    DCP rejects those plans as out-of-bounds. Keep the padded shard data intact
    and make the checkpoint metadata agree on the padded bucket length globally.
    """

    try:
        from areal.engine.megatron_utils.checkpointer import (  # pyright: ignore[reportMissingImports]
            MegatronCheckpointManager,
        )
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]
        from megatron.core.dist_checkpointing.mapping import (  # pyright: ignore[reportMissingImports]
            ShardedTensor,
        )
    except Exception:
        return

    original = MegatronCheckpointManager.generate_state_dict
    if getattr(original, "__platoon_megatron_optim_metadata_patch__", False):
        return

    def _iter_sharded_tensors(obj):
        if isinstance(obj, ShardedTensor):
            yield obj
        elif isinstance(obj, dict):
            for value in obj.values():
                yield from _iter_sharded_tensors(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from _iter_sharded_tensors(value)

    def _pad_distributed_optimizer_bucket_global_shapes(optimizer_state_dict):
        local_extents = {}
        for sharded_tensor in _iter_sharded_tensors(optimizer_state_dict):
            key = getattr(sharded_tensor, "key", "")
            global_shape = tuple(getattr(sharded_tensor, "global_shape", ()))
            local_shape = tuple(getattr(sharded_tensor, "local_shape", ()))
            global_offset = tuple(getattr(sharded_tensor, "global_offset", ()))
            if (
                "optimizer.distributed" not in key
                or len(global_shape) != 1
                or len(local_shape) != 1
                or len(global_offset) != 1
            ):
                continue
            local_end = global_offset[0] + local_shape[0]
            if local_end > global_shape[0]:
                local_extents[key] = max(local_extents.get(key, global_shape[0]), local_end)

        if dist.is_available() and dist.is_initialized():
            gathered_extents = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_extents, local_extents)
            padded_extents = {}
            for rank_extents in gathered_extents:
                if not rank_extents:
                    continue
                for key, padded_size in rank_extents.items():
                    padded_extents[key] = max(padded_extents.get(key, 0), padded_size)
        else:
            padded_extents = local_extents

        if not padded_extents:
            return optimizer_state_dict

        for sharded_tensor in _iter_sharded_tensors(optimizer_state_dict):
            padded_size = padded_extents.get(getattr(sharded_tensor, "key", ""))
            if padded_size is None:
                continue
            sharded_tensor.global_shape = (padded_size,)

        return optimizer_state_dict

    @wraps(original)
    def _generate_state_dict_with_optimizer_metadata(self, *args, **kwargs):
        optimizer = getattr(self, "optimizer", None)
        original_sharded_state_dict = getattr(optimizer, "sharded_state_dict", None)
        if original_sharded_state_dict is None:
            return original(self, *args, **kwargs)

        @wraps(original_sharded_state_dict)
        def _sharded_state_dict_with_dp_reshardable_metadata(*inner_args, **inner_kwargs):
            inner_kwargs.setdefault(
                "metadata",
                {"distrib_optim_sharding_type": "dp_reshardable"},
            )
            return _pad_distributed_optimizer_bucket_global_shapes(
                original_sharded_state_dict(*inner_args, **inner_kwargs)
            )

        try:
            optimizer.sharded_state_dict = _sharded_state_dict_with_dp_reshardable_metadata
        except Exception:
            return original(self, *args, **kwargs)

        try:
            return original(self, *args, **kwargs)
        finally:
            optimizer.sharded_state_dict = original_sharded_state_dict

    _generate_state_dict_with_optimizer_metadata.__platoon_megatron_optim_metadata_patch__ = True
    MegatronCheckpointManager.generate_state_dict = _generate_state_dict_with_optimizer_metadata


def _qwen35_gdn_cp_enabled() -> bool:
    return _env_truthy("PLATOON_QWEN35_GDN_CP")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _is_global_rank_zero_or_unknown() -> bool:
    rank = os.environ.get("RANK", os.environ.get("SLURM_PROCID"))
    return rank is None or rank == "0"


def _log_qwen35_gdn_cp_once(owner: object, key: str, message: str) -> None:
    if not _is_global_rank_zero_or_unknown():
        return
    attr = f"__platoon_qwen35_gdn_cp_logged_{key}__"
    if getattr(owner, attr, False):
        return
    setattr(owner, attr, True)
    logging.getLogger("PlatoonQwen35GDNCP").warning(message)


def _patch_triton_cache_for_qwen35_gdn_cp() -> None:
    """Avoid shared Triton/FLA autotune cache races across many actor ranks.

    FLA's GDN kernels autotune through Triton the first time they run. With 80
    actor processes starting together, the default shared cache can expose a
    metadata file before the matching ``.cubin`` is visible, which surfaces as
    ``KeyError: "Unknown key: 'cubin'"`` during Triton ``CompiledKernel`` load.
    Isolate caches per process and disable FLA's autotune result cache for this
    opt-in experimental path.
    """

    if not _qwen35_gdn_cp_enabled():
        return

    rank = (
        os.environ.get("SLURM_PROCID")
        or os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("CUDA_VISIBLE_DEVICES", "unknown").replace(",", "_")
    )
    cache_dir = os.path.join(
        tempfile.gettempdir(),
        "platoon-triton-cache",
        f"rank-{rank}-pid-{os.getpid()}",
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_AUTOTUNING"] = "0"
    os.environ["FLA_CACHE_RESULTS"] = "0"

    triton_module = sys.modules.get("triton")
    if triton_module is not None:
        try:
            triton_module.knobs.cache.dir = cache_dir
            triton_module.knobs.autotuning.cache = False
        except Exception:
            pass


def _get_cp_sequence_lengths(cu_seqlens, cp_size: int, local_total_len: int | None = None):
    global_seq_lengths = [(cu_seqlens[i + 1] - cu_seqlens[i]).item() for i in range(len(cu_seqlens) - 1)]
    local_seq_lengths = []
    for global_seq_len in global_seq_lengths:
        if global_seq_len % cp_size != 0:
            raise ValueError(f"Expected sequence length {global_seq_len} to be divisible by cp_size={cp_size}")
        local_seq_lengths.append(global_seq_len // cp_size)

    if local_total_len is not None and sum(local_seq_lengths) != local_total_len:
        raise ValueError(f"Expected local total length {local_total_len}, got {sum(local_seq_lengths)}")
    return global_seq_lengths, local_seq_lengths


def _build_zigzag_cp_text_position_ids(cu_seqlens, cp_rank: int, cp_size: int, device, dtype):
    import torch  # pyright: ignore[reportMissingImports]

    pieces = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        seq_len = int((end - start).item())
        if seq_len % (2 * cp_size) != 0:
            raise ValueError(
                f"Expected sequence length {seq_len} to be divisible by 2 * cp_size={2 * cp_size} "
                "for zigzag CP text position IDs."
            )
        local_len = seq_len // cp_size
        half_len = local_len // 2
        pieces.append(torch.arange(half_len * cp_rank, half_len * (cp_rank + 1), device=device, dtype=dtype))
        pieces.append(torch.arange(seq_len - half_len * (cp_rank + 1), seq_len - half_len * cp_rank, device=device, dtype=dtype))
    return torch.cat(pieces, dim=0) if pieces else torch.empty(0, device=device, dtype=dtype)


_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT = threading.local()


def _patch_megatron_bridge_qwen3vl_already_cp_local_packed_input() -> None:
    """Avoid double CP preprocessing inside Qwen3-VL for AReaL-packed text batches."""

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen3vl_model  # pyright: ignore[reportMissingImports]
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import (  # pyright: ignore[reportMissingImports]
            Qwen3VLModel,
        )
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron Bridge Qwen3-VL model.") from exc

    original_preprocess_packed_seqs = qwen3vl_model.preprocess_packed_seqs
    if not getattr(original_preprocess_packed_seqs, "__platoon_qwen35_already_cp_local_patch__", False):

        @wraps(original_preprocess_packed_seqs)
        def _preprocess_without_second_cp_split(input_ids, attention_mask, pre_process=True, pg_collection=None):
            context = getattr(_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT, "value", None)
            if (
                context is not None
                and pre_process
                and input_ids.dim() >= 2
                and input_ids.size(0) == 1
                and input_ids.size(1) == context["local_total_len"]
            ):
                return input_ids.contiguous(), context["packed_seq_params"]
            return original_preprocess_packed_seqs(input_ids, attention_mask, pre_process, pg_collection)

        _preprocess_without_second_cp_split.__platoon_qwen35_already_cp_local_patch__ = True
        qwen3vl_model.preprocess_packed_seqs = _preprocess_without_second_cp_split

    original_get_rope_index = qwen3vl_model.get_rope_index
    if not getattr(original_get_rope_index, "__platoon_qwen35_already_cp_local_rope_patch__", False):

        @wraps(original_get_rope_index)
        def _get_rope_index_for_already_cp_local_text(
            spatial_merge_size,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            input_ids=None,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
            packed_seq_params=None,
        ):
            context = getattr(_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT, "value", None)
            if (
                context is not None
                and input_ids is not None
                and image_grid_thw is None
                and video_grid_thw is None
                and input_ids.dim() == 2
                and input_ids.size(0) == 1
                and input_ids.size(1) == context["local_total_len"]
            ):
                positions = _build_zigzag_cp_text_position_ids(
                    context["cu_seqlens"],
                    context["cp_rank"],
                    context["cp_size"],
                    input_ids.device,
                    input_ids.dtype,
                )
                position_ids = positions.view(1, 1, -1).expand(3, input_ids.size(0), -1).contiguous()
                mrope_position_deltas = input_ids.new_zeros((input_ids.size(0), 1))
                return position_ids, mrope_position_deltas
            return original_get_rope_index(
                spatial_merge_size,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )

        _get_rope_index_for_already_cp_local_text.__platoon_qwen35_already_cp_local_rope_patch__ = True
        qwen3vl_model.get_rope_index = _get_rope_index_for_already_cp_local_text

    original_forward = Qwen3VLModel.forward
    if getattr(original_forward, "__platoon_qwen35_already_cp_local_forward_patch__", False):
        return

    @wraps(original_forward)
    def _forward_with_already_cp_local_context(self, input_ids, *args, **kwargs):
        packed_seq_params = kwargs.get("packed_seq_params", None)
        if packed_seq_params is None and len(args) >= 6:
            packed_seq_params = args[5]

        context = None
        if (
            packed_seq_params is not None
            and input_ids is not None
            and input_ids.dim() == 2
            and input_ids.size(0) == 1
            and getattr(self.pg_collection, "cp", None) is not None
            and self.pg_collection.cp.size() > 1
            and kwargs.get("pixel_values", None) is None
            and kwargs.get("pixel_values_videos", None) is None
            and kwargs.get("image_grid_thw", None) is None
            and kwargs.get("video_grid_thw", None) is None
        ):
            cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
            cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else packed_seq_params.cu_seqlens_q
            cp_size = self.pg_collection.cp.size()
            _, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size)
            local_total_len = sum(local_seq_lengths)
            if input_ids.numel() == local_total_len:
                context = {
                    "packed_seq_params": packed_seq_params,
                    "cu_seqlens": cu_seqlens,
                    "cp_rank": self.pg_collection.cp.rank(),
                    "cp_size": cp_size,
                    "local_total_len": local_total_len,
                }

        previous = getattr(_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT, "value", None)
        rotary_pos_emb = getattr(getattr(self, "language_model", None), "rotary_pos_emb", None)
        previous_is_thd_format = getattr(rotary_pos_emb, "is_thd_format", None)
        if context is not None:
            _QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = context
            if previous_is_thd_format is not None:
                # Position IDs are already CP-local for AReaL's packed THD path.
                # Avoid Qwen's RoPE module applying a second CP slice to them.
                rotary_pos_emb.is_thd_format = True
        try:
            return original_forward(self, input_ids, *args, **kwargs)
        finally:
            if previous_is_thd_format is not None:
                rotary_pos_emb.is_thd_format = previous_is_thd_format
            _QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = previous

    _forward_with_already_cp_local_context.__platoon_qwen35_already_cp_local_forward_patch__ = True
    Qwen3VLModel.forward = _forward_with_already_cp_local_context


def _patch_megatron_core_qwen35_mtp_local_thd_rope() -> None:
    """Let MTP consume Qwen's already-CP-local THD RoPE table without re-slicing it."""

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        from megatron.core.models.common.embeddings import rope_utils  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron-Core RoPE utilities.") from exc

    original_apply_thd = rope_utils._apply_rotary_pos_emb_thd
    if getattr(original_apply_thd, "__platoon_qwen35_mtp_local_thd_rope_patch__", False):
        return

    @wraps(original_apply_thd)
    def _apply_rotary_pos_emb_thd_with_local_freqs(
        t,
        cu_seqlens,
        freqs,
        rotary_interleaved=False,
        multi_latent_attention=False,
        mscale=1.0,
        cp_group=None,
    ):
        context = getattr(_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT, "value", None)
        if (
            context is not None
            and cp_group is not None
            and cp_group.size() > 1
            and freqs is not None
            and freqs.dim() >= 1
            and context.get("local_total_len") == t.size(0)
            and freqs.size(0) == t.size(0)
        ):
            return rope_utils._apply_rotary_pos_emb_bshd(
                t.unsqueeze(1),
                freqs,
                rotary_interleaved=rotary_interleaved,
                multi_latent_attention=multi_latent_attention,
                mscale=mscale,
            ).squeeze(1)
        return original_apply_thd(
            t,
            cu_seqlens,
            freqs,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=multi_latent_attention,
            mscale=mscale,
            cp_group=cp_group,
        )

    _apply_rotary_pos_emb_thd_with_local_freqs.__platoon_qwen35_mtp_local_thd_rope_patch__ = True
    rope_utils._apply_rotary_pos_emb_thd = _apply_rotary_pos_emb_thd_with_local_freqs


def _patch_megatron_core_mtp_checkpoint_non_tensor_kwargs() -> None:
    """Keep MTP activation recompute from saving PackedSeqParams in autograd state."""

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.distributed as dist  # pyright: ignore[reportMissingImports]
        from megatron.core import tensor_parallel  # pyright: ignore[reportMissingImports]
        from megatron.core.transformer.multi_token_prediction import (  # pyright: ignore[reportMissingImports]
            MultiTokenPredictionLayer,
        )
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron-Core MTP checkpointing.") from exc

    original_checkpointed_forward = MultiTokenPredictionLayer._checkpointed_forward
    if getattr(original_checkpointed_forward, "__platoon_qwen35_mtp_checkpoint_non_tensor_patch__", False):
        return

    @wraps(original_checkpointed_forward)
    def _checkpointed_forward_with_non_tensor_closure(self, forward_func, *args, **kwargs):
        if self.config.fp8:
            return original_checkpointed_forward(self, forward_func, *args, **kwargs)

        def checkpoint_handler():
            tensor_args = []
            arg_specs = []
            for arg in args:
                if torch.is_tensor(arg):
                    arg_specs.append(("tensor", len(tensor_args)))
                    tensor_args.append(arg)
                else:
                    arg_specs.append(("value", arg))

            kw_specs = {}
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    kw_specs[key] = ("tensor", len(tensor_args))
                    tensor_args.append(value)
                else:
                    kw_specs[key] = ("value", value)

            def _forward_with_closed_non_tensors(*runtime_tensor_args):
                rebuilt_args = [
                    runtime_tensor_args[spec[1]] if spec[0] == "tensor" else spec[1] for spec in arg_specs
                ]
                rebuilt_kwargs = {
                    key: runtime_tensor_args[spec[1]] if spec[0] == "tensor" else spec[1]
                    for key, spec in kw_specs.items()
                }

                packed_seq_params = rebuilt_kwargs.get("packed_seq_params", None)
                cp_group = getattr(self, "cp_group", None)
                context = None
                if packed_seq_params is not None and cp_group is not None and cp_group.size() > 1:
                    cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
                    cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else packed_seq_params.cu_seqlens_q
                    _, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_group.size())
                    local_total_len = sum(local_seq_lengths)
                    context = {
                        "packed_seq_params": packed_seq_params,
                        "cu_seqlens": cu_seqlens,
                        "cp_rank": dist.get_rank(group=cp_group),
                        "cp_size": cp_group.size(),
                        "local_total_len": local_total_len,
                    }

                previous = getattr(_QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT, "value", None)
                if context is not None:
                    _QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = context
                try:
                    return forward_func(*rebuilt_args, **rebuilt_kwargs)
                finally:
                    _QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = previous

            return tensor_parallel.checkpoint(
                _forward_with_closed_non_tensors,
                self.config.distribute_saved_activations,
                *tensor_args,
            )

        if self.config.recompute_method == "uniform":
            assert self.config.recompute_num_layers == 1, "recompute_num_layers must be 1 for MTP recompute"
            return checkpoint_handler()
        if self.config.recompute_method == "block":
            return forward_func(*args, **kwargs)
        raise ValueError("Invalid activation recompute method.")

    _checkpointed_forward_with_non_tensor_closure.__platoon_qwen35_mtp_checkpoint_non_tensor_patch__ = True
    MultiTokenPredictionLayer._checkpointed_forward = _checkpointed_forward_with_non_tensor_closure


def _patch_megatron_core_mtp_aux_loss_scaling_for_areal_cp() -> None:
    """Normalize Megatron auxiliary losses explicitly for AReaL's PPO loss path.

    Megatron's per-token-loss mode expects the main loss function to return
    ``(loss_sum, num_tokens, metrics)`` so ``finalize_model_grads`` can divide
    all gradients, including MTP and MoE auxiliary-loss autoscaler gradients, by
    global tokens. AReaL's PPO path returns ``(already_normalized_loss,
    metrics)`` instead, so ``num_tokens`` stays zero and auxiliary losses need
    their own explicit normalization. Under CP this shows up as huge grad norms.
    """

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        from megatron.core import parallel_state  # pyright: ignore[reportMissingImports]
        from megatron.core.pipeline_parallel import schedules  # pyright: ignore[reportMissingImports]
        from megatron.core.transformer.moe import moe_utils  # pyright: ignore[reportMissingImports]
        from megatron.core.transformer.moe.router import TopKRouter  # pyright: ignore[reportMissingImports]
        import megatron.core.transformer.multi_token_prediction as mtp  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron-Core aux-loss scaling.") from exc

    original_process_mtp_loss = mtp.process_mtp_loss
    if not getattr(original_process_mtp_loss, "__platoon_qwen35_mtp_areal_cp_process_loss_patch__", False):

        @wraps(original_process_mtp_loss)
        def _process_mtp_loss_with_explicit_areal_cp_normalization(*args, **kwargs):
            config = kwargs.get("config", None)
            if config is None and len(args) >= 10:
                config = args[9]
            cp_group = kwargs.get("cp_group", None)
            if cp_group is None and len(args) >= 11:
                cp_group = args[10]

            should_use_explicit_normalization = (
                config is not None
                and getattr(config, "calculate_per_token_loss", False)
                and cp_group is not None
                and cp_group.size() > 1
            )
            if not should_use_explicit_normalization:
                return original_process_mtp_loss(*args, **kwargs)

            previous = config.calculate_per_token_loss
            config.calculate_per_token_loss = False
            try:
                # Use MTP's legacy explicit normalization path:
                # sum(loss * mask) / rolled_num_tokens. AReaL's main PPO loss
                # is already normalized and does not drive finalize_model_grads'
                # token-count division.
                return original_process_mtp_loss(*args, **kwargs)
            finally:
                config.calculate_per_token_loss = previous

        _process_mtp_loss_with_explicit_areal_cp_normalization.__platoon_qwen35_mtp_areal_cp_process_loss_patch__ = True
        mtp.process_mtp_loss = _process_mtp_loss_with_explicit_areal_cp_normalization
        try:
            import megatron.core.models.gpt.gpt_model as gpt_model  # pyright: ignore[reportMissingImports]

            # GPTModel imports process_mtp_loss by value, so update the already
            # imported module reference as well as the source MTP module.
            gpt_model.process_mtp_loss = _process_mtp_loss_with_explicit_areal_cp_normalization
        except Exception:
            pass

    def _router_needs_explicit_areal_cp_normalization(router) -> bool:
        cp_group = getattr(router, "cp_group", None)
        return (
            getattr(getattr(router, "config", None), "calculate_per_token_loss", False)
            and cp_group is not None
            and cp_group.size() > 1
        )

    def _disable_moe_aux_for_diagnostics(router) -> bool:
        return _env_truthy("PLATOON_QWEN35_GDN_CP_DISABLE_MOE_AUX") and _router_needs_explicit_areal_cp_normalization(router)

    original_attach_aux_loss = TopKRouter.attach_and_log_load_balancing_loss
    if not getattr(original_attach_aux_loss, "__platoon_qwen35_moe_areal_cp_aux_patch__", False):

        @wraps(original_attach_aux_loss)
        def _attach_aux_loss_with_explicit_areal_cp_normalization(self, *args, **kwargs):
            if _disable_moe_aux_for_diagnostics(self):
                _log_qwen35_gdn_cp_once(
                    _attach_aux_loss_with_explicit_areal_cp_normalization,
                    "disable_moe_aux",
                    "Disabling MoE aux load-balancing autograd hooks for Qwen3.5 GDN CP diagnostic run.",
                )
                return args[0] if args else kwargs["activation"]
            if not _router_needs_explicit_areal_cp_normalization(self):
                return original_attach_aux_loss(self, *args, **kwargs)

            previous = self.calculate_per_token_loss
            self.calculate_per_token_loss = False
            try:
                # AReaL's PPO loss path does not drive Megatron's final
                # token-count gradient divisor, so avoid multiplying MoE aux
                # loss by local token count here.
                return original_attach_aux_loss(self, *args, **kwargs)
            finally:
                self.calculate_per_token_loss = previous

        _attach_aux_loss_with_explicit_areal_cp_normalization.__platoon_qwen35_moe_areal_cp_aux_patch__ = True
        TopKRouter.attach_and_log_load_balancing_loss = _attach_aux_loss_with_explicit_areal_cp_normalization

    original_apply_z_loss = TopKRouter.apply_z_loss
    if not getattr(original_apply_z_loss, "__platoon_qwen35_moe_areal_cp_z_patch__", False):

        @wraps(original_apply_z_loss)
        def _apply_z_loss_with_explicit_areal_cp_normalization(self, *args, **kwargs):
            if _disable_moe_aux_for_diagnostics(self):
                _log_qwen35_gdn_cp_once(
                    _apply_z_loss_with_explicit_areal_cp_normalization,
                    "disable_moe_z",
                    "Disabling MoE router z-loss autograd hooks for Qwen3.5 GDN CP diagnostic run.",
                )
                return args[0] if args else kwargs["logits"]
            if not _router_needs_explicit_areal_cp_normalization(self):
                return original_apply_z_loss(self, *args, **kwargs)

            previous = self.calculate_per_token_loss
            self.calculate_per_token_loss = False
            try:
                return original_apply_z_loss(self, *args, **kwargs)
            finally:
                self.calculate_per_token_loss = previous

        _apply_z_loss_with_explicit_areal_cp_normalization.__platoon_qwen35_moe_areal_cp_z_patch__ = True
        TopKRouter.apply_z_loss = _apply_z_loss_with_explicit_areal_cp_normalization

    original_forward_step_calc_loss = schedules.forward_step_calc_loss
    if getattr(original_forward_step_calc_loss, "__platoon_qwen35_mtp_areal_cp_scale_patch__", False):
        if getattr(original_forward_step_calc_loss, "__platoon_qwen35_moe_areal_cp_scale_patch__", False):
            return
        # Older live interpreters may already have the MTP-only wrapper
        # installed. Re-wrap the original function instead of nesting the old
        # wrapper, which would double-apply MTP scaling.
        original_forward_step_calc_loss = getattr(
            original_forward_step_calc_loss,
            "__wrapped__",
            original_forward_step_calc_loss,
        )

    @wraps(original_forward_step_calc_loss)
    def _forward_step_calc_loss_with_areal_cp_mtp_scale(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size=None,
        is_last_stage=None,
    ):
        original_set_loss_scale = mtp.MTPLossAutoScaler.set_loss_scale
        original_set_moe_loss_scale = moe_utils.MoEAuxLossAutoScaler.set_loss_scale

        def _set_mtp_loss_scale_with_areal_cp_normalization(scale):
            if (
                getattr(config, "calculate_per_token_loss", False)
                and getattr(config, "mtp_num_layers", None) is not None
                and (cp_group_size or parallel_state.get_context_parallel_world_size()) > 1
            ):
                dp_cp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
                scale = scale / max(1, int(num_microbatches)) / max(1, int(dp_cp_size))
            return original_set_loss_scale(scale)

        def _set_moe_loss_scale_with_areal_cp_normalization(scale):
            if (
                getattr(config, "calculate_per_token_loss", False)
                and getattr(config, "num_moe_experts", None) is not None
                and (cp_group_size or parallel_state.get_context_parallel_world_size()) > 1
            ):
                cp_size = cp_group_size or parallel_state.get_context_parallel_world_size()
                dp_cp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=True)
                # Match Megatron's non-per-token MoE aux-loss semantics:
                # scale by cp_size/num_microbatches, with the missing DDP
                # average supplied explicitly because calculate_per_token_loss
                # disables DDP's built-in 1/dp_cp scaling.
                scale = scale * int(cp_size) / max(1, int(num_microbatches)) / max(1, int(dp_cp_size))
            return original_set_moe_loss_scale(scale)

        mtp.MTPLossAutoScaler.set_loss_scale = staticmethod(_set_mtp_loss_scale_with_areal_cp_normalization)
        moe_utils.MoEAuxLossAutoScaler.set_loss_scale = staticmethod(_set_moe_loss_scale_with_areal_cp_normalization)
        try:
            return original_forward_step_calc_loss(
                model,
                output_tensor,
                loss_func,
                config,
                vp_stage,
                collect_non_loss_data,
                num_microbatches,
                forward_data_store,
                cp_group_size=cp_group_size,
                is_last_stage=is_last_stage,
            )
        finally:
            mtp.MTPLossAutoScaler.set_loss_scale = original_set_loss_scale
            moe_utils.MoEAuxLossAutoScaler.set_loss_scale = original_set_moe_loss_scale

    _forward_step_calc_loss_with_areal_cp_mtp_scale.__platoon_qwen35_mtp_areal_cp_scale_patch__ = True
    _forward_step_calc_loss_with_areal_cp_mtp_scale.__platoon_qwen35_moe_areal_cp_scale_patch__ = True
    schedules.forward_step_calc_loss = _forward_step_calc_loss_with_areal_cp_mtp_scale


def _gather_cp_tensors(tensor, cp_group):
    import torch  # pyright: ignore[reportMissingImports]
    import torch.distributed as dist  # pyright: ignore[reportMissingImports]

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=cp_group))]
    dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
    return gathered


def _zigzag_to_packed_shard_impl(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
    import torch  # pyright: ignore[reportMissingImports]

    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    gathered_by_rank = [
        gathered.split(local_seq_lengths, dim=0) for gathered in _gather_cp_tensors(hidden_states, cp_group)
    ]

    full_sequences = []
    for seq_idx, global_seq_len in enumerate(global_seq_lengths):
        per_rank = [rank_seqs[seq_idx] for rank_seqs in gathered_by_rank]
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            full_seq = torch.cat(
                [seq[:subchunk_len] for seq in per_rank] + [seq[subchunk_len:] for seq in per_rank][::-1],
                dim=0,
            )
        else:
            full_seq = torch.cat(per_rank, dim=0)
        full_sequences.append(full_seq)

    full_stream = torch.cat(full_sequences, dim=0) if full_sequences else hidden_states[:0]
    shard_len = hidden_states.size(0)
    return full_stream[cp_rank * shard_len : (cp_rank + 1) * shard_len]


def _packed_shard_to_zigzag_impl(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
    import torch  # pyright: ignore[reportMissingImports]

    global_seq_lengths, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size, hidden_states.size(0))
    full_stream = torch.cat(_gather_cp_tensors(hidden_states, cp_group), dim=0)
    full_sequences = full_stream.split(global_seq_lengths, dim=0)

    local_sequences = []
    for full_seq, global_seq_len, local_seq_len in zip(
        full_sequences,
        global_seq_lengths,
        local_seq_lengths,
        strict=True,
    ):
        if global_seq_len % (2 * cp_size) == 0:
            subchunk_len = global_seq_len // (2 * cp_size)
            parts = full_seq.split(subchunk_len, dim=0)
            local_sequences.append(torch.cat([parts[cp_rank], parts[2 * cp_size - 1 - cp_rank]], dim=0))
        else:
            local_sequences.append(full_seq.split(local_seq_len, dim=0)[cp_rank])

    return torch.cat(local_sequences, dim=0) if local_sequences else hidden_states[:0]


class _ZigzagToPackedShard:
    @staticmethod
    def apply(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
        import torch  # pyright: ignore[reportMissingImports]

        class _AutogradFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, states, seq_lens):
                ctx.cp_group = cp_group
                ctx.cp_rank = cp_rank
                ctx.cp_size = cp_size
                ctx.save_for_backward(seq_lens)
                return _zigzag_to_packed_shard_impl(states, seq_lens, cp_group, cp_rank, cp_size)

            @staticmethod
            def backward(ctx, grad_output):
                (seq_lens,) = ctx.saved_tensors
                result = _packed_shard_to_zigzag_impl(
                    grad_output,
                    seq_lens,
                    ctx.cp_group,
                    ctx.cp_rank,
                    ctx.cp_size,
                )
                return result, None

        return _AutogradFn.apply(hidden_states, cu_seqlens)


class _PackedShardToZigzag:
    @staticmethod
    def apply(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
        import torch  # pyright: ignore[reportMissingImports]

        class _AutogradFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, states, seq_lens):
                ctx.cp_group = cp_group
                ctx.cp_rank = cp_rank
                ctx.cp_size = cp_size
                ctx.save_for_backward(seq_lens)
                return _packed_shard_to_zigzag_impl(states, seq_lens, cp_group, cp_rank, cp_size)

            @staticmethod
            def backward(ctx, grad_output):
                (seq_lens,) = ctx.saved_tensors
                result = _zigzag_to_packed_shard_impl(
                    grad_output,
                    seq_lens,
                    ctx.cp_group,
                    ctx.cp_rank,
                    ctx.cp_size,
                )
                return result, None

        return _AutogradFn.apply(hidden_states, cu_seqlens)


def _zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
    return _ZigzagToPackedShard.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def _packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank: int, cp_size: int):
    return _PackedShardToZigzag.apply(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size)


def _maybe_gather_sequence_parallel_for_gdn(module, hidden_states, expected_seq_len: int):
    """Gather GDN activations that are still sequence-parallel sharded.

    Some Megatron Bridge Qwen3-VL paths feed GDN with SP-sharded activations even
    after the input projection. The packed CP relayout operates on the complete
    CP-local sequence, so gather along sequence dimension when the observed shape
    shows an extra SP split.
    """

    if hidden_states.size(0) == expected_seq_len:
        return hidden_states
    if not getattr(module.config, "sequence_parallel", False):
        raise ValueError(f"Packed GDN CP expected local sequence length {expected_seq_len}, got {hidden_states.size(0)}")
    if expected_seq_len % hidden_states.size(0) != 0:
        raise ValueError(f"Packed GDN CP expected local sequence length {expected_seq_len}, got {hidden_states.size(0)}")

    from megatron.core import tensor_parallel  # pyright: ignore[reportMissingImports]

    gathered = tensor_parallel.gather_from_sequence_parallel_region(
        hidden_states,
        tensor_parallel_output_grad=True,
        group=module.pg_collection.tp,
    )
    if gathered.size(0) != expected_seq_len:
        raise ValueError(
            "Packed GDN CP sequence-parallel gather produced length "
            f"{gathered.size(0)}, expected {expected_seq_len}; "
            f"input length was {hidden_states.size(0)} and TP group size is {module.pg_collection.tp.size()}."
        )
    return gathered


def _get_gdn_cp_group_info(module):
    from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]

    try:
        cp_group = module.pg_collection.cp
        return cp_group, cp_group.rank(), cp_group.size()
    except Exception:
        cp_group = mpu.get_context_parallel_group()
        return cp_group, mpu.get_context_parallel_rank(), mpu.get_context_parallel_world_size()


def _get_gdn_cp_group_candidates(module, packed_seq_params=None):
    from megatron.core import parallel_state as mpu  # pyright: ignore[reportMissingImports]

    candidates = []

    cp_group = getattr(packed_seq_params, "cp_group", None)
    local_cp_size = getattr(packed_seq_params, "local_cp_size", None)
    if cp_group is not None:
        candidates.append((cp_group, cp_group.rank(), cp_group.size(), "packed_seq_params.cp_group"))
    elif local_cp_size is not None and local_cp_size <= 1:
        candidates.append((None, 0, int(local_cp_size), "packed_seq_params.local_cp_size"))

    try:
        cp_group = mpu.get_context_parallel_group()
        candidates.append(
            (
                cp_group,
                mpu.get_context_parallel_rank(),
                mpu.get_context_parallel_world_size(),
                "parallel_state.context_parallel_group",
            )
        )
    except Exception:
        pass

    try:
        cp_group = module.pg_collection.cp
        candidates.append((cp_group, cp_group.rank(), cp_group.size(), "module.pg_collection.cp"))
    except Exception:
        pass

    unique = []
    seen = set()
    for cp_group, cp_rank, cp_size, source in candidates:
        key = (id(cp_group), cp_rank, cp_size)
        if key in seen:
            continue
        seen.add(key)
        unique.append((cp_group, cp_rank, cp_size, source))
    return unique


def _select_gdn_cp_group_for_tensor(module, packed_seq_params, cu_seqlens, hidden_states):
    import torch.distributed as dist  # pyright: ignore[reportMissingImports]

    diagnostics = []
    for cp_group, cp_rank, cp_size, source in _get_gdn_cp_group_candidates(module, packed_seq_params):
        if cp_size <= 1:
            return cp_group, cp_rank, cp_size, hidden_states
        if cp_group is None:
            diagnostics.append(f"{source}: missing process group for cp_size={cp_size}")
            continue
        if dist.get_world_size(group=cp_group) != cp_size:
            diagnostics.append(
                f"{source}: group world size {dist.get_world_size(group=cp_group)} does not match cp_size={cp_size}"
            )
            continue

        _, local_seq_lengths = _get_cp_sequence_lengths(cu_seqlens, cp_size)
        expected_seq_len = sum(local_seq_lengths)
        if hidden_states.size(0) == expected_seq_len:
            return cp_group, cp_rank, cp_size, hidden_states

        try:
            gathered = _maybe_gather_sequence_parallel_for_gdn(module, hidden_states, expected_seq_len)
        except Exception as exc:
            diagnostics.append(
                f"{source}: expected local sequence length {expected_seq_len}, "
                f"observed {hidden_states.size(0)} ({exc})"
            )
            continue
        return cp_group, cp_rank, cp_size, gathered

    raise ValueError(
        "Packed GDN CP could not find a CP group matching the packed tensor shape. "
        f"cu_seqlens total length is {int(cu_seqlens[-1].item())}, observed local length is {hidden_states.size(0)}. "
        f"Candidates: {'; '.join(diagnostics) if diagnostics else 'none'}"
    )


def _build_gdn_cp_context(module, cu_seqlens, device, cp_group=None, cp_size: int | None = None):
    import torch  # pyright: ignore[reportMissingImports]

    if cp_group is None or cp_size is None:
        cp_group, _, cp_size = _get_gdn_cp_group_info(module)
    if cp_size <= 1:
        return None

    try:
        from fla.ops.cp import build_cp_context  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        raise RuntimeError(
            "PLATOON_QWEN35_GDN_CP=1 requires flash-linear-attention with "
            "`fla.ops.cp.build_cp_context` available."
        ) from exc

    return build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=module.conv_kernel_dim,
    )


def _apply_packed_causal_conv1d_for_gdn(
    module,
    qkv,
    cu_seqlens,
    cp_group=None,
    cp_rank: int | None = None,
    cp_size: int | None = None,
):
    """Apply GDN's depthwise causal conv without bleeding across packed samples.

    Megatron-Core's current GDN conv is an ``nn.Conv1d`` over the full sequence.
    For packed THD input that would mix state across sample boundaries. For the
    opt-in CP path, gather the contiguous CP shards, run the real module weights
    independently per packed sequence, then return this CP rank's contiguous
    shard. This is correctness-first and can be replaced by FLA ShortConvolution
    once the runtime exposes a parameter-sharing CP convolution path.
    """

    import torch  # pyright: ignore[reportMissingImports]
    import torch.distributed.nn.functional as dist_F  # pyright: ignore[reportMissingImports]

    if cp_group is None or cp_rank is None or cp_size is None:
        cp_group, cp_rank, cp_size = _get_gdn_cp_group_info(module)

    if qkv.size(0) != 1:
        raise ValueError(f"Packed GDN expects dummy batch dimension 1, got qkv shape {tuple(qkv.shape)}")

    local_qkv = qkv.squeeze(0)
    if cp_size > 1:
        gathered = dist_F.all_gather(local_qkv.contiguous(), group=cp_group)
        full_qkv = torch.cat(gathered, dim=0)
    else:
        full_qkv = local_qkv

    pieces = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        seq = full_qkv[int(start.item()) : int(end.item())]
        if seq.numel() == 0:
            pieces.append(seq)
            continue
        seq_bds = seq.transpose(0, 1).unsqueeze(0).contiguous()
        conv = module.conv1d(seq_bds)[..., : seq.size(0)]
        pieces.append(module.act_fn(conv).squeeze(0).transpose(0, 1))

    conv_full = torch.cat(pieces, dim=0) if pieces else full_qkv[:0]
    if cp_size <= 1:
        return conv_full.unsqueeze(0)

    shard_len = local_qkv.size(0)
    return conv_full[cp_rank * shard_len : (cp_rank + 1) * shard_len].unsqueeze(0)


def _patch_megatron_core_gdn_context_parallel_config_validation() -> None:
    """Bypass MCore's GDN CP construction guard under the opt-in CP patch.

    Megatron-Core 0.17.0 rejects ``experimental_attention_variant="gated_delta_net"``
    with CP>1 during ``TransformerConfig.__post_init__``. The forward patch below
    supplies the missing packed GDN CP path, so let provider construction complete
    while preserving the real CP size on the finalized config.
    """

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        from megatron.core.transformer.transformer_config import (  # pyright: ignore[reportMissingImports]
            TransformerConfig,
        )
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron-Core TransformerConfig.") from exc

    original = TransformerConfig.__post_init__
    if getattr(original, "__platoon_qwen35_gdn_cp_config_patch__", False):
        return

    @wraps(original)
    def _post_init_allowing_gdn_context_parallel(self, *args, **kwargs):
        if (
            getattr(self, "experimental_attention_variant", None) == "gated_delta_net"
            and getattr(self, "context_parallel_size", 1) > 1
        ):
            context_parallel_size = self.context_parallel_size
            try:
                self.context_parallel_size = 1
                return original(self, *args, **kwargs)
            finally:
                self.context_parallel_size = context_parallel_size
        return original(self, *args, **kwargs)

    _post_init_allowing_gdn_context_parallel.__platoon_qwen35_gdn_cp_config_patch__ = True
    TransformerConfig.__post_init__ = _post_init_allowing_gdn_context_parallel


def _patch_megatron_core_gated_delta_net_context_parallel() -> None:
    """Opt-in packed THD + CP support for Megatron-Core GatedDeltaNet.

    This mirrors the MILES approach at the layout/recurrence boundary while
    keeping Megatron-Core's TP-sharded GDN weights. It is intentionally guarded
    by ``PLATOON_QWEN35_GDN_CP=1`` because upstream MCore still marks this area
    experimental and the duplicated convolution fallback trades memory for
    correctness.
    """

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        import torch  # pyright: ignore[reportMissingImports]
        import torch.nn.functional as F  # pyright: ignore[reportMissingImports]
        from megatron.core.ssm import gated_delta_net as gdn_module  # pyright: ignore[reportMissingImports]
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # pyright: ignore[reportMissingImports]
        from megatron.core.utils import (  # pyright: ignore[reportMissingImports]
            deprecate_inference_params,
            nvtx_range_pop,
            nvtx_range_push,
        )
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import Megatron-Core GDN dependencies.") from exc

    original = GatedDeltaNet.forward
    if getattr(original, "__platoon_qwen35_gdn_cp_patch__", False):
        return

    @wraps(original)
    def _gdn_forward_with_packed_context_parallel(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        **kwargs,
    ):
        if packed_seq_params is None:
            return original(
                self,
                hidden_states,
                attention_mask,
                key_value_states=key_value_states,
                inference_context=inference_context,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                inference_params=inference_params,
                **kwargs,
            )

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if inference_context is not None:
            raise NotImplementedError("Packed GDN CP does not support inference contexts.")
        if key_value_states is not None:
            raise NotImplementedError("Packed GDN CP does not support cross-attention key/value states.")
        if getattr(packed_seq_params, "qkv_format", None) != "thd":
            raise NotImplementedError(
                f"Packed GDN CP only supports THD packed sequences, got {packed_seq_params.qkv_format!r}."
            )
        cu_seqlens_padded = getattr(packed_seq_params, "cu_seqlens_q_padded", None)
        cu_seqlens = cu_seqlens_padded if cu_seqlens_padded is not None else packed_seq_params.cu_seqlens_q
        if gdn_module.chunk_gated_delta_rule is None:
            raise RuntimeError("Packed GDN CP requires `fla.ops.gated_delta_rule.chunk_gated_delta_rule`.")
        nvtx_range_push(suffix="in_proj")
        qkvzba, _ = self.in_proj(hidden_states)
        nvtx_range_pop(suffix="in_proj")

        cp_group, cp_rank, cp_size, qkvzba = _select_gdn_cp_group_for_tensor(self, packed_seq_params, cu_seqlens, qkvzba)
        if cp_size > 1:
            qkvzba = _zigzag_to_packed_shard(qkvzba, cu_seqlens, cp_group, cp_rank, cp_size)

        qkvzba = qkvzba.transpose(0, 1)
        batch, seq_len, _ = qkvzba.shape
        if batch != 1:
            raise ValueError(f"Packed GDN CP expects dummy batch dimension 1, got qkvzba shape {tuple(qkvzba.shape)}")

        qkv, gate, beta, alpha = torch.split(
            qkvzba,
            [
                (self.qk_dim * 2 + self.v_dim) // self.tp_size,
                self.v_dim // self.tp_size,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            dim=-1,
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)

        nvtx_range_push(suffix="conv1d")
        qkv = _apply_packed_causal_conv1d_for_gdn(self, qkv, cu_seqlens, cp_group, cp_rank, cp_size)
        nvtx_range_pop(suffix="conv1d")

        query, key, value = torch.split(
            qkv,
            [self.qk_dim // self.tp_size, self.qk_dim // self.tp_size, self.v_dim // self.tp_size],
            dim=-1,
        )
        query = query.reshape(batch, seq_len, -1, self.key_head_dim)
        key = key.reshape(batch, seq_len, -1, self.key_head_dim)
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)

        if self.use_qk_l2norm:
            query = gdn_module.l2norm(query.contiguous())
            key = gdn_module.l2norm(key.contiguous())
        if self.num_value_heads // self.num_key_heads > 1:
            query = query.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)
            key = key.repeat_interleave(self.num_value_heads // self.num_key_heads, dim=2)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        gate = gate.contiguous()
        beta = beta.contiguous()
        alpha = alpha.contiguous()

        nvtx_range_push(suffix="g_and_beta")
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)
        beta = beta.sigmoid()
        nvtx_range_pop(suffix="g_and_beta")

        # Even when AReaL sets config.deterministic_mode for MoE stability, the
        # deterministic torch GDN reference path has no CP state passing. Keep
        # the global deterministic settings and use FLA only for packed GDN CP.
        cp_context = _build_gdn_cp_context(self, cu_seqlens, hidden_states.device, cp_group, cp_size)
        nvtx_range_push(suffix="gated_delta_rule")
        if cp_context is not None:
            core_attn_out, _ = gdn_module.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cp_context.cu_seqlens,
                cp_context=cp_context,
            )
        else:
            core_attn_out, _ = gdn_module.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=False,
                cu_seqlens=cu_seqlens,
            )
        nvtx_range_pop(suffix="gated_delta_rule")

        nvtx_range_push(suffix="gated_norm")
        norm_out = self._apply_gated_norm(core_attn_out, gate)
        nvtx_range_pop(suffix="gated_norm")

        norm_out = norm_out.reshape(batch, seq_len, -1)
        norm_out = norm_out.transpose(0, 1).contiguous()
        if cp_size > 1:
            norm_out = _packed_shard_to_zigzag(norm_out, cu_seqlens, cp_group, cp_rank, cp_size)

        nvtx_range_push(suffix="out_proj")
        out, out_bias = self.out_proj(norm_out)
        nvtx_range_pop(suffix="out_proj")
        return out, out_bias

    _gdn_forward_with_packed_context_parallel.__platoon_qwen35_gdn_cp_patch__ = True
    GatedDeltaNet.forward = _gdn_forward_with_packed_context_parallel


def _patch_areal_qwen35_gdn_cp_guards() -> None:
    """Let text-only Qwen3.5 use AReaL's packed CP path under opt-in."""

    if not _qwen35_gdn_cp_enabled():
        return

    try:
        import areal.engine.core.model as core_model  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        raise RuntimeError("PLATOON_QWEN35_GDN_CP=1 could not import AReaL model guards.") from exc

    original_requires_padded_seq = core_model.requires_padded_seq
    if not getattr(original_requires_padded_seq, "__platoon_qwen35_gdn_cp_guard_patch__", False):

        @wraps(original_requires_padded_seq)
        def _requires_padded_seq_with_qwen35_gdn_cp(model_type: str) -> bool:
            if core_model.is_qwen3_5_model(model_type):
                return False
            return original_requires_padded_seq(model_type)

        _requires_padded_seq_with_qwen35_gdn_cp.__platoon_qwen35_gdn_cp_guard_patch__ = True
        core_model.requires_padded_seq = _requires_padded_seq_with_qwen35_gdn_cp

    original_is_valid_vision_model = core_model.is_valid_vision_model
    if not getattr(original_is_valid_vision_model, "__platoon_qwen35_gdn_cp_vision_patch__", False):

        @wraps(original_is_valid_vision_model)
        def _is_valid_vision_model_with_text_only_qwen35_gdn_cp(model_type: str) -> bool:
            if core_model.is_qwen3_5_model(model_type):
                return False
            return original_is_valid_vision_model(model_type)

        _is_valid_vision_model_with_text_only_qwen35_gdn_cp.__platoon_qwen35_gdn_cp_vision_patch__ = True
        core_model.is_valid_vision_model = _is_valid_vision_model_with_text_only_qwen35_gdn_cp

    megatron_engine = sys.modules.get("areal.engine.megatron_engine")
    if megatron_engine is not None:
        megatron_engine.requires_padded_seq = core_model.requires_padded_seq
        megatron_engine.is_valid_vision_model = core_model.is_valid_vision_model


def _patch_batch_task_dispatcher_idle_submit() -> None:
    """Avoid silent dispatcher stalls when no rollout tasks are in flight.

    AReaL's dispatcher reserves ``batch_size`` slots in the async runner queue
    before submitting more work. If that reservation leaves zero apparent
    capacity, ``active_submit_and_wait`` can wait for results even though no
    tasks were ever submitted. Platoon's long rollouts make that look like a
    generation hang, so patch the capacity calculation to use actual free queue
    slots and emit periodic state when dispatch is blocked.
    """

    from areal.infra.workflow_executor import BatchTaskDispatcher  # pyright: ignore[reportMissingImports]

    original = BatchTaskDispatcher.active_submit_and_wait
    if getattr(original, "__platoon_idle_submit_patch__", False):
        return

    @wraps(original)
    def _active_submit_and_wait_with_idle_guard(
        self,
        input_generator,
        batch_size: int,
        dynamic_bs: bool = False,
    ) -> list[Any]:
        accepted_cnt = 0
        total_attempts = 0
        results = []
        last_blocked_log = 0.0

        while True:
            self._check_thread_exception()

            with self._input_cv:
                pending_inputs = len(self._pending_inputs)
            runner_input_size = self.runner.get_input_queue_size()
            cap_staleness = self.staleness_manager.get_pending_limit() - pending_inputs

            if self.runner.max_queue_size < batch_size:
                raise ValueError(
                    "Inference engine config's queue size is too small: "
                    f"{self.runner.max_queue_size} < batch size {batch_size}."
                )

            free_runner_slots = self.runner.max_queue_size - runner_input_size
            capacity = min(cap_staleness, free_runner_slots)

            if capacity > 0 and not self.runner.paused.is_set():
                for _ in range(min(batch_size, capacity)):
                    try:
                        self.submit_task_input(next(input_generator))
                    except StopIteration:
                        raise RuntimeError(
                            "Input generator exhausted before batch completion. "
                            "Use cycle_dataloader() or provide an infinite generator."
                        ) from None
            else:
                now = time.monotonic()
                if now - last_blocked_log >= 30.0:
                    last_blocked_log = now
                    stats = self.staleness_manager.get_stats()
                    self.logger.warning(
                        "Rollout dispatch is waiting without submit capacity: "
                        "batch_size=%s accepted=%s total_attempts=%s "
                        "pending_inputs=%s runner_input_queue=%s "
                        "max_queue_size=%s cap_staleness=%s free_runner_slots=%s "
                        "paused=%s stats=%s",
                        batch_size,
                        accepted_cnt,
                        total_attempts,
                        pending_inputs,
                        runner_input_size,
                        self.runner.max_queue_size,
                        cap_staleness,
                        free_runner_slots,
                        self.runner.paused.is_set(),
                        stats,
                    )

            try:
                arrived = self.wait_results(count=batch_size - accepted_cnt, timeout=1)
            except TimeoutError:
                arrived = []

            for res in arrived:
                is_accepted = res is not None
                if not is_accepted:
                    if dynamic_bs:
                        total_attempts += 1
                        if total_attempts >= batch_size:
                            break
                    continue

                accepted_cnt += 1
                total_attempts += 1
                results.append(res)

                if dynamic_bs:
                    if total_attempts >= batch_size:
                        break
                elif accepted_cnt >= batch_size:
                    break
            else:
                continue
            break

        return results

    _active_submit_and_wait_with_idle_guard.__platoon_idle_submit_patch__ = True
    BatchTaskDispatcher.active_submit_and_wait = _active_submit_and_wait_with_idle_guard


def _patch_local_scheduler_fork_ready_timeout() -> None:
    """Give forked proxy workers enough time to import and start serving.

    AReaL hardcodes fork readiness checks to 60 seconds. Platoon's proxy
    workers import the workflow stack and can take longer on cold starts, even
    though the workers are healthy once the server finishes booting.
    """

    from areal.infra.scheduler.local import LocalScheduler  # pyright: ignore[reportMissingImports]

    original = LocalScheduler._wait_for_fork_ready
    if getattr(original, "__platoon_fork_ready_timeout_patch__", False):
        return

    @wraps(original)
    async def _wait_for_fork_ready_with_platoon_timeout(
        session,
        host: str,
        port: int,
        timeout: float = 60,
    ) -> bool:
        if timeout == 60:
            timeout = float(os.environ.get("PLATOON_AREAL_FORK_READY_TIMEOUT", "900"))
        return await original(session, host, port, timeout=timeout)

    _wait_for_fork_ready_with_platoon_timeout.__platoon_fork_ready_timeout_patch__ = True
    LocalScheduler._wait_for_fork_ready = staticmethod(_wait_for_fork_ready_with_platoon_timeout)


def _patch_remote_inf_engine_asyncio_teardown_race() -> None:
    """Run inference-server fan-out coroutines without asyncio's racy teardown.

    ``areal.infra.remote_inf_engine`` issues control-plane requests (pause/
    resume/offload/onload and weight-update fan-outs) through ``uvloop.run``,
    whose ``asyncio.Runner`` teardown calls ``asyncio.all_tasks()``. That
    snapshots a process-global WeakSet shared by every event loop in the
    process; with the workflow executor churning thousands of rollout tasks in
    another thread, the snapshot fails with ``RuntimeError: Set changed size
    during iteration`` even after CPython's 1000 internal retries (bpo-36607;
    only structurally fixed by per-thread task lists in Python 3.14). At
    recursive-workflow concurrency the churn is continuous, so retrying the
    fan-out does not help either - observed 5 consecutive failures over 2.5
    minutes. Instead, replace the module's ``uvloop.run`` with a runner that
    drives a private event loop directly and skips the cancel-all sweep. The
    fan-out coroutines await everything they spawn before returning, so the
    sweep (the only ``all_tasks()`` caller on this path) is dead weight.
    """

    import areal.infra.remote_inf_engine as remote_inf_engine  # pyright: ignore[reportMissingImports]
    import uvloop  # pyright: ignore[reportMissingImports]

    if getattr(remote_inf_engine.uvloop, "__platoon_asyncio_teardown_race_patch__", False):
        return

    class _RaceFreeUvloop:
        """Module-local ``uvloop`` stand-in whose ``run`` skips Runner teardown."""

        __platoon_asyncio_teardown_race_patch__ = True

        @staticmethod
        def run(coro):
            loop = uvloop.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()

        def __getattr__(self, name):
            return getattr(uvloop, name)

    remote_inf_engine.uvloop = _RaceFreeUvloop()


def _patch_remote_inf_engine_proxy_resolution() -> None:
    """Let custom RolloutWorkflow instances receive worker-local proxy URLs.

    Upstream AReaL already threads a per-worker ``proxy_addr`` through
    ``RemoteInfEngine.submit()``, but only injects it when wrapping agent-like
    workflows in ``OpenAIProxyWorkflow``. Platoon's custom workflows are already
    ``RolloutWorkflow`` instances, so inline mode needs a small patch to bind the
    same worker-local proxy URL onto those workflow objects before execution.
    """

    from areal.api import RolloutWorkflow  # pyright: ignore[reportMissingImports]
    from areal.infra.remote_inf_engine import RemoteInfEngine  # pyright: ignore[reportMissingImports]

    original = RemoteInfEngine._resolve_workflow
    if getattr(original, "__platoon_proxy_patch__", False):
        return

    def _inject_proxy_addr(workflow: RolloutWorkflow, proxy_addr: str) -> None:
        setter = getattr(workflow, "set_proxy_base_url", None)
        if callable(setter):
            setter(proxy_addr)
            return
        if hasattr(workflow, "proxy_base_url"):
            setattr(workflow, "proxy_base_url", proxy_addr)

    @wraps(original)
    def _resolve_workflow_with_proxy_addr(
        self,
        workflow: Any,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved = original(
            self,
            workflow,
            workflow_kwargs,
            group_size=group_size,
            proxy_addr=proxy_addr,
        )
        if proxy_addr is not None and isinstance(resolved, RolloutWorkflow):
            _inject_proxy_addr(resolved, proxy_addr)
        return resolved

    _resolve_workflow_with_proxy_addr.__platoon_proxy_patch__ = True
    RemoteInfEngine._resolve_workflow = _resolve_workflow_with_proxy_addr


def _flatten_message_list_content(messages: list[dict[str, Any]]) -> None:
    """Convert OpenAI list-shaped text content blocks to plain strings.

    OpenHands (and other clients) may send ``content`` as
    ``[{"type": "text", "text": "..."}]``. Hugging Face ``apply_chat_template``
    and AReaL's interaction cache expect string ``content`` for text-only turns.
    """

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if any(
            isinstance(item, dict)
            and item.get("type") in ("image_url", "image", "input_image")
            for item in content
        ):
            continue
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        message["content"] = "".join(text_parts)


def _decode_tool_call_arguments(messages: list[dict[str, Any]]) -> None:
    """Decode assistant tool-call ``arguments`` from JSON strings into dicts.

    OpenAI-format messages (e.g. from OpenHands native tool calling) carry
    ``tool_calls[].function.arguments`` as a JSON string per spec. Chat
    templates such as Qwen3 render them with the Jinja ``items`` filter, which
    requires a mapping and otherwise raises ``TypeError: Can only get item
    pairs from a mapping`` once the conversation history contains a tool call.
    Decode the string into a dict in place so the template sees a mapping;
    leave non-JSON or non-object payloads untouched.
    """

    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                decoded = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, dict):
                function["arguments"] = decoded


def _patch_areal_openai_message_content_flatten() -> None:
    """Normalize OpenHands-style messages before HF chat templates and proxy cache.

    Two adjustments are applied to the proxy's incoming messages:

    - Flatten list-shaped text ``content`` blocks to plain strings.
    - Decode tool-call ``arguments`` from JSON strings into dicts so chat
      templates that iterate them as mappings (e.g. Qwen3) do not crash.
    """

    import areal.experimental.openai.client as client_module  # pyright: ignore[reportMissingImports]

    original_ensure = client_module._ensure_message_dict_list
    if getattr(original_ensure, "__platoon_message_content_patch__", False):
        return

    @wraps(original_ensure)
    def _ensure_message_dict_list_with_flatten(
        name: str,
        value: list[Any],
    ) -> list[dict[str, Any]]:
        normalized = original_ensure(name, value)
        _flatten_message_list_content(normalized)
        _decode_tool_call_arguments(normalized)
        return normalized

    _ensure_message_dict_list_with_flatten.__platoon_message_content_patch__ = True
    client_module._ensure_message_dict_list = _ensure_message_dict_list_with_flatten


# ---------------------------------------------------------------------------
# Process stall instrumentation
# ---------------------------------------------------------------------------
# Workers have wedged in ways that left no post-mortem evidence (e.g. a
# rollout worker stopped answering `pause` RPCs entirely and the run died
# without a single stack trace). The watchdog below makes the next wedge
# self-diagnosing from the worker's own log.

_STALL_WATCHDOG_STARTED = False
_ENGINE_CALL_LOCK = threading.Lock()
_ENGINE_CALL_STATE: dict[str, Any] = {}


def _watchdog_log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[platoon-stall-watchdog pid={os.getpid()}] {timestamp} {message}",
        file=sys.stderr,
        flush=True,
    )


def _dump_all_thread_stacks(reason: str) -> None:
    name_by_ident = {t.ident: t.name for t in threading.enumerate()}
    lines = [f"all thread stacks ({reason}):"]
    for ident, frame in sys._current_frames().items():
        name = name_by_ident.get(ident, "unknown")
        stack = "".join(traceback.format_stack(frame))
        lines.append(f"--- thread {name!r} (ident={ident}) ---\n{stack}")
    _watchdog_log("\n".join(lines))


def _patch_engine_rpc_call_tracking() -> None:
    """Record which engine RPC method is running on the worker engine thread.

    All engine RPCs (including trivial ones like ``pause``) are serialized
    through a single engine thread, so one stuck method makes the whole worker
    unresponsive to the trainer. Tracking the current method lets the stall
    watchdog name the offender and dump its stack instead of the trainer only
    seeing opaque RPC timeouts.
    """

    try:
        import areal.infra.rpc.guard.engine_blueprint as engine_blueprint  # pyright: ignore[reportMissingImports]
    except Exception:
        return

    original = engine_blueprint._submit_to_engine_thread
    if getattr(original, "__platoon_engine_call_tracking__", False):
        return

    @wraps(original)
    def _submit_with_tracking(func_name: str, func, *args: Any, **kwargs: Any) -> Any:
        @wraps(func)
        def _tracked(*func_args: Any, **func_kwargs: Any) -> Any:
            with _ENGINE_CALL_LOCK:
                _ENGINE_CALL_STATE.update(
                    name=func_name,
                    started=time.monotonic(),
                    active=True,
                )
            try:
                return func(*func_args, **func_kwargs)
            finally:
                with _ENGINE_CALL_LOCK:
                    _ENGINE_CALL_STATE["active"] = False

        return original(func_name, _tracked, *args, **kwargs)

    _submit_with_tracking.__platoon_engine_call_tracking__ = True
    engine_blueprint._submit_to_engine_thread = _submit_with_tracking


def _install_process_stall_watchdog() -> None:
    """Start a watchdog that makes process wedges self-diagnosing.

    Installs, in every Platoon AReaL process (trainer, train workers, rollout
    workers, proxy workers):

    - ``SIGUSR1`` -> faulthandler dump of all thread stacks to stderr, for
      on-demand inspection of a live process (``kill -USR1 <pid>``).
    - A dead-man timer re-armed every few seconds by a heartbeat thread. If
      Python threads cannot run for ``PLATOON_STALL_DUMP_SECS`` (default 180s;
      e.g. a stop-the-world GC pause or a GIL-holding native call),
      faulthandler's C watchdog thread dumps all thread stacks to stderr
      without needing the GIL.
    - A post-hoc warning when the heartbeat thread itself was frozen, which
      timestamps GC/GIL stalls even when they end before the dump fires.
    - A warning plus all-thread stack dump when one engine RPC method has been
      running for over ``PLATOON_ENGINE_STALL_SECS`` (default 600s).
    - A warning when open file descriptors exceed 80% of the soft limit
      (leaked sockets exhaust FDs long before the process dies).

    Disable with ``PLATOON_STALL_WATCHDOG=0``.
    """

    global _STALL_WATCHDOG_STARTED

    if os.environ.get("PLATOON_STALL_WATCHDOG", "1") != "1":
        return
    if _STALL_WATCHDOG_STARTED:
        return
    _STALL_WATCHDOG_STARTED = True

    import faulthandler
    import signal

    try:
        # chain=False: with no prior Python handler installed, chaining would
        # fall through to the default action and terminate the process.
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except Exception:
        pass

    _patch_engine_rpc_call_tracking()

    heartbeat_interval = 5.0
    freeze_warn_slack = 30.0
    freeze_dump_secs = float(os.environ.get("PLATOON_STALL_DUMP_SECS", "180"))
    engine_stall_secs = float(os.environ.get("PLATOON_ENGINE_STALL_SECS", "600"))
    fd_check_period = 60.0
    fd_warn_fraction = 0.8
    fd_warn_cooldown = 300.0

    def _maybe_warn_fd_usage(last_warn: float) -> float:
        try:
            import resource

            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            open_fds = len(os.listdir("/proc/self/fd"))
        except Exception:
            return last_warn
        now = time.monotonic()
        if soft_limit > 0 and open_fds > fd_warn_fraction * soft_limit and now - last_warn > fd_warn_cooldown:
            _watchdog_log(
                f"high file descriptor usage: {open_fds}/{soft_limit} open; "
                "leaked sockets can wedge this process before any crash"
            )
            return now
        return last_warn

    def _watchdog_loop() -> None:
        last_fd_check = 0.0
        last_fd_warn = float("-inf")
        last_engine_report = float("-inf")
        while True:
            try:
                faulthandler.dump_traceback_later(freeze_dump_secs, exit=False, file=sys.stderr)
            except Exception:
                pass

            before_sleep = time.monotonic()
            time.sleep(heartbeat_interval)
            now = time.monotonic()

            gap = now - before_sleep
            if gap > heartbeat_interval + freeze_warn_slack:
                _watchdog_log(
                    f"Python threads could not run for {gap:.0f}s "
                    "(stop-the-world GC pause or GIL-holding native call); "
                    f"stalls over {freeze_dump_secs:.0f}s dump all thread stacks via faulthandler"
                )

            with _ENGINE_CALL_LOCK:
                engine_call = dict(_ENGINE_CALL_STATE)
            if engine_call.get("active"):
                elapsed = now - engine_call["started"]
                if elapsed > engine_stall_secs and now - last_engine_report > engine_stall_secs:
                    last_engine_report = now
                    _dump_all_thread_stacks(
                        f"engine RPC method {engine_call['name']!r} has been running for {elapsed:.0f}s; "
                        "all other engine RPCs (e.g. pause) are queued behind it"
                    )

            if now - last_fd_check > fd_check_period:
                last_fd_check = now
                last_fd_warn = _maybe_warn_fd_usage(last_fd_warn)

    threading.Thread(target=_watchdog_loop, daemon=True, name="platoon-stall-watchdog").start()
    _watchdog_log(
        f"started (freeze dump after {freeze_dump_secs:.0f}s, engine RPC stall report after "
        f"{engine_stall_secs:.0f}s); run `kill -USR1 {os.getpid()}` for an on-demand stack dump"
    )


def apply_all_patches() -> None:
    """Apply Platoon compatibility patches for the current AReaL release.

    Two historical patches were dropped when upgrading to AReaL HEAD
    (``a0f3dca``) because upstream now handles those cases natively:

    - ``apply_chat_template`` return-type coercion: ``areal.utils.hf_utils``
      now provides an ``apply_chat_template`` wrapper that normalizes the
      Transformers 5 dict return to ``list[int]``. Re-patching the tokenizer
      method globally would make that wrapper's ``result["input_ids"]`` fail.
    - FSDP wrap-class set/tuple compatibility: ``areal.engine.fsdp_utils``'s
      ``apply_fsdp2`` now normalizes ``_no_split_modules`` and
      ``transformer_layer_cls_to_wrap`` internally.
    """

    _patch_hf_tokenizer_download_race()
    _patch_model_response_custom_stop_sequences()
    _patch_triton_cache_for_qwen35_gdn_cp()
    _patch_megatron_bridge_attention_backend()
    _patch_megatron_bridge_qwen35_tp_validation()
    _patch_megatron_bridge_qwen35_drop_mtp_for_rl()
    _patch_megatron_bridge_qwen35_cp_per_token_loss()
    _patch_megatron_checkpoint_optimizer_metadata()
    _patch_areal_qwen35_gdn_cp_guards()
    _patch_megatron_bridge_qwen3vl_already_cp_local_packed_input()
    _patch_megatron_core_qwen35_mtp_local_thd_rope()
    _patch_megatron_core_mtp_checkpoint_non_tensor_kwargs()
    _patch_megatron_core_mtp_aux_loss_scaling_for_areal_cp()
    _patch_megatron_core_gdn_context_parallel_config_validation()
    _patch_megatron_core_gated_delta_net_context_parallel()
    _patch_batch_task_dispatcher_idle_submit()
    _patch_local_scheduler_fork_ready_timeout()
    _patch_remote_inf_engine_asyncio_teardown_race()
    _patch_remote_inf_engine_proxy_resolution()
    _patch_areal_openai_message_content_flatten()
    _install_process_stall_watchdog()
