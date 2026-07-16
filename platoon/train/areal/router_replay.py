"""Feature-gated routed-expert replay for Platoon's Megatron actor.

The rollout/training interface is intentionally small:

``routed_experts``
    Integer tensor ``[batch, sequence, global_layers, topk]``. Layer indices
    are global transformer-layer indices, not PP-local or MoE-only ordinals.

``routed_experts_valid``
    Boolean tensor ``[batch, sequence]``. Expert zero is valid, so missing
    routing decisions must be represented by this mask rather than a magic
    expert id. For SGLang, every real token except the terminal token must be
    valid; terminal and padding rows are replayed with the live router.

Megatron-Core 0.17 already implements ``RouterReplay``. This module supplies
the AReaL plumbing around that native API: micro-batch reordering, packed and
padded layout alignment, CP/TP partitioning, PP/VP layer selection, and the
forward-to-checkpoint-recompute action lifecycle.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import torch

logger = logging.getLogger("PlatoonRouterReplay")

ROUTED_EXPERTS_FIELD = "routed_experts"
ROUTED_EXPERTS_VALID_FIELD = "routed_experts_valid"
_MICROBATCH_FIELD = "_platoon_router_replay_microbatch"
_PENDING_ATTR = "_platoon_router_replay_pending"
_STATE_ATTR = "_platoon_router_replay_state"
_BINDINGS_ATTR = "_platoon_router_replay_bindings"


class RouterReplayError(RuntimeError):
    """Raised when replay cannot cover the configured training batch exactly."""


@dataclass(frozen=True)
class RouterReplayBatch:
    routes: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class RouterBinding:
    layer_index: int
    router: Any
    replay: Any


@dataclass
class RouterReplayMicrobatch:
    routes: torch.Tensor
    valid: torch.Tensor
    expected_layers: frozenset[int]
    seen_layers: set[int] = field(default_factory=set)


@dataclass
class RouterReplayEngineState:
    bindings: tuple[RouterBinding, ...]
    expected_layers: frozenset[int]
    num_layers: int
    topk: int
    num_experts: int
    full_recompute: bool
    sequence_parallel: bool


def _recompute_mode(config: Any) -> tuple[str | None, str | None]:
    """Return the effective MCore activation-recompute mode for one config."""

    return (
        getattr(config, "recompute_granularity", None),
        getattr(config, "recompute_method", None),
    )


def _uses_sinkhorn(routing_type: Any) -> bool:
    return routing_type == "sinkhorn" or (isinstance(routing_type, (list, tuple)) and "sinkhorn" in routing_type)


def _record_metrics(**values: float) -> None:
    try:
        from areal.utils import stats_tracker

        stats_tracker.scalar(**{f"r3/{key}": value for key, value in values.items()})
    except Exception:
        logger.debug("Could not record R3 metrics", exc_info=True)


def _as_local_tensor(value: Any, *, field_name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        value = to_local()
        if torch.is_tensor(value):
            return value
    try:
        return torch.as_tensor(value)
    except Exception as exc:
        raise RouterReplayError(f"{field_name} cannot be converted to a tensor") from exc


def _validate_actor_payload(
    data: Mapping[str, Any],
    routes: torch.Tensor,
    valid: torch.Tensor,
    *,
    expected_layers: int,
    expected_topk: int,
    expected_experts: int | None,
) -> tuple[int, int]:
    attention_mask = data.get("attention_mask")
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        raise RouterReplayError("R3 requires a 2D attention_mask tensor")
    if routes.ndim != 4:
        raise RouterReplayError(f"{ROUTED_EXPERTS_FIELD} must have shape [B,S,L,K], got {tuple(routes.shape)}")
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise RouterReplayError(
            f"{ROUTED_EXPERTS_VALID_FIELD} must be a bool [B,S] tensor, got "
            f"shape={tuple(valid.shape)} dtype={valid.dtype}"
        )
    if tuple(routes.shape[:2]) != tuple(attention_mask.shape) or tuple(valid.shape) != tuple(attention_mask.shape):
        raise RouterReplayError(
            "R3 batch/sequence dimensions must match attention_mask: "
            f"routes={tuple(routes.shape)}, valid={tuple(valid.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if routes.shape[2:] != (expected_layers, expected_topk):
        raise RouterReplayError(
            f"R3 layer/top-k mismatch: expected ({expected_layers}, {expected_topk}), got {tuple(routes.shape[2:])}"
        )
    integral_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if routes.dtype not in integral_dtypes:
        raise RouterReplayError(f"{ROUTED_EXPERTS_FIELD} must use an integer dtype, got {routes.dtype}")

    attention = attention_mask.bool()
    lengths = attention.sum(dim=1, dtype=torch.long)
    if torch.any(lengths <= 0):
        raise RouterReplayError("R3 does not support empty sequences")
    positions = torch.arange(attention.shape[1], device=attention.device).unsqueeze(0)
    prefix_attention = positions < lengths.unsqueeze(1)
    if not torch.equal(attention, prefix_attention):
        raise RouterReplayError("R3 requires the same right-padded, left-aligned token layout used by AReaL packing")

    required = prefix_attention.clone()
    rows = torch.arange(required.shape[0], device=required.device)
    required[rows, lengths - 1] = False
    valid_on_attention_device = valid.to(device=required.device)
    missing = required & ~valid_on_attention_device
    unexpected = valid_on_attention_device & ~required
    required_count = int(required.sum().item())
    valid_required_count = int((required & valid_on_attention_device).sum().item())
    route_coverage = valid_required_count / max(required_count, 1)
    _record_metrics(
        route_coverage=float(route_coverage),
        required_routes=float(required_count),
        missing_required_routes=float(missing.sum().item()),
        unexpected_valid_routes=float(unexpected.sum().item()),
    )
    if missing.any() or unexpected.any():
        raise RouterReplayError(
            "R3 route coverage is incomplete: every real non-terminal token must be valid, "
            "and terminal/padding tokens must be invalid; "
            f"missing={int(missing.sum().item())}, unexpected={int(unexpected.sum().item())}, "
            f"coverage={route_coverage:.6f}"
        )

    valid_for_routes = valid.to(device=routes.device)
    selected = routes[valid_for_routes]
    if selected.numel():
        min_id = int(selected.min().item())
        max_id = int(selected.max().item())
        if min_id < 0:
            raise RouterReplayError(f"R3 expert ids must be non-negative, observed {min_id}")
        if expected_experts is not None and max_id >= expected_experts:
            raise RouterReplayError(f"R3 expert id {max_id} is outside configured range [0, {expected_experts})")
        if expected_topk > 1:
            sorted_ids = torch.sort(selected, dim=-1).values
            duplicate_rows = torch.any(sorted_ids[..., 1:] == sorted_ids[..., :-1], dim=-1)
            if torch.any(duplicate_rows):
                raise RouterReplayError(
                    "R3 top-k expert IDs must be unique within every token/layer row; "
                    f"found {int(duplicate_rows.sum().item())} rows with duplicates"
                )
    return required_count, valid_required_count


def pop_and_split_actor_router_replay(
    data: dict[str, Any],
    mb_list: Any,
    config: Any,
) -> list[RouterReplayBatch | None]:
    """Consume the public batch fields and mirror the actor minibatch ordering."""

    enabled = bool(getattr(config, "enable_router_replay", False))
    routes_value = data.pop(ROUTED_EXPERTS_FIELD, None)
    valid_value = data.pop(ROUTED_EXPERTS_VALID_FIELD, None)
    if not enabled:
        if routes_value is not None or valid_value is not None:
            raise RouterReplayError("Received routed-expert data while actor.enable_router_replay is false")
        return [None] * len(mb_list.mbs)
    if routes_value is None or valid_value is None:
        raise RouterReplayError(
            f"actor.enable_router_replay requires both {ROUTED_EXPERTS_FIELD!r} and {ROUTED_EXPERTS_VALID_FIELD!r}"
        )

    expected_layers = getattr(config, "router_replay_num_layers", None)
    expected_topk = getattr(config, "router_replay_topk", None)
    expected_experts = getattr(config, "router_replay_num_experts", None)
    if not isinstance(expected_layers, int) or expected_layers <= 0:
        raise RouterReplayError("actor.router_replay_num_layers must be a positive integer")
    if not isinstance(expected_topk, int) or expected_topk <= 0:
        raise RouterReplayError("actor.router_replay_topk must be a positive integer")
    if expected_experts is not None and (not isinstance(expected_experts, int) or expected_experts <= 0):
        raise RouterReplayError("actor.router_replay_num_experts must be positive or null")

    routes = _as_local_tensor(routes_value, field_name=ROUTED_EXPERTS_FIELD)
    valid = _as_local_tensor(valid_value, field_name=ROUTED_EXPERTS_VALID_FIELD)
    _validate_actor_payload(
        data,
        routes,
        valid,
        expected_layers=expected_layers,
        expected_topk=expected_topk,
        expected_experts=expected_experts,
    )

    batch_size = routes.shape[0]
    forward_indices = getattr(mb_list, "forward_indices", None)
    if forward_indices is not None:
        order = torch.as_tensor(forward_indices, dtype=torch.long, device=routes.device)
        routes = routes.index_select(0, order)
        valid = valid.index_select(0, order.to(valid.device))

    split: list[RouterReplayBatch | None] = []
    offset = 0
    for mb in mb_list.mbs:
        attention = mb.get("attention_mask") if isinstance(mb, dict) else None
        if not torch.is_tensor(attention) or attention.ndim != 2:
            raise RouterReplayError("Cannot infer actor minibatch sample count for R3")
        count = attention.shape[0]
        split.append(
            RouterReplayBatch(
                routes=routes[offset : offset + count].contiguous(),
                valid=valid[offset : offset + count].contiguous(),
            )
        )
        offset += count
    if offset != batch_size:
        raise RouterReplayError(f"Actor minibatch split covered {offset} samples, expected {batch_size}")
    return split


def stage_engine_router_replay_batch(engine: Any, batch: RouterReplayBatch | None) -> None:
    if batch is None:
        return
    if not bool(getattr(engine.config, "enable_router_replay", False)):
        raise RouterReplayError("Cannot stage R3 data on a disabled engine")
    if getattr(engine, _PENDING_ATTR, None) is not None:
        raise RouterReplayError("Previous R3 batch was not consumed")
    setattr(engine, _PENDING_ATTR, batch)


def discard_staged_engine_router_replay_batch(engine: Any) -> None:
    setattr(engine, _PENDING_ATTR, None)


def assert_engine_router_replay_batch_consumed(engine: Any) -> None:
    if getattr(engine, _PENDING_ATTR, None) is not None:
        raise RouterReplayError("Megatron train_batch returned without consuming staged R3 data")


def split_packed_for_context_parallel(
    tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor:
    """Zigzag-split packed dim 0 while preserving every trailing dimension."""

    if cp_size <= 1:
        return tensor
    if not 0 <= cp_rank < cp_size:
        raise RouterReplayError(f"Invalid CP rank {cp_rank} for size {cp_size}")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if any(int(length.item()) % (2 * cp_size) for length in lengths):
        raise RouterReplayError(f"R3 packed sequence lengths must be divisible by 2*CP ({2 * cp_size}): {lengths}")
    output_len = tensor.shape[0] // cp_size
    output = torch.empty((output_len, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    for i, length_tensor in enumerate(lengths):
        length = int(length_tensor.item())
        local_length = length // cp_size
        half = local_length // 2
        source_start = int(cu_seqlens[i].item())
        destination_start = source_start // cp_size
        first_start = source_start + cp_rank * half
        second_start = source_start + length - (cp_rank + 1) * half
        output[destination_start : destination_start + half] = tensor[first_start : first_start + half]
        output[destination_start + half : destination_start + local_length] = tensor[second_start : second_start + half]
    return output


def _pack_router_replay(
    batch: RouterReplayBatch,
    *,
    old_cu_seqlens: torch.Tensor,
    padded_cu_seqlens: torch.Tensor,
    use_padded_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    routes = batch.routes.to(device=padded_cu_seqlens.device)
    valid = batch.valid.to(device=padded_cu_seqlens.device)
    batch_size, source_seqlen, num_layers, topk = routes.shape
    if old_cu_seqlens.shape[0] != batch_size + 1:
        raise RouterReplayError(
            "R3 sample count does not match Megatron microbatch: "
            f"routes B={batch_size}, old_cu entries={old_cu_seqlens.shape[0]}"
        )
    old_lengths = old_cu_seqlens[1:] - old_cu_seqlens[:-1]
    if int(old_lengths.max().item()) > source_seqlen:
        raise RouterReplayError(
            f"R3 source sequence length {source_seqlen} is shorter than model input {int(old_lengths.max().item())}"
        )

    if use_padded_seq:
        if padded_cu_seqlens.shape[0] < old_cu_seqlens.shape[0]:
            raise RouterReplayError("Padded cu_seqlens lost real R3 sequences")
        padded_lengths = padded_cu_seqlens[1:] - padded_cu_seqlens[:-1]
        max_seqlen = int(padded_lengths.max().item())
        padded_batch = padded_lengths.shape[0]
        packed_routes = torch.zeros(
            padded_batch,
            max_seqlen,
            num_layers,
            topk,
            dtype=routes.dtype,
            device=routes.device,
        )
        packed_valid = torch.zeros(padded_batch, max_seqlen, dtype=torch.bool, device=routes.device)
        for i, length_tensor in enumerate(old_lengths):
            length = int(length_tensor.item())
            packed_routes[i, :length] = routes[i, :length]
            packed_valid[i, :length] = valid[i, :length]
        return packed_routes.flatten(0, 1), packed_valid.flatten()

    total_padded = int(padded_cu_seqlens[-1].item())
    packed_routes = torch.zeros(total_padded, num_layers, topk, dtype=routes.dtype, device=routes.device)
    packed_valid = torch.zeros(total_padded, dtype=torch.bool, device=routes.device)
    for i, length_tensor in enumerate(old_lengths):
        length = int(length_tensor.item())
        start = int(padded_cu_seqlens[i].item())
        packed_routes[start : start + length] = routes[i, :length]
        packed_valid[start : start + length] = valid[i, :length]
    return packed_routes, packed_valid


def build_local_router_replay_microbatch(
    batch: RouterReplayBatch,
    *,
    old_cu_seqlens: torch.Tensor,
    padded_cu_seqlens: torch.Tensor,
    expected_layers: frozenset[int],
    use_padded_seq: bool,
    cp_size: int,
    cp_rank: int,
    sequence_parallel: bool,
    tp_size: int,
    scatter_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> RouterReplayMicrobatch:
    routes, valid = _pack_router_replay(
        batch,
        old_cu_seqlens=old_cu_seqlens,
        padded_cu_seqlens=padded_cu_seqlens,
        use_padded_seq=use_padded_seq,
    )
    if use_padded_seq and cp_size > 1:
        raise RouterReplayError("R3 padded BSHD layout cannot be combined with context parallelism")
    if cp_size > 1:
        routes = split_packed_for_context_parallel(routes, padded_cu_seqlens, cp_size=cp_size, cp_rank=cp_rank)
        valid = split_packed_for_context_parallel(valid, padded_cu_seqlens, cp_size=cp_size, cp_rank=cp_rank)
    if sequence_parallel and tp_size > 1:
        if scatter_fn is None:
            from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region

            scatter_fn = scatter_to_sequence_parallel_region
        routes = scatter_fn(routes)
        valid = scatter_fn(valid.to(torch.uint8).unsqueeze(-1)).squeeze(-1).bool()
    return RouterReplayMicrobatch(
        routes=routes.contiguous(),
        valid=valid.contiguous(),
        expected_layers=expected_layers,
    )


def _remove_from_native_global_registry(replay: Any, replay_cls: type) -> None:
    """Keep Platoon's PP/VP-local replay objects out of MCore's global registry.

    MCore registers every ``RouterReplay`` instance in construction order. That
    ordering is not stable across PP/VP chunks, and Platoon never uses MCore's
    global replay helpers, so retaining these objects there is both misleading
    and a potential lifetime leak.
    """

    instances = replay_cls.global_router_replay_instances
    instances[:] = [candidate for candidate in instances if candidate is not replay]


def _collect_router_bindings(
    model: Any,
    *,
    attach_missing: bool = False,
    replay_cls: type | None = None,
) -> tuple[RouterBinding, ...]:
    unwrapped = getattr(model, "module", model)
    bindings: list[RouterBinding] = []
    seen_replays: set[int] = set()
    if attach_missing and replay_cls is None:
        from megatron.core.transformer.moe.router_replay import RouterReplay

        replay_cls = RouterReplay
    for module in unwrapped.modules():
        # TopKRouter initializes this attribute to None when native replay is
        # disabled. Attaching after model construction avoids relying on the
        # transient Megatron-Bridge provider used to construct the model.
        if not hasattr(module, "router_replay"):
            continue
        replay = getattr(module, "router_replay", None)
        if replay is None:
            if not attach_missing:
                continue
            assert replay_cls is not None
            replay = replay_cls()
            module.router_replay = replay
        if attach_missing:
            assert replay_cls is not None
            if not isinstance(replay, replay_cls):
                raise RouterReplayError(
                    f"Router layer {getattr(module, 'layer_number', None)} has an unsupported "
                    f"replay object {type(replay).__name__}"
                )
            router_config = getattr(module, "config", None)
            if router_config is None or not hasattr(router_config, "moe_enable_routing_replay"):
                raise RouterReplayError("MCore router config lacks moe_enable_routing_replay")
            router_config.moe_enable_routing_replay = True
            _remove_from_native_global_registry(replay, replay_cls)
        layer_number = getattr(module, "layer_number", None)
        if not isinstance(layer_number, int) or layer_number <= 0:
            raise RouterReplayError("MCore router replay instance has no positive global layer_number")
        if id(replay) in seen_replays:
            continue
        seen_replays.add(id(replay))
        bindings.append(RouterBinding(layer_index=layer_number - 1, router=module, replay=replay))
    return tuple(sorted(bindings, key=lambda item: item.layer_index))


def _collect_chunk_transformer_layers(model: Any) -> frozenset[int]:
    """Return global layer indices owned by one PP/VP model chunk."""

    unwrapped = model
    seen_wrappers: set[int] = set()
    while isinstance(getattr(unwrapped, "module", None), torch.nn.Module):
        if id(unwrapped) in seen_wrappers:
            break
        seen_wrappers.add(id(unwrapped))
        unwrapped = unwrapped.module

    language_model = getattr(unwrapped, "language_model", unwrapped)
    while isinstance(getattr(language_model, "module", None), torch.nn.Module):
        language_model = language_model.module
    decoder = getattr(language_model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
        return frozenset()

    indices: set[int] = set()
    for layer in layers:
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int) and layer_number > 0:
            indices.add(layer_number - 1)
    return frozenset(indices)


def _layer_coverage_errors(payloads: Sequence[Mapping[str, Any]], num_layers: int) -> list[str]:
    errors = [f"rank {payload['rank']}: {error}" for payload in payloads for error in payload.get("errors", ())]
    expected = set(range(num_layers))
    union = {layer for payload in payloads for layer in payload.get("layers", ())}
    if union != expected:
        errors.append(
            "distributed PP/VP layer union mismatch: "
            f"missing={sorted(expected - union)}, unexpected={sorted(union - expected)}"
        )

    by_pp_rank: dict[int, dict[tuple[int, ...], list[int]]] = {}
    for payload in payloads:
        variants = by_pp_rank.setdefault(int(payload.get("pp_rank", -1)), {})
        layers = tuple(payload.get("layers", ()))
        variants.setdefault(layers, []).append(int(payload.get("rank", -1)))
    for pp_rank, variants in by_pp_rank.items():
        if len(variants) > 1:
            errors.append(f"PP rank {pp_rank} has inconsistent router layers across DP/TP/CP/EP replicas: {variants}")
    canonical_pp_layers = {
        pp_rank: set(next(iter(variants))) for pp_rank, variants in by_pp_rank.items() if len(variants) == 1
    }
    owners: dict[int, list[int]] = {}
    for pp_rank, layers in canonical_pp_layers.items():
        for layer in layers:
            owners.setdefault(layer, []).append(pp_rank)
    overlaps = {layer: ranks for layer, ranks in owners.items() if len(ranks) > 1}
    if overlaps:
        errors.append(f"transformer layers are owned by multiple PP ranks: {overlaps}")
    return errors


def _validate_distributed_layer_coverage(
    local_layers: frozenset[int],
    num_layers: int,
    local_errors: Sequence[str],
    *,
    dist_module: Any | None = None,
    pp_rank: int | None = None,
) -> None:
    if dist_module is None:
        import torch.distributed as dist_module

    distributed = bool(dist_module.is_available() and dist_module.is_initialized())
    rank = int(dist_module.get_rank()) if distributed else 0
    if pp_rank is None:
        try:
            from megatron.core import parallel_state as mpu

            pp_rank = int(mpu.get_pipeline_model_parallel_rank())
        except Exception:
            pp_rank = 0
    payload = {
        "rank": rank,
        "pp_rank": pp_rank,
        "layers": tuple(sorted(local_layers)),
        "errors": tuple(local_errors),
    }
    if distributed:
        payloads: list[dict[str, Any] | None] = [None] * int(dist_module.get_world_size())
        dist_module.all_gather_object(payloads, payload)
        gathered = [item for item in payloads if item is not None]
    else:
        gathered = [payload]
    errors = _layer_coverage_errors(gathered, num_layers)
    if errors:
        raise RouterReplayError("R3 model binding validation failed: " + "; ".join(errors))


def _collective_raise_on_error(
    local_error: str | None,
    *,
    phase: str,
    dist_module: Any | None = None,
    device: torch.device | str | None = None,
) -> None:
    """Make every rank fail before leaving a replay validation phase."""

    if dist_module is None:
        import torch.distributed as dist_module

    distributed = bool(dist_module.is_available() and dist_module.is_initialized())
    rank = int(dist_module.get_rank()) if distributed else 0
    payload = {"rank": rank, "error": local_error}
    if distributed:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else "cpu"
        failed = torch.tensor([local_error is not None], dtype=torch.int32, device=device)
        dist_module.all_reduce(failed, op=dist_module.ReduceOp.MAX)
        if not bool(failed.item()):
            return
        gathered: list[dict[str, Any] | None] = [None] * int(dist_module.get_world_size())
        dist_module.all_gather_object(gathered, payload)
        failures = [item for item in gathered if item is not None and item.get("error")]
    else:
        failures = [payload] if local_error else []
    if failures:
        details = "; ".join(f"rank {item['rank']}: {item['error']}" for item in failures)
        raise RouterReplayError(f"R3 {phase} failed collectively: {details}")


def _reset_replay_runtime(replay: Any) -> None:
    replay.clear_router_replay_action()
    replay.clear_indices()
    replay._platoon_target_valid = None
    replay._platoon_backward_valid = []
    replay._platoon_forward_calls = 0
    replay._platoon_backward_calls = 0
    replay._platoon_device_counters = None


def _set_binding_forward_data(binding: RouterBinding, microbatch: RouterReplayMicrobatch, action: Any) -> None:
    if binding.layer_index >= microbatch.routes.shape[1]:
        raise RouterReplayError(
            f"R3 has {microbatch.routes.shape[1]} global layers but local router requests layer {binding.layer_index}"
        )
    # Keep rollout IDs compact in the checkpoint-recompute queue. The native
    # hot path casts the active slice to int64 immediately before gather.
    target = microbatch.routes[:, binding.layer_index, :].contiguous()
    mask = microbatch.valid.contiguous()
    binding.replay.set_target_indices(target)
    binding.replay._platoon_target_valid = mask
    binding.replay._platoon_backward_valid.append(mask)
    binding.replay.set_router_replay_action(action)
    microbatch.seen_layers.add(binding.layer_index)


def _patched_native_get_replay_topk(
    replay: Any,
    scores: torch.Tensor,
    topk: int,
    num_groups: int | None,
    group_topk: int | None,
    default_compute_topk: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    replay_forward_action: Any,
    replay_backward_action: Any,
    original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    action = replay.router_replay_action
    if action not in (replay_forward_action, replay_backward_action):
        return original(replay, scores, topk, num_groups, group_topk, default_compute_topk)

    if action == replay_forward_action:
        target = replay.target_topk_idx
        valid = getattr(replay, "_platoon_target_valid", None)
        replay._platoon_forward_calls = getattr(replay, "_platoon_forward_calls", 0) + 1
    else:
        masks = getattr(replay, "_platoon_backward_valid", None)
        if not replay.replay_backward_list or not masks:
            raise RouterReplayError("R3 checkpoint recompute queue underflow")
        target = replay.replay_backward_list.pop(0)
        valid = masks.pop(0)
        replay._platoon_backward_calls = getattr(replay, "_platoon_backward_calls", 0) + 1

    if target is None or valid is None:
        raise RouterReplayError("R3 replay action has no target indices or validity mask")
    target = target.to(device=scores.device, dtype=torch.long)
    valid = valid.to(device=scores.device, dtype=torch.bool)
    if target.shape != (scores.shape[0], topk) or valid.shape != (scores.shape[0],):
        raise RouterReplayError(
            "R3 local token alignment mismatch: "
            f"scores={tuple(scores.shape)}, target={tuple(target.shape)}, valid={tuple(valid.shape)}"
        )

    invalid = ~valid
    # Strict R3 batches always contain terminal/padding rows that must use the
    # live router. Computing live top-k unconditionally avoids a device-to-host
    # predicate sync in every MoE layer. It also gives us a free diagnostic on
    # valid rows; no extra top-k invocation is made for that comparison.
    _, live_indices = default_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
    zero = torch.zeros((), dtype=torch.int64, device=scores.device)
    if action == replay_forward_action:
        compared = valid.sum(dtype=torch.int64)
        mismatch = (
            valid
            & torch.any(
                torch.sort(target, dim=1).values != torch.sort(live_indices, dim=1).values,
                dim=1,
            )
        ).sum(dtype=torch.int64)
    else:
        compared = zero
        mismatch = zero
    counter_delta = torch.stack(
        (
            invalid.sum(dtype=torch.int64),
            torch.full((), scores.shape[0], dtype=torch.int64, device=scores.device),
            compared,
            mismatch,
        )
    ).detach()
    counters = getattr(replay, "_platoon_device_counters", None)
    replay._platoon_device_counters = counter_delta if counters is None else counters + counter_delta
    target = torch.where(valid.unsqueeze(1), target, live_indices)
    return scores.gather(1, target), target


def install_native_router_replay_fallback() -> None:
    """Teach native MCore replay to live-route explicitly invalid rows."""

    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

    original = RouterReplay.get_replay_topk
    if getattr(original, "__platoon_router_replay_fallback__", False):
        return

    @wraps(original)
    def _get_replay_topk_with_fallback(
        self,
        scores,
        topk,
        num_groups=None,
        group_topk=None,
        default_compute_topk=None,
    ):
        return _patched_native_get_replay_topk(
            self,
            scores,
            topk,
            num_groups,
            group_topk,
            default_compute_topk,
            replay_forward_action=RouterReplayAction.REPLAY_FORWARD,
            replay_backward_action=RouterReplayAction.REPLAY_BACKWARD,
            original=original,
        )

    _get_replay_topk_with_fallback.__platoon_router_replay_fallback__ = True
    RouterReplay.get_replay_topk = _get_replay_topk_with_fallback


def install_areal_router_replay_hooks() -> None:
    """Install the inert hook around AReaL's packed model forward."""

    import areal.engine.megatron_engine as megatron_engine

    original_packed_forward = megatron_engine.packed_context_parallel_forward
    if getattr(original_packed_forward, "__platoon_router_replay_forward__", False):
        return

    @wraps(original_packed_forward)
    def _packed_forward_with_router_replay(model, input_, *args, **kwargs):
        microbatch = input_.get(_MICROBATCH_FIELD)
        if microbatch is None:
            return original_packed_forward(model, input_, *args, **kwargs)
        bindings = getattr(model, _BINDINGS_ATTR, None)
        if bindings is None:
            bindings = getattr(getattr(model, "module", None), _BINDINGS_ATTR, ())
        if not bindings:
            raise RouterReplayError("R3 microbatch reached a PP/VP model chunk with no router binding metadata")

        from megatron.core.transformer.moe.router_replay import RouterReplayAction

        before_calls = {
            id(binding.replay): getattr(binding.replay, "_platoon_forward_calls", 0) for binding in bindings
        }
        for binding in bindings:
            _set_binding_forward_data(binding, microbatch, RouterReplayAction.REPLAY_FORWARD)
        try:
            output = original_packed_forward(model, input_, *args, **kwargs)
            for binding in bindings:
                observed = getattr(binding.replay, "_platoon_forward_calls", 0) - before_calls[id(binding.replay)]
                if observed != 1:
                    raise RouterReplayError(
                        f"R3 router layer {binding.layer_index} was invoked {observed} times; expected once"
                    )
            return output
        finally:
            for binding in bindings:
                binding.replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    _packed_forward_with_router_replay.__platoon_router_replay_forward__ = True
    megatron_engine.packed_context_parallel_forward = _packed_forward_with_router_replay


@contextmanager
def router_replay_initialization(engine: Any):
    enabled = bool(getattr(engine.config, "enable_router_replay", False))
    if enabled:
        install_areal_router_replay_hooks()
        install_native_router_replay_fallback()
    yield


def configure_router_replay_engine(engine: Any) -> None:
    """Validate the constructed MCore model and cache its PP/VP bindings."""

    enabled = bool(getattr(engine.config, "enable_router_replay", False))
    setattr(engine, _PENDING_ATTR, None)
    if not enabled:
        return
    tf_config = engine.tf_config
    if not hasattr(tf_config, "moe_enable_routing_replay"):
        raise RouterReplayError("Pinned Megatron-Core lacks TransformerConfig.moe_enable_routing_replay")
    if bool(getattr(tf_config, "moe_router_fusion", False)):
        raise RouterReplayError(
            "R3 is incompatible with moe_router_fusion: MCore's fused top-k path bypasses RouterReplay"
        )
    routing_type = getattr(tf_config, "moe_router_load_balancing_type", None)
    if _uses_sinkhorn(routing_type):
        raise RouterReplayError("R3 is incompatible with sinkhorn routing, which bypasses RouterReplay")
    checkpointing_requested = bool(getattr(engine.config, "gradient_checkpointing", False))
    requested_recompute_config = getattr(engine, "mcore_config", None)
    if requested_recompute_config is None:
        # Engines outside AReaL's Megatron implementation may expose only one
        # config object. The real AReaL engine always has mcore_config.
        requested_recompute_config = tf_config
    requested_recompute_mode = _recompute_mode(requested_recompute_config)
    if checkpointing_requested and requested_recompute_mode != ("full", "uniform"):
        raise RouterReplayError(
            "R3 gradient checkpointing requires full/uniform recompute so native FIFO replay "
            "consumes every queued microbatch in order; configured "
            f"recompute_granularity={requested_recompute_mode[0]!r}, "
            f"recompute_method={requested_recompute_mode[1]!r}"
        )
    configured_layers = getattr(engine.config, "router_replay_num_layers", None)
    configured_topk = getattr(engine.config, "router_replay_topk", None)
    configured_experts = getattr(engine.config, "router_replay_num_experts", None)
    actual_layers = int(tf_config.num_layers)
    actual_topk = int(tf_config.moe_router_topk)
    actual_experts = int(tf_config.num_moe_experts)
    if configured_layers != actual_layers:
        raise RouterReplayError(
            f"actor.router_replay_num_layers={configured_layers} but model has {actual_layers} layers"
        )
    if configured_topk != actual_topk:
        raise RouterReplayError(f"actor.router_replay_topk={configured_topk} but model uses topk={actual_topk}")
    if configured_experts is not None and configured_experts != actual_experts:
        raise RouterReplayError(
            f"actor.router_replay_num_experts={configured_experts} but model has {actual_experts} experts"
        )

    all_bindings: list[RouterBinding] = []
    seen_layers: set[int] = set()
    local_errors: list[str] = []
    runtime_recompute_modes: set[tuple[str | None, str | None]] = set()
    runtime_sequence_parallel_values: set[bool] = set()
    models = list(engine.model or [])
    for chunk_index, model in enumerate(models):
        try:
            expected_chunk_layers = _collect_chunk_transformer_layers(model)
        except Exception as exc:
            expected_chunk_layers = frozenset()
            local_errors.append(f"chunk {chunk_index} transformer-layer discovery failed: {exc}")
        try:
            bindings = _collect_router_bindings(model, attach_missing=True)
        except Exception as exc:
            bindings = ()
            local_errors.append(f"chunk {chunk_index} router binding failed: {exc}")
        setattr(model, _BINDINGS_ATTR, bindings)
        module = getattr(model, "module", None)
        if module is not None:
            setattr(module, _BINDINGS_ATTR, bindings)
        binding_layers = frozenset(binding.layer_index for binding in bindings)
        if expected_chunk_layers and binding_layers != expected_chunk_layers:
            local_errors.append(
                f"chunk {chunk_index} router coverage mismatch: "
                f"missing={sorted(expected_chunk_layers - binding_layers)}, "
                f"unexpected={sorted(binding_layers - expected_chunk_layers)}"
            )
        for binding in bindings:
            if binding.layer_index >= actual_layers:
                local_errors.append(
                    f"Router layer index {binding.layer_index} is outside the {actual_layers} transformer layers; "
                    "MTP or another auxiliary router cannot consume rollout R3 data"
                )
            router_config = binding.router.config
            runtime_recompute_mode = _recompute_mode(router_config)
            runtime_recompute_modes.add(runtime_recompute_mode)
            runtime_sequence_parallel_values.add(bool(getattr(router_config, "sequence_parallel", False)))
            if checkpointing_requested and runtime_recompute_mode != ("full", "uniform"):
                local_errors.append(
                    f"R3 router layer {binding.layer_index} runtime config must use full/uniform recompute; "
                    f"got recompute_granularity={runtime_recompute_mode[0]!r}, "
                    f"recompute_method={runtime_recompute_mode[1]!r}"
                )
            elif runtime_recompute_mode[0] is not None and runtime_recompute_mode != ("full", "uniform"):
                local_errors.append(
                    f"R3 router layer {binding.layer_index} has unsupported runtime recompute mode "
                    f"recompute_granularity={runtime_recompute_mode[0]!r}, "
                    f"recompute_method={runtime_recompute_mode[1]!r}"
                )
            if bool(getattr(router_config, "moe_router_fusion", False)):
                local_errors.append(
                    f"R3 router layer {binding.layer_index} has moe_router_fusion enabled, which bypasses replay"
                )
            runtime_routing_type = getattr(router_config, "moe_router_load_balancing_type", None)
            if _uses_sinkhorn(runtime_routing_type):
                local_errors.append(
                    f"R3 router layer {binding.layer_index} uses sinkhorn routing, which bypasses replay"
                )
            runtime_num_layers = int(getattr(router_config, "num_layers", actual_layers))
            runtime_topk = int(getattr(router_config, "moe_router_topk", actual_topk))
            runtime_num_experts = int(getattr(router_config, "num_moe_experts", actual_experts))
            if runtime_num_layers != actual_layers:
                local_errors.append(
                    f"R3 router layer {binding.layer_index} reports {runtime_num_layers} global layers, "
                    f"but the engine snapshot reports {actual_layers}"
                )
            if runtime_topk != actual_topk:
                local_errors.append(
                    f"R3 router layer {binding.layer_index} config top-k {runtime_topk} does not match "
                    f"the engine snapshot {actual_topk}"
                )
            if runtime_num_experts != actual_experts:
                local_errors.append(
                    f"R3 router layer {binding.layer_index} reports {runtime_num_experts} experts, "
                    f"but the engine snapshot reports {actual_experts}"
                )
            if int(getattr(binding.router, "topk", actual_topk)) != actual_topk:
                local_errors.append(
                    f"R3 router layer {binding.layer_index} top-k does not match model top-k {actual_topk}"
                )
            if binding.layer_index in seen_layers:
                local_errors.append(f"Duplicate local RouterReplay binding for global layer {binding.layer_index}")
            seen_layers.add(binding.layer_index)
            all_bindings.append(binding)
    if not all_bindings:
        local_errors.append("this Megatron pipeline rank constructed no MoE routers")
    if len(runtime_recompute_modes) > 1:
        local_errors.append(
            f"R3 router configs have inconsistent runtime recompute modes: {sorted(runtime_recompute_modes, key=repr)}"
        )
    if len(runtime_sequence_parallel_values) > 1:
        local_errors.append(
            f"R3 router configs disagree on sequence_parallel: {sorted(runtime_sequence_parallel_values)}"
        )
    _validate_distributed_layer_coverage(
        frozenset(seen_layers),
        actual_layers,
        local_errors,
    )
    # Distributed validation above guarantees that this rank has bindings and
    # that their runtime modes agree. These constructed router configs are the
    # source of truth: with megatron-bridge, engine.tf_config is an earlier
    # snapshot created before AReaL applies recompute and sequence-parallel
    # overrides to the provider that actually materializes the model.
    assert len(runtime_recompute_modes) == 1
    assert len(runtime_sequence_parallel_values) == 1
    runtime_recompute_mode = next(iter(runtime_recompute_modes))
    full_recompute = runtime_recompute_mode == ("full", "uniform")
    sequence_parallel = next(iter(runtime_sequence_parallel_values))
    # This flag is introspection only: the per-router native objects above are
    # the runtime source of truth. No router loss coefficients are modified.
    tf_config.moe_enable_routing_replay = True
    state = RouterReplayEngineState(
        bindings=tuple(all_bindings),
        expected_layers=frozenset(seen_layers),
        num_layers=actual_layers,
        topk=actual_topk,
        num_experts=actual_experts,
        full_recompute=full_recompute,
        sequence_parallel=sequence_parallel,
    )
    setattr(engine, _STATE_ATTR, state)
    logger.info(
        "Enabled native MCore R3 on this rank: local_layers=%s global_layers=%d topk=%d experts=%d "
        "full_recompute=%s sequence_parallel=%s",
        sorted(seen_layers),
        actual_layers,
        actual_topk,
        actual_experts,
        full_recompute,
        sequence_parallel,
    )


def _split_engine_microbatches(batch: RouterReplayBatch, mb_list: Any) -> list[RouterReplayBatch]:
    routes, valid = batch.routes, batch.valid
    forward_indices = getattr(mb_list, "forward_indices", None)
    if forward_indices is not None:
        order = torch.as_tensor(forward_indices, dtype=torch.long, device=routes.device)
        routes = routes.index_select(0, order)
        valid = valid.index_select(0, order.to(valid.device))
    result: list[RouterReplayBatch] = []
    offset = 0
    for mb in mb_list.mbs:
        cu_seqlens = mb.get("cu_seqlens")
        if not torch.is_tensor(cu_seqlens):
            raise RouterReplayError("R3 requires packed Megatron microbatches with cu_seqlens")
        count = cu_seqlens.shape[0] - 1
        result.append(
            RouterReplayBatch(
                routes=routes[offset : offset + count],
                valid=valid[offset : offset + count],
            )
        )
        offset += count
    if offset != routes.shape[0]:
        raise RouterReplayError(f"Megatron R3 microbatch split covered {offset} samples, expected {routes.shape[0]}")
    return result


def _prepare_engine_microbatches(engine: Any, mb_list: Any, batch: RouterReplayBatch) -> list[RouterReplayMicrobatch]:
    from megatron.core import parallel_state as mpu

    state: RouterReplayEngineState = getattr(engine, _STATE_ATTR)
    split = _split_engine_microbatches(batch, mb_list)
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    prepared: list[RouterReplayMicrobatch] = []
    for index, sub_batch in enumerate(split):
        padded_mb = mb_list.padded_mbs[index]
        padded_cu = padded_mb.get("cu_seqlens")
        old_cu = (
            mb_list.old_cu_seqlens_list[index]
            if mb_list.old_cu_seqlens_list is not None
            else mb_list.mbs[index].get("cu_seqlens")
        )
        if not torch.is_tensor(padded_cu) or not torch.is_tensor(old_cu):
            raise RouterReplayError("R3 could not resolve padded/original cu_seqlens")
        prepared.append(
            build_local_router_replay_microbatch(
                sub_batch,
                old_cu_seqlens=old_cu,
                padded_cu_seqlens=padded_cu,
                expected_layers=state.expected_layers,
                use_padded_seq=bool(getattr(engine, "use_padded_seq", False)),
                cp_size=cp_size,
                cp_rank=cp_rank,
                sequence_parallel=state.sequence_parallel,
                tp_size=tp_size,
            )
        )
    return prepared


def _materialize_device_counters(bindings: Sequence[RouterBinding]) -> tuple[int, int, int, int]:
    """Reduce all hot-path counters with one device synchronization per batch."""

    counters = [
        binding.replay._platoon_device_counters
        for binding in bindings
        if getattr(binding.replay, "_platoon_device_counters", None) is not None
    ]
    if not counters:
        return 0, 0, 0, 0
    totals = torch.stack(counters).sum(dim=0).to(device="cpu").tolist()
    return tuple(int(value) for value in totals)


def _validate_and_record_cycle(engine: Any, microbatches: Sequence[RouterReplayMicrobatch]) -> None:
    state: RouterReplayEngineState = getattr(engine, _STATE_ATTR)
    expected_coverage = len(state.expected_layers) * len(microbatches)
    observed_coverage = sum(len(microbatch.seen_layers) for microbatch in microbatches)
    missing = [
        (index, sorted(state.expected_layers - microbatch.seen_layers))
        for index, microbatch in enumerate(microbatches)
        if microbatch.seen_layers != state.expected_layers
    ]
    forward_calls = sum(getattr(binding.replay, "_platoon_forward_calls", 0) for binding in state.bindings)
    backward_calls = sum(getattr(binding.replay, "_platoon_backward_calls", 0) for binding in state.bindings)
    fallback_tokens, total_tokens, live_compared_tokens, live_mismatch_tokens = _materialize_device_counters(
        state.bindings
    )
    full_recompute = state.full_recompute
    expected_calls = expected_coverage
    queue_mismatches = []
    for binding in state.bindings:
        index_count = len(binding.replay.replay_backward_list)
        mask_count = len(getattr(binding.replay, "_platoon_backward_valid", []))
        if index_count != mask_count:
            queue_mismatches.append((binding.layer_index, index_count, mask_count))

    _record_metrics(
        local_layer_coverage=float(observed_coverage / max(expected_coverage, 1)),
        forward_replay_calls=float(forward_calls),
        backward_replay_calls=float(backward_calls),
        live_fallback_fraction=float(fallback_tokens / max(total_tokens, 1)),
        live_route_set_mismatch_fraction=float(live_mismatch_tokens / max(live_compared_tokens, 1)),
        live_route_set_compared_tokens=float(live_compared_tokens),
    )
    errors = []
    if missing:
        errors.append(f"missing PP/VP layer coverage {missing}")
    if forward_calls != expected_calls:
        errors.append(f"forward replay calls={forward_calls}, expected={expected_calls}")
    if queue_mismatches:
        errors.append(f"index/mask queue mismatches={queue_mismatches}")
    if full_recompute:
        if backward_calls != expected_calls:
            errors.append(f"checkpoint replay calls={backward_calls}, expected={expected_calls}")
        unconsumed = [
            (binding.layer_index, len(binding.replay.replay_backward_list))
            for binding in state.bindings
            if binding.replay.replay_backward_list
        ]
        if unconsumed:
            errors.append(f"unconsumed checkpoint replay entries={unconsumed}")
    elif backward_calls > forward_calls:
        errors.append(f"checkpoint replay calls={backward_calls} exceed forward calls={forward_calls}")
    if errors:
        raise RouterReplayError("R3 fail-closed lifecycle check failed: " + "; ".join(errors))


def run_router_replay_forward_backward(
    engine: Any,
    original: Callable[..., Any],
    mb_list: Any,
    process_output_fn: Callable[..., Any],
    *,
    forward_only: bool = False,
    gather_cp_output: bool = False,
) -> Any:
    """Wrap AReaL's scheduler while keeping its forward/backward implementation intact."""

    enabled = bool(getattr(engine.config, "enable_router_replay", False))
    pending = getattr(engine, _PENDING_ATTR, None)
    setattr(engine, _PENDING_ATTR, None)
    if forward_only:
        if pending is not None:
            raise RouterReplayError("R3 training data was staged for a forward-only call")
        return original(
            mb_list,
            process_output_fn,
            forward_only=forward_only,
            gather_cp_output=gather_cp_output,
        )
    if not enabled:
        if pending is not None:
            raise RouterReplayError("R3 data reached a disabled Megatron engine")
        return original(
            mb_list,
            process_output_fn,
            forward_only=forward_only,
            gather_cp_output=gather_cp_output,
        )
    state: RouterReplayEngineState | None = getattr(engine, _STATE_ATTR, None)
    microbatches: list[RouterReplayMicrobatch] = []
    preflight_error = None
    try:
        if pending is None:
            raise RouterReplayError("R3 is enabled but this training batch has no staged routed experts")
        if state is None:
            raise RouterReplayError("R3 engine state was not configured after model initialization")
        valid_routes = pending.routes[pending.valid.to(pending.routes.device)]
        if valid_routes.numel():
            min_id = int(valid_routes.min().item())
            max_id = int(valid_routes.max().item())
            if min_id < 0 or max_id >= state.num_experts:
                raise RouterReplayError(
                    f"R3 expert ids [{min_id}, {max_id}] exceed model range [0, {state.num_experts})"
                )

        for binding in state.bindings:
            _reset_replay_runtime(binding.replay)
        microbatches = _prepare_engine_microbatches(engine, mb_list, pending)
        for padded_mb, microbatch in zip(mb_list.padded_mbs, microbatches, strict=True):
            padded_mb[_MICROBATCH_FIELD] = microbatch
    except Exception as exc:
        preflight_error = f"{type(exc).__name__}: {exc}"
    try:
        _collective_raise_on_error(preflight_error, phase="preflight")
    except Exception:
        for padded_mb in mb_list.padded_mbs:
            padded_mb.pop(_MICROBATCH_FIELD, None)
        if state is not None:
            for binding in state.bindings:
                _reset_replay_runtime(binding.replay)
        raise
    assert state is not None
    try:
        result = original(
            mb_list,
            process_output_fn,
            forward_only=forward_only,
            gather_cp_output=gather_cp_output,
        )
        lifecycle_error = None
        try:
            _validate_and_record_cycle(engine, microbatches)
        except Exception as exc:
            lifecycle_error = f"{type(exc).__name__}: {exc}"
        _collective_raise_on_error(lifecycle_error, phase="post-backward lifecycle validation")
        return result
    finally:
        for padded_mb in mb_list.padded_mbs:
            padded_mb.pop(_MICROBATCH_FIELD, None)
        for binding in state.bindings:
            _reset_replay_runtime(binding.replay)
