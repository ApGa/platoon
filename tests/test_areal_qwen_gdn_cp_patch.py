from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


def _load_patches_module(module_name: str = "platoon_areal_patches_test"):
    spec = importlib.util.spec_from_file_location(module_name, PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_subchunk(sample_id: int, sub_id: int, chunk_len: int) -> torch.Tensor:
    base = sample_id * 1000 + sub_id * 100
    return torch.arange(base, base + chunk_len, dtype=torch.float32).view(-1, 1, 1)


def _build_rank_inputs(rank: int, world_size: int):
    chunk_lens = [3, 5]
    tail_pad_local_len = 3
    zigzag_chunks = []
    full_sequences = []
    cu_seqlens = [0]

    for sample_id, chunk_len in enumerate(chunk_lens):
        subchunks = [_make_subchunk(sample_id, sub_id, chunk_len) for sub_id in range(2 * world_size)]
        zigzag_chunks.extend([subchunks[rank], subchunks[2 * world_size - 1 - rank]])
        full_sequences.append(torch.cat(subchunks, dim=0))
        cu_seqlens.append(cu_seqlens[-1] + 2 * world_size * chunk_len)

    tail_pad = (rank * 10000 + torch.arange(tail_pad_local_len, dtype=torch.float32)).view(-1, 1, 1)
    zigzag_chunks.append(tail_pad)
    full_sequences.append(
        torch.cat(
            [
                (r * 10000 + torch.arange(tail_pad_local_len, dtype=torch.float32)).view(-1, 1, 1)
                for r in range(world_size)
            ],
            dim=0,
        )
    )
    cu_seqlens.append(cu_seqlens[-1] + world_size * tail_pad_local_len)

    zigzag = torch.cat(zigzag_chunks, dim=0).requires_grad_(True)
    packed_full = torch.cat(full_sequences, dim=0)
    local_len = zigzag.size(0)
    packed_shard = packed_full[rank * local_len : (rank + 1) * local_len]
    return zigzag, packed_shard, torch.tensor(cu_seqlens, dtype=torch.int32)


def _relayout_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        patches = _load_patches_module(f"platoon_areal_patches_test_rank_{rank}")
        zigzag, expected_packed_shard, cu_seqlens = _build_rank_inputs(rank, world_size)
        packed_shard = patches._zigzag_to_packed_shard(zigzag, cu_seqlens, dist.group.WORLD, rank, world_size)

        # Check the adjoint of the relayout itself, independently of the inverse
        # relayout.  A round trip alone could hide paired forward/backward bugs.
        packed_weight = 2 * packed_shard.detach() + 3
        (packed_shard * packed_weight).sum().backward(retain_graph=True)
        weighted_adjoint_ok = torch.equal(zigzag.grad, 2 * zigzag.detach() + 3)
        zigzag.grad = None

        roundtrip = patches._packed_shard_to_zigzag(packed_shard, cu_seqlens, dist.group.WORLD, rank, world_size)

        loss = roundtrip.sum()
        loss.backward()

        # The packed Conv1d path reconstructs the global contiguous stream on
        # every CP rank, but each rank only keeps its output shard. Verify both
        # sides of that distributed backward: all-gather's reduce-scatter must
        # return the reference input-gradient shard, and summing the replicated
        # weight gradients across CP must reproduce the full-sequence result.
        channels = 3
        kernel_size = 3
        conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size - 1,
            groups=channels,
            bias=True,
            dtype=torch.float64,
        )
        with torch.no_grad():
            conv.weight.copy_(torch.linspace(-0.3, 0.4, channels * kernel_size).view(channels, 1, kernel_size))
            conv.bias.copy_(torch.linspace(-0.1, 0.2, channels))
        reference_conv = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size - 1,
            groups=channels,
            bias=True,
            dtype=torch.float64,
        )
        reference_conv.load_state_dict(conv.state_dict())
        module = types.SimpleNamespace(conv1d=conv, act_fn=torch.tanh)
        reference_module = types.SimpleNamespace(conv1d=reference_conv, act_fn=torch.tanh)

        conv_cu_seqlens = torch.tensor([0, 5, 12], dtype=torch.int32)
        full_input = torch.linspace(-1.0, 1.0, 12 * channels, dtype=torch.float64).view(12, channels)
        conv_local_len = full_input.size(0) // world_size
        local_input = full_input[rank * conv_local_len : (rank + 1) * conv_local_len].clone().requires_grad_(True)
        local_output = patches._apply_packed_causal_conv1d_reference_for_gdn(
            module,
            local_input.unsqueeze(0),
            conv_cu_seqlens,
            dist.group.WORLD,
            rank,
            world_size,
        )
        full_output_weight = torch.linspace(0.2, 1.7, 12 * channels, dtype=torch.float64).view(1, 12, channels)
        local_output_weight = full_output_weight[:, rank * conv_local_len : (rank + 1) * conv_local_len]
        (local_output * local_output_weight).sum().backward()

        dist.all_reduce(conv.weight.grad, op=dist.ReduceOp.SUM)
        dist.all_reduce(conv.bias.grad, op=dist.ReduceOp.SUM)

        reference_input = full_input.clone().requires_grad_(True)
        reference_output = patches._apply_packed_causal_conv1d_reference_for_gdn(
            reference_module,
            reference_input.unsqueeze(0),
            conv_cu_seqlens,
            cp_group=types.SimpleNamespace(),
            cp_rank=0,
            cp_size=1,
        )
        (reference_output * full_output_weight).sum().backward()
        reference_slice = slice(rank * conv_local_len, (rank + 1) * conv_local_len)

        result = {
            "packed_ok": torch.equal(packed_shard, expected_packed_shard),
            "roundtrip_ok": torch.equal(roundtrip, zigzag),
            "grad_ok": torch.equal(zigzag.grad, torch.ones_like(zigzag)),
            "weighted_adjoint_ok": weighted_adjoint_ok,
            "conv_output_ok": torch.allclose(local_output, reference_output[:, reference_slice], atol=1e-12),
            "conv_input_grad_ok": torch.allclose(
                local_input.grad,
                reference_input.grad[reference_slice],
                atol=1e-12,
            ),
            "conv_weight_grad_ok": torch.allclose(conv.weight.grad, reference_conv.weight.grad, atol=1e-12),
            "conv_bias_grad_ok": torch.allclose(conv.bias.grad, reference_conv.bias.grad, atol=1e-12),
        }
        Path(result_dir, f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _fla_conv_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from fla.ops.cp import build_cp_context
        from torch.utils.checkpoint import checkpoint

        os.environ["PLATOON_QWEN35_GDN_CP_CONV_BACKEND"] = "fla"
        patches = _load_patches_module(f"platoon_areal_patches_fla_conv_rank_{rank}")
        channels = 4
        kernel_size = 4
        total_tokens = 16
        local_tokens = total_tokens // world_size
        device = torch.device("cuda", rank)
        dtype = torch.bfloat16
        results = {}

        for case_name, (boundaries, use_checkpoint) in {
            # The CP cut at token 8 crosses a sequence that begins at token 6.
            # Only two predecessor tokens are valid; token 5 must not leak in.
            "crossing_boundary": ([0, 6, 9, 16], False),
            # The CP cut is exactly a sequence boundary, so rank 1 must ignore
            # every halo token received from rank 0.
            "exact_boundary": ([0, 8, 16], False),
            # Production uses full activation recomputation, which executes
            # the convolution collective again during backward.
            "checkpointed_crossing_boundary": ([0, 6, 9, 16], True),
        }.items():
            conv = torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size - 1,
                groups=channels,
                bias=False,
                device=device,
                dtype=dtype,
            )
            reference_conv = torch.nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size - 1,
                groups=channels,
                bias=False,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                weights = torch.linspace(
                    -0.2,
                    0.3,
                    channels * kernel_size,
                    device=device,
                    dtype=torch.float32,
                ).to(dtype)
                conv.weight.copy_(weights.view(channels, 1, kernel_size))
                reference_conv.weight.copy_(conv.weight)

            module = types.SimpleNamespace(
                conv1d=conv,
                conv_kernel_dim=kernel_size,
                act_fn=torch.nn.functional.silu,
            )
            reference_module = types.SimpleNamespace(
                conv1d=reference_conv,
                act_fn=torch.nn.functional.silu,
            )
            cu_seqlens = torch.tensor(boundaries, device=device, dtype=torch.int32)
            full_input = (
                torch.linspace(
                    -1.0,
                    1.0,
                    total_tokens * channels,
                    device=device,
                    dtype=torch.float32,
                )
                .to(dtype)
                .view(total_tokens, channels)
            )
            local_slice = slice(rank * local_tokens, (rank + 1) * local_tokens)
            local_input = full_input[local_slice].clone().requires_grad_(True)
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens,
                group=dist.group.WORLD,
                conv1d_kernel_size=kernel_size,
            )

            def apply_local_conv(x):
                return patches._apply_packed_causal_conv1d_for_gdn(
                    module,
                    x,
                    cu_seqlens,
                    dist.group.WORLD,
                    rank,
                    world_size,
                    cp_context,
                )

            local_qkv = local_input.unsqueeze(0)
            local_output = (
                checkpoint(apply_local_conv, local_qkv, use_reentrant=False)
                if use_checkpoint
                else apply_local_conv(local_qkv)
            )

            reference_input = full_input.clone().requires_grad_(True)
            reference_output = patches._apply_packed_causal_conv1d_reference_for_gdn(
                reference_module,
                reference_input.unsqueeze(0),
                cu_seqlens,
                cp_group=types.SimpleNamespace(),
                cp_rank=0,
                cp_size=1,
            )
            output_weight = (
                torch.linspace(
                    0.2,
                    1.7,
                    total_tokens * channels,
                    device=device,
                    dtype=torch.float32,
                )
                .to(dtype)
                .view(1, total_tokens, channels)
            )
            (local_output * output_weight[:, local_slice]).sum().backward()
            dist.all_reduce(conv.weight.grad, op=dist.ReduceOp.SUM)
            (reference_output * output_weight).sum().backward()

            results[case_name] = {
                "output": torch.allclose(
                    local_output.float(),
                    reference_output[:, local_slice].float(),
                    atol=2e-2,
                    rtol=2e-2,
                ),
                "input_grad": torch.allclose(
                    local_input.grad.float(),
                    reference_input.grad[local_slice].float(),
                    atol=2e-2,
                    rtol=2e-2,
                ),
                "weight_grad": torch.allclose(
                    conv.weight.grad.float(),
                    reference_conv.weight.grad.float(),
                    atol=2e-2,
                    rtol=2e-2,
                ),
            }

        Path(result_dir, f"rank_{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_qwen_gdn_cp_relayout_roundtrip_and_backward(tmp_path: Path):
    world_size = 2
    init_file = tmp_path / "dist_init"
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    mp.spawn(
        _relayout_worker,
        args=(world_size, str(init_file), str(result_dir)),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result = json.loads(Path(result_dir, f"rank_{rank}.json").read_text())
        assert result == {
            "packed_ok": True,
            "roundtrip_ok": True,
            "grad_ok": True,
            "weighted_adjoint_ok": True,
            "conv_output_ok": True,
            "conv_input_grad_ok": True,
            "conv_weight_grad_ok": True,
            "conv_bias_grad_ok": True,
        }


def test_qwen_gdn_cp_fla_convolution_matches_reference_on_two_gpus(tmp_path: Path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("FLA CP convolution parity requires two CUDA GPUs")
    if importlib.util.find_spec("fla") is None:
        pytest.skip("flash-linear-attention is not installed")

    world_size = 2
    init_file = tmp_path / "fla_dist_init"
    result_dir = tmp_path / "fla_results"
    result_dir.mkdir()
    mp.spawn(
        _fla_conv_worker,
        args=(world_size, str(init_file), str(result_dir)),
        nprocs=world_size,
        join=True,
    )

    expected = {
        "crossing_boundary": {"output": True, "input_grad": True, "weight_grad": True},
        "exact_boundary": {"output": True, "input_grad": True, "weight_grad": True},
        "checkpointed_crossing_boundary": {"output": True, "input_grad": True, "weight_grad": True},
    }
    for rank in range(world_size):
        assert json.loads(Path(result_dir, f"rank_{rank}.json").read_text()) == expected


def test_qwen_gdn_cp_uses_fla_local_convolution_by_default(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_fla_conv_test")
    monkeypatch.delenv("PLATOON_QWEN35_GDN_CP_CONV_BACKEND", raising=False)

    calls = {}

    def causal_conv1d(*, x, weight, bias, activation, cp_context):
        calls.update(
            x=x,
            weight=weight,
            bias=bias,
            activation=activation,
            cp_context=cp_context,
        )
        bias_term = 0 if bias is None else bias.sum()
        return x + weight.sum() + bias_term, None

    fla_mod = types.ModuleType("fla")
    fla_mod.__path__ = []
    modules_mod = types.ModuleType("fla.modules")
    modules_mod.__path__ = []
    convolution_mod = types.ModuleType("fla.modules.convolution")
    convolution_mod.causal_conv1d = causal_conv1d
    fla_mod.modules = modules_mod
    modules_mod.convolution = convolution_mod
    monkeypatch.setitem(sys.modules, "fla", fla_mod)
    monkeypatch.setitem(sys.modules, "fla.modules", modules_mod)
    monkeypatch.setitem(sys.modules, "fla.modules.convolution", convolution_mod)

    conv = torch.nn.Conv1d(3, 3, 4, groups=3, bias=False)
    module = types.SimpleNamespace(conv1d=conv, act_fn=torch.tanh)
    qkv = torch.randn(1, 6, 3, requires_grad=True)
    cp_context = types.SimpleNamespace(conv1d_kernel_size=4)

    output = patches._apply_packed_causal_conv1d_for_gdn(
        module,
        qkv,
        torch.tensor([0, 12], dtype=torch.int32),
        cp_group=object(),
        cp_rank=0,
        cp_size=2,
        cp_context=cp_context,
    )
    output.sum().backward()

    assert calls["x"].shape == qkv.shape
    assert calls["weight"].shape == (3, 4)
    assert calls["weight"].untyped_storage().data_ptr() == conv.weight.untyped_storage().data_ptr()
    assert calls["bias"] is None
    assert calls["activation"] is None
    assert calls["cp_context"] is cp_context
    assert torch.equal(output, torch.tanh(calls["x"] + calls["weight"].sum()))
    assert qkv.grad is not None
    assert conv.weight.grad is not None


def test_qwen_gdn_cp_reference_convolution_is_an_explicit_rollback(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_reference_conv_test")
    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP_CONV_BACKEND", "reference")

    expected = torch.randn(1, 4, 2)
    calls = []

    def reference(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(patches, "_apply_packed_causal_conv1d_reference_for_gdn", reference)
    output = patches._apply_packed_causal_conv1d_for_gdn(
        types.SimpleNamespace(),
        torch.randn(1, 4, 2),
        torch.tensor([0, 8], dtype=torch.int32),
        cp_group=object(),
        cp_rank=0,
        cp_size=2,
        cp_context=object(),
    )

    assert output is expected
    assert len(calls) == 1


def test_qwen_gdn_cp_local_thd_rope_bypass_survives_recompute(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_recompute_rope_test")

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    common_mod = types.ModuleType("megatron.core.models.common")
    embeddings_mod = types.ModuleType("megatron.core.models.common.embeddings")
    rope_utils_mod = types.ModuleType("megatron.core.models.common.embeddings.rope_utils")

    calls = {"local": 0, "upstream": 0}

    def apply_bshd(t, freqs, **_kwargs):
        calls["local"] += 1
        return t + freqs

    def apply_thd_upstream(t, _cu_seqlens, _freqs, **_kwargs):
        calls["upstream"] += 1
        return t - 1

    rope_utils_mod._apply_rotary_pos_emb_bshd = apply_bshd
    rope_utils_mod._apply_rotary_pos_emb_thd = apply_thd_upstream
    embeddings_mod.rope_utils = rope_utils_mod
    common_mod.embeddings = embeddings_mod
    models_mod.common = common_mod
    core_mod.models = models_mod
    megatron_mod.core = core_mod

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.models", models_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common", common_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common.embeddings", embeddings_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.models.common.embeddings.rope_utils", rope_utils_mod)
    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP", "1")

    patches._patch_megatron_core_qwen35_already_cp_local_thd_rope()

    cp_group = types.SimpleNamespace(size=lambda: 2)
    t = torch.zeros(8, 2, 4)
    local_freqs = torch.ones(8, 1, 1, 4)
    global_cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    # Normal model forward: the Qwen wrapper has installed its thread-local.
    patches._QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = {"local_total_len": t.size(0)}
    normal_forward = rope_utils_mod._apply_rotary_pos_emb_thd(
        t,
        global_cu_seqlens,
        local_freqs,
        cp_group=cp_group,
    )

    # Full activation checkpointing replays the attention layer after the
    # outer Qwen forward has returned and cleared this context.  Tensor shape
    # metadata must still select the same already-local path.
    patches._QWEN35_GDN_ALREADY_CP_LOCAL_CONTEXT.value = None
    recompute_forward = rope_utils_mod._apply_rotary_pos_emb_thd(
        t,
        global_cu_seqlens,
        local_freqs,
        cp_group=cp_group,
    )

    assert torch.equal(normal_forward, torch.ones_like(t))
    assert torch.equal(recompute_forward, normal_forward)
    assert calls == {"local": 2, "upstream": 0}

    # A full/global frequency table does not satisfy the already-local shape
    # contract and must retain Megatron's upstream THD behavior.
    global_freqs = torch.ones(16, 1, 1, 4)
    upstream_result = rope_utils_mod._apply_rotary_pos_emb_thd(
        t,
        global_cu_seqlens,
        global_freqs,
        cp_group=cp_group,
    )
    assert torch.equal(upstream_result, t - 1)
    assert calls == {"local": 2, "upstream": 1}


def test_qwen_gdn_cp_guard_patch_is_opt_in(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_guard_test")
    areal_mod = types.ModuleType("areal")
    engine_mod = types.ModuleType("areal.engine")
    core_mod = types.ModuleType("areal.engine.core")
    model_mod = types.ModuleType("areal.engine.core.model")

    model_mod.is_qwen3_5_model = lambda model_type: model_type in {"qwen3_5", "qwen3_5_moe"}
    model_mod.requires_padded_seq = lambda model_type: model_type == "qwen3_5_moe"
    model_mod.is_valid_vision_model = lambda model_type: model_type == "qwen3_5_moe"

    monkeypatch.setitem(sys.modules, "areal", areal_mod)
    monkeypatch.setitem(sys.modules, "areal.engine", engine_mod)
    monkeypatch.setitem(sys.modules, "areal.engine.core", core_mod)
    monkeypatch.setitem(sys.modules, "areal.engine.core.model", model_mod)

    monkeypatch.delenv("PLATOON_QWEN35_GDN_CP", raising=False)
    patches._patch_areal_qwen35_gdn_cp_guards()
    assert model_mod.requires_padded_seq("qwen3_5_moe") is True
    assert model_mod.is_valid_vision_model("qwen3_5_moe") is True

    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP", "1")
    patches._patch_areal_qwen35_gdn_cp_guards()
    assert model_mod.requires_padded_seq("qwen3_5_moe") is False
    assert model_mod.is_valid_vision_model("qwen3_5_moe") is False
    assert model_mod.requires_padded_seq("qwen3") is False
    assert model_mod.is_valid_vision_model("qwen3") is False


def test_qwen_gdn_cp_config_validation_patch_preserves_cp(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_config_validation_test")
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    config_mod = types.ModuleType("megatron.core.transformer.transformer_config")

    class FakeTransformerConfig:
        def __init__(self, experimental_attention_variant: str | None, context_parallel_size: int):
            self.experimental_attention_variant = experimental_attention_variant
            self.context_parallel_size = context_parallel_size
            self.post_init_seen_cp_sizes = []
            self.__post_init__()

        def __post_init__(self):
            self.post_init_seen_cp_sizes.append(self.context_parallel_size)
            if self.experimental_attention_variant == "gated_delta_net":
                assert self.context_parallel_size == 1, (
                    f"Gated delta net does not support context parallel for now, but got {self.context_parallel_size=}."
                )

    config_mod.TransformerConfig = FakeTransformerConfig
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer", transformer_mod)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.transformer_config", config_mod)

    monkeypatch.delenv("PLATOON_QWEN35_GDN_CP", raising=False)
    patches._patch_megatron_core_gdn_context_parallel_config_validation()
    with pytest.raises(AssertionError, match="Gated delta net does not support context parallel"):
        FakeTransformerConfig("gated_delta_net", 2)

    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP", "1")
    patches._patch_megatron_core_gdn_context_parallel_config_validation()
    cfg = FakeTransformerConfig("gated_delta_net", 2)
    assert cfg.context_parallel_size == 2
    assert cfg.post_init_seen_cp_sizes == [1]

    non_gdn_cfg = FakeTransformerConfig(None, 2)
    assert non_gdn_cfg.context_parallel_size == 2
    assert non_gdn_cfg.post_init_seen_cp_sizes == [2]


def test_qwen_gdn_cp_provider_uses_constructor_only_per_token_loss(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_provider_loss_test")
    megatron_mod = types.ModuleType("megatron")
    bridge_mod = types.ModuleType("megatron.bridge")
    models_mod = types.ModuleType("megatron.bridge.models")
    qwen_vl_mod = types.ModuleType("megatron.bridge.models.qwen_vl")
    provider_mod = types.ModuleType("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

    class FakeRouter(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            # TopKRouter caches this value instead of consulting config later.
            self.calculate_per_token_loss = config.calculate_per_token_loss

    class FakeModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.router = FakeRouter(self.config)

    class FakeProvider:
        def __init__(
            self,
            experimental_attention_variant: str | None,
            context_parallel_size: int,
            *,
            fail: bool = False,
        ):
            self.experimental_attention_variant = experimental_attention_variant
            self.context_parallel_size = context_parallel_size
            self.calculate_per_token_loss = False
            self.fail = fail
            self.loss_modes_seen_by_provide = []

        def provide(self):
            self.loss_modes_seen_by_provide.append(self.calculate_per_token_loss)
            if self.fail:
                raise RuntimeError("constructor failed")
            return FakeModel(self)

    class FakeMoEProvider(FakeProvider):
        pass

    provider_mod.Qwen35VLModelProvider = FakeProvider
    provider_mod.Qwen35VLMoEModelProvider = FakeMoEProvider
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge", bridge_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.models", models_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.qwen_vl", qwen_vl_mod)
    monkeypatch.setitem(sys.modules, "megatron.bridge.models.qwen_vl.qwen35_vl_provider", provider_mod)

    monkeypatch.delenv("PLATOON_QWEN35_GDN_CP", raising=False)
    patches._patch_megatron_bridge_qwen35_cp_constructor_loss_mode()
    unpatched_provider = FakeProvider("gated_delta_net", 2)
    unpatched_model = unpatched_provider.provide()
    assert unpatched_provider.loss_modes_seen_by_provide == [False]
    assert unpatched_provider.calculate_per_token_loss is False
    assert unpatched_model.config.calculate_per_token_loss is False
    assert unpatched_model.router.calculate_per_token_loss is False

    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP", "1")
    patches._patch_megatron_bridge_qwen35_cp_constructor_loss_mode()

    for provider_cls in (FakeProvider, FakeMoEProvider):
        provider = provider_cls("gated_delta_net", 2)
        model = provider.provide()

        # The Qwen constructor assertion sees per-token mode, but the provider,
        # model config, and modules that cache the flag all return to AReaL's
        # legacy normalized-loss mode before DDP wrapping.
        assert provider.loss_modes_seen_by_provide == [True]
        assert provider.calculate_per_token_loss is False
        assert model.config.calculate_per_token_loss is False
        assert model.router.config.calculate_per_token_loss is False
        assert model.router.calculate_per_token_loss is False

        dp_cp_world_size = 10
        ddp_gradient_scaling_factor = 1.0 if model.config.calculate_per_token_loss else 1.0 / dp_cp_world_size
        assert ddp_gradient_scaling_factor == pytest.approx(0.1)

    for attention_variant, cp_size in (("gated_delta_net", 1), (None, 2)):
        provider = FakeProvider(attention_variant, cp_size)
        model = provider.provide()
        assert provider.loss_modes_seen_by_provide == [False]
        assert provider.calculate_per_token_loss is False
        assert model.config.calculate_per_token_loss is False
        assert model.router.calculate_per_token_loss is False

    failed_provider = FakeProvider("gated_delta_net", 2, fail=True)
    with pytest.raises(RuntimeError, match="constructor failed"):
        failed_provider.provide()
    assert failed_provider.loss_modes_seen_by_provide == [True]
    assert failed_provider.calculate_per_token_loss is False


def test_qwen_gdn_cp_rejects_sequence_parallel_gather_after_input_projection(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_post_projection_sp_test")

    class FakeGroup:
        pass

    group = FakeGroup()
    module = types.SimpleNamespace(
        config=types.SimpleNamespace(sequence_parallel=True),
        pg_collection=types.SimpleNamespace(tp=types.SimpleNamespace(size=lambda: 2)),
    )
    monkeypatch.setattr(
        patches,
        "_get_gdn_cp_group_candidates",
        lambda _module, _packed_seq_params: [(group, 0, 2, "test.cp_group")],
    )
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    gather_calls = []
    tensor_parallel = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda *args, **kwargs: gather_calls.append((args, kwargs))
        or torch.zeros(8, 1, 4)
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.tensor_parallel = tensor_parallel
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    # cu_seqlens describes 16 global tokens, hence 8 tokens on each CP rank.
    # A 4-token qkvzba tensor is already feature-sharded by in_proj and cannot
    # safely be gathered along sequence: TP ranks own different head slices.
    with pytest.raises(ValueError) as exc_info:
        patches._select_gdn_cp_group_for_tensor(
            module,
            packed_seq_params=types.SimpleNamespace(),
            cu_seqlens=torch.tensor([0, 16], dtype=torch.int32),
            hidden_states=torch.zeros(4, 1, 4),
        )

    error = str(exc_info.value)
    assert "sequence length 8, observed 4" in error
    assert "feature-partitioned" in error
    assert gather_calls == []


def test_checkpoint_bucket_shape_patch_preserves_upstream_load_metadata(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_checkpoint_shape_test")

    class FakeShardedTensor:
        def __init__(self, key, global_shape, local_shape, global_offset):
            self.key = key
            self.global_shape = global_shape
            self.local_shape = local_shape
            self.global_offset = global_offset

    padded_shard = FakeShardedTensor(
        "optimizer.distributed.bucket.exp_avg",
        global_shape=(8,),
        local_shape=(4,),
        global_offset=(8,),
    )
    model_shard = FakeShardedTensor(
        "model.weight",
        global_shape=(8,),
        local_shape=(4,),
        global_offset=(8,),
    )

    class FakeOptimizer:
        def __init__(self):
            self.calls = []

        def sharded_state_dict(self, state_dict, **kwargs):
            self.calls.append((state_dict, kwargs))
            return {"optimizer": [padded_shard], "model": [model_shard]}

    upstream_metadata = {
        "distrib_optim_sharding_type": "dp_reshardable",
        "upstream_sentinel": True,
    }

    class FakeCheckpointManager:
        def __init__(self):
            self.optimizer = FakeOptimizer()

        def generate_state_dict(self, *, is_loading=False):
            return self.optimizer.sharded_state_dict(
                {"model": "state"},
                is_loading=is_loading,
                metadata=upstream_metadata,
            )

    areal_mod = types.ModuleType("areal")
    engine_mod = types.ModuleType("areal.engine")
    megatron_utils_mod = types.ModuleType("areal.engine.megatron_utils")
    checkpointer_mod = types.ModuleType("areal.engine.megatron_utils.checkpointer")
    checkpointer_mod.MegatronCheckpointManager = FakeCheckpointManager
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    dist_checkpointing_mod = types.ModuleType("megatron.core.dist_checkpointing")
    mapping_mod = types.ModuleType("megatron.core.dist_checkpointing.mapping")
    mapping_mod.ShardedTensor = FakeShardedTensor

    for name, module in {
        "areal": areal_mod,
        "areal.engine": engine_mod,
        "areal.engine.megatron_utils": megatron_utils_mod,
        "areal.engine.megatron_utils.checkpointer": checkpointer_mod,
        "megatron": megatron_mod,
        "megatron.core": core_mod,
        "megatron.core.dist_checkpointing": dist_checkpointing_mod,
        "megatron.core.dist_checkpointing.mapping": mapping_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dist, "is_available", lambda: False)

    patches._patch_megatron_checkpoint_optimizer_bucket_shapes()
    manager = FakeCheckpointManager()
    original_sharded_state_dict = manager.optimizer.sharded_state_dict
    state = manager.generate_state_dict(is_loading=True)

    assert manager.optimizer.calls == [
        (
            {"model": "state"},
            {"is_loading": True, "metadata": upstream_metadata},
        )
    ]
    assert state["optimizer"][0].global_shape == (12,)
    assert state["model"][0].global_shape == (8,)
    assert manager.optimizer.sharded_state_dict == original_sharded_state_dict


def test_openai_content_patch_leaves_tool_argument_decoding_to_upstream(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_openai_content_test")

    client_mod = types.ModuleType("areal.experimental.openai.client")

    def ensure_message_dict_list(_name, value):
        return value

    client_mod._ensure_message_dict_list = ensure_message_dict_list
    for name in ("areal", "areal.experimental", "areal.experimental.openai"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "areal.experimental.openai.client", client_mod)

    arguments = '{"query":"weather"}'
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello "},
                "world",
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "search", "arguments": arguments},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}],
        },
    ]

    patches._patch_areal_openai_message_content_flatten()
    normalized = client_mod._ensure_message_dict_list("messages", messages)

    assert normalized[0]["content"] == "hello world"
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == arguments
    assert isinstance(normalized[2]["content"], list)


def test_openai_non_thinking_hint_reaches_chat_template(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_non_thinking_test")

    proxy_mod = types.ModuleType(
        "areal.experimental.openai.proxy.proxy_rollout_server"
    )
    captured = []

    async def call_client_create(
        create_fn,
        request,
        session_id,
        extra_ignored_args=None,
        stream=False,
    ):
        captured.append(
            (create_fn, request, session_id, extra_ignored_args, stream)
        )
        return "response"

    proxy_mod._call_client_create = call_client_create
    for name in (
        "areal",
        "areal.experimental",
        "areal.experimental.openai",
        "areal.experimental.openai.proxy",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "areal.experimental.openai.proxy.proxy_rollout_server",
        proxy_mod,
    )

    patches._patch_areal_openai_reasoning_chat_template()
    patched = proxy_mod._call_client_create
    assert patched is not call_client_create
    patches._patch_areal_openai_reasoning_chat_template()
    assert proxy_mod._call_client_create is patched

    result = asyncio.run(
        patched(
            create_fn="create",
            request={
                "messages": [{"role": "user", "content": "summarize"}],
                "reasoning_effort": "none",
                "chat_template_kwargs": {"ignored": True},
            },
            session_id="session",
            extra_ignored_args=["ignored"],
            stream=True,
        )
    )

    assert result == "response"
    create_fn, request, session_id, ignored, stream = captured.pop()
    assert create_fn == "create"
    assert session_id == "session"
    assert ignored == ["ignored"]
    assert stream is True
    assert "reasoning_effort" not in request
    assert "chat_template_kwargs" not in request
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": False,
    }


def test_openai_reasoning_hint_enables_chat_template(monkeypatch):
    patches = _load_patches_module(
        "platoon_areal_patches_reasoning_enabled_test"
    )

    proxy_mod = types.ModuleType(
        "areal.experimental.openai.proxy.proxy_rollout_server"
    )
    captured = []

    async def call_client_create(**kwargs):
        captured.append(kwargs)
        return "response"

    proxy_mod._call_client_create = call_client_create
    for name in (
        "areal",
        "areal.experimental",
        "areal.experimental.openai",
        "areal.experimental.openai.proxy",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "areal.experimental.openai.proxy.proxy_rollout_server",
        proxy_mod,
    )

    patches._patch_areal_openai_reasoning_chat_template()
    request = {"reasoning_effort": "high"}
    asyncio.run(
        proxy_mod._call_client_create(
            create_fn="create",
            request=request,
            session_id="session",
        )
    )

    forwarded = captured[0]["request"]
    assert forwarded is not request
    assert "reasoning_effort" not in forwarded
    assert forwarded["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": False,
    }


def test_proxy_fork_command_uses_platoon_entrypoint(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_proxy_fork_test")

    class LocalScheduler:
        def fork_workers(self, role, target_role, command=None):
            return role, target_role, command

    class SlurmScheduler:
        def fork_workers(self, role, target_role, command=None):
            return role, target_role, command

    local_mod = types.ModuleType("areal.infra.scheduler.local")
    local_mod.LocalScheduler = LocalScheduler
    slurm_mod = types.ModuleType("areal.infra.scheduler.slurm")
    slurm_mod.SlurmScheduler = SlurmScheduler
    for name in (
        "areal",
        "areal.infra",
        "areal.infra.scheduler",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "areal.infra.scheduler.local", local_mod)
    monkeypatch.setitem(sys.modules, "areal.infra.scheduler.slurm", slurm_mod)

    patches._patch_areal_proxy_rollout_fork_command()
    local_patched = LocalScheduler.fork_workers
    slurm_patched = SlurmScheduler.fork_workers
    patches._patch_areal_proxy_rollout_fork_command()
    assert LocalScheduler.fork_workers is local_patched
    assert SlurmScheduler.fork_workers is slurm_patched

    upstream = "areal.experimental.openai.proxy.proxy_rollout_server"
    assert LocalScheduler().fork_workers("proxy", "rollout", upstream) == (
        "proxy",
        "rollout",
        "platoon.areal_proxy_rollout",
    )
    assert SlurmScheduler().fork_workers(
        role="proxy",
        target_role="rollout",
        command=upstream,
    ) == ("proxy", "rollout", "platoon.areal_proxy_rollout")
    assert LocalScheduler().fork_workers("other", "rollout", "custom.module") == (
        "other",
        "rollout",
        "custom.module",
    )


@pytest.mark.parametrize(
    "config_name",
    [
        "toolathlon_openhands_areal_prealloc_8node.yaml",
        "toolathlon_openhands_areal_prealloc_16node.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-bs16.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-ptc-recursive.yaml",
    ],
)
def test_qwen_megatron_configs_explicitly_disable_mtp(config_name):
    config_path = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal" / config_name
    config = yaml.safe_load(config_path.read_text())

    assert config["actor"]["megatron"]["enable_mtp"] is False
