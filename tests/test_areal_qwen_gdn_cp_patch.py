from __future__ import annotations

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
        roundtrip = patches._packed_shard_to_zigzag(packed_shard, cu_seqlens, dist.group.WORLD, rank, world_size)

        loss = roundtrip.sum()
        loss.backward()
        result = {
            "packed_ok": torch.equal(packed_shard, expected_packed_shard),
            "roundtrip_ok": torch.equal(roundtrip, zigzag),
            "grad_ok": torch.equal(zigzag.grad, torch.ones_like(zigzag)),
        }
        Path(result_dir, f"rank_{rank}.json").write_text(json.dumps(result))
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
        assert result == {"packed_ok": True, "roundtrip_ok": True, "grad_ok": True}


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
                    "Gated delta net does not support context parallel for now,"
                    f" but got {self.context_parallel_size=}."
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


def test_qwen_gdn_cp_provider_enables_per_token_loss(monkeypatch):
    patches = _load_patches_module("platoon_areal_patches_provider_loss_test")
    megatron_mod = types.ModuleType("megatron")
    bridge_mod = types.ModuleType("megatron.bridge")
    models_mod = types.ModuleType("megatron.bridge.models")
    qwen_vl_mod = types.ModuleType("megatron.bridge.models.qwen_vl")
    provider_mod = types.ModuleType("megatron.bridge.models.qwen_vl.qwen35_vl_provider")

    class FakeProvider:
        def __init__(self, experimental_attention_variant: str | None, context_parallel_size: int):
            self.experimental_attention_variant = experimental_attention_variant
            self.context_parallel_size = context_parallel_size
            self.calculate_per_token_loss = False

        def provide(self):
            return self.calculate_per_token_loss

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
    patches._patch_megatron_bridge_qwen35_cp_per_token_loss()
    assert FakeProvider("gated_delta_net", 2).provide() is False

    monkeypatch.setenv("PLATOON_QWEN35_GDN_CP", "1")
    patches._patch_megatron_bridge_qwen35_cp_per_token_loss()
    assert FakeProvider("gated_delta_net", 2).provide() is True
    assert FakeMoEProvider("gated_delta_net", 2).provide() is True
    assert FakeProvider("gated_delta_net", 1).provide() is False
    assert FakeProvider(None, 2).provide() is False


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("PLATOON_RUN_GDN_CP_CUDA_TESTS") != "1",
    reason="GDN forward/backward CP correctness requires CUDA and an explicit opt-in.",
)
def test_qwen_gdn_cp_cuda_correctness_placeholder():
    pytest.skip("Run the full Qwen GDN CP smoke on allocated GPUs with PLATOON_RUN_GDN_CP_CUDA_TESTS=1.")
