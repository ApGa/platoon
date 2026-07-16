from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"
SGLANG_COMPAT_PATH = REPO_ROOT / "platoon/sglang_scheduler_compat.py"


def _load_patches_module():
    spec = importlib.util.spec_from_file_location("platoon_areal_route_patches", PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sglang_compat_module(monkeypatch):
    scheduler_module = ModuleType("areal.v2.inference_service.sglang.scheduler")

    def original_scheduler(*args, **kwargs):
        return (args, kwargs)

    scheduler_module.areal_run_scheduler_process = original_scheduler
    monkeypatch.setitem(sys.modules, "areal", ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.v2", ModuleType("areal.v2"))
    monkeypatch.setitem(sys.modules, "areal.v2.inference_service", ModuleType("areal.v2.inference_service"))
    sglang_package = ModuleType("areal.v2.inference_service.sglang")
    sglang_package.scheduler = scheduler_module
    monkeypatch.setitem(sys.modules, "areal.v2.inference_service.sglang", sglang_package)
    monkeypatch.setitem(sys.modules, "areal.v2.inference_service.sglang.scheduler", scheduler_module)

    spec = importlib.util.spec_from_file_location("platoon_sglang_scheduler_compat", SGLANG_COMPAT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("values", "expected_dtype"),
    [
        ([0, 255], torch.uint8),
        ([-1, 255], torch.int16),
        ([0, 256], torch.int16),
        ([-40_000, 40_000], torch.int32),
        ([-3_000_000_000, 3_000_000_000], torch.int64),
    ],
)
def test_routed_expert_compaction_uses_smallest_lossless_integer_dtype(values, expected_dtype):
    patches = _load_patches_module()
    source = torch.tensor(values, dtype=torch.int64)

    compact = patches._compact_routed_expert_ids(source)

    assert compact.dtype == expected_dtype
    assert torch.equal(compact.to(torch.int64), source)


def test_routed_expert_compaction_rejects_non_integral_values():
    patches = _load_patches_module()

    with pytest.raises(ValueError, match="finite integers"):
        patches._compact_routed_expert_ids(torch.tensor([0.0, 1.5]))


def test_sglang_route_capture_uses_prefill_token_capacity_when_chunking_is_disabled(monkeypatch):
    compat = _load_sglang_compat_module(monkeypatch)
    calls = []

    class FakeCapturer:
        @staticmethod
        def create(
            enable,
            model_config,
            num_fused_shared_experts,
            num_tokens,
            max_running_requests,
            device,
        ):
            calls.append(
                {
                    "enable": enable,
                    "model_config": model_config,
                    "num_fused_shared_experts": num_fused_shared_experts,
                    "num_tokens": num_tokens,
                    "max_running_requests": max_running_requests,
                    "device": device,
                }
            )
            return "capturer"

    capturer_module = SimpleNamespace(RoutedExpertsCapturer=FakeCapturer)
    server_args = SimpleNamespace(chunked_prefill_size=-1, max_prefill_tokens=32_768, dp_size=1)

    assert compat.install_routed_experts_capture_capacity_patch(
        capturer_module, lambda: server_args
    )
    assert not compat.install_routed_experts_capture_capacity_patch(
        capturer_module, lambda: server_args
    )
    result = FakeCapturer.create(True, object(), 0, 100_000, 3_335, "cuda")

    assert result == "capturer"
    assert calls[-1]["max_running_requests"] == 32_768


def test_sglang_route_capture_capacity_patch_is_noop_when_capture_is_disabled(monkeypatch):
    compat = _load_sglang_compat_module(monkeypatch)
    calls = []

    class FakeCapturer:
        @staticmethod
        def create(
            enable,
            model_config,
            num_fused_shared_experts,
            num_tokens,
            max_running_requests,
            device,
        ):
            calls.append(max_running_requests)
            return "noop"

    capturer_module = SimpleNamespace(RoutedExpertsCapturer=FakeCapturer)

    def unexpected_server_args_lookup():
        raise AssertionError("disabled capture must not inspect or change SGLang server args")

    compat.install_routed_experts_capture_capacity_patch(
        capturer_module, unexpected_server_args_lookup
    )
    assert FakeCapturer.create(False, object(), 0, 100_000, 3_335, "cuda") == "noop"
    assert calls == [3_335]


def test_sglang_route_capture_capacity_patch_preserves_positive_chunking(monkeypatch):
    compat = _load_sglang_compat_module(monkeypatch)
    calls = []

    class FakeCapturer:
        @staticmethod
        def create(
            enable,
            model_config,
            num_fused_shared_experts,
            num_tokens,
            max_running_requests,
            device,
        ):
            calls.append(max_running_requests)
            return "capturer"

    capturer_module = SimpleNamespace(RoutedExpertsCapturer=FakeCapturer)
    compat.install_routed_experts_capture_capacity_patch(
        capturer_module,
        lambda: SimpleNamespace(chunked_prefill_size=8_192, max_prefill_tokens=32_768, dp_size=1),
    )
    FakeCapturer.create(True, object(), 0, 100_000, 3_335, "cuda")
    assert calls == [3_335]


def test_areal_sglang_launcher_redirect_is_r3_only_and_idempotent():
    patches = _load_patches_module()
    baseline_command = ["python", "-m", "areal.v2.inference_service.sglang.launch_server", "--model", "qwen"]

    class FakeSGLangConfig:
        @staticmethod
        def build_cmd_from_args(args):
            return baseline_command

    assert patches._patch_areal_sglang_routed_experts_launcher(FakeSGLangConfig)
    assert not patches._patch_areal_sglang_routed_experts_launcher(FakeSGLangConfig)
    assert FakeSGLangConfig.build_cmd_from_args({"enable_return_routed_experts": False}) is baseline_command
    assert FakeSGLangConfig.build_cmd_from_args({"enable_return_routed_experts": True}) == [
        "python",
        "-m",
        "platoon.sglang_server",
        "--model",
        "qwen",
    ]


def test_interrupted_route_accumulator_keeps_prefix_and_appends_only_new_suffix():
    patches = _load_patches_module()
    accumulator = patches._RoutedExpertsPrefixAccumulator()
    first = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.int32)
    # The overlapping prefix deliberately differs. Routes captured by the
    # original behavior request must win; only rows 3+ come from the resume.
    resumed = np.array(
        [[101, 101], [102, 102], [103, 103], [4, 4], [5, 5]],
        dtype=np.int32,
    )

    accumulator.append(first)
    accumulator.append(resumed)

    assert np.array_equal(np.concatenate(accumulator), np.concatenate([first, resumed[3:]]))
    assert not np.shares_memory(accumulator[1], resumed)


def test_interrupted_route_accumulator_rejects_shorter_or_changed_width_resume():
    patches = _load_patches_module()
    accumulator = patches._RoutedExpertsPrefixAccumulator()
    accumulator.append(np.zeros((3, 4), dtype=np.int32))

    with pytest.raises(RuntimeError, match="shorter than the already captured prefix"):
        accumulator.append(np.zeros((2, 4), dtype=np.int32))
    with pytest.raises(RuntimeError, match="width changed"):
        accumulator.append(np.zeros((4, 5), dtype=np.int32))


def test_remote_engine_patch_stitches_full_prefix_matrices():
    patches = _load_patches_module()

    class FakeRemoteInfEngine:
        async def agenerate(self, req):
            accumulated_routed_experts: list[np.ndarray] = []
            for gen_result in req:
                if gen_result.routed_experts is not None:
                    accumulated_routed_experts.append(gen_result.routed_experts)
            return np.concatenate(accumulated_routed_experts) if accumulated_routed_experts else None

    fake_module = ModuleType("fake_remote_inf_engine")
    fake_module.RemoteInfEngine = FakeRemoteInfEngine
    fake_module.np = np

    assert patches._patch_remote_inf_engine_routed_expert_stitching(fake_module)
    first = SimpleNamespace(routed_experts=np.array([[1], [2], [3]], dtype=np.int32))
    resumed = SimpleNamespace(routed_experts=np.array([[9], [9], [9], [4], [5]], dtype=np.int32))
    result = asyncio.run(FakeRemoteInfEngine().agenerate([first, resumed]))

    assert np.array_equal(result, np.array([[1], [2], [3], [4], [5]], dtype=np.int32))
    assert patches._patch_remote_inf_engine_routed_expert_stitching(fake_module)


def test_openai_proxy_patch_transports_compact_routes_and_explicit_validity(monkeypatch):
    patches = _load_patches_module()

    class FakeInteraction:
        def __init__(self, routes):
            self.model_response = SimpleNamespace(routed_experts=routes)

        def to_tensor_dict(self):
            return {"input_ids": torch.tensor([[10, 11, 12]])}

    areal_module = ModuleType("areal")
    experimental_module = ModuleType("areal.experimental")
    openai_module = ModuleType("areal.experimental.openai")
    types_module = ModuleType("areal.experimental.openai.types")
    types_module.InteractionWithTokenLogpReward = FakeInteraction
    monkeypatch.setitem(sys.modules, "areal", areal_module)
    monkeypatch.setitem(sys.modules, "areal.experimental", experimental_module)
    monkeypatch.setitem(sys.modules, "areal.experimental.openai", openai_module)
    monkeypatch.setitem(sys.modules, "areal.experimental.openai.types", types_module)

    patches._patch_areal_openai_routed_experts_transport()
    raw = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [6, 255]],
        ],
        dtype=torch.int32,
    )
    interaction = FakeInteraction(raw)
    result = interaction.to_tensor_dict()

    assert result["routed_experts"].shape == (1, 2, 4)
    assert result["routed_experts"].dtype == torch.uint8
    assert torch.equal(result["routed_experts"].squeeze(0), raw.flatten(start_dim=1).to(torch.uint8))
    assert torch.equal(result["routed_experts_valid"], torch.ones((1, 2), dtype=torch.bool))
    assert interaction.model_response.routed_experts.dtype == np.uint8


def test_openai_proxy_patch_is_inert_when_capture_is_disabled(monkeypatch):
    patches = _load_patches_module()

    class FakeInteraction:
        model_response = SimpleNamespace(routed_experts=None)

        def to_tensor_dict(self):
            return {"input_ids": torch.tensor([[10, 11]])}

    types_module = ModuleType("areal.experimental.openai.types")
    types_module.InteractionWithTokenLogpReward = FakeInteraction
    monkeypatch.setitem(sys.modules, "areal.experimental.openai.types", types_module)

    patches._patch_areal_openai_routed_experts_transport()
    result = FakeInteraction().to_tensor_dict()

    assert set(result) == {"input_ids"}


def _fake_upstream_split_and_unpad(
    result,
    n_trajs,
    traj_group_sizes=1,
    traj_seqlens=None,
):
    if isinstance(traj_group_sizes, int):
        traj_group_sizes = [traj_group_sizes] * n_trajs
    if traj_seqlens is None:
        attention_splits = result["attention_mask"].split(traj_group_sizes, dim=0)
        traj_seqlens = [int(mask.sum(-1).max().item()) for mask in attention_splits]
    output = [{} for _ in range(n_trajs)]
    total = sum(traj_group_sizes)
    for key, value in result.items():
        if torch.is_tensor(value) and value.shape[0] == total:
            splits = list(value.split(traj_group_sizes, dim=0))
            for index, (split, seq_len) in enumerate(zip(splits, traj_seqlens, strict=True)):
                # Reproduce AReaL's bug: generic tensors are trimmed on their
                # last dimension, which is K rather than S for [B,S,L,K].
                if split.ndim >= 2 and split.shape[-1] > seq_len:
                    split = split[..., :seq_len]
                output[index][key] = split
        else:
            for item in output:
                item[key] = copy.deepcopy(value)
    return output


def _pad_sequence_dim(tensor: torch.Tensor, width: int) -> torch.Tensor:
    output = tensor.new_zeros((tensor.shape[0], width, *tensor.shape[2:]))
    output[:, : tensor.shape[1]] = tensor
    return output


def test_route_aware_unpadding_survives_heterogeneous_dp_split_and_reconcat():
    patches = _load_patches_module()
    data_module = ModuleType("fake_areal_data")
    data_module.split_and_unpad_tensor = _fake_upstream_split_and_unpad
    dist_module = ModuleType("fake_dist_rollout")
    dist_module.split_and_unpad_tensor = _fake_upstream_split_and_unpad

    assert patches._patch_areal_routed_experts_unpadding(data_module, dist_module) == {
        "data": True,
        "dist_rollout_alias": True,
    }
    assert dist_module.split_and_unpad_tensor is data_module.split_and_unpad_tensor
    assert patches._patch_areal_routed_experts_unpadding(data_module, dist_module) == {
        "data": False,
        "dist_rollout_alias": False,
    }

    routes = torch.arange(3 * 6 * 3 * 8, dtype=torch.uint8).reshape(3, 6, 3, 8)
    attention = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    valid = attention.clone()
    valid[:, -1] = False
    batch = {
        "attention_mask": attention,
        "routed_experts": routes,
        "routed_experts_valid": valid,
        "metadata": "preserved",
    }

    split = dist_module.split_and_unpad_tensor(batch, 2, [2, 1])

    assert split[0]["routed_experts"].shape == (2, 5, 3, 8)
    # Regression: S=2 is smaller than K=8. The upstream last-dimension
    # unpad would irreversibly truncate K to two without stashing routes first.
    assert split[1]["routed_experts"].shape == (1, 2, 3, 8)
    torch.testing.assert_close(split[0]["routed_experts"], routes[:2, :5])
    torch.testing.assert_close(split[1]["routed_experts"], routes[2:, :2])
    assert split[0]["routed_experts_valid"].shape == (2, 5)
    assert split[1]["routed_experts_valid"].shape == (1, 2)

    # Simulate DP redistribution selecting/reordering trajectory groups, then
    # concat+split as batched_call does. Trailing [L,K] data must survive both.
    width = 5
    recombined = {
        "attention_mask": torch.cat(
            (
                _pad_sequence_dim(split[1]["attention_mask"], width),
                _pad_sequence_dim(split[0]["attention_mask"], width),
            ),
            dim=0,
        ),
        "routed_experts": torch.cat(
            (
                _pad_sequence_dim(split[1]["routed_experts"], width),
                _pad_sequence_dim(split[0]["routed_experts"], width),
            ),
            dim=0,
        ),
        "routed_experts_valid": torch.cat(
            (
                _pad_sequence_dim(split[1]["routed_experts_valid"], width),
                _pad_sequence_dim(split[0]["routed_experts_valid"], width),
            ),
            dim=0,
        ),
    }
    resplit = data_module.split_and_unpad_tensor(recombined, 2, [1, 2], [2, 5])

    torch.testing.assert_close(resplit[0]["routed_experts"], routes[2:, :2])
    torch.testing.assert_close(resplit[1]["routed_experts"], routes[:2, :5])
    assert resplit[0]["routed_experts"].shape[-2:] == (3, 8)
    assert resplit[1]["routed_experts"].shape[-2:] == (3, 8)


def test_route_aware_unpadding_is_noop_for_baseline_batches():
    patches = _load_patches_module()
    data_module = ModuleType("fake_areal_data_baseline")
    data_module.split_and_unpad_tensor = _fake_upstream_split_and_unpad
    patches._patch_areal_routed_experts_unpadding(data_module, ModuleType("unused_dist"))
    batch = {
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        "input_ids": torch.tensor([[10, 11, 0], [20, 0, 0]]),
    }

    result = data_module.split_and_unpad_tensor(batch, 2, [1, 1])

    assert set(result[0]) == {"attention_mask", "input_ids"}
    torch.testing.assert_close(result[0]["input_ids"], torch.tensor([[10, 11]]))
    torch.testing.assert_close(result[1]["input_ids"], torch.tensor([[20]]))


def test_areal_proxy_serialization_round_trip_preserves_uint8_routes_and_mask():
    proxy_server = pytest.importorskip("areal.experimental.openai.proxy.server")
    interaction_types = pytest.importorskip("areal.experimental.openai.types")

    interaction = interaction_types.InteractionWithTokenLogpReward()
    interaction._cache = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "routed_experts": torch.arange(8, dtype=torch.uint8).reshape(1, 2, 4),
        "routed_experts_valid": torch.tensor([[True, True]]),
    }
    interaction.reward = 0.5
    interaction.interaction_id = "completion-a"

    payload = proxy_server.serialize_interactions({"completion-a": interaction})
    restored = proxy_server.deserialize_interactions(payload)["completion-a"].to_tensor_dict()

    assert restored["routed_experts"].dtype == torch.uint8
    assert torch.equal(restored["routed_experts"], interaction._cache["routed_experts"])
    assert torch.equal(restored["routed_experts_valid"], interaction._cache["routed_experts_valid"])
