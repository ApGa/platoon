from __future__ import annotations

import importlib.util
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


def _load_patches_module():
    spec = importlib.util.spec_from_file_location("platoon_areal_stats_patches", PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ReduceType(Enum):
    SCALAR = auto()
    SUM = auto()


class _FakeScalarTensor:
    def __init__(self, value):
        self.value = float(value)

    def __truediv__(self, other):
        if other.value == 0:
            return _FakeScalarTensor(float("nan"))
        return _FakeScalarTensor(self.value / other.value)

    def __float__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class _FakeTorch:
    float32 = object()

    def __init__(self):
        self.devices = []

    def tensor(self, value, *, dtype, device):
        assert dtype is self.float32
        self.devices.append(device)
        return _FakeScalarTensor(value)


def _fake_stats_module():
    class FakeDistributedStatsTracker:
        def __init__(self):
            self.reduce_types = {}
            self.reduce_groups = {}
            self.stats = {}
            self.original_calls = []

        def _effective_reduce_group(self, key, default_reduce_group):
            return self.reduce_groups.get(key, default_reduce_group)

        def _aggregate(self, key, reduce_group):
            self.original_calls.append((key, reduce_group))
            return {"delegated": True}

    return SimpleNamespace(
        DistributedStatsTracker=FakeDistributedStatsTracker,
        ReduceType=_ReduceType,
        torch=_FakeTorch(),
    )


def test_local_scalar_stats_are_aggregated_on_cpu():
    patches = _load_patches_module()
    stats_module = _fake_stats_module()

    assert patches._patch_areal_local_scalar_stats_export(stats_module)
    tracker = stats_module.DistributedStatsTracker()
    tracker.stats["latency"] = [1.0, 2.0, 3.0]

    assert tracker._aggregate("latency", None) == {
        "latency": 2.0,
        "latency__count": 3,
    }
    assert tracker.original_calls == []
    assert stats_module.torch.devices == ["cpu", "cpu"]


def test_stats_patch_delegates_distributed_and_non_scalar_reductions():
    patches = _load_patches_module()
    stats_module = _fake_stats_module()
    patches._patch_areal_local_scalar_stats_export(stats_module)
    tracker = stats_module.DistributedStatsTracker()
    tracker.stats["distributed"] = [1.0]
    tracker.stats["summed"] = [1.0]
    tracker.stats["default_group"] = [1.0]
    default_group = object()
    tracker.reduce_groups["distributed"] = object()
    tracker.reduce_types["summed"] = _ReduceType.SUM

    assert tracker._aggregate("distributed", None) == {"delegated": True}
    assert tracker._aggregate("summed", None) == {"delegated": True}
    assert tracker._aggregate("default_group", default_group) == {"delegated": True}
    assert tracker.original_calls == [
        ("distributed", None),
        ("summed", None),
        ("default_group", default_group),
    ]


def test_stats_patch_is_idempotent_and_preserves_empty_scalar_behavior():
    patches = _load_patches_module()
    stats_module = _fake_stats_module()

    assert patches._patch_areal_local_scalar_stats_export(stats_module)
    patched = stats_module.DistributedStatsTracker._aggregate
    assert not patches._patch_areal_local_scalar_stats_export(stats_module)
    assert stats_module.DistributedStatsTracker._aggregate is patched

    tracker = stats_module.DistributedStatsTracker()
    tracker.stats["empty"] = []
    result = tracker._aggregate("empty", None)
    assert result["empty"] != result["empty"]
    assert result["empty__count"] == 0
