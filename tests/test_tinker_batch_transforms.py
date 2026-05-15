"""Focused regression tests for Tinker trainer-side batch transforms."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class DummyTensorData:
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def to_torch(self) -> torch.Tensor:
        return self._tensor

    @classmethod
    def from_torch(cls, tensor: torch.Tensor) -> "DummyTensorData":
        return cls(tensor)


class DummyDatum:
    def __init__(self, loss_fn_inputs: dict):
        self.loss_fn_inputs = loss_fn_inputs
        self.model_input = {}


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_tinker_batch_transforms_module():
    tinker_mod = types.ModuleType("tinker")
    tinker_mod.Datum = DummyDatum
    tinker_mod.TensorData = DummyTensorData
    sys.modules["tinker"] = tinker_mod
    return _load_module(
        "platoon_tinker_batch_transforms_test",
        REPO_ROOT / "platoon/train/tinker/batch_transforms.py",
    )


def test_tinker_depth_transform_matches_existing_microbatch_formula():
    batch_transforms = _load_tinker_batch_transforms_module()
    transform = batch_transforms.DepthLevelWeightingTransform()
    context = batch_transforms.BatchTransformContext(
        config=SimpleNamespace(train=SimpleNamespace(workflow_config=SimpleNamespace(depth_level_weighting=True))),
        train_step=5,
        minibatch_num=1,
        microbatch_num=2,
    )

    datums = [
        DummyDatum(
            {
                "traj_depth": DummyTensorData(torch.tensor([0])),
                "traj_start": DummyTensorData(torch.tensor([1.0])),
                "mask": DummyTensorData(torch.tensor([1.0, 1.0])),
                "advantages": DummyTensorData(torch.tensor([2.0, 2.0])),
            }
        ),
        DummyDatum(
            {
                "traj_depth": DummyTensorData(torch.tensor([1])),
                "traj_start": DummyTensorData(torch.tensor([1.0])),
                "mask": DummyTensorData(torch.tensor([1.0, 0.0, 1.0])),
                "advantages": DummyTensorData(torch.tensor([3.0, 3.0, 3.0])),
            }
        ),
        DummyDatum(
            {
                "traj_depth": DummyTensorData(torch.tensor([1])),
                "traj_start": DummyTensorData(torch.tensor([0.0])),
                "mask": DummyTensorData(torch.tensor([1.0])),
                "advantages": DummyTensorData(torch.tensor([4.0])),
            }
        ),
    ]

    transformed = transform(datums, context)
    assert transformed is datums

    # Depth 0: traj_count=1, action_tokens=2 => raw_weight=1
    # Depth 1: traj_count=1, action_tokens=3 => raw_weight=1
    # Normalization keeps effective total action-token mass unchanged, so weights stay 1.
    assert torch.equal(datums[0].loss_fn_inputs["advantages"].to_torch(), torch.tensor([2.0, 2.0]))
    assert torch.equal(datums[1].loss_fn_inputs["advantages"].to_torch(), torch.tensor([3.0, 3.0, 3.0]))
    assert torch.equal(datums[2].loss_fn_inputs["advantages"].to_torch(), torch.tensor([4.0]))


def test_tinker_batch_transform_runner_passes_context_and_allows_custom_transform():
    batch_transforms = _load_tinker_batch_transforms_module()

    seen = {}

    def custom_transform(datums, context):
        seen["count"] = len(datums)
        seen["train_step"] = context.train_step
        datums[0].loss_fn_inputs["advantages"] = DummyTensorData(
            datums[0].loss_fn_inputs["advantages"].to_torch() + 1.0
        )
        return datums

    datums = [
        DummyDatum(
            {
                "mask": DummyTensorData(torch.tensor([1.0])),
                "advantages": DummyTensorData(torch.tensor([5.0])),
            }
        )
    ]
    context = batch_transforms.BatchTransformContext(
        config=SimpleNamespace(train=SimpleNamespace(workflow_config=SimpleNamespace(depth_level_weighting=False))),
        train_step=11,
        minibatch_num=0,
        microbatch_num=0,
    )

    transformed = batch_transforms.run_batch_transforms(datums, [custom_transform], context)
    assert transformed is datums
    assert seen == {"count": 1, "train_step": 11}
    assert torch.equal(datums[0].loss_fn_inputs["advantages"].to_torch(), torch.tensor([6.0]))
