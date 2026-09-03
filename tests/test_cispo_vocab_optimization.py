"""Focused tests for CISPO's post-logits and pre-logits Megatron paths."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_path(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    fused = _load_path(
        "platoon.train.areal.fused_linear_logprobs",
        "platoon/train/areal/fused_linear_logprobs.py",
    )
    loss_stub = types.ModuleType("platoon.train.areal.loss_functions")
    loss_stub.supports_logprobs_only_training = lambda fn: bool(getattr(fn, "supports_logprobs_only_training", False))
    sys.modules[loss_stub.__name__] = loss_stub
    optimization = _load_path(
        "platoon_cispo_vocab_optimization_test",
        "platoon/train/areal/cispo_vocab_optimization.py",
    )
    return fused, optimization


class _Mpu:
    @staticmethod
    def get_tensor_model_parallel_world_size():
        return 1

    @staticmethod
    def get_tensor_model_parallel_group():
        raise AssertionError("TP group is not queried at world size one")


def _engine_module(**overrides):
    values = {
        "mpu": _Mpu,
        "gather_logprobs": lambda output, labels, **_: (
            F.log_softmax(output.float(), -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ),
        "reassemble_cp_packed_logprobs": lambda values, _: values + 10,
        "unpad_logits": lambda values, padding_length, *_: values[:-padding_length],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _engine():
    return SimpleNamespace(
        config=SimpleNamespace(is_critic=False, temperature=1.0),
        enable_tree_training=False,
    )


def _recording_loss(records):
    def loss(logprobs, entropy, inputs, **diagnostics):
        records.append((logprobs.detach().clone(), entropy, inputs, diagnostics))
        return logprobs.sum()

    loss.supports_logprobs_only_training = True
    return loss


def test_post_logits_path_omits_entropy_and_vocab_diagnostics_but_preserves_cp_loss():
    _, optimization = _load_modules()
    records = []
    output = torch.randn(4, 7, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 4])
    inputs = {
        "input_ids": torch.tensor([9, 9, 9]),
        "_cp_local_labels": labels,
        "_cp_padded_cu_seqlens": torch.tensor([0, 4]),
        "_cp_padding_length": 1,
        "_cp_old_cu_seqlens": torch.tensor([0, 3]),
    }

    result = optimization.compute_logprobs_only_megatron_loss(
        _engine(),
        output,
        inputs,
        _recording_loss(records),
        lambda _: torch.tensor(2.0),
        torch.tensor(4.0),
        loss_multiplier=3.0,
        megatron_engine_module=_engine_module(),
    )
    expected_local = F.log_softmax(output.float(), -1).gather(-1, labels[:, None]).squeeze(-1)
    expected_logprobs = (expected_local + 10)[:-1]

    torch.testing.assert_close(result, expected_logprobs.sum() * 1.5)
    assert len(records) == 1
    torch.testing.assert_close(records[0][0], expected_logprobs)
    assert records[0][1] is None
    assert not any(key.startswith("_cp_") for key in records[0][2])
    assert set(records[0][3].values()) == {None}


def test_pre_logits_path_matches_dense_selected_logprobs_and_gradient():
    fused, optimization = _load_modules()
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(6, 4, generator=generator, requires_grad=True)
    weight = torch.randn(9, 4, generator=generator, requires_grad=True)
    input_ids = torch.tensor([2, 4, 1, 8, 3, 6])
    labels = torch.roll(input_ids, shifts=-1)
    records = []
    state = fused.FusedLinearLogprobsHeadState(None, lambda: None, None, active_weight=weight)

    result = optimization.compute_fused_linear_megatron_cispo_loss(
        _engine(),
        hidden,
        {"input_ids": input_ids},
        _recording_loss(records),
        lambda _: torch.tensor(1.0),
        torch.tensor(1.0),
        state,
        chunk_size=2,
        megatron_engine_module=_engine_module(),
    )
    result.backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_logprobs = F.log_softmax(F.linear(reference_hidden, reference_weight), -1)
    reference_logprobs = reference_logprobs.gather(-1, labels[:, None]).squeeze(-1)
    reference_logprobs.sum().backward()

    torch.testing.assert_close(records[0][0], reference_logprobs)
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad, rtol=2e-6, atol=2e-6)


def test_pre_logits_zero_weight_still_runs_hidden_and_output_weight_hooks():
    fused, optimization = _load_modules()
    hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    state = fused.FusedLinearLogprobsHeadState(None, lambda: None, None, active_weight=weight)
    loss = _recording_loss([])

    result = optimization.compute_fused_linear_megatron_cispo_loss(
        _engine(),
        hidden,
        {"input_ids": torch.tensor([1, 2, 3])},
        loss,
        lambda _: torch.tensor(0.0),
        torch.tensor(1.0),
        state,
        megatron_engine_module=_engine_module(),
    )
    result.backward()

    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) == 0
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0
