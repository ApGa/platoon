"""CPU tests for the pre-logits CISPO vocabulary optimization."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "platoon/train/areal/fused_linear_logprobs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("platoon_fused_linear_logprobs_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
def test_chunked_fused_linear_matches_dense_forward_and_arbitrary_weighted_backward(
    temperature,
):
    fused = _load_module()
    generator = torch.Generator().manual_seed(123)
    hidden = torch.randn(2, 5, 7, generator=generator, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(11, 7, generator=generator, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, 11, (2, 5), generator=generator)
    upstream = torch.randn(2, 5, generator=generator)
    observed_shapes = []

    actual = fused.vocab_parallel_fused_linear_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        chunk_size=3,
        _chunk_observer=lambda logits: observed_shapes.append(tuple(logits.shape)),
    )
    (actual * upstream).sum().backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = (
        F.log_softmax(
            F.linear(reference_hidden, reference_weight).float() / temperature,
            dim=-1,
        )
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    (reference * upstream).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_hidden_grad, reference_hidden.grad, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_weight_grad, reference_weight.grad, rtol=2e-6, atol=2e-6)
    # Four forward chunks only; backward intentionally recomputes without the
    # observer. No projection spans all ten tokens.
    assert observed_shapes == [(3, 11), (3, 11), (3, 11), (1, 11)]


def test_fused_linear_validates_shapes_dtype_and_scalars():
    fused = _load_module()
    hidden = torch.randn(4, 3)
    weight = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 2, 3])

    with pytest.raises(ValueError, match="leading dimensions"):
        fused.vocab_parallel_fused_linear_logprobs(hidden[:3], weight, labels)
    with pytest.raises(TypeError, match="torch.long"):
        fused.vocab_parallel_fused_linear_logprobs(hidden, weight, labels.int())
    with pytest.raises(ValueError, match="chunk_size"):
        fused.vocab_parallel_fused_linear_logprobs(hidden, weight, labels, chunk_size=0)
    with pytest.raises(ValueError, match="temperature"):
        fused.vocab_parallel_fused_linear_logprobs(hidden, weight, labels, temperature=0.0)


def test_bfloat16_projection_preserves_fp32_softmax_boundary():
    fused = _load_module()
    generator = torch.Generator().manual_seed(44)
    hidden = torch.randn(9, 6, generator=generator, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(13, 6, generator=generator, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 13, (9,), generator=generator)
    upstream = torch.randn(9, generator=generator)

    actual = fused.vocab_parallel_fused_linear_logprobs(
        hidden,
        weight,
        labels,
        temperature=0.6,
        chunk_size=3,
    )
    (actual * upstream).sum().backward()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = F.log_softmax(F.linear(reference_hidden, reference_weight).float() / 0.6, -1)
    reference = reference.gather(-1, labels[:, None]).squeeze(-1)
    (reference * upstream).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    ("hidden_dtype", "weight_dtype"),
    [
        (torch.float32, torch.bfloat16),
        (torch.float64, torch.float32),
    ],
)
def test_mixed_hidden_and_weight_dtypes_match_explicit_megatron_compute_cast(
    hidden_dtype,
    weight_dtype,
):
    fused = _load_module()
    generator = torch.Generator().manual_seed(72)
    hidden = torch.randn(8, 5, generator=generator, dtype=hidden_dtype, requires_grad=True)
    weight = torch.randn(13, 5, generator=generator, dtype=weight_dtype, requires_grad=True)
    labels = torch.randint(0, 13, (8,), generator=generator)
    upstream = torch.randn(8, generator=generator)

    actual = fused.vocab_parallel_fused_linear_logprobs(
        hidden,
        weight,
        labels,
        temperature=0.85,
        chunk_size=3,
    )
    (actual * upstream).sum().backward()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_logits = F.linear(reference_hidden.to(weight_dtype), reference_weight)
    reference = F.log_softmax(reference_logits.float() / 0.85, -1)
    reference = reference.gather(-1, labels[:, None]).squeeze(-1)
    (reference * upstream).sum().backward()

    tolerance = 2e-2 if weight_dtype == torch.bfloat16 else 2e-6
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        hidden.grad,
        reference_hidden.grad,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        weight.grad,
        reference_weight.grad,
        rtol=tolerance,
        atol=tolerance,
    )
    assert hidden.grad.dtype == hidden_dtype
    assert weight.grad.dtype == weight_dtype


def _tp_gradient_worker(rank, world_size, init_file):
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fused = _load_module()
        generator = torch.Generator().manual_seed(91)
        reference_hidden = torch.randn(7, 5, generator=generator, requires_grad=True)
        reference_weight = torch.randn(12, 5, generator=generator, requires_grad=True)
        labels = torch.tensor([0, 7, 2, 11, 6, 3, 9])
        upstream = torch.tensor([0.5, -1.0, 0.2, 1.7, -0.3, 0.9, -0.4])

        hidden = reference_hidden.detach().clone().requires_grad_(True)
        weight = reference_weight.detach().chunk(world_size, dim=0)[rank].clone().requires_grad_(True)
        actual = fused.vocab_parallel_fused_linear_logprobs(
            hidden,
            weight,
            labels,
            temperature=0.8,
            tp_group=dist.group.WORLD,
            chunk_size=2,
        )
        (actual * upstream).sum().backward()
        # The production wrapper places Megatron's copy-to-TP autograd mapping
        # before this primitive; emulate its dHidden SUM here.
        dist.all_reduce(hidden.grad)

        reference = F.log_softmax(F.linear(reference_hidden, reference_weight) / 0.8, -1)
        reference = reference.gather(-1, labels[:, None]).squeeze(-1)
        (reference * upstream).sum().backward()

        torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=2e-6, atol=2e-6)
        expected_weight_grad = reference_weight.grad.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(weight.grad, expected_weight_grad, rtol=2e-6, atol=2e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for the CPU TP test")
def test_two_rank_vocab_parallel_forward_and_gradients_match_dense(tmp_path):
    mp.spawn(
        _tp_gradient_worker,
        args=(2, str(tmp_path / "tp-init")),
        nprocs=2,
        join=True,
    )


def test_weight_gradient_reaches_standard_parameter_accumulation_hook():
    fused = _load_module()
    hidden = torch.randn(4, 3, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(7, 3))
    labels = torch.tensor([0, 2, 6, 1])
    main_grad = torch.zeros_like(weight, dtype=torch.float32)
    hook_calls = []

    def accumulate(parameter):
        hook_calls.append(1)
        main_grad.add_(parameter.grad.float())
        parameter.grad = None

    weight.register_post_accumulate_grad_hook(accumulate)
    output = fused.vocab_parallel_fused_linear_logprobs(hidden, weight, labels, chunk_size=2)
    output.sum().backward()

    reference_weight = weight.detach().clone().requires_grad_(True)
    reference = F.log_softmax(F.linear(hidden.detach(), reference_weight), -1)
    reference.gather(-1, labels[:, None]).sum().backward()
    assert hook_calls == [1]
    assert weight.grad is None
    torch.testing.assert_close(main_grad, reference_weight.grad, rtol=2e-6, atol=2e-6)


class _FakeOutputLayer(torch.nn.Module):
    def __init__(self, *, sequence_parallel=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(5, 3))
        self.bias = None
        self.gather_output = False
        self.disable_grad_reduce = False
        self.sequence_parallel = sequence_parallel
        self.tp_group = "tp"
        self.config = SimpleNamespace(defer_embedding_wgrad_compute=False)
        self.original_calls = 0

    def forward(self, input_, weight=None, runtime_gather_output=None):
        del runtime_gather_output
        self.original_calls += 1
        return F.linear(input_, self.weight if weight is None else weight), None


class _FakeModel:
    def __init__(self, *, sequence_parallel=False):
        self.output_layer = _FakeOutputLayer(sequence_parallel=sequence_parallel)
        self.post_process = True
        self.mtp_process = False
        self.config = SimpleNamespace(use_mup=False)


@pytest.mark.parametrize(
    ("sequence_parallel", "expected_mapping"),
    [(False, "copy"), (True, "gather")],
)
def test_conditional_head_preserves_normal_calls_and_megatron_tp_mapping(sequence_parallel, expected_mapping):
    fused = _load_module()
    model = _FakeModel(sequence_parallel=sequence_parallel)
    mapping_calls = []

    def copy_mapping(value, *, group):
        mapping_calls.append(("copy", group))
        return value

    def gather_mapping(value, *, group):
        mapping_calls.append(("gather", group))
        return value

    state = fused.install_fused_linear_logprobs_head(
        [model],
        copy_to_tp_region=copy_mapping,
        gather_from_sp_region=gather_mapping,
    )
    assert state is not None
    hidden = torch.randn(4, 3)

    ordinary_output, _ = model.output_layer(hidden)
    torch.testing.assert_close(ordinary_output, F.linear(hidden, model.output_layer.weight))
    assert model.output_layer.original_calls == 1
    assert mapping_calls == []

    with fused.activate_fused_linear_logprobs_head(state):
        mapped_hidden, bias = model.output_layer(hidden)
        assert fused.fused_linear_logprobs_head_is_active(state)
    torch.testing.assert_close(mapped_hidden, hidden)
    assert bias is None
    assert mapping_calls == [(expected_mapping, "tp")]
    assert state.active_weight is model.output_layer.weight
    assert state.fused_forward_calls == 1
    assert model.output_layer.original_calls == 1
    assert (
        fused.install_fused_linear_logprobs_head(
            [model],
            copy_to_tp_region=copy_mapping,
            gather_from_sp_region=gather_mapping,
        )
        is state
    )


def test_conditional_head_fails_closed_for_full_vocab_gather():
    fused = _load_module()
    model = _FakeModel()
    state = fused.install_fused_linear_logprobs_head(
        [model],
        copy_to_tp_region=lambda value, **_: value,
        gather_from_sp_region=lambda value, **_: value,
    )
    assert state is not None

    with fused.activate_fused_linear_logprobs_head(state):
        with pytest.raises(RuntimeError, match="cannot gather"):
            model.output_layer(torch.randn(4, 3), runtime_gather_output=True)


def test_conditional_head_rejects_preinstalled_fp32_output_hook():
    fused = _load_module()
    model = _FakeModel()
    model.output_layer._platoon_fp32_lm_head_hook_handle = object()

    with pytest.raises(RuntimeError, match="Remove the FP32 LM-head output hook"):
        fused.install_fused_linear_logprobs_head(
            [model],
            copy_to_tp_region=lambda value, **_: value,
            gather_from_sp_region=lambda value, **_: value,
        )
