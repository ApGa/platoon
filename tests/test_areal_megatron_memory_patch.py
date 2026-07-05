from __future__ import annotations

import dis
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


def _load_patches_module():
    spec = importlib.util.spec_from_file_location("platoon_areal_memory_patches", PATCHES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _UnsafeVocabReductionEngine:
    def _compute_logprobs_and_loss(
        self,
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        loss_multiplier=1.0,
    ):
        del self, inputs, loss_fn, loss_weight_fn, total_loss_weight, loss_multiplier
        vocab_mean_logits = output.detach().float().mean(-1)
        vocab_norm_logits = output.detach().float().norm(dim=-1)
        return vocab_mean_logits, vocab_norm_logits


class _ChangedVocabReductionEngine:
    def _compute_logprobs_and_loss(
        self,
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        loss_multiplier=1.0,
        new_upstream_parameter=None,
    ):
        del (
            self,
            inputs,
            loss_fn,
            loss_weight_fn,
            total_loss_weight,
            loss_multiplier,
            new_upstream_parameter,
        )
        return output.detach().float().mean(-1)


class _PartialVocabReductionEngine:
    def _compute_logprobs_and_loss(
        self,
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        loss_multiplier=1.0,
    ):
        del self, inputs, loss_fn, loss_weight_fn, total_loss_weight, loss_multiplier
        return output.detach().float().mean(-1)


class _FakeMpu:
    @staticmethod
    def get_tensor_model_parallel_world_size():
        return 2

    @staticmethod
    def get_tensor_model_parallel_group():
        return "fake-tp-group"


class _ForwardEngine:
    def __init__(self):
        self.config = SimpleNamespace(is_critic=False, temperature=0.7)
        self.enable_tree_training = False
        self.is_vision_model = False
        self.use_padded_seq = False
        self.gather_history = []

    def _compute_logprobs_and_loss(
        self,
        output,
        inputs,
        loss_fn,
        loss_weight_fn,
        total_loss_weight,
        loss_multiplier=1.0,
    ):
        del self, inputs, loss_fn, loss_weight_fn, total_loss_weight, loss_multiplier
        return output.sum()

    def _compute_forward_result(self, output, inputs):
        return "upstream", output, inputs

    def forward_backward_batch(
        self,
        mb_list,
        process_output_fn,
        forward_only=False,
        gather_cp_output=False,
    ):
        del mb_list, process_output_fn, forward_only
        self.gather_history.append(gather_cp_output)
        return gather_cp_output

    def forward_batch(self, input_=None):
        del input_
        return self.forward_backward_batch(
            None,
            lambda output, inputs: None,
            forward_only=True,
            gather_cp_output=True,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp32_vocab_reductions_match_materialized_reference(dtype):
    patches = _load_patches_module()
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn(23, 257, generator=generator, dtype=torch.float32).to(dtype)

    expected_mean = logits.float().mean(-1)
    expected_norm = logits.float().norm(dim=-1)
    # Force eight chunks so this also exercises cross-chunk result assembly.
    scratch_bytes = 3 * logits.shape[-1] * torch.float32.itemsize
    actual_mean = patches._fp32_vocab_mean_without_materializing_logits(logits, max_scratch_bytes=scratch_bytes)
    actual_norm = patches._fp32_vocab_norm_without_materializing_logits(logits, max_scratch_bytes=scratch_bytes)

    torch.testing.assert_close(actual_mean, expected_mean, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("reduction", ["mean", "norm"])
def test_vocab_reduction_fp32_casts_are_bounded_by_complete_rows(reduction):
    patches = _load_patches_module()
    logits = torch.randn(10, 31, dtype=torch.bfloat16)
    row_bytes = logits.shape[-1] * torch.float32.itemsize
    max_scratch_bytes = 3 * row_bytes
    observed_chunk_bytes = []

    patches._bounded_fp32_vocab_reduction(
        logits,
        reduction,
        max_scratch_bytes=max_scratch_bytes,
        _chunk_observer=lambda chunk: observed_chunk_bytes.append(chunk.numel() * chunk.element_size()),
    )

    assert len(observed_chunk_bytes) == 4
    assert max(observed_chunk_bytes) <= max_scratch_bytes
    assert observed_chunk_bytes == [3 * row_bytes, 3 * row_bytes, 3 * row_bytes, row_bytes]

    observed_chunk_bytes.clear()
    patches._bounded_fp32_vocab_reduction(
        logits[:2],
        reduction,
        max_scratch_bytes=1,
        _chunk_observer=lambda chunk: observed_chunk_bytes.append(chunk.numel() * chunk.element_size()),
    )
    assert observed_chunk_bytes == [row_bytes, row_bytes]


def test_vocab_reduction_fails_closed_for_noncontiguous_logits():
    patches = _load_patches_module()
    logits = torch.randn(5, 7, 11, dtype=torch.bfloat16).transpose(0, 1)
    assert not logits.is_contiguous()

    with pytest.raises(RuntimeError, match="non-contiguous vocab logits"):
        patches._fp32_vocab_mean_without_materializing_logits(logits)


def test_vocab_rewrite_changes_only_the_two_guarded_reductions():
    patches = _load_patches_module()
    engine_module = SimpleNamespace(MegatronEngine=_UnsafeVocabReductionEngine)
    original = _UnsafeVocabReductionEngine._compute_logprobs_and_loss
    logits = torch.randn(11, 101, dtype=torch.bfloat16)
    expected = original(None, logits, None, None, None, None)

    result = patches._patch_areal_megatron_memory_compatibility(engine_module)
    rewritten = _UnsafeVocabReductionEngine._compute_logprobs_and_loss
    actual = rewritten(None, logits, None, None, None, None)

    assert result == {"vocab_reductions": True, "cp_forward_scalars": False}
    assert rewritten is not original
    instruction_names = {
        instruction.argval
        for instruction in dis.get_instructions(rewritten)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_METHOD"}
    }
    assert "_platoon_fp32_vocab_mean" in instruction_names
    assert "_platoon_fp32_vocab_norm" in instruction_names
    assert "float" not in instruction_names
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)


def test_vocab_rewrite_skips_an_unknown_upstream_signature():
    patches = _load_patches_module()
    engine_module = SimpleNamespace(MegatronEngine=_ChangedVocabReductionEngine)
    original = _ChangedVocabReductionEngine._compute_logprobs_and_loss

    result = patches._patch_areal_megatron_memory_compatibility(engine_module)

    assert result == {"vocab_reductions": False, "cp_forward_scalars": False}
    assert _ChangedVocabReductionEngine._compute_logprobs_and_loss is original


def test_vocab_rewrite_fails_clearly_on_a_partial_guard_match():
    patches = _load_patches_module()
    engine_module = SimpleNamespace(MegatronEngine=_PartialVocabReductionEngine)

    with pytest.raises(RuntimeError, match="exactly one unsafe mean and one unsafe norm"):
        patches._patch_areal_megatron_memory_compatibility(engine_module)


def test_implicit_megatron_import_is_gated_on_qwen_gdn_cp(monkeypatch):
    patches = _load_patches_module()
    monkeypatch.delenv("PLATOON_QWEN35_GDN_CP", raising=False)

    assert patches._patch_areal_megatron_memory_compatibility() == {
        "vocab_reductions": False,
        "cp_forward_scalars": False,
    }


def test_forward_batch_uses_cp_local_scalars_only_for_supported_layouts():
    patches = _load_patches_module()
    calls = []

    def gather_logprobs(output, labels, *, temperature, tp_group):
        calls.append(("gather_logprobs", labels.clone(), temperature, tp_group))
        return output[:, 0].float() + labels.float()

    def reassemble_cp_packed_logprobs(result, cu_seqlens):
        calls.append(("reassemble", cu_seqlens.clone()))
        return result + 10

    def unpad_logits(result, padding_length, cu_seqlens, old_cu_seqlens):
        calls.append(("unpad", padding_length, cu_seqlens.clone(), old_cu_seqlens.clone()))
        return result[:-padding_length]

    engine_module = SimpleNamespace(
        MegatronEngine=_ForwardEngine,
        gather_logprobs=gather_logprobs,
        reassemble_cp_packed_logprobs=reassemble_cp_packed_logprobs,
        unpad_logits=unpad_logits,
        mpu=_FakeMpu,
    )
    patch_result = patches._patch_areal_megatron_memory_compatibility(engine_module)
    engine = _ForwardEngine()

    # A direct caller's explicit gather request remains untouched.
    assert engine.forward_backward_batch(None, lambda *_: None, True, True) is True
    # Packed text forward_batch can select one scalar per token before CP gather.
    assert engine.forward_batch() is False

    for unsafe_attribute in ("enable_tree_training", "is_vision_model", "use_padded_seq"):
        setattr(engine, unsafe_attribute, True)
        assert engine.forward_batch() is True
        setattr(engine, unsafe_attribute, False)

    assert patch_result == {"vocab_reductions": False, "cp_forward_scalars": True}
    assert engine.gather_history == [True, False, True, True, True]

    output = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    labels = torch.tensor([1, 2, 3, 4])
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    old_cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    inputs = {
        "_cp_local_labels": labels,
        "_cp_padded_cu_seqlens": cu_seqlens,
        "_cp_padding_length": 1,
        "_cp_old_cu_seqlens": old_cu_seqlens,
    }
    actor_result = engine._compute_forward_result(output, inputs)
    expected_actor = (output[:, 0].float() + labels.float() + 10)[:-1]
    torch.testing.assert_close(actor_result, expected_actor)
    assert calls[0][0] == "gather_logprobs"
    assert calls[0][2:] == (0.7, "fake-tp-group")
    assert [call[0] for call in calls] == ["gather_logprobs", "reassemble", "unpad"]

    calls.clear()
    engine.config.is_critic = True
    critic_output = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    critic_result = engine._compute_forward_result(critic_output, inputs)
    torch.testing.assert_close(critic_result, (critic_output.squeeze(-1) + 10)[:-1])
    assert [call[0] for call in calls] == ["reassemble", "unpad"]

    ordinary_inputs = {"input_ids": torch.tensor([1, 2, 3])}
    assert engine._compute_forward_result(output, ordinary_inputs) == (
        "upstream",
        output,
        ordinary_inputs,
    )
