from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
FEATURE_CONFIG = "toolathlon_openhands_areal_prealloc_16node-cp-fp32-lm-head.yaml"
FP32_LM_HEAD_PATH = REPO_ROOT / "platoon/train/areal/fp32_lm_head.py"

_SPEC = importlib.util.spec_from_file_location("platoon_areal_fp32_lm_head", FP32_LM_HEAD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
install_fp32_lm_head_output_hooks = _MODULE.install_fp32_lm_head_output_hooks


class _TupleOutputHead(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias_result = bias
        self.scale = nn.Parameter(torch.tensor(2.0, dtype=bias.dtype))

    def forward(self, inputs: torch.Tensor):
        return inputs * self.scale, self.bias_result


class _LanguageModel(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.output_layer = head


class _LanguageModelContainer(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model


class _DDPContainer(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp32_lm_head_hook_casts_logits_and_preserves_gradient(dtype):
    bias = torch.tensor([3.0], dtype=dtype)
    head = _TupleOutputHead(bias)
    model = _DDPContainer(_LanguageModelContainer(_LanguageModel(head)))

    assert install_fp32_lm_head_output_hooks([model], enabled=True, is_critic=False) == 1
    # Installation is idempotent on repeated initialization attempts.
    assert install_fp32_lm_head_output_hooks([model], enabled=True, is_critic=False) == 0

    inputs = torch.tensor([1.0, -2.0], dtype=dtype, requires_grad=True)
    logits, returned_bias = head(inputs)
    assert logits.dtype == torch.float32
    assert returned_bias is bias

    logits.sum().backward()
    assert inputs.grad is not None
    assert inputs.grad.dtype == dtype
    torch.testing.assert_close(inputs.grad.float(), torch.full((2,), 2.0))
    assert head.scale.dtype == dtype
    assert head.scale.grad is not None
    assert head.scale.grad.dtype == dtype
    torch.testing.assert_close(head.scale.grad.float(), torch.tensor(-1.0))


@pytest.mark.parametrize(
    ("enabled", "is_critic"),
    [(False, False), (True, True)],
)
def test_fp32_lm_head_hook_does_not_change_disabled_or_critic_outputs(enabled, is_critic):
    head = _TupleOutputHead(torch.tensor([0.0], dtype=torch.bfloat16))
    model = _LanguageModel(head)

    assert (
        install_fp32_lm_head_output_hooks(
            [model],
            enabled=enabled,
            is_critic=is_critic,
        )
        == 0
    )
    logits, _ = head(torch.ones(2, dtype=torch.bfloat16))
    assert logits.dtype == torch.bfloat16


def test_fp32_lm_head_hook_fails_closed_on_post_process_chunk_without_head():
    model = nn.Module()
    model.post_process = True

    with pytest.raises(RuntimeError, match="post-process Megatron model chunk has no output_layer"):
        install_fp32_lm_head_output_hooks([model], enabled=True, is_critic=False)


def test_fp32_lm_head_hook_allows_zero_hooks_on_non_last_pipeline_stage():
    model = nn.Module()
    model.post_process = False

    assert install_fp32_lm_head_output_hooks([model], enabled=True, is_critic=False) == 0


def test_fp32_lm_head_feature_config_is_explicit_and_isolated():
    config = yaml.safe_load((CONFIG_DIR / FEATURE_CONFIG).read_text())

    assert config["defaults"] == ["toolathlon_openhands_areal_prealloc_16node-cp", "_self_"]
    assert config["trial_name"] == "toolathlon-openhands-16node-qwen3.6-35B-fp32-lm-head-trial0"
    assert "openreward" in config  # Required by the launcher's pre-Hydra section check.
    assert config["actor"]["backend"].startswith("megatron:")
    assert config["actor"]["path"] == "Qwen/Qwen3.6-35B-A3B"
    assert config["actor"]["megatron"]["enable_mtp"] is False
    assert config["actor"]["megatron"]["enable_fp32_lm_head"] is True
    assert config["ref"]["megatron"]["enable_fp32_lm_head"] is True


@pytest.mark.parametrize(
    "baseline_name",
    [
        "toolathlon_openhands_areal_prealloc_16node-cp.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-bs16.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-ptc-recursive.yaml",
        "toolathlon_openhands_areal_prealloc_16node.yaml",
        "toolathlon_openhands_areal_prealloc_8node.yaml",
    ],
)
def test_existing_qwen_resume_configs_do_not_enable_fp32_lm_head(baseline_name):
    config = yaml.safe_load((CONFIG_DIR / baseline_name).read_text())

    assert "enable_fp32_lm_head" not in config.get("actor", {}).get("megatron", {})
    assert "enable_fp32_lm_head" not in config.get("ref", {}).get("megatron", {})
