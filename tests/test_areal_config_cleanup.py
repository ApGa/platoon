"""Focused tests for Platoon's AReaL config cleanup."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_config_module():
    existing = sys.modules.get("platoon_areal_config_defs_test")
    if existing is not None:
        return existing

    cli_args_mod = types.ModuleType("areal.api.cli_args")
    sys.modules["areal.api.cli_args"] = cli_args_mod

    @dataclass
    class FakeSchedulerConfig:
        type: str | None = None

    @dataclass
    class FakeOpenAIConfig:
        mode: str = "offline"
        admin_api_key: str = "test-key"

    @dataclass
    class FakePPOActorConfig:
        backend: str = ""
        behave_imp_weight_cap: float | None = None

        def __post_init__(self):
            pass

    @dataclass
    class FakeInferenceEngineConfig:
        backend: str = ""
        experiment_name: str | None = None
        trial_name: str | None = None
        fileroot: str | None = None
        tokenizer_path: str = ""
        openai: FakeOpenAIConfig | None = None
        scheduling_strategy: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)

    @dataclass
    class FakeGRPOConfig:
        scheduler: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)

        def __post_init__(self):
            pass

    for cls in (
        FakeSchedulerConfig,
        FakeOpenAIConfig,
        FakePPOActorConfig,
        FakeInferenceEngineConfig,
        FakeGRPOConfig,
    ):
        cls.__module__ = "areal.api.cli_args"

    cli_args_mod.FakeSchedulerConfig = FakeSchedulerConfig
    cli_args_mod.FakeOpenAIConfig = FakeOpenAIConfig
    cli_args_mod.GRPOConfig = FakeGRPOConfig
    cli_args_mod.PPOActorConfig = FakePPOActorConfig
    cli_args_mod.InferenceEngineConfig = FakeInferenceEngineConfig

    dist_rollout_mod = types.ModuleType("areal.core.dist_rollout")
    dist_rollout_mod.redistribute = lambda local_batch, granularity: types.SimpleNamespace(data=local_batch)
    sys.modules["areal.core.dist_rollout"] = dist_rollout_mod

    utils_data_mod = types.ModuleType("areal.utils.data")
    utils_data_mod.all_gather_tensor_container = lambda batch, group=None: [batch]
    utils_data_mod.broadcast_tensor_container = lambda batch, src_rank=0: batch
    utils_data_mod.concat_padded_tensors = lambda items: items[0] if items else {}
    utils_data_mod.get_batch_size = lambda batch: len(next(iter(batch.values()))) if batch else 0
    utils_data_mod.tensor_container_to = lambda batch, device=None: batch
    sys.modules["areal.utils.data"] = utils_data_mod

    return _load_module(
        "platoon_areal_config_defs_test",
        REPO_ROOT / "platoon/train/areal/config_defs.py",
    )


def test_minimal_platoon_first_config_parses_and_injects_loss_settings():
    config_mod = _load_config_module()

    cfg = OmegaConf.merge(
        OmegaConf.structured(config_mod.PlatoonArealRLTrainerConfig),
        OmegaConf.create(
            {
                "rollout": {"backend": "sglang:d4p1t1"},
                "actor": {"backend": "fsdp:d2p1t1"},
                "loss_fn_config": {
                    "loss_fn": "cispo",
                    "loss_fn_kwargs": {
                        "clip_low_threshold": 0.1,
                        "clip_high_threshold": 4.2,
                        "alpha": 3.0,
                    },
                },
            }
        ),
    )

    parsed = OmegaConf.to_object(cfg)

    assert parsed.rollout.backend == "sglang:d4p1t1"
    assert parsed.actor.backend == "fsdp:d2p1t1"
    assert parsed.actor.loss_fn == "cispo"
    assert parsed.actor.loss_fn_kwargs["clip_low_threshold"] == 0.1
    assert parsed.actor.loss_fn_kwargs["clip_high_threshold"] == 4.2
    assert parsed.actor.loss_fn_kwargs["alpha"] == 3.0
    assert parsed.scheduler.type == "local"
    assert parsed.eval_gconfig.lora_name == parsed.gconfig.lora_name
    assert not hasattr(parsed.train_dataset, "path")
    assert not hasattr(parsed.valid_dataset, "type")


def test_removed_top_level_legacy_keys_are_rejected():
    config_mod = _load_config_module()
    schema = OmegaConf.structured(config_mod.PlatoonArealRLTrainerConfig)

    for payload in (
        {"allocation_mode": "sglang:d4p1t1+d2p1t1"},
        {"launcher": {"trainer_mem_per_gpu": 32768}},
        {"gconfig": {"n_samples": 8}},
    ):
        try:
            OmegaConf.merge(schema, OmegaConf.create(payload))
        except Exception:
            continue
        raise AssertionError(f"Expected payload to be rejected: {payload}")


def test_removed_nested_legacy_keys_are_rejected():
    config_mod = _load_config_module()
    schema = OmegaConf.structured(config_mod.PlatoonArealRLTrainerConfig)

    for payload in (
        {"actor": {"group_size": 8}},
        {"actor": {"dynamic_sampling": False}},
        {"actor": {"clip_low_threshold": 0.0}},
        {"loss_fn_config": {"clip_low_threshold": 0.0}},
        {"train_dataset": {"path": ""}},
        {"train_dataset": {"type": "rl"}},
        {"valid_dataset": {"path": ""}},
    ):
        try:
            OmegaConf.merge(schema, OmegaConf.create(payload))
        except Exception:
            continue
        raise AssertionError(f"Expected payload to be rejected: {payload}")
