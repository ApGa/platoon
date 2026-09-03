"""CPU-only tests for the opt-in Megatron CISPO post-logits fast path."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOSS_FUNCTIONS_PATH = REPO_ROOT / "platoon/train/areal/loss_functions.py"
CISPO_OPTIMIZATION_PATH = REPO_ROOT / "platoon/train/areal/cispo_vocab_optimization.py"
CONFIG_DEFS_PATH = REPO_ROOT / "platoon/train/areal/config_defs.py"
LOGPROBS_ONLY_ATTRIBUTE = "__platoon_supports_logprobs_only_training__"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _RecordingStatsTracker:
    def __init__(self) -> None:
        self.denominator_calls: list[dict] = []
        self.stat_calls: list[dict] = []

    def denominator(self, **kwargs) -> None:
        self.denominator_calls.append(kwargs)

    def stat(self, **kwargs) -> None:
        self.stat_calls.append(kwargs)

    def reset(self) -> None:
        self.denominator_calls.clear()
        self.stat_calls.clear()


@pytest.fixture
def cispo_modules(monkeypatch):
    """Load the two focused modules with only AReaL's small loss API mocked."""

    tracker = _RecordingStatsTracker()

    areal_module = types.ModuleType("areal")
    areal_module.__path__ = []
    trainer_module = types.ModuleType("areal.trainer")
    trainer_module.__path__ = []
    ppo_module = types.ModuleType("areal.trainer.ppo")
    ppo_module.__path__ = []
    actor_module = types.ModuleType("areal.trainer.ppo.actor")
    actor_module.grpo_loss_fn = lambda logprobs, entropy, input_data, **kwargs: logprobs.sum()
    stats_module = types.ModuleType("areal.trainer.ppo.stats")
    stats_module.infer_token_denominator = lambda input_data, loss_mask: loss_mask
    utils_module = types.ModuleType("areal.utils")
    utils_module.stats_tracker = tracker

    for name, module in {
        "areal": areal_module,
        "areal.trainer": trainer_module,
        "areal.trainer.ppo": ppo_module,
        "areal.trainer.ppo.actor": actor_module,
        "areal.trainer.ppo.stats": stats_module,
        "areal.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    loss_functions = _load_module(
        "platoon_areal_cispo_loss_functions_test",
        LOSS_FUNCTIONS_PATH,
    )
    # The optimization imports this canonical name. Supplying the already-loaded
    # module avoids importing platoon.train.areal and its controller integration.
    monkeypatch.setitem(
        sys.modules,
        "platoon.train.areal.loss_functions",
        loss_functions,
    )
    fused_linear_module = types.ModuleType("platoon.train.areal.fused_linear_logprobs")
    fused_linear_module.FusedLinearLogprobsHeadState = object
    fused_linear_module.vocab_parallel_fused_linear_logprobs = lambda *args, **kwargs: pytest.fail(
        "The post-logits tests must not enter the fused-linear path"
    )
    monkeypatch.setitem(
        sys.modules,
        "platoon.train.areal.fused_linear_logprobs",
        fused_linear_module,
    )
    optimization = _load_module(
        "platoon_areal_cispo_vocab_optimization_test",
        CISPO_OPTIMIZATION_PATH,
    )
    return loss_functions, optimization, tracker


def _selected_logprobs(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    return (
        torch.log_softmax(logits.float() / temperature, dim=-1)
        .gather(
            -1,
            labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )


def _logprobs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_distribution = torch.log_softmax(logits.float() / temperature, dim=-1)
    logprobs = log_distribution.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_distribution.exp() * log_distribution).sum(dim=-1)
    return logprobs, entropy


def test_logprobs_only_cispo_matches_detached_entropy_loss_and_gradients(cispo_modules):
    loss_functions, optimization, tracker = cispo_modules
    generator = torch.Generator().manual_seed(41)
    initial_logits = torch.randn(7, 11, generator=generator, dtype=torch.float64)
    input_ids = torch.tensor([1, 7, 3, 9, 4, 6, 2])
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    temperature = 0.73

    with torch.no_grad():
        initial_new_logprobs = _selected_logprobs(initial_logits, labels, temperature)
    inputs = {
        "input_ids": input_ids,
        "logprobs": initial_new_logprobs + torch.tensor([0.2, -0.4, 0.0, 0.7, -0.1, 0.3, -0.6]),
        "advantages": torch.tensor([1.2, -0.3, 0.8, -1.1, 0.5, 2.0, -0.7]),
        "loss_mask": torch.tensor([True, True, False, True, True, False, True]),
    }
    loss_fn = loss_functions.build_loss_fn(
        "cispo",
        loss_fn_kwargs={"clip_low_threshold": 0.15, "clip_high_threshold": 1.6},
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(is_critic=False, temperature=temperature),
        enable_tree_training=False,
    )

    def gather_logprobs(output, selected_labels, *, temperature, tp_group):
        assert tp_group is None
        return _selected_logprobs(output, selected_labels, temperature)

    megatron_module = SimpleNamespace(
        mpu=SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 1,
            get_tensor_model_parallel_group=lambda: pytest.fail("TP group should not be requested at TP=1"),
        ),
        gather_logprobs=gather_logprobs,
    )
    local_weight = torch.tensor(3.0)
    total_weight = torch.tensor(8.0)
    loss_multiplier = 2.5

    old_logits = initial_logits.clone().requires_grad_(True)
    old_logprobs, entropy = _logprobs_and_entropy(old_logits, labels, temperature)
    old_loss = loss_fn(
        old_logprobs,
        entropy,
        inputs,
        vocab_min_logits=old_logits.detach().min(-1).values,
        vocab_max_logits=old_logits.detach().max(-1).values,
        vocab_mean_logits=old_logits.detach().mean(-1),
        vocab_norm_logits=old_logits.detach().norm(dim=-1),
    )
    old_loss = old_loss * (local_weight / total_weight * loss_multiplier)
    old_loss.backward()
    old_gradient = old_logits.grad.detach().clone()
    assert "entropy" in tracker.stat_calls[-1]

    tracker.reset()
    new_logits = initial_logits.clone().requires_grad_(True)
    new_loss = optimization.compute_logprobs_only_megatron_loss(
        engine,
        new_logits,
        inputs,
        loss_fn,
        lambda _: local_weight,
        total_weight,
        loss_multiplier,
        megatron_engine_module=megatron_module,
    )
    new_loss.backward()

    torch.testing.assert_close(new_loss, old_loss.detach(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(new_logits.grad, old_gradient, rtol=1e-12, atol=1e-12)
    assert "entropy" not in tracker.stat_calls[-1]
    assert len(tracker.denominator_calls) == 1


def test_logprobs_only_uses_cp_local_labels_then_reassembles_and_unpads(cispo_modules):
    _, optimization, _ = cispo_modules
    calls: list[tuple] = []

    class FakeMpu:
        @staticmethod
        def get_tensor_model_parallel_world_size():
            return 2

        @staticmethod
        def get_tensor_model_parallel_group():
            return "tp-group"

    def gather_logprobs(output, labels, *, temperature, tp_group):
        calls.append(("gather", labels.clone(), temperature, tp_group))
        return _selected_logprobs(output, labels, temperature)

    def reassemble_cp_packed_logprobs(local_logprobs, cu_seqlens):
        calls.append(("reassemble", cu_seqlens.clone()))
        return torch.cat((local_logprobs[:1], local_logprobs, local_logprobs[-1:]))

    def unpad_logits(logprobs, padding_length, cu_seqlens, old_cu_seqlens):
        calls.append(
            (
                "unpad",
                padding_length,
                cu_seqlens.clone(),
                old_cu_seqlens.clone(),
            )
        )
        return logprobs[:-padding_length]

    megatron_module = SimpleNamespace(
        mpu=FakeMpu,
        gather_logprobs=gather_logprobs,
        reassemble_cp_packed_logprobs=reassemble_cp_packed_logprobs,
        unpad_logits=unpad_logits,
    )
    engine = SimpleNamespace(
        config=SimpleNamespace(is_critic=False, temperature=0.5),
        enable_tree_training=False,
    )
    output = torch.tensor(
        [[0.2, -0.1, 0.7, 0.4], [-0.3, 0.8, 0.1, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cp_labels = torch.tensor([2, 1])
    padded_cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    old_cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    inputs = {
        "input_ids": torch.tensor([9, 9, 9]),
        "ordinary": "preserved",
        "_cp_local_labels": cp_labels,
        "_cp_padded_cu_seqlens": padded_cu_seqlens,
        "_cp_padding_length": 1,
        "_cp_old_cu_seqlens": old_cu_seqlens,
    }
    observed_loss_call = {}

    def loss_fn(logprobs, entropy, loss_inputs, **kwargs):
        observed_loss_call.update(
            logprobs=logprobs,
            entropy=entropy,
            inputs=loss_inputs,
            kwargs=kwargs,
        )
        return logprobs.sum()

    setattr(loss_fn, LOGPROBS_ONLY_ATTRIBUTE, True)
    result = optimization.compute_logprobs_only_megatron_loss(
        engine,
        output,
        inputs,
        loss_fn,
        lambda _: torch.tensor(2.0),
        torch.tensor(5.0),
        loss_multiplier=3.0,
        megatron_engine_module=megatron_module,
    )
    result.backward()

    assert [call[0] for call in calls] == ["gather", "reassemble", "unpad"]
    torch.testing.assert_close(calls[0][1], cp_labels)
    assert calls[0][2:] == (0.5, "tp-group")
    assert calls[2][1] == 1
    assert observed_loss_call["entropy"] is None
    assert observed_loss_call["inputs"] == {
        "input_ids": inputs["input_ids"],
        "ordinary": "preserved",
    }
    assert set(observed_loss_call["kwargs"]) == {
        "vocab_min_logits",
        "vocab_max_logits",
        "vocab_mean_logits",
        "vocab_norm_logits",
    }
    assert all(value is None for value in observed_loss_call["kwargs"].values())
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()


@pytest.mark.parametrize("loss_name", ["grpo", "ppo"])
def test_logprobs_only_fails_closed_for_losses_without_explicit_capability(cispo_modules, loss_name):
    loss_functions, optimization, _ = cispo_modules
    loss_fn = loss_functions.build_loss_fn(loss_name)
    gather_called = False

    def gather_logprobs(*args, **kwargs):
        nonlocal gather_called
        gather_called = True
        return torch.tensor(0.0)

    megatron_module = SimpleNamespace(
        mpu=SimpleNamespace(get_tensor_model_parallel_world_size=lambda: 1),
        gather_logprobs=gather_logprobs,
    )

    assert not loss_functions.supports_logprobs_only_training(loss_fn)
    with pytest.raises(RuntimeError, match="does not declare entropy"):
        optimization.compute_logprobs_only_megatron_loss(
            SimpleNamespace(config=SimpleNamespace(is_critic=False), enable_tree_training=False),
            torch.randn(2, 3),
            {"input_ids": torch.tensor([1, 2])},
            loss_fn,
            lambda _: torch.tensor(1.0),
            torch.tensor(1.0),
            megatron_engine_module=megatron_module,
        )
    assert gather_called is False


def test_only_built_cispo_loss_advertises_logprobs_only_capability(cispo_modules):
    loss_functions, _, _ = cispo_modules

    assert loss_functions.supports_logprobs_only_training(loss_functions.build_loss_fn("cispo"))
    assert not loss_functions.supports_logprobs_only_training(loss_functions.build_loss_fn("grpo"))
    assert not loss_functions.supports_logprobs_only_training(loss_functions.cispo_loss_fn)
    assert not loss_functions.supports_logprobs_only_training(lambda *_args, **_kwargs: torch.tensor(0.0))


def _load_config_module(monkeypatch):
    areal_module = types.ModuleType("areal")
    areal_module.__path__ = []
    api_module = types.ModuleType("areal.api")
    api_module.__path__ = []
    cli_args_module = types.ModuleType("areal.api.cli_args")

    @dataclass
    class FakeSchedulerConfig:
        type: str | None = None

    @dataclass
    class FakeMegatronConfig:
        enable_mtp: bool = False
        recompute_granularity: str | None = "full"
        recompute_method: str | None = "uniform"

    @dataclass
    class FakePPOActorConfig:
        backend: str = ""
        enable_tree_training: bool = False
        gradient_checkpointing: bool = True
        recompute_logprob: bool = False
        use_decoupled_loss: bool = False
        scheduling_spec: list = field(default_factory=list)
        megatron: FakeMegatronConfig = field(default_factory=FakeMegatronConfig)

        def __post_init__(self):
            pass

        def should_compute_prox_logp(self):
            return self.recompute_logprob or self.use_decoupled_loss

    @dataclass
    class FakeGRPOConfig:
        scheduler: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)
        critic: FakePPOActorConfig | None = None
        teacher: FakePPOActorConfig | None = None

        def __post_init__(self):
            pass

    @dataclass
    class FakeInferenceEngineConfig:
        backend: str = ""
        return_routed_experts: bool = False
        scheduling_spec: list = field(default_factory=list)

    for cls in (
        FakeSchedulerConfig,
        FakeMegatronConfig,
        FakePPOActorConfig,
        FakeGRPOConfig,
        FakeInferenceEngineConfig,
    ):
        cls.__module__ = "areal.api.cli_args"

    cli_args_module.GRPOConfig = FakeGRPOConfig
    cli_args_module.PPOActorConfig = FakePPOActorConfig
    for name, module in {
        "areal": areal_module,
        "areal.api": api_module,
        "areal.api.cli_args": cli_args_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    shared_config_module = types.ModuleType("platoon.config_defs")

    @dataclass
    class FakeRolloutConfig:
        pass

    shared_config_module.RolloutConfig = FakeRolloutConfig
    monkeypatch.setitem(sys.modules, "platoon.config_defs", shared_config_module)

    components_module = types.ModuleType("platoon.train.components")

    @dataclass
    class FakeEnvironmentConfig:
        pass

    components_module.EnvironmentConfig = FakeEnvironmentConfig
    components_module.normalize_environment_configs = lambda environments: list(environments)
    monkeypatch.setitem(sys.modules, "platoon.train.components", components_module)

    train_utils_module = types.ModuleType("platoon.utils.train")
    train_utils_module.VariableBatchInferenceEngineConfig = FakeInferenceEngineConfig
    monkeypatch.setitem(sys.modules, "platoon.utils.train", train_utils_module)

    return _load_module("platoon_areal_cispo_config_defs_test", CONFIG_DEFS_PATH)


def _make_trainer_config(
    config_module,
    *,
    enabled: bool,
    actor_backend: str,
    loss_name: str,
    tree_training: bool = False,
    fused_linear: bool = False,
    fused_chunk_size: int = 512,
):
    return config_module.PlatoonArealRLTrainerConfig(
        rollout=config_module.VariableBatchInferenceEngineConfig(backend="sglang:d1p1t1"),
        actor=config_module.PlatoonPPOActorConfig(
            backend=actor_backend,
            enable_tree_training=tree_training,
            enable_cispo_logprobs_only=enabled,
            enable_cispo_fused_linear_logprobs=fused_linear,
            cispo_fused_linear_chunk_size=fused_chunk_size,
        ),
        loss_fn_config=config_module.LossFnConfig(loss_fn=loss_name),
    )


def test_cispo_logprobs_only_is_default_off_and_does_not_constrain_other_losses(monkeypatch):
    config_module = _load_config_module(monkeypatch)

    config = _make_trainer_config(
        config_module,
        enabled=False,
        actor_backend="fsdp:d1p1t1",
        loss_name="grpo",
        tree_training=True,
    )

    assert config.actor.enable_cispo_logprobs_only is False
    assert config.actor.enable_cispo_fused_linear_logprobs is False
    assert config.actor.cispo_fused_linear_chunk_size == 512
    assert config.actor.loss_fn == "grpo"


def test_cispo_logprobs_only_accepts_only_the_supported_configuration(monkeypatch):
    config_module = _load_config_module(monkeypatch)

    config = _make_trainer_config(
        config_module,
        enabled=True,
        actor_backend="megatron:d1p1t1",
        loss_name="cispo",
    )

    assert config.actor.enable_cispo_logprobs_only is True
    assert config.actor.loss_fn == "cispo"


@pytest.mark.parametrize(
    ("actor_backend", "loss_name", "tree_training", "error"),
    [
        ("fsdp:d1p1t1", "cispo", False, "require the Megatron actor backend"),
        ("megatron:d1p1t1", "grpo", False, "require loss_fn_config.loss_fn=cispo"),
        ("megatron:d1p1t1", "cispo", True, "do not support tree training"),
    ],
)
def test_cispo_logprobs_only_config_validation_fails_closed(
    monkeypatch,
    actor_backend,
    loss_name,
    tree_training,
    error,
):
    config_module = _load_config_module(monkeypatch)

    with pytest.raises(ValueError, match=error):
        _make_trainer_config(
            config_module,
            enabled=True,
            actor_backend=actor_backend,
            loss_name=loss_name,
            tree_training=tree_training,
        )

def test_cispo_vocab_optimization_config_rejects_ambiguous_or_invalid_modes(monkeypatch):
    config_module = _load_config_module(monkeypatch)

    with pytest.raises(ValueError, match="distinct alternatives and cannot both be enabled"):
        _make_trainer_config(
            config_module,
            enabled=True,
            fused_linear=True,
            actor_backend="megatron:d1p1t1",
            loss_name="cispo",
        )

    with pytest.raises(ValueError, match="cispo_fused_linear_chunk_size must be positive"):
        _make_trainer_config(
            config_module,
            enabled=False,
            actor_backend="megatron:d1p1t1",
            loss_name="cispo",
            fused_chunk_size=0,
        )
