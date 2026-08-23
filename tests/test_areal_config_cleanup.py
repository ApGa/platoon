"""Focused tests for Platoon's AReaL config cleanup."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rollout_root_success_spelling_and_legacy_config_compatibility():
    from platoon.config_defs import RolloutConfig

    canonical = RolloutConfig(propagate_root_success=True)
    assert canonical.propagate_root_success is True

    legacy_schema = OmegaConf.merge(
        OmegaConf.structured(RolloutConfig),
        OmegaConf.create({"propogate_root_success": True}),
    )
    legacy = OmegaConf.to_object(legacy_schema)
    assert legacy.propagate_root_success is True

    with pytest.raises(ValueError, match="Conflicting rollout propagation settings"):
        RolloutConfig(
            propagate_root_success=False,
            propogate_root_success=True,
        )


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

    areal_mod = types.ModuleType("areal")
    areal_api_mod = types.ModuleType("areal.api")
    cli_args_mod = types.ModuleType("areal.api.cli_args")
    sys.modules["areal"] = areal_mod
    sys.modules["areal.api"] = areal_api_mod
    sys.modules["areal.api.cli_args"] = cli_args_mod

    @dataclass
    class FakeSchedulerConfig:
        type: str | None = None

    @dataclass
    class FakeOpenAIConfig:
        mode: str = "offline"
        admin_api_key: str = "test-key"

    @dataclass
    class FakeMegatronEngineConfig:
        enable_mtp: bool = False
        moe_router_fusion: bool = False
        moe_z_loss_coeff: float | None = None
        recompute_granularity: str | None = "full"
        recompute_method: str | None = "uniform"

    @dataclass
    class FakePPOActorConfig:
        backend: str = ""
        behave_imp_weight_cap: float | None = None
        megatron: FakeMegatronEngineConfig = field(default_factory=FakeMegatronEngineConfig)
        recompute_logprob: bool = False
        use_decoupled_loss: bool = False
        gradient_checkpointing: bool = True

        def __post_init__(self):
            pass

        def should_compute_prox_logp(self):
            return self.recompute_logprob or self.use_decoupled_loss

    @dataclass
    class FakeInferenceEngineConfig:
        backend: str = ""
        experiment_name: str | None = None
        trial_name: str | None = None
        fileroot: str | None = None
        tokenizer_path: str = ""
        openai: FakeOpenAIConfig | None = None
        scheduling_strategy: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)
        return_routed_experts: bool = False

    @dataclass
    class FakeGRPOConfig:
        scheduler: FakeSchedulerConfig = field(default_factory=FakeSchedulerConfig)
        # Match the optional engine fields supplied by AReaL's real PPOConfig,
        # which Platoon's post-init visits to propagate worker allocator flags.
        critic: FakePPOActorConfig | None = None
        teacher: FakePPOActorConfig | None = None

        def __post_init__(self):
            pass

    for cls in (
        FakeSchedulerConfig,
        FakeOpenAIConfig,
        FakeMegatronEngineConfig,
        FakePPOActorConfig,
        FakeInferenceEngineConfig,
        FakeGRPOConfig,
    ):
        cls.__module__ = "areal.api.cli_args"

    cli_args_mod.FakeSchedulerConfig = FakeSchedulerConfig
    cli_args_mod.FakeOpenAIConfig = FakeOpenAIConfig
    cli_args_mod.FakeMegatronEngineConfig = FakeMegatronEngineConfig
    cli_args_mod.FakePPOActorConfig = FakePPOActorConfig
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
    serialized = asdict(parsed)
    assert serialized["actor"]["loss_fn"] == "cispo"
    assert serialized["actor"]["loss_fn_kwargs"]["clip_high_threshold"] == 4.2
    assert parsed.scheduler.type == "local"
    assert parsed.eval_gconfig.lora_name == parsed.gconfig.lora_name
    assert parsed.workflow_config.filter_zero_advantage_datums is True
    assert not hasattr(parsed.train_dataset, "path")
    assert not hasattr(parsed.valid_dataset, "type")


def test_token_efficiency_reward_config_parses_and_validates():
    config_mod = _load_config_module()

    workflow = config_mod.WorkflowConfig(
        token_efficiency_reward={
            "enabled": True,
            "coefficient": 0.05,
            "reference_tokens": 20_000,
            "max_penalty": 0.2,
            "input_token_weight": 0.01,
            "output_token_weight": 1.0,
            "attribution": "policy_subtree",
        }
    )
    assert workflow.token_efficiency_reward.enabled is True
    assert workflow.token_efficiency_reward.reference_tokens == 20_000
    assert workflow.token_efficiency_reward.input_token_weight == 0.01

    with pytest.raises(ValueError, match="reference_tokens must be positive"):
        config_mod.TokenEfficiencyRewardConfig(enabled=True, reference_tokens=0)
    with pytest.raises(ValueError, match="at least one positive token weight"):
        config_mod.TokenEfficiencyRewardConfig(
            enabled=True,
            input_token_weight=0,
            output_token_weight=0,
        )


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


def _router_replay_payload() -> dict:
    return {
        "rollout": {
            "backend": "sglang:d6p1t8",
            "return_routed_experts": True,
        },
        "actor": {
            "backend": "megatron:(attn:d5p2t4c2|ffn:d5p2t1e8)",
            "enable_router_replay": True,
            "router_replay_num_layers": 40,
            "router_replay_topk": 8,
            "router_replay_num_experts": 256,
            "megatron": {"enable_mtp": False},
        },
    }


def test_router_replay_structured_config_requires_safe_backends_and_capture():
    config_mod = _load_config_module()
    schema = OmegaConf.structured(config_mod.PlatoonArealRLTrainerConfig)
    parsed = OmegaConf.to_object(OmegaConf.merge(schema, OmegaConf.create(_router_replay_payload())))

    assert parsed.actor.enable_router_replay is True
    assert parsed.workflow_config.enable_router_replay is True
    assert parsed.workflow_config.router_replay_num_layers == 40
    assert parsed.workflow_config.router_replay_topk == 8
    assert parsed.actor.megatron.enable_mtp is False
    assert parsed.actor.megatron.moe_z_loss_coeff is None


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor.backend", "fsdp:d5p2t4c2", "Megatron actor backend"),
        ("rollout.return_routed_experts", False, "return_routed_experts=true"),
        ("actor.megatron.enable_mtp", True, "enable_mtp=false"),
        ("actor.recompute_logprob", True, "recomputation to be disabled"),
        ("actor.megatron.recompute_method", "block", "recompute_method=uniform"),
    ],
)
def test_router_replay_structured_config_rejects_unsafe_combinations(path, value, message):
    config_mod = _load_config_module()
    schema = OmegaConf.structured(config_mod.PlatoonArealRLTrainerConfig)
    payload = OmegaConf.create(_router_replay_payload())
    OmegaConf.update(payload, path, value, merge=False)

    with pytest.raises(ValueError, match=message):
        OmegaConf.to_object(OmegaConf.merge(schema, payload))


def test_r3_yaml_composes_over_baseline_without_changing_moe_coefficients():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    baseline_path = config_dir / "toolathlon_openhands_areal_prealloc_16node-cp.yaml"
    derived_path = config_dir / "toolathlon_openhands_areal_prealloc_16node-cp-r3.yaml"
    baseline = OmegaConf.load(baseline_path)
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=derived_path.stem)

    assert composed.rollout.return_routed_experts is True
    assert composed.actor.enable_router_replay is True
    assert composed.actor.router_replay_num_layers == 40
    assert composed.actor.router_replay_topk == 8
    assert composed.actor.router_replay_num_experts == 256
    assert composed.actor.megatron.enable_mtp is False
    # Toolathlon uses custom textual stop sequences; SGLang needs its tokenizer
    # initialized to decode tail strings. Exact S-1 route validation guards
    # alignment instead of PR #1207's skip-tokenizer setting.
    assert composed.sglang.skip_tokenizer_init is False
    assert composed.actor.backend == "megatron:(attn:d5p2t4c2|ffn:d5p2t1e8)"
    assert composed.actor.path == "Qwen/Qwen3.6-35B-A3B"
    assert "moe_aux_loss_coeff" not in composed.actor.megatron
    assert "moe_z_loss_coeff" not in composed.actor.megatron
    assert "enable_router_replay" not in baseline.actor
    assert "return_routed_experts" not in baseline.rollout

    # Match the launcher's pre-Hydra gate: a derived file must carry its own
    # literal top-level marker even though Hydra later merges inherited values.
    assert any(line.startswith("openreward:") for line in derived_path.read_text().splitlines())
    assert composed.openreward.env_name == baseline.openreward.env_name


def test_judged_recursive_r3_fp32_yaml_composes_all_features():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = "toolathlon_openhands_areal_prealloc_16node-cp-ptc-recursive-judged-r3-fp32-lm-head"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    assert composed.openreward.enable_programmatic_tool_calling is True
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.subagent_default_max_steps == 50
    assert composed.openreward.subagent_max_depth == 2
    assert composed.openreward.enable_subagent_reward_judging is True
    assert composed.openreward.subagent_reward_judge_max_steps == 50
    assert composed.openreward.subagent_delegation_reward_coefficient == 0.4
    assert composed.trial_name.endswith("trial2")
    assert composed.workflow_config.depth_level_weighting is True
    assert composed.workflow_config.leave_one_out_baseline is True
    assert composed.workflow_config.rollout_config.propagate_root_success is False
    assert composed.workflow_config.rollout_config.timeout == 5400
    assert composed.workflow_config.rollout_config.step_timeout == 2700
    assert composed.rollout.return_routed_experts is True
    assert composed.rollout.request_timeout == 12600
    assert composed.rollout.agent.session_timeout_seconds == 7200
    subprocess_hard_timeout = composed.workflow_config.rollout_config.timeout + 180
    assert 2 * subprocess_hard_timeout < composed.rollout.request_timeout
    assert composed.sglang.skip_tokenizer_init is False
    assert composed.actor.enable_router_replay is True
    assert composed.actor.router_replay_num_layers == 40
    assert composed.actor.router_replay_topk == 8
    assert composed.actor.router_replay_num_experts == 256
    assert composed.actor.megatron.enable_mtp is False
    assert composed.actor.megatron.enable_fp32_lm_head is True
    assert composed.ref.megatron.enable_fp32_lm_head is True
    assert "moe_aux_loss_coeff" not in composed.actor.megatron
    assert "moe_z_loss_coeff" not in composed.actor.megatron


def test_mixed_recursive_r3_fp32_yaml_composes_balancing_and_child_policies():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = (
        "toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-ptc-recursive-"
        "r3-fp32-lm-head"
    )
    config_path = config_dir / f"{config_name}.yaml"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    literal = OmegaConf.load(config_path)
    assert literal.defaults[0] == "toolathlon_tmax_swe_openhands_areal_prealloc_16node-cp-r3-fp32-lm-head"
    assert "openreward" in literal
    assert "actor" in literal

    environments = {
        environment.label: environment
        for environment in composed.openreward.environments
    }
    assert list(environments) == ["toolathlon", "tmax", "swe_rebench"]
    assert {label: environment.sampling_weight for label, environment in environments.items()} == {
        "toolathlon": 1.0,
        "tmax": 1.0,
        "swe_rebench": 1.0,
    }
    assert all(
        "subagent_environment_access" not in environment
        for environment in environments.values()
    )

    assert composed.openreward.enable_programmatic_tool_calling is True
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.subagent_environment_access == "shared"
    assert composed.openreward.subagent_default_max_steps == 50
    assert composed.openreward.subagent_max_depth == 2
    assert composed.openreward.balance_accepted_batches is False
    assert composed.train_dataset.batch_size == 8
    assert composed.rollout.consumer_batch_size == 8
    assert composed.workflow_config.group_size == 8
    assert composed.workflow_config.straggler_quorum == 6
    assert composed.workflow_config.min_successful_group_size == 4
    assert composed.rollout.request_timeout == 12600
    assert composed.rollout.agent.session_timeout_seconds == 7200
    assert composed.recover.freq_epochs is None
    assert composed.recover.freq_steps == 1
    assert composed.recover.freq_secs is None
    assert composed.workflow_config.depth_level_weighting is True
    assert composed.workflow_config.depth_level_discount_gamma is None
    assert composed.workflow_config.leave_one_out_baseline is True
    assert composed.workflow_config.rollout_config.propagate_root_success is True
    assert composed.trial_name.endswith("ptc-recursive-r3-fp32-lm-head-trial0")

    assert composed.rollout.return_routed_experts is True
    assert composed.actor.enable_router_replay is True
    assert composed.actor.megatron.enable_mtp is False
    assert composed.actor.megatron.enable_fp32_lm_head is True
    assert composed.ref.megatron.enable_fp32_lm_head is True


def test_32node_judged_recursive_config_has_balanced_topology_and_timeouts():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-fp32-lm-head"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    assert composed.cluster.n_nodes == 32
    assert composed.trial_name == (
        "toolathlon-openhands-32node-qwen3.6-35B-ptc-recursive-judged-r3-fp32-lm-head-trial3"
    )
    assert composed.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert composed.ref.backend == composed.actor.backend
    assert composed.rollout.backend == "sglang:d12p1t8"
    assert composed.rollout.max_concurrent_rollouts == 12
    assert composed.rollout.consumer_batch_size == 4
    assert composed.train_dataset.batch_size == 4
    assert composed.rollout.request_timeout == 12600
    subprocess_hard_timeout = composed.workflow_config.rollout_config.timeout + 180
    assert subprocess_hard_timeout < composed.rollout.request_timeout
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.enable_subagent_reward_judging is True
    assert composed.actor.enable_router_replay is True
    assert composed.actor.megatron.enable_fp32_lm_head is True
    assert composed.ref.megatron.enable_fp32_lm_head is True


def test_32node_recursive_bs8_config_composes_all_features_with_preserved_thinking():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = (
        "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-"
        "fp32-lm-head-bs8"
    )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    model_path = "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    assert composed.cluster.n_nodes == 32
    assert composed.trial_name.endswith("fp32-lm-head-bs8-trial2")
    assert composed.train_dataset.batch_size == 8
    assert composed.rollout.consumer_batch_size == 8
    assert composed.workflow_config.group_size == 8
    assert composed.workflow_config.subagent_datum_keep_probability == 0.25
    assert composed.workflow_config.subagent_datum_sampling_seed == composed.seed
    assert composed.rollout.max_concurrent_rollouts == 12
    assert composed.rollout.max_head_offpolicyness == 3
    assert composed.rollout.scheduling_spec[0].mem == 128
    assert composed.actor.scheduling_spec[0].mem == 32
    assert composed.workflow_config.straggler_quorum == 6
    assert composed.workflow_config.straggler_timeout_seconds == 900
    assert composed.workflow_config.subprocess_shutdown_grace_seconds == 10
    assert composed.workflow_config.min_successful_group_size == 4
    assert composed.workflow_config.rollout_config.timeout == 3600
    assert composed.workflow_config.rollout_config.step_timeout == 2700
    assert composed.actor.path == model_path
    assert composed.ref.path == model_path
    assert composed.sglang.model_path == model_path
    assert composed.tokenizer_path == model_path
    assert composed.rollout.tokenizer_path == model_path
    assert composed.workflow_config.rollout_config.model_name == model_path
    assert composed.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert composed.ref.backend == composed.actor.backend
    assert composed.rollout.backend == "sglang:d12p1t8"
    assert composed.openreward.enable_programmatic_tool_calling is True
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.enable_subagent_reward_judging is True
    assert composed.openreward.subagent_delegation_reward_coefficient == 0.4
    assert composed.workflow_config.leave_one_out_baseline is True
    assert composed.actor.enable_router_replay is True
    assert composed.actor.megatron.enable_mtp is False
    assert composed.actor.megatron.enable_fp32_lm_head is True
    assert composed.ref.megatron.enable_fp32_lm_head is True


def test_32node_toolathlon_rootprop_config_composes_requested_ablation():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = (
        "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-rootprop-"
        "r3-fp32-lm-head-bs8"
    )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    assert composed.trial_name == "ta32-rec-rootprop-v1-trial0"
    assert composed.cluster.n_nodes == 32
    assert composed.openreward.env_name == "toolathlongym"
    assert OmegaConf.select(composed, "openreward.environments") is None
    assert composed.openreward.train_task_limit is None
    assert composed.openreward.eval_split is None
    assert composed.openreward.eval_task_limit is None
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.subagent_environment_access == "shared"
    assert composed.openreward.subagent_default_max_steps == 50
    assert composed.openreward.subagent_max_depth == 2
    assert composed.openreward.enable_subagent_reward_judging is False
    assert composed.openreward.subagent_delegation_reward_coefficient == 0.0

    workflow_config_type = _load_config_module().WorkflowConfig
    workflow = workflow_config_type(
        **OmegaConf.to_container(composed.workflow_config, resolve=True)
    )
    assert workflow.group_size == 8
    assert workflow.subagent_datum_keep_probability == 0.25
    assert workflow.rollout_config.propagate_root_success is True
    assert workflow.token_efficiency_reward.enabled is False
    assert workflow.straggler_quorum == 6
    assert workflow.straggler_timeout_seconds == 900
    assert workflow.min_successful_group_size == 4
    assert workflow.rollout_config.timeout == 3600
    assert workflow.rollout_config.step_timeout == 2700

    assert composed.train_dataset.batch_size == 8
    assert composed.rollout.backend == "sglang:d12p1t8"
    assert composed.rollout.max_concurrent_rollouts == 12
    assert composed.rollout.scheduling_spec[0].mem == 128
    assert composed.sglang.mem_fraction_static == 0.70
    assert composed.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert composed.actor.path == "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    assert composed.recover.freq_steps == 1
    assert composed.valid_dataset is None
    assert composed.evaluator.eval_before_train is False
    assert composed.stats_logger.wandb.project == "toolathlon-openhands"
    assert composed.stats_logger.wandb.group == "ta32-rec-rootprop-v1"


def test_32node_toolathlon_behavior_gate_config_composes_requested_ablation():
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    config_name = (
        "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-"
        "behavior-gated-r3-fp32-lm-head-bs8"
    )
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=config_name)

    assert composed.trial_name == "ta32-rec-behavior-gate-v1-trial0"
    assert composed.cluster.n_nodes == 32
    assert composed.openreward.env_name == "toolathlongym"
    assert OmegaConf.select(composed, "openreward.environments") is None
    assert composed.openreward.train_task_limit is None
    assert composed.openreward.eval_split is None
    assert composed.openreward.eval_task_limit is None
    assert composed.openreward.enable_recursive_subagents is True
    assert composed.openreward.subagent_environment_access == "shared"
    assert composed.openreward.subagent_default_max_steps == 50
    assert composed.openreward.subagent_max_depth == 2
    assert composed.openreward.enable_subagent_reward_judging is True
    assert composed.openreward.subagent_reward_judge_max_steps == 50
    assert composed.openreward.enable_subagent_behavior_judging is True
    assert composed.openreward.subagent_behavior_judge_max_prompt_tokens == 24_576
    assert composed.openreward.subagent_behavior_judge_max_output_tokens == 4_096
    assert (
        composed.openreward.subagent_behavior_judge_max_prompt_tokens
        + composed.openreward.subagent_behavior_judge_max_output_tokens
        <= 32_768
    )
    assert composed.openreward.subagent_behavior_judge_timeout_seconds == 300.0
    assert composed.openreward.subagent_delegation_reward_coefficient == 0.0

    workflow_config_type = _load_config_module().WorkflowConfig
    workflow = workflow_config_type(
        **OmegaConf.to_container(composed.workflow_config, resolve=True)
    )
    assert workflow.group_size == 8
    assert workflow.subagent_datum_keep_probability == 0.25
    assert workflow.rollout_config.propagate_root_success is False
    assert workflow.token_efficiency_reward.enabled is False
    assert workflow.straggler_quorum == 6
    assert workflow.straggler_timeout_seconds == 900
    assert workflow.min_successful_group_size == 4
    assert workflow.rollout_config.timeout == 3600
    assert workflow.rollout_config.step_timeout == 2700

    assert composed.train_dataset.batch_size == 8
    assert composed.rollout.backend == "sglang:d12p1t8"
    assert composed.rollout.max_concurrent_rollouts == 12
    assert composed.rollout.scheduling_spec[0].mem == 128
    assert composed.sglang.mem_fraction_static == 0.70
    assert composed.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert composed.actor.path == "apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    assert composed.recover.freq_steps == 1
    assert composed.valid_dataset is None
    assert composed.evaluator.eval_before_train is False
    assert composed.stats_logger.wandb.project == "toolathlon-openhands"
    assert composed.stats_logger.wandb.group == "ta32-rec-behavior-gate-v1"


def test_32node_recursive_wrapper_preserves_allocation_for_successors():
    script_name = "openreward-toolathlon-prealloc-32node-ptc-recursive.sh"
    script = (REPO_ROOT / "slurm-scripts" / script_name).read_text()

    assert "#SBATCH --nodes=32" in script
    assert "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-fp32-lm-head.yaml" in script
    assert f"OPENREWARD_JOB_SCRIPT=${{REPO_ROOT}}/slurm-scripts/{script_name}" in script
    assert '"${SLURM_NNODES:-32}" -ne 32' in script
    assert "OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}" in script


def test_32node_recursive_bs8_wrapper_preserves_experiment_for_successors():
    script_name = "openreward-toolathlon-prealloc-32node-ptc-recursive-bs8.sh"
    script = (REPO_ROOT / "slurm-scripts" / script_name).read_text()

    assert "#SBATCH --nodes=32" in script
    assert "#SBATCH --job-name=openreward-qwen36-32n-recursive-bs8" in script
    assert (
        "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-"
        "fp32-lm-head-bs8.yaml"
    ) in script
    assert f"OPENREWARD_JOB_SCRIPT=${{REPO_ROOT}}/slurm-scripts/{script_name}" in script
    assert '"${SLURM_NNODES:-32}" -ne 32' in script
    assert "OPENREWARD_CONTROLLER_CPUS=${OPENREWARD_CONTROLLER_CPUS:-64}" in script
    assert "NCCL_RAS_ENABLE=${NCCL_RAS_ENABLE:-0}" in script


def test_32node_rootprop_wrapper_is_toolathlon_only_and_self_contained():
    script_name = "openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-rootprop.sh"
    script_path = REPO_ROOT / "slurm-scripts" / script_name
    script = script_path.read_text()

    assert script_path.stat().st_mode & 0o111
    assert "#SBATCH --nodes=32" in script
    assert "openreward-toolathlon-prealloc-base.sh" in script
    assert "openreward-multienv" not in script
    assert "ta32-rec-rootprop-v1-trial0" in script
    assert "recursive-rootprop-r3-fp32-lm-head-bs8.yaml" in script
    assert f"OPENREWARD_JOB_SCRIPT=${{REPO_ROOT}}/slurm-scripts/{script_name}" in script
    assert "OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0" in script
    assert "OPENREWARD_WANDB_MODE=online" in script
    assert "apga+toolathlon-gym+18e62c0d041.sqsh" in script
    assert "OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800" in script
    assert "OPENREWARD_DEADLINE_SAFETY_SECONDS:-600" in script
    assert 'value=${value:1:${#value}-2}' in script


def test_32node_behavior_gate_wrapper_is_toolathlon_only_and_self_contained():
    script_name = (
        "openreward-toolathlon-prealloc-32node-ptc-recursive-bs8-"
        "behavior-gated.sh"
    )
    script_path = REPO_ROOT / "slurm-scripts" / script_name
    script = script_path.read_text()

    assert script_path.stat().st_mode & 0o111
    assert "#SBATCH --nodes=32" in script
    assert "openreward-toolathlon-prealloc-base.sh" in script
    assert "openreward-multienv" not in script
    assert "ta32-rec-behavior-gate-v1-trial0" in script
    assert "recursive-behavior-gated-r3-fp32-lm-head-bs8.yaml" in script
    assert f"OPENREWARD_JOB_SCRIPT=${{REPO_ROOT}}/slurm-scripts/{script_name}" in script
    assert "OPENREWARD_SUBAGENT_DELEGATION_REWARD_COEFFICIENT=0.0" in script
    assert "OPENREWARD_WANDB_MODE=online" in script
    assert "apga+toolathlon-gym+18e62c0d041.sqsh" in script
    assert "OPENREWARD_DEADLINE_INITIAL_STEP_SECONDS:-1800" in script
    assert "OPENREWARD_DEADLINE_SAFETY_SECONDS:-600" in script
    assert 'value=${value:1:${#value}-2}' in script
    assert "OPENREWARD_SUBAGENT_BEHAVIOR_JUDGE_MODEL" not in script
    assert "inference-api.nvidia.com" not in script


def test_prealloc_launcher_retries_plain_exit_one_with_bounded_atomic_successor():
    script = (REPO_ROOT / "slurm-scripts/openreward-toolathlon-prealloc.sh").read_text()

    assert "OPENREWARD_MAX_INFRA_RESTARTS=${OPENREWARD_MAX_INFRA_RESTARTS:-3}" in script
    assert 'elif [[ "${status}" -eq 1 ]]; then' in script
    assert 'restart_reason="trainer/controller runtime failure (exit 1)"' in script
    assert 'submit_successor "${restart_reason}" 1 || true' in script
    assert 'exit "${status}"' in script

    # The claim must remain per allocation so every member of a continuation
    # chain can create at most one successor without blocking later members.
    assert "OPENREWARD_JOB_STATE_DIR=${OPENREWARD_JOB_STATE_ROOT}/${JOB_INSTANCE_ID}" in script
    assert "SUCCESSOR_CLAIM_DIR=${OPENREWARD_JOB_STATE_DIR}/successor-claimed" in script
    assert 'if ! mkdir "${SUCCESSOR_CLAIM_DIR}" 2>/dev/null; then' in script
    assert '--dependency="afterany:${SLURM_JOB_ID}"' in script

    # Once the trainer has exited, signal handlers must not interrupt the small
    # claim-to-sbatch critical section. EXIT cleanup and the captured status stay.
    mask_position = script.index("trap '' TERM INT USR1", script.index("restart_reason="))
    submit_position = script.index('submit_successor "${restart_reason}" 1 || true')
    exit_position = script.index('exit "${status}"', submit_position)
    assert mask_position < submit_position < exit_position


@pytest.mark.parametrize(
    "filename",
    [
        "toolathlon_openhands_areal_prealloc_16node-cp-r3.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-fp32-lm-head.yaml",
        "toolathlon_openhands_areal_prealloc_16node-cp-r3-fp32-lm-head.yaml",
        ("toolathlon_openhands_areal_prealloc_16node-cp-ptc-recursive-judged-r3-fp32-lm-head.yaml"),
        ("toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-fp32-lm-head.yaml"),
        (
            "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-judged-r3-"
            "fp32-lm-head-bs8.yaml"
        ),
        (
            "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-rootprop-"
            "r3-fp32-lm-head-bs8.yaml"
        ),
        (
            "toolathlon_openhands_areal_prealloc_32node-cp-ptc-recursive-"
            "behavior-gated-r3-fp32-lm-head-bs8.yaml"
        ),
    ],
)
def test_derived_experiment_configs_pass_launcher_pre_hydra_checks(filename):
    config_dir = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
    path = config_dir / filename
    lines = path.read_text().splitlines()

    # OmegaConf's strict loader also catches accidental duplicate YAML keys.
    parsed = OmegaConf.load(path)
    assert sum(line.startswith("openreward:") for line in lines) == 1
    assert any(line.strip().startswith("backend: megatron") for line in lines)
    assert parsed.actor.backend.startswith("megatron:")
