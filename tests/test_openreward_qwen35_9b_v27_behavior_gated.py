from __future__ import annotations

import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
CONFIG_NAME = (
    "abci_toolathlon_openhands_areal_prealloc_16node_recursive_h200_"
    "qwen35_9b_dp64_c12_v27_behavior_gated"
)
BASE = (
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen36_moe_cp2_c24_ta20_v26_fresh_d2_noeff"
)


def _merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return overlay


def _compose(name: str, ancestry: tuple[str, ...] = ()) -> dict:
    yaml = pytest.importorskip("yaml")
    if name in ancestry:
        raise AssertionError(f"recursive defaults: {name}")
    leaf = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text())
    defaults = leaf.pop("defaults", [])
    config: dict = {}
    for default in defaults:
        if default == "_self_":
            config = _merge(config, leaf)
        else:
            config = _merge(config, _compose(default, (*ancestry, name)))
    if "_self_" not in defaults:
        config = _merge(config, leaf)
    return config


def test_qwen35_9b_v27_leaf_composes_exact_dp64_runtime_contract() -> None:
    yaml = pytest.importorskip("yaml")
    literal = yaml.safe_load((CONFIG_ROOT / f"{CONFIG_NAME}.yaml").read_text())
    config = _compose(CONFIG_NAME)

    assert literal["defaults"] == [BASE, "_self_"]
    assert config["experiment_name"] == "abci-or-ta16-q35-9b-dp-behavior-v27"
    assert config["trial_name"] == "abci-ta16-rec-q359b-dp-c12-d2-behavior-v27-t0"
    assert config["stats_logger"]["wandb"]["group"] == (
        "abci-ta16-rec-q359b-dp-c12-d2-behavior-v27"
    )

    assert config["cluster"]["n_nodes"] == 16
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["actor"]["backend"] == "megatron:d64p1t1c1"
    assert config["ref"]["backend"] == "${actor.backend}"
    assert config["rollout"]["backend"] == "sglang:d64p1t1"
    assert 64 + 64 == config["cluster"]["n_nodes"] * 8

    assert config["actor"]["enable_router_replay"] is False
    assert config["workflow_config"]["enable_router_replay"] is False
    assert config["rollout"]["return_routed_experts"] is False
    assert config["actor"]["megatron"]["enable_mtp"] is False
    assert config["actor"]["megatron"]["enable_fp32_lm_head"] is False
    assert config["ref"]["megatron"]["enable_fp32_lm_head"] is False
    assert config["actor"]["megatron"]["use_deterministic_algorithms"] is False
    assert config["sglang"]["disable_radix_cache"] is False

    for engine in (config["actor"], config["ref"]):
        assert engine["mb_spec"] == {
            "n_mbs": 1,
            "granularity": 1,
            "max_tokens_per_mb": 65_536,
            "n_mbs_divisor": 1,
        }
    assert config["rollout"]["agent"]["engine_max_tokens"] == 65_536
    assert config["sglang"]["context_length"] == 65_537
    assert config["sglang"]["max_prefill_tokens"] == 65_536
    assert config["openreward"]["condenser_max_tokens"] == 52_428


def test_qwen35_9b_v27_dp64_floor_is_protected_by_bounded_recollection() -> None:
    config = _compose(CONFIG_NAME)
    workflow = config["workflow_config"]
    rollout = config["rollout"]
    actor = config["actor"]

    dispatch_dp = 64
    pp = 1
    ppo_minibatches = actor["ppo_n_minibatches"]
    granularity = actor["mb_spec"]["granularity"]
    engine_divisor = math.lcm(pp, actor["mb_spec"]["n_mbs_divisor"])
    engine_min = math.ceil(
        max(actor["mb_spec"]["n_mbs"], 1) / engine_divisor
    ) * engine_divisor
    actor_quantum = dispatch_dp * ppo_minibatches * granularity
    divisor = math.lcm(
        actor_quantum * engine_divisor,
        rollout["ensure_batch_divisible_by"],
    )
    minimum = math.ceil(
        actor_quantum * engine_min / divisor
    ) * divisor

    assert (engine_divisor, engine_min, divisor, minimum) == (1, 1, 64, 64)
    assert config["dynamic_bs"] is True
    assert config["train_dataset"]["batch_size"] == 8
    assert workflow["group_size"] == 8
    assert workflow["min_successful_group_size"] == 5
    assert workflow["dynamic_batch_recollection_max_rounds"] == 2
    assert workflow["straggler_timeout_seconds"] == 900
    assert workflow["straggler_quorum"] == 6
    assert workflow["subprocess_shutdown_grace_seconds"] == 10
    full_recollection_root_capacity = (
        config["train_dataset"]["batch_size"]
        * workflow["min_successful_group_size"]
        * workflow["dynamic_batch_recollection_max_rounds"]
    )
    assert full_recollection_root_capacity == 80 >= minimum
    assert rollout["consumer_batch_size"] == (
        config["train_dataset"]["batch_size"]
        * workflow["dynamic_batch_recollection_max_rounds"]
    ) == 16
    assert rollout["max_concurrent_rollouts"] == 12
    assert workflow["filter_zero_advantage_datums"] is False
    assert workflow["filter_zero_variance_groups"] is False

    # Dynamic sizing can return fewer accepted tasks than a full chunk. The
    # trainer must therefore fail closed after bounded recollection instead of
    # dispatching an undersized global batch to DP64.
    trainer_source = (REPO_ROOT / "platoon/train/areal/rl.py").read_text()
    assert "target_datums=actor_batch_requirements.minimum_datums" in trainer_source
    assert "if global_trainable < min_per_step:" in trainer_source
    assert "if total < requirements.minimum_datums:" in trainer_source


def test_qwen35_9b_v27_uses_toolathlon_behavior_gate_reward_contract() -> None:
    config = _compose(CONFIG_NAME)
    workflow = config["workflow_config"]
    openreward = config["openreward"]

    assert [environment["label"] for environment in openreward["environments"]] == [
        "toolathlon"
    ]
    assert openreward["environments"][0]["env_name"] == "toolathlongym"
    assert openreward["subagent_environment_access"] == "shared"
    assert openreward["subagent_default_max_steps"] == 100
    assert openreward["subagent_max_depth"] == 2
    assert openreward["enable_subagent_reward_judging"] is True
    assert openreward["subagent_reward_judge_max_steps"] == 100
    assert openreward["enable_subagent_behavior_judging"] is True
    assert openreward["subagent_behavior_judge_max_prompt_tokens"] == 24_576
    assert openreward["subagent_behavior_judge_max_output_tokens"] == 4_096
    assert openreward["subagent_behavior_judge_timeout_seconds"] == 300.0
    assert openreward["subagent_delegation_reward_coefficient"] == 0.0
    assert workflow["subagent_datum_keep_probability"] == 0.25
    assert workflow["rollout_config"]["max_steps"] == 200
    assert workflow["rollout_config"]["timeout"] == 3600
    assert workflow["rollout_config"]["step_timeout"] == 2700
    assert workflow["rollout_config"]["propagate_root_success"] is False
    assert workflow["token_efficiency_reward"]["enabled"] is False
    assert config["valid_dataset"] is None
    assert config["evaluator"]["eval_before_train"] is False
