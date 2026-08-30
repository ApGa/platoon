from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
BASE = (
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen36_moe_cp2_c24_ta20_v26_fresh_d2_noeff"
)
CONFIGS = {
    "rootprop": (
        "abci_toolathlon_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen36_moe_cp2_c12_v27_rootprop"
    ),
    "behavior_gated": (
        "abci_toolathlon_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen36_moe_cp2_c12_v27_behavior_gated"
    ),
}


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


@pytest.mark.parametrize("strategy", CONFIGS)
def test_v27_qwen36_reward_leaf_preserves_abci_64k_topology(strategy: str) -> None:
    yaml = pytest.importorskip("yaml")
    name = CONFIGS[strategy]
    literal = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text())
    config = _compose(name)

    assert literal["defaults"] == [BASE, "_self_"]
    assert config["cluster"]["n_nodes"] == 32
    assert config["rollout"]["backend"] == "sglang:d12p1t8"
    assert config["rollout"]["max_concurrent_rollouts"] == 12
    assert config["actor"]["backend"] == (
        "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    )
    assert config["actor"]["enable_router_replay"] is True
    assert config["rollout"]["return_routed_experts"] is True
    assert config["actor"]["megatron"]["use_deterministic_algorithms"] is True

    assert config["rollout"]["agent"]["engine_max_tokens"] == 65_536
    assert config["actor"]["mb_spec"]["max_tokens_per_mb"] == 65_536
    assert config["ref"]["mb_spec"]["max_tokens_per_mb"] == 65_536
    assert config["sglang"]["context_length"] == 65_537
    assert config["sglang"]["max_prefill_tokens"] == 65_536
    assert config["openreward"]["condenser_max_tokens"] == 52_428


@pytest.mark.parametrize("strategy", CONFIGS)
def test_v27_qwen36_reward_leaf_has_shared_toolathlon_contract(strategy: str) -> None:
    config = _compose(CONFIGS[strategy])
    workflow = config["workflow_config"]
    openreward = config["openreward"]

    assert config["train_dataset"]["batch_size"] == 8
    assert config["rollout"]["consumer_batch_size"] == 8
    assert workflow["group_size"] == 8
    assert workflow["straggler_timeout_seconds"] == 900
    assert workflow["straggler_quorum"] == 6
    assert workflow["subprocess_shutdown_grace_seconds"] == 10
    assert workflow["min_successful_group_size"] == 4
    assert workflow["subagent_datum_keep_probability"] == 0.25
    assert workflow["rollout_config"]["max_steps"] == 200
    assert workflow["rollout_config"]["timeout"] == 3600
    assert workflow["rollout_config"]["step_timeout"] == 2700
    assert workflow["token_efficiency_reward"]["enabled"] is False

    assert openreward["subagent_environment_access"] == "shared"
    assert openreward["subagent_default_max_steps"] == 100
    assert openreward["subagent_max_depth"] == 2
    assert openreward["subagent_delegation_reward_coefficient"] == 0.0
    assert [env["label"] for env in openreward["environments"]] == ["toolathlon"]
    assert openreward["environments"][0]["env_name"] == "toolathlongym"
    assert config["valid_dataset"] is None
    assert config["evaluator"]["eval_before_train"] is False


def test_v27_rootprop_disables_judges_and_propagates_root_success() -> None:
    config = _compose(CONFIGS["rootprop"])

    assert config["openreward"]["enable_subagent_reward_judging"] is False
    assert config["openreward"]["enable_subagent_behavior_judging"] is False
    assert config["workflow_config"]["rollout_config"]["propagate_root_success"] is True


def test_v27_behavior_gate_uses_same_policy_verifier_contract() -> None:
    config = _compose(CONFIGS["behavior_gated"])
    openreward = config["openreward"]

    assert openreward["enable_subagent_reward_judging"] is True
    assert openreward["subagent_reward_judge_max_steps"] == 100
    assert openreward["enable_subagent_behavior_judging"] is True
    assert openreward["subagent_behavior_judge_max_prompt_tokens"] == 24_576
    assert openreward["subagent_behavior_judge_max_output_tokens"] == 4_096
    assert openreward["subagent_behavior_judge_timeout_seconds"] == 300.0
    assert config["workflow_config"]["rollout_config"]["propagate_root_success"] is False


def test_v27_reward_strategy_identities_are_distinct() -> None:
    configs = [_compose(name) for name in CONFIGS.values()]

    assert len({config["experiment_name"] for config in configs}) == 2
    assert len({config["trial_name"] for config in configs}) == 2
    assert len({config["stats_logger"]["wandb"]["group"] for config in configs}) == 2
