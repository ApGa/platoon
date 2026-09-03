from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
CONFIG_NAME = (
    "abci_toolathlon_openhands_areal_prealloc_15node_recursive_h200_"
    "qwen36_moe_cp1_c24_s3_v34_behavior_usage_gated_partial"
)
BASE_NAME = (
    "abci_toolathlon_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen36_moe_cp2_c12_v33_behavior_usage_gated"
)
CONFIG = CONFIG_ROOT / f"{CONFIG_NAME}.yaml"
BASE_CONFIG = CONFIG_ROOT / f"{BASE_NAME}.yaml"
LAUNCHER = (
    REPO_ROOT
    / "pbs-scripts/openreward-toolathlon-15node-recursive-qwen36-cp1-"
    "behavior-usage-partial-64k-h200-v34-abci.pbs"
)
COMMON = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
NODE_SERVICES = REPO_ROOT / "pbs-scripts/openreward-node-services.sh"
ENV = REPO_ROOT / "pbs-scripts/openreward-v34-abci.env"

EXPERIMENT = "abci-or-ta15-q36-cp1-behavior-usage-partial-v34"
TRIAL = "abci-ta15-rec-q36m-cp1-c24-s3-d2-behavior-usage-partial-v34-t0"
BINDING = "qwen36-15-v34-behavior-usage-partial64"


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
        raise AssertionError(f"config default cycle: {ancestry + (name,)}")
    value = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text()) or {}
    defaults = value.pop("defaults", [])
    result: dict = {}
    for item in defaults:
        if item == "_self_":
            result = _merge(result, value)
        elif isinstance(item, str):
            result = _merge(result, _compose(item, ancestry + (name,)))
        else:
            raise AssertionError(f"unsupported config default in {name}: {item!r}")
    if "_self_" not in defaults:
        result = _merge(result, value)
    return result


def _actor_allocations(
    backend: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    match = re.fullmatch(
        r"megatron:\(attn:d(\d+)p(\d+)t(\d+)c(\d+)\|"
        r"ffn:d(\d+)p(\d+)t(\d+)e(\d+)\)",
        backend,
    )
    assert match is not None, backend
    values = tuple(int(value) for value in match.groups())
    return values[:4], values[4:]


def _rollout_allocation(backend: str) -> tuple[int, int]:
    match = re.fullmatch(r"sglang:d(\d+)p(\d+)t(\d+)", backend)
    assert match is not None, backend
    data, pipeline, tensor = (int(value) for value in match.groups())
    return data * pipeline * tensor, data


def test_v34_config_has_exact_15_node_cp1_topology_and_queue_contract() -> None:
    yaml = pytest.importorskip("yaml")
    leaf = yaml.safe_load(CONFIG.read_text())
    config = _compose(CONFIG_NAME)

    assert leaf["defaults"] == [BASE_NAME, "_self_"]
    assert config["experiment_name"] == EXPERIMENT
    assert config["trial_name"] == TRIAL
    assert config["cluster"]["n_nodes"] == 15
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["scheduler"]["type"] == "pbs_prealloc"
    assert config["cluster"]["fileroot"] == "${oc.env:PLATOON_EXPERIMENT_ROOT}"

    attn, ffn = _actor_allocations(config["actor"]["backend"])
    assert attn == (16, 1, 2, 1)
    assert ffn == (4, 1, 1, 8)
    assert config["ref"]["backend"] == "${actor.backend}"
    train_gpus = attn[0] * attn[1] * attn[2] * attn[3]
    assert train_gpus == ffn[0] * ffn[1] * ffn[2] * ffn[3] == 32

    rollout_gpus, rollout_replicas = _rollout_allocation(config["rollout"]["backend"])
    assert (rollout_gpus, rollout_replicas) == (88, 11)
    assert train_gpus + rollout_gpus == config["cluster"]["n_nodes"] * 8
    assert config["train_dataset"]["batch_size"] == 8
    assert config["rollout"]["consumer_batch_size"] == 8
    assert config["workflow_config"]["group_size"] == 8
    assert config["rollout"]["max_concurrent_rollouts"] == 24
    assert config["rollout"]["max_head_offpolicyness"] == 3
    assert config["rollout"]["ensure_batch_divisible_by"] == 16

    assert config["rollout"]["agent"]["engine_max_tokens"] == 65_536
    assert config["actor"]["mb_spec"]["max_tokens_per_mb"] == 65_536
    assert config["ref"]["mb_spec"]["max_tokens_per_mb"] == 65_536
    assert config["sglang"]["context_length"] == 65_537
    assert config["sglang"]["max_prefill_tokens"] == 65_536


def test_v34_config_enables_validated_fused_r3_full_checkpoint_contract() -> None:
    config = _compose(CONFIG_NAME)
    actor = config["actor"]
    megatron = actor["megatron"]

    assert actor["gradient_checkpointing"] is True
    assert megatron["recompute_granularity"] == "full"
    assert megatron["recompute_method"] == "uniform"
    assert megatron["recompute_num_layers"] == 1
    assert megatron["distribute_saved_activations"] is None
    assert megatron["recompute_modules"] is None

    assert actor["enable_cispo_fused_linear_logprobs"] is True
    assert actor["cispo_fused_linear_chunk_size"] == 512
    assert megatron["enable_fp32_lm_head"] is False
    assert config["ref"]["megatron"]["enable_fp32_lm_head"] is True
    assert megatron["moe_permute_fusion"] is True
    assert megatron["moe_router_fusion"] is False
    assert megatron["use_deterministic_algorithms"] is True

    assert actor["enable_router_replay"] is True
    assert actor["router_replay_num_layers"] == 40
    assert actor["router_replay_topk"] == 8
    assert actor["router_replay_num_experts"] == 256
    assert config["rollout"]["return_routed_experts"] is True
    assert megatron["enable_mtp"] is False


def test_v34_config_preserves_both_child_gates_and_toolathlon_only() -> None:
    config = _compose(CONFIG_NAME)
    openreward = config["openreward"]

    assert config["workflow_config"]["rollout_config"]["propagate_root_success"] is False
    assert openreward["enable_recursive_subagents"] is True
    assert openreward["subagent_max_depth"] == 2
    assert openreward["enable_subagent_reward_judging"] is True
    assert openreward["enable_subagent_behavior_judging"] is True
    assert openreward["enable_subagent_parent_usage_eligibility_gate"] is True
    assert openreward["subagent_parent_usage_judge_max_concurrency"] == 4
    assert [entry["label"] for entry in openreward["environments"]] == ["toolathlon"]


def test_v34_launcher_is_fresh_immutable_and_attests_exact_inputs() -> None:
    source = LAUNCHER.read_text()

    assert os.stat(LAUNCHER).st_mode & 0o111
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert "#PBS -l select=15:mpiprocs=8" in source
    assert "RELEASE_REF=refs/tags/abci-v8r34-07bebf9-057108f" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r34-07bebf9-057108f" in source
    assert f"CONFIG_SHA256={hashlib.sha256(CONFIG.read_bytes()).hexdigest()}" in source
    assert f"BASE_CONFIG_SHA256={hashlib.sha256(BASE_CONFIG.read_bytes()).hexdigest()}" in source
    assert "require_component external/AReaL d99124ec15102ca2fcd4960cc8beaef3950c2672" in source
    assert "require_component external/openhands-agent-sdk 057108f1f1235940e5d04401b4c294e4777c3baf" in source
    assert "require_component external/swe-rebench-v2-openrewardenv 538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7" in source
    assert "require_component external/toolathlon-gym 18e62c0d0419da88209af911528f2f8a0b44a5ed" in source
    assert f"EXPERIMENT={EXPERIMENT}" in source
    assert f"TRIAL={TRIAL}" in source
    assert f"FRESH_BINDING={BINDING}" in source
    assert "OPENREWARD_EXPECTED_NNODES=15" in source
    assert "OPENREWARD_SERVICE_NODE_COUNT=11" in source
    assert "OPENREWARD_EXPECTED_MODEL_SHARDS=26" in source
    assert "OPENREWARD_EXPECTED_RECURSIVE=1" in source
    assert "OPENREWARD_EXPECTED_PTC=1" in source
    assert "OPENREWARD_ENVIRONMENT_SET=toolathlon_only" in source
    assert "OPENREWARD_NODE_MEMORY_GUARD_ENABLED=0" in source
    assert "refusing stale checkpoint" in source
    assert "refusing stale success artifact" in source
    assert "OPENREWARD_REQUIRE_EXISTING_RECOVERY" in source
    assert 'exec "${COMMON}" "${CONFIG}"' in source
    assert "openreward_update_gate.py" not in source
    assert "qsub" not in source.lower()


def test_v34_partial_reward_mode_is_fail_closed_across_service_boundary() -> None:
    launcher = LAUNCHER.read_text()
    common = COMMON.read_text()
    services = NODE_SERVICES.read_text()

    mode_export = "export OPENREWARD_REWARD_MODE=partial"
    service_start = (
        'echo "[pbs-openreward] starting one Toolathlon service on '
        "${SERVICE_NODE_COUNT}"
    )
    pbsdsh_forward = '"OPENREWARD_REWARD_MODE=${OPENREWARD_REWARD_MODE}"'
    reward_capture = "REWARD_MODE=${OPENREWARD_REWARD_MODE:-}"
    reward_validation = '[[ "${REWARD_MODE}" == partial ]]'
    cleanenv_start = '"${RUNTIME}" exec --cleanenv --no-home --overlay'
    cleanenv_injection = 'OPENREWARD_REWARD_MODE="${REWARD_MODE}"'

    assert common.count(mode_export) == 1
    assert common.count(pbsdsh_forward) == 1
    assert common.index(mode_export) < common.index(service_start) < common.index(pbsdsh_forward)
    assert services.count(reward_capture) == 1
    assert services.count(reward_validation) == 1
    assert services.count(cleanenv_injection) == 1
    assert services.index(reward_validation) < services.index(cleanenv_start)
    assert services.index(cleanenv_start) < services.index(cleanenv_injection)

    for literal in (
        "Partial credit is a launch invariant",
        "require_literal_count \"${COMMON}\" 1 'export OPENREWARD_REWARD_MODE=partial'",
        'require_literal_count "${NODE_SERVICES}" 1 \'REWARD_MODE=${OPENREWARD_REWARD_MODE:-}\'',
        'require_literal_count "${NODE_SERVICES}" 1 \'        OPENREWARD_REWARD_MODE="${REWARD_MODE}" \\\'',
        "partial-reward export/forward/validation/cleanenv injection order is invalid",
    ):
        assert literal in launcher


def test_v34_fresh_identity_is_authorized_by_controller_and_supervisor() -> None:
    common = COMMON.read_text()
    services = NODE_SERVICES.read_text()

    assert common.count(BINDING) == 3
    assert services.count(BINDING) == 2
    for source in (common, services):
        assert f'"${{EXPECTED_EXPERIMENT}}" == {EXPERIMENT}' in source
        assert f'"${{EXPECTED_TRIAL}}" == {TRIAL}' in source
        assert "15:1:1:qwen36" in source
    subprocess.run(["bash", "-n", str(COMMON)], check=True)
    subprocess.run(["bash", "-n", str(NODE_SERVICES)], check=True)


def test_v34_env_reuses_v33_v9_pins_and_rebinds_only_v34_code() -> None:
    source = ENV.read_text()

    assert re.search(
        r"source \"\$\{_OPENREWARD_V34_BASE_ENV\}\".*"
        r"PLATOON_REPO_ROOT=/home/aci18912it/rasp/releases/"
        r"platoon-v8r34-07bebf9-057108f",
        source,
        flags=re.DOTALL,
    )
    assert (
        "_OPENREWARD_V34_BASE_ENV=/home/aci18912it/rasp/releases/"
        "platoon-v8r33-07bebf9-057108f/pbs-scripts/openreward-v9-abci.env"
    ) in source
    assert os.stat(ENV).st_mode & 0o111
    subprocess.run(["bash", "-n", str(ENV)], check=True)
