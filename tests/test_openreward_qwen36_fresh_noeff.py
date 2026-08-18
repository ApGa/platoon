from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "pbs-scripts"
CONFIG_DIR = ROOT / "plugins/openreward/platoon/openreward/configs/areal"
CONFIG_NAME = (
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen36_moe_cp2_c24_ta20_v26_fresh_d2_noeff"
)
CONFIG = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
WRAPPER = (
    PBS
    / "openreward-ta-swe-32node-recursive-qwen36-cp2-fresh-d2-noeff-h200-v13-abci.pbs"
)
MANIFEST = PBS / "openreward-v9-release.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fresh_qwen36_noeff_config_composes_as_exact_ablation() -> None:
    hydra = pytest.importorskip("hydra")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = hydra.compose(config_name=CONFIG_NAME)

    assert config.experiment_name == "abci-or-ta-swe32-q36-cp2-effoff-v26"
    assert config.trial_name == "abci-ta-swe32-rec-q36m-cp2-c24-d2-effoff-v26-t0"
    assert config.cluster.n_nodes == 32
    assert config.dynamic_bs is False
    assert config.workflow_config.filter_zero_advantage_datums is False
    assert config.workflow_config.filter_zero_variance_groups is False
    assert config.workflow_config.token_efficiency_reward.enabled is False
    assert config.workflow_config.token_efficiency_reward.coefficient == 0.05
    assert config.workflow_config.token_efficiency_reward.attribution == "policy_subtree"
    assert config.rollout.max_concurrent_rollouts == 24
    assert config.rollout.consumer_batch_size == 8
    assert config.actor.backend == "megatron:(attn:d10p2t4c2|ffn:d10p2t1e8)"
    assert config.rollout.backend == "sglang:d12p1t8"
    assert config.openreward.enable_recursive_subagents is True
    assert config.openreward.subagent_max_depth == 2
    assert config.openreward.enable_programmatic_tool_calling is True
    assert config.recover.mode == "auto"
    assert config.recover.no_load_optim is False
    assert config.recover.no_save_optim is False
    assert config.stats_logger.wandb.group == "abci-ta-swe32-rec-q36m-cp2-d2-effoff-v26"


def test_fresh_qwen36_noeff_wrapper_is_fail_closed_and_fresh() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()

    assert stat.S_IMODE(WRAPPER.stat().st_mode) == 0o755
    assert "#PBS -l select=32:mpiprocs=8" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r26-e11d720-057108f" in source
    assert "RELEASE_REF=refs/tags/abci-v8r26-e11d720-057108f" in source
    assert f"CONFIG_SHA256={_sha256(CONFIG)}" in source
    assert "EXPERIMENT=abci-or-ta-swe32-q36-cp2-effoff-v26" in source
    assert "TRIAL=abci-ta-swe32-rec-q36m-cp2-c24-d2-effoff-v26-t0" in source
    assert "FRESH_BINDING=qwen36-32-v26-fresh-d2-noeff" in source
    assert "export OPENREWARD_EXPECTED_FRESH_RUN_BINDING=${FRESH_BINDING}" in source
    assert "export OPENREWARD_NODE_MEMORY_GUARD_ENABLED=0" in source
    assert "    '    enabled: false'" in source
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in source
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in source
    assert "OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" not in source
    assert "OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=" not in source
    assert 'exec "${COMMON}" "${CONFIG}"' in source


def test_fresh_qwen36_noeff_release_contract_binds_assets() -> None:
    manifest = json.loads(MANIFEST.read_text())
    contract = manifest["training_semantics"][
        "fresh_recursive_32node_qwen36_cp2_noeff_contract"
    ]
    assert contract["fresh_start"] is True
    assert contract["token_efficiency_reward_enabled"] is False
    assert contract["subagent_max_depth"] == 2
    assert contract["node_memory_guard_enabled"] is False
    assert _sha256(ROOT / contract["config_relative"]) == contract["config_sha256"]
    assert _sha256(ROOT / contract["wrapper_relative"]) == contract["wrapper_sha256"]
