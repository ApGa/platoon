from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "plugins/openreward/platoon/openreward/configs/areal"
    / (
        "abci_toolathlon_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen36_moe_cp2_c12_v33_behavior_usage_gated.yaml"
    )
)
LAUNCHER = (
    REPO_ROOT
    / "pbs-scripts/openreward-toolathlon-32node-recursive-qwen36-cp2-behavior-usage-gated-64k-h200-v33-abci.pbs"
)
COMMON = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
ENV = REPO_ROOT / "pbs-scripts/openreward-v33-abci.env"


def test_v33_behavior_usage_config_is_fresh_64k_toolathlon_only() -> None:
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load(CONFIG.read_text())

    assert config["experiment_name"] == ("abci-or-ta32-q36-cp2-behavior-usage-gate-v33")
    assert config["trial_name"] == ("abci-ta32-rec-q36m-cp2-c12-d2-behavior-usage-gate-v33-t0")
    assert config["workflow_config"]["rollout_config"]["propagate_root_success"] is False
    assert config["openreward"]["subagent_max_depth"] == 2
    assert config["openreward"]["enable_subagent_reward_judging"] is True
    assert config["openreward"]["enable_subagent_behavior_judging"] is True
    assert config["openreward"]["enable_subagent_parent_usage_eligibility_gate"] is True
    assert config["openreward"]["subagent_parent_usage_judge_max_concurrency"] == 4
    assert [entry["label"] for entry in config["openreward"]["environments"]] == ["toolathlon"]


def test_v33_launcher_attests_config_and_skips_only_redundant_external_gates() -> None:
    source = LAUNCHER.read_text()
    digest = hashlib.sha256(CONFIG.read_bytes()).hexdigest()

    assert f"CONFIG_SHA256={digest}" in source
    assert "RELEASE_REF=refs/tags/abci-v8r33-07bebf9-057108f" in source
    assert "FRESH_BINDING=qwen36-32-v33-behavior-usage64" in source
    assert "OPENREWARD_NODE_MEMORY_GUARD_ENABLED=0" in source
    assert "OPENREWARD_EXPECTED_NNODES=32" in source
    assert "OPENREWARD_SERVICE_NODE_COUNT=12" in source
    assert "GATE2_RESULT" not in source
    assert "ALL_MCP_RESULT" not in source
    assert "require_component external/AReaL d99124ec" in source
    assert "require_component external/toolathlon-gym 18e62c0d" in source
    assert os.stat(LAUNCHER).st_mode & 0o111
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_v33_common_runtime_authorizes_exact_fresh_identity_and_disabled_guard() -> None:
    source = COMMON.read_text()
    binding = "qwen36-32-v33-behavior-usage64"

    assert source.count(binding) == 3
    assert '"${EXPECTED_EXPERIMENT}" == abci-or-ta32-q36-cp2-behavior-usage-gate-v33' in source
    assert '"${EXPECTED_TRIAL}" == abci-ta32-rec-q36m-cp2-c12-d2-behavior-usage-gate-v33-t0' in source


def test_v33_env_rebinds_only_the_code_release_after_v32_asset_overlay() -> None:
    source = ENV.read_text()

    assert re.search(
        r"source \"\$\{_OPENREWARD_V33_BASE_ENV\}\".*"
        r"PLATOON_REPO_ROOT=/home/aci18912it/rasp/releases/"
        r"platoon-v8r33-07bebf9-057108f",
        source,
        flags=re.DOTALL,
    )
    assert os.stat(ENV).st_mode & 0o111
    subprocess.run(["bash", "-n", str(ENV)], check=True)
