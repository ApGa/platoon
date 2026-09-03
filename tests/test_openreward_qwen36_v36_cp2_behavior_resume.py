from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "pbs-scripts"
CONFIG = (
    ROOT
    / "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_openhands_areal_prealloc_15node_recursive_h200_"
    "qwen36_moe_cp2_c24_s3_v35_behavior_usage_gated_partial.yaml"
)
WRAPPER = (
    PBS
    / "openreward-toolathlon-15node-recursive-qwen36-cp2-behavior-usage-"
    "partial-resume-gs14-64k-h200-v36-abci.pbs"
)
ENV = PBS / "openreward-v36-abci.env"
VERIFIER = PBS / "verify_openreward_resume_checkpoint.py"
BINDING = "qwen36-15-v36-cp2-behavior-gs14"
EXPERIMENT = "abci-or-ta15-q36-cp2-behavior-usage-partial-v35"
TRIAL = "abci-ta15-rec-q36m-cp2-c24-s3-d2-behavior-usage-partial-v35-t0"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("resume_verifier_v36", VERIFIER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v36_resume_wrapper_preserves_the_exact_v35_run_contract() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert os.stat(WRAPPER).st_mode & 0o111
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    assert "#PBS -l select=15:mpiprocs=8" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r36-07bebf9-057108f" in source
    assert "RELEASE_REF=refs/tags/abci-v8r36-07bebf9-057108f" in source
    assert f"EXPERIMENT={EXPERIMENT}" in source
    assert f"TRIAL={TRIAL}" in source
    assert f"SUCCESSOR_BINDING={BINDING}" in source
    assert f"CONFIG_SHA256={hashlib.sha256(CONFIG.read_bytes()).hexdigest()}" in source
    assert "SOURCE_PBS_JOB_ID=2218557.pbs1" in source
    assert "EXPECTED_GLOBAL_STEP=14" in source
    assert "EXPECTED_START_GLOBAL_STEP=15" in source
    assert "export OPENREWARD_EXPECTED_RESUME_DCP_SHARDS=32" in source
    assert "export OPENREWARD_EXPECTED_NNODES=15" in source
    assert "export OPENREWARD_SERVICE_NODE_COUNT=11" in source
    assert "export OPENREWARD_ENVIRONMENT_SET=toolathlon_only" in source
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in source
    assert "export OPENREWARD_NODE_MEMORY_GUARD_ENABLED=0" in source
    assert "OPENREWARD_EXPECTED_FRESH_RUN_BINDING" in source
    assert 'exec "${COMMON}" "${CONFIG}"' in source


def test_v36_resume_binding_pins_the_durable_step_14_checkpoint() -> None:
    binding = _load_verifier().V22_SUCCESSOR_BINDINGS[BINDING]

    assert binding["experiment"] == EXPERIMENT
    assert binding["trial"] == TRIAL
    assert binding["source_job"] == "2218557.pbs1"
    assert (binding["global_step"], binding["start_global_step"]) == (14, 15)
    assert binding["shards"] == 32
    assert binding["total_bytes"] == 491_703_860_304
    assert binding["stats_commit_step"] == 13
    assert binding["state_keys"] == 21_671
    assert binding["storage_entries"] == 116_818
    assert binding["optimizer_moment_keys"] == 18
    assert binding["optimizer_layout"] == "two_group"
    assert binding["lr_num_steps"] == 15
    assert (binding["samples_yielded"], binding["iterator_yielded"]) == (208, 26)


def test_v36_overlay_reuses_pinned_v33_environment_and_rebinds_code() -> None:
    source = ENV.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(ENV)], check=True)
    assert (
        "_OPENREWARD_V36_BASE_ENV=/home/aci18912it/rasp/releases/"
        "platoon-v8r33-07bebf9-057108f/pbs-scripts/openreward-v9-abci.env"
    ) in source
    assert (
        "export PLATOON_REPO_ROOT=/home/aci18912it/rasp/releases/"
        "platoon-v8r36-07bebf9-057108f"
    ) in source
