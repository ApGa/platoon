from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PBS = ROOT / "pbs-scripts"
CONFIGS = ROOT / "plugins/openreward/platoon/openreward/configs/areal"
MANIFEST = PBS / "openreward-v9-release.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_resume_verifier():
    path = PBS / "verify_openreward_resume_checkpoint.py"
    spec = importlib.util.spec_from_file_location("openreward_v22_resume_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v22_manifest_binds_three_successors_and_exact_assets() -> None:
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["release_root"] == ("/home/aci18912it/rasp/releases/platoon-v8r23-e11d720-057108f")
    assert manifest["release_ref"] == "refs/tags/abci-v8r23-e11d720-057108f"
    contracts = manifest["training_semantics"]["v8r22_successor_contracts"]
    assert set(contracts) == {
        "qwen38_32node_gs10",
        "qwen38_16node_gs11",
        "qwen36_32node_gs1_depth2",
    }
    assert [contracts[name]["max_concurrent_rollouts"] for name in contracts] == [
        24,
        12,
        24,
    ]
    for contract in contracts.values():
        for kind in ("config", "wrapper"):
            path = ROOT / contract[f"{kind}_relative"]
            assert _sha256(path) == contract[f"{kind}_sha256"]
        assert stat.S_IMODE((ROOT / contract["wrapper_relative"]).stat().st_mode) == 0o755


def test_v22_config_semantics_preserve_requested_concurrency_and_lineage() -> None:
    q38_32 = (
        CONFIGS / "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen38_dense_c24_ta20_v22_resume_gs10.yaml"
    ).read_text()
    q38_16 = (
        CONFIGS / "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
        "qwen38_dense_c12_ta20_v22_resume_gs11.yaml"
    ).read_text()
    q36 = (
        CONFIGS / "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
        "qwen36_moe_cp2_c24_ta20_v22_resume_d2.yaml"
    ).read_text()
    assert "  max_concurrent_rollouts: 24\n" in q38_32
    assert "  subagent_max_depth: 2\n" in q38_32
    assert "    timeout: 7200\n" in q38_32
    assert "    step_timeout: 5400\n" in q38_32
    assert "  max_concurrent_rollouts: 12\n" in q38_16
    assert "  no_load_optim: false\n" in q38_16
    assert "  max_concurrent_rollouts: 24\n" in q36
    assert "  subagent_max_depth: 2\n" in q36
    assert "dynamic_bs: false\n" in q36
    assert "  filter_zero_advantage_datums: false\n" in q36
    assert "  filter_zero_variance_groups: false\n" in q36
    assert "experiment_name: abci-or-ta-swe32-q36-cp2-v21\n" in q36
    assert "trial_name: abci-ta-swe32-rec-q36m-cp2-c24-d1-v21-t0\n" in q36
    assert "    group: abci-ta-swe32-rec-q36m-cp2-d1-v21\n" in q36


def test_v22_resume_verifier_exact_binding_counts() -> None:
    verifier = _load_resume_verifier()
    bindings = verifier.V22_SUCCESSOR_BINDINGS
    assert set(bindings) == {
        "qwen38-32-v22-gs10",
        "qwen38-16-v22-gs11-ordinary",
        "qwen36-32-v22-depth2",
    }
    q38_32 = bindings["qwen38-32-v22-gs10"]
    q38_16 = bindings["qwen38-16-v22-gs11-ordinary"]
    q36 = bindings["qwen36-32-v22-depth2"]
    assert (q38_32["global_step"], q38_32["lr_num_steps"]) == (10, 11)
    assert (q38_16["global_step"], q38_16["lr_num_steps"]) == (11, 8)
    assert (q36["global_step"], q36["lr_num_steps"]) == (1, 2)
    assert q36["optimizer_layout"] == "two_group"
    assert [
        q38_32["optimizer_moment_keys"],
        q38_16["optimizer_moment_keys"],
        q36["optimizer_moment_keys"],
    ] == [16, 16, 48]


def test_v22_common_forwards_binding_and_guard_without_changing_concurrency() -> None:
    common = (PBS / "openreward-ta-swe-prealloc-common.sh").read_text()
    services = (PBS / "openreward-node-services.sh").read_text()
    assert "OPENREWARD_EXPECTED_RESUME_SUCCESSOR_BINDING" in common
    for binding in (
        "qwen38-32-v22-gs10",
        "qwen38-16-v22-gs11-ordinary",
        "qwen36-32-v22-depth2",
    ):
        assert binding in common
    for literal in (
        "2061584302080",
        "274877906944",
        "NODE_MEMORY_TRIGGER_PERCENT:-85",
        "NODE_MEMORY_POLL_SECONDS:-2",
        "NODE_MEMORY_TELEMETRY_SECONDS:-60",
    ):
        assert literal in common
        assert literal in services
