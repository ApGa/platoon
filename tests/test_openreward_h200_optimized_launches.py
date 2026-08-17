from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins" / "openreward" / "platoon" / "openreward" / "configs" / "areal"
PBS_ROOT = REPO_ROOT / "pbs-scripts"

CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20",
        "parent": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_ta20",
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": "abci-openreward-ta-swe-h200-16",
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v1-trial0",
        "actor": "megatron:(attn:d10p2t4c1|ffn:d5p2t1e8)",
        "actor_gpus": 80,
        "rollout": "sglang:d6p1t8",
        "rollout_gpus": 48,
        "concurrency": 12,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20",
        "parent": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_ta20",
        "wrapper": "openreward-ta-swe-32node-recursive-h200-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": "abci-openreward-ta-swe-h200-32",
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v1-trial0",
        "actor": "megatron:(attn:d20p2t4c1|ffn:d10p2t1e8)",
        "actor_gpus": 160,
        "rollout": "sglang:d12p1t8",
        "rollout_gpus": 96,
        "concurrency": 24,
    },
)

V2_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v2",
        "parent": CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v2-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v2-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v2",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v2",
        "parent": CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v2-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v2-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v2",
        "service_nodes": 12,
    },
)

V3_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v3",
        "parent": V2_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v3-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v3-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v3",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v3",
        "parent": V2_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v3-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v3-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v3",
        "service_nodes": 12,
    },
)

V4_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v4",
        "parent": V3_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v4-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v4-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v4",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v4",
        "parent": V3_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v4-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v4-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v4",
        "service_nodes": 12,
    },
)

V5_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v5",
        "parent": V4_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v5-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v5-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v5",
        "service_nodes": 6,
        "failed_remote_workers": 72,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v5",
        "parent": V4_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v5-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v5-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v5",
        "service_nodes": 12,
        "failed_remote_workers": 152,
    },
)

V6_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v6",
        "parent": V5_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v6-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v6-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v6",
        "service_nodes": 6,
        "rollout_mem": 32,
        "mem_fraction_static": 0.8,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v6",
        "parent": V5_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v6-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v6-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v6",
        "service_nodes": 12,
        "rollout_mem": 128,
        "mem_fraction_static": 0.7,
    },
)

V7_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v7",
        "parent": V6_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v7-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v7-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v7",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v7",
        "parent": V6_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v7-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v7-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v7",
        "service_nodes": 12,
    },
)

V8_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v8",
        "parent": V7_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v8-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "ptc": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v8-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v8",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v8",
        "parent": V7_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v8-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "ptc": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v8-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v8",
        "service_nodes": 12,
    },
)

V9_CASES = (
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v9",
        "parent": V8_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "ptc": False,
        "experiment": CASES[0]["experiment"],
        "trial": "abci-ta-swe16-nonrec-h200-cp1-c12-v9-trial0",
        "wandb_group": "abci-ta-swe16-nonrec-h200-cp1-c12-v9",
        "service_nodes": 6,
    },
    {
        "config": "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v9",
        "parent": V8_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "ptc": True,
        "experiment": CASES[1]["experiment"],
        "trial": "abci-ta-swe32-rec-h200-cp1-c24-v9-trial0",
        "wandb_group": "abci-ta-swe32-rec-h200-cp1-c24-v9",
        "service_nodes": 12,
    },
)

CP1_PADDED_CASES = (
    {
        "config": ("abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v10_padded"),
        "parent": V9_CASES[0]["config"],
        "wrapper": "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs",
        "nodes": 16,
        "recursive": False,
        "ptc": False,
        "experiment": "abci-or-ta-swe16-cp1-v14",
        "trial": "abci-ta-swe16-nr-cp1-c12-v14-t0",
        "wandb_group": "abci-ta-swe16-nr-cp1-c12-v14",
        "job_name": "rasp-h200-16sf-v14",
        "service_nodes": 6,
        "max_concurrent_rollouts": 12,
        "consumer_batch_size": 8,
        "recollection_max_rounds": 1,
        "actor_safe_minimum_datums": 40,
    },
    {
        "config": ("abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v10_padded"),
        "parent": V9_CASES[1]["config"],
        "wrapper": "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs",
        "nodes": 32,
        "recursive": True,
        "ptc": True,
        "experiment": "abci-or-ta-swe32-cp1-v15",
        "trial": "abci-ta-swe32-rec-cp1-c24-v15-t0",
        "wandb_group": "abci-ta-swe32-rec-cp1-c24-v15",
        "job_name": "rasp-h200-32sf-v15",
        "service_nodes": 12,
        "max_concurrent_rollouts": 24,
        "consumer_batch_size": 16,
        "recollection_max_rounds": 2,
        "actor_safe_minimum_datums": 80,
        "subagent_datum_keep_probability": 0.25,
    },
)

QWEN38_DENSE_CONFIG = "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_qwen38_dense_c24_ta20_v16"
QWEN38_DENSE_WRAPPER = "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs"
QWEN38_DENSE_16_CONFIG = (
    "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
    "qwen38_dense_c12_ta20_v20_gs0_no_load_optim"
)
QWEN38_DENSE_16_WRAPPER = (
    "openreward-ta-swe-16node-nonrecursive-qwen38-gs0-nooptim-h200-v10-abci.pbs"
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


def _actor_sizes(backend: str) -> tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]:
    match = re.fullmatch(
        r"megatron:\(attn:d(\d+)p(\d+)t(\d+)c(\d+)\|ffn:d(\d+)p(\d+)t(\d+)e(\d+)\)",
        backend,
    )
    assert match is not None, backend
    values = tuple(int(value) for value in match.groups())
    attn = values[:4]
    ffn = values[4:]
    return (
        attn[0] * attn[1] * attn[2] * attn[3],
        ffn[0] * ffn[1] * ffn[2] * ffn[3],
        attn,
        ffn,
    )


def _rollout_size(backend: str) -> tuple[int, int]:
    match = re.fullmatch(r"sglang:d(\d+)p(\d+)t(\d+)", backend)
    assert match is not None, backend
    data, pipeline, tensor = (int(value) for value in match.groups())
    return data * pipeline * tensor, data


@pytest.mark.parametrize("case", CASES, ids=("16node", "32node"))
def test_h200_leaf_has_exact_safe_topology_and_preserves_training_semantics(case: dict) -> None:
    config = _compose(case["config"])
    parent = _compose(case["parent"])

    assert config["experiment_name"] == case["experiment"]
    assert config["trial_name"] == case["trial"]
    assert config["trial_name"] != parent["trial_name"]
    assert config["trial_name"].startswith("abci-")
    assert config["scheduler"]["type"] == "pbs_prealloc"
    assert config["cluster"]["n_nodes"] == case["nodes"]
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["actor"]["backend"] == case["actor"]

    attn_gpus, ffn_gpus, attn, ffn = _actor_sizes(config["actor"]["backend"])
    assert attn_gpus == ffn_gpus == case["actor_gpus"]
    assert attn[3] == 1  # CP is explicitly disabled.
    assert attn[1] == ffn[1] == 2  # Hybrid attention/FFN PP must match.
    assert attn[2] == 4  # Keep the parity TP rather than introducing TP8 risk.
    assert ffn[2:] == (1, 8)  # Keep expert TP1/EP8 unchanged.

    rollout_gpus, rollout_replicas = _rollout_size(config["rollout"]["backend"])
    assert config["rollout"]["backend"] == case["rollout"]
    assert rollout_gpus == case["rollout_gpus"]
    assert config["rollout"]["max_concurrent_rollouts"] == case["concurrency"]
    assert case["concurrency"] // rollout_replicas == 2
    assert attn_gpus + rollout_gpus == case["nodes"] * 8

    # The optimization changes only topology, concurrency, and fresh identity.
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config["openreward"]["environments"] == parent["openreward"]["environments"]
    assert config["train_dataset"]["batch_size"] == parent["train_dataset"]["batch_size"] == 8
    assert config["workflow_config"] == parent["workflow_config"]
    assert config["actor"]["mb_spec"]["max_tokens_per_mb"] == 32768
    assert config["ref"]["mb_spec"]["max_tokens_per_mb"] == 32768
    assert config["sglang"]["context_length"] == 32769
    assert config["recover"]["freq_steps"] == 1
    assert config["valid_dataset"] is None


@pytest.mark.parametrize("case", CASES, ids=("16node", "32node"))
def test_h200_wrapper_is_fresh_fixed_and_receipt_enabled(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert "refusing stale checkpoint" in wrapper
    assert "refusing stale success artifact" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()
    assert "openreward_update_gate.py" not in wrapper


def test_recursive_h200_wrapper_has_no_nonrecursive_parity_dependency() -> None:
    wrapper = (PBS_ROOT / CASES[1]["wrapper"]).read_text()
    assert "GATE_MARKER=" not in wrapper
    assert "GATE_EXPERIMENT=" not in wrapper
    assert "GATE_TRIAL=" not in wrapper
    assert "openreward_update_gate.py" not in wrapper


@pytest.mark.parametrize("case", V2_CASES, ids=("16node-v2", "32node-v2"))
def test_h200_v2_leaf_changes_only_fresh_identity(case: dict) -> None:
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "trial_name": case["trial"],
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert config["experiment_name"] == case["experiment"]
    assert config["trial_name"] == case["trial"]
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]


@pytest.mark.parametrize("case", V2_CASES, ids=("16node-v2", "32node-v2"))
def test_h200_v2_service_tail_matches_disjoint_role_topology(case: dict) -> None:
    config = _compose(case["config"])
    actor_gpus, ffn_gpus, _attn, _ffn = _actor_sizes(config["actor"]["backend"])
    rollout_gpus, _rollout_replicas = _rollout_size(config["rollout"]["backend"])
    assert actor_gpus == ffn_gpus
    assert actor_gpus % 8 == rollout_gpus % 8 == 0

    actor_nodes = actor_gpus // 8
    rollout_nodes = rollout_gpus // 8
    assert rollout_nodes == case["service_nodes"]
    assert actor_nodes + rollout_nodes == case["nodes"]

    ordered_nodes = tuple(range(case["nodes"]))
    actor_prefix = ordered_nodes[:actor_nodes]
    service_tail = ordered_nodes[-case["service_nodes"] :]
    assert actor_prefix == ordered_nodes[: case["nodes"] - case["service_nodes"]]
    assert service_tail == ordered_nodes[case["nodes"] - case["service_nodes"] :]
    assert set(actor_prefix).isdisjoint(service_tail)
    assert actor_prefix + service_tail == ordered_nodes

    wrapper_path = PBS_ROOT / case["wrapper"]
    subprocess.run(["bash", "-n", str(wrapper_path)], check=True)
    wrapper = wrapper_path.read_text()
    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V3_CASES, ids=("16node-v3", "32node-v3"))
def test_h200_v3_leaf_repeats_preflight_invariants_without_semantic_drift(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    # These literals are intentionally present in the leaf itself. The common
    # shell preflight runs before the pinned Hydra/controller environment is
    # resolved and must not compose live source with host-site dependencies.
    assert leaf["experiment_name"] == case["experiment"]
    assert leaf["trial_name"] == case["trial"]
    assert leaf["scheduler"]["type"] == "pbs_prealloc"
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    assert leaf["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config == expected

    text = path.read_text()
    assert re.search(rf"^experiment_name:\s*{re.escape(case['experiment'])}$", text, re.MULTILINE)
    assert re.search(rf"^trial_name:\s*{re.escape(case['trial'])}$", text, re.MULTILINE)
    assert re.search(r"^\s+type:\s*pbs_prealloc$", text, re.MULTILINE)
    assert re.search(
        r"^\s+fileroot:\s*\$\{oc\.env:PLATOON_EXPERIMENT_ROOT\}$",
        text,
        re.MULTILINE,
    )
    assert re.search(
        rf"^\s+enable_recursive_subagents:\s*{str(case['recursive']).lower()}$",
        text,
        re.MULTILINE,
    )


@pytest.mark.parametrize("case", V3_CASES, ids=("16node-v3", "32node-v3"))
def test_h200_v3_wrapper_has_fresh_fail_closed_identity_and_service_tail(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert "v2 failed only the literal-leaf config preflight" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V4_CASES, ids=("16node-v4", "32node-v4"))
def test_h200_v4_leaf_repeats_preflight_invariants_without_semantic_drift(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert leaf["experiment_name"] == case["experiment"]
    assert leaf["trial_name"] == case["trial"]
    assert leaf["scheduler"]["type"] == "pbs_prealloc"
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    assert leaf["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config == expected


@pytest.mark.parametrize("case", V4_CASES, ids=("16node-v4", "32node-v4"))
def test_h200_v4_wrapper_has_fresh_fail_closed_identity_and_service_tail(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert "v3 passed literal preflight, immutable runtime setup" in wrapper
    assert "failed controller task discovery" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V5_CASES, ids=("16node-v5", "32node-v5"))
def test_h200_v5_leaf_changes_only_fresh_identity(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert leaf["experiment_name"] == case["experiment"]
    assert leaf["trial_name"] == case["trial"]
    assert leaf["scheduler"]["type"] == "pbs_prealloc"
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    assert leaf["openreward"]["enable_recursive_subagents"] is case["recursive"]


@pytest.mark.parametrize("case", V5_CASES, ids=("16node-v5", "32node-v5"))
def test_h200_v5_wrapper_is_fresh_fail_closed_and_audits_bind_failure(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    assert path.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert f"{case['failed_remote_workers']} remote containers" in wrapper
    assert "controller-created /local bind source" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V6_CASES, ids=("16node-v6", "32node-v6"))
def test_h200_v6_leaf_changes_only_identity_and_rollout_allocator(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    rollout_spec = [
        {
            "task_type": "worker",
            "port_count": 2,
            "gpu": 1,
            "mem": case["rollout_mem"],
            "cmd": "python -m areal.infra.rpc.rpc_server",
            "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"},
        }
    ]
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "rollout": {"scheduling_spec": rollout_spec},
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert config["rollout"]["scheduling_spec"] == rollout_spec
    assert config["actor"]["scheduling_spec"] == parent["actor"]["scheduling_spec"]
    assert config["ref"]["scheduling_spec"] == parent["ref"]["scheduling_spec"]
    assert config["sglang"]["mem_fraction_static"] == case["mem_fraction_static"]
    assert config["sglang"] == parent["sglang"]
    assert leaf["rollout"]["scheduling_spec"] == rollout_spec
    assert leaf["experiment_name"] == case["experiment"]
    assert leaf["trial_name"] == case["trial"]
    assert leaf["scheduler"]["type"] == "pbs_prealloc"
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    assert leaf["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert "sglang" not in leaf
    assert "disable_custom_all_reduce" not in path.read_text()


@pytest.mark.parametrize("case", V6_CASES, ids=("16node-v6", "32node-v6"))
def test_h200_v6_wrapper_is_fresh_fail_closed_and_keeps_custom_ar(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    assert path.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert "expandable segments could not be exported" in wrapper
    assert "Preserve custom all-reduce, CUDA graphs" in wrapper
    assert "disable_custom_all_reduce" not in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V7_CASES, ids=("16node-v7", "32node-v7"))
def test_h200_v7_leaf_preserves_horizon_and_explicitly_disables_ptc(
    case: dict,
) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "openreward": {"enable_programmatic_tool_calling": False},
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert leaf == {
        "defaults": [case["parent"], "_self_"],
        "experiment_name": case["experiment"],
        "trial_name": case["trial"],
        "scheduler": {"type": "pbs_prealloc"},
        "cluster": {
            "n_nodes": case["nodes"],
            "n_gpus_per_node": 8,
            "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
            "name_resolve": {
                "type": "nfs",
                "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
            },
        },
        "openreward": {
            "enable_recursive_subagents": case["recursive"],
            "enable_programmatic_tool_calling": False,
        },
        "stats_logger": {"wandb": {"group": case["wandb_group"]}},
    }
    # V7 changes identity and intentionally disables PTC. The complete horizon,
    # topology, data, rollout, optimizer, checkpoint, and logging stay inherited.
    assert config.get("total_train_steps") == parent.get("total_train_steps")
    assert config["total_train_epochs"] == parent["total_train_epochs"]
    assert config["actor"] == parent["actor"]
    assert config["rollout"] == parent["rollout"]
    assert config["workflow_config"] == parent["workflow_config"]
    assert config["recover"] == parent["recover"]
    assert config["valid_dataset"] is None
    assert parent["openreward"]["enable_programmatic_tool_calling"] is True
    assert config["openreward"]["enable_programmatic_tool_calling"] is False
    assert config["openreward"]["enable_task_tracker"] is True
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]


@pytest.mark.parametrize("case", V7_CASES, ids=("16node-v7", "32node-v7"))
def test_h200_v7_real_structured_composition_disables_ptc(case: dict, monkeypatch) -> None:
    cli_args = pytest.importorskip("areal.api.cli_args")
    omegaconf = pytest.importorskip("omegaconf")
    config_module = pytest.importorskip("platoon.openreward.areal_config")
    for name in ("PLATOON_EXPERIMENT_ROOT", "PLATOON_MODEL_SNAPSHOT", "PLATOON_NAME_RESOLVE_ROOT"):
        monkeypatch.setenv(name, f"/tmp/platoon-v7-config-test/{name.lower()}")

    raw, _ = cli_args.parse_cli_args(["--config", str(CONFIG_ROOT / f"{case['config']}.yaml")])
    structured = cli_args.to_structured_cfg(raw, config_module.OpenRewardArealTrainerConfig)
    config = omegaconf.OmegaConf.to_object(structured)

    assert isinstance(config, config_module.OpenRewardArealTrainerConfig)
    assert config.openreward.enable_programmatic_tool_calling is False
    assert config.openreward.enable_task_tracker is True
    assert config.openreward.enable_recursive_subagents is case["recursive"]


@pytest.mark.parametrize("case", V7_CASES, ids=("16node-v7", "32node-v7"))
def test_h200_v7_wrapper_is_fresh_fail_closed_long_run_with_live_candidate(
    case: dict,
) -> None:
    path = PBS_ROOT / case["wrapper"]
    assert path.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert (
        f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml"
    ) in wrapper
    assert f"EXPERIMENT={case['experiment']}" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_EXPECTED_NNODES={case['nodes']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert "host-controller Python" in wrapper
    assert "immutable worker-container interpreter" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT=${RECEIPT}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=$(date +%s)" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert '"${COMMON}" "${CONFIG}" || status=$?' not in wrapper
    assert "HOST_PYTHON=" not in wrapper
    assert "training_success_receipt verify" not in wrapper
    assert "OPENREWARD_H200_V7_ONE_UPDATE_OK" not in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V8_CASES, ids=("16node-v8", "32node-v8"))
def test_h200_v8_leaf_advances_only_identity_and_explicit_ptc_matrix(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    openreward = {
        "enable_recursive_subagents": case["recursive"],
        "enable_programmatic_tool_calling": case["ptc"],
    }
    if case["ptc"]:
        openreward |= {
            "programmatic_tool_calling_mode": "orchestration_only",
            "programmatic_tool_calling_max_tool_calls_per_execution": 1024,
        }
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "openreward": openreward,
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert leaf["scheduler"] == {"type": "pbs_prealloc"}
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    assert config.get("total_train_steps") == parent.get("total_train_steps")
    assert config["total_train_epochs"] == parent["total_train_epochs"] == 10
    for invariant in ("actor", "rollout", "workflow_config", "recover"):
        assert config[invariant] == parent[invariant]
    assert config["openreward"]["enable_task_tracker"] is True
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config["openreward"]["enable_programmatic_tool_calling"] is case["ptc"]
    if case["ptc"]:
        assert config["openreward"]["programmatic_tool_calling_mode"] == "orchestration_only"
        assert config["openreward"]["programmatic_tool_calling_max_tool_calls_per_execution"] == 1024


@pytest.mark.parametrize("case", V8_CASES, ids=("16node-v8", "32node-v8"))
def test_h200_v8_wrapper_pins_isolated_release_image_and_fresh_long_run(case: dict) -> None:
    path = PBS_ROOT / case["wrapper"]
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert path.stat().st_mode & 0o111 == 0o111
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r2-376d1b9-18e62c0" in wrapper
    assert "REPO_ROOT=/home/aci18912it/rasp/source/platoon" not in wrapper
    assert "RELEASE_REF=refs/tags/abci-v8r2-376d1b9-18e62c0" in wrapper
    assert "RELEASE_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_v8_release.py" in wrapper
    assert "ABCI_ENV=${REPO_ROOT}/pbs-scripts/openreward-v8-abci.env" in wrapper
    assert "export PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" in wrapper
    assert "status --porcelain=v1 --untracked-files=all" in wrapper
    assert '"${RELEASE_VERIFY}" --release-root "${REPO_ROOT}"' in wrapper
    assert f"#PBS -l select={case['nodes']}:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert f"TRIAL={case['trial']}" in wrapper
    assert f"OPENREWARD_SERVICE_NODE_COUNT={case['service_nodes']}" in wrapper
    assert f"OPENREWARD_EXPECTED_RECURSIVE={int(case['recursive'])}" in wrapper
    assert f"OPENREWARD_EXPECTED_PTC={int(case['ptc'])}" in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()


@pytest.mark.parametrize("case", V9_CASES, ids=("16node-v9", "32node-v9"))
def test_h200_v9_leaf_is_fresh_full_horizon_and_has_exact_ptc_matrix(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{case['config']}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    openreward = {
        "enable_recursive_subagents": case["recursive"],
        "enable_programmatic_tool_calling": case["ptc"],
    }
    if case["ptc"]:
        openreward |= {
            "programmatic_tool_calling_mode": "orchestration_only",
            "programmatic_tool_calling_max_tool_calls_per_execution": 1024,
        }
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "total_train_epochs": 10,
            "total_train_steps": None,
            "tokenizer_path": "${oc.env:PLATOON_MODEL_SNAPSHOT}",
            "openreward": openreward,
            "actor": {"path": "${oc.env:PLATOON_MODEL_SNAPSHOT}"},
            "sglang": {"model_path": "${oc.env:PLATOON_MODEL_SNAPSHOT}"},
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert leaf["total_train_epochs"] == 10
    assert leaf["total_train_steps"] is None
    assert leaf["tokenizer_path"] == "${oc.env:PLATOON_MODEL_SNAPSHOT}"
    assert leaf["actor"] == {"path": "${oc.env:PLATOON_MODEL_SNAPSHOT}"}
    assert leaf["sglang"] == {"model_path": "${oc.env:PLATOON_MODEL_SNAPSHOT}"}
    assert leaf["scheduler"] == {"type": "pbs_prealloc"}
    assert leaf["cluster"] == {
        "n_nodes": case["nodes"],
        "n_gpus_per_node": 8,
        "fileroot": "${oc.env:PLATOON_EXPERIMENT_ROOT}",
        "name_resolve": {
            "type": "nfs",
            "nfs_record_root": "${oc.env:PLATOON_NAME_RESOLVE_ROOT}",
        },
    }
    for invariant in ("actor", "rollout", "workflow_config", "recover"):
        assert config[invariant] == parent[invariant]
    assert config["openreward"]["enable_task_tracker"] is True
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config["openreward"]["enable_programmatic_tool_calling"] is case["ptc"]
    if case["ptc"]:
        assert config["openreward"]["programmatic_tool_calling_mode"] == "orchestration_only"
        assert config["openreward"]["programmatic_tool_calling_max_tool_calls_per_execution"] == 1024


@pytest.mark.parametrize("case", CP1_PADDED_CASES, ids=("16node-cp1-padded", "32node-cp1-padded"))
def test_h200_cp1_padded_leaf_enables_dynamic_signal_filtered_batches(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    leaf = yaml.safe_load((CONFIG_ROOT / f"{case['config']}.yaml").read_text())
    parent_leaf = yaml.safe_load((CONFIG_ROOT / f"{case['parent']}.yaml").read_text())
    config = _compose(case["config"])
    parent = _compose(case["parent"])
    expected = _merge(
        parent,
        {
            "experiment_name": case["experiment"],
            "trial_name": case["trial"],
            "dynamic_bs": True,
            "workflow_config": {
                "filter_zero_advantage_datums": True,
                "filter_zero_variance_groups": True,
                "dynamic_batch_recollection_max_rounds": case["recollection_max_rounds"],
            },
            "rollout": {"consumer_batch_size": case["consumer_batch_size"]},
            "stats_logger": {"wandb": {"group": case["wandb_group"]}},
        },
    )

    assert config == expected
    assert leaf["defaults"] == [case["parent"], "_self_"]
    assert leaf["scheduler"] == parent_leaf["scheduler"]
    assert leaf["cluster"] == parent_leaf["cluster"]
    assert leaf["openreward"] == parent_leaf["openreward"]
    assert leaf["dynamic_bs"] is True
    assert leaf["workflow_config"] == {
        "filter_zero_advantage_datums": True,
        "filter_zero_variance_groups": True,
        "dynamic_batch_recollection_max_rounds": case["recollection_max_rounds"],
    }
    assert leaf["rollout"] == {"consumer_batch_size": case["consumer_batch_size"]}
    assert parent.get("dynamic_bs") is None
    assert parent["workflow_config"]["filter_zero_advantage_datums"] is False
    assert parent["workflow_config"]["filter_zero_variance_groups"] is False
    assert config["cluster"]["n_nodes"] == case["nodes"]
    assert config["openreward"]["enable_recursive_subagents"] is case["recursive"]
    assert config["openreward"]["enable_programmatic_tool_calling"] is case["ptc"]
    assert config["openreward"]["balance_accepted_batches"] is False
    assert config["rollout"]["max_head_offpolicyness"] == 3
    assert config["rollout"]["consumer_batch_size"] == case["consumer_batch_size"]
    assert config["train_dataset"]["batch_size"] == 8
    assert config["rollout"]["max_concurrent_rollouts"] == case["max_concurrent_rollouts"]
    assert config["workflow_config"]["dynamic_batch_recollection_max_rounds"] == case["recollection_max_rounds"]
    assert config["rollout"]["consumer_batch_size"] >= (
        config["train_dataset"]["batch_size"] * case["recollection_max_rounds"]
    )

    _, _, attn, _ = _actor_sizes(config["actor"]["backend"])
    actor_dp_size, actor_pp_size = attn[:2]
    actor_mb_spec = config["actor"].get("mb_spec", {})
    configured_n_mbs = actor_mb_spec.get("n_mbs", 1) or 1
    ppo_n_minibatches = config["actor"].get("ppo_n_minibatches", 1)
    microbatch_granularity = actor_mb_spec.get("granularity", 1)
    pipeline_minimum = 2 * actor_pp_size if actor_pp_size > 1 else 1
    engine_min_microbatches = math.ceil(max(configured_n_mbs, pipeline_minimum) / actor_pp_size) * actor_pp_size
    actor_safe_minimum = actor_dp_size * ppo_n_minibatches * microbatch_granularity * engine_min_microbatches
    assert actor_safe_minimum == case["actor_safe_minimum_datums"]
    if case["recursive"]:
        assert config["workflow_config"]["subagent_datum_keep_probability"] == case["subagent_datum_keep_probability"]
    assert config["total_train_epochs"] == 10
    assert config["total_train_steps"] is None
    for invariant in ("actor", "recover"):
        assert config[invariant] == parent[invariant]
    assert config["rollout"] == _merge(
        parent["rollout"],
        {"consumer_batch_size": case["consumer_batch_size"]},
    )


def test_qwen38_dense_recursive_leaf_has_exact_safe_training_contract() -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{QWEN38_DENSE_CONFIG}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(QWEN38_DENSE_CONFIG)
    parent_name = CP1_PADDED_CASES[1]["config"]
    parent = _compose(parent_name)

    assert leaf["defaults"] == [parent_name, "_self_"]
    assert config["experiment_name"] == "abci-or-ta-swe32-q38-v16"
    assert config["trial_name"] == "abci-ta-swe32-rec-q38d-c24-v16-t0"
    assert config["stats_logger"]["wandb"]["group"] == "abci-ta-swe32-rec-q38d-c24-v16"
    snapshot = "${oc.env:PLATOON_MODEL_SNAPSHOT}"
    assert config["tokenizer_path"] == snapshot
    assert config["actor"]["path"] == snapshot
    assert config["sglang"]["model_path"] == snapshot

    actor_match = re.fullmatch(r"megatron:d(\d+)p(\d+)t(\d+)c(\d+)", config["actor"]["backend"])
    assert actor_match is not None
    actor_dp, actor_pp, actor_tp, actor_cp = (int(value) for value in actor_match.groups())
    actor_gpus = actor_dp * actor_pp * actor_tp * actor_cp
    rollout_gpus, rollout_replicas = _rollout_size(config["rollout"]["backend"])
    assert (actor_dp, actor_pp, actor_tp, actor_cp) == (20, 2, 4, 1)
    assert actor_gpus == 160
    assert config["rollout"]["backend"] == "sglang:d12p1t8"
    assert rollout_gpus == 96
    assert rollout_replicas == 12
    assert actor_gpus + rollout_gpus == config["cluster"]["n_nodes"] * 8 == 256
    assert config["ref"]["backend"] == "${actor.backend}"
    assert "attn:" not in config["actor"]["backend"]
    assert "ffn:" not in config["actor"]["backend"]

    mb_spec = config["actor"]["mb_spec"]
    assert mb_spec["n_mbs"] == 8
    assert mb_spec["max_tokens_per_mb"] == 32768
    ppo_n_minibatches = config["actor"]["ppo_n_minibatches"]
    granularity = mb_spec.get("granularity", 1)
    engine_divisor = math.lcm(actor_pp, mb_spec.get("n_mbs_divisor", 1))
    engine_min_microbatches = math.ceil(max(mb_spec["n_mbs"], 2 * actor_pp) / engine_divisor) * engine_divisor
    datum_divisor = actor_dp * ppo_n_minibatches * granularity * engine_divisor
    minimum_datums = actor_dp * ppo_n_minibatches * granularity * engine_min_microbatches
    assert engine_divisor == 2
    assert datum_divisor == 40
    assert minimum_datums == 160

    assert config["dynamic_bs"] is True
    assert config["workflow_config"]["filter_zero_advantage_datums"] is True
    assert config["workflow_config"]["filter_zero_variance_groups"] is True
    assert config["workflow_config"]["dynamic_batch_recollection_max_rounds"] == 2
    assert config["workflow_config"]["subagent_datum_keep_probability"] == 0.25
    assert config["rollout"]["consumer_batch_size"] == 16
    assert config["rollout"]["max_concurrent_rollouts"] == 24
    assert config["rollout"]["max_concurrent_rollouts"] // rollout_replicas == 2
    assert config["rollout"]["max_head_offpolicyness"] == 1

    assert config["rollout"]["return_routed_experts"] is False
    assert config["actor"]["enable_router_replay"] is False
    assert config["actor"]["router_replay_num_layers"] is None
    assert config["actor"]["router_replay_topk"] is None
    assert config["actor"]["router_replay_num_experts"] is None
    assert config["workflow_config"]["enable_router_replay"] is False
    assert config["workflow_config"]["router_replay_num_layers"] is None
    assert config["workflow_config"]["router_replay_topk"] is None
    assert config["actor"]["megatron"]["enable_mtp"] is False
    assert config["actor"]["megatron"]["use_deterministic_algorithms"] is False

    assert config["openreward"]["enable_recursive_subagents"] is True
    assert config["openreward"]["enable_programmatic_tool_calling"] is True
    assert config["openreward"]["programmatic_tool_calling_mode"] == "orchestration_only"
    assert config["openreward"]["programmatic_tool_calling_max_tool_calls_per_execution"] == 1024
    assert config["openreward"]["environments"] == parent["openreward"]["environments"]
    assert config["openreward"]["balance_accepted_batches"] is False
    assert config["recover"] == parent["recover"]
    assert config["total_train_epochs"] == parent["total_train_epochs"] == 10
    assert config["total_train_steps"] is parent["total_train_steps"] is None


def test_qwen38_dense_nonrecursive_leaf_has_exact_safe_training_contract() -> None:
    yaml = pytest.importorskip("yaml")
    path = CONFIG_ROOT / f"{QWEN38_DENSE_16_CONFIG}.yaml"
    leaf = yaml.safe_load(path.read_text())
    config = _compose(QWEN38_DENSE_16_CONFIG)
    source_name = (
        "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
        "qwen38_dense_c12_ta20_v18"
    )
    parent = _compose(source_name)

    assert leaf["defaults"] == [source_name, "_self_"]
    assert config["experiment_name"] == "abci-or-ta-swe16-q38-v18"
    assert config["trial_name"] == "abci-ta-swe16-nr-q38d-c12-v18-t0"
    assert config["stats_logger"]["wandb"]["group"] == "abci-ta-swe16-nr-q38d-c12-v18"
    snapshot = "${oc.env:PLATOON_MODEL_SNAPSHOT}"
    assert config["tokenizer_path"] == snapshot
    assert config["actor"]["path"] == snapshot
    assert config["sglang"]["model_path"] == snapshot

    actor_match = re.fullmatch(r"megatron:d(\d+)p(\d+)t(\d+)c(\d+)", config["actor"]["backend"])
    assert actor_match is not None
    actor_dp, actor_pp, actor_tp, actor_cp = (int(value) for value in actor_match.groups())
    actor_gpus = actor_dp * actor_pp * actor_tp * actor_cp
    rollout_gpus, rollout_replicas = _rollout_size(config["rollout"]["backend"])
    assert (actor_dp, actor_pp, actor_tp, actor_cp) == (10, 2, 4, 1)
    assert actor_gpus == 80
    assert config["rollout"]["backend"] == "sglang:d6p1t8"
    assert rollout_gpus == 48
    assert rollout_replicas == 6
    assert actor_gpus + rollout_gpus == config["cluster"]["n_nodes"] * 8 == 128
    assert config["ref"]["backend"] == "${actor.backend}"
    assert "attn:" not in config["actor"]["backend"]
    assert "ffn:" not in config["actor"]["backend"]

    mb_spec = config["actor"]["mb_spec"]
    assert mb_spec["n_mbs"] == 8
    assert mb_spec["max_tokens_per_mb"] == 32768
    ppo_n_minibatches = config["actor"]["ppo_n_minibatches"]
    granularity = mb_spec.get("granularity", 1)
    engine_divisor = math.lcm(actor_pp, mb_spec.get("n_mbs_divisor", 1))
    engine_min_microbatches = math.ceil(max(mb_spec["n_mbs"], 2 * actor_pp) / engine_divisor) * engine_divisor
    datum_divisor = actor_dp * ppo_n_minibatches * granularity * engine_divisor
    minimum_datums = actor_dp * ppo_n_minibatches * granularity * engine_min_microbatches
    assert engine_divisor == 2
    assert datum_divisor == 20
    assert minimum_datums == 80

    assert config["dynamic_bs"] is True
    assert config["workflow_config"]["filter_zero_advantage_datums"] is True
    assert config["workflow_config"]["filter_zero_variance_groups"] is True
    assert config["workflow_config"]["dynamic_batch_recollection_max_rounds"] == 1
    assert config["rollout"]["consumer_batch_size"] == 8
    assert config["rollout"]["max_concurrent_rollouts"] == 12
    assert config["rollout"]["max_concurrent_rollouts"] // rollout_replicas == 2
    assert config["rollout"]["max_head_offpolicyness"] == 1

    assert config["rollout"]["return_routed_experts"] is False
    assert config["actor"]["enable_router_replay"] is False
    assert config["actor"]["router_replay_num_layers"] is None
    assert config["actor"]["router_replay_topk"] is None
    assert config["actor"]["router_replay_num_experts"] is None
    assert config["workflow_config"]["enable_router_replay"] is False
    assert config["workflow_config"]["router_replay_num_layers"] is None
    assert config["workflow_config"]["router_replay_topk"] is None
    assert config["actor"]["megatron"]["enable_mtp"] is False
    assert config["actor"]["megatron"]["use_deterministic_algorithms"] is False

    assert config["openreward"]["enable_recursive_subagents"] is False
    assert config["openreward"]["enable_programmatic_tool_calling"] is False
    assert config["openreward"]["environments"] == parent["openreward"]["environments"]
    assert config["openreward"]["balance_accepted_batches"] is False
    expected_recover = dict(parent["recover"])
    expected_recover.update(no_load_optim=True, no_save_optim=False)
    assert config["recover"] == expected_recover
    assert config["total_train_epochs"] == parent["total_train_epochs"] == 10
    assert config["total_train_steps"] is parent["total_train_steps"] is None


def test_common_launcher_validates_parameterized_index_declared_model_shards(tmp_path: Path) -> None:
    common = (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_text()
    overlay = (PBS_ROOT / "openreward-v9-abci.env").read_text()
    assert 'source "${ABCI_ENV}"' in common
    assert 'case "${OPENREWARD_MODEL_FAMILY:-}" in' in common
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN36_MODEL_SNAPSHOT}" in overlay
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in overlay
    assert "EXPECTED_MODEL_SHARDS=${OPENREWARD_EXPECTED_MODEL_SHARDS:-26}" in common
    assert 'require_uint OPENREWARD_EXPECTED_MODEL_SHARDS "${EXPECTED_MODEL_SHARDS}" 1' in common
    assert '"${PLATOON_MODEL_SNAPSHOT}" "${EXPECTED_MODEL_SHARDS}"' in common
    assert '[[ "${model_shards}" -eq 26 ]]' not in common

    match = re.search(
        r'^if ! "\$\{CONTROLLER_PYTHON_BASE\}" -I -c \'\n(?P<source>.*?)\n\' '
        r'"\$\{PLATOON_MODEL_SNAPSHOT\}" "\$\{EXPECTED_MODEL_SHARDS\}"; then$',
        common,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    validator = match.group("source")

    shard_names = [f"model-{ordinal:05d}-of-00003.safetensors" for ordinal in range(1, 4)]
    weight_map = {f"model.layers.{index}.weight": name for index, name in enumerate(shard_names)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    for name in shard_names:
        (tmp_path / name).touch()

    command = [sys.executable, "-I", "-c", validator, str(tmp_path), "3"]
    subprocess.run(command, check=True, capture_output=True, text=True)

    unexpected = tmp_path / "model-00004-of-00003.safetensors"
    unexpected.touch()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "unexpected=['model-00004-of-00003.safetensors']" in result.stderr
    unexpected.unlink()

    missing = tmp_path / shard_names[-1]
    missing.unlink()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert f"missing=['{shard_names[-1]}']" in result.stderr


@pytest.mark.parametrize(
    ("case", "expected_wandb_id", "expected_length"),
    (
        (
            CP1_PADDED_CASES[0],
            "abci-or-ta-swe16-cp1-v14_abci-ta-swe16-nr-cp1-c12-v14-t0_train",
            62,
        ),
        (
            CP1_PADDED_CASES[1],
            "abci-or-ta-swe32-cp1-v15_abci-ta-swe32-rec-cp1-c24-v15-t0_train",
            63,
        ),
    ),
    ids=("16node-v14", "32node-v15"),
)
def test_h200_cp1_padded_identity_fits_areal_wandb_limit_with_margin(
    case: dict,
    expected_wandb_id: str,
    expected_length: int,
) -> None:
    wandb_id = f"{case['experiment']}_{case['trial']}_train"
    assert wandb_id == expected_wandb_id
    assert len(wandb_id) == expected_length
    assert len(wandb_id) <= 128
    assert 128 - len(wandb_id) >= 32



def test_qwen38_dense_nonrecursive_wrapper_binds_step0_resume_release_and_leaf_contract() -> None:
    path = PBS_ROOT / QWEN38_DENSE_16_WRAPPER
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert path.stat().st_mode & 0o111 == 0o111
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r21-376d1b9-18e62c0" in wrapper
    assert "REPO_ROOT=/home/aci18912it/rasp/source/platoon" not in wrapper
    assert "RELEASE_REF=refs/tags/abci-v8r21-376d1b9-18e62c0" in wrapper
    assert "RELEASE_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_v9_release.py" in wrapper
    assert ("GATE2_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_v9_gate2_result.py") in wrapper
    assert "GATE2_RESULT=${OPENREWARD_V9_GATE2_RESULT:-}" in wrapper
    assert ("ALL_MCP_VERIFY=${REPO_ROOT}/pbs-scripts/verify_toolathlon_all_mcp_health_result.py") in wrapper
    assert "ALL_MCP_RESULT=${OPENREWARD_TOOLATHLON_ALL_MCP_HEALTH_RESULT:-}" in wrapper
    assert '${ALL_MCP_RESULT}" == /* && -f "${ALL_MCP_RESULT}" && ! -L "${ALL_MCP_RESULT}' in wrapper
    assert "submission must pass one absolute final all-MCP RESULT" in wrapper
    assert "submission must pass one absolute final v9 Gate2 RESULT" in wrapper
    assert "export OPENREWARD_V9_GATE2_RESULT" not in wrapper
    assert "export OPENREWARD_TOOLATHLON_ALL_MCP_HEALTH_RESULT" not in wrapper
    assert "ABCI_ENV=${REPO_ROOT}/pbs-scripts/openreward-v9-abci.env" in wrapper
    assert "export PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" in wrapper
    assert '"${RELEASE_VERIFY}" --release-root "${REPO_ROOT}"' in wrapper
    assert ('"${GATE2_VERIFY}" --release-root "${REPO_ROOT}" --result "${GATE2_RESULT}"') in wrapper
    assert (
        '"${ALL_MCP_VERIFY}" --release-root "${REPO_ROOT}" '
        '--result "${ALL_MCP_RESULT}" \\\n'
        '    --gate2-result "${GATE2_RESULT}"'
    ) in wrapper
    assert "#PBS -l select=16:mpiprocs=8" in wrapper
    assert "#PBS -N rasp-h200-16q38-rs0n-v20" in wrapper
    assert "#PBS -l walltime=72:00:00" in wrapper
    assert (
        f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{QWEN38_DENSE_16_CONFIG}.yaml"
        in wrapper
    )
    assert "EXPERIMENT=abci-or-ta-swe16-q38-v18" in wrapper
    assert "TRIAL=abci-ta-swe16-nr-q38d-c12-v18-t0" in wrapper
    for literal in (
        "  no_load_optim: true",
        "  no_save_optim: false",
        "  backend: megatron:d10p2t4c1",
        "  backend: sglang:d6p1t8",
        "  enable_recursive_subagents: false",
        "  enable_programmatic_tool_calling: false",
    ):
        assert f"    '{literal}'" in wrapper
    assert "step-zero recovery leaf lacks exact literal" in wrapper
    assert "-v7-trial" not in wrapper
    assert "OPENREWARD_SERVICE_NODE_COUNT=6" in wrapper
    assert "OPENREWARD_EXPECTED_RECURSIVE=0" in wrapper
    assert "OPENREWARD_EXPECTED_PTC=0" in wrapper
    assert "OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "OPENREWARD_ALLOW_PREUPDATE_RESUME=1" in wrapper
    assert "OPENREWARD_PREUPDATE_RESUME_BINDING=${PREUPDATE_BINDING}" in wrapper
    assert "OPENREWARD_PREUPDATE_LOAD_POLICY=${PREUPDATE_LOAD_POLICY}" in wrapper
    assert "OPENREWARD_PREUPDATE_NO_LOAD_OPTIM=1" in wrapper
    assert "OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=0" in wrapper
    assert "OPENREWARD_EXPECTED_START_GLOBAL_STEP=1" in wrapper
    assert "OPENREWARD_EXPECTED_RESUME_OPTIMIZER_MOMENT_KEYS=0" in wrapper
    assert "--allow-preupdate-global-step-zero" in wrapper
    assert "--expected-start-global-step 1" in wrapper
    assert '--preupdate-load-policy "${PREUPDATE_LOAD_POLICY}"' in wrapper
    assert "resume-gs0-nooptim-v8r20.success.json" in wrapper
    assert "OPENREWARD_QWEN38_MODEL_REPOSITORY:-}" in wrapper
    assert "apurvaga/Qwen3.8-27B-preserve-thinking" in wrapper
    assert "OPENREWARD_QWEN38_MODEL_REVISION:-}" in wrapper
    assert "97c725615b8aa025b800da4ceed38811358b30f1" in wrapper
    assert "OPENREWARD_QWEN38_UPSTREAM_SOURCE:-}" in wrapper
    assert "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in wrapper
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in wrapper
    assert "export OPENREWARD_EXPECTED_MODEL_SHARDS=18" in wrapper
    assert "export OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" in wrapper
    assert (
        "export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="
        "${OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
    ) in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert "PLATOON_QWEN35_GDN_PACKED" not in wrapper
    assert "OPENREWARD_EXPECTED_QWEN35_GDN_PACKED" not in wrapper
    assert "PLATOON_QWEN35_GDN_CP=" not in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' not in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' not in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()

    wandb_id = "abci-or-ta-swe16-q38-v18_abci-ta-swe16-nr-q38d-c12-v18-t0_train"
    assert len(wandb_id) == 63
    assert 128 - len(wandb_id) >= 32


def test_qwen38_dense_wrapper_binds_exact_step5_resume_and_leaf_contract() -> None:
    path = PBS_ROOT / QWEN38_DENSE_WRAPPER
    subprocess.run(["bash", "-n", str(path)], check=True)
    wrapper = path.read_text()

    assert path.stat().st_mode & 0o111 == 0o111
    assert "#PBS -l select=32:mpiprocs=8" in wrapper
    assert "#PBS -l walltime=168:00:00" in wrapper
    assert "#PBS -N rasp-h200-32q38-rs5-v16" in wrapper
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r19-376d1b9-18e62c0" in wrapper
    assert "RELEASE_REF=refs/tags/abci-v8r19-376d1b9-18e62c0" in wrapper
    assert "RELEASE_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_v9_release.py" in wrapper
    assert "GATE2_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_v9_gate2_result.py" in wrapper
    assert "ALL_MCP_VERIFY=${REPO_ROOT}/pbs-scripts/verify_toolathlon_all_mcp_health_result.py" in wrapper
    assert "ABCI_ENV=${REPO_ROOT}/pbs-scripts/openreward-v9-abci.env" in wrapper
    assert (
        f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{QWEN38_DENSE_CONFIG}.yaml"
        in wrapper
    )
    assert "EXPERIMENT=abci-or-ta-swe32-q38-v16" in wrapper
    assert "TRIAL=abci-ta-swe32-rec-q38d-c24-v16-t0" in wrapper
    assert "OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in wrapper
    assert "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" not in wrapper

    assert "OPENREWARD_QWEN38_MODEL_REPOSITORY:-}" in wrapper
    assert "apurvaga/Qwen3.8-27B-preserve-thinking" in wrapper
    assert "OPENREWARD_QWEN38_MODEL_REVISION:-}" in wrapper
    assert "97c725615b8aa025b800da4ceed38811358b30f1" in wrapper
    assert "OPENREWARD_QWEN38_UPSTREAM_SOURCE:-}" in wrapper
    assert "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" in wrapper
    assert "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}" in wrapper
    assert "export OPENREWARD_EXPECTED_MODEL_SHARDS=18" in wrapper

    for literal in (
        "  dynamic_batch_recollection_max_rounds: 2",
        "  consumer_batch_size: 16",
        "  max_head_offpolicyness: 1",
        "  return_routed_experts: false",
        "    use_deterministic_algorithms: false",
    ):
        assert f"grep -Fxc '{literal}' \"${{CONFIG}}\"" in wrapper

    assert "export OPENREWARD_EXPECTED_NNODES=32" in wrapper
    assert "export OPENREWARD_SERVICE_NODE_COUNT=12" in wrapper
    assert "export OPENREWARD_EXPECTED_RECURSIVE=1" in wrapper
    assert "export OPENREWARD_EXPECTED_PTC=1" in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=5" in wrapper
    assert "RESUME_VERIFY=${REPO_ROOT}/pbs-scripts/verify_openreward_resume_checkpoint.py" in wrapper
    assert "--expected-shards 160" in wrapper
    assert "--expected-storage-entries" not in wrapper
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in wrapper
    assert '[[ ! -e "${prior}" && ! -L "${prior}" ]]' not in wrapper
    assert '[[ ! -e "${CHECKPOINT_ROOT}" && ! -L "${CHECKPOINT_ROOT}" ]]' not in wrapper
    assert "resume-gs5-v8r16.success.json" in wrapper
    assert 'exec "${COMMON}" "${CONFIG}"' in wrapper
    assert "qsub" not in wrapper.lower()

    wandb_id = "abci-or-ta-swe32-q38-v16_abci-ta-swe32-rec-q38d-c24-v16-t0_train"
    assert len(wandb_id) == 64
    assert 128 - len(wandb_id) >= 32


def test_common_launcher_leaves_qwen35_layout_to_model_and_topology() -> None:
    common = (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_text()

    assert "OPENREWARD_EXPECTED_QWEN35_GDN_PACKED" not in common
    assert "PLATOON_QWEN35_GDN_PACKED" not in common
    assert "export PLATOON_QWEN35_GDN_CP=" not in common
    assert "checkpoint model family and actual context-parallel topology" in common
    assert "export PLATOON_QWEN35_GDN_CP_CONV_BACKEND=" in common
    forward_line = next(
        line for line in common.splitlines() if line.startswith("export PLATOON_AREAL_PBS_FORWARD_ENV=")
    )
    forwarded = forward_line.split('"', 2)[1].split()
    assert "PLATOON_QWEN35_GDN_PACKED" not in forwarded
    assert "PLATOON_QWEN35_GDN_CP" not in forwarded
    assert forwarded.count("PLATOON_QWEN35_GDN_CP_CONV_BACKEND") == 1


def test_qwen38_retry_heartbeat_is_fail_closed_and_reaches_contained_workers(tmp_path: Path) -> None:
    common = (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_text()
    scheduler = (REPO_ROOT / "platoon/train/areal/preallocated_pbs.py").read_text()
    supervisor = (REPO_ROOT / "platoon/train/areal/pbs_role_supervisor.py").read_text()
    construction = re.search(
        r"^# BEGIN OPENREWARD_WORKER_PREAMBLE_CONSTRUCTION\n"
        r"(?P<body>.*?)"
        r"^# END OPENREWARD_WORKER_PREAMBLE_CONSTRUCTION$",
        common,
        re.MULTILINE | re.DOTALL,
    )

    assert construction is not None

    def construct_preamble(expected_heartbeat: str) -> str:
        script = (
            "set -euo pipefail\n"
            "EXPECTED_NCCL_HEARTBEAT_TIMEOUT=${TEST_EXPECTED_HEARTBEAT}\n"
            "PLATOON_NODE_LOCAL_ROOT=${TEST_NODE_LOCAL_ROOT}\n"
            "RUNTIME_GUARD_BIN=${TEST_RUNTIME_GUARD_BIN}\n"
            "OPENREWARD_WORKER_JOB_VENV=${TEST_WORKER_VENV}\n"
            "cuda_lib_dirs=${TEST_CUDA_LIB_DIRS}\n"
            "JOB_STATE_DIR=${TEST_JOB_STATE_DIR}\n"
            + construction.group("body")
            + "printf '%s' \"${PLATOON_AREAL_PBS_WORKER_PREAMBLE}\"\n"
        )
        completed = subprocess.run(
            ["/bin/bash", "-c", script],
            env={
                "TEST_EXPECTED_HEARTBEAT": expected_heartbeat,
                "TEST_NODE_LOCAL_ROOT": str(tmp_path / 'node root "quoted"'),
                "TEST_RUNTIME_GUARD_BIN": str(tmp_path / "runtime guard"),
                "TEST_WORKER_VENV": str(tmp_path / "worker venv"),
                "TEST_CUDA_LIB_DIRS": str(tmp_path / "cuda libs"),
                "TEST_JOB_STATE_DIR": str(tmp_path / "job state"),
            },
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    preamble = construct_preamble("3600")
    assert 'echo "[pbs-openreward] ERROR: contained worker NCCL heartbeat retry contract is missing"' in preamble
    assert "export ERROR:" not in preamble

    inner_driver = 'mkdir() { :; }\ncd() { :; }\neval "${TEST_PREAMBLE}"\n'
    inner_environment = {
        "TEST_PREAMBLE": preamble,
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "3600",
        "PLATOON_ROLE": "actor",
        "PLATOON_WORKER_INDEX": "0",
    }
    subprocess.run(["/bin/bash", "-c", inner_driver], env=inner_environment, check=True, capture_output=True)

    missing_heartbeat = dict(inner_environment)
    missing_heartbeat.pop("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC")
    rejected = subprocess.run(
        ["/bin/bash", "-c", inner_driver],
        env=missing_heartbeat,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert "contained worker NCCL heartbeat retry contract is missing" in rejected.stderr

    mismatched_heartbeat = dict(inner_environment)
    mismatched_heartbeat["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "3599"
    rejected_mismatch = subprocess.run(
        ["/bin/bash", "-c", inner_driver],
        env=mismatched_heartbeat,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected_mismatch.returncode == 2
    assert "contained worker NCCL heartbeat retry contract is missing" in rejected_mismatch.stderr

    optional_preamble = construct_preamble("")
    subprocess.run(
        ["/bin/bash", "-c", inner_driver],
        env={
            "TEST_PREAMBLE": optional_preamble,
            "PLATOON_ROLE": "rollout",
            "PLATOON_WORKER_INDEX": "0",
        },
        check=True,
        capture_output=True,
    )

    assert (
        "EXPECTED_NCCL_HEARTBEAT_TIMEOUT="
        "${OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-}"
    ) in common
    assert "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC must equal the retry contract" in common
    assert 'PLATOON_AREAL_PBS_FORWARD_ENV+=" TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"' in common
    assert '"worker_environment": self._worker_environment(role, spec)' in scheduler
    assert 'result[f"SINGULARITYENV_{name}"] = value' in supervisor
    assert 'result[f"APPTAINERENV_{name}"] = value' in supervisor


def test_common_launcher_routes_urls_and_service_lifecycle_only_to_validated_tail() -> None:
    common = (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_text()

    assert "SERVICE_NODE_COUNT=${OPENREWARD_SERVICE_NODE_COUNT:-${EXPECTED_NNODES}}" in common
    assert 'require_uint OPENREWARD_SERVICE_NODE_COUNT "${SERVICE_NODE_COUNT}" 1' in common
    assert "SERVICE_NODE_COUNT=$((10#${SERVICE_NODE_COUNT}))" in common
    assert "((SERVICE_NODE_COUNT <= EXPECTED_NNODES))" in common
    assert "SERVICE_NODE_START=$((EXPECTED_NNODES - SERVICE_NODE_COUNT))" in common
    assert 'awk -v first="$((SERVICE_NODE_START + 1))"' in common
    assert 'mapfile -t SERVICE_NODES < <(cut -f1 "${SERVICE_NODE_MAP}")' in common

    urls_body = re.search(r"^urls_for_port\(\) \{(?P<body>.*?)^\}", common, re.MULTILINE | re.DOTALL)
    assert urls_body is not None
    assert 'for node in "${SERVICE_NODES[@]}"' in urls_body.group("body")
    assert "ALLOC_NODES" not in urls_body.group("body")

    toolathlon_pool = 'OPENREWARD_SESSION_URLS_TOOLATHLON=$(urls_for_port "${OPENREWARD_PORT}")'
    swe_pool = 'OPENREWARD_SESSION_URLS_SWE_REBENCH=$(urls_for_port "${OPENREWARD_SWE_REBENCH_PORT}")'
    controller_launch = common.index('echo "[pbs-openreward] starting host controller')
    assert toolathlon_pool in common
    assert swe_pool in common
    assert common.index(toolathlon_pool) < controller_launch
    assert common.index(swe_pool) < controller_launch

    launch_start = common.index('echo "[pbs-openreward] starting one Toolathlon and one SWE service')
    launch_end = common.index("\nendpoint_healthy()", launch_start)
    launch_body = common[launch_start:launch_end]
    assert 'done <"${SERVICE_NODE_MAP}"' in launch_body
    assert 'done <"${NODE_MAP}"' not in launch_body
    assert 'service_pids+=("$!")' in launch_body
    assert 'service_hosts+=("${host}")' in launch_body
    assert '"${#service_pids[@]}" -eq "${SERVICE_NODE_COUNT}"' in launch_body
    assert '"${#service_hosts[@]}" -eq "${SERVICE_NODE_COUNT}"' in launch_body

    health_body = re.search(
        r"^service_pools_healthy\(\) \{(?P<body>.*?)^\}",
        common,
        re.MULTILINE | re.DOTALL,
    )
    assert health_body is not None
    assert 'for index in "${!service_pids[@]}"' in health_body.group("body")
    assert "host=${service_hosts[index]}" in health_body.group("body")
    assert "ALLOC_NODES" not in health_body.group("body")


def test_v1_h200_wrappers_keep_default_all_node_service_behavior() -> None:
    for case in CASES:
        wrapper = (PBS_ROOT / case["wrapper"]).read_text()
        assert "OPENREWARD_SERVICE_NODE_COUNT" not in wrapper
