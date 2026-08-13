"""Exact contracts for the bounded latest-stack 16-node H200 benchmark."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "plugins/openreward/platoon/openreward/configs/areal"
PBS_ROOT = REPO_ROOT / "pbs-scripts"
BASE_CONFIG = "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v8"
CONTROL_CONFIG = "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_bench28_v8"

CASES = (
    {
        "slug": "cp1-c12",
        "config": CONTROL_CONFIG,
        "experiment": "abci-openreward-ta-swe-h200-16-bench28-v8-cp1-c12",
        "trial": "abci-ta-swe16-bench28-v8-cp1-c12-trial0",
        "group": "abci-ta-swe16-bench28-v8-cp1-c12",
        "actor": "megatron:(attn:d10p2t4c1|ffn:d5p2t1e8)",
        "concurrency": 12,
    },
    {
        "slug": "cp2-c12",
        "config": ("abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp2_c12_bench28_v8"),
        "experiment": "abci-openreward-ta-swe-h200-16-bench28-v8-cp2-c12",
        "trial": "abci-ta-swe16-bench28-v8-cp2-c12-trial0",
        "group": "abci-ta-swe16-bench28-v8-cp2-c12",
        "actor": "megatron:(attn:d5p2t4c2|ffn:d5p2t1e8)",
        "concurrency": 12,
    },
    {
        "slug": "cp1-c18",
        "config": ("abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c18_bench28_v8"),
        "experiment": "abci-openreward-ta-swe-h200-16-bench28-v8-cp1-c18",
        "trial": "abci-ta-swe16-bench28-v8-cp1-c18-trial0",
        "group": "abci-ta-swe16-bench28-v8-cp1-c18",
        "actor": "megatron:(attn:d10p2t4c1|ffn:d5p2t1e8)",
        "concurrency": 18,
    },
    {
        "slug": "cp1-c24",
        "config": ("abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c24_bench28_v8"),
        "experiment": "abci-openreward-ta-swe-h200-16-bench28-v8-cp1-c24",
        "trial": "abci-ta-swe16-bench28-v8-cp1-c24-trial0",
        "group": "abci-ta-swe16-bench28-v8-cp1-c24",
        "actor": "megatron:(attn:d10p2t4c1|ffn:d5p2t1e8)",
        "concurrency": 24,
    },
)


def _merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = deepcopy(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else deepcopy(value)
        return result
    return deepcopy(overlay)


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


def _without_identity(config: dict) -> dict:
    result = deepcopy(config)
    result.pop("experiment_name")
    result.pop("trial_name")
    result["stats_logger"]["wandb"].pop("group")
    return result


def _actor_branch_gpus(backend: str) -> tuple[int, int]:
    match = re.fullmatch(
        r"megatron:\(attn:d(\d+)p(\d+)t(\d+)c(\d+)\|ffn:d(\d+)p(\d+)t(\d+)e(\d+)\)",
        backend,
    )
    assert match is not None, backend
    values = tuple(int(value) for value in match.groups())
    return (
        values[0] * values[1] * values[2] * values[3],
        values[4] * values[5] * values[6] * values[7],
    )


def _wrapper(case: dict) -> Path:
    return PBS_ROOT / (f"openreward-ta-swe-16node-nonrecursive-h200-{case['slug']}-bench28-v8-abci.pbs")


def _wrapper_without_arm_identity(case: dict) -> str:
    source = _wrapper(case).read_text()
    for value, replacement in (
        (case["config"], "{CONFIG}"),
        (case["experiment"], "{EXPERIMENT}"),
        (case["trial"], "{TRIAL}"),
        (case["slug"].upper().replace("-", "_"), "{MARKER}"),
        (case["slug"].replace("-", ""), "{PBS_NAME}"),
        (case["slug"], "{SLUG}"),
    ):
        source = source.replace(value, replacement)
    return source


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["slug"])
def test_bench28_composed_matrix_is_bounded_fresh_and_ptc_off(case: dict) -> None:
    yaml = pytest.importorskip("yaml")
    leaf = yaml.safe_load((CONFIG_ROOT / f"{case['config']}.yaml").read_text())
    config = _compose(case["config"])

    assert leaf["total_train_steps"] == 28
    assert config["experiment_name"] == case["experiment"]
    assert config["trial_name"] == case["trial"]
    assert config["stats_logger"]["wandb"]["group"] == case["group"]
    assert config["total_train_steps"] == 28
    assert config["total_train_epochs"] == 10
    assert config["scheduler"]["type"] == "pbs_prealloc"
    assert config["cluster"]["n_nodes"] == 16
    assert config["cluster"]["n_gpus_per_node"] == 8
    assert config["openreward"]["enable_recursive_subagents"] is False
    assert config["openreward"]["enable_programmatic_tool_calling"] is False
    assert config["openreward"]["enable_task_tracker"] is True
    environments = config["openreward"]["environments"]
    assert [(environment["label"], environment["sampling_start_step"]) for environment in environments] == [
        ("toolathlon", 0),
        ("swe_rebench", 20),
    ]
    assert config["actor"]["backend"] == case["actor"]
    assert _actor_branch_gpus(config["actor"]["backend"]) == (80, 80)
    assert config["rollout"]["backend"] == "sglang:d6p1t8"
    assert config["rollout"]["max_concurrent_rollouts"] == case["concurrency"]
    assert config["recover"]["freq_steps"] == 1
    assert config["valid_dataset"] is None

    # AReaL derives its W&B ID from these identities plus `_train`.
    wandb_id = f"{case['experiment']}_{case['trial']}_train"
    assert len(wandb_id) <= 128


def test_bench28_storage_discovery_and_wandb_identities_are_pairwise_unique() -> None:
    for key in ("experiment", "trial", "group"):
        assert len({case[key] for case in CASES}) == len(CASES)
    receipt_paths = {f"benchmarks/{case['trial']}.step27.json" for case in CASES}
    checkpoint_paths = {f"{case['experiment']}/{case['trial']}" for case in CASES}
    name_resolve_paths = {f"{case['experiment']}/{case['trial']}" for case in CASES}
    wandb_ids = {f"{case['experiment']}_{case['trial']}_train" for case in CASES}
    assert all(len(values) == len(CASES) for values in (receipt_paths, checkpoint_paths, name_resolve_paths, wandb_ids))


def test_bench28_wrappers_share_one_lifecycle_architecture() -> None:
    control = _wrapper_without_arm_identity(CASES[0])
    assert all(_wrapper_without_arm_identity(case) == control for case in CASES[1:])


def test_bench28_arms_change_one_measured_variable_from_fresh_control() -> None:
    control = _compose(CONTROL_CONFIG)
    cp2 = _without_identity(_compose(CASES[1]["config"]))
    c18 = _without_identity(_compose(CASES[2]["config"]))
    c24 = _without_identity(_compose(CASES[3]["config"]))
    semantic_control = _without_identity(control)

    expected_cp2 = deepcopy(semantic_control)
    expected_cp2["actor"]["backend"] = "megatron:(attn:d5p2t4c2|ffn:d5p2t1e8)"
    assert cp2 == expected_cp2

    expected_c18 = deepcopy(semantic_control)
    expected_c18["rollout"]["max_concurrent_rollouts"] = 18
    assert c18 == expected_c18

    expected_c24 = deepcopy(semantic_control)
    expected_c24["rollout"]["max_concurrent_rollouts"] = 24
    assert c24 == expected_c24

    base = _compose(BASE_CONFIG)
    expected_control = _merge(
        base,
        {
            "experiment_name": CASES[0]["experiment"],
            "trial_name": CASES[0]["trial"],
            "total_train_steps": 28,
            "stats_logger": {"wandb": {"group": CASES[0]["group"]}},
        },
    )
    assert control == expected_control


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["slug"])
def test_bench28_real_structured_composition_matches_matrix(case: dict, monkeypatch) -> None:
    cli_args = pytest.importorskip("areal.api.cli_args")
    omegaconf = pytest.importorskip("omegaconf")
    config_module = pytest.importorskip("platoon.openreward.areal_config")
    for name in ("PLATOON_EXPERIMENT_ROOT", "PLATOON_MODEL_SNAPSHOT", "PLATOON_NAME_RESOLVE_ROOT"):
        monkeypatch.setenv(name, f"/tmp/platoon-bench28-config-test/{name.lower()}")

    raw, _ = cli_args.parse_cli_args(["--config", str(CONFIG_ROOT / f"{case['config']}.yaml")])
    structured = cli_args.to_structured_cfg(raw, config_module.OpenRewardArealTrainerConfig)
    config = omegaconf.OmegaConf.to_object(structured)

    assert isinstance(config, config_module.OpenRewardArealTrainerConfig)
    assert config.total_train_steps == 28
    assert config.total_train_epochs == 10
    assert config.actor.backend == case["actor"]
    assert config.rollout.max_concurrent_rollouts == case["concurrency"]
    assert config.openreward.enable_recursive_subagents is False
    assert config.openreward.enable_programmatic_tool_calling is False


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["slug"])
def test_bench28_wrapper_waits_for_cleanup_then_verifies_strong_step27_receipt(case: dict) -> None:
    path = _wrapper(case)
    assert path.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(path)], check=True)
    source = path.read_text()

    assert "#PBS -l select=16:mpiprocs=8" in source
    assert "#PBS -l walltime=168:00:00" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8-376d1b9-18e62c0" in source
    assert f"CONFIG=${{REPO_ROOT}}/plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml" in source
    assert f"EXPERIMENT={case['experiment']}" in source
    assert f"TRIAL={case['trial']}" in source
    assert "FINAL_STEP=27" in source
    assert "OPENREWARD_EXPECTED_RECURSIVE=0" in source
    assert "OPENREWARD_EXPECTED_PTC=0" in source
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_MIN_GLOBAL_STEP=${FINAL_STEP}" in source
    assert "PLATOON_TRAINING_SUCCESS_RECEIPT_FINALIZE=0" in source
    assert "PLATOON_TRAINING_CONFIG_PATH=${CONFIG}" in source
    assert "PLATOON_TRAINING_SUCCESS_NOT_BEFORE_EPOCH=${launch_epoch}" in source
    assert '"${COMMON}" "${CONFIG}" || status=$?' in source
    assert "platoon.training_success_receipt verify" in source
    assert '--global-step "${FINAL_STEP}"' in source
    assert "training SIF digest mismatch" in source
    assert "refusing stale benchmark state" in source
    assert '"${RECEIPT}" "${CANDIDATE}" "${ERROR_RECEIPT}"' in source
    assert '"${CHECKPOINT_ROOT}" "${NAME_RESOLVE_TRIAL}" "${CONTROLLER_RUNTIME_RECEIPT}"' in source
    assert "openreward-controller-runtime-receipt.py" in source
    assert "read-controller-python" in source
    assert '--controller-cache-root "${PLATOON_PROJECT_ROOT}/cache/openreward-controller-venvs"' in source
    assert '"${CONTROLLER_PYTHON}" -I -m platoon.training_success_receipt finalize-candidate' in source
    assert '--candidate "${CANDIDATE}"' in source
    assert '--receipt "${RECEIPT}"' in source
    assert "--remote-timeout-seconds 300" in source
    assert "openreward_update_gate" not in source
    assert "qsub" not in source.lower()
    assert "qsig" not in source.lower()
    assert 'exec "${COMMON}"' not in source

    common_call = source.index('"${COMMON}" "${CONFIG}" || status=$?')
    cleanup_check = source.index('"${HOST_PYTHON}" -I -c', common_call)
    runtime_read = source.index("read-controller-python", cleanup_check)
    receipt_finalize = source.index("platoon.training_success_receipt finalize-candidate", runtime_read)
    receipt_verify = source.index("platoon.training_success_receipt verify", receipt_finalize)
    final_success = source.index("OPENREWARD_BENCH28_", receipt_verify)
    assert common_call < cleanup_check < runtime_read < receipt_finalize < receipt_verify < final_success


def _empty_trial_pruner() -> str:
    source = _wrapper(CASES[0]).read_text()
    match = re.search(
        r'"\$\{HOST_PYTHON\}" -I -c \'(?P<program>.*?)\' "\$\{NAME_RESOLVE_TRIAL\}"',
        source,
        re.DOTALL,
    )
    assert match is not None
    return match.group("program")


def test_bench28_empty_name_resolution_tree_is_pruned_only_bottom_up(tmp_path: Path) -> None:
    trial = tmp_path / "user/experiment/trial"
    (trial / "workers/actor/0").mkdir(parents=True)

    subprocess.run([sys.executable, "-I", "-c", _empty_trial_pruner(), str(trial)], check=True)

    assert not trial.exists()
    assert trial.parent.is_dir()


def test_bench28_trial_pruner_is_a_noop_when_cleanup_removed_the_tree(tmp_path: Path) -> None:
    trial = tmp_path / "already-absent"

    subprocess.run([sys.executable, "-I", "-c", _empty_trial_pruner(), str(trial)], check=True)

    assert not trial.exists()


def test_bench28_trial_pruner_rejects_an_indirect_parent(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_trial = real_root / "trial"
    real_trial.mkdir(parents=True)
    indirect_root = tmp_path / "indirect"
    indirect_root.symlink_to(real_root, target_is_directory=True)
    indirect_trial = indirect_root / "trial"

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _empty_trial_pruner(), str(indirect_trial)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert real_trial.is_dir()


@pytest.mark.parametrize("kind", ("file", "symlink", "fifo"))
def test_bench28_trial_pruner_rejects_every_surviving_entry(tmp_path: Path, kind: str) -> None:
    trial = tmp_path / "trial"
    workers = trial / "workers"
    workers.mkdir(parents=True)
    entry = workers / "ENTRY"
    if kind == "file":
        entry.write_text("live record")
    elif kind == "symlink":
        entry.symlink_to(tmp_path / "missing")
    else:
        os.mkfifo(entry)

    completed = subprocess.run(
        [sys.executable, "-I", "-c", _empty_trial_pruner(), str(trial)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert trial.is_dir()
    assert os.path.lexists(entry)


def test_gate2_requires_every_benchmark_leaf_and_host_validates_but_does_not_archive_wrappers() -> None:
    gate2 = (PBS_ROOT / "openreward-immutable-runtime-gate2.pbs").read_text()
    for case in CASES:
        member = f"plugins/openreward/platoon/openreward/configs/areal/{case['config']}.yaml"
        assert gate2.count(member) == 1
    for case in CASES:
        wrapper = _wrapper(case).name
        assert gate2.count(wrapper) == 1
        assert (
            f"pbs-scripts/{wrapper}"
            in gate2[gate2.index("for shell_asset in") : gate2.index("done", gate2.index("for shell_asset in"))]
        )
        assert (
            f"pbs-scripts/{wrapper}"
            not in gate2[
                gate2.index("for archive_member in") : gate2.index("done", gate2.index("for archive_member in"))
            ]
        )
    assert Path(__file__).name not in gate2


def test_bench28_receipt_gate_defers_argument_evaluation_until_final_step() -> None:
    receipt = (REPO_ROOT / "platoon/training_success_receipt.py").read_text()
    trainer = (REPO_ROOT / "platoon/train/areal/rl.py").read_text()

    assert 'MIN_GLOBAL_STEP_ENV = "PLATOON_TRAINING_SUCCESS_RECEIPT_MIN_GLOBAL_STEP"' in receipt
    assert "def candidate_publication_enabled(global_step:" in receipt
    assert "candidate_publication_enabled(global_step)" in trainer
