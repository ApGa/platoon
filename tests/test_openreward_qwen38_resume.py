from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
VERIFY = PBS_ROOT / "verify_openreward_resume_checkpoint.py"
COMMON = PBS_ROOT / "openreward-ta-swe-prealloc-common.sh"
WRAPPER = PBS_ROOT / "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs"
SUCCESSOR_WRAPPER = PBS_ROOT / "openreward-ta-swe-32node-recursive-qwen38-resume-h200-v11-abci.pbs"
WRAPPER_16 = PBS_ROOT / "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs"
NOOPTIM_WRAPPER_16 = PBS_ROOT / "openreward-ta-swe-16node-nonrecursive-qwen38-gs0-nooptim-h200-v10-abci.pbs"
EXPERIMENT = "abci-or-ta-swe32-q38-v16"
TRIAL = "abci-ta-swe32-rec-q38d-c24-v16-t0"
EXPERIMENT_16 = "abci-or-ta-swe16-q38-v18"
TRIAL_16 = "abci-ta-swe16-nr-q38d-c12-v18-t0"
MODEL_REPOSITORY = "apurvaga/Qwen3.8-27B-preserve-thinking"
MODEL_REVISION = "97c725615b8aa025b800da4ceed38811358b30f1"
CONFIG_RELATIVE = Path(
    "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen38_dense_c24_ta20_v16.yaml"
)
CONFIG_RELATIVE_16 = Path(
    "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
    "qwen38_dense_c12_ta20_v18.yaml"
)
NOOPTIM_CONFIG_RELATIVE_16 = Path(
    "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_"
    "qwen38_dense_c12_ta20_v20_gs0_no_load_optim.yaml"
)
SUCCESSOR_CONFIG_RELATIVE = Path(
    "plugins/openreward/platoon/openreward/configs/areal/"
    "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_"
    "qwen38_dense_c24_ta20_v20.yaml"
)
ANCESTOR_SHA256 = "b432e3a12ff78921e7aac6cff3914f1c4c5aeffd9f08e3c9918ad5daa7714a22"


def _merge(base, overlay):
    if isinstance(base, dict) and isinstance(overlay, dict):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = _merge(result[key], value) if key in result else value
        return result
    return overlay


def _compose_config(relative: Path, ancestry: tuple[Path, ...] = ()) -> dict:
    yaml = pytest.importorskip("yaml")
    if relative in ancestry:
        raise AssertionError(f"recursive defaults: {relative}")
    path = REPO_ROOT / relative
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = value.pop("defaults", [])
    result: dict = {}
    for default in defaults:
        if default == "_self_":
            result = _merge(result, value)
        else:
            parent = relative.parent / f"{default}.yaml"
            result = _merge(result, _compose_config(parent, (*ancestry, relative)))
    if "_self_" not in defaults:
        result = _merge(result, value)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _size_manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode() + b"\0" + str(path.stat().st_size).encode() + b"\n")
    return digest.hexdigest()


def _content_manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode() + b"\0" + _sha256(path).encode() + b"\n")
    return digest.hexdigest()


def _sample_manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        size = path.stat().st_size
        sample = hashlib.sha256()
        sample.update(b"openreward-resume-checkpoint-sample-v2\0")
        sample.update(str(size).encode() + b"\0")
        with path.open("rb") as stream:
            if size <= 3 * 1024 * 1024:
                sample.update(stream.read())
            else:
                sample.update(stream.read(1024 * 1024))
                stream.seek((size - 1024 * 1024) // 2)
                sample.update(stream.read(1024 * 1024))
                stream.seek(size - 1024 * 1024)
                sample.update(stream.read(1024 * 1024))
        digest.update(path.name.encode() + b"\0" + str(size).encode() + b"\0" + sample.hexdigest().encode() + b"\n")
    return digest.hexdigest()


def _candidate_sampled_sha256(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(b"platoon-checkpoint-sample-v1\0")
    digest.update(str(size).encode() + b"\0")
    with path.open("rb") as stream:
        if size <= 2 * 1024 * 1024:
            digest.update(stream.read())
        else:
            digest.update(stream.read(1024 * 1024))
            stream.seek(size - 1024 * 1024)
            digest.update(stream.read(1024 * 1024))
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _run_common_static_config_inspection(path: Path) -> subprocess.CompletedProcess[str]:
    source = COMMON.read_text(encoding="utf-8")
    begin = source.index('CONFIG=$(readlink -f "${CONFIG}")')
    end = source.index("for model_file in config.json", begin)
    harness = """#!/usr/bin/env bash
set -euo pipefail
CONFIG=$1
EXPECTED_EXPERIMENT=abci-or-ta-swe32-q38-v16
EXPECTED_TRIAL=abci-ta-swe32-rec-q38d-c24-v16-t0
EXPECTED_RECURSIVE=1
EXPECTED_PTC=1
ENVIRONMENT_SET=toolathlon_only
die() {
    echo "[pbs-openreward] ERROR: $*" >&2
    exit 2
}
""" + source[begin:end]
    return subprocess.run(
        ["bash", "-c", harness, "common-static-inspection", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_qwen38_v20_leaf_executes_common_static_launch_inspection(tmp_path: Path) -> None:
    config = REPO_ROOT / SUCCESSOR_CONFIG_RELATIVE
    result = _run_common_static_config_inspection(config)
    assert result.returncode == 0, result.stderr

    inherited_only = tmp_path / "inherited-only.yaml"
    inherited_only.write_text(
        "defaults:\n"
        "  - abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_"
        "h200_qwen38_dense_c24_ta20_v16\n"
        "  - _self_\n"
        "workflow_config:\n"
        "  rollout_config:\n"
        "    max_steps: 150\n",
        encoding="utf-8",
    )
    result = _run_common_static_config_inspection(inherited_only)
    assert result.returncode == 2
    assert "config experiment '' does not match 'abci-or-ta-swe32-q38-v16'" in result.stderr


def test_qwen38_v20_successor_composes_exact_timeout_and_step_contract() -> None:
    config = _compose_config(SUCCESSOR_CONFIG_RELATIVE)
    source = _compose_config(CONFIG_RELATIVE)
    rollout_config = config["workflow_config"]["rollout_config"]

    assert config["experiment_name"] == source["experiment_name"] == EXPERIMENT
    assert config["trial_name"] == source["trial_name"] == TRIAL
    assert rollout_config["max_steps"] == 150
    assert rollout_config["timeout"] == 7200
    assert rollout_config["step_timeout"] == 5400
    assert config["openreward"]["subagent_default_max_steps"] == 100
    assert config["openreward"]["subagent_reward_judge_max_steps"] == 100
    assert config["openreward"]["subagent_max_depth"] == source["openreward"]["subagent_max_depth"] == 2

    # The controller timeout admits one 7,380s child cohort ahead, this
    # workflow's cohort, and the established 1,440s export/cleanup margin.
    child_hard_timeout = rollout_config["timeout"] + 120 + 60
    parent_future_timeout = child_hard_timeout + 30
    assert (child_hard_timeout, parent_future_timeout) == (7380, 7410)
    assert config["rollout"]["request_timeout"] == 2 * child_hard_timeout + 1440 == 16200
    assert config["rollout"]["agent"]["session_timeout_seconds"] == 9000

    # All requested non-timeout behavior stays on the active dense lineage.
    assert config["dynamic_bs"] is source["dynamic_bs"] is True
    assert config["actor"] == source["actor"]
    assert config["ref"] == source["ref"]
    assert config["sglang"] == source["sglang"]
    assert config["recover"] == source["recover"]
    assert config["workflow_config"]["filter_zero_advantage_datums"] is True
    assert config["workflow_config"]["filter_zero_variance_groups"] is True
    assert config["workflow_config"]["dynamic_batch_recollection_max_rounds"] == 2
    assert config["rollout"]["backend"] == source["rollout"]["backend"] == "sglang:d12p1t8"
    assert config["rollout"]["return_routed_experts"] is False
    assert config["actor"]["mb_spec"]["max_tokens_per_mb"] == 32768
    assert config["sglang"]["context_length"] == 32769

    workflow_source = (REPO_ROOT / "platoon/train/areal/workflows/group_rollout_workflow.py").read_text(
        encoding="utf-8"
    )
    child_source = (REPO_ROOT / "platoon/train/areal/subprocess_worker.py").read_text(encoding="utf-8")
    assert "_SUBPROCESS_INIT_BUDGET_SECONDS = 120" in workflow_source
    assert "_SUBPROCESS_CLEANUP_GRACE_SECONDS = 60" in workflow_source
    assert "_PARENT_FUTURE_GRACE_SECONDS = 30" in workflow_source
    assert '(config_dict.get("timeout") or 900)' in child_source
    assert "+ 120  # AppWorld init budget" in child_source
    assert "+ 60  # cleanup grace" in child_source


def test_qwen38_16_gs0_nooptim_leaf_changes_only_one_shot_recovery_load() -> None:
    source = _compose_config(CONFIG_RELATIVE_16)
    successor = _compose_config(NOOPTIM_CONFIG_RELATIVE_16)

    assert successor["experiment_name"] == source["experiment_name"] == EXPERIMENT_16
    assert successor["trial_name"] == source["trial_name"] == TRIAL_16
    assert successor["recover"]["no_load_optim"] is True
    assert successor["recover"]["no_save_optim"] is False
    assert source["recover"].get("no_load_optim", False) is False
    assert source["recover"].get("no_save_optim", False) is False

    normalized = dict(successor)
    normalized_recover = dict(successor["recover"])
    normalized_recover.pop("no_load_optim")
    normalized_recover.pop("no_save_optim")
    normalized["recover"] = normalized_recover
    assert normalized == source

    assert successor["cluster"]["n_nodes"] == 16
    assert successor["actor"]["backend"] == "megatron:d10p2t4c1"
    assert successor["rollout"]["backend"] == "sglang:d6p1t8"
    assert successor["openreward"]["enable_recursive_subagents"] is False
    assert successor["openreward"]["enable_programmatic_tool_calling"] is False


def test_successor_stats_cursor_matches_save_before_commit_resume_order() -> None:
    trainer_source = (REPO_ROOT / "platoon/train/areal/rl.py").read_text(encoding="utf-8")
    loop = trainer_source.index("for global_step in range(start_step, max_steps):")
    checkpoint = trainer_source.index("self._save_recover_checkpoint(", loop)
    commit = trainer_source.index("self._export_and_commit_stats(", checkpoint)
    candidate = trainer_source.index("maybe_publish_training_success_candidate(", commit)
    assert checkpoint < commit < candidate

    areal_candidates = (
        REPO_ROOT / "external/AReaL",
        Path("/home/aci18912it/rasp/releases/platoon-v8r17-376d1b9-18e62c0/external/AReaL"),
    )
    areal_root = next((path for path in areal_candidates if path.is_dir()), None)
    if areal_root is None:
        pytest.skip("materialized pinned AReaL checkout is unavailable")
    stats_source = (areal_root / "areal/utils/stats_logger.py").read_text(encoding="utf-8")
    recover_source = (areal_root / "areal/utils/recover.py").read_text(encoding="utf-8")
    assert "stats_logger_info=stats_logger.state_dict()" in recover_source
    assert "stats_logger.load_state_dict(recover_info.stats_logger_info)" in recover_source
    assert "log_step = max(global_step, self._last_commit_step + 1)" in stats_source
    # gs5 persisted cursor4; gs6 saves cursor4 before commit6. Loading that
    # exact state for the next step still logs at 7, never replaying 5 or 6.
    assert max(7, 4 + 1) == 7


def test_areal_no_load_optim_skips_only_load_and_normal_save_keeps_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patches_path = REPO_ROOT / "platoon/train/areal/patches.py"
    spec = importlib.util.spec_from_file_location("nooptim_patch_fixture", patches_path)
    assert spec is not None and spec.loader is not None
    patches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patches)

    class SaveLoadMeta:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    checkpoint = tmp_path / "recover_checkpoint"

    class Saver:
        @staticmethod
        def get_recover_checkpoint_path(*_args, **_kwargs):
            return str(checkpoint)

    class Handler:
        def _save_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("unpatched save")

        def _load_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("unpatched load")

    class Logger:
        @staticmethod
        def info(_message):
            pass

        @staticmethod
        def warning(_message):
            pass

    areal = types.ModuleType("areal")
    api = types.ModuleType("areal.api")
    api.SaveLoadMeta = SaveLoadMeta
    utils = types.ModuleType("areal.utils")
    recover = types.ModuleType("areal.utils.recover")
    recover.RecoverHandler = Handler
    recover.logger = Logger()
    saver = types.ModuleType("areal.utils.saver")
    saver.Saver = Saver
    areal.api = api
    areal.utils = utils
    utils.recover = recover
    utils.saver = saver
    for name, module in {
        "areal": areal,
        "areal.api": api,
        "areal.utils": utils,
        "areal.utils.recover": recover,
        "areal.utils.saver": saver,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    patches._patch_areal_recover_checkpoint_rotation()
    handler = Handler()
    handler.config = SimpleNamespace(
        experiment_name="experiment",
        trial_name="trial",
        fileroot=str(tmp_path),
        no_load_optim=False,
        no_save_optim=False,
    )
    checkpoint.mkdir()

    class Engine:
        def __init__(self):
            self.loaded = []
            self.saved = []

        def load(self, meta):
            self.loaded.append(meta)
            if meta.with_optim:
                raise RuntimeError("optimizer moments are absent")

        def save(self, meta):
            self.saved.append(meta)
            Path(meta.path, "saved").write_text("ok\n", encoding="utf-8")

    engine = Engine()
    with pytest.raises(RuntimeError, match="optimizer moments are absent"):
        handler._load_checkpoint(engine)
    assert engine.loaded[-1].with_optim is True

    handler.config.no_load_optim = True
    handler._load_checkpoint(engine)
    assert engine.loaded[-1].with_optim is False

    # no_load_optim never feeds the save plan. With the independent normal
    # no_save_optim=False setting, the first gs>=1 checkpoint includes optimizer
    # state and therefore becomes eligible for ordinary optimizer-loading resume.
    handler._save_checkpoint(engine)
    assert engine.saved[-1].with_optim is True
    assert checkpoint.joinpath("saved").read_text(encoding="utf-8") == "ok\n"


def test_qwen38_v20_successor_wrapper_binds_frozen_gs6_before_services() -> None:
    subprocess.run(["bash", "-n", str(SUCCESSOR_WRAPPER)], check=True)
    source = SUCCESSOR_WRAPPER.read_text(encoding="utf-8")
    assert SUCCESSOR_WRAPPER.stat().st_mode & 0o111 == 0o111
    assert "#PBS -N rasp-h200-32q38-rsp6-v20" in source
    assert "BINDING_STATE=FROZEN_POST_STEP6_CHECKPOINT" in source
    assert "UNOBSERVED" not in source
    assert "--successor-binding qwen38-32-v20-post-step6" in source
    assert "--expected-source-config-sha256" in source
    assert "--authorized-ancestor-receipt" in source
    assert "resume-post-step6-v8r20.success.json" in source
    assert "resume-gs5-v8r16.success.candidate.json" in source
    assert "SOURCE_PBS_JOB_ID=2156202.pbs1" in source
    assert "EXPECTED_GLOBAL_STEP=6" in source
    assert "EXPECTED_DCP_STATE_KEYS=1472" in source
    assert "EXPECTED_DCP_STORAGE_ENTRIES=18297" in source
    assert "EXPECTED_LR_NUM_STEPS=7" in source
    assert "EXPECTED_SAMPLES_YIELDED=336" in source
    assert "EXPECTED_ITERATOR_YIELDED=42" in source
    assert "12a5a385826dbb2efb4a6dbffcdc057dce7063def8ade0c0d131dae1f802cd94" in source
    assert "08a2837d33589671fae43e796a981ad41e16ca335ab696a1748cd660f1a1b991" in source
    assert '[[ "${EXPECTED_GLOBAL_STEP}" == 6 && \\' in source
    assert '"${EXPECTED_GLOBAL_STEP}" -ge 6' not in source
    assert "6699027815f1628a6522c0520d3e30b17df124a41e3583b6ff4a3e4937b556f8" in source
    assert "57410a3d50b2a3bf45991568abc687f334f4d02179a7d207da9c32e1872735c4" in source
    assert "9417c54953c5651d5c9a73649657ca50da74995847d77258e0a256e193674251" in source
    assert "00c8d7ccdd27f49bd248369ffa45416d89b53dcb23a45ec9d5ea523aeeb7bddb" in source
    assert "export OPENREWARD_MODEL_FAMILY=qwen38" in source
    assert source.index("export OPENREWARD_MODEL_FAMILY=qwen38") < source.index('source "${ABCI_ENV}"')
    guard = source.index('[[ "${BINDING_STATE}" == FROZEN_POST_STEP6_CHECKPOINT ]]')
    assert guard < source.index('source "${ABCI_ENV}"')
    assert guard < source.index('"${RESUME_VERIFY}"')
    assert guard < source.index('exec "${COMMON}"')

    source_log_check = source.index('sha256sum -- "${SOURCE_LOG}"')
    assert guard < source_log_check < source.index('source "${ABCI_ENV}"')
    assert "qsub" not in source.lower()


def _build_authorized_fixture(
    tmp_path: Path, *, preupdate: bool = False, successor_config: bool = False
) -> tuple[list[str], dict[str, Path]]:
    if preupdate and successor_config:
        raise ValueError("preupdate fixtures do not use a successor config")
    experiment = EXPERIMENT_16 if preupdate else EXPERIMENT
    trial = TRIAL_16 if preupdate else TRIAL
    config_relative = CONFIG_RELATIVE_16 if preupdate else CONFIG_RELATIVE
    shard_count = 80 if preupdate else 160
    global_step = 0 if preupdate else (6 if successor_config else 5)
    source_job_id = "2155389.pbs1" if preupdate else "2154492.pbs1"
    checkpoint_root = tmp_path / "checkpoints" / "test-user" / experiment / trial
    checkpoint = checkpoint_root / "default" / "recover_checkpoint"
    recover_info = checkpoint_root / "recover_info"
    checkpoint.mkdir(parents=True)
    recover_info.mkdir()

    checkpoint_files: list[Path] = []
    for ordinal in range(shard_count):
        shard = checkpoint / f"__{ordinal}_0.distcp"
        shard.write_bytes(f"shard-{ordinal}".encode())
        checkpoint_files.append(shard)
    dcp_metadata = checkpoint / ".metadata"
    dcp_metadata.write_bytes(b"prefix\0" + f"{checkpoint}_new".encode() + b"\0suffix")
    common = checkpoint / "common.pt"
    common.write_bytes(b"common")
    metadata = checkpoint / "metadata.json"
    _write_json(
        metadata,
        {
            "sharded_backend": "torch_dist",
            "sharded_backend_version": 1,
            "common_backend": "torch",
            "common_backend_version": 1,
        },
    )
    checkpoint_files.extend((dcp_metadata, common, metadata))
    for path in checkpoint_files:
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))

    _write_json(
        recover_info / "step_info.json",
        {
            "epoch": 0,
            "epoch_step": global_step,
            "global_step": global_step,
            "steps_per_epoch": 846,
        },
    )
    _write_json(
        recover_info / "stats_logger_info.json",
        {"last_commit_step": 4 if successor_config else global_step - 1},
    )
    _write_json(
        recover_info / "checkpoint_info.json",
        {
            "step": {
                "frequency_steps": 1,
                "steps": global_step + 1,
                "last_steps": global_step + 1,
                "interval_steps": 1,
            }
        },
    )
    _write_json(recover_info / "evaluator_info.json", {})
    _write_json(recover_info / "saver_info.json", {})
    (recover_info / "dataloader_info.pkl").write_bytes(b"pickle")
    for path in recover_info.iterdir():
        os.utime(path, ns=(2_000_000_000, 2_000_000_000))

    model_snapshot = tmp_path / "model-snapshot"
    model_snapshot.mkdir()
    source_release = tmp_path / "source-release"
    source_config = source_release / config_relative
    source_config.parent.mkdir(parents=True)
    config_bytes = (
        (REPO_ROOT / (CONFIG_RELATIVE_16 if preupdate else CONFIG_RELATIVE)).read_bytes()
        if successor_config or preupdate
        else (f"experiment_name: {experiment}\ntrial_name: {trial}\nrecover:\n  mode: auto\n").encode()
    )
    source_config.write_bytes(config_bytes)
    current_config = tmp_path / "current-config.yaml"
    current_config.write_bytes(
        (REPO_ROOT / SUCCESSOR_CONFIG_RELATIVE).read_bytes()
        if successor_config
        else ((REPO_ROOT / NOOPTIM_CONFIG_RELATIVE_16).read_bytes() if preupdate else config_bytes)
    )
    source_ref = "refs/tags/source-release"
    _write_json(
        source_release / "pbs-scripts" / "openreward-v9-release.json",
        {
            "release_root": str(source_release),
            "release_ref": source_ref,
            "model_artifacts": {
                ("nonrecursive_16node" if preupdate else "recursive_32node"): {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "snapshot_path": str(model_snapshot),
                    "architecture": "dense_transformer",
                    "safetensors_shard_count": 18,
                }
            },
        },
    )
    overlay = source_release / "pbs-scripts" / "openreward-v9-abci.env"
    overlay.write_text(
        "\n".join(
            (
                f"export OPENREWARD_QWEN38_MODEL_REPOSITORY={MODEL_REPOSITORY}",
                f"export OPENREWARD_QWEN38_MODEL_REVISION={MODEL_REVISION}",
                f"export OPENREWARD_QWEN38_MODEL_SNAPSHOT={model_snapshot}",
                "export PLATOON_MODEL_SNAPSHOT=${OPENREWARD_QWEN38_MODEL_SNAPSHOT}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    source_wrapper = (
        source_release
        / "pbs-scripts"
        / (
            "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs"
            if preupdate
            else "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs"
        )
    )
    source_wrapper.write_text(
        "\n".join(
            (
                f"REPO_ROOT={source_release}",
                f"RELEASE_REF={source_ref}",
                f"EXPERIMENT={experiment}",
                f"TRIAL={trial}",
                f"CONFIG=${{REPO_ROOT}}/{config_relative}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(source_release)], check=True)
    subprocess.run(["git", "-C", str(source_release), "config", "user.name", "test"], check=True)
    subprocess.run(
        ["git", "-C", str(source_release), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source_release), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source_release), "commit", "-qm", "source"], check=True)
    subprocess.run(["git", "-C", str(source_release), "tag", "source-release"], check=True)

    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    source_candidate = receipt_dir / (
        f"{trial}.resume-gs5-v8r16.success.candidate.json" if successor_config else f"{trial}.success.candidate.json"
    )
    ancestor_candidate = receipt_dir / f"{trial}.success.candidate.json"
    source_log = tmp_path / f"{source_job_id}.OU"
    if preupdate:
        source_commit = subprocess.run(
            ["git", "-C", str(source_release), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_log.write_text(
            "\n".join(
                (
                    f"openreward_v9_release=PASS release_commit={source_commit}",
                    "[h200-16-v9] accepted gate2=/gate2 all-MCP=/all-mcp",
                    (f"[pbs-openreward] starting host controller experiment={experiment} trial={trial} nodes=16"),
                    (f"Recover INFO: Saved recover checkpoint to {checkpoint_root}/default/recover_checkpoint"),
                    ("RLTrainer ERROR: Training failed with exception: optimizer_batch_datums must be an integer >= 1"),
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        candidate_step = global_step if successor_config else 0
        recover_evidence: dict[str, object] = {
            "checkpoint_root": str(checkpoint_root),
            "global_step": candidate_step,
        }
        if successor_config:
            recover_evidence = {
                "fileroot": str(checkpoint_root.parent.parent.parent.parent),
                "checkpoint_root": str(checkpoint_root),
                "recover_info_path": str(recover_info),
                "recover_info": {
                    path.name: {"size": path.stat().st_size, "sha256": _sha256(path)} for path in recover_info.iterdir()
                },
                "recover_checkpoint_path": str(checkpoint),
                "recover_checkpoint_file_count": len(checkpoint_files),
                "recover_checkpoint_total_bytes": sum(path.stat().st_size for path in checkpoint_files),
                "recover_checkpoint_manifest": {
                    path.name: {
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "sample_sha256": _candidate_sampled_sha256(path),
                    }
                    for path in checkpoint_files
                },
                "global_step": candidate_step,
            }
        _write_json(
            source_candidate,
            {
                "schema_version": 2,
                "gate": "abci-openreward-one-update-success",
                "kind": "candidate",
                "pbs_job_id": source_job_id,
                "experiment_name": experiment,
                "trial_name": trial,
                "config_path": str(source_config),
                "config_sha256": _sha256(source_config),
                "global_step": candidate_step,
                **(
                    {
                        "stats_logger_last_commit_step": candidate_step,
                        "optimizer_update_successful": True,
                        "optimizer_batch_datums": 40,
                        "optimizer_action_tokens": 100,
                        "weight_sync_succeeded": True,
                        "expected_version": candidate_step + 1,
                        "actor_version": candidate_step + 1,
                        "rollout_version": candidate_step + 1,
                    }
                    if successor_config
                    else {}
                ),
                "recover": recover_evidence,
                "wandb": {"run_id": f"{experiment}_{trial}_train"},
            },
        )
        if successor_config:
            _write_json(ancestor_candidate, {"authorized_ancestor": True})
    target_receipt = receipt_dir / (
        f"{trial}.resume-gs0-nooptim-v8r20.success.json"
        if preupdate
        else (
            f"{trial}.resume-post-step6-v8r20.success.json"
            if successor_config
            else f"{trial}.resume-gs5-v8r16.success.json"
        )
    )
    total_bytes = sum(path.stat().st_size for path in checkpoint.iterdir())
    command = [
        str(VERIFY),
        "--checkpoint-root",
        str(checkpoint_root),
        "--expected-checkpoint-root",
        str(checkpoint_root),
        "--expected-global-step",
        str(global_step),
        "--expected-start-global-step",
        str(global_step + 1),
        "--expected-steps-per-epoch",
        "846",
        "--expected-shards",
        str(shard_count),
        "--expected-total-bytes",
        str(total_bytes),
        "--expected-checkpoint-layout-sha256",
        _size_manifest_sha256(list(checkpoint.iterdir())),
        "--expected-checkpoint-sample-manifest-sha256",
        _sample_manifest_sha256(list(checkpoint.iterdir())),
        "--expected-fixed-content-sha256",
        _content_manifest_sha256([dcp_metadata, common, metadata]),
        "--expected-recover-info-content-sha256",
        _content_manifest_sha256(list(recover_info.iterdir())),
        "--expected-experiment",
        experiment,
        "--expected-trial",
        trial,
        "--current-config",
        str(current_config),
        "--expected-config-sha256",
        _sha256(current_config),
        "--source-pbs-job-id",
        source_job_id,
        "--source-release-root",
        str(source_release),
        "--source-release-ref",
        source_ref,
        "--source-config-relative",
        str(config_relative),
        "--model-repository",
        MODEL_REPOSITORY,
        "--model-revision",
        MODEL_REVISION,
        "--model-snapshot",
        str(model_snapshot),
        "--target-receipt",
        str(target_receipt),
    ]
    if successor_config:
        command[command.index("--source-pbs-job-id") : command.index("--source-pbs-job-id")] = [
            "--expected-source-config-sha256",
            _sha256(source_config),
            "--successor-binding",
            "qwen38-32-v20-post-step6",
            "--authorized-ancestor-receipt",
            str(ancestor_candidate),
            "--expected-authorized-ancestor-receipt-sha256",
            _sha256(ancestor_candidate),
        ]
        # Production pins the real ancestor digest. This isolated executable
        # copy substitutes only that immutable evidence value so synthetic
        # checkpoint fixtures can exercise the rest of the exact binding.
        fixture_verify = tmp_path / "verify-successor-fixture.py"
        verifier_source = VERIFY.read_text(encoding="utf-8")
        # The v8r22 bindings legitimately reuse this historical ancestor. Only
        # substitute the first occurrence, which is the legacy v20 constant
        # exercised by this synthetic fixture.
        assert verifier_source.count(ANCESTOR_SHA256) >= 1
        fixture_verify.write_text(
            verifier_source.replace(ANCESTOR_SHA256, _sha256(ancestor_candidate), 1),
            encoding="utf-8",
        )
        fixture_verify.chmod(0o755)
        command[0] = str(fixture_verify)
    if preupdate:
        command[command.index("--source-pbs-job-id") : command.index("--source-pbs-job-id")] = [
            "--expected-source-config-sha256",
            _sha256(source_config),
            "--source-log",
            str(source_log),
            "--expected-source-log-sha256",
            _sha256(source_log),
            "--allow-preupdate-global-step-zero",
            "--preupdate-binding",
            "qwen38-16-v18-job2155389",
            "--preupdate-load-policy",
            "qwen38-16-v20-gs0-no-load-optim",
        ]
    else:
        command[command.index("--source-pbs-job-id") : command.index("--source-pbs-job-id")] = [
            "--source-candidate",
            str(source_candidate),
            "--expected-source-candidate-sha256",
            _sha256(source_candidate),
        ]
    return command, {
        "checkpoint": checkpoint,
        "recover_info": recover_info,
        "receipt_dir": receipt_dir,
        "target_receipt": target_receipt,
        "source_log": source_log,
        "source_config": source_config,
        "current_config": current_config,
        "ancestor_candidate": ancestor_candidate,
        "source_candidate": source_candidate,
    }


def test_resume_verifier_executes_complete_authorization(tmp_path: Path) -> None:
    command, _ = _build_authorized_fixture(tmp_path)
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "openreward_resume_checkpoint=PASS global_step=5 shards=160 " in result.stdout


def test_resume_verifier_binds_distinct_source_and_successor_config_hashes(
    tmp_path: Path,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    paths["current_config"].write_text("changed after binding\n", encoding="utf-8")
    rejected = subprocess.run(command, check=False, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "current resume config identity mismatch" in rejected.stderr


def test_successor_binding_rejects_later_checkpoint_step(tmp_path: Path) -> None:
    command, _ = _build_authorized_fixture(tmp_path, successor_config=True)
    command[command.index("--expected-global-step") + 1] = "7"

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "expected steps/counts/digests are invalid" in result.stderr


def test_successor_binding_rejects_self_consistent_alternative_config(
    tmp_path: Path,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    paths["current_config"].write_text("alternative successor\n", encoding="utf-8")
    command[command.index("--expected-config-sha256") + 1] = _sha256(paths["current_config"])

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "expected steps/counts/digests are invalid" in result.stderr


def test_successor_binding_rejects_unhashed_receipt_lineage_entries(tmp_path: Path) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    unexpected = paths["receipt_dir"] / f"{TRIAL}.unexpected"
    unexpected.write_text("not authorized\n", encoding="utf-8")

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "receipt state is missing or ambiguous" in result.stderr


def test_successor_binding_rejects_same_step_candidate_for_different_checkpoint(
    tmp_path: Path,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    candidate = json.loads(paths["source_candidate"].read_text(encoding="utf-8"))
    candidate["recover"]["recover_checkpoint_manifest"]["__0_0.distcp"]["size"] += 1
    _write_json(paths["source_candidate"], candidate)
    digest_index = command.index("--expected-source-candidate-sha256") + 1
    command[digest_index] = _sha256(paths["source_candidate"])

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "does not exactly bind the current recovery checkpoint" in result.stderr


@pytest.mark.parametrize("persisted_commit", [3, 5])
def test_successor_binding_rejects_any_other_recover_stats_cursor(
    tmp_path: Path,
    persisted_commit: int,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    _write_json(
        paths["recover_info"] / "stats_logger_info.json",
        {"last_commit_step": persisted_commit},
    )
    digest_index = command.index("--expected-recover-info-content-sha256") + 1
    command[digest_index] = _content_manifest_sha256(list(paths["recover_info"].iterdir()))

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "stats logger recovery state is not coherent" in result.stderr


def test_successor_binding_requires_post_checkpoint_stats_commit_in_candidate(
    tmp_path: Path,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, successor_config=True)
    candidate = json.loads(paths["source_candidate"].read_text(encoding="utf-8"))
    candidate["stats_logger_last_commit_step"] = 5
    _write_json(paths["source_candidate"], candidate)
    digest_index = command.index("--expected-source-candidate-sha256") + 1
    command[digest_index] = _sha256(paths["source_candidate"])

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "lacks the exact post-checkpoint commit/update evidence" in result.stderr


def test_preupdate_resume_verifier_executes_candidate_free_step_zero_authorization(
    tmp_path: Path,
) -> None:
    command, _ = _build_authorized_fixture(tmp_path, preupdate=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "openreward_resume_checkpoint=PASS global_step=0 shards=80 " in result.stdout


def test_preupdate_binding_rejects_changed_current_config_and_any_advanced_step(
    tmp_path: Path,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, preupdate=True)
    paths["current_config"].write_text("changed after binding\n", encoding="utf-8")
    changed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert changed.returncode != 0
    assert "current resume config identity mismatch" in changed.stderr

    command, _ = _build_authorized_fixture(tmp_path / "advanced", preupdate=True)
    command[command.index("--expected-global-step") + 1] = "1"
    command[command.index("--expected-start-global-step") + 1] = "2"
    advanced = subprocess.run(command, check=False, capture_output=True, text=True)
    assert advanced.returncode != 0
    assert "expected steps/counts/digests are invalid" in advanced.stderr


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (
        ("missing_shard", "shard ordinals mismatch"),
        ("same_size_shard", "bounded-sample manifest digest mismatch"),
        ("wrong_step", "recover step info does not equal"),
        ("unexpected_source_candidate", "source receipt/candidate state is not absent"),
        ("unexpected_source_success", "source receipt/candidate state is not absent"),
    ),
)
def test_preupdate_resume_rejects_missing_corrupt_wrong_step_and_source_candidate(
    tmp_path: Path,
    corruption: str,
    expected_error: str,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path, preupdate=True)
    if corruption == "missing_shard":
        (paths["checkpoint"] / "__79_0.distcp").unlink()
    elif corruption == "same_size_shard":
        shard = paths["checkpoint"] / "__79_0.distcp"
        shard.write_bytes(b"X" * shard.stat().st_size)
    elif corruption == "wrong_step":
        _write_json(
            paths["recover_info"] / "step_info.json",
            {"epoch": 0, "epoch_step": 1, "global_step": 1, "steps_per_epoch": 846},
        )
    elif corruption == "unexpected_source_candidate":
        _write_json(paths["receipt_dir"] / f"{TRIAL_16}.success.candidate.json", {})
    else:
        _write_json(paths["receipt_dir"] / f"{TRIAL_16}.success.json", {})

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_step_zero_resume_requires_explicit_preupdate_mode(tmp_path: Path) -> None:
    command, _ = _build_authorized_fixture(tmp_path, preupdate=True)
    command.remove("--allow-preupdate-global-step-zero")

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "expected steps/counts/digests are invalid" in result.stderr


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (
        ("missing_shard", "shard ordinals mismatch"),
        ("zero_shard", "identity/owner/size mismatch"),
        ("same_size_shard", "bounded-sample manifest digest mismatch"),
        ("wrong_step", "recover step info does not equal"),
        ("extra_receipt", "receipt state is missing or ambiguous"),
    ),
)
def test_resume_verifier_rejects_corruption_and_ambiguous_state(
    tmp_path: Path,
    corruption: str,
    expected_error: str,
) -> None:
    command, paths = _build_authorized_fixture(tmp_path)
    if corruption == "missing_shard":
        (paths["checkpoint"] / "__159_0.distcp").unlink()
    elif corruption == "zero_shard":
        (paths["checkpoint"] / "__159_0.distcp").write_bytes(b"")
    elif corruption == "same_size_shard":
        shard = paths["checkpoint"] / "__159_0.distcp"
        shard.write_bytes(b"X" * shard.stat().st_size)
    elif corruption == "wrong_step":
        _write_json(
            paths["recover_info"] / "step_info.json",
            {"epoch": 0, "epoch_step": 4, "global_step": 4, "steps_per_epoch": 846},
        )
    else:
        (paths["receipt_dir"] / f"{TRIAL}.unexpected").write_text("stale\n")

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_runtime_semantic_verifier_executes_gap_free_dcp_and_progress_checks(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("resume_verifier", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.QWEN38_SUCCESSOR_ANCESTOR_SHA256 == ANCESTOR_SHA256

    class Value:
        pass

    checkpoint = tmp_path / "recover_checkpoint"
    checkpoint.mkdir()
    for ordinal in range(2):
        (checkpoint / f"__{ordinal}_0.distcp").write_bytes(b"0123456789")
    state_metadata = {"first": object(), "second": object()}
    storage_data = {}
    for ordinal, fqn in enumerate(state_metadata):
        index = Value()
        index.fqn = fqn
        storage = Value()
        storage.relative_path = f"__{ordinal}_0.distcp"
        storage.offset = 0
        storage.length = 10
        storage_data[index] = storage
    storage_meta = Value()
    storage_meta.checkpoint_id = str(checkpoint) + "_new"
    storage_meta.save_id = "00000000-0000-0000-0000-000000000000"
    storage_meta.load_id = None
    storage_meta.modules = []
    metadata = Value()
    metadata.state_dict_metadata = state_metadata
    metadata.storage_data = storage_data
    metadata.storage_meta = storage_meta
    metadata.version = "1.0.0"
    common = {
        "optimizer": {"param_state_sharding_type": "dp_reshardable"},
        "lr_scheduler": {"num_steps": 6},
    }
    dataloader = [
        {
            "_sampler_iter_state": {"samples_yielded": 208},
            "_sampler_iter_yielded": 26,
            "_num_yielded": 26,
            "_iterator_finished": False,
        }
    ]
    arguments = {
        "checkpoint": checkpoint,
        "metadata": metadata,
        "common": common,
        "dataloader": dataloader,
        "expected_state_keys": 2,
        "expected_storage_entries": 2,
        "expected_shards": 2,
        "expected_lr_num_steps": 6,
        "expected_samples_yielded": 208,
        "expected_iterator_yielded": 26,
    }

    module._verify_runtime_objects(**arguments)
    next(iter(storage_data.values())).offset = 1
    with pytest.raises(RuntimeError, match="not gap-free"):
        module._verify_runtime_objects(**arguments)


def test_preupdate_runtime_schema_requires_zero_moments_and_eight_groups() -> None:
    spec = importlib.util.spec_from_file_location("resume_verifier_moments", VERIFY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    pristine: dict[str, object] = {}
    for group in range(8):
        prefix = f"optimizer.distributed.dp_group_idx_{group}"
        pristine[f"{prefix}.optimizer/shard_0_1"] = object()
        pristine[f"{prefix}.per_bucket_numel/shard_0_1"] = object()
        pristine[f"{prefix}.per_bucket_numel_unpadded/shard_0_1"] = object()
        pristine[f"{prefix}.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.param"] = object()

    module._verify_optimizer_moment_metadata(pristine, 0)
    with_moments = dict(pristine)
    with_moments[
        "optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.exp_avg"
    ] = object()
    with pytest.raises(RuntimeError, match="optimizer-moment key count mismatch"):
        module._verify_optimizer_moment_metadata(with_moments, 0)

    missing_group = {key: value for key, value in pristine.items() if "dp_group_idx_7" not in key}
    with pytest.raises(RuntimeError, match="eight-group skeleton"):
        module._verify_optimizer_moment_metadata(missing_group, 0)

    unexpected_bookkeeping = dict(pristine)
    unexpected_bookkeeping["optimizer.distributed.dp_group_idx_0.extra_bookkeeping/shard_0_1"] = object()
    with pytest.raises(RuntimeError, match="eight-group skeleton"):
        module._verify_optimizer_moment_metadata(unexpected_bookkeeping, 0)


def _health_functions() -> str:
    source = COMMON.read_text(encoding="utf-8")
    begin = source.index("# BEGIN OPENREWARD_HEALTH_MONITOR_FUNCTIONS")
    end = source.index("# END OPENREWARD_HEALTH_MONITOR_FUNCTIONS")
    return source[begin:end]


@pytest.mark.parametrize(("failed_workers", "expected_status"), ((3, 0), (4, 1)))
def test_worker_aware_toolathlon_health_quorum_executes(
    tmp_path: Path,
    failed_workers: int,
    expected_status: int,
) -> None:
    header_log = tmp_path / "headers"
    url_log = tmp_path / "urls"
    harness = tmp_path / "health-harness.sh"
    harness.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
HEALTH_CONNECT_TIMEOUT_SECONDS=3
HEALTH_MAX_TIME_SECONDS=15
TOOLATHLON_WORKER_BASE_PORT=28100
TOOLATHLON_WORKERS=8
TOOLATHLON_WORKER_QUORUM=5
TOOLATHLON_PUBLIC_PROBES=4
TOOLATHLON_PUBLIC_QUORUM=1
OPENREWARD_PORT=8082
OPENREWARD_SWE_REBENCH_PORT=8084
JOB_ID_SAFE=test-job
service_pids=($$)
service_hosts=(node001)
HEADER_LOG=$1
URL_LOG=$2
FAILED_WORKERS=$3
curl() {
    local header= url= worker
    while (($#)); do
        case $1 in
            --header)
                header=$2
                shift 2
                ;;
            http://*)
                url=$1
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    printf '%s\n' "${url}" >>"${URL_LOG}"
    if [[ -n "${header}" ]]; then
        printf '%s\n' "${header}" >>"${HEADER_LOG}"
    fi
    sleep 0.2
    if [[ "${url}" =~ :2810([0-7])/health$ ]]; then
        worker=${BASH_REMATCH[1]}
        if ((worker < FAILED_WORKERS)); then
            printf '000\t15.000'
            return 28
        fi
    fi
    [[ "${url}" == http://node001:8082/health || \
       "${url}" == http://node001:8084/health || \
       "${url}" =~ ^http://node001:2810[0-7]/health$ ]]
    printf '200\t0.010'
}
"""
        + _health_functions()
        + "\nservice_pools_healthy 7\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(harness), str(header_log), str(url_log), str(failed_workers)],
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started

    assert result.returncode == expected_status, result.stderr
    headers = header_log.read_text().splitlines()
    assert len(headers) == 4
    assert len(set(headers)) == 4
    assert all(value.startswith("X-Session-ID: openreward-health-") for value in headers)
    urls = url_log.read_text().splitlines()
    assert len(urls) == 13
    assert len({url for url in urls if ":2810" in url}) == 8
    assert elapsed < 1.2, "health probes did not execute concurrently"
    assert "host=node001 port=28100 url=http://node001:28100/health" in result.stderr
    assert "rc=28 http=000 elapsed=15.000" in result.stderr
    if expected_status:
        assert "rc=quorum" in result.stderr
        assert "successes=4 failures=4 required=5 workers=8" in result.stderr


def test_resume_wrapper_and_common_bind_explicit_recovery_without_yaml_change() -> None:
    wrapper = WRAPPER.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    subprocess.run(["bash", "-n", str(COMMON)], check=True)
    assert "#PBS -N rasp-h200-32q38-rs5-v16" in wrapper
    assert "EXPERIMENT=abci-or-ta-swe32-q38-v16" in wrapper
    assert "TRIAL=abci-ta-swe32-rec-q38d-c24-v16-t0" in wrapper
    assert "--expected-global-step 5" in wrapper
    assert "--expected-shards 160" in wrapper
    assert "--expected-total-bytes 383216730907" in wrapper
    assert "--expected-checkpoint-layout-sha256 6699027815f1628" in wrapper
    assert "--expected-checkpoint-sample-manifest-sha256 23aea8bf8cc9329" in wrapper
    assert "--expected-fixed-content-sha256 ff7dccef8552800b" in wrapper
    assert "--expected-recover-info-content-sha256 86e7b2726bcbc863" in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=5" in wrapper
    assert "resume-gs5-v8r16.success.json" in wrapper
    assert "refusing stale checkpoint; use a fresh trial" not in wrapper
    assert "controller_args+=(recover.mode=on)" in common
    assert 'case "${OPENREWARD_MODEL_FAMILY:-}" in' in common
    assert '"OPENREWARD_MODEL_FAMILY=${OPENREWARD_MODEL_FAMILY}"' in common
    assert "OPENREWARD_MODEL_FAMILY PLATOON_MODEL_SNAPSHOT" in common
    assert '"${RESUME_CHECKPOINT_VERIFY}" runtime' in common
    assert '"${PLATOON_TRAINING_SIF}"' in common
    assert "http://${host}:${OPENREWARD_PORT}/health" in common
    assert "http://${host}:${OPENREWARD_SWE_REBENCH_PORT}/health" in common


def test_preupdate_nooptim_wrapper_binds_exact_step_zero_checkpoint_and_heartbeat() -> None:
    wrapper = NOOPTIM_WRAPPER_16.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(NOOPTIM_WRAPPER_16)], check=True)
    assert NOOPTIM_WRAPPER_16.stat().st_mode & 0o111 == 0o111
    assert "#PBS -N rasp-h200-16q38-rs0n-v20" in wrapper
    assert "#PBS -l walltime=72:00:00" in wrapper
    assert "EXPERIMENT=abci-or-ta-swe16-q38-v18" in wrapper
    assert "TRIAL=abci-ta-swe16-nr-q38d-c12-v18-t0" in wrapper
    assert "--expected-global-step 0" in wrapper
    assert "--expected-shards 80" in wrapper
    assert "--expected-total-bytes 164220069555" in wrapper
    assert "--allow-preupdate-global-step-zero" in wrapper
    assert "PREUPDATE_BINDING=qwen38-16-v18-job2155389" in wrapper
    assert "PREUPDATE_LOAD_POLICY=qwen38-16-v20-gs0-no-load-optim" in wrapper
    assert '--preupdate-binding "${PREUPDATE_BINDING}"' in wrapper
    assert '--preupdate-load-policy "${PREUPDATE_LOAD_POLICY}"' in wrapper
    assert "--expected-source-config-sha256" in wrapper
    assert "--expected-start-global-step 1" in wrapper
    assert '--source-log "${SOURCE_LOG}"' in wrapper
    assert "--source-candidate" not in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_ALLOW_PREUPDATE_RESUME=1" in wrapper
    assert "export OPENREWARD_PREUPDATE_RESUME_BINDING=${PREUPDATE_BINDING}" in wrapper
    assert "export OPENREWARD_PREUPDATE_LOAD_POLICY=${PREUPDATE_LOAD_POLICY}" in wrapper
    assert "export OPENREWARD_PREUPDATE_NO_LOAD_OPTIM=1" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=0" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_LR_NUM_STEPS=0" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_OPTIMIZER_MOMENT_KEYS=0" in wrapper
    assert "resume-gs0-nooptim-v8r20.success.json" in wrapper
    assert "OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" in wrapper
    assert "preupdate resume requires exact recovered global step 0" in common
    assert "preupdate no-load-optimizer resume requires start step 1 and no optimizer moments" in common
    assert "preupdate optimizer-load policy is forbidden for normal resume" in common
    assert "controller_args+=(recover.no_load_optim=true recover.no_save_optim=false)" in common
    assert '"${RESUME_CHECKPOINT_VERIFY}" composition' in common
