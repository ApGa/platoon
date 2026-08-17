from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
VERIFY = PBS_ROOT / "verify_openreward_resume_checkpoint.py"
COMMON = PBS_ROOT / "openreward-ta-swe-prealloc-common.sh"
WRAPPER = PBS_ROOT / "openreward-ta-swe-32node-recursive-h200-v9-abci.pbs"
WRAPPER_16 = PBS_ROOT / "openreward-ta-swe-16node-nonrecursive-h200-v9-abci.pbs"
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
        digest.update(
            path.name.encode()
            + b"\0"
            + str(size).encode()
            + b"\0"
            + sample.hexdigest().encode()
            + b"\n"
        )
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _build_authorized_fixture(
    tmp_path: Path, *, preupdate: bool = False
) -> tuple[list[str], dict[str, Path]]:
    experiment = EXPERIMENT_16 if preupdate else EXPERIMENT
    trial = TRIAL_16 if preupdate else TRIAL
    config_relative = CONFIG_RELATIVE_16 if preupdate else CONFIG_RELATIVE
    shard_count = 80 if preupdate else 160
    global_step = 0 if preupdate else 5
    source_job_id = "2155389.pbs1" if preupdate else "2154492.pbs1"
    checkpoint_root = tmp_path / "checkpoints" / experiment / trial
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
        recover_info / "stats_logger_info.json", {"last_commit_step": global_step - 1}
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
        f"experiment_name: {experiment}\n"
        f"trial_name: {trial}\n"
        "recover:\n  mode: auto\n"
    ).encode()
    source_config.write_bytes(config_bytes)
    current_config = tmp_path / "current-config.yaml"
    current_config.write_bytes(config_bytes)
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
    source_candidate = receipt_dir / f"{trial}.success.candidate.json"
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
                    (
                        "[pbs-openreward] starting host controller "
                        f"experiment={experiment} trial={trial} nodes=16"
                    ),
                    (
                        "Recover INFO: Saved recover checkpoint to "
                        f"{checkpoint_root}/default/recover_checkpoint"
                    ),
                    (
                        "RLTrainer ERROR: Training failed with exception: "
                        "optimizer_batch_datums must be an integer >= 1"
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
    else:
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
                "global_step": 0,
                "recover": {"checkpoint_root": str(checkpoint_root), "global_step": 0},
                "wandb": {"run_id": f"{experiment}_{trial}_train"},
            },
        )
    target_receipt = receipt_dir / (
        f"{trial}.resume-gs0-v8r17.success.json"
        if preupdate
        else f"{trial}.resume-gs5-v8r16.success.json"
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
        _sha256(source_config),
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
    if preupdate:
        command[command.index("--source-pbs-job-id"):command.index("--source-pbs-job-id")] = [
            "--source-log",
            str(source_log),
            "--expected-source-log-sha256",
            _sha256(source_log),
            "--allow-preupdate-global-step-zero",
            "--preupdate-binding",
            "qwen38-16-v18-job2155389",
        ]
    else:
        command[command.index("--source-pbs-job-id"):command.index("--source-pbs-job-id")] = [
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
    }


def test_resume_verifier_executes_complete_authorization(tmp_path: Path) -> None:
    command, _ = _build_authorized_fixture(tmp_path)
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert (
        "openreward_resume_checkpoint=PASS global_step=5 shards=160 "
        in result.stdout
    )


def test_preupdate_resume_verifier_executes_candidate_free_step_zero_authorization(
    tmp_path: Path,
) -> None:
    command, _ = _build_authorized_fixture(tmp_path, preupdate=True)
    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "openreward_resume_checkpoint=PASS global_step=0 shards=80 " in result.stdout


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
    assert '"${RESUME_CHECKPOINT_VERIFY}" runtime' in common
    assert '"${PLATOON_TRAINING_SIF}"' in common
    assert "http://${host}:${OPENREWARD_PORT}/health" in common
    assert "http://${host}:${OPENREWARD_SWE_REBENCH_PORT}/health" in common


def test_preupdate_resume_wrapper_binds_exact_step_zero_checkpoint_and_heartbeat() -> None:
    wrapper = WRAPPER_16.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")

    subprocess.run(["bash", "-n", str(WRAPPER_16)], check=True)
    assert "#PBS -N rasp-h200-16q38-rs0-v18" in wrapper
    assert "#PBS -l walltime=72:00:00" in wrapper
    assert "EXPERIMENT=abci-or-ta-swe16-q38-v18" in wrapper
    assert "TRIAL=abci-ta-swe16-nr-q38d-c12-v18-t0" in wrapper
    assert "--expected-global-step 0" in wrapper
    assert "--expected-shards 80" in wrapper
    assert "--expected-total-bytes 164220069555" in wrapper
    assert "--allow-preupdate-global-step-zero" in wrapper
    assert "--preupdate-binding qwen38-16-v18-job2155389" in wrapper
    assert "--source-log \"${SOURCE_LOG}\"" in wrapper
    assert "--source-candidate" not in wrapper
    assert "export OPENREWARD_REQUIRE_EXISTING_RECOVERY=1" in wrapper
    assert "export OPENREWARD_ALLOW_PREUPDATE_RESUME=1" in wrapper
    assert "export OPENREWARD_PREUPDATE_RESUME_BINDING=qwen38-16-v18-job2155389" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_GLOBAL_STEP=0" in wrapper
    assert "export OPENREWARD_EXPECTED_RESUME_LR_NUM_STEPS=0" in wrapper
    assert "resume-gs0-v8r17.success.json" in wrapper
    assert "OPENREWARD_EXPECTED_TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600" in wrapper
    assert "preupdate resume requires exact recovered global step 0" in common
    assert "normal resume requires LR scheduler step equal recovered global step plus one" in common
