from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import stat
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "pbs-scripts" / "openreward-node-local-init.py"
COMMON_PATH = REPO_ROOT / "pbs-scripts" / "openreward-ta-swe-prealloc-common.sh"


def _load_helper():
    spec = importlib.util.spec_from_file_location("openreward_node_local_init_test", HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    job_id = "12345.pbs1"
    host = socket.gethostname().split(".", 1)[0]
    localdir = tmp_path / "local" / job_id
    localdir.mkdir(parents=True, mode=0o700)
    root = localdir / "platoon" / job_id
    receipts = tmp_path / "shared" / "receipts"
    receipts.mkdir(parents=True, mode=0o700)
    node_map = tmp_path / "shared" / "ordered-nodes.tsv"
    node_map.write_text(f"{host}\t0\n")
    monkeypatch.setenv("PBS_JOBID", job_id)
    monkeypatch.setenv("PBS_LOCALDIR", str(localdir))
    return job_id, host, localdir, root, receipts, node_map


def _init_arguments(
    *,
    job_id: str,
    host: str,
    root: Path,
    receipts: Path,
    training_sif: Path,
    digest: str,
    stage: int,
) -> list[str]:
    return [
        "init",
        "--expected-host",
        host,
        "--task-index",
        "0",
        "--expected-root",
        str(root),
        "--pbs-job-id",
        job_id,
        "--generation",
        "a" * 32,
        "--receipt",
        str(receipts / f"{host}.json"),
        "--stage-training-sif",
        str(stage),
        "--training-sif",
        str(training_sif),
        "--training-sif-sha256",
        digest,
    ]


def _verify_arguments(
    *,
    job_id: str,
    root: Path,
    receipts: Path,
    node_map: Path,
    digest: str,
    stage: int,
) -> list[str]:
    return [
        "verify",
        "--node-map",
        str(node_map),
        "--receipt-dir",
        str(receipts),
        "--expected-nodes",
        "1",
        "--expected-root",
        str(root),
        "--pbs-job-id",
        job_id,
        "--generation",
        "a" * 32,
        "--stage-training-sif",
        str(stage),
        "--training-sif-sha256",
        digest,
    ]


def test_initializer_creates_exact_worker_bind_sources_and_durable_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, node_map = _fixture_paths(tmp_path, monkeypatch)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"unused when staging is disabled")
    digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()

    assert (
        helper.main(
            _init_arguments(
                job_id=job_id,
                host=host,
                root=root,
                receipts=receipts,
                training_sif=training_sif,
                digest=digest,
                stage=0,
            )
        )
        == 0
    )
    assert stat.S_IMODE(root.lstat().st_mode) == 0o700
    assert stat.S_IMODE((root / "worker-home").lstat().st_mode) == 0o700
    assert stat.S_IMODE((root / "worker-tmp").lstat().st_mode) == 0o1777
    receipt = receipts / f"{host}.json"
    assert stat.S_IMODE(receipt.lstat().st_mode) == 0o600
    value = json.loads(receipt.read_text())
    assert value["node_local_root"] == str(root)
    assert value["worker_tmp"] == str(root / "worker-tmp")
    assert value["training_sif_staged"] is False
    assert (
        helper.main(
            _verify_arguments(
                job_id=job_id,
                root=root,
                receipts=receipts,
                node_map=node_map,
                digest=digest,
                stage=0,
            )
        )
        == 0
    )


@pytest.mark.parametrize("mismatch", ("host", "job", "path"))
def test_initializer_rejects_wrong_remote_identity_or_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, _node_map = _fixture_paths(tmp_path, monkeypatch)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"unused")
    digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()
    requested_job = "99999.pbs1" if mismatch == "job" else job_id
    requested_host = f"not-{host}" if mismatch == "host" else host
    requested_root = tmp_path / "wrong-root" if mismatch == "path" else root

    status = helper.main(
        _init_arguments(
            job_id=requested_job,
            host=requested_host,
            root=requested_root,
            receipts=receipts,
            training_sif=training_sif,
            digest=digest,
            stage=0,
        )
    )
    assert status == 2
    assert not (receipts / f"{requested_host}.json").exists()


def test_initializer_rejects_symlink_worker_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, _node_map = _fixture_paths(tmp_path, monkeypatch)
    root.parent.mkdir(mode=0o700)
    target = tmp_path / "redirected-root"
    target.mkdir()
    root.symlink_to(target, target_is_directory=True)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"unused")
    digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()

    assert (
        helper.main(
            _init_arguments(
                job_id=job_id,
                host=host,
                root=root,
                receipts=receipts,
                training_sif=training_sif,
                digest=digest,
                stage=0,
            )
        )
        == 2
    )
    assert not (receipts / f"{host}.json").exists()


def test_runtime_attestation_requires_exact_executable_hash_version_owner_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, node_map = _fixture_paths(tmp_path, monkeypatch)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"unused")
    training_digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()
    runtime = tmp_path / "singularity"
    runtime.write_text("#!/bin/sh\nprintf '%s\\n' 4.4.1-test\n")
    runtime.chmod(0o555)
    digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
    monkeypatch.setattr(helper, "EXPECTED_CONTAINER_RUNTIME_PATH", runtime)
    monkeypatch.setattr(helper, "EXPECTED_CONTAINER_RUNTIME_SHA256", digest)
    monkeypatch.setattr(helper, "EXPECTED_CONTAINER_RUNTIME_VERSION", "4.4.1-test")
    monkeypatch.setattr(helper, "TRUSTED_CONTAINER_RUNTIME_UIDS", {os.getuid()})
    arguments = Namespace(
        container_runtime=str(runtime),
        container_runtime_sha256=digest,
        container_runtime_version="4.4.1-test",
    )

    assert helper._attest_container_runtime(arguments) == {
        "container_runtime_path": str(runtime),
        "container_runtime_sha256": digest,
        "container_runtime_version": "4.4.1-test",
        "container_runtime_uid": os.getuid(),
        "container_runtime_mode": 0o555,
    }
    runtime_arguments = [
        "--attest-container-runtime",
        "1",
        "--container-runtime",
        str(runtime),
        "--container-runtime-sha256",
        digest,
        "--container-runtime-version",
        "4.4.1-test",
    ]
    init_arguments = _init_arguments(
        job_id=job_id,
        host=host,
        root=root,
        receipts=receipts,
        training_sif=training_sif,
        digest=training_digest,
        stage=0,
    )
    assert helper.main([*init_arguments, *runtime_arguments]) == 0
    receipt = receipts / f"{host}.json"
    receipt_value = json.loads(receipt.read_text())
    assert {key: receipt_value[key] for key in helper.RUNTIME_RECEIPT_KEYS} == {
        "container_runtime_path": str(runtime),
        "container_runtime_sha256": digest,
        "container_runtime_version": "4.4.1-test",
        "container_runtime_uid": os.getuid(),
        "container_runtime_mode": 0o555,
    }
    verify_arguments = _verify_arguments(
        job_id=job_id,
        root=root,
        receipts=receipts,
        node_map=node_map,
        digest=training_digest,
        stage=0,
    )
    assert helper.main([*verify_arguments, *runtime_arguments]) == 0
    receipt_value["container_runtime_sha256"] = "0" * 64
    receipt.write_text(json.dumps(receipt_value))
    assert helper.main([*verify_arguments, *runtime_arguments]) == 2

    runtime.chmod(0o575)
    with pytest.raises(helper.InitError, match="ownership or mode"):
        helper._attest_container_runtime(arguments)


def test_verifier_rejects_any_extra_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, node_map = _fixture_paths(tmp_path, monkeypatch)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(b"unused")
    digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()
    assert (
        helper.main(
            _init_arguments(
                job_id=job_id,
                host=host,
                root=root,
                receipts=receipts,
                training_sif=training_sif,
                digest=digest,
                stage=0,
            )
        )
        == 0
    )
    extra = receipts / "unexpected.json"
    extra.write_text("{}\n")
    extra.chmod(0o600)
    assert (
        helper.main(
            _verify_arguments(
                job_id=job_id,
                root=root,
                receipts=receipts,
                node_map=node_map,
                digest=digest,
                stage=0,
            )
        )
        == 2
    )


def test_stage_one_puts_digest_verified_training_sif_in_exact_worker_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = _load_helper()
    job_id, host, _localdir, root, receipts, node_map = _fixture_paths(tmp_path, monkeypatch)
    training_sif = tmp_path / "training.sif"
    training_sif.write_bytes(os.urandom(1024 * 1024))
    digest = hashlib.sha256(training_sif.read_bytes()).hexdigest()
    assert (
        helper.main(
            _init_arguments(
                job_id=job_id,
                host=host,
                root=root,
                receipts=receipts,
                training_sif=training_sif,
                digest=digest,
                stage=1,
            )
        )
        == 0
    )

    local_sif = root / "images" / "platoon-training.sif"
    assert hashlib.sha256(local_sif.read_bytes()).hexdigest() == digest
    assert stat.S_IMODE(local_sif.lstat().st_mode) == 0o400
    assert (
        helper.main(
            _verify_arguments(
                job_id=job_id,
                root=root,
                receipts=receipts,
                node_map=node_map,
                digest=digest,
                stage=1,
            )
        )
        == 0
    )


def test_common_initializes_every_ordered_node_before_builds_services_and_workers() -> None:
    common = COMMON_PATH.read_text()
    init_start = common.index(
        'echo "[pbs-openreward] initializing worker bind sources on all ${EXPECTED_NNODES} ordered nodes"'
    )
    init_end = common.index("\nurls_for_port()", init_start)
    body = common[init_start:init_end]
    assert 'done <"${NODE_MAP}"' in body
    assert '"${PBSDSH}" -n "${vnode_index}" --' in body
    assert '"${NODE_LOCAL_INIT_PYTHON}" "${NODE_LOCAL_INIT_SNAPSHOT}" init' in body
    assert '--stage-training-sif "${PLATOON_STAGE_TRAINING_SIF}"' in body
    assert '--attest-container-runtime "${ATTEST_CONTAINER_RUNTIME}"' in body
    assert '--container-runtime "${RUNTIME}"' in body
    assert '--container-runtime-sha256 "${EXPECTED_CONTAINER_RUNTIME_SHA256}"' in body
    assert '--container-runtime-version "${EXPECTED_CONTAINER_RUNTIME_VERSION}"' in body
    assert '"${CONTROLLER_PYTHON_BASE}" -I "${NODE_LOCAL_INIT_SNAPSHOT}" verify' in body
    assert '"${node_init_count}" -eq "${EXPECTED_NNODES}"' in body
    assert common.index("NODE_LOCAL_INIT_SNAPSHOT=") < init_start
    assert common.index("chmod 0500") < init_start
    assert init_end < common.index("resolving immutable SIF worker environment")
    assert init_end < common.index("starting one Toolathlon and one SWE service")
    assert init_end < common.index("PLATOON_AREAL_PBS_CONTAINER_BINDS=")
    for secret in (
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        "RUBRIC_DEFAULT_LLM",
        "HF_TOKEN",
    ):
        assert f"-u {secret}" in body


def test_common_and_service_helper_pin_v9_runtime_without_changing_v8_fallback() -> None:
    common = COMMON_PATH.read_text()
    services = (COMMON_PATH.parent / "openreward-node-services.sh").read_text()
    runtime_sha = "085a757f172dcabda66b44d086913314d6aa574055a8fea90523be82b770cec6"

    for source in (common, services):
        assert "EXPECTED_CONTAINER_RUNTIME_PATH=/usr/bin/singularity" in source
        assert f"EXPECTED_CONTAINER_RUNTIME_SHA256={runtime_sha}" in source
        assert "EXPECTED_CONTAINER_RUNTIME_VERSION=4.4.1-1.el9" in source
        assert 'if [[ -n "${TOOLATHLON_PORTABILITY_PATCH_SHA256}" ]]; then' in source
        assert '( "${runtime_uid}" == 0 || "${runtime_uid}" == 65534 )' in source
        assert "$((8#${runtime_mode} & 8#022)) -eq 0" in source
        assert '"$(sha256sum "${RUNTIME}" | awk \'{print $1}\')"' in source
        assert '"$("${RUNTIME}" version)" == "${EXPECTED_CONTAINER_RUNTIME_VERSION}"' in source
    assert "ATTEST_CONTAINER_RUNTIME=0" in common
