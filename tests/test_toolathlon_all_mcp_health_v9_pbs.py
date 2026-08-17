"""Structural checks for the one-node v9 all-MCP PBS acceptance wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
WRAPPER = PBS_ROOT / "openreward-toolathlon-all-mcp-health-v9.pbs"
BUILD_WRAPPER = PBS_ROOT / "openreward-toolathlon-pbs-portable-build.pbs"
DRIVER = PBS_ROOT / "toolathlon_all_mcp_health.py"


def test_all_mcp_wrapper_uses_one_node_exact_future_release_and_no_submission() -> None:
    source = WRAPPER.read_text()

    assert WRAPPER.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    assert "#PBS -l select=1:mpiprocs=8" in source
    assert "#PBS -l walltime=02:00:00" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r23-e11d720-057108f" in source
    assert "RELEASE_REF=refs/tags/abci-v8r23-e11d720-057108f" in source
    assert "readonly LOCAL_ROOT=${PBS_LOCALDIR:?PBS_LOCALDIR is required}/platoon/${PBS_JOBID}" in source
    assert "/platoon/toolathlon-all-mcp-v9/${PBS_JOBID}" not in source
    assert "openreward-v9-abci.env" in source
    assert "export OPENREWARD_MODEL_FAMILY=qwen38" in source
    assert source.index("export OPENREWARD_MODEL_FAMILY=qwen38") < source.index(
        'source "${ABCI_ENV}"'
    )
    assert '"OPENREWARD_MODEL_FAMILY=${OPENREWARD_MODEL_FAMILY}"' in source
    assert "verify_openreward_v9_release.py" in source
    assert '"${RELEASE_VERIFY}" --release-root "${REPO_ROOT}"' in source
    assert "qsub" not in source.lower()


def test_all_mcp_wrapper_runs_exact_production_service_driver_and_normal_cleanup() -> None:
    source = WRAPPER.read_text()
    launch = '"${PBSDSH}" -n 0 -- /usr/bin/env'
    driver = "/opt/venv/bin/python /opt/openreward-health/toolathlon_all_mcp_health.py"
    request = 'request_service_shutdown || die "natural node-service shutdown request publication failed"'
    teardown = 'teardown_service || die "normal node-service cleanup attestation failed"'
    candidate = '"${RESULT_PUBLISHER}" candidate'

    for marker in (launch, driver, request, teardown, candidate):
        assert marker in source
    assert source.index(launch) < source.index(driver) < source.index(request)
    assert source.index(request) < source.index(teardown) < source.index(candidate)
    assert "PLATOON_NODE_SERVICES_CLEANUP_RECEIPT" in source
    assert "PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST" in source
    assert "export PLATOON_NODE_LOCAL_ROOT=${LOCAL_ROOT}" in source
    assert "export PLATOON_NODE_LOCAL_ROOT=${LOCAL_ROOT}/node-services" not in source
    assert '"${PLATOON_NODE_LOCAL_ROOT:-}" == "${LOCAL_ROOT}"' in source
    assert '"${SINGULARITY_CACHEDIR:-}" == "${LOCAL_ROOT}/singularity-cache"' in source
    assert '"${SINGULARITY_TMPDIR:-}" == "${LOCAL_ROOT}/singularity-tmp"' in source
    assert '"${APPTAINER_CACHEDIR:-}" == "${SINGULARITY_CACHEDIR}"' in source
    assert '"${APPTAINER_TMPDIR:-}" == "${SINGULARITY_TMPDIR}"' in source
    assert '"${SWE_APPTAINER_SESSION_ROOT:-}" == "${LOCAL_ROOT}/swe-sessions"' in source
    assert '"${SWE_SINGULARITY_SESSION_ROOT:-}" == "${SWE_APPTAINER_SESSION_ROOT}"' in source
    assert source.index("v9 overlay node-local paths") < source.index("export PLATOON_NODE_LOCAL_ROOT=${LOCAL_ROOT}")
    assert '--primary-status "${expected_primary_status}"' in source
    assert "assert-tcp-closed" in source
    assert "assert-empty-swe" in source
    assert "OPENREWARD_TOOLATHLON_BUILD_RECEIPT" in source
    assert "health-report.json" in source
    assert "RESULT.candidate" in source
    assert "${LOCAL_ROOT}/driver-tmp:/mnt/driver-tmp" in source
    assert "TMPDIR=/mnt/driver-tmp" in source
    assert "UV_CACHE_DIR=/mnt/driver-home/uv-cache" in source
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv/python" in source
    assert "UV_PYTHON_PREFERENCE=only-managed" in source
    assert "UV_PYTHON_DOWNLOADS=never" in source
    assert '[[ "${OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH:-}" == /opt/ms-playwright ]]' in source
    assert '"OPENREWARD_TOOLATHLON_PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright"' in source
    assert "PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright" in source
    assert 'OPENREWARD_SESSION_URL="http://${HOST}:${OPENREWARD_PORT}"' in source
    assert 'OPENREWARD_API_URL="http://${HOST}:${OPENREWARD_PORT}"' in source
    assert '"PLATOON_CONTAINER_RUNTIME=${RUNTIME}"' in source
    assert "--catalog /app/tool_schemas.json" in source
    assert "--tasks-root /app/tasks" in source
    assert "http://127.0.0.1:${OPENREWARD_PORT}" not in source


def test_all_mcp_wrapper_never_self_publishes_final_result_or_claims_pbs_exit() -> None:
    source = WRAPPER.read_text()

    assert "RESULT.candidate" in source
    assert "toolathlon_all_mcp_health_candidate=PASS" in source
    assert "publish_toolathlon_all_mcp_health_result.py" in source
    assert '"${RESULT_PUBLISHER}" finalize' not in source
    assert "Exit_status 0" in source
    assert "post-job finalizer" in source
    assert "node_services_pbsdsh_smoke=PASS" not in source


def test_all_mcp_wrapper_scrubs_all_controller_credentials() -> None:
    source = WRAPPER.read_text()

    assert 'unset "${secret_name}"' in source
    for name in (
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        "RUBRIC_DEFAULT_LLM",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert name in source
        assert f"-u {name}" in source


def test_all_mcp_driver_has_exact_25_server_cover_and_email_regression() -> None:
    source = DRIVER.read_text()

    assert "EXPECTED_SERVERS = (" in source
    assert '"snowflake",' in source
    assert "if len(sessions) != 7:" in source
    assert '"expected_server_count": len(EXPECTED_SERVERS)' in source
    assert '"failure_count": 0' in source
    assert '"emails.check_connection"' in source
    assert '"emails.send_email"' in source
    assert '"emails.get_emails"' in source
    assert '"playwright_with_chunk": ("playwright_with_chunk.browser_tab_list", {})' in source
    assert "Email sent successfully" in source
    assert "subject not in readback" in source
    assert os.access(DRIVER, os.R_OK)


def test_portable_build_wrapper_is_commit_pinned_atomic_and_never_submits() -> None:
    source = BUILD_WRAPPER.read_text()

    assert BUILD_WRAPPER.stat().st_mode & 0o111 == 0o111
    subprocess.run(["bash", "-n", str(BUILD_WRAPPER)], check=True)
    assert "#PBS -l select=1:mpiprocs=8" in source
    assert "PLATOON_TOOLATHLON_BUILD_INPUT_ROOT" in source
    assert "PLATOON_TOOLATHLON_BUILD_INPUT_COMMIT" in source
    assert '"${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$' in source
    assert 'rev-parse --verify HEAD)" == "${EXPECTED_COMMIT}"' in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "build_toolathlon_pbs_portable_sif.sh" in source
    assert "publish_toolathlon_pbs_image.py" in source
    assert "publish_toolathlon_pbs_build_receipt.py" in source
    assert "openreward-toolathlon-18e62c0d-pbs-${patch_identity:0:12}.sif" in source
    assert 'OUTPUT_DIR}/build-receipts/$(basename "${OUTPUT_SIF}").json' in source
    assert '"$(stat -c %a "${OUTPUT_SIF}")" == 550' in source
    assert '"$(stat -c %a "${BUILD_RECEIPT}")" == 400' in source
    assert "refusing stale derived image or build receipt" in source
    assert "qsub" not in source.lower()
