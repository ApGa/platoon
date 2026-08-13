"""Cross-repository provenance and compatibility checks for the v8 release."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
EXTERNAL = REPO_ROOT / "external"

EXPECTED_REVISIONS = {
    "platoon_upstream": "376d1b9885ea40464766fb3fea74815c6a698fbf",
    "areal": "d99124ec15102ca2fcd4960cc8beaef3950c2672",
    "openhands": "b1acf52ee362884f092b9eaf5e701a1bb5d1b335",
    "swe_rebench": "538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7",
    "toolathlon": "18e62c0d0419da88209af911528f2f8a0b44a5ed",
}
EXPECTED_UPSTREAM_REFRESH = {
    "fetched_at_utc": "2026-08-13T07:54:55Z",
    "platoon": {
        "remote": "https://github.com/ApGa/platoon.git",
        "ref": "refs/heads/apga/update-areal-with-registry",
        "sha": EXPECTED_REVISIONS["platoon_upstream"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "openhands": {
        "remote": "https://github.com/apga/openhands-agent-sdk.git",
        "ref": "refs/heads/apga/ptc",
        "sha": EXPECTED_REVISIONS["openhands"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "toolathlon": {
        "remote": "https://github.com/ApGa/toolathlon-gym.git",
        "ref": "refs/heads/apga/session-db-shim",
        "sha": EXPECTED_REVISIONS["toolathlon"],
        "release_ahead": 0,
        "release_behind": 0,
    },
    "swe_rebench": {
        "remote": "https://github.com/ApGa/SWE-rebench-V2-OpenRewardEnv.git",
        "ref": "refs/heads/main",
        "sha": "25b14c06b9236c075a4ede25bff6979e5783bb09",
        "release_ahead": 1,
        "release_behind": 0,
    },
}


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _require_release_gate() -> None:
    if os.environ.get("PLATOON_V8_RELEASE_GATE") != "1":
        pytest.skip("materialized external checkout validation is release-gate only")
    if REPO_ROOT != Path("/home/aci18912it/rasp/releases/platoon-v8-376d1b9-18e62c0"):
        pytest.fail("PLATOON_V8_RELEASE_GATE requires the canonical release root")


def test_v8_manifest_records_exact_component_and_image_revisions() -> None:
    manifest = json.loads((PBS_ROOT / "openreward-v8-release.json").read_text())
    assert manifest["components"] == EXPECTED_REVISIONS
    assert manifest["upstream_refresh"] == EXPECTED_UPSTREAM_REFRESH
    assert manifest["toolathlon_image"] == {
        "path": "/groups/gai51740/aci18912it/rasp/images/openreward-toolathlon-18e62c0d-c1e60b36.sif",
        "sha256": "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267",
        "oci_index_digest": "sha256:494af57e8a0f158d668982bf985b5937e73ba58eb468715f884de03550bd677c",
        "oci_amd64_digest": "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd",
    }
    assert manifest["swe_runtime"] == {
        "swe_revision": EXPECTED_REVISIONS["swe_rebench"],
        "requirements_sha256": "9d953d984c4d9095054d666492b51674797572b44f7bb02245a98611e38b6a7a",
        "lock_path": "containers/swe-rebench-538b5c5-py311.lock.txt",
        "lock_sha256": "dd33fc4a252140a3976c2b989c7769405781b863a21fdbb7a77ad5adc52de836",
        "python_real_executable": (
            "/home/aci18912it/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11"
        ),
        "python_executable_sha256": "c0b74b270ea823c760917e64fbde1945e67ba739d97fa716a48a5b69e6505d62",
        "python_library_sha256": "46da5d861633bd1627474c0d870592d8ec7817bef573ddc89b2128ed21721857",
        "python_build_sha256": "aae4c341cd4a142780552b76d2f7e41c6e26b9317a414ac7c6252fa7848146e2",
        "python_version": "3.11.15",
        "uv_executable": "/home/aci18912it/.local/bin/uv",
        "uv_executable_sha256": "92face6b1f0462ad911857957bd168cd4ae45515e2a2cb3fcc3ecbda3d4d82b1",
        "uv_version": "0.12.1",
        "package_count": 62,
        "freeze_sha256": "96c72a454e95bb0da0df14861d23909ff00d9345d301c9277300948f67cf1c4c",
        "package_pins": {
            "openreward": "0.1.137",
            "pyarrow": "25.0.1",
            "pydantic": "2.13.4",
            "pyright": "1.1.411",
        },
    }


def test_v8_preserves_latest_upstream_toolathlon_image_pin() -> None:
    expected = "apga+toolathlon-gym+18e62c0d041.sqsh"
    old = "apga+openreward-toolathlon-gym+43c1179fd7b8.sqsh"
    paths = (
        REPO_ROOT / "slurm-scripts/openreward-toolathlon-prealloc-base.sh",
        REPO_ROOT / "slurm-scripts/openreward-toolathlon-swe-prealloc-16node-ptc-task-tracker-ta20-curriculum.sh",
        REPO_ROOT / "slurm-scripts/openreward-toolathlon-swe-prealloc-32node-ptc-recursive-ta20-curriculum.sh",
    )
    for path in paths:
        source = path.read_text()
        assert expected in source
        assert old not in source


def test_v8_swe_host_runtime_is_hash_locked_and_launch_verified() -> None:
    manifest = json.loads((PBS_ROOT / "openreward-v8-release.json").read_text())
    runtime = manifest["swe_runtime"]
    lock = REPO_ROOT / runtime["lock_path"]
    lock_bytes = lock.read_bytes()
    env = (PBS_ROOT / "openreward-v8-abci.env").read_text()
    server = (PBS_ROOT / "openreward-swe-singularity-server.sh").read_text()
    materializer = (PBS_ROOT / "materialize_openreward_v8_swe_env.py").read_text()
    verifier = (PBS_ROOT / "verify_openreward_v8_swe_env.py").read_text()

    assert hashlib.sha256(lock_bytes).hexdigest() == runtime["lock_sha256"]
    lock_text = lock_bytes.decode()
    assert lock_text.count("--hash=sha256:") > runtime["package_count"]
    for name, version in runtime["package_pins"].items():
        assert f"{name}=={version}" in lock_text
    assert f"SWE_REBENCH_SOURCE_REVISION={EXPECTED_REVISIONS['swe_rebench']}" in env
    assert "OPENREWARD_VERIFY_PINNED_SWE_RUNTIME=1" in env
    assert "RUNTIME_VERIFY=${SCRIPT_DIR}/verify_openreward_v8_swe_env.py" in server
    assert "VERIFY_PINNED_RUNTIME=${OPENREWARD_VERIFY_PINNED_SWE_RUNTIME:-0}" in server
    assert 'if [[ "${VERIFY_PINNED_RUNTIME}" == 1 ]]; then' in server
    assert '"${RUNTIME_VERIFY}" --release-root "${PLATOON_REPO_ROOT}"' in server
    assert "--require-hashes" in materializer
    assert '"--link-mode",\n        "copy"' in materializer
    assert "refusing an existing SWE runtime" in materializer
    assert "freeze_sha256" in verifier
    assert '"prefix": str(environment)' in verifier
    assert "SWE runtime key package pins mismatch" in verifier
    assert "SWE runtime root is not immutable, direct, and user-owned" in verifier
    assert "writable or foreign-owned published entry" in verifier


def test_v8_release_gate_materialized_external_revisions_are_exact_and_clean() -> None:
    _require_release_gate()

    paths = {
        "areal": EXTERNAL / "AReaL",
        "openhands": EXTERNAL / "openhands-agent-sdk",
        "swe_rebench": EXTERNAL / "swe-rebench-v2-openrewardenv",
        "toolathlon": EXTERNAL / "toolathlon-gym",
    }
    for name, path in paths.items():
        assert _git(path, "rev-parse", "HEAD") == EXPECTED_REVISIONS[name]
        assert _git(path, "status", "--porcelain=v1", "--untracked-files=all") == ""


def test_v8_openhands_lock_is_exactly_the_materialized_parallel_call_fix() -> None:
    revision = EXPECTED_REVISIONS["openhands"]
    project = (REPO_ROOT / "plugins/openreward/pyproject.toml").read_text()
    lock = (REPO_ROOT / "plugins/openreward/uv.lock").read_text()
    assert project.count(f'rev = "{revision}"') == 4
    assert "d2677bf750ab202efc2808a2b05d4dcd8efa98b5" not in project
    assert lock.count(revision) >= 12


def test_v8_release_gate_openhands_has_parallel_call_stuck_fix() -> None:
    _require_release_gate()
    detector = (EXTERNAL / "openhands-agent-sdk/openhands-sdk/openhands/sdk/conversation/stuck_detector.py").read_text()

    assert "ResponseBatch = tuple[list[ActionEvent], list[ObservationBaseEvent]]" in detector
    assert "_get_complete_response_batches" in detector
    assert "Parallel executors may complete siblings in any order" in detector


def test_v8_tracked_bridge_translates_declared_tool_error() -> None:
    bridge = (REPO_ROOT / "plugins/openreward/platoon/openreward/mcp_bridge.py").read_text()
    contract = 'TOOL_ERROR_META_KEY = "openhands.dev/tool-error"'

    assert contract in bridge
    assert "error_message = _tool_error_message(payload)" in bridge
    assert "raise RuntimeError(error_message)" in bridge


def test_v8_release_gate_binds_toolathlon_and_openhands_error_path() -> None:
    _require_release_gate()
    toolathlon = (EXTERNAL / "toolathlon-gym/server.py").read_text()
    mcp_definition = (EXTERNAL / "openhands-agent-sdk/openhands-sdk/openhands/sdk/mcp/definition.py").read_text()
    parallel_executor = (
        EXTERNAL / "openhands-agent-sdk/openhands-sdk/openhands/sdk/agent/parallel_executor.py"
    ).read_text()

    contract = 'TOOL_ERROR_META_KEY = "openhands.dev/tool-error"'
    assert contract in toolathlon
    assert "metadata={" in toolathlon and "TOOL_ERROR_META_KEY:" in toolathlon
    # FastMCP turns that bridge exception into CallToolResult.isError. OpenHands
    # preserves the bit on its observation, while unexpected parallel-executor
    # exceptions become an AgentErrorEvent instead of killing the conversation.
    assert "is_error=result.isError" in mcp_definition
    assert "Converts exceptions to ``AgentErrorEvent``" in parallel_executor
    assert "AgentErrorEvent(" in parallel_executor


def test_v8_env_and_launchers_pin_exact_image_labels_fail_closed() -> None:
    env = (PBS_ROOT / "openreward-v8-abci.env").read_text()
    common = (PBS_ROOT / "openreward-ta-swe-prealloc-common.sh").read_text()
    services = (PBS_ROOT / "openreward-node-services.sh").read_text()

    assert "set +x" in env
    assert "PLATOON_REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8-376d1b9-18e62c0" in env
    assert "openreward-toolathlon-18e62c0d-c1e60b36.sif" in env
    for expected in (
        "e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267",
        "18e62c0d0419da88209af911528f2f8a0b44a5ed",
        "https://github.com/ApGa/toolathlon-gym",
        "sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd",
    ):
        assert expected in env
    for source in (common, services):
        assert "Toolathlon SIF source URL mismatch" in source
        assert "Toolathlon SIF source revision mismatch" in source
        assert "Toolathlon SIF OCI digest mismatch" in source
    assert '"${PBSDSH}" -n "${vnode_index}" -- /usr/bin/env' in common
    assert '"PLATOON_ABCI_ENV_FILE=${ABCI_ENV}"' in common
    assert '"PLATOON_NODE_SERVICES_CLEANUP_RECEIPT=${service_receipt}"' in common
    assert '"${NODE_SERVICES}"' in common


def test_v8_release_verifier_is_fail_closed_and_manifest_driven() -> None:
    verifier = (PBS_ROOT / "verify_openreward_v8_release.py").read_text()

    assert 'RELEASE_ROOT = Path("/home/aci18912it/rasp/releases/platoon-v8-376d1b9-18e62c0")' in verifier
    assert 'RELEASE_REF = "refs/tags/abci-v8-376d1b9-18e62c0"' in verifier
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in verifier
    assert "release HEAD does not match its pinned tag" in verifier
    assert "release component revisions mismatch" in verifier
    assert "release Toolathlon image identity mismatch" in verifier
    assert "materialized external checkout is dirty" in verifier
    assert "release tracks retired migration debris" in verifier

    # During isolated release preparation, deletions are intentionally reviewed
    # before they are staged.  Check the effective source tree here; the launch
    # verifier repeats this against the committed index in the tagged release.
    tracked = [path for path in _git(REPO_ROOT, "ls-files").splitlines() if (REPO_ROOT / path).exists()]
    assert not any(path.endswith(".orig") for path in tracked)
    assert not any("2132698" in path for path in tracked)

    completed = subprocess.run(
        [
            sys.executable,
            str(PBS_ROOT / "verify_openreward_v8_release.py"),
            "--release-root",
            str(REPO_ROOT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode != 0
    assert "release root is not the exact canonical path" in completed.stderr
