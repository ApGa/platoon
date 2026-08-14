"""Structural checks for production patch validation in immutable Gate2."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE2 = REPO_ROOT / "pbs-scripts/openreward-immutable-runtime-gate2.pbs"
GATE2_V9 = REPO_ROOT / "pbs-scripts/openreward-immutable-runtime-gate2-v9.pbs"


def test_gate2_uses_the_canonical_explicit_cluster_environment():
    source = GATE2.read_text()

    assert "ABCI_ENV=${PLATOON_ABCI_ENV_FILE:-/home/aci18912it/rasp/config/abci.env}" in source
    assert '[[ "${ABCI_ENV}" == /* && -r "${ABCI_ENV}" ]]' in source
    assert '[[ "$(readlink -f "${ABCI_ENV}")" == "${ABCI_ENV}" ]]' in source
    assert "export PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" in source


def test_gate2_validates_release_only_assets_before_source_resolution():
    source = GATE2.read_text()

    release = '"${RELEASE_VERIFY}" --release-root "${REPO_ROOT}"'
    bash_loop = "for shell_asset in \\\n"
    python_probe = '"${CONTROLLER_PYTHON_BASE}" -I - "${REPO_ROOT}"'
    resolve = 'echo "[gate2] resolving exact source/content identity"'
    assert release in source
    assert bash_loop in source
    for asset in (
        "openreward-ta-swe-prealloc-common.sh",
        "openreward-node-services.sh",
        "openreward-ta-swe-16node-nonrecursive-h200-v8-abci.pbs",
        "openreward-ta-swe-32node-recursive-h200-v8-abci.pbs",
        "openreward-node-services-pbsdsh-smoke.pbs",
        "openreward-v8-ptc-e2e-smoke.pbs",
    ):
        assert asset in source
    assert "/bin/bash -n" in source
    assert python_probe in source
    assert "ast.parse(path.read_text(), filename=str(path))" in source
    assert "openreward-service-cleanup-receipt.py" in source
    assert "openreward-service-shutdown-request.py" in source
    assert "openreward-controller-runtime-receipt.py" in source
    assert "openreward_v8_ptc_e2e.py" in source
    assert source.index(release) < source.index(bash_loop) < source.index(python_probe) < source.index(resolve)


def test_gate2_archive_requires_v8_provenance_and_both_fresh_leaf_configs():
    source = GATE2.read_text()

    assert "    pbs-scripts/openreward-v8-abci.env \\\n" in source
    assert "    pbs-scripts/openreward-v8-release.json \\\n" in source
    assert "    pbs-scripts/verify_openreward_v8_release.py \\\n" in source
    assert "    pbs-scripts/verify_openreward_v8_swe_env.py \\\n" in source
    assert "    pbs-scripts/materialize_openreward_v8_swe_env.py \\\n" in source
    assert "host-only launch asset" in source
    assert "for host_only_asset in \\\n" in source
    assert "pbs-scripts/openreward_v8_ptc_e2e.py \\\n" in source
    assert "pbs-scripts/openreward-service-cleanup-receipt.py" in source
    assert "pbs-scripts/openreward-service-shutdown-request.py" in source
    assert "pbs-scripts/openreward-controller-runtime-receipt.py" in source
    assert "    containers/swe-rebench-538b5c5-py311.lock.txt \\\n" in source
    assert "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v8.yaml" in source
    assert "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v8.yaml" in source
    assert "immutable source archive contains retired migration debris" in source
    assert r"[^/]+\.orig$" in source
    assert "cleanup_openreward_2132698_generation" in source


def test_additive_v9_gate2_selects_future_overlay_without_replacing_v8_entrypoint():
    source = GATE2.read_text()
    wrapper = GATE2_V9.read_text()

    assert "GATE2_RELEASE_FLAVOR=${OPENREWARD_GATE2_RELEASE_FLAVOR:-v8}" in source
    assert "openreward_v8_release.py" in source
    assert "verify_openreward_v9_release.py" in source
    assert "GATE_ROOT_BASENAME=gate2-immutable-runtime-v9" in source
    assert "publish_toolathlon_pbs_build_receipt.py" in source
    assert 'TOOLATHLON_SIF_SHA256=$(sha256sum "${OPENREWARD_TOOLATHLON_SIF}"' in source
    assert "--toolathlon-build-receipt-sha256" in source
    assert "--toolathlon-portability-patch-sha256" in source
    assert "gate2_immutable_runtime_v9_candidate=passed" in source
    assert "publish_openreward_v9_gate2_result.py" in source
    assert 'publish_openreward_v9_gate2_result.py" finalize' not in source
    assert 'cat >"${GATE_ROOT}/RESULT" <<EOF' in source
    assert "OPENREWARD_GATE2_RELEASE_FLAVOR=v9" in wrapper
    assert "openreward-v9-abci.env" in wrapper
    assert "openreward-immutable-runtime-gate2.pbs" in wrapper
    assert "qsub" not in wrapper.lower()


def test_v9_gate2_archive_has_explicit_allow_and_host_only_deny_sets():
    source = GATE2.read_text()
    adapter = (REPO_ROOT / "pbs-scripts/prepare_openreward_env.sh").read_text()

    for asset in (
        "openreward-v9-abci.env",
        "openreward-v9-release.json",
        "verify_openreward_v9_release.py",
        "verify_toolathlon_all_mcp_health_result.py",
        "verify_openreward_v9_gate2_result.py",
    ):
        assert asset in source
        assert f'print "    pbs-scripts/{asset}"' in adapter
    for config in (
        "abci_toolathlon_swe_openhands_areal_prealloc_16node_nonrecursive_h200_cp1_c12_ta20_v9.yaml",
        "abci_toolathlon_swe_openhands_areal_prealloc_32node_recursive_h200_cp1_c24_ta20_v9.yaml",
    ):
        assert config in source
    for host_only in (
        "openreward-toolathlon-pbs-portable-build.pbs",
        "openreward-toolathlon-all-mcp-health-v9.pbs",
        "openreward-node-local-init.py",
        "toolathlon_all_mcp_health.py",
        "publish_toolathlon_all_mcp_health_result.py",
        "publish_openreward_v9_gate2_result.py",
        "publish_toolathlon_pbs_build_receipt.py",
        "launch_toolathlon_python_mcp.py",
        "probe_toolathlon_python_mcp_launchers.py",
    ):
        assert host_only in source
    assert '"pbs-scripts/openreward-node-local-init.py",' in source
    assert '"containers/launch_toolathlon_python_mcp.py",' in source
    assert '"containers/probe_toolathlon_python_mcp_launchers.py",' in source
    assert 'print "    pbs-scripts/publish_openreward_v9_gate2_result.py"' not in adapter
    assert 'print "    pbs-scripts/openreward-node-local-init.py"' not in adapter


def test_gate2_archives_and_imports_rollout_rpc_patch_entrypoint():
    source = GATE2.read_text()

    assert "    platoon/areal_rpc_server.py \\\n" in source
    assert "import platoon.areal_rpc_server as areal_rpc_server" in source
    assert "areal_rpc_server._apply_platoon_areal_rollout_worker_patches()" in source


def test_gate2_archives_and_validates_pbs_worker_python_fork_override():
    source = GATE2.read_text()

    assert "    platoon/train/areal/fork_lifecycle.py \\\n" in source
    assert "    platoon/train/areal/preallocated_pbs.py \\\n" in source
    assert 'PreallocatedPBSScheduler.__dict__["_fork_single_worker"]' in source
    assert '"__platoon_worker_python_fork_patch__"' in source
    assert '"self", "session", "role", "idx", "target_wi", "target_role", "command",' in source
    assert 'assert "_ForkWorkerPythonSession" in fork_source' in source
    assert 'assert "super()._fork_single_worker" in fork_source' in source
    assert '"PLATOON_AREAL_PBS_WORKER_PYTHON"' in source


def test_gate2_proves_shared_fork_alias_cleanup_on_all_scheduler_paths():
    source = GATE2.read_text()

    assert "areal_local.LocalScheduler" in source
    assert "areal_slurm.SlurmScheduler" in source
    assert '"__platoon_fork_alias_cleanup_patch__"' in source
    assert "PreallocatedSlurmScheduler" in source
    assert "PreallocatedPBSScheduler" in source
    assert 'assert "cleanup_forked_alias_workers" in delete_source' in source
    assert 'assert "cleanup_dependent_aliases_before_owner" in delete_source' in source
    assert "preallocated_slurm.cleanup_forked_alias_workers is preallocated_pbs.cleanup_forked_alias_workers" in source
    assert "preallocated_slurm.cleanup_dependent_aliases_before_owner" in source
    assert "fork_lifecycle.cleanup_forked_alias_workers" in source
    assert "fork_lifecycle.cleanup_dependent_aliases_before_owner" in source
    assert "platoon_areal_patches._patch_rollout_controller_cleanup_error_propagation" in source
    assert '"__platoon_cleanup_error_propagation_patch__"' in source
    assert "RolloutController.destroy" in source
    assert "platoon_areal_patches._patch_rollout_controller_cleanup_error_propagation" in source
    assert 'assert "_CleanupRecordingScheduler" in destroy_patch_source' in source
    assert 'assert "cleanup_errors" in destroy_patch_source' in source
    assert 'assert "self.scheduler = scheduler" in destroy_patch_source' in source


def test_gate2_validates_worker_python_target_and_proxy_entrypoint_in_sif():
    source = GATE2.read_text()

    assert 'assert os.readlink(worker_python) == "/usr/bin/python3.12"' in source
    assert "import platoon.areal_proxy_rollout as areal_proxy_rollout" in source
    assert "assert callable(areal_proxy_rollout.main)" in source
    assert '"__platoon_name_resolve_cleanup__"' in source
    assert 'assert "_reset_name_resolve" in proxy_main_source' in source


def test_gate2_proves_actual_worker_child_exit_and_cleanup_methods_are_patched():
    source = GATE2.read_text()

    assert "remote_inf_engine.RemoteInfEngine._wait_for_server" in source
    assert "remote_inf_engine.RemoteInfEngine.launch_server" in source
    assert '"__platoon_server_child_exit_patch__"' in source
    assert '"__platoon_server_child_exit_cleanup_patch__"' in source
    assert '"self", "address", "process",' in source
    assert '"self", "server_args",' in source


def test_gate2_proves_actual_controller_sync_and_async_methods_are_patched():
    source = GATE2.read_text()

    assert "areal_slurm.SlurmScheduler.call_engine" in source
    assert "areal_slurm.SlurmScheduler.async_call_engine" in source
    assert source.count('"__platoon_engine_call_error_passthrough_patch__"') == 2
    assert "patched_call_engine.__wrapped__" in source
    assert "patched_async_call_engine.__wrapped__" in source
