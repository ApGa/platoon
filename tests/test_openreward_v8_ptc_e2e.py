"""Structural and receipt tests for the v8 latest-stack PTC acceptance."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PBS_ROOT = REPO_ROOT / "pbs-scripts"
WRAPPER = PBS_ROOT / "openreward-v8-ptc-e2e-smoke.pbs"
DRIVER = PBS_ROOT / "openreward_v8_ptc_e2e.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("openreward_v8_ptc_e2e", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v8_ptc_wrapper_uses_one_node_exact_release_and_passed_gate2() -> None:
    source = WRAPPER.read_text()

    assert "#PBS -l select=1:mpiprocs=8" in source
    assert "#PBS -l walltime=00:30:00" in source
    assert "REPO_ROOT=/home/aci18912it/rasp/releases/platoon-v8r2-376d1b9-18e62c0" in source
    assert "OPENREWARD_V8_PTC_GATE2_RESULT" in source
    assert '[[ "$(result_value "${GATE2_RESULT}" gate2)" == passed ]]' in source
    assert '"${current_source_sha}" == "${source_sha}"' in source
    assert '"${current_worker_key}" == "${worker_key}"' in source
    assert 'grep -Fxq "source_sha256=${source_sha}" "${controller_manifest}"' in source
    assert '"${RELEASE_VERIFY}" --release-root "${REPO_ROOT}"' in source
    assert "export PLATOON_ABCI_ENV_FILE=${ABCI_ENV}" in source
    assert "OPENREWARD_VERIFY_PINNED_SWE_RUNTIME" in source
    assert "qsub" not in source.lower()


def test_v8_ptc_wrapper_runs_exact_production_service_and_cleans_it_before_pass() -> None:
    source = WRAPPER.read_text()

    launch = '"${PBSDSH}" -n 0 -- /usr/bin/env'
    health = "exact production service endpoints are healthy"
    driver = '"${controller_python}" -I "${DRIVER_SNAPSHOT}"'
    request = '"${controller_python}" -I "${SHUTDOWN_REQUEST_HELPER}" publish'
    teardown = 'teardown_service || die "production node-service cleanup handshake attestation failed"'
    publish = "openreward_v8_ptc_e2e=PASS"
    for marker in (launch, health, driver, request, teardown, publish):
        assert marker in source
    assert source.index(launch) < source.index(health) < source.index(driver)
    natural_request_call = source.rindex("request_service_shutdown || die")
    assert source.rindex(driver) < natural_request_call < source.rindex(teardown) < source.rindex(publish)
    assert "production node-service cleanup handshake attestation failed" in source
    assert "node-service helper died after the PTC driver" in source
    assert "an environment endpoint became unhealthy after the PTC driver" in source
    assert "PLATOON_NODE_SERVICES_CLEANUP_RECEIPT" in source
    assert "expected_primary_status=0" in source
    assert "expected_primary_status=143" in source
    assert '--primary-status "${expected_primary_status}"' in source
    assert "assert-tcp-closed" in source
    assert "assert-empty-swe" in source
    assert "trap cleanup EXIT" in source
    assert "trap 'exit 143' TERM" in source
    assert "trap 'exit 130' INT" in source
    assert "service_shutdown_requested=1" in source
    assert '"PLATOON_NODE_SERVICES_SHUTDOWN_REQUEST=${SERVICE_REQUEST}"' in source
    assert "PLATOON_STAGE_TOOLATHLON_SIF=1" in source
    assert "PLATOON_STAGE_TRAINING_SIF=0" in source
    assert '"PLATOON_ABCI_ENV_FILE=${ABCI_ENV}"' in source
    assert '"${NODE_SERVICES}"' in source


def test_v8_ptc_wrapper_scrubs_secrets_and_snapshots_driver() -> None:
    source = WRAPPER.read_text()

    for name in (
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        "RUBRIC_DEFAULT_LLM",
        "HF_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        assert f"-u {name}" in source
        assert f'unset "${{{name}}}"' not in source
    assert 'driver_sha=$(sha256sum "${DRIVER}"' in source
    assert 'chmod 0500 "${DRIVER_SNAPSHOT}.partial"' in source
    assert '"$(sha256sum "${DRIVER_SNAPSHOT}.partial"' in source
    assert '--driver-sha256 "${driver_sha}"' in source


def test_v8_ptc_driver_binds_latest_toolathlon_bridge_openhands_and_ptc() -> None:
    source = DRIVER.read_text()

    assert 'EXPECTED_OPENHANDS_REVISION = "b1acf52ee362884f092b9eaf5e701a1bb5d1b335"' in source
    assert 'EXPECTED_TOOLATHLON_REVISION = "18e62c0d0419da88209af911528f2f8a0b44a5ed"' in source
    assert "_direct_url_revision(distribution)" in source
    assert "_build_mcp_config(_Task(), openreward, environment" in source
    assert "_configure_openreward_urls(environment, allow_legacy_global_pool=False)" in source
    assert "OPENREWARD_ENVIRONMENT_LABEL_KEY" in source
    assert 'server["args"][:2] != ["-m", "platoon.openreward.mcp_bridge"]' in source
    assert "resources.enter_context(create_mcp_tools(mcp_config, timeout=180.0))" in source
    assert "ProgrammaticToolCallingTool.create(" in source
    assert "ProgrammaticToolCallingMode.ORCHESTRATION_ONLY" in source
    assert "max_tool_calls_per_execution=1024" in source
    assert '"environment_max_tool_calls": 8' in source
    assert 'os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"' in source
    assert "def _closing_executor(executor: Any):" in source
    assert "finally:\n        executor.close()" in source
    assert "_closing_executor(programmatic_tool.executor)" in source
    assert "resources.enter_context(" in source


def test_v8_ptc_driver_lazy_runtime_imports_and_production_mcp_config(tmp_path: Path) -> None:
    """Use a real controller env when explicitly supplied by the release gate."""

    controller_python = os.environ.get("OPENREWARD_V8_TEST_CONTROLLER_PYTHON")
    controller_site = os.environ.get("OPENREWARD_V8_TEST_CONTROLLER_SITE_PACKAGES")
    if not controller_python or not controller_site:
        import pytest

        pytest.skip("exact Gate2 controller environment was not supplied")
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        import platoon.openreward.tasks as tasks
        from platoon.openreward.config_defs import OpenRewardConfig, OpenRewardEnvironmentConfig
        from platoon.openreward.rollout import _build_mcp_config

        assert tasks.OPENREWARD_ENVIRONMENT_LABEL_KEY == "openreward_environment_label"
        env = OpenRewardEnvironmentConfig(
            label="toolathlon", env_name="toolathlongym", session_urls=["http://node:8082"]
        )
        tasks._configure_openreward_urls(env, allow_legacy_global_pool=False)
        assert os.environ["OPENREWARD_SESSION_URL"] == "http://node:8082"
        class Task:
            id = "probe"
            misc = {{
                tasks.OPENREWARD_ENVIRONMENT_LABEL_KEY: "toolathlon",
                tasks.OPENREWARD_TASK_INDEX_KEY: 0,
                tasks.OPENREWARD_TASK_NAME_KEY: "probe",
                tasks.OPENREWARD_TASK_SPLIT_KEY: "train",
            }}
        config = _build_mcp_config(Task(), OpenRewardConfig(environments=[env]), env, {str(tmp_path)!r})
        server = config["mcpServers"]["openreward"]
        assert server["args"][:2] == ["-m", "platoon.openreward.mcp_bridge"]
        print("v8_ptc_driver_lazy_imports=PASS")
        """
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": f"{REPO_ROOT / 'plugins/openreward'}:{controller_site}",
        "OPENHANDS_SUPPRESS_BANNER": "1",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    }
    completed = subprocess.run(
        [controller_python, "-P", "-c", script],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        env=environment,
    )
    assert "v8_ptc_driver_lazy_imports=PASS" in completed.stdout


def test_v8_ptc_driver_requires_concurrent_error_success_and_live_followup() -> None:
    source = DRIVER.read_text()

    assert "bad, good = await asyncio.gather(" in source
    assert "atools.python_execute" in source
    assert 'first.failed_tool_names != ("python_execute",)' in source
    assert ") != (2, 2, 2, 0):" in source
    assert 'if "True False" not in first.text:' in source
    assert "persistent_probe" in source
    assert "ptc_sibling.txt" in source
    assert ") != (1, 1, 1, 0):" in source
    assert 'typed[0].get("kind") != "nonzero_exit"' in source
    assert 'state.get("turn") != 3 or state.get("finished") is not False' in source


def test_v8_ptc_driver_publishes_private_result_atomically(tmp_path: Path) -> None:
    driver = _load_driver()
    result = tmp_path / "RESULT.json"
    payload = {"status": "passed", "marker": "safe"}

    driver._private_json(result, payload)

    assert json.loads(result.read_text()) == payload
    assert stat.S_IMODE(result.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.tmp"))


def test_v8_ptc_assets_are_executable_and_wrapper_has_valid_shell() -> None:
    assert os.access(WRAPPER, os.X_OK)
    assert os.access(DRIVER, os.X_OK)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
