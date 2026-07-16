"""Regression coverage for the resilient Toolathlon node service."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = REPO_ROOT / "plugins" / "openreward" / "scripts" / "openreward-toolathlon-resilient-entrypoint.sh"
LAUNCHER = REPO_ROOT / "slurm-scripts" / "openreward-toolathlon-prealloc-base.sh"
MULTIENV_LAUNCHER = REPO_ROOT / "slurm-scripts" / "openreward-multienv-prealloc.sh"
KEEPALIVE = REPO_ROOT / "slurm-scripts" / "gpu_keepalive.py"
PREPARE_ENV = REPO_ROOT / "slurm-scripts" / "prepare_openreward_env.sh"


def test_toolathlon_supervisor_has_stable_port_bounded_restart_semantics():
    subprocess.run(["bash", "-n", str(SUPERVISOR)], check=True)
    script = SUPERVISOR.read_text()

    assert SUPERVISOR.stat().st_mode & 0o111
    assert "local port=$((BASE_PORT + index))" in script
    assert "worker_pids[index]=${pid}" in script
    assert "pid_to_worker[${pid}]=${index}" in script
    assert 'if wait -n -p exited_pid "${wait_pids[@]}"' in script
    assert 'start_worker "${index}"' in script
    assert "worker ${index} exhausted restart budget" in script
    assert 'if [[ "${exited_pid}" == "${nginx_pid}" ]]' in script
    assert "nginx exited unexpectedly" in script
    assert "trap cleanup EXIT" in script
    assert "PG_CONF=$(find /etc/postgresql -name postgresql.conf -print -quit)" in script


def test_toolathlon_launcher_mounts_supervisor_without_bad_exit_cascade():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    launcher = LAUNCHER.read_text()

    assert LAUNCHER.stat().st_mode & 0o111
    assert KEEPALIVE.is_file()
    assert PREPARE_ENV.stat().st_mode & 0o111
    assert "/users/apurvag" not in launcher
    for secret_name in (
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "LITELLM_API_KEY",
        "HF_TOKEN",
    ):
        assert f"export {secret_name}=" not in launcher

    assert (
        "TOOLATHLON_SERVER_ENTRYPOINT=${OPENREWARD_TOOLATHLON_SERVER_ENTRYPOINT:-"
        "${REPO_ROOT}/plugins/openreward/scripts/"
        "openreward-toolathlon-resilient-entrypoint.sh}"
    ) in launcher
    assert ("TOOLATHLON_CONTAINER_ENTRYPOINT=/app/openreward-toolathlon-resilient-entrypoint.sh") in launcher
    assert "--kill-on-bad-exit=0" in launcher
    assert "env_servers_healthy" in launcher
    assert "pool=Toolathlon endpoint=${failed_endpoint}" in launcher
    assert "srun_pid=${server_pid} srun_state=${srun_state}" in launcher
    assert "WARNING: Toolathlon env-server step died" in launcher
    assert "WARNING: Toolathlon env-server endpoint is unreachable" in launcher
    assert "stop_gpu_keepalive_before_training" in launcher
    assert ("${TOOLATHLON_SERVER_ENTRYPOINT}:${TOOLATHLON_CONTAINER_ENTRYPOINT}:ro") in launcher
    assert "/app/entrypoint.sh" not in launcher


def test_multienv_launcher_isolates_rank_loss_and_reports_failed_pool_endpoint():
    subprocess.run(["bash", "-n", str(MULTIENV_LAUNCHER)], check=True)
    launcher = MULTIENV_LAUNCHER.read_text()

    assert MULTIENV_LAUNCHER.stat().st_mode & 0o111
    assert "--kill-on-bad-exit=0" in launcher
    assert "--kill-on-bad-exit=1" not in launcher
    assert "server_pool_healthy" in launcher
    assert "pool=${name} endpoint=${failed_endpoint}" in launcher
    assert "srun_pid=${pid} srun_state=${srun_state}" in launcher
    assert 'server_pool_healthy TMax "${TMAX_PORT}" "${tmax_pid}"' in launcher
    assert 'server_pool_healthy SWE-rebench "${SWE_PORT}" "${swe_pid}"' in launcher
    assert "${issue_summary}" in launcher


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def _read_count(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def test_toolathlon_supervisor_restarts_only_failed_port_and_stops_cleanly(tmp_path):
    worker = tmp_path / "worker.sh"
    nginx = tmp_path / "nginx.sh"
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    _write_executable(
        worker,
        """#!/bin/bash
set -euo pipefail
count_file="${TEST_STATE_DIR}/worker-${PORT}.count"
count=0
if [[ -f "${count_file}" ]]; then
    count=$(cat "${count_file}")
fi
count=$((count + 1))
printf '%s\n' "${count}" >"${count_file}"
if [[ "${PORT}" == "${TEST_FAIL_PORT}" && "${count}" -eq 1 ]]; then
    exit 143
fi
trap 'exit 0' TERM INT
while true; do
    sleep 0.1
done
""",
    )
    _write_executable(
        nginx,
        """#!/bin/bash
set -euo pipefail
if [[ "$*" == *" -t "* || " $*" == *" -t "* ]]; then
    exit 0
fi
if [[ "${1:-}" == "-s" ]]; then
    if [[ -f "${TEST_STATE_DIR}/nginx.pid" ]]; then
        kill -TERM "$(cat "${TEST_STATE_DIR}/nginx.pid")" 2>/dev/null || true
    fi
    exit 0
fi
printf '%s\n' "$$" >"${TEST_STATE_DIR}/nginx.pid"
trap 'exit 0' TERM INT
while true; do
    sleep 0.1
done
""",
    )

    env = {
        **os.environ,
        "OPENREWARD_ENTRYPOINT_SKIP_POSTGRES": "1",
        "OPENREWARD_PORT": "19999",
        "OPENREWARD_WORKERS": "2",
        "OPENREWARD_WORKER_BASE_PORT": "19000",
        "OPENREWARD_WORKER_RESTART_MAX_ATTEMPTS": "2",
        "OPENREWARD_WORKER_RESTART_RESET_SECS": "60",
        "OPENREWARD_WORKER_RESTART_BACKOFF_INITIAL_SECS": "0",
        "OPENREWARD_WORKER_RESTART_BACKOFF_MAX_SECS": "0",
        "OPENREWARD_SERVER_SHUTDOWN_GRACE_SECS": "2",
        "OPENREWARD_SERVER_PYTHON": str(worker),
        "OPENREWARD_SERVER_APP": "unused",
        "OPENREWARD_NGINX_BIN": str(nginx),
        "OPENREWARD_NGINX_CONFIG": str(tmp_path / "nginx.conf"),
        "OPENREWARD_NGINX_PID_FILE": str(tmp_path / "nginx.pid"),
        "OPENREWARD_NGINX_LOG_DIR": str(tmp_path / "nginx-log"),
        "TEST_STATE_DIR": str(state_dir),
        "TEST_FAIL_PORT": "19000",
    }
    proc = subprocess.Popen(
        [str(SUPERVISOR)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    output = ""
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if _read_count(state_dir / "worker-19000.count") >= 2:
                break
            if proc.poll() is not None:
                output = proc.communicate()[0]
                raise AssertionError(f"supervisor exited before restart:\n{output}")
            time.sleep(0.05)
        else:
            raise AssertionError("timed out waiting for failed worker restart")

        assert _read_count(state_dir / "worker-19000.count") == 2
        assert _read_count(state_dir / "worker-19001.count") == 1
        assert proc.poll() is None

        proc.terminate()
        output = proc.communicate(timeout=8)[0]
        assert proc.returncode == 0
        assert "worker 0 exited unexpectedly" in output
        assert "status=143" in output
        assert "restart 1/2" in output
        assert "received TERM; stopping" in output
        assert "[entrypoint] shutting down" in output
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
