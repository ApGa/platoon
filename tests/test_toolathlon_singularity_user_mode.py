"""ABCI coverage for Toolathlon's unprivileged Singularity service path."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "plugins/openreward/scripts/openreward-toolathlon-resilient-entrypoint.sh"
STAGER = REPO_ROOT / "plugins/openreward/scripts/openreward-toolathlon-stage-pgdata.sh"
PBS_SMOKE = REPO_ROOT / "pbs-scripts/openreward-toolathlon-singularity-smoke.pbs"
SESSION_SMOKE = REPO_ROOT / "pbs-scripts/toolathlon_session_smoke.py"
RECOVERY_SMOKE = REPO_ROOT / "pbs-scripts/toolathlon_worker_recovery_smoke.py"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def test_pgdata_stager_publishes_owned_cluster_atomically(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    (seed / "global").mkdir(parents=True)
    (seed / "base/1").mkdir(parents=True)
    (seed / "PG_VERSION").write_text("14\n")
    (seed / "global/pg_control").write_bytes(b"control")
    (seed / "base/1/2673").write_bytes(b"relation")
    destination = tmp_path / "node-local/postgresql/14/main"

    result = subprocess.run(
        [str(STAGER), str(destination)],
        env={**os.environ, "OPENREWARD_PG_SEED_DIR": str(seed)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert destination.stat().st_uid == os.getuid()
    assert destination.stat().st_mode & 0o777 == 0o700
    assert (destination / "PG_VERSION").read_text() == "14\n"
    assert (destination / "base/1/2673").read_bytes() == b"relation"
    assert not list(destination.parent.glob(".openreward-pgdata-stage.*"))

    again = subprocess.run(
        [str(STAGER), str(destination)],
        env={**os.environ, "OPENREWARD_PG_SEED_DIR": str(seed)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert again.returncode == 0
    assert "existing PostgreSQL cluster is ready" in again.stdout


def test_pgdata_stager_refuses_nonempty_invalid_destination(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    (seed / "global").mkdir(parents=True)
    (seed / "PG_VERSION").write_text("14\n")
    (seed / "global/pg_control").write_bytes(b"control")
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "do-not-delete"
    sentinel.write_text("preserve\n")

    result = subprocess.run(
        [str(STAGER), str(destination)],
        env={**os.environ, "OPENREWARD_PG_SEED_DIR": str(seed)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "refusing nonempty, invalid destination" in result.stderr
    assert sentinel.read_text() == "preserve\n"


def test_user_mode_starts_direct_postgres_and_uses_only_writable_paths(tmp_path: Path) -> None:
    state = tmp_path / "state"
    pgdata = state / "postgresql/14/main"
    (pgdata / "global").mkdir(parents=True)
    (pgdata / "PG_VERSION").write_text("14\n")
    (pgdata / "global/pg_control").write_bytes(b"control")
    tools = tmp_path / "tools"
    tools.mkdir()
    calls = tmp_path / "calls"
    calls.mkdir()

    pg_ctl = tools / "pg_ctl"
    _write_executable(
        pg_ctl,
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"${TEST_CALLS}/pg_ctl"
action=${!#}
case "${action}" in
  start) touch "${TEST_CALLS}/postgres.started" ;;
  status) test -f "${TEST_CALLS}/postgres.started" ;;
  stop) rm -f "${TEST_CALLS}/postgres.started" ;;
  *) exit 2 ;;
esac
""",
    )
    pg_isready = tools / "pg_isready"
    _write_executable(
        pg_isready,
        """#!/bin/bash
test -f "${TEST_CALLS}/postgres.started"
""",
    )
    psql = tools / "psql"
    _write_executable(
        psql,
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"${TEST_CALLS}/psql"
""",
    )
    worker = tools / "worker"
    _write_executable(
        worker,
        """#!/bin/bash
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
    )
    nginx = tools / "nginx"
    _write_executable(
        nginx,
        """#!/bin/bash
set -euo pipefail
printf '%s\n' "$*" >>"${TEST_CALLS}/nginx.args"
if [[ " $* " == *" -t "* ]]; then
  exit 0
fi
if [[ "${1:-}" == "-s" ]]; then
  kill -TERM "$(cat "${TEST_CALLS}/nginx.pid")" 2>/dev/null || true
  exit 0
fi
printf '%s\n' "$$" >"${TEST_CALLS}/nginx.pid"
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
    )

    env = {
        **os.environ,
        "OPENREWARD_RUNTIME_MODE": "user",
        "OPENREWARD_STATE_DIR": str(state),
        "OPENREWARD_PGDATA": str(pgdata),
        "OPENREWARD_PG_CTL": str(pg_ctl),
        "OPENREWARD_PG_ISREADY": str(pg_isready),
        "OPENREWARD_PSQL_BIN": str(psql),
        "OPENREWARD_SERVER_PYTHON": str(worker),
        "OPENREWARD_SERVER_APP": "unused",
        "OPENREWARD_NGINX_BIN": str(nginx),
        "OPENREWARD_WORKERS": "1",
        "OPENREWARD_PORT": "19998",
        "OPENREWARD_WORKER_BASE_PORT": "19900",
        "OPENREWARD_PG_PORT": "15432",
        "OPENREWARD_PG_MAX_CONNECTIONS": "37",
        "OPENREWARD_SERVER_SHUTDOWN_GRACE_SECS": "2",
        "TEST_CALLS": str(calls),
    }
    proc = subprocess.Popen(
        [str(ENTRYPOINT)],
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
            if (calls / "nginx.pid").exists() and (calls / "postgres.started").exists():
                break
            if proc.poll() is not None:
                output = proc.communicate()[0]
                raise AssertionError(f"entrypoint exited before readiness:\n{output}")
            time.sleep(0.05)
        else:
            raise AssertionError("timed out waiting for mocked user-mode service")

        nginx_config = (state / "nginx/nginx.conf").read_text()
        nginx_error_log = state / "nginx/log/error.log"
        pg_config = (pgdata / "postgresql.conf").read_text()
        assert "user root;" not in nginx_config
        assert str(state / "nginx/tmp/client-body") in nginx_config
        assert f"error_log {nginx_error_log} warn;" in nginx_config
        assert "error_log /dev/stderr" not in nginx_config
        assert "access_log off;" in nginx_config
        assert nginx_error_log.is_file()
        assert nginx_error_log.stat().st_mode & 0o777 == 0o600
        assert f"data_directory = '{pgdata}'" in pg_config
        assert "port = 15432" in pg_config
        assert "max_connections = 37" in pg_config
        assert "ssl = off" in pg_config
        worker_pid_file = state / "worker-pids/worker-0.pid"
        assert int(worker_pid_file.read_text().strip()) > 1

        proc.terminate()
        output = proc.communicate(timeout=8)[0]
        assert proc.returncode == 0
        assert not (calls / "postgres.started").exists()
        pg_ctl_calls = (calls / "pg_ctl").read_text()
        assert "-l " in pg_ctl_calls and " start" in pg_ctl_calls
        assert "-m fast" in pg_ctl_calls and " stop" in pg_ctl_calls
        assert "service postgresql" not in output
        assert "runtime_mode=user" in output
        assert "starting user PostgreSQL" in output
        assert "received TERM; stopping" in output
        assert not worker_pid_file.exists()
        nginx_calls = (calls / "nginx.args").read_text()
        assert " -e " not in f" {nginx_calls} "
        assert "-t -c" in nginx_calls
        assert "-s quit -c" in nginx_calls
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


def test_nginx_log_path_override_cannot_inject_configuration(tmp_path: Path) -> None:
    state = tmp_path / "state"
    injected_config = tmp_path / "injected.conf"
    malicious_log_dir = f"{tmp_path}/logs; include {injected_config};"

    result = subprocess.run(
        [str(ENTRYPOINT)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "OPENREWARD_RUNTIME_MODE": "user",
            "OPENREWARD_STATE_DIR": str(state),
            "OPENREWARD_NGINX_LOG_DIR": malicious_log_dir,
            "OPENREWARD_WORKERS": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "NGINX_LOG_DIR must be an absolute nginx-safe path" in result.stderr
    assert not (state / "nginx/nginx.conf").exists()
    assert not injected_config.exists()


def test_nginx_config_failure_reports_bounded_sanitized_log_tail(tmp_path: Path) -> None:
    state = tmp_path / "state"
    tools = tmp_path / "tools"
    tools.mkdir()

    worker = tools / "worker"
    _write_executable(
        worker,
        """#!/bin/bash
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
    )
    nginx = tools / "nginx"
    _write_executable(
        nginx,
        """#!/bin/bash
set -euo pipefail
is_test=0
while (($#)); do
  case "$1" in
    -t)
      is_test=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done
if ((is_test)); then
  {
    printf 'OLDEST_LOG_CONTENT_MUST_BE_TRUNCATED\\n'
    head -c 24576 /dev/zero | tr '\\000' A
    printf '\\n\\033[31mNGINX_CONFIG_TEST_FAILURE\\033[0m\\rCARRIAGE_RETURN_REMOVED\\n'
  } >"${OPENREWARD_NGINX_LOG_DIR}/error.log"
  exit 1
fi
exit 99
""",
    )

    result = subprocess.run(
        [str(ENTRYPOINT)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "OPENREWARD_RUNTIME_MODE": "user",
            "OPENREWARD_STATE_DIR": str(state),
            "OPENREWARD_NGINX_LOG_DIR": str(state / "nginx/log"),
            "OPENREWARD_ENTRYPOINT_SKIP_POSTGRES": "1",
            "OPENREWARD_WORKER_PID_ISOLATION": "0",
            "OPENREWARD_SERVER_PYTHON": str(worker),
            "OPENREWARD_SERVER_APP": "unused",
            "OPENREWARD_NGINX_BIN": str(nginx),
            "OPENREWARD_WORKERS": "1",
            "OPENREWARD_PORT": "19998",
            "OPENREWARD_WORKER_BASE_PORT": "19900",
            "OPENREWARD_SERVER_SHUTDOWN_GRACE_SECS": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert "nginx configuration test failed" in result.stderr
    assert "last 100 lines, at most 16384 bytes" in result.stderr
    assert "NGINX_CONFIG_TEST_FAILURE" in result.stderr
    assert "OLDEST_LOG_CONTENT_MUST_BE_TRUNCATED" not in result.stderr
    assert "\x1b" not in result.stderr
    assert "\r" not in result.stderr
    assert len(result.stderr.encode()) < 18_000
    assert "[entrypoint] shutting down" in result.stdout


def test_abci_smoke_uses_fakeroot_only_to_stage_seed() -> None:
    subprocess.run(["bash", "-n", str(PBS_SMOKE)], check=True)
    script = PBS_SMOKE.read_text()

    assert PBS_SMOKE.stat().st_mode & 0o111
    assert "#PBS -q R9920261300" in script
    assert "#PBS -v RTYPE=rt_HF" in script
    assert "#PBS -l select=1:mpiprocs=8" in script
    assert script.count("--fakeroot") == 1
    assert script.index("--fakeroot") < script.index("--overlay")
    service_launch = script[script.index('echo "[smoke] starting non-fakeroot'):]
    assert "--fakeroot" not in service_launch
    assert "OPENREWARD_RUNTIME_MODE=user" in service_launch
    assert "OPENREWARD_WORKER_PID_ISOLATION=0" in service_launch
    assert "OPENREWARD_WORKERS=2" in service_launch
    assert "OPENREWARD_WORKER_PID_DIR=/mnt/state/worker-pids" in service_launch
    assert "toolathlon_session_smoke.py" in service_launch
    assert "toolathlon_worker_recovery_smoke.py" in service_launch
    assert "OPENREWARD_SMOKE_HEALTH_SECONDS:-1800" in script
    assert "health window passed" in service_launch
    assert 'test "${psql_identity}" = eigent:postgres' in service_launch
    assert "toolathlon_singularity_user_mode_smoke=ok" in service_launch
    for secret in ("WANDB_API_KEY", "HF_TOKEN", "OPENAI_API_KEY", "LITELLM_API_KEY"):
        assert secret not in script


def test_session_smoke_has_concurrency_isolation_and_reward_oracles() -> None:
    source = SESSION_SMOKE.read_text()

    assert "ThreadPoolExecutor(max_workers=2)" in source
    assert "Barrier(2)" in source
    assert '"isolated-a"' in source and '"isolated-b"' in source
    assert "Canvas_Announcements.xlsx" in source
    assert "expected_reward = 1.0 if install_gold else 0.0" in source
    assert 'session.call_tool("claim_done", {})' in source


def test_worker_recovery_smoke_uses_exact_pid_and_preserves_peer_session() -> None:
    source = RECOVERY_SMOKE.read_text()

    assert "OPENREWARD_KILL_WORKER_PID_FILE" in source
    assert "os.kill(old_pid, signal.SIGKILL)" in source
    assert "candidate != old_pid" in source
    assert "unrelated-worker-session-survived" in source
    assert '"unrelated_session_preserved": True' in source
