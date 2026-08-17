from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = REPO_ROOT / "pbs-scripts/openreward-node-memory-guard.py"
NODE_SERVICES_PATH = REPO_ROOT / "pbs-scripts/openreward-node-services.sh"
COMMON_PATH = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"


def _load_guard_module():
    spec = importlib.util.spec_from_file_location("openreward_node_memory_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


guard = _load_guard_module()


def _settings():
    return guard.Settings(
        node_limit_bytes=guard.EXPECTED_NODE_LIMIT_BYTES,
        trigger_percent=guard.EXPECTED_TRIGGER_PERCENT,
        fallback_available_bytes=guard.EXPECTED_FALLBACK_AVAILABLE_BYTES,
        poll_seconds=guard.EXPECTED_POLL_SECONDS,
        telemetry_seconds=guard.EXPECTED_TELEMETRY_SECONDS,
    )


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="ascii")


def test_exact_contract_constants_and_threshold() -> None:
    settings = _settings()

    settings.validate_exact_contract()
    assert settings.node_limit_bytes == 2_061_584_302_080
    assert settings.cgroup_trigger_bytes == 1_752_346_656_768
    assert settings.fallback_available_bytes == 274_877_906_944
    assert settings.poll_seconds == 2
    assert settings.telemetry_seconds == 60

    with pytest.raises(guard.GuardFailure, match="exact PBS service-node contract"):
        guard.Settings(
            node_limit_bytes=settings.node_limit_bytes,
            trigger_percent=84,
            fallback_available_bytes=settings.fallback_available_bytes,
            poll_seconds=settings.poll_seconds,
            telemetry_seconds=settings.telemetry_seconds,
        ).validate_exact_contract()


def test_selects_highest_aggregate_ancestor_with_exact_finite_limit(
    tmp_path: Path,
) -> None:
    settings = _settings()
    proc_self_cgroup = tmp_path / "proc/self/cgroup"
    cgroup_root = tmp_path / "sys/fs/cgroup"
    meminfo = tmp_path / "proc/meminfo"
    _write(proc_self_cgroup, "0::/pbs/job-42/task-7\n")
    _write(cgroup_root / "memory.max", "max\n")
    _write(cgroup_root / "pbs/memory.max", "max\n")
    _write(cgroup_root / "pbs/job-42/memory.max", f"{settings.node_limit_bytes}\n")
    _write(cgroup_root / "pbs/job-42/memory.current", "123\n")
    _write(cgroup_root / "pbs/job-42/task-7/memory.max", f"{settings.node_limit_bytes}\n")
    _write(cgroup_root / "pbs/job-42/task-7/memory.current", "12\n")

    source = guard.resolve_memory_source(
        settings,
        proc_self_cgroup=proc_self_cgroup,
        cgroup_root=cgroup_root,
        proc_meminfo=meminfo,
    )

    assert isinstance(source, guard.CgroupMemorySource)
    assert source.cgroup_dir == cgroup_root / "pbs/job-42"
    assert source.sample().current_bytes == 123


def test_falls_back_to_explicit_memavailable_when_no_exact_cgroup_limit(
    tmp_path: Path,
) -> None:
    settings = _settings()
    proc_self_cgroup = tmp_path / "proc/self/cgroup"
    cgroup_root = tmp_path / "sys/fs/cgroup"
    meminfo = tmp_path / "proc/meminfo"
    _write(proc_self_cgroup, "0::/pbs/job-42/task-7\n")
    _write(cgroup_root / "memory.max", "max\n")
    _write(cgroup_root / "pbs/job-42/task-7/memory.max", "1073741824\n")
    _write(meminfo, "MemTotal: 1 kB\nMemAvailable: 268435456 kB\n")

    source = guard.resolve_memory_source(
        settings,
        proc_self_cgroup=proc_self_cgroup,
        cgroup_root=cgroup_root,
        proc_meminfo=meminfo,
    )

    assert isinstance(source, guard.MeminfoMemorySource)
    reading = source.sample()
    assert reading.available_bytes == settings.fallback_available_bytes
    assert reading.triggered is False

    _write(meminfo, "MemAvailable: 268435455 kB\n")
    assert source.sample().triggered is True


def test_valid_nonunified_membership_uses_meminfo_fallback(tmp_path: Path) -> None:
    settings = _settings()
    proc_self_cgroup = tmp_path / "proc/self/cgroup"
    meminfo = tmp_path / "proc/meminfo"
    _write(proc_self_cgroup, "7:memory:/pbs/job-42\n")
    _write(meminfo, "MemAvailable: 300000000 kB\n")

    source = guard.resolve_memory_source(
        settings,
        proc_self_cgroup=proc_self_cgroup,
        cgroup_root=tmp_path / "sys/fs/cgroup",
        proc_meminfo=meminfo,
    )

    assert isinstance(source, guard.MeminfoMemorySource)
    assert source.sample().triggered is False


@pytest.mark.parametrize(
    "membership",
    (
        "malformed\n",
        "0:memory:/pbs/job-42\n",
        "0::relative/path\n",
        "0::/pbs/job-42\n0::/pbs/job-43\n",
    ),
)
def test_rejects_malformed_or_ambiguous_cgroup_membership(tmp_path: Path, membership: str) -> None:
    proc_self_cgroup = tmp_path / "proc/self/cgroup"
    _write(proc_self_cgroup, membership)

    with pytest.raises(guard.GuardFailure):
        guard.read_unified_cgroup_path(proc_self_cgroup)


def test_malformed_finite_memory_max_fails_instead_of_falling_back(
    tmp_path: Path,
) -> None:
    settings = _settings()
    proc_self_cgroup = tmp_path / "proc/self/cgroup"
    cgroup_root = tmp_path / "sys/fs/cgroup"
    _write(proc_self_cgroup, "0::/pbs/job-42\n")
    _write(cgroup_root / "pbs/job-42/memory.max", "not-a-limit\n")

    with pytest.raises(guard.GuardFailure, match="nonnegative integer"):
        guard.resolve_memory_source(
            settings,
            proc_self_cgroup=proc_self_cgroup,
            cgroup_root=cgroup_root,
            proc_meminfo=tmp_path / "proc/meminfo",
        )


def test_cgroup_boundary_and_post_start_contract_changes_fail_closed(
    tmp_path: Path,
) -> None:
    settings = _settings()
    cgroup_dir = tmp_path / "job"
    _write(cgroup_dir / "memory.max", f"{settings.node_limit_bytes}\n")
    _write(cgroup_dir / "memory.current", f"{settings.cgroup_trigger_bytes - 1}\n")
    source = guard.CgroupMemorySource(cgroup_dir, settings)

    assert source.sample().triggered is False
    _write(cgroup_dir / "memory.current", f"{settings.cgroup_trigger_bytes}\n")
    assert source.sample().triggered is True

    _write(cgroup_dir / "memory.max", "max\n")
    with pytest.raises(guard.GuardFailure, match="changed from finite to unlimited"):
        source.sample()

    _write(cgroup_dir / "memory.max", f"{settings.node_limit_bytes}\n")
    (cgroup_dir / "memory.current").unlink()
    with pytest.raises(guard.GuardFailure, match="could not read"):
        source.sample()


@pytest.mark.parametrize(
    "contents",
    (
        "MemFree: 999999999 kB\n",
        "MemAvailable: unknown kB\n",
        "MemAvailable: 300000000 MB\n",
        "MemAvailable: 300000000 kB\nMemAvailable: 300000001 kB\n",
    ),
)
def test_meminfo_fallback_rejects_missing_malformed_or_duplicate_value(tmp_path: Path, contents: str) -> None:
    meminfo = tmp_path / "proc/meminfo"
    _write(meminfo, contents)
    source = guard.MeminfoMemorySource(meminfo, _settings())

    with pytest.raises(guard.GuardFailure, match="exactly one"):
        source.sample()


def test_runtime_read_failure_after_start_returns_nonzero_and_uses_two_second_poll() -> None:
    settings = _settings()

    class FailingAfterStart:
        description = "mode=test"

        def __init__(self) -> None:
            self.calls = 0

        def sample(self):
            self.calls += 1
            if self.calls == 1:
                return guard.Reading(mode="meminfo", triggered=False, available_bytes=300)
            raise guard.GuardFailure("injected post-start read failure")

    source = FailingAfterStart()
    waits = []
    output = io.StringIO()
    error = io.StringIO()
    status = guard.run_monitor(
        source,
        settings,
        threading.Event(),
        output=output,
        error=error,
        monotonic=lambda: 0.0,
        wait_for_stop=lambda seconds: waits.append(seconds) or False,
    )

    assert status == guard.EXIT_RUNTIME_FAILURE
    assert waits == [2]
    assert "event=start mode=test" in output.getvalue()
    assert "injected post-start read failure" in error.getvalue()


def test_threshold_exit_is_nonzero_and_periodic_telemetry_is_sixty_seconds() -> None:
    settings = _settings()

    class StaticSource:
        description = "mode=test"

        def __init__(self, triggered: bool) -> None:
            self.triggered = triggered

        def sample(self):
            return guard.Reading(
                mode="cgroup",
                triggered=self.triggered,
                current_bytes=settings.cgroup_trigger_bytes,
                limit_bytes=settings.node_limit_bytes,
                trigger_bytes=settings.cgroup_trigger_bytes,
            )

    threshold_error = io.StringIO()
    threshold_status = guard.run_monitor(
        StaticSource(True),
        settings,
        threading.Event(),
        output=io.StringIO(),
        error=threshold_error,
        monotonic=lambda: 0.0,
    )
    assert threshold_status == guard.EXIT_THRESHOLD
    assert "event=threshold" in threshold_error.getvalue()

    stop = threading.Event()
    wait_calls = 0

    def wait_once_then_stop(seconds: float) -> bool:
        nonlocal wait_calls
        assert seconds == 2
        wait_calls += 1
        if wait_calls == 2:
            stop.set()
            return True
        return False

    clock_values = iter((0.0, 60.0))
    telemetry_output = io.StringIO()
    telemetry_status = guard.run_monitor(
        StaticSource(False),
        settings,
        stop,
        output=telemetry_output,
        error=io.StringIO(),
        monotonic=lambda: next(clock_values),
        wait_for_stop=wait_once_then_stop,
    )
    assert telemetry_status == 0
    assert telemetry_output.getvalue().count("event=telemetry") == 1
    assert "event=stop reason=signal" in telemetry_output.getvalue()


def test_node_supervisor_and_common_launcher_wire_guard_fail_closed() -> None:
    subprocess.run(["bash", "-n", str(NODE_SERVICES_PATH)], check=True)
    subprocess.run(["bash", "-n", str(COMMON_PATH)], check=True)
    assert GUARD_PATH.stat().st_mode & 0o111

    node_services = NODE_SERVICES_PATH.read_text()
    common = COMMON_PATH.read_text()
    exact_knobs = {
        "OPENREWARD_NODE_MEMORY_GUARD_ENABLED": "1",
        "OPENREWARD_NODE_MEMORY_LIMIT_BYTES": "2061584302080",
        "OPENREWARD_NODE_MEMORY_TRIGGER_PERCENT": "85",
        "OPENREWARD_NODE_MEMORY_FALLBACK_AVAILABLE_BYTES": "274877906944",
        "OPENREWARD_NODE_MEMORY_POLL_SECONDS": "2",
        "OPENREWARD_NODE_MEMORY_TELEMETRY_SECONDS": "60",
    }
    for name, value in exact_knobs.items():
        assert f"${{{name}:-{value}}}" in common
        assert f'"{name}=${{NODE_' in common
        assert f"${{{name}:-{value}}}" in node_services

    launch_start = common.index('echo "[pbs-openreward] starting one Toolathlon and one SWE service')
    launch_end = common.index("\nHEALTH_CONNECT_TIMEOUT_SECONDS=", launch_start)
    launch_body = common[launch_start:launch_end]
    for name in exact_knobs:
        assert f'"{name}=' in launch_body

    assert "local -a child_names=(toolathlon swe-rebench)" in node_services
    assert 'local -a child_pids=("${toolathlon_pid}" "${swe_pid}")' in node_services
    assert 'kill -0 "${memory_guard_pid}"' in node_services
    assert 'wait "${memory_guard_pid}"' in node_services
    assert "cleanup-guard" in node_services
    assert "memory guard was already dead before natural cleanup" in node_services
    assert '"${primary_status}" -eq 0 && "${guard_wait_status}" != 0' in node_services
    assert 'OPENREWARD_WORKERS="${OPENREWARD_WORKERS:-8}"' in node_services
    assert "export OPENREWARD_WORKERS=${OPENREWARD_WORKERS:-8}" in common
    assert "TOOLATHLON_WORKER_QUORUM=${OPENREWARD_TOOLATHLON_WORKER_QUORUM:-5}" in common
