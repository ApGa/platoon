#!/usr/bin/env python3
"""Rootless exact-command startup probe for Toolathlon's baked Python MCPs."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

CONFIGS = Path("/app/configs/mcp_servers")
CATALOG = Path("/app/tool_schemas.json")
MANIFEST = Path("/usr/local/share/openreward/toolathlon-pbs-portable.json")
LOCAL_SERVERS = "/opt/local_servers"
POSTGRES_BIN = Path("/usr/lib/postgresql/14/bin")
PRODUCTION_PATH = "/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
PRODUCTION_PYTHONPATH = "/app/runtime_patches"
PROTOCOL_VERSION = "2024-11-05"
STABILITY_SECONDS = 0.25
RESERVED_ENVIRONMENT = {
    "HOME",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONPATH",
    "TMPDIR",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "VIRTUAL_ENV",
    "XDG_CACHE_HOME",
}
PYTHON_CONFIGS = {
    "arxiv-latex": "arxiv-latex-mcp.yaml",
    "arxiv_local": "arxiv_local.yaml",
    "emails": "emails.yaml",
    "excel": "excel.yaml",
    "google_sheet": "google_sheet.yaml",
    "pdf-tools": "pdf-tools.yaml",
    "pptx": "pptx.yaml",
    "scholarly": "scholarly_search.yaml",
    "snowflake": "snowflake.yaml",
    "terminal": "terminal.yaml",
    "word": "word.yaml",
    "yahoo-finance": "yahoo-finance.yaml",
    "youtube-transcript": "youtube_transcript.yaml",
}


class PostgresState:
    __slots__ = (
        "cluster",
        "log",
        "environment",
        "process",
        "minimum_start_ticks",
        "start_ticks",
    )

    def __init__(
        self,
        cluster: Path,
        log: Path,
        environment: dict[str, str],
        process: subprocess.Popen[bytes],
        minimum_start_ticks: int,
    ) -> None:
        self.cluster = cluster
        self.log = log
        self.environment = environment
        self.process = process
        self.minimum_start_ticks = minimum_start_ticks
        self.start_ticks: int | None = None


ACTIVE_PROCESS_GROUPS: set[int] = set()
ACTIVE_POSTGRES: PostgresState | None = None


class ProbeInterrupted(RuntimeError):
    """Raised by TERM/INT so normal finally blocks clean every child."""


def require_cleanup_signals_unblocked() -> None:
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    forbidden = blocked.intersection({signal.SIGTERM, signal.SIGINT})
    if forbidden:
        raise RuntimeError(f"Python MCP probe inherited blocked cleanup signals: {sorted(map(int, forbidden))}")


def interrupt_probe(signum: int, _frame: Any) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise ProbeInterrupted(f"Python MCP portability probe interrupted by signal {signum}")


def emergency_cleanup() -> None:
    """Last-resort cleanup if normal async/context teardown cannot complete."""

    global ACTIVE_POSTGRES
    errors: list[str] = []
    if ACTIVE_POSTGRES is not None:
        state = ACTIVE_POSTGRES
        try:
            stop_private_postgres(state, require_running=False)
        except BaseException as exc:
            errors.append(f"PostgreSQL emergency cleanup failed: {exc}")
        pid_absent = not (state.cluster / "postmaster.pid").exists()
        group_absent = sync_process_group_absent(state.process.pid, timeout=0)
        if pid_absent and group_absent:
            ACTIVE_POSTGRES = None

    for process_group in tuple(ACTIVE_PROCESS_GROUPS):
        try:
            forced_kill = terminate_sync_process_group(process_group)
            if forced_kill:
                errors.append(f"process group {process_group} required SIGKILL during emergency cleanup")
        except BaseException as exc:
            errors.append(f"process-group emergency cleanup failed: {exc}")
    if ACTIVE_PROCESS_GROUPS:
        errors.append(f"process groups remain after emergency cleanup: {sorted(ACTIVE_PROCESS_GROUPS)}")
    if ACTIVE_POSTGRES is not None:
        errors.append("PostgreSQL state remains active after emergency cleanup")
    if errors:
        os.write(2, ("; ".join(errors) + "\n").encode(errors="replace"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: Any, workspace: Path) -> Any:
    if not isinstance(value, str):
        return value
    return value.replace("${local_servers_paths}", LOCAL_SERVERS).replace("${agent_workspace}", str(workspace))


def project_from_command(command: str) -> Path:
    path = Path(command)
    try:
        relative = path.relative_to(LOCAL_SERVERS)
    except ValueError as exc:
        raise RuntimeError(f"Python MCP command is outside the baked server tree: {command}") from exc
    if len(relative.parts) < 4 or relative.parts[1:3] != (".venv", "bin"):
        raise RuntimeError(f"Python MCP command is not a baked venv launcher: {command}")
    project = Path(LOCAL_SERVERS) / relative.parts[0]
    if not project.is_dir() or project.is_symlink():
        raise RuntimeError(f"Python MCP project is missing or indirect: {project}")
    return project


def tree_metadata(projects: set[Path]) -> str:
    """Identity every project entry without reading package payload contents."""

    digest = hashlib.sha256()
    for project in sorted(projects):
        for path in [project, *sorted(project.rglob("*"))]:
            value = path.lstat()
            relative = path.relative_to(project).as_posix() if path != project else "."
            if stat.S_ISLNK(value.st_mode):
                kind = "L"
                extra = os.readlink(path)
            elif stat.S_ISDIR(value.st_mode):
                kind = "D"
                extra = ""
            elif stat.S_ISREG(value.st_mode):
                kind = "F"
                extra = ""
            else:
                kind = "O"
                extra = ""
            record = (
                f"{project.name}\t{relative}\t{kind}\t{stat.S_IMODE(value.st_mode):04o}\t"
                f"{value.st_uid}\t{value.st_gid}\t{value.st_size}\t{value.st_mtime_ns}\t"
                f"{value.st_ctime_ns}\t{value.st_nlink}\t{extra}\n"
            )
            digest.update(record.encode())
    return digest.hexdigest()


async def response(
    process: asyncio.subprocess.Process,
    request_id: int,
    method: str,
    params: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("MCP subprocess pipes are unavailable")
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    process.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode())
    await process.stdin.drain()
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"{method} timed out after {timeout:.0f}s")
        line = await asyncio.wait_for(process.stdout.readline(), timeout=remaining)
        if not line:
            returncode = await process.wait()
            raise RuntimeError(f"MCP exited before {method} response: rc={returncode}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("id") != request_id:
            continue
        if value.get("jsonrpc") != "2.0":
            raise RuntimeError(f"{method} response has an invalid JSON-RPC version")
        if "error" in value:
            raise RuntimeError(f"{method} response contains an error: {value['error']!r}")
        if not isinstance(value.get("result"), dict):
            raise RuntimeError(f"{method} response has no result object")
        return value


async def spawn_registered_process(
    command: str,
    arguments: list[str],
    *,
    stderr: Any,
    cwd: Path,
    environment: dict[str, str],
) -> asyncio.subprocess.Process:
    """Atomically register an isolated child without leaking a blocked signal mask."""

    process: asyncio.subprocess.Process | None = None
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        {signal.SIGTERM, signal.SIGINT},
    )

    def restore_child_signal_mask() -> None:
        # The probe is single-threaded at fork. Restore the exact pre-block
        # mask in the child immediately before exec; otherwise TERM cleanup is
        # inherited as blocked even though the child starts a new session.
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=stderr,
            cwd=cwd,
            env=environment,
            limit=8 * 1024 * 1024,
            start_new_session=True,
            preexec_fn=restore_child_signal_mask,
        )
        ACTIVE_PROCESS_GROUPS.add(process.pid)
    finally:
        try:
            # A pending TERM/INT is delivered only after the new process group
            # has entered the cleanup registry.
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException:
            # The caller cannot receive ``process`` if signal delivery raises
            # during unmasking, so this helper must close the registered group.
            if process is not None:
                await close_process("spawn-interrupted", process, protocol_succeeded=False)
            raise
    if process is None:
        raise RuntimeError("MCP subprocess creation returned no process")
    return process


async def require_stable_before_eof(
    name: str,
    process: asyncio.subprocess.Process,
    *,
    duration: float = STABILITY_SECONDS,
) -> None:
    """Reject a server that self-exits after replying but before client EOF."""

    await asyncio.sleep(duration)
    if process.returncode is not None:
        raise RuntimeError(f"{name}: MCP self-exited before client EOF: rc={process.returncode}")


async def process_group_absent(process_group: int, *, timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify absence of MCP process group {process_group}") from exc
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.05)


async def terminate_process_group(
    process: asyncio.subprocess.Process,
    process_group: int,
    *,
    term_wait: float = 5,
    term_group_wait: float = 1,
    kill_wait: float = 5,
    kill_group_wait: float = 2,
) -> tuple[int, bool]:
    """Boundedly terminate the whole isolated group and report SIGKILL use."""

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        returncode = await asyncio.wait_for(process.wait(), timeout=term_wait)
    except asyncio.TimeoutError:
        returncode = process.returncode
    if await process_group_absent(process_group, timeout=term_group_wait):
        return (returncode if returncode is not None else process.returncode), False

    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        returncode = await asyncio.wait_for(process.wait(), timeout=kill_wait)
    except asyncio.TimeoutError as exc:
        raise RuntimeError("MCP process leader survived process-group SIGKILL") from exc
    if not await process_group_absent(process_group, timeout=kill_group_wait):
        raise RuntimeError("MCP process group survived process-group SIGKILL")
    return returncode, True


async def close_process(
    name: str,
    process: asyncio.subprocess.Process,
    *,
    protocol_succeeded: bool,
) -> dict[str, Any]:
    """Close via EOF, then clean the isolated group under a pinned contract."""

    process_group = process.pid
    exited_on_eof = process.returncode is not None
    if exited_on_eof and protocol_succeeded:
        group_absent = await process_group_absent(process_group, timeout=1)
        if not group_absent:
            _, forced_kill = await terminate_process_group(process, process_group)
            ACTIVE_PROCESS_GROUPS.discard(process_group)
            if forced_kill:
                raise RuntimeError(f"{name}: self-exited leader left descendants requiring SIGKILL")
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        raise RuntimeError(f"{name}: MCP self-exited before client EOF: rc={process.returncode}")
    if not exited_on_eof:
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
            await process.stdin.wait_closed()
            # Give the subprocess transport one event-loop turn to observe
            # pipe EOF before waiting on the child watcher (needed by 3.9 too).
            await asyncio.sleep(0.05)
        try:
            await asyncio.wait_for(process.wait(), timeout=6)
            exited_on_eof = True
        except asyncio.TimeoutError:
            exited_on_eof = False

    if exited_on_eof:
        returncode = process.returncode
        group_absent = await process_group_absent(process_group, timeout=1)
        if not group_absent:
            _, forced_kill = await terminate_process_group(process, process_group)
            ACTIVE_PROCESS_GROUPS.discard(process_group)
            if forced_kill:
                raise RuntimeError("orphaned MCP process group required SIGKILL after EOF")
            raise RuntimeError("orphaned MCP process group remained after its leader exited")
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        if protocol_succeeded:
            expected_returncode = 0
            if returncode != expected_returncode:
                raise RuntimeError(f"unexpected MCP EOF return code: rc={returncode}, expected={expected_returncode}")
        return {"method": "eof", "returncode": returncode, "process_group_absent": True}

    returncode, forced_kill = await terminate_process_group(process, process_group)
    ACTIVE_PROCESS_GROUPS.discard(process_group)
    if forced_kill:
        raise RuntimeError("MCP process group required forbidden SIGKILL teardown")
    if protocol_succeeded:
        raise RuntimeError(f"{name} ignored EOF and required process-group SIGTERM")
    return {"method": "sigterm", "returncode": returncode, "process_group_absent": True}


async def probe_one(
    name: str,
    filename: str,
    workspace: Path,
    expected_launcher: dict[str, str],
    expected_tools: list[dict[str, Any]],
    postgres_env: dict[str, str],
) -> dict[str, Any]:
    config_path = CONFIGS / filename
    if sha256(config_path) != expected_launcher["config_sha256"]:
        raise RuntimeError(f"{name}: configured launch digest differs from manifest")
    value = yaml.safe_load(config_path.read_text())
    if not isinstance(value, dict) or value.get("name") != name:
        raise RuntimeError(f"{name}: malformed or mismatched MCP config")
    params = value.get("params")
    if not isinstance(params, dict):
        raise RuntimeError(f"{name}: MCP config params are absent")
    command = resolve(params.get("command"), workspace)
    arguments = [resolve(item, workspace) for item in params.get("args", [])]
    if command != expected_launcher["config_command"] or Path(command).name == "uv":
        raise RuntimeError(f"{name}: command is not the exact immutable launcher: {command!r}")
    project = project_from_command(expected_launcher["exec_command"])
    if str(project) != expected_launcher["project_cwd"]:
        raise RuntimeError(f"{name}: manifest project cwd does not match its baked venv")
    if name == "youtube-transcript":
        if command != expected_launcher["exec_command"]:
            raise RuntimeError("YouTube transcript MCP must execute its baked Python directly")
    elif not arguments or arguments[0] != name:
        raise RuntimeError(f"{name}: whitelist launcher key is absent or reordered")
    cwd = Path(resolve(params.get("cwd", str(workspace)), workspace))
    if cwd == workspace or workspace in cwd.parents:
        cwd.mkdir(parents=True, exist_ok=True)
    elif not cwd.is_dir():
        raise RuntimeError(f"{name}: immutable configured cwd is absent: {cwd}")

    configured_env = params.get("env", {})
    if not isinstance(configured_env, dict):
        raise RuntimeError(f"{name}: configured environment is malformed")
    reserved = RESERVED_ENVIRONMENT.intersection(str(key) for key in configured_env)
    if reserved:
        raise RuntimeError(f"{name}: YAML overrides a probe-controlled environment key: {sorted(reserved)}")
    child_home = workspace / ".home"
    child_tmp = workspace / ".tmp"
    child_cache = workspace / ".cache"
    child_uv_cache = workspace / ".uv-cache"
    child_home.mkdir(mode=0o700)
    child_tmp.mkdir(mode=0o700)
    child_cache.mkdir(mode=0o700)
    child_uv_cache.mkdir(mode=0o700)
    environment = {
        "PATH": PRODUCTION_PATH,
        "HOME": str(child_home),
        "TMPDIR": str(child_tmp),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": PRODUCTION_PYTHONPATH,
        "XDG_CACHE_HOME": str(child_cache),
        "UV_CACHE_DIR": str(child_uv_cache),
        "UV_PYTHON_DOWNLOADS": "never",
    }
    environment.update({str(key): str(resolve(item, workspace)) for key, item in configured_env.items()})
    # The production bridge applies its per-session database environment after
    # YAML values.  Mirror that precedence with a private Unix-socket fixture.
    environment.update(postgres_env)
    stderr_path = workspace / f"{name}.stderr"
    with stderr_path.open("wb") as stderr:
        process: asyncio.subprocess.Process | None = None
        try:
            process = await spawn_registered_process(
                command,
                arguments,
                stderr=stderr,
                cwd=cwd,
                environment=environment,
            )
            initialized = await response(
                process,
                1,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "openreward-portability-probe", "version": "1.0"},
                },
                timeout=30,
            )
            initialize_result = initialized["result"]
            server_info = initialize_result.get("serverInfo")
            if (
                initialize_result.get("protocolVersion") != PROTOCOL_VERSION
                or not isinstance(initialize_result.get("capabilities"), dict)
                or not isinstance(server_info, dict)
                or not isinstance(server_info.get("name"), str)
                or not server_info["name"]
                or not isinstance(server_info.get("version"), str)
                or not server_info["version"]
            ):
                raise RuntimeError(f"{name}: initialize result has an invalid protocol/capability/server shape")
            if process.stdin is None:
                raise RuntimeError(f"{name}: MCP stdin disappeared after initialize")
            notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            process.stdin.write((json.dumps(notification, separators=(",", ":")) + "\n").encode())
            await process.stdin.drain()
            listed = await response(process, 2, "tools/list", {}, timeout=30)
            tools = listed["result"].get("tools")
            if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
                raise RuntimeError(f"{name}: tools/list result has no tool object list")
            live_names = [tool.get("name") for tool in tools]
            expected_names = [tool.get("name") for tool in expected_tools]
            if (
                not all(isinstance(item, str) and item for item in live_names)
                or not all(isinstance(item, str) and item for item in expected_names)
                or len(live_names) != len(set(live_names))
                or len(expected_names) != len(set(expected_names))
                or set(live_names) != set(expected_names)
            ):
                raise RuntimeError(
                    f"{name}: tools/list parity failed: "
                    f"missing={sorted(set(expected_names) - set(live_names))}, "
                    f"unexpected={sorted(set(live_names) - set(expected_names))}"
                )
            await require_stable_before_eof(name, process)
        except BaseException as exc:
            if process is not None:
                try:
                    await close_process(name, process, protocol_succeeded=False)
                except BaseException as cleanup_exc:
                    stderr.flush()
                    detail = stderr_path.read_text(errors="replace")[-4000:]
                    raise RuntimeError(f"{name}: {exc}; cleanup failed: {cleanup_exc}; stderr={detail!r}") from exc
            stderr.flush()
            detail = stderr_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"{name}: {exc}; stderr={detail!r}") from exc
        if process is None:
            raise RuntimeError(f"{name}: MCP subprocess creation returned no process")
        teardown = await close_process(name, process, protocol_succeeded=True)
        teardown["alive_before_eof"] = True
    return {"name": name, "tool_count": len(expected_names), "teardown": teardown}


def sync_process_group_absent(process_group: int, *, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        except PermissionError as exc:
            raise RuntimeError(f"cannot verify absence of helper process group {process_group}") from exc
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def terminate_sync_process_group(process_group: int) -> bool:
    """Terminate and verify an isolated helper group; return whether KILL was used."""

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        return False
    if sync_process_group_absent(process_group, timeout=5):
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        return False
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        return False
    if not sync_process_group_absent(process_group, timeout=2):
        raise RuntimeError(f"helper process group {process_group} survived SIGKILL")
    ACTIVE_PROCESS_GROUPS.discard(process_group)
    return True


def terminate_sync_subprocess(process: subprocess.Popen[str]) -> bool:
    """Terminate a registered helper while also reaping its direct leader."""

    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if sync_process_group_absent(process_group, timeout=1):
        ACTIVE_PROCESS_GROUPS.discard(process_group)
        return False
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("helper process leader survived process-group SIGKILL") from exc
    if not sync_process_group_absent(process_group, timeout=2):
        raise RuntimeError("helper process group survived process-group SIGKILL")
    ACTIVE_PROCESS_GROUPS.discard(process_group)
    return True


def spawn_registered_sync(
    command: list[str],
    *,
    environment: dict[str, str],
    stdin: Any = subprocess.DEVNULL,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    text: bool = True,
    registered_callback: Callable[[subprocess.Popen[Any]], None] | None = None,
) -> subprocess.Popen[Any]:
    """Atomically spawn and register a synchronous isolated-session helper."""

    process: subprocess.Popen[Any] | None = None
    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        {signal.SIGTERM, signal.SIGINT},
    )

    def restore_child_signal_mask() -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    try:
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=text,
            env=environment,
            start_new_session=True,
            preexec_fn=restore_child_signal_mask,
        )
        ACTIVE_PROCESS_GROUPS.add(process.pid)
        if registered_callback is not None:
            registered_callback(process)
    except BaseException:
        if process is not None:
            terminate_sync_subprocess(process)
        raise
    finally:
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException:
            if process is not None:
                terminate_sync_subprocess(process)
            raise
    if process is None:
        raise RuntimeError("helper subprocess creation returned no process")
    return process


def run_checked(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout: int = 60,
) -> int:
    """Run a helper in a registered session and fail closed on all descendants."""

    process = spawn_registered_sync(command, environment=environment)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException as exc:
        forced_kill = terminate_sync_subprocess(process)
        if isinstance(exc, subprocess.TimeoutExpired):
            raise RuntimeError(f"command timed out after {timeout}s (forced_kill={forced_kill}): {command!r}") from exc
        raise

    if process.returncode != 0:
        forced_kill = False
        if not sync_process_group_absent(process.pid, timeout=0):
            forced_kill = terminate_sync_subprocess(process)
        else:
            ACTIVE_PROCESS_GROUPS.discard(process.pid)
        raise RuntimeError(
            f"command failed rc={process.returncode}, forced_kill={forced_kill}: {command!r}: {stderr[-4000:]!r}"
        )
    if not sync_process_group_absent(process.pid, timeout=1):
        forced_kill = terminate_sync_subprocess(process)
        raise RuntimeError(
            f"successful helper left a process-group descendant (forced_kill={forced_kill}): {command!r}"
        )
    ACTIVE_PROCESS_GROUPS.discard(process.pid)
    return process.pid


def boot_ticks() -> int:
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return int(uptime * ticks_per_second)


def process_start_ticks(pid: int) -> int:
    source = Path(f"/proc/{pid}/stat").read_text()
    closing_parenthesis = source.rfind(")")
    if closing_parenthesis < 0:
        raise RuntimeError(f"malformed /proc stat for PID {pid}")
    fields = source[closing_parenthesis + 2 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"missing process start ticks for PID {pid}") from exc


@contextmanager
def private_postgres(workspace: Path) -> Iterator[dict[str, str]]:
    """Yield a private Unix-socket database owned by a tracked foreground process."""

    global ACTIVE_POSTGRES
    for executable in ("initdb", "postgres", "pg_isready", "createdb"):
        path = POSTGRES_BIN / executable
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"pinned PostgreSQL fixture executable is absent: {path}")
    cluster = workspace / "postgres-data"
    socket = workspace / "postgres-socket"
    log = workspace / "postgres.log"
    fixture_home = workspace / "postgres-home"
    fixture_tmp = workspace / "postgres-tmp"
    socket.mkdir(mode=0o700)
    fixture_home.mkdir(mode=0o700)
    fixture_tmp.mkdir(mode=0o700)
    fixture_environment = {
        "PATH": f"{POSTGRES_BIN}:/usr/bin:/bin",
        "HOME": str(fixture_home),
        "TMPDIR": str(fixture_tmp),
        "LANG": "C",
        "LC_ALL": "C",
    }
    run_checked(
        [
            str(POSTGRES_BIN / "initdb"),
            "-D",
            str(cluster),
            "-U",
            "postgres",
            "-A",
            "trust",
            "--no-locale",
            "--encoding=UTF8",
        ],
        environment=fixture_environment,
        timeout=120,
    )
    command = [
        str(POSTGRES_BIN / "postgres"),
        "-D",
        str(cluster),
        "-k",
        str(socket),
        "-p",
        "5432",
        "-h",
        "",
        "-c",
        "unix_socket_permissions=0700",
        "-c",
        "fsync=off",
        "-c",
        "synchronous_commit=off",
        "-c",
        "full_page_writes=off",
    ]
    minimum_start_ticks = boot_ticks()

    def register_postgres(process: subprocess.Popen[Any]) -> None:
        global ACTIVE_POSTGRES
        ACTIVE_POSTGRES = PostgresState(
            cluster,
            log,
            fixture_environment,
            process,
            minimum_start_ticks,
        )

    with log.open("ab", buffering=0) as log_stream:
        process = spawn_registered_sync(
            command,
            environment=fixture_environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=False,
            registered_callback=register_postgres,
        )
    state = ACTIVE_POSTGRES
    if state is None or state.process is not process:
        raise RuntimeError("private PostgreSQL process was not atomically registered")
    try:
        validate_foreground_postgres(state, command)
        readiness_error = "not attempted"
        deadline = time.monotonic() + 120
        while True:
            if process.poll() is not None:
                detail = log.read_text(errors="replace")[-4000:] if log.is_file() else ""
                raise RuntimeError(
                    f"private PostgreSQL exited before readiness: rc={process.returncode}, log={detail!r}"
                )
            try:
                run_checked(
                    [
                        str(POSTGRES_BIN / "pg_isready"),
                        "-h",
                        str(socket),
                        "-p",
                        "5432",
                        "-d",
                        "postgres",
                        "-U",
                        "postgres",
                    ],
                    environment=fixture_environment,
                    timeout=5,
                )
                break
            except RuntimeError as exc:
                readiness_error = str(exc)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"private PostgreSQL readiness timed out: {readiness_error}") from exc
                time.sleep(0.1)
        run_checked(
            [
                str(POSTGRES_BIN / "createdb"),
                "-h",
                str(socket),
                "-p",
                "5432",
                "-U",
                "postgres",
                "toolathlon",
            ],
            environment=fixture_environment,
        )
        yield {
            "PGHOST": str(socket),
            "PGPORT": "5432",
            "PGDATABASE": "toolathlon",
            "PGUSER": "postgres",
            "PGPASSWORD": "",
            "PG_HOST": str(socket),
            "PG_PORT": "5432",
            "PG_DATABASE": "toolathlon",
            "PG_USER": "postgres",
            "PG_PASSWORD": "",
        }
    finally:
        try:
            stop_private_postgres(state, require_running=True)
        finally:
            if sync_process_group_absent(process.pid, timeout=0) and not (cluster / "postmaster.pid").exists():
                ACTIVE_POSTGRES = None


def validate_foreground_postgres(state: PostgresState, command: list[str]) -> None:
    process = state.process
    if process.pid <= 1 or process.poll() is not None:
        raise RuntimeError("private PostgreSQL foreground leader is absent or unsafe")
    try:
        process_group = os.getpgid(process.pid)
        session = os.getsid(process.pid)
    except ProcessLookupError as exc:
        raise RuntimeError("private PostgreSQL foreground identity disappeared") from exc
    if process_group != process.pid or session != process.pid:
        raise RuntimeError(
            f"private PostgreSQL is not its trusted session leader: "
            f"pid={process.pid}, pgid={process_group}, sid={session}"
        )
    if process_group in {os.getpid(), os.getpgrp(), os.getsid(0)}:
        raise RuntimeError("private PostgreSQL overlaps the probe process group/session")
    process_root = Path(f"/proc/{process.pid}")
    if process_root.stat().st_uid != os.geteuid():
        raise RuntimeError("private PostgreSQL is owned by a different uid")
    executable = (process_root / "exe").resolve(strict=True)
    if executable != (POSTGRES_BIN / "postgres").resolve(strict=True):
        raise RuntimeError(f"private PostgreSQL executable is unexpected: {executable}")
    if process.args != command:
        raise RuntimeError("private PostgreSQL Popen command differs from the exact fixture command")
    live_command = (process_root / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    if live_command != [item.encode() for item in command]:
        raise RuntimeError("private PostgreSQL live command differs from the exact fixture command")
    start_ticks = process_start_ticks(process.pid)
    if start_ticks < state.minimum_start_ticks:
        raise RuntimeError("private PostgreSQL predates its trusted spawn boundary")
    if state.start_ticks is None:
        state.start_ticks = start_ticks
    elif state.start_ticks != start_ticks:
        raise RuntimeError("private PostgreSQL PID start identity changed")
    pid_path = state.cluster / "postmaster.pid"
    deadline = time.monotonic() + 10
    while not pid_path.exists() and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("private PostgreSQL exited before writing postmaster.pid")
        time.sleep(0.05)
    if not pid_path.exists():
        raise RuntimeError("private PostgreSQL did not write postmaster.pid")
    metadata = pid_path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or pid_path.is_symlink() or metadata.st_uid != os.geteuid():
        raise RuntimeError("private PostgreSQL postmaster.pid is not a direct owned file")
    try:
        recorded_pid = int(pid_path.read_text().splitlines()[0])
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError("private PostgreSQL postmaster.pid is malformed") from exc
    if recorded_pid != process.pid:
        raise RuntimeError(f"private PostgreSQL PID file mismatch: {recorded_pid} != {process.pid}")


def validate_postgres_group_members(state: PostgresState) -> None:
    """Authorize only exact same-uid postgres members before fallback signals."""

    members = []
    group_leader = Path(f"/proc/{state.process.pid}")
    if group_leader.exists():
        live_start_ticks = process_start_ticks(state.process.pid)
        if state.start_ticks is None or live_start_ticks != state.start_ticks:
            raise RuntimeError("private PostgreSQL process-group leader start identity changed")
    expected_executable = (POSTGRES_BIN / "postgres").resolve(strict=True)
    for process_root in Path("/proc").iterdir():
        if not process_root.name.isdigit():
            continue
        pid = int(process_root.name)
        try:
            if os.getpgid(pid) != state.process.pid:
                continue
            if (
                os.getsid(pid) != state.process.pid
                or process_root.stat().st_uid != os.geteuid()
                or (process_root / "exe").resolve(strict=True) != expected_executable
            ):
                raise RuntimeError(f"untrusted process found in private PostgreSQL group: {pid}")
            members.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
    if not members:
        raise RuntimeError("private PostgreSQL group disappeared before signalling")


def stop_private_postgres(
    state: PostgresState,
    *,
    require_running: bool,
) -> None:
    """Fast-stop the exact foreground postmaster and prove the whole group absent."""

    process = state.process
    process_group = process.pid
    used_fallback: str | None = None
    was_running = process.poll() is None
    group_present = not sync_process_group_absent(process_group, timeout=0)
    if was_running:
        validate_foreground_postgres(state, list(process.args))
    if group_present:
        validate_postgres_group_members(state)
        try:
            os.killpg(process_group, signal.SIGINT)
        except ProcessLookupError:
            pass
    if process.returncode is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            used_fallback = "SIGTERM"
    if not sync_process_group_absent(process_group, timeout=1):
        if used_fallback is None:
            used_fallback = "SIGTERM"
        validate_postgres_group_members(state)
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if process.returncode is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    if not sync_process_group_absent(process_group, timeout=1):
        used_fallback = "SIGKILL"
        validate_postgres_group_members(state)
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.returncode is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("private PostgreSQL leader survived process-group SIGKILL") from exc
    if not sync_process_group_absent(process_group, timeout=2):
        raise RuntimeError("private PostgreSQL process group remains after shutdown")
    if process.returncode is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("private PostgreSQL leader was not reaped") from exc
    ACTIVE_PROCESS_GROUPS.discard(process_group)
    pid_path = state.cluster / "postmaster.pid"
    if pid_path.exists():
        detail = state.log.read_text(errors="replace")[-4000:] if state.log.is_file() else ""
        raise RuntimeError(f"private PostgreSQL left postmaster.pid after shutdown: {detail!r}")
    if process.returncode != 0:
        raise RuntimeError(f"private PostgreSQL fast shutdown returned rc={process.returncode}")
    if used_fallback is not None:
        raise RuntimeError(f"private PostgreSQL required forbidden {used_fallback} fallback")
    if require_running and not was_running:
        raise RuntimeError("private PostgreSQL self-exited before requested shutdown")


async def run(workspace: Path, postgres_env: dict[str, str]) -> list[dict[str, Any]]:
    if os.geteuid() == 0:
        raise RuntimeError("Python MCP portability probe must run as an unprivileged uid")
    if os.environ.get("PYTHONPATH") != PRODUCTION_PYTHONPATH:
        raise RuntimeError("Python MCP portability probe requires the exact runtime-patch PYTHONPATH")
    if "VIRTUAL_ENV" in os.environ or "UV_PROJECT_ENVIRONMENT" in os.environ:
        raise RuntimeError("Python MCP portability probe forbids inherited virtual-project state")
    outer_root = workspace.parent
    expected_outer_environment = {
        "HOME": str(outer_root),
        "TMPDIR": str(outer_root),
        "XDG_CACHE_HOME": str(outer_root / ".cache"),
        "UV_CACHE_DIR": str(outer_root / ".uv-cache"),
    }
    mismatched_outer = {
        key: (os.environ.get(key), value)
        for key, value in expected_outer_environment.items()
        if os.environ.get(key) != value
    }
    if mismatched_outer:
        raise RuntimeError(f"Python MCP probe environment mismatch: {mismatched_outer!r}")
    for value in expected_outer_environment.values():
        path = Path(value)
        metadata = path.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or path.is_symlink()
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise RuntimeError(f"Python MCP probe environment directory is not private: {path}")
    runtime_patches = Path(PRODUCTION_PYTHONPATH)
    if not runtime_patches.is_dir() or runtime_patches.is_symlink():
        raise RuntimeError("pinned Python runtime-patch directory is missing or indirect")
    manifest = json.loads(MANIFEST.read_text())
    launch = manifest.get("python_mcp_launchers")
    if not isinstance(launch, dict) or launch.get("policy") != "baked_venv_no_uv":
        raise RuntimeError("immutable Python MCP launch manifest is missing")
    mapping = launch.get("launcher_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(PYTHON_CONFIGS):
        raise RuntimeError("immutable Python MCP launcher map is incomplete")
    catalog = json.loads(CATALOG.read_text())
    projects = {project_from_command(item["exec_command"]) for item in mapping.values()}
    before = tree_metadata(projects)
    results = []
    for name, filename in PYTHON_CONFIGS.items():
        results.append(
            await probe_one(
                name,
                filename,
                workspace / name,
                mapping[name],
                catalog[name],
                postgres_env,
            )
        )
    after = tree_metadata(projects)
    if after != before:
        raise RuntimeError("a Python MCP launch mutated its immutable project or .venv tree")
    return results


def main() -> None:
    require_cleanup_signals_unblocked()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt_probe)
    signal.signal(signal.SIGINT, interrupt_probe)
    atexit.register(emergency_cleanup)
    workspace = args.workspace_root.absolute()
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError(f"probe workspace must not already exist: {workspace}")
    workspace.mkdir(mode=0o700, parents=False)
    try:
        with private_postgres(workspace) as postgres_env:
            results = asyncio.run(run(workspace, postgres_env))
    finally:
        if ACTIVE_POSTGRES is None and not ACTIVE_PROCESS_GROUPS and workspace.is_dir() and not workspace.is_symlink():
            shutil.rmtree(workspace)
    if ACTIVE_POSTGRES is not None or ACTIVE_PROCESS_GROUPS:
        raise RuntimeError(
            "Python MCP probe cleanup is incomplete: "
            f"postgres_active={ACTIVE_POSTGRES is not None}, "
            f"process_groups={sorted(ACTIVE_PROCESS_GROUPS)}"
        )
    if workspace.exists() or workspace.is_symlink():
        raise RuntimeError("Python MCP probe workspace remains after successful cleanup")
    print(
        json.dumps(
            {
                "toolathlon_python_mcp_rootless_probe": "ok",
                "server_count": len(results),
                "servers": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
