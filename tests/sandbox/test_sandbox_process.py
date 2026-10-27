from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from platoon.sandbox import Mount, SandboxSnapshot, SandboxSpec, get_provider
from platoon.sandbox import process as process_module
from platoon.sandbox._exec import run_command
from platoon.sandbox.process import SANDBOX_ROOT_ENV, PathMap, ProcessProvider, ProcessSandbox
from platoon.utils.stopping import process_start_time, registered_process_groups


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


async def _start(provider: ProcessProvider, **spec_fields) -> ProcessSandbox:
    return await provider.start(SandboxSpec(image="local", **spec_fields))


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_group_gone(pgid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while _group_alive(pgid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


async def _wait_gone(pid: int, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while _running(pid):
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.05)
    return True


async def _plant(sandbox: ProcessSandbox) -> tuple[int, int]:
    """A background process in a command's group, and one that moved to a session of its own (setsid)."""

    grouped = await sandbox.exec("echo $$; sleep 30 >/dev/null 2>&1 &", timeout_s=10)
    await sandbox.exec(
        "setsid sh -c 'echo $$ > /tmp/detached.pid; exec sleep 30' </dev/null >/dev/null 2>&1 & sleep 0.3",
        timeout_s=10,
    )
    group, detached = int(grouped.stdout.split()[0]), int((sandbox.root / "tmp" / "detached.pid").read_text())
    assert _group_alive(group) and _running(detached)
    return group, detached


def test_path_map_rewrites_sandbox_paths_at_path_boundaries():
    paths = PathMap(Path("/host/sb/fs"), ["/app", "/tests", "/tmp"])

    assert paths.rewrite("cd /app && cat /tests/test.sh > /tmp/out") == (
        "cd /host/sb/fs/app && cat /host/sb/fs/tests/test.sh > /host/sb/fs/tmp/out"
    )
    assert paths.rewrite('X="/app/a b" --out=/tmp PATH=/usr/bin:/app/bin') == (
        'X="/host/sb/fs/app/a b" --out=/host/sb/fs/tmp PATH=/usr/bin:/host/sb/fs/app/bin'
    )
    # Not sandbox paths: other directories sharing a prefix, relative and home paths, URLs, expansions.
    for text in ["/application /tmpdir /app.txt", "./app ~/app a/app", "http://host/app", "${ROOT}/app $(pwd)/tmp"]:
        assert paths.rewrite(text) == text
    assert paths.to_host("/app/../app/x") == "/host/sb/fs/app/x"
    assert paths.to_host("/usr/bin/env") == "/usr/bin/env"


def test_path_map_leaves_host_paths_below_its_root_alone():
    paths = PathMap(Path("/tmp/sandboxes/sb/fs"), ["/tmp", "/app"])

    assert paths.rewrite("ls /tmp/sandboxes/sb/fs/app /app") == "ls /tmp/sandboxes/sb/fs/app /tmp/sandboxes/sb/fs/app"
    assert paths.to_host("/tmp/sandboxes/sb/fs/app/x") == "/tmp/sandboxes/sb/fs/app/x"
    with pytest.raises(ValueError, match="other than '/'"):
        PathMap(Path("/x"), ["/"])


@pytest.mark.asyncio
async def test_exec_runs_in_the_workdir_with_translated_paths_env_and_stdin(provider: ProcessProvider):
    sandbox = await _start(provider, env={"DATA": "/app/data.txt"})
    try:
        result = await sandbox.exec('pwd; echo hello > /app/data.txt; cat "$DATA"; echo "$EXTRA"', env={"EXTRA": "x"})
        assert result.exit_code == 0 and not result.timed_out
        assert result.stdout.splitlines() == [str(sandbox.root / "app"), "hello", "x"]
        assert (sandbox.root / "app" / "data.txt").read_text() == "hello\n"

        listed = await sandbox.exec(["cat", "/app/data.txt"])
        assert listed.stdout == "hello\n"

        piped = await sandbox.exec("cat > /tmp/in.txt && wc -c < /tmp/in.txt", stdin=b"abc")
        assert piped.stdout.strip() == "3"

        failing = await sandbox.exec("echo oops >&2; exit 3", cwd="/tmp")
        assert (failing.exit_code, failing.stderr) == (3, "oops\n")

        roots = await sandbox.exec(f'echo "${SANDBOX_ROOT_ENV}" "$TMPDIR"')
        assert roots.stdout.split() == [str(sandbox.root), str(sandbox.root / "tmp")]
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_exec_timeout_kills_the_whole_process_group(provider: ProcessProvider):
    sandbox = await _start(provider)
    try:
        started = time.monotonic()
        result = await sandbox.exec("echo $$; sleep 30 & sleep 30", timeout_s=0.5)
        assert time.monotonic() - started < 5
        assert result.timed_out and result.exit_code < 0
        assert await _wait_group_gone(int(result.stdout.split()[0]))
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_background_processes_do_not_block_exec_and_die_with_the_sandbox(provider: ProcessProvider):
    sandbox = await _start(provider)
    result = await sandbox.exec("echo $$; sleep 30 &", timeout_s=10)
    pgid = int(result.stdout.split()[0])
    assert result.exit_code == 0 and not result.timed_out
    assert _group_alive(pgid)

    await sandbox.stop()

    assert await _wait_group_gone(pgid)
    assert not sandbox.root.exists()


@pytest.mark.asyncio
async def test_kill_processes_ends_what_commands_left_and_keeps_the_sandbox(provider: ProcessProvider):
    sandbox = await _start(provider)
    unrelated = subprocess.Popen(["sleep", "30"], start_new_session=True)  # this user's, not the sandbox's
    try:
        group, detached = await _plant(sandbox)

        await sandbox.kill_processes()

        assert await _wait_group_gone(group) and await _wait_gone(detached)
        assert unrelated.poll() is None
        assert (await sandbox.exec("cat /tmp/detached.pid")).stdout == f"{detached}\n"  # the files stay
    finally:
        unrelated.kill()
        unrelated.wait()
        await sandbox.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        await sandbox.kill_processes()


@pytest.mark.asyncio
async def test_kept_processes_outlive_kill_processes_until_the_sandbox_stops(provider: ProcessProvider):
    """A task's init, started by a command, and what it started (a child, a daemon in a session of its own) are spared
    by ``kill_processes`` once kept, with their group; ``stop`` ends them."""

    sandbox = await _start(provider)
    try:
        started = await sandbox.exec(
            'sh -c \'echo $$ > /tmp/init.pid; setsid sh -c "echo \\$$ > /tmp/daemon.pid; exec sleep 30" '
            "</dev/null >/dev/null 2>&1 & exec sleep 30' </dev/null >/dev/null 2>&1 & sleep 0.3",
            timeout_s=10,
        )
        assert started.exit_code == 0
        init = int((sandbox.root / "tmp" / "init.pid").read_text())
        daemon = int((sandbox.root / "tmp" / "daemon.pid").read_text())
        start = process_start_time(init)
        sandbox.keep_processes([f"{init}:{start}", f"{daemon}:{process_start_time(daemon)}"])
        group, detached = await _plant(sandbox)

        await sandbox.kill_processes()

        assert await _wait_group_gone(group) and await _wait_gone(detached)
        assert _running(init) and _running(daemon)
    finally:
        await sandbox.stop()
    assert await _wait_gone(init) and await _wait_gone(daemon)


@pytest.mark.asyncio
async def test_stop_also_ends_processes_that_left_their_group(provider: ProcessProvider):
    sandbox = await _start(provider)
    group, detached = await _plant(sandbox)

    await sandbox.stop()

    assert await _wait_group_gone(group) and await _wait_gone(detached)


class _OsWithoutKill:
    """The os module as platoon.sandbox.process sees it, but SIGKILL ends nothing: a process that outlives the kill,
    as one that keeps forking might."""

    def __getattr__(self, name: str):
        return getattr(os, name)

    @staticmethod
    def kill(pid: int, signum: int) -> None:
        return None


@pytest.mark.asyncio
async def test_kill_processes_fails_when_a_process_outlives_it(provider: ProcessProvider, monkeypatch):
    sandbox = await _start(provider)
    try:
        group, detached = await _plant(sandbox)
        with monkeypatch.context() as patch:
            patch.setattr(process_module, "os", _OsWithoutKill())
            patch.setattr(process_module, "_KILL_PASSES", 3)
            with pytest.raises(RuntimeError, match=f"still alive after SIGKILL: {detached}$"):
                await sandbox.kill_processes()
        assert await _wait_group_gone(group) and _running(detached)  # its group died with the kill all the same
    finally:
        await sandbox.stop()
    assert await _wait_gone(detached)


@pytest.mark.asyncio
async def test_a_command_group_stays_registered_while_it_has_members(provider: ProcessProvider):
    # In a session of its own, a command is out of reach of a stopping rollout worker's process-group kill: the
    # worker ends the groups still registered (platoon.utils.stopping).
    sandbox = await _start(provider)
    try:
        done = await sandbox.exec("echo $$", timeout_s=10)
        left = await sandbox.exec("echo $$; sleep 30 > /dev/null 2>&1 &", timeout_s=10)
        timed_out = await sandbox.exec("echo $$; sleep 30", timeout_s=0.5)
        groups = registered_process_groups()

        assert int(done.stdout) not in groups and int(timed_out.stdout.split()[0]) not in groups
        assert int(left.stdout.split()[0]) in groups  # its background sleep is still there
    finally:
        await sandbox.stop()
    assert int(left.stdout.split()[0]) not in registered_process_groups()


@pytest.mark.asyncio
async def test_cancelled_exec_kills_its_process_group(provider: ProcessProvider):
    sandbox = await _start(provider)
    try:
        task = asyncio.create_task(sandbox.exec("echo $$ > /tmp/pid; sleep 30"))
        pid_file = sandbox.root / "tmp" / "pid"
        for _ in range(100):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _wait_group_gone(int(pid_file.read_text()))
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_upload_and_download_files_and_directories(provider: ProcessProvider, tmp_path: Path):
    source = tmp_path / "tests"
    (source / "nested").mkdir(parents=True)
    (source / "test.sh").write_text("echo ok\n")
    (source / "nested" / "data.txt").write_text("data\n")
    sandbox = await _start(provider)
    try:
        await sandbox.upload(source, "/tests")
        await sandbox.upload(source / "test.sh", "/logs/copy/test.sh")
        await sandbox.upload(source / "test.sh", "relative.sh")
        assert (await sandbox.exec("bash /tests/test.sh")).stdout == "ok\n"
        assert (sandbox.root / "tests" / "nested" / "data.txt").read_text() == "data\n"
        assert (sandbox.root / "logs" / "copy" / "test.sh").exists()
        assert (sandbox.root / "app" / "relative.sh").exists()

        await sandbox.download("/tests", tmp_path / "out")
        await sandbox.download("/tests/nested/data.txt", tmp_path / "single" / "data.txt")
        assert (tmp_path / "out" / "nested" / "data.txt").read_text() == "data\n"
        assert (tmp_path / "single" / "data.txt").read_text() == "data\n"
        with pytest.raises(FileNotFoundError):
            await sandbox.download("/logs/missing.txt", tmp_path / "missing.txt")
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_snapshot_restores_an_independent_copy_until_the_source_stops(provider: ProcessProvider):
    sandbox = await _start(provider)
    await sandbox.exec("echo before > /app/state.txt")
    snapshot = await sandbox.snapshot()
    await sandbox.exec("echo after > /app/state.txt")

    restored = await provider.restore(snapshot, sandbox.spec)
    try:
        assert restored.id != sandbox.id
        assert (await restored.exec("cat /app/state.txt")).stdout == "before\n"
        await restored.exec("echo restored > /app/state.txt")
        assert (await sandbox.exec("cat /app/state.txt")).stdout == "after\n"

        await sandbox.stop()
        assert not Path(snapshot.path).exists()
        assert (await restored.exec("cat /app/state.txt")).stdout == "restored\n"
    finally:
        await restored.stop()

    with pytest.raises(ValueError, match="docker"):
        await provider.restore(SandboxSnapshot(id="x", provider="docker", path="img", image="local"), sandbox.spec)


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_ends_the_sandbox(provider: ProcessProvider):
    sandbox = await _start(provider)
    await sandbox.stop()
    await sandbox.stop()

    with pytest.raises(RuntimeError, match="stopped"):
        await sandbox.exec("true")
    assert list(provider.root.iterdir()) == []


@pytest.mark.asyncio
async def test_mounts_and_custom_workdirs_are_mapped(provider: ProcessProvider, tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "input.txt").write_text("mounted\n")
    sandbox = await _start(provider, workdir="/workspace", mounts=[Mount(source=str(data), target="/srv/inputs")])
    try:
        result = await sandbox.exec("pwd; cat /srv/inputs/input.txt")
        assert result.stdout.splitlines() == [str(sandbox.root / "workspace"), "mounted"]
    finally:
        await sandbox.stop()
    assert (data / "input.txt").exists()


@pytest.mark.asyncio
async def test_run_command_bounds_captured_output():
    result = await run_command(["bash", "-c", "head -c 5000 /dev/zero | tr '\\0' x"], max_output_bytes=100)

    assert result.stdout.startswith("x" * 100)
    assert result.stdout.endswith("[4900 bytes of output dropped]")


def test_get_provider_resolves_names_and_import_paths(tmp_path: Path):
    assert isinstance(get_provider("process", root=tmp_path), ProcessProvider)
    provider = get_provider("platoon.sandbox.process:ProcessProvider", root=tmp_path)
    assert provider.root == tmp_path
    assert provider.capabilities == frozenset({"snapshot", "routes", "host_loopback", "stacks"})


@pytest.mark.asyncio
async def test_a_start_cancelled_while_its_directory_is_made_leaves_nothing(
    provider: ProcessProvider, monkeypatch: pytest.MonkeyPatch
):
    """The thread that makes a sandbox's directory runs on when its start is cancelled: the directory it makes is
    deleted once it is made, never left behind (a stack's start cancelled midway cancels its services' starts)."""

    made = asyncio.Event()
    loop = asyncio.get_running_loop()
    create = ProcessSandbox.create

    def slow_create(cls, base_dir: Path, spec: SandboxSpec, source: Path | None = None) -> ProcessSandbox:
        time.sleep(0.3)
        sandbox = create(base_dir, spec, source)
        loop.call_soon_threadsafe(made.set)
        return sandbox

    monkeypatch.setattr(ProcessSandbox, "create", classmethod(slow_create))
    starting = asyncio.ensure_future(_start(provider))
    await asyncio.sleep(0.05)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await asyncio.wait_for(made.wait(), timeout=5.0)
    await asyncio.sleep(0.1)  # the deletion, once the thread is done
    assert list(provider.root.iterdir()) == []
