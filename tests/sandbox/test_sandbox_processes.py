"""The in-sandbox process scripts (platoon.sandbox._processes) in a real PID namespace.

``NamespaceSandbox`` is a container as enroot makes one for hpc-sandbox: an unprivileged user, PID and mount
namespace with its own ``/proc``, whose init is a keep-alive loop, and commands that enter it with ``nsenter``.
The tests that use it skip where unprivileged user namespaces are not allowed.  ``HostSandbox`` has no namespace
of its own: only the process listing ever runs in it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from platoon.envs.harbor.verifier import VerifierFile, collect
from platoon.sandbox._exec import run_command
from platoon.sandbox._processes import kill_other_processes, own_processes
from platoon.sandbox.base import ExecResult

pytestmark = pytest.mark.asyncio

KEEPALIVE = "trap '' HUP INT QUIT TERM USR1 USR2; while :; do sleep 86400 & wait $!; done"
# A loop, a daemon in a session of its own, an orphan, a process named like the keep-alive's sleep, and a loop
# that starts a detached child every 10 ms.
AGENT_PROCESSES = (
    "nohup sh -c 'while :; do sleep 0.01; done' >/dev/null 2>&1 & "
    "setsid sh -c 'sleep 1000' </dev/null >/dev/null 2>&1 & "
    "( (sleep 1001 </dev/null >/dev/null 2>&1 &) & ) ; "
    "sh -c 'exec sleep 86400' </dev/null >/dev/null 2>&1 & "
    "nohup sh -c 'while :; do (sleep 1002 &); sleep 0.01; done' >/dev/null 2>&1 & "
    "sleep 0.3; echo planted"
)


def _namespaces_work() -> bool:
    if not (shutil.which("unshare") and shutil.which("nsenter")):
        return False
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "/bin/sh", "-c", "kill -0 1"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


requires_namespaces = pytest.mark.skipif(not _namespaces_work(), reason="unprivileged user namespaces are not allowed")


class NamespaceSandbox:
    """A keep-alive loop as the init of its own user, PID and mount namespaces; commands enter them."""

    def __init__(self) -> None:
        self.id = f"ns-{uuid.uuid4().hex[:8]}"
        self._unshare = subprocess.Popen(
            ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--mount-proc", "/bin/sh", "-c", KEEPALIVE],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10
        children: list[str] = []
        while not children and time.monotonic() < deadline:
            time.sleep(0.02)
            pid = self._unshare.pid
            with open(f"/proc/{pid}/task/{pid}/children") as listing:
                children = listing.read().split()
        assert children, "the namespace's init did not start"
        self.init = int(children[0])

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        argv = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        enter = ["nsenter", f"--target={self.init}", "--user", "--pid", "--mount", "--preserve-credentials"]
        return await run_command([*enter, f"--wd={cwd or '/'}", "--", *argv], timeout_s=timeout_s or 60, stdin=stdin)

    async def processes(self) -> list[str]:
        """``pid (comm) state`` of every live process in the namespace but the lister."""

        listing = await self.exec(
            "for d in /proc/[0-9]*; do p=${d#/proc/}; [ $p = $$ ] && continue; "
            'read -r s < $d/stat 2>/dev/null && echo "$p ${s#* }"; done'
        )
        entries = [line.split()[:3] for line in listing.stdout.splitlines()]
        return [" ".join(entry) for entry in entries if entry[2] not in ("Z", "X")]

    def stop(self) -> None:
        os.killpg(self._unshare.pid, signal.SIGKILL)
        try:
            os.kill(self.init, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self._unshare.wait()


@pytest.fixture
def sandbox():
    box = NamespaceSandbox()
    try:
        yield box
    finally:
        box.stop()


@requires_namespaces
async def test_the_sandbox_lists_only_its_keepalive_processes(sandbox: NamespaceSandbox):
    own = await own_processes(sandbox)

    assert own is not None and len(own) == 2  # the init and its sleep, never the lister
    assert {entry.split(":")[0] for entry in own} == {"1", "2"}


@requires_namespaces
async def test_every_process_the_commands_left_dies_and_the_sandbox_lives_on(sandbox: NamespaceSandbox):
    own = await own_processes(sandbox)
    planted = await sandbox.exec(AGENT_PROCESSES)
    assert planted.stdout.strip() == "planted"
    assert len(await sandbox.processes()) >= 9

    started = time.monotonic()
    await kill_other_processes(sandbox, own)

    assert time.monotonic() - started < 10
    left = await sandbox.processes()
    assert [entry.split()[0] for entry in left] == ["1", "2"], left  # the keep-alive loop and its sleep
    assert (await sandbox.exec("echo alive")).stdout == "alive\n"
    await asyncio.sleep(0.2)
    assert len(await sandbox.processes()) == 2  # nothing came back


@requires_namespaces
async def test_killing_without_the_sandbox_own_processes_is_refused(sandbox: NamespaceSandbox):
    with pytest.raises(RuntimeError, match="could not list its own processes"):
        await kill_other_processes(sandbox, None)


class HostSandbox:
    """Commands run on the host, in this process's PID namespace: a runtime that gave the sandbox none of its own
    (Singularity where the site says ``allow pid ns = no``), in which ``kill -1`` would reach the user's every
    process."""

    id = "host"

    async def exec(self, cmd, *, timeout_s=None, env=None, cwd=None, stdin=None) -> ExecResult:
        argv = ["/bin/sh", "-c", cmd] if isinstance(cmd, str) else list(cmd)
        return await run_command(argv, timeout_s=timeout_s or 60, cwd=cwd, stdin=stdin)


async def test_a_sandbox_without_a_pid_namespace_of_its_own_is_never_killed_in(caplog):
    sandbox = HostSandbox()

    with caplog.at_level("ERROR", logger="platoon.sandbox._processes"):
        assert await own_processes(sandbox) is None  # it lists this process
    assert f"shares the PID namespace of this process ({os.getpid()}:" in caplog.text
    with pytest.raises(RuntimeError, match="no PID namespace of its own"):
        await kill_other_processes(sandbox, None)  # so the kill never runs


@requires_namespaces
async def test_a_proc_mounted_over_fails_closed(sandbox: NamespaceSandbox):
    """A command running as root in the sandbox's user namespace can hide its processes behind another /proc."""

    own = await own_processes(sandbox)
    hidden = await sandbox.exec("mount -t tmpfs none /proc && nohup sleep 1000 >/dev/null 2>&1 & echo ok")
    assert hidden.exit_code == 0

    with pytest.raises(RuntimeError, match="does not list its processes"):
        await kill_other_processes(sandbox, own)
    assert await own_processes(sandbox) is None


@requires_namespaces
async def test_harbor_never_reads_a_reward_file_with_something_mounted_over_it(
    sandbox: NamespaceSandbox, tmp_path: Path
):
    """Root in an enroot sandbox can bind a file that reads 1 (a host file in reach) over the reward file, which then
    reads as a regular file; collect refuses a verifier directory with anything mounted on or below it."""

    verifier, one = tmp_path / "verifier", tmp_path / "one"
    verifier.mkdir()
    (verifier / "reward.txt").write_text("0\n")
    one.write_text("1\n")
    assert (await collect(sandbox, str(verifier), tail=()))["reward.txt"] == VerifierFile("file", "0\n")

    mounted = await sandbox.exec(["mount", "--bind", str(one), str(verifier / "reward.txt")])
    assert mounted.exit_code == 0, mounted.stderr

    assert await collect(sandbox, str(verifier), tail=()) == {
        "reward.json": VerifierFile("mount"),
        "reward.txt": VerifierFile("mount"),
    }
