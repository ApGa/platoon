"""What a stopping process ends -- the process groups registered, never one whose id another process has now, and
the sessions its descendants started -- and the cleanups it lets finish, which a rollout's cleanup scope collects
(``platoon.utils.stopping``)."""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from platoon.utils import stopping
from platoon.utils.stopping import (
    CleanupScope,
    cleanups,
    end_process_groups,
    in_cleanup_scope,
    kill_process_groups,
    let_finish,
    register_process_group,
    registered_process_groups,
    unregister_process_group,
)


@pytest.fixture(autouse=True)
def _a_registry_of_its_own(monkeypatch):
    """Each test sees only what it registers: the registries are process-wide."""

    monkeypatch.setattr(stopping, "_groups", {})
    monkeypatch.setattr(stopping, "_cleanups", set())


def _in_a_session_of_its_own(command: str) -> subprocess.Popen:
    return subprocess.Popen(["sh", "-c", command], start_new_session=True)


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.02)


def test_ending_the_groups_asks_first_and_kills_what_ignores_it(tmp_path):
    ready, asked = tmp_path / "ready", tmp_path / "asked"
    process = _in_a_session_of_its_own(f"trap 'echo >> {asked}' TERM; touch {ready}; while :; do sleep 0.05; done")
    register_process_group(process.pid)
    try:
        _wait_for(ready)
        started = time.monotonic()

        assert end_process_groups(0.5) == [process.pid]

        assert process.wait(timeout=10) == -signal.SIGKILL
        assert time.monotonic() - started >= 0.5  # it had its grace
        assert asked.exists() and registered_process_groups() == []
    finally:
        process.kill()
        process.wait()


def test_a_group_that_exits_on_sigterm_is_not_waited_for():
    process = _in_a_session_of_its_own("exec sleep 30")
    register_process_group(process.pid)
    started = time.monotonic()

    assert end_process_groups(10.0) == [process.pid]

    assert time.monotonic() - started < 5
    assert process.wait(timeout=10) == -signal.SIGTERM


def test_a_group_whose_id_another_process_has_now_is_never_signalled():
    process = _in_a_session_of_its_own("exec sleep 30")
    try:
        register_process_group(process.pid)
        stopping._groups[process.pid] += 1  # as if its leader had gone and the pid been given to another process

        assert kill_process_groups() == [] and end_process_groups(0.1) == []

        assert process.poll() is None and registered_process_groups() == []
    finally:
        process.kill()
        process.wait()


def test_kill_process_groups_kills_at_once_and_forgets_them():
    process = _in_a_session_of_its_own("trap '' TERM; while :; do sleep 0.05; done")
    register_process_group(process.pid)
    unregister_process_group(12345678)  # unknown: ignored

    assert kill_process_groups() == [process.pid]

    assert process.wait(timeout=10) == -signal.SIGKILL and registered_process_groups() == []


def test_the_sessions_its_descendants_started_are_found_and_ended():
    # In a process of its own: this test's process may have descendants of its own.
    script = textwrap.dedent(
        """
        import subprocess, time
        from platoon.utils.stopping import end_process_groups, register_descendant_groups

        shell = subprocess.Popen(["sh", "-c", "setsid sleep 1000 & echo $!; wait"], stdout=subprocess.PIPE, text=True)
        daemon = int(shell.stdout.readline())
        time.sleep(0.2)
        found = register_descendant_groups()
        assert found == [daemon], (found, daemon)
        assert register_descendant_groups() == []  # registered already
        assert end_process_groups(0.2) == [daemon]
        shell.wait(timeout=10)  # its wait returned: the daemon is gone
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)

    assert done.returncode == 0 and done.stdout == "ok\n", done.stdout + done.stderr


@pytest.mark.asyncio
async def test_cleanups_are_listed_until_they_end():
    gate = asyncio.Event()
    cleanup = let_finish(asyncio.ensure_future(gate.wait()))
    other = asyncio.ensure_future(asyncio.sleep(10))

    assert cleanups() == [cleanup] and cleanups(asyncio.get_running_loop()) == [cleanup]
    another = asyncio.new_event_loop()
    try:
        assert cleanups(another) == []
    finally:
        another.close()
    gate.set()
    await cleanup
    assert cleanups() == []
    other.cancel()


@pytest.mark.asyncio
async def test_a_cleanup_scope_collects_the_cleanups_its_code_starts_then_or_later_on_its_own_loop():
    gate = asyncio.Event()
    outside = let_finish(asyncio.ensure_future(gate.wait()))  # no scope is open

    async def later() -> None:  # a task of the scope's code that starts a cleanup after the block
        await asyncio.sleep(0)
        let_finish(asyncio.ensure_future(gate.wait()))

    async def another_loop() -> None:
        let_finish(asyncio.ensure_future(asyncio.sleep(0)))

    scope = CleanupScope()
    with scope:
        first = let_finish(asyncio.ensure_future(gate.wait()))
        waits = in_cleanup_scope(asyncio.ensure_future(gate.wait()))  # the scope's only
        starter = asyncio.ensure_future(later())
        await asyncio.to_thread(asyncio.run, another_loop())  # the thread copies the context, not the loop
    await starter

    running = scope.running()
    assert len(running) == 3 and first in running and waits in running and outside not in running
    assert waits not in cleanups() and first in cleanups()
    gate.set()
    assert await scope.finish(5) == 0 and scope.running() == []


@pytest.mark.asyncio
async def test_finishing_a_scope_waits_for_the_cleanups_its_cleanups_start_within_a_bound_and_cancels_none():
    gate = asyncio.Event()

    async def teardown() -> None:  # as it ends it starts another, as an env's that waits for its code does
        await asyncio.sleep(0.05)
        let_finish(asyncio.ensure_future(gate.wait()))

    scope = CleanupScope()
    with scope:
        let_finish(asyncio.ensure_future(teardown()))

    assert await scope.finish(0.3) == 1  # the second still runs at the bound
    (hung,) = scope.running()
    waiting = asyncio.ensure_future(scope.finish(10))
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not hung.done()  # cancelling the wait cancels no cleanup
    gate.set()
    assert await scope.finish(1) == 0
