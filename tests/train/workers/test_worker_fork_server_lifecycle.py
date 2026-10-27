"""A rollout worker supervisor's fork server, as ``platoon.train.workers.fork_servers`` starts, waits for, replaces and
closes it: on a real supervisor, with a fake server in place of the process (``test_worker_forkserver.py`` runs the
real one).  Its lock is one per event loop; a dead server is retired without a word; one still importing past the
startup budget leaves the worker to exec and is kept, one that failed is closed and forgotten; each new server is
announced again; and a supervisor that started as many servers as it starts takes no lock to say so."""

from __future__ import annotations

import asyncio
import logging
import os

import pytest
from fakes.workers import worker_request

from platoon.train.workers import fork_servers
from platoon.train.workers.protocol import RolloutWorkerRequest
from platoon.train.workers.supervisor import RolloutWorkerSupervisor
from platoon.utils.forkserver import ForkServer, ForkServerError


class _Server:
    """A fork server's state as the supervisor reads it; ``ready()`` raises ``failure``, when given."""

    def __init__(self, *, alive=True, serving=False, error=None, failure=None, loop=None, env=None, cwd=None):
        self.alive, self.serving, self.error, self.failure = alive, serving, error, failure
        self.loop, self.env, self.cwd = loop, env, cwd
        self.pid, self.seconds, self.prepare_errors = 4242, 0.1, []
        self.closed = False

    async def ready(self, timeout):
        if self.failure is not None:
            raise self.failure

    def close(self):
        self.closed = True


def _supervisor(tmp_path) -> RolloutWorkerSupervisor:
    return RolloutWorkerSupervisor(scratch_dir=str(tmp_path), start_method="forkserver")


def _request(tmp_path) -> RolloutWorkerRequest:
    return worker_request(tmp_path, "task-r0-forkserver", startup_budget_seconds=1.0, rollout_budget_seconds=2.0)


def test_the_lock_that_starts_a_server_is_one_per_event_loop(tmp_path):
    supervisor = _supervisor(tmp_path)

    async def locks() -> tuple[asyncio.Lock, asyncio.Lock]:
        loop = asyncio.get_running_loop()
        return fork_servers._lock(supervisor, loop), fork_servers._lock(supervisor, loop)

    first, again = asyncio.run(locks())
    other, _ = asyncio.run(locks())
    assert first is again and other is not first


def test_a_dead_server_is_retired_without_a_word(tmp_path, caplog):
    supervisor = _supervisor(tmp_path)

    async def retire() -> _Server:
        loop = asyncio.get_running_loop()
        server = _Server(alive=False, loop=loop, env={}, cwd=os.getcwd())
        supervisor._fork_server = server
        fork_servers._retire(supervisor, server, loop, {}, os.getcwd())
        return server

    with caplog.at_level(logging.WARNING):
        server = asyncio.run(retire())

    assert server.closed and supervisor._fork_server is None
    assert "started with another" not in caplog.text


def test_a_server_still_importing_past_the_startup_budget_is_kept_and_the_worker_execs(tmp_path, caplog):
    supervisor = _supervisor(tmp_path)
    server = _Server(alive=True, failure=ForkServerError("still importing"))
    supervisor._fork_server = server

    with caplog.at_level(logging.WARNING):
        serves = asyncio.run(fork_servers._await_imports(supervisor, server, _request(tmp_path)))

    assert serves is False and supervisor._fork_server is server and not server.closed
    assert "has not imported the worker's stack" in caplog.text


@pytest.mark.parametrize("permanent", [False, True], ids=["died importing", "cannot serve"])
def test_a_server_that_failed_its_imports_is_closed_and_forgotten(tmp_path, permanent):
    """One that died while importing is closed and forgotten (another may start); one that says it cannot serve turns
    forking off as well."""

    supervisor = _supervisor(tmp_path)
    server = _Server(alive=False, failure=ForkServerError("it died", permanent=permanent))
    supervisor._fork_server = server

    serves = asyncio.run(fork_servers._await_imports(supervisor, server, _request(tmp_path)))

    assert serves is False and server.closed and supervisor._fork_server is None
    assert (supervisor._fork_server_starts == supervisor._max_fork_server_starts) is permanent


def test_each_new_server_is_announced_and_waited_for_afresh(tmp_path, monkeypatch):
    supervisor = _supervisor(tmp_path)
    supervisor._fork_server_late = supervisor._fork_server_announced = True  # of the server before
    monkeypatch.setattr(ForkServer, "launch", classmethod(lambda cls, **kwargs: _Server()))

    server = fork_servers._launch_fork_server(supervisor, _request(tmp_path), {}, os.getcwd())

    assert supervisor._fork_server is server
    assert (supervisor._fork_server_late, supervisor._fork_server_announced) == (False, False)


def test_closing_the_supervisor_s_server_forgets_it(tmp_path):
    supervisor = _supervisor(tmp_path)
    server = _Server()
    supervisor._fork_server = server

    fork_servers.close(supervisor)

    assert server.closed and supervisor._fork_server is None


def test_a_supervisor_that_started_its_last_server_execs_without_waiting_for_the_lock(tmp_path):
    """Its lock may be held, while a request waits for a server's imports: a request that finds no server and no start
    left goes on by exec at once."""

    supervisor = _supervisor(tmp_path)
    supervisor._fork_server_starts = supervisor._max_fork_server_starts

    async def run():
        lock = fork_servers._lock(supervisor, asyncio.get_running_loop())
        async with lock:  # held, as by a request waiting for a server's imports
            return await asyncio.wait_for(fork_servers._fork_server_for(supervisor, _request(tmp_path)), 1.0)

    assert asyncio.run(run()) is None
