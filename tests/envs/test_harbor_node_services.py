"""Node services (platoon.envs.harbor.node_services) on the process provider: a declaration, the per-node supervisor
that a run starts, owns the service under a node lock and dies with the run, the front's scopes, and Harbor stacks that
reach a node service in place of a sidecar, whose failure makes their rollouts invalid.  The node service is the
fixtures' judge (``fixtures/harbor-node-services``, a standard-library server written for platoon)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from site_cache import use_site_cache

from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.node_services import (
    NodeServiceError,
    NodeServiceRunner,
    NodeServiceSpec,
    NodeServiceUnavailable,
    declared_node_services,
    node_dir,
    node_services_of,
    open_scope,
    ping,
    staging,
)
from platoon.envs.harbor.node_services import runner as node_runner
from platoon.envs.harbor.node_services.front import Front
from platoon.envs.service import EnvSetupError
from platoon.sandbox.process import ProcessProvider
from platoon.train.components import normalize_environment_configs
from platoon.train.startup import RolloutStartup

pytestmark = pytest.mark.asyncio

FIXTURES = Path(__file__).parent / "fixtures" / "harbor-node-services"
NAME = "judge-pool"


@pytest.fixture(autouse=True)
def scratch(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Node scratch of the test's own, short enough for the sockets' addresses below it."""

    path = Path(tempfile.mkdtemp(prefix="ns", dir="/tmp"))
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(path))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def data(scratch: Path) -> Path:
    """A directory the judge writes to (``/state``): its starts' pids, in ``starts.log``."""

    path = scratch / "data"
    path.mkdir()
    return path


def _declaration(data: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "image": "local",
        "command": [sys.executable, "/judge/judge.py"],
        "port": 8081,
        "binds": [
            {"source": str(FIXTURES / "judge"), "target": "/judge"},
            {"source": str(FIXTURES / "problems"), "target": "/problems", "stage": True},
            {"source": str(data), "target": "/state", "read_only": False},
        ],
        "health": {"interval": 0.2, "timeout": 5, "start_period": 30, "retries": 2},
        "front": {"routes": ["POST /evaluate", "GET /health"]},
        **overrides,
    }


def _options(scratch: Path) -> dict[str, Any]:
    return {"root": str(scratch / "sandboxes")}


def _spec(data: Path, scratch: Path, **overrides: Any) -> NodeServiceSpec:
    return NodeServiceSpec.parse(
        NAME, _declaration(data, **overrides), provider="process", provider_options=_options(scratch)
    )


def _starts(data: Path) -> list[int]:
    log = data / "starts.log"
    return [int(line) for line in log.read_text().split()] if log.exists() else []


def _requests(data: Path) -> list[str]:
    """The problems of the requests the judge began, in order."""

    log = data / "requests.log"
    return log.read_text().split() if log.exists() else []


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


async def _until(condition: Any, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the condition never held")


async def _request(socket: str, method: str, path: str, *, body: Any = None, headers: dict | None = None):
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=socket)) as http:
        async with http.request(method, f"http://scope{path}", json=body, headers=headers or {}) as response:
            return response.status, await response.json(content_type=None)


@pytest_asyncio.fixture
async def runner(data: Path, scratch: Path) -> Any:
    started = NodeServiceRunner([_spec(data, scratch)])
    await started.start()
    await started.ready(timeout_s=60)
    yield started
    await started.stop()


# --- the declaration -----------------------------------------------------------------------------------------------


async def test_a_declaration_is_checked_and_keyed_by_what_it_runs(data: Path, scratch: Path):
    spec = _spec(data, scratch, cpus="0", env={"MODE": "x"})
    assert (spec.port, spec.network, spec.cpus, spec.front.routes) == (
        8081,
        "none",
        "0",
        ("POST /evaluate", "GET /health"),
    )
    assert NodeServiceSpec.from_json(spec.to_json()) == spec
    assert spec.key == NodeServiceSpec.from_json(spec.to_json()).key != _spec(data, scratch, env={"MODE": "y"}).key
    assert node_dir(spec).name == f"{NAME}-{spec.key}"
    assert spec.allows("post", "/evaluate") and not spec.allows("GET", "/admin")
    for overrides, message in (
        ({"port": 0}, "port must be the TCP port"),
        ({"cpus": "a-b"}, "cpus is a number of CPUs or a list"),
        ({"network": "egress"}, "network is none or host"),
        ({"front": {"routes": ["/evaluate"]}}, "front.routes must list routes"),
        ({"binds": [{"source": "relative", "target": "/x"}]}, "source must be an absolute host path"),
        ({"binds": [{"source": "/x", "target": "/x", "stage": True, "read_only": False}]}, "staged bind is shared"),
        ({"sidecar": True}, "unknown keys"),
    ):
        with pytest.raises(NodeServiceError, match=message):
            _spec(data, scratch, **overrides)


async def test_an_environment_may_take_the_site_profile_s_declaration(data: Path, scratch: Path, tmp_path, monkeypatch):
    profile = tmp_path / "site.yaml"
    profile.write_text(json.dumps({"node_services": {NAME: _declaration(data)}}))
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    [spec] = declared_node_services({NAME: "site"}, provider="process", provider_options=_options(scratch)).values()
    assert spec == _spec(data, scratch)
    with pytest.raises(NodeServiceError, match="the site profile .* declares none"):
        declared_node_services({"other": "site"}, provider="process")
    environments = normalize_environment_configs(
        [
            {
                "name": "a",
                "service": "harbor",
                "service_kwargs": {"provider": "process", "node_services": {NAME: "site"}},
            },
            {
                "name": "b",
                "service": "harbor",
                "service_kwargs": {"provider": "process", "node_services": {NAME: "site"}},
            },
        ]
    )
    assert [spec.name for spec in node_services_of(environments)] == [NAME]  # one service for both entries


# --- the supervisor and the front ----------------------------------------------------------------------------------


async def test_a_scope_reaches_the_service_s_declared_routes_for_its_own_problem_alone(runner, data: Path):
    [spec] = runner.specs
    assert (await ping(spec))["state"] == "up"
    scope = await open_scope(spec, {"problem_id": "sum-a"})
    try:
        assert await _request(scope.socket, "POST", "/evaluate", body={"code": "42"}) == (
            200,
            {"status": "done", "score": 100.0, "problem_id": "sum-a"},
        )
        # The rollout cannot name another problem, nor reach a route the declaration does not list, nor carry a token.
        other = {"X-Platoon-Scope": json.dumps({"problem_id": "sum-b"}), "Authorization": "Bearer guessed"}
        status, answer = await _request(scope.socket, "POST", "/evaluate", body={"code": "7"}, headers=other)
        assert (status, answer["score"], answer["problem_id"]) == (200, 0.0, "sum-a")
        assert (await _request(scope.socket, "GET", "/admin"))[0] == 404
        assert (await _request(scope.socket, "POST", "/health"))[0] == 404
        assert await scope.status() == (True, "")
    finally:
        await scope.close()
    assert not Path(scope.socket).exists()
    assert len(_starts(data)) == 1


async def test_the_service_starts_once_per_node_whoever_asks_for_it(data: Path, scratch: Path):
    # Three runs on one node declare the service at once, and their workers open scopes at once: one start serves all.
    runners = [NodeServiceRunner([_spec(data, scratch)]) for _ in range(3)]
    try:
        for each in runners:
            await each.start()
        await asyncio.gather(*(each.ready(timeout_s=60) for each in runners))
        spec = runners[0].specs[0]
        scopes = await asyncio.gather(*(open_scope(spec, {"problem_id": "sum-b"}) for _ in range(8)))
        answers = await asyncio.gather(
            *(_request(scope.socket, "POST", "/evaluate", body={"code": "7"}) for scope in scopes)
        )
        assert {(status, answer["score"]) for status, answer in answers} == {(200, 100.0)}
        assert len(_starts(data)) == 1 and len({scope.socket for scope in scopes}) == 8
        for scope in scopes:
            await scope.close()
    finally:
        for each in runners:
            await each.stop()
    assert not any(_alive(pid) for pid in _starts(data))


async def test_the_front_tries_again_a_request_that_the_service_s_full_socket_did_not_take(data: Path, scratch: Path):
    # More rollouts at once than the service's socket queues: the socket refuses a connection, so the request never
    # reached the service, and the front tries it again once the service takes connections, rather than answer 502.
    backend = scratch / "service.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(backend))
    listener.listen(0)
    queued = []
    for _ in range(4):  # the socket's queue, full
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.setblocking(False)
        try:
            client.connect(str(backend))
        except BlockingIOError:
            pass
        queued.append(client)

    async def evaluate(request: web.Request) -> web.Response:
        return web.json_response({"scope": json.loads(request.headers["X-Platoon-Scope"])})

    app = web.Application()
    app.router.add_post("/evaluate", evaluate)
    service = web.AppRunner(app)
    await service.setup()
    (scratch / "node").mkdir()

    async def serving(wait_s: float) -> int:
        return 1

    front = Front(_spec(data, scratch), scratch / "node", backend=backend, token="t", serving=serving)
    await front.start()
    try:
        scope = await front.open({"problem_id": "sum-a"}, generation=1)
        asking = asyncio.ensure_future(_request(str(scope.socket), "POST", "/evaluate", body={"code": "42"}))
        await asyncio.sleep(0.2)
        assert not asking.done()  # still trying
        await web.SockSite(service, listener).start()  # the service takes connections again
        assert await asking == (200, {"scope": {"problem_id": "sum-a"}})
    finally:
        await front.close()
        for client in queued:
            client.close()
        await service.cleanup()
        listener.close()


async def test_another_run_s_supervisor_takes_over_when_the_owner_s_run_ends(data: Path, scratch: Path):
    first, second = NodeServiceRunner([_spec(data, scratch)]), NodeServiceRunner([_spec(data, scratch)])
    try:
        await first.start()
        await first.ready(timeout_s=60)
        await second.start()
        await second.ready(timeout_s=60)  # served by the first one's start
        spec = first.specs[0]
        scope = await open_scope(spec, {"problem_id": "sum-a"})
        assert len(_starts(data)) == 1
        # The second run's workers open scopes all through the handover: each waits for the next owner, none fails.
        refused: list[Exception] = []
        opened: list[int] = []

        async def keep_opening() -> None:
            while not opened or opened[-1] < 2:
                try:
                    opening = await open_scope(spec, {"problem_id": "sum-a"}, wait_s=60)
                except NodeServiceUnavailable as exc:
                    refused.append(exc)
                else:
                    opened.append(opening.generation)
                    await opening.close()
                await asyncio.sleep(0.05)

        workers = asyncio.ensure_future(keep_opening())
        await asyncio.sleep(0.3)
        await first.stop()
        assert (await scope.status())[0] is False  # the scopes of the owner's front went with it
        await scope.close()
        await asyncio.wait_for(workers, 60)
        assert refused == [] and opened[0] == 1 and opened[-1] == 2
        await second.ready(timeout_s=60)  # the second one's own start now, counted on the node
        assert (await ping(spec))["generation"] == 2
        assert len(_starts(data)) == 2 and not _alive(_starts(data)[0])
        again = await open_scope(spec, {"problem_id": "sum-a"})
        assert (await _request(again.socket, "POST", "/evaluate", body={"code": "42"}))[1]["score"] == 100.0
        await again.close()
    finally:
        await first.stop()
        await second.stop()
    assert not any(_alive(pid) for pid in _starts(data))


async def test_a_failed_start_is_down_for_the_scopes_that_used_it_and_the_others_go_on(runner, data: Path):
    [spec] = runner.specs
    used, idle = await open_scope(spec, {"problem_id": "sum-a"}), await open_scope(spec, {"problem_id": "sum-b"})
    [judge] = _starts(data)
    asking = asyncio.ensure_future(_request(used.socket, "POST", "/evaluate", body={"code": "slow"}))
    await _until(lambda: _requests(data) == ["sum-a"])
    os.kill(judge, signal.SIGKILL)  # while it grades the used scope's request
    status, answer = await asking
    assert status in (502, 503), answer  # never an answer of another start
    await _until(lambda: len(_starts(data)) == 2)
    await runner.ready(timeout_s=60)
    up, detail = await used.status()
    assert up is False and "node service" in detail  # for good
    # A rollout that did not use the start that failed goes on with the next one: no work of its is lost.
    assert await idle.status() == (True, "")
    assert (await _request(idle.socket, "POST", "/evaluate", body={"code": "7"}))[1]["score"] == 100.0
    assert await idle.status() == (True, "") and idle.served == [2]
    fresh = await open_scope(spec, {"problem_id": "sum-a"})
    assert fresh.generation == used.generation + 1
    assert (await _request(fresh.socket, "POST", "/evaluate", body={"code": "42"}))[1]["score"] == 100.0
    for scope in (used, idle, fresh):
        await scope.close()


async def test_a_failed_start_makes_down_only_the_scopes_it_may_have_answered_wrongly(data: Path, scratch: Path):
    # A scope with a request in flight as the start failed, or one since the start last passed its health check, is
    # down; one idle since then is not.
    async def serving(wait_s: float) -> int:
        return 1

    (scratch / "node").mkdir()
    front = Front(_spec(data, scratch), scratch / "node", backend=scratch / "none.sock", token="t", serving=serving)
    await front.start()
    try:
        idle, before, after, busy = [await front.open({"problem_id": "sum-a"}, generation=1) for _ in range(4)]
        before.last_active, after.last_active, busy.active, busy.last_active = 10.0, 30.0, 1, 5.0
        assert front.start_failed("the judge died", since=20.0) == 2
        assert [scope.down for scope in (idle, before, after, busy)] == [None, None, "the judge died", "the judge died"]
        # The front's own failure to serve a request makes its scope down too.
        assert (await _request(str(idle.socket), "POST", "/evaluate", body={"code": "42"}))[0] == 502
        assert idle.down == "the node service judge-pool could not be reached"
    finally:
        await front.close()


async def test_a_probe_never_restarts_the_service_on_a_failed_health_check(data: Path, scratch: Path):
    # A judge's bad moment, seen by one rollout's probe, must not restart the service under every other rollout: only
    # the supervisor's own checks, an interval apart, count.
    health = {"interval": 60, "timeout": 5, "start_period": 30, "retries": 2}
    runner = NodeServiceRunner([_spec(data, scratch, health=health)])
    await runner.start()
    try:
        await runner.ready(timeout_s=60)
        [spec] = runner.specs
        mine, other = await open_scope(spec, {"problem_id": "sum-a"}), await open_scope(spec, {"problem_id": "sum-b"})
        (data / "unhealthy").write_text("")
        assert await mine.status() == (True, "")
        await asyncio.sleep(2.5)
        (data / "unhealthy").unlink()
        assert (await ping(spec))["generation"] == 1 and len(_starts(data)) == 1
        assert await other.status() == (True, "")
        for scope in (mine, other):
            await scope.close()
    finally:
        await runner.stop()


async def test_a_service_that_never_starts_fails_the_run_s_start_with_its_log(data: Path, scratch: Path):
    command = [sys.executable, "-c", "import sys; print('no judge here'); sys.exit(3)"]
    failing = NodeServiceRunner([_spec(data, scratch, command=command, restarts=1)])
    await failing.start()
    try:
        with pytest.raises(NodeServiceError, match="failed to start 1 times in a row.*no judge here"):
            await failing.ready(timeout_s=60)
        with pytest.raises(NodeServiceUnavailable, match="opened no scope"):
            await open_scope(failing.specs[0], {}, wait_s=1)
    finally:
        await failing.stop()


async def test_a_service_that_no_supervisor_serves_here_is_unavailable(data: Path, scratch: Path):
    with pytest.raises(NodeServiceUnavailable, match="is not running on this node"):
        await open_scope(_spec(data, scratch), {"problem_id": "sum-a"})


async def test_a_service_that_keeps_failing_soon_after_it_starts_fails_the_run(data: Path, scratch: Path):
    # A judge that crashes soon after each start is started again, at once and then less and less often; after
    # `restarts` such starts in a row its supervisor gives up, and the run hears of it.
    failing = NodeServiceRunner([_spec(data, scratch, env={"CRASH_AFTER_S": "3"}, restarts=2)])
    heard: list[str] = []
    failing.on_failure(heard.append)
    await failing.start()
    try:
        await failing.ready(timeout_s=60)
        await _until(lambda: bool(heard), 60)
        assert "failed 2 times in a row" in heard[0] and failing.failure == heard[0]
        assert len(_starts(data)) == 2 and (await ping(failing.specs[0]))["state"] == "failed"
    finally:
        await failing.stop()
    assert not any(_alive(pid) for pid in _starts(data))


async def test_the_run_replaces_a_supervisor_that_died_and_reaps_what_it_left(runner, data: Path):
    [spec] = runner.specs
    [judge] = _starts(data)
    [supervisor] = runner._processes.values()
    supervisor.kill()
    supervisor.wait()
    await runner.ready(timeout_s=60)  # once the supervisor that replaces it serves
    await _until(lambda: len(_starts(data)) == 2 and not _alive(judge), 60)
    assert runner.failure is None and (await ping(spec))["generation"] == 2
    scope = await open_scope(spec, {"problem_id": "sum-a"})
    assert (await _request(scope.socket, "POST", "/evaluate", body={"code": "42"}))[1]["score"] == 100.0
    await scope.close()


async def test_a_supervisor_that_keeps_dying_fails_the_run(runner, data: Path):
    heard: list[str] = []
    runner.on_failure(heard.append)
    [spec] = runner.specs
    deaths = node_runner.SUPERVISOR_DEATHS
    for death in range(deaths):
        [supervisor] = runner._processes.values()
        supervisor.kill()
        supervisor.wait()
        if death + 1 < deaths:
            await _until(lambda: runner._processes[spec.key] is not supervisor, 30)
            await runner.ready(timeout_s=60)
    await _until(lambda: bool(heard), 30)
    assert heard == [f"the node service {NAME} failed: its supervisor died {deaths} times"]
    await _until(lambda: not any(_alive(pid) for pid in _starts(data)))


async def test_a_signal_to_the_run_s_process_group_leaves_its_node_services_serving(
    data: Path, scratch: Path, tmp_path: Path
):
    # A terminal's Ctrl+C, or a scheduler's SIGTERM, asks the run to stop at its next step boundary, once it graded its
    # last rollouts: its node services serve on until the run stops them, and so they do after a SIGTERM of their own.
    declaration = tmp_path / "spec.json"
    declaration.write_text(json.dumps(_spec(data, scratch).to_json()))
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, json, signal, sys
            from platoon.envs.harbor.node_services import NodeServiceRunner, NodeServiceSpec

            async def main():
                for signum in (signal.SIGINT, signal.SIGTERM):  # as platoon train: a first signal stops it gracefully
                    signal.signal(signum, lambda *args: None)
                runner = NodeServiceRunner([NodeServiceSpec.from_json(json.loads(open(sys.argv[1]).read()))])
                await runner.start()
                await runner.ready(timeout_s=60)
                print("ready", flush=True)
                await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
                await runner.stop()
                print("stopped", flush=True)

            asyncio.run(main())
            """
        )
    )
    run = subprocess.Popen(
        [sys.executable, str(script), str(declaration)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert run.stdin is not None and run.stdout is not None
        assert run.stdout.readline().strip() == b"ready"
        [judge] = _starts(data)
        supervisor = _child_of(run.pid)
        for signum in (signal.SIGINT, signal.SIGTERM):
            os.killpg(run.pid, signum)
        os.kill(supervisor, signal.SIGTERM)
        await asyncio.sleep(2)
        assert _alive(supervisor) and _alive(judge)
        scope = await open_scope(_spec(data, scratch), {"problem_id": "sum-a"})
        assert (await _request(scope.socket, "POST", "/evaluate", body={"code": "42"}))[1]["score"] == 100.0
        await scope.close()
        run.stdin.write(b"go\n")
        run.stdin.flush()
        assert run.stdout.readline().strip() == b"stopped"
        await _until(lambda: not _alive(supervisor) and not _alive(judge))
    finally:
        if run.poll() is None:
            run.kill()


async def test_the_supervisor_and_its_service_die_with_the_run(data: Path, scratch: Path, tmp_path: Path):
    # A run's process that is killed outright takes its supervisor along, which stops the service.
    declaration = tmp_path / "spec.json"
    declaration.write_text(json.dumps(_spec(data, scratch).to_json()))
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, json, sys
            from platoon.envs.harbor.node_services import NodeServiceRunner, NodeServiceSpec

            async def main():
                runner = NodeServiceRunner([NodeServiceSpec.from_json(json.loads(open(sys.argv[1]).read()))])
                await runner.start()
                await runner.ready(timeout_s=60)
                print("ready", flush=True)
                await asyncio.sleep(3600)

            asyncio.run(main())
            """
        )
    )
    run = subprocess.Popen([sys.executable, str(script), str(declaration)], stdout=subprocess.PIPE)
    try:
        assert run.stdout is not None and run.stdout.readline().strip() == b"ready"
        [judge] = _starts(data)
        supervisor = _child_of(run.pid)
        assert supervisor and _alive(supervisor)
        run.kill()
        run.wait()
        await _until(lambda: not _alive(supervisor) and not _alive(judge))
    finally:
        if run.poll() is None:
            run.kill()


async def test_what_a_supervisor_killed_outright_left_is_reaped_when_the_run_stops(data: Path, scratch: Path):
    # A supervisor that cannot stop its service (SIGKILL) leaves it running; the run reaps it by its labels.
    runner = NodeServiceRunner([_spec(data, scratch)])
    await runner.start()
    try:
        await runner.ready(timeout_s=60)
        [judge] = _starts(data)
        [supervisor] = runner._processes.values()
        supervisor.kill()
        supervisor.wait()
    finally:
        await runner.stop()
    await _until(lambda: not _alive(judge))


def _child_of(parent: int) -> int:
    """A process whose parent was ``parent`` (the supervisor the run started), 0 for none."""

    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[1]) == parent:
            return int(stat.parent.name)
    return 0


async def test_a_run_starts_its_environments_node_services_before_any_rollout(data: Path, scratch: Path):
    environments = normalize_environment_configs(
        [
            {
                "service": "harbor",
                "service_kwargs": {
                    "provider": "process",
                    "provider_options": _options(scratch),
                    "node_services": {NAME: _declaration(data)},
                },
            }
        ]
    )
    startup = await RolloutStartup.begin(environments)
    try:
        assert startup.node_services is not None
        await startup.ready()
        assert (await ping(startup.node_services.specs[0]))["state"] == "up"
    finally:
        assert await startup.close() is None  # as the run ends: they did not fail
    assert not any(_alive(pid) for pid in _starts(data))
    assert (await RolloutStartup.begin(normalize_environment_configs([{"service": "table"}]))).node_services is None


async def test_node_services_take_whole_cores_of_the_run_s_cpus_each_its_own():
    def siblings(cpu: int) -> list[int]:  # two hyperthreads a core: CPU n and n + 8
        return [cpu % 8, cpu % 8 + 8]

    allowed = list(range(16))
    plan_cpus = node_runner.plan_cpus
    plan, rest = plan_cpus([("a", "judge", 4), ("b", "other", 2), ("c", "listed", "6")], allowed, siblings=siblings)
    assert plan == {"a": [0, 1, 8, 9], "b": [2, 10], "c": [6]}
    assert rest == [3, 4, 5, 7, 11, 12, 13, 15]  # 14, the sibling of the listed CPU 6, is no one's
    with pytest.raises(NodeServiceError, match="which another node service takes"):
        plan_cpus([("a", "judge", 4), ("b", "other", "0-1")], allowed, siblings=siblings)
    with pytest.raises(NodeServiceError, match="which this run may not use"):
        plan_cpus([("a", "judge", "16")], allowed, siblings=siblings)


async def test_the_run_s_rollouts_run_on_the_cpus_its_node_services_leave(data: Path, scratch: Path):
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 8:
        pytest.skip("needs 8 CPUs")
    environments = normalize_environment_configs(
        [
            {
                "service": "harbor",
                "service_kwargs": {
                    "provider": "process",
                    "provider_options": _options(scratch),
                    "node_services": {name: _declaration(data, cpus=1) for name in ("judge-a", "judge-b")},
                },
            }
        ]
    )
    workers: list[set[int]] = []

    class Workflow:  # its rollout workers start after the node services took their CPUs
        async def warm_up(self) -> None:
            probe = "import os; print(*sorted(os.sched_getaffinity(0)))"
            output = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout
            workers.append({int(cpu) for cpu in output.split()})

    startup = await RolloutStartup.begin(environments, [Workflow()])
    try:
        await startup.ready()
        judges = [os.sched_getaffinity(pid) for pid in _starts(data)]
        assert len(judges) == 2 and judges[0] and judges[1] and not judges[0] & judges[1]
        assert workers == [os.sched_getaffinity(0)] and not workers[0] & (judges[0] | judges[1])
        assert workers[0] | judges[0] | judges[1] <= allowed
    finally:
        await startup.close()
    assert os.sched_getaffinity(0) == allowed  # the run's CPUs again


async def test_a_staged_copy_that_a_service_reads_outlives_a_new_version_of_its_source(tmp_path: Path):
    source = tmp_path / "problems"
    source.mkdir()
    (source / "answer").write_text("1")
    first = staging.stage(source)
    again = staging.stage(source)
    assert again.path == first.path and (first.path / "answer").read_text() == "1"
    again.release()
    (source / "answer").write_text("22")
    second = staging.stage(source)
    assert second.path != first.path and (second.path / "answer").read_text() == "22"
    assert (first.path / "answer").read_text() == "1"  # still read by the service that holds it
    first.release()
    third = staging.stage(source)
    assert third.path == second.path and not first.path.exists()  # no one held the old one any more
    for copy in (second, third):
        copy.release()


# --- Harbor stacks that reach a node service --------------------------------------------------------------------


@pytest.fixture
def fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The fixture tasks, fetched with the overlay that serves their judge from the node's judge pool."""

    use_site_cache(monkeypatch, tmp_path / "cache")
    settings: dict[str, Any] = {
        "dataset": str(FIXTURES / "tasks"),
        "lock": str(tmp_path / "lock.json"),
        "overlays": [str(FIXTURES / "overlays" / "judge-pool")],
        "refuse": {"sidecars": False},
    }
    lock = fetch(fetch_config(**settings))
    for task in lock.tasks:
        assert task.refused is None and task.compose is not None
        assert task.compose["services"]["judge"] == {"external": NAME, "scope": {"problem_id": "${PROBLEM_ID}"}}
    return settings


def _service(fetched: dict[str, Any], data: Path, scratch: Path, **options: Any) -> HarborEnvironmentService:
    return HarborEnvironmentService(
        provider="process",
        provider_options=_options(scratch),
        node_services={NAME: _declaration(data)},
        **fetched,
        **options,
    )


async def _graded(service: HarborEnvironmentService, task_id: str, *, answer: str | None = None):
    session = await service.open(await service.get_task(task_id), rollout_id=f"rollout-{task_id}")
    try:
        if answer is None:
            await session.run_solution()
        else:
            assert session.sandbox is not None
            await session.sandbox.exec(["/bin/sh", "-c", f'echo {answer} > "$PLATOON_SANDBOX_ROOT/app/answer.txt"'])
        return await session.grade(), session
    finally:
        await session.close()


async def test_stacks_reach_the_node_service_through_scopes_of_their_own_tasks(runner, fetched, data, scratch):
    service = _service(fetched, data, scratch)
    await service.check_tasks("train", ["sum-a", "sum-b"])
    (right_a, session), (right_b, _), (wrong, _) = await asyncio.gather(
        _graded(service, "sum-a"), _graded(service, "sum-b"), _graded(service, "sum-a", answer="7")
    )
    assert (right_a.status, right_a.rewards) == ("scored", {"reward": 1.0}), right_a.reason
    assert (right_b.status, right_b.rewards) == ("scored", {"reward": 1.0}), right_b.reason
    assert (wrong.status, wrong.rewards) == ("scored", {"reward": 0.0}), wrong.reason
    assert right_a.metadata["node_services"] == {"judge": {"node_service": NAME, "generation": 1, "served_by": [1]}}
    assert set(session.stack.services) == {"main"}  # the judge is the node's, not the stack's
    assert len(_starts(data)) == 1


async def test_the_agents_reach_nothing_of_the_node_service_but_their_scope(runner, fetched, data, scratch):
    service = _service(fetched, data, scratch)
    session = await service.open(await service.get_task("sum-a"), rollout_id="rollout")
    try:
        probe = (
            "import sys, urllib.request\n"
            "try:\n"
            "    urllib.request.urlopen(sys.argv[1], timeout=10)\n"
            "except urllib.error.HTTPError as exc:\n"
            "    print(exc.code)\n"
        )
        assert session.stack is not None and session.sandbox is not None
        admin = session.stack.rewrite("http://judge:8081/admin", service="main")
        result = await session.sandbox.exec(["python3", "-c", probe, admin])
        assert result.stdout.strip() == "404", result
    finally:
        await session.close()


async def test_a_node_service_that_fails_while_the_rollout_uses_it_makes_it_invalid(runner, fetched, data, scratch):
    service = _service(fetched, data, scratch)
    session = await service.open(await service.get_task("sum-a"), rollout_id="rollout")
    try:
        assert session.sandbox is not None
        await session.sandbox.exec(["/bin/sh", "-c", 'echo slow > "$PLATOON_SANDBOX_ROOT/app/answer.txt"'])
        grading = asyncio.ensure_future(session.grade())
        await _until(lambda: _requests(data) == ["sum-a"], 60)
        os.kill(_starts(data)[0], signal.SIGKILL)  # while it grades the tests' request
        outcome = await grading
    finally:
        await session.close()
    assert (outcome.status, outcome.failure_kind) == ("invalid", "node_service_down"), outcome.reason
    assert outcome.metadata["node_service"] == NAME and outcome.metadata["service"] == "judge"


async def test_a_node_service_that_failed_before_the_rollout_used_it_grades_it_on_its_next_start(
    runner, fetched, data, scratch
):
    service = _service(fetched, data, scratch)
    session = await service.open(await service.get_task("sum-a"), rollout_id="rollout")
    try:
        await session.run_solution()
        os.kill(_starts(data)[0], signal.SIGKILL)  # another rollout's submission, say, as this one's agent works
        outcome = await session.grade()
    finally:
        await session.close()
    assert (outcome.status, outcome.rewards) == ("scored", {"reward": 1.0}), outcome.reason
    assert outcome.metadata["node_services"] == {"judge": {"node_service": NAME, "generation": 1, "served_by": [2]}}


async def test_a_stack_whose_node_service_is_not_running_does_not_open(fetched, data, scratch):
    service = _service(fetched, data, scratch)
    with pytest.raises(EnvSetupError, match="is not running on this node") as error:
        await service.open(await service.get_task("sum-a"), rollout_id="rollout")
    assert error.value.failure_kind == "env_setup"


async def test_stacks_that_reach_a_node_service_are_refused_where_it_cannot_be_reached(fetched, data, scratch):
    undeclared = HarborEnvironmentService(provider="process", provider_options=_options(scratch), **fetched)
    with pytest.raises(EnvSetupError, match="reach the node service judge-pool .* which the environment does not"):
        await undeclared.check_tasks("train", ["sum-a"])
    with pytest.raises(EnvSetupError, match="which the environment does not declare") as error:
        await undeclared.open(await undeclared.get_task("sum-a"), rollout_id="rollout")
    assert error.value.failure_kind == "env_config"
    on_host = _service(fetched, data, scratch, stacks="host_network")
    with pytest.raises(EnvSetupError, match="every process of the node would reach that port"):
        await on_host.check_tasks("train", ["sum-a"])


async def test_stacks_that_reach_a_node_service_are_refused_where_the_provider_does_not_say_it_reaches_one(
    fetched, data, scratch, monkeypatch: pytest.MonkeyPatch
):
    # A provider written before stacks reached node services would start the stack without its relay: every rollout
    # would be invalid for want of its judge.  One that does not declare the capability is refused.
    monkeypatch.setattr(ProcessProvider, "capabilities", ProcessProvider.capabilities - {"externals"})
    service = _service(fetched, data, scratch)
    with pytest.raises(EnvSetupError, match="reach no service outside them") as error:
        await service.check_tasks("train", ["sum-a"])
    assert error.value.failure_kind == "env_config"


async def test_a_scope_that_names_a_variable_the_task_lacks_does_not_open(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data: Path, scratch: Path
):
    use_site_cache(monkeypatch, tmp_path / "cache")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    text = (FIXTURES / "overlays" / "judge-pool" / "overlay.toml").read_text()
    (overlay / "overlay.toml").write_text(text.replace("${PROBLEM_ID}", "${PROBLEM_NAME}"))
    settings: dict[str, Any] = {
        "dataset": str(FIXTURES / "tasks"),
        "lock": str(tmp_path / "lock.json"),
        "overlays": [str(overlay)],
    }
    fetch(fetch_config(**settings))
    service = _service(settings, data, scratch)
    with pytest.raises(EnvSetupError, match=r"names \$\{PROBLEM_NAME\}, which the task's") as error:
        await service.open(await service.get_task("sum-a"), rollout_id="rollout")
    assert error.value.failure_kind == "env_config"


async def test_without_the_overlay_the_privileged_judge_is_refused_at_fetch(tmp_path: Path, monkeypatch):
    use_site_cache(monkeypatch, tmp_path / "cache")
    lock = fetch(fetch_config(str(FIXTURES / "tasks"), lock=tmp_path / "lock.json", refuse={"sidecars": False}))
    assert {task.name: (task.refused or "").split(":")[0] for task in lock.tasks} == {
        "sum-a": "compose",
        "sum-b": "compose",
    }
