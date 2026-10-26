"""Harbor stacks on Docker, end to end (RFC 0010, M6's exit check), with docker compose.

- Harbor's hello-mcp (vendored in ``tests/envs/fixtures/harbor-compose``), whose two images ``platoon prepare`` builds
  with the docker builder, its sidecar's from its build context, scores its oracle through ``platoon-oracle`` (1.0)
  and its no-op through ``platoon-noop`` (0.0), on the rollout path training takes; its solution and tests install
  packages, so it needs the network.
- ``app-redis`` (``tests/envs/fixtures/harbor-stacks-docker``) runs a ``redis:7-alpine`` sidecar whose entrypoint
  takes its data directory and drops root for the ``redis`` user, in a stack without a network; a sidecar that is
  killed, or that the agents shut down, makes the outcome invalid, and one that never turns healthy fails the start.
- ``app-privileges``, fetched with ``refuse: {privileges: false}``: the sidecar given ``NET_ADMIN`` and the privileged
  one change their network, and ``main`` cannot; what a sidecar wrote as another user is deleted with the stack.
- The stack fixtures written for the process provider (``tests/envs/fixtures/harbor-stacks``) score as there: a
  sidecar's log directory, an init job, a separate verifier with collect hooks and a sidecar's artifacts.
- ``main`` reaches the model gateway's Unix socket through its route; a stack's egress follows its network; and a
  SIGKILLed worker's stack is removed by a reap of the worker's label.
- The tool plane serves the tasks' MCP servers (hello-mcp's, and the fixtures ``mcp-http`` and ``mcp-sse``) through a
  relay of the stack's: a function harness solves each through the plane (``tests/mcp_solver.py``), with its calls in
  its trajectory's tool-call records, and a tool that an agent's access level hides cannot be called; a CLI harness in
  ``main`` solves each at the URL that ``{task_mcp_servers}`` gives it.
- After each test, nothing of it is left: no container, network or volume, and no project directory.

It needs Docker with the compose plugin, and ``PLATOON_TEST_DOCKER_STACKS``: ``1`` runs it where Docker is, and
``require``, which CI's ``docker-stacks`` job sets, fails it where Docker is not, rather than skip it.  Without either
value, or without Docker at ``1``, it is skipped, saying why.  It pulls and builds images, so its first run takes
minutes.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

WANTED = os.environ.get("PLATOON_TEST_DOCKER_STACKS", "")


def _docker_problem() -> str | None:
    """Why Docker cannot run stacks here: no CLI, no daemon it reaches, no compose plugin; None when it can."""

    docker = shutil.which("docker")
    if docker is None:
        return "there is no docker CLI on PATH"
    for command in (["info", "--format", "{{.ServerVersion}}"], ["compose", "version"]):
        try:
            result = subprocess.run([docker, *command], capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"`docker {command[0]}` fails: {exc}"
        if result.returncode != 0:
            return f"`docker {command[0]}` fails: {(result.stderr or result.stdout).strip()[-300:]}"
    return None


if WANTED not in ("1", "require"):
    pytest.skip(
        "set PLATOON_TEST_DOCKER_STACKS=1 to run Harbor stacks on Docker end to end: it pulls and builds images (CI's "
        "docker-stacks job runs it)",
        allow_module_level=True,
    )
if (PROBLEM := _docker_problem()) is not None:
    if WANTED == "require":
        pytest.fail(f"PLATOON_TEST_DOCKER_STACKS=require, and Docker cannot run stacks here: {PROBLEM}", pytrace=False)
    pytest.skip(f"Docker cannot run stacks here: {PROBLEM} (CI's docker-stacks job runs it)", allow_module_level=True)

import yaml  # noqa: E402
from mcp_solver import (  # noqa: E402
    DIRECT,
    ROTATE_TOOL,
    SECRET_TOOL,
    SOLVER,
    hidden_from_read_only,
    solved,
    solved_directly,
)
from no_model import no_model_gateway  # noqa: E402
from site_cache import use_site_cache  # noqa: E402

from platoon.config_defs import RolloutConfig  # noqa: E402
from platoon.envs.harbor import HarborEnvironmentService, HarborSession  # noqa: E402
from platoon.envs.service import EnvSetupError, SessionNeeds  # noqa: E402
from platoon.harness.rollout import run_harness_rollout  # noqa: E402
from platoon.harness.runner import HarnessRunner  # noqa: E402
from platoon.prepare import load_environments, prepare, summary  # noqa: E402
from platoon.sandbox.docker import DockerProvider  # noqa: E402
from platoon.task import Task  # noqa: E402
from platoon.train.components import EnvironmentRoute  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "envs" / "fixtures"
TASKS = {
    "hello-mcp": FIXTURES / "harbor-compose" / "upstream" / "examples" / "hello-mcp",
    **{
        name: FIXTURES / "harbor-stacks-docker" / "tasks" / name
        for name in ("app-redis", "app-redis-crash", "app-redis-unhealthy", "app-privileges")
    },
    **{
        name: FIXTURES / "harbor-stacks" / "tasks" / name
        for name in ("app-kv", "app-kv-seeded", "sidecar-artifacts", "mcp-http", "mcp-sse")
    },
}
# Whether the internet answers from main: a connection to the Python package index.
REACH_THE_WEB = "import socket; socket.create_connection(('pypi.org', 443), timeout=15).close()"
# Whether a service may change its network: its loopback's MTU (SIOCSIFMTU), which takes NET_ADMIN.
CHANGE_THE_NETWORK = (
    "import fcntl, socket, struct; fcntl.ioctl(socket.socket(), 0x8922, struct.pack('16si', b'lo', 1400))"
)
# A worker that opens app-redis's stack under its label and waits to be killed: $1 the label, $2 the service's settings.
WORKER = """
import asyncio, json, sys
from platoon.envs.harbor import HarborEnvironmentService
from platoon.sandbox import cleanup

async def main():
    cleanup.set_labels({"platoon.worker": sys.argv[1]})
    service = HarborEnvironmentService(**json.loads(sys.argv[2]))
    await service.open(await service.get_task("app-redis"), rollout_id="doomed")
    print("ready", flush=True)
    await asyncio.sleep(3600)

asyncio.run(main())
"""

pytestmark = pytest.mark.asyncio


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=300, check=True)


def _objects() -> dict[str, set[str]]:
    """Every container, network and volume Docker holds now."""

    return {
        "containers": set(_docker("ps", "--all", "--quiet", "--no-trunc").stdout.split()),
        "networks": set(_docker("network", "ls", "--quiet", "--no-trunc").stdout.split()),
        "volumes": set(_docker("volume", "ls", "--quiet").stdout.split()),
    }


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A site cache and node scratch of the module's own, where the stacks' project directories go too."""

    root = tmp_path_factory.mktemp("docker-stacks")
    with pytest.MonkeyPatch.context() as patch:
        use_site_cache(patch, root / "cache")
        patch.setenv("PLATOON_NODE_SCRATCH", str(root / "scratch"))
        yield root


@pytest.fixture(scope="module")
def settings(site: Path) -> dict[str, Any]:
    """The Harbor entry's settings, once ``platoon prepare`` fetched its tasks and built their images."""

    dataset = site / "dataset"
    for name, source in TASKS.items():
        shutil.copytree(source, dataset / name)
    kwargs = {
        "dataset": str(dataset),
        "refuse": {"sidecars": False, "privileges": False},  # app-privileges asks for some
        "provider": "docker",
        "provider_options": {"stack_root": str(site / "stacks")},
    }
    config = site / "environments.yaml"
    entry = {"name": "stacks", "service": "harbor", "service_kwargs": kwargs, "harness": "platoon-oracle"}
    config.write_text(yaml.safe_dump({"environments": [entry]}))
    lines: list[str] = []
    steps = prepare(load_environments(config), out=lines.append)
    line, code = summary(steps, dry_run=False)
    assert code == 0, [*lines, line]
    # hello-mcp's two images, main's and its sidecar's, from its build context; the registry images that the other
    # tasks' sidecars run are pinned, by digest with their configuration, and pulled.
    assert [(step.step, step.status) for step in steps] == [
        ("dataset", "done"),
        ("images", "done"),
        ("registry images", "done"),
    ], lines
    assert "built 2 with the docker builder" in lines[1], lines
    return kwargs


@pytest.fixture(autouse=True)
def nothing_left(settings: dict[str, Any]) -> Iterator[None]:
    """Each test leaves no container, network, volume or project directory behind."""

    before = _objects()
    yield
    left = {kind: sorted(now - before[kind]) for kind, now in _objects().items()}
    stacks = Path(settings["provider_options"]["stack_root"])
    left["directories"] = sorted(path.name for path in stacks.iterdir()) if stacks.exists() else []
    assert left == {"containers": [], "networks": [], "volumes": [], "directories": []}


async def _open(settings: dict[str, Any], name: str) -> HarborSession:
    service = HarborEnvironmentService(**settings)
    return await service.open(await service.get_task(name), rollout_id=f"docker-{name}")


def _main(session: HarborSession) -> Any:
    """The session's main sandbox, where the agents run."""

    assert session.sandbox is not None
    return session.sandbox


async def _graded(settings: dict[str, Any], name: str, *, solve: bool) -> Any:
    session = await _open(settings, name)
    try:
        if solve:
            await session.run_solution()
        return await session.grade()
    finally:
        await session.close()


async def _rollout(settings: dict[str, Any], harness: str, task: str, tmp_path: Path) -> dict[str, Any]:
    route = EnvironmentRoute(
        name="stacks", service="harbor", service_kwargs=settings, harness=harness, resolve_task=True
    )
    runner = HarnessRunner(
        logs_dir=tmp_path / "logs",
        state_root=tmp_path / "state",
        sandbox_state_root=str(tmp_path / "sandbox-state"),
        kill_grace_s=0.2,
    )
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False)
    collection = await run_harness_rollout(
        route.attach(Task(id=task)), config, gateway=no_model_gateway(), runner=runner
    )
    assert isinstance(collection, dict)  # return_dict
    (root,) = collection["trajectories"].values()
    return root


# --- hello-mcp, on the rollout path --------------------------------------------------------------------------------


async def test_hello_mcp_scores_its_oracle_and_its_no_op(settings: dict[str, Any], tmp_path: Path):
    oracle = await _rollout(settings, "platoon-oracle", "hello-mcp", tmp_path / "oracle")
    noop = await _rollout(settings, "platoon-noop", "hello-mcp", tmp_path / "noop")

    assert (oracle["misc"]["outcome"]["status"], oracle["reward"]) == ("scored", 1.0), oracle["misc"]
    assert oracle["misc"]["oracle"]["exit_code"] == 0
    assert (noop["misc"]["outcome"]["status"], noop["reward"]) == ("scored", 0.0), noop["misc"]


# --- task MCP servers, through the tool plane ---------------------------------------------------------------------


@pytest.mark.parametrize("task", ["hello-mcp", "mcp-http", "mcp-sse"])
async def test_a_function_harness_solves_the_mcp_tasks_through_the_tool_plane(
    settings: dict[str, Any], tmp_path: Path, task: str
):
    assert solved(await _rollout(settings, SOLVER, task, tmp_path))  # with its tool calls in the trajectory


@pytest.mark.parametrize("task", ["hello-mcp", "mcp-http", "mcp-sse"])
async def test_a_harness_in_main_solves_the_mcp_tasks_at_the_urls_it_is_given(
    settings: dict[str, Any], tmp_path: Path, task: str
):
    assert solved_directly(await _rollout(settings, DIRECT, task, tmp_path))  # {task_mcp_servers}, from main


async def test_a_tool_the_agents_access_hides_cannot_be_called(settings: dict[str, Any]):
    """hello-mcp's get_secret says nothing of being read-only, nor does the fixtures' rotate_secret: an agent with
    ``read_only`` access sees neither, and cannot call them; the relay goes with the stack (``nothing_left``)."""

    for task, tool in (("hello-mcp", SECRET_TOOL), ("mcp-sse", ROTATE_TOOL)):
        session = await _open(settings, task)
        try:
            assert await hidden_from_read_only(session, tool)
            stack = session.stack
            assert stack is not None
            relays = _docker("ps", "--quiet", "--filter", f"label=platoon.stack={stack.id}", "--filter", "name=relay")
            assert len(relays.stdout.split()) == 1  # the listing reached the server through one relay
        finally:
            await session.close()


# --- app-redis -----------------------------------------------------------------------------------------------------


async def test_app_redis_runs_its_sidecar_as_its_image_does(settings: dict[str, Any]):
    session = await _open(settings, "app-redis")
    try:
        stack = session.stack
        assert stack is not None and stack.network == "none"
        # The entrypoint, run as root, took /data and became the redis user (uid 999) for the server, PID 1.
        status = await stack.services["cache"].exec(["sh", "-c", "grep '^Uid:' /proc/1/status; stat -c %u /data"])
        assert status.stdout.split() == ["Uid:", "999", "999", "999", "999", "999"], status
        offline = await _main(session).exec(["python3", "-c", REACH_THE_WEB])
        assert offline.exit_code != 0  # a stack without a network reaches only its own services
        noop = await session.grade()
        assert (noop.status, noop.reward) == ("scored", 0.0), noop
        assert (await session.run_solution()).exit_code == 0
        oracle = await session.grade()
        assert (oracle.status, oracle.reward) == ("scored", 1.0), oracle
        assert oracle.metadata["services"] == {
            "main": {"run_as": "subids"},
            "cache": {"run_as": "subids", "state": "healthy"},
        }
    finally:
        await session.close()


async def test_a_killed_sidecar_makes_the_outcome_invalid(settings: dict[str, Any]):
    session = await _open(settings, "app-redis")
    try:
        assert (await session.run_solution()).exit_code == 0
        assert session.stack is not None
        _docker("kill", "--signal", "KILL", session.stack.services["cache"].id)
        outcome = await session.grade()
    finally:
        await session.close()

    assert (outcome.status, outcome.failure_kind, outcome.trainable) == ("invalid", "sidecar_unhealthy", False)
    assert outcome.metadata["sidecar"] == "cache" and outcome.metadata["sidecar_exit_code"] == 137, outcome


async def test_a_sidecar_the_agents_shut_down_makes_the_outcome_invalid(settings: dict[str, Any]):
    outcome = await _graded(settings, "app-redis-crash", solve=True)

    assert (outcome.status, outcome.failure_kind) == ("invalid", "sidecar_unhealthy"), outcome
    assert outcome.reason.startswith("the sidecar cache exited with status 0 before the tests")


async def test_a_sidecar_that_never_turns_healthy_fails_the_start(settings: dict[str, Any]):
    with pytest.raises(EnvSetupError, match="the service cache never turned healthy") as error:
        await _open(settings, "app-redis-unhealthy")

    assert error.value.failure_kind == "sidecar_start"


# --- privileges ----------------------------------------------------------------------------------------------------


async def test_privileges_reach_only_the_services_given_them(settings: dict[str, Any]):
    session = await _open(settings, "app-privileges")
    try:
        stack = session.stack
        assert stack is not None
        for service, granted in (("main", False), ("router", True), ("judge", True)):
            changed = await stack.services[service].exec(["python3", "-c", CHANGE_THE_NETWORK])
            assert (changed.exit_code == 0) is granted, (service, changed)
        # As another user, in the volume judge shares with main: which this process cannot delete, but the teardown's
        # container that runs rm as root does (the module's nothing_left looks).
        owned = "mkdir /shared/owned && touch /shared/owned/file && chown -R 12345:12345 /shared/owned"
        written = await stack.services["judge"].exec(["sh", "-c", f"{owned} && chmod 700 /shared/owned"])
        assert written.exit_code == 0, written
        with pytest.raises(PermissionError):
            os.listdir(stack.volume_path("shared") / "owned")
    finally:
        await session.close()


# --- the process provider's stack fixtures, on Docker ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["app-kv", "app-kv-seeded", "sidecar-artifacts"])
async def test_the_stack_fixtures_score_their_oracles_and_their_no_ops(settings: dict[str, Any], name: str):
    noop = await _graded(settings, name, solve=False)
    oracle = await _graded(settings, name, solve=True)

    assert (noop.status, noop.reward) == ("scored", 0.0), noop
    assert (oracle.status, oracle.reward) == ("scored", 1.0), oracle


class _Pong(http.server.BaseHTTPRequestHandler):
    """Answers every GET with ``pong``: the model gateway, as far as a route can tell."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "4")
        self.end_headers()
        self.wfile.write(b"pong")

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


async def test_main_reaches_the_model_gateway_through_its_route(settings: dict[str, Any], tmp_path: Path):
    """As a harness in main does, in a stack without a network: the socket's directory is bound into main read-only,
    and a forwarder there relays 127.0.0.1:18080 to it."""

    sockets = tmp_path / "gateway"
    sockets.mkdir()
    server = _UnixServer(str(sockets / "gateway.sock"), _Pong)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    ask = "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18080/', timeout=15).read().decode())"
    try:
        service = HarborEnvironmentService(**settings)
        needs = SessionNeeds(upstreams={"gateway": f"unix://{sockets}/gateway.sock"})
        session = await service.open(await service.get_task("app-redis"), rollout_id="docker-routed", needs=needs)
        try:
            assert _main(session).routes == {"gateway": "http://127.0.0.1:18080"}
            reply = await _main(session).exec(["python3", "-c", ask])
            assert reply.stdout.strip() == "pong", reply
        finally:
            await session.close()
    finally:
        server.shutdown()
        server.server_close()


async def test_a_stacks_egress_follows_its_network(settings: dict[str, Any]):
    session = await _open(settings, "app-kv")
    reach_kv = "import os, urllib.request; urllib.request.urlopen(os.environ['KV_URL'] + '/answer', timeout=10)"
    try:
        stack = session.stack
        assert stack is not None and stack.network == "egress"
        assert (await _main(session).exec(["python3", "-c", REACH_THE_WEB])).exit_code == 0
        await stack.set_network("none")
        assert (await _main(session).exec(["python3", "-c", REACH_THE_WEB])).exit_code != 0
        kv = await _main(session).exec(["python3", "-c", reach_kv])
        assert "HTTP Error 404" in kv.stderr  # kv answers by name, with nothing stored yet
        await stack.set_network("egress")
        assert (await _main(session).exec(["python3", "-c", REACH_THE_WEB])).exit_code == 0
    finally:
        await session.close()


# --- reaping ---------------------------------------------------------------------------------------------------------


async def test_a_sigkilled_workers_stack_is_removed_by_a_reap_of_its_label(settings: dict[str, Any], tmp_path: Path):
    label = f"docker-stacks-{os.getpid()}"
    worker = subprocess.Popen(
        [sys.executable, "-c", WORKER, label, json.dumps(settings)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert worker.stdout is not None
        line = await asyncio.wait_for(asyncio.to_thread(worker.stdout.readline), timeout=300)
        assert line.strip() == "ready", line
    finally:
        worker.send_signal(signal.SIGKILL)
        worker.wait()
    selector = f"label=platoon.worker={label}"
    assert len(_docker("ps", "--all", "--quiet", "--filter", selector).stdout.split()) == 2  # main and cache

    provider = DockerProvider(stack_root=settings["provider_options"]["stack_root"])
    assert await provider.reap({"platoon.worker": label}) == 1

    assert _docker("ps", "--all", "--quiet", "--filter", selector).stdout.split() == []
    assert _docker("network", "ls", "--quiet", "--filter", selector).stdout.split() == []
