"""Routes from a sandbox to host services: upstream validation, the forwarder, and the providers' plumbing.

The process provider runs the forwarder for real (a host process on a free loopback port, as in a sandbox that shares
the host's network), so a request from inside the sandbox crosses the route to a Unix-socket server on the host.
Docker and hpc-sandbox are faked: their tests check the binds and the commands that set the route up.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from platoon.gateway.server import socket_directory
from platoon.sandbox import Mount, SandboxSpec, get_provider
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.routes import (
    FIRST_PORT,
    FORWARDER,
    ROUTE_MARKER,
    RouteError,
    SandboxRoutes,
    parse_upstreams,
    python_candidates,
)

FETCH = "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1] + '/hello', timeout=10).read().decode())"


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    """Each test starts with an empty sandbox registry and no process labels."""

    monkeypatch.setattr(sandbox_cleanup, "_live", {})
    monkeypatch.setattr(sandbox_cleanup, "_labels", {})
    monkeypatch.setattr(sandbox_cleanup, "_mirror_path", None)
    monkeypatch.setattr(sandbox_cleanup, "_shut_down", False)


@pytest_asyncio.fixture
async def unix_server() -> AsyncIterator[str]:
    """An HTTP server on a Unix socket in a private directory; yields the socket's path.

    The directory is the gateway's own (:func:`socket_directory`), short enough for an ``AF_UNIX`` address wherever
    the tests' temporary directory is: pytest's paths under xdist and a job's scratch can exceed 108 bytes."""

    directory = socket_directory()
    path = str(directory / "gw.sock")

    async def hello(request: web.Request) -> web.Response:
        return web.json_response({"host": request.headers.get("Host")})

    app = web.Application()
    app.router.add_get("/hello", hello)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.UnixSite(runner, path).start()
        yield path
    finally:
        await runner.cleanup()
        shutil.rmtree(directory, ignore_errors=True)


# --------------------------------------------------------------------------- upstreams


def test_upstreams_are_validated_in_name_order():
    upstreams = parse_upstreams({"zeta": "http://10.0.0.1:80", "gateway": "unix:///run/gw/gw.sock"})

    assert [(upstream.name, upstream.socket) for upstream in upstreams] == [
        ("gateway", "/run/gw/gw.sock"),
        ("zeta", None),
    ]
    assert upstreams[0].target == "/run/platoon/gateway"


@pytest.mark.parametrize(
    ("upstreams", "message"),
    [
        ({"-bad": "unix:///s/gw.sock"}, "must match"),
        ({"gateway": "tcp://host:1"}, "unix://, http:// or https://"),
        ({"gateway": "unix://relative/gw.sock"}, "absolute path"),
        ({"gateway": "unix:///"}, "absolute path"),
        ({"gateway": "http:///v1"}, "no host"),
    ],
)
def test_bad_upstreams_are_refused(upstreams, message):
    with pytest.raises(ValueError, match=message):
        parse_upstreams(upstreams)


def test_a_sandbox_on_the_host_network_needs_no_forwarder():
    spec = SandboxSpec(image="x", upstreams={"gateway": "unix:///s/gw.sock", "judge": "http://127.0.0.1:9000"})
    routes = SandboxRoutes(spec, own_network=True)

    assert routes.forwarded == () and routes.mounts() == []
    assert routes.routes == {"judge": "http://127.0.0.1:9000"}  # a TCP upstream is its own route


def test_a_sandbox_without_a_network_binds_the_socket_directory_and_refuses_tcp_upstreams():
    spec = SandboxSpec(image="x", network="none", upstreams={"gateway": "unix:///s/dir/gw.sock"})
    routes = SandboxRoutes(spec, own_network=True)

    assert routes.forwarded == ("gateway",)
    assert routes.mounts() == [Mount(source="/s/dir", target="/run/platoon/gateway", read_only=True)]
    assert routes.routes == {}  # until its forwarder runs

    with pytest.raises(ValueError, match="cannot reach the TCP upstream"):
        SandboxRoutes(dataclasses.replace(spec, upstreams={"judge": "http://h:1"}), own_network=True)


def test_the_forwarder_prefers_a_toolchain_python():
    mounts = [Mount("/t/oh", "/opt/oh"), Mount("/data", "/data", read_only=False), Mount("/t/x", "/opt/x")]

    assert python_candidates(mounts) == ["/opt/oh/bin/python3", "/opt/x/bin/python3", "python3", "python"]


# --------------------------------------------------------------------------- the forwarder script


def test_the_forwarder_relays_to_a_socket_that_may_appear_later(tmp_path: Path):
    """The forwarder publishes its port at once, connects per connection, and keeps long socket paths short."""

    deep = tmp_path / ("d" * 60) / ("e" * 60)  # beyond an AF_UNIX address when absolute
    deep.mkdir(parents=True)
    socket_path, port_file = deep / "gw.sock", tmp_path / "port"
    forwarder = subprocess.Popen(
        [
            sys.executable,
            "-I",
            str(FORWARDER),
            "--socket",
            str(socket_path),
            "--port",
            "0",
            "--port-file",
            str(port_file),
        ]
    )
    server = None
    try:
        deadline = time.monotonic() + 30
        while not port_file.exists():
            assert forwarder.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        port = int(port_file.read_text())

        refused = subprocess.run([sys.executable, "-c", FETCH, f"http://127.0.0.1:{port}"], capture_output=True)
        assert refused.returncode != 0  # no socket yet: the connection is closed, the forwarder lives on
        assert forwarder.poll() is None

        server = subprocess.Popen(
            [sys.executable, "-c", _UNIX_HTTP_SERVER, str(socket_path)], stdout=subprocess.PIPE, cwd=deep
        )
        assert server.stdout is not None and server.stdout.readline().strip() == b"ready"
        fetched = subprocess.run(
            [sys.executable, "-c", FETCH, f"http://127.0.0.1:{port}"], capture_output=True, text=True, check=True
        )
        assert json.loads(fetched.stdout) == {"host": f"127.0.0.1:{port}"}
    finally:
        for process in (forwarder, server):
            if process is not None:
                process.kill()
                process.wait()


# A minimal HTTP server on a Unix socket, bound by its name from inside its directory (its path is too long).
_UNIX_HTTP_SERVER = """
import http.server, json, os, socketserver, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"host": self.headers.get("Host")}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
    def address_string(self):
        return "unix"
class Server(socketserver.UnixStreamServer):
    pass
server = Server(os.path.basename(sys.argv[1]), Handler)
print("ready", flush=True)
server.serve_forever()
"""


# --------------------------------------------------------------------------- the process provider


@pytest.mark.asyncio
async def test_a_process_sandbox_without_a_network_reaches_the_host_through_its_route(tmp_path, unix_server):
    provider = get_provider("process", root=str(tmp_path / "sandboxes"))
    spec = SandboxSpec(image="local", network="none", upstreams={"gateway": f"unix://{unix_server}"})

    sandbox = await provider.start(spec)
    try:
        route = sandbox.routes["gateway"]
        assert route.startswith("http://127.0.0.1:")
        result = await sandbox.exec(["python3", "-c", FETCH, route])
        assert result.exit_code == 0, result.stderr
        assert json.loads(result.stdout) == {"host": route.removeprefix("http://")}

        # The forwarder carries the infrastructure marker, not an agent's.
        pid = (await sandbox.exec("cat /tmp/platoon-route/gateway.pid")).stdout.strip()
        environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        assert ROUTE_MARKER.encode() in environ
    finally:
        await sandbox.stop()
    assert not sandbox_cleanup.live_sandboxes()
    deadline = time.monotonic() + 10
    while _running(int(pid)):  # it died with the sandbox
        assert time.monotonic() < deadline
        await asyncio.sleep(0.05)


def _running(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


@pytest.mark.asyncio
async def test_a_killed_forwarder_is_started_again_on_its_port(tmp_path, unix_server):
    provider = get_provider("process", root=str(tmp_path / "sandboxes"))
    sandbox = await provider.start(
        SandboxSpec(image="local", network="none", upstreams={"gateway": f"unix://{unix_server}"})
    )
    try:
        route = sandbox.routes["gateway"]
        assert await sandbox.ensure_routes() == {"gateway": route}  # alive: left alone
        pid = int((await sandbox.exec("cat /tmp/platoon-route/gateway.pid")).stdout)
        os.kill(pid, signal.SIGKILL)  # the policy killed it

        assert await sandbox.ensure_routes() == {"gateway": route}
        assert int((await sandbox.exec("cat /tmp/platoon-route/gateway.pid")).stdout) != pid
        result = await sandbox.exec(["python3", "-c", FETCH, route])
        assert result.exit_code == 0, result.stderr
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_a_route_outlives_the_workdir_the_agent_removed(tmp_path, unix_server):
    """The route's own commands run in "/", as the runner's do: once the agent has removed its workdir, where no
    command starts, the runner's check before a launch still finds the route alive, and starts it again after the
    policy killed it."""

    provider = get_provider("process", root=str(tmp_path / "sandboxes"))
    sandbox = await provider.start(
        SandboxSpec(image="local", network="none", upstreams={"gateway": f"unix://{unix_server}"})
    )
    try:
        route = sandbox.routes["gateway"]
        shutil.rmtree(sandbox.host_path(sandbox.spec.workdir))
        with pytest.raises(OSError):  # the process provider cannot start a command there
            await sandbox.exec("true")

        assert await sandbox.ensure_routes() == {"gateway": route}  # alive: left alone
        pid = int((await sandbox.exec("cat /tmp/platoon-route/gateway.pid", cwd="/")).stdout)
        os.kill(pid, signal.SIGKILL)
        assert await sandbox.ensure_routes() == {"gateway": route}  # started again
        assert int((await sandbox.exec("cat /tmp/platoon-route/gateway.pid", cwd="/")).stdout) != pid
        result = await sandbox.exec(["python3", "-c", FETCH, route], cwd="/")
        assert result.exit_code == 0, result.stderr
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_a_sandbox_without_python_for_the_forwarder_does_not_start(tmp_path, unix_server):
    bin_dir = tmp_path / "bin"  # sh and the tools of the start script, but no Python
    bin_dir.mkdir()
    for tool in ("sh", "mkdir", "rm", "cat", "sleep"):
        found = next(Path(directory, tool) for directory in ("/bin", "/usr/bin") if Path(directory, tool).exists())
        (bin_dir / tool).symlink_to(found)
    provider = get_provider("process", root=str(tmp_path / "sandboxes"))
    # The sandbox sees those tools at a path of its own: the process provider translates env values under its mapped
    # roots, /tmp among them, and pytest's tmp_path is under /tmp on most Linux machines.
    spec = SandboxSpec(
        image="local",
        network="none",
        env={"PATH": "/opt/no-python/bin"},
        mounts=[Mount(str(bin_dir), "/opt/no-python/bin")],
        upstreams={"gateway": f"unix://{unix_server}"},
    )

    with pytest.raises(RouteError, match="no Python 3"):
        await provider.start(spec)
    assert list((tmp_path / "sandboxes").iterdir()) == []  # the sandbox was stopped
    assert not sandbox_cleanup._live


@pytest.mark.asyncio
async def test_a_restored_sandbox_gets_its_own_forwarder(tmp_path, unix_server):
    provider = get_provider("process", root=str(tmp_path / "sandboxes"))
    spec = SandboxSpec(image="local", network="none", upstreams={"gateway": f"unix://{unix_server}"})
    sandbox = await provider.start(spec)
    try:
        restored = await provider.restore(await sandbox.snapshot(), spec)
        try:
            assert restored.routes["gateway"] != sandbox.routes["gateway"]  # two forwarders on the host's loopback
            result = await restored.exec(["python3", "-c", FETCH, restored.routes["gateway"]])
            assert result.exit_code == 0, result.stderr
        finally:
            await restored.stop()
    finally:
        await sandbox.stop()


# --------------------------------------------------------------------------- docker (a fake CLI)

# Records each CLI call; answers the route's start script with the port it was asked for.
FAKE_DOCKER = """\
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == "exec" and "platoon-route" in args:
    print(args[args.index("platoon-route") + 4])
"""


@pytest.mark.asyncio
async def test_a_docker_sandbox_without_a_network_gets_the_socket_bind_and_a_forwarder(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "docker"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    binary.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("PATH", f"{binary.parent}:{Path(sys.executable).parent}:/usr/bin:/bin")
    provider = get_provider("docker")
    spec = SandboxSpec(
        image="python:3.12-slim",
        network="none",
        mounts=[Mount("/host/oh", "/opt/oh")],
        upstreams={"gateway": "unix:///host/pgw/gw.sock"},
    )

    sandbox = await provider.start(spec)

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    run = calls[0]
    assert run[0] == "run" and run[run.index("--network") + 1] == "none"
    assert "/host/pgw:/run/platoon/gateway:ro" in run  # read-only: connecting to the socket writes nothing
    copy = next(call for call in calls if call[0] == "cp")
    assert copy[1] == str(FORWARDER) and copy[2].endswith(":/tmp/platoon-route/forwarder.py")
    start = next(call for call in calls if call[0] == "exec" and "platoon-route" in call)
    tail = start[start.index("platoon-route") + 1 :]
    assert tail == [
        "/tmp/platoon-route",
        "gateway",
        "/run/platoon/gateway/gw.sock",
        str(FIRST_PORT),
        "/opt/oh/bin/python3",
        "python3",
        "python",
    ]
    assert sandbox.routes == {"gateway": f"http://127.0.0.1:{FIRST_PORT}"}
    assert "routes" in provider.capabilities
    await sandbox.stop()


# --------------------------------------------------------------------------- hpc-sandbox (a fake package)


def _fake_hpc_sandbox(calls: list[tuple]) -> types.ModuleType:
    module = types.ModuleType("hpc_sandbox")
    module.Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
    module.SandboxSpec = dataclasses.make_dataclass(
        "SandboxSpec", ["image", "workdir", "env", "mounts", "network", "cpus", "memory_mb", "labels"]
    )
    module.ExecResult = dataclasses.make_dataclass("ExecResult", ["exit_code", "stdout", "stderr", "timed_out"])
    module.JanitorReport = dataclasses.make_dataclass(
        "JanitorReport", ["stopped", "removed", "live", "errors"], namespace={}
    )

    class Sandbox:
        id = "hpcsb-1"

        async def exec(self, cmd, **kwargs):
            calls.append(("exec", cmd))
            if isinstance(cmd, list) and "platoon-route" in cmd:
                return module.ExecResult(0, f"{cmd[cmd.index('platoon-route') + 4]}\n", "", False)
            return module.ExecResult(0, "", "", False)

        async def upload(self, src, dst):
            calls.append(("upload", str(src), dst))

        async def stop(self):
            calls.append(("stop",))

    class Provider:
        def __init__(self, **options):
            self.capabilities = frozenset({"snapshot", "isolation", "network_none"})

        async def start(self, spec):
            calls.append(("start", spec))
            return Sandbox()

        async def reap(self, labels):
            calls.append(("reap", labels))
            return module.JanitorReport(["hpcsb-a"], ["hpcsb-a", "hpcsb-b"], ["hpcsb-c"], ["one error"])

    module.SingularityProvider = Provider
    return module


@pytest.mark.asyncio
async def test_an_hpc_sandbox_without_a_network_gets_the_socket_bind_and_a_forwarder(monkeypatch):
    calls: list[tuple] = []
    fake = _fake_hpc_sandbox(calls)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", fake)
    provider = get_provider("singularity", allow_network_none=True)
    spec = SandboxSpec(
        image="x.sif",
        network="none",
        mounts=[Mount("/stage/oh", "/build/oh")],
        labels={"platoon.task": "t"},
        upstreams={"gateway": "unix:///host/pgw/gw.sock"},
    )

    sandbox = await provider.start(spec)

    started = calls[0][1]
    assert started.mounts == [
        fake.Mount("/stage/oh", "/build/oh", True),
        fake.Mount("/host/pgw", "/run/platoon/gateway", True),
    ]
    assert started.network == "none" and started.labels == {"platoon.task": "t"}
    # The sandbox's own processes are listed first, so that grading never mistakes the forwarder for one of them.
    assert calls[1][0] == "exec" and "platoon-list" in calls[1][1]
    assert calls[2] == ("upload", str(FORWARDER), "/tmp/platoon-route/forwarder.py")
    assert calls[3][1][-3:] == ["/build/oh/bin/python3", "python3", "python"]
    assert sandbox.routes == {"gateway": f"http://127.0.0.1:{FIRST_PORT}"}
    assert "routes" in provider.capabilities


@pytest.mark.asyncio
async def test_an_hpc_sandbox_carries_this_process_labels(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    sandbox_cleanup.set_labels({sandbox_cleanup.WORKER_LABEL: "req-1"})
    provider = get_provider("singularity")

    sandbox = await provider.start(SandboxSpec(image="x.sif", labels={"platoon.task": "t"}))

    assert calls[0][1].labels == {"platoon.task": "t", sandbox_cleanup.WORKER_LABEL: "req-1"}
    assert sandbox_cleanup.live_sandboxes() == [sandbox]
    await sandbox.stop()
    assert sandbox_cleanup.live_sandboxes() == []


@pytest.mark.asyncio
async def test_an_hpc_provider_reaps_through_hpc_sandbox(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _fake_hpc_sandbox(calls))
    provider = get_provider("singularity")

    assert await provider.reap({"platoon.worker": "req-1"}) == 2  # stopped or removed, each once
    assert calls == [("reap", {"platoon.worker": "req-1"})]
    with pytest.raises(ValueError, match="needs labels"):
        await provider.reap({})
