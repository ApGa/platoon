"""The host-network fallback (platoon.sandbox.host_network_stacks) over the process provider's sandboxes: services
on the host's network at ports their stack holds, a service that listens beyond the loopback, or whose port another
user took, refused, no stack without a network, providers whose sandboxes run other users refused, a root of this
user's alone, and nothing left after a stop or a reap."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import socket
import stat
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from platoon.sandbox import cleanup, host_network_stacks
from platoon.sandbox.base import SandboxSpec
from platoon.sandbox.host_network_stacks import (
    NO_NETWORK,
    HostNetworkStackProvider,
    beyond_loopback,
    host_network_refusal,
)
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.sandbox.process_stacks import STACK_RECORD
from platoon.sandbox.stacks import Dependency, Healthcheck, ServiceSpec, Stack, StackSpec, StackStartError

pytestmark = pytest.mark.asyncio

# Serves "ok" on WEB_HOST (default 127.0.0.1) at WEB_PORT, which the stack sets as the spec's listen says.
SERVER = (
    "import http.server, os\n"
    "class Answer(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        self.send_response(200); self.send_header('Content-Length', '2'); self.end_headers()\n"
    "        self.wfile.write(b'ok')\n"
    "    def log_message(self, *args): pass\n"
    "address = (os.environ.get('WEB_HOST', '127.0.0.1'), int(os.environ['WEB_PORT']))\n"
    "http.server.ThreadingHTTPServer(address, Answer).serve_forever()\n"
)
FETCH = "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())"


@pytest.fixture
def inner(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


@pytest.fixture
def provider(inner: ProcessProvider, tmp_path: Path) -> HostNetworkStackProvider:
    return HostNetworkStackProvider(inner, root=tmp_path / "stacks")


@pytest.fixture
def before() -> set[int]:
    return {id(sandbox) for sandbox in cleanup.live_sandboxes()}


def _web_stack(*, web_env: dict[str, str] | None = None, network: Any = "egress") -> StackSpec:
    check = Healthcheck(command=(sys.executable, "-c", FETCH, "http://localhost:8000/"), interval_s=0.05, retries=100)
    return StackSpec(
        services={
            "main": ServiceSpec(
                sandbox=SandboxSpec(image="local", env={"WEB": "http://web:8000/", "WEB_ADDRESS": "WEB:8000"}),
                depends_on={"web": Dependency("service_healthy")},
            ),
            "web": ServiceSpec(
                sandbox=SandboxSpec(image="local", env=web_env or {}),
                command=[sys.executable, "-c", SERVER],
                healthcheck=check,  # names its own port on its loopback
                ports=(8000,),
                listen={8000: "WEB_PORT"},
            ),
        },
        network=network,
        labels={"platoon.task": "web"},
    )


def _spec(stack: Stack, service: str) -> SandboxSpec:
    sandbox = stack.services[service]
    assert isinstance(sandbox, ProcessSandbox)
    return sandbox.spec


def _left(inner: ProcessProvider, provider: HostNetworkStackProvider, before: set[int]) -> list[Any]:
    roots = [path for root in (inner.root, provider.root) if root.exists() for path in root.iterdir()]
    return [*roots, *(sandbox for sandbox in cleanup.live_sandboxes() if id(sandbox) not in before)]


async def test_services_run_on_the_host_network_at_ports_their_stack_holds(
    provider: HostNetworkStackProvider, inner: ProcessProvider, before: set[int]
):
    stack = await provider.start_stack(_web_stack())
    try:
        port = int((await stack.connect("web", 8000)).rsplit(":", 1)[1])
        web, main = _spec(stack, "web"), _spec(stack, "main")
        assert web.env["WEB_PORT"] == web.env["PLATOON_PORT_8000"] == str(port)  # its listen variables
        assert main.env["WEB"] == f"http://127.0.0.1:{port}/"
        assert main.env["WEB_ADDRESS"] == f"127.0.0.1:{port}"  # without a scheme, in any case
        fetched = await stack.main.exec([sys.executable, "-c", FETCH, main.env["WEB"]])
        assert fetched.stdout.strip() == "ok", fetched.stderr
        assert stack.run_modes == {"main": "caller", "web": "caller"}
        states = await stack.health()  # the healthcheck reaches the port the stack holds, not 8000
        assert {name: state.status for name, state in states.items()} == {"main": "running", "web": "healthy"}
        assert list(provider.root.iterdir()) == [provider.root / stack.id]  # its directory is the fallback's
        assert (provider.root / stack.id / STACK_RECORD).is_file()
        with pytest.raises(ValueError, match="cannot take away"):
            await stack.set_network("none")
    finally:
        await stack.stop()
    assert _left(inner, provider, before) == []


async def test_a_sidecar_that_listens_beyond_the_loopback_never_starts(
    provider: HostNetworkStackProvider, inner: ProcessProvider, before: set[int]
):
    with pytest.raises(StackStartError, match=r"listens on 0\.0\.0\.0 at port \d+ \(its port 8000\), beyond"):
        await provider.start_stack(_web_stack(web_env={"WEB_HOST": "0.0.0.0"}))
    assert _left(inner, provider, before) == []


async def test_main_listening_beyond_the_loopback_at_its_own_port_never_starts(
    provider: HostNetworkStackProvider, inner: ProcessProvider, before: set[int]
):
    """``main`` is checked as its sidecars are: a port it declares is a port of the node."""

    spec = _web_stack()
    main = spec.services["main"]
    spec.services["main"] = dataclasses.replace(
        main,
        sandbox=dataclasses.replace(main.sandbox, env={**main.sandbox.env, "WEB_HOST": "0.0.0.0"}),
        command=[sys.executable, "-c", SERVER],
        healthcheck=Healthcheck(
            command=(sys.executable, "-c", FETCH, "http://localhost:9000/"), interval_s=0.05, retries=100
        ),
        ports=(9000,),
        listen={9000: "WEB_PORT"},
    )
    with pytest.raises(StackStartError, match=r"main.* listens on 0\.0\.0\.0 at port \d+ \(its port 9000\), beyond"):
        await provider.start_stack(spec)
    assert _left(inner, provider, before) == []


def _tcp_table(path: Path, *rows: tuple[str, int, int]) -> Path:
    """A ``/proc/net/tcp`` that lists a listener at each (address, port, uid) of ``rows``."""

    lines = ["  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode"]
    for number, (address, port, uid) in enumerate(rows):
        word = format(int.from_bytes(socket.inet_aton(address), sys.byteorder), "08X")  # as the kernel writes it
        lines.append(
            f"{number:4d}: {word}:{port:04X} 00000000:0000 0A 00000000:00000000 00:00000000 00000000 {uid:5d}"
            "        0 4242 1 0000000000000000 100 0 0 10 0"
        )
    path.write_text("\n".join(lines) + "\n")
    return path


async def test_another_user_s_listener_at_a_service_s_port_makes_it_unhealthy(
    provider: HostNetworkStackProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A process of another user that took the port its stack holds for a service (claims are visible to the node in
    /proc/net/unix) would get what the stack sends there, the token included: the service is unhealthy."""

    stack = await provider.start_stack(_web_stack())
    try:
        port = int((await stack.connect("web", 8000)).rsplit(":", 1)[1])
        other = os.getuid() + 1
        table = _tcp_table(tmp_path / "tcp", ("127.0.0.1", port, other))
        monkeypatch.setattr(host_network_stacks, "_TABLES", ((str(table), 4),))
        state = (await stack.health(["web"]))["web"]
        assert state.status == "unhealthy"
        assert f"a process of another user (uid {other}) listens at port {port} (its port 8000)" in (state.detail or "")
        assert host_network_stacks.listeners({port}) == [host_network_stacks.Listener(port, "127.0.0.1", other)]
    finally:
        await stack.stop()


async def test_the_root_is_this_user_s_alone_and_a_planted_one_is_refused(
    inner: ProcessProvider, tmp_path: Path, before: set[int]
):
    """In a node's shared scratch: made 0700 whatever the umask, so that other users neither read a stack's volumes
    nor swap them before a runtime binds them; and one that is not this user's directory (a symlink planted where it
    goes) is refused, with nothing made there or reaped from it."""

    umask = os.umask(0o022)
    try:
        provider = HostNetworkStackProvider(inner, root=tmp_path / "shared" / "stacks")
        stack = await provider.start_stack(_web_stack())
        try:
            assert stat.S_IMODE(provider.root.stat().st_mode) == 0o700
            assert stat.S_IMODE((provider.root / stack.id).stat().st_mode) == 0o700
        finally:
            await stack.stop()
    finally:
        os.umask(umask)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "stack-0123456789ab").mkdir(parents=True)
    (elsewhere / "stack-0123456789ab" / STACK_RECORD).write_text(json.dumps({"labels": {"platoon.task": "web"}}))
    (tmp_path / "planted").symlink_to(elsewhere)
    planted = HostNetworkStackProvider(inner, root=tmp_path / "planted")
    with pytest.raises(RuntimeError, match="planted is not a directory of this user's, so no sandbox or stack"):
        await planted.start_stack(_web_stack())
    assert await planted.reap({"platoon.task": "web"}) == 0
    assert [path.name for path in elsewhere.iterdir()] == ["stack-0123456789ab"]  # nothing made, nothing deleted
    assert _left(inner, provider, before) == []


async def test_a_stack_without_a_network_is_refused(provider: HostNetworkStackProvider):
    with pytest.raises(ValueError, match="cannot take away"):
        await provider.start_stack(_web_stack(network="none"))
    spec = _web_stack()
    spec.services["web"].sandbox = dataclasses.replace(spec.services["web"].sandbox, network="none")
    with pytest.raises(ValueError, match="cannot take away"):
        await provider.start_stack(spec)
    assert "cannot take away" in NO_NETWORK


async def test_reaping_deletes_the_directories_of_its_stacks(
    provider: HostNetworkStackProvider, inner: ProcessProvider, tmp_path: Path
):
    stack = await provider.start_stack(_web_stack())
    try:
        # A killed worker's supervisor reaps it as the stack registered it: the fallback, built from its root.
        record = cleanup.SandboxRecord(
            provider=cleanup.provider_ref(provider),
            options={"root": str(provider.root)},
            labels={"platoon.task": "web"},
        )
        assert await cleanup.reap_records([record], timeout_s=30.0) >= 1
        assert list(provider.root.iterdir()) == []
        reaper = HostNetworkStackProvider(root=tmp_path / "stacks")
        with pytest.raises(ValueError, match="built to reap"):
            await reaper.start_stack(_web_stack())
        with pytest.raises(ValueError, match="Reaping needs labels"):
            await reaper.reap({})
    finally:
        await stack.stop()  # what reaping took is gone already: stopping is quiet


async def test_a_start_cancelled_midway_leaves_nothing(
    provider: HostNetworkStackProvider, inner: ProcessProvider, before: set[int]
):
    """Cancelled while main is prepared, once its sidecar runs: the start raises only once what it made is gone."""

    prepared = asyncio.Event()

    async def setup(sandbox: object) -> None:
        prepared.set()
        await asyncio.sleep(30)

    spec = _web_stack()
    spec.services["main"].setup = setup
    starting = asyncio.ensure_future(provider.start_stack(spec))
    await asyncio.wait_for(prepared.wait(), timeout=30)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    for _ in range(100):  # what a cancelled start's threads made, they delete
        if not _left(inner, provider, before):
            break
        await asyncio.sleep(0.05)
    assert _left(inner, provider, before) == []


class _Provider:
    """A built provider whose sandboxes run as your own uid, with the capabilities it is given."""

    name = "fake"

    def __init__(self, capabilities: set[str]) -> None:
        self.capabilities = frozenset(capabilities)
        self.options: dict[str, Any] = {}

    @classmethod
    def root_refusal(cls, options: Any) -> str | None:
        return "it runs every command as your own uid"


async def test_the_fallback_runs_over_providers_whose_sandboxes_share_the_host_network_as_your_uid():
    assert host_network_refusal("process") is None
    docker = host_network_refusal("docker") or ""
    assert "runs a sandbox's commands as root" in docker
    assert docker.endswith("run its stacks with stacks: auto, as compose projects or pods")  # docker has no fakeroot
    fakeroot = host_network_refusal("singularity", {"fakeroot": True}) or ""
    assert "runs a sandbox's commands as root" in fakeroot and "run it without fakeroot" in fakeroot
    assert host_network_refusal("singularity", {}) is None  # without fakeroot: your own uid
    elsewhere = cast(Any, _Provider(set()))
    assert "do not share this host's network" in (host_network_refusal(elsewhere) or "")
    assert host_network_refusal(cast(Any, _Provider({"host_loopback"}))) is None
    with pytest.raises(ValueError, match="do not share this host's network"):
        await HostNetworkStackProvider(elsewhere).start_stack(_web_stack())


async def test_listeners_beyond_the_loopback_are_found_at_the_ports_asked_about():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as everywhere, socket.socket() as loopback:
        everywhere.bind(("0.0.0.0", 0))
        everywhere.listen()
        loopback.bind(("127.0.0.1", 0))
        loopback.listen()
        exposed, local = everywhere.getsockname()[1], loopback.getsockname()[1]
        assert beyond_loopback({exposed, local}) == [(exposed, "0.0.0.0")]
        assert beyond_loopback({local}) == []
    if socket.has_ipv6:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as any6, socket.socket(socket.AF_INET6) as local6:
            try:
                any6.bind(("::", 0))
                local6.bind(("::1", 0))
            except OSError:
                pytest.skip("no IPv6 here")
            any6.listen()
            local6.listen()
            port6, local6_port = any6.getsockname()[1], local6.getsockname()[1]
            assert beyond_loopback({port6, local6_port}) == [(port6, "::")]
