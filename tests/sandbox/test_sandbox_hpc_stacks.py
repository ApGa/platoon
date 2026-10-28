"""Stacks as hpc-sandbox pods (platoon.sandbox.hpc_stacks), against the fake ``hpc_sandbox`` package that
``tests/train/fake_hpc_sandbox`` holds (0.5's API): the pod and services a stack's spec becomes, run modes, relayed
upstreams, volumes in the pod's directory, what an hpc-sandbox without 0.5's pods refuses, and nothing left after a
stop, a failed or cancelled start, or a reap."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import json
import logging
import os
import stat
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any, cast

import pytest

from platoon.sandbox import cleanup, get_provider, hpc_stacks, stack_refusal
from platoon.sandbox.base import SandboxSpec
from platoon.sandbox.hpc import HpcProvider, HpcSandbox
from platoon.sandbox.stacks import (
    MAIN,
    SERVICE_LABEL,
    STACK_LABEL,
    STACKS,
    Dependency,
    ExternalService,
    Healthcheck,
    ServiceSpec,
    StackSpec,
    StackStartError,
    Volume,
)

pytestmark = pytest.mark.asyncio

FAKE_HPC_SANDBOX = Path(__file__).resolve().parents[1] / "train" / "fake_hpc_sandbox" / "hpc_sandbox" / "__init__.py"
GATEWAY = "unix:///run/gateway/gw.sock"


@pytest.fixture
def hpc_sandbox(monkeypatch) -> types.ModuleType:
    """The fake hpc-sandbox 0.5, on a node that nests user namespaces."""

    spec = importlib.util.spec_from_file_location("hpc_sandbox", FAKE_HPC_SANDBOX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(hpc_stacks, "nesting_refusal", lambda: None)
    monkeypatch.setattr(hpc_stacks, "_CALLERS_NOTED", set())
    return module


@pytest.fixture
def provider(hpc_sandbox, tmp_path: Path) -> HpcProvider:
    built = get_provider("singularity", scratch_root=str(tmp_path / "scratch"))
    assert isinstance(built, HpcProvider)
    return built


@pytest.fixture
def before() -> set[int]:
    return set(cleanup._live)


def _registered(before: set[int]) -> list[object]:
    """What was registered for cleanup since ``before``: each one's stack, or None while it is not attached."""

    return [entry.sandbox for token, entry in cleanup._live.items() if token not in before]


def _stack(**services: ServiceSpec) -> StackSpec:
    main = ServiceSpec(sandbox=SandboxSpec(image="docker://main", upstreams={"gateway": GATEWAY}))
    return StackSpec(services={MAIN: main, **services}, labels={"platoon.task": "t"})


def _sidecar(*, ports: tuple[int, ...] = (), user: str | None = None, **fields) -> ServiceSpec:
    return ServiceSpec(
        sandbox=SandboxSpec(image="docker://db", workdir="/"), command=["db"], ports=ports, user=user, **fields
    )


def _only_pod(provider: HpcProvider):
    [pod] = provider.inner.started_pods
    return pod


def _socket(url: str) -> str:
    """The socket a ``unix://`` URL of ``connect`` names: where its link points, for a link."""

    path = url.removeprefix("unix://")
    return os.readlink(path) if os.path.islink(path) else path


async def _until(condition, timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


async def test_the_singularity_provider_runs_stacks_with_hpc_sandbox_s_pods(provider: HpcProvider):
    assert STACKS in provider.capabilities
    assert stack_refusal(cast(Any, provider)) is None
    assert stack_refusal("singularity", {}) is None and stack_refusal("enroot", {}) is None


async def test_a_stack_starts_as_one_pod_whose_services_run_by_their_run_modes(provider: HpcProvider, before):
    spec = _stack(
        kv=_sidecar(ports=(6379,), aliases=("cache",), shm_mb=128, ulimits={"nofile": (1024, 2048)}),
        web=_sidecar(ports=(8000,), user="node", depends_on={"kv": Dependency("service_started")}),
    )
    stack = await provider.start_stack(spec)
    pod = _only_pod(provider)
    try:
        assert pod.spec.network == "egress" and pod.spec.users == "subids" and pod.spec.hosts == ["cache"]
        assert pod.spec.upstreams == {"gateway": GATEWAY} and pod.spec.upstream_ports == {"gateway": 18080}
        assert pod.spec.labels[STACK_LABEL] == stack.id and pod.spec.labels["platoon.task"] == "t"
        assert [service.name for service in pod.started] == ["main", "kv", "web"]
        started = {service.name: service for service in pod.started}
        # main as root, a sidecar whose image runs as root with your subordinate ids, one whose image does not as you.
        assert {name: service.run_as for name, service in started.items()} == {
            "main": "root",
            "kv": "subids",
            "web": "caller",
        }
        assert stack.run_modes == {"main": "root", "kv": "subids", "web": "caller"}
        assert started["kv"].command == ["db"] and started["main"].command is None
        assert started["kv"].shm_mb == 128 and started["kv"].ulimits == {"nofile": (1024, 2048)}
        assert started["kv"].sandbox.labels[SERVICE_LABEL] == "kv"
        assert started["kv"].sandbox.labels[STACK_LABEL] == stack.id
        # main reaches the gateway through the pod's relay, which nothing in the pod can stop.
        assert stack.main.routes == {"gateway": "http://127.0.0.1:18080"}
        assert await cast(HpcSandbox, stack.main).ensure_routes() == {"gateway": "http://127.0.0.1:18080"}
        assert stack.services["kv"].routes == {}
        assert stack.host_loopback is False  # its services share the pod's network, not this host's
        assert stack.rewrite("redis://kv:6379") == "redis://kv:6379"  # names resolve in a pod
        assert _registered(before) == [stack]
    finally:
        await stack.stop()
    assert pod.stops == 1 and not pod.path.exists() and not _registered(before)


async def test_the_relay_s_ports_avoid_the_ports_the_services_declare(provider: HpcProvider):
    spec = _stack(web=_sidecar(ports=(18080, 18081)))
    spec.services[MAIN].sandbox.upstreams["judge"] = "unix:///run/judge/judge.sock"
    stack = await provider.start_stack(spec)
    try:
        assert _only_pod(provider).spec.upstream_ports == {"gateway": 18082, "judge": 18083}
        assert stack.main.routes == {"gateway": "http://127.0.0.1:18082", "judge": "http://127.0.0.1:18083"}
    finally:
        await stack.stop()


async def test_an_external_service_is_relayed_at_its_own_port_and_named_like_the_service_it_serves(
    provider: HpcProvider,
):
    # A node service in place of the stack's judge: main reaches it as judge:8081 (or its alias), as it would the
    # judge's own container, through the pod's relay to the rollout's scope; the other upstreams keep away from 8081.
    spec = _stack(web=_sidecar(ports=(18080,)))
    spec.externals = {"judge": ExternalService(socket="/run/node/scope.sock", port=8081, aliases=("grader",))}
    stack = await provider.start_stack(spec)
    pod = _only_pod(provider)
    try:
        assert pod.spec.upstreams == {"gateway": GATEWAY, "judge": "unix:///run/node/scope.sock"}
        assert pod.spec.upstream_ports == {"gateway": 18081, "judge": 8081}
        assert pod.spec.hosts == ["grader", "judge"]
        assert stack.main.routes == {"gateway": "http://127.0.0.1:18081"}  # the judge is no route of main's
        assert stack.rewrite("http://judge:8081/evaluate") == "http://judge:8081/evaluate"
        assert [service.name for service in pod.started] == ["main", "web"]
    finally:
        await stack.stop()


async def test_an_external_service_that_clashes_with_the_stack_is_refused_before_anything_starts(provider: HpcProvider):
    gateway_named = _stack()
    gateway_named.externals = {"gateway": ExternalService(socket="/run/node/scope.sock", port=8081)}
    with pytest.raises(ValueError, match="named like an upstream of its services"):
        await provider.start_stack(gateway_named)
    declared = _stack(web=_sidecar(ports=(8081,)))
    declared.externals = {"judge": ExternalService(socket="/run/node/scope.sock", port=8081)}
    with pytest.raises(ValueError, match="at port 8081, which the service 'web' declares"):
        await provider.start_stack(declared)
    assert provider.inner.started_pods == []


async def test_an_upstream_that_is_not_a_unix_socket_is_refused_before_anything_starts(provider: HpcProvider):
    spec = _stack()
    spec.services[MAIN].sandbox.upstreams["gateway"] = "http://127.0.0.1:8000"

    with pytest.raises(ValueError, match="nothing in a pod reaches: give it as a unix:// socket"):
        await provider.start_stack(spec)
    assert provider.inner.started_pods == []


async def test_without_subordinate_ids_sidecars_run_as_your_own_uid(hpc_sandbox, provider: HpcProvider, caplog):
    hpc_sandbox.SUBORDINATE_IDS = False
    caplog.set_level(logging.WARNING)
    for _ in range(2):
        stack = await provider.start_stack(_stack(kv=_sidecar()))
        await stack.stop()

    first, second = provider.inner.started_pods
    assert first.spec.users == "single" and [service.run_as for service in first.started] == ["root", "caller"]
    assert caplog.text.count("runs the sidecars of stacks as your own uid") == 1  # said once


async def test_enroot_runs_sidecars_as_your_own_uid(hpc_sandbox, provider: HpcProvider, monkeypatch):
    mapped: list[bool] = []
    monkeypatch.setattr(hpc_sandbox.PodService, "run_modes", ("root", "caller"))  # enroot's: one id per container
    monkeypatch.setattr(provider.inner, "map_subordinate_ids", lambda: mapped.append(True))

    stack = await provider.start_stack(_stack(kv=_sidecar()))
    await stack.stop()
    assert stack.run_modes == {"main": "root", "kv": "caller"} and not mapped


async def test_volumes_are_made_and_filled_in_the_pod_s_directory_before_any_service_starts(provider: HpcProvider):
    filled: list[Path] = []

    def fill(directory: Path) -> None:
        assert provider.inner.started_pods[0].started == []  # before any service starts
        filled.append(directory)
        (directory / "tree").mkdir()
        (directory / "tree" / "seed").write_text("1")

    copy = Volume(target="/srv/copy", name="platoon:copy-0", subpath="tree", read_only=True, fill=fill)
    spec = _stack(
        kv=_sidecar(volumes=[Volume(target="/data", name="data"), Volume(target="/cache"), copy]),
        web=_sidecar(volumes=[Volume(target="/shared", name="data"), dataclasses.replace(copy, target="/copy")]),
    )
    stack = await provider.start_stack(spec)
    pod = _only_pod(provider)
    try:
        root = pod.path / hpc_stacks.VOLUMES_DIR
        assert filled == [root / "volumes" / "platoon%3Acopy-0"]  # once, for both; no colon, which binds refuse
        mounts = {service.name: service.sandbox.mounts for service in pod.started}
        kv = {mount.target: (mount.source, mount.read_only) for mount in mounts["kv"]}
        assert kv == {
            "/data": (str(root / "volumes" / "data"), False),
            "/cache": (str(root / "services" / "kv" / "volume-1"), False),
            "/srv/copy": (str(root / "volumes" / "platoon%3Acopy-0" / "tree"), True),
        }
        web = {mount.target: mount.source for mount in mounts["web"]}
        assert web["/shared"] == kv["/data"][0] and web["/copy"] == kv["/srv/copy"][0]
        assert stack.volume_path("data") == root / "volumes" / "data"
    finally:
        await stack.stop()
    assert not pod.path.exists()  # the pod's deletion took the volumes


async def test_states_logs_network_and_connections_go_through_the_pod(provider: HpcProvider):
    check = Healthcheck(command=("probe",), interval_s=0.01, retries=2)
    spec = _stack(kv=_sidecar(ports=(8000,), healthcheck=check))
    stack = await provider.start_stack(spec)
    pod = _only_pod(provider)
    try:
        states = await stack.health()
        assert {name: state.status for name, state in states.items()} == {"main": "running", "kv": "healthy"}
        assert await stack.logs("kv", max_bytes=3) == "kv\n"  # the tail
        await stack.set_network("none")
        assert pod.network == "none" and stack.network == "none"
        assert _socket(await stack.connect("kv", 8000)) == f"{pod.path}/exposed/8000.sock"
        pod.services["kv"].exit_status = 137
        [state] = (await stack.health(["kv"])).values()
        assert (state.status, state.exit_code) == ("exited", 137)
    finally:
        await stack.stop()


async def test_a_failed_start_stops_the_pod_and_leaves_nothing_registered(provider: HpcProvider, before):
    async def setup(sandbox) -> None:
        raise RuntimeError("main cannot be prepared")

    spec = _stack(kv=_sidecar())
    spec.services[MAIN].setup = setup
    spec.services[MAIN].depends_on = {"kv": Dependency("service_started")}

    with pytest.raises(RuntimeError, match="main cannot be prepared"):
        await provider.start_stack(spec)
    pod = _only_pod(provider)
    assert pod.stops == 1 and pod.services["main"].stopped and not pod.path.exists() and not _registered(before)


async def test_a_sidecar_that_never_turns_healthy_fails_the_start(hpc_sandbox, provider: HpcProvider, monkeypatch):
    async def failing(self, cmd, **kwargs):
        listing = cmd[-1:] == ["platoon-list"]
        return hpc_sandbox.ExecResult(0 if listing else 1, "1:10\n" if listing else "", "", False)

    monkeypatch.setattr(hpc_sandbox.PodService, "exec", failing)
    check = Healthcheck(command=("probe",), interval_s=0.01, retries=2)

    with pytest.raises(StackStartError, match="the service kv never turned healthy"):
        await provider.start_stack(_stack(kv=_sidecar(healthcheck=check)))
    assert _only_pod(provider).stops == 1


async def test_a_start_cancelled_while_the_volumes_are_made_stops_the_pod_once_they_are(provider: HpcProvider, before):
    filling, release = threading.Event(), threading.Event()

    def fill(directory: Path) -> None:
        filling.set()
        release.wait(10)

    starting = asyncio.ensure_future(provider.start_stack(_stack(kv=_sidecar(volumes=[Volume("/v", fill=fill)]))))
    await asyncio.to_thread(filling.wait, 10)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    pod = _only_pod(provider)
    assert pod.stops == 0 and _registered(before) == [None]  # its directory is still being filled: still reaped
    release.set()
    for _ in range(200):
        if pod.stops:
            break
        await asyncio.sleep(0.01)
    assert pod.stops == 1 and pod.started == [] and not pod.path.exists() and not _registered(before)


async def test_volumes_have_docker_s_modes_whatever_the_umask(provider: HpcProvider):
    """A sidecar's users other than root read its volumes and write its tmpfs, as in Docker (0755, and 1777), also
    where this process's umask keeps others out of what it makes."""

    volumes = [Volume("/data", name="data"), Volume("/cache"), Volume("/run", tmpfs=True)]
    previous = os.umask(0o077)
    try:
        stack = await provider.start_stack(_stack(kv=_sidecar(volumes=volumes)))
    finally:
        os.umask(previous)
    try:
        root = _only_pod(provider).path / hpc_stacks.VOLUMES_DIR
        made = ("volumes/data", "services/kv/volume-1", "services/kv/volume-2")
        modes = {path: stat.S_IMODE((root / path).stat().st_mode) for path in made}
        assert modes == {"volumes/data": 0o755, "services/kv/volume-1": 0o755, "services/kv/volume-2": 0o1777}
    finally:
        await stack.stop()


async def test_a_connection_s_socket_too_long_for_an_address_is_handed_out_as_a_short_link(
    hpc_sandbox, provider: HpcProvider, tmp_path: Path, monkeypatch
):
    """A Unix socket's address holds 107 bytes, which a pod's path in a deep scratch root exceeds: ``connect`` gives a
    link to it in a private directory of node scratch, which the stack's stop removes."""

    deep = tmp_path / ("d" * 100) / "exposed" / "8000.sock"

    async def expose(self, port: int) -> str:
        return str(deep)

    monkeypatch.setattr(hpc_sandbox.SessionPod, "expose", expose)
    node = Path(tempfile.mkdtemp(prefix="pt-", dir="/tmp"))  # node scratch with a short path, as on a node
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(node))
    try:
        stack = await provider.start_stack(_stack(kv=_sidecar(ports=(8000,))))
        try:
            url = await stack.connect("kv", 8000)
            link = Path(url.removeprefix("unix://"))
            assert len(os.fsencode(link)) < 108 and link.parent.parent == node
            assert os.readlink(link) == str(deep) and stat.S_IMODE(link.parent.stat().st_mode) == 0o700
            assert await stack.connect("kv", 8000) == url
        finally:
            await stack.stop()
        assert not link.parent.exists()
    finally:
        node.rmdir()


async def test_connections_made_at_once_share_one_directory_of_links(
    hpc_sandbox, provider: HpcProvider, tmp_path: Path, monkeypatch
):
    """Two first connections at once, each to a socket too long for an address, link from one private directory of
    node scratch, which the stack's stop removes: none is left behind."""

    async def expose(self, port: int) -> str:
        return str(tmp_path / ("d" * 100) / "exposed" / f"{port}.sock")

    monkeypatch.setattr(hpc_sandbox.SessionPod, "expose", expose)
    node = Path(tempfile.mkdtemp(prefix="pt-", dir="/tmp"))
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(node))
    try:
        stack = await provider.start_stack(_stack(kv=_sidecar(ports=(8000, 8001))))
        try:
            urls = await asyncio.gather(stack.connect("kv", 8000), stack.connect("kv", 8001))
            links = [Path(url.removeprefix("unix://")) for url in urls]
            assert links[0].parent == links[1].parent and list(node.iterdir()) == [links[0].parent]
        finally:
            await stack.stop()
        assert list(node.iterdir()) == []
    finally:
        node.rmdir()


async def test_a_service_s_log_tail_is_its_last_bytes(hpc_sandbox, provider: HpcProvider, monkeypatch):
    async def logs(self, name: str) -> str:
        return "\u00e9t\u00e9\n"  # six bytes

    monkeypatch.setattr(hpc_sandbox.SessionPod, "logs", logs)
    stack = await provider.start_stack(_stack(kv=_sidecar()))
    try:
        assert await stack.logs("kv", max_bytes=4) == "t\u00e9\n"
    finally:
        await stack.stop()


async def test_a_start_cancelled_again_while_it_rolls_back_stays_registered_until_the_pod_stopped(
    hpc_sandbox, provider: HpcProvider, before, monkeypatch
):
    """A start cancelled a second time stops waiting for its rollback, which runs on: the pod stays registered for
    cleanup (a stopping process stops it; a process killed meanwhile is reaped by its labels) until it has stopped."""

    preparing, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stop = hpc_sandbox.SessionPod.stop

    async def slow_stop(self) -> None:
        stopping.set()
        await release.wait()
        await stop(self)

    async def setup(sandbox) -> None:
        preparing.set()
        await asyncio.Event().wait()  # until cancelled

    monkeypatch.setattr(hpc_sandbox.SessionPod, "stop", slow_stop)
    spec = _stack(kv=_sidecar())
    spec.services[MAIN].setup = setup
    starting = asyncio.ensure_future(provider.start_stack(spec))
    await asyncio.wait_for(preparing.wait(), 5)
    starting.cancel()
    await asyncio.wait_for(stopping.wait(), 5)  # the rollback stops the pod
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert _registered(before) == [None]
    release.set()
    await _until(lambda: not _registered(before))
    assert _only_pod(provider).stops == 1


async def test_a_reap_stops_the_pods_of_this_process_and_sweeps_those_of_a_process_that_is_gone(
    provider: HpcProvider, tmp_path: Path
):
    stack = await provider.start_stack(_stack(kv=_sidecar()))
    labels = {STACK_LABEL: stack.id}
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    left = provider.inner.pod_root / "hpcsb-left.json"
    left.write_text(json.dumps({"owner": int(dead.stdout), "labels": {"platoon.run": "/srv/run"}}))
    (provider.inner.pod_root / "hpcsb-left").mkdir()

    assert await provider.reap(labels) == 1  # this process's own pod
    assert await provider.reap({"platoon.run": "/srv/run"}) == 1  # one a process that is gone left
    assert list(provider.inner.pod_root.iterdir()) == []
    await stack.stop()  # stopped already: nothing more happens


def _hpc_sandbox_0_4(fake: types.ModuleType) -> types.ModuleType:
    """The fake as hpc-sandbox 0.4 has it: pods whose services all run as root, without host aliases or exposed
    ports."""

    pod = [
        (name, object, dataclasses.field(default=None)) for name in ("network", "upstreams", "upstream_ports", "labels")
    ]
    module = types.ModuleType("hpc_sandbox")
    module.__dict__.update({name: value for name, value in vars(fake).items() if not name.startswith("__")})
    module.__dict__.update(
        __version__="0.4.0",
        PodSpec=dataclasses.make_dataclass("PodSpec", pod),
        ServiceSpec=dataclasses.make_dataclass(
            "ServiceSpec", ["name", "sandbox", ("command", object, None), ("shm_mb", object, None)]
        ),
    )
    return module


async def test_an_hpc_sandbox_without_0_5_s_pods_refuses_every_stack_naming_0_5(
    hpc_sandbox, monkeypatch, tmp_path: Path
):
    monkeypatch.setitem(sys.modules, "hpc_sandbox", _hpc_sandbox_0_4(hpc_sandbox))
    provider = cast(HpcProvider, get_provider("singularity", scratch_root=str(tmp_path)))

    refusal = stack_refusal("singularity", {})
    assert refusal == (
        "the singularity provider runs a multi-container stack as an hpc-sandbox pod, and the installed hpc-sandbox "
        "(0.4.0) has no pod services that run as your uid or with your subordinate ids, host aliases or ports exposed "
        "to the host: install hpc-sandbox 0.5"
    )
    assert STACKS not in provider.capabilities and stack_refusal(cast(Any, provider)) == refusal
    with pytest.raises(ValueError, match="install hpc-sandbox 0.5"):
        await provider.start_stack(_stack(kv=_sidecar()))
    assert provider.inner.started_pods == []


async def test_an_hpc_sandbox_without_pods_or_a_node_that_does_not_nest_refuses_stacks(hpc_sandbox, monkeypatch):
    monkeypatch.setattr(hpc_stacks, "nesting_refusal", lambda: "this node does not nest user namespaces")
    assert stack_refusal("singularity", {}) == "this node does not nest user namespaces"
    monkeypatch.delattr(hpc_sandbox.SingularityProvider, "start_pod")
    assert "has no pods: install hpc-sandbox 0.5" in (stack_refusal("singularity", {}) or "")


async def test_a_build_of_hpc_sandbox_0_5_without_what_pods_need_names_it(hpc_sandbox, monkeypatch):
    monkeypatch.setattr(hpc_sandbox, "__version__", "0.5.0.dev0", raising=False)
    monkeypatch.delattr(hpc_sandbox.SingularityProvider, "map_subordinate_ids")
    monkeypatch.delattr(hpc_sandbox.SessionPod, "expose")

    assert stack_refusal("singularity", {}) == (
        "the singularity provider runs a multi-container stack as an hpc-sandbox pod, and the installed hpc-sandbox "
        "(0.5.0.dev0) has no pod services that run as your uid or with your subordinate ids or ports exposed to the "
        "host: install a release of hpc-sandbox 0.5"
    )


async def test_the_nesting_probe_runs_once_per_process(monkeypatch):
    runs: list[list[str]] = []

    def run(argv, **kwargs):
        runs.append(argv)
        stderr = "Traceback ...\nPermissionError: [Errno 1] Operation not permitted\n"
        return subprocess.CompletedProcess(argv, 1, "", stderr)

    hpc_stacks.nesting_refusal.cache_clear()
    monkeypatch.setattr(hpc_stacks.subprocess, "run", run)
    try:
        for _ in range(2):
            assert hpc_stacks.nesting_refusal() == (
                "this node does not let an unprivileged user nest user namespaces, which pods need (PermissionError: "
                "[Errno 1] Operation not permitted)"
            )
    finally:
        hpc_stacks.nesting_refusal.cache_clear()
    assert len(runs) == 1 and runs[0][:2] == [sys.executable, "-I"]


@pytest.mark.parametrize(
    ("user", "root"),
    [
        (None, True),
        ("", True),
        ("0", True),
        ("root:root", True),
        ("0:1000", True),
        ("00", True),  # uid 0, as Docker reads it
        ("01", False),
        ("redis", False),
        ("999:999", False),
    ],
)
async def test_an_image_s_user_runs_as_root_only_when_it_is_root(user: str | None, root: bool):
    assert hpc_stacks.runs_as_root(user) is root
