"""Stacks on the process provider (platoon.sandbox.process_stacks): ports each stack holds, the listen variable,
references rewritten to those ports, volumes shared by name and filled before the services start, a root of this user's
alone, a service whose sandbox does not answer never called exited, and nothing left after a stop, a cancelled start or
a reap."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from platoon.sandbox import cleanup, process_stacks
from platoon.sandbox.base import ExecResult, SandboxSpec
from platoon.sandbox.process import ProcessProvider, ProcessSandbox
from platoon.sandbox.process_stacks import PORT_ENV, STACK_LABEL, STACK_RECORD, HostService
from platoon.sandbox.stacks import (
    Dependency,
    Healthcheck,
    ServiceSpec,
    Stack,
    StackSpec,
    StackStartError,
    Volume,
    service_references,
)
from platoon.utils.stopping import CleanupScope

pytestmark = pytest.mark.asyncio

# Serves "<name>:<the port it listens on>" on its listen variable, PLATOON_PORT_8000, else 8000.
SERVER = (
    "import http.server, os\n"
    "port = int(os.environ.get('PLATOON_PORT_8000', '8000'))\n"
    "class Answer(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        body = f\"{os.environ['NAME']}:{port}\".encode()\n"
    "        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()\n"
    "        self.wfile.write(body)\n"
    "    def log_message(self, *args): pass\n"
    "http.server.ThreadingHTTPServer(('127.0.0.1', port), Answer).serve_forever()\n"
)
FETCH = "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1], timeout=5).read().decode())"


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


@pytest.fixture
def before() -> set[int]:
    """What is registered for cleanup before a test (other tests of the process may leave registrations)."""

    return {id(sandbox) for sandbox in cleanup.live_sandboxes()}


def _registered(before: set[int]) -> list[object]:
    """What the test registered for cleanup and left registered."""

    return [sandbox for sandbox in cleanup.live_sandboxes() if id(sandbox) not in before]


def _web_stack(name: str, **main_env: str) -> StackSpec:
    check = Healthcheck(command=(sys.executable, "-c", FETCH, "http://web:8000/"), interval_s=0.05, retries=100)
    return StackSpec(
        services={
            "main": ServiceSpec(
                sandbox=SandboxSpec(image="local", env={"WEB": "http://web:8000/", **main_env}),
                depends_on={"web": Dependency("service_healthy")},
            ),
            "web": ServiceSpec(
                sandbox=SandboxSpec(image="local", workdir="/", env={"NAME": name}),
                command=[sys.executable, "-c", SERVER],
                healthcheck=check,
                ports=(8000,),
                aliases=("www",),
            ),
        },
        labels={"platoon.task": name},
    )


def _claimed(port: int) -> bool:
    """Whether a stack holds ``port``: its claim cannot be bound by anyone else."""

    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(f"\0platoon-stack-port-{port}")
    except OSError:
        return True
    finally:
        probe.close()
    return False


def _spec(stack: Stack, service: str) -> SandboxSpec:
    sandbox = stack.services[service]
    assert isinstance(sandbox, ProcessSandbox)
    return sandbox.spec


async def _fetch(stack, url_variable: str = "WEB") -> str:
    result = await stack.main.exec(f'{sys.executable} -c "{FETCH}" "${url_variable}"')
    assert result.exit_code == 0, result.stderr
    return result.stdout.strip()


async def test_each_stack_gets_its_own_ports_and_its_services_reach_one_another_by_name(
    provider: ProcessProvider, before: set[int]
):
    first, second = await asyncio.gather(
        provider.start_stack(_web_stack("one")), provider.start_stack(_web_stack("two"))
    )
    try:
        one, two = await _fetch(first), await _fetch(second)
        assert one.startswith("one:") and two.startswith("two:") and one.split(":")[1] != two.split(":")[1]
        port = int(one.split(":")[1])
        low, high = map(int, Path("/proc/sys/net/ipv4/ip_local_port_range").read_text().split())
        assert not low <= port <= high  # neither a bind to port 0 nor an outgoing connection takes it
        if sys.platform.startswith("linux"):
            assert _claimed(port)  # held for the stack's life, however its service binds it
        web, main = _spec(first, "web"), _spec(first, "main")
        assert web.env[PORT_ENV.format(port=8000)] == str(port)  # its listen variable
        assert main.env["WEB"] == f"http://127.0.0.1:{port}/"  # main's reference, rewritten
        assert first.rewrite("redis://user@web:8000/0 and http://www:8000 and http://web:9/") == (
            f"redis://user@127.0.0.1:{port}/0 and http://127.0.0.1:{port} and http://web:9/"  # an undeclared port stays
        )
        assert await first.connect("web", 8000) == f"tcp://127.0.0.1:{port}"
        with pytest.raises(ValueError, match="declares no port 9"):
            await first.connect("web", 9)
        assert first.run_modes == {"main": "caller", "web": "caller"}
        assert web.workdir == "/tmp"  # its working directory was /
        assert main.labels[STACK_LABEL] == first.id and main.labels["platoon.service"] == "main"
        states = await first.health()
        assert {name: state.status for name, state in states.items()} == {"main": "running", "web": "healthy"}
    finally:
        await asyncio.gather(first.stop(), second.stop())
    assert list(provider.root.iterdir()) == [] and not _registered(before)
    assert not _claimed(port)  # released with the stack


async def test_a_named_volume_is_one_directory_for_every_service_that_names_it(provider: ProcessProvider):
    writer = [sys.executable, "-c", "open('/shared/note', 'w').write('hi')"]  # the provider maps the path
    filled: list[Path] = []

    def fill(directory: Path) -> None:  # once, before any service that mounts it starts
        filled.append(directory)
        (directory / "tree").mkdir()
        (directory / "tree" / "seed").write_text("seeded")

    copy = Volume(target="/opt/copy", name="platoon:copy-0", subpath="tree", fill=fill)
    spec = StackSpec(
        services={
            "main": ServiceSpec(
                sandbox=SandboxSpec(image="local"),
                volumes=[Volume(target="/shared", name="data"), Volume(target="/own"), copy],
                depends_on={"writer": Dependency("service_completed_successfully")},
            ),
            "writer": ServiceSpec(
                sandbox=SandboxSpec(image="local"),
                command=writer,
                volumes=[
                    Volume(target="/shared", name="data"),
                    Volume(target="/seen", name=copy.name, subpath="tree", read_only=True),
                ],
            ),
        }
    )
    stack = await provider.start_stack(spec)
    try:
        assert (await stack.main.exec("cat /shared/note")).stdout == "hi"
        assert (await stack.main.exec("ls -A /own")).stdout == ""  # an anonymous volume starts empty
        assert (await stack.health(["writer"]))["writer"].status == "completed"
        assert (await stack.main.exec("cat /opt/copy/seed")).stdout == "seeded"  # its fill's subpath, mounted
        await stack.main.exec("echo from-main > /opt/copy/note")
        assert (await stack.services["writer"].exec("cat /seen/note")).stdout == "from-main\n"  # one directory
        assert filled == [stack.volume_path("platoon:copy-0")] and filled[0].is_relative_to(provider.root / stack.id)
        with pytest.raises(KeyError):
            stack.volume_path("nope")
    finally:
        await stack.stop()
    assert list(provider.root.iterdir()) == []


async def test_a_service_whose_command_fails_fails_the_start_with_its_log_and_leaves_nothing(
    provider: ProcessProvider, before: set[int]
):
    spec = StackSpec(
        services={
            "main": ServiceSpec(sandbox=SandboxSpec(image="local"), depends_on={"db": Dependency("service_healthy")}),
            "db": ServiceSpec(
                sandbox=SandboxSpec(image="local"),
                command=["sh", "-c", "echo cannot open the data directory >&2; exit 3"],
                healthcheck=Healthcheck(command=("test", "-f", "/tmp/ready"), interval_s=0.05, retries=100),
                ports=(5432,),
            ),
        }
    )
    with pytest.raises(StackStartError, match="the service db exited with status 3 before its healthcheck") as error:
        await provider.start_stack(spec)
    assert "cannot open the data directory" in error.value.log
    assert list(provider.root.iterdir()) == [] and not _registered(before)


async def test_reaping_by_label_deletes_a_stack_s_sandboxes_and_its_directory(provider: ProcessProvider):
    """Reaping takes the stacks of this process too, as those of a process that is gone (a live other's it leaves:
    below)."""

    stack = await provider.start_stack(_web_stack("orphan"))
    record = provider.root / stack.id / STACK_RECORD
    data = json.loads(record.read_text())
    assert data["labels"]["platoon.task"] == "orphan" and data["owner_pid"] == os.getpid()
    try:
        assert await provider.reap({"platoon.task": "orphan"}) == 3  # its two sandboxes and its directory
        assert list(provider.root.iterdir()) == []
    finally:
        await stack.stop()  # what reaping took is gone already: stopping is quiet


async def test_a_stack_of_a_live_owner_is_left_alone(provider: ProcessProvider):
    directory = provider.root / "stack-other"
    directory.mkdir(parents=True)
    owner = await asyncio.create_subprocess_exec("sleep", "30")
    try:
        from platoon.utils.stopping import process_start_time

        record = {"labels": {"x": "1"}, "owner_pid": owner.pid, "owner_start_time": process_start_time(owner.pid)}
        (directory / STACK_RECORD).write_text(json.dumps(record))
        assert await provider.reap({"x": "1"}) == 0 and directory.is_dir()
    finally:
        owner.kill()
        await owner.wait()
    assert await provider.reap({"x": "1"}) == 1 and not directory.exists()


async def test_a_start_cancelled_while_main_is_prepared_leaves_nothing_once_it_raises(
    provider: ProcessProvider, before: set[int], monkeypatch: pytest.MonkeyPatch
):
    """main's sandbox runs and its setup is under way when the start is cancelled: the start raises only once main's
    sandbox is gone (its removal is slow here), with what the stack holds, so a rollout's permit that comes back once
    its cleanup scope is done finds nothing of it."""

    remove = ProcessSandbox._remove

    def slow_remove(self: ProcessSandbox) -> None:
        if self.spec.labels.get("platoon.service") == "main":
            time.sleep(0.5)
        remove(self)

    monkeypatch.setattr(ProcessSandbox, "_remove", slow_remove)
    prepared = asyncio.Event()

    async def setup(sandbox: object) -> None:
        prepared.set()
        await asyncio.sleep(30)

    spec = StackSpec(
        services={
            "main": ServiceSpec(sandbox=SandboxSpec(image="local"), depends_on={"kv": Dependency()}, setup=setup),
            "kv": ServiceSpec(sandbox=SandboxSpec(image="local"), command=["sleep", "300"], ports=(6379,)),
        }
    )
    scope = CleanupScope()

    async def rollout() -> None:
        with scope:
            await provider.start_stack(spec)

    task = asyncio.ensure_future(rollout())
    await asyncio.wait_for(prepared.wait(), timeout=30)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert list(provider.root.iterdir()) == [] and not _registered(before)  # main's sandbox included
    assert await scope.finish(10.0) == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ports are claimed through Linux's abstract sockets")
async def test_a_port_another_stack_holds_is_never_given_out(provider: ProcessProvider, tmp_path: Path, monkeypatch):
    """With room for two ports only, a stack gets the one that no claim holds, and a second stack none."""

    lowest = 23456
    ephemeral = tmp_path / "ip_local_port_range"
    ephemeral.write_text(f"{lowest + 2}\t65535\n")  # leaves lowest and lowest + 1 outside the ephemeral range
    monkeypatch.setattr(process_stacks, "_LOWEST_PORT", lowest)
    monkeypatch.setattr(process_stacks, "_EPHEMERAL_PORTS", str(ephemeral))
    if not all(process_stacks._bindable(port) for port in (lowest, lowest + 1)):
        pytest.skip("this host uses the test's ports")
    other = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    other.bind(f"\0platoon-stack-port-{lowest}")  # another stack's claim, in another process
    spec = StackSpec(services={"main": ServiceSpec(sandbox=SandboxSpec(image="local"), ports=(8000,))})
    try:
        stack = await provider.start_stack(spec)
        try:
            assert await stack.connect("main", 8000) == f"tcp://127.0.0.1:{lowest + 1}"
            with pytest.raises(OSError, match="No port is free for a stack"):
                await provider.start_stack(spec)
        finally:
            await stack.stop()
        again = await provider.start_stack(spec)  # released with the stack that held it
        assert await again.connect("main", 8000) == f"tcp://127.0.0.1:{lowest + 1}"
        await again.stop()
    finally:
        other.close()
    assert list(provider.root.iterdir()) == []


async def test_references_to_a_service_and_to_a_service_s_own_loopback_reach_the_ports_the_stack_holds(
    provider: ProcessProvider,
):
    """Every service's environment values, command and healthcheck are rewritten: a reference to a service at a port
    it declares, with or without a scheme and in any case, and a service's own ``localhost:<port>``; a reference to an
    undeclared port, to another service's port on a service's own loopback, and to a longer name stay as written."""

    spec = _web_stack("one")
    web = spec.services["web"]
    web.healthcheck = Healthcheck(command=(sys.executable, "-c", FETCH, "http://localhost:8000/"), interval_s=0.05)
    web.listen = {8000: "WEB_PORT"}
    spec.services["main"].sandbox.env.update(
        {"PLAIN": "WEB:8000", "LOOPBACK": "http://localhost:8000/", "OTHER": "myweb:8000 web:9 web.example:8000"}
    )
    spec.services["main"].command = [sys.executable, "-c", "import time; time.sleep(300)", "--peer=www:8000"]
    stack = await provider.start_stack(spec)
    try:
        port = int((await stack.connect("web", 8000)).rsplit(":", 1)[1])
        main = _spec(stack, "main")
        assert main.env["PLAIN"] == f"127.0.0.1:{port}"
        assert main.env["LOOPBACK"] == "http://localhost:8000/"  # main declares no port 8000: not its own
        assert main.env["OTHER"] == "myweb:8000 web:9 web.example:8000"
        assert _spec(stack, "web").env["WEB_PORT"] == str(port)  # the variable its listen names
        assert stack.rewrite("http://localhost:8000/", service="web") == f"http://127.0.0.1:{port}/"
        assert stack.rewrite("redis://u@0.0.0.0:8000", service="web") == f"redis://u@127.0.0.1:{port}"
        assert stack.rewrite("localhost:8000") == "localhost:8000"  # no service: no loopback of its own
        assert (await stack.health(["web"]))["web"].status == "healthy"  # its healthcheck reached its own port
        shell = int(cast(Any, stack)._handles["main"]._entry.split(":")[0])  # the shell that runs main's command
        [child] = Path(f"/proc/{shell}/task/{shell}/children").read_text().split()
        assert Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")[3] == f"--peer=127.0.0.1:{port}".encode()
    finally:
        await stack.stop()


async def test_service_references_have_a_scheme_or_a_port_and_a_name_of_their_own():
    found = [
        (reference.text, reference.host, reference.port)
        for reference in service_references(
            "http://kv/a kv:6379 KV:1 -h kv mykv:6379 kv.example.com:1 at kv. postgres://u:p@kv:5432/db kv:99999",
            {"kv"},
        )
    ]
    assert found == [
        ("http://kv", "kv", None),
        ("kv:6379", "kv", 6379),
        ("KV:1", "kv", 1),
        ("postgres://u:p@kv:5432", "kv", 5432),
        ("kv:99999", "kv", None),  # not a port
    ]


async def test_the_provider_s_root_must_be_this_user_s_alone(tmp_path: Path):
    """A root that another user of a node's shared scratch made first, or a symlink planted where it goes, is refused:
    no sandbox is made there, nor anything reaped from it."""

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "planted").symlink_to(elsewhere)
    provider = ProcessProvider(root=tmp_path / "planted")
    with pytest.raises(RuntimeError, match="planted is not a directory of this user's"):
        await provider.start(SandboxSpec(image="local"))
    assert await provider.reap({"platoon.task": "web"}) == 0
    assert list(elsewhere.iterdir()) == []


async def test_a_service_whose_sandbox_does_not_answer_in_time_is_not_called_exited():
    """Its sandbox too slow to tell whether its command runs says nothing of the service: a probe fails as
    infrastructure (the exception), never reads a sidecar that may run as one that exited."""

    class Slow:
        id = "slow"

        async def exec(self, *args: Any, **kwargs: Any) -> ExecResult:
            return ExecResult(exit_code=-1, stdout="", stderr="", timed_out=True)

    with pytest.raises(TimeoutError, match="did not answer"):
        await HostService(cast(Any, Slow()), "4242:1", {}).status()
