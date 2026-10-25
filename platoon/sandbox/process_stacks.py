"""Stacks on the ``process`` provider: each service a process sandbox on this host, for tests (RFC 0010, section 6).

:meth:`ProcessProvider.start_stack <platoon.sandbox.process.ProcessProvider.start_stack>` starts a stack with the
generic driver (:func:`platoon.sandbox.stacks.drive_stack`): each service is a process sandbox of the provider, with
the stack's labels and ``platoon.stack`` and ``platoon.service``, and its command runs in the background there,
its output in ``/tmp/platoon-service/service.log`` of the service's sandbox and its exit status written next to it
when it ends.  As everything the provider runs, a stack is not isolated: its services' images, networks, resources,
users and capabilities are not enforced, nor are read-only mounts, and the services share this host's network.

**Addressing.**  Each port a service declares is a port of this host that the stack holds for its whole life, so that
stacks running at once never take one another's: one outside the kernel's ephemeral range (``ip_local_port_range``),
which neither a bind to port 0 nor an outgoing connection takes, that nothing on the host has bound when the stack
starts, and that no other stack holds, in any process (a claim the kernel arbitrates: an abstract Unix socket named for
the port, which goes with the process that holds it).  A program outside platoon that binds that very port itself can
still take it.  The service learns its port from ``PLATOON_PORT_<declared port>`` in its environment
(``PLATOON_PORT_6379=21234``), its **listen variable**: a service that must also run here listens on
``${PLATOON_PORT_6379:-6379}``, which real stacks leave unset, as a script prefixes paths with
``$PLATOON_SANDBOX_ROOT``.  A ``scheme://<service>:<port>`` reference (the service's name, or one of its aliases, and a
port it declares, also with a ``user@`` before the host) in any service's environment values and healthchecks, and in
what a caller rewrites with :meth:`Stack.rewrite <platoon.sandbox.stacks.Stack.rewrite>`, becomes
``scheme://127.0.0.1:<its port>``.  A service that shares another's network (``network_of``) declares its ports there.
``connect(service, port)`` is ``tcp://127.0.0.1:<its port>``.

**Volumes** are directories below the stack's own, ``<root>/stack-<id>/``, made (and filled, for a volume with a
``fill``) before any service starts, and linked at their targets as the provider links mounts: a named volume one
directory for every service that names it, an anonymous one or a ``tmpfs`` a directory per service.  The stack's
directory records its labels and its owner (``stack.json``), so that :meth:`ProcessProvider.reap
<platoon.sandbox.process.ProcessProvider.reap>` deletes the directories of a process that is gone, as it does its
sandboxes', and what its volumes hold with them.  A service whose working directory is ``/`` runs in its own ``/tmp``,
since the provider maps no path to the host's root.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import posixpath
import random
import re
import shutil
import socket
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from platoon.sandbox import cleanup
from platoon.sandbox._processes import start_background
from platoon.sandbox.base import Mount
from platoon.sandbox.stacks import DrivenStack, ServiceSpec, StackNetwork, StackSpec, drive_stack
from platoon.utils.stopping import let_finish, process_start_time

if TYPE_CHECKING:
    from platoon.sandbox.process import ProcessProvider, ProcessSandbox

#: The label of every sandbox of a stack: the stack's id; ``platoon.service`` names the service.
STACK_LABEL = "platoon.stack"
SERVICE_LABEL = "platoon.service"
#: The record a stack's directory holds: its labels and its owner, which reaping reads.
STACK_RECORD = "stack.json"
STACK_PREFIX = "stack-"
#: A service's listen variable: ``PLATOON_PORT_<declared port>`` = the port it listens on in its stack.
PORT_ENV = "PLATOON_PORT_{port}"
SERVICE_LOG = "/tmp/platoon-service/service.log"
SERVICE_STATUS = "/tmp/platoon-service/status"
LOOPBACK = "127.0.0.1"
_ROOT_WORKDIR = "/tmp"  # where a service whose working directory is / runs
_LOWEST_PORT = 10000  # below it are the ports that software is set to listen on (5432, 6379, 8080, ...)
_EPHEMERAL_PORTS = "/proc/sys/net/ipv4/ip_local_port_range"
_DEFAULT_EPHEMERAL = (32768, 60999)  # Linux's own default
# A stack's claim of a port: an abstract Unix socket of this name (Linux; elsewhere ports are only checked as free).
_CLAIM = "\0platoon-stack-port-{port}"
_CLAIMS = sys.platform.startswith("linux")
# scheme://[user@]host:port, the port not followed by another digit.
_URL = re.compile(
    r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^@/\s]*@)?(?P<host>[A-Za-z0-9_][A-Za-z0-9_.-]*):"
    r"(?P<port>[0-9]{1,5})(?![0-9])"
)


async def start_process_stack(provider: ProcessProvider, spec: StackSpec) -> DrivenStack:
    """Start ``spec`` on ``provider`` (the module docstring says how): registered for cleanup like its services, so
    that a stopping process stops it and reaping by label deletes what it leaves."""

    spec.check()
    stack_id = f"{STACK_PREFIX}{uuid.uuid4().hex[:12]}"
    labels = {**spec.labels, **cleanup.labels(), STACK_LABEL: stack_id}
    token = cleanup.register(
        provider=cleanup.provider_ref(provider), options={"root": str(provider.root)}, labels=labels
    )
    # The thread runs to its end whatever its caller does: a stopping process spares it, and a start given up
    # meanwhile undoes what it made once it is done, a cleanup of the rollout's own.
    creating = let_finish(
        asyncio.ensure_future(asyncio.to_thread(_Backend.create, provider, spec, provider.root / stack_id, labels))
    )
    try:
        backend = await asyncio.shield(creating)
    except BaseException:
        cleanup.unregister(token)
        let_finish(asyncio.ensure_future(_remove_made(creating)))
        raise
    try:
        stack = await drive_stack(backend, spec, id=stack_id, on_stop=lambda: cleanup.unregister(token))
    except BaseException:
        cleanup.unregister(token)  # the driver stopped what it started
        raise
    cleanup.attach(token, stack)
    return stack


async def _remove_made(creating: asyncio.Future[_Backend]) -> None:
    """Release what a backend whose start was given up made, once its thread is done: its ports, and its directory
    (with the copies its volumes hold, which may be large), in a thread."""

    await asyncio.wait([creating])
    if creating.cancelled() or creating.exception() is not None:
        return  # it made nothing it did not undo itself
    backend = creating.result()
    backend.release_ports()
    await asyncio.to_thread(shutil.rmtree, backend.directory, True)


def reap_stack_directories(root: Path, labels: Mapping[str, str]) -> int:
    """Delete the stack directories below ``root`` whose labels include ``labels`` and whose owner is this process or
    gone, with what their volumes hold; returns how many.  Their services are process sandboxes, which reaping by the
    same labels ends."""

    reaped = 0
    for directory in sorted(root.glob(f"{STACK_PREFIX}*")):
        try:
            record = json.loads((directory / STACK_RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or not cleanup.labels_match(record.get("labels"), labels):
            continue
        owner = record.get("owner_pid")
        if isinstance(owner, int) and owner != os.getpid():
            started = process_start_time(owner)
            if started is not None and started == record.get("owner_start_time"):
                continue  # another live process holds it
        shutil.rmtree(directory, ignore_errors=True)
        reaped += 1
    return reaped


class _Service:
    """A started service: its process sandbox, and the background command it runs (None: it only idles)."""

    run_as = "caller"

    def __init__(self, sandbox: ProcessSandbox, entry: str | None) -> None:
        self.sandbox = sandbox
        self._entry = entry

    async def status(self) -> tuple[bool, int | None]:
        if self._entry is None:
            return True, None
        pid, _, start = self._entry.partition(":")
        if _running(int(pid), int(start)):
            return True, None
        try:
            text = self.sandbox.host_path(SERVICE_STATUS).read_text().strip()
        except (OSError, RuntimeError):
            return False, None
        return False, int(text) if text.lstrip("-").isdigit() else None

    async def logs(self, max_bytes: int) -> str:
        return await asyncio.to_thread(_tail, self.sandbox.host_path(SERVICE_LOG), max_bytes)

    async def stop(self) -> None:
        await self.sandbox.stop()


class _Backend:
    """The services of one stack on the process provider: their ports, names and volumes."""

    def __init__(self, provider: ProcessProvider, spec: StackSpec, directory: Path, labels: dict[str, str]) -> None:
        self.provider = provider
        self.directory = directory
        self.labels = labels
        # Every service's network: its own, or the one it shares (network_of, followed to its end).
        self.network_of: dict[str, str] = {}
        for name in spec.services:
            owner, seen = name, {name}
            while (shared := spec.services[owner].network_of) is not None and shared not in seen:
                owner = shared
                seen.add(owner)
            self.network_of[name] = owner
        # The names each network answers to: its services' names and aliases.
        self.names: dict[str, str] = {}
        for name, service in spec.services.items():
            for alias in (name, *service.aliases):
                self.names.setdefault(alias, self.network_of[name])
        self.ports: dict[tuple[str, int], int] = {}  # (network, declared port) -> the host's port
        self._claims: list[socket.socket] = []  # what holds those ports for the stack, until it is released
        # Each volume's directory: by name for a named one, by (service, its index) for a service's own.
        self.named: dict[str, Path] = {}
        self._own: dict[tuple[str, int], Path] = {}

    @classmethod
    def create(cls, provider: ProcessProvider, spec: StackSpec, directory: Path, labels: dict[str, str]) -> _Backend:
        backend = cls(provider, spec, directory, labels)
        directory.mkdir(parents=True)
        try:
            record = {"labels": labels, "owner_pid": os.getpid(), "owner_start_time": process_start_time(os.getpid())}
            (directory / STACK_RECORD).write_text(json.dumps(record), encoding="utf-8")
            declared = dict.fromkeys(
                (backend.network_of[name], port) for name, service in spec.services.items() for port in service.ports
            )
            ports, backend._claims = _claim_ports(len(declared))
            backend.ports = dict(zip(declared, ports, strict=True))
            backend._make_volumes(spec)
        except BaseException:
            backend.release_ports()
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return backend

    def _make_volumes(self, spec: StackSpec) -> None:
        """Make every volume's directory, and fill the ones with a ``fill``, once each, before any service starts."""

        for name, service in spec.services.items():
            for number, volume in enumerate(service.volumes):
                if volume.name is not None:
                    directory = self.named.get(volume.name)
                    if directory is not None:
                        continue  # made and filled for the first service that names it
                    directory = self.named[volume.name] = self.directory / "volumes" / _volume_dir(volume.name)
                else:
                    directory = self._own[(name, number)] = self.directory / "services" / name / f"volume-{number}"
                directory.mkdir(parents=True)
                if volume.fill is not None:
                    volume.fill(directory)

    async def start_service(self, name: str, spec: ServiceSpec) -> _Service:
        network = self.network_of[name]
        listen = {PORT_ENV.format(port=port): str(self.ports[(network, port)]) for port in spec.ports}
        mounts = list(spec.sandbox.mounts)
        for number, volume in enumerate(spec.volumes):
            directory = self.named[volume.name] if volume.name is not None else self._own[(name, number)]
            source = directory / volume.subpath if volume.subpath else directory
            mounts.append(Mount(source=str(source), target=volume.target, read_only=volume.read_only))
        workdir = spec.sandbox.workdir if posixpath.normpath(spec.sandbox.workdir) != "/" else _ROOT_WORKDIR
        sandbox_spec = dataclasses.replace(
            spec.sandbox,
            workdir=workdir,
            env={**spec.sandbox.env, **listen},
            mounts=mounts,
            labels={**self.labels, **spec.sandbox.labels, SERVICE_LABEL: name},
        )
        sandbox = await self.provider.start(sandbox_spec)
        try:
            if spec.setup is not None:
                await spec.setup(sandbox)
            entry = None
            if spec.command:
                entry = await start_background(
                    sandbox, list(spec.command), log=SERVICE_LOG, cwd=workdir, status=SERVICE_STATUS
                )
        except BaseException:
            # A cleanup of its own, which a stopping process spares: the stack's stop waits for this rollback.
            await asyncio.shield(let_finish(asyncio.ensure_future(sandbox.stop())))
            raise
        return _Service(sandbox, entry)

    async def set_network(self, network: StackNetwork) -> None:
        """The process provider enforces no network: nothing to change."""

    async def connect(self, service: str, port: int) -> str:
        allocated = self.ports.get((self.network_of[service], port))
        if allocated is None:
            raise ValueError(f"The service {service!r} declares no port {port}")
        return f"tcp://{LOOPBACK}:{allocated}"

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        def replace(match: re.Match[str]) -> str:
            network = self.names.get(match.group("host"))
            allocated = self.ports.get((network, int(match.group("port")))) if network is not None else None
            if allocated is None:
                return match.group(0)
            return f"{match.group('scheme')}{match.group('user') or ''}{LOOPBACK}:{allocated}"

        return _URL.sub(replace, value) if "://" in value else value

    def volume_path(self, name: str) -> Path:
        return self.named[name]

    async def close(self) -> None:
        self.release_ports()
        await asyncio.to_thread(shutil.rmtree, self.directory, True)

    def release_ports(self) -> None:
        while self._claims:
            self._claims.pop().close()


def _volume_dir(name: str) -> str:
    """A named volume's directory below ``volumes/``: its name, with what a path cannot hold replaced."""

    return name.replace("/", "%2F") if name not in (".", "..") else f"%{name}"


def _claim_ports(count: int) -> tuple[list[int], list[socket.socket]]:
    """``count`` ports for a stack (the module docstring says which), and the claims that hold them until closed."""

    if count == 0:
        return [], []
    try:
        low, high = (int(part) for part in Path(_EPHEMERAL_PORTS).read_text().split())
    except (OSError, ValueError):
        low, high = _DEFAULT_EPHEMERAL
    spans = [span for span in (range(_LOWEST_PORT, low), range(max(high + 1, _LOWEST_PORT), 65536)) if len(span)]
    spans = spans or [range(_LOWEST_PORT, 65536)]  # an ephemeral range that leaves nothing out: all of it
    total = sum(len(span) for span in spans)
    first = random.SystemRandom().randrange(total)  # not the module's generator, which a test may seed
    ports: list[int] = []
    claims: list[socket.socket] = []
    try:
        for step in range(total):
            index = (first + step) % total
            for span in spans:
                if index < len(span):
                    port = span[index]
                    break
                index -= len(span)
            claim = _claim(port) if _CLAIMS else None
            if _CLAIMS and claim is None:
                continue  # another stack holds it
            if not _bindable(port):
                if claim is not None:
                    claim.close()
                continue
            ports.append(port)
            if claim is not None:
                claims.append(claim)
            if len(ports) == count:
                return ports, claims
        raise OSError(f"No port is free for a stack: {count} wanted, {len(ports)} found")
    except BaseException:
        for claim in claims:
            claim.close()
        raise


def _claim(port: int) -> socket.socket | None:
    """Claim ``port`` for a stack (on Linux): the bound socket that holds the claim, None when another process holds
    it."""

    claim = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        claim.bind(_CLAIM.format(port=port))
    except OSError:
        claim.close()
        return None
    return claim


def _bindable(port: int) -> bool:
    """Whether nothing on this host holds ``port``: binding it on every address works (a listener, a connection that
    lingers, or another bind of it on any address makes it fail)."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("", port))
        except OSError:
            return False
    return True


def _running(pid: int, start: int) -> bool:
    """Whether the host process ``pid`` that started at ``start`` still runs (a zombie does not)."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b") ", 1)[1].split()
    except (OSError, IndexError):
        return False
    try:
        return fields[0] not in (b"Z", b"X", b"x") and int(fields[19]) == start
    except (IndexError, ValueError):
        return False


def _tail(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = [
    "PORT_ENV",
    "SERVICE_LABEL",
    "STACK_LABEL",
    "reap_stack_directories",
    "start_process_stack",
]
