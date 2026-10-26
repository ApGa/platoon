"""Stacks whose services are sandboxes on this host's network: the ``process`` provider's, and the host-network
fallback's (RFC 0010, section 6).

:func:`start_host_stack` starts a stack with the generic driver (:func:`platoon.sandbox.stacks.drive_stack`), each
service a sandbox of a provider whose sandboxes share this host's network: :meth:`ProcessProvider.start_stack
<platoon.sandbox.process.ProcessProvider.start_stack>` starts its stacks so, each service a process sandbox, and the
host-network fallback (:mod:`platoon.sandbox.host_network_stacks`) over the single sandboxes of a provider that
isolates them.  Each service's sandbox has the stack's labels with ``platoon.stack`` and ``platoon.service``, and its
command runs in the background there, its output in ``/tmp/platoon-service/service.log`` of the sandbox and its exit
status written next to it when it ends, which the stack reads through the sandbox (:class:`HostService`).  On the
``process`` provider a stack is not isolated: its services' images, networks, resources, users and capabilities are not
enforced, nor are read-only mounts.

**Addressing** (:class:`HostAddresses`).  The services share this host's network, so each port a service declares is a
port of this host that the stack holds for its whole life, so that stacks running at once never take one another's: one
outside the kernel's ephemeral range (``ip_local_port_range``), which neither a bind to port 0 nor an outgoing
connection takes, that nothing on the host has bound when the stack starts, and that no other stack holds, in any
process (a claim the kernel arbitrates: an abstract Unix socket named for the port, which goes with the process that
holds it).  A program outside platoon that binds that very port itself can still take it.  A service learns its port
from its **listen variables**: ``PLATOON_PORT_<declared port>`` (``PLATOON_PORT_6379=21234``), and each variable its
spec's ``listen`` names for that port (``REDIS_PORT``), so a service written to run here too listens on
``${PLATOON_PORT_6379:-6379}``, which real stacks leave unset, as a script prefixes paths with
``$PLATOON_SANDBOX_ROOT``.  A **reference** to a service (:func:`~platoon.sandbox.stacks.service_references`),
``[scheme://][user@]<host>:<port>`` whose host is the service's name or one of its aliases (in any case, as host names
resolve) and whose port is one it declares, becomes ``[scheme://][user@]127.0.0.1:<its port>`` in every service's
environment values, command and healthcheck, and in what a caller rewrites with :meth:`Stack.rewrite
<platoon.sandbox.stacks.Stack.rewrite>`; so does a service's reference to a port it declares itself on its own loopback
(``localhost``, ``127.0.0.1``, ``0.0.0.0``), such as its healthcheck's.  A service that shares another's network
(``network_of``) declares its ports there.  ``connect(service, port)`` is ``tcp://127.0.0.1:<its port>``.

**Volumes** are directories below the stack's own, ``<root>/stack-<id>/``, made (and filled, for a volume with a
``fill``) before any service starts, and mounted at their targets (``root`` is a directory of your uid's alone,
:func:`private_root`, since what they hold is the stack's): a named volume one directory for every service that
names it, an anonymous one or a ``tmpfs`` a directory per service.  The stack's directory records its labels and its
owner (``stack.json``), so that reaping by label (:func:`reap_stack_directories`, which the provider whose ``root`` it
is runs) deletes the directories of a process that is gone, and what its volumes hold with them; the sandboxes of its
services are their provider's to reap.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import shutil
import socket
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from platoon.sandbox import cleanup
from platoon.sandbox._processes import alive_processes, start_background
from platoon.sandbox.base import Sandbox, SandboxProvider
from platoon.sandbox.stacks import (
    LOOPBACK,
    LOOPBACK_HOSTS,
    SERVICE_LABEL,
    STACK_LABEL,
    DrivenStack,
    ServiceSpec,
    StackNetwork,
    StackSpec,
    StackVolumes,
    drive_stack,
    make_volumes,
    service_references,
)
from platoon.utils.scratch import private_dir
from platoon.utils.stopping import let_finish, process_start_time

if TYPE_CHECKING:
    from platoon.sandbox.process import ProcessProvider

logger = logging.getLogger(__name__)

#: The record a stack's directory holds: its labels and its owner, which reaping reads.
STACK_RECORD = "stack.json"
STACK_PREFIX = "stack-"
#: A service's listen variable: ``PLATOON_PORT_<declared port>`` = the port it listens on in its stack.
PORT_ENV = "PLATOON_PORT_{port}"
SERVICE_LOG = "/tmp/platoon-service/service.log"
SERVICE_STATUS = "/tmp/platoon-service/status"
_LOWEST_PORT = 10000  # below it are the ports that software is set to listen on (5432, 6379, 8080, ...)
_EPHEMERAL_PORTS = "/proc/sys/net/ipv4/ip_local_port_range"
_DEFAULT_EPHEMERAL = (32768, 60999)  # Linux's own default
# A stack's claim of a port: an abstract Unix socket of this name (Linux; elsewhere ports are only checked as free).
_CLAIM = "\0platoon-stack-port-{port}"
_CLAIMS = sys.platform.startswith("linux")


class HostAddresses:
    """The ports and names of one stack whose services share this host's network (the module docstring says how):
    the ports are claimed by :meth:`claim`, and held until :meth:`release`."""

    def __init__(self, spec: StackSpec) -> None:
        self.services = spec.services
        # Every service's network: its own, or the one it shares (network_of, followed to its end).
        self.network_of: dict[str, str] = {}
        for name in spec.services:
            owner, seen = name, {name}
            while (shared := spec.services[owner].network_of) is not None and shared not in seen:
                owner = shared
                seen.add(owner)
            self.network_of[name] = owner
        # The names each network answers to, in lower case: its services' names and aliases.
        self.names: dict[str, str] = {}
        for name, service in spec.services.items():
            for alias in (name, *service.aliases):
                self.names.setdefault(alias.lower(), self.network_of[name])
        self.ports: dict[tuple[str, int], int] = {}  # (network, declared port) -> the host's port
        self._claims: list[socket.socket] = []  # what holds those ports for the stack, until they are released

    def claim(self) -> None:
        """Claim a port of this host for each port a network declares."""

        declared = dict.fromkeys(
            (self.network_of[name], port) for name, service in self.services.items() for port in service.ports
        )
        ports, self._claims = _claim_ports(len(declared))
        self.ports = dict(zip(declared, ports, strict=True))

    def own_ports(self, name: str) -> dict[int, int]:
        """The ports the service ``name`` declares, with the host's port of each."""

        network = self.network_of[name]
        return {port: self.ports[(network, port)] for port in self.services[name].ports}

    def listen_env(self, name: str, service: ServiceSpec) -> dict[str, str]:
        """The service's listen variables: ``PLATOON_PORT_<port>``, and those its ``listen`` names, for each port it
        declares."""

        env = {}
        for port, allocated in self.own_ports(name).items():
            env[PORT_ENV.format(port=port)] = str(allocated)
            if (variable := service.listen.get(port)) is not None:
                env[variable] = str(allocated)
        return env

    def connect(self, service: str, port: int) -> str:
        allocated = self.ports.get((self.network_of[service], port))
        if allocated is None:
            raise ValueError(f"The service {service!r} declares no port {port}")
        return f"tcp://{LOOPBACK}:{allocated}"

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        """``value`` with each reference to a service at a port it declares, and to a port ``service`` declares on its
        own loopback, at the host's port that the stack holds for it."""

        if ":" not in value:
            return value
        own = self.network_of.get(service) if service is not None else None
        hosts = {*self.names, *(LOOPBACK_HOSTS if own is not None else ())}
        pieces, last = [], 0
        for reference in service_references(value, hosts):
            network = own if reference.host in LOOPBACK_HOSTS else self.names[reference.host]
            allocated = self.ports.get((network, reference.port)) if reference.port is not None else None
            if allocated is None:
                continue
            pieces += [value[last : reference.start], f"{reference.scheme}{reference.user}{LOOPBACK}:{allocated}"]
            last = reference.end
        return "".join([*pieces, value[last:]]) if pieces else value

    def release(self) -> None:
        while self._claims:
            self._claims.pop().close()


class HostService:
    """A started service of a stack on this host's network: its sandbox, and the background command it runs (None: it
    only idles), watched through the sandbox (a :class:`~platoon.sandbox.stacks.ServiceHandle`)."""

    run_as = "caller"

    def __init__(self, sandbox: Sandbox, entry: str | None, ports: Mapping[int, int]) -> None:
        self.sandbox = sandbox
        self.ports = dict(ports)  # the ports it declares, with the host's port of each
        self._entry = entry  # "<pid>:<start time>" of the shell that runs its command and records its exit status

    async def status(self) -> tuple[bool, int | None]:
        """Whether its command runs, else its exit status (None: not known); TimeoutError when its sandbox does not
        answer in time, which says nothing of the service (a probe then fails as infrastructure)."""

        if self._entry is None:
            return True, None
        try:
            if await alive_processes(self.sandbox, {self._entry}):
                return True, None
            result = await self.sandbox.exec(["cat", SERVICE_STATUS], cwd="/", timeout_s=60.0)
        except TimeoutError:
            raise
        except Exception:  # a sandbox that is gone runs nothing
            return False, None
        text = result.stdout.strip()
        return False, int(text) if result.exit_code == 0 and text.lstrip("-").isdigit() else None

    async def logs(self, max_bytes: int) -> str:
        tail = ["/bin/sh", "-c", 'tail -c "$1" -- "$2" 2>/dev/null || cat -- "$2"', "platoon-log"]
        result = await self.sandbox.exec([*tail, str(max_bytes), SERVICE_LOG], cwd="/", timeout_s=60.0)
        return result.stdout[-max_bytes:] if result.exit_code == 0 else ""

    async def stop(self) -> None:
        await self.sandbox.stop()


class HostBackend:
    """The services of one stack on this host's network, each a sandbox of ``provider``: their ports and names
    (:class:`HostAddresses`), and their volumes in the stack's ``directory``.  :meth:`watch` says how a started service
    is watched; the ``process`` provider enforces no network, so :meth:`set_network` changes nothing."""

    def __init__(self, provider: SandboxProvider, spec: StackSpec, directory: Path, labels: dict[str, str]) -> None:
        self.provider = provider
        self.directory = directory
        self.labels = labels
        self.addresses = HostAddresses(spec)
        self.volumes = StackVolumes()

    @classmethod
    def create(cls, provider: SandboxProvider, spec: StackSpec, directory: Path, labels: dict[str, str]) -> Self:
        """The backend of ``spec``, with its directory, its record, its ports and its volumes made; in a thread."""

        backend = cls(provider, spec, directory, labels)
        private_root(directory.parent)
        directory.mkdir(mode=0o700)
        try:
            record = {"labels": labels, "owner_pid": os.getpid(), "owner_start_time": process_start_time(os.getpid())}
            (directory / STACK_RECORD).write_text(json.dumps(record), encoding="utf-8")
            backend.addresses.claim()
            backend.volumes = make_volumes(spec, directory)
        except BaseException:
            backend.addresses.release()
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return backend

    async def start_service(self, name: str, spec: ServiceSpec) -> HostService:
        sandbox_spec = dataclasses.replace(
            spec.sandbox,
            env={**spec.sandbox.env, **self.addresses.listen_env(name, spec)},
            mounts=[*spec.sandbox.mounts, *self.volumes.mounts(name, spec)],
            labels={**self.labels, **spec.sandbox.labels, SERVICE_LABEL: name},
        )
        sandbox = await self.provider.start(sandbox_spec)
        try:
            if spec.setup is not None:
                await spec.setup(sandbox)
            entry = None
            if spec.command:
                entry = await start_background(sandbox, list(spec.command), log=SERVICE_LOG, status=SERVICE_STATUS)
        except BaseException:
            # A cleanup of its own, which a stopping process spares: the stack's stop waits for this rollback.
            await asyncio.shield(let_finish(asyncio.ensure_future(sandbox.stop())))
            raise
        return self.watch(name, sandbox, entry)

    def watch(self, name: str, sandbox: Sandbox, entry: str | None) -> HostService:
        """The handle of the started service ``name``: its sandbox, and its command's shell (``entry``)."""

        return HostService(sandbox, entry, self.addresses.own_ports(name))

    async def set_network(self, network: StackNetwork) -> None:
        """The process provider enforces no network: nothing to change."""

    async def connect(self, service: str, port: int) -> str:
        return self.addresses.connect(service, port)

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        return self.addresses.rewrite(value, service=service)

    def volume_path(self, name: str) -> Path:
        return self.volumes.named[name]

    async def close(self) -> None:
        self.addresses.release()
        await asyncio.to_thread(shutil.rmtree, self.directory, True)


async def start_host_stack(
    provider: SandboxProvider,
    spec: StackSpec,
    *,
    root: Path,
    reaper: Any,
    backend_class: type[HostBackend] = HostBackend,
) -> DrivenStack:
    """Start ``spec``, each service a sandbox of ``provider`` (the module docstring says how), its directory below
    ``root``: registered for cleanup like its services, through ``reaper``, the provider that reaps the stack
    directories below ``root`` (built from ``{"root": root}``), so that a stopping process stops it and reaping by label
    deletes what it leaves."""

    spec.check()
    stack_id = f"{STACK_PREFIX}{uuid.uuid4().hex[:12]}"
    labels = {**spec.labels, **cleanup.labels(), STACK_LABEL: stack_id}
    token = cleanup.register(provider=cleanup.provider_ref(reaper), options={"root": str(root)}, labels=labels)
    # The thread runs to its end whatever its caller does: a stopping process spares it, and a start given up
    # meanwhile undoes what it made once it is done, a cleanup of the rollout's own.
    creating = let_finish(
        asyncio.ensure_future(asyncio.to_thread(backend_class.create, provider, spec, root / stack_id, labels))
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


async def start_process_stack(provider: ProcessProvider, spec: StackSpec) -> DrivenStack:
    """Start ``spec`` on the process provider, each service a process sandbox of it, the stack's directory below its
    ``root``, which its reaping covers."""

    return await start_host_stack(provider, spec, root=provider.root, reaper=provider)


async def _remove_made(creating: asyncio.Future[HostBackend]) -> None:
    """Release what a backend whose start was given up made, once its thread is done: its ports, and its directory
    (with the copies its volumes hold, which may be large), in a thread."""

    await asyncio.wait([creating])
    if creating.cancelled() or creating.exception() is not None:
        return  # it made nothing it did not undo itself
    backend = creating.result()
    backend.addresses.release()
    await asyncio.to_thread(shutil.rmtree, backend.directory, True)


def reap_stack_directories(root: Path, labels: Mapping[str, str]) -> int:
    """Delete the stack directories below ``root`` whose labels include ``labels`` and whose owner is this process or
    gone, with what their volumes hold; returns how many.  Their services' sandboxes are their provider's to reap, by
    the same labels."""

    if not os.path.lexists(root):
        return 0
    try:
        private_root(root)
    except RuntimeError as exc:
        logger.warning("Not reaping the stack directories there: %s", exc)
        return 0
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


def private_root(root: Path) -> Path:
    """``root``, made 0700 where it is missing: the directory of a provider's sandboxes or stacks, which must be a
    directory of this user's alone (``RuntimeError`` otherwise, :func:`platoon.utils.scratch.private_dir`): in a node's
    shared scratch another user who made it first could read what its stacks' volumes hold, or swap a volume's directory
    for a symlink before a runtime binds it, and a symlink in its place would put them, and a reap's deletions,
    elsewhere."""

    try:
        return private_dir(root)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}, so no sandbox or stack of this provider's may go there: remove it, or set PLATOON_NODE_SCRATCH "
            "(or the provider's root) to a directory of your own"
        ) from None


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


__all__ = [
    "PORT_ENV",
    "SERVICE_LABEL",
    "STACK_LABEL",
    "STACK_RECORD",
    "HostAddresses",
    "HostBackend",
    "HostService",
    "private_root",
    "reap_stack_directories",
    "start_host_stack",
    "start_process_stack",
]
