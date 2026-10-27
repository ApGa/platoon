"""Stacks: a task's services, each in a sandbox of its own, started and stopped together (RFC 0010, section 6).

A multi-container task, such as a Harbor task whose compose file names services besides ``main``, runs as a
**stack**: ``main``, the agents' sandbox, and its **sidecars** (a database, a judge, an MCP server), which reach one
another by name.  A provider that runs stacks declares the ``stacks`` capability (:data:`STACKS`) and has
``start_stack(spec)`` (:class:`StackProvider`): it starts every service of a :class:`StackSpec` and returns a running
:class:`Stack`, or raises, having stopped what it started.

**The spec.**  Each :class:`ServiceSpec` holds the sandbox the service runs in (a
:class:`~platoon.sandbox.base.SandboxSpec`: its image, working directory, environment, binds and resources; ``main``'s
carries the upstreams its harness needs), its command as the final argv (compose's ``entrypoint`` and ``command`` over
its image's, which the caller resolves; None for a service that only idles, as ``main`` does), its healthcheck, its
dependencies with their conditions, its volumes (directories the stack owns: a named one shared by every service that
names it, an anonymous one or a ``tmpfs`` the service's own; a caller may have one filled before any service mounts it,
such as with a copy of files it binds, which the stack then owns and deletes), the TCP ports it declares (with the
variables that tell it the port it listens on where its stack gives it another: ``listen``), the names it answers to
besides its own, and what only some runtimes apply (its run mode, the user its image runs its command as,
``shm_size``, ``ulimits``, added capabilities, ``privileged``, its stop signal: the driver passes them to the backend,
and the ``process`` provider's applies none; a provider with the ``privileged`` capability grants every privilege and
capability a spec asks for).  A ``setup`` coroutine, when given, runs once the service's sandbox runs and before its
command starts: the Harbor service prepares ``main`` that way (its directories, its image's deferred files, its init).
A backend makes its volumes' directories with :func:`make_volumes`.

**Starting.**  :func:`drive_stack` is the generic driver a provider may start its stacks with, over a
:class:`StackBackend` that starts one service at a time.  It starts each service once its dependencies have reached
their conditions, as compose does: ``service_started`` (its command started), ``service_healthy`` (its healthcheck
passed, by Docker's rules: failures in the start period do not count, and ``retries`` failures in a row make it
unhealthy) and ``service_completed_successfully`` (its command exited with status 0 within the service's
``timeout_s``: an init job, such as a seeder); services that do not wait for one another start at once.  It returns
once every service has started, each one with a healthcheck has passed it and each init job has completed, as ``docker
compose up --wait`` does, and a last probe of every service but the init jobs finds each one up: a service whose
command exits by then, or fails, or never turns healthy (or no longer is), fails the start (:class:`StackStartError`,
with the tail of its log).  On any failure, and when the start is cancelled, it stops every service it started, and
what the backend holds for the stack, before it raises: a service caught in the middle of its start stops what it
started itself, which the stop waits for; a ``setup`` coroutine's own exception passes through as it is.

**Probes.**  :meth:`Stack.health` tells each service's state (:class:`ServiceState`): an init job ``completed`` (or
``exited`` with another status), any other service ``exited``, ``unhealthy`` (its healthcheck failed ``retries`` times
in a row, or its backend found it serving where it must not: a handle's ``problem()``, which the host-network fallback
has), ``healthy``, or ``running`` when it has no healthcheck.  Callers probe around what depends on the services, such
as grading.

**Addressing.**  In a stack whose services share a network of their own (a pod, a compose project), each service
reaches the others by name at the ports they declare, and :meth:`Stack.rewrite` changes nothing; nor do they share
this host's loopback (a stack's ``host_loopback`` is false), so a harness in ``main`` reaches the model gateway only
through its route.  Where they share the host's network instead, as the ``process`` provider's and the host-network
fallback's do (:mod:`platoon.sandbox.process_stacks`, :mod:`platoon.sandbox.host_network_stacks`), each declared port
is a port of the host that the stack holds for its life, which the service learns from its listen variables, and
``rewrite`` turns a reference to a service at a port it declares (``scheme://<service>:<port>``, ``<service>:<port>``),
and a service's own ``localhost:<port>``, into the ``127.0.0.1:<allocated port>`` that reaches it.  The driver rewrites
every service's environment values, command and healthcheck so; a caller rewrites what else it runs in a service,
such as Harbor's collect hooks.  What such a rewrite acts on, a **reference** (:func:`service_references`), is
``[scheme://][user@]<host>:<port>`` or ``scheme://<host>``, the host's name in any case, as host names resolve; a
host's name alone (``-h postgres``) is none.

**Teardown.**  :meth:`Stack.stop` stops every service, then what the backend holds; its own task, and each service's
stop, are cleanups that a stopping process lets finish (:func:`platoon.utils.stopping.let_finish`), inside the cleanup
scope of the rollout that started them, so a rollout's permit comes back only once its stack is gone.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Collection, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from platoon.sandbox.base import ExecResult, Mount, Sandbox, SandboxSpec
from platoon.utils import scratch
from platoon.utils.stopping import let_finish

logger = logging.getLogger(__name__)

#: The capability of a provider that runs stacks (``start_stack``).
STACKS = "stacks"
#: The service that is the agents' sandbox; every stack has it.
MAIN = "main"
#: The label of every sandbox of a stack: the stack's id; ``platoon.service`` names the service.
STACK_LABEL = "platoon.stack"
SERVICE_LABEL = "platoon.service"
Condition = Literal["service_started", "service_healthy", "service_completed_successfully"]
CONDITIONS: tuple[Condition, ...] = ("service_started", "service_healthy", "service_completed_successfully")
#: A stack's network: ``egress`` reaches beyond the stack (the web), ``none`` only the stack's own services.
StackNetwork = Literal["egress", "none"]
#: How a service's processes run (RFC 0010, b.4): as root of a user namespace that maps one id, as the caller's own
#: uid, or as root with the caller's subordinate ids; a provider that runs everything as the caller says ``caller``.
RunMode = Literal["root", "caller", "subids"]
ServiceStatus = Literal["running", "healthy", "unhealthy", "exited", "completed"]
#: How much of a failed service's log a :class:`StackStartError` carries.
LOG_TAIL_BYTES = 2000
_HEALTH_DETAIL_CHARS = 500
_COMPLETION_POLL_S = (0.05, 1.0)  # how often an init job is looked at: first, and at most (doubling)
SOCKET_ADDRESS_BYTES = 108  # a Unix socket's path (sun_path), its terminating NUL included
#: This host's loopback address, where the services of a stack on this host's network reach one another.
LOOPBACK = "127.0.0.1"
#: The names of a service's own loopback, whose references to a port it declares are its own.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0"})
# [scheme://][user@]host[:port]: not inside a longer name, and the host ends where a name would (a dot ends it only
# before something that is no part of a name, as at the end of a sentence).
_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9_.+-])"
    r"(?:(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?P<user>[^@/\s]*@)?)?"
    r"(?P<host>[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?)"
    r"(?::(?P<port>[0-9]{1,5}))?"
    r"(?![A-Za-z0-9_]|[.-][A-Za-z0-9_]|:[0-9])"
)


@dataclass(frozen=True)
class Healthcheck:
    """A service's readiness probe, with Docker's defaults: ``command`` (an argv) runs in the service's sandbox, every
    ``interval_s`` (``start_interval_s`` during the start period), each run at most ``timeout_s``; failures in the
    first ``start_period_s`` do not count, and ``retries`` failures in a row make the service unhealthy."""

    command: tuple[str, ...]
    interval_s: float = 30.0
    timeout_s: float = 30.0
    start_period_s: float = 0.0
    start_interval_s: float = 5.0
    retries: int = 3

    @property
    def worst_case_s(self) -> float:
        """How long waiting for it to pass may take at most: its start period, then each retry's timeout and
        interval."""

        return self.start_period_s + (self.retries + 1) * (self.timeout_s + self.interval_s)

    @property
    def probe_worst_case_s(self) -> float:
        """How long a probe of a running service may take at most (:meth:`Stack.health`): ``retries`` runs, no start
        period."""

        return self.retries * (self.timeout_s + self.interval_s)


@dataclass(frozen=True)
class Dependency:
    """What a service waits for before it starts: the other service's ``condition`` (``required`` is compose's, which
    the driver keeps for the record: a service that is started and fails always fails the stack)."""

    condition: Condition = "service_started"
    required: bool = True


@dataclass(frozen=True)
class Volume:
    """A directory the stack owns, mounted at ``target`` in a service: a named volume (``name``), shared by every
    service of the stack that names it, or the service's own (``name`` None: an anonymous volume, or a ``tmpfs``),
    deleted with the stack.  It is empty when the stack starts unless ``fill`` fills it: called once with the volume's
    directory on this host, in a thread, before any service that mounts it starts (the first service's ``fill``, for a
    named volume several name), such as with a copy of files the caller binds, which the stack then owns.  ``subpath``
    mounts what ``fill`` made at that path of the volume (a file or a directory) rather than the whole of it, and
    ``read_only`` mounts it read-only where the provider enforces that."""

    target: str
    name: str | None = None
    tmpfs: bool = False
    read_only: bool = False
    subpath: str | None = None
    fill: Callable[[Path], None] | None = None


@dataclass
class ServiceSpec:
    """One service of a stack (the module docstring says what each field means)."""

    sandbox: SandboxSpec
    command: list[str] | None = None  # the final argv; None: the service only idles (main)
    healthcheck: Healthcheck | None = None
    depends_on: dict[str, Dependency] = field(default_factory=dict)
    volumes: list[Volume] = field(default_factory=list)
    ports: tuple[int, ...] = ()  # the TCP ports it declares (compose's expose and ports)
    # By declared port, the variables that tell it the port it listens on, where its stack gives it another (stacks on
    # this host's network set them, besides PLATOON_PORT_<port>; elsewhere it listens on the port it declares).
    listen: dict[int, str] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()  # names it answers to besides its own (compose's hostname, network aliases)
    network_of: str | None = None  # the service whose network it shares (compose's network_mode: service:<x>)
    # What only some runtimes apply (the driver passes them to the backend; the process provider applies none).
    run_as: RunMode | None = None  # None: the provider's choice
    user: str | None = None  # the user its image runs its command as (USER: a name, uid or uid:gid); None: root
    shm_mb: int | None = None
    ulimits: dict[str, tuple[int, int]] = field(default_factory=dict)
    cap_add: tuple[str, ...] = ()
    privileged: bool = False  # every privilege of the host: only a provider with the privileged capability grants it
    stop_signal: str | None = None
    # Awaited with the service's sandbox once it runs, before its command starts; its exception fails the start.
    setup: Callable[[Sandbox], Awaitable[None]] | None = None
    # An init job's longest run, from its start to its exit, before the stack's start fails; None: no bound.
    timeout_s: float | None = None


@dataclass
class StackSpec:
    """The services of a stack, ``main`` among them, its network, and the labels every service's sandbox carries."""

    services: dict[str, ServiceSpec]
    network: StackNetwork = "egress"
    labels: dict[str, str] = field(default_factory=dict)

    def order(self) -> list[str]:
        """The services in an order that starts each after its dependencies (the spec's order where that allows), once
        :meth:`check` has found nothing wrong; ``ValueError`` names what is."""

        self.check()
        placed: list[str] = []
        remaining = list(self.services)
        while remaining:
            ready = next(name for name in remaining if set(self.services[name].depends_on) <= set(placed))
            placed.append(ready)
            remaining.remove(ready)
        return placed

    def check(self) -> None:
        """Raise ``ValueError`` for a spec no stack can start: no ``main``, a dependency on a service it lacks, on
        itself or in a cycle, ``service_healthy`` on a service without a healthcheck, a network of a service it lacks,
        or a listen variable for a port the service does not declare."""

        if MAIN not in self.services:
            raise ValueError(f"A stack needs a {MAIN!r} service, the agents' sandbox; it has {sorted(self.services)}")
        if self.network not in ("egress", "none"):
            raise ValueError(f"A stack's network is egress or none, not {self.network!r}")
        for name, service in self.services.items():
            for other, dependency in service.depends_on.items():
                if other == name:
                    raise ValueError(f"The service {name!r} depends on itself")
                if other not in self.services:
                    raise ValueError(f"The service {name!r} depends on {other!r}, which the stack does not have")
                if dependency.condition not in CONDITIONS:
                    raise ValueError(f"The service {name!r} waits for {other!r} to be {dependency.condition!r}")
                if dependency.condition == "service_healthy" and self.services[other].healthcheck is None:
                    raise ValueError(
                        f"The service {name!r} waits for {other!r} to be healthy, and {other!r} has no healthcheck"
                    )
            if service.network_of is not None and (
                service.network_of == name or service.network_of not in self.services
            ):
                raise ValueError(
                    f"The service {name!r} shares the network of {service.network_of!r}, not another service"
                )
            if undeclared := sorted(set(service.listen) - set(service.ports)):
                raise ValueError(
                    f"The service {name!r} has listen variables for ports it does not declare: {undeclared}"
                )
        if cycle := dependency_cycle({name: set(service.depends_on) for name, service in self.services.items()}):
            raise ValueError(f"The services depend on one another in a cycle: {' -> '.join(cycle)}")

    def init_jobs(self) -> frozenset[str]:
        """The services some service waits for to complete successfully: init jobs, which run to their end."""

        return frozenset(
            other
            for service in self.services.values()
            for other, dependency in service.depends_on.items()
            if dependency.condition == "service_completed_successfully"
        )


def dependency_cycle(graph: Mapping[str, Iterable[str]]) -> list[str] | None:
    """A cycle of ``graph`` (each node's dependencies), as the nodes along it with the first again at its end; None
    when it has none.  Dependencies the graph does not list are leaves."""

    state: dict[str, int] = {}  # 1: on the current path, 2: done
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        state[node] = 1
        path.append(node)
        for other in sorted(graph.get(node, ())):
            if state.get(other) == 1:
                return [*path[path.index(other) :], other]
            if other not in state and (found := visit(other)) is not None:
                return found
        path.pop()
        state[node] = 2
        return None

    for node in graph:
        if node not in state and (found := visit(node)) is not None:
            return found
    return None


def longest_path(graph: Mapping[str, Iterable[str]], weights: Mapping[str, float]) -> float:
    """The heaviest path of an acyclic ``graph`` (each node's dependencies), each node counting its weight: how long
    starting every node takes when each starts once its dependencies have."""

    finish: dict[str, float] = {}

    def done(node: str) -> float:
        if node not in finish:
            finish[node] = weights.get(node, 0.0) + max((done(other) for other in graph.get(node, ())), default=0.0)
        return finish[node]

    return max((done(node) for node in graph), default=0.0)


# --- references to services ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """A reference to a host in a text: ``[scheme://][user@]host[:port]`` (:func:`service_references`)."""

    text: str
    start: int
    end: int
    scheme: str
    user: str
    host: str
    port: int | None


def service_references(text: str, hosts: Collection[str]) -> Iterator[Reference]:
    """The references in ``text`` to any of ``hosts`` (lower-case names: host names resolve in any case) that a
    stack's addressing could act on: with a scheme (``http://kv``, ``http://kv:6379``) or a port (``kv:6379``).  A
    host's name alone (``-h postgres``) is none."""

    if not hosts:
        return
    for match in _REFERENCE.finditer(text):
        port = match.group("port")
        if match.group("host").lower() not in hosts or (port is None and match.group("scheme") is None):
            continue
        yield Reference(
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            scheme=match.group("scheme") or "",
            user=match.group("user") or "",
            host=match.group("host").lower(),
            port=int(port) if port is not None and int(port) < 65536 else None,
        )


# --- volumes ---------------------------------------------------------------------------------------------------------


@dataclass
class StackVolumes:
    """Where a stack's volumes are on this host (:func:`make_volumes`): each named volume's directory, and each
    service's own (an anonymous volume or a ``tmpfs``), by the service and the volume's place in its list."""

    named: dict[str, Path] = field(default_factory=dict)
    own: dict[tuple[str, int], Path] = field(default_factory=dict)

    def mounts(self, name: str, service: ServiceSpec) -> list[Mount]:
        """The mounts of the service ``name``'s volumes: each one's directory, or what its ``fill`` made at its
        ``subpath``, at its target."""

        mounts = []
        for number, volume in enumerate(service.volumes):
            directory = self.named[volume.name] if volume.name is not None else self.own[(name, number)]
            source = directory / volume.subpath if volume.subpath else directory
            mounts.append(Mount(source=str(source), target=volume.target, read_only=volume.read_only))
        return mounts


def make_volumes(spec: StackSpec, root: Path) -> StackVolumes:
    """Make the directories of ``spec``'s volumes below ``root``, ``volumes/<name>`` for a named volume (once, for
    every service that names it) and ``services/<service>/volume-<n>`` for a service's own, with Docker's modes
    whatever the umask (0755, and 1777 for a ``tmpfs``, so that a service's users other than root use them as there),
    and fill the ones with a ``fill`` (a named volume with the first service's), before any service starts; in a
    thread, as it may copy files."""

    volumes = StackVolumes()
    for name, service in spec.services.items():
        for number, volume in enumerate(service.volumes):
            if volume.name is None:
                directory = volumes.own[(name, number)] = root / "services" / name / f"volume-{number}"
            elif volume.name in volumes.named:
                continue  # made and filled for the first service that names it
            else:
                directory = volumes.named[volume.name] = root / "volumes" / _volume_dir(volume.name)
            directory.mkdir(parents=True)
            directory.chmod(0o1777 if volume.tmpfs else 0o755)
            if volume.fill is not None:
                volume.fill(directory)
    return volumes


def _volume_dir(name: str) -> str:
    """A named volume's directory below ``volumes/``: its name, with each byte of a character other than a letter, a
    digit, ``_``, ``.`` or ``-`` written as ``%XX``, as a path cannot hold ``/`` and runtimes' binds take no ``:`` or
    ``,`` (platoon's own volumes are named ``platoon:...``); ``.`` and ``..`` get a ``%`` before them."""

    escaped = "".join(
        char
        if char.isascii() and (char.isalnum() or char in "_.-")
        else "".join(f"%{byte:02X}" for byte in char.encode())
        for char in name
    )
    return f"%{escaped}" if escaped in (".", "..") else escaped


class SocketLinks:
    """Paths short enough for a Unix socket's address (:data:`SOCKET_ADDRESS_BYTES`) to the sockets on this host
    through which a stack's :meth:`~Stack.connect` reaches its services: a socket's own path where it fits, else a link
    to it in a private directory (0700) of node scratch, or of ``/tmp`` where node scratch's path is too long too, made
    when a first link needs it and removed with the links by :meth:`close`."""

    def __init__(self) -> None:
        self._directory: Path | None = None

    async def short(self, socket: str | Path, name: str) -> str:
        """``socket``'s path, or a link to it named ``name`` (one name for one socket)."""

        if len(os.fsencode(socket)) < SOCKET_ADDRESS_BYTES:
            return str(socket)
        if self._directory is None:
            directory = await asyncio.to_thread(_links_directory, len(os.fsencode(name)))
            if self._directory is None:
                self._directory = directory
            else:  # another connection made one meanwhile, which close() removes
                await asyncio.to_thread(directory.rmdir)
        link = self._directory / name
        if not link.is_symlink():
            link.symlink_to(socket)
        return str(link)

    async def close(self) -> None:
        if self._directory is not None:
            await asyncio.to_thread(shutil.rmtree, self._directory, ignore_errors=True)


def _links_directory(name_bytes: int) -> Path:
    """A new private directory for links to a stack's sockets, whose path leaves room in a socket's address for a link
    named with ``name_bytes`` bytes: in node scratch, else in ``/tmp``."""

    for root in (scratch.scratch_dir(), Path("/tmp")):
        if len(os.fsencode(root)) + len("/platoon-sockets-XXXXXXXX/") + name_bytes < SOCKET_ADDRESS_BYTES:
            return Path(tempfile.mkdtemp(prefix="platoon-sockets-", dir=root))
    raise OSError("Neither node scratch nor /tmp has a path short enough for links to a stack's sockets")


@dataclass(frozen=True)
class ServiceState:
    """A service's state as a probe found it (:meth:`Stack.health`): ``status``, its command's exit status once it
    exited (when known), and what the healthcheck or the backend said."""

    status: ServiceStatus
    exit_code: int | None = None
    detail: str = ""

    @property
    def up(self) -> bool:
        """Whether the service serves: it runs (healthy, where it has a healthcheck), or an init job completed."""

        return self.status in ("running", "healthy", "completed")

    def describe(self) -> str:
        """``exited with status 1``, ``is unhealthy: <detail>``, ``is healthy``, ..."""

        if self.status == "exited":
            return "exited" + (f" with status {self.exit_code}" if self.exit_code is not None else "")
        if self.status == "unhealthy":
            return "is unhealthy" + (f": {self.detail}" if self.detail else "")
        return f"is {self.status}"


class StackStartError(RuntimeError):
    """A service of a stack failed to start: its sandbox did not start, its command exited or failed, or its
    healthcheck never passed.  ``service`` names it and ``log`` holds the tail of its log."""

    def __init__(self, service: str, message: str, *, log: str = "") -> None:
        super().__init__(
            f"the service {service} {message}" + (f"; its log ends: {log.strip()[-500:]}" if log.strip() else "")
        )
        self.service = service
        self.log = log


@runtime_checkable
class Stack(Protocol):
    """A running stack: its services' sandboxes, their states, and its teardown."""

    id: str

    @property
    def main(self) -> Sandbox:
        """The agents' sandbox, ``services["main"]``."""
        ...

    @property
    def services(self) -> Mapping[str, Sandbox]:
        """Every service's sandbox by name, ``main``'s too: where a caller runs a command in a service."""
        ...

    @property
    def network(self) -> StackNetwork:
        """The stack's network now (:meth:`set_network`)."""
        ...

    @property
    def run_modes(self) -> Mapping[str, str]:
        """How each service's processes run (:data:`RunMode`)."""
        ...

    async def health(self, services: Iterable[str] | None = None) -> dict[str, ServiceState]:
        """Probe ``services`` (default: every one) and say what each one's state is."""
        ...

    async def logs(self, service: str, *, max_bytes: int = LOG_TAIL_BYTES) -> str:
        """The last ``max_bytes`` of what the service's command wrote."""
        ...

    async def set_network(self, network: StackNetwork) -> None:
        """Give the whole stack ``network`` from now on (a provider that enforces no network records it)."""
        ...

    async def connect(self, service: str, port: int) -> str:
        """A host endpoint that reaches ``port`` of ``service``: ``unix:///<socket>`` or ``tcp://<host>:<port>``."""
        ...

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        """``value`` (a command, an environment value) as it must read in ``service`` to reach the stack's services by
        name (the module docstring's addressing)."""
        ...

    def volume_path(self, name: str) -> Path:
        """The directory on this host of the stack's named volume ``name``, which the caller may read (what a sidecar
        wrote there); ``KeyError`` for a volume the stack does not have."""
        ...

    async def stop(self) -> None:
        """Stop every service and release what the stack holds; idempotent."""
        ...


@runtime_checkable
class StackProvider(Protocol):
    """A provider that runs stacks (it declares :data:`STACKS`)."""

    async def start_stack(self, spec: StackSpec) -> Stack: ...


class ServiceHandle(Protocol):
    """One started service, as a :class:`StackBackend` runs it.  A handle may also have ``async def problem() -> str |
    None``, which says why the running service is unhealthy besides its healthcheck (the host-network fallback's: it
    listens beyond this host's loopback), and makes a probe find it so."""

    @property
    def sandbox(self) -> Sandbox: ...

    @property
    def run_as(self) -> str: ...

    async def status(self) -> tuple[bool, int | None]:
        """Whether its command still runs (always, for a service without one, while it runs), and its exit status once
        it exited, when known."""
        ...

    async def logs(self, max_bytes: int) -> str: ...

    async def stop(self) -> None: ...


class StackBackend(Protocol):
    """What a provider gives :func:`drive_stack`: one service at a time, and the stack's own resources."""

    async def start_service(self, name: str, spec: ServiceSpec) -> ServiceHandle:
        """Start the service's sandbox, await ``spec.setup`` with it when given, then start ``spec.command`` when
        given; on any failure, and when cancelled, stop what it started before it raises (a stop that a stopping
        process lets finish: :func:`platoon.utils.stopping.let_finish`)."""
        ...

    async def set_network(self, network: StackNetwork) -> None: ...

    async def connect(self, service: str, port: int) -> str: ...

    def rewrite(self, value: str, *, service: str | None = None) -> str: ...

    def volume_path(self, name: str) -> Path: ...

    async def close(self) -> None:
        """Release what the stack holds besides its services (its volumes, ...); idempotent."""
        ...


class SetupError(Exception):
    """Wraps what a service's ``setup`` raised while the driver unwinds; never seen outside :func:`drive_stack`."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


class DrivenStack:
    """A :class:`Stack` whose services :func:`drive_stack` started over a :class:`StackBackend`.  ``host_loopback``
    says whether its services share this host's network, loopback included (the module docstring's addressing)."""

    def __init__(
        self,
        backend: StackBackend,
        spec: StackSpec,
        *,
        id: str,
        on_stop: Callable[[], None] | None = None,
        host_loopback: bool = True,
    ) -> None:
        self.id = id
        self.spec = spec
        self.host_loopback = host_loopback
        self._backend = backend
        self._network: StackNetwork = spec.network
        # False after a switch that failed midway, which may have switched some services: the next one runs in full.
        self._network_settled = True
        self._jobs = spec.init_jobs()
        self._handles: dict[str, ServiceHandle] = {}
        self._checks: dict[str, Healthcheck] = {}  # each service's healthcheck as it runs there (rewritten)
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping: asyncio.Task[None] | None = None
        self._on_stop = on_stop

    @property
    def main(self) -> Sandbox:
        return self._handles[MAIN].sandbox

    @property
    def services(self) -> dict[str, Sandbox]:
        return {name: handle.sandbox for name, handle in self._handles.items()}

    @property
    def network(self) -> StackNetwork:
        return self._network

    @property
    def run_modes(self) -> dict[str, str]:
        return {name: handle.run_as for name, handle in self._handles.items()}

    async def health(self, services: Iterable[str] | None = None) -> dict[str, ServiceState]:
        names = list(self._handles if services is None else services)
        states = await asyncio.gather(*(self._state(name) for name in names))
        return dict(zip(names, states, strict=True))

    async def logs(self, service: str, *, max_bytes: int = LOG_TAIL_BYTES) -> str:
        handle = self._handles.get(service)
        if handle is None:
            return ""
        try:
            return await handle.logs(max_bytes)
        except Exception as exc:  # a log that cannot be read says nothing about the service
            return f"[cannot read the log: {type(exc).__name__}: {exc}]"

    async def set_network(self, network: StackNetwork) -> None:
        if network not in ("egress", "none"):
            raise ValueError(f"A stack's network is egress or none, not {network!r}")
        if network != self._network or not self._network_settled:
            self._network_settled = False
            await self._backend.set_network(network)
            self._network, self._network_settled = network, True

    async def connect(self, service: str, port: int) -> str:
        if service not in self._handles:
            raise KeyError(f"The stack has no service {service!r}")
        return await self._backend.connect(service, port)

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        return self._backend.rewrite(value, service=service)

    def volume_path(self, name: str) -> Path:
        return self._backend.volume_path(name)

    async def stop(self) -> None:
        """Stop every service, then what the backend holds; idempotent, and it finishes when its caller stops
        waiting (a stopping process lets it finish)."""

        if self._stopping is None:
            self._stopping = let_finish(asyncio.ensure_future(self._stop()))
        await asyncio.shield(self._stopping)

    async def _stop(self) -> None:
        # A start that is unwinding: no service may start once the stop has begun.  A service's start that was
        # cancelled already is stopping what it started (the backend's rollback): it is waited for, never cancelled
        # again, which would leave that rollback running on its own after the stack counts as stopped.
        starting = [task for task in self._tasks if not task.done()]
        for task in starting:
            if not task.cancelling():
                task.cancel()
        if starting:
            await asyncio.wait(starting)  # unlike gather, a wait that is cancelled cancels none of them
        for task in self._tasks:  # what they raised, the start reported already
            if task.done() and not task.cancelled():
                task.exception()
        handles = list(self._handles.items())
        # Each a cleanup of its own: a stopping process cancels every other task, again and again.
        stops = [let_finish(asyncio.ensure_future(handle.stop())) for _, handle in handles]
        if stops:
            await asyncio.wait(stops)
        for (name, _), stop in zip(handles, stops, strict=True):
            error = asyncio.CancelledError() if stop.cancelled() else stop.exception()
            if error is not None:
                logger.warning("Cannot stop the service %s of stack %s: %r", name, self.id, error)
        try:
            await self._backend.close()
        finally:
            if self._on_stop is not None:
                self._on_stop()

    # Starting ---------------------------------------------------------------------------------------------------

    async def _start(self) -> None:
        order = self.spec.order()
        loop = asyncio.get_running_loop()
        reached: dict[tuple[str, str], asyncio.Future[None]] = {
            (name, condition): loop.create_future() for name in order for condition in CONDITIONS
        }
        self._tasks = [asyncio.ensure_future(self._start_service(name, reached)) for name in order]
        try:
            pending: set[asyncio.Task[None]] = set(self._tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
                for task in done:
                    if not task.cancelled() and (error := task.exception()) is not None:
                        raise error
        finally:
            for task in self._tasks:
                task.cancel()
            for future in reached.values():
                future.cancel()
        self._tasks = []
        # As docker compose up --wait: every service that runs on is up, healthy where it has a healthcheck, also the
        # ones that passed theirs long before the last one started: the probe before the agents.
        running = [name for name in order if name not in self._jobs]
        states = await self.health(running)
        for name in running:
            if not (state := states[name]).up:
                raise StackStartError(name, f"{state.describe()} as the stack started", log=await self.logs(name))

    async def _start_service(self, name: str, reached: Mapping[tuple[str, str], asyncio.Future[None]]) -> None:
        service = self.spec.services[name]
        for other, dependency in service.depends_on.items():
            await asyncio.shield(reached[(other, dependency.condition)])
        prepared = self._prepared(name, service)
        try:
            handle = await self._backend.start_service(name, prepared)
        except SetupError as wrapped:
            raise wrapped.error from None
        except StackStartError:
            raise
        except Exception as exc:
            raise StackStartError(name, f"did not start: {type(exc).__name__}: {exc}") from exc
        self._handles[name] = handle
        reached[(name, "service_started")].set_result(None)
        if name in self._jobs:
            await self._wait_completed(name, handle, service.timeout_s)
            reached[(name, "service_completed_successfully")].set_result(None)
            reached[(name, "service_healthy")].set_result(None)  # what waits for it to be healthy waits for its end
            return
        if (check := self._checks.get(name)) is not None:
            await self._wait_healthy(name, handle, check)
        reached[(name, "service_healthy")].set_result(None)

    def _prepared(self, name: str, service: ServiceSpec) -> ServiceSpec:
        """``service`` as it starts: its environment values, command and healthcheck rewritten for the stack's
        addressing, and its setup's exception marked as its own."""

        rewrite = self._backend.rewrite
        sandbox = dataclasses.replace(
            service.sandbox, env={key: rewrite(value, service=name) for key, value in service.sandbox.env.items()}
        )
        command = [rewrite(arg, service=name) for arg in service.command] if service.command is not None else None
        if service.healthcheck is not None:
            self._checks[name] = dataclasses.replace(
                service.healthcheck, command=tuple(rewrite(arg, service=name) for arg in service.healthcheck.command)
            )
        setup = service.setup
        if setup is not None:
            original = setup

            async def setup(sandbox_: Sandbox) -> None:
                try:
                    await original(sandbox_)
                except Exception as exc:
                    raise SetupError(exc) from exc

        return dataclasses.replace(
            service, sandbox=sandbox, command=command, healthcheck=self._checks.get(name), setup=setup
        )

    async def _wait_healthy(self, name: str, handle: ServiceHandle, check: Healthcheck) -> None:
        """Wait for the service's healthcheck to pass, by Docker's rules; :class:`StackStartError` once its command
        exits or the check failed ``retries`` times in a row after the start period."""

        loop = asyncio.get_running_loop()
        started = loop.time()
        failures = 0
        while True:
            passed, detail = await _run_check(handle, check)
            if passed:
                return
            running, code = await handle.status()
            if not running:
                raise StackStartError(name, f"{_exited(code)} before its healthcheck passed", log=await self.logs(name))
            starting = loop.time() - started < check.start_period_s
            if not starting:
                failures += 1
                if failures >= check.retries:
                    raise StackStartError(
                        name,
                        f"never turned healthy: its healthcheck failed {failures} times: {detail}",
                        log=await self.logs(name),
                    )
            await asyncio.sleep(check.start_interval_s if starting else check.interval_s)

    async def _wait_completed(self, name: str, handle: ServiceHandle, timeout_s: float | None) -> None:
        """Wait for an init job's command to exit, at most ``timeout_s`` (None: no bound); :class:`StackStartError`
        unless it exits with status 0 by then."""

        loop = asyncio.get_running_loop()
        deadline = None if timeout_s is None else loop.time() + timeout_s
        delay, longest = _COMPLETION_POLL_S
        while True:
            running, code = await handle.status()
            if not running:
                if code == 0:
                    return
                raise StackStartError(
                    name, f"{_exited(code)}, and others wait for it to complete successfully", log=await self.logs(name)
                )
            if deadline is not None and loop.time() >= deadline:
                raise StackStartError(
                    name,
                    f"did not complete within {timeout_s:g}s, and others wait for it to complete successfully",
                    log=await self.logs(name),
                )
            await asyncio.sleep(delay if deadline is None else max(0.0, min(delay, deadline - loop.time())))
            delay = min(delay * 2, longest)

    # Probing ----------------------------------------------------------------------------------------------------

    async def _state(self, name: str) -> ServiceState:
        handle = self._handles.get(name)
        if handle is None:
            return ServiceState("exited", detail="it never started")
        running, code = await handle.status()
        if name in self._jobs:
            if running:
                return ServiceState("running")
            return ServiceState("completed" if code == 0 else "exited", code)
        if not running:
            return ServiceState("exited", code)
        if (problem := getattr(handle, "problem", None)) is not None and (detail := await problem()) is not None:
            return ServiceState("unhealthy", detail=detail)
        check = self._checks.get(name)
        if check is None:
            return ServiceState("running")
        failures = 0
        while True:
            passed, detail = await _run_check(handle, check)
            if passed:
                return ServiceState("healthy")
            failures += 1
            if failures >= check.retries:
                return ServiceState("unhealthy", detail=detail)
            running, code = await handle.status()
            if not running:
                return ServiceState("exited", code)
            await asyncio.sleep(check.interval_s)


async def drive_stack(
    backend: StackBackend,
    spec: StackSpec,
    *,
    id: str,
    on_stop: Callable[[], None] | None = None,
    host_loopback: bool = True,
) -> DrivenStack:
    """Start ``spec``'s services over ``backend`` (the module docstring says how) and return the running stack; on any
    failure, and when cancelled, stop what started and ``backend``'s own resources first.  ``on_stop`` runs once the
    stack has stopped; ``host_loopback`` says whether the backend's services share this host's network."""

    stack = DrivenStack(backend, spec, id=id, on_stop=on_stop, host_loopback=host_loopback)
    try:
        await stack._start()
    except BaseException:
        await asyncio.shield(stack.stop())
        raise
    return stack


async def _run_check(handle: ServiceHandle, check: Healthcheck) -> tuple[bool, str]:
    """Run ``check`` once in the service's sandbox: whether it passed, and what it said when it did not."""

    try:
        result: ExecResult = await handle.sandbox.exec(list(check.command), timeout_s=check.timeout_s)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[-_HEALTH_DETAIL_CHARS:]
    if result.exit_code == 0 and not result.timed_out:
        return True, ""
    if result.timed_out:
        return False, f"it timed out after {check.timeout_s:g}s"
    output = (result.stderr or result.stdout).strip()
    return False, (f"exit {result.exit_code}" + (f": {output}" if output else ""))[-_HEALTH_DETAIL_CHARS:]


def _exited(code: int | None) -> str:
    return "exited" + (f" with status {code}" if code is not None else "")


__all__ = [
    "CONDITIONS",
    "LOOPBACK",
    "LOOPBACK_HOSTS",
    "MAIN",
    "SERVICE_LABEL",
    "SOCKET_ADDRESS_BYTES",
    "STACKS",
    "STACK_LABEL",
    "Condition",
    "Dependency",
    "DrivenStack",
    "Healthcheck",
    "Reference",
    "RunMode",
    "ServiceHandle",
    "ServiceSpec",
    "ServiceState",
    "SocketLinks",
    "Stack",
    "StackBackend",
    "StackNetwork",
    "StackProvider",
    "StackSpec",
    "StackStartError",
    "StackVolumes",
    "Volume",
    "dependency_cycle",
    "drive_stack",
    "longest_path",
    "make_volumes",
    "service_references",
]
