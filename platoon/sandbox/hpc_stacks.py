"""Stacks on the singularity and enroot providers: each stack an hpc-sandbox pod (RFC 0010, section 6).

A **pod** (hpc-sandbox's ``docs/pods.md``) is a user and network namespace of its own, which a small process, the
holder, keeps, and each service of the stack is a container in it: the services share one loopback, so each reaches
the others by name at the ports they declare, as in a compose project, and nothing outside the pod reaches them.
:meth:`HpcProvider.start_stack <platoon.sandbox.hpc.HpcProvider.start_stack>` starts the pod, then its services with
the generic driver (:func:`platoon.sandbox.stacks.drive_stack`), which orders them and runs their healthchecks:

- **The pod** has the stack's network (``egress``, through slirp4netns, or ``none``), the stack's labels with this
  process's cleanup labels and ``platoon.stack``, the services' aliases among its host names, and the upstreams of the
  services' sandboxes (``main``'s gateway), which the holder relays from ports of the pod's loopback to the host's Unix
  sockets: the first ports from 18080 on that no service declares.  An upstream that is not a Unix socket is refused,
  since nothing in a pod reaches another host service.
- **Each service** runs its image with its final argv (or idles, as ``main`` does), its environment, working
  directory, mounts and memory limit, its ``shm_size`` and ``ulimits``, and a run mode (:data:`RunMode`): ``main`` runs
  as ``root`` (your uid, as a single sandbox's commands do); a sidecar whose image runs its command as a user other
  than root (``USER``) as ``caller``, your own uid without root, who owns the image's files; any other sidecar as
  ``subids``, root with your subordinate ids, which switches users as Docker's root does (the official database images
  give their data to a user of their own and switch to it), where the runtime gives that mode (Singularity and
  Apptainer, not enroot) and your user has subordinate ids, else as ``caller``.  Each service is an
  :class:`~platoon.sandbox.hpc.HpcSandbox`, and ``main``'s routes are the pod's, which nothing in the pod can stop.
- **Volumes** are directories in the pod's own directory, made and filled before any service starts
  (:func:`~platoon.sandbox.stacks.make_volumes`), so that the pod's deletion takes them, with what a ``subids``
  service wrote there as another user, which hpc-sandbox gives back to your uid first.  A ``tmpfs`` is one too.

``set_network`` switches the pod's egress, ``connect(service, port)`` exposes the pod's port to this host as a Unix
socket in the pod's directory (``unix://<path>``, which only your uid can reach; a link to it in node scratch where
that path is too long for a socket's address), ``rewrite`` changes nothing, and ``stop`` stops the pod, and every
service with it.  The stack is registered for cleanup as a sandbox is, until its pod has stopped: a stopping process
stops it, a reap by its labels stops it, and once its process is gone, hpc-sandbox's janitor sweeps what it left.

**What pods need** (:func:`pods_refusal`): hpc-sandbox 0.5, whose pods run services as your uid or with your
subordinate ids, take host aliases and expose ports to the host, and a node that lets an unprivileged user nest user
namespaces (the pod's, each service's inside it, and its runtime's own).
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import importlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platoon.sandbox import cleanup
from platoon.sandbox.routes import FIRST_PORT, parse_upstreams
from platoon.sandbox.stacks import (
    MAIN,
    SERVICE_LABEL,
    STACK_LABEL,
    DrivenStack,
    RunMode,
    ServiceSpec,
    StackNetwork,
    StackSpec,
    StackVolumes,
    drive_stack,
    make_volumes,
)
from platoon.utils import scratch
from platoon.utils.stopping import let_finish

if TYPE_CHECKING:
    from platoon.sandbox.hpc import HpcProvider, HpcSandbox

logger = logging.getLogger(__name__)

#: The directory of a pod's own that holds the stack's volumes (hpc-sandbox keeps its files beside it).
VOLUMES_DIR = "platoon"
# Three user namespaces, each made in the one before by an unprivileged user (the first with a network namespace): a
# pod's, a service's in it, and the service's runtime's own (Singularity's with run_as="caller", enroot's).
_NESTING_PROBE = """
import os
uid, gid = os.getuid(), os.getgid()
for depth in range(3):
    os.unshare(os.CLONE_NEWUSER | (os.CLONE_NEWNET if depth == 0 else 0))
    for name, text in (("setgroups", "deny"), ("uid_map", f"0 {uid} 1"), ("gid_map", f"0 {gid} 1")):
        with open(f"/proc/self/{name}", "w") as handle:
            handle.write(text)
    uid = gid = 0
"""
_NESTING_TIMEOUT_S = 60.0
_SOCKET_ADDRESS_BYTES = 108  # a Unix socket's path (sun_path), its terminating NUL included
_LINK_BYTES = len("/platoon-pod-XXXXXXXX/65535.sock")  # what a link adds to its root
_CALLERS_NOTED: set[str] = set()  # the providers whose sidecars run as the caller, warned about once


def pods_refusal(provider: str, inner_class: str) -> str | None:
    """Why the ``provider`` provider, over hpc-sandbox's ``inner_class``, would start no stack here: the installed
    hpc-sandbox lacks what its pods need (hpc-sandbox 0.5 has it), or this node does not nest user namespaces; None
    when it starts them, or cannot tell (without hpc-sandbox, building the provider says how to install it)."""

    try:
        hpc_sandbox = importlib.import_module("hpc_sandbox")
    except ImportError:
        return None
    if gaps := _pod_gaps(hpc_sandbox, inner_class):
        version = getattr(hpc_sandbox, "__version__", None) or "an older release"
        # A 0.5 that lacks them is a development build.
        release = "a release of hpc-sandbox 0.5" if version.startswith("0.5") else "hpc-sandbox 0.5"
        return (
            f"the {provider} provider runs a multi-container stack as an hpc-sandbox pod, and the installed "
            f"hpc-sandbox ({version}) has no {_either(gaps)}: install {release}"
        )
    return nesting_refusal()


def _pod_gaps(hpc_sandbox: Any, inner_class: str) -> list[str]:
    """What the ``hpc_sandbox`` package lacks of the pods that stacks run as; nothing, from hpc-sandbox 0.5 on."""

    runtime = getattr(hpc_sandbox, inner_class, None)
    if not callable(getattr(runtime, "start_pod", None)):
        return ["pods"]
    pod = _fields(getattr(hpc_sandbox, "PodSpec", None))
    service = _fields(getattr(hpc_sandbox, "ServiceSpec", None))
    gaps = []
    if "users" not in pod or "run_as" not in service or not callable(getattr(runtime, "map_subordinate_ids", None)):
        gaps.append("pod services that run as your uid or with your subordinate ids")
    if "hosts" not in pod:
        gaps.append("host aliases")
    if "exposes" not in pod or not callable(getattr(getattr(hpc_sandbox, "SessionPod", None), "expose", None)):
        gaps.append("ports exposed to the host")
    return gaps


@functools.cache
def nesting_refusal() -> str | None:
    """Why this node runs no pod: an unprivileged user cannot nest user namespaces as pods do, which a child Python
    tries once per process; None when it can."""

    try:
        probe = subprocess.run(
            [sys.executable, "-I", "-c", _NESTING_PROBE], capture_output=True, text=True, timeout=_NESTING_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"whether this node nests user namespaces, which pods need, could not be told: {exc}"
    if probe.returncode == 0:
        return None
    detail = (probe.stderr.strip().splitlines() or [f"exit {probe.returncode}"])[-1]
    return f"this node does not let an unprivileged user nest user namespaces, which pods need ({detail})"


def runs_as_root(user: str | None) -> bool:
    """Whether an image that runs its command as ``user`` (its ``USER``: a name, a uid or ``uid:gid``) runs it as
    root, as one without a ``USER`` does."""

    name = (user or "").partition(":")[0].strip()
    return name in ("", "root") or (name.isascii() and name.isdigit() and int(name) == 0)


async def start_pod_stack(provider: HpcProvider, spec: StackSpec) -> DrivenStack:
    """Start ``spec`` as a pod of ``provider`` (the module docstring says how), registered for cleanup like a sandbox;
    ``ValueError`` for a spec no pod can run."""

    spec.check()
    upstreams, ports = _pod_upstreams(spec)
    run_modes = await _run_modes(provider, spec)
    stack_id = f"stack-{uuid.uuid4().hex[:12]}"
    labels = {**spec.labels, **cleanup.labels(), STACK_LABEL: stack_id}
    pod_spec = provider.hpc_sandbox.PodSpec(
        network=spec.network,
        upstreams=upstreams,
        upstream_ports=ports,
        labels=labels,
        users="subids" if "subids" in run_modes.values() else "single",
        hosts=_host_aliases(spec),
    )
    token = cleanup.register(provider=cleanup.provider_ref(provider), options=provider.options, labels=labels)
    # Registered until the pod has stopped, also after a failed or cancelled start, whose rollback may outlast it.
    unregister = functools.partial(cleanup.unregister, token)
    backend = await _PodBackend.start(provider, pod_spec, spec, run_modes, labels, on_stop=unregister)
    stack = await drive_stack(backend, spec, id=stack_id, on_stop=unregister, host_loopback=False)
    cleanup.attach(token, stack)
    return stack


def _pod_upstreams(spec: StackSpec) -> tuple[dict[str, str], dict[str, int]]:
    """The upstreams of the stack's services, by name, and the pod's port for each: the first from 18080 on that no
    service declares.  ``ValueError`` for one that is not a Unix socket, or a name given two sockets."""

    upstreams: dict[str, str] = {}
    for name, service in spec.services.items():
        for upstream in parse_upstreams(service.sandbox.upstreams):
            if upstream.socket is None:
                raise ValueError(
                    f"The service {name!r} of a pod names the upstream {upstream.name!r} at {upstream.url}, which "
                    "nothing in a pod reaches: give it as a unix:// socket, which the pod relays"
                )
            if upstreams.setdefault(upstream.name, upstream.url) != upstream.url:
                raise ValueError(f"The services of a pod name two sockets for the upstream {upstream.name!r}")
    declared = {port for service in spec.services.values() for port in service.ports}
    ports: dict[str, int] = {}
    port = FIRST_PORT
    for name in sorted(upstreams):
        while port in declared:
            port += 1
        ports[name] = port
        port += 1
    return upstreams, ports


def _host_aliases(spec: StackSpec) -> list[str]:
    """The names the services answer to besides their own (compose's ``hostname`` and network aliases), which every
    service of a pod resolves to the pod's one loopback address, as it resolves their names."""

    return sorted({alias for service in spec.services.values() for alias in service.aliases} - set(spec.services))


async def _run_modes(provider: HpcProvider, spec: StackSpec) -> dict[str, RunMode]:
    """Each service's run mode (the module docstring says which), mapping your subordinate ids now when a service
    would run with them (once per process: what makes a process's first such pod slow)."""

    def mode(name: str, service: ServiceSpec, subids: bool) -> RunMode:
        if service.run_as is not None:
            return service.run_as
        if name == MAIN:
            return "root"
        return "subids" if subids and runs_as_root(service.user) else "caller"

    wanted = any(mode(name, service, True) == "subids" for name, service in spec.services.items())
    subids = wanted and await _maps_subordinate_ids(provider)
    return {name: mode(name, service, subids) for name, service in spec.services.items()}


async def _maps_subordinate_ids(provider: HpcProvider) -> bool:
    """Whether ``provider``'s pods map your subordinate ids: its runtime gives services that mode (enroot maps one id
    into a container), and your user has them, with the tools that map them, which it maps now unless this process
    did.  A mapping that fails or does not finish in time fails the stack's start."""

    inner, hpc_sandbox = provider.inner, provider.hpc_sandbox
    if "subids" not in getattr(inner.service_type, "run_modes", ()):
        return False
    try:
        await inner.map_subordinate_ids()
    except hpc_sandbox.SandboxStartError:
        raise
    except (hpc_sandbox.SandboxError, ValueError) as exc:
        if provider.name not in _CALLERS_NOTED:
            _CALLERS_NOTED.add(provider.name)
            logger.warning(
                "The %s provider runs the sidecars of stacks as your own uid, not as root with your subordinate ids, "
                "so images that switch users run without switching: %s",
                provider.name,
                exc,
            )
        return False
    return True


class _PodBackend:
    """The services of one stack in a pod, started one by one, and the pod's network, ports and volumes."""

    def __init__(
        self,
        provider: HpcProvider,
        pod: Any,
        volumes: StackVolumes,
        run_modes: Mapping[str, RunMode],
        labels: Mapping[str, str],
    ) -> None:
        self.provider = provider
        self.pod = pod
        self.volumes = volumes
        self.run_modes = dict(run_modes)
        self.labels = dict(labels)
        self._links: Path | None = None  # where connect() links to the pod's sockets, made when one needs it

    @classmethod
    async def start(
        cls,
        provider: HpcProvider,
        pod_spec: Any,
        spec: StackSpec,
        run_modes: Mapping[str, RunMode],
        labels: Mapping[str, str],
        *,
        on_stop: Callable[[], None],
    ) -> _PodBackend:
        """Start the pod, and make and fill the stack's volumes in its directory; on a failure or a cancellation the
        pod stops, once the volumes' thread is done, as a cleanup that a stopping process lets finish, and then
        ``on_stop`` runs (at once for a pod that did not start)."""

        try:
            pod = await provider.inner.start_pod(pod_spec)
        except BaseException:
            on_stop()
            raise
        making = let_finish(asyncio.ensure_future(asyncio.to_thread(make_volumes, spec, Path(pod.path) / VOLUMES_DIR)))
        try:
            volumes = await asyncio.shield(making)
        except BaseException:
            # The thread runs to its end whatever its caller does: the pod stops once it is done, a cleanup of the
            # rollout's own.
            let_finish(asyncio.ensure_future(_stop_when_done(making, pod, on_stop)))
            raise
        return cls(provider, pod, volumes, run_modes, labels)

    async def start_service(self, name: str, spec: ServiceSpec) -> _PodService:
        run_as = self.run_modes[name]
        sandbox = dataclasses.replace(
            spec.sandbox,
            mounts=[*spec.sandbox.mounts, *self.volumes.mounts(name, spec)],
            labels={**self.labels, **spec.sandbox.labels, SERVICE_LABEL: name},
        )
        inner = await self.pod.start_service(
            self.provider.hpc_sandbox.ServiceSpec(
                name=name,
                sandbox=self.provider.inner_spec(sandbox),
                command=list(spec.command) if spec.command is not None else None,
                shm_mb=spec.shm_mb,
                ulimits=dict(spec.ulimits),
                run_as=run_as,
            )
        )
        routes = _PodRoutes({upstream: self.pod.routes[upstream] for upstream in spec.sandbox.upstreams})
        service = self.provider.sandbox_for(inner, routes=routes)
        try:
            if spec.command is None:  # it idles, as main does: what runs there now keeps it alive
                await service.record_own_processes()
            if spec.setup is not None:
                await spec.setup(service)
        except BaseException:
            # A cleanup of its own, which a stopping process spares: the stack's stop waits for this rollback.
            await asyncio.shield(let_finish(asyncio.ensure_future(inner.stop())))
            raise
        return _PodService(self.pod, name, service, inner, run_as)

    async def set_network(self, network: StackNetwork) -> None:
        await self.pod.set_network(network)

    async def connect(self, service: str, port: int) -> str:
        """The pod's ``port``, whichever service listens there (they share one loopback), as a Unix socket that only
        your uid can reach: the pod's own, or a link to it in a private directory of node scratch (else of ``/tmp``)
        when the pod's path is too long for a socket's address."""

        socket = await self.pod.expose(port)
        if len(os.fsencode(socket)) < _SOCKET_ADDRESS_BYTES:
            return f"unix://{socket}"
        if self._links is None:
            links = await asyncio.to_thread(_links_directory)
            if self._links is None:
                self._links = links
            else:  # another connection's made one meanwhile, which the stop removes
                await asyncio.to_thread(links.rmdir)
        link = self._links / f"{port}.sock"
        if not link.is_symlink():
            link.symlink_to(socket)
        return f"unix://{link}"

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        return value  # every service reaches the others by name, at the ports they declare

    def volume_path(self, name: str) -> Path:
        return self.volumes.named[name]

    async def close(self) -> None:
        try:
            await self.pod.stop()
        finally:
            if self._links is not None:
                await asyncio.to_thread(shutil.rmtree, self._links, ignore_errors=True)


class _PodService:
    """A started service of a pod: its sandbox and run mode, and its state and log as the pod tells them."""

    def __init__(self, pod: Any, name: str, sandbox: HpcSandbox, inner: Any, run_as: RunMode) -> None:
        self.sandbox = sandbox
        self.run_as = run_as
        self._pod = pod
        self._name = name
        self._inner = inner

    async def status(self) -> tuple[bool, int | None]:
        if await self._pod.health(self._name) == "exited":
            return False, self._inner.exit_status
        return True, None

    async def logs(self, max_bytes: int) -> str:
        return (await self._pod.logs(self._name)).encode()[-max_bytes:].decode(errors="replace")

    async def stop(self) -> None:
        """Nothing: the pod stops its services with it (the backend's ``close``)."""


class _PodRoutes:
    """A pod service's routes: the pod's, which its holder relays, out of every service's reach."""

    def __init__(self, routes: Mapping[str, str]) -> None:
        self._routes = dict(routes)

    @property
    def routes(self) -> dict[str, str]:
        return dict(self._routes)

    async def ensure(self, sandbox: Any) -> dict[str, str]:
        return self.routes


async def _stop_when_done(making: asyncio.Future[Any], pod: Any, on_stop: Callable[[], None]) -> None:
    await asyncio.wait([making])
    try:
        await pod.stop()
    finally:
        on_stop()


def _links_directory() -> Path:
    """A new private directory for links to a pod's sockets, whose path leaves room in a socket's address for a link's
    name: in node scratch, else in ``/tmp``."""

    for root in (scratch.scratch_dir(), Path("/tmp")):
        if len(os.fsencode(root)) + _LINK_BYTES < _SOCKET_ADDRESS_BYTES:
            return Path(tempfile.mkdtemp(prefix="platoon-pod-", dir=root))
    raise OSError("Neither node scratch nor /tmp has a path short enough for the links to a pod's sockets")


def _fields(kind: Any) -> set[str]:
    return {field.name for field in dataclasses.fields(kind)} if dataclasses.is_dataclass(kind) else set()


def _either(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


__all__ = ["VOLUMES_DIR", "nesting_refusal", "pods_refusal", "runs_as_root", "start_pod_stack"]
