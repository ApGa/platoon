"""Stacks on the ``docker`` provider: each stack a docker compose project (RFC 0010, section 6).

:meth:`DockerProvider.start_stack <platoon.sandbox.docker.DockerProvider.start_stack>` starts a stack with the generic
driver (:func:`platoon.sandbox.stacks.drive_stack`), as every provider does: the services start in dependency order
under compose's three conditions, their healthchecks run in their containers (``docker exec``) by Docker's rules, and
a start that fails or is cancelled removes what it made.  A stack is a compose **project** of its own,
``platoon-stack-<id>``, whose directory below the provider's ``stack_root`` (``<stack_root>/<project>/``, 0700) holds
the project's file (``compose.yaml``), written from the stack's spec, and its volumes.  The file is JSON, which compose
reads as the YAML it also is, so that no value changes its type on the way (YAML 1.2 reads a plain ``1e3`` as a
number).  It gives each service its image, its final command as the ``entrypoint`` (``sleep infinity`` for one that
idles, as ``main`` does), working directory, environment, binds, volumes, network, resources, capabilities and labels,
every ``$`` written ``$$`` so that compose interpolates nothing; it has no ``depends_on`` and no ``healthcheck``, which
the driver applies.  ``stack_root`` must be a directory of this user's own: on a node's shared scratch, another user
who made it could change a project's file before compose reads it; and its path may hold no ``:``, which ends a bind's
source where Docker reads one.

Each service is a container, ``<project>-<service>``: ``docker compose create`` makes it once its dependencies hold
(one create at a time, the first making the project's network), ``docker start`` starts it, and it runs as a
:class:`~platoon.sandbox.docker.DockerSandbox`.  A docker command that changes what the daemon holds (a create, a
start, a network's) runs to its end even when the start that asked for it is given up, and the rollback waits for it:
a CLI killed midway would leave its request to the daemon, which completes it unseen.  ``main`` reaches its upstreams
(the model gateway) as a container without a network does: its socket's directory bound read-only, and a forwarder on
``127.0.0.1:18080`` (:mod:`platoon.sandbox.routes`).  Every service runs as its image's user, with Docker's root free
to switch to any other: the run mode that pods call ``subids``.

**Networks.**  The services share a network of the project's own, ``<project>-internal``, which is ``internal`` and
gives the host no address on it (``inhibit_ipv4``), so that it reaches nothing beyond the stack, the host's own services
included; :func:`compose_refusal` warns of a daemon before 25.0.5, which passes an internal network's DNS queries on to
the host's resolvers.  On it each service answers to its name and aliases, at the ports it declares, as in compose, so
``rewrite`` changes nothing; a service that shares another's network (``network_of``, compose's ``network_mode:
service:<x>``) starts after it, as compose orders them.  While the stack has egress, each service is also connected to
``<project>-egress``, a bridge network that reaches the web (and the host's addresses other than its loopback), made
when the stack first needs it and disconnected again for a phase without egress
(:meth:`~platoon.sandbox.stacks.Stack.set_network`).

**Reaching a service from this host.**  ``connect(service, port)`` starts a **relay**: a container of its own,
``<project>-relay-<service>-<port>``, that shares the network of the service that listens there (``--network
container:``), so that it reaches the port on that network's loopback whatever the stack's networks are, and serves a
Unix socket in the project's directory, ``relay/<service>-<port>.sock``, which only your uid reaches (through a link
in node scratch where that path is too long for a socket's address), also from a rootless daemon, whose networks this
host does not reach.  The relay
runs platoon's forwarder (``platoon/sandbox/forwarder.py``) with the Python 3 of that service's image, else of
``main``'s, as root without capabilities, and carries the stack's labels, so that the teardown and a reap remove it
with the services; a stack whose images have no Python 3 cannot be reached so (a ``RuntimeError`` says why).

**Volumes** are directories of the project's, a named one below ``volumes/`` for every service that names it, a
service's own ones below ``services/``: made, and filled where they have a ``fill``, before any service starts, and
bound into the services, whose users may all write them (0777).  So they start empty, as on every provider, and
``volume_path`` names a directory on this host.  ``tmpfs`` volumes are Docker's.

**Teardown** removes the containers that carry the stack's label, with their anonymous volumes, and then its networks,
whatever the stack knows it made: also what a CLI that failed or timed out left the daemon to make.  Then the directory
goes, what the services wrote there as other users (root, a database's own user) through a container of ``main``'s
image that runs ``rm`` as root, and its record last.  The record (``stack.json``, written first) holds the stack's
labels and its owner, so that :meth:`DockerProvider.reap <platoon.sandbox.docker.DockerProvider.reap>` removes the
stacks a process left, by their containers' labels and by their records; a stack whose teardown failed stays
registered for that.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import urllib.parse
import uuid
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any

from platoon.sandbox import cleanup
from platoon.sandbox._exec import run_command
from platoon.sandbox.base import ExecResult, Mount
from platoon.sandbox.docker import DockerProvider, DockerSandbox, image_reference
from platoon.sandbox.routes import FORWARDER, SandboxRoutes
from platoon.sandbox.stacks import (
    MAIN,
    SERVICE_LABEL,
    STACK_LABEL,
    Dependency,
    DrivenStack,
    ServiceSpec,
    SocketLinks,
    StackNetwork,
    StackSpec,
    Volume,
    drive_stack,
)
from platoon.utils.scratch import private_dir
from platoon.utils.stopping import let_finish, process_start_time

logger = logging.getLogger(__name__)

PROJECT_PREFIX = "platoon-stack-"
#: The file compose reads, and the record reaping reads, in a project's directory.
PROJECT_FILE = "compose.yaml"
STACK_RECORD = "stack.json"
#: How a service idles: a container's command must run for as long as the container.
IDLE = ("sleep", "infinity")
#: The run mode of every service: Docker's root, which can switch to any user, as a pod's ``subids`` mode gives.
RUN_MODE = "subids"
_INTERNAL = "internal"  # the project's network, as its file names it
# The internal network's option that gives the host no address on it, so that no service reaches the host's own
# services (Docker 28's gateway_mode_ipv4: isolated does too, but older daemons refuse it).
_NO_HOST_ADDRESS = {"com.docker.network.bridge.inhibit_ipv4": "true"}
# What a JSON string may hold as it is but YAML, as which compose reads the file, may not, or reads as a line break
# (U+2028, U+2029), which drops the spaces around it: written as JSON's escapes.
_NOT_YAML = re.compile("[\x7f-\x9f\u2028\u2029\ufeff\ufffe\uffff]")
_LOG_LINES = 200  # how much of a container's log a stack's logs() reads
_REMAINS = "/platoon-remains"  # where the container that deletes what services wrote mounts the directory
_DNS_KEPT_IN = (25, 0, 5)  # the first daemon whose internal networks keep their DNS queries in (CVE-2024-29018)
RELAY_DIR = "relay"  # the directory of a project's relay sockets, which each relay binds at _RELAY_TARGET
_RELAY_TARGET = "/run/platoon-relay"
_RELAY_START_TIMEOUT_S = 60.0
# A relay's command: the first Python 3.6 or later on its image's PATH runs platoon's forwarder, $3, which serves the
# socket $1 and relays each connection to port $2 of the loopback; exit 127 when the image has none.
_RELAY_SCRIPT = r"""
for python in python3 python; do
  if command -v "$python" >/dev/null 2>&1 && "$python" -c 'import sys; sys.exit(sys.version_info < (3, 6))'; then
    exec "$python" -I -c "$3" --serve-socket "$1" --port "$2"
  fi
done
echo "no Python 3 to run platoon's relay" >&2
exit 127
"""


async def start_compose_stack(provider: DockerProvider, spec: StackSpec) -> DrivenStack:
    """Start ``spec`` as a compose project (the module docstring says how): registered for cleanup like the provider's
    sandboxes, so that a stopping process stops it and reaping by label removes what it leaves."""

    if spec.externals:
        raise ValueError(
            f"A compose project cannot reach a host socket, and the stack reaches {', '.join(sorted(spec.externals))} "
            "outside it (a node service): run it as a pod (singularity, enroot) or on the process provider"
        )
    spec = _shared_networks_first(spec)
    spec.check()
    name = f"{PROJECT_PREFIX}{uuid.uuid4().hex[:12]}"
    labels = {**spec.labels, **cleanup.labels(), STACK_LABEL: name}
    project = _Project(provider, spec, name, labels)  # ValueError for a spec the provider cannot run
    token = cleanup.register(provider=cleanup.provider_ref(provider), options=provider.options, labels=labels)

    def forget() -> None:  # once the stack stopped: a teardown that left something leaves it to a reap by label
        if project.removed:
            cleanup.unregister(token)

    # The thread runs to its end whatever its caller does: a stopping process spares it, and a start given up
    # meanwhile deletes what it made once it is done, a cleanup of the rollout's own.
    making = let_finish(asyncio.ensure_future(asyncio.to_thread(project.make)))
    try:
        await asyncio.shield(making)
    except BaseException:
        cleanup.unregister(token)
        let_finish(asyncio.ensure_future(_remove_made(making, project.directory)))
        raise
    # Its services share the project's network, not this host's: main reaches the gateway through its route alone.
    stack = await drive_stack(project, spec, id=name, on_stop=forget, host_loopback=False)  # one that fails stops
    cleanup.attach(token, stack)
    return stack


@functools.cache
def compose_refusal(binary: str) -> str | None:
    """Why the docker CLI ``binary`` runs no stack: it has no compose plugin (``docker compose version`` fails); None
    when it has one.  Asked once per process, which also warns of a daemon whose internal networks let their DNS
    queries out (before 25.0.5), since a stack's phases without egress rely on them."""

    try:
        result = subprocess.run([binary, "compose", "version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"the docker CLI ({binary}) cannot tell its compose version: {exc}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-300:]
        return f"the docker CLI ({binary}) has no compose plugin: `{binary} compose version` fails ({detail})"
    _warn_of_dns_out(binary)
    return None


async def reap_compose_stacks(provider: DockerProvider, labels: Mapping[str, str]) -> int:
    """Remove the stacks whose labels include ``labels``: the containers and networks of a stack that carry them, and
    the directories below the provider's ``stack_root`` whose record has them and whose owner is this process or gone
    (a stack whose start was cut short may have no container yet).  Returns how many stacks."""

    filters = [*(f"--filter=label={key}={value}" for key, value in labels.items()), f"--filter=label={STACK_LABEL}"]
    listed = await _run(provider, "ps", "--all", *filters, "--format", f'{{{{.Label "{STACK_LABEL}"}}}}')
    names = set(listed.stdout.split()) if listed.exit_code == 0 else set()
    directories = await asyncio.to_thread(_left_directories, provider.stack_root, labels)
    await _remove_labelled(provider, filters)
    for name, image in sorted(directories.items()):
        await remove_directory(provider, provider.stack_root / name, image=image)
    names |= set(directories)
    if names:
        logger.info("Removed %d docker compose stack(s) labelled %s", len(names), dict(labels))
    return len(names)


async def remove_directory(provider: DockerProvider, directory: Path, *, image: str | None) -> bool:
    """Delete a project's ``directory``: what this process cannot delete there itself (what the services wrote as root
    or as another user) through a container of ``image`` that runs ``rm`` as root, and its record last, so that a reap
    still finds a directory whose contents could not all be deleted.  Whether it is gone."""

    try:
        info = await asyncio.to_thread(directory.lstat)
    except FileNotFoundError:
        return True
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():  # no rm as root is ever pointed elsewhere
        logger.warning("Not deleting %s, which is no directory of this user's own", directory)
        return False
    remains = await asyncio.to_thread(_empty, directory)
    if remains and image is not None:
        await _run(
            provider,
            *("run", "--rm", "--pull=never", "--network=none", "--user=0:0", "--entrypoint=rm"),
            f"--volume={directory}:{_REMAINS}",
            image,
            *("-rf", "--", *(f"{_REMAINS}/{entry}" for entry in remains)),
        )
        remains = await asyncio.to_thread(_empty, directory)
    if remains:
        logger.warning("Cannot delete the docker compose project directory %s", directory)
        return False
    await asyncio.to_thread(shutil.rmtree, directory, True)
    return True


class _Project:
    """One stack's compose project: its directory and file, its containers, networks and volumes (the stack's
    :class:`~platoon.sandbox.stacks.StackBackend`)."""

    def __init__(self, provider: DockerProvider, spec: StackSpec, name: str, labels: dict[str, str]) -> None:
        self.provider = provider
        self.spec = spec
        self.name = name
        self.labels = labels
        self.directory = provider.stack_root / name
        self.network: StackNetwork = spec.network
        self.internal = f"{name}-internal"
        self.egress = f"{name}-egress"
        self.removed = False  # its teardown removed everything
        for service, each in spec.services.items():
            image_reference(each.sandbox.image)  # ValueError for an image docker cannot run
            if each.setup is not None and each.command:
                raise ValueError(
                    f"The service {service!r} has a setup and a command: the docker provider runs a setup only in a "
                    "service that idles, since a service's command is its container's and starts with it"
                )
        self._image = image_reference(spec.services[MAIN].sandbox.image)  # what deletes what services wrote
        # The upstreams of a service (main's gateway), reached as from a container without a network: ValueError for
        # one it cannot reach so.
        self._routes = {
            service: SandboxRoutes(dataclasses.replace(each.sandbox, network="none"), own_network=True)
            for service, each in spec.services.items()
            if each.sandbox.upstreams
        }
        self._volumes = _volume_directories(spec, self.directory)
        self._started: list[str] = []
        self._egress_made = False
        self._creating = asyncio.Lock()  # one create at a time: the first makes the internal network
        self._changes: set[asyncio.Task[ExecResult]] = set()  # the docker commands still changing what the daemon holds
        self._relays: dict[tuple[str, int], Path] = {}  # each relay's socket, by the service whose port it reaches
        self._relaying = asyncio.Lock()  # one relay started at a time
        self._links = SocketLinks()

    def container(self, service: str) -> str:
        return f"{self.name}-{service}"

    # Making the project (in a thread) ---------------------------------------------------------------------------

    def make(self) -> None:
        """Make the project's directory, its record (first, so that a reap finds whatever a process killed meanwhile
        leaves), its volumes (filled where they have a fill) and its file."""

        _private_root(self.directory.parent)
        self.directory.mkdir(mode=0o700)
        try:
            record = {
                "labels": self.labels,
                "image": self._image,
                "owner_pid": os.getpid(),
                "owner_start_time": process_start_time(os.getpid()),
            }
            (self.directory / STACK_RECORD).write_text(json.dumps(record), encoding="utf-8")
            for directory, volume in self._volumes.made:
                directory.mkdir(parents=True)
                directory.chmod(0o777)  # whichever user a service runs as writes there, as in a new volume
                if volume.fill is not None:
                    volume.fill(directory)
            text = json.dumps(self.document(), indent=2, ensure_ascii=False)
            (self.directory / PROJECT_FILE).write_text(_NOT_YAML.sub(_escaped, text), encoding="utf-8")
        except BaseException:
            shutil.rmtree(self.directory, ignore_errors=True)
            raise

    def document(self) -> dict[str, Any]:
        """The project's file, as compose reads it."""

        internal = {"name": self.internal, "internal": True, "driver_opts": _NO_HOST_ADDRESS, "labels": self.labels}
        return _literal(
            {
                "name": self.name,
                "services": {service: self._service(service, each) for service, each in self.spec.services.items()},
                "networks": {_INTERNAL: internal},
            }
        )

    def _service(self, name: str, spec: ServiceSpec) -> dict[str, Any]:
        sandbox = spec.sandbox
        entry: dict[str, Any] = {
            "image": image_reference(sandbox.image),
            "container_name": self.container(name),
            "entrypoint": list(spec.command or IDLE),
            "working_dir": sandbox.workdir,
            "labels": {**sandbox.labels, **self.labels, SERVICE_LABEL: name},
        }
        if sandbox.env:
            entry["environment"] = dict(sandbox.env)
        routes = self._routes.get(name)
        mounts = [*sandbox.mounts, *(routes.mounts() if routes is not None else ())]
        volumes = [_bind(mount) for mount in mounts]
        volumes += [self._volumes.mount(name, number, volume) for number, volume in enumerate(spec.volumes)]
        if volumes:
            entry["volumes"] = volumes
        if spec.network_of is not None:
            entry["network_mode"] = f"service:{spec.network_of}"
        else:
            entry["networks"] = {_INTERNAL: {"aliases": list(spec.aliases)} if spec.aliases else {}}
        entry.update(_runtime_options(spec))
        return entry

    # The stack's backend ----------------------------------------------------------------------------------------

    async def start_service(self, name: str, spec: ServiceSpec) -> _Service:
        container = self.container(name)
        try:
            async with self._creating:
                created = await self._run_through(self._compose("create", "--no-recreate", name))
                _checked(created, f"create the service {name}")
                if self.network == "egress" and spec.network_of is None:
                    await self._make_egress()
                    connected = await self._run_through(self._docker("network", "connect", self.egress, container))
                    _checked(connected, f"connect {container} to {self.egress}")
                self._started.append(name)
            _checked(await self._run_through(self._docker("start", container)), f"start {container}")
            routes = self._routes.get(name)
            sandbox = DockerSandbox(
                container,
                spec.sandbox,
                binary=self.provider.binary,
                cli_timeout_s=self.provider.cli_timeout_s,
                routes=routes,
            )
            if spec.command is None:  # it idles, as main does: what runs in it now keeps it alive
                await sandbox.note_own_processes()
            if routes is not None:
                await routes.start(sandbox)
            if spec.setup is not None:
                await spec.setup(sandbox)
        except BaseException:
            # A cleanup of its own, which a stopping process spares: the stack's stop waits for this rollback.
            await asyncio.shield(let_finish(asyncio.ensure_future(self._remove(container))))
            raise
        return _Service(self, sandbox)

    async def set_network(self, network: StackNetwork) -> None:
        async with self._creating:
            self.network = network  # what a service that starts from now on gets, also if a switch below fails
            if network == "egress":
                await self._make_egress()
            if self._egress_made:
                attached = [name for name in self._started if self.spec.services[name].network_of is None]
                switches = [self._switch(name, network) for name in attached]
                results = await asyncio.gather(*switches, return_exceptions=True)  # each tried, whatever one raises
                if errors := [error for error in results if isinstance(error, BaseException)]:
                    raise errors[0]

    async def connect(self, service: str, port: int) -> str:
        """A Unix socket on this host that reaches ``port`` of ``service``, through the relay of the service whose
        network it is (the module docstring says how), started the first time."""

        owner = service
        while (shared := self.spec.services[owner].network_of) is not None:
            owner = shared
        async with self._relaying:
            socket = self._relays.get((owner, port))
            if socket is None:
                socket = self._relays[(owner, port)] = await self._start_relay(owner, port)
        return f"unix://{await self._links.short(socket, socket.name)}"

    async def _start_relay(self, owner: str, port: int) -> Path:
        """Start the relay to ``port`` of ``owner``'s network, with the image of ``owner``, else of ``main``, that has a
        Python 3, and return its socket on this host once it accepts connections."""

        directory = self.directory / RELAY_DIR
        await asyncio.to_thread(_relay_directory, directory)
        name = f"{owner}-{port}.sock"
        container = f"{self.name}-relay-{owner}-{port}"
        images = dict.fromkeys(image_reference(self.spec.services[each].sandbox.image) for each in (owner, MAIN))
        failures = []
        for image in images:
            run = self._docker(
                *("run", "--detach", f"--name={container}", f"--network=container:{self.container(owner)}"),
                *("--user=0:0", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pull=never"),
                *(f"--label={key}={value}" for key, value in self.labels.items()),
                *("--entrypoint=/bin/sh", f"--volume={directory}:{_RELAY_TARGET}", image),
                *("-c", _RELAY_SCRIPT, "platoon-relay", f"{_RELAY_TARGET}/{name}", str(port), _forwarder_source()),
            )
            result = await self._run_through(run)
            if result.exit_code == 0 and not result.timed_out:
                failure = await self._relay_failure(container, directory / name)
                if failure is None:
                    return directory / name
            else:
                failure = "it timed out" if result.timed_out else result.stderr.strip()
            failures.append(f"with {image}, {failure}")
            await self._remove(container)
        raise RuntimeError(f"Could not relay this host to port {port} of the service {owner}: {'; '.join(failures)}")

    async def _relay_failure(self, container: str, socket: Path) -> str | None:
        """None once the relay ``container`` serves ``socket``; else why it does not: its command exited, or it did not
        listen in time."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _RELAY_START_TIMEOUT_S
        while not await asyncio.to_thread(os.path.exists, socket):
            running, code = await self.state(container)
            if not running:
                log = (await self.logs(container, 500)).strip()
                return f"its relay exited with status {code}" + (f": {log}" if log else "")
            if loop.time() > deadline:
                return f"its relay did not listen within {_RELAY_START_TIMEOUT_S:g}s"
            await asyncio.sleep(0.05)
        return None

    def rewrite(self, value: str, *, service: str | None = None) -> str:
        """The services reach one another by name on the project's network: nothing to rewrite."""

        return value

    def volume_path(self, name: str) -> Path:
        return self._volumes.named[name]

    async def close(self) -> None:
        await self._settled()  # what a start given up asked of the daemon is there to remove, or never will be
        await self._links.close()
        removed = await _remove_labelled(self.provider, [f"--filter=label={STACK_LABEL}={self.name}"])
        removed = await remove_directory(self.provider, self.directory, image=self._image) and removed
        if not removed:
            logger.warning("The docker compose project %s is not gone: a reap by its labels removes it", self.name)
        self.removed = removed

    # The CLI ----------------------------------------------------------------------------------------------------

    async def state(self, container: str) -> tuple[bool, int | None]:
        """Whether the container runs, and its command's exit status once it exited; a ``RuntimeError`` when Docker
        cannot tell (a probe's infrastructure failure, not a service that is down)."""

        result = await self._docker("inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", container)
        if result.exit_code != 0 and not result.timed_out and "no such" in result.stderr.lower():
            return False, None  # gone
        status, _, code = _checked(result, f"inspect {container}").stdout.strip().partition(" ")
        if status == "running":
            return True, None
        return False, int(code) if code.lstrip("-").isdigit() else None

    async def logs(self, container: str, max_bytes: int) -> str:
        result = await self._docker("logs", "--tail", str(_LOG_LINES), container)
        return (result.stdout + result.stderr)[-max_bytes:]

    async def _switch(self, service: str, network: StackNetwork) -> None:
        """Connect the service's container to the egress network, or disconnect it, as ``network`` says.  Docker
        refuses to connect a container twice, or to disconnect one that is not connected: one that is where it should
        be counts as switched, so that a switch that failed midway completes when it is tried again."""

        container = self.container(service)
        verb, way = ("connect", "to") if network == "egress" else ("disconnect", "from")
        result = await self._run_through(self._docker("network", verb, self.egress, container))
        if result.exit_code != 0 and (self.egress in await self._networks(container)) == (network == "egress"):
            return
        _checked(result, f"{verb} the service {service} {way} {self.egress}")

    async def _networks(self, container: str) -> dict[str, Any]:
        """The networks the container is on, by name, as ``docker inspect`` tells them (its address on each)."""

        result = _checked(
            await self._docker("inspect", "--format", "{{json .NetworkSettings.Networks}}", container),
            f"look up the networks of {container}",
        )
        return json.loads(result.stdout or "null") or {}

    async def _make_egress(self) -> None:
        if not self._egress_made:
            labels = [f"--label={key}={value}" for key, value in self.labels.items()]
            create = self._docker("network", "create", "--driver=bridge", *labels, self.egress)
            _checked(await self._run_through(create), f"create the network {self.egress}")
            self._egress_made = True

    async def _remove(self, container: str) -> None:
        await self._settled()  # its create or start, given up, is done first
        result = await self._docker("rm", "--force", "--volumes", container)
        if result.exit_code != 0 and "no such container" not in result.stderr.lower():
            logger.warning("Cannot remove the container %s: %s", container, result.stderr.strip())

    async def _run_through(self, command: Coroutine[Any, Any, ExecResult]) -> ExecResult:
        """``command``, a docker command that changes what the daemon holds, run to its end whatever its caller does:
        a CLI killed midway would leave its request to the daemon, which completes it unseen, after a rollback looked.
        A stopping process spares it, and the rollback and the teardown wait for it (:meth:`_settled`)."""

        running = let_finish(asyncio.ensure_future(command))
        self._changes.add(running)
        running.add_done_callback(self._changes.discard)
        return await asyncio.shield(running)

    async def _settled(self) -> None:
        """Return once the docker commands that a start or a switch given up left running are done."""

        if self._changes:
            await asyncio.wait(set(self._changes))

    async def _compose(self, *arguments: str) -> ExecResult:
        file = str(self.directory / PROJECT_FILE)
        argv = [self.provider.binary, "compose", "--project-name", self.name, "--file", file, *arguments]
        return await run_command(argv, timeout_s=self.provider.cli_timeout_s)

    async def _docker(self, *arguments: str) -> ExecResult:
        return await _run(self.provider, *arguments)


class _Service:
    """A started service: its container, as a docker sandbox (a :class:`~platoon.sandbox.stacks.ServiceHandle`)."""

    run_as = RUN_MODE

    def __init__(self, project: _Project, sandbox: DockerSandbox) -> None:
        self.sandbox = sandbox
        self._project = project

    async def status(self) -> tuple[bool, int | None]:
        return await self._project.state(self.sandbox.id)

    async def logs(self, max_bytes: int) -> str:
        return await self._project.logs(self.sandbox.id, max_bytes)

    async def stop(self) -> None:
        await self.sandbox.stop()


@dataclasses.dataclass
class _VolumeDirectories:
    """Where each volume of a stack lives: named ones by name, a service's own ones by (service, position); ``made``
    lists each directory to make once, with the volume whose ``fill`` fills it."""

    named: dict[str, Path]
    own: dict[tuple[str, int], Path]
    made: list[tuple[Path, Volume]]

    def mount(self, service: str, number: int, volume: Volume) -> dict[str, Any]:
        """The project file's entry of a service's ``number``-th volume."""

        if volume.tmpfs and volume.name is None:
            return {"type": "tmpfs", "target": volume.target}
        directory = self.named[volume.name] if volume.name is not None else self.own[(service, number)]
        source = directory / volume.subpath if volume.subpath else directory
        return _bind(Mount(source=str(source), target=volume.target, read_only=volume.read_only))


def _volume_directories(spec: StackSpec, directory: Path) -> _VolumeDirectories:
    """Where each volume of a stack whose project directory is ``directory`` lives: a named one below ``volumes/``, a
    service's own ones below ``services/<service>/``, apart, whatever the names."""

    named: dict[str, Path] = {}
    own: dict[tuple[str, int], Path] = {}
    made: list[tuple[Path, Volume]] = []
    for service, each in spec.services.items():
        for number, volume in enumerate(each.volumes):
            if volume.name is not None:
                if volume.name not in named:  # made and filled for the first service that names it
                    named[volume.name] = directory / "volumes" / urllib.parse.quote(volume.name, safe="")
                    made.append((named[volume.name], volume))
            elif not volume.tmpfs:
                own[(service, number)] = directory / "services" / urllib.parse.quote(service, safe="") / str(number)
                made.append((own[(service, number)], volume))
    return _VolumeDirectories(named, own, made)


def _relay_directory(directory: Path) -> None:
    """Make a project's relay directory, where a relay's root writes its socket whichever uid the daemon maps it to
    (0777: a daemon that remaps users makes it a uid of its own), inside the project's own 0700 directory, which no
    other user enters."""

    directory.mkdir(exist_ok=True)
    directory.chmod(0o777)


@functools.cache
def _forwarder_source() -> str:
    return FORWARDER.read_text(encoding="utf-8")


def _bind(mount: Mount) -> dict[str, Any]:
    return {"type": "bind", "source": mount.source, "target": mount.target, "read_only": mount.read_only}


def _runtime_options(spec: ServiceSpec) -> dict[str, Any]:
    """What the project file gives a service of what only some runtimes apply: its resources, capabilities,
    privileges and stop signal."""

    options: dict[str, Any] = {}
    if spec.sandbox.cpus is not None:
        options["cpus"] = spec.sandbox.cpus
    if spec.sandbox.memory_mb is not None:
        options["mem_limit"] = f"{spec.sandbox.memory_mb}m"
    if spec.shm_mb is not None:
        options["shm_size"] = f"{spec.shm_mb}m"
    if spec.ulimits:
        options["ulimits"] = {name: {"soft": soft, "hard": hard} for name, (soft, hard) in spec.ulimits.items()}
    if spec.cap_add:
        options["cap_add"] = list(spec.cap_add)
    if spec.privileged:
        options["privileged"] = True
    if spec.stop_signal:
        options["stop_signal"] = spec.stop_signal
    return options


def _literal(value: Any) -> Any:
    """``value`` with every ``$`` in its strings written ``$$``, which compose reads as one ``$``: the spec's values are
    final, and compose would otherwise interpolate them again."""

    if isinstance(value, str):
        return value.replace("$", "$$")
    if isinstance(value, Mapping):
        return {key: _literal(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_literal(item) for item in value]
    return value


def _escaped(character: re.Match[str]) -> str:
    return f"\\u{ord(character[0]):04x}"


def _shared_networks_first(spec: StackSpec) -> StackSpec:
    """``spec`` with each service that shares another's network depending on it (started), as compose makes it: a
    container joins the network of one that runs."""

    services = {}
    for name, service in spec.services.items():
        other = service.network_of
        if other is not None and other not in service.depends_on:
            service = dataclasses.replace(service, depends_on={**service.depends_on, other: Dependency()})
        services[name] = service
    return dataclasses.replace(spec, services=services)


def _warn_of_dns_out(binary: str) -> None:
    """Warn when the daemon is older than 25.0.5, whose internal networks pass their containers' DNS queries on to
    the host's resolvers (CVE-2024-29018): a stack's services could send data out that way in a phase without egress."""

    argv = [binary, "version", "--format", "{{.Server.Version}}"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return
    version = re.match(r"(\d+)\.(\d+)\.(\d+)", result.stdout.strip())
    if result.returncode == 0 and version is not None and tuple(map(int, version.groups())) < _DNS_KEPT_IN:
        logger.warning(
            "The Docker daemon is %s: before 25.0.5, an internal network passes its containers' DNS queries on to the "
            "host's resolvers (CVE-2024-29018), so a stack's services may send data out through DNS in a phase "
            "without egress; upgrade Docker, unless your distribution's build carries the fix",
            version[0],
        )


async def _remove_labelled(provider: DockerProvider, filters: list[str]) -> bool:
    """Remove the containers that ``filters`` select (``--filter=label=...``), with their anonymous volumes, then the
    networks; whether none is left."""

    gone = True
    for listing, removal in (
        (("ps", "--all", "--quiet"), ("rm", "--force", "--volumes")),
        (("network", "ls", "--quiet"), ("network", "rm")),
    ):
        listed = await _run(provider, *listing, *filters)
        if listed.exit_code != 0 or listed.timed_out:
            gone = False
        elif found := listed.stdout.split():
            removed = await _run(provider, *removal, *found)
            gone = gone and removed.exit_code == 0 and not removed.timed_out
    return gone


async def _run(provider: DockerProvider, *arguments: str) -> ExecResult:
    return await run_command([provider.binary, *arguments], timeout_s=provider.cli_timeout_s)


def _checked(result: ExecResult, action: str) -> ExecResult:
    """``result``, unless its command failed: then a ``RuntimeError`` that says what could not be done, and why."""

    if result.exit_code != 0 or result.timed_out:
        detail = "it timed out" if result.timed_out else (result.stderr.strip() or result.stdout.strip())
        raise RuntimeError(f"Could not {action}: {detail}")
    return result


def _empty(directory: Path) -> list[str]:
    """Delete what a project's ``directory`` holds but its record, as far as this process can: what is left, by name
    (nothing once the directory is gone)."""

    for entry in _entries(directory):
        path = directory / entry
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                path.unlink()
    return _entries(directory)


def _entries(directory: Path) -> list[str]:
    """What a project's ``directory`` holds but its record, by name; nothing once it is gone."""

    try:
        return sorted(entry for entry in os.listdir(directory) if entry != STACK_RECORD)
    except FileNotFoundError:
        return []


async def _remove_made(making: asyncio.Future[None], directory: Path) -> None:
    """Delete what a project whose start was given up made, once its thread is done (it holds no container yet)."""

    await asyncio.wait([making])
    if making.cancelled() or making.exception() is not None:
        return  # it made nothing it did not delete itself
    await asyncio.to_thread(shutil.rmtree, directory, True)


def _left_directories(root: Path, labels: Mapping[str, str]) -> dict[str, str | None]:
    """The project directories below ``root`` whose record's labels include ``labels`` and whose owner is this process
    or gone, by project, with the image that deletes what their services wrote."""

    left: dict[str, str | None] = {}
    if not os.path.lexists(root):
        return left
    try:
        directories = sorted(_private_root(root).glob(f"{PROJECT_PREFIX}*"))
    except RuntimeError as exc:
        logger.warning("Not reaping the docker compose projects there: %s", exc)
        return left
    for directory in directories:
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
        image = record.get("image")
        left[directory.name] = image if isinstance(image, str) else None
    return left


def _private_root(root: Path) -> Path:
    """``root``, made 0700 where it is missing: the provider's ``stack_root``, which must be a directory of this user's
    (``RuntimeError`` otherwise), since another user of a node's shared scratch who made it, or a symlink in its
    place, could change a project's file before compose reads it, or point a teardown's ``rm`` elsewhere.  Its path
    may hold no ``:``, which ends a bind's source where Docker reads one (compose passes its binds on so)."""

    if ":" in str(root):  # before anything is made: a service with a volume, a directory below it, would not create
        raise RuntimeError(
            f"{root} has a ':', which Docker reads as the end of a bind's source, so no stack's compose project may go "
            "there: set the docker provider's stack_root to a path without one"
        )
    try:
        return private_dir(root)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}, so no stack's compose project may go there: remove it, or set the docker provider's stack_root to "
            "a directory of your own"
        ) from None


__all__ = [
    "IDLE",
    "PROJECT_FILE",
    "PROJECT_PREFIX",
    "RELAY_DIR",
    "RUN_MODE",
    "STACK_RECORD",
    "compose_refusal",
    "reap_compose_stacks",
    "remove_directory",
    "start_compose_stack",
]
