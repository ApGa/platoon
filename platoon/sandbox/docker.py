"""The docker provider: one long-lived container per sandbox, driven through the docker CLI (or podman's).

Killing a ``docker exec`` client leaves its command running in the container,
so every command carries a marker of its own in its environment
(``PLATOON_EXEC``), and a command that times out, or whose caller is
cancelled, is killed inside the container with everything that still carries
its marker (:func:`platoon.sandbox._processes.kill_marked_command`).  Processes that
drop the marker end when the container is removed, or when ``kill_processes``
kills every process in the container but the ones that keep it alive, which
the sandbox lists when it starts, and the ones ``keep_processes`` keeps, with
their descendants (:mod:`platoon.sandbox._processes`).
Snapshots are committed images, removed when the sandbox they came from stops.

A container without a network (``--network none``) reaches a Unix-socket
upstream through its directory, bound at ``/run/platoon/<name>``, and a
forwarder on port 18080 of its own loopback (:mod:`platoon.sandbox.routes`).
Every container carries the sandbox's labels, so :meth:`DockerProvider.reap`
removes the containers a process that is gone left running, with their
anonymous volumes.

The provider also runs stacks, each a docker compose project of its own
(:meth:`DockerProvider.start_stack`, :mod:`platoon.sandbox.compose`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import posixpath
import shutil
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platoon.sandbox import cleanup
from platoon.sandbox._exec import run_command
from platoon.sandbox._processes import KILL_TIMEOUT_S, kill_marked_command, kill_other_processes, own_processes
from platoon.sandbox.base import ExecResult, SandboxSnapshot, SandboxSpec
from platoon.sandbox.routes import SandboxRoutes
from platoon.utils.scratch import node_scratch
from platoon.utils.stopping import let_finish

if TYPE_CHECKING:
    from platoon.sandbox.stacks import DrivenStack, StackSpec

logger = logging.getLogger(__name__)

DEFAULT_CLI_TIMEOUT_S = 600.0  # bounds `docker run` (including an image pull), cp and commit
EXEC_MARKER_ENV = "PLATOON_EXEC"  # each command's own marker, by which a timed-out one is killed in the container
_PATH_KIND_SCRIPT = 'if [ -d "$1" ]; then echo dir; elif [ -e "$1" ]; then echo file; fi'
# How docker and podman report removing a container that does not exist (in lower case).
_NO_SUCH_CONTAINER = "no such container"


def image_reference(image: str) -> str:
    """The docker image for a spec image: ``docker://repo:tag`` or a plain reference."""

    if image.startswith("docker://"):
        return image.removeprefix("docker://")
    if "://" in image or image.endswith(".sif") or image == "local":
        raise ValueError(f"The docker provider cannot run image {image!r}")
    return image


def run_arguments(spec: SandboxSpec, *, name: str, image: str) -> list[str]:
    """``docker run`` arguments for a container that idles until commands are exec'd into it."""

    arguments = ["run", "--detach", "--name", name, "--workdir", spec.workdir, "--network", spec.network]
    arguments += [f"--env={key}={value}" for key, value in spec.env.items()]
    arguments += [f"--label={key}={value}" for key, value in spec.labels.items()]
    for mount in spec.mounts:
        arguments += ["--volume", f"{mount.source}:{mount.target}{':ro' if mount.read_only else ''}"]
    if spec.cpus is not None:
        arguments += ["--cpus", f"{spec.cpus:g}"]
    if spec.memory_mb is not None:
        arguments += ["--memory", f"{spec.memory_mb}m"]
    return [*arguments, "--entrypoint", "sleep", image, "infinity"]


class DockerSandbox:
    def __init__(
        self,
        container: str,
        spec: SandboxSpec,
        *,
        binary: str,
        cli_timeout_s: float,
        routes: SandboxRoutes | None = None,
    ) -> None:
        self.id = container
        self.spec = spec
        self._binary = binary
        self._cli_timeout_s = cli_timeout_s
        self._snapshot_images: list[str] = []
        self._stopped = False
        self._routes = routes
        self._token: int | None = None
        self._own_processes: frozenset[str] | None = None  # the container's own, listed when it starts
        self._kept: frozenset[str] = frozenset()  # what kill_processes spares too, with descendants (keep_processes)

    @property
    def routes(self) -> dict[str, str]:
        """Upstream name -> the URL a process in the container calls (:mod:`platoon.sandbox.routes`)."""

        return self._routes.routes if self._routes is not None else {}

    async def ensure_routes(self) -> dict[str, str]:
        """Start again any route forwarder that is gone; returns :attr:`routes`."""

        self._check_running()
        return await self._routes.ensure(self) if self._routes is not None else {}

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        self._check_running()
        marker = f"{EXEC_MARKER_ENV}={uuid.uuid4().hex}"
        try:
            result = await self._docker_exec(cmd, marker, timeout_s=timeout_s, env=env, cwd=cwd, stdin=stdin)
        except BaseException:  # cancelled: killing the client left the command running in the container
            await asyncio.shield(self._kill_command(marker))
            raise
        if result.timed_out:
            await self._kill_command(marker)
        return result

    async def _docker_exec(
        self,
        cmd: str | list[str],
        marker: str,
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        argv = [self._binary, "exec", "--workdir", self._path(cwd or self.spec.workdir), f"--env={marker}"]
        argv += [f"--env={key}={value}" for key, value in (env or {}).items()]
        if stdin is not None:
            argv.append("--interactive")
        argv += [self.id, *(["bash", "-c", cmd] if isinstance(cmd, str) else cmd)]
        return await run_command(argv, timeout_s=timeout_s, stdin=stdin)

    async def _kill_command(self, marker: str) -> None:
        """Kill what a command whose client was killed left running in the container.  A failure is logged: the
        command's result stands either way, and the container's removal ends it at the latest."""

        if self._stopped:
            return
        kill = kill_marked_command(marker)
        try:
            result = await self._docker_exec(kill, f"{EXEC_MARKER_ENV}=kill", cwd="/", timeout_s=KILL_TIMEOUT_S)
        except Exception as exc:
            result = ExecResult(exit_code=-1, stdout="", stderr=f"{type(exc).__name__}: {exc}")
        if result.exit_code != 0 or result.timed_out:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.exit_code}"
            logger.warning("A timed-out or cancelled command may still run in container %s: %s", self.id, detail)

    async def upload(self, src: str | Path, dst: str) -> None:
        self._check_running()
        source, target = Path(src), self._path(dst)
        parent = target if source.is_dir() else posixpath.dirname(target)
        await self._checked(await self.exec(["mkdir", "-p", parent]), f"create {parent}")
        await self._docker("cp", f"{source}/." if source.is_dir() else str(source), f"{self.id}:{target}")

    async def download(self, src: str, dst: str | Path) -> None:
        self._check_running()
        source, target = self._path(src), Path(dst)
        kind = (await self.exec(["bash", "-c", _PATH_KIND_SCRIPT, "kind", source])).stdout.strip()
        if not kind:
            raise FileNotFoundError(f"No such file or directory in sandbox {self.id}: {source}")
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
            await self._docker("cp", f"{self.id}:{source}/.", str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            await self._docker("cp", f"{self.id}:{source}", str(target))

    async def note_own_processes(self) -> None:
        """Note what runs in the container now as its own, which :meth:`kill_processes` spares: once, as it starts,
        before any command or route forwarder runs."""

        self._own_processes = await own_processes(self)

    def keep_processes(self, entries: Iterable[str]) -> None:
        """Spare the processes ``entries`` ("<pid>:<start time>" in the container) and their descendants from
        :meth:`kill_processes`."""

        self._kept = self._kept | frozenset(entries)

    async def kill_processes(self) -> None:
        self._check_running()
        await kill_other_processes(self, self._own_processes, self._kept)

    async def snapshot(self) -> SandboxSnapshot:
        self._check_running()
        snapshot_id = uuid.uuid4().hex[:12]
        image = f"platoon-snapshot:{snapshot_id}"
        await self._docker("commit", self.id, image)
        self._snapshot_images.append(image)
        return SandboxSnapshot(id=snapshot_id, provider=DockerProvider.name, path=image, image=self.spec.image)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        await run_command([self._binary, "rm", "--force", "--volumes", self.id], timeout_s=self._cli_timeout_s)
        if self._snapshot_images:
            # Best effort: an image that a restored sandbox still runs stays until that sandbox is removed.
            await run_command([self._binary, "rmi", *self._snapshot_images], timeout_s=self._cli_timeout_s)
        cleanup.unregister(self._token)

    def _path(self, path: str) -> str:
        return posixpath.join(self.spec.workdir, path)

    def _check_running(self) -> None:
        if self._stopped:
            raise RuntimeError(f"Sandbox {self.id} is stopped")

    async def _docker(self, *arguments: str) -> None:
        result = await run_command([self._binary, *arguments], timeout_s=self._cli_timeout_s)
        await self._checked(result, f"{arguments[0]} {' '.join(arguments[1:])}")

    @staticmethod
    async def _checked(result: ExecResult, action: str) -> None:
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError(f"Could not {action}: {result.stderr.strip() or result.stdout.strip()}")


class DockerProvider:
    """A container per sandbox, and a compose project per stack (:mod:`platoon.sandbox.compose`), whose directories
    go below ``stack_root`` (default: ``platoon-docker-stacks-<uid>`` in node scratch)."""

    name = "docker"
    capabilities = frozenset(
        {"snapshot", "network_none", "isolation", "routes", "host_loopback", "stacks", "privileged"}
    )
    # A container per sandbox: a site profile may choose it (platoon.site).
    isolates = True

    @classmethod
    def unavailable(cls, options: Mapping[str, Any]) -> str | None:
        """Why this host cannot run the provider built with ``options``: no docker CLI on PATH; None otherwise."""

        binary = options.get("binary") or "docker"
        return None if shutil.which(str(binary)) else f"the docker CLI ({binary}) is not on PATH"

    @classmethod
    def stack_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why the provider built with ``options`` runs no stack: a docker CLI without the compose plugin
        (:func:`platoon.sandbox.stack_refusal`); None when it has it, and when there is no CLI, which building the
        provider reports."""

        from platoon.sandbox.compose import compose_refusal

        binary = shutil.which(str(options.get("binary") or "docker"))
        return compose_refusal(binary) if binary is not None else None

    def __init__(
        self,
        binary: str = "docker",
        cli_timeout_s: float = DEFAULT_CLI_TIMEOUT_S,
        stack_root: str | Path | None = None,
    ) -> None:
        resolved = shutil.which(binary)
        if resolved is None:
            raise RuntimeError(f"The docker provider needs the {binary!r} CLI on PATH")
        self.binary = resolved
        self.cli_timeout_s = cli_timeout_s
        # In node-local scratch, which a node's users may share: a directory of this user's own.
        default_root = node_scratch() / f"platoon-docker-stacks-{os.getuid()}"
        self.stack_root = Path(stack_root) if stack_root is not None else default_root

    @property
    def options(self) -> dict[str, Any]:
        """The options that build this provider again, as a reap of what it started does."""

        return {"binary": self.binary, "cli_timeout_s": self.cli_timeout_s, "stack_root": str(self.stack_root)}

    async def start(self, spec: SandboxSpec) -> DockerSandbox:
        return await self._run(spec, image_reference(spec.image))

    async def start_stack(self, spec: StackSpec) -> DrivenStack:
        """Start a stack as a docker compose project (:mod:`platoon.sandbox.compose`)."""

        from platoon.sandbox.compose import start_compose_stack

        return await start_compose_stack(self, spec)

    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> DockerSandbox:
        if snapshot.provider != self.name:
            raise ValueError(f"Cannot restore a {snapshot.provider!r} snapshot with the docker provider")
        return await self._run(spec, snapshot.path)

    async def reap(self, labels: Mapping[str, str]) -> int:
        """Remove every stack whose labels include ``labels`` (:func:`platoon.sandbox.compose.reap_compose_stacks`),
        then every container whose labels include them, with its anonymous volumes, and the snapshot images committed
        from them; returns how many stacks and containers.  Docker cannot tell which process a container belongs to,
        so ``labels`` must select only sandboxes nobody uses any more (a stopped rollout worker's ``platoon.worker``
        label)."""

        from platoon.sandbox.compose import reap_compose_stacks

        if not labels:
            raise ValueError("Reaping needs labels: an empty set would select every container")
        stacks = await reap_compose_stacks(self, labels)
        filters = [f"--filter=label={key}={value}" for key, value in labels.items()]
        listed = await run_command([self.binary, "ps", "--all", "--quiet", *filters], timeout_s=self.cli_timeout_s)
        containers = listed.stdout.split() if listed.exit_code == 0 else []
        if containers:
            await run_command([self.binary, "rm", "--force", "--volumes", *containers], timeout_s=self.cli_timeout_s)
        images = await run_command(
            [self.binary, "images", "--quiet", "--filter=reference=platoon-snapshot", *filters],
            timeout_s=self.cli_timeout_s,
        )
        if images.exit_code == 0 and images.stdout.split():
            await run_command([self.binary, "rmi", "--force", *images.stdout.split()], timeout_s=self.cli_timeout_s)
        if containers:
            logger.info("Removed %d container(s) labelled %s", len(containers), dict(labels))
        return stacks + len(containers)

    async def _run(self, spec: SandboxSpec, image: str) -> DockerSandbox:
        routes = SandboxRoutes(spec, own_network=True)
        labels = {**spec.labels, **cleanup.labels()}
        token = cleanup.register(provider=cleanup.provider_ref(self), options=self.options, labels=labels)
        container = f"platoon-{uuid.uuid4().hex[:12]}"
        try:
            run_spec = dataclasses.replace(spec, mounts=[*spec.mounts, *routes.mounts()], labels=labels)
            result = await run_command(
                [self.binary, *run_arguments(run_spec, name=container, image=image)], timeout_s=self.cli_timeout_s
            )
            if result.exit_code != 0 or result.timed_out:
                raise RuntimeError(f"Could not start a container from {image}: {result.stderr.strip()}")
        except BaseException:
            # `docker run` may have created the container although it failed, timed out or was cancelled: remove it
            # by name, to the end also when this start is cancelled.
            await asyncio.shield(let_finish(asyncio.ensure_future(self._remove_unstarted(container, token))))
            raise
        sandbox = DockerSandbox(
            container, run_spec, binary=self.binary, cli_timeout_s=self.cli_timeout_s, routes=routes
        )
        sandbox._token = token
        try:
            await sandbox.note_own_processes()  # before any route's forwarder starts
            await routes.start(sandbox)
        except BaseException:  # cancelled too: the container must not outlive the start
            await asyncio.shield(sandbox.stop())
            raise
        cleanup.attach(token, sandbox)
        return sandbox

    async def _remove_unstarted(self, container: str, token: int) -> None:
        """Remove the container of a start that did not succeed, if there is one, and then forget the sandbox.  A
        container that cannot be removed stays registered, so a reap by its labels still finds it."""

        try:
            result = await run_command(
                [self.binary, "rm", "--force", "--volumes", container], timeout_s=self.cli_timeout_s
            )
        except Exception:
            logger.warning("Cannot remove container %s, whose start failed", container, exc_info=True)
            return
        if not result.timed_out and (result.exit_code == 0 or _NO_SUCH_CONTAINER in result.stderr.lower()):
            cleanup.unregister(token)
        else:
            logger.warning("Cannot remove container %s, whose start failed: %s", container, result.stderr.strip())
