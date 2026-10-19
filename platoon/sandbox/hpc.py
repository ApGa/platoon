"""Providers over the ``hpc-sandbox`` package: rootless HPC sandboxes on Singularity/Apptainer or enroot.

``hpc_sandbox`` mirrors :mod:`platoon.sandbox.base` without importing platoon: its ``SandboxSpec``, ``Mount``,
``ExecResult``, ``SandboxSnapshot``, ``DownloadReport`` and ``DroppedLink`` have the same fields, which the adapter
translates one by one, and its ``CWD_ERROR_*`` constants the same values.
The package is imported only when a provider is built (the ``hpc`` extra: ``pip install 'platoon[hpc]'``, or
``uv sync`` with ``--extra hpc`` among the extras it is given in a platoon checkout), and the provider's options
(``scratch_root``, ``image_cache``, ...) pass through to the ``hpc_sandbox`` provider it wraps.  An adapter takes
its capabilities from that provider, refuses to restore another provider's snapshot, and, as the other providers
do, deletes the snapshots a sandbox took when that sandbox stops (when the inner provider can).  Errors pass
through as they are: a failure of the sandbox rather than of the command raises ``hpc_sandbox.SandboxError``
(``SandboxLostError`` when the container died) instead of returning an exit status, and so does a command that never
started for a reason of the sandbox's own.  A command whose working directory (the spec's ``workdir`` when ``exec``
gets no ``cwd``) is gone or cannot be entered fails as the command's own failure, as with ``docker exec``: exit
status 126 with hpc-sandbox's line on stderr (:func:`platoon.sandbox.base.cwd_error`), and nothing falls back to
another directory.  ``upload`` and ``download`` run from ``/``, so absolute paths still transfer once the workdir is
gone.  ``download`` returns what it left out (:class:`~platoon.sandbox.base.DownloadReport`): hpc-sandbox copies no
symlink that could lead out of the copy, no owner and no setuid, setgid or sticky bit.  Harbor grading checks the
workdir with a command run in ``/`` and scores an attempt whose agents removed it 0 rather than run its tests
(:mod:`platoon.envs.harbor.session`).

Of its provider's capabilities, an adapter keeps those of platoon's contract
(:data:`platoon.sandbox.base.CAPABILITIES`), not hpc-sandbox 0.3's ``pods`` (services that share one network, which
platoon never starts), and adds ``routes`` and ``host_loopback``: it implements the routes, and its sandboxes run on
this node.

A string command runs through ``bash -c``, as :mod:`platoon.sandbox.base` says and the other providers do, not through
hpc-sandbox's ``/bin/sh -c``: the ``bash`` tool of a Harbor task then takes bash syntax (``[[ ]]``, ``source``,
``set -o pipefail``) on every provider, and an image without bash runs only argv commands.

The adapter implements what hpc-sandbox leaves to its caller: routes to Unix-socket upstreams from a sandbox
without a network (a bind of the socket's directory plus a forwarder on port 18080 of the sandbox's own loopback,
:mod:`platoon.sandbox.routes`), this process's labels on every sandbox, and reaping by label
(:meth:`HpcProvider.reap`, hpc-sandbox's janitor: it stops the sessions whose owner process is gone).

:mod:`platoon.sandbox.singularity` and :mod:`platoon.sandbox.enroot` name the two providers.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from platoon.sandbox import cleanup
from platoon.sandbox._processes import kill_other_processes, own_processes
from platoon.sandbox.base import CAPABILITIES, DownloadReport, DroppedLink, ExecResult, SandboxSnapshot, SandboxSpec
from platoon.sandbox.routes import SandboxRoutes

logger = logging.getLogger(__name__)


def import_hpc_sandbox(provider: str) -> Any:
    """The ``hpc_sandbox`` package, or an ImportError that says how to install it."""

    try:
        return importlib.import_module("hpc_sandbox")
    except ImportError as exc:
        raise ImportError(
            f"The {provider} provider needs the hpc-sandbox package: install the hpc extra, e.g. pip install "
            "'platoon[hpc]', or in a platoon checkout uv sync --extra tinker --extra ors --extra hpc (uv sync removes "
            "the extras it is not given)"
        ) from exc


class HpcSandbox:
    """A sandbox of an :class:`HpcProvider`, over the ``hpc_sandbox`` sandbox the provider started.

    Both runtimes give a sandbox its own PID namespace (Singularity only where the site allows PID namespaces), in
    which hpc-sandbox keeps the container alive with processes of its own before any command runs: Singularity's
    ``sinit``, the instance's keep-alive startscript and its ``sleep``; enroot's keep-alive loop as the namespace's
    init and its ``sleep``.  The adapter lists them when the sandbox starts, before its routes' forwarders, and
    :meth:`kill_processes` kills every other process from inside the sandbox (:mod:`platoon.sandbox._processes`),
    or refuses to in a sandbox that shares platoon's PID namespace.
    """

    def __init__(self, provider: HpcProvider, inner: Any, *, routes: SandboxRoutes | None = None) -> None:
        self.id: str = inner.id
        self._provider = provider
        self._inner = inner
        self._snapshots: list[Any] = []
        self._routes = routes
        self._token: int | None = None
        self._own_processes: frozenset[str] | None = None

    @property
    def routes(self) -> dict[str, str]:
        """Upstream name -> the URL a process in the sandbox calls (:mod:`platoon.sandbox.routes`)."""

        return self._routes.routes if self._routes is not None else {}

    async def ensure_routes(self) -> dict[str, str]:
        """Start again any route forwarder that is gone; returns :attr:`routes`."""

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
        # platoon's contract runs a string through bash, as docker and the process provider do; hpc-sandbox's own
        # string runs through /bin/sh, which is dash in Debian and Ubuntu images and rejects bash's syntax.
        argv = ["bash", "-c", cmd] if isinstance(cmd, str) else cmd
        result = await self._inner.exec(argv, timeout_s=timeout_s, env=env, cwd=cwd, stdin=stdin)
        return ExecResult(
            exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr, timed_out=result.timed_out
        )

    async def upload(self, src: str | Path, dst: str) -> None:
        await self._inner.upload(src, dst)

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        report = await self._inner.download(src, dst)
        if report is None:
            return None
        return DownloadReport(
            dropped_links=[DroppedLink(link.path, link.target, link.reason) for link in report.dropped_links],
            dropped_total=report.dropped_total,
        )

    async def kill_processes(self) -> None:
        await kill_other_processes(self, self._own_processes)

    async def snapshot(self) -> SandboxSnapshot:
        snapshot = await self._inner.snapshot()
        self._snapshots.append(snapshot)
        return SandboxSnapshot(
            id=snapshot.id,
            provider=snapshot.provider,
            path=snapshot.path,
            image=snapshot.image,
            metadata=dict(snapshot.metadata),
        )

    async def stop(self) -> None:
        await self._inner.stop()
        while self._snapshots:
            await self._provider.delete_inner_snapshot(self._snapshots.pop())
        cleanup.unregister(self._token)
        self._token = None


class HpcProvider:
    """A :class:`~platoon.sandbox.base.SandboxProvider` over ``hpc_sandbox.<inner_class>(**options)``.

    Subclasses set ``name``, the provider name that snapshots carry, and ``inner_class``.
    """

    name: ClassVar[str]
    inner_class: ClassVar[str]
    # A rootless container per sandbox: a site profile may choose it (platoon.site).
    isolates: ClassVar[bool] = True
    # The runtime executables a node needs, any one of them, unless the options name one (``binary``).
    executables: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def unavailable(cls, options: Mapping[str, Any]) -> str | None:
        """Why this host cannot run the provider built with ``options``: no hpc-sandbox package, or no runtime
        executable on PATH; None otherwise.  Read without building the provider."""

        try:
            import_hpc_sandbox(cls.name)
        except ImportError as exc:
            return str(exc)
        wanted = [str(options["binary"])] if options.get("binary") else list(cls.executables)
        if wanted and not any(shutil.which(binary) for binary in wanted):
            return f"no {' or '.join(wanted)} executable is on PATH"
        return None

    def __init__(self, **options: Any) -> None:
        self._hpc_sandbox = import_hpc_sandbox(self.name)
        self.options = dict(options)  # as hpc-sandbox's provider takes them: what reaping rebuilds it from
        self.inner = getattr(self._hpc_sandbox, self.inner_class)(**options)
        # Of the inner provider's capabilities, those of platoon's contract (not hpc-sandbox 0.3's "pods"), and the
        # adapter's own: both runtimes run a sandbox on this node, in this host's network namespace unless it has none
        # of its own.
        inner = frozenset(self.inner.capabilities) & CAPABILITIES
        self.capabilities: frozenset[str] = inner | {"routes", "host_loopback"}

    @classmethod
    def network_none_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why a provider built with ``options`` would refuse a sandbox without a network, or None.  hpc-sandbox
        refuses one wherever its provider lacks the ``network_none`` capability; a subclass whose options decide
        that says so here (:func:`platoon.sandbox.network_none_refusal`)."""

        return None

    @classmethod
    def rootfs_layout_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why a provider built with ``options`` could not give its sandboxes the root filesystem layout they ask
        for, or None (:func:`platoon.sandbox.rootfs_layout_refusal`); a subclass with layouts says so here."""

        return None

    @classmethod
    def root_refusal(cls, options: Mapping[str, Any]) -> str | None:
        """Why a provider built with ``options`` could not run a sandbox's commands as root with other users to
        switch to, or None (:func:`platoon.sandbox.root_refusal`); each subclass says so for its runtime."""

        return None

    async def start(self, spec: SandboxSpec) -> HpcSandbox:
        return await self._open(spec, lambda inner_spec: self.inner.start(inner_spec))

    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> HpcSandbox:
        if snapshot.provider != self.name:
            raise ValueError(
                f"The {self.name} provider cannot restore snapshot {snapshot.id}, which the {snapshot.provider!r} "
                "provider took"
            )
        inner_snapshot = self._hpc_sandbox.SandboxSnapshot(
            id=snapshot.id,
            provider=snapshot.provider,
            path=snapshot.path,
            image=snapshot.image,
            metadata=dict(snapshot.metadata),
        )
        return await self._open(spec, lambda inner_spec: self.inner.restore(inner_snapshot, inner_spec))

    async def reap(self, labels: Mapping[str, str]) -> int:
        """Stop this provider's sandboxes whose labels include ``labels``, and sweep the sessions on this node that
        carry them and whose owner process is gone (hpc-sandbox's janitor); returns how many it stopped or
        removed.  Sessions another live process holds are left alone."""

        if not labels:
            raise ValueError("Reaping needs labels: an empty set would select every abandoned session")
        reap = getattr(self.inner, "reap", None)
        if reap is None:
            return 0
        report = await reap(dict(labels))
        for error in getattr(report, "errors", ()):
            logger.warning("Reaping %s sandboxes labelled %s: %s", self.name, dict(labels), error)
        reaped = {*getattr(report, "stopped", ()), *getattr(report, "removed", ())}
        if reaped:
            logger.info("Reaped %d %s sandbox(es) labelled %s", len(reaped), self.name, dict(labels))
        return len(reaped)

    async def _open(self, spec: SandboxSpec, start: Any) -> HpcSandbox:
        routes = SandboxRoutes(spec, own_network=True)
        labels = {**spec.labels, **cleanup.labels()}
        token = cleanup.register(provider=cleanup.provider_ref(self), options=self.options, labels=labels)
        try:
            inner = await start(self._spec(spec, routes=routes, labels=labels))
        except BaseException:
            cleanup.unregister(token)
            raise
        sandbox = HpcSandbox(self, inner, routes=routes)
        sandbox._token = token
        try:
            # Before any route's forwarder starts: what runs in the sandbox now keeps it alive.
            sandbox._own_processes = await own_processes(sandbox)
            await routes.start(sandbox)
        except BaseException:  # cancelled too: the sandbox must not outlive the start
            await asyncio.shield(sandbox.stop())
            raise
        cleanup.attach(token, sandbox)
        return sandbox

    async def delete_inner_snapshot(self, snapshot: Any) -> None:
        """Delete an ``hpc_sandbox`` snapshot; nothing when the inner provider cannot delete snapshots."""

        delete = getattr(self.inner, "delete_snapshot", None)
        if delete is not None:
            await delete(snapshot)

    def _spec(
        self, spec: SandboxSpec, *, routes: SandboxRoutes | None = None, labels: Mapping[str, str] | None = None
    ) -> Any:
        hpc_sandbox = self._hpc_sandbox
        mounts = [*spec.mounts, *(routes.mounts() if routes is not None else ())]
        return hpc_sandbox.SandboxSpec(
            image=spec.image,
            workdir=spec.workdir,
            env=dict(spec.env),
            mounts=[hpc_sandbox.Mount(source=m.source, target=m.target, read_only=m.read_only) for m in mounts],
            network=spec.network,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
            labels=dict(spec.labels if labels is None else labels),
        )


__all__ = ["HpcProvider", "HpcSandbox", "import_hpc_sandbox"]
