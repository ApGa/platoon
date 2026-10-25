"""Sandbox provider interface: start, exec in, transfer files to and from, snapshot and stop task sandboxes.

Paths passed to a sandbox (``cwd``, ``upload`` destinations, ``download`` sources)
are paths inside the sandbox; local paths are host paths. A string command runs
through ``bash -c``; a list runs as an argv.

Provider capabilities are a subset of {"snapshot", "network_none", "gpu", "isolation", "routes", "host_loopback"}
(:data:`CAPABILITIES`): what platoon may ask of a provider's sandboxes.
``host_loopback`` says that a sandbox on the host network (``network: host``) shares this host's network namespace,
so its processes reach a service on the host's loopback, such as the model gateway, as they are.  A provider whose
sandboxes run elsewhere (on another node, in a cloud) must not declare it: a harness placed in one of its sandboxes
needs a route to the gateway, and without one platoon refuses to place it there.

**Upstreams and routes.** A spec's ``upstreams`` name the host services that processes in the sandbox must reach,
such as the model gateway's Unix socket (``{"gateway": "unix:///abs/dir/gw.sock"}``), and the started sandbox's
``routes`` give, for each one that needs it, the URL that works inside the sandbox (``{"gateway":
"http://127.0.0.1:18080"}``).  The built-in providers implement them as :mod:`platoon.sandbox.routes` describes and
declare the ``routes`` capability; a sandbox on the host network needs no route to a Unix socket on the host, whose
service its processes reach on the host's loopback as they are.

**Reaping.** A provider may also implement :class:`SupportsReap`: stop the sandboxes whose labels include the
given ones, also those a process that died without stopping them left behind (:mod:`platoon.sandbox.cleanup`).

``kill_processes`` ends what commands left running without stopping the sandbox, so that nothing they started runs
alongside a verifier.  Every built-in provider has it; a sandbox of a provider written before it existed may lack it,
and callers check.  ``keep_processes(entries)``, optional too, spares processes from it, each with its descendants:
those a session started for the task before any agent ran ("<pid>:<start time>", as ``/proc`` in the sandbox lists
them), such as a main-only compose file's command, the task's init.  The Harbor service refuses such a task on a
sandbox without it.

**A working directory that cannot be entered.**  A command whose working directory (``cwd``, else the spec's
``workdir``) is gone, is not a directory or cannot be entered never starts, and fails like a command: the docker and
hpc-sandbox providers return exit status 126 (:data:`CWD_ERROR_EXIT_CODE`), as ``docker exec`` does, and the
``process`` provider raises the host's ``OSError``.  hpc-sandbox (0.2 on) also says so: empty stdout and, on stderr,
only :data:`CWD_ERROR_PREFIX` + ``"<directory>: <reason>"``, which :func:`cwd_error` recognizes, so a caller can tell
a command that never started from one that exited 126 itself.

**What a download leaves out.**  ``download`` returns a :class:`DownloadReport` where the provider says what it left
out of the copy (hpc-sandbox: symlinks that could lead outside it, absolute ones or ones that climb out with ``..``),
and ``None`` where it does not say (``process`` and ``docker`` copy symlinks as they are).  A link a download keeps
may still reach what its destination held before, so download into a new or empty directory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

#: Every capability a provider may declare (docs/extending/sandbox-providers.md says what each means).
CAPABILITIES = frozenset({"snapshot", "network_none", "gpu", "isolation", "routes", "host_loopback"})


@dataclass
class Mount:
    source: str  # host path
    target: str  # sandbox path
    read_only: bool = True


@dataclass
class SandboxSpec:
    image: str  # "docker://repo:tag", a .sif (singularity) or .sqsh (enroot) path, or "local" (process provider)
    workdir: str = "/app"
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[Mount] = field(default_factory=list)
    network: Literal["host", "none"] = "host"
    cpus: float | None = None
    memory_mb: int | None = None
    labels: dict[str, str] = field(default_factory=dict)  # rollout_id etc., for janitors
    # name -> "unix:///abs/dir/name.sock" (a host Unix socket) or "http(s)://host:port"; see Sandbox.routes
    upstreams: dict[str, str] = field(default_factory=dict)


@dataclass
class ExecResult:
    exit_code: int  # provider-specific after a timeout; check timed_out
    stdout: str
    stderr: str
    timed_out: bool = False


# The exit status of a command that could not enter its working directory, on docker (as `docker exec` gives it) and
# on hpc-sandbox; a command may also exit with it by itself.
CWD_ERROR_EXIT_CODE = 126
# hpc-sandbox's stderr for one, whole: this prefix, the directory, ": " and one of the reasons (hpc_sandbox's constants
# of the same names, which tests/sandbox/test_sandbox_hpc.py keeps equal to these).
CWD_ERROR_PREFIX = "hpc-sandbox: cannot enter the working directory "
CWD_ERROR_REASONS = (
    "it does not exist or cannot be reached",
    "not a directory",
    "permission denied",
)


def cwd_error(result: ExecResult) -> str | None:
    """The provider's line saying that the command never started because it could not enter its working directory,
    as hpc-sandbox reports it (exit status 126, empty stdout, and on stderr only that line); None for any other
    result, a command's own exit status 126 included, and for providers that do not say why (docker)."""

    if result.exit_code != CWD_ERROR_EXIT_CODE or result.timed_out or result.stdout:
        return None
    if not result.stderr.startswith(CWD_ERROR_PREFIX):
        return None
    line = result.stderr.removesuffix("\n")  # the directory may hold line breaks: all of stderr is the one line
    return line if line.endswith(tuple(f": {reason}" for reason in CWD_ERROR_REASONS)) else None


@dataclass(frozen=True)
class DroppedLink:
    """A symlink a download left out of its copy, so that nothing on the host follows it out of the copy.

    ``path`` is relative to the download's destination, ``target`` is the link's text, and ``reason`` says why
    (hpc-sandbox: ``"absolute"``, ``"absolute into the tree"``, ``"escapes with .."`` or ``"climbs after a name"``).
    """

    path: str
    target: str
    reason: str


@dataclass
class DownloadReport:
    """What a download left out: the first links (hpc-sandbox lists at most 100) and how many in all."""

    dropped_links: list[DroppedLink] = field(default_factory=list)
    dropped_total: int = 0


@dataclass
class SandboxSnapshot:
    """A restorable copy of a sandbox's files (not its processes); valid until the sandbox it came from stops."""

    id: str
    provider: str  # the provider that can restore it
    path: str  # where that provider keeps it: a host directory, or a committed image for docker
    image: str  # the image of the sandbox it came from
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Sandbox(Protocol):
    id: str

    @property
    def routes(self) -> Mapping[str, str]:
        """Upstream name -> the URL a process inside calls to reach it (``{"gateway": "http://127.0.0.1:18080"}``);
        only the upstreams that need a route have one.  Sandboxes of providers that predate routes may lack it, so
        callers read it with ``getattr(sandbox, "routes", {})``."""
        ...

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        """Run a command to completion; on timeout it is killed and ``timed_out`` is set."""
        ...

    async def upload(self, src: str | Path, dst: str) -> None:
        """Copy a host file to ``dst``, or a host directory's contents into ``dst``."""
        ...

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        """Copy a sandbox file to ``dst``, or a sandbox directory's contents into ``dst``; what the copy left out, or
        None from a provider that does not say.  Raises FileNotFoundError when ``src`` does not exist."""
        ...

    async def snapshot(self) -> SandboxSnapshot:
        """Capture the filesystem; only if "snapshot" is in the provider's capabilities."""
        ...

    async def kill_processes(self) -> None:
        """SIGKILL every process the commands started, detached ones included, and keep the sandbox running with its
        files; raises RuntimeError when one it found survives or it cannot tell them from the sandbox's own."""
        ...

    async def stop(self) -> None:
        """Kill everything running in the sandbox and release it; idempotent."""
        ...


@runtime_checkable
class SandboxProvider(Protocol):
    name: str
    capabilities: frozenset[str]

    async def start(self, spec: SandboxSpec) -> Sandbox: ...

    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> Sandbox: ...


@runtime_checkable
class SupportsReap(Protocol):
    """A provider that can stop sandboxes by label, its own and those a dead process left behind."""

    async def reap(self, labels: Mapping[str, str]) -> int:
        """Stop and remove every sandbox whose labels include ``labels`` (never empty) that this provider started,
        or that a process which is gone started on this host; returns how many.  Sandboxes that another live
        process holds are left alone."""
        ...
