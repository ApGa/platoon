"""The process provider: a host directory stands in for the sandbox filesystem and commands run as host processes.

There is no isolation, so it suits tests and laptops; the image, network,
resource limits and read-only mounts are not enforced: a mount is a link to its
source, so what a command writes through a read-only one (a task's files bound
at its working directory, say) reaches that source, a fetched dataset's cache
included. Sandbox paths under the
mapped roots (``/app``, ``/tests``, ``/logs``, ``/solution``, ``/tmp``, the
workdir and mount targets) live under a per-sandbox host directory and are
translated in commands, ``cwd``, env values and file transfers. File contents
are not translated: a script that must also run here can prefix sandbox paths
with ``$PLATOON_SANDBOX_ROOT``, which real sandboxes leave unset.

A spec with ``network: none`` still runs on the host's network, but its
Unix-socket upstreams get routes as in an isolated sandbox: a forwarder on a
free loopback port (:mod:`platoon.sandbox.routes`), so the route works the same
way here as on providers that enforce the network. Each sandbox directory
records its labels, its owner and its commands' process groups, so
:meth:`ProcessProvider.reap` can end the sandboxes of a process that is gone.

Every command runs in a process group of its own, which the sandbox kills when
it stops or :meth:`ProcessSandbox.kill_processes` is called; a process that
left its group (``setsid``) is found by the ``PLATOON_SANDBOX_ROOT`` in its
environment, and one that also dropped that variable escapes both.  The
processes :meth:`ProcessSandbox.keep_processes` keeps (a task's init) are
spared with their descendants and their groups, until the sandbox stops.

The provider also runs stacks, each service a sandbox of its own on the host's network, with ports that each stack holds
for its life (:meth:`ProcessProvider.start_stack`, :mod:`platoon.sandbox.process_stacks`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import posixpath
import re
import shutil
import signal
import time
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

from platoon.sandbox import cleanup
from platoon.sandbox._exec import _has_members, kill_process_group, run_command
from platoon.sandbox.base import ExecResult, SandboxSnapshot, SandboxSpec
from platoon.sandbox.process_stacks import reap_stack_directories, start_process_stack
from platoon.sandbox.routes import SandboxRoutes
from platoon.sandbox.stacks import DrivenStack, StackSpec
from platoon.utils.scratch import node_scratch
from platoon.utils.stopping import let_finish, process_start_time, signal_process_group

logger = logging.getLogger(__name__)

SANDBOX_ROOT_ENV = "PLATOON_SANDBOX_ROOT"
DEFAULT_MAPPED_ROOTS = ("/app", "/tests", "/logs", "/solution", "/tmp")
# In a sandbox's directory, next to fs/: its labels and owner, and the process groups of its commands.
RECORD_FILE = "sandbox.json"
GROUPS_FILE = "process-groups"
# How many times kill_processes looks for processes that left their group, and how long it waits between looks.
_KILL_PASSES = 50
_KILL_PAUSE_S = 0.02
# How long a reap waits for the processes it SIGKILLed to exit, which give back their memory, files and ports first.
_EXIT_WAIT_S = 10.0
_GONE = (b"Z", b"X", b"x")  # a zombie or a dead process holds nothing any more
# A mapped root starts a path when no path character precedes it, and it ends at "/" or a non-path character.
_PATH_START = r"(?<![\w.~/})-])"
_PATH_END = r"(?![\w.-])"


class PathMap:
    """Translates sandbox paths under a set of roots to host paths below ``host_root``."""

    def __init__(self, host_root: Path, roots: Iterable[str]) -> None:
        normalized = {posixpath.normpath(root) for root in roots}
        if any(not posixpath.isabs(root) or root == "/" for root in normalized):
            raise ValueError(f"Mapped roots must be absolute directories other than '/': {sorted(normalized)}")
        self.host_root = str(host_root)
        self._roots = sorted(normalized, key=len, reverse=True)
        alternatives = "|".join(re.escape(root) for root in self._roots)
        self._pattern = re.compile(f"{_PATH_START}(?:{alternatives}){_PATH_END}")

    def to_host(self, path: str) -> str:
        """Map one absolute sandbox path; paths outside the roots, or below ``host_root``, are host paths already."""

        normalized = posixpath.normpath(path)
        if _is_within(normalized, self.host_root):
            return normalized
        if any(_is_within(normalized, root) for root in self._roots):
            return self.host_root + normalized
        return path

    def rewrite(self, text: str) -> str:
        """Map every sandbox path in a command or env value."""

        def replace(match: re.Match[str]) -> str:
            if text.startswith(self.host_root, match.start()):
                return match.group(0)  # already a host path below the root
            return self.host_root + match.group(0)

        return self._pattern.sub(replace, text)


def _is_within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory + "/")


def _kill_marked(entry: bytes, spare: frozenset[str] = frozenset(), killed: dict[int, int] | None = None) -> list[int]:
    """SIGKILL every process of this user whose environment holds ``entry`` (``NAME=value``), but the ``spare`` ones
    ("<pid>:<start time>") and their descendants, looking again until none is left; the ones still there after the
    last look.  ``killed`` collects those it signals (pid -> start time).

    The environment of a process that has begun to exit can no longer be read, so no later look finds it, although it
    still gives back what it held until it has exited (:func:`_wait_exited`)."""

    for _ in range(_KILL_PASSES):
        marked = _marked(entry, spare)
        if not marked:
            return []
        for pid in marked:
            if killed is not None and pid not in killed and (start := process_start_time(pid)) is not None:
                killed[pid] = start
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(_KILL_PAUSE_S)
    return _marked(entry, spare)


def _marked(entry: bytes, spare: frozenset[str] = frozenset()) -> list[int]:
    """The processes of this user but this one whose environment holds ``entry`` (a dead one's reads as empty), but
    the ``spare`` ones and their descendants."""

    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as environ:
                if entry in environ.read().split(b"\0"):
                    pids.append(int(name))
        except OSError:  # gone, or another user's
            continue
    return [pid for pid in pids if not (spare and _descends(pid, spare))]


def _descends(pid: int, roots: frozenset[str]) -> bool:
    """Whether the process ``pid``, or one of its ancestors, is one of ``roots`` ("<pid>:<start time>")."""

    while pid > 1:
        stat = _stat(pid)
        if stat is None:
            return False
        parent, start = stat
        if f"{pid}:{start}" in roots:
            return True
        pid = parent
    return False


def _stat(pid: int) -> tuple[int, int] | None:
    """The parent pid and start time of process ``pid``, None when it is gone."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            fields = handle.read().rsplit(b") ", 1)[1].split()
        return int(fields[1]), int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _wait_exited(groups: Mapping[int, int | None], pids: Mapping[int, int], timeout_s: float) -> list[int]:
    """Wait, at most ``timeout_s``, until no process of the process ``groups`` (pgid -> its leader's start time, None
    when unknown) and none of ``pids`` (pid -> start time) runs any more; the ones that still run then.

    A SIGKILLed process runs on in the kernel while it gives back its memory, which takes a while when it holds much,
    and then its files and ports; only then is it a zombie, or gone."""

    deadline = time.monotonic() + timeout_s
    while running := _running(groups, pids):
        if time.monotonic() >= deadline:
            return running
        time.sleep(_KILL_PAUSE_S)
    return []


def _running(groups: Mapping[int, int | None], pids: Mapping[int, int]) -> list[int]:
    """The processes that run (not a zombie) and are one of ``pids`` or in one of ``groups``: those of
    :func:`_wait_exited`."""

    groups = {group: start for group, start in groups.items() if _has_members(group)}
    if not groups and not pids:  # every group is gone, the usual case: no look through /proc
        return []
    table: dict[int, tuple[bytes, int, int]] = {}  # pid -> state, process group, start time
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as handle:
                fields = handle.read().rsplit(b") ", 1)[1].split()
            table[int(name)] = (fields[0], int(fields[2]), int(fields[19]))
        except (OSError, IndexError, ValueError):  # gone
            continue

    def signalled(group: int) -> bool:
        # A process group's id is the pid of its leader, which no new process gets while the group has a member: a
        # leader that started at another time leads another group, and the one signalled is gone.
        leader = table.get(group)
        return leader is None or groups[group] is None or leader[2] == groups[group]

    return sorted(
        pid
        for pid, (state, group, start) in table.items()
        if state not in _GONE and (pids.get(pid) == start or (group in groups and signalled(group)))
    )


def _root_marker(root: Path) -> bytes:
    """The environment entry every command of the sandbox whose filesystem is ``root`` carries."""

    return os.fsencode(f"{SANDBOX_ROOT_ENV}={root}")


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


class ProcessSandbox:
    """A sandbox whose filesystem is ``<base_dir>/fs`` and whose commands are host processes."""

    def __init__(self, base_dir: Path, spec: SandboxSpec) -> None:
        self.id = base_dir.name
        self.spec = spec
        self.root = base_dir / "fs"
        self._base_dir = base_dir
        self._paths = PathMap(self.root, (*DEFAULT_MAPPED_ROOTS, spec.workdir, *(m.target for m in spec.mounts)))
        self._process_groups: set[int] = set()
        self._kept: frozenset[str] = frozenset()  # processes kill_processes spares, with their descendants
        self._kept_groups: set[int] = set()  # their process groups, which only stop kills
        self._stopped = False
        self._routes: SandboxRoutes | None = None
        self._token: int | None = None

    @property
    def routes(self) -> dict[str, str]:
        """Upstream name -> the URL a process in the sandbox calls (:mod:`platoon.sandbox.routes`)."""

        return self._routes.routes if self._routes is not None else {}

    async def ensure_routes(self) -> dict[str, str]:
        """Start again any route forwarder that is gone; returns :attr:`routes`."""

        self._check_running()
        return await self._routes.ensure(self) if self._routes is not None else {}

    def host_path(self, path: str) -> Path:
        """The host path behind a sandbox path; relative paths start at the workdir."""

        return Path(self._paths.to_host(posixpath.join(self.spec.workdir, path)))

    @classmethod
    def create(cls, base_dir: Path, spec: SandboxSpec, source: Path | None = None) -> ProcessSandbox:
        """Lay out a sandbox below ``base_dir``, starting from a snapshot directory when given."""

        sandbox = cls(base_dir, spec)
        if source is not None:
            shutil.copytree(source, sandbox.root, symlinks=True)
        else:
            sandbox.root.mkdir(parents=True)
        record = {"labels": spec.labels, "owner_pid": os.getpid(), "owner_start_time": process_start_time(os.getpid())}
        (base_dir / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
        for mount in spec.mounts:  # before the workdir, which may be a mount's target (a stack's volume)
            link = sandbox.host_path(mount.target)
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(Path(mount.source).resolve())
        for directory in (spec.workdir, "/tmp"):
            if not any(
                _is_within(posixpath.normpath(directory), mount.target) for mount in spec.mounts if mount.read_only
            ):
                sandbox.host_path(directory).mkdir(parents=True, exist_ok=True)  # never into a read-only mount's source
        return sandbox

    def _environment(self, env: dict[str, str] | None) -> dict[str, str]:
        environment = {**os.environ, "TMPDIR": str(self.host_path("/tmp")), SANDBOX_ROOT_ENV: str(self.root)}
        for key, value in {**self.spec.env, **(env or {})}.items():
            environment[key] = self._paths.rewrite(value)
        return environment

    def _check_running(self) -> None:
        if self._stopped:
            raise RuntimeError(f"Sandbox {self.id} is stopped")

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
        if isinstance(cmd, str):
            argv = ["bash", "-c", self._paths.rewrite(cmd)]
        else:
            argv = [self._paths.rewrite(arg) for arg in cmd]
        return await run_command(
            argv,
            timeout_s=timeout_s,
            env=self._environment(env),
            cwd=self.host_path(cwd or self.spec.workdir),
            stdin=stdin,
            process_groups=self._process_groups,
            on_group=self._record_group,
        )

    def _record_group(self, pgid: int) -> None:
        """Note a command's process group on disk, where :meth:`ProcessProvider.reap` finds it."""

        try:
            with (self._base_dir / GROUPS_FILE).open("a", encoding="utf-8") as groups:
                groups.write(f"{pgid} {process_start_time(pgid)}\n")
        except OSError:
            pass

    async def upload(self, src: str | Path, dst: str) -> None:
        self._check_running()
        await asyncio.to_thread(_copy, Path(src), self.host_path(dst))

    async def download(self, src: str, dst: str | Path) -> None:
        self._check_running()
        await asyncio.to_thread(_copy, self.host_path(src), Path(dst))

    def keep_processes(self, entries: Iterable[str]) -> None:
        """Spare the processes ``entries`` ("<pid>:<start time>", host pids) and their descendants from
        :meth:`kill_processes`, and their process groups too, until the sandbox stops."""

        self._check_running()
        entries = tuple(entries)
        self._kept = self._kept | frozenset(entries)
        for entry in entries:
            try:
                group = os.getpgid(int(entry.split(":", 1)[0]))
            except (OSError, ValueError):
                continue
            if group in self._process_groups:
                self._process_groups.discard(group)
                self._kept_groups.add(group)

    async def kill_processes(self) -> None:
        """SIGKILL the process group of every command, and every process left of them that carries this sandbox's
        ``PLATOON_SANDBOX_ROOT`` (one that moved to a group of its own), but the kept processes
        (:meth:`keep_processes`); the sandbox's files stay.  Raises RuntimeError when such a process is still there
        after the last look."""

        self._check_running()
        self._kill_groups()
        if left := await asyncio.to_thread(_kill_marked, _root_marker(self.root), self._kept):
            pids = " ".join(map(str, left))
            raise RuntimeError(
                f"Could not kill the processes left in sandbox {self.id}: still alive after SIGKILL: {pids}"
            )

    async def snapshot(self) -> SandboxSnapshot:
        self._check_running()
        snapshot_id = uuid.uuid4().hex[:12]
        target = self._base_dir / "snapshots" / snapshot_id
        await asyncio.to_thread(shutil.copytree, self.root, target, symlinks=True)
        return SandboxSnapshot(id=snapshot_id, provider=ProcessProvider.name, path=str(target), image=self.spec.image)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._kill_groups(kept=True)
        await asyncio.to_thread(self._remove)  # one thread: it finishes even when the caller stops waiting
        cleanup.unregister(self._token)

    def _kill_groups(self, *, kept: bool = False) -> None:
        groups = list(self._process_groups)
        self._process_groups.clear()
        if kept:
            groups += self._kept_groups
            self._kept_groups.clear()
        for pgid in groups:
            kill_process_group(pgid)

    def _remove(self) -> None:
        _kill_marked(_root_marker(self.root))
        shutil.rmtree(self._base_dir, ignore_errors=True)


class ProcessProvider:
    """Sandboxes as host directories below ``root``; snapshots are directory copies; stacks of such sandboxes."""

    name = "process"
    capabilities = frozenset({"snapshot", "routes", "host_loopback", "stacks"})
    # Commands run on the host as this user: a site profile may not choose it (platoon.site), and platoon train
    # refuses it for a Harbor entry unless the run config names it.
    isolates = False

    def __init__(self, root: str | Path | None = None) -> None:
        # In node-local scratch, which a node's users may share: a directory of this user's own.
        self.root = Path(root) if root is not None else node_scratch() / f"platoon-sandboxes-{os.getuid()}"

    @classmethod
    def root_refusal(cls, options: Mapping[str, object]) -> str | None:
        """Commands run on the host as this process's user, never as a sandbox's root
        (:func:`platoon.sandbox.root_refusal`)."""

        return "the process provider runs commands on the host as this process's user, not as a sandbox's root"

    async def start(self, spec: SandboxSpec) -> ProcessSandbox:
        return await self._open(spec)

    async def start_stack(self, spec: StackSpec) -> DrivenStack:
        """Start a stack, each of its services a sandbox of this provider (:mod:`platoon.sandbox.process_stacks`)."""

        return await start_process_stack(self, spec)

    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> ProcessSandbox:
        if snapshot.provider != self.name:
            raise ValueError(f"Cannot restore a {snapshot.provider!r} snapshot with the process provider")
        return await self._open(spec, Path(snapshot.path))

    async def reap(self, labels: Mapping[str, str]) -> int:
        """End every sandbox below ``root`` whose labels include ``labels`` and whose owner is this process or gone:
        SIGKILL the process groups its commands left, and delete its directory, and the directories of such stacks
        (their volumes).  Returns how many, once the processes it killed have exited, or after at most 10 seconds of
        waiting for them (a warning names those still running)."""

        if not labels:
            raise ValueError("Reaping needs labels: an empty set would select every sandbox")
        return await asyncio.to_thread(self._reap, dict(labels))

    def _reap(self, labels: dict[str, str]) -> int:
        reaped = 0
        groups: dict[int, int | None] = {}  # the process groups it kills: pgid -> its leader's start time
        killed: dict[int, int] = {}  # and the processes that had left them: pid -> start time
        for base_dir in sorted(self.root.glob("process-*")):
            try:
                record = json.loads((base_dir / RECORD_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or not cleanup.labels_match(record.get("labels"), labels):
                continue
            owner = record.get("owner_pid")
            if isinstance(owner, int) and owner != os.getpid() and _alive(owner, record.get("owner_start_time")):
                continue  # another live process holds it
            for pgid, start_time in _recorded_groups(base_dir / GROUPS_FILE):
                if signal_process_group(pgid, start_time, signal.SIGKILL):
                    groups[pgid] = start_time
            _kill_marked(_root_marker(base_dir / "fs"), killed=killed)  # and those that left their group
            shutil.rmtree(base_dir, ignore_errors=True)
            reaped += 1
        reaped += reap_stack_directories(self.root, labels)  # what the stacks of those sandboxes held: their volumes
        # What runs next, such as the run that reaps before it starts, needs what they held.
        if running := _wait_exited(groups, killed, _EXIT_WAIT_S):
            logger.warning(
                "Processes of the sandboxes labelled %s still run %gs after SIGKILL: %s",
                labels,
                _EXIT_WAIT_S,
                " ".join(map(str, running)),
            )
        if reaped:
            logger.info("Reaped %d process sandbox(es) labelled %s", reaped, labels)
        return reaped

    async def _open(self, spec: SandboxSpec, source: Path | None = None) -> ProcessSandbox:
        routes = SandboxRoutes(spec, own_network=False)
        labels = {**spec.labels, **cleanup.labels()}
        options = {"root": str(self.root)}
        token = cleanup.register(provider=cleanup.provider_ref(self), options=options, labels=labels)
        base_dir = self._new_dir()
        creating: asyncio.Future[ProcessSandbox] | None = None
        try:
            sandbox_spec = dataclasses.replace(spec, mounts=[*spec.mounts, *routes.mounts()], labels=labels)
            # The thread runs to its end whatever its caller does: a stopping process spares it.
            creating = let_finish(
                asyncio.ensure_future(asyncio.to_thread(ProcessSandbox.create, base_dir, sandbox_spec, source))
            )
            sandbox = await asyncio.shield(creating)
        except BaseException:
            cleanup.unregister(token)
            if creating is not None:  # a start given up while its directory is made: delete it once it is made
                let_finish(asyncio.ensure_future(_remove_when_made(creating, base_dir)))
            raise
        sandbox._routes, sandbox._token = routes, token
        try:
            await routes.start(sandbox)
        except BaseException:
            await asyncio.shield(let_finish(asyncio.ensure_future(sandbox.stop())))  # spared by a stopping process
            raise
        cleanup.attach(token, sandbox)
        return sandbox

    def _new_dir(self) -> Path:
        return self.root / f"process-{uuid.uuid4().hex[:12]}"


async def _remove_when_made(creating: asyncio.Future[ProcessSandbox], base_dir: Path) -> None:
    """Delete the directory of a sandbox whose start was given up, once the thread that makes it is done."""

    await asyncio.wait([creating])
    if not creating.cancelled():
        creating.exception()  # what it raised, its caller had
    await asyncio.to_thread(shutil.rmtree, base_dir, True)


def _alive(pid: int, start_time: object) -> bool:
    current = process_start_time(pid)
    return current is not None and (start_time is None or current == start_time)


def _recorded_groups(path: Path) -> list[tuple[int, int | None]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    groups = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            groups.append((int(parts[0]), int(parts[1]) if parts[1].isdigit() else None))
    return groups
