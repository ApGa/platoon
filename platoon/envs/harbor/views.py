"""A sandbox as one of its image's users sees it: the commands that a Harbor session runs as the users its task names
(``agent_user: honor``, ``verifier_user: honor``; :mod:`platoon.envs.harbor.users`).

A sandbox that honors users gives root to platoon's own commands, and runs each phase's commands as its user, as
Harbor does (harbor ``environments/base.py``, ``trial/trial.py``): the agents' (env tools, harness, oracle) as
``[agent].user``, else the image's ``USER``, else root; the tests as ``[verifier].user``, else the image's ``USER``,
else root.  A :class:`SandboxView` is the sandbox as one of them sees it, with Docker's umask, which a rootless
sandbox would otherwise take from the host (027 on many shared machines):

- **root** (:func:`root_view`): platoon's commands, and those of a phase that runs as root;
- **a user** (:func:`user_views`): each command runs as root and switches to the user before it starts, with the
  sandbox's own ``exec_as`` where it has one (Docker's ``exec --user``), else with a tool of the image that the view
  finds when it opens: ``setpriv`` (util-linux), else ``su``, else a Python's standard library (the image's, or a
  toolchain's).  The switch also sets ``HOME``, ``USER`` and ``LOGNAME`` from the image's ``/etc/passwd`` (``HOME``
  only where the sandbox's spec sets none), since a rootless sandbox gives every command root's ``HOME``.

A session's views run its commands in its working directory unless they name another (a relative one, and a relative
path to transfer, starts there): the sandbox may have started elsewhere, so that its start made no working directory
the image lacks, which the image's deferred ``WORKDIR`` gives its ``USER`` (:mod:`platoon.envs.harbor.images`).

A view gives what it uploads to its user: a user's uploads become that user's (``chown -h``, which never follows a
link), and root's keep the host's modes, with every user given the read and search access that the group has, as a
checkout under Docker's umask has them (a file the host keeps from its group, such as one of mode 600, stays so).  The
copy of a directory's contents gives its destination the directory's mode, so the destination is given too, while the
directories made above it are root's, with Docker's modes.  A root view changes what it uploaded by path once the
upload is done, so it uploads only while no code of another user runs: at a session's start, and while grading, once
the agents' processes are killed.  A user's view never downloads, which would read as root what the user may not.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platoon.envs.harbor.users import DOCKER_UMASK, ROOT, User, hook_users, parse_user, phase_users, read_accounts
from platoon.sandbox.base import DownloadReport, ExecResult, Sandbox, SandboxSnapshot

if TYPE_CHECKING:
    from platoon.envs.harbor.task import HarborTask

# Runs "$@" under Docker's umask: the shell a switch, or a root command, starts from.
_UMASK = ("/bin/sh", "-c", f'umask {DOCKER_UMASK} && exec "$@"')
# Prints the first tool that switches to uid $1 and gid $2 here: setpriv, else su (to the user $3, where su may),
# else one of the Python interpreters named after them.
_PROBE_SCRIPT = r"""
uid=$1; gid=$2; name=$3; shift 3
if setpriv --reuid="$uid" --regid="$gid" --clear-groups -- /bin/sh -c : >/dev/null 2>&1; then echo setpriv; exit 0; fi
if [ -n "$name" ] && su -m -s /bin/sh -c : "$name" >/dev/null 2>&1; then echo su; exit 0; fi
for python; do
  if "$python" -I -c 'import os' >/dev/null 2>&1; then echo "python $python"; exit 0; fi
done
"""
# Switches to uid argv[1], gid argv[2] and the groups argv[3] (comma-separated), then runs argv[4:].
_PYTHON_SWITCH = (
    "import os, sys; os.setgroups([int(g) for g in sys.argv[3].split(',') if g]); os.setgid(int(sys.argv[2])); "
    "os.setuid(int(sys.argv[1])); os.execvp(sys.argv[4], sys.argv[4:])"
)
_COMMAND_TIMEOUT_S = 120.0
_PATHS_PER_COMMAND = 500


@dataclass(frozen=True)
class _Switch:
    """How a view runs a command: ``prefix`` before its argv, ``user`` for the sandbox's ``exec_as``, and the
    environment it adds."""

    prefix: tuple[str, ...]
    user: str | None = None
    env: Mapping[str, str] | None = None


class SandboxView:
    """``sandbox`` as ``user`` sees it: its commands run as that user under Docker's umask, and what it uploads becomes
    that user's (root's, readable by every user).  With a ``workdir``, its commands run there unless they name another
    directory, and relative paths start there.  Everything but commands and transfers is the sandbox's own."""

    def __init__(self, sandbox: Sandbox, user: User, switch: _Switch, workdir: str | None = None) -> None:
        self.id = sandbox.id
        self.user = user
        self.sandbox = sandbox
        self.workdir = workdir
        self._switch = switch
        self._exec_as = getattr(sandbox, "exec_as", None) if switch.user is not None else None

    def _in_workdir(self, path: str | None) -> str | None:
        """``path`` as the sandbox reads it: below the view's workdir when relative, the workdir itself for None."""

        if self.workdir is None:
            return path
        return posixpath.join(self.workdir, path) if path else self.workdir

    @property
    def routes(self) -> Mapping[str, str]:
        return getattr(self.sandbox, "routes", None) or {}

    async def ensure_routes(self) -> Mapping[str, str]:
        ensure = getattr(self.sandbox, "ensure_routes", None)
        return await ensure() if ensure is not None else self.routes

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult:
        argv = [*self._switch.prefix, *(["bash", "-c", cmd] if isinstance(cmd, str) else cmd)]
        options: dict[str, Any] = {"timeout_s": timeout_s, "cwd": self._in_workdir(cwd), "stdin": stdin}
        options["env"] = {**(self._switch.env or {}), **(env or {})} or None
        if self._exec_as is not None:
            return await self._exec_as(self._switch.user, argv, **options)
        return await self.sandbox.exec(argv, **options)

    async def upload(self, src: str | Path, dst: str) -> None:
        source = Path(src)
        dst = self._in_workdir(dst) or dst
        directory = dst if source.is_dir() else posixpath.dirname(dst) or "/"
        result = await root_view(self.sandbox).exec(
            ["mkdir", "-p", "--", directory], cwd="/", timeout_s=_COMMAND_TIMEOUT_S
        )
        if result.exit_code != 0 or result.timed_out:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"Cannot make {directory} in sandbox {self.id}: {detail}")
        await self.sandbox.upload(source, dst)
        uploaded = uploaded_paths(source, dst)
        if self.user.uid == 0:
            await _open_to_every_user(self.sandbox, uploaded)
        else:
            await give(self.sandbox, f"{self.user.uid}:{self.user.gid}", [path for path, _ in uploaded])

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        if self.user.uid != 0:
            raise PermissionError(
                f"A view of sandbox {self.id} as uid {self.user.uid} does not download: the sandbox would read {src} "
                "as root; read the user's files through exec"
            )
        return await self.sandbox.download(self._in_workdir(src) or src, dst)

    @property
    def keep_processes(self) -> Callable[[Iterable[str]], None] | None:
        """The sandbox's own ``keep_processes``, or None where it has none (callers look it up with ``getattr``)."""

        return getattr(self.sandbox, "keep_processes", None)

    async def kill_processes(self) -> None:
        await self.sandbox.kill_processes()

    async def snapshot(self) -> SandboxSnapshot:
        return await self.sandbox.snapshot()

    async def stop(self) -> None:
        await self.sandbox.stop()


@dataclass(frozen=True)
class SessionUsers:
    """Who runs what in one of a session's sandboxes: platoon's own commands as ``root``, the agents' as ``agents``,
    the tests as ``tests``, the image's own commands (``main``'s init and healthchecks, collect hooks that name no
    user) as ``default``, the image's ``USER`` or root, and each collect hook that names a user as that user
    (``hooks``).  In a sandbox that honors no user, each of them is the sandbox itself (:meth:`alone`)."""

    root: Sandbox
    agents: Sandbox
    tests: Sandbox
    default: Sandbox
    hooks: Mapping[str, Sandbox] = field(default_factory=dict)

    @classmethod
    def alone(cls, sandbox: Sandbox) -> SessionUsers:
        return cls(root=sandbox, agents=sandbox, tests=sandbox, default=sandbox)

    @property
    def honors(self) -> bool:
        """Whether the sandbox honors users: its root runs as a view (:func:`root_view`)."""

        return isinstance(self.root, SandboxView)

    @property
    def tests_as_user(self) -> bool:
        """Whether the tests run as a user other than root, who must write ``/logs/verifier`` and read ``/tests``."""

        return isinstance(self.tests, SandboxView) and self.tests.user.uid != 0


def unviewed(sandbox: Sandbox) -> Sandbox:
    """The sandbox that ``sandbox`` is a view of, or ``sandbox`` itself."""

    return sandbox.sandbox if isinstance(sandbox, SandboxView) else sandbox


def root_view(sandbox: Sandbox, *, workdir: str | None = None) -> SandboxView:
    """``sandbox``'s commands as root of its user namespace, under Docker's umask: as root already where the sandbox
    gives root (Singularity's ``--fakeroot``), through ``exec_as("0")`` where it runs commands as users itself
    (Docker).  ``workdir`` is where its commands run unless they name a directory (else the sandbox's own)."""

    if isinstance(sandbox, SandboxView):
        sandbox = sandbox.sandbox
    user = "0" if callable(getattr(sandbox, "exec_as", None)) else None
    return SandboxView(sandbox, ROOT, _Switch(prefix=(*_UMASK, "platoon-root"), user=user), workdir)


async def user_views(
    sandbox: Sandbox,
    specs: Iterable[str | None],
    *,
    spec_env: Mapping[str, str] | None = None,
    pythons: Sequence[str] = (),
    workdir: str | None = None,
) -> dict[str | None, SandboxView]:
    """Views of ``sandbox`` as each user ``specs`` names, by spec, its accounts read once: one of Docker's ``--user``
    forms (:func:`~platoon.envs.harbor.users.parse_user`), or None for the sandbox's own user, the container's
    ``USER`` where the sandbox runs commands as users itself, else root.  ``spec_env`` is the sandbox's own
    environment (a ``HOME`` there stays), ``pythons`` are interpreters besides the image's ``python3`` that may
    switch users, such as a mounted toolchain's, and ``workdir`` is where their commands run unless they name a
    directory.  ``ValueError`` says why no command could run as one: an image without the user, or without a way to
    switch to it."""

    root = root_view(sandbox, workdir=workdir)
    native = callable(getattr(sandbox, "exec_as", None))
    accounts: tuple[str, str] | None = None
    views: dict[str | None, SandboxView] = {}
    for spec in dict.fromkeys(specs):
        if spec is None and not native:
            views[spec] = root
            continue
        accounts = accounts or await read_accounts(root)
        # The container's own user is read for what it uploads, while its commands run as Docker's plain exec runs
        # them, with the user's supplementary groups.
        user = parse_user(spec if spec is not None else await _own_user(sandbox), *accounts)
        if user.uid == 0:
            views[spec] = root
            continue
        env = {"HOME": user.home} if "HOME" not in (spec_env or {}) else {}
        if user.name is not None:
            env |= {"USER": user.name, "LOGNAME": user.name}
        if native:
            switch = _Switch(prefix=(*_UMASK, "platoon-user"), user=spec, env=env)
        else:
            switch = _Switch(prefix=await _switch(root, user, pythons), env=env)
        views[spec] = SandboxView(sandbox, user, switch, workdir)
    return views


async def session_users(
    sandbox: Sandbox,
    harbor_task: HarborTask,
    *,
    image_user: str | None,
    verifier: bool = False,
    spec_env: Mapping[str, str] | None = None,
    pythons: Sequence[str] = (),
    workdir: str | None = None,
) -> SessionUsers:
    """Who runs what in ``sandbox``, a sandbox of ``harbor_task`` that honors users (its separate verifier's with
    ``verifier``), whose image's ``USER`` is ``image_user``, as Harbor's rule says
    (:func:`~platoon.envs.harbor.users.phase_users`; the sandbox's own user is root, or the container's ``USER`` where
    the sandbox runs commands as users itself), each running its commands in ``workdir``, the session's working
    directory, unless they name another.  ``ValueError`` when one of them cannot run (:func:`user_views`)."""

    users = phase_users(harbor_task, image_user=image_user, verifier=verifier)
    hooks = hook_users(harbor_task, verifier=verifier)
    views = await user_views(sandbox, [*users.values(), *hooks], spec_env=spec_env, pythons=pythons, workdir=workdir)
    return SessionUsers(
        root=root_view(sandbox, workdir=workdir),
        agents=views[users["agents"]],
        tests=views[users["tests"]],
        default=views[users["default"]],
        hooks={user: views[user] for user in hooks},
    )


async def _own_user(sandbox: Sandbox) -> str:
    """The uid and gid that ``sandbox``'s commands run as when they name no user (``uid:gid``)."""

    result = await sandbox.exec(
        ["/bin/sh", "-c", 'echo "$(id -u):$(id -g)"', "platoon-own-user"], cwd="/", timeout_s=_COMMAND_TIMEOUT_S
    )
    if result.exit_code != 0 or result.timed_out:
        raise ValueError(f"cannot tell which user the image runs as: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


async def _switch(root: SandboxView, user: User, pythons: Sequence[str]) -> tuple[str, ...]:
    """The argv prefix that runs a command as ``user`` in the sandbox of ``root``, with the first tool that switches to
    it there (:data:`_PROBE_SCRIPT`)."""

    login = user.login or ""
    probe = await root.exec(
        ["/bin/sh", "-c", _PROBE_SCRIPT, "platoon-switch", str(user.uid), str(user.gid), login, "python3", *pythons],
        cwd="/",
        timeout_s=_COMMAND_TIMEOUT_S,
    )
    tool, _, python = probe.stdout.strip().partition(" ")
    umask = (*_UMASK, "platoon-user")
    if tool == "setpriv":
        groups = "--init-groups" if user.login is not None else "--clear-groups"
        return (*umask, "setpriv", f"--reuid={user.uid}", f"--regid={user.gid}", groups, "--")
    if tool == "su":
        return (*umask, "su", "-m", "-s", "/bin/sh", "-c", 'exec "$@"', login, "--", "platoon-user")
    if tool == "python" and python:
        groups = ",".join(str(group) for group in user.groups)
        return (*umask, python, "-I", "-c", _PYTHON_SWITCH, str(user.uid), str(user.gid), groups)
    raise ValueError(
        f"the image has no way to switch to the user {user.spec} (uid {user.uid}): no setpriv that takes --reuid "
        f"(util-linux's), {'no su' if login else 'no su for a user that /etc/passwd lists with its own group'}, and "
        "no Python 3"
    )


def uploaded_paths(source: Path, dst: str) -> list[tuple[str, Path]]:
    """The sandbox paths that an upload of ``source`` to ``dst`` writes, each with the host's path it copies: ``dst``
    for a file; for a directory, ``dst``, whose mode and owner the copy of the directory's contents gives it too
    (``cp -p``), and every entry below it.  Read from the host's copy, never from the sandbox."""

    paths = [(dst, source)]
    if not source.is_dir():
        return paths
    for directory, dirnames, filenames in os.walk(source):  # never into a linked directory
        base = posixpath.join(dst, Path(directory).relative_to(source).as_posix())
        for name in sorted([*dirnames, *filenames]):
            paths.append((posixpath.normpath(posixpath.join(base, name)), Path(directory, name)))
    return paths


async def _open_to_every_user(sandbox: Sandbox, uploaded: Sequence[tuple[str, Path]]) -> None:
    """Give every user of ``sandbox`` the read and search access to what root ``uploaded`` that the host gives its
    group, as a checkout under Docker's umask gives it, where the host's (often 027) took it away; set every path's mode
    so, since the copy of a directory may have given its destination another.  Links are left alone."""

    paths_by_mode: dict[int, list[str]] = {}
    for path, host in uploaded:
        if host.is_symlink():
            continue
        mode = stat.S_IMODE(host.stat().st_mode)
        paths_by_mode.setdefault(mode | (mode & 0o050) >> 3, []).append(path)
    for mode, paths in paths_by_mode.items():
        await _each(sandbox, ["chmod", format(mode, "o"), "--"], paths)


async def give(sandbox: Sandbox, owner: str, paths: Sequence[str]) -> None:
    """Give ``paths`` of ``sandbox`` to ``owner`` (a user, optionally ``:group``, as ``chown`` takes one), as its root;
    a link gets the owner itself, never what it points to."""

    await _each(sandbox, ["chown", "-h", owner, "--"], list(paths))


async def _each(sandbox: Sandbox, command: list[str], paths: list[str]) -> None:
    """Run ``command`` as root in ``sandbox`` on ``paths``, a few hundred at a time; RuntimeError when it fails."""

    for start in range(0, len(paths), _PATHS_PER_COMMAND):
        chunk = paths[start : start + _PATHS_PER_COMMAND]
        result = await root_view(sandbox).exec([*command, *chunk], cwd="/", timeout_s=_COMMAND_TIMEOUT_S)
        if result.exit_code != 0 or result.timed_out:
            shown = shlex.join(command[:-1])
            raise RuntimeError(f"Could not {shown} what was uploaded: {(result.stderr or result.stdout).strip()}")
