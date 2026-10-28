"""A Harbor task's compose services when a session opens: their binds, and the stack of a task with sidecars, with
its probes, its collect hooks and what its sidecars write in the session's log directories (RFC 0010, section 6).

**Binds** of the task's own files (:class:`~platoon.envs.harbor.compose.Bind`, any service's): a read-only one binds
the cached task (:func:`bind_mounts`); a masked one binds a filtered copy, in which what its volumes mask is an empty
directory, a writable one binds a copy, so that the content-addressed cache is never written, and a missing source an
empty directory, as Docker creates one; a host path that a listed host variable names binds read-only, resolved now.
A single sandbox's copies are the session's, in node scratch (:func:`bind_copies`); a stack's are its own volumes
(:func:`copied_binds`), one for each path and masks that every service binding it shares, as compose shares the one
host directory, so a sidecar sees what ``main`` writes there and reaping a stack deletes its copies too.  A stack's
copies have the modes a Docker host's checkout has (every user enters their directories and reads their files), and a
sidecar's read-only bind of files that not every user may read (a task fetched under a umask such as 027) binds such a
copy too, since a sidecar may read them as a user of its own, as the official database images do.

**Stacks.**  A task whose compose file names services besides ``main`` (``flags.stack: sidecars``) runs as a stack on a
sandbox provider that runs stacks (:mod:`platoon.sandbox.stacks`), and :func:`stack_spec` turns its plan into the
stack's spec, making nothing yet (the stack makes its volumes when it starts), so it also says, before any rollout,
why a stack cannot start:

- ``main`` is the sandbox a single-container session of the task would get (its image, what the compose file gives
  it, the harness's needs), idle; the session's preparation of it (its directories, its image's deferred files, its
  init and healthchecks) is the service's ``setup``, which runs once its dependencies are up;
- each sidecar runs its image, as compose runs it: a registry reference as the image map pinned it (by digest, with
  its configuration, which ``platoon prepare`` records), else as written, the image map's image of a build context's
  whole build key, or ``main``'s image for ``${MAIN_IMAGE_NAME}``; compose's ``command`` and ``entrypoint`` over the
  image's ``ENTRYPOINT`` and ``CMD``; its environment over the image's and the Harbor service's ``env``; its
  healthcheck, else its image's; the user its image runs as (``USER``); its dependencies, volumes, ports, names and
  resources.  An unpinned registry image's configuration is unknown (a dataset directory has no image map), so such a
  service runs compose's command alone, and one that gives none is refused; an init job's run is bounded
  (``job_timeout_s``);
- a sidecar's bind of a session log directory (``${HOST_VERIFIER_LOGS_PATH}``, the agent's or the artifacts') is a
  volume of the stack, one per log directory for the sidecars that bind it: grading copies its regular files, never a
  link, into that directory of the sandbox the tests run in, once it made the directory afresh (:func:`snapshot_logs`,
  :func:`put_logs`), so that the agents can neither erase nor forge what a sidecar records there; through a sidecar
  that runs with your subordinate ids, as its root, since its own users may keep their files from yours
  (:func:`log_source`), and a file it cannot read fails the grade as infrastructure, never left out.  A sidecar does
  not see what ``main`` writes in its own, so a read-only bind of one is refused when the task loads.

**Node services.**  A service that a dataset's overlay marks ``external`` is served by a node service
(:mod:`platoon.envs.harbor.node_services`), and not started: the stack reaches it outside it (``StackSpec.externals``),
through the rollout's scope of the node service, at the node service's port, by the service's name, and the other
services' dependencies on it are met before the stack starts (the node service is up, or no scope opens).

**On the host network** (``host_network``, the fallback of :mod:`platoon.sandbox.host_network_stacks`), each service
also gets what the overlay's rules for it name (:class:`~platoon.envs.harbor.compose.HostNetworkRule`): the stack's
token, a secret of the stack's own, in every token variable that a rule names, in every service of the stack (so that
each service, and the tests in ``main``, reach the others), ``127.0.0.1`` in its ``bind`` variable, and its ``listen``
variables, which the stack sets to the ports it holds.

**Probes** (:func:`probe`): before and after the tests (``shared``), or before and after the artifacts are collected
(``separate``), every sidecar must be up: an init job completed, any other running, and healthy where it has a
healthcheck.  One that is down makes the outcome ``invalid`` (``sidecar_unhealthy``: its name, its state and the tail
of its log), whatever the reward files say, unless the environment sets ``on_sidecar_down: grade``, which grades
anyway and records it.  A sidecar that fails to start, or is down as the stack's start ends, fails the session's start
(``sidecar_start``, an ``error``).  A rollout's deadline counts the waits along the services' dependencies
(:func:`start_bound_s`) and each probe's longest (:func:`probe_bound_s`), from the healthchecks the stack runs.

**Collect hooks** (``[[verifier.collect]]``, :func:`run_collect_hooks`): commands run in a service once the agents are
done, best effort, as Harbor runs them: ``main``'s first, then the sidecars'.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import posixpath
import secrets
import shutil
import stat
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from platoon.envs.harbor.compose import (
    LOG_DIRS,
    MAIN,
    Bind,
    HostNetworkRule,
    ServiceHealthcheck,
    declared_ports,
    image_healthcheck,
    resolve_argv,
    resolve_deferred,
    service_aliases,
    service_binds,
)
from platoon.envs.harbor.images import SessionImage, docker_image_reference
from platoon.envs.harbor.task import CollectHook, HarborTask
from platoon.envs.service import Outcome
from platoon.sandbox.base import Mount, Sandbox, SandboxSpec
from platoon.sandbox.stacks import (
    LOG_TAIL_BYTES,
    LOOPBACK,
    Dependency,
    ExternalService,
    Healthcheck,
    ServiceSpec,
    ServiceState,
    Stack,
    StackNetwork,
    StackSpec,
    Volume,
    longest_path,
)
from platoon.utils import scratch
from platoon.utils.stopping import let_finish

logger = logging.getLogger(__name__)
T = TypeVar("T")

#: The stack's own volumes, named so that no compose file's volume can be called the same (compose's names start with
#: a letter or a digit, and hold no colon).
COPY_VOLUME = "platoon:copy-{number}"
LOGS_VOLUME = "platoon:logs-{logs}"
_COPY_TREE = "tree"  # where a copy volume holds the copy: what the bind mounts (a file or a directory)
_COPY_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
CopyKey = tuple[str, tuple[str, ...]]  # a bind's task path and masks: the binds that share one copy


# --- binds ---------------------------------------------------------------------------------------------------------


def copy_key(bind: Bind) -> CopyKey:
    """What identifies the copy a bind of the task's files shares with others: its path in the task, and its masks."""

    assert bind.task is not None
    return posixpath.normpath(bind.task), tuple(sorted(posixpath.normpath(mask) for mask in bind.masks))


def needs_copy(bind: Bind) -> bool:
    """Whether a bind of the task's files cannot bind the cache itself: a writable, masked or missing source's."""

    return bind.task is not None and (not bind.read_only or bool(bind.masks) or bind.missing)


def bind_mounts(harbor_task: HarborTask, binds: Sequence[Bind], *, copied: Collection[CopyKey] = ()) -> list[Mount]:
    """The read-only ``binds`` that mount what they name: the task's own files (but masked ones, missing sources and
    the ``copied`` ones of a stack, which bind copies: :func:`bind_copies`, :func:`copied_binds`), and host paths that
    listed host variables name, resolved now (``KeyError`` names an unset one).  Log directories are the caller's."""

    mounts = []
    for bind in binds:
        if not bind.read_only or bind.missing or bind.masks or bind.logs is not None:
            continue
        if bind.task is not None:
            if copy_key(bind) not in copied:
                mounts.append(Mount(source=str(harbor_task.task_dir / bind.task), target=bind.target, read_only=True))
        elif bind.host is not None:
            source = resolve_deferred(bind.host, harbor_task.env, os.environ)
            if not os.path.isabs(source) or not os.path.exists(source):
                raise ValueError(
                    f"Harbor task {harbor_task.name!r} binds the host path {source!r} at {bind.target}, which is not "
                    "an existing absolute path on this host"
                )
            mounts.append(Mount(source=source, target=bind.target, read_only=True))
    return mounts


def bind_copies(harbor_task: HarborTask, binds: Sequence[Bind]) -> tuple[list[Mount], list[Path]]:
    """The mounts of a single sandbox's ``binds`` of the task's files that need a copy of their own, and those copies,
    in scratch, for the caller to delete with the sandbox: a masked read-only bind binds a filtered copy (hard links
    where the filesystem allows), in which the masked paths are empty directories, so that the masked files never
    enter the sandbox; a writable bind binds a copy (never links: the content-addressed cache is never written); a
    missing source binds an empty directory, as Docker creates one."""

    copies: list[Path] = []
    mounts: list[Mount] = []
    try:
        for bind in binds:
            if not needs_copy(bind):
                continue
            directory = Path(scratch.mkdtemp("platoon-harbor-bind-"))
            copies.append(directory)
            _fill_copy(harbor_task, bind.task or ".", bind.masks, bind.missing, link=bind.read_only)(directory)
            mounts.append(Mount(source=str(directory / _COPY_TREE), target=bind.target, read_only=bind.read_only))
    except BaseException:
        for directory in copies:
            shutil.rmtree(directory, ignore_errors=True)
        raise
    return mounts, copies


def copied_binds(harbor_task: HarborTask) -> dict[CopyKey, bool]:
    """The binds of a stack's task files that bind a copy the stack owns (a volume of its own), by path and masks, one
    copy for every service that binds that path with those masks: those a service binds writable, masked or from a
    missing source, the read-only binds of what another service writes, and what a sidecar binds that not every user
    may read, whose copy every user reads (:func:`_fill_copy`).  Each says whether the copy may hard-link the task's
    files: every bind of it is read-only, and every user may read them or no sidecar binds them.  Empty for a task
    that is not a stack."""

    plan = harbor_task.compose
    if plan is None or plan.kind != "sidecars":
        return {}
    shared: dict[CopyKey, list[Bind]] = {}
    sidecars: set[CopyKey] = set()  # what a sidecar binds
    for name, data in plan.services.items():
        for bind in service_binds(data):
            if bind.task is not None:
                shared.setdefault(copy_key(bind), []).append(bind)
                if name != MAIN:
                    sidecars.add(copy_key(bind))
    copies = {}
    for key, binds in sorted(shared.items()):
        readable = key not in sidecars or _readable_by_all(harbor_task.task_dir / key[0])  # by whoever binds it
        if any(needs_copy(bind) for bind in binds) or not readable:
            copies[key] = all(bind.read_only for bind in binds) and readable
    return copies


def _readable_by_all(path: Path) -> bool:
    """Whether every user may read ``path``, as a sidecar that reads it as a user of its own needs: a file others may
    read, or a directory others may enter and list, with everything below it so (links are neither followed nor
    counted: a copy keeps them as links).  A path that does not exist has nothing to read."""

    def others_read(entry: str, *, directory: bool) -> bool:
        mode = os.lstat(entry).st_mode
        wanted = 0o005 if directory else 0o004
        return stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)) or mode & wanted == wanted

    if not os.path.lexists(path):
        return True
    if not path.is_dir() or path.is_symlink():
        return others_read(str(path), directory=False)
    for directory, _names, files in os.walk(path):  # links to directories are among the names, never walked
        if not others_read(directory, directory=True):
            return False
        if not all(others_read(os.path.join(directory, name), directory=False) for name in files):
            return False
    return True


def _fill_copy(
    harbor_task: HarborTask, path: str, masks: Sequence[str], missing: bool, *, link: bool, readable: bool = False
) -> Callable[[Path], None]:
    """What fills a copy's directory: the task's ``path`` at its ``tree`` (filtered by ``masks``, hard-linked when
    ``link``), or an empty directory for a missing one.  With ``readable`` (a stack's), every user may then enter the
    copy's directories and read its files (``chmod a+rX``), whatever this process's umask, as on a Docker host, whose
    checkouts have those modes; a hard link keeps the cache's modes, which every user reads already."""

    def fill(directory: Path) -> None:
        tree = directory / _COPY_TREE
        if missing:
            tree.mkdir()
        else:
            filtered_copy(harbor_task.task_dir / path, tree, masks, link=link)
        if readable:
            _readable_modes(tree)

    return fill


def _readable_modes(tree: Path) -> None:
    """``chmod a+rX`` on ``tree`` and everything below it but links and the hard links that :func:`filtered_copy`
    made (they share the cache's inodes, never changed)."""

    def add(entry: str) -> None:
        status = os.lstat(entry)
        if stat.S_ISDIR(status.st_mode) or (stat.S_ISREG(status.st_mode) and status.st_nlink == 1):
            mode = stat.S_IMODE(status.st_mode)
            executable = stat.S_ISDIR(status.st_mode) or mode & 0o111
            os.chmod(entry, mode | 0o444 | (0o111 if executable else 0))

    add(str(tree))
    if tree.is_dir() and not tree.is_symlink():
        for directory, names, files in os.walk(tree):
            for name in [*names, *files]:
                add(os.path.join(directory, name))


def filtered_copy(source: Path, target: Path, masks: Sequence[str], *, link: bool) -> None:
    """Copy ``source`` (a file or a directory) to ``target``, with hard links where possible when ``link``, leaving
    each of ``masks`` (paths relative to ``source``) an empty directory.  A symlink is copied as a link, as a bind
    shows it."""

    hidden = {posixpath.normpath(mask) for mask in masks}

    def copy(path: str, destination: str) -> None:
        if os.path.islink(path):
            os.symlink(os.readlink(path), destination)
            return
        if link:
            try:
                os.link(path, destination)
                return
            except OSError:
                pass
        shutil.copy2(path, destination)

    if not source.is_dir() or source.is_symlink():
        target.parent.mkdir(parents=True, exist_ok=True)
        copy(str(source), str(target))
        return
    for directory, names, files in os.walk(source):  # symlinks to directories are among the names, never walked
        relative = os.path.relpath(directory, source)
        (target / relative).mkdir(parents=True, exist_ok=True)
        for name in list(names):
            path = posixpath.normpath(posixpath.join(relative, name))
            if path in hidden:
                names.remove(name)
                (target / path).mkdir()
            elif os.path.islink(os.path.join(directory, name)):
                names.remove(name)
                copy(os.path.join(directory, name), str(target / path))
        for name in files:
            path = posixpath.normpath(posixpath.join(relative, name))
            if path not in hidden:
                copy(os.path.join(directory, name), str(target / path))
    for mask in hidden:  # a mask the task has nothing at is an empty directory too, as Docker's volume makes one
        (target / mask).mkdir(parents=True, exist_ok=True)


async def in_thread(work: Callable[[], T], *, undo: Callable[[T | None], None]) -> T:
    """``work()`` in a thread.  A caller cancelled while it runs gets the cancellation at once, but the thread runs on
    to its end (a stopping process spares it), and ``undo`` then gets what it made (None when it failed), in a thread,
    as a cleanup of the rollout's (:func:`~platoon.utils.stopping.let_finish`): what a cancelled caller leaves in node
    scratch, which no one else would delete."""

    running = let_finish(asyncio.ensure_future(asyncio.to_thread(work)))
    try:
        return await asyncio.shield(running)
    except asyncio.CancelledError:
        let_finish(asyncio.ensure_future(_undo_when_done(running, undo)))
        raise


async def _undo_when_done(running: asyncio.Future[T], undo: Callable[[T | None], None]) -> None:
    await asyncio.wait([running])
    made = None if running.cancelled() or running.exception() is not None else running.result()
    await asyncio.to_thread(undo, made)


# --- the stack's spec ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SidecarLogs:
    """One of the session's log directories (``verifier``, ``agent`` or ``artifacts``) that sidecars bind: a volume of
    the stack, bound in each of ``services`` at the path there (its target) where its compose file binds the
    directory."""

    logs: str
    volume: str
    services: Mapping[str, str]  # each sidecar that binds it, with its target


@dataclass
class StackPlan:
    """What :func:`stack_spec` made: the spec, and the sidecars' log directories."""

    spec: StackSpec
    logs: list[SidecarLogs]


def stack_spec(
    harbor_task: HarborTask,
    main: SandboxSpec,
    *,
    main_image: SessionImage | None,
    service_image: Callable[[str, Mapping[str, Any]], SessionImage],
    env: Mapping[str, str],
    mounts: Sequence[Mount],
    network: StackNetwork,
    labels: Mapping[str, str],
    setup: Callable[[Sandbox], Awaitable[None]],
    job_timeout_s: float | None = None,
    pulled_image: Callable[[str], SessionImage | None] = lambda reference: None,
    on_host_network: bool = False,
    externals: Mapping[str, ExternalService] | None = None,
) -> StackPlan:
    """The spec of the task's stack (the module docstring says what each service gets): ``main`` is the agents'
    sandbox, prepared by ``setup``; ``main_image`` is its image as the session runs it (None: its ``docker_image``,
    whose configuration platoon does not know), ``service_image(name, build)`` the image of a service that builds from
    a context, ``pulled_image(reference)`` the image map's pin of a registry image (None: none), ``env`` and ``mounts``
    the Harbor service's (every sidecar gets its read-only mounts), ``job_timeout_s`` an init job's longest run,
    ``on_host_network`` whether it runs on the host-network fallback, with what the overlays' rules say, and
    ``externals`` how the stack reaches each service that a node service serves (its scope's socket and the node
    service's port), by the service's name.  It makes nothing.  ``ValueError`` says why a service cannot run (an image
    without a known command or not in the image map, an unset host variable, a node service the stack has no way to,
    a spec no stack can start)."""

    plan = harbor_task.compose
    assert plan is not None and plan.kind == "sidecars"
    served = plan.external
    if missing := sorted(set(served) - set(externals or {})):
        raise ValueError(
            f"Harbor task {harbor_task.name!r}: the service {missing[0]} is served by the node service "
            f"{served[missing[0]]}, which the stack has no way to"
        )
    copies = {
        key: Volume(
            target="",
            name=COPY_VOLUME.format(number=number),
            subpath=_COPY_TREE,
            fill=_fill_copy(harbor_task, key[0], key[1], _missing(harbor_task, key[0]), link=link, readable=True),
        )
        for number, (key, link) in enumerate(copied_binds(harbor_task).items())
    }
    logs: dict[str, dict[str, str]] = {}  # each log directory's sidecars, with their targets
    services: dict[str, ServiceSpec] = {}
    for name, data in plan.services.items():
        if name in served:
            continue  # served by a node service: not started
        binds = service_binds(data)
        own = [
            replace(copies[copy_key(bind)], target=bind.target, read_only=bind.read_only)
            for bind in binds
            if bind.task is not None and copy_key(bind) in copies
        ]
        if name == MAIN:
            check = _healthcheck(data, main_image.healthcheck if main_image is not None else None)
            services[MAIN] = _service(
                data, sandbox=main, command=None, healthcheck=check, setup=setup, volumes=own, served=served
            )
            continue
        for bind in binds:
            if bind.logs is not None:
                logs.setdefault(bind.logs, {}).setdefault(name, bind.target)
                own.append(Volume(target=bind.target, name=LOGS_VOLUME.format(logs=bind.logs)))
        services[name] = _sidecar(
            harbor_task,
            name,
            data,
            image=_image(harbor_task, name, data, main, main_image, service_image, pulled_image),
            env=env,
            mounts=[*(mount for mount in mounts if mount.read_only), *bind_mounts(harbor_task, binds, copied=copies)],
            volumes=own,
            network=network,
            labels=labels,
            timeout_s=job_timeout_s,
            served=served,
        )
    if on_host_network:
        services = _on_host_network(services, plan.host_network_rules)
    reached = {name: external for name, external in (externals or {}).items() if name in served}
    spec = StackSpec(services=services, network=network, labels=dict(labels), externals=reached)
    spec.check()
    directories = [
        SidecarLogs(logs=kind, volume=LOGS_VOLUME.format(logs=kind), services=targets) for kind, targets in logs.items()
    ]
    return StackPlan(spec, directories)


def _on_host_network(
    services: Mapping[str, ServiceSpec], rules: Mapping[str, HostNetworkRule]
) -> dict[str, ServiceSpec]:
    """``services`` with what the overlays' ``rules`` for them give them on the host-network fallback: a new token of
    the stack's in every token variable a rule names, in every service; ``127.0.0.1`` in a service's ``bind`` variable;
    and its ``listen`` variables, which the stack sets."""

    token = secrets.token_urlsafe(32)
    tokens = dict.fromkeys(sorted({rule.token for rule in rules.values() if rule.token}), token)
    on_host: dict[str, ServiceSpec] = {}
    for name, service in services.items():
        rule = rules.get(name) or HostNetworkRule()
        env = {**service.sandbox.env, **tokens, **({rule.bind: LOOPBACK} if rule.bind else {})}
        on_host[name] = replace(service, sandbox=replace(service.sandbox, env=env), listen=dict(rule.listen))
    return on_host


def _sidecar(
    harbor_task: HarborTask,
    name: str,
    data: Mapping[str, Any],
    *,
    image: SessionImage,
    env: Mapping[str, str],
    mounts: Sequence[Mount],
    volumes: Sequence[Volume],
    network: StackNetwork,
    labels: Mapping[str, str],
    timeout_s: float | None,
    served: Collection[str] = (),
) -> ServiceSpec:
    """A sidecar's service, as compose runs it (the module docstring says what it gets); ``served`` are the services
    that node services serve, whose dependencies are met before the stack starts."""

    environment = {
        key: resolve_deferred(value, harbor_task.env, os.environ)
        for key, value in (data.get("environment") or {}).items()
    }
    resources = data.get("resources") or {}
    sandbox = SandboxSpec(
        image=image.image or "",
        workdir=data.get("working_dir") or image.workdir or "/",
        env={**image.env, **env, **environment},
        mounts=list(mounts),
        network="none" if network == "none" else "host",
        cpus=resources.get("cpus"),
        memory_mb=resources.get("memory_mb"),
        labels=dict(labels),
    )
    return _service(
        data,
        sandbox=sandbox,
        command=_argv(harbor_task, name, data, image),
        healthcheck=_healthcheck(data, image.healthcheck),
        volumes=volumes,
        timeout_s=timeout_s,
        user=image.user,
        served=served,
    )


def _service(
    data: Mapping[str, Any],
    *,
    sandbox: SandboxSpec,
    command: list[str] | None,
    healthcheck: Healthcheck | None,
    setup: Callable[[Sandbox], Awaitable[None]] | None = None,
    volumes: Sequence[Volume] = (),
    timeout_s: float | None = None,
    user: str | None = None,
    served: Collection[str] = (),
) -> ServiceSpec:
    network_mode = str(data.get("network_mode") or "")
    return ServiceSpec(
        sandbox=sandbox,
        command=command,
        healthcheck=healthcheck,
        depends_on={
            other: Dependency(condition=settings["condition"], required=settings.get("required", True))
            for other, settings in (data.get("depends_on") or {}).items()
            if other not in served  # a node service is up before the stack starts
        },
        volumes=[
            *(
                Volume(target=mount["target"], name=mount.get("name"), tmpfs=mount["type"] == "tmpfs")
                for mount in data.get("volumes") or ()
                if mount["type"] in ("volume", "tmpfs") and not mount.get("masks")  # one that masks a bind: its copy's
            ),
            *volumes,
        ],
        ports=declared_ports(data),
        aliases=service_aliases(data),
        network_of=network_mode.removeprefix("service:") if network_mode.startswith("service:") else None,
        shm_mb=data.get("shm_mb"),
        ulimits={name: (int(limit[0]), int(limit[1])) for name, limit in (data.get("ulimits") or {}).items()},
        cap_add=tuple(data.get("cap_add") or ()),
        privileged=bool(data.get("privileged")),
        stop_signal=data.get("stop_signal"),
        setup=setup,
        timeout_s=timeout_s,
        user=user,
    )


def _missing(harbor_task: HarborTask, path: str) -> bool:
    """Whether the stack plan says the task has nothing at ``path`` (a bind of a missing source)."""

    plan = harbor_task.compose
    assert plan is not None
    return any(
        bind.missing
        for data in plan.services.values()
        for bind in service_binds(data)
        if bind.task is not None and posixpath.normpath(bind.task) == path
    )


def _image(
    harbor_task: HarborTask,
    name: str,
    data: Mapping[str, Any],
    main: SandboxSpec,
    main_image: SessionImage | None,
    service_image: Callable[[str, Mapping[str, Any]], SessionImage],
    pulled_image: Callable[[str], SessionImage | None],
) -> SessionImage:
    """A sidecar's image: a registry reference (as the image map pinned it, else as written, its configuration
    unknown), ``main``'s, or a build context's."""

    image = data.get("image")
    if isinstance(image, Mapping) and image.get("main"):
        if main_image is not None:
            return replace(main_image, files=())  # a service's command starts with its sandbox: nothing is deferred
        return SessionImage(image=main.image, env={}, workdir=None)
    if isinstance(image, str):
        return pulled_image(image) or SessionImage(image=docker_image_reference(image), env={}, workdir=None)
    build = data.get("build")
    if not isinstance(build, Mapping):
        raise ValueError(f"Harbor task {harbor_task.name!r}: the service {name} has neither an image nor a build")
    return service_image(name, build)


def _argv(harbor_task: HarborTask, name: str, data: Mapping[str, Any], image: SessionImage) -> list[str]:
    """A sidecar's command as compose runs it (:func:`~platoon.envs.harbor.compose.resolve_argv`)."""

    entrypoint, command = data.get("entrypoint"), data.get("command")
    if entrypoint is None and command is None and image.entrypoint is None and image.cmd is None:
        raise ValueError(
            f"Harbor task {harbor_task.name!r}: the service {name} runs the command of its image {image.image}, "
            "which platoon does not know (no image map pinned the image with its configuration, as platoon prepare "
            "pins a fetched dataset's on every provider but process): give the service a command (an overlay's "
            "[[compose]] replace)"
        )
    argv = resolve_argv(image_entrypoint=image.entrypoint, image_cmd=image.cmd, entrypoint=entrypoint, command=command)
    if not argv:
        raise ValueError(f"Harbor task {harbor_task.name!r}: the service {name} has an empty command")
    return argv


def _healthcheck(data: Mapping[str, Any], image: Mapping[str, Any] | None) -> Healthcheck | None:
    """A service's healthcheck: compose's (``disable`` turns it off), else its image's ``HEALTHCHECK``."""

    raw = data.get("healthcheck")
    if raw and raw.get("disable"):
        return None
    check = ServiceHealthcheck.from_json(raw) if raw and "test" in raw else image_healthcheck(image)
    if check is None:
        return None
    return Healthcheck(
        command=tuple(check.command),
        interval_s=check.interval_s,
        timeout_s=check.timeout_s,
        start_period_s=check.start_period_s,
        start_interval_s=check.start_interval_s,
        retries=check.retries,
    )


# --- deadlines -----------------------------------------------------------------------------------------------------


def healthchecks(
    harbor_task: HarborTask,
    *,
    main_image: SessionImage | None,
    service_image: Callable[[str, Mapping[str, Any]], SessionImage],
    pulled_image: Callable[[str], SessionImage | None] = lambda reference: None,
) -> dict[str, Healthcheck | None]:
    """Each service's healthcheck as its stack runs it (compose's, else its image's); a service whose image cannot be
    had (a build context the image map lacks, whose session never opens) counts compose's alone."""

    plan = harbor_task.compose
    assert plan is not None
    checks: dict[str, Healthcheck | None] = {}
    for name, data in plan.services.items():
        if data.get("external"):  # a node service's: not the stack's to wait for
            checks[name] = None
            continue
        image: Mapping[str, Any] | None = None
        try:
            if name == MAIN:
                image = main_image.healthcheck if main_image is not None else None
            else:
                main = SandboxSpec(image="")
                image = _image(harbor_task, name, data, main, main_image, service_image, pulled_image).healthcheck
        except ValueError:
            image = None
        checks[name] = _healthcheck(data, image)
    return checks


def start_bound_s(
    harbor_task: HarborTask,
    checks: Mapping[str, Healthcheck | None],
    *,
    start_s: float,
    main_health_s: float,
    job_s: float,
) -> float:
    """How long the task's stack may take to start, along its dependencies' longest path: ``start_s`` per service
    start, each sidecar's healthcheck's longest wait (``checks``, :func:`healthchecks`), an init job's run (``job_s``)
    and ``main_health_s``, what ``main``'s preparation and its own healthcheck wait for."""

    plan = harbor_task.compose
    assert plan is not None
    jobs = {
        other
        for data in plan.services.values()
        for other, settings in (data.get("depends_on") or {}).items()
        if settings.get("condition") == "service_completed_successfully"
    }
    weights = {}
    for name, data in plan.services.items():
        if data.get("external"):  # a node service's: up before the stack starts
            weights[name] = 0.0
            continue
        if name == MAIN:
            wait = main_health_s
        elif name in jobs:
            wait = job_s
        else:
            wait = check.worst_case_s if (check := checks.get(name)) is not None else 0.0
        weights[name] = start_s + wait
    graph = {name: set(data.get("depends_on") or {}) for name, data in plan.services.items()}
    return longest_path(graph, weights)


def probe_bound_s(checks: Mapping[str, Healthcheck | None]) -> float:
    """How long one probe of a stack's sidecars may take at most: the slowest healthcheck's ``retries`` runs (they run
    side by side)."""

    return max((check.probe_worst_case_s for name, check in checks.items() if name != MAIN and check), default=0.0)


# --- probes --------------------------------------------------------------------------------------------------------


async def probe(stack: Stack, *, when: str) -> tuple[Outcome | None, dict[str, ServiceState]]:
    """Probe every sidecar of ``stack``; an ``invalid`` outcome (``sidecar_unhealthy``) when one is down, naming it,
    its state and the tail of its log (``when``: "before the tests", ...), and every sidecar's state."""

    sidecars = [name for name in stack.services if name != MAIN]
    states = await stack.health(sidecars)
    down = {name: state for name, state in states.items() if not state.up}
    if not down:
        return None, states
    name, state = next(iter(down.items()))
    log = await stack.logs(name, max_bytes=LOG_TAIL_BYTES)
    others = sorted(set(down) - {name})
    reason = f"the sidecar {name} {state.describe()} {when}" + (f" (and {', '.join(others)})" if others else "")
    logger.info("Harbor stack %s: %s", stack.id, reason)
    return (
        Outcome.infra(
            "sidecar_unhealthy",
            status="invalid",
            reason=reason,
            metadata={
                "sidecar": name,
                "sidecar_status": state.status,
                **({"sidecar_exit_code": state.exit_code} if state.exit_code is not None else {}),
                **({"sidecar_detail": state.detail} if state.detail else {}),
                "sidecar_log": log[-LOG_TAIL_BYTES:],
            },
        ),
        states,
    )


def services_metadata(stack: Stack, states: Mapping[str, ServiceState]) -> dict[str, dict[str, Any]]:
    """What an outcome records of a stack's services: each one's run mode and, for a sidecar that a probe of this grade
    looked at, its state then."""

    services: dict[str, dict[str, Any]] = {}
    for name, mode in stack.run_modes.items():
        entry: dict[str, Any] = {"run_as": mode}
        if (state := states.get(name)) is not None:
            entry["state"] = state.status
        services[name] = entry
    return services


# --- collect hooks -------------------------------------------------------------------------------------------------


async def run_collect_hooks(
    hooks: Sequence[CollectHook],
    *,
    main: Sandbox,
    stack: Stack | None,
    task_name: str,
    users: Mapping[str, Sandbox] | None = None,
) -> list[str]:
    """Run ``hooks`` in order, each in its service (``main``'s sandbox without a stack, or its view as the user a hook
    names, from ``users``), best effort as Harbor runs them: a hook that fails or times out is logged, and the
    grading goes on.  The hooks that failed, as ``<service>: <command>``."""

    failed = []
    for hook in hooks:
        if stack is not None:
            sandbox = stack.services.get(hook.service)
            command = stack.rewrite(hook.command, service=hook.service)
        else:
            sandbox, command = (users or {}).get(hook.user or "", main), hook.command
        if sandbox is None:
            failed.append(f"{hook.service}: {hook.command}")
            continue
        try:
            result = await sandbox.exec(command, timeout_s=hook.timeout_s)
        except Exception as exc:
            logger.warning("Harbor task %s: the collect hook in %s failed: %s", task_name, hook.service, exc)
            failed.append(f"{hook.service}: {hook.command}")
            continue
        if result.exit_code != 0 or result.timed_out:
            detail = "timed out" if result.timed_out else f"exit {result.exit_code}"
            logger.warning(
                "Harbor task %s: the collect hook in %s %s: %s",
                task_name,
                hook.service,
                detail,
                (result.stderr or result.stdout).strip()[-500:],
            )
            failed.append(f"{hook.service}: {hook.command}")
    return failed


# --- the sidecars' log directories ---------------------------------------------------------------------------------


async def log_source(stack: Stack, logs: SidecarLogs, downloads: Path) -> Path:
    """Where grading copies a log directory of the sidecars from: the stack's volume on this host, or, where a sidecar
    that binds it runs with your subordinate ids, whose own users' files this host's user may not read, a copy that
    sidecar downloads, as its root, into ``downloads`` (a sidecar that is down leaves the volume)."""

    for service, target in logs.services.items():
        sandbox = stack.services.get(service)
        if sandbox is None or stack.run_modes.get(service) != "subids":
            continue
        copy = downloads / logs.logs / service
        try:
            await sandbox.download(target, copy)
        except Exception as exc:
            logger.warning("Cannot download the sidecars' %s logs through %s: %s", logs.logs, service, exc)
            continue
        return copy
    return stack.volume_path(logs.volume)


def snapshot_logs(directories: Mapping[str, Path], into: Path) -> dict[str, Path]:
    """Copy the regular files of the sidecars' log directories (``directories``: by log directory, ``verifier``, ...,
    where :func:`log_source` found it) into ``into``, by log directory (``into/verifier``, ...), never through a link at
    any depth, nor a link, a device or anything else, also while a sidecar swaps what is there, so that nothing a
    sidecar left there leads a copy elsewhere; the directories that hold something, by name.  ``OSError`` for what it
    cannot read, which the tests must not go without; what is gone since it was listed is left out."""

    def unless_gone(error: OSError) -> None:  # what fwalk cannot list or enter
        if not isinstance(error, FileNotFoundError):
            raise error

    copied: dict[str, Path] = {}
    for logs, source in directories.items():
        target = into / logs
        if not source.is_dir() or source.is_symlink():
            continue
        # Never follows a link to a directory.
        for directory, _names, files, directory_fd in os.fwalk(source, onerror=unless_gone):
            relative = os.path.relpath(directory, source)
            for name in files:
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        continue  # a link, a device, a FIFO: never opened
                    fd = os.open(name, _COPY_FLAGS, dir_fd=directory_fd)
                except FileNotFoundError:
                    continue  # gone since it was listed
                except OSError as exc:
                    if exc.errno == errno.ELOOP:
                        continue  # swapped for a link since it was listed (O_NOFOLLOW)
                    raise
                with os.fdopen(fd, "rb") as handle:
                    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                        continue  # swapped for something else since its stat
                    destination = target / relative / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("wb") as out:
                        shutil.copyfileobj(handle, out)
                copied[logs] = target
    return copied


async def put_logs(sandbox: Sandbox, snapshot: Mapping[str, Path]) -> None:
    """Upload a :func:`snapshot_logs` copy into the session's log directories of ``sandbox`` (``/logs/verifier``,
    ...), over what is there."""

    for logs, directory in snapshot.items():
        await sandbox.upload(directory, LOG_DIRS[logs])


__all__ = [
    "COPY_VOLUME",
    "LOGS_VOLUME",
    "SidecarLogs",
    "StackPlan",
    "bind_copies",
    "bind_mounts",
    "copied_binds",
    "filtered_copy",
    "healthchecks",
    "in_thread",
    "log_source",
    "probe",
    "probe_bound_s",
    "put_logs",
    "run_collect_hooks",
    "services_metadata",
    "snapshot_logs",
    "stack_spec",
    "start_bound_s",
]
