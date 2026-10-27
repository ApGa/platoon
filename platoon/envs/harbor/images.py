"""Task images at session open: the site's image map, and the deferred Dockerfile steps a session applies.

A lock (``platoon harbor fetch``) records each task's build plan: its build key and the trailing Dockerfile steps
deferred to session open (:mod:`platoon.envs.harbor.dockerfile`). The **image map** is a site file binding build
keys to built images, written by ``platoon harbor images build``::

    {"version": 1, "images": {"sha256:<key>": {
       "image": "/shared/images/<key>.sif", "sha256": "<file digest>", "builder": "oci",
       "env": {"PATH": "/opt/extra/bin:...", "FOO": "bar"}, "workdir": "/workdir", "user": null,
       "entrypoint": null, "cmd": null, "healthcheck": null, "exposed_ports": [], "volumes": [],
       "stop_signal": null, "file_owners": "<key>.owners", "receipt": {...}}}}

An entry's ``file_owners`` is its image's owner manifest (:mod:`platoon.envs.harbor.owners`), which the builder that
keeps an image's owners records (``oci``): what a sandbox that extracts the image as one user applies when it honors
users.  A *pulled* entry, keyed by a registry reference as a task names its ``docker_image`` (``docker://repo:tag``;
a sidecar's pin, below, is keyed ``pulled:<reference>``), pins the image a sandbox that honors users runs for it
(``docker://repo@sha256:...``), with its configuration's ``USER`` and the owner manifest that its registry's layers
give (``platoon prepare`` records them for such entries).

When a session opens, :func:`session_image` picks the image (the map's entry for the task's key, the task's
``docker_image``, or the service's default), merges the image's environment with the deferred ``ENV`` steps, lays the
deferred metadata (``ENTRYPOINT``, ``CMD``, ``HEALTHCHECK``) over the image's own, and lists the file operations
(``WORKDIR`` directories, ``COPY`` uploads) that :func:`apply_image_files` performs once the sandbox runs. A task that
builds from a Dockerfile but has no entry in the map is refused with ``env_config``, never run on a default image. A
prebuilt-image task without a Dockerfile gets its ``environment/`` directory uploaded into its working directory, as
Harbor does.

The map also **pins** the registry images that stacks' sidecars run as written (``image: redis:7-alpine``), under the
key ``pulled:<reference>`` (:func:`pulled_key`): an entry of the ``pulled`` builder whose image is the reference by the
digest its registry served (``docker://redis@sha256:...``), with that image's configuration, from which sessions
resolve the sidecar's command on every provider, and which a pod's runtimes do not apply by themselves
(:func:`platoon.envs.harbor.builders.pin_images`).

A lock-backed Harbor entry uses its **default image map**, ``images/<provider>.json`` in its Harbor cache
(:func:`default_image_map`), when that file exists: ``platoon prepare`` builds the entry's images for its sandbox
provider into ``images/<provider>/`` there and records them in it. An entry that serves a directory (``dataset_dir``,
``task_dirs``) has no image map: its tasks run their ``docker_image`` or the service's ``default_image``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import posixpath
import re
import stat
import uuid
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from platoon.envs.harbor.dockerfile import copy_operands, expand, has_variables
from platoon.envs.harbor.users import copy_owner, read_accounts
from platoon.envs.harbor.views import give, uploaded_paths

IMAGE_MAP_VERSION = 1
# What a receipt records as ``builder_revision`` when the ``oci`` builder built the image: 1 since it unpacks the image
# under umask 0, so that its files keep the read and execute bits the image gives other users.  The receipts of the
# builder before (platoon before 0.3.0) record none (:func:`withholds_modes`).
OCI_BUILDER_REVISION = 1
ENVIRONMENT_DIR = "environment"
IMAGES_DIR = "images"  # where a Harbor cache keeps its default image maps and their images
PULLED = "pulled:"  # the prefix of the key of a registry image that sidecars run
_LOCAL_IMAGE_SUFFIXES = (".sif", ".sqsh", ".squashfs", ".img")
# Prints how many directories of the path $1, from its end, are missing: those a copy into $1 makes.
_MISSING_SCRIPT = (
    'path=$1; missing=0; while [ ! -e "$path" ]; do missing=$((missing + 1)); path=$(dirname -- "$path"); done; '
    'echo "$missing"'
)


class ImageMapError(ValueError):
    """An image map that cannot be read, or a task whose image it does not have."""


@dataclass(frozen=True)
class ImageEntry:
    """A built image: where it is, and the configuration its runtime does not apply by itself."""

    image: str  # a .sif or .sqsh path (never given a docker:// prefix), or a registry reference
    sha256: str | None = None  # receipt: the image file's SHA-256
    # "oci", "def" or "docker"; "pulled" for a registry image that sidecars run, or that a sandbox honoring users runs
    builder: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)  # from the image config; applied to every command
    workdir: str | None = None
    user: str | None = None
    entrypoint: tuple[str, ...] | None = None  # what runs main's or a compose service's command (stacks)
    cmd: tuple[str, ...] | None = None
    # The image's HEALTHCHECK as compose writes a healthcheck (durations in seconds), its EXPOSE ports ("80/tcp"),
    # VOLUME paths and STOPSIGNAL, from its configuration.
    healthcheck: Mapping[str, Any] | None = None
    exposed_ports: tuple[str, ...] = ()
    volumes: tuple[str, ...] = ()
    stop_signal: str | None = None
    # Its owner manifest (a file beside the map, or an absolute path); None where its builder recorded none.
    file_owners: str | None = None
    receipt: Mapping[str, Any] = field(default_factory=dict)  # builder version, time, log, size, ...

    @classmethod
    def from_json(cls, data: Mapping[str, Any], where: str = "image map entry") -> ImageEntry:
        if not isinstance(data, Mapping) or not isinstance(data.get("image"), str) or not data["image"]:
            raise ImageMapError(f"{where} needs an image")
        env = data.get("env") or {}
        if not isinstance(env, Mapping) or not all(isinstance(v, str) for v in env.values()):
            raise ImageMapError(f"{where}: env must map names to strings")
        return cls(
            image=data["image"],
            sha256=data.get("sha256"),
            builder=data.get("builder"),
            env=dict(env),
            workdir=data.get("workdir") or None,
            user=data.get("user") or None,
            entrypoint=tuple(data["entrypoint"]) if data.get("entrypoint") else None,
            cmd=tuple(data["cmd"]) if data.get("cmd") else None,
            healthcheck=dict(data["healthcheck"]) if data.get("healthcheck") else None,
            exposed_ports=tuple(data.get("exposed_ports") or ()),
            volumes=tuple(data.get("volumes") or ()),
            stop_signal=data.get("stop_signal") or None,
            file_owners=data.get("file_owners") or None,
            receipt=dict(data.get("receipt") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "sha256": self.sha256,
            "builder": self.builder,
            "env": dict(self.env),
            "workdir": self.workdir,
            "user": self.user,
            "entrypoint": list(self.entrypoint) if self.entrypoint else None,
            "cmd": list(self.cmd) if self.cmd else None,
            **({"healthcheck": dict(self.healthcheck)} if self.healthcheck else {}),
            **({"exposed_ports": list(self.exposed_ports)} if self.exposed_ports else {}),
            **({"volumes": list(self.volumes)} if self.volumes else {}),
            **({"stop_signal": self.stop_signal} if self.stop_signal else {}),
            **({"file_owners": self.file_owners} if self.file_owners else {}),
            "receipt": dict(self.receipt),
        }


def withholds_modes(entry: ImageEntry) -> bool:
    """Whether the image of ``entry`` may keep its files from other users: the ``oci`` builder of platoon before 0.3.0
    built it (its receipt records no ``builder_revision``), and unpacked it under the host's umask, which on a host
    whose umask is 027 took from its files the read and execute bits the image gives other users.  A sandbox whose
    commands switch users, such as a verifier that runs as root, needs them: build such an image again."""

    return entry.builder == "oci" and not entry.receipt.get("builder_revision")


class ImageMap:
    """Build keys (``sha256:<hex>``) to :class:`ImageEntry`. Relative image paths resolve against the map file's
    directory, so a map and its images can move together."""

    def __init__(self, entries: Mapping[str, ImageEntry] | None = None, *, path: Path | None = None) -> None:
        self.entries: dict[str, ImageEntry] = dict(entries or {})
        self.path = path

    @classmethod
    def load(cls, source: str | Path | Mapping[str, Any]) -> ImageMap:
        if isinstance(source, Mapping):
            return cls._from_json(source, path=None, where="the image map")
        path = Path(source)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise ImageMapError(f"image map {path} does not exist") from None
        except json.JSONDecodeError as exc:
            raise ImageMapError(f"image map {path} is not JSON: {exc}") from None
        return cls._from_json(data, path=path, where=str(path))

    @classmethod
    def _from_json(cls, data: Mapping[str, Any], *, path: Path | None, where: str) -> ImageMap:
        if not isinstance(data, Mapping) or data.get("version", IMAGE_MAP_VERSION) != IMAGE_MAP_VERSION:
            raise ImageMapError(f"{where}: expected an image map of version {IMAGE_MAP_VERSION}")
        images = data.get("images") or {}
        if not isinstance(images, Mapping):
            raise ImageMapError(f"{where}: images must map build keys to entries")
        return cls({key: ImageEntry.from_json(value, f"{where} [{key}]") for key, value in images.items()}, path=path)

    def get(self, key: str) -> ImageEntry | None:
        entry = self.entries.get(key)
        if entry is None or self.path is None:
            return entry
        if _is_relative_local(entry.image):
            entry = replace(entry, image=str((self.path.parent / entry.image).resolve()))
        if entry.file_owners is not None and not Path(entry.file_owners).is_absolute():
            entry = replace(entry, file_owners=str((self.path.parent / entry.file_owners).resolve()))
        return entry

    def to_json(self) -> dict[str, Any]:
        return {
            "version": IMAGE_MAP_VERSION,
            "images": {key: self.entries[key].to_json() for key in sorted(self.entries)},
        }


def has_image(image_map: ImageMap | None, key: str) -> bool:
    """Whether ``image_map`` has an image for build ``key`` that a sandbox can start from: an entry, and, for a local
    image file (``.sif``, ``.sqsh``, ...), the file, whole (:func:`has_image_file`)."""

    entry = image_map.get(key) if image_map is not None else None
    return entry is not None and has_image_file(entry)


def has_image_file(entry: ImageEntry) -> bool:
    """Whether the image of a map's ``entry`` (as :meth:`ImageMap.get` gives it) is there, as far as a ``stat`` tells: a
    reference that is not a local file, or a local image file that exists and has the size its build recorded
    (``receipt.size_bytes``), so that a truncated copy counts as missing.  ``platoon harbor images verify`` checks the
    content against the recorded SHA-256."""

    path = local_image_path(entry.image)
    if path is None:
        return True
    size = entry.receipt.get("size_bytes")
    try:
        found = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(found.st_mode):
        return False
    return not isinstance(size, int) or isinstance(size, bool) or found.st_size == size


def pulled_key(reference: str) -> str:
    """The image map's key of a registry image that a stack's sidecar runs, as its compose file names it
    (``redis:7-alpine``): ``pulled:redis:7-alpine``."""

    return f"{PULLED}{reference}"


def pulled_image(entry: ImageEntry) -> SessionImage:
    """What a sidecar runs of a pinned registry image (``entry``, :func:`pulled_key`'s): the image by its digest,
    with its configuration, and nothing deferred."""

    return SessionImage(
        image=entry.image,
        env=dict(entry.env),
        workdir=entry.workdir,
        user=entry.user,
        entrypoint=entry.entrypoint,
        cmd=entry.cmd,
        healthcheck=entry.healthcheck,
    )


def default_image_map(cache: str | Path, provider: str) -> Path:
    """The image map of a lock-backed Harbor entry: ``images/<provider>.json`` in its Harbor cache.
    A provider named as ``module:Class`` is written with ``-`` for the characters a file name should not hold."""

    return Path(cache) / IMAGES_DIR / f"{_file_name(provider)}.json"


def default_image_dir(cache: str | Path, provider: str) -> Path:
    """Where ``platoon prepare`` puts the image files of a default image map (:func:`default_image_map`):
    ``images/<provider>/`` in the Harbor cache."""

    return Path(cache) / IMAGES_DIR / _file_name(provider)


def _file_name(provider: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", provider) or "provider"


def update_image_map(path: str | Path, entries: Mapping[str, ImageEntry]) -> ImageMap:
    """Add ``entries`` to the map file at ``path`` (created if missing), under a lock: build shards running at
    once each add their own images."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _flock(path.with_name(f".{path.name}.lock")):  # a hidden lock file next to the map
        current = ImageMap.load(path) if path.exists() else ImageMap(path=path)
        current.entries.update(entries)
        write_json_atomic(path, current.to_json())
    return current


@dataclass(frozen=True)
class FileOp:
    """A file operation of a deferred step: ``mkdir`` a directory, or ``copy`` a context path to a destination."""

    kind: str  # "mkdir" or "copy"
    target: str  # absolute sandbox path
    source: str | None = None  # copy: path relative to the build context ("." for all of it)
    into: bool = False  # copy: the destination is a directory the source goes into
    # copy: its --chown; mkdir: the USER in effect at its WORKDIR (None for none); expanded (a user, optionally
    # :group), which copy_owner reads
    owner: str | None = None


@dataclass(frozen=True)
class SessionImage:
    """What a session's sandbox runs: the image, its environment and working directory, and the deferred file
    operations to perform once it runs."""

    image: str | None
    env: dict[str, str]  # the image's environment and the deferred ENV steps (task and service env go on top)
    workdir: str | None  # the deferred WORKDIR, else the image's
    user: str | None = None
    key: str | None = None  # the build key, when the image comes from the map
    context: Path | None = None  # the build context the copies read from
    files: tuple[FileOp, ...] = ()
    # The image's ENTRYPOINT and CMD (the map's, with the deferred ones over them), and its HEALTHCHECK as compose
    # writes one; None where platoon does not know them, as for a task's prebuilt docker_image.
    entrypoint: tuple[str, ...] | None = None
    cmd: tuple[str, ...] | None = None
    healthcheck: Mapping[str, Any] | None = None
    # The image's owner manifest (ImageEntry.file_owners); None where none is known.
    file_owners: str | None = None


def session_image(
    task_dir: str | Path,
    *,
    plan: Mapping[str, Any] | None,
    image_map: ImageMap | None,
    docker_image: str | None = None,
    default_image: str | None = None,
    context: str = ENVIRONMENT_DIR,
    pulled: bool = False,
) -> SessionImage:
    """The image a session of the task at ``task_dir`` runs, from its build plan (a lock row's ``image``, or one
    computed at open) and the site's image map.  With ``pulled``, a ``docker_image`` that the map has a pulled entry
    for runs that entry's pinned image, with its ``USER`` and owner manifest (a sandbox that honors users).

    Raises :class:`ImageMapError` for a task that builds from a Dockerfile and has no image in the map.
    """

    task_dir = Path(task_dir)
    if plan is not None and plan.get("key"):
        key = plan["key"]
        entry = image_map.get(key) if image_map is not None else None
        if entry is None:
            if image_map is None:
                missing = "and the service has no image map"
            else:
                where = f"image map {image_map.path}" if image_map.path else "the image map"
                missing = f"which {where} has no image for"
            raise ImageMapError(
                f"Harbor task {task_dir.name} builds from {context}/Dockerfile (build key {key}), {missing}; build "
                "it with `platoon prepare` on the run's config, which builds it into the entry's image map"
            )
        env, workdir, files = deferred_effects(
            plan.get("deferred") or (),
            env=entry.env,
            workdir=entry.workdir,
            context=task_dir / context,
            user=entry.user,
        )
        config = plan.get("config") or {}
        entrypoint = config.get("entrypoint", entry.entrypoint)
        cmd = config.get("cmd", entry.cmd)
        return SessionImage(
            image=entry.image,
            env=env,
            workdir=workdir,
            user=planned_user(config, entry, env),
            key=key,
            context=task_dir / context,
            files=files,
            entrypoint=tuple(entrypoint) if entrypoint is not None else None,
            cmd=tuple(cmd) if cmd is not None else None,
            healthcheck=config.get("healthcheck", entry.healthcheck),
            file_owners=entry.file_owners,
        )
    image = docker_image_reference(docker_image) if docker_image else default_image
    files: tuple[FileOp, ...] = ()
    if docker_image and plan is not None and plan.get("upload_environment"):
        files = (FileOp("copy", target="", source=".", into=True),)  # into the session's working directory
    entry = image_map.get(image) if pulled and image is not None and image_map is not None else None
    if entry is not None:
        return SessionImage(
            image=entry.image,
            env={},
            workdir=None,
            user=entry.user,
            context=task_dir / context,
            files=files,
            file_owners=entry.file_owners,
        )
    return SessionImage(image=image, env={}, workdir=None, context=task_dir / context, files=files)


def planned_user(config: Mapping[str, Any], entry: ImageEntry, env: Mapping[str, str]) -> str | None:
    """The ``USER`` of an image whose build plan records ``config`` (its suffix's metadata) and that the map's
    ``entry`` built: the plan's, its variables expanded against the image's environment ``env`` as a build expands
    them, else the built image's own; None for none, or an empty one (root)."""

    user = config.get("user", entry.user)
    if user and has_variables(user):
        user = expand(user, env)
    return user or None


def docker_image_reference(image: str) -> str:
    """A task's ``docker_image`` as a provider reference: local image files (``.sif``, ``.sqsh``) and absolute
    paths as written, references without a scheme as ``docker://<image>``."""

    if "://" in image or image.startswith(("/", "./", "../", "~")) or image.endswith(_LOCAL_IMAGE_SUFFIXES):
        return image
    return f"docker://{image}"


def local_image_path(image: str) -> Path | None:
    """The path of a local image file (an image ending in ``.sif``, ``.sqsh``, ``.squashfs`` or ``.img``), or None
    for a registry reference or a name such as ``local``."""

    if "://" in image or not image.endswith(_LOCAL_IMAGE_SUFFIXES):
        return None
    return Path(image).expanduser()


def deferred_effects(
    steps: Sequence[Sequence[str]],
    *,
    env: Mapping[str, str],
    workdir: str | None,
    context: Path,
    user: str | None = None,
) -> tuple[dict[str, str], str | None, tuple[FileOp, ...]]:
    """Apply deferred steps on paper: the environment and working directory after them, and their file
    operations. Variables expand as in a build, against ``ARG`` values and the environment so far (``ENV`` wins).
    A ``WORKDIR``'s ``mkdir`` names the ``USER`` in effect there as its owner, which BuildKit makes its directories
    as: the last deferred ``USER`` before it, else ``user``, the built image's own."""

    environment = dict(env)
    arguments: dict[str, str] = {}
    current = workdir or "/"
    current_user = user or None
    files: list[FileOp] = []
    for step in steps:
        op, *operands = step
        variables = {**arguments, **environment}
        if op == "ARG":
            arguments[operands[0]] = expand(operands[1], variables) if len(operands) > 1 else ""
        elif op == "ENV":
            environment[operands[0]] = expand(operands[1], variables)
        elif op == "USER":
            current_user = expand(operands[0], variables) or None
        elif op == "WORKDIR":
            current = posixpath.normpath(posixpath.join(current, expand(operands[0], variables)))
            workdir = current
            files.append(FileOp("mkdir", target=current, owner=current_user))
        elif op == "COPY":
            owner, sources, destination = copy_operands(step)
            owner = expand(owner, variables) if owner is not None else None
            destination = expand(destination, variables)
            target = posixpath.join(current, destination) if not destination.startswith("/") else destination
            into_dir = destination.endswith("/") or len(sources) > 1
            target = posixpath.normpath(target)
            for source in sources:
                is_dir = source == "." or (context / source.rstrip("/")).is_dir()
                if is_dir:
                    files.append(FileOp("copy", target, source=source.rstrip("/") or ".", into=True, owner=owner))
                elif into_dir:
                    name = posixpath.basename(source.rstrip("/"))
                    files.append(FileOp("copy", posixpath.join(target, name), source=source, owner=owner))
                else:
                    files.append(FileOp("copy", target, source=source, owner=owner))
        else:
            raise ImageMapError(f"unknown deferred step {op!r}")
    return environment, workdir, tuple(files)


async def apply_image_files(sandbox: Any, image: SessionImage, *, workdir: str, owners: bool = False) -> None:
    """Perform the image's deferred file operations in a running sandbox: create ``WORKDIR`` directories and upload
    ``COPY`` sources from the task's build context (``workdir`` is the session's working directory, where a prebuilt
    task's ``environment/`` goes).  With ``owners`` (a sandbox that honors users), a copy's ``--chown`` gives its owner
    (:func:`~platoon.envs.harbor.users.copy_owner`) what it copies and the directories it makes, and a ``WORKDIR``
    gives the directories it makes to the ``USER`` in effect there, read as BuildKit reads it, the same way; a
    destination or a directory that was there keeps its own, as BuildKit does.  Elsewhere every file is the sandbox's
    own user's."""

    accounts: tuple[str, str] | None = None
    for op in image.files:
        _check_contained(sandbox, op.target or workdir)
        if op.kind == "mkdir":
            made = await _missing_directories(sandbox, op.target) if owners and op.owner else []
            result = await sandbox.exec(["mkdir", "-p", op.target])
            if result.exit_code != 0:
                raise RuntimeError(f"Cannot create {op.target}: {result.stderr.strip()}")
            if made:
                assert op.owner is not None
                accounts = accounts or await read_accounts(sandbox)
                if (owner := copy_owner(op.owner, *accounts)) != "0:0":
                    await give(sandbox, owner, made)
            continue
        assert image.context is not None and op.source is not None
        source = image.context if op.source == "." else image.context / op.source
        target = op.target or workdir
        if source.is_dir() or op.into:
            if not source.is_dir():
                raise RuntimeError(f"COPY source {op.source} is not a directory")
        elif not op.into:
            # A file copied to a path that is a directory in the image goes into it, as in a build.
            probe = await sandbox.exec(["test", "-d", target])
            if probe.exit_code == 0:
                target = posixpath.join(target, source.name)
        if not (owners and op.owner):
            await sandbox.upload(source, target)
            continue
        accounts = accounts or await read_accounts(sandbox)
        owner = copy_owner(op.owner, *accounts)
        made = await _missing_directories(sandbox, target if source.is_dir() else posixpath.dirname(target))
        await sandbox.upload(source, target)
        copied = [path for path, _ in uploaded_paths(source, target)]
        if source.is_dir():
            copied = copied[1:]  # the destination: given with the directories the copy made, else kept as it was
        await give(sandbox, owner, [*made, *copied])


async def _missing_directories(sandbox: Any, directory: str) -> list[str]:
    """The directories of the path ``directory`` that the sandbox lacks, from the top down to ``directory``: those that
    a copy into it makes."""

    directory = posixpath.normpath(directory)
    result = await sandbox.exec(["/bin/sh", "-c", _MISSING_SCRIPT, "platoon-missing", directory], cwd="/")
    if result.exit_code != 0 or not result.stdout.strip().isdigit():
        raise RuntimeError(f"Cannot look for {directory}: {(result.stderr or result.stdout).strip()}")
    missing: list[str] = []
    for _ in range(int(result.stdout)):
        missing.insert(0, directory)
        directory = posixpath.dirname(directory)
    return missing


def _check_contained(sandbox: Any, target: str) -> None:
    """Refuse a destination the ``process`` provider would leave on the host: it maps only some sandbox paths
    (``/app``, ``/tests``, ``/logs``, ``/solution``, ``/tmp``, the workdir) below its root."""

    host_path, root = getattr(sandbox, "host_path", None), getattr(sandbox, "root", None)
    if host_path is None or root is None:
        return
    resolved = Path(host_path(target))
    if resolved != Path(root) and Path(root) not in resolved.parents:
        raise RuntimeError(
            f"The {type(sandbox).__name__} maps only some sandbox paths below its root, so a deferred Dockerfile step "
            f"would write {target} on the host; use a provider with a real sandbox for this task"
        )


def write_json_atomic(path: str | Path, data: Any, *, indent: int | None = 1) -> None:
    """Write JSON to ``path`` through a new file in the same directory (created with the umask's mode) and a
    rename."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(data, stream, indent=indent, sort_keys=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


@contextlib.contextmanager
def _flock(path: Path) -> Generator[None, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_relative_local(image: str) -> bool:
    return "://" not in image and not image.startswith(("/", "~")) and image.endswith(_LOCAL_IMAGE_SUFFIXES)
