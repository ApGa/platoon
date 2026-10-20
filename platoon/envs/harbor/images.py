"""Task images at session open: the site's image map, and the deferred Dockerfile steps a session applies.

A lock (``platoon harbor fetch``) records each task's build plan: its build key and the trailing Dockerfile steps
deferred to session open (:mod:`platoon.envs.harbor.dockerfile`). The **image map** is a site file binding build
keys to built images, written by ``platoon harbor images build``::

    {"version": 1, "images": {"sha256:<key>": {
       "image": "/shared/images/<key>.sif", "sha256": "<file digest>", "builder": "oci",
       "env": {"PATH": "/opt/extra/bin:...", "FOO": "bar"}, "workdir": "/workdir", "user": null,
       "entrypoint": null, "cmd": null, "receipt": {...}}}}

When a session opens, :func:`session_image` picks the image (the map's entry for the task's key, the task's
``docker_image``, or the service's default), merges the image's environment with the deferred ``ENV`` steps, and
lists the file operations (``WORKDIR`` directories, ``COPY`` uploads) that :func:`apply_image_files` performs once
the sandbox runs. A task that builds from a Dockerfile but has no entry in the map is refused with ``env_config``,
never run on a default image. A prebuilt-image task without a Dockerfile gets its ``environment/`` directory
uploaded into its working directory, as Harbor does.

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

from platoon.envs.harbor.dockerfile import expand

IMAGE_MAP_VERSION = 1
# What a receipt records as ``builder_revision`` when the ``oci`` builder built the image: 1 since it unpacks the image
# under umask 0, so that its files keep the read and execute bits the image gives other users.  The receipts of the
# builder before (platoon before 0.3.0) record none (:func:`withholds_modes`).
OCI_BUILDER_REVISION = 1
ENVIRONMENT_DIR = "environment"
IMAGES_DIR = "images"  # where a Harbor cache keeps its default image maps and their images
_LOCAL_IMAGE_SUFFIXES = (".sif", ".sqsh", ".squashfs", ".img")


class ImageMapError(ValueError):
    """An image map that cannot be read, or a task whose image it does not have."""


@dataclass(frozen=True)
class ImageEntry:
    """A built image: where it is, and the configuration its runtime does not apply by itself."""

    image: str  # a .sif or .sqsh path (never given a docker:// prefix), or a registry reference
    sha256: str | None = None  # receipt: the image file's SHA-256
    builder: str | None = None  # "oci", "def" or "docker"
    env: Mapping[str, str] = field(default_factory=dict)  # from the image config; applied to every command
    workdir: str | None = None
    user: str | None = None
    entrypoint: tuple[str, ...] | None = None  # for sidecars (stacks)
    cmd: tuple[str, ...] | None = None
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
        if entry is None or self.path is None or not _is_relative_local(entry.image):
            return entry
        return replace(entry, image=str((self.path.parent / entry.image).resolve()))

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


def session_image(
    task_dir: str | Path,
    *,
    plan: Mapping[str, Any] | None,
    image_map: ImageMap | None,
    docker_image: str | None = None,
    default_image: str | None = None,
    context: str = ENVIRONMENT_DIR,
) -> SessionImage:
    """The image a session of the task at ``task_dir`` runs, from its build plan (a lock row's ``image``, or one
    computed at open) and the site's image map.

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
            plan.get("deferred") or (), env=entry.env, workdir=entry.workdir, context=task_dir / context
        )
        return SessionImage(
            image=entry.image,
            env=env,
            workdir=workdir,
            user=entry.user,
            key=key,
            context=task_dir / context,
            files=files,
        )
    image = docker_image_reference(docker_image) if docker_image else default_image
    files: tuple[FileOp, ...] = ()
    if docker_image and plan is not None and plan.get("upload_environment"):
        files = (FileOp("copy", target="", source=".", into=True),)  # into the session's working directory
    return SessionImage(image=image, env={}, workdir=None, context=task_dir / context, files=files)


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
) -> tuple[dict[str, str], str | None, tuple[FileOp, ...]]:
    """Apply deferred steps on paper: the environment and working directory after them, and their file
    operations. Variables expand as in a build, against ``ARG`` values and the environment so far (``ENV`` wins)."""

    environment = dict(env)
    arguments: dict[str, str] = {}
    current = workdir or "/"
    files: list[FileOp] = []
    for step in steps:
        op, *operands = step
        variables = {**arguments, **environment}
        if op == "ARG":
            arguments[operands[0]] = expand(operands[1], variables) if len(operands) > 1 else ""
        elif op == "ENV":
            environment[operands[0]] = expand(operands[1], variables)
        elif op == "WORKDIR":
            current = posixpath.normpath(posixpath.join(current, expand(operands[0], variables)))
            workdir = current
            files.append(FileOp("mkdir", target=current))
        elif op == "COPY":
            *sources, destination = operands
            destination = expand(destination, variables)
            target = posixpath.join(current, destination) if not destination.startswith("/") else destination
            into_dir = destination.endswith("/") or len(sources) > 1
            target = posixpath.normpath(target)
            for source in sources:
                is_dir = source == "." or (context / source.rstrip("/")).is_dir()
                if is_dir:
                    files.append(FileOp("copy", target=target, source=source.rstrip("/") or ".", into=True))
                elif into_dir:
                    name = posixpath.basename(source.rstrip("/"))
                    files.append(FileOp("copy", target=posixpath.join(target, name), source=source))
                else:
                    files.append(FileOp("copy", target=target, source=source, into=False))
        else:
            raise ImageMapError(f"unknown deferred step {op!r}")
    return environment, workdir, tuple(files)


async def apply_image_files(sandbox: Any, image: SessionImage, *, workdir: str, umask: str | None = None) -> None:
    """Perform the image's deferred file operations in a running sandbox: create ``WORKDIR`` directories (with
    ``umask``, when given, as a build's are: Docker's is 022) and upload ``COPY`` sources from the task's build context
    (``workdir`` is the session's working directory, where a prebuilt task's ``environment/`` goes)."""

    for op in image.files:
        _check_contained(sandbox, op.target or workdir)
        if op.kind == "mkdir":
            mkdir = ["mkdir", "-p", op.target]
            if umask is not None:
                mkdir = ["/bin/sh", "-c", f'umask {umask} && mkdir -p -- "$1"', "platoon-mkdir", op.target]
            result = await sandbox.exec(mkdir)
            if result.exit_code != 0:
                raise RuntimeError(f"Cannot create {op.target}: {result.stderr.strip()}")
            continue
        assert image.context is not None and op.source is not None
        source = image.context if op.source == "." else image.context / op.source
        target = op.target or workdir
        if source.is_dir() or op.into:
            if not source.is_dir():
                raise RuntimeError(f"COPY source {op.source} is not a directory")
            await sandbox.upload(source, target)
            continue
        if not op.into:
            # A file copied to a path that is a directory in the image goes into it, as in a build.
            probe = await sandbox.exec(["test", "-d", target])
            if probe.exit_code == 0:
                target = posixpath.join(target, source.name)
        await sandbox.upload(source, target)


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
