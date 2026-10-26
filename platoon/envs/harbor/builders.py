"""Build the images a lock's tasks need, once per site: ``platoon harbor images plan``, ``build``, ``pin``, ``verify``.

:func:`plan_images` groups a lock's runnable tasks by build key (:mod:`platoon.envs.harbor.dockerfile`) and says
which keys a site's image map lacks; an entry whose image file is gone counts as missing. The keys are those of the
tasks' own images, their separate verifiers' and their stacks' sidecars' that build from a context of the task
(``build:`` in the compose file, planned whole). :func:`build_images`
builds those from a representative task's Dockerfile prefix (a sidecar's whole Dockerfile, with its build arguments)
and records each image in the map with a receipt
(builder, version, time, log, size, SHA-256, and the digest of each base image; for ``oci``, the builder's revision,
:data:`~platoon.envs.harbor.images.OCI_BUILDER_REVISION`). :func:`verify_images` checks the map's image files against
the SHA-256 their entries record.

The registry images that sidecars run as written (``image: redis:7-alpine``) are planned too (:class:`PlannedPull`),
and :func:`pin_images` **pins** each in the map: the digest its registry serves for it, and the configuration the
registry gives its image for this host's platform (``ENTRYPOINT``, ``CMD``, ``ENV``, ``WORKDIR``, ``USER``,
``HEALTHCHECK``, ...), which sessions run it with, since platoon gives every runtime a sidecar's whole command and a
pod's runtimes apply none of an image's configuration by themselves (:func:`registry_config`).

A build resolves each base image (``FROM``) to the digest its registry serves for it (:func:`registry_digest`)
and builds from ``reference@digest``, so the receipt names exactly what the image was built from. The build key,
computed offline when the lock was fetched, does not change. A base whose digest cannot be resolved (no network,
a private registry) is built as written, with a warning, and its digest in the receipt is null.

Builders:

- ``oci`` (SingularityCE 4.1 or later, the default): ``singularity build --oci`` runs the Dockerfile with rootless
  BuildKit, as Docker builds it: multi-stage builds, step flags, and ``ADD`` of a URL (verifying its ``--checksum``,
  never extracting a remote archive) or of a git repository. SingularityCE's BuildKit has no gateway frontend to
  load the frontend image a ``# syntax=`` directive names, but its built-in frontend implements Docker's own
  Dockerfile syntax: a directive that names a 1.x release of that (``docker/dockerfile:1.7``) is left out of the
  Dockerfile it builds, and one that names another frontend (labs, a third party's) is refused, naming the
  ``docker`` builder. The native runtime cannot run the OCI-SIF it writes, so the builder extracts its single
  squashfs layer
  and image config (``singularity sif list``/``dump``), unpacks the layer (``unsquashfs``) and builds a native SIF
  from the tree (``singularity build``). The image's ``ENV``, ``WORKDIR``, ``USER``, ``ENTRYPOINT`` and ``CMD``
  go into the map from the config, since a SIF built from a directory carries none of them.
- ``def``: a translation to a ``Bootstrap: docker`` definition file built with ``--fakeroot``. As in a Docker
  build, each ``RUN`` runs in a shell of its own that starts in the current ``WORKDIR`` and sees the ``ARG``
  values, and ``ENV`` values are literal but for their variable references (:func:`definition_file`). It refuses
  what it cannot translate faithfully (multi-stage builds, ``ARG`` without a default, ``USER``, ``SHELL``, flags,
  here-documents, ``ADD`` of a URL, a git repository or an archive), naming the ``oci`` builder, and never falls
  back to the base image.  A definition file keeps none of the Dockerfile's image configuration, so the map records
  its base image's (from the registry, as a pin does), under what the Dockerfile itself sets, which the lock's plan
  holds.  It is the builder of the ``singularity`` provider where the runtime is Apptainer, which has no BuildKit
  (:func:`provider_builder`).
- ``docker``: ``docker build`` (or ``podman`` with ``PLATOON_DOCKER=podman``) tagged ``platoon-harbor:<key>``, for
  the docker provider; with ``push``, pushed and recorded by digest (enroot sites import that digest).  It alone passes
  a sidecar's build arguments (``--build-arg``); the others refuse an image that has any.
"""

from __future__ import annotations

import contextlib
import fcntl
import functools
import hashlib
import http.client
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from platoon.envs.harbor.dockerfile import (
    DOCKERFILE,
    DockerfilePlan,
    expand,
    image_config,
    parse_assignments,
    plan_dockerfile,
    remote_source,
    shell_word,
    variables_defined,
    variables_in_scope,
)
from platoon.envs.harbor.hashing import file_sha256
from platoon.envs.harbor.images import (
    OCI_BUILDER_REVISION,
    ImageEntry,
    ImageMap,
    docker_image_reference,
    has_image_file,
    local_image_path,
    pulled_key,
    update_image_map,
)
from platoon.envs.harbor.lock import DatasetLock, LockedTask
from platoon.utils.scratch import node_scratch

logger = logging.getLogger(__name__)

# Resolves a base image reference to the digest its registry serves for it: None (or a DigestError) when it cannot.
DigestResolver = Callable[[str], str | None]
# Resolves an image reference to that digest and the configuration of its image for this host (registry_config).
ConfigResolver = Callable[[str], tuple[str, Mapping[str, Any]]]

BUILDERS = ("oci", "def", "docker")
# The builder that builds images a sandbox provider runs: SIF files for singularity (def where the runtime is
# Apptainer: provider_builder), local images for docker.  The other providers pull their images from a registry
# (enroot), or run none (process).
PROVIDER_BUILDERS = {"singularity": "oci", "docker": "docker"}
_SQUASHFS_MAGIC = b"hsqs"
# A "# syntax=" parser directive, and the frontend images of Docker's own Dockerfile syntax, 1.x releases (and
# latest) without labs features, which BuildKit's built-in Dockerfile frontend implements.
_SYNTAX_DIRECTIVE = re.compile(r"#\s*syntax\s*=\s*(?P<image>\S+)", re.IGNORECASE)
_DOCKERFILE_FRONTEND = re.compile(
    r"(?:docker\.io/)?docker/dockerfile:(?:1(?:\.[0-9]+){0,2}|latest)(?:@sha256:[0-9a-f]{64})?"
)
# singularity-buildkitd listens on $XDG_RUNTIME_DIR/<this, with its pid>, a path it refuses beyond 104 bytes.
_BUILDKIT_SOCKET = "/singularity-buildkitd/singularity-buildkitd-4194304.sock"
_SOCKET_PATH_MAX = 104


class BuildError(RuntimeError):
    """An image that could not be built, with the log that says why."""


@dataclass
class PlannedImage:
    """One build key of a lock: the tasks that use it and whether the image map has it."""

    key: str
    role: str  # "environment", "verifier" or "service" (a stack's sidecar): the first role that needs the key
    tasks: list[str] = field(default_factory=list)
    base: list[str] = field(default_factory=list)
    present: bool = False  # the map has an entry, and its image file exists if the image is a local file
    example: LockedTask | None = None  # the task whose Dockerfile builds it (the first by name)
    notes: list[str] = field(default_factory=list)
    # A service's build context and Dockerfile in the example task, and its build arguments.
    context: str | None = None
    dockerfile: str = DOCKERFILE
    build_args: dict[str, str] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "shared" if len(self.tasks) > 1 else "per_task"

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "kind": self.kind,
            "tasks": len(self.tasks),
            "base": self.base,
            "present": self.present,
            "example": self.example.name if self.example else None,
            **({"context": f"{self.context}/{self.dockerfile}"} if self.context else {}),
            **({"notes": self.notes} if self.notes else {}),
        }


@dataclass
class PlannedPull:
    """A registry image that the sidecars of a lock's stacks run: the tasks that run it, and whether the image map
    pinned it."""

    reference: str  # as the compose file names it
    tasks: list[str] = field(default_factory=list)
    pinned: bool = False
    role: ClassVar[str] = "service"

    @property
    def key(self) -> str:
        """Its key in the image map (:func:`~platoon.envs.harbor.images.pulled_key`)."""

        return pulled_key(self.reference)

    def to_json(self) -> dict[str, Any]:
        return {"reference": self.reference, "tasks": len(self.tasks), "pinned": self.pinned}


@dataclass
class ImagePlan:
    images: list[PlannedImage]
    prebuilt: dict[str, int]  # registry references of prebuilt-image tasks -> task count
    default_image: int  # tasks without docker_image or Dockerfile (they run on the service's default_image)
    refused: dict[str, int]  # refusal kind -> task count (they need no image)
    pulls: list[PlannedPull] = field(default_factory=list)  # the registry images sidecars run, most-used first

    @property
    def missing(self) -> list[PlannedImage]:
        return [image for image in self.images if not image.present]

    @property
    def unpinned(self) -> list[PlannedPull]:
        return [pull for pull in self.pulls if not pull.pinned]

    def to_json(self) -> dict[str, Any]:
        return {
            "images": [image.to_json() for image in self.images],
            "missing": len(self.missing),
            "prebuilt": self.prebuilt,
            "default_image": self.default_image,
            "refused": self.refused,
            **(
                {"pulls": [pull.to_json() for pull in self.pulls], "unpinned": len(self.unpinned)} if self.pulls else {}
            ),
        }


def plan_images(lock: DatasetLock, image_map: ImageMap | None = None) -> ImagePlan:
    """The build keys the lock's runnable tasks need (their own images', their separate verifiers' and their stacks'
    sidecars'), most-used first, and which the image map has, and the registry images their sidecars run, and which
    the map pinned. An entry whose image is a local file (``.sif``, ``.sqsh``, ...) counts only while the file
    exists."""

    images: dict[str, PlannedImage] = {}
    pulls: dict[str, PlannedPull] = {}
    prebuilt: dict[str, int] = {}
    refused: dict[str, int] = {}
    default_image = 0
    for task in lock.tasks:
        if task.refused is not None:
            kind = task.refused.split(":", 1)[0]
            refused[kind] = refused.get(kind, 0) + 1
            continue
        for role, plan in (("environment", task.image), ("verifier", task.verifier_image)):
            if not plan:
                continue
            if plan.get("kind") == "prebuilt":
                prebuilt[plan["ref"]] = prebuilt.get(plan["ref"], 0) + 1
            elif plan.get("key"):
                image = images.setdefault(plan["key"], PlannedImage(key=plan["key"], role=role))
                image.tasks.append(task.name)
                if image.example is None or task.name < image.example.name:
                    image.example = task
                    image.base = list(plan.get("base") or [])
                    image.notes = list(plan.get("notes") or [])
            elif role == "environment":
                default_image += 1
        for build in task.service_builds():
            image = images.setdefault(build["key"], PlannedImage(key=build["key"], role="service"))
            if not image.tasks or image.tasks[-1] != task.name:  # two sidecars of one task may share an image
                image.tasks.append(task.name)
            if image.example is None or task.name < image.example.name:
                image.example = task
                image.base = list(build.get("base") or [])
                image.context, image.dockerfile = build["context"], build.get("dockerfile") or DOCKERFILE
                image.build_args = dict(build.get("args") or {})
        for reference in dict.fromkeys(task.service_images()):
            pulls.setdefault(reference, PlannedPull(reference)).tasks.append(task.name)
    for image in images.values():
        entry = image_map.get(image.key) if image_map is not None else None
        image.present = entry is not None and has_image_file(entry)
    for pull in pulls.values():
        pull.pinned = image_map is not None and image_map.get(pulled_key(pull.reference)) is not None
    ordered = sorted(images.values(), key=lambda image: (-len(image.tasks), image.key))
    by_use = sorted(pulls.values(), key=lambda pull: (-len(pull.tasks), pull.reference))
    return ImagePlan(images=ordered, prebuilt=prebuilt, default_image=default_image, refused=refused, pulls=by_use)


@dataclass(frozen=True)
class ImageCheck:
    """One image of a lock checked against the image map."""

    image: PlannedImage | PlannedPull
    status: str  # "ok", "missing" (no entry, no file, another SHA-256, no pin) or "unchecked" (nothing to hash)
    detail: str


def verify_images(lock: DatasetLock, image_map: ImageMap, *, pins: bool = True) -> list[ImageCheck]:
    """Check the image map's entry for each build key the lock's runnable tasks use: it must exist, and a local
    image file must exist and hash to the SHA-256 the entry records. A registry reference cannot be hashed here,
    nor can a file whose entry records no SHA-256: those are ``unchecked``.  With ``pins`` (the providers that run
    images), each registry image that sidecars run must be pinned, as a run checks: one that is not is ``missing``, a
    pin ``unchecked``."""

    hashes: dict[Path, str] = {}  # keys aliased to one file hash it once
    checks: list[ImageCheck] = []
    plan = plan_images(lock, image_map)
    for image in plan.images:
        entry = image_map.get(image.key)
        path = local_image_path(entry.image) if entry is not None else None
        if entry is None:
            status, detail = "missing", "not in the image map"
        elif path is None:
            status, detail = "unchecked", f"{entry.image} is not a local image file, so it is not checked"
        elif not path.is_file():
            status, detail = "missing", f"{path} does not exist"
        elif not entry.sha256:
            status, detail = "unchecked", f"{path} has no SHA-256 in the image map, so it is not checked"
        else:
            if path not in hashes:
                hashes[path] = file_sha256(path)
            expected = entry.sha256.lower().removeprefix("sha256:")
            if hashes[path] == expected:
                status, detail = "ok", f"{path} matches its SHA-256"
            else:
                status, detail = "missing", f"{path} has SHA-256 {hashes[path]}, the image map says {expected}"
        checks.append(ImageCheck(image=image, status=status, detail=detail))
    for pull in plan.pulls if pins else ():
        if (entry := image_map.get(pull.key)) is None:
            status, detail = "missing", "not pinned in the image map (platoon prepare, or platoon harbor images pin)"
        else:
            status, detail = "unchecked", f"pinned as {entry.image}, a registry reference, so it is not checked"
        checks.append(ImageCheck(image=pull, status=status, detail=detail))
    return checks


def build_images(
    lock: DatasetLock,
    *,
    cache: str | Path,
    image_map: str | Path,
    out_dir: str | Path | None = None,
    builder: str = "oci",
    shard: tuple[int, int] = (0, 1),
    keys: Sequence[str] | None = None,
    scratch: str | Path | None = None,
    timeout_s: float = 3600.0,
    push: str | None = None,
    log: Callable[[str], None] | None = None,
    resolve_digest: DigestResolver | None = None,
) -> dict[str, ImageEntry]:
    """Build the images ``image_map`` lacks (or ``keys``), this shard's share of them, and add each to the map as
    soon as it is built. Returns the new entries; raises :class:`BuildError` after trying them all if any failed.

    Each image is built under a file lock of its own next to the map, so builds of one map that run at once, on one
    node or on several that share its filesystem, build each image once: a build that waited for another finds the
    image in the map and leaves it (unless ``keys`` asks for it again).

    Base images resolve to digests through ``resolve_digest`` (default :func:`registry_digest`), once per
    reference, so the images of one run build from the same digest of a shared base."""

    say = log or logger.info
    resolve = _remembered(resolve_digest or registry_digest)
    if builder not in BUILDERS:
        raise ValueError(f"builder must be one of {BUILDERS}, got {builder!r}")
    index, count = shard
    if not 0 <= index < count:
        raise ValueError(f"shard must be i/n with 0 <= i < n, got {index}/{count}")
    # Absolute, so the images are recorded by paths the map finds (it resolves a relative image path against its own
    # directory, not against the working directory a relative image_map or --out was given in), and so the builds,
    # which run their tools in the build context, find their scratch.
    map_path = Path(image_map).absolute()
    current = ImageMap.load(map_path) if map_path.exists() else None
    plan = plan_images(lock, current)
    if keys and (unknown := sorted(set(keys) - {image.key for image in plan.images})):
        raise ValueError(f"The lock's runnable tasks use no build key {unknown}")
    # Shards split all of the lock's keys, so a shard's share does not move as other shards fill the map.
    assigned = [
        image for position, image in enumerate(sorted(plan.images, key=lambda i: i.key)) if position % count == index
    ]
    wanted = [image for image in assigned if (image.key in keys if keys else not image.present)]
    out = Path(out_dir).absolute() if out_dir is not None else map_path.parent / "images"
    scratch_root = Path(scratch).absolute() if scratch is not None else node_scratch()
    built: dict[str, ImageEntry] = {}
    failures: list[str] = []
    say(f"{len(wanted)} image(s) to build with the {builder} builder (shard {index}/{count})")
    for image in wanted:
        assert image.example is not None
        with _build_lock(map_path, image.key):
            if not keys and (present := _mapped(map_path, image.key)) is not None:
                say(f"{image.key} was built by another build while this one waited: {present}")
                continue
            started = time.monotonic()
            try:
                entry = build_image(
                    Path(cache) / image.example.path,
                    image.key,
                    role=image.role,
                    context=image.context,
                    dockerfile=image.dockerfile,
                    build_args=image.build_args,
                    builder=builder,
                    out_dir=out,
                    scratch=scratch_root,
                    timeout_s=timeout_s,
                    push=push,
                    example=image.example.name,
                    log=say,
                    resolve_digest=resolve,
                )
            except BuildError as exc:
                failures.append(f"{image.key} ({image.example.name}): {exc}")
                say(f"FAILED {image.key} after {time.monotonic() - started:.0f}s: {exc}")
                continue
            update_image_map(map_path, {image.key: entry})
        built[image.key] = entry
        say(f"built {image.key} for {len(image.tasks)} task(s) in {time.monotonic() - started:.0f}s: {entry.image}")
    if failures:
        raise BuildError("; ".join(failures))
    return built


@contextlib.contextmanager
def _build_lock(map_path: Path, key: str) -> Generator[None, None, None]:
    """The builds of one image for one map take turns on a hidden lock file next to the map."""

    lock = map_path.with_name(f".{map_path.name}.{key.removeprefix('sha256:')[:16]}.build.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _mapped(map_path: Path, key: str) -> str | None:
    """The image the map at ``map_path`` has for ``key`` now, when its file (for a local image) is there, whole
    (:func:`~platoon.envs.harbor.images.has_image_file`); else None."""

    if not map_path.is_file():
        return None
    entry = ImageMap.load(map_path).get(key)
    return entry.image if entry is not None and has_image_file(entry) else None


def build_image(
    task_dir: Path,
    key: str,
    *,
    role: str = "environment",
    context: str | None = None,
    dockerfile: str = DOCKERFILE,
    build_args: Mapping[str, str] | None = None,
    builder: str = "oci",
    out_dir: Path,
    scratch: Path,
    timeout_s: float = 3600.0,
    push: str | None = None,
    example: str | None = None,
    log: Callable[[str], None] | None = None,
    resolve_digest: DigestResolver | None = None,
    resolve_config: ConfigResolver | None = None,
) -> ImageEntry:
    """Build the image of ``key`` from the Dockerfile prefix of the cached task at ``task_dir``, its base images
    named by the digests ``resolve_digest`` (default :func:`registry_digest`) resolves them to.  A stack's sidecar
    (``role="service"``) builds its whole Dockerfile, ``dockerfile`` in its build ``context`` (relative to the task),
    with its ``build_args``; the task's own image builds from ``environment/``, a verifier's from ``tests/``.  The
    ``def`` builder records its base image's configuration, which ``resolve_config`` (default
    :func:`registry_config`) gives."""

    context_name = context or ("tests" if role == "verifier" else "environment")
    source = task_dir / context_name
    plan = plan_dockerfile(source / dockerfile, source, whole=role == "service", build_args=build_args)
    if plan.key != key:
        raise BuildError(f"{task_dir}/{context_name}/{dockerfile} plans build key {plan.key}, not {key}")
    if build_args and builder != "docker":
        raise BuildError(
            f"{context_name}/{dockerfile} is built with the build arguments {sorted(build_args)}, which only the "
            "docker builder passes (--builder docker)"
        )
    digests = _base_digests(plan, resolve_digest or registry_digest)
    hexkey = key.removeprefix("sha256:")
    # Private (0700): the oci builder unpacks the image there with its own modes, files any user may write included.
    work = scratch / f"build-{hexkey[:16]}-{uuid.uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / f"{hexkey}.{builder}.log"
    started = time.monotonic()
    runner = _Runner(log_path, timeout_s=timeout_s, env=_build_env(scratch))
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        work.mkdir(mode=0o700)
        build_context = work / "context"
        directives = list(plan.directives)
        if builder == "oci":
            directives, frontend = _oci_directives(plan)
            if frontend is not None:
                runner.note(
                    f"left out {frontend!r}: SingularityCE's BuildKit cannot load a frontend image, and builds the "
                    "Dockerfile with its built-in frontend, which implements that syntax"
                )
        _stage_context(plan, source, build_context, _pinned_dockerfile(plan, digests, directives))
        if builder == "oci":
            sif, config = _build_oci(runner, build_context, work, out_dir / f"{hexkey}.sif")
            entry = _entry_from_config(str(sif), config, builder="oci", sha256=file_sha256(sif))
        elif builder == "def":
            sif, workdir = _build_def(runner, plan, build_context, work, out_dir / f"{hexkey}.sif", digests)
            env = _sif_environment(runner, sif)
            entry = ImageEntry(image=str(sif), sha256=file_sha256(sif), builder="def", env=env, workdir=workdir)
            entry = _with_base_config(
                entry, plan, digests, resolve_config or registry_config, required=role == "service"
            )
        else:
            entry = _build_docker(runner, build_context, hexkey, push=push, build_args=build_args or {})
    except BuildError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise BuildError(f"{type(exc).__name__}: {exc} (log: {log_path})") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)
    receipt = {
        "builder_version": runner.version(builder),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - started, 1),
        "log": str(log_path),
        "size_bytes": Path(entry.image).stat().st_size if Path(entry.image).is_file() else None,
        "base": list(plan.base_images),  # as the Dockerfile names them, ARG defaults expanded
        "base_digests": digests,  # the digest each base image (not an earlier stage or scratch) was built from, or null
        "task": example or task_dir.name,
        "role": role,
    }
    if builder == "oci":
        receipt["builder_revision"] = OCI_BUILDER_REVISION
    return replace(entry, receipt=receipt)


# --- base image digests --------------------------------------------------------------------------------------

# The manifest types a digest request accepts, multi-platform indexes first: the digest of an index is what a pull
# of the tag starts from on every platform.
_MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}|sha512:[0-9a-f]{128}")
_REPOSITORY = re.compile(r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*(?:/[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*)*")
_TAG = re.compile(r"\w[\w.-]{0,127}")
_CHALLENGE_PARAMETER = re.compile(r'([A-Za-z][\w-]*)=(?:"((?:[^"\\]|\\.)*)"|([^\s,"]*))')
_REGISTRY_TIMEOUT_S = 30.0


class DigestError(RuntimeError):
    """A base image whose registry did not give its digest."""


def registry_digest(
    reference: str, *, timeout_s: float = _REGISTRY_TIMEOUT_S, urlopen: Callable[..., Any] | None = None
) -> str:
    """The digest the registry of an image reference serves for it: the ``Docker-Content-Digest`` of its manifest
    (``HEAD``, else ``GET``), asked anonymously, with the token the registry's ``Bearer`` challenge points to when
    it asks for one. A reference pinned by digest needs no request.

    ``urlopen`` (default :func:`urllib.request.urlopen`) makes the requests. Raises :class:`DigestError` when the
    registry cannot be reached or gives no digest.
    """

    host, repository, tag, digest = _parse_reference(reference)
    if digest is not None:
        return digest
    url = f"https://{host}/v2/{repository}/manifests/{tag}"
    client = _RegistryClient(urlopen or urllib.request.urlopen, repository, timeout_s)
    try:
        try:
            digest = client.manifest_digest(url, "HEAD")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise
            digest = None  # a registry that does not answer HEAD: ask with GET
        if digest is None:
            digest = client.manifest_digest(url, "GET")
    except urllib.error.HTTPError as exc:
        raise DigestError(f"{url} answered HTTP {exc.code}") from None
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise DigestError(f"{url}: {exc}") from None
    if digest is None or not _DIGEST.fullmatch(digest):
        raise DigestError(f"{url} gave no digest")
    return digest


class _RegistryClient:
    """Requests to one repository of a registry, with the anonymous token it asks for, once it asks."""

    def __init__(self, urlopen: Callable[..., Any], repository: str, timeout_s: float) -> None:
        self._urlopen = urlopen
        self._repository = repository
        self._timeout_s = timeout_s
        self._token: str | None = None

    def manifest_digest(self, url: str, method: str) -> str | None:
        """The ``Docker-Content-Digest`` of the manifest at ``url``; for a ``GET`` without one, the SHA-256 of the
        manifest it serves, which is the digest by definition."""

        with self._open(url, method) as response:
            digest = response.headers.get("Docker-Content-Digest")
            if digest or method == "HEAD" or response.headers.get_content_type() not in _MANIFEST_TYPES:
                return digest or None
            return "sha256:" + hashlib.sha256(response.read()).hexdigest()

    def json(self, url: str) -> tuple[Any, str]:
        """The JSON document at ``url`` (a manifest or a blob), and its digest: the ``Docker-Content-Digest`` the
        registry gives, else the SHA-256 of what it served."""

        with self._open(url, "GET") as response:
            body = response.read()
            digest = response.headers.get("Docker-Content-Digest") or "sha256:" + hashlib.sha256(body).hexdigest()
        return json.loads(body), digest

    def _open(self, url: str, method: str) -> Any:
        request = urllib.request.Request(url, method=method, headers={"Accept": ", ".join(_MANIFEST_TYPES)})
        if self._token is not None:  # never sent on to where a redirect leads, such as a blob's storage
            request.add_unredirected_header("Authorization", f"Bearer {self._token}")
        try:
            return self._urlopen(request, timeout=self._timeout_s)
        except urllib.error.HTTPError as exc:
            challenge = (exc.headers.get("WWW-Authenticate") if exc.headers is not None else None) or ""
            if exc.code != 401 or self._token is not None or not challenge.lower().startswith("bearer "):
                raise
            exc.close()
        self._token = self._anonymous_token(challenge)
        return self._open(url, method)

    def _anonymous_token(self, challenge: str) -> str:
        """A pull token from the ``realm`` of a ``Bearer realm=...,service=...,scope=...`` challenge."""

        parameters: dict[str, str] = {}
        for match in _CHALLENGE_PARAMETER.finditer(challenge[len("bearer ") :]):
            quoted, bare = match.group(2), match.group(3)
            parameters[match.group(1).lower()] = re.sub(r"\\(.)", r"\1", quoted) if quoted is not None else bare
        realm = parameters.get("realm") or ""
        if not realm.startswith(("https://", "http://")):
            raise DigestError(f"the registry's token realm {realm!r} is not an HTTP URL")
        query = {
            "service": parameters.get("service"),
            "scope": parameters.get("scope") or f"repository:{self._repository}:pull",
        }
        query_string = urllib.parse.urlencode({name: value for name, value in query.items() if value})
        url = realm + ("&" if "?" in realm else "?") + query_string
        try:
            with self._urlopen(urllib.request.Request(url), timeout=self._timeout_s) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise DigestError(f"{realm} answered HTTP {exc.code} to a token request") from None
        token = (data.get("token") or data.get("access_token")) if isinstance(data, dict) else None
        if not isinstance(token, str) or not token:
            raise DigestError(f"{realm} gave no token")
        return token


def _parse_reference(reference: str) -> tuple[str, str, str, str | None]:
    """An image reference (``[registry/]repository[:tag][@digest]``) as the registry host to ask, the repository,
    the tag and the digest, with Docker's defaults: no registry is Docker Hub (``registry-1.docker.io``, with
    ``library/`` before a single name), and no tag is ``latest``."""

    name, at, digest = reference.partition("@")
    if at and not _DIGEST.fullmatch(digest):
        raise DigestError(f"{reference!r} has a malformed digest")
    first, slash, rest = name.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost" or first != first.lower()):
        host, repository = first, rest
    else:
        host, repository = "docker.io", name
    tag = "latest"
    if ":" in repository:
        repository, _, tag = repository.rpartition(":")
    if host in ("docker.io", "index.docker.io"):
        host = "registry-1.docker.io"
        if "/" not in repository:
            repository = f"library/{repository}"
    if not _REPOSITORY.fullmatch(repository) or not _TAG.fullmatch(tag):
        raise DigestError(f"{reference!r} is not an image reference")
    return host, repository, tag, digest or None


def _pinned_reference(reference: str, digest: str) -> str:
    """``reference`` by ``digest``, without its tag (some builders refuse a reference with both):
    ``python:3.12-slim`` becomes ``python@sha256:...``."""

    head, slash, last = reference.partition("@")[0].rpartition("/")
    return f"{head}{slash}{last.partition(':')[0]}@{digest}"


# --- registry images: their configurations, and the pins of the ones sidecars run ------------------------------

# The architecture an image index names for this host's machine (platform.machine()).
_ARCHITECTURES = {"x86_64": "amd64", "aarch64": "arm64"}


def registry_config(
    reference: str,
    *,
    timeout_s: float = _REGISTRY_TIMEOUT_S,
    urlopen: Callable[..., Any] | None = None,
    machine: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """The digest the registry of an image reference serves for it, as :func:`registry_digest` gives it (an image
    index's, for an image of several platforms), and the configuration of its image for Linux on ``machine`` (default:
    this host's): the image's config blob, whose ``config`` holds the ``Env``, ``Entrypoint``, ``Cmd``,
    ``WorkingDir``, ``User``, ``Healthcheck``, ... that its containers start with.  ``urlopen`` makes the requests, as
    for :func:`registry_digest`.  Raises :class:`DigestError` when the registry cannot be reached or does not serve
    them, or has no image for this platform."""

    host, repository, tag, pinned = _parse_reference(reference)
    client = _RegistryClient(urlopen or urllib.request.urlopen, repository, timeout_s)
    base = f"https://{host}/v2/{repository}"
    try:
        manifest, digest = client.json(f"{base}/manifests/{pinned or tag}")
        if isinstance(manifest, dict) and isinstance(manifest.get("manifests"), list):  # an index of platforms
            manifest, _ = client.json(f"{base}/manifests/{_platform_manifest(manifest, reference, machine)}")
        blob = (manifest.get("config") or {}).get("digest") if isinstance(manifest, dict) else None
        if not isinstance(blob, str) or not _DIGEST.fullmatch(blob):
            raise DigestError(f"{base}/manifests/{pinned or tag} names no image configuration")
        config, _ = client.json(f"{base}/blobs/{blob}")
    except urllib.error.HTTPError as exc:
        raise DigestError(f"{exc.filename or base} answered HTTP {exc.code}") from None
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise DigestError(f"{base}: {exc}") from None
    if not isinstance(config, dict) or not isinstance(config.get("config") or {}, dict):
        raise DigestError(f"{base}/blobs/{blob} is not an image configuration")
    return pinned or digest, config


def _platform_manifest(index: Mapping[str, Any], reference: str, machine: str | None) -> str:
    """The digest of the manifest that an image index lists for Linux on ``machine``'s architecture."""

    machine = machine or platform.machine()
    architecture = _ARCHITECTURES.get(machine, machine)
    for entry in index.get("manifests") or ():
        found = (entry.get("platform") or {}) if isinstance(entry, dict) else {}
        if found.get("os") == "linux" and found.get("architecture") == architecture:
            digest = entry.get("digest")
            if isinstance(digest, str) and _DIGEST.fullmatch(digest):
                return digest
    raise DigestError(f"{reference} has no image for linux/{architecture}")


def pinned_entry(reference: str, resolve: ConfigResolver | None = None) -> ImageEntry:
    """The image map's entry of a registry image that sidecars run: by the digest its registry serves for
    ``reference``, with its image's configuration (``resolve``, default :func:`registry_config`)."""

    digest, config = (resolve or registry_config)(reference)
    image = docker_image_reference(_pinned_reference(reference, digest))
    entry = _entry_from_config(image, config, builder="pulled", sha256=None)
    pinned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return replace(entry, receipt={"reference": reference, "digest": digest, "pinned_at": pinned_at})


def pin_images(
    lock: DatasetLock,
    *,
    image_map: str | Path,
    references: Sequence[str] | None = None,
    log: Callable[[str], None] | None = None,
    resolve: ConfigResolver | None = None,
) -> dict[str, ImageEntry]:
    """Pin the registry images that the lock's runnable stacks' sidecars run and ``image_map`` has not pinned (or
    ``references``, again), adding each to the map as soon as it is pinned (:func:`pinned_entry`).  Returns the new
    entries by reference; raises :class:`DigestError` after trying them all if any failed."""

    say = log or logger.info
    map_path = Path(image_map).absolute()
    plan = plan_images(lock, ImageMap.load(map_path) if map_path.exists() else None)
    if references and (unknown := sorted(set(references) - {pull.reference for pull in plan.pulls})):
        raise ValueError(f"The lock's runnable stacks run no registry image {unknown}")
    wanted = [pull for pull in plan.pulls if (pull.reference in references if references else not pull.pinned)]
    pinned: dict[str, ImageEntry] = {}
    failures: list[str] = []
    for pull in wanted:
        try:
            entry = pinned_entry(pull.reference, resolve)
        except DigestError as exc:
            failures.append(f"{pull.reference}: {exc}")
            continue
        update_image_map(map_path, {pulled_key(pull.reference): entry})
        pinned[pull.reference] = entry
        say(f"pinned {pull.reference} for {len(pull.tasks)} task(s): {entry.image}")
    if failures:
        raise DigestError("; ".join(failures))
    return pinned


def _from_images(plan: DockerfilePlan) -> dict[int, str]:
    """The base image of each ``FROM`` of the prefix that names one (not an earlier stage, nor ``scratch``), by the
    instruction's position in the prefix."""

    images: dict[int, str] = {}
    stages: set[str] = set()
    bases = iter(plan.base_images)  # one per FROM, with its ARG defaults expanded
    for position, instruction in enumerate(plan.prefix):
        if instruction.keyword != "FROM":
            continue
        _, operands = instruction.flags_and_operands()
        if not operands:
            continue
        base = next(bases, "")
        if base and base.lower() not in stages and base.lower() != "scratch":
            images[position] = base
        if len(operands) >= 3 and operands[1].lower() == "as":
            stages.add(operands[2].lower())
    return images


def _base_digests(plan: DockerfilePlan, resolve: DigestResolver) -> dict[str, str | None]:
    """The digest ``resolve`` finds for each base image of ``plan``: None, with a warning, where it finds none."""

    digests: dict[str, str | None] = {}
    for reference in dict.fromkeys(_from_images(plan).values()):
        try:
            digest, why = resolve(reference), "none found"
        except DigestError as exc:
            digest, why = None, str(exc)
        if digest is None:
            logger.warning(
                "Cannot resolve base image %s to a digest (%s); building from it as written, with a null digest in "
                "the receipt",
                reference,
                why,
            )
        digests[reference] = digest
    return digests


def _pinned_dockerfile(
    plan: DockerfilePlan, digests: Mapping[str, str | None], directives: Sequence[str] | None = None
) -> str:
    """``plan``'s prefix as a Dockerfile, its ``FROM`` steps naming their base images by the digests resolved for
    them, under ``directives`` (default: ``plan``'s parser directives)."""

    images = _from_images(plan)
    lines = list(plan.directives if directives is None else directives)
    for position, instruction in enumerate(plan.prefix):
        digest = digests.get(images[position]) if position in images else None
        if digest is None:
            lines.append(instruction.text)
            continue
        flags, operands = instruction.flags_and_operands()
        words = [f"--{name}={value}" if value else f"--{name}" for name, value in flags.items()]
        lines.append(" ".join(["FROM", *words, _pinned_reference(images[position], digest), *operands[1:]]))
    return "\n".join(lines) + "\n"


def _oci_directives(plan: DockerfilePlan) -> tuple[list[str], str | None]:
    """``plan``'s parser directives as the oci builder hands them to SingularityCE's BuildKit, which has no gateway
    frontend to load the image a ``# syntax=`` directive names (it fails with "invalid frontend: gateway.v0"), and
    the directive it leaves out: one that names a 1.x release of Docker's own Dockerfile frontend, which BuildKit's
    built-in frontend implements. Raises :class:`BuildError` for a directive that names another frontend."""

    kept: list[str] = []
    left_out: str | None = None
    for directive in plan.directives:
        syntax = _SYNTAX_DIRECTIVE.fullmatch(directive.strip())
        if syntax is None:
            kept.append(directive)
        elif _DOCKERFILE_FRONTEND.fullmatch(syntax.group("image")):
            left_out = directive
        else:
            raise BuildError(
                f"{directive}: the Dockerfile asks for the frontend {syntax.group('image')}, which the oci builder "
                "cannot load (SingularityCE's BuildKit builds with its built-in frontend, docker/dockerfile:1 without "
                "labs features); build this image with the docker builder (--builder docker)"
            )
    return kept, left_out


def _remembered(resolve: DigestResolver) -> DigestResolver:
    """``resolve``, asked once per reference."""

    answers: dict[str, str | None | DigestError] = {}

    def lookup(reference: str) -> str | None:
        if reference not in answers:
            try:
                answers[reference] = resolve(reference)
            except DigestError as exc:
                answers[reference] = exc
        answer = answers[reference]
        if isinstance(answer, DigestError):
            raise answer
        return answer

    return lookup


# --- builders ------------------------------------------------------------------------------------------------


class _Runner:
    """Runs build commands, appending their output to one log file."""

    def __init__(self, log_path: Path, *, timeout_s: float, env: Mapping[str, str]) -> None:
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.env = dict(env)
        log_path.write_text("")

    def run(self, argv: Sequence[str], *, cwd: Path | None = None, stdout: Path | None = None, umask: int = -1) -> str:
        """Run ``argv`` (in ``cwd``, with ``umask`` when it is not -1), logging it and its output; its stdout, unless
        ``stdout`` names a file to write it to.  ``BuildError`` when it fails or times out."""

        with self.log_path.open("a") as log:
            log.write(f"$ {shlex.join(argv)}\n")
            log.flush()
            sink = stdout.open("wb") if stdout is not None else None
            try:
                result = subprocess.run(
                    list(argv),
                    cwd=cwd,
                    env=self.env,
                    stdout=sink if sink is not None else subprocess.PIPE,
                    stderr=log if sink is not None else subprocess.STDOUT,
                    timeout=self.timeout_s,
                    check=False,
                    umask=umask,
                )
            except subprocess.TimeoutExpired:
                raise BuildError(f"{argv[0]} timed out after {self.timeout_s:.0f}s (log: {self.log_path})") from None
            finally:
                if sink is not None:
                    sink.close()
            output = "" if stdout is not None else (result.stdout or b"").decode(errors="replace")
            log.write(output)
            if result.returncode != 0:
                tail = output.strip().splitlines()[-5:] if output else []
                raise BuildError(
                    f"{shlex.join(argv[:3])} ... exited {result.returncode}: {' | '.join(tail)} (log: {self.log_path})"
                )
            return output

    def note(self, text: str) -> None:
        """Say ``text`` in the log, and to this module's logger."""

        with self.log_path.open("a") as log:
            log.write(f"platoon: {text}\n")
        logger.info("%s (log: %s)", text, self.log_path)

    def version(self, builder: str) -> str | None:
        binary = _docker_binary() if builder == "docker" else _singularity_binary()
        try:
            result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60, env=self.env)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() or None


def _build_oci(runner: _Runner, context: Path, work: Path, target: Path) -> tuple[Path, dict[str, Any]]:
    singularity = _singularity_binary()
    oci = work / "image.oci.sif"
    runner.run([singularity, "build", "--oci", str(oci), "Dockerfile"], cwd=context)
    listing = runner.run([singularity, "sif", "list", str(oci)])
    layers: list[Path] = []
    config: dict[str, Any] | None = None
    for descriptor, kind in _sif_descriptors(listing):
        if kind != "OCI.Blob":
            continue
        blob = work / f"blob-{descriptor}"
        runner.run([singularity, "sif", "dump", str(descriptor), str(oci)], stdout=blob)
        with blob.open("rb") as handle:
            head = handle.read(4)
        if head == _SQUASHFS_MAGIC:
            layers.append(blob)
        elif blob.stat().st_size < 16 * 1024 * 1024:
            try:
                data = json.loads(blob.read_text())
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "rootfs" in data and isinstance(data.get("config"), dict):
                config = data
    if len(layers) != 1:
        raise BuildError(f"the OCI-SIF holds {len(layers)} squashfs layers, expected 1 (log: {runner.log_path})")
    if config is None:
        raise BuildError(f"the OCI-SIF holds no image config (log: {runner.log_path})")
    rootfs = work / "rootfs"
    # Run by anyone but root, unsquashfs creates each file with the image's mode through the process's umask and never
    # sets it after: under a host umask such as 027, the image's programs and libraries would lose the read and execute
    # bits it gives other users, which a sandbox that switches users (a verifier run as root) needs.
    runner.run([_tool("unsquashfs"), "-no-xattrs", "-d", str(rootfs), str(layers[0])], umask=0)
    layers[0].unlink()
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        runner.run([singularity, "build", str(partial), str(rootfs)])
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target, config


def _build_def(
    runner: _Runner,
    plan: DockerfilePlan,
    context: Path,
    work: Path,
    target: Path,
    digests: Mapping[str, str | None],
) -> tuple[Path, str | None]:
    definition, workdir = definition_file(plan, context, digests=digests)
    path = context / "image.def"
    path.write_text(definition)
    # Under --fakeroot the build runs as a subordinate uid, which may not write to shared storage: build in the
    # node-local work directory, then copy the image into place as the caller.
    built = work / "image.sif"
    runner.run([_singularity_binary(), "build", "--fakeroot", str(built), "image.def"], cwd=context)
    partial = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.partial")
    try:
        shutil.copyfile(built, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target, workdir


# Variables the runtime sets in every container, not the image's own.
_RUNTIME_VARIABLES = frozenset(
    {"PWD", "OLDPWD", "SHLVL", "_", "PS1", "PROMPT_COMMAND", "HOME", "USER", "LOGNAME", "TERM", "HOSTNAME"}
)


def _sif_environment(runner: _Runner, sif: Path) -> dict[str, str]:
    """The environment a SIF's own scripts set (base image and ``%environment``), for the image map: deferred
    ``ENV`` steps such as ``PATH=/opt/bin:$PATH`` expand against it."""

    try:
        output = runner.run([_singularity_binary(), "exec", "--cleanenv", str(sif), "env", "-0"])
    except BuildError:
        logger.warning("Cannot read the environment of %s; its map entry has none (log: %s)", sif, runner.log_path)
        return {}
    env = {}
    for item in output.split("\0"):
        name, separator, value = item.partition("=")
        if not separator or name in _RUNTIME_VARIABLES or name.startswith(("SINGULARITY", "APPTAINER")):
            continue
        env[name] = value
    if env.get("LD_LIBRARY_PATH") == "/.singularity.d/libs":
        del env["LD_LIBRARY_PATH"]
    return env


def _with_base_config(
    entry: ImageEntry,
    plan: DockerfilePlan,
    digests: Mapping[str, str | None],
    resolve: ConfigResolver,
    *,
    required: bool,
) -> ImageEntry:
    """A ``def`` build's ``entry`` with the image configuration a definition file keeps none of: its base image's
    (from the registry, by the digest it was built from), under what the built steps set (``CMD``, ``ENTRYPOINT``,
    ``USER``, ``HEALTHCHECK``, ``EXPOSE``, ``VOLUME``, ``STOPSIGNAL``), with the build's own environment and working
    directory.  Without the registry, a ``BuildError`` where the configuration is ``required`` (a sidecar's image, whose
    command may be its base image's), else it records none of them, with a warning."""

    base = next(iter(_from_images(plan).values()), None)  # a definition file has one FROM
    if base is None:
        return entry
    digest = digests.get(base)
    try:
        _, config = resolve(_pinned_reference(base, digest) if digest else base)
    except DigestError as exc:
        if required:
            raise BuildError(
                f"Cannot read the configuration of {base}, the base of {entry.image}, from its registry: {exc}"
            ) from None
        logger.warning(
            "Cannot read the configuration of %s, the base of %s, from its registry (%s)", base, entry.image, exc
        )
        return entry
    inherited = _entry_from_config(entry.image, config, builder="def", sha256=entry.sha256)
    own = image_config(list(variables_in_scope(plan.prefix)))
    return replace(
        inherited,
        env=entry.env,
        workdir=entry.workdir or inherited.workdir,
        entrypoint=tuple(own["entrypoint"]) if "entrypoint" in own else inherited.entrypoint,
        cmd=tuple(own["cmd"]) if "cmd" in own else inherited.cmd,
        user=own.get("user", inherited.user),
        healthcheck=own.get("healthcheck", inherited.healthcheck),
        exposed_ports=tuple(sorted({*inherited.exposed_ports, *own.get("expose", ())})),
        volumes=tuple(sorted({*inherited.volumes, *own.get("volumes", ())})),
        stop_signal=own.get("stop_signal", inherited.stop_signal),
    )


def _build_docker(
    runner: _Runner, context: Path, hexkey: str, *, push: str | None, build_args: Mapping[str, str]
) -> ImageEntry:
    docker = _docker_binary()
    tag = f"{push}:{hexkey[:32]}" if push else f"platoon-harbor:{hexkey[:32]}"
    arguments = [f"--build-arg={name}={value}" for name, value in build_args.items()]
    runner.run([docker, "build", "-t", tag, *arguments, "-f", "Dockerfile", "."], cwd=context)
    config = json.loads(runner.run([docker, "image", "inspect", "--format", "{{json .Config}}", tag]) or "{}")
    image = tag
    if push:
        runner.run([docker, "push", tag])
        digests = json.loads(runner.run([docker, "image", "inspect", "--format", "{{json .RepoDigests}}", tag]) or "[]")
        if not digests:
            raise BuildError(f"{tag} has no repository digest after the push (log: {runner.log_path})")
        image = f"docker://{digests[0]}"
    return _entry_from_config(image, {"config": config}, builder="docker", sha256=None)


def definition_file(
    plan: DockerfilePlan, context: Path, *, digests: Mapping[str, str | None] | None = None
) -> tuple[str, str | None]:
    """A ``Bootstrap: docker`` definition file equivalent to ``plan``'s prefix (its build context at ``context``),
    and its final ``WORKDIR``. The base image is named by its digest where ``digests`` has one.

    ``%post`` runs the steps as a Docker build does:

    - each ``RUN`` runs in a subshell of its own that starts in the current ``WORKDIR`` (``/`` before one), the
      shell form through ``/bin/sh -c``, so a ``cd`` or a variable it sets does not reach the next ``RUN``;
    - ``ARG`` values are exported for the ``RUN`` steps after them, unless an ``ENV`` of the same name is set;
    - ``ENV`` values, exported in ``%post`` and in ``%environment``, are literal but for their variable references:
      the stage's ``ARG`` values are substituted (they do not reach the image's runtime environment), and other
      references, such as ``$PATH`` to the base image's environment, are left for the shell to expand;
    - ``COPY`` sources go in through ``%files`` to a staging directory and are copied into place in their
      Dockerfile order, with ``COPY``'s directory semantics; the staging directory goes after the last of them.
    """

    variables: dict[str, str] = {}
    arguments: dict[str, str] = {}  # the stage's ARG values, as its RUN steps see them
    env_names: set[str] = set()  # the names the stage's ENV steps set
    base: str | None = None
    files: list[str] = []
    post: list[str] = ["set -e"]
    environment: list[str] = []
    labels: list[str] = []
    workdir: str | None = None
    staged = 0
    copied = 0  # the %post lines up to the last COPY or ADD, which the staged files must outlive
    for instruction, in_scope in variables_in_scope(plan.prefix):
        keyword = instruction.keyword
        flags, operands = instruction.flags_and_operands()
        if instruction.heredocs:
            raise BuildError(f"line {instruction.line}: here-documents need the oci builder")
        if keyword == "FROM":
            if base is not None:
                raise BuildError(f"line {instruction.line}: multi-stage builds need the oci builder")
            if flags:
                raise BuildError(f"line {instruction.line}: FROM flags need the oci builder")
            base = expand(operands[0], variables)
        elif keyword == "ARG":
            for name, value in parse_assignments(instruction.arguments, "ARG"):
                if value is None:
                    raise BuildError(
                        f"line {instruction.line}: ARG {name} has no default; the def builder takes no build args"
                    )
                variables[name] = value
                if base is not None and name not in env_names:  # an ENV of the same name wins, as in Docker
                    arguments[name] = expand(value, variables)
                    post.append(f"export {name}={shlex.quote(arguments[name])}")
        elif base is None:
            raise BuildError(f"line {instruction.line}: {keyword} before FROM")
        elif keyword == "RUN":
            if flags:
                raise BuildError(f"line {instruction.line}: RUN flags need the oci builder")
            command = _exec_form(instruction.arguments)
            if command:
                run = shlex.join(command)
            elif command is None and instruction.arguments:
                run = f"/bin/sh -c {shlex.quote(instruction.arguments)}"
            else:
                raise BuildError(f"line {instruction.line}: RUN needs a command")
            post.append(f"( cd {shlex.quote(workdir or '/')} && {run} )")
        elif keyword == "ENV":
            # One export per ENV step: its values see the variables as they were before it, as in Docker.
            pairs = parse_assignments(instruction.arguments, "ENV")
            known = {name: value for name, value in arguments.items() if name not in env_names}
            export = "export " + " ".join(f"{name}={shell_word(value, known)}" for name, value in pairs)
            post.append(export)
            environment.append(export)
            env_names.update(name for name, _ in pairs)
        elif keyword == "WORKDIR":
            path = instruction.arguments.strip()
            workdir = path if path.startswith("/") else f"{workdir or ''}/{path}"
            post.append(f"mkdir -p {shlex.quote(workdir)} && cd {shlex.quote(workdir)}")
        elif keyword in ("COPY", "ADD"):
            for source in operands[:-1]:
                if remote := remote_source(source, in_scope):
                    shown = expand(source, in_scope) if variables_defined(source, in_scope) else source
                    what = "the git repository" if remote == "git" else "the URL"
                    if keyword == "COPY":  # no builder can: Docker refuses it too
                        raise BuildError(f"line {instruction.line}: COPY cannot fetch {what} {shown}; only ADD can")
                    raise BuildError(
                        f"line {instruction.line}: {keyword} fetches {what} {shown}, which the def builder cannot "
                        "do: build this image with the oci builder (--builder oci), which runs the Dockerfile with "
                        "BuildKit as Docker does"
                    )
            if flags:
                raise BuildError(f"line {instruction.line}: {keyword} flags need the oci builder")
            *sources, destination = operands
            destination = expand(destination, variables)
            many = len(sources) > 1 or destination.endswith("/")
            for source in sources:
                if "://" in source or any(char in source for char in "*?[$"):
                    raise BuildError(f"line {instruction.line}: {keyword} {source} needs the oci builder")
                local = context / source
                if keyword == "ADD" and local.is_file() and _looks_archive(local):
                    raise BuildError(f"line {instruction.line}: ADD of an archive needs the oci builder")
                staging = f"/.platoon-files/{staged}"
                staged += 1
                files.append(f"    {shlex.quote(source.rstrip('/') or '.')} {staging}")
                quoted = shlex.quote(destination)
                if local.is_dir():
                    post.append(f"mkdir -p {quoted} && cp -a {staging}/. {quoted}/")
                elif many:
                    post.append(f"mkdir -p {quoted} && cp -a {staging} {quoted}/{shlex.quote(local.name)}")
                else:
                    post.append(f'mkdir -p "$(dirname {quoted})" && cp -a {staging} {quoted}')
            copied = len(post)
        elif keyword == "LABEL":
            labels += [f"    {name} {value}" for name, value in parse_assignments(instruction.arguments, "LABEL")]
        elif keyword in ("USER", "SHELL"):
            raise BuildError(f"line {instruction.line}: {keyword} needs the oci builder")
        # EXPOSE, CMD, ENTRYPOINT, HEALTHCHECK, STOPSIGNAL, VOLUME: image metadata a sandbox does not use
    if base is None:
        raise BuildError("the Dockerfile prefix has no FROM")
    if staged:
        post.insert(copied, "rm -rf /.platoon-files")  # once every COPY and ADD has read its files
    digest = (digests or {}).get(base)
    sections = ["Bootstrap: docker", f"From: {_pinned_reference(base, digest) if digest else base}", ""]
    if files:
        sections += ["%files", *files, ""]
    sections += ["%post", *(f"    {line}" for line in post), ""]
    if environment:
        sections += ["%environment", *(f"    {line}" for line in environment), ""]
    if labels:
        sections += ["%labels", *labels, ""]
    return "\n".join(sections), workdir


def _entry_from_config(image: str, blob: Mapping[str, Any], *, builder: str, sha256: str | None) -> ImageEntry:
    config = blob.get("config") or {}
    env = dict(item.split("=", 1) for item in config.get("Env") or [] if "=" in item)
    return ImageEntry(
        image=image,
        sha256=sha256,
        builder=builder,
        env=env,
        workdir=config.get("WorkingDir") or None,
        user=config.get("User") or None,
        entrypoint=tuple(config["Entrypoint"]) if config.get("Entrypoint") else None,
        cmd=tuple(config["Cmd"]) if config.get("Cmd") else None,
        healthcheck=_healthcheck_of(config.get("Healthcheck")),
        exposed_ports=tuple(sorted(config.get("ExposedPorts") or ())),
        volumes=tuple(sorted(config.get("Volumes") or ())),
        stop_signal=config.get("StopSignal") or None,
    )


def _healthcheck_of(healthcheck: Any) -> dict[str, Any] | None:
    """An image configuration's ``Healthcheck`` as compose writes a healthcheck, durations (nanoseconds there) in
    seconds; None for none."""

    if not isinstance(healthcheck, Mapping) or not healthcheck.get("Test"):
        return None
    data: dict[str, Any] = {"test": list(healthcheck["Test"])}
    for name, key in (("interval", "Interval"), ("timeout", "Timeout"), ("start_period", "StartPeriod")):
        if healthcheck.get(key):
            data[name] = healthcheck[key] / 1e9
    if healthcheck.get("StartInterval"):
        data["start_interval"] = healthcheck["StartInterval"] / 1e9
    if healthcheck.get("Retries"):
        data["retries"] = int(healthcheck["Retries"])
    return data


def _exec_form(command: str) -> list[str] | None:
    """A ``RUN`` command's exec form (a JSON array of strings), or None for the shell form."""

    if not command.startswith("["):
        return None
    try:
        words = json.loads(command)
    except json.JSONDecodeError:
        return None  # such as `[ -f x ] && ...`: the shell form, as in Docker
    return words if isinstance(words, list) and all(isinstance(word, str) for word in words) else None


def _stage_context(plan: DockerfilePlan, context: Path, target: Path, dockerfile: str) -> None:
    """Copy what the prefix reads (or the whole context) into ``target``, and ``dockerfile`` (the prefix) as its
    ``Dockerfile``."""

    target.mkdir(parents=True)
    if "." in plan.context_files:
        shutil.copytree(context, target, dirs_exist_ok=True, symlinks=True)
    else:
        for relative in plan.context_files:
            source = context / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True)
            else:
                shutil.copy2(source, destination)
    (target / "Dockerfile").write_text(dockerfile)


def _sif_descriptors(listing: str) -> list[tuple[int, str]]:
    descriptors = []
    for line in listing.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) >= 5 and cells[0].isdigit():
            descriptors.append((int(cells[0]), cells[-1]))
    return descriptors


def _build_env(scratch: Path) -> dict[str, str]:
    """The environment of build commands: Singularity's caches and BuildKit's runtime directories in scratch
    (``SINGULARITY_CACHEDIR``/``SINGULARITY_TMPDIR`` as set, to share base layers across builds; the runtime
    directory under ``/tmp`` when scratch is too deep for BuildKit's socket), and the system directories that hold
    ``unsquashfs``."""

    root = scratch / "platoon-images"
    env = dict(os.environ)
    for name, default in (("SINGULARITY_CACHEDIR", "cache"), ("SINGULARITY_TMPDIR", "tmp")):
        env.setdefault(name, str(root / default))
    for name, directory in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data")):
        env[name] = str(root / directory)
    env["XDG_RUNTIME_DIR"] = str(_runtime_dir(root / "run"))
    for name in ("SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR", "XDG_RUNTIME_DIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_RUNTIME_DIR"]).chmod(0o700)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), "/usr/sbin", "/sbin"])
    return env


def _runtime_dir(preferred: Path) -> Path:
    """BuildKit's runtime directory: ``preferred``, unless the daemon's socket below it would not fit in a Unix
    socket address (a deep scratch directory); then a private directory of this user under ``/tmp``."""

    if len(os.fsencode(preferred)) + len(_BUILDKIT_SOCKET) <= _SOCKET_PATH_MAX:
        return preferred
    fallback = Path("/tmp") / f"platoon-images-{os.getuid()}"
    fallback.mkdir(mode=0o700, exist_ok=True)
    info = fallback.lstat()
    if not fallback.is_dir() or fallback.is_symlink() or info.st_uid != os.getuid():
        raise BuildError(f"{fallback} is not a directory of this user, so BuildKit cannot use it for its socket")
    return fallback


def _singularity_binary() -> str:
    return (
        os.environ.get("PLATOON_SINGULARITY")
        or shutil.which("singularity")
        or shutil.which("apptainer")
        or "singularity"
    )


def _docker_binary() -> str:
    return os.environ.get("PLATOON_DOCKER") or "docker"


def provider_builder(provider: str) -> str | None:
    """The builder of the images that the sandbox provider ``provider`` runs (:data:`PROVIDER_BUILDERS`): for
    ``singularity`` ``oci``, but ``def`` where its runtime is Apptainer, which has no BuildKit; None for a provider
    that pulls its images from a registry (``enroot``) or runs none (``process``)."""

    builder = PROVIDER_BUILDERS.get(provider)
    return "def" if builder == "oci" and _runtime_is_apptainer() else builder


@functools.cache
def _runtime_is_apptainer() -> bool:
    """Whether the Singularity runtime here is Apptainer (``apptainer version ...``), which builds no OCI image."""

    try:
        result = subprocess.run([_singularity_binary(), "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip().lower().startswith("apptainer")


def _tool(name: str) -> str:
    return shutil.which(name, path=os.pathsep.join([os.environ.get("PATH", ""), "/usr/sbin", "/sbin"])) or name


def _looks_archive(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(512)
    return head[:2] == b"\x1f\x8b" or head[:3] == b"BZh" or head[:6] == b"\xfd7zXZ\x00" or head[257:262] == b"ustar"
