"""Owner manifests: the paths an image gives to owners other than root, which a sandbox that loses them gets back.

hpc-sandbox extracts every image as one user, and SingularityCE's conversion of a registry image records every file as
root's, so a sandbox that switches users (``agent_user: honor``) would give none of its own to a user that the image
gives a directory (its home, or a workdir that its Dockerfile gives the user).  An image's **owner manifest** lists
them, one ``UID GID PATH`` line each (hpc-sandbox 0.5's ``SandboxSpec.file_owners`` format, which its ``fakeroot``
sandboxes apply before anything runs), and the image map records one beside each image that such sandboxes run
(:mod:`platoon.envs.harbor.images`):

- **a built image** (the ``oci`` builder): its OCI-SIF's layer keeps the owners its Dockerfile gives, which the builder
  lists (``unsquashfs -lln``, :func:`from_squashfs_listing`) before it unpacks the layer as you;
- **a pulled image**, a registry reference that a task runs as its ``docker_image``: the owners its layers' tar headers
  give, the layers read in order with their whiteouts (:func:`pull_owners`), only the headers kept.

A manifest lists only what its format can hold: a path with a line break is left out.
"""

from __future__ import annotations

import io
import json
import logging
import os
import posixpath
import re
import tarfile
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from platoon.envs.harbor.images import ImageEntry, ImageMap, update_image_map

if TYPE_CHECKING:
    from platoon.envs.harbor.builders import RegistryClient

logger = logging.getLogger(__name__)

# One entry of `unsquashfs -lln`: the mode, uid/gid, the size (or a device's major and minor), the time, and the path
# below the listing's root, followed by " -> " and its target for a link.
_LISTING = re.compile(
    r"^(?P<mode>\S{10})\s+(?P<uid>[0-9]+)/(?P<gid>[0-9]+)\s+[0-9]+(?:,\s*[0-9]+)?\s+"
    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\s(?P<path>squashfs-root(?:/.*)?)$"
)
_INDEX_TYPES = ("application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json")
_WHITEOUT = ".wh."
_OPAQUE = ".wh..wh..opq"
_ARCHITECTURES = {"x86_64": "amd64", "aarch64": "arm64"}  # the kernel's names, as images name them
_REGISTRY_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class Owner:
    """The owner an image gives a path: its uid and gid."""

    path: str
    uid: int
    gid: int


def write_manifest(owners: Iterable[Owner], path: Path) -> int:
    """Write the manifest of ``owners`` that are not root's user and group to ``path`` (through a new file and a
    rename), sorted by path; how many it lists.  A path with a line break is left out, with a warning."""

    lines = []
    for owner in sorted(owners, key=lambda each: each.path):
        if owner.uid == 0 and owner.gid == 0:
            continue
        if "\n" in owner.path or "\r" in owner.path:
            logger.warning("An owner manifest leaves out %r, whose name holds a line break", owner.path)
            continue
        lines.append(f"{owner.uid} {owner.gid} {owner.path}\n")
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes("".join(lines).encode("utf-8", "surrogateescape"))  # names need not be UTF-8
    os.replace(partial, path)
    return len(lines)


def from_squashfs_listing(listing: str) -> list[Owner]:
    """The owners an ``unsquashfs -lln`` listing gives each path, its paths made absolute (``squashfs-root`` is
    ``/``); lines that are not entries are skipped."""

    owners = []
    for line in listing.splitlines():
        if (match := _LISTING.match(line)) is None:
            continue
        path = match.group("path").removeprefix("squashfs-root") or "/"
        if match.group("mode").startswith("l"):
            path = path.partition(" -> ")[0]
        owners.append(Owner(path, int(match.group("uid")), int(match.group("gid"))))
    return owners


def layer_owners(layers: Iterable[IO[bytes]]) -> list[Owner]:
    """The owners the tar ``layers`` give their paths, applied in order as an image's layers are: a later layer's entry
    replaces an earlier one's, a whiteout (``.wh.<name>``) removes ``<name>`` and what lies below it, and an opaque
    directory (``.wh..wh..opq``) what earlier layers put below its directory.  Each stream is read once, in order."""

    found: dict[str, Owner] = {}
    for stream in layers:
        layer: dict[str, Owner] = {}
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                path = posixpath.normpath("/" + member.name)
                directory, name = posixpath.split(path)
                if name == _OPAQUE:
                    found = {key: owner for key, owner in found.items() if not _below(key, directory)}
                elif name.startswith(_WHITEOUT):
                    gone = posixpath.join(directory, name.removeprefix(_WHITEOUT))
                    found = {key: owner for key, owner in found.items() if key != gone and not _below(key, gone)}
                else:
                    layer[path] = Owner(path, member.uid, member.gid)
        found.update(layer)
    return list(found.values())


@dataclass(frozen=True)
class PulledImage:
    """What a registry gives of an image: its pinned reference and digest, its configuration's ``USER``, its owners."""

    image: str  # docker://repo@sha256:...
    digest: str
    user: str | None
    owners: list[Owner]


def pull_owners(
    reference: str,
    *,
    urlopen: Callable[..., Any] | None = None,
    architecture: str | None = None,
    timeout_s: float = _REGISTRY_TIMEOUT_S,
) -> PulledImage:
    """The image ``reference`` names (``docker://repo:tag``, or as Docker names one), as its registry serves it for
    ``architecture`` (Docker's names: ``amd64``, ``arm64``; this host's by default), asked anonymously: its digest, its
    configuration's ``USER`` and the owners its layers give (:func:`layer_owners`).  ``urlopen`` makes the requests,
    whose redirects (a blob's to a CDN, say) never carry the registry's token
    (:class:`~platoon.envs.harbor.builders.RegistryClient`).  Raises :class:`~platoon.envs.harbor.builders.DigestError`,
    ``OSError`` or ``ValueError`` when the registry cannot say."""

    from platoon.envs.harbor.builders import RegistryClient, parse_reference, pinned_reference, registry_digest

    name = reference.removeprefix("docker://")
    host, repository, _, _ = parse_reference(name)
    opener = urlopen or urllib.request.urlopen
    digest = registry_digest(name, timeout_s=timeout_s, urlopen=opener)
    client = RegistryClient(opener, repository, timeout_s)
    base = f"https://{host}/v2/{repository}"
    manifest = _json(client, f"{base}/manifests/{digest}")
    if manifest.get("mediaType") in _INDEX_TYPES or "manifests" in manifest:
        wanted = architecture or _ARCHITECTURES.get(os.uname().machine, os.uname().machine)
        platforms = [entry for entry in manifest.get("manifests") or [] if _platform(entry) == ("linux", wanted)]
        if not platforms:
            raise ValueError(f"{reference} has no image for linux/{wanted}")
        manifest = _json(client, f"{base}/manifests/{platforms[0]['digest']}")
    config = _json(client, f"{base}/blobs/{manifest['config']['digest']}")

    def layers() -> Iterator[IO[bytes]]:  # each blob streamed in turn, its response closed once it is read
        for layer in manifest.get("layers") or []:
            with io.BufferedReader(_Stream(client.open(f"{base}/blobs/{layer['digest']}"))) as stream:
                yield stream

    return PulledImage(
        image=f"docker://{pinned_reference(name, digest)}",
        digest=digest,
        user=(config.get("config") or {}).get("User") or None,
        owners=layer_owners(layers()),
    )


def record_pulled(
    references: Iterable[str],
    *,
    image_map: Path,
    out_dir: Path,
    log: Callable[[str], None] | None = None,
    pull: Callable[[str], PulledImage] | None = None,
) -> dict[str, ImageEntry]:
    """Record in ``image_map`` a pulled entry for each of ``references`` (``docker://repo:tag``) that it lacks: the
    image its registry serves now, pinned, its ``USER``, and its owner manifest, written to ``out_dir/pulled``
    (``pull``, by default :func:`pull_owners`).  The new entries; ``ValueError`` names those it could not pull, after
    trying them all."""

    say = log or logger.info
    pull = pull or pull_owners
    current = ImageMap.load(image_map) if image_map.exists() else ImageMap(path=image_map)
    recorded: dict[str, ImageEntry] = {}
    failures: list[str] = []
    for reference in dict.fromkeys(references):
        if current.get(reference) is not None:
            continue
        try:
            pulled = pull(reference)
        except (OSError, ValueError, RuntimeError) as exc:  # RuntimeError: the registry's DigestError
            failures.append(f"{reference}: {exc}")
            continue
        manifest = out_dir / "pulled" / f"{pulled.digest.partition(':')[2]}.owners"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        listed = write_manifest(pulled.owners, manifest)
        entry = ImageEntry(
            image=pulled.image,
            builder="pulled",
            user=pulled.user,
            file_owners=str(manifest),
            receipt={"reference": reference, "digest": pulled.digest, "file_owners": listed},
        )
        update_image_map(image_map, {reference: entry})
        recorded[reference] = entry
        say(f"recorded {reference} as {pulled.image}: its layers give {listed} path(s) other owners than root")
    if failures:
        raise ValueError("; ".join(failures))
    return recorded


def _platform(entry: Mapping[str, Any]) -> tuple[str | None, str | None]:
    platform = entry.get("platform") or {}
    return platform.get("os"), platform.get("architecture")


def _json(client: RegistryClient, url: str) -> Mapping[str, Any]:
    with client.open(url) as response:
        return json.loads(response.read())


class _Stream(io.RawIOBase):
    """A response read as it streams in; closing the stream closes the response."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        data = self._response.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._response.close()
        super().close()


def _below(path: str, directory: str) -> bool:
    return path.startswith(directory.rstrip("/") + "/")
