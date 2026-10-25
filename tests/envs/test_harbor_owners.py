"""Owner manifests (:mod:`platoon.envs.harbor.owners`): the paths an image gives users other than root, as an
OCI-SIF's layer lists them, or as a registry's layers give them, whiteouts and all; how a pulled image is recorded in
an image map; and that no redirect carries the registry's token."""

from __future__ import annotations

import email.message
import gzip
import hashlib
import http.client
import io
import json
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from platoon.envs.harbor.images import ImageMap
from platoon.envs.harbor.owners import (
    Owner,
    PulledImage,
    from_squashfs_listing,
    layer_owners,
    pull_owners,
    record_pulled,
    write_manifest,
)

# As `unsquashfs -lln` lists a layer (SquashFS 4.4): its header, then one line an entry, a device's major and minor
# where a file's size goes, and a link's target after its name.
LISTING = """Parallel unsquashfs: Using 8 processors
5 inodes (3 blocks) to write

drwxr-xr-x 0/0                    72 2026-10-26 03:38 squashfs-root
crw-rw-rw- 0/0                 1,  3 2026-10-26 03:38 squashfs-root/dev/null
drwxr-xr-x 1000/1000              30 2026-10-26 03:38 squashfs-root/srv/node
-rw-r--r-- 1/2                     1 2026-10-26 03:38 squashfs-root/dir with space/f x
lrwxrwxrwx 1000/1000               9 2026-10-26 03:38 squashfs-root/srv/link -> node
"""


def test_a_layers_listing_gives_each_path_its_owner():
    assert from_squashfs_listing(LISTING) == [
        Owner("/", 0, 0),
        Owner("/dev/null", 0, 0),
        Owner("/srv/node", 1000, 1000),
        Owner("/dir with space/f x", 1, 2),
        Owner("/srv/link", 1000, 1000),  # the link itself
    ]


def test_a_manifest_lists_what_root_does_not_own_sorted(tmp_path: Path):
    owners = [
        Owner("/srv/node", 1000, 1000),
        Owner("/etc/shadow", 0, 42),
        Owner("/usr", 0, 0),
        Owner("/bad\nname", 5, 5),
        Owner("/caf\udce9", 7, 7),  # a name that is not UTF-8, as unsquashfs prints it
    ]

    assert write_manifest(owners, tmp_path / "x.owners") == 3
    assert (tmp_path / "x.owners").read_bytes() == b"7 7 /caf\xe9\n0 42 /etc/shadow\n1000 1000 /srv/node\n"
    assert not list(tmp_path.glob(".*"))  # written through a file of its own, then renamed


def _layer(entries: list[tuple[str, int, int]], *, directories: tuple[str, ...] = ()) -> io.BytesIO:
    """A gzip-compressed tar layer of ``entries`` (empty files, by path, uid and gid) and ``directories``."""

    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for path in directories:
            info = tarfile.TarInfo(path)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for path, uid, gid in entries:
            info = tarfile.TarInfo(path)
            info.uid, info.gid = uid, gid
            archive.addfile(info, io.BytesIO(b""))
    data.seek(0)
    return data


def test_layers_apply_in_order_with_their_whiteouts():
    base = _layer(
        [
            ("./srv/node/.profile", 1000, 1000),
            ("etc/passwd", 0, 0),
            ("var/cache/app/x", 100, 101),
            ("opt/a/b", 5, 5),
            ("opt/ab", 6, 6),
        ],
        directories=("srv/node",),
    )
    top = _layer(
        [
            ("opt/.wh.a", 0, 0),  # removes /opt/a and what lies below it, not /opt/ab
            ("var/.wh..wh..opq", 0, 0),  # /var is opaque: what the base put below it is gone
            ("var/new", 7, 7),
            ("srv/node/.profile", 0, 0),  # replaced, root's now
        ]
    )

    owners = {owner.path: (owner.uid, owner.gid) for owner in layer_owners([base, top])}

    assert owners == {
        "/srv/node": (0, 0),
        "/srv/node/.profile": (0, 0),
        "/etc/passwd": (0, 0),
        "/opt/ab": (6, 6),
        "/var/new": (7, 7),
    }


class _Response:
    """What urlopen returns: a status, headers and a body, read in pieces or at once."""

    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        self._body = io.BytesIO(body)
        self.headers = email.message.Message()
        for name, value in headers.items():
            self.headers[name] = value

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        pass

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class _Registry:
    """A registry of one repository (``library/python``) that asks for an anonymous token, serves ``manifests`` (by tag
    and digest) and ``blobs`` (by digest), as urlopen does, and records the requests made of it (``requests``)."""

    def __init__(self, blobs: dict[str, bytes], manifests: dict[str, Any], *, tag: str = "3.12-slim") -> None:
        self.blobs, self.manifests, self.tag = blobs, manifests, tag
        self.by_digest = {
            "sha256:" + hashlib.sha256(json.dumps(body).encode()).hexdigest(): body for body in manifests.values()
        }
        self.requests: list[tuple[str, str, str | None]] = []
        self.sent: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> _Response:
        url, method = request.full_url, request.get_method()
        self.requests.append((method, url, request.get_header("Authorization")))
        self.sent.append(request)
        if url.startswith("https://auth.example/token"):
            return _Response(json.dumps({"token": "t0ken"}).encode(), {})
        if request.get_header("Authorization") != "Bearer t0ken":
            challenge = email.message.Message()
            challenge["WWW-Authenticate"] = 'Bearer realm="https://auth.example/token"'
            raise urllib.error.HTTPError(url, 401, "unauthorized", challenge, None)
        reference = url.rsplit("/", 1)[1]
        if "/manifests/" in url:
            manifest = self.manifests.get(reference) or self.by_digest[reference]
            body = json.dumps(manifest).encode()
            digest = "sha256:" + hashlib.sha256(body).hexdigest() if reference == self.tag else reference
            media = manifest.get("mediaType", "application/vnd.oci.image.index.v1+json")
            return _Response(
                b"" if method == "HEAD" else body, {"Docker-Content-Digest": digest, "Content-Type": media}
            )
        return _Response(self.blobs[reference], {})


def _image(user: str | None = "1000") -> tuple[dict[str, bytes], dict[str, Any]]:
    """An image of two layers for linux/amd64, beside one for linux/arm64, behind an index."""

    layers = [
        _layer([("srv/node/a", 1000, 1000), ("etc/shadow", 0, 42)]).read(),
        _layer([("srv/node/.wh.a", 0, 0), ("srv/node/b", 1000, 1000)]).read(),
    ]
    config = json.dumps({"config": {"User": user} if user else {}, "rootfs": {}}).encode()
    blobs = {"sha256:" + hashlib.sha256(blob).hexdigest(): blob for blob in [*layers, config]}
    digests = list(blobs)
    manifest = {
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": digests[2]},
        "layers": [{"digest": digests[0]}, {"digest": digests[1]}],
    }
    index = {
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {"digest": "sha256:" + "a" * 64, "platform": {"os": "linux", "architecture": "arm64"}},
            {"digest": "sha256:" + "b" * 64, "platform": {"os": "linux", "architecture": "amd64"}},
        ],
    }
    return blobs, {"3.12-slim": index, "sha256:" + "b" * 64: manifest, "sha256:" + "a" * 64: {"layers": []}}


def test_a_pulled_image_gives_the_owners_its_layers_give_and_its_user():
    blobs, manifests = _image()
    urlopen = _Registry(blobs, manifests)

    pulled = pull_owners("docker://python:3.12-slim", urlopen=urlopen, architecture="amd64")

    index_digest = "sha256:" + hashlib.sha256(json.dumps(manifests["3.12-slim"]).encode()).hexdigest()
    assert pulled.digest == index_digest and pulled.image == f"docker://python@{index_digest}"
    assert pulled.user == "1000"
    assert {(owner.path, owner.uid, owner.gid) for owner in pulled.owners} == {
        ("/etc/shadow", 0, 42),
        ("/srv/node/b", 1000, 1000),
    }
    asked = [url for _, url, _ in urlopen.requests]
    assert f"https://registry-1.docker.io/v2/library/python/manifests/sha256:{'b' * 64}" in asked  # amd64's
    assert f"https://registry-1.docker.io/v2/library/python/manifests/sha256:{'a' * 64}" not in asked
    with pytest.raises(ValueError, match="has no image for linux/s390x"):
        pull_owners("docker://python:3.12-slim", urlopen=urlopen, architecture="s390x")


def test_pulled_images_are_recorded_in_the_image_map_once(tmp_path: Path):
    image_map, out = tmp_path / "images" / "singularity.json", tmp_path / "images" / "singularity"
    pulled: list[str] = []

    def pull(reference: str) -> PulledImage:
        pulled.append(reference)
        if "broken" in reference:
            raise OSError("the registry is down")
        return PulledImage("docker://python@sha256:" + "c" * 64, "sha256:" + "c" * 64, "1000", [Owner("/srv", 1, 1)])

    recorded = record_pulled(["docker://python:3.12-slim"], image_map=image_map, out_dir=out, pull=pull)
    entry = ImageMap.load(image_map).get("docker://python:3.12-slim")

    assert recorded == {"docker://python:3.12-slim": entry} and entry is not None
    assert (entry.image, entry.builder, entry.user) == ("docker://python@sha256:" + "c" * 64, "pulled", "1000")
    assert Path(entry.file_owners or "").read_text() == "1 1 /srv\n"
    assert entry.receipt == {"reference": "docker://python:3.12-slim", "digest": "sha256:" + "c" * 64, "file_owners": 1}
    with pytest.raises(ValueError, match="docker://broken:1: the registry is down"):
        record_pulled(["docker://python:3.12-slim", "docker://broken:1"], image_map=image_map, out_dir=out, pull=pull)
    assert pulled == ["docker://python:3.12-slim", "docker://broken:1"]  # the recorded one is not pulled again


def test_no_redirect_carries_the_registrys_token():
    """The registry's token goes in a header that urllib leaves behind when it follows a redirect, so a blob's
    redirect to another host (a CDN, a bucket), which may refuse a request that carries one, never gets it."""

    blobs, manifests = _image()
    urlopen = _Registry(blobs, manifests)

    pull_owners("docker://python:3.12-slim", urlopen=urlopen, architecture="amd64")

    tokened = [request for request in urlopen.sent if request.get_header("Authorization") == "Bearer t0ken"]
    assert len(tokened) >= 4  # the manifests' and the blobs'
    handler = urllib.request.HTTPRedirectHandler()
    for request in tokened:
        redirected = handler.redirect_request(
            request, io.BytesIO(), 307, "redirect", http.client.HTTPMessage(), "https://cdn.example/blob?signed"
        )
        assert redirected is not None and not redirected.has_header("Authorization")


def test_a_gzip_stream_read_in_pieces_is_read_whole():
    """Layers stream through tarfile's ``r|*``, which reads compressed data in pieces; one larger than a piece reads
    whole."""

    big = _layer([(f"srv/{n:05}", 1, 1) for n in range(3000)]).read()
    assert len(gzip.decompress(big)) > 1 << 20
    assert len(layer_owners([io.BytesIO(big)])) == 3000
