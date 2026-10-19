"""Base image digests: what a registry answers (a stand-in registry, never the network), and the Dockerfiles that
builds run, their base images named by digest."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import urllib.error
import urllib.parse
import urllib.request
import urllib.response
from pathlib import Path

import pytest

from platoon.envs.harbor import builders
from platoon.envs.harbor.builders import DigestError, registry_digest
from platoon.envs.harbor.dockerfile import plan_dockerfile

INDEX = "sha256:" + "1" * 64
MANIFEST_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
}


def _headers(**fields: str) -> email.message.Message:
    headers = email.message.Message()
    for name, value in fields.items():
        headers[name.replace("_", "-")] = value
    return headers


class Registry:
    """A registry that, as Docker Hub does, wants an anonymous token from its realm before it serves a manifest.

    ``digests`` maps manifest URLs to the digest they serve; ``digest_header`` says whether responses carry it as
    ``Docker-Content-Digest`` (else a GET serves the manifest, whose SHA-256 is its digest)."""

    realm = "https://auth.registry.example/token"

    def __init__(self, digests: dict[str, str], *, digest_header: bool = True, manifest: bytes = b"{}") -> None:
        self.digests = digests
        self.digest_header = digest_header
        self.manifest = manifest
        self.requests: list[tuple[str, str, dict[str, str]]] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> urllib.response.addinfourl:
        assert timeout is not None
        url, headers = request.full_url, dict(request.header_items())
        self.requests.append((request.get_method(), url, headers))
        if url.startswith(self.realm + "?"):
            scope = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))["scope"]
            return self._answer(request, json.dumps({"token": f"token for {scope}"}).encode(), "application/json")
        scope = "repository:" + url.split("/v2/", 1)[1].rsplit("/manifests/", 1)[0] + ":pull"
        if headers.get("Authorization") != f"Bearer token for {scope}":
            challenge = f'Bearer realm="{self.realm}",service="registry.example",scope="{scope}"'
            raise urllib.error.HTTPError(url, 401, "Unauthorized", _headers(WWW_Authenticate=challenge), None)
        if url not in self.digests:
            raise urllib.error.HTTPError(url, 404, "Not Found", _headers(), None)
        assert set(headers["Accept"].split(", ")) == MANIFEST_TYPES
        return self._answer(request, self.manifest, "application/vnd.oci.image.index.v1+json")

    def _answer(self, request: urllib.request.Request, body: bytes, content_type: str) -> urllib.response.addinfourl:
        headers = _headers(Content_Type=content_type)
        if self.digest_header and request.full_url in self.digests:
            headers["Docker-Content-Digest"] = self.digests[request.full_url]
        payload = b"" if request.get_method() == "HEAD" else body
        return urllib.response.addinfourl(io.BytesIO(payload), headers, request.full_url, 200)


def test_digests_come_from_the_registry_with_an_anonymous_token():
    url = "https://registry-1.docker.io/v2/library/python/manifests/3.12-slim"
    registry = Registry({url: INDEX})
    assert registry_digest("python:3.12-slim", urlopen=registry) == INDEX
    token = f"{Registry.realm}?service=registry.example&scope=repository%3Alibrary%2Fpython%3Apull"
    assert [(method, target) for method, target, _ in registry.requests] == [
        ("HEAD", url),  # 401, with the challenge that names the realm
        ("GET", token),
        ("HEAD", url),
    ]
    assert "Authorization" not in registry.requests[1][2]  # anonymous


def test_a_registry_without_digest_headers_serves_the_manifest_that_is_hashed():
    url = "https://ghcr.io/v2/org/tool/manifests/v1"
    manifest = b'{"schemaVersion": 2, "manifests": []}'
    registry = Registry({url: INDEX}, digest_header=False, manifest=manifest)
    assert registry_digest("ghcr.io/org/tool:v1", urlopen=registry) == "sha256:" + hashlib.sha256(manifest).hexdigest()
    assert [method for method, _, _ in registry.requests] == ["HEAD", "GET", "HEAD", "GET"]


def test_a_reference_pinned_by_digest_needs_no_request():
    registry = Registry({})
    assert registry_digest(f"python:3.12@{INDEX}", urlopen=registry) == INDEX
    assert registry.requests == []


@pytest.mark.parametrize(
    ("reference", "parsed"),
    [
        ("python", ("registry-1.docker.io", "library/python", "latest", None)),
        ("docker.io/library/python:3.12", ("registry-1.docker.io", "library/python", "3.12", None)),
        ("index.docker.io/org/tool:1", ("registry-1.docker.io", "org/tool", "1", None)),
        ("org/tool:1.0-rc", ("registry-1.docker.io", "org/tool", "1.0-rc", None)),
        ("ghcr.io/org/team/tool:v1", ("ghcr.io", "org/team/tool", "v1", None)),
        ("localhost:5000/tool", ("localhost:5000", "tool", "latest", None)),
        (f"quay.io/org/tool:2@{INDEX}", ("quay.io", "org/tool", "2", INDEX)),
    ],
)
def test_references_resolve_with_dockers_defaults(reference: str, parsed: tuple):
    assert builders._parse_reference(reference) == parsed


def test_registries_that_cannot_answer_raise_digest_errors():
    def offline(request: urllib.request.Request, timeout: float | None = None):
        raise urllib.error.URLError("no network")

    with pytest.raises(DigestError, match="registry-1.docker.io/v2/library/python/manifests/3.12: .*no network"):
        registry_digest("python:3.12", urlopen=offline)
    with pytest.raises(DigestError, match="HTTP 404"):
        registry_digest("python:3.99", urlopen=Registry({}))

    def private(request: urllib.request.Request, timeout: float | None = None):
        challenge = _headers(WWW_Authenticate='Basic realm="private"')
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", challenge, None)

    with pytest.raises(DigestError, match="HTTP 401"):
        registry_digest("registry.example/private/tool:1", urlopen=private)

    def tokens_down(request: urllib.request.Request, timeout: float | None = None):
        if request.full_url.startswith(Registry.realm):
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", _headers(), None)
        return Registry({})(request, timeout)

    with pytest.raises(DigestError, match=f"{Registry.realm} answered HTTP 503 to a token request"):
        registry_digest("python:3.12", urlopen=tokens_down)
    with pytest.raises(DigestError, match="not an image reference"):
        registry_digest("Not An/Image", urlopen=offline)
    with pytest.raises(DigestError, match="malformed digest"):
        registry_digest("python@sha256:abc", urlopen=offline)


def test_builds_run_dockerfiles_with_each_base_image_named_by_digest(tmp_path: Path):
    (tmp_path / "Dockerfile").write_text(
        "# syntax=docker/dockerfile:1\n"
        "ARG BASE=python:3.12-slim\n"
        "FROM --platform=linux/amd64 golang:1.22 AS build\n"
        "RUN go build -o /app .\n"
        "FROM $BASE\n"
        "COPY --from=build /app /usr/local/bin/app\n"
        "FROM build AS again\n"
        "RUN true\n"
        "FROM scratch\n"
        "COPY --from=again /app /app\n"
    )
    plan = plan_dockerfile(tmp_path / "Dockerfile")
    golang, python = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    digests = builders._base_digests(plan, {"golang:1.22": golang, "python:3.12-slim": python}.get)
    assert digests == {"golang:1.22": golang, "python:3.12-slim": python}  # not the stage, nor scratch
    assert builders._pinned_dockerfile(plan, digests).splitlines() == [
        "# syntax=docker/dockerfile:1",
        "ARG BASE=python:3.12-slim",
        f"FROM --platform=linux/amd64 golang@{golang} AS build",
        "RUN go build -o /app .",
        f"FROM python@{python}",
        "COPY --from=build /app /usr/local/bin/app",
        "FROM build AS again",
        "RUN true",
        "FROM scratch",
        "COPY --from=again /app /app",
    ]
    unresolved = builders._pinned_dockerfile(plan, {"golang:1.22": None, "python:3.12-slim": python})
    assert "FROM --platform=linux/amd64 golang:1.22 AS build\n" in unresolved  # built as written
    assert builders._pinned_reference("localhost:5000/team/tool:1", golang) == f"localhost:5000/team/tool@{golang}"


def test_a_run_asks_the_registry_once_per_reference():
    asked: list[str] = []

    def resolve(reference: str) -> str:
        asked.append(reference)
        if reference == "private.example/tool:1":
            raise DigestError("HTTP 401")
        return INDEX

    lookup = builders._remembered(resolve)
    assert [lookup("python:3.12"), lookup("python:3.12")] == [INDEX, INDEX]
    for _ in range(2):
        with pytest.raises(DigestError, match="HTTP 401"):
            lookup("private.example/tool:1")
    assert asked == ["python:3.12", "private.example/tool:1"]
