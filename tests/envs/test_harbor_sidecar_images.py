"""The images of stacks' sidecars (platoon.envs.harbor.builders): build contexts planned and built whole, and the
registry images that sidecars run, pinned in the image map by the digest their registry serves, with the configuration
it gives their image (a stand-in registry here, never the network), which sessions run them with; the ``def`` builder's
images get their base image's configuration, and the ``singularity`` provider builds with it on Apptainer."""

from __future__ import annotations

import email.message
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from typing import Any

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import builders
from platoon.envs.harbor.builders import (
    BuildError,
    DigestError,
    build_image,
    pin_images,
    plan_images,
    provider_builder,
    registry_config,
    verify_images,
)
from platoon.envs.harbor.dockerfile import plan_dockerfile
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.images import ImageEntry, ImageMap, pulled_key, update_image_map
from platoon.envs.harbor.locked import LockedTasks, TaskImages

INDEX = "sha256:" + "1" * 64
AMD64 = "sha256:" + "2" * 64
ARM64 = "sha256:" + "3" * 64
CONFIG = "sha256:" + "4" * 64
POSTGRES = {
    "architecture": "amd64",
    "config": {
        "Entrypoint": ["docker-entrypoint.sh"],
        "Cmd": ["postgres"],
        "Env": ["PATH=/usr/local/bin:/usr/bin", "PGDATA=/var/lib/postgresql/data"],
        "ExposedPorts": {"5432/tcp": {}},
        "Volumes": {"/var/lib/postgresql/data": {}},
        "StopSignal": "SIGINT",
    },
    "rootfs": {"type": "layers", "diff_ids": []},
}
DB = "services:\n  db:\n    image: postgres:16-alpine\n"
STACK = DB + "  api:\n    build: ./api\n    command: [python3, /srv/api.py]\n"
LOCK_STACKS: dict[str, bool] = {}  # a fetch's default, which locks stacks as runnable


def _task(root: Path, name: str, compose: str) -> Path:
    task = root / name
    (task / "environment" / "api").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text('[environment]\ndocker_image = "python:3.12-slim"\n')
    (task / "instruction.md").write_text("Use the database.\n")
    (task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    (task / "environment" / "docker-compose.yaml").write_text(compose)
    (task / "environment" / "api" / "Dockerfile").write_text("FROM python:3.12-slim\nCOPY api.py /srv/api.py\n")
    (task / "environment" / "api" / "api.py").write_text(f"print({name!r})\n")
    return task


@pytest.fixture
def lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    use_site_cache(monkeypatch, tmp_path / "cache")
    _task(tmp_path / "dataset", "one", STACK)
    _task(tmp_path / "dataset", "two", DB)
    return fetch(fetch_config(str(tmp_path / "dataset"), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))


def _headers(**fields: str) -> email.message.Message:
    headers = email.message.Message()
    for name, value in fields.items():
        headers[name.replace("_", "-")] = value
    return headers


class Registry:
    """A registry of one image of two platforms, which asks for an anonymous token first, as Docker Hub does."""

    realm = "https://auth.registry.example/token"
    base = "https://registry-1.docker.io/v2/library/postgres"

    def __init__(self) -> None:
        index = {
            "manifests": [
                {"digest": "sha256:" + "9" * 64, "platform": {"os": "unknown", "architecture": "unknown"}},
                {"digest": AMD64, "platform": {"os": "linux", "architecture": "amd64"}},
                {"digest": ARM64, "platform": {"os": "linux", "architecture": "arm64", "variant": "v8"}},
            ]
        }
        manifest = {"config": {"digest": CONFIG}, "layers": []}
        self.documents = {
            f"{self.base}/manifests/16-alpine": (index, INDEX),
            f"{self.base}/manifests/{INDEX}": (index, INDEX),
            f"{self.base}/manifests/{AMD64}": (manifest, AMD64),
            f"{self.base}/manifests/{ARM64}": ({"config": {"digest": "sha256:" + "5" * 64}}, ARM64),
            f"{self.base}/blobs/{CONFIG}": (POSTGRES, None),
        }
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float | None = None) -> urllib.response.addinfourl:
        self.requests.append(request)
        url = request.full_url
        if url.startswith(self.realm + "?"):
            body = json.dumps({"token": "anonymous"}).encode()
            return urllib.response.addinfourl(io.BytesIO(body), _headers(Content_Type="application/json"), url, 200)
        if request.get_header("Authorization") != "Bearer anonymous":
            challenge = f'Bearer realm="{self.realm}",service="registry.docker.io"'
            raise urllib.error.HTTPError(url, 401, "Unauthorized", _headers(WWW_Authenticate=challenge), None)
        if url not in self.documents:
            raise urllib.error.HTTPError(url, 404, "Not Found", _headers(), None)
        document, digest = self.documents[url]
        headers = _headers(Content_Type="application/json", **({"Docker_Content_Digest": digest} if digest else {}))
        return urllib.response.addinfourl(io.BytesIO(json.dumps(document).encode()), headers, url, 200)


def test_a_lock_plans_its_sidecars_build_contexts_and_registry_images(lock, tmp_path: Path):
    plan = plan_images(lock)
    [api] = [image for image in plan.images if image.role == "service"]
    assert api.tasks == ["one"] and api.context == "environment/api" and api.dockerfile == "Dockerfile"
    assert api.to_json()["context"] == "environment/api/Dockerfile" and api.base == ["python:3.12-slim"]
    assert [(pull.reference, pull.tasks, pull.pinned) for pull in plan.pulls] == [
        ("postgres:16-alpine", ["one", "two"], False)
    ]
    assert plan.to_json()["unpinned"] == 1
    pinned = ImageMap.load({"images": {pulled_key("postgres:16-alpine"): {"image": "docker://postgres@" + INDEX}}})
    assert [pull.pinned for pull in plan_images(lock, pinned).pulls] == [True]


def test_a_sidecar_s_build_context_builds_whole_and_only_the_docker_builder_passes_build_arguments(
    lock, tmp_path: Path
):
    task_dir = tmp_path / "cache" / "harbor" / lock.get("one").path
    context = task_dir / "environment" / "api"
    whole = plan_dockerfile(context / "Dockerfile", context, whole=True, build_args={"MODE": "1"})
    with pytest.raises(BuildError, match="build arguments \\['MODE'\\], which only the docker builder passes"):
        build_image(
            task_dir,
            whole.key,
            role="service",
            context="environment/api",
            build_args={"MODE": "1"},
            builder="oci",
            out_dir=tmp_path / "out",
            scratch=tmp_path / "scratch",
        )
    prefix = plan_dockerfile(context / "Dockerfile", context).key
    with pytest.raises(BuildError, match=f"environment/api/Dockerfile plans build key {whole.key}, not {prefix}"):
        build_image(
            task_dir,
            prefix,
            role="service",
            context="environment/api",
            build_args={"MODE": "1"},
            out_dir=tmp_path / "out",
            scratch=tmp_path / "scratch",
        )


def test_the_registry_gives_the_image_s_configuration_for_this_host_s_platform():
    registry = Registry()

    digest, config = registry_config("postgres:16-alpine", urlopen=registry, machine="x86_64")

    assert digest == INDEX and config == POSTGRES  # the index's digest, which a pull of the tag starts from
    asked = [request.full_url.removeprefix(Registry.base) for request in registry.requests]
    assert asked[-3:] == ["/manifests/16-alpine", f"/manifests/{AMD64}", f"/blobs/{CONFIG}"]
    for request in registry.requests:  # the token is never sent on where a redirect leads, such as a blob's storage
        assert "Authorization" not in request.headers
    with pytest.raises(DigestError, match="names no image configuration|HTTP 404"):
        registry_config("postgres:16-alpine", urlopen=registry, machine="aarch64")  # its config blob is not served
    with pytest.raises(DigestError, match="has no image for linux/riscv64"):
        registry_config("postgres:16-alpine", urlopen=registry, machine="riscv64")
    assert registry_config(f"postgres@{INDEX}", urlopen=registry, machine="x86_64")[0] == INDEX


def _resolve(reference: str) -> tuple[str, dict[str, Any]]:
    if reference.startswith("missing"):
        raise DigestError(f"{reference}: HTTP 404")
    return "sha256:" + hashlib.sha256(reference.encode()).hexdigest(), POSTGRES


def test_pins_record_each_registry_image_by_digest_with_its_configuration(lock, tmp_path: Path):
    image_map = tmp_path / "images.json"

    pinned = pin_images(lock, image_map=image_map, resolve=_resolve)

    digest = _resolve("postgres:16-alpine")[0]
    entry = ImageMap.load(image_map).entries[pulled_key("postgres:16-alpine")]
    assert pinned == {"postgres:16-alpine": entry} and entry.builder == "pulled"
    assert entry.image == f"docker://postgres@{digest}"
    assert (entry.entrypoint, entry.cmd, entry.stop_signal) == (("docker-entrypoint.sh",), ("postgres",), "SIGINT")
    assert entry.env["PGDATA"] == "/var/lib/postgresql/data" and entry.exposed_ports == ("5432/tcp",)
    assert entry.receipt["reference"] == "postgres:16-alpine" and entry.receipt["digest"] == digest
    assert pin_images(lock, image_map=image_map, resolve=_resolve) == {}  # pinned already
    assert list(pin_images(lock, image_map=image_map, references=["postgres:16-alpine"], resolve=_resolve)) == [
        "postgres:16-alpine"
    ]
    with pytest.raises(ValueError, match="run no registry image \\['redis:7'\\]"):
        pin_images(lock, image_map=image_map, references=["redis:7"], resolve=_resolve)


def test_a_pin_that_fails_is_named_after_the_others_are_pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    use_site_cache(monkeypatch, tmp_path / "cache")
    _task(tmp_path / "dataset", "one", "services:\n  db:\n    image: missing:1\n  kv:\n    image: redis:7\n")
    lock = fetch(fetch_config(str(tmp_path / "dataset"), lock=tmp_path / "lock.json", refuse=LOCK_STACKS))

    with pytest.raises(DigestError, match="missing:1: missing:1: HTTP 404"):
        pin_images(lock, image_map=tmp_path / "images.json", resolve=_resolve)
    assert list(ImageMap.load(tmp_path / "images.json").entries) == [pulled_key("redis:7")]


def test_sessions_take_a_registry_image_as_the_image_map_pinned_it(lock, tmp_path: Path):
    locked = LockedTasks(lock=tmp_path / "lock.json")
    images = TaskImages(locked, "singularity")
    assert images.for_pulled("postgres:16-alpine") is None  # no map yet
    entry = ImageEntry(image="docker://postgres@" + INDEX, builder="pulled", user="postgres", cmd=("postgres",))
    assert images.path is not None
    update_image_map(images.path, {pulled_key("postgres:16-alpine"): entry})
    pulled = TaskImages(locked, "singularity").for_pulled("postgres:16-alpine")
    assert pulled is not None and (pulled.image, pulled.user, pulled.cmd) == (entry.image, "postgres", ("postgres",))


def test_def_builds_record_their_base_image_s_configuration_under_what_their_steps_set(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        'FROM postgres:16\nCMD ["postgres", "-c", "fsync=off"]\nEXPOSE 9000\nRUN true\n'
    )
    plan = plan_dockerfile(context / "Dockerfile", context)
    built = ImageEntry(image="/srv/images/x.sif", builder="def", env={"PATH": "/usr/bin"}, workdir=None)

    entry = builders._with_base_config(
        built, plan, {"postgres:16": INDEX}, lambda reference: (INDEX, POSTGRES), required=True
    )

    assert (entry.entrypoint, entry.cmd) == (("docker-entrypoint.sh",), ("postgres", "-c", "fsync=off"))
    assert entry.exposed_ports == ("5432/tcp", "9000/tcp") and entry.stop_signal == "SIGINT"
    assert entry.env == {"PATH": "/usr/bin"} and entry.workdir is None  # the build's own

    def offline(reference: str) -> tuple[str, dict[str, Any]]:
        raise DigestError("no network")

    with caplog.at_level(logging.WARNING, logger="platoon.envs.harbor.builders"):
        assert builders._with_base_config(built, plan, {}, offline, required=False) == built  # a task's own image
    assert "Cannot read the configuration of postgres:16" in caplog.text


def test_a_def_build_of_a_sidecar_s_image_fails_without_its_base_image_s_configuration(
    lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A sidecar runs its image's ENTRYPOINT and CMD, which a definition file keeps none of: where the registry cannot
    give its base image's configuration, the build fails rather than record an image that would run another command
    (a task's own image, whose command no session runs, is recorded, with a warning)."""

    def build_def(runner: Any, plan: Any, context: Path, work: Path, target: Path, digests: Any) -> tuple[Path, None]:
        target.write_text("def\n")
        return target, None

    def offline(reference: str) -> tuple[str, dict[str, Any]]:
        raise DigestError("registry-1.docker.io answered HTTP 429")

    monkeypatch.setattr(builders, "_build_def", build_def)
    monkeypatch.setattr(builders, "_sif_environment", lambda runner, sif: {})
    task_dir = tmp_path / "cache" / "harbor" / lock.get("one").path
    context = task_dir / "environment" / "api"
    with pytest.raises(BuildError, match="Cannot read the configuration of python:3.12-slim, the base of"):
        build_image(
            task_dir,
            plan_dockerfile(context / "Dockerfile", context, whole=True).key,
            role="service",
            context="environment/api",
            builder="def",
            out_dir=tmp_path / "out",
            scratch=tmp_path / "scratch",
            resolve_digest=lambda reference: INDEX,
            resolve_config=offline,
        )


def test_images_verify_finds_the_registry_images_of_sidecars_that_the_image_map_has_not_pinned(lock):
    """As a run's check does, on every provider that runs images (not ``process``)."""

    def pulls(image_map: ImageMap, **options: bool) -> list[tuple[str, str]]:
        checks = verify_images(lock, image_map, **options)
        return [(check.image.key, check.status) for check in checks if check.image.role == "service"]

    key = pulled_key("postgres:16-alpine")
    built = {plan_images(lock).images[0].key: {"image": "docker://built"}}  # the api's build context
    assert pulls(ImageMap.load({"images": built})) == [(plan_images(lock).images[0].key, "unchecked"), (key, "missing")]
    pinned = ImageMap.load({"images": {**built, key: {"image": "docker://postgres@" + INDEX}}})
    assert pulls(pinned)[-1] == (key, "unchecked")  # a registry reference, which is not hashed here
    assert [check for check in verify_images(lock, ImageMap(), pins=False) if check.image.key == key] == []


@pytest.mark.parametrize(
    ("version", "builder"), [("apptainer version 1.3.6", "def"), ("singularity-ce version 4.4.1", "oci")]
)
def test_the_singularity_provider_builds_with_def_where_its_runtime_is_apptainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, builder: str
):
    runtime = tmp_path / "singularity"
    runtime.write_text(f"#!/bin/sh\necho '{version}'\n")
    runtime.chmod(0o755)
    monkeypatch.setenv("PLATOON_SINGULARITY", str(runtime))
    builders._runtime_is_apptainer.cache_clear()
    try:
        assert provider_builder("singularity") == builder
    finally:
        builders._runtime_is_apptainer.cache_clear()
    assert (provider_builder("docker"), provider_builder("enroot"), provider_builder("process")) == (
        "docker",
        None,
        None,
    )
