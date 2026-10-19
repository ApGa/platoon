"""Session images: the site's image map and the deferred Dockerfile steps a session applies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platoon.envs.harbor.images import (
    FileOp,
    ImageEntry,
    ImageMap,
    ImageMapError,
    apply_image_files,
    deferred_effects,
    docker_image_reference,
    session_image,
    update_image_map,
)
from platoon.sandbox import SandboxSpec
from platoon.sandbox.process import ProcessProvider

KEY = "sha256:" + "a" * 64


def _task(root: Path) -> Path:
    for relative, text in {
        "environment/input/data.csv": "x\n1\n",
        "environment/input/sub/more.csv": "y\n",
        "environment/instance.json": "{}",
        "environment/Dockerfile": "FROM x\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_image_map_round_trip_relative_paths_and_locked_updates(tmp_path: Path):
    path = tmp_path / "site" / "images.json"
    update_image_map(path, {KEY: ImageEntry(image="images/a.sif", sha256="00", builder="oci", env={"A": "1"})})
    update_image_map(path, {"sha256:" + "b" * 64: ImageEntry(image="docker://python:3.12")})
    data = json.loads(path.read_text())
    assert data["version"] == 1 and sorted(data["images"]) == [KEY, "sha256:" + "b" * 64]
    loaded = ImageMap.load(path)
    assert loaded.get(KEY).image == str((tmp_path / "site" / "images" / "a.sif").resolve())  # next to the map
    assert loaded.get(KEY).env == {"A": "1"}
    assert loaded.get("sha256:" + "b" * 64).image == "docker://python:3.12"
    assert loaded.get("sha256:" + "c" * 64) is None
    inline = ImageMap.load({"images": {KEY: {"image": "local"}}})
    assert inline.get(KEY).image == "local"
    with pytest.raises(ImageMapError, match="needs an image"):
        ImageMap.load({"images": {KEY: {"env": {}}}})
    with pytest.raises(ImageMapError, match="does not exist"):
        ImageMap.load(tmp_path / "missing.json")


def test_deferred_effects_expand_like_a_build(tmp_path: Path):
    context = _task(tmp_path)
    env, workdir, files = deferred_effects(
        [
            ["ARG", "VERSION", "1.0"],
            ["ENV", "PATH", "/opt/v$VERSION/bin:$PATH"],
            ["ENV", "LITERAL", "\\$HOME"],
            ["WORKDIR", "data"],
            ["COPY", "input/", "/data/input/"],
            ["COPY", "instance.json", "instance.json"],
            ["COPY", "instance.json", "input/data.csv", "$HOME/many/"],
        ],
        env={"PATH": "/usr/bin", "HOME": "/root"},
        workdir="/workdir",
        context=context / "environment",
    )
    assert env == {"PATH": "/opt/v1.0/bin:/usr/bin", "HOME": "/root", "LITERAL": "$HOME"}
    assert workdir == "/workdir/data"
    assert files == (
        FileOp("mkdir", target="/workdir/data"),
        FileOp("copy", target="/data/input", source="input", into=True),
        FileOp("copy", target="/workdir/data/instance.json", source="instance.json", into=False),
        FileOp("copy", target="/root/many/instance.json", source="instance.json"),
        FileOp("copy", target="/root/many/data.csv", source="input/data.csv"),
    )


def test_session_image_from_the_map_and_refusal_without_it(tmp_path: Path):
    task_dir = _task(tmp_path)
    image_map = ImageMap({KEY: ImageEntry(image="/images/a.sif", env={"P": "1"}, workdir="/workdir", user="u")})
    plan = {"kind": "shared", "key": KEY, "deferred": [["COPY", "instance.json", "/workdir/instance.json"]]}
    image = session_image(task_dir, plan=plan, image_map=image_map, default_image="docker://default")
    assert (image.image, image.env, image.workdir, image.user, image.key) == (
        "/images/a.sif",
        {"P": "1"},
        "/workdir",
        "u",
        KEY,
    )
    assert image.files == (FileOp("copy", target="/workdir/instance.json", source="instance.json"),)
    with pytest.raises(ImageMapError, match="has no image for"):
        session_image(task_dir, plan=plan, image_map=ImageMap(), default_image="docker://default")
    with pytest.raises(ImageMapError, match="no image map"):
        session_image(task_dir, plan=plan, image_map=None, default_image="docker://default")


def test_prebuilt_and_default_images(tmp_path: Path):
    task_dir = _task(tmp_path)
    prebuilt = session_image(
        task_dir,
        plan={"kind": "prebuilt", "ref": "python:3.12", "upload_environment": True},
        image_map=None,
        docker_image="python:3.12",
    )
    assert prebuilt.image == "docker://python:3.12"
    assert prebuilt.files == (FileOp("copy", target="", source=".", into=True),)
    default = session_image(task_dir, plan={"kind": "none"}, image_map=None, default_image="docker://d")
    assert (default.image, default.files) == ("docker://d", ())
    assert docker_image_reference("/x/task.sif") == "/x/task.sif"
    assert docker_image_reference("images/task.sqsh") == "images/task.sqsh"
    assert docker_image_reference("oras://r/x:1") == "oras://r/x:1"
    assert docker_image_reference("ubuntu:24.04") == "docker://ubuntu:24.04"


@pytest.mark.asyncio
async def test_apply_image_files_in_a_sandbox(tmp_path: Path):
    task_dir = _task(tmp_path / "task")
    provider = ProcessProvider(root=tmp_path / "sandboxes")
    sandbox = await provider.start(SandboxSpec(image="local", workdir="/app"))
    try:
        await sandbox.exec(["mkdir", "-p", "/app/existing"])
        plan = {
            "key": KEY,
            "deferred": [
                ["WORKDIR", "/app/work"],
                ["COPY", "input/", "/app/inputs/"],
                ["COPY", "instance.json", "/app/existing"],
                ["COPY", "instance.json", "/tmp/renamed.json"],
            ],
        }
        image = session_image(task_dir, plan=plan, image_map=ImageMap({KEY: ImageEntry(image="local")}))
        await apply_image_files(sandbox, image, workdir="/app")
        assert sandbox.host_path("/app/work").is_dir()
        assert sandbox.host_path("/app/inputs/data.csv").read_text() == "x\n1\n"
        assert sandbox.host_path("/app/inputs/sub/more.csv").read_text() == "y\n"
        assert sandbox.host_path("/app/existing/instance.json").read_text() == "{}"  # into an existing directory
        assert sandbox.host_path("/tmp/renamed.json").read_text() == "{}"
        prebuilt = session_image(
            task_dir, plan={"kind": "prebuilt", "upload_environment": True}, image_map=None, docker_image="x:1"
        )
        await apply_image_files(sandbox, prebuilt, workdir="/app")
        assert sandbox.host_path("/app/instance.json").read_text() == "{}"  # environment/ into the workdir
        outside = session_image(
            task_dir,
            plan={"key": KEY, "deferred": [["COPY", "input/", "/data/input/"]]},
            image_map=ImageMap({KEY: ImageEntry(image="local")}),
        )
        with pytest.raises(RuntimeError, match="on the host"):
            await apply_image_files(sandbox, outside, workdir="/app")
    finally:
        await sandbox.stop()
