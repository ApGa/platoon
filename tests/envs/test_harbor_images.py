"""Session images: the site's image map and the deferred Dockerfile steps a session applies."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from platoon.envs.harbor.images import (
    FileOp,
    ImageEntry,
    ImageMap,
    ImageMapError,
    SessionImage,
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


@pytest.mark.parametrize(
    ("planned", "user"), [("${NB_UID}", "1000"), ("${MISSING}", None), ("app", "app"), (None, "built")]
)
def test_a_planned_user_expands_against_the_images_environment(tmp_path: Path, planned: str | None, user: str | None):
    """A suffix's ``USER ${NB_UID}``, whose variable the base image's ENV sets, is that user, as a build makes it; one
    that expands to nothing is root's (None); without one, the built image's own."""

    entry = ImageEntry(image="/images/a.sif", env={"NB_UID": "1000"}, user="built")
    plan = {"kind": "shared", "key": KEY, "deferred": [], **({"config": {"user": planned}} if planned else {})}

    image = session_image(_task(tmp_path), plan=plan, image_map=ImageMap({KEY: entry}))

    assert image.user == user


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


def test_a_copys_owner_expands_like_its_destination(tmp_path: Path):
    _, _, files = deferred_effects(
        [["ARG", "USERNAME", "agent"], ["COPY", "--chown=${USERNAME}:staff", "input/", "/app/inputs/"]],
        env={},
        workdir=None,
        context=_task(tmp_path / "task") / "environment",
    )

    assert files == (FileOp("copy", "/app/inputs", source="input", into=True, owner="agent:staff"),)


def test_a_workdir_names_the_user_in_effect_there(tmp_path: Path):
    """The directories a WORKDIR makes are, as BuildKit makes them, the USER's in effect there: the built image's own
    before any deferred USER, then each deferred USER's, expanded as a build expands it (an empty one is root)."""

    _, workdir, files = deferred_effects(
        [
            ["WORKDIR", "/app"],
            ["ARG", "NAME", "agent"],
            ["USER", "${NAME}:staff"],
            ["WORKDIR", "work"],
            ["USER", "${NB_UID}"],
            ["WORKDIR", "/data"],
            ["USER", "${UNSET}"],
            ["WORKDIR", "/root/x"],
        ],
        env={"NB_UID": "1000"},
        workdir=None,
        context=_task(tmp_path / "task") / "environment",
        user="built",
    )

    assert workdir == "/root/x"
    assert files == (
        FileOp("mkdir", "/app", owner="built"),
        FileOp("mkdir", "/app/work", owner="agent:staff"),
        FileOp("mkdir", "/data", owner="1000"),
        FileOp("mkdir", "/root/x"),
    )


def test_a_workdir_before_any_deferred_user_is_the_built_images_users(tmp_path: Path):
    image_map = ImageMap({KEY: ImageEntry(image="/images/a.sif", user="jovyan")})  # USER set in the base or prefix

    image = session_image(_task(tmp_path), plan={"key": KEY, "deferred": [["WORKDIR", "/work"]]}, image_map=image_map)

    assert image.files == (FileOp("mkdir", "/work", owner="jovyan"),)


@pytest.mark.asyncio
async def test_a_workdirs_user_gets_the_directories_it_makes_where_users_are_honored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Where a sandbox honors users, the directories a deferred WORKDIR makes are the USER's in effect there, as
    BuildKit makes them, while those that were there keep their own; elsewhere they are the sandbox's own user's."""

    task_dir = _task(tmp_path / "task")
    sandbox = await ProcessProvider(root=tmp_path / "sandboxes").start(SandboxSpec(image="local", workdir="/app"))
    owner = f"{os.getuid()}:{os.getgid()}"  # the only owner this test may give
    given: list[list[str]] = []
    real = sandbox.exec

    async def exec(cmd, **options):
        if isinstance(cmd, list) and "chown" in cmd:
            given.append(cmd[cmd.index("chown") :])
        return await real(cmd, **options)

    image = session_image(
        task_dir,
        plan={"key": KEY, "deferred": [["USER", owner], ["WORKDIR", "/app"], ["WORKDIR", "made/deep"]]},
        image_map=ImageMap({KEY: ImageEntry(image="local")}),
    )
    monkeypatch.setattr(sandbox, "exec", exec)
    try:
        await apply_image_files(sandbox, image, workdir="/app", owners=True)
        assert given == [["chown", "-h", owner, "--", "/app/made", "/app/made/deep"]]  # /app was there
        assert sandbox.host_path("/app/made/deep").is_dir()
        given.clear()
        unhonored = replace(image, files=(FileOp("mkdir", "/app/other", owner=owner),))
        await apply_image_files(sandbox, unhonored, workdir="/app")
        assert given == [] and sandbox.host_path("/app/other").is_dir()
    finally:
        await sandbox.stop()


@pytest.mark.asyncio
async def test_a_copys_owner_applies_where_users_are_honored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Where a sandbox honors users, a deferred copy's --chown gives its owner what it copies and the directories it
    makes, while a destination that was there keeps its own, as BuildKit does; elsewhere every file is the sandbox's
    own user's."""

    task_dir = _task(tmp_path / "task")
    sandbox = await ProcessProvider(root=tmp_path / "sandboxes").start(SandboxSpec(image="local", workdir="/app"))
    owner = f"{os.getuid()}:{os.getgid()}"  # the only owner this test may give
    given: list[list[str]] = []
    real = sandbox.exec

    async def exec(cmd, **options):
        if isinstance(cmd, list) and "chown" in cmd:
            given.append(cmd[cmd.index("chown") :])
        return await real(cmd, **options)

    def copying(destination: str) -> SessionImage:
        return session_image(
            task_dir,
            plan={"key": KEY, "deferred": [["COPY", f"--chown={owner}", "input/", destination]]},
            image_map=ImageMap({KEY: ImageEntry(image="local")}),
        )

    monkeypatch.setattr(sandbox, "exec", exec)
    try:
        await apply_image_files(sandbox, copying("/app/inputs/"), workdir="/app")
        assert given == []
        await apply_image_files(sandbox, copying("/app/inputs/"), workdir="/app", owners=True)
        await apply_image_files(sandbox, copying("/app/made/inputs/"), workdir="/app", owners=True)
        copied = ["data.csv", "sub", "sub/more.csv"]
        assert given == [
            ["chown", "-h", owner, "--", *(f"/app/inputs/{name}" for name in copied)],  # /app/inputs was there
            [
                "chown",
                "-h",
                owner,
                "--",
                "/app/made",
                "/app/made/inputs",
                *(f"/app/made/inputs/{name}" for name in copied),
            ],
        ]
    finally:
        await sandbox.stop()


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
