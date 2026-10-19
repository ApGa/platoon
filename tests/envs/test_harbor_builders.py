"""Planning, building and verifying the images of a lock: grouping by build key, definition files, base images
pinned by digest, and the OCI pipeline with stand-ins for singularity, unsquashfs and the registry."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shlex
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from platoon.envs.harbor import builders
from platoon.envs.harbor.builders import (
    BuildError,
    DigestError,
    build_images,
    definition_file,
    plan_images,
    verify_images,
)
from platoon.envs.harbor.dockerfile import parse_dockerfile, plan_dockerfile
from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.envs.harbor.hashing import file_sha256
from platoon.envs.harbor.images import ImageEntry, ImageMap, has_image

SHARED = "FROM python:3.12-slim\nRUN pip install numpy\nWORKDIR /app\nCOPY instance.json /app/instance.json\n"
OWN = "FROM python:3.12-slim\nCOPY seed.py /opt/seed.py\nRUN python /opt/seed.py\n"


def _digest(reference: str) -> str:
    return "sha256:" + hashlib.sha256(reference.encode()).hexdigest()


@pytest.fixture(autouse=True)
def registry(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A stand-in for every registry, so that no build here asks a real one: a base image resolves to a digest made
    from its reference. Returns the references asked for."""

    asked: list[str] = []

    def resolve(reference: str, **_: object) -> str:
        asked.append(reference)
        return _digest(reference)

    monkeypatch.setattr(builders, "registry_digest", resolve)
    return asked


def _dataset(root: Path) -> Path:
    tasks = {
        "a": (SHARED, {"environment/instance.json": "{}"}),
        "b": (SHARED, {"environment/instance.json": '{"b": 1}'}),
        "c": (OWN, {"environment/seed.py": "print(3)"}),
        "prebuilt": (None, {}),
        "gpu": (SHARED, {"environment/instance.json": "{}"}),
    }
    for name, (dockerfile, files) in tasks.items():
        toml = {"prebuilt": '[environment]\ndocker_image = "python:3.12"\n', "gpu": "[environment]\ngpus = 2\n"}
        contents = {
            "task.toml": toml.get(name, ""),
            "instruction.md": "Do it.\n",
            "tests/test.sh": "echo 1\n",
            **({"environment/Dockerfile": dockerfile} if dockerfile else {}),
            **files,
        }
        for relative, text in contents.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    return root


@pytest.fixture
def lock(tmp_path: Path):
    return fetch(fetch_config(str(_dataset(tmp_path / "dataset")), cache=tmp_path / "cache", lock=tmp_path / "l.json"))


def test_plan_groups_runnable_tasks_by_build_key(lock, tmp_path: Path):
    plan = plan_images(lock)
    assert [(image.kind, sorted(image.tasks)) for image in plan.images] == [("shared", ["a", "b"]), ("per_task", ["c"])]
    assert plan.images[0].base == ["python:3.12-slim"] and plan.images[0].example.name == "a"
    assert plan.prebuilt == {"python:3.12": 1} and plan.refused == {"gpus": 1} and plan.default_image == 0
    assert len(plan.missing) == 2
    sif = tmp_path / "shared.sif"
    sif.write_bytes(b"image")
    present = plan_images(lock, ImageMap.load({"images": {plan.images[0].key: {"image": str(sif)}}}))
    assert [image.present for image in present.images] == [True, False]
    assert present.to_json()["missing"] == 1


def test_plan_counts_a_local_image_only_while_its_file_exists(lock, tmp_path: Path):
    shared, own = (image.key for image in plan_images(lock).images)
    image_map = ImageMap.load({"images": {shared: {"image": str(tmp_path / "gone.sif")}, own: {"image": "local"}}})
    assert [image.present for image in plan_images(lock, image_map).images] == [False, True]
    (tmp_path / "gone.sif").write_bytes(b"image")
    assert [image.present for image in plan_images(lock, image_map).images] == [True, True]
    registry_map = ImageMap.load({"images": {shared: {"image": "docker://registry.example/shared:1"}}})
    assert [image.present for image in plan_images(lock, registry_map).images] == [True, False]


def test_plan_counts_a_truncated_local_image_as_missing(lock, tmp_path: Path):
    """A local image file that is not the size its build recorded (a truncated copy) counts as missing, in the plan and
    in a run's check, so prepare builds it again; an entry that records no size counts while its file exists."""

    shared, own = (image.key for image in plan_images(lock).images)
    sif = tmp_path / "shared.sif"
    sif.write_bytes(b"a whole image")
    whole = {"image": str(sif), "receipt": {"size_bytes": sif.stat().st_size}}
    image_map = ImageMap.load({"images": {shared: whole, own: {"image": str(sif)}}})
    assert [image.present for image in plan_images(lock, image_map).images] == [True, True]
    sif.write_bytes(b"a whole")
    assert [image.present for image in plan_images(lock, image_map).images] == [False, True]
    assert not has_image(image_map, shared) and has_image(image_map, own)


def test_verify_images_hashes_local_image_files(lock, tmp_path: Path):
    shared, own = (image.key for image in plan_images(lock).images)
    sif = tmp_path / "shared.sif"
    sif.write_bytes(b"image")
    entries = {
        shared: {"image": str(sif), "sha256": file_sha256(sif)},
        own: {"image": "docker://registry.example/own@sha256:" + "0" * 64},
    }

    def statuses(images: dict) -> list[tuple[str, str]]:
        return [(check.status, check.detail) for check in verify_images(lock, ImageMap.load({"images": images}))]

    not_checked = f"{entries[own]['image']} is not a local image file, so it is not checked"
    assert statuses(entries) == [("ok", f"{sif} matches its SHA-256"), ("unchecked", not_checked)]
    assert statuses({shared: {**entries[shared], "sha256": "sha256:" + file_sha256(sif)}})[0][0] == "ok"
    no_digest = f"{sif} has no SHA-256 in the image map, so it is not checked"
    assert statuses({shared: {"image": str(sif)}})[0] == ("unchecked", no_digest)
    sif.write_bytes(b"other")
    mismatch = f"{sif} has SHA-256 {file_sha256(sif)}, the image map says {entries[shared]['sha256']}"
    assert statuses(entries)[0] == ("missing", mismatch)
    sif.unlink()
    assert statuses(entries)[0] == ("missing", f"{sif} does not exist")
    assert statuses({}) == [("missing", "not in the image map")] * 2


def test_definition_files_translate_the_prefix_faithfully(tmp_path: Path):
    context = tmp_path / "environment"
    (context / "input").mkdir(parents=True)
    (context / "input" / "a.csv").write_text("1")
    (context / "run.sh").write_text("#!/bin/sh\n")
    (context / "Dockerfile").write_text(
        "ARG BASE=python:3.12-slim\nFROM $BASE\nARG MV=0.9.0\nENV PATH=/opt/bin:$PATH\nWORKDIR /workdir\n"
        "COPY input/ /data/input/\nCOPY run.sh /opt/bin/\nRUN pip install math-verify==$MV\n"
        'LABEL a=b\nEXPOSE 80\nRUN ["echo", "done"]\nCOPY late.txt /x\n'
    )
    (context / "late.txt").write_text("deferred")
    plan = plan_dockerfile(context / "Dockerfile")
    text, workdir = definition_file(plan, context)
    assert workdir == "/workdir"
    assert text.startswith("Bootstrap: docker\nFrom: python:3.12-slim\n")
    assert "%files\n    input /.platoon-files/0\n    run.sh /.platoon-files/1\n" in text
    post = text.split("%post\n", 1)[1].split("\n\n", 1)[0]
    assert post.splitlines() == [
        "    set -e",
        "    export MV=0.9.0",
        '    export PATH="/opt/bin:${PATH}"',
        "    mkdir -p /workdir && cd /workdir",
        "    mkdir -p /data/input/ && cp -a /.platoon-files/0/. /data/input//",
        "    mkdir -p /opt/bin/ && cp -a /.platoon-files/1 /opt/bin//run.sh",
        "    rm -rf /.platoon-files",  # once the last COPY has read its files
        "    ( cd /workdir && /bin/sh -c 'pip install math-verify==$MV' )",
        "    ( cd /workdir && echo done )",
    ]
    assert '%environment\n    export PATH="/opt/bin:${PATH}"\n' in text and "%labels\n    a b\n" in text
    assert "late.txt" not in text  # deferred: applied at session open
    pinned, _ = definition_file(plan, context, digests={"python:3.12-slim": "sha256:" + "f" * 64})
    assert pinned.startswith(f"Bootstrap: docker\nFrom: python@sha256:{'f' * 64}\n")


def _section(definition: str, name: str) -> str:
    body = definition.split(f"\n{name}\n", 1)[1].split("\n\n", 1)[0]
    return "\n".join(line.removeprefix("    ") for line in body.splitlines())


def test_definition_file_steps_run_as_in_a_docker_build(tmp_path: Path):
    context, work, out = tmp_path / "environment", tmp_path / "work", tmp_path / "out"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "ARG GREETING=hello\n"
        'ENV QUOTED="say \\"hi\\" `id` $(id) \\\\ \\$HOME" PATH=/opt/bin:$PATH WHO=$GREETING LEVEL=1\n'
        "ENV LEVEL=2 BEFORE=$LEVEL MODE=env\n"
        "ARG MODE=arg\n"
        f"WORKDIR {work}\n"
        f"RUN mkdir -p {out} && cd / && FOO=1 && export BAR=2\n"
        f'RUN pwd > {out}/pwd && echo "[$FOO$BAR] $GREETING $MODE" > {out}/run && printenv QUOTED > {out}/quoted\n'
        f'RUN [ -d {work} ] && echo "$PATH" > {out}/path\n'
        f'RUN ["sh", "-c", "pwd > {out}/exec"]\n'
    )
    text, _ = definition_file(plan_dockerfile(context / "Dockerfile"), context)
    quoted = 'say "hi" `id` $(id) \\ $HOME'  # literal: no command runs, no variable expands

    subprocess.run(["/bin/sh", "-c", _section(text, "%post")], env={"PATH": "/usr/bin:/bin"}, check=True)
    assert (out / "pwd").read_text() == f"{work}\n"  # the RUN before cd'd elsewhere, in a shell of its own
    assert (out / "run").read_text() == "[] hello env\n"  # its variables stay there; ARGs reach RUN, an ENV wins
    assert (out / "quoted").read_text() == f"{quoted}\n"
    assert (out / "path").read_text() == "/opt/bin:/usr/bin:/bin\n"  # a shell form that starts with [
    assert (out / "exec").read_text() == f"{work}\n"

    runtime = subprocess.run(
        ["/bin/sh", "-c", _section(text, "%environment") + "\nprintenv QUOTED PATH WHO LEVEL BEFORE MODE"],
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    # The ARG is substituted, since it does not reach the runtime; $PATH expands against the runtime's.
    assert runtime.stdout.splitlines() == [quoted, "/opt/bin:/usr/bin:/bin", "hello", "2", "1", "env"]


def test_definition_file_copies_each_find_their_staged_files(tmp_path: Path):
    context, stage, out = tmp_path / "environment", tmp_path / "stage", tmp_path / "out"
    (context / "input").mkdir(parents=True)
    (context / "input" / "a.csv").write_text("1")
    (context / "run.sh").write_text("#!/bin/sh\n")
    (context / "notes.txt").write_text("notes")
    (context / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        f"COPY input/ {out}/data/input/\n"
        f"COPY run.sh notes.txt {out}/opt/bin/\n"
        f"RUN ls {out}/opt/bin > {out}/listing\n"
    )
    text, _ = definition_file(plan_dockerfile(context / "Dockerfile"), context)

    stage.mkdir()
    for line in _section(text, "%files").splitlines():  # what the build's %files section puts in place
        source, target = shlex.split(line)
        staged = stage / Path(target).relative_to("/.platoon-files")
        if (context / source).is_dir():
            shutil.copytree(context / source, staged)
        else:
            shutil.copy2(context / source, staged)
    post = _section(text, "%post").replace("/.platoon-files", str(stage))
    subprocess.run(["/bin/sh", "-c", post], env={"PATH": "/usr/bin:/bin"}, check=True)

    assert (out / "data" / "input" / "a.csv").read_text() == "1"
    assert (out / "listing").read_text() == "notes.txt\nrun.sh\n"
    assert not stage.exists()  # removed once the last COPY has read its files


@pytest.mark.parametrize(
    ("dockerfile", "message"),
    [
        ("FROM a AS x\nRUN true\nFROM b\nRUN true\n", "multi-stage"),
        ("ARG V\nFROM x\nRUN echo $V\n", "no default"),
        ("FROM x\nUSER nobody\nRUN true\n", "USER"),
        ("FROM x\nRUN --mount=type=cache,target=/c true\n", "RUN flags"),
        ("FROM x\nRUN <<EOF\ntrue\nEOF\n", "here-documents"),
        (
            "FROM x\nADD https://example.com/a.tar.gz /a.tar.gz\n",
            r"line 2: ADD fetches the URL https://example\.com/a\.tar\.gz, which the def builder cannot do: build "
            r"this image with the oci builder \(--builder oci\)",
        ),
        (
            f"FROM x\nARG PDG=https://example.com/pkg\nADD --checksum=sha256:{'3e' * 32} ${{PDG}}/a.py /a.py\n",
            r"line 3: ADD fetches the URL https://example\.com/pkg/a\.py, .*--builder oci",
        ),
        (
            "FROM x\nENV REPO=https://github.com/org/repo.git\nADD ${REPO}#v1 /src\n",
            r"line 3: ADD fetches the git repository https://github\.com/org/repo\.git#v1, .*--builder oci",
        ),
    ],
)
def test_definition_files_refuse_what_they_cannot_translate(tmp_path: Path, dockerfile: str, message: str):
    (tmp_path / "Dockerfile").write_text(dockerfile)
    with pytest.raises(BuildError, match=message):
        definition_file(plan_dockerfile(tmp_path / "Dockerfile"), tmp_path)


def test_definition_files_say_that_no_builder_copies_remote_content(tmp_path: Path):
    # The plan refuses such a COPY first; a plan made some other way gets the same answer, not "use oci".
    (tmp_path / "Dockerfile").write_text("FROM x\nRUN true\n")
    plan = plan_dockerfile(tmp_path / "Dockerfile")
    copy = parse_dockerfile("FROM x\nARG U=https://example.com/x\nCOPY ${U} /x\n").instructions
    with pytest.raises(BuildError, match=r"^line 3: COPY cannot fetch the URL https://example\.com/x; only ADD can$"):
        definition_file(dataclasses.replace(plan, prefix=tuple(copy)), tmp_path)


FAKE_SINGULARITY = r"""#!/bin/bash
# Stand-in for singularity: an OCI-SIF of a squashfs layer, an image config and a manifest.
set -e
echo "$@" >> "$FAKE_LOG"
case "$1 $2" in
  "build --oci") test -f Dockerfile; cat Dockerfile >> "$FAKE_LOG"; echo built > "$3"; exit 0 ;;
  "sif list")
    echo 'ID |GROUP |LINK |SIF POSITION (start-end) |TYPE'
    echo '----'
    for id in 1 2 3; do echo "$id |1 |NONE |$id-$((id + 1)) |OCI.Blob"; done
    echo '4 |1 |NONE |4-5 |OCI.RootIndex'
    exit 0 ;;
  "sif dump")
    case "$3" in
      1) printf 'hsqs-layer-bytes' ;;
      2) printf '%s' '{"rootfs":{"type":"layers"},"config":{"Env":["PATH=/opt/extra/bin:/usr/bin","FOO=bar"],'
         printf '%s' '"WorkingDir":"/workdir","Cmd":["python3"]}}' ;;
      3) printf '{"schemaVersion":2,"layers":[]}' ;;
    esac
    exit 0 ;;
  "--version ") echo "singularity-ce version 4.4.1"; exit 0 ;;
esac
if [ "$1" = build ] && [ "$2" = --fakeroot ]; then
  test -f "$4"
  cat "$4" >> "$FAKE_LOG"
  case "$3" in */image.sif) ;; *) exit 3 ;; esac  # built in the work directory, then copied into place
  echo def > "$3"; exit 0
fi
if [ "$1" = build ]; then test -d "$3"; echo native > "$2"; exit 0; fi
if [ "$1" = exec ]; then printf 'PATH=/usr/local/bin:/usr/bin\0PWD=/x\0SINGULARITY_NAME=x\0LANG=C.UTF-8\0'; exit 0; fi
if [ "$1" = --version ]; then echo "singularity-ce version 4.4.1"; exit 0; fi
echo "unexpected: $*" >&2; exit 2
"""
FAKE_UNSQUASHFS = r"""#!/bin/bash
set -e
echo "unsquashfs $*" >> "$FAKE_LOG"
mkdir -p "$3/workdir"
"""


def _install(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def fake_singularity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand-ins for singularity and unsquashfs on PATH; returns the log of their calls (with each Dockerfile that
    singularity build --oci is given)."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", FAKE_SINGULARITY)
    _install(bin_dir, "unsquashfs", FAKE_UNSQUASHFS)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    return tmp_path / "calls.log"


def test_oci_builds_extract_a_native_sif_and_record_the_config(lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", FAKE_SINGULARITY)
    _install(bin_dir, "unsquashfs", FAKE_UNSQUASHFS)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    image_map = tmp_path / "site" / "images.json"
    built = build_images(
        lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "scratch", shard=(0, 1)
    )
    assert len(built) == 2
    loaded = ImageMap.load(image_map)
    shared = plan_images(lock, loaded).images[0]
    entry = loaded.get(shared.key)
    assert entry.image == str(tmp_path / "site" / "images" / f"{shared.key.removeprefix('sha256:')}.sif")
    assert Path(entry.image).read_text() == "native\n"
    assert entry.env == {"PATH": "/opt/extra/bin:/usr/bin", "FOO": "bar"} and entry.workdir == "/workdir"
    assert entry.cmd == ("python3",) and entry.builder == "oci" and len(entry.sha256) == 64
    assert entry.receipt["builder_version"] == "singularity-ce version 4.4.1"
    assert entry.receipt["task"] == "a" and Path(entry.receipt["log"]).is_file()
    calls = (tmp_path / "calls.log").read_text()
    assert "build --oci" in calls and "sif dump 1" in calls and "unsquashfs -no-xattrs -d" in calls
    assert plan_images(lock, ImageMap.load(image_map)).missing == []
    assert build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "scratch") == {}
    assert not list((tmp_path / "scratch").glob("build-*"))  # build trees are removed


# A separate verifier that fetches a file at build time, as nuclear-fusion-rl's b03, b04 and b31 do: through an ARG,
# checked by its SHA-256, in a build context without a .dockerignore.
FETCHING_TESTS = (
    "# syntax=docker/dockerfile:1.7\n"
    "FROM python:3.11.6-slim AS base\n"
    "ARG PDG=https://raw.githubusercontent.com/org/repo/0123456789abcdef0123456789abcdef01234567\n"
    f"ADD --checksum=sha256:{'3e' * 32} ${{PDG}}/pkg/__init__.py /opt/pkg/__init__.py\n"
    "COPY . /tests/\n"
    "RUN mkdir -p /verifier_data && mv /tests/hidden.json /verifier_data/ && chmod 700 /verifier_data\n"
    "ENV GRADER_DATA_DIR=/verifier_data\n"
)


def test_a_verifier_image_that_fetches_files_plans_and_builds_with_the_oci_builder(
    tmp_path: Path, fake_singularity: Path
):
    task = tmp_path / "dataset" / "fetching"
    for relative, text in {
        "task.toml": '[verifier]\nenvironment_mode = "separate"\n',
        "instruction.md": "Do it.\n",
        "environment/Dockerfile": SHARED,
        "environment/instance.json": "{}",
        "tests/Dockerfile": FETCHING_TESTS,
        "tests/test.sh": "echo 1\n",
        "tests/hidden.json": "{}",
    }.items():
        (task / relative).parent.mkdir(parents=True, exist_ok=True)
        (task / relative).write_text(text)
    lock = fetch(fetch_config(str(tmp_path / "dataset"), cache=tmp_path / "cache", lock=tmp_path / "l.json"))
    (row,) = lock.tasks
    # Before, the fetch refused it: "image: tests/Dockerfile: line 4: ADD source 'https://...' is missing".
    assert row.refused is None and row.verifier_image is not None
    pdg = "https://raw.githubusercontent.com/org/repo/0123456789abcdef0123456789abcdef01234567"
    assert row.verifier_image["deferred"] == [["ARG", "PDG", pdg], ["ENV", "GRADER_DATA_DIR", "/verifier_data"]]
    (verifier,) = [image for image in plan_images(lock).images if image.role == "verifier"]
    image_map = tmp_path / "site" / "images.json"

    def build(builder: str) -> dict[str, ImageEntry]:
        scratch = tmp_path / "scratch"
        return build_images(
            lock, cache=tmp_path / "cache", image_map=image_map, scratch=scratch, builder=builder, keys=[verifier.key]
        )

    with pytest.raises(BuildError) as refused:
        build("def")
    assert f"line 4: ADD fetches the URL {pdg}/pkg/__init__.py, which the def builder cannot do" in str(refused.value)
    assert "(--builder oci)" in str(refused.value)
    built = build("oci")

    entry = ImageMap.load(image_map).get(verifier.key)
    assert list(built) == [verifier.key] and entry is not None
    dockerfile = fake_singularity.read_text().split("build --oci", 1)[1]  # what singularity build --oci was given
    # BuildKit gets the step as written: it fetches the file and checks its checksum, and the key covers the text.
    assert f"ADD --checksum=sha256:{'3e' * 32} ${{PDG}}/pkg/__init__.py /opt/pkg/__init__.py\n" in dockerfile
    # SingularityCE's BuildKit cannot load the frontend image the directive names: its built-in frontend builds it.
    assert "# syntax=docker/dockerfile:1.7" not in dockerfile
    assert "platoon: left out '# syntax=docker/dockerfile:1.7'" in Path(entry.receipt["log"]).read_text()


@pytest.mark.parametrize(
    ("directive", "left_out"),
    [
        ("# syntax=docker/dockerfile:1.7", True),
        ("#syntax=docker.io/docker/dockerfile:1", True),
        (f"# syntax=docker/dockerfile:1.7.1@sha256:{'a' * 64}", True),
        ("# syntax=docker/dockerfile:1.7-labs", False),
        ("# syntax=example.com/frontends/dockerfile:2", False),
    ],
)
def test_oci_builds_build_docker_dockerfile_syntax_with_buildkits_own_frontend(
    tmp_path: Path, fake_singularity: Path, directive: str, left_out: bool
):
    task = tmp_path / "dataset" / "t"
    for relative, text in {
        "task.toml": "",
        "instruction.md": "Do it.\n",
        "tests/test.sh": "echo 1\n",
        "environment/Dockerfile": f"{directive}\nFROM python:3.12-slim\nRUN true\n",
    }.items():
        (task / relative).parent.mkdir(parents=True, exist_ok=True)
        (task / relative).write_text(text)
    lock = fetch(fetch_config(str(tmp_path / "dataset"), cache=tmp_path / "cache", lock=tmp_path / "l.json"))

    def build() -> dict[str, ImageEntry]:
        return build_images(lock, cache=tmp_path / "cache", image_map=tmp_path / "images.json", scratch=tmp_path / "s")

    if not left_out:
        with pytest.raises(BuildError, match=r"which the oci builder cannot load .* \(--builder docker\)"):
            build()
        assert not fake_singularity.exists()  # refused before singularity runs
        return
    (entry,) = build().values()
    dockerfile = fake_singularity.read_text().split("build --oci", 1)[1]
    assert directive not in dockerfile and "\nRUN true\n" in dockerfile
    assert f"platoon: left out {directive!r}" in Path(entry.receipt["log"]).read_text()


def test_shards_split_the_missing_keys(lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", FAKE_SINGULARITY)
    _install(bin_dir, "unsquashfs", FAKE_UNSQUASHFS)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    image_map = tmp_path / "images.json"
    first = build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", shard=(0, 2))
    second = build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", shard=(1, 2))
    assert len(first) == len(second) == 1 and set(first) != set(second)
    assert len(json.loads(image_map.read_text())["images"]) == 2


def test_a_failed_build_names_its_log(lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", "#!/bin/bash\necho 'FATAL: no space left' >&2\nexit 1\n")
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    with pytest.raises(BuildError, match="exited 1: FATAL: no space left .*log: "):
        build_images(lock, cache=tmp_path / "cache", image_map=tmp_path / "m.json", scratch=tmp_path / "s")
    assert not (tmp_path / "m.json").exists()


def test_def_builds_record_the_images_own_environment(lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", FAKE_SINGULARITY)
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    image_map = tmp_path / "images.json"
    built = build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", builder="def")
    shared = built[plan_images(lock).images[0].key]
    assert shared.builder == "def" and Path(shared.image).read_text() == "def\n" and shared.workdir is None
    assert shared.env == {"PATH": "/usr/local/bin:/usr/bin", "LANG": "C.UTF-8"}
    assert "build --fakeroot" in (tmp_path / "calls.log").read_text()


def test_def_image_environments_leave_out_the_variables_the_runtime_sets(tmp_path: Path):
    runtime = {
        "PWD": "/root",
        "OLDPWD": "/",
        "SHLVL": "1",
        "_": "/usr/bin/env",
        "PS1": "Singularity> ",
        "PROMPT_COMMAND": 'PS1="Singularity> "; unset PROMPT_COMMAND',
        "HOME": "/root",
        "USER": "root",
        "LOGNAME": "root",
        "TERM": "xterm",
        "HOSTNAME": "builder",
        "SINGULARITY_NAME": "image.sif",
        "APPTAINER_CONTAINER": "/images/image.sif",
        "LD_LIBRARY_PATH": "/.singularity.d/libs",
    }
    own = {"PATH": "/opt/conda/bin:/usr/bin", "LANG": "C.UTF-8", "PROMPT": "the image's own"}

    class Runner:
        log_path = tmp_path / "build.log"

        def run(self, argv: list[str]) -> str:
            assert argv[1:3] == ["exec", "--cleanenv"]
            return "".join(f"{name}={value}\0" for name, value in {**runtime, **own}.items())

    assert builders._sif_environment(Runner(), tmp_path / "image.sif") == own


def _fake_build_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put the stand-ins for singularity and unsquashfs in place; returns the log of their calls."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir, "singularity", FAKE_SINGULARITY)
    _install(bin_dir, "unsquashfs", FAKE_UNSQUASHFS)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PLATOON_SINGULARITY", str(bin_dir / "singularity"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    return tmp_path / "calls.log"


@pytest.mark.parametrize("builder", ["oci", "def"])
def test_builds_name_their_base_images_by_digest(
    lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: list[str], builder: str
):
    calls = _fake_build_tools(tmp_path, monkeypatch)
    image_map = tmp_path / "images.json"
    built = build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", builder=builder)
    assert set(built) == {image.key for image in plan_images(lock).images}  # the keys the lock was fetched with
    digest = _digest("python:3.12-slim")
    for key in built:
        receipt = ImageMap.load(image_map).get(key).receipt
        assert receipt["base"] == ["python:3.12-slim"] and receipt["base_digests"] == {"python:3.12-slim": digest}
    built_from = f"FROM python@{digest}\n" if builder == "oci" else f"From: python@{digest}\n"
    assert calls.read_text().count(built_from) == 2 and "python:3.12-slim" not in calls.read_text()
    assert registry == ["python:3.12-slim"]  # once for the two images that share it


def test_a_base_image_without_a_digest_builds_as_written_with_a_warning(
    lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    calls = _fake_build_tools(tmp_path, monkeypatch)
    asked: list[str] = []

    def offline(reference: str) -> str:
        asked.append(reference)
        raise DigestError("https://registry-1.docker.io/v2/library/python/manifests/3.12-slim: no network")

    image_map = tmp_path / "images.json"
    with caplog.at_level(logging.WARNING, logger=builders.__name__):
        built = build_images(
            lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", resolve_digest=offline
        )
    assert len(built) == 2 and asked == ["python:3.12-slim"]
    for key in built:
        assert ImageMap.load(image_map).get(key).receipt["base_digests"] == {"python:3.12-slim": None}
    assert calls.read_text().count("FROM python:3.12-slim\n") == 2
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings and all("Cannot resolve base image python:3.12-slim" in warning for warning in warnings)
    assert "no network" in warnings[0]


def test_a_build_refuses_a_task_whose_dockerfile_no_longer_plans_its_key(
    lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _fake_build_tools(tmp_path, monkeypatch)  # a build that went ahead would run only the stand-ins
    task = lock.get("c")
    key = task.image["key"]
    (tmp_path / "cache" / task.path / "environment" / "seed.py").write_text("print(4)\n")  # changed since the fetch
    image_map = tmp_path / "images.json"
    with pytest.raises(BuildError, match=f"environment/Dockerfile plans build key sha256:[0-9a-f]{{64}}, not {key}"):
        build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s", keys=[key])
    assert not image_map.exists() and not calls.exists()


def test_buildkit_gets_a_runtime_directory_its_socket_path_fits_in(tmp_path: Path):
    node_local = Path("/local/1234567.pbs1/platoon-images/run")  # a scheduler's node-local scratch: kept
    assert builders._runtime_dir(node_local) == node_local
    deep = tmp_path / ("d" * 40)  # singularity-buildkitd refuses a socket path beyond 104 bytes
    env = builders._build_env(deep)
    runtime = Path(env["XDG_RUNTIME_DIR"])
    assert runtime == Path("/tmp") / f"platoon-images-{os.getuid()}" and runtime.is_dir()
    assert len(str(runtime)) + len(builders._BUILDKIT_SOCKET) <= builders._SOCKET_PATH_MAX
    assert (runtime.stat().st_mode & 0o777) == 0o700
    assert Path(env["XDG_CONFIG_HOME"]).is_relative_to(deep)  # the rest stays in scratch


def test_builds_of_one_map_at_once_build_each_image_once(lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two builds of one image map at once, as two prepares on two nodes run them, take turns on each image: the one
    that waited finds the image in the map and builds nothing."""

    import threading
    import time

    calls: list[str] = []

    def build_image(task_dir: Path, key: str, **_: object) -> ImageEntry:
        calls.append(key)
        time.sleep(0.2)
        return ImageEntry(image="local", builder="oci")

    monkeypatch.setattr(builders, "build_image", build_image)
    image_map = tmp_path / "site" / "images.json"
    results: list[dict] = []

    def build() -> None:
        results.append(build_images(lock, cache=tmp_path / "cache", image_map=image_map, scratch=tmp_path / "s"))

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    keys = sorted(image.key for image in plan_images(lock).images)
    assert sorted(calls) == keys and sorted(key for built in results for key in built) == keys  # each built once
    assert plan_images(lock, ImageMap.load(image_map)).missing == []
    assert build_images(lock, cache=tmp_path / "cache", image_map=image_map, keys=keys[:1]) == {
        keys[0]: ImageEntry(image="local", builder="oci")
    }  # a key asked for by name is built again


def test_a_relative_image_map_records_its_images_where_it_finds_them(
    lock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An image map, or an image directory, named relative to the working directory (an entry's image_map, which runs
    resolve there, or --image-map and --out): the map finds the images built into it, though it resolves a relative
    image path against its own directory."""

    _fake_build_tools(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    for image_map, out_dir in ((Path("site/images.json"), None), (Path("other/map.json"), Path("sifs"))):
        built = build_images(lock, cache=tmp_path / "cache", image_map=image_map, out_dir=out_dir, scratch=Path("s"))
        assert len(built) == 2 and all(Path(entry.image).is_absolute() for entry in built.values())
        assert plan_images(lock, ImageMap.load(image_map)).missing == []
