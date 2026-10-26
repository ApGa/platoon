"""``platoon prepare``: a config's datasets fetched into their locks, the images their tasks build recorded in each
entry's image map, and the harnesses' toolchains built, each step a no-op once current, also when two prepares run at
once; ``--dry-run`` says what is missing and changes nothing; and ``platoon train``, ``eval`` and ``rollout`` with
``--prepare``, which run it first.

A fake Hugging Face hub serves a dataset shaped like a large Harbor dataset at a commit (``test_harbor_fetch``),
image builds write a file instead of running a builder, and a fake uv builds toolchains around this interpreter."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml
from site_cache import use_site_cache
from test_harbor_fetch import COMMIT, FakeHub, _overlay
from test_harbor_fetch import hub as fake_hub  # noqa: F401 - a fixture: the fake hub, serving the dataset

from platoon import cli
from platoon.envs.harbor import builders
from platoon.envs.harbor import fetch as fetch_module
from platoon.envs.harbor.images import ImageEntry, ImageMap, update_image_map
from platoon.envs.harbor.lock import read_lock
from platoon.harness.toolchains import read_manifest
from platoon.prepare import load_environments, prepare, prepare_command, summary
from platoon.utils.config import ConfigError

PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
FAKE_UV = """#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print("uv 0.0.0 (fake)")
elif args[:2] == ["python", "install"]:
    root = pathlib.Path(args[args.index("--install-dir") + 1]) / "cpython-{version}.99-linux-x86_64-gnu"
    (root / "bin").mkdir(parents=True)
    python3 = root / "bin" / "python3"
    python3.write_text('#!/bin/sh\\nexec {python} "$@"\\n')
    python3.chmod(0o755)
elif args[:2] == ["pip", "install"]:
    prefix = pathlib.Path(args[args.index("--python") + 1]).parent.parent
    (prefix / "installed.txt").write_text(pathlib.Path(args[args.index("-r") + 1]).read_text())
"""


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    code = exit_info.value.code
    assert isinstance(code, int)
    return code


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """A site cache of the test's own, ``site/`` (the Harbor cache ``site/harbor``, the toolchains
    ``site/toolchains``), a fake uv on the PATH and image builds that write a file, with what each was asked to do."""

    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(FAKE_UV.format(python=sys.executable, version=PYTHON))
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{uv.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_UV_LOG", str(tmp_path / "uv.log"))
    use_site_cache(monkeypatch, tmp_path / "site")
    builds: list = []

    def build_image(task_dir: Path, key: str, *, builder: str, out_dir: Path, **_: object) -> ImageEntry:
        builds.append((key, builder))
        out_dir.mkdir(parents=True, exist_ok=True)
        image = out_dir / f"{key.removeprefix('sha256:')}.sif"
        image.write_bytes(b"an image")
        return ImageEntry(image=str(image), builder=builder)

    monkeypatch.setattr(builders, "build_image", build_image)
    return {"builds": builds}


def _uv_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "uv.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def _spec(tmp_path: Path) -> Path:
    specs = tmp_path / "specs"
    specs.mkdir(exist_ok=True)
    (specs / "agent.txt").write_text("pkg==1.0 --hash=sha256:00\n")
    path = specs / "agent.toml"
    path.write_text(
        textwrap.dedent(
            f"""\
            name = "agent"
            version = "1.0"
            [install]
            toolchain = "agent-1.0"
            [install.recipe]
            python = "{PYTHON}"
            lock = "agent.txt"
            [launch]
            argv = ["{{toolchain}}/bin/python3", "-m", "agent"]
            """
        )
    )
    return path


def _config(tmp_path: Path, **service_kwargs) -> Path:
    """A training config with one Harbor entry on the fake dataset, at a commit, with a hold-out; it names no
    log_path, which its launcher would pass."""

    entry = {
        "name": "harbor-dataset",
        "service": "harbor",
        "service_kwargs": {
            "dataset": f"hf://org/harbor-dataset@{COMMIT}",
            "select": {
                "source_dataset": ["FineEnvs/SmolDataEnvs-harbor-train", "amphora/MathConstructOptimize-Envs-harbor"]
            },
            "overlays": [str(_overlay_once(tmp_path))],
            "index": [{"path": "data/tasks.jsonl", "fields": ["split_group"]}],
            "holdout": {"fraction": 0.5, "group_by": "split_group", "stratify_by": "source_dataset"},
            "provider": "singularity",
            **service_kwargs,
        },
        "harness": str(_spec(tmp_path)),
    }
    config = tmp_path / "configs" / "train.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(yaml.safe_dump({"environments": [entry], "train": {"model_name": "m"}}))
    return config


def _overlay_once(tmp_path: Path) -> Path:
    directory = tmp_path / "overlays" / "smoldata-run-before-copy"
    return directory if directory.is_dir() else _overlay(tmp_path)


def _run(config: Path, *overrides: str, dry_run: bool = False) -> tuple[list, list[str], int]:
    lines: list[str] = []
    steps = prepare(load_environments(config, overrides), dry_run=dry_run, out=lines.append, say=lambda line: None)
    line, code = summary(steps, dry_run=dry_run)
    return steps, [*lines, line], code


def test_prepare_fetches_builds_and_a_second_run_does_nothing(fake_hub: FakeHub, site, tmp_path: Path):  # noqa: F811
    config = _config(tmp_path)

    steps, lines, code = _run(config)

    assert code == 0 and [(step.step, step.status) for step in steps] == [
        ("dataset", "done"),
        ("images", "done"),
        ("toolchain agent-1.0", "done"),
    ]
    (lock_path,) = (tmp_path / "site" / "harbor" / "locks").iterdir()
    lock = read_lock(lock_path)
    default = tmp_path / "site" / "harbor" / "images" / "singularity.json"
    assert [builder for _, builder in site["builds"]] == ["oci", "oci"]  # SmolData's shared image, and MCO's
    entries = ImageMap.load(default).entries
    assert sorted(entries) == sorted(key for key, _ in site["builds"])
    assert {Path(entry.image).parent for entry in entries.values()} == {
        tmp_path / "site" / "harbor" / "images" / "singularity"
    }
    assert read_manifest(tmp_path / "site" / "toolchains" / "agent-1.0") is not None
    assert lines[0].startswith(f"environments[harbor-dataset] dataset: fetched hf://org/harbor-dataset@{COMMIT} into")
    assert lines[1].startswith("environments[harbor-dataset] hold-out: 50% of each source_dataset")  # its report
    assert f"built 2 with the oci builder into the image map {default}" in lines[-3]
    assert lines[-1] == "ready: 3 made now, 0 already current"
    assert len(lock.tasks) == len(lock.runnable()) == 6  # SmolData's 3 tasks and MathConstructOptimize's 3

    downloads, uv_calls = sum(fake_hub.downloads.values()), len(_uv_calls(tmp_path))

    def never(*args, **kwargs):
        raise AssertionError("prepare made something that was current")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("fetch", "file_sha256", "_sha256_and_blob_id"):
            patch.setattr(fetch_module, name, never)
        patch.setattr(builders, "build_image", never)
        steps, lines, code = _run(config)

    assert code == 0 and [step.status for step in steps] == ["current", "current", "current"]
    assert lines[-1] == "ready: 0 made now, 3 already current"
    assert sum(fake_hub.downloads.values()) == downloads and len(_uv_calls(tmp_path)) == uv_calls


def test_a_dry_run_says_what_is_missing_and_changes_nothing(fake_hub: FakeHub, site, tmp_path: Path):  # noqa: F811
    config = _config(tmp_path)

    steps, lines, code = _run(config, dry_run=True)

    assert code == 1 and [(step.step, step.status) for step in steps] == [
        ("dataset", "missing"),
        ("images", "missing"),
        ("toolchain agent-1.0", "missing"),
    ]
    assert "would fetch hf://org/harbor-dataset@" in lines[0] and "(" in lines[0] and "does not exist" in lines[0]
    assert "not planned: the dataset is not fetched yet" in lines[1]
    assert "would build it at" in lines[2] and "it is not built there" in lines[2]
    assert lines[-1] == "not ready: 3 step(s) would make something (a dry run changes nothing)"
    assert (
        not (tmp_path / "site" / "harbor").exists()
        and not (tmp_path / "site" / "toolchains").exists()
        and not site["builds"]
    )
    assert sum(fake_hub.downloads.values()) == 0

    _run(config)
    steps, lines, code = _run(config, dry_run=True)
    assert code == 0 and [step.status for step in steps] == ["current", "current", "current"]
    assert lines[-1] == "ready: 3 already current"


def test_two_prepares_at_once_make_each_thing_once(fake_hub: FakeHub, site, tmp_path: Path):  # noqa: F811
    """Two prepares at once, as on two nodes of one site, take turns: the dataset is fetched once, each image built
    once, the toolchain built once, and both end ready."""

    config = _config(tmp_path)
    resolved = load_environments(config)
    results: list[list] = []

    def run() -> None:
        results.append(prepare(resolved, out=lambda line: None, say=lambda line: None))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2 and all(summary(steps, dry_run=False)[1] == 0 for steps in results)
    assert len(site["builds"]) == 2 and len({key for key, _ in site["builds"]}) == 2  # each image once
    assert sum(call[:2] == ["python", "install"] for call in _uv_calls(tmp_path)) == 1  # the toolchain once
    statuses = sorted(tuple(step.status for step in steps) for steps in results)
    assert statuses[0][0] == "current" and statuses[1][0] == "done"  # one fetched, the other found its lock
    # a step that found everything made by the other prepare says so, as current, and counts nothing as made
    assert not any("built 0 " in step.detail for steps in results for step in steps)
    made = [step for steps in results for step in steps if step.status == "done"]
    assert len([step for step in made if step.step == "images"]) in (1, 2)  # one built both, or each built one


def test_a_truncated_image_is_built_again(fake_hub: FakeHub, site, tmp_path: Path):  # noqa: F811
    """An image file that is not the size its build recorded (a truncated copy) is missing: prepare builds it again,
    and the run's check refuses it until then."""

    from platoon.envs.harbor.images import has_image
    from platoon.envs.service import EnvNotPrepared
    from platoon.prepare import harbor_service

    def build_image(task_dir: Path, key: str, *, builder: str, out_dir: Path, **_: object) -> ImageEntry:
        site["builds"].append((key, builder))
        out_dir.mkdir(parents=True, exist_ok=True)
        image = out_dir / f"{key.removeprefix('sha256:')}.sif"
        image.write_bytes(b"a whole image")
        return ImageEntry(image=str(image), builder=builder, receipt={"size_bytes": image.stat().st_size})

    config = _config(tmp_path, provider_options={"allow_network_none": True})  # the tasks run without a network
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builders, "build_image", build_image)
        assert _run(config)[2] == 0
        default = tmp_path / "site" / "harbor" / "images" / "singularity.json"
        key, entry = sorted(ImageMap.load(default).entries.items())[0]
        Path(entry.image).write_bytes(b"a whole")  # truncated
        assert not has_image(ImageMap.load(default), key)
        service = harbor_service(load_environments(config).config.environments[0])
        assert service is not None
        names = asyncio.run(service.list_tasks("train"))
        refusal = re.escape(f"{default} lacks, or names by a file that is gone or truncated: 1 image(s), such as {key}")
        with pytest.raises(EnvNotPrepared, match=refusal):
            asyncio.run(service.check_tasks("train", names))  # the run's check
        steps, lines, code = _run(config)

    images = next(step for step in steps if step.step == "images")
    assert code == 0 and images.status == "done" and images.detail.startswith("built 1 with the oci builder")
    assert Path(entry.image).read_bytes() == b"a whole image" and has_image(ImageMap.load(default), key)


def test_prepare_never_replaces_a_toolchain_built_from_another_recipe(site, tmp_path: Path, capsys):
    """A toolchain built from another recipe (another platoon's spec, or the spec before an edit) may be what a live run
    stages its rollouts from: prepare leaves it as it is, and the step fails naming the forced build that replaces it,
    in a dry run too.  Once forced, prepare finds it current."""

    spec = _spec(tmp_path)
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id": "t1"}\n')
    config = tmp_path / "configs" / "table.yaml"
    config.parent.mkdir()
    entry = {"service": "table", "service_kwargs": {"path": str(tasks)}, "harness": str(spec)}
    config.write_text(yaml.safe_dump({"environments": [entry]}))
    assert _run(config)[2] == 0
    toolchain = tmp_path / "site" / "toolchains" / "agent-1.0"
    built = read_manifest(toolchain)
    calls = len(_uv_calls(tmp_path))
    (spec.parent / "agent.txt").write_text("pkg==2.0 --hash=sha256:11\n")  # another recipe for the same name

    for dry_run in (True, False):
        steps, lines, code = _run(config, dry_run=dry_run)
        assert code == 1 and [step.status for step in steps] == ["failed"]
        assert "was built from another recipe (lock sha256" in lines[0]
        force = f"platoon toolchain build {spec} --root {tmp_path / 'site' / 'toolchains'} --force"
        assert lines[0].endswith(f"rebuild it in place, while no job uses it, with: {force}")
    assert read_manifest(toolchain) == built and len(_uv_calls(tmp_path)) == calls  # left as it was

    cli.main(["toolchain", "build", str(spec), "--root", str(tmp_path / "site" / "toolchains"), "--force"])
    capsys.readouterr()
    steps, lines, code = _run(config)
    assert code == 0 and [step.status for step in steps] == ["current"]


def test_the_process_provider_records_its_images_as_local(fake_hub: FakeHub, site, tmp_path: Path):  # noqa: F811
    config = _config(tmp_path, provider="process")

    steps, lines, code = _run(config)

    assert code == 0 and steps[1].status == "done" and not site["builds"]
    process = ImageMap.load(tmp_path / "site" / "harbor" / "images" / "process.json")
    assert {entry.image for entry in process.entries.values()} == {"local"} and len(process.entries) == 2
    assert "recorded 2 of the 2 image(s) the runnable tasks build as local" in lines[-3]


def test_a_provider_that_pulls_from_a_registry_names_the_command_that_builds_and_pushes(
    fake_hub: FakeHub,  # noqa: F811
    site,
    tmp_path: Path,
    capsys,
):
    config = _config(tmp_path)
    override = "environments.0.service_kwargs.provider=enroot"

    assert _exit_code(["prepare", str(config), override]) == 1

    out = capsys.readouterr().out
    command = f"platoon harbor images build {config} {override} --builder docker --push REGISTRY/NAME"
    assert (
        f"pulls its images from a registry, which prepare does not guess: build and push them with `{command}`" in out
    )
    assert f"then run `{prepare_command(config, [override])}` again" in out
    assert out.rstrip().endswith("not ready: 1 step(s) failed (environments[harbor-dataset] images)")
    assert not site["builds"]


def test_an_entry_that_serves_a_directory_fetches_and_builds_nothing(site, tmp_path: Path, capsys):
    """A dataset directory is not fetched, and its tasks run their own images (no default image map, I-R10); its
    harness's toolchain is still built."""

    task = tmp_path / "tasks" / "hello"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\nRUN true\n")
    (task / "task.toml").write_text("")
    (task / "instruction.md").write_text("Say hello.\n")
    config = tmp_path / "configs" / "dir.yaml"
    config.parent.mkdir()
    entry = {
        "service": "harbor",
        "service_kwargs": {"dataset_dir": str(tmp_path / "tasks")},
        "harness": str(_spec(tmp_path)),
    }
    config.write_text(yaml.safe_dump({"environments": [entry]}))

    cli.main(["prepare", str(config)])

    out = capsys.readouterr().out.splitlines()
    assert out[0] == "environments[default] dataset: serves a directory, which is not fetched"
    assert out[1].startswith("environments[default] toolchain agent-1.0: built at")
    assert out[-1] == "ready: 1 made now, 0 already current" and not site["builds"]


def test_prepare_refuses_the_images_of_root_verifiers_that_the_oci_builder_made_before_0_3_0(site, tmp_path: Path):
    """With verifier_user: honor, an image that a verifier runs as root on, and whose receipt records no oci builder
    revision (as the fake builds here, and the oci builder's before 0.3.0), may keep its files from the users the tests
    switch to, and prepare replaces no image: the step fails, naming the build that makes it again.  Built again, it
    passes; the task's own image, which only the agents run, never mattered."""

    task = tmp_path / "dataset" / "rooted"
    files = {
        "task.toml": '[agent]\nuser = "nobody"\n[verifier]\nenvironment_mode = "separate"\nuser = "root"\n',
        "instruction.md": "Do the task.\n",
        "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        "tests/Dockerfile": "FROM python:3.12-slim\nRUN echo grader > /grader.txt\n",
        "environment/Dockerfile": "FROM python:3.12-slim\nRUN echo built > /built.txt\n",
    }
    for relative, text in files.items():
        (task / relative).parent.mkdir(parents=True, exist_ok=True)
        (task / relative).write_text(text)
    config = tmp_path / "configs" / "rooted.yaml"
    config.parent.mkdir()
    entry = {
        "name": "rooted",
        "service": "harbor",
        "service_kwargs": {
            "dataset": str(tmp_path / "dataset"),
            "refuse": {"agent_user": False},
            "agent_user": "ignore",
            "verifier_user": "honor",
            "provider": "singularity",
            "verifier_provider_options": {"rootfs_layout": "copy", "fakeroot": True},
        },
        "harness": str(_spec(tmp_path)),
    }
    config.write_text(yaml.safe_dump({"environments": [entry]}))

    steps, lines, code = _run(config)

    (lock_path,) = (tmp_path / "site" / "harbor" / "locks").iterdir()
    verifier_key = read_lock(lock_path).get("rooted").verifier_image["key"]
    (images,) = [step for step in steps if step.step == "images"]
    assert code == 1 and images.status == "failed" and len(site["builds"]) == 2, images.detail
    assert images.detail.startswith("built 2 with the oci builder")
    assert f"build them again with `platoon harbor images build CONFIG --key {verifier_key}`" in images.detail
    assert lines[-1] == "not ready: 1 step(s) failed (environments[rooted] images)"

    default = tmp_path / "site" / "harbor" / "images" / "singularity.json"
    built = ImageMap.load(default).entries[verifier_key]
    again = ImageEntry(image=built.image, builder="oci", receipt={"builder_revision": 1})  # built since
    update_image_map(default, {verifier_key: again})
    steps, lines, code = _run(config)
    assert code == 0 and [step.status for step in steps if step.step == "images"] == ["current"], lines


def test_a_moving_reference_is_prepared_only_once_pinned(fake_hub: FakeHub, site, tmp_path: Path, capsys):  # noqa: F811
    """A dataset named by a branch moves: prepare never resolves it, but uses the lock that platoon harbor fetch
    --update wrote for it."""

    config = _config(tmp_path, dataset="hf://org/harbor-dataset@main")

    steps, lines, code = _run(config, dry_run=True)
    assert steps[0].status == "missing" and "is a moving reference, which no current lock answers" in lines[0]
    assert _exit_code(["prepare", str(config)]) == 78
    assert "pin a commit, or resolve it with `platoon harbor fetch CONFIG --update`" in capsys.readouterr().err

    cli.main(["harbor", "fetch", str(config), "--update"])
    capsys.readouterr()
    steps, lines, code = _run(config)
    assert code == 0 and steps[0].status == "current" and lines[0].endswith("6 task(s), 6 runnable")


def test_configuration_problems_exit_78(fake_hub: FakeHub, site, tmp_path: Path, capsys):  # noqa: F811
    config = _config(tmp_path)
    assert _exit_code(["prepare", str(config), "environments.0.service_kwargs.holdout.fraction=2"]) == 78
    assert "holdout.fraction must be a number between 0 and 1" in capsys.readouterr().err
    assert _exit_code(["prepare", str(config), "environments.0.harness=nope"]) == 78
    assert _exit_code(["prepare", str(config), 'environments.0.service_kwargs.image_map={"images": {}}']) == 78
    assert "environments.0.service_kwargs.image_map: unknown key; platoon 0.3.0 removed it" in capsys.readouterr().err
    (tmp_path / "configs" / "empty.yaml").write_text("train: {model_name: m}\n")
    assert _exit_code(["prepare", str(tmp_path / "configs" / "empty.yaml")]) == 78
    assert "lists no environments" in capsys.readouterr().err


def test_the_policys_tokenizer_is_loaded_through_its_renderer(site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The tokenizer step loads the model of the config's train (or inference) section through its renderer, as the
    run's processes will, into the Hugging Face cache, and is current once a load finds everything there; without a
    renderer, or for a hosted model, there is none."""

    from huggingface_hub import constants

    proxy = pytest.importorskip("platoon.train.tinker.proxy")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    loaded: list[tuple] = []

    def build_renderer(*args) -> None:  # downloads the tokenizer's file once, as a load does
        loaded.append(args)
        blobs = tmp_path / "hub" / "models--org--model" / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        (blobs / "tokenizer-config").write_text("{}")

    monkeypatch.setattr(proxy, "build_renderer", build_renderer)
    config = tmp_path / "configs" / "eval.yaml"
    config.parent.mkdir()
    entry = {"service": "table", "service_kwargs": {"path": str(tmp_path / "tasks.jsonl")}}
    inference = {"model_name": "org/model", "renderer_name": "chat_template", "renderer_kwargs": {"x": 1}}
    config.write_text(yaml.safe_dump({"environments": [entry], "inference": inference}))

    assert _run(config, dry_run=True)[1][-2] == (
        "policy tokenizer: would load org/model through the chat_template renderer into the Hugging Face cache"
    )
    steps, lines, code = _run(config)
    assert code == 0 and loaded == [("org/model", "chat_template", {"x": 1})] and steps[-1].status == "done"
    assert (
        lines[-2]
        == "policy tokenizer: loaded org/model through the chat_template renderer, into the Hugging Face cache"
    )
    steps, lines, code = _run(config)
    assert steps[-1].status == "current" and lines[-1] == "ready: 0 made now, 1 already current"
    assert _run(config, "inference.hosted={base_url: http://h, api_key_env: K}")[0] == []

    def fail(*args):
        raise OSError("no access to org/model")

    monkeypatch.setattr(proxy, "build_renderer", fail)
    steps, lines, code = _run(config)
    assert code == 1 and steps[-1].status == "failed" and "OSError: no access to org/model" in lines[-2]


def test_a_renderer_setting_that_the_renderer_refuses_is_a_configuration_error(
    site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A tool-call parser, or a renderer_kwargs key, that the renderer refuses is a configuration error (exit 78), as
    when a run starts, and not a tokenizer step that failed (exit 1), which a launcher would start again; a renderer
    that does not load for another reason still fails the step."""

    proxy = pytest.importorskip("platoon.train.tinker.proxy")
    from platoon.train.tinker.renderers import ToolCallParserError

    def build_renderer(model: str, renderer: str, kwargs: dict) -> None:
        if "tool_call_parser" in kwargs:
            raise ToolCallParserError(f"no tool-call parser is named {kwargs['tool_call_parser']!r}")
        if kwargs:
            raise ValueError(f"Renderer '{renderer}' does not support renderer_kwargs['{next(iter(kwargs))}']")
        raise ValueError(f"Unknown renderer: {renderer}")

    monkeypatch.setattr(proxy, "build_renderer", build_renderer)
    config = tmp_path / "configs" / "eval.yaml"
    config.parent.mkdir()
    entry = {"service": "table", "service_kwargs": {"path": str(tmp_path / "tasks.jsonl")}}
    inference = {"model_name": "org/model", "renderer_name": "chat_template", "renderer_kwargs": {"x": 1}}
    config.write_text(yaml.safe_dump({"environments": [entry], "inference": inference}))

    parser = r"inference\.renderer_kwargs\.tool_call_parser: no tool-call parser is named 'nope'"
    with pytest.raises(ConfigError, match=parser):
        _run(config, "inference.renderer_kwargs={tool_call_parser: nope}")
    unknown = r"inference\.renderer_kwargs: Renderer 'chat_template' does not support renderer_kwargs\['x'\]"
    with pytest.raises(ConfigError, match=unknown):
        _run(config)
    del inference["renderer_kwargs"]
    config.write_text(yaml.safe_dump({"environments": [entry], "inference": inference}))
    steps, lines, code = _run(config)
    assert code == 1 and steps[-1].status == "failed" and "ValueError: Unknown renderer: chat_template" in lines[-2]


def _run_config(tmp_path: Path, kind: str) -> Path:
    """A training (``train``) or evaluation (``eval``) config on _config's entry, with the process provider, which
    platoon train or eval start on and refuse when their tasks' images or the harness's toolchain are not made."""

    data = yaml.safe_load(_config(tmp_path, provider="process").read_text())
    if kind == "train":
        data["train"] = {"model_name": "m", "renderer_name": "qwen3", "batch_size": 1, "max_training_steps": 1}
        data["eval"] = {"every": 0}
        data["stats"] = {"experiment_name": "exp", "trial_name": "trial"}
    else:
        del data["train"]
        data["inference"] = {"model_name": "m", "renderer_name": "qwen3", "output_dir": str(tmp_path / "eval")}
        data["split"] = "train"  # the fake dataset's hold-out holds out nothing runnable
    config = tmp_path / "configs" / f"{kind}.yaml"
    config.write_text(yaml.safe_dump(data))
    return config


def test_train_and_eval_refuse_what_prepare_has_not_made_with_the_prepare_command(
    fake_hub: FakeHub,  # noqa: F811
    site,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    """Before any rollout, and before Tinker opens: a toolchain that is not built stops the run before its tasks are
    listed (nothing is fetched), and a task image the map lacks once they are; each with exit 78 and the platoon
    prepare command for the config the run was given, overrides included."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-not-a-key")
    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", lambda *args: None)  # prepare's tokenizer step
    train = _run_config(tmp_path, "train")
    log_path = f"log_path={tmp_path / 'runs'}"

    assert _exit_code(["train", str(train), log_path]) == 78
    err = capsys.readouterr().err
    assert "what the run needs is not prepared on this site:" in err and "toolchain agent-1.0, which a run never" in err
    assert err.rstrip().endswith(f"prepare it with: {prepare_command(train, [log_path])}")
    assert sum(fake_hub.downloads.values()) == 0  # refused before its tasks were listed, so nothing was fetched

    cli.main(["prepare", str(train)])
    capsys.readouterr()
    default = tmp_path / "site" / "harbor" / "images" / "process.json"
    images = json.loads(default.read_text())
    key = sorted(images["images"])[0]
    del images["images"][key]  # one image of the two missing again
    default.write_text(json.dumps(images))

    assert _exit_code(["train", str(train), log_path]) == 78
    err = capsys.readouterr().err
    assert f"build their images from Dockerfiles, which the image map {default} lacks: 1 image(s), such as {key}" in err
    assert err.rstrip().endswith(f"prepare it with: {prepare_command(train, [log_path])}")

    evaluation = _run_config(tmp_path, "eval")
    assert _exit_code(["eval", str(evaluation)]) == 78
    err = capsys.readouterr().err
    assert f"which the image map {default} lacks" in err
    assert err.rstrip().endswith(f"prepare it with: {prepare_command(evaluation)}")

    cli.main(["prepare", str(evaluation)])
    assert capsys.readouterr().out.splitlines()[-1] == "ready: 2 made now, 2 already current"  # the image again


def test_rollout_refuses_a_dataset_that_is_not_fetched(
    fake_hub: FakeHub,  # noqa: F811
    site,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    """platoon rollout, which runs one task and never fetches, stops with the prepare command when no lock answers the
    entry, and when the task's image is missing."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-not-a-key")
    data = yaml.safe_load(_run_config(tmp_path, "train").read_text())
    data["environments"][0]["harness"] = "platoon-noop"  # no toolchain
    data["log_path"] = str(tmp_path / "runs")
    config = tmp_path / "configs" / "rollout.yaml"
    config.write_text(yaml.safe_dump(data))
    task = "fineenvs--smoldataenvs-harbor-train--task-0"
    rollout = ["rollout", str(config), "--task", task, "--output-dir", str(tmp_path / "run")]

    assert _exit_code(rollout) == 78
    err = capsys.readouterr().err
    assert "does not exist: run `platoon prepare` on the run's config" in err
    assert err.rstrip().endswith(f"prepare it with: {prepare_command(config)}")

    cli.main(["harbor", "fetch", str(config)])  # the dataset, not its images
    capsys.readouterr()
    assert _exit_code(rollout) == 78
    assert "build their images from Dockerfiles, which no image map holds" in capsys.readouterr().err


# --- --prepare: a run that prepares first ------------------------------------------------------------------------

ROLLOUT_TASK = "fineenvs--smoldataenvs-harbor-train--task-0"


class Reached(Exception):
    """What the stand-in for the trainer raises: training passed every check of the run and would open Tinker now."""


@dataclass(frozen=True)
class _Policy:
    """Stands in for the policy platoon rollout serves, whose install would start a Tinker sampler."""

    model_name: str
    renderer_name: str | None = None
    model_path: str | None = None
    renderer_kwargs: dict = field(default_factory=dict)
    context_window_length: int | None = None
    base_url: str | None = None

    def install(self) -> None:
        pass


@pytest.fixture
def opened(site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    """Stand-ins for what each run opens once its checks pass, with what reached them: the trainer (which raises
    Reached with the training tasks it was given), the evaluation's runner, and a rollout's policy and route.  Also a
    Tinker key, and a tokenizer load that puts the model's file in the test's own Hugging Face cache, so that the
    next prepare finds it there."""

    from huggingface_hub import constants

    from platoon.train.auto import AutoRollout

    monkeypatch.setenv("TINKER_API_KEY", "tml-not-a-key")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def build_renderer(model: str, *args) -> None:
        blob = tmp_path / "hub" / f"models--{model}" / "blobs" / "tokenizer"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text("{}")

    reached: dict[str, list] = {"train": [], "eval": [], "rollout": []}

    def trainer(*, config, train_dataset, eval_dataset, train_order=None, train_sampling=None):
        reached["train"].append(len(train_dataset))
        raise Reached

    class Runner:
        def __init__(self, *, workflow, output_dir) -> None:
            pass

        async def arun(self, *, dataset, **kwargs) -> dict:
            reached["eval"].append(len(dataset))
            return {"summary": {"total_tasks": len(dataset)}}

    def route(settings, split):
        def rollout(task, config) -> dict:
            reached["rollout"].append(task.id)
            root = {"id": "root", "task": {"id": task.id}, "parent_info": None, "reward": 1.0, "steps": [{}]}
            return {"id": "rollout-1", "trajectories": {"root": root}}

        return rollout

    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", build_renderer)
    monkeypatch.setattr("platoon.train.launch.PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setattr("platoon.eval.run.EvalRunner", Runner)
    monkeypatch.setattr(cli, "_eval_policy_class", lambda: _Policy)
    monkeypatch.setattr(AutoRollout, "from_config", route)
    return reached


def _prepare_lines(text: str) -> list[str]:
    """The lines of prepare's steps and its verdict in a command's output; its fetch and build logs aside."""

    return [line for line in text.splitlines() if line.startswith(("environments[", "policy ", "ready:", "not ready:"))]


def test_train_with_prepare_prepares_then_trains_and_a_second_run_prepares_nothing(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    capsys,
):
    """platoon train --prepare first runs what platoon prepare runs on its config and overrides, with prepare's lines
    on stderr, and then trains, past every check that refuses the same run without the flag.  A second run's
    prepare makes nothing, and prints what platoon prepare prints then, line for line."""

    config = _run_config(tmp_path, "train")
    log_path = f"log_path={tmp_path / 'runs'}"
    assert _exit_code(["train", str(config), log_path]) == 78  # without the flag nothing is made: refused
    capsys.readouterr()

    with pytest.raises(Reached):
        cli.main(["train", str(config), "--prepare", log_path])

    out, err = capsys.readouterr()
    lines = _prepare_lines(err)
    assert out == "" and [line.split(":")[0] for line in lines] == [
        "environments[harbor-dataset] dataset",
        "environments[harbor-dataset] hold-out",
        "environments[harbor-dataset] images",
        "environments[harbor-dataset] toolchain agent-1.0",
        "policy tokenizer",
        "ready",
    ]
    assert lines[0].startswith(f"environments[harbor-dataset] dataset: fetched hf://org/harbor-dataset@{COMMIT} into")
    assert lines[-1] == "ready: 4 made now, 0 already current"
    default = tmp_path / "site" / "harbor" / "images" / "process.json"
    assert len(ImageMap.load(default).entries) == 2 and read_manifest(tmp_path / "site" / "toolchains" / "agent-1.0")
    assert len(opened["train"]) == 1 and opened["train"][0] > 0  # the run listed its tasks and reached the trainer

    cli.main(["prepare", str(config), log_path])
    expected = capsys.readouterr().out
    assert expected.splitlines()[-1] == "ready: 0 made now, 4 already current"
    downloads, uv_calls = sum(fake_hub.downloads.values()), len(_uv_calls(tmp_path))

    def never(*args, **kwargs):
        raise AssertionError("prepare made something that was current")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("fetch", "file_sha256", "_sha256_and_blob_id"):
            patch.setattr(fetch_module, name, never)
        patch.setattr(builders, "build_image", never)
        with pytest.raises(Reached):
            cli.main(["train", str(config), "--prepare", log_path])

    out, err = capsys.readouterr()
    assert out == "" and expected in err  # platoon prepare's stdout, on stderr
    assert sum(fake_hub.downloads.values()) == downloads and len(_uv_calls(tmp_path)) == uv_calls
    assert len(opened["train"]) == 2


def test_the_overrides_of_a_run_with_prepare_reach_its_prepare_and_the_run(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    capsys,
):
    """The overrides a run with --prepare is given reach its prepare, which makes everything where they say, and the
    run, which finds it there (its checks would refuse it anywhere else) and records them."""

    config = _run_config(tmp_path, "train")
    lock = tmp_path / "elsewhere" / "dataset.lock.json"
    overrides = [
        f"log_path={tmp_path / 'runs'}",
        f"environments.0.service_kwargs.lock={lock}",
        "stats.trial_name=prepared",
    ]

    with pytest.raises(Reached):
        cli.main(["train", str(config), "--prepare", *overrides])

    assert _prepare_lines(capsys.readouterr().err)[-1] == "ready: 4 made now, 0 already current"
    assert len(read_lock(lock).tasks) == 6  # where the override says, which the run read
    assert not list((tmp_path / "site" / "harbor" / "locks").glob("*.lock.json"))  # not the entry's default lock
    snapshot = yaml.safe_load((tmp_path / "runs" / "exp" / "prepared" / "config.resolved.yaml").read_text())
    assert snapshot["environments"][0]["service_kwargs"]["lock"] == str(lock)
    assert len(opened["train"]) == 1


def test_eval_and_rollout_with_prepare_keep_stdout_for_what_they_print(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    capsys,
):
    """platoon eval and rollout with --prepare prepare first too, with prepare's lines on stderr, so their stdout
    holds what it holds without the flag: the evaluation's JSON summary, and the rollout's summary."""

    cli.main(["eval", str(_run_config(tmp_path, "eval")), "--prepare"])

    out, err = capsys.readouterr()
    assert json.loads(out) == {"total_tasks": opened["eval"][0]} and opened["eval"][0] > 0
    assert _prepare_lines(err)[-1] == "ready: 4 made now, 0 already current"

    config = _run_config(tmp_path, "train")  # the same entry: what the evaluation's prepare made is current
    run_dir = tmp_path / "rollout"
    argv = ["rollout", str(config), "--task", ROLLOUT_TASK, "--output-dir", str(run_dir), "--json-output"]
    cli.main([*argv, "--prepare", f"log_path={tmp_path / 'runs'}"])

    out, err = capsys.readouterr()
    assert json.loads(out)["task"] == opened["rollout"][0] and ROLLOUT_TASK in opened["rollout"][0]
    assert _prepare_lines(err)[-1] == "ready: 0 made now, 4 already current"
    assert (run_dir / "trajectory_collection.json").is_file()


def test_eval_with_prepare_prepares_nothing_for_a_report(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    capsys,
):
    """An evaluation with stage=report builds its report from finished rollouts and needs nothing that prepare makes:
    --prepare says it is ignored and prepares nothing, so it neither delays the report nor fails it, as prepare would
    here (a registry provider, whose images prepare does not build)."""

    config = _run_config(tmp_path, "eval")

    cli.main(["eval", str(config), "stage=report", "--prepare", "environments.0.service_kwargs.provider=enroot"])

    out, err = capsys.readouterr()
    assert json.loads(out) == {"total_tasks": 0} and opened["eval"] == [0]
    assert "platoon: --prepare is ignored: stage=report runs no rollout" in err and not _prepare_lines(err)
    assert sum(fake_hub.downloads.values()) == 0


def test_rollout_with_prepare_loads_the_tokenizer_of_the_model_it_samples(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    """platoon rollout's --model and --renderer replace the config's model and renderer, so its prepare loads their
    tokenizer, which the rollout loads, and not the config's, which this machine may not be able to load."""

    loaded: list[tuple[str, str]] = []

    def build_renderer(model: str, renderer: str, *args) -> None:
        loaded.append((model, renderer))
        if model == "m":
            raise OSError("401 Client Error: m is a gated repository")
        blob = tmp_path / "hub" / f"models--{model.replace('/', '--')}" / "blobs" / "tokenizer"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text("{}")

    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", build_renderer)
    config = _run_config(tmp_path, "train")
    run_dir = tmp_path / "rollout"
    argv = ["rollout", str(config), "--task", ROLLOUT_TASK, "--output-dir", str(run_dir), "--json-output"]
    flags = ["--model", "org/other", "--renderer", "chat_template", "--prepare"]

    cli.main([*argv, *flags, f"log_path={tmp_path / 'runs'}"])

    out, err = capsys.readouterr()
    assert json.loads(out)["task"] == opened["rollout"][0]
    lines = _prepare_lines(err)
    assert "policy tokenizer: loaded org/other through the chat_template renderer, into the Hugging Face cache" in lines
    assert lines[-1] == "ready: 4 made now, 0 already current"
    assert loaded[0] == ("org/other", "chat_template") and all(model != "m" for model, _ in loaded)


@pytest.mark.parametrize("command", ["train", "eval", "rollout"])
def test_a_prepare_that_does_not_end_ready_stops_the_run_with_its_exit_status(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    command: str,
):
    """When the prepare of a run with --prepare fails, the run does not start, nothing is written in its run
    directory, and the command exits with prepare's status: 1 for a step that failed, 78 for a configuration
    error, also one that only prepare's last step finds, the tokenizer's, which the run without the flag refuses
    with 78 too."""

    log_path = f"log_path={tmp_path / 'runs'}"
    if command == "eval":
        config, overrides, run_dir = _run_config(tmp_path, "eval"), [], tmp_path / "eval"
        argv = ["eval", str(config)]
    else:
        config, overrides = _run_config(tmp_path, "train"), [log_path]
        run_dir = tmp_path / "runs" / "exp" / "trial" if command == "train" else tmp_path / "rollout"
        argv = [command, str(config)]
        if command == "rollout":
            argv += ["--task", ROLLOUT_TASK, "--output-dir", str(run_dir)]
    registry = "environments.0.service_kwargs.provider=enroot"  # pulls its images from a registry: the step fails

    assert _exit_code([*argv, "--prepare", *overrides, registry]) == 1

    err = capsys.readouterr().err
    assert _prepare_lines(err)[-1] == "not ready: 1 step(s) failed (environments[harbor-dataset] images)"
    assert f"then run `{prepare_command(config, [*overrides, registry])}` again" in err
    assert not opened[command] and not run_dir.exists()

    moving = "environments.0.service_kwargs.dataset=hf://org/harbor-dataset@main"  # a configuration error
    assert _exit_code([*argv, "--prepare", *overrides, moving]) == 78
    assert "is a moving reference, which no current lock answers" in capsys.readouterr().err
    assert not opened[command] and not run_dir.exists()

    from platoon.train.tinker.renderers import ToolCallParserError

    def build_renderer(*args) -> None:  # the renderer has no such tool-call parser: a configuration error
        raise ToolCallParserError("no tool-call parser is named 'nope'")

    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", build_renderer)
    assert _exit_code([*argv, "--prepare", *overrides]) == 78
    err = capsys.readouterr().err
    assert ".renderer_kwargs.tool_call_parser: no tool-call parser is named 'nope'" in err
    assert not opened[command] and not run_dir.exists()


def test_a_supervised_run_with_prepare_prepares_once_before_its_first_start(
    fake_hub: FakeHub,  # noqa: F811
    opened,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
):
    """With watchdog.max_restarts, platoon train --prepare prepares once, in platoon train, before the first start.
    Each start, platoon.train.supervise's main in a process of its own, runs the run's checks and reaches the trainer
    without preparing, and the restarts after two failed starts prepare nothing either."""

    from platoon.train import supervise

    calls, starts = tmp_path / "prepares", tmp_path / "starts"
    prepare_in_train = cli._prepare

    def counting(*args, **kwargs):
        with calls.open("a") as log:
            log.write("platoon train\n")
        return prepare_in_train(*args, **kwargs)

    start = tmp_path / "start.py"  # the real start (supervise.main) with a prepare that tells, and a stand-in trainer
    start.write_text(
        textwrap.dedent(
            f"""\
            import sys
            from pathlib import Path

            import platoon.prepare
            import platoon.train.launch as training
            from platoon.train import supervise

            def prepare(*args, **kwargs):
                with open({str(calls)!r}, "a") as log:
                    log.write("a start\\n")
                raise AssertionError("a start prepared")

            def trainer(*, config, train_dataset, eval_dataset, train_order=None, train_sampling=None):
                path = Path({str(starts)!r})
                n = int(path.read_text()) + 1 if path.exists() else 1
                path.write_text(str(n))
                if n < 3:
                    raise RuntimeError(f"start {{n}} fails after the run's checks, with {{len(train_dataset)}} tasks")
                raise SystemExit(0)

            platoon.prepare.prepare = prepare
            training.PlatoonTinkerRLTrainer = trainer
            sys.exit(supervise.main(sys.argv[1:]))
            """
        )
    )
    monkeypatch.setattr(cli, "_prepare", counting)
    monkeypatch.setattr(supervise, "start_command", lambda config: [sys.executable, str(start), str(config)])
    config = _run_config(tmp_path, "train")

    cli.main(["train", str(config), "--prepare", f"log_path={tmp_path / 'runs'}", "watchdog.max_restarts=2"])

    assert calls.read_text() == "platoon train\n" and starts.read_text() == "3"
    assert _prepare_lines(capsys.readouterr().err).count("ready: 4 made now, 0 already current") == 1


def test_a_supervised_start_that_is_not_prepared_names_a_prepare_command_that_prepares_it(
    fake_hub: FakeHub,  # noqa: F811
    site,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd,
):
    """A supervised run's starts run in processes of their own, which never see the command line: a start that is
    not prepared names the platoon prepare command for the run's snapshot, which holds its overrides, and that
    command prepares what the run needs."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-not-a-key")
    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", lambda *args: None)
    config = _run_config(tmp_path, "train")
    snapshot = tmp_path / "runs" / "exp" / "trial" / "config.resolved.yaml"

    assert _exit_code(["train", str(config), f"log_path={tmp_path / 'runs'}", "watchdog.max_restarts=1"]) == 78

    err = capfd.readouterr().err
    assert "toolchain agent-1.0, which a run never" in err and f"prepare it with: {prepare_command(snapshot)}" in err
    cli.main(["prepare", str(snapshot)])
    assert capfd.readouterr().out.splitlines()[-1].startswith("ready: ")
