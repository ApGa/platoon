"""Fetching Harbor datasets into a content-addressed cache, with a lock: local directories and a fake Hub."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from site_cache import use_site_cache

from platoon.envs.harbor import fetch as fetch_module
from platoon.envs.harbor.dataset import dataset_tasks
from platoon.envs.harbor.fetch import FetchError, Overlay, ensure_lock, fetch, fetch_config, is_pinned
from platoon.envs.harbor.hashing import content_hash
from platoon.envs.harbor.inputs import input_files
from platoon.envs.harbor.lock import LockError, check_entry, read_lock, verify_lock
from platoon.envs.harbor.task import HarborTask

COMMIT = "f216b86ca1a18d44433d6cb45f70bec81f47c66e"
SMOLDATA_DOCKERFILE = """FROM docker.io/savatar101/env-data-agent-train:base
WORKDIR /workdir
COPY input/ /data/input/
RUN pip install --no-cache-dir math-verify==0.9.0
"""
MCO_DOCKERFILE = """FROM python:3.11.16-slim
RUN pip install --no-cache-dir numpy==2.4.6
WORKDIR /workdir
COPY instance.json /workdir/instance.json
"""
OVERLAY = """description = "Install math-verify before copying the inputs"
[match]
source_dataset = "FineEnvs/SmolDataEnvs-harbor-train"
[[edit]]
file = "environment/Dockerfile"
pattern = '^(COPY input/ /data/input/\\n)(RUN pip install --no-cache-dir math-verify==[^\\n]+\\n)'
replace = '\\2\\1'
"""


def _task(root: Path, *, dockerfile: str, toml: str = "", files: dict[str, str] | None = None) -> None:
    contents = {
        "task.toml": toml or '[verifier]\nenvironment_mode = "separate"\n[environment]\nnetwork_mode = "no-network"\n',
        "instruction.md": "Answer the question.\n",
        "environment/Dockerfile": dockerfile,
        "tests/test.sh": "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n",
        "tests/grader.py": "print(1)\n",
        "solution/solve.sh": "echo solve\n",
        "inputs-provenance.json": "{}",  # outside Harbor's package: never fetched
        **(files or {}),
    }
    for relative, text in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def harbor_dataset(root: Path) -> Path:
    """A dataset shaped like a large Harbor one: registry.json, task-index.json and nested tasks/<source>/<id>/."""

    index, registry = {}, []
    sources = {
        "fineenvs--smoldataenvs-harbor-train": ("FineEnvs/SmolDataEnvs-harbor-train", SMOLDATA_DOCKERFILE),
        "amphora--mathconstructoptimize-envs-harbor": ("amphora/MathConstructOptimize-Envs-harbor", MCO_DOCKERFILE),
    }
    for slug, (source, dockerfile) in sources.items():
        for number in range(3):
            name = f"{slug}--task-{number}"
            path = f"tasks/{slug}/{name}"
            files = (
                {"environment/input/shared.csv": "a,b\n1,2\n", f"environment/input/own-{number}.csv": f"{number}\n"}
                if "smol" in slug
                else {"environment/instance.json": json.dumps({"n": number})}
            )
            _task(root / path, dockerfile=dockerfile, files=files)
            registry.append({"name": name, "path": path})
            index[name] = {"task_path": path, "adapter": "native_harbor", "source_dataset": source, "gpus": 0}
    _task(
        root / "tasks" / "gpu-task",
        dockerfile=MCO_DOCKERFILE,
        toml="[environment]\ngpus = 1\n",
        files={"environment/instance.json": "{}"},
    )
    registry.append({"name": "gpu-task", "path": "tasks/gpu-task"})
    index["gpu-task"] = {
        "task_path": "tasks/gpu-task",
        "adapter": "native_harbor",
        "source_dataset": "x/gpu",
        "gpus": 1,
    }
    (root / "registry.json").write_text(
        json.dumps([{"name": "harbor-dataset", "version": "2.0.0", "description": "", "tasks": registry}])
    )
    (root / "task-index.json").write_text(json.dumps(index))
    (root / "data").mkdir()
    (root / "data" / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_id": name, "split_group": f"g{position % 2}", "instruction": "long"}) + "\n"
            for position, name in enumerate(index)
        )
    )
    return root


def _entry_not_found(message: str, url: str) -> Exception:
    """What huggingface_hub raises for a file that a repository lacks, as the installed release raises it: from 1.0
    on, ``RemoteEntryNotFoundError`` with an httpx response; before, ``EntryNotFoundError`` with a requests one (the
    oldest release platoon's requirements admit is one of those)."""

    import huggingface_hub

    error = fetch_module._remote_entry_not_found()
    if int(huggingface_hub.__version__.split(".")[0]) >= 1:
        import httpx

        return error(message, response=httpx.Response(404, request=httpx.Request("GET", url)))
    import requests

    response = requests.Response()
    response.status_code, response.url = 404, url
    return error(message, response=response)


class FakeHub:
    """Stands in for huggingface_hub: serves a local directory as a dataset repository at one commit.

    ``tampered`` maps repository paths to bytes served instead of theirs (a transfer that goes wrong every time);
    ``forced`` counts the downloads asked to bypass the local cache."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import huggingface_hub
        from huggingface_hub.hf_api import RepoFile

        self.root = root
        self.downloads: Counter[str] = Counter()
        self.forced: Counter[str] = Counter()
        self.tampered: dict[str, bytes] = {}
        self.listings: list[str | None] = []
        hub = self

        class Api:
            def repo_info(self, repo_id, *, repo_type, revision):
                assert repo_type == "dataset"
                return type("Info", (), {"sha": COMMIT})()

            def list_repo_tree(self, repo_id, *, path_in_repo=None, recursive, revision, repo_type):
                assert recursive and revision == COMMIT and repo_type == "dataset"
                hub.listings.append(path_in_repo)
                base = hub.root / (path_in_repo or "")
                for path in sorted(base.rglob("*")):
                    if path.is_file():
                        data = path.read_bytes()
                        relative = path.relative_to(hub.root).as_posix()
                        oid = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
                        lfs = None
                        if path.suffix == ".csv":
                            lfs = {"size": len(data), "oid": hashlib.sha256(data).hexdigest(), "pointerSize": 134}
                        yield RepoFile(path=relative, size=len(data), oid=oid, lfs=lfs)

        def download(repo_id, filename, *, repo_type, revision, cache_dir, force_download=False):
            assert revision == COMMIT
            source = hub.root / filename
            if not source.is_file():
                raise _entry_not_found(f"{filename} is not in {repo_id}", f"https://hub.test/{filename}")
            hub.downloads[filename] += 1
            if force_download:
                hub.forced[filename] += 1
            data = source.read_bytes()
            blob = Path(cache_dir) / "blobs" / hashlib.sha256(data).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            if force_download or not blob.exists():  # a new file renamed over the old one, as huggingface_hub does
                temporary = blob.with_name(f"{blob.name}.incomplete")
                temporary.write_bytes(hub.tampered.get(filename, data))
                os.replace(temporary, blob)
            link = Path(cache_dir) / "snapshots" / COMMIT / filename
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                link.symlink_to(blob)
            return str(link)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    monkeypatch.setattr(fetch_module, "_PER_TASK_LISTING", 2)
    return FakeHub(harbor_dataset(tmp_path / "repo"), monkeypatch)


def _overlay(tmp_path: Path) -> Path:
    directory = tmp_path / "overlays" / "smoldata-run-before-copy"
    directory.mkdir(parents=True)
    (directory / "overlay.toml").write_text(OVERLAY)
    return directory


def test_hub_fetch_selects_downloads_each_content_once_and_locks(hub: FakeHub, tmp_path: Path):
    config = fetch_config(
        f"hf://org/harbor-dataset@{COMMIT}",
        cache=tmp_path / "cache",
        lock=tmp_path / "phase0.lock.json",
        select={"source_dataset": ["FineEnvs/SmolDataEnvs-harbor-train", "amphora/MathConstructOptimize-Envs-harbor"]},
        overlays=[_overlay(tmp_path)],
        refuse={"per_task_build": True},
        index=[{"path": "data/tasks.jsonl", "fields": ["split_group"]}],
    )
    lock = fetch(config)
    assert lock == read_lock(tmp_path / "phase0.lock.json")
    assert lock.resolved == COMMIT and lock.hash_rule == "harbor-packager-v1"
    assert [task.name for task in lock.tasks] == sorted(
        [f"amphora--mathconstructoptimize-envs-harbor--task-{n}" for n in range(3)]
        + [f"fineenvs--smoldataenvs-harbor-train--task-{n}" for n in range(3)]
    )
    assert "gpu-task" not in lock  # not selected, never downloaded
    assert not any("gpu-task" in name for name in hub.downloads)
    assert not any(name.endswith("inputs-provenance.json") for name in hub.downloads)
    # One download per distinct content: the shared input, grader and test script once each.
    assert sum(count for name, count in hub.downloads.items() if name.endswith("shared.csv")) == 1
    assert sum(count for name, count in hub.downloads.items() if name.endswith("tests/test.sh")) == 1
    smol = lock.get("fineenvs--smoldataenvs-harbor-train--task-1")
    assert smol.index == {
        "adapter": "native_harbor",
        "source_dataset": "FineEnvs/SmolDataEnvs-harbor-train",
        "gpus": 0,
        "split_group": "g1",
    }
    assert smol.flags["verifier"] == "separate" and smol.flags["network"] == "no-network"
    assert smol.overlays == ("smoldata-run-before-copy",) and smol.upstream_digest not in (None, smol.digest)
    assert smol.image["kind"] == "shared" and smol.image["deferred"] == [["COPY", "input/", "/data/input/"]]
    # The files the image takes from the task, by their text (these end their lines already, so their SHA-256):
    # never its Dockerfile, tests or solution.
    assert smol.input_files == {
        "environment/input/own-1.csv": "sha256:" + hashlib.sha256(b"1\n").hexdigest(),
        "environment/input/shared.csv": "sha256:" + hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
    }
    mco = lock.get("amphora--mathconstructoptimize-envs-harbor--task-0")
    # As text: the SHA-256 of '{"n": 0}' with its line ended.
    assert mco.input_files == {"environment/instance.json": "sha256:" + hashlib.sha256(b'{"n": 0}\n').hexdigest()}
    assert mco.image["kind"] == "shared" and mco.image["key"] != smol.image["key"]
    assert mco.upstream_digest is None and mco.overlays == ()
    assert len({task.image["key"] for task in lock.tasks}) == 2
    assert all(task.refused is None for task in lock.tasks)
    entry = tmp_path / "cache" / smol.path
    assert "sha256:" + content_hash(entry) == smol.digest
    assert os.access(entry / "tests" / "test.sh", os.X_OK) and not os.access(entry / "tests" / "grader.py", os.X_OK)
    assert not (entry / "inputs-provenance.json").exists()
    shared = [tmp_path / "cache" / lock.get(f"fineenvs--smoldataenvs-harbor-train--task-{n}").path for n in range(3)]
    inodes = {(path / "environment" / "input" / "shared.csv").stat().st_ino for path in shared}
    assert len(inodes) == 1  # hard links to one download
    assert (entry / "environment" / "Dockerfile").read_text().index("RUN") < (
        entry / "environment" / "Dockerfile"
    ).read_text().index("COPY")
    assert list(verify_lock(lock, tmp_path / "cache")) == []


def test_a_refetch_reuses_listings_and_downloads(hub: FakeHub, tmp_path: Path):
    config = fetch_config(
        f"hf://org/harbor-dataset@{COMMIT}",
        cache=tmp_path / "cache",
        lock=tmp_path / "l.json",
        select={"source_dataset": "amphora/MathConstructOptimize-Envs-harbor"},
    )
    first = fetch(config)
    listings, downloads = list(hub.listings), sum(hub.downloads.values())
    hub.downloads.clear()
    second = fetch(config)
    assert hub.listings == listings  # cached per commit
    assert [task.digest for task in second.tasks] == [task.digest for task in first.tasks]
    assert downloads > 0


def test_allowlists_list_parents_or_tasks(hub: FakeHub, tmp_path: Path):
    allowlist = tmp_path / "splits.json"
    allowlist.write_text(json.dumps({"train": ["gpu-task"], "val": ["fineenvs--smoldataenvs-harbor-train--task-0"]}))
    lock = fetch(
        fetch_config(
            f"hf://org/harbor-dataset@{COMMIT}", cache=tmp_path / "c", lock=tmp_path / "l.json", tasks=[allowlist]
        )
    )
    assert [task.name for task in lock.tasks] == ["fineenvs--smoldataenvs-harbor-train--task-0", "gpu-task"]
    assert lock.get("gpu-task").refused.startswith("gpus:")
    assert lock.get("fineenvs--smoldataenvs-harbor-train--task-0").image["kind"] == "per_task"
    assert set(hub.listings) == {
        "tasks/gpu-task",
        "tasks/fineenvs--smoldataenvs-harbor-train/fineenvs--smoldataenvs-harbor-train--task-0",
    }
    allowlist.write_text(json.dumps(["nope"]))
    with pytest.raises(FetchError, match="not in"):
        fetch(
            fetch_config(
                f"hf://org/harbor-dataset@{COMMIT}", cache=tmp_path / "c", lock=tmp_path / "l.json", tasks=[allowlist]
            )
        )


def test_moving_references_need_an_update(hub: FakeHub, tmp_path: Path):
    config = fetch_config(
        "hf://org/harbor-dataset@main",
        cache=tmp_path / "c",
        lock=tmp_path / "l.json",
        select={"source_dataset": "x/gpu"},
    )
    assert not is_pinned(config.dataset) and is_pinned(f"hf://a/b@{COMMIT}") and is_pinned(str(tmp_path))
    with pytest.raises(FetchError, match="moving reference"):
        fetch(config)
    with pytest.raises(LockError, match="moving reference"):
        ensure_lock(config, fetch_allowed=True)
    lock = fetch(config, update=True)
    assert lock.resolved == COMMIT and lock.dataset == "hf://org/harbor-dataset@main"
    assert ensure_lock(config, fetch_allowed=False) == lock  # the lock pins it now


def test_ensure_lock_fetches_pinned_references_only_where_allowed(tmp_path: Path):
    dataset = tmp_path / "local"
    _task(dataset / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    config = fetch_config(str(dataset), cache=tmp_path / "cache")
    with pytest.raises(LockError, match="does not exist"):
        ensure_lock(config, fetch_allowed=False)
    lock = ensure_lock(config, fetch_allowed=True)
    assert config.lock_path().parent == tmp_path / "cache" / "locks"
    assert ensure_lock(config, fetch_allowed=False) == lock
    changed = fetch_config(str(dataset), cache=tmp_path / "cache", lock=config.lock_path(), refuse={"gpus": False})
    with pytest.raises(LockError, match="other fetch settings"):
        ensure_lock(changed, fetch_allowed=False)


def test_local_directories_keep_directory_ids_and_refuse_what_cannot_run(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    _task(
        root / "group" / "deep",
        dockerfile=MCO_DOCKERFILE,
        toml='[task]\nname = "org/deep"\n',
        files={"environment/instance.json": "{}"},
    )
    _task(root / "group" / "plain", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    _task(
        root / "stack",
        dockerfile=MCO_DOCKERFILE,
        files={
            "environment/instance.json": "{}",
            "environment/docker-compose.yaml": "services:\n  db: {image: postgres:16}\n",
        },
    )
    _task(
        root / "trivial",
        dockerfile=MCO_DOCKERFILE,
        files={"environment/instance.json": "{}", "environment/docker-compose.yaml": "services:\n  main: {}\n"},
    )
    _task(
        root / "user",
        dockerfile=MCO_DOCKERFILE,
        toml='[agent]\nuser = "learner"\n',
        files={"environment/instance.json": "{}"},
    )
    _task(root / "suffix-user", dockerfile=MCO_DOCKERFILE + "USER learner\n", files={"environment/instance.json": "{}"})
    _task(
        root / "steps",
        dockerfile=MCO_DOCKERFILE,
        toml='[[steps]]\nname = "one"\n',
        files={"environment/instance.json": "{}"},
    )
    _task(root / "broken", dockerfile="FROM x\nRUN true\nCOPY gone.json /w/\n")
    _task(
        root / "prebuilt",
        dockerfile="",
        toml='[environment]\ndocker_image = "python:3.12"\n',
        files={"environment/data.txt": "d"},
    )
    (root / "prebuilt" / "environment" / "Dockerfile").unlink()
    lock = fetch(fetch_config(str(root), cache=tmp_path / "cache", lock=tmp_path / "l.json"))
    names = [task.name for task in lock.tasks]
    assert names == sorted(
        ["alpha", "group/deep", "group/plain", "stack", "trivial", "user", "suffix-user", "steps", "broken", "prebuilt"]
    )
    assert names == [name for name, _ in dataset_tasks(root)]  # the ids dataset_dir gives the same directory
    refused = {task.name: (task.refused or "").split(":")[0] for task in lock.tasks}
    assert refused == {
        "alpha": "",
        "group/deep": "",
        "group/plain": "",
        "stack": "stack",
        "trivial": "",
        "user": "agent_user",
        "suffix-user": "agent_user",
        "steps": "task_config",
        "broken": "image",
        "prebuilt": "",
    }
    assert lock.get("trivial").flags["stack"] == "trivial"
    assert lock.get("prebuilt").image == {"kind": "prebuilt", "ref": "python:3.12", "upload_environment": True}
    assert lock.get("alpha").image["kind"] == "shared"  # alpha, group/deep, group/plain, trivial share it


def test_overlay_mismatches_refuse_the_task(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(
        root / "fixed",
        dockerfile="FROM x\nRUN pip install --no-cache-dir math-verify==0.9.0\nCOPY input/ /i/\n",
        files={"environment/input/a.csv": "1"},
    )
    (root / "task-index.json").write_text(
        json.dumps({"fixed": {"task_path": "fixed", "source_dataset": "FineEnvs/SmolDataEnvs-harbor-train"}})
    )
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", overlays=[_overlay(tmp_path)]))
    assert lock.get("fixed").refused.startswith(
        "overlay: smoldata-run-before-copy: environment/Dockerfile: the pattern matched 0 time(s)"
    )
    assert lock.overlay.startswith("sha256:")


def test_copy_overlays_and_their_digest(tmp_path: Path):
    directory = tmp_path / "overlay"
    (directory / "files").mkdir(parents=True)
    (directory / "files" / "test.sh").write_text("#!/bin/bash\necho patched\n")
    (directory / "overlay.toml").write_text('[[copy]]\nfile = "tests/test.sh"\nsource = "files/test.sh"\n')
    first = Overlay.load(directory)
    (directory / "files" / "test.sh").write_text("#!/bin/bash\necho patched again\n")
    assert Overlay.load(directory).digest != first.digest
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", overlays=[directory]))
    entry = tmp_path / "c" / lock.get("alpha").path
    assert (entry / "tests" / "test.sh").read_text().endswith("patched again\n")
    assert os.access(entry / "tests" / "test.sh", os.X_OK)
    assert (root / "alpha" / "tests" / "test.sh").read_text().startswith("#!/bin/bash\necho 1")  # source untouched
    with pytest.raises(FetchError, match=r"no \[\[edit\]\], \[\[copy\]\], \[\[compose\]\] or compose_host_env"):
        (directory / "overlay.toml").write_text("description = 'empty'\n")
        Overlay.load(directory)


def test_a_refetch_of_copies_of_one_size_and_time_gives_each_task_its_own_digest(tmp_path: Path):
    """A fetch of a local directory copies each task into a staging directory and hashes the copies; a second fetch
    finds every task cached and removes them.  The next task's copies may take the inodes freed, with the same size
    and modification time (a copy keeps its source's): each task must hash to its own files, never to a hash
    remembered for another's."""

    root = tmp_path / "dataset"
    for n in range(40):
        _task(root / f"t{n:02d}", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": f'{{"n": "{n:02d}"}}'})
    stamp = 1_700_000_000_000_000_000
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, ns=(stamp, stamp))
    config = fetch_config(str(root), cache=tmp_path / "cache")

    for lock in (fetch(config), fetch(config)):
        for task in lock.tasks:
            assert task.digest == "sha256:" + content_hash(root / task.name), task.name
            assert task.input_files == input_files(root / task.name), task.name  # its own, never another's
    assert list(verify_lock(read_lock(config.lock_path()), tmp_path / "cache")) == []


def test_a_one_byte_change_fails_verify(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json"))
    assert list(verify_lock(lock, tmp_path / "c")) == []
    instance = tmp_path / "c" / lock.get("alpha").path / "environment" / "instance.json"
    instance.write_text("{ }")
    problems = list(verify_lock(lock, tmp_path / "c"))
    assert len(problems) == 1 and "the lock says" in problems[0]


def test_fetch_settings_resolve_relative_paths_against_the_working_directory(tmp_path: Path, monkeypatch):
    """An entry's paths are absolute once its config loads (relative to the file that names them); the settings a
    caller gives relative paths resolve against the working directory, and the cache is the site's Harbor cache."""

    from platoon.envs.harbor.lock import default_cache

    monkeypatch.chdir(tmp_path)
    config = fetch_config(
        "dataset", lock="locks/x.lock.json", select={"index.source_dataset": "a"}, refuse={"per_task_build": True}
    )
    assert config.dataset == str(tmp_path / "dataset") and config.lock == tmp_path / "locks" / "x.lock.json"
    assert config.cache == default_cache()
    assert config.select == {"source_dataset": ("a",)} and config.refuse.per_task_build
    with pytest.raises(FetchError, match="flags"):
        fetch_config("x", select={"flags.verifier": "separate"})


def test_a_task_toml_inside_a_tasks_files_is_not_another_task(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "outer", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    _task(root / "outer" / "tests" / "fixture", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    _task(root / "steps", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})  # a name, not a dir
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json"))
    assert [task.name for task in lock.tasks] == ["outer", "steps"]


def test_a_subdirectory_of_a_hub_repository(hub: FakeHub, tmp_path: Path):
    lock = fetch(
        fetch_config(
            f"hf://org/harbor-dataset@{COMMIT}#tasks",
            cache=tmp_path / "c",
            lock=tmp_path / "l.json",
            select={"name": ["gpu-task"]},
            refuse={"gpus": False},
        )
    )
    assert [task.name for task in lock.tasks] == ["gpu-task"] and lock.get("gpu-task").refused is None
    assert lock.get("gpu-task").source_path == "gpu-task" and hub.listings[-1] == "tasks/gpu-task"


def test_the_trainer_fetches_cache_entries_a_matching_lock_lacks(tmp_path: Path):
    dataset = tmp_path / "local"
    _task(dataset / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    config = fetch_config(str(dataset), cache=tmp_path / "cache", lock=tmp_path / "l.json")
    lock = ensure_lock(config, fetch_allowed=True)
    entry = tmp_path / "cache" / lock.get("alpha").path
    shutil.rmtree(entry)
    assert ensure_lock(config, fetch_allowed=False) == lock  # workers only read; their lookup reports the entry
    assert ensure_lock(config, fetch_allowed=True).get("alpha").digest == lock.get("alpha").digest
    assert entry.is_dir()


def test_one_rollout_per_node_hashes_an_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import threading
    import time

    from platoon.envs.harbor import lock as lock_module

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "{}"})
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json"))
    entry, digest = tmp_path / "c" / lock.get("alpha").path, lock.get("alpha").digest
    calls = []
    real = lock_module.content_hash

    def slow_hash(directory):
        calls.append(directory)
        time.sleep(0.2)
        return real(directory)

    monkeypatch.setattr(lock_module, "content_hash", slow_hash)
    threads = [threading.Thread(target=lock_module.check_entry, args=(entry, digest)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(calls) == 1
    with pytest.raises(LockError, match="the lock says"):
        lock_module.check_entry(entry, "sha256:" + "0" * 64)


SEPARATE = '[verifier]\nenvironment_mode = "separate"\n'
INSTANCE = {"environment/instance.json": "{}"}


def test_a_fetch_refuses_what_the_task_loader_refuses(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "network", dockerfile=MCO_DOCKERFILE, toml='[environment]\nnetwork_mode = "bridge"\n', files=INSTANCE)
    _task(
        root / "compose-verifier",
        dockerfile=MCO_DOCKERFILE,
        toml=SEPARATE,
        files={**INSTANCE, "tests/docker-compose.yaml": "services:\n  main: {}\n"},
    )
    _task(root / "dotdot", dockerfile=MCO_DOCKERFILE, toml='artifacts = ["../secret"]\n' + SEPARATE, files=INSTANCE)
    _task(
        root / "other-service",
        dockerfile=MCO_DOCKERFILE,
        toml='artifacts = [{source = "/out", service = "db"}]\n' + SEPARATE,
        files=INSTANCE,
    )
    _task(root / "fine", dockerfile=MCO_DOCKERFILE, toml=SEPARATE, files=INSTANCE)
    lock = fetch(fetch_config(str(root), cache=tmp_path / "cache", lock=tmp_path / "l.json"))
    refused = {task.name: task.refused for task in lock.tasks}
    assert refused.pop("fine") is None
    assert all(reason.startswith("task_config: ") for reason in refused.values())
    assert "network_mode must be public, allowlist or no-network, got 'bridge'" in refused["network"]
    assert refused["compose-verifier"].startswith("task_config: compose-verifier: tests/docker-compose.yaml makes")
    assert "'../secret' must not contain '..'" in refused["dotdot"]
    assert "comes from the compose service 'db'" in refused["other-service"]
    assert not any(str(tmp_path) in reason or ".tmp" in reason for reason in refused.values())
    assert [task.name for task in lock.runnable()] == ["fine"]
    assert HarborTask.load(tmp_path / "cache" / lock.get("fine").path).verifier_mode == "separate"


def test_sidecar_stacks_stay_refused_when_the_policy_lets_them_through(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(
        root / "with-db",
        dockerfile=MCO_DOCKERFILE,
        files={**INSTANCE, "environment/docker-compose.yaml": "services:\n  db: {image: postgres:16}\n"},
    )
    by_policy = fetch(fetch_config(str(root), cache=tmp_path / "cache", lock=tmp_path / "a.json"))
    assert by_policy.get("with-db").refused.startswith("stack: ")
    allowed = fetch(
        fetch_config(str(root), cache=tmp_path / "cache", lock=tmp_path / "b.json", refuse={"sidecars": False})
    )
    reason = allowed.get("with-db").refused
    assert reason.startswith("task_config: with-db: environment/docker-compose.yaml makes the environment a multi")
    assert str(tmp_path) not in reason and allowed.runnable() == []


def test_refusals_name_the_task_and_its_files_never_the_directory_it_was_checked_in(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "no-from", dockerfile="# nothing to build\n")
    _task(root / "no-instruction", dockerfile=MCO_DOCKERFILE, files=INSTANCE)
    (root / "no-instruction" / "instruction.md").unlink()
    lock = fetch(fetch_config(str(root), cache=tmp_path / "cache", lock=tmp_path / "l.json"))
    assert (
        lock.get("no-from").refused == "image: environment/Dockerfile: environment/Dockerfile has no FROM instruction"
    )
    assert (
        lock.get("no-from").image["error"] == "environment/Dockerfile: environment/Dockerfile has no FROM instruction"
    )
    assert lock.get("no-instruction").refused == "task_config: no-instruction has no instruction.md"
    assert not any(str(tmp_path) in json.dumps(task.to_json()) for task in lock.tasks)


def test_a_directory_that_is_one_task_is_named_by_its_task_name_else_its_directory(tmp_path: Path):
    _task(tmp_path / "solo", dockerfile=MCO_DOCKERFILE, files=INSTANCE)
    _task(tmp_path / "named", dockerfile=MCO_DOCKERFILE, toml='[task]\nname = "org/named"\n', files=INSTANCE)
    for directory, name in (("solo", "solo"), ("named", "org/named")):
        lock = fetch(fetch_config(str(tmp_path / directory), cache=tmp_path / "c", lock=tmp_path / f"{directory}.json"))
        assert [task.name for task in lock.tasks] == [name] and lock.get(name).source_path == "."


@pytest.mark.parametrize(
    ("index_file", "index"),
    [
        ("registry.json", [{"name": "d", "tasks": [{"name": "escape", "path": "../outside"}]}]),
        ("registry.json", {"name": "d", "tasks": [{"name": "escape", "path": "/srv/outside"}]}),
        ("task-index.json", {"escape": {"task_path": "tasks/../../outside", "source_dataset": "x"}}),
    ],
)
def test_index_paths_that_leave_the_dataset_fail_the_fetch(tmp_path: Path, index_file: str, index):
    _task(tmp_path / "outside", dockerfile=MCO_DOCKERFILE, files=INSTANCE)  # what the paths would reach
    root = tmp_path / "dataset"
    root.mkdir()
    (root / index_file).write_text(json.dumps(index))
    with pytest.raises(FetchError, match=f"{index_file}: task 'escape' has a path outside the dataset"):
        fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json"))


def test_a_refetch_replaces_a_damaged_cache_entry_and_keeps_an_intact_one(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files=INSTANCE)
    _task(root / "beta", dockerfile=MCO_DOCKERFILE, files={"environment/instance.json": "[]"})
    config = fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json")
    lock = fetch(config)
    damaged, intact = (tmp_path / "c" / lock.get(name).path for name in ("alpha", "beta"))
    (damaged / "environment" / "instance.json").write_text("{ }")
    inode = intact.stat().st_ino
    assert len(list(verify_lock(lock, tmp_path / "c"))) == 1
    again = fetch(config)
    assert [task.digest for task in again.tasks] == [task.digest for task in lock.tasks]
    assert list(verify_lock(again, tmp_path / "c")) == []
    assert (damaged / "environment" / "instance.json").read_text() == "{}"
    assert intact.stat().st_ino == inode  # a matching entry is reused as it is
    assert list((tmp_path / "c" / ".tmp").iterdir()) == []  # the damaged copy is deleted


def test_a_refetch_downloads_again_what_no_longer_matches_the_listing(hub: FakeHub, tmp_path: Path):
    config = fetch_config(
        f"hf://org/dataset@{COMMIT}",
        cache=tmp_path / "cache",
        lock=tmp_path / "l.json",
        select={"source_dataset": "FineEnvs/SmolDataEnvs-harbor-train"},
    )
    lock = fetch(config)
    assert not hub.forced
    entry = tmp_path / "cache" / lock.get("fineenvs--smoldataenvs-harbor-train--task-0").path
    for relative in ("environment/input/shared.csv", "tests/grader.py"):  # stored with LFS, and in git
        # Cached files are hard links to the downloads: writing one in place damages the download too.
        (entry / relative).write_text("damaged\n")
    assert len(list(verify_lock(lock, tmp_path / "cache"))) == 3  # every task links the shared files
    again = fetch(config)
    assert sorted(path.rsplit("/", 1)[-1] for path in hub.forced.elements()) == ["grader.py", "shared.csv"]
    assert [task.digest for task in again.tasks] == [task.digest for task in lock.tasks]
    assert list(verify_lock(again, tmp_path / "cache")) == []
    assert (entry / "environment" / "input" / "shared.csv").read_text() == "a,b\n1,2\n"


@pytest.mark.parametrize("relative", ["environment/input/shared.csv", "tests/grader.py"])  # LFS, git
def test_a_download_that_never_matches_its_listing_fails_the_fetch(hub: FakeHub, tmp_path: Path, relative: str):
    task = "fineenvs--smoldataenvs-harbor-train--task-0"
    path = f"tasks/fineenvs--smoldataenvs-harbor-train/{task}/{relative}"
    hub.tampered[path] = b"not what the listing says\n"
    config = fetch_config(
        f"hf://org/dataset@{COMMIT}", cache=tmp_path / "c", lock=tmp_path / "l.json", select={"name": [task]}
    )
    with pytest.raises(FetchError, match=f"{re.escape(path)} does not match the repository's listing"):
        fetch(config)
    assert hub.forced == Counter({path: 1})  # downloaded once more, bypassing the local cache
    assert not (tmp_path / "l.json").exists()


def test_hub_fetches_leave_out_what_a_gitignore_leaves_out_of_the_hash(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    _task(
        repo / "tasks" / "ignoring",
        dockerfile=MCO_DOCKERFILE,
        files={**INSTANCE, ".gitignore": "environment/data/\n", "environment/data/big.bin": "x" * 4096},
    )
    _task(  # no .gitignore: Harbor's default ignores apply
        repo / "tasks" / "defaults",
        dockerfile=MCO_DOCKERFILE,
        files={"environment/instance.json": "[]", "tests/__pycache__/grader.cpython-312.pyc": "compiled"},
    )
    hub = FakeHub(repo, monkeypatch)
    lock = fetch(fetch_config(f"hf://org/dataset@{COMMIT}#tasks", cache=tmp_path / "cache", lock=tmp_path / "h.json"))
    assert not any("environment/data" in path or "__pycache__" in path for path in hub.downloads)
    assert hub.downloads["tasks/ignoring/.gitignore"] == 1
    ignoring = tmp_path / "cache" / lock.get("ignoring").path
    assert (ignoring / ".gitignore").is_file() and not (ignoring / "environment" / "data").exists()
    assert not (tmp_path / "cache" / lock.get("defaults").path / "tests" / "__pycache__").exists()
    assert lock.get("ignoring").digest == "sha256:" + content_hash(repo / "tasks" / "ignoring")
    local = fetch(fetch_config(str(repo / "tasks"), cache=tmp_path / "local", lock=tmp_path / "local.json"))
    assert [(task.name, task.digest) for task in lock.tasks] == [(task.name, task.digest) for task in local.tasks]

    def files(entry: Path) -> list[str]:
        return sorted(path.relative_to(entry).as_posix() for path in entry.rglob("*") if path.is_file())

    for task in lock.tasks:  # the same files as a fetch of the directory
        assert files(tmp_path / "cache" / task.path) == files(tmp_path / "local" / task.path)


def test_a_verification_marker_counts_only_for_the_cache_directory_it_was_written_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    root = tmp_path / "dataset"
    _task(root / "alpha", dockerfile=MCO_DOCKERFILE, files=INSTANCE)
    first = fetch(fetch_config(str(root), cache=tmp_path / "a", lock=tmp_path / "a.json")).get("alpha")
    second = fetch(fetch_config(str(root), cache=tmp_path / "b", lock=tmp_path / "b.json")).get("alpha")
    assert first.digest == second.digest
    check_entry(tmp_path / "a" / first.path, first.digest)  # marks the digest verified for cache a on this node
    damaged = tmp_path / "b" / second.path
    (damaged / "environment" / "instance.json").write_text("{ }")
    with pytest.raises(LockError, match="the lock says"):
        check_entry(damaged, second.digest)


def test_a_hub_fetch_reads_each_distinct_file_once_and_so_does_a_refetch(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reads: Counter[int] = Counter()  # by inode: hard links are one file

    def counting(hash_file):
        def counted(path: Path):
            reads[path.stat().st_ino] += 1
            return hash_file(path)

        return counted

    monkeypatch.setattr(fetch_module, "file_sha256", counting(fetch_module.file_sha256))
    monkeypatch.setattr(fetch_module, "_sha256_and_blob_id", counting(fetch_module._sha256_and_blob_id))
    config = fetch_config(f"hf://org/dataset@{COMMIT}", cache=tmp_path / "c", lock=tmp_path / "l.json")
    fetch(config)  # checking each download hashes it; caching the tasks reuses those hashes
    assert reads and set(reads.values()) == {1}
    reads.clear()
    fetch(config)  # the cache entries are hashed again, but they are hard links to the downloads
    assert reads and set(reads.values()) == {1}


# --- index fields from each task's task.toml [metadata] --------------------------------------------------------


def _metadata_dataset(root: Path, metadata: dict[str, str], index: dict[str, dict] | None = None) -> Path:
    """Tasks whose task.toml carries the given [metadata] lines, and a task-index.json with ``index`` rows."""

    for name, lines in metadata.items():
        _task(root / name, dockerfile=MCO_DOCKERFILE, toml=f"[metadata]\n{lines}\n" if lines else "", files=INSTANCE)
    rows = {name: {"task_path": name, "source_dataset": "data", **(index or {}).get(name, {})} for name in metadata}
    (root / "task-index.json").write_text(json.dumps(rows))
    return root


def test_metadata_fields_are_recorded_as_index_fields(tmp_path: Path):
    root = _metadata_dataset(
        tmp_path / "dataset",
        {
            "alpha": 'kaggle_dataset_name = "sales"\ndifficulty = 3\nnotes = "long"',
            "beta": 'kaggle_dataset_name = "weather"\nreviewed = true',
            "gamma": "",  # no [metadata] table: nothing to record
        },
    )
    lock = fetch(
        fetch_config(
            str(root),
            cache=tmp_path / "c",
            lock=tmp_path / "l.json",
            index=[{"metadata": ["kaggle_dataset_name", "difficulty", "reviewed"]}],
        )
    )
    assert lock.get("alpha").index == {"source_dataset": "data", "kaggle_dataset_name": "sales", "difficulty": 3}
    assert lock.get("beta").index == {"source_dataset": "data", "kaggle_dataset_name": "weather", "reviewed": True}
    assert lock.get("gamma").index == {"source_dataset": "data"}  # a field a task lacks is not recorded for it
    assert "notes" not in lock.get("alpha").index  # only the named fields
    assert read_lock(tmp_path / "l.json").get("alpha").index["kaggle_dataset_name"] == "sales"


def test_a_metadata_field_that_is_not_a_scalar_fails_the_fetch(tmp_path: Path):
    root = _metadata_dataset(tmp_path / "dataset", {"alpha": 'tags = ["a", "b"]', "beta": 'tags = "b"'})
    with pytest.raises(FetchError, match=r"'alpha': task.toml \[metadata\] tags is \['a', 'b'\]; an index field"):
        fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", index=[{"metadata": "tags"}]))
    assert not (tmp_path / "l.json").exists()


def test_a_metadata_field_the_index_gives_another_value_fails_the_fetch_naming_both(tmp_path: Path):
    root = _metadata_dataset(
        tmp_path / "dataset",
        {"alpha": 'split_group = "g1"', "beta": 'split_group = "g2"'},
        index={"alpha": {"split_group": "g1"}, "beta": {"split_group": "g9"}},
    )
    config = fetch_config(
        str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", index=[{"metadata": ["split_group"]}]
    )
    with pytest.raises(
        FetchError, match=r"'beta': the index field 'split_group' is 'g9' in the dataset's index and 'g2'"
    ):
        fetch(config)
    rows = json.loads((root / "task-index.json").read_text())
    rows["beta"]["split_group"] = "g2"  # the same value twice is no conflict
    (root / "task-index.json").write_text(json.dumps(rows))
    assert fetch(config).get("beta").index["split_group"] == "g2"


def test_select_cannot_name_a_metadata_field(tmp_path: Path):
    root = _metadata_dataset(tmp_path / "dataset", {"alpha": 'kaggle_dataset_name = "sales"'})
    config = fetch_config(
        str(root),
        cache=tmp_path / "c",
        lock=tmp_path / "l.json",
        select={"kaggle_dataset_name": "sales"},
        index=[{"metadata": ["kaggle_dataset_name"]}],
    )
    with pytest.raises(FetchError, match="has the index field 'kaggle_dataset_name'; fields from task.toml"):
        fetch(config)


def test_metadata_entries_are_inputs_of_the_lock(tmp_path: Path):
    root = _metadata_dataset(tmp_path / "dataset", {"alpha": 'kaggle_dataset_name = "sales"'})
    plain = fetch_config(str(root), cache=tmp_path / "c")
    with_metadata = fetch_config(str(root), cache=tmp_path / "c", index=[{"metadata": ["kaggle_dataset_name"]}])
    assert plain.inputs_digest() != with_metadata.inputs_digest()
    assert plain.lock_path() != with_metadata.lock_path()
    first = ensure_lock(with_metadata, fetch_allowed=True)
    assert ensure_lock(with_metadata, fetch_allowed=False) == first  # its own lock answers it
    with pytest.raises(LockError, match="does not exist"):
        ensure_lock(plain, fetch_allowed=False)  # and not the config without the entry


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"metadata": []}, "metadata must list field names"),
        ({"metadata": [""]}, "metadata must list field names"),
        ({"metadata": ["a"], "path": "x.json"}, "an entry with metadata takes no other key"),
    ],
)
def test_metadata_entries_are_checked(tmp_path: Path, entry, message):
    with pytest.raises(FetchError, match=message):
        fetch_config(str(tmp_path), cache=tmp_path / "c", index=[entry])


# --- refusing tasks whose sessions would upload too much -----------------------------------------------------

COPY_DOCKERFILE = (
    "FROM python:3.12-slim\nRUN pip install numpy\nWORKDIR /app\nCOPY input/ /data/input/\nCOPY small.txt /app/\n"
)
KIB = 1024


def _sized_task(root: Path, name: str, *, input_kib: int, toml: str = "", files: dict[str, str] | None = None) -> None:
    _task(
        root / name,
        dockerfile=COPY_DOCKERFILE,
        toml=toml or "[environment]\n",
        files={
            "environment/input/data.bin": "x" * (input_kib * KIB),
            "environment/small.txt": "s" * 10,
            **(files or {}),
        },
    )


def test_a_task_whose_deferred_copies_exceed_the_limit_is_refused_with_its_size(tmp_path: Path):
    root = tmp_path / "dataset"
    _sized_task(root, "small", input_kib=100)
    _sized_task(root, "large", input_kib=3 * 1024)
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", refuse={"uploads_over_mb": 1}))
    assert lock.get("small").refused is None
    assert lock.get("small").flags["upload_bytes"] == 100 * KIB + 10
    assert lock.get("large").flags["upload_bytes"] == 3 * 1024 * KIB + 10
    assert lock.get("large").refused == (
        "uploads_over_mb: a session uploads 3.0 MiB into one of the task's sandboxes, over the limit of 1 MiB"
    )
    assert [task.name for task in lock.runnable()] == ["small"]


def test_a_prebuilt_task_counts_the_environment_it_uploads(tmp_path: Path):
    root = tmp_path / "dataset"
    _task(
        root / "prebuilt",
        dockerfile="",
        toml='[environment]\ndocker_image = "python:3.12"\n',
        files={"environment/data.bin": "x" * (2 * 1024 * KIB)},
    )
    (root / "prebuilt" / "environment" / "Dockerfile").unlink()
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", refuse={"uploads_over_mb": 1}))
    row = lock.get("prebuilt")
    assert row.image["upload_environment"] and row.flags["upload_bytes"] == 2 * 1024 * KIB
    assert row.refused.startswith("uploads_over_mb: a session uploads 2.0 MiB")


def test_a_separate_verifier_counts_once_per_sandbox(tmp_path: Path):
    """Each sandbox gets its own copy: the agents' and a separate verifier's.  The larger one counts, never their
    sum: a verifier that runs the task's image uploads the same inputs again, into its own sandbox."""

    root = tmp_path / "dataset"
    separate = '[verifier]\nenvironment_mode = "separate"\n'
    _sized_task(root, "same-image", input_kib=700, toml=separate)
    _sized_task(
        root,
        "own-image",
        input_kib=200,
        toml=separate,
        files={
            "tests/Dockerfile": "FROM python:3.12-slim\nRUN pip install pytest\nCOPY expected.bin /grader/\n",
            "tests/expected.bin": "e" * (900 * KIB),
        },
    )
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json", refuse={"uploads_over_mb": 1}))
    assert lock.get("same-image").flags["upload_bytes"] == 700 * KIB + 10  # not twice
    assert lock.get("own-image").flags["upload_bytes"] == 900 * KIB  # the verifier's copy is the larger
    assert lock.get("same-image").refused is None and lock.get("own-image").refused is None


def test_without_a_limit_nothing_is_refused_and_the_size_is_still_recorded(tmp_path: Path):
    root = tmp_path / "dataset"
    _sized_task(root, "large", input_kib=3 * 1024)
    lock = fetch(fetch_config(str(root), cache=tmp_path / "c", lock=tmp_path / "l.json"))
    assert lock.get("large").refused is None and lock.get("large").flags["upload_bytes"] == 3 * 1024 * KIB + 10


@pytest.mark.parametrize(
    ("refuse", "message"),
    [
        ({"gpus": "false"}, "refuse.gpus must be true or false, got 'false'"),
        ({"per_task_build": 1}, "refuse.per_task_build must be true or false, got 1"),
        ({"uploads_over_mb": 0}, "refuse.uploads_over_mb must be a positive number of MiB, got 0"),
        ({"uploads_over_mb": -5}, "must be a positive number of MiB"),
        ({"uploads_over_mb": True}, "must be a positive number of MiB"),
        ({"uploads_over_mb": "100"}, "must be a positive number of MiB"),
        ({"uploads_over_mb": float("inf")}, "must be a positive number of MiB"),
        ({"upload_mb": 100}, "refuse: unknown keys \\['upload_mb'\\]"),
    ],
)
def test_refuse_takes_booleans_and_a_positive_size(tmp_path: Path, refuse, message):
    with pytest.raises(FetchError, match=message):
        fetch_config(str(tmp_path), cache=tmp_path / "c", refuse=refuse)


def test_the_size_limit_is_an_input_of_the_lock_only_when_set(tmp_path: Path):
    unset = fetch_config("hf://org/data@" + COMMIT, cache=tmp_path / "c")
    limited = fetch_config("hf://org/data@" + COMMIT, cache=tmp_path / "c", refuse={"uploads_over_mb": 100})
    assert unset.refuse.inputs() == {"sidecars": True, "gpus": True, "agent_user": True, "per_task_build": False}
    assert limited.refuse.inputs()["uploads_over_mb"] == 100.0
    assert unset.inputs_digest() != limited.inputs_digest()
    assert (
        limited.inputs_digest()
        == fetch_config(
            "hf://org/data@" + COMMIT, cache=tmp_path / "c", refuse={"uploads_over_mb": 100.0}
        ).inputs_digest()
    )  # 100 and 100.0 are one limit


# The inputs digests that platoon computed for these configs before the size limit and the metadata entries existed
# (tests/envs fixture values only, so the digests do not depend on the checkout's path): a lock written then still
# answers the same config.
GOLDEN_FULL = "cdd5eb84b2e7f78ec1413f98d8cd8be0d18fa290e574a9dd41ee2f4f52c7cc0f"
GOLDEN_PLAIN = "22cc32a399cec05db28dbe3fab09d5a417356971c3702245343ebf58e27463e0"


def test_a_lock_written_before_the_new_keys_still_answers_its_config(tmp_path: Path):
    allowlist = tmp_path / "allow.json"
    allowlist.write_text(json.dumps(["a", "b"]))
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "overlay.toml").write_text(OVERLAY)
    full = fetch_config(
        f"hf://org/harbor-dataset@{COMMIT}",
        cache=tmp_path / "c",
        tasks=[allowlist],
        select={"source_dataset": ["FineEnvs/SmolDataEnvs-harbor-train", "amphora/MathConstructOptimize-Envs-harbor"]},
        overlays=[overlay],
        refuse={"sidecars": True, "gpus": True, "agent_user": True, "per_task_build": True},
        index=[
            "data/extra.json",
            {"path": "data/tasks.jsonl", "key": "task_id", "fields": ["split_group", "source_id"]},
        ],
    )
    plain = fetch_config(f"hf://org/harbor-dataset@{COMMIT}", cache=tmp_path / "c")

    assert (full.inputs_digest(), plain.inputs_digest()) == (GOLDEN_FULL, GOLDEN_PLAIN)
    assert full.lock_path() == tmp_path / "c" / "locks" / f"{GOLDEN_FULL[:16]}.lock.json"  # the default lock is found


# --- platoon harbor fetch CONFIG: a training or evaluation config's entries ------------------------------------------


def test_fetching_a_pinned_dataset_again_changes_nothing_until_asked_to_repair(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """A dataset pinned to a commit cannot change: once its lock answers the entry and the cache holds its tasks,
    `platoon harbor fetch CONFIG` downloads and hashes nothing, and prints the same lock and hold-out."""

    import yaml

    from platoon import cli

    use_site_cache(monkeypatch, tmp_path / "cache")
    config = tmp_path / "train.yaml"
    entry = {
        "name": "science",
        "service": "harbor",
        "service_kwargs": {
            "dataset": f"hf://org/harbor-dataset@{COMMIT}",
            "select": {
                "source_dataset": ["FineEnvs/SmolDataEnvs-harbor-train", "amphora/MathConstructOptimize-Envs-harbor"]
            },
            "index": [{"path": "data/tasks.jsonl", "fields": ["split_group"]}],
            "holdout": {"fraction": 0.5, "group_by": "split_group", "stratify_by": "source_dataset"},
        },
        "harness": "platoon-bash",
    }
    config.write_text(yaml.safe_dump({"environments": [entry], "train": {"model_name": "m", "renderer_name": "r"}}))
    cli.main(["harbor", "fetch", str(config)])
    first = json.loads(capsys.readouterr().out)
    downloads = sum(hub.downloads.values())

    def never(*args, **kwargs):
        raise AssertionError("a current lock was fetched or hashed again")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("fetch", "file_sha256", "_sha256_and_blob_id"):
            patch.setattr(fetch_module, name, never)
        cli.main(["harbor", "fetch", str(config)])
    second, err = capsys.readouterr()

    assert json.loads(second) == {**first, "fetched": False}  # the same lock, tasks and hold-out
    assert "nothing to fetch (--repair hashes them again)" in err and sum(hub.downloads.values()) == downloads
    assert first["holdout"]["eval"] + first["holdout"]["train"] == first["runnable"] == 6

    cli.main(["harbor", "fetch", str(config), "--repair"])
    assert json.loads(capsys.readouterr().out)["fetched"] is True


def test_a_pinned_lock_written_before_input_files_is_fetched_again_to_record_them(
    hub: FakeHub, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`platoon harbor fetch` leaves a current pinned lock alone, but not one written before platoon recorded the
    tasks' input files: it says so and fetches again, and the lock it writes holds them; the hold-out is the same."""

    import yaml

    from platoon import cli
    from platoon.envs.harbor.lock import write_lock

    use_site_cache(monkeypatch, tmp_path / "cache")
    config = tmp_path / "train.yaml"
    entry = {
        "name": "science",
        "service": "harbor",
        "service_kwargs": {
            "dataset": f"hf://org/harbor-dataset@{COMMIT}",
            "select": {"source_dataset": "amphora/MathConstructOptimize-Envs-harbor"},
            "index": [{"path": "data/tasks.jsonl", "fields": ["split_group"]}],
            "holdout": {"fraction": 0.5, "group_by": "split_group"},
        },
        "harness": "platoon-bash",
    }
    config.write_text(yaml.safe_dump({"environments": [entry], "train": {"model_name": "m", "renderer_name": "r"}}))
    cli.main(["harbor", "fetch", str(config)])
    first = json.loads(capsys.readouterr().out)
    lock_path = Path(first["lock"])
    lock = read_lock(lock_path)
    write_lock(replace(lock, tasks=tuple(replace(task, input_files=None) for task in lock.tasks)), lock_path)

    cli.main(["harbor", "fetch", str(config)])
    second, err = capsys.readouterr()

    assert "records no input files, as locks written before platoon recorded them do" in err
    assert json.loads(second) == {**first, "fetched": True}  # the same lock path, tasks and hold-out
    assert read_lock(lock_path).records_input_files


def test_fetches_of_one_pinned_lock_at_once_fetch_it_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two fetches of one lock at once, as two prepares on two nodes run them, take turns on the lock's file lock:
    the one that waited finds the lock the other wrote current and fetches nothing."""

    import threading
    import time

    from platoon.envs.harbor.lock import DatasetLock, write_lock

    config = fetch_config(f"hf://org/harbor-dataset@{COMMIT}", cache=tmp_path / "cache")
    calls: list[str] = []

    def slow_fetch(settings, *, update: bool = False, log=None) -> DatasetLock:
        calls.append(settings.dataset)
        time.sleep(0.2)
        lock = DatasetLock(
            dataset=settings.dataset, resolved=COMMIT, fetched_with="", tasks=(), inputs=settings.inputs_digest()
        )
        write_lock(lock, settings.lock_path())
        return lock

    monkeypatch.setattr(fetch_module, "fetch", slow_fetch)
    results: list[tuple] = []
    threads = [threading.Thread(target=lambda: results.append(fetch_module.fetch_if_needed(config))) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == [config.dataset] and sorted(fetched for _, fetched in results) == [False, True]
    assert fetch_module.fetch_if_needed(config)[1] is False  # current: nothing to fetch
    assert fetch_module.fetch_if_needed(config, repair=True)[1] is True  # unless asked to repair
    local = fetch_config(str(tmp_path / "anywhere"), cache=tmp_path / "cache")
    assert fetch_module.fetch_if_needed(local)[1] is True  # a local directory is fetched every time
