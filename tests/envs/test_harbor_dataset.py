from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborDataset, load_dataset, load_task
from platoon.envs.harbor.dataset import read_registry, task_dir_of
from platoon.envs.harbor.task import HARBOR_MISC_KEY
from platoon.registry import resolve_component

FIXTURE_TASKS = ["broken-verifier", "failing-check", "hello-file", "invalid-grade"]


def test_indexes_a_dataset_directory_in_name_order(harbor_fixtures: Path):
    dataset = HarborDataset(harbor_fixtures)

    assert dataset.names("train") == FIXTURE_TASKS
    assert dataset.names("anything") == FIXTURE_TASKS
    assert dataset.get("hello-file").name == "platoon/hello-file"
    with pytest.raises(KeyError, match="missing"):
        dataset.get("missing")


def test_splits_come_from_a_manifest_file_or_mapping(harbor_fixtures: Path):
    from_file = HarborDataset(harbor_fixtures, splits="splits.json")
    from_mapping = HarborDataset(harbor_fixtures, splits={"eval": ["hello-file"]})

    assert from_file.names("train") == ["hello-file", "failing-check", "broken-verifier"]
    assert from_file.names("eval") == ["invalid-grade"]
    assert from_mapping.names("eval") == ["hello-file"]
    with pytest.raises(KeyError, match="Unknown split 'test'"):
        from_file.names("test")
    with pytest.raises(ValueError, match="unknown Harbor tasks: \\['nope'\\]"):
        HarborDataset(harbor_fixtures, splits={"train": ["nope"]})
    with pytest.raises(ValueError, match="must list task names"):
        HarborDataset(harbor_fixtures, splits={"train": "hello-file"})


def test_toml_manifests_and_explicit_task_dirs(harbor_fixtures: Path, tmp_path: Path):
    manifest = tmp_path / "splits.toml"
    manifest.write_text('train = ["hello-file"]\n')
    dataset = HarborDataset(
        task_dirs=[harbor_fixtures / "hello-file", harbor_fixtures / "invalid-grade"], splits=manifest
    )

    assert dataset.names("train") == ["hello-file"]
    assert dataset.get("invalid-grade").task_dir == (harbor_fixtures / "invalid-grade").resolve()
    with pytest.raises(ValueError, match="named 'hello-file'"):
        HarborDataset(task_dirs=[harbor_fixtures / "hello-file", harbor_fixtures / "hello-file"])
    with pytest.raises(ValueError, match="exactly one"):
        HarborDataset()


def test_skips_tasks_platoon_cannot_run(make_task, tmp_path: Path, caplog):
    make_task("good")
    make_task("multi-step", task_toml='[[steps]]\nname = "one"\n')
    (tmp_path / "tasks" / "not-a-task").mkdir()

    with caplog.at_level(logging.WARNING):
        dataset = HarborDataset(tmp_path / "tasks")

    assert dataset.names("train") == ["good"]
    assert "multi-step" in caplog.text
    with pytest.raises(ValueError, match="No runnable Harbor tasks"):
        HarborDataset(tmp_path / "tasks" / "not-a-task")


def test_loaders_use_absolute_task_dirs_as_ids(harbor_fixtures: Path):
    task_ids = load_dataset(
        None, "eval", dataset_dir=harbor_fixtures, splits="splits.json", split_name="train", limit=2
    )

    assert task_ids == [str((harbor_fixtures / name).resolve()) for name in ["hello-file", "failing-check"]]
    task = load_task(task_ids[0])
    assert task.id == task_ids[0]
    assert task.misc[HARBOR_MISC_KEY]["task_dir"] == task_ids[0]
    assert resolve_component("dataset_loader", "harbor") is load_dataset
    assert resolve_component("task_loader", "platoon.envs.harbor.loaders:load_task") is load_task


def test_multi_container_tasks_are_refused_by_name_and_in_manifests(make_task, caplog):
    runnable = make_task("single", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    stack = make_task("stack", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    (stack / "environment" / "docker-compose.yaml").write_text("services:\n  judge:\n    image: judge\n")
    dataset_dir = runnable.parent

    with caplog.at_level(logging.WARNING):
        dataset = HarborDataset(dataset_dir)
    assert dataset.names("train") == ["single"]  # never listed, so never run
    assert "Skipping Harbor task" in caplog.text and "multi-container" in caplog.text
    with pytest.raises(ValueError, match="multi-container"):  # by name: why, not "unknown"
        dataset.get("stack")
    with pytest.raises(ValueError, match="names Harbor tasks platoon cannot run: .*multi-container"):
        HarborDataset(dataset_dir, splits={"train": ["single", "stack"]})
    with pytest.raises(ValueError, match="No runnable Harbor tasks in .*multi-container"):
        HarborDataset(task_dirs=[stack])


def test_a_refused_task_shares_its_name_with_no_other(make_task, tmp_path: Path):
    """Two directories named alike are ambiguous even when platoon refuses one: names() would list the runnable one
    while get() and a manifest naming it reported the other's refusal."""

    stack = make_task("stack", task_toml='[environment]\ndocker_image = "ubuntu:24.04"\n')
    (stack / "environment" / "docker-compose.yaml").write_text("services:\n  judge:\n    image: judge\n")
    runnable = shutil.copytree(stack, tmp_path / "elsewhere" / "stack")
    (runnable / "environment" / "docker-compose.yaml").unlink()
    refused_too = shutil.copytree(stack, tmp_path / "again" / "stack")

    for task_dirs in ([stack, runnable], [runnable, stack], [stack, refused_too]):
        with pytest.raises(ValueError, match="Two Harbor tasks are named 'stack'"):
            HarborDataset(task_dirs=task_dirs)
    assert HarborDataset(task_dirs=[runnable]).get("stack").task_dir == runnable.resolve()


# ------------------------------------------------------------------------------------------ nested and registered


def _nest(make_task, relative: str, task_toml: str = "") -> Path:
    """A task at ``tasks/<relative>``."""

    task_dir = make_task(relative.replace("/", "__"), task_toml=task_toml)
    target = task_dir.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    task_dir.rename(target)
    return target


def test_tasks_are_found_at_any_depth_and_named_by_their_relative_path(make_task, tmp_path: Path):
    _nest(make_task, "top")
    _nest(make_task, "source-a/one")
    _nest(make_task, "source-b/deeper/two")
    (tmp_path / "tasks" / "top" / "environment" / "task.toml").write_text("")  # a task's own files are not scanned
    _nest(make_task, ".cache/hidden")
    _nest(make_task, "__pycache__/compiled")

    dataset = HarborDataset(tmp_path / "tasks")

    assert dataset.names("train") == ["source-a/one", "source-b/deeper/two", "top"]
    assert dataset.get("source-b/deeper/two").task_dir == (tmp_path / "tasks" / "source-b" / "deeper" / "two").resolve()


def test_a_registry_names_the_tasks_and_an_index_describes_them(make_task, tmp_path: Path, caplog):
    root = tmp_path / "tasks"
    _nest(make_task, "tasks/smol/smol--1")
    _nest(make_task, "tasks/mco/mco--1")
    _nest(make_task, "tasks/unlisted/x")  # not in the registry: not part of the dataset
    registry = [
        {
            "name": "science",
            "version": "1.0",
            "description": "",
            "tasks": [
                {"name": "smol--1", "path": "tasks/smol/smol--1"},
                {"name": "mco--1", "path": "tasks/mco/mco--1"},
                {"name": "not-downloaded", "path": "tasks/smol/not-downloaded"},
            ],
        }
    ]
    (root / "registry.json").write_text(json.dumps(registry))
    (root / "task-index.json").write_text(json.dumps({"smol--1": {"source_dataset": "FineEnvs/SmolData"}}))

    with caplog.at_level(logging.INFO, logger="platoon.envs.harbor.dataset"):
        dataset = HarborDataset(root)

    assert dataset.names("train") == ["mco--1", "smol--1"]
    assert dataset.get("smol--1").index == {"source_dataset": "FineEnvs/SmolData"}
    assert dataset.get("mco--1").index == {}
    assert "lists 1 Harbor tasks that are not under" in caplog.text
    assert read_registry(root) == {
        "smol--1": "tasks/smol/smol--1",
        "mco--1": "tasks/mco/mco--1",
        "not-downloaded": "tasks/smol/not-downloaded",
    }
    assert task_dir_of(root, "smol--1") == root / "tasks/smol/smol--1"
    assert task_dir_of(root, "unknown") is None


@pytest.mark.parametrize(
    ("registry", "message"),
    [
        ({"tasks": [{"name": "a", "path": "a"}, {"name": "a", "path": "b"}]}, "Two Harbor tasks are named 'a'"),
        ({"tasks": [{"name": "a", "path": "../a"}]}, "outside the dataset"),
        ({"tasks": [{"name": "a", "path": "/abs/a"}]}, "outside the dataset"),
        ({"tasks": [{"name": "a", "path": "a", "git_url": "https://example.com/r.git"}]}, "lives in https://"),
        ({"tasks": [{"name": "a"}]}, "every task needs a name and a path"),
        ([{"name": "d"}], "each with a list of tasks"),
        ("not a registry", "each with a list of tasks"),
    ],
)
def test_malformed_registries_are_refused(tmp_path: Path, registry, message: str):
    (tmp_path / "registry.json").write_text(json.dumps(registry))

    with pytest.raises(ValueError, match=message):
        read_registry(tmp_path)


def test_task_dir_of_finds_relative_paths_without_a_registry(make_task, tmp_path: Path):
    _nest(make_task, "group/name")
    root = tmp_path / "tasks"

    assert task_dir_of(root, "group/name") == root / "group" / "name"
    for task_id in ("../group/name", "/group/name", "group/./name", "group", "missing", ""):
        assert task_dir_of(root, task_id) is None, task_id


def test_the_dataset_passes_agent_user_to_its_tasks(make_task, tmp_path: Path, caplog):
    make_task("runs-as-user", task_toml='[agent]\nuser = "agent"\n')
    make_task("plain")

    with caplog.at_level(logging.WARNING):
        assert HarborDataset(tmp_path / "tasks").names("train") == ["plain"]
    assert "rootless sandboxes run every process as one user" in caplog.text
    ignoring = HarborDataset(tmp_path / "tasks", agent_user="ignore")
    assert ignoring.names("train") == ["plain", "runs-as-user"]
    assert ignoring.get("runs-as-user").ignored_users == {"agent": "agent"}
