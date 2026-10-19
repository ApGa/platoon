"""A Harbor dataset's index fields as the run's dataset columns, beside ``task_id``.

A listing gives each task its index fields (a lock row's, or a dataset directory's ``task-index.json`` row) as text,
so that ``train.label_by`` can name one.  A field named like one of platoon's own columns (``task_id``,
``_platoon*``) is left out, and an error when a key that reads columns names it.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.train.auto import AutoDataset
from platoon.train.components import ENVIRONMENT_COLUMN, EnvironmentConfig
from platoon.train.labels import check_labels, dataset_labels
from platoon.utils.config import ConfigError

DOCKERFILE = "FROM python:3.12-slim\nWORKDIR /app\nCOPY instance.json /app/instance.json\n"


def _dataset(root: Path, tasks: dict[str, dict[str, Any]]) -> Path:
    """A dataset directory whose ``task-index.json`` gives each task ``tasks[name]`` as its index fields."""

    for name in tasks:
        for relative, text in {
            "task.toml": "",
            "instruction.md": f"Task {name}.\n",
            "environment/Dockerfile": DOCKERFILE,
            "environment/instance.json": json.dumps({"name": name}),
            "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        }.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    (root / "task-index.json").write_text(
        json.dumps({name: {"task_path": name, **fields} for name, fields in tasks.items()})
    )
    return root


FIELDS = {
    "smol-0": {"source_dataset": "Fine/Smol", "gpus": 0, "verified": True, "tags": ["a"], "note": ""},
    "smol-1": {"source_dataset": "Fine/Smol", "gpus": 0, "verified": False},
    "mco-0": {"source_dataset": "amphora/MCO", "gpus": 1, "weight": 0.5},
}


def _locked(tmp_path: Path, fields: dict[str, dict[str, Any]] = FIELDS) -> dict[str, Any]:
    """A fetched dataset's service options, its tasks' images prepared: a run lists a fetched dataset's tasks only once
    the images they build are (``check_tasks``)."""

    dataset, cache = _dataset(tmp_path / "dataset", fields), tmp_path / "cache"
    _prepare_images(dataset, cache)
    return {"dataset": str(dataset), "cache": str(cache)}


def _prepare_images(dataset: Path, cache: Path, provider: str = "docker") -> None:
    """What platoon prepare does for the images of a fetched dataset, which a run checks when it lists its tasks:
    the dataset's lock fetched, and each image its tasks build recorded in the default image map (as ``local``)."""

    from platoon.envs.harbor.builders import plan_images
    from platoon.envs.harbor.fetch import fetch, fetch_config
    from platoon.envs.harbor.images import ImageEntry, default_image_map, update_image_map

    lock = fetch(fetch_config(str(dataset), cache=cache))
    entries = {image.key: ImageEntry(image="local") for image in plan_images(lock).images}
    update_image_map(default_image_map(cache, provider), entries)


def _config(*entries: EnvironmentConfig, label_by: str = ENVIRONMENT_COLUMN, **train: Any) -> SimpleNamespace:
    return SimpleNamespace(
        environments=list(entries),
        train=SimpleNamespace(
            batch_size=None,
            task_sampler="auto",
            label_by=label_by,
            rollout_group_limits={},
            workflow_config=SimpleNamespace(straggler_policies={}, straggler_policy_key=None),
            **train,
        ),
        eval=SimpleNamespace(workflow_config=SimpleNamespace(straggler_policies={}, straggler_policy_key=None)),
    )


@pytest.mark.asyncio
async def test_a_locked_and_a_directory_service_give_each_tasks_index_fields(tmp_path: Path):
    locked = HarborEnvironmentService(**_locked(tmp_path))
    directory = HarborEnvironmentService(str(tmp_path / "dataset"))

    for service in (locked, directory):
        columns = await service.task_columns(await service.list_tasks("train"))
        assert columns["mco-0"] == FIELDS["mco-0"]  # as the index gives them; the listing makes them text
        assert set(columns) == set(FIELDS)


def test_a_listing_gives_the_index_fields_as_text_columns_beside_the_task_id(tmp_path: Path):
    entry = EnvironmentConfig(service="harbor", service_kwargs=_locked(tmp_path))

    rows = {row["task_id"]: row for row in AutoDataset.from_config(_config(entry), "train").to_list()}

    # tags (not a scalar) and note (empty) give no value, and no other task has them: no column.
    assert rows["smol-0"] == {
        "task_id": "smol-0",
        "source_dataset": "Fine/Smol",
        "gpus": "0",
        "verified": "true",
        "weight": None,
    }
    assert (rows["mco-0"]["gpus"], rows["mco-0"]["weight"], rows["smol-1"]["verified"]) == ("1", "0.5", "false")


def test_a_mixture_keeps_each_entrys_columns_beside_its_name(tmp_path: Path):
    harbor = EnvironmentConfig(name="science", service="harbor", service_kwargs=_locked(tmp_path))
    other = EnvironmentConfig(
        name="other", service="harbor", service_kwargs={"dataset_dir": str(_dataset(tmp_path / "other", {"t": {}}))}
    )

    rows = AutoDataset.from_config(_config(harbor, other), "eval").to_list()

    by_id = {row["task_id"]: row for row in rows}
    assert by_id["science::mco-0"]["source_dataset"] == "amphora/MCO"
    assert by_id["science::mco-0"][ENVIRONMENT_COLUMN] == "science"
    assert by_id["other::t"]["source_dataset"] is None and by_id["other::t"][ENVIRONMENT_COLUMN] == "other"


def test_label_by_names_an_index_field(tmp_path: Path):
    entry = EnvironmentConfig(service="harbor", service_kwargs=_locked(tmp_path))
    config = _config(entry, label_by="source_dataset")

    dataset = AutoDataset.from_config(config, "train")
    check_labels(config, dataset)  # every task has a source_dataset: no error, no warning

    assert sorted(dataset_labels(dataset, "source_dataset")) == ["Fine/Smol", "Fine/Smol", "amphora/MCO"]


def test_an_index_field_named_like_platoons_own_column_is_left_out_and_an_error_when_a_key_names_it(tmp_path: Path):
    fields = {"a": {"task_id": "elsewhere-a", "_platoon_environment": "x", "suite": "s"}, "b": {"suite": "s"}}
    entry = EnvironmentConfig(name="science", service="harbor", service_kwargs=_locked(tmp_path, fields))

    rows = AutoDataset.from_config(_config(entry), "train").to_list()  # no key names them: left out
    assert sorted(rows, key=lambda row: row["task_id"]) == [
        {"task_id": "a", "suite": "s"},
        {"task_id": "b", "suite": "s"},
    ]

    with pytest.raises(ConfigError) as raised:
        AutoDataset.from_config(_config(entry, label_by="task_id"), "train")
    assert raised.value.problems == [
        "environments[science]: train.label_by names 'task_id', a column this environment's tasks have (an index "
        "field) whose name is that of one of platoon's own columns, which the dataset keeps instead, so the key "
        "would not read the task's: name another column"
    ]

    straggler = _config(entry)
    straggler.eval.workflow_config.straggler_policy_key = "task_id"  # deprecated, and a column key too
    with pytest.raises(ConfigError, match=r"eval\.workflow_config\.straggler_policy_key names 'task_id'"):
        AutoDataset.from_config(straggler, "train")
