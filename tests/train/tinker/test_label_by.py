"""``train.label_by``: the one column that labels a run's tasks, which group limits and straggler policies key on.

It replaced ``train.rollout_group_key`` (the same key's former name) and ``workflow_config.straggler_policy_key``,
which platoon 0.3.0 removed (``tests/test_removed_forms.py``), and a label is a column's value as text.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from platoon.train.components import ENVIRONMENT_COLUMN, task_ids_to_dataset
from platoon.train.labels import check_labels, dataset_labels, label_text, task_label
from platoon.train.tinker.config_defs import ENVIRONMENT_LABEL, PlatoonTinkerRLTrainerConfig, TrainConfig
from platoon.train.tinker.dispatcher import Dispatcher
from platoon.train.tinker.permits import PermitPool
from platoon.utils.config import ConfigError, resolve_config


def _config_file(tmp_path: Path, train: dict[str, Any], eval_section: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", **train},
                "eval": eval_section or {"every": 0},
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    return path


def _load(tmp_path: Path, train: dict[str, Any], eval_section: dict[str, Any] | None = None, **kwargs: Any):
    return resolve_config(_config_file(tmp_path, train, eval_section), [], PlatoonTinkerRLTrainerConfig, **kwargs)


def test_tasks_are_labelled_by_their_mixture_entry_by_default():
    config = TrainConfig(model_name="m", renderer_name="r")

    assert config.label_by == ENVIRONMENT_LABEL == ENVIRONMENT_COLUMN


def test_label_by_names_the_column(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = _load(tmp_path, {"label_by": "source_dataset", "rollout_group_limits": {"slow": 2}})

    assert resolved.deprecations == ()
    assert (resolved.config.train.label_by, resolved.config.train.rollout_group_limits) == (
        "source_dataset",
        {"slow": 2},
    )


def test_label_by_must_name_a_column():
    with pytest.raises(ValueError, match="train.label_by must name a dataset column"):
        TrainConfig(model_name="m", renderer_name="r", label_by="")


def test_straggler_policies_need_no_column_of_their_own(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = _load(tmp_path, {"workflow_config": {"straggler_policies": {"slow": {"timeout_seconds": 60}}}})

    assert resolved.config.train.workflow_config.straggler_policies["slow"].timeout_seconds == 60


def test_a_label_is_a_columns_value_as_text():
    assert [label_text(value) for value in ("a", "", None, True, False, 3, 2.5, [1], {"a": 1})] == [
        "a",
        None,
        None,
        "true",
        "false",
        "3",
        "2.5",
        None,
        None,
    ]
    assert task_label({"suite": 3}, "suite") == "3"
    assert task_label({"task_id": "t"}, "suite") is None
    rows = [{"task_id": "a", "suite": "x"}, {"task_id": "b"}]
    assert dataset_labels(rows, "suite") == ["x", None]
    assert dataset_labels(task_ids_to_dataset(["a", "b"]), "suite") == [None, None]


def test_rollout_groups_default_to_the_environment_column_mixtures_write():
    config = TrainConfig(model_name="Qwen/Qwen3-4B-Instruct-2507", renderer_name="qwen3_instruct")
    assert config.label_by == ENVIRONMENT_COLUMN == "_platoon_environment"


def test_group_limits_apply_to_a_columns_values_as_labels():
    """The dispatcher reads each row's label as text, so limits keyed on ``3`` and ``true`` apply to a column's int and
    boolean values: one group of each label at a time, the next of a label once its group ends."""

    async def run():
        rows = [{"task_id": str(index), "level": level} for index, level in enumerate([3, 3, True, True])]
        started: list[str] = []
        gates: dict[str, asyncio.Event] = {}

        class Rows:
            def __init__(self) -> None:
                self.permits = PermitPool(100)

            def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
                return None, task_label(data, "level"), 1

        async def run_group(draw, release):
            started.append(draw.task_id)
            await gates.setdefault(draw.task_id, asyncio.Event()).wait()
            release()

        dispatcher = Dispatcher(
            rows=Rows(),
            order=lambda position: rows[position],
            batch_size=4,
            max_staleness=None,
            steps=1,
            limits={"3": 1, "true": 1},
            labels=("3", "true"),
        )
        groups: list[asyncio.Task] = []
        loop = asyncio.ensure_future(
            dispatcher.run(spawn=lambda coroutine: groups.append(asyncio.ensure_future(coroutine)), run_group=run_group)
        )
        for _ in range(20):
            await asyncio.sleep(0)
        assert started == ["0", "2"]  # one of each label at a time
        gates.setdefault("0", asyncio.Event()).set()
        for _ in range(20):
            await asyncio.sleep(0)
        assert started == ["0", "2", "1"]
        for task in [loop, *groups]:
            task.cancel()
        await asyncio.gather(loop, *groups, return_exceptions=True)

    asyncio.run(run())


def _config(label_by: str = ENVIRONMENT_COLUMN, *, limits=None, train_policies=None, eval_policies=None):
    def workflow(policies):
        return SimpleNamespace(straggler_policies=policies or {})

    return SimpleNamespace(
        train=SimpleNamespace(
            label_by=label_by, rollout_group_limits=limits or {}, workflow_config=workflow(train_policies)
        ),
        eval=SimpleNamespace(workflow_config=workflow(eval_policies)),
    )


def test_a_label_column_no_training_task_has_stops_the_run():
    tasks = [{"task_id": "a", "source_dataset": "x"}]

    with pytest.raises(ConfigError, match=r"train.label_by: no training task has a 'sorce_dataset' value") as raised:
        check_labels(_config("sorce_dataset"), tasks)

    assert "their columns are task_id, source_dataset; did you mean source_dataset?" in str(raised.value)
    check_labels(_config(), task_ids_to_dataset(["a", "b"]))  # a single environment: no entry column, no error


def test_unlabelled_tasks_and_settings_for_labels_no_task_has_are_warned_of(caplog):
    tasks = [{"task_id": "a", "suite": "x"}, {"task_id": "b", "suite": "y"}, {"task_id": "c"}]
    evaluated = [{"task_id": "d", "suite": "x"}]
    config = _config(
        "suite",
        limits={"x": 1, "z": 1},
        train_policies={"y": {}, "w": {}},
        eval_policies={"y": {}},
    )

    with caplog.at_level(logging.WARNING, logger="platoon.train.labels"):
        check_labels(config, tasks, evaluated)

    assert caplog.messages == [
        "train.label_by: 1 of the 3 training tasks have no 'suite' value, so no group limit or straggler policy "
        "applies to them",
        "train.rollout_group_limits: no training task has the 'suite' value z, so the setting applies to no task",
        "train.workflow_config.straggler_policies: no training task has the 'suite' value w, so the setting applies "
        "to no task",
        "eval.workflow_config.straggler_policies: no evaluation task has the 'suite' value y, so the setting applies "
        "to no task",
    ]


def test_group_workflows_get_the_label_column():
    from platoon.train.tinker.train import label_kwargs
    from platoon.train.tinker.workflows import GroupRolloutWorkflow

    config = SimpleNamespace(train=SimpleNamespace(label_by="suite"))
    assert label_kwargs(GroupRolloutWorkflow, config) == {"label_by": "suite"}
    assert label_kwargs(object, config) == {}  # a workflow of another kind gets none


def test_metrics_are_named_by_the_column_and_label_and_capped_at_64_labels(caplog):
    from platoon.train.labels import MAX_METRIC_LABELS, metric_labels, metric_prefix

    assert metric_prefix(ENVIRONMENT_COLUMN, "math") == "environment/math"
    assert metric_prefix("source_dataset", "Fine/Smol") == "source_dataset/Fine--Smol"
    assert metric_prefix("a/b", "c") == "a--b/c"
    rows = [{"task_id": "x", "kind": label} for label in ["b", "a", "b", None]]
    assert metric_labels(rows, "kind") == ["b", "a"]  # the most frequent first, ties in label order
    assert metric_labels(None, "kind") == [] and metric_labels(object(), "kind") == []

    many = [{"kind": f"k{index:03d}"} for index in range(100)] + [{"kind": "k099"}]
    with caplog.at_level(logging.WARNING, logger="platoon.train.labels"):
        logged = metric_labels(many, "kind", split="evaluation")
    assert len(logged) == MAX_METRIC_LABELS == 64
    assert logged[:3] == ["k099", "k000", "k001"]
    assert caplog.messages == [
        "train.label_by: the evaluation tasks have 100 'kind' values; their metrics break down by the 64 most "
        "frequent, which label 65 of the 101 labelled tasks (group limits and straggler policies apply to every value)"
    ]
