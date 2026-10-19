"""``environments[].exclude_tasks``: tasks left out of an entry's task lists by id or by name, for every kind of task
list (ORS environments, task tables, Harbor datasets, dataset loaders) and on an entry that selects a named
environment; checked when the tasks are listed, and recorded in the run's ``lock.json``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ors_toys
import pytest
import yaml
from extension_site import sandbox

from platoon import cli
from platoon.envs.harbor import HarborEnvironmentService
from platoon.envs.ors import OrsEnvironmentService
from platoon.eval.run import PlatoonEvalConfig
from platoon.task import Task
from platoon.train.auto import AutoDataset
from platoon.train.components import ENVIRONMENT_COLUMN, EnvironmentConfig, normalize_environment_configs
from platoon.train.exclusion import TaskExclusion, record_exclusions
from platoon.train.tinker import train as train_module
from platoon.utils.config import ConfigError, resolve_config, update_run_lock

# Splits train and test, each with the tasks <split>:0..2, named <split>-0..2.
COUNTER = "python://ors_toys:CounterEnv"
HERE = __name__


def _config(*environments: EnvironmentConfig) -> SimpleNamespace:
    return SimpleNamespace(environments=list(environments), train=SimpleNamespace(batch_size=None, task_order="auto"))


def _ids(config: SimpleNamespace, split: str = "train") -> list[str]:
    return list(AutoDataset.from_config(config, split)["task_id"])


def _problem(entry: EnvironmentConfig, split: str = "train") -> str:
    with pytest.raises(ConfigError) as error:
        AutoDataset.from_config(_config(entry), split)
    (problem,) = error.value.problems
    return problem


# --- the config surface ------------------------------------------------------------------------------------------


def test_exclude_tasks_holds_task_ids_or_names_each_once_in_order():
    assert EnvironmentConfig().exclude_tasks == []
    entry = EnvironmentConfig(service=COUNTER, exclude_tasks=["train:2", "train-0", "train:2"])
    assert entry.exclude_tasks == ["train:2", "train-0"]
    for bad in ("train:2", {"train:2": True}, ["train:2", ""], [" train:2"], [3]):
        with pytest.raises(ValueError, match="exclude_tasks"):
            EnvironmentConfig(service=COUNTER, exclude_tasks=bad)


def test_config_files_and_overrides_set_exclude_tasks_and_loading_checks_the_items(tmp_path):
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {"inference": {"model_name": "m"}, "environments": [{"service": COUNTER, "exclude_tasks": ["train-0"]}]}
        )
    )

    loaded = resolve_config(path, ["environments.0.exclude_tasks+=train:2"], PlatoonEvalConfig)
    assert loaded.config.environments[0].exclude_tasks == ["train-0", "train:2"]
    with pytest.raises(ConfigError, match=r"environments\.0\.exclude_tasks\.0: 8 is not a str"):
        resolve_config(path, ["environments.0.exclude_tasks=[8]"], PlatoonEvalConfig)
    with pytest.raises(ConfigError, match="without surrounding spaces"):
        resolve_config(path, ["environments.0.exclude_tasks=['train-0 ']"], PlatoonEvalConfig)


# --- ORS environments --------------------------------------------------------------------------------------------


def test_an_ors_entry_leaves_tasks_out_of_every_list_by_id_and_by_name():
    entry = EnvironmentConfig(
        service=COUNTER, exclude_tasks=["train-1", "test:0"], eval_dataset_kwargs={"split_name": "test"}
    )

    assert _ids(_config(entry), "train") == ["train:0", "train:2"]
    assert _ids(_config(entry), "eval") == ["test:1", "test:2"]


def test_tasks_leave_after_the_seeded_shuffle_and_before_the_limit():
    def listed(exclude_tasks: list[str], **dataset_kwargs) -> list[str]:
        both = {"split_name": ["train", "test"], **dataset_kwargs}
        return _ids(_config(EnvironmentConfig(service=COUNTER, exclude_tasks=exclude_tasks, dataset_kwargs=both)))

    shuffled = listed([], shuffle_seed=3)
    remaining = [task_id for task_id in shuffled if task_id not in ("train:1", "test:2")]
    assert listed(["train:1", "test-2"], shuffle_seed=3) == remaining  # the seeded order of the rest is kept
    assert listed(["train:1", "test-2"], shuffle_seed=3, limit=4) == remaining[:4]  # the limit counts the rest
    assert listed(["train:0"], limit=2) == ["train:1", "train:2"]


def test_items_that_name_no_task_are_a_config_error_naming_each_one():
    entry = EnvironmentConfig(service=COUNTER, exclude_tasks=["train-1", "train-9", "train:7", "default::train:0"])

    problem = _problem(entry)
    assert problem.startswith(
        "environments[default].exclude_tasks: ['train-9', 'train:7', 'default::train:0'] match no task id or name "
        "of the environment (its tasks have ids and names like 'train:0' and 'train-0')"
    )
    assert "without the mixture's '<name>::' prefix" in problem


def test_an_item_may_name_a_task_that_the_list_does_not_hold_but_the_environment_has():
    # train-2 is a task the service's own selection leaves out; test:1 and test-2 belong to the other split.
    entry = EnvironmentConfig(
        service=COUNTER, service_kwargs={"task_indices": [0, 1]}, exclude_tasks=["train-2", "test:1", "test-2"]
    )

    assert _ids(_config(entry)) == ["train:0", "train:1"]


def test_leaving_no_task_is_a_config_error():
    entry = EnvironmentConfig(service=COUNTER, exclude_tasks=["train:0", "train-1", "train:2"])

    assert _problem(entry) == "environments[default].exclude_tasks leaves none of the 3 tasks of split 'train'"


@pytest.mark.asyncio
async def test_an_ors_service_names_every_task_of_a_split_whatever_its_selection():
    service = OrsEnvironmentService(ors_toys.CounterEnv, task_indices=[2])

    assert await service.list_task_names("test") == {"test:0": "test-0", "test:1": "test-1", "test:2": "test-2"}


# --- task tables, Harbor datasets and dataset loaders --------------------------------------------------------------


def test_a_task_table_leaves_rows_out_by_id():
    rows = [{"id": "a", "goal": "A."}, {"id": "b", "goal": "B."}, {"id": "c", "goal": "C.", "split": "test"}]
    table = {"service": "table", "service_kwargs": {"rows": rows}}

    assert _ids(_config(EnvironmentConfig(**table, exclude_tasks=["b", "c"]))) == ["a"]
    assert _problem(EnvironmentConfig(**table, exclude_tasks=["d"])) == (
        "environments[default].exclude_tasks: ['d'] match no task id or name of the environment (its tasks have "
        "ids and names like 'a')"
    )


@pytest.mark.asyncio
async def test_a_harbor_service_lists_its_manifest_splits(harbor_fixtures: Path):
    assert await HarborEnvironmentService(harbor_fixtures, splits="splits.json").list_splits() == ["train", "eval"]
    assert await HarborEnvironmentService(harbor_fixtures).list_splits() == []


def test_a_harbor_dataset_leaves_task_directories_out_by_name(harbor_fixtures: Path):
    def entry(exclude_tasks: list[str], **service_kwargs) -> EnvironmentConfig:
        kwargs = {"dataset_dir": str(harbor_fixtures), **service_kwargs}
        return EnvironmentConfig(service="harbor", service_kwargs=kwargs, exclude_tasks=exclude_tasks)

    # invalid-grade is in the manifest's eval split only.
    split = entry(["failing-check", "invalid-grade"], splits="splits.json")
    assert _ids(_config(split)) == ["hello-file", "broken-verifier"]
    assert _ids(_config(entry(["failing-check"]))) == ["broken-verifier", "hello-file", "invalid-grade"]
    assert "['nope'] match no task id or name" in _problem(entry(["hello-file", "nope"], splits="splits.json"))
    assert "['nope'] match no task id or name" in _problem(entry(["nope"]))


def load_ids(config, split, **kwargs) -> list[str]:
    return ["x0", "x1", "x2"]


def load_rows(config, split) -> list[dict]:
    return [{"task_id": "r0", "level": 1}, {"task_id": "r1", "level": 2}]


def load_hf_dataset(config, split):
    from datasets import Dataset

    return Dataset.from_list([{"task_id": "d0"}, {"task_id": "d1"}])


def load_opaque(config, split):
    return ("dataset", split)


def test_a_dataset_loaders_tasks_leave_by_id_and_items_it_does_not_return_are_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="platoon.train.exclusion"):
        assert _ids(_config(EnvironmentConfig(dataset_loader=f"{HERE}:load_ids", exclude_tasks=["x1", "y"]))) == [
            "x0",
            "x2",
        ]
    assert "environments[default].exclude_tasks: ['y'] match none of the 3 task ids the dataset loader" in caplog.text
    assert _ids(_config(EnvironmentConfig(dataset_loader=f"{HERE}:load_hf_dataset", exclude_tasks=["d1"]))) == ["d0"]

    rows = EnvironmentConfig(name="rows", dataset_loader=f"{HERE}:load_rows", exclude_tasks=["r0"])
    hf = EnvironmentConfig(name="hf", dataset_loader=f"{HERE}:load_hf_dataset", exclude_tasks=["d1"])
    listed = AutoDataset.from_config(_config(rows, hf), "eval").to_list()
    assert [(row["task_id"], row["level"]) for row in listed] == [("rows::r1", 2), ("hf::d0", None)]

    opaque = EnvironmentConfig(dataset_loader=f"{HERE}:load_opaque", exclude_tasks=["x"])
    assert "needs the task ids its dataset loader returns" in _problem(opaque)
    assert AutoDataset.from_config(_config(EnvironmentConfig(dataset_loader=f"{HERE}:load_opaque")), "eval") == (
        "dataset",
        "eval",
    )


# --- mixtures and records ------------------------------------------------------------------------------------------


def test_mixture_entries_leave_out_their_own_tasks_and_every_entrys_unknown_items_are_reported_together():
    def entry(name: str, exclude_tasks: list[str]) -> EnvironmentConfig:
        return EnvironmentConfig(
            name=name, service=COUNTER, exclude_tasks=exclude_tasks, eval_dataset_kwargs={"split_name": "train"}
        )

    rows = AutoDataset.from_config(_config(entry("a", ["train-0"]), entry("b", ["train:2"])), "eval").to_list()
    assert [(row["task_id"], row[ENVIRONMENT_COLUMN]) for row in rows] == [
        ("a::train:1", "a"),
        ("a::train:2", "a"),
        ("b::train:0", "b"),
        ("b::train:1", "b"),
    ]
    with pytest.raises(ConfigError) as error:
        AutoDataset.from_config(_config(entry("a", ["nope-a"]), entry("b", ["nope-b"])), "eval")
    assert [problem.split(":")[0] for problem in error.value.problems] == [
        "environments[a].exclude_tasks",
        "environments[b].exclude_tasks",
    ]


def test_each_list_reports_what_it_left_out():
    test_split = {"split_name": "test"}
    counter = EnvironmentConfig(
        name="counter", service=COUNTER, exclude_tasks=["train-2", "train:0"], eval_dataset_kwargs=test_split
    )
    plain = EnvironmentConfig(name="plain", service=COUNTER, eval_dataset_kwargs=test_split)
    exclusions: list[TaskExclusion] = []

    AutoDataset.from_config(_config(counter, plain), "train", exclusions=exclusions)
    AutoDataset.from_config(_config(counter, plain), "eval", exclusions=exclusions)
    assert exclusions == [
        TaskExclusion("counter", "train", 3, ("train:0", "train:2")),
        TaskExclusion("counter", "eval", 3, ()),
    ]


def test_the_exclusions_go_into_the_lock_json_a_command_wrote(tmp_path):
    exclusions = [TaskExclusion("a", "train", 3, ("train:1",)), TaskExclusion("a", "eval", 2, ())]
    record_exclusions(tmp_path, exclusions)
    assert not (tmp_path / "lock.json").exists()  # no command started this run

    (tmp_path / "lock.json").write_text(json.dumps({"argv": ["platoon", "train"]}))
    record_exclusions(tmp_path, exclusions)
    assert json.loads((tmp_path / "lock.json").read_text()) == {
        "argv": ["platoon", "train"],
        "excluded_tasks": {
            "a": {"train": {"listed": 3, "excluded": ["train:1"]}, "eval": {"listed": 2, "excluded": []}}
        },
    }
    (tmp_path / "lock.json").write_text("[]")
    assert update_run_lock(tmp_path, {"excluded_tasks": {}}) is False
    assert (tmp_path / "lock.json").read_text() == "[]"


# --- the commands --------------------------------------------------------------------------------------------------


async def rollout(task: Task, config) -> dict:
    return {}


async def answer(task: Task, config) -> dict:
    return {
        "trajectories": {
            "root": {"id": "root", "task": {"id": task.id}, "parent_info": None, "reward": 1.0, "steps": [{}]}
        }
    }


class ReachedTrainer(Exception):
    """The trainer was built: it opens Tinker, so a refused config must stop before it."""


@pytest.fixture
def train_config(tmp_path, monkeypatch):
    """Write a training config over the counter environment; building the trainer raises ``ReachedTrainer``."""

    def trainer(**kwargs):
        raise ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")

    def write(exclude_tasks: list[str], environment: dict[str, Any] | None = None) -> str:
        """The config's path; its entry is ``environment`` (default: the counter environment's) with
        ``exclude_tasks``."""

        path = tmp_path / "train.yaml"
        if environment is None:
            environment = {"name": "counter", "service": COUNTER, "rollout": f"{HERE}:rollout"}
        settings = {
            "environments": [{**environment, "exclude_tasks": exclude_tasks}],
            "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 1},
            "eval": {"every": 0},
            "log_path": str(tmp_path / "logs"),
            "stats": {"experiment_name": "exp", "trial_name": "trial"},
        }
        path.write_text(yaml.safe_dump(settings))
        return str(path)

    return write


def test_platoon_train_records_the_left_out_tasks_in_lock_json(train_config, tmp_path):
    with pytest.raises(ReachedTrainer):
        cli.main(["train", train_config(["train-1"])])

    lock = json.loads((tmp_path / "logs" / "exp" / "trial" / "lock.json").read_text())
    assert lock["argv"][:2] == ["platoon", "train"]
    assert lock["excluded_tasks"] == {"counter": {"train": {"listed": 3, "excluded": ["train:1"]}}}


def test_platoon_train_exits_78_naming_every_unknown_item_before_the_trainer(train_config, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", train_config(["train-1", "trian-2", "train:9"])])

    assert exit_info.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "environments[counter].exclude_tasks: ['trian-2', 'train:9'] match no task id or name" in err
    assert "Traceback" not in err


def test_platoon_eval_runs_the_tasks_that_remain_and_records_the_rest(tmp_path, capsys):
    path = tmp_path / "eval.yaml"
    environment = {"service": COUNTER, "rollout": f"{HERE}:answer", "eval_dataset_kwargs": {"split_name": "test"}}
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {"model_name": "fake-model", "output_dir": str(tmp_path / "results")},
                "environments": [{**environment, "exclude_tasks": ["test-0"]}],
            }
        )
    )

    cli.main(["eval", str(path)])

    assert json.loads(capsys.readouterr().out)["total_tasks"] == 2
    lock = json.loads((tmp_path / "results" / "lock.json").read_text())
    assert lock["excluded_tasks"] == {"default": {"eval": {"listed": 3, "excluded": ["test:0"]}}}


# --- named environments --------------------------------------------------------------------------------------------

# acme-counter@1: the counter environment, whose training tasks are a seeded sample of four from both splits.
NAMED_COUNTER = f"""
import platoon as pt

COUNTER = pt.EnvironmentEntry(
    "acme-counter",
    version="1",
    description="Count up to each task's target.",
    service="{COUNTER}",
    dataset_kwargs={{"split_name": ["train", "test"], "shuffle_seed": 3, "limit": 4}},
    eval_dataset_kwargs={{"split_name": "test"}},
)
"""


@pytest.fixture
def acme_counter(tmp_path, monkeypatch):
    """``acme-counter@1``, which the distribution ``acme-counter`` 1.2.0 installs into a scratch site."""

    with sandbox(tmp_path, monkeypatch) as site:
        site.module("acme_counter/entries.py", NAMED_COUNTER)
        entry_points = {"platoon.environments": {"acme-counter": "acme_counter.entries:COUNTER"}}
        site.install("acme-counter", "1.2.0", entry_points=entry_points)
        yield site


def test_an_entry_that_selects_a_named_environment_leaves_tasks_out_before_its_limit(acme_counter):
    (entry,) = normalize_environment_configs([{"env": "acme-counter@1", "exclude_tasks": ["train:1", "test-2"]}])
    sample = EnvironmentConfig(service=COUNTER, dataset_kwargs={"split_name": ["train", "test"], "shuffle_seed": 3})
    remaining = [task_id for task_id in _ids(_config(sample)) if task_id not in ("train:1", "test:2")]

    assert (entry.env, entry.exclude_tasks) == ("acme-counter@1", ["train:1", "test-2"])
    # The named environment's seeded sample and limit, drawn from the tasks that remain.
    assert _ids(_config(entry)) == remaining[:4]
    assert _ids(_config(entry), "eval") == ["test:0", "test:1"]


def test_one_lock_json_holds_the_components_the_parser_and_the_tasks_left_out(acme_counter, train_config, tmp_path):
    """The run record lists the components and the extensions behind them, the tool-call parser joins it, and the
    tasks left out join it once they are listed: none replaces another."""

    environment = {"name": "counter", "env": "acme-counter@1", "rollout": f"{HERE}:rollout"}
    with pytest.raises(ReachedTrainer):
        cli.main(["train", train_config(["train-1"], environment)])

    lock = json.loads((tmp_path / "logs" / "exp" / "trial" / "lock.json").read_text())
    components = {(item["kind"], item["ref"]): item for item in lock["components"]}
    named = components[("environment", "acme-counter@1")]
    assert (named["source"], named["distribution"], named["distribution_version"]) == (
        "acme_counter.entries:COUNTER",
        "acme-counter",
        "1.2.0",
    )
    assert {("service", COUNTER), ("rollout", f"{HERE}:rollout")} <= components.keys()
    assert lock["extensions"]["acme-counter"]["version"] == "1.2.0"
    assert lock["tool_call_parsing"] == {
        "model_name": "m",
        "renderer_name": "r",
        "parser": None,
        "reason": "the renderer parses tool calls itself",
    }
    assert lock["excluded_tasks"] == {"counter": {"train": {"listed": 6, "excluded": ["train:1"]}}}


@pytest.mark.parametrize(
    ("keys", "problem", "validate_code"),
    [
        # Found while the config loads, so config validate finds them too.
        (
            {"exclude_tasks": ["train-1 "]},
            "exclude_tasks items must be task ids or names, non-empty strings without surrounding spaces",
            cli.EXIT_CONFIG,
        ),
        ({"env": "acme-counter@2"}, "environments.0.env: Unknown environment 'acme-counter@2'", cli.EXIT_CONFIG),
        # Found while the components resolve, which config validate does too.
        (
            {"reward_processor": "acme-rewards"},
            "environments[counter].reward_processor: Unknown reward processor 'acme-rewards'",
            cli.EXIT_CONFIG,
        ),
        # Found while the tasks are listed, which only train and eval do.
        (
            {"exclude_tasks": ["train-1", "trian-2"]},
            "environments[counter].exclude_tasks: ['trian-2'] match no task id or name of the environment",
            0,
        ),
        (
            {"exclude_tasks": [f"{split}:{index}" for split in ("train", "test") for index in range(3)]},
            "environments[counter].exclude_tasks leaves none of the 6 tasks of split 'train'",
            0,
        ),
    ],
)
def test_each_kind_of_error_stops_a_run_on_a_named_environment_with_exit_78(
    acme_counter, train_config, capsys, keys, problem, validate_code
):
    environment = {"name": "counter", "env": "acme-counter@1", "rollout": f"{HERE}:rollout", **keys}
    path = train_config(environment.pop("exclude_tasks", []), environment)

    assert cli.config_validate(Path(path)) == validate_code
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", path])

    assert exit_info.value.code == cli.EXIT_CONFIG
    out, err = capsys.readouterr()
    assert problem in err and "Traceback" not in err
    assert (problem in out) == bool(validate_code)
