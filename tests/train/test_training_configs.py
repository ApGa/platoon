"""Training configs that load but cannot train are configuration errors, raised by ``arun_tinker_training``
before it lists tasks or opens Tinker; ``platoon train`` reports them in one line with exit code 78."""

from __future__ import annotations

import asyncio

import pytest
import yaml

from platoon import cli
from platoon.task import Task
from platoon.train import launch as train_module
from platoon.utils.config import ConfigError

TASKS = {"train": ["a", "b", "c", "d"], "eval": ["e"]}
LISTED: list[str] = []


def load_tasks(config, split, **kwargs) -> list[str]:
    LISTED.append(split)
    if split not in TASKS:
        raise ValueError(f"no split {split!r}")
    return TASKS[split]


def load_task(task_id: str) -> Task:
    return Task(id=task_id)


async def rollout(task: Task, config) -> dict:
    return {}


class ReachedTrainer(Exception):
    """The trainer was built: it opens Tinker, so a refused config must stop before it."""


@pytest.fixture
def train_config(tmp_path, monkeypatch):
    """Write a training config over a four-task environment; building the trainer raises ``ReachedTrainer``."""

    def trainer(**kwargs):
        raise ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    LISTED.clear()

    def write(train: dict | None = None, **top) -> str:
        path = tmp_path / "train.yaml"
        environment = {
            "dataset_loader": f"{__name__}:load_tasks",
            "task_loader": f"{__name__}:load_task",
            "rollout": f"{__name__}:rollout",
        }
        settings = {
            "environments": [environment],
            "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, **(train or {})},
            "eval": {"every": 0},
            "log_path": str(tmp_path / "logs"),
            **top,
        }
        path.write_text(yaml.safe_dump(settings))
        return str(path)

    return write


def _train(path: str) -> None:
    asyncio.run(train_module.arun_tinker_training(["--config", path]))


@pytest.mark.parametrize("duration", [{"num_epochs": 1}, {"max_training_steps": 1}])
def test_a_config_with_a_duration_reaches_the_trainer(train_config, duration):
    with pytest.raises(ReachedTrainer):
        _train(train_config(duration))


def test_a_config_without_a_duration_is_a_config_error_before_anything_runs(train_config):
    with pytest.raises(ConfigError, match="train: set num_epochs or max_training_steps"):
        _train(train_config())
    assert LISTED == []


def test_a_config_that_cannot_train_is_refused_before_the_key_is_asked_for(train_config, monkeypatch):
    """The config's own problems come first: without a duration and without ``TINKER_API_KEY``, the run is told of the
    duration, which it can fix in its config before it needs a key."""

    monkeypatch.delenv("TINKER_API_KEY")
    with pytest.raises(ConfigError, match="train: set num_epochs or max_training_steps"):
        _train(train_config())


def test_platoon_train_reports_a_config_without_a_duration_in_one_line(train_config, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", train_config()])

    assert exit_info.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert "train: set num_epochs or max_training_steps" in err and "Traceback" not in err


@pytest.mark.parametrize("section", ["train", "eval"])
def test_a_training_config_that_turns_judges_off_is_refused(train_config, section):
    """``rollout_config.judges`` is an evaluation's switch: training runs its judges (verifiers have their own
    switches), so a training config that turns them off stops before anything runs."""

    judges_off = {"workflow_config": {"rollout_config": {"judges": False}}}
    if section == "train":
        path = train_config({"max_training_steps": 1, **judges_off})
    else:
        path = train_config({"max_training_steps": 1}, eval={"every": 0, **judges_off})
    with pytest.raises(ConfigError, match=rf"{section}\.workflow_config\.rollout_config\.judges: false is an"):
        _train(path)
    assert LISTED == []


@pytest.mark.parametrize("evaluation", [{"every": 0}, {}], ids=["every-0", "the-default"])
def test_with_evaluation_off_the_eval_split_need_not_exist(train_config, monkeypatch, evaluation):
    monkeypatch.delitem(TASKS, "eval")
    with pytest.raises(ReachedTrainer):
        _train(train_config({"max_training_steps": 1}, eval=evaluation))
    assert LISTED == ["train"]


def test_with_evaluation_on_the_eval_split_is_listed(train_config):
    with pytest.raises(ReachedTrainer):
        _train(train_config({"max_training_steps": 1}, eval={"every": 1}))
    assert LISTED == ["train", "eval"]


class _RecordingWorkflow:
    """Stands in for the rollout workflow class: records the ``filter_errors`` each phase is built with."""

    built: dict[str, bool] = {}

    def __init__(self, **kwargs):
        self.built[kwargs["stats_scope"]] = kwargs["filter_errors"]


class _Trainer:
    def __init__(self, **kwargs):
        self.model_info, self.run_log_path = None, "logs"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def train(self, **workflows):
        pass


@pytest.fixture
def workflow_filters(train_config, monkeypatch):
    """``filter_errors`` of the train and eval workflows ``arun_tinker_training`` builds for a config."""

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", _Trainer)
    monkeypatch.setattr(
        train_module.AutoWorkflow, "from_config", classmethod(lambda cls, config, default: _RecordingWorkflow)
    )

    def built(**top) -> dict[str, bool]:
        _RecordingWorkflow.built = {}
        _train(train_config({"max_training_steps": 1}, **top))
        return dict(_RecordingWorkflow.built)

    return built


def test_eval_filter_errors_comes_from_the_eval_section(workflow_filters):
    assert workflow_filters(eval={"every": 0, "workflow_config": {"filter_errors": True}})["eval"] is True
    # A partial eval section and no eval section keep the eval default, false, as the hard-coded value was.
    assert workflow_filters(eval={"every": 0, "workflow_config": {"group_size": 1}})["eval"] is False
    assert workflow_filters(eval={"every": 0})["eval"] is False


def test_an_environments_eval_workflow_kwargs_still_override_the_eval_section(workflow_filters, tmp_path):
    environment = {
        "dataset_loader": f"{__name__}:load_tasks",
        "task_loader": f"{__name__}:load_task",
        "rollout": f"{__name__}:rollout",
        "eval_workflow_kwargs": {"filter_errors": True},
    }
    assert workflow_filters(environments=[environment], eval={"every": 0})["eval"] is True


def test_a_batch_larger_than_the_training_set_is_a_config_error(train_config):
    with pytest.raises(ConfigError, match="train.batch_size: 8 is more than the 4 training tasks"):
        _train(train_config({"batch_size": 8, "max_training_steps": 1}))


def test_a_batch_the_training_set_fills_reaches_the_trainer(train_config):
    with pytest.raises(ReachedTrainer):
        _train(train_config({"batch_size": 4, "max_training_steps": 1}))


def test_the_weighted_order_fills_a_batch_larger_than_the_training_set(train_config):
    with pytest.raises(ReachedTrainer):
        _train(train_config({"batch_size": 8, "max_training_steps": 1, "task_sampler": "weighted"}))
