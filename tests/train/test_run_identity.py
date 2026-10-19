"""One run per run directory (``identity.json``): a resume that would change the model, the renderer or the held-out
tasks stops with exit 78 before the trainer is built, and records why in ``lock.json``; and the warning for an
evaluation that lists training tasks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.envs.harbor.holdout import HoldoutRow, HoldoutRule, hold_out
from platoon.task import Task
from platoon.train.components import EnvironmentConfig
from platoon.train.identity import (
    IDENTITY_NAME,
    LOCK_KEY,
    check_run_identity,
    entry_task_ids,
    identity_changes,
    warn_eval_overlap,
)
from platoon.train.tinker import train as train_module
from platoon.utils.config import ConfigError

GROUPS = 20  # one task per group, in two sources
HOLDOUT = {"fraction": 0.25, "group_by": "group", "stratify_by": "source"}


async def rollout(task: Task, config) -> dict:
    return {}


class ReachedTrainer(Exception):
    """The trainer was built: it opens Tinker, so a refused start must stop before it."""


def _dataset(root: Path) -> Path:
    rows = {}
    for n in range(GROUPS):
        name = f"task-{n:02d}"
        for relative, text in {
            "task.toml": "",
            "instruction.md": f"Task {n}.\n",
            "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        }.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        rows[name] = {"source": f"s{n % 2}", "group": f"g{n}", "family": f"f{n // 4}"}
    (root / "task-index.json").write_text(json.dumps(rows))
    return root


def _checkpoint(run_dir: Path) -> None:
    """A checkpoint the trainer would resume from."""

    run_dir.mkdir(parents=True, exist_ok=True)
    row = {"name": "000003", "batch": 3, "state_path": "tinker://run/state/3", "sampler_path": "tinker://run/s/3"}
    (run_dir / "checkpoints.jsonl").write_text(json.dumps(row) + "\n")


@pytest.fixture
def train(tmp_path, monkeypatch):
    """``platoon train`` on a Harbor dataset directory with a hold-out; building the trainer raises
    ``ReachedTrainer``.  Returns a function of overrides that gives None when the trainer was reached, else the exit
    status."""

    def trainer(**kwargs):
        raise ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    dataset = _dataset(tmp_path / "dataset")
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environments": [
                    {
                        "name": "science",
                        "service": "harbor",
                        "service_kwargs": {"dataset_dir": str(dataset), "holdout": HOLDOUT},
                        "rollout": f"{__name__}:rollout",
                    }
                ],
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 1},
                "eval": {"every": 1},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
            }
        )
    )

    def run(*overrides: str) -> int | None:
        try:
            cli.main(["train", str(config), *overrides])
        except ReachedTrainer:
            return None
        except SystemExit as exit_info:
            return int(exit_info.code or 0)
        raise AssertionError("platoon train neither reached the trainer nor exited")

    return run


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "exp" / "trial"


def _held(seed: int, group_by: str = "group") -> set[str]:
    rows = [
        HoldoutRow(f"task-{n:02d}", {"source": f"s{n % 2}", "group": f"g{n}", "family": f"f{n // 4}"}, input_files={})
        for n in range(GROUPS)  # the tasks have no environment/: no input files
    ]
    return set(hold_out(rows, HoldoutRule.coerce({**HOLDOUT, "seed": seed, "group_by": group_by})).held_out)


def test_a_fresh_start_writes_the_identity_and_the_same_config_resumes(train, tmp_path):
    assert train() is None
    record = json.loads((_run_dir(tmp_path) / IDENTITY_NAME).read_text())
    assert (record["model"], record["renderer"]) == ("m", "r")
    science = record["environments"]["science"]
    assert science["holdout"] == {"fraction": 0.25, "group_by": ["group"], "stratify_by": "source", "seed": 0}
    assert set(science["held_out"]) == _held(0) and science["eval"] == sorted(_held(0))

    _checkpoint(_run_dir(tmp_path))
    assert train() is None  # the same run resumes
    assert json.loads((_run_dir(tmp_path) / IDENTITY_NAME).read_text())["written_at"] == record["written_at"]


@pytest.mark.parametrize(
    ("override", "problem"),
    [
        ("environments.0.service_kwargs.holdout.seed=5", "task(s) the run held out would now train"),
        ("environments.0.service_kwargs.holdout.fraction=0.5", "task(s) are now held out, and the run did not hold"),
        ("environments.0.service_kwargs.holdout.group_by=family", "task(s) the run held out would now train"),
        ("train.model_name=m2", "the run trains model 'm', and this start 'm2'"),
        ("train.renderer_name=r2", "the run trains renderer 'r', and this start 'r2'"),
    ],
)
def test_a_resume_that_changes_the_run_exits_78_and_records_why(train, tmp_path, capsys, override, problem):
    assert _held(0) != _held(5) and _held(0) != _held(0, "family")  # the overrides do move held-out tasks
    assert train() is None
    _checkpoint(_run_dir(tmp_path))
    capsys.readouterr()

    assert train(override) == cli.EXIT_CONFIG  # never reached the trainer

    err = capsys.readouterr().err
    assert "holds a run this start would change, so it does not resume it" in err and problem in err
    assert "Traceback" not in err
    refused = json.loads((_run_dir(tmp_path) / "lock.json").read_text())[LOCK_KEY]
    assert refused["identity"] == str(_run_dir(tmp_path) / IDENTITY_NAME)
    assert any(problem in line for line in refused["problems"])


def load_many(config, split):
    return [f"other-{n}" for n in range(100)]


def test_a_resume_reads_every_task_a_mixture_trains_on_not_only_its_first_epoch(tmp_path, monkeypatch, capsys):
    """A mixture's light Harbor entry gets no slot of the first epoch, but the next epochs go on with the order and
    draw its tasks: a changed hold-out that would train tasks the run held out is refused, though the first epoch
    holds none of them."""

    from platoon.train.auto import AutoDataset
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    def trainer(**kwargs):
        raise ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    dataset = _dataset(tmp_path / "dataset")
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "environments": [
                    {
                        "name": "science",
                        "service": "harbor",
                        "service_kwargs": {"dataset_dir": str(dataset), "holdout": HOLDOUT},
                        "rollout": f"{__name__}:rollout",
                        "weight": 1,
                    },
                    {
                        "name": "other",
                        "dataset_loader": f"{__name__}:load_many",
                        "rollout": f"{__name__}:rollout",
                        "weight": 1000,
                    },
                ],
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 1},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "stats": {"experiment_name": "exp", "trial_name": "trial"},
            }
        )
    )
    moved = "environments.0.service_kwargs.holdout.seed=5"

    def run(*overrides: str) -> int | None:
        try:
            cli.main(["train", str(config), *overrides])
        except ReachedTrainer:
            return None
        except SystemExit as exit_info:
            return int(exit_info.code or 0)
        raise AssertionError("platoon train neither reached the trainer nor exited")

    assert run() is None
    _checkpoint(_run_dir(tmp_path))
    resolved = resolve_config(config, [moved], PlatoonTinkerRLTrainerConfig).config
    training = AutoDataset.training(resolved)
    first_epoch = {task_id for task_id in training.dataset["task_id"] if task_id.startswith("science::")}
    assert first_epoch == set() and _held(0) - _held(5)  # the first epoch holds none of the tasks that move
    capsys.readouterr()

    assert run(moved) == cli.EXIT_CONFIG

    assert "task(s) the run held out would now train" in capsys.readouterr().err


def test_a_hold_out_added_to_a_running_run_is_refused(train, tmp_path, capsys):
    assert train("environments.0.service_kwargs.holdout=null") is None
    assert json.loads((_run_dir(tmp_path) / IDENTITY_NAME).read_text())["environments"] == {}
    _checkpoint(_run_dir(tmp_path))
    capsys.readouterr()

    assert train() == cli.EXIT_CONFIG

    assert "task(s) are now held out, and the run held out no task" in capsys.readouterr().err


def test_a_resume_where_only_a_refusal_changed_goes_on_and_says_so(train, tmp_path, caplog):
    """A task that cannot load now is refused: it leaves the lists, but the walk, which counts refused tasks too, holds
    out the same tasks, so the run resumes, and logs that the task left the evaluation."""

    assert train() is None
    _checkpoint(_run_dir(tmp_path))
    held = sorted(_held(0))
    (tmp_path / "dataset" / held[0] / "instruction.md").unlink()  # refused from now on (task_config)

    with caplog.at_level(logging.INFO, logger="platoon.train.identity"):
        assert train() is None
    assert f"1 held-out task(s) are refused now and leave the evaluation, such as ['{held[0]}']" in caplog.text
    assert set(json.loads((_run_dir(tmp_path) / IDENTITY_NAME).read_text())["environments"]["science"]["held_out"]) == (
        _held(0)
    )


def test_without_a_checkpoint_the_identity_is_written_afresh(train, tmp_path):
    assert train() is None
    assert train("environments.0.service_kwargs.holdout.seed=5") is None  # nothing trained yet: a new run

    record = json.loads((_run_dir(tmp_path) / IDENTITY_NAME).read_text())
    assert set(record["environments"]["science"]["held_out"]) == _held(5)


def test_a_resume_without_a_record_may_not_add_a_hold_out(tmp_path, monkeypatch):
    """A run started before platoon wrote identity.json resumes and gets one, unless it holds tasks out now."""

    run_dir = tmp_path / "run"
    _checkpoint(run_dir)
    identity: dict[str, Any] = {"model": "m", "renderer": "r", "environments": {}}
    check_run_identity(run_dir, identity, {})
    assert json.loads((run_dir / IDENTITY_NAME).read_text())["model"] == "m"

    (run_dir / IDENTITY_NAME).unlink()
    held = {"science": {"held_out": ["a"], "eval": ["a"]}}
    with pytest.raises(ConfigError, match=r"environments\[science\] holds tasks out, and the run recorded no hold-out"):
        check_run_identity(run_dir, {**identity, "environments": held}, {})
    (run_dir / IDENTITY_NAME).write_text("not json")
    with pytest.raises(ConfigError, match="cannot be read"):
        check_run_identity(run_dir, identity, {})


def test_held_out_tasks_that_only_left_the_selection_are_logged(tmp_path, caplog):
    before = {"model": "m", "renderer": "r", "environments": {"science": {"held_out": ["a", "b", "c"]}}}
    now = {"model": "m", "renderer": "r", "environments": {"science": {"held_out": ["a", "b"]}}}

    problems, notes = identity_changes(before, now, {"science": {"d", "e"}})
    assert problems == [] and notes == [
        "environments[science]: 1 task(s) the run held out left the selection since it started, such as ['c']; the "
        "evaluation goes on without them"
    ]
    rejoined = identity_changes(
        {**before, "environments": {"science": {"held_out": ["a", "b"], "eval": ["a"]}}},
        {**now, "environments": {"science": {"held_out": ["a", "b"], "eval": ["a", "b"]}}},
        {},
    )
    assert rejoined == (
        [],
        ["environments[science]: 1 held-out task(s) are runnable again and join the evaluation, such as ['b']"],
    )

    run_dir = tmp_path / "run"
    _checkpoint(run_dir)
    (run_dir / IDENTITY_NAME).write_text(json.dumps(before))
    with caplog.at_level(logging.INFO, logger="platoon.train.identity"):
        check_run_identity(run_dir, now, {"science": {"d", "e"}})
    assert "1 task(s) the run held out left the selection since it started, such as ['c']" in caplog.text


# --- the evaluation-overlap warning ----------------------------------------------------------------------------------


def _entries(*names: str) -> list[EnvironmentConfig]:
    return [EnvironmentConfig(name=name, service="harbor") for name in names]


def test_the_overlap_warning_names_each_entry_whose_evaluation_lists_training_tasks(caplog):
    mixture = _entries("plain", "held")
    train = [{"task_id": "plain::a"}, {"task_id": "plain::b"}, {"task_id": "held::x"}]
    evaluate = [{"task_id": "plain::a"}, {"task_id": "plain::b"}, {"task_id": "held::y"}]

    with caplog.at_level(logging.WARNING, logger="platoon.train.identity"):
        warned = warn_eval_overlap(mixture, train, evaluate)

    assert warned == ["plain"]
    assert (
        "environments[plain]: the evaluation lists 2 of its 2 task(s) among the tasks training draws from"
        in caplog.text
    )
    assert entry_task_ids(train, mixture) == {"plain": {"a", "b"}, "held": {"x"}}
    assert warn_eval_overlap(_entries("single"), ["a", "b"], ["c"]) == []


def test_platoon_train_warns_when_an_entry_without_a_hold_out_evaluates_its_training_tasks(train, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="platoon.train.identity"):
        assert train("environments.0.service_kwargs.holdout=null") is None
    assert "environments[science]: the evaluation lists 20 of its 20 task(s)" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="platoon.train.identity"):
        assert train("stats.trial_name=held") is None
    assert "the evaluation lists" not in caplog.text


def test_entry_task_ids_reads_lists_rows_and_datasets():
    from datasets import Dataset

    single = _entries("only")
    assert entry_task_ids(Dataset.from_list([{"task_id": "a"}, {"task_id": "b"}]), single) == {"only": {"a", "b"}}
    assert entry_task_ids(["a", {"task_id": "b"}], single) == {"only": {"a", "b"}}
    assert entry_task_ids(None, single) == {"only": set()}
    assert entry_task_ids(SimpleNamespace(), single) == {"only": set()}
