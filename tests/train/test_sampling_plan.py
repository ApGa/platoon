"""How a run draws its training tasks (``platoon.train.sampling``): each entry's share, how often each of its tasks is
drawn over the planned steps, and the warnings for weights that leave tasks out of the run or draw some far more often
than others, which ``platoon train`` logs and ``platoon config validate --tasks`` prints."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import yaml

from platoon import cli
from platoon.train.auto import AutoDataset
from platoon.train.components import EnvironmentConfig
from platoon.train.sampling import REPEAT_RATIO_LIMIT, SamplingPlan, log_sampling_plan, planned_steps, sampling_plan

HERE = __name__


def load_ids(config, split, *, prefix, count):
    return [f"{prefix}{index}" for index in range(count)]


async def rollout(task, config) -> dict:
    return {}


def _entries(*sizes: tuple[str, int], weight: float | Literal["tasks"] = 1.0) -> list[EnvironmentConfig]:
    return [
        EnvironmentConfig(
            name=name, dataset_loader=f"{HERE}.load_ids", dataset_kwargs={"prefix": name, "count": count}, weight=weight
        )
        for name, count in sizes
    ]


def _plan(
    *sizes: tuple[str, int], weight: float | Literal["tasks"] = 1.0, batch_size=32, epochs=10, steps=None
) -> SamplingPlan:
    """The plan AutoDataset.training builds for a mixture of entries of the given sizes."""

    train = SimpleNamespace(batch_size=batch_size, task_sampler="auto", num_epochs=epochs, max_training_steps=steps)
    plan = AutoDataset.training(SimpleNamespace(environments=_entries(*sizes, weight=weight), train=train)).plan
    assert plan is not None
    return plan


def test_equal_weights_over_475_tasks_and_22_warn_that_the_22_repeat_about_22_times_as_often():
    plan = _plan(("science", 475), ("small", 22))

    assert (plan.tasks, plan.epoch_size, plan.steps, plan.slots) == (497, 512, 160, 5120)
    assert plan.describe() == [
        "training tasks: 497, drawn in epochs of 512 (16 batches of 32), 160 steps planned",
        "environments[science]: 475 tasks, weight 1, 50.0% of the slots: each task drawn about 5.4 times over the "
        "run's 160 steps",
        "environments[small]: 22 tasks, weight 1, 50.0% of the slots: each task drawn about 116.4 times over the "
        "run's 160 steps",
    ]
    assert plan.warnings() == [
        "environments[small]'s 22 training tasks are each drawn about 21.6 times as often as environments[science]'s "
        "475 (about 116.4 and 5.4 times each over the run's 160 steps, at weights 1 and 1); weight: tasks on every "
        "entry draws every task of the mixture equally often"
    ]


def test_a_run_too_short_for_an_entrys_share_warns_how_many_of_its_tasks_never_train():
    """20 steps of 32 are 640 slots, enough for each of the 497 tasks once with weight: tasks, but half of them are 320
    of science's 475 tasks."""

    plan = _plan(("science", 475), ("small", 22), epochs=None, steps=20)

    assert plan.warnings()[0] == (
        "environments[science]: its weight (1, 50.0% of the slots) gives it about 320 of the run's 640 task slots (20 "
        "steps of 32), fewer than its 475 training tasks, so about 155 of them never train; weight: tasks on every "
        "entry draws every task of the mixture equally often"
    )
    # A run too short for every task once whatever the weights is not the weights' doing: no such warning.
    short = _plan(("a", 475), ("b", 22), epochs=None, steps=10)
    assert not any("never train" in warning for warning in short.warnings())


def test_weight_tasks_draws_every_task_equally_often_and_warns_of_nothing():
    plan = _plan(("science", 475), ("small", 22), weight="tasks", epochs=None, steps=20)

    assert [round(plan.draws(entry), 6) for entry in plan.entries] == [round(640 / 497, 6)] * 2
    assert [entry.share for entry in plan.entries] == pytest.approx([475 / 497, 22 / 497])
    assert plan.describe()[1].startswith("environments[science]: 475 tasks, weight tasks, 95.6% of the slots")
    assert plan.warnings() == []


@pytest.mark.parametrize(
    ("weights", "warned"),
    [((1.0, 1.0), True), ((10.0, 1.0), False), ((475.0, 88.0), False), ((475.0, 89.0), True)],
)
def test_the_repeat_warning_starts_above_the_limit(weights, warned):
    """22 tasks drawn 4 times as often as 475 is the limit: weights 475 and 88; 89 is above it."""

    entries = [
        EnvironmentConfig(name=name, weight=weight) for name, weight in zip(("science", "small"), weights, strict=True)
    ]
    plan = sampling_plan(
        entries,
        {"science": 475, "small": 22},
        dict(zip(("science", "small"), weights, strict=True)),
        train=SimpleNamespace(batch_size=32, num_epochs=10, max_training_steps=None),
        epoch_size=512,
    )

    assert any("times as often as" in warning for warning in plan.warnings()) == warned
    assert REPEAT_RATIO_LIMIT == 4.0


def test_without_a_planned_length_the_plan_counts_per_epoch():
    plan = _plan(("science", 475), ("small", 22), epochs=None, steps=None)

    assert plan.steps is None and plan.slots is None
    assert plan.describe()[0] == "training tasks: 497, drawn in epochs of 512 (16 batches of 32)"
    assert plan.describe()[2].endswith("each task drawn about 11.6 times per epoch")
    assert "times each per epoch" in plan.warnings()[0] and len(plan.warnings()) == 1


def test_a_single_environment_is_described_and_warns_of_nothing():
    train = SimpleNamespace(batch_size=32, task_sampler="auto", num_epochs=2, max_training_steps=None)
    plan = AutoDataset.training(SimpleNamespace(environments=_entries(("solo", 475))[:1], train=train)).plan

    assert plan is not None
    assert plan.describe() == [
        "training tasks: 475, drawn in epochs of 448 (14 batches of 32), 28 steps planned",
        "environments[solo]: 475 tasks, weight 1, 100.0% of the slots: each task drawn about 1.9 times over the run's "
        "28 steps",
    ]
    assert plan.warnings() == []


@pytest.mark.parametrize(
    ("epochs", "steps", "expected"),
    [(3, None, 30), (None, 7, 7), (3, 25, 25), (3, 40, 30), (None, None, None)],
)
def test_planned_steps_count_as_the_trainer_does(epochs, steps, expected):
    from platoon.train.tinker.rl import TrainLoopSharedState

    train = SimpleNamespace(num_epochs=epochs, max_training_steps=steps)
    assert planned_steps(train, 10) == expected
    state: Any = object.__new__(TrainLoopSharedState)
    state.config = SimpleNamespace(train=train)
    state.train_dataloader = [None] * 10  # 10 batches an epoch
    if expected is None:
        with pytest.raises(ValueError, match="num_epochs or max_training_steps"):
            state.num_train_batches
    else:
        assert state.num_train_batches == expected


def test_the_plan_is_logged_with_its_warnings(caplog):
    with caplog.at_level(logging.INFO, logger="platoon.train.sampling"):
        log_sampling_plan(_plan(("science", 475), ("small", 22)))

    infos = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert infos[0] == "training tasks: 497, drawn in epochs of 512 (16 batches of 32), 160 steps planned"
    assert len(infos) == 3 and len(warnings) == 1 and "about 21.6 times as often" in warnings[0]


# --- platoon train and platoon config validate --tasks ---------------------------------------------------------------


class _ReachedTrainer(Exception):
    pass


def _config(tmp_path: Path, **weights) -> Path:
    environments = [
        {
            "name": name,
            "dataset_loader": f"{HERE}:load_ids",
            "dataset_kwargs": {"prefix": name, "count": count},
            "rollout": f"{HERE}:rollout",
            **({"weight": weights[name]} if name in weights else {}),
        }
        for name, count in (("science", 475), ("small", 22))
    ]
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "environments": environments,
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 32, "num_epochs": 10},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    return path


def test_platoon_train_logs_the_plan_and_warns_once_it_lists_the_tasks(monkeypatch, tmp_path, caplog):
    from platoon.train.tinker import train as train_module

    def trainer(**kwargs):
        raise _ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")

    with caplog.at_level(logging.INFO, logger="platoon.train.sampling"), pytest.raises(_ReachedTrainer):
        cli.main(["train", str(_config(tmp_path, science=1, small=1))])

    messages = [record.getMessage() for record in caplog.records if record.name == "platoon.train.sampling"]
    assert "training tasks: 497, drawn in epochs of 512 (16 batches of 32), 160 steps planned" in messages
    assert any("about 21.6 times as often as environments[science]'s 475" in message for message in messages)


def test_config_validate_lists_the_tasks_and_prints_the_plan_only_when_asked(tmp_path, capsys):
    path = _config(tmp_path, science=1, small=1)

    assert cli.config_validate(path) == 0
    assert "training tasks" not in capsys.readouterr().out  # listing reads datasets and env servers: on request
    assert cli.config_validate(path, tasks=True) == 0

    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"ok   {path} (PlatoonTinkerRLTrainerConfig)"
    assert out[1] == "  training tasks: 497, drawn in epochs of 512 (16 batches of 32), 160 steps planned"
    assert out[2].startswith("  environments[science]: 475 tasks, weight 1, 50.0% of the slots")
    assert out[4].startswith("  warning: environments[small]'s 22 training tasks are each drawn about 21.6 times")
    assert cli.config_validate(path, "environments.0.weight=tasks", "environments.1.weight=tasks", tasks=True) == 0
    assert "warning" not in capsys.readouterr().out


def test_config_validate_tasks_fails_on_what_listing_the_tasks_finds(tmp_path, capsys):
    path = _config(tmp_path)
    none_left = ["environments.1.dataset_kwargs.count=2", "environments.1.exclude_tasks=[small0, small1]"]

    assert cli.config_validate(path, *none_left) == 0  # found while listing the tasks, which validate does when asked
    assert cli.config_validate(path, *none_left, tasks=True) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert f"FAIL {path}" in out and "environments[small].exclude_tasks leaves none of the 2 tasks" in out
    one = tmp_path / "one.yaml"
    one.write_text(path.read_text())
    single = [
        f"environments=[{{dataset_loader: {HERE}:load_ids, dataset_kwargs: {{prefix: a, count: 20}}, "
        f"rollout: {HERE}:rollout}}]"
    ]
    assert cli.config_validate(one, *single, tasks=True) == cli.EXIT_CONFIG
    assert "train.batch_size: 32 is more than the 20 training tasks" in capsys.readouterr().out
    assert cli.config_validate(one, *single, "train.batch_size=8", tasks=True) == 0
    assert "  training tasks: 20, drawn in epochs of 16 (2 batches of 8), 20 steps planned" in capsys.readouterr().out
