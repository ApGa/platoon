"""Training draws a mixture's tasks epoch after epoch of their weighted order, and a resumed run goes on with it.

The trainer's own loop (``PlatoonTinkerRLTrainer.train``) runs with Tinker faked (``fakes.trainer``) on the order that
``AutoDataset.training`` builds for a mixture: handed the order alone, the trainer draws its dataloader's batches, which
are recorded; handed the task samplers too (``sampling``), as ``platoon train`` hands them, it draws through them, and
the rows the groups run are recorded by position.  The checkpoint a run resumes from records the batch it stopped at,
and the draws depend on their position alone, so a resumed run draws what the run it resumes would have drawn.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fakes.config import make_config, trainer_config
from fakes.trainer import StepLog, Workflow, make_trainer

from platoon.train import dataloader as dataloader_module
from platoon.train.auto import AutoDataset
from platoon.train.dispatch.dispatcher import POSITION_COLUMN

HERE = __name__


def load_ids(config, split, *, prefix, count):
    return [f"{prefix}{index}" for index in range(count)]


def _training(batch_size: int):
    """A mixture of 6 tasks and 2, weighted 1 to 1, as AutoDataset.training lists and orders it."""

    def environment(name: str, count: int) -> dict:
        return {
            "name": name,
            "dataset_loader": f"{HERE}.load_ids",
            "dataset_kwargs": {"prefix": name, "count": count},
            "seed": 3,
            "weight": 1.0,
        }

    config = make_config(
        environments=[environment("a", 6), environment("b", 2)],
        train={"batch_size": batch_size, "task_sampler": "auto"},
    )
    return AutoDataset.training(config)


def _record_batches(monkeypatch) -> list[tuple[int, list[str]]]:
    drawn: list[tuple[int, list[str]]] = []
    get_batch = dataloader_module.PlatoonTinkerDataloader.get_batch

    def recording(self, batch_num):
        batch = get_batch(self, batch_num)
        drawn.append((batch_num, [row["task_id"] for row in batch]))
        return batch

    monkeypatch.setattr(dataloader_module.PlatoonTinkerDataloader, "get_batch", recording)
    return drawn


def _train(tmp_path: Path, training, *overrides: str) -> StepLog:
    """Train on ``training``'s first epoch and order; the steps logged, with their stats."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    trainer, _saves, logged = make_trainer(
        trainer_config(tmp_path, *overrides), train_dataset=training.dataset, train_order=training.order
    )
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Workflow(delay=0.01)), 60))
    return logged


class _Positions(Workflow):
    """Records the task each position's group runs."""

    def __init__(self) -> None:
        super().__init__(delay=0.01)
        self.tasks: dict[int, str] = {}

    async def arun_episode(self, data):
        self.tasks[data[POSITION_COLUMN]] = data["task_id"]
        return await super().arun_episode(data)


def _train_through_samplers(tmp_path: Path, training, *overrides: str, resume_at: int | None = None):
    """Train as platoon train has the trainer train, through ``training``'s samplers, from batch ``resume_at``
    (the checkpoint a resume starts from) when given; the task each position ran."""

    tmp_path.mkdir(parents=True, exist_ok=True)
    trainer, _saves, _logged = make_trainer(
        trainer_config(tmp_path, *overrides),
        train_dataset=training.dataset,
        train_order=training.order,
        train_sampling=training.sampling,
    )
    if resume_at is not None:
        run_dir = Path(trainer.run_log_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "name": f"{resume_at:06d}",
            "batch": resume_at,
            "state_path": f"tinker://m/weights/{resume_at}",
            "sampler_path": f"tinker://m/s/{resume_at}",
        }
        (run_dir / "checkpoints.jsonl").write_text(json.dumps(record) + "\n")
    workflow = _Positions()
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
    return workflow.tasks


def _batches(order, *, epochs: int, batch_size: int) -> list[list[str]]:
    rows = [row["task_id"] for epoch in range(epochs) for row in order.epoch(epoch)]
    return [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]


def test_training_draws_each_epoch_of_the_weighted_order_in_turn(monkeypatch, tmp_path):
    training = _training(batch_size=2)
    drawn = _record_batches(monkeypatch)

    _train(tmp_path, training, "train.max_training_steps=12")

    expected = _batches(training.order, epochs=3, batch_size=2)
    assert [number for number, _ in drawn] == list(range(12))  # 4 batches per epoch, 3 epochs
    assert [batch for _, batch in drawn] == expected
    epochs = [[task for batch in expected[start : start + 4] for task in batch] for start in (0, 4, 8)]
    assert epochs[1] != epochs[0] and epochs[2] != epochs[0]  # not the first epoch again
    # a gets 4 of each epoch's 8 slots: its 6 tasks are all drawn within two epochs, each once before any twice.
    a_tasks = [task for epoch in epochs[:2] for task in epoch if task.startswith("a::")]
    assert sorted(a_tasks[:6]) == [f"a::a{index}" for index in range(6)]


def test_through_the_samplers_training_draws_each_epoch_of_the_weighted_order_in_turn(monkeypatch, tmp_path):
    """Handed the samplers, as platoon train hands them, the trainer draws no batch of its dataloader: the rows the
    groups run are, position by position, those of the weighted order's epochs."""

    training = _training(batch_size=2)
    drawn = _record_batches(monkeypatch)

    tasks = _train_through_samplers(tmp_path, training, "train.max_training_steps=12")

    expected = [task for batch in _batches(training.order, epochs=3, batch_size=2) for task in batch]
    assert [tasks[position] for position in range(24)] == expected
    assert drawn == []  # the samplers drew the order's rows, not the dataloader


def test_through_the_samplers_a_resumed_run_draws_what_the_run_it_resumes_would_have(tmp_path):
    """A resume after batch 6, from a checkpoint that saved no state of the draws, recomputes where each entry's
    sampler stands from the position alone."""

    uninterrupted = _train_through_samplers(tmp_path / "first", _training(2), "train.max_training_steps=12")
    resumed = _train_through_samplers(tmp_path / "second", _training(2), "train.max_training_steps=12", resume_at=6)

    assert sorted(resumed) == list(range(12, 24))
    assert resumed == {position: uninterrupted[position] for position in range(12, 24)}


def test_a_resumed_run_draws_the_batches_the_run_it_resumes_would_have(monkeypatch, tmp_path):
    training = _training(batch_size=2)
    drawn = _record_batches(monkeypatch)
    _train(tmp_path / "first", training, "train.max_training_steps=12")
    uninterrupted = [batch for _, batch in drawn]
    drawn.clear()

    # The second run resumes after batch 6, in the middle of the second epoch, from an order built afresh.
    resumed_dir = tmp_path / "second"
    resumed_dir.mkdir()
    config = trainer_config(resumed_dir, "train.max_training_steps=12")
    again = _training(batch_size=2)
    second, _saves, _logged = make_trainer(config, train_dataset=again.dataset, train_order=again.order)
    run_dir = Path(second.run_log_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"name": "000006", "batch": 6, "state_path": "tinker://m/weights/6", "sampler_path": "tinker://m/s/6"}
    (run_dir / "checkpoints.jsonl").write_text(json.dumps(record) + "\n")
    asyncio.run(asyncio.wait_for(second.train(train_workflow=Workflow(delay=0.01)), 60))

    assert [number for number, _ in drawn] == list(range(6, 12))
    assert [batch for _, batch in drawn] == uninterrupted[6:]


class _FailingOrder:
    """An order of six rows that cannot give its second epoch: it raises, as an order of one's own may."""

    epoch_size = 6

    def epoch(self, index: int) -> list[dict]:
        if index >= 1:
            raise ValueError(f"the order has no epoch {index}")
        return [{"task_id": f"t{position}"} for position in range(6)]


def _errors(error: BaseException) -> list[BaseException]:
    return [error, *(found for nested in getattr(error, "exceptions", ()) for found in _errors(nested))]


def test_an_order_handed_to_the_trainer_that_fails_stops_the_run_with_its_own_error(tmp_path):
    """A trainer built by hand with an order of its own and no task samplers draws each position of the order as it
    is: an epoch the order cannot give stops the run with the order's error, as platoon 0.3.0's did, and no row of
    another position trains in its place."""

    from datasets import Dataset

    config = trainer_config(tmp_path, "train.batch_size=3", "train.max_training_steps=4", "train.max_staleness=0")
    trainer, _saves, _logged = make_trainer(
        config, train_dataset=Dataset.from_list(_FailingOrder().epoch(0)), train_order=_FailingOrder()
    )
    workflow = _Positions()
    with pytest.raises(Exception) as raised:
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
    assert any(isinstance(error, ValueError) and "no epoch 1" in str(error) for error in _errors(raised.value))
    assert workflow.tasks == {position: f"t{position}" for position in range(6)}


class _ReachedTrainer(Exception):
    pass


async def rollout(task, config) -> dict:
    return {}


def test_platoon_train_hands_the_trainer_the_mixtures_order(monkeypatch, tmp_path):
    """``platoon train`` lists the tasks once, and hands the trainer the first epoch as its dataset and the weighted
    order its dataloader draws the epochs from."""

    import yaml

    from platoon import cli
    from platoon.train import launch as train_module
    from platoon.train.components import WeightedOrder
    from platoon.train.dataset_order import PRESERVE_DATASET_ORDER_COLUMN

    built: dict = {}

    def trainer(**kwargs):
        built.update(kwargs)
        raise _ReachedTrainer()

    monkeypatch.setattr(train_module, "PlatoonTinkerRLTrainer", trainer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    environments = [
        {
            "name": name,
            "dataset_loader": f"{HERE}:load_ids",
            "dataset_kwargs": {"prefix": name, "count": count},
            "rollout": f"{HERE}:rollout",
            "seed": 3,
            "weight": 1,
        }
        for name, count in (("a", 6), ("b", 2))
    ]
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "environments": environments,
                "train": {"model_name": "m", "renderer_name": "r", "batch_size": 2, "max_training_steps": 12},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
            }
        )
    )

    with pytest.raises(_ReachedTrainer):
        cli.main(["train", str(path)])

    order = built["train_order"]
    assert isinstance(order, WeightedOrder) and order.epoch_size == 8
    assert built["train_dataset"].to_list() == [{**row, PRESERVE_DATASET_ORDER_COLUMN: True} for row in order.epoch(0)]
    assert [row["task_id"] for row in order.epoch(1)] == [row["task_id"] for row in _training(2).order.epoch(1)]


def load_kinds(config, split, *, prefix, count):
    return [{"task_id": f"{prefix}{index}", "kind": f"{prefix}-{index}"} for index in range(count)]


def test_each_steps_metrics_break_down_by_the_labels_of_tasks_the_first_epoch_leaves_out(tmp_path):
    """The run labels its tasks by a column of their own, one label each: the labels whose metrics a step logs are
    every task's, not only those of the tasks the first epoch draws."""

    config = make_config(
        environments=[
            {
                "name": name,
                "dataset_loader": f"{HERE}.load_kinds",
                "dataset_kwargs": {"prefix": name, "count": count},
                "seed": 3,
                "weight": 1.0,
            }
            for name, count in (("a", 6), ("b", 2))
        ],
        train={"batch_size": 2, "task_sampler": "auto"},
    )
    training = AutoDataset.training(config)
    kinds = sorted(row["kind"] for row in training.tasks)
    assert training.order is not None and sorted(row["kind"] for row in training.order.items) == kinds
    assert {row["kind"] for row in training.dataset} < set(kinds)  # a gets 4 of the first epoch's 8 slots, for 6 tasks

    logged = _train(tmp_path, training, "train.label_by=kind")

    labelled = {key.split("/")[3] for stats in logged.stats for key in stats if key.startswith("train/batch/kind/")}
    assert sorted(labelled) == kinds
    assert all(f"train/batch/kind/{kind}/groups" in stats for stats in logged.stats for kind in kinds)  # also at zero
