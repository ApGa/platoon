"""The trainer's runs that the tests of several packages drive on fakes (``dispatch/``, ``tasks/``, ``workflow/``).

- Groups whose one datum's first token is the position of their row in the order (:class:`PositionGroups`; some fail
  in :class:`FailingGroups`), so the datums a step trains say which positions it consumed, and a training client that
  records them per step (:class:`RecordingClient`).
- A mixture's first epoch and weighted order (:func:`weighted_mixture`).
- One start on a fake Tinker service, with tinker-cookbook's own checkpoint saving, whose training calls record the
  positions each step trains (:func:`service_start`), run within a minute (:func:`run_start`); the settings of a run
  that a crash can cut after any of its steps with draws outstanding (:data:`RECOVERY_RUN`); and what a crash right
  after a checkpoint leaves (:func:`crash_after`), with the dispatcher's state that checkpoint saved
  (:func:`dispatch_state`); and that a run gave each position one fate at most (:func:`check_fates_once`).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset

from fakes.config import trainer_config
from fakes.tinker_service import make_service_trainer
from fakes.trainer import StepLog, TrainingClient, datum
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.components import ENVIRONMENT_COLUMN, WeightedOrder
from platoon.train.dataset_order import PRESERVE_DATASET_ORDER_COLUMN
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.task_stats import read_events


class PositionGroups:
    """A training workflow: each group takes ``delay(row)`` seconds, then, when ``hold(row)`` gives an event, waits for
    it, and returns one datum of its row's position at the policy version it started at; ``on_start(row, count)`` runs
    as each group starts (``count`` groups started so far)."""

    def __init__(self, llm, *, delay=lambda row: 0.01, hold=None, on_start=None) -> None:
        self.llm = llm
        self.delay = delay
        self.hold = hold
        self.on_start = on_start
        self.started: list[int] = []
        self.tasks: dict[int, str] = {}

    async def arun_episode(self, data):
        self.started.append(data[POSITION_COLUMN])
        self.tasks[data[POSITION_COLUMN]] = data["task_id"]
        if self.on_start is not None:
            self.on_start(data, len(self.started))
        version = self.llm.version
        await asyncio.sleep(self.delay(data))
        if self.hold is not None and (event := self.hold(data)) is not None:
            await event.wait()
        return [datum(data[POSITION_COLUMN], version=version)]


class FailingGroups(PositionGroups):
    """Groups of which those at ``positions`` fail: they raise ``error`` when the others would return."""

    def __init__(self, llm, positions, error=None, **kwargs) -> None:
        super().__init__(llm, **kwargs)
        self.positions = set(positions)
        self.error = error if error is not None else RuntimeError("the group failed")

    async def arun_episode(self, data):
        if data[POSITION_COLUMN] in self.positions:
            self.started.append(data[POSITION_COLUMN])
            await asyncio.sleep(self.delay(data))
            raise self.error
        return await super().arun_episode(data)


class RecordingClient(TrainingClient):
    """Records the positions each step trains (``steps``; one forward and backward per step here), each taking
    ``delay``: as the training client of a trainer ``fakes.trainer.make_trainer`` builds, or, through :meth:`record`, as
    what the training clients of a fake Tinker service run first (:func:`service_start`)."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self.steps: list[list[int]] = []

    async def record(self, datums) -> None:
        self.steps.append([datum.model_input.to_ints()[0] for datum in datums])
        if self.delay:
            await asyncio.sleep(self.delay)

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        await self.record(datums)
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def weighted_mixture(**counts: int) -> tuple[Dataset, WeightedOrder]:
    """The first epoch and the weighted order of a mixture of ``counts`` tasks per entry, weighted by their tasks."""

    pools = {
        name: [{"task_id": f"{name}::{name}{index}", ENVIRONMENT_COLUMN: name} for index in range(count)]
        for name, count in counts.items()
    }
    size = sum(counts.values())
    order = WeightedOrder(
        pools, {name: float(count) for name, count in counts.items()}, seeds=dict.fromkeys(counts, 3), epoch_size=size
    )
    first = Dataset.from_list([{**row, PRESERVE_DATASET_ORDER_COLUMN: True} for row in order.epoch(0)])
    return first, order


# A run of four steps of two groups that saves a recovery checkpoint after each step while groups up to two steps ahead
# run (48 rollouts, six groups, at once): a crash after any of its checkpoints leaves draws outstanding.
RECOVERY_RUN = (
    "train.batch_size=2",
    "train.max_training_steps=4",
    "train.max_staleness=2",
    "train.max_concurrent_rollouts=48",
    "checkpoint.recovery_every=1",
)


def service_start(
    tmp_path: Path, service, *overrides: str, tasks: int = 8, train_delay: float = 0.01, **trainer_options
) -> tuple[Any, RecordingClient, StepLog]:
    """``(trainer, client, logged)``: one start of a run on ``service`` (``fakes.tinker_service.make_service_trainer``,
    with the config ``trainer_config`` loads with ``overrides``); ``client`` records the positions each step trains
    (``client.steps``), each step's training taking ``train_delay``, and ``logged`` the steps the start logs.
    ``trainer_options`` are the trainer constructor's (``train_dataset``, ...)."""

    trainer, logged = make_service_trainer(
        trainer_config(tmp_path, *overrides), service, tasks=tasks, **trainer_options
    )
    client = RecordingClient(delay=train_delay)
    service.on_forward_backward = client.record
    return trainer, client, logged


def run_start(trainer, workflow) -> None:
    """Train ``trainer`` with ``workflow`` to its last step; a start that does not end within a minute fails."""

    async def run() -> None:
        task = asyncio.ensure_future(trainer.train(train_workflow=workflow))
        done, _ = await asyncio.wait({task}, timeout=60)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise AssertionError("training did not finish within 60 s")
        task.result()

    asyncio.run(run())


def crash_after(run_dir: Path, name: str) -> list[dict[str, Any]]:
    """Leave the run as a crash right after checkpoint ``name`` would: its rows up to it, the files they name."""

    rows = read_checkpoint_records(run_dir)
    last = max(index for index, row in enumerate(rows) if row["name"] == name)
    for row in rows[last + 1 :]:
        for section, key in (("tasks", "file"), ("dispatch", "file"), ("dispatch", "kept")):
            if isinstance(file := (row.get(section) or {}).get(key), str):
                (run_dir / file).unlink(missing_ok=True)
    (run_dir / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[: last + 1]))
    return rows[: last + 1]


def dispatch_state(run_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    """The dispatcher's state a checkpoint row names."""

    return json.loads((run_dir / row["dispatch"]["file"]).read_bytes())


def check_fates_once(trainer) -> dict[int, str]:
    """Every position has at most one fate in the event log, and the statistics' per-task fates are the events'; the
    fate of each position."""

    events = list(read_events(Path(trainer.run_log_path)))
    position = {(e["start"], e["seq"]): e["position"] for e in events if e["event"] == "draw"}
    task = {(e["start"], e["seq"]): e["task_id"] for e in events if e["event"] == "draw"}
    fates = [
        (position[(e["start"], e["seq"])], task[(e["start"], e["seq"])], e["fate"])
        for e in events
        if e["event"] == "fate"
    ]
    by_position = Counter(position for position, _task, _fate in fates)
    assert all(count == 1 for count in by_position.values()), by_position.most_common(3)
    store = trainer._recorder.store
    counted: Counter[tuple[str, str]] = Counter((task_id, fate) for _position, task_id, fate in fates)
    for task_id in store.task_ids():
        for fate, count in store.task(task_id).fates.items():
            assert counted[(task_id, fate)] == count, (task_id, fate)
    return {position: fate for position, _task, fate in fates}
