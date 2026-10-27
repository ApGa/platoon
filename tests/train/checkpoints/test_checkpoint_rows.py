"""A full checkpoint's row of the run's ``checkpoints.jsonl``, as a real run writes it (tinker-cookbook's own saving, on
a fake Tinker service): its name, its step and the paths the save wrote, then the run's other counters in their order
(``COUNTER_KEYS``), then the entries of the start's draws (the per-task statistics' snapshot, then the dispatcher's
state).  A row's key order is what a reader of the file sees; the golden trace sorts it away."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fakes.config import trainer_config
from fakes.tinker_service import FakeTinkerService, make_service_trainer
from fakes.trainer import Workflow

from platoon.train.checkpoints.participants import COUNTER_KEYS


def _rows(tmp_path: Path, *overrides: str) -> list[list[str]]:
    """The keys of each row a run of two steps, a recovery checkpoint after each, writes, in the file's order."""

    config = trainer_config(tmp_path, "train.max_training_steps=2", "checkpoint.recovery_every=1", *overrides)
    trainer, _logged = make_service_trainer(config, FakeTinkerService())
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Workflow(delay=0.01)), 60))
    lines = (Path(trainer.run_log_path) / "checkpoints.jsonl").read_text().splitlines()
    return [list(json.loads(line)) for line in lines]


def test_each_row_holds_its_name_the_run_s_counters_the_draws_entries_and_its_paths_in_that_order(tmp_path):
    rows = _rows(tmp_path)

    # The launch save, a recovery checkpoint after each step, and the final one: tinker-cookbook's record puts the
    # fields it knows first (the name, the step, the paths), then the rest of loop_state in its order.
    counters = [key for key in COUNTER_KEYS if key != "batch"]
    assert rows == [["name", "batch", "state_path", "sampler_path", *counters, "tasks", "dispatch"]] * 4
