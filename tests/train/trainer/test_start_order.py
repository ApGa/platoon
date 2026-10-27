"""What a start does that a user sees, in the order they see it: each save's line before the save, the run's last line
after the final save, and the rollout workers' first binding published under the start's limit on the rollouts
starting at once.  The golden trace sorts its records and keys, so it misses these orders (``tests/train/parity``)."""

from __future__ import annotations

import asyncio
import logging

from fakes.config import trainer_config
from fakes.trainer import SaveLog, Workflow, make_trainer


class _Events(logging.Handler):
    """The start's lines, each an event of ``events``."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(logging.INFO)
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(record.getMessage())


def test_each_save_s_line_comes_before_the_save_and_the_run_s_last_line_after_the_final_one(tmp_path):
    events: list[str] = []
    trainer, _saves, logged = make_trainer(trainer_config(tmp_path, "checkpoint={every: 100}"), saves=SaveLog(events))
    handler = _Events(events)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        asyncio.run(trainer.train(train_workflow=Workflow(delay=0.01)))
    finally:
        root.removeHandler(handler)
        root.setLevel(level)

    shown = [
        event
        for event in events
        if event.startswith(("save ", "Saving recovery checkpoint", "Saving permanent checkpoint", "Training complete"))
    ]
    assert shown == [
        "Saving recovery checkpoint recovery-000000 at launch (batch 0)",
        "save recovery-000000",
        "Saving permanent checkpoint final at step 3...",
        "save final",
        "Training complete successfully!",
    ]
    assert logged == [1, 2, 3]


class _Bound(Workflow):
    """A workflow that takes the start's limit on the rollouts starting at once, and records the limit it holds as
    each binding of the rollout workers is published to it."""

    takes_start_limit = True

    def __init__(self) -> None:
        super().__init__(delay=0.01)
        self.start_limit = None
        self.bindings: list[object] = []

    def publish_model_binding(self, sampling_client, version) -> None:
        self.bindings.append(self.start_limit)


def test_the_rollout_workers_first_binding_is_published_under_the_start_s_limit(tmp_path):
    """The limit is set before the loops start, so the first binding, published before any group starts, and every one
    after it, already find it."""

    trainer, _saves, _logged = make_trainer(trainer_config(tmp_path, "train.max_starting_rollouts=2"))
    workflow = _Bound()

    asyncio.run(trainer.train(train_workflow=workflow))

    assert workflow.bindings and all(limit is not None for limit in workflow.bindings)
    assert len({id(limit) for limit in workflow.bindings}) == 1  # one limit for the start
