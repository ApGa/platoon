"""The first SIGINT or SIGTERM stops ``platoon train`` at a step boundary and saves ``<step>_interrupted``.

The signal handler only requests a shutdown, on which the rollout feeders stop.  A train loop that kept waiting
for its batch's rollouts never returned: no interrupt checkpoint was written, and the watchdog later killed the
process with the exit code the restart wrapper restarts on.  Now a batch that has applied no update yet is
abandoned (a resume from the last completed step redoes it, so no update is applied twice), and one that has
finishes first.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest
from fakes.config import make_config, trainer_config
from fakes.loop import loop_state, loop_trainer
from fakes.trainer import TrainingClient, Workflow, interrupt_at, make_trainer

from platoon.train.evaluation.loop import eval_worker
from platoon.train.trainer import signals as signals_module
from platoon.utils.stats_tracker import get as get_tracker

NO_PERMANENT_CHECKPOINTS = "checkpoint={every: 100}"


def _train(trainer, workflow) -> None:
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), timeout=20))


def _interrupt_at_optimizer_step(step: int):
    """An ``on_optim_step`` hook that sends this process SIGINT as optimizer step ``step`` is submitted."""

    hook = interrupt_at(step)
    return lambda count: hook(None, count)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_the_first_signal_stops_training_and_saves_the_interrupt_checkpoint(tmp_path, sig):
    config = trainer_config(tmp_path, "train.max_training_steps=20", NO_PERMANENT_CHECKPOINTS)
    client = TrainingClient()
    trainer, saves, logged = make_trainer(config, training_client=client)
    # Two workers, batch 2: the fifth rollout starts once step 1 is done and a rollout of step 2 has returned.
    workflow = Workflow(delay=0.2, on_start=interrupt_at(5, sig))

    _train(trainer, workflow)

    assert logged == [1]
    assert client.optim_steps == 1  # step 2 applied nothing: resuming at step 1 redoes it
    assert saves == ["recovery-000000", "000001_interrupted"]


def test_a_batch_that_applied_an_update_finishes_before_the_interrupt_checkpoint(tmp_path):
    config = trainer_config(
        tmp_path, "train.max_training_steps=20", "train.num_minibatches=2", NO_PERMANENT_CHECKPOINTS
    )
    # Two optimizer steps per batch: the third is step 2's first, after which step 2 must still finish.
    client = TrainingClient(on_optim_step=_interrupt_at_optimizer_step(3))
    trainer, saves, logged = make_trainer(config, training_client=client)

    _train(trainer, Workflow(delay=0.2))

    assert logged == [1, 2]
    assert client.optim_steps == 4
    assert saves == ["recovery-000000", "000002_interrupted"]


@pytest.mark.parametrize(
    "sigs",
    [(signal.SIGUSR2,), (signal.SIGUSR2, signal.SIGTERM), (signal.SIGTERM, signal.SIGUSR2), (signal.SIGUSR2,) * 2],
    ids=["stop", "stop-then-own", "own-then-stop", "stop-twice"],
)
def test_the_stop_a_supervisor_passes_on_is_a_first_signal_that_never_counts_as_a_second(tmp_path, monkeypatch, sigs):
    """``platoon train`` passes its first SIGINT or SIGTERM on as SIGUSR2 (``platoon.train.supervise``), which the
    start may get besides the signal itself, in either order: one stop at the step boundary either way."""

    exits: list[int] = []
    monkeypatch.setattr(os, "_exit", exits.append)
    config = trainer_config(tmp_path, "train.max_training_steps=20", NO_PERMANENT_CHECKPOINTS)
    client = TrainingClient()
    trainer, saves, logged = make_trainer(config, training_client=client)

    _train(trainer, Workflow(delay=0.2, on_start=interrupt_at(5, *sigs)))

    assert exits == []
    assert logged == [1]
    assert saves == ["recovery-000000", "000001_interrupted"]


def test_the_stop_at_once_a_supervisor_passes_on_forces_the_exit(tmp_path, monkeypatch):
    exits: list[int] = []
    stopped: list[bool] = []
    monkeypatch.setattr(os, "_exit", exits.append)  # records instead of exiting; training then goes on
    monkeypatch.setattr(signals_module, "_stop_rollout_workers", lambda: stopped.append(True))
    config = trainer_config(tmp_path, "train.max_training_steps=2", NO_PERMANENT_CHECKPOINTS)
    trainer, saves, logged = make_trainer(config)

    _train(trainer, Workflow(delay=0.05, on_start=interrupt_at(3, signal.SIGUSR1)))

    assert exits == [1] and stopped == [True]


@pytest.mark.parametrize("num_minibatches", [1, 2])
def test_after_a_shutdown_request_an_evaluation_s_rollout_workers_take_no_task(num_minibatches):
    # Training never waits for an evaluation after the request, whatever its updates per batch (training's groups,
    # which a batch with several updates may still need, start through the dispatcher): they return at once.
    trainer = loop_trainer(make_config(train={"num_minibatches": num_minibatches}))
    shared = loop_state(trainer)
    shared.shutdown_event.set()

    async def serve() -> tuple[int, int]:
        tasks, results = asyncio.Queue(), asyncio.Queue()
        tasks.put_nowait({"task_id": "a"})
        worker = asyncio.ensure_future(eval_worker(shared, Workflow(0), tasks, results, tracker=get_tracker("eval")))
        try:
            await asyncio.wait({worker}, timeout=1)  # one that stops returns at once; one that serves blocks
            return tasks.qsize(), results.qsize()
        finally:
            worker.cancel()

    assert asyncio.run(serve()) == (1, 0)
