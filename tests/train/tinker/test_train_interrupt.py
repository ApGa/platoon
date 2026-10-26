"""The first SIGINT or SIGTERM stops ``platoon train`` at a step boundary and saves ``<step>_interrupted``.

The signal handler only requests a shutdown, on which the rollout feeders stop.  A train loop that kept waiting
for its batch's rollouts never returned: no interrupt checkpoint was written, and the watchdog later killed the
process with the exit code the restart wrapper restarts on.  Now a batch that has applied no update yet is
abandoned (a resume from the last completed step redoes it, so no update is applied twice), and one that has
finishes first.
"""

from __future__ import annotations

import asyncio
import signal
from types import SimpleNamespace
from typing import Any

import pytest
from trainer_fakes import TrainingClient, Workflow, make_trainer, trainer_config

from platoon.train.trainer import rl as rl_module
from platoon.train.trainer import signals as signals_module
from platoon.utils.stats_tracker import get as get_tracker

NO_PERMANENT_CHECKPOINTS = "checkpoint={every: 100}"


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT, SIGTERM, SIGUSR1 and SIGUSR2 handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _interrupt_at(count: int, sig: signal.Signals = signal.SIGINT):
    """A hook that delivers ``sig`` to this process on its ``count``-th call."""

    def hook(calls: int) -> None:
        if calls == count:
            signal.raise_signal(sig)

    return hook


def _train(trainer, workflow) -> None:
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), timeout=20))


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_the_first_signal_stops_training_and_saves_the_interrupt_checkpoint(tmp_path, monkeypatch, sig):
    config = trainer_config(tmp_path, "train.max_training_steps=20", NO_PERMANENT_CHECKPOINTS)
    client = TrainingClient()
    trainer, saves, logged = make_trainer(monkeypatch, config, training_client=client)
    # Two workers, batch 2: the fifth rollout starts once step 1 is done and a rollout of step 2 has returned.
    workflow = Workflow(delay=0.2, on_start=_interrupt_at(5, sig))

    _train(trainer, workflow)

    assert logged == [1]
    assert client.optim_steps == 1  # step 2 applied nothing: resuming at step 1 redoes it
    assert saves == ["recovery-000000", "000001_interrupted"]


def test_a_batch_that_applied_an_update_finishes_before_the_interrupt_checkpoint(tmp_path, monkeypatch):
    config = trainer_config(
        tmp_path, "train.max_training_steps=20", "train.num_minibatches=2", NO_PERMANENT_CHECKPOINTS
    )
    # Two optimizer steps per batch: the third is step 2's first, after which step 2 must still finish.
    client = TrainingClient(on_optim_step=_interrupt_at(3))
    trainer, saves, logged = make_trainer(monkeypatch, config, training_client=client)

    _train(trainer, Workflow(delay=0.2))

    assert logged == [1, 2]
    assert client.optim_steps == 4
    assert saves == ["recovery-000000", "000002_interrupted"]


def _signals_at(count: int, *sigs: signal.Signals):
    """A hook that delivers ``sigs`` to this process, in order, on its ``count``-th call."""

    def hook(calls: int) -> None:
        if calls == count:
            for sig in sigs:
                signal.raise_signal(sig)

    return hook


@pytest.mark.parametrize(
    "sigs",
    [(signal.SIGUSR2,), (signal.SIGUSR2, signal.SIGTERM), (signal.SIGTERM, signal.SIGUSR2), (signal.SIGUSR2,) * 2],
    ids=["stop", "stop-then-own", "own-then-stop", "stop-twice"],
)
def test_the_stop_a_supervisor_passes_on_is_a_first_signal_that_never_counts_as_a_second(tmp_path, monkeypatch, sigs):
    """``platoon train`` passes its first SIGINT or SIGTERM on as SIGUSR2 (``platoon.train.supervise``), which the
    start may get besides the signal itself, in either order: one stop at the step boundary either way."""

    exits: list[int] = []
    monkeypatch.setattr(rl_module.os, "_exit", exits.append)
    config = trainer_config(tmp_path, "train.max_training_steps=20", NO_PERMANENT_CHECKPOINTS)
    client = TrainingClient()
    trainer, saves, logged = make_trainer(monkeypatch, config, training_client=client)

    _train(trainer, Workflow(delay=0.2, on_start=_signals_at(5, *sigs)))

    assert exits == []
    assert logged == [1]
    assert saves == ["recovery-000000", "000001_interrupted"]


def test_the_stop_at_once_a_supervisor_passes_on_forces_the_exit(tmp_path, monkeypatch):
    exits: list[int] = []
    stopped: list[bool] = []
    monkeypatch.setattr(rl_module.os, "_exit", exits.append)  # records instead of exiting; training then goes on
    monkeypatch.setattr(signals_module, "_stop_rollout_workers", lambda: stopped.append(True))
    config = trainer_config(tmp_path, "train.max_training_steps=2", NO_PERMANENT_CHECKPOINTS)
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, Workflow(delay=0.05, on_start=_signals_at(3, signal.SIGUSR1)))

    assert exits == [1] and stopped == [True]


@pytest.mark.parametrize("num_minibatches", [1, 2])
def test_after_a_shutdown_request_an_evaluation_s_rollout_workers_take_no_task(num_minibatches):
    # Training never waits for an evaluation after the request, whatever its updates per batch (training's groups,
    # which a batch with several updates may still need, start through the dispatcher): they return at once.
    trainer: Any = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(train=SimpleNamespace(num_minibatches=num_minibatches))
    shared: Any = SimpleNamespace(shutdown_event=asyncio.Event(), train_tracker=get_tracker("train"), watchdog=None)
    shared.shutdown_event.set()

    async def serve() -> tuple[int, int]:
        tasks, results = asyncio.Queue(), asyncio.Queue()
        tasks.put_nowait({"task_id": "a"})
        worker = asyncio.ensure_future(
            trainer._rollout_workflow_worker_loop(shared, Workflow(0), tasks, results, tracker=get_tracker("eval"))
        )
        try:
            await asyncio.wait({worker}, timeout=1)  # one that stops returns at once; one that serves blocks
            return tasks.qsize(), results.qsize()
        finally:
            worker.cancel()

    assert asyncio.run(serve()) == (1, 0)
