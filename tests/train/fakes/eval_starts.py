"""The starts of one run with evaluation during training (``tests/train/evaluation``): each start is ``train()`` on the
run directory on a fake Tinker service (``fakes.tinker_service``), which resumes from the newest checkpoint that
``checkpoints.jsonl`` lists, as ``platoon train`` runs it again after a failure; and a crash where the test says, as
an error no handler of the trainer catches (:class:`Crash`), so the start ends without an interrupt checkpoint.

- :func:`evaluated_start`: one start, ``(trainer, steps it evaluated, its crash or None)``;
- :class:`Evaluation`: an evaluation workflow that crashes the start at a rollout, or never ends;
- :func:`optimizer_step_crash`: a crash at a training client's optimizer step;
- :func:`resume_point`, :func:`run_directory`: the step a start resumes at, and a config's run directory.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fakes.tinker_service import make_service_trainer
from fakes.trainer import Workflow
from platoon.train.components import task_ids_to_dataset
from platoon.utils.jsonl import read_jsonl


class Crash(BaseException):
    """The process dying: no handler of the trainer catches it, so the start saves nothing more."""


class Evaluation:
    """An evaluation workflow whose rollouts take a moment; with ``crash_at``, the rollout of that number (counted from
    1 over the start) crashes the start, and with ``stuck``, every rollout waits for good."""

    def __init__(self, *, crash_at: int | None = None, stuck: bool = False) -> None:
        self.crash_at, self.stuck = crash_at, stuck
        self.started = 0

    async def arun_episode(self, data):
        self.started += 1
        if self.started == self.crash_at:
            raise Crash(f"crashed in evaluation rollout {self.started}")
        if self.stuck:
            await asyncio.Event().wait()
        await asyncio.sleep(0.01)
        return []


def run_directory(config) -> Path:
    """The run directory of ``config``: where each of its starts writes."""

    return Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)


def optimizer_step_crash(update: int, once: Callable[[], bool] = lambda: True):
    """A fake service's ``on_optim_step``: a training client crashes at its ``update``-th optimizer step, once
    ``once()`` holds."""

    async def crash(client) -> None:
        if client.optim_steps + 1 == update:
            while not once():
                await asyncio.sleep(0.01)
            raise Crash(f"crashed at optimizer step {update}")

    return crash


def evaluated_start(config, service, *, evaluation=None, crash_at_update=None, once=lambda: True):
    """One start of the run: ``(trainer, steps it evaluated, its crash or None)``, the steps as each evaluation it
    logged gives its ``eval/step``."""

    service.on_optim_step = optimizer_step_crash(crash_at_update, once) if crash_at_update is not None else None
    trainer, logged = make_service_trainer(config, service, eval_dataset=task_ids_to_dataset(["val:0", "val:1"]))

    async def run() -> BaseException | None:
        task = asyncio.ensure_future(
            trainer.train(train_workflow=Workflow(delay=0.01), eval_workflow=evaluation or Evaluation())
        )
        done, _ = await asyncio.wait({task}, timeout=60)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise AssertionError("the start did not end within 60 s")
        return task.exception()

    error = asyncio.run(run())
    if error is not None and (not isinstance(error, BaseExceptionGroup) or error.subgroup(Crash) is None):
        raise error  # a failure of the test, not the crash it set up
    return trainer, [stats["eval/step"] for stats in logged.stats if "eval/step" in stats], error


def resume_point(trainer) -> int:
    """The step a start of the run resumes at: the ``batch`` of the newest checkpoint with weights and optimizer."""

    rows = read_jsonl(Path(trainer.run_log_path) / "checkpoints.jsonl")
    return [row for row in rows if row.get("state_path")][-1]["batch"]
