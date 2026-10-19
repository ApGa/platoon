"""Every sandbox a training run starts carries the run's label (``platoon.run``: the run directory), in the trainer's
own process and, through the worker request, in its rollout workers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.task import Task
from platoon.train.tinker import rl

RUN = sandbox_cleanup.RUN_LABEL


@pytest.fixture(autouse=True)
def no_labels(monkeypatch):
    monkeypatch.setattr(sandbox_cleanup, "_labels", {})


class _StatsLogger:
    def __init__(self, config, exp_config=None) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _trainer(tmp_path, monkeypatch) -> rl.PlatoonTinkerRLTrainer:
    """A trainer whose ``async with`` runs without Tinker or W&B."""

    monkeypatch.setattr(rl.checkpoint_utils, "get_last_checkpoint", lambda log_path: None)
    monkeypatch.setattr(rl, "StatsLogger", _StatsLogger)
    trainer = object.__new__(rl.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        log_path="logs",  # relative: the label is absolute
        stats=SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            to_stats_logger_config=lambda log_path: SimpleNamespace(wandb=SimpleNamespace(resume_run_id=None)),
        ),
    )
    trainer._stats_logger = None
    trainer._train_tracker = None
    trainer._sandbox_labels = None
    return trainer


def test_the_trainer_labels_its_runs_sandboxes_with_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sandbox_cleanup.set_labels({"keep": "1"})
    trainer = _trainer(tmp_path, monkeypatch)
    inside: list[dict[str, str]] = []

    async def run() -> None:
        async with trainer:
            inside.append(sandbox_cleanup.labels())

    asyncio.run(run())

    assert inside == [{"keep": "1", RUN: str(tmp_path / "logs" / "exp" / "trial")}]
    assert sandbox_cleanup.labels() == {"keep": "1"}  # put back when the run ends
    assert trainer._stats_logger.closed


def test_the_trainer_puts_back_the_labels_when_training_fails(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, monkeypatch)

    async def run() -> None:
        async with trainer:
            assert RUN in sandbox_cleanup.labels()
            raise RuntimeError("training failed")

    with pytest.raises(RuntimeError, match="training failed"):
        asyncio.run(run())
    assert sandbox_cleanup.labels() == {}


def test_a_rollout_workers_request_carries_the_labels_of_the_trainer(tmp_path):
    from platoon.train.tinker.config_defs import WorkflowConfig
    from platoon.train.tinker.workflows import GroupRolloutWorkflow

    async def rollout(task, rollout_config):
        raise AssertionError("not run")

    workflow = GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=WorkflowConfig(group_size=1),
        model_info=SimpleNamespace(model_name="m", base_url=None, api_key=None, llm=SimpleNamespace(version=0)),
    )

    assert workflow._build_worker_request({"task_id": "t"}, 0).sandbox_labels == {}
    with sandbox_cleanup.labelled(sandbox_cleanup.run_label(tmp_path / "run")):
        request = workflow._build_worker_request({"task_id": "t"}, 0)

    assert request.sandbox_labels == {RUN: str(tmp_path / "run")}
