"""``platoon train`` readies its rollouts before the first one (:mod:`platoon.train.startup`): it starts its workflows'
rollout workers and the staging of its harness toolchains as soon as its training client is open, so that both run
while it saves at launch, and starts no rollout, training's or an evaluation's, before the staging is done, or
``STAGING_WAIT_S`` passed, or a shutdown was asked for."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time

from trainer_fakes import Workflow, make_trainer, trainer_config

from platoon.train import startup
from platoon.train.trainer import rl as rl_module


class _WarmWorkflow(Workflow):
    def __init__(self, events: list[str], name: str, **kwargs) -> None:
        super().__init__(on_start=lambda count: events.append(f"{name} rollout {count}"), **kwargs)
        self.events, self.name = events, name

    async def warm_up(self) -> None:
        self.events.append(f"{self.name} warm_up")


def _staging(monkeypatch, events: list[str], seconds: float = 0.3) -> None:
    def prestage(names, log):
        events.append(f"staging {list(names)}")
        time.sleep(seconds)
        events.append("staged")
        return {}

    monkeypatch.setattr("platoon.train.auto.harness_toolchains", lambda environments: ["tc-1.0"])
    monkeypatch.setattr("platoon.harness.toolchains.prestage", prestage)


def _record_saves(monkeypatch, events: list[str]) -> None:
    save = rl_module.checkpoint_utils.save_checkpoint_async

    async def recording(**kwargs):
        events.append(f"save {kwargs['name']}")
        return await save(**kwargs)

    monkeypatch.setattr(rl_module.checkpoint_utils, "save_checkpoint_async", recording)


def test_no_rollout_starts_before_the_toolchains_are_staged_and_both_begin_before_the_launch_save(
    tmp_path, monkeypatch
):
    events: list[str] = []
    trainer, _, logged = make_trainer(monkeypatch, trainer_config(tmp_path))
    _staging(monkeypatch, events)
    _record_saves(monkeypatch, events)

    asyncio.run(trainer.train(train_workflow=_WarmWorkflow(events, "train")))

    first_rollout = next(index for index, event in enumerate(events) if " rollout " in event)
    assert events.index("train warm_up") < events.index("save recovery-000000")  # the fork server starts first
    assert events.index("staging ['tc-1.0']") < events.index("staged") < first_rollout
    assert events.index("save recovery-000000") < events.index("staged")  # the launch save ran meanwhile
    assert logged == [1, 2, 3]


def test_an_evaluation_s_workers_are_started_ahead_and_its_rollouts_wait_for_the_staging_too(tmp_path, monkeypatch):
    events: list[str] = []
    config = trainer_config(tmp_path, "eval.every=1", "eval.at_start=true", "train.max_training_steps=1")
    trainer, _, _ = make_trainer(monkeypatch, config, eval_tasks=2)
    _staging(monkeypatch, events)

    asyncio.run(
        trainer.train(train_workflow=_WarmWorkflow(events, "train"), eval_workflow=_WarmWorkflow(events, "eval"))
    )

    assert {"train warm_up", "eval warm_up"} <= set(events)
    staged = events.index("staged")
    assert all(events.index(event) > staged for event in events if " rollout " in event)
    assert any(event.startswith("eval rollout") for event in events)


def test_without_an_evaluation_its_workers_are_not_started(tmp_path, monkeypatch):
    events: list[str] = []
    trainer, _, _ = make_trainer(monkeypatch, trainer_config(tmp_path))
    _staging(monkeypatch, events, seconds=0.0)

    asyncio.run(
        trainer.train(train_workflow=_WarmWorkflow(events, "train"), eval_workflow=_WarmWorkflow(events, "eval"))
    )

    assert "train warm_up" in events and "eval warm_up" not in events


def _hung_staging(monkeypatch, events: list[str]) -> threading.Event:
    release = threading.Event()

    def prestage(names, log):
        events.append("staging")
        release.wait(60)
        events.append("staged")
        return {}

    monkeypatch.setattr("platoon.train.auto.harness_toolchains", lambda environments: ["tc-1.0"])
    monkeypatch.setattr("platoon.harness.toolchains.prestage", prestage)
    return release


def test_a_staging_that_hangs_holds_the_first_rollouts_for_the_staging_wait_at_most(tmp_path, monkeypatch):
    events: list[str] = []
    trainer, _, logged = make_trainer(monkeypatch, trainer_config(tmp_path, "train.max_training_steps=2"))
    release = _hung_staging(monkeypatch, events)
    monkeypatch.setattr(startup, "STAGING_WAIT_S", 0.5)

    try:
        began = time.monotonic()
        asyncio.run(trainer.train(train_workflow=_WarmWorkflow(events, "train")))
        took, seen = time.monotonic() - began, list(events)
    finally:
        release.set()

    assert any(" rollout " in event for event in seen) and "staged" not in seen  # the rollouts did not wait on
    assert logged == [1, 2] and took < 30


def test_a_shutdown_asked_for_during_the_staging_ends_the_wait(tmp_path, monkeypatch):
    events: list[str] = []
    trainer, _, logged = make_trainer(monkeypatch, trainer_config(tmp_path, "train.max_training_steps=3"))
    release = _hung_staging(monkeypatch, events)

    def interrupt() -> None:
        while "staging" not in events:
            time.sleep(0.01)
        time.sleep(0.2)
        events.append("sigterm")
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=interrupt, daemon=True).start()
    try:
        began = time.monotonic()
        asyncio.run(trainer.train(train_workflow=_WarmWorkflow(events, "train")))
        took, seen = time.monotonic() - began, list(events)
    finally:
        release.set()

    assert "sigterm" in seen and "staged" not in seen and took < 30
    assert not any(" rollout " in event for event in seen[seen.index("sigterm") :]) and logged == []
