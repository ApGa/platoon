"""``every`` counts training steps, and ``every: 0``, the default, turns a training event off: ``eval.every: 0`` trains
without evaluating, and ``checkpoint.every: 0`` trains with weights-only updates between the launch, recovery and
final checkpoints.  A negative ``every`` is a configuration error.  ``trainer_fakes`` runs 4 steps an epoch.

``strategy: epoch`` counts ``every`` in passes over the training tasks instead (``test_event_strategy.py``)."""

from __future__ import annotations

import asyncio
import signal

import pytest
from trainer_fakes import Workflow, make_trainer, trainer_config

from platoon.train.tinker.config_defs import EvalConfig
from platoon.utils.config import ConfigError


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT and SIGTERM handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _train(trainer, **workflows) -> None:
    asyncio.run(asyncio.wait_for(trainer.train(**workflows), timeout=60))


def test_eval_every_0_trains_without_evaluating(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "eval={every: 0}")
    trainer, saves, logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = Workflow()

    _train(trainer, train_workflow=Workflow(), eval_workflow=evaluation)

    assert logged == [1, 2, 3]
    assert evaluation.started == 0
    assert saves == ["recovery-000000", "final"]


def test_an_eval_section_that_sets_no_every_does_not_evaluate(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "eval={}")
    trainer, saves, logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = Workflow()

    _train(trainer, train_workflow=Workflow(), eval_workflow=evaluation)

    assert evaluation.started == 0


def test_without_a_checkpoint_cadence_a_run_saves_no_permanent_checkpoint(tmp_path, monkeypatch):
    # Four steps make one epoch, at whose end a run saved a permanent checkpoint while ``every`` counted epochs.
    config = trainer_config(tmp_path, "train.max_training_steps=4")
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, train_workflow=Workflow())

    assert logged == [1, 2, 3, 4]
    assert saves == ["recovery-000000", "final"]


def test_checkpoint_every_0_keeps_the_recovery_cadence(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "checkpoint={every: 0, recovery_every: 2}")
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, train_workflow=Workflow())

    assert saves == ["recovery-000000", "recovery-000002", "final"]


def test_every_alone_counts_training_steps_not_epochs(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "checkpoint={every: 2}")
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, train_workflow=Workflow())

    assert saves == ["recovery-000000", "000002", "final"]


def test_eval_at_start_is_off_by_default_and_takes_true_or_false(tmp_path):
    """``eval.at_start`` also evaluates the starting policy, at step 0 (``test_eval_cadence.py`` and
    ``test_eval_resume.py``); by default the first evaluation is at step ``every``."""

    assert trainer_config(tmp_path, "eval.every=5").eval.at_start is False
    assert trainer_config(tmp_path, "eval.every=5", "eval.at_start=true").eval.at_start is True
    with pytest.raises(ConfigError, match="eval.at_start"):
        trainer_config(tmp_path, "eval.at_start=sometimes")
    with pytest.raises(ValueError, match="eval.at_start must be true or false, got 'yes'"):
        EvalConfig(every=5, at_start="yes")  # a config built in Python, which no loader checked


@pytest.mark.parametrize("section", ["checkpoint", "eval"])
def test_a_negative_every_is_a_config_error(tmp_path, section):
    with pytest.raises(ConfigError, match=rf"{section}\.every must be a number of training steps, or 0 for never"):
        trainer_config(tmp_path, f"{section}.every=-1")
