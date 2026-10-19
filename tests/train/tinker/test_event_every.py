"""``every`` counts training steps, and ``every: 0``, the default, turns a training event off: ``eval.every: 0`` trains
without evaluating, and ``checkpoint.every: 0`` trains with weights-only updates between the launch, recovery and
final checkpoints.  A negative ``every`` is a configuration error.

``strategy``, which gave ``every`` its unit (epochs by default), is deprecated: a config that still sets it keeps its
old meaning until platoon 0.3.0 removes it, with a deprecation warning, an ``every`` it leaves out included, which was
1.  ``trainer_fakes`` runs 4 steps an epoch."""

from __future__ import annotations

import asyncio
import signal

import pytest
from trainer_fakes import Workflow, make_trainer, trainer_config

from platoon.deprecations import PlatoonDeprecationWarning
from platoon.train.tinker.config_defs import EvalConfig, PlatoonTinkerRLTrainerConfig, evaluates, event_interval
from platoon.utils.config import ConfigError, resolve_config


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


def test_a_deprecated_epoch_strategy_still_counts_epochs(tmp_path, monkeypatch):
    with pytest.warns(PlatoonDeprecationWarning, match=r"checkpoint\.strategy: deprecated; use every alone"):
        config = trainer_config(tmp_path, "train.max_training_steps=8", "checkpoint={strategy: epoch, every: 1}")
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, train_workflow=Workflow())

    assert saves == ["recovery-000000", "000004", "000008", "final"]


def test_a_deprecated_strategy_none_still_turns_the_event_off(tmp_path, monkeypatch):
    with pytest.warns(PlatoonDeprecationWarning) as warned:
        config = trainer_config(tmp_path, "eval={strategy: none, every: 1}", "checkpoint={strategy: none, every: 1}")
    deprecated = sorted(
        str(item.message).partition(":")[0] for item in warned if item.category is PlatoonDeprecationWarning
    )
    assert deprecated == ["checkpoint.strategy", "eval.strategy"]
    trainer, saves, logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = Workflow()

    _train(trainer, train_workflow=Workflow(), eval_workflow=evaluation)

    assert evaluation.started == 0
    assert saves == ["recovery-000000", "final"]


def _without_every(tmp_path, evaluation: str, checkpoint: str) -> PlatoonTinkerRLTrainerConfig:
    """A config whose eval and checkpoint sections are these (YAML flow mappings), which may leave ``every`` out."""

    path = tmp_path / "train.yaml"
    path.write_text(
        "train: {model_name: m, renderer_name: r, batch_size: 2, max_training_steps: 8, max_concurrent_rollouts: 16}\n"
        f"eval: {evaluation}\n"
        f"checkpoint: {checkpoint}\n"
        "watchdog: {enabled: false}\n"
        f"log_path: {tmp_path / 'logs'}\n"
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig, warn=False).config


@pytest.mark.parametrize("strategy", ["step", "epoch"])
def test_a_deprecated_strategy_keeps_the_every_of_1_it_had_when_every_is_left_out(tmp_path, strategy):
    # Before every counted steps it defaulted to 1: eval: {strategy: step} evaluated every step, and checkpoint:
    # {strategy: epoch} saved a permanent checkpoint every epoch.
    config = _without_every(tmp_path, f"{{strategy: {strategy}}}", f"{{strategy: {strategy}}}")

    assert (config.eval.every, config.checkpoint.every) == (1, 1)
    assert evaluates(config)
    assert event_interval(config.checkpoint, 4) == (4 if strategy == "epoch" else 1)


def test_left_out_every_is_0_without_a_counting_strategy(tmp_path):
    plain = _without_every(tmp_path, "{}", "{}")
    assert (plain.eval.every, plain.checkpoint.every) == (0, 0)
    none = _without_every(tmp_path, "{strategy: none}", "{strategy: none}")
    assert not evaluates(none) and event_interval(none.checkpoint, 4) is None
    explicit = _without_every(tmp_path, "{strategy: step, every: 0}", "{strategy: step, every: 0}")
    assert not evaluates(explicit) and event_interval(explicit.checkpoint, 4) is None


def test_a_deprecated_epoch_strategy_without_every_saves_every_epoch(tmp_path, monkeypatch):
    config = _without_every(tmp_path, "{every: 0}", "{strategy: epoch}")
    trainer, saves, logged = make_trainer(monkeypatch, config)

    _train(trainer, train_workflow=Workflow())

    assert saves == ["recovery-000000", "000004", "000008", "final"]


def test_a_deprecated_step_strategy_without_every_evaluates_every_step(tmp_path, monkeypatch):
    config = _without_every(tmp_path, "{strategy: step}", "{every: 0}")
    trainer, saves, logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = Workflow()

    _train(trainer, train_workflow=Workflow(), eval_workflow=evaluation)

    assert evaluation.started >= 2  # the last step's evaluation at least, of two tasks; every: 0 runs none


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


def test_a_deprecated_strategy_none_ignores_every(tmp_path):
    with pytest.warns(PlatoonDeprecationWarning):
        config = trainer_config(tmp_path, "checkpoint={strategy: none, every: -1}")

    assert config.checkpoint.strategy == "none"


def test_the_deprecated_strategy_is_reported_with_its_replacement(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        "train: {model_name: m, renderer_name: r}\n"
        "eval: {strategy: step, every: 20}\n"
        "checkpoint: {strategy: step, every: 50}\n"
        f"log_path: {tmp_path}\n"
    )

    resolved = resolve_config(path, [], PlatoonTinkerRLTrainerConfig, warn=False)

    assert [(item.path, item.replacement, item.removed_in) for item in resolved.deprecations] == [
        ("eval.strategy", "every alone, which counts training steps (0: never)", "0.3.0"),
        ("checkpoint.strategy", "every alone, which counts training steps (0: never)", "0.3.0"),
    ]
    assert (resolved.config.eval.every, resolved.config.checkpoint.every) == (20, 50)
