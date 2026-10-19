"""``strategy`` gives ``every`` its unit in ``eval`` and ``checkpoint``: ``step``, a training step, which is what a
config that leaves it out counts, or ``epoch``, a pass over the training tasks.  A config that sets ``strategy`` and
leaves ``every`` out triggers the event every 1 of its units; one that leaves both out never does.  A start that
resumes evaluates at the steps a run that never stopped would, the multiples of the interval, and does not run again
an evaluation that an earlier start completed; it counts the interval of permanent checkpoints from the step it
resumes at, as it counts ``recovery_every`` (``test_checkpoint_cadence.py``).  ``strategy: none``, which platoon 0.3.0
removed, is refused, with ``every: 0`` in its place.

Each run here has 8 training tasks in batches of 2, so an epoch is 4 steps, and a recovery checkpoint after every step,
so that a crash resumes from the step before it (``test_eval_resume.py``).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import warnings
from pathlib import Path

import pytest
import yaml
from test_eval_resume import _resume_point, _run_dir, _start
from tinker_service_fakes import FakeTinkerService
from trainer_fakes import Workflow, make_trainer, trainer_config

from platoon import cli
from platoon.train.tinker import rl
from platoon.train.tinker.config_defs import (
    CheckpointConfig,
    EvalConfig,
    PlatoonTinkerRLTrainerConfig,
    default_every,
    evaluates,
    event_interval,
)
from platoon.train.tinker.evaluation_record import completed_evaluation_steps
from platoon.utils.config import ConfigError, resolve_config
from platoon.utils.jsonl import read_jsonl


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT, SIGTERM, SIGUSR1 and SIGUSR2 handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _config(tmp_path: Path, steps: int, *overrides: str) -> PlatoonTinkerRLTrainerConfig:
    """``steps`` steps of two tasks, a recovery checkpoint after each, an evaluation of two held-out tasks at a time,
    and the cadences ``overrides`` give (``trainer_fakes`` sets ``eval.every: 0``, and no checkpoint section)."""

    return trainer_config(
        tmp_path,
        f"train.max_training_steps={steps}",
        "checkpoint.recovery_every=1",
        "eval.max_concurrent_rollouts=2",
        *overrides,
    )


def _permanent(trainer) -> list[int]:
    """The steps of the permanent checkpoints that the run's ``checkpoints.jsonl`` lists, over all its starts: those
    named by their step, not ``recovery-<step>`` or ``final``."""

    rows = read_jsonl(Path(trainer.run_log_path) / "checkpoints.jsonl")
    return [row["batch"] for row in rows if str(row["name"]).isdigit()]


# --- the units -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("strategy", "every", "steps"),
    [(None, 3, 3), ("step", 3, 3), ("epoch", 2, 8), (None, 0, None), ("step", 0, None), ("epoch", 0, None)],
)
def test_every_counts_in_the_units_of_strategy(strategy, every, steps):
    for config in (EvalConfig(strategy=strategy, every=every), CheckpointConfig(strategy=strategy, every=every)):
        assert event_interval(config, 4) == steps  # 4 steps an epoch


def test_left_out_every_is_1_under_a_strategy_and_0_without_one(tmp_path):
    """The rule of a config that leaves ``every`` out (``default_every``): ``eval: {strategy: step}`` evaluates every
    step and ``checkpoint: {strategy: epoch}`` saves a permanent checkpoint every epoch, as when ``every`` counted in
    ``strategy``'s units by default; ``eval: {}`` never evaluates."""

    assert (default_every(None), default_every("step"), default_every("epoch")) == (0, 1, 1)
    config = _config(tmp_path, 8, "eval.every=null", "eval.strategy=step", "checkpoint.strategy=epoch")
    assert (config.eval.every, config.checkpoint.every) == (1, 1)
    assert evaluates(config) and event_interval(config.eval, 4) == 1 and event_interval(config.checkpoint, 4) == 4
    unset = _config(tmp_path, 8, "eval.every=null")
    assert (unset.eval.strategy, unset.eval.every, unset.checkpoint.every) == (None, 0, 0) and not evaluates(unset)


def test_strategy_is_a_key_of_its_own_with_no_deprecation(tmp_path, capsys):
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r"},
                "eval": {"strategy": "epoch", "every": 2},
                "checkpoint": {"strategy": "step", "every": 50},
                "log_path": str(tmp_path / "runs"),
            }
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = resolve_config(path, [], PlatoonTinkerRLTrainerConfig)

    assert resolved.deprecations == ()
    assert (resolved.config.eval.strategy, resolved.config.eval.every) == ("epoch", 2)
    assert (resolved.config.checkpoint.strategy, resolved.config.checkpoint.every) == ("step", 50)
    cli.main(["config", "validate", "--strict", "--no-resolve", str(path)])
    assert capsys.readouterr().out == f"ok   {path} (PlatoonTinkerRLTrainerConfig)\n"


def test_strategy_none_is_refused_naming_every_0(tmp_path):
    """``strategy: none`` turned an event off whatever ``every`` said; ``every: 0`` says the same, and platoon 0.3.0
    removed it: a config file, an override and a config built in Python that set it are refused, naming ``every: 0``."""

    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r"},
                "eval": {"strategy": "none", "every": 5},
                "checkpoint": {"strategy": "none"},
                "log_path": str(tmp_path / "runs"),
            },
            sort_keys=False,
        )
    )
    evaluation = (
        "eval.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use eval.every: 0, which never "
        "evaluates"
    )
    checkpoint = (
        "checkpoint.strategy: 'none' is not one of 'step', 'epoch'; platoon 0.3.0 removed it; use checkpoint.every: 0, "
        "which saves no permanent checkpoint"
    )

    with pytest.raises(ConfigError) as refused:
        resolve_config(path, [], PlatoonTinkerRLTrainerConfig, warn=False)
    assert refused.value.problems == [evaluation, checkpoint]

    with pytest.raises(ConfigError) as refused:
        trainer_config(tmp_path, "eval={every: 3, strategy: none}")
    assert refused.value.problems == [f"override eval={{every: 3, strategy: none}}: {evaluation}"]
    with pytest.raises(ConfigError) as refused:  # as an override writes it, none is the removed value, not a null
        trainer_config(tmp_path, "eval.every=3", "eval.strategy=none")
    assert refused.value.problems == [f"override eval.strategy=none: {evaluation}"]
    unset = trainer_config(tmp_path, "eval.every=3", "eval.strategy=null")  # null leaves it unset: every counts steps
    assert (unset.eval.strategy, unset.eval.every) == (None, 3)

    with pytest.raises(ValueError) as built:
        EvalConfig(strategy="none", every=5)  # type: ignore[arg-type]
    assert str(built.value) == evaluation
    with pytest.raises(ValueError) as built:
        CheckpointConfig(strategy="none")  # type: ignore[arg-type]
    assert str(built.value) == checkpoint


@pytest.mark.parametrize("section", ["checkpoint", "eval"])
def test_a_negative_every_under_an_epoch_strategy_names_its_unit(tmp_path, section):
    with pytest.raises(ConfigError, match=rf"{section}\.every must be a number of passes over the training tasks"):
        trainer_config(tmp_path, f"{section}.strategy=epoch", f"{section}.every=-1")


# --- the schedules, over the starts of a run ------------------------------------------------------------------------


def test_without_a_strategy_every_counts_training_steps(tmp_path, monkeypatch):
    config = _config(tmp_path, 4, "eval.every=2", "checkpoint.every=2")

    trainer, evaluated, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and evaluated == [2, 4] and _permanent(trainer) == [2, 4]


def test_a_config_that_leaves_every_out_and_sets_no_strategy_neither_evaluates_nor_saves_a_permanent_checkpoint(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, 4, "eval.every=null")

    trainer, evaluated, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and evaluated == [] and _permanent(trainer) == []


def test_a_step_strategy_without_every_evaluates_every_step(tmp_path, monkeypatch):
    config = trainer_config(tmp_path, "eval.every=null", "eval.strategy=step")  # three steps
    trainer, saves, logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = Workflow()

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Workflow(), eval_workflow=evaluation), timeout=60))

    # The last step's evaluation at least, of two tasks (one that a later step overtakes does not run); every: 0
    # would run none.
    assert evaluation.started >= 2


def test_an_epoch_strategy_evaluates_and_saves_at_the_end_of_each_pass(tmp_path, monkeypatch):
    config = _config(tmp_path, 8, "eval.every=null", "eval.strategy=epoch", "checkpoint.strategy=epoch")

    trainer, evaluated, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and evaluated == [4, 8] and _permanent(trainer) == [4, 8]


def test_a_run_shorter_than_its_evaluation_epoch_evaluates_nothing_and_says_so_in_epochs(tmp_path, monkeypatch, caplog):
    """A run whose last step comes before the end of its first epoch has no evaluation due: it warns, naming
    ``every`` in epochs, as the config gives it, and in steps."""

    config = _config(tmp_path, 3, "eval.every=1", "eval.strategy=epoch")

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        _trainer, evaluated, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and evaluated == []
    assert (
        "No evaluation runs: eval.every is 1 epoch(s) (strategy: epoch), 4 steps, past the run's last step, 3; "
        "eval.at_start: true would evaluate the starting policy at step 0"
    ) in caplog.text


def test_a_start_that_resumes_at_an_epochs_end_goes_on_with_the_next_epochs(tmp_path, monkeypatch):
    """The crash comes during step 5, once the first epoch's evaluation completed: the restart resumes at step 4, from
    its permanent checkpoint, and evaluates and saves a permanent checkpoint at step 8, the end of the second epoch,
    as a run that never stopped would."""

    config = _config(tmp_path, 8, "eval.every=1", "eval.strategy=epoch", "checkpoint.strategy=epoch")
    service, run_dir = FakeTinkerService(), _run_dir(config)
    first, evaluated, crash = _start(
        monkeypatch, config, service, crash_at_update=5, once=lambda: 4 in completed_evaluation_steps(run_dir)
    )
    assert crash is not None and evaluated == [4] and _resume_point(first) == 4

    second, evaluated, crash = _start(monkeypatch, config, service)

    assert crash is None and evaluated == [8]
    assert _permanent(second) == [4, 8]  # over both starts: step 4's by the first, step 8's by the second
    assert completed_evaluation_steps(run_dir) == {4, 8}


def test_a_start_that_resumes_within_an_epoch_evaluates_at_each_epochs_end(tmp_path, monkeypatch):
    """The crash comes during step 6, once the first epoch's evaluation completed: the restart resumes at step 5, from
    the recovery checkpoint of step 5, and evaluates at steps 8 and 12, the ends of the second and third epochs.  It
    saves its first permanent checkpoint an epoch after the step it resumed at, at step 9."""

    config = _config(tmp_path, 12, "eval.every=1", "eval.strategy=epoch", "checkpoint.strategy=epoch")
    service, run_dir = FakeTinkerService(), _run_dir(config)
    first, evaluated, crash = _start(
        monkeypatch, config, service, crash_at_update=6, once=lambda: 4 in completed_evaluation_steps(run_dir)
    )
    assert crash is not None and evaluated == [4] and _resume_point(first) == 5

    second, evaluated, crash = _start(monkeypatch, config, service)

    assert crash is None and evaluated == [8, 12]
    assert _permanent(second) == [4, 9]
    assert completed_evaluation_steps(run_dir) == {4, 8, 12}


def test_a_resumed_step_schedule_skips_the_evaluation_an_earlier_start_completed(tmp_path, monkeypatch):
    """With ``strategy: step`` and ``every: 2`` in both sections, the crash comes during step 3, after step 2's
    evaluation completed: the restart resumes at step 2, does not evaluate it again, and evaluates and saves a
    permanent checkpoint at step 4."""

    config = _config(
        tmp_path, 4, "eval.strategy=step", "eval.every=2", "checkpoint.strategy=step", "checkpoint.every=2"
    )
    service, run_dir = FakeTinkerService(), _run_dir(config)
    first, evaluated, crash = _start(
        monkeypatch, config, service, crash_at_update=3, once=lambda: 2 in completed_evaluation_steps(run_dir)
    )
    assert crash is not None and evaluated == [2] and _resume_point(first) == 2

    second, evaluated, crash = _start(monkeypatch, config, service)

    assert crash is None and evaluated == [4]
    assert _permanent(second) == [2, 4]
    assert completed_evaluation_steps(run_dir) == {2, 4}
