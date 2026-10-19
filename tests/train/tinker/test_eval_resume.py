"""Evaluation during training over the starts of one run: a fresh start, with ``eval.at_start`` and without, and with
an ``eval.every`` past its last step; a start that resumes after a crash, at a multiple of ``eval.every`` and between
two; one that resumes a run whose last step was trained; and the record that tells them apart, the run directory's
``evaluations.jsonl``.

Each start is ``train()`` on the run directory, which resumes from the newest checkpoint that ``checkpoints.jsonl``
lists: what ``platoon train`` runs, and what it runs again after a failure with ``watchdog.max_restarts``
(``platoon.train.supervise``).  A crash here is an error no handler of the trainer catches, as none outlives a killed
process: the start ends without an interrupt checkpoint.

    HF_HUB_OFFLINE=1 python -m pytest tests/train/tinker/test_eval_resume.py -q
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from collections.abc import Callable
from pathlib import Path

import pytest
from tinker_service_fakes import FakeTinkerService, make_pruning_trainer
from trainer_fakes import Workflow, trainer_config

from platoon.deprecations import PlatoonDeprecationWarning
from platoon.train.components import task_ids_to_dataset
from platoon.train.tinker import rl
from platoon.train.tinker.evaluation_record import EVALUATIONS_NAME, completed_evaluation_steps, record_evaluation
from platoon.utils.jsonl import read_jsonl


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT, SIGTERM, SIGUSR1 and SIGUSR2 handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _Crash(BaseException):
    """The process dying: no handler of the trainer catches it, so the start saves nothing more."""


class _Evaluation:
    """An evaluation workflow whose rollouts take a moment; with ``crash_at``, the rollout of that number (counted from
    1 over the start) crashes the start, and with ``stuck``, every rollout waits for good."""

    def __init__(self, *, crash_at: int | None = None, stuck: bool = False) -> None:
        self.crash_at, self.stuck = crash_at, stuck
        self.started = 0

    async def arun_episode(self, data):
        self.started += 1
        if self.started == self.crash_at:
            raise _Crash(f"crashed in evaluation rollout {self.started}")
        if self.stuck:
            await asyncio.Event().wait()
        await asyncio.sleep(0.01)
        return []


def _run_config(tmp_path: Path, *overrides: str):
    """Four steps of two tasks, an evaluation of two held-out tasks every 2 steps, and a recovery checkpoint after every
    step, so that a crash resumes from the step before it."""

    return trainer_config(
        tmp_path,
        "train.max_training_steps=4",
        "eval.every=2",
        "eval.max_concurrent_rollouts=2",
        "checkpoint.recovery_every=1",
        *overrides,
    )


def _run_dir(config) -> Path:
    return Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)


def _crash_at_update(trainer, update: int, once: Callable[[], bool] = lambda: True) -> None:
    """Have the start's training client crash at its ``update``-th optimizer step, once ``once()`` holds."""

    create = trainer._create_training_client

    async def create_training_client(service_client, resume_info):
        client = await create(service_client, resume_info)
        step = client.optim_step_async

        async def optim_step_async(params):
            if client.optim_steps + 1 == update:
                while not once():
                    await asyncio.sleep(0.01)
                raise _Crash(f"crashed at optimizer step {update}")
            return await step(params)

        client.optim_step_async = optim_step_async
        return client

    trainer._create_training_client = create_training_client


def _start(monkeypatch, config, service, *, evaluation=None, crash_at_update=None, once=lambda: True):
    """One start of the run: ``(trainer, steps it evaluated, its crash or None)``, the steps as each evaluation it
    logged gives its ``eval/step``."""

    trainer, logged = make_pruning_trainer(monkeypatch, config, service)
    trainer.eval_dataset = task_ids_to_dataset(["val:0", "val:1"])
    if crash_at_update is not None:
        _crash_at_update(trainer, crash_at_update, once)

    async def run() -> BaseException | None:
        task = asyncio.ensure_future(
            trainer.train(train_workflow=Workflow(delay=0.01), eval_workflow=evaluation or _Evaluation())
        )
        done, _ = await asyncio.wait({task}, timeout=60)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise AssertionError("the start did not end within 60 s")
        return task.exception()

    error = asyncio.run(run())
    if error is not None and (not isinstance(error, BaseExceptionGroup) or error.subgroup(_Crash) is None):
        raise error  # a failure of the test, not the crash it set up
    return trainer, [stats["eval/step"] for _step, stats in logged if "eval/step" in stats], error


def _resume_point(trainer) -> int:
    """The step a start of the run resumes at: the ``batch`` of the newest checkpoint with weights and optimizer."""

    rows = read_jsonl(Path(trainer.run_log_path) / "checkpoints.jsonl")
    return [row for row in rows if row.get("state_path")][-1]["batch"]


@pytest.mark.parametrize(("at_start", "evaluated"), [(False, [2, 4]), (True, [0, 2, 4])])
def test_a_fresh_run_evaluates_its_starting_policy_only_with_at_start(
    tmp_path, monkeypatch, caplog, at_start, evaluated
):
    config = _run_config(tmp_path, f"eval.at_start={str(at_start).lower()}")

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        trainer, steps, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and steps == evaluated
    assert completed_evaluation_steps(trainer.run_log_path) == set(evaluated)
    if at_start:
        assert "Evaluating the starting policy at step 0, before the first update (eval.at_start)" in caplog.text
    else:
        assert "eval.at_start: true would evaluate the starting policy at step 0 too" in caplog.text


def test_a_restart_does_not_run_again_an_evaluation_that_completed_where_it_resumes(tmp_path, monkeypatch, caplog):
    """The crash comes during step 3, after the evaluation of step 2 completed: the restart resumes at step 2, and
    evaluates at step 4 only."""

    config, service = _run_config(tmp_path), FakeTinkerService()
    run_dir = _run_dir(config)
    first, steps, crash = _start(
        monkeypatch, config, service, crash_at_update=3, once=lambda: 2 in completed_evaluation_steps(run_dir)
    )
    assert crash is not None and steps == [2] and _resume_point(first) == 2

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        second, steps, crash = _start(monkeypatch, config, service)

    assert crash is None and steps == [4]
    assert [line["step"] for line in read_jsonl(run_dir / EVALUATIONS_NAME)] == [2, 4]
    assert "Resuming at step 2, whose evaluation an earlier start completed (evaluations.jsonl)" in caplog.text


def test_a_restart_runs_again_the_evaluation_a_crash_cut_short_where_it_resumes(tmp_path, monkeypatch, caplog):
    """The crash comes during step 3 while the evaluation of step 2 runs: the restart resumes at step 2 and evaluates
    it, then at step 4."""

    config, service = _run_config(tmp_path), FakeTinkerService()
    evaluation = _Evaluation(stuck=True)
    first, steps, crash = _start(
        monkeypatch, config, service, evaluation=evaluation, crash_at_update=3, once=lambda: evaluation.started > 0
    )
    assert crash is not None and steps == [] and _resume_point(first) == 2
    assert completed_evaluation_steps(first.run_log_path) == set()

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        _second, steps, crash = _start(monkeypatch, config, service)

    assert crash is None and steps == [2, 4]
    assert "Resuming at step 2, whose evaluation is due and no earlier start completed" in caplog.text


def test_a_restart_between_two_multiples_evaluates_at_the_next(tmp_path, monkeypatch, caplog):
    """The crash comes during step 4: the restart resumes at step 3, which no evaluation is due at, and evaluates at
    step 4 only."""

    config, service = _run_config(tmp_path), FakeTinkerService()
    run_dir = _run_dir(config)
    first, steps, crash = _start(
        monkeypatch, config, service, crash_at_update=4, once=lambda: 2 in completed_evaluation_steps(run_dir)
    )
    assert crash is not None and steps == [2] and _resume_point(first) == 3

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        _second, steps, crash = _start(monkeypatch, config, service)

    assert crash is None and steps == [4]
    assert "Resuming at step 3: the next evaluation is at step 4" in caplog.text


def test_a_restart_after_a_crash_in_the_last_evaluation_runs_it_and_a_later_start_does_not(
    tmp_path, monkeypatch, caplog
):
    """The crash comes during the evaluation due at the last step, after that step's checkpoint: the restart resumes
    there with nothing left to train, runs that evaluation and ends, and says so; a start after it evaluates
    nothing."""

    config, service = _run_config(tmp_path), FakeTinkerService()
    first, steps, crash = _start(monkeypatch, config, service, evaluation=_Evaluation(crash_at=3))  # step 4's first
    assert crash is not None and steps == [2] and _resume_point(first) == 4

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        second, steps, crash = _start(monkeypatch, config, service)
    assert crash is None and steps == [4]
    assert "Training was already complete; this start ran the evaluation due at step 4" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=rl.__name__):
        third, steps, crash = _start(monkeypatch, config, service)
    assert crash is None and steps == []
    assert "no other is due by the run's last step, 4" in caplog.text
    assert "Training was already complete; nothing to do" in caplog.text
    assert [line["step"] for line in read_jsonl(Path(third.run_log_path) / EVALUATIONS_NAME)] == [2, 4]


@pytest.mark.parametrize(("at_start", "evaluated"), [(False, []), (True, [0])])
def test_an_every_past_the_last_step_evaluates_nothing_but_the_starting_policy_and_says_so(
    tmp_path, monkeypatch, caplog, at_start, evaluated
):
    """``eval.every`` past a run's last step leaves no step to evaluate but step 0, and that only with
    ``eval.at_start``: a run without it evaluates nothing, and warns."""

    config = _run_config(tmp_path, "eval.every=10", f"eval.at_start={str(at_start).lower()}")

    with caplog.at_level(logging.INFO, logger=rl.__name__):
        _trainer, steps, crash = _start(monkeypatch, config, FakeTinkerService())

    assert crash is None and steps == evaluated
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == rl.__name__ and record.levelno >= logging.WARNING
    ]
    if at_start:
        assert "(eval.at_start), and no other by the run's last step, 4" in caplog.text and not warnings
    else:
        assert warnings == [
            "No evaluation runs: eval.every is 10, past the run's last step, 4; eval.at_start: true would evaluate "
            "the starting policy at step 0"
        ]


def test_a_restart_after_a_crash_in_the_evaluation_of_the_starting_policy_runs_it_again(tmp_path, monkeypatch):
    """With ``eval.at_start``, a crash during the evaluation of the starting policy leaves only the checkpoint the
    start saved at launch: the restart resumes at step 0, evaluates the starting policy again before its first update,
    and goes on."""

    config, service = _run_config(tmp_path, "eval.at_start=true"), FakeTinkerService()
    first, steps, crash = _start(monkeypatch, config, service, evaluation=_Evaluation(crash_at=1))
    assert crash is not None and steps == [] and _resume_point(first) == 0
    assert first._model_info.llm.version == 0  # the first update waited for the evaluation, which never ended

    _second, steps, crash = _start(monkeypatch, config, service)

    assert crash is None and steps == [0, 2, 4]


def test_a_restart_at_step_0_after_the_starting_policy_was_evaluated_does_not_evaluate_it_again(tmp_path, monkeypatch):
    """With ``eval.at_start``, a crash at the first update, after the evaluation of the starting policy completed,
    resumes the run at step 0: the restart neither evaluates the starting policy again nor holds its first update."""

    config, service = _run_config(tmp_path, "eval.at_start=true"), FakeTinkerService()
    first, steps, crash = _start(monkeypatch, config, service, crash_at_update=1)
    assert crash is not None and steps == [0] and _resume_point(first) == 0

    _second, steps, crash = _start(monkeypatch, config, service)

    assert crash is None and steps == [2, 4]


@pytest.mark.parametrize("off", ["eval.every=0", "eval={every: 3, strategy: none}"])
def test_at_start_without_evaluation_is_ignored_with_a_warning(tmp_path, monkeypatch, caplog, off):
    if "strategy" in off:  # the deprecated strategy: none turns evaluation off whatever every says
        with pytest.warns(PlatoonDeprecationWarning, match="eval.strategy"):
            config = _run_config(tmp_path, off, "eval.at_start=true")
    else:
        config = _run_config(tmp_path, off, "eval.at_start=true")

    with caplog.at_level(logging.WARNING, logger=rl.__name__):
        trainer, logged = make_pruning_trainer(monkeypatch, config, FakeTinkerService())
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Workflow(delay=0.01)), timeout=60))

    reason = "eval.every is 0" if off == "eval.every=0" else "eval.strategy is none"
    assert f"eval.at_start is ignored: {reason}, which turns evaluation during training off" in caplog.text
    assert not [stats for _step, stats in logged if "eval/step" in stats]
    assert not (Path(trainer.run_log_path) / EVALUATIONS_NAME).exists()


# --- the record ---------------------------------------------------------------------------------------------------


def test_the_record_skips_what_does_not_parse_and_a_new_line_starts_on_its_own(tmp_path):
    path = tmp_path / EVALUATIONS_NAME
    assert completed_evaluation_steps(tmp_path) == set()  # no record: none completed
    lines = [
        json.dumps({"step": 2, "logged_at": 3, "tasks": 2, "seconds": 1.0}),
        json.dumps({"step": True}),  # no step
        json.dumps(["step", 4]),  # no object
        '{"step": 6, "logged_',  # cut short by a crash
    ]
    path.write_text("\n".join(lines))

    record_evaluation(path, step=8, logged_at=9, tasks=2, seconds=12.345)

    assert completed_evaluation_steps(tmp_path) == {2, 8}
    last = json.loads(path.read_text().splitlines()[-1])
    assert {key: last[key] for key in ("step", "logged_at", "tasks", "seconds")} == {
        "step": 8,
        "logged_at": 9,
        "tasks": 2,
        "seconds": 12.3,
    }


def test_a_record_that_cannot_be_written_is_logged_and_training_goes_on(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        record_evaluation(tmp_path / "gone" / EVALUATIONS_NAME, step=4, logged_at=5, tasks=2, seconds=1.0)

    assert "Could not record the evaluation of step 4" in caplog.text
    assert "a start that resumes at step 4 evaluates it again" in caplog.text
