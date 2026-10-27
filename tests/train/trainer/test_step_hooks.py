"""Step hooks (RFC 0021, section 9) and the rules that come with them, on ``platoon.testing``'s fake Tinker service.

A hook drains before its step publishes, adds optimizer updates through ``end.train`` or ``end.learner`` that platoon
counts, during its ``update()`` only, one update of a policy at a time and never beside an update left open, and the
step publishes once, its own updates and the hooks' together; it hears every publication, the launch's too.  A critical
hook's error in drain or update fails the start with nothing published or saved; one that is not critical has its
errors logged.  A stop during the hooks saves nothing of the step when the trainer's steps are atomic (rule 10), and
lets the step end first when they are not, where an interrupt that comes as the hooks run all the same saves nothing of
the step either.  A hook with state is a checkpoint participant, which takes up its state before the training client
opens; a key that the checkpoint row holds itself, or a hook's of the draws, is refused as it registers; a saved key
that nothing claims stops the start; the run closes its hooks and participants in reverse, also after a failure, whose
error it raises as it was.  And what comes with them: the per-update Adam parameters, updates
that fail closed, a trainer's own steps through ``run.learner`` and ``run.end_step``, the policy's snapshot, a warm
start with the optimizer's state, the draws' provenance, and the hooks the rl trainer attaches from its options."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import signal
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import tinker
from fakes.trainer import Future
from tinker_cookbook.checkpoint_utils import CheckpointRecord

import platoon.experimental as px
from platoon.config_defs import RolloutConfig
from platoon.programs import DRAW_KEY, DrawInfo, Episode
from platoon.registry import register_component
from platoon.task import Task
from platoon.testing import check_trainer
from platoon.testing import trainers as checks
from platoon.train.checkpoints.participants import CORE_KEYS, RunCheckpoints, row_fields
from platoon.train.components import task_ids_to_dataset
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.dispatch.source import RolloutSource
from platoon.train.policy.learner import Learner, UpdateRejected
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import MAIN, PolicySpec, policy_specs
from platoon.train.trainer import signals
from platoon.train.trainer.hook_config import HookSpec
from platoon.train.trainer.hooks import RunSide, StepEnd, StepHooks
from platoon.train.trainer.registry import RLOptions, trainer_options
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import ConfigError, build_config
from platoon.utils.stats_logger import StatsLogger

# --- one start on the fakes -----------------------------------------------------------------------------------------


@dataclass
class _Start:
    rows: list[dict[str, Any]]
    error: BaseException | None
    service: checks.FakePolicyService


def _start(
    trainer: Any,
    root: Path,
    *,
    steps: int = 2,
    hooks: tuple[Any, ...] = (),
    settings: dict[str, Any] | None = None,
    service: checks.FakePolicyService | None = None,
    workflow: Any = None,
    options: Any = None,
) -> _Start:
    """One start of ``trainer`` on the check's fakes: ``steps`` steps of two one-datum groups (``workflow``'s, when
    given), with ``hooks`` registered after the trainer's own check, ``settings`` laid over the check's config,
    section by section, and the trainer built with ``options``."""

    given = checks._settings(root, steps)
    for section, values in (settings or {}).items():
        given[section] = {**given[section], **values} if isinstance(values, dict) and section in given else values
    config = build_config(PlatoonTinkerRLTrainerConfig, given)
    run_dir = Path(config.log_path, config.stats.experiment_name, config.stats.trial_name)
    before = len(checks._rows(run_dir))
    setup = px.TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        tasks=px.TaskSources(train_dataset=task_ids_to_dataset([f"task-{index}" for index in range(16)])),
        workflow_builder=lambda policies: px.Workflows(workflow or checks._OneDatumGroups(), None),
        options=options,
    )
    service = service or checks.FakePolicyService()
    error = None
    with checks._faked(service), checks._restored_signals():
        try:
            built = trainer.from_setup(setup)
            _register_after_check(built, hooks)
            asyncio.run(asyncio.wait_for(checks._run(built), 60))
        except Exception as exc:  # noqa: BLE001 - the tests read what failed
            error = exc
    return _Start(checks._rows(run_dir)[before:], error, service)


def _register_after_check(trainer: Any, hooks: tuple[Any, ...]) -> None:
    check = trainer.check

    def checked(run: Any) -> None:
        check(run)
        for hook in hooks:
            run.hooks.register(hook)

    trainer.check = checked


def _names(start: _Start) -> list[str]:
    return [row["name"] for row in start.rows]


def _errors(start: _Start) -> list[str]:
    """What the start failed with, each error of a group (the rl trainer's loops raise a step's error in one)."""

    error = start.error
    if isinstance(error, BaseExceptionGroup):
        return [str(inner) for inner in error.exceptions]
    return [] if error is None else [str(error)]


class _Atomic(PlatoonTinkerRLTrainer):
    """The rl trainer, its steps atomic and its updates failing closed, as an outside trainer may ask (rule 10)."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        return dataclasses.replace(super().needs(config, options), atomic_steps=True, fail_closed_updates=True)


# --- hooks -----------------------------------------------------------------------------------------------------------


class _Recorder(px.StepHookBase):
    """Records what it does in ``log`` (the fake service's calls, so that the order shows), and takes ``updates``
    optimizer updates at each step's end, with ``adam`` as their Adam parameters; ``fail`` (a method and a step) raises
    there; ``stop`` (a method and a step) asks for a stop there, then waits a moment, interruptibly when
    ``interruptible``."""

    def __init__(
        self,
        log: list[str],
        *,
        name: str = "recorder",
        updates: int = 0,
        adam: Any = None,
        fail: tuple[str, int] | None = None,
        stop: tuple[str, int] | None = None,
        interruptible: bool = True,
    ) -> None:
        self.log, self.name, self.updates, self.adam = log, name, updates, adam
        self.fail, self.stop, self.interruptible = fail, stop, interruptible

    async def drain(self, end: px.StepEnd) -> None:
        self.log.append(f"{self.name} drains {end.step}")
        await self._maybe("drain", end)

    async def update(self, end: px.StepEnd) -> None:
        self.log.append(f"{self.name} updates {end.step} after {dict(end.step_updates)}")
        await self._maybe("update", end)
        for _ in range(self.updates):
            await end.train(MAIN, [checks._probe_datum()], objective=checks._ProbeObjective(), adam_params=self.adam)

    async def published(self, end: px.StepEnd) -> None:
        self.log.append(f"{self.name} hears {end.kind} {end.step}: {dict(end.published)} {end.hook_updates}")
        if end.kind == "step":
            await self._maybe("published", end)

    async def _maybe(self, method: str, end: px.StepEnd) -> None:
        if self.fail == (method, end.step):
            raise RuntimeError(f"{self.name} failed in {method}")
        if self.stop == (method, end.step):
            signal.raise_signal(signal.SIGUSR2)
            waited = asyncio.sleep(0.2)
            await end.wait(waited, name=f"{self.name}'s side work", timeout=10, interruptible=self.interruptible)


class _Deaf(px.StepHookBase):
    """A hook that is not critical, whose hearing fails."""

    name = "deaf"
    critical = False

    async def published(self, end: px.StepEnd) -> None:
        raise RuntimeError("deaf cannot hear")


class _Cursor(px.StepHookBase):
    """A hook with state to resume, a checkpoint participant: how many steps' ends it saw, which it records in the
    fake service's calls (``log``) as it takes it up."""

    name = "cursor"
    keys = ("cursor",)
    log: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.seen = 0

    @classmethod
    def from_run(cls, run: Any, options: Any) -> _Cursor:
        hook = cls()
        hook.seen = options.get("from", 0)
        return hook

    def resume(self, start: Any) -> None:
        if start.record is not None:
            self.seen = start.loop_state["cursor"]["seen"]
        self.log.append(f"cursor resumes at {self.seen}")

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {"cursor": {"seen": self.seen}}

    async def update(self, end: px.StepEnd) -> None:
        self.seen += 1


register_component("step_hook", "test-step-hooks-cursor", _Cursor, exist_ok=True)


# --- the sequence: counted updates, one publication, the order ------------------------------------------------------


def _step_calls(step: int, *, hook_updates: int = 2) -> list[str]:
    """What the fake service and the recorder see of one rl step whose end the recorder adds ``hook_updates``
    updates to."""

    return [
        "forward_backward 2",  # the step's own update, over its two groups
        "optim_step",
        f"recorder drains {step}",
        f"recorder updates {step} after {{'main': 1}}",
        *["forward_backward 1", "optim_step"] * hook_updates,
        "save_weights_and_get_sampling_client",  # one publication, after the drain and every update
        f"recorder hears step {step}: {{'main': {step + 1}}} {{'main': {hook_updates}}}",
    ]


def _after_launch(service: checks.FakePolicyService) -> list[str]:
    return service.calls[service.calls.index("save_state recovery-000000") + 1 :]


def test_the_hooks_updates_are_counted_and_the_step_publishes_them_with_its_own_once(tmp_path, monkeypatch):
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(StatsLogger, "log", lambda self, step, stats, **kwargs: logged.append(dict(stats)))
    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=(_Recorder(service.calls, updates=2),), service=service)

    assert start.error is None
    assert _after_launch(service) == [
        "recorder hears launch 0: {'main': 0} {'main': 0}",
        *_step_calls(0),
        *_step_calls(1),
        "save_state final",
    ]
    assert [(row["name"], row["policy_version"], row["optimizer_steps"]) for row in start.rows] == [
        ("recovery-000000", 0, 0),
        ("final", 2, 6),  # two versions; each step's own update and its hook's two
    ]
    counted = ("train/batch/optimizer_updates", "train/batch/hook_optimizer_updates", "train/optim/updates_total")
    assert [tuple(row[name] for name in counted) for row in logged] == [(1.0, 2.0, 3.0), (1.0, 2.0, 6.0)]


class _Zero(checks._OneDatumGroups):
    """Groups whose every advantage is zero: a step takes no update of its own."""

    async def arun_episode(self, data: dict) -> list[Any]:
        import torch

        (datum,) = await super().arun_episode(data)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(torch.zeros(3))
        return [datum]


def test_a_step_that_only_its_hooks_updated_publishes_and_one_that_nothing_updated_does_not(tmp_path):
    service = checks.FakePolicyService()
    hooks = (_Recorder(service.calls, updates=1),)
    start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=hooks, service=service, workflow=_Zero())
    assert start.error is None
    assert service.calls.count("save_weights_and_get_sampling_client") == 2 and start.rows[-1]["policy_version"] == 2

    quiet = checks.FakePolicyService()
    start = _start(
        PlatoonTinkerRLTrainer, tmp_path / "quiet", hooks=(_Recorder(quiet.calls),), service=quiet, workflow=_Zero()
    )
    assert start.error is None
    assert "save_weights_and_get_sampling_client" not in quiet.calls and start.rows[-1]["policy_version"] == 0
    assert "recorder hears step 1: {} {'main': 0}" in quiet.calls  # the hooks hear every step's end


def test_the_adam_parameters_of_one_update_reach_its_optimizer_step_alone(tmp_path):
    service = checks.FakePolicyService()
    hooks = (_Recorder(service.calls, updates=1, adam=tinker.AdamParams(learning_rate=0.5)),)
    start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=hooks, service=service)

    assert start.error is None
    own = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(tmp_path, 2)).train.optimizer.learning_rate
    assert [params.learning_rate for params in service.adam] == [own, 0.5, own, 0.5]


# --- failures --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["drain", "update"])
def test_a_critical_hook_s_error_fails_the_start_with_nothing_published_or_saved(tmp_path, method):
    service = checks.FakePolicyService()
    hooks = (_Recorder(service.calls, fail=(method, 1)),)
    start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=hooks, service=service)

    # The error ends the rl trainer's loops, whose task group raises it, as it raises a failed update's.
    assert isinstance(start.error, ExceptionGroup) and [str(error) for error in start.error.exceptions] == [
        f"recorder failed in {method}"
    ]
    assert _names(start) == ["recovery-000000"]  # neither final nor <step>_interrupted
    assert service.calls.count("save_weights_and_get_sampling_client") == 1  # step 0's alone
    assert not any(call.startswith("recorder hears step 1") for call in service.calls)


def test_a_hook_that_is_not_critical_has_its_errors_logged_and_training_goes_on(tmp_path, caplog):
    with caplog.at_level(logging.ERROR, logger="platoon.train.trainer.hooks"):
        start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=(_Deaf(),))

    assert start.error is None and _names(start) == ["recovery-000000", "final"]
    assert [record.getMessage() for record in caplog.records] == [
        f"Step hook deaf failed to hear the end of step {step}; it is not critical, so training goes on"
        for step in (0, 0, 1)  # the launch's, then each step's
    ]


def test_the_run_refuses_a_hook_without_a_name_a_name_twice_and_one_not_critical_that_updates():
    hooks = StepHooks()
    hooks.register(_Recorder([]))
    with pytest.raises(ValueError, match="two step hooks are named 'recorder'"):
        hooks.register(_Recorder([]))
    with pytest.raises(ValueError, match="a step hook has a name"):
        hooks.register(px.StepHookBase())

    class _Hasty(_Deaf):
        name = "hasty"

        async def update(self, end: px.StepEnd) -> None: ...

    with pytest.raises(ValueError, match="'hasty' is not critical, so it only hears publications"):
        hooks.register(_Hasty())


class _Keeper(px.StepHookBase):
    """Keeps the learner its update() was given and the update it applied with it, and as it hears the step's end uses
    them again (``call``): a submission, an optimizer step, or the applied update's count, once more; a submission
    through the kept learner under another objective (``for_objective``), or through the trainer's learner of the run
    it holds (``run_learner``).  With ``call`` "next_update", step 1's update() submits through step 0's learner."""

    name = "keeper"

    def __init__(self, call: str, run: Any = None) -> None:
        self.call, self.run = call, run

    async def update(self, end: px.StepEnd) -> None:
        if self.call == "next_update" and end.step > 0:
            await self.kept.update().submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        self.kept = end.learner(MAIN, objective=checks._ProbeObjective())
        self.applied = self.kept.update()
        await self.applied.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        await self.applied.apply()

    async def published(self, end: px.StepEnd) -> None:
        if end.kind != "step" or self.call == "next_update":
            return
        if self.call == "submit":
            await self.kept.update().submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        elif self.call == "for_objective":
            update = self.kept.for_objective(checks._ProbeObjective()).update()
            await update.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        elif self.call == "run_learner":
            update = self.run.learner(MAIN, objective=checks._ProbeObjective()).update()
            await update.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        elif self.call == "submit_optim_step":
            await self.kept.update().submit_optim_step()
        else:
            await self.applied.finish()


@pytest.mark.parametrize("call", ["submit", "submit_optim_step", "finish"])
def test_a_hook_s_learner_sends_and_counts_during_its_update_only(tmp_path, call):
    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path, hooks=(_Keeper(call),), service=service)

    assert _errors(start) == ["step hook 'keeper' updates the policy during its update() only"]
    # Step 0's own update and the hook's, in its update(); nothing sent after the step published.
    assert _after_launch(service) == [
        "forward_backward 2",
        "optim_step",
        "forward_backward 1",
        "optim_step",
        "save_weights_and_get_sampling_client",
    ]
    run = RunSide(policies={}, learner_of=cast(Any, None), wait=cast(Any, None), interruptible=False)
    end = StepEnd(kind="step", step=0, step_updates={}, consumed={}, tracker=cast(Any, None), _run=run)
    with pytest.raises(RuntimeError, match=r"end.learner\(...\) is a step hook's, during its update\(\) only"):
        end.learner(MAIN, objective=checks._ProbeObjective())


def _keeping(trainer: Any, call: str) -> Any:
    """``trainer``, whose check registers a :class:`_Keeper` that holds the run, as one the config names would."""

    class _Keeping(trainer):
        def check(self, run: Any) -> None:
            super().check(run)
            run.hooks.register(_Keeper(call, run))

    return _Keeping


_REFUSALS = {
    # The kept learner under another objective, as the hook hears the step's end.
    "for_objective": "step hook 'keeper' updates the policy during its update() only",
    # The trainer's learner, which the run the hook holds gives, as the hook hears the step's end.
    "run_learner": (
        "run.learner() is the trainer's, which sends nothing as a step's end runs its step hooks: a hook updates the "
        "policy through end.train(...) or end.learner(...), during its update()"
    ),
    # Step 0's learner, in step 1's update().
    "next_update": (
        "step hook 'keeper' sends through a learner of an earlier step, step 0: a learner updates the policy during "
        "the update() that took it only, so take one in each update()"
    ),
}


@pytest.mark.parametrize("trainer", ["rl", "atomic"])
@pytest.mark.parametrize("call", list(_REFUSALS))
def test_a_hook_s_learner_refuses_under_another_objective_and_a_step_later_and_the_trainer_s_as_the_hooks_run(
    tmp_path, monkeypatch, trainer, call
):
    monkeypatch.setattr(_Closing, "log", [])
    service = checks.FakePolicyService()
    base = PlatoonTinkerRLTrainer if trainer == "rl" else _ThreeUpdates
    start = _start(_keeping(base, call), tmp_path, service=service)

    assert _errors(start) == [_REFUSALS[call]]
    own = ["forward_backward 2", "optim_step"] if trainer == "rl" else ["forward_backward 1", "optim_step"] * 3
    step_0 = [*own, "forward_backward 1", "optim_step", "save_weights_and_get_sampling_client"]
    # Nothing sent after step 0 published, or, a step later, after step 1's own updates.
    assert _after_launch(service) == (step_0 + own if call == "next_update" else step_0)


class _Leaves(px.StepHookBase):
    """Sends an update in its update() and leaves it open: no optimizer step takes it in."""

    name = "leaves"

    async def update(self, end: px.StepEnd) -> None:
        update = end.learner(MAIN, objective=checks._ProbeObjective()).update()
        await update.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)


class _LeavesItsOwn(px.TrainerBase):
    """A trainer whose step sends an update through run.learner and ends without applying it."""

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        return px.StepPlan(end_step=run.config.train.max_training_steps)

    async def run_steps(self, run: Any) -> None:
        update = run.learner(MAIN, objective=checks._ProbeObjective()).update()
        await update.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
        await run.end_step()


class _Background(px.StepHookBase):
    """Leaves its update running in the background as its update() returns, once the update's optimizer step was
    sent: the step's end would publish an optimizer step that no count holds yet."""

    name = "background"

    def __init__(self, service: checks.FakePolicyService) -> None:
        self.service = service

    async def update(self, end: px.StepEnd) -> None:
        sent = self.service.optimizer_steps
        update = end.train(MAIN, [checks._probe_datum()], objective=checks._ProbeObjective())
        self.running = asyncio.ensure_future(update)
        while self.service.optimizer_steps == sent:
            await asyncio.sleep(0)


_LEFT_OPEN = (
    "left an update of policy 'main' open, sent but not applied (submit without apply, or an apply still running)"
)


@pytest.mark.parametrize("hooked", [True, False])
def test_a_step_s_end_refuses_an_update_the_step_left_open_with_hooks_or_without(tmp_path, hooked):
    service = checks.FakePolicyService()
    hooks = (_Recorder(service.calls, updates=1),) if hooked else ()
    start = _start(_LeavesItsOwn, tmp_path, hooks=hooks, service=service)

    assert str(start.error).startswith(f"the step {_LEFT_OPEN}")
    calls = _after_launch(service)  # the hooks heard the launch, then nothing ran, applied or published
    assert calls[-1] == "forward_backward 1" and "optim_step" not in calls and "recorder drains 0" not in calls


def test_a_step_s_end_refuses_an_update_a_hook_left_open_or_running(tmp_path):
    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path / "open", hooks=(_Leaves(),), service=service)
    assert [error.split(":")[0] for error in _errors(start)] == [f"a step hook {_LEFT_OPEN}"]
    assert service.calls.count("save_weights_and_get_sampling_client") == 0  # step 0 never published

    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path / "running", hooks=(_Background(service),), service=service)
    assert [error.split(":")[0] for error in _errors(start)] == [f"a step hook {_LEFT_OPEN}"]
    assert service.calls.count("save_weights_and_get_sampling_client") == 0  # its optimizer step sent, not counted


class _Together(px.StepHookBase):
    """Sends two updates of one policy at once."""

    name = "together"

    async def update(self, end: px.StepEnd) -> None:
        datum, objective = checks._probe_datum(), checks._ProbeObjective()
        await asyncio.gather(*(end.train(MAIN, [datum] * size, objective=objective) for size in (1, 3)))


def test_a_policy_takes_one_update_at_a_time(tmp_path, monkeypatch):
    sends = checks._FakeTrainingClient.forward_backward_async

    async def yielding(self: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)  # as a real service's call yields: the other update's submission runs meanwhile
        return await sends(self, *args, **kwargs)

    monkeypatch.setattr(checks._FakeTrainingClient, "forward_backward_async", yielding)
    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path, steps=1, hooks=(_Together(),), service=service)

    assert _errors(start) == [
        "an update of policy 'main' is open, sent and not applied yet: apply it before sending another, or one "
        "optimizer step would take in both"
    ]
    assert "forward_backward 3" not in service.calls  # the second update never reached the service


class _FakeLearner:
    """What a hook's update() sends through a learner, recorded: each submission's datums, and each apply."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    def update(self) -> _FakeLearner:
        return self

    async def submit(self, datums: list[Any], originals: list[Any], *, minibatch: int, microbatch: int) -> None:
        self.sent.append(len(datums))

    async def apply(self, tally: Any = None, *, adam_params: Any = None) -> None:
        self.sent.append(("apply", adam_params))


def test_a_hook_alone_is_tested_on_a_step_s_end_built_for_the_test():
    learner = _FakeLearner()
    hook = _Recorder([], updates=2, adam="adam")
    asyncio.run(hook.update(StepEnd.for_test(step=3, step_updates={MAIN: 1}, learner=learner)))
    assert learner.sent == [1, ("apply", "adam"), 1, ("apply", "adam")]
    assert hook.log == ["recorder updates 3 after {'main': 1}"]

    with pytest.raises(RuntimeError, match=r"built by hand: StepEnd.for_test\(learner=...\) gives a learner"):
        asyncio.run(_Recorder([], updates=1).update(StepEnd.for_test()))


def test_a_step_s_end_shows_its_hooks_what_they_read_and_keeps_the_run_s_side_private():
    public = [item.name for item in dataclasses.fields(px.StepEnd) if not item.name.startswith("_")]
    assert public == ["kind", "step", "step_updates", "consumed", "tracker", "published"]
    reference = Path(__file__).resolve().parents[3] / "docs" / "reference" / "experimental.md"
    assert "| `_" not in reference.read_text(encoding="utf-8")  # no private field in any table
    assert px.TrainInterrupted is signals.TrainInterrupted  # what an interruptible wait raises


# --- interrupts (rule 10) --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["drain", "update", "published"])
def test_an_interrupt_during_the_hooks_of_steps_that_are_not_atomic_lets_the_step_end_first(tmp_path, method):
    hooks = (_Recorder([], updates=1, stop=(method, 1)),)
    start = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, hooks=hooks)

    # Step 1 ends, its hook's update and its publication included; the stop takes effect at step 2's boundary.
    assert start.error is None and _names(start) == ["recovery-000000", "000002_interrupted"]
    assert {key: start.rows[-1][key] for key in ("batch", "policy_version", "optimizer_steps")} == {
        "batch": 2,
        "policy_version": 2,
        "optimizer_steps": 4,  # two steps, each its own update and its hook's
    }
    resumed = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, hooks=(_Recorder([], updates=1),))
    assert resumed.error is None and resumed.rows[-1]["optimizer_steps"] == 8  # as four steps uninterrupted: none twice


@pytest.mark.parametrize("method", ["drain", "update", "published"])
def test_an_interrupt_during_the_hooks_of_atomic_steps_saves_nothing_of_the_step(tmp_path, caplog, method):
    with caplog.at_level(logging.INFO, logger="platoon.train.trainer.run"):
        start = _start(_Atomic, tmp_path, steps=4, hooks=(_Recorder([], stop=(method, 1)),))
    assert start.error is None and _names(start) == ["recovery-000000"]
    assert (
        "Training was interrupted inside step 1, which had sent training work: no checkpoint saves it, and the run "
        "resumes from checkpoint recovery-000000"
    ) in caplog.messages


class _Interrupts(px.StepHookBase):
    """Updates the policy once at each step's end, and in ``method`` of step 1 interrupts the step after that update:
    it raises ``TrainInterrupted`` itself (``how`` "raise"), or asks for a stop and waits, interruptibly, on the run
    that ``from_run`` gave it (``how`` "run.wait")."""

    name = "interrupts"

    def __init__(self, run: Any, method: str, how: str) -> None:
        self.run, self.method, self.how = run, method, how

    @classmethod
    def from_run(cls, run: Any, options: Any) -> _Interrupts:
        return cls(run, options["method"], options["how"])

    async def drain(self, end: px.StepEnd) -> None:
        await self._maybe("drain", end)

    async def update(self, end: px.StepEnd) -> None:
        await end.train(MAIN, [checks._probe_datum()], objective=checks._ProbeObjective())
        await self._maybe("update", end)

    async def published(self, end: px.StepEnd) -> None:
        if end.kind == "step":
            await self._maybe("published", end)

    async def _maybe(self, method: str, end: px.StepEnd) -> None:
        if (method, end.step) != (self.method, 1):
            return
        if self.how == "raise":
            raise px.TrainInterrupted()
        signal.raise_signal(signal.SIGUSR2)
        await self.run.wait(asyncio.sleep(10), name="the hook's side work", timeout=20, interruptible=True)


register_component("step_hook", "test-step-hooks-interrupts", _Interrupts, exist_ok=True)


@pytest.mark.parametrize(
    ("method", "how"), [("drain", "raise"), ("update", "raise"), ("published", "raise"), ("update", "run.wait")]
)
def test_an_interrupt_a_hook_makes_under_steps_that_are_not_atomic_saves_nothing_of_the_step(
    tmp_path, caplog, method, how
):
    hooks = [{"hook": "test-step-hooks-interrupts", "options": {"method": method, "how": how}}]
    with caplog.at_level(logging.INFO, logger="platoon.train.trainer.run"):
        start = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, options={"hooks": hooks})

    # Step 1's own update and its commit are done, its hooks' perhaps half: no <step>_interrupted saves it.
    assert start.error is None and _names(start) == ["recovery-000000"]
    assert (
        "Training was interrupted inside step 1, as its step hooks ran: no checkpoint saves it, and the run resumes "
        "from checkpoint recovery-000000"
    ) in caplog.messages
    resumed = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, hooks=(_Recorder([], updates=1),))
    assert resumed.error is None and resumed.rows[-1]["optimizer_steps"] == 8  # as four steps uninterrupted: none twice


@pytest.mark.parametrize("trainer", [PlatoonTinkerRLTrainer, _Atomic])
def test_a_stop_during_a_hook_s_wait_that_is_not_interruptible_lets_the_step_end_first(tmp_path, trainer):
    stop = _Recorder([], stop=("drain", 1), interruptible=False)
    start = _start(trainer, tmp_path, steps=4, hooks=(stop,))

    # Step 1 publishes; the stop takes effect at the next step's first wait, at its boundary.
    assert start.error is None and _names(start) == ["recovery-000000", "000002_interrupted"]
    assert start.rows[-1]["policy_version"] == 2


# --- participants: their state, their keys, their close --------------------------------------------------------------


def test_a_hook_s_participant_state_resumes_before_the_training_client_opens(tmp_path, monkeypatch):
    attach = {"trainer_options": {"hooks": [{"hook": "test-step-hooks-cursor", "options": {"from": 5}}]}}
    service = checks.FakePolicyService()
    monkeypatch.setattr(_Cursor, "log", service.calls)
    first = _start(PlatoonTinkerRLTrainer, tmp_path, settings=attach, service=service)
    assert first.error is None and first.rows[-1]["cursor"] == {"seen": 7}  # built with its options, then 2 steps

    resumed = checks.FakePolicyService()
    monkeypatch.setattr(_Cursor, "log", resumed.calls)
    second = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, settings=attach, service=resumed)
    assert second.error is None
    assert resumed.calls[:2] == [
        "cursor resumes at 7",  # registered in check(): taken up before the training client opens
        "create_training_client_from_state_with_optimizer tinker://fake/weights/final",
    ]
    assert [row["cursor"] for row in second.rows] == [{"seen": 7}, {"seen": 9}]


def test_a_saved_key_that_no_part_of_the_run_claims_stops_the_start(tmp_path, monkeypatch):
    monkeypatch.setattr(_Cursor, "log", [])
    attach = {"trainer_options": {"hooks": ["test-step-hooks-cursor"]}}
    assert _start(PlatoonTinkerRLTrainer, tmp_path, settings=attach).error is None

    resumed = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4)
    assert isinstance(resumed.error, ConfigError)
    assert "checkpoint final saved loop_state key 'cursor', which no part of this run takes up" in str(resumed.error)
    assert resumed.rows == []  # stopped before its launch save


def test_a_participant_s_keys_are_its_own():
    policies: Any = SimpleNamespace()
    checkpoints = RunCheckpoints("unused", policies, wandb_run_id=lambda: None)
    cursor: Any = _Cursor()
    checkpoints.register(cursor)
    other: Any = SimpleNamespace(name="other", keys=("cursor",))
    with pytest.raises(ConfigError, match="loop_state key 'cursor' is 'cursor''s: 'other' may not hold it too"):
        checkpoints.register(other)
    for key in ("batch", "policies"):
        owner: Any = SimpleNamespace(name="owner", keys=(key,))
        with pytest.raises(ConfigError, match=f"loop_state key '{key}' is the run's own"):
            checkpoints.register(owner)
    checkpoints.register(cursor)  # again: nothing changes
    assert checkpoints.participants == [cursor]
    assert set(RolloutSource.keys) <= set(CORE_KEYS)  # the draws' keys, which every release's rows may hold


# The fields of a checkpoints.jsonl row besides batch (a counter), which a loop_state entry of the same name replaced
# (name, epoch, final) or which the save's paths replaced (state_path, sampler_path).
_ROW_FIELDS = ("name", "epoch", "final", "state_path", "sampler_path")
_ROW = "is a field of the checkpoint row itself"


class _Claims(px.StepHookBase):
    """A hook with state, a checkpoint participant, that holds the keys its options name."""

    name = "claims"

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys

    @classmethod
    def from_run(cls, run: Any, options: Any) -> _Claims:
        return cls(tuple(options["keys"]))

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {key: {} for key in self.keys}

    def resume(self, start: Any) -> None:
        pass


register_component("step_hook", "test-step-hooks-claims", _Claims, exist_ok=True)


class _TrainerState:
    """A trainer's participant, which has no name: an error names its class."""

    def __init__(self, *keys: str) -> None:
        self.keys = keys

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {key: {} for key in self.keys}

    def resume(self, start: Any) -> None:
        pass


@pytest.mark.parametrize("key", _ROW_FIELDS)
def test_no_participant_holds_a_key_that_the_checkpoint_row_holds_itself(key):
    # The row's own fields, as tinker_cookbook writes the row: an entry of one of their names is not the entry's.
    assert key in row_fields() and "batch" in row_fields() and "cursor" not in row_fields()
    assert key not in CheckpointRecord.from_dict({"name": "n", key: "entry"}).extra
    assert "cursor" in CheckpointRecord.from_dict({"name": "n", "cursor": "entry"}).extra

    policies: Any = SimpleNamespace()
    checkpoints = RunCheckpoints("unused", policies, wandb_run_id=lambda: None)
    refused = f"loop_state key '{key}' {_ROW}"
    with pytest.raises(ConfigError, match=f"{refused}: '_TrainerState' may not hold it"):
        checkpoints.register(cast(Any, _TrainerState("mine", key)))
    hooks = StepHooks(checkpoints)
    with pytest.raises(ConfigError, match=f"{refused}: 'claims' may not hold it"):
        hooks.register(_Claims(("cursor", key)))
    assert checkpoints.participants == [] and len(hooks) == 0  # refused whole, as it registers


@pytest.mark.parametrize("key", ["tasks", "next_position", "dispatch", "credit"])
def test_a_hook_holds_none_of_the_draws_keys_which_a_trainer_s_participant_may(key):
    policies: Any = SimpleNamespace()
    checkpoints = RunCheckpoints("unused", policies, wandb_run_id=lambda: None)
    hooks = StepHooks(checkpoints)
    with pytest.raises(ConfigError, match=f"loop_state key '{key}' is the draws' .*: step hook 'claims' may not hold"):
        hooks.register(_Claims((key,)))
    assert checkpoints.participants == [] and len(hooks) == 0

    own = _TrainerState(key)  # a trainer's draws of its own: the run claims the key, the trainer holds it
    checkpoints.register(cast(Any, own))
    assert checkpoints.participants == [own]


class _HoldsFinal(px.TrainerBase):
    """A trainer whose check registers a participant that holds ``final``, a field of the checkpoint row."""

    def check(self, run: Any) -> None:
        run.checkpoints.register(_TrainerState("final"))

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        return px.StepPlan(end_step=run.config.train.max_training_steps)

    async def run_steps(self, run: Any) -> None:
        while run.clock.step < run.plan.end_step:
            await run.end_step()


@pytest.mark.parametrize(
    ("trainer", "key", "problem"),
    [
        (PlatoonTinkerRLTrainer, "name", f"trainer_options.hooks[0]: loop_state key 'name' {_ROW}: 'claims' may not"),
        (PlatoonTinkerRLTrainer, "state_path", f"trainer_options.hooks[0]: loop_state key 'state_path' {_ROW}"),
        (
            PlatoonTinkerRLTrainer,
            "tasks",
            "trainer_options.hooks[0]: loop_state key 'tasks' is the draws' (the rollout source's): step hook 'claims'",
        ),
        (_HoldsFinal, None, f"loop_state key 'final' {_ROW}: '_TrainerState' may not hold it"),
    ],
)
def test_a_key_of_the_row_or_a_hook_s_of_the_draws_stops_the_start_before_the_service_opens_naming_its_holder(
    tmp_path, trainer, key, problem
):
    service = checks.FakePolicyService()
    hooks = [{"hook": "test-step-hooks-claims", "options": {"keys": [key]}}]
    start = _start(trainer, tmp_path, settings={"trainer_options": {"hooks": hooks}} if key else None, service=service)
    assert isinstance(start.error, ConfigError) and problem in str(start.error), start.error  # exit 78
    assert str(start.error).count("invalid config") == 1  # the hook's item names the problem once
    assert service.calls == [] and start.rows == []


class _Closing(px.TrainerBase):
    """A trainer whose check registers three parts that record their close: a hook, a participant, and a hook with
    keys, which registering it as a hook makes a participant too; each step ends with run.end_step, and its close
    records too."""

    log: ClassVar[list[str]] = []

    def check(self, run: Any) -> None:
        run.hooks.register(_Part("hook", self.log))
        run.checkpoints.register(_Part("participant", self.log, keys=("participant",)))
        run.hooks.register(_Part("both", self.log, keys=("both",)))

    async def prepare(self, run: Any, start: Any) -> px.StepPlan:
        return px.StepPlan(end_step=run.config.train.max_training_steps)

    async def run_steps(self, run: Any) -> None:
        while run.clock.step < run.plan.end_step:
            await run.end_step()

    async def close(self, run: Any) -> None:
        self.log.append("trainer closes")


class _Part(px.StepHookBase):
    def __init__(self, name: str, log: list[str], *, keys: tuple[str, ...] = (), fails: str | None = None) -> None:
        self.name, self.log, self.keys, self.fails = name, log, keys, fails

    async def update(self, end: px.StepEnd) -> None:
        if self.fails == "update":
            raise RuntimeError(f"{self.name} failed")

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {key: {} for key in self.keys}

    def resume(self, start: Any) -> None:
        pass

    async def aclose(self) -> None:
        self.log.append(f"{self.name} closes")
        if self.fails == "aclose":
            raise RuntimeError(f"{self.name} failed to close")


def test_the_run_closes_its_hooks_and_participants_in_reverse_after_the_trainer_also_after_a_failure(
    tmp_path, monkeypatch
):
    log: list[str] = []
    monkeypatch.setattr(_Closing, "log", log)
    ends = _start(_Closing, tmp_path / "ends")
    assert ends.error is None and log == ["trainer closes", "both closes", "participant closes", "hook closes"]
    assert all({"participant", "both"} <= set(row) for row in ends.rows)  # the hook with keys is a participant

    log.clear()
    failing = _start(_Closing, tmp_path / "fails", hooks=(_Part("late", log, fails="update"),))
    assert isinstance(failing.error, RuntimeError) and str(failing.error) == "late failed"  # raised as it was
    assert log == ["trainer closes", "late closes", "both closes", "participant closes", "hook closes"]

    # A start that failed raises its error as it was, the close's errors noted on it; one that ended raises the
    # close's error, or its errors together.
    hooks = (_Part("late", log, fails="aclose"), _Part("worse", log, fails="update"))
    both = _start(_Closing, tmp_path / "both", hooks=hooks)
    assert isinstance(both.error, RuntimeError) and str(both.error) == "worse failed"
    assert both.error.__notes__ == ["Closing the start failed too: RuntimeError('late failed to close')"]
    closing = _start(_Closing, tmp_path / "closing", hooks=(_Part("late", log, fails="aclose"),))
    assert isinstance(closing.error, RuntimeError) and str(closing.error) == "late failed to close"
    hooks = (_Part("late", log, fails="aclose"), _Part("later", log, fails="aclose"))
    twice = _start(_Closing, tmp_path / "twice", hooks=hooks)
    assert isinstance(twice.error, ExceptionGroup)
    assert [str(error) for error in twice.error.exceptions] == ["later failed to close", "late failed to close"]


def test_the_rl_trainer_stops_its_recovery_pruning_after_steps_that_ended_not_after_a_failure(tmp_path, monkeypatch):
    stopped: list[str] = []

    async def close_after_training(self: Any) -> None:
        stopped.append("pruning stops")

    monkeypatch.setattr(PlatoonTinkerRLTrainer, "_close_after_training", close_after_training)
    assert _start(PlatoonTinkerRLTrainer, tmp_path / "ends").error is None and stopped == ["pruning stops"]
    stopped.clear()
    failed = _start(PlatoonTinkerRLTrainer, tmp_path / "fails", hooks=(_Recorder([], fail=("update", 0)),))
    assert _errors(failed) == ["recorder failed in update"] and stopped == []  # as before: a failed start's teardown


def test_a_start_that_a_config_error_stops_exits_78_also_when_a_part_fails_to_close(tmp_path, monkeypatch):
    monkeypatch.setattr(_Cursor, "log", [])
    attach = {"trainer_options": {"hooks": ["test-step-hooks-cursor"]}}
    assert _start(PlatoonTinkerRLTrainer, tmp_path, settings=attach).error is None

    resumed = _start(PlatoonTinkerRLTrainer, tmp_path, steps=4, hooks=(_Part("late", [], fails="aclose"),))
    assert isinstance(resumed.error, ConfigError)  # the key 'cursor', which nothing claims: exit 78
    assert resumed.error.__notes__ == ["Closing the start failed too: RuntimeError('late failed to close')"]


# --- updates that fail closed ----------------------------------------------------------------------------------------


class _Client:
    """A training client whose results carry the given metrics."""

    def __init__(self, fb_metrics: dict[str, float], optim_metrics: dict[str, Any]) -> None:
        self.fb_metrics, self.optim_metrics = fb_metrics, optim_metrics

    async def forward_backward_async(self, datums: Any, loss_fn: str, loss_fn_config: Any) -> Future:
        return Future(SimpleNamespace(loss_fn_outputs=[], metrics=self.fb_metrics))

    async def optim_step_async(self, params: Any) -> Future:
        return Future(SimpleNamespace(metrics=self.optim_metrics))


def _failing_closed(client: _Client) -> Learner:
    info: Any = SimpleNamespace()
    policy = Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), info)
    policy.optimizer_params = tinker.AdamParams()
    policy.training_client = cast(Any, client)

    async def wait(awaitable: Any, *, name: str, total_timeout: float | None = None) -> Any:
        return await awaitable

    timeouts = SimpleNamespace(forward_backward_timeout_seconds=1.0, optim_step_timeout_seconds=1.0)
    tracker: Any = SimpleNamespace(scalar=lambda **metrics: None)
    objective = checks._ProbeObjective()
    return Learner(policy, objective=objective, timeouts=timeouts, wait=wait, tracker=tracker, fail_closed=True)


@pytest.mark.parametrize(
    ("fb_metrics", "optim_metrics", "refused"),
    [
        ({}, {"skipped": 1.0}, "skipped the optimizer step"),
        ({}, {"grad_norm": math.nan}, "optim_step returned grad_norm = nan"),
        ({"loss:sum": math.inf}, {}, "forward_backward returned loss:sum = inf"),
    ],
)
def test_an_update_that_fails_closed_refuses_a_skipped_or_nonfinite_result_before_it_counts(
    fb_metrics, optim_metrics, refused
):
    learner = _failing_closed(_Client(fb_metrics, optim_metrics))
    tally = SimpleNamespace(optimizer_steps=0)

    async def update() -> None:
        sent = learner.update()
        await sent.submit([checks._probe_datum()], [checks._probe_datum()], minibatch=0, microbatch=0)
        await sent.apply(tally)

    with pytest.raises(UpdateRejected, match=refused):
        asyncio.run(update())
    assert tally.optimizer_steps == 0 and learner.policy.optimizer_steps == 0


def test_a_trainer_s_own_learners_fail_closed_and_the_rl_trainer_s_do_not(tmp_path, monkeypatch):
    learners: list[Learner] = []

    class _Learning(_Closing):
        def check(self, run: Any) -> None:
            pass

        async def run_steps(self, run: Any) -> None:
            learners.append(run.learner(MAIN))
            await super().run_steps(run)

    monkeypatch.setattr(_Closing, "log", [])
    assert _start(_Learning, tmp_path).error is None
    assert learners[0].fail_closed and learners[0].clock is not None
    assert not PlatoonTinkerRLTrainer.needs(None, None).fail_closed_updates


class _ThreeUpdates(_Closing):
    """A trainer of its own whose every step takes three updates through run.learner, then ends with run.end_step."""

    def check(self, run: Any) -> None:
        pass

    async def run_steps(self, run: Any) -> None:
        learner = run.learner(MAIN, objective=checks._ProbeObjective())
        while run.clock.step < run.plan.end_step:
            for _ in range(3):
                update = learner.update()
                await update.submit([checks._probe_datum()], [], minibatch=0, microbatch=0)
                await update.apply()
            await run.end_step()


def test_a_trainer_s_own_step_ends_with_the_updates_its_policies_took_published_once(tmp_path, monkeypatch):
    monkeypatch.setattr(_Closing, "log", [])
    service = checks.FakePolicyService()
    start = _start(_ThreeUpdates, tmp_path, service=service)

    assert start.error is None and service.calls.count("save_weights_and_get_sampling_client") == 2
    assert [(row["name"], row["policy_version"], row["optimizer_steps"]) for row in start.rows] == [
        ("recovery-000000", 0, 0),
        ("final", 2, 6),  # a version a step, counted from the policy's optimizer steps
    ]


def test_a_trainer_s_learner_is_built_once_the_run_s_clock_started(tmp_path, monkeypatch):
    class _Early(_ThreeUpdates):
        async def prepare(self, run: Any, start: Any) -> px.StepPlan:
            run.learner(MAIN)  # before ready(): its submissions would open no step (rule 10)
            return await super().prepare(run, start)

    monkeypatch.setattr(_Closing, "log", [])
    start = _start(_Early, tmp_path)
    assert isinstance(start.error, RuntimeError) and "build it once the clock started" in str(start.error)
    assert start.rows == []  # stopped before its launch save


# --- the policy's snapshot -------------------------------------------------------------------------------------------


def test_a_policy_s_snapshot_refuses_a_sample_once_the_policy_published_again():
    llm = checks._FakeLLM("fake-model")
    llm.context_window_length = 4096
    info: Any = SimpleNamespace(llm=llm)
    policy = Policy(PolicySpec(MAIN, model_name="fake-model", renderer_name="r"), info)
    sampled: list[Any] = []

    async def sample_async(*, prompt: Any, num_samples: int, sampling_params: Any) -> str:
        sampled.append(prompt)
        if prompt == "moves":
            llm.update_sampling_client("newer")  # the policy publishes while the call runs
        return "response"

    policy.sampling_client = cast(Any, SimpleNamespace(sample_async=sample_async))
    snapshot = policy.snapshot()
    assert (snapshot.version, snapshot.current(), snapshot.model_name) == (0, True, "fake-model")
    assert asyncio.run(snapshot.sample("prompt", None)) == "response"
    with pytest.raises(px.PolicyMoved, match="published a new version while it ran"):
        asyncio.run(snapshot.sample("moves", None))
    with pytest.raises(px.PolicyMoved, match="before the call"):
        asyncio.run(snapshot.sample("prompt", None))
    assert sampled == ["prompt", "moves"] and not snapshot.current()
    assert policy.snapshot().gateway(context_window_length=1024).context_window_length == 1024
    assert policy.snapshot().gateway(context_window_length=8192).context_window_length == 4096


# --- a warm start with the optimizer's state -------------------------------------------------------------------------


def test_a_new_run_from_a_checkpoint_takes_its_optimizer_state_when_the_config_says_so(tmp_path):
    warm = {"checkpoint": {"load_checkpoint_path": "tinker://other/weights/000010", "load_optimizer_state": True}}
    service = checks.FakePolicyService()
    start = _start(PlatoonTinkerRLTrainer, tmp_path, settings=warm, service=service)

    assert start.error is None
    assert service.calls[0] == "create_training_client_from_state_with_optimizer tinker://other/weights/000010"
    assert start.rows[0]["batch"] == start.rows[0]["optimizer_steps"] == 0  # a new run: its counters start at 0

    weights = checks.FakePolicyService()
    only = {"checkpoint": {"load_checkpoint_path": "tinker://x"}}
    assert _start(PlatoonTinkerRLTrainer, tmp_path / "weights", settings=only, service=weights).error is None
    assert weights.calls[0] == "create_training_client_from_state tinker://x"

    alone = {**checks._settings(tmp_path, 2), "checkpoint": {"load_optimizer_state": True}}
    with pytest.raises(ValueError, match="load_optimizer_state takes the optimizer state of checkpoint.load_check"):
        build_config(PlatoonTinkerRLTrainerConfig, alone)


# --- the draws' provenance -------------------------------------------------------------------------------------------


class _Provenance(PlatoonTinkerRLTrainer):
    """The rl trainer, its training episodes told their draw."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        return dataclasses.replace(super().needs(config, options), draw_provenance=True)


class _Rows(checks._OneDatumGroups):
    """One-datum groups that keep each row's draw, as a workflow passes it on to its episodes."""

    def __init__(self) -> None:
        self.draws: list[Any] = []

    async def arun_episode(self, data: dict) -> list[Any]:
        self.draws.append(data.get(DRAW_KEY))
        return await super().arun_episode(data)


class _Consumed(px.StepHookBase):
    """Keeps the draws each step's end names."""

    name = "consumed"

    def __init__(self) -> None:
        self.steps: list[dict[Any, str]] = []

    async def update(self, end: px.StepEnd) -> None:
        self.steps.append(dict(end.consumed))


def test_training_episodes_see_their_draw_when_the_trainer_asks_and_a_step_s_end_names_those_it_committed(tmp_path):
    rows, consumed = _Rows(), _Consumed()
    start = _start(_Provenance, tmp_path, hooks=(consumed,), workflow=rows)

    assert start.error is None
    run_dir = str(Path(tmp_path, "logs", "platoon_tinker", "run"))
    assert rows.draws and all(isinstance(draw, DrawInfo) and draw.run_dir == run_dir for draw in rows.draws)
    by_key = {draw.key: draw for draw in rows.draws}
    assert [len(step) for step in consumed.steps] == [2, 2] and all(
        key in by_key and fate == "consumed" for step in consumed.steps for key, fate in step.items()
    )
    assert not set(consumed.steps[0]) & set(consumed.steps[1])  # each draw is committed once

    plain = _Rows()
    assert _start(PlatoonTinkerRLTrainer, tmp_path / "plain", workflow=plain).error is None
    assert plain.draws and set(plain.draws) == {None}  # off by default: the rows are as they always were


def test_a_draw_travels_to_its_episode_in_the_rollout_config_and_an_episode_without_one_has_none(tmp_path):
    model_info: Any = SimpleNamespace(llm=SimpleNamespace(version=0), model_name="", renderer_name="")
    workflow = GroupRolloutWorkflow(
        rollout_fn=lambda task, config: {},
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=WorkflowConfig(group_size=1, rollout_config=RolloutConfig(output_dir=str(tmp_path))),
        model_info=model_info,
    )
    draw = DrawInfo(run_dir="/runs/r", start=2, seq=7, position=12, step=3)
    request = workflow._build_worker_request({"task_id": "t", DRAW_KEY: draw}, 0)
    assert request.rollout_config["extra"][DRAW_KEY] == dataclasses.asdict(draw)
    assert DRAW_KEY not in workflow._build_worker_request({"task_id": "t"}, 0).rollout_config["extra"]

    def episode(extra: dict[str, Any]) -> Episode:
        unused: Any = None
        task = Task(goal="g", id="t")
        return Episode(
            task=task, session=unused, config=RolloutConfig(extra=extra), route=unused, runtime=unused, root=unused
        )

    assert episode({DRAW_KEY: dataclasses.asdict(draw)}).draw == draw and draw.key == (2, 7)
    assert episode({}).draw is None


# --- the hooks the rl trainer attaches from its options --------------------------------------------------------------


def test_the_rl_trainer_s_options_name_its_hooks_and_are_checked_when_the_config_loads(tmp_path):
    options = RLOptions.parse({"hooks": ["pkg.mod:Hook", {"hook": "pkg.mod:Other", "options": {"every": 2}}]})
    assert [(spec.ref, dict(spec.options)) for spec in options.hooks] == [
        ("pkg.mod:Hook", {}),
        ("pkg.mod:Other", {"every": 2}),
    ]
    config = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(tmp_path, 2))
    assert trainer_options(config.trainer_options) == trainer_options(None) == RLOptions()
    for wrong, problem in [
        ({"hooks": [3]}, "hooks[0]: expected a step hook"),
        ({"hooks": "pkg.mod:Hook"}, "hooks: expected a list of step hooks"),
        ({"hooks": [{"hook": "pkg.mod:Hook", "every": 2}]}, "hooks[0]: expected a step hook"),
        ({"batch_size": 4}, "the rl trainer takes hooks, not batch_size"),
    ]:
        settings = {**checks._settings(tmp_path, 2), "trainer_options": wrong}
        with pytest.raises(ConfigError) as refused:
            build_config(PlatoonTinkerRLTrainerConfig, settings)
        assert problem in str(refused.value)


class _Hasty(px.StepHookBase):
    """A hook that is not critical, yet updates: the run refuses it."""

    name = "hasty"
    critical = False

    async def update(self, end: px.StepEnd) -> None: ...


class _Bare:
    """A class of the StepHook protocol without from_run."""

    name = "bare"


register_component("step_hook", "test-step-hooks-hasty", _Hasty, exist_ok=True)
register_component("step_hook", "test-step-hooks-bare", _Bare, exist_ok=True)


def test_a_hook_the_config_names_that_does_not_resolve_build_or_register_stops_the_start_before_the_service_opens(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(_Cursor, "log", [])
    (tmp_path / "platoon_test_broken_hooks.py").write_text('raise RuntimeError("broken as it imports")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    for hooks, problem in [
        (["no.such.module:Hook"], "trainer_options.hooks[0]: 'no.such.module:Hook' names no step hook"),
        (
            ["platoon_test_broken_hooks:Hook"],
            "trainer_options.hooks[0]: 'platoon_test_broken_hooks:Hook' names no step hook (RuntimeError: broken as",
        ),
        (
            [{"hook": "platoon.experimental:StepHookBase", "options": {"every": 2}}],
            "trainer_options.hooks[0]: platoon.experimental:StepHookBase cannot be built (ValueError: StepHookBase",
        ),
        (
            [{"hook": "test-step-hooks-interrupts", "options": {}}],
            "trainer_options.hooks[0]: test-step-hooks-interrupts cannot be built (KeyError: 'method')",
        ),
        (["test-step-hooks-bare"], "trainer_options.hooks[0]: test-step-hooks-bare has no from_run(run, options)"),
        (["test-step-hooks-hasty"], "trainer_options.hooks[0]: step hook 'hasty' is not critical"),
        (
            ["test-step-hooks-cursor", "test-step-hooks-cursor"],
            "trainer_options.hooks[1]: two step hooks are named 'cursor'",
        ),
    ]:
        service = checks.FakePolicyService()
        settings = {"trainer_options": {"hooks": hooks}}
        start = _start(PlatoonTinkerRLTrainer, tmp_path, settings=settings, service=service)
        assert isinstance(start.error, ConfigError) and problem in str(start.error), start.error
        assert service.calls == [] and start.rows == []


@dataclass(frozen=True)
class _InterleavedOptions:
    """A subclass's options of its own, which name the hooks the rl trainer attaches beside its own settings."""

    notes_every: int = 1
    hooks: tuple[HookSpec, ...] = ()


def test_the_rl_trainer_reads_its_hooks_from_the_options_it_was_built_with_whatever_their_form(tmp_path, monkeypatch):
    monkeypatch.setattr(_Cursor, "log", [])
    for name, options in [
        ("mapping", {"hooks": ["test-step-hooks-cursor"]}),
        ("parsed", RLOptions(hooks=(HookSpec("test-step-hooks-cursor"),))),
        ("own", _InterleavedOptions(hooks=(HookSpec("test-step-hooks-cursor"),))),
    ]:
        start = _start(PlatoonTinkerRLTrainer, tmp_path / name, options=options)
        assert start.error is None and start.rows[-1]["cursor"] == {"seen": 2}, name

    wrong = _start(PlatoonTinkerRLTrainer, tmp_path / "wrong", options={"hook": []})
    assert isinstance(wrong.error, ConfigError) and "the rl trainer takes hooks, not hook" in str(wrong.error)
    check_trainer(PlatoonTinkerRLTrainer, options={"hooks": []})
    check_trainer(_NotesAndRewrites, options=_InterleavedOptions())


# --- the conformance check, and the outside interleaved trainer it serves first --------------------------------------


class _NotesAndRewrites(PlatoonTinkerRLTrainer):
    """An interleaved trainer as an outside package writes it on this API: the rl trainer's step, unchanged, plus two
    hooks that are participants, registered in check(): note NTP and strategy rewrites, each one cross-entropy update
    at every step's end on the same learner, their producers paused before the step publishes and following the new
    sampler after; its steps atomic, its updates failing closed, and its episodes told their draw."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> px.TrainerNeeds:
        needs = super().needs(config, options)
        return dataclasses.replace(needs, atomic_steps=True, fail_closed_updates=True, draw_provenance=True)

    def check(self, run: Any) -> None:
        super().check(run)
        policy = run.policies[run.policies.leader]
        for phase in (_Phase("note_ntp", policy), _Phase("strategy_rewrite", policy)):
            run.hooks.register(phase)  # and with the run's checkpoints: it has keys


class _Phase(px.StepHookBase):
    """One objective's phase: a producer that samples a snapshot of the policy, paused in drain() and following the
    new sampler in published(); one cross-entropy update at each step's end; its count resumed before the client
    opens, and refused when it is not the model's step."""

    def __init__(self, name: str, policy: Any) -> None:
        self.name, self.keys, self.policy = name, (name,), policy
        self.updates, self.snapshot = 0, None

    def resume(self, start: Any) -> None:
        if start.record is not None:
            self.updates = start.loop_state[self.name]["updates"]

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {self.name: {"updates": self.updates}}

    async def drain(self, end: px.StepEnd) -> None:
        assert self.snapshot is not None and self.snapshot.current()  # nothing published since it followed

    async def update(self, end: px.StepEnd) -> None:
        update = end.learner(self.policy.id, objective=checks._ProbeObjective()).update()
        datum = checks._probe_datum()
        await update.submit([datum], [datum], minibatch=0, microbatch=0)
        await update.apply()
        self.updates += 1

    async def published(self, end: px.StepEnd) -> None:
        self.snapshot = self.policy.snapshot()


def test_the_conformance_check_runs_step_hooks_and_interrupts_inside_a_step():
    check_trainer(PlatoonTinkerRLTrainer)  # its steps are not atomic: a stop inside one lets it end first
    check_trainer(_Atomic)
    check_trainer(_NotesAndRewrites)


def test_the_conformance_check_names_an_atomic_trainer_whose_own_updates_never_open_its_step(monkeypatch):
    class _Unopened(_Closing):
        """Each step one optimizer step sent by hand, outside a learner of the run's, which never says so."""

        def check(self, run: Any) -> None:
            pass

        async def run_steps(self, run: Any) -> None:
            policy = run.policies[run.policies.leader]
            while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
                await (await policy.training_client.optim_step_async(None)).result_async()
                policy.optimizer_steps += 1
                await run.end_step()

    monkeypatch.setattr(_Closing, "log", [])
    with pytest.raises(AssertionError) as failed:
        check_trainer(_Unopened)
    assert str(failed.value).splitlines()[1:] == [
        "  keeps steps whole: its steps are atomic, but a stop as step 1 drained its hooks, after the step's own 1 "
        "update(s), saved ['recovery-000000', '000001_interrupted'], not the launch checkpoint alone: a step is open "
        "once it sends training work (run.learner's open it; work sent by hand sets run.clock.step_open)"
    ]


def test_the_conformance_check_names_a_trainer_whose_steps_never_end_with_the_run(monkeypatch):
    class _ByHand(_Closing):
        def check(self, run: Any) -> None:
            pass

        async def run_steps(self, run: Any) -> None:
            while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
                run.clock.advance()

    monkeypatch.setattr(_Closing, "log", [])
    with pytest.raises(AssertionError, match="runs step hooks: its steps never ran the step hooks"):
        check_trainer(_ByHand)
