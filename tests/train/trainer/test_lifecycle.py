"""The lifecycle every trainer goes through (RFC 0021, ``TrainerRun``): its phases in order, the run's full checkpoints
and their participants, the step clock, the ``trainer:`` key, and the conformance check of a trainer
(``platoon.testing.check_trainer``), which the rl trainer passes."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from fakes.config import make_config
from fakes.trainer import fake_policies

import platoon.experimental as px
from platoon.testing import check_trainer
from platoon.testing import trainers as checks
from platoon.train.checkpoints.participants import COUNTER_KEYS, RunCheckpoints
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy import spec as spec_module
from platoon.train.policy.spec import MAIN, policy_specs
from platoon.train.trainer import api, registry
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.train.trainer.run import StepClock, TrainerRun
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.utils.config import ConfigError, build_config


class _Counting(px.TrainerBase):
    """A trainer of its own on TrainerBase's defaults: each step one optimizer step, published as a new version; it
    records each hook as the run calls it (in ``calls``, which a test may share with the fake service's), and stops at
    a step boundary once a stop is asked for, saying it was interrupted."""

    calls: list[str]

    def __init__(self, setup: px.TrainerSetup) -> None:
        super().__init__(setup)
        self.calls = []

    def check(self, run: TrainerRun) -> None:
        self.calls.append("check")
        run.checkpoints.register(_Cursor("early", self.calls))

    async def prepare(self, run: TrainerRun, start: Any) -> px.StepPlan:
        self.calls.append("prepare")
        run.checkpoints.register(_Cursor("late", self.calls))
        return px.StepPlan(end_step=run.config.train.max_training_steps)

    async def started(self, run: TrainerRun) -> None:
        self.calls.append(f"started {sorted(run.launch_save)}")

    async def ready(self, run: TrainerRun) -> None:
        self.calls.append("ready")
        await super().ready(run)

    async def run_steps(self, run: TrainerRun) -> None:
        self.calls.append("run_steps")
        while run.clock.step < run.plan.end_step:
            if run.shutdown.is_set():
                run.interrupted = True
                return
            await _step(run)

    async def close(self, run: TrainerRun) -> None:
        self.calls.append("close")


async def _step(run: TrainerRun) -> None:
    """One step of the leader policy: an optimizer step on the fake service, published as a new version."""

    policy = run.policies[run.policies.leader]
    client: Any = policy.training_client  # the fake service's
    await (await client.optim_step_async(None)).result_async()
    policy.optimizer_steps += 1
    policy.info.llm.update_sampling_client(await policy.save_for_sampler())
    run.clock.advance()


class _AsS4Sketches(_Counting):
    """Its steps stop on a stop request without saying they were interrupted, as the loops of RFC 0021's S4 sketches
    do."""

    async def run_steps(self, run: TrainerRun) -> None:
        while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
            await _step(run)


class _Unstoppable(_Counting):
    """It never hears a stop request."""

    async def run_steps(self, run: TrainerRun) -> None:
        run.shutdown = asyncio.Event()
        await super().run_steps(run)


class _Cursor:
    """A participant of the run's checkpoints that records when it takes up the start point."""

    def __init__(self, name: str, calls: list[str]) -> None:
        self.name, self.calls = name, calls
        self.keys: tuple[str, ...] = (name,)

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {self.name: {"checkpoint": name}}

    def resume(self, start: Any) -> None:
        taken = f" from {start.record.name}, which saved {start.loop_state[self.name]}" if start.record else ""
        self.calls.append(f"resume {self.name} at {start.step}{taken}")


def _rows(root: Path) -> list[dict[str, Any]]:
    path = root / "logs" / "platoon_tinker" / "run" / "checkpoints.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _counted(root: Path, service: checks.FakePolicyService, steps: int = 2) -> list[str]:
    """One start of a counting trainer on ``service``: the trainer's hooks and what the service was asked, in one log,
    in order."""

    config = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(root, steps))
    setup = px.TrainerSetup(config=config, policy_specs=policy_specs(config))
    with checks._faked(service), checks._restored_signals():
        trainer = _Counting.from_setup(setup)
        trainer.calls = service.calls
        asyncio.run(checks._run(trainer))
    return service.calls


def test_a_trainer_on_the_base_runs_through_the_phases_in_order_and_saves_at_launch_and_at_its_end(tmp_path):
    assert _counted(tmp_path, checks.FakePolicyService()) == [
        "check",
        "resume early at 0",  # a participant registered in check() takes up the start point before any client opens
        "create_lora_training_client fake-model",
        "prepare",
        "resume late at 0",  # one registered later takes it up as it registers
        "save_state recovery-000000",
        "started ['main']",
        "ready",
        "run_steps",
        *["optim_step", "save_weights_and_get_sampling_client"] * 2,
        "save_state final",  # the save at exit, before the trainer closes
        "close",
    ]
    rows = _rows(tmp_path)
    assert [(row["name"], row["batch"], row["policy_version"], row["optimizer_steps"]) for row in rows] == [
        ("recovery-000000", 0, 0, 0),
        ("final", 2, 2, 2),
    ]
    # The run's counters first, then each participant's entries, in the order they registered.
    assert [key for key in rows[-1] if key in (*COUNTER_KEYS, "early", "late")] == [*COUNTER_KEYS, "early", "late"]
    assert rows[-1]["late"] == {"checkpoint": "final"}


def test_participants_take_up_the_row_a_start_resumes_from_by_its_step_and_its_record(tmp_path):
    _counted(tmp_path, checks.FakePolicyService(stop_at=3), steps=4)
    resumed = _counted(tmp_path, checks.FakePolicyService(), steps=4)

    taken = "from 000003_interrupted, which saved {'checkpoint': '000003_interrupted'}"
    assert resumed[:5] == [
        "check",
        f"resume early at 3 {taken}",
        "create_training_client_from_state_with_optimizer tinker://fake/weights/000003_interrupted",
        "prepare",
        f"resume late at 3 {taken}",
    ]


def test_the_conformance_check_passes_a_trainer_that_keeps_the_lifecycle_and_names_what_one_breaks():
    check_trainer(_Counting)
    with pytest.raises(
        AssertionError, match=r"fails 1 check\(s\):\n  interrupts: saved \['recovery-000000', 'final'\]"
    ):
        check_trainer(_Unstoppable)


def test_steps_that_a_stop_request_stopped_short_of_their_end_save_an_interrupted_checkpoint_not_the_final_one():
    check_trainer(_AsS4Sketches)  # trains to its end, stops at a step boundary and saves <step>_interrupted, resumes


def test_the_rl_trainer_keeps_the_lifecycle():
    check_trainer(PlatoonTinkerRLTrainer)


class _RLWithCursor(PlatoonTinkerRLTrainer):
    """An rl trainer whose check() replaces the rl trainer's (without super()) and registers a participant; it records
    where the run's clock stands once the trainer is ready."""

    clocks: ClassVar[list[tuple[int, int | None]]] = []

    def check(self, run: TrainerRun) -> None:
        self._require_entered()
        run.checkpoints.register(_Cursor("cursor", []))

    async def ready(self, run: TrainerRun) -> None:
        await super().ready(run)
        self.clocks.append((run.clock.step, run.clock.end))


def test_the_rl_trainer_s_steps_save_through_the_start_it_runs_whatever_a_subclass_s_hooks_do(tmp_path, monkeypatch):
    settings = checks._settings
    monkeypatch.setattr(checks, "_settings", lambda root, steps: {**settings(root, steps), "checkpoint": {"every": 1}})
    monkeypatch.setattr(_RLWithCursor, "clocks", [])
    subclass: Any = _RLWithCursor
    with checks._restored_signals():
        start = checks._start(subclass, None, tmp_path, steps=2)

    assert start.error is None
    # Every full checkpoint holds the participant, those of the cadence too.
    assert [(row["name"], row.get("cursor")) for row in start.rows] == [
        (name, {"checkpoint": name}) for name in ("recovery-000000", "000001", "000002", "final")
    ]
    assert _RLWithCursor.clocks == [(0, 2)]  # the run's clock, from the start's step to the plan's end


def test_a_start_s_steps_never_save_outside_the_start_that_train_runs():
    config = make_config()
    trainer: Any = PlatoonTinkerRLTrainer(config, [], policies=fake_policies(config))
    trainer._rollouts = object()  # a start's prepare() opened its draws, but train() did not record that start
    with pytest.raises(RuntimeError, match="steps save through the start train"):
        trainer._checkpoints(object())


def test_the_rl_trainer_refuses_a_setup_whose_policies_are_not_its_config_s(tmp_path):
    config = build_config(PlatoonTinkerRLTrainerConfig, checks._settings(tmp_path, 2))
    other = {MAIN: dataclasses.replace(policy_specs(config)[MAIN], model_name="other-model")}
    setup = px.TrainerSetup(config=config, policy_specs=other, tasks=px.TaskSources(train_dataset=[]))
    with checks._faked(checks.FakePolicyService()), pytest.raises(ValueError, match="policies its config declares"):
        PlatoonTinkerRLTrainer.from_setup(setup)


def test_the_run_s_checkpoints_hold_the_counters_then_each_participant_s_entries_in_registration_order(tmp_path):
    saved: list[dict[str, Any]] = []

    class Leader:
        id = "main"

        async def save(self, name: str, *, log_path: str, loop_state: dict[str, Any], wait: Any) -> dict[str, str]:
            saved.append(loop_state)
            return {"state_path": f"tinker://x/{name}", "sampler_path": f"tinker://x/s/{name}"}

    policies: Any = _Policies(main=Leader())
    checkpoints = RunCheckpoints(str(tmp_path), policies, wandb_run_id=lambda: "w")
    calls: list[str] = []
    checkpoints.register(_Cursor("b", calls))
    checkpoints.register(_Cursor("a", calls))
    start: Any = SimpleNamespace(step=3, record=None)
    checkpoints.resume(start)
    checkpoints.register(_Cursor("c", calls))
    unused: Any = None  # the leader above saves without waiting
    paths = asyncio.run(checkpoints.save("000004", step=4, policy_version=5, optimizer_steps=6, wait=unused))

    assert calls == ["resume b at 3", "resume a at 3", "resume c at 3"]
    assert list(saved[0]) == ["batch", "wandb_run_id", "policy_version", "optimizer_steps", "b", "a", "c"]
    assert saved[0]["batch"] == 4 and saved[0]["wandb_run_id"] == "w" and saved[0]["optimizer_steps"] == 6
    assert paths == {"main": {"state_path": "tinker://x/000004", "sampler_path": "tinker://x/s/000004"}}


class _Policies(dict):
    """The run's policies as the checkpoints read them: by id, with the leader's."""

    leader = "main"


def test_the_step_clock_advances_the_step_then_the_horizon_then_the_event_then_the_watchdog():
    order: list[str] = []
    event = asyncio.Event()
    watchdog: Any = SimpleNamespace(heartbeat=lambda: order.append("fed"))
    clock = StepClock(step=3, updated=event, end=5, watchdog=watchdog)
    consumed: Any = SimpleNamespace(step_started=lambda: order.append(f"horizon at step {clock.step}"))

    clock.advance(consumed=consumed)
    assert (clock.step, clock.end, event.is_set(), order) == (4, 5, True, ["horizon at step 4", "fed"])


def test_the_loops_state_reads_and_writes_the_step_and_the_event_on_the_run_s_clock():
    clock = StepClock(step=2, updated=asyncio.Event())
    parts: dict[str, Any] = {
        "config": make_config(),
        "shutdown_event": asyncio.Event(),
        "train_dataloader": SimpleNamespace(),
        "eval_dataloader": None,
        "sampling_client_step": 0,
        "stats_logger": SimpleNamespace(),
        "train_tracker": SimpleNamespace(),
    }
    state = TrainLoopSharedState(**parts, clock=clock, model_info=SimpleNamespace())
    state.train_step += 1
    assert (clock.step, state.sampling_client_updated_event) == (3, clock.updated)
    with pytest.raises(TypeError, match="clock or the loops' step and event, not both: given clock and train_step"):
        TrainLoopSharedState(**parts, clock=clock, train_step=0)

    # Made as platoon 0.4.0 made it: a clock of its own, over the step and the event, which feeds the state's watchdog.
    event = asyncio.Event()
    watchdog: Any = SimpleNamespace()
    own: Any = TrainLoopSharedState(
        **parts, train_step=5, sampling_client_updated_event=event, watchdog=watchdog, model_info=SimpleNamespace()
    )
    assert (own.clock.step, own.clock.updated is event, own.clock.watchdog is watchdog) == (5, True, True)


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("rl", None),
        ("sft", "trainer: 'sft' is reserved for a trainer platoon will add; this release trains with rl (the default)"),
        ("acme-dpo", "trainer: 'acme-dpo' is not a trainer: this release has one, rl (the default)"),
    ],
)
def test_the_trainer_key_takes_rl_and_refuses_any_other_name_when_the_config_loads(tmp_path, name, problem):
    settings = {**checks._settings(tmp_path, 2), "trainer": name}
    if problem is None:
        assert build_config(PlatoonTinkerRLTrainerConfig, settings).trainer == "rl"
        return
    assert registry.trainer_problems(SimpleNamespace(trainer=name)) == [problem]
    with pytest.raises(ConfigError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        build_config(PlatoonTinkerRLTrainerConfig, settings)


def test_the_experimental_api_exports_the_trainer_api_and_the_policy_spec_as_they_are():
    assert (px.TrainerBase, px.TrainerSetup, px.StepPlan, px.TrainerNeeds) == (
        api.TrainerBase,
        api.TrainerSetup,
        api.StepPlan,
        api.TrainerNeeds,
    )
    assert px.PolicySpec is spec_module.PolicySpec and px.MAIN is spec_module.MAIN
    assert issubclass(PlatoonTinkerRLTrainer, px.TrainerBase)
    assert PlatoonTinkerRLTrainer.needs(None, None) == px.TrainerNeeds(rollouts=True)
