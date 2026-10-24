"""``pass_rate`` in the trainer's own loop (Tinker faked), with the real group workflow, its rollouts faked at the
worker boundary (RFC 0019, section 5): one environment of 40 tasks with known pass rates (25 nearly always solved, 15
spread from 0.1 to 0.9), where each rollout passes when a hash of its task, its draw's position and its rollout number
falls below its task's rate, so a draw drawn again gets the outcomes it had, and no two draws share theirs.  Each group
waits a time drawn from a seed of the test's before it runs, which chooses the order the groups complete in.

- ``pass_rate`` trains more informative groups than ``uniform``, draws every task, and concentrates no more than its
  window's cap allows;
- at ``max_staleness: 0`` with permits for the whole batch, two runs whose groups complete in different orders make
  the same decisions (task, reason, position, and score bit for bit), and a run resumed at a step goes on with the
  decisions of the run that was not interrupted;
- a resume under a changed sampler warns, records it, and goes on: a new sampler starts afresh on the statistics the
  run has, an entry's waiting slots are bound by its static sampler now, or dropped (each logged) when it no longer
  gets slots, changed options keep the state, the same options written out change nothing, and a state that does not
  load starts the sampler afresh;
- a run with ``pass_rate`` on a capped entry, interrupted twice, consumes every position once;
- with its statistics stopped, ``pass_rate`` explores only, and says so once.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import math
import random
import signal
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
import yaml
from tinker_service_fakes import FakeTinkerService, make_pruning_trainer
from trainer_fakes import TrainingClient, make_trainer

from platoon.experimental import Draw, Sampler
from platoon.task import Task
from platoon.train import entry_settings
from platoon.train.auto import AutoDataset
from platoon.train.task_stats import TaskStatsStore, read_events
from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.checkpoint_pruning import read_checkpoint_records
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.dispatcher import POSITION_COLUMN
from platoon.train.tinker.task_records import CHANGED_KEY
from platoon.train.tinker.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.train.tinker.workflows import group_rollout_workflow as workflow_module
from platoon.utils.config import resolve_config
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

HERE = __name__
G = 8
# The pass rates of the 40 tasks: 25 at 0.99, 15 evenly from 0.1 to 0.9.
RATES = [0.99] * 25 + [0.1 + 0.8 * index / 14 for index in range(15)]
# The position of the draw whose group runs, which its rollouts inherit.
POSITION: contextvars.ContextVar[int] = contextvars.ContextVar("position")


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def tasks(config: Any, split: str, *, count: int = 40) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}"} for index in range(count)]


def rate(task_id: str) -> float:
    return RATES[int(task_id.split("::")[-1][1:]) % len(RATES)]


def passes(task_id: str, position: int, rollout: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{position}:{rollout}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate(task_id)


def _entry(name: str | None = None, count: int = 40, **fields: Any) -> dict[str, Any]:
    entry = {"dataset_loader": f"{HERE}:tasks", "dataset_kwargs": {"count": count}, "rollout": "x:y", **fields}
    if name:
        entry["name"] = name
    return entry


def _config(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]], **extra: Any) -> Any:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "model_name": "m",
                    "renderer_name": "r",
                    "batch_size": 8,
                    "max_training_steps": 30,
                    "max_staleness": 0,
                    "max_concurrent_rollouts": 64,
                    "workflow_config": {"group_size": G},
                    **train,
                },
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": environments,
                **extra,
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def _datum(position: int, rollout: int, reward: float, version: int) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, rollout, 1]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, reward, reward],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


def _fake_rollouts(monkeypatch: pytest.MonkeyPatch, policy: Any) -> None:
    """Each rollout passes by the hash of its task, its draw's position and its rollout number."""

    async def run_rollout_request(request, llm, *, rollout_fn=None, reward_processor=None):
        position = POSITION.get()
        success = 1.0 if passes(request.task_id, position, request.rollout_number) else 0.0
        stats = TrajectoryStats(
            trajectory_id="root",
            reward=success,
            num_steps=2,
            num_input_tokens=3,
            num_output_tokens=4,
            num_datums=1,
            is_root=True,
        )
        result = TrajectoryCollectionResult(
            datums=[_datum(position, request.rollout_number, success, policy.version)],
            task_reward=success,
            trajectory_stats=[stats],
            root_rewards_dict={},
            task_reward_valid=True,
        )
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status="completed",
            result=result,
            workload=RolloutWorkload(
                trajectories=1, postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1
            ),
            observed=True,
            outcome_status="scored",
            success_reward=success,
            rollout_seconds=0.1,
        )

    monkeypatch.setattr(workflow_module, "run_rollout_request", run_rollout_request)


class _Workflow(GroupRolloutWorkflow):
    """The real group workflow; each group first waits a time drawn from ``seed`` and its position, and puts its
    position where its rollouts read it.  ``completed`` holds the positions in the order their groups completed;
    ``on_start(position, started)`` runs as each group starts."""

    def __init__(self, config: Any, *, seed: int = 0, on_start: Any = None) -> None:
        super().__init__(
            rollout_fn=lambda task, rollout_config: None,
            get_task_fn=lambda task_id: Task(id=task_id),
            config=config.train.workflow_config,
            model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0, model_name="m")),
        )
        self.entry_configs = {label: own.train for label, own in entry_settings.entry_settings(config).items()}
        self.seed = seed
        self.on_start = on_start
        self.started = 0
        self.completed: list[int] = []

    async def arun_episode(self, data):
        position = data[POSITION_COLUMN]
        POSITION.set(position)
        self.started += 1
        if self.on_start is not None:
            self.on_start(position, self.started)
        await asyncio.sleep(random.Random(f"{self.seed}:{position}").uniform(0.0, 0.01))
        output = await super().arun_episode(data)
        self.completed.append(position)
        return output


class _Recording(TrainingClient):
    """Records each step's datums: their tokens and advantages."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[list[tuple[tuple[int, ...], tuple[float, ...]]]] = []

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.steps.append(
            [
                (tuple(datum.model_input.to_ints()), tuple(datum.loss_fn_inputs["advantages"].to_torch().tolist()))
                for datum in datums
            ]
        )
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def _built(trainer: Any, config: Any) -> Any:
    training = AutoDataset.training(config)
    trainer.train_dataset, trainer.train_order, trainer.train_sampling = (
        training.dataset,
        training.order,
        training.sampling,
    )
    return training


def _run_once(tmp_path: Path, monkeypatch, train: dict[str, Any], environments: list[dict[str, Any]], *, seed: int = 0):
    """One run on fakes: ``(trainer, each step's logged stats, the workflow, the training client)``."""

    config = _config(tmp_path, train, environments)
    client = _Recording()
    trainer, _saves, _logged = make_trainer(monkeypatch, config, training_client=client)
    logged: list[dict[str, float]] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(dict(stats))
    )
    _built(trainer, config)
    _fake_rollouts(monkeypatch, trainer._model_info.llm)
    workflow = _Workflow(config, seed=seed)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 120))
    return trainer, logged, workflow, client


def _draws(run_dir: str | Path) -> list[dict[str, Any]]:
    return [event for event in read_events(run_dir) if event["event"] == "draw"]


# --- pass_rate trains more informative groups ------------------------------------------------------------------


def test_pass_rate_raises_the_informative_share_over_uniform(tmp_path, monkeypatch):
    shares, runs = {}, {}
    for sampler in ("uniform", "pass_rate"):
        environment = _entry("pool", task_sampler=sampler)
        trainer, logged, _workflow, _client = _run_once(
            tmp_path / sampler, monkeypatch, {"task_sampler": "weighted"}, [environment]
        )
        assert len(logged) == 30
        informative = sum(step["train/batch/informative_groups"] for step in logged)
        shares[sampler] = informative / (30 * 8)
        runs[sampler] = (trainer, logged)
    print(f"informative share over the run's 240 groups: {shares}")
    assert shares["pass_rate"] >= shares["uniform"] + 0.15, shares
    assert shares["pass_rate"] >= 1.3 * shares["uniform"], shares
    trainer, logged = runs["pass_rate"]
    draws = _draws(trainer.run_log_path)
    counts = Counter(event["task_id"] for event in draws)
    assert len(counts) == 40  # every task drawn
    top = sorted(counts.values(), reverse=True)[:4]
    assert sum(top) / len(draws) <= 0.35
    assert {event["reason"] for event in draws} <= {"explore", "exploit"}
    assert all(event["sampler"] == "pass_rate@pool" for event in draws)
    prefix = "train/sampler/environment/pool/"
    assert all(step.get(f"{prefix}draws/fallback", 0) == 0 for step in logged)
    last = logged[-1]
    assert last[f"{prefix}top10_share"] <= 0.35 and 0 < last[f"{prefix}eligible_share"] <= 1
    assert 0 <= last[f"{prefix}predicted_informative"] <= 1 and 0 <= last[f"{prefix}informative_share"] <= 1
    assert sum(step[f"{prefix}draws"] for step in logged) == len(draws)
    assert all(step[f"{prefix}decision_ms"] >= 0 for step in logged if f"{prefix}decision_ms" in step)


# --- replays at k = 0 -----------------------------------------------------------------------------------------


REPLAYED = {
    "per-entry": ({"task_sampler": "weighted", "max_training_steps": 10}, [_entry("pool", task_sampler="pass_rate")]),
    "pool-wide": ({"task_sampler": "pass_rate", "max_training_steps": 10}, [_entry()]),
}


def _decisions(draws: list[dict[str, Any]]) -> list[tuple[str, str, int, float | None]]:
    return [(event["task_id"], event["reason"], event["position"], event.get("score")) for event in draws]


@pytest.mark.parametrize("form", sorted(REPLAYED))
def test_a_k0_run_replays_decision_for_decision(tmp_path, monkeypatch, form):
    """Two runs whose groups complete in different orders: the same draws, scores bit for bit, each step's datums the
    same, in the order of completion, and the same statistics at the end, byte for byte (their decayed means of the
    members' times and tokens are fractional sums, whose bits show the order the groups were taken in)."""

    train, environments = REPLAYED[form]
    first = _run_once(tmp_path / "first", monkeypatch, train, environments, seed=1)
    second = _run_once(tmp_path / "second", monkeypatch, train, environments, seed=2)
    assert first[2].completed != second[2].completed  # the groups did complete in other orders
    assert sorted(first[2].completed) == sorted(second[2].completed)
    draws = _decisions(_draws(first[0].run_log_path))
    assert draws == _decisions(_draws(second[0].run_log_path))
    assert {reason for _task, reason, _position, _score in draws} <= {"explore", "exploit"}
    assert all(score is not None and math.isfinite(score) for *_rest, score in draws)
    assert [sorted(step) for step in first[3].steps] == [sorted(step) for step in second[3].steps]
    final = [Path(run[0].run_log_path) / "tasks" / "state-s1-final.json" for run in (first, second)]
    assert final[0].read_bytes() == final[1].read_bytes()


def _start(tmp_path: Path, monkeypatch, train: dict[str, Any], environments: list[dict[str, Any]], service: Any, **kw):
    """One start of a run on a fake Tinker service: ``(trainer, each step's logged stats)``."""

    config = _config(tmp_path, train, environments, checkpoint={"every": 100})
    trainer, logged = make_pruning_trainer(monkeypatch, config, service)
    _built(trainer, config)
    _fake_rollouts(monkeypatch, trainer._model_info.llm)
    workflow = _Workflow(config, **kw)
    lock = Path(trainer.run_log_path) / "lock.json"
    if not lock.exists():
        lock.write_text("{}\n")  # platoon train writes it before the trainer starts
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 120))
    return trainer, logged


def _interrupt_at(count: int):
    def hook(position: int, started: int) -> None:
        if started == count:
            signal.raise_signal(signal.SIGINT)

    return hook


def test_a_k0_run_resumed_at_a_step_makes_the_uninterrupted_run_s_decisions(tmp_path, monkeypatch):
    """Interrupted as its fourth step's groups start, then resumed: every position's decision (not the draws a resume
    drew again) is the uninterrupted run's, tasks, reasons and positions exactly and scores to a relative 1e-12: the
    interrupted step's groups enter the statistics in another order, which rounds float sums differently."""

    train, environments = (
        {"task_sampler": "weighted", "max_training_steps": 8},
        [_entry("pool", task_sampler="pass_rate")],
    )
    whole, _logged = _start(tmp_path / "whole", monkeypatch, train, environments, FakeTinkerService(), seed=3)
    expected = {event["position"]: event for event in _draws(whole.run_log_path)}
    service = FakeTinkerService()
    _start(tmp_path / "parts", monkeypatch, train, environments, service, seed=3, on_start=_interrupt_at(3 * 8 + 1))
    resumed, _logged = _start(tmp_path / "parts", monkeypatch, train, environments, service, seed=4)
    draws = [event for event in _draws(resumed.run_log_path) if event["reason"] != "reissue"]
    assert {event["start"] for event in draws} == {1, 2}
    by_position = {event["position"]: event for event in draws}
    assert sorted(by_position) == sorted(expected) == list(range(64))
    for position, event in by_position.items():
        want = expected[position]
        assert (event["task_id"], event["reason"]) == (want["task_id"], want["reason"]), position
        assert event["score"] == pytest.approx(want["score"], rel=1e-12, abs=0.0)


# --- resumes under a changed sampler --------------------------------------------------------------------------------


class Unloadable(Sampler):
    """An adaptive sampler whose saved state never loads: its load_state_dict raises KeyError."""

    adaptive = True

    def __init__(self, setup, options=None) -> None:
        super().__init__(setup, options)
        self.made = 0

    def next(self, ctx):
        eligible = [task for task in ctx.task_ids if ctx.eligible(task)]
        if not eligible:
            return None
        self.made += 1
        return Draw(eligible[ctx.decision % len(eligible)], "turn")

    def state_dict(self):
        return {"made": self.made}

    def load_state_dict(self, state):
        self.made = state["count"]  # the wrong key: a KeyError


def _capped(sampler: str | None, **options: Any) -> list[dict[str, Any]]:
    hard = _entry("hard", count=12)
    if sampler is not None:
        hard["task_sampler"] = sampler
    if options:
        hard["task_sampler_options"] = options
    return [hard, _entry("easy", count=12)]


CAPPED = {"task_sampler": "weighted", "batch_size": 4, "max_staleness": 2, "max_training_steps": 12}
LIMITED = {**CAPPED, "rollout_group_limits": {"hard": 1}}


def _changed_resume(tmp_path, monkeypatch, caplog, before: tuple[dict, list], after: tuple[dict, list], interrupt: int):
    service = FakeTinkerService()
    first, _ = _start(tmp_path, monkeypatch, *before, service, on_start=_interrupt_at(interrupt))
    run_dir = Path(first.run_log_path)
    rows = read_checkpoint_records(run_dir)
    state = json.loads((run_dir / rows[-1]["dispatch"]["file"]).read_bytes())
    with caplog.at_level(logging.INFO):
        second, logged = _start(tmp_path, monkeypatch, *after, service)
    lock = json.loads((run_dir / "lock.json").read_text())
    return state, second, logged, list(read_events(run_dir)), lock.get(CHANGED_KEY, [])


def test_a_resume_that_switches_an_entry_to_pass_rate_starts_it_afresh_on_the_run_s_statistics(
    tmp_path, monkeypatch, caplog
):
    state, second, _logged, events, changed = _changed_resume(
        tmp_path, monkeypatch, caplog, (CAPPED, _capped(None)), (CAPPED, _capped("pass_rate")), 21
    )
    assert state["samplers"]["entries"]["hard"]["name"] == "uniform@hard"
    assert "environments[hard].task_sampler: uniform -> pass_rate" in caplog.text
    assert any("environments[hard].task_sampler: uniform -> pass_rate" in entry["changes"] for entry in changed)
    made = [
        e
        for e in events
        if e["event"] == "draw" and e["start"] == 2 and e["sampler"] == "pass_rate@hard" and e["reason"] != "reissue"
    ]
    member = second._dispatcher.samplers.entries["hard"]
    assert made and member.decisions == len(made)  # afresh, at decision 0
    # It read the statistics the snapshot carried: its first decision saw groups of hard's tasks.
    assert any(
        score is not None and score != pytest.approx(0.778, abs=5e-4) for score in (e.get("score") for e in made)
    )


def test_a_resume_that_makes_an_adaptive_entry_static_binds_its_waiting_slots_in_order(tmp_path, monkeypatch, caplog):
    state, _second, _logged, events, _changed = _changed_resume(
        tmp_path, monkeypatch, caplog, (LIMITED, _capped("pass_rate")), (LIMITED, _capped(None)), 21
    )
    tokens = state["tokens"].get("hard", [])
    assert tokens, state["tokens"]
    bound = [e for e in events if e["event"] == "draw" and e["start"] == 2 and e["position"] in tokens]
    assert [e["position"] for e in bound] == tokens
    assert all((e["sampler"], e["reason"]) == ("uniform@hard", "order") for e in bound)


def test_a_resume_whose_entry_no_longer_gets_slots_drops_its_waiting_slots_each_logged(tmp_path, monkeypatch, caplog):
    uniform = {key: value for key, value in LIMITED.items() if key != "task_sampler"} | {"task_sampler": "uniform"}
    state, _second, _logged, events, _changed = _changed_resume(
        tmp_path, monkeypatch, caplog, (LIMITED, _capped("pass_rate")), (uniform, _capped(None)), 21
    )
    tokens = state["tokens"].get("hard", [])
    assert tokens
    skipped = [e for e in events if e["event"] == "slot_skipped"]
    assert [(e["position"], e["entry"], e["why"]) for e in skipped] == [
        (position, "hard", "no_slots") for position in tokens
    ]
    assert f"{len(tokens)} slots that waited for an entry's task sampler are dropped" in caplog.text


def test_changed_options_keep_the_state_and_the_same_options_written_out_change_nothing(tmp_path, monkeypatch, caplog):
    state, second, _logged, events, changed = _changed_resume(
        tmp_path / "changed",
        monkeypatch,
        caplog,
        (CAPPED, _capped("pass_rate")),
        (CAPPED, _capped("pass_rate", explore=0.5)),
        21,
    )
    saved = state["samplers"]["entries"]["hard"]["decisions"]
    made = [
        e
        for e in events
        if e["event"] == "draw" and e["start"] == 2 and e["sampler"] == "pass_rate@hard" and e["reason"] != "reissue"
    ]
    assert second._dispatcher.samplers.entries["hard"].decisions == saved + len(made)  # its state loaded
    assert any("environments[hard].task_sampler_options:" in change for entry in changed for change in entry["changes"])
    caplog.clear()
    _state, _second, _logged, _events, unchanged = _changed_resume(
        tmp_path / "same",
        monkeypatch,
        caplog,
        (CAPPED, _capped("pass_rate")),
        (CAPPED, _capped("pass_rate", explore=0.2, max_draws_per_epoch=3)),
        21,
    )
    assert unchanged == [] and "sampling inputs changed" not in caplog.text


def test_a_sampler_state_that_does_not_load_starts_the_sampler_afresh(tmp_path, monkeypatch, caplog):
    environments = _capped(f"{HERE}:Unloadable")
    _state, second, _logged, _events, _changed = _changed_resume(
        tmp_path, monkeypatch, caplog, (CAPPED, environments), (CAPPED, environments), 21
    )
    assert "does not load (KeyError: 'count'): it starts afresh" in caplog.text
    assert second._dispatcher.samplers.entries["hard"].sampler.made >= 1


# --- exactly once across two interrupts, with pass_rate on a capped entry -------------------------------------


def test_a_run_with_pass_rate_on_a_capped_entry_interrupted_twice_consumes_every_position_once(tmp_path, monkeypatch):
    service = FakeTinkerService()
    restored: list[tuple[dict, dict]] = []
    original = rl_module.PlatoonTinkerRLTrainer._restore_dispatch

    def restore(self, dispatcher, resume_info, policy_version):
        original(self, dispatcher, resume_info, policy_version)
        record = resume_info.extra["dispatch"]
        saved = json.loads((Path(self.run_log_path) / record["file"]).read_bytes())
        restored.append((saved["samplers"], json.loads(json.dumps(dispatcher.samplers.state()))))

    monkeypatch.setattr(rl_module.PlatoonTinkerRLTrainer, "_restore_dispatch", restore)
    for interrupt in (17, 17, None):
        trainer, logged = _start(
            tmp_path,
            monkeypatch,
            LIMITED,
            _capped("pass_rate"),
            service,
            on_start=_interrupt_at(interrupt) if interrupt else None,
        )
    run_dir = Path(trainer.run_log_path)
    assert len(restored) == 2
    for saved, after in restored:
        for level in ("run",):
            assert {key: saved[level][key] for key in ("name", "decisions", "state")} == {
                key: after[level][key] for key in ("name", "decisions", "state")
            }
        assert after["entries"]["hard"]["state"] == saved["entries"]["hard"]["state"]
        assert after["entries"]["hard"]["decisions"] == saved["entries"]["hard"]["decisions"]
    events = list(read_events(run_dir))
    draws = {(e["start"], e["seq"]): e for e in events if e["event"] == "draw"}
    settles = [(e["start"], e["seq"]) for e in events if e["event"] == "settle"]
    assert len(settles) == len(set(settles))  # each group settled at most once
    consumed = [draws[(e["start"], e["seq"])]["position"] for e in events if e.get("fate") == "consumed"]
    skipped = [e["position"] for e in events if e["event"] == "slot_skipped"]
    assert sorted(consumed) == sorted(set(consumed))  # trained once each
    assert len(consumed) == 12 * 4
    final = json.loads((run_dir / read_checkpoint_records(run_dir)[-1]["dispatch"]["file"]).read_bytes())
    waiting = [position for positions in final.get("tokens", {}).values() for position in positions]
    outstanding = [position for _start, _seq, _task, position in final["outstanding"]]
    lost = [draws[(e["start"], e["seq"])]["position"] for e in events if e.get("fate") in ("skipped", "lost")]
    assert sorted(consumed + skipped + waiting + outstanding + lost) == list(range(final["next_position"]))
    # The last start, which no interrupt cut short, logged each of its draws by reason.
    by_reason = Counter(e["reason"] for e in draws.values() if e["start"] == 3)
    logged_reasons: Counter[str] = Counter()
    for _step, stats in logged:
        for key, value in stats.items():
            if key.startswith("train/sampler/draws/") and "__" not in key:
                logged_reasons[key.removeprefix("train/sampler/draws/")] += int(value)
    assert logged_reasons == by_reason


# --- without statistics -----------------------------------------------------------------------------------------


def test_with_its_statistics_stopped_pass_rate_explores_and_says_so_once(tmp_path, monkeypatch, caplog):
    def settle(self, record):
        raise RuntimeError("a bug in the statistics")

    monkeypatch.setattr(TaskStatsStore, "settle", settle)
    with caplog.at_level(logging.WARNING):
        trainer, logged, _workflow, _client = _run_once(
            tmp_path,
            monkeypatch,
            {"task_sampler": "weighted", "max_training_steps": 6},
            [_entry("pool", task_sampler="pass_rate")],
        )
    assert len(logged) == 6 and trainer._recorder.stopped
    member = trainer._dispatcher.samplers.entries["pool"]
    assert member.fallbacks == 0
    assert caplog.text.count("statistics are not available") == 1
    # The statistics stopped at the first step's metrics: from the second step on, every decision explored.
    prefix = "train/sampler/environment/pool/draws/"
    assert all(step.get(f"{prefix}exploit", 0) == 0 for step in logged[1:])
    assert sum(step.get(f"{prefix}explore", 0) for step in logged[1:]) == 5 * 8


def test_a_trainer_given_no_sampling_refuses_a_sampler_it_cannot_draw(tmp_path, monkeypatch):
    config = _config(tmp_path, {}, [_entry("a", task_sampler="pass_rate"), _entry("b")])
    trainer, _saves, _logged = make_trainer(monkeypatch, config)
    with pytest.raises(ValueError, match="entry and adaptive samplers need AutoDataset.training's sampling"):
        trainer._samplers(None, "auto")
