"""``informative`` in the trainer's own loop on M4's fake environment (Tinker faked, the real group workflow, rollouts
faked at the worker boundary): 40 tasks, 25 nearly always solved and 15 spread from 0.1 to 0.9, each rollout passing by
a hash of its task, its draw's position and its rollout number (``test_pass_rate_run.py``).  RFC 0019's exit check for
M5 on CPU: ``informative`` raises the informative share of the groups a step trains to 1, at the cost the arithmetic
gives, one over the informative share in draws per admitted group; and ``pass_rate`` with ``informative`` needs fewer
draws per admitted group than ``uniform`` with it."""

from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_pass_rate_run import G, _built, _config, _entry, _fake_rollouts, _Recording, _Workflow, passes
from trainer_fakes import make_trainer

from platoon.train.task_stats import read_events

STEPS = 30
B = 8


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _run(tmp_path: Path, monkeypatch, sampler: str, rule: str, *, seed: int = 0) -> tuple[Any, list[dict[str, Any]]]:
    config = _config(
        tmp_path,
        {"task_sampler": "weighted", "max_training_steps": STEPS, "group_admission": rule},
        [_entry("pool", task_sampler=sampler)],
    )
    trainer, _saves, _logged = make_trainer(monkeypatch, config, training_client=_Recording())
    logged: list[dict[str, Any]] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(stats)
    )
    _built(trainer, config)
    _fake_rollouts(monkeypatch, trainer._model_info.llm)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=_Workflow(config, seed=seed)), 300))
    return trainer, logged


def informative(task_id: str, position: int) -> bool:
    """Whether the group of ``task_id`` at ``position`` trains: its rollouts neither all pass nor all fail."""

    outcomes = {passes(task_id, position, rollout) for rollout in range(G)}
    return len(outcomes) == 2


def test_informative_trains_only_informative_groups_at_one_over_p_rollouts_each(tmp_path, monkeypatch):
    runs = {
        name: _run(tmp_path / name, monkeypatch, sampler, rule)
        for name, (sampler, rule) in {
            "uniform": ("uniform", "all"),
            "uniform+informative": ("uniform", "informative"),
            "pass_rate+informative": ("pass_rate", "informative"),
        }.items()
    }
    # All mode: the informative share of the groups each step trained, as M4's fake run measured it.
    _trainer, plain = runs["uniform"]
    plain_share = sum(stats["train/batch/informative_groups"] for stats in plain) / (STEPS * B)
    for name in ("uniform+informative", "pass_rate+informative"):
        trainer, logged = runs[name]
        assert len(logged) == STEPS
        for stats in logged:  # every group a step trained was informative
            assert stats["train/batch/informative_groups"] == stats["train/admission/groups_admitted"]
            for kind in ("all_pass", "all_fail", "uniform_mixed"):
                assert stats[f"train/batch/{kind}_groups"] == 0
            assert stats["train/admission/errors"] == 0
        assert all(stats.get("train/sampler/environment/pool/draws/fallback", 0) == 0 for stats in logged)
    # The static arm: each step's draws and admitted groups are a walk of the order with the fake's outcomes.
    trainer, logged = runs["uniform+informative"]
    draws = [event for event in read_events(Path(trainer.run_log_path)) if event["event"] == "draw"]
    task = {event["position"]: event["task_id"] for event in draws}
    position = 0
    for stats in logged:
        taken = admitted = 0
        while admitted < B and taken < 3 * B:
            admitted += informative(task[position], position)
            taken += 1
            position += 1
        assert (stats["train/admission/counted"], stats["train/admission/groups_admitted"]) == (taken, admitted)
    drawn = len(draws)
    admitted_total = sum(stats["train/admission/groups_admitted"] for stats in logged)
    share = sum(informative(task[p], p) for p in range(drawn)) / drawn
    assert drawn / admitted_total == pytest.approx(1 / share)  # Wald: one over the informative share
    standard_error = (plain_share * (1 - plain_share) / (STEPS * B)) ** 0.5
    assert abs(share - plain_share) <= 3 * standard_error + 3 * (share * (1 - share) / drawn) ** 0.5
    # The adaptive arm needs fewer draws per admitted group.
    adaptive_trainer, adaptive = runs["pass_rate+informative"]
    adaptive_draws = sum(1 for event in read_events(Path(adaptive_trainer.run_log_path)) if event["event"] == "draw")
    adaptive_admitted = sum(stats["train/admission/groups_admitted"] for stats in adaptive)
    print(
        f"draws per admitted group: uniform+informative {drawn / admitted_total:.2f}, pass_rate+informative "
        f"{adaptive_draws / adaptive_admitted:.2f}; all-mode informative share {plain_share:.3f}"
    )
    # Calibrated over 8 seeds of the completion order: about 1.4 draws per admitted group (the refills make it vary
    # from run to run with completion timing) against the static arm's 2.77 (all mode's share 0.354, 2.82 draws per
    # informative group).
    assert adaptive_draws / adaptive_admitted <= 1.6
    assert adaptive_draws / adaptive_admitted < 0.7 * drawn / admitted_total


def test_pass_rate_with_informative_on_a_capped_entry_sees_every_settled_group(tmp_path, monkeypatch):
    """``pass_rate`` on a capped entry under ``weighted``, with ``informative``, at ``max_staleness: 2``: the sampler's
    tokens wait for the capped label as under M4, every settled group, rejected ones included, enters the statistics it
    decides from, its predicted and realized shares count the rejected groups beside the consumed ones (so the share
    is below 1 though every consumed group trained), and every position has one fate."""

    from test_admission_run import check_fates_once

    config = _config(
        tmp_path,
        {
            "task_sampler": "weighted",
            "batch_size": 4,
            "max_staleness": 2,
            "max_training_steps": 12,
            "rollout_group_limits": {"hard": 1},
            "group_admission": "informative",
        },
        [_entry("hard", task_sampler="pass_rate"), _entry("easy")],
    )
    trainer, _saves, _logged = make_trainer(monkeypatch, config, training_client=_Recording())
    logged: list[dict[str, Any]] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(stats)
    )
    _built(trainer, config)
    _fake_rollouts(monkeypatch, trainer._model_info.llm)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=_Workflow(config)), 300))
    assert len(logged) == 12
    events = list(read_events(Path(trainer.run_log_path)))
    store = trainer._recorder.store
    settled = sum(1 for event in events if event["event"] == "settle")
    assert sum(store.counts(task).groups for task in store.task_ids()) == settled  # rejected groups included
    prefix = "train/sampler/environment/hard/"
    assert any(f"{prefix}tokens" in stats for stats in logged)
    assert all(stats.get(f"{prefix}draws/fallback", 0) == 0 for stats in logged)
    shares = [stats[f"{prefix}informative_share"] for stats in logged if f"{prefix}informative_share" in stats]
    assert shares and min(shares) < 1.0 and all(0 <= share <= 1 for share in shares)
    for stats in logged:  # every group a step trained was informative
        assert stats["train/batch/informative_groups"] == stats["train/admission/groups_admitted"]
    check_fates_once(trainer)
