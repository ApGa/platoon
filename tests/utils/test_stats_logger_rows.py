"""The stats logger's rows without a W&B step of their own (``StatsLogger.log_row``), as the steps of policies that
step independently log them: W&B numbers the rows, and is told once per metric prefix which metric of the row is its
axis (``define_metric``): ``policies/<id>/...`` against the policy's own step, every other prefix against the row's
axis; a row on the run's axis carries the elapsed time, as a stepped row does."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from platoon.utils.stats_logger import StatsLogger, StatsLoggerConfig, WandBConfig


def test_each_prefix_is_told_its_axis_once_and_the_rows_carry_no_step(tmp_path, monkeypatch):
    calls: list[tuple[str, Any, Any]] = []
    wandb: Any = SimpleNamespace(
        define_metric=lambda name, step_metric=None: calls.append(("define", name, step_metric)),
        log=lambda stats, **kwargs: calls.append(("log", sorted(stats), kwargs)),
    )
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = StatsLoggerConfig(log_dir=str(tmp_path), wandb=WandBConfig(mode="disabled"), print_stats=False)
    stats_logger = StatsLogger(config)
    started: Any = object()
    stats_logger._wandb_run = started  # as if W&B had started

    stats_logger.log_row({"train/loss": 1.0, "run/step": 1}, axis="run/step")
    stats_logger.log_row({"policies/worker/train/loss": 2.0, "policies/worker/step": 1}, axis="policies/worker/step")
    stats_logger.log_row({"train/loss": 0.5, "run/step": 2}, axis="run/step")

    defined = [(name, step) for kind, name, step in calls if kind == "define"]
    assert defined == [
        ("run/step", None),
        ("run/*", "run/step"),
        ("time/*", "run/step"),
        ("train/*", "run/step"),
        ("policies/worker/step", None),
        ("policies/worker/*", "policies/worker/step"),
    ]
    logs = [(names, kwargs) for kind, names, kwargs in calls if kind == "log"]
    assert all(kwargs == {} for _, kwargs in logs)  # no W&B step: W&B numbers the rows
    assert logs[1][0] == ["policies/worker/step", "policies/worker/train/loss"]  # a policy's row holds its own only


def test_a_row_on_the_run_s_axis_carries_the_elapsed_time_whatever_its_policies_metrics(tmp_path, monkeypatch):
    logged: list[dict[str, Any]] = []
    wandb: Any = SimpleNamespace(define_metric=lambda *args, **kwargs: None, log=lambda stats: logged.append(stats))
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    config = StatsLoggerConfig(log_dir=str(tmp_path), wandb=WandBConfig(mode="disabled"), print_stats=False)
    stats_logger = StatsLogger(config)
    started: Any = object()
    stats_logger._wandb_run = started  # as if W&B had started

    # The leader's row: its metrics, the run's, and its own under policies/main/.
    stats_logger.log_row({"train/loss": 1.0, "policies/main/share_wait_s": 0.1, "run/step": 1}, axis="run/step")
    stats_logger.log_row({"policies/worker/step": 1}, axis="policies/worker/step")
    assert "time/elapsed_seconds" in logged[0] and "time/elapsed_seconds" not in logged[1]
