"""Worker scalars, and the cell-guard metric that would otherwise invert.

``_ptc_cell_guard_stats`` swallows every exception and returns ``{}``, so a
regression in it is invisible under unchanged key names.  Once rollouts run in
worker processes the trainer's own guard registry is permanently empty, and
naively summing the guard's *gauges* across workers would report zero
``ptc_cells_stuck`` precisely when cells are being killed for hanging -- the
worker with the stuck cell is exactly the one that gets SIGKILLed and writes no
response.  Hence: cumulative counters only, plus ``*_at_exit`` gauges, plus the
``.diag`` marker so a killed worker still contributes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace

import pytest

import platoon.train.tinker  # noqa: F401
from platoon.envs.base import Task
from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.rl import _ptc_cell_guard_stats
from platoon.train.tinker.workers import telemetry as telemetry_module
from platoon.train.tinker.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.tinker.workers.supervisor import WorkerOutcome
from platoon.train.tinker.workers.telemetry import (
    record_worker_cell_guard,
    reset_for_tests,
    worker_cell_guard_stats,
)
from platoon.train.tinker.workflows import group_rollout_workflow as grw
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import get as get_tracker


@pytest.fixture(autouse=True)
def _clean_aggregator():
    reset_for_tests()
    yield
    reset_for_tests()


def _workflow(scope: str):
    return grw.GroupRolloutWorkflow(
        rollout_fn=None,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=WorkflowConfig(group_size=1),
        model_info=SimpleNamespace(
            llm=SimpleNamespace(version=0, model_name=""),
            model_name="",
            base_url="None",
            api_key="None",
            renderer_name="",
            renderer_kwargs={},
        ),
        stats_scope=scope,
    )


def _outcome(disposition, *, response=None, **kwargs):
    base = dict(
        response=response,
        returncode=0,
        disposition=disposition,
        killed_by_parent=False,
        diag=None,
        launch_seconds=1.5,
        wall_seconds=30.0,
        retried=False,
        worker_pid=1234,
        log_path=None,
        request_id="task-r0-abc",
    )
    base.update(kwargs)
    return WorkerOutcome(**base)


def _response(**kwargs):
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0-abc",
        status="completed",
        result=None,
        workload=RolloutWorkload(),
        observed=True,
    )
    base.update(kwargs)
    return RolloutWorkerResponse(**base)


def _scalars(scope):
    exported = get_tracker(scope).export(reset=True)
    return {key.removeprefix(f"{scope}/"): value for key, value in exported.items()}


@pytest.mark.parametrize(
    "disposition,expected",
    [
        ("hard_timeout", "rollout_worker_hard_timeout"),
        ("worker_crashed", "rollout_worker_crashed"),
        ("rollout_worker_oom", "rollout_worker_oom"),
        ("worker_launch_failed", "rollout_worker_launch_failed"),
    ],
)
def test_each_failure_disposition_emits_exactly_its_own_scalar(disposition, expected):
    scope = f"tel_{disposition}"
    workflow = _workflow(scope)
    workflow._record_worker_telemetry(_outcome(disposition))
    scalars = _scalars(scope)

    assert scalars[expected] == 1.0
    others = {
        "rollout_worker_hard_timeout",
        "rollout_worker_crashed",
        "rollout_worker_oom",
        "rollout_worker_launch_failed",
    } - {expected}
    assert not (others & scalars.keys())
    # The cost of isolation is always reported, on every path.
    assert scalars["rollout_worker_wall_seconds"] == 30.0
    assert scalars["rollout_worker_launch_seconds"] == 1.5


def test_a_completed_rollout_emits_no_failure_scalar():
    workflow = _workflow("tel_ok")
    workflow._record_worker_telemetry(_outcome("completed", response=_response()))
    scalars = _scalars("tel_ok")
    assert "rollout_worker_hard_timeout" not in scalars
    assert "rollout_worker_crashed" not in scalars
    assert "rollout_worker_killed" not in scalars


def test_a_kill_is_recorded_separately_from_the_verdict():
    """A worker signalled during its teardown still counts as completed."""

    workflow = _workflow("tel_killed")
    workflow._record_worker_telemetry(
        _outcome("completed", response=_response(), killed_by_parent=True, retried=True)
    )
    scalars = _scalars("tel_killed")
    assert scalars["rollout_worker_killed"] == 1.0
    assert scalars["rollout_worker_launch_retried"] == 1.0
    assert "rollout_worker_crashed" not in scalars


def test_routing_coverage_below_one_is_reported_and_warned(caplog):
    """R3: a dropped routing id raises nothing anywhere; only this says so."""

    workflow = _workflow("tel_routing")
    response = _response(model_calls=4, routing_id_present=2, extra_allow_verified=True)
    with caplog.at_level(logging.WARNING):
        workflow._record_worker_telemetry(_outcome("completed", response=response))
    scalars = _scalars("tel_routing")

    assert scalars["rollout_worker_routing_id_coverage"] == 0.5
    assert any("routing_handle coverage" in record.message for record in caplog.records)

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        workflow._record_worker_telemetry(
            _outcome(
                "completed",
                response=_response(model_calls=4, routing_id_present=4, extra_allow_verified=True),
            )
        )
    assert _scalars("tel_routing")["rollout_worker_routing_id_coverage"] == 1.0
    assert not any("routing_handle coverage" in record.message for record in caplog.records)


def test_a_worker_with_telemetry_enabled_is_an_error(caplog):
    workflow = _workflow("tel_telemetry")
    with caplog.at_level(logging.ERROR):
        workflow._record_worker_telemetry(
            _outcome("completed", response=_response(telemetry_disabled=False))
        )
    assert any("telemetry ENABLED" in record.message for record in caplog.records)


def test_the_aggregator_sums_cumulative_counters_across_workers():
    record_worker_cell_guard(
        "agg",
        {"ptc_cells_started": 5, "ptc_cells_finished": 5, "ptc_cells_stuck_at_exit": 0},
    )
    record_worker_cell_guard(
        "agg",
        {"ptc_cells_started": 3, "ptc_cells_finished": 2, "ptc_cells_stuck_at_exit": 1},
    )
    stats = worker_cell_guard_stats("agg", reset=False)

    assert stats["ptc_cells_started"] == 8.0
    assert stats["ptc_cells_finished"] == 7.0
    assert stats["ptc_cells_stuck_at_exit"] == 1.0
    assert stats["ptc_worker_reports"] == 2.0
    # ``ptc_cells_running`` deliberately does not cross: an exited worker runs
    # nothing, so summing it would be meaningless.
    assert "ptc_cells_running" not in stats

    assert _ptc_cell_guard_stats("agg")["ptc_cells_started"] == 8.0
    # Exported once per step, then reset.
    assert worker_cell_guard_stats("agg") == {}


def test_a_killed_worker_contributes_through_its_diag_marker():
    """THE regression this whole item exists for.

    The worker with a stuck cell is exactly the one that gets SIGKILLed and
    writes no response, so without the marker ``ptc_cells_stuck`` reads zero
    precisely when a cell was killed for hanging.
    """

    workflow = _workflow("tel_diag")
    workflow._record_worker_telemetry(
        _outcome(
            "hard_timeout",
            response=None,
            diag={
                "phase": "rollout",
                "elapsed": "3720.0",
                "ptc_cells_started": "2",
                "ptc_cells_finished": "1",
                "ptc_cells_stuck_at_exit": "1",
                "ptc_threads_leaked_at_exit": "1",
                "stuck_cells": "while True: pass@3600s",
            },
        )
    )
    stats = worker_cell_guard_stats("tel_diag", reset=False)
    assert stats["ptc_cells_stuck_at_exit"] == 1.0
    assert stats["ptc_threads_leaked_at_exit"] == 1.0
    assert stats["ptc_cells_started"] == 2.0
    assert _scalars("tel_diag")["rollout_worker_hard_timeout"] == 1.0


def test_ptc_cell_guard_stats_never_raises(monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("aggregator is broken")

    monkeypatch.setattr(telemetry_module, "worker_cell_guard_stats", explode)
    # Falls through to the local snapshot (or {}), and never raises.
    assert isinstance(_ptc_cell_guard_stats("whatever"), dict)


def test_worker_counters_take_precedence_over_the_empty_local_snapshot():
    """In process mode the trainer's own registry is permanently empty."""

    record_worker_cell_guard("precedence", {"ptc_cells_started": 9, "ptc_cells_finished": 9})
    stats = _ptc_cell_guard_stats("precedence")
    assert stats["ptc_cells_started"] == 9.0
    assert "ptc_cells_running" not in stats


def test_malformed_worker_counters_are_ignored_not_fatal():
    record_worker_cell_guard("junk", {"ptc_cells_started": "not a number"})
    assert worker_cell_guard_stats("junk", reset=False) == {}
    record_worker_cell_guard("junk", None)
    record_worker_cell_guard("junk", {"unrelated": 1})
    assert worker_cell_guard_stats("junk", reset=False) == {}


def test_an_over_allocating_worker_is_classified_as_an_oom_not_a_crash(tmp_path, monkeypatch):
    """A real worker, tripped by its own RSS watchdog.

    The distinct exit code plus the ``phase=rss`` marker is what makes this a
    deterministic classification rather than an anonymous ``rc=-9`` that could
    equally be a hard timeout or an external sweep.
    """

    from dataclasses import asdict

    from platoon.config_defs import RolloutConfig
    from platoon.train.tinker.workers.protocol import (
        ConversionPolicy,
        ModelBinding,
        RolloutWorkerRequest,
    )
    from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor

    stub_dir = os.path.dirname(os.path.abspath(__file__))
    scratch = tmp_path / "ipc"
    scratch.mkdir()
    supervisor = RolloutWorkerSupervisor(
        scratch_dir=str(scratch),
        log_dir=str(tmp_path / "logs"),
        parent_grace_seconds=120.0,
        launch_retries=0,
        env_overrides={
            "PYTHONPATH": os.pathsep.join([stub_dir, os.environ.get("PYTHONPATH", "")]),
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
        },
    )
    request = RolloutWorkerRequest(
        protocol_version=PROTOCOL_VERSION,
        request_id="rss-0",
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path))),
        rollout_fn_ref="worker_stub_rollouts:sleep_forever",
        model=ModelBinding(
            binding_path=str(scratch / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(scratch / "rss-0.res"),
        ready_path=str(scratch / "rss-0.ready"),
        startup_budget_seconds=300.0,
        rollout_budget_seconds=120.0,
        # Any real worker is far above this, so the watchdog trips on its first
        # sample after readiness.
        rss_soft_limit_bytes=1,
    )
    outcome = asyncio.run(supervisor.run(request))

    assert outcome.disposition == "rollout_worker_oom"
    assert outcome.diag is not None and outcome.diag["phase"] == "rss"
    assert int(outcome.diag["rss_bytes"]) > 1

    workflow = _workflow("tel_oom")
    workflow._record_worker_telemetry(outcome)
    assert _scalars("tel_oom")["rollout_worker_oom"] == 1.0
