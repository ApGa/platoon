"""Worker scalars, and the cell-guard metric that would otherwise invert.

``_ptc_cell_guard_stats`` swallows every exception and returns ``{}``, so a
regression in it is invisible under unchanged key names.  Once rollouts run in
worker processes the trainer's own guard registry is permanently empty, and
naively summing the guard's *gauges* across workers would report zero
``ptc_cells_stuck`` precisely when cells are being killed for hanging -- the
worker with the stuck cell is exactly the one that gets SIGKILLed and writes no
response.  Hence: cumulative counters only, plus ``*_at_exit`` gauges, plus the
``.diag`` marker so a killed worker still contributes.

The R3 guard warns about each rollout whose routing ids were dropped, but only
once for each start of training about a model service that sends none, such as
a dense model's: warning about each of its rollouts buried the log (18,125
warnings in a 70-step run).
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import platoon.hooks
import platoon.train.tinker  # noqa: F401
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.step.metrics import _ptc_cell_guard_stats
from platoon.train.workers import telemetry as telemetry_module
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.workers.run import cell_guard_cumulative
from platoon.train.workers.supervisor import WorkerOutcome
from platoon.train.workers.telemetry import (
    local_telemetry,
    record_worker_cell_guard,
    reset_for_tests,
    worker_cell_guard_stats,
)
from platoon.train.workflow import group as group_module
from platoon.train.workflow import telemetry as workflow_telemetry
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import get as get_tracker


@pytest.fixture(autouse=True)
def _clean_aggregator(monkeypatch):
    monkeypatch.setattr(platoon.hooks, "_HOOKS", {})
    reset_for_tests()
    yield
    reset_for_tests()


def _cell_guard_snapshot():
    return {
        "ptc_cells_started": 3,
        "ptc_cells_finished": 1,
        "ptc_cells_running": 2,
        "ptc_cells_stuck": 1,
        "ptc_threads_leaked": 0,
    }


def _workflow(scope: str):
    return group_module.GroupRolloutWorkflow(
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
    # Every counter is present on every member, so the exported value is a
    # fraction with a real denominator rather than a constant 1.0.
    others = {
        "rollout_worker_hard_timeout",
        "rollout_worker_crashed",
        "rollout_worker_oom",
        "rollout_worker_launch_failed",
    } - {expected}
    assert all(scalars[name] == 0.0 for name in others)
    assert scalars[f"{expected}__count"] == 1.0
    # The cost of isolation is always reported, on every path.
    assert scalars["rollout_worker_wall_seconds"] == 30.0
    assert scalars["rollout_worker_launch_seconds"] == 1.5


def test_a_completed_rollout_emits_no_failure_scalar():
    workflow = _workflow("tel_ok")
    workflow._record_worker_telemetry(_outcome("completed", response=_response()))
    scalars = _scalars("tel_ok")
    assert scalars["rollout_worker_hard_timeout"] == 0.0
    assert scalars["rollout_worker_crashed"] == 0.0
    assert scalars["rollout_worker_killed"] == 0.0


def test_a_kill_is_recorded_separately_from_the_verdict():
    """A worker signalled during its teardown still counts as completed."""

    workflow = _workflow("tel_killed")
    workflow._record_worker_telemetry(_outcome("completed", response=_response(), killed_by_parent=True, retried=True))
    scalars = _scalars("tel_killed")
    assert scalars["rollout_worker_killed"] == 1.0
    assert scalars["rollout_worker_launch_retried"] == 1.0
    assert scalars["rollout_worker_crashed"] == 0.0


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


def _routing_response(present: int, calls: int):
    """A response that passes the R3 patch and telemetry checks, so that every warning left is about routing ids."""

    return _response(model_calls=calls, routing_id_present=present, extra_allow_verified=True, telemetry_disabled=True)


def _report(workflow, *present: int, calls: int = 4, first: int = 0) -> None:
    """Record one finished rollout of ``calls`` model calls for each count of calls that got a routing id: the rollouts
    ``rollout-<first>``, ``rollout-<first + 1>`` and so on."""

    for number, count in enumerate(present, start=first):
        workflow._record_worker_telemetry(
            _outcome("completed", response=_routing_response(count, calls), request_id=f"rollout-{number}")
        )


def _routing_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "routing id" in record.getMessage()
    ]


def test_a_service_that_sends_no_routing_ids_is_warned_about_once_not_for_each_rollout(caplog):
    """A dense model's service sends no routing ids: one warning says so, and no rollout warns on its own."""

    workflow = _workflow("tel_no_routing_ids")
    with caplog.at_level(logging.WARNING):
        _report(workflow, 0, 0, 0, 0, 0)

    [warning] = _routing_warnings(caplog)
    assert "rollout worker rollout-0 got no routing ids for its 4 model calls" in warning
    assert "the model service sends none" in warning
    assert "dense model" in warning and "mixture-of-experts" in warning and "rollout routing replay" in warning
    assert "the metric tel_no_routing_ids/rollout_worker_routing_id_coverage" in warning
    # The metric still counts every rollout.
    scalars = _scalars("tel_no_routing_ids")
    assert scalars["rollout_worker_routing_id_coverage"] == 0.0
    assert scalars["rollout_worker_routing_id_coverage__count"] == 5.0


def test_each_start_of_training_warns_on_its_own_and_says_so(caplog):
    """Each start of training (a resume, or a restart with ``watchdog.max_restarts``) builds its workflows anew, so a
    warning speaks for its own start alone: the service of an earlier start may have sent routing ids."""

    with caplog.at_level(logging.WARNING):
        _report(_workflow("tel_routing_starts"), 4, 4)  # a start whose service sent routing ids
        _report(_workflow("tel_routing_starts"), 0, 0, 0)  # the next start, whose service sends none
    [warning] = _routing_warnings(caplog)
    assert "nor did any tel_routing_starts rollout before it in this start of training" in warning
    assert "This is logged once for each start of training, not for each rollout" in warning
    assert "this run" not in warning


def test_each_rollout_with_routing_ids_for_some_model_calls_but_not_all_is_warned_about(caplog):
    workflow = _workflow("tel_partial_routing_ids")
    with caplog.at_level(logging.WARNING):
        _report(workflow, 2, 4, 3, 1)

    warnings = _routing_warnings(caplog)
    assert len(warnings) == 3
    for warning, shown in zip(warnings, ("2/4", "3/4", "1/4")):
        assert f"returned routing ids for only {shown} model calls" in warning


def test_a_rollout_without_routing_ids_is_warned_about_after_the_service_has_sent_some(caplog):
    """The service sends routing ids, so a rollout that got none lost them all."""

    workflow = _workflow("tel_dropped_routing_ids")
    with caplog.at_level(logging.WARNING):
        _report(workflow, 4, 0, 4, 0)

    warnings = _routing_warnings(caplog)
    assert len(warnings) == 2
    for warning in warnings:
        assert "returned routing ids for only 0/4 model calls" in warning
        assert "the model service sent routing ids to earlier tel_dropped_routing_ids rollouts" in warning


def test_routing_ids_that_first_come_after_rollouts_without_them_are_reported_once(caplog):
    workflow = _workflow("tel_late_routing_ids")
    with caplog.at_level(logging.WARNING):
        # A rollout that made no model call has no coverage, and is no rollout without routing ids.
        workflow._record_worker_telemetry(_outcome("completed", response=_routing_response(0, 0)))
        _report(workflow, 0, 0, 0)
    [warning] = _routing_warnings(caplog)
    assert "the model service sends none" in warning

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _report(workflow, 4, 4, 2, first=3)
    late, partial = _routing_warnings(caplog)
    assert "rollout worker rollout-3 got routing ids for 4/4 model calls" in late
    assert (
        "the first tel_late_routing_ids rollout in this start of training to get any, after 3 rollout(s) without them "
        "(rollout-0, rollout-1, rollout-2): either the model service began to send routing ids, or those rollouts lost "
        "theirs"
    ) in late
    assert "rollout worker rollout-5 returned routing ids for only 2/4 model calls" in partial

    # Routing ids have come: each rollout without them lost them.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _report(workflow, 0, 0, first=6)
    assert len(_routing_warnings(caplog)) == 2


def test_the_first_rollout_to_get_routing_ids_logs_one_line_that_names_the_first_rollouts_without(caplog):
    """A first rollout with routing ids for some of its model calls but not all, after five rollouts without any, logs
    the one report, which shows its share of model calls, and not also the warning for some but not all."""

    workflow = _workflow("tel_first_routing_ids_partial")
    with caplog.at_level(logging.WARNING):
        _report(workflow, 0, 0, 0, 0, 0, 2)
    no_ids, late = _routing_warnings(caplog)
    assert "rollout worker rollout-0 got no routing ids" in no_ids
    assert "rollout worker rollout-5 got routing ids for 2/4 model calls" in late
    assert "after 5 rollout(s) without them (rollout-0, rollout-1, rollout-2 and 2 more)" in late


def test_training_and_evaluation_rollouts_keep_their_own_routing_id_state(caplog):
    """``platoon train`` builds one workflow for each stats scope; neither scope's rollouts speak for the other's."""

    train, evaluation = _workflow("tel_routing_train"), _workflow("tel_routing_eval")
    with caplog.at_level(logging.WARNING):
        _report(train, 4, 4)
        _report(evaluation, 0, 0, 0)
    [warning] = _routing_warnings(caplog)
    assert "nor did any tel_routing_eval rollout before it in this start of training" in warning
    assert "the metric tel_routing_eval/rollout_worker_routing_id_coverage" in warning
    # Nothing trains on evaluation rollouts, whose metric is logged once for each evaluation, not for each step.
    assert "trains" not in warning and "per-step" not in warning

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _report(evaluation, 0)
        _report(train, 0)
    [warning] = _routing_warnings(caplog)
    assert "returned routing ids for only 0/4 model calls" in warning
    assert "earlier tel_routing_train rollouts" in warning


def test_the_routing_id_coverage_metric_is_unchanged():
    workflow = _workflow("tel_routing_metric")
    _report(workflow, 0, 0, 2, 4)
    scalars = _scalars("tel_routing_metric")
    assert scalars["rollout_worker_routing_id_coverage"] == (0.0 + 0.0 + 0.5 + 1.0) / 4
    assert scalars["rollout_worker_routing_id_coverage__count"] == 4.0

    workflow._record_worker_telemetry(_outcome("completed", response=_routing_response(0, 0)))
    assert "rollout_worker_routing_id_coverage" not in _scalars("tel_routing_metric")


def test_each_run_in_one_process_keeps_its_own_routing_id_state(caplog):
    """Several trainers can run in one process (tests do): no run's rollouts may silence another run's warning, or
    turn it into one for each rollout."""

    with caplog.at_level(logging.WARNING):
        _report(_workflow("tel_routing_runs"), 4, 0)
    [warning] = _routing_warnings(caplog)
    assert "returned routing ids for only 0/4 model calls" in warning

    for _ in range(2):  # two more runs of the same scope, each with a service that sends none
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _report(_workflow("tel_routing_runs"), 0, 0, 0)
        [warning] = _routing_warnings(caplog)
        assert "the model service sends none" in warning


def test_a_workflow_built_without_its_constructor_keeps_its_routing_id_state(caplog):
    """Custom workflows may build the class with ``__new__`` and set only the attributes they use."""

    workflow = group_module.GroupRolloutWorkflow.__new__(group_module.GroupRolloutWorkflow)
    workflow.stats_scope = "tel_routing_new"
    workflow.tracker = get_tracker("tel_routing_new")
    with caplog.at_level(logging.WARNING):
        _report(workflow, 0, 0, 0)
    [warning] = _routing_warnings(caplog)
    assert "nor did any tel_routing_new rollout before it in this start of training" in warning
    assert _scalars("tel_routing_new")["rollout_worker_routing_id_coverage__count"] == 3.0


def test_a_failing_routing_id_check_costs_the_rollout_none_of_its_other_telemetry(caplog, monkeypatch):
    """The check runs last: what it raises is logged at DEBUG, and the rollout loses only the check's own warning."""

    workflow = _workflow("tel_routing_check_fails")

    def broken(*args):
        raise RuntimeError("the routing id check is broken")

    monkeypatch.setattr(workflow, "_warn_of_missing_routing_ids", broken)
    with caplog.at_level(logging.INFO, logger=workflow_telemetry.__name__):
        _report(workflow, 0)
    scalars = _scalars("tel_routing_check_fails")
    assert scalars["rollout_worker_routing_id_coverage"] == 0.0
    assert scalars["rollout_worker_wall_seconds"] == 30.0
    assert any("rollout worker rollout-0 pid=1234 disposition=completed" in r.getMessage() for r in caplog.records)


def test_two_rollouts_that_report_at_once_from_threads_warn_once(caplog):
    """One rollout at a time reads the workflow's routing id state, logs what it says and records what it got.

    The handler below holds the thread that logs the first warning about routing ids for half a second, before that
    thread records its rollout. Without the lock, the other thread's rollout, which got no routing ids either, would
    read the state meanwhile and warn as well.
    """

    workflow = _workflow("tel_routing_threads")
    held: list[logging.LogRecord] = []

    class Holding(logging.Handler):
        def emit(self, record):
            if "routing id" in record.getMessage() and not held:
                held.append(record)
                time.sleep(0.5)

    handler = Holding(logging.WARNING)
    workflow_telemetry.logger.addHandler(handler)
    start = threading.Barrier(2, timeout=30)  # a thread that never comes fails the test rather than hang it

    def report(number):
        start.wait()
        _report(workflow, 0, first=number)

    try:
        with caplog.at_level(logging.WARNING):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(report, range(2)))
    finally:
        workflow_telemetry.logger.removeHandler(handler)

    assert len(held) == 1
    assert len(_routing_warnings(caplog)) == 1
    assert _scalars("tel_routing_threads")["rollout_worker_routing_id_coverage__count"] == 2.0


def test_a_worker_with_telemetry_enabled_is_an_error(caplog):
    workflow = _workflow("tel_telemetry")
    with caplog.at_level(logging.ERROR):
        workflow._record_worker_telemetry(_outcome("completed", response=_response(telemetry_disabled=False)))
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
    """A killed worker still reports its stuck cells.

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
    from platoon.train.workers.protocol import ConversionPolicy, ModelBinding, RolloutWorkerRequest
    from platoon.train.workers.supervisor import RolloutWorkerSupervisor

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


def test_local_telemetry_is_empty_without_an_extension():
    assert local_telemetry() == {}
    assert cell_guard_cumulative(at_exit=True) == {}
    assert _ptc_cell_guard_stats("no_extension") == {}


def test_local_telemetry_merges_hooks_and_skips_a_failing_one():
    def broken():
        raise RuntimeError("guard is broken")

    platoon.hooks.register("rollout_worker.telemetry", broken)
    platoon.hooks.register("rollout_worker.telemetry", _cell_guard_snapshot)
    platoon.hooks.register("rollout_worker.telemetry", lambda: {"other_counter": "2"})

    assert local_telemetry() == {
        "ptc_cells_started": 3.0,
        "ptc_cells_finished": 1.0,
        "ptc_cells_running": 2.0,
        "ptc_cells_stuck": 1.0,
        "ptc_threads_leaked": 0.0,
        "other_counter": 2.0,
    }


def test_a_worker_forwards_cumulative_counters_and_gauges_sampled_at_exit():
    platoon.hooks.register("rollout_worker.telemetry", _cell_guard_snapshot)

    assert cell_guard_cumulative() == {"ptc_cells_started": 3, "ptc_cells_finished": 1}
    assert cell_guard_cumulative(at_exit=True) == {
        "ptc_cells_started": 3,
        "ptc_cells_finished": 1,
        "ptc_cells_stuck_at_exit": 1,
        "ptc_threads_leaked_at_exit": 0,
    }


def test_inprocess_stats_fall_back_to_the_local_snapshot():
    """Without worker reports the trainer's own guard keeps the instantaneous names."""

    platoon.hooks.register("rollout_worker.telemetry", _cell_guard_snapshot)
    stats = _ptc_cell_guard_stats("inprocess")
    assert stats["ptc_cells_running"] == 2.0
    assert stats["ptc_cells_stuck"] == 1.0
