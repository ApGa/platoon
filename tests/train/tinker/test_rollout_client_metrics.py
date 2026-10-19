"""The client-side throughput metrics -- the numbers that name the bottleneck.

A training step on this path is rollout-bound while the serving engines can
sit idle.  These metrics measure the client directly: concurrency actually
achieved, the queue wait the trainer experiences, where an agent step's wall
clock goes (model call vs in-client tokenizing vs the environment), and
whether this process and its workers are CPU saturated.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

import platoon.train.tinker  # noqa: F401
from platoon.train.tinker.rollout_metrics import (
    ClientThroughputMonitor,
    RolloutTimeline,
    RolloutTimingSummary,
    current_timeline,
    monitor,
    read_process_cpu_seconds,
    read_rss_bytes,
    record_model_call,
    reset_monitors_for_tests,
    step_metrics,
    timeline_scope,
)


@pytest.fixture(autouse=True)
def _fresh_monitors():
    reset_monitors_for_tests()
    yield
    reset_monitors_for_tests()


# -- the per-rollout timeline -------------------------------------------


def test_a_timeline_splits_an_agent_step_into_its_three_parts():
    timeline = RolloutTimeline(started_at=100.0)
    # Two calls, with 5 s of tool/environment work between them.
    timeline.record(started_at=101.0, render_seconds=0.4, sample_seconds=2.0, parse_seconds=0.1, ended_at=103.5)
    timeline.record(started_at=108.5, render_seconds=0.6, sample_seconds=3.0, parse_seconds=0.2, ended_at=112.3)

    summary = timeline.summary(now=120.0)
    assert summary.model_calls == 2
    # The first call has no predecessor, so its gap is attributed to startup.
    assert summary.first_model_call_seconds == pytest.approx(1.0)
    assert summary.tool_seconds_total == pytest.approx(5.0)
    assert summary.sample_seconds_total == pytest.approx(5.0)
    assert summary.render_seconds_total == pytest.approx(1.0)
    assert summary.parse_seconds_total == pytest.approx(0.3)
    assert summary.rollout_seconds == pytest.approx(20.0)
    assert len(summary.samples) == 2


def test_a_rollout_that_never_called_the_model_reports_no_first_call():
    summary = RolloutTimeline(started_at=0.0).summary(now=5.0)
    assert summary.model_calls == 0
    # -1.0, not 0.0: "no data" must never read as "instant".
    assert summary.first_model_call_seconds == -1.0


def test_samples_are_strided_so_a_long_episode_stays_cheap_on_the_wire():
    timeline = RolloutTimeline(started_at=0.0)
    for index in range(5000):
        timeline.record(
            started_at=float(index),
            render_seconds=0.01,
            sample_seconds=0.02,
            parse_seconds=0.01,
            ended_at=index + 0.5,
        )
    summary = timeline.summary(now=6000.0)
    assert summary.model_calls == 5000  # exact
    assert len(summary.samples) == 512  # bounded
    # The sums stay exact even though the samples are downsampled.
    assert summary.sample_seconds_total == pytest.approx(100.0)


def test_record_model_call_uses_the_active_scope_and_is_a_no_op_without_one():
    assert current_timeline() is None
    record_model_call(started_at=0.0, render_seconds=1.0, sample_seconds=1.0, parse_seconds=1.0, ended_at=3.0)

    with timeline_scope() as timeline:
        assert current_timeline() is timeline
        record_model_call(
            started_at=time.monotonic(),
            render_seconds=0.1,
            sample_seconds=0.2,
            parse_seconds=0.05,
            ended_at=time.monotonic(),
        )
        assert timeline.summary().model_calls == 1
    assert current_timeline() is None


def test_a_timeline_survives_a_nonsense_record():
    timeline = RolloutTimeline(started_at=0.0)
    # Negative phase durations are clamped, not propagated as negative time.
    timeline.record(started_at=1.0, render_seconds=-5.0, sample_seconds=1.0, parse_seconds=-1.0, ended_at=2.0)
    summary = timeline.summary(now=3.0)
    assert summary.render_seconds_total == 0.0
    assert summary.parse_seconds_total == 0.0


def test_subagent_calls_land_on_the_root_rollouts_timeline():
    """Recursive rollouts must not each get their own bucket."""

    async def subagent():
        record_model_call(started_at=1.0, render_seconds=0.1, sample_seconds=0.2, parse_seconds=0.1, ended_at=1.4)

    async def root():
        with timeline_scope() as timeline:
            record_model_call(started_at=0.0, render_seconds=0.1, sample_seconds=0.2, parse_seconds=0.1, ended_at=0.4)
            await asyncio.gather(asyncio.create_task(subagent()), asyncio.create_task(subagent()))
            return timeline.summary()

    summary = asyncio.run(root())
    assert summary.model_calls == 3


# -- the parent-side monitor --------------------------------------------


def test_concurrency_is_time_weighted_not_a_count():
    """The metric that answers "are 24 rollouts actually overlapping?"."""

    async def scenario():
        client = ClientThroughputMonitor("test")
        client.set_configured_concurrency(4.0)

        async def rollout(seconds):
            with client.rollout_in_flight():
                await asyncio.sleep(seconds)

        await asyncio.gather(rollout(0.3), rollout(0.3), rollout(0.3), rollout(0.05))
        return client.export()

    metrics = asyncio.run(scenario())
    assert metrics["rollout_client/rollouts_in_flight_max"] == 4.0
    assert metrics["rollout_client/rollouts_in_flight_configured"] == 4.0
    # Mean is below the peak because one member finished early: a count would
    # have said 4 and hidden exactly the effect we are trying to see.
    assert 2.0 < metrics["rollout_client/rollouts_in_flight_mean"] < 4.0
    assert metrics["rollout_client/concurrency_achieved_fraction"] < 1.0


def test_serialised_rollouts_show_a_concurrency_of_about_one():
    """The signature of a client that is not actually overlapping work."""

    async def scenario():
        client = ClientThroughputMonitor("test")
        client.set_configured_concurrency(8.0)
        for _ in range(4):
            with client.rollout_in_flight():
                await asyncio.sleep(0.1)
        return client.export()

    metrics = asyncio.run(scenario())
    assert metrics["rollout_client/rollouts_in_flight_max"] == 1.0
    assert metrics["rollout_client/rollouts_in_flight_mean"] == pytest.approx(1.0, abs=0.2)
    assert metrics["rollout_client/concurrency_achieved_fraction"] < 0.2


def test_the_trainer_queue_wait_distinguishes_starved_from_saturated():
    starved = ClientThroughputMonitor("starved")
    for wait in (30.0, 45.0, 60.0):
        starved.record_queue_wait(wait, immediate=False)
    starved_metrics = starved.export()
    assert starved_metrics["rollout_client/trainer_queue_wait_seconds_total"] == 135.0
    assert starved_metrics["rollout_client/trainer_queue_wait_seconds_max"] == 60.0
    assert starved_metrics["rollout_client/trainer_queue_ready_fraction"] == 0.0

    saturated = ClientThroughputMonitor("saturated")
    for _ in range(10):
        saturated.record_queue_wait(0.0, immediate=True)
    saturated_metrics = saturated.export()
    assert saturated_metrics["rollout_client/trainer_queue_ready_fraction"] == 1.0
    assert saturated_metrics["rollout_client/trainer_queue_wait_seconds_total"] == 0.0


def test_agent_step_percentiles_split_model_client_and_tool_time():
    client = ClientThroughputMonitor("split")
    # Ten identical steps: 1 s of environment, 0.5 s of in-client
    # render+parse, 2 s in the gateway.
    calls = [(1.0, 0.3, 2.0, 0.2)] * 9 + [(9.0, 0.3, 2.0, 0.2)]
    client.record_rollout_timing(
        RolloutTimingSummary.from_calls(calls, rollout_seconds=40.0, first_model_call_seconds=0.7)
    )
    metrics = client.export()

    assert metrics["rollout_client/agent_step_seconds_p50"] == pytest.approx(3.5)
    assert metrics["rollout_client/agent_step_model_seconds_p50"] == pytest.approx(2.0)
    assert metrics["rollout_client/agent_step_client_seconds_p50"] == pytest.approx(0.5)
    assert metrics["rollout_client/agent_step_tool_seconds_p50"] == pytest.approx(1.0)
    # p90 catches the tail the mean would hide.
    assert metrics["rollout_client/agent_step_tool_seconds_p90"] > 1.0
    # And the shares say where the episode went, in one number each.  Nine
    # steps waited 1 s on the environment and one waited 9 s, so the totals are
    # 18 s tool, 20 s model, 5 s client -- 43 s accounted.
    assert metrics["rollout_client/agent_step_seconds_accounted_total"] == pytest.approx(43.0)
    assert metrics["rollout_client/agent_step_model_fraction"] == pytest.approx(20.0 / 43.0)
    assert metrics["rollout_client/agent_step_client_fraction"] == pytest.approx(5.0 / 43.0)
    assert metrics["rollout_client/agent_step_tool_fraction"] == pytest.approx(18.0 / 43.0)
    assert metrics["rollout_client/model_calls_per_rollout_mean"] == 10.0


def test_a_client_bound_episode_is_visible_as_a_large_client_fraction():
    """What a GIL-serialised client looks like in the metrics."""

    client = ClientThroughputMonitor("gil")
    client.record_rollout_timing(
        RolloutTimingSummary.from_calls(
            [(0.2, 6.0, 1.0, 3.0)] * 20, rollout_seconds=210.0, first_model_call_seconds=1.0
        )
    )
    metrics = client.export()
    assert metrics["rollout_client/agent_step_client_fraction"] > 0.8
    assert metrics["rollout_client/agent_step_model_fraction"] < 0.15


def test_startup_offset_makes_the_two_isolation_modes_comparable():
    client = ClientThroughputMonitor("startup")
    summary = RolloutTimingSummary.from_calls(
        [(0.0, 0.1, 1.0, 0.1)], rollout_seconds=10.0, first_model_call_seconds=2.0
    )
    client.record_rollout_timing(summary)  # in-process: no launch cost
    client.record_rollout_timing(summary, startup_offset=25.0)  # a worker's exec->ready
    metrics = client.export()

    # Two observations, 2.0 s and 27.0 s: the worker's 25 s of launch is
    # visible where the in-process run has none.
    assert metrics["rollout_client/dequeue_to_first_model_call_seconds_p50"] == pytest.approx(14.5)
    assert metrics["rollout_client/dequeue_to_first_model_call_seconds_p90"] == pytest.approx(24.5)


def test_cpu_and_memory_are_measured_for_this_process_and_its_children():
    client = ClientThroughputMonitor("cpu")
    deadline = time.monotonic() + 0.3
    total = 0
    while time.monotonic() < deadline:
        total += 1  # burn measurable CPU in this process
    assert total > 0

    metrics = client.export()
    assert metrics["rollout_client/cpu_percent"] > 0.0
    assert metrics["rollout_client/rss_bytes"] > 0.0
    assert metrics["rollout_client/threads"] >= 1.0
    assert metrics["rollout_client/total_cpu_percent"] >= metrics["rollout_client/cpu_percent"]
    assert metrics["rollout_client/cpu_cores_busy"] == pytest.approx(
        metrics["rollout_client/total_cpu_percent"] / 100.0
    )
    assert 0.0 <= metrics["rollout_client/cpu_saturation"] <= 1.0 + 1e-9


def test_worker_cpu_shows_up_as_children_cpu():
    """With process isolation the rollout CPU moves out of this process.

    ``RUSAGE_CHILDREN`` accounts for reaped children exactly, and the
    supervisor waits on every worker it launches, so this needs no sampling and
    no extra thread.
    """

    import subprocess
    import sys

    client = ClientThroughputMonitor("children")
    client.export()  # establish the window baseline
    subprocess.run(
        [sys.executable, "-c", "x=0\nfor i in range(4_000_000): x+=i"],
        check=True,
        timeout=300,
    )
    metrics = client.export()
    assert metrics["rollout_client/worker_cpu_percent"] > 0.0
    assert metrics["rollout_client/total_cpu_percent"] >= metrics["rollout_client/worker_cpu_percent"]


def test_live_worker_rss_is_summed_over_registered_pids():
    client = ClientThroughputMonitor("rss")
    client.register_worker(os.getpid())
    metrics = client.export()
    assert metrics["rollout_client/worker_processes_live"] == 1.0
    assert metrics["rollout_client/worker_rss_bytes"] > 0.0

    client.unregister_worker(os.getpid())
    assert "rollout_client/worker_rss_bytes" not in client.export()


def test_worker_peak_rss_is_reported_per_step():
    client = ClientThroughputMonitor("peak")
    client.record_worker_peak_rss(1_000)
    client.record_worker_peak_rss(3_000)
    metrics = client.export()
    assert metrics["rollout_client/worker_peak_rss_bytes_sum"] == 4000.0
    assert metrics["rollout_client/worker_peak_rss_bytes_max"] == 3000.0
    # Windowed, not cumulative for the run.
    assert "rollout_client/worker_peak_rss_bytes_sum" not in client.export()


def test_export_resets_the_window():
    client = ClientThroughputMonitor("window")
    client.record_queue_wait(5.0, immediate=False)
    assert client.export()["rollout_client/trainer_queue_wait_seconds_total"] == 5.0
    assert "rollout_client/trainer_queue_wait_seconds_total" not in client.export()


def test_step_metrics_never_raises_and_is_scope_keyed():
    monitor("train").record_queue_wait(1.0, immediate=False)
    train_metrics = step_metrics("train")
    assert train_metrics["rollout_client/trainer_queue_wait_seconds_total"] == 1.0
    # A scope with no activity is still safe and still reports resources.
    eval_metrics = step_metrics("eval")
    assert "rollout_client/trainer_queue_wait_seconds_total" not in eval_metrics
    assert "rollout_client/cpu_percent" in eval_metrics


def test_metric_collection_never_breaks_a_step(monkeypatch):
    client = ClientThroughputMonitor("broken")

    def explode(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(client, "_export", explode)
    assert client.export() == {}


def test_proc_readers_degrade_to_zero_rather_than_raising():
    assert read_rss_bytes(pid=1 << 30) == 0
    assert read_process_cpu_seconds(pid=1 << 30) == 0.0
    assert read_rss_bytes() > 0
    assert read_process_cpu_seconds() > 0.0
