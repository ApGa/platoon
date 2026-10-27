"""The shared single-rollout body: branch parity, diagnostics, safety rails.

``run_rollout_request`` is the one implementation of what
``arun_episode_single`` used to inline.  These tests pin the branch order and
the status/observed pairs it must produce, the R3 routing-id accounting it
reports, and the two rails that exist because a wrong answer here is silent:
the reward processor is never inferred, and the process-global Python-overlay
is never entered outside a worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import tinker
import torch
from fakes.workers import FakeProxySession
from tinker import TensorData

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.envs.service import EnvSetupError
from platoon.task import Task
from platoon.train.tinker.rollout_metrics import current_timeline, record_model_call
from platoon.train.workers import run as run_module
from platoon.train.workers.binding import FollowingSamplingClient, publish_binding
from platoon.train.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RefResolutionError,
    RolloutWorkerRequest,
    dump_response,
)
from platoon.train.workers.run import run_rollout_request
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def named_reward_processor(trajectory: dict) -> tuple[float, dict]:
    """A module-level, non-default reward processor that resolves by reference."""

    return trajectory["reward"], {"reward/named": trajectory["reward"]}


def _interaction(routing_id, input_tokens=100, output_tokens=10):
    return SimpleNamespace(
        obs=SimpleNamespace(length=input_tokens),
        action=SimpleNamespace(tokens=list(range(output_tokens))),
        routing_id=routing_id,
    )


def _datum() -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([2, 3, 4])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
        },
    )


def _result(*, num_datums: int, stats_datums: int, policy_excluded: int = 0):
    return TrajectoryCollectionResult(
        datums=[_datum() for _ in range(num_datums)],
        task_reward=1.0,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=1.0,
                num_steps=1,
                num_input_tokens=100,
                num_output_tokens=10,
                num_datums=stats_datums,
                is_root=True,
            )
        ],
        root_rewards_dict={},
        num_policy_excluded_datums=policy_excluded,
    )


def _request(tmp_path, **overrides) -> RolloutWorkerRequest:
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id="task-r0-abc",
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path), timeout=60)),
        rollout_fn_ref="platoon.harness.rollout:harness_rollout",
        model=ModelBinding(
            binding_path=str(tmp_path / "binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="Qwen/Qwen3",
            renderer_name="qwen3_6_preserve_thinking",
            checkpoint_version=3,
        ),
        conversion=ConversionPolicy(),
        response_path=str(tmp_path / "res.pkl"),
        ready_path=str(tmp_path / "ready"),
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


@pytest.fixture(autouse=True)
def _fake_proxy_session(monkeypatch):
    monkeypatch.setattr(run_module, "TinkerLLMProxySession", FakeProxySession)
    FakeProxySession.interactions_to_inject = {}
    yield
    FakeProxySession.interactions_to_inject = {}


def _run(request, rollout_fn, result=None, monkeypatch=None, **kwargs):
    if result is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", lambda **_kwargs: result)
    return asyncio.run(run_rollout_request(request, None, rollout_fn=rollout_fn, **kwargs))


class _SamplingStub:
    async def sample_async(self):
        return None


def test_conversion_stamps_oldest_actual_sample_not_request_or_latest(tmp_path, monkeypatch):
    request = _request(tmp_path)  # Request made at version 3.
    path = request.model.binding_path
    publish_binding(_SamplingStub(), 3, path)
    follower = FollowingSamplingClient(path, poll_seconds=0)
    llm = SimpleNamespace(sampling_client=follower)
    captured = []

    def convert(**kwargs):
        captured.append(kwargs["checkpoint_version"])
        return _result(num_datums=1, stats_datums=1)

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", convert)

    async def rollout_fn(_task, _config):
        publish_binding(_SamplingStub(), 8, path)  # Environment setup waited.
        await follower.sample_async()
        publish_binding(_SamplingStub(), 10, path)
        await follower.sample_async()  # Later child/judge sample.
        return {"trajectories": {"root": {"steps": [{}]}}}

    response = asyncio.run(run_rollout_request(request, llm, rollout_fn=rollout_fn))
    assert response.status == "completed"
    assert captured == [8]


def test_no_trajectories_branch(tmp_path):
    async def rollout_fn(_task, _config):
        return {"trajectories": {}}

    response = _run(_request(tmp_path), rollout_fn)
    assert response.status == "no_trajectories"
    assert response.observed is False
    assert response.result is None
    assert response.workload.trajectories == 0


def test_pre_sampling_zero_is_conversion_empty_but_observed(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=0, stats_datums=0), monkeypatch=monkeypatch
    )
    assert response.status == "conversion_empty"
    assert response.observed is True
    assert response.result is None


def test_empty_datums_without_sampler_or_exclusion_is_conversion_empty(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=0, stats_datums=2), monkeypatch=monkeypatch
    )
    assert response.status == "conversion_empty"
    assert response.observed is True
    assert response.result is None
    # The generation work is still accounted for.
    assert response.workload.postmerge_datums == 2


def test_empty_datums_with_a_sampler_stays_a_baseline_member(tmp_path, monkeypatch):
    """Sampling may legitimately remove every datum; the reward still counts."""

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    request = _request(
        tmp_path,
        conversion=ConversionPolicy(
            subagent_datum_sampler=DeterministicSubagentDatumSampler(keep_probability=0.5, seed=1)
        ),
    )
    response = _run(request, rollout_fn, result=_result(num_datums=0, stats_datums=2), monkeypatch=monkeypatch)
    assert response.status == "completed"
    assert response.observed is True
    assert response.result is not None and response.result.datums == []


def test_completed_rollout_carries_the_funnel(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}, {}]}}}

    FakeProxySession.interactions_to_inject = {"c1": _interaction(11), "c2": _interaction(12)}
    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=2, stats_datums=3), monkeypatch=monkeypatch
    )
    assert response.status == "completed"
    assert response.observed is True
    assert len(response.result.datums) == 2
    assert response.workload.postmerge_datums == 3
    assert (
        0
        < len(response.result.datums)
        <= response.workload.post_sampling_datums
        <= response.workload.policy_eligible_datums
        <= response.workload.postmerge_datums
    )
    # The funnel gate accepts it, which is the whole point.
    dump_response(response, str(tmp_path / "ok.pkl"))
    assert response.conversion_seconds >= 0.0
    assert response.rollout_seconds > 0.0


def test_finished_rollout_writes_its_viewer_event_file(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"id": "collection-7", "trajectories": {"root": {"id": "root", "steps": [{}], "reward": 1.0}}}

    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=1, stats_datums=1), monkeypatch=monkeypatch
    )
    assert response.status == "completed"
    events = (tmp_path / "events" / "events_task_collection-7.jsonl").read_text().splitlines()
    assert [json.loads(line)["type"] for line in events] == [
        "trajectory_created",
        "trajectory_task_set",
        "trajectory_step_added",
        "trajectory_finished",
    ]


def test_an_unwritable_event_file_does_not_fail_the_rollout(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"id": "collection-8", "trajectories": {"root": {"id": "root", "steps": [{}]}}}

    def refuse(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(run_module, "emit_rollout_events", refuse)
    result = _result(num_datums=1, stats_datums=1)
    assert _run(_request(tmp_path), rollout_fn, result=result, monkeypatch=monkeypatch).status == "completed"


def test_exception_yields_error_with_observed_following_trajectories(tmp_path):
    async def raising(_task, _config):
        raise RuntimeError("boom")

    response = _run(_request(tmp_path), raising)
    assert response.status == "error"
    assert response.result is None
    assert response.error_type == "RuntimeError"
    assert "boom" in response.error_repr
    # No trajectory was produced, so this rollout was not observed.
    assert response.observed is False


def test_a_rollout_that_raises_still_reports_the_model_calls_it_made(tmp_path):
    """The session's interactions are taken whether the rollout returned or raised: an error after two model calls
    reports both, and their routing ids."""

    FakeProxySession.interactions_to_inject = {"c1": _interaction(7), "c2": _interaction(None)}

    async def raising(_task, _config):
        raise RuntimeError("boom")

    response = _run(_request(tmp_path), raising)
    assert response.status == "error"
    assert (response.model_calls, response.routing_id_present) == (2, 1)


def test_a_converted_rollout_reports_how_long_its_conversion_took(tmp_path, monkeypatch):
    def slow(**_kwargs):
        time.sleep(0.02)
        return _result(num_datums=1, stats_datums=1)

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", slow)

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    response = _run(_request(tmp_path), rollout_fn)
    assert response.status == "completed" and response.conversion_seconds >= 0.02


def test_error_after_a_trajectory_is_observed(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    def explode(**_kwargs):
        raise ValueError("conversion blew up")

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", explode)
    response = _run(_request(tmp_path), rollout_fn)
    assert response.status == "error"
    assert response.observed is True
    assert response.workload.trajectories == 1


def _root(misc: dict) -> dict:
    return {"trajectories": {"root": {"steps": [{}], "misc": misc}, "child": {"steps": [{}], "misc": {}}}}


def test_the_response_reports_the_status_of_the_outcome_its_root_recorded(tmp_path, monkeypatch):
    async def graded(_task, _config):
        return _root({"outcome": {"status": "scored", "rewards": {"reward": 1.0}, "reason": None}})

    response = _run(_request(tmp_path), graded, result=_result(num_datums=1, stats_datums=1), monkeypatch=monkeypatch)

    assert response.status == "completed" and response.outcome_status == "scored"


def test_a_rollout_that_left_no_result_still_reports_its_outcome_status(tmp_path, monkeypatch):
    """A harness that could not be set up in its sandbox: an error outcome and no model call, so conversion keeps
    nothing."""

    async def failed_setup(_task, _config):
        return _root({"outcome": {"status": "error", "failure_kind": "harness_setup"}, "trajectory_invalid": True})

    response = _run(
        _request(tmp_path), failed_setup, result=_result(num_datums=0, stats_datums=0), monkeypatch=monkeypatch
    )

    assert response.status == "conversion_empty" and response.result is None
    assert response.outcome_status == "error"


def test_without_a_recorded_outcome_the_status_is_the_markers_or_none(tmp_path, monkeypatch):
    async def timed_out(_task, _config):
        return _root({"trajectory_timed_out": True})

    async def unrecorded(_task, _config):
        return _root({})

    async def raising(_task, _config):
        raise RuntimeError("boom")

    result = _result(num_datums=1, stats_datums=1)
    assert _run(_request(tmp_path), timed_out, result=result, monkeypatch=monkeypatch).outcome_status == "timeout"
    assert _run(_request(tmp_path), unrecorded, result=result, monkeypatch=monkeypatch).outcome_status is None
    assert _run(_request(tmp_path), raising).outcome_status is None


@pytest.mark.parametrize("failure_kind", ["sandbox_start", "env_connect", "env_config"])
def test_a_rollout_whose_session_never_opened_counts_as_an_error(tmp_path, failure_kind):
    """The environment could not open the rollout's session (a sandbox that did not start, an env server that did
    not answer, a task it cannot set up): the rollout raised before it recorded any outcome, and it counts as an
    error of its environment, not with the rollouts that recorded none."""

    async def never_opened(_task, _config):
        raise EnvSetupError(failure_kind, "the session did not open")

    response = _run(_request(tmp_path), never_opened)

    assert (response.status, response.error_type, response.result) == ("error", "EnvSetupError", None)
    assert response.observed is False
    assert response.outcome_status == "error"


def test_routing_id_present_counts_only_non_none(tmp_path, monkeypatch):
    """R3 guard: a dropped routing_id is a silently worse gradient, not an error."""

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    FakeProxySession.interactions_to_inject = {
        "c1": _interaction(7),
        "c2": _interaction(None),
        "c3": _interaction(9),
    }
    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=1, stats_datums=1), monkeypatch=monkeypatch
    )
    assert response.model_calls == 3
    assert response.routing_id_present == 2


def test_non_default_reward_processor_never_falls_back(tmp_path):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    unresolvable = _request(
        tmp_path,
        conversion=ConversionPolicy(
            reward_processor_is_default=False,
            reward_processor_ref=f"{__name__}:not_a_real_processor",
        ),
    )
    with pytest.raises(RefResolutionError):
        _run(unresolvable, rollout_fn)

    missing_ref = _request(
        tmp_path,
        conversion=ConversionPolicy(reward_processor_is_default=False, reward_processor_ref=None),
    )
    with pytest.raises(ValueError):
        _run(missing_ref, rollout_fn)


def test_reward_processor_ref_is_used_when_it_resolves(tmp_path, monkeypatch):
    seen = {}

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    def capture(**kwargs):
        seen["reward_processor"] = kwargs["reward_processor"]
        seen["checkpoint_version"] = kwargs["checkpoint_version"]
        seen["filter_errors"] = kwargs["filter_errors"]
        return _result(num_datums=1, stats_datums=1)

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", capture)
    request = _request(
        tmp_path,
        conversion=ConversionPolicy(
            filter_errors=True,
            reward_processor_is_default=False,
            reward_processor_ref=f"{__name__}:named_reward_processor",
        ),
    )
    response = _run(request, rollout_fn)

    assert seen["reward_processor"] is named_reward_processor
    assert seen["checkpoint_version"] == 3
    assert seen["filter_errors"] is True
    assert response.status == "completed"


def test_default_reward_processor_is_the_documented_identity(tmp_path, monkeypatch):
    seen = {}

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    def capture(**kwargs):
        seen["reward_processor"] = kwargs["reward_processor"]
        return _result(num_datums=1, stats_datums=1)

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", capture)
    _run(_request(tmp_path), rollout_fn)
    assert seen["reward_processor"] is run_module.DEFAULT_REWARD_PROCESSOR
    assert seen["reward_processor"]({"reward": 0.25}) == (0.25, {})


def test_python_env_isolation_is_ignored_outside_a_worker(tmp_path, monkeypatch):
    """The overlay mutates sys.path/sys.executable process-globally.

    With N concurrent in-process rollouts the first to finish would tear it
    down under the other N-1, breaking their imports and every in-flight MCP
    bridge launch (which uses ``sys.executable``).
    """

    monkeypatch.delenv(run_module.WORKER_ENV_MARKER, raising=False)
    entered = []

    import contextlib

    @contextlib.contextmanager
    def tripwire():
        entered.append(True)
        yield tmp_path

    monkeypatch.setattr("platoon.utils.rollout_python_env.isolated_rollout_python_environment", tripwire)

    async def rollout_fn(_task, _config):
        return {"trajectories": {}}

    before_path = list(sys.path)
    before_executable = sys.executable
    response = _run(_request(tmp_path, python_env_isolation=True), rollout_fn)

    assert entered == []
    assert sys.path == before_path
    assert sys.executable == before_executable
    assert response.status == "no_trajectories"


def test_python_env_isolation_is_entered_inside_a_worker(tmp_path, monkeypatch):
    monkeypatch.setenv(run_module.WORKER_ENV_MARKER, "1")
    entered = []

    import contextlib

    @contextlib.contextmanager
    def tripwire():
        entered.append(True)
        yield tmp_path

    monkeypatch.setattr("platoon.utils.rollout_python_env.isolated_rollout_python_environment", tripwire)

    async def rollout_fn(_task, _config):
        return {"trajectories": {}}

    _run(_request(tmp_path, python_env_isolation=True), rollout_fn)
    assert entered == [True]


def test_diagnostics_report_the_worker_invariants(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_TELEMETRY", "0")

    async def rollout_fn(_task, _config):
        return {"trajectories": {}}

    response = _run(_request(tmp_path), rollout_fn, extra_allow=True, startup_seconds=12.5)
    assert response.telemetry_disabled is True
    assert response.extra_allow_verified is True
    assert response.startup_seconds == 12.5
    assert response.worker_pid == os.getpid()
    assert response.torch_threads >= 1
    assert response.timing is not None

    monkeypatch.setenv("TINKER_TELEMETRY", "1")
    assert _run(_request(tmp_path), rollout_fn).telemetry_disabled is False


def test_rollout_config_round_trips_through_the_request(tmp_path):
    seen = {}

    async def rollout_fn(_task, config):
        seen["config"] = config
        return {"trajectories": {}}

    original = RolloutConfig(
        output_dir=str(tmp_path),
        timeout=1800,
        step_timeout=250,
        propagate_root_success=True,
        extra={"environment": {"a": 1}},
    )
    request = _request(tmp_path, rollout_config=asdict(original))
    _run(request, rollout_fn)
    assert seen["config"] == original
    assert seen["config"].propagate_root_success is True
    assert seen["config"].inference_params == original.inference_params


def test_status_and_observed_table_matches_the_pre_split_outcomes(tmp_path, monkeypatch):
    """One table, so a future edit to the branch order fails visibly."""

    async def empty(_task, _config):
        return {"trajectories": {}}

    async def one(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    cases = [
        (empty, None, "no_trajectories", False, False),
        (one, _result(num_datums=0, stats_datums=0), "conversion_empty", True, False),
        (one, _result(num_datums=0, stats_datums=1), "conversion_empty", True, False),
        (one, _result(num_datums=1, stats_datums=1), "completed", True, True),
    ]
    for rollout_fn, result, status, observed, has_result in cases:
        if result is not None:
            monkeypatch.setattr(
                run_module,
                "get_train_data_for_trajectory_collection",
                lambda _result=result, **_kwargs: _result,
            )
        response = _run(_request(tmp_path), rollout_fn)
        assert (response.status, response.observed, response.result is not None) == (
            status,
            observed,
            has_result,
        ), status


def test_model_call_timings_reach_the_response(tmp_path, monkeypatch):
    """REGRESSION GATE: the session owns the timeline the proxy writes into.

    ``TinkerLLMProxySession`` binds a timeline exactly as it binds the
    interaction map.  A second timeline bound around the session would be
    shadowed by it, and every client-throughput timing would silently read zero
    -- the metrics would look healthy and measure nothing.
    """

    async def rollout_fn(_task, _config):
        # Stands in for what the model gateway records on every model call.
        for index in range(3):
            record_model_call(
                started_at=float(index) * 10.0,
                render_seconds=0.5,
                sample_seconds=2.0,
                parse_seconds=0.25,
                ended_at=float(index) * 10.0 + 2.75,
            )
        return {"trajectories": {"root": {"steps": [{}]}}}

    response = _run(
        _request(tmp_path), rollout_fn, result=_result(num_datums=1, stats_datums=1), monkeypatch=monkeypatch
    )
    assert response.timing is not None
    assert response.timing.model_calls == 3
    assert response.timing.sample_seconds_total == pytest.approx(6.0)
    assert response.timing.render_seconds_total == pytest.approx(1.5)
    assert response.timing.parse_seconds_total == pytest.approx(0.75)
    # Two gaps between three calls, each the time between one call ending and
    # the next beginning.
    assert response.timing.tool_seconds_total == pytest.approx(2 * 7.25)
    assert len(response.timing.samples) == 3


def test_on_first_call_reaches_the_session_s_timeline_and_without_it_nothing_changes(tmp_path, monkeypatch):
    """The start limit's signal: the session's timeline calls ``on_first_call`` once, at the first call it is told was
    issued. Told so without one, it changes nothing: its first call is still the first one recorded."""

    fired: list[str] = []

    async def rollout_fn(_task, _config):
        timeline = current_timeline()
        assert timeline is not None
        timeline.call_started()
        timeline.call_started()
        record_model_call(
            started_at=timeline.started_at + 1.0,
            render_seconds=0.1,
            sample_seconds=0.2,
            parse_seconds=0.1,
            ended_at=timeline.started_at + 1.4,
        )
        return {"trajectories": {"root": {"steps": [{}]}}}

    result = _result(num_datums=1, stats_datums=1)
    signalled = _run(
        _request(tmp_path), rollout_fn, result=result, monkeypatch=monkeypatch, on_first_call=lambda: fired.append("1")
    )
    plain = _run(_request(tmp_path), rollout_fn, result=result, monkeypatch=monkeypatch)
    assert fired == ["1"]
    for response in (signalled, plain):
        assert response.status == "completed" and response.timing.model_calls == 1
        assert response.timing.first_model_call_seconds == pytest.approx(1.0)


def test_a_rollout_that_fails_before_the_session_still_returns_a_summary(tmp_path):
    async def rollout_fn(_task, _config):
        raise RuntimeError("boom")

    response = _run(_request(tmp_path), rollout_fn)
    assert response.timing is not None
    assert response.timing.model_calls == 0
    assert response.timing.first_model_call_seconds == -1.0


def test_the_response_names_the_reward_key_its_roots_scored_outcome_trained(tmp_path, monkeypatch):
    async def graded(_task, _config):
        outcome = {"status": "scored", "rewards": {"correctness": 1.0}, "reason": None, "reward_key": "correctness"}
        return _root({"outcome": outcome})

    async def platoons_zero(_task, _config):
        return _root({"outcome": {"status": "scored", "rewards": {"reward": 0.0}, "reason": "x", "reward_key": None}})

    result = _result(num_datums=1, stats_datums=1)
    assert _run(_request(tmp_path), graded, result=result, monkeypatch=monkeypatch).reward_key == "correctness"
    assert _run(_request(tmp_path), platoons_zero, result=result, monkeypatch=monkeypatch).reward_key is None
    empty = _result(num_datums=0, stats_datums=0)  # no result: the key still travels
    assert _run(_request(tmp_path), graded, result=empty, monkeypatch=monkeypatch).reward_key == "correctness"
