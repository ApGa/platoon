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
import os
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
import pytest
import tinker
import torch
from tinker import TensorData

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.train.tinker.workers import run as run_module
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RefResolutionError,
    RolloutWorkerRequest,
    dump_response,
)
from platoon.train.tinker.workers.run import run_rollout_request
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


class _FakeSession:
    """Stands in for TinkerLLMProxySession with an injectable interaction map."""

    interactions_to_inject: dict = {}

    def __init__(self) -> None:
        self.interactions = dict(type(self).interactions_to_inject)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


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
        rollout_fn_ref="platoon.openreward.rollout:run_rollout",
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
    monkeypatch.setattr(run_module, "TinkerLLMProxySession", _FakeSession)
    _FakeSession.interactions_to_inject = {}
    yield
    _FakeSession.interactions_to_inject = {}


def _run(request, rollout_fn, result=None, monkeypatch=None, **kwargs):
    if result is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(
            run_module, "get_train_data_for_trajectory_collection", lambda **_kwargs: result
        )
    return asyncio.run(run_rollout_request(request, None, rollout_fn=rollout_fn, **kwargs))


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
    response = _run(
        request, rollout_fn, result=_result(num_datums=0, stats_datums=2), monkeypatch=monkeypatch
    )
    assert response.status == "completed"
    assert response.observed is True
    assert response.result is not None and response.result.datums == []


def test_completed_rollout_carries_the_funnel(tmp_path, monkeypatch):
    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}, {}]}}}

    _FakeSession.interactions_to_inject = {"c1": _interaction(11), "c2": _interaction(12)}
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


def test_routing_id_present_counts_only_non_none(tmp_path, monkeypatch):
    """R3 guard: a dropped routing_id is a silently worse gradient, not an error."""

    async def rollout_fn(_task, _config):
        return {"trajectories": {"root": {"steps": [{}]}}}

    _FakeSession.interactions_to_inject = {
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
            reward_processor_ref="platoon.openreward.rewards:not_a_real_processor",
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
            reward_processor_ref="platoon.openreward.rewards:reward_processor",
        ),
    )
    response = _run(request, rollout_fn)
    from platoon.openreward.rewards import reward_processor

    assert seen["reward_processor"] is reward_processor
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

    monkeypatch.setattr(
        "platoon.utils.rollout_python_env.isolated_rollout_python_environment", tripwire
    )

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

    monkeypatch.setattr(
        "platoon.utils.rollout_python_env.isolated_rollout_python_environment", tripwire
    )

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


def test_rollout_config_round_trips_through_the_request(tmp_path, monkeypatch):
    seen = {}

    async def rollout_fn(_task, config):
        seen["config"] = config
        return {"trajectories": {}}

    original = RolloutConfig(
        output_dir=str(tmp_path),
        timeout=1800,
        step_timeout=250,
        propogate_root_success=True,
        extra={"openreward": {"a": 1}},
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
