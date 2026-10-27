"""The run's policies (``platoon.train.policy``): main, as ``train:`` declares it, registered as the process's model
gateway, opened from the weights its spec names, bound for the rollout workers that sample it, and holding the state
the training loops read and write through their shared state."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.train import task_stats
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy import service as service_module
from platoon.train.policy import set as set_module
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec, policy_specs
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import build_config

SETTINGS = {
    "train": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer_name": "qwen3_instruct",
        "renderer_kwargs": {"enable_thinking": False},
        "context_window_length": 32768,
        "sample_timeout_seconds": 300,
        "lora_rank": 16,
    },
    "tinker_base_url": "http://self-hosted:8000",
    "checkpoint": {"load_checkpoint_path": "tinker://run-1/weights/000010"},
    "eval": {"every": 0},
    "log_path": "logs",
}


def test_main_is_the_policy_train_declares_under_its_train_names():
    config = build_config(PlatoonTinkerRLTrainerConfig, SETTINGS)

    assert policy_specs(config) == {
        MAIN: PolicySpec(
            id=MAIN,
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            renderer_name="qwen3_instruct",
            renderer_kwargs={"enable_thinking": False},
            context_window_length=32768,
            sample_timeout_seconds=300,
            tinker_base_url="http://self-hosted:8000",
            lora_rank=16,
            load_checkpoint_path="tinker://run-1/weights/000010",
        )
    }
    assert MAIN is task_stats.MAIN  # the per-task statistics' snapshots keep their format


def test_a_spec_hashes_as_a_frozen_value():
    spec = PolicySpec(MAIN, model_name="m", renderer_name="r", renderer_kwargs={"enable_thinking": False})

    assert hash(spec) == hash(dataclasses.replace(spec)) and {spec: 1}[dataclasses.replace(spec)] == 1


def test_building_the_policies_registers_main_s_gateway_as_the_trainer_did(monkeypatch):
    calls = []

    def register(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(llm=SimpleNamespace(version=0))

    monkeypatch.setattr(set_module, "register_tinker_llm", register)
    policies = PolicySet.build(policy_specs(build_config(PlatoonTinkerRLTrainerConfig, SETTINGS)))

    assert list(policies) == [MAIN] and policies.leader == MAIN
    assert calls == [
        (
            ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct"),
            {
                "context_window_length": 32768,
                "renderer_kwargs": {"enable_thinking": False},
                "base_url": "http://self-hosted:8000",
                "sample_timeout_seconds": 300,
            },
        )
    ]


def test_building_refuses_more_than_one_policy_which_would_each_become_the_default_gateway(monkeypatch):
    calls = []
    monkeypatch.setattr(set_module, "register_tinker_llm", lambda *args, **kwargs: calls.append(args))
    main = PolicySpec(MAIN, model_name="m", renderer_name="r")
    teacher = PolicySpec("teacher", model_name="t", renderer_name="r")

    with pytest.raises(ValueError, match="one policy, main, not 2: main, teacher"):
        PolicySet.build({MAIN: main, "teacher": teacher})
    with pytest.raises(ValueError, match="one policy, main, not 0"):
        PolicySet.build({})
    assert calls == []  # no gateway registered


def test_a_run_has_at_least_one_policy():
    with pytest.raises(ValueError, match="at least one policy"):
        PolicySet([])


class _WarmService:
    """A Tinker service that is restarting at the first ask for a training client, then answers."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, Any]] = []

    async def create_training_client_from_state_async(self, path: str) -> str:
        self.asked.append(("weights", path))
        if len(self.asked) == 1:
            raise ConnectionError("the service restarts")
        return "warm-client"

    async def create_lora_training_client_async(self, model_name: str, rank: int) -> str:
        self.asked.append(("new", model_name))
        return "new-client"


def test_a_new_run_starts_from_the_weights_its_spec_names_and_logs_them_once_loaded(caplog):
    caplog.set_level(logging.INFO, logger=service_module.__name__)
    weights = "tinker://run-1/weights/000010"
    spec = PolicySpec(MAIN, model_name="m", renderer_name="r", lora_rank=8, load_checkpoint_path=weights)
    service: Any = _WarmService()

    def logged() -> list[str]:
        return [record.getMessage() for record in caplog.records if record.name == service_module.__name__]

    with pytest.raises(ConnectionError):
        asyncio.run(service_module.create_training_client(service, None, spec))
    assert logged() == []  # not loaded: nothing logged
    assert asyncio.run(service_module.create_training_client(service, None, spec)) == "warm-client"

    assert service.asked == [("weights", weights), ("weights", weights)]  # its weights, never a new adapter
    assert logged() == [f"Loaded weights from {weights}"]


def test_a_rollout_worker_follows_the_policy_s_binding_at_the_poll_interval_the_config_sets():
    llm = SimpleNamespace(
        model_name="m", context_window_length=32768, sample_timeout_seconds=300.0, max_tokens=512, temperature=0.7
    )
    info = SimpleNamespace(llm=llm, model_name="m", renderer_name="r", renderer_kwargs={"enable_thinking": False})
    workflow: Any = SimpleNamespace(  # the method's self
        _isolation="process",
        _binding_path="/srv/run/model_binding.pkl",
        model_info=info,
        config=SimpleNamespace(worker_binding_poll_seconds=0.5),
    )

    binding = GroupRolloutWorkflow._model_binding(workflow, 7)

    assert (binding.binding_path, binding.follow_poll_seconds) == ("/srv/run/model_binding.pkl", 0.5)
    assert (binding.model_name, binding.renderer_name) == ("m", "r")
    assert binding.renderer_kwargs == {"enable_thinking": False}
    assert (binding.context_window_length, binding.sample_timeout_seconds) == (32768, 300.0)
    assert (binding.max_tokens, binding.temperature, binding.checkpoint_version) == (512, 0.7, 7)


def _state(**given: Any) -> TrainLoopSharedState:
    fakes: dict[str, Any] = {
        "config": SimpleNamespace(),
        "train_dataloader": [],
        "stats_logger": SimpleNamespace(),
        "train_tracker": SimpleNamespace(),
    }
    return TrainLoopSharedState(
        shutdown_event=asyncio.Event(),
        train_step=0,
        eval_dataloader=None,
        sampling_client_step=0,
        sampling_client_updated_event=asyncio.Event(),
        **fakes,
        **given,
    )


def test_the_shared_state_reads_and_writes_the_leader_policy_s_state():
    info: Any = SimpleNamespace(model_name="m")
    client: Any = object()
    policy = Policy(PolicySpec(MAIN, model_name="m", renderer_name="r"), info)
    policy.training_client, policy.optimizer_steps = client, 3
    state = _state(policies=PolicySet([policy]))

    assert state.model_info is info and state.training_client is client and state.optimizer_steps == 3
    state.optimizer_steps += 1
    sampler: Any = object()
    state.sampling_client = sampler
    assert policy.optimizer_steps == 4 and policy.sampling_client is sampler


def test_a_shared_state_given_the_policy_s_state_keeps_it_in_a_policy_of_its_own():
    """As platoon 0.4.0's code and the tests make one: the six keyword arguments, and no policies."""

    info, client = SimpleNamespace(model_name="m", renderer_name="r"), object()
    state = _state(model_info=info, training_client=client, sampling_client=None, optimizer_steps=2)

    assert state.policies is not None and list(state.policies) == [MAIN]
    leader = state.policies[MAIN]
    assert (leader.spec.model_name, leader.spec.renderer_name) == ("m", "r")
    assert state.model_info is info and state.training_client is client and state.sampling_client is None
    assert state.optimizer_steps == 2 and state.optimizer_params is None and state.service_client is None


def test_a_shared_state_takes_the_policies_or_the_policy_s_state_not_both():
    """The part of the policy's state given beside the policies would be dropped.  ``dataclasses.replace()`` gives
    both."""

    state = _state(
        model_info=SimpleNamespace(model_name="m"),
        service_client="service",
        training_client="tc-1",
        sampling_client="sc-1",
        optimizer_params="adam",
    )

    with pytest.raises(TypeError, match="not both: given policies and training_client, optimizer_steps"):
        _state(policies=state.policies, training_client="tc-2", optimizer_steps=0)
    with pytest.raises(TypeError, match="not both"):
        dataclasses.replace(state, training_client="tc-2")
    assert state.training_client == "tc-1" and state.optimizer_steps == 0
