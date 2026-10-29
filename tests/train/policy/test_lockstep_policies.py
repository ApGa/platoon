"""A run's policies trained in lockstep (RFC 0021, S7), as the policy package holds them: what loading reads and refuses
of a trained ``policies:`` entry, the bound the dispatcher's horizon counts, a policy's publications and subscribers,
its work done at once with the others' (naming the one that fails), its optimizer state loaded with its checkpoint, and
a resume that changes which policies train.  Their checkpoints, the step's lanes, their credit and their evaluations
are tested beside those packages' own tests; whole starts on the fake Tinker service in
``trainer/test_lockstep_run.py``."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy import service
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import (
    MAIN,
    PolicySpec,
    failure_note,
    horizon_bound,
    policy_specs,
    trained_policies,
)
from platoon.utils.config import ConfigError, build_config

SETTINGS: dict[str, Any] = {
    "train": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer_name": "qwen3_instruct",
        "optimizer": {"learning_rate": 3e-5},
        "loss_fn": "cispo",
        "loss_fn_config": {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0},
        "max_staleness": 3,
    },
    "eval": {"every": 0},
    "log_path": "logs",
}
WORKER = {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"}
BOUND = {"agents": {"roles": {"subagent": {"policy": "worker"}}}}


def config(**sections: Any) -> PlatoonTinkerRLTrainerConfig:
    return build_config(PlatoonTinkerRLTrainerConfig, {**copy.deepcopy(SETTINGS), **sections})


def problems(**sections: Any) -> list[str]:
    with pytest.raises(ConfigError) as raised:
        config(**sections)
    return raised.value.problems


def worker(**settings: Any) -> dict[str, Any]:
    """A trained worker policy bound to the subagents' role, with ``settings``."""

    return {"policies": {"worker": {**WORKER, **settings}}, **BOUND}


# --- loading ----------------------------------------------------------------------------------------------------------


def test_a_trained_policy_takes_train_s_adapter_rank_optimizer_and_loss_unless_it_sets_its_own():
    specs = policy_specs(config(train={**SETTINGS["train"], "lora_rank": 16}, **worker()))
    assert trained_policies(config(**worker())) == (MAIN, "worker")
    spec = specs["worker"]
    assert spec.trainable and spec.lora_rank == 16 and spec.max_staleness is None and not spec.load_optimizer_state
    assert spec.optimizer == specs[MAIN].optimizer and (spec.loss_fn, spec.loss_fn_config) == (
        "cispo",
        {"clip_low_threshold": 0.0, "clip_high_threshold": 5.0},
    )

    own = policy_specs(
        config(**worker(lora_rank=8, optimizer={"learning_rate": 1e-5}, loss_fn="importance_sampling", max_staleness=2))
    )["worker"]
    assert own.optimizer is not None and (own.lora_rank, own.optimizer.learning_rate, own.max_staleness) == (8, 1e-5, 2)
    # Another loss than train:'s takes none of train:'s loss config, whose keys are for train:'s loss.
    assert (own.loss_fn, own.loss_fn_config) == ("importance_sampling", {})
    given = policy_specs(config(**worker(loss_fn="ppo", loss_fn_config={"clip_low_threshold": 0.8})))["worker"]
    assert given.loss_fn_config == {"clip_low_threshold": 0.8}


@pytest.mark.parametrize("key", ["batch_size", "checkpoint"])
def test_a_policy_s_own_batch_or_cadence_is_refused_in_lockstep_naming_independent_stepping(key):
    [problem] = problems(**worker(**{key: 4 if key == "batch_size" else {"recovery_every": 5}}))
    assert problem.startswith(f"policies.worker.{key}: a policy's own ")
    assert "(train.stepping: independent); stepping together (in lockstep, as train.stepping or the" in problem


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"lora_rank": 0}, "policies.worker: lora_rank must be a whole number, 1 or more, or null, got 0"),
        ({"max_staleness": -1}, "policies.worker: max_staleness must be a whole number of versions, 0 or more"),
        ({"loss_fn": ""}, "policies.worker: loss_fn must name a Tinker loss, or be null, got ''"),
    ],
)
def test_a_trained_policy_s_keys_out_of_range_are_refused_naming_the_policy(settings, message):
    with pytest.raises(ValueError) as raised:
        config(**worker(**settings))
    assert str(raised.value).startswith(message)
    # Two entries of one model: the message says which one is wrong.
    with pytest.raises(ValueError, match=r"^policies\.critic: lora_rank"):
        config(policies={"worker": WORKER, "critic": {**WORKER, **settings, "lora_rank": 0}})


def test_a_frozen_policy_with_a_trained_policy_s_keys_is_refused():
    judge = {**WORKER, "trainable": False, "lora_rank": 8, "optimizer": {"learning_rate": 1.0}, "max_staleness": 1}
    [problem] = problems(policies={"judge": judge}, agents={"roles": {"judge": {"policy": "judge"}}})
    assert problem == (
        "policies.judge: lora_rank, optimizer, max_staleness are for a policy the run trains, and this one is frozen "
        "(trainable: false), so it never steps: leave them out"
    )
    [problem] = problems(policies={"judge": {**WORKER, "trainable": False, "lora_rank": 8}}, **_judged())
    assert problem.endswith(
        "lora_rank is for a policy the run trains, and this one is frozen (trainable: false), so it never steps: "
        "leave it out"
    )


def _judged() -> dict[str, Any]:
    return {"agents": {"roles": {"judge": {"policy": "judge"}}}}


def test_an_optimizer_state_to_load_needs_the_checkpoint_it_is_in():
    [problem] = problems(**worker(load_optimizer_state=True))
    assert problem.startswith("policies.worker.load_optimizer_state: it loads the optimizer state of the checkpoint")
    loaded = config(**worker(load_optimizer_state=True, load_checkpoint_path="tinker://run/weights/000010"))
    assert policy_specs(loaded)["worker"].load_optimizer_state


def test_the_root_may_sample_any_trained_policy():
    loaded = config(
        policies={"worker": WORKER}, agents={"policy": "worker", "roles": {"lead": {"policy": "main"}}}
    )  # loads
    assert trained_policies(loaded) == (MAIN, "worker")


def test_a_trained_policy_no_agent_samples_is_refused():
    [problem] = problems(policies={"worker": WORKER})
    assert problem.startswith("policies.worker: the run trains it, and no agent samples it, so no row would ever train")
    [problem] = problems(policies={"worker": WORKER}, agents={"policy": "worker"})
    assert problem.startswith("agents.policy: the root agent samples worker, and no role samples main")


@pytest.mark.parametrize(
    ("run_bound", "own", "horizon"),
    [(3, None, 3), (3, 1, 1), (3, 5, 3), (None, 2, 2), (None, None, None)],
)
def test_the_horizon_counts_the_smallest_bound_of_the_trained_policies(run_bound, own, horizon):
    sections = worker(max_staleness=own)
    loaded = config(train={**SETTINGS["train"], "max_staleness": run_bound}, **sections)
    assert horizon_bound(loaded) == horizon
    alone = config(train={**SETTINGS["train"], "max_staleness": run_bound})
    assert horizon_bound(alone) == run_bound  # one policy: train.max_staleness, as before


def test_keep_recovery_must_span_each_trained_policy_s_own_bound():
    found = problems(checkpoint={"keep_recovery": 1, "recovery_every": 2}, **worker(max_staleness=4))
    start = (
        "checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 2 = 2 must exceed policies.worker.max_staleness: 4"
    )
    assert any(problem.startswith(start) for problem in found), found


# --- policies ---------------------------------------------------------------------------------------------------------


class _LLM:
    def __init__(self, calls: list[str]) -> None:
        self.calls, self.version = calls, 0

    def set_version(self, version: int) -> None:
        self.calls.append(f"set_version {version}")
        self.version = version

    def update_sampling_client(self, sampling_client: Any, increment_version: bool = True) -> None:
        self.calls.append(f"sampler {sampling_client}")
        self.version += increment_version


def _policy(policy_id: str, calls: list[str] | None = None, *, url: str | None = None) -> Policy:
    info: Any = SimpleNamespace(llm=_LLM(calls if calls is not None else []))
    return Policy(PolicySpec(policy_id, model_name="m", renderer_name="r", tinker_base_url=url), info)


def test_a_policy_hands_each_sampler_it_publishes_to_its_subscribers_at_its_new_version():
    calls: list[str] = []
    policy = _policy("worker", calls)
    end = policy.subscribe(lambda sampler, version: calls.append(f"first {sampler} at {version}"))
    policy.subscribe(lambda sampler, version: calls.append(f"second {sampler} at {version}"))

    first: Any = "s1"
    policy.publish(first)
    assert (policy.sampling_client, policy.version) == ("s1", 1)
    assert calls == ["sampler s1", "first s1 at 1", "second s1 at 1"]
    end()
    calls.clear()
    launch: Any = "s0"
    policy.publish(launch, version=7)  # the launch's: at the version the start resumes at
    assert policy.version == 7 and calls == ["set_version 7", "sampler s0", "second s0 at 7"]


def test_the_policies_work_at_once_and_the_first_that_fails_names_its_policy_and_service():
    policies = PolicySet([_policy(MAIN), _policy("worker", url="http://worker:8000")])
    order: list[str] = []

    async def main_work() -> str:
        order.append("main starts")
        await asyncio.sleep(10)
        return "never"

    async def worker_work() -> str:
        order.append("worker starts")
        raise RuntimeError("503 Service Unavailable")

    with pytest.raises(RuntimeError, match="503") as raised:
        asyncio.run(policies.each({MAIN: main_work(), "worker": worker_work()}, doing="saving checkpoint final"))
    assert order == ["main starts", "worker starts"]
    assert raised.value.__notes__ == ["policy worker, on http://worker:8000: saving checkpoint final failed"]

    async def done(value: str) -> str:
        return value

    assert asyncio.run(policies.each({MAIN: done("a"), "worker": done("b")}, doing="x")) == {MAIN: "a", "worker": "b"}


def test_the_trained_policies_open_at_once_and_a_stop_before_one_opened_stops_the_start(monkeypatch):
    policies = PolicySet([_policy(MAIN), _policy("worker")])
    opened: list[str] = []

    async def open_policy(self: Policy, start: Any, *, stop: Any = None) -> bool:
        opened.append(self.id)
        return self.id == MAIN

    monkeypatch.setattr(Policy, "open", open_policy)
    start: Any = SimpleNamespace(policies={MAIN: None, "worker": None})
    assert asyncio.run(policies.open(start)) is False and opened == [MAIN, "worker"]


def test_a_trained_policy_loads_its_checkpoint_s_optimizer_state_when_it_asks():
    calls: list[str] = []

    class Service:
        async def create_training_client_from_state_with_optimizer_async(self, path: str) -> str:
            calls.append(f"with optimizer {path}")
            return "client"

        async def create_training_client_from_state_async(self, path: str) -> str:
            calls.append(f"weights {path}")
            return "client"

    path = "tinker://run/weights/000010"
    spec = PolicySpec("worker", model_name="m", renderer_name="r", load_checkpoint_path=path)
    client: Any = Service()
    asyncio.run(service.create_training_client(client, None, spec))
    asyncio.run(service.create_training_client(client, None, dataclasses.replace(spec, load_optimizer_state=True)))
    assert calls == [f"weights {path}", f"with optimizer {path}"]


def test_a_failure_note_names_the_policy_and_the_service_its_client_reaches(monkeypatch):
    spec = PolicySpec("worker", model_name="m", renderer_name="r")
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    assert failure_note(spec, "optim_step") == "policy worker, on hosted Tinker: optim_step failed"
    monkeypatch.setenv("TINKER_BASE_URL", "http://tinker:8000")
    assert (
        failure_note(spec, "optim_step") == "policy worker, on http://tinker:8000 (TINKER_BASE_URL): optim_step failed"
    )
    own = dataclasses.replace(spec, tinker_base_url="http://worker:8000")
    assert failure_note(own, "optim_step") == "policy worker, on http://worker:8000: optim_step failed"


def test_one_policy_s_work_among_several_trained_policies_names_it_when_it_fails():
    policies = PolicySet([_policy(MAIN), _policy("worker", url="http://worker:8000"), _policy("judge")])
    policies["judge"].spec = PolicySpec("judge", model_name="m", renderer_name="r", trainable=False)

    async def failing() -> str:
        raise RuntimeError("503 Service Unavailable")

    with pytest.raises(RuntimeError, match="503") as raised:
        asyncio.run(policies.each({"worker": failing()}, doing="saving checkpoint recovery-000001"))
    assert raised.value.__notes__ == ["policy worker, on http://worker:8000: saving checkpoint recovery-000001 failed"]

    alone = PolicySet([_policy(MAIN), policies["judge"]])  # one trained policy: its error is raised as it always was
    with pytest.raises(RuntimeError, match="503") as raised:
        asyncio.run(alone.each({MAIN: failing()}, doing="saving its weights"))
    assert not hasattr(raised.value, "__notes__")


# --- resume -----------------------------------------------------------------------------------------------------------


def test_a_resume_that_trains_a_frozen_policy_or_freezes_a_trained_one_is_refused():
    from platoon.train import identity

    trained = identity.run_identity(config(**worker()), [])
    frozen = identity.run_identity(config(policies={"worker": {**WORKER, "trainable": False}}, **BOUND), [])
    assert identity.policy_changes(trained, frozen) == [
        "policies.worker.trainable: the run's is True, and this start's False"
    ]
    assert identity.policy_changes(frozen, trained) == [
        "policies.worker.trainable: the run's is False, and this start's True"
    ]
