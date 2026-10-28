"""Rollouts that sample several policies (``agents.roles`` bound to a frozen policy): the group workflow's worker
request carries each other policy's binding, which agent samples which and the policies the run trains (protocol 3);
each policy's binding is a file of its own, which the trainer publishes before the first rollout and a worker follows
with a runtime of its own; and the conversion stamps each policy's rows with that policy's version.  A run of one
sampled policy sends protocol 2's request, as before."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import pickle
import sys
import time
from types import SimpleNamespace
from typing import Any

from fakes import WORKER_STUBS

import platoon.train.tinker  # noqa: F401
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.policy.binding import publish_frozen_bindings
from platoon.train.policy.spec import MAIN, AgentBindings
from platoon.train.workers import entrypoint
from platoon.train.workers import run as run_module
from platoon.train.workers.binding import FollowingSamplingClient, publish_binding, read_binding
from platoon.train.workers.protocol import (
    POLICY_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    ModelBinding,
    PolicyRolloutWorkerRequest,
)
from platoon.train.workers.supervisor import RolloutWorkerSupervisor
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

STUB_DIR = str(WORKER_STUBS)
if STUB_DIR not in sys.path:
    sys.path.insert(0, STUB_DIR)

import worker_stub_rollouts  # noqa: E402


def _info(model: str, version: int) -> Any:
    llm = SimpleNamespace(version=version, model_name=model, max_tokens=4096, temperature=1.0, top_k=-1, top_p=1.0)
    return SimpleNamespace(llm=llm, model_name=model, renderer_name="qwen3_instruct", renderer_kwargs={})


def _workflow(tmp_path, *, isolation: Any = "inprocess", judge: bool = True) -> GroupRolloutWorkflow:
    config = WorkflowConfig(
        group_size=2, rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")), rollout_isolation=isolation
    )
    rollout: Any = worker_stub_rollouts.varied_reward_rollout
    workflow = GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=config,
        model_info=_info("student", 3),
        log_path=str(tmp_path / "logs"),
    )
    if judge:
        workflow.sampled_policies = {"judge": _info("judge-model", 0)}
        workflow.agents = AgentBindings(default=MAIN, roles={"judge": "judge"})
        workflow.trained_policies = frozenset({MAIN})
    return workflow


def test_a_rollout_of_one_policy_is_protocol_2_s_request_as_before(tmp_path):
    request = _workflow(tmp_path, judge=False)._build_worker_request({"task_id": "t"}, 0)
    assert type(request).__name__ == "RolloutWorkerRequest" and request.protocol_version == PROTOCOL_VERSION


def test_a_rollout_of_several_policies_carries_each_one_s_binding(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    for isolation in ("inprocess", "process"):
        workflow = _workflow(tmp_path, isolation=isolation)
        request = workflow._build_worker_request({"task_id": "t"}, 0)

        assert isinstance(request, PolicyRolloutWorkerRequest)
        assert request.protocol_version == POLICY_PROTOCOL_VERSION
        assert (request.model.model_name, request.model.checkpoint_version) == ("student", 3)
        [(policy, binding)] = request.policies.items()
        assert (policy, binding.model_name, binding.checkpoint_version) == ("judge", "judge-model", 0)
        assert (request.agents, request.trained, request.tag_datums) == (workflow.agents, {MAIN}, False)
        if isolation == "process":  # a worker follows each policy's binding at its own file
            assert os.path.basename(request.model.binding_path) == "model_binding.pkl"
            assert os.path.basename(binding.binding_path) == "model_binding.judge.pkl"
        else:
            assert binding.binding_path == request.model.binding_path == ""
    warm = workflow._warm_up_request()  # the fork server builds each policy's renderer ahead
    assert isinstance(warm, PolicyRolloutWorkerRequest)
    assert list(warm.policies) == ["judge"]


def test_each_policy_s_binding_is_published_to_its_own_file(tmp_path):
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path))
    supervisor.publish_binding("default sampler", 3)
    supervisor.publish_binding("judge sampler", 0, policy="judge")

    assert read_binding(str(tmp_path / "model_binding.pkl")) == ("default sampler", 3)
    assert read_binding(str(tmp_path / "model_binding.judge.pkl")) == ("judge sampler", 0)
    # A day later, a new supervisor's sweep keeps every policy's binding, which its workers read.
    old = time.time() - 2 * 24 * 3600
    for name in ("model_binding.pkl", "model_binding.judge.pkl"):
        os.utime(tmp_path / name, (old, old))
    RolloutWorkerSupervisor(scratch_dir=str(tmp_path))
    assert {"model_binding.pkl", "model_binding.judge.pkl"} <= set(os.listdir(tmp_path))


def test_the_fork_server_builds_every_sampled_policy_s_renderer_ahead(tmp_path):
    supervisor = RolloutWorkerSupervisor(scratch_dir=str(tmp_path))
    request = _workflow(tmp_path)._warm_up_request()

    models = [step["kwargs"]["model_name"] for step in supervisor._fork_server_prepare(request)]
    assert models == ["student", "judge-model"]


def test_the_trainer_publishes_the_frozen_policies_bindings_before_the_first_rollout():
    published: list[tuple] = []
    judge = SimpleNamespace(id="judge", sampling_client="judge sampler", version=0)
    # The agents' default, main, is not the first policy their sampling lists here: it is named, not placed.
    policies: Any = SimpleNamespace(sampled=lambda: [judge, SimpleNamespace(id=MAIN)], agents=AgentBindings(MAIN))

    def publish(*args: Any, **kwargs: Any) -> None:
        published.append((args, kwargs))

    publish_frozen_bindings(policies, "workflow", publish, required=True)
    assert published == [(("workflow", "judge sampler", 0), {"required": True, "policy": "judge"})]


def test_an_evaluation_publishes_the_frozen_bindings_of_its_state_s_policies_if_it_has_them(monkeypatch):
    from platoon.train.evaluation.loop import EvaluationLoop, Publication

    published: list[tuple] = []
    trainer: Any = SimpleNamespace(
        config=SimpleNamespace(train=SimpleNamespace(label_by=None)),
        eval_dataset=None,
        _publish_worker_binding=lambda *args, **kwargs: published.append((args, kwargs)),
    )
    workflow: Any = SimpleNamespace(rollout_isolation="process")
    main_sampler: Any = "main sampler"
    publication = Publication(updated=asyncio.Event(), version=lambda: 1, sampling_client=lambda: main_sampler)

    async def no_groups() -> tuple[int, dict]:
        return 0, {}

    def evaluated(state: Any) -> list[tuple]:
        published.clear()
        loop = EvaluationLoop(trainer, state, workflow, publication)
        monkeypatch.setattr(loop, "_run_groups", no_groups)
        asyncio.run(loop.evaluate(0))
        return list(published)

    logger = SimpleNamespace(log=lambda **kwargs: None)
    # A state a test made as platoon 0.4.0 did, without the run's policies: main's binding alone, as the train loop.
    state: Any = SimpleNamespace(train_step=1, stats_logger=logger, evaluations_path=None)
    assert evaluated(state) == [((workflow, "main sampler", 1), {})]
    judge = SimpleNamespace(id="judge", sampling_client="judge sampler", version=0)
    state.policies = SimpleNamespace(sampled=lambda: [SimpleNamespace(id=MAIN), judge], agents=AgentBindings(MAIN))
    assert evaluated(state) == [
        ((workflow, "main sampler", 1), {}),
        ((workflow, "judge sampler", 0), {"required": False, "policy": "judge"}),
    ]


def test_the_group_workflow_binds_each_policy_to_its_own_file(tmp_path):
    workflow = _workflow(tmp_path, isolation="process")
    calls: list[tuple] = []
    supervisor: Any = SimpleNamespace(publish_binding=lambda *args, **kwargs: calls.append((args, kwargs)))
    workflow._supervisor = supervisor

    workflow.publish_model_binding("sampler", 4)
    workflow.publish_model_binding("judge sampler", 0, policy="judge")
    assert calls == [(("sampler", 4), {}), (("judge sampler", 0), {"policy": "judge"})]


class _Stub:
    async def sample_async(self, **kwargs: Any) -> None:
        return None


def _request(tmp_path) -> Any:
    """A worker's request of several policies, its bindings at files in ``tmp_path``."""

    request = _workflow(tmp_path)._build_worker_request({"task_id": "t"}, 0)
    model = ModelBinding(
        str(tmp_path / "model_binding.pkl"), b"", 0.0, "student", "qwen3_instruct", checkpoint_version=3
    )
    judge = ModelBinding(str(tmp_path / "model_binding.judge.pkl"), b"", 0.0, "judge-model", "qwen3_instruct")
    return dataclasses.replace(request, model=model, policies={"judge": judge})


def test_a_worker_serves_each_policy_from_its_own_binding(tmp_path, monkeypatch):
    request = _request(tmp_path)
    publish_binding(_Stub(), 3, request.model.binding_path)
    publish_binding(_Stub(), 0, request.policies["judge"].binding_path)
    built: list[str] = []

    def following(binding: ModelBinding) -> Any:
        built.append(binding.binding_path)
        sampler = FollowingSamplingClient(binding.binding_path, poll_seconds=0)
        return SimpleNamespace(gateway=SimpleNamespace(sampler=sampler, served=[]))

    monkeypatch.setattr(entrypoint, "_following_llm", following)
    llm = following(request.model)
    llm.gateway.serve_policies = lambda policy, *, runtimes, roles: llm.gateway.served.append((policy, runtimes, roles))
    entrypoint._serve_policies(request, llm)

    [(policy, runtimes, roles)] = llm.gateway.served
    assert (policy, list(runtimes), roles) == (MAIN, ["judge"], {"judge": "judge"})
    assert built == [request.model.binding_path, request.policies["judge"].binding_path]


def test_the_conversion_stamps_each_policy_s_rows_with_its_own_version(tmp_path, monkeypatch):
    request = _request(tmp_path)
    paths = {MAIN: request.model.binding_path, "judge": request.policies["judge"].binding_path}
    publish_binding(_Stub(), 3, paths[MAIN])
    publish_binding(_Stub(), 0, paths["judge"])
    followers = {policy: FollowingSamplingClient(path, poll_seconds=0) for policy, path in paths.items()}
    runtimes = {MAIN: SimpleNamespace(sampler=followers[MAIN]), "judge": SimpleNamespace(sampler=followers["judge"])}
    llm = SimpleNamespace(
        sampling_client=followers[MAIN], gateway=SimpleNamespace(runtime=lambda policy: runtimes[policy or MAIN])
    )
    seen: dict = {}

    def convert(**kwargs: Any) -> TrajectoryCollectionResult:
        seen.update(kwargs)
        return _converted()

    async def rollout(task: Task, config: RolloutConfig) -> dict:
        publish_binding(_Stub(), 5, paths[MAIN])  # main published twice before its first sample
        await followers[MAIN].sample_async()
        await followers["judge"].sample_async()
        return {"trajectories": {"root": {"steps": [{}]}}}

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", convert)
    asyncio.run(run_module.run_rollout_request(request, llm, rollout_fn=rollout))

    policies = seen["policies"]
    assert (policies.default, dict(policies.versions), policies.trained) == (MAIN, {MAIN: 5, "judge": 0}, {MAIN})
    assert pickle.loads(pickle.dumps(request)).agents == request.agents  # the request crosses to the worker


def _converted() -> TrajectoryCollectionResult:
    stats = TrajectoryStats(
        trajectory_id="root", reward=1.0, num_steps=1, num_input_tokens=1, num_output_tokens=1, num_datums=0
    )
    return TrajectoryCollectionResult(datums=[], task_reward=1.0, trajectory_stats=[stats], root_rewards_dict={})


def test_a_rollout_of_one_policy_converts_as_before(tmp_path, monkeypatch):
    seen: dict = {}

    def convert(**kwargs: Any) -> TrajectoryCollectionResult:
        seen.update(kwargs)
        return _converted()

    async def rollout(task: Task, config: RolloutConfig) -> dict:
        return {"trajectories": {"root": {"steps": [{}]}}}

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", convert)
    request = _workflow(tmp_path, judge=False)._build_worker_request({"task_id": "t"}, 0)
    asyncio.run(run_module.run_rollout_request(request, None, rollout_fn=rollout))
    assert (seen["policies"], seen["checkpoint_version"]) == (None, 3)
