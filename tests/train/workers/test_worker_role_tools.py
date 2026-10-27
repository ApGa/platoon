"""A run whose agent roles have limited tools (``agents.roles.<role>.tools``): the group workflow gets the lists from
its setup, its worker requests carry them (protocol 3's ``role_tools``, sent only then), a rollout worker's gateway
limits each role's tools from them before its rollout runs, and the trainer's gateway, which rollouts in its process
sample, from the config.
Such a request converts as a rollout of one policy, without tags.  A run without lists sends protocol 2's request."""

from __future__ import annotations

import asyncio
import pickle
import sys
from types import SimpleNamespace
from typing import Any, cast

from fakes import WORKER_STUBS

import platoon.train.tinker  # noqa: F401
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train import launch
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.policy.spec import MAIN
from platoon.train.workers import entrypoint
from platoon.train.workers import run as run_module
from platoon.train.workers.protocol import POLICY_PROTOCOL_VERSION, PROTOCOL_VERSION, PolicyRolloutWorkerRequest
from platoon.train.workflow import protocol
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import build_config
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

STUB_DIR = str(WORKER_STUBS)
if STUB_DIR not in sys.path:
    sys.path.insert(0, STUB_DIR)

import worker_stub_rollouts  # noqa: E402

ROLE_TOOLS = {"reviewer": ("read_*", "view"), "judge": ()}


def _workflow(tmp_path, *, isolation: Any = "inprocess") -> GroupRolloutWorkflow:
    config = WorkflowConfig(
        group_size=2, rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")), rollout_isolation=isolation
    )
    llm = SimpleNamespace(version=3, model_name="student", max_tokens=4096, temperature=1.0, top_k=-1, top_p=1.0)
    info = SimpleNamespace(llm=llm, model_name="student", renderer_name="qwen3_instruct", renderer_kwargs={})
    rollout: Any = worker_stub_rollouts.varied_reward_rollout  # named by module, as a worker resolves it
    return GroupRolloutWorkflow(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=config,
        model_info=cast(Any, info),
        log_path=str(tmp_path / "logs"),
    )


def test_the_group_workflow_takes_the_roles_tools_from_its_setup():
    async def rollout(task: Any, rollout_config: Any) -> dict:
        raise AssertionError("not run")

    given: dict[str, Any] = {
        "phase": "train",
        "rollout_fn": rollout,
        "get_task_fn": lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        "config": WorkflowConfig(group_size=2),
        "entry_configs": None,
        "policies": {MAIN: SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0))},
        "log_path": "logs",
        "stats_scope": "train",
        "filter_errors": False,
        "reward_processor": None,
    }
    assert dict(GroupRolloutWorkflow.from_setup(protocol.WorkflowSetup(**given)).role_tools) == {}
    built = GroupRolloutWorkflow.from_setup(protocol.WorkflowSetup(**given, role_tools=ROLE_TOOLS))
    assert dict(built.role_tools) == ROLE_TOOLS


def test_a_run_without_role_tools_sends_protocol_2_s_request_as_before(tmp_path):
    request = _workflow(tmp_path)._build_worker_request({"task_id": "t"}, 0)
    assert type(request).__name__ == "RolloutWorkerRequest" and request.protocol_version == PROTOCOL_VERSION


def test_each_request_of_a_run_with_role_tools_carries_them(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    for isolation in ("inprocess", "process"):
        workflow = _workflow(tmp_path, isolation=isolation)
        workflow.role_tools = ROLE_TOOLS
        request = workflow._build_worker_request({"task_id": "t"}, 0)

        assert isinstance(request, PolicyRolloutWorkerRequest)
        assert request.protocol_version == POLICY_PROTOCOL_VERSION
        assert request.role_tools == ROLE_TOOLS
        # One policy, and no tags: the conversion is a rollout of one policy's (below).
        assert (request.tag_datums, request.policies, request.agents) == (False, {}, None)
        assert pickle.loads(pickle.dumps(request)).role_tools == ROLE_TOOLS  # it crosses to the worker


def test_a_worker_s_gateway_limits_the_roles_tools_of_its_request(tmp_path):
    limited: list[Any] = []
    llm = SimpleNamespace(gateway=SimpleNamespace(limit_tools=limited.append))
    workflow = _workflow(tmp_path)

    entrypoint._limit_tools(workflow._build_worker_request({"task_id": "t"}, 0), llm)  # protocol 2: nothing to limit
    workflow.role_tools = ROLE_TOOLS
    request = workflow._build_worker_request({"task_id": "t"}, 0)
    entrypoint._limit_tools(request, None)  # no model bound: no gateway
    entrypoint._limit_tools(request, llm)

    assert limited == [ROLE_TOOLS]


def test_a_worker_limits_its_gateway_s_tools_before_its_rollout_runs(tmp_path, monkeypatch):
    workflow = _workflow(tmp_path)
    workflow.role_tools = ROLE_TOOLS
    request = workflow._build_worker_request({"task_id": "t"}, 0)
    steps: list[Any] = []
    llm = SimpleNamespace(gateway=SimpleNamespace(limit_tools=lambda roles: steps.append(("limit_tools", roles))))
    for step in ("_become_worker_process", "_arm_startup_deadline", "_import_environment_packages", "_print_header"):
        monkeypatch.setattr(entrypoint, step, lambda *args, **kwargs: None)
    monkeypatch.setattr(entrypoint, "_load_request", lambda path: request)
    monkeypatch.setattr(entrypoint, "_import_stack", lambda: False)
    monkeypatch.setattr(entrypoint, "_build_llm", lambda model: llm)
    monkeypatch.setattr(entrypoint, "_run", lambda request, llm, **kwargs: steps.append("run") or 0)

    assert entrypoint.main([str(tmp_path / "request.pkl")]) == 0
    assert steps == [("limit_tools", ROLE_TOOLS), "run"]


def test_a_rollout_with_role_tools_converts_as_one_policy_s(tmp_path, monkeypatch):
    seen: dict = {}

    def convert(**kwargs: Any) -> TrajectoryCollectionResult:
        seen.update(kwargs)
        stats = TrajectoryStats(
            trajectory_id="root", reward=1.0, num_steps=1, num_input_tokens=1, num_output_tokens=1, num_datums=0
        )
        return TrajectoryCollectionResult(datums=[], task_reward=1.0, trajectory_stats=[stats], root_rewards_dict={})

    async def rollout(task: Task, config: RolloutConfig) -> dict:
        return {"trajectories": {"root": {"steps": [{}]}}}

    monkeypatch.setattr(run_module, "get_train_data_for_trajectory_collection", convert)
    workflow = _workflow(tmp_path)
    workflow.role_tools = ROLE_TOOLS
    request = workflow._build_worker_request({"task_id": "t"}, 0)
    asyncio.run(run_module.run_rollout_request(request, None, rollout_fn=rollout))

    assert (seen["policies"], seen["tag_datums"], seen["checkpoint_version"]) == (None, False, 3)


def test_the_trainer_s_gateway_limits_the_roles_tools_of_its_config(monkeypatch):
    limited: list[Any] = []
    monkeypatch.setattr(launch, "default_gateway", lambda: SimpleNamespace(limit_tools=limited.append))
    settings = {"train": {"model_name": "m", "renderer_name": "qwen3_instruct"}, "eval": {"every": 0}, "log_path": "l"}

    launch._limit_tools(build_config(PlatoonTinkerRLTrainerConfig, settings))
    roles = {"reviewer": {"tools": ["read_*", "view"]}, "judge": {"tools": []}, "helper": {"policy": "main"}}
    launch._limit_tools(build_config(PlatoonTinkerRLTrainerConfig, {**settings, "agents": {"roles": roles}}))

    assert limited == [ROLE_TOOLS]  # nothing without a list; the roles that list theirs
