"""A group's settings (``GroupSettings``) and the switch between in-process rollouts and rollout workers.

A workflow config that lacks a setting reads ``WorkflowConfig``'s default, converted as the group's code uses it; the
group's code reads where its rollouts run under one name, ``GroupRolloutWorkflow._isolation``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.train.config_defs import StragglerPolicy, WorkflowConfig
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.settings import GroupSettings

# Each setting a config may lack, with the value (and type) a config without it reads.
DEFAULTS = {
    "min_successful_group_size": 1,
    "straggler_timeout_seconds": None,
    "straggler_quorum": None,
    "straggler_policies": {},
    "filter_message_actions": False,
    "depth_level_discount_gamma": None,
    "subagent_sampling": (1.0, 0),
    "worker_start_method": "forkserver",
    "worker_launch_concurrent_groups": 2,
    "worker_startup_budget_seconds": 300.0,
    "worker_finalize_grace_seconds": 120.0,
    "worker_unbounded_rollout_budget_seconds": 24 * 3600.0,
    "worker_parent_grace_seconds": 60.0,
    "worker_termination_grace_seconds": 10.0,
    "worker_rss_soft_limit_bytes": 24 * 1024**3,
    "worker_mem_available_floor_bytes": 0,
    "worker_binding_poll_seconds": 5.0,
    "worker_python_env_isolation": False,
}


def _partial(**settings: Any) -> Any:
    """A workflow config with only ``settings``, as tests build one."""

    return SimpleNamespace(**settings)


@pytest.mark.parametrize("name", sorted(DEFAULTS))
def test_a_config_without_a_setting_reads_the_workflow_config_default(name):
    value = getattr(GroupSettings(_partial()), name)
    assert value == DEFAULTS[name]
    assert type(value) is type(DEFAULTS[name])
    assert value == getattr(GroupSettings(WorkflowConfig()), name)


def test_a_config_s_values_are_converted_as_the_group_uses_them():
    config = _partial(
        worker_startup_budget_seconds="12.5",
        worker_binding_poll_seconds=2,
        worker_launch_concurrent_groups="3",
        worker_rss_soft_limit_bytes=7.0,
        worker_start_method="exec",
        worker_python_env_isolation=1,
    )
    settings = GroupSettings(config)
    assert settings.worker_startup_budget_seconds == 12.5
    assert type(settings.worker_binding_poll_seconds) is float
    assert settings.worker_launch_concurrent_groups == 3
    assert type(settings.worker_rss_soft_limit_bytes) is int
    assert settings.worker_start_method == "exec"
    assert settings.worker_python_env_isolation is True


def test_a_label_s_straggler_policy_replaces_the_group_s():
    config = WorkflowConfig(
        straggler_timeout_seconds=30.0,
        straggler_quorum=3,
        straggler_policies={"slow": StragglerPolicy(timeout_seconds=90.0)},
    )
    settings = GroupSettings(config)
    assert settings.straggler_policy("slow") == (90.0, None)
    assert settings.straggler_policy("other") == (30.0, 3)
    assert settings.straggler_policy(None) == (30.0, 3)
    assert GroupSettings(_partial()).straggler_policy("slow") == (None, None)


class _Supervisor:
    def __init__(self) -> None:
        self.warmed: list = []
        self.published: list = []

    async def warm(self, request) -> None:
        self.warmed.append(request)

    def publish_binding(self, sampling_client, version: int) -> None:
        self.published.append((sampling_client, version))


def _workflow(tmp_path, isolation: str, cls: type = GroupRolloutWorkflow, **config: Any) -> Any:
    """A workflow built with ``__new__``, as a custom workflow may build it, with a recording supervisor."""

    workflow: Any = object.__new__(cls)
    workflow.rollout_isolation = isolation
    workflow.stats_scope = "group-settings-test"
    workflow.config = _partial(**config)
    workflow.model_info = SimpleNamespace(llm=None, model_name="m")
    workflow._ipc_dir_cache = str(tmp_path)
    workflow._supervisor = _Supervisor()
    return workflow


def test_rollout_workers_are_warmed_up_and_follow_the_policy_only_in_process_mode(tmp_path):
    workflow = _workflow(tmp_path, "process")
    asyncio.run(workflow.warm_up())
    workflow.publish_model_binding("client", 3)
    [request] = workflow._supervisor.warmed
    assert request.request_id == "group-settings-test-warm-up"
    assert request.startup_budget_seconds == 300.0
    assert request.model.binding_path == str(tmp_path / "model_binding.pkl")
    assert workflow._supervisor.published == [("client", 3)]

    in_process = _workflow(tmp_path, "inprocess")
    exec_workers = _workflow(tmp_path, "process", worker_start_method="exec")
    for other in (in_process, exec_workers):
        asyncio.run(other.warm_up())
        assert other._supervisor.warmed == []
    in_process.publish_model_binding("client", 4)
    assert in_process._supervisor.published == []
    assert in_process._model_binding(0).binding_path == ""


async def _holds_the_startup_gate(workflow: Any) -> bool:
    startup_pass = await workflow._enter_startup_gate(4)
    try:
        return startup_pass._semaphore is not None and startup_pass._timer is not None
    finally:
        startup_pass.release()


def test_only_process_mode_holds_the_startup_gate(tmp_path):
    assert asyncio.run(_holds_the_startup_gate(_workflow(tmp_path, "process", worker_startup_budget_seconds=20)))
    assert not asyncio.run(_holds_the_startup_gate(_workflow(tmp_path, "inprocess")))


class _ProcessWorkflow(GroupRolloutWorkflow):
    @property
    def _isolation(self) -> str:
        return "process"


def test_one_name_switches_every_reader(tmp_path):
    """A workflow whose ``_isolation`` says ``process`` runs its rollout workers whatever its ``rollout_isolation``."""

    workflow = _workflow(tmp_path, "inprocess", cls=_ProcessWorkflow)
    asyncio.run(workflow.warm_up())
    workflow.publish_model_binding("client", 5)
    assert len(workflow._supervisor.warmed) == 1
    assert workflow._supervisor.published == [("client", 5)]
    assert workflow._model_binding(0).binding_path == str(tmp_path / "model_binding.pkl")
    assert asyncio.run(_holds_the_startup_gate(workflow))
