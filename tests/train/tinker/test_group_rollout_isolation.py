"""Group semantics must not change when rollouts move into their own processes.

Several of these launch four real worker processes, each of which imports the
tinker/torch stack, so the file takes a couple of minutes.  That cost buys the
only end-to-end evidence that a rollout really does run in its own interpreter.

The blast radius is the point.  A worker death is a *member* failure -- decided
by ``min_successful_group_size`` like any other -- never a group failure and
never a trainer failure.  These tests exercise that with real worker processes
(slow, deliberately) and pin the parity between the two isolation modes.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest
import torch

import platoon.train.tinker  # noqa: F401
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.workers.protocol import RefResolutionError
from platoon.train.workflow import context as context_module
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
from platoon.train.workflow.start_limit import StartLimit, held_slot
from platoon.utils.stats_tracker import get as get_tracker

STUB_DIR = os.path.dirname(os.path.abspath(__file__))
if STUB_DIR not in sys.path:
    sys.path.insert(0, STUB_DIR)

import worker_stub_rollouts  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_on_the_child_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("OPENHANDS_SUPPRESS_BANNER", "1")


def _model_info():
    llm = SimpleNamespace(
        version=3,
        model_name="",  # no model bound: workers start no gateway session
        context_window_length=None,
        max_tokens=4096,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
        seed=None,
    )
    return SimpleNamespace(
        llm=llm,
        # Empty on purpose: a binding with no model name registers no sampling
        # provider, so these tests contact no gateway and load no tokenizer.
        model_name="",
        base_url="None",
        api_key="None",
        renderer_name="",
        renderer_kwargs={},
    )


def _workflow(tmp_path, scope, *, isolation, rollout_fn=None, **config_kwargs):
    config_kwargs.setdefault("min_successful_group_size", 1)
    config = WorkflowConfig(
        group_size=4,
        rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts"), timeout=180),
        filter_zero_advantage_datums=False,
        leave_one_out_baseline=True,
        rollout_isolation=isolation,
        worker_startup_budget_seconds=240.0,
        worker_finalize_grace_seconds=60.0,
        worker_parent_grace_seconds=30.0,
        worker_termination_grace_seconds=1.0,
        **config_kwargs,
    )
    return group_module.GroupRolloutWorkflow(
        rollout_fn=rollout_fn or worker_stub_rollouts.varied_reward_rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=config,
        model_info=_model_info(),
        log_path=str(tmp_path / "logs"),
        stats_scope=scope,
    )


def _scalars(scope):
    exported = get_tracker(scope).export(reset=True)
    return {key.removeprefix(f"{scope}/"): value for key, value in exported.items()}


def _members_with(scalars, key) -> float:
    """How many members hit a counter.

    The counters are emitted on every member as 0 or 1 and the tracker averages
    them, so the count is the mean times the denominator.
    """

    return scalars.get(key, 0.0) * scalars.get(f"{key}__count", 0.0)


def _advantages(output):
    return sorted(round(float(datum.loss_fn_inputs["advantages"].to_torch().sum().item()), 6) for datum in output)


def _run(workflow, task_id="task"):
    return asyncio.run(workflow.arun_episode({"task_id": task_id}))


def test_process_mode_matches_inprocess_mode_datum_for_datum(tmp_path, monkeypatch):
    """Parity gate: same fakes, same group result, different processes."""

    monkeypatch.setenv("STUB_SLOT_DIR", str(tmp_path / "slots-inprocess"))
    inprocess = _workflow(tmp_path / "a", "iso_inprocess", isolation="inprocess")
    in_output = _run(inprocess)
    _scalars("iso_inprocess")

    monkeypatch.setenv("STUB_SLOT_DIR", str(tmp_path / "slots-process"))
    process = _workflow(tmp_path / "b", "iso_process", isolation="process")
    out_output = _run(process)
    scalars = _scalars("iso_process")

    assert len(in_output) == 4
    assert len(out_output) == len(in_output)
    assert out_output.observed_rollouts == in_output.observed_rollouts == 4
    assert out_output.requested_rollouts == in_output.requested_rollouts == 4
    assert out_output.workload.model_calls == in_output.workload.model_calls
    assert out_output.workload.postmerge_datums == in_output.workload.postmerge_datums
    # Leave-one-out centering over the same reward multiset must give the same
    # advantages regardless of which process produced each member.
    assert _advantages(out_output) == _advantages(in_output)
    # R3: the routing handle survived the process boundary.
    for datum in out_output:
        handle = datum.loss_fn_inputs["routing_handle"].to_torch()
        assert handle.dtype == torch.int64 and int(handle.item()) >= 1000
    assert scalars.get("rollout_worker_routing_id_coverage") == 1.0

    logs = tmp_path / "b" / "logs" / "rollouts" / "iso_process" / "3" / "workers"
    log_files = list(logs.glob("*.log"))
    assert len(log_files) == 4
    headers = [
        line
        for path in log_files
        for line in path.read_text().splitlines()
        if line.startswith("[rollout-worker] request=")
    ]
    assert len(headers) == 4
    pids = {header.split("pid=")[1].split()[0] for header in headers}
    # Every rollout ran in its own interpreter, and no interpreter was reused.
    assert len(pids) == 4


def test_one_member_s_launch_failure_costs_only_that_member(tmp_path, monkeypatch):
    """GATE: an escape here would cost the whole group, not one rollout.

    ``_run_group_rollouts`` re-raises through ``task.result()``, so an
    exception from request building would discard the three members that
    completed and would have satisfied ``min_successful_group_size``.
    """

    from platoon.train.workers import supervisor as supervisor_module

    monkeypatch.setenv("STUB_SLOT_DIR", str(tmp_path / "slots"))
    real_dump = supervisor_module.dump_request

    def failing_for_rollout_one(request, path):
        if "-r1-" in request.request_id:
            raise OSError(28, "No space left on device")
        return real_dump(request, path)

    monkeypatch.setattr(supervisor_module, "dump_request", failing_for_rollout_one)
    workflow = _workflow(tmp_path, "iso_launch_fail", isolation="process")
    output = _run(workflow)
    scalars = _scalars("iso_launch_fail")

    assert output.requested_rollouts == 4
    assert output.observed_rollouts == 3
    assert len(output) == 3
    assert _members_with(scalars, "rollout_worker_launch_failed") == 1.0
    assert _members_with(scalars, "rollout_worker_launch_retried") == 1.0
    assert scalars["rollout_worker_launch_failed__count"] == 4.0


def test_a_segfaulting_member_does_not_stop_its_siblings(tmp_path, monkeypatch):
    """Crash containment: a segfault kills only its worker; in-process it would kill the trainer."""

    monkeypatch.setenv("STUB_SLOT_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("STUB_SEGFAULT_ROLLOUT", "1")
    workflow = _workflow(tmp_path, "iso_segfault", isolation="process")

    # One member segfaults; the rest run the normal rollout.
    original = workflow.rollout_fn_ref
    calls = {"n": 0}
    real_build = workflow._build_worker_request

    def build(data, rollout_number):
        request = real_build(data, rollout_number)
        if rollout_number == 2:
            calls["n"] += 1
            return type(request)(
                **{
                    **request.__dict__,
                    "rollout_fn_ref": "worker_stub_rollouts:segfault_rollout",
                }
            )
        return request

    monkeypatch.setattr(workflow, "_build_worker_request", build)
    output = _run(workflow)
    scalars = _scalars("iso_segfault")

    assert original == "worker_stub_rollouts:varied_reward_rollout"
    assert calls["n"] == 1
    assert output.requested_rollouts == 4
    assert output.observed_rollouts == 3
    assert len(output) == 3
    assert _members_with(scalars, "rollout_worker_crashed") == 1.0
    assert _members_with(scalars, "rollout_worker_hard_timeout") == 0.0


def test_a_spinning_member_is_killed_by_the_straggler_policy(tmp_path, monkeypatch):
    """Straggler cancellation kills the worker.

    In-process, a cancelled straggler abandons its coroutine while the runaway
    work keeps a thread and a core forever.
    """

    monkeypatch.setenv("STUB_SLOT_DIR", str(tmp_path / "slots"))
    workflow = _workflow(
        tmp_path,
        "iso_straggler",
        isolation="process",
        straggler_timeout_seconds=10.0,
        straggler_quorum=3,
        min_successful_group_size=2,
    )
    real_build = workflow._build_worker_request

    def build(data, rollout_number):
        request = real_build(data, rollout_number)
        if rollout_number == 3:
            return type(request)(**{**request.__dict__, "rollout_fn_ref": "worker_stub_rollouts:spin_rollout"})
        return request

    monkeypatch.setattr(workflow, "_build_worker_request", build)
    output = _run(workflow)
    scalars = _scalars("iso_straggler")

    assert scalars.get("straggler_cancelled") == 1.0
    assert output.requested_rollouts == 4
    assert output.observed_rollouts == 3
    # The group still trains: a killed member is one member, not the group.
    assert len(output) == 3


def test_construction_rejects_a_callable_a_worker_could_not_import(tmp_path):
    def closure_reward_processor(trajectory):
        return trajectory["reward"], {}

    with pytest.raises(RefResolutionError):
        _workflow(tmp_path, "iso_refs", isolation="process")._build_ref(
            closure_reward_processor, what="reward_processor"
        )

    # ... and in-process mode is unaffected: an unnameable callable is only
    # recorded, because in-process behaviour is unchanged.
    inprocess = _workflow(tmp_path, "iso_refs2", isolation="inprocess")
    assert inprocess._build_ref(closure_reward_processor, what="reward_processor") is None


def test_construction_resolves_a_main_defined_reward_processor(tmp_path, monkeypatch):
    """A train script may define ``reward_processor`` in the script itself."""

    import importlib.util
    import types

    reward_processor = worker_stub_rollouts.doubled_reward
    shim = types.ModuleType("__main__")
    shim.__spec__ = importlib.util.find_spec("worker_stub_rollouts")
    monkeypatch.setitem(sys.modules, "__main__", shim)

    main_scoped = types.FunctionType(reward_processor.__code__, reward_processor.__globals__, "doubled_reward")
    main_scoped.__module__ = "__main__"
    main_scoped.__qualname__ = "doubled_reward"

    workflow = group_module.GroupRolloutWorkflow(
        rollout_fn=worker_stub_rollouts.varied_reward_rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=WorkflowConfig(group_size=1, rollout_isolation="process"),
        model_info=_model_info(),
        stats_scope="iso_main_ref",
        reward_processor=main_scoped,
    )
    assert workflow.reward_processor_ref == "worker_stub_rollouts:doubled_reward"
    assert workflow._reward_processor_is_default is False


@pytest.mark.parametrize("limit", [None, 3], ids=["no-start-limit", "start-limit-of-a-group"])
def test_group_atomic_admission_admits_whole_groups(tmp_path, limit):
    """Members of a group must start together, and groups must not overlap.

    A per-rollout readiness gate would inject start-time skew inside a group,
    and the straggler policy has no per-member reference point: it starts its
    tail-grace clock when the quorum settles.  The same rollout numbers would
    then deterministically start last and be cancelled having had a fraction of
    their budget, biasing the leave-one-out baseline.

    So also with a start limit of a group's size (``train.max_starting_rollouts``), each member's first model call
    following its readiness.
    """

    events: list[tuple[str, str]] = []

    class Recorder(group_module.GroupRolloutWorkflow):
        async def arun_episode_single(self, data, rollout_number=0):
            group = data["task_id"]
            events.append(("start", group))
            slot = context_module._group_ready_callback.get()
            await asyncio.sleep(0.15)
            events.append(("ready", group))
            if slot is not None:
                slot()
            if (start := held_slot()) is not None:
                start.first_call()
            await asyncio.sleep(0.15)
            events.append(("done", group))
            return output_module._SingleRolloutOutcome(
                result=None, workload=group_module.RolloutWorkload(), observed=False
            )

    config = WorkflowConfig(
        group_size=3,
        rollout_isolation="process",
        worker_launch_concurrent_groups=1,
        worker_startup_budget_seconds=30.0,
    )
    workflow = Recorder(
        rollout_fn=worker_stub_rollouts.varied_reward_rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=config,
        model_info=_model_info(),
        stats_scope="iso_admission",
    )
    workflow.start_limit = StartLimit(limit) if limit else None

    async def scenario():
        await asyncio.gather(
            workflow._run_group_rollouts({"task_id": "A"}),
            workflow._run_group_rollouts({"task_id": "B"}),
        )

    asyncio.run(scenario())

    starts = [group for kind, group in events if kind == "start"]
    # Every member of the first admitted group starts before any member of the
    # second: no interleaving.
    assert starts in (["A"] * 3 + ["B"] * 3, ["B"] * 3 + ["A"] * 3), starts
    first_group = starts[0]
    readies = [group for kind, group in events if kind == "ready"]
    # The permit is handed over at readiness, not at completion, so the second
    # group starts while the first is still running.
    first_group_done = events.index(("done", first_group))
    second_group_start = events.index(("start", readies[-1] if False else starts[-1]))
    assert second_group_start < first_group_done


def test_workflow_config_validates_the_isolation_keys():
    with pytest.raises(ValueError, match="rollout_isolation"):
        WorkflowConfig(rollout_isolation="threads")
    with pytest.raises(ValueError, match="worker_python_env_isolation"):
        WorkflowConfig(worker_python_env_isolation=True)
    with pytest.raises(ValueError, match="worker_startup_budget_seconds"):
        WorkflowConfig(worker_startup_budget_seconds=-1.0)
    with pytest.raises(ValueError, match="worker_launch_concurrent_groups"):
        WorkflowConfig(worker_launch_concurrent_groups=0)
    # The combination that IS allowed.
    config = WorkflowConfig(rollout_isolation="process", worker_python_env_isolation=True)
    assert config.worker_python_env_isolation is True


def test_worker_requests_carry_the_environments_packages(tmp_path):
    """A rollout worker imports the environments' packages the trainer hands its workflows (train.py)."""

    from platoon.train.launch import worker_package_kwargs

    workflow = _workflow(tmp_path, "iso_packages", isolation="inprocess")
    request = workflow._build_worker_request({"task_id": "task"}, 0)
    assert request.packages == ()

    explicit = group_module.GroupRolloutWorkflow(
        rollout_fn=worker_stub_rollouts.varied_reward_rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=workflow.config,
        model_info=_model_info(),
        packages=["my_project.envs"],
    )
    request = explicit._build_worker_request({"task_id": "task"}, 0)
    assert request.packages == ("my_project.envs",)

    config = SimpleNamespace(
        environments=[
            {"name": "a", "service": "python://my_project.envs:A", "package": "my_project.envs"},
            {"name": "b", "service": "python://other.envs:B"},
        ]
    )
    assert worker_package_kwargs(group_module.GroupRolloutWorkflow, config) == {"packages": ("my_project.envs",)}
    assert worker_package_kwargs(object, config) == {}  # a workflow of another kind gets none
