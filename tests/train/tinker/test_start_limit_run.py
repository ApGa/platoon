"""The start limit in the group workflow and the trainer (``train.max_starting_rollouts``; Tinker faked).

- A slot comes back when the rollout tree's first model call is issued, the root's or a subagent's, even one whose
  sampling never returns; under process isolation, when the rollout worker marks it; at the end of a rollout that fails
  before any call or makes none; and on its timer, the group's startup budget plus the grace.
- No deadlock with fewer slots than a group's members, under rolling starts and co-start, nor with groups nested in a
  member's rollout, which take no slots; a member asks for its slot only inside its lease block; a slot wait joins the
  member's start delay, and a member that waited is not cut before its own grace; cancellation and failures while
  waiting for or holding a slot leak nothing.
- Evaluation during training shares the limit; a ``k = 0`` run with a binding limit trains the same datums each step; a
  limit that never binds changes nothing; without the key, the rollout hosts are called as before and no metric is
  added.
- The startup gate still admits whole groups (``test_group_rollout_isolation.py``); the key's checks and hints; what a
  run says as it starts.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_dispatch_run import _Recording
from test_rollout_permits_run import _group, _logging_stats, _Members, _outcome, _until
from trainer_fakes import make_trainer, trainer_config

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.envs.service import EnvSetupError
from platoon.gateway.core import ChatRequest
from platoon.task import Task
from platoon.testing import ScriptedPolicy, make_gateway
from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.config_defs import TrainConfig, WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.leases import holding
from platoon.train.tinker import rollout_metrics
from platoon.train.tinker.rollout_metrics import current_timeline
from platoon.train.trainer import rl as rl_module
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.workflow import group as group_module
from platoon.train.workflow import members as members_module
from platoon.train.workflow.start_limit import StartLimit, held_slot
from platoon.utils.config import ConfigError
from platoon.utils.stats_tracker import get as get_tracker

STUB_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _fresh_monitors():
    """The client monitors are the process's, one per stats scope: each test starts and ends with none, so that no other
    test's start-limit metrics show here, nor these elsewhere."""

    rollout_metrics.reset_monitors_for_tests()
    yield
    rollout_metrics.reset_monitors_for_tests()


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT, SIGTERM, SIGUSR1 and SIGUSR2 handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _first_call() -> None:
    """What a rollout's host does as the rollout's first model call is issued: give its start slot back."""

    start = held_slot()
    if start is not None:
        start.first_call()


def _workflow(tmp_path: Path, rollout_fn: Any, scope: str, **config: Any) -> group_module.GroupRolloutWorkflow:
    """The group workflow itself, in process, its rollouts run by ``run_rollout_request``."""

    config.setdefault("filter_zero_advantage_datums", False)
    return group_module.GroupRolloutWorkflow(
        rollout_fn=rollout_fn,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=WorkflowConfig(rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")), **config),
        model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0)),
        log_path=str(tmp_path / "logs"),
        stats_scope=scope,
    )


class _HeldPolicy(ScriptedPolicy):
    """A scripted policy that answers its calls only once ``go`` is set, noting each call as it begins."""

    def __init__(self, events: list[tuple[str, int]]) -> None:
        super().__init__(lambda _prompt: "ok")
        self.events, self.begun = events, 0
        self.go = asyncio.Event()

    async def sample_async(self, *, prompt: Any, sampling_params: Any, num_samples: int = 1) -> Any:
        self.events.append(("call", self.begun))
        self.begun += 1
        await self.go.wait()
        return await super().sample_async(prompt=prompt, sampling_params=sampling_params, num_samples=num_samples)


# --- when a slot comes back ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("caller", ["root", "subagent"])
def test_in_process_a_slot_comes_back_when_the_first_model_call_is_issued_though_it_never_returns(tmp_path, caller):
    """One slot and a group of 3, each rollout making one model call through a real model gateway (its root's, or a
    subagent's when the root makes none) that samples only once all three calls have begun: each member's rollout
    starts only once the one before it has issued its call, so every slot came back at its call."""

    events: list[tuple[str, int]] = []
    policy = _HeldPolicy(events)
    gateway = make_gateway(policy)
    members = itertools.count()

    async def rollout(task, config):
        member = next(members)
        events.append(("rollout", member))
        root = gateway.open_agent(
            rollout_id=f"r{member}", trajectory_id="root", interactions={}, timeline=current_timeline()
        )
        agent = root if caller == "root" else gateway.open_subagent(root, role="helper")
        await gateway.chat(agent, ChatRequest.from_params([{"role": "user", "content": "hi"}], {}))
        return {"trajectories": {}}

    workflow = _workflow(tmp_path, rollout, f"test_sl_first_call_{caller}", group_size=3)
    workflow.start_limit = limit = StartLimit(1)

    async def run():
        group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t"}))
        await _until(lambda: policy.begun == 3)
        policy.go.set()
        return await asyncio.wait_for(group, 10)

    asyncio.run(run())
    assert events == [("rollout", 0), ("call", 0), ("rollout", 1), ("call", 1), ("rollout", 2), ("call", 2)]
    assert limit.releases == {"first_call": 3} and limit.held == 0
    get_tracker(f"test_sl_first_call_{caller}").export(reset=True)


def test_a_rollout_that_fails_before_any_call_or_makes_none_gives_its_slot_back_at_its_end(tmp_path):
    """One slot, a group of 2: member 0's environment cannot open its session, member 1's rollout calls no model (as
    ``platoon-oracle`` does). Each slot comes back as its rollout ends."""

    members = itertools.count()

    async def rollout(task, config):
        if next(members) == 0:
            raise EnvSetupError("sandbox_start", "the sandbox did not start")
        return {"trajectories": {}}

    workflow = _workflow(tmp_path, rollout, "test_sl_fallbacks", group_size=2)
    workflow.start_limit = limit = StartLimit(1)
    asyncio.run(asyncio.wait_for(workflow.arun_episode({"task_id": "t"}), 10))
    assert limit.releases == {"end": 2} and limit.held == 0
    get_tracker("test_sl_fallbacks").export(reset=True)


def test_a_member_s_slot_timer_is_its_group_s_startup_budget_plus_the_grace(tmp_path, monkeypatch):
    """One entry keeps the default budget (300 s) and another sets 900 s: their groups' slots come back on timers of
    600 and 1200 s without a model call."""

    timers: list[float] = []

    class Recording(StartLimit):
        def slot(self, *, timeout_seconds, monitor):
            timers.append(timeout_seconds)
            return super().slot(timeout_seconds=timeout_seconds, monitor=monitor)

    async def script(row, member):
        return 1.0

    workflow = _group(WorkflowConfig(group_size=2, filter_zero_advantage_datums=False), script, "test_sl_timers")
    workflow.entry_configs = {
        "quick": WorkflowConfig(group_size=2),
        "slow": WorkflowConfig(group_size=2, worker_startup_budget_seconds=900.0),
    }
    workflow.start_limit = Recording(4)

    async def run():
        await workflow.arun_episode({"task_id": "a", POSITION_COLUMN: 0, ENVIRONMENT_COLUMN: "quick"})
        await workflow.arun_episode({"task_id": "b", POSITION_COLUMN: 1, ENVIRONMENT_COLUMN: "slow"})

    asyncio.run(run())
    assert timers == [600.0, 600.0, 1200.0, 1200.0]
    get_tracker("test_sl_timers").export(reset=True)


def test_under_process_isolation_a_slot_comes_back_when_the_worker_marks_its_first_model_call(tmp_path, monkeypatch):
    """Real rollout workers, one slot, a group of 2: each rollout tells its timeline that its first model call is
    issued, as the model gateway does, then waits until both members have (60 s at most). Member 1 starts only once
    member 0's worker has marked its call, so both slots come back at their calls, and no marker is left. Then without
    the limit no worker writes one."""

    import worker_stub_rollouts

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    meetings = {"two": (tmp_path / "meet-two", 2), "one": (tmp_path / "meet-one", 1)}
    for directory, _count in meetings.values():
        directory.mkdir()
    ipc = tmp_path / "ipc"
    ipc.mkdir(mode=0o700)
    scope = "test_sl_process"
    workflow = group_module.GroupRolloutWorkflow(
        rollout_fn=worker_stub_rollouts.call_then_meet,
        get_task_fn=lambda task_id: Task(
            goal="g", id=task_id, misc={"meet": str(meetings[task_id][0]), "count": meetings[task_id][1]}
        ),
        config=WorkflowConfig(
            group_size=2,
            rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts")),
            rollout_isolation="process",
            worker_startup_budget_seconds=240.0,
            worker_termination_grace_seconds=1.0,
        ),
        # No model name: the workers bind no model and contact no service.
        model_info=SimpleNamespace(
            model_name="", renderer_name="", renderer_kwargs={}, llm=SimpleNamespace(version=0, model_name="")
        ),
        log_path=str(tmp_path / "logs"),
        stats_scope=scope,
    )
    workflow._ipc_dir_cache = str(ipc)
    workflow.start_limit = limit = StartLimit(1)

    def markers() -> list[str]:
        return [name for name in os.listdir(ipc) if name.endswith(".first_call")]

    async def run():
        limited = await workflow.arun_episode({"task_id": "two"})
        assert markers() == []
        workflow.start_limit = None
        return limited, await workflow.arun_episode({"task_id": "one"})

    try:
        limited, unlimited = asyncio.run(asyncio.wait_for(run(), 600))
    finally:
        workflow.supervisor.close()
    assert limited.requested_rollouts == unlimited.requested_rollouts == 2
    assert limit.releases == {"first_call": 2} and limit.held == 0
    assert markers() == []  # without the limit none was written, and none was there to remove
    get_tracker(scope).export(reset=True)


# --- deadlock freedom, the order of acquisition and the straggler deadlines -------------------------------------------


@pytest.mark.parametrize("release", ["group", "rollout"])
def test_two_slots_for_groups_of_4_run_to_the_end_under_co_start_and_rolling_starts(tmp_path, monkeypatch, release):
    """Permits for two groups of 4 and two start slots: every member takes its slot inside its permit, and each run
    trains its steps, never more than 2 rollouts starting at once while more than 2 run."""

    config = trainer_config(
        tmp_path,
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.max_staleness=1",
        "train.max_concurrent_rollouts=8",
        "train.workflow_config.group_size=4",
        f"train.permit_release={release}",
        "train.max_starting_rollouts=2",
    )
    trainer, _saves, logged = make_trainer(monkeypatch, config, tasks=8)
    now = {"starting": 0, "running": 0}
    peak = dict(now)

    async def script(row, member):
        now["starting"] += 1
        now["running"] += 1
        peak.update({key: max(peak[key], value) for key, value in now.items()})
        await asyncio.sleep(0.02)  # its start-up
        now["starting"] -= 1
        _first_call()
        await asyncio.sleep(0.1)
        now["running"] -= 1
        return float(member % 2)

    workflow = _Members(config.train.workflow_config, trainer._model_info.llm, script)
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
    assert logged == [1, 2]
    assert peak["starting"] <= 2 < peak["running"]
    assert workflow.start_limit.held == 0 and not workflow.start_limit._waiters


def test_a_member_asks_for_its_slot_only_inside_its_lease_block():
    """A lease that notes which tasks are inside a block: every member asks for its slot from inside its own."""

    inside: set[asyncio.Task] = set()
    asked: list[bool] = []

    class Lease:
        size = 3

        def rollout(self, index):
            @asynccontextmanager
            async def block():
                await asyncio.sleep(0.01 * index)  # granted one by one, as under rolling starts
                task = asyncio.current_task()
                assert task is not None
                inside.add(task)
                try:
                    yield 0.01 * index
                finally:
                    inside.discard(task)

            return block()

    class Asking(_Members):
        def _member_start_slot(self):
            asked.append(asyncio.current_task() in inside)
            return super()._member_start_slot()

    async def script(row, member):
        _first_call()
        return 1.0

    workflow = Asking(
        WorkflowConfig(group_size=3, filter_zero_advantage_datums=False),
        SimpleNamespace(version=0),
        script,
        scope="test_sl_order",
    )
    workflow.start_limit = StartLimit(1)

    async def run():
        with holding(Lease()):
            await workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0})

    asyncio.run(run())
    assert asked == [True, True, True]
    get_tracker("test_sl_order").export(reset=True)


@pytest.mark.parametrize("same_instance", [False, True], ids=["another-workflow", "itself"])
def test_a_group_that_a_member_s_rollout_runs_takes_no_start_slots(same_instance):
    """One slot; each member of a group of 3 runs, in its rollout, a group of 3 of another workflow sharing the limit,
    or of the same workflow, whose members call the model. The nested groups take no slots, which would wait for the
    one their member holds until its timer, and their first call gives their member's slot back."""

    limit = StartLimit(1)

    async def script(row, member):
        if row.get("depth", 0) == 0:
            nested = await (outer if same_instance else inner).arun_episode(
                {"task_id": f"t{member}", POSITION_COLUMN: member, "depth": 1}
            )
            return float(nested.observed_rollouts == 3)
        _first_call()
        return float(member % 2)

    config = WorkflowConfig(group_size=3, filter_zero_advantage_datums=False)
    inner, outer = _group(config, script, "test_sl_inner"), _group(config, script, "test_sl_outer")
    inner.start_limit = outer.start_limit = limit
    output = asyncio.run(asyncio.wait_for(outer.arun_episode({"task_id": "t", POSITION_COLUMN: 0}), 10))
    assert output.observed_rollouts == 3 and limit.releases == {"first_call": 3} and limit.held == 0
    for scope in ("test_sl_inner", "test_sl_outer"):
        get_tracker(scope).export(reset=True)


def test_a_member_waiting_for_its_slot_is_waiting_and_its_slot_wait_joins_its_start_delay():
    """Co-start, a group of 4 on 2 slots: members 2 and 3 wait for theirs, so they are waiting (no offset) and the group
    did not start together; once member 0 calls, member 2 starts with its wait as its offset. With 4 slots every wait
    is exactly 0, and the group started together."""

    calls = {member: asyncio.Event() for member in range(4)}
    admissions: list[members_module._GroupAdmission] = []

    class Recording(_Members):
        async def _admit_group(self, group_size):
            admissions.append(await super()._admit_group(group_size))
            return admissions[-1]

    async def script(row, member):
        await calls[member].wait()
        _first_call()
        return 1.0

    def workflow(size: int) -> Recording:
        made = Recording(
            WorkflowConfig(group_size=4, filter_zero_advantage_datums=False),
            SimpleNamespace(version=0),
            script,
            scope="test_sl_offsets",
        )
        made.start_limit = StartLimit(size)
        return made

    async def run():
        group = asyncio.ensure_future(workflow(2).arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await _until(lambda: admissions and len(admissions[0].starts.waiting) == 2)
        starts = admissions[0].starts
        assert starts.waiting == {2, 3} and starts.offset(2) is None and not starts.together()
        await asyncio.sleep(0.05)
        calls[0].set()
        await _until(lambda: 2 not in starts.waiting)
        assert (offset := starts.offset(2)) is not None and offset >= 0.05
        assert starts.offset(0) == starts.offset(1) == 0.0
        for event in calls.values():
            event.set()
        await asyncio.wait_for(group, 10)
        admissions.clear()
        await asyncio.wait_for(workflow(4).arun_episode({"task_id": "t", POSITION_COLUMN: 1}), 10)
        assert admissions[0].starts.together() and not any(admissions[0].starts._waited.values())

    asyncio.run(run())
    get_tracker("test_sl_offsets").export(reset=True)


def test_a_member_that_waited_for_its_slot_is_not_cut_before_its_own_grace(caplog):
    """Co-start, a group of 4 on 2 slots, quorum 2, grace 0.3 s. Members 0 and 1 start up for 0.5 s, call, and end at
    0.6 and 0.7 s, settling the quorum; members 2 and 3 took their slots at 0.5 s. Member 2 runs past the group's
    shared deadline (1.0 s) and ends: its own carries its 0.5 s start delay. Member 3 runs on and is cut at its own."""

    config = WorkflowConfig(
        group_size=4, straggler_timeout_seconds=0.3, straggler_quorum=2, filter_zero_advantage_datums=False
    )
    quorum, shared_deadline_passed = asyncio.Event(), asyncio.Event()

    async def script(row, member):
        if member < 2:
            await asyncio.sleep(0.5)
            _first_call()
            await asyncio.sleep(0.1 + 0.1 * member)
            if member == 1:
                quorum.set()
        elif member == 2:
            _first_call()
            await shared_deadline_passed.wait()
        else:
            _first_call()
            await asyncio.sleep(30)
        return float(member % 2)

    scope = "test_sl_late_member_grace"
    workflow = _group(config, script, scope)
    workflow.start_limit = StartLimit(2)

    async def run():
        group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await quorum.wait()
        await asyncio.sleep(0.4)  # past the group's shared deadline, 0.3 s after the quorum
        assert ("cancelled", 0, 2) not in workflow.log
        shared_deadline_passed.set()
        return await asyncio.wait_for(group, 10)

    with caplog.at_level(logging.WARNING):
        output = asyncio.run(run())
    assert ("end", 0, 2) in workflow.log and ("cancelled", 0, 3) in workflow.log
    assert output.observed_rollouts == 3 and workflow.start_limit.held == 0
    assert get_tracker(scope).export(reset=True)[f"{scope}/straggler_cancelled"] == 1.0


def test_cancelling_a_group_gives_back_the_slot_it_holds_and_its_waiting_members_take_none():
    started = asyncio.Event()

    async def script(row, member):
        started.set()
        await asyncio.sleep(30)

    workflow = _group(WorkflowConfig(group_size=3, filter_zero_advantage_datums=False), script, "test_sl_cancel")
    workflow.start_limit = limit = StartLimit(1)

    async def run():
        group = asyncio.ensure_future(workflow.arun_episode({"task_id": "t", POSITION_COLUMN: 0}))
        await started.wait()
        assert limit.held == 1 and len(limit._waiters) == 2
        group.cancel()
        await asyncio.gather(group, return_exceptions=True)

    asyncio.run(run())
    assert limit.held == 0 and not limit._waiters and limit.releases == {"cancelled": 1}
    get_tracker("test_sl_cancel").export(reset=True)


def test_a_rollout_that_crashes_while_it_holds_its_slot_gives_it_back(tmp_path, monkeypatch):
    """One slot, a group of 2, its rollouts' host faked: member 0's raises, member 1's issues its first call."""

    async def run_rollout_request(request, llm, *, rollout_fn=None, reward_processor=None, on_first_call: Any = None):
        if request.rollout_number == 0:
            raise RuntimeError("the rollout's host crashed")
        on_first_call()
        outcome = _outcome(0, 1, reward=1.0, version=0)
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status="completed",
            result=outcome.result,
            workload=outcome.workload,
            observed=True,
        )

    monkeypatch.setattr(members_module, "run_rollout_request", run_rollout_request)
    workflow = _workflow(tmp_path, lambda task, config: None, "test_sl_crash", group_size=2)
    workflow.start_limit = limit = StartLimit(1)
    output = asyncio.run(asyncio.wait_for(workflow.arun_episode({"task_id": "t"}), 10))
    assert output.observed_rollouts == 1
    assert limit.releases == {"end": 1, "first_call": 1} and limit.held == 0
    get_tracker("test_sl_crash").export(reset=True)


# --- the trainer: evaluation, replay, a limit that never binds, the metrics ------------------------------------------


class _Scripted(_Members):
    """``_Members`` for rows with or without the dispatcher's position (an evaluation's have none)."""

    async def arun_episode_single(self, data, rollout_number=0):
        return await super().arun_episode_single({POSITION_COLUMN: 0, **data}, rollout_number)


def test_evaluation_during_training_shares_the_limit(tmp_path, monkeypatch):
    """Two slots, groups of 2, an evaluation of 4 tasks after every step: training's and evaluation's starting
    rollouts together never exceed 2, both take slots, both log their metrics, and the run ends."""

    config = trainer_config(
        tmp_path,
        "eval.every=1",
        "train.workflow_config.group_size=2",
        "train.max_concurrent_rollouts=8",
        "train.max_starting_rollouts=2",
    )
    trainer, _saves, _logged = make_trainer(monkeypatch, config, eval_tasks=4)
    logged = _logging_stats(trainer)
    starting = {"train": 0, "eval": 0}
    seen: list[tuple[int, int]] = []

    async def script(row, member):
        scope = "eval" if row["task_id"].startswith("val") else "train"
        starting[scope] += 1
        seen.append((starting["train"], starting["eval"]))
        await asyncio.sleep(0.02)
        starting[scope] -= 1
        _first_call()
        await asyncio.sleep(0.05)
        return float(member % 2)

    llm = trainer._model_info.llm
    train = _Scripted(config.train.workflow_config, llm, script, scope="train")
    evaluate = _Scripted(config.eval.workflow_config, llm, script, scope="eval")
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=train, eval_workflow=evaluate), 60))
    assert train.start_limit is evaluate.start_limit and train.start_limit.held == 0
    assert max(t + e for t, e in seen) <= 2 and any(e for _t, e in seen)
    assert any("rollout_client/start_limit/releases/first_call" in stats for stats in logged)
    evaluations = [stats for stats in logged if "eval/time" in stats]
    assert evaluations and all(
        stats["eval/rollout_client/start_limit/releases/first_call"] == 4.0 for stats in evaluations
    )


def test_at_k_0_a_binding_limit_trains_the_same_datums_each_step_and_no_limit_logs_no_metric(tmp_path, monkeypatch):
    """Groups of 2 with the default permits, members that take a varied number of event-loop turns to start up: two
    slots change which groups settle first, not what each step trains."""

    steps: dict[str, list[list[int]]] = {}
    keys: dict[str, set[str]] = {}
    for limit in ("off", "2"):
        where = tmp_path / limit
        where.mkdir()
        overrides = [
            "train.batch_size=4",
            "train.max_training_steps=5",
            "train.max_staleness=0",
            "train.workflow_config.group_size=2",
            "train.max_concurrent_rollouts=8",
        ]
        if limit != "off":
            overrides.append(f"train.max_starting_rollouts={limit}")
        config = trainer_config(where, *overrides)
        client = _Recording()
        trainer, _saves, _logged = make_trainer(monkeypatch, config, tasks=10, training_client=client)
        logged = _logging_stats(trainer)

        async def script(row, member):
            for _ in range((row[POSITION_COLUMN] * 7 + member * 3) % 5):
                await asyncio.sleep(0)
            _first_call()
            return float(member % 2)

        workflow = _Members(config.train.workflow_config, trainer._model_info.llm, script)
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
        steps[limit] = [sorted(step) for step in client.steps]
        keys[limit] = {key for stats in logged for key in stats if "start_limit" in key}

    assert steps["2"] == steps["off"] and len(steps["off"]) == 5
    assert keys["off"] == set() and "rollout_client/start_limit/waiting_mean" in keys["2"]


def test_a_limit_that_never_binds_trains_and_requests_as_without_one(tmp_path, monkeypatch):
    """The real group workflow, its rollouts' host faked as the parity harnesses fake it: without the key the host is
    called exactly as before (a fake with the old signature), and with a limit of the whole budget every slot is taken
    at once, so the run trains the same datums and requests the same rollouts in the same order."""

    def run(limit: str | None) -> tuple[list[list[int]], list[tuple[str, int]]]:
        where = tmp_path / (limit or "off")
        where.mkdir()
        overrides = [
            "train.batch_size=2",
            "train.max_training_steps=3",
            "train.workflow_config.group_size=4",
            "train.max_concurrent_rollouts=8",
        ]
        if limit is not None:
            overrides.append(f"train.max_starting_rollouts={limit}")
        config = trainer_config(where, *overrides)
        requests: list[tuple[str, int]] = []

        async def respond(request) -> RolloutWorkerResponse:
            requests.append((request.task_id, request.rollout_number))
            for _ in range(request.rollout_number % 3):
                await asyncio.sleep(0)
            index = int(request.task_id.split(":")[1])
            outcome = _outcome(index, request.rollout_number, reward=float(request.rollout_number % 2), version=0)
            return RolloutWorkerResponse(
                protocol_version=PROTOCOL_VERSION,
                request_id=request.request_id,
                status="completed",
                result=outcome.result,
                workload=outcome.workload,
                observed=True,
            )

        async def as_before(request, llm, *, rollout_fn=None, reward_processor=None):
            return await respond(request)

        async def with_signal(request, llm, *, rollout_fn=None, reward_processor=None, on_first_call: Any = None):
            on_first_call()
            return await respond(request)

        monkeypatch.setattr(members_module, "run_rollout_request", as_before if limit is None else with_signal)
        client = _Recording()
        trainer, _saves, _logged = make_trainer(monkeypatch, config, tasks=8, training_client=client)
        workflow = group_module.GroupRolloutWorkflow(
            rollout_fn=lambda task, rollout_config: None,
            get_task_fn=lambda task_id: Task(id=task_id),
            config=config.train.workflow_config,
            model_info=SimpleNamespace(model_name="m", llm=trainer._model_info.llm),
        )
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
        return client.steps, requests

    assert run("8") == run(None)


# --- the key and what a run says as it starts -----------------------------------------------------------------------


def test_max_starting_rollouts_takes_a_whole_number_of_one_or_more_or_null(tmp_path):
    assert trainer_config(tmp_path).train.max_starting_rollouts is None
    for value in (1, 256):
        assert trainer_config(tmp_path, f"train.max_starting_rollouts={value}").train.max_starting_rollouts == value
    assert trainer_config(tmp_path, "train.max_starting_rollouts=null").train.max_starting_rollouts is None
    for value in ("0", "-1", "true", "1.5", "'8'"):
        with pytest.raises(ConfigError, match="max_starting_rollouts"):
            trainer_config(tmp_path, f"train.max_starting_rollouts={value}")
    message = "train.max_starting_rollouts must be a whole number, 1 or more, or null for no limit"
    for value in (0, -1, True, 1.5, "8"):
        with pytest.raises(ValueError, match=message):
            TrainConfig(model_name="m", renderer_name="r", max_starting_rollouts=value)


@pytest.mark.parametrize(
    ("key", "hint"),
    [
        ("eval.max_starting_rollouts=4", "limits evaluation's start-ups too"),
        ("train.workflow_config.max_starting_rollouts=4", "a key of train, not of a workflow config"),
        ("eval.workflow_config.max_starting_rollouts=4", "a key of train, not of a workflow config"),
    ],
)
def test_the_key_anywhere_else_is_refused_with_a_hint(tmp_path, key, hint):
    with pytest.raises(ConfigError, match=hint):
        trainer_config(tmp_path, key)


def test_a_run_says_how_its_rollouts_start_up(tmp_path, caplog):
    """The limit's line, and the warnings and notes it earns: below the largest group size (training's, or an
    evaluation's during training), at or above the rollouts that run at once, and a training workflow that takes no
    slots; nothing and no gate without the key, which also clears an earlier start's gate."""

    def lines(*overrides: str, train: Any = None, evaluation: Any = None) -> list[str]:
        config = trainer_config(tmp_path, *overrides)
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="platoon.train.workflow.start_limit"):
            rl_module._limit_starts(config, train, evaluation)
        return [record.getMessage() for record in caplog.records]

    group = _group(WorkflowConfig(group_size=8), None, "test_sl_lines")
    said = lines("train.max_starting_rollouts=4", train=group)
    assert len(said) == 2 and "limited to 4 at once" in said[0] and "below the largest group size (8)" in said[1]
    assert isinstance(group.start_limit, StartLimit) and group.start_limit.limit == 4
    evaluation = _group(WorkflowConfig(group_size=1), None, "test_sl_lines_eval")
    said = lines(
        "train.max_starting_rollouts=8",
        "eval.workflow_config.group_size=16",
        "eval.max_concurrent_rollouts=16",
        train=group,
        evaluation=evaluation,
    )
    assert any("below the largest group size (16)" in line for line in said)
    assert evaluation.start_limit is group.start_limit
    said = lines(
        "train.max_starting_rollouts=272", "eval.max_concurrent_rollouts=256", train=group, evaluation=evaluation
    )
    assert any("can never bind: no more than 272 rollouts run at once" in line for line in said)
    said = lines("train.max_starting_rollouts=8", train=SimpleNamespace())
    assert any("has no effect on training" in line for line in said)
    assert lines(train=group, evaluation=evaluation) == []
    assert group.start_limit is None and evaluation.start_limit is None
