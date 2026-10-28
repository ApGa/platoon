"""Credit estimators in a training run (RFC 0021, section 6): ``GroupRolloutWorkflow`` awaits the entry's estimator
over the group, within its time limit, tags the rows of an estimator that reads agents (``PolicyGroupOutput``, which a
checkpoint keeps whole), whatever order the estimator leaves the members in, and refuses rows without tags; its
requests ask the workers for tags (protocol 3) only then; a subclass with credit hooks of its own keeps them, unless it
declares that it applies credit; the run settles a stateful estimator as it files each group and saves its state with
the groups a checkpoint keeps; an estimator sees the run's policies and each one's version; and an estimator that
raises, in assign or in settle, or runs out of time, stops the run with an error that names it, while a policy's failed
scoring fails only its group."""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.train.checkpoints import saving
from platoon.train.checkpoints.kept_groups import decode_group, encode_group
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import CreditEstimatorBase, CreditGroup, CreditSetup, PolicyView
from platoon.train.credit.group import GroupCredit
from platoon.train.credit.keyed import Keyed, KeyedOptions
from platoon.train.credit.registry import workflow_problems
from platoon.train.credit.runtime import CreditEstimatorError, settle_credit
from platoon.train.dispatch import runner
from platoon.train.dispatch.settled import carries_datums, emptied
from platoon.train.dispatch.source import RolloutSource
from platoon.train.leases import NO_BLOCK
from platoon.train.policy.spec import MAIN
from platoon.train.workers import protocol
from platoon.train.workers.protocol import PolicyRolloutWorkerRequest, RolloutWorkerRequest
from platoon.train.workflow.context import _group_entry
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import GroupOutput, PolicyGroupOutput
from platoon.train.workflow.protocol import capabilities_of, report_of, rows_by_policy
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.tinker_data_processing import AgentTag, TrajectoryCollectionResult, TrajectoryStats

_SETUP = CreditSetup(name="test", entry="default", settings=WorkflowConfig(), policies=(MAIN,))


def _datum(reward: float, actions: float = 1.0) -> tinker.Datum:
    """One observation token, then two action tokens that carry ``reward`` (with ``actions`` 0, two tokens of an agent
    message, which train nothing)."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor([0.0, actions, actions])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward * actions, reward * actions])),
            "checkpoint_version": TensorData.from_torch(torch.tensor([0])),
        },
    )


def _rollout(root: float, child: float, *, tagged: bool = True, message: bool = False) -> TrajectoryCollectionResult:
    """A rollout whose root (role ``policy``) delegated to one subagent (role ``helper``), one row each; with
    ``message``, the root has a second row, an agent message, which the error-token filter drops."""

    root_rows = [_datum(root), _datum(root, actions=0.0)] if message else [_datum(root)]
    stats = [
        TrajectoryStats(
            trajectory_id=trajectory,
            reward=reward,
            num_steps=1,
            num_input_tokens=1,
            num_output_tokens=2,
            num_datums=rows,
            is_root=trajectory == "root",
        )
        for trajectory, reward, rows in (("root", root, len(root_rows)), ("root.1", child, 1))
    ]
    return TrajectoryCollectionResult(
        datums=[*root_rows, _datum(child)],
        task_reward=root,
        trajectory_stats=stats,
        root_rewards_dict={},
        agents=(
            AgentTag("root", "agent-root", "policy", None, 0, root),
            AgentTag("root.1", "agent-helper", "helper", "root", 1, child),
        )
        if tagged
        else None,
        datum_trajectories=(*["root"] * len(root_rows), "root.1") if tagged else None,
    )


def _workflow(estimator: Any = None, workflow_cls: type = GroupRolloutWorkflow, *, group_size: int = 2) -> Any:
    async def rollout(task: Any, rollout_config: Any) -> dict:
        raise AssertionError("not run")

    workflow = workflow_cls(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=group_size),
        model_info=SimpleNamespace(
            model_name="m", renderer_name="qwen3_instruct", renderer_kwargs={}, llm=SimpleNamespace(version=4)
        ),
    )
    if estimator is not None:
        workflow.entry_configs = {"default": WorkflowConfig(group_size=group_size)}
        workflow.entry_credit = {"default": estimator}
    return workflow


def _run(workflow: Any, results: Sequence[TrajectoryCollectionResult]) -> Any:
    async def single(data: dict, rollout_number: int) -> TrajectoryCollectionResult:
        return results[rollout_number]

    workflow.arun_episode_single = single
    return asyncio.run(workflow.arun_episode({"task_id": "task"}))


def _action_values(datums: Sequence[tinker.Datum]) -> list[float]:
    return [float(datum.loss_fn_inputs["advantages"].to_torch()[1]) for datum in datums]


# --- the workflow ----------------------------------------------------------------------------------------------------


def test_an_estimator_that_reads_agents_gets_each_member_s_agents_and_the_output_keeps_the_tags():
    output = _run(_workflow(Keyed(_SETUP, KeyedOptions(key=["role"]))), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])

    assert isinstance(output, PolicyGroupOutput)
    assert [(sample.member, sample.role, sample.agent_id, sample.depth) for sample in output.samples] == [
        (0, "policy", "agent-root", 0),
        (0, "helper", "agent-helper", 1),
        (1, "policy", "agent-root", 0),
        (1, "helper", "agent-helper", 1),
    ]
    rows = rows_by_policy(output, MAIN)
    # Roots centered on the roots' mean (0.5), the helpers on theirs (0): the helpers' rows train nothing.
    assert list(rows) == [MAIN] and _action_values(rows[MAIN]) == [0.5, 0.0, -0.5, 0.0]
    report = report_of(output)
    assert (report.entry, report.verdict, report.trainable_rollouts, report.task_retained_datums) == (
        "default",
        "trained",
        2,
        4,
    )
    assert len(output) == 4


def test_the_group_estimator_gives_the_workflow_s_own_credit_and_its_output():
    own = _run(_workflow(), [_rollout(1.0, 1.0), _rollout(0.0, 1.0)])
    estimated = _run(_workflow(GroupCredit(_SETUP)), [_rollout(1.0, 1.0), _rollout(0.0, 1.0)])

    assert type(own) is type(estimated) is GroupOutput
    assert _action_values(own) == _action_values(estimated) == [0.5, 0.5, -0.5, 0.5]


def test_rows_without_tags_fail_the_group_under_an_estimator_that_reads_agents():
    with pytest.raises(ValueError, match="carry no tags of the agents that produced them"):
        _run(_workflow(Keyed(_SETUP)), [_rollout(1.0, 0.0, tagged=False), _rollout(0.0, 0.0, tagged=False)])


class _OwnBaselines(GroupRolloutWorkflow):
    def compute_baselines(self, results: list[TrajectoryCollectionResult]) -> list[float]:
        return [0.0] * len(results)


def test_a_subclass_with_credit_hooks_of_its_own_keeps_them_and_refuses_an_estimator():
    assert GroupRolloutWorkflow.applies_credit is True and _OwnBaselines.applies_credit is False
    assert capabilities_of(_OwnBaselines).applies_credit is False
    output = _run(_workflow(Keyed(_SETUP), _OwnBaselines), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert isinstance(output, GroupOutput) and _action_values(output) == [1.0, 0.0, 0.0, 0.0]

    choices: Any = {"default": SimpleNamespace()}
    [problem] = workflow_problems(choices, _OwnBaselines, "acme:OwnBaselines")
    assert problem.startswith("environments[default].credit: the run's workflow, acme:OwnBaselines, does not apply")
    assert workflow_problems(choices, GroupRolloutWorkflow, "group_rollout") == []


class _DeclaresItApplies(GroupRolloutWorkflow):
    """A subclass that overrides assign_advantages, for groups without an estimator, and declares that it applies an
    entry's estimator."""

    capabilities = replace(capabilities_of(GroupRolloutWorkflow), applies_credit=True)

    def assign_advantages(self, results: list[TrajectoryCollectionResult]) -> None:
        super().assign_advantages(results)


def test_a_subclass_that_declares_it_applies_credit_credits_with_the_entry_s_estimator():
    choices: Any = {"default": SimpleNamespace()}
    assert workflow_problems(choices, _DeclaresItApplies, "acme:DeclaresItApplies") == []
    output = _run(_workflow(_Settling(_SETUP), _DeclaresItApplies), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert _action_values(output) == [1.0, 1.0, 1.0, 1.0]  # the estimator's, as loading said


class _Settling(CreditEstimatorBase):
    needs_agents = False

    def __init__(self, setup: CreditSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self.assigned: list[str] = []
        self.settled: list[str] = []

    async def assign(self, group: CreditGroup) -> None:
        await asyncio.sleep(0)  # an estimator may await
        self.assigned.append(group.task_id)
        for sample in group.samples():
            sample.set_advantage(1.0)

    def settle(self, group: CreditGroup) -> None:
        self.settled.append(group.task_id)

    def state(self) -> dict:
        return {"settled": list(self.settled)}

    def load_state(self, state: dict) -> None:
        self.settled = list(state["settled"])


def test_a_stateful_estimator_is_settled_by_the_run_as_it_files_the_group_once():
    estimator = _Settling(_SETUP)
    output = _run(_workflow(estimator), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert estimator.assigned == ["task"] and estimator.settled == [] and output.credit is not None
    settle_credit(output)
    settle_credit(output)
    assert estimator.settled == ["task"] and output.credit is None
    assert _run(_workflow(GroupCredit(_SETUP)), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)]).credit is None  # no settle


class _RanksByReward(CreditEstimatorBase):
    """A rank-based estimator that sorts the group's members by reward, in place, before it ranks them."""

    needs_agents = True

    async def assign(self, group: CreditGroup) -> None:
        group.members.sort(key=lambda member: member.task_reward)
        for rank, member in enumerate(group.members):
            for sample in member.samples:
                sample.set_advantage(float(rank))


def test_an_estimator_that_reorders_the_members_still_sends_the_rows_the_filter_kept():
    first, second = _rollout(1.0, 0.0, message=True), _rollout(0.0, 0.0)
    trained = [first.datums[0], first.datums[2], *second.datums]  # all but the message
    output = _run(_workflow(_RanksByReward(_SETUP)), [first, second])
    assert [id(sample.datum) for sample in output.samples] == [id(datum) for datum in trained]
    assert [(sample.member, sample.role) for sample in output.samples] == [
        (0, "policy"),
        (0, "helper"),
        (1, "policy"),
        (1, "helper"),
    ]
    assert _action_values(rows_by_policy(output, MAIN)[MAIN]) == [1.0, 1.0, 0.0, 0.0]


class _Waits(CreditEstimatorBase):
    """A service call that does not answer in time."""

    assign_timeout_seconds = 0.05
    cancelled = False

    async def assign(self, group: CreditGroup) -> None:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            type(self).cancelled = True
            raise


def test_an_assign_that_runs_out_of_time_is_cancelled_and_stops_the_run():
    assert CreditEstimatorBase.assign_timeout_seconds == 600.0
    with pytest.raises(CreditEstimatorError, match=r"assign\(group\) did not return within 0.05 s, on task task$"):
        _run(_workflow(_Waits(_SETUP)), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert _Waits.cancelled

    class Unlimited(_Settling):
        assign_timeout_seconds = None

    assert _action_values(_run(_workflow(Unlimited(_SETUP)), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])) == [1.0] * 4


class _Raises(CreditEstimatorBase):
    async def assign(self, group: CreditGroup) -> None:
        raise KeyError("a bug")


def test_an_estimator_that_raises_stops_the_run_and_its_group_counts_as_no_failed_rollout():
    workflow = _workflow(_Raises(_SETUP))

    async def single(data: dict, rollout_number: int) -> TrajectoryCollectionResult:
        return [_rollout(1.0, 0.0), _rollout(0.0, 0.0)][rollout_number]

    workflow.arun_episode_single = single
    recorded: list[dict] = []
    tracker: Any = SimpleNamespace(scalar=lambda **values: recorded.append(values))
    with pytest.raises(
        CreditEstimatorError,
        match=r"^the credit estimator test of environments\[default\]: assign\(group\) raised KeyError: 'a bug', on "
        r"task task$",
    ):
        asyncio.run(runner._run_group(workflow, {"task_id": "task"}, watchdog=None, tracker=tracker))
    assert recorded == []


class _ScoresWithTheTeacher(CreditEstimatorBase):
    """Scores the group's rows with the teacher, as reverse_kl does, and records what it saw of the policies."""

    seen: list[tuple[dict, list]] = []

    async def assign(self, group: CreditGroup) -> None:
        type(self).seen.append((dict(group.versions), sorted(group.policies)))
        await group.policies["teacher"].score([sample.datum for sample in group.samples()])


async def _unavailable(datums: Sequence[tinker.Datum]) -> list[torch.Tensor]:
    raise RuntimeError("503 Service Unavailable")


def test_a_policy_s_failed_scoring_fails_only_its_group(caplog):
    workflow = _workflow(_ScoresWithTheTeacher(_SETUP))
    versions = {MAIN: 6}
    workflow.policy_views = {
        MAIN: PolicyView(MAIN, current_version=lambda: versions[MAIN]),
        "teacher": PolicyView("teacher", scorer=_unavailable),
    }

    async def single(data: dict, rollout_number: int) -> TrajectoryCollectionResult:
        return [_rollout(1.0, 0.0), _rollout(0.0, 0.0)][rollout_number]

    workflow.arun_episode_single = single
    tracker: Any = SimpleNamespace(scalar=lambda **values: None)
    output = asyncio.run(runner._run_group(workflow, {"task_id": "task"}, watchdog=None, tracker=tracker))

    # The run goes on: the group's output says it failed, and it trains nothing.
    report = report_of(output)
    assert report.verdict == "failed" and not carries_datums(output)
    stats = StatsTracker()
    assert report.group_stats is not None
    report.group_stats.record(stats)
    assert stats.export()["credit/failures"] == 1.0
    assert "Task task group trains nothing: its credit failed, policy teacher could not score 4 row(s)" in caplog.text
    # The estimator saw the run's policies, each at its version when assign began (a frozen one's: 0).
    assert _ScoresWithTheTeacher.seen[-1] == ({MAIN: 6, "teacher": 0}, [MAIN, "teacher"])


def test_without_the_run_s_policies_a_group_has_main_s_from_the_workflow():
    estimator = _ScoresWithTheTeacher(_SETUP)
    with pytest.raises(
        CreditEstimatorError, match="raised KeyError: 'teacher'"
    ):  # an estimator's own bug stops the run
        _run(_workflow(estimator), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert _ScoresWithTheTeacher.seen[-1] == ({MAIN: 4}, [MAIN])


def test_a_tagged_output_is_kept_whole_by_a_checkpoint():
    output = _run(_workflow(Keyed(_SETUP)), [_rollout(1.0, 0.0), _rollout(0.0, 1.0)])
    encoded = encode_group((0, 3), entry="default", label=None, record=None, output=output)
    assert encoded is not None and encoded["output"]["type"] == "policy_group"
    _, _, _, _, decoded = decode_group(pickle.loads(pickle.dumps(encoded)))

    assert isinstance(decoded, PolicyGroupOutput)
    assert replace(decoded.report, group_stats=None) == replace(output.report, group_stats=None)
    assert decoded.report.group_stats.to_json() == output.report.group_stats.to_json()
    assert [(s.policy, s.member, s.agent_id, s.role, s.trajectory_id, s.depth) for s in decoded.samples] == [
        (s.policy, s.member, s.agent_id, s.role, s.trajectory_id, s.depth) for s in output.samples
    ]
    assert _action_values(rows_by_policy(decoded, MAIN)[MAIN]) == _action_values(rows_by_policy(output, MAIN)[MAIN])


# --- the worker requests ---------------------------------------------------------------------------------------------


def _request(workflow: Any) -> RolloutWorkerRequest:
    token = _group_entry.set("default")
    try:
        return workflow._build_worker_request({"task_id": "task"}, 0)
    finally:
        _group_entry.reset(token)


def test_requests_ask_for_tags_only_for_an_estimator_that_reads_agents(tmp_path):
    plain, grouped = _request(_workflow()), _request(_workflow(GroupCredit(_SETUP)))
    tagged = _request(_workflow(Keyed(_SETUP)))

    assert type(plain) is type(grouped) is RolloutWorkerRequest and plain.protocol_version == protocol.PROTOCOL_VERSION
    assert isinstance(tagged, PolicyRolloutWorkerRequest) and tagged.tag_datums
    assert tagged.protocol_version == protocol.POLICY_PROTOCOL_VERSION == 3
    # Every field of protocol 2 as it is, and the request round-trips through the workers' file.
    assert {name: getattr(tagged, name) for name in vars(plain) if name != "request_id"} == {
        name: getattr(plain, name) for name in vars(plain) if name != "request_id"
    } | {"protocol_version": 3}
    path = str(tmp_path / "request.pkl")
    protocol.dump_request(tagged, path)
    assert protocol.load_request(path) == tagged


def test_a_request_of_another_protocol_is_refused(tmp_path):
    request = _request(_workflow())
    path = str(tmp_path / "request.pkl")
    fields: dict[str, Any] = dict(vars(request))
    fields["protocol_version"] = 3
    protocol.dump_request(RolloutWorkerRequest(**fields), path)
    with pytest.raises(protocol.ProtocolVersionError, match="protocol 3 != 2"):
        protocol.load_request(path)


# --- the run ---------------------------------------------------------------------------------------------------------


class _Lease:
    """A draw's lease that holds no permit, called once the group's workflow returned."""

    size = 0

    def __call__(self) -> None:
        pass

    def rollout(self, index: int) -> Any:
        return NO_BLOCK


class _Dispatcher:
    def __init__(self, estimator: _Settling) -> None:
        self.estimator, self.filed = estimator, []

    def settle(self, draw: Any, item: Any) -> None:
        self.filed.append((draw.task_id, list(self.estimator.settled)))


def _run_training_group(estimator: _Settling, dispatcher: _Dispatcher) -> None:
    """Run a group of task ``task`` under ``estimator`` as the rollout source runs a training group, which files it
    with ``dispatcher``."""

    workflow = _workflow(estimator)

    async def single(data: dict, rollout_number: int) -> TrajectoryCollectionResult:
        return [_rollout(1.0, 0.0), _rollout(0.0, 0.0)][rollout_number]

    workflow.arun_episode_single = single
    source: Any = SimpleNamespace(
        dispatcher=dispatcher,
        tracker=SimpleNamespace(scalar=lambda **_: None),
        watchdog=None,
        workflow=workflow,
        recorder=None,
        admissions=None,
        episode_row=lambda draw: draw.row,  # no draw provenance: the group runs on the draw's own row
    )
    draw: Any = SimpleNamespace(row={"task_id": "task"}, task_id="task", entry="default", label=None, key=(0, 1))
    lease: Any = _Lease()
    asyncio.run(runner._run_training_group(source, draw, lease))


def test_the_runner_settles_the_credit_before_it_files_the_group():
    estimator = _Settling(_SETUP)
    dispatcher = _Dispatcher(estimator)
    _run_training_group(estimator, dispatcher)
    assert dispatcher.filed == [("task", ["task"])]


class _SettleRaises(_Settling):
    def settle(self, group: CreditGroup) -> None:
        raise RuntimeError("a bug in settle")


def test_a_settle_that_raises_stops_the_run_before_its_group_is_filed():
    estimator = _SettleRaises(_SETUP)
    dispatcher = _Dispatcher(estimator)
    with pytest.raises(CreditEstimatorError, match=r"settle\(group\) raised RuntimeError: a bug in settle, on task"):
        _run_training_group(estimator, dispatcher)
    assert estimator.assigned == ["task"] and dispatcher.filed == []


class _Kept:
    def state(self) -> dict:
        return {"next_position": 0}

    def kept(self) -> list:
        return []


def test_a_checkpoint_takes_the_estimators_state_with_the_groups_it_keeps(tmp_path, monkeypatch):
    estimator = _Settling(_SETUP)
    estimator.settled.append("g1")

    def save_state(run_dir: str, checkpoint: str, state: dict, kept: list) -> dict:
        estimator.settled.append("g2")  # a group that settles while the dispatcher's files are written
        return {"file": "dispatch.json"}

    monkeypatch.setattr(saving, "save_state", save_state)
    source: Any = SimpleNamespace(
        recorder=None,
        dispatcher=_Kept(),
        admissions=None,
        run_dir=str(tmp_path),
        workflow=SimpleNamespace(entry_credit={"default": estimator}),
    )
    loop_state = asyncio.run(saving._checkpoint_state(source, "000003", {}))
    state = {"class": f"{__name__}:_Settling", "state": {"settled": ["g1"]}}
    assert loop_state == {"dispatch": {"file": "dispatch.json"}, "credit": {"default": state}}

    plain: Any = SimpleNamespace(**{**vars(source), "workflow": _workflow()})
    assert "credit" not in asyncio.run(saving._checkpoint_state(plain, "000003", {}))


class _KeepsOne(_Kept):
    def kept(self) -> list:
        return [((0, 1), SimpleNamespace())]


@pytest.mark.parametrize(
    ("dispatcher", "record", "saved"),
    [
        (_Kept(), None, False),  # the dispatcher's state could not be written: the resume rolls its draws out again
        (_KeepsOne(), {"file": "dispatch.json"}, False),  # its kept groups could not be written
        (_KeepsOne(), {"file": "dispatch.json", "kept": "kept.json.gz"}, True),
    ],
)
def test_a_checkpoint_saves_the_estimators_state_only_beside_the_groups_it_settled(
    tmp_path, monkeypatch, dispatcher, record, saved
):
    monkeypatch.setattr(saving, "save_state", lambda run_dir, checkpoint, state, kept: record)
    estimator = _Settling(_SETUP)
    source: Any = SimpleNamespace(
        recorder=None,
        dispatcher=dispatcher,
        admissions=None,
        run_dir=str(tmp_path),
        workflow=SimpleNamespace(entry_credit={"default": estimator}),
    )
    assert ("credit" in asyncio.run(saving._checkpoint_state(source, "000003", {}))) is saved


def test_a_resumed_source_gives_the_estimators_their_saved_state(monkeypatch):
    estimator = _Settling(_SETUP)
    source: Any = RolloutSource.__new__(RolloutSource)
    source.workflow = SimpleNamespace(entry_credit={"default": estimator})
    source.dispatcher = object()
    monkeypatch.setattr(RolloutSource, "restore", lambda self, dispatcher, record, version: None)
    saved = {"default": {"class": f"{__name__}:_Settling", "state": {"settled": ["g1"]}}}
    start: Any = SimpleNamespace(record=object(), policy_version=3, loop_state={"credit": saved})
    source.resume(start)
    assert estimator.settled == ["g1"] and "credit" in RolloutSource.keys


def test_rows_tagged_with_a_trajectory_of_no_agent_fail_the_group():
    rollout = _rollout(1.0, 0.0)
    rollout.datum_trajectories = ("root", "elsewhere")
    with pytest.raises(ValueError, match=r"tagged with trajectories of no agent of its tree: \['elsewhere'\]"):
        _run(_workflow(Keyed(_SETUP)), [rollout, _rollout(0.0, 0.0)])


def test_a_tagged_output_goes_through_an_admission_rule_s_settling_as_a_list_does():
    output = _run(_workflow(Keyed(_SETUP)), [_rollout(1.0, 0.0), _rollout(0.0, 0.0)])
    assert carries_datums(output) and not carries_datums(PolicyGroupOutput((), output.report))
    marker = emptied(output)  # a rejected group's marker: no rows, the same report
    assert list(marker) == [] and report_of(marker) == output.report
