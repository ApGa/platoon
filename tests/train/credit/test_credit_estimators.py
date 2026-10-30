"""Core's credit estimators and the conformance check: ``group`` reproduces the workflow's own credit value for value
(advantages and ``credit/*`` metrics), ``keyed`` centers each agent on the agents of its key, and
``platoon.testing.check_credit_estimator`` passes on both and names what an estimator breaks."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

from platoon.testing import check_credit_estimator
from platoon.testing.credit import _datum, fixture_group, run_checks
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import (
    CreditCheckContext,
    CreditEstimatorBase,
    CreditGroup,
    CreditSetup,
    PolicyView,
    TrainingSample,
)
from platoon.train.credit.group import GroupCredit
from platoon.train.credit.keyed import Keyed, KeyedOptions
from platoon.train.policy.spec import MAIN
from platoon.train.workflow.crediting import member_rollout
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.tinker_data_processing import AgentTag, TrajectoryCollectionResult

_SETUP = CreditSetup(name="test", entry="default", settings=WorkflowConfig(), policies=(MAIN,))


def _group(results: list[TrajectoryCollectionResult], *, tagged: bool, leave_one_out: bool = False) -> CreditGroup:
    return CreditGroup(
        task_id="task",
        label=None,
        entry="default",
        settings=WorkflowConfig(leave_one_out_baseline=leave_one_out),
        members=[member_rollout(index, result, MAIN, tagged=tagged) for index, result in enumerate(results)],
        policies={MAIN: PolicyView(MAIN)},
        versions={MAIN: 0},
        stats=DeferredStats(),
    )


def _results(seed: int) -> list[TrajectoryCollectionResult]:
    """The members of check_credit_estimator's fixture group ``seed``, as the converter returns them, with tags."""

    results = []
    for member in fixture_group(seed, tagged=True).members:
        result = member.result
        result.agents = tuple(
            AgentTag(agent.trajectory_id, agent.agent_id, agent.role, agent.parent_id, agent.depth, agent.reward)
            for agent in member.agents
        )
        result.datum_trajectories = tuple(sample.trajectory_id for sample in member.samples)
        results.append(result)
    return results


def _recorded(stats: Any) -> dict[str, Any]:
    tracker = StatsTracker()
    stats.record(tracker)
    return tracker.export()


# --- group -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("leave_one_out", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_group_reproduces_the_workflow_s_own_credit_value_for_value(seed, leave_one_out):
    own, estimated = _results(seed), _results(seed)
    workflow = GroupRolloutWorkflow.__new__(GroupRolloutWorkflow)
    workflow.config = WorkflowConfig(leave_one_out_baseline=leave_one_out)
    own_stats = DeferredStats()
    setattr(workflow, "_group_stats", lambda: own_stats)
    workflow.assign_advantages(own)

    group = _group(estimated, tagged=False, leave_one_out=leave_one_out)
    asyncio.run(GroupCredit(_SETUP).assign(group))

    for left, right in zip(own, estimated, strict=True):
        for a, b in zip(left.datums, right.datums, strict=True):
            assert torch.equal(a.loss_fn_inputs["advantages"].to_torch(), b.loss_fn_inputs["advantages"].to_torch())
            assert a.loss_fn_inputs["advantages"].to_torch().dtype == torch.float32
    assert _recorded(group.stats) == _recorded(own_stats)


# --- keyed -----------------------------------------------------------------------------------------------------------


def _agent_advantages(group: CreditGroup) -> dict[str, list[float]]:
    """Each agent's action-token advantage, of each of its rows, by agent id."""

    return {
        agent.agent_id: [float(sample.advantages[sample.mask > 0][0]) for sample in agent.samples]
        for agent in group.agents()
    }


def test_keyed_on_task_and_role_centers_each_role_on_its_own_agents():
    group = _group(_results(0), tagged=True)
    asyncio.run(Keyed(_SETUP, KeyedOptions(key=["task", "role"])).assign(group))

    for role in ("policy", "helper"):
        scored = [(agent, agent.reward) for agent in group.agents() if agent.role == role and agent.reward is not None]
        mean = sum(reward for _, reward in scored) / len(scored)
        for agent, reward in scored:
            assert all(value == pytest.approx(reward - mean) for value in _agent_advantages(group)[agent.agent_id])
    metrics = _recorded(group.stats)
    assert metrics["credit/keyed/baselines"] == 2.0


def test_keyed_leave_one_out_and_an_agent_alone_in_its_key():
    group = _group(_results(1), tagged=True)
    asyncio.run(Keyed(_SETUP, KeyedOptions(key=["depth"], leave_one_out=True)).assign(group))

    roots = [(agent, agent.reward) for agent in group.agents() if agent.depth == 0 and agent.reward is not None]
    total = sum(reward for _, reward in roots)
    for agent, reward in roots:
        expected = reward - (total - reward) / (len(roots) - 1)
        assert _agent_advantages(group)[agent.agent_id] == pytest.approx([expected] * len(agent.samples))

    alone = _group(_results(1)[:1], tagged=True)
    asyncio.run(Keyed(_SETUP, KeyedOptions(key=["role"], leave_one_out=True)).assign(alone))
    assert all(value == 0.0 for values in _agent_advantages(alone).values() for value in values)


def test_keyed_by_default_keys_on_task_policy_and_role_and_an_unscored_agent_trains_nothing():
    assert KeyedOptions().key == ["task", "policy", "role"] and not KeyedOptions().leave_one_out
    group = _group(_results(2), tagged=True)
    unscored = group.members[0].agents[0]
    unscored.reward = None
    # A row of an agent that was not scored, were it to have one: the converter loaded a reward onto it.
    unscored.samples = [TrainingSample(_datum(0.75, (0, 1, 1, 0, 0, 0, 0, 0)), MAIN, 0)]
    asyncio.run(Keyed(_SETUP).assign(group))
    assert float(unscored.samples[0].advantages.abs().sum()) == 0.0


@pytest.mark.parametrize("key", [[], ["role", "role"], ["member"]])
def test_keyed_refuses_a_key_it_cannot_make(key):
    with pytest.raises(ValueError, match="key: one or more of task, policy, role, depth, each once"):
        KeyedOptions(key=key)


# --- the conformance check -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimator", "options"),
    [("group", None), ("keyed", None), ("keyed", {"key": ["depth"], "leave_one_out": True}), (Keyed, None)],
)
def test_core_s_estimators_pass_the_conformance_check(estimator, options):
    check_credit_estimator(estimator, options=options)


class _OnEveryToken(CreditEstimatorBase):
    async def assign(self, group: CreditGroup) -> None:
        for sample in group.samples():
            sample._set(torch.ones_like(sample.advantages))


class _Shorter(CreditEstimatorBase):
    async def assign(self, group: CreditGroup) -> None:
        for sample in group.samples():
            sample._set(sample.advantages[:-1])


class _Remembers(CreditEstimatorBase):
    """A baseline that runs over the groups it saw, without giving its state: a resume would credit otherwise."""

    def __init__(self, setup: CreditSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self.seen = 0

    async def assign(self, group: CreditGroup) -> None:
        self.seen += 1
        for sample in group.samples():
            sample.set_advantage(float(self.seen))


class _Forgets(_Remembers):
    """A running baseline, changed in settle, whose state it gives but does not take back."""

    async def assign(self, group: CreditGroup) -> None:
        for sample in group.samples():
            sample.set_advantage(float(self.seen))

    def settle(self, group: CreditGroup) -> None:
        self.seen += 1

    def state(self) -> dict:
        return {"seen": self.seen}

    def load_state(self, state: dict) -> None:
        pass


class _Resumes(_Forgets):
    """A running baseline, changed in settle and taken back on resume: what the check asks of a stateful estimator."""

    def load_state(self, state: dict) -> None:
        self.seen = int(state["seen"])


class _StateInAssign(_Resumes):
    """A running baseline it gives and takes back, but changes in assign: a group credited and never filed (the run
    stopped before) would count in it."""

    async def assign(self, group: CreditGroup) -> None:
        self.seen += 1
        await super().assign(group)

    def settle(self, group: CreditGroup) -> None:
        pass


@pytest.mark.parametrize(
    ("cls", "check", "says"),
    [
        (_OnEveryToken, "action_tokens", "advantages changed on tokens that are no actions"),
        (_Shorter, "action_tokens", "advantages of (7,)"),
        (_Remembers, "resumes", "(it keeps no state())"),
        (_Forgets, "resumes", "after taking up its state"),
        (_StateInAssign, "resumes", "assign(group) changed state(): an estimator changes its state in settle(group)"),
    ],
)
def test_the_check_names_what_an_estimator_breaks(cls, check, says):
    failures = run_checks(cls)
    assert check in failures and any(says in what for what in failures[check]), failures
    with pytest.raises(AssertionError, match=f"fails {len(failures)} check"):
        check_credit_estimator(cls)


def test_a_stateful_estimator_that_settles_and_resumes_passes():
    check_credit_estimator(_Resumes)


class _ChecksTheConfig(_Resumes):
    """An estimator whose load-time check cross-checks the loaded config, as the guide suggests."""

    seen: list[tuple[str, int, list[str]]] = []

    @staticmethod
    def check(options: Any, ctx: CreditCheckContext) -> None:
        settings = ctx.config.train.workflow_config
        _ChecksTheConfig.seen.append((ctx.entry, settings.group_size, [env.label for env in ctx.config.environments]))
        if settings.leave_one_out_baseline:
            raise ValueError("centers on the group mean: leave_one_out_baseline must be false")


def test_the_check_gives_a_load_time_check_a_loaded_config_of_the_fixture_s_entry():
    check_credit_estimator(_ChecksTheConfig)
    assert _ChecksTheConfig.seen == [("fixture", WorkflowConfig().group_size, ["fixture"])]


def test_options_the_schema_refuses_and_an_unknown_name_fail_the_check():
    assert "options" in run_checks("keyed", options={"key": ["member"]})
    assert "options" in run_checks("keyed", options={"keys": ["role"]})
    assert "protocol" in run_checks("no-such-estimator")


def test_the_fixture_has_two_roles_an_interrupted_root_and_an_agent_without_rows():
    group = fixture_group(0, tagged=True)
    agents = group.agents()
    assert {agent.role for agent in agents} == {"policy", "helper"}
    assert any(agent.reward is None and not agent.samples for agent in agents)
    assert any(agent.reward is not None and not agent.samples for agent in agents)
    untagged = fixture_group(0, tagged=False)
    assert not untagged.agents() and all(sample.role is None for sample in untagged.samples())
