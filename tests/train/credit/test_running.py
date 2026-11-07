"""``credit: running``: each scored agent's rows credited against decayed statistics of the groups that settled before
it sampled, by role and by stratum: the baseline shrunk toward its parent stratum's, the scale a decayed root mean
square, both read at the oldest version the agent sampled at, as they were when its policy reached it; the priors, the
warm-up, the clip, the penalty cap and the versions kept, as many as the rows' staleness bound; rewards that are not
finite; its options, its priors file and its load-time check; its state, small and exact through a resume; and the
conformance check."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

import pytest
import tinker
import torch
from tinker import TensorData

from platoon.options import schema_options
from platoon.testing import check_credit_estimator
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import AgentRun, CreditGroup, CreditSetup, MemberRollout, PolicyView, TrainingSample
from platoon.train.credit.registry import BUILTIN_CREDIT_ESTIMATORS, resolve_credit_estimator
from platoon.train.credit.running import (
    DEFAULT_WINDOW,
    MIN_SCALE,
    WARMUP_SAMPLES,
    Running,
    RunningOptions,
    RunningPriors,
    StratumPrior,
    read_priors,
)
from platoon.train.policy.spec import MAIN
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

_SETUP = CreditSetup(name="running", entry="solo", settings=WorkflowConfig(group_size=1), policies=(MAIN, "worker"))


def _bounded(bound: int) -> CreditSetup:
    """The setup of an entry whose rows of either policy may be ``bound`` versions old and still train."""

    return dataclasses.replace(_SETUP, max_staleness={MAIN: bound, "worker": bound})


# A priors' scale of 1 and a negligible weight: an advantage is its reward less its baseline while the residuals stay
# about 1, with no warm-up.
_FLAT = RunningPriors(mean=0.0, scale=1.0, weight=1e-9)


class Clock:
    """The policies' versions, which a test moves on as training would: each group reads them when it is built (its
    ``versions``) and when it settles (its views)."""

    def __init__(self) -> None:
        self.versions = {MAIN: 0, "worker": 0}

    def views(self) -> dict[str, PolicyView]:
        return {
            policy: PolicyView(policy, current_version=lambda policy=policy: self.versions[policy])
            for policy in self.versions
        }


def _datum(reward: float) -> tinker.Datum:
    """An observation token, then two action tokens that carry ``reward``, as the converter loads it."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor([0.0, 1.0, 1.0])),
            "advantages": TensorData.from_torch(torch.tensor([0.0, reward, reward])),
            "checkpoint_version": TensorData.from_torch(torch.tensor([0])),
        },
    )


def _agent(
    reward: float | None,
    sampled: int = 0,
    *,
    policy: str = MAIN,
    rows: int = 1,
    name: str = "a",
    role: str = "policy",
    parent: AgentRun | None = None,
) -> AgentRun:
    """An agent of ``policy`` whose rows were sampled at version ``sampled``: a root, or a child of ``parent`` in
    ``role``."""

    depth = 0 if parent is None else parent.depth + 1
    parent_id = None if parent is None else parent.trajectory_id
    agent = AgentRun(f"agent-{name}", f"t-{name}", parent_id, role, policy, depth, reward)
    agent.samples = [
        TrainingSample(_datum(reward or 0.0), policy, 0, agent.agent_id, role, agent.trajectory_id, depth, sampled)
        for _ in range(rows)
    ]
    return agent


def _group(
    clock: Clock, *agents: AgentRun | tuple[AgentRun, ...], columns: dict[str, str] | None = None
) -> CreditGroup:
    """A group of one member per agent (or per tuple of a root and its children), at the clock's versions, of a task
    with ``columns``."""

    members = []
    for index, tree in enumerate(agents):
        tree = tree if isinstance(tree, tuple) else (tree,)
        samples = [sample for agent in tree for sample in agent.samples]
        for sample in samples:
            sample.member = index
        root = tree[0]
        result = TrajectoryCollectionResult(
            datums=[sample.datum for sample in samples],
            task_reward=root.reward or 0.0,
            trajectory_stats=[],
            root_rewards_dict={},
            task_reward_valid=root.reward is not None,
        )
        members.append(MemberRollout(index, samples, list(tree), result.task_reward, root.reward is not None, result))
    return CreditGroup(
        "task",
        None,
        "solo",
        WorkflowConfig(group_size=1),
        members,
        clock.views(),
        dict(clock.versions),
        DeferredStats(),
        dict(columns or {}),
    )


def _metrics(group: CreditGroup) -> dict[str, Any]:
    tracker = StatsTracker()
    group.stats.record(tracker)
    return tracker.export()


def _assigned(estimator: Running, group: CreditGroup) -> dict[str, Any]:
    """Credit ``group`` (no settle), and return its metrics."""

    asyncio.run(estimator.assign(group))
    return _metrics(group)


def _credited(estimator: Running, group: CreditGroup) -> dict[str, Any]:
    """Credit ``group`` and settle it, as the run does, and return its metrics."""

    metrics = _assigned(estimator, group)
    estimator.settle(group)
    return metrics


def _advantage(agent: AgentRun) -> float:
    [value] = {float(sample.advantages[1]) for sample in agent.samples}
    return value


def _settle(estimator: Running, clock: Clock, *rewards: float, columns: dict[str, str] | None = None) -> None:
    """Settle one group of one agent per reward, each sampled at the clock's version."""

    for reward in rewards:
        _credited(estimator, _group(clock, _agent(reward, clock.versions[MAIN]), columns=columns))


# --- the estimator ---------------------------------------------------------------------------------------------------


def test_running_is_core_s_reads_agents_and_credits_groups_of_one():
    assert BUILTIN_CREDIT_ESTIMATORS["running"] == "platoon.train.credit.running:Running"
    assert resolve_credit_estimator("running") is Running
    assert Running.needs_agents is True and Running.credits_single_members is True
    assert RunningOptions() == RunningOptions(
        strata=[], priors=None, half_life=4.0, shrink=8.0, clip=3.0, cap_penalties=True
    )


@pytest.mark.parametrize(
    "options",
    [
        None,
        {"priors": {"mean": 0.25, "scale": 0.5}},
        {
            "strata": ["kind", "mode"],
            "priors": {"mean": 0.25, "scale": 0.5, "weight": 2, "strata": [{"stratum": ["a"], "mean": 1.0}]},
            "half_life": 1,
            "shrink": 0,
            "clip": 1,
            "cap_penalties": False,
        },
    ],
)
def test_running_keeps_the_credit_contract(options):
    check_credit_estimator("running", options=options)


def test_without_priors_agents_train_at_zero_until_what_they_read_holds_enough_settled_samples():
    clock, estimator = Clock(), Running(_SETUP)
    for index in range(WARMUP_SAMPLES - 1):
        assert _credited(estimator, _group(clock, _agent(float(index % 2))))["credit/running/warmup"] == 1.0

    clock.versions[MAIN] = 1  # an agent of version 1 reads what settled at version 0: one sample too few
    early = _agent(1.0, sampled=1)
    metrics = _credited(estimator, _group(clock, early))
    assert _advantage(early) == 0.0 and metrics["credit/running/warmup"] == 1.0
    assert "credit/running/baseline/avg" in metrics  # read, as the warm-up's metrics show it

    clock.versions[MAIN] = 2  # its sample settled all the same: an agent of version 2 reads enough
    late, older = _agent(1.0, sampled=2), _agent(1.0, sampled=1, name="b")
    metrics = _assigned(estimator, _group(clock, late, older))
    assert _advantage(late) > 0 and _advantage(older) == 0.0  # the one of version 1 still reads too few
    assert metrics["credit/running/warmup"] == 0.5


def test_an_agent_reads_the_statistics_as_they_were_when_its_policy_reached_the_version_it_sampled_at():
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)
    clock.versions[MAIN] = 1
    then = _agent(0.5, sampled=1)
    _assigned(estimator, _group(clock, then))

    _settle(estimator, clock, 0.0, 0.0)  # settle at version 1, and then at 2: after the agent began sampling
    clock.versions[MAIN] = 2
    _settle(estimator, clock, 0.0)
    clock.versions[MAIN] = 3
    late, fresh = _agent(0.5, sampled=1), _agent(0.5, sampled=3, name="b")
    metrics = _assigned(estimator, _group(clock, late, fresh))

    assert _advantage(late) == pytest.approx(_advantage(then))  # what settled since it sampled changes nothing
    assert _advantage(then) < 0 < _advantage(fresh)  # the baseline it read was 1; since then it fell
    assert metrics["credit/running/age/avg"] == 1.0  # versions 3 - 1 and 3 - 3


def test_an_agent_reads_at_the_oldest_of_its_rows_versions():
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)
    for version in (1, 2, 3):
        clock.versions[MAIN] = version
        _settle(estimator, clock, 0.0)
    two_rows = _agent(0.5, sampled=1, rows=2)
    two_rows.samples[1].version = 3  # its second row sampled two versions later
    one_row = _agent(0.5, sampled=1, name="b")
    _assigned(estimator, _group(clock, two_rows, one_row))
    assert _advantage(two_rows) == pytest.approx(_advantage(one_row)) == pytest.approx(-0.5)


def test_a_group_settles_at_its_policy_s_version_now_so_a_snapshot_never_changes_once_read():
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)  # version 0
    slow = _group(clock, _agent(0.0, sampled=0))
    asyncio.run(estimator.assign(slow))  # credited at version 0, filed later
    clock.versions[MAIN] = 2
    first = _agent(0.5, sampled=2)
    _assigned(estimator, _group(clock, first))
    estimator.settle(slow)  # filed at version 2: what an agent of version 2 read must stay as it was
    clock.versions[MAIN] = 3
    _settle(estimator, clock, 1.0)
    again = _agent(0.5, sampled=2, name="b")
    _assigned(estimator, _group(clock, again))
    assert _advantage(again) == pytest.approx(_advantage(first))


def test_a_sample_counts_at_full_weight_from_when_it_settles_and_halves_every_half_life():
    clock, estimator = Clock(), Running(_SETUP)
    _settle(estimator, clock, 1.0)
    clock.versions[MAIN] = 4
    slow = _agent(0.0, sampled=0)  # sampled at version 0, settled at 4: it enters at full weight at 4
    _credited(estimator, _group(clock, slow))
    clock.versions[MAIN] = 5
    metrics = _assigned(estimator, _group(clock, _agent(1.0, sampled=5)))

    # The first sample weighs 2^(-5/4) at version 5 and the second 2^(-1/4): the baseline is 1/3. Weighed from when
    # they were sampled, the two would weigh alike, 1/2.
    assert metrics["credit/running/baseline/avg"] == pytest.approx(1 / 3)


def test_a_stratum_s_baseline_is_its_mean_shrunk_toward_its_parent_s_from_the_priors_on():
    priors = RunningPriors(mean=0.0, scale=1.0, weight=4.0, strata=[StratumPrior(["masked"], 1.0)])
    options = RunningOptions(strata=["kind", "mode"], priors=priors, shrink=2.0)
    clock, estimator = Clock(), Running(_SETUP, options)
    numeric, lifesci = {"kind": "masked", "mode": "numeric"}, {"kind": "lifesci"}

    def baseline(columns: dict[str, str]) -> float:
        group = _group(clock, _agent(0.0, sampled=clock.versions[MAIN]), columns=columns)
        return _assigned(estimator, group)["credit/running/baseline/avg"]

    # The root's prior is 0, masked's 1 at weight 4, shrunk toward the root's at weight 2: 4/6. Below it, (masked,
    # numeric) has no weight of its own, and takes its parent's; lifesci has none either, and takes the root's.
    assert baseline(numeric) == pytest.approx(2 / 3)
    assert baseline({"kind": "masked"}) == pytest.approx(2 / 3)
    assert baseline(lifesci) == pytest.approx(0.0) and baseline({}) == pytest.approx(0.0)

    _settle(estimator, clock, 1.0, columns=numeric)
    clock.versions[MAIN] = 1
    decay = 0.5 ** (1 / 4)
    root = (decay * 1.0) / (decay * 5.0)  # the prior's weight 4 at mean 0, and the sample
    masked = (decay * 5.0 + 2 * root) / (decay * 5.0 + 2)
    assert baseline({}) == pytest.approx(root)
    assert baseline(numeric) == pytest.approx((decay * 1.0 + 2 * masked) / (decay * 1.0 + 2))


def test_the_scale_is_the_decayed_root_mean_square_of_the_residuals_and_at_least_its_floor():
    clock = Clock()
    estimator = Running(_SETUP, RunningOptions(priors=RunningPriors(mean=0.0, scale=1.0, weight=1.0)))
    _settle(estimator, clock, 3.0)  # credited against the prior's baseline, 0: a residual of 3
    clock.versions[MAIN] = 1
    assert _assigned(estimator, _group(clock, _agent(0.0, 1)))["credit/running/scale/avg"] == pytest.approx(
        math.sqrt((1.0 + 9.0) / 2)
    )

    tiny = Running(_SETUP, RunningOptions(priors=RunningPriors(mean=0.0, scale=0.001)))
    agent = _agent(0.05)
    metrics = _assigned(tiny, _group(Clock(), agent))
    assert metrics["credit/running/scale/avg"] == MIN_SCALE and _advantage(agent) == pytest.approx(0.5)


def test_the_scale_decays_at_twice_the_half_life_and_takes_each_residual_against_the_baseline_it_read():
    clock = Clock()
    estimator = Running(_SETUP, RunningOptions(priors=RunningPriors(mean=0.0, scale=1.0, weight=1.0)))
    _settle(estimator, clock, 3.0)  # version 0: a residual of 3 against the prior's 0
    slow = _group(clock, _agent(1.0, sampled=0))  # reads version 0's baseline, the prior's 0
    asyncio.run(estimator.assign(slow))
    clock.versions[MAIN] = 4
    estimator.settle(slow)  # settles at 4 with its residual 1 - 0, not 1 - 1.5, the baseline at 4
    clock.versions[MAIN] = 5
    read = _assigned(estimator, _group(clock, _agent(0.0, 5)))
    old, new = 0.5 ** (5 / 8), 0.5 ** (1 / 8)
    expected = math.sqrt((old * (1.0 + 9.0) + new * 1.0) / (old * 2 + new))
    assert read["credit/running/scale/avg"] == pytest.approx(expected)


def test_the_clip_option_bounds_the_advantage():
    estimator = Running(_SETUP, RunningOptions(priors=RunningPriors(mean=0.0, scale=1.0), clip=1.0))
    agent = _agent(2.0)
    metrics = _assigned(estimator, _group(Clock(), agent))
    assert _advantage(agent) == pytest.approx(1.0) and metrics["credit/running/clipped"] == 1.0


@pytest.mark.parametrize(
    ("reward", "cap_penalties", "advantage", "clipped", "capped"),
    [
        (-0.5, True, 0.0, 1.0, 1.0),  # above a baseline of -1, but a penalty: it never pays
        (-0.5, False, 3.0, 1.0, 0.0),
        (-0.95, True, 0.0, 0.0, 1.0),
        (2.0, True, 3.0, 1.0, 0.0),
        (-2.0, True, -3.0, 1.0, 0.0),
        (-1.1, True, -1.0, 0.0, 0.0),
    ],
)
def test_advantages_are_clipped_and_a_penalty_never_pays(reward, cap_penalties, advantage, clipped, capped):
    priors = RunningPriors(mean=-1.0, scale=0.1)
    estimator = Running(_SETUP, RunningOptions(priors=priors, cap_penalties=cap_penalties))
    agent = _agent(reward)
    metrics = _assigned(estimator, _group(Clock(), agent))
    assert _advantage(agent) == pytest.approx(advantage)
    assert (metrics["credit/running/clipped"], metrics["credit/running/capped"]) == (clipped, capped)


def test_the_statistics_are_kept_as_far_back_as_the_rows_may_be_old_and_an_agent_that_sampled_earlier_trains_at_0():
    # Rows that may be 3 versions old: a read reaches back 3 versions from the last settle's, and no further.
    clock, estimator = Clock(), Running(_bounded(3), RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)
    clock.versions[MAIN] = 5
    _settle(estimator, clock, 1.0)  # the versions kept are now 2 to 5
    gone, kept = _agent(0.5, sampled=1), _agent(0.5, sampled=2, name="b")
    metrics = _credited(estimator, _group(clock, gone, kept))

    assert _advantage(gone) == 0.0 and _advantage(kept) == pytest.approx(-0.5)  # it read the first sample's 1
    assert metrics["credit/running/out_of_ring"] == 0.5
    [[policy, role, tracker]] = estimator.state()["trackers"]
    assert (policy, role, tracker["base_version"]) == (MAIN, "policy", 0)  # what the reads at 2 to 5 start from
    assert [settle[0] for settle in tracker["settles"]] == [5, 5, 5]  # every settle since: the agent at 1 settled too

    # A setup that gives no bound (outside a run) keeps DEFAULT_WINDOW versions.
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)
    clock.versions[MAIN] = DEFAULT_WINDOW
    edge, older = _agent(0.5, sampled=1), _agent(0.5, sampled=0, name="b")
    _settle(estimator, clock, 1.0)
    _assigned(estimator, _group(clock, edge, older))
    assert _advantage(edge) == pytest.approx(-0.5) and _advantage(older) == 0.0


def test_a_stratum_without_recent_samples_leaves_the_statistics_and_reads_its_parent_s_baseline():
    clock, estimator = Clock(), Running(_bounded(3), RunningOptions(strata=["kind"], priors=_FLAT))
    _settle(estimator, clock, 1.0, columns={"kind": "a"})
    clock.versions[MAIN] = 80  # 20 half-lives: its weight 2^-20 is about a millionth
    _settle(estimator, clock, 0.0, columns={"kind": "b"})
    clock.versions[MAIN] = 84
    _settle(estimator, clock, 0.0, columns={"kind": "b"})  # version 80's statistics are now the oldest kept
    [[_, _, tracker]] = estimator.state()["trackers"]
    assert [stratum for stratum, _, _ in tracker["base"]["sums"]] == [[], ["b"]]

    clock.versions[MAIN] = 85
    root, gone = (
        _assigned(estimator, _group(clock, _agent(0.0, 85), columns=columns))["credit/running/baseline/avg"]
        for columns in ({}, {"kind": "a"})
    )
    assert gone == root


def test_an_unscored_agent_trains_nothing_and_moves_nothing_and_one_without_rows_settles():
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    unscored = _agent(None)
    unscored.samples[0].set_advantage(0.75)  # a reward the converter loaded, which it must not train on
    _credited(estimator, _group(clock, unscored))
    assert _advantage(unscored) == 0.0 and estimator.state() == {"trackers": []}

    sampled_out = _agent(1.0, rows=0)
    _credited(estimator, _group(clock, sampled_out))
    [[_, _, tracker]] = estimator.state()["trackers"]
    assert tracker["settles"] == [[0, [], 1.0, 1.0]]  # against the statistics now: the priors' 0


@pytest.mark.parametrize("reward", [math.nan, math.inf, -math.inf])
def test_an_agent_whose_reward_is_not_finite_trains_nothing_and_moves_nothing(reward, caplog):
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _settle(estimator, clock, 1.0)
    before = estimator.state()
    broken, other = _agent(reward), _agent(reward, name="b")
    with caplog.at_level(logging.WARNING, logger="platoon.train.credit.running"):
        _credited(estimator, _group(clock, broken, other))

    assert _advantage(broken) == 0.0 == _advantage(other)
    assert estimator.state() == before
    [warning] = [record.getMessage() for record in caplog.records]
    assert f"agent agent-a of task task has a reward of {reward}, which is no finite number" in warning
    clock.versions[MAIN] = 1
    later = _agent(0.5, sampled=1, name="c")
    _assigned(estimator, _group(clock, later))
    assert _advantage(later) == pytest.approx(-0.5)  # the statistics as they were: the one sample's 1


def test_each_policy_s_agents_have_statistics_of_their_own_aged_in_its_versions():
    clock, estimator = Clock(), Running(_SETUP, RunningOptions(priors=_FLAT))
    _credited(estimator, _group(clock, _agent(1.0), _agent(-1.0, policy="worker", name="w")))
    clock.versions["worker"] = 1  # the worker's policy moved on, main's did not
    ours, theirs = _agent(1.0, sampled=1), _agent(1.0, policy="worker", sampled=1, name="w")
    clock.versions[MAIN] = 1
    _assigned(estimator, _group(clock, ours, theirs))

    assert _advantage(ours) == pytest.approx(0.0, abs=1e-6)  # main's baseline: main's sample alone, 1
    assert _advantage(theirs) > 0  # the worker's baseline: its own sample, -1
    assert [(policy, role) for policy, role, _ in estimator.state()["trackers"]] == [
        (MAIN, "policy"),
        ("worker", "policy"),
    ]


def test_each_role_of_a_policy_has_statistics_of_its_own():
    # A root whose reward is always 0 and its subagent's always 1: each at its own role's mean, so once the warm-up
    # is over both train at exactly 0. Sharing statistics, each would be centered on the other's rewards too.
    clock, estimator = Clock(), Running(_bounded(2))
    for index in range(WARMUP_SAMPLES + 4):
        clock.versions[MAIN] = index // 4
        root = _agent(0.0, clock.versions[MAIN], name=f"r{index}")
        helper = _agent(1.0, clock.versions[MAIN], name=f"h{index}", role="helper", parent=root)
        metrics = _credited(estimator, _group(clock, (root, helper)))

    assert metrics["credit/running/warmup"] == 0.0
    assert _advantage(root) == 0.0 and _advantage(helper) == 0.0
    assert [(policy, role) for policy, role, _ in estimator.state()["trackers"]] == [
        (MAIN, "helper"),
        (MAIN, "policy"),
    ]


def test_a_task_without_a_strata_column_is_credited_at_the_stratum_above_and_warned_of_once(caplog):
    priors = RunningPriors(mean=0.0, scale=1.0, weight=4.0, strata=[StratumPrior(["masked"], 1.0)])
    estimator = Running(_SETUP, RunningOptions(strata=["kind", "mode"], priors=priors, shrink=2.0))
    with caplog.at_level(logging.WARNING, logger="platoon.train.credit.running"):
        for _ in range(2):
            metrics = _assigned(estimator, _group(Clock(), _agent(0.0), columns={"kind": "masked"}))
            assert metrics["credit/running/baseline/avg"] == pytest.approx(2 / 3)  # masked's
            assert _assigned(estimator, _group(Clock(), _agent(0.0), columns={"mode": "numeric"}))[
                "credit/running/baseline/avg"
            ] == pytest.approx(0.0)  # no kind: the root's
    warned = [record.getMessage() for record in caplog.records]
    assert warned == [
        "environments[solo].credit_options.strata: task task has no mode column, so the agents of a task without it "
        "are credited at the stratum above it (warned once per column)",
        "environments[solo].credit_options.strata: task task has no kind column, so the agents of a task without it "
        "are credited at the stratum above it (warned once per column)",
    ]


def test_a_resume_from_its_state_credits_as_the_run_would_have():
    options = RunningOptions(strata=["kind"], priors=RunningPriors(mean=0.0, scale=0.5))
    clock, running = Clock(), Running(_SETUP, options)
    for version in range(6):
        clock.versions[MAIN] = version
        _settle(running, clock, version / 5, columns={"kind": "a" if version % 2 else "b"})
    resumed = Running(_SETUP, options)
    resumed.load_state(json.loads(json.dumps(running.state())))

    clock.versions[MAIN] = 7
    credits = []
    for estimator in (running, resumed):
        agents = [_agent(0.5, sampled=version, name=str(version)) for version in (3, 6, 7)]
        _credited(estimator, _group(clock, *agents, columns={"kind": "a"}))
        credits.append([_advantage(agent) for agent in agents])
    assert credits[0] == credits[1] and len(set(credits[0])) == 3
    assert running.state() == resumed.state()


def _busy_run(estimator: Running, clock: Clock, versions: range, rng: random.Random) -> list[float]:
    """Credit and settle 16 groups at each of ``versions``, each of one agent that sampled up to 3 versions before, of
    a task of one of 90 strata (``[kind, mode, domain]``); the advantages, in order."""

    advantages = []
    for version in versions:
        clock.versions[MAIN] = version
        for index in range(16):
            columns = {"kind": rng.choice("abc"), "mode": rng.choice("xyz"), "domain": f"d{rng.randrange(10)}"}
            agent = _agent(rng.random(), sampled=max(version - rng.randrange(4), 0), name=f"{version}-{index}")
            _credited(estimator, _group(clock, agent, columns=columns))
            advantages.append(_advantage(agent))
    return advantages


def test_the_state_holds_one_copy_of_the_statistics_and_the_settles_the_bound_reaches_and_resumes_exactly():
    options = RunningOptions(strata=["kind", "mode", "domain"], priors=RunningPriors(mean=0.5, scale=0.3))
    clock, running = Clock(), Running(_bounded(10), options)
    _busy_run(running, clock, range(100), random.Random(0))

    state = running.state()
    [[_, _, tracker]] = state["trackers"]
    # The window ends at version 99 and reaches 10 versions back: 89 to 99, 11 versions of 16 settles, and the
    # statistics after version 88's, where the reads before any settle at 89 start. A copy of the statistics for each
    # version kept would be about 20 times as large.
    assert tracker["base_version"] == 88 and len(tracker["settles"]) == 11 * 16
    assert len(json.dumps(state)) < 30_000

    resumed = Running(_bounded(10), options)
    resumed.load_state(json.loads(json.dumps(state)))
    assert resumed.state() == state
    after = [_busy_run(estimator, Clock(), range(100, 130), random.Random(1)) for estimator in (running, resumed)]
    assert after[0] == after[1] and len(set(after[0])) > 1  # to the bit
    assert running.state() == resumed.state()


# --- its options -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "says"),
    [
        ({"strata": ["kind", "kind"]}, "credit_options.strata: each column once"),
        ({"strata": ["task_id"]}, "credit_options.strata: task_id is platoon's own column"),
        ({"strata": ["_platoon_environment"]}, "credit_options.strata: _platoon_environment is platoon's own column"),
        ({"strata": "kind"}, "credit_options.strata: expected a list"),
        ({"half_life": 0}, "credit_options.half_life: a finite number above 0, got 0"),
        ({"shrink": -1}, "credit_options.shrink: a finite number of 0 or more, got -1"),
        ({"clip": 0}, "credit_options.clip: a number above 0, got 0"),
        ({"priors": {"mean": 0.0}}, "credit_options.priors.scale: required key is missing"),
        ({"priors": {"mean": 0.0, "scale": 0.0}}, "credit_options.priors: scale: a finite number above 0, got 0.0"),
        ({"priors": {"mean": 0.0, "scale": 1.0, "wieght": 2}}, "did you mean credit_options.priors.weight?"),
        (
            {"strata": ["kind"], "priors": {"mean": 0, "scale": 1, "strata": [{"stratum": ["a", "b"], "mean": 0}]}},
            "credit_options.priors: strata.0.stratum has 2 value(s), more than strata names columns (kind)",
        ),
        (
            {"priors": {"mean": 0, "scale": 1, "strata": [{"stratum": ["a"], "mean": 0}]}},
            "credit_options.priors: strata.0.stratum has 1 value(s), more than strata names columns (none)",
        ),
        (
            {
                "strata": ["kind"],
                "priors": {
                    "mean": 0,
                    "scale": 1,
                    "strata": [{"stratum": ["a"], "mean": 0}, {"stratum": ["a"], "mean": 1}],
                },
            },
            "credit_options.priors: strata: each stratum once, got [['a'], ['a']]",
        ),
        (
            {"strata": ["kind"], "priors": {"mean": 0, "scale": 1, "strata": [{"stratum": [], "mean": 0}]}},
            "stratum: one or more",
        ),
    ],
)
def test_options_it_cannot_serve_are_refused_under_their_path(options, says):
    _, problems = schema_options(RunningOptions, name="running").check(options, "credit_options")
    assert any(says in problem for problem in problems), problems


def test_a_prior_s_stratum_is_written_as_the_task_s_columns_hold_their_values():
    from platoon.train.workflow.crediting import task_columns

    priors = {"mean": 0, "scale": 1, "strata": [{"stratum": [3], "mean": 1}, {"stratum": [True], "mean": -1}]}
    parsed, problems = schema_options(RunningOptions, name="running").check(
        {"strata": ["level"], "priors": priors}, "credit_options"
    )
    assert not problems and [prior.stratum for prior in parsed.priors.strata] == [["3"], ["true"]]

    # A task row holds the integer 3 and the boolean true, which an estimator reads as text (CreditGroup.columns).
    estimator = Running(_SETUP, parsed)
    baselines = [
        _assigned(estimator, _group(Clock(), _agent(0.0), columns=task_columns(row)))["credit/running/baseline/avg"]
        for row in ({"task_id": "t", "level": 3}, {"task_id": "t", "level": True})
    ]
    assert baselines == pytest.approx([2 / 3, -2 / 3])  # each prior's 1 or -1 at weight 16, shrunk toward the root's 0

    _, problems = schema_options(RunningOptions, name="running").check(
        {"strata": ["level"], "priors": {"mean": 0, "scale": 1, "strata": [{"stratum": [""], "mean": 1}]}},
        "credit_options",
    )
    assert any("stratum: one or more values of the strata columns" in problem for problem in problems), problems


def test_an_entry_whose_rows_have_no_staleness_bound_is_refused():
    from platoon.testing.credit import fixture_config
    from platoon.train.credit.api import CreditCheckContext

    unbounded = fixture_config()
    with pytest.raises(ValueError, match="they have no staleness bound; set train.max_staleness"):
        Running.check(RunningOptions(), CreditCheckContext(entry="fixture", policies=(MAIN,), config=unbounded))
    bounded = dataclasses.replace(unbounded, train=dataclasses.replace(unbounded.train, max_staleness=2))
    Running.check(RunningOptions(), CreditCheckContext(entry="fixture", policies=(MAIN,), config=bounded))


def test_the_priors_may_come_from_a_yaml_or_json_file(tmp_path):
    written = {"mean": -0.25, "scale": 0.5, "weight": 8, "strata": [{"stratum": ["masked"], "mean": 0.1}]}
    (tmp_path / "priors.json").write_text(json.dumps(written))
    (tmp_path / "priors.yaml").write_text(
        "mean: -0.25\nscale: 0.5\nweight: 8\nstrata:\n  - {stratum: [masked], mean: 0.1}\n"
    )
    expected = RunningPriors(mean=-0.25, scale=0.5, weight=8, strata=[StratumPrior(["masked"], 0.1)])
    for name in ("priors.json", "priors.yaml"):
        options = RunningOptions(strata=["kind"], priors=str(tmp_path / name))
        assert options.priors == expected == read_priors(tmp_path / name)


@pytest.mark.parametrize(
    ("content", "says"),
    [
        (None, "cannot be read (No such file or directory)"),
        ("mean: [", "is not YAML or JSON"),
        ("- 1\n", "holds list, not a mapping of priors"),
        ("mean: 0\n", "priors.yaml: scale: required key is missing"),
        ("mean: 0\nscale: 1\nstrata: [{stratum: [a], mean: 0}]\n", "priors.yaml: strata.0.stratum has 1 value(s)"),
    ],
)
def test_a_priors_file_that_holds_no_priors_is_refused_with_why(tmp_path, content, says):
    path = tmp_path / "priors.yaml"
    if content is not None:
        path.write_text(content)
    _, problems = schema_options(RunningOptions, name="running").check({"priors": str(path)}, "credit_options")
    assert any(problem.startswith(f"credit_options.priors: {path}") for problem in problems), problems
    assert any(says in problem for problem in problems), problems


def test_the_priors_need_not_name_every_stratum_and_their_scale_and_weight_seed_the_statistics():
    priors = RunningPriors(mean=0.5, scale=2.0, weight=3.0, strata=[StratumPrior(["a", "x"], -1.0)])
    estimator = Running(_SETUP, RunningOptions(strata=["kind", "mode"], priors=priors, shrink=0.0))
    agent = _agent(0.5)
    metrics = _assigned(estimator, _group(Clock(), agent, columns={"kind": "a", "mode": "x"}))
    # Without shrinkage, (a, x) is its prior's mean; the scale is the priors'.
    assert metrics["credit/running/baseline/avg"] == pytest.approx(-1.0)
    assert metrics["credit/running/scale/avg"] == pytest.approx(2.0) and _advantage(agent) == pytest.approx(0.75)


def test_a_priors_file_is_read_when_the_options_are_parsed(tmp_path):
    path = Path(tmp_path / "later.yaml")
    path.write_text("mean: 0\nscale: 1\n")
    options = RunningOptions(priors=str(path))
    path.unlink()  # what was read stays
    assert isinstance(options.priors, RunningPriors) and Running(_SETUP, options).options.priors == options.priors
