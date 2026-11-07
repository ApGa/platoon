"""``credit: roles``: each role of an entry's trees credited by its own estimator, on a view of the group's agents of
that role, its metrics under its own prefix, the rows of a role it does not name at 0; a reverse KL term per role,
scored as ``reverse_kl`` scores; a share of the advantage mass per role, held by a weight read before each group's
credit from the masses as they were when the role began sampling, whatever settled since, moved only as groups settle,
per policy and aged in its versions, as the roles' masses drift, starting over when a role with a share first carries
mass; its metrics; its options, its load-time check, its build-time one and its estimators as ``lock.json`` records
them; errors that name the role; its state, exact through a resume; and the conformance check."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

import pytest
import tinker
import torch
from tinker import TensorData

import platoon.experimental as px
from platoon.options import schema_options
from platoon.testing import check_credit_estimator
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import (
    AgentRun,
    CreditGroup,
    CreditSetup,
    MemberRollout,
    PolicyView,
    ScoringFailed,
    TrainingSample,
)
from platoon.train.credit.keyed import Keyed
from platoon.train.credit.registry import resolve_credit_estimator
from platoon.train.credit.roles import MASS_HALF_LIFE, RoleCredit, RoleKL, Roles, RolesOptions
from platoon.train.credit.running import Running
from platoon.train.policy.spec import MAIN
from platoon.utils.config import ConfigError
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

_SETUP = CreditSetup(name="roles", entry="envgen", settings=WorkflowConfig(group_size=1), policies=(MAIN, "worker"))
# A running baseline that trains from the first group on: a prior of mean 0 and scale 1 that weighs nothing.
_FLAT = {"priors": {"mean": 0.0, "scale": 1.0, "weight": 1e-9}}


class AsReward(px.CreditEstimatorBase):
    """Each scored agent's rows at its reward: what an estimator gives, so that a test sets each role's mass."""

    needs_agents = True

    async def assign(self, group: px.CreditGroup) -> None:
        for agent in group.agents():
            for sample in agent.samples:
                sample.set_advantage(agent.reward or 0.0)


class Recording(AsReward):
    """:class:`AsReward`, recording the roles of each view it credits and settles, and a metric."""

    seen: list[tuple[str, list[str]]] = []

    async def assign(self, group: px.CreditGroup) -> None:
        await super().assign(group)
        Recording.seen.append(("assign", [agent.role for agent in group.agents()]))
        group.stats.scalar(**{"credit/recording/agents": float(len(group.agents()))})

    def settle(self, group: px.CreditGroup) -> None:
        Recording.seen.append(("settle", [agent.role for agent in group.agents()]))


class Clock:
    """The policies' versions, which a test moves on as training would."""

    def __init__(self) -> None:
        self.versions = {MAIN: 0, "worker": 0}

    def views(self) -> dict[str, PolicyView]:
        return {
            policy: PolicyView(policy, current_version=lambda policy=policy: self.versions[policy])
            for policy in self.versions
        }


def _datum(tokens: int, logprob: float = -1.0) -> tinker.Datum:
    """An observation token, then ``tokens`` action tokens sampled at ``logprob``, whose advantages the estimator
    sets."""

    mask = [0.0] + [1.0] * tokens
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(mask)))),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor(mask)),
            "advantages": TensorData.from_torch(torch.zeros(len(mask))),
            "logprobs": TensorData.from_torch(torch.full((len(mask),), logprob)),
            "checkpoint_version": TensorData.from_torch(torch.tensor([0])),
        },
    )


def _agent(
    name: str, role: str, reward: float | None, *, tokens: int = 2, policy: str = MAIN, parent: AgentRun | None = None
) -> AgentRun:
    """An agent of ``role`` with one row of ``tokens`` action tokens (none for an unscored one)."""

    depth, parent_id = (0, None) if parent is None else (parent.depth + 1, parent.trajectory_id)
    agent = AgentRun(name, f"t-{name}", parent_id, role, policy, depth, reward)
    if reward is not None:
        agent.samples = [TrainingSample(_datum(tokens), policy, 0, name, role, agent.trajectory_id, depth, 0)]
    return agent


def _group(clock: Clock, *trees: tuple[AgentRun, ...]) -> CreditGroup:
    """A group of one member per tree (a root, then its children), sampled and credited at the clock's versions."""

    members = []
    for index, tree in enumerate(trees):
        samples = [sample for agent in tree for sample in agent.samples]
        for sample in samples:
            sample.member, sample.version = index, clock.versions[sample.policy]  # sampled now
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
        "envgen",
        WorkflowConfig(group_size=1),
        members,
        clock.views(),
        dict(clock.versions),
        DeferredStats(),
    )


def _production(generator: float | None, solvers: list[float], *, tokens: int = 2) -> tuple[AgentRun, ...]:
    """A generator root and its solver children."""

    root = _agent("generator", "generator", generator, tokens=tokens)
    return (root, *(_agent(f"solver-{index}", "solver", reward, parent=root) for index, reward in enumerate(solvers)))


def _estimator(setup: CreditSetup = _SETUP, **by_role: dict[str, Any]) -> Roles:
    options = schema_options(RolesOptions).parse({"by_role": by_role}, "credit_options")
    return Roles(setup, options)


def _credit(estimator: Roles, group: CreditGroup) -> CreditGroup:
    asyncio.run(estimator.assign(group))
    estimator.settle(group)
    return group


def _advantages(agent: AgentRun) -> list[list[float]]:
    return [sample.advantages.tolist() for sample in agent.samples]


def _mass(group: CreditGroup, role: str) -> float:
    rows = [sample for agent in group.agents() if agent.role == role for sample in agent.samples]
    return sum(float(sample.advantages[sample.mask > 0].abs().sum()) for sample in rows)


def _exported(group: CreditGroup) -> dict[str, float]:
    tracker = StatsTracker()
    group.stats.record(tracker)
    return tracker.export()


_AS_REWARD = f"{__name__}:AsReward"


# --- the component ---------------------------------------------------------------------------------------------------


def test_roles_is_core_s_reads_agents_credits_groups_of_one_and_limits_no_assign_of_its_own():
    assert resolve_credit_estimator("roles") is Roles
    assert Roles.needs_agents is True and Roles.credits_single_members is True
    assert Roles.assign_timeout_seconds is None  # each role's estimator awaits under its own limit
    with pytest.raises(ValueError, match="is core's"):
        px.credit_estimator("Roles")


@pytest.mark.parametrize(
    "by_role",
    [
        {"policy": {"credit": "running", "options": _FLAT, "share": 0.25}, "helper": {"credit": "keyed"}},
        {"policy": {"credit": "keyed", "options": {"key": ["role"]}}, "helper": {"credit": "running", "share": 0.4}},
        {"helper": {"credit": "group"}},  # the roots, which by_role does not name, train at 0
    ],
)
def test_roles_keeps_the_credit_contract(by_role):
    check_credit_estimator("roles", options={"by_role": by_role})


# --- each role's estimator --------------------------------------------------------------------------------------------


def test_each_role_is_credited_by_its_own_estimator_on_a_view_of_its_agents_alone():
    clock, Recording.seen = Clock(), []
    estimator = _estimator(
        generator={"credit": "running", "options": _FLAT},
        solver={"credit": "keyed", "options": {"key": ["role"], "leave_one_out": True}},
        judge={"credit": f"{__name__}:Recording"},
    )
    group = _credit(estimator, _group(clock, _production(0.5, [1.0, 0.0, 1.0, 0.0])))

    root, *solvers = group.agents()
    assert _advantages(root) == [[0.0, 0.5, 0.5]]  # running: (0.5 - 0) / 1, as the generator's only estimator
    # keyed with leave_one_out among the solvers alone: each against the mean of the other three
    assert [_advantages(solver)[0][1] for solver in solvers] == pytest.approx([1 - 1 / 3, -2 / 3, 1 - 1 / 3, -2 / 3])
    assert Recording.seen == []  # a role whose agents are not in the group is not credited, nor settled
    assert isinstance(estimator._estimators["generator"], Running)
    assert isinstance(estimator._estimators["solver"], Keyed)

    judge = _agent("judge", "judge", 1.0)
    _credit(estimator, _group(clock, (*_production(0.5, [1.0]), judge)))
    assert Recording.seen == [("assign", ["judge"]), ("settle", ["judge"])]  # its view: its agents alone


def test_each_role_s_estimator_records_its_metrics_under_its_own_prefix():
    clock = Clock()
    estimator = _estimator(
        generator={"credit": "running", "options": _FLAT},
        solver={"credit": "keyed", "options": {"key": ["role"], "leave_one_out": True}},
    )
    metrics = _exported(_credit(estimator, _group(clock, _production(0.5, [1.0, 0.0]))))

    assert metrics["credit/roles/solver/keyed/baselines"] == 1.0
    assert metrics["credit/roles/solver/agent_advantage/abs_avg"] == pytest.approx(1.0)
    assert metrics["credit/roles/generator/running/advantage/avg"] == pytest.approx(0.5)
    assert metrics["credit/roles/generator/agent_advantage/abs_avg"] == pytest.approx(0.5)
    assert not [name for name in metrics if name.startswith(("credit/keyed/", "credit/running/", "credit/agent_"))]


def test_the_agents_of_a_role_by_role_does_not_name_train_nothing_and_are_warned_of_once(caplog):
    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD})
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            group = _credit(estimator, _group(clock, _production(1.0, [1.0, 0.5])))
    root, *solvers = group.agents()
    assert _advantages(root) == [[0.0, 1.0, 1.0]]
    assert all(_advantages(solver) == [[0.0, 0.0, 0.0]] for solver in solvers)
    [line] = [record.getMessage() for record in caplog.records if "train nothing" in record.getMessage()]
    assert "the agents of role solver train nothing" in line and "by_role names generator" in line


class Raises(AsReward):
    async def assign(self, group: px.CreditGroup) -> None:
        raise KeyError("a bug")

    def settle(self, group: px.CreditGroup) -> None:
        raise KeyError("a bug in settle")


class Slow(AsReward):
    assign_timeout_seconds = 0.01

    async def assign(self, group: px.CreditGroup) -> None:
        await asyncio.sleep(1)


class Fails(AsReward):
    async def assign(self, group: px.CreditGroup) -> None:
        raise ScoringFailed("the teacher did not answer")


@pytest.mark.parametrize(
    ("cls", "error", "says"),
    [
        (Raises, RuntimeError, r"by_role\.solver: .*Raises's assign\(group\) raised KeyError: 'a bug'"),
        (Slow, RuntimeError, r"by_role\.solver: .*Slow's assign\(group\) did not return within 0.01 s"),
        (Fails, ScoringFailed, "the teacher did not answer"),
    ],
)
def test_an_estimator_that_raises_or_runs_out_of_time_names_its_role_and_a_failed_scoring_fails_the_group(
    cls, error, says
):
    estimator = _estimator(generator={"credit": _AS_REWARD}, solver={"credit": f"{__name__}:{cls.__name__}"})
    with pytest.raises(error, match=says) as raised:
        asyncio.run(estimator.assign(_group(Clock(), _production(1.0, [1.0]))))
    assert type(raised.value) is error  # a failed scoring goes through unwrapped, and fails the group alone


def test_a_settle_that_raises_names_its_role():
    estimator = _estimator(generator={"credit": _AS_REWARD}, solver={"credit": f"{__name__}:Raises"})
    group = _group(Clock(), _production(1.0, []))  # no solver: its estimator neither credits nor settles it
    _credit(estimator, group)
    with pytest.raises(RuntimeError, match=r"by_role\.solver: .*Raises's settle\(group\) raised KeyError"):
        estimator.settle(_group(Clock(), _production(1.0, [1.0])))


# --- the KL term -----------------------------------------------------------------------------------------------------


def _scorer(logprob: float, calls: list[int]) -> PolicyView:
    """A frozen policy that scores every target token at ``logprob``, counting the rows it was asked to score."""

    async def score(datums):
        calls.append(len(datums))
        return [torch.full_like(datum.loss_fn_inputs["logprobs"].to_torch(), logprob) for datum in datums]

    return PolicyView("reference", scorer=score)


def test_a_kl_term_adds_the_reverse_kl_to_its_role_s_scored_rows_after_its_estimator():
    calls: list[int] = []
    setup = CreditSetup(
        name="roles",
        entry="envgen",
        settings=WorkflowConfig(),
        policies=(MAIN,),
        policy_views={"reference": _scorer(-1.5, calls)},
    )
    estimator = _estimator(
        setup, generator={"credit": _AS_REWARD, "kl": {"to": "reference", "coef": 0.5}}, solver={"credit": _AS_REWARD}
    )
    unscored = _agent("lost", "generator", None)
    group = _credit(estimator, _group(Clock(), _production(1.0, [1.0]), (unscored,)))

    root, solver, _ = group.agents()
    # sampled at -1, scored at -1.5: a reverse KL of 0.5 at each token, so 1 - 0.5 * 0.5 on each action token
    assert _advantages(root) == [[0.0, 0.75, 0.75]]
    assert _advantages(solver) == [[0.0, 1.0, 1.0]]  # its role has no KL term
    assert calls == [1]  # the scored agent's one row; an unscored agent's are none
    assert _exported(group)["credit/role_kl/generator"] == pytest.approx(0.5)


def test_a_kl_term_whose_scoring_fails_fails_the_group():
    async def fails(datums):
        raise RuntimeError("the reference's service is down")

    setup = CreditSetup(
        name="roles",
        entry="envgen",
        settings=WorkflowConfig(),
        policies=(MAIN,),
        policy_views={"reference": PolicyView("reference", scorer=fails)},
    )
    estimator = _estimator(setup, generator={"credit": _AS_REWARD, "kl": {"to": "reference", "coef": 1.0}})
    with pytest.raises(ScoringFailed, match="could not score 1 row"):
        asyncio.run(estimator.assign(_group(Clock(), _production(1.0, []))))


# --- the share -------------------------------------------------------------------------------------------------------


def test_a_role_trains_at_weight_1_until_its_mass_and_the_unshared_roles_mass_are_known():
    """Every group here began sampling at version 0, before anything settled: each reads the masses as they are when
    its credit begins."""

    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.1}, solver={"credit": _AS_REWARD})
    group = _group(clock, _production(2.0, [1.0, 1.0]))
    asyncio.run(estimator.assign(group))
    assert _advantages(group.agents()[0]) == [[0.0, 2.0, 2.0]]  # no group settled yet
    assert "credit/role_weight/generator" not in _exported(group)

    estimator.settle(group)  # masses: the generator's 4, the solvers' 4
    nothing = _credit(estimator, _group(clock, _production(0.0, [1.0])))  # a generator whose rows carry no mass
    assert _advantages(nothing.agents()[0]) == [[0.0, 0.0, 0.0]]
    after = _group(clock, _production(2.0, [1.0, 1.0]))
    asyncio.run(estimator.assign(after))
    weight = 0.1 / 0.9 * (4 + 2) / 4  # share / (1 - share) * M_solver / M_generator
    assert _advantages(after.agents()[0]) == [pytest.approx([0.0, 2 * weight, 2 * weight])]
    assert [_advantages(solver) for solver in after.agents()[1:]] == [[[0.0, 1.0, 1.0]]] * 2  # unshared: as credited
    assert _exported(after)["credit/role_weight/generator"] == pytest.approx(weight)


def test_the_weight_is_read_before_the_group_is_credited_keeps_signs_and_moves_only_in_settle():
    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.2}, solver={"credit": _AS_REWARD})
    _credit(estimator, _group(clock, _production(1.0, [1.0])))
    before = json.dumps(estimator.state())
    first, second = _group(clock, _production(-4.0, [1.0])), _group(clock, _production(4.0, [-1.0]))
    asyncio.run(estimator.assign(first))
    asyncio.run(estimator.assign(second))
    assert json.dumps(estimator.state()) == before  # assign alone moves nothing
    weight = 0.2 / 0.8 * 2 / 2
    assert _advantages(first.agents()[0]) == [[0.0, -4.0 * weight, -4.0 * weight]]  # its own mass changes nothing
    assert _advantages(second.agents()[0]) == [[0.0, 4.0 * weight, 4.0 * weight]]
    estimator.settle(first)
    assert json.dumps(estimator.state()) != before


def test_a_weight_reads_the_masses_as_they_were_when_its_role_began_sampling_whatever_settled_since():
    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.2}, solver={"credit": _AS_REWARD})
    _credit(estimator, _group(clock, _production(1.0, [1.0])))  # at version 0: masses of 2 and 2
    clock.versions[MAIN] = 1
    early, late = _group(clock, _production(1.0, [1.0])), _group(clock, _production(-1.0, [1.0]))  # sampled at 1
    _credit(estimator, early)
    for version in (1, 2, 2):  # then groups whose solvers carry ever more mass settle, into version 2
        clock.versions[MAIN] = version
        _credit(estimator, _group(clock, _production(1.0, [3.0, 3.0])))
    asyncio.run(estimator.assign(late))  # credited two versions later, it reads what settled before version 1

    weight = 0.2 / 0.8 * 2 / 2
    assert _exported(early)["credit/role_weight/generator"] == pytest.approx(weight)
    assert _exported(late)["credit/role_weight/generator"] == pytest.approx(weight)
    assert _advantages(late.agents()[0]) == [[0.0, -weight, -weight]]
    fresh = _group(clock, _production(1.0, [1.0]))  # sampled at version 2: what settled before it, early's included
    asyncio.run(estimator.assign(fresh))
    assert _exported(fresh)["credit/role_weight/generator"] > 1.5 * weight


def test_every_group_sampled_at_one_version_trains_at_one_weight_however_long_it_took_to_settle():
    """Productions sampled at each version, whose failures settle at once and whose successes six versions later, while
    the solvers' mass grows: a weight read when each group is credited would differ between the two and lean toward
    the late ones; read at the version each began sampling, it is the same for both (but at version 0, before anything
    settled, where each reads the masses as they are), and the weighted outcomes keep the sign of the raw ones."""

    clock, rng = Clock(), random.Random(5)
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.1}, solver={"credit": _AS_REWARD})
    pending: list[tuple[int, int, CreditGroup]] = []
    weights: dict[int, set[float]] = {}
    outcomes, weighted = [], []
    for version in range(60):
        clock.versions[MAIN] = version
        for _ in range(8):
            outcome = rng.choice([-1.0, 1.0])
            solvers = [rng.choice([-1, 1]) * (1 + version / 30) * 0.5 for _ in range(8)]
            pending.append((version + (0 if outcome < 0 else 6), version, _group(clock, _production(outcome, solvers))))
        for due, sampled, group in [item for item in pending if item[0] <= version]:
            _credit(estimator, group)
            weight = _exported(group).get("credit/role_weight/generator")
            weights.setdefault(sampled, set()).add(round(weight, 12) if weight is not None else 1.0)
            outcomes.append(group.agents()[0].reward)
            weighted.append(group.agents()[0].samples[0].advantages[1].item())
        pending = [item for item in pending if item[0] > version]

    assert [version for version, seen in sorted(weights.items()) if len(seen) > 1] == [0]
    assert sum(outcomes) < 0 and sum(weighted) < 0


def test_a_role_with_a_share_starts_its_policy_s_masses_over_from_the_first_group_where_it_carries_mass():
    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.1}, solver={"credit": _AS_REWARD})
    for _ in range(3):  # a generator that carries no mass yet, as in its estimator's warm-up: solvers' mass of 12
        _credit(estimator, _group(clock, _production(0.0, [1.0, 1.0])))
    first = _credit(estimator, _group(clock, _production(1.0, [1.0])))
    assert "credit/role_weight/generator" not in _exported(first)  # its mass was not known: weight 1
    second = _credit(estimator, _group(clock, _production(1.0, [1.0])))
    # Against the first group's masses alone, 2 and 2, not the solvers' 14 of every group so far.
    assert _exported(second)["credit/role_weight/generator"] == pytest.approx(0.1 / 0.9 * 2 / 2)
    _credit(estimator, _group(clock, _production(0.0, [1.0])))  # no mass once it has some: nothing starts over
    assert estimator._masses[MAIN].live() == {"generator": 4.0, "solver": 6.0}


def test_the_weight_after_a_running_role_s_warm_up_is_near_its_level_after():
    """A generator under running without priors, which trains at 0 for its first settles, beside eight solvers
    leave-one-out: had the solvers' mass kept the warm-up's groups, against a generator's mass of a group or two, the
    first weights after it would be several times their later level."""

    setup = CreditSetup(
        name="roles", entry="envgen", settings=WorkflowConfig(group_size=1), policies=(MAIN,), max_staleness={MAIN: 10}
    )
    clock, rng = Clock(), random.Random(1)
    estimator = _estimator(
        setup,
        generator={"credit": "running", "share": 0.1},
        solver={"credit": "keyed", "options": {"key": ["role"], "leave_one_out": True}},
    )
    weights = []
    for index in range(160):
        solvers = [rng.choice([0.0, 1.0]) for _ in range(8)]
        group = _credit(estimator, _group(clock, _production(rng.choice([-1.0, 0.0, 0.5, 1.0]), solvers)))
        weights.append(_exported(group).get("credit/role_weight/generator"))
        if index % 8 == 7:
            clock.versions[MAIN] += 1

    first = next(index for index, weight in enumerate(weights) if weight is not None)
    later = sorted(weight for weight in weights[80:] if weight is not None)
    level = later[len(later) // 2]
    assert all(level / 2 < weight < 2 * level for weight in weights[first : first + 12] if weight is not None)


def test_a_share_holds_its_role_s_share_of_the_mass_as_the_roles_masses_drift():
    """A generator's rewards that grow as the run goes on, and solvers' that shrink: without a share the generator's
    part of the mass more than doubles; with one it stays at its share."""

    def realized(by_role: dict[str, Any]) -> tuple[float, float]:
        clock, rng = Clock(), random.Random(7)
        estimator = _estimator(**by_role)
        parts = []
        for index in range(800):
            progress = index / 800
            generator = rng.choice([-1, 1]) * (1 + 1.5 * progress) * rng.uniform(0.5, 1.5)
            solvers = [rng.choice([-1, 1]) * (1 - 0.4 * progress) * rng.uniform(0, 1) for _ in range(8)]
            group = _credit(estimator, _group(clock, _production(generator, solvers, tokens=4)))
            parts.append((_mass(group, "generator"), _mass(group, "solver")))
            if index % 8 == 7:
                clock.versions[MAIN] += 1  # a step every 8 groups
        early, late = parts[100:300], parts[600:]
        return tuple(sum(g for g, _ in part) / sum(g + s for g, s in part) for part in (early, late))

    plain = {"generator": {"credit": _AS_REWARD}, "solver": {"credit": _AS_REWARD}}
    early, late = realized(plain)
    assert late > 1.4 * early
    shared = {**plain, "generator": {"credit": _AS_REWARD, "share": 0.1}}
    assert realized(shared) == (pytest.approx(0.1, abs=0.015), pytest.approx(0.1, abs=0.015))


def test_several_shares_each_hold_their_part_beside_the_unshared_roles():
    clock = Clock()
    estimator = _estimator(
        generator={"credit": _AS_REWARD, "share": 0.1},
        judge={"credit": _AS_REWARD, "share": 0.3},
        solver={"credit": _AS_REWARD},
    )
    for _ in range(3):
        group = _credit(estimator, _group(clock, (*_production(5.0, [1.0, 1.0]), _agent("judge", "judge", 0.5))))
    total = sum(_mass(group, role) for role in ("generator", "judge", "solver"))
    assert _mass(group, "generator") / total == pytest.approx(0.1)
    assert _mass(group, "judge") / total == pytest.approx(0.3)


def _generators(clock: Clock) -> CreditGroup:
    """A group of a generator of main and one of the worker, each sampled at its policy's version now."""

    return _group(clock, _production(1.0, []), (_agent("generator", "generator", 1.0, policy="worker"),))


def test_each_policy_s_masses_are_its_own_and_halve_every_half_life_of_its_versions():
    clock = Clock()
    estimator = _estimator(generator={"credit": _AS_REWARD, "share": 0.5}, solver={"credit": _AS_REWARD})
    worker = (_agent("generator", "generator", 3.0, policy="worker"), _agent("solver", "solver", 1.0, policy="worker"))
    _credit(estimator, _group(clock, _production(1.0, [1.0]), worker))
    masses = {policy: held.live() for policy, held in estimator._masses.items()}
    assert masses == {MAIN: {"generator": 2.0, "solver": 2.0}, "worker": {"generator": 6.0, "solver": 2.0}}

    clock.versions["worker"] = int(2 * MASS_HALF_LIFE)
    _credit(estimator, _group(clock, (_agent("solver", "solver", 1.0, policy="worker"),)))
    assert estimator._masses["worker"].live() == pytest.approx({"generator": 1.5, "solver": 2.5})
    assert estimator._masses[MAIN].live() == {"generator": 2.0, "solver": 2.0}  # main's versions did not move
    # A worker's generator that began sampling at version 8 reads the masses of what settled before it: version 0's.
    assert estimator._weights(_generators(clock)) == {
        (MAIN, "generator"): pytest.approx(1.0),
        ("worker", "generator"): pytest.approx(2 / 6),
    }
    clock.versions["worker"] += 1
    assert estimator._weights(_generators(clock))[("worker", "generator")] == pytest.approx(2.5 / 1.5)


def test_role_masses_are_summed_over_a_step_s_groups_as_they_train():
    clock, tracker = Clock(), StatsTracker()
    estimator = _estimator(generator={"credit": _AS_REWARD}, solver={"credit": _AS_REWARD})
    for generator, solvers in ((1.0, [0.5, -0.5]), (-2.0, [])):
        _credit(estimator, _group(clock, _production(generator, solvers))).stats.record(tracker)
    metrics = tracker.export()
    assert metrics["credit/role_groups"] == 2.0
    assert metrics["credit/role_mass/generator"] == pytest.approx(2 + 4)
    assert metrics["credit/role_mass/solver"] == pytest.approx(2.0)


# --- its state -------------------------------------------------------------------------------------------------------


def test_a_resume_from_its_state_credits_as_the_run_would_have():
    by_role = {
        "generator": {"credit": "running", "options": {"strata": ["kind"], **_FLAT}, "share": 0.2},
        "solver": {"credit": "keyed", "options": {"key": ["role"], "leave_one_out": True}},
    }
    setup = CreditSetup(
        name="roles", entry="envgen", settings=WorkflowConfig(), policies=(MAIN,), max_staleness={MAIN: 4}
    )
    clock, rng = Clock(), random.Random(3)
    running, resumed = _estimator(setup, **by_role), _estimator(setup, **by_role)
    for index in range(12):
        group = _group(clock, _production(rng.uniform(-1, 1), [rng.choice([0.0, 1.0]) for _ in range(4)]))
        group.columns = {"kind": rng.choice(["masked", "grounded"])}
        _credit(running, group)
        clock.versions[MAIN] += index % 2
    resumed.load_state(json.loads(json.dumps(running.state())))
    assert resumed.state() == running.state()

    advantages = []
    for estimator in (running, resumed):
        group = _group(clock, _production(0.3, [1.0, 0.0, 0.0, 1.0]))
        group.columns = {"kind": "masked"}
        _credit(estimator, group)
        advantages.append([_advantages(agent) for agent in group.agents()])
    assert advantages[0] == advantages[1] and advantages[0][0] != [[0.0, 0.0, 0.0]]
    assert running.state() == resumed.state()


def test_a_resume_gives_each_role_s_estimator_its_own_state_and_starts_one_of_another_class_afresh(caplog):
    clock = Clock()
    by_role = {"generator": {"credit": "running", "options": _FLAT}, "solver": {"credit": _AS_REWARD}}
    estimator = _estimator(**by_role)
    _credit(estimator, _group(clock, _production(1.0, [1.0])))
    state = json.loads(json.dumps(estimator.state()))
    assert list(state["estimators"]) == ["generator"]  # the solvers' estimator keeps no state
    assert state["estimators"]["generator"]["class"] == "platoon.train.credit.running:Running"
    assert state["masses"] == [[MAIN, [[0, [["generator", 2.0], ["solver", 2.0]]]]]]  # by policy, then version

    changed = _estimator(generator={"credit": f"{__name__}:Stateful"}, solver={"credit": _AS_REWARD})
    with caplog.at_level(logging.WARNING):
        changed.load_state(state)
    assert "by_role.generator starts afresh" in caplog.text and changed._estimators["generator"].loaded is None
    assert changed._masses[MAIN].live() == {"generator": 2.0, "solver": 2.0}


class Stateful(AsReward):
    """An estimator with a state, which records what it was given back."""

    def __init__(self, setup: px.CreditSetup, options: Any = None) -> None:
        super().__init__(setup, options)
        self.loaded: Any = None

    def state(self) -> dict[str, Any]:
        return {"loaded": self.loaded}

    def load_state(self, state: dict[str, Any]) -> None:
        self.loaded = state


# --- its options and checks ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("by_role", "says"),
    [
        ({}, "credit_options.by_role: one or more roles"),
        ({"solver": {"credit": "keyd"}}, "credit_options.by_role.solver.credit: unknown credit estimator 'keyd'"),
        ({"solver": {"credit": "roles"}}, "credit_options.by_role.solver.credit: roles credits the roles of a tree"),
        ({"solver": {"credit": "keyed", "options": {"key": ["agent"]}}}, "credit_options.by_role.solver.options.key"),
        ({"solver": {"credit": "keyed", "options": {"leave": True}}}, "by_role.solver.options.leave: unknown key"),
        ({"solver": {"credit": "group", "options": {"x": 1}}}, "by_role.solver.options: group takes no options"),
        ({"solver": {"credit": "keyed", "shares": 0.1}}, "credit_options.by_role.solver.shares"),
        (
            {"a": {"credit": "keyed", "share": 1.5}, "b": {"credit": "keyed"}},
            "credit_options.by_role.a.share: a number above 0 and below 1, got 1.5",
        ),
        (
            {"a": {"credit": "keyed", "share": 0.6}, "b": {"credit": "keyed", "share": 0.4}, "c": {"credit": "keyed"}},
            "the shares add up to 1",
        ),
        ({"a": {"credit": "keyed", "share": 0.5}}, "every role has a share"),
        (
            {"a": {"credit": "keyed", "kl": {"to": "reference", "coef": 0}}},
            "credit_options.by_role.a.kl.coef: a finite number above 0, got 0",
        ),
        ({"a": {"credit": "keyed", "kl": {"to": "reference"}}}, "by_role.a.kl"),
    ],
)
def test_options_it_cannot_serve_are_refused_under_their_path(by_role, says):
    parsed, problems = schema_options(RolesOptions).check({"by_role": by_role}, "credit_options")
    assert parsed is None and any(says in problem for problem in problems), problems


def test_its_options_as_lock_json_records_them_hold_each_role_s_options_with_their_defaults():
    options = RolesOptions(by_role={"solver": RoleCredit("keyed", {"leave_one_out": True}, kl=RoleKL("ref", 0.1))})
    assert schema_options(RolesOptions).as_json(options) == {
        "by_role": {
            "solver": {
                "credit": "keyed",
                "options": {"key": ["task", "policy", "role"], "leave_one_out": True},
                "share": None,
                "kl": {"to": "ref", "coef": 0.1},
            }
        }
    }


def test_a_kl_term_s_policy_whose_tokenizer_the_run_loaded_otherwise_is_refused_as_it_is_built():
    class Tokenizer:
        def __init__(self, vocabulary: dict[str, int]) -> None:
            self.vocabulary = vocabulary

        def get_vocab(self) -> dict[str, int]:
            return self.vocabulary

    views = {
        MAIN: PolicyView(MAIN, tokenizer=Tokenizer({"a": 0})),
        "ref": PolicyView("ref", tokenizer=Tokenizer({"b": 0})),
    }
    setup = CreditSetup(name="roles", entry="envgen", settings=WorkflowConfig(), policies=(MAIN,), policy_views=views)
    with pytest.raises(ConfigError, match=r"by_role\.solver\.kl\.to: policies\.ref's tokenizer, as the run loaded it"):
        _estimator(setup, solver={"credit": "keyed", "kl": {"to": "ref", "coef": 0.1}})
    same = {**views, "ref": PolicyView("ref", tokenizer=views[MAIN].tokenizer)}
    _estimator(
        CreditSetup(name="roles", entry="envgen", settings=WorkflowConfig(), policies=(MAIN,), policy_views=same),
        solver={"credit": "keyed", "kl": {"to": "ref", "coef": 0.1}},
    )


def test_the_weights_add_up_exactly_whatever_order_the_masses_settled_in():
    """The weights add up the masses in the order of their roles' names, so that a resume, which rebuilds the masses in
    that order, reads them alike to the last bit, as the run would have."""

    roles = {"z": {"credit": _AS_REWARD, "share": 0.25}, **{role: {"credit": _AS_REWARD} for role in "abc"}}
    estimator, resumed, clock = _estimator(**roles), _estimator(**roles), Clock()
    for _ in range(4):  # masses that settle in the order z, c, b, a, decayed as the versions go by
        trees = (_agent("z", "z", 0.1), _agent("c", "c", 0.1), _agent("b", "b", 0.2), _agent("a", "a", 0.3))
        _credit(estimator, _group(clock, trees))
        clock.versions[MAIN] += 1
    resumed.load_state(json.loads(json.dumps(estimator.state())))
    masses = estimator._masses[MAIN].live()
    assert list(masses) == ["z", "c", "b", "a"] and list(resumed._masses[MAIN].live()) == ["a", "b", "c", "z"]
    in_names_order = (masses["a"] + masses["b"]) + masses["c"]
    assert in_names_order != (masses["c"] + masses["b"]) + masses["a"]  # in the order they settled: another sum
    probe = _group(clock, (_agent("z", "z", 0.1),))  # sampled after the last settle: reads the masses it left
    expected = {(MAIN, "z"): 0.25 / 0.75 * in_names_order / masses["z"]}
    assert resumed._weights(probe) == estimator._weights(probe) == expected
