"""A step of a run that trains several policies in lockstep, policy by policy (``platoon.train.step.lanes``): a group's
rows split by the trained policies, a policy's own staleness bound, the metrics a lane records under its prefix, and
each trained policy's lane (its learner, its stale filter, its version, its rows of a microbatch), the leader's
first."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

from fakes.trainer import named_policy

from platoon.train.policy.learner import PrefixedTracker
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.step.datums import policy_rows
from platoon.train.step.lanes import PolicyLane, policy_lanes
from platoon.train.step.microbatch import MicrobatchFill
from platoon.train.step.staleness import StaleFilter


class _Rows:
    def __init__(self, rows: dict[str, Any]) -> None:
        self._rows = rows

    def rows(self) -> dict[str, Any]:
        return self._rows


def test_a_group_s_rows_split_by_the_trained_policies_and_another_policy_s_are_left_out(caplog):
    main_rows, worker_rows, judge_rows = ["m1", "m2"], ["w1"], ["j1"]

    by_policy = policy_rows(_Rows({"worker": worker_rows, MAIN: main_rows, "judge": judge_rows}), [MAIN, "worker"])
    assert by_policy == {MAIN: main_rows, "worker": worker_rows}
    assert "gives rows of judge, which this run does not train (it trains main, worker)" in caplog.text
    plain = ["d"]
    assert policy_rows(plain, [MAIN, "worker"]) == {MAIN: plain, "worker": ()}  # a list of datums: the leader's


def test_a_policy_with_a_bound_of_its_own_judges_its_rows_against_it():
    run = StaleFilter(3, {"slow": 5}, per_datum=True)
    assert run.for_policy(None) is run
    own = run.for_policy(1)
    assert (own.run_bound, dict(own.bounds), own.per_datum) == (1, {}, True)


def test_a_lane_records_its_metrics_under_its_prefix():
    recorded: list[dict[str, float]] = []
    tracker: Any = SimpleNamespace(scalar=lambda **values: recorded.append(values))
    PrefixedTracker(tracker, "policies/worker/").scalar(stale_rollouts=1.0, **{"optim/lr": 2.0})
    assert recorded == [{"policies/worker/stale_rollouts": 1.0, "policies/worker/optim/lr": 2.0}]


def test_each_trained_policy_has_a_lane_the_leader_s_first_each_other_one_s_under_its_own_bound_and_prefix():
    main, trained = named_policy(MAIN), named_policy("worker")
    trained.spec = dataclasses.replace(trained.spec, max_staleness=1)
    llm: Any = trained.info.llm
    llm.version = 5
    tracker: Any = SimpleNamespace(scalar=lambda **values: None)
    run_filter = StaleFilter(3, {}, per_datum=False)

    def objective(spec: PolicySpec) -> Any:
        return f"objective of {spec.id}"

    def lanes(others: list[Policy]) -> tuple[PolicyLane, ...]:
        timeouts: Any = SimpleNamespace()
        return policy_lanes(
            main,
            others,
            objective=objective,
            timeouts=timeouts,
            wait=_no_wait,
            tracker=tracker,
            stale_filter=run_filter,
            leader_version=lambda: 7,
        )

    leader, other = lanes([trained])
    assert (leader.policy_id, leader.prefix, leader.leads, leader.named, leader.version()) == (MAIN, "", True, True, 7)
    assert leader.stale_filter is run_filter and leader.tracker is tracker and leader.policy_suffix == " of policy main"
    assert (other.policy_id, other.prefix, other.leads, other.version()) == ("worker", "policies/worker/", False, 5)
    assert other.stale_filter.run_bound == 1 and other.learner.objective == "objective of worker"
    assert other.learner.tracker is other.tracker and other.policy_suffix == " of policy worker"
    fill = MicrobatchFill()
    worker_rows: Any = ["w1"]
    main_rows: Any = ["m1"]
    other.add_rows(fill, worker_rows)
    leader.add_rows(fill, main_rows)
    assert (fill.datums, fill.others) == (["m1"], {"worker": ["w1"]})
    assert other.rows_in(MicrobatchFill()) == [] and leader.rows_in(fill) == ["m1"]

    [alone] = lanes([])  # a run of one trained policy: nothing it logs or raises names a policy
    assert not alone.named and alone.policy_suffix == ""


async def _no_wait(awaitable: Any, *, name: str, total_timeout: float | None = None) -> Any:
    return await awaitable
