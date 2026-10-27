"""The per-task statistics store (``platoon.train.task_stats``): passes judged on the scored reward, what each verdict
counts as, decay, the posterior's closed forms, snapshots that round-trip byte for byte, and the step metrics."""

from __future__ import annotations

import dataclasses
import json
import math
from fractions import Fraction
from itertools import product

import pytest

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.task_stats import (
    DecayedCounts,
    EntryInfo,
    EventLog,
    GroupRecord,
    MemberResult,
    TaskStatsStore,
    band,
    informative_at,
    next_start,
    pass_at,
    read_events,
    read_snapshot,
    readable_snapshot,
    snapshot_bytes,
    snapshot_name,
    write_snapshot,
)

POOL = [("quick::a", "quick", "quick"), ("quick::b", "quick", "quick"), ("slow::c", "slow", "slow")]
ENTRIES = {"quick": EntryInfo(group_size=4), "slow": EntryInfo(group_size=2, success_threshold=0.5)}


def member(scored: float | None, *, reward: float | None = None, valid: bool = True, version: int = 5) -> MemberResult:
    return MemberResult(
        reward=scored if reward is None else reward,
        scored=scored,
        valid=valid,
        outcome="scored" if valid else "error",
        disposition="completed",
        reward_key="reward" if valid else None,
        steps=3,
        output_tokens=40,
        seconds=2.0,
        versions={"main": (version, version + 1)} if valid else {},
    )


def record(task_id: str, members: list[MemberResult], verdict: str, *, seq: int, start: int = 1) -> GroupRecord:
    entry = task_id.partition("::")[0]
    return GroupRecord(
        start=start,
        seq=seq,
        task_id=task_id,
        entry=entry,
        labels={ENVIRONMENT_COLUMN: entry},
        harness=None,
        group_size=len(members),
        sampler="auto",
        reason="order",
        score=None,
        position=seq,
        drawn_at={"main": 5},
        members=tuple(members),
        verdict=verdict,
    )


def store() -> TaskStatsStore:
    return TaskStatsStore(POOL, ENTRIES, column=ENVIRONMENT_COLUMN)


def test_passes_are_judged_on_the_scored_reward_so_a_shaping_reward_processor_changes_none():
    """A reward processor that scales every reward by 0.98 trains 0.98 for a correct rollout; its environment still
    scored 1.0, so every pass count is the one without the processor."""

    plain, scaled = store(), store()
    rewards = [1.0, 1.0, 1.0, 0.0]
    plain.settle(record("quick::a", [member(score) for score in rewards], "trained", seq=0))
    scaled.settle(record("quick::a", [member(score, reward=0.98 * score) for score in rewards], "trained", seq=0))

    for stats in (plain.task("quick::a"), scaled.task("quick::a")):
        assert (stats.attempts, stats.passes, stats.informative_groups) == (4, 3, 1)
    assert scaled.task("quick::a").reward_sum == pytest.approx(3 * 0.98)
    # A partial-credit entry counts by its own threshold.
    partial = store()
    partial.settle(record("slow::c", [member(0.6), member(0.4)], "trained", seq=0))
    assert partial.task("slow::c").passes == 1


def test_each_settled_group_counts_once_by_its_verdict_and_success():
    stats = store()
    stats.settle(record("quick::a", [member(1.0)] * 4, "uniform", seq=0))  # all pass
    stats.settle(record("quick::a", [member(0.0)] * 4, "uniform", seq=1))  # all fail
    # Uniform, yet mixed by success: no datum kept the signal, or the trained reward and the success disagree.
    stats.settle(record("quick::a", [member(1.0, reward=0.0), member(0.0)] * 2, "uniform", seq=2))
    assert stats.task("quick::a").uniform_streak == 3
    stats.settle(record("quick::a", [member(1.0), member(0.0)] * 2, "trained", seq=3))
    stats.settle(record("quick::a", [member(None, valid=False)] * 4, "no_results", seq=4))
    stats.settle(record("quick::a", [], "failed", seq=5))

    task = stats.task("quick::a")
    assert (task.groups, task.attempts, task.passes) == (6, 16, 4 + 2 + 2)
    assert (task.informative_groups, task.all_pass_groups, task.all_fail_groups) == (1, 1, 1)
    assert task.uniform_mixed_groups == 1
    assert (task.uniform_streak, task.no_evidence_streak) == (0, 2)
    assert [summary.verdict for summary in task.history] == ["uniform"] * 3 + ["trained", "no_results", "failed"]
    totals = stats.entry("quick")
    assert (totals.groups, totals.informative_groups, totals.uniform_mixed_groups) == (6, 1, 1)
    assert stats.label("quick").passes == 8


def test_evidence_decays_by_policy_version():
    counts = DecayedCounts(32.0)
    counts.add(1.0, 0.0, version=0)
    assert counts.at(32) == pytest.approx((0.5, 0.0))
    counts.add(0.0, 1.0, version=32)  # newer: the old evidence decays to its version, then it adds
    assert counts.at(32) == pytest.approx((0.5, 1.0))
    counts.add(1.0, 0.0, version=0)  # older, arriving late: added at its discount
    assert counts.at(32) == pytest.approx((1.0, 1.0))
    assert DecayedCounts(None, 2.0, 1.0, 0).at(1000) == (2.0, 1.0)  # no half-life: no decay


@pytest.mark.parametrize(("alpha", "beta", "k"), [(1.0, 1.0, 1), (1.5, 2.25, 4), (0.3, 7.0, 8), (12.0, 0.5, 3)])
def test_pass_at_k_matches_the_beta_binomial_expectation(alpha, beta, k):
    """E[pass@k] = 1 - B(alpha, beta + k) / B(alpha, beta), and E[informative_k] = E[pass@k] - E[p^k]."""

    def beta_function(a: float, b: float) -> float:
        return math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    expected = 1 - beta_function(alpha, beta + k) / beta_function(alpha, beta)
    assert pass_at(alpha, beta, k) == pytest.approx(expected)
    all_pass = beta_function(alpha + k, beta) / beta_function(alpha, beta)
    assert informative_at(alpha, beta, k) == pytest.approx(expected - all_pass)
    assert pass_at(alpha, beta, 1) == pytest.approx(alpha / (alpha + beta))


def _beta_binomial(a: Fraction, b: Fraction, k: int) -> list[Fraction]:
    """``P(X = x)`` for ``x`` from 0 to ``k``, ``X ~ BetaBinomial(k, a, b)``, exactly, for rational ``a`` and ``b``:
    ``C(k, x) prod_{i<x} (a + i) prod_{j<k-x} (b + j) / prod_{i<k} (a + b + i)``."""

    denominator = math.prod((a + b + i for i in range(k)), start=Fraction(1))
    return [
        math.comb(k, x)
        * math.prod((a + i for i in range(x)), start=Fraction(1))
        * math.prod((b + j for j in range(k - x)), start=Fraction(1))
        / denominator
        for x in range(k + 1)
    ]


# Two grids of (alpha, beta, k): fractional, decayed counts and group sizes up to 32; the second reaches the counts of
# a near-solved entry's prior against long evidence.
_GRIDS = [
    (
        [
            Fraction(value)
            for value in ("0.02", "0.05", "0.1", "0.28", "0.5", "1", "1.72", "2", "3.3", "7.5", "40.5", "189")
        ],
        range(1, 17),
    ),
    (
        [Fraction(value) for value in ("0.02", "0.1", "0.5", "1", "1.72", "2", "7.5", "40.5", "189", "1/3", "12000")],
        (1, 2, 4, 8, 16, 32),
    ),
]


def test_closed_forms_match_exact_beta_binomial_sums():
    """``pass_at`` and ``informative_at``, the closed forms the statistics and ``pass_rate`` use, against the
    Beta-binomial summed exactly in rationals: ``1 - P(X = 0)`` and ``1 - P(X = 0) - P(X = k)``.  Every distribution
    sums to exactly 1.  The tolerance is absolute: where ``alpha`` is tiny and ``beta`` large, ``1 - prod`` cancels, so
    the relative error of a probability near 0 can reach 1e-7 while the absolute error stays near 1e-15."""

    cases = 0
    for values, sizes in _GRIDS:
        for a, b in product(values, values):
            for k in sizes:
                probabilities = _beta_binomial(a, b, k)
                assert sum(probabilities) == 1
                exact_pass = 1 - probabilities[0]
                exact_informative = 1 - probabilities[0] - probabilities[k] if k > 1 else Fraction(0)
                assert abs(pass_at(float(a), float(b), k) - float(exact_pass)) < 1e-12, (a, b, k)
                assert abs(informative_at(float(a), float(b), k) - float(exact_informative)) < 1e-12, (a, b, k)
                cases += 1
    assert cases == 12 * 12 * 16 + 11 * 11 * 6


def test_the_designs_worked_numbers():
    """Worked numbers of the posterior of RFC 0019, section 3.3 (a prior of strength 2 at the entry's rate):
    ``E[informative_8]`` for an unseen task and after 0, 1, 4 and 8 passes of 8, under an entry at a pass rate of 0.5,
    0.05 and 0.86."""

    worked = {
        0.5: [0.778, 0.471, 0.735, 0.959, 0.471],
        0.05: [0.168, 0.059, 0.505, 0.944, 0.715],
        0.86: [0.407, 0.676, 0.845, 0.950, 0.159],
    }
    evidence = [(0, 0), (0, 8), (1, 7), (4, 4), (8, 0)]
    for rate, numbers in worked.items():
        prior = (2 * rate, 2 * (1 - rate))
        got = [informative_at(prior[0] + passes, prior[1] + fails, 8) for passes, fails in evidence]
        assert [round(value, 3) for value in got] == numbers, rate


def test_the_posterior_borrows_its_prior_from_the_entry():
    stats = store()
    stats.settle(record("quick::a", [member(1.0)] * 3 + [member(0.0)], "trained", seq=0))

    # quick::b has no group: its prior alone, centered on quick's rate (1 + 3) / (2 + 4) with weight 2.
    alpha, beta = stats.posterior("quick::b", now=5)
    assert (alpha, beta) == pytest.approx((2 * 4 / 6, 2 * 2 / 6))
    alpha, beta = stats.posterior("quick::a", now=5)
    assert (alpha, beta) == pytest.approx((2 * 4 / 6 + 3, 2 * 2 / 6 + 1))
    # 32 versions on, the evidence weighs half: the prior moves to (1 + 1.5) / (2 + 2), and the task's own halves.
    alpha, beta = stats.posterior("quick::a", now=37)
    assert (alpha, beta) == pytest.approx((2 * 2.5 / 4 + 1.5, 2 * 1.5 / 4 + 0.5))


def test_a_task_of_no_entry_takes_the_run_s_group_size():
    stats = TaskStatsStore([("t", None, None)], {}, column=ENVIRONMENT_COLUMN, group_size=8)
    stats.settle(record("t", [member(1.0), member(0.0)], "trained", seq=0))

    alpha, beta = stats.posterior("t", now=5)
    assert stats.step_metrics(now=5, prefixes={})["tasks/pass_at_G"] == pytest.approx(pass_at(alpha, beta, 8))


def test_a_snapshot_round_trips_byte_for_byte():
    stats = store()
    for seq, task_id in enumerate(["quick::a", "quick::b", "slow::c", "quick::a"]):
        stats.draw(1, seq, task_id, position=seq)
    stats.settle(record("quick::a", [member(1.0), member(0.0)] * 2, "trained", seq=0))
    stats.settle(record("slow::c", [member(0.7), member(None, valid=False)], "below_min", seq=2))
    stats.fate(1, 0, "consumed")
    stats.fate(1, 2, "stale")
    data = stats.snapshot(start=1, next_seq=4)
    saved = snapshot_bytes(data)

    loaded = store()
    loaded.load(json.loads(saved))

    assert snapshot_bytes(loaded.snapshot(start=1, next_seq=4)) == saved
    assert snapshot_bytes(TaskStatsStore.from_snapshot(json.loads(saved)).snapshot(start=1, next_seq=4)) == saved
    assert loaded.outstanding == [(1, 1, "quick::b", 1), (1, 3, "quick::a", 3)]
    assert loaded.task("quick::a").fates == {"consumed": 1}
    assert loaded.task("slow::c").fates == {"stale": 1}


def test_a_resume_counts_the_outstanding_draws_as_lost_and_follows_a_changed_pool():
    stats = store()
    stats.draw(1, 0, "quick::a")
    stats.settle(record("quick::a", [member(1.0)] * 4, "uniform", seq=0))
    data = stats.snapshot(start=1, next_seq=1)

    resumed = TaskStatsStore([("quick::a", "quick", "quick"), ("quick::new", "quick", "quick")], ENTRIES, column="c")
    resumed.load(data)

    assert resumed.lose_outstanding() == [(1, 0)]
    assert resumed.task("quick::a").fates == {"lost": 1} and resumed.task("quick::a").outstanding == 0
    assert resumed.task("quick::a").groups == 1  # its evidence stays: it settled before the checkpoint
    assert resumed.task("quick::new").groups == 0  # a task new to the pool starts empty
    assert resumed.task_ids() == ["quick::a", "quick::new"]  # one that left it is dropped
    with pytest.raises(ValueError, match="cannot read the task statistics snapshot"):
        store().load({**data, "format": 99})


def test_a_resume_draws_an_outstanding_draw_again_under_a_new_number_neither_drawn_twice_nor_given_a_fate():
    stats = store()
    stats.draw(1, 0, "quick::a", position=4)
    stats.draw(1, 1, "slow::c", position=5)

    assert stats.reissue((1, 0), (2, 0)) and stats.reissue([1, 1], (2, 1))
    assert not stats.reissue((1, 9), (2, 2))  # not outstanding: nothing changes

    assert stats.outstanding == [(2, 0, "quick::a", 4), (2, 1, "slow::c", 5)]
    task = stats.task("quick::a")
    assert (task.drawn, task.outstanding, task.fates) == (1, 1, {})
    stats.fate(2, 0, "consumed")
    assert stats.task("quick::a").fates == {"consumed": 1} and stats.task("quick::a").outstanding == 0


def test_a_group_record_and_its_members_read_back_from_json():
    group = record("slow::c", [member(0.7), member(None, valid=False)], "below_min", seq=3, start=2)

    assert GroupRecord.from_json(json.loads(json.dumps(group.to_json()))) == group
    flagged = dataclasses.replace(member(1.0), flag=True)
    assert MemberResult.from_json(flagged.to_json()) == flagged


def test_the_step_metrics_average_the_seen_tasks_posteriors_by_label():
    stats = store()
    stats.settle(record("quick::a", [member(1.0)] * 4, "uniform", seq=0, start=1))
    for seq in range(1, 4):
        stats.settle(record("slow::c", [member(None, valid=False)] * 2, "no_results", seq=seq))

    metrics = stats.step_metrics(now=6, prefixes={"quick": "environment/quick", "slow": "environment/slow"})

    alpha, beta = stats.posterior("quick::a", now=6)
    assert metrics["tasks/environment/quick/pass_at_1"] == pytest.approx(alpha / (alpha + beta))
    assert metrics["tasks/environment/quick/pass_at_G"] == pytest.approx(pass_at(alpha, beta, 4))
    assert metrics["tasks/environment/quick/coverage"] == 0.5
    assert metrics[f"tasks/environment/quick/bands/{band(alpha / (alpha + beta))}"] == 1.0
    assert metrics["tasks/environment/slow/no_evidence"] == 1.0  # three groups in a row without a valid member
    assert metrics["tasks/coverage"] == pytest.approx(2 / 3)
    assert "tasks/pass_at_1" in metrics and "tasks/pass_at_G" in metrics
    assert [band(rate) for rate in (0.05, 0.2, 0.5, 0.9, 0.95)] == ["zero", "low", "mid", "high", "solved"]


def test_a_workflow_that_says_nothing_of_its_members_gives_no_evidence():
    """A custom workflow's plain list of datums records no members and no verdict (``unknown``): its groups count, but
    they make no task seen and move no streak, so no task reads as without evidence and no pass@k is logged from the
    prior alone.  A workflow that raised (``failed``) does say something: a group without a valid member."""

    stats = store()
    prefixes = {"quick": "environment/quick", "slow": "environment/slow"}
    for seq in range(4):
        stats.settle(record("quick::a", [], "unknown", seq=seq))

    task = stats.task("quick::a")
    assert (task.groups, task.reported_groups, task.no_evidence_streak, task.uniform_streak) == (4, 0, 0, 0)
    metrics = stats.step_metrics(now=6, prefixes=prefixes)
    assert metrics["tasks/coverage"] == 0.0 and metrics["tasks/no_evidence"] == 0.0
    assert not [key for key in metrics if key.endswith(("pass_at_1", "pass_at_G"))]
    for seq in range(4, 7):
        stats.settle(record("quick::b", [], "failed", seq=seq))
    metrics = stats.step_metrics(now=6, prefixes=prefixes)
    assert metrics["tasks/environment/quick/no_evidence"] == 1.0 and metrics["tasks/coverage"] == 0.0


def test_the_chance_of_an_informative_group_is_never_below_zero():
    """For a group of one it is 0 by definition; the closed form, ``E[pass@1] - E[p]``, rounds a hair either way."""

    import random

    rng = random.Random(0)
    for _ in range(5000):
        alpha, beta = rng.uniform(0.01, 50.0), rng.uniform(0.01, 50.0)
        assert 0.0 <= informative_at(alpha, beta, 1) < 1e-12
        assert informative_at(alpha, beta, 4) >= 0.0


def test_a_rollout_s_own_success_flag_passes_at_1_whatever_the_threshold():
    """As platoon eval reads it: the flag of a rollout whose trace records no outcome passes at 1, the scored reward of
    a traced one at the entry's threshold."""

    def flagged(scored: float, *, flag: bool) -> MemberResult:
        return MemberResult(
            reward=scored,
            scored=scored,
            valid=True,
            outcome="none" if flag else "scored",
            disposition="completed",
            reward_key=None,
            steps=None,
            output_tokens=None,
            seconds=None,
            flag=flag,
        )

    assert not flagged(0.7, flag=True).passes(0.5) and flagged(0.7, flag=False).passes(0.5)
    assert flagged(1.0, flag=True).passes(2.0) and not flagged(1.0, flag=False).passes(2.0)


def test_the_newest_snapshot_that_reads_back_is_found_past_the_newer_ones_that_do_not(tmp_path):
    data = store().snapshot(start=1, next_seq=0)
    kept = write_snapshot(tmp_path, snapshot_name(1, "000002"), data)
    gone = write_snapshot(tmp_path, snapshot_name(1, "recovery-000003"), data)
    (tmp_path / gone["file"]).unlink()
    rows = [
        {"name": "000002", "tasks": kept},
        {"name": "recovery-000003", "tasks": gone},
        {"name": "recovery-000004"},  # its snapshot could not be written
    ]

    found = readable_snapshot(tmp_path, rows)

    assert found is not None and found.row["name"] == "000002" and found.data == json.loads(snapshot_bytes(data))
    assert found.passed_over[0] == "checkpoint recovery-000004 names no snapshot of the per-task statistics"
    assert found.passed_over[1].startswith("checkpoint recovery-000003: ") and "cannot be read" in found.passed_over[1]
    assert readable_snapshot(tmp_path, rows[1:]) is None


def test_snapshots_are_written_durably_and_read_back_against_their_sha256(tmp_path):
    data = store().snapshot(start=2, next_seq=0)

    record_ = write_snapshot(tmp_path, snapshot_name(2, "recovery-000120"), data)

    assert record_["file"] == "tasks/state-s2-recovery-000120.json" and record_["start"] == 2
    assert read_snapshot(tmp_path, record_) == json.loads(snapshot_bytes(data))
    path = tmp_path / record_["file"]
    path.write_text(path.read_text().replace('"start":2', '"start":3'))
    with pytest.raises(ValueError, match="does not have the SHA-256 its checkpoint recorded"):
        read_snapshot(tmp_path, record_)
    path.unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        read_snapshot(tmp_path, record_)
    assert not list((tmp_path / "tasks").glob(".*partial"))


def test_a_new_start_numbers_itself_after_the_starts_the_rows_name():
    assert next_start([]) == 1
    rows = [{"name": "recovery-000000"}, {"name": "x", "tasks": {"start": 2}}, {"name": "y", "tasks": {"start": 1}}]
    assert next_start(rows) == 3
    # A start whose statistics stopped names its start in the state of the dispatcher's draws alone.
    assert next_start([*rows, {"name": "z", "dispatch": {"start": 4}}]) == 5


def test_the_event_log_appends_a_line_per_event(tmp_path):
    log = EventLog(tmp_path)
    for event, seq in product(("draw", "settle"), range(2)):
        log.write(event, start=1, seq=seq)
    log.close()

    assert [(event["event"], event["seq"]) for event in read_events(tmp_path)] == [
        ("draw", 0),
        ("draw", 1),
        ("settle", 0),
        ("settle", 1),
    ]
    EventLog(None).write("draw", start=1, seq=0)  # a run without a directory logs nothing


def test_a_line_torn_by_a_crash_keeps_the_next_start_s_first_event_whole(tmp_path):
    """A start that died mid-write leaves the log's last line without its newline: the next start begins a line of its
    own, so the torn line alone is lost."""

    log = EventLog(tmp_path)
    log.write("draw", start=1, seq=0, task_id="t")
    log.close()
    with open(tmp_path / "tasks" / "events.jsonl", "a") as handle:
        handle.write('{"event":"settle","start":1,"se')  # torn
    log = EventLog(tmp_path)
    log.write("start", start=2, batch=1, checkpoint="recovery-000001", state_lost=None)
    log.write("draw", start=2, seq=0, task_id="t")
    log.close()

    assert [(event["event"], event["start"]) for event in read_events(tmp_path)] == [
        ("draw", 1),
        ("start", 2),
        ("draw", 2),
    ]


def test_an_entry_s_success_threshold_is_a_finite_number_that_a_named_environment_may_ship():
    from platoon.envs import named
    from platoon.train.components import EnvironmentConfig

    assert EnvironmentConfig(name="a", rollout="x:y").success_threshold == 1.0
    assert EnvironmentConfig(name="a", rollout="x:y", success_threshold=1).success_threshold == 1.0
    for value in (True, float("inf"), float("nan"), "high"):
        with pytest.raises(ValueError, match="success_threshold must be a finite number"):
            EnvironmentConfig(name="a", rollout="x:y", success_threshold=value)
    assert named.EnvironmentEntry("acme-partial", rollout="x:y", success_threshold=0.5).keys["success_threshold"] == 0.5


def test_a_task_s_counts_are_read_without_building_its_record_and_agree_with_it():
    """``counts`` is the cheap read a task sampler makes of every task: the same numbers as ``task``, and the version
    of the task's evidence (None before any)."""

    stats = store()
    assert stats.counts("quick::a") == (0, 0, 0, 0, 0, None)
    stats.draw(1, 0, "quick::a", 0)
    stats.draw(1, 1, "quick::a", 1)
    stats.settle(record("quick::a", [member(1.0), member(0.0), member(None, valid=False)], "trained", seq=0))
    stats.settle(record("quick::a", [member(None, valid=False)] * 4, "no_results", seq=1))
    counts, task = stats.counts("quick::a"), stats.task("quick::a")
    assert (counts.groups, counts.outstanding, counts.drawn) == (task.groups, task.outstanding, task.drawn) == (2, 2, 2)
    assert (counts.uniform_streak, counts.no_evidence_streak) == (task.uniform_streak, task.no_evidence_streak)
    assert counts.evidence_version == 5
    # The record's means are copies of the store's: changing one changes nothing the store holds.
    task.means["reward"].add(100.0, 6)
    assert stats.task("quick::a").means["reward"].at(6) == pytest.approx(0.5)
