"""``pass_rate`` decision by decision (RFC 0019, section 5): its rules, each lifted when it would leave nothing; its
exploration (the least recently drawn first) and its exploitation (Thompson sampling on each task's posterior, scored
``1 - p^G - (1 - p)^G``); the cheap read it makes of every task; its state; and its metrics."""

from __future__ import annotations

from typing import Any

import pytest

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.sampler_api import TaskStatsView
from platoon.train.sampler_runtime import Owed, SamplerSet, TaskPool, TaskSampling
from platoon.train.samplers import PassRateOptions, PassRateSampler, SamplerChoice, WeightedSampler
from platoon.train.task_stats import (
    MAIN,
    EntryInfo,
    GroupRecord,
    MemberResult,
    TaskCounts,
    TaskStatsStore,
    informative_at,
)

G = 4


class _Statistics:
    def __init__(self, store: TaskStatsStore) -> None:
        self.store = store
        self.stopped = False


def _set(count: int = 8, *, ids: list[str] | None = None, **options: Any) -> tuple[SamplerSet, TaskStatsStore]:
    """``pass_rate`` as the one entry's sampler under ``weighted``, over ``count`` tasks, with a statistics store."""

    ids = ids or [f"t{index}" for index in range(count)]
    rows = [{"task_id": task, ENVIRONMENT_COLUMN: "e"} for task in ids]
    pool = TaskPool.from_rows(rows, entries=["e"], column=ENVIRONMENT_COLUMN)
    info = EntryInfo(group_size=G, name="e", weight=1.0, seed=11, index=0, task_ids=pool.task_ids)
    sampling = TaskSampling(
        pool=pool,
        run=SamplerChoice("weighted", WeightedSampler),
        entries={"e": SamplerChoice("pass_rate", PassRateSampler, PassRateOptions(**options))},
        infos={"e": info},
        epoch_size=len(ids),
    )
    samplers = SamplerSet.build(sampling, batch_size=4, label_column=ENVIRONMENT_COLUMN)
    store = TaskStatsStore([(task, "e", "e") for task in dict.fromkeys(ids)], {"e": info}, column=ENVIRONMENT_COLUMN)
    samplers.attach(_Statistics(store))
    return samplers, store


def _decide(samplers: SamplerSet, *, version: int = 0, room: Any = None) -> Any:
    position = samplers.run.decisions
    owed = samplers.static(position)
    assert isinstance(owed, Owed)
    return samplers.decide(owed.entry, room=room or (lambda label: True), versions={MAIN: version})


def _settle(store: TaskStatsStore, seq: int, task: str, passes: int, *, version: int = 0, valid: bool = True) -> None:
    members = tuple(
        MemberResult(
            reward=float(index < passes) if valid else None,
            scored=float(index < passes) if valid else None,
            valid=valid,
            outcome="scored" if valid else "error",
            disposition="completed",
            reward_key=None,
            steps=None,
            output_tokens=None,
            seconds=None,
            versions={MAIN: (version, version)},
        )
        for index in range(G)
    )
    verdict = "trained" if valid and 0 < passes < G else "uniform" if valid else "no_results"
    store.draw(9, seq, task, None)
    store.settle(
        GroupRecord(
            start=9,
            seq=seq,
            task_id=task,
            entry="e",
            labels={ENVIRONMENT_COLUMN: "e"},
            harness=None,
            group_size=G,
            sampler="test",
            reason="order",
            score=None,
            position=None,
            drawn_at={MAIN: version},
            members=members,
            verdict=verdict,
        )
    )
    store.fate(9, seq, "consumed")


def test_with_explore_1_it_draws_the_least_recently_drawn_first_a_round_robin():
    samplers, _store = _set(explore=1.0, max_draws_per_epoch=None)
    first = [_decide(samplers).row["task_id"] for _ in range(8)]
    second = [_decide(samplers).row["task_id"] for _ in range(8)]
    assert sorted(first) == [f"t{index}" for index in range(8)]  # every task once, the never drawn first
    assert second == first  # then the least recently drawn: the same turn again
    assert {decided.reason for decided in [_decide(samplers)]} == {"explore"}


def test_with_explore_0_it_never_explores():
    samplers, store = _set(explore=0.0)
    for seq, task in enumerate(f"t{index}" for index in range(8)):
        _settle(store, seq, task, passes=seq % (G + 1))
    reasons = {_decide(samplers).reason for _ in range(24)}
    assert reasons == {"exploit"}


def test_it_prefers_tasks_whose_next_group_mixes_passes_and_failures():
    """Two tasks a group passes half of, six that every group passes or fails: exploitation draws the first two far
    more often, and each draw's score is its task's E[informative_G]."""

    samplers, store = _set(explore=0.0, max_draws_per_epoch=None)
    for repeat in range(6):
        for index in range(8):
            _settle(store, 8 * repeat + index, f"t{index}", passes=2 if index < 2 else (G if index % 2 else 0))
    decisions = [_decide(samplers) for _ in range(40)]
    drawn = [decided.row["task_id"] for decided in decisions]
    assert sum(task in ("t0", "t1") for task in drawn) >= 32
    for decided in decisions[:5]:
        alpha, beta = store.posterior(decided.row["task_id"], now=0)
        assert decided.score == informative_at(alpha, beta, G)


def test_an_unseen_task_with_a_draw_outstanding_waits_for_its_first_result():
    samplers, store = _set(explore=1.0, max_draws_per_epoch=None)
    for index in range(7):
        store.draw(9, index, f"t{index}", None)  # t0..t6 drawn, none settled
    assert _decide(samplers).row["task_id"] == "t7"
    store.draw(9, 7, "t7", None)
    # Every task is unseen with a draw outstanding: those with the fewest stay (all, one each).
    assert _decide(samplers).row["task_id"] in {f"t{index}" for index in range(8)}


def test_max_outstanding_per_task_leaves_out_a_busy_task_unless_every_task_is():
    samplers, store = _set(explore=0.0, max_outstanding_per_task=1, max_draws_per_epoch=None)
    for index in range(8):
        _settle(store, index, f"t{index}", passes=2)
        store.draw(9, 100 + index, f"t{index}", None)  # each with one draw outstanding now
    store.fate(9, 103, "consumed")  # but t3
    assert {_decide(samplers).row["task_id"] for _ in range(6)} == {"t3"}


def test_max_draws_per_epoch_caps_a_task_s_draws_in_its_window_and_is_lifted_when_every_task_is_at_it():
    samplers, store = _set(count=4, explore=0.0, max_draws_per_epoch=1)
    for index in range(4):
        _settle(store, index, f"t{index}", passes=2 if index == 0 else 0)
    window = [_decide(samplers).row["task_id"] for _ in range(4)]
    assert sorted(window) == ["t0", "t1", "t2", "t3"]  # one each in a window of four decisions
    member = samplers.entries["e"]
    assert member.sampler.metrics(None)["cap_lifted"] == 0.0
    member.sampler._window_draws = {task: 1 for task in ("t0", "t1", "t2", "t3")}
    member.decisions = 5  # still the second window
    member.sampler._window = 1
    _decide(samplers)
    assert member.sampler.metrics(None)["cap_lifted"] == 1.0


def test_a_task_without_evidence_three_groups_running_is_drawn_by_exploration_only():
    samplers, store = _set(count=2, explore=0.0, max_draws_per_epoch=None)
    _settle(store, 0, "t0", passes=2)
    for seq in range(1, 4):
        _settle(store, seq, "t1", passes=0, valid=False)  # three groups without a valid member
    assert store.counts("t1").no_evidence_streak == 3
    assert {_decide(samplers).row["task_id"] for _ in range(8)} == {"t0"}


def test_its_decisions_read_each_task_s_counts_never_its_whole_record(monkeypatch):
    samplers, store = _set(count=16)
    for index in range(16):
        _settle(store, index, f"t{index}", passes=index % (G + 1))

    def refuse(self, task_id):
        raise AssertionError("pass_rate read task(); a decision reads counts() for every task")

    monkeypatch.setattr(TaskStatsView, "task", refuse)
    assert all(_decide(samplers) is not None for _ in range(40))


def test_a_decision_keeps_no_task_s_counts_alive_through_it(monkeypatch):
    """A decision reads every eligible task's counts but keeps only the numbers its rules need: an object kept for each
    task through a decision over a large pool would be traced and promoted by the garbage collector, whose full
    collections then hold up the event loop that decides."""

    samplers, store = _set(count=300, explore=0.5)
    for index in range(300):
        _settle(store, index, f"t{index}", passes=index % (G + 1))
    live = {"now": 0, "most": 0}

    class Counted(TaskCounts):
        __slots__ = ()

        def __new__(cls, counts: TaskCounts) -> Counted:
            live["now"] += 1
            live["most"] = max(live["most"], live["now"])
            return tuple.__new__(cls, counts)

        def __del__(self) -> None:
            live["now"] -= 1

    read = TaskStatsStore.counts
    monkeypatch.setattr(TaskStatsStore, "counts", lambda self, task_id: Counted(read(self, task_id)))
    reasons = {_decide(samplers).reason for _ in range(12)}
    assert reasons == {"explore", "exploit"} and live["most"] <= 2


def test_a_listing_that_repeats_an_id_draws_its_first_row():
    samplers, _store = _set(ids=["a", "b", "a", "c"], explore=1.0, max_draws_per_epoch=None)
    drawn = [_decide(samplers) for _ in range(3)]
    assert sorted(decided.row["task_id"] for decided in drawn) == ["a", "b", "c"]
    assert next(decided.index for decided in drawn if decided.row["task_id"] == "a") == 0


def test_its_state_round_trips_drops_tasks_that_left_and_refuses_what_it_cannot_read():
    samplers, store = _set(count=6)
    for index in range(6):
        _settle(store, index, f"t{index}", passes=1)
    for _ in range(9):
        _decide(samplers)
    sampler = samplers.entries["e"].sampler
    state = sampler.state_dict()
    assert set(state) == {"last_drawn", "window", "window_draws", "previous_window_draws"}
    fresh, _ = _set(count=6)
    fresh.entries["e"].sampler.load_state_dict(state)
    assert fresh.entries["e"].sampler.state_dict() == state
    smaller, _ = _set(count=3)
    smaller.entries["e"].sampler.load_state_dict(state)
    assert set(smaller.entries["e"].sampler.state_dict()["last_drawn"]) <= {"t0", "t1", "t2"}
    for broken in ({"last_drawn": [1]}, {**state, "window": -1}, {**state, "window_draws": {"t0": "2"}}, {"x": 1}):
        with pytest.raises(ValueError):
            fresh.entries["e"].sampler.load_state_dict(broken)
    fresh.entries["e"].sampler.load_state_dict({})  # starts afresh


def test_its_metrics_say_how_concentrated_its_draws_are_and_what_its_decisions_rested_on():
    samplers, store = _set(count=10, explore=0.0, max_draws_per_epoch=None)
    for index in range(10):
        _settle(store, index, f"t{index}", passes=2 if index == 0 else 0, version=index)
    for _ in range(10):
        _decide(samplers, version=12)
    metrics = samplers.entries["e"].sampler.metrics(samplers.view(12))
    assert set(metrics) == {"top10_share", "eligible_share", "cap_lifted", "prior_share", "evidence_age"}
    assert 0.1 <= metrics["top10_share"] <= 1.0 and metrics["eligible_share"] == 1.0
    assert 0 < metrics["prior_share"] < 1 and metrics["evidence_age"] >= 0
    assert samplers.entries["e"].sampler.metrics(None) == {"top10_share": metrics["top10_share"]}  # reset


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"explore": 1.5}, "explore: must be between 0 and 1, got 1.5"),
        ({"explore": True}, "explore: must be between 0 and 1, got True"),
        ({"max_draws_per_epoch": 0}, "max_draws_per_epoch: must be an integer from 1, or null for no limit, got 0"),
        ({"max_outstanding_per_task": 0}, "max_outstanding_per_task: must be an integer from 1"),
        ({"candidates": 0}, "candidates: must be an integer from 1, got 0"),
        ({"seed": 1.5}, "seed: must be an integer, got 1.5"),
    ],
)
def test_its_options_check_their_ranges(options, message):
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        PassRateOptions(**options)


def test_candidates_bound_how_many_tasks_an_exploit_decision_scores(monkeypatch):
    samplers, store = _set(count=12, explore=0.0, candidates=3, max_draws_per_epoch=None)
    for index in range(12):
        _settle(store, index, f"t{index}", passes=2)
    scored: list[str] = []
    original = TaskStatsView.posterior

    def counting(self, task_id):
        scored.append(task_id)
        return original(self, task_id)

    monkeypatch.setattr(TaskStatsView, "posterior", counting)
    _decide(samplers)
    assert len(scored) == 3 + 1  # three scored, and the chosen one's score
