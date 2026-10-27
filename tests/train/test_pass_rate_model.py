"""``pass_rate`` and ``uniform`` in an event-driven model of a training run (RFC 0019, section 5), the model its
defaults were chosen with, here with the real samplers deciding: groups are decided when a concurrency slot and the
horizon allow, run for lognormal times, enter the statistics when they settle, and are trained ``batch_size`` a step
in the order they complete; a step that trained an informative group moves the policy on a version, and a group more
than ``k`` versions old at its step is stale.  Each group is ``G`` Bernoulli rollouts at its task's fixed pass rate:
selection only, no learning.  The samplers are reached through the real sampler set's contexts, read a real statistics
store fed real group records, and a task in flight is held back only by ``pass_rate``'s own rules.

In CI: two pools of 40 tasks at 20 groups at once, 12 seeds each, against the model's own means (within 0.03 for
``uniform``, 0.04 for ``pass_rate``) and its concentration.  With ``PLATOON_PASS_RATE_MODEL_TABLE=1``, every cell of
the model's table of defaults, 60 seeds each (10 on 475 tasks), printed as a table.
"""

from __future__ import annotations

import heapq
import os
import random
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from platoon.train.components import ENVIRONMENT_COLUMN
from platoon.train.sampler_runtime import Owed, SamplerSet, TaskPool, TaskSampling
from platoon.train.samplers import PassRateSampler, SamplerChoice, UniformSampler, WeightedSampler
from platoon.train.task_stats import MAIN, EntryInfo, GroupRecord, MemberResult, TaskStatsStore

G = 8
# The calibration pool: 40 training tasks of an agentic environment, passing 1, 2, 3 or 4 of 5 attempts (3, 6, 6 and 25
# of them), made stronger as a policy that solves each in one of three tries: 1 - (1 - k / 5) ** 3.
CALIB40 = tuple([1 / 5] * 3 + [2 / 5] * 6 + [3 / 5] * 6 + [4 / 5] * 25)
STRONG40 = tuple(1 - (1 - rate) ** 3 for rate in CALIB40)
# 40 pass rates drawn from the Beta distribution whose all-pass and all-fail shares at a group of 8 are those a live
# run measured (0.751 and 0.073): Beta(0.31426, 0.05601), the model's draw.
LIVE_BETA = (0.3142620752398056, 0.05600992955911855)
LIVE40 = (
    0.6085852639845434, 0.999999999993805, 0.19533023819104062, 1.8905841432923775e-06, 0.9999973396242522,
    0.9999999945577064, 0.9999590485204407, 0.9997715073244681, 0.9999999999991628, 1.0, 0.7764763042677238,
    0.2922307704264594, 0.9999696951774208, 0.9999999998619817, 0.9999997728136659, 0.9999999999988708, 1.0,
    0.999999999999994, 0.12795111489467784, 0.6937354454865282, 0.9803865054487878, 0.9662296407970163,
    0.5750205987826766, 0.9973606702008143, 0.9996395618462994, 0.9999999999999273, 0.9999999415759913,
    0.9999132366838884, 0.9983824165438988, 0.9999999999999925, 0.9999999998868184, 1.0, 0.9999999999986612,
    0.9999999983204663, 0.9990234706112194, 0.030845881743848346, 0.8352618339491569, 0.9999999999999932,
    0.019692615208506525, 0.9999999999996471,
)  # fmt: skip


@dataclass(frozen=True)
class Setup:
    """A cell of the model's table: the training steps, the batch, the groups at once, ``k`` and the seeds."""

    steps: int = 60
    batch: int = 20
    slots: int = 20
    k: int = 3
    seeds: int = 12
    roll_mean: float = 10.0
    roll_sigma: float = 0.5
    train_time: float = 2.0


@dataclass(frozen=True)
class Outcome:
    informative_share: float
    top10_share: float
    stale_share: float
    coverage: float


class _Statistics:
    """The run's statistics, as a sampler set reads them."""

    def __init__(self, store: TaskStatsStore) -> None:
        self.store = store
        self.stopped = False


def run(pool: Sequence[float], sampler: str, setup: Setup, seed: int) -> Outcome:
    """One run of the model with the real ``sampler`` (``uniform`` or ``pass_rate``, as an entry's sampler under
    ``weighted``) choosing every group's task."""

    rng = random.Random(seed)
    count = len(pool)
    rows = [{"task_id": f"t{index}", ENVIRONMENT_COLUMN: "pool"} for index in range(count)]
    tasks = TaskPool.from_rows(rows, entries=["pool"], column=ENVIRONMENT_COLUMN)
    info = EntryInfo(group_size=G, name="pool", weight=1.0, seed=seed, index=0, task_ids=tasks.task_ids)
    cls = PassRateSampler if sampler == "pass_rate" else UniformSampler
    sampling = TaskSampling(
        pool=tasks,
        run=SamplerChoice("weighted", WeightedSampler),
        entries={"pool": SamplerChoice(sampler, cls, None)},
        infos={"pool": info},
        epoch_size=count,
    )
    samplers = SamplerSet.build(sampling, batch_size=setup.batch, label_column=ENVIRONMENT_COLUMN)
    store = TaskStatsStore(
        [(task, "pool", "pool") for task in tasks.task_ids], {"pool": info}, column=ENVIRONMENT_COLUMN
    )
    samplers.attach(_Statistics(store))
    rate = {task: pool[index] for index, task in enumerate(tasks.task_ids)}

    version = pipeline = admitted = running = position = step = consumed = informative = stale = 0
    step_signal = False
    now = train_end = 0.0
    collecting = True
    events: list[tuple[float, int, str, int, int]] = []
    results: deque[tuple[int, str, int, int]] = deque()
    draws = dict.fromkeys(tasks.task_ids, 0)

    def decide() -> str:
        versions = {MAIN: version}
        decided = samplers.static(position, step=step, versions=versions)
        if isinstance(decided, Owed):
            decided = samplers.decide(decided.entry, room=lambda label: True, step=step, versions=versions)
        assert decided is not None and decided.reason != "fallback"
        return str(decided.row["task_id"])

    while step < setup.steps:
        while running < setup.slots and pipeline + admitted < (setup.k + 1) * setup.batch:
            task = decide()
            passes = sum(rng.random() < rate[task] for _ in range(G))
            duration = setup.roll_mean * rng.lognormvariate(-(setup.roll_sigma**2) / 2, setup.roll_sigma)
            store.draw(1, position, task, position)
            heapq.heappush(events, (now + duration, position, task, passes, version))
            draws[task] += 1
            position += 1
            running += 1
            pipeline += 1
        if collecting:
            while results and admitted < setup.batch:
                seq, task, passes, drawn_at = results.popleft()
                pipeline -= 1
                admitted += 1
                consumed += 1
                late = version - drawn_at > setup.k
                store.fate(1, seq, "stale" if late else "consumed")
                if late:
                    stale += 1
                else:
                    mixed = 0 < passes < G
                    informative += mixed
                    step_signal = step_signal or mixed
            if admitted == setup.batch:
                collecting, train_end = False, now + setup.train_time
                continue
        next_settle = events[0][0] if events else float("inf")
        if not collecting and train_end <= next_settle:
            now = train_end
            version += step_signal
            step += 1
            step_signal = False
            admitted = 0
            collecting = True
            continue
        assert events, "nothing in flight and the step is not full"
        now, seq, task, passes, drawn_at = heapq.heappop(events)
        running -= 1
        store.settle(_record(seq, task, passes, drawn_at))
        results.append((seq, task, passes, drawn_at))

    trained = consumed - stale
    ranked = sorted(draws.values(), reverse=True)
    return Outcome(
        informative_share=informative / trained if trained else 0.0,
        top10_share=sum(ranked[: max(1, count // 10)]) / sum(ranked),
        stale_share=stale / consumed if consumed else 0.0,
        coverage=sum(1 for value in ranked if value) / count,
    )


def _record(seq: int, task: str, passes: int, version: int) -> GroupRecord:
    members = tuple(
        MemberResult(
            reward=float(index < passes),
            scored=float(index < passes),
            valid=True,
            outcome="scored",
            disposition="completed",
            reward_key=None,
            steps=None,
            output_tokens=None,
            seconds=None,
            versions={MAIN: (version, version)},
        )
        for index in range(G)
    )
    return GroupRecord(
        start=1,
        seq=seq,
        task_id=task,
        entry="pool",
        labels={ENVIRONMENT_COLUMN: "pool"},
        harness=None,
        group_size=G,
        sampler="model",
        reason="order",
        score=None,
        position=seq,
        drawn_at={MAIN: version},
        members=members,
        verdict="trained" if 0 < passes < G else "uniform",
    )


def mean_outcome(pool: Sequence[float], sampler: str, setup: Setup) -> dict[str, float]:
    outcomes = [run(pool, sampler, setup, seed) for seed in range(setup.seeds)]
    return {
        name: statistics.fmean(getattr(outcome, name) for outcome in outcomes)
        for name in ("informative_share", "top10_share", "stale_share", "coverage")
    }


# The model's means (60 seeds each), and the tolerance of a mean of 12 seeds.
CI_CELLS = {
    "strong40": (STRONG40, {"uniform": 0.306, "pass_rate": 0.649}),
    "live40": (LIVE40, {"uniform": 0.194, "pass_rate": 0.525}),
}
TOLERANCE = {"uniform": 0.03, "pass_rate": 0.04}


@pytest.mark.parametrize("pool_name", sorted(CI_CELLS))
@pytest.mark.parametrize("sampler", ["uniform", "pass_rate"])
def test_the_real_samplers_reproduce_the_model_s_informative_shares(pool_name, sampler):
    pool, expected = CI_CELLS[pool_name]
    found = mean_outcome(pool, sampler, Setup())
    assert abs(found["informative_share"] - expected[sampler]) <= TOLERANCE[sampler], found
    if sampler == "pass_rate":
        assert 0.25 <= found["top10_share"] <= 0.35, found
        assert found["coverage"] == 1.0
    else:
        assert found["top10_share"] < 0.15


# --- the model's whole table of defaults, on request ------------------------------------------------------------------


def _live475() -> tuple[float, ...]:
    """475 pass rates drawn from the live run's Beta distribution, as the model drew them (numpy's generator, seed
    9)."""

    numpy = pytest.importorskip("numpy")
    return tuple(float(value) for value in numpy.random.default_rng(9).beta(*LIVE_BETA, size=475))


TABLE: dict[str, tuple[Any, Setup, dict[str, float]]] = {
    # name: (pool, setup, the model's means: uniform, pass_rate with the defaults)
    "calib40, B 20, 20 at once": (CALIB40, Setup(seeds=60), {"uniform": 0.8745, "pass_rate": 0.9370}),
    "strong40, B 20, 20 at once": (STRONG40, Setup(seeds=60), {"uniform": 0.3059, "pass_rate": 0.6486}),
    "live40, B 20, 20 at once": (LIVE40, Setup(seeds=60), {"uniform": 0.1935, "pass_rate": 0.5246}),
    "strong40, B 20, 40 at once": (STRONG40, Setup(slots=40, seeds=60), {"uniform": 0.3050, "pass_rate": 0.6435}),
    "live40, B 20, 40 at once": (LIVE40, Setup(slots=40, seeds=60), {"uniform": 0.1930, "pass_rate": 0.5170}),
    "live475, B 32, 32 at once": (
        _live475,
        Setup(steps=100, batch=32, slots=32, seeds=10),
        {"uniform": 0.1867, "pass_rate": 0.4102},
    ),
}


@pytest.mark.skipif(not os.environ.get("PLATOON_PASS_RATE_MODEL_TABLE"), reason="the whole table runs on request")
def test_the_whole_table_of_defaults(capsys):
    lines = [
        "| cell | uniform (model) | uniform (real) | pass_rate (model) | pass_rate (real) | top10_share (real) | "
        "coverage (real) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, (pool, setup, model) in TABLE.items():
        rates = pool() if callable(pool) else pool
        uniform = mean_outcome(rates, "uniform", setup)
        adaptive = mean_outcome(rates, "pass_rate", setup)
        lines.append(
            f"| {name} ({setup.seeds} seeds) | {model['uniform']:.3f} | {uniform['informative_share']:.3f} | "
            f"{model['pass_rate']:.3f} | {adaptive['informative_share']:.3f} | {adaptive['top10_share']:.3f} | "
            f"{adaptive['coverage']:.3f} |"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))
