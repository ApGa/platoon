"""The built-in task samplers draw platoon 0.3.0's rows, position for position (RFC 0019, milestone M4).

platoon 0.3.0's order classes stay in the tree, untouched, and are the oracles here: ``WeightedOrder`` with its cycling
pools (``weighted``, and ``auto`` for a mixture), ``ShuffledPasses`` and ``PoolPasses`` (``auto`` for a single
environment, and ``uniform``) and ``FixedOrder`` (a listing marked to keep its order), as the trainer's dataloader draws
them.  The samplers repeat their arithmetic; these tests hold them to it on random configs and heterogeneous rows, and,
for the orders that are pure Python, to the digests of the rows that platoon 0.3.0 itself drew
(``tests/fixtures/orders-0.3.0.json``, written by ``orders_0_3_0.py``), which no change to the oracles can move.  The
orders that shuffle a dataset are compared in process only: their permutation is the installed ``datasets``'s.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import orders_0_3_0
import pytest
import yaml

from platoon.train import components, dataset_order, samplers
from platoon.train.auto import AutoDataset
from platoon.train.components import ENVIRONMENT_COLUMN, EnvironmentConfig, WeightedOrder
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.sampler_api import SamplerContext, SamplerSetup, decision_key
from platoon.train.sampler_runtime import Owed, SamplerSet, TaskPool, TaskSampling
from platoon.train.samplers import SamplerChoice, UniformSampler, WeightedSampler
from platoon.train.task_stats import EntryInfo
from platoon.utils.config import resolve_config

HERE = __name__
GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "orders-0.3.0.json"


def _json(row: Any) -> str:
    return json.dumps(row, sort_keys=True, default=str)


def _draws(sampling: TaskSampling, batch_size: int, positions: range) -> list[dict[str, Any]]:
    """The rows a new set of ``sampling``'s samplers draws at ``positions``, asked from the first of them on."""

    samplers_ = SamplerSet.build(sampling, batch_size=batch_size, label_column=ENVIRONMENT_COLUMN)
    return [samplers_.static(position).row for position in positions]


# --- the seeds ------------------------------------------------------------------------------------------------------


def test_the_samplers_seeds_are_those_of_0_3_0_s_orders():
    assert samplers.SHUFFLE_SEED == dataset_order.SHUFFLE_SEED == 42
    assert samplers.PASS_SEED_STRIDE == dataset_order.PASS_SEED_STRIDE == components._PASS_SEED_STRIDE == 101
    assert samplers.ENTRY_SEED_STRIDE == components._GROUP_SEED_STRIDE == 10_007


# --- weighted, and each entry's uniform, against WeightedOrder ------------------------------------------------------


def _random_mixture(rng: random.Random) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], dict[str, int]]:
    """One to eight entries, a quarter of them without tasks, weighted by their tasks, by numbers that tie, or by
    fractions, each with a seed."""

    labels = [f"e{index}" for index in range(rng.randint(1, 8))]
    groups = {
        label: [
            {"task_id": f"{label}::t{index}", ENVIRONMENT_COLUMN: label}
            for index in range(0 if rng.random() < 0.25 else rng.randint(1, 25))
        ]
        for label in labels
    }
    if not any(groups.values()):
        groups[labels[0]] = [{"task_id": f"{labels[0]}::t0", ENVIRONMENT_COLUMN: labels[0]}]
    mode = rng.random()
    if mode < 0.4:
        weights = {label: float(len(groups[label])) for label in labels}
    elif mode < 0.7:
        weights = {label: rng.choice([1.0, 2.0, 0.5, 1 / 3, 0.1, 3.0]) for label in labels}
    else:
        weights = dict.fromkeys(labels, 1.0)
    seeds = {label: rng.randint(0, 10**6) for label in labels}
    return groups, weights, seeds


def _weighted_sampling(
    groups: dict[str, list[dict[str, Any]]], weights: dict[str, float], seeds: dict[str, int], epoch_size: int
) -> TaskSampling:
    """What AutoDataset.training builds for a mixture's weighted order: a pool of every entry's rows, ``weighted``
    over the entries with tasks, and each one's ``uniform``."""

    labels = list(groups)
    pool = TaskPool.from_rows(
        [row for rows in groups.values() for row in rows], entries=labels, column=ENVIRONMENT_COLUMN
    )
    infos: dict[str, EntryInfo] = {}
    for label in labels:
        if groups[label]:
            infos[label] = EntryInfo(
                group_size=1,
                name=label,
                weight=weights[label],
                seed=seeds[label],
                index=len(infos),
                task_ids=tuple(row["task_id"] for row in groups[label]),
            )
    uniform = SamplerChoice("uniform", UniformSampler)
    return TaskSampling(
        pool=pool,
        run=SamplerChoice("weighted", WeightedSampler),
        entries=dict.fromkeys(infos, uniform),
        infos=infos,
        epoch_size=epoch_size,
    )


@pytest.mark.parametrize("chunk", range(6))
def test_weighted_and_each_entry_s_uniform_draw_the_weighted_order_s_items(chunk):
    """300 random mixtures (50 per chunk), four epochs each: the sampler set draws, position for position, the items
    WeightedOrder's epochs hold.  Then a resume at a random position, from that position alone, goes on with the same
    items, whether its counts are recomputed when it first draws or when it takes up a checkpoint that saved none."""

    rng = random.Random(1000 + chunk)
    for _ in range(50):
        groups, weights, seeds = _random_mixture(rng)
        epoch = rng.randint(1, 80)
        order = WeightedOrder(groups, weights, seeds=seeds, epoch_size=epoch)
        expected = [row["task_id"] for index in range(4) for row in order.epoch(index)]
        sampling = _weighted_sampling(groups, weights, seeds, epoch)
        assert [row["task_id"] for row in _draws(sampling, 1, range(len(expected)))] == expected

        resume = rng.randint(0, len(expected))
        tail = range(resume, len(expected))
        assert [row["task_id"] for row in _draws(sampling, 1, tail)] == expected[resume:]
        restored = SamplerSet.build(sampling, batch_size=1, label_column=ENVIRONMENT_COLUMN)
        restored.restore(None, next_position=resume)
        assert [restored.static(position).row["task_id"] for position in tail] == expected[resume:]


def test_the_weighted_sampler_answers_any_decision_alone_as_in_sequence():
    """A static sampler is a function of its decision index: asked at random decisions, out of order, it answers what
    it answers in sequence."""

    rng = random.Random(7)
    for _ in range(20):
        groups, weights, seeds = _random_mixture(rng)
        sampling = _weighted_sampling(groups, weights, seeds, 10)
        setup = SamplerSetup(
            name="weighted",
            level="run",
            entry=None,
            task_ids=sampling.pool.task_ids,
            entries=sampling.infos,
            batch_size=1,
            label_column=ENVIRONMENT_COLUMN,
            seed=0,
        )
        sequence = WeightedSampler(setup)
        answers = [sequence.next(_context(decision)).entry for decision in range(200)]
        jumping = WeightedSampler(setup)
        for decision in rng.sample(range(200), 60):
            assert jumping.next(_context(decision)).entry == answers[decision]


def _context(decision: int) -> SamplerContext:
    return SamplerContext(
        task_ids=(),
        entry=None,
        decision=decision,
        key=decision_key(0, "test", decision),
        eligible=lambda task_id: True,
        stats=None,
        step=0,
        versions={"main": 0},
    )


# --- through AutoDataset.training, as platoon train lists and orders the tasks --------------------------------------


@pytest.mark.parametrize("name", sorted(orders_0_3_0.CONFIGS))
def test_the_samplers_draw_the_rows_0_3_0_s_dataloader_draws(name, tmp_path):
    """Each config through AutoDataset.training: the sampler set its ``sampling`` builds draws, position for
    position, the rows the dataloader draws from its ``dataset`` and ``order`` (three epochs and a batch), as JSON;
    a listing that repeats task ids is drawn row by row; and a resume at any batch from its position alone goes on
    with the same rows."""

    config = orders_0_3_0.config(name, tmp_path)
    training = AutoDataset.training(config)
    batch_size = config.train.batch_size
    expected = [_json(row) for row in orders_0_3_0.drawn_rows(training, batch_size)]
    assert training.sampling is not None
    drawn = [_json(row) for row in _draws(training.sampling, batch_size, range(len(expected)))]
    assert drawn == expected
    for resume in range(0, len(expected), batch_size):
        tail = [_json(row) for row in _draws(training.sampling, batch_size, range(resume, len(expected)))]
        assert tail == expected[resume:]


@pytest.mark.parametrize("name", sorted(name for name in orders_0_3_0.CONFIGS if orders_0_3_0.pure_python(name)))
def test_the_pure_python_orders_draw_the_rows_platoon_0_3_0_drew(name, tmp_path):
    """The rows of the weighted orders, as platoon 0.3.0 itself drew them (their task ids and digests, which
    ``orders_0_3_0.py`` wrote running on platoon 0.3.0)."""

    golden = json.loads(GOLDEN.read_text())[name]
    config = orders_0_3_0.config(name, tmp_path)
    training = AutoDataset.training(config)
    assert training.sampling is not None
    drawn = _draws(training.sampling, config.train.batch_size, range(len(golden["rows"])))
    assert [[str(row["task_id"]), orders_0_3_0.digest(row)] for row in drawn] == golden["rows"]


def test_the_golden_orders_cover_every_pure_python_config():
    golden = json.loads(GOLDEN.read_text())
    assert sorted(golden) == sorted(name for name in orders_0_3_0.CONFIGS if orders_0_3_0.pure_python(name))
    assert {entry["order"] for entry in golden.values()} == {"WeightedOrder"}


@pytest.mark.parametrize(
    ("run", "entry"),
    [("weighted", None), ("weighted", "pass_rate"), ("auto", "pass_rate"), ("pass_rate", None), ("uniform", None)],
)
@pytest.mark.parametrize("values", [["x"], [None, "x", "y"]], ids=["every-row", "some-rows"])
def test_a_single_environment_s_rows_are_its_own_whatever_their_environment_column_says(tmp_path, run, entry, values):
    """A single environment whose rows name an environment of their own in platoon's environment column (the rows of a
    mixture's run, exported): every row is the environment's, as platoon 0.3.0's orders took them, so its entry's
    sampler is asked for each slot and chooses among all of them, and an adaptive run sampler's pool is the entry's."""

    environment = orders_0_3_0.entry(None, 7, seed=4, loader="env_column", values=values)
    if entry is not None:
        environment["task_sampler"] = entry
    path = tmp_path / "train.yaml"
    train = {"model_name": "m", "renderer_name": "r", "batch_size": 3, "max_training_steps": 4}
    path.write_text(
        yaml.safe_dump(
            {
                "train": {**train, "task_sampler": run, "workflow_config": {"group_size": 2}},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": [environment],
            }
        )
    )
    config = resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config
    training = AutoDataset.training(config)
    label = config.environments[0].label
    pool = training.sampling.pool
    assert pool.entry_indices(label) == tuple(range(len(pool))) == tuple(range(7))
    built = SamplerSet.build(training.sampling, batch_size=3, label_column=ENVIRONMENT_COLUMN)
    if built.run.slots:
        assert list(built.entries) == [label] and built.entries[label].setup.task_ids == pool.task_ids
        if entry == "pass_rate":
            assert all(isinstance(built.static(position), Owed) for position in range(5))
        else:
            assert {built.static(position).reason for position in range(14)} == {"order"}
    assert set(built.fallbacks().values()) == {0}


# --- heterogeneous rows of a dataset ---------------------------------------------------------------------------------


def heterogeneous(config: Any, split: str, *, seed: int, count: int) -> list[dict[str, Any]]:
    """``count`` rows whose columns differ: missing keys, nested mappings with different keys, mixed numbers, lists,
    nulls."""

    rng = random.Random(seed)
    listed = []
    for index in range(count):
        row: dict[str, Any] = {"task_id": f"t{index}"}
        if rng.random() < 0.5:
            row["kind"] = rng.choice(["hard", "easy"])
        if rng.random() < 0.5:
            row["meta"] = {"n": index} if index % 2 else {"m": str(index)}
        if rng.random() < 0.3:
            row["score"] = index if index % 3 else index + 0.5
        if rng.random() < 0.3:
            row["tags"] = [f"x{index}"] * (index % 3)
        listed.append(row)
    return listed


def _training(sampler: str, entries: list[tuple[str | None, int, int]], batch_size: int) -> Any:
    environments = [
        EnvironmentConfig(
            name=name,
            dataset_loader=f"{HERE}:heterogeneous",
            dataset_kwargs={"seed": seed, "count": count},
            rollout="x:y",
        )
        for name, seed, count in entries
    ]
    config = SimpleNamespace(
        environments=environments, train=SimpleNamespace(batch_size=batch_size, task_sampler=sampler)
    )
    return AutoDataset.training(config)


@pytest.mark.parametrize("sampler", ["auto", "uniform"])
def test_heterogeneous_rows_of_a_dataset_are_drawn_as_0_3_0_drew_them(sampler):
    """60 pools of rows whose columns differ, as one environment's listing (``auto``: the dataloader's shuffled
    passes of the listing) and as a mixture of two (``uniform``: PoolPasses over every entry's rows, every row with
    every column): the samplers draw the same rows, as JSON, for three epochs."""

    rng = random.Random(5)
    for trial in range(60):
        count = rng.randint(1, 40)
        batch_size = rng.randint(1, count)
        if sampler == "auto":
            entries = [(None, trial, count)]
        else:
            entries = [("a", trial, count), ("b", 1000 + trial, rng.randint(1, 10))]
        training = _training(sampler, entries, batch_size)
        expected = [_json(row) for row in orders_0_3_0.drawn_rows(training, batch_size)]
        assert [_json(row) for row in _draws(training.sampling, batch_size, range(len(expected)))] == expected


@pytest.mark.parametrize("sampler", ["auto", "uniform"])
def test_two_draws_of_a_task_from_a_dataset_share_no_nested_value(sampler):
    """A dataset's row is converted afresh for every draw, as 0.3.0's passes converted every epoch afresh: a group
    that changes its row's nested values changes no other draw's."""

    entries = [(None, 3, 6)] if sampler == "auto" else [("a", 3, 6), ("b", 4, 2)]
    training = _training(sampler, entries, 2)
    pool = training.sampling.pool
    drawn = _draws(training.sampling, 2, range(4 * len(pool)))
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in drawn:
        by_task.setdefault(row["task_id"], []).append(row)
    nested = [rows for rows in by_task.values() if isinstance(rows[0].get("meta"), dict)]
    assert nested
    for rows in nested:
        assert len(rows) >= 2 and rows[0]["meta"] == rows[1]["meta"] and rows[0]["meta"] is not rows[1]["meta"]


# --- a static sampler keeps no state ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["single-auto", "single-auto-marked", "mix-weighted-numeric", "mix-uniform"])
def test_a_static_built_in_keeps_no_state_and_refuses_one(name, tmp_path):
    config = orders_0_3_0.config(name, tmp_path)
    training = AutoDataset.training(config)
    built = SamplerSet.build(training.sampling, batch_size=config.train.batch_size, label_column=ENVIRONMENT_COLUMN)
    for _ in range(9):
        built.static(built.run.decisions)
    for member in built.members():
        assert member.sampler.state_dict() == {}
        member.sampler.load_state_dict({})
        with pytest.raises(ValueError, match="keeps no state"):
            member.sampler.load_state_dict({"position": 3})
