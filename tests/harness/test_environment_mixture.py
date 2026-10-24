from __future__ import annotations

import random
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest
from harness_fakes import FakeService

from platoon.task import Task
from platoon.train.auto import (
    AutoDataset,
    AutoEnvironment,
    AutoRewardProcessor,
    AutoRollout,
    AutoTaskLoader,
    AutoWorkflow,
    TaskRouter,
    mixture_eval_rollout,
    mixture_reward_processor,
    mixture_rollout,
)
from platoon.train.components import (
    ENVIRONMENT_COLUMN,
    HARNESS_ROLLOUT,
    EnvironmentConfig,
    EnvironmentRoute,
    WeightedOrder,
    choose_harness,
    interleave_by_weight,
    normalize_environment_configs,
)
from platoon.train.tinker.dataset_order import PRESERVE_DATASET_ORDER_COLUMN, PoolPasses
from platoon.train.tinker.workers.protocol import validated_ref
from platoon.train.tinker.workers.run import DEFAULT_REWARD_PROCESSOR
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection

HERE = __name__


def load_ids(config, split, *, prefix, count=4):
    return [f"{prefix}{index}" for index in range(count)] if split == "train" else [f"{prefix}-eval"]


def load_rows(config, split):
    return [{"task_id": "row-0", "difficulty": "hard"}, {"task_id": "row-1", "difficulty": "easy"}]


def load_task(task_id):
    return Task(id=task_id, goal=f"solve {task_id}", misc={"shared": True})


async def rollout_a(task, config):
    return {"env": "a", "task": task.id, "config": config}


async def rollout_b(task, config):
    return {"env": "b", "task": task.id}


async def eval_rollout_b(task, config):
    return {"env": "b-eval", "task": task.id}


def double_reward(trajectory):
    return 2 * trajectory["reward"], {"reward/double": 2 * trajectory["reward"]}


def environment(name: str, **overrides) -> EnvironmentConfig:
    values = {
        "name": name,
        "dataset_loader": f"{HERE}.load_ids",
        "dataset_kwargs": {"prefix": name},
        "eval_dataset_kwargs": {"prefix": name},
        "task_loader": f"{HERE}.load_task",
        "rollout": f"{HERE}.rollout_{name}",
    }
    return EnvironmentConfig(**{**values, **overrides})


def config(*environments: EnvironmentConfig, batch_size: int | None = 4, task_sampler: str = "auto") -> SimpleNamespace:
    return SimpleNamespace(
        environments=list(environments), train=SimpleNamespace(batch_size=batch_size, task_sampler=task_sampler)
    )


def weighted_order(mixed: SimpleNamespace) -> WeightedOrder:
    """The weighted order AutoDataset.training builds for a mixture (or a single environment under task_sampler:
    weighted)."""

    order = AutoDataset.training(mixed).order
    assert order is not None
    return order


def mixture() -> SimpleNamespace:
    return config(
        environment("a", weight=3.0, dataset_kwargs={"prefix": "a", "count": 2}),
        environment(
            "b",
            weight=1.0,
            dataset_kwargs={"prefix": "b", "count": 5},
            eval_rollout=f"{HERE}.eval_rollout_b",
            reward_processor=f"{HERE}.double_reward",
            harness={"platoon-bash": 1.0, "mini-swe-agent": 1.0},
            service="harness_fakes:FakeService",
        ),
    )


def test_single_environment_resolves_exactly_the_named_components():
    single = config(environment("a", reward_processor=f"{HERE}.double_reward"))

    assert AutoEnvironment.from_config(single) is single.environments[0]
    assert AutoDataset.from_config(single, "train")["task_id"] == ["a0", "a1", "a2", "a3"]
    assert AutoTaskLoader.from_config(single) is load_task
    assert AutoRollout.from_config(single, "train") is rollout_a
    assert AutoRollout.from_config(single, "eval") is rollout_a
    assert AutoRewardProcessor.from_config(single) is double_reward
    assert AutoRewardProcessor.from_config(config(environment("a"))) is DEFAULT_REWARD_PROCESSOR


def test_mixture_training_order_interleaves_by_weight_with_labeled_ids():
    dataset = AutoDataset.from_config(mixture(), "train")
    rows = dataset.to_list()

    assert len(rows) == 8  # 7 tasks, filled to whole batches of 4
    assert all(row[PRESERVE_DATASET_ORDER_COLUMN] for row in rows)
    labels = [row[ENVIRONMENT_COLUMN] for row in rows]
    assert labels == ["a", "b", "a", "a", "a", "b", "a", "a"]  # 3:1 in every batch of four
    assert all(row["task_id"].startswith(f"{row[ENVIRONMENT_COLUMN]}::") for row in rows)
    a_ids = [row["task_id"] for row in rows if row[ENVIRONMENT_COLUMN] == "a"]
    assert Counter(a_ids) == {"a::a0": 3, "a::a1": 3}  # the small, heavy environment cycles
    assert rows == AutoDataset.from_config(mixture(), "train").to_list()  # deterministic


def balanced_sampler_order(
    environment_ids: list[str], weights: dict[str, float], *, batch_size: int, seed: int, epochs: int = 1
) -> list[int]:
    """A balanced environment sampler's training order over ``epochs`` epochs, as indices into its records, written
    out independently of platoon: weighted fair slots over the environments (ties rotate), each environment's records
    shuffled per pass with ``seed + index * 10_007 + pass * 101``, in epochs of the records filled to whole batches,
    the last batch from the next pass; each epoch goes on with the slots' credits and the passes where the one before
    it stopped."""

    order = list(dict.fromkeys(environment_ids))
    indices = {name: [i for i, value in enumerate(environment_ids) if value == name] for name in order}
    total = sum(weights[environment] for environment in order)
    probabilities = [weights[environment] / total for environment in order]
    size = -(-len(environment_ids) // batch_size) * batch_size * epochs
    credits, cursor, slots = [0.0] * len(order), 0, []
    for _ in range(size):
        credits = [credit + probability for credit, probability in zip(credits, probabilities)]
        candidates = {i for i, credit in enumerate(credits) if abs(credit - max(credits)) <= 1e-12}
        chosen = next(i for offset in range(len(order)) if (i := (cursor + offset) % len(order)) in candidates)
        credits[chosen] -= 1.0
        cursor = (chosen + 1) % len(order)
        slots.append(chosen)

    def pool(environment_index: int, cycle: int) -> list[int]:
        shuffled = list(indices[order[environment_index]])
        random.Random(seed + environment_index * 10_007 + cycle * 101).shuffle(shuffled)
        return shuffled

    pools = [pool(i, 0) for i in range(len(order))]
    positions, cycles, result = [0] * len(order), [0] * len(order), []
    for i in slots:
        if positions[i] == len(pools[i]):
            cycles[i] += 1
            pools[i], positions[i] = pool(i, cycles[i]), 0
        result.append(pools[i][positions[i]])
        positions[i] += 1
    return result


def test_mixture_training_order_is_the_balanced_samplers():
    mixed = config(
        environment("a", weight=3.0, seed=343, dataset_kwargs={"prefix": "a", "count": 5}),
        environment("b", weight=1.0, seed=343, dataset_kwargs={"prefix": "b", "count": 7}),
        batch_size=8,
    )
    records = [f"a::a{index}" for index in range(5)] + [f"b::b{index}" for index in range(7)]

    rows = AutoDataset.from_config(mixed, "train").to_list()

    expected = balanced_sampler_order([record[0] for record in records], {"a": 3.0, "b": 1.0}, batch_size=8, seed=343)
    assert [row["task_id"] for row in rows] == [records[index] for index in expected]


def test_each_epoch_takes_up_the_weighted_order_where_the_one_before_stopped():
    """The dataloader draws epoch after epoch of the order AutoDataset.training builds: epoch 0 is the dataset that
    AutoDataset.from_config gives, and each later epoch goes on with the slots and each environment's passes, so the
    light environment's tasks that epoch 0 leaves out come next, instead of epoch 0 again."""

    mixed = config(
        environment("a", weight=3.0, seed=343, dataset_kwargs={"prefix": "a", "count": 5}),
        environment("b", weight=1.0, seed=343, dataset_kwargs={"prefix": "b", "count": 7}),
        batch_size=8,
    )
    records = [f"a::a{index}" for index in range(5)] + [f"b::b{index}" for index in range(7)]

    training = AutoDataset.training(mixed)

    order = training.order
    assert order is not None and order.epoch_size == 16
    expected = balanced_sampler_order(
        [record[0] for record in records], {"a": 3.0, "b": 1.0}, batch_size=8, seed=343, epochs=4
    )
    epochs = [[row["task_id"] for row in order.epoch(index)] for index in range(4)]
    assert [task_id for epoch in epochs for task_id in epoch] == [records[index] for index in expected]
    assert training.dataset.to_list() == [{**row, PRESERVE_DATASET_ORDER_COLUMN: True} for row in order.epoch(0)]
    assert AutoDataset.from_config(mixed, "train").to_list() == training.dataset.to_list()
    assert epochs[1] != epochs[0]
    # b gets 4 of each epoch's 16 slots: epoch 0 draws 4 of its 7 tasks, epoch 1 the other 3 first.
    first, second = [[task_id for task_id in epoch if task_id.startswith("b::")] for epoch in epochs[:2]]
    assert len(set(first)) == 4 and set(second[:3]) == {f"b::b{index}" for index in range(7)} - set(first)
    # Every task once, those epoch 0 leaves out included, for the checks that read the run's tasks.
    assert sorted(row["task_id"] for row in training.tasks) == sorted(records)


def test_a_subsampled_environment_draws_every_task_once_before_any_twice_across_epochs():
    """Equal weights over 95 tasks and 5: the large environment gets 52 of each epoch's 104 slots, fewer than its
    tasks, yet over 10 epochs each of its tasks is drawn 5 or 6 times; and within each environment no task is drawn
    twice more than another (one pass after another)."""

    mixed = config(
        environment("a", weight=1.0, seed=11, dataset_kwargs={"prefix": "a", "count": 95}),
        environment("b", weight=1.0, seed=11, dataset_kwargs={"prefix": "b", "count": 5}),
        batch_size=8,
    )
    order = weighted_order(mixed)

    draws = Counter(row["task_id"] for epoch in range(10) for row in order.epoch(epoch))

    a = [draws[f"a::a{index}"] for index in range(95)]
    b = [draws[f"b::b{index}"] for index in range(5)]
    assert sum(a) == sum(b) == 520
    assert (min(a), max(a)) == (5, 6) and (min(b), max(b)) == (104, 104)
    # Epoch 0 alone leaves 43 of a's tasks out; the next epoch draws them first.
    first = {row["task_id"] for row in order.epoch(0) if row[ENVIRONMENT_COLUMN] == "a"}
    assert len(first) == 52
    second = [row["task_id"] for row in order.epoch(1) if row[ENVIRONMENT_COLUMN] == "a"]
    assert set(second[:43]) == {f"a::a{index}" for index in range(95)} - first


def test_the_weighted_order_is_the_same_however_its_epochs_are_read():
    """The order follows from the listing, the weights, the seeds and the batch size alone: two builds give the same
    epochs, and reading an earlier epoch again draws it as before (a resumed run reads its epochs from the one it
    resumes in)."""

    def build():
        return weighted_order(
            config(
                environment("a", weight=2.0, seed=5, dataset_kwargs={"prefix": "a", "count": 9}),
                environment("b", weight=1.0, seed=6, dataset_kwargs={"prefix": "b", "count": 4}),
                batch_size=4,
            )
        )

    first, second = build(), build()
    in_sequence = [first.epoch(index) for index in range(5)]

    assert [second.epoch(index) for index in (3, 1, 4, 0, 2)] == [in_sequence[index] for index in (3, 1, 4, 0, 2)]
    assert first.epoch(2) == in_sequence[2] and first.epoch(0) == in_sequence[0]
    other_seed = weighted_order(
        config(
            environment("a", weight=2.0, seed=7, dataset_kwargs={"prefix": "a", "count": 9}),
            environment("b", weight=1.0, seed=6, dataset_kwargs={"prefix": "b", "count": 4}),
            batch_size=4,
        )
    )
    assert [other_seed.epoch(index) for index in range(5)] != in_sequence


def test_weight_tasks_draws_every_task_of_the_mixture_equally_often():
    """weight: tasks gives 95 tasks and 5 shares of 95 and 5: over 25 epochs of 104 slots each of the 100 tasks is
    drawn 26 times."""

    mixed = config(
        environment("a", weight="tasks", seed=11, dataset_kwargs={"prefix": "a", "count": 95}),
        environment("b", weight="tasks", seed=11, dataset_kwargs={"prefix": "b", "count": 5}),
        batch_size=8,
    )
    order = weighted_order(mixed)

    epochs = [Counter(row["task_id"] for row in order.epoch(index)) for index in range(25)]

    draws = sum(epochs, Counter())
    assert len(draws) == 100 and set(draws.values()) == {26}
    # It is the default: a mixture whose entries set no weight draws the same order.
    unweighted = config(
        environment("a", seed=11, dataset_kwargs={"prefix": "a", "count": 95}),
        environment("b", seed=11, dataset_kwargs={"prefix": "b", "count": 5}),
        batch_size=8,
    )
    assert EnvironmentConfig().weight == "tasks"
    assert weighted_order(unweighted).epoch(3) == order.epoch(3)
    # Equal weights would have drawn each of b's tasks 19 times as often as each of a's.
    equal = weighted_order(
        config(
            environment("a", weight=1.0, seed=11, dataset_kwargs={"prefix": "a", "count": 95}),
            environment("b", weight=1.0, seed=11, dataset_kwargs={"prefix": "b", "count": 5}),
            batch_size=8,
        )
    )
    equal_draws = Counter(row["task_id"] for index in range(25) for row in equal.epoch(index))
    assert equal_draws["b::b0"] / equal_draws["a::a0"] == pytest.approx(19, rel=0.1)


def test_weight_tasks_counts_the_tasks_an_entry_trains_on():
    """The share counts the tasks an entry lists for training, once exclude_tasks have left: 10 listed less 3 left
    out against 7 is an even split."""

    mixed = config(
        environment("a", weight="tasks", dataset_kwargs={"prefix": "a", "count": 10}, exclude_tasks=["a0", "a1", "a2"]),
        environment("b", weight="tasks", dataset_kwargs={"prefix": "b", "count": 7}),
        batch_size=2,
    )

    training = AutoDataset.training(mixed)

    order = training.order
    assert order is not None and order.epoch_size == 14
    first = Counter(row[ENVIRONMENT_COLUMN] for row in order.epoch(0))
    assert first == {"a": 7, "b": 7}
    assert {row["task_id"] for row in order.epoch(0)} == {row["task_id"] for row in training.tasks}
    assert "a::a0" not in {row["task_id"] for row in training.tasks}


def test_a_mixtures_weights_are_all_numbers_or_all_tasks():
    with pytest.raises(
        ValueError, match=r"Mixture entries \['b'\] weigh by their tasks \(weight: tasks, the weight of"
    ):
        normalize_environment_configs([{"name": "a", "weight": 2}, {"name": "b"}])  # b sets none: tasks
    with pytest.raises(ValueError, match=r"give every entry a number \(1 each for equal shares\), or no entry one"):
        normalize_environment_configs([{"name": "a", "weight": 2}, {"name": "b", "weight": "tasks"}])
    by_tasks = normalize_environment_configs([{"name": "a", "weight": "tasks"}, {"name": "b", "weight": "tasks"}])
    assert [entry.weight for entry in by_tasks] == ["tasks", "tasks"]
    assert normalize_environment_configs([{"weight": "tasks"}])[0].weight == "tasks"  # one entry: any weight
    bads: tuple[Any, ...] = ("task", "Tasks", 0, -1.0, float("nan"), float("inf"), True, None)
    for bad in bads:
        with pytest.raises(ValueError, match="weight must be a finite positive number or tasks"):
            EnvironmentConfig(weight=bad)


def test_weight_tasks_loads_from_a_config_file_and_an_override(tmp_path):
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import ConfigError, resolve_config

    path = tmp_path / "train.yaml"
    path.write_text(
        "train: {model_name: m, renderer_name: r}\n"
        "eval: {every: 0}\n"
        "log_path: runs\n"
        "environments:\n"
        "  - {name: a, weight: tasks, rollout: x:y}\n"
        "  - {name: b, weight: 2, rollout: x:y}\n"
    )

    loaded = resolve_config(path, ["environments.1.weight=tasks"], PlatoonTinkerRLTrainerConfig).config
    assert [entry.weight for entry in loaded.environments] == ["tasks", "tasks"]
    with pytest.raises(ConfigError, match="weigh by their tasks"):
        resolve_config(path, [], PlatoonTinkerRLTrainerConfig)
    with pytest.raises(ConfigError, match=r"'many' is not a float \| 'tasks'"):
        resolve_config(path, ["environments.1.weight=many"], PlatoonTinkerRLTrainerConfig)
    path.write_text(path.read_text().replace("weight: 2", "weight: task"))
    with pytest.raises(ConfigError, match=r"environments.1.weight: 'task' is not a float \| 'tasks'"):
        resolve_config(path, [], PlatoonTinkerRLTrainerConfig)


@pytest.mark.parametrize("count", [5, 8, 11])
def test_a_single_environment_trains_in_the_weighted_order_when_asked(count):
    """``train.task_sampler: weighted``: a mixture of one, in the order of any ``environments`` list: shuffled with
    the environment's seed, the last batch filled from the next pass, ids unlabeled."""

    single = environment("a", seed=343, dataset_kwargs={"prefix": "a", "count": count})

    rows = AutoDataset.from_config(config(single, task_sampler="weighted"), "train").to_list()

    expected = balanced_sampler_order(["a"] * count, {"a": 1.0}, batch_size=4, seed=343, epochs=3)
    epoch_size = len(rows)
    assert [row["task_id"] for row in rows] == [f"a{index}" for index in expected[:epoch_size]]
    assert epoch_size % 4 == 0 and {row["task_id"] for row in rows} == {f"a{index}" for index in range(count)}
    # The next epochs go on with the passes, so every task is drawn as often as every other, give or take one.
    order = weighted_order(config(single, task_sampler="weighted"))
    later = [row["task_id"] for epoch in range(3) for row in order.epoch(epoch)]
    assert later == [f"a{index}" for index in expected]
    draws = Counter(later)
    assert max(draws.values()) - min(draws.values()) <= 1 and len(draws) == count
    assert all(row[PRESERVE_DATASET_ORDER_COLUMN] for row in rows)
    assert all(ENVIRONMENT_COLUMN not in row for row in rows)
    # The router loads the unlabeled ids, as for any single environment.
    router = AutoTaskLoader.from_config(config(single, task_sampler="weighted"))
    assert router(rows[0]["task_id"]).id == rows[0]["task_id"]
    # Evaluation, and training without the option, keep the listing as it is (the dataloader shuffles training).
    listed = [f"a{index}" for index in range(count)]
    assert AutoDataset.from_config(config(single), "train")["task_id"] == listed
    assert AutoDataset.from_config(config(single, task_sampler="weighted"), "eval")["task_id"] == ["a-eval"]


def test_uniform_draws_every_task_of_every_entry_in_seeded_passes_without_entry_shares():
    """``train.task_sampler: uniform``: every training task of every entry once per pass, the passes
    ``Dataset.shuffle(42 + 101 p)`` of the listing, in epochs of as many tasks as there are, filled to whole batches
    from the next pass; each entry's share is its tasks'.  The entries' weights and seeds play no part."""

    from datasets import Dataset

    mixed = config(
        environment("a", dataset_kwargs={"prefix": "a", "count": 3}, seed=5),
        environment("b", dataset_kwargs={"prefix": "b", "count": 6}, seed=9),
        task_sampler="uniform",
    )

    training = AutoDataset.training(mixed)

    listing = [f"a::a{index}" for index in range(3)] + [f"b::b{index}" for index in range(6)]
    assert [row["task_id"] for row in training.tasks] == listing
    order, plan = training.order, training.plan
    assert isinstance(order, PoolPasses) and plan is not None
    assert order.epoch_size == 12  # 9 tasks, filled to whole batches of 4
    passes = [Dataset.from_dict({"id": listing}).shuffle(seed=42 + 101 * number)["id"] for number in range(4)]
    stream = [task for drawn in passes for task in drawn]
    assert [row["task_id"] for epoch in range(3) for row in order.epoch(epoch)] == stream[:36]
    assert [row["task_id"] for row in training.dataset.to_list()] == stream[:12]
    assert all(row[PRESERVE_DATASET_ORDER_COLUMN] for row in training.dataset.to_list())
    assert {entry.label: entry.share for entry in plan.entries} == {"a": 3 / 9, "b": 6 / 9}
    assert not plan.warnings()
    # The entries' seeds do not change it.
    reseeded = config(
        environment("a", dataset_kwargs={"prefix": "a", "count": 3}, seed=1),
        environment("b", dataset_kwargs={"prefix": "b", "count": 6}, seed=2),
        task_sampler="uniform",
    )
    reseeded_order = AutoDataset.training(reseeded).order
    assert reseeded_order is not None and [row["task_id"] for row in reseeded_order.epoch(1)] == stream[12:24]


@pytest.mark.parametrize("count", [5, 8, 10])
def test_uniform_draws_a_single_environment_in_the_batches_auto_draws(count):
    """For one environment, ``uniform`` draws the batches the dataloader's shuffled passes draw under ``auto``; only its
    epochs differ, which hold every task, the last batch filled from the next pass."""

    from platoon.train.tinker.rl import PlatoonTinkerDataloader

    single = environment("a", dataset_kwargs={"prefix": "a", "count": count})
    auto = AutoDataset.training(config(single))
    uniform = AutoDataset.training(config(single, task_sampler="uniform"))

    assert isinstance(uniform.order, PoolPasses)
    batches = PlatoonTinkerDataloader(auto.dataset, 4)
    drawn = PlatoonTinkerDataloader(uniform.dataset, 4, order=uniform.order)
    assert [[row["task_id"] for row in drawn.get_batch(number)] for number in range(12)] == [
        [row["task_id"] for row in batches.get_batch(number)] for number in range(12)
    ]
    assert len(drawn) == -(-count // 4) and uniform.order.items == auto.tasks.to_list()


def test_uniform_takes_no_entry_weights(tmp_path):
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, task_sampler_problems
    from platoon.utils.config import ConfigError, resolve_config

    path = tmp_path / "train.yaml"
    path.write_text(
        "train: {model_name: m, renderer_name: r, task_sampler: uniform}\nlog_path: runs\n"
        "environments:\n  - {name: a, rollout: 'x:y', weight: 3}\n  - {name: b, rollout: 'x:y', weight: 1}\n"
    )
    with pytest.raises(ConfigError) as error:
        resolve_config(path, [], PlatoonTinkerRLTrainerConfig)
    assert error.value.problems == [
        f"environments[{name}].weight: {weight}: train.task_sampler: uniform draws every training task of every "
        "entry equally often, with no entry's share, so it takes no weights; leave weight out (weight: tasks, the "
        "default, is uniform's own share), or set train.task_sampler: weighted to share the slots by weight"
        for name, weight in (("a", 3), ("b", 1))
    ]
    path.write_text(path.read_text().replace(", weight: 3", "").replace(", weight: 1", ""))
    loaded = resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config
    assert loaded.train.task_sampler == "uniform" and task_sampler_problems(loaded) == []


def test_an_unknown_task_sampler_is_refused():
    """A name that is neither core's nor ``module:Class``, while no entry names a package that could register it: the
    trainer config refuses it with the line the loader reports, and AutoDataset, which resolves the run's sampler,
    with resolution's message.  ``TrainConfig`` alone takes any name, which a package may register."""

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, TrainConfig
    from platoon.utils.config import ConfigError

    with pytest.raises(ConfigError, match="unknown task sampler 'shuffled'"):
        AutoDataset.from_config(config(environment("a"), task_sampler="shuffled"), "train")
    train = TrainConfig(model_name="m", renderer_name="r", task_sampler="shuffled")
    assert train.task_sampler == "shuffled"
    with pytest.raises(ConfigError) as error:
        PlatoonTinkerRLTrainerConfig(train=train, log_path="runs")
    assert error.value.problems == [
        "train.task_sampler: 'shuffled' is not one of 'auto', 'weighted', 'uniform', 'pass_rate', a name an "
        "environment's package registers, or module:Class"
    ]
    with pytest.raises(ValueError, match="takes 'auto' or 'weighted', got 'shuffled'"):  # the deprecated name's
        TrainConfig(model_name="m", renderer_name="r", task_order="shuffled")
    assert TrainConfig(model_name="m", renderer_name="r").task_sampler == "auto"
    # The deprecated name takes only the values it had: uniform came after it.
    uniform: Any = "uniform"
    with pytest.raises(ValueError, match="takes 'auto' or 'weighted', got 'uniform': set train.task_sampler instead"):
        TrainConfig(model_name="m", renderer_name="r", task_order=uniform)


@pytest.mark.parametrize(
    ("key", "line"),
    [
        (
            "task_sampler",
            "train.task_sampler: 'random' is not one of 'auto', 'weighted', 'uniform', 'pass_rate', a name an "
            "environment's package registers, or module:Class",
        ),
        ("task_order", "train.task_order: 'random' is not one of 'auto', 'weighted'"),
    ],
)
def test_a_config_file_s_unknown_sampler_names_the_samplers_under_either_name(tmp_path, key, line):
    """An unknown sampler, in the config's one report, in a file and in an override: ``task_sampler``'s line names
    core's samplers, a package's and ``module:Class``, for an override too, which parses it as any name; the deprecated
    ``task_order`` lists the values it takes, which ``uniform`` is not, and its override keeps the parser's
    message."""

    import warnings

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import ConfigError, resolve_config

    path = tmp_path / "train.yaml"
    path.write_text(f"train: {{model_name: m, renderer_name: r, {key}: random}}\nlog_path: runs\n")
    with warnings.catch_warnings(), pytest.raises(ConfigError) as error:
        warnings.simplefilter("ignore")  # task_order's deprecation
        resolve_config(path, [], PlatoonTinkerRLTrainerConfig)
    assert line in error.value.problems
    path.write_text("train: {model_name: m, renderer_name: r}\nlog_path: runs\n")
    if key == "task_sampler":
        with pytest.raises(ConfigError) as error:
            resolve_config(path, [f"train.{key}=random"], PlatoonTinkerRLTrainerConfig)
        assert line in error.value.problems
    else:
        with pytest.raises(ConfigError, match=rf"train\.{key} must be one of 'auto', 'weighted', got 'random'"):
            resolve_config(path, [f"train.{key}=random"], PlatoonTinkerRLTrainerConfig)


def test_task_order_is_the_deprecated_name_of_task_sampler():
    """``train.task_order``, ``task_sampler``'s name before platoon 0.3.0, is read as ``task_sampler`` is, and then
    cleared: the run has one sampler.  Beside ``task_sampler``, the same sampler agrees, and ``auto``, its default
    before 0.3.0, yields; ``weighted`` beside another sampler is an error."""

    from platoon.train.tinker.config_defs import TrainConfig

    def sampler(**names: str) -> tuple[str | None, str | None]:
        built = TrainConfig(model_name="m", renderer_name="r", **names)
        return built.task_sampler, built.task_order

    assert sampler(task_order="weighted") == ("weighted", None)
    assert sampler(task_order="auto") == ("auto", None)
    assert sampler(task_sampler="weighted", task_order="weighted") == ("weighted", None)
    assert sampler(task_sampler="auto", task_order="auto") == ("auto", None)
    assert sampler(task_sampler="weighted", task_order="auto") == ("weighted", None)  # 0.2.0's default yields
    with pytest.raises(
        ValueError,
        match=r"train.task_sampler is 'auto' and train.task_order, its deprecated name, is "
        r"'weighted'; set task_sampler alone",
    ):
        TrainConfig(model_name="m", renderer_name="r", task_sampler="auto", task_order="weighted")


TASK_ORDER_DEPRECATION = "train.task_order: deprecated; use train.task_sampler instead (platoon 0.4.0 removes it)"


@pytest.mark.parametrize(
    ("task_order", "task_sampler", "message"),
    [
        (
            "auto",
            "weighted",
            f"{TASK_ORDER_DEPRECATION}; its value 'auto' is ignored, since train.task_sampler is 'weighted'",
        ),
        ("auto", "auto", TASK_ORDER_DEPRECATION),
        ("weighted", "weighted", TASK_ORDER_DEPRECATION),
    ],
)
def test_a_config_file_may_set_both_names_and_the_warning_says_when_task_order_is_ignored(
    tmp_path, task_order, task_sampler, message
):
    """Both names load when they agree, or when ``task_order`` is ``auto``, its default before 0.3.0 (as a dump of
    0.2.0's ``config show --defaults`` spells it): ``task_sampler`` decides, and the deprecation warning then says
    that ``task_order`` was ignored.  ``lock.json``'s record of the deprecation stays the same."""

    from platoon.deprecations import PlatoonDeprecationWarning
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    path = tmp_path / "train.yaml"
    path.write_text(
        f"train: {{model_name: m, renderer_name: r, task_order: {task_order}, task_sampler: {task_sampler}}}\n"
        "log_path: runs\n"
    )

    with pytest.warns(PlatoonDeprecationWarning) as warned:
        resolved = resolve_config(path, [], PlatoonTinkerRLTrainerConfig)

    assert [str(item.message) for item in warned if item.category is PlatoonDeprecationWarning] == [message]
    assert [deprecation.message for deprecation in resolved.deprecations] == [message]
    assert [deprecation.to_dict() for deprecation in resolved.deprecations] == [
        {"path": "train.task_order", "replacement": "train.task_sampler", "removed_in": "0.4.0"}
    ]
    assert (resolved.config.train.task_sampler, resolved.config.train.task_order) == (task_sampler, None)


def test_a_config_file_whose_task_order_weighted_contradicts_task_sampler_does_not_load(tmp_path):
    import warnings

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import ConfigError, resolve_config

    path = tmp_path / "train.yaml"
    path.write_text("train: {model_name: m, renderer_name: r, task_order: weighted}\nlog_path: runs\n")

    with warnings.catch_warnings(), pytest.raises(ConfigError, match="set task_sampler alone") as error:
        warnings.simplefilter("ignore")  # task_order's deprecation
        resolve_config(path, ["train.task_sampler=auto"], PlatoonTinkerRLTrainerConfig)
    assert error.value.problems == [
        "ValueError: train.task_sampler is 'auto' and train.task_order, its deprecated name, is 'weighted'; set "
        "task_sampler alone"
    ]


def test_a_config_that_sets_task_order_loads_with_a_deprecation_and_draws_the_same_rows(tmp_path):
    """A config file that still sets ``train.task_order: weighted`` loads with a deprecation warning that names
    ``train.task_sampler``, and its run draws the rows of one that sets ``task_sampler: weighted``."""

    from platoon.deprecations import PlatoonDeprecationWarning
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    def write(name: str, key: str) -> Any:
        path = tmp_path / name
        path.write_text(
            f"train: {{model_name: m, renderer_name: r, batch_size: 4, {key}: weighted}}\n"
            "log_path: runs\n"
            "environments:\n"
            f"  - {{name: a, seed: 343, dataset_loader: {HERE}.load_ids, dataset_kwargs: {{prefix: a, count: 7}},\n"
            f"     task_loader: {HERE}.load_task, rollout: {HERE}.rollout_a}}\n"
        )
        return path

    with pytest.warns(PlatoonDeprecationWarning, match=r"train.task_order: deprecated; use train.task_sampler"):
        legacy = resolve_config(write("legacy.yaml", "task_order"), [], PlatoonTinkerRLTrainerConfig)
    current = resolve_config(write("current.yaml", "task_sampler"), [], PlatoonTinkerRLTrainerConfig)

    assert [deprecation.to_dict() for deprecation in legacy.deprecations] == [
        {"path": "train.task_order", "replacement": "train.task_sampler", "removed_in": "0.4.0"}
    ]
    assert current.deprecations == ()
    assert legacy.config.train.task_sampler == current.config.train.task_sampler == "weighted"
    rows = AutoDataset.training(legacy.config).order
    expected = AutoDataset.training(current.config).order
    assert rows is not None and expected is not None
    assert [[row["task_id"] for row in rows.epoch(epoch)] for epoch in range(3)] == [
        [row["task_id"] for row in expected.epoch(epoch)] for epoch in range(3)
    ]


def test_mixture_evaluation_visits_every_task_once_in_config_order():
    rows = AutoDataset.from_config(mixture(), "eval").to_list()

    assert [row["task_id"] for row in rows] == ["a::a-eval", "b::b-eval"]
    assert [row[ENVIRONMENT_COLUMN] for row in rows] == ["a", "b"]
    assert PRESERVE_DATASET_ORDER_COLUMN not in rows[0]


def test_an_evaluation_config_visits_every_task_of_a_mixtures_train_split_once():
    """An evaluation config (no train section) that evaluates the train split lists every task once, in config order,
    as for any other split: the weighted training order, which repeats a heavy entry's tasks and leaves out some of a
    light one's, is a training config's."""

    evaluation = SimpleNamespace(environments=mixture().environments)

    rows = AutoDataset.from_config(evaluation, "train").to_list()

    assert [row["task_id"] for row in rows] == ["a::a0", "a::a1"] + [f"b::b{index}" for index in range(5)]
    assert all(PRESERVE_DATASET_ORDER_COLUMN not in row for row in rows)
    # The training config's train split is the weighted order: a (weight 3) repeats, filled to whole batches.
    assert len(AutoDataset.from_config(mixture(), "train")) == 8


def test_mixture_keeps_extra_dataset_columns():
    rows_environment = environment(
        "rows", dataset_loader=f"{HERE}.load_rows", dataset_kwargs={}, eval_dataset_kwargs={}
    )
    mixed = config(environment("a"), rows_environment)

    rows = AutoDataset.from_config(mixed, "eval").to_list()

    assert {row["task_id"]: row["difficulty"] for row in rows if row[ENVIRONMENT_COLUMN] == "rows"} == {
        "rows::row-0": "hard",
        "rows::row-1": "easy",
    }


def test_task_router_loads_through_the_environment_and_attaches_its_route():
    router = AutoTaskLoader.from_config(mixture())
    shared = load_task("x")

    task = router("b::b3")

    assert isinstance(router, TaskRouter)
    assert task.id == "b3" and task.goal == "solve b3" and task.misc["shared"] is True
    route = EnvironmentRoute.of_task(task)
    assert route.name == "b"
    assert route.rollout == f"{HERE}:rollout_b" and route.eval_rollout == f"{HERE}:eval_rollout_b"
    assert route.reward_processor == f"{HERE}:double_reward"
    assert route.harness == choose_harness({"platoon-bash": 1, "mini-swe-agent": 1}, task_id="b3", seed=0)
    assert route.service == "harness_fakes:FakeService" and not route.resolve_task
    assert EnvironmentRoute.of_task(router("a::a0")).harness is None
    assert "platoon_environment" not in shared.misc  # loaders' own tasks are never mutated
    with pytest.raises(ValueError, match="unknown environment 'c'"):
        router("c::1")
    with pytest.raises(ValueError, match="does not start with"):
        router("a0")


@pytest.mark.asyncio
async def test_rollout_dispatch_follows_the_task_route():
    mixed = mixture()
    router = AutoTaskLoader.from_config(mixed)
    train = AutoRollout.from_config(mixed, "train")
    evaluate = AutoRollout.from_config(mixed, "eval")

    assert (train, evaluate) == (mixture_rollout, mixture_eval_rollout)
    assert validated_ref(train, what="rollout") == "platoon.train.auto:mixture_rollout"
    assert await train(router("a::a1"), "cfg") == {"env": "a", "task": "a1", "config": "cfg"}
    assert await evaluate(router("a::a1"), "cfg") == {"env": "a", "task": "a1", "config": "cfg"}
    assert await train(router("b::b0"), "cfg") == {"env": "b", "task": "b0"}
    assert await evaluate(router("b::b0"), "cfg") == {"env": "b-eval", "task": "b0"}
    with pytest.raises(ValueError, match="carries no environment route"):
        await train(Task(id="plain"), "cfg")


def test_reward_processing_follows_each_trajectory_route():
    mixed = mixture()
    router = AutoTaskLoader.from_config(mixed)
    assert AutoRewardProcessor.from_config(mixed) is mixture_reward_processor
    collection = TrajectoryCollection()
    for task_id, reward in (("a::a0", 1.0), ("b::b0", 0.5)):
        root = Trajectory(id=f"{task_id}/root", task=router(task_id), reward=reward)
        child = Trajectory(
            id=f"{task_id}/child", task=root.task.fork("delegate"), reward=reward, parent_info=ParentInfo(root.id, 0)
        )
        collection.trajectories.update({root.id: root, child.id: child})
    trajectories = list(collection.to_dict()["trajectories"].values())

    processed = [mixture_reward_processor(trajectory) for trajectory in trajectories]

    assert processed == [(1.0, {}), (1.0, {}), (1.0, {"reward/double": 1.0}), (1.0, {"reward/double": 1.0})]
    with pytest.raises(ValueError, match="carries no environment route"):
        mixture_reward_processor({"reward": 1.0, "task": {"misc": {}}})
    no_processors = config(environment("a"), environment("b"))
    assert AutoRewardProcessor.from_config(no_processors) is DEFAULT_REWARD_PROCESSOR


def test_weighted_harness_choice_is_deterministic_and_follows_weights():
    weights = {"platoon-bash": 3.0, "mini-swe-agent": 1.0}
    picks = [choose_harness(weights, task_id=f"task-{index}", seed=7) for index in range(4000)]

    assert picks == [choose_harness(weights, task_id=f"task-{index}", seed=7) for index in range(4000)]
    assert 0.72 < picks.count("platoon-bash") / len(picks) < 0.78
    assert picks != [choose_harness(weights, task_id=f"task-{index}", seed=8) for index in range(4000)]
    assert choose_harness(dict(reversed(weights.items())), task_id="task-1", seed=7) == picks[1]
    assert choose_harness("mini-swe-agent", task_id="x", seed=0) == "mini-swe-agent"
    assert choose_harness(None, task_id="x", seed=0) is None


def test_interleave_rotates_ties_and_cycles_pools():
    groups = {"x": ["x1"], "y": ["y1", "y2"], "z": ["z1"]}

    order = interleave_by_weight(groups, {"x": 1, "y": 1, "z": 1}, size=8, seeds={"x": 0, "y": 0, "z": 0})

    assert [item[0] for item in order] == ["x", "y", "z", "x", "y", "z", "x", "y"]
    assert order.count("x1") == 3 and order.count("z1") == 2
    assert {item for item in order if item.startswith("y")} == {"y1", "y2"}


def test_mixture_validation():
    with pytest.raises(ValueError, match="needs a `name`"):
        normalize_environment_configs([{"rollout": "x"}, {"name": "b"}])
    with pytest.raises(ValueError, match=r"repeated: \['a'\]"):
        normalize_environment_configs([{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValueError, match="different `workflow_kwargs`"):
        normalize_environment_configs(
            [{"name": "a", "workflow_kwargs": {"x": 1}}, {"name": "b", "workflow_kwargs": {"x": 2}}]
        )
    with pytest.raises(ValueError, match="weight must be a finite positive number"):
        EnvironmentConfig(weight=0)
    with pytest.raises(ValueError, match="placement must be one of"):
        EnvironmentConfig(placement="cloud")
    with pytest.raises(ValueError, match="builtins must be one of"):
        EnvironmentConfig(builtins="none")
    with pytest.raises(ValueError, match="a harness needs an environment `service`"):
        EnvironmentConfig(harness="platoon-bash")
    with pytest.raises(ValueError, match="each with a positive weight"):
        EnvironmentConfig(harness={"platoon-bash": -1}, service="harbor")
    with pytest.raises(ValueError, match="must not contain '::'"):
        EnvironmentConfig(name="a::b")


def test_run_level_settings_come_from_whichever_entry_sets_them():
    mixed = config(environment("a"), environment("b", workflow="custom/workflow", workflow_kwargs={"stats_scope": "x"}))

    merged = AutoEnvironment.from_config(mixed)

    assert merged.name == "a" and merged.workflow == "custom/workflow"
    assert merged.workflow_kwargs == {"stats_scope": "x"}

    class Default:
        pass

    assert AutoWorkflow.from_config(config(environment("a"), environment("b")), default=Default) is Default


def test_service_environments_list_and_reference_tasks_and_default_to_harness_rollout(tmp_path):
    FakeService.root = tmp_path
    service_only = config(
        EnvironmentConfig(
            service="harness_fakes:FakeService",
            harness="platoon-bash",
            eval_dataset_kwargs={"split_name": "test", "limit": 2},
        )
    )

    assert AutoDataset.from_config(service_only, "train")["task_id"] == ["train-0", "train-1", "train-2"]
    assert AutoDataset.from_config(service_only, "eval")["task_id"] == ["test-0", "test-1"]
    task = AutoTaskLoader.from_config(service_only)("train-1")
    route = EnvironmentRoute.of_task(task)
    assert task.id == "train-1" and route.resolve_task and route.name == "default"
    assert route.rollout == HARNESS_ROLLOUT and route.harness == "platoon-bash"
    from platoon.harness.rollout import harness_rollout

    assert AutoRollout.from_config(service_only, "train") is harness_rollout


def test_service_task_lists_concatenate_splits_then_take_a_seeded_sample():
    def sampled(**kwargs) -> SimpleNamespace:
        return config(
            EnvironmentConfig(service="harness_fakes:FakeService", harness="platoon-bash", eval_dataset_kwargs=kwargs)
        )

    listed = ["a-0", "a-1", "a-2", "b-0", "b-1", "b-2"]
    assert AutoDataset.from_config(sampled(split_name=["a", "b"]), "eval")["task_id"] == listed
    shuffled = list(listed)
    random.Random(3).shuffle(shuffled)
    sample = AutoDataset.from_config(sampled(split_name=["a", "b"], shuffle_seed=3, limit=4), "eval")
    assert sample["task_id"] == shuffled[:4]
    for bad in ({"shuffle_seed": "3"}, {"shuffle_seed": True}, {"split_name": []}, {"split_name": ["a", ""]}):
        with pytest.raises(ValueError, match="shuffle_seed|split_name"):
            AutoDataset.from_config(sampled(**bad), "eval")


def test_disabling_builtins_requires_a_harness_that_can():
    disabled = config(
        EnvironmentConfig(service="harness_fakes:FakeService", harness="platoon-bash", builtins="disable")
    )

    with pytest.raises(ValueError, match="cannot do"):
        AutoTaskLoader.from_config(disabled)
