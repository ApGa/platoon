from __future__ import annotations

import pytest

from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler


def test_deterministic_sampler_keeps_roots_and_allows_empty_subagent_trajectory():
    sampler = DeterministicSubagentDatumSampler(keep_probability=0.0, seed=7)

    assert sampler.sample_mask(
        task_id="task",
        trajectory_id="root",
        depth=0,
        num_datums=4,
    ) == [True, True, True, True]
    assert sampler.sample_mask(
        task_id="task",
        trajectory_id="child",
        depth=1,
        num_datums=4,
    ) == [False, False, False, False]
    assert sampler.sample_mask(
        task_id="task",
        trajectory_id="grandchild",
        depth=2,
        num_datums=2,
    ) == [False, False]


def test_probability_one_keeps_every_nonroot_datum():
    sampler = DeterministicSubagentDatumSampler(keep_probability=1.0, seed=7)

    assert sampler.sample_mask(
        task_id="task",
        trajectory_id="child",
        depth=3,
        num_datums=4,
    ) == [True, True, True, True]


def test_deterministic_sampler_is_reproducible_and_draws_per_datum():
    sampler = DeterministicSubagentDatumSampler(keep_probability=0.5, seed=19)
    kwargs = {
        "task_id": "task",
        "trajectory_id": "child",
        "depth": 2,
        "num_datums": 128,
    }

    first = sampler.sample_mask(**kwargs)
    second = sampler.sample_mask(**kwargs)

    assert first == second
    assert any(first)
    assert not all(first)
    assert sampler.sample_mask(**(kwargs | {"trajectory_id": "other-child"})) != first


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_deterministic_sampler_rejects_invalid_probability(probability: float):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        DeterministicSubagentDatumSampler(keep_probability=probability)


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_deterministic_sampler_requires_integer_seed(seed):
    with pytest.raises(ValueError, match="integer"):
        DeterministicSubagentDatumSampler(seed=seed)
