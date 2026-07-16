"""Deterministic, distributed-aware sampling for OpenReward mixtures."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from torch.utils.data import Sampler


class BalancedEnvironmentSampler(Sampler[int]):
    """Construct balanced global input batches, then shard them across ranks.

    The sampler guarantees the configured mix for submitted global batches.
    AReaL executes rollouts asynchronously, so completion-order batching and
    rejected rollouts can still shift the accepted optimizer-step mix.
    """

    def __init__(
        self,
        environment_ids: Sequence[str],
        sampling_weights: Sequence[float],
        *,
        global_batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        if len(environment_ids) != len(sampling_weights):
            raise ValueError("Environment ids and sampling weights must have the same length")
        if not environment_ids:
            raise ValueError("A balanced environment sampler requires at least one task")
        if global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in range 0..{num_replicas - 1}")
        if global_batch_size % num_replicas != 0:
            raise ValueError(
                f"global batch size ({global_batch_size}) must be divisible by "
                f"world size ({num_replicas})"
            )

        self.environment_ids = [str(value) for value in environment_ids]
        self.global_batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        self.environment_order = list(dict.fromkeys(self.environment_ids))
        self._indices_by_environment = {
            environment: [
                index
                for index, value in enumerate(self.environment_ids)
                if value == environment
            ]
            for environment in self.environment_order
        }
        weights_by_environment: dict[str, float] = {}
        for environment, raw_weight in zip(self.environment_ids, sampling_weights):
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Sampling weights must be finite and positive")
            previous = weights_by_environment.setdefault(environment, weight)
            if previous != weight:
                raise ValueError(
                    f"Environment {environment!r} has inconsistent sampling weights"
                )
        total_weight = sum(weights_by_environment.values())
        self._probabilities = [
            weights_by_environment[environment] / total_weight
            for environment in self.environment_order
        ]

        task_count = len(self.environment_ids)
        if drop_last:
            self.num_global_batches = task_count // global_batch_size
        else:
            self.num_global_batches = math.ceil(task_count / global_batch_size)
        if self.num_global_batches == 0:
            raise ValueError(
                "The mixed training dataset is smaller than one global batch; "
                "lower train_dataset.batch_size or select more tasks"
            )
        self.total_size = self.num_global_batches * global_batch_size
        self.num_samples = self.total_size // num_replicas
        self._slot_environments = self._build_slot_schedule(self.total_size)

    def _build_slot_schedule(self, size: int) -> list[str]:
        """Weighted fair queue with rotating tie breaks.

        Equal weights for three environments and batch size eight yield
        3/3/2, then 3/2/3, then 2/3/3 instead of always assigning the two
        remainder slots to the first labels.
        """

        credits = [0.0] * len(self.environment_order)
        cursor = 0
        result: list[str] = []
        for _ in range(size):
            credits = [
                credit + probability
                for credit, probability in zip(credits, self._probabilities)
            ]
            maximum = max(credits)
            candidates = {
                index
                for index, credit in enumerate(credits)
                if math.isclose(credit, maximum, rel_tol=0.0, abs_tol=1e-12)
            }
            chosen = next(
                index
                for offset in range(len(self.environment_order))
                if (index := (cursor + offset) % len(self.environment_order)) in candidates
            )
            credits[chosen] -= 1.0
            cursor = (chosen + 1) % len(self.environment_order)
            result.append(self.environment_order[chosen])
        return result

    def environment_batches(self) -> list[list[str]]:
        return [
            self._slot_environments[start : start + self.global_batch_size]
            for start in range(0, self.total_size, self.global_batch_size)
        ]

    def _shuffled_pool(self, environment_index: int, cycle: int) -> list[int]:
        environment = self.environment_order[environment_index]
        pool = list(self._indices_by_environment[environment])
        if self.shuffle:
            cycle_seed = (
                self.seed
                + self.epoch * 1_000_003
                + environment_index * 10_007
                + cycle * 101
            )
            random.Random(cycle_seed).shuffle(pool)
        return pool

    def __iter__(self) -> Iterator[int]:
        pools = [self._shuffled_pool(index, 0) for index in range(len(self.environment_order))]
        positions = [0] * len(pools)
        cycles = [0] * len(pools)
        environment_indices = {
            environment: index
            for index, environment in enumerate(self.environment_order)
        }
        global_indices: list[int] = []
        for environment in self._slot_environments:
            environment_index = environment_indices[environment]
            if positions[environment_index] == len(pools[environment_index]):
                cycles[environment_index] += 1
                pools[environment_index] = self._shuffled_pool(
                    environment_index,
                    cycles[environment_index],
                )
                positions[environment_index] = 0
            position = positions[environment_index]
            global_indices.append(pools[environment_index][position])
            positions[environment_index] += 1
        return iter(global_indices[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def materialize_balanced_record_order(
    records: Sequence[Mapping[str, Any]],
    *,
    environment_key: str,
    sampling_weight_key: str,
    global_batch_size: int,
    seed: int = 0,
    preserve_order_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return complete batches ordered by the configured environment mixture."""

    sampler = BalancedEnvironmentSampler(
        [str(record[environment_key]) for record in records],
        [float(record[sampling_weight_key]) for record in records],
        global_batch_size=global_batch_size,
        seed=seed,
        shuffle=True,
        drop_last=True,
    )
    ordered = [dict(records[index]) for index in sampler]
    if preserve_order_key is not None:
        ordered = [
            {**record, preserve_order_key: True}
            for record in ordered
        ]
    return ordered
