"""Deterministic, distributed-aware sampling for OpenReward mixtures."""

from __future__ import annotations

import math
import random
import time
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, TypeVar

from torch.utils.data import Sampler

TInput = TypeVar("TInput")
TResult = TypeVar("TResult")


class AcceptedEnvironmentBatchObserver:
    """Observe native asynchronous accepted batches without changing dispatch.

    The balanced sampler controls submission order, while AReaL's native
    dispatcher returns whichever valid groups finish first. Results retain a
    task ID but not their original input row, so remember that small piece of
    metadata as inputs are consumed and summarize it after the native method
    returns. Delegating to the original method preserves its prefetch,
    staleness, rejection, and dynamic-batch behavior exactly.
    """

    def __init__(
        self,
        environment_order: Sequence[str],
        *,
        input_environment: Callable[[TInput], str],
    ) -> None:
        self.environment_order = list(dict.fromkeys(map(str, environment_order)))
        if not self.environment_order:
            raise ValueError("Accepted-environment observation requires at least one environment")
        self._known_environments = frozenset(self.environment_order)
        self._input_environment = input_environment
        self._pending_environments: dict[int, str] = {}
        self.last_accepted_counts: Counter[str] = Counter()
        self.last_accepted_total = 0
        self.last_unknown_results = 0

    def _observed_inputs(
        self,
        input_generator: Generator[TInput, None, None],
    ) -> Iterator[TInput]:
        for task_input in input_generator:
            task_id = getattr(task_input, "task_id", None)
            if not isinstance(task_id, bool) and isinstance(task_id, int):
                try:
                    environment = str(self._input_environment(task_input))
                except Exception:
                    # Telemetry must never make an otherwise valid rollout fail.
                    pass
                else:
                    self._pending_environments[task_id] = environment
            yield task_input

    def _record_results(self, dispatcher: Any, results: Sequence[TResult]) -> None:
        accepted_counts: Counter[str] = Counter()
        unknown_results = 0
        for result in results:
            task_id = getattr(result, "task_id", None)
            environment = self._pending_environments.pop(task_id, None)
            if environment not in self._known_environments:
                unknown_results += 1
                continue
            accepted_counts[environment] += 1

        # Native AReaL removes accepted and rejected IDs from this set. Using
        # it when available prevents rejected-input labels from accumulating;
        # pending prefetched tasks remain mapped for a later optimizer batch.
        active_task_ids = getattr(dispatcher, "_active_task_ids", None)
        if isinstance(active_task_ids, set):
            try:
                active_snapshot = set(active_task_ids)
            except RuntimeError:
                # A concurrent mutation is harmless; defer cleanup to the next
                # completed batch instead of risking training for observability.
                active_snapshot = None
            if active_snapshot is not None:
                self._pending_environments = {
                    task_id: environment
                    for task_id, environment in self._pending_environments.items()
                    if task_id in active_snapshot
                }

        self.last_accepted_counts = accepted_counts
        self.last_accepted_total = len(results)
        self.last_unknown_results = unknown_results

    def install(self, dispatcher: Any) -> None:
        existing = getattr(dispatcher, "_platoon_accepted_environment_observer", None)
        if existing is not None and existing is not self:
            raise RuntimeError("An accepted-environment observer is already installed")

        native_active_submit_and_wait = dispatcher.active_submit_and_wait

        def _active_submit_and_wait(
            input_generator: Generator[TInput, None, None],
            batch_size: int,
            dynamic_bs: bool = False,
        ) -> list[TResult]:
            results = native_active_submit_and_wait(
                self._observed_inputs(input_generator),
                batch_size,
                dynamic_bs,
            )
            self._record_results(dispatcher, results)
            return results

        dispatcher._platoon_accepted_environment_observer = self
        dispatcher.active_submit_and_wait = _active_submit_and_wait


class EnvironmentSamplingStartGate:
    """Skip environment inputs until their configured logical model version.

    AReaL's asynchronous dispatcher consumes ahead of the optimizer and keeps
    work in flight across accepted batches.  Gating the persistent input stream
    at dispatch time therefore gives the curriculum a durable clock (the
    rollout/controller version) without coupling it to a dataloader cursor or
    buffering inactive tasks.  Relative ``sampling_weight`` behavior is
    unchanged among environments which are currently admitted.

    Install this *after* :class:`AcceptedEnvironmentBatchObserver` so telemetry
    records only inputs which pass the curriculum gate and can actually be
    submitted.
    """

    def __init__(
        self,
        environment_start_steps: Mapping[str, int],
        *,
        input_environment: Callable[[TInput], str],
        current_step: Callable[[], int],
    ) -> None:
        if not environment_start_steps:
            raise ValueError("Environment sampling start gate requires at least one environment")
        self.environment_order = list(map(str, environment_start_steps))
        if len(set(self.environment_order)) != len(self.environment_order):
            raise ValueError("Environment sampling start gate labels must be unique")
        self._known_environments = frozenset(self.environment_order)
        self.environment_start_steps: dict[str, int] = {}
        for environment, raw_start_step in environment_start_steps.items():
            if (
                isinstance(raw_start_step, bool)
                or not isinstance(raw_start_step, int)
                or raw_start_step < 0
            ):
                raise ValueError(
                    "Environment sampling start steps must be non-negative integers"
                )
            self.environment_start_steps[str(environment)] = raw_start_step
        if not any(step == 0 for step in self.environment_start_steps.values()):
            raise ValueError("At least one environment sampling start step must be zero")

        self._input_environment = input_environment
        self._current_step = current_step
        self.last_step = 0
        self.last_admitted_environments = frozenset(
            environment
            for environment, start_step in self.environment_start_steps.items()
            if start_step == 0
        )
        self.last_admitted_input_counts: Counter[str] = Counter()
        self.last_skipped_input_counts: Counter[str] = Counter()
        self.total_skipped_input_counts: Counter[str] = Counter()

    def _validated_step(self) -> int:
        step = self._current_step()
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                "Environment sampling curriculum requires a non-negative integer logical step"
            )
        return step

    def _admitted_inputs(
        self,
        input_generator: Generator[TInput, None, None],
        *,
        step: int,
    ) -> Iterator[TInput]:
        for task_input in input_generator:
            environment = str(self._input_environment(task_input))
            if environment not in self._known_environments:
                raise ValueError(
                    f"Unknown input environment {environment!r}; expected one of "
                    f"{self.environment_order}"
                )
            if step < self.environment_start_steps[environment]:
                self.last_skipped_input_counts[environment] += 1
                self.total_skipped_input_counts[environment] += 1
                continue
            self.last_admitted_input_counts[environment] += 1
            yield task_input

    def install(self, dispatcher: Any) -> None:
        existing = getattr(dispatcher, "_platoon_environment_sampling_start_gate", None)
        if existing is not None and existing is not self:
            raise RuntimeError("An environment sampling start gate is already installed")

        native_active_submit_and_wait = dispatcher.active_submit_and_wait

        def _active_submit_and_wait(
            input_generator: Generator[TInput, None, None],
            batch_size: int,
            dynamic_bs: bool = False,
        ) -> list[TResult]:
            step = self._validated_step()
            self.last_step = step
            self.last_admitted_environments = frozenset(
                environment
                for environment, start_step in self.environment_start_steps.items()
                if step >= start_step
            )
            self.last_admitted_input_counts = Counter()
            self.last_skipped_input_counts = Counter()
            return native_active_submit_and_wait(
                self._admitted_inputs(input_generator, step=step),
                batch_size,
                dynamic_bs,
            )

        dispatcher._platoon_environment_sampling_start_gate = self
        dispatcher.active_submit_and_wait = _active_submit_and_wait


class StrictEnvironmentBatchCoordinator:
    """Admit exact per-step environment quotas without stale result backlogs.

    AReaL normally prefetches several batches and consumes whichever rollout
    groups finish first. A fast environment can then crowd a slower one out of
    an optimizer step. This coordinator submits only the current sampler quota,
    fully drains each round, and retries only labels which remain below quota.
    No accepted surplus crosses a model-version boundary.
    """

    def __init__(
        self,
        environment_batches: Sequence[Sequence[str]],
        *,
        input_environment: Callable[[TInput], str],
        start_batch_index: int = 0,
        max_replacement_rounds: int = 8,
    ) -> None:
        if not environment_batches:
            raise ValueError("Strict environment balancing requires target batches")
        self.environment_batches = [list(map(str, batch)) for batch in environment_batches]
        batch_sizes = {len(batch) for batch in self.environment_batches}
        if len(batch_sizes) != 1 or not next(iter(batch_sizes)):
            raise ValueError("Strict environment target batches must have one non-zero size")
        self.batch_size = next(iter(batch_sizes))
        self.environment_order = list(
            dict.fromkeys(environment for batch in self.environment_batches for environment in batch)
        )
        if (
            isinstance(max_replacement_rounds, bool)
            or not isinstance(max_replacement_rounds, int)
            or max_replacement_rounds < 0
        ):
            raise ValueError("max_replacement_rounds must be a non-negative integer")
        self.max_replacement_rounds = max_replacement_rounds
        self._known_environments = frozenset(self.environment_order)
        self._input_environment = input_environment
        self._buffer_limit_per_environment = self.batch_size
        self._buffered_inputs: dict[str, deque[TInput]] = {
            environment: deque() for environment in self.environment_order
        }
        self._batch_index = int(start_batch_index) % len(self.environment_batches)
        self.last_target_counts: Counter[str] = Counter()
        self.last_accepted_counts: Counter[str] = Counter()
        self.last_attempt_counts: Counter[str] = Counter()
        self.last_discarded_input_counts: Counter[str] = Counter()
        self.total_discarded_input_counts: Counter[str] = Counter()
        self.last_retry_rounds = 0

    def _validated_environment(self, value: str, *, source: str) -> str:
        environment = str(value)
        if environment not in self._known_environments:
            raise ValueError(f"Unknown {source} environment {environment!r}; expected one of {self.environment_order}")
        return environment

    def _take_inputs(
        self,
        input_generator: Generator[TInput, None, None],
        slots: Sequence[str],
        discarded_counts: Counter[str],
    ) -> list[TInput]:
        selected: list[TInput] = []
        for raw_environment in slots:
            environment = self._validated_environment(raw_environment, source="target")
            buffered = self._buffered_inputs[environment]
            if buffered:
                selected.append(buffered.popleft())
                continue

            while True:
                try:
                    task_input = next(input_generator)
                except StopIteration:
                    raise RuntimeError(
                        "Input generator exhausted before strict environment quota completion. "
                        "Use cycle_dataloader() or provide an infinite generator."
                    ) from None
                task_environment = self._validated_environment(
                    self._input_environment(task_input),
                    source="input",
                )
                if task_environment == environment:
                    selected.append(task_input)
                    break
                task_buffer = self._buffered_inputs[task_environment]
                if len(task_buffer) < self._buffer_limit_per_environment:
                    task_buffer.append(task_input)
                else:
                    discarded_counts[task_environment] += 1
        return selected

    @staticmethod
    def _wait_for_round(dispatcher: Any, count: int) -> list[Any]:
        last_log = 0.0
        while True:
            try:
                return dispatcher.wait_results(count=count, timeout=1)
            except TimeoutError:
                now = time.monotonic()
                if now - last_log < 30.0:
                    continue
                last_log = now
                logger = getattr(dispatcher, "logger", None)
                if logger is not None:
                    logger.warning(
                        "Strict environment batch is waiting for %s current-quota results; stats=%s",
                        count,
                        dispatcher.staleness_manager.get_stats(),
                    )

    @staticmethod
    def _missing_slots(
        target_slots: Sequence[str],
        accepted_counts: Counter[str],
    ) -> list[str]:
        remaining = Counter(target_slots)
        remaining.subtract(accepted_counts)
        missing: list[str] = []
        for environment in target_slots:
            if remaining[environment] <= 0:
                continue
            missing.append(environment)
            remaining[environment] -= 1
        return missing

    def _record_outcome(
        self,
        target_counts: Counter[str],
        accepted_counts: Counter[str],
        attempt_counts: Counter[str],
        discarded_input_counts: Counter[str],
        retry_rounds: int,
    ) -> None:
        self.last_target_counts = target_counts.copy()
        self.last_accepted_counts = accepted_counts.copy()
        self.last_attempt_counts = attempt_counts.copy()
        self.last_discarded_input_counts = discarded_input_counts.copy()
        self.total_discarded_input_counts.update(discarded_input_counts)
        self.last_retry_rounds = retry_rounds

    def prepare_batch(
        self,
        dispatcher: Any,
        input_generator: Generator[TInput, None, None],
        batch_size: int,
        dynamic_bs: bool = False,
    ) -> list[TResult]:
        if dynamic_bs:
            raise ValueError("Strict accepted-environment balancing is incompatible with dynamic_bs=true")
        if batch_size != self.batch_size:
            raise ValueError(
                f"Strict environment target batch size does not match the dispatcher: {self.batch_size} != {batch_size}"
            )
        if dispatcher.runner.max_queue_size < batch_size:
            raise ValueError(
                "Inference engine config's queue size is too small for strict environment "
                f"balance: {dispatcher.runner.max_queue_size} < {batch_size}."
            )

        target_slots = self.environment_batches[self._batch_index]
        target_counts = Counter(target_slots)
        accepted_counts: Counter[str] = Counter()
        attempt_counts: Counter[str] = Counter()
        discarded_input_counts: Counter[str] = Counter()
        accepted_results: list[TResult] = []
        retry_rounds = 0
        round_slots = list(target_slots)

        while round_slots:
            round_inputs = self._take_inputs(input_generator, round_slots, discarded_input_counts)
            submitted_labels: dict[int, str] = {}
            for task_input, environment in zip(round_inputs, round_slots, strict=True):
                task_id = getattr(task_input, "task_id", None)
                if isinstance(task_id, bool) or not isinstance(task_id, int):
                    raise TypeError("Strict environment task inputs require an integer task_id")
                if task_id in submitted_labels:
                    raise ValueError(f"Duplicate strict environment task id {task_id}")
                submitted_labels[task_id] = environment
                attempt_counts[environment] += 1
                dispatcher.submit_task_input(task_input)

            arrived = self._wait_for_round(dispatcher, len(round_inputs))
            if len(arrived) != len(round_inputs):
                raise RuntimeError(
                    "Strict environment dispatcher returned an incomplete drained round: "
                    f"expected={len(round_inputs)}, received={len(arrived)}"
                )
            for result in arrived:
                if result is None:
                    continue
                task_id = getattr(result, "task_id", None)
                if task_id not in submitted_labels:
                    raise RuntimeError(
                        f"Strict environment dispatcher returned an unknown accepted task id: {task_id!r}"
                    )
                environment = submitted_labels[task_id]
                if accepted_counts[environment] >= target_counts[environment]:
                    raise RuntimeError(
                        f"Strict environment batch received an accepted result beyond its quota for {environment!r}"
                    )
                accepted_counts[environment] += 1
                accepted_results.append(result)

            round_slots = self._missing_slots(target_slots, accepted_counts)
            if not round_slots:
                break
            if retry_rounds >= self.max_replacement_rounds:
                self._record_outcome(
                    target_counts,
                    accepted_counts,
                    attempt_counts,
                    discarded_input_counts,
                    retry_rounds,
                )
                raise RuntimeError(
                    "Strict environment batch exhausted replacement rounds after draining "
                    f"all attempts: target={dict(target_counts)}, "
                    f"accepted={dict(accepted_counts)}, attempts={dict(attempt_counts)}, "
                    f"input_discards={dict(discarded_input_counts)}"
                )
            retry_rounds += 1

        if len(accepted_results) != batch_size or accepted_counts != target_counts:
            raise RuntimeError(
                "Strict environment batch completed with inconsistent quotas: "
                f"target={dict(target_counts)}, accepted={dict(accepted_counts)}"
            )

        self._record_outcome(
            target_counts,
            accepted_counts,
            attempt_counts,
            discarded_input_counts,
            retry_rounds,
        )
        self._batch_index = (self._batch_index + 1) % len(self.environment_batches)
        return accepted_results

    def install(self, dispatcher: Any) -> None:
        existing = getattr(
            dispatcher,
            "_platoon_strict_environment_batch_coordinator",
            None,
        )
        if existing is not None and existing is not self:
            raise RuntimeError("A strict environment batch coordinator is already installed")

        def _active_submit_and_wait(
            input_generator: Generator[TInput, None, None],
            batch_size: int,
            dynamic_bs: bool = False,
        ) -> list[TResult]:
            return self.prepare_batch(
                dispatcher,
                input_generator,
                batch_size,
                dynamic_bs,
            )

        dispatcher._platoon_strict_environment_batch_coordinator = self
        dispatcher.active_submit_and_wait = _active_submit_and_wait


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
                f"global batch size ({global_batch_size}) must be divisible by world size ({num_replicas})"
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
            environment: [index for index, value in enumerate(self.environment_ids) if value == environment]
            for environment in self.environment_order
        }
        weights_by_environment: dict[str, float] = {}
        for environment, raw_weight in zip(self.environment_ids, sampling_weights):
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Sampling weights must be finite and positive")
            previous = weights_by_environment.setdefault(environment, weight)
            if previous != weight:
                raise ValueError(f"Environment {environment!r} has inconsistent sampling weights")
        total_weight = sum(weights_by_environment.values())
        self._probabilities = [
            weights_by_environment[environment] / total_weight for environment in self.environment_order
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
            credits = [credit + probability for credit, probability in zip(credits, self._probabilities)]
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
            cycle_seed = self.seed + self.epoch * 1_000_003 + environment_index * 10_007 + cycle * 101
            random.Random(cycle_seed).shuffle(pool)
        return pool

    def __iter__(self) -> Iterator[int]:
        pools = [self._shuffled_pool(index, 0) for index in range(len(self.environment_order))]
        positions = [0] * len(pools)
        cycles = [0] * len(pools)
        environment_indices = {environment: index for index, environment in enumerate(self.environment_order)}
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
        ordered = [{**record, preserve_order_key: True} for record in ordered]
    return ordered
