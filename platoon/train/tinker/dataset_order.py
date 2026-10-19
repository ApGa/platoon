"""How the training dataloader orders a dataset's rows, epoch by epoch.

The dataloader hands out batch ``b`` from epoch ``b // batches_per_epoch`` of the dataset's order, so the tasks a
batch holds depend only on its number: a resumed run, which starts at the batch its checkpoint recorded, draws what
the run it resumes would have drawn.  The order is one of:

- **A given order** (:class:`EpochOrder`), such as the weighted fair order of a mixture's tasks
  (:class:`platoon.train.components.WeightedOrder`), which ``platoon train`` hands the trainer: each epoch is that
  order's epoch.
- **The rows as they are**, every epoch, for a dataset whose rows are all marked with
  :data:`PRESERVE_DATASET_ORDER_COLUMN` (the marker is left out of the rows), and for one that is not shuffled.
- **Seeded shuffled passes** over the rows, for a dataset that is shuffled (:class:`ShuffledPasses`): pass ``p`` is
  ``Dataset.shuffle(seed + p * 101)``'s order, and each epoch takes up the passes where the epoch before it stopped.
  An epoch that holds only whole batches leaves the rows of its last partial batch to start the next one, so every
  row is drawn as often as every other, give or take one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from datasets import Dataset

PRESERVE_DATASET_ORDER_COLUMN = "_platoon_preserve_dataset_order"
# A shuffled dataset's seed stride per pass over its rows: pass 0 is Dataset.shuffle(seed), the first epoch's order.
PASS_SEED_STRIDE = 101


class EpochOrder(Protocol):
    """Rows, epoch by epoch: ``epoch(i)`` holds epoch ``i``'s ``epoch_size`` rows, in the order they are drawn."""

    @property
    def epoch_size(self) -> int: ...

    def epoch(self, index: int) -> Sequence[Mapping[str, Any]]: ...


def epoch_order(dataset: Dataset, *, shuffle_seed: int | None, epoch_size: int | None = None) -> EpochOrder:
    """The order a dataloader draws ``dataset``'s rows in, ``epoch_size`` rows per epoch (default: every row once):
    the rows as they are when they are marked as ordered or ``shuffle_seed`` is None, seeded shuffled passes
    otherwise.  Rows marked as ordered must all be marked true."""

    size = len(dataset) if epoch_size is None else epoch_size
    if PRESERVE_DATASET_ORDER_COLUMN in dataset.column_names:
        marker_values = dataset[PRESERVE_DATASET_ORDER_COLUMN]
        if not marker_values or any(value is not True for value in marker_values):
            raise ValueError(f"{PRESERVE_DATASET_ORDER_COLUMN} must be true for every ordered record")
        return FixedOrder(dataset.remove_columns(PRESERVE_DATASET_ORDER_COLUMN).to_list()[:size])
    if shuffle_seed is None:
        return FixedOrder(dataset.to_list()[:size])
    return ShuffledPasses(dataset, seed=shuffle_seed, epoch_size=size)


class FixedOrder:
    """The same rows every epoch."""

    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        self._rows = list(rows)

    @property
    def epoch_size(self) -> int:
        return len(self._rows)

    def epoch(self, index: int) -> Sequence[Mapping[str, Any]]:
        return self._rows


class ShuffledPasses:
    """A dataset's rows in seeded shuffled passes, read ``epoch_size`` rows at a time: pass ``p`` is
    ``dataset.shuffle(seed=seed + p * 101)``'s order, and epoch ``i`` holds positions ``i * epoch_size`` up to
    ``(i + 1) * epoch_size`` of the passes one after another.  Any epoch is read directly, without the ones before
    it."""

    def __init__(self, dataset: Dataset, *, seed: int, epoch_size: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"A shuffle seed is a whole number, 0 or more, got {seed!r}")
        if len(dataset) < 1 or epoch_size < 1:
            raise ValueError(f"Cannot draw epochs of {epoch_size} rows from a dataset of {len(dataset)}")
        self._dataset = dataset
        self._seed = seed
        self._epoch_size = epoch_size
        self._passes: dict[int, Dataset] = {}

    @property
    def epoch_size(self) -> int:
        return self._epoch_size

    def epoch(self, index: int) -> Sequence[Mapping[str, Any]]:
        rows: list[dict[str, Any]] = []
        position, end = index * self._epoch_size, (index + 1) * self._epoch_size
        count = len(self._dataset)
        while position < end:
            number, offset = divmod(position, count)
            take = min(count - offset, end - position)
            columns = self._pass(number)[offset : offset + take]
            rows.extend({name: values[row] for name, values in columns.items()} for row in range(take))
            position += take
        return rows

    def _pass(self, number: int) -> Dataset:
        if number not in self._passes:
            if len(self._passes) >= 2:  # an epoch spans two passes at most while it is shorter than one
                self._passes.pop(min(self._passes))
            self._passes[number] = self._dataset.shuffle(seed=self._seed + number * PASS_SEED_STRIDE)
        return self._passes[number]
