"""The training tasks in batches, epoch after epoch (``PlatoonTinkerDataloader``), and the steps a run plans."""

from collections.abc import Mapping, Sequence
from typing import Any

from datasets import Dataset

from platoon.train.dataset_order import SHUFFLE_SEED, EpochOrder, epoch_order
from platoon.train.sampling import planned_steps


def _run_steps(config: Any, batches_per_epoch: int) -> int:
    """The training steps a run plans, with ``batches_per_epoch`` batches an epoch (:func:`planned_steps`)."""

    steps = planned_steps(config.train, batches_per_epoch)
    if steps is None:
        raise ValueError("Must specify at least one of num_epochs or max_training_steps")
    return steps


class PlatoonTinkerDataloader:
    """Batches of task rows, epoch after epoch.

    Batch ``b`` is the ``b % len(self)``-th batch of epoch ``b // len(self)`` of the dataset's order
    (:mod:`platoon.train.dataset_order`): ``order`` when given, such as the weighted order of a mixture's tasks,
    which ``platoon train`` hands the trainer; else the dataset's rows in seeded shuffled passes (``shuffle_seed``),
    or as they are.  With ``drop_last`` every batch is whole: an epoch holds the whole batches its rows fill, and a
    shuffled dataset's last partial batch starts the next epoch instead of being left out of every one.  Without it,
    an epoch is every row once, and its last batch may be partial.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle_seed: int | None = SHUFFLE_SEED,
        drop_last: bool = True,
        order: EpochOrder | None = None,
    ):
        if order is None:
            if drop_last and len(dataset) < batch_size:
                raise ValueError(
                    f"batch_size {batch_size} is more than the dataset's {len(dataset)} rows, so no full batch exists"
                )
            epoch_size = len(dataset) // batch_size * batch_size if drop_last else len(dataset)
            order = epoch_order(dataset, shuffle_seed=shuffle_seed, epoch_size=epoch_size)
        elif order.epoch_size < 1 or (drop_last and order.epoch_size % batch_size):
            raise ValueError(f"An epoch of {order.epoch_size} rows is not a whole number of batches of {batch_size}")
        self._order = order
        self._batch_size = batch_size
        self._batches_per_epoch = -(-order.epoch_size // batch_size)
        self._epoch: tuple[int, Sequence[Mapping[str, Any]]] | None = None

    def get_batch(self, batch_num: int) -> list[dict]:
        epoch, index = divmod(batch_num, self._batches_per_epoch)
        if self._epoch is None or self._epoch[0] != epoch:
            self._epoch = (epoch, self._order.epoch(epoch))
        rows = self._epoch[1][index * self._batch_size : (index + 1) * self._batch_size]
        return [dict(row) for row in rows]

    def __len__(self) -> int:
        return self._batches_per_epoch
