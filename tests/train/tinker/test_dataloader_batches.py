"""The training dataloader hands out whole batches of task rows, and refuses a batch size the dataset cannot fill.

``datasets.Dataset.batch(n, drop_last_batch=True)`` returns the rows themselves, unbatched, when there are fewer
than ``n``; ``get_batch`` would then read one task id as a batch of one-character task ids.
"""

from __future__ import annotations

import pytest

from platoon.train.components import task_ids_to_dataset
from platoon.train.tinker.rl import PlatoonTinkerDataloader

TASK_IDS = ["train:0", "train:1", "train:2", "train:3"]


def test_the_dataloader_refuses_a_batch_it_cannot_fill():
    with pytest.raises(ValueError, match="batch_size 8 is more than the dataset's 4 rows"):
        PlatoonTinkerDataloader(task_ids_to_dataset(TASK_IDS), batch_size=8, shuffle_seed=None)


def test_whole_batches_and_an_evaluation_loader_still_load():
    loader = PlatoonTinkerDataloader(task_ids_to_dataset(TASK_IDS), batch_size=4, shuffle_seed=None)
    assert [row["task_id"] for row in loader.get_batch(0)] == TASK_IDS

    evaluation = PlatoonTinkerDataloader(
        task_ids_to_dataset(TASK_IDS), batch_size=8, shuffle_seed=None, drop_last=False
    )
    assert [row["task_id"] for row in evaluation.get_batch(0)] == TASK_IDS


class _Epochs:
    """An order whose epoch ``i`` is ``i``'s tasks: ``e<i>-<n>``."""

    def __init__(self, epoch_size: int):
        self.epoch_size = epoch_size
        self.read: list[int] = []

    def epoch(self, index: int):
        self.read.append(index)
        return [{"task_id": f"e{index}-{n}"} for n in range(self.epoch_size)]


def test_a_given_order_gives_each_epoch_its_own_batches():
    order = _Epochs(epoch_size=4)
    loader = PlatoonTinkerDataloader(task_ids_to_dataset(TASK_IDS), batch_size=2, order=order)

    assert len(loader) == 2  # batches per epoch
    batches = [[row["task_id"] for row in loader.get_batch(number)] for number in range(5)]

    assert batches == [["e0-0", "e0-1"], ["e0-2", "e0-3"], ["e1-0", "e1-1"], ["e1-2", "e1-3"], ["e2-0", "e2-1"]]
    assert order.read == [0, 1, 2]  # each epoch drawn once, as its first batch is asked for
    # A resumed run starts at the batch its checkpoint recorded: the batch is the same as in the run it resumes.
    resumed = PlatoonTinkerDataloader(task_ids_to_dataset(TASK_IDS), batch_size=2, order=_Epochs(epoch_size=4))
    assert [row["task_id"] for row in resumed.get_batch(3)] == ["e1-2", "e1-3"]


def test_an_order_whose_epochs_are_not_whole_batches_is_refused():
    with pytest.raises(ValueError, match="An epoch of 5 rows is not a whole number of batches of 2"):
        PlatoonTinkerDataloader(task_ids_to_dataset(TASK_IDS), batch_size=2, order=_Epochs(epoch_size=5))


def test_a_shuffled_dataset_draws_every_task_as_often_as_every_other_across_epochs():
    """23 tasks in batches of 4 fill 5 batches an epoch: the first epoch is the dataset shuffled with seed 42, as it
    always was, and the 3 tasks it leaves out start the next epoch instead of being left out of every one, so over
    23 epochs each task is drawn 20 times; a dataloader asked for a later batch first (a resumed run) hands out the
    same batch."""

    from collections import Counter

    task_ids = [f"train:{index}" for index in range(23)]
    dataset = task_ids_to_dataset(task_ids)
    loader = PlatoonTinkerDataloader(dataset, batch_size=4)

    assert len(loader) == 5
    first_epoch = [row["task_id"] for number in range(5) for row in loader.get_batch(number)]
    assert first_epoch == dataset.shuffle(seed=42)["task_id"][:20]
    draws = Counter(row["task_id"] for number in range(5 * 23) for row in loader.get_batch(number))
    assert set(draws) == set(task_ids) and set(draws.values()) == {20}
    left_out = set(task_ids) - set(first_epoch)
    assert {row["task_id"] for row in loader.get_batch(5)[:3]} == left_out
    resumed = PlatoonTinkerDataloader(dataset, batch_size=4)
    assert resumed.get_batch(37) == loader.get_batch(37) and resumed.get_batch(6) == loader.get_batch(6)
