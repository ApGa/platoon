"""Where an sft run is in its conversations (:class:`DatasetCursor`), and how many steps they make
(:func:`steps_per_epoch`, :func:`last_step`).  Imports no Tinker.

Each epoch walks every conversation once, in an order that the run's seed and the epoch's number decide; a step takes
the next ``batch_size`` of them, and the epoch's last step what is left.  The cursor takes part in every full
checkpoint (``loop_state["sft"]``: the epoch, how far into it the run is, and the seed and the dataset that decide the
order), so that a resume goes on with the next conversation and trains each one once per epoch.  A resume with another
seed or other conversations would walk another order, and is refused.
"""

import math
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from platoon.train.supervised.options import SupervisedOptions
from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint

# The cursor's key of a checkpoint row's loop_state.
KEY = "sft"


def shuffled(count: int, seed: str) -> list[int]:
    """``0 .. count - 1`` in the order ``seed`` decides: a Fisher-Yates shuffle that draws only ``random()``, whose
    sequence for a seed Python keeps from release to release (``random.shuffle``'s own algorithm it does not), so that
    a resume on another Python walks the same order."""

    draw = random.Random(seed)
    order = list(range(count))
    for last in range(count - 1, 0, -1):
        other = int(draw.random() * (last + 1))
        order[last], order[other] = order[other], order[last]
    return order


def steps_per_epoch(conversations: int, batch_size: int) -> int:
    """The steps of one pass over ``conversations``: whole batches, and a last, shorter one for what is left."""

    return math.ceil(conversations / batch_size)


def last_step(options: SupervisedOptions, cursor: "DatasetCursor", step: int) -> int:
    """The step after the run's last one, for a start at ``step`` whose cursor is where the start resumes: the step
    that ends ``num_epochs`` passes, counted on from the cursor at the options' ``batch_size``, or
    ``max_training_steps``, whichever comes first.  Counted from the cursor, a resume with another ``batch_size`` still
    trains each conversation once an epoch."""

    ends = [options.max_training_steps]
    if options.num_epochs is not None:
        ends.append(step + cursor.steps_left(options.num_epochs, options.batch_size))
    return min(end for end in ends if end is not None)


class DatasetCursor:
    """The run's place in ``conversations`` conversations: the epoch, and how many of its conversations were taken
    (``position``).  Each epoch's order is :func:`shuffled` by ``seed`` and the epoch.  A participant of
    the run's full checkpoints (:class:`~platoon.train.checkpoints.participants.CheckpointParticipant`)."""

    keys: tuple[str, ...] = (KEY,)

    def __init__(self, conversations: int, *, seed: int, fingerprint: str) -> None:
        self.conversations = conversations
        self.seed = seed
        self.fingerprint = fingerprint
        self.epoch = 0
        self.position = 0
        self._order: tuple[int, list[int]] | None = None  # the epoch whose order it is, and the order

    def take(self, count: int) -> list[int]:
        """The next ``count`` conversations, by their index (fewer at the end of an epoch); the cursor moves past
        them, to the next epoch once it took the last of this one."""

        taken = self.order(self.epoch)[self.position : self.position + count]
        self.position += len(taken)
        if self.position == self.conversations:
            self.epoch, self.position = self.epoch + 1, 0
        return taken

    def steps_left(self, epochs: int, batch_size: int) -> int:
        """The steps that take what is left of the first ``epochs`` epochs, ``batch_size`` conversations a step."""

        if self.epoch >= epochs:
            return 0
        rest = math.ceil((self.conversations - self.position) / batch_size)
        return rest + (epochs - self.epoch - 1) * steps_per_epoch(self.conversations, batch_size)

    def order(self, epoch: int) -> list[int]:
        """Epoch ``epoch``'s order of the conversations."""

        if self._order is None or self._order[0] != epoch:
            self._order = (epoch, shuffled(self.conversations, f"{self.seed}/{epoch}"))
        return self._order[1]

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return {KEY: {**self._walk(), "epoch": self.epoch, "position": self.position}}

    def resume(self, start: "StartPoint") -> None:
        """Go on from where the checkpoint the start resumes from left the conversations (a fresh run starts at the
        first); refuse a checkpoint that walked another order, or that no sft run saved."""

        if start.record is None:
            return
        saved = start.loop_state.get(KEY)
        if not isinstance(saved, Mapping):
            raise ConfigError(
                f"trainer: sft cannot resume from {start.record.name}: its row holds no loop_state.{KEY} (where an "
                "sft run was in its conversations); resume a run with the trainer that started it"
            )
        if (walked := {key: saved.get(key) for key in self._walk()}) != self._walk():
            raise ConfigError(
                f"trainer: sft cannot resume from {start.record.name}: it walked the conversations {walked}, and this "
                f"start has {self._walk()}: their order would differ, and some conversations would train twice in an "
                "epoch and others never. Resume with the run's datasets and seed, or start a new run "
                "(checkpoint.load_checkpoint_path)"
            )
        self.epoch, self.position = int(saved["epoch"]), int(saved["position"])

    def _walk(self) -> dict[str, Any]:
        """What decides the order the cursor walks: the conversations and the seed."""

        return {"conversations": self.conversations, "fingerprint": self.fingerprint, "seed": self.seed}
