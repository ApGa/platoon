"""The sft trainer's options (:class:`SupervisedOptions`): ``trainer_options`` under ``trainer: sft``, checked when the
config loads.  Imports no Tinker."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SupervisedOptions:
    """``trainer_options`` under ``trainer: sft``: the datasets to train on, named as LLaMA-Factory names them
    (``dataset`` in the ``dataset_info.json`` of ``dataset_dir``), how many conversations a step trains on and for how
    long, and how each conversation is cut and which of its turns train."""

    # The options that name files, which the config loader makes absolute against the config file's directory, the
    # default too (platoon.utils.config).
    path_options: ClassVar[tuple[str, ...]] = ("dataset_dir",)

    # The datasets to train on, by their names in dataset_dir's dataset_info.json: one name, several separated by
    # commas (as LLaMA-Factory's dataset argument), or a list. Their conversations are put together in this order.
    dataset: str | list[str]
    # The directory that holds dataset_info.json and the files its entries name (LLaMA-Factory's dataset_dir),
    # relative to the config file; left out, the data directory beside the config file.
    dataset_dir: str = "data"
    # Conversations per training step, which take one optimizer step together.
    batch_size: int = 32
    # Passes over the conversations. With max_training_steps as well, training runs the shorter of the two; one of
    # them is required.
    num_epochs: int | None = None
    # Training steps to run. With num_epochs as well, training runs the shorter of the two.
    max_training_steps: int | None = None
    # How many forward_backward calls a step's conversations are sent in, before the step's one optimizer step.
    num_microbatches: int = 1
    # Tokens a rendered conversation is cut to, as LLaMA-Factory's cutoff_len cuts it: turn by turn from the first
    # (from the last with mask_history), each prompt and reply sharing what is left; null: no cut.
    max_length: int | None = None
    # Train on the last model turn of each conversation only (LLaMA-Factory's mask_history), not on every one.
    mask_history: bool = False
    # Seeds each epoch's order of the conversations, and which rows a dataset's num_samples takes.
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("dataset: name at least one dataset of dataset_dir's dataset_info.json")
        for name in ("batch_size", "num_microbatches"):
            if not _counts(getattr(self, name)):
                raise ValueError(f"{name}: must be an integer from 1, got {getattr(self, name)!r}")
        if self.num_epochs is None and self.max_training_steps is None:
            raise ValueError("set num_epochs or max_training_steps (with both, training runs the shorter)")
        for name in ("num_epochs", "max_training_steps"):
            value = getattr(self, name)
            if value is not None and not _counts(value):
                raise ValueError(f"{name}: must be an integer from 1, or null, got {value!r}")
        if self.num_microbatches > self.batch_size:
            raise ValueError(f"num_microbatches: {self.num_microbatches} is more than batch_size: {self.batch_size}")
        if self.max_length is not None and (not _counts(self.max_length) or self.max_length < 2):
            raise ValueError(f"max_length: must be an integer from 2, or null for no cut, got {self.max_length!r}")

    @property
    def datasets(self) -> list[str]:
        """The datasets' names, in order."""

        names = self.dataset.split(",") if isinstance(self.dataset, str) else self.dataset
        return [name.strip() for name in names if name.strip()]


def _counts(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1
