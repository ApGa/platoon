"""The conversations an sft run trains on (:func:`load_conversations`): the rows of each dataset the options name,
loaded as LLaMA-Factory loads them (``datasets.load_dataset``: its files in the dataset directory, or its dataset on
the Hugging Face Hub, then its split), ``num_samples`` of them sampled, each turned into a conversation in its
dataset's format; then all of them, in the order the options name the datasets.  Imports no Tinker."""

import hashlib
import logging
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import datasets

from platoon.train.supervised.cursor import shuffled
from platoon.train.supervised.formats import Conversation, SkippedRow, convert
from platoon.train.supervised.options import SupervisedOptions
from platoon.train.supervised.registry import FILE_TYPES, DatasetEntry, read_entries
from platoon.utils.config import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SupervisedDataset:
    """The conversations a run trains on, in order, and their fingerprint, by which a resume knows them for the ones
    its checkpoint walked."""

    conversations: tuple[Conversation, ...]
    fingerprint: str


def load_conversations(options: SupervisedOptions) -> SupervisedDataset:
    """The conversations of the datasets ``options`` name.  A row LLaMA-Factory would skip is skipped, with one
    warning per dataset that counts them and says why the first was.  Raises
    :class:`~platoon.utils.config.ConfigError` for a dataset that does not load or has no conversation to train on."""

    conversations: list[Conversation] = []
    for entry in read_entries(options.dataset_dir, options.datasets):
        if entry.ignored:
            ignored = ", ".join(entry.ignored)
            logger.warning(
                "Dataset %s: its entry's %s, which LLaMA-Factory does not read, is ignored", entry.name, ignored
            )
        rows = _rows(entry, options.dataset_dir)
        converted = _converted(entry, rows, _sampled(len(rows), entry.num_samples, seed=options.seed))
        if not converted:
            raise ConfigError(f"trainer_options.dataset: {entry.name} has no conversation to train on")
        logger.info("Dataset %s: %d conversations (%s format)", entry.name, len(converted), entry.formatting)
        conversations += converted
    return SupervisedDataset(tuple(conversations), _fingerprint(conversations))


def dataset_lines(options: SupervisedOptions) -> list[str]:
    """What ``platoon config validate --tasks`` lists of an sft run's datasets: each one's conversations, loaded as the
    run loads them (:func:`load_conversations`, which raises :class:`~platoon.utils.config.ConfigError`)."""

    counts = Counter(conversation.dataset for conversation in load_conversations(options).conversations)
    return [f"dataset {name}: {count} conversations" for name, count in counts.items()]


def _rows(entry: DatasetEntry, dataset_dir: str) -> list[dict[str, Any]]:
    """The entry's rows: its split of its Hub dataset (``hf_hub_url``, with ``subset`` and ``folder``), else of its
    files (``file_name``: a file, or every file of a folder, all of one type)."""

    try:
        if entry.hf_hub_url is not None:
            dataset = datasets.load_dataset(
                entry.hf_hub_url, name=entry.subset, data_dir=entry.folder, split=entry.split
            )
        else:
            loader, files = _files(entry, dataset_dir)
            dataset = datasets.load_dataset(loader, data_files=files, split=entry.split)
    except ConfigError:
        raise
    except Exception as exc:  # a missing file, a split it has not, a Hub dataset that does not load
        raise ConfigError(f"trainer_options.dataset: {entry.name} does not load: {type(exc).__name__}: {exc}") from exc
    return [dict(row) for row in dataset]


def _files(entry: DatasetEntry, dataset_dir: str) -> tuple[str, list[str]]:
    """The loader and the files of an entry's ``file_name``, in the dataset directory."""

    assert entry.file_name is not None  # an entry has hf_hub_url or file_name (registry)
    path = os.path.join(dataset_dir, entry.file_name)
    if os.path.isdir(path):
        files = sorted(os.path.join(path, name) for name in os.listdir(path))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise ConfigError(f"trainer_options.dataset: {entry.name}: {path} does not exist")
    loaders = {FILE_TYPES.get(os.path.splitext(file)[1][1:], "") for file in files}
    if len(loaders) != 1 or "" in loaders:
        raise ConfigError(
            f"trainer_options.dataset: {entry.name}: {path} must be files of one type, "
            f"one of {', '.join(sorted(FILE_TYPES))}"
        )
    (loader,) = loaders
    return loader, files


def _sampled(rows: int, num_samples: int | None, *, seed: int) -> list[int]:
    """Which of ``rows`` rows to take, by index: every one, in order, without ``num_samples``; else ``num_samples`` of
    them, every row once before any twice, as LLaMA-Factory samples them: in an order the seed decides, and again in
    that order while more rows are asked for than there are."""

    if num_samples is None or rows == 0:
        return list(range(rows))
    order = shuffled(rows, f"{seed}/num_samples")
    return [order[index % rows] for index in range(num_samples)]


def _converted(entry: DatasetEntry, rows: Sequence[dict[str, Any]], taken: list[int]) -> list[Conversation]:
    """The rows ``taken`` (by index) as conversations, the skipped ones left out, with a warning that counts them."""

    conversations: list[Conversation] = []
    skipped, first = 0, ""
    for index in taken:
        try:
            conversations.append(convert(entry, rows[index], index))
        except SkippedRow as exc:
            skipped, first = skipped + 1, first or f"row {index}: {exc}"
    if skipped:
        logger.warning(
            "Dataset %s: skipped %d of its %d rows, which LLaMA-Factory would not train on either; the first, %s",
            entry.name,
            skipped,
            len(taken),
            first,
        )
    return conversations


def _fingerprint(conversations: Sequence[Conversation]) -> str:
    """A short digest of the conversations, in order."""

    digest = hashlib.sha256()
    for conversation in conversations:
        digest.update(repr(conversation).encode())
    return digest.hexdigest()[:16]
