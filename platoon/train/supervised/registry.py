"""LLaMA-Factory's dataset registry: the ``dataset_info.json`` of a dataset directory, which names each dataset and says
where its rows are and how to read them (:func:`read_entries`, one :class:`DatasetEntry` each).  Imports no Tinker.

The keys are those of LLaMA-Factory's ``data/README.md``, read as its ``DatasetAttr`` reads them
(``src/llamafactory/data/parser.py``): ``hf_hub_url`` (a dataset on the Hugging Face Hub, which wins over the rest) or
``file_name`` (a file, or a folder of files of one type, in the dataset directory: json, jsonl, csv, parquet or
arrow); ``formatting``, ``alpaca`` (the default) or ``sharegpt``; ``split`` (``train``), ``subset``, ``folder`` and
``num_samples``; ``columns``, which column holds what; and for ``sharegpt``, ``tags``, how a message names its role
and its text.  An entry's ``columns`` (or ``tags``) replace LLaMA-Factory's defaults whole: a key they leave out names
nothing.  What this release does not read is refused by name: an entry whose rows come from another source
(``ms_hub_url``, ``om_hub_url``, ``script_url``, ``cloud_file_name``, which LLaMA-Factory reads over ``file_name``),
``formatting: openai``, a preference dataset (``ranking``, and the ``chosen``, ``rejected`` and ``kto_tag`` columns)
and a multimodal one (the ``images``, ``videos`` and ``audios`` columns).  A key LLaMA-Factory does not read either
(such as ``file_sha1``) is ignored, as it ignores it.
"""

import json
import os
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

from platoon.utils.config import ConfigError

if TYPE_CHECKING:
    from platoon.train.supervised.options import SupervisedOptions

# The registry file of a dataset directory.
INFO_NAME = "dataset_info.json"
# The file types a file_name may have, by extension, as datasets.load_dataset names their loaders.
FILE_TYPES = {"arrow": "arrow", "csv": "csv", "json": "json", "jsonl": "json", "parquet": "parquet"}
FORMATS = ("alpaca", "sharegpt")
# The sources besides hf_hub_url and file_name, in the order LLaMA-Factory prefers them when an entry has no
# hf_hub_url: each wins over file_name there.
_OTHER_SOURCES = ("ms_hub_url", "om_hub_url", "script_url", "cloud_file_name")
# The keys LLaMA-Factory reads of an entry.
_ENTRY_KEYS = {"hf_hub_url", "file_name", *_OTHER_SOURCES, "formatting", "ranking", "subset", "split", "folder"}
_ENTRY_KEYS |= {"num_samples", "columns", "tags"}
# The columns of preference and multimodal datasets.
_UNREAD_COLUMNS = ("chosen", "rejected", "kto_tag", "images", "videos", "audios")


@dataclass(frozen=True)
class Columns:
    """Which column of a row holds what (an entry's ``columns``): ``alpaca``'s prompt, query, response and history,
    ``sharegpt``'s messages, and either format's system prompt and tools; None: the rows have no such column.  The
    defaults are LLaMA-Factory's, for an entry without ``columns``."""

    prompt: str | None = "instruction"
    query: str | None = "input"
    response: str | None = "output"
    history: str | None = None
    messages: str | None = "conversations"
    system: str | None = None
    tools: str | None = None


@dataclass(frozen=True)
class Tags:
    """How a ``sharegpt`` message names its role and its text (an entry's ``tags``): the keys, then the role of each
    speaker: the user, the assistant, a tool's result, a tool call and the system prompt; None: no message has that
    role.  The defaults are LLaMA-Factory's, for an entry without ``tags``."""

    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset of a registry: its name, where its rows are (``hf_hub_url``, else ``file_name`` in the dataset
    directory), which of them to read (``subset``, ``folder``, ``split``, and ``num_samples`` rows of them, sampled),
    how to read each (``formatting``, ``columns``, ``tags``), and the keys of its entry that nothing reads."""

    name: str
    hf_hub_url: str | None = None
    file_name: str | None = None
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    subset: str | None = None
    folder: str | None = None
    split: str = "train"
    num_samples: int | None = None
    columns: Columns = field(default_factory=Columns)
    tags: Tags = field(default_factory=Tags)
    ignored: tuple[str, ...] = ()


def read_entries(dataset_dir: str, names: list[str]) -> list[DatasetEntry]:
    """The entries ``names`` of ``dataset_dir``'s ``dataset_info.json``, in order.  Raises
    :class:`~platoon.utils.config.ConfigError` with every problem: a registry that does not read, a name it does not
    hold, an entry this release does not read."""

    path = os.path.join(dataset_dir, INFO_NAME)
    where = "trainer_options.dataset"
    try:
        with open(path, encoding="utf-8") as file:
            registry = json.load(file)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"trainer_options.dataset_dir: cannot read {path}: {exc}") from exc
    if not isinstance(registry, dict):
        raise ConfigError(f"trainer_options.dataset_dir: {path} holds no JSON object of datasets")
    problems: list[str] = []
    entries: list[DatasetEntry] = []
    for name in names:
        if name not in registry:
            problems.append(f"{where}: {path} has no dataset {name!r}")
            continue
        try:
            entries.append(_entry(name, registry[name]))
        except ValueError as exc:
            problems.append(f"{where}: {name} in {path}: {exc}")
    if problems:
        raise ConfigError(problems)
    return entries


def entry_problems(options: "SupervisedOptions") -> list[str]:
    """What the config's load finds wrong with the datasets sft's ``options`` name, without loading them: each
    problem :func:`read_entries` raises."""

    try:
        read_entries(options.dataset_dir, options.datasets)
    except ConfigError as exc:
        return exc.problems
    return []


def _entry(name: str, info: Any) -> DatasetEntry:
    """``name``'s entry, from its object in the registry; a ``ValueError`` names what is wrong with it."""

    if not isinstance(info, dict):
        raise ValueError(f"expected an object, got {info!r}")
    if not info.get("hf_hub_url"):
        if other := [key for key in _OTHER_SOURCES if info.get(key)]:
            raise ValueError(
                f"its rows come from {other[0]}, which LLaMA-Factory reads over file_name and this release does not "
                "read: give it hf_hub_url, or file_name alone"
            )
        if not info.get("file_name"):
            raise ValueError("give it hf_hub_url or file_name")
    if info.get("ranking"):
        raise ValueError("ranking: preference datasets are not supported yet: trainer: sft trains on conversations")
    formatting = info.get("formatting", "alpaca")
    if formatting == "openai":
        raise ValueError(
            "formatting: 'openai' (messages whose tool calls are an assistant's tool_calls) is not supported yet; "
            "messages without tool calls read as sharegpt with the tags role_tag: role, content_tag: content, "
            "user_tag: user, assistant_tag: assistant and system_tag: system"
        )
    if formatting not in FORMATS:
        raise ValueError(f"formatting: {formatting!r} is not one of {', '.join(FORMATS)}")
    num_samples = info.get("num_samples")
    if num_samples is not None and (
        isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples < 1
    ):
        raise ValueError(f"num_samples: must be an integer from 1, got {num_samples!r}")
    columns = _section(Columns, info, "columns")
    tags = _section(Tags, info, "tags")
    _check_reads_rows(formatting, columns, tags)
    return DatasetEntry(
        name=name,
        hf_hub_url=info.get("hf_hub_url") or None,
        file_name=info.get("file_name") or None,
        formatting=formatting,
        subset=info.get("subset"),
        folder=info.get("folder"),
        split=info.get("split", "train"),
        num_samples=num_samples,
        columns=columns,
        tags=tags,
        ignored=tuple(sorted(set(info) - _ENTRY_KEYS)),
    )


def _section(cls: type[Columns] | type[Tags], info: dict[str, Any], key: str) -> Any:
    """The entry's ``columns`` or ``tags`` (``cls``): LLaMA-Factory's defaults without them; with them, what they
    name, and None for each key they leave out (as LLaMA-Factory's ``DatasetAttr.join`` sets it)."""

    if key not in info:
        return cls()
    values = info[key]
    if not isinstance(values, dict):
        raise ValueError(f"{key}: expected an object, got {values!r}")
    if unread := [column for column in _UNREAD_COLUMNS if values.get(column)]:
        raise ValueError(
            f"{key}: {', '.join(unread)}: preference and multimodal datasets are not supported yet: trainer: sft "
            "trains on text conversations"
        )
    known = [item.name for item in fields(cls)]
    if unknown := sorted(set(values) - set(known) - set(_UNREAD_COLUMNS)):
        raise ValueError(f"{key}: unknown keys {unknown}; it takes {known}")
    return cls(**{name: values.get(name) for name in known})


def _check_reads_rows(formatting: str, columns: Columns, tags: Tags) -> None:
    """Refuse ``columns`` or ``tags`` that leave out what every row of the format is read by: alpaca's response,
    sharegpt's messages and the keys of a message's role and text (LLaMA-Factory would stop on the first row)."""

    needed = {
        "alpaca": {"columns.response": columns.response},
        "sharegpt": {
            "columns.messages": columns.messages,
            "tags.role_tag": tags.role_tag,
            "tags.content_tag": tags.content_tag,
        },
    }[formatting]
    if missing := [key for key, value in needed.items() if not value]:
        raise ValueError(
            f"{', '.join(missing)}: name it: an entry's columns and tags replace LLaMA-Factory's defaults, so a key "
            "they leave out names nothing"
        )
