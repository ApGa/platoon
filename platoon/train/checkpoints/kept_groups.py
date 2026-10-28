"""The groups a full checkpoint keeps: those that settled and that no committed step consumed (RFC 0019, question 4).

A group runs for minutes or hours; it settles, waits in the result queue for the step that dequeues it, and that step
commits.  So that a resume from a checkpoint saved in between need not roll such a group out again, each full
checkpoint writes them to ``tasks/kept-s<start>-<checkpoint>.json.gz`` in the run directory (:func:`write_kept_groups`),
and the state of the dispatcher's draws that it saves beside them names the file with its SHA-256
(:func:`platoon.train.checkpoints.dispatch_state.save_state`); a start that resumes reads them back (:func:`read_kept`)
and the steps after it train them as they are, each once.

What a group keeps: its datums, exactly (token ids, and every loss input; floats as their bytes), and what the step
that consumes it reads of it besides, by the codec of its output's exact type (:func:`register_codec`): for a
``GroupRolloutWorkflow`` group (``GroupOutput``, whose codec :mod:`platoon.train.workflow.output` registers), its
workload, rollout counts, rollout statistics, deferred group statistics, label, entry, members and verdict, and for one
whose rows are tagged by who produced them (``PolicyGroupOutput``, under a credit estimator that reads agents), each
row's tags and the same report; for a workflow that returns a plain list of datums, the list; for a group whose
workflow failed, that it failed, so that the
step after a resume consumes it as failed, as the run would have without one.  An output of any other type is not
kept: a resume draws its task again.  The file is gzip-compressed JSON, as the snapshot is JSON: no pickle, so reading
a run directory runs no code.  Writing it costs about ten bytes per token of the groups it keeps, which every full save
writes again while they wait for a step.
"""

from __future__ import annotations

import base64
import gzip
import json
import logging
import math
import os
import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tinker

from platoon.train.task_stats import GroupRecord, read_tasks_file, write_tasks_file

logger = logging.getLogger(__name__)

# The format of a kept-groups file, which the snapshot names.
KEPT_FORMAT = 1


def kept_name(start: int, checkpoint: str) -> str:
    """The file of the groups a start's checkpoint keeps: ``kept-s<start>-<checkpoint>.json.gz``."""

    return f"kept-s{start}-{checkpoint}.json.gz"


def encode_group(
    key: tuple[int, int], *, entry: str | None, label: str | None, record: GroupRecord | None, output: Any
) -> dict[str, Any] | None:
    """A settled group as JSON data (``output`` None for a group whose workflow failed), or None for one that cannot
    be kept: no codec keeps an output of its type (:func:`register_codec`), or its codec does not keep this one."""

    encoded = _encode_output(output)
    if encoded is None:
        return None
    return {
        "key": list(key),
        "entry": entry,
        "label": label,
        "record": record.to_json() if record is not None else None,
        "output": encoded,
    }


def decode_group(data: Mapping[str, Any]) -> tuple[tuple[int, int], str | None, str | None, GroupRecord | None, Any]:
    """``(key, entry, label, record, output)`` of a group :func:`encode_group` wrote; a ``ValueError`` for data it
    cannot read."""

    try:
        start, seq = data["key"]
        record = data["record"]
        return (
            (int(start), int(seq)),
            None if data["entry"] is None else str(data["entry"]),
            None if data["label"] is None else str(data["label"]),
            GroupRecord.from_json(record) if record is not None else None,
            _decode_output(data["output"]),
        )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"cannot read a kept group: {type(exc).__name__}: {exc}") from exc


def write_kept(run_dir: str | os.PathLike[str], name: str, groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write ``groups`` (:func:`encode_group`'s) as ``tasks/<name>`` in ``run_dir``, durably (a file synced, then
    renamed into place), and return what the dispatcher's state records of it: ``{file, sha256, format, groups}``, the
    groups' keys in their order."""

    content = gzip.compress(
        (json.dumps({"format": KEPT_FORMAT, "groups": list(groups)}, separators=(",", ":")) + "\n").encode(),
        compresslevel=1,
        mtime=0,
    )
    return {
        **write_tasks_file(run_dir, name, content),
        "format": KEPT_FORMAT,
        "groups": [list(group["key"]) for group in groups],
    }


def write_kept_groups(
    run_dir: str | os.PathLike[str], start: int, checkpoint: str, kept: Sequence[tuple[tuple[int, int], Any]]
) -> dict[str, Any] | None:
    """Write the groups that start ``start``'s checkpoint ``checkpoint`` keeps (``kept``: ``(key, item)`` each, an item
    as a step dequeues it, with its ``entry``, ``label``, ``record`` and ``rollout``), those that can be kept, and
    return :func:`write_kept`'s record; None when none can, or, with a warning, when the disk refuses the file: a
    resume from the checkpoint then rolls their tasks out again."""

    groups = []
    for key, item in kept:
        try:
            group = encode_group(
                key,
                entry=item.entry,
                label=item.label,
                record=item.record,
                output=item.rollout,
            )
        except Exception:  # a group that cannot be kept is rolled out again, and costs no other
            logger.warning("Cannot keep the settled group %s with checkpoint %s", key, checkpoint, exc_info=True)
            group = None
        if group is not None:
            groups.append(group)
    if not groups:
        return None
    try:
        return write_kept(run_dir, kept_name(start, checkpoint), groups)
    except OSError as exc:
        logger.warning(
            "Could not write the %d groups that settled and no step has trained yet with checkpoint %s (%s): a "
            "resume from it rolls their tasks out again",
            len(groups),
            checkpoint,
            exc,
        )
        return None


def read_kept(run_dir: str | os.PathLike[str], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The groups of the file a dispatcher's state's ``kept`` record names, checked against its SHA-256; a
    ``ValueError`` that says why when it is missing, changed or unreadable."""

    if not isinstance(record.get("file"), str) or not record.get("file"):
        raise ValueError(f"the checkpoint names no kept-groups file: {dict(record)!r}")
    content = read_tasks_file(run_dir, record)
    path = record["file"]
    try:
        data = json.loads(gzip.decompress(content))
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"{path} is not gzip-compressed JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != KEPT_FORMAT or not isinstance(data.get("groups"), list):
        raise ValueError(f"{path} holds no kept groups of format {KEPT_FORMAT}")
    return data["groups"]


# --- outputs: a codec for each type of output -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Codec:
    """How a full checkpoint keeps an output of one type (``cls``): its tag in the file (its ``type``), its data as
    JSON (``encode``; None for an output it does not keep), and the output read back from that data (``decode``)."""

    cls: type
    tag: str
    encode: Callable[[Any], dict[str, Any] | None]
    decode: Callable[[Mapping[str, Any]], Any]


# The codecs, by the exact type of the output they keep (a subclass, whose own fields would be lost, is not kept) and
# by their tag.
_BY_TYPE: dict[type, _Codec] = {}
_BY_TAG: dict[str, _Codec] = {}


def register_codec(
    cls: type, tag: str, encode: Callable[[Any], dict[str, Any] | None], decode: Callable[[Mapping[str, Any]], Any]
) -> None:
    """Keep each output whose type is exactly ``cls`` as ``{"type": tag, **encode(output)}`` (``encode`` returns None
    for one it does not keep, and its data names no ``type`` of its own), and read it back with ``decode``, which gets
    that data.  Datums keep exactly with :func:`encode_datum` and :func:`decode_datum`.  A tag is one class's: another
    class's is refused, and ``failed`` is a failed group's.  The same class registered again, as a module run again
    registers it (a reload, or a test that loads the module afresh), keeps the codec it was first registered with for
    reading."""

    if tag == "failed":
        raise ValueError(f"the kept-group tag 'failed' is a failed group's, not {_name(cls)}'s")
    first = _BY_TAG.get(tag)
    if first is not None and _name(first.cls) != _name(cls):
        raise ValueError(f"the kept-group tag {tag!r} is {_name(first.cls)}'s, not {_name(cls)}'s")
    codec = _Codec(cls, tag, encode, decode)
    _BY_TYPE[cls] = codec
    _BY_TAG.setdefault(tag, codec)


def _name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _encode_output(output: Any) -> dict[str, Any] | None:
    if output is None:  # its workflow failed
        return {"type": "failed"}
    codec = _BY_TYPE.get(type(output))
    if codec is None:
        return None
    data = codec.encode(output)
    if data is None:
        return None
    if "type" in data:
        raise ValueError(f"the {codec.tag!r} codec's data names a type of its own")
    return {"type": codec.tag, **data}


def _decode_output(data: Mapping[str, Any]) -> Any:
    if data["type"] == "failed":
        return None
    codec = _BY_TAG.get(data["type"])
    if codec is None:
        raise ValueError(f"not a kept output: {data['type']!r}")
    return codec.decode(data)


def _encode_datums(output: list[Any]) -> dict[str, Any] | None:
    """A workflow's plain list of datums as it is kept; None for a list that holds anything else."""

    if not all(isinstance(datum, tinker.Datum) for datum in output):
        return None
    return {"datums": [encode_datum(datum) for datum in output]}


def _decode_datums(data: Mapping[str, Any]) -> list[tinker.Datum]:
    return [decode_datum(datum) for datum in data["datums"]]


register_codec(list, "datums", _encode_datums, _decode_datums)


def _encode_datum(datum: tinker.Datum) -> dict[str, Any]:
    """A datum as JSON data that reads back exactly (:func:`decode_datum`): its model input, and every loss input."""

    return {
        "model_input": datum.model_input.model_dump(mode="json"),
        "loss_fn_inputs": {name: _encode_tensor(value) for name, value in datum.loss_fn_inputs.items()},
    }


def _decode_datum(data: Mapping[str, Any]) -> tinker.Datum:
    """The datum :func:`encode_datum` wrote."""

    return tinker.Datum(
        model_input=tinker.ModelInput.model_validate(data["model_input"]),
        loss_fn_inputs={str(name): _decode_tensor(value) for name, value in data["loss_fn_inputs"].items()},
    )


# A datum as a codec keeps it (register_codec), and reads it back.  The private names are the module's in platoon
# 0.4.0, which its old path still reaches.
encode_datum = _encode_datum
decode_datum = _decode_datum


def _encode_tensor(value: tinker.TensorData) -> dict[str, Any]:
    """A tensor's values: integers as they are, floats as the little-endian bytes of their float32 values (float64 for
    values a float32 does not hold exactly), so that they read back exactly, NaN and the infinities too."""

    encoded: dict[str, Any] = {"dtype": value.dtype, "shape": value.shape}
    if value.dtype != "float32":
        encoded["data"] = list(value.data)
        return encoded
    values = [float(item) for item in value.data]
    try:
        single = struct.pack(f"<{len(values)}f", *values)
    except OverflowError:  # beyond a float32's range
        single = None
    if single is not None and all(
        kept == item or (math.isnan(kept) and math.isnan(item))
        for kept, item in zip(struct.unpack(f"<{len(values)}f", single), values)
    ):
        encoded["f4"] = base64.b64encode(single).decode()
    else:
        encoded["f8"] = base64.b64encode(struct.pack(f"<{len(values)}d", *values)).decode()
    return encoded


def _decode_tensor(data: Mapping[str, Any]) -> tinker.TensorData:
    values: list[Any]
    if "f4" in data or "f8" in data:
        code, raw = ("f", base64.b64decode(data["f4"])) if "f4" in data else ("d", base64.b64decode(data["f8"]))
        values = list(struct.unpack(f"<{len(raw) // struct.calcsize(code)}{code}", raw))
    else:
        values = list(data["data"])
    return tinker.TensorData(data=values, dtype=data["dtype"], shape=data["shape"])
