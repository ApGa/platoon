"""The JSON Lines records a training run appends to as it goes and its later starts read back:
``checkpoints_deleted.jsonl`` (:mod:`platoon.train.tinker.checkpoint_pruning`) and ``evaluations.jsonl``
(:mod:`platoon.train.tinker.evaluation_record`).

A process may die while it writes a line, so a reader skips a line that does not parse, and a writer starts its line on
a line of its own.
"""

from __future__ import annotations

import json
import os
from typing import Any


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """The JSON objects of the file at ``path``, one a line, in order; a line that does not parse (one being written,
    or one a crash cut short) or that holds no object is skipped, and a missing file has none."""

    try:
        with open(path) as file:
            lines = file.read().splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def append_jsonl_line(path: str | os.PathLike[str], line: str) -> None:
    """Append ``line`` to the file at ``path`` as a line of its own, also after a last line that a crash cut short."""

    with open(path, "a+b") as file:
        if file.seek(0, os.SEEK_END):
            file.seek(-1, os.SEEK_END)
            if file.read(1) != b"\n":
                file.write(b"\n")
        file.write(line.encode() + b"\n")
