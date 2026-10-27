"""Writes a golden trace down: what the fakes recorded as JSON data that two runs of the same code reproduce byte for
byte, with what varies from run to run masked (:func:`to_data`, :class:`Masks`, :func:`metric_tables`); a scenario's
two golden files (:func:`render`); and where two traces differ (:func:`describe_difference`)."""

from __future__ import annotations

import dataclasses
import difflib
import json
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

GOLDEN_DIR = Path(__file__).parent / "golden"
# A trace in memory, as run_scenario returns it; render() writes it as two files.
Trace = dict[str, Any]
# A golden file's lines are at most this wide, unless they hold a datum or an event.
WIDTH = 120


class OneLine(dict):
    """A dict that a golden file holds on one line, whatever its length: a datum, an event."""


class Expanded(dict):
    """A dict that a golden file holds one entry per line, whatever its length: the metrics under one parent path."""


# --------------------------------------------------------------------------- masks

_REQUEST_ID = re.compile(r"(-r\d+-)[0-9a-f]{8}\b")
_ADDRESS = re.compile(r"\b0x[0-9a-f]{6,}\b")
# The log messages that report how long something took.  Their durations are in virtual time, so they would not
# vary, but they would change with every change of the fakes' timing; any other number in a message stays.
_DURATION = re.compile(r"\b(Batch time: |Total time: |completed in |took |Waited |failed after )\d+(?:\.\d+)?s\b")


@dataclass(frozen=True)
class Masks:
    """What a trace masks in the text it holds: the run's temporary directory (``<tmp>``) and the import name of the
    module whose functions the configs name (``<fakes>``), which another layout of the tests would change; in a log
    message also the random part of request ids, object addresses and reported durations."""

    root: Path
    module: str

    def text(self, value: str) -> str:
        for prefix in sorted({str(self.root), os.path.realpath(self.root)}, key=len, reverse=True):
            value = value.replace(prefix, "<tmp>")
        return value.replace(f"{self.module}:", "<fakes>:")

    def message(self, value: str) -> str:
        value = _REQUEST_ID.sub(r"\1<id>", self.text(value))
        return _ADDRESS.sub("0x?", _DURATION.sub(r"\1<t>s", value))

    def strings(self, value: Any) -> Any:
        """``value`` with :meth:`text` applied to every string it holds, keeping each dict's and list's type."""

        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.strings(item) for item in value]
        if isinstance(value, dict):
            return type(value)((key, self.strings(item)) for key, item in value.items())
        return value


# --------------------------------------------------------------------------- data


def float_data(value: float) -> float | str:
    """A float as a trace holds it: ``nan``, ``inf`` and ``-inf`` as text; a float32 value as the shortest decimal
    that reads back as it in float32 (``0.1`` rather than ``0.10000000149011612``); any other as it is."""

    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    try:
        single = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return value
    if single != value:
        return value
    for digits in range(1, 10):
        text = float(f"{value:.{digits}g}")
        if struct.unpack("<f", struct.pack("<f", text))[0] == value:
            return text
    return value


def to_data(value: Any, masks: Masks | None = None) -> Any:
    """``value`` as JSON data: dataclasses, pydantic models and other objects as dicts of their fields, tuples as
    lists, tensors as lists, bytes by their length, floats by :func:`float_data`, strings masked by ``masks``."""

    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return float_data(value)
    if isinstance(value, torch.Tensor):
        return to_data(value.tolist(), masks)
    if isinstance(value, str):
        return masks.text(value) if masks is not None else value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Mapping):
        return {str(key): to_data(item, masks) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_data(item, masks) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_data(getattr(value, item.name), masks) for item in dataclasses.fields(value)}
    if hasattr(value, "model_dump"):  # a pydantic model, such as tinker.AdamParams
        return to_data(value.model_dump(), masks)
    if hasattr(value, "__dict__"):
        return {"class": type(value).__name__, **to_data(vars(value), masks)}
    return repr(value)


# --------------------------------------------------------------------------- metrics

# Metrics of the machine, or weighed by the wall-clock time between a rollout client's samples.
_MACHINE_METRICS = frozenset(
    "rollout_client/" + name
    for name in (
        "cpu_cores_busy",
        "cpu_percent",
        "cpu_saturation",
        "total_cpu_percent",
        "worker_cpu_percent",
        "rss_bytes",
        "threads",
        "concurrency_achieved_fraction",
        "groups_in_flight_mean",
        "rollouts_in_flight_mean",
        "trainer_queue_ready_fraction",
    )
)
# Metrics that measure time, by the end of their name (a group's rollout time, a sampler's decisions, a wait).
_TIMING_ENDINGS = ("_time", "_seconds", "_seconds_max", "_seconds_mean", "_seconds_total", "_ms", "_ms_max")
# Training metrics that the fakes' numbers leave inexact: means and extremes of e ** (log ratio), whose last bits
# depend on the CPU's vector width.  The clip fractions, which count the same weights, stay.
_INEXACT_METRICS = frozenset(f"optim/importance_weight_{name}" for name in ("mean", "std", "min", "max"))


def masked_value(key: str) -> str | None:
    """What a trace holds in place of metric ``key``'s values when they are not a function of the code: ``"timing"``
    for a metric of the wall clock or the machine, ``"inexact"`` for one whose last bits depend on the CPU; None
    otherwise.  A count (``__count``, ``/count``) is never masked.  A new metric of the wall clock belongs here."""

    if key.endswith(("__count", "/count")):
        return None
    path = key.split("/")
    name = path[-1]
    unscoped = "/".join(path[1:]) if path[0] in ("train", "eval") else key
    if unscoped in _INEXACT_METRICS:
        return "inexact"
    timing = (
        key in _MACHINE_METRICS
        or key.startswith("timing/")
        or name == "time"
        or name.endswith(_TIMING_ENDINGS)
        or "rollout_time" in path[:-1]
    )
    return "timing" if timing else None


def metric_tables(logged: Sequence[tuple[int, bool, Mapping[str, Any]]]) -> dict[str, Any]:
    """The metrics of each logged step, those of training (``train``) and of the evaluations (``eval``, logged with
    ``force``): the steps they were logged at, and each metric's values grouped by the metric's parent path
    (``train/batch/groups`` is ``groups`` under ``train/batch``).  A metric with the same value at every step holds
    that value, any other its value at each step, null where it was not logged; :func:`masked_value` replaces the
    values of the metrics that vary from run to run.  A ``__count`` that is 1 wherever it was logged is left out: a
    metric written twice in a step shows as its count of 2."""

    tables: dict[str, dict[str, Any]] = {}
    for step, force, stats in logged:
        table = tables.setdefault("eval" if force else "train", {"steps": [], "values": {}})
        column = len(table["steps"])
        table["steps"].append(step)
        for key, value in stats.items():
            values = table["values"].setdefault(key, [None] * column)
            values.extend([None] * (column - len(values)))
            values.append(masked_value(key) or to_data(value))
    result: dict[str, Any] = {}
    for kind, table in sorted(tables.items()):
        groups: dict[str, Expanded] = {}
        for key, values in sorted(table["values"].items()):
            values.extend([None] * (len(table["steps"]) - len(values)))
            if key.endswith("__count") and all(value is None or value == 1 for value in values):
                continue
            parent, _, name = key.rpartition("/")
            same = len({json.dumps(value) for value in values}) == 1
            groups.setdefault(parent, Expanded())[name] = values[0] if same else values
        result[kind] = {"steps": table["steps"], "values": groups}
    return result


def series(trace: Trace, key: str, *, start: int = 0, kind: str = "train") -> list[Any]:
    """Metric ``key``'s value at each step a start of ``trace`` logged (empty for a metric it never logged)."""

    table = trace["starts"][start]["metrics"].get(kind)
    parent, _, name = key.rpartition("/")
    if table is None or name not in table["values"].get(parent, {}):
        return []
    value = table["values"][parent][name]
    return value if isinstance(value, list) else [value] * len(table["steps"])


# --------------------------------------------------------------------------- the golden files


def golden_names(name: str) -> tuple[str, str]:
    """The names of scenario ``name``'s two golden files: its trace, and each start's metrics."""

    return f"{name}.json", f"{name}.metrics.json"


def render(trace: Trace) -> dict[str, str]:
    """The golden files of ``trace``, by name: its trace without the metrics, then each start's metrics, so that a
    pull request's file list shows whether training changed or only its metrics did."""

    main, metrics = golden_names(trace["scenario"])
    starts = [{key: value for key, value in start.items() if key != "metrics"} for start in trace["starts"]]
    return {
        main: _render({**trace, "starts": starts}, "") + "\n",
        metrics: _render({"scenario": trace["scenario"], "starts": [s["metrics"] for s in trace["starts"]]}, "") + "\n",
    }


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _render(value: Any, indent: str) -> str:
    """Pretty JSON that diffs line by line, keys sorted: a :class:`OneLine` on one line; an :class:`Expanded` one
    entry per line; a list of numbers on one line; a list of texts (messages, files) one per line; anything else on
    one line when it fits in :data:`WIDTH` characters, else one item per line."""

    flat = json.dumps(value, ensure_ascii=False, separators=(", ", ": "), sort_keys=True)
    if isinstance(value, OneLine) or not value or not isinstance(value, (dict, list)):
        return flat
    inner = indent + " "
    if isinstance(value, dict):
        if not isinstance(value, Expanded) and len(flat) <= WIDTH:
            return flat
        items = [
            f"{inner}{json.dumps(key, ensure_ascii=False)}: {_render(item, inner)}"
            for key, item in sorted(value.items())
        ]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    texts = all(isinstance(item, str) for item in value)
    if texts:
        expand = len(value) > 2 or len(flat) > WIDTH or any(len(item) > 60 for item in value)
    elif all(map(_scalar, value)):
        expand = False
    else:
        expand = len(flat) > WIDTH or any(isinstance(item, OneLine) for item in value)
    if not expand:
        return flat
    return "[\n" + ",\n".join(inner + _render(item, inner) for item in value) + "\n" + indent + "]"


# --------------------------------------------------------------------------- differences


def describe_difference(
    expected: Mapping[str, str], actual: Mapping[str, str], *, labels: tuple[str, str] = ("golden", "this run")
) -> str:
    """Where the golden files ``actual`` (name to text) differ from ``expected``: the parts of each start that
    differ, the first Tinker call, request and message that differ, the metrics added, removed and changed, then the
    first lines of each file's diff."""

    summary: list[str] = []
    diffs: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        old_text, new_text = expected.get(name, ""), actual.get(name, "")
        if old_text == new_text:
            continue
        old, new = _parse(old_text), _parse(new_text)
        summary += _metric_changes(name, old, new) if name.endswith(".metrics.json") else _trace_changes(name, old, new)
        diff = list(
            difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(), f"{labels[0]}/{name}", f"{labels[1]}/{name}", lineterm=""
            )
        )
        more = f"\n... ({len(diff) - 40} more lines of diff)" if len(diff) > 40 else ""
        diffs.append("\n".join(diff[:40]) + more)
    return "\n".join(["Differences:", *(f"- {line}" for line in summary), *diffs])


def _parse(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text) if text else {}
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _first(old: Sequence[Any], new: Sequence[Any]) -> int:
    return next((index for index, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))


def _short(value: Any, width: int = 100) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= width else text[: width - 3] + "..."


def _change(old: Any, new: Any, path: str = "") -> str:
    """``old -> new``: both whole when they are short, else where they first differ and how."""

    if not path and len(_short(old, 10**6)) <= 100 and len(_short(new, 10**6)) <= 100:
        return f"{_short(old)} -> {_short(new)}"
    if isinstance(old, dict) and isinstance(new, dict):
        key = next(key for key in sorted(set(old) | set(new)) if old.get(key) != new.get(key))
        return _change(old.get(key), new.get(key), f"{path}.{key}")
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        index = _first(old, new)
        return _change(old[index], new[index], f"{path}[{index}]")
    where = f"{path.lstrip('.')}: " if path else ""
    if isinstance(old, str) and isinstance(new, str):  # from a little before the first character that differs
        cut = max(0, _first(old, new) - 20)
        dots = "..." if cut else ""
        return f"{where}{dots}{_short(old[cut:], 70)} -> {dots}{_short(new[cut:], 70)}"
    return f"{where}{_short(old, 60)} -> {_short(new, 60)}"


def _trace_changes(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines = [
        f"{name}: {key}: {_change(old.get(key), new.get(key))}"
        for key in sorted(set(old) | set(new))
        if key != "starts" and old.get(key) != new.get(key)
    ]
    old_starts, new_starts = old.get("starts", []), new.get("starts", [])
    if len(old_starts) != len(new_starts):
        lines.append(f"{name}: {len(old_starts)} start(s) -> {len(new_starts)}")
    for index, (before, after) in enumerate(zip(old_starts, new_starts)):
        for part in sorted(set(before) | set(after)):
            was, now = before.get(part), after.get(part)
            if was == now:
                continue
            where = f"starts[{index}].{part}"
            if isinstance(was, list) and isinstance(now, list):
                at = _first(was, now)
                change = _change(was[at], now[at]) if at < min(len(was), len(now)) else "an item more or fewer"
                lines.append(f"{where}[{at}], the first that differs ({len(was)} -> {len(now)} in all): {change}")
            elif part == "requests" and isinstance(was, dict) and isinstance(now, dict):  # by scope
                for scope in sorted(set(was) | set(now)):
                    rows, new_rows = was.get(scope, []), now.get(scope, [])
                    if rows != new_rows:
                        at = _first(rows, new_rows)
                        both = at < min(len(rows), len(new_rows))
                        change = _change(rows[at], new_rows[at]) if both else "a row more or fewer"
                        count = f"{len(rows)} -> {len(new_rows)} in all"
                        lines.append(f"{where}.{scope}[{at}], the first that differs ({count}): {change}")
            else:
                lines.append(f"{where}: {_short(was)} -> {_short(now)}")
    return lines


def _flat_metrics(start: Mapping[str, Any]) -> dict[str, Any]:
    flat = {}
    for kind, table in start.items():
        for parent, group in table.get("values", {}).items():
            for key, value in group.items():
                flat[f"{kind}: {parent}/{key}" if parent else f"{kind}: {key}"] = value
        flat[f"{kind}: (steps)"] = table.get("steps")
    return flat


def _metric_changes(name: str, old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    lines = []
    old_starts, new_starts = old.get("starts", []), new.get("starts", [])
    if len(old_starts) != len(new_starts):
        lines.append(f"{name}: {len(old_starts)} start(s) -> {len(new_starts)}")
    for index, (before, after) in enumerate(zip(old_starts, new_starts)):
        was, now = _flat_metrics(before), _flat_metrics(after)
        for what, keys in (
            ("added", sorted(set(now) - set(was))),
            ("removed", sorted(set(was) - set(now))),
            ("changed", sorted(key for key in set(was) & set(now) if was[key] != now[key])),
        ):
            if keys:
                shown = ", ".join(keys[:8]) + (f" and {len(keys) - 8} more" if len(keys) > 8 else "")
                lines.append(f"starts[{index}] metrics {what}: {shown}")
    return lines
