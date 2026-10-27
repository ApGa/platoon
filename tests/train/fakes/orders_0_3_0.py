"""The training configs whose orders platoon 0.3.0 drew, for the tests that hold the built-in task samplers to them
(``test_builtin_samplers.py``), and the rows 0.3.0's dataloader draws for each.

It imports only what platoon 0.3.0 has, so that the same configs run on it: run as a script against a checkout of
platoon 0.3.0, it writes the digests of the rows its pure-Python orders drew (``weighted``, and ``auto`` for a
mixture), ``tests/fixtures/orders-0.3.0.json``:

    PYTHONPATH=<platoon 0.3.0>:tests/train python tests/train/fakes/orders_0_3_0.py tests/fixtures/orders-0.3.0.json
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

HERE = __name__
# Each config draws three epochs and one batch more.
EPOCHS = 3


def rows(config: Any, split: str, *, prefix: str, count: int, extra: bool = False) -> list[dict[str, Any]]:
    """``count`` tasks ``<prefix><index>``; with ``extra``, columns that only some rows have, and nested mappings with
    different keys."""

    listed = []
    for index in range(count):
        row: dict[str, Any] = {"task_id": f"{prefix}{index}"}
        if extra:
            row["kind"] = "hard" if index % 3 == 0 else "easy"
            row["meta"] = {"n": index} if index % 2 else {"m": str(index)}
        listed.append(row)
    return listed


def repeated(config: Any, split: str, *, count: int) -> list[dict[str, Any]]:
    """A listing that repeats task ids, each row with content of its own: 0.3.0 draws each row as a position."""

    return [{"task_id": f"d{index % 3}", "variant": index} for index in range(count)]


def marked(config: Any, split: str, *, count: int) -> list[dict[str, Any]]:
    """A listing whose rows are all marked to keep their order, as a custom loader may give it."""

    return [
        {"task_id": f"m{index}", "kind": "hard" if index % 2 else "easy", "_platoon_preserve_dataset_order": True}
        for index in range(count)
    ]


def env_column(config: Any, split: str, *, count: int, values: list[str | None]) -> list[dict[str, Any]]:
    """A listing whose rows name an environment of their own in platoon's environment column (``values`` in turn, None
    leaving it out), as the rows of a mixture's run exported with ``AutoDataset.from_config(config, "train")`` do: a
    single environment draws them all as its own, whatever they say."""

    listed = []
    for index in range(count):
        row: dict[str, Any] = {"task_id": f"e{index}"}
        if values[index % len(values)] is not None:
            row["_platoon_environment"] = values[index % len(values)]
        listed.append(row)
    return listed


def entry(
    name: str | None,
    count: int,
    *,
    seed: int | None = None,
    weight: float | None = None,
    extra: bool = False,
    loader: str = "rows",
    values: list[str | None] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"count": count}
    if loader == "rows":
        kwargs.update(prefix=f"{name or 't'}-", extra=extra)
    if values is not None:
        kwargs["values"] = values
    data: dict[str, Any] = {"dataset_loader": f"{HERE}:{loader}", "dataset_kwargs": kwargs, "rollout": "x:y"}
    if name:
        data["name"] = name
    if seed is not None:
        data["seed"] = seed
    if weight is not None:
        data["weight"] = weight
    return data


# name -> (train section, environments)
CONFIGS: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
    "single-auto": ({"batch_size": 4}, [entry(None, 10)]),
    "single-auto-B3-extra": ({"batch_size": 3}, [entry(None, 10, extra=True)]),
    "single-auto-marked": ({"batch_size": 4}, [entry(None, 10, loader="marked")]),
    "single-auto-repeated-ids": ({"batch_size": 2}, [entry(None, 6, loader="repeated")]),
    "single-weighted": ({"batch_size": 4, "task_sampler": "weighted"}, [entry(None, 10, seed=7)]),
    "single-weighted-repeated-ids": (
        {"batch_size": 2, "task_sampler": "weighted"},
        [entry(None, 6, loader="repeated")],
    ),
    "single-weighted-env-column": (
        {"batch_size": 3, "task_sampler": "weighted"},
        [entry(None, 7, seed=4, loader="env_column", values=["x"])],
    ),
    "single-weighted-env-column-some": (
        {"batch_size": 3, "task_sampler": "weighted"},
        [entry(None, 7, seed=4, loader="env_column", values=[None, "x"])],
    ),
    "single-uniform": ({"batch_size": 4, "task_sampler": "uniform"}, [entry(None, 10)]),
    "mix-auto-tasks": ({"batch_size": 4}, [entry("a", 7, seed=0), entry("b", 5, seed=5)]),
    "mix-auto-extra": ({"batch_size": 4}, [entry("a", 5, extra=True), entry("b", 6)]),
    "mix-auto-3-B5": ({"batch_size": 5}, [entry("a", 3, seed=1), entry("b", 9, seed=2), entry("c", 4, seed=3)]),
    "mix-auto-empty-entry": ({"batch_size": 3}, [entry("a", 5, seed=4), entry("b", 0), entry("c", 4, seed=6)]),
    "mix-weighted-numeric": (
        {"batch_size": 4, "task_sampler": "weighted"},
        [entry("a", 7, seed=3, weight=2), entry("b", 5, seed=5, weight=1)],
    ),
    "mix-weighted-fractional": (
        {"batch_size": 6, "task_sampler": "weighted"},
        [entry("a", 4, seed=9, weight=0.5), entry("b", 11, seed=1, weight=1 / 3), entry("c", 2, weight=0.1)],
    ),
    "mix-uniform": ({"batch_size": 4, "task_sampler": "uniform"}, [entry("a", 7), entry("b", 5)]),
    "mix-uniform-extra": ({"batch_size": 4, "task_sampler": "uniform"}, [entry("a", 5, extra=True), entry("b", 6)]),
}


def config(name: str, directory: Path) -> Any:
    """Config ``name``, loaded as platoon train loads it (its file written to ``directory``)."""

    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    train, environments = CONFIGS[name]
    path = Path(directory) / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", "max_training_steps": 12, **train},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "log_path": str(Path(directory) / "logs"),
                "environments": environments,
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def digest(row: Any) -> str:
    """A row's digest: of its JSON, keys sorted."""

    return hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]


def drawn_rows(training: Any, batch_size: int) -> list[dict[str, Any]]:
    """The rows 0.3.0's dataloader draws from ``AutoDataset.training``'s result, position by position: three epochs
    and a batch."""

    from platoon.train.dataloader import PlatoonTinkerDataloader

    loader = PlatoonTinkerDataloader(training.dataset, batch_size, order=training.order)
    return [row for batch in range(EPOCHS * len(loader) + 1) for row in loader.get_batch(batch)]


def pure_python(name: str) -> bool:
    """Whether config ``name`` draws a pure-Python order (``weighted``, or ``auto`` for a mixture), whose rows do not
    depend on the installed ``datasets``: the orders of a single environment's ``auto`` and of ``uniform`` are its
    shuffles."""

    train, environments = CONFIGS[name]
    sampler = train.get("task_sampler", "auto")
    return sampler == "weighted" or (sampler == "auto" and len(environments) > 1)


def main(out: str) -> None:
    from platoon.train.auto import AutoDataset

    golden: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as directory:
        for name in CONFIGS:
            if not pure_python(name):
                continue
            loaded = config(name, Path(directory))
            training = AutoDataset.training(loaded)
            drawn = drawn_rows(training, loaded.train.batch_size)
            golden[name] = {
                "order": type(training.order).__name__,
                "rows": [[str(row["task_id"]), digest(row)] for row in drawn],
            }
    Path(out).write_text(golden_text(golden))


def golden_text(golden: dict[str, Any]) -> str:
    """The digests as JSON, one row a line."""

    parts = []
    for name in sorted(golden):
        rows = ",\n".join(json.dumps(row) for row in golden[name]["rows"])
        parts.append(f' {json.dumps(name)}: {{"order": {json.dumps(golden[name]["order"])}, "rows": [\n{rows}\n ]}}')
    return "{\n" + ",\n".join(parts) + "\n}\n"


if __name__ == "__main__":
    main(sys.argv[1])
