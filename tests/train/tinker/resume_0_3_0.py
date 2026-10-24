"""Runs that platoon 0.3.0 interrupted, for the test that resumes them here (``test_resume_from_0_3_0.py``).

Each run lists its tasks and orders them as ``platoon train`` does (``AutoDataset.training``), trains on a fake Tinker
service with tinker-cookbook's own checkpoint saving (``tinker_service_fakes``), and is interrupted in the middle of a
step, so that its ``<step>_interrupted`` checkpoint holds draws that were running, draws that waited for a capped
label, and groups that had settled and that no step had trained.  Each group returns one datum whose first token is its
row's position in the order, so the datums a step trains say which positions it consumed.

It imports only what platoon 0.3.0 has.  Run as a script against a checkout of platoon 0.3.0, it writes the fixture
``tests/fixtures/resume-0.3.0/``: for each run, the files a resume reads (``checkpoints.jsonl`` and ``tasks/``), and
``rows.json``, the row 0.3.0 draws at each position of the run's order, which an uninterrupted run trains:

    PYTHONPATH=<platoon 0.3.0>:tests/train/tinker python tests/train/tinker/resume_0_3_0.py tests/fixtures/resume-0.3.0
"""

from __future__ import annotations

import asyncio
import json
import shutil
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import tinker
import torch
import yaml
from tinker_service_fakes import FakeTinkerService, make_pruning_trainer

HERE = __name__
# The column a dispatcher stamps a row's place in the order into.
POSITION = "_platoon_position"
STAMPS = ("_platoon_seq", POSITION)


def tasks(config: Any, split: str, *, prefix: str, count: int, nested: bool = False) -> list[dict[str, Any]]:
    """``count`` tasks ``<prefix><index>``; with ``nested``, a nested mapping whose keys differ from row to row and a
    column that only some rows have."""

    listed = []
    for index in range(count):
        row: dict[str, Any] = {"task_id": f"{prefix}{index}"}
        if nested:
            row["meta"] = {"n": index} if index % 2 else {"m": str(index)}
            if index % 3 == 0:
                row["kind"] = "hard"
        listed.append(row)
    return listed


def _entry(name: str | None, count: int, **fields: Any) -> dict[str, Any]:
    nested = fields.pop("nested", False)
    entry: dict[str, Any] = {
        "dataset_loader": f"{HERE}:tasks",
        "dataset_kwargs": {"prefix": f"{name or 't'}-", "count": count, "nested": nested},
        "rollout": "x:y",
        **fields,
    }
    if name:
        entry["name"] = name
    return entry


# name -> the run's train section, its environments, the group whose start interrupts it, and how long a group of a
# row takes (by the row's position and label).
RUNS: dict[str, dict[str, Any]] = {
    # A weighted mixture with seeds and numeric weights whose slow entry is capped at one group at a time.
    "weighted-capped": {
        "train": {
            "batch_size": 3,
            "max_training_steps": 6,
            "max_staleness": 2,
            "max_concurrent_rollouts": 64,
            "rollout_group_limits": {"slow": 1},
        },
        "environments": [_entry("quick", 6, seed=3, weight=2), _entry("slow", 4, seed=5, weight=1)],
        "interrupt": 11,
        "slow": {"labels": ["slow"], "every": None},
    },
    # One environment, drawn in the dataloader's shuffled passes.
    "single": {
        "train": {"batch_size": 2, "max_training_steps": 6, "max_staleness": 2, "max_concurrent_rollouts": 48},
        "environments": [_entry(None, 7)],
        "interrupt": 8,
        "slow": {"labels": [], "every": 3},
    },
    # Two entries under uniform, whose rows have nested mappings with different keys.
    "uniform-mixture": {
        "train": {
            "batch_size": 3,
            "max_training_steps": 6,
            "max_staleness": 2,
            "max_concurrent_rollouts": 64,
            "task_sampler": "uniform",
        },
        "environments": [_entry("a", 5, nested=True), _entry("b", 6, nested=True)],
        "interrupt": 11,
        "slow": {"labels": [], "every": 4},
    },
}
# The positions whose rows rows.json holds: more than a run draws.
ROWS = 64


def config(name: str, directory: Path) -> Any:
    """Run ``name``'s config, loaded as platoon train loads it (its file written to ``directory``, its run directory
    ``directory/logs/platoon_tinker/run``)."""

    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    run = RUNS[name]
    path = Path(directory) / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r", **run["train"]},
                "eval": {"every": 0},
                "watchdog": {"enabled": False},
                "checkpoint": {"every": 100},
                "log_path": str(Path(directory) / "logs"),
                "environments": run["environments"],
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def row_json(row: Any) -> str:
    """A drawn row as JSON, without the dispatcher's stamps, keys sorted."""

    return json.dumps({key: value for key, value in row.items() if key not in STAMPS}, sort_keys=True, default=str)


def order_rows(name: str, directory: Path) -> list[str]:
    """The row the run's order holds at each of its first ``ROWS`` positions, as the trainer's dataloader draws it."""

    from platoon.train.auto import AutoDataset
    from platoon.train.tinker.rl import PlatoonTinkerDataloader

    loaded = config(name, directory)
    training = AutoDataset.training(loaded)
    batch_size = loaded.train.batch_size
    loader = PlatoonTinkerDataloader(training.dataset, batch_size, order=training.order)
    return [row_json(row) for batch in range(ROWS // batch_size + 1) for row in loader.get_batch(batch)][:ROWS]


def _datum(position: int, version: int) -> tinker.Datum:
    """A datum whose first token is the position of the row its group ran."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, 1, 2]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


class Groups:
    """A training workflow: each group returns one datum of its row's position, after ``delay(row)`` seconds; the
    ``interrupt``-th group to start interrupts the run (SIGINT).  ``rows`` holds each group's row, by position."""

    def __init__(self, llm: Any, *, slow: dict[str, Any], interrupt: int | None = None) -> None:
        self.llm = llm
        self.slow = slow
        self.interrupt = interrupt
        self.started: list[int] = []
        self.rows: dict[int, str] = {}

    def delay(self, row: dict[str, Any]) -> float:
        every = self.slow["every"]
        if row.get("_platoon_environment") in self.slow["labels"] or (every and row[POSITION] % every == 0):
            return 0.3
        return 0.01

    async def arun_episode(self, data: dict[str, Any]) -> list[tinker.Datum]:
        self.started.append(data[POSITION])
        self.rows[data[POSITION]] = row_json(data)
        if len(self.started) == self.interrupt:
            signal.raise_signal(signal.SIGINT)
        version = self.llm.version
        await asyncio.sleep(self.delay(data))
        return [_datum(data[POSITION], version)]


def start(monkeypatch: Any, name: str, directory: Path, *, interrupt: int | None = None) -> tuple[Any, Groups, list]:
    """One start of run ``name`` in ``directory``: ``(trainer, workflow, steps)``, ``steps`` the positions each step
    trained.  The trainer draws as platoon train has it draw, from what ``AutoDataset.training`` gives (its
    ``sampling`` too, where there is one)."""

    from platoon.train.auto import AutoDataset

    loaded = config(name, directory)
    training = AutoDataset.training(loaded)
    trainer, _logged = make_pruning_trainer(monkeypatch, loaded, FakeTinkerService())
    trainer.train_dataset, trainer.train_order = training.dataset, training.order
    if getattr(training, "sampling", None) is not None:
        trainer.train_sampling = training.sampling
    steps: list[list[int]] = []

    async def create_training_client(service_client, resume_info):
        made = service_client.training_client()
        forward_backward = made.forward_backward_async

        async def recording(datums, loss_fn, loss_fn_config):
            steps.append([datum.model_input.to_ints()[0] for datum in datums])
            await asyncio.sleep(0.02)
            return await forward_backward(datums, loss_fn, loss_fn_config)

        made.forward_backward_async = recording
        return made

    trainer._create_training_client = create_training_client
    workflow = Groups(trainer._model_info.llm, slow=RUNS[name]["slow"], interrupt=interrupt)
    return trainer, workflow, steps


def run(trainer: Any, workflow: Groups) -> None:
    """Train until the run ends or its interrupt checkpoint is saved, within a minute; the process's signal handlers
    are put back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)}
    try:
        asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


# The files of a run directory that a resume reads, and the event log it appends to.
RESUME_FILES = ("checkpoints.jsonl", "tasks")


def main(out: str) -> None:
    target = Path(out)
    for name, spec in RUNS.items():
        with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as monkeypatch:
            trainer, workflow, _steps = start(monkeypatch, name, Path(directory), interrupt=spec["interrupt"])
            run(trainer, workflow)
            run_dir = Path(trainer.run_log_path)
            names = [json.loads(line)["name"] for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()]
            if not names[-1].endswith("_interrupted"):
                raise SystemExit(f"{name}: the run was not interrupted: {names}")
            destination = target / name
            shutil.rmtree(destination, ignore_errors=True)
            destination.mkdir(parents=True)
            for item in RESUME_FILES:
                source = run_dir / item
                if source.is_dir():
                    shutil.copytree(source, destination / item)
                else:
                    shutil.copy2(source, destination / item)
            rows = order_rows(name, Path(directory))
            (destination / "rows.json").write_text(json.dumps(rows, indent=0) + "\n")
            print(f"{name}: checkpoints {names}, {len(workflow.started)} groups started")


if __name__ == "__main__":
    main(sys.argv[1])
