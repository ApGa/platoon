"""The built-in task samplers in the trainer's own loop (Tinker faked), the trainer built as ``platoon train`` builds
it: from what ``AutoDataset.training`` lists and orders, its ``sampling`` included (RFC 0019, milestone M4).

M3's exit checks, again on the samplers' path: at ``max_staleness: 0`` every step trains exactly the rows of its batch
of 0.3.0's order; at ``k >= 1`` the groups started and not consumed never exceed ``(k + 1) × batch_size``; a run
interrupted twice consumes every position exactly once and rolls out each once, the starts after the first resuming
from the dispatcher's state of format 2.  And a resume whose pool changed draws each outstanding draw again from its
own task's row.  Each group returns one datum whose first token is its row's position in the order.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import pytest
from fakes.config import mixture_config
from fakes.runs import RecordingClient
from fakes.tinker_service import FakeTinkerService, make_service_trainer
from fakes.trainer import datum, draws_of, interrupt_at, make_trainer

from platoon.train import dataloader as dataloader_module
from platoon.train.auto import AutoDataset
from platoon.train.checkpoints.pruning import read_checkpoint_records
from platoon.train.dispatch.dispatcher import POSITION_COLUMN, SEQ_COLUMN
from platoon.train.task_records import CHANGED_KEY
from platoon.train.task_stats import read_events

HERE = __name__


def tasks(config: Any, split: str, *, prefix: str, count: int, nested: bool = False, reverse: bool = False) -> list:
    """``count`` tasks ``<prefix><index>``, listed as numbered, or in reverse; with ``nested``, nested mappings with
    different keys and a column that only some rows have."""

    listed: list[dict[str, Any]] = []
    for index in range(count):
        row: dict[str, Any] = {"task_id": f"{prefix}{index}"}
        if nested:
            row["meta"] = {"n": index} if index % 2 else {"m": str(index)}
            if index % 3 == 0:
                row["kind"] = "hard"
        listed.append(row)
    return listed[::-1] if reverse else listed


def marked(config: Any, split: str, *, count: int) -> list[dict[str, Any]]:
    return [{"task_id": f"m{index}", "_platoon_preserve_dataset_order": True} for index in range(count)]


def _entry(name: str | None, count: int, loader: str = "tasks", **fields: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"count": count}
    if loader == "tasks":
        kwargs["prefix"] = f"{name or 't'}-"
        for key in ("nested", "reverse"):
            if key in fields:
                kwargs[key] = fields.pop(key)
    entry: dict[str, Any] = {"dataset_loader": f"{HERE}:{loader}", "dataset_kwargs": kwargs, "rollout": "x:y", **fields}
    if name:
        entry["name"] = name
    return entry


def _handed(training: Any) -> dict[str, Any]:
    """What platoon train hands the trainer of ``training`` (``AutoDataset.training``'s), as its constructor's
    arguments: the first epoch, the order and the sampling."""

    assert training.sampling is not None
    return {"train_dataset": training.dataset, "train_order": training.order, "train_sampling": training.sampling}


def _row_json(row: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in (SEQ_COLUMN, POSITION_COLUMN)}, sort_keys=True)


class _Groups:
    """A training workflow: each group takes ``delay(row)`` seconds and returns one datum of its row's position;
    ``on_start(row, count)`` runs as each group starts.  ``rows`` holds the row each position ran, as JSON."""

    def __init__(self, llm, *, delay=lambda row: 0.01, on_start=None) -> None:
        self.llm = llm
        self.delay = delay
        self.on_start = on_start
        self.started: list[int] = []
        self.rows: dict[int, str] = {}

    async def arun_episode(self, data):
        self.started.append(data[POSITION_COLUMN])
        self.rows[data[POSITION_COLUMN]] = _row_json(data)
        if self.on_start is not None:
            self.on_start(data, len(self.started))
        version = self.llm.version
        await asyncio.sleep(self.delay(data))
        return [datum(data[POSITION_COLUMN], version=version)]


# name -> (train section beyond the run's, environments)
CONFIGS: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
    "weighted-mixture": (
        {"task_sampler": "weighted"},
        [_entry("a", 7, seed=3, weight=2), _entry("b", 5, seed=5, weight=1), _entry("c", 3, seed=8, weight=0.5)],
    ),
    "weighted-mixture-long": (  # five epochs and a batch of its order
        {"task_sampler": "weighted", "max_training_steps": 21},
        [_entry("a", 7, seed=3, weight=2), _entry("b", 5, seed=5, weight=1), _entry("c", 3, seed=8, weight=0.5)],
    ),
    "uniform-mixture": ({"task_sampler": "uniform"}, [_entry("a", 5, nested=True), _entry("b", 6, nested=True)]),
    "one-environment": ({}, [_entry(None, 9, nested=True)]),
    "marked-listing": ({}, [_entry(None, 9, loader="marked")]),
}


# --- k = 0: each step trains exactly the rows of its batch -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_at_k_0_every_step_trains_exactly_the_rows_of_its_batch_of_0_3_0_s_order(tmp_path, name):
    """Groups that take random times, two of a step's four at once: each step trains the four positions of its
    batch, and each position ran the row 0.3.0's dataloader draws there (over several epochs of a mixture's order, in
    the long one)."""

    train, environments = CONFIGS[name]
    config = mixture_config(
        tmp_path,
        {"batch_size": 4, "max_training_steps": 6, "max_staleness": 0, "max_concurrent_rollouts": 16, **train},
        environments,
    )
    steps = config.train.max_training_steps
    assert steps is not None
    client = RecordingClient()
    training = AutoDataset.training(config)
    trainer, _saves, logged = make_trainer(config, training_client=client, **_handed(training))
    pace = random.Random(7)
    workflow = _Groups(trainer.model_info.llm, delay=lambda row: pace.uniform(0.0, 0.03))

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 120))

    assert logged == list(range(1, steps + 1))
    assert [sorted(step) for step in client.steps] == [list(range(4 * step, 4 * step + 4)) for step in range(steps)]
    assert sorted(workflow.started) == list(range(4 * steps))
    dataloader = dataloader_module.PlatoonTinkerDataloader(training.dataset, 4, order=training.order)
    expected = [_row_json(row) for step in range(steps) for row in dataloader.get_batch(step)]
    assert [workflow.rows[position] for position in range(4 * steps)] == expected


# --- k >= 1: the pipeline never exceeds (k + 1) × batch_size ---------------------------------------------------------


@pytest.mark.parametrize("k", [1, 3])
def test_at_k_the_groups_started_and_not_consumed_never_exceed_k_plus_one_batches(tmp_path, k):
    train, environments = CONFIGS["weighted-mixture"]
    config = mixture_config(
        tmp_path,
        {"batch_size": 3, "max_training_steps": 10, "max_staleness": k, "max_concurrent_rollouts": 1024, **train},
        environments,
    )
    client = RecordingClient(delay=0.02)
    trainer, _saves, logged = make_trainer(config, training_client=client, **_handed(AutoDataset.training(config)))
    pace = random.Random(k)
    ahead: list[int] = []
    workflow = _Groups(
        trainer.model_info.llm,
        delay=lambda row: pace.uniform(0.0, 0.02),
        on_start=lambda row, count: ahead.append(count - 3 * len(logged)),
    )

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))

    assert logged == list(range(1, 11))
    assert max(ahead) <= (k + 1) * 3
    assert draws_of(trainer).dispatcher.peak == (k + 1) * 3
    assert sorted(position for step in client.steps for position in step) == list(range(30))


# --- exactly once across two interrupts ------------------------------------------------------------------------------


def _start(tmp_path: Path, name: str, **overrides: Any) -> tuple[Any, Any, list[list[int]]]:
    """One start of a run of config ``name`` on a fake Tinker service, built as platoon train builds it: ``(trainer,
    AutoDataset.training's result, the positions each step trains)``."""

    train, environments = CONFIGS[name]
    if "environments" in overrides:
        environments = overrides.pop("environments")
    config = mixture_config(
        tmp_path,
        {
            "batch_size": 2,
            "max_training_steps": 10,
            "max_staleness": 2,
            "max_concurrent_rollouts": 48,
            **train,
            **overrides,
        },
        environments,
        checkpoint={"every": 100},
    )
    steps: list[list[int]] = []

    async def recording(datums):
        steps.append([datum.model_input.to_ints()[0] for datum in datums])
        await asyncio.sleep(0.01)

    training = AutoDataset.training(config)
    trainer, _logged = make_service_trainer(
        config, FakeTinkerService(on_forward_backward=recording), **_handed(training)
    )
    return trainer, training, steps


def _run(trainer, workflow) -> None:
    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=workflow), 60))


@pytest.mark.parametrize("name", ["weighted-mixture", "uniform-mixture"])
def test_a_run_interrupted_twice_consumes_every_position_exactly_once(tmp_path, name):
    """Interrupted at its seventh group twice, with groups that had settled and that no step had trained, and a group
    running: every position of the order is rolled out to its end once and trained once over the three starts, the
    second and third resuming from a dispatcher's state of format 2, each sampler at the count it saved; and every
    group ran the row the order holds at its position."""

    trained: list[int] = []
    rows: dict[int, str] = {}
    for interrupt in (7, 7, None):
        trainer, training, steps = _start(tmp_path, name)
        workflow = _Groups(trainer.model_info.llm, on_start=interrupt_at(interrupt) if interrupt else None)
        _run(trainer, workflow)
        trained += [position for step in steps for position in step]
        rows.update(workflow.rows)

    run_dir = Path(trainer.run_log_path)
    records = read_checkpoint_records(run_dir)
    interrupted = [row for row in records if row["name"].endswith("_interrupted")]
    assert len(interrupted) == 2
    for row in interrupted:
        state = json.loads((run_dir / row["dispatch"]["file"]).read_bytes())
        assert state["format"] == 2 and state["kept"]["groups"]
        assert [draw[:2] for draw in state["drawn"]] == [draw[:2] for draw in state["outstanding"]]
        assert all(index is not None for _start, _seq, index, _by in state["drawn"])
        assert state["samplers"]["run"]["decisions"] == state["next_position"]
    assert sorted(trained) == list(range(20))
    events = list(read_events(run_dir))
    position = {(e["start"], e["seq"]): e["position"] for e in events if e["event"] == "draw"}
    assert sorted(position[(e["start"], e["seq"])] for e in events if e["event"] == "settle") == list(range(20))
    fates = [(position[(e["start"], e["seq"])], e["fate"]) for e in events if e["event"] == "fate"]
    assert sorted(p for p, fate in fates if fate == "consumed") == list(range(20)) and {f for _p, f in fates} == {
        "consumed"
    }
    draws = [e for e in events if e["event"] == "draw"]
    assert {e["reason"] for e in draws} == {"order", "reissue"} and all("sampler" in e for e in draws)
    dataloader = dataloader_module.PlatoonTinkerDataloader(training.dataset, 2, order=training.order)
    expected = [_row_json(row) for step in range(10) for row in dataloader.get_batch(step)]
    assert all(row == expected[at] for at, row in rows.items())


# --- a resume over a pool that changed -------------------------------------------------------------------------------


def test_a_resume_whose_pool_changed_draws_each_outstanding_draw_again_from_its_own_task_s_row(tmp_path):
    """The listing comes back reversed at the resume: the sampling inputs changed (warned of, recorded), and every
    saved pool index now holds another task, so each draw drawn again takes its own task's row, never the row at its
    old index."""

    environments = [_entry(None, 8, nested=True)]
    first, _training, _steps = _start(tmp_path, "one-environment", environments=environments)
    workflow = _Groups(
        first.model_info.llm,
        delay=lambda row: 0.3 if row[POSITION_COLUMN] % 3 == 0 else 0.01,
        on_start=interrupt_at(7),
    )
    _run(first, workflow)
    run_dir = Path(first.run_log_path)
    state = json.loads((run_dir / read_checkpoint_records(run_dir)[-1]["dispatch"]["file"]).read_bytes())
    saved = {(start, seq): (task, position) for start, seq, task, position in state["outstanding"]}
    moved = [_entry(None, 8, nested=True, reverse=True)]
    pool = [row["task_id"] for row in tasks(None, "train", prefix="t-", count=8, reverse=True)]
    assert state["drawn"] and all(pool[index] != saved[(start, seq)][0] for start, seq, index, _by in state["drawn"])

    (run_dir / "lock.json").write_text("{}\n")  # platoon train writes it before the trainer starts
    second, _training, _steps = _start(tmp_path, "one-environment", environments=moved)
    resumed = _Groups(second.model_info.llm)
    _run(second, resumed)

    events = list(read_events(run_dir))
    reissued = [e for e in events if e["event"] == "draw" and e.get("reason") == "reissue"]
    assert reissued
    for event in reissued:
        task, at = saved[tuple(event["reissues"])]
        assert event["task_id"] == task and event["position"] == at
        assert json.loads(resumed.rows[at])["task_id"] == task
    [changed] = json.loads((run_dir / "lock.json").read_text())[CHANGED_KEY]
    assert changed["changes"] == ["environments[default]: its training tasks (as many, 8, but others)"]


# --- a state whose rows or tokens do not read ------------------------------------------------------------------------


def _rewrite_state(run_dir: Path, key: str, value: Any) -> dict[str, Any]:
    """Set ``key`` of the dispatcher's state the newest checkpoint names to ``value``, its SHA-256 recorded anew, as a
    state written by another tool, or damaged, would read; the state as it was."""

    path = run_dir / "checkpoints.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    record = rows[-1]["dispatch"]
    state = json.loads((run_dir / record["file"]).read_bytes())
    content = (json.dumps({**state, key: value}, separators=(",", ":")) + "\n").encode()
    (run_dir / record["file"]).write_bytes(content)
    record["sha256"] = hashlib.sha256(content).hexdigest()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return state


@pytest.mark.parametrize(("key", "value"), [("drawn", [[1, 2, "3", None]]), ("tokens", {"default": "positions"})])
def test_a_state_whose_rows_or_tokens_do_not_read_still_draws_its_outstanding_draws_again(tmp_path, caplog, key, value):
    """A dispatcher's state whose rows of the outstanding draws (``drawn``) or tokens do not read loses that part
    alone, with a warning: the draws it left outstanding are still trained, those that were running drawn again
    (from the first row of their task, without the rows), and the order goes on at its position."""

    environments = [_entry(None, 8, nested=True)]
    first, _training, steps = _start(tmp_path, "one-environment", environments=environments)
    workflow = _Groups(
        first.model_info.llm,
        delay=lambda row: 0.3 if row[POSITION_COLUMN] % 3 == 0 else 0.01,
        on_start=interrupt_at(7),
    )
    _run(first, workflow)
    trained = [position for step in steps for position in step]
    run_dir = Path(first.run_log_path)
    state = _rewrite_state(run_dir, key, value)
    saved = {(start, seq): (task, position) for start, seq, task, position in state["outstanding"]}
    assert saved

    second, _training, steps = _start(tmp_path, "one-environment", environments=environments)
    with caplog.at_level(logging.WARNING):
        _run(second, _Groups(second.model_info.llm))
    trained += [position for step in steps for position in step]

    assert sorted(trained) == list(range(20))  # every position trained once, none lost
    events = list(read_events(run_dir))
    assert not [event for event in events if event["event"] == "fate" and event["fate"] == "lost"]
    reissued = [event for event in events if event["event"] == "draw" and event.get("reason") == "reissue"]
    assert reissued and all(saved[tuple(event["reissues"])][0] == event["task_id"] for event in reissued)
    said = [record.getMessage() for record in caplog.records]
    assert any("cannot read the outstanding draws' rows" in line or "cannot read the tokens" in line for line in said)
    assert not any("the draws it left outstanding are lost" in line for line in said)
