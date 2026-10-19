"""``task_table``: tasks with no tools and no grader, for self-grading harnesses."""

from __future__ import annotations

import asyncio
import json
import logging
import threading

import pytest

from platoon.envs import EnvironmentService, EnvSession, task_table
from platoon.envs import table as table_module
from platoon.envs.service import get_service
from platoon.task import Task

pytestmark = pytest.mark.asyncio


async def test_rows_become_tasks_with_splits_from_the_rows():
    table = task_table(
        [
            {"id": "a", "goal": "Say hi.", "split": "train", "answer": "hi", "max_steps": 3},
            {"task_id": "b", "prompt": "Say bye.", "split": "test"},
            "Count to three.",
        ]
    )

    assert isinstance(table, EnvironmentService)
    assert await table.list_splits() == ["train", "test"]
    assert await table.list_tasks("train") == ["a", "task-2"]
    assert await table.list_tasks("test") == ["b"]
    task = await table.get_task("a")
    assert (task.goal, task.max_steps, task.misc) == ("Say hi.", 3, {"answer": "hi"})
    task.misc["answer"] = "changed"
    assert (await table.get_task("a")).misc == {"answer": "hi"}  # callers get copies
    with pytest.raises(ValueError, match="no split"):
        await table.list_tasks("validation")
    with pytest.raises(ValueError, match="no task"):
        await table.get_task("zzz")


async def test_explicit_splits_and_duplicate_ids():
    table = task_table(["one", "two"], splits={"train": ["task-1"], "test": ["task-0"]})
    assert await table.list_tasks("train") == ["task-1"]
    with pytest.raises(ValueError, match="unknown task ids"):
        task_table(["one"], splits={"train": ["missing"]})
    with pytest.raises(ValueError, match="two rows"):
        task_table([{"id": "x", "goal": "a"}, {"id": "x", "goal": "b"}])


async def test_sessions_have_the_goal_no_tools_and_no_grade():
    table = task_table([{"id": "a", "goal": "Say hi."}])
    session = await table.open(Task(id="a"), rollout_id="r")

    assert isinstance(session, EnvSession) and session.sandbox is None
    assert session.instruction == "Say hi." and await session.tools() == []
    assert (await session.call_tool("bash", {"command": "ls"})).is_error
    outcome = await session.grade()
    assert (outcome.status, outcome.reason, outcome.trainable) == ("invalid", "self_grade", False)
    child = await session.fork("fresh", task=Task(id="c", goal="Help."))
    assert child.instruction == "Help."
    await child.close()
    await session.close()


async def test_the_table_service_builds_from_config_rows_or_a_jsonl_path(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"t{index}", "goal": f"Task {index}."}) for index in range(2)) + "\n")

    from_path = get_service("table", path=str(path))
    assert await from_path.list_tasks("train") == ["t0", "t1"]
    from_rows = get_service("table", rows=[{"id": "x", "goal": "X."}], name="inline")
    assert from_rows.name == "inline" and (await from_rows.get_task("x")).goal == "X."
    with pytest.raises(ValueError, match="rows or a JSONL path"):
        get_service("table")


async def test_concurrent_rollouts_share_one_parse_of_a_path_table_off_the_loop(tmp_path, monkeypatch):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "".join(
            json.dumps({"id": f"t{index}", "goal": f"Task {index}.", "meta": {"n": index}}) + "\n"
            for index in range(1_000)
        )
    )
    parse = table_module._parse_file
    parses = []
    # Off the loop, the loop keeps running while the table is parsed: its watcher sees the parse in progress, and the
    # parse waits for that before it ends. Parsing on the loop, or blocking the loop until the parse is done, keeps
    # the watcher away, and the parse waits in vain. A handshake, not a bound on how long a tick may stall, which the
    # load of other processes on the node stretches (a parallel run saw 0.66 s with the parse off the loop).
    parsing, seen = threading.Event(), threading.Event()
    loop_ran_during_the_parse = []

    def parse_while_the_loop_runs(real):
        parses.append(real)
        parsing.set()
        rows = parse(real)
        loop_ran_during_the_parse.append(seen.wait(timeout=30))
        return rows

    monkeypatch.setattr(table_module, "_parse_file", parse_while_the_loop_runs)

    async def watch_the_loop():
        while True:
            await asyncio.sleep(0.005)
            if parsing.is_set():
                seen.set()

    async def rollout(index: int) -> Task:
        service = get_service("table", path=str(path))  # as harness_rollout builds it, once per rollout
        task = await service.get_task(f"t{index}")
        session = await service.open(task, rollout_id=f"r{index}")
        await session.close()
        return task

    watcher = asyncio.create_task(watch_the_loop())
    tasks = await asyncio.gather(*(rollout(index) for index in range(64)))
    watcher.cancel()

    assert len(parses) == 1
    assert loop_ran_during_the_parse == [True]
    assert [task.goal for task in tasks[:2]] == ["Task 0.", "Task 1."]
    tasks[0].misc["meta"]["n"] = "changed"
    assert (await get_service("table", path=str(path)).get_task("t0")).misc["meta"] == {"n": 0}  # deep copies

    path.write_text(json.dumps({"id": "t0", "goal": "Changed."}) + "\n")  # a changed file is read again
    assert (await get_service("table", path=str(path)).get_task("t0")).goal == "Changed."
    assert len(parses) == 2


async def test_many_inline_rows_are_reported_once(monkeypatch, caplog):
    monkeypatch.setattr(table_module, "_warned_inline_rows", False)
    rows = [f"Task {index}." for index in range(table_module.INLINE_ROWS_WARNING + 1)]
    with caplog.at_level(logging.WARNING, logger="platoon.envs.table"):
        task_table(rows[:-1])
        task_table(rows)
        task_table(rows)
    warnings = [record.getMessage() for record in caplog.records if "inline rows" in record.getMessage()]
    assert len(warnings) == 1 and "path" in warnings[0]
