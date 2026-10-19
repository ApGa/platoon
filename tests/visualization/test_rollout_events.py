"""Rollout event files: what rollouts write after they finish, for ``python -m platoon.visualization.cli``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from platoon.task import Task
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.visualization.event_sinks import (
    emit_rollout_events,
    remove_temporary_events,
    rollout_events_path,
    trajectory_collection_dump_to_events,
)


def _collection() -> TrajectoryCollection:
    collection = TrajectoryCollection(id="collection-1")
    root = Trajectory(id="root", task=Task(goal="find the number", id="number_search.val.0"), reward=1.0)
    root.steps.append(TrajectoryStep(misc={"action_misc": {"completion_id": "call-1"}}))
    child = Trajectory(
        id="child",
        task=Task(goal="check the low half", id="number_search.val.0"),
        parent_info=ParentInfo(id="root", fork_step=1),
        finish_message="checked",
    )
    collection.trajectories = {"root": root, "child": child}
    return collection


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_events_cover_every_trajectory_in_viewer_order(tmp_path):
    path = emit_rollout_events(_collection(), tmp_path / "events.jsonl", process_id=7)

    records = _records(path)
    assert [(record["type"], record.get("trajectory_id") or record["trajectory"]["id"]) for record in records] == [
        ("trajectory_created", "root"),
        ("trajectory_task_set", "root"),
        ("trajectory_step_added", "root"),
        ("trajectory_finished", "root"),
        ("trajectory_created", "child"),
        ("trajectory_task_set", "child"),
        ("trajectory_finished", "child"),
    ]
    assert {record["collection_id"] for record in records} == {"collection-1"}
    assert {record["process_id"] for record in records} == {7}
    timestamps = [record["ts"] for record in records]
    assert timestamps == sorted(timestamps)
    step = records[2]
    assert step["step_index"] == 0 and step["reward"] == 1.0
    assert step["step"]["misc"]["action_misc"]["completion_id"] == "call-1"
    assert records[1]["task"]["goal"] == "find the number"
    assert records[4]["trajectory"]["parent_info"] == {"id": "root", "fork_step": 1}
    assert records[6]["finish_message"] == "checked"


def test_a_dump_and_its_collection_write_the_same_events(tmp_path):
    collection = _collection()
    from_object = _records(emit_rollout_events(collection, tmp_path / "a.jsonl", process_id=1))
    from_dump = _records(emit_rollout_events(collection.to_dict(), tmp_path / "b.jsonl", process_id=1))

    def strip(records):
        return [{key: value for key, value in record.items() if key != "ts"} for record in records]

    assert strip(from_object) == strip(from_dump)
    assert strip(from_dump) == [
        {**record, "process_id": 1}
        for record in strip(trajectory_collection_dump_to_events(json.loads(json.dumps(collection.to_dict()))))
    ]


def test_rewriting_replaces_the_file_and_leaves_no_temporary(tmp_path):
    path = tmp_path / "events" / "events_task_collection.jsonl"
    emit_rollout_events(_collection(), path)
    emit_rollout_events(_collection(), path)
    assert len(_records(path)) == 7
    assert os.listdir(path.parent) == [path.name]
    assert {record["process_id"] for record in _records(path)} == {os.getpid()}


class _Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text for this leaf")


def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    events = tmp_path / "events"
    collection = _collection()
    collection.trajectories["root"].misc["leaf"] = _Unprintable()
    with pytest.raises(RuntimeError, match="no text"):
        emit_rollout_events(collection, events / "unprintable.jsonl")
    assert not events.exists() or not any(events.iterdir())

    def full_disk(self, target):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "replace", full_disk)
    with pytest.raises(OSError, match="No space"):
        emit_rollout_events(_collection(), events / "full.jsonl")
    assert not any(events.iterdir())


def test_the_temporaries_of_a_killed_writer_are_removed_by_pid(tmp_path):
    events = tmp_path / "events"
    events.mkdir()
    for name in (".events_a_c.jsonl.41.tmp", ".events_b_c.jsonl.41.tmp", ".events_a_c.jsonl.42.tmp", "events_a.jsonl"):
        (events / name).write_text("{}\n")

    assert remove_temporary_events(tmp_path, 41) == 2
    assert sorted(path.name for path in events.iterdir()) == [".events_a_c.jsonl.42.tmp", "events_a.jsonl"]
    assert remove_temporary_events(tmp_path / "missing", 41) == 0


def test_event_paths_are_safe_file_names_under_the_rollout_dir(tmp_path):
    path = rollout_events_path(tmp_path, "mixture::tasks/a b:1", "c/1")
    assert path.parent == tmp_path / "events"
    assert path.name == "events_mixture__tasks_a_b_1_c_1.jsonl"


def _verified_collection() -> TrajectoryCollection:
    """A root that delegated to a child, a verifier of that child, and steps with annotated misc."""

    collection = _collection()
    root, child = collection.trajectories["root"], collection.trajectories["child"]
    root.steps.append(TrajectoryStep(misc={"code": "guess(5)", "thought": "too low?", "output": "Too low"}))
    root.misc = {"outcome": {"status": "scored", "rewards": {"reward": 1.0}, "reason": None}, "agent": {"role": "root"}}
    child.steps.append(TrajectoryStep(misc={"action_misc": {"completion_id": "call-2"}}))
    verifier = Trajectory(
        id="verifier",
        task=Task(goal="check the child", id="number_search.val.0"),
        parent_info=ParentInfo(id="child", fork_step=1),
        reward=0.0,
        misc={"platoon_subagent_verifies_trajectory_id": "child", "agent": {"role": "verifier"}},
    )
    verifier.steps.append(TrajectoryStep(misc={"thought": "it checked"}))
    collection.trajectories["verifier"] = verifier
    return collection


def test_created_records_carry_no_steps_task_or_misc_beyond_the_viewers_keys(tmp_path):
    collection = _verified_collection()
    records = _records(emit_rollout_events(collection, tmp_path / "events.jsonl", process_id=1))

    created = {
        record["trajectory"]["id"]: record["trajectory"] for record in records if record["type"] == "trajectory_created"
    }
    assert set(created) == {"root", "child", "verifier"}
    for trajectory in created.values():
        assert trajectory["steps"] == [] and trajectory["task"] is None
    assert created["root"]["misc"] == {} and created["root"]["reward"] == 1.0
    assert created["verifier"]["misc"] == {"platoon_subagent_verifies_trajectory_id": "child"}
    assert created["child"]["parent_info"] == {"id": "root", "fork_step": 1}
    assert created["child"]["finish_message"] == "checked"
    finished = {record["trajectory_id"]: record for record in records if record["type"] == "trajectory_finished"}
    assert finished["root"]["misc"] == collection.trajectories["root"].misc  # the whole misc, once


def _legacy_events(collection: TrajectoryCollection, process_id: int) -> list[dict]:
    """The events in their first format: each created record embedded its whole trajectory."""

    dump = json.loads(json.dumps(collection.to_dict()))
    records = trajectory_collection_dump_to_events(dump)
    for record in records:
        record["process_id"] = process_id
        if record["type"] == "trajectory_created":
            record["trajectory"] = dump["trajectories"][record["trajectory"]["id"]]
    return records


def _viewer_tree(path) -> list:
    """The tree a headless TrajectoryViewer builds from an event file: every node's label, type and payload,
    except the steps and task a trajectory node's payload held only in the legacy files."""

    import asyncio

    from platoon.visualization.tui import TrajectoryViewer

    async def build() -> list:
        app = TrajectoryViewer(jsonl_paths=[path], replay_delay=0.0)
        async with app.run_test() as pilot:
            while app._polling_task is None or not app._polling_task.done():
                await pilot.pause(0.05)

            def walk(node) -> list:
                data = node.data if isinstance(node.data, dict) else {}
                payload = data.get("payload")
                if data.get("type") == "trajectory" and isinstance(payload, dict):
                    payload = {key: value for key, value in payload.items() if key not in ("steps", "task")}
                return [str(node.label), data.get("type"), payload, [walk(child) for child in node.children]]

            return walk(app.tree_widget.tree_widget.root)

    return asyncio.run(build())


def test_the_viewer_builds_the_same_tree_from_legacy_and_current_event_files(tmp_path):
    collection = _verified_collection()
    current = emit_rollout_events(collection, tmp_path / "current" / "events.jsonl", process_id=3)
    legacy = tmp_path / "legacy" / "events.jsonl"
    legacy.parent.mkdir()
    legacy.write_text("".join(json.dumps(record) + "\n" for record in _legacy_events(collection, 3)))

    assert len(current.read_text()) < len(legacy.read_text())
    tree = _viewer_tree(current)
    assert tree == _viewer_tree(legacy)
    text = json.dumps(tree)
    assert "verifier:verifier" in text and "step 1: code: guess(5); thought: too low?; output" in text
