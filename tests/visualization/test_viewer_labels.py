"""The viewer shows task ids, goals and step text as written: tree labels are literal text, not Rich markup or
emoji codes (``task:train:41`` is not the ``:train:`` emoji, and ``[/tmp/out]`` is not a closing tag)."""

from __future__ import annotations

import asyncio

import pytest
from rich.text import Text

from platoon.task import Task
from platoon.trajectory import Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.visualization.event_sinks import emit_rollout_events
from platoon.visualization.tui import PlayPauseFriendlyTree, _collection_display_label

GOAL = "Move list[int] results to [/tmp/out] and report :fire: counts as [bold]text[/bold]"


@pytest.mark.parametrize("task_id", ["train:41", "repo[/fix]-1", "swe/[bold]x", "eval:7"])
def test_a_collection_label_shows_its_task_id_verbatim(task_id):
    label = _collection_display_label("c0ffee", task_id)

    shown = PlayPauseFriendlyTree("collections").process_label(label)

    assert shown.plain == label
    assert f"task:{task_id}" in shown.plain


def test_a_styled_label_keeps_its_style():
    label = Text("traj:root · reward 1.0", style="dim")

    shown = PlayPauseFriendlyTree("collections").process_label(label)

    assert shown.plain == label.plain and shown.style == "dim"


def _viewer(path, explore) -> list[str]:
    """Build a headless viewer from an event file, run ``explore(app, pilot, nodes)`` on it, and return the tree's
    labels."""

    from platoon.visualization.tui import TrajectoryViewer

    async def build() -> list[str]:
        app = TrajectoryViewer(jsonl_paths=[path], replay_delay=0.0)
        async with app.run_test() as pilot:
            while app._polling_task is None or not app._polling_task.done():
                await pilot.pause(0.05)
            nodes: list = []

            def walk(node) -> None:
                nodes.append(node)
                for child in node.children:
                    walk(child)

            walk(app.tree_widget.tree_widget.root)
            await explore(app, pilot, nodes)
            assert app.is_running  # nothing the viewer rendered crashed it
            return [str(node.label) for node in nodes]

    return asyncio.run(build())


def _events(tmp_path):
    collection = TrajectoryCollection(id="rollout-1")
    root = Trajectory(id="root", task=Task(goal=GOAL, id="train:41"), reward=1.0)
    root.steps.append(TrajectoryStep(misc={"action_misc": {"completion_id": "call-1"}, "output": "ls [/tmp/out]"}))
    collection.trajectories = {"root": root}
    return emit_rollout_events(collection, tmp_path / "events.jsonl", process_id=1)


async def _look(app, pilot, nodes) -> None:
    return None


def test_the_viewer_shows_task_ids_and_goals_literally(tmp_path):
    labels = _viewer(_events(tmp_path), _look)

    text = "\n".join(labels)
    assert "collection: task:train:41" in text  # not the :train: emoji
    task_labels = [label for label in labels if label.startswith("task:train:41 · ")]
    assert task_labels == [f"task:train:41 · {GOAL}"]  # the task node exists, with its goal as written


def _content(widget):
    return getattr(widget, "content", None) or widget.renderable


def test_details_and_search_results_show_the_text_literally(tmp_path):
    seen: dict[str, object] = {}

    async def select_and_search(app, pilot, nodes) -> None:
        (task_node,) = [node for node in nodes if str(node.label).startswith("task:train:41 · ")]
        app.tree_widget.tree_widget.select_node(task_node)
        await pilot.pause(0.1)
        seen["details title"] = _content(app.details_panel).title.plain
        app.perform_search("tmp/out")  # matches the task's goal and the step's output
        await pilot.pause(0.1)
        seen["results"] = sorted({str(_content(widget)) for widget in app.search_panel.result_widgets})
        app.perform_search("[/tmp")
        await pilot.pause(0.1)
        seen["results label"] = str(_content(app.search_panel.results_label))

    _viewer(_events(tmp_path), select_and_search)

    assert seen["details title"] == f"task:train:41 · {GOAL} · task"
    results = "\n".join(seen["results"])
    assert "[/tmp/out]" in results and "ls [/tmp/out]" in results
    assert seen["results label"].startswith("Search results for '[/tmp'")
