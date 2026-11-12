"""The viewer renders any harness's tool use, the ``tool_results`` and ``inputs`` a step's misc holds: in ``auto``
mode a step is labelled by its tools, their counts and what ended other than ``completed``, and its details show each
result, with the arguments its step's message gave the call, and each input, each with its ref."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from textual.widgets.tree import TreeNode

from platoon.task import Task
from platoon.trajectory import Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.visualization.event_sinks import emit_rollout_events
from platoon.visualization.tui import DetailsPanel, Event, PlayPauseFriendlyTree, TrajectoryTree

MESSAGE = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "call-1", "type": "function", "function": {"name": "unlock", "arguments": '{"digits": "1"}'}},
        {"id": "call-2", "type": "function", "function": {"name": "unlock", "arguments": '{"digits": "2"}'}},
        {"id": "call-3", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
    ],
}
RESULTS = [
    {"name": "unlock", "tool_call_id": "call-2", "outcome": "completed", "output": "still locked", "ref": "turn/w1@5"},
    {"name": "unlock", "tool_call_id": "call-1", "outcome": "completed", "output": "open", "ref": "turn/w1@6"},
    {"name": "bash", "tool_call_id": "call-3", "outcome": "failed", "output": None, "ref": "turn/w1@7"},
]
INPUTS = [
    {"ref": "agent/w1@1", "from": "agent:lead", "text": "Open the box. Its code is 1."},
    {"ref": "inbox/w1@3", "from": "agent:lead", "text": "[bold]try[/bold] the second code"},
]


def _step(**misc: object) -> TrajectoryStep:
    return TrajectoryStep(misc={"action_misc": {"completion_id": "c1"}, "message": MESSAGE, **misc})


def _viewed(tmp_path: Path, *steps: TrajectoryStep, mode: str = "auto") -> list[TreeNode]:
    """The step nodes a viewer in ``mode`` makes of ``steps``, from their rollout's event file."""

    collection = TrajectoryCollection(id="rollout-1")
    root = Trajectory(id="root", task=Task(goal="open the box", id="test:1"))
    root.steps.extend(steps)
    collection.trajectories = {"root": root}
    path = emit_rollout_events(collection, tmp_path / f"events-{mode}.jsonl", process_id=1)
    tree = TrajectoryTree(mode=mode)
    tree.tree_widget = PlayPauseFriendlyTree("collections")
    for line in path.read_text().splitlines():
        record = json.loads(line)
        tree.ingest(Event(type=record["type"], data=record))
    return list(tree.traj_steps_nodes["root"].children)


def _labels(tmp_path: Path, *steps: TrajectoryStep, mode: str = "auto") -> list[str]:
    return [str(node.label) for node in _viewed(tmp_path, *steps, mode=mode)]


def _details(tmp_path: Path, step: TrajectoryStep, mode: str = "auto") -> str:
    (node,) = _viewed(tmp_path, step, mode=mode)
    console = Console(record=True, width=160)
    console.print(DetailsPanel(mode=mode)._render_data(node.data["payload"]))
    return console.export_text()


def test_a_tool_use_step_is_labelled_by_its_tools_their_counts_and_what_did_not_complete(tmp_path):
    labels = _labels(tmp_path, _step(tool_results=RESULTS, inputs=INPUTS), _step(tool_results=[], inputs=INPUTS[:1]))

    assert labels == ["step 0: tools: unlock x2, bash; bash failed; inputs: 2", "step 1: inputs: 1"]


def test_a_tool_use_step_shows_each_result_with_its_arguments_and_each_input_with_its_ref(tmp_path):
    shown = _details(tmp_path, _step(tool_results=RESULTS, inputs=INPUTS))

    assert "tools: unlock x2, bash; bash failed; inputs: 2" in shown
    assert shown.index("1. unlock · completed · turn/w1@5") < shown.index("2. unlock · completed · turn/w1@6")
    assert (
        shown.index('"digits": "2"') < shown.index("still locked") < shown.index('"digits": "1"') < shown.index("open")
    )
    assert "3. bash · failed · turn/w1@7" in shown and "<unknown>" in shown  # its output is unknown, not empty
    assert "1. agent/w1@1 · from agent:lead" in shown and "Open the box. Its code is 1." in shown
    assert "[bold]try[/bold] the second code" in shown  # inputs are data, not Rich markup


def test_a_tool_result_with_nothing_known_but_its_end_still_renders(tmp_path):
    labels = _labels(tmp_path, _step(tool_results=[{"outcome": None}, {"outcome": "rejected"}]))

    assert labels == ["step 0: tools: tool x2; tool rejected"]
    assert "1. tool" in _details(tmp_path, _step(tool_results=[{}]))


def test_tool_use_renders_only_in_auto_mode(tmp_path):
    step = _step(tool_results=RESULTS, output="the code ran")

    assert _labels(tmp_path, step, mode="codeact") == ["step 0: output"]
    assert "tool results" not in _details(tmp_path, step, mode="codeact")


def test_openhands_events_render_as_openhands_whatever_tool_use_a_step_holds(tmp_path):
    action = {"kind": "ActionEvent", "tool_name": "search", "tool_call_id": "call-9"}
    step = _step(tool_results=RESULTS, action_events=[action], observation_events=[])

    assert _labels(tmp_path, step) == ["step 0: search"]
    assert "actions and observations" in _details(tmp_path, step)
