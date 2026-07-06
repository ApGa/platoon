from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from platoon.visualization.tui import (  # noqa: E402
    DetailsPanel,
    Event,
    PlayPauseFriendlyTree,
    TrajectoryTree,
    _collection_display_label,
    _observation_error_summary,
    _openhands_event_summary,
    _openhands_search_text,
    _openhands_step_summary,
    _step_action_events,
    _task_display_id,
    _tool_call_display,
)


def _tool_action(name: str, arguments: dict | None = None) -> dict:
    return {
        "kind": "ActionEvent",
        "tool_name": "call_tool",
        "tool_call_id": f"call-{name}",
        "tool_call": {
            "name": "call_tool",
            "arguments": json.dumps({"name": name, "arguments": arguments or {}}),
        },
    }


def _tool_observation(tool_name: str, payload: object, *, tool_call_id: str | None = None) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "kind": "ObservationEvent",
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "observation": {
            "content": [
                {"text": f"[Tool '{tool_name}' executed.]"},
                {"text": json.dumps({"text": text})},
            ]
        },
    }


def test_nested_action_events_are_normalized():
    step = {"action_events": {"action_events": [{"kind": "ActionEvent", "tool_name": "get_task"}]}}

    assert _step_action_events(step) == [{"kind": "ActionEvent", "tool_name": "get_task"}]


def test_collection_label_prefers_task_id_when_available():
    label = _collection_display_label("0338e49c-8572-4348-af60-1fb38d686bc0", "canvas-assessment-quality-audit")

    assert label == "collection:canvas-assessment-quality-audit · id:0338e..."


def test_task_display_id_uses_task_id_field():
    assert _task_display_id({"id": "canvas-assessment-quality-audit", "goal": "Call get_task"}) == (
        "canvas-assessment-quality-audit"
    )


def test_trajectory_tree_nests_child_trajectories_by_parent_info():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    root = {"id": "root", "reward": 0.0}
    child = {
        "id": "child",
        "reward": 1.0,
        "parent_info": {"id": "root", "fork_step": 2},
    }
    verifier = {
        "id": "verifier",
        "reward": 0.0,
        "parent_info": {"id": "root", "fork_step": 3},
        "misc": {
            "subagent_reward_verifier_task": True,
            "subagent_reward_verifies_trajectory_id": "child",
            "exclude_from_training": True,
        },
    }

    for trajectory in (root, child, verifier):
        tree.ingest(
            Event(
                type="trajectory_created",
                data={"collection_id": collection_id, "trajectory": trajectory},
            )
        )

    assert tree.traj_nodes["root"].parent is tree.group_nodes[f"collection:{collection_id}"]
    assert tree.traj_nodes["child"].parent is tree.traj_nodes["root"]
    assert tree.traj_nodes["verifier"].parent is tree.traj_nodes["child"]
    assert "verifier:verifier" in str(tree.traj_nodes["verifier"].label)
    assert getattr(tree.traj_nodes["verifier"].label, "style", None) == "dim"

    for event_type, extra in (
        ("trajectory_task_set", {"task": {"id": "verifier-task", "goal": "verify child"}}),
        ("trajectory_step_added", {"step_index": 0, "step": {"misc": {}}, "reward": 0.0}),
        ("trajectory_finished", {"reward": 0.0, "finish_message": "{}", "misc": verifier["misc"]}),
    ):
        tree.ingest(
            Event(
                type=event_type,
                data={"collection_id": collection_id, "trajectory_id": "verifier", **extra},
            )
        )

    assert tree.traj_nodes["verifier"].parent is tree.traj_nodes["child"]
    assert "verifier:verifier" in str(tree.traj_nodes["verifier"].label)
    assert getattr(tree.traj_nodes["verifier"].label, "style", None) == "dim"


def test_trajectory_tree_repairs_verifier_parent_from_task_misc():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    root = {"id": "root", "reward": 0.0}
    child = {
        "id": "child",
        "reward": 1.0,
        "parent_info": {"id": "root", "fork_step": 2},
    }
    verifier = {
        "id": "verifier",
        "reward": 0.0,
        "parent_info": {"id": "root", "fork_step": 3},
    }
    verifier_misc = {
        "subagent_reward_verifier_task": True,
        "subagent_reward_verifies_trajectory_id": "child",
        "exclude_from_training": True,
    }

    for trajectory in (root, child, verifier):
        tree.ingest(
            Event(
                type="trajectory_created",
                data={"collection_id": collection_id, "trajectory": trajectory},
            )
        )

    assert tree.traj_nodes["verifier"].parent is tree.traj_nodes["root"]

    tree.ingest(
        Event(
            type="trajectory_task_set",
            data={
                "collection_id": collection_id,
                "trajectory_id": "verifier",
                "task": {"id": "verifier-task", "goal": "verify child", "misc": verifier_misc},
            },
        )
    )

    assert tree.traj_nodes["verifier"].parent is tree.traj_nodes["child"]


def test_trajectory_tree_updates_reward_from_late_finished_event():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    tree.ingest(
        Event(
            type="trajectory_created",
            data={"collection_id": collection_id, "trajectory": {"id": "child", "reward": 0.0}},
        )
    )
    tree.ingest(
        Event(
            type="trajectory_finished",
            data={"collection_id": collection_id, "trajectory_id": "child", "reward": 0.0},
        )
    )
    tree.ingest(
        Event(
            type="trajectory_finished",
            data={
                "collection_id": collection_id,
                "trajectory_id": "child",
                "reward": 0.8,
                "misc": {"subagent_reward_judgment": {"score": 0.8}},
            },
        )
    )

    assert tree.traj_rewards["child"] == 0.8
    assert "reward:0.800" in str(tree.traj_nodes["child"].label)
    payload = tree.traj_nodes["child"].data["payload"]
    assert payload["reward"] == 0.8
    assert payload["misc"]["subagent_reward_judgment"]["score"] == 0.8


def test_trajectory_tree_repairs_out_of_order_deep_parent_chain():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    root = {"id": "root", "reward": 0.0}
    child = {
        "id": "child",
        "reward": 0.0,
        "parent_info": {"id": "root", "fork_step": 1},
    }
    grandchild = {
        "id": "grandchild",
        "reward": 1.0,
        "parent_info": {"id": "child", "fork_step": 2},
    }

    for trajectory in (grandchild, child, root):
        tree.ingest(
            Event(
                type="trajectory_created",
                data={"collection_id": collection_id, "trajectory": trajectory},
            )
        )

    assert tree.traj_nodes["root"].parent is tree.group_nodes[f"collection:{collection_id}"]
    assert tree.traj_nodes["child"].parent is tree.traj_nodes["root"]
    assert tree.traj_nodes["grandchild"].parent is tree.traj_nodes["child"]


def test_call_tool_display_uses_catalog_tool_name():
    tool_name, arguments = _tool_call_display(_tool_action("excel.create_workbook", {"path": "plan.xlsx"}))

    assert tool_name == "excel.create_workbook"
    assert arguments == {"path": "plan.xlsx"}


def test_tool_call_display_keeps_direct_action_data_arguments():
    event = {
        "kind": "ActionEvent",
        "tool_name": "python_execute",
        "action": {"data": {"code": "print('hello')"}, "kind": "MCPToolAction"},
    }

    tool_name, arguments = _tool_call_display(event)

    assert tool_name == "python_execute"
    assert arguments == {"code": "print('hello')"}


def test_setup_step_summary_collapses_system_prompt():
    step = {
        "action_events": None,
        "observation_events": [
            {"kind": "SystemPromptEvent", "system_prompt": {"text": "very long prompt"}},
            {"kind": "MessageEvent", "llm_message": {"content": [{"text": "Call get_task first."}]}},
        ],
    }

    assert _openhands_step_summary(step) == "setup: system prompt + user message"


def test_get_task_observation_summary_extracts_task_name():
    step = {
        "action_events": {"action_events": [{"kind": "ActionEvent", "tool_name": "get_task"}]},
        "observation_events": [
            _tool_observation(
                "get_task",
                {"task_name": "course-enrollment-analytics-dashboard", "prompt": "Do work"},
            )
        ],
    }

    assert _openhands_step_summary(step) == "get_task -> course-enrollment-analytics-dashboard"


def test_parallel_tool_calls_are_counted():
    step = {
        "action_events": {
            "action_events": [
                _tool_action("emails.send_email", {"to": "a@example.com"}),
                _tool_action("emails.send_email", {"to": "b@example.com"}),
                _tool_action("emails.send_email", {"to": "c@example.com"}),
            ]
        },
        "observation_events": [],
    }

    assert _openhands_step_summary(step) == "tools: emails.send_email x3"


def test_observation_errors_are_surfaced():
    observation = _tool_observation("python_execute", "Traceback: RuntimeError('boom')", tool_call_id="call-python")
    step = {
        "action_events": {
            "action_events": [{"kind": "ActionEvent", "tool_name": "python_execute", "tool_call_id": "call-python"}]
        },
        "observation_events": [observation],
    }

    assert "Traceback" in (_observation_error_summary([observation]) or "")
    assert "->" in (_openhands_step_summary(step) or "")


def test_claim_done_summary_uses_final_reward_payload():
    step = {
        "misc": {
            "reward_misc": {
                "openreward/final_payload": {
                    "finished": True,
                    "reward": 0.0,
                    "text": "FAIL\n\n=== Check 1 ===",
                }
            }
        },
        "action_events": {"action_events": [{"kind": "ActionEvent", "tool_name": "claim_done"}]},
        "observation_events": [],
    }

    assert _openhands_step_summary(step) == "claim_done -> FAIL (reward=0.0)"


def test_openhands_search_text_includes_tool_and_observation_payload():
    step = {
        "action_events": {"action_events": [_tool_action("woocommerce.woo_reports_top_sellers")]},
        "observation_events": [_tool_observation("call_tool", {"products": ["Widget A"]})],
    }

    search_text = _openhands_search_text(step)

    assert "woocommerce.woo_reports_top_sellers" in search_text
    assert "Widget A" in search_text


def test_condensation_summary_is_surfaced():
    event = {
        "kind": "Condensation",
        "summary": "USER_CONTEXT: KPI report work has been summarized.",
        "forgotten_event_ids": ["event-1", "event-2"],
    }
    step = {"action_events": None, "observation_events": [event]}

    assert _openhands_event_summary(event) == "condensation: USER_CONTEXT: KPI report work has been summarized."
    assert _openhands_step_summary(step) == "condensation: USER_CONTEXT: KPI report work has been summarized."


def test_condensation_summary_is_included_with_action_summary():
    step = {
        "action_events": {
            "action_events": [
                _tool_action("snowflake.write_query"),
                _tool_action("snowflake.write_query"),
            ]
        },
        "observation_events": [
            {
                "kind": "Condensation",
                "summary": "Warehouse context and KPI calculations were condensed.",
            }
        ],
    }

    summary = _openhands_step_summary(step)

    assert summary is not None
    assert "tools: snowflake.write_query x2" in summary
    assert "condensation: Warehouse context and KPI calculations were condensed." in summary


def test_openhands_search_text_includes_condensation_summary():
    step = {
        "action_events": None,
        "observation_events": [
            {
                "kind": "Condensation",
                "summary": "The condensed state includes revenue and support KPI calculations.",
            }
        ],
    }

    search_text = _openhands_search_text(step)

    assert "revenue and support KPI calculations" in search_text


def test_condensation_detail_panel_renders_summary():
    panel = DetailsPanel(mode="openhands")
    rendered = panel._render_openhands_condensation_summary(
        {
            "kind": "Condensation",
            "summary": "## KPI State\n\n- Revenue: **Near**\n- Support: Met",
            "forgotten_event_ids": ["event-1"],
        },
        "condensation summary",
    )
    console = Console(record=True, width=120)

    console.print(rendered)
    output = console.export_text()

    assert isinstance(rendered, Panel)
    assert isinstance(rendered.renderable, Group)
    assert any(isinstance(renderable, Markdown) for renderable in rendered.renderable.renderables)
    assert "condensation summary" in output
    assert "forgotten events: 1" in output
    assert "KPI State" in output
    assert "Revenue" in output
    assert "Near" in output


def test_python_execute_arguments_render_code_panel():
    panel = DetailsPanel(mode="openhands")
    rendered = panel._render_openhands_argument_panels("python_execute", {"code": "print('hello')", "timeout": 30})

    titles = [str(item.title) for item in rendered]
    assert "python_execute code" in titles
    assert "arguments" in titles
