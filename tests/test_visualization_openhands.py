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
    _observation_text,
    _openhands_event_summary,
    _openhands_search_text,
    _openhands_step_summary,
    _step_action_events,
    _task_display_id,
    _task_display_metadata,
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

    assert label == ("collection: task:canvas-assessment-quality-audit · trajs:solver=0,verifier=0 · id:0338e49c")


def test_collection_label_keeps_named_task_and_numeric_index_distinct():
    label = _collection_display_label(
        "0338e49c-8572-4348-af60-1fb38d686bc0",
        "canvas-assessment-quality-audit",
        environment="toolathlon",
        task_name="canvas-assessment-quality-audit",
        task_index="114",
        split="train",
    )

    assert label == (
        "collection: toolathlon · task:canvas-assessment-quality-audit "
        "· index:train#114 · trajs:solver=0,verifier=0 · id:0338e49c"
    )


def test_task_display_id_uses_task_id_field():
    assert _task_display_id({"id": "canvas-assessment-quality-audit", "goal": "Call get_task"}) == (
        "canvas-assessment-quality-audit"
    )


def test_task_display_metadata_prefers_openreward_semantic_fields():
    task = {
        "id": ("openreward:v1:eyJlbnZpcm9ubWVudCI6InN3ZV9yZWJlbmNoIiwiaW5kZXgiOjI0NDQsInNwbGl0IjoidHJhaW4ifQ"),
        "misc": {
            "openreward_environment_label": "swe_rebench",
            "openreward_task_index": 2444,
            "openreward_task_split": "train",
        },
    }

    metadata = _task_display_metadata(task)

    assert metadata.environment == "swe_rebench"
    assert metadata.display_id == "2444"
    assert metadata.task_index == "2444"
    assert metadata.split == "train"
    assert _task_display_id(task) == "2444"


def test_encoded_openreward_id_is_a_backward_compatible_metadata_fallback():
    task = {"id": ("openreward:v1:eyJlbnZpcm9ubWVudCI6InRtYXgiLCJpbmRleCI6NDc4OSwic3BsaXQiOiJ0cmFpbiJ9")}

    metadata = _task_display_metadata(task)

    assert metadata.environment == "tmax"
    assert metadata.display_id == "4789"
    assert metadata.split == "train"


def test_malformed_encoded_openreward_id_falls_back_to_the_raw_id():
    raw_task_id = "openreward:v1:not-valid-base64"

    metadata = _task_display_metadata({"id": raw_task_id})

    assert metadata.environment is None
    assert metadata.display_id == raw_task_id


def test_collection_label_shows_env_task_and_ignores_recursive_child_ids():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "068cf326-e654-4035-8cc3-8872bfd08fed"
    root_id = "root"
    child_id = "child"
    verifier_id = "verifier"
    task_misc = {
        "openreward_environment_label": "swe_rebench",
        "openreward_task_index": 2444,
        "openreward_task_split": "train",
    }
    root_task_id = "openreward:v1:eyJlbnZpcm9ubWVudCI6InN3ZV9yZWJlbmNoIiwiaW5kZXgiOjI0NDQsInNwbGl0IjoidHJhaW4ifQ"

    for trajectory in (
        {"id": root_id, "reward": 0.0},
        {
            "id": child_id,
            "reward": 0.0,
            "parent_info": {"id": root_id, "fork_step": 2},
        },
        {
            "id": verifier_id,
            "reward": 0.0,
            "parent_info": {"id": child_id, "fork_step": 3},
        },
    ):
        tree.ingest(
            Event(
                type="trajectory_created",
                data={"collection_id": collection_id, "trajectory": trajectory},
            )
        )

    for trajectory_id, task_id, extra_misc in (
        (root_id, root_task_id, {}),
        (child_id, "422de74e-27de-4921-9471-97c12663ec40", {}),
        (
            verifier_id,
            "2603c91e-0fd7-4eba-b636-fac017d543c5",
            {"subagent_reward_verifier_task": True},
        ),
    ):
        tree.ingest(
            Event(
                type="trajectory_task_set",
                data={
                    "collection_id": collection_id,
                    "trajectory_id": trajectory_id,
                    "task": {
                        "id": task_id,
                        "goal": "Fix the pingsource service-account configuration.",
                        "misc": {**task_misc, **extra_misc},
                    },
                },
            )
        )

    group = tree.group_nodes[f"collection:{collection_id}"]
    label = str(group.label)

    assert label == ("collection: swe_rebench · task:train#2444 · trajs:solver=2,verifier=1 · id:068cf326")
    assert "422de74e" not in label
    assert "2603c91e" not in label
    assert group.data["payload"]["environment"] == "swe_rebench"
    assert group.data["payload"]["task_index"] == "2444"
    assert group.data["payload"]["raw_task_id"] == root_task_id
    assert group.data["payload"]["solver_trajectory_count"] == 2
    assert group.data["payload"]["verifier_trajectory_count"] == 1
    root_task_labels = [str(node.label) for node in tree.traj_nodes[root_id].children]
    assert any("env:swe_rebench" in label and "task:2444" in label for label in root_task_labels)
    assert "subtree:solver=2,verifier=1" in str(tree.traj_nodes[root_id].label)
    assert "subtree:solver=1,verifier=1" in str(tree.traj_nodes[child_id].label)
    assert "subtree:solver=0,verifier=1" in str(tree.traj_nodes[verifier_id].label)


def test_missing_collection_id_reuses_the_trajectory_collection():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    tree.ingest(
        Event(
            type="trajectory_created",
            data={"collection_id": collection_id, "trajectory": {"id": "root", "reward": 0.0}},
        )
    )
    tree.ingest(
        Event(
            type="trajectory_task_set",
            data={
                "trajectory_id": "root",
                "task": {"id": "openreward:v1:eyJlbnZpcm9ubWVudCI6InRtYXgiLCJpbmRleCI6NDc4OSwic3BsaXQiOiJ0cmFpbiJ9"},
            },
        )
    )

    assert set(tree.group_nodes) == {f"collection:{collection_id}"}
    assert tree.traj_nodes["root"].parent is tree.group_nodes[f"collection:{collection_id}"]
    assert "collection: tmax · task:train#4789" in str(tree.group_nodes[f"collection:{collection_id}"].label)


def test_late_collection_repairs_an_unlabeled_placeholder_group():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    tree.ingest(
        Event(
            type="trajectory_step_added",
            data={"trajectory_id": "root", "step_index": 0, "step": {"output": "working"}},
        )
    )
    assert tree.traj_nodes["root"].parent is tree.group_nodes["unlabeled"]

    tree.ingest(
        Event(
            type="trajectory_created",
            data={"collection_id": collection_id, "trajectory": {"id": "root", "reward": 0.0}},
        )
    )

    assert "unlabeled" not in tree.group_nodes
    assert tree.traj_nodes["root"].parent is tree.group_nodes[f"collection:{collection_id}"]
    assert "trajs:solver=1,verifier=0" in str(tree.group_nodes[f"collection:{collection_id}"].label)


def test_child_collection_repairs_an_unlabeled_parent_placeholder():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"

    tree.ingest(
        Event(
            type="trajectory_step_added",
            data={"trajectory_id": "root", "step_index": 0, "step": {"output": "working"}},
        )
    )
    tree.ingest(
        Event(
            type="trajectory_created",
            data={
                "collection_id": collection_id,
                "trajectory": {
                    "id": "child",
                    "reward": 0.0,
                    "parent_info": {"id": "root", "fork_step": 1},
                },
            },
        )
    )

    group = tree.group_nodes[f"collection:{collection_id}"]
    assert "unlabeled" not in tree.group_nodes
    assert tree.traj_nodes["root"].parent is group
    assert tree.traj_nodes["child"].parent is tree.traj_nodes["root"]
    assert "trajs:solver=2,verifier=0" in str(group.label)
    assert "subtree:solver=2,verifier=0" in str(tree.traj_nodes["root"].label)


def test_root_task_metadata_wins_when_task_events_precede_creation():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"
    root_task_id = "openreward:v1:eyJlbnZpcm9ubWVudCI6InN3ZV9yZWJlbmNoIiwiaW5kZXgiOjQyLCJzcGxpdCI6InRyYWluIn0"
    task_misc = {
        "openreward_environment_label": "swe_rebench",
        "openreward_task_index": 42,
        "openreward_task_split": "train",
    }

    for trajectory_id, task_id in (
        ("child", "opaque-child-task-id"),
        ("root", root_task_id),
    ):
        tree.ingest(
            Event(
                type="trajectory_task_set",
                data={
                    "collection_id": collection_id,
                    "trajectory_id": trajectory_id,
                    "task": {"id": task_id, "misc": task_misc},
                },
            )
        )
    tree.ingest(
        Event(
            type="trajectory_created",
            data={
                "collection_id": collection_id,
                "trajectory": {
                    "id": "child",
                    "reward": 0.0,
                    "parent_info": {"id": "root", "fork_step": 1},
                },
            },
        )
    )
    tree.ingest(
        Event(
            type="trajectory_created",
            data={"collection_id": collection_id, "trajectory": {"id": "root", "reward": 0.0}},
        )
    )

    group = tree.group_nodes[f"collection:{collection_id}"]
    assert group.data["payload"]["raw_task_id"] == root_task_id
    assert "task:train#42" in str(group.label)
    assert "subtree:solver=2,verifier=0" in str(tree.traj_nodes["root"].label)


def test_bridge_session_records_share_one_semantic_collection():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")

    for event_type, data in (
        ("session_started", {"env": "tmax", "split": "train"}),
        ("task_requested", {"prompt_chars": 123}),
        ("tool_call", {"tool_name": "get_task", "arguments": {}}),
        ("session_closing", {"finished": True}),
    ):
        tree.ingest(Event(type=event_type, data=data))

    assert set(tree.group_nodes) == {"collection:bridge:tmax"}
    group = tree.group_nodes["collection:bridge:tmax"]
    assert str(group.label) == "collection: bridge:tmax · trajs:solver=0,verifier=0"
    assert len(group.children) == 4
    assert tree.active_bridge_collection_id is None


def test_every_trajectory_node_shows_both_subtree_count_categories():
    tree = TrajectoryTree()
    tree.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
    collection_id = "collection-1"
    trajectories = (
        {"id": "root", "reward": 0.0},
        {
            "id": "verifier",
            "reward": 0.0,
            "parent_info": {"id": "root", "fork_step": 1},
            "misc": {"subagent_reward_verifier_task": True},
        },
        {
            "id": "verifier-child",
            "reward": 0.0,
            "parent_info": {"id": "verifier", "fork_step": 1},
        },
    )

    for trajectory in trajectories:
        tree.ingest(
            Event(
                type="trajectory_created",
                data={"collection_id": collection_id, "trajectory": trajectory},
            )
        )

    assert "subtree:solver=2,verifier=1" in str(tree.traj_nodes["root"].label)
    assert "subtree:solver=1,verifier=1" in str(tree.traj_nodes["verifier"].label)
    assert "subtree:solver=1,verifier=0" in str(tree.traj_nodes["verifier-child"].label)


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
    assert "subtree:solver=2,verifier=1" in str(tree.traj_nodes["root"].label)
    assert "subtree:solver=1,verifier=1" in str(tree.traj_nodes["child"].label)
    assert "subtree:solver=0,verifier=1" in str(tree.traj_nodes["verifier"].label)
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
    assert "subtree:solver=2,verifier=1" in str(tree.traj_nodes["root"].label)
    assert "subtree:solver=1,verifier=1" in str(tree.traj_nodes["child"].label)
    assert "subtree:solver=0,verifier=1" in str(tree.traj_nodes["verifier"].label)
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
    assert "subtree:solver=2,verifier=1" in str(tree.traj_nodes["root"].label)
    assert "subtree:solver=1,verifier=1" in str(tree.traj_nodes["child"].label)
    assert "subtree:solver=0,verifier=1" in str(tree.traj_nodes["verifier"].label)


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
    assert "subtree:solver=3,verifier=0" in str(tree.traj_nodes["root"].label)
    assert "subtree:solver=2,verifier=0" in str(tree.traj_nodes["child"].label)
    assert "subtree:solver=1,verifier=0" in str(tree.traj_nodes["grandchild"].label)


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


def test_agent_error_event_is_rendered_and_surfaced():
    observation = {
        "kind": "AgentErrorEvent",
        "tool_name": "call_tool",
        "tool_call_id": "call-invalid",
        "error": "Error validating tool 'call_tool': missing required argument 'name'",
    }
    step = {
        "action_events": {
            "action_events": [
                {
                    "kind": "ActionEvent",
                    "tool_name": "call_tool",
                    "tool_call_id": "call-invalid",
                }
            ]
        },
        "observation_events": [observation],
    }

    assert "missing required argument" in (_observation_text(observation) or "")
    assert "missing required argument" in (_observation_error_summary([observation]) or "")
    assert "missing required argument" in (_openhands_step_summary(step) or "")


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


def test_condensation_reasoning_is_searchable_and_rendered_separately():
    step = {
        "misc": {
            "synthetic_step_type": "openhands_condensation",
            "condensation_reasoning": ("Review the forgotten events and identify the durable implementation state."),
        },
        "action_events": None,
        "observation_events": [
            {
                "kind": "Condensation",
                "summary": (
                    "USER_CONTEXT: Fix the parser.\n"
                    "COMPLETED: Located the implementation.\n"
                    "PENDING: Apply and test the patch."
                ),
            }
        ],
    }
    panel = DetailsPanel(mode="openhands")
    rendered = panel._render_openhands_step(step)
    console = Console(record=True, width=120)

    console.print(rendered)
    output = console.export_text()

    assert "condensation reasoning" in output
    assert "Review the forgotten events" in output
    assert "Review the forgotten events" in _openhands_search_text(step)
    assert "Review the forgotten events" not in step["observation_events"][0]["summary"]


def test_python_execute_arguments_render_code_panel():
    panel = DetailsPanel(mode="openhands")
    rendered = panel._render_openhands_argument_panels("python_execute", {"code": "print('hello')", "timeout": 30})

    titles = [str(item.title) for item in rendered]
    assert "python_execute code" in titles
    assert "arguments" in titles
