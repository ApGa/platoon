from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from platoon.visualization.tui import (  # noqa: E402
    DetailsPanel,
    _collection_display_label,
    _observation_error_summary,
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
            "action_events": [
                {"kind": "ActionEvent", "tool_name": "python_execute", "tool_call_id": "call-python"}
            ]
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


def test_python_execute_arguments_render_code_panel():
    panel = DetailsPanel(mode="openhands")
    rendered = panel._render_openhands_argument_panels("python_execute", {"code": "print('hello')", "timeout": 30})

    titles = [str(item.title) for item in rendered]
    assert "python_execute code" in titles
    assert "arguments" in titles
