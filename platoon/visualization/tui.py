from __future__ import annotations

import asyncio
import json
import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

try:
    from textual.containers import HorizontalScroll  # type: ignore
except Exception:
    # Fallback for older Textual versions without HorizontalScroll
    HorizontalScroll = VerticalScroll  # type: ignore
from rich.console import Group
from rich.json import JSON as RichJSON
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.binding import Binding
from textual.events import MouseDown, MouseMove, MouseUp

BRIDGE_EVENT_TYPES = {
    "session_started",
    "task_requested",
    "tool_call",
    "tool_result",
    "max_tool_calls_exceeded",
    "session_closing",
}

VISUALIZATION_MODES = {"auto", "codeact", "openhands"}

MOUSE_CAPTURE_ENABLE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1006h"
MOUSE_CAPTURE_DISABLE = "\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1005l\x1b[?1006l\x1b[?1015l"


def _shorten_text(value: Any, max_chars: int = 240) -> str:
    text = str(value).replace("\n", " ").strip()
    text = " ".join(text.split())
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _normalize_visualization_mode(mode: str | None) -> str:
    if mode in VISUALIZATION_MODES:
        return str(mode)
    return "auto"


def _task_display_id(task: Any) -> str | None:
    if not isinstance(task, dict):
        return None
    for key in ("id", "task_id", "name"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _collection_display_label(collection_id: Any, task_id: str | None = None) -> str:
    if task_id:
        short_id = _shorten_text(collection_id, 8) if collection_id else "unlabeled"
        return f"collection:{task_id} · id:{short_id}"
    return f"collection:{collection_id}" if collection_id else "unlabeled"


def _is_openhands_step_payload(data: Any) -> bool:
    return isinstance(data, dict) and ("action_events" in data or "observation_events" in data)


def _should_render_openhands(mode: str, data: Any) -> bool:
    if mode == "openhands":
        return _is_openhands_step_payload(data)
    return mode == "auto" and _is_openhands_step_payload(data)


def _as_dict_list(value: Any, nested_key: str | None = None) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if nested_key and isinstance(value.get(nested_key), list):
            return [item for item in value[nested_key] if isinstance(item, dict)]
        if isinstance(value.get("action_events"), list):
            return [item for item in value["action_events"] if isinstance(item, dict)]
        if isinstance(value.get("observation_events"), list):
            return [item for item in value["observation_events"] if isinstance(item, dict)]
        if "kind" in value:
            return [value]
    return []


def _step_action_events(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _as_dict_list(step.get("action_events"), "action_events")


def _step_observation_events(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _as_dict_list(step.get("observation_events"), "observation_events")


def _text_block_value(block: Any) -> str | None:
    if isinstance(block, dict):
        text = block.get("text")
        if isinstance(text, str):
            return text
        if isinstance(block.get("content"), str):
            return block["content"]
    elif isinstance(block, str):
        return block
    return None


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _format_inline_value(value: Any, max_chars: int = 80) -> str:
    parsed = _parse_json_string(value)
    if isinstance(parsed, str):
        return _shorten_text(parsed)
    if isinstance(parsed, (int, float, bool)) or parsed is None:
        return str(parsed)
    if isinstance(parsed, list):
        inner = ", ".join(_format_inline_value(item) for item in parsed)
        return f"[{inner}]"
    if isinstance(parsed, dict):
        return _format_args_inline(parsed)
    return _shorten_text(parsed)


def _format_args_inline(value: Any, max_items: int = 4, max_chars: int = 180) -> str:
    parsed = _parse_json_string(value)
    if parsed in (None, "", {}):
        return ""
    if not isinstance(parsed, dict):
        return _format_inline_value(parsed)

    parts: List[str] = []
    for index, (key, item) in enumerate(parsed.items()):
        if index >= max_items:
            parts.append("...")
            break
        parts.append(f"{key}={_format_inline_value(item, 56)}")
    return _shorten_text(", ".join(parts), max_chars)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _tool_call_display(event: Dict[str, Any]) -> tuple[str | None, Any]:
    action = event.get("action")
    action_data = action.get("data") if isinstance(action, dict) else None
    if isinstance(action_data, dict) and action_data:
        tool_name = action_data.get("name") or event.get("tool_name")
        arguments = action_data.get("arguments") if "arguments" in action_data else action_data
    else:
        tool_name = event.get("tool_name")
        arguments = None
        tool_call = event.get("tool_call")
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name") or tool_name
            arguments = tool_call.get("arguments")

    arguments = _parse_json_string(arguments)
    if tool_name == "call_tool" and isinstance(arguments, dict):
        catalog_name = arguments.get("name")
        if isinstance(catalog_name, str) and catalog_name:
            return catalog_name, arguments.get("arguments") or {}
    return tool_name if isinstance(tool_name, str) else None, arguments


def _observation_payload(event: Dict[str, Any]) -> Any:
    observation = event.get("observation")
    if not isinstance(observation, dict):
        return None
    content = observation.get("content")
    if isinstance(content, str):
        return _parse_json_string(content)
    if not isinstance(content, list):
        return None

    payloads: List[Any] = []
    for block in content:
        text = _text_block_value(block)
        if not text or text.startswith("[Tool "):
            continue
        parsed = _parse_json_string(text)
        if isinstance(parsed, dict) and "text" in parsed:
            payloads.append(_parse_json_string(parsed.get("text")))
        elif isinstance(parsed, dict) and "blocks" in parsed:
            payloads.append(parsed.get("blocks"))
        else:
            payloads.append(parsed)
    if not payloads:
        return None
    if len(payloads) == 1:
        return payloads[0]
    return payloads


def _observation_preview(event: Dict[str, Any]) -> str | None:
    payload = _observation_payload(event)
    if payload is None:
        return None
    if isinstance(payload, str):
        return _shorten_text(payload)
    return _shorten_text(_pretty_json(payload))


def _condensation_summary_text(event: Dict[str, Any]) -> str | None:
    if event.get("kind") != "Condensation":
        return None
    summary = event.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return None


def _message_text(event: Dict[str, Any]) -> str | None:
    message = event.get("llm_message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_text_block_value(block) for block in content]
        text = " ".join(part for part in parts if part)
        return text or None
    return None


def _thought_text(event: Dict[str, Any]) -> str | None:
    thought = event.get("thought")
    if isinstance(thought, str):
        return thought
    if isinstance(thought, list):
        parts = [_text_block_value(block) for block in thought]
        text = " ".join(part for part in parts if part)
        return text or None
    return None


def _reasoning_text(event: Dict[str, Any]) -> str | None:
    reasoning = event.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    return None


def _observation_text(event: Dict[str, Any]) -> str | None:
    condensation_summary = _condensation_summary_text(event)
    if condensation_summary:
        return condensation_summary

    payload = _observation_payload(event)
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    return _pretty_json(payload)


def _find_nested_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_nested_value(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_nested_value(item, key)
            if found is not None:
                return found
    return None


def _event_has_error(event: Dict[str, Any]) -> bool:
    payload = _observation_payload(event)
    preview = _observation_preview(event) or ""
    lowered = preview.lower()
    if any(token in lowered for token in ("error", "traceback", "exception", "failed")):
        return True
    if isinstance(payload, dict):
        for key in ("error", "exception", "traceback"):
            value = _find_nested_value(payload, key)
            if value:
                return True
    return False


def _observation_error_summary(events: List[Dict[str, Any]]) -> str | None:
    for event in events:
        if _event_has_error(event):
            return _openhands_event_summary(event)
    return None


def _first_condensation_summary(events: List[Dict[str, Any]]) -> str | None:
    for event in events:
        summary = _condensation_summary_text(event)
        if summary:
            return f"condensation: {_shorten_text(summary)}"
    return None


def _task_discovery_summary(events: List[Dict[str, Any]]) -> str | None:
    for event in events:
        raw_tool_name = event.get("tool_name")
        payload = _observation_payload(event)
        task_name = _find_nested_value(payload, "task_name")
        if not task_name and raw_tool_name == "get_task":
            task_name = _find_nested_value(payload, "name")
        prompt = _find_nested_value(payload, "prompt")
        if raw_tool_name == "get_task" or task_name or prompt:
            if isinstance(task_name, str) and task_name:
                return f"get_task -> {task_name}"
            if isinstance(prompt, str) and prompt:
                return f"get_task -> {_shorten_text(prompt, 160)}"
            return "get_task"
    return None


def _event_tool_names(events: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for event in events:
        tool_name, _ = _tool_call_display(event)
        if tool_name:
            names.append(tool_name)
    return names


def _tool_batch_summary(events: List[Dict[str, Any]]) -> str | None:
    if not events:
        return None
    names = _event_tool_names(events)
    if not names:
        return _openhands_event_summary(events[0])
    if len(names) == 1:
        return _openhands_event_summary(events[0])

    counts: Dict[str, int] = {}
    order: List[str] = []
    for name in names:
        if name not in counts:
            order.append(name)
            counts[name] = 0
        counts[name] += 1
    parts = [f"{name} x{counts[name]}" if counts[name] > 1 else name for name in order]
    return "tools: " + _shorten_text(", ".join(parts), 180)


def _reward_misc(step: Dict[str, Any]) -> Dict[str, Any]:
    misc = step.get("misc")
    if not isinstance(misc, dict):
        return {}
    reward_misc = misc.get("reward_misc")
    return reward_misc if isinstance(reward_misc, dict) else {}


def _final_payload(step: Dict[str, Any]) -> Any:
    reward_misc = _reward_misc(step)
    payload = reward_misc.get("openreward/final_payload")
    return _parse_json_string(payload)


def _final_evaluation_summary(step: Dict[str, Any]) -> str | None:
    payload = _final_payload(step)
    if not payload:
        return None

    status: str | None = None
    reward = None
    text = None
    if isinstance(payload, dict):
        reward = payload.get("reward")
        text = payload.get("text")
        if isinstance(text, str):
            first = text.strip().splitlines()[0].strip()
            if first:
                status = first
        if status is None and payload.get("finished") is True:
            status = "finished"
    elif isinstance(payload, str):
        text = payload
        first = payload.strip().splitlines()[0].strip()
        if first:
            status = first

    parts = ["claim_done"]
    if status:
        parts.append(str(status))
    if reward is not None:
        parts.append(f"reward={reward}")
    return " -> ".join(parts[:2]) + (f" ({parts[2]})" if len(parts) > 2 else "")


def _openhands_event_summary(event: Dict[str, Any]) -> str:
    kind = event.get("kind")
    condensation_summary = _condensation_summary_text(event)
    if condensation_summary:
        return f"condensation: {_shorten_text(condensation_summary)}"

    if kind == "ObservationEvent":
        observation = _observation_preview(event)
        if observation:
            raw_tool_name = event.get("tool_name")
            if isinstance(raw_tool_name, str) and raw_tool_name:
                return f"{raw_tool_name}: {observation}"
            return f"observation: {observation}"

    tool_name, arguments = _tool_call_display(event)
    if isinstance(tool_name, str) and tool_name:
        args_text = _format_args_inline(arguments)
        return f"{tool_name}: {args_text}" if args_text else tool_name

    observation = _observation_preview(event)
    if observation:
        return f"observation: {observation}"

    thought = _thought_text(event)
    if thought:
        return f"thought: {_shorten_text(thought)}"

    reasoning = _reasoning_text(event)
    if reasoning:
        return f"reasoning: {_shorten_text(reasoning)}"

    message = _message_text(event)
    if message:
        return f"message: {_shorten_text(message)}"

    if isinstance(kind, str) and kind:
        return kind
    return _shorten_text(event)


def _openhands_step_summary(step: Dict[str, Any]) -> str | None:
    action_events = _step_action_events(step)
    observation_events = _step_observation_events(step)
    non_system_observations = [event for event in observation_events if event.get("kind") != "SystemPromptEvent"]
    observation_summaries = [_openhands_event_summary(event) for event in non_system_observations or observation_events]

    final_summary = _final_evaluation_summary(step)
    if final_summary:
        return final_summary

    if not action_events:
        setup_messages = [
            event
            for event in observation_events
            if event.get("kind") == "MessageEvent" and (_message_text(event) or "").strip()
        ]
        has_system = any(event.get("kind") == "SystemPromptEvent" for event in observation_events)
        if has_system and setup_messages:
            return "setup: system prompt + user message"
        if has_system:
            return "setup: system prompt"

    task_summary = _task_discovery_summary(non_system_observations or observation_events)
    if task_summary:
        return task_summary

    parts: List[str] = []
    action_summary = _tool_batch_summary(action_events)
    if action_summary:
        parts.append(action_summary)

    # Surface tool failures directly in the tree label. Successful observation text
    # is usually redundant with the action summary and can be very large.
    error_summary = _observation_error_summary(non_system_observations or observation_events)
    if error_summary:
        parts.append(f"-> {error_summary}")

    condensation_summary = _first_condensation_summary(non_system_observations or observation_events)
    if condensation_summary:
        parts.append(condensation_summary)

    if not parts and observation_summaries:
        parts.append(observation_summaries[0])
    if not parts:
        return None
    return " ".join(parts)


def _openhands_search_text(step: Dict[str, Any]) -> str:
    parts: List[str] = []
    summary = _openhands_step_summary(step)
    if summary:
        parts.append(summary)
    for event in _step_action_events(step):
        parts.append(_openhands_event_summary(event))
        tool_name, arguments = _tool_call_display(event)
        if tool_name:
            parts.append(tool_name)
        if arguments:
            parts.append(_format_args_inline(arguments, max_items=8, max_chars=400))
        for text in (_thought_text(event), _reasoning_text(event), _message_text(event)):
            if text:
                parts.append(text)
    for event in _step_observation_events(step):
        parts.append(_openhands_event_summary(event))
        text = _observation_text(event)
        if text:
            parts.append(text)
        message = _message_text(event)
        if message:
            parts.append(message)
    return " ".join(part for part in parts if part)


def _bridge_collection_id(record: Dict[str, Any]) -> str:
    task_name = record.get("task_name")
    if isinstance(task_name, str) and task_name:
        return f"bridge:{task_name}"
    env = record.get("env")
    if isinstance(env, str) and env:
        return f"bridge:{env}"
    return "bridge"


def _bridge_record_summary(record: Dict[str, Any]) -> str:
    record_type = record.get("type", "bridge")
    if record_type == "tool_call":
        args_text = _format_args_inline(record.get("arguments"))
        suffix = f": {args_text}" if args_text else ""
        return f"call {record.get('tool_name', 'tool')}{suffix}"
    if record_type == "tool_result":
        result = record.get("result")
        text = result.get("text") if isinstance(result, dict) else result
        return f"result {record.get('tool_name', 'tool')}: {_shorten_text(text)}"
    if record_type == "task_requested":
        return f"task requested · {record.get('prompt_chars', '?')} prompt chars"
    if record_type == "session_started":
        return f"session · {record.get('env', '?')} · {record.get('split', '?')}"
    if record_type == "session_closing":
        return f"session closing · finished:{record.get('finished')}"
    return _shorten_text(record)


def _record_sort_key(record: Dict[str, Any]) -> float:
    value = record.get("ts", record.get("time", 0.0))
    try:
        return float(value)
    except Exception:
        return 0.0


class PlayPauseFriendlyTree(Tree):
    """A Tree widget that doesn't capture space key, allowing it to bubble up for play/pause."""

    BINDINGS = [
        # Override space binding to do nothing, letting it bubble up to app level
        # The default Tree uses space for toggle_node (expand/collapse)
        Binding("space", "noop", show=False),
        # Keep enter for toggling nodes instead
        Binding("enter", "toggle_node", "Toggle", show=False),
    ]

    def action_noop(self) -> None:
        """Do nothing - let the key bubble up to the app."""
        pass


class ClickableResult(Static):
    """A clickable search result item with enhanced styling."""

    def __init__(
        self, text: str, index: int, search_panel: "SearchPanel", result_type: str = "unknown", context: str = ""
    ) -> None:
        # Create rich content with icons and colors
        content = self._create_rich_content(text, result_type, context)
        super().__init__(content)
        self.index = index
        self.search_panel = search_panel
        self.result_type = result_type
        self.is_highlighted = False

        self._apply_normal_styling()

    def _apply_normal_styling(self) -> None:
        """Apply normal (non-highlighted) styling."""
        try:
            bg_color, border_color = self._get_type_colors(self.result_type)
            # Completely reset all styling properties
            self.styles.padding = (0, 1)
            self.styles.background = bg_color
            self.styles.color = None  # Reset to default
            self.styles.text_style = None  # Reset to default
            self.styles.border = None  # Reset to default
            self.styles.margin = (0, 0, 0, 0)
        except Exception:
            pass

    def _apply_highlighted_styling(self) -> None:
        """Apply highlighted styling."""
        try:
            self.styles.padding = (0, 1)
            self.styles.background = "blue"
            self.styles.color = "white"
            self.styles.text_style = "bold"
            self.styles.border = ("thick", "bright_white")
            self.styles.margin = (0, 0, 0, 0)
        except Exception:
            pass

    def set_highlighted(self, highlighted: bool) -> None:
        """Set whether this result is highlighted."""
        if self.is_highlighted != highlighted:
            self.is_highlighted = highlighted
            if highlighted:
                self._apply_highlighted_styling()
            else:
                self._apply_normal_styling()
            # Force a refresh to ensure styling changes are visible
            try:
                self.refresh()
                # Also refresh the parent container to ensure changes propagate
                if self.parent:
                    self.parent.refresh()
            except Exception:
                pass

    def _get_type_colors(self, result_type: str) -> tuple[str, str]:
        """Get subtle background colors based on result type."""
        type_colors = {
            "trajectory": ("grey17", "grey50"),
            "step": ("grey15", "grey45"),
            "task": ("grey19", "grey50"),
            "collection": ("grey16", "grey48"),
            "fork": ("grey14", "grey46"),
            "unknown": ("grey18", "grey48"),
        }
        return type_colors.get(result_type, ("grey18", "grey48"))

    def _create_rich_content(self, text: str, result_type: str, context: str) -> str:
        """Create clean content with minimal icons."""
        # Simpler icons for different types
        icons = {"trajectory": "▶", "step": "•", "task": "◦", "collection": "▼", "fork": "↳", "unknown": "·"}

        icon = icons.get(result_type, "·")

        # Clean single-line format
        if context and len(context.strip()) > 0:
            # Truncate context if too long and clean it up
            context = context.strip()
            if len(context) > 40:
                context = context[:37] + "..."
            return f"{icon} {text} — {context}"
        else:
            return f"{icon} {text}"

    def on_click(self) -> None:
        """Handle click on this result."""
        try:
            self.search_panel.viewer.focus_search_result(self.index)
        except Exception:
            pass


@dataclass
class Event:
    type: str
    data: Dict[str, Any]


class SearchPanel(Static):
    """Search panel with input field and search results list."""

    def __init__(self, viewer: "TrajectoryViewer") -> None:
        super().__init__(id="search_panel")
        self.viewer = viewer
        self.search_input: Optional[Input] = None
        self.results_container: Optional[VerticalScroll] = None
        self.results_label: Optional[Label] = None
        self.visible = False
        self.current_results: List[TreeNode[str]] = []
        self.result_widgets: List[ClickableResult] = []
        self.highlighted_index: int = -1  # Track currently highlighted result
        try:
            # Start completely hidden with no height
            self.styles.height = 0
            self.styles.max_height = 20
            self.styles.background = "grey11"
            self.styles.border = ("thick", "cyan")
            self.styles.display = "none"
            self.styles.overflow_y = "hidden"
        except Exception:
            pass

    def compose(self) -> ComposeResult:  # type: ignore[override]
        with Vertical():
            self.search_input = Input(placeholder="Search node labels and content...", id="search_input")
            yield self.search_input

            # Results count label
            self.results_label = Label("No search results", id="search_results_label")
            try:
                self.results_label.styles.height = 1
                self.results_label.styles.color = "cyan"
            except Exception:
                pass
            yield self.results_label

            # Results container
            self.results_container = VerticalScroll(id="search_results_container")
            try:
                self.results_container.styles.height = 10
                self.results_container.styles.border = ("round", "grey")
                self.results_container.styles.margin = (1, 0)
            except Exception:
                pass
            yield self.results_container

    def show(self) -> None:
        """Show the search panel."""
        self.visible = True
        try:
            self.styles.display = "block"
            self.styles.height = "auto"
            self.styles.max_height = 20
        except Exception:
            pass
        if self.search_input:
            self.search_input.focus()

    def hide(self) -> None:
        """Hide the search panel."""
        self.visible = False
        try:
            self.styles.display = "none"
            self.styles.height = 0
        except Exception:
            pass

    def toggle(self) -> None:
        """Toggle search panel visibility."""
        if self.visible:
            self.hide()
        else:
            self.show()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search when user presses Enter."""
        if event.input == self.search_input:
            query = event.value.strip()
            if query:
                self.viewer.perform_search(query)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle real-time search as user types."""
        if event.input == self.search_input:
            query = event.value.strip()
            self.viewer.perform_search(query)

    def update_results(
        self, results: List[TreeNode[str]], query: str, total_counts: Optional[Dict[str, int]] = None
    ) -> None:
        """Update the search results display with grouping and enhanced styling."""
        self.current_results = results

        # Group results by type
        grouped_results = self._group_results_by_type(results)

        # Update results count label with breakdown including denominators
        if self.results_label:
            if not query:
                self.results_label.update("No search query")
            elif not results:
                self.results_label.update(f"No results for '{query}'")
            else:
                breakdown = self._create_results_breakdown(grouped_results, total_counts)
                self.results_label.update(f"Search results for '{query}': {breakdown}")

        # Update results container
        if self.results_container:
            # Clear ALL previous content (including headers and results)
            try:
                # Remove all children from the container
                for child in list(self.results_container.children):
                    child.remove()
            except Exception:
                pass
            self.result_widgets.clear()

            # Add results grouped by type
            result_index = 0
            for result_type, type_results in grouped_results.items():
                if type_results:
                    # Add colorful type header only if there are multiple types
                    if len([t for t, r in grouped_results.items() if r]) > 1:
                        # Get appropriate color for each type
                        type_colors = {
                            "collection": "magenta",
                            "trajectory": "bright_blue",
                            "task": "yellow",
                            "step": "green",
                            "fork": "cyan",
                            "unknown": "grey",
                        }
                        header_color = type_colors.get(result_type, "cyan")
                        # Create proper plural forms
                        plurals = {
                            "collection": "Collections",
                            "trajectory": "Trajectories",
                            "task": "Tasks",
                            "step": "Steps",
                            "fork": "Forks",
                            "unknown": "Unknown",
                        }
                        header_text = plurals.get(result_type, f"{result_type.title()}s")

                        # Create the header and force color styling
                        type_header = Label(header_text)
                        try:
                            # Set multiple color properties to ensure it works
                            type_header.styles.color = header_color
                            type_header.styles.text_style = "bold"
                            type_header.styles.margin = (1, 0, 0, 1)
                            # Also try setting it on the Label's renderable
                            from rich.text import Text as RichText

                            colored_text = RichText(header_text, style=f"bold {header_color}")
                            type_header.update(colored_text)
                        except Exception:
                            pass
                        self.results_container.mount(type_header)

                    # Add results for this type
                    for node in type_results:
                        label_text = self._format_result_label(node, result_index)
                        context = self._extract_result_context(node)
                        result_widget = ClickableResult(label_text, result_index, self, result_type, context)

                        self.result_widgets.append(result_widget)
                        self.results_container.mount(result_widget)
                        result_index += 1

    def _format_result_label(self, node: TreeNode[str], index: int) -> str:
        """Format a search result for display in the list."""
        # Get simplified node path - just the node itself and immediate parent
        current_label = str(node.label)

        # For steps, show the step info more clearly
        if "step" in current_label.lower():
            # Extract step number and summary
            parts = current_label.split(":", 1)
            if len(parts) > 1:
                step_info = parts[0].strip()
                summary = parts[1].strip()
                if len(summary) > 50:
                    summary = summary[:47] + "..."
                return f"{step_info}: {summary}"

        # For other types, show a clean version
        if len(current_label) > 60:
            current_label = current_label[:57] + "..."

        return current_label

    def _group_results_by_type(self, results: List[TreeNode[str]]) -> Dict[str, List[TreeNode[str]]]:
        """Group search results by their type."""
        groups: Dict[str, List[TreeNode[str]]] = {
            "collection": [],
            "trajectory": [],
            "task": [],
            "step": [],
            "fork": [],
            "unknown": [],
        }

        for node in results:
            node_type = self._detect_node_type(node)
            groups[node_type].append(node)

        return groups

    def _detect_node_type(self, node: TreeNode[str]) -> str:
        """Detect the type of a tree node."""
        label = str(node.label).lower()

        if "collection:" in label:
            return "collection"
        elif "traj:" in label:
            return "trajectory"
        elif "task:" in label:
            return "task"
        elif "step" in label:
            return "step"
        elif "fork" in label:
            return "fork"
        else:
            return "unknown"

    def _extract_result_context(self, node: TreeNode[str]) -> str:
        """Extract context/preview from node data."""
        if not hasattr(node, "data") or not node.data:
            return ""

        try:
            data = node.data
            if isinstance(data, dict):
                payload = data.get("payload", {})
                if isinstance(payload, dict):
                    if _is_openhands_step_payload(payload):
                        context = _openhands_search_text(payload)
                        if context:
                            return _shorten_text(context, 120)
                    # Extract relevant context based on type
                    if "code" in payload:
                        code = payload["code"]
                        if isinstance(code, str):
                            # Get first line of code
                            first_line = code.split("\n")[0].strip()
                            return first_line
                    elif "thought" in payload:
                        thought = payload["thought"]
                        if isinstance(thought, str):
                            # Get first sentence of thought
                            first_sentence = thought.split(".")[0].strip()
                            return first_sentence
                    elif "goal" in payload:
                        goal = payload["goal"]
                        if isinstance(goal, str):
                            return goal
                    elif "output" in payload:
                        output = payload["output"]
                        if isinstance(output, str):
                            # Get first line of output
                            first_line = output.split("\n")[0].strip()
                            return first_line
        except Exception:
            pass

        return ""

    def _create_results_breakdown(
        self, grouped_results: Dict[str, List[TreeNode[str]]], total_counts: Optional[Dict[str, int]] = None
    ) -> str:
        """Create a breakdown string of results by type with denominators."""
        # Proper plural forms for count display
        plurals = {
            "collection": "collections",
            "trajectory": "trajectories",  # Correct spelling!
            "task": "tasks",
            "step": "steps",
            "fork": "forks",
            "unknown": "unknown",
        }

        breakdown_parts = []
        for result_type, type_results in grouped_results.items():
            if type_results:
                match_count = len(type_results)
                plural_name = plurals.get(result_type, f"{result_type}s")

                # Add denominator if total counts are available
                if total_counts and result_type in total_counts:
                    total_count = total_counts[result_type]
                    if total_count > 0:  # Only show types that exist in the tree
                        breakdown_parts.append(f"{match_count}/{total_count} {plural_name}")
                else:
                    # Fallback to original format if no totals available
                    breakdown_parts.append(f"{match_count} {plural_name}")

        return ", ".join(breakdown_parts) if breakdown_parts else "no results"

    def clear_results(self) -> None:
        """Clear the search results display."""
        self.current_results.clear()
        self.highlighted_index = -1  # Reset highlight
        if self.results_label:
            self.results_label.update("No search results")
        if self.results_container:
            # Clear ALL children from the container (headers and results)
            try:
                for child in list(self.results_container.children):
                    child.remove()
            except Exception:
                pass
            self.result_widgets.clear()

    def highlight_result(self, index: int) -> None:
        """Highlight a specific search result by index."""
        # Clear ALL previous highlights (defensive approach)
        for i, widget in enumerate(self.result_widgets):
            if widget.is_highlighted:
                widget.set_highlighted(False)

        # Set new highlight
        self.highlighted_index = index
        if 0 <= index < len(self.result_widgets):
            self.result_widgets[index].set_highlighted(True)

            # Scroll to make the highlighted result visible
            try:
                highlighted_widget = self.result_widgets[index]
                if self.results_container:
                    # Try to scroll the highlighted widget into view
                    # Use a more compatible scrolling approach
                    if hasattr(self.results_container, "scroll_to_widget"):
                        self.results_container.scroll_to_widget(highlighted_widget)
                    elif hasattr(highlighted_widget, "scroll_visible"):
                        highlighted_widget.scroll_visible()
                    # Force a refresh of the entire container
                    self.results_container.refresh()
            except Exception:
                pass


class TrajectoryTree(Static):
    def __init__(self, *, mode: str = "auto") -> None:
        super().__init__()
        self.mode = _normalize_visualization_mode(mode)
        self.tree_widget: Optional[PlayPauseFriendlyTree[str]] = None
        self.traj_nodes: Dict[str, TreeNode[str]] = {}
        # Map grouping label -> group node to avoid duplicate "unlabeled" nodes
        self.group_nodes: Dict[str, TreeNode[str]] = {}
        # Map collection_id -> task id once trajectory_task_set arrives. Group
        # nodes are created before task metadata, so labels are upgraded later.
        self.collection_task_ids: Dict[str, str] = {}
        # Remember which group label a trajectory belongs to so later events are
        # attached consistently even if they don't repeat collection/process/task.
        self.traj_to_group_label: Dict[str, str] = {}
        # Maintain a single "steps" container per trajectory to avoid duplicates
        self.traj_steps_nodes: Dict[str, TreeNode[str]] = {}
        # Maintain a single "steps" container per trajectory to avoid duplicates
        # Map (trajectory_id, step_index) -> step node to enable focusing/scrolling
        self.step_nodes: Dict[tuple[str, int], TreeNode[str]] = {}
        # Map (bridge collection, turn) -> node for raw OpenReward bridge JSONL files.
        self.bridge_turn_nodes: Dict[tuple[str, int], TreeNode[str]] = {}
        # Track the latest known reward per trajectory for quick label updates
        self.traj_rewards: Dict[str, float] = {}
        # Track whether a trajectory has reached a terminal state.
        self.traj_finished: Dict[str, bool] = {}
        # Track which trajectories are currently expanded (for efficient collapse_all_except)
        self.expanded_trajs: set[str] = set()
        # Bulk loading mode - when True, don't expand new trajectory nodes
        self.bulk_loading: bool = False
        # Search functionality
        self.search_results: List[TreeNode[str]] = []
        self.current_search_index: int = -1
        self.current_search_query: str = ""

    def reset(self) -> None:
        if self.tree_widget is None:
            return
        # Remove all children under the root
        try:
            for child in list(self.tree_widget.root.children):
                try:
                    child.remove()
                except Exception:
                    pass
        except Exception:
            pass
        # Clear internal maps
        self.traj_nodes.clear()
        self.group_nodes.clear()
        self.collection_task_ids.clear()
        self.traj_to_group_label.clear()
        self.traj_steps_nodes.clear()
        self.step_nodes.clear()
        self.bridge_turn_nodes.clear()
        self.traj_rewards.clear()
        self.traj_finished.clear()
        self.expanded_trajs.clear()
        self.bulk_loading = False
        self.search_results.clear()
        self.current_search_index = -1
        self.current_search_query = ""

    def compose(self) -> ComposeResult:  # type: ignore[override]
        self.tree_widget = PlayPauseFriendlyTree("Trajectory Collections")
        yield self.tree_widget

    def search_nodes(self, query: str) -> List[TreeNode[str]]:
        """Search for nodes matching the query in labels and content."""
        if not query or not self.tree_widget:
            return []

        query_lower = query.lower()
        results: List[TreeNode[str]] = []

        def search_node_recursive(node: TreeNode[str]) -> None:
            # Search in node label
            label_text = str(node.label).lower()
            if query_lower in label_text:
                results.append(node)

            # Search in node data/content
            if hasattr(node, "data") and node.data:
                content_text = self._extract_searchable_content(node.data).lower()
                if query_lower in content_text:
                    results.append(node)

            # Recursively search children
            for child in node.children:
                search_node_recursive(child)

        # Start search from root
        search_node_recursive(self.tree_widget.root)
        return results

    def _extract_searchable_content(self, data: Any) -> str:
        """Extract searchable text content from node data."""
        if not data:
            return ""

        content_parts = []

        if isinstance(data, dict):
            # Extract payload content
            payload = data.get("payload", {})
            if isinstance(payload, dict):
                if _is_openhands_step_payload(payload):
                    content_parts.append(_openhands_search_text(payload))
                for key, value in payload.items():
                    if isinstance(value, str):
                        content_parts.append(value)
                    elif isinstance(value, (dict, list)):
                        try:
                            import json

                            content_parts.append(json.dumps(value))
                        except Exception:
                            content_parts.append(str(value))
                    else:
                        content_parts.append(str(value))
        elif isinstance(data, str):
            content_parts.append(data)
        else:
            content_parts.append(str(data))

        return " ".join(content_parts)

    def perform_search(self, query: str) -> None:
        """Perform search and update search results."""
        self.current_search_query = query
        self.search_results = self.search_nodes(query)
        self.current_search_index = -1

        # If we have results, highlight the first one
        if self.search_results:
            self.current_search_index = 0
            self._highlight_search_result()

        # Return results for SearchPanel to display
        return self.search_results

    def next_search_result(self) -> None:
        """Navigate to next search result."""
        if not self.search_results:
            return

        self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
        self._highlight_search_result()

    def previous_search_result(self) -> None:
        """Navigate to previous search result."""
        if not self.search_results:
            return

        self.current_search_index = (self.current_search_index - 1) % len(self.search_results)
        self._highlight_search_result()

    def _highlight_search_result(self) -> None:
        """Highlight and focus current search result."""
        if not self.search_results or self.current_search_index < 0:
            return

        current_node = self.search_results[self.current_search_index]

        # Expand parents to make the node visible
        node = current_node
        while node is not None:
            try:
                node.expand()
            except Exception:
                pass
            node = getattr(node, "parent", None)

        # Select and focus the node
        if self.tree_widget:
            for method_name in ("select_node", "select"):
                method = getattr(self.tree_widget, method_name, None)
                if method is not None:
                    try:
                        method(current_node)
                        break
                    except Exception:
                        pass

            # Scroll to make the node visible
            try:
                scroll_to_node = getattr(self.tree_widget, "scroll_to_node", None)
                if scroll_to_node is not None:
                    scroll_to_node(current_node)
            except Exception:
                pass

    def clear_search(self) -> None:
        """Clear search results."""
        self.search_results.clear()
        self.current_search_index = -1
        self.current_search_query = ""

    def focus_result_by_index(self, index: int) -> None:
        """Focus on a specific search result by index."""
        if 0 <= index < len(self.search_results):
            self.current_search_index = index
            self._highlight_search_result()

    def get_total_counts_by_type(self) -> Dict[str, int]:
        """Get total count of each node type in the tree."""
        if not self.tree_widget:
            return {}

        counts: Dict[str, int] = {"collection": 0, "trajectory": 0, "task": 0, "step": 0, "fork": 0, "unknown": 0}

        def count_nodes_recursive(node: TreeNode[str]) -> None:
            # Determine node type using the same logic as SearchPanel
            label = str(node.label).lower()
            if "collection:" in label:
                counts["collection"] += 1
            elif "traj:" in label:
                counts["trajectory"] += 1
            elif "task:" in label:
                counts["task"] += 1
            elif "step" in label:
                counts["step"] += 1
            elif "fork" in label:
                counts["fork"] += 1
            else:
                counts["unknown"] += 1

            # Recursively count children
            for child in node.children:
                count_nodes_recursive(child)

        # Start counting from root
        count_nodes_recursive(self.tree_widget.root)
        return counts


class SplitDivider(Static):
    """A simple draggable divider to resize left/right panes."""

    def __init__(self, viewer: "TrajectoryViewer") -> None:
        super().__init__(id="split_divider")
        self.viewer = viewer
        self._dragging: bool = False
        try:
            self.styles.cursor = "col-resize"  # type: ignore[attr-defined]
            self.styles.background = "grey23"  # type: ignore[attr-defined]
            self.styles.height = "100%"  # type: ignore[attr-defined]
        except Exception:
            pass

    def render(self) -> Text:  # type: ignore[override]
        # Draw a full-height bar; background does the main visual, but provide a character too
        return Text("│\n" * 2000)

    def on_mouse_down(self, event: MouseDown) -> None:  # type: ignore[override]
        self._dragging = True
        try:
            self.capture_mouse()
        except Exception:
            pass
        try:
            event.stop()
        except Exception:
            pass

    def on_mouse_up(self, event: MouseUp) -> None:  # type: ignore[override]
        self._dragging = False
        try:
            self.release_mouse()
        except Exception:
            pass
        try:
            event.stop()
        except Exception:
            pass

    def on_mouse_move(self, event: MouseMove) -> None:  # type: ignore[override]
        if not self._dragging:
            return
        try:
            row = getattr(self, "parent", None)
            region = getattr(row, "region", None)
            width = getattr(region, "width", None)
            screen_left = getattr(region, "x", 0)
            screen_x = getattr(event, "screen_x", None)
            if screen_x is None or width is None or width <= 0:
                return
            relative_x = max(0, min(screen_x - screen_left, width))
            pct = int((relative_x / float(width)) * 100)
            self.viewer.set_split(pct)
            event.stop()
        except Exception:
            pass

    def ensure_traj_node(
        self,
        traj_id: str,
        *,
        label: Optional[str] = None,
        parent: Optional[TreeNode[str]] = None,
        expand: bool = True,
    ) -> TreeNode[str]:
        """Return (and create if necessary) the tree node for *traj_id*.

        If *parent* is supplied, the trajectory node will be created as a child
        of that parent instead of at the tree root.  Subsequent calls will
        return the same node regardless of *parent*.

        If *expand* is True (default), the node will be expanded when created.
        Set to False during bulk loading for better performance.
        """
        assert self.tree_widget is not None

        if traj_id not in self.traj_nodes:
            target_parent: TreeNode[str] = parent if parent is not None else self.tree_widget.root
            node = target_parent.add(label or f"traj:{traj_id}")
            # New trajectory nodes should start expanded so that children (steps)
            # are visible immediately (unless bulk loading).
            if expand:
                node.expand()
                self.expanded_trajs.add(traj_id)
            self.traj_nodes[traj_id] = node

        return self.traj_nodes[traj_id]

    def _reward_color(self, reward: Optional[float]) -> Optional[str]:
        if reward is None:
            return None
        try:
            r = float(reward)
        except Exception:
            return None
        if r <= 0.0:
            return "red"
        if r >= 1.0:
            return "green"
        return "yellow"

    def _format_traj_label(self, traj_id: str) -> Text | str:
        reward = self.traj_rewards.get(traj_id)
        base = f"traj:{traj_id}"
        label = f"{base} · reward:{reward:.3f}" if reward is not None else base
        # Color only when finished
        if self.traj_finished.get(traj_id):
            color = self._reward_color(reward)
            if color:
                return Text(label, style=color)
        return label

    def _set_node_label(self, node: TreeNode[str], label: Text | str) -> None:
        # Textual versions differ; support both set_label and attribute assignment
        if hasattr(node, "set_label"):
            try:
                node.set_label(label)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        try:
            node.label = label  # type: ignore[assignment]
        except Exception:
            # Best-effort: if neither works, leave existing label
            pass

    def ingest(self, event: Event) -> None:
        # Group strictly by collection_id to ensure all trajectories from the same
        # dump or live session appear under a single root node.
        collection_id = event.data.get("collection_id")
        if not collection_id and event.type in BRIDGE_EVENT_TYPES:
            collection_id = _bridge_collection_id(event.data)

        # Extract trajectory id (if present) for stable grouping between events
        traj_id_for_group: Optional[str] = None
        if event.type == "trajectory_created":
            traj = event.data.get("trajectory")
            if isinstance(traj, dict):
                traj_id_for_group = traj.get("id")
        else:
            traj_id_for_group = event.data.get("trajectory_id")

        group_node: Optional[TreeNode[str]] = None
        if self.tree_widget is not None:
            label = f"collection:{collection_id}" if collection_id else "unlabeled"
            # find or create grouping node via our map to avoid duplicates
            group_node = self.group_nodes.get(label)
            if group_node is None:
                task_id = self.collection_task_ids.get(str(collection_id)) if collection_id else None
                group_node = self.tree_widget.root.add(_collection_display_label(collection_id, task_id))
                group_node.expand()
                group_node.data = {
                    "type": "collection",
                    "payload": {"collection_id": collection_id, "task_id": task_id},
                }
                self.group_nodes[label] = group_node
            # Remember association of this trajectory to the chosen group label
            if traj_id_for_group is not None:
                self.traj_to_group_label[traj_id_for_group] = label

        if event.type == "trajectory_created":
            traj = event.data["trajectory"]
            traj_id = traj["id"]
            parent_info = traj.get("parent_info")
            # Initialize known reward (if present) and label accordingly
            try:
                self.traj_rewards[traj_id] = float(traj.get("reward", 0.0))
            except Exception:
                self.traj_rewards[traj_id] = 0.0
            # Initialize terminal status from payload if present.
            try:
                finish_msg = traj.get("finish_message")
                error_msg = traj.get("error_message")
                self.traj_finished[traj_id] = finish_msg is not None or error_msg is not None
            except Exception:
                self.traj_finished[traj_id] = False
            label = self._format_traj_label(traj_id)
            node = self.ensure_traj_node(traj_id, label=label, parent=group_node, expand=not self.bulk_loading)
            node.data = {"type": "trajectory", "payload": traj}
            # Render forks as a dedicated child under the child trajectory for clarity
            if parent_info:
                parent_id = parent_info.get("id")
                fork_step = parent_info.get("fork_step")
                if parent_id:
                    fork_node = node.add(f"fork from {parent_id} @ step {fork_step}")
                    fork_node.data = {"type": "fork", "payload": parent_info}
            # Add a stable "Steps" subgroup to unclutter the trajectory root
            if traj_id not in self.traj_steps_nodes:
                steps_node = node.add("steps")
                steps_node.data = None
                self.traj_steps_nodes[traj_id] = steps_node

        elif event.type == "trajectory_task_set":
            traj_id = event.data["trajectory_id"]
            task = event.data.get("task")
            task_id = _task_display_id(task)
            if task_id and collection_id and group_node is not None:
                self.collection_task_ids[str(collection_id)] = task_id
                self._set_node_label(group_node, _collection_display_label(collection_id, task_id))
                group_node.data = {
                    "type": "collection",
                    "payload": {"collection_id": collection_id, "task_id": task_id},
                }
            node = self.ensure_traj_node(traj_id, parent=group_node, expand=not self.bulk_loading)
            if task:
                task_goal = task.get("goal")
                task_label = f"task: {task_id}" if task_id else "task"
                if isinstance(task_goal, str) and task_goal.strip():
                    task_label = f"{task_label} · {_shorten_text(task_goal, 160)}"
            else:
                task_label = "task: None"
            task_node = node.add(task_label)
            task_node.data = {"type": "task", "payload": task} if task else None

        elif event.type == "trajectory_step_added":
            traj_id = event.data["trajectory_id"]
            step_index = event.data.get("step_index")
            step = event.data.get("step")
            node = self.ensure_traj_node(traj_id, parent=group_node, expand=not self.bulk_loading)
            # Update cached reward from event if available; fallback to existing value
            new_reward = event.data.get("reward")
            if isinstance(new_reward, (int, float)):
                try:
                    self.traj_rewards[traj_id] = float(new_reward)
                except Exception:
                    pass
            # Update finish/error metadata on the trajectory payload if present
            try:
                finish_msg = event.data.get("finish_message")
                error_msg = event.data.get("error_message")
                if isinstance(node.data, dict):
                    payload = node.data.get("payload") if isinstance(node.data.get("payload"), dict) else None
                    if isinstance(payload, dict):
                        if finish_msg is not None:
                            payload["finish_message"] = finish_msg
                        if error_msg is not None:
                            payload["error_message"] = error_msg
                # Update terminal status if we received any terminal info.
                if finish_msg is not None or error_msg is not None:
                    self.traj_finished[traj_id] = True
            except Exception:
                pass
            # Update the trajectory node label to reflect latest reward
            self._set_node_label(node, self._format_traj_label(traj_id))
            # Keep payload's reward in sync if present so DetailsPanel shows latest
            try:
                if isinstance(node.data, dict) and isinstance(node.data.get("payload"), dict):
                    node.data["payload"]["reward"] = self.traj_rewards.get(
                        traj_id, node.data["payload"].get("reward", 0.0)
                    )
            except Exception:
                pass
            # Ensure steps subgroup exists (via mapping)
            steps_group = self.traj_steps_nodes.get(traj_id)
            if steps_group is None:
                steps_group = node.add("steps")
                self.traj_steps_nodes[traj_id] = steps_group
            # Build a concise summary label for the step
            summary_parts: List[str] = []
            if isinstance(step, dict):
                if _should_render_openhands(getattr(self, "mode", "auto"), step):
                    openhands_summary = _openhands_step_summary(step)
                    if openhands_summary:
                        summary_parts.append(openhands_summary)
                if not summary_parts:
                    if "code" in step and isinstance(step["code"], str):
                        code_lines = step["code"].splitlines()
                        if code_lines:
                            code_line = code_lines[0].strip()
                            # Do not truncate; let the UI scroll horizontally
                            summary_parts.append(f"code: {code_line}")
                    if "thought" in step and isinstance(step["thought"], str):
                        thought_lines = step["thought"].splitlines()
                        if thought_lines:
                            thought_line = thought_lines[0].strip()
                            if thought_line:
                                summary_parts.append(f"thought: {thought_line}")
                    for k in ("output", "error"):
                        v = step.get(k)
                        if v:
                            summary_parts.append(k)
                if not summary_parts:
                    openhands_summary = _openhands_step_summary(step)
                    if openhands_summary:
                        summary_parts.append(openhands_summary)
            step_summary = "; ".join(summary_parts) if summary_parts else "(details)"
            step_node = steps_group.add(f"step {step_index}: {step_summary}")
            step_node.data = {"type": "step", "payload": step}
            try:
                if isinstance(step_index, int):
                    self.step_nodes[(traj_id, step_index)] = step_node
            except Exception:
                pass

        elif event.type == "trajectory_finished":
            traj_id = event.data["trajectory_id"]
            node = self.ensure_traj_node(traj_id, parent=group_node, expand=not self.bulk_loading)
            # Update cached reward and terminal status
            try:
                new_reward = event.data.get("reward")
                if isinstance(new_reward, (int, float)):
                    self.traj_rewards[traj_id] = float(new_reward)
            except Exception:
                pass
            try:
                finish_msg = event.data.get("finish_message")
                error_msg = event.data.get("error_message")
                if isinstance(node.data, dict):
                    payload = node.data.get("payload") if isinstance(node.data.get("payload"), dict) else None
                    if isinstance(payload, dict):
                        if finish_msg is not None:
                            payload["finish_message"] = finish_msg
                        if error_msg is not None:
                            payload["error_message"] = error_msg
                # A trajectory_finished event is authoritative even when finish_message == "".
                self.traj_finished[traj_id] = True
            except Exception:
                pass
            # Refresh label to reflect final reward and status
            self._set_node_label(node, self._format_traj_label(traj_id))

        elif event.type in BRIDGE_EVENT_TYPES:
            if group_node is None:
                return
            turn = event.data.get("turn")
            if isinstance(turn, int):
                turn_key = (str(collection_id), turn)
                turn_node = self.bridge_turn_nodes.get(turn_key)
                if turn_node is None:
                    turn_node = group_node.add(f"turn {turn}")
                    turn_node.data = {"type": "bridge_turn", "payload": {"turn": turn}}
                    self.bridge_turn_nodes[turn_key] = turn_node
                bridge_node = turn_node.add(_bridge_record_summary(event.data))
            else:
                bridge_node = group_node.add(_bridge_record_summary(event.data))
            bridge_node.data = {"type": "bridge_event", "payload": event.data}

    def focus_step(self, traj_id: str, step_index: int) -> None:
        if self.tree_widget is None:
            return
        node = self.step_nodes.get((traj_id, step_index))
        if node is None:
            return
        # Expand parents to make sure it's visible
        cur = node
        while cur is not None:
            try:
                cur.expand()
            except Exception:
                pass
            cur = getattr(cur, "parent", None)
        # Try selection APIs across Textual versions
        for method_name in ("select_node", "select"):
            method = getattr(self.tree_widget, method_name, None)
            if method is not None:
                try:
                    method(node)
                    break
                except Exception:
                    pass
        # Fallback to setting cursor_node if available
        try:
            setattr(self.tree_widget, "cursor_node", node)
        except Exception:
            pass
        # Smoothly ensure visibility and center the node if possible
        try:
            # First, if Tree provides a direct helper
            scroll_to_node = getattr(self.tree_widget, "scroll_to_node", None)
            if scroll_to_node is not None:
                try:
                    scroll_to_node(node)
                except Exception:
                    pass
            # Then, center the node within the viewport if we can access region/size
            region = getattr(node, "region", None)
            height = getattr(getattr(self.tree_widget, "size", None), "height", None)
            scroll_to = getattr(self.tree_widget, "scroll_to", None)
            # Determine current vertical offset
            offset_y = 0
            scroll_offset = getattr(self.tree_widget, "scroll_offset", None)
            if scroll_offset is not None:
                offset_y = getattr(scroll_offset, "y", 0) or 0
            if region is not None and isinstance(height, int) and scroll_to is not None:
                node_y = getattr(region, "y", None)
                if isinstance(node_y, int):
                    top_visible = offset_y
                    bottom_visible = offset_y + max(height - 1, 1)
                    if node_y < top_visible + 2 or node_y > bottom_visible - 2:
                        target_y = max(node_y - height // 2, 0)
                        try:
                            scroll_to(y=target_y)
                        except Exception:
                            pass
        except Exception:
            pass

    def highlight_step(self, traj_id: str, step_index: int, *, duration: float = 0.6) -> None:
        if self.tree_widget is None:
            return
        node = self.step_nodes.get((traj_id, step_index))
        if node is None:
            return
        try:
            # Preserve original label for later restore
            if not isinstance(getattr(node, "data", None), dict):
                node.data = {}
            original_label = str(node.label)
            node.data["__orig_label"] = original_label
            # Apply a temporary highlight style
            from rich.text import Text as RichText

            highlighted = RichText(original_label, style="bold reverse")
            self._set_node_label(node, highlighted)  # type: ignore[arg-type]
            # Schedule highlight removal
            import asyncio as _asyncio

            async def _clear() -> None:
                try:
                    await _asyncio.sleep(duration)
                    # Node may have been removed or updated; re-fetch
                    n = self.step_nodes.get((traj_id, step_index))
                    if n is None:
                        return
                    orig = None
                    if isinstance(n.data, dict):
                        orig = n.data.pop("__orig_label", None)
                    if orig is not None:
                        self._set_node_label(n, orig)
                        try:
                            self.refresh()
                        except Exception:
                            pass
                except Exception:
                    pass

            _asyncio.create_task(_clear())
        except Exception:
            return

    def collapse_all_except(self, keep_traj_id: str) -> None:
        # Collapse only currently expanded trajectory nodes (O(k) where k = expanded, not O(n) total)
        # First, expand the one we want to keep
        tnode = self.traj_nodes.get(keep_traj_id)
        if tnode is not None:
            try:
                tnode.expand()
            except Exception:
                pass
            self.expanded_trajs.add(keep_traj_id)
            # Also expand steps under the kept trajectory
            steps_node = self.traj_steps_nodes.get(keep_traj_id)
            if steps_node is not None:
                try:
                    steps_node.expand()
                except Exception:
                    pass

        # Collapse all other expanded trajectories
        to_collapse = [tid for tid in self.expanded_trajs if tid != keep_traj_id]
        for tid in to_collapse:
            tnode = self.traj_nodes.get(tid)
            if tnode is not None:
                try:
                    tnode.collapse()
                except Exception:
                    pass
            self.expanded_trajs.discard(tid)


# Bind methods accidentally nested under SplitDivider back onto TrajectoryTree
try:
    TrajectoryTree.ensure_traj_node = SplitDivider.ensure_traj_node  # type: ignore[attr-defined]
    TrajectoryTree._reward_color = SplitDivider._reward_color  # type: ignore[attr-defined]
    TrajectoryTree._format_traj_label = SplitDivider._format_traj_label  # type: ignore[attr-defined]
    TrajectoryTree._set_node_label = SplitDivider._set_node_label  # type: ignore[attr-defined]
    TrajectoryTree.ingest = SplitDivider.ingest  # type: ignore[attr-defined]
    TrajectoryTree.focus_step = SplitDivider.focus_step  # type: ignore[attr-defined]
    TrajectoryTree.highlight_step = SplitDivider.highlight_step  # type: ignore[attr-defined]
    TrajectoryTree.collapse_all_except = SplitDivider.collapse_all_except  # type: ignore[attr-defined]

except Exception:
    pass


class TrajectoryViewer(App):
    CSS_PATH = None
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("right", "step", "Next"),
        Binding("n", "step", show=False),
        Binding("r", "restart", "Restart"),
        Binding("ctrl+f", "toggle_search", "Search"),
        Binding("m", "toggle_mouse_capture", "Mouse/Select"),
        Binding("d", "toggle_details_only", "Details Only"),
        Binding("f3", "next_search", "Next Result"),
        Binding("shift+f3", "prev_search", "Prev Result"),
        Binding("escape", "close_search", "Close Search"),
    ]

    def __init__(
        self,
        event_queue: Optional[queue.Queue] = None,
        jsonl_path: Optional[str | Path] = None,
        jsonl_paths: Optional[List[str | Path]] = None,
        *,
        # Default to reading from the beginning so that existing events are visible
        # when opening an already-populated JSONL file.
        start_at_end: bool = False,
        # If provided, replay events from JSONL(s) with this delay between events
        # (seconds). When set, files are read from the beginning and stop at EOF
        # rather than tailing indefinitely.
        replay_delay: Optional[float] = None,
        mode: str = "auto",
        selectable_text: bool = False,
    ) -> None:
        super().__init__()
        self.mode = _normalize_visualization_mode(mode)
        self.selectable_text = selectable_text
        self.mouse_capture_enabled = not selectable_text
        self.event_queue = event_queue
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.jsonl_paths = [Path(p) for p in jsonl_paths] if jsonl_paths else None
        self.tree_widget = TrajectoryTree(mode=self.mode)
        self.details_panel = DetailsPanel(mode=self.mode)
        self.search_panel = SearchPanel(self)
        self.start_at_end = start_at_end
        self.replay_delay = replay_delay
        self._polling_task: Optional[asyncio.Task] = None
        self._tail_tasks: List[asyncio.Task] = []
        # Replay control state
        self._replay_records: List[Dict[str, Any]] = []
        self._replay_index: int = 0
        self._replay_running: bool = False
        # Split view state
        self._split_pct: int = 60
        self._details_only: bool = False
        self._left_container: Optional[HorizontalScroll] = None
        self._right_outer_container: Optional[HorizontalScroll] = None
        self._right_container: Optional[VerticalScroll] = None
        self._divider: Optional[Static] = None

    def compose(self) -> ComposeResult:  # type: ignore[override]
        yield Header(show_clock=True)
        yield self.search_panel
        # Two-pane layout with a draggable divider and no gap between panes
        row = Horizontal()
        try:
            row.styles.gap = 0
            row.styles.padding = 0
            row.styles.margin = 0
        except Exception:
            pass
        with row:
            # Left pane: Tree within a horizontal scroller to avoid truncation
            self._left_container = HorizontalScroll(id="left_pane")
            try:
                self._left_container.styles.width = f"{self._split_pct}%"
                self._left_container.styles.min_width = 20
            except Exception:
                pass
            with self._left_container:
                try:
                    self.tree_widget.styles.overflow_x = "auto"  # type: ignore[attr-defined]
                except Exception:
                    pass
                yield self.tree_widget

            # Draggable divider between panes
            self._divider = SplitDivider(self)
            try:
                self._divider.styles.width = 2
                self._divider.styles.min_width = 2
            except Exception:
                pass
            yield self._divider

            # Right pane: Details with vertical + horizontal scrolling
            outer_h = HorizontalScroll()
            self._right_outer_container = outer_h
            try:
                outer_h.styles.flex = 1
                # Allow vertical scrolling to propagate to the inner VerticalScroll
                outer_h.styles.overflow_y = "visible"  # type: ignore[attr-defined]
            except Exception:
                pass
            with outer_h:
                self._right_container = VerticalScroll(id="right_pane")
                try:
                    self._right_container.styles.flex = 1
                    self._right_container.styles.overflow_x = "auto"  # type: ignore[attr-defined]
                except Exception:
                    pass
                with self._right_container:
                    try:
                        self.details_panel.styles.overflow_x = "auto"  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    yield self.details_panel
        yield row
        yield Footer()

    def set_split(self, pct: int) -> None:
        if self._details_only:
            return
        pct = max(10, min(90, int(pct)))
        self._split_pct = pct
        if self._left_container is not None:
            try:
                self._left_container.styles.width = f"{pct}%"
            except Exception:
                pass
        try:
            self.refresh()
        except Exception:
            pass

    def action_toggle_play(self) -> None:
        if not self._replay_records:
            return
        # Debounce to avoid key auto-repeat glitches blanking the screen
        now = time.time()
        if now - getattr(self, "_last_toggle_ts", 0.0) < 0.15:
            return
        self._last_toggle_ts = now
        if self._replay_running:
            self._replay_running = False
        else:
            delay = float(self.replay_delay or 0.5)
            asyncio.create_task(self._replay_autoplay_loop(delay))

    def action_step(self) -> None:
        if not self._replay_records:
            return
        # Pause autoplay if running before single-step
        self._replay_running = False
        asyncio.create_task(self._replay_step_forward())

    def action_restart(self) -> None:
        if not self._replay_records:
            return
        self._replay_running = False
        self._replay_index = 0
        # Reset UI and render first record again
        self.tree_widget.reset()
        asyncio.create_task(self._handle_record(self._replay_records[0]))

    def action_toggle_search(self) -> None:
        """Toggle search panel visibility."""
        self.search_panel.toggle()

    def action_close_search(self) -> None:
        """Close search panel and clear search results."""
        self.search_panel.hide()
        self.search_panel.clear_results()
        self.tree_widget.clear_search()

    def action_next_search(self) -> None:
        """Navigate to next search result."""
        self.tree_widget.next_search_result()
        # Update highlight in search panel
        if hasattr(self.tree_widget, "current_search_index"):
            self.search_panel.highlight_result(self.tree_widget.current_search_index)

    def action_prev_search(self) -> None:
        """Navigate to previous search result."""
        self.tree_widget.previous_search_result()
        # Update highlight in search panel
        if hasattr(self.tree_widget, "current_search_index"):
            self.search_panel.highlight_result(self.tree_widget.current_search_index)

    def action_toggle_mouse_capture(self) -> None:
        """Toggle terminal mouse capture so text can be selected with the mouse."""
        self.mouse_capture_enabled = not self.mouse_capture_enabled
        self._apply_mouse_capture()
        try:
            if self.mouse_capture_enabled:
                self.notify("Mouse capture enabled")
            else:
                self.notify("Mouse capture disabled; drag-select text in your terminal")
        except Exception:
            pass

    def action_toggle_details_only(self) -> None:
        """Expand details to full width so terminal selection stays within details."""
        self._details_only = not self._details_only
        self._apply_details_only_layout()
        try:
            if self._details_only:
                self.notify("Details-only view enabled")
            else:
                self.notify("Split view enabled")
        except Exception:
            pass

    def _apply_details_only_layout(self) -> None:
        try:
            if self._left_container is not None:
                self._left_container.styles.width = 0 if self._details_only else f"{self._split_pct}%"
                self._left_container.styles.min_width = 0 if self._details_only else 20
            if self._divider is not None:
                self._divider.styles.width = 0 if self._details_only else 2
                self._divider.styles.min_width = 0 if self._details_only else 2
            if self._right_outer_container is not None:
                self._right_outer_container.styles.flex = 1
        except Exception:
            pass
        try:
            self.refresh()
        except Exception:
            pass

    def _apply_mouse_capture(self) -> None:
        sequence = MOUSE_CAPTURE_ENABLE if self.mouse_capture_enabled else MOUSE_CAPTURE_DISABLE
        try:
            sys.__stdout__.write(sequence)
            sys.__stdout__.flush()
        except Exception:
            pass

    def perform_search(self, query: str) -> None:
        """Perform search in tree widget and update search panel."""
        if query:
            results = self.tree_widget.perform_search(query)
            total_counts = self.tree_widget.get_total_counts_by_type()
            self.search_panel.update_results(results, query, total_counts)
            # Highlight the first result if any results found
            if results and hasattr(self.tree_widget, "current_search_index"):
                self.search_panel.highlight_result(self.tree_widget.current_search_index)
        else:
            self.tree_widget.clear_search()
            self.search_panel.clear_results()

    def focus_search_result(self, index: int) -> None:
        """Focus on a specific search result by index."""
        self.tree_widget.focus_result_by_index(index)
        # Also update the highlight in the search panel
        self.search_panel.highlight_result(index)

    async def on_mount(self) -> None:  # type: ignore[override]
        self._apply_mouse_capture()
        if self.event_queue is not None:
            self._polling_task = asyncio.create_task(self._poll_queue())
        elif self.jsonl_path is not None:
            if self.replay_delay is not None:
                # If delay is 0 or negative, load instantly without autoplay
                if float(self.replay_delay) <= 0.0:
                    self._polling_task = asyncio.create_task(self._replay_jsonl_instant())
                else:
                    self._polling_task = asyncio.create_task(self._replay_jsonl())
            else:
                self._polling_task = asyncio.create_task(self._tail_jsonl())
        elif self.jsonl_paths is not None:
            if self.replay_delay is not None:
                # Load and sort all, start paused; if delay <= 0 load instantly
                paths = [Path(p) for p in self.jsonl_paths]
                if float(self.replay_delay) <= 0.0:
                    self._polling_task = asyncio.create_task(self._replay_jsonls_instant(paths))
                else:
                    self._polling_task = asyncio.create_task(self._replay_jsonls(paths))
            else:
                for p in self.jsonl_paths:
                    self._tail_tasks.append(asyncio.create_task(self._tail_one_jsonl(Path(p))))

    async def _poll_queue(self) -> None:
        while True:
            try:
                record = self.event_queue.get(timeout=0.2)  # type: ignore[attr-defined]
                await self._handle_record(record)
            except queue.Empty:
                await asyncio.sleep(0.05)

    async def _tail_jsonl(self) -> None:
        assert self.jsonl_path is not None
        await self._tail_file_loop(self.jsonl_path, self.start_at_end)

    async def _tail_one_jsonl(self, path: Path) -> None:
        await self._tail_file_loop(path, self.start_at_end)

    async def _tail_file_loop(self, path: Path, start_at_end: bool) -> None:
        """Tail a JSONL file, handling truncation and replacement.

        - If the file is truncated (size < current position), seek to start.
        - If the file is replaced (inode change), reopen and seek to start.
        - If start_at_end is True on first open, seek to end; otherwise, begin at start.
        - Uses bulk loading for existing content to avoid per-record UI refresh.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        last_inode: Optional[int] = None
        pos: int = 0
        file_obj: Optional[object] = None
        first_open: bool = True
        bulk_load_done: bool = False

        while True:
            try:
                stat = path.stat()
            except FileNotFoundError:
                # Wait for the file to appear
                await asyncio.sleep(0.2)
                continue

            inode = getattr(stat, "st_ino", None)
            # Open or reopen file if needed
            if file_obj is None or getattr(file_obj, "closed", False) or inode != last_inode:
                try:
                    if file_obj and not getattr(file_obj, "closed", False):
                        file_obj.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                file_obj = path.open("r")
                if first_open and start_at_end:
                    file_obj.seek(0, 2)  # type: ignore[attr-defined]
                    bulk_load_done = True  # No bulk load needed if starting at end
                else:
                    file_obj.seek(0)  # type: ignore[attr-defined]
                pos = file_obj.tell()  # type: ignore[attr-defined]
                last_inode = inode
                first_open = False

            # Bulk load existing content on first open (without per-record refresh)
            if not bulk_load_done:
                records_loaded = 0
                if self.tree_widget is not None:
                    self.tree_widget.bulk_loading = True
                while True:
                    line = file_obj.readline()  # type: ignore[attr-defined]
                    if not line:
                        break
                    pos = file_obj.tell()  # type: ignore[attr-defined]
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    # Ingest without full UI refresh
                    await self._handle_record_bulk(record)
                    records_loaded += 1
                    # Yield periodically to keep UI responsive
                    if records_loaded % 100 == 0:
                        await asyncio.sleep(0)
                # Single refresh after bulk load
                if self.tree_widget is not None:
                    self.tree_widget.bulk_loading = False
                    # Expand root and collection nodes so trajectories are visible
                    try:
                        self.tree_widget.tree_widget.root.expand()
                        for group_node in self.tree_widget.group_nodes.values():
                            group_node.expand()
                    except Exception:
                        pass
                    self.tree_widget.refresh()
                bulk_load_done = True
                continue

            # Normal tailing mode: read a line
            line = file_obj.readline()  # type: ignore[attr-defined]
            if line:
                pos = file_obj.tell()  # type: ignore[attr-defined]
                try:
                    record = json.loads(line)
                except Exception:
                    # Skip malformed line
                    await asyncio.sleep(0)
                    continue
                await self._handle_record(record)
                await asyncio.sleep(0)
                continue

            # No line; check for truncation or replacement
            try:
                stat_now = path.stat()
            except FileNotFoundError:
                # File removed; close and wait
                try:
                    if file_obj and not getattr(file_obj, "closed", False):
                        file_obj.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
                file_obj = None
                last_inode = None
                await asyncio.sleep(0.2)
                continue

            # Truncation detected
            if stat_now.st_size < pos:
                try:
                    file_obj.seek(0)  # type: ignore[attr-defined]
                    pos = file_obj.tell()  # type: ignore[attr-defined]
                except Exception:
                    # Reopen if seek fails
                    try:
                        file_obj.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    file_obj = None
                    last_inode = None
                bulk_load_done = False  # Re-do bulk load on truncation
            else:
                # Idle briefly before next poll
                await asyncio.sleep(0.2)

    async def _replay_jsonl(self) -> None:
        """Replay events from a single JSONL file with a fixed delay and ordering by timestamp."""
        assert self.jsonl_path is not None
        delay = float(self.replay_delay or 0.0)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        # Load all records, sort by ts if present
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except Exception:
                    continue
        records.sort(key=_record_sort_key)
        self._replay_records = records
        self._replay_index = 0
        # If running auto-play, loop until done; otherwise, render first frame if any
        if delay <= 0:
            delay = 0.5
        # Start paused initially
        self._replay_running = False
        if self._replay_records:
            await self._handle_record(self._replay_records[0])

    async def _replay_jsonl_instant(self) -> None:
        """Load all records and render them immediately without delay."""
        assert self.jsonl_path is not None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path.touch(exist_ok=True)
        records: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except Exception:
                    continue
        records.sort(key=_record_sort_key)
        for record in records:
            await self._handle_record(record)

    async def _replay_one_jsonl(self, path: Path) -> None:
        """Replay events from one JSONL among many, with a fixed delay and ordering by timestamp."""
        delay = float(self.replay_delay or 0.0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        records: List[Dict[str, Any]] = []
        with path.open("r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except Exception:
                    continue
        records.sort(key=_record_sort_key)
        # For multi-file replay we can just stream sorted records
        for record in records:
            await self._handle_record(record)
            if delay > 0:
                await asyncio.sleep(delay)

    async def _replay_jsonls(self, paths: List[Path]) -> None:
        """Replay events from multiple JSONL files, ordered by timestamp, starting paused."""
        all_records: List[Dict[str, Any]] = []
        for p in paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch(exist_ok=True)
                with p.open("r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            all_records.append(rec)
                        except Exception:
                            continue
            except Exception:
                continue
        all_records.sort(key=_record_sort_key)
        self._replay_records = all_records
        self._replay_index = 0
        # Start paused
        self._replay_running = False
        if self._replay_records:
            # Reset UI and render first record
            self.tree_widget.reset()
            await self._handle_record(self._replay_records[0])

    async def _replay_jsonls_instant(self, paths: List[Path]) -> None:
        """Load all records from multiple files and render them immediately."""
        all_records: List[Dict[str, Any]] = []
        for p in paths:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch(exist_ok=True)
                with p.open("r") as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                            all_records.append(rec)
                        except Exception:
                            continue
            except Exception:
                continue
        all_records.sort(key=_record_sort_key)
        # Bulk load without per-record refresh
        if self.tree_widget is not None:
            self.tree_widget.bulk_loading = True
        for i, rec in enumerate(all_records):
            await self._handle_record_bulk(rec)
            # Yield periodically to keep UI responsive
            if i % 100 == 0:
                await asyncio.sleep(0)
        # Single refresh after all records loaded
        if self.tree_widget is not None:
            self.tree_widget.bulk_loading = False
            # Expand root and collection nodes so trajectories are visible
            try:
                self.tree_widget.tree_widget.root.expand()
                for group_node in self.tree_widget.group_nodes.values():
                    group_node.expand()
            except Exception:
                pass
            self.tree_widget.refresh()

    async def _handle_record_bulk(self, record: Dict[str, Any]) -> None:
        """Handle a record during bulk loading - ingest only, no refresh or auto-expand."""
        ev = Event(type=record.get("type", "unknown"), data=record)
        self.tree_widget.ingest(ev)

    async def _handle_record(self, record: Dict[str, Any]) -> None:
        ev = Event(type=record.get("type", "unknown"), data=record)
        self.tree_widget.ingest(ev)
        # Refresh synchronously; Textual's refresh is not awaitable.
        # Using the widget's refresh ensures the tree reflects newly-added nodes.
        if self.tree_widget is not None:
            self.tree_widget.refresh()

        # Auto-expand the affected node and its parents when new events arrive
        try:
            traj_id = record.get("trajectory_id")
            if ev.type == "trajectory_created":
                traj = record.get("trajectory", {})
                traj_id = traj.get("id")
            if traj_id and self.tree_widget is not None:
                node = self.tree_widget.traj_nodes.get(traj_id)
                if node is not None:
                    # Expand node and all parents
                    cur = node
                    while cur is not None:
                        try:
                            cur.expand()
                        except Exception:
                            pass
                        cur = getattr(cur, "parent", None)
                    # Track this trajectory as expanded
                    self.tree_widget.expanded_trajs.add(traj_id)
                    # If a step event, focus that step, center it, and briefly highlight.
                    # Also collapse other trajectories to reduce clutter.
                    if ev.type == "trajectory_step_added":
                        step_index = record.get("step_index")
                        if isinstance(step_index, int):
                            # Collapse is now O(k) where k = expanded trajectories, not O(n) total
                            try:
                                self.tree_widget.collapse_all_except(traj_id)
                            except Exception:
                                pass
                            self.tree_widget.focus_step(traj_id, step_index)
                            self.tree_widget.highlight_step(traj_id, step_index)
        except Exception:
            pass

    async def _replay_step_forward(self) -> None:
        if self._replay_index < len(self._replay_records) - 1:
            self._replay_index += 1
            await self._handle_record(self._replay_records[self._replay_index])

    async def _replay_autoplay_loop(self, delay: float) -> None:
        self._replay_running = True
        try:
            while self._replay_running and self._replay_index < len(self._replay_records) - 1:
                await self._replay_step_forward()
                await asyncio.sleep(delay)
        finally:
            self._replay_running = False

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:  # type: ignore[override]
        node = event.node
        label = str(node.label)
        payload = getattr(node, "data", None)
        self.details_panel.show(label, payload)


def run_viewer_from_queue(event_queue: queue.Queue, *, mode: str = "auto", selectable_text: bool = False) -> None:
    app = TrajectoryViewer(event_queue=event_queue, mode=mode, selectable_text=selectable_text)
    app.run()


def run_viewer_from_jsonl(
    path: str | Path, *, start_at_end: bool = False, mode: str = "auto", selectable_text: bool = False
) -> None:
    """Launch a TrajectoryViewer for a single JSONL file.

    By default, the viewer starts reading from the **beginning** of the file so that
    you can inspect previously-recorded events.  Set ``start_at_end=True`` to mimic
    a *tail -f* style live view that ignores existing lines and only shows new
    events appended after the viewer starts.
    """
    app = TrajectoryViewer(jsonl_path=path, start_at_end=start_at_end, mode=mode, selectable_text=selectable_text)
    app.run()


def run_viewer_from_jsonls(
    paths: List[str | Path], *, start_at_end: bool = False, mode: str = "auto", selectable_text: bool = False
) -> None:
    """Launch a TrajectoryViewer that tails multiple JSONL files in parallel."""
    app = TrajectoryViewer(jsonl_paths=paths, start_at_end=start_at_end, mode=mode, selectable_text=selectable_text)
    app.run()


def run_replay_from_jsonl(
    path: str | Path, *, delay: float = 0.5, mode: str = "auto", selectable_text: bool = False
) -> None:
    app = TrajectoryViewer(
        jsonl_path=path,
        start_at_end=False,
        replay_delay=delay,
        mode=mode,
        selectable_text=selectable_text,
    )
    app.run()


def run_replay_from_jsonls(
    paths: List[str | Path], *, delay: float = 0.5, mode: str = "auto", selectable_text: bool = False
) -> None:
    app = TrajectoryViewer(
        jsonl_paths=paths,
        start_at_end=False,
        replay_delay=delay,
        mode=mode,
        selectable_text=selectable_text,
    )
    app.run()


class DetailsPanel(Static):
    """Renders the details of the selected node with minimal clutter.

    - For step payloads (dicts), shows key panels; code fields are syntax highlighted.
    - For other payloads, pretty-prints JSON when possible.
    """

    def __init__(self, *, mode: str = "auto") -> None:
        super().__init__()
        self.mode = _normalize_visualization_mode(mode)
        self.current_content: str = ""
        self.search_query: str = ""

    CODE_KEYS_TO_LANG = {
        "code": "python",
        "python": "python",
        "py": "python",
        "bash": "bash",
        "sh": "bash",
        "shell": "bash",
        "sql": "sql",
        "javascript": "javascript",
        "js": "javascript",
    }

    def show(self, label: str, payload: Any) -> None:
        if not payload:
            self.current_content = ""
            self.update(Panel(Text("No details"), title=label))
            return

        payload_type = payload.get("type") if isinstance(payload, dict) else None
        data = payload.get("payload") if isinstance(payload, dict) else payload

        # Store content for searching
        self.current_content = self._extract_content_text(data)

        renderable = self._render_data(data)
        title = f"{label}" if payload_type is None else f"{label} · {payload_type}"
        self.update(Panel(renderable, title=title, border_style="cyan"))

    def _extract_content_text(self, data: Any) -> str:
        """Extract all text content for searching."""
        if not data:
            return ""

        content_parts = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    content_parts.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    try:
                        import json

                        content_parts.append(f"{key}: {json.dumps(value)}")
                    except Exception:
                        content_parts.append(f"{key}: {str(value)}")
                else:
                    content_parts.append(f"{key}: {str(value)}")
        elif isinstance(data, str):
            content_parts.append(data)
        else:
            content_parts.append(str(data))

        return " ".join(content_parts)

    def search_content(self, query: str) -> bool:
        """Search for query in current content. Returns True if found."""
        if not query or not self.current_content:
            return False
        return query.lower() in self.current_content.lower()

    def _is_openhands_step(self, data: Dict[str, Any]) -> bool:
        return _should_render_openhands(self.mode, data)

    def _render_openhands_step(self, data: Dict[str, Any]) -> Any:
        panels: List[Any] = []
        summary = _openhands_step_summary(data) or "OpenHands step"
        panels.append(Panel(Text(summary, no_wrap=False, overflow="fold"), title="summary"))

        action_events = _step_action_events(data)
        observation_events = _step_observation_events(data)
        system_events = [event for event in observation_events if event.get("kind") == "SystemPromptEvent"]
        visible_observations = [event for event in observation_events if event.get("kind") != "SystemPromptEvent"]

        if system_events:
            panels.append(self._render_openhands_setup(system_events))

        if action_events:
            panels.append(self._render_openhands_action_observation_pairs(action_events, visible_observations))
        elif visible_observations:
            panels.append(self._render_openhands_events("observations", visible_observations))

        for key, value in data.items():
            if key in {"action_events", "observation_events"}:
                continue
            panels.append(self._render_key_value(key, value))
        return Group(*panels) if panels else Text("<empty>")

    def _render_openhands_setup(self, events: List[Dict[str, Any]]) -> Panel:
        lines = Text()
        for event in events:
            system_prompt = event.get("system_prompt")
            if isinstance(system_prompt, dict):
                text = system_prompt.get("text")
                tools = event.get("tools")
                if isinstance(text, str):
                    lines.append(f"system prompt: {len(text)} chars\n")
                if isinstance(tools, list):
                    lines.append(f"tools advertised: {len(tools)}\n")
                continue
            lines.append(_openhands_event_summary(event) + "\n")
        return Panel(lines or Text("setup event"), title="setup")

    def _render_openhands_action_observation_pairs(
        self, action_events: List[Dict[str, Any]], observation_events: List[Dict[str, Any]]
    ) -> Panel:
        cards: List[Any] = []
        matched_observation_ids: set[int] = set()

        for index, action in enumerate(action_events, start=1):
            tool_name, arguments = _tool_call_display(action)
            title = f"{index}. {tool_name or _openhands_event_summary(action)}"
            sections: List[Any] = []

            metadata = Text()
            event_id = action.get("id")
            timestamp = action.get("timestamp")
            if event_id or timestamp:
                metadata.append(f"{event_id or ''} {timestamp or ''}\n", style="dim")
            security_risk = action.get("security_risk")
            if security_risk:
                metadata.append(f"security: {security_risk}\n", style="dim")
            summary = action.get("summary")
            if isinstance(summary, str) and summary.strip():
                metadata.append(f"summary: {_shorten_text(summary, 300)}\n")
            if metadata.plain:
                sections.append(metadata)

            thought = _thought_text(action)
            if thought:
                sections.append(Panel(Text(thought, no_wrap=False, overflow="fold"), title="thought"))

            reasoning = _reasoning_text(action)
            if reasoning and reasoning != thought:
                sections.append(Panel(Text(reasoning, no_wrap=False, overflow="fold"), title="reasoning"))

            if arguments not in (None, "", {}):
                sections.extend(self._render_openhands_argument_panels(tool_name, arguments))

            matched = self._matching_observations(action, observation_events)
            for observation in matched:
                matched_observation_ids.add(id(observation))
                sections.append(self._render_openhands_observation_card(observation))

            cards.append(Panel(Group(*sections) if sections else Text("<empty>"), title=title))

        unmatched = [event for event in observation_events if id(event) not in matched_observation_ids]
        if unmatched:
            cards.append(self._render_openhands_events("unmatched observations", unmatched))
        return (
            Panel(Group(*cards), title="actions and observations") if cards else Panel(Text("<empty>"), title="actions")
        )

    def _render_openhands_argument_panels(self, tool_name: str | None, arguments: Any) -> List[Panel]:
        parsed = _parse_json_string(arguments)
        panels: List[Panel] = []

        if isinstance(parsed, dict):
            remaining = dict(parsed)
            for key in ("code", "python", "script"):
                value = remaining.pop(key, None)
                if isinstance(value, str) and value.strip():
                    panels.append(
                        Panel(
                            Syntax(value, "python", word_wrap=True, line_numbers=True),
                            title=f"{tool_name or 'tool'} {key}",
                        )
                    )

            command = remaining.pop("command", None)
            if isinstance(command, str) and command.strip():
                lang = "python" if tool_name == "python_execute" else "bash"
                panels.append(
                    Panel(
                        Syntax(command, lang, word_wrap=True, line_numbers=True),
                        title=f"{tool_name or 'tool'} command",
                    )
                )

            if remaining:
                panels.append(Panel(Syntax(_pretty_json(remaining), "json", word_wrap=True), title="arguments"))
            return panels or [Panel(Text("<empty>"), title="arguments")]

        if isinstance(parsed, str) and tool_name == "python_execute":
            return [Panel(Syntax(parsed, "python", word_wrap=True, line_numbers=True), title="python_execute code")]

        try:
            argument_view: Any = Syntax(_pretty_json(parsed), "json", word_wrap=True)
        except Exception:
            argument_view = Text(str(arguments), no_wrap=False, overflow="fold")
        return [Panel(argument_view, title="arguments")]

    def _matching_observations(
        self, action: Dict[str, Any], observation_events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        action_id = action.get("id")
        tool_call_id = action.get("tool_call_id")
        matches: List[Dict[str, Any]] = []
        for observation in observation_events:
            if tool_call_id and observation.get("tool_call_id") == tool_call_id:
                matches.append(observation)
            elif action_id and observation.get("action_id") == action_id:
                matches.append(observation)
        return matches

    def _render_openhands_condensation_summary(self, event: Dict[str, Any], title: str) -> Panel:
        summary = _condensation_summary_text(event)
        if summary is None:
            return Panel(Text("<empty>"), title=title)

        metadata = Text()
        event_id = event.get("id")
        timestamp = event.get("timestamp")
        if event_id or timestamp:
            metadata.append(f"{event_id or ''} {timestamp or ''}\n", style="dim")
        forgotten_event_ids = event.get("forgotten_event_ids")
        if isinstance(forgotten_event_ids, list):
            metadata.append(f"forgotten events: {len(forgotten_event_ids)}\n", style="dim")
        summary_offset = event.get("summary_offset")
        if summary_offset is not None:
            metadata.append(f"summary offset: {summary_offset}\n", style="dim")

        body: Any = Markdown(summary)
        if metadata.plain:
            body = Group(metadata, body)
        return Panel(body, title=title)

    def _render_openhands_observation_card(self, event: Dict[str, Any]) -> Panel:
        condensation_summary = _condensation_summary_text(event)
        if condensation_summary:
            return self._render_openhands_condensation_summary(event, "condensation summary")

        payload = _observation_payload(event)
        title = _openhands_event_summary(event)
        if payload is not None:
            if isinstance(payload, str):
                payload_view: Any = Text(payload, no_wrap=False, overflow="fold")
            else:
                payload_view = Syntax(_pretty_json(payload), "json", word_wrap=True)
            return Panel(payload_view, title=title)

        lines = Text()
        event_id = event.get("id")
        timestamp = event.get("timestamp")
        if event_id or timestamp:
            lines.append(f"{event_id or ''} {timestamp or ''}\n", style="dim")
        message = _message_text(event)
        if message:
            lines.append(f"message: {_shorten_text(message, 800)}\n")
        return Panel(lines or Text("<empty>"), title=title)

    def _render_openhands_events(self, title: str, events: List[Dict[str, Any]]) -> Panel:
        cards: List[Any] = []
        for index, event in enumerate(events, start=1):
            lines = Text()

            event_id = event.get("id")
            timestamp = event.get("timestamp")
            if event_id or timestamp:
                lines.append(f"{event_id or ''} {timestamp or ''}\n", style="dim")

            condensation_summary = _condensation_summary_text(event)
            if condensation_summary:
                cards.append(self._render_openhands_condensation_summary(event, f"{index}. condensation summary"))
                continue

            thought = _thought_text(event)
            if thought:
                lines.append(f"thought: {thought}\n")

            reasoning = _reasoning_text(event)
            if reasoning and reasoning != thought:
                lines.append(f"reasoning: {reasoning}\n", style="dim")

            message = _message_text(event)
            if message:
                lines.append(f"message: {_shorten_text(message, 800)}\n")

            tool_name, arguments = _tool_call_display(event)
            if tool_name and arguments not in (None, "", {}):
                argument_panels = self._render_openhands_argument_panels(tool_name, arguments)
                body = Group(lines, *argument_panels) if lines.plain else Group(*argument_panels)
                cards.append(Panel(body, title=f"{index}. {tool_name}"))
                continue

            payload = _observation_payload(event)
            if payload is not None:
                if isinstance(payload, str):
                    payload_view: Any = Text(payload, no_wrap=False, overflow="fold")
                else:
                    payload_view = Syntax(_pretty_json(payload), "json", word_wrap=True)
                cards.append(Panel(payload_view, title=f"{index}. {_openhands_event_summary(event)}"))
                continue

            cards.append(Panel(lines or Text("<empty>"), title=f"{index}. {_openhands_event_summary(event)}"))
        return Panel(Group(*cards), title=title) if cards else Panel(Text("<empty>"), title=title)

    def _render_data(self, data: Any) -> Any:
        if isinstance(data, dict):
            if self._is_openhands_step(data):
                return self._render_openhands_step(data)
            # Render as a set of panels for each field to keep things readable
            panels: List[Any] = []
            for key, value in data.items():
                panels.append(self._render_key_value(key, value))
            return Group(*panels) if panels else Text("<empty>")
        elif isinstance(data, list):
            try:
                import json as _json

                dumped = _json.dumps(data, indent=2, ensure_ascii=False)
                return Syntax(dumped, "json", word_wrap=True, line_numbers=False)
            except Exception:
                return Text(str(data), no_wrap=False, overflow="fold")
        elif isinstance(data, str):
            # Default: render Markdown for richer formatting
            try:
                return Markdown(data)
            except Exception:
                return Text(data, no_wrap=False, overflow="fold")
        else:
            # Fallback to JSON renderer
            try:
                return RichJSON.from_data(data)
            except Exception:
                return Text(repr(data))

    def _render_key_value(self, key: str, value: Any) -> Panel:
        # Code-like keys: render with syntax highlighting
        if isinstance(value, str) and key in self.CODE_KEYS_TO_LANG:
            lang = self.CODE_KEYS_TO_LANG[key]
            syn = Syntax(value, lang, word_wrap=True, line_numbers=False)
            return Panel(syn, title=key)
        # Structured data: pretty JSON with wrapping and syntax highlight
        if isinstance(value, (dict, list)):
            try:
                import json as _json

                dumped = _json.dumps(value, indent=2, ensure_ascii=False)
                json_view = Syntax(dumped, "json", word_wrap=True, line_numbers=False)
            except Exception:
                json_view = Text(str(value), no_wrap=False, overflow="fold")
            return Panel(json_view, title=key)
        # Plain scalars or non-code strings: render Markdown when string
        if isinstance(value, str):
            try:
                return Panel(Markdown(value), title=key)
            except Exception:
                return Panel(Text(value, no_wrap=False, overflow="fold"), title=key)
        return Panel(Text(str(value), no_wrap=False, overflow="fold"), title=key)
