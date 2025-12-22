from __future__ import annotations

import asyncio
import json
import queue
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Static, Input, Label
from textual.widgets.tree import TreeNode
from textual.containers import Horizontal, VerticalScroll, Vertical
try:
    from textual.containers import HorizontalScroll  # type: ignore
except Exception:
    # Fallback for older Textual versions without HorizontalScroll
    HorizontalScroll = VerticalScroll  # type: ignore
from textual.binding import Binding
from textual.events import MouseDown, MouseMove, MouseUp

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.json import JSON as RichJSON
from rich.markdown import Markdown

class ClickableResult(Static):
    """A clickable search result item with enhanced styling."""
    
    def __init__(self, text: str, index: int, search_panel: "SearchPanel", result_type: str = "unknown", context: str = "") -> None:
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
            "unknown": ("grey18", "grey48")
        }
        return type_colors.get(result_type, ("grey18", "grey48"))
    
    def _create_rich_content(self, text: str, result_type: str, context: str) -> str:
        """Create clean content with minimal icons."""
        # Simpler icons for different types
        icons = {
            "trajectory": "▶",
            "step": "•",
            "task": "◦",
            "collection": "▼",
            "fork": "↳",
            "unknown": "·"
        }
        
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
    

    
    def update_results(self, results: List[TreeNode[str]], query: str, total_counts: Optional[Dict[str, int]] = None) -> None:
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
                            "unknown": "grey"
                        }
                        header_color = type_colors.get(result_type, "cyan")
                        # Create proper plural forms
                        plurals = {
                            "collection": "Collections",
                            "trajectory": "Trajectories", 
                            "task": "Tasks",
                            "step": "Steps",
                            "fork": "Forks",
                            "unknown": "Unknown"
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
            "unknown": []
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
        if not hasattr(node, 'data') or not node.data:
            return ""
        
        try:
            data = node.data
            if isinstance(data, dict):
                payload = data.get("payload", {})
                if isinstance(payload, dict):
                    # Extract relevant context based on type
                    if "code" in payload:
                        code = payload["code"]
                        if isinstance(code, str):
                            # Get first line of code
                            first_line = code.split('\n')[0].strip()
                            return first_line
                    elif "thought" in payload:
                        thought = payload["thought"]
                        if isinstance(thought, str):
                            # Get first sentence of thought
                            first_sentence = thought.split('.')[0].strip()
                            return first_sentence
                    elif "goal" in payload:
                        goal = payload["goal"]
                        if isinstance(goal, str):
                            return goal
                    elif "output" in payload:
                        output = payload["output"]
                        if isinstance(output, str):
                            # Get first line of output
                            first_line = output.split('\n')[0].strip()
                            return first_line
        except Exception:
            pass
        
        return ""
    
    def _create_results_breakdown(self, grouped_results: Dict[str, List[TreeNode[str]]], total_counts: Optional[Dict[str, int]] = None) -> str:
        """Create a breakdown string of results by type with denominators."""
        # Proper plural forms for count display
        plurals = {
            "collection": "collections",
            "trajectory": "trajectories",  # Correct spelling!
            "task": "tasks",
            "step": "steps", 
            "fork": "forks",
            "unknown": "unknown"
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
                    if hasattr(self.results_container, 'scroll_to_widget'):
                        self.results_container.scroll_to_widget(highlighted_widget)
                    elif hasattr(highlighted_widget, 'scroll_visible'):
                        highlighted_widget.scroll_visible()
                    # Force a refresh of the entire container
                    self.results_container.refresh()
            except Exception:
                pass


class TrajectoryTree(Static):
    def __init__(self) -> None:
        super().__init__()
        self.tree_widget: Optional[Tree[str]] = None
        self.traj_nodes: Dict[str, TreeNode[str]] = {}
        # Map grouping label -> group node to avoid duplicate "unlabeled" nodes
        self.group_nodes: Dict[str, TreeNode[str]] = {}
        # Remember which group label a trajectory belongs to so later events are
        # attached consistently even if they don't repeat collection/process/task.
        self.traj_to_group_label: Dict[str, str] = {}
        # Maintain a single "steps" container per trajectory to avoid duplicates
        self.traj_steps_nodes: Dict[str, TreeNode[str]] = {}
        # Maintain a single "steps" container per trajectory to avoid duplicates
        # Map (trajectory_id, step_index) -> step node to enable focusing/scrolling
        self.step_nodes: Dict[tuple[str, int], TreeNode[str]] = {}
        # Track the latest known reward per trajectory for quick label updates
        self.traj_rewards: Dict[str, float] = {}
        # Track whether a trajectory is finished (finish_message or error_message present)
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
        self.traj_to_group_label.clear()
        self.traj_steps_nodes.clear()
        self.step_nodes.clear()
        self.traj_rewards.clear()
        self.traj_finished.clear()
        self.expanded_trajs.clear()
        self.bulk_loading = False
        self.search_results.clear()
        self.current_search_index = -1
        self.current_search_query = ""

    def compose(self) -> ComposeResult:  # type: ignore[override]
        self.tree_widget = Tree("Trajectory Collections")
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
            if hasattr(node, 'data') and node.data:
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
        
        counts: Dict[str, int] = {
            "collection": 0,
            "trajectory": 0,
            "task": 0,
            "step": 0,
            "fork": 0,
            "unknown": 0
        }
        
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
                group_node = self.tree_widget.root.add(label)
                group_node.expand()
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
            # Initialize finished status from payload if present
            try:
                finish_msg = traj.get("finish_message")
                error_msg = traj.get("error_message")
                self.traj_finished[traj_id] = bool(finish_msg) or bool(error_msg)
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
            node = self.ensure_traj_node(traj_id, parent=group_node, expand=not self.bulk_loading)
            task_label = f"task: {task.get('goal')}" if task else "task: None"
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
                # Update finished map if we received any terminal info
                if finish_msg is not None or error_msg is not None:
                    self.traj_finished[traj_id] = bool(finish_msg) or bool(error_msg)
            except Exception:
                pass
            # Update the trajectory node label to reflect latest reward
            self._set_node_label(node, self._format_traj_label(traj_id))
            # Keep payload's reward in sync if present so DetailsPanel shows latest
            try:
                if isinstance(node.data, dict) and isinstance(node.data.get("payload"), dict):
                    node.data["payload"]["reward"] = self.traj_rewards.get(traj_id, node.data["payload"].get("reward", 0.0))
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
                self.traj_finished[traj_id] = bool(finish_msg) or bool(error_msg)
            except Exception:
                pass
            # Refresh label to reflect final reward and status
            self._set_node_label(node, self._format_traj_label(traj_id))

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
    ) -> None:
        super().__init__()
        self.event_queue = event_queue
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.jsonl_paths = [Path(p) for p in jsonl_paths] if jsonl_paths else None
        self.tree_widget = TrajectoryTree()
        self.details_panel = DetailsPanel()
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
        self._left_container: Optional[HorizontalScroll] = None
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
        if hasattr(self.tree_widget, 'current_search_index'):
            self.search_panel.highlight_result(self.tree_widget.current_search_index)
    
    def action_prev_search(self) -> None:
        """Navigate to previous search result."""
        self.tree_widget.previous_search_result()
        # Update highlight in search panel
        if hasattr(self.tree_widget, 'current_search_index'):
            self.search_panel.highlight_result(self.tree_widget.current_search_index)
    
    def perform_search(self, query: str) -> None:
        """Perform search in tree widget and update search panel."""
        if query:
            results = self.tree_widget.perform_search(query)
            total_counts = self.tree_widget.get_total_counts_by_type()
            self.search_panel.update_results(results, query, total_counts)
            # Highlight the first result if any results found
            if results and hasattr(self.tree_widget, 'current_search_index'):
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
        records.sort(key=lambda r: r.get("ts", 0.0))
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
        records.sort(key=lambda r: r.get("ts", 0.0))
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
        records.sort(key=lambda r: r.get("ts", 0.0))
        # For multi-file replay we can just stream sorted records
        for record in records:
            await self._handle_record(record)
            if delay > 0:
                await asyncio.sleep(delay)

    async def _replay_jsonls(self, paths: List[Path]) -> None:
        """Replay events from multiple JSONL files, ordered by timestamp, starting paused."""
        delay = float(self.replay_delay or 0.0)
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
        all_records.sort(key=lambda r: r.get("ts", 0.0))
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
        all_records.sort(key=lambda r: r.get("ts", 0.0))
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


def run_viewer_from_queue(event_queue: queue.Queue) -> None:
    app = TrajectoryViewer(event_queue=event_queue)
    app.run()


def run_viewer_from_jsonl(path: str | Path, *, start_at_end: bool = False) -> None:
    """Launch a TrajectoryViewer for a single JSONL file.

    By default, the viewer starts reading from the **beginning** of the file so that
    you can inspect previously-recorded events.  Set ``start_at_end=True`` to mimic
    a *tail -f* style live view that ignores existing lines and only shows new
    events appended after the viewer starts.
    """
    app = TrajectoryViewer(jsonl_path=path, start_at_end=start_at_end)
    app.run()


def run_viewer_from_jsonls(paths: List[str | Path], *, start_at_end: bool = False) -> None:
    """Launch a TrajectoryViewer that tails multiple JSONL files in parallel."""
    app = TrajectoryViewer(jsonl_paths=paths, start_at_end=start_at_end)
    app.run()

def run_replay_from_jsonl(path: str | Path, *, delay: float = 0.5) -> None:
    app = TrajectoryViewer(jsonl_path=path, start_at_end=False, replay_delay=delay)
    app.run()

def run_replay_from_jsonls(paths: List[str | Path], *, delay: float = 0.5) -> None:
    app = TrajectoryViewer(jsonl_paths=paths, start_at_end=False, replay_delay=delay)
    app.run()


class DetailsPanel(Static):
    """Renders the details of the selected node with minimal clutter.

    - For step payloads (dicts), shows key panels; code fields are syntax highlighted.
    - For other payloads, pretty-prints JSON when possible.
    """
    
    def __init__(self) -> None:
        super().__init__()
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

    def _render_data(self, data: Any) -> Any:
        if isinstance(data, dict):
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

