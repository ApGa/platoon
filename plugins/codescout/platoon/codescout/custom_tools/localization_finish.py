"""Custom finish tool for code localization tasks.

This tool allows the agent to submit localization results in a structured format where:
- File path is required
- Class name is optional
- Function name is optional
"""

import sys

# This module lives at two importable paths simultaneously:
#   - "platoon.codescout.custom_tools.localization_finish"  (via the platoon package)
#   - "custom_tools.localization_finish"                    (via the .pth entry for plugins/codescout)
#
# Python treats these as two separate modules and will load the file twice,
# producing two distinct class objects that both claim __module__ =
# "custom_tools.localization_finish" after our __module__ override.
# Pydantic then sees two Action/Observation subclasses with the same name
# and raises "Duplicate class definition".
#
# Fix: whichever path loads us first, immediately register both keys in
# sys.modules pointing to the SAME module object, so the second import
# finds the cached version instead of re-executing the file.
_SHORT = "custom_tools.localization_finish"
_LONG  = "platoon.codescout.custom_tools.localization_finish"
_PLUGINS = "plugins.codescout.platoon.codescout.custom_tools.localization_finish"
_SHORT_PKG = "custom_tools"
_LONG_PKG  = "platoon.codescout.custom_tools"

_this_module = sys.modules[__name__]
if _SHORT not in sys.modules:
    sys.modules[_SHORT] = _this_module
if _LONG not in sys.modules:
    sys.modules[_LONG] = _this_module
if _PLUGINS not in sys.modules:
    sys.modules[_PLUGINS] = _this_module
if _SHORT_PKG not in sys.modules and _LONG_PKG in sys.modules:
    sys.modules[_SHORT_PKG] = sys.modules[_LONG_PKG]

import json
from typing import TYPE_CHECKING
from collections.abc import Sequence

from pydantic import BaseModel, Field, computed_field
from rich.text import Text

from openhands.sdk import (
    Action,
    Observation,
    ToolDefinition
)
from openhands.sdk.tool import ToolExecutor, ToolAnnotations, register_tool

from openhands.sdk.conversation.state import ConversationExecutionStatus

if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation

class CodeLocation(BaseModel):
    """A single code location with optional class and function."""

    file: str = Field(description="Path to the file (required)")
    class_name: str | None = Field(default=None, description="Class name (optional)")
    function_name: str | None = Field(default=None, description="Function/method name (optional)")

class LocalizationFinishAction(Action):
    """Action for submitting final localization results."""

    locations: list[CodeLocation] = Field(
        description="""List of code locations to modify. Each location in this list must have:
- file: Path to the file relative to the repository root (required)
- class_name: Class name (optional, omit for changes to imports, global variables, and global functions)
- function_name: Function/method name (optional, omit for changes that edit parts of a file outside of any particular function)
"""
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this action."""
        content = Text()
        content.append("Submitting localization results:\n", style="bold blue")
        content.append(f"Found {len(self.locations)} location(s):\n", style="green")
        for i, loc in enumerate(self.locations, 1):
            content.append(f"  {i}. {loc.file}", style="cyan")
            if loc.class_name:
                content.append(f" → {loc.class_name}", style="yellow")
            if loc.function_name:
                content.append(f".{loc.function_name}", style="magenta")
            content.append("\n")
        return content

# Override __module__ so the client-side pydantic discriminator registry uses the
# same module path as the server container ("custom_tools.localization_finish").
# This prevents a "Duplicate class definition" error when the server sends events
# back and the client tries to deserialize them — both sides see the same name.
LocalizationFinishAction.__module__ = "custom_tools.localization_finish"


class LocalizationFinishObservation(Observation):
    """Observation returned after submitting localization results. No observation is needed since the agent will exit after this action."""

    @property
    def visualize(self) -> Text:
        """Return an empty Text representation since the message is in the action."""
        return Text()

LocalizationFinishObservation.__module__ = "custom_tools.localization_finish"
    
def locations_to_dict_list(locations: list[CodeLocation]) -> list[dict]:
    """Convert CodeLocation objects to dictionary format.

    Args:
        locations: List of CodeLocation objects

    Returns:
        List of dictionaries with 'file', 'class_name', 'function_name' keys
    """
    return [
        {
            "file": loc.file,
            "class_name": loc.class_name,
            "function_name": loc.function_name,
        }
        for loc in locations
    ]

class LocalizationFinishExecutor(ToolExecutor):
    def __call__(
        self,
        action: LocalizationFinishAction,
        conversation: "BaseConversation | None" = None,  # noqa: ARG002
    ) -> LocalizationFinishObservation:
        try:
            loc_dict = locations_to_dict_list(action.locations)
            text = json.dumps(loc_dict, indent=2)
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
            return LocalizationFinishObservation.from_text(text=text)
        except Exception as _:
            return LocalizationFinishObservation.from_text(text="")

TOOL_DESCRIPTION = """Submit your final code localization results.

Use this tool when you have identified all relevant files, classes, and functions that need to be modified to address the issue described in the problem statement.

Provide a structured list of locations. Each location must have:
- file: Path to the file relative to the root of the repository (required)
- class_name: Class name (optional)
- function_name: Function/method name (optional)

You must submit a list of locations that require modification and for each location you must follow the below rules in your output:
1. If the required modifications belong to a specific function that belongs to a class, provide the file path, class name, and function name.
2. If the required modification belongs to a function that is not part of any class, provide the file path and function name.
3. If the required modification does not belong to any specific class or a function (e.g. global variables, imports, new class, new global function etc.), it is sufficient to provide only the file path.
4. If the required modification belongs to a class (e.g. adding a new method to a class, changing the class inheritance), provide the file path and class name. If you are modifying the __init__ method of a class, you should provide the function name as well.

IMPORTANT:
1. If multiple different edits need to be edited in the same file, you should create separate entries for each edit, specifying the same file path but different class/function names as applicable. Each entry should compulsorily include the file path.
2. Do NOT include duplicate entries in your output for which the file, class, and function names are all identical.
3. Ensure that the file paths are accurate and relative to the root of the repository without any leading "./" or "/". All locations must be valid and exist in the codebase and this applies to class and function names as well.
4. Aim for high precision (all returned locations are relevant) and high recall (no relevant locations missed).
5. The agent will terminate its execution after you call this tool.
"""

class LocalizationFinishTool(ToolDefinition[LocalizationFinishAction, LocalizationFinishObservation]):
    """Tool for submitting final localization results."""

    """Tool for submitting final code localization results."""

    @classmethod
    def create(
        cls,
        conv_state, # noqa: ARG003
        **params
    ) -> Sequence["LocalizationFinishTool"]:
        """Create LocalizationFinishTool instance.

        Args:
            conv_state: Conversation state (provides workspace info)
            workspace_dir: Optional workspace directory override
            **params: Additional parameters

        Returns:
            A sequence containing a single LocalizationFinishTool instance.
        """
        if params:
            raise ValueError("LocalizationFinishTool doesn't accept parameters")
        
        return [
            cls(
                action_type=LocalizationFinishAction,
                observation_type=LocalizationFinishObservation,
                description=TOOL_DESCRIPTION,
                executor=LocalizationFinishExecutor(),
                annotations=ToolAnnotations(
                    title="localization_finish",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]

# Override __module__ on all classes so both client and server use the same
# module path ("custom_tools.localization_finish") in every registry:
# - the pydantic discriminator registry (prevents "Duplicate class definition")
# - the tool registry _MODULE_QUALNAMES (tells the server what to import)
LocalizationFinishTool.__module__ = "custom_tools.localization_finish"

register_tool("LocalizationFinishTool", LocalizationFinishTool)