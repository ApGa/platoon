"""Exercise actual FastMCP argument validation and default injection."""
import json

import pytest

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from platoon.openreward.mcp_bridge import _make_environment_tool


@pytest.mark.asyncio
async def test_fastmcp_preserves_environment_defaults_and_omitted_deletion_text():
    calls = []

    class Runtime:
        def call_openreward_tool(self, name, arguments):
            calls.append((name, arguments))
            if name == "str_replace":
                return {"text": "remove this".replace(arguments["old_str"], arguments["new_str"])}
            return {"text": "ok"}

    server = FastMCP("environment-defaults")
    schemas = {
        "bash": {
            "command": {"type": "string"},
            "description": {"type": "string", "default": ""},
            "limit": {"type": "integer", "default": 0},
            "verbose": {"type": "boolean", "default": False},
            "tags": {"type": "array", "items": {"type": "string"}, "default": []},
            "timeout": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "str_replace": {
            "old_str": {"type": "string"},
            "new_str": {"type": "string", "default": ""},
            "description": {"type": "string", "default": ""},
        },
    }
    for name, properties in schemas.items():
        server.add_tool(_make_environment_tool(Runtime(), {
            "name": name,
            "parameters": {"type": "object", "properties": properties,
                           "required": ["command" if name == "bash" else "old_str"]},
        }))
    await server.call_tool("bash", {"command": "echo ok"})
    result = await server.call_tool("str_replace", {"old_str": "remove "})
    assert calls == [
        ("bash", {"command": "echo ok", "description": "", "limit": 0,
                  "verbose": False, "tags": [], "timeout": None}),
        ("str_replace", {"old_str": "remove ", "new_str": "", "description": ""}),
    ]
    content, _ = result
    assert any(json.loads(block.text) == {"text": "this"} for block in content)


@pytest.mark.asyncio
async def test_fastmcp_does_not_fill_a_required_argument_from_its_schema_default():
    class Runtime:
        def call_openreward_tool(self, name, arguments):
            raise AssertionError("Missing required command must never reach the environment")

    server = FastMCP("required-environment-argument")
    server.add_tool(_make_environment_tool(Runtime(), {
        "name": "bash",
        "parameters": {"type": "object", "required": ["command"],
                       "properties": {"command": {"type": "string", "default": "echo wrong"}}},
    }))
    with pytest.raises(ToolError, match="command.*|Field required"):
        await server.call_tool("bash", {})
