"""Generic ``bash``, ``view``, ``write`` and ``edit`` tools that act on a sandbox.

They are the env tools of a sandboxed session, for harnesses that do not act on
task state natively. Relative paths start at the sandbox workdir.
"""

from __future__ import annotations

import posixpath
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from platoon.envs.service import ToolResult, ToolSpec
from platoon.sandbox.base import Sandbox
from platoon.utils import scratch

DEFAULT_COMMAND_TIMEOUT_S = 300.0
DEFAULT_MAX_OUTPUT_CHARS = 30_000
_EDIT_CONTEXT_LINES = 4

# Prints "dir" or "file" before the listing or contents, so one exec tells them apart.
_VIEW_SCRIPT = (
    'if [ -d "$1" ]; then echo dir; cd "$1" && find . -maxdepth 2 -not -path "*/.*" | sort; '
    'elif [ -f "$1" ]; then echo file; cat -- "$1"; '
    'else echo "No such file or directory: $1" >&2; exit 2; fi'
)
_WRITE_SCRIPT = 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"'

TOOL_SPECS = [
    ToolSpec(
        name="bash",
        description="Run a bash command in the task environment and return its output and exit code.",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to run."}},
            "required": ["command"],
        },
    ),
    ToolSpec(
        name="view",
        description="Show a file with line numbers, or list a directory two levels deep.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File or directory path."},
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional [start, end] line numbers (1-based, inclusive); end -1 means the end.",
                },
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        name="write",
        description="Create or overwrite a file with the given content, creating parent directories.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    ToolSpec(
        name="edit",
        description="Replace the only occurrence of old_str in a file with new_str.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}},
            "required": ["path", "old_str", "new_str"],
        },
    ),
]


class _ToolError(Exception):
    """A tool call that could not do what it was asked; reported to the model as an error result."""


class SandboxTools:
    """Dispatches tool calls to the sandbox; failures come back as error results, never exceptions."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        workdir: str,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._sandbox = sandbox
        self._workdir = workdir
        self._command_timeout_s = command_timeout_s
        self._max_output_chars = max_output_chars
        self._handlers: dict[str, Callable[[dict[str, Any]], Awaitable[str]]] = {
            "bash": self._bash,
            "view": self._view,
            "write": self._write,
            "edit": self._edit,
        }

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult.text(f"Unknown tool {name!r}; available: {sorted(self._handlers)}", is_error=True)
        try:
            return ToolResult.text(self._truncate(await handler(arguments)))
        except _ToolError as exc:
            return ToolResult.text(str(exc), is_error=True)
        except Exception as exc:  # the sandbox failed, not the model
            return ToolResult.text(f"The sandbox failed: {type(exc).__name__}: {exc}", is_error=True)

    async def _bash(self, arguments: dict[str, Any]) -> str:
        result = await self._sandbox.exec(_string(arguments, "command"), timeout_s=self._command_timeout_s)
        output = result.stdout + (f"\n[stderr]\n{result.stderr}" if result.stderr else "") or "[no output]"
        if result.timed_out:
            return f"{output}\n[command timed out after {self._command_timeout_s:g} s]"
        return f"{output}\n[exit code {result.exit_code}]" if result.exit_code else output

    async def _view(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        view_range = arguments.get("view_range")
        result = await self._sandbox.exec(["bash", "-c", _VIEW_SCRIPT, "view", path])
        if result.exit_code != 0:
            raise _ToolError(result.stderr.strip() or f"Cannot view {path}")
        kind, _, body = result.stdout.partition("\n")
        if kind == "dir":
            return f"{path}:\n{body}"
        lines = body.splitlines()
        if not lines:
            return f"{path} is empty"
        start, end = 1, len(lines)
        if view_range is not None:
            if not (isinstance(view_range, list) and len(view_range) == 2 and all(type(n) is int for n in view_range)):
                raise _ToolError("view_range must be [start, end] line numbers")
            start, end = view_range[0], len(lines) if view_range[1] == -1 else view_range[1]
            if not 1 <= start <= max(end, 1) or end > len(lines):
                raise _ToolError(f"view_range {view_range} is outside the file's {len(lines)} lines")
        return "\n".join(f"{number:6}\t{lines[number - 1]}" for number in range(start, end + 1))

    async def _write(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        content = _string(arguments, "content")
        await self._write_file(path, content)
        return f"Wrote {len(content)} characters to {path}"

    async def _edit(self, arguments: dict[str, Any]) -> str:
        path = self._path(arguments)
        old, new = _string(arguments, "old_str"), _string(arguments, "new_str")
        if not old:
            raise _ToolError("old_str must not be empty")
        original = await self._read_file(path)
        occurrences = original.count(old)
        if occurrences != 1:
            raise _ToolError(f"old_str must occur exactly once in {path}; found {occurrences} occurrences")
        content = original.replace(old, new)
        await self._write_file(path, content)
        first = original[: original.index(old)].count("\n")
        lines = content.splitlines()
        last = first + new.count("\n")
        window = range(max(0, first - _EDIT_CONTEXT_LINES), min(len(lines), last + 1 + _EDIT_CONTEXT_LINES))
        snippet = "\n".join(f"{number + 1:6}\t{lines[number]}" for number in window)
        return f"Edited {path}:\n{snippet}"

    async def _read_file(self, path: str) -> str:
        # A download copies exact bytes, unlike exec output, which is decoded leniently and capped.
        with scratch.temporary_directory() as local_dir:
            local = Path(local_dir) / "file"
            try:
                await self._sandbox.download(path, local)
                return local.read_bytes().decode("utf-8")
            except FileNotFoundError:
                raise _ToolError(f"No such file: {path}") from None
            except (IsADirectoryError, UnicodeDecodeError):
                raise _ToolError(f"{path} is not a UTF-8 text file") from None

    async def _write_file(self, path: str, content: str) -> None:
        # Writing through the shell keeps an existing file's mode, which an upload would replace.
        result = await self._sandbox.exec(["bash", "-c", _WRITE_SCRIPT, "write", path], stdin=content.encode())
        if result.exit_code != 0:
            raise _ToolError(result.stderr.strip() or f"Cannot write {path}")

    def _path(self, arguments: dict[str, Any]) -> str:
        return posixpath.join(self._workdir, _string(arguments, "path"))

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        half = self._max_output_chars // 2
        return f"{text[:half]}\n[... {len(text) - 2 * half} characters truncated ...]\n{text[-half:]}"


def _string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise _ToolError(f"Argument {key!r} must be a string")
    return value
