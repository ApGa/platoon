"""``hermes``: JSON tool calls inside ``<tool_call>`` tags (Qwen2.5, Qwen3 and Qwen3-2507, Hermes models).

::

    <tool_call>
    {"name": NAME, "arguments": {...}}
    </tool_call>

Each block holds one JSON call object (or a JSON list of them); ``parameters`` is read as ``arguments``.  A block
starts at a ``<tool_call>`` followed by JSON, or at one that starts a line; elsewhere the tag is just text.  The
JSON is decoded as a whole, so a string inside it may hold a literal ``</tool_call>``.  A block whose JSON does not
decode to call objects is an unparsed call on its own, and the response's other calls still parse.

Lenient rules, which mark the call recovered: several call objects one after another in one block (JSON after
them that does not decode is an unparsed call of its own), a missing ``</tool_call>`` after the JSON (text between
the JSON and a late ``</tool_call>`` goes to the content), raw control characters (newlines) inside a JSON string,
and arguments written as a JSON string that holds the object.
"""

from __future__ import annotations

import re

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import (
    ParseResult,
    ResultBuilder,
    ToolCallParser,
    json_call,
    json_value,
    skip_whitespace,
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
_REAL_START = re.compile(r"<tool_call>\s*[\[{]")


class HermesParser(ToolCallParser):
    name = "hermes"
    aliases = ("qwen25", "qwen")
    call_start = _REAL_START

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        pos = 0
        while True:
            start = _next_block(text, pos)
            if start is None:
                out.text(text[pos:])
                return out.result()
            out.text(text[pos:start])
            pos = _block(text, start, out)


def _next_block(text: str, pos: int) -> int | None:
    while (start := text.find(TOOL_CALL_OPEN, pos)) != -1:
        line_start = text.rfind("\n", 0, start) + 1
        if _REAL_START.match(text, start) or not text[line_start:start].strip():
            return start
        pos = start + 1
    return None


def _block(text: str, start: int, out: ResultBuilder) -> int:
    """Parse the block at ``start`` into ``out``; returns where it ends."""

    body = start + len(TOOL_CALL_OPEN)
    try:
        value, end, lenient = json_value(text, body)
    except ValueError as e:
        block_end = _close(text, body)
        out.fail(text[start:block_end], f"the tool call is not valid JSON: {e}")
        return block_end
    values, broken = [value], None
    after = skip_whitespace(text, end)
    while broken is None and text.startswith(("{", "["), after):  # more JSON: the format has one call per block
        try:
            value, end, _ = json_value(text, after)
        except ValueError as e:
            broken = f"the tool call is not valid JSON: {e}"
        else:
            values.append(value)
            lenient = True
            after = skip_whitespace(text, end)
    trailing = ""  # text between the JSON and a late </tool_call>
    if text.startswith(TOOL_CALL_CLOSE, after):
        block_end, recovered = after + len(TOOL_CALL_CLOSE), lenient
    elif broken is not None:  # JSON that does not decode after the calls: a broken call, up to its </tool_call>
        block_end, recovered = _close(text, after), True
    else:
        block_end, recovered = _close(text, end, missing=end), True  # </tool_call> is missing, or late
        trailing = text[end : max(end, block_end - len(TOOL_CALL_CLOSE))]
    calls = []
    try:
        for value in values:
            for item in value if isinstance(value, list) and value else [value]:
                calls.append(json_call(item))
    except ValueError as e:
        out.fail(text[start:block_end], f"the tool call is not a call object: {e}")
        return block_end
    for name, arguments, _, decoded in calls:
        out.call(name, arguments, recovered=recovered or decoded)
    if broken is not None:
        out.fail(text[after:block_end], broken)
    out.text(trailing)
    return block_end


def _close(text: str, pos: int, *, missing: int | None = None) -> int:
    """Where a block ends: after the first ``</tool_call>`` from ``pos`` on, unless the next block starts before it;
    then at that block (or the end), or at ``missing`` when it is given."""

    following = _next_block(text, pos)
    limit = len(text) if following is None else following
    close = text.find(TOOL_CALL_CLOSE, pos, limit)
    if close != -1:
        return close + len(TOOL_CALL_CLOSE)
    return limit if missing is None else missing
