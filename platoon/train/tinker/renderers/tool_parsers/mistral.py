"""``mistral``: Mistral's tool calls, in both of its tokenizer formats.

::

    [TOOL_CALLS][{"name": NAME, "arguments": {...}, "id": "a1b2c3d4e"}, ...]      (tokenizer v3 and v7)
    [TOOL_CALLS]NAME[ARGS]{...}[TOOL_CALLS]NAME[ARGS]{...}                        (tokenizer v11 and later)
    [TOOL_CALLS]NAME[CALL_ID]a1b2c3d4e[ARGS]{...}

Text before the first ``[TOOL_CALLS]`` is content.  Reasoning is between ``[THINK]`` and ``[/THINK]``, as Magistral
writes it; a response that opens with ``<think>`` instead is read with those tags.  Mistral's chat templates only
accept tool-call ids of nine letters and digits, so a call keeps the id it was sampled with when it is one, and
otherwise gets a new one of that shape.

A call whose JSON does not decode is an unparsed call, from its ``[TOOL_CALLS]`` (in a JSON list, from that item on)
to the next ``[TOOL_CALLS]`` or the end.  Lenient rules, which mark the call recovered: raw control characters
(newlines) inside a JSON string, and arguments written as a JSON string that holds the object.
"""

from __future__ import annotations

import re
import secrets
import string

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import (
    THINK_TAGS,
    ParseResult,
    ResultBuilder,
    ToolCallParser,
    json_call,
    json_value,
    skip_whitespace,
)

TOOL_CALLS = "[TOOL_CALLS]"
ARGS = "[ARGS]"
CALL_ID = "[CALL_ID]"
_ID = re.compile(r"[A-Za-z0-9]{9}")
_ID_ALPHABET = string.ascii_letters + string.digits


class MistralParser(ToolCallParser):
    name = "mistral"
    default_reasoning_tags = ("[THINK]", "[/THINK]")
    call_start = re.compile(re.escape(TOOL_CALLS))

    def split_reasoning(self, text: str, *, in_reasoning: bool = False) -> tuple[str, str]:
        default_tags = self.reasoning_tags == self.default_reasoning_tags
        if not in_reasoning and default_tags and text.lstrip().startswith(THINK_TAGS[0]):
            return ToolCallParser(reasoning_tags=THINK_TAGS).split_reasoning(text)
        return super().split_reasoning(text, in_reasoning=in_reasoning)

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        start = text.find(TOOL_CALLS)
        if start == -1:
            out.text(text)
            return out.result()
        out.text(text[:start])
        pos = start
        while pos < len(text):
            if not text.startswith(TOOL_CALLS, pos):
                following = text.find(TOOL_CALLS, pos)
                end = len(text) if following == -1 else following
                out.text(text[pos:end])
                pos = end
                continue
            body = skip_whitespace(text, pos + len(TOOL_CALLS))
            if text.startswith("[", body):
                pos = _json_list(text, pos, body, out)
            else:
                pos = _compact(text, pos, body, out)
        return out.result()


def _next(text: str, pos: int) -> int:
    following = text.find(TOOL_CALLS, pos)
    return len(text) if following == -1 else following


def _json_list(text: str, start: int, body: int, out: ResultBuilder) -> int:
    """``[TOOL_CALLS][{...}, ...]`` at ``start``, item by item, so the items before one that does not parse still
    parse; that item and the rest of the list are one unparsed call.  Returns where the list ends."""

    pos, parsed = body + 1, 0
    while True:
        item = skip_whitespace(text, pos)
        if text.startswith("]", item) and parsed:
            return item + 1
        try:
            value, end, lenient = json_value(text, item)
            name, arguments, call_id, decoded = json_call(value)
        except ValueError as e:
            limit = _next(text, item)
            out.fail(text[start if not parsed else item : limit], f"the tool call is not valid JSON: {e}")
            return limit
        out.call(name, arguments, id=_mistral_id(call_id), recovered=lenient or decoded)
        parsed += 1
        pos = skip_whitespace(text, end)
        if text.startswith(",", pos):
            pos += 1
        elif not text.startswith("]", pos):
            limit = _next(text, pos)
            out.fail(text[pos:limit], "the list of tool calls is not closed")
            return limit


def _compact(text: str, start: int, body: int, out: ResultBuilder) -> int:
    """``[TOOL_CALLS]NAME[CALL_ID]ID[ARGS]{...}`` at ``start``; returns where it ends."""

    limit = _next(text, body)
    args = text.find(ARGS, body, limit)
    if args == -1:
        out.fail(text[start:limit], "the tool call has no [ARGS]")
        return limit
    name, call_id = text[body:args], None
    if CALL_ID in name:
        name, call_id = name.split(CALL_ID, 1)
        call_id = call_id.strip()
    name = name.strip()
    try:
        if not name:
            raise ValueError("the tool call has no function name")
        value, end, lenient = json_value(text, args + len(ARGS))
        name, arguments, _, decoded = json_call({"name": name, "arguments": value})
    except ValueError as e:
        out.fail(text[start:limit], f"the tool call is not valid: {e}")
        return limit
    out.call(name, arguments, id=_mistral_id(call_id), recovered=lenient or decoded)
    return end


def _mistral_id(call_id: str | None) -> str:
    """The sampled id when it has Mistral's shape (nine letters and digits), else a new one."""

    if call_id is not None and _ID.fullmatch(call_id):
        return call_id
    return "".join(secrets.choice(_ID_ALPHABET) for _ in range(9))
