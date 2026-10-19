"""``kimi_k2``: the tool calls of Kimi K2 and K2.5.

::

    <|tool_calls_section_begin|><|tool_call_begin|>functions.NAME:N<|tool_call_argument_begin|>{...}<|tool_call_end|>
    <|tool_calls_section_end|>

The function name is in the call id (``functions.NAME:N``, or ``NAME:N``), and the call keeps that id: Kimi's chat
template writes it back into the conversation, so a new one would change what the model sees.  A call whose id
holds no name (a bare counter) is an unparsed call, as is one whose JSON does not decode; each runs from its
``<|tool_call_begin|>`` to its ``<|tool_call_end|>``, else to the next call or the end.  Text outside the section is
content; reasoning is between ``<think>`` and ``</think>``.  Lenient rules, which mark the call recovered: a missing
``<|tool_call_end|>``, raw control characters inside a JSON string, and arguments written as a JSON string that holds
the object.

The id shape follows SGLang's ``kimi_k2`` detector (Apache-2.0; see ``NOTICE`` in this directory); the code is
written for platoon.
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

SECTION_BEGIN = "<|tool_calls_section_begin|>"
SECTION_END = "<|tool_calls_section_end|>"
CALL_BEGIN = "<|tool_call_begin|>"
ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
CALL_END = "<|tool_call_end|>"
_CALL_ID = re.compile(r"(?:functions\.)?(?P<name>[\w.\-]+):\d+")


class KimiK2Parser(ToolCallParser):
    name = "kimi_k2"
    call_start = re.compile(f"{re.escape(SECTION_BEGIN)}|{re.escape(CALL_BEGIN)}")

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        pos = 0
        while True:
            start = text.find(CALL_BEGIN, pos)
            if start == -1:
                out.text(_without_section_tags(text[pos:]))
                return out.result()
            out.text(_without_section_tags(text[pos:start]))
            body = start + len(CALL_BEGIN)
            following = text.find(CALL_BEGIN, body)
            limit = len(text) if following == -1 else following
            try:
                call_id, name, arguments, end, recovered = _call(text, body, limit)
            except ValueError as e:
                close = text.find(CALL_END, body, limit)
                end = limit if close == -1 else close + len(CALL_END)
                out.fail(text[start:end], f"the tool call is not valid: {e}")
            else:
                out.call(name, arguments, id=call_id, recovered=recovered)
            pos = end


def _call(text: str, body: int, limit: int) -> tuple[str, str, dict, int, bool]:
    arguments_at = text.find(ARGUMENT_BEGIN, body, limit)
    if arguments_at == -1:
        raise ValueError("no <|tool_call_argument_begin|>")
    call_id = text[body:arguments_at].strip()
    match = _CALL_ID.fullmatch(call_id)
    if match is None:
        raise ValueError(f"the call id {call_id!r} holds no function name")
    value, end, lenient = json_value(text, arguments_at + len(ARGUMENT_BEGIN))
    name, arguments, _, decoded = json_call({"name": match["name"], "arguments": value})
    after = skip_whitespace(text, end)
    if text.startswith(CALL_END, after):
        return call_id, name, arguments, after + len(CALL_END), lenient or decoded
    if after >= limit or text.startswith(SECTION_END, after):
        return call_id, name, arguments, end, True  # <|tool_call_end|> is missing
    raise ValueError(f"unexpected text after the arguments: {text[after : after + 30]!r}")


def _without_section_tags(text: str) -> str:
    return text.replace(SECTION_BEGIN, "").replace(SECTION_END, "")
