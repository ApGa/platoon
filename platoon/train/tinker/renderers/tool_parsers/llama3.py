"""``llama3``: the JSON tool calls of Llama 3.1, 3.2 and 3.3 (vLLM's ``llama3_json``).

::

    {"name": NAME, "parameters": {...}}

The call is the whole message, or follows ``<|python_tag|>``; several calls may follow one another, separated by
``;``.  ``arguments`` is read as ``parameters``.  A message that does not start with a JSON object (after any text
before ``<|python_tag|>``) is content, and so is a JSON object without both a ``name`` and ``parameters`` (or
``arguments``), as vLLM's ``llama3_json`` reads it: ``{"name": "John Smith", "age": 42}`` is an answer.  JSON that
starts like a call (``{"name":``) but does not decode is an unparsed call, from there to the end of the message.

Lenient rules, which mark the call recovered: raw control characters (newlines) inside a JSON string, and arguments
written as a JSON string that holds the object.
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

PYTHON_TAG = "<|python_tag|>"
_CALL_LIKE = re.compile(r"\{\s*\"name\"\s*:")
_SEPARATOR = re.compile(r"\s*[;,]?\s*")


class Llama3Parser(ToolCallParser):
    name = "llama3"
    aliases = ("llama3_json", "llama4_json")
    call_start = re.compile(re.escape(PYTHON_TAG))

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        tag = text.find(PYTHON_TAG)
        if tag != -1:
            out.text(text[:tag])
            pos = tag + len(PYTHON_TAG)
        else:
            pos = 0
        pos = skip_whitespace(text, pos)
        if not text.startswith("{", pos):
            out.text(text[pos:] if tag != -1 else text)
            return out.result()
        while pos < len(text) and text.startswith("{", pos):
            try:
                value, end, lenient = json_value(text, pos)
            except ValueError as e:
                if _CALL_LIKE.match(text, pos) or out.calls:
                    out.fail(text[pos:], f"the tool call is not valid JSON: {e}")
                else:
                    out.text(text[pos:])
                return out.result()
            if not (isinstance(value, dict) and "name" in value and ("parameters" in value or "arguments" in value)):
                out.text(text[pos:end])  # a JSON answer, not a call
            else:
                try:
                    name, arguments, _, decoded = json_call(value)
                except ValueError as e:
                    out.fail(text[pos:end], f"the tool call is not a call object: {e}")
                else:
                    out.call(name, arguments, recovered=lenient or decoded)
            match = _SEPARATOR.match(text, end)
            pos = match.end() if match else end
        out.text(text[pos:])
        return out.result()
