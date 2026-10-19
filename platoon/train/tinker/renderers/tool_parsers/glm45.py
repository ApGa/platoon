"""``glm45``: the tool calls of GLM-4.5 and later (also SGLang's ``glm47`` shape).

::

    <tool_call>NAME
    <arg_key>KEY</arg_key>
    <arg_value>VALUE</arg_value>
    </tool_call>

GLM-4.7 and later write the same tags without the newlines.  A call starts at a ``<tool_call>`` followed by a
function name and then an argument, the close tag, a new line or the end; elsewhere the tag is just text.  A value
is text, typed by the request's tool schemas (:mod:`.arguments`), and kept as written.  A call with no arguments is
``<tool_call>NAME</tool_call>``.  A value ends at the first ``</arg_value>`` that the next markup confirms (the next
``<arg_key>``, or the call's ``</tool_call>`` followed by nothing that belongs inside a call before the next call or
the end), so it may hold a literal close tag.  Reasoning is between ``<think>`` and ``</think>``.

Lenient rules, which mark the call recovered: a missing ``</arg_value>`` before the next ``<arg_key>`` (of an
argument the tool declares and the call does not have yet; any such key when the tool's schema is unknown) or
before the call's ``</tool_call>``, and a missing ``</tool_call>`` after the call's last argument, at the end of the
response or before the next call.  A call that does not parse otherwise, such as one with no arguments and no
``</tool_call>`` (it may be cut off inside its name), is an unparsed call, from its ``<tool_call>`` to its
``</tool_call>``, else to the next call or the end.
"""

from __future__ import annotations

import re

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import (
    ParseResult,
    ResultBuilder,
    ToolCallParser,
    skip_whitespace,
)

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
KEY_OPEN, KEY_CLOSE = "<arg_key>", "</arg_key>"
VALUE_OPEN, VALUE_CLOSE = "<arg_value>", "</arg_value>"
_REAL_START = re.compile(r"<tool_call>\s*[^\s<>]+\s*(?:<arg_key>|</tool_call>|$)")
_VALUE_ENDS = re.compile(r"</arg_value>|<arg_key>|</tool_call>")
_CALL_MARKUP = re.compile(r"</arg_value>|<arg_key>|</arg_key>|<arg_value>|</tool_call>")


class _Malformed(Exception):
    """A call block that does not parse, and why."""


class Glm45Parser(ToolCallParser):
    name = "glm45"
    aliases = ("glm", "glm47")
    call_start = _REAL_START

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        out = ResultBuilder()
        pos = 0
        while True:
            match = _REAL_START.search(text, pos)
            if match is None:
                out.text(text[pos:])
                return out.result()
            start = match.start()
            out.text(text[pos:start])
            try:
                name, arguments, end, recovered = _parse_call(text, start, tools)
            except _Malformed as e:
                end = _bad_block_end(text, start)
                out.fail(text[start:end], str(e))
            else:
                out.call(name, tools.coerce(name, arguments), recovered=recovered)
            pos = end


def _parse_call(text: str, start: int, tools: ToolSchemas) -> tuple[str, dict[str, str], int, bool]:
    pos = skip_whitespace(text, start + len(TOOL_CALL_OPEN))
    name_end = min(_find(text, "<", pos), _find(text, "\n", pos))
    name = text[pos:name_end].strip()
    pos, recovered = name_end, False
    arguments: dict[str, str] = {}
    while True:
        tag = skip_whitespace(text, pos)
        if text.startswith(KEY_OPEN, tag):
            key, value_start = _key(text, tag, name)
            value_end, pos, relaxed = _value_end(text, value_start, name, key, arguments, tools)
            value = text[value_start:value_end]
            # Without its close tag, a value ends before the newline the format puts between the tags.
            arguments[key] = value[:-1] if relaxed and value.endswith("\n") else value
            recovered = recovered or relaxed
        elif text.startswith(TOOL_CALL_CLOSE, tag):
            return name, arguments, tag + len(TOOL_CALL_CLOSE), recovered
        elif tag >= len(text) or _REAL_START.match(text, tag):
            if not arguments:  # nothing shows that the call ended: it may be cut off inside its name
                raise _Malformed(f"function {name!r} is cut off")
            return name, arguments, pos, True  # </tool_call> is missing
        else:
            raise _Malformed(f"function {name!r} has unexpected text {text[tag : tag + 30]!r}")


def _key(text: str, tag: int, name: str) -> tuple[str, int]:
    """The key of the ``<arg_key>`` at ``tag`` and where its value starts."""

    key_end = text.find(KEY_CLOSE, tag)
    if key_end == -1:
        raise _Malformed(f"an <arg_key> of function {name!r} is not closed")
    key = text[tag + len(KEY_OPEN) : key_end].strip()
    value_tag = skip_whitespace(text, key_end + len(KEY_CLOSE))
    if not key or not text.startswith(VALUE_OPEN, value_tag):
        raise _Malformed(f"argument {key!r} of function {name!r} has no <arg_value>")
    return key, value_tag + len(VALUE_OPEN)


def _value_end(
    text: str, start: int, name: str, key: str, arguments: dict[str, str], tools: ToolSchemas
) -> tuple[int, int, bool]:
    """``(value end, position to go on from, relaxed)`` for the value that starts at ``start``."""

    for match in _VALUE_ENDS.finditer(text, start):
        found = match.group()
        if found == VALUE_CLOSE:
            following = skip_whitespace(text, match.end())
            if (
                text.startswith(KEY_OPEN, following)
                or (text.startswith(TOOL_CALL_CLOSE, following) and _block_ends(text, following + len(TOOL_CALL_CLOSE)))
                or following >= len(text)
                or _REAL_START.match(text, following)
            ):
                return match.start(), match.end(), False
        elif found == KEY_OPEN:  # </arg_value> is missing before the next argument
            try:
                next_key, _ = _key(text, match.start(), name)
            except _Malformed:
                continue
            if next_key != key and next_key not in arguments and tools.declares(name, next_key) is not False:
                return match.start(), match.start(), True
        elif _block_ends(text, match.end()):  # </arg_value> is missing before </tool_call>
            return match.start(), match.start(), True
    raise _Malformed(f"argument {key!r} of function {name!r} is not closed")


def _block_ends(text: str, pos: int) -> bool:
    following = _REAL_START.search(text, pos)
    return _CALL_MARKUP.search(text, pos, len(text) if following is None else following.start()) is None


def _bad_block_end(text: str, start: int) -> int:
    following = _REAL_START.search(text, start + 1)
    limit = len(text) if following is None else following.start()
    close = text.find(TOOL_CALL_CLOSE, start, limit)
    return limit if close == -1 else close + len(TOOL_CALL_CLOSE)


def _find(text: str, needle: str, pos: int) -> int:
    found = text.find(needle, pos)
    return len(text) if found == -1 else found
