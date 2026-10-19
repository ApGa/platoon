"""``deepseekv3``: the tool calls of DeepSeek-V3 and DeepSeek-R1 (vLLM's ``deepseek_v3``).

::

    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
    ```json
    {...}
    ```<｜tool▁call▁end｜>
    <｜tool▁call▁begin｜>...<｜tool▁call▁end｜><｜tool▁calls▁end｜>

Text before the calls is content; reasoning is between ``<think>`` and ``</think>`` (DeepSeek-R1's generation
prompt ends in ``<think>``).  The JSON arguments are decoded as a whole, so they may hold a literal fence or tag.
A call that does not parse is an unparsed call, from its ``<｜tool▁call▁begin｜>`` to its ``<｜tool▁call▁end｜>``
(else to the next call or the end).  Lenient rules, which mark the call recovered: a missing
``<｜tool▁call▁end｜>``, a missing closing fence, raw control characters inside a JSON string, and arguments written
as a JSON string that holds the object.  A missing ``<｜tool▁calls▁begin｜>`` or ``<｜tool▁calls▁end｜>`` is not an
error.  :mod:`.deepseek_v31` reads DeepSeek-V3.1's simpler call shape with the same section markup.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import (
    ParseResult,
    ResultBuilder,
    ToolCallParser,
    json_call,
    json_value,
    skip_whitespace,
)

CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
CALLS_END = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"
FENCE = "```"

# (name, arguments, end, recovered) of the call whose body starts at a position; raises ValueError.
CallReader = Callable[[str, int, int], tuple[str, dict, int, bool]]


class DeepSeekV3Parser(ToolCallParser):
    name = "deepseekv3"
    aliases = ("deepseek_v3",)
    call_start = re.compile(f"{re.escape(CALLS_BEGIN)}|{re.escape(CALL_BEGIN)}")

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        return parse_sections(text, _fenced_call)


def parse_sections(text: str, read_call: CallReader) -> ParseResult:
    """Content and calls of a response in DeepSeek's section markup, each call read by ``read_call``."""

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
            name, arguments, end, recovered = read_call(text, body, limit)
        except ValueError as e:
            close = text.find(CALL_END, body, limit)
            end = limit if close == -1 else close + len(CALL_END)
            out.fail(text[start:end], f"the tool call is not valid: {e}")
        else:
            out.call(name, arguments, recovered=recovered)
        pos = end


def call_end(text: str, pos: int, limit: int) -> tuple[int, bool]:
    """The end of a call whose arguments end at ``pos``: after its ``<｜tool▁call▁end｜>``, or at ``pos`` with
    ``recovered`` when that is missing.  Text before a late close tag is an error."""

    after = skip_whitespace(text, pos)
    if text.startswith(CALL_END, after):
        return after + len(CALL_END), False
    if after >= limit or text.startswith(CALLS_END, after):
        return pos, True
    raise ValueError(f"unexpected text after the arguments: {text[after : after + 30]!r}")


def json_arguments(text: str, pos: int, name: str) -> tuple[dict, int, bool]:
    """The JSON object of arguments at ``pos``: ``(arguments, end, recovered)``."""

    value, end, lenient = json_value(text, pos)
    _, arguments, _, decoded = json_call({"name": name, "arguments": value})
    return arguments, end, lenient or decoded


def _fenced_call(text: str, body: int, limit: int) -> tuple[str, dict, int, bool]:
    """``function<｜tool▁sep｜>NAME\\n```json\\n{...}\\n```<｜tool▁call▁end｜>``."""

    sep = text.find(SEP, body, limit)
    if sep == -1:
        raise ValueError("no <｜tool▁sep｜>")
    newline = text.find("\n", sep, limit)
    if newline == -1:
        raise ValueError("no arguments after the function name")
    name = text[sep + len(SEP) : newline].strip()
    if not name:
        raise ValueError("the tool call has no function name")
    fence = skip_whitespace(text, newline)
    if not text.startswith(FENCE, fence):
        raise ValueError("the arguments are not in a ``` fence")
    language = text.find("\n", fence, limit)
    if language == -1 or text[fence + len(FENCE) : language].strip() not in ("", "json"):
        raise ValueError("the arguments fence is not a JSON fence")
    arguments, end, recovered = json_arguments(text, language, name)
    closing = skip_whitespace(text, end)
    if text.startswith(FENCE, closing):
        end = closing + len(FENCE)
    else:
        recovered = True  # the closing fence is missing
    end, missing_close = call_end(text, end, limit)
    return name, arguments, end, recovered or missing_close


def _without_section_tags(text: str) -> str:
    return text.replace(CALLS_BEGIN, "").replace(CALLS_END, "")
