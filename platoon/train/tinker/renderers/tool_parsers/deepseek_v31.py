"""``deepseekv31``: the tool calls of DeepSeek-V3.1 (vLLM's ``deepseek_v31``).

::

    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>NAME<｜tool▁sep｜>{...}<｜tool▁call▁end｜>...<｜tool▁calls▁end｜>

The section markup, the reasoning tags (V3.1's thinking prompt ends in ``<think>``) and the lenient rules are those
of :mod:`.deepseek_v3`; only the call's shape differs: the name comes before ``<｜tool▁sep｜>`` and the JSON
arguments right after it, with no fence.
"""

from __future__ import annotations

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import ParseResult, ToolCallParser
from platoon.train.tinker.renderers.tool_parsers.deepseek_v3 import (
    SEP,
    DeepSeekV3Parser,
    call_end,
    json_arguments,
    parse_sections,
)


class DeepSeekV31Parser(ToolCallParser):
    name = "deepseekv31"
    aliases = ("deepseek_v31",)
    call_start = DeepSeekV3Parser.call_start

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        return parse_sections(text, _call)


def _call(text: str, body: int, limit: int) -> tuple[str, dict, int, bool]:
    """``NAME<｜tool▁sep｜>{...}<｜tool▁call▁end｜>``."""

    sep = text.find(SEP, body, limit)
    if sep == -1:
        raise ValueError("no <｜tool▁sep｜>")
    name = text[body:sep].strip()
    if not name:
        raise ValueError("the tool call has no function name")
    arguments, end, recovered = json_arguments(text, sep + len(SEP), name)
    end, missing_close = call_end(text, end, limit)
    return name, arguments, end, recovered or missing_close
