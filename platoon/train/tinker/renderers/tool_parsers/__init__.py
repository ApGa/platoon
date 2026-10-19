"""Tool-call parsers for the chat-template renderer: one small module per format, chosen by name or by a round trip.

A parser reads a complete sampled response and returns a :class:`ParseResult`: the reasoning, the content, the tool
calls and the call blocks it could not parse (never silently dropped).  The built-in parsers, in the order
``tool_call_parser: auto`` tries them (:mod:`.probe`), with their names (SGLang's ``--tool-call-parser`` names) and
aliases (vLLM's spellings and SGLang's other names for the same format):

* ``qwen3_coder`` (``qwen3_xml``, ``step3p5``, ``nanbeige``): XML calls of Qwen3-Coder, Qwen3.5 and later;
* ``hermes`` (``qwen25``, ``qwen``): JSON in ``<tool_call>`` tags, Qwen2.5, Qwen3 and Qwen3-2507, Hermes;
* ``glm45`` (``glm``, ``glm47``): GLM-4.5 and later;
* ``deepseekv31`` (``deepseek_v31``) and ``deepseekv3`` (``deepseek_v3``): DeepSeek-V3.1, and DeepSeek-V3 and R1;
* ``kimi_k2``: Kimi K2 and K2.5;
* ``mistral``: both of Mistral's tokenizer formats;
* ``gpt-oss`` (``openai``, ``gpt_oss``): Harmony;
* ``llama3`` (``llama3_json``, ``llama4_json``): Llama 3.1 to 3.3 JSON;
* ``pythonic`` (``llama4_pythonic``): Llama 4's Python call lists.

:func:`get_parser` also takes ``none`` (reasoning only, no tool calls) and ``package.module:attr`` for a parser of
your own: an object with :meth:`~SupportsParse.parse`, a class to build without arguments, or a function with
``parse``'s signature.  Argument typing for formats that write values as text is shared (:mod:`.arguments`).
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas, coerce_value
from platoon.train.tinker.renderers.tool_parsers.base import (
    THINK_TAGS,
    ParsedToolCall,
    ParseResult,
    SupportsParse,
    ToolCallParser,
    UnparsedToolCall,
)
from platoon.train.tinker.renderers.tool_parsers.deepseek_v3 import DeepSeekV3Parser
from platoon.train.tinker.renderers.tool_parsers.deepseek_v31 import DeepSeekV31Parser
from platoon.train.tinker.renderers.tool_parsers.glm45 import Glm45Parser
from platoon.train.tinker.renderers.tool_parsers.gpt_oss import GptOssParser
from platoon.train.tinker.renderers.tool_parsers.hermes import HermesParser
from platoon.train.tinker.renderers.tool_parsers.kimi_k2 import KimiK2Parser
from platoon.train.tinker.renderers.tool_parsers.llama3 import Llama3Parser
from platoon.train.tinker.renderers.tool_parsers.mistral import MistralParser
from platoon.train.tinker.renderers.tool_parsers.pythonic import PythonicParser
from platoon.train.tinker.renderers.tool_parsers.qwen3_coder import Qwen3CoderParser

AUTO = "auto"
NONE = "none"

ReasoningTags = tuple[str, str] | None

_PARSERS: dict[str, type[ToolCallParser]] = {}
_ALIASES: dict[str, str] = {}


class NoToolCalls(ToolCallParser):
    """``none``: reasoning in ``<think>`` tags and content, no tool calls."""

    name = NONE


def register_parser(parser: type[ToolCallParser]) -> type[ToolCallParser]:
    """Add a :class:`ToolCallParser` subclass under its ``name`` and ``aliases``; ``auto`` tries parsers in the
    order they were registered.  Usable as a class decorator."""

    names = [parser.name, *parser.aliases]
    taken = [name for name in names if name in (AUTO, NONE) or canonical_name(name) not in (None, parser.name)]
    if not parser.name or taken:
        raise ValueError(f"cannot register {parser.__name__}: its name is empty or taken ({taken})")
    _PARSERS[parser.name] = parser
    for alias in parser.aliases:
        _ALIASES[alias] = parser.name
    return parser


def parser_names() -> list[str]:
    """The registered parsers' names, in the order ``auto`` tries them."""

    return list(_PARSERS)


def parser_aliases() -> dict[str, str]:
    """Each alias and the name it stands for."""

    return dict(_ALIASES)


def canonical_name(name: str) -> str | None:
    """The registered name that ``name`` is or stands for; None when it is neither."""

    if name in _PARSERS:
        return name
    return _ALIASES.get(name)


def reasoning_tags_setting(value: Any) -> ReasoningTags | str:
    """A ``reasoning_tags`` setting as the parsers take it: ``auto``, None (``none``) or an ``(open, close)`` pair."""

    if value in (None, NONE):
        return None
    if value == AUTO:
        return AUTO
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        open_tag, close_tag = value
        if isinstance(open_tag, str) and isinstance(close_tag, str) and open_tag and close_tag:
            return (open_tag, close_tag)
    raise ValueError(f"reasoning_tags must be auto, none or a pair of tags like [<think>, </think>], not {value!r}")


def get_parser(spec: str, *, reasoning_tags: ReasoningTags | str = AUTO) -> SupportsParse:
    """The parser ``spec`` names: a registered name or alias, ``none``, or ``package.module:attr``.

    ``reasoning_tags`` overrides the format's reasoning tags (:class:`ToolCallParser`); it applies to platoon's
    parsers and to user parsers that subclass :class:`ToolCallParser`.
    """

    reasoning_tags = reasoning_tags_setting(reasoning_tags)
    if spec == NONE:
        return NoToolCalls(reasoning_tags=reasoning_tags)
    name = canonical_name(spec)
    if name is not None:
        return _PARSERS[name](reasoning_tags=reasoning_tags)
    if ":" not in spec:
        raise ValueError(
            f"Unknown tool_call_parser {spec!r}: use {AUTO}, {NONE}, one of {parser_names()} (or an alias: "
            f"{sorted(_ALIASES)}), or package.module:attr for a parser of your own"
        )
    return _load_user_parser(spec, reasoning_tags)


def _load_user_parser(spec: str, reasoning_tags: ReasoningTags | str) -> SupportsParse:
    module_name, _, attr = spec.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"tool_call_parser {spec!r} does not resolve: {e}") from e
    if inspect.isclass(target):
        if issubclass(target, ToolCallParser):
            return target(reasoning_tags=reasoning_tags)
        target = target()
    elif isinstance(target, ToolCallParser) and reasoning_tags != AUTO:
        target = type(target)(reasoning_tags=reasoning_tags)
    if reasoning_tags != AUTO and not isinstance(target, ToolCallParser):
        raise ValueError(f"reasoning_tags applies to parsers that subclass ToolCallParser, and {spec!r} does not")
    if isinstance(target, SupportsParse):
        return target
    if callable(target):
        return _FunctionParser(target, spec)
    raise ValueError(f"tool_call_parser {spec!r} is neither a parser (with parse()) nor a function")


class _FunctionParser:
    """A plain function with :meth:`SupportsParse.parse`'s signature, as a parser."""

    def __init__(self, function: Callable[..., ParseResult], spec: str) -> None:
        self.function = function
        self.spec = spec

    def __repr__(self) -> str:
        return f"_FunctionParser({self.spec})"

    def parse(
        self, text: str, *, tools: Sequence[Mapping[str, Any]] | None = None, in_reasoning: bool = False
    ) -> ParseResult:
        return self.function(text, tools=tools, in_reasoning=in_reasoning)


# The built-in parsers, in the order auto tries them.
for _parser in (
    Qwen3CoderParser,
    HermesParser,
    Glm45Parser,
    DeepSeekV31Parser,
    DeepSeekV3Parser,
    KimiK2Parser,
    MistralParser,
    GptOssParser,
    Llama3Parser,
    PythonicParser,
):
    register_parser(_parser)

__all__ = [
    "AUTO",
    "NONE",
    "THINK_TAGS",
    "NoToolCalls",
    "ParseResult",
    "ParsedToolCall",
    "SupportsParse",
    "ToolCallParser",
    "ToolSchemas",
    "UnparsedToolCall",
    "canonical_name",
    "coerce_value",
    "get_parser",
    "parser_aliases",
    "parser_names",
    "reasoning_tags_setting",
    "register_parser",
]
