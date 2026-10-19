"""What a tool-call parser returns, and the steps every parser shares.

A parser reads the text a model sampled after the generation prompt, cut at the stop token, and returns a
:class:`ParseResult`: the reasoning, the visible content, the tool calls, and every call block it could not parse
(an :class:`UnparsedToolCall` with the block's raw text and the reason).  Parsers read complete responses only;
the gateway synthesizes a stream after the parse.  A parser never raises on what a model wrote: whatever it cannot
read is content or an unparsed call.

:class:`ToolCallParser` subclasses implement :meth:`~ToolCallParser.parse_calls` for their format and inherit the
reasoning split.  Text between the format's reasoning tags (``<think>`` and ``</think>`` unless a parser says
otherwise) at the start of the response is reasoning, and so is the start of a response whose generation prompt
ended in the open tag (``in_reasoning``); a response that starts with the close tag has empty reasoning.  A
reasoning block that never closes ends where a real tool call starts (the parser's ``call_start``), as SGLang's
reasoning parser has it (Apache-2.0; see ``NOTICE`` in this directory), since models sometimes open a call without
closing their reasoning; without a call, the whole response is reasoning.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from platoon.train.tinker.renderers.tool_parsers.arguments import LENIENT_JSON, STRICT_JSON, ToolSchemas

THINK_TAGS = ("<think>", "</think>")

_WHITESPACE = re.compile(r"\s*")
_JSON_START = re.compile(r'[{\["\-0-9tfn]')  # what a JSON value can start with


@dataclass(frozen=True)
class ParsedToolCall:
    """One tool call: its function name and arguments.

    ``id`` is the call id the model sampled, for formats that carry one (Kimi K2's ``functions.NAME:N``, Mistral's
    nine-character ids); None lets the renderer mint one.  ``recovered`` marks a call that parsed only by a lenient
    rule, such as a missing close tag.
    """

    name: str
    arguments: dict[str, Any]
    id: str | None = None
    recovered: bool = False


@dataclass(frozen=True)
class UnparsedToolCall:
    """A tool-call block that did not parse: its raw text, exactly as sampled, and why."""

    raw_text: str
    error: str


@dataclass
class ParseResult:
    """A parsed response: ``reasoning`` and ``content`` (both stripped), the tool calls in order, and the call blocks
    that did not parse, whose raw text is in neither ``content`` nor ``reasoning``."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ParsedToolCall] = field(default_factory=list)
    unparsed: list[UnparsedToolCall] = field(default_factory=list)

    @property
    def recovered(self) -> int:
        """How many of the tool calls parsed only by a lenient rule."""

        return sum(call.recovered for call in self.tool_calls)


@runtime_checkable
class SupportsParse(Protocol):
    """What platoon needs of a tool-call parser, built in or user-supplied (``tool_call_parser: pkg.mod:attr``).

    ``text`` is the sampled text before the stop token.  ``tools`` are the request's OpenAI tool definitions (for
    typing argument values sampled as text).  ``in_reasoning`` is true when the generation prompt ended in the
    format's open reasoning tag, so the text starts inside the reasoning block.
    """

    def parse(
        self, text: str, *, tools: Sequence[Mapping[str, Any]] | None = None, in_reasoning: bool = False
    ) -> ParseResult: ...


class ToolCallParser:
    """A tool-call format and the reasoning convention that goes with it.

    Subclasses set ``name`` (SGLang's ``--tool-call-parser`` name), ``aliases`` (vLLM's spellings and other names),
    ``default_reasoning_tags`` and ``call_start`` (the pattern that starts a real call, which ends an unclosed
    reasoning block), and implement :meth:`parse_calls`.  ``reasoning_tags`` overrides the format's reasoning tags:
    an ``(open, close)`` pair, or None for a model that does not reason in tags.
    """

    name: ClassVar[str] = ""
    aliases: ClassVar[tuple[str, ...]] = ()
    default_reasoning_tags: ClassVar[tuple[str, str] | None] = THINK_TAGS
    call_start: ClassVar[re.Pattern[str] | None] = None

    def __init__(self, *, reasoning_tags: tuple[str, str] | None | Literal["auto"] = "auto") -> None:
        self.reasoning_tags = self.default_reasoning_tags if reasoning_tags == "auto" else reasoning_tags

    def __repr__(self) -> str:
        return f"{type(self).__name__}(reasoning_tags={self.reasoning_tags!r})"

    @property
    def reasoning_open(self) -> str | None:
        """The tag that opens reasoning, which a generation prompt may end in; None without reasoning tags."""

        return self.reasoning_tags[0] if self.reasoning_tags else None

    def parse(
        self, text: str, *, tools: Sequence[Mapping[str, Any]] | ToolSchemas | None = None, in_reasoning: bool = False
    ) -> ParseResult:
        """The reasoning, content, tool calls and unparsed call blocks of ``text`` (see :class:`SupportsParse`)."""

        reasoning, body = self.split_reasoning(text, in_reasoning=in_reasoning)
        result = self.parse_calls(body, ToolSchemas.of(tools))
        result.reasoning = reasoning
        return result

    def split_reasoning(self, text: str, *, in_reasoning: bool = False) -> tuple[str, str]:
        """``(reasoning, rest)``: the reasoning block at the start of ``text`` and the text after it."""

        if self.reasoning_tags is None:
            return "", text
        open_tag, close_tag = self.reasoning_tags
        if not in_reasoning:
            stripped = text.lstrip()
            if stripped.startswith(close_tag):  # an empty reasoning block whose open tag was in the prompt
                return "", stripped[len(close_tag) :]
            if not stripped.startswith(open_tag):
                return "", text
            text = stripped[len(open_tag) :]
        end = text.find(close_tag)
        if end == -1:
            start = self.call_start.search(text) if self.call_start is not None else None
            if start is None:
                return _reasoning(text, open_tag), ""
            return _reasoning(text[: start.start()], open_tag), text[start.start() :]
        return _reasoning(text[:end], open_tag), text[end + len(close_tag) :]

    def parse_calls(self, text: str, tools: ToolSchemas) -> ParseResult:
        """The content, tool calls and unparsed call blocks of the response after its reasoning."""

        return ParseResult(content=text.strip())


def _reasoning(text: str, open_tag: str) -> str:
    """Reasoning text without the open tags a model re-emitted at its start."""

    text = text.strip()
    while text.startswith(open_tag):
        text = text[len(open_tag) :].lstrip()
    return text


class ResultBuilder:
    """Collects a response's content pieces, calls and unparsed blocks in order, for :meth:`parse_calls`."""

    def __init__(self) -> None:
        self.content: list[str] = []
        self.calls: list[ParsedToolCall] = []
        self.unparsed: list[UnparsedToolCall] = []

    def text(self, text: str) -> None:
        if text:
            self.content.append(text)

    def call(self, name: str, arguments: dict[str, Any], *, id: str | None = None, recovered: bool = False) -> None:
        self.calls.append(ParsedToolCall(name=name, arguments=arguments, id=id, recovered=recovered))

    def fail(self, raw_text: str, error: str) -> None:
        if raw_text.strip():
            self.unparsed.append(UnparsedToolCall(raw_text=raw_text, error=error))

    def result(self) -> ParseResult:
        return ParseResult(content="".join(self.content).strip(), tool_calls=self.calls, unparsed=self.unparsed)


# --------------------------------------------------------------------------- JSON helpers


def skip_whitespace(text: str, pos: int) -> int:
    match = _WHITESPACE.match(text, pos)
    return match.end() if match else pos


def json_value(text: str, pos: int = 0) -> tuple[Any, int, bool]:
    """Decode the JSON value at ``pos`` (after whitespace): ``(value, end, lenient)``.

    ``lenient`` is true when the value decoded only with raw control characters (a newline, say) inside a string,
    which strict JSON forbids.  Raises ``ValueError`` when no JSON value starts there, when the value holds what JSON
    has no value for (``NaN``, ``Infinity``, a number out of a float's range), and when it nests more than
    :data:`~.arguments.MAX_JSON_DEPTH` levels deep (or deeper than the thread's stack lets Python decode).
    """

    pos = skip_whitespace(text, pos)
    if not _JSON_START.match(text, pos):  # fail at once: a decode error counts the lines before ``pos``
        raise ValueError(f"no JSON value starts at {text[pos : pos + 20]!r}")
    try:
        try:
            value, end = STRICT_JSON.raw_decode(text, pos)
            return value, end, False
        except ValueError:
            value, end = LENIENT_JSON.raw_decode(text, pos)
            return value, end, True
    except RecursionError:
        raise ValueError("the JSON is nested too deeply to decode") from None


def call_arguments(value: Any) -> tuple[dict[str, Any], bool]:
    """A call's JSON arguments as a mapping: ``(arguments, recovered)``.

    Missing or null arguments are ``{}``; a JSON string that holds an object is decoded, as a recovered call.
    Raises ``ValueError`` for anything else.
    """

    if value is None:
        return {}, False
    if isinstance(value, dict):
        return value, False
    if isinstance(value, str):
        try:
            decoded = STRICT_JSON.decode(value) if value.strip() else {}
        except (ValueError, RecursionError) as e:
            raise ValueError(f"arguments are a string that is not a JSON object: {e}") from None
        if isinstance(decoded, dict):
            return decoded, True
    raise ValueError(f"arguments must be a JSON object, not {type(value).__name__}")


def json_call(value: Any) -> tuple[str, dict[str, Any], str | None, bool]:
    """A ``{"name": ..., "arguments": ...}`` call object (``parameters`` for ``arguments``, as Llama writes it, and
    an optional ``id``): ``(name, arguments, id, recovered)``.  Raises ``ValueError`` when it is not one."""

    if not isinstance(value, dict):
        raise ValueError(f"a call must be a JSON object, not {type(value).__name__}")
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("the call has no function name")
    arguments, recovered = call_arguments(value.get("arguments", value.get("parameters")))
    call_id = value.get("id")
    return name.strip(), arguments, call_id if isinstance(call_id, str) and call_id else None, recovered


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
