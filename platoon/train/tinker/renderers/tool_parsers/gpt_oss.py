"""``gpt-oss``: the Harmony messages of gpt-oss (vLLM's ``openai``).

The generation prompt ends in ``<|start|>assistant``, so a response is one or more Harmony messages, each a header,
``<|message|>`` and a body that ends in ``<|end|>`` (or a stop token: ``<|call|>``, ``<|return|>``)::

    <|channel|>analysis<|message|>REASONING<|end|>
    <|start|>assistant<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{...}
    <|start|>assistant<|channel|>final<|message|>ANSWER

The ``analysis`` channel is reasoning.  A message with a recipient (``to=functions.NAME`` in the header, before or
after the channel) is a tool call with JSON arguments; the name drops the ``functions.`` namespace.  The ``final``
channel, and a ``commentary`` message without a recipient (a preamble), are content.  Text that holds no Harmony
markup is content.  Reasoning tags do not apply: Harmony's channels say what is reasoning.  A tool call whose JSON
does not decode, and a header without ``<|message|>``, are unparsed calls (that message).  Lenient rules, which mark
the call recovered: raw control characters inside a JSON string, and arguments written as a JSON string that holds
the object.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from platoon.train.tinker.renderers.tool_parsers.arguments import ToolSchemas
from platoon.train.tinker.renderers.tool_parsers.base import (
    ParseResult,
    ResultBuilder,
    ToolCallParser,
    json_call,
    json_value,
    skip_whitespace,
)

START = "<|start|>"
MESSAGE = "<|message|>"
CHANNEL = "<|channel|>"
_ENDS = ("<|end|>", "<|call|>", "<|return|>")
_MARKUP = re.compile(r"<\|(?:start|channel|message|end|call|return|constrain)\|>")
_RECIPIENT = re.compile(r"\bto=(\S+?)(?=\s|<\||$)")
_CHANNEL_NAME = re.compile(r"<\|channel\|>\s*(\w+)")


class GptOssParser(ToolCallParser):
    name = "gpt-oss"
    aliases = ("openai", "gpt_oss")
    default_reasoning_tags = None

    def parse(
        self, text: str, *, tools: Sequence[Mapping[str, Any]] | ToolSchemas | None = None, in_reasoning: bool = False
    ) -> ParseResult:
        if not _MARKUP.search(text):
            return ParseResult(content=text.strip())
        out = ResultBuilder()
        reasoning: list[str] = []
        for start, end in _messages(text):
            chunk = text[start:end]
            header, found, body = chunk.partition(MESSAGE)
            if not found:
                out.fail(chunk, "a Harmony message has no <|message|>")
                continue
            body = _body(body)
            recipient = _RECIPIENT.search(header)
            channel = _CHANNEL_NAME.search(header)
            if recipient is not None:
                try:
                    value, value_end, lenient = json_value(body) if body.strip() else ({}, 0, False)
                    if body[value_end:].strip():
                        raise ValueError(f"unexpected text after the arguments: {body[value_end:][:30]!r}")
                    name, arguments, _, decoded = json_call(
                        {"name": recipient[1].removeprefix("functions."), "arguments": value}
                    )
                except ValueError as e:
                    out.fail(chunk, f"the tool call is not valid: {e}")
                else:
                    out.call(name, arguments, recovered=lenient or decoded)
            elif channel is not None and channel[1] == "analysis":
                reasoning.append(body)
            else:
                out.text(body if not out.content else f"\n\n{body}")
        result = out.result()
        result.reasoning = "\n\n".join(part.strip() for part in reasoning if part.strip())
        return result


def _messages(text: str) -> list[tuple[int, int]]:
    """The spans of the response's messages: the first starts the response, each other one after a ``<|start|>``."""

    spans, start = [], 0
    while True:
        following = text.find(START, start)
        end = len(text) if following == -1 else following
        if text[start:end].strip():
            spans.append((start, end))
        if following == -1:
            return spans
        start = skip_whitespace(text, following + len(START))


def _body(body: str) -> str:
    """A message body without its end marker."""

    for end in _ENDS:
        cut = body.find(end)
        if cut != -1:
            body = body[:cut]
    return body
