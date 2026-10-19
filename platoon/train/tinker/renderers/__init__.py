"""Platoon-side tinker_cookbook renderers.

Call :func:`register` before ``tinker_cookbook.renderers.get_renderer`` so that ``train.renderer_name`` can name
``chat_template`` or one of its preset aliases (``qwen3_6_preserve_thinking``, ``qwen3_8_preserve_thinking`` and
their ``_disable_thinking`` variants).  The chat-template renderer parses responses with the tool-call parsers of
:mod:`.tool_parsers`.
"""

from __future__ import annotations

from typing import Any

from platoon.train.tinker.renderers.chat_template import (
    ALIASES,
    RENDERER_FACTORIES,
    RENDERER_NAME,
    ChatTemplateRenderer,
    register,
)
from platoon.train.tinker.renderers.tool_parsers.probe import ParserChoice, ToolCallParserError


def tool_call_parsing(renderer: Any) -> ParserChoice | None:
    """Choose a chat-template renderer's tool-call parser now and say which and why; None for other renderers,
    which parse tool calls their own way.  Raises :class:`ToolCallParserError` as the renderer does."""

    if isinstance(renderer, ChatTemplateRenderer):
        return renderer.resolve_tool_call_parser()[1]
    return None


__all__ = [
    "ALIASES",
    "RENDERER_FACTORIES",
    "RENDERER_NAME",
    "ChatTemplateRenderer",
    "ParserChoice",
    "ToolCallParserError",
    "register",
    "tool_call_parsing",
]
