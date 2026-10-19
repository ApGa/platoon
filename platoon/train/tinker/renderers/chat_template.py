"""A renderer that renders prompts with the model's own chat template and parses responses with a tool-call parser.

``build_generation_prompt`` cleans the OpenAI-shaped messages (below), then encodes, without special tokens::

    tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                  **template_kwargs, **request_overrides)

The template kwargs are the renderer's ``enable_thinking``, ``preserve_thinking`` and ``reasoning_effort``
attributes (``None`` leaves a setting to the template) plus ``chat_template_kwargs``; ``train.renderer_kwargs``
sets any of them.  A request may override ``enable_thinking`` and ``preserve_thinking``.  Stop tokens are
``generation_config.json``'s ``eos_token_id``.

``parse_response`` reads the sampled text with a tool-call parser (:mod:`.tool_parsers`), set by the renderer's
``tool_call_parser`` attribute (``train.renderer_kwargs.tool_call_parser``, and the evaluation config's
``inference.renderer_kwargs``): ``auto`` (the default) keeps the parser that reads the template's own rendering of a
probe turn back exactly (:mod:`.tool_parsers.probe`), and refuses when none does and the template renders tools;
a parser's name or alias, ``none`` or ``package.module:attr`` picks one.  ``reasoning_tags`` overrides the format's
reasoning tags (``auto``, ``none`` or a pair).  The choice is made once per tokenizer, template and setting in a
process (:meth:`ChatTemplateRenderer.resolve_tool_call_parser`).  Whether a response starts inside reasoning comes
from its prompt: the gateway passes it, and when nothing does, a render of the generation prompt under the request's
``enable_thinking`` decides.  A parse never fails the model call: when the parser raises, or returns arguments that
are not JSON (a user parser's bug, say), the whole response is one unparsed call, and the error is logged.

Message cleanup, so the Qwen templates accept what clients send: content parts are joined into one string (a
cookbook thinking part becomes an inline ``<think>`` block, image and video parts are refused, since this
renderer is text-only), and tool calls (OpenAI dicts, flat dicts or cookbook ``ToolCall`` objects) become OpenAI
dicts whose ``arguments`` is a mapping: a JSON string is parsed, and ``""`` or ``None`` become ``{}``.

The gateway picks its code path from this class's signatures (``platoon.gateway.core.ModelGateway.render``), so
``build_generation_prompt`` spells out ``tools``, ``enable_thinking`` and ``preserve_thinking`` rather than taking
``**kwargs``.

The names ``qwen3_6_preserve_thinking`` and ``qwen3_8_preserve_thinking`` (each with a ``_disable_thinking``
variant) are this renderer with preset settings: they are the custom renderers it replaced, token for token
(``tests/renderers``), as long as the model is the ``apurvaga/*-preserve-thinking`` snapshot whose template they
copied.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jinja2
import tinker
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    UnparsedToolCall,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

from platoon.train.tinker.renderers.tool_parsers import AUTO, ParseResult, SupportsParse, reasoning_tags_setting
from platoon.train.tinker.renderers.tool_parsers import UnparsedToolCall as UnparsedBlock
from platoon.train.tinker.renderers.tool_parsers.base import new_call_id
from platoon.train.tinker.renderers.tool_parsers.probe import (
    ParserChoice,
    ToolCallParserError,
    choose_parser,
    prompt_ends_in,
    reasoning_open,
)

logger = logging.getLogger(__name__)

RENDERER_NAME = "chat_template"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
# Prompt tokens decoded to see whether a generation prompt ends in an open reasoning tag.
_PROMPT_TAIL_TOKENS = 16
# Parser errors are logged with their traceback the first few times in a process, then every thousandth.
_PARSER_ERROR_TRACEBACKS = 3
_parser_errors = itertools.count(1)

# Parser choices made in this process: (tokenizer, template, template kwargs, stop strings, settings) -> choice.
_CHOICES: dict[tuple[Any, ...], tuple[SupportsParse, ParserChoice]] = {}
_CHOICES_LOCK = threading.Lock()

# The custom renderers' names, kept so configs and run records resolve to the same prompts.
ALIASES: dict[str, dict[str, Any]] = {
    "qwen3_6_preserve_thinking": {"preserve_thinking": True},
    "qwen3_6_preserve_thinking_disable_thinking": {"preserve_thinking": True, "enable_thinking": False},
    "qwen3_8_preserve_thinking": {"preserve_thinking": True, "reasoning_effort": "xhigh"},
    "qwen3_8_preserve_thinking_disable_thinking": {
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
        "enable_thinking": False,
    },
}


class ChatTemplateRenderer(Renderer):
    """Prompts from ``tokenizer.apply_chat_template``; responses through a tool-call parser (``tool_call_parser``)."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: Any | None = None,
        *,
        enable_thinking: bool = True,
        preserve_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        tool_call_parser: str = AUTO,
        reasoning_tags: Any = AUTO,
    ):
        super().__init__(tokenizer)
        self.image_processor = image_processor  # accepted for get_renderer() parity; text-only
        self.enable_thinking = enable_thinking
        self.preserve_thinking = preserve_thinking
        self.reasoning_effort = reasoning_effort
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tool_call_parser = tool_call_parser
        self.reasoning_tags = reasoning_tags
        self._stop_tokens = generation_stop_tokens(tokenizer)
        self._generation_prompts: dict[str, str] = {}  # template kwargs -> a generation prompt rendered with them

    # ------------------------------------------------------------------ render
    def template_kwargs(self, *, enable_thinking: bool | None = None, preserve_thinking: bool | None = None) -> dict:
        """The kwargs the template sees: the renderer's settings, then the request's overrides."""
        kwargs = dict(self.chat_template_kwargs)
        kwargs["enable_thinking"] = bool(self.enable_thinking if enable_thinking is None else enable_thinking)
        preserve = self.preserve_thinking if preserve_thinking is None else preserve_thinking
        if preserve is not None:
            kwargs["preserve_thinking"] = bool(preserve)
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def render_text(
        self,
        messages: list[Message],
        *,
        tools: Iterable[Any] | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """The prompt as text, ending in the assistant generation prompt (or the conversation as it is, without)."""
        if not messages:
            raise RendererError("No messages provided.")
        try:
            return self.tokenizer.apply_chat_template(
                clean_messages(messages),
                tools=list(tools) if tools and not isinstance(tools, Mapping) else None,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                **self.template_kwargs(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking),
            )
        except jinja2.TemplateError as e:  # the template's raise_exception: a bad request, like RendererError
            raise RendererError(str(e)) from e

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
        *,
        tools: Iterable[Any] | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
    ) -> tinker.ModelInput:
        if role != "assistant":
            raise RendererError(f"{type(self).__name__} only generates assistant turns, not {role!r}")
        text = self.render_text(
            messages, tools=tools, enable_thinking=enable_thinking, preserve_thinking=preserve_thinking
        )
        tokens = self._encode(text)
        if prefill:
            tokens += self._encode(prefill)
        return tinker.ModelInput.from_ints(tokens)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        raise NotImplementedError(f"{type(self).__name__} renders whole conversations with the chat template")

    def get_stop_sequences(self) -> list[int]:
        return list(self._stop_tokens)

    # ------------------------------------------------------------------ parse
    def parse_response(
        self,
        response: list[int],
        tools: Iterable[Any] | None = None,
        *,
        enable_thinking: bool | None = None,
        prompt: tinker.ModelInput | Sequence[int] | None = None,
    ) -> tuple[Message, bool]:
        """Parse sampled tokens (after the generation prompt) into an assistant Message.

        ``success`` is False when no stop token was sampled; the message is still parsed on a best-effort basis
        (truncated reasoning ends up in ``reasoning_content`` with empty ``content``).  ``tools`` type the
        arguments of formats that write values as text.  The response starts inside reasoning when ``prompt`` (the
        prompt it was sampled for) ends in the parser's open reasoning tag; without a prompt, when the generation
        prompt under ``enable_thinking`` (the request's, else the renderer's) does.

        Besides ``content``, ``reasoning_content`` and ``tool_calls``, the message holds the call blocks that did
        not parse as ``unparsed_tool_calls`` (their raw text is in neither ``content`` nor the reasoning) and, when
        some calls parsed only by a lenient rule, their number as ``recovered_tool_calls``.  When the parser raises,
        or returns arguments that are not JSON, the whole text is one unparsed call.
        """
        text, success = split_at_stop(self.tokenizer, response, self._stop_tokens)
        parser, _ = self.resolve_tool_call_parser()
        in_reasoning = self._starts_in_reasoning(reasoning_open(parser), prompt, enable_thinking)
        try:
            result = parser.parse(text, tools=list(tools) if tools else None, in_reasoning=in_reasoning)
            return assistant_message(result), success
        except Exception as e:  # a parser's bug must not fail the model call
            _log_parser_error(parser, e)
            error = f"the tool-call parser failed: {type(e).__name__}: {e}"
            unparsed = [UnparsedBlock(raw_text=text, error=error)] if text.strip() else []
            return assistant_message(ParseResult(unparsed=unparsed)), success

    def resolve_tool_call_parser(self) -> tuple[SupportsParse, ParserChoice]:
        """The tool-call parser for this renderer's settings and why it was chosen (see the module docstring).

        Chosen once per tokenizer, template, template settings and parser settings in a process; raises
        :class:`~.tool_parsers.probe.ToolCallParserError` when ``auto`` finds no parser for a template that
        renders tools, when ``tool_call_parser`` names no parser, or when ``reasoning_tags`` is not a setting.
        """
        settings = (
            self.tool_call_parser,
            json.dumps(self.reasoning_tags, default=str),
            json.dumps(self.template_kwargs(), sort_keys=True, default=str),
        )
        resolved = self.__dict__.get("_resolved_parser")
        if resolved is not None and resolved[0] == settings:
            return resolved[1]
        template = json.dumps(self.tokenizer.chat_template, sort_keys=True, default=str)
        key = (
            getattr(self.tokenizer, "name_or_path", None),
            hashlib.sha256(template.encode()).hexdigest(),
            tuple(self._stop_tokens),
            *settings,
        )
        setting = self.tool_call_parser or AUTO
        if not isinstance(setting, str):
            raise ToolCallParserError(
                f"tool_call_parser must be a parser's name or package.module:attr, not {setting!r}"
            )
        try:
            reasoning_tags = reasoning_tags_setting(self.reasoning_tags)
        except ValueError as e:
            raise ToolCallParserError(str(e)) from e
        with _CHOICES_LOCK:
            chosen = _CHOICES.get(key)
            if chosen is None:
                chosen = choose_parser(
                    self._render_probe,
                    [self.tokenizer.decode([token]) for token in self._stop_tokens],
                    setting=setting,
                    reasoning_tags=reasoning_tags,
                    label=str(getattr(self.tokenizer, "name_or_path", "") or ""),
                )
                _CHOICES[key] = chosen
        self._resolved_parser = (settings, chosen)
        return chosen

    @property
    def tool_call_parser_choice(self) -> ParserChoice:
        """Which tool-call parser this renderer uses and why (``lock.json`` records it)."""
        return self.resolve_tool_call_parser()[1]

    def _render_probe(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, generate: bool) -> str:
        return self.render_text(messages, tools=tools, add_generation_prompt=generate)  # type: ignore[arg-type]

    def _starts_in_reasoning(
        self, open_tag: str | None, prompt: tinker.ModelInput | Sequence[int] | None, enable_thinking: bool | None
    ) -> bool:
        if not open_tag:
            return False
        if prompt is not None:
            try:
                tokens = list(prompt.to_ints() if isinstance(prompt, tinker.ModelInput) else prompt)
                return prompt_ends_in(str(self.tokenizer.decode(tokens[-_PROMPT_TAIL_TOKENS:])), open_tag)
            except (ValueError, TypeError):
                logger.debug("Could not read the prompt's last tokens; using the generation prompt instead")
        enable = bool(self.enable_thinking if enable_thinking is None else enable_thinking)
        key = json.dumps(self.template_kwargs(enable_thinking=enable), sort_keys=True, default=str)
        if key not in self._generation_prompts:
            conversation: list[Message] = [{"role": "user", "content": "Hi"}]
            try:
                self._generation_prompts[key] = self.render_text(conversation, enable_thinking=enable)
            except RendererError:  # a template that needs more than one user message: go by the setting
                return enable
        return prompt_ends_in(self._generation_prompts[key], open_tag)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def assistant_message(result: ParseResult) -> Message:
    """A parsed response as a cookbook assistant Message (see :meth:`ChatTemplateRenderer.parse_response`).

    Raises ``ValueError`` or ``TypeError`` when a call's arguments are not JSON (``NaN``, a set).
    """
    message: dict[str, Any] = {"role": "assistant", "content": result.content, "reasoning_content": result.reasoning}
    if result.tool_calls:
        message["tool_calls"] = [
            ToolCall(
                id=call.id or new_call_id(),
                function=ToolCall.FunctionBody(
                    name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False, allow_nan=False)
                ),
            )
            for call in result.tool_calls
        ]
    if result.unparsed:
        message["unparsed_tool_calls"] = [
            UnparsedToolCall(raw_text=call.raw_text, error=call.error) for call in result.unparsed
        ]
    if result.recovered:
        message["recovered_tool_calls"] = result.recovered
    return message  # type: ignore[return-value]


def _log_parser_error(parser: SupportsParse, error: Exception) -> None:
    count = next(_parser_errors)
    if count <= _PARSER_ERROR_TRACEBACKS:
        logger.warning("Tool-call parser %r failed; the whole response is one unparsed call", parser, exc_info=error)
    elif count % 1000 == 0:
        logger.warning("Tool-call parser %r failed on %d responses so far, the last time: %r", parser, count, error)


def split_at_stop(tokenizer: Any, response: Sequence[int], stop_tokens: Iterable[int]) -> tuple[str, bool]:
    """The decoded text before the first stop token, and whether a stop token was sampled."""
    stop_ids = set(stop_tokens)
    for i, token in enumerate(response):
        if token in stop_ids:
            return str(tokenizer.decode(list(response[:i]))), True
    return str(tokenizer.decode(list(response))), False


def generation_stop_tokens(tokenizer: Tokenizer) -> list[int]:
    """``generation_config.json``'s ``eos_token_id`` for the tokenizer's model (its eos token when there is none)."""
    from transformers import GenerationConfig

    eos: Any = None
    try:
        eos = GenerationConfig.from_pretrained(tokenizer.name_or_path).eos_token_id
    except OSError:
        pass
    if eos is None:
        eos = tokenizer.eos_token_id
        logger.warning("No generation_config.json eos_token_id for %s; stopping on %s", tokenizer.name_or_path, eos)
    if eos is None:
        raise RendererError(f"No stop tokens for {tokenizer.name_or_path}")
    return [eos] if isinstance(eos, int) else list(eos)


# --------------------------------------------------------------------- cleanup
def clean_messages(messages: Iterable[Message]) -> list[dict[str, Any]]:
    """Messages as the chat template expects them (see the module docstring)."""
    cleaned = []
    for message in messages:
        out = dict(message)
        out["content"] = content_text(message.get("content"), is_system=message.get("role") == "system")
        tool_calls = out.pop("tool_calls", None)
        if tool_calls and not isinstance(tool_calls, Mapping) and isinstance(tool_calls, Iterable):
            out["tool_calls"] = [_openai_tool_call(tool_call) for tool_call in tool_calls]
        cleaned.append(out)
    return cleaned


def content_text(content: Any, *, is_system: bool = False) -> str:
    """Message content as one string: text parts joined, thinking parts inlined, images and videos refused."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        raise RendererError("Unexpected content type.")
    if isinstance(content, Iterable):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, Mapping):
                raise RendererError("Unexpected item type in content.")
            item_type = item.get("type")
            if "image" in item or "image_url" in item or item_type in ("image", "input_image"):
                if is_system:
                    raise RendererError("System message cannot contain images.")
                raise RendererError("Image content is not supported by the text-only chat-template renderer.")
            if "video" in item or item_type == "video":
                if is_system:
                    raise RendererError("System message cannot contain videos.")
                raise RendererError("Video content is not supported by the text-only chat-template renderer.")
            if "text" in item:
                text = item["text"]
                parts.append("" if text is None else str(text))
            elif item_type == "thinking":
                # cookbook ThinkingPart -> inline think block, which the template splits out of the content
                parts.append(f"{THINK_OPEN}{item.get('thinking', '')}{THINK_CLOSE}")
            else:
                raise RendererError("Unexpected item type in content.")
        return "".join(parts)
    raise RendererError("Unexpected content type.")


def tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Mapping[str, Any]]:
    """Accept OpenAI dicts (nested ``function`` or flat), cookbook/LiteLLM objects; arguments as a mapping."""
    if isinstance(tool_call, Mapping):
        function = tool_call.get("function")
        body: Any = function if function is not None else tool_call
        name = body.get("name") if isinstance(body, Mapping) else None
        arguments = body.get("arguments") if isinstance(body, Mapping) else None
    else:
        body = getattr(tool_call, "function", tool_call)
        name = getattr(body, "name", None)
        arguments = getattr(body, "arguments", None)
    if not isinstance(name, str) or not name:
        raise RendererError(f"Tool call is missing a string function name: {tool_call!r}")
    if arguments is None:
        return name, {}
    if isinstance(arguments, str):
        if not arguments.strip():
            return name, {}
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            raise RendererError(f"Tool call {name!r} arguments are not valid JSON: {e}") from e
    if not isinstance(arguments, Mapping):
        raise RendererError(f"Tool call {name!r} arguments must be a JSON object, got {type(arguments).__name__}")
    return name, arguments


def _openai_tool_call(tool_call: Any) -> dict[str, Any]:
    name, arguments = tool_call_name_and_arguments(tool_call)
    call_id = tool_call.get("id") if isinstance(tool_call, Mapping) else getattr(tool_call, "id", None)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": dict(arguments)}}


# --------------------------------------------------------------------- registry
RENDERER_FACTORIES = {
    RENDERER_NAME: ChatTemplateRenderer,
    **{name: functools.partial(ChatTemplateRenderer, **preset) for name, preset in ALIASES.items()},
}


def register(force: bool = False) -> list[str]:
    """Register the renderer and its aliases with ``tinker_cookbook.renderers.register_renderer`` (idempotent).

    Returns the registered names.  Must run in every process that calls ``get_renderer`` with these names
    (including processes that unpickle a renderer).
    """
    from tinker_cookbook.renderers import is_renderer_registered, register_renderer

    for name, factory in RENDERER_FACTORIES.items():
        if force or not is_renderer_registered(name):
            register_renderer(name, factory)
    return list(RENDERER_FACTORIES)
