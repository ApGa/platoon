"""Qwen3.6 *preserve-thinking* renderer for platoon's Tinker path.

Token-for-token mirror of the chat template shipped with
``apurvaga/Qwen3.6-35B-A3B-preserve-thinking`` (``chat_template.jinja``,
snapshot ``ce815485d640a33461c25e3d5c505b5346e8c857``), so that

    renderer.build_generation_prompt(messages, tools=tools).to_ints()
    == tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                     add_generation_prompt=True)

for OpenAI / OpenHands shaped conversations.  Under the AReaL backend the very
same template was applied by AReaL's OpenAI client (``areal.experimental.openai
.client`` -> ``tokenizer.apply_chat_template``), so this renderer keeps the
prompt distribution identical and keeps consecutive sampling calls exact
prefix extensions (prefix-merged datums, R3 routing coverage).

Template facts mirrored here (see the jinja for the source of truth):

* ``<think>`` blocks are rendered for **every** assistant turn when
  ``preserve_thinking`` is true (the template default).  With
  ``preserve_thinking=False`` only assistant turns *after* the last real user
  query keep their block (upstream Qwen behaviour).  A "real user query" is a
  ``user`` message whose trimmed content is not wrapped in
  ``<tool_response>...</tool_response>``; if none exists the template raises.
* Assistant reasoning comes from ``reasoning_content`` when it is a string;
  otherwise it is split out of ``content`` at ``</think>`` exactly like the
  template does (``content.split('</think>')[0].rstrip('\\n').split('<think>')
  [-1].lstrip('\\n')`` / ``content.split('</think>')[-1].lstrip('\\n')``).
  Reasoning and all message contents are ``|trim``-med, so an assistant turn
  without reasoning renders as ``<think>\\n\\n</think>\\n\\n``.
* Tools are declared in a leading system turn: ``# Tools ... <tools>`` with one
  ``json.dumps(tool, ensure_ascii=False)`` line per tool **verbatim** (the
  template's ``tojson``; OpenAI ``{"type": "function", "function": {...}}``
  dicts are dumped as given, key order preserved), followed by the
  ``<IMPORTANT>`` instructions and, if present, the user's system prompt.
* Tool calls use the Qwen3-Coder XML form (NOT JSON)::

      <tool_call>
      <function=NAME>
      <parameter=KEY>
      VALUE
      </parameter>
      </function>
      </tool_call>

  String argument values are emitted verbatim, everything else through
  ``json.dumps(value, ensure_ascii=False)`` (``3``, ``true``, ``null``,
  ``{"a": 1}``).  The template iterates ``arguments|items`` so arguments MUST
  be a mapping; OpenAI JSON-string arguments are ``json.loads``-ed first (AReaL
  and SGLang do the same server-side).  The first call is preceded by
  ``\\n\\n`` when the visible content is non-empty, later calls by ``\\n``.
* Tool results render as ``<|im_start|>user\\n<tool_response>\\n...\\n
  </tool_response>``; consecutive tool messages share one user turn.
* Generation prompt: ``<|im_start|>assistant\\n<think>\\n`` or, with
  ``enable_thinking=False``, ``<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n``.

Platoon-side API (the pinned tinker_cookbook ``Renderer`` API has no ``tools``
parameter; tools are usually folded into a system message):

* ``build_generation_prompt(messages, role="assistant", prefill=None, *,
  tools=None, enable_thinking=None, preserve_thinking=None)`` -- pass the
  request's OpenAI ``tools`` list as-is; per-call overrides default to the
  instance attributes.  ``build_generation_prompt_with_tools(messages, tools)``
  is an alias.
* ``build_supervised_example(messages, train_on_what, *, tools=None, ...)``.
* ``parse_response(tokens, tools=None) -> (Message, success)``.  Returns an
  assistant ``Message`` with ``content`` (visible text, tool-call blocks
  removed, stripped), ``reasoning_content`` (stripped ``<think>`` span) and
  ``tool_calls`` as ``tinker_cookbook.renderers.ToolCall`` objects (OpenAI
  shape: ``.id``, ``.type == "function"``, ``.function.name``,
  ``.function.arguments`` JSON string).  Pass ``tools`` to get AReaL's
  schema-typed argument coercion (``integer``/``number``/``boolean``/
  ``array``/``object``); without a schema every argument stays a string.
  Parsing mirrors AReaL's ``_process_tool_calls_qwen3_coder_xml`` with the
  SGLang ``qwen3_coder`` detector as fallback.
* ``to_openai_message(message)`` converts a parsed ``Message`` back to an
  OpenAI dict (``reasoning_content`` + ``tool_calls`` dicts) that renders
  identically when echoed back into ``build_generation_prompt``.
* ``render_conversation_text(...)`` returns the rendered *string* (equal to
  ``apply_chat_template(..., tokenize=False)``) for debugging.
* ``register()`` registers the factories under ``qwen3_6_preserve_thinking``
  and ``qwen3_6_preserve_thinking_disable_thinking`` with
  ``tinker_cookbook.renderers.register_renderer`` so ``get_renderer(name,
  tokenizer, image_processor=..., model_name=...)`` (used by
  ``platoon.train.tinker.proxy.register_tinker_llm``) can build them.  Call it
  in every process that unpickles a renderer.  The attributes
  ``preserve_thinking`` / ``enable_thinking`` are plain instance attributes so
  ``train.renderer_kwargs`` can override them.

Content parts: ``{"type": "text", "text": ...}`` (any item with a ``text`` key)
and bare strings are concatenated; cookbook ``ThinkingPart`` items on
assistant turns are folded into ``<think>...</think>`` text (then split like
the template); image / video parts raise ``RendererError`` (text-only).
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import tinker
import torch
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    Role,
    ToolCall,
    TrainOnWhat,
    UnparsedToolCall,
)
from tinker_cookbook.tokenizer_utils import Tokenizer

RENDERER_NAME = "qwen3_6_preserve_thinking"
RENDERER_NAME_DISABLE_THINKING = "qwen3_6_preserve_thinking_disable_thinking"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
FUNCTION_PREFIX = "<function="
PARAMETER_PREFIX = "<parameter="

# Markers that must tokenize atomically for per-message tokenization to equal
# whole-string tokenization (every chunk boundary we introduce is adjacent to
# one of these, or to "\n" followed by trimmed, non-whitespace text).
_ATOMIC_MARKERS = (IM_START, IM_END, THINK_OPEN, THINK_CLOSE, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)

# Verbatim copies of the template's string literals.
_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)

# AReaL primary parser (areal/experimental/openai/tool_call_parser.py @ d99124e).
_AREAL_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)
_AREAL_FUNCTION_RE = re.compile(r"<function=(?P<name>[^>\s]+)>\s*(?P<body>.*?)\s*</function>", re.DOTALL)
_AREAL_PARAMETER_RE = re.compile(r"<parameter=(?P<name>[^>\s]+)>(?P<value>.*?)</parameter>", re.DOTALL)
# SGLang qwen3_coder fallback (sglang/srt/function_call/qwen3_coder_detector.py @ 0.5.10.post1).
_SGL_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_SGL_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
_SGL_PARAMETER_RE = re.compile(r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)", re.DOTALL)


def _tojson(value: Any) -> str:
    """transformers' ``tojson`` jinja filter: json.dumps(ensure_ascii=False, default separators)."""
    return json.dumps(value, ensure_ascii=False)


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class Qwen3_6RenderContext(RenderContext):
    """RenderContext carrying the template-level knobs for one build call."""

    last_query_index: int = -1
    """Index of the last ``user`` message that is not a ``<tool_response>`` wrapper."""
    preserve_thinking: bool = True
    enable_thinking: bool = True


@dataclass(frozen=True)
class _RenderedText:
    header: str | None
    output: str
    stop_overlap: str | None = None


class Qwen3_6PreserveThinkingRenderer(Renderer):
    """tinker_cookbook renderer mirroring the Qwen3.6 preserve-thinking chat template."""

    supports_streaming = True

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: Any | None = None,
        *,
        preserve_thinking: bool = True,
        enable_thinking: bool = True,
    ):
        super().__init__(tokenizer)
        self.image_processor = image_processor  # accepted for get_renderer() parity; text-only
        self.preserve_thinking = bool(preserve_thinking)
        self.enable_thinking = bool(enable_thinking)
        self._marker_tokens = {marker: self._single_token(marker) for marker in _ATOMIC_MARKERS}
        self._im_end_token = self._marker_tokens[IM_END]
        endoftext = self._encode(ENDOFTEXT)
        self._endoftext_token = endoftext[0] if len(endoftext) == 1 else None

    # ------------------------------------------------------------------ utils
    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _single_token(self, text: str) -> int:
        tokens = self._encode(text)
        if len(tokens) != 1:
            raise RendererError(
                f"{type(self).__name__} requires {text!r} to be a single (added) token; got {tokens}. "
                "Is this the Qwen3.6 preserve-thinking tokenizer?"
            )
        return tokens[0]

    @property
    def has_extension_property(self) -> bool:
        return self.preserve_thinking

    @property
    def _end_message_token(self) -> int:
        return self._im_end_token

    @property
    def _stop_token_ids(self) -> list[int]:
        return [t for t in (self._im_end_token, self._endoftext_token) if t is not None]

    def get_stop_sequences(self) -> list[int]:
        # generation_config.json eos_token_id = [<|im_end|>, <|endoftext|>]; AReaL/SGLang stop on both.
        return self._stop_token_ids

    # ------------------------------------------------------- template mirror
    @staticmethod
    def _render_content(content: Any, *, is_system: bool = False) -> str:
        """Mirror of the template's ``render_content`` macro (text-only)."""
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
                    raise RendererError("Image content is not supported by the text-only Qwen3.6 renderer.")
                if "video" in item or item_type == "video":
                    if is_system:
                        raise RendererError("System message cannot contain videos.")
                    raise RendererError("Video content is not supported by the text-only Qwen3.6 renderer.")
                if "text" in item:
                    text = item["text"]
                    parts.append("" if text is None else str(text))
                elif item_type == "thinking":
                    # cookbook ThinkingPart -> inline think block (split again by _assistant_parts)
                    parts.append(f"{THINK_OPEN}{item.get('thinking', '')}{THINK_CLOSE}")
                else:
                    raise RendererError("Unexpected item type in content.")
            return "".join(parts)
        raise RendererError("Unexpected content type.")

    @classmethod
    def _assistant_parts(cls, message: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(reasoning, content)`` exactly as the template derives them."""
        content = cls._render_content(message.get("content")).strip()
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str):
            reasoning = reasoning_content
        else:
            reasoning = ""
            if THINK_CLOSE in content:
                reasoning = content.split(THINK_CLOSE)[0].rstrip("\n").split(THINK_OPEN)[-1].lstrip("\n")
                content = content.split(THINK_CLOSE)[-1].lstrip("\n")
        return reasoning.strip(), content

    @staticmethod
    def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Mapping[str, Any] | None]:
        """Accept OpenAI dicts (nested ``function`` or flat), cookbook/LiteLLM objects."""
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
            return name, None
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

    @classmethod
    def _format_tool_calls(cls, message: Mapping[str, Any], content: str) -> str:
        tool_calls = message.get("tool_calls")
        if not tool_calls or isinstance(tool_calls, Mapping) or not isinstance(tool_calls, Iterable):
            return ""
        out: list[str] = []
        for i, tool_call in enumerate(tool_calls):
            name, arguments = cls._tool_call_name_and_arguments(tool_call)
            if i == 0:
                out.append(("\n\n" if content.strip() else "") + f"{TOOL_CALL_OPEN}\n{FUNCTION_PREFIX}{name}>\n")
            else:
                out.append(f"\n{TOOL_CALL_OPEN}\n{FUNCTION_PREFIX}{name}>\n")
            if arguments is not None:
                for key, value in arguments.items():
                    out.append(f"{PARAMETER_PREFIX}{key}>\n")
                    out.append(value if isinstance(value, str) else _tojson(value))
                    out.append("\n</parameter>\n")
            out.append(f"</function>\n{TOOL_CALL_CLOSE}")
        return "".join(out)

    @staticmethod
    def format_tools_block(tools: Iterable[Any]) -> str:
        """The ``# Tools ... </IMPORTANT>`` text the template puts in the system turn."""
        return _TOOLS_HEADER + "".join("\n" + _tojson(tool) for tool in tools) + _TOOLS_FOOTER

    def create_conversation_prefix_with_tools(self, tools: list[Any], system_prompt: str = "") -> list[Message]:
        """cookbook API: system message carrying the tool declarations (tools dumped verbatim)."""
        prompt = self._render_content(system_prompt, is_system=True).strip()
        if tools and not isinstance(tools, Mapping):
            content = self.format_tools_block(tools) + (f"\n\n{prompt}" if prompt else "")
        else:
            content = prompt
        return [Message(role="system", content=content)]

    def _prepare_messages(self, messages: list[Message], tools: Iterable[Any] | None) -> list[Message]:
        if not messages:
            raise RendererError("No messages provided.")
        prepared = list(messages)
        if tools and not isinstance(tools, Mapping):
            first = prepared[0]
            if first.get("role") == "system":
                prefix = self.create_conversation_prefix_with_tools(list(tools), first.get("content"))[0]
                prepared[0] = {**first, "content": prefix["content"]}  # type: ignore[typeddict-item]
            else:
                prepared.insert(0, self.create_conversation_prefix_with_tools(list(tools))[0])
        return prepared

    @classmethod
    def last_query_index(cls, messages: list[Message]) -> int:
        """Index of the last user message that is not a ``<tool_response>`` wrapper.

        Mirrors the template's ``ns.last_query_index``; raises when no such user turn exists.
        """
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.get("role") == "user":
                content = cls._render_content(message.get("content")).strip()
                if not (content.startswith(TOOL_RESPONSE_OPEN) and content.endswith(TOOL_RESPONSE_CLOSE)):
                    return index
        raise RendererError("No user query found in messages.")

    def _contexts(
        self,
        messages: list[Message],
        *,
        preserve_thinking: bool | None,
        enable_thinking: bool | None,
    ) -> list[Qwen3_6RenderContext]:
        preserve = self.preserve_thinking if preserve_thinking is None else bool(preserve_thinking)
        enable = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        last_query = self.last_query_index(messages)
        last_user = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        contexts = []
        for idx in range(len(messages) + 1):  # +1: suffix context
            contexts.append(
                Qwen3_6RenderContext(
                    idx=idx,
                    is_last=(idx >= len(messages) - 1),
                    prev_message=messages[idx - 1] if idx > 0 else None,
                    last_user_index=last_user,
                    last_query_index=last_query,
                    preserve_thinking=preserve,
                    enable_thinking=enable,
                )
            )
        return contexts

    @staticmethod
    def _lead(ctx: RenderContext) -> str:
        """Text between the previous message and this ``<|im_start|>``."""
        if ctx.idx <= 0:
            return ""
        prev = ctx.prev_message
        return (IM_END if prev is not None and prev.get("role") == "tool" else "") + "\n"

    def _render_message_text(self, message: Message, ctx: RenderContext) -> _RenderedText:
        role = message.get("role")
        lead = self._lead(ctx)
        if role == "system":
            if ctx.idx != 0:
                raise RendererError("System message must be at the beginning.")
            content = self._render_content(message.get("content"), is_system=True).strip()
            return _RenderedText(header=f"{IM_START}system\n", output=f"{content}{IM_END}")
        if role == "user":
            content = self._render_content(message.get("content")).strip()
            return _RenderedText(header=f"{lead}{IM_START}user\n", output=f"{content}{IM_END}")
        if role == "assistant":
            preserve = getattr(ctx, "preserve_thinking", self.preserve_thinking)
            last_query = getattr(ctx, "last_query_index", ctx.last_user_index)
            reasoning, content = self._assistant_parts(message)
            header = f"{lead}{IM_START}assistant\n"
            if preserve or ctx.idx > last_query:
                if reasoning:
                    # Header ends where the generation prompt ends; safe because reasoning is trimmed.
                    header += f"{THINK_OPEN}\n"
                    body = f"{reasoning}\n{THINK_CLOSE}\n\n{content}"
                else:
                    # "<think>\n" + "\n</think>" must be tokenized together ("\n\n" is one token).
                    body = f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n{content}"
            else:
                body = content
            body += self._format_tool_calls(message, content)
            return _RenderedText(header=header, output=f"{body}{IM_END}")
        if role == "tool":
            content = self._render_content(message.get("content")).strip()
            prev = ctx.prev_message
            wrapped = f"{TOOL_RESPONSE_OPEN}\n{content}\n{TOOL_RESPONSE_CLOSE}"
            if prev is not None and prev.get("role") == "tool":
                return _RenderedText(header=None, output=f"\n{wrapped}", stop_overlap=IM_END)
            return _RenderedText(header=f"{lead}{IM_START}user\n", output=wrapped, stop_overlap=IM_END)
        raise RendererError("Unexpected message role.")

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        text = self._render_message_text(message, ctx)
        header = tinker.types.EncodedTextChunk(tokens=self._encode(text.header)) if text.header else None
        output: list[tinker.types.ModelInputChunk] = []
        if text.output:
            output.append(tinker.types.EncodedTextChunk(tokens=self._encode(text.output)))
        stop_overlap = (
            tinker.types.EncodedTextChunk(tokens=self._encode(text.stop_overlap)) if text.stop_overlap else None
        )
        return RenderedMessage(header=header, output=output, stop_overlap=stop_overlap)

    def _generation_suffix_text(self, role: Role, ctx: RenderContext) -> str:
        enable = getattr(ctx, "enable_thinking", self.enable_thinking)
        think = f"{THINK_OPEN}\n" if enable else f"{THINK_OPEN}\n\n{THINK_CLOSE}\n\n"
        return f"{self._lead(ctx)}{IM_START}{role}\n{think}"

    def _get_generation_suffix(self, role: Role, ctx: RenderContext) -> list[int]:
        return self._encode(self._generation_suffix_text(role, ctx))

    # ------------------------------------------------------------ public API
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
        prepared = self._prepare_messages(messages, tools)
        contexts = self._contexts(prepared, preserve_thinking=preserve_thinking, enable_thinking=enable_thinking)
        chunks: list[tinker.types.ModelInputChunk] = []
        for message, ctx in zip(prepared, contexts[:-1], strict=True):
            rendered = self.render_message(message, ctx)
            if rendered.header is not None and rendered.header.tokens:
                chunks.append(rendered.header)
            chunks.extend(x for x in rendered.output if not isinstance(x, tinker.types.EncodedTextChunk) or x.tokens)
        chunks.append(tinker.types.EncodedTextChunk(tokens=self._get_generation_suffix(role, contexts[-1])))
        if prefill:
            chunks.append(tinker.types.EncodedTextChunk(tokens=self._encode(prefill)))
        return tinker.ModelInput(chunks=chunks)

    def build_generation_prompt_with_tools(
        self, messages: list[Message], tools: Iterable[Any] | None, **kwargs: Any
    ) -> tinker.ModelInput:
        """Alias for ``build_generation_prompt(messages, tools=tools, **kwargs)``."""
        return self.build_generation_prompt(messages, tools=tools, **kwargs)

    def render_conversation_text(
        self,
        messages: list[Message],
        *,
        tools: Iterable[Any] | None = None,
        add_generation_prompt: bool = True,
        role: Role = "assistant",
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
    ) -> str:
        """String form; equals ``tokenizer.apply_chat_template(..., tokenize=False)``."""
        prepared = self._prepare_messages(messages, tools)
        contexts = self._contexts(prepared, preserve_thinking=preserve_thinking, enable_thinking=enable_thinking)
        pieces: list[str] = []
        last_stop_overlap: str | None = None
        for message, ctx in zip(prepared, contexts[:-1], strict=True):
            text = self._render_message_text(message, ctx)
            pieces.append((text.header or "") + text.output)
            last_stop_overlap = text.stop_overlap
        if add_generation_prompt:
            pieces.append(self._generation_suffix_text(role, contexts[-1]))
        else:
            pieces.append((last_stop_overlap or "") + "\n")  # template ends every turn with <|im_end|>\n
        return "".join(pieces)

    def build_supervised_example(
        self,
        messages: list[Message],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        *,
        tools: Iterable[Any] | None = None,
        preserve_thinking: bool | None = None,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Same tokens as the concatenated ``build_generation_prompt`` prefixes; base weight semantics."""
        prepared = self._prepare_messages(messages, tools)
        contexts = self._contexts(prepared, preserve_thinking=preserve_thinking, enable_thinking=None)
        chunk_weights: list[tuple[tinker.types.ModelInputChunk, float]] = []
        last_user_idx = contexts[0].last_user_index
        for idx, (message, ctx) in enumerate(zip(prepared, contexts[:-1], strict=True)):
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                assert "trainable" in message, "CUSTOMIZED train_on_what requires a trainable field on every message"
            else:
                assert "trainable" not in message, "trainable field is only allowed with CUSTOMIZED train_on_what"
            is_last = idx == len(prepared) - 1
            is_assistant = message.get("role") == "assistant"
            is_user_or_system = message.get("role") in ("user", "system")
            is_after_last_user = last_user_idx == -1 or idx > last_user_idx
            rendered = self.render_message(message, ctx)
            if rendered.header is not None:
                chunk_weights.append((rendered.header, float(train_on_what == TrainOnWhat.ALL_TOKENS)))
            match train_on_what:
                case TrainOnWhat.LAST_ASSISTANT_MESSAGE:
                    has_weight = is_last and is_assistant
                case TrainOnWhat.LAST_ASSISTANT_TURN:
                    has_weight = is_assistant and is_after_last_user
                case TrainOnWhat.ALL_ASSISTANT_MESSAGES:
                    has_weight = is_assistant
                case TrainOnWhat.ALL_MESSAGES | TrainOnWhat.ALL_TOKENS:
                    has_weight = True
                case TrainOnWhat.ALL_USER_AND_SYSTEM_MESSAGES:
                    has_weight = is_user_or_system
                case TrainOnWhat.CUSTOMIZED:
                    has_weight = bool(message.get("trainable", False))
                case _:
                    raise RendererError(f"Unknown train_on_what: {train_on_what}")
            chunk_weights.extend((part, float(has_weight)) for part in rendered.output if part)
            if is_last and rendered.stop_overlap is not None:
                chunk_weights.append((rendered.stop_overlap, float(has_weight)))
        weights = torch.tensor([w for chunk, w in chunk_weights for _ in range(chunk.length)])
        return tinker.ModelInput(chunks=[chunk for chunk, _ in chunk_weights]), weights

    # --------------------------------------------------------------- parsing
    def _normalize_response_tokens(self, response: list[int]) -> list[int]:
        """Restore the prefilled ``<think>\\n`` so the cookbook streaming parser sees the think block."""
        if not self.enable_thinking:
            return response
        prefix = self._encode(f"{THINK_OPEN}\n")
        if response[: len(prefix)] == prefix or (response and response[0] == self._marker_tokens[THINK_OPEN]):
            return response
        return prefix + list(response)

    def _split_at_stop(self, response: list[int]) -> tuple[str, bool]:
        stop_ids = set(self._stop_token_ids)
        for i, token in enumerate(response):
            if token in stop_ids:
                return str(self.tokenizer.decode(response[:i])), True
        return str(self.tokenizer.decode(list(response))), False

    def parse_response(
        self, response: list[int], tools: Iterable[Any] | None = None, *, enable_thinking: bool | None = None
    ) -> tuple[Message, bool]:
        """Parse sampled tokens (after the ``<think>\\n`` generation prompt) into an assistant Message.

        ``success`` is False when no stop token (``<|im_end|>``/``<|endoftext|>``) was sampled;
        the message is still parsed on a best-effort basis (truncated reasoning ends up in
        ``reasoning_content`` with empty ``content``, like SGLang's force-reasoning mode).
        ``enable_thinking`` must match the request's generation prompt when it overrides
        the renderer default (for example, a nonthinking utility summary).
        """
        text, success = self._split_at_stop(response)
        return self._parse_text(text, tools, enable_thinking=enable_thinking), success

    def _parse_response_for_streaming(self, response: list[int]) -> tuple[Message, bool]:
        text, success = self._split_at_stop(response)
        return self._parse_text(text, None), success

    def _parse_text(
        self, text: str, tools: Iterable[Any] | None, *, enable_thinking: bool | None = None
    ) -> Message:
        enable = self.enable_thinking if enable_thinking is None else enable_thinking
        if THINK_CLOSE in text:
            # Template semantics on re-render: reasoning before the FIRST </think>, content after the LAST.
            reasoning = text.split(THINK_CLOSE)[0].split(THINK_OPEN)[-1].strip()
            body = text.split(THINK_CLOSE)[-1]
        elif enable or THINK_OPEN in text:
            # Prompt ended in <think>\n and the model never closed it: everything is reasoning.
            reasoning = text.replace(THINK_OPEN, "").strip()
            body = ""
        else:
            reasoning = ""
            body = text
        tool_calls, unparsed, content = self._parse_tool_calls(body, list(tools) if tools else None)
        message: dict[str, Any] = {"role": "assistant", "content": content.strip(), "reasoning_content": reasoning}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if unparsed:
            message["unparsed_tool_calls"] = unparsed
        return message  # type: ignore[return-value]

    def _parse_tool_calls(
        self, body: str, tools: list[Any] | None
    ) -> tuple[list[ToolCall], list[UnparsedToolCall], str]:
        if TOOL_CALL_OPEN not in body:
            return [], [], body
        areal = self._parse_tool_calls_areal(body, _tool_argument_schemas(tools))
        if areal is not None:
            return areal[0], [], areal[1]
        return self._parse_tool_calls_sglang(body, tools)

    @staticmethod
    def _parse_tool_calls_areal(
        body: str, schemas: dict[str, dict[str, dict[str, Any]]]
    ) -> tuple[list[ToolCall], str] | None:
        """Port of AReaL ``_process_tool_calls_qwen3_coder_xml``; None => fall back to the SGLang parser."""
        calls: list[ToolCall] = []
        spans: list[tuple[int, int]] = []
        for tool_match in _AREAL_TOOL_CALL_RE.finditer(body):
            block_calls: list[ToolCall] = []
            for fn_match in _AREAL_FUNCTION_RE.finditer(tool_match.group("body")):
                name = fn_match.group("name")
                fn_body = fn_match.group("body")
                if fn_body.count(PARAMETER_PREFIX) != fn_body.count("</parameter>"):
                    return None
                args: dict[str, Any] = {}
                for param in _AREAL_PARAMETER_RE.finditer(fn_body):
                    raw_value = param.group("value")
                    if PARAMETER_PREFIX in raw_value or "</parameter>" in raw_value:
                        return None
                    param_name = param.group("name")
                    args[param_name] = _coerce_areal(
                        _strip_one_newline(raw_value), schemas.get(name, {}).get(param_name)
                    )
                if not args:
                    continue  # zero-argument calls are left to the SGLang fallback
                block_calls.append(_make_tool_call(name, args))
            if block_calls:
                calls.extend(block_calls)
                spans.append(tool_match.span())
        if not calls:
            return None
        chunks: list[str] = []
        last = 0
        for start, end in spans:
            chunks.append(body[last:start])
            last = end
        chunks.append(body[last:])
        return calls, "".join(chunks).replace(IM_END, "")

    @staticmethod
    def _parse_tool_calls_sglang(
        body: str, tools: list[Any] | None
    ) -> tuple[list[ToolCall], list[UnparsedToolCall], str]:
        """Port of SGLang ``Qwen3CoderDetector.detect_and_parse`` (non-streaming)."""
        calls: list[ToolCall] = []
        try:
            raw_tool_calls = _SGL_TOOL_CALL_RE.findall(body)
            if not raw_tool_calls and FUNCTION_PREFIX in body:
                raw_tool_calls = [body]
            for tool_content in raw_tool_calls:
                for func_match in _SGL_FUNCTION_RE.findall(tool_content):
                    func_body = func_match[0] or func_match[1]
                    if ">" not in func_body:
                        continue
                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]
                    param_config = _sglang_param_config(func_name, tools)
                    parsed: dict[str, Any] = {}
                    for p_match in _SGL_PARAMETER_RE.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        parsed[p_match[:p_idx]] = _convert_sglang(
                            _strip_one_newline(p_match[p_idx + 1 :]), p_match[:p_idx], param_config
                        )
                    calls.append(_make_tool_call(func_name, parsed))
        except Exception as e:  # SGLang logs and returns the text unchanged
            return [], [UnparsedToolCall(raw_text=body, error=f"qwen3_coder fallback parser failed: {e}")], body
        start = body.find(TOOL_CALL_OPEN)
        if start == -1:
            start = body.find(FUNCTION_PREFIX)
        content = body[:start] if start > 0 else ""
        unparsed = (
            []
            if calls
            else [UnparsedToolCall(raw_text=body[max(start, 0) :], error="Malformed Qwen3-Coder tool call XML")]
        )
        return calls, unparsed, content

    # ------------------------------------------------------------- OpenAI IO
    @classmethod
    def _tool_call_to_openai(cls, tool_call: Any) -> dict[str, Any]:
        if isinstance(tool_call, ToolCall):
            return {
                "type": "function",
                "id": tool_call.id,
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
        name, arguments = cls._tool_call_name_and_arguments(tool_call)
        tool_id = tool_call.get("id") if isinstance(tool_call, Mapping) else getattr(tool_call, "id", None)
        return {
            "type": "function",
            "id": tool_id,
            "function": {"name": name, "arguments": _tojson(dict(arguments) if arguments is not None else {})},
        }

    def to_openai_message(self, message: Message) -> dict:
        """OpenAI dict (``reasoning_content`` + JSON-string ``arguments``) that re-renders identically."""
        result: dict[str, Any] = {"role": message["role"]}
        content = message.get("content")
        reasoning = message.get("reasoning_content")  # type: ignore[typeddict-item]
        if isinstance(content, list):
            texts: list[str] = []
            thinks: list[str] = []
            for part in content:
                if part["type"] == "text":
                    texts.append(part["text"])
                elif part["type"] == "thinking":
                    thinks.append(part["thinking"])
                else:
                    raise NotImplementedError("to_openai_message does not support image content")
            result["content"] = "".join(texts)
            if thinks and not isinstance(reasoning, str):
                reasoning = "".join(thinks)
        else:
            result["content"] = "" if content is None else content
        if isinstance(reasoning, str):
            result["reasoning_content"] = reasoning
        tool_calls = message.get("tool_calls")
        if tool_calls:
            result["tool_calls"] = [self._tool_call_to_openai(tc) for tc in tool_calls]
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]
        return result


class Qwen3_6PreserveThinkingDisableThinkingRenderer(Qwen3_6PreserveThinkingRenderer):
    """``enable_thinking=False`` variant: generation prompt ends with ``<think>\\n\\n</think>\\n\\n``.

    History turns still render their (possibly empty) think blocks exactly like the template
    (``enable_thinking`` only affects the generation prompt).  Intended for the condenser /
    judges; combine with ``preserve_thinking=False`` via ``renderer_kwargs`` to reproduce the
    AReaL condenser request (``chat_template_kwargs={"enable_thinking": False,
    "preserve_thinking": False}``).
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        image_processor: Any | None = None,
        *,
        preserve_thinking: bool = True,
        enable_thinking: bool = False,
    ):
        super().__init__(
            tokenizer, image_processor, preserve_thinking=preserve_thinking, enable_thinking=enable_thinking
        )


# ------------------------------------------------------------- parse helpers
def _strip_one_newline(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _make_tool_call(name: str, arguments: Mapping[str, Any]) -> ToolCall:
    return ToolCall(
        id=_new_tool_call_id(),
        function=ToolCall.FunctionBody(name=name, arguments=json.dumps(arguments, ensure_ascii=False)),
    )


def _iter_tool_definitions(tools: list[Any] | None) -> list[dict[str, Any]]:
    """OpenAI-wrapped or flat function definitions -> flat dicts (AReaL ``_iter_tool_definitions``)."""
    defs: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            continue
        if isinstance(tool.get("function"), Mapping):
            defs.append(dict(tool["function"]))
        elif tool.get("type") == "function":
            defs.append(dict(tool))
    return defs


def _tool_argument_schemas(tools: list[Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    schemas: dict[str, dict[str, dict[str, Any]]] = {}
    for tool_def in _iter_tool_definitions(tools):
        name = tool_def.get("name")
        parameters = tool_def.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, Mapping):
            schemas[name] = {k: v for k, v in properties.items() if isinstance(v, Mapping)}
    return schemas


def _coerce_areal(value: str, schema: Mapping[str, Any] | None) -> Any:
    """AReaL ``_coerce_qwen3_coder_parameter``: schema-typed coercion, else the raw string."""
    if not schema:
        return value
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    stripped = value.strip()
    try:
        if typ == "boolean":
            lowered = stripped.lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        if typ == "integer":
            return int(stripped)
        if typ == "number":
            return float(stripped)
        if typ in ("array", "object"):
            return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value
    return value


def _sglang_param_config(func_name: str, tools: list[Any] | None) -> dict[str, Any]:
    """SGLang ``Qwen3CoderDetector._get_arguments_config``."""
    if tools is None:
        return {}
    for tool_def in _iter_tool_definitions(tools):
        if tool_def.get("name") != func_name:
            continue
        params = tool_def.get("parameters")
        if isinstance(params, Mapping) and "properties" in params:
            return dict(params["properties"])
        if isinstance(params, Mapping):
            return dict(params)
        return {}
    return {}


def _convert_sglang(param_value: str, param_name: str, param_config: Mapping[str, Any]) -> Any:
    """SGLang ``Qwen3CoderDetector._convert_param_value``."""
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    config = param_config[param_name]
    param_type = str(config["type"]).strip().lower() if isinstance(config, Mapping) and "type" in config else "string"
    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(param_value)
        except Exception:
            return param_value
    if param_type.startswith(("num", "float")):
        try:
            maybe_convert = not ("." in param_value or "e" in param_value.lower())
            converted = float(param_value)
            if maybe_convert and converted.is_integer():
                return int(converted)
            return converted
        except Exception:
            return param_value
    if param_type in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    if param_type in ("object", "array", "arr") or param_type.startswith(("dict", "list")):
        try:
            return json.loads(param_value)
        except Exception:
            pass
    try:
        return ast.literal_eval(param_value)
    except Exception:
        return param_value


# ---------------------------------------------------------------- registry
def _factory(tokenizer: Tokenizer, image_processor: Any | None = None) -> Renderer:
    return Qwen3_6PreserveThinkingRenderer(tokenizer, image_processor)


def _factory_disable_thinking(tokenizer: Tokenizer, image_processor: Any | None = None) -> Renderer:
    return Qwen3_6PreserveThinkingDisableThinkingRenderer(tokenizer, image_processor)


RENDERER_FACTORIES = {
    RENDERER_NAME: _factory,
    RENDERER_NAME_DISABLE_THINKING: _factory_disable_thinking,
}


def register(force: bool = False) -> list[str]:
    """Register both renderers with ``tinker_cookbook.renderers.register_renderer`` (idempotent).

    Returns the registered names.  Must run in every process that calls ``get_renderer``
    with these names (including processes that unpickle a renderer).
    """
    from tinker_cookbook.renderers import is_renderer_registered, register_renderer

    for name, factory in RENDERER_FACTORIES.items():
        if force or not is_renderer_registered(name):
            register_renderer(name, factory)
    return list(RENDERER_FACTORIES)


__all__ = [
    "RENDERER_FACTORIES",
    "RENDERER_NAME",
    "RENDERER_NAME_DISABLE_THINKING",
    "Qwen3_6PreserveThinkingDisableThinkingRenderer",
    "Qwen3_6PreserveThinkingRenderer",
    "Qwen3_6RenderContext",
    "register",
]
