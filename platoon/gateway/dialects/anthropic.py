"""Anthropic Messages (Claude Code), experimental.

``system`` becomes the system prompt; an assistant turn's ``thinking``, ``text`` and ``tool_use`` blocks become
its reasoning, content and tool calls; a user turn's ``tool_result`` blocks become tool messages followed by its
text.  Thinking blocks carry a gateway-issued ``signature`` that the gateway never checks, since a rejected
signature makes Claude Code strip earlier thinking.  ``thinking.type == "disabled"`` renders without thinking.
``tool_choice`` ``any``/``tool``, ``stop_sequences`` and image blocks are rejected.  A prompt that overflows the
context window gets Anthropic's "prompt is too long" error, which triggers Claude Code's reactive compaction.

A hosted gateway (:mod:`platoon.gateway.hosted`) forwards a Messages body (and a ``count_tokens`` one) as
:func:`hosted_body` leaves it, with the tool rules :meth:`AnthropicMessages.chat_request` applies
(:func:`tool_functions`, :func:`check_assistant_blocks`); :func:`assemble_events` and :func:`reply_fields` read the
provider's reply for the call's record.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from platoon.gateway.core import (
    ChatRequest,
    ChatResponse,
    ContextWindowExceededError,
    SamplingOverrides,
    check_tool_choice,
    tool_function,
)
from platoon.gateway.dialects import ErrorKind, error_kind, sse

_ERROR_TYPES: dict[ErrorKind, str] = {
    "authentication": "authentication_error",
    "permission": "permission_error",
    "agent_closed": "permission_error",
    "context_length": "invalid_request_error",
    "invalid_request": "invalid_request_error",
    "timeout": "timeout_error",
    "server": "api_error",
}
_STOP_REASONS = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}
# The content blocks of an assistant turn that training renders; any other belongs to a tool training never offers
# (a server tool, an MCP connector, a container) and is refused.
ASSISTANT_BLOCK_TYPES = ("thinking", "redacted_thinking", "text", "tool_use")


def _blocks(content: Any) -> list[Mapping[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or not all(isinstance(block, Mapping) for block in content):
        raise ValueError("Message content must be a string or an array of content blocks")
    return content


def _text(content: Any) -> str:
    texts = []
    for block in _blocks(content):
        if block.get("type") != "text":
            raise ValueError(f"Unsupported content block type {block.get('type')!r}")
        texts.append(block.get("text") or "")
    return "".join(texts)


def check_assistant_blocks(content: Any) -> list[Mapping[str, Any]]:
    """An assistant turn's content blocks; a block of a tool that training never offers (anything but
    :data:`ASSISTANT_BLOCK_TYPES`: ``server_tool_use``, a server tool's result, ``mcp_tool_use``, ...) is a
    ValueError, for training and the hosted pass-through alike."""

    blocks = _blocks(content)
    for block in blocks:
        if block.get("type") not in ASSISTANT_BLOCK_TYPES:
            raise ValueError(f"Unsupported assistant content block type {block.get('type')!r}")
    return blocks


def _assistant_message(content: Any) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    for block in check_assistant_blocks(content):
        kind = block.get("type")
        if kind == "thinking":
            message["reasoning_content"] = message.get("reasoning_content", "") + (block.get("thinking") or "")
        elif kind == "text":
            message["content"] += block.get("text") or ""
        elif kind == "tool_use":
            function = {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}
            message.setdefault("tool_calls", []).append(
                {"id": block.get("id"), "type": "function", "function": function}
            )
    return message


def _user_messages(content: Any) -> list[dict[str, Any]]:
    """Tool results first, as tool messages, then the turn's text (Anthropic puts results before text)."""
    messages: list[dict[str, Any]] = []
    texts: list[str] = []
    for block in _blocks(content):
        kind = block.get("type")
        if kind == "tool_result":
            messages.append(
                {"role": "tool", "tool_call_id": block.get("tool_use_id"), "content": _text(block.get("content") or "")}
            )
        elif kind == "text":
            texts.append(block.get("text") or "")
        else:
            raise ValueError(f"Unsupported user content block type {kind!r}")
    if texts:
        messages.append({"role": "user", "content": "".join(texts)})
    return messages


def _tool_input(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except ValueError:
        parsed = None
    # Never fabricate structure the model did not produce: unparseable arguments stay visible as sampled.
    return parsed if isinstance(parsed, dict) else {"_raw_arguments": arguments}


def tool_functions(tools: Any) -> list[dict[str, Any]]:
    """The tools a policy is shown of a Messages request's ``tools``, as OpenAI function tools: every entry with a
    name, whatever its ``type``, becomes a function of its ``name``, its ``description`` and its ``input_schema``
    (:func:`~platoon.gateway.core.tool_function`, which gives one without a schema the empty-object schema), and every
    other field is dropped.  So a server tool (web search, code execution, tool search, ...) is a function of its name,
    which nothing runs at the provider, and a deferred tool (``defer_loading``) is shown up front.  An entry without a
    name (an MCP toolset) is left out, since no model can call it.  ``tools`` that is not a list, and an entry that is
    not an object, are a ValueError, for training and the hosted pass-through alike."""

    if tools and not isinstance(tools, list):
        raise ValueError(f"Invalid tools type: {type(tools)}")
    functions: list[dict[str, Any]] = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            raise ValueError(f"Invalid tool definition: {tool!r}")
        function = tool_function(
            {"name": tool.get("name"), "description": tool.get("description"), "parameters": tool.get("input_schema")}
        )
        if function is not None:
            functions.append({"type": "function", "function": function})
    return functions


def tool_choice_type(tool_choice: Any) -> Any:
    """A Messages ``tool_choice`` as the gateway checks it (:func:`~platoon.gateway.core.check_tool_choice`): an
    object's ``type`` (``auto``, ``any``, ``tool`` or ``none``)."""

    return tool_choice.get("type") if isinstance(tool_choice, Mapping) else tool_choice


class AnthropicMessages:
    """The Messages dialect (``POST /v1/messages`` and ``/v1/messages/count_tokens``)."""

    name = "anthropic_messages"

    def chat_request(self, body: Mapping[str, Any]) -> ChatRequest:
        messages: list[dict[str, Any]] = []
        if body.get("system"):
            messages.append({"role": "system", "content": _text(body["system"])})
        turns = body.get("messages")
        if not isinstance(turns, list) or not turns:
            raise ValueError("'messages' must be a non-empty array")
        for turn in turns:
            role = turn.get("role") if isinstance(turn, Mapping) else None
            if role == "assistant":
                messages.append(_assistant_message(turn.get("content")))
            elif role == "user":
                messages.extend(_user_messages(turn.get("content")))
            else:
                raise ValueError(f"Unsupported message role {role!r}")
        if body.get("stop_sequences"):
            raise ValueError("stop_sequences are not supported: they would replace the renderer's end-of-turn stops")
        params: dict[str, Any] = {
            "tools": tool_functions(body.get("tools")),
            "tool_choice": tool_choice_type(body.get("tool_choice")),
            "max_tokens": body.get("max_tokens"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
            "top_k": body.get("top_k"),
        }
        if (body.get("thinking") or {}).get("type") == "disabled":
            params["reasoning_effort"] = "none"
        return ChatRequest.from_params(messages, params)

    def response(self, response: ChatResponse, body: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": response.call_id,
            "type": "message",
            "role": "assistant",
            "model": response.model,
            "content": self._content(response),
            "stop_reason": _STOP_REASONS[response.finish_reason],
            "stop_sequence": None,
            "usage": {"input_tokens": response.prompt_tokens, "output_tokens": response.completion_tokens},
        }

    def events(self, response: ChatResponse, body: Mapping[str, Any]) -> Iterator[bytes]:
        message = self.response(response, body)
        start = {**message, "content": [], "stop_reason": None, "usage": {**message["usage"], "output_tokens": 0}}
        yield sse({"type": "message_start", "message": start}, event="message_start")
        for index, block in enumerate(message["content"]):
            if block["type"] == "thinking":
                opened = {"type": "thinking", "thinking": "", "signature": ""}
                deltas = [
                    {"type": "thinking_delta", "thinking": block["thinking"]},
                    {"type": "signature_delta", "signature": block["signature"]},
                ]
            elif block["type"] == "text":
                opened, deltas = {"type": "text", "text": ""}, [{"type": "text_delta", "text": block["text"]}]
            else:
                opened = {**block, "input": {}}
                deltas = [{"type": "input_json_delta", "partial_json": json.dumps(block["input"])}]
            yield sse(
                {"type": "content_block_start", "index": index, "content_block": opened}, event="content_block_start"
            )
            for delta in deltas:
                yield sse({"type": "content_block_delta", "index": index, "delta": delta}, event="content_block_delta")
            yield sse({"type": "content_block_stop", "index": index}, event="content_block_stop")
        delta = {"stop_reason": message["stop_reason"], "stop_sequence": None}
        usage = {"output_tokens": message["usage"]["output_tokens"]}
        yield sse({"type": "message_delta", "delta": delta, "usage": usage}, event="message_delta")
        yield sse({"type": "message_stop"}, event="message_stop")

    def error(self, exc: BaseException) -> dict[str, Any]:
        kind = error_kind(exc)
        message = str(exc)
        if isinstance(exc, ContextWindowExceededError):
            message = f"prompt is too long: {exc.prompt_tokens} tokens > {exc.context_window} maximum"
        elif kind == "server":
            message = f"{type(exc).__name__}: {exc}"
        return {"type": "error", "error": {"type": _ERROR_TYPES[kind], "message": message}}

    def error_event(self, exc: BaseException) -> bytes:
        return sse(self.error(exc), event="error")

    @staticmethod
    def _content(response: ChatResponse) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if response.reasoning_content is not None:
            signature = f"platoon:{response.call_id}"
            content.append({"type": "thinking", "thinking": response.reasoning_content, "signature": signature})
        if response.content or not response.tool_calls:
            content.append({"type": "text", "text": response.content})
        for tool_call in response.tool_calls or []:
            function = tool_call["function"]
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": function["name"],
                    "input": _tool_input(function["arguments"]),
                }
            )
        return content


# --------------------------------------------------------------------------- forwarding to a hosted model
#
# A hosted gateway (platoon.gateway.hosted) forwards a Messages request in this dialect to its base URL.  The hosted
# model gets the tools AnthropicMessages.chat_request shows a trained policy, written as Messages tools, and the
# request carries nothing that constrains its calls more than training does.

# Fields a hosted model never gets: none of them reaches a trained policy.  ``mcp_servers`` and ``container`` would
# give the provider remote MCP servers and a code-execution container, and ``context_management`` and ``compaction``
# would edit the conversation at the provider.  ``tool_choice`` goes too once training's check passed it (``auto``
# or ``none``), and with it ``disable_parallel_tool_use``.
HOSTED_DROPPED_FIELDS = ("tool_choice", "mcp_servers", "container", "context_management", "compaction")
# Stop reasons that end a message at its token limit.
_LENGTH_STOPS = ("max_tokens", "model_context_window_exceeded")


def _server_tool_block(kind: Any) -> bool:
    """Whether a content block belongs to a tool training never offers: a server tool's use or result, an MCP
    connector's, or a file uploaded to a code-execution container."""

    if not isinstance(kind, str):
        return False
    if kind in ("server_tool_use", "mcp_tool_use", "mcp_tool_result", "container_upload"):
        return True
    return kind.endswith("_tool_result") and kind != "tool_result"


def _check_history(turns: Any) -> None:
    for turn in turns if isinstance(turns, list) else []:
        role = turn.get("role") if isinstance(turn, Mapping) else None
        if role == "assistant":
            check_assistant_blocks(turn.get("content"))
        elif role == "user":
            for block in _blocks(turn.get("content")):
                if _server_tool_block(block.get("type")):
                    raise ValueError(f"Unsupported user content block type {block.get('type')!r}")


def _hosted_tool(function: Mapping[str, Any]) -> dict[str, Any]:
    """A function a policy is shown (:func:`tool_functions`) as a Messages tool: its parameters are its
    ``input_schema``."""

    return {("input_schema" if key == "parameters" else key): value for key, value in function.items()}


def hosted_body(
    body: Mapping[str, Any],
    *,
    model: str,
    defaults: SamplingOverrides | None = None,
    count_tokens: bool = False,
) -> dict[str, Any]:
    """A Messages body (or a ``count_tokens`` one) as the hosted model gets it: ``model`` in place of the request's,
    the tools :func:`tool_functions` shows a trained policy, as Messages tools of a ``name``, a ``description`` and an
    ``input_schema`` (the empty-object schema for one that gives none; a toolset, which has no name, is left out),
    none of :data:`HOSTED_DROPPED_FIELDS`, and the sampling parameters it leaves unset taken from ``defaults`` (the
    rollout's ``inference_params``: ``max_tokens``, ``temperature``, ``top_p``; not for ``count_tokens``, which takes
    none).  Everything else passes unchanged, images, stop sequences and thinking included.

    Refuses, with training's ValueError and message, what training refuses of tools (tools that are not a list, an
    entry that is not an object), the history of tools training never offers (:func:`check_assistant_blocks`, and such
    blocks in a user turn) and a ``tool_choice`` other than ``auto`` and ``none``.
    """

    _check_history(body.get("messages"))
    functions = [tool["function"] for tool in tool_functions(body.get("tools"))]
    check_tool_choice(tool_choice_type(body.get("tool_choice")))
    out = {key: value for key, value in body.items() if key not in HOSTED_DROPPED_FIELDS}
    tools = [_hosted_tool(function) for function in functions]
    if tools:
        out["tools"] = tools
    else:
        out.pop("tools", None)
    out["model"] = model
    if defaults is not None and not count_tokens:
        sampling = {"max_tokens": defaults.max_tokens, "temperature": defaults.temperature, "top_p": defaults.top_p}
        for key, value in sampling.items():
            if out.get(key) is None and value is not None:
                out[key] = value
    return out


def assemble_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Streamed Messages events as the message they make up: its content blocks with their text, thinking,
    signatures, citations and tool input, its stop reason and its usage.  Raises ``ValueError`` for a stream that
    ended before ``message_stop``."""

    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}
    stopped = False
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = dict(event.get("message") or {})
            message["usage"] = dict(message.get("usage") or {})
        elif kind == "content_block_start":
            index = int(event.get("index", len(blocks)))
            block = dict(event.get("content_block") or {})
            blocks[index] = block
            if "input" in block and not block["input"]:
                partial_json[index] = ""
        elif kind == "content_block_delta":
            index = int(event.get("index", 0))
            block = blocks.setdefault(index, {})
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
            elif delta_type == "input_json_delta":
                partial_json[index] = partial_json.get(index, "") + delta.get("partial_json", "")
            elif delta_type == "citations_delta":
                block.setdefault("citations", []).append(delta.get("citation"))
        elif kind == "content_block_stop":
            index = int(event.get("index", 0))
            if partial_json.get(index):
                blocks[index]["input"] = _tool_input(partial_json.pop(index))
        elif kind == "message_delta" and message is not None:
            message.update({key: value for key, value in (event.get("delta") or {}).items()})
            # Cumulative counts that replace message_start's; one left null (or out) keeps message_start's.
            counts = event.get("usage") or {}
            message["usage"].update({key: value for key, value in counts.items() if value is not None})
        elif kind == "message_stop":
            stopped = True
    if message is None or not stopped:
        raise ValueError("the response stream ended before message_stop")
    message["content"] = [blocks[index] for index in sorted(blocks)]
    return message


def reply_fields(message: Mapping[str, Any]) -> dict[str, Any]:
    """What a call's record takes from a Messages message: ``content`` (its text blocks), ``reasoning_content`` (its
    thinking), ``tool_calls`` (its ``tool_use`` blocks as OpenAI tool calls), ``finish_reason`` (the stop reason as
    sent), ``truncated`` (stopped at the token limit), ``refusal``, ``usage`` (normalized: ``prompt_tokens`` counts
    cache reads and writes too, which ``cached_prompt_tokens`` and ``cache_write_tokens`` repeat) and
    ``usage_reported`` (false when the message carries no token counts)."""

    texts, thinking, tool_calls = [], [], []
    for block in message.get("content") or []:
        kind = block.get("type") if isinstance(block, Mapping) else None
        if kind == "text":
            texts.append(block.get("text") or "")
        elif kind == "thinking":
            thinking.append(block.get("thinking") or "")
        elif kind == "tool_use":
            arguments = json.dumps(block.get("input") if block.get("input") is not None else {})
            function = {"name": block.get("name"), "arguments": arguments}
            tool_calls.append({"id": block.get("id"), "type": "function", "function": function})
    stop_reason = message.get("stop_reason")
    usage = message.get("usage") or {}
    read = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + read + written
    completion = int(usage.get("output_tokens") or 0)
    normalized = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if read:
        normalized["cached_prompt_tokens"] = read
    if written:
        normalized["cache_write_tokens"] = written
    details = usage.get("output_tokens_details")
    thought = details.get("thinking_tokens") if isinstance(details, Mapping) else None
    if thought:  # the output tokens spent thinking, which output_tokens includes
        normalized["reasoning_tokens"] = int(thought)
    return {
        "content": "".join(texts),
        "reasoning_content": "".join(thinking) or None,
        "tool_calls": tool_calls or None,
        "finish_reason": stop_reason,
        "truncated": stop_reason in _LENGTH_STOPS,
        "refusal": stop_reason == "refusal",
        "usage": normalized,
        "usage_reported": any(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens")),
    }
