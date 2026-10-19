"""OpenAI Chat Completions: messages with tool calls and tool results, ``tools`` with ``tool_choice`` auto/none,
``reasoning_content`` (as vLLM and DeepSeek return it), usage, finish reasons and optional logprobs.

Streaming is synthetic: the role, reasoning, content and tool-call chunks, the finish chunk, the usage chunk when
``stream_options.include_usage`` asks for it, and ``[DONE]``.

A hosted gateway (:mod:`platoon.gateway.hosted`) forwards a Chat Completions body as :func:`hosted_body` leaves it,
and :func:`in_process_body` is ``ctx.chat``'s call as such a body; :func:`assemble_chunks` and :func:`reply_fields`
read the provider's reply for the call's record.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

from platoon.gateway.core import (
    ChatRequest,
    ChatResponse,
    ContextWindowExceededError,
    ModelGateway,
    RenderRequest,
    SamplingOverrides,
)
from platoon.gateway.dialects import ErrorKind, error_kind, sse

_ERROR_CODES: dict[ErrorKind, str | None] = {
    "authentication": "invalid_api_key",
    "permission": "permission_denied",
    "agent_closed": "agent_closed",
    "context_length": "context_length_exceeded",
    "invalid_request": None,
    "timeout": "sample_timeout",
    "server": None,
}
_ERROR_TYPES: dict[ErrorKind, str] = {"permission": "permission_error", "timeout": "timeout", "server": "server_error"}


class OpenAIChat:
    """The Chat Completions dialect; it decodes sampled tokens with ``gateway``'s tokenizer for logprobs."""

    name = "openai_chat"

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def chat_request(self, body: Mapping[str, Any]) -> ChatRequest:
        if body.get("n") not in (None, 1):
            raise ValueError("Only n=1 is supported")
        if not isinstance(body.get("stream_options") or {}, dict):
            raise ValueError("'stream_options' must be an object")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' must be a non-empty array")
        if not all(isinstance(message, dict) and isinstance(message.get("role"), str) for message in messages):
            raise ValueError("Every message must be an object with a string 'role'")
        params = dict(body)
        if isinstance(params.get("stop"), str):
            params["stop"] = [params["stop"]]
        # Newer OpenAI clients send instructions as "developer"; chat templates only know "system".
        messages = [
            {**message, "role": "system"} if message["role"] == "developer" else message for message in messages
        ]
        return ChatRequest.from_params(messages, params)

    def response(self, response: ChatResponse, body: Mapping[str, Any]) -> dict[str, Any]:
        message = response.message()
        if response.tool_calls and not response.content:
            message["content"] = None
        choice = {
            "index": 0,
            "message": message,
            "logprobs": self._logprobs(response, body),
            "finish_reason": response.finish_reason,
        }
        return {
            "id": response.call_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [choice],
            "usage": response.usage(),
        }

    def events(self, response: ChatResponse, body: Mapping[str, Any]) -> Iterator[bytes]:
        deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
        if response.reasoning_content:
            deltas.append({"reasoning_content": response.reasoning_content})
        if response.content:
            deltas.append({"content": response.content})
        for index, tool_call in enumerate(response.tool_calls or []):
            deltas.append({"tool_calls": [{"index": index, **tool_call}]})
        choices: list[dict[str, Any]] = [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": None} for delta in deltas
        ]
        logprobs = self._logprobs(response, body)
        choices.append({"index": 0, "delta": {}, "logprobs": logprobs, "finish_reason": response.finish_reason})
        created = int(time.time())
        base = {"id": response.call_id, "object": "chat.completion.chunk", "created": created, "model": response.model}
        # With include_usage every chunk carries "usage": null and a last chunk without choices the totals.
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        usage = {"usage": None} if include_usage else {}
        for choice in choices:
            yield sse({**base, "choices": [choice], **usage})
        if include_usage:
            yield sse({**base, "choices": [], "usage": response.usage()})
        yield b"data: [DONE]\n\n"

    def error(self, exc: BaseException) -> dict[str, Any]:
        kind = error_kind(exc)
        message = str(exc)
        if isinstance(exc, ContextWindowExceededError):
            message = (
                f"This model's maximum context length is {exc.context_window} tokens. However, your messages "
                f"resulted in {exc.prompt_tokens} tokens. Please reduce the length of the messages."
            )
        elif kind == "server":
            message = f"{type(exc).__name__}: {exc}"
        return {
            "error": {
                "message": message,
                "type": _ERROR_TYPES.get(kind, "invalid_request_error"),
                "param": "messages" if kind == "context_length" else None,
                "code": _ERROR_CODES[kind],
            }
        }

    def error_event(self, exc: BaseException) -> bytes:
        return sse(self.error(exc))

    def _logprobs(self, response: ChatResponse, body: Mapping[str, Any]) -> dict[str, Any] | None:
        if not body.get("logprobs"):
            return None
        tokens = self.gateway.decode_tokens(response.tokens)
        return {
            "content": [
                {"token": token, "logprob": logprob, "bytes": list(token.encode("utf-8")), "top_logprobs": []}
                for token, logprob in zip(tokens, response.logprobs or [], strict=False)
            ]
        }


# --------------------------------------------------------------------------- forwarding to a hosted model
#
# A hosted gateway (platoon.gateway.hosted) forwards a Chat Completions request in this dialect to its base URL.  The
# tools a hosted model gets are the ones a trained policy is shown (RenderRequest.from_params, as chat_request reads
# them), and the request carries nothing that constrains its calls more than training does.

# Fields a hosted model never gets: none of them reaches a trained policy.  ``web_search_options`` would search the
# web at the provider, ``functions`` and ``function_call`` are legacy function calling, which training never reads,
# and every call the renderer parses is emitted, whatever ``parallel_tool_calls`` says.  ``tool_choice`` goes too
# once training's check passed it (``auto`` or ``none``): the renderer forces no call.
HOSTED_DROPPED_FIELDS = ("tool_choice", "parallel_tool_calls", "functions", "function_call", "web_search_options")
# Message fields that stream as text deltas, in the order a completion's chunks carry them.
_TEXT_FIELDS = ("reasoning_content", "reasoning", "content", "refusal")


def hosted_body(body: Mapping[str, Any], *, model: str, defaults: SamplingOverrides | None = None) -> dict[str, Any]:
    """A Chat Completions body as the hosted model gets it: ``model`` in place of the request's, the tools a trained
    policy is shown (``RenderRequest.from_params``: plain functions of a name, a description and parameters, so no
    ``strict`` schema, no other field and no other type), none of :data:`HOSTED_DROPPED_FIELDS`, and the sampling
    parameters it leaves unset taken from ``defaults`` (the rollout's ``inference_params``: ``max_completion_tokens``,
    ``temperature``, ``top_p``).  Everything else passes unchanged.

    Refuses, with training's ValueError and message, what training refuses of tools (tools that are not a list, a tool
    that names no function or is not an object, a ``tool_choice`` other than ``auto`` and ``none``) and ``n`` other
    than 1.
    """

    if body.get("n") not in (None, 1):
        raise ValueError("Only n=1 is supported")
    tools = RenderRequest.from_params(body).tools  # with training's checks of tools and tool_choice
    out = {key: value for key, value in body.items() if key not in HOSTED_DROPPED_FIELDS}
    if tools:
        out["tools"] = tools
    else:
        out.pop("tools", None)  # an empty list: no tools to training, and a provider may refuse one
    out["model"] = model
    if defaults is not None:
        budget = out.get("max_tokens") if out.get("max_completion_tokens") is None else out["max_completion_tokens"]
        sampling = {
            "max_completion_tokens": defaults.max_tokens if budget is None else None,
            "temperature": defaults.temperature,
            "top_p": defaults.top_p,
        }
        for key, value in sampling.items():
            if out.get(key) is None and value is not None:
                out[key] = value
    return out


def in_process_body(messages: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> dict[str, Any]:
    """``ctx.chat``'s call (:attr:`ChatRequest.params`) as the Chat Completions body an OpenAI client sends: the
    messages and the keyword parameters, with ``extra_body`` merged in (its fields win, as the OpenAI SDK sends them),
    unstreamed."""

    body = {key: value for key, value in params.items() if key not in ("extra_body", "stream", "stream_options")}
    extra = params.get("extra_body")
    if isinstance(extra, Mapping):
        body.update({key: value for key, value in extra.items() if key not in ("stream", "stream_options")})
    body["messages"] = [dict(message) for message in messages]
    return body


def assemble_chunks(chunks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Streamed ``chat.completion.chunk`` objects as the ``chat.completion`` they make up (the first choice).

    Text deltas (content, ``reasoning_content`` and ``reasoning``, refusal) are joined, tool calls merged by their
    index, logprobs concatenated; any other field of a delta is kept as it last came.  Raises ``ValueError`` for a
    stream that ended before its finish reason.
    """

    completion: dict[str, Any] | None = None
    message: dict[str, Any] = {"role": "assistant"}
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    logprobs: dict[str, Any] | None = None
    usage = None
    for chunk in chunks:
        if completion is None:
            completion = {key: value for key, value in chunk.items() if key not in ("choices", "usage", "object")}
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            for key, value in (choice.get("delta") or {}).items():
                if key == "tool_calls":
                    for delta in value or []:
                        _merge_tool_call(tool_calls, delta)
                elif key in _TEXT_FIELDS and isinstance(value, str):
                    message[key] = message.get(key, "") + value
                elif value is not None:
                    message[key] = value
            if choice.get("logprobs"):
                logprobs = logprobs or {}
                for key, items in choice["logprobs"].items():
                    if isinstance(items, list):
                        logprobs.setdefault(key, []).extend(items)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if completion is None or finish_reason is None:
        raise ValueError("the response stream ended before its finish reason")
    message.setdefault("content", None)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    completion["object"] = "chat.completion"
    completion["choices"] = [{"index": 0, "message": message, "logprobs": logprobs, "finish_reason": finish_reason}]
    if usage is not None:
        completion["usage"] = usage
    return completion


def _merge_tool_call(tool_calls: dict[int, dict[str, Any]], delta: Mapping[str, Any]) -> None:
    index = int(delta.get("index") or 0)
    call = tool_calls.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
    if delta.get("id"):
        call["id"] = delta["id"]
    if delta.get("type"):
        call["type"] = delta["type"]
    function = delta.get("function") or {}
    for key in ("name", "arguments"):
        if isinstance(function.get(key), str):
            call["function"][key] += function[key]


def _tool_call(tool_call: Any) -> dict[str, Any]:
    function = tool_call.get("function") if isinstance(tool_call, Mapping) else None
    function = function if isinstance(function, Mapping) else {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments if arguments is not None else {})
    return {
        "id": tool_call.get("id") if isinstance(tool_call, Mapping) else None,
        "type": "function",
        "function": {"name": function.get("name"), "arguments": arguments},
    }


def reply_fields(completion: Mapping[str, Any]) -> dict[str, Any]:
    """What a call's record takes from a ``chat.completion``: ``content`` (a refusal's text when there is no other),
    ``reasoning_content``, ``tool_calls``, ``finish_reason`` (as sent), ``truncated``, ``refusal``, ``usage``
    (normalized: ``prompt_tokens`` counts cached tokens too, which ``cached_prompt_tokens`` repeats) and
    ``usage_reported`` (false when the reply has no usage, as a stream that did not ask for it)."""

    choices = completion.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, Mapping))
    content = content if isinstance(content, str) else ""
    refusal_text = message.get("refusal") if isinstance(message.get("refusal"), str) else None
    if not content and refusal_text:
        content = refusal_text
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = message.get("reasoning") if isinstance(message.get("reasoning"), str) else None
    tool_calls = [_tool_call(tool_call) for tool_call in message.get("tool_calls") or []] or None
    return {
        "content": content,
        "reasoning_content": reasoning or None,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "refusal": bool(refusal_text) or finish_reason == "content_filter",
        "usage": _usage(completion.get("usage")),
        "usage_reported": isinstance(completion.get("usage"), Mapping),
    }


def _usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached:
        out["cached_prompt_tokens"] = int(cached)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning:
        out["reasoning_tokens"] = int(reasoning)
    return out
