"""OpenAI Responses (Codex), experimental.

Input items map onto chat messages: ``instructions`` and system/developer messages become the system prompt,
consecutive ``reasoning``, assistant ``message`` and ``function_call`` items one assistant turn, and each
``function_call_output`` a tool message.  Only ``function`` tools are rendered; other tool types are dropped.

Reasoning round-trips exactly: the output ``reasoning`` item carries the sampled reasoning in
``encrypted_content`` (``pg1:`` + base64), which Codex keeps and sends back on reasoning items, so the next
prompt re-renders the thinking the model actually produced.  A generation cut off by the token limit still ends
with ``response.completed`` (status ``incomplete``): Codex treats ``response.incomplete`` as a stream failure and
would resample.

A hosted gateway (:mod:`platoon.gateway.hosted`) forwards a Responses body as :func:`hosted_body` leaves it, with
the tool rules :meth:`OpenAIResponses.chat_request` applies (:func:`function_tools`, :func:`check_input_items`);
:func:`final_response` and :func:`reply_fields` read the provider's reply for the call's record.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from platoon.gateway.core import (
    TOOL_FUNCTION_FIELDS,
    ChatRequest,
    ChatResponse,
    SamplingOverrides,
    check_tool_choice,
    tool_function,
)
from platoon.gateway.dialects import sse
from platoon.gateway.dialects.openai_chat import OpenAIChat

REASONING_PREFIX = "pg1:"
_TEXT_PARTS = ("input_text", "output_text", "text", "summary_text", "reasoning_text")
# The input items training renders; any other belongs to a tool training never offers (a built-in or custom tool's
# call, an MCP call) or to state kept at the provider (``item_reference``), and is refused.
INPUT_ITEM_TYPES = ("message", "reasoning", "function_call", "function_call_output")


def seal_reasoning(text: str) -> str:
    return REASONING_PREFIX + base64.b64encode(text.encode("utf-8")).decode("ascii")


def _reasoning_text(item: Mapping[str, Any]) -> str:
    sealed = item.get("encrypted_content")
    if isinstance(sealed, str) and sealed.startswith(REASONING_PREFIX):
        return base64.b64decode(sealed.removeprefix(REASONING_PREFIX)).decode("utf-8")
    return _text(item.get("content")) or _text(item.get("summary"))


def _text(content: Any) -> str:
    """Text of a string or of a list of content parts; images are not supported."""
    if content is None or isinstance(content, str):
        return content or ""
    if not isinstance(content, list):
        raise ValueError(f"Unsupported content: {content!r}")
    texts: list[str] = []
    for part in content:
        part_type = part.get("type") if isinstance(part, Mapping) else None
        if part_type in _TEXT_PARTS:
            texts.append(part.get("text") or "")
        elif part_type != "refusal":
            raise ValueError(f"Unsupported content part type {part_type!r}")
    return "".join(texts)


def check_input_item(item: Any) -> str:
    """An input item's type; an item that is not an object, or whose type is not one of :data:`INPUT_ITEM_TYPES`,
    is a ValueError, for training and the hosted pass-through alike."""

    if not isinstance(item, Mapping):
        raise ValueError("Every input item must be an object")
    kind = item.get("type", "message")
    if kind not in INPUT_ITEM_TYPES:
        raise ValueError(f"Unsupported input item type {kind!r}")
    return kind


def check_input_items(items: Iterable[Any]) -> None:
    """:func:`check_input_item` of every item."""

    for item in items:
        check_input_item(item)


def input_messages(items: list[Any]) -> list[dict[str, Any]]:
    """Responses input items -> OpenAI chat messages."""
    messages: list[dict[str, Any]] = []
    turn: dict[str, Any] | None = None  # the assistant turn being assembled from consecutive output items

    def assistant_turn(*, fresh: bool = False) -> dict[str, Any]:
        nonlocal turn
        if turn is None or fresh:
            turn = {"role": "assistant", "content": ""}
            messages.append(turn)
        return turn

    for item in items:
        kind = check_input_item(item)
        if kind == "reasoning":
            assistant_turn(fresh=True)["reasoning_content"] = _reasoning_text(item)
        elif kind == "message" and item.get("role") == "assistant":
            current = assistant_turn()
            if current["content"] or "tool_calls" in current:
                current = assistant_turn(fresh=True)
            current["content"] = _text(item.get("content"))
        elif kind == "function_call":
            function = {"name": item.get("name"), "arguments": item.get("arguments") or ""}
            tool_call = {"id": item.get("call_id"), "type": "function", "function": function}
            assistant_turn().setdefault("tool_calls", []).append(tool_call)
        elif kind == "message":
            turn = None
            role = item.get("role")
            if role not in ("user", "system", "developer"):
                raise ValueError(f"Unsupported message role {role!r}")
            messages.append({"role": "user" if role == "user" else "system", "content": _text(item.get("content"))})
        else:  # function_call_output
            turn = None
            messages.append({"role": "tool", "tool_call_id": item.get("call_id"), "content": _text(item.get("output"))})
    return messages


def function_tools(tools: Any) -> list[dict[str, Any]]:
    """The tools a policy is shown of a Responses request's ``tools``, as OpenAI function tools: the ``function``
    entries, each a function of its name, description and parameters (:func:`~platoon.gateway.core.tool_function`,
    which gives one without a schema the empty-object schema), in that order whatever order the entry writes them in,
    as training has always shown them (the OpenAI Agents SDK writes ``parameters`` before ``description``).  Every
    other type (built-in tools such as web search, ``mcp``, ``custom``, ``tool_search``) is dropped, so nothing runs
    at the provider, and so is a function entry without a name, which no model can call."""

    functions: list[dict[str, Any]] = []
    for tool in tools or []:
        if not (isinstance(tool, Mapping) and tool.get("type") == "function"):
            continue
        function = tool_function({key: tool[key] for key in TOOL_FUNCTION_FIELDS if key in tool})
        if function is not None:
            functions.append({"type": "function", "function": function})
    return functions


class OpenAIResponses:
    """The Responses dialect; errors use the OpenAI error object, as in Chat Completions."""

    name = "openai_responses"

    def __init__(self, chat: OpenAIChat) -> None:
        self._chat = chat

    def chat_request(self, body: Mapping[str, Any]) -> ChatRequest:
        messages: list[dict[str, Any]] = []
        if body.get("instructions"):
            messages.append({"role": "system", "content": _text(body["instructions"])})
        items = body.get("input")
        if isinstance(items, str):
            messages.append({"role": "user", "content": items})
        elif isinstance(items, list):
            messages.extend(input_messages(items))
        else:
            raise ValueError("'input' must be a string or an array of items")
        params: dict[str, Any] = {
            "tools": function_tools(body.get("tools")),
            "tool_choice": body.get("tool_choice"),
            "max_tokens": body.get("max_output_tokens"),
            "temperature": body.get("temperature"),
            "top_p": body.get("top_p"),
        }
        if (body.get("reasoning") or {}).get("effort") == "none":
            params["reasoning_effort"] = "none"
        return ChatRequest.from_params(messages, params)

    def response(self, response: ChatResponse, body: Mapping[str, Any]) -> dict[str, Any]:
        truncated = response.finish_reason == "length"
        return {
            "id": response.call_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "incomplete" if truncated else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "error": None,
            "model": response.model,
            "output": self._output(response),
            "parallel_tool_calls": True,
            "tool_choice": body.get("tool_choice", "auto"),
            "tools": body.get("tools") or [],
            "usage": {
                "input_tokens": response.prompt_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": response.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": response.prompt_tokens + response.completion_tokens,
            },
        }

    def events(self, response: ChatResponse, body: Mapping[str, Any]) -> Iterator[bytes]:
        final = self.response(response, body)
        sequence = 0

        def event(kind: str, **fields: Any) -> bytes:
            nonlocal sequence
            sequence += 1
            return sse({"type": kind, "sequence_number": sequence - 1, **fields}, event=kind)

        yield event("response.created", response={**final, "status": "in_progress", "output": []})
        for index, item in enumerate(final["output"]):
            yield event("response.output_item.added", output_index=index, item=_pending(item))
            if item["type"] == "message":
                text = item["content"][0]["text"]
                where = {"item_id": item["id"], "output_index": index, "content_index": 0}
                yield event("response.output_text.delta", **where, delta=text)
                yield event("response.output_text.done", **where, text=text)
            elif item["type"] == "function_call":
                where = {"item_id": item["id"], "output_index": index}
                yield event("response.function_call_arguments.delta", **where, delta=item["arguments"])
                yield event("response.function_call_arguments.done", **where, arguments=item["arguments"])
            yield event("response.output_item.done", output_index=index, item=item)
        yield event("response.completed", response=final)

    def error(self, exc: BaseException) -> dict[str, Any]:
        return self._chat.error(exc)

    def error_event(self, exc: BaseException) -> bytes:
        """``response.failed``: Codex ignores bare ``error`` events but fails (and may retry) on this one."""
        return failed_event(self.error(exc)["error"])

    @staticmethod
    def _output(response: ChatResponse) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        if response.reasoning_content is not None:
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{response.call_id}",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": response.reasoning_content}],
                    "encrypted_content": seal_reasoning(response.reasoning_content),
                }
            )
        if response.content or not response.tool_calls:
            text = {"type": "output_text", "text": response.content, "annotations": []}
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{response.call_id}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [text],
                }
            )
        for tool_call in response.tool_calls or []:
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{tool_call['id']}",
                    "call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                    "status": "completed",
                }
            )
        return output


def failed_event(error: Mapping[str, Any]) -> bytes:
    """An OpenAI error object (``{"message", "type", "code"}``) as a ``response.failed`` event."""

    failed = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "status": "failed",
        "error": {"code": error.get("code") or error.get("type"), "message": error.get("message")},
    }
    return sse({"type": "response.failed", "sequence_number": 0, "response": failed}, event="response.failed")


def _pending(item: dict[str, Any]) -> dict[str, Any]:
    """An output item as ``response.output_item.added`` announces it, before its text or arguments."""
    if item["type"] == "message":
        return {**item, "status": "in_progress", "content": []}
    if item["type"] == "function_call":
        return {**item, "status": "in_progress", "arguments": ""}
    return item


# --------------------------------------------------------------------------- forwarding to a hosted model
#
# A hosted gateway (platoon.gateway.hosted) forwards a Responses request in this dialect to its base URL.  The hosted
# model gets the function tools OpenAIResponses.chat_request shows a trained policy, and the request carries nothing
# that constrains its calls more than training does, nor anything that keeps or edits a conversation at the provider.

# Fields a hosted model never gets: none of them reaches a trained policy.  ``previous_response_id``,
# ``conversation``, ``prompt`` (a stored prompt), ``background`` and ``truncation`` keep or edit a conversation at the
# provider, while the model must see exactly the input the harness sent.  Every call the renderer parses is
# emitted, whatever ``parallel_tool_calls`` and ``max_tool_calls`` say, and ``tool_choice`` goes once training's check
# passed it (``auto`` or ``none``).
HOSTED_DROPPED_FIELDS = (
    "tool_choice",
    "parallel_tool_calls",
    "max_tool_calls",
    "previous_response_id",
    "conversation",
    "prompt",
    "background",
    "truncation",
)
# Stream events that end a response with its whole object.
FINAL_EVENTS = ("response.completed", "response.incomplete")


def _hosted_tool(function: Mapping[str, Any]) -> dict[str, Any]:
    """A function a policy is shown (:func:`function_tools`) as a Responses function tool."""

    # strict: training constrains no call to its schema, and the API requires the field.
    return {"type": "function", **function, "strict": False}


def hosted_body(body: Mapping[str, Any], *, model: str, defaults: SamplingOverrides | None = None) -> dict[str, Any]:
    """A Responses body as the hosted model gets it: ``model`` in place of the request's, the function tools
    :func:`function_tools` shows a trained policy (with ``strict: false``), none of :data:`HOSTED_DROPPED_FIELDS`, and
    the sampling parameters it leaves unset taken from ``defaults`` (the rollout's ``inference_params``:
    ``max_output_tokens``, ``temperature``, ``top_p``).  Everything else passes unchanged: instructions, input with its
    images and encrypted reasoning, reasoning settings, ``include``, ``store`` and text formats.

    Refuses, with training's ValueError and message, input items training never renders (:func:`check_input_items`)
    and a ``tool_choice`` other than ``auto`` and ``none``.
    """

    items = body.get("input")
    if isinstance(items, list):
        check_input_items(items)
    functions = [tool["function"] for tool in function_tools(body.get("tools"))]
    check_tool_choice(body.get("tool_choice"))
    out = {key: value for key, value in body.items() if key not in HOSTED_DROPPED_FIELDS}
    tools = [_hosted_tool(function) for function in functions]
    if tools:
        out["tools"] = tools
    else:
        out.pop("tools", None)
    out["model"] = model
    if defaults is not None:
        sampling = {
            "max_output_tokens": defaults.max_tokens,
            "temperature": defaults.temperature,
            "top_p": defaults.top_p,
        }
        for key, value in sampling.items():
            if out.get(key) is None and value is not None:
                out[key] = value
    return out


def final_response(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The response object a Responses stream ends with (its ``response.completed`` or ``response.incomplete``
    event).  Raises ``ValueError`` for a stream that ended before it."""

    final = None
    for event in events:
        if event.get("type") in FINAL_EVENTS and isinstance(event.get("response"), Mapping):
            final = dict(event["response"])
    if final is None:
        raise ValueError("the response stream ended before response.completed")
    return final


def _part_texts(parts: Any, kinds: tuple[str, ...]) -> list[str]:
    parts = parts if isinstance(parts, list) else []
    return [str(part.get("text") or "") for part in parts if isinstance(part, Mapping) and part.get("type") in kinds]


def reply_fields(response: Mapping[str, Any]) -> dict[str, Any]:
    """What a call's record takes from a Responses response: ``content`` (its output text, or a refusal's when there
    is no other), ``reasoning_content`` (its reasoning text or summary, when the provider returns any),
    ``tool_calls`` (its ``function_call`` items as OpenAI tool calls), ``finish_reason`` (``completed``, or why it is
    incomplete), ``truncated`` (cut off at ``max_output_tokens``), ``refusal``, ``usage`` (normalized:
    ``prompt_tokens`` counts cached tokens too, which ``cached_prompt_tokens`` repeats) and ``usage_reported``."""

    texts: list[str] = []
    refusals: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        kind = item.get("type") if isinstance(item, Mapping) else None
        if kind == "message":
            texts += _part_texts(item.get("content"), ("output_text", "text"))
            refusals += [
                str(part.get("refusal") or "")
                for part in item.get("content") or []
                if isinstance(part, Mapping) and part.get("type") == "refusal"
            ]
        elif kind == "reasoning":
            reasoning += _part_texts(item.get("content"), ("reasoning_text", "text")) or _part_texts(
                item.get("summary"), ("summary_text", "text")
            )
        elif kind == "function_call":
            function = {"name": item.get("name"), "arguments": item.get("arguments") or ""}
            tool_calls.append({"id": item.get("call_id"), "type": "function", "function": function})
    incomplete = (response.get("incomplete_details") or {}).get("reason")
    status = response.get("status")
    finish_reason = incomplete if status == "incomplete" and incomplete else status
    usage = response.get("usage")
    counts = usage if isinstance(usage, Mapping) else {}
    prompt = int(counts.get("input_tokens") or 0)
    completion = int(counts.get("output_tokens") or 0)
    normalized = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    cached = (counts.get("input_tokens_details") or {}).get("cached_tokens")
    if cached:
        normalized["cached_prompt_tokens"] = int(cached)
    thought = (counts.get("output_tokens_details") or {}).get("reasoning_tokens")
    if thought:
        normalized["reasoning_tokens"] = int(thought)
    content = "".join(texts) or "".join(refusals)
    return {
        "content": content,
        "reasoning_content": "".join(reasoning) or None,
        "tool_calls": tool_calls or None,
        "finish_reason": finish_reason,
        "truncated": status == "incomplete" and incomplete == "max_output_tokens",
        "refusal": bool(refusals),
        "usage": normalized,
        "usage_reported": isinstance(usage, Mapping),
    }
