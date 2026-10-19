"""A fake hosted model API for the hosted-gateway tests: one aiohttp server, on a loopback port in a thread of its own,
that speaks OpenAI Chat Completions, OpenAI Responses and Anthropic Messages (and ``count_tokens``) under ``/v1``, so
a test (or a whole ``platoon eval`` run in the test's process, worker processes included) reaches it over real HTTP.

It records every request (path, query string, headers with lower-cased names, body), answers each dialect with a
scripted reply (a function of the request body), records any other request (another route, or another method, such as
the GET a followed 301 makes) in ``others`` and answers it 404, and can be told to misbehave first:

* ``faults``: answers it gives, one per request, before it answers normally (``Fault``: a status, a JSON body in
  which ``{key}`` becomes the key the request presented, headers, a delay, a path suffix it applies to);
* ``cut_streams``: that many streams are cut off midway (the connection drops before the last event);
* ``error_streams``: that many streams carry an error event midway, in their dialect's form;
* ``delay``: seconds before every answer.

``max_in_flight`` is the most requests it was answering at once.  The ``bisect_*`` players play number search
(``platoon.envs.number_search``): they read the range from the task's prompt and the env's "Too low" / "Too high"
replies, and call ``guess`` with the middle, in each dialect's own form.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

MODEL = "fake-model"
SERVED_MODEL = "fake-model-2026-01-01"
FAKE_KEY = "sk-fake-0123456789abcdef-SECRET"
REQUEST_ID = "req_fake"
CHAT, RESPONSES, MESSAGES, COUNT_TOKENS = "/chat/completions", "/responses", "/messages", "/messages/count_tokens"

Reply = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Fault:
    """One failing answer: ``status`` with a JSON ``body`` (``{key}`` in a string becomes the presented key) and
    ``headers``, after ``delay`` seconds.  With ``path``, only a request whose path ends with it gets it; with
    ``when``, only one whose body it accepts (say a harness's request, not the probe's)."""

    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    path: str | None = None
    when: Callable[[dict[str, Any]], bool] | None = None

    def applies(self, path: str, body: dict[str, Any]) -> bool:
        return (self.path is None or path.endswith(self.path)) and (self.when is None or self.when(body))


def with_tools(body: dict[str, Any]) -> bool:
    """A request that offers tools: a harness's, never the probe's."""

    return bool(body.get("tools"))


def openai_error(message: str, *, kind: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def anthropic_error(message: str, *, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


# --------------------------------------------------------------------------- default replies


def chat_text(body: dict[str, Any]) -> dict[str, Any]:
    return {"content": "OK"}


def messages_text(body: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": "OK"}]}


def responses_text(body: dict[str, Any]) -> dict[str, Any]:
    return {"output": [output_message("OK")]}


def output_message(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def tool_call(name: str, arguments: Mapping[str, Any], *, call_id: str | None = None) -> dict[str, Any]:
    return {
        "id": call_id or f"call_{uuid.uuid4().hex[:10]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(dict(arguments))},
    }


def tool_use(name: str, arguments: Mapping[str, Any], *, call_id: str | None = None) -> dict[str, Any]:
    return {
        "type": "tool_use",
        "id": call_id or f"toolu_{uuid.uuid4().hex[:16]}",
        "name": name,
        "input": dict(arguments),
    }


def function_call(name: str, arguments: Mapping[str, Any], *, call_id: str | None = None) -> dict[str, Any]:
    call_id = call_id or f"call_{uuid.uuid4().hex[:10]}"
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(dict(arguments)),
        "status": "completed",
    }


# --------------------------------------------------------------------------- the server


class FakeProvider:
    """The fake API (see the module docstring); ``base_url`` is ``http://127.0.0.1:<port>/v1`` once started."""

    def __init__(
        self,
        *,
        chat: Reply = chat_text,
        messages: Reply = messages_text,
        responses: Reply = responses_text,
        api_key: str | None = FAKE_KEY,
        models: tuple[str, ...] = (MODEL,),
    ) -> None:
        self.replies = {CHAT: chat, MESSAGES: messages, RESPONSES: responses}
        self.api_key = api_key
        self.models = models
        self.requests: list[dict[str, Any]] = []
        self.others: list[dict[str, Any]] = []  # requests to routes it does not serve: method, path, headers
        self.faults: list[Fault] = []
        self.cut_streams = 0
        self.error_streams = 0
        self.delay = 0.0
        self.in_flight = 0
        self.max_in_flight = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None
        self.origin = ""

    @property
    def base_url(self) -> str:
        return f"{self.origin}/v1"

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> FakeProvider:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> str:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def serve() -> None:
            asyncio.set_event_loop(loop)
            runner = web.AppRunner(self._app(), access_log=None)
            loop.run_until_complete(runner.setup())
            loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", 0).start())
            self._runner = runner
            self.origin = f"http://127.0.0.1:{runner.addresses[0][1]}"
            ready.set()
            loop.run_forever()

        self._loop = loop
        self._thread = threading.Thread(target=serve, daemon=True)
        self._thread.start()
        if not ready.wait(15):
            raise RuntimeError("the fake provider did not start")
        return self.base_url

    def stop(self) -> None:
        loop, runner = self._loop, self._runner
        if loop is None:
            return
        if runner is not None:
            asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(15)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(15)
        loop.close()
        self._loop = None

    def _app(self) -> web.Application:
        app = web.Application(client_max_size=64 * 1024**2)
        for path in (CHAT, RESPONSES, MESSAGES, COUNT_TOKENS):
            app.router.add_post(f"/v1{path}", self._handle)
        app.router.add_route("*", "/{tail:.*}", self._other)
        return app

    async def _other(self, request: web.Request) -> web.Response:
        headers = {name.lower(): value for name, value in request.headers.items()}
        self.others.append({"method": request.method, "path": request.path, "headers": headers})
        return web.json_response(openai_error("Not Found"), status=404)

    # ------------------------------------------------------------------ helpers for tests

    def calls(self, path: str) -> list[dict[str, Any]]:
        """The recorded requests to ``path`` (a suffix such as :data:`CHAT`)."""
        return [request for request in self.requests if request["path"].endswith(path)]

    # ------------------------------------------------------------------ requests

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        path = request.path.removeprefix("/v1")
        raw = await request.read()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {"_unparsed": raw.decode("utf-8", errors="replace")}
        headers = {name.lower(): value for name, value in request.headers.items()}
        self.requests.append({"path": request.path, "query": request.query_string, "headers": headers, "body": body})
        messages_api = path in (MESSAGES, COUNT_TOKENS)
        presented = headers.get("x-api-key") if messages_api else _bearer(headers)
        if self.api_key is not None and presented != self.api_key:
            error = (
                anthropic_error("invalid x-api-key", kind="authentication_error")
                if messages_api
                else openai_error("Incorrect API key provided.", code="invalid_api_key")
            )
            return web.json_response(error, status=401)
        fault = next((fault for fault in self.faults if fault.applies(path, body)), None)
        if fault is not None:
            self.faults.remove(fault)
            if fault.delay:
                await asyncio.sleep(fault.delay)
            return web.json_response(_fill_key(fault.body, presented or ""), status=fault.status, headers=fault.headers)
        if body.get("model") not in self.models:
            missing = f"The model `{body.get('model')}` does not exist"
            error = anthropic_error(missing, kind="not_found_error") if messages_api else openai_error(missing)
            return web.json_response(error, status=404)
        if path == COUNT_TOKENS:
            return web.json_response({"input_tokens": 1000 + len(json.dumps(body.get("messages")))})
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            reply = self.replies[path](body)
            if path == CHAT:
                answer, events = _chat(body, reply)
            elif path == MESSAGES:
                answer, events = _messages(body, reply)
            else:
                answer, events = _responses(body, reply)
            request_headers = {"request-id" if messages_api else "x-request-id": REQUEST_ID}
            if not body.get("stream"):
                return web.json_response(answer, headers=request_headers)
            return await self._stream(request, path, events, request_headers)
        finally:
            self.in_flight -= 1

    async def _stream(
        self, request: web.Request, path: str, events: list[bytes], headers: dict[str, str]
    ) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", **headers})
        await response.prepare(request)
        if self.cut_streams > 0:
            self.cut_streams -= 1
            for event in events[: len(events) // 2]:
                await response.write(event)
            if request.transport is not None:
                request.transport.close()  # the connection drops before the last event
            return response
        if self.error_streams > 0:
            self.error_streams -= 1
            events = [*events[: len(events) // 2], _error_event(path)]
        for event in events:
            await response.write(event)
        await response.write_eof()
        return response


def _bearer(headers: Mapping[str, str]) -> str | None:
    scheme, _, token = headers.get("authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" else None


def _fill_key(value: Any, key: str) -> Any:
    if isinstance(value, str):
        return value.replace("{key}", key)
    if isinstance(value, Mapping):
        return {name: _fill_key(item, key) for name, item in value.items()}
    if isinstance(value, list):
        return [_fill_key(item, key) for item in value]
    return value


def _sse(data: Any, event: str | None = None) -> bytes:
    name = f"event: {event}\n" if event else ""
    return f"{name}data: {json.dumps(data)}\n\n".encode()


def _error_event(path: str) -> bytes:
    if path == MESSAGES:
        return _sse(anthropic_error("Overloaded", kind="overloaded_error"), event="error")
    if path == RESPONSES:
        failed = {"id": "resp_x", "status": "failed", "error": {"code": "server_error", "message": "upstream failed"}}
        return _sse({"type": "response.failed", "sequence_number": 1, "response": failed}, event="response.failed")
    return _sse(openai_error("upstream failed", kind="server_error"))


def _halves(text: str) -> list[str]:
    return [text[: len(text) // 2], text[len(text) // 2 :]]


# --------------------------------------------------------------------------- Chat Completions


def _chat(body: dict[str, Any], reply: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
    answer = dict(reply)
    finish = answer.pop("finish_reason", None) or ("tool_calls" if answer.get("tool_calls") else "stop")
    size = len(json.dumps(body.get("messages"))) // 10
    usage = answer.pop("usage", None) or {
        "prompt_tokens": 100 + size,
        "completion_tokens": 7,
        "total_tokens": 107 + size,
        "prompt_tokens_details": {"cached_tokens": 64},
    }
    model = answer.pop("model", None) or SERVED_MODEL
    message: dict[str, Any] = {"role": "assistant", "content": answer.pop("content", None), **answer}
    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
        "usage": usage,
    }
    base = {"id": completion["id"], "object": "chat.completion.chunk", "created": 1, "model": model}
    deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
    for key in ("reasoning_content", "content", "refusal"):
        text = message.get(key)
        if isinstance(text, str) and text:
            deltas += [{key: half} for half in _halves(text)]
    for index, call in enumerate(message.get("tool_calls") or []):
        first, second = _halves(call["function"]["arguments"])
        opened = {"index": index, "id": call["id"], "type": "function"}
        deltas.append({"tool_calls": [{**opened, "function": {"name": call["function"]["name"], "arguments": first}}]})
        deltas.append({"tool_calls": [{"index": index, "function": {"arguments": second}}]})
    chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]} for delta in deltas]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if (body.get("stream_options") or {}).get("include_usage"):
        chunks.append({**base, "choices": [], "usage": usage})
    return completion, [*(_sse(chunk) for chunk in chunks), b"data: [DONE]\n\n"]


# --------------------------------------------------------------------------- Messages


def _messages(body: dict[str, Any], reply: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
    answer = dict(reply)
    content = answer.pop("content")
    stop = answer.pop("stop_reason", None) or (
        "tool_use" if any(block.get("type") == "tool_use" for block in content) else "end_turn"
    )
    usage = answer.pop("usage", None) or {
        "input_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 10,
        "output_tokens": 9,
    }
    message = {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": answer.pop("model", None) or SERVED_MODEL,
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": usage,
    }
    start = {**message, "content": [], "stop_reason": None, "usage": {**usage, "output_tokens": 1}}
    events = [_sse({"type": "message_start", "message": start}, event="message_start")]
    for index, block in enumerate(content):
        kind = block["type"]
        if kind == "text":
            opened, deltas = (
                {"type": "text", "text": ""},
                [{"type": "text_delta", "text": half} for half in _halves(block["text"])],
            )
        elif kind == "thinking":
            opened = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif kind == "tool_use":
            opened = {**block, "input": {}}
            deltas = [
                {"type": "input_json_delta", "partial_json": half} for half in _halves(json.dumps(block["input"]))
            ]
        else:
            opened, deltas = block, []
        events.append(
            _sse({"type": "content_block_start", "index": index, "content_block": opened}, "content_block_start")
        )
        for delta in deltas:
            events.append(_sse({"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta"))
        events.append(_sse({"type": "content_block_stop", "index": index}, "content_block_stop"))
    delta = {"stop_reason": stop, "stop_sequence": None}
    events.append(
        _sse(
            {"type": "message_delta", "delta": delta, "usage": {"output_tokens": usage["output_tokens"]}},
            "message_delta",
        )
    )
    events.append(_sse({"type": "message_stop"}, "message_stop"))
    return message, events


# --------------------------------------------------------------------------- Responses


def _responses(body: dict[str, Any], reply: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
    answer = dict(reply)
    output = answer.pop("output")
    status = answer.pop("status", None) or "completed"
    usage = answer.pop("usage", None) or {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 32},
        "output_tokens": 11,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 131,
    }
    response = {
        "id": f"resp_{uuid.uuid4().hex[:12]}",
        "object": "response",
        "created_at": 1,
        "status": status,
        "incomplete_details": answer.pop("incomplete_details", None),
        "error": None,
        "model": answer.pop("model", None) or SERVED_MODEL,
        "output": output,
        "parallel_tool_calls": True,
        "usage": usage,
    }
    sequence = iter(range(1000))

    def event(kind: str, **fields: Any) -> bytes:
        return _sse({"type": kind, "sequence_number": next(sequence), **fields}, event=kind)

    events = [event("response.created", response={**response, "status": "in_progress", "output": []})]
    for index, item in enumerate(output):
        events.append(event("response.output_item.added", output_index=index, item=item))
        if item["type"] == "message":
            text = item["content"][0].get("text", "")
            for half in _halves(text):
                events.append(event("response.output_text.delta", item_id=item["id"], output_index=index, delta=half))
        elif item["type"] == "function_call":
            for half in _halves(item["arguments"]):
                events.append(
                    event("response.function_call_arguments.delta", item_id=item["id"], output_index=index, delta=half)
                )
        events.append(event("response.output_item.done", output_index=index, item=item))
    final = "response.incomplete" if status == "incomplete" else "response.completed"
    events.append(event(final, response=response))
    return response, events


# --------------------------------------------------------------------------- number search players


def _bisect(goal: str, guesses: list[int], replies: list[str]) -> int:
    found = re.search(r"between (\d+) and (\d+)", goal)
    low, high = (int(found[1]), int(found[2])) if found else (1, 100)
    for guess, reply in zip(guesses, replies, strict=False):
        if "Too low" in reply:
            low = guess + 1
        elif "Too high" in reply:
            high = guess - 1
    return (low + high) // 2


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for block in content or []:
        if isinstance(block, Mapping):
            if isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif "content" in block:
                texts.append(_text(block["content"]))
    return "".join(texts)


def bisect_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Number search on Chat Completions: the next guess, as a ``guess`` tool call."""

    messages = body.get("messages") or []
    goal = next((_text(m.get("content")) for m in messages if "between" in _text(m.get("content"))), "")
    guesses, replies = [], []
    for message in messages:
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] == "guess":
                guesses.append(int(json.loads(call["function"]["arguments"])["number"]))
        if message.get("role") == "tool":
            replies.append(_text(message.get("content")))
    return {"content": "", "tool_calls": [tool_call("guess", {"number": _bisect(goal, guesses, replies)})]}


def bisect_messages(body: dict[str, Any], *, thinking: bool = False) -> dict[str, Any]:
    """Number search on Messages: the next guess, as a ``guess`` tool use (after a signed thinking block when
    ``thinking``)."""

    messages = body.get("messages") or []
    goal = _text(body.get("system")) + "".join(
        _text(m.get("content")) for m in messages if m.get("role") == "user" and "between" in _text(m.get("content"))
    )
    guesses, replies = [], []
    for message in messages:
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "tool_use" and block.get("name") == "guess":
                guesses.append(int(block["input"]["number"]))
            elif block.get("type") == "tool_result":
                replies.append(_text(block.get("content")))
    content = [tool_use("guess", {"number": _bisect(goal, guesses, replies)})]
    if thinking:
        content.insert(0, {"type": "thinking", "thinking": "bisect the range", "signature": f"sig-{uuid.uuid4().hex}"})
    return {"content": content}


def bisect_responses(body: dict[str, Any]) -> dict[str, Any]:
    """Number search on Responses: the next guess, as a ``guess`` function call."""

    items = body.get("input")
    items = [{"type": "message", "role": "user", "content": items}] if isinstance(items, str) else items or []
    goal = _text(body.get("instructions")) + "".join(
        _text(item.get("content")) for item in items if item.get("type", "message") == "message"
    )
    guesses, replies = [], []
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == "guess":
            guesses.append(int(json.loads(item["arguments"])["number"]))
        elif item.get("type") == "function_call_output":
            replies.append(_text(item.get("output")))
    return {"output": [function_call("guess", {"number": _bisect(goal, guesses, replies)})]}
