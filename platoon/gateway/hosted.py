"""A model gateway whose policy is a hosted model, reached at one base URL in each harness's own dialect.

:class:`HostedGateway` is a :class:`~platoon.gateway.core.ModelGateway` whose calls go to ``inference.hosted.base_url``
instead of a Tinker sampler.  Agents, their keys and tree, call kinds, step budgets, the judges switch, recording
and shutdown stay the gateway's own, so harnesses, episode programs, trajectories and evaluation reports work as they
do with Tinker.  Nothing is translated between dialects:

* **Over HTTP** (:meth:`HostedGateway.forward`), a harness's request in its dialect (Chat Completions, Responses or
  Messages, and Messages ``count_tokens``) goes to the same path under the base URL with the provider's key in place
  of the agent's.  The body changes only as the dialect's ``hosted_body`` says: the model, the tools normalized as
  training normalizes them (a request training refuses is refused with training's error), the sampling parameters
  it leaves unset from the rollout's defaults, and a few fields no trained policy ever gets.  The provider's reply
  goes back as it came: its status, content type and body, stream included, with the key redacted.
* **In process** (:meth:`HostedGateway.chat`), ``ctx.chat``'s call is a Chat Completions body
  (:func:`~platoon.gateway.dialects.openai_chat.in_process_body`), sent unstreamed and normalized the same way.

Every attempt has ``timeout_s``; 408, 409, 429 and 5xx replies, connection errors, timeouts, streams cut off before
their last event and error events in a stream are retried with backoff (or as ``retry-after`` asks), all within one
budget of ``max_retries + 1`` attempts.  A stream is read whole before it is relayed, so a broken one can be sent
again and the call recorded from the whole reply.  A call is recorded on its agent like a Tinker call, with a
:class:`~platoon.gateway.core.ProviderRecord` (usage, served model, ids, attempts, cost) instead of token ids, so it
never trains: hosted models serve evaluation and ``platoon rollout`` only.

A call that fails after its retries is counted in its agent's misc (:data:`BACKEND_FAILURES_MISC_KEY`), and an
evaluation errors its rollout; a refused key, payment, permission or model stops the evaluation, and so does a
redirect, which the gateway never follows, so the key and the request go to the base URL only.  The key is read
from the variable ``api_key_env`` names in each process that sends requests, never stored, logged or given to a
harness, and redacted from every reply.  The HTTP client reads nothing else from the environment but the proxy
variables (``HTTPS_PROXY``, ``HTTP_PROXY``, ``NO_PROXY``).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import email.utils
import json
import logging
import os
import random
import re
import time
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import aiohttp

from platoon.gateway.core import (
    AgentSession,
    CallTiming,
    ChatRequest,
    ChatResponse,
    HostedCallError,
    HostedRefusedError,
    HostedRequestError,
    ModelGateway,
    PreparedCall,
    ProviderContextWindowError,
    ProviderRecord,
    RenderRequest,
    check_call_kind,
    generate_id,
)
from platoon.gateway.dialects import anthropic, openai_chat, responses

logger = logging.getLogger(__name__)

HostedDialect = Literal["openai_chat", "openai_responses", "anthropic_messages"]
DIALECTS: tuple[HostedDialect, ...] = ("openai_chat", "openai_responses", "anthropic_messages")
# Where a call in each dialect goes under the base URL: the path it has under the gateway's own /v1.
PATHS: dict[str, str] = {
    "openai_chat": "/chat/completions",
    "openai_responses": "/responses",
    "anthropic_messages": "/messages",
}
COUNT_TOKENS_PATH = "/messages/count_tokens"
# A request's completion budget in each dialect: what a policy call drops under sampling_precedence: config when the
# rollout sets max_completion_tokens, which the dialect then adds under its own name.
_COMPLETION_BUDGETS: dict[str, tuple[str, ...]] = {
    "openai_chat": ("max_completion_tokens", "max_tokens"),
    "openai_responses": ("max_output_tokens",),
    "anthropic_messages": ("max_tokens",),
}
# The client headers forwarded with a request in each dialect, which are part of its request (the Messages API
# refuses a request without its version).  No other client header is forwarded.
FORWARDED_HEADERS: dict[str, tuple[str, ...]] = {
    "openai_chat": ("openai-beta",),
    "openai_responses": ("openai-beta",),
    "anthropic_messages": ("anthropic-version", "anthropic-beta"),
}
# The reply headers relayed to the harness with a reply that failed, besides its content type.
RELAYED_HEADERS = ("retry-after", "retry-after-ms")
# Replies that refuse the run itself (the key, the payment, the permission, the model or the route): never retried.
# A redirect (3xx) refuses it too.  The gateway follows none: following one would send the request, and with it the
# key, wherever the base URL points (aiohttp drops an Authorization header on the way to another origin, but keeps
# Messages' x-api-key).
REFUSED_STATUSES = frozenset({401, 402, 403, 404})
# Backoff before attempt n + 1 of a call: base * 2 ** (n - 1) seconds, at most cap, plus up to a quarter of jitter.
RETRY_BACKOFF_S = (0.5, 8.0)
# The longest wait a reply's retry-after (or retry-after-ms) gets.
MAX_RETRY_AFTER_S = 60.0
# The probe's retries and its timeout per attempt (at most timeout_s): enough for a hiccup, few enough that a wrong
# URL fails fast.
PROBE_RETRIES = 2
PROBE_TIMEOUT_S = 120.0
# The Messages API version the probe sends (harnesses send their own).
ANTHROPIC_VERSION = "2023-06-01"
# An agent's misc record of its calls that failed: ``count`` (failed after their retries, or refused), ``kinds``
# (``rate_limit``, ``timeout``, ``server``, ``refused``), ``fatal`` (the first refusal of the run's key or model) and
# ``rejected`` (requests the provider refused as invalid, which are the harness's).
BACKEND_FAILURES_MISC_KEY = "backend_failures"
# What a reply, record or log shows in place of the key.
REDACTED = "[redacted]"
# Keys shorter than this are not looked for in replies, where they would match ordinary text.
MIN_SECRET_LENGTH = 8


class HostedSettings(Protocol):
    """What a hosted gateway reads of ``inference.hosted`` (``platoon.eval.config_defs.HostedConfig``)."""

    base_url: str
    api_key_env: str | None
    timeout_s: float
    max_retries: int
    max_concurrent_requests: int | None
    pricing: Any


# --------------------------------------------------------------------------- helpers


def url_origin(url: str | None) -> str | None:
    """The scheme, host and port of ``url``: what ``lock.json`` and the report record of a base URL, never its
    path, query or user info."""

    if not url:
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{host}{port}" if parts.scheme and host else None


def call_cost(usage: Mapping[str, Any], pricing: Any) -> float | None:
    """A call's cost in USD from ``pricing`` (USD per million tokens: ``input``, ``output``, and ``cached_input`` and
    ``cache_write``, which default to ``input``), or None without prices."""

    if pricing is None:
        return None

    def price(name: str, default: float) -> float:
        value = pricing.get(name) if isinstance(pricing, Mapping) else getattr(pricing, name, None)
        return default if value is None else float(value)

    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int(usage.get("cached_prompt_tokens") or 0)
    written = int(usage.get("cache_write_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    input_price = price("input", 0.0)
    total = (
        max(prompt - cached - written, 0) * input_price
        + cached * price("cached_input", input_price)
        + written * price("cache_write", input_price)
        + output * price("output", 0.0)
    )
    return total / 1_000_000


def secret_forms(secret: str) -> list[str]:
    """The forms a reply may echo ``secret`` in, longest first: as it is, JSON-escaped (with ``/`` escaped too, as
    some encoders write it) and percent-encoded (either case of hex digits).  They differ only for a key with
    characters such as ``/``, ``+`` or ``=``."""

    escaped = json.dumps(secret)[1:-1]
    encoded = quote(secret, safe="")
    forms = {secret, escaped, escaped.replace("/", "\\/"), encoded, re.sub(r"%[0-9A-F]{2}", _lower, encoded)}
    return sorted(forms, key=len, reverse=True)


def _lower(match: re.Match[str]) -> str:
    return match[0].lower()


def redact(text: str, secrets: Sequence[str]) -> str:
    """``text`` with each of ``secrets``, in each of its :func:`secret_forms`, replaced by :data:`REDACTED`."""

    for secret in secrets:
        for form in secret_forms(secret):
            text = text.replace(form, REDACTED)
    return text


def redact_bytes(data: bytes, secrets: Sequence[str]) -> bytes:
    """:func:`redact` of a reply's bytes."""

    for secret in secrets:
        for form in secret_forms(secret):
            data = data.replace(form.encode(), REDACTED.encode())
    return data


def env_proxy(url: str) -> tuple[str | None, dict[str, str] | None]:
    """The proxy for ``url`` from the standard variables (``HTTPS_PROXY`` or ``HTTP_PROXY``, then ``ALL_PROXY``,
    either case, unless ``NO_PROXY`` covers the host), without its credentials, and the ``Proxy-Authorization``
    header they make; ``(None, None)`` for none.  Nothing else is read from the environment: aiohttp's own
    ``trust_env`` would also send ``.netrc`` credentials to the base URL's host."""

    proxies = urllib.request.getproxies_environment()
    parts = urlsplit(url)
    proxy = proxies.get(parts.scheme) or proxies.get("all")
    if not proxy or _no_proxy(parts.hostname or "", proxies.get("no", "")):
        return None, None
    proxy_parts = urlsplit(proxy if "://" in proxy else f"http://{proxy}")
    if proxy_parts.scheme != "http":
        logger.warning("Ignoring the %s proxy for hosted calls: only http proxies are supported", proxy_parts.scheme)
        return None, None
    if proxy_parts.username is None:
        return urlunsplit(proxy_parts), None
    host = proxy_parts.hostname or ""
    netloc = f"{host}:{proxy_parts.port}" if proxy_parts.port else host
    credentials = f"{unquote(proxy_parts.username)}:{unquote(proxy_parts.password or '')}".encode()
    stripped = urlunsplit((proxy_parts.scheme, netloc, proxy_parts.path, proxy_parts.query, proxy_parts.fragment))
    return stripped, {"Proxy-Authorization": "Basic " + base64.b64encode(credentials).decode("ascii")}


def _no_proxy(host: str, no_proxy: str) -> bool:
    """Whether ``NO_PROXY`` (comma-separated hosts and domain suffixes, ``*`` for all) covers ``host``."""

    host = host.lower().strip("[]")
    for entry in no_proxy.split(","):
        name = entry.strip().lower()
        if name == "*":
            return True
        if name.startswith("[") and "]" in name:  # an IPv6 address, perhaps with a port
            name = name[1 : name.index("]")]
        elif name.count(":") == 1:  # a host with a port
            name = name.partition(":")[0]
        name = name.lstrip(".")
        if name and (host == name or host.endswith(f".{name}")):
            return True
    return False


def _redirect_message(url: str, headers: Mapping[str, str]) -> str:
    """What a redirect of a call to ``url`` says, for an error or a log: where it points (its origin only, which
    holds no credentials), and that the gateway follows no redirect."""

    location = headers.get("location")
    target = url_origin(urljoin(url, location)) if location else None
    where = f" to {target}" if target else ""
    return (
        f"the base URL redirects{where}, and hosted calls follow no redirect: set inference.hosted.base_url to the "
        "URL that answers"
    )


def _request_id(headers: Mapping[str, str]) -> str | None:
    for name in ("x-request-id", "request-id"):
        if headers.get(name):
            return str(headers[name])
    return None


def _json(data: bytes) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return None


def error_message(data: bytes) -> str:
    """The provider's message in an error reply (``{"error": {"message"}}``, ``{"type": "error", "error": {...}}``,
    ``{"message"}``, ``{"detail"}``), else the reply's text, shortened."""

    body = _json(data)
    if isinstance(body, Mapping):
        error = body.get("error", body)
        if isinstance(error, Mapping) and isinstance(error.get("message"), str):
            return error["message"]
        if isinstance(error, str):
            return error
        for key in ("message", "detail"):
            if isinstance(body.get(key), str):
                return body[key]
    text = data.decode("utf-8", errors="replace").strip()
    return text if len(text) <= 500 else text[:500] + "..."


def _error_type(data: bytes) -> str | None:
    """The type or code of an error reply, to log in place of its text."""

    body = _json(data)
    if not isinstance(body, Mapping):
        return None
    error = body.get("error", body)
    if isinstance(error, Mapping):
        for key in ("type", "code"):
            if error.get(key) is not None:
                return str(error[key])
    return None


_CONTEXT_LENGTH = re.compile(
    r"context[ _]?(?:length|window|limit)|maximum context|prompt is too long|input (?:is )?too long|"
    r"longer than the model|reduce the length of the (?:messages|input)|input tokens exceed",
    re.I,
)
_TOKEN_COUNTS = (
    # vLLM: "maximum context length is 8192 tokens. However, you requested 100 output tokens and your prompt contains
    # at least 8100 input tokens, for a total of at least 8200 tokens."
    re.compile(r"maximum context length is (?P<window>\d+) tokens.*?prompt contains at least (?P<prompt>\d+)", re.S),
    # OpenAI, older vLLM, OpenRouter: "maximum context length is 8192 tokens. However, you requested (about) 9000
    # tokens" or "your messages resulted in 9000 tokens"
    re.compile(
        r"maximum context length is (?P<window>\d+) tokens.*?(?:requested|resulted in) (?:about )?(?P<prompt>\d+)", re.S
    ),
    # OpenAI: "Input tokens exceed the configured limit of 272000 tokens. Your messages resulted in 280000 tokens."
    re.compile(r"limit of (?P<window>\d+) tokens.*?resulted in (?P<prompt>\d+)", re.S),
    # Anthropic: "prompt is too long: 208000 tokens > 200000 maximum"
    re.compile(r"prompt is too long: (?P<prompt>\d+) tokens > (?P<window>\d+)"),
    # Anthropic: "input length and `max_tokens` exceed context limit: 198000 + 4096 > 200000"
    re.compile(r"exceed context limit: (?P<prompt>\d+) \+ \d+ > (?P<window>\d+)"),
    # SGLang: "The input (9000 tokens) is longer than the model's context length (8192 tokens)."
    re.compile(r"input \((?P<prompt>\d+) tokens\) is longer than the model's context length \((?P<window>\d+)"),
)


def context_length_error(
    status: int, message: str, *, request_id: str | None = None
) -> ProviderContextWindowError | None:
    """A provider's refusal as a context-length error, when its status (400, 413 or 422) and wording say it is one;
    else None.  Each server words it its own way: OpenAI's ``context_length_exceeded``, Anthropic's "prompt is too
    long", vLLM's, SGLang's, OpenRouter's and a LiteLLM proxy's are recognized."""

    if status not in (400, 413, 422) or not _CONTEXT_LENGTH.search(message):
        return None
    if status == 413 and "request_too_large" in message:  # the request's bytes, not its tokens
        return None
    for pattern in _TOKEN_COUNTS:
        found = pattern.search(message)
        if found:
            return ProviderContextWindowError(
                message,
                prompt_tokens=int(found["prompt"]),
                context_window=int(found["window"]),
                status=status,
                request_id=request_id,
            )
    return ProviderContextWindowError(message, status=status, request_id=request_id)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The wait a reply asks for, in seconds (``retry-after-ms``, else ``retry-after`` in seconds or as a date)."""

    milliseconds = headers.get("retry-after-ms")
    if milliseconds:
        with contextlib.suppress(ValueError):
            return max(float(milliseconds) / 1000, 0.0)
    value = headers.get("retry-after")
    if not value:
        return None
    with contextlib.suppress(ValueError):
        return max(float(value), 0.0)
    with contextlib.suppress(TypeError, ValueError, IndexError):
        when = email.utils.parsedate_to_datetime(value)
        return max(when.timestamp() - time.time(), 0.0)
    return None


def backoff_seconds(attempt: int, headers: Mapping[str, str] | None = None) -> float:
    """How long to wait after failed attempt ``attempt`` (1-based): what the reply's retry-after asks, at most
    :data:`MAX_RETRY_AFTER_S`, else :data:`RETRY_BACKOFF_S` with jitter."""

    asked = _retry_after(headers or {})
    if asked is not None:
        return min(asked, MAX_RETRY_AFTER_S)
    base, cap = RETRY_BACKOFF_S
    delay = min(base * 2 ** (attempt - 1), cap)
    return delay + random.uniform(0, delay / 4)


# --------------------------------------------------------------------------- reading replies


class _ErrorEvent(Exception):
    """A stream that carries an error event: the provider failed after its stream started."""


def sse_events(data: bytes) -> list[tuple[str | None, str]]:
    """The events of an SSE body: ``(event name or None, data)``; comments and events without data are skipped."""

    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    events: list[tuple[str | None, str]] = []
    for block in text.split("\n\n"):
        name: str | None = None
        lines: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if key == "event":
                name = value
            elif key == "data":
                lines.append(value)
        if lines:
            events.append((name, "\n".join(lines)))
    return events


def _stream_reply(dialect: str, data: bytes) -> dict[str, Any]:
    """The reply a stream's events make up; :class:`_ErrorEvent` for a stream with an error event, ``ValueError`` for
    one cut off before its last event or that is not JSON."""

    objects = []
    for _, payload in sse_events(data):
        if payload.strip() == "[DONE]":
            continue
        event = json.loads(payload)
        if not isinstance(event, Mapping):
            continue
        if dialect == "openai_chat":
            failed = event.get("error") is not None or event.get("object") == "error"
        elif dialect == "anthropic_messages":
            failed = event.get("type") == "error"
        else:
            failed = event.get("type") in ("error", "response.failed")
        if failed:
            response = event.get("response")
            raise _ErrorEvent(error_message(json.dumps(response if isinstance(response, Mapping) else event).encode()))
        objects.append(event)
    if dialect == "openai_chat":
        return openai_chat.assemble_chunks(objects)
    if dialect == "anthropic_messages":
        return anthropic.assemble_events(objects)
    return responses.final_response(objects)


_REPLY_FIELDS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "openai_chat": openai_chat.reply_fields,
    "openai_responses": responses.reply_fields,
    "anthropic_messages": anthropic.reply_fields,
}


# --------------------------------------------------------------------------- exchanges


@dataclass(frozen=True)
class Relay:
    """A provider's reply as the gateway relays it to a harness: its status, content type and body (the key
    redacted), the headers relayed with it (``retry-after``), and the call recorded from it (a reply that succeeded
    to a model call) or None."""

    status: int
    content_type: str
    body: bytes = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict)
    response: ChatResponse | None = None

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        """The body as JSON, or None when it is not."""
        return _json(self.body)


Outcome = Literal["ok", "retry", "failed", "refused", "invalid", "context_length"]


@dataclass
class _Reply:
    status: int
    headers: dict[str, str]
    body: bytes = field(repr=False)

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type") or "application/json"


@dataclass
class _Attempt:
    """One HTTP attempt: its reply (None without one), what it came to, and why it failed (``problem``: a status or
    an error type, never a body; ``message``: the provider's own message, redacted)."""

    reply: _Reply | None
    outcome: Outcome
    kind: str | None = None  # a retried failure's: rate_limit, timeout or server
    problem: str = ""
    message: str = ""
    payload: Any = None  # the reply, assembled from its stream, when it succeeded
    relayable: bool = True  # whether the reply can reach the harness as it came (a cut stream cannot)


@dataclass
class Exchange:
    """A call's HTTP exchange with the provider: its last attempt, and how many attempts and 429s it took."""

    last: _Attempt
    attempts: int = 0
    rate_limited: int = 0
    seconds: float = 0.0

    @property
    def outcome(self) -> Outcome:
        return "failed" if self.last.outcome == "retry" else self.last.outcome

    @property
    def request_id(self) -> str | None:
        return _request_id(self.last.reply.headers) if self.last.reply is not None else None

    def relay(self, response: ChatResponse | None = None) -> Relay | None:
        """The last reply as the harness gets it, or None when there is none to relay."""

        reply = self.last.reply
        if reply is None or not self.last.relayable:
            return None
        headers = {} if self.outcome == "ok" else {k: reply.headers[k] for k in RELAYED_HEADERS if k in reply.headers}
        return Relay(reply.status, reply.content_type, reply.body, headers, response)

    def describe(self) -> str:
        """One line for an error or a log: the status and the provider's message, or what went wrong."""

        reply = self.last.reply
        if reply is not None and self.last.outcome != "ok":
            message = self.last.message or self.last.problem
            return f"{reply.status} {message}".strip()
        return self.last.problem


@dataclass
class _LoopState:
    """One event loop's HTTP session and request limit: both belong to the loop that made them, and a subprocess
    worker runs each rollout under an ``asyncio.run`` of its own."""

    session: aiohttp.ClientSession
    limit: asyncio.Semaphore | None
    closer: Any = None


class _PerLoop:
    """:class:`_LoopState` by event loop, each closed when its loop shuts down (``asyncio.run`` closes the loop's
    async generators: one per state closes it)."""

    def __init__(self, make: Callable[[], _LoopState]) -> None:
        self._make = make
        self._states: dict[asyncio.AbstractEventLoop, _LoopState] = {}

    async def get(self) -> _LoopState:
        loop = asyncio.get_running_loop()
        for stale in [other for other in self._states if other.is_closed()]:
            self._states.pop(stale, None)  # a loop closed without shutting down its generators
        current = self._states.get(loop)
        if current is None:
            current = self._states[loop] = self._make()
            current.closer = self._close_with_loop(loop, current)
            await current.closer.__anext__()  # runs up to its yield: the loop's shutdown resumes it
        return current

    def _close_with_loop(self, loop: asyncio.AbstractEventLoop, current: _LoopState) -> Any:
        states = self._states

        async def closer() -> Any:
            try:
                yield
            finally:
                states.pop(loop, None)
                try:
                    await current.session.close()
                except Exception:
                    logger.debug("Closing a hosted gateway's HTTP session failed", exc_info=True)

        return closer()

    async def aclose(self) -> None:
        """Close the running loop's session now."""

        current = self._states.pop(asyncio.get_running_loop(), None)
        if current is not None and current.closer is not None:
            await current.closer.aclose()


def _probe_body(dialect: str, model: str) -> dict[str, Any]:
    if dialect == "anthropic_messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "Reply with OK."}]}
    if dialect == "openai_responses":
        return {"model": model, "input": "Reply with OK.", "max_output_tokens": 16, "store": False}
    return {"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_completion_tokens": 16}


class ProbeError(HostedCallError):
    """The base URL did not answer the probe of one dialect (``dialect``) as a model call succeeds: ``detail`` is the
    status and the provider's message, or what went wrong."""

    def __init__(self, dialect: str, detail: str, *, status: int | None = None) -> None:
        super().__init__(f"the {dialect} probe failed: {detail}", status=status)
        self.dialect = dialect
        self.detail = detail


# --------------------------------------------------------------------------- the gateway


class _NoRenderer:
    """A hosted gateway's stand-in renderer: the provider renders its own prompts."""

    def build_generation_prompt(self, messages: Any, **kwargs: Any) -> Any:
        raise TypeError("A hosted model renders its own prompts")

    def parse_response(self, response: Any, **kwargs: Any) -> Any:
        raise TypeError("A hosted model parses its own replies")

    def get_stop_sequences(self) -> list[Any]:
        return []


class _NoSampler:
    """A hosted gateway's stand-in sampler: its base URL answers every call."""

    async def sample_async(self, **kwargs: Any) -> Any:
        raise TypeError("A hosted gateway's base URL answers its calls")

    def sample(self, **kwargs: Any) -> Any:
        raise TypeError("A hosted gateway's base URL answers its calls")


class HostedGateway(ModelGateway):
    """A model gateway over a hosted model at ``settings.base_url`` (see the module docstring).

    ``max_concurrent_requests`` is this process's share of the run's request limit (default:
    ``settings.max_concurrent_requests``, None for no limit).
    """

    def __init__(
        self, settings: HostedSettings, model_name: str, *, max_concurrent_requests: int | None = None
    ) -> None:
        # The stand-ins only satisfy ModelGateway's constructor: a hosted gateway renders and samples nothing.
        super().__init__(
            sampler=_NoSampler(), renderer=cast(Any, _NoRenderer()), tokenizer=cast(Any, None), model_name=model_name
        )
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")
        limit = settings.max_concurrent_requests if max_concurrent_requests is None else max_concurrent_requests
        self.max_concurrent_requests = limit
        self._states = _PerLoop(self._loop_state)

    # ------------------------------------------------------------------ what a hosted model does itself

    def render(self, messages: Any, request: RenderRequest | None = None) -> Any:
        raise TypeError("A hosted model renders its own prompts: there are no prompt tokens to count")

    def fit_max_tokens(self, prompt: Any, max_tokens: int) -> int:
        raise TypeError("A hosted model's context window is the provider's")

    def prepare(self, request: ChatRequest, *, defaults: Any = None, kind: str = "policy") -> PreparedCall:
        raise TypeError("A hosted model's calls are forwarded, not prepared")

    async def sample(self, call: PreparedCall) -> ChatResponse:
        raise TypeError("A hosted model's calls are forwarded, not sampled")

    def sample_sync(self, call: PreparedCall) -> ChatResponse:
        raise TypeError("A hosted model is called asynchronously")

    def parse(self, prompt: Any, result: Any, request: RenderRequest | None = None) -> ChatResponse:
        raise TypeError("A hosted model's replies are read by its dialect")

    def decode_tokens(self, tokens: Sequence[int]) -> list[str]:
        return []  # a hosted call has no sampled token ids

    # ------------------------------------------------------------------ calls

    async def forward(
        self,
        agent: AgentSession,
        dialect: str,
        path: str,
        query: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        kind: str = "policy",
    ) -> Relay:
        """Forward a harness's request in ``dialect`` to ``path`` under the base URL (with ``query``), record the
        call on ``agent`` as a ``kind`` call when it succeeds, and return the provider's reply as the harness gets it.

        Raises before anything is sent what the gateway refuses as it does with Tinker: a kind of call the agent may
        not make (``CallKindRefusedError``), a request training refuses (a ValueError with training's message), a
        judge call while judges are off (``JudgesOffError``), a closed agent or a gateway that is shut down.  Raises
        :class:`HostedCallError` for a call that failed after its retries without a reply to relay (a timeout, a
        connection error, a stream cut off), and :class:`HostedRefusedError` when the key's variable is not set.  A
        ``policy`` call counts against the agent's call cap while it is in flight, as a Tinker sample does; the
        server answers one past the cap with the budget stop instead of forwarding it.
        """

        self._admit(agent, kind)
        started = time.perf_counter()
        sent = self._hosted_body(dialect, body, agent, kind)
        prepared = time.perf_counter()
        exchange = await self._send(agent, kind, dialect, path, query, sent, headers)
        response = None
        if exchange.outcome == "ok":
            response = self._response(dialect, exchange, started, prepared, in_process=False)
            self._ensure_open(agent)  # an agent closed while its call was in flight records nothing
            agent.record(response, kind=kind)
        relay = exchange.relay(response)
        if relay is None:
            raise self._error(exchange)
        return relay

    async def chat(
        self, agent: AgentSession, request: ChatRequest | PreparedCall, *, kind: str = "policy"
    ) -> ChatResponse:
        """``ctx.chat``'s call for ``agent``: its messages and parameters (:attr:`ChatRequest.params`) as an
        unstreamed Chat Completions request, normalized as one over HTTP, and recorded as a ``kind`` call.

        Raises what a Tinker call raises: :class:`ProviderContextWindowError` (a ``ContextWindowExceededError``) for
        a context-length refusal, :class:`HostedRequestError` (a ValueError) for another refused request,
        ``CallKindRefusedError`` for a kind of call the agent may not make, ``JudgesOffError`` for a judge call
        while judges are off, and :class:`HostedCallError` (or :class:`HostedRefusedError`) for a call that failed.
        A reply cut off at its token limit runs no tools, as with Tinker: its tool calls are counted, not emitted.
        """

        if isinstance(request, PreparedCall):
            raise TypeError("A hosted gateway forwards requests; it does not sample prepared calls")
        self._admit(agent, kind)
        started = time.perf_counter()
        body = openai_chat.in_process_body(request.messages, request.params)
        sent = self._hosted_body("openai_chat", body, agent, kind)
        prepared = time.perf_counter()
        exchange = await self._send(agent, kind, "openai_chat", PATHS["openai_chat"], "", sent, {})
        if exchange.outcome != "ok":
            raise self._error(exchange)
        response = self._response("openai_chat", exchange, started, prepared, in_process=True)
        self._ensure_open(agent)
        agent.record(response, kind=kind)
        return response

    async def count_tokens(
        self, agent: AgentSession, body: Mapping[str, Any], query: str, headers: Mapping[str, str]
    ) -> Relay:
        """Forward a Messages ``count_tokens`` request, normalized as a Messages request, and return the provider's
        reply.  It is not recorded, takes no step and is never counted as a failed call, as with Tinker, where the
        renderer counts."""

        self._ensure_running()
        self._ensure_open(agent)
        sent = anthropic.hosted_body(body, model=self.model_name, count_tokens=True)
        exchange = await self._exchange("anthropic_messages", COUNT_TOKENS_PATH, query, sent, headers, parse=False)
        relay = exchange.relay()
        if relay is None:
            raise self._error(exchange)
        return relay

    async def probe(self, dialects: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Send one minimal request in each of ``dialects`` for the model (a few retries, a short timeout), and return
        the model each answered with (``{dialect: {"served_model": ...}}``).  Raises :class:`ProbeError` for the first
        dialect that is not answered as a model call succeeds: a wrong key, a model the server does not serve, a base
        URL without the dialect's route, or nothing answering at all."""

        timeout = min(float(self.settings.timeout_s), PROBE_TIMEOUT_S)
        served: dict[str, dict[str, Any]] = {}
        for dialect in dialects:
            headers = {"anthropic-version": ANTHROPIC_VERSION} if dialect == "anthropic_messages" else {}
            exchange = await self._exchange(
                dialect,
                PATHS[dialect],
                "",
                _probe_body(dialect, self.model_name),
                headers,
                max_retries=PROBE_RETRIES,
                timeout_s=timeout,
            )
            if exchange.outcome != "ok":
                status = exchange.last.reply.status if exchange.last.reply is not None else None
                raise ProbeError(dialect, exchange.describe(), status=status)
            model = exchange.last.payload.get("model") if isinstance(exchange.last.payload, Mapping) else None
            served[dialect] = {"served_model": model if isinstance(model, str) else None}
        return served

    async def aclose(self) -> None:
        """Close this event loop's HTTP session."""

        await self._states.aclose()

    # ------------------------------------------------------------------ internals

    def _admit(self, agent: AgentSession, kind: str) -> None:
        """Refuse what ``ModelGateway.chat`` refuses before it samples: an unknown call kind, a kind the agent may
        not make (``AgentSession.check_kind``), a judge call while judges are off, a gateway that is shut down, a
        closed agent."""

        check_call_kind(kind)
        agent.check_kind(kind)
        if kind == "judge" and not agent.judges:
            raise agent.refuse_judge("A judge call")
        self._ensure_running()
        self._ensure_open(agent)

    async def _send(
        self,
        agent: AgentSession,
        kind: str,
        dialect: str,
        path: str,
        query: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Exchange:
        """:meth:`_exchange` for one of ``agent``'s calls, counted as a Tinker sample is (``ModelGateway.chat``): a
        ``policy`` call counts against the agent's call cap while it is in flight (``AgentSession.call_tally``),
        and a call the backend failed (:meth:`_count`) or that could not be sent counts in ``gateway_errors``, which
        ``on_harness_exit``'s ``grade_if_gateway_clean`` reads."""

        counted = 1 if kind == "policy" else 0
        agent.policy_calls_in_flight += counted
        agent.call_tally.in_flight += counted
        try:
            exchange = await self._exchange(dialect, path, query, body, headers)
        except Exception:
            agent.gateway_errors += 1
            raise
        finally:
            agent.policy_calls_in_flight -= counted
            agent.call_tally.in_flight -= counted
        self._count(agent, dialect, exchange)
        return exchange

    def _hosted_body(
        self, dialect: str, body: Mapping[str, Any], agent: AgentSession, kind: str = "policy"
    ) -> dict[str, Any]:
        defaults = agent.sampling_defaults
        if defaults is not None and defaults.precedence == "config" and agent.sampling_kind(kind) == "policy":
            # sampling_precedence: config, as SamplingOverrides.for_call applies it with Tinker: the rollout's
            # sampling decides a policy call's, never the harness's own (an aux call of an agent with policy_only
            # samples as a policy call: AgentSession.sampling_kind).  The call's temperature, top_p and top_k go, and
            # so does its completion budget when the rollout sets one; the dialect then adds the rollout's (top_k never
            # reaches a hosted model), and the server's defaults stand for what the rollout leaves unset.
            dropped = {"temperature", "top_p", "top_k"}
            if defaults.max_tokens is not None:
                dropped.update(_COMPLETION_BUDGETS.get(dialect, ()))
            body = {key: value for key, value in body.items() if key not in dropped}
        if dialect == "openai_chat":
            return openai_chat.hosted_body(body, model=self.model_name, defaults=defaults)
        if dialect == "openai_responses":
            return responses.hosted_body(body, model=self.model_name, defaults=defaults)
        if dialect == "anthropic_messages":
            return anthropic.hosted_body(body, model=self.model_name, defaults=defaults)
        raise ValueError(f"Unknown dialect {dialect!r}; expected one of {DIALECTS}")

    def _loop_state(self) -> _LoopState:
        limit = self.max_concurrent_requests
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),  # max_concurrent_requests is the limit
            cookie_jar=aiohttp.DummyCookieJar(),  # every call stands alone
            trust_env=False,  # env_proxy reads the proxy variables, and nothing else is read
        )
        return _LoopState(session=session, limit=asyncio.Semaphore(limit) if limit else None)

    def _key(self) -> str:
        """The key, from the variable ``api_key_env`` names ("" without one), without the whitespace around it: a
        variable read from a file often ends in a newline, which no header may carry."""

        name = self.settings.api_key_env
        return os.environ.get(name, "").strip() if name else ""

    def _secrets(self) -> list[str]:
        value = self._key()
        return [value] if len(value) >= MIN_SECRET_LENGTH else []

    def _request_headers(self, dialect: str, headers: Mapping[str, str]) -> dict[str, str] | None:
        """The forwarded client headers and the key's; None when the key's variable is not set."""

        allowed = FORWARDED_HEADERS.get(dialect, ())
        out = {name.lower(): value for name, value in headers.items() if name.lower() in allowed}
        out["content-type"] = "application/json"
        if self.settings.api_key_env:
            key = self._key()
            if not key:
                return None
            if dialect == "anthropic_messages":
                out["x-api-key"] = key
            else:
                out["authorization"] = f"Bearer {key}"
        return out

    async def _exchange(
        self,
        dialect: str,
        path: str,
        query: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        *,
        max_retries: int | None = None,
        timeout_s: float | None = None,
        parse: bool = True,
    ) -> Exchange:
        """Send ``body`` within the request limit, retrying what failed within ``max_retries + 1`` attempts."""

        started = time.perf_counter()
        request_headers = self._request_headers(dialect, headers)
        if request_headers is None:
            problem = f"the variable {self.settings.api_key_env} that inference.hosted.api_key_env names is not set"
            return Exchange(_Attempt(None, "refused", problem=problem, relayable=False))
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        data = json.dumps(body).encode()
        stream = bool(body.get("stream"))
        allowed = (self.settings.max_retries if max_retries is None else max_retries) + 1
        timeout = float(self.settings.timeout_s if timeout_s is None else timeout_s)
        state = await self._states.get()
        exchange = Exchange(_Attempt(None, "retry"))
        async with state.limit if state.limit is not None else contextlib.nullcontext():
            while True:
                exchange.attempts += 1
                attempt = await self._attempt(
                    state.session, url, request_headers, data, dialect, stream, timeout, parse
                )
                exchange.last = attempt
                if attempt.reply is not None and attempt.reply.status == 429:
                    exchange.rate_limited += 1
                if attempt.outcome != "retry" or exchange.attempts >= allowed:
                    break
                delay = backoff_seconds(exchange.attempts, attempt.reply.headers if attempt.reply else None)
                logger.warning(
                    "A %s call to %s failed (%s); retrying in %.1fs (attempt %d of %d)",
                    dialect,
                    url_origin(self.base_url),
                    self._what(attempt),
                    delay,
                    exchange.attempts + 1,
                    allowed,
                )
                await asyncio.sleep(delay)
        exchange.seconds = time.perf_counter() - started
        return exchange

    async def _attempt(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: Mapping[str, str],
        data: bytes,
        dialect: str,
        stream: bool,
        timeout: float,
        parse: bool,
    ) -> _Attempt:
        proxy, proxy_headers = env_proxy(url)
        secrets = self._secrets()
        try:
            async with session.post(
                url,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                proxy=proxy,
                proxy_headers=proxy_headers,
                allow_redirects=False,  # a redirect refuses the call (REFUSED_STATUSES): the key goes nowhere else
            ) as response:
                raw = await response.read()
                status = response.status
                reply_headers = {name.lower(): value for name, value in response.headers.items()}
        except TimeoutError:
            return _Attempt(None, "retry", kind="timeout", problem=f"no reply within timeout_s ({timeout:g}s)")
        except aiohttp.ClientError as exc:  # a refused or reset connection, a reply cut off midway
            return _Attempt(None, "retry", kind="server", problem=redact(f"{type(exc).__name__}: {exc}", secrets))
        reply = _Reply(status, reply_headers, redact_bytes(raw, secrets))
        if 200 <= status < 300:
            if not parse:
                return _Attempt(reply, "ok")
            event_stream = reply.content_type.startswith("text/event-stream")
            try:
                payload = _stream_reply(dialect, reply.body) if stream and event_stream else json.loads(reply.body)
            except _ErrorEvent as exc:
                return _Attempt(reply, "retry", kind="server", problem="an error event in the stream", message=str(exc))
            except ValueError as exc:  # a stream that ended before its last event, or a body that is not JSON
                return _Attempt(reply, "retry", kind="server", problem=f"an incomplete reply: {exc}", relayable=False)
            return _Attempt(reply, "ok", payload=payload)
        if 300 <= status < 400:
            return _Attempt(reply, "refused", problem=f"status {status}", message=_redirect_message(url, reply.headers))
        message = error_message(reply.body)
        if status == 429:
            return _Attempt(reply, "retry", kind="rate_limit", problem="status 429", message=message)
        if status in (408, 504):
            return _Attempt(reply, "retry", kind="timeout", problem=f"status {status}", message=message)
        if status == 409 or status >= 500:
            return _Attempt(reply, "retry", kind="server", problem=f"status {status}", message=message)
        if status in REFUSED_STATUSES:
            return _Attempt(reply, "refused", problem=f"status {status}", message=message)
        if context_length_error(status, message) is not None:
            return _Attempt(reply, "context_length", problem=f"status {status}", message=message)
        return _Attempt(reply, "invalid", problem=f"status {status}", message=message)

    @staticmethod
    def _what(attempt: _Attempt) -> str:
        """An attempt's failure for a log: its status, error type and request id, or its error; never a body."""

        reply = attempt.reply
        if reply is None:
            return attempt.problem
        details = [attempt.problem]
        error_type = _error_type(reply.body)
        if error_type:
            details.append(f"error type {error_type}")
        request_id = _request_id(reply.headers)
        if request_id:
            details.append(f"request id {request_id}")
        return ", ".join(details)

    def _count(self, agent: AgentSession, dialect: str, exchange: Exchange) -> None:
        """Count on ``agent`` what the provider refused or failed: a context-length refusal as a context overflow
        (``context_overflows``), and failed, refused and rejected calls (:data:`BACKEND_FAILURES_MISC_KEY`), the
        failed and refused ones also as the gateway's errors (``gateway_errors``), as a failed Tinker sample is."""

        outcome = exchange.outcome
        if outcome == "ok":
            return
        if outcome == "context_length":
            agent.context_overflows += 1
            return
        record = agent.misc.setdefault(BACKEND_FAILURES_MISC_KEY, {"count": 0, "kinds": {}, "rejected": 0})
        where = url_origin(self.base_url)
        if outcome == "invalid":
            record["rejected"] += 1  # an invalid request: the harness's, not an outage
            logger.warning("%s refused an %s request as invalid (%s)", where, dialect, self._what(exchange.last))
            return
        kind = "refused" if outcome == "refused" else exchange.last.kind or "server"
        agent.gateway_errors += 1  # the backend failed or refused the call, not the harness
        record["count"] += 1
        record["kinds"][kind] = record["kinds"].get(kind, 0) + 1
        if outcome == "refused" and not record.get("fatal"):
            record["fatal"] = f"{where} refused an {dialect} call: {exchange.describe()}"
        logger.error(
            "A %s call to %s failed after %d attempt(s): %s",
            dialect,
            where,
            exchange.attempts,
            self._what(exchange.last),
        )

    def _error(self, exchange: Exchange) -> Exception:
        """The exception a call that did not succeed raises in process, or over HTTP without a reply to relay."""

        reply = exchange.last.reply
        status = reply.status if reply is not None else None
        request_id = exchange.request_id
        detail = exchange.describe()
        outcome = exchange.outcome
        if outcome == "context_length" and reply is not None:
            error = context_length_error(reply.status, exchange.last.message, request_id=request_id)
            if error is not None:
                return error
        if outcome == "invalid":
            return HostedRequestError(f"{url_origin(self.base_url)} refused the request: {detail}", status=status)
        if outcome == "refused":
            return HostedRefusedError(
                f"{url_origin(self.base_url)} refused the call: {detail}",
                status=status,
                request_id=request_id,
                attempts=exchange.attempts,
            )
        kind: Literal["timeout", "server"] = "timeout" if exchange.last.kind == "timeout" else "server"
        return HostedCallError(
            f"{url_origin(self.base_url)} did not answer after {exchange.attempts} attempt(s): {detail}",
            kind=kind,
            status=status,
            request_id=request_id,
            attempts=exchange.attempts,
        )

    def _response(
        self, dialect: str, exchange: Exchange, started: float, prepared: float, *, in_process: bool
    ) -> ChatResponse:
        parse_started = time.perf_counter()
        payload = exchange.last.payload if isinstance(exchange.last.payload, Mapping) else {}
        fields = _REPLY_FIELDS[dialect](payload)
        usage = fields["usage"]
        reported = bool(fields["usage_reported"])
        served, response_id, stop = payload.get("model"), payload.get("id"), fields["finish_reason"]
        record = ProviderRecord(
            dialect=dialect,
            usage=usage,
            usage_reported=reported,
            served_model=served if isinstance(served, str) else None,
            response_id=response_id if isinstance(response_id, str) else None,
            request_id=exchange.request_id,
            raw_stop_reason=stop if isinstance(stop, str) else None,
            refusal=bool(fields["refusal"]),
            attempts=exchange.attempts,
            rate_limited=exchange.rate_limited,
            cost_usd=call_cost(usage, self.settings.pricing) if reported else None,
        )
        tool_calls = fields["tool_calls"]
        truncated = bool(fields["truncated"])
        # Over HTTP the harness gets the reply as it came, tool calls and all; ctx.chat's reply cut off at its token
        # limit runs no tools, as with Tinker.
        emitted = tool_calls if tool_calls and not (in_process and truncated) else None
        return ChatResponse(
            call_id=generate_id("hosted-"),
            model=self.model_name,
            prompt=None,
            tokens=[],
            logprobs=None,
            stop_reason="length" if truncated else "stop",
            parse_success=True,
            role="assistant",
            content=fields["content"] or "",
            reasoning_content=fields["reasoning_content"],
            tool_calls=emitted,
            parsed_tool_calls=len(tool_calls or []),
            routing_id=None,
            timing=CallTiming(started, prepared - started, exchange.seconds, parse_started, time.perf_counter()),
            provider=record,
        )


__all__ = [
    "BACKEND_FAILURES_MISC_KEY",
    "COUNT_TOKENS_PATH",
    "DIALECTS",
    "FORWARDED_HEADERS",
    "PATHS",
    "REDACTED",
    "Exchange",
    "HostedCallError",
    "HostedGateway",
    "HostedRefusedError",
    "HostedRequestError",
    "HostedSettings",
    "ProbeError",
    "ProviderContextWindowError",
    "Relay",
    "backoff_seconds",
    "call_cost",
    "context_length_error",
    "env_proxy",
    "error_message",
    "redact",
    "redact_bytes",
    "secret_forms",
    "sse_events",
    "url_origin",
]
