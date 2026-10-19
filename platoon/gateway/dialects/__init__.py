"""Wire dialects of the model gateway.

A dialect maps its request body onto a :class:`~platoon.gateway.core.ChatRequest` and a
:class:`~platoon.gateway.core.ChatResponse` back onto its response body and SSE events; the server owns
authentication, keepalives and status codes, so every dialect records into the same agent through the same core.

* :mod:`~platoon.gateway.dialects.openai_chat`: OpenAI Chat Completions, the reference dialect.
* :mod:`~platoon.gateway.dialects.responses`: OpenAI Responses (Codex), experimental.
* :mod:`~platoon.gateway.dialects.anthropic`: Anthropic Messages (Claude Code), experimental.

Each module also holds what a hosted gateway (:mod:`platoon.gateway.hosted`) needs to forward a request in its
dialect: the request's normalization (``hosted_body``), which shares its tool rules with ``chat_request``, and what a
call's record takes from a reply (``reply_fields``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from typing import Any, Literal, Protocol

from platoon.gateway.core import (
    AgentClosedError,
    ChatRequest,
    ChatResponse,
    ContextWindowExceededError,
    HostedCallError,
    JudgesOffError,
    UnknownAgentError,
)

ErrorKind = Literal[
    "authentication", "permission", "agent_closed", "context_length", "invalid_request", "timeout", "server"
]

ERROR_STATUS: dict[ErrorKind, int] = {
    "authentication": 401,
    "permission": 403,
    "agent_closed": 410,
    "context_length": 400,
    "invalid_request": 400,
    "timeout": 504,
    "server": 500,
}


def error_kind(exc: BaseException) -> ErrorKind:
    """What went wrong, independent of how a dialect words it.  A hosted call that failed after its retries
    (:class:`~platoon.gateway.core.HostedCallError`) is a timeout or a server error, as its ``kind`` says."""
    if isinstance(exc, UnknownAgentError):
        return "authentication"
    if isinstance(exc, JudgesOffError):
        return "permission"
    if isinstance(exc, AgentClosedError):
        return "agent_closed"
    if isinstance(exc, ContextWindowExceededError):
        return "context_length"
    if isinstance(exc, HostedCallError):
        return exc.kind
    if isinstance(exc, ValueError):
        return "invalid_request"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return "server"


class Dialect(Protocol):
    # The dialect's name, as a harness spec names it (``platoon.harness.spec.Dialect``).
    name: str

    def chat_request(self, body: Mapping[str, Any]) -> ChatRequest:
        """The request body as a gateway request; ValueError for what the gateway cannot honour."""
        ...

    def response(self, response: ChatResponse, body: Mapping[str, Any]) -> dict[str, Any]:
        """The non-streaming response body."""
        ...

    def events(self, response: ChatResponse, body: Mapping[str, Any]) -> Iterator[bytes]:
        """The whole response as SSE events, sent once the call has been sampled."""
        ...

    def error(self, exc: BaseException) -> dict[str, Any]:
        """The error body for ``exc`` (sent with ``ERROR_STATUS[error_kind(exc)]``)."""
        ...

    def error_event(self, exc: BaseException) -> bytes:
        """``exc`` as an SSE event, for failures after a stream has started."""
        ...


def sse(data: Mapping[str, Any], event: str | None = None) -> bytes:
    """One SSE event; ``event`` names it, as the Responses and Anthropic dialects do."""
    name = f"event: {event}\n" if event else ""
    return f"{name}data: {json.dumps(data)}\n\n".encode()
