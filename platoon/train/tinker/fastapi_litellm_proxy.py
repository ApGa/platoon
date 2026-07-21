"""FastAPI proxy that routes Apptainer LLM calls through LiteLLM.

The trainer owns TinkerLLM initialization and LiteLLM custom-provider
registration. This server only provides an OpenAI-compatible HTTP endpoint,
wraps each request in ``TinkerLLMProxySession``, and stores captured
interactions by rollout session id.
"""

from __future__ import annotations

import logging
import socket
import threading
from collections import defaultdict
from typing import Any

import litellm
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from litellm.exceptions import APIConnectionError, BadRequestError, ContextWindowExceededError

from platoon.train.tinker.proxy import TinkerLLMInteraction, TinkerLLMProxySession

logger = logging.getLogger(__name__)

SESSION_HEADER = "X-Platoon-Tinker-Session"
_active_proxy: FastAPILiteLLMTinkerHTTPProxyServer | None = None


def get_active_tinker_http_proxy() -> FastAPILiteLLMTinkerHTTPProxyServer | None:
    return _active_proxy


def set_active_tinker_http_proxy(proxy: FastAPILiteLLMTinkerHTTPProxyServer | None) -> None:
    global _active_proxy
    _active_proxy = proxy


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _is_context_window_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "prompt length plus max_tokens exceeds the model's context window" in message
        or "input exceeds the context window" in message
        or "maximum context length" in message
        or "contextwindowexceedederror" in message
    )


class FastAPILiteLLMTinkerHTTPProxyServer:
    def __init__(
        self,
        litellm_model_name: str,
        host: str = "127.0.0.1",
        port: int | None = None,
        context_window_length: int | None = None,
    ):
        self.host = host
        self.port = port if port is not None else _find_free_port(host)
        self.litellm_model_name = litellm_model_name
        self.context_window_length = context_window_length
        self.app = FastAPI(title="Platoon Tinker LiteLLM Proxy")
        self._lock = threading.Lock()
        self._interactions_by_session: defaultdict[str, dict[str, TinkerLLMInteraction]] = defaultdict(dict)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._add_routes()

    def _add_routes(self) -> None:
        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.get("/v1/models")
        @self.app.get("/models")
        async def models() -> dict[str, Any]:
            return {"object": "list", "data": [{"id": "platoon-tinker", "object": "model"}]}

        @self.app.post("/v1/chat/completions")
        @self.app.post("/chat/completions")
        async def chat_completions(
            payload: dict[str, Any],
            x_platoon_tinker_session: str | None = Header(default=None, alias=SESSION_HEADER),
        ) -> dict[str, Any]:
            if not x_platoon_tinker_session:
                raise HTTPException(status_code=400, detail=f"Missing required header: {SESSION_HEADER}")
            return await self.chat_completion(payload, x_platoon_tinker_session)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def model_name(self) -> str:
        return "openai/platoon-tinker"

    @property
    def api_key(self) -> str:
        return "sk-xxx"

    async def chat_completion(self, payload: dict[str, Any], session_id: str) -> dict[str, Any]:
        messages = payload.get("messages")
        if messages is None:
            raise HTTPException(status_code=400, detail="Missing required field: messages")
        if payload.get("stream"):
            raise HTTPException(status_code=400, detail="Streaming is not supported by this proxy")

        completion_kwargs = {k: v for k, v in payload.items() if k not in {"model", "messages", "stream"}}

        async with TinkerLLMProxySession() as proxy_session:
            try:
                response = await litellm.acompletion(
                    model=self.litellm_model_name,
                    messages=messages,
                    **completion_kwargs,
                )
            except (ValueError, APIConnectionError, BadRequestError, ContextWindowExceededError) as e:
                if _is_context_window_error(e):
                    raise HTTPException(status_code=400, detail=str(e)) from e
                raise
            interactions = dict(proxy_session.interactions)

        with self._lock:
            self._interactions_by_session[session_id].update(interactions)

        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=True)
        return dict(response)

    def start(self) -> None:
        if self._server is not None:
            return
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning", workers=1)
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        set_active_tinker_http_proxy(self)
        logger.info("Started FastAPI LiteLLM Tinker proxy at %s", self.base_url)

    def stop(self) -> None:
        if self._server is None:
            return
        try:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=5.0)
        finally:
            self._server = None
            self._thread = None
            if get_active_tinker_http_proxy() is self:
                set_active_tinker_http_proxy(None)
            logger.info("Stopped FastAPI LiteLLM Tinker proxy")

    def pop_interactions(self, session_id: str) -> dict[str, TinkerLLMInteraction]:
        with self._lock:
            return self._interactions_by_session.pop(session_id, {})

    def discard_session(self, session_id: str) -> None:
        with self._lock:
            self._interactions_by_session.pop(session_id, None)