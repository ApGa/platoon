"""Proxy helpers for the modern AReaL OpenAI session flow."""

from __future__ import annotations

import aiohttp
from areal.experimental.openai.proxy.client_session import OpenAIProxyClient


class ArealProxySession:
    """Small wrapper around AReaL's session-key based proxy client."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        task_id: str,
        admin_api_key: str,
    ):
        self._client = OpenAIProxyClient(
            session=session,
            base_url=base_url,
            task_id=task_id,
            admin_api_key=admin_api_key,
        )

    @property
    def session_id(self) -> str | None:
        return self._client.session_id

    @property
    def session_api_key(self) -> str:
        return self._client.session_api_key

    async def export_interactions(self):
        return await self._client.export_interactions(discount=1.0, style="individual")

    async def __aenter__(self) -> "ArealProxySession":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._client.__aexit__(exc_type, exc_value, traceback)
