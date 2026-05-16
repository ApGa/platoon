"""Proxy helpers for the modern AReaL OpenAI session flow."""

from __future__ import annotations

import aiohttp  # pyright: ignore[reportMissingImports]
from areal.experimental.openai.proxy.client_session import OpenAIProxyClient  # pyright: ignore[reportMissingImports]
from areal.experimental.openai.proxy.server import GRANT_CAPACITY_PATHNAME  # pyright: ignore[reportMissingImports]
from areal.infra.utils.http import ensure_end_with_slash  # pyright: ignore[reportMissingImports]


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
        self._session = session
        self._base_url = ensure_end_with_slash(base_url)
        self._admin_api_key = admin_api_key

    @property
    def session_id(self) -> str | None:
        return self._client.session_id

    @property
    def session_api_key(self) -> str:
        return self._client.session_api_key

    async def export_interactions(self):
        return await self._client.export_interactions(discount=1.0, style="individual")

    async def set_last_reward(self, reward: float) -> None:
        await self._client.set_last_reward(reward)

    async def __aenter__(self) -> "ArealProxySession":
        await self._grant_capacity()
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self._set_default_proxy_reward()
        await self._client.__aexit__(exc_type, exc_value, traceback)

    async def _grant_capacity(self) -> None:
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        async with self._session.post(
            f"{self._base_url}{GRANT_CAPACITY_PATHNAME}",
            headers=headers,
        ) as response:
            response.raise_for_status()

    async def _set_default_proxy_reward(self) -> None:
        """Use AReaL's public API to avoid missing-reward export warnings.

        Platoon computes rewards from completed trajectories after export, so
        the proxy-side reward value is only a placeholder.
        """
        try:
            await self.set_last_reward(0.0)
        except aiohttp.ClientResponseError as exc:
            if exc.status != 400:
                raise
