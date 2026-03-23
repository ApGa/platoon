from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import logging

from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)

TAVILY_CLIENT = AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

_TAVILY_LIMITER: "_TavilyLimiter | None" = None
_TAVILY_LIMITER_PID: int | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _TavilyLimiter:
    def __init__(self, max_requests_per_minute: int, max_concurrency: int):
        self._max_requests_per_minute = max_requests_per_minute
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        await self._semaphore.acquire()
        try:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    cutoff = now - 60.0
                    while self._timestamps and self._timestamps[0] <= cutoff:
                        self._timestamps.popleft()

                    if len(self._timestamps) < self._max_requests_per_minute:
                        self._timestamps.append(now)
                        logger.debug(
                            "[TavilyLimiter pid=%s] acquired slot (%d/%d used in window)",
                            os.getpid(),
                            len(self._timestamps),
                            self._max_requests_per_minute,
                        )
                        return

                    wait_time = max(0.0, 60.0 - (now - self._timestamps[0]))
                logger.info(
                    "[TavilyLimiter pid=%s] rate limit reached (%d/%d), waiting %.1fs",
                    os.getpid(),
                    len(self._timestamps),
                    self._max_requests_per_minute,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
        except Exception:
            self._semaphore.release()
            raise

    def release(self) -> None:
        self._semaphore.release()


def _get_tavily_limiter() -> "_TavilyLimiter | None":
    if not _env_flag("PLATOON_TAVILY_RATE_LIMIT_ENABLED", default=False):
        return None

    global _TAVILY_LIMITER, _TAVILY_LIMITER_PID

    try:
        max_requests_per_minute = int(os.getenv("PLATOON_TAVILY_MAX_REQUESTS_PER_MINUTE", "200"))
        max_concurrency = int(os.getenv("PLATOON_TAVILY_MAX_CONCURRENCY", "1000"))
    except ValueError:
        return None

    if max_requests_per_minute <= 0 or max_concurrency <= 0:
        return None

    pid = os.getpid()
    if _TAVILY_LIMITER is None or _TAVILY_LIMITER_PID != pid:
        _TAVILY_LIMITER = _TavilyLimiter(
            max_requests_per_minute=max_requests_per_minute,
            max_concurrency=max_concurrency,
        )
        _TAVILY_LIMITER_PID = pid

    return _TAVILY_LIMITER


@asynccontextmanager
async def _maybe_rate_limited_tavily_request() -> AsyncIterator[None]:
    limiter = _get_tavily_limiter()
    if limiter is None:
        if not getattr(_maybe_rate_limited_tavily_request, "_warned", False):
            logger.warning(
                "[TavilyLimiter pid=%s] Rate limiter is DISABLED "
                "(set PLATOON_TAVILY_RATE_LIMIT_ENABLED=1 to enable)",
                os.getpid(),
            )
            _maybe_rate_limited_tavily_request._warned = True
        yield
        return

    await limiter.acquire()
    try:
        yield
    finally:
        limiter.release()


async def search_web(query: str, max_results: int = 5) -> dict:
    """Search the web for information related to the query.
    Args:
        query: The query to search for.
        max_results: (optional) The maximum number of results to return. Betweem 1 and 20. Defaults to 5.

    Returns:
        A dictionary containing the search results in the following format:
            {
                "query": str,
                "follow_up_questions": list[str],
                "answer": str,
                "images": list[str],
                "results": list[dict],
                "response_time": float,
                "request_id": str,
            }

            A single result is a dictionary with the following keys:
            {
                "url": str,
                "title": str,
                "content": str,
                "score": float,
                "raw_content": str | None,
            }
    """
    async with _maybe_rate_limited_tavily_request():
        response = await TAVILY_CLIENT.search(query=query, max_results=max_results)
    return response


async def view_webpage_content(url: str) -> str:
    """View the content of a webpage.
    Args:
        url: The URL of the webpage to view.

    Returns:
        A string containing the content of the webpage. This may be very long.
        It is wise to first inspect the size of the response before deciding to print it out
        as it may exceed your context window. A good rule of thumb: if the response is greater than
        32K characters, you may want to just look at the first 32K characters or take some other 
        reasonable approach to avoid printing out the entire response.
    """
    async with _maybe_rate_limited_tavily_request():
        response = await TAVILY_CLIENT.extract(urls=url)

    results = response["results"]
    if not results:
        raise ValueError("Content extraction failed for the given URL.")

    return results[0]["raw_content"]
