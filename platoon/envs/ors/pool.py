"""Env-server URL pools and per-rollout sharding: the SHA-1 of a rollout id picks its env server.

A pool comes from, in order: the service's static URLs (the URI's hosts or
``urls=``), ``OPENREWARD_SESSION_URLS_<LABEL>`` (the label upper-cased, other
characters as ``_``; or a named variable), the legacy process-wide
``OPENREWARD_SESSION_URLS`` (single-environment runs only: in a mixture every
entry needs its own pool, since servers serve different environments), and
finally ``OPENREWARD_SESSION_URL`` or ``http://localhost:8080``. Variables are
read when a session opens, so launchers can export live pools after config load.
A run checks a mixture's pools when it starts, before it contacts any server:
entries without a variant that would reach one server under different env names
(say, one whose variable is missing falls back to a pool another entry uses) are
refused (:func:`platoon.train.auto.check_mixture_pools`).
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence

DEFAULT_SESSION_URL = "http://localhost:8080"
SESSION_URL_ENV_VAR = "OPENREWARD_SESSION_URL"
LEGACY_POOL_ENV_VAR = "OPENREWARD_SESSION_URLS"
POOL_ENV_VAR_PREFIX = "OPENREWARD_SESSION_URLS_"


def pool_env_var(label: str) -> str:
    """``OPENREWARD_SESSION_URLS_<LABEL>`` for an environment label."""

    return POOL_ENV_VAR_PREFIX + "".join(char.upper() if char.isalnum() else "_" for char in label.strip())


def resolve_session_urls(
    *,
    urls: Sequence[str] | None,
    label: str,
    env_var: str | None = None,
    legacy_global_pool: bool = True,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """The live URL pool for one environment (see the module docstring for the order)."""

    return resolve_session_pool(
        urls=urls, label=label, env_var=env_var, legacy_global_pool=legacy_global_pool, environ=environ
    )[0]


def resolve_session_pool(
    *,
    urls: Sequence[str] | None,
    label: str,
    env_var: str | None = None,
    legacy_global_pool: bool = True,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[str], str | None]:
    """:func:`resolve_session_urls`, and where the pool came from when it is not the environment's own: the
    process-wide ``OPENREWARD_SESSION_URLS`` or ``OPENREWARD_SESSION_URL``, or the default URL, which every
    environment without a pool of its own shares.  None for its static URLs or its variable's."""

    environ = os.environ if environ is None else environ
    own = [url.strip() for url in urls or () if url.strip()] or _split(environ.get(env_var or pool_env_var(label), ""))
    if own:
        return [_with_scheme(url) for url in own], None
    if legacy_global_pool and (legacy := _split(environ.get(LEGACY_POOL_ENV_VAR, ""))):
        return [_with_scheme(url) for url in legacy], LEGACY_POOL_ENV_VAR
    if SESSION_URL_ENV_VAR in environ:
        return [_with_scheme(environ[SESSION_URL_ENV_VAR])], SESSION_URL_ENV_VAR
    return [DEFAULT_SESSION_URL], DEFAULT_SESSION_URL


def select_session_url(candidates: Sequence[str], shard_key: str) -> str:
    """The pool member for ``shard_key`` (a rollout id): stable per key, spread across keys."""

    if not candidates:
        raise ValueError("An env-server pool needs at least one URL")
    digest = hashlib.sha1(shard_key.encode("utf-8")).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]


def _split(value: str) -> list[str]:
    return [url.strip() for url in value.split(",") if url.strip()]


def _with_scheme(url: str) -> str:
    return url if "://" in url else f"http://{url}"
