"""The Tinker service's refusal of ``TINKER_API_KEY``, as one line instead of the client's traceback.

A ``tinker.ServiceClient`` opens its session as it is built: the first request a process makes of the service, and
where a key the service does not take is refused (HTTP 401).  A key it takes may still be refused a request (HTTP
403), as a self-hosted service refuses a checkpoint that another key saved.  Every later request of the run would be
refused alike, so where platoon opens the service (:func:`platoon.train.tinker.proxy.register_tinker_llm`, which the
trainer, ``platoon eval`` and ``platoon rollout`` sample through, and the trainer's training client) it turns either
answer into :class:`~platoon.eval.config_defs.TinkerKeyRefusedError` (:func:`key_refusals`): a configuration error
(exit 78) whose message names the key and the service.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Mapping
from typing import TYPE_CHECKING, Any

import tinker

if TYPE_CHECKING:
    from platoon.eval.config_defs import TinkerKeyRefusedError

__all__ = ["KEY_REFUSALS", "key_refusal", "key_refusals"]

#: The service's answers that refuse the key: one it does not take (401), and one that may not do what was asked (403).
KEY_REFUSALS = frozenset({401, 403})
# Where a JSON answer says why, in the shapes services give it (FastAPI's detail, an error object or string).
_DETAIL_FIELDS = ("detail", "error", "message")


def key_refusal(exc: BaseException, *, asked: str | None = None) -> TinkerKeyRefusedError | None:
    """``exc``, which a Tinker client raised, as the service's refusal of ``TINKER_API_KEY`` (HTTP 401 or 403) when
    it is one, naming the service it came from and ``asked`` (what was asked for, unless a session); None for any
    other error.  The Tinker SDK raises a ``ValueError`` for a request whose result it could not retrieve, with the
    service's answer as its cause."""

    error: BaseException = exc
    if isinstance(error, ValueError) and isinstance(error.__cause__, tinker.TinkerError):
        error = error.__cause__
    if not isinstance(error, tinker.APIStatusError) or error.status_code not in KEY_REFUSALS:
        return None
    # The key's rules, imported only for a refusal: the processes that sample through this module (rollout workers
    # among them) need not import the evaluation package.
    from platoon.eval.config_defs import tinker_key_refused

    return tinker_key_refused(error.status_code, _detail(error), service=_service(error), asked=asked)


@contextlib.contextmanager
def key_refusals(asked: str | None = None) -> Generator[None]:
    """Raise :class:`~platoon.eval.config_defs.TinkerKeyRefusedError` for the service's refusal of the key in the
    block (:func:`key_refusal`; ``asked`` is what the block asks the service for, unless a session), and anything
    else the block raises as it is."""

    try:
        yield
    except Exception as exc:
        refusal = key_refusal(exc, asked=asked)
        if refusal is None:
            raise
        raise refusal from exc


def _detail(error: tinker.APIStatusError) -> str:
    """Why the service refused, from its answer: a JSON answer's own words, else its text."""

    body: Any = error.body
    if isinstance(body, Mapping):
        for name in _DETAIL_FIELDS:
            value = body.get(name)
            if isinstance(value, Mapping):
                value = value.get("message")
            if isinstance(value, str) and value.strip():
                return value
        return str(dict(body)) if body else ""
    if isinstance(body, str):
        return body
    return ""


def _service(error: tinker.APIStatusError) -> str | None:
    """The origin (scheme, host and port) of the service that refused, from the request it answered."""

    from platoon.gateway.hosted import url_origin

    request = getattr(error, "request", None)
    return url_origin(str(request.url)) if request is not None else None
