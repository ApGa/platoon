"""Named hooks through which extensions plug into platoon core.

Core never imports an extension.  At a few fixed seams it calls the functions
registered under a hook name instead, and an extension registers its functions
when its modules are imported.  Configs name extension components by module
path, so the trainer and every rollout worker import those modules -- and so
register the same hooks -- before any seam runs.

Hook names and the signatures core calls them with:

``rollout_worker.telemetry``
    ``fn() -> Mapping[str, float]``: counters of the current process that every
    rollout host reports (for example, a code-execution guard's counters).
``rollout_worker.diagnostics``
    ``fn() -> Mapping[str, str]``: short strings written into a rollout worker's
    diagnostic marker, so a worker killed for hanging still says why (for
    example, the source of an overdue code cell).
``trajectory.error_completion_ids``
    ``fn(steps) -> set[str]``: ids of the sampled completions whose actions
    failed, on top of the generic detection in
    :mod:`platoon.utils.trajectory_error_filtering` (for example, a harness's
    own error events).
``trajectory.message_completion_ids``
    ``fn(steps) -> set[str]``: ids of the sampled completions that only sent a
    message and ran no action (for example, a harness's message-only turns).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["hooks", "register"]

_HOOKS: dict[str, list[Callable[..., Any]]] = {}


def register(name: str, fn: Callable[..., Any]) -> None:
    """Register ``fn`` under ``name``.

    Registering the same function twice keeps one registration, so importing a
    module again (``importlib.reload``, a module loaded under two names) does
    not double its effect.
    """

    registered = _HOOKS.setdefault(name, [])
    if fn not in registered:
        registered.append(fn)


def hooks(name: str) -> list[Callable[..., Any]]:
    """Return the functions registered under ``name`` in registration order."""

    return list(_HOOKS.get(name, ()))
