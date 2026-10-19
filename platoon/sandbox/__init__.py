"""Task sandboxes: the provider interface and the built-in process, docker, singularity and enroot providers.

The singularity and enroot providers adapt the ``hpc-sandbox`` package (the ``hpc`` extra), imported when one is
built.  :mod:`platoon.sandbox.routes` says how a sandbox reaches a service on the host (``SandboxSpec.upstreams``
and ``Sandbox.routes``), and :mod:`platoon.sandbox.cleanup` how the sandboxes a process leaves behind are reaped.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from platoon.registry import import_from_string
from platoon.sandbox.base import (
    CWD_ERROR_EXIT_CODE,
    CWD_ERROR_PREFIX,
    CWD_ERROR_REASONS,
    DownloadReport,
    DroppedLink,
    ExecResult,
    Mount,
    Sandbox,
    SandboxProvider,
    SandboxSnapshot,
    SandboxSpec,
    SupportsReap,
    cwd_error,
)

_BUILTIN_PROVIDERS = {
    "process": "platoon.sandbox.process:ProcessProvider",
    "docker": "platoon.sandbox.docker:DockerProvider",
    "singularity": "platoon.sandbox.singularity:SingularityProvider",
    "enroot": "platoon.sandbox.enroot:EnrootProvider",
}


def get_provider(name: str, **options: Any) -> SandboxProvider:
    """Build a sandbox provider from a built-in name or a ``module:Class`` path."""

    return provider_class(name)(**options)


def provider_class(name: str) -> Any:
    """The provider class (or factory) a built-in name or a ``module:Class`` path names, imported without building
    it: the hpc-sandbox package, for one, is imported only when a provider is built."""

    return import_from_string(_BUILTIN_PROVIDERS.get(name, name))


def network_none_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why ``provider`` (a name with its ``options``, or a built provider) would refuse every sandbox without a
    network, read from its options without building it; None when it isolates such sandboxes, runs them on the host
    network as the ``process`` provider does, or cannot tell.  A provider class tells through a
    ``network_none_refusal(options)`` class method."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
    else:
        kind, options = type(provider), getattr(provider, "options", None)
    refusal = getattr(kind, "network_none_refusal", None)
    return refusal(dict(options or {})) if callable(refusal) else None


def provider_unavailable(provider: str, options: Mapping[str, Any] | None = None) -> str | None:
    """Why this host cannot run the provider ``provider`` names with ``options`` (no runtime executable, no package),
    read without building it; None when it can, or cannot tell.  A provider class tells through an
    ``unavailable(options)`` class method; one that does not import says so here."""

    try:
        kind = provider_class(provider)
    except (ImportError, AttributeError, ValueError) as exc:
        return f"{provider!r} does not import ({type(exc).__name__}: {exc})"
    unavailable = getattr(kind, "unavailable", None)
    return unavailable(dict(options or {})) if callable(unavailable) else None


def rootfs_layout_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why ``provider`` (a name with its ``options``, or a built provider) could not give its sandboxes the root
    filesystem layout they ask for (the singularity provider's ``rootfs_layout``), read from its options without
    building it; None when nothing stops it, or when it cannot tell.  A provider class tells through a
    ``rootfs_layout_refusal(options)`` class method."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
    else:
        kind, options = type(provider), getattr(provider, "options", None)
    refusal = getattr(kind, "rootfs_layout_refusal", None)
    return refusal(dict(options or {})) if callable(refusal) else None


__all__ = [
    "CWD_ERROR_EXIT_CODE",
    "CWD_ERROR_PREFIX",
    "CWD_ERROR_REASONS",
    "DownloadReport",
    "DroppedLink",
    "ExecResult",
    "Mount",
    "Sandbox",
    "SandboxProvider",
    "SandboxSnapshot",
    "SandboxSpec",
    "SupportsReap",
    "cwd_error",
    "get_provider",
    "network_none_refusal",
    "provider_class",
    "provider_unavailable",
    "rootfs_layout_refusal",
]
