"""Task sandboxes: the provider interface and the built-in process, docker, singularity and enroot providers.

The singularity and enroot providers adapt the ``hpc-sandbox`` package (the ``hpc`` extra), imported when one is
built.  :mod:`platoon.sandbox.routes` says how a sandbox reaches a service on the host (``SandboxSpec.upstreams``
and ``Sandbox.routes``), :mod:`platoon.sandbox.cleanup` how the sandboxes a process leaves behind are reaped, and
:mod:`platoon.sandbox.stacks` how a provider with the ``stacks`` capability runs a task's services together.
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
from platoon.sandbox.stacks import (
    EXTERNALS,
    STACKS,
    Dependency,
    Healthcheck,
    ServiceSpec,
    ServiceState,
    Stack,
    StackProvider,
    StackSpec,
    StackStartError,
    Volume,
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


def stack_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why ``provider`` (a name with its ``options``, or a built provider) would start no stack, a task's services
    together (:mod:`platoon.sandbox.stacks`), read without building it; None when it starts stacks, or cannot tell
    before it is built.  A built provider tells by its capabilities (``stacks``, with a ``start_stack`` method), then
    as its class does, which also says why not where it has one; a provider class by a ``stack_refusal(options)``
    class method (the ``docker`` provider's says whether its CLI has the compose plugin, an hpc-sandbox provider's
    whether its pods run here), else by capabilities it declares on the class."""

    if not isinstance(provider, str):
        refusal = getattr(type(provider), "stack_refusal", None)
        reason = refusal(dict(getattr(provider, "options", None) or {})) if callable(refusal) else None
        if STACKS not in provider.capabilities or not callable(getattr(provider, "start_stack", None)):
            return reason or f"the {provider.name!r} sandbox provider runs no multi-container stacks"
        return reason
    try:
        kind = provider_class(provider)
    except (ImportError, AttributeError, ValueError):
        return None  # building the provider reports it
    refusal = getattr(kind, "stack_refusal", None)
    if callable(refusal):
        return refusal(dict(options or {}))
    capabilities = getattr(kind, "capabilities", None)
    if not isinstance(capabilities, frozenset | set):
        return None  # set when it is built: the session's opening checks it
    if STACKS in capabilities and callable(getattr(kind, "start_stack", None)):
        return None
    return f"the {provider!r} sandbox provider runs no multi-container stacks"


def externals_refusal(provider: str | SandboxProvider) -> str | None:
    """Why the stacks of ``provider`` (a name, or a built provider) would not reach services outside them (a stack
    spec's ``externals``, node services): it does not declare the ``externals`` capability, as a provider written before
    externals existed does not, whose stacks would start without them.  None when it declares it, and when it cannot
    tell before it is built (an hpc-sandbox provider's capabilities: its stacks refuse what they cannot reach)."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
        name, capabilities = provider, getattr(kind, "capabilities", None)
    else:
        name, capabilities = provider.name, provider.capabilities
    if not isinstance(capabilities, frozenset | set) or EXTERNALS in capabilities:
        return None
    return (
        f"the stacks of the {name!r} sandbox provider reach no service outside them (it lacks the {EXTERNALS!r} "
        "capability)"
    )


def privilege_refusal(provider: str | SandboxProvider) -> str | None:
    """Why ``provider`` (a name, or a built provider) would not give a stack's services the privileges a Harbor task
    fetched with ``refuse: {privileges: false}`` may ask for: a privileged container, or a capability that a rootless
    service does not hold over the stack (``NET_ADMIN``, ...); None when it declares the ``privileged`` capability
    (``docker``).  Read from the class without building it; a provider that does not import grants none."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return f"the {provider!r} sandbox provider does not import"
        name, capabilities = provider, getattr(kind, "capabilities", None)
    else:
        name, capabilities = provider.name, provider.capabilities
    if isinstance(capabilities, frozenset | set) and "privileged" in capabilities:
        return None
    return f"the {name!r} sandbox provider gives no service such privileges (the docker provider does)"


def root_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why ``provider`` (a name with its ``options``, or a built provider) could not run a sandbox's commands as root
    with other users to switch to, as a Harbor verifier that runs as root needs (it may run the agent's code as
    another user), read from its options without building it; None when it can, or cannot tell.  A provider class
    tells through a ``root_refusal(options)`` class method; where it cannot tell (the image's own user decides, as
    with ``docker``), the Harbor service checks the started sandbox."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
    else:
        kind, options = type(provider), getattr(provider, "options", None)
    refusal = getattr(kind, "root_refusal", None)
    return refusal(dict(options or {})) if callable(refusal) else None


def loses_file_owners(provider: str | SandboxProvider) -> bool:
    """Whether ``provider`` (a name, or a built provider) gives its sandboxes' files other owners than their image does,
    as hpc-sandbox, which extracts every image as one user: their files then keep the image's owners only through its
    owner manifest (``SandboxSpec.file_owners``).  A provider class says so with ``loses_file_owners = True``; False
    for one that does not, or that cannot be imported (building it reports that)."""

    if isinstance(provider, str):
        try:
            provider = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return False
    return bool(getattr(provider, "loses_file_owners", False))


def file_owners_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why ``provider`` (a name with its ``options``, or a built provider) could not give a sandbox's files the owners
    its image's manifest lists (``SandboxSpec.file_owners``), read without building it; None when it can, or when it
    cannot tell.  A provider class tells through a ``file_owners_refusal(options)`` class method."""

    if isinstance(provider, str):
        try:
            kind = provider_class(provider)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
    else:
        kind, options = type(provider), getattr(provider, "options", None)
    refusal = getattr(kind, "file_owners_refusal", None)
    return refusal(dict(options or {})) if callable(refusal) else None


__all__ = [
    "CWD_ERROR_EXIT_CODE",
    "CWD_ERROR_PREFIX",
    "CWD_ERROR_REASONS",
    "EXTERNALS",
    "STACKS",
    "Dependency",
    "DownloadReport",
    "DroppedLink",
    "ExecResult",
    "Healthcheck",
    "Mount",
    "Sandbox",
    "SandboxProvider",
    "SandboxSnapshot",
    "SandboxSpec",
    "ServiceSpec",
    "ServiceState",
    "Stack",
    "StackProvider",
    "StackSpec",
    "StackStartError",
    "SupportsReap",
    "Volume",
    "cwd_error",
    "externals_refusal",
    "file_owners_refusal",
    "get_provider",
    "loses_file_owners",
    "network_none_refusal",
    "privilege_refusal",
    "provider_class",
    "provider_unavailable",
    "root_refusal",
    "rootfs_layout_refusal",
    "stack_refusal",
]
