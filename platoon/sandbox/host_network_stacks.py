"""The host-network fallback: stacks where pods cannot run, each service a single sandbox on this host's network
(RFC 0010, section 6).

A node that does not let an unprivileged user nest user namespaces runs no pod (:mod:`platoon.sandbox.hpc_stacks`).
:class:`HostNetworkStackProvider` runs a stack there all the same, over the single sandboxes of a provider such as
``singularity`` or ``enroot``, as the ``process`` provider runs its stacks (:mod:`platoon.sandbox.process_stacks`): each
service a sandbox of its own on this host's network, its command in the background there; each port a service declares
a port of this host that the stack holds for its life, which the service learns from its listen variables
(``PLATOON_PORT_<port>``, and those its spec's ``listen`` names); and the references to the services in their
environment values, commands and healthchecks rewritten to reach those ports, a service's own ``localhost:<port>``
included.

What a pod gives a stack and the host network does not, the fallback makes up for, or refuses:

- **A network of its own.**  On the host network every process of the node reaches a service's port, other
  rollouts' agents included, and so do other hosts where it listens on more than this host's loopback.  So each
  service must listen on ``127.0.0.1`` alone and take a token that only its own stack's services know: its caller says
  how each one takes them (Harbor's overlays, :mod:`platoon.envs.harbor.host_network`), and the fallback checks what
  listens at the ports a service declares (:meth:`_Checked.problem`): a listener there beyond the loopback, or one
  that another user's process holds (it bound the port first, the stacks' claims being visible to the node in
  ``/proc/net/unix``), makes the service unhealthy, so its stack does not start, and a sidecar's probe at grading makes
  the outcome invalid.  Nothing tells a process of your own uid from the service, another rollout's agent included,
  and a port a service listens at without declaring it is not looked at.  The token is the caller's to enforce:
  nothing checks that a service refuses a request without it.
- **No network.**  A stack whose network is ``none`` is refused, as is a switch to ``none``: its services have the
  host's, which the fallback cannot take away.
- **Node services.**  A stack that reaches a node service (``externals``) is refused: the relay to it would be a port
  of this host that every process of the node reaches.
- **Your own uid.**  The stack's volumes are directories of yours on this host, which its services write as your uid.
  A provider whose sandboxes run other users (``docker``, whose stacks run as compose projects anyway, and
  ``singularity`` with ``fakeroot``) is refused, since what those users wrote there your uid could not delete
  (:func:`host_network_refusal`).

The stacks' directories are below the provider's ``root`` (``platoon-stacks-<uid>`` in node scratch, a directory of
your uid's alone: :func:`~platoon.sandbox.process_stacks.private_root`), which it reaps by label; their services'
sandboxes are their own provider's to reap.  A Harbor environment selects the fallback with
``stacks: host_network``; nothing chooses it by itself, since its services are reachable from the node.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import sys
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platoon.sandbox.base import Sandbox, SandboxProvider
from platoon.sandbox.process_stacks import (
    HostBackend,
    HostService,
    reap_stack_directories,
    start_host_stack,
)
from platoon.sandbox.stacks import STACKS, DrivenStack, StackNetwork, StackSpec
from platoon.utils.scratch import node_scratch

#: Why the fallback runs no stack without a network.
NO_NETWORK = (
    "the host-network fallback runs a stack's services on this host's network, which it cannot take away: a stack "
    "that runs without a network (a task's no-network phase, internal networks only, network_policy: none) runs as a "
    "pod or a compose project"
)
#: Why the fallback runs no stack that reaches a node service.
NO_EXTERNALS = (
    "the host-network fallback relays a port of this host to what a stack reaches outside it (a node service), and "
    "every process of the node would reach that port: a stack that reaches a node service runs as a pod, or on the "
    "process provider"
)
_LISTEN = "0A"  # a socket's state in /proc/net/tcp: TCP_LISTEN
_TABLES = (("/proc/net/tcp", 4), ("/proc/net/tcp6", 6))  # this process's network namespace: the host's


class HostNetworkStackProvider:
    """Stacks on this host's network over ``provider``'s single sandboxes (the module docstring says how), their
    directories below ``root``.  Built with ``root`` alone, as reaping builds it, it only reaps those directories."""

    name = "host_network"
    capabilities = frozenset({STACKS})

    def __init__(self, provider: SandboxProvider | None = None, *, root: str | Path | None = None) -> None:
        self.provider = provider
        self.root = Path(root) if root is not None else node_scratch() / f"platoon-stacks-{os.getuid()}"

    async def start_stack(self, spec: StackSpec) -> DrivenStack:
        """Start ``spec``, each service a sandbox of the provider on this host's network; ``ValueError`` for a provider
        or a stack that the fallback cannot run."""

        if self.provider is None:
            raise ValueError("This host-network stack provider was built to reap stack directories: it has no provider")
        if (refusal := host_network_refusal(self.provider)) is not None:
            raise ValueError(refusal)
        isolated = [name for name, service in spec.services.items() if service.sandbox.network != "host"]
        if spec.network == "none" or isolated:
            raise ValueError(NO_NETWORK)
        if spec.externals:
            raise ValueError(NO_EXTERNALS)
        return await start_host_stack(self.provider, spec, root=self.root, reaper=self, backend_class=_Backend)

    async def reap(self, labels: Mapping[str, str]) -> int:
        """Delete the stack directories below ``root`` whose labels include ``labels`` and whose owner is this process
        or gone; returns how many."""

        if not labels:
            raise ValueError("Reaping needs labels: an empty set would select every stack")
        return await asyncio.to_thread(reap_stack_directories, self.root, dict(labels))


def host_network_refusal(provider: str | SandboxProvider, options: Mapping[str, Any] | None = None) -> str | None:
    """Why the host-network fallback would run no stack over ``provider`` (a name with its ``options``, or a built
    provider), read without building it; None when it runs them, or cannot tell before the provider is built.  Its
    sandboxes must share this host's network (the ``host_loopback`` capability), and run their commands as your own
    uid (its ``root_refusal`` says it gives no root that switches users)."""

    from platoon.sandbox import provider_class, root_refusal

    if isinstance(provider, str):
        name = provider
        try:
            capabilities = getattr(provider_class(provider), "capabilities", None)
        except (ImportError, AttributeError, ValueError):
            return None  # building the provider reports it
    else:
        name, capabilities = provider.name, provider.capabilities
    if isinstance(capabilities, frozenset | set) and "host_loopback" not in capabilities:
        return (
            f"the {name!r} sandbox provider's sandboxes do not share this host's network (it does not declare "
            "host_loopback), where the host-network fallback runs a stack's services"
        )
    if root_refusal(provider, options) is None:
        settings = options if isinstance(provider, str) else getattr(provider, "options", None)
        fix = (
            "run it without fakeroot, or run the stacks as pods (stacks: auto)"
            if (settings or {}).get("fakeroot")
            else "run its stacks with stacks: auto, as compose projects or pods"
        )
        return (
            f"the {name!r} sandbox provider runs a sandbox's commands as root that can switch users (or does not say "
            "otherwise), and the host-network fallback runs a stack's services as your own uid, whose volumes are "
            f"directories of yours that another user's files would stay in: {fix}"
        )
    return None


class _Backend(HostBackend):
    """A stack of the fallback: the ``process`` provider's way, with what listens at its services' ports checked, and
    no switch to ``none``."""

    def watch(self, name: str, sandbox: Sandbox, entry: str | None) -> HostService:
        return _Checked(sandbox, entry, self.addresses.own_ports(name))

    async def set_network(self, network: StackNetwork) -> None:
        if network == "none":
            raise ValueError(NO_NETWORK)


class _Checked(HostService):
    """A service of the fallback, unhealthy while something listens at one of its ports that must not."""

    async def problem(self) -> str | None:
        """Why the service must not serve as it does: a listener at one of its ports that another user's process holds
        (it took the port its stack holds for the service, and would get what the stack sends there, the token
        included), or one on an address other than this host's loopback (``0.0.0.0``, the node's own), which other
        hosts reach; None when there is none."""

        if not self.ports:
            return None
        uid = os.getuid()
        found = await asyncio.to_thread(listeners, set(self.ports.values()))
        for listener in sorted(found, key=lambda listener: (listener.uid == uid, listener.port)):  # others' first
            declared = next(declared for declared, allocated in self.ports.items() if allocated == listener.port)
            if listener.uid != uid:
                return (
                    f"a process of another user (uid {listener.uid}) listens at port {listener.port} (its port "
                    f"{declared}), which its stack holds for it: what its stack sends there, its token included, would "
                    "reach that process"
                )
            if not listener.loopback:
                return (
                    f"it listens on {listener.address} at port {listener.port} (its port {declared}), beyond this "
                    "host's loopback, where other hosts reach it: on the host network a service listens on 127.0.0.1 "
                    "alone"
                )
        return None


@dataclass(frozen=True)
class Listener:
    """A listening TCP socket of this host's network namespace: its port, its address, and the uid that owns it (as
    this process's user namespace sees it)."""

    port: int
    address: str
    uid: int

    @property
    def loopback(self) -> bool:
        """Whether only this host reaches it: its address is a loopback one (``127.0.0.1``, ``::1``, or
        ``::ffff:127.0.0.1``)."""

        ip = ipaddress.ip_address(self.address)
        mapped = ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) else None
        return ip.is_loopback or (mapped is not None and mapped.is_loopback)


def listeners(ports: Collection[int]) -> list[Listener]:
    """The listening TCP sockets of this host's network namespace at any of ``ports``, read from ``/proc/net/tcp``
    and ``tcp6``."""

    found = []
    for table, family in _TABLES:
        try:
            lines = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 8 or fields[3] != _LISTEN:
                continue
            address, _, port = fields[1].partition(":")
            if int(port, 16) in ports:
                found.append(Listener(int(port, 16), str(_address(address, family)), int(fields[7])))
    return sorted(found, key=lambda listener: (listener.port, listener.address))


def beyond_loopback(ports: Collection[int]) -> list[tuple[int, str]]:
    """The listening TCP sockets of this host's network namespace at any of ``ports`` whose address is not a loopback
    one (``0.0.0.0``, ``::``, the node's own addresses), as (port, address)."""

    return [(listener.port, listener.address) for listener in listeners(ports) if not listener.loopback]


def _address(text: str, family: int) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """An address as ``/proc/net/tcp`` writes it: hex, each 32-bit word in the host's byte order."""

    words = bytes.fromhex(text)
    if sys.byteorder == "little":
        words = b"".join(words[index : index + 4][::-1] for index in range(0, len(words), 4))
    return ipaddress.IPv4Address(words) if family == 4 else ipaddress.IPv6Address(words)


__all__ = [
    "NO_EXTERNALS",
    "NO_NETWORK",
    "HostNetworkStackProvider",
    "Listener",
    "beyond_loopback",
    "host_network_refusal",
    "listeners",
]
