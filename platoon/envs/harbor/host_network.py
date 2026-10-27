"""Which Harbor stacks the host-network fallback runs (``stacks: host_network``), and why it refuses the others.

The fallback (:mod:`platoon.sandbox.host_network_stacks`) runs each service of a stack as a single sandbox on this
host's network, where every process of the node reaches its ports and a service's name resolves to nothing.  So it runs
a stack only as a dataset's overlays make it safe there, service by service (``host_network = {listen, token, bind}``
in a ``[[compose]]`` entry, a :class:`~platoon.envs.harbor.compose.HostNetworkRule`, which the plan records with the
service), and :func:`host_network_problem` refuses the rest, when a run lists its tasks and when a task loads:

- a sidecar that declares ports and that no overlay authenticates (no ``token``), and a declared TCP port without a
  ``listen`` rule; a declared UDP port, which the fallback gives no port of its own;
- a stack that names its services where the fallback cannot rewrite them.  A fetch scans each stack for such
  references (:func:`with_unrewritable_references`), in the task's text files and where the fallback rewrites nothing,
  and where it rewrites (environment values, commands, healthchecks, ``[verifier].env``, MCP servers' URLs, collect
  hooks), those it cannot rewrite; the plan records them (:data:`UNREWRITABLE_REFERENCES`), and an overlay edits them
  away.

A token rule is a promise the overlay makes, that the service refuses a request without the token: nothing checks it.
A task's MCP server that is a sidecar cannot keep it, since an MCP client sends no token (a task's MCP server carries no
headers), so :func:`mcp_problem` refuses it.
"""

from __future__ import annotations

import codecs
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from platoon.envs.harbor.compose import COMPOSE_FILE, ENVIRONMENT_DIR, MAIN, ComposePlan, declared_ports
from platoon.envs.harbor.hashing import task_files
from platoon.sandbox.stacks import LOOPBACK_HOSTS, service_references

#: Where a stack's plan records the references to its services that the fallback could not rewrite, from a fetch's
#: scan: ``{"first": ["<where>: <reference>", ...], "count": n}`` (the first few, and how many in all).
UNREWRITABLE_REFERENCES = "unrewritable_references"
_REFERENCES_RECORDED = 10
# A task file is read a piece at a time, each with the start of the next (longer than any reference); one with a NUL
# byte in its first bytes is binary, as git tells, and not read.
_SCAN_CHUNK = 1 << 20
_SCAN_OVERLAP = 1024
_BINARY_PROBE = 8192


def host_network_problem(plan: ComposePlan) -> str | None:
    """Why the host-network fallback cannot run a stack plan, as the end of a sentence that names the stack; None when
    it can.  Every sidecar that declares ports needs an overlay's token for it (on the host network every process of
    the node reaches them), every declared TCP port a listen rule, no port may be UDP, and the plan must have been
    scanned for references to its services that the fallback cannot rewrite, and hold none."""

    rules = plan.host_network_rules
    for name, service in plan.services.items():
        if service.get("external"):
            continue
        if other := [port for port in service.get("expose") or () if not port.endswith("/tcp")]:
            return (
                f"whose service {name} declares the port {other[0]}, which the host-network fallback gives no port of "
                "its own (the stacks of a task on one node would share it): an overlay's [[compose]] entry replaces "
                "its expose without it, or the stack runs as a pod or a compose project"
            )
        rule = rules.get(name)
        if name != MAIN and declared_ports(service) and (rule is None or rule.token is None):
            return (
                f"whose sidecar {name} no overlay authenticates: on the host network every process of the node reaches "
                "its ports, so an overlay's [[compose]] entry for it gives it the stack's token "
                '(host_network = {token = "VARIABLE"}, with bind = "VARIABLE" where a variable sets the address it '
                "listens on), and the service refuses a request without it"
            )
        listen = rule.listen if rule is not None else {}
        if missing := [port for port in declared_ports(service) if port not in listen]:
            return (
                f"whose service {name} declares port {missing[0]} and no listen rule: on the host network it listens "
                "on a port of this host that its stack holds, which an overlay's [[compose]] entry names the variable "
                f'that sets (host_network = {{listen = {{{missing[0]} = "VARIABLE"}}}})'
            )
    scan = plan.data.get(UNREWRITABLE_REFERENCES)
    if not isinstance(scan, Mapping):
        return (
            "whose plan was not scanned for the references to its services that the host-network fallback cannot "
            "rewrite: fetch the dataset again"
        )
    first = list(scan.get("first") or ())
    if count := int(scan.get("count") or 0):
        more = f"; and {count - len(first)} more" if count > len(first) else ""
        return (
            "that names its services where the host-network fallback cannot rewrite them "
            f"({'; '.join(first)}{more}): it rewrites a service's name at a port the service declares in "
            "environment values, commands, healthchecks, [verifier].env, MCP servers' URLs and collect hooks, so an "
            "overlay edits the files ([[edit]]) to read the address from an environment variable"
        )
    return None


def mcp_problem(plan: ComposePlan, urls: Mapping[str, str]) -> str | None:
    """Why the host-network fallback cannot run the stack for the task's MCP servers (``urls``: each one's URL, by
    its name), as the end of a sentence that names the stack: one that is a sidecar, which an MCP client cannot
    authenticate (it sends no token: a task's MCP server carries no headers), so the sidecar would answer every process
    of the node; None when there is none."""

    references = _References(plan.services)
    main = references.networks.get(MAIN)
    sidecars = frozenset(host for host, network in references.hosts.items() if network != main)
    for server, url in urls.items():
        if (named := next(service_references(url, sidecars), None)) is not None:
            return (
                f"whose MCP server {server} is its sidecar {named.host} ({url}): an MCP client sends no token (a "
                "task's MCP server carries no headers yet), so on the host network the sidecar would answer every "
                "process of the node"
            )
    return None


def with_unrewritable_references(plan: ComposePlan | None, task_dir: Path) -> ComposePlan | None:
    """``plan`` (the stack plan of the task at ``task_dir``, as a fetch makes it) with the references to its services
    that the fallback could not rewrite (:func:`unrewritable_references`) recorded; another plan as it is."""

    if plan is None or plan.kind != "sidecars":
        return plan
    return ComposePlan({**plan.data, UNREWRITABLE_REFERENCES: unrewritable_references(task_dir, plan.services)})


def unrewritable_references(task_dir: Path, services: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The references to a stack's services (``[scheme://]<service>:<port>``, ``scheme://<service>``, and a loopback
    address at a port a service declares) that the host-network fallback could not rewrite, as
    :data:`UNREWRITABLE_REFERENCES` records them: those in the task's files, but for its compose file and
    ``task.toml``; those in the parts of ``task.toml`` and of the plan where it rewrites nothing (``main``'s command,
    which starts as its init); and those it cannot rewrite where it does (the services' environment values, commands
    and healthchecks, ``[environment].env``, ``[verifier].env``, the task's MCP servers' URLs and its collect hooks): a
    service named without a port, or at a port it does not declare, and another service's port on a service's own
    loopback."""

    references = _References(services)
    for relative in task_files(task_dir):
        if relative not in (f"{ENVIRONMENT_DIR}/{COMPOSE_FILE}", "task.toml"):
            for text, until in _texts(task_dir / relative):
                references.look(text, relative, service=None, until=until, prose=True)
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        config = {}  # the task is refused for it
    hooks = (config.get("verifier") or {}).get("collect") or []
    for path, text in _toml_strings(config, ()):
        references.look(text, f"task.toml {'.'.join(map(str, path))}", service=_rewritten_in(path, hooks))
    for name, service in services.items():
        if service.get("external"):
            continue
        for key, value in (service.get("environment") or {}).items():
            references.look(str(value), f"compose {name}.environment.{key}", service=name)
        for arg in (service.get("healthcheck") or {}).get("test") or ():
            references.look(str(arg), f"compose {name}.healthcheck", service=name)
        for key in ("entrypoint", "command"):
            for arg in service.get(key) or ():  # main's runs as its init, which nothing rewrites
                references.look(str(arg), f"compose {name}.{key}", service=None if name == MAIN else name)
    found = list(references.found)
    return {"first": found[:_REFERENCES_RECORDED], "count": len(found)}


class _References:
    """The references to a stack's services that the host-network fallback could not rewrite, as they are found."""

    def __init__(self, services: Mapping[str, Mapping[str, Any]]) -> None:
        started = {name: service for name, service in services.items() if not service.get("external")}
        self.networks: dict[str, str] = {}  # each service's network: its own, or the one it shares
        for name in started:
            owner, seen = name, {name}
            while (mode := str(started.get(owner, {}).get("network_mode") or "")).startswith("service:"):
                owner = mode.removeprefix("service:")
                if owner in seen:
                    break
                seen.add(owner)
            self.networks[name] = owner
        self.hosts: dict[str, str] = {}  # each name and alias, in lower case, with its network
        self.declared: dict[str, set[int]] = {}  # each network's declared ports
        for name, service in started.items():
            aliases = [name, service.get("hostname") or ""]
            aliases += [
                alias
                for settings in (service.get("networks") or {}).values()
                for alias in settings.get("aliases") or ()
            ]
            for alias in filter(None, aliases):
                self.hosts.setdefault(alias.lower(), self.networks[name])
            self.declared.setdefault(self.networks[name], set()).update(declared_ports(service))
        self.ports = {port for ports in self.declared.values() for port in ports}
        self.lookup = frozenset({*self.hosts, *LOOPBACK_HOSTS})
        self.found: dict[str, None] = {}  # "<where>: <reference>", in the order found

    def look(
        self, text: str, where: str, *, service: str | None, until: int | None = None, prose: bool = False
    ) -> None:
        """Note the references in ``text`` (at ``where``) that the fallback leaves as they are: each one where it
        rewrites nothing (``service`` None), else those it cannot rewrite in ``service``'s environment.  References
        that start at ``until`` or later are left for the next piece of the text.  In ``prose`` (a task's files, where
        a service's name may be a word), one without a scheme counts only at a port the service declares."""

        own = self.networks.get(service) if service is not None else None
        for reference in service_references(text, self.lookup):
            if until is not None and reference.start >= until:
                break
            if prose and not reference.scheme and reference.host not in LOOPBACK_HOSTS:
                if reference.port not in self.declared[self.hosts[reference.host]]:
                    continue  # "kv:1" in a text: no address the stack's
            if reference.host in LOOPBACK_HOSTS:
                if reference.port not in self.ports:
                    continue  # not a port of the stack's: the service's own business
                if own is not None and reference.port in self.declared[own]:
                    continue
            elif own is not None and reference.port in self.declared[self.hosts[reference.host]]:
                continue
            self.found.setdefault(f"{where}: {reference.text}")


def _texts(path: Path) -> Iterator[tuple[str, int]]:
    """``path``'s contents as text (undecodable bytes replaced), a piece at a time, each with where the next piece
    begins: a piece ends with the start of the next one, so a reference across the two is found whole in one.  Nothing
    for a binary file (a NUL byte in its first 8 KiB)."""

    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    carried = ""
    try:
        with path.open("rb") as handle:
            if b"\0" in handle.read(_BINARY_PROBE):
                return
            handle.seek(0)
            while True:
                chunk = handle.read(_SCAN_CHUNK)
                text = carried + decoder.decode(chunk, final=not chunk)
                if not chunk:
                    yield text, len(text)
                    return
                yield text, max(0, len(text) - _SCAN_OVERLAP)
                carried = text[-_SCAN_OVERLAP:]
    except OSError:
        return


def _toml_strings(value: Any, path: tuple[Any, ...]) -> Iterator[tuple[tuple[Any, ...], str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _toml_strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _toml_strings(item, (*path, index))
    elif isinstance(value, str):
        yield path, value


def _rewritten_in(path: tuple[Any, ...], hooks: Sequence[Any]) -> str | None:
    """The service in whose environment a stack rewrites the ``task.toml`` value at ``path``, or None where nothing
    rewrites it: ``[environment].env`` and ``[verifier].env`` and MCP servers' URLs in ``main``'s, a collect hook's
    command in its service's."""

    match path:
        case ("environment" | "verifier", "env", _):
            return MAIN
        case ("environment", "mcp_servers", int(), "url"):
            return MAIN
        case ("verifier", "collect", int() as index, "command") if index < len(hooks) and isinstance(
            hooks[index], dict
        ):
            return str(hooks[index].get("service") or MAIN)
    return None


__all__ = [
    "UNREWRITABLE_REFERENCES",
    "host_network_problem",
    "mcp_problem",
    "unrewritable_references",
    "with_unrewritable_references",
]
