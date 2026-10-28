"""Which tools an agent may see and call, by its role: ``agents.roles.<role>.tools`` (experimental).

A run's config may give an agent role a list of tool names, each a name or an ``fnmatch`` pattern (``read_*``),
matched case-sensitively against the names agents see: an env's tools, and those the tool plane serves from a task's
MCP servers (``<server>__<tool>``, :func:`platoon.envs.harbor.session.task_tool_name`).  The gateway holds the lists
by role (``ModelGateway.limit_tools``), and gives each agent it opens its role's (:func:`allowlists_of`; the root
agent's role is ``policy``) on top of its parent's: an agent under a limited one never sees more than its parent, as
with tool access.  A role without a list keeps every tool, and an empty list allows none.

The tool plane (:mod:`platoon.gateway.mcp`) applies them where it lists and where it runs an agent's calls, for MCP
clients and inline harnesses alike: a listing holds only the tools that every allowlist of the agent allows
(:func:`visible`), and a call to any other tool is refused unrun (:func:`refusal`): the agent gets an error result
that says why, the env never sees the call, and the agent's misc counts it under :data:`TOOLS_REFUSED_MISC_KEY`.

Standard library only, so that the gateway and the training config (whose loading warns of a pattern that matches no
tool, :func:`unmatched`) both read it.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from platoon.envs.service import ToolSpec
    from platoon.gateway.core import AgentSession

# Trajectory misc key of an agent's refused tool calls: ``{"count": <calls refused>, "first": <the first one's tool>}``.
TOOLS_REFUSED_MISC_KEY = "tools_refused"


@dataclass(frozen=True)
class Allowlist:
    """The tools the agents of ``role``, and the agents under them, may see and call: those a pattern matches."""

    role: str
    patterns: tuple[str, ...]

    def allows(self, name: str) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in self.patterns)


def allowlists_of(
    role: str, roles: Mapping[str, Sequence[str]], inherited: tuple[Allowlist, ...] = ()
) -> tuple[Allowlist, ...]:
    """The allowlists of an agent of ``role`` opened under an agent with ``inherited`` (none for a root agent): those,
    and its role's when ``roles`` lists one."""

    patterns = roles.get(role)
    return inherited if patterns is None else (*inherited, Allowlist(role, tuple(patterns)))


def visible(agent: AgentSession, tools: Iterable[ToolSpec]) -> list[ToolSpec]:
    """The tools of ``tools`` that every allowlist of ``agent`` allows."""

    return [tool for tool in tools if _refusing(agent, tool.name) is None]


def refusal(agent: AgentSession, name: str) -> str | None:
    """None when ``agent`` may call the tool ``name``; else why it may not, for the error result that refuses the call,
    which this counts in the agent's misc under :data:`TOOLS_REFUSED_MISC_KEY`."""

    allowlist = _refusing(agent, name)
    if allowlist is None:
        return None
    refused = agent.misc.setdefault(TOOLS_REFUSED_MISC_KEY, {"count": 0, "first": name})
    refused["count"] += 1
    whose = "agents of role" if allowlist.role == agent.role else "agents under an agent of role"
    allowed = f"only {', '.join(allowlist.patterns)}" if allowlist.patterns else "no tools"
    return f"Tool {name!r} is not allowed: {whose} {allowlist.role!r} may call {allowed}. The call was not run."


def unmatched(patterns: Iterable[str], names: Iterable[str]) -> list[str]:
    """The patterns of ``patterns`` that match none of the tool names ``names``."""

    names = list(names)
    return [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(name, pattern) for name in names)]


def _refusing(agent: AgentSession, name: str) -> Allowlist | None:
    """The first allowlist of ``agent`` that does not allow the tool ``name``; None when every one does."""

    return next((allowlist for allowlist in agent.tool_allowlists if not allowlist.allows(name)), None)
