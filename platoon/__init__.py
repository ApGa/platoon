"""platoon: RL for LLM agents through the Tinker API.

The public API, re-exported lazily so ``import platoon`` stays cheap::

    import platoon as pt

    @pt.harness("my-agent")                      # a function harness (platoon.harness.function)
    async def my_agent(ctx: pt.AgentContext) -> pt.AgentResult: ...

    @pt.program("my-program")                    # an episode program (platoon.programs)
    async def my_program(ep: pt.EpisodeContext) -> pt.Outcome: ...

    MY_ENV = pt.EnvironmentEntry("my-env", service="python://my_pkg.env:MyEnv")   # a named environment

Installed distributions provide these to configs by name through entry points (``platoon.extensions``); each ships
packages of its own, outside ``platoon``.  ``pt.gym`` adapts step environments.  ``pt.__version__`` is the running
platoon's version: a source checkout's ``pyproject.toml``, else the installed distribution's.
"""

from typing import TYPE_CHECKING, Any

# Public name -> the module that defines it.
_EXPORTS = {
    **dict.fromkeys(
        (
            "AgentAPI",
            "AgentContext",
            "AgentFn",
            "AgentResult",
            "AgentStatus",
            "Budget",
            "CallKind",
            "ChatReply",
            "McpEndpoint",
            "ModelEndpoint",
            "SubagentResult",
            "ToolAccess",
        ),
        "platoon.harness.agent",
    ),
    "EnvironmentEntry": "platoon.envs.named",
    "FunctionHarness": "platoon.harness.function",
    "PlatoonDeprecationWarning": "platoon.deprecations",
    "get_harness": "platoon.harness.registry",
    **dict.fromkeys(("EpisodeContext", "EpisodeProgram", "program"), "platoon.programs"),
    **dict.fromkeys(
        ("EnvSession", "EnvironmentService", "ForkMode", "Outcome", "SessionNeeds", "ToolResult", "ToolSpec"),
        "platoon.envs.service",
    ),
    "Task": "platoon.task",
    **dict.fromkeys(("Trajectory", "TrajectoryCollection"), "platoon.trajectory"),
}
# Public submodules: ``pt.harness`` (callable: the @harness decorator) and ``pt.gym``.
_SUBMODULES = ("harness", "gym")

__all__ = [
    "AgentAPI",
    "AgentContext",
    "AgentFn",
    "AgentResult",
    "AgentStatus",
    "Budget",
    "CallKind",
    "ChatReply",
    "EnvSession",
    "EnvironmentEntry",
    "EnvironmentService",
    "EpisodeContext",
    "EpisodeProgram",
    "ForkMode",
    "FunctionHarness",
    "McpEndpoint",
    "ModelEndpoint",
    "Outcome",
    "PlatoonDeprecationWarning",
    "SessionNeeds",
    "SubagentResult",
    "Task",
    "ToolAccess",
    "ToolResult",
    "ToolSpec",
    "Trajectory",
    "TrajectoryCollection",
    "get_harness",
    "gym",
    "harness",
    "program",
]


def __getattr__(name: str) -> Any:
    import importlib

    if name == "__version__":
        from platoon.utils.config import distribution_version

        version = globals()["__version__"] = distribution_version("platoon") or "0+unknown"
        return version
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    __version__: str

    from platoon import gym
    from platoon.deprecations import PlatoonDeprecationWarning
    from platoon.envs.named import EnvironmentEntry
    from platoon.envs.service import (
        EnvironmentService,
        EnvSession,
        ForkMode,
        Outcome,
        SessionNeeds,
        ToolResult,
        ToolSpec,
    )
    from platoon.harness.agent import (
        AgentAPI,
        AgentContext,
        AgentFn,
        AgentResult,
        AgentStatus,
        Budget,
        CallKind,
        ChatReply,
        McpEndpoint,
        ModelEndpoint,
        SubagentResult,
        ToolAccess,
    )
    from platoon.harness.function import FunctionHarness
    from platoon.harness.function import harness as harness
    from platoon.harness.registry import get_harness
    from platoon.programs import EpisodeContext, EpisodeProgram, program
    from platoon.task import Task
    from platoon.trajectory import Trajectory, TrajectoryCollection
