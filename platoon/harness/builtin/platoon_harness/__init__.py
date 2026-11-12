"""The ``platoon-harness`` harness and episode program: a team of agents that platoon-harness runs, rolled out and
trained as one platoon episode.

Importing this package registers both, and imports nothing of platoon-harness, which the ``platoon-harness`` extra
installs: the harness, the program and the program's load check import :mod:`.bridge` when they run, and without the
extra they raise naming it.  :mod:`.bridge` runs the team, :mod:`.authority` admits each of its agents to platoon's
gateway, and :mod:`.finalize` marks what of a rollout trains.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any

from platoon.harness.function import harness
from platoon.programs import program

if TYPE_CHECKING:
    from platoon.envs.service import Outcome
    from platoon.harness.agent import AgentContext, AgentResult
    from platoon.options import CheckContext
    from platoon.programs import EpisodeContext

# The name of both the harness and the program.
NAME = "platoon-harness"
INSTALL = (
    "the platoon-harness harness and program need the platoon-harness package: install the platoon-harness extra, "
    "e.g. pip install 'platoon[platoon-harness]', or in a platoon checkout uv sync --extra tinker --extra ors --extra "
    "platoon-harness (uv sync removes the extras it is not given)"
)


def load_bridge() -> ModuleType:
    """:mod:`.bridge`, or an ImportError that says how to install platoon-harness."""

    try:
        import platoon_harness  # noqa: F401
    except ImportError as exc:
        raise ImportError(INSTALL) from exc
    from . import bridge

    return bridge


def check(options: Mapping[str, Any], ctx: CheckContext) -> None:
    """The program's load check (:func:`.bridge.check_entry`); without platoon-harness, a refusal that names the
    extra."""

    try:
        found = load_bridge()
    except ImportError as exc:
        raise ValueError(str(exc)) from exc
    found.check_entry(options, ctx)


@harness(NAME, executes_model_code=True, budget_enforcement="gateway", path_options=("team",))
async def team_harness(ctx: AgentContext) -> AgentResult:
    """Run a team of agents that platoon-harness runs, each on a platoon agent of its own: as the root, the whole
    session; as a member, the one agent it was spawned for.  It runs under ``program: platoon-harness`` alone, and its
    options are ``team``, a team file or a trusted ``module:attr`` naming a ``Team``, and ``rewards``, the producer
    each role trains on."""

    return await load_bridge().run_agent(ctx)


@program(NAME, check=check)
async def team_program(ep: EpisodeContext) -> Outcome:
    """Run a team as one episode: ``single``'s finalizers and the team's, the root in an empty repository, and
    platoon's grades and the producers ``rewards`` names, each written into the ended session."""

    return await load_bridge().run_episode(ep)
