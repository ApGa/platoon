"""Run one episode end to end on CPU: an environment, a harness and an episode program, with a scripted policy.

:func:`run_episode` takes the keys of one ``environments`` entry of a config (``service``, ``harness``, ``program``,
``agents``, ...) and runs one task through ``harness_rollout``, the path training and evaluation take: the env
session opens, the program runs the root agent and its subagents through a real model gateway (served over HTTP as
well, for harnesses that call it there), the session is graded, the agent tree becomes a trajectory collection, and
the program's finalizers run on it.  Only sampling is scripted.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from platoon.testing.policy import Script, ScriptedPolicy
from platoon.utils import scratch

if TYPE_CHECKING:
    from platoon.config_defs import RolloutConfig
    from platoon.gateway import ModelGateway
    from platoon.gateway.core import TinkerLLMInteraction
    from platoon.task import Task
    from platoon.trajectory import Trajectory, TrajectoryCollection


@dataclass
class EpisodeRun:
    """What one episode produced."""

    # Every agent's trajectory, by trajectory id, the root's first.
    collection: TrajectoryCollection
    # Every recorded model call's exact prompt and sampled tokens, by call id: what training data is built from.
    interactions: dict[str, TinkerLLMInteraction] = field(repr=False)
    # The gateway the episode sampled through; its sampler is the scripted policy.
    gateway: ModelGateway = field(repr=False)
    # Where the rollout wrote its files (harness logs of a command-line harness).
    output_dir: Path

    @property
    def trajectories(self) -> list[Trajectory]:
        """Every trajectory, the root's first."""
        return list(self.collection.trajectories.values())

    @property
    def root(self) -> Trajectory:
        """The root agent's trajectory."""
        return self.trajectories[0]

    @property
    def children(self) -> list[Trajectory]:
        """The subagents' trajectories (every trajectory but the root's)."""
        return self.trajectories[1:]

    @property
    def outcome(self) -> dict[str, Any]:
        """The episode's graded outcome, as recorded on the root: ``status``, ``rewards`` and ``reason``."""
        return self.root.misc["outcome"]

    @property
    def reward(self) -> float:
        """The root's reward."""
        return self.root.reward

    @property
    def policy(self) -> ScriptedPolicy:
        """The scripted policy the episode sampled."""
        sampler = self.gateway.sampler
        if not isinstance(sampler, ScriptedPolicy):
            raise TypeError(f"The episode's gateway samples {type(sampler).__name__}, not a ScriptedPolicy")
        return sampler


async def run_episode(
    task: Task | str,
    *,
    service: str,
    harness: str,
    policy: ScriptedPolicy | Script | ModelGateway = (),
    service_kwargs: Mapping[str, Any] | None = None,
    harness_options: Mapping[str, Any] | None = None,
    program: str = "single",
    agents: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
    rollout: RolloutConfig | Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> EpisodeRun:
    """Run ``task`` with ``harness`` on ``service`` through ``program``, sampling ``policy``.

    Args:
        task: A ``Task``, or a task id for the service to resolve, as ``platoon rollout --task`` does (an ORS
            environment's ids are ``<split>:<index>``, such as ``val:0``).
        service: The environment service, as an environment entry names it: ``python://pkg.module:EnvClass``
            for an ORS environment class, ``module:ServiceClass`` for an ``EnvironmentService``, and so on.
        harness: The harness, as an environment entry names it (``platoon-react``, ``my-agent@2``, a spec path).
        policy: What the model writes: a ``ScriptedPolicy``, a script to wrap in one, or a gateway from
            :func:`~platoon.testing.make_gateway` (for other gateway settings).
        service_kwargs, harness_options, program, agents: The environment entry's keys of the same names.
        max_steps: The root agent's step budget (default: the task's, else unbounded).
        rollout: Other rollout settings (``RolloutConfig`` fields such as ``timeout`` or ``inference_params``).
        output_dir: Where the rollout writes its files (default: a new temporary directory, kept).

    Returns:
        An :class:`EpisodeRun` with the trajectory collection and the recorded model calls.
    """

    from platoon.config_defs import RolloutConfig
    from platoon.gateway import ModelGateway
    from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
    from platoon.task import Task
    from platoon.testing.gateway import make_gateway
    from platoon.train.components import AgentTreeConfig, EnvironmentRoute
    from platoon.train.tinker.proxy import proxy_interactions

    gateway = policy if isinstance(policy, ModelGateway) else make_gateway(policy)
    if isinstance(rollout, RolloutConfig):
        config = copy.deepcopy(rollout)
    else:
        settings: dict[str, Any] = {"verbose": False, **dict(rollout or {})}
        config = RolloutConfig(**settings)
    directory = Path(output_dir) if output_dir is not None else Path(scratch.mkdtemp("platoon-episode-"))
    config.output_dir = str(directory)
    config.return_dict = False
    if max_steps is not None:
        config.max_steps = max_steps
    route = EnvironmentRoute(
        name="testing",
        service=service,
        service_kwargs=dict(service_kwargs or {}),
        harness=harness,
        harness_options=dict(harness_options or {}),
        program=program,
        agents=AgentTreeConfig.coerce(agents),
        resolve_task=isinstance(task, str),
    )
    task = Task(id=task) if isinstance(task, str) else copy.copy(task)
    if config.max_steps is not None:
        task.max_steps = config.max_steps
    interactions: dict[str, TinkerLLMInteraction] = {}
    token = proxy_interactions.set(interactions)
    try:
        collection = await run_harness_rollout(route.attach(task), config, gateway=EmbeddedGateway(gateway))
    finally:
        proxy_interactions.reset(token)
    return EpisodeRun(
        collection=cast("TrajectoryCollection", collection),  # not a dict: config.return_dict is off
        interactions=interactions,
        gateway=gateway,
        output_dir=directory,
    )
