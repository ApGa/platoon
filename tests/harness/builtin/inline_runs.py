"""Run a function-harness agent tree on the inline runtime (``platoon.harness.inline``), as ``harness_rollout``
does: a root agent on a gateway, the env session's grade as the episode outcome, and the per-agent trace.

The built-in harness tests use it, and so do the TextCraft extension's (``extensions/textcraft/tests``).
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from platoon.envs.service import EnvSession, Outcome, get_service
from platoon.gateway import build_trajectory_collection
from platoon.gateway.core import ModelGateway
from platoon.harness.agent import AgentResult
from platoon.harness.inline import InlineAgentContext, InlineRuntime, grade_session, run_agent
from platoon.harness.registry import get_harness
from platoon.rewards import delegation_capped
from platoon.task import Task
from platoon.train.components import AgentTreeConfig
from platoon.trajectory import TrajectoryCollection
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection


@dataclass
class InlineRun:
    result: AgentResult
    outcome: Outcome
    collection: TrajectoryCollection
    interactions: dict[str, Any]

    @property
    def root(self):
        return next(iter(self.collection.trajectories.values()))

    @property
    def children(self):
        return [t for t in self.collection.trajectories.values() if t.parent_info is not None]


async def run_tree(
    gateway: ModelGateway,
    harness: str,
    session: EnvSession,
    task: Task,
    *,
    max_steps: int | None,
    options: Mapping[str, Any] | None = None,
    agents: Mapping[str, Any] | None = None,
) -> InlineRun:
    runtime = InlineRuntime(gateway=gateway, base_url=None, agents=AgentTreeConfig.coerce(agents))
    interactions: dict[str, Any] = {}
    rollout_id = f"rollout-{uuid.uuid4().hex[:8]}"
    agent = gateway.open_agent(
        rollout_id=rollout_id, trajectory_id=str(uuid.uuid4()), interactions=interactions, max_steps=max_steps
    )
    ctx = InlineAgentContext(
        runtime,
        agent,
        task=task,
        session=session,
        harness=get_harness(harness),
        instruction=session.instruction,
        options=options,
    )
    result = await run_agent(ctx)
    outcome = await grade_session(session)
    agent.outcome = outcome
    collection = build_trajectory_collection(gateway.close_rollout(rollout_id), task=task, outcome=outcome)
    return InlineRun(result, outcome, collection, interactions)


async def open_service_task(service_name: str, task_id: str) -> tuple[Task, EnvSession]:
    """Task ``task_id`` of the environment service ``service_name``, and a session opened on it."""

    service = get_service(service_name)
    task = await service.get_task(task_id)
    return task, await service.open(task, rollout_id="test")


def train_datums(run: InlineRun, task_id: str) -> Any:
    """The training data of a run's agent tree, scored with ``platoon-delegation`` and with each datum's depth."""

    return get_train_data_for_trajectory_collection(
        run.collection.to_dict(),
        run.interactions,
        task_id,
        checkpoint_version=0,
        reward_processor=delegation_capped,
        include_traj_depth=True,
    )
