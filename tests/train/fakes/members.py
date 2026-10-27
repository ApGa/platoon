"""The group workflow with scripted members (:class:`ScriptedMembers`), for the tests of the rollout permits and of the
start limit: each member's rollout is a script the test writes, run inside the member's lease block as a rollout is,
whose one datum's first two tokens are its group's position in the order and its member (:func:`member_outcome`); and a
wait for a condition the run reaches (:func:`until`).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import tinker
import torch
from tinker import TensorData

from platoon.train.config_defs import WorkflowConfig
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.workflow import group as group_module
from platoon.train.workflow import output as output_module
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats


def member_datum(position: int, member: int, version: int) -> tinker.Datum:
    """A datum whose first two tokens are the position of its group's row and its member."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, member, 2]),
        loss_fn_inputs={
            name: TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, 1.0, 1.0],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


def member_outcome(
    position: int, member: int, *, reward: float, version: int, seconds: float | None = None
) -> output_module.MemberOutcome:
    result = TrajectoryCollectionResult(
        datums=[member_datum(position, member, version)],
        task_reward=reward,
        trajectory_stats=[
            TrajectoryStats(
                trajectory_id="root",
                reward=reward,
                num_steps=1,
                num_input_tokens=3,
                num_output_tokens=2,
                num_datums=1,
                is_root=True,
            )
        ],
        root_rewards_dict={},
    )
    workload = RolloutWorkload(
        environment_steps=1,
        model_calls=1,
        input_tokens=3,
        output_tokens=2,
        trajectories=1,
        postmerge_datums=1,
        policy_eligible_datums=1,
        post_sampling_datums=1,
    )
    return output_module.MemberOutcome(result=result, workload=workload, observed=True, seconds=seconds)


async def unscripted_rollout(task: Any, config: Any) -> dict:
    """The rollout function of :class:`ScriptedMembers`, which a workflow under process isolation must be able to name
    for its workers: never called, since each member's rollout is its script."""

    raise AssertionError("a scripted member's rollout is its script")


class ScriptedMembers(group_module.GroupRolloutWorkflow):
    """The group workflow, each member's rollout a script: ``await script(row, member)`` is the rollout, and returns
    its reward, or ``(reward, seconds)``.  It runs inside the member's lease block, as a rollout does, counted as a
    rollout in flight; ``log`` records each member's start, end and cancellation."""

    def __init__(self, config: WorkflowConfig, llm: Any, script: Any, *, scope: str = "train") -> None:
        super().__init__(
            rollout_fn=unscripted_rollout,
            get_task_fn=None,
            config=config,
            model_info=SimpleNamespace(model_name="m", base_url="u", api_key="k", llm=llm),
            stats_scope=scope,
        )
        self.llm = llm
        self.script = script
        self.log: list[tuple[str, int, int]] = []

    async def arun_episode_single(self, data: dict, rollout_number: int = 0):
        position = data[POSITION_COLUMN]
        version = self.llm.version
        self.log.append(("start", position, rollout_number))
        try:
            with self._monitor().rollout_in_flight():
                answer = await self.script(data, rollout_number)
        except asyncio.CancelledError:
            self.log.append(("cancelled", position, rollout_number))
            raise
        self.log.append(("end", position, rollout_number))
        reward, seconds = answer if isinstance(answer, tuple) else (answer, None)
        return member_outcome(position, rollout_number, reward=reward, version=version, seconds=seconds)

    def started(self, position: int) -> list[int]:
        return sorted(member for kind, where, member in self.log if kind == "start" and where == position)


async def until(condition, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not condition():
        if loop.time() > end:
            raise AssertionError("the run never got there")
        await asyncio.sleep(0.01)


def scripted_group(config: WorkflowConfig, script: Any, scope: str, cls: Any = ScriptedMembers) -> Any:
    """A group workflow outside the trainer, in its own stats scope."""

    return cls(config, SimpleNamespace(version=0), script, scope=scope)
