"""RFC 0019's fake environment of milestone M4, which the tests of ``pass_rate`` and ``informative`` run in the
trainer's own loop (Tinker faked, the real group workflow, rollouts faked at the worker boundary): 40 tasks with known
pass rates (25 nearly always solved, 15 spread from 0.1 to 0.9), where each rollout passes when a hash of its task, its
draw's position and its rollout number falls below its task's rate (:func:`passes`), so a draw drawn again gets the
outcomes it had, and no two draws share theirs.

- :func:`pool_entry`, :func:`run_config`, :func:`trainer_arguments`: an entry of the tasks, a run's config over entries,
  and what ``platoon train`` hands the trainer of that config;
- :func:`fake_rollouts`: each rollout's outcome by :func:`passes`;
- :class:`SeededWorkflow`: the real group workflow, each group's start delayed by a seed, which chooses the order the
  groups complete in;
- :class:`DatumRecordingClient`: a training client that records each step's datums.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch

from fakes.config import mixture_config
from fakes.trainer import TrainingClient
from platoon.task import Task
from platoon.train import entry_settings
from platoon.train.auto import AutoDataset
from platoon.train.dispatch.dispatcher import POSITION_COLUMN
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerResponse
from platoon.train.workflow import members as members_module
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import TrajectoryCollectionResult, TrajectoryStats

HERE = __name__
G = 8
# The pass rates of the 40 tasks: 25 at 0.99, 15 evenly from 0.1 to 0.9.
RATES = [0.99] * 25 + [0.1 + 0.8 * index / 14 for index in range(15)]
# The position of the draw whose group runs, which its rollouts inherit.
POSITION: contextvars.ContextVar[int] = contextvars.ContextVar("position")


def tasks(config: Any, split: str, *, count: int = 40) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}"} for index in range(count)]


def rate(task_id: str) -> float:
    return RATES[int(task_id.split("::")[-1][1:]) % len(RATES)]


def passes(task_id: str, position: int, rollout: int) -> bool:
    digest = hashlib.sha256(f"{task_id}:{position}:{rollout}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate(task_id)


def pool_entry(name: str | None = None, count: int = 40, **fields: Any) -> dict[str, Any]:
    """An entry of ``count`` of the tasks (:func:`tasks`), named ``name`` when given."""

    entry = {"dataset_loader": f"{HERE}:tasks", "dataset_kwargs": {"count": count}, "rollout": "x:y", **fields}
    if name:
        entry["name"] = name
    return entry


def run_config(tmp_path: Path, train: dict[str, Any], environments: list[dict[str, Any]], **sections: Any) -> Any:
    """A run of 30 steps of 8 groups of :data:`G`, none stale, unless ``train`` says otherwise
    (``fakes.config.mixture_config``)."""

    run = {"batch_size": 8, "max_training_steps": 30, "max_staleness": 0, "max_concurrent_rollouts": 64}
    return mixture_config(tmp_path, {**run, "workflow_config": {"group_size": G}, **train}, environments, **sections)


def rollout_datum(position: int, rollout: int, reward: float, version: int) -> tinker.Datum:
    """The datum of ``rollout`` of the draw at ``position``, its tokens the two, its advantage its ``reward``."""

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([position, rollout, 1]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {
                "target_tokens": [1, 2, 3],
                "logprobs": [0.0, -0.1, -0.2],
                "advantages": [0.0, reward, reward],
                "mask": [0.0, 1.0, 1.0],
                "checkpoint_version": [version],
            }.items()
        },
    )


def fake_rollouts(monkeypatch: pytest.MonkeyPatch, policy: Any) -> None:
    """Each rollout passes by the hash of its task, its draw's position and its rollout number."""

    async def run_rollout_request(request, llm, *, rollout_fn=None, reward_processor=None):
        position = POSITION.get()
        success = 1.0 if passes(request.task_id, position, request.rollout_number) else 0.0
        stats = TrajectoryStats(
            trajectory_id="root",
            reward=success,
            num_steps=2,
            num_input_tokens=3,
            num_output_tokens=4,
            num_datums=1,
            is_root=True,
        )
        result = TrajectoryCollectionResult(
            datums=[rollout_datum(position, request.rollout_number, success, policy.version)],
            task_reward=success,
            trajectory_stats=[stats],
            root_rewards_dict={},
            task_reward_valid=True,
        )
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status="completed",
            result=result,
            workload=RolloutWorkload(
                trajectories=1, postmerge_datums=1, policy_eligible_datums=1, post_sampling_datums=1
            ),
            observed=True,
            outcome_status="scored",
            success_reward=success,
            rollout_seconds=0.1,
        )

    monkeypatch.setattr(members_module, "run_rollout_request", run_rollout_request)


class SeededWorkflow(GroupRolloutWorkflow):
    """The real group workflow; each group first waits a time drawn from ``seed`` and its position, and puts its
    position where its rollouts read it.  ``completed`` holds the positions in the order their groups completed;
    ``on_start(row, count)`` runs as each group starts."""

    def __init__(self, config: Any, *, seed: int = 0, on_start: Any = None) -> None:
        super().__init__(
            rollout_fn=lambda task, rollout_config: None,
            get_task_fn=lambda task_id: Task(id=task_id),
            config=config.train.workflow_config,
            model_info=SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0, model_name="m")),
        )
        self.entry_configs = {label: own.train for label, own in entry_settings.entry_settings(config).items()}
        self.seed = seed
        self.on_start = on_start
        self.started = 0
        self.completed: list[int] = []

    async def arun_episode(self, data):
        position = data[POSITION_COLUMN]
        POSITION.set(position)
        self.started += 1
        if self.on_start is not None:
            self.on_start(data, self.started)
        await asyncio.sleep(random.Random(f"{self.seed}:{position}").uniform(0.0, 0.01))
        output = await super().arun_episode(data)
        self.completed.append(position)
        return output


class DatumRecordingClient(TrainingClient):
    """Records each step's datums: their tokens and advantages."""

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[list[tuple[tuple[int, ...], tuple[float, ...]]]] = []

    async def forward_backward_async(self, datums, loss_fn, loss_fn_config):
        self.steps.append(
            [
                (tuple(datum.model_input.to_ints()), tuple(datum.loss_fn_inputs["advantages"].to_torch().tolist()))
                for datum in datums
            ]
        )
        return await super().forward_backward_async(datums, loss_fn, loss_fn_config)


def trainer_arguments(config: Any) -> dict[str, Any]:
    """What platoon train hands the trainer of ``config``, as its constructor's arguments: ``AutoDataset.training``'s
    first epoch, order and sampling."""

    training = AutoDataset.training(config)
    return {"train_dataset": training.dataset, "train_order": training.order, "train_sampling": training.sampling}
