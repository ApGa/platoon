"""Each evaluation during training breaks its metrics down by the run's label column (``train.label_by``), as each
training step does: ``eval/<column>/<label>/{groups, task_reward, outcome/<status>}``, every label of the evaluation's
tasks, also at zero (RFC 0018, section 1.5)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import tinker
import torch

from platoon.train.tinker import rl as rl_module
from platoon.train.tinker.config_defs import CheckpointConfig, EvalConfig
from platoon.train.tinker.workflows.group_rollout_workflow import _TaskRolloutOutput
from platoon.utils.rollout_stats import OUTCOME_STATUSES, outcome_counts, summarize_rollout_group
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_tracker import StatsTracker


def _output(reward: float, label: str, status: str = "scored") -> _TaskRolloutOutput:
    stats = summarize_rollout_group({"task_reward": [reward]})
    stats.counts.update(outcome_counts([status]))
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            name: tinker.TensorData.from_torch(torch.tensor(values))
            for name, values in {"advantages": [0.0, 0.0], "mask": [1.0, 1.0], "target_tokens": [2, 3]}.items()
        },
    )
    return _TaskRolloutOutput(
        [datum],
        workload=RolloutWorkload(),
        requested_rollouts=1,
        observed_rollouts=1,
        trainable_rollouts=0,
        task_retained_datums=1,
        rollout_stats=stats,
        label=label,
    )


class _Batches:
    """An eval dataloader of one-task batches."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def get_batch(self, index: int) -> list[dict]:
        return [self.rows[index]]


def test_an_evaluation_logs_every_label_of_its_tasks_also_at_zero():
    tasks = [
        {"task_id": "a", "source_dataset": "Fine/Smol"},
        {"task_id": "b", "source_dataset": "Fine/Smol"},
        {"task_id": "c", "source_dataset": "MCO"},
    ]
    outputs = {"a": _output(1.0, "Fine/Smol"), "b": _output(0.0, "Fine/Smol", "timeout"), "c": None}  # c's failed
    trainer = object.__new__(rl_module.PlatoonTinkerRLTrainer)
    trainer.config = SimpleNamespace(
        train=SimpleNamespace(label_by="source_dataset"),
        eval=EvalConfig(every=1, at_start=True, max_concurrent_rollouts=1),
        checkpoint=CheckpointConfig(every=0),
    )
    trainer.eval_dataset = tasks
    logged: list[dict] = []

    async def worker(shared_state, workflow, queue, results, tracker=None, rows=None):
        while True:
            data = await queue.get()
            results.put_nowait(outputs[data["task_id"]])

    trainer._rollout_workflow_worker_loop = worker

    async def run() -> None:
        state = rl_module.TrainLoopSharedState(
            config=trainer.config,
            shutdown_event=asyncio.Event(),
            train_step=0,
            train_dataloader=[None],
            eval_dataloader=_Batches(tasks),
            optimizer_params=None,
            service_client=None,
            training_client=None,
            sampling_client=None,
            sampling_client_step=0,
            sampling_client_updated_event=asyncio.Event(),
            model_info=None,
            stats_logger=None,
            train_tracker=StatsTracker("train"),
        )

        def log(*, step, stats, force=False):
            logged.append(stats)
            state.shutdown_event.set()  # one evaluation is enough
            state.sampling_client_updated_event.set()

        state.stats_logger = SimpleNamespace(log=log)
        state.sampling_client_updated_event.set()  # as at launch: the starting policy's evaluation (at_start)
        await asyncio.wait_for(trainer._eval_loop(state, object()), timeout=30)

    asyncio.run(run())

    (stats,) = logged
    smol, mco = "eval/source_dataset/Fine--Smol", "eval/source_dataset/MCO"
    assert (stats[f"{smol}/groups"], stats[f"{smol}/task_reward/avg"]) == (2, 0.5)
    assert (stats[f"{smol}/outcome/scored"], stats[f"{smol}/outcome/timeout"]) == (1, 1)
    assert stats[f"{mco}/groups"] == 0  # its only group failed: logged at zero, with no reward
    assert all(stats[f"{mco}/outcome/{status}"] == 0 for status in OUTCOME_STATUSES)
    assert f"{mco}/task_reward/avg" not in stats
    assert (stats["eval/step"], stats["eval/num_tasks"]) == (0, 3)
