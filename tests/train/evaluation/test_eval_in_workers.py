"""An evaluation whose rollouts run in worker processes while training's run in the trainer's: the workers read the
sampling client from the binding published to them, which each step publishes to training's workflow alone, so each
evaluation publishes the policy it evaluates to its own workflow before any of its groups starts
(``platoon.train.evaluation.loop``)."""

from __future__ import annotations

import asyncio

from fakes.config import trainer_config
from fakes.trainer import Workflow, make_trainer


class _InWorkers(Workflow):
    """An evaluation workflow whose rollouts run in worker processes: each binding published to it and each group it
    runs is an event of ``events``."""

    rollout_isolation = "process"

    def __init__(self, events: list[str]) -> None:
        super().__init__(delay=0.01)
        self.events = events

    def publish_model_binding(self, sampling_client, version) -> None:
        self.events.append("binding")

    async def arun_episode(self, data):
        self.events.append("group")
        return await super().arun_episode(data)


def test_each_evaluation_publishes_its_binding_to_its_workers_before_its_groups_start(tmp_path):
    events: list[str] = []
    trainer, _saves, logged = make_trainer(trainer_config(tmp_path, "eval.every=1"), eval_tasks=2)

    run = trainer.train(train_workflow=Workflow(delay=0.01), eval_workflow=_InWorkers(events))
    asyncio.run(asyncio.wait_for(run, 60))

    evaluations = sum("eval/step" in stats for stats in logged.stats)
    assert evaluations >= 1
    assert events == ["binding", "group", "group"] * evaluations  # each evaluation's binding, then its two groups
