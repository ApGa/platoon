from __future__ import annotations

import asyncio

from platoon.config_defs import RolloutConfig
from platoon.eval.config_defs import EvalWorkflowConfig
from platoon.eval.subprocess_worker import (
    KILL_BACKSTOP_SECONDS,
    PARENT_GRACE_SECONDS,
    SANDBOX_STOP_SECONDS,
    _hard_timeout_seconds,
)
from platoon.eval.workflow import DefaultEvalGroupWorkflow, _ProcessSlots
from platoon.task import Task


def test_eval_subprocess_hard_timeout_includes_cleanup_grace():
    assert _hard_timeout_seconds({"timeout": 10}) == 190
    assert _hard_timeout_seconds({}) == 1080
    assert DefaultEvalGroupWorkflow._subprocess_hard_timeout_seconds(RolloutConfig(timeout=10)) == 190


async def unfinished_rollout(task: Task, config: RolloutConfig) -> dict:
    raise AssertionError("never runs")


def test_the_parent_waits_past_the_hard_timeout_while_the_rollout_stops_its_sandboxes(monkeypatch, tmp_path):
    assert SANDBOX_STOP_SECONDS < KILL_BACKSTOP_SECONDS < PARENT_GRACE_SECONDS
    graces = []

    async def run(self, timeout_s, fn, *args, grace_s=0.0):
        graces.append(grace_s)
        raise TimeoutError(f"Subprocess hard timeout after {timeout_s}s")

    monkeypatch.setattr(_ProcessSlots, "run", run)
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=unfinished_rollout,
        get_task_fn=lambda task_id: Task(goal="g", id=task_id),
        config=EvalWorkflowConfig(use_subprocesses=True, rollout_config=RolloutConfig(timeout=10)),
        model_name="m",
    )

    record = asyncio.run(workflow.arun_rollout({"task_id": "a"}, 0, str(tmp_path)))

    assert graces == [PARENT_GRACE_SECONDS]
    assert "hard timeout" in record.error
