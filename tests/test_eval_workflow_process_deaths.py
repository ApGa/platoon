"""``platoon eval`` subprocess rollouts whose process dies or outlives its hard timeout fail alone: the next
rollout runs and the siblings complete.

Slow: each test starts subprocess rollouts and waits them out.  The rollouts and the workflow are
``test_eval_workflow.py``'s.
"""

from __future__ import annotations

import asyncio

from test_eval_workflow import _errors, _workflow, fragile_rollout

from platoon.eval.runner import EvalRunner
from platoon.eval.workflow import DefaultEvalGroupWorkflow


def _run_tasks(tmp_path, task_ids: list[str], *, workers: int) -> dict:
    workflow = _workflow(use_subprocesses=True, rollout_fn=fragile_rollout, workers=workers)
    runner = EvalRunner(workflow=workflow, output_dir=str(tmp_path / "results"))
    return asyncio.run(runner.arun(dataset=[{"task_id": task_id} for task_id in task_ids], resume=False))


def test_a_rollout_whose_process_dies_fails_alone(tmp_path):
    report = _run_tasks(tmp_path, ["a", "dies", "b", "c"], workers=1)

    assert list(_errors(report)) == ["dies"]
    assert report["summary"]["successful_rollouts"] == 3


def test_the_siblings_of_a_rollout_whose_process_dies_complete(tmp_path):
    report = _run_tasks(tmp_path, ["slow-4-1", "slow-4-2", "slow-4-3", "dies"], workers=4)

    assert list(_errors(report)) == ["dies"]
    assert report["summary"]["successful_rollouts"] == 3


def test_the_siblings_of_a_rollout_past_its_hard_timeout_complete(tmp_path, monkeypatch):
    workflow = _workflow(use_subprocesses=True, rollout_fn=fragile_rollout, workers=4)

    async def run():
        try:
            # Start every slot's process first, so the timings below leave interpreter start-up out.
            warm = [
                workflow.arun_rollout({"task_id": f"warm-{index}"}, 0, str(tmp_path / "warm")) for index in range(4)
            ]
            assert all(record.success for record in await asyncio.gather(*warm))
            # The hung rollout times out at 10 s and its process dies at 12 s, while siblings started at 8 s run on.
            monkeypatch.setattr(
                DefaultEvalGroupWorkflow, "_subprocess_hard_timeout_seconds", staticmethod(lambda *_: 10)
            )
            hung = asyncio.create_task(workflow.arun_rollout({"task_id": "hangs"}, 0, str(tmp_path / "hangs")))
            await asyncio.sleep(8)
            siblings = [
                workflow.arun_rollout({"task_id": f"slow-6-{index}"}, 0, str(tmp_path / f"slow-{index}"))
                for index in range(3)
            ]
            return await hung, await asyncio.gather(*siblings)
        finally:
            await workflow.aclose()

    hung, siblings = asyncio.run(run())
    assert "hard timeout" in hung.error
    assert [(sibling.success, sibling.error) for sibling in siblings] == [(True, None)] * 3
