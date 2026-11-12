"""A team's rollouts under ``rollout_isolation: process`` (slow: each rollout worker imports the tinker and torch stack
as it starts): each rollout runs its session in a rollout worker of its own, under the worker's settings, and its rows
come back to the group as in-process rollouts' do."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import teams_workers
from platoon_harness import Session
from teams_toys import prove

from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.config_defs import WorkflowConfig
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import GroupOutput

HERE = Path(__file__).resolve().parent
TESTING = HERE.parents[2] / "testing"


def test_a_rollout_runs_under_process_isolation(_sdk_home, tmp_path, monkeypatch):
    prove(_sdk_home)
    facts = tmp_path / "facts"
    facts.mkdir()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(HERE), str(TESTING), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv(teams_workers.FACTS, str(facts))
    workflow = GroupRolloutWorkflow(
        rollout_fn=teams_workers.team_rollout,
        get_task_fn=lambda task_id: Task(goal="the lockbox", id=task_id),
        config=WorkflowConfig(
            group_size=2,
            rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts"), timeout=600),
            rollout_isolation="process",
            worker_startup_budget_seconds=300.0,
            worker_termination_grace_seconds=1.0,
        ),
        # No model name: the workers bind no policy and contact no service, and each rollout samples its own.
        model_info=SimpleNamespace(
            model_name="", renderer_name="", renderer_kwargs={}, llm=SimpleNamespace(version=0, model_name="")
        ),
        log_path=str(tmp_path / "logs"),
        stats_scope="test_teams_isolation",
    )

    try:
        output = asyncio.run(asyncio.wait_for(workflow.arun_episode({"task_id": "test:1"}), 900))
    finally:
        workflow.supervisor.close()

    assert isinstance(output, GroupOutput)
    noted = [json.loads(path.read_text()) for path in facts.iterdir()]
    states = [asyncio.run(_state(note["session"])) for note in noted]
    assert output.observed_rollouts == 2 and len(output) == 4  # each root's two calls, a row each
    assert [member.valid for member in output.members] == [True, True]
    assert len({note["pid"] for note in noted} | {os.getpid()}) == 3  # two workers, neither the test's process
    assert states == [("closed", {"solver": "ended"})] * 2


async def _state(name: str) -> tuple[str, dict[str, str]]:
    async with Session.attach(name) as ended:
        return (await ended.status())["state"], {row["agent"]: row["state"] for row in await ended.agents()}
