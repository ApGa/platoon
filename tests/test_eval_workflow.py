"""``platoon eval`` rollouts: tasks from the task loader, the policy installed where rollouts run, event files."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field

import pytest

from platoon.config_defs import RolloutConfig
from platoon.eval.config_defs import EvalConfig, EvalWorkflowConfig
from platoon.eval.runner import EvalRunner
from platoon.eval.workflow import DefaultEvalGroupWorkflow
from platoon.task import Task


def load_task(task_id: str) -> Task:
    return Task(goal=f"solve {task_id}", id=task_id, misc={"route": "toy"})


async def fragile_rollout(task: Task, config: RolloutConfig) -> dict:
    """``toy_rollout``, except for tasks named to die (``os._exit``), to hang and then die, or to take a while
    (``slow-<seconds>-<n>``)."""

    if task.id.startswith("dies"):
        os._exit(3)
    if task.id.startswith("hangs"):
        time.sleep(12)  # past the hard timeout the test sets, then dead, as a worker's own alarm kills it
        os.kill(os.getpid(), signal.SIGKILL)
    if task.id.startswith("slow-"):
        await asyncio.sleep(float(task.id.split("-")[1]))
    return await toy_rollout(task, config)


async def setup_failing_rollout(task: Task, config: RolloutConfig) -> dict:
    """``toy_rollout``, except that task ``bad`` cannot reach its env server (an exception that does not pickle)."""

    from platoon.envs.service import EnvSetupError

    if task.id == "bad":
        raise EnvSetupError("env_connect", f"no env server for {task.id}")
    return await toy_rollout(task, config)


async def toy_rollout(task: Task, config: RolloutConfig) -> dict:
    """A finished one-trajectory rollout whose reward says whether it saw its task and config."""

    seen = task.misc.get("route") == "toy" and config.return_dict and not config.train
    return {
        "id": f"collection-{task.id}",
        "trajectories": {
            "root": {
                "id": "root",
                "task": {"id": task.id, "goal": task.goal, "max_steps": task.max_steps},
                "steps": [{"misc": {}}],
                "reward": 1.0 if seen else 0.0,
                "misc": {},
            }
        },
    }


@dataclass(frozen=True)
class CountingPolicy:
    installs: list[int] = field(default_factory=list)

    def install(self) -> None:
        self.installs.append(1)


def _workflow(
    *, use_subprocesses: bool = False, policy=None, rollout_fn=toy_rollout, workers: int = 1
) -> DefaultEvalGroupWorkflow:
    config = EvalWorkflowConfig(
        use_subprocesses=use_subprocesses,
        subprocess_max_workers=workers,
        rollout_config=RolloutConfig(max_steps=5, timeout=60),
    )
    return DefaultEvalGroupWorkflow(
        rollout_fn=rollout_fn, get_task_fn=load_task, config=config, model_name="Qwen/Qwen3-4B", policy=policy
    )


def _events(output_dir, task_id):
    (path,) = (output_dir / "events").glob(f"events_{task_id}_collection-{task_id}.jsonl")
    return [json.loads(line)["type"] for line in path.read_text().splitlines()]


def test_in_process_rollouts_install_the_policy_once_and_write_events(tmp_path):
    policy = CountingPolicy()
    workflow = _workflow(policy=policy)

    async def run():
        return [await workflow.arun_rollout({"task_id": task_id}, 0, str(tmp_path / task_id)) for task_id in ("a", "b")]

    records = asyncio.run(run())
    assert [(record.task_id, record.success, record.error) for record in records] == [
        ("a", True, None),
        ("b", True, None),
    ]
    assert records[0].trajectory_collection["trajectories"]["root"]["task"]["max_steps"] == 5
    assert policy.installs == [1]
    assert _events(tmp_path / "a", "a") == [
        "trajectory_created",
        "trajectory_task_set",
        "trajectory_step_added",
        "trajectory_finished",
    ]


def test_subprocess_rollouts_ship_the_loaded_task_and_a_rollout_reference(tmp_path):
    workflow = _workflow(use_subprocesses=True)

    async def run():
        try:
            return await workflow.arun_rollout({"task_id": "c"}, 0, str(tmp_path))
        finally:
            await workflow.aclose()

    record = asyncio.run(run())
    assert (record.success, record.error) == (True, None)
    assert _events(tmp_path, "c")[0] == "trajectory_created"


def test_the_policy_needs_a_renderer():
    assert EvalConfig(model_name="Qwen/Qwen3-4B").policy() is None
    policy = EvalConfig(
        model_name="Qwen/Qwen3-4B", renderer_name="qwen3", model_path="tinker://run/sampler_weights/final"
    ).policy()
    assert (policy.model_name, policy.renderer_name, policy.model_path) == (
        "Qwen/Qwen3-4B",
        "qwen3",
        "tinker://run/sampler_weights/final",
    )


def test_a_subprocess_rollout_failure_says_what_failed_and_the_next_rollout_runs(tmp_path):
    workflow = _workflow(use_subprocesses=True, rollout_fn=setup_failing_rollout)

    async def run():
        try:
            return [
                await workflow.arun_rollout({"task_id": task_id}, 0, str(tmp_path / task_id))
                for task_id in ("bad", "c")
            ]
        finally:
            await workflow.aclose()

    bad, good = asyncio.run(run())
    assert bad.error == "EnvSetupError: no env server for bad" and not bad.success
    assert (good.success, good.error) == (True, None)


def _errors(report: dict) -> dict[str, str]:
    return {
        task["task_id"]: rollout["error"]
        for task in report["tasks"]
        for rollout in task["rollouts"]
        if rollout["error"] is not None
    }


@pytest.mark.parametrize(
    ("package", "harness", "program"),
    [
        ("eval_pkg.harnesses", "eval-submitter", "single"),
        ("eval_pkg.programs", "eval_pkg.harnesses:submitter", "eval-program"),
    ],
    ids=["harness", "program"],
)
def test_subprocess_rollouts_import_the_environments_packages(tmp_path, package, harness, program):
    from eval_pkg.policy import InertPolicy

    from platoon.eval.run import PlatoonEvalConfig
    from platoon.train.auto import AutoDataset, AutoEnvironment, AutoRollout, AutoTaskLoader

    environment = {
        "name": "pkg",
        "service": "python://eval_pkg.envs:OneShotEnv",
        "harness": harness,
        "program": program,
        "package": package,  # registers what the entry names; nothing else imports it
    }
    config = PlatoonEvalConfig(inference=EvalConfig(model_name="inert"), environments=[environment], split="test")
    AutoEnvironment.load(config)
    packages = AutoEnvironment.packages(config)
    assert packages == (package,)
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=AutoRollout.from_config(config, "test"),
        get_task_fn=AutoTaskLoader.from_config(config),
        config=EvalWorkflowConfig(
            use_subprocesses=True, subprocess_max_workers=1, rollout_config=RolloutConfig(max_steps=5, timeout=60)
        ),
        model_name="inert",
        policy=InertPolicy(),
        packages=packages,
    )
    runner = EvalRunner(workflow=workflow, output_dir=str(tmp_path / "results"))

    report = asyncio.run(runner.arun(dataset=list(AutoDataset.from_config(config, "test")), resume=False))

    assert _errors(report) == {}
    assert (report["summary"]["valid_rollouts"], report["summary"]["success_rate"]) == (2, 1.0)
