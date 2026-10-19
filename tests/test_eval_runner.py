"""The evaluation report and resume: unscored outcomes are errored rollouts, never valid failures."""

from __future__ import annotations

import asyncio
import json
from collections import Counter

from eval_pkg.policy import _NoRenderer

from platoon.config_defs import RolloutConfig
from platoon.envs.service import EnvSetupError, Outcome
from platoon.eval.config_defs import EvalWorkflowConfig
from platoon.eval.runner import EvalRunner
from platoon.eval.workflow import DefaultEvalGroupWorkflow
from platoon.gateway import ModelGateway, build_trajectory_collection
from platoon.task import Task

# What each task's rollout ends in, as harness_rollout grades it.
OUTCOMES = {
    "solved": Outcome(status="scored", rewards={"reward": 1.0}),
    "crash": Outcome.infra("harness_crash", reason="custom@1 exited with code 2"),
    "grader": Outcome.infra("unexpected:RuntimeError", reason="the grader raised"),
    "invalid": Outcome.infra("env_invalid", status="invalid", reason="verifier_declared"),
    "timeout": Outcome(status="timeout", reason="agent_timeout"),
}
CALLS: Counter[str] = Counter()


async def graded_rollout(task: Task, config: RolloutConfig) -> dict:
    """The collection the gateway trace builds for the task's outcome (one root agent, no model calls); the task
    ``raises`` raises instead, and the session of the task ``never-opened`` does not open."""

    CALLS[task.id] += 1
    if task.id == "raises":
        raise RuntimeError("the harness could not start")
    if task.id == "never-opened":
        raise EnvSetupError("sandbox_start", "the sandbox did not start")
    gateway = ModelGateway(sampler=None, renderer=_NoRenderer(), tokenizer=None, model_name="m")
    gateway.open_agent(rollout_id=f"rollout-{task.id}", trajectory_id=f"root-{task.id}", interactions={})
    agents = gateway.close_rollout(f"rollout-{task.id}")
    return build_trajectory_collection(agents, task=task, outcome=OUTCOMES[str(task.id)]).to_dict()


def _runner(tmp_path, **config) -> EvalRunner:
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=graded_rollout,
        get_task_fn=lambda task_id: Task(id=task_id, goal=f"solve {task_id}"),
        config=EvalWorkflowConfig(**config),
        model_name="m",
    )
    return EvalRunner(workflow=workflow, output_dir=str(tmp_path / "results"))


def test_unscored_outcomes_are_errored_rollouts_and_resume_retries_them(tmp_path):
    CALLS.clear()
    dataset = [{"task_id": task_id} for task_id in OUTCOMES]

    report = asyncio.run(_runner(tmp_path).arun(dataset=dataset))

    summary = report["summary"]
    assert (summary["valid_rollouts"], summary["errored_rollouts"], summary["successful_rollouts"]) == (1, 4, 1)
    assert summary["success_rate"] == 1.0 and summary["success_at_k"] == 1.0  # only the measured task counts
    assert summary["reward_at_k_mean"] == 1.0 and summary["tasks_without_valid_rollouts"] == 4
    assert summary["outcome_status_counts"] == {"error": 2, "invalid": 1, "scored": 1, "timeout": 1}
    errors = {task["task_id"]: task["rollouts"][0]["error"] for task in report["tasks"]}
    assert errors["solved"] is None
    assert errors["crash"] == "Outcome error: custom@1 exited with code 2"
    assert errors["timeout"] == "Outcome timeout: agent_timeout"
    metadata = json.loads((tmp_path / "results/rollouts/invalid/rollout_0/metadata.json").read_text())
    assert (metadata["outcome_status"], metadata["status"]) == ("invalid", "errored")

    # A second run resumes: the scored rollout is reused, the errored ones run again.
    report = asyncio.run(_runner(tmp_path).arun(dataset=dataset, resume=True))
    assert CALLS == {"solved": 1, "crash": 2, "grader": 2, "invalid": 2, "timeout": 2}
    assert report["summary"]["errored_rollouts"] == 4
    # Without retry_errored they are reused as they are, and a report-only run reads them the same way.
    asyncio.run(_runner(tmp_path).arun(dataset=dataset, resume=True, retry_errored=False))
    assert CALLS == {"solved": 1, "crash": 2, "grader": 2, "invalid": 2, "timeout": 2}
    report = asyncio.run(_runner(tmp_path).arun(dataset=[], run_rollouts=False))
    assert (report["summary"]["valid_rollouts"], report["summary"]["errored_rollouts"]) == (1, 4)


def test_timeouts_can_count_as_failures(tmp_path):
    dataset = [{"task_id": "solved"}, {"task_id": "timeout"}]

    report = asyncio.run(_runner(tmp_path, count_timeout_as_failure=True).arun(dataset=dataset))

    summary = report["summary"]
    assert (summary["valid_rollouts"], summary["errored_rollouts"], summary["success_rate"]) == (2, 0, 0.5)
    assert summary["success_at_k"] == 0.5 and summary["outcome_status_counts"] == {"scored": 1, "timeout": 1}


def test_a_rollout_whose_session_never_opened_counts_as_raised(tmp_path):
    """The report counts each rollout that raised as ``raised``, also one whose environment could not open its session
    (``EnvSetupError``), which training counts as an ``error``."""

    dataset = [{"task_id": "solved"}, {"task_id": "raises"}, {"task_id": "never-opened"}]

    report = asyncio.run(_runner(tmp_path).arun(dataset=dataset))

    summary = report["summary"]
    assert summary["outcome_status_counts"] == {"raised": 2, "scored": 1}
    assert (summary["valid_rollouts"], summary["errored_rollouts"]) == (1, 2)


def test_rollouts_on_disk_are_judged_by_the_current_timeout_setting(tmp_path):
    """A report or a resumed run derives each outcome's verdict again from its collection; only a rollout that
    raised keeps the error stored with it."""

    CALLS.clear()
    dataset = [{"task_id": "solved"}, {"task_id": "timeout"}, {"task_id": "raises"}]
    report = asyncio.run(_runner(tmp_path).arun(dataset=dataset))
    assert (report["summary"]["valid_rollouts"], report["summary"]["errored_rollouts"]) == (1, 2)

    report = asyncio.run(_runner(tmp_path, count_timeout_as_failure=True).arun(dataset=[], run_rollouts=False))
    summary = report["summary"]
    assert (summary["valid_rollouts"], summary["errored_rollouts"], summary["success_rate"]) == (2, 1, 0.5)
    errors = {task["task_id"]: task["rollouts"][0]["error"] for task in report["tasks"]}
    assert errors == {"solved": None, "timeout": None, "raises": "the harness could not start"}

    # A resumed run reuses the timeout as the valid failure it now is and runs only the raised rollout again.
    asyncio.run(_runner(tmp_path, count_timeout_as_failure=True).arun(dataset=dataset, resume=True))
    assert CALLS == {"solved": 1, "timeout": 1, "raises": 2}
    # Without the setting, the timeout on disk is an errored rollout again.
    report = asyncio.run(_runner(tmp_path).arun(dataset=[], run_rollouts=False))
    assert (report["summary"]["valid_rollouts"], report["summary"]["errored_rollouts"]) == (1, 2)

    # A retry that raised leaves the earlier attempt's collection on disk: the retry's error still stands.
    metadata_path = tmp_path / "results/rollouts/timeout/rollout_0/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps({**metadata, "error": "the retry raised", "outcome_status": None}))
    report = asyncio.run(_runner(tmp_path, count_timeout_as_failure=True).arun(dataset=[], run_rollouts=False))
    errors = {task["task_id"]: task["rollouts"][0]["error"] for task in report["tasks"]}
    assert errors["timeout"] == "the retry raised"


def test_the_rollouts_label_their_sandboxes_with_the_evaluations_output_directory(tmp_path, monkeypatch):
    """Every sandbox the rollouts start carries ``platoon.run``: the output directory as an absolute path, while the
    rollouts run, and this process's labels are as they were once they are done."""

    from platoon.sandbox import cleanup as sandbox_cleanup

    monkeypatch.chdir(tmp_path)
    seen: list[dict[str, str]] = []

    async def labelled_rollout(task: Task, config: RolloutConfig) -> dict:
        seen.append(sandbox_cleanup.labels())
        return await graded_rollout(task, config)

    before = sandbox_cleanup.labels()
    workflow = DefaultEvalGroupWorkflow(
        rollout_fn=labelled_rollout,
        get_task_fn=lambda task_id: Task(id=task_id, goal=f"solve {task_id}"),
        config=EvalWorkflowConfig(),
        model_name="m",
    )
    asyncio.run(EvalRunner(workflow=workflow, output_dir="results").arun(dataset=[{"task_id": "solved"}]))

    assert seen == [{**before, sandbox_cleanup.RUN_LABEL: str(tmp_path / "results")}]
    assert sandbox_cleanup.labels() == before


def test_each_rollouts_record_names_the_reward_key_it_trained_and_the_report_counts_them(tmp_path, monkeypatch):
    correct = Outcome(status="scored", rewards={"correctness": 1.0, "submission": 1.0}, primary="correctness")
    monkeypatch.setitem(OUTCOMES, "correct", correct)
    dataset = [{"task_id": "solved"}, {"task_id": "correct"}, {"task_id": "crash"}]

    report = asyncio.run(_runner(tmp_path).arun(dataset=dataset))

    assert report["summary"]["reward_key_counts"] == {"correctness": 1, "reward": 1}  # the crash trained none
    keys = {
        task: json.loads((tmp_path / f"results/rollouts/{task}/rollout_0/metadata.json").read_text())["reward_key"]
        for task in ("solved", "correct", "crash")
    }
    assert keys == {"solved": "reward", "correct": "correctness", "crash": None}
