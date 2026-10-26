"""A training config whose ``eval`` names only ``every`` evaluates with training's rollout and worker settings (RFC
0018, milestone M3's exit check).

The run is the trainer's own loop, ``PlatoonTinkerRLTrainer.train``, with Tinker faked (``trainer_fakes``), on a
Harbor dataset directory whose held-out tasks are the eval split.  Training's workflow config runs each rollout in a
worker process (``rollout_isolation: process``) with worker settings of its own, and sets the rollout's step limit,
time limit and sampling; the evaluation sets none of them.  ``GroupRolloutWorkflow`` builds every rollout's worker
request as a run does; a stand-in for the worker supervisor records how each workflow built it and every request it
got, round-trips the request through pickle as the process boundary does, and runs it with the worker's own body,
``run_rollout_request``, in this process.  ``platoon-react`` writes ``hello.txt`` with one ``bash`` call in a
``process`` sandbox, sampled from a scripted policy through the real model gateway, which records the sampling
parameters of each call, and each task's own tests grade it.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from trainer_fakes import Workflow, datum, make_trainer, trainer_config

from platoon.testing import ScriptedPolicy, serve_policy, tool_call
from platoon.train.auto import AutoDataset, AutoRollout, AutoTaskLoader
from platoon.train.components import rollout_overhead_s
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.labels import label_column
from platoon.train.workers.run import WORKER_ENV_MARKER, run_rollout_request
from platoon.train.workers.supervisor import WorkerOutcome
from platoon.train.workflow import group as group_module
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.utils.config import resolve_config

TASK_TOML = """version = "1.0"

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 30.0

[environment]
docker_image = "ubuntu:24.04"
workdir = "/app"
"""
GRADE = 'reward=0; [ "$(cat hello.txt)" = hello ] && reward=1; echo $reward > "$verifier_dir/reward.txt"'
# Training's rollout and worker settings, each away from its default; the evaluation sets none of them.
WORKFLOW = {
    "group_size": 2,
    "rollout_isolation": "process",
    "worker_start_method": "exec",
    "worker_startup_budget_seconds": 123.0,
    "worker_finalize_grace_seconds": 11.0,
    "worker_parent_grace_seconds": 7.0,
    "worker_termination_grace_seconds": 3.0,
    "worker_rss_soft_limit_bytes": 12345678901,
    "worker_mem_available_floor_bytes": 1024,
    "worker_python_env_isolation": True,
    "rollout_config": {
        "max_steps": 5,
        "timeout": 120,
        "sampling_precedence": "config",
        "inference_params": {"temperature": 0.7, "top_p": 0.9, "max_completion_tokens": 321},
    },
}


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT and SIGTERM handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


class _Supervisor:
    """Stands in for ``RolloutWorkerSupervisor``: records the settings each workflow built it with and the requests
    it got, and runs each request with the worker's own body in this process, after a pickle round trip."""

    built: list[_Supervisor] = []

    def __init__(self, **settings) -> None:
        self.settings = settings
        self.requests: list = []
        _Supervisor.built.append(self)

    def publish_binding(self, sampling_client, version: int) -> None:
        pass

    async def run(self, request, *, log_dir=None, on_ready=None, on_launch=None, on_reap=None) -> WorkerOutcome:
        request = pickle.loads(pickle.dumps(request))  # what reaches a worker process
        self.requests.append(request)
        if on_ready is not None:
            on_ready()
        response = await run_rollout_request(request)  # resolves the rollout function by name, as a worker does
        return WorkerOutcome(
            response=response,
            returncode=0,
            disposition=response.status,
            killed_by_parent=False,
            diag=None,
            launch_seconds=0.0,
            wall_seconds=0.0,
            retried=False,
            worker_pid=None,
            request_id=request.request_id,
        )

    def close(self) -> None:
        pass


def _answer(prompt: str) -> str:
    """The scripted policy: write hello.txt with one bash call, then finish once its result is in the prompt."""

    return "I wrote hello.txt." if "<tool_response>" in prompt else tool_call("bash", command="echo hello > hello.txt")


def _dataset(root: Path) -> Path:
    """Four tasks of two sources, each task its own group; each instruction names its task."""

    index = {}
    for source in ("alpha", "beta"):
        for number in range(2):
            name = f"{source}-{number}"
            files = {
                "task.toml": TASK_TOML,
                "instruction.md": f"Task {name}: write hello to hello.txt.\n",
                "environment/Dockerfile": "FROM ubuntu:24.04\n",
                # PLATOON_SANDBOX_ROOT is set by the process provider only.
                "tests/test.sh": f'#!/bin/bash\nverifier_dir="${{PLATOON_SANDBOX_ROOT:-}}/logs/verifier"\n{GRADE}\n',
            }
            for relative, text in files.items():
                path = root / name / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            index[name] = {"task_path": name, "source_dataset": source, "task_group": name}
    (root / "task-index.json").write_text(json.dumps(index))
    return root


def _config(tmp_path: Path) -> PlatoonTinkerRLTrainerConfig:
    environment = {
        "service": "harbor",
        "service_kwargs": {
            "dataset_dir": str(_dataset(tmp_path / "dataset")),
            "holdout": {"fraction": 0.5, "group_by": "task_group", "stratify_by": "source_dataset"},
            "provider": "process",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        "harness": "platoon-react",
    }
    path = tmp_path / "train.yaml"
    path.write_text(
        json.dumps(
            {
                "train": {
                    "model_name": "m",
                    "renderer_name": "r",
                    "batch_size": 1,
                    "max_training_steps": 1,
                    "workflow_config": WORKFLOW,
                },
                "eval": {"every": 1},  # nothing else: how it rolls out comes from training
                "watchdog": {"enabled": False},
                "environments": [environment],
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def test_an_evaluation_that_names_only_every_rolls_out_with_trainings_rollout_and_worker_settings(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    train_dataset = AutoDataset.from_config(config, "train")
    eval_dataset = AutoDataset.from_config(config, "eval")
    held_out = set(eval_dataset["task_id"])
    assert len(held_out) == 2 and len(train_dataset) == 2

    trainer, _saves, _logged = make_trainer(monkeypatch, config)
    trainer.train_dataset, trainer.eval_dataset = train_dataset, eval_dataset
    trainer._model_info = SimpleNamespace(model_name="m", llm=trainer._model_info.llm)
    trainer._stats_logger = SimpleNamespace(wandb_run_id=None, log=lambda step, stats, force=False: None)
    monkeypatch.setattr(group_module, "RolloutWorkerSupervisor", _Supervisor)
    monkeypatch.setattr(_Supervisor, "built", [])
    # The stand-in runs the requests in this process, which is no rollout worker: without the worker marker, the
    # requests' python_env_isolation (training's) leaves alone the per-process Python overlay, which a process enters
    # once at a time and this test's concurrent rollouts would share.
    monkeypatch.delenv(WORKER_ENV_MARKER, raising=False)

    def workflow(split: str) -> GroupRolloutWorkflow:  # as platoon train builds each phase's workflow
        section = config.train if split == "train" else config.eval
        return GroupRolloutWorkflow(
            rollout_fn=AutoRollout.from_config(config, split),
            get_task_fn=AutoTaskLoader.from_config(config, split),
            config=section.workflow_config,
            model_info=trainer.model_info,
            log_path=trainer.run_log_path,
            stats_scope=split,
            label_by=label_column(config),
        )

    policy = ScriptedPolicy(_answer)
    with serve_policy(policy):  # the process's policy, which harness rollouts sample
        asyncio.run(
            asyncio.wait_for(trainer.train(train_workflow=workflow("train"), eval_workflow=workflow("eval")), 300)
        )

    supervisors: dict[str, list[_Supervisor]] = {"train": [], "eval": []}
    for supervisor in _Supervisor.built:  # each workflow's, by where its rollouts write
        scopes = {
            request.rollout_config["output_dir"].split("/rollouts/")[1].split("/")[0] for request in supervisor.requests
        }
        [scope] = scopes
        supervisors[scope].append(supervisor)
    [training], [evaluation] = supervisors["train"], supervisors["eval"]
    # Each phase's worker supervisor, built with training's worker settings.
    assert evaluation.settings == training.settings
    assert (evaluation.settings["start_method"], evaluation.settings["termination_grace_seconds"]) == ("exec", 3.0)
    assert (evaluation.settings["parent_grace_seconds"], evaluation.settings["mem_available_floor_bytes"]) == (
        7.0,
        1024,
    )
    # The evaluation after the step (none of the starting policy without eval.at_start): each held-out task once,
    # one rollout a task.
    assert sorted(request.task_id for request in evaluation.requests) == sorted(held_out)
    assert {request.rollout_number for request in evaluation.requests} == {0}
    assert sorted(request.rollout_number for request in training.requests) == [0, 1]  # training's group of 2
    for request in evaluation.requests:
        rollout = request.rollout_config
        assert (request.task.max_steps, rollout["max_steps"], rollout["timeout"]) == (5, 5, 120)
        assert rollout["inference_params"] == {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": None,
            "max_completion_tokens": 321,
        }
        assert rollout["sampling_precedence"] == "config"
        assert (request.startup_budget_seconds, request.rss_soft_limit_bytes) == (123.0, 12345678901)
        assert request.python_env_isolation is True
        assert request.rollout_budget_seconds == 120 + rollout_overhead_s(request.task) + 11.0
    # Every model call of the evaluation sampled as training's calls did.
    calls = {"train": [], "eval": []}
    for call in policy.calls:
        task = call.text.partition("Task ")[2].partition(":")[0]
        calls["eval" if any(task_id.endswith(task) for task_id in held_out) else "train"].append(call)
    assert len(calls["eval"]) == 4 and len(calls["train"]) == 4  # two calls a rollout
    for call in calls["eval"] + calls["train"]:
        params = call.sampling_params
        assert (params.temperature, params.top_p, params.max_tokens) == (0.7, 0.9, 321)


class _ProcessEvaluation:
    """An evaluation workflow whose rollouts run in worker processes: records each binding it is given."""

    rollout_isolation = "process"

    def __init__(self) -> None:
        self.bindings: list[tuple[object, int]] = []

    def publish_model_binding(self, sampling_client, version: int) -> None:
        self.bindings.append((sampling_client, version))

    async def arun_episode(self, data):
        return [datum()]


def test_an_evaluation_in_worker_processes_gets_the_binding_while_training_runs_in_process(monkeypatch, tmp_path):
    """Workers read the sampling client from the binding training publishes for its own workers; when only the
    evaluation runs in worker processes, each evaluation publishes it first."""

    config = trainer_config(tmp_path, "train.max_training_steps=1", "eval.every=1", "eval.at_start=true")
    trainer, _saves, _logged = make_trainer(monkeypatch, config, eval_tasks=2)
    evaluation = _ProcessEvaluation()

    asyncio.run(asyncio.wait_for(trainer.train(train_workflow=Workflow(), eval_workflow=evaluation), timeout=60))

    # The starting policy's evaluation (eval.at_start), then the one after the step, each with the policy it samples.
    assert [version for _client, version in evaluation.bindings] == [0, 1]
