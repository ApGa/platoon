"""A CPU training run on a Harbor dataset of two sources logs both sources' metrics every step and in each
evaluation (RFC 0018, milestone M2's exit check).

The run is the trainer's own loop, ``PlatoonTinkerRLTrainer.train``, with Tinker faked (``trainer_fakes``).  Its tasks
are listed from a Harbor dataset directory through ``AutoDataset``, each task's ``source_dataset`` index field a
column of the dataset, and held out by source.  ``GroupRolloutWorkflow`` rolls them out in process: ``platoon-react``
writes ``hello.txt`` with one ``bash`` call in a ``process`` sandbox, sampled from a scripted policy through the real
model gateway, and each task's own tests grade it: one source's tasks score 1 in ``reward.txt``, the other's 0 as
``correctness`` in ``reward.json``, under ``reward: [reward, correctness]``.
"""

from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from trainer_fakes import make_trainer

from platoon.testing import ScriptedPolicy, serve_policy, tool_call
from platoon.train.auto import AutoDataset, AutoRollout, AutoTaskLoader
from platoon.train.labels import check_labels, label_column
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker.workflows import GroupRolloutWorkflow
from platoon.utils.config import resolve_config

# Each source's tests: alpha's tasks score 1 as reward in reward.txt when hello.txt says hello, beta's 0 as
# correctness in reward.json.
SOURCES = {
    "alpha/one": 'reward=0; [ "$(cat hello.txt)" = hello ] && reward=1; echo $reward > "$verifier_dir/reward.txt"',
    "beta": 'echo \'{"correctness": 0}\' > "$verifier_dir/reward.json"',
}
TASK_TOML = """version = "1.0"

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 30.0

[environment]
docker_image = "ubuntu:24.04"
workdir = "/app"
"""


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """``train()`` installs SIGINT and SIGTERM handlers; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _answer(prompt: str) -> str:
    """The scripted policy: write hello.txt with one bash call, then finish once its result is in the prompt."""

    return "I wrote hello.txt." if "<tool_response>" in prompt else tool_call("bash", command="echo hello > hello.txt")


def _two_sources(root: Path, per_source: int = 4) -> Path:
    """A dataset directory of ``per_source`` tasks of each source, each its own group, whose tests write the source's
    reward file."""

    index = {}
    for source, grade in SOURCES.items():
        for number in range(per_source):
            name = f"{source.split('/')[0]}-{number}"
            files = {
                "task.toml": TASK_TOML,
                "instruction.md": "Write hello to hello.txt.\n",
                "environment/Dockerfile": "FROM ubuntu:24.04\n",
                # PLATOON_SANDBOX_ROOT is set by the process provider only.
                "tests/test.sh": f'#!/bin/bash\nverifier_dir="${{PLATOON_SANDBOX_ROOT:-}}/logs/verifier"\n{grade}\n',
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
            "dataset_dir": str(_two_sources(tmp_path / "dataset")),
            "holdout": {"fraction": 0.25, "group_by": "task_group", "stratify_by": "source_dataset"},
            "reward": ["reward", "correctness"],
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
                    "batch_size": 2,
                    "max_training_steps": 3,
                    "max_concurrent_rollouts": 4,
                    "label_by": "source_dataset",
                    "workflow_config": {"group_size": 2, "rollout_config": {"timeout": 120}},
                },
                "eval": {"every": 1, "max_concurrent_rollouts": 2},
                "checkpoint": {"every": 0},
                "watchdog": {"enabled": False},
                "environments": [environment],
                "log_path": str(tmp_path / "logs"),
            }
        )
    )
    return resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config


def test_a_run_on_two_sources_logs_both_sources_every_step_and_in_each_evaluation(monkeypatch, tmp_path):
    config = _config(tmp_path)
    train_dataset = AutoDataset.from_config(config, "train")
    eval_dataset = AutoDataset.from_config(config, "eval")
    check_labels(config, train_dataset, eval_dataset)  # as platoon train does: source_dataset labels every task
    assert sorted(eval_dataset["source_dataset"]) == ["alpha/one", "beta"]  # one held-out task of each source
    assert len(train_dataset) == 6

    trainer, _saves, _steps = make_trainer(monkeypatch, config)
    trainer.train_dataset, trainer.eval_dataset = train_dataset, eval_dataset
    trainer._model_info = SimpleNamespace(model_name="m", llm=trainer._model_info.llm)
    logged: list[dict] = []
    trainer._stats_logger = SimpleNamespace(
        wandb_run_id=None, log=lambda step, stats, force=False: logged.append(stats)
    )

    def workflow(split: str) -> GroupRolloutWorkflow:
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

    with serve_policy(ScriptedPolicy(_answer)):  # the process's policy, which harness rollouts sample
        asyncio.run(
            asyncio.wait_for(trainer.train(train_workflow=workflow("train"), eval_workflow=workflow("eval")), 300)
        )

    steps = [stats for stats in logged if "train/step" in stats]
    evaluations = [stats for stats in logged if "eval/step" in stats]
    alpha, beta = "source_dataset/alpha--one", "source_dataset/beta"
    assert [stats["train/step"] for stats in steps] == [1, 2, 3]
    for stats in steps:  # both sources, every step: each step's two groups between them, a source it took none at 0
        alpha_groups, beta_groups = stats[f"train/batch/{alpha}/groups"], stats[f"train/batch/{beta}/groups"]
        assert alpha_groups + beta_groups == 2
        assert stats[f"train/batch/{alpha}/outcome/scored"] == 2 * alpha_groups
        if alpha_groups:
            assert stats[f"train/batch/{alpha}/task_reward/avg"] == 1.0
        if beta_groups:
            assert stats[f"train/batch/{beta}/task_reward/avg"] == 0.0
        # Each rollout counted by the reward key it trained: alpha's reward.txt trains reward, beta's correctness.
        assert stats.get("train/batch/reward_key/reward", 0) == 2 * alpha_groups
        assert stats.get("train/batch/reward_key/correctness", 0) == 2 * beta_groups
        assert stats.get(f"train/batch/{beta}/reward_key/reward", 0) == 0
        assert stats["train/batch/reward_key_mismatch"] == 0
    # The last step's evaluation, and any before it that training did not overtake; none of step 0 without at_start.
    assert evaluations and evaluations[-1]["eval/step"] == 3
    assert all(stats["eval/step"] in (1, 2, 3) for stats in evaluations)
    for stats in evaluations:  # each evaluation: one held-out task of each source, each source's reward its own
        assert (stats[f"eval/{alpha}/groups"], stats[f"eval/{beta}/groups"]) == (1, 1)
        assert (stats[f"eval/{alpha}/task_reward/avg"], stats[f"eval/{beta}/task_reward/avg"]) == (1.0, 0.0)
        assert stats[f"eval/{alpha}/outcome/scored"] == stats[f"eval/{beta}/outcome/scored"] == 1
        assert stats[f"eval/{alpha}/reward_key/reward"] == stats[f"eval/{beta}/reward_key/correctness"] == 1
