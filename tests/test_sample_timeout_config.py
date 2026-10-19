"""The sampling timeout is a config key beside the model, with ``TINKER_SAMPLE_TIMEOUT_SECONDS`` as its fallback.

``train.sample_timeout_seconds`` bounds each sampling call of the policy ``platoon train`` serves, in its rollouts, its
rollout workers and its evaluations; ``inference.sample_timeout_seconds`` bounds those of the policy ``platoon eval``
serves; ``platoon rollout`` takes the config's.  The key wins over the variable, which a gateway job may export for
every client it runs; the variable bounds a config that sets no key, and 600 s one with neither.

The policies here are real model gateways over the toy tokenizer (``platoon/toy``) behind a fake Tinker service
client: no model download and no Tinker service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the toy model's processor lookup must not reach the Hub

from platoon import cli  # noqa: E402
from platoon.config_defs import RolloutConfig  # noqa: E402
from platoon.eval.config_defs import EvalConfig, HostedConfig, sample_timeout_note  # noqa: E402
from platoon.eval.run import arun_eval  # noqa: E402
from platoon.gateway import core  # noqa: E402
from platoon.task import Task  # noqa: E402
from platoon.testing import TOY_MODEL_NAME, toy_tokenizer  # noqa: E402
from platoon.train.tinker import proxy, rl  # noqa: E402
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig  # noqa: E402
from platoon.train.tinker.train import arun_tinker_training  # noqa: E402
from platoon.train.tinker.workflows.group_rollout_workflow import GroupRolloutWorkflow  # noqa: E402
from platoon.utils.config import ConfigError, build_config, load_config  # noqa: E402

VARIABLE = "TINKER_SAMPLE_TIMEOUT_SECONDS"
KEY = "train.sample_timeout_seconds"
RENDERER = "qwen3_instruct"
# The bound the policy's gateway sampled under, as each rollout below saw it.
BOUNDS: list[float] = []


@pytest.fixture
def tinker(monkeypatch) -> list[dict]:
    """A fake Tinker service: each policy that opens a session gets a real gateway over the toy tokenizer, which
    ``toy_tokenizer`` registers as ``platoon/toy`` with tinker-cookbook.  The variable starts unset."""

    toy_tokenizer()
    sessions: list[dict] = []

    class ServiceClient:
        def __init__(self, **kwargs):
            sessions.append(kwargs)

        def create_sampling_client(self, base_model, model_path=None):
            return object()

    monkeypatch.setattr(proxy.tinker, "ServiceClient", ServiceClient)
    monkeypatch.setattr(core, "_default_gateway", None)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("TINKER_TELEMETRY", "1")  # restored afterwards: an evaluation's policy turns it off
    monkeypatch.delenv(VARIABLE, raising=False)
    BOUNDS.clear()
    return sessions


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["t1"]


def load_task(task_id: str) -> Task:
    return Task(id=task_id, goal="g")


def no_rollout(task: Task, config: RolloutConfig) -> dict:
    """A workflow's rollout that the tests here never run."""

    raise AssertionError("never runs")


async def bound_rollout(task: Task, config) -> dict:
    """Records the bound of the policy this process serves, and scores 1."""

    BOUNDS.append(core.default_gateway().sample_timeout_seconds)
    return {"id": "c", "trajectories": {"root": {"id": "root", "task": {"id": task.id}, "reward": 1.0, "steps": []}}}


ENVIRONMENT = {
    "dataset_loader": f"{__name__}:load_tasks",
    "task_loader": f"{__name__}:load_task",
    "rollout": f"{__name__}:bound_rollout",
}


def _train_settings(tmp_path: Path, **train) -> dict:
    return {
        "train": {"model_name": TOY_MODEL_NAME, "renderer_name": RENDERER, "max_training_steps": 1, **train},
        "eval": {"every": 0},
        "log_path": str(tmp_path / "logs"),
        "environments": [ENVIRONMENT],
    }


def _eval_settings(tmp_path: Path, **inference) -> dict:
    return {
        "inference": {
            "model_name": TOY_MODEL_NAME,
            "renderer_name": RENDERER,
            "output_dir": str(tmp_path / "results"),
            **inference,
        },
        "environments": [ENVIRONMENT],
    }


def _write(path: Path, settings: dict) -> str:
    path.write_text(yaml.safe_dump(settings))
    return str(path)


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


# --------------------------------------------------------------------------- the keys


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_a_key_that_is_not_a_positive_number_of_seconds_is_refused_as_the_config_loads(tmp_path, value):
    with pytest.raises(ValueError, match=r"train\.sample_timeout_seconds must be a positive number of seconds or null"):
        build_config(PlatoonTinkerRLTrainerConfig, _train_settings(tmp_path, sample_timeout_seconds=value))
    with pytest.raises(ValueError, match=r"inference\.sample_timeout_seconds must be a positive number of seconds"):
        EvalConfig(model_name=TOY_MODEL_NAME, renderer_name=RENDERER, sample_timeout_seconds=value)


def test_config_validate_refuses_a_bad_key_and_takes_a_good_one(tmp_path, capsys):
    bad = _write(tmp_path / "bad.yaml", _train_settings(tmp_path, sample_timeout_seconds=0))
    good = _write(tmp_path / "good.yaml", _eval_settings(tmp_path, sample_timeout_seconds=1800))

    assert _exit_code(["config", "validate", "--no-resolve", bad]) == cli.EXIT_CONFIG
    assert f"{KEY} must be a positive number of seconds or null, got 0" in capsys.readouterr().out
    cli.main(["config", "validate", "--no-resolve", good])
    assert f"ok   {good} (PlatoonEvalConfig)" in capsys.readouterr().out


def test_a_training_configs_eval_section_points_at_the_train_key(tmp_path):
    path = _write(tmp_path / "train.yaml", {**_train_settings(tmp_path), "eval": {"sample_timeout_seconds": 60}})

    with pytest.raises(ConfigError, match="eval.sample_timeout_seconds: unknown key; evaluation during") as info:
        load_config(args=["--config", path], config_class=PlatoonTinkerRLTrainerConfig)
    assert "so train.sample_timeout_seconds bounds its sampling calls too" in str(info.value)


def test_a_hosted_model_refuses_the_key_and_names_its_own():
    hosted = HostedConfig(base_url="https://api.openai.com/v1")
    with pytest.raises(ValueError, match=r"inference\.sample_timeout_seconds configure a Tinker policy") as info:
        EvalConfig(model_name="gpt-5.1", hosted=hosted, sample_timeout_seconds=60)
    assert str(info.value).endswith("leave them out (inference.hosted.timeout_s bounds a hosted model's calls)")


# --------------------------------------------------------------------------- which wins


@pytest.mark.parametrize(
    ("key", "variable", "bound", "source"),
    [
        (1800, None, 1800.0, KEY),
        (1800, "300", 1800.0, f"{KEY}, which wins over {VARIABLE}=300"),
        (None, "300", 300.0, f"{VARIABLE}; {KEY} is not set"),
        (None, None, 600.0, f"the default; neither {KEY} nor {VARIABLE} is set"),
    ],
)
def test_the_trainers_policy_samples_under_the_key_else_the_variable_else_600(
    tmp_path, tinker, monkeypatch, key, variable, bound, source
):
    if variable is not None:
        monkeypatch.setenv(VARIABLE, variable)
    config = build_config(PlatoonTinkerRLTrainerConfig, _train_settings(tmp_path, sample_timeout_seconds=key))

    trainer = rl.PlatoonTinkerRLTrainer(config=config, train_dataset=[])

    assert trainer.model_info.llm.sample_timeout_seconds == bound
    assert core.default_gateway().sample_timeout_seconds == bound  # the gateway harness rollouts are served
    assert sample_timeout_note(KEY, key) == f"Sampling timeout: {bound:g} s ({source})"


def test_a_rollout_worker_is_bound_by_the_trainers_bound_not_its_own_environment(tmp_path, tinker, monkeypatch):
    """Each worker request carries the bound the trainer resolved: a worker builds its gateway with it
    (``test_worker_sample_timeout.py`` runs one), so the variable is read once, in the trainer."""

    monkeypatch.setenv(VARIABLE, "300")
    config = build_config(PlatoonTinkerRLTrainerConfig, _train_settings(tmp_path))
    trainer = rl.PlatoonTinkerRLTrainer(config=config, train_dataset=[])
    workflow = GroupRolloutWorkflow(
        rollout_fn=no_rollout,
        get_task_fn=load_task,
        config=WorkflowConfig(rollout_config=RolloutConfig(output_dir=str(tmp_path / "rollouts"))),
        model_info=trainer.model_info,
        log_path=str(tmp_path / "logs"),
    )
    monkeypatch.setenv(VARIABLE, "5")  # what a worker started now would read; it never does

    request = workflow._build_worker_request({"task_id": "t1"}, 0)

    assert request.model.sample_timeout_seconds == 300.0


@pytest.mark.parametrize(("key", "variable", "bound"), [(1800, "300", 1800), (None, "300", 300), (None, None, 600)])
def test_an_evaluation_policy_samples_under_the_key_else_the_variable(tinker, monkeypatch, key, variable, bound):
    if variable is not None:
        monkeypatch.setenv(VARIABLE, variable)
    policy = EvalConfig(model_name=TOY_MODEL_NAME, renderer_name=RENDERER, sample_timeout_seconds=key).policy()
    assert policy is not None and policy.sample_timeout_seconds == key

    # What a worker process of use_subprocesses gets, and installs there.
    pickle.loads(pickle.dumps(policy)).install()

    assert core.default_gateway().sample_timeout_seconds == bound


# --------------------------------------------------------------------------- the commands


def test_platoon_eval_samples_under_the_key_and_logs_which_bound_won(tmp_path, tinker, monkeypatch, caplog):
    monkeypatch.setenv(VARIABLE, "300")
    path = _write(tmp_path / "eval.yaml", _eval_settings(tmp_path, sample_timeout_seconds=1800))

    with caplog.at_level(logging.INFO, logger="platoon.eval.run"):
        report = asyncio.run(arun_eval(["--config", path]))

    assert report["summary"]["success_rate"] == 1.0 and BOUNDS == [1800.0]
    assert (
        "Sampling timeout: 1800 s (inference.sample_timeout_seconds, which wins over TINKER_SAMPLE_TIMEOUT_SECONDS=300)"
        in caplog.messages
    )


@pytest.mark.parametrize("variable", ["0", "half an hour"])
def test_a_bad_variable_stops_platoon_eval_and_platoon_train_before_they_run(tmp_path, tinker, monkeypatch, variable):
    monkeypatch.setenv(VARIABLE, variable)
    evaluation = _write(tmp_path / "eval.yaml", _eval_settings(tmp_path))
    training = _write(tmp_path / "train.yaml", _train_settings(tmp_path))

    with pytest.raises(ConfigError, match=r"The sample timeout \(TINKER_SAMPLE_TIMEOUT_SECONDS\) must be"):
        asyncio.run(arun_eval(["--config", evaluation]))
    with pytest.raises(ConfigError, match=r"The sample timeout \(TINKER_SAMPLE_TIMEOUT_SECONDS\) must be"):
        asyncio.run(arun_tinker_training(["--config", training]))
    assert BOUNDS == [] and tinker == []  # no rollout ran and no Tinker session opened

    # The key wins, so a config that sets it never reads the variable.
    keyed = _write(tmp_path / "keyed.yaml", _eval_settings(tmp_path, sample_timeout_seconds=1800))
    asyncio.run(arun_eval(["--config", keyed]))
    assert BOUNDS == [1800.0]


def test_platoon_rollout_samples_under_the_configs_key(tmp_path, tinker, monkeypatch, capsys):
    monkeypatch.setenv(VARIABLE, "300")
    training = _write(tmp_path / "train.yaml", _train_settings(tmp_path, sample_timeout_seconds=1800))
    evaluation = _write(tmp_path / "eval.yaml", _eval_settings(tmp_path, sample_timeout_seconds=900))
    unset = _write(tmp_path / "unset.yaml", _eval_settings(tmp_path))

    for config in (training, evaluation, unset):
        cli.main(["rollout", config, "--task", "t1", "--output-dir", str(tmp_path / Path(config).stem)])

    assert BOUNDS == [1800.0, 900.0, 300.0]


def test_the_key_needs_a_served_policy(tmp_path, tinker, capsys):
    """Without a renderer no Tinker policy is served, so the key would bound nothing: refused, as the other policy
    settings are (platoon rollout takes one from --renderer)."""

    settings = _eval_settings(tmp_path, sample_timeout_seconds=1800)
    del settings["inference"]["renderer_name"]
    path = _write(tmp_path / "eval.yaml", settings)

    with pytest.raises(ConfigError, match=r"inference\.sample_timeout_seconds configure the policy"):
        asyncio.run(arun_eval(["--config", path]))
    argv = ["rollout", path, "--task", "t1", "--output-dir", str(tmp_path / "run")]
    assert _exit_code(argv) == cli.EXIT_CONFIG
    assert "sample_timeout_seconds configure the policy, which is served only with" in capsys.readouterr().err
    cli.main([*argv[:-1], str(tmp_path / "rendered"), "--renderer", RENDERER])
    assert BOUNDS == [1800.0]
