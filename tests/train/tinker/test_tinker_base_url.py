"""A training config's ``tinker_base_url`` is the Tinker server of every Tinker client its commands open: the
trainer's policy sampler as well as its training client, and the policy ``platoon rollout`` serves.

``tinker.ServiceClient`` opens its session as it is constructed, at ``TINKER_BASE_URL`` or else the hosted
service, so a client built without the configured URL reaches another server than the one the config names.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from platoon import cli
from platoon.eval.config_defs import EvalPolicy
from platoon.task import Task
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.tinker import proxy
from platoon.train.trainer import rl as rl_module
from platoon.utils.config import build_config

BASE_URL = "http://self-hosted:8000"


@pytest.fixture
def sessions(monkeypatch) -> list[dict]:
    """The keyword arguments of every ``tinker.ServiceClient`` created: ``{}`` reads ``TINKER_BASE_URL``."""

    created: list[dict] = []

    class ServiceClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def create_sampling_client(self, base_model, model_path=None):
            return object()

    monkeypatch.setattr(proxy.tinker, "ServiceClient", ServiceClient)
    monkeypatch.setattr(proxy, "build_tinker_llm", lambda **kwargs: SimpleNamespace(set_as_default=lambda: None))
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    return created


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["t1"]


def load_task(task_id: str) -> Task:
    return Task(id=task_id, goal="g")


async def rollout(task: Task, config) -> dict:
    return {"id": "c", "trajectories": {"root": {"id": "root", "task": {"id": task.id}, "reward": 1.0, "steps": []}}}


def _settings(**top) -> dict:
    return {
        "train": {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"},
        "eval": {"every": 0},
        "log_path": "logs",
        "environments": [
            {
                "dataset_loader": f"{__name__}:load_tasks",
                "task_loader": f"{__name__}:load_task",
                "rollout": f"{__name__}:rollout",
            }
        ],
        **top,
    }


def test_register_tinker_llm_opens_its_session_on_the_given_server(sessions):
    proxy.register_tinker_llm("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct", base_url=BASE_URL)
    proxy.register_tinker_llm("Qwen/Qwen3-4B-Instruct-2507", "qwen3_instruct")

    assert sessions == [{"base_url": BASE_URL}, {}]


def test_the_trainer_registers_its_policy_on_the_configured_server(sessions):
    config = build_config(PlatoonTinkerRLTrainerConfig, _settings(tinker_base_url=BASE_URL))

    rl_module.PlatoonTinkerRLTrainer(config=config, train_dataset=[])

    assert sessions == [{"base_url": BASE_URL}]


def test_the_recovery_pruner_deletes_on_the_server_the_trainer_trains_on(tmp_path, sessions, monkeypatch):
    """The trainer reads its policy's server from the config once, as it is built: its training session and its
    recovery pruner's session open on that server, also after the config changes."""

    settings = _settings(tinker_base_url=BASE_URL, log_path=str(tmp_path / "logs"), checkpoint={"keep_recovery": 2})
    settings["train"]["max_staleness"] = 1
    config = build_config(PlatoonTinkerRLTrainerConfig, settings)
    no_tasks: Any = []
    trainer = rl_module.PlatoonTinkerRLTrainer(config=config, train_dataset=no_tasks)
    pruners: list[dict] = []
    monkeypatch.setattr(rl_module, "RecoveryCheckpointPruner", lambda **kwargs: pruners.append(kwargs))
    config.tinker_base_url = "http://moved:8000"

    trainer.policies.connect()
    trainer._start_recovery_pruning()
    pruners[0]["service_client_factory"]()

    assert sessions == [{"base_url": BASE_URL}] * 3  # the model gateway's, the training session's and the pruner's


def test_rollout_of_a_training_config_samples_the_configured_server(tmp_path, sessions, capsys):
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(_settings(tinker_base_url=BASE_URL, log_path=str(tmp_path / "logs"))))

    cli.main(["rollout", str(config), "--task", "t1", "--output-dir", str(tmp_path / "run")])

    assert sessions == [{"base_url": BASE_URL}]


def test_an_eval_policy_opens_its_session_on_its_base_url(sessions):
    EvalPolicy(model_name="m", renderer_name="r", base_url=BASE_URL).install()
    EvalPolicy(model_name="m", renderer_name="r").install()  # no base_url: TINKER_BASE_URL, as before

    assert sessions == [{"base_url": BASE_URL}, {}]


def test_without_tinker_base_url_every_client_reads_the_environment_as_before(tmp_path, sessions, capsys):
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump(_settings(log_path=str(tmp_path / "logs"))))

    rl_module.PlatoonTinkerRLTrainer(config=build_config(PlatoonTinkerRLTrainerConfig, _settings()), train_dataset=[])
    cli.main(["rollout", str(config), "--task", "t1", "--output-dir", str(tmp_path / "run")])

    assert sessions == [{}, {}]
