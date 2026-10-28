"""``platoon eval``'s policy: without one, refused for harness rollouts before anything runs and fine for custom
ones; a Tinker one opens its session with the Tinker SDK's telemetry off.  And an evaluation whose node services fail
stops."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from platoon.eval.config_defs import EvalConfig, EvalPolicy
from platoon.eval.run import arun_eval, harness_rollout_environments
from platoon.task import Task
from platoon.train.components import HARNESS_ROLLOUT
from platoon.utils.config import ConfigError


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["solved"]


def load_task(task_id: str) -> Task:
    return Task(id=task_id, goal=f"solve {task_id}")


async def scored_rollout(task: Task, config) -> dict:
    return {"id": "c", "trajectories": {"root": {"id": "root", "task": {"id": task.id}, "reward": 1.0, "steps": []}}}


# A remote service nobody serves: a check that runs after listing tasks would fail on the connection instead.
UNREACHABLE = {"service": "ors+http://127.0.0.1:9/Nothing", "harness": "platoon-react"}
CUSTOM = {"dataset_loader": f"{__name__}:load_tasks", "task_loader": f"{__name__}:load_task"}


def _config(tmp_path: Path, environment: dict, inference: dict | None = None, **top) -> list[str]:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {"model_name": "m", "output_dir": str(tmp_path / "results"), **(inference or {})},
                "environments": [{"name": "env", **environment}],
                **top,
            }
        )
    )
    return ["--config", str(path)]


def test_a_harness_environment_without_a_renderer_is_refused_before_anything_runs(tmp_path):
    with pytest.raises(ConfigError, match=r"environments \['env'\] roll out with harness_rollout"):
        asyncio.run(arun_eval(_config(tmp_path, UNREACHABLE)))
    assert not list((tmp_path / "results").rglob("metadata.json"))


def test_a_custom_rollout_without_a_renderer_runs(tmp_path):
    report = asyncio.run(arun_eval(_config(tmp_path, {**CUSTOM, "rollout": f"{__name__}:scored_rollout"})))
    assert report["summary"]["success_rate"] == 1.0


def test_a_report_needs_no_renderer(tmp_path):
    report = asyncio.run(arun_eval(_config(tmp_path, UNREACHABLE, stage="report")))
    assert report["summary"]["total_rollouts"] == 0


def test_which_environments_roll_out_with_harness_rollout():
    from platoon.eval.run import PlatoonEvalConfig

    config = PlatoonEvalConfig(
        inference=EvalConfig(model_name="m"),
        environments=[
            {"name": "harness", **UNREACHABLE},
            {"name": "named", **UNREACHABLE, "eval_rollout": HARNESS_ROLLOUT},
            {"name": "custom-eval", **UNREACHABLE, "eval_rollout": f"{__name__}:scored_rollout"},
            {"name": "custom", **CUSTOM, "rollout": f"{__name__}:scored_rollout"},
        ],
    )
    assert harness_rollout_environments(config, "test") == ["harness", "named"]
    assert harness_rollout_environments(config, "train") == ["harness", "named", "custom-eval"]


@pytest.mark.parametrize(
    "setting",
    [
        {"model_path": "tinker://run/final"},
        {"renderer_kwargs": {"a": 1}},
        {"context_window_length": 4096},
        {"sample_timeout_seconds": 1800},
    ],
)
def test_policy_settings_need_a_renderer(tmp_path, setting):
    custom = {**CUSTOM, "rollout": f"{__name__}:scored_rollout"}  # would run without a policy
    with pytest.raises(ConfigError, match=rf"inference\.{next(iter(setting))} configure the policy"):
        asyncio.run(arun_eval(_config(tmp_path, custom, inference=setting)))
    assert not list((tmp_path / "results").rglob("metadata.json"))
    # The section itself loads either way: platoon rollout can give it a renderer with --renderer.
    assert EvalConfig(model_name="m", **setting).policy() is None
    assert EvalConfig(model_name="m", renderer_name="qwen3", **setting).policy() is not None


def test_a_tinker_policy_opens_its_session_with_the_sdks_telemetry_off(monkeypatch):
    """With telemetry on, the Tinker SDK reports a call that fails for good with a SESSION_END event, which a
    self-hosted gateway (miles-tinker) takes for the client's exit: it reclaims the session, and every later call of
    the process fails ("unknown sampling session"). In Harbor gate G2 (2026-10-17), one sample the engine refused
    ended an evaluation worker's session that way. The SDK reads TINKER_TELEMETRY as a session opens."""

    from platoon.train.tinker import proxy

    telemetry: list[str | None] = []

    class ServiceClient:
        def __init__(self, **kwargs):
            telemetry.append(os.environ.get("TINKER_TELEMETRY"))

        def create_sampling_client(self, base_model, model_path=None):
            return object()

    monkeypatch.setattr(proxy.tinker, "ServiceClient", ServiceClient)
    monkeypatch.setattr(proxy, "build_tinker_llm", lambda **kwargs: SimpleNamespace(set_as_default=lambda: None))
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    monkeypatch.setenv("TINKER_TELEMETRY", "1")  # what the run's environment asked for

    EvalPolicy(model_name="m", renderer_name="r", base_url="http://self-hosted:8000").install()

    assert telemetry == ["0"]


def test_an_evaluation_whose_node_services_fail_starts_no_further_rollout_and_stops_them(tmp_path, monkeypatch):
    from platoon.train.startup import RolloutStartup

    closed: list[bool] = []

    class Startup:  # the rollouts' startup of a run whose node services failed as it began
        node_services = object()

        async def ready(self) -> None:
            pass

        def on_node_service_failure(self, callback) -> None:
            callback("the judge could not be started again")

        async def close(self) -> str:
            closed.append(True)
            return "the judge could not be started again"

    async def begin(environments, workflows=()):
        return Startup()

    monkeypatch.setattr(RolloutStartup, "begin", staticmethod(begin))
    with pytest.raises(RuntimeError, match=r"0 rollout\(s\), 1 not started: the judge could not be started again"):
        asyncio.run(arun_eval(_config(tmp_path, {**CUSTOM, "rollout": f"{__name__}:scored_rollout"})))
    assert closed == [True]  # its node services stopped before the evaluation ended
