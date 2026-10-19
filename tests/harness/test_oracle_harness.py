"""``platoon-oracle`` and ``platoon-noop`` check a dataset through the rollout path training uses: on a sound Harbor
task the oracle (``solution/solve.sh``) scores 1 and the no-op 0, without a single model call."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from harness_fakes import FakeService
from test_harness_rollout import rollout_config, runner

from platoon.gateway import ModelGateway
from platoon.harness.rollout import EmbeddedGateway, run_harness_rollout
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

HELLO = Path(__file__).resolve().parents[1] / "envs" / "fixtures" / "harbor" / "hello-file"


class NoModel:
    """A renderer for a gateway that must never render: these harnesses make no model calls."""

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        raise AssertionError("no model call expected")

    def parse_response(self, tokens):
        raise AssertionError("no model call expected")

    def get_stop_sequences(self):
        return []


def gateway() -> EmbeddedGateway:
    return EmbeddedGateway(ModelGateway(sampler=None, renderer=NoModel(), tokenizer=None, model_name="none"))


def dataset(tmp_path: Path, *, solve: str | None = None, agent_timeout: float | None = None) -> Path:
    task_dir = tmp_path / "dataset" / "hello-file"
    shutil.copytree(HELLO, task_dir)
    if solve is not None:
        (task_dir / "solution" / "solve.sh").write_text(f"#!/bin/bash\n{solve}\n")
    if agent_timeout is not None:
        toml = task_dir / "task.toml"
        toml.write_text(toml.read_text().replace("timeout_sec = 120.0", f"timeout_sec = {agent_timeout}"))
    return task_dir.parent


async def play(tmp_path: Path, harness: str, data: Path) -> dict:
    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            "dataset_dir": str(data),
            "provider": "process",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        harness=harness,
        resolve_task=True,
    )
    collection = await run_harness_rollout(
        route.attach(Task(id="hello-file")), rollout_config(tmp_path), gateway=gateway(), runner=runner(tmp_path)
    )
    (root,) = collection["trajectories"].values()
    return root


@pytest.mark.asyncio
async def test_the_oracle_scores_a_sound_task_and_the_noop_does_not(tmp_path):
    oracle = await play(tmp_path, "platoon-oracle", dataset(tmp_path))
    noop = await play(tmp_path / "noop", "platoon-noop", dataset(tmp_path / "noop"))

    assert (oracle["misc"]["outcome"]["status"], oracle["reward"]) == ("scored", 1.0)
    assert oracle["misc"]["oracle"]["exit_code"] == 0 and oracle["steps"] == []
    assert oracle["misc"]["harness"]["spec"] == "platoon-oracle@1"
    assert (noop["misc"]["outcome"]["status"], noop["reward"]) == ("scored", 0.0) and noop["steps"] == []


@pytest.mark.asyncio
async def test_a_failing_solution_is_still_graded_as_harbor_does(tmp_path):
    root = await play(tmp_path, "platoon-oracle", dataset(tmp_path, solve="echo broken >&2; exit 3"))

    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 0.0)
    assert root["misc"]["oracle"]["exit_code"] == 3 and "broken" in root["misc"]["oracle"]["output"]


@pytest.mark.asyncio
async def test_a_solution_past_the_agent_timeout_is_never_graded(tmp_path):
    root = await play(tmp_path, "platoon-oracle", dataset(tmp_path, solve="sleep 10", agent_timeout=0.5))

    # The rollout ends the oracle at the task's agent timeout, as any harness: a timeout, not graded by default.
    outcome = root["misc"]["outcome"]
    assert (outcome["status"], outcome["reason"], outcome["rewards"]) == ("timeout", "agent_timeout", {})
    assert root["misc"].get("oracle", {"timed_out": True})["timed_out"] is True


@pytest.mark.asyncio
async def test_the_oracle_needs_a_session_with_a_reference_solution(tmp_path):
    FakeService.root = tmp_path / "sandboxes"
    route = EnvironmentRoute(
        name="fake", service="harness_fakes:FakeService", harness="platoon-oracle", resolve_task=True
    )

    collection = await run_harness_rollout(
        route.attach(Task(id="train-0")), rollout_config(tmp_path), gateway=gateway(), runner=runner(tmp_path)
    )

    (root,) = collection["trajectories"].values()
    outcome = root["misc"]["outcome"]
    assert (outcome["status"], outcome["failure_kind"]) == ("error", "harness_crash")
    assert "needs an env session with a reference solution; FakeSession has none" in outcome["reason"]
