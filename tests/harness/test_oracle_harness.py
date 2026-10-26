"""``platoon-oracle`` and ``platoon-noop`` check a dataset through the rollout path training uses: on a sound Harbor
task the oracle (``solution/solve.sh``) scores 1 and the no-op 0, without a single model call."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from harness_fakes import FakeService
from no_model import no_model_gateway
from site_cache import use_site_cache
from test_harness_rollout import rollout_config, runner

from platoon.envs.harbor.fetch import fetch, fetch_config
from platoon.harness.rollout import run_harness_rollout
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

HELLO = Path(__file__).resolve().parents[1] / "envs" / "fixtures" / "harbor" / "hello-file"
STACKS = Path(__file__).resolve().parents[1] / "envs" / "fixtures" / "harbor-stacks" / "tasks"


def dataset(tmp_path: Path, *, solve: str | None = None, agent_timeout: float | None = None) -> Path:
    task_dir = tmp_path / "dataset" / "hello-file"
    shutil.copytree(HELLO, task_dir)
    if solve is not None:
        (task_dir / "solution" / "solve.sh").write_text(f"#!/bin/bash\n{solve}\n")
    if agent_timeout is not None:
        toml = task_dir / "task.toml"
        toml.write_text(toml.read_text().replace("timeout_sec = 120.0", f"timeout_sec = {agent_timeout}"))
    return task_dir.parent


async def play(tmp_path: Path, harness: str, data: Path | None, *, task_id: str = "hello-file", **source: Any) -> dict:
    """One rollout of ``harness`` on the task ``task_id`` of the dataset directory ``data``, or of the fetched dataset
    that ``source`` names (``dataset``, ``refuse``, ...), on the process provider."""

    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            **(source or {"dataset_dir": str(data)}),
            "provider": "process",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
        },
        harness=harness,
        resolve_task=True,
    )
    collection = await run_harness_rollout(
        route.attach(Task(id=task_id)), rollout_config(tmp_path), gateway=no_model_gateway(), runner=runner(tmp_path)
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


def stack_repository(root: Path) -> str:
    """A git repository at ``root`` holding the stack fixture ``app-kv`` below ``tasks/`` (main stores a value through
    a key-value sidecar, which records the write; the tests read both back): the ``git+`` dataset of that directory at
    its commit."""

    shutil.copytree(STACKS / "app-kv", root / "tasks" / "app-kv")

    def git(*args: str) -> str:
        command = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git("commit", "-q", "-m", "tasks")
    return f"git+{root.as_uri()}@{git('rev-parse', 'HEAD')}#tasks"


@pytest.mark.asyncio
async def test_a_stack_fetched_from_git_scores_with_the_oracle_and_not_with_the_noop(tmp_path, monkeypatch):
    """A multi-container task fetched from a git repository, which ``refuse: {sidecars: false}`` locks as a stack, runs
    through the rollout path on the process provider with its sidecar: the oracle scores 1 and the no-op 0, both
    graded (a sidecar that a probe found down would make them invalid)."""

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "scratch"))
    use_site_cache(monkeypatch, tmp_path / "site")
    source: dict[str, Any] = {"dataset": stack_repository(tmp_path / "upstream"), "refuse": {"sidecars": False}}
    lock = fetch(fetch_config(**source))  # as the trainer fetches it when it lists the tasks
    task = lock.get("app-kv")
    assert (lock.source["kind"], task.flags["stack"], task.refused) == ("git", "sidecars", None)

    oracle = await play(tmp_path / "oracle", "platoon-oracle", None, task_id="app-kv", **source)
    noop = await play(tmp_path / "noop", "platoon-noop", None, task_id="app-kv", **source)

    assert (oracle["misc"]["outcome"]["status"], oracle["reward"]) == ("scored", 1.0), oracle["misc"]["outcome"]
    assert oracle["misc"]["oracle"]["exit_code"] == 0 and oracle["steps"] == []
    assert (noop["misc"]["outcome"]["status"], noop["reward"]) == ("scored", 0.0), noop["misc"]["outcome"]
    for run in ("oracle", "noop"):
        assert list((tmp_path / run / "sandboxes").iterdir()) == []  # the stack's services and volumes are gone


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
        route.attach(Task(id="train-0")), rollout_config(tmp_path), gateway=no_model_gateway(), runner=runner(tmp_path)
    )

    (root,) = collection["trajectories"].values()
    outcome = root["misc"]["outcome"]
    assert (outcome["status"], outcome["failure_kind"]) == ("error", "harness_crash")
    assert "needs an env session with a reference solution; FakeSession has none" in outcome["reason"]
