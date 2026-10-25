"""Users honored on Docker, end to end (``agent_user: honor``): the user fixtures, their images built with ``docker
build``, each score their reference solution 1.0 and nothing 0.0; the agent's commands run as its user (``docker exec
--user``), with its own ``HOME``, and its ``edit`` tool cannot read the answer that only root may read.

GitHub CI's runners have Docker, and there a missing Docker fails this module rather than skipping it; elsewhere it
skips without one.  It pulls ``python:3.12-slim``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from user_fixtures import TASKS, fetch_and_build, fetched_service, oracle_and_noop, the_agents_view

IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"


def _docker_problem() -> str | None:
    """Why this host cannot run containers; None when it can."""

    if shutil.which("docker") is None:
        return "no docker CLI on PATH"
    info = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=120, check=False)
    return None if info.returncode == 0 else f"docker info failed: {(info.stderr or info.stdout).strip()[:300]}"


PROBLEM = _docker_problem()
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(PROBLEM is not None and not IN_CI, reason=f"needs Docker ({PROBLEM})"),
]


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if PROBLEM is not None:  # in CI only: never skipped there
        pytest.fail(f"GitHub CI runs this module on Docker, which this runner cannot run: {PROBLEM}")
    return fetched_service(tmp_path, monkeypatch, provider="docker", provider_options={})


async def test_the_user_fixtures_score_their_oracles_and_no_ops_as_docker_runs_them(service, tmp_path: Path):
    assert await fetch_and_build(service, builder="docker", scratch=tmp_path / "scratch") == sorted(TASKS)
    await service.check_tasks("train", list(TASKS))

    for name in TASKS:
        oracle, noop = await oracle_and_noop(service, name)
        assert (oracle.status, oracle.rewards) == ("scored", {"reward": 1.0}), (name, oracle.metadata)
        assert (noop.status, noop.rewards) == ("scored", {"reward": 0.0}), (name, noop.metadata)

    view = await the_agents_view(service)
    assert view["who"] == "1000\n/srv/agent", view
    assert view["edit_error"] == "True" and "Permission denied" in view["edit"], view
    assert "occurrence" not in view["edit"]  # nothing of the file's contents reaches the agent
    listed = ["docker", "ps", "--all", "--filter", "label=platoon.env=harbor", "--format", '{{.Label "platoon.task"}}']
    left = subprocess.run(listed, capture_output=True, text=True, check=True).stdout.split()
    assert not [task for task in left if task.removeprefix("platoon/") in TASKS]  # every session's container is gone
