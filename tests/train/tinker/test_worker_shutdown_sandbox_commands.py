"""A worker ends what a sandbox command its rollout ran left in the background, when its trainer dies and when it
exits.

The command runs in a session of its own, out of reach of the worker's process-group kill
(``worker_shutdown_scenarios`` runs the scenario; ``test_worker_shutdown.py`` says why these tests exist).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import pytest
from worker_shutdown_scenarios import (
    EXIT_SLACK_SECONDS,
    STUB_DIR,
    _gone_soon,
    _kill_leftover,
    _kill_trainer_once_ready,
    _request,
)

from platoon.train.tinker.workers.entrypoint import TERMINATION_SECONDS
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor


@pytest.mark.parametrize("releases_it", [True, False], ids=["released", "never-released"])
def test_a_worker_whose_trainer_is_killed_ends_what_a_sandbox_command_left_running(tmp_path, releases_it):
    """A sandbox command left a process in the background that ignores SIGTERM.  The env's teardown, which a Harbor
    lease runs in the background and which takes a while (``docker rm`` does), must not be cut short by the worker's
    cancelling again; when the rollout never releases its env, the worker must end the command's group itself."""

    run = _kill_trainer_once_ready(
        tmp_path, "sandbox_command_rollout", **({} if releases_it else {"STUB_STUBBORN": "1"})
    )
    command = run["facts"]["command_pid"]
    try:
        assert _gone_soon(command), "the sandbox command outlived its worker"
        assert run["gone_at"] - run["killed_at"] <= TERMINATION_SECONDS + EXIT_SLACK_SECONDS
        if releases_it:
            assert not Path(run["facts"]["sandbox_dir"]).exists()  # the teardown ran to its end
        else:
            assert "the rollout did not unwind in time" in run["log"]
            assert re.search(r"ended \d+ process group\(s\) the rollout started", run["log"]), run["log"]
    finally:
        _kill_leftover(command)


def test_a_worker_ends_what_its_rollout_left_running_as_it_exits(tmp_path):
    """A rollout that ended without releasing its env: nothing else ends the command it left in the background once
    the worker is gone."""

    request = _request(tmp_path, "sandbox_command_rollout")
    (tmp_path / "state").mkdir()
    supervisor = RolloutWorkerSupervisor(
        scratch_dir=str(tmp_path / "ipc"),
        log_dir=str(tmp_path / "logs"),
        env_overrides={
            "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
            "STUB_FACTS_PATH": str(tmp_path / "facts.json"),
            "STUB_STATE_DIR": str(tmp_path / "state"),
            "STUB_LEAVE_IT": "1",
            "HF_HUB_OFFLINE": "1",
        },
        parent_grace_seconds=600.0,
    )

    outcome = asyncio.run(supervisor.run(request))

    command = json.loads((tmp_path / "facts.json").read_text())["command_pid"]
    try:
        assert outcome.returncode == 0 and outcome.response is not None, outcome
        assert _gone_soon(command), "the sandbox command outlived its worker"
        log = (tmp_path / "logs" / f"{request.request_id}.log").read_text(errors="replace")
        assert re.search(r"ended \d+ process group\(s\) the rollout left running", log), log[-2000:]
    finally:
        _kill_leftover(command)
