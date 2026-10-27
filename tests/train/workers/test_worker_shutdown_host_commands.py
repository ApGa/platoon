"""A worker whose trainer dies ends the harness CLI its rollout started on the rollout host, in a session of its own.

The command runs out of reach of the worker's process-group kill (``fakes.worker_shutdown`` runs the scenario;
``test_worker_shutdown.py`` says why these tests exist).
"""

from __future__ import annotations

import re

import pytest
from fakes.worker_shutdown import EXIT_SLACK_SECONDS, gone_soon, kill_leftover, kill_trainer_once_ready

from platoon.train.workers.entrypoint import TERMINATION_SECONDS


@pytest.mark.parametrize("holds_its_loop", [False, True], ids=["cancelled", "holding-its-loop"])
def test_a_worker_whose_trainer_is_killed_ends_a_host_command_that_ignores_sigterm(tmp_path, holds_its_loop):
    """A harness CLI on the rollout host runs in a session of its own.  Cancelled, its cleanup (SIGTERM, a 5 s grace,
    SIGKILL) must run to its end although the worker cancels again a second later; when the rollout holds its loop,
    the worker's alarm must take the command with it."""

    run = kill_trainer_once_ready(
        tmp_path, "host_command_rollout", **({"STUB_HOLD_LOOP": "1"} if holds_its_loop else {})
    )
    command = run["facts"]["command_pid"]
    try:
        assert gone_soon(command), "the host command outlived its worker"
        assert run["gone_at"] - run["killed_at"] <= TERMINATION_SECONDS + EXIT_SLACK_SECONDS
        if not holds_its_loop:
            assert (run["state"] / "command.term").exists()  # asked to stop before it was killed
            assert re.search(r"the rollout unwound in \d+\.\ds; killing the process group", run["log"]), run["log"]
    finally:
        kill_leftover(command)
