"""A rollout worker whose trainer dies stops within its deadline, and calls nothing once it is stopping.

A trainer killed (SIGTERM, then SIGKILL) in the middle of a recursive rollout: its worker got SIGTERM through
PDEATHSIG, deleted its env session and cancelled its rollout -- and ran on, creating subagent conversations and
calling the deleted session, because its harness (OpenHands conversations) swallowed the cancellation.  Nothing but
an external sweep would have stopped it, and under the restart wrapper it would have run next to the restarted
trainer.

These tests run the real supervisor in a stand-in trainer process and the real worker entrypoint, with a recursive
rollout through platoon's own harness path (``harness_rollout``, spawn, the gateway and its tool plane) whose agents
log every model call, tool call, env session and fork (``worker_stub_rollouts``).  The trainer is SIGKILLed mid
rollout.  So are rollouts whose commands run in sessions of their own, out of reach of the worker's process-group
kill: a harness CLI on the rollout host (``test_worker_shutdown_host_commands.py``) and a sandbox command
(``test_worker_shutdown_sandbox_commands.py``); ``test_worker_shutdown_cancellation.py`` covers how a stopping
worker reaches its rollout.  Slow: each process imports the tinker/torch stack, so the scenarios
(``fakes.worker_shutdown``) are spread over files that parallel test runs share out.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import pytest
from fakes.worker_shutdown import (
    EXIT_SLACK_SECONDS,
    STUB_DIR,
    assert_stopped_cleanly,
    kill_trainer_mid_rollout,
    recorded_calls,
    shutdown_request,
)

from platoon.train.workers.entrypoint import TERMINATION_SECONDS
from platoon.train.workers.supervisor import RolloutWorkerSupervisor


def test_a_worker_whose_trainer_is_killed_stops_a_rollout_that_swallows_cancellation(tmp_path):
    """Agents that catch the cancellation and call on.  The worker refuses their calls, and kills its process group
    at its deadline."""

    run = kill_trainer_mid_rollout(tmp_path, "stubborn_recursive_rollout")

    assert_stopped_cleanly(run)
    assert "the rollout did not unwind in time" in run["log"]
    assert run["gone_at"] - run["killed_at"] <= TERMINATION_SECONDS + EXIT_SLACK_SECONDS


def test_a_worker_whose_trainer_is_killed_unwinds_a_cooperative_rollout_at_once(tmp_path):
    run = kill_trainer_mid_rollout(tmp_path, "polite_recursive_rollout")

    assert_stopped_cleanly(run)
    assert re.search(r"the rollout unwound in \d+\.\ds", run["log"]), run["log"][-2000:]
    assert run["gone_at"] - run["killed_at"] < 4.0


def test_a_straggler_kill_stops_a_rollout_that_swallows_cancellation_inside_the_supervisors_grace(tmp_path):
    """The supervisor's own SIGTERM (a straggler kill, a cancelled group): the worker stops itself before the
    supervisor's SIGKILL would have to, and calls nothing meanwhile."""

    request = shutdown_request(tmp_path, "stubborn_recursive_rollout")
    calls_path = tmp_path / "calls.log"
    logs = tmp_path / "logs"
    supervisor = RolloutWorkerSupervisor(
        scratch_dir=str(tmp_path / "ipc"),
        log_dir=str(logs),
        env_overrides={
            "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
            "STUB_CALL_LOG": str(calls_path),
            "STUB_FACTS_PATH": str(tmp_path / "facts.json"),
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
        },
        parent_grace_seconds=600.0,
    )
    assert TERMINATION_SECONDS < supervisor.termination_grace_seconds

    async def scenario() -> tuple[float, float]:
        running = asyncio.ensure_future(supervisor.run(request))
        deadline = time.monotonic() + 300
        while not any(goal.startswith("short-d2") for _, _, goal in recorded_calls(calls_path)):
            assert time.monotonic() < deadline, "the recursive tree never got busy"
            await asyncio.sleep(0.1)
        cancelled_at = time.time()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        return cancelled_at, time.time()

    cancelled_at, reaped_at = asyncio.run(scenario())

    log = (logs / f"{request.request_id}.log").read_text(errors="replace")
    started = re.search(r"\[rollout-worker\] terminating at (\d+\.\d+) \(SIGTERM\)", log)
    assert started and "the rollout did not unwind in time; killing the process group" in log, log[-3000:]
    # The worker ended itself: the supervisor's SIGKILL would have come only after its grace.
    assert reaped_at - cancelled_at < supervisor.termination_grace_seconds
    late = [call for call in recorded_calls(calls_path) if call[0] > float(started.group(1))]
    assert late == [], late[:10]
