"""How a stopping worker reaches its rollout: it cancels again a rollout that swallowed the first cancellation, and
its alarm ends one that holds the event loop.

``fakes.worker_shutdown`` runs the scenario; ``test_worker_shutdown.py`` says why these tests exist.
"""

from __future__ import annotations

import re

from fakes.worker_shutdown import kill_trainer_once_ready

from platoon.train.workers.entrypoint import RECANCEL_AFTER_SECONDS, TERMINATION_BACKSTOP_SECONDS, TERMINATION_SECONDS
from platoon.train.workers.protocol import read_diag


def test_a_worker_cancels_again_a_rollout_that_swallowed_the_first_cancellation(tmp_path):
    """One cancellation does not stop a rollout that catches it and goes on: the worker cancels it again a second
    later, so it unwinds and runs its own cleanup instead of being killed at the deadline."""

    run = kill_trainer_once_ready(tmp_path, "swallow_the_first_cancellation")

    assert (run["state"] / "cancelled once").exists()
    assert (run["state"] / "unwound").exists(), run["log"][-2000:]
    assert re.search(r"the rollout unwound in \d+\.\ds; killing the process group", run["log"]), run["log"][-2000:]
    assert run["gone_at"] - run["killed_at"] < RECANCEL_AFTER_SECONDS + 3.0


def test_a_worker_whose_rollout_holds_its_loop_is_ended_by_its_alarm_at_the_deadline(tmp_path):
    """A rollout that holds the event loop in a blocking call gets no cancellation, and the loop never starts the
    worker's termination: the alarm armed as the worker began stopping ends it at its deadline, ahead of the monitor
    thread's backstop.  Its SIGTERM handler arms it, or the monitor thread does when its check of ``getppid()`` sees
    the trainer gone first (``fakes.worker_shutdown.kill_trainer_mid_rollout`` says why either can)."""

    run = kill_trainer_once_ready(tmp_path, "hold_the_loop")

    elapsed = run["gone_at"] - run["killed_at"]
    # The monitor thread's backstop kills TERMINATION_BACKSTOP_SECONDS after the deadline at the earliest: within a
    # second of the deadline only the alarm can have ended it.
    assert TERMINATION_BACKSTOP_SECONDS > 1.0
    assert TERMINATION_SECONDS - 0.5 < elapsed < TERMINATION_SECONDS + 1.0, elapsed
    assert "terminating at" not in run["log"]  # the loop never ran the termination
    diag = read_diag(run["diag"])
    assert diag is not None and diag["phase"] == "terminated", diag
    assert diag["reason"] in ("SIGTERM", "its parent is gone"), diag
