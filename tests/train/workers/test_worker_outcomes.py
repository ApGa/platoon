"""What a rollout worker's exit comes to (``platoon.train.workers.outcomes.classify``), for a worker that left no
response: the kind of its exit, checked in order (out of memory by its exit code or its marker, its own deadline's
marker, a stop on SIGTERM, the parent's deadline without a marker, a crash, a clean exit with nothing readable)."""

from __future__ import annotations

import logging
import os

import pytest
from fakes.workers import worker_request

from platoon.train.workers.outcomes import classify
from platoon.train.workers.protocol import EXIT_RSS_LIMIT, RolloutWorkerRequest, encode_diag


def _request(tmp_path) -> RolloutWorkerRequest:
    return worker_request(tmp_path, "task-r0-outcome", startup_budget_seconds=20.0, rollout_budget_seconds=2.0)


def _disposition(tmp_path, *, returncode, phase=None, timed_out=False) -> str:
    request = _request(tmp_path)
    if phase is not None:
        with open(request.diag_path, "wb") as handle:
            handle.write(encode_diag({"phase": phase}))
    elif os.path.exists(request.diag_path):
        os.unlink(request.diag_path)
    outcome = classify(
        request,
        returncode=returncode,
        killed_by_parent=timed_out,
        timed_out=timed_out,
        launch_seconds=0.0,
        wall_seconds=1.0,
        retried=False,
        worker_pid=4242,
        log_path=None,
    )
    assert outcome.response is None
    return outcome.disposition


@pytest.mark.parametrize(
    ("returncode", "phase", "timed_out", "disposition"),
    [
        (EXIT_RSS_LIMIT, None, False, "rollout_worker_oom"),  # its RSS limit's exit code
        (-9, "rss", False, "rollout_worker_oom"),  # its marker: it SIGKILLed itself over the limit
        (-9, "rss", True, "rollout_worker_oom"),  # out of memory first, also past the parent's deadline
        (-9, "rollout", False, "hard_timeout"),  # its own rollout deadline's marker
        (-9, "startup", False, "hard_timeout"),
        (-15, "terminated", False, "worker_crashed"),  # stopped on a SIGTERM that was not this supervisor's deadline
        (-15, "terminated", True, "hard_timeout"),  # stopped by this supervisor's deadline
        (-9, None, True, "hard_timeout"),  # the parent's deadline, without a marker
        (None, None, False, "worker_crashed"),  # no return code: a forked worker whose fork server died
        (1, None, False, "worker_crashed"),
        (0, None, False, "worker_launch_failed"),  # a clean exit that wrote nothing readable
    ],
)
def test_a_worker_without_a_response_is_classified_by_the_kind_of_its_exit(
    tmp_path, returncode, phase, timed_out, disposition
):
    assert _disposition(tmp_path, returncode=returncode, phase=phase, timed_out=timed_out) == disposition


def test_the_parent_s_deadline_without_a_marker_is_warned_of(tmp_path, caplog):
    """The worker's own alarm fires before the parent's deadline: a worker the parent had to stop without a marker
    says that the timeout ladder is misconfigured."""

    with caplog.at_level(logging.WARNING):
        assert _disposition(tmp_path, returncode=-9, timed_out=True) == "hard_timeout"

    assert "hit the parent deadline without leaving a diagnostic marker" in caplog.text
