"""What a rollout worker's exit comes to: the outcome its supervisor reports, and removing what it left behind.

Once :class:`~platoon.train.workers.supervisor.RolloutWorkerSupervisor` has reaped a worker, it classifies it here,
response-first (below): a readable response for the worker's request is the outcome, whatever the return code or
whoever signalled it; else the kind of exit decides the disposition: out of memory, a hard timeout, a stop on SIGTERM,
a crash, or a clean exit that wrote nothing readable.  Then it removes what the worker wrote beside its request.  The
workflow turns the :class:`WorkerOutcome` into a member of the rollout's group.

Classification is **response-first**
====================================

A worker needs ~1.9 s of interpreter teardown *after* it ``os.replace``s its
response, and during that window its parent task is still not done and can be
selected as a straggler.  Classifying on ``(returncode, we_killed_it)`` would
therefore discard a complete rollout's datums because its group's tail grace
expired two seconds after it finished.  So the supervisor always tries to read
the response first; a readable response with a matching protocol wins
regardless of the return code or who sent the signal, and the kill is recorded
separately as ``rollout_worker_killed``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from platoon.train.workers.protocol import (
    EXIT_RSS_LIMIT,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    load_response,
    read_diag,
    unlink_quietly,
)

logger = logging.getLogger(__name__)

__all__ = ["WorkerOutcome", "classify"]


@dataclass(frozen=True)
class WorkerOutcome:
    """Everything the workflow needs to turn one worker into a group member."""

    response: RolloutWorkerResponse | None
    returncode: int | None
    disposition: str
    killed_by_parent: bool
    diag: dict | None
    launch_seconds: float
    wall_seconds: float
    retried: bool
    worker_pid: int | None
    log_path: str | None = None
    request_id: str = ""


def classify(
    request: RolloutWorkerRequest,
    *,
    returncode: int | None,
    killed_by_parent: bool,
    timed_out: bool,
    launch_seconds: float,
    wall_seconds: float,
    retried: bool,
    worker_pid: int,
    log_path: str | None,
) -> WorkerOutcome:
    """The outcome of the worker of ``request``, which has exited: its response's status when it wrote a readable
    response for the request, else the disposition of its exit (:func:`_exit_disposition`)."""

    response = _read_response(request)
    diag = read_diag(request.diag_path)
    if response is not None:
        # Response-first: a complete rollout that was signalled during its
        # ~1.9 s of interpreter teardown still counts, and the kill is
        # recorded separately.
        disposition = response.status
    else:
        disposition = _exit_disposition(request, returncode=returncode, diag=diag, timed_out=timed_out)
    return WorkerOutcome(
        response=response,
        returncode=returncode,
        disposition=disposition,
        killed_by_parent=killed_by_parent,
        diag=diag,
        launch_seconds=launch_seconds,
        wall_seconds=wall_seconds,
        retried=retried,
        worker_pid=worker_pid,
        log_path=log_path,
        request_id=request.request_id,
    )


def _read_response(request: RolloutWorkerRequest) -> RolloutWorkerResponse | None:
    """The response the worker wrote for ``request``; None for none, one that cannot be read, or one for another
    request (each of the last two logged)."""

    try:
        if os.path.exists(request.response_path):
            candidate = load_response(request.response_path)
            if candidate.request_id == request.request_id:
                return candidate
            logger.warning(
                "rollout worker response at %s is for request %s, not %s; ignoring",
                request.response_path,
                candidate.request_id,
                request.request_id,
            )
    except Exception:
        logger.warning(
            "rollout worker %s left an unreadable response at %s",
            request.request_id,
            request.response_path,
            exc_info=True,
        )
    return None


def _exit_disposition(
    request: RolloutWorkerRequest, *, returncode: int | None, diag: dict | None, timed_out: bool
) -> str:
    """The disposition of a worker that left no response, by the kind of its exit, checked in this order: out of
    memory (its RSS limit's exit code or marker), a hard timeout (its own deadline's marker), a stop on SIGTERM (its
    ``terminated`` marker), the parent's deadline without a marker, a crash (no return code, or one other than 0), and
    a clean exit that wrote nothing readable."""

    if returncode == EXIT_RSS_LIMIT or (diag and diag.get("phase") == "rss"):
        return "rollout_worker_oom"
    if diag and diag.get("phase") in ("startup", "rollout"):
        return "hard_timeout"
    if diag and diag.get("phase") == "terminated":
        # It stopped on a SIGTERM (this supervisor's, or its trainer's death through PDEATHSIG) and SIGKILLed
        # its own process group at the end: a crash as far as the group is concerned, unless it was this
        # supervisor's deadline that stopped it.
        return "hard_timeout" if timed_out else "worker_crashed"
    if timed_out:
        # The child's own alarm should have fired first; if it did not, the
        # timeout ladder is misconfigured.
        logger.warning(
            "rollout worker %s hit the parent deadline without leaving a diagnostic marker; "
            "check that the child budget is below the parent budget",
            request.request_id,
        )
        return "hard_timeout"
    if returncode != 0:  # none (a forked worker whose fork server died), or a failing one
        return "worker_crashed"
    # Exited cleanly but wrote nothing readable: treat as a launch
    # failure, never as a partial unpickle.
    return "worker_launch_failed"


def _remove_worker_files(request: RolloutWorkerRequest) -> None:
    """Delete what a worker writes beside its request: its response, diagnostics, readiness and, for a request that
    asks for it, first-call marker."""

    for path in (request.response_path, request.diag_path, request.ready_path):
        unlink_quietly(path)
    if request.signal_first_call:
        unlink_quietly(request.first_call_path)


def _remove_event_temporaries(request: RolloutWorkerRequest, pid: int) -> None:
    """Delete the hidden event files of a worker that was killed while writing its rollout's events."""

    output_dir = (request.rollout_config or {}).get("output_dir")
    if not output_dir:
        return
    # Imported here, for a run that writes events: importing the supervisor does not import the viewer's event sinks.
    from platoon.visualization.event_sinks import remove_temporary_events

    try:
        removed = remove_temporary_events(output_dir, pid)
    except OSError:
        return
    if removed:
        logger.info("removed %d temporary event files rollout worker %s left behind", removed, request.request_id)
