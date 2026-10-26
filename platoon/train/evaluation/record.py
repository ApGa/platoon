"""The run directory's record of the evaluations during training that completed: ``evaluations.jsonl``.

A run saves the checkpoint of a step before that step's evaluation starts, so a start that resumes at the step cannot
tell from the checkpoints whether an earlier start completed its evaluation.  So the evaluation loop appends a line
here for each evaluation once it has logged the evaluation's metrics::

    {"step": 40, "logged_at": 43, "tasks": 120, "seconds": 3912.4, "time": "2026-10-19T04:21:07+00:00"}

``step`` is the step it evaluated (its ``eval/step``), ``logged_at`` the step training had reached when its metrics
were logged, and ``tasks`` and ``seconds`` its size and how long it took.  A start that resumes reads the steps back
(:func:`completed_evaluation_steps`) and does not evaluate the step it resumes at again when an earlier start
completed that evaluation; one that a crash cut short runs again.  A crash between the metrics and the line leaves a
logged evaluation out, so a start that resumes at its step runs it once more.  The record names steps, not policies:
an evaluation of a step that a later start trained again still counts for it.  A line that does not parse, such as one
a crash cut short, is skipped, and a run directory without the file has completed none: a fresh run, or one that a
platoon without the record started.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from platoon.utils.jsonl import append_jsonl_line, read_jsonl

logger = logging.getLogger(__name__)

# The record, in the run directory, of the evaluations during training that completed.
EVALUATIONS_NAME = "evaluations.jsonl"


def completed_evaluation_steps(run_dir: str | os.PathLike[str]) -> set[int]:
    """The steps whose evaluation completed in the run in ``run_dir``, as its ``evaluations.jsonl`` lists them."""

    return {
        step
        for record in read_jsonl(os.path.join(run_dir, EVALUATIONS_NAME))
        if isinstance(step := record.get("step"), int) and not isinstance(step, bool)
    }


def record_evaluation(path: str | os.PathLike[str], *, step: int, logged_at: int, tasks: int, seconds: float) -> None:
    """Append the evaluation of ``step``, which completed, to the record at ``path``.  A record that cannot be written
    is logged and training goes on: a start that resumes at that step then evaluates it again."""

    record = {
        "step": step,
        "logged_at": logged_at,
        "tasks": tasks,
        "seconds": round(seconds, 1),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        append_jsonl_line(path, json.dumps(record))
    except OSError as exc:
        logger.warning(
            "Could not record the evaluation of step %s in %s: %s; a start that resumes at step %s evaluates it again",
            step,
            path,
            exc,
            step,
        )
