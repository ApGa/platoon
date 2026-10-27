"""The rollout workers' telemetry, recorded as each member's worker is reaped (``rollout_isolation: process``): a scalar
per disposition, the cost, and the R3 routing-id coverage warnings.

The functions here that take the workflow first are its methods still: ``GroupRolloutWorkflow`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from platoon.train.workers.supervisor import WorkerOutcome
from platoon.train.workers.telemetry import record_worker_cell_guard

if TYPE_CHECKING:
    from platoon.train.workflow.group import GroupRolloutWorkflow

logger = logging.getLogger(__name__)


# Guards each workflow's ``_RoutingIdState``, so that one rollout at a time reads it, logs what it says and records what
# the rollout got. Only threads need this: asyncio tasks cannot interleave in that code, which never awaits, and platoon
# reports each rollout from its event loop's thread; a custom workflow may report from threads of its own.
_routing_id_lock = threading.Lock()
# At most this many of the rollouts without routing ids are named when a later rollout is the first to get any.
_NAMED_ROLLOUTS = 3


@dataclass
class _RoutingIdState:
    """What the rollouts of one workflow, which is one stats scope of one start of training, got of routing ids (R3).

    ``seen``: a rollout got a routing id for at least one of its model calls.  ``without``: how many rollouts with model
    calls got none before one did, and ``first_without`` the request ids of the first ``_NAMED_ROLLOUTS`` of them.
    """

    seen: bool = False
    without: int = 0
    first_without: list[str] = field(default_factory=list)


def _warn_of_missing_routing_ids(workflow: "GroupRolloutWorkflow", request_id: str, present: int, calls: int) -> None:
    """Warn about each rollout whose routing ids (R3) were dropped, and once about a service that sends none.

    A server with rollout routing replay sends a routing id with each sampled sequence. A datum whose last model
    call got none has no ``routing_handle``, which is an error nowhere in the stack: for a mixture-of-experts model,
    only a quietly worse gradient. So a rollout that got routing ids for some of its ``calls`` model calls but not
    all, or for none of them after the service has sent some, is warned about on its own. A service that has sent
    none, such as a dense model's, is warned about once, at the first rollout, not for each one. The first rollout
    to get any after rollouts without them says so and names the first few: either the service began to send
    routing ids, or those rollouts lost theirs. A rollout logs one of these warnings at most, and the metric
    ``rollout_worker_routing_id_coverage`` counts every rollout either way.

    What the rollouts got so far is this workflow's own, and ``platoon train`` builds one workflow for each stats
    scope (``train``, ``eval``) at each start of training, a resume or a restart (``watchdog.max_restarts``)
    included. So each scope's rollouts are known from the current start of training on, and another run in the
    same process keeps its own.
    """

    scope = workflow.stats_scope
    with _routing_id_lock:
        # Made here rather than in ``__init__``, since custom workflows may build this class with ``__new__``.
        state = workflow.__dict__.setdefault("_routing_id_state", _RoutingIdState())
        if present > 0 and not state.seen and state.without:
            named = ", ".join(state.first_without)
            if state.without > len(state.first_without):
                named += f" and {state.without - len(state.first_without)} more"
            logger.warning(
                "rollout worker %s got routing ids for %s/%s model calls, the first %s rollout in this start of "
                "training to get any, after %s rollout(s) without them (%s): either the model service began to "
                "send routing ids, or those rollouts lost theirs, and their datums have no routing_handle (R3). "
                "From now on, each rollout without routing ids is logged",
                request_id,
                present,
                calls,
                scope,
                state.without,
                named,
            )
        elif 0 < present < calls:
            logger.warning(
                "rollout worker %s returned routing ids for only %s/%s model calls (R3 routing_handle coverage %.3f)",
                request_id,
                present,
                calls,
                present / calls,
            )
        elif present == 0 and state.seen:
            logger.warning(
                "rollout worker %s returned routing ids for only 0/%s model calls (R3 routing_handle coverage "
                "0.000), though the model service sent routing ids to earlier %s rollouts",
                request_id,
                calls,
                scope,
            )
        elif present == 0 and not state.without:
            logger.warning(
                "rollout worker %s got no routing ids for its %s model calls, nor did any %s rollout before it in "
                "this start of training: the model service sends none. That is expected for a dense model, but "
                "means no rollout routing replay (R3) for a mixture-of-experts model. This is logged once for "
                "each start of training, not for each rollout; the metric %s/rollout_worker_routing_id_coverage "
                "shows each rollout's coverage",
                request_id,
                calls,
                scope,
                scope,
            )
        if present > 0:
            state.seen = True
        elif not state.seen:
            state.without += 1
            if len(state.first_without) < _NAMED_ROLLOUTS:
                state.first_without.append(request_id)


def _record_worker_telemetry(workflow: "GroupRolloutWorkflow", outcome: WorkerOutcome) -> None:
    """One scalar per disposition, plus the cost and the R3 coverage guard.

    Never raises: a metric must not be able to fail a rollout.
    """

    try:
        # Every counter is emitted on EVERY member, as 0.0 or 1.0.  The
        # stats tracker averages a scalar over the calls in a step and
        # publishes the call count alongside it, so this reads as "fraction
        # of workers that hard-timed-out" with ``__count`` as the number of
        # workers.  Emitting only on failure would make the value a
        # constant 1.0 and hide the denominator.
        scalars: dict[str, float] = {
            "rollout_worker_wall_seconds": outcome.wall_seconds,
            "rollout_worker_launch_seconds": outcome.launch_seconds,
            "rollout_worker_killed": float(outcome.killed_by_parent),
            "rollout_worker_launch_retried": float(outcome.retried),
        }
        counted = {
            "hard_timeout": "rollout_worker_hard_timeout",
            "worker_crashed": "rollout_worker_crashed",
            "rollout_worker_oom": "rollout_worker_oom",
            "worker_launch_failed": "rollout_worker_launch_failed",
        }
        for disposition, name in counted.items():
            scalars[name] = float(outcome.disposition == disposition)
        response = outcome.response
        if response is not None:
            scalars["rollout_worker_startup_seconds"] = response.startup_seconds
            scalars["rollout_worker_peak_rss_bytes"] = float(response.peak_rss_bytes)
            if response.model_calls:
                coverage = response.routing_id_present / float(response.model_calls)
                scalars["rollout_worker_routing_id_coverage"] = coverage
            if not response.extra_allow_verified:
                logger.warning(
                    "rollout worker %s did not verify the R3 extra='allow' patch",
                    outcome.request_id,
                )
            if not response.telemetry_disabled:
                logger.error(
                    "rollout worker %s ran with tinker telemetry ENABLED; a shadow client can "
                    "end the trainer's gateway session",
                    outcome.request_id,
                )
            record_worker_cell_guard(workflow.stats_scope, response.cell_guard)
            workflow._monitor().record_rollout_timing(response.timing, startup_offset=outcome.launch_seconds)
            workflow._monitor().record_worker_peak_rss(response.peak_rss_bytes)
        elif outcome.diag:
            # A SIGKILLed worker writes its cumulative cell-guard counters
            # into the marker precisely because it cannot write a response;
            # without this the metric would read zero exactly when a cell
            # was killed for hanging.
            record_worker_cell_guard(workflow.stats_scope, outcome.diag)
        workflow.tracker.scalar(**scalars)
        logger.info(
            "rollout worker %s pid=%s disposition=%s rc=%s launch=%.1fs wall=%.1fs log=%s",
            outcome.request_id,
            outcome.worker_pid,
            outcome.disposition,
            outcome.returncode,
            outcome.launch_seconds,
            outcome.wall_seconds,
            outcome.log_path,
        )
        if response is not None and response.model_calls:
            # Last, so that a failure in it costs the rollout none of the telemetry above.
            workflow._warn_of_missing_routing_ids(outcome.request_id, response.routing_id_present, response.model_calls)
    except Exception:  # pragma: no cover - telemetry must never break a rollout
        logger.debug("failed to record rollout worker telemetry", exc_info=True)
