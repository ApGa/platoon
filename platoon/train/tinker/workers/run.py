"""One rollout's body -- the single implementation both hosts use.

``GroupRolloutWorkflow.arun_episode_single`` used to inline this.  It now
builds a :class:`RolloutWorkerRequest` and hands it either to this function
directly (``rollout_isolation: inprocess``) or to a worker process that calls
exactly this function (``rollout_isolation: process``).  **One code path, two
hosts** is what makes in-process mode a real fallback instead of a divergent
branch that rots.

The branch order below reproduces the pre-split behaviour exactly, including
``add_datum_funnel`` immediately after conversion.  See
:mod:`platoon.train.tinker.workers.workload` for why omitting it is total data
loss rather than a metrics regression.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import Any, Callable

from platoon.config_defs import RolloutConfig
from platoon.train.tinker.proxy import TinkerLLMProxySession
from platoon.train.tinker.rollout_metrics import RolloutTimeline, timeline_scope
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    resolve_ref,
)
from platoon.train.tinker.workers.workload import add_datum_funnel, workload_from_raw_rollout
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_REWARD_PROCESSOR",
    "cell_guard_cumulative",
    "run_rollout_request",
]

#: The environment marker the supervisor sets in every worker.  Anything that
#: mutates process-global interpreter state keys off this, so the invariant is
#: asserted where it is used and not merely where it is configured.
WORKER_ENV_MARKER = "PLATOON_TINKER_ROLLOUT_WORKER"


def default_reward_processor(trajectory: dict) -> tuple[float, dict]:
    """The historical default: the trajectory's own reward, no components."""

    return trajectory["reward"], {}


#: Identity used by the workflow to set ``ConversionPolicy.reward_processor_is_default``.
DEFAULT_REWARD_PROCESSOR: Callable[[dict], tuple[float, dict]] = default_reward_processor


def in_rollout_worker() -> bool:
    return os.environ.get(WORKER_ENV_MARKER) == "1"


def cell_guard_cumulative(*, at_exit: bool = False) -> dict[str, int]:
    """Cumulative programmatic-tool-calling counters for this process.

    ``CellGuardRegistry.counts()`` recomputes ``ptc_cells_stuck`` /
    ``ptc_threads_leaked`` / ``ptc_cells_running`` from the live set and
    documents them as *current, not cumulative*.  Summing those gauges across
    short-lived workers would invert the metric: healthy workers all report 0
    at exit, and the one worker that had a stuck cell is exactly the one that
    gets SIGKILLed and writes no response.  So workers forward the genuinely
    cumulative counters, plus the gauges sampled **once at exit** under
    distinct ``*_at_exit`` names.  Never raises.
    """

    try:
        from platoon.openreward.cell_guard import snapshot

        counts = {key: int(value) for key, value in snapshot().items()}
    except Exception:
        return {}
    forwarded = {
        "ptc_cells_started": counts.get("ptc_cells_started", 0),
        "ptc_cells_finished": counts.get("ptc_cells_finished", 0),
    }
    if at_exit:
        forwarded["ptc_cells_stuck_at_exit"] = counts.get("ptc_cells_stuck", 0)
        forwarded["ptc_threads_leaked_at_exit"] = counts.get("ptc_threads_leaked", 0)
    return forwarded


def _routing_id_present(interactions: dict) -> int:
    present = 0
    for interaction in interactions.values():
        if getattr(interaction, "routing_id", None) is not None:
            present += 1
    return present


def _torch_threads() -> int:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return 0


def _peak_rss_bytes() -> int:
    try:
        import resource

        # ru_maxrss is kilobytes on Linux.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def _resolve_reward_processor(request: RolloutWorkerRequest) -> Callable[[dict], tuple[float, dict]]:
    policy = request.conversion
    if policy.reward_processor_is_default:
        return DEFAULT_REWARD_PROCESSOR
    ref = policy.reward_processor_ref
    if not ref:
        # Never fall back to the default: substituting
        # ``lambda traj: (traj["reward"], {})`` for a real reward processor
        # would change task_reward, root_rewards_dict and every advantage with
        # no error and entirely plausible metrics.
        raise ValueError(
            "conversion policy declares a non-default reward processor but carries no reference"
        )
    return resolve_ref(ref)


async def run_rollout_request(
    request: RolloutWorkerRequest,
    llm: Any = None,
    *,
    startup_seconds: float = 0.0,
    extra_allow: bool = False,
    rollout_fn: Callable | None = None,
    reward_processor: Callable[[dict], tuple[float, dict]] | None = None,
) -> RolloutWorkerResponse:
    """Run one rollout and convert it, returning a serializable outcome.

    ``llm`` is the already-registered ``TinkerLLM`` of whichever process hosts
    the rollout; the rollout itself reaches it through litellm's custom
    provider map, so this function only reads it for diagnostics.

    ``rollout_fn`` / ``reward_processor`` let the **in-process** host pass the
    objects it already holds.  A worker passes neither and resolves both from
    the request's refs.  This is deliberate: resolving by ref in-process would
    make a run whose reward processor is a closure or a ``__main__``-defined
    function start failing in a mode that works today, which phase 0 must not
    do.  The rest of the body -- every branch and its order -- is identical
    either way.
    """

    started = time.monotonic()
    if rollout_fn is None:
        rollout_fn = resolve_ref(request.rollout_fn_ref)
    if reward_processor is None:
        reward_processor = _resolve_reward_processor(request)
    config = RolloutConfig(**dict(request.rollout_config))
    task = request.task
    task_id = request.task_id
    rollout_number = request.rollout_number

    trajectory_collection: dict | None = None
    interactions: dict = {}
    conversion_seconds = 0.0
    timeline = RolloutTimeline()

    def _response(
        status: str,
        result,
        workload: RolloutWorkload,
        *,
        observed: bool,
        error: BaseException | None = None,
    ) -> RolloutWorkerResponse:
        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=request.request_id,
            status=status,  # type: ignore[arg-type]
            result=result,
            workload=workload,
            observed=observed,
            error_type=type(error).__name__ if error is not None else None,
            error_repr=repr(error)[:2000] if error is not None else None,
            startup_seconds=float(startup_seconds),
            rollout_seconds=time.monotonic() - started,
            conversion_seconds=conversion_seconds,
            model_calls=len(interactions),
            routing_id_present=_routing_id_present(interactions),
            extra_allow_verified=bool(extra_allow),
            telemetry_disabled=os.environ.get("TINKER_TELEMETRY") == "0",
            torch_threads=_torch_threads(),
            peak_rss_bytes=_peak_rss_bytes(),
            cell_guard=cell_guard_cumulative(at_exit=True),
            worker_pid=os.getpid(),
            worker_pgid=os.getpgrp() if hasattr(os, "getpgrp") else 0,
            timing=timeline.summary(),
        )

    # ``isolated_rollout_python_environment`` mutates sys.path, sys.executable
    # and os.environ process-globally and restores them on exit, so with N
    # concurrent in-process rollouts the first to finish would tear the overlay
    # down under the other N-1.  It is only ever entered in a dedicated worker.
    if request.python_env_isolation and in_rollout_worker():
        from platoon.utils.rollout_python_env import isolated_rollout_python_environment

        environment_stack = isolated_rollout_python_environment()
    else:
        environment_stack = contextlib.nullcontext()

    with environment_stack:
        try:
            with timeline_scope(timeline):
                async with TinkerLLMProxySession() as session:
                    try:
                        trajectory_collection = await asyncio.create_task(rollout_fn(task, config))
                    finally:
                        # Copy before the context manager resets its ContextVar.
                        interactions = dict(session.interactions)

                    workload = workload_from_raw_rollout(trajectory_collection, interactions)

                    if not trajectory_collection.get("trajectories"):
                        logger.warning(
                            "No trajectories found for task %s and rollout %s", task_id, rollout_number
                        )
                        return _response("no_trajectories", None, workload, observed=False)

                    conversion_started = time.monotonic()
                    result = get_train_data_for_trajectory_collection(
                        trajectory_collection=trajectory_collection,
                        interactions=interactions,
                        task_id=task_id,
                        checkpoint_version=request.model.checkpoint_version,
                        filter_errors=request.conversion.filter_errors,
                        reward_processor=reward_processor,
                        include_traj_depth=request.conversion.include_traj_depth,
                        include_traj_start=request.conversion.include_traj_start,
                        subagent_datum_sampler=request.conversion.subagent_datum_sampler,
                    )
                    workload = add_datum_funnel(workload, result)
                    conversion_seconds = time.monotonic() - conversion_started

                    # Sampling is allowed to remove every converted datum, but it
                    # must not turn a genuinely unusable rollout into a baseline
                    # member merely because sampling happens to be enabled.  The
                    # per-trajectory counts are deliberately pre-sampling counts.
                    num_pre_sampling_datums = sum(stats.num_datums for stats in result.trajectory_stats)
                    if num_pre_sampling_datums == 0:
                        logger.warning(
                            "No pre-sampling train data found for task %s and rollout %s",
                            task_id,
                            rollout_number,
                        )
                        return _response("conversion_empty", None, workload, observed=True)

                    if not result.datums:
                        logger.warning(
                            "No train data found for task %s and rollout %s", task_id, rollout_number
                        )
                        # Retain an empty result produced by Bernoulli sampling or
                        # policy exclusion: its reward and trajectory stats still
                        # belong in group baselines/metrics, and a sibling rollout
                        # may provide the group's actual training datums.
                        if (
                            request.conversion.subagent_datum_sampler is None
                            and result.num_policy_excluded_datums == 0
                        ):
                            return _response("conversion_empty", None, workload, observed=True)

                    return _response("completed", result, workload, observed=True)

        except Exception as exc:
            logger.exception(
                "Error in tinker workflow for task %s and rollout %s: %s", task_id, rollout_number, exc
            )
            workload = workload_from_raw_rollout(trajectory_collection, interactions)
            return _response(
                "error",
                None,
                workload,
                observed=workload.trajectories > 0,
                error=exc,
            )
