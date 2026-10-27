"""One rollout's body -- the single implementation both hosts use.

``GroupRolloutWorkflow.arun_episode_single`` builds a
:class:`RolloutWorkerRequest` and hands it either to this function
directly (``rollout_isolation: inprocess``) or to a worker process that calls
exactly this function (``rollout_isolation: process``).  **One code path, two
hosts** is what makes in-process mode a real fallback instead of a divergent
branch that rots.

The rollout, its conversion into datums and ``add_datum_funnel`` run here in
that order, the funnel immediately after conversion.  See
:mod:`platoon.train.workers.workload` for why omitting it is total data
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
from platoon.envs.service import EnvSetupError
from platoon.train.tinker.proxy import TinkerLLMProxySession
from platoon.train.tinker.rollout_metrics import RolloutTimeline
from platoon.train.workers.protocol import PROTOCOL_VERSION, RolloutWorkerRequest, RolloutWorkerResponse, resolve_ref
from platoon.train.workers.telemetry import local_telemetry
from platoon.train.workers.workload import add_datum_funnel, workload_from_raw_rollout
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import outcome_status, scored_success, trained_reward_key
from platoon.visualization.event_sinks import emit_rollout_events, rollout_events_path

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
    """The default: the trajectory's own reward, no components."""

    return trajectory["reward"], {}


#: Identity used by the workflow to set ``ConversionPolicy.reward_processor_is_default``.
DEFAULT_REWARD_PROCESSOR: Callable[[dict], tuple[float, dict]] = default_reward_processor


def in_rollout_worker() -> bool:
    return os.environ.get(WORKER_ENV_MARKER) == "1"


def cell_guard_cumulative(*, at_exit: bool = False) -> dict[str, int]:
    """Cumulative programmatic-tool-calling counters for this process.

    The counters come from the ``rollout_worker.telemetry`` hooks.  The cell
    guard recomputes ``ptc_cells_stuck`` /
    ``ptc_threads_leaked`` / ``ptc_cells_running`` from the live set and
    documents them as *current, not cumulative*.  Summing those gauges across
    short-lived workers would invert the metric: healthy workers all report 0
    at exit, and the one worker that had a stuck cell is exactly the one that
    gets SIGKILLed and writes no response.  So workers forward the genuinely
    cumulative counters, plus the gauges sampled **once at exit** under
    distinct ``*_at_exit`` names.  Never raises.
    """

    try:
        counts = {key: int(value) for key, value in local_telemetry().items()}
    except (OverflowError, ValueError):  # a non-finite counter
        return {}
    if not counts:
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


def _outcome_status(collection: Any, error: BaseException | None) -> str | None:
    """The status a rollout counts as (``outcome/<status>``): ``error`` for one whose environment could not open its
    session, which raised :class:`~platoon.envs.service.EnvSetupError` (a sandbox that did not start, an env server
    that did not answer) and so recorded no outcome; else the status of the outcome its root trajectory recorded,
    None without a root or one.  Never raises."""

    if isinstance(error, EnvSetupError):
        return "error"
    root = _root_trajectory(collection)
    return outcome_status(root) if root is not None else None


def _reward_key(collection: Any) -> str | None:
    """The reward key the rollout's root trained (``reward_key/<key>``, and what the members of a group must agree on):
    its scored outcome's, None without one.  Never raises."""

    root = _root_trajectory(collection)
    return trained_reward_key(root) if root is not None else None


def _success_reward(collection: Any) -> float | None:
    """The reward the rollout's root's environment scored, before any reward processor (its last step's
    ``reward/success``, or a legacy rollout's own ``success`` flag), None without one.  Never raises."""

    root = _root_trajectory(collection)
    try:
        return scored_success(root) if root is not None else None
    except Exception:
        logger.debug("Cannot read the scored reward of a rollout", exc_info=True)
        return None


def _root_trajectory(collection: Any) -> dict | None:
    """The root trajectory of a rollout's trajectory collection, None without one.  Never raises: what is read of it
    is a diagnostic or a count, which must not fail the rollout."""

    try:
        trajectories = collection.get("trajectories") if isinstance(collection, dict) else None
        root = next(iter(trajectories.values()), None) if isinstance(trajectories, dict) else None
        return root if isinstance(root, dict) else None
    except Exception:
        logger.debug("Cannot read the root trajectory of a rollout", exc_info=True)
        return None


def _write_rollout_events(config: RolloutConfig, task_id: str, collection: dict) -> None:
    """Write the finished rollout's event file for the trajectory viewer; never fails the rollout."""

    try:
        emit_rollout_events(collection, rollout_events_path(config.output_dir, task_id, str(collection.get("id"))))
    except Exception:
        logger.warning("Could not write the event file of a rollout of task %s", task_id, exc_info=True)


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
        raise ValueError("conversion policy declares a non-default reward processor but carries no reference")
    return resolve_ref(ref)


async def run_rollout_request(
    request: RolloutWorkerRequest,
    llm: Any = None,
    *,
    startup_seconds: float = 0.0,
    extra_allow: bool = False,
    rollout_fn: Callable | None = None,
    reward_processor: Callable[[dict], tuple[float, dict]] | None = None,
    on_first_call: Callable[[], None] | None = None,
) -> RolloutWorkerResponse:
    """Run one rollout and convert it, returning a serializable outcome.

    ``llm`` is the ``TinkerLLM`` of whichever process hosts the rollout; the
    rollout itself reaches its gateway as the process default
    (``platoon.gateway.default_gateway``), so this function only reads it for
    diagnostics.

    ``rollout_fn`` / ``reward_processor`` let the **in-process** host pass the
    objects it already holds.  A worker passes neither and resolves both from
    the request's refs.  This is deliberate: resolving by ref in-process would
    make a run whose reward processor is a closure or a ``__main__``-defined
    function fail in in-process mode, which supports them.  The rest of the
    body -- every branch and its order -- is identical either way.

    ``on_first_call`` becomes the session timeline's: it is called once, as the
    rollout tree's first model call is issued (``RolloutTimeline``).

    The steps, in order: roll out in the policy's session, write its events
    for the trajectory viewer, convert the trajectories to datums, classify
    the result.  A rollout without trajectories is not converted, and one
    that raised reports its error.
    :class:`_RolloutRun` holds what the steps share.
    """

    started = time.monotonic()
    if rollout_fn is None:
        rollout_fn = resolve_ref(request.rollout_fn_ref)
    if reward_processor is None:
        reward_processor = _resolve_reward_processor(request)
    run = _RolloutRun(request, started, startup_seconds=startup_seconds, extra_allow=extra_allow)
    with _python_environment(request):
        try:
            async with TinkerLLMProxySession() as session:
                await run.roll_out(session, rollout_fn, on_first_call)
                workload = workload_from_raw_rollout(run.collection, run.interactions)
                if not run.collection.get("trajectories"):
                    logger.warning(
                        "No trajectories found for task %s and rollout %s", request.task_id, request.rollout_number
                    )
                    return run.response("no_trajectories", None, workload, observed=False)
                _write_rollout_events(run.config, request.task_id, run.collection)
                result, workload = run.convert(llm, reward_processor, workload)
                return run.classify(result, workload)
        except Exception as exc:
            return run.failed(exc)


def _python_environment(request: RolloutWorkerRequest) -> contextlib.AbstractContextManager:
    """The rollout's own Python overlay when its request asks for one and this is a rollout worker; else nothing.

    ``isolated_rollout_python_environment`` mutates sys.path, sys.executable
    and os.environ process-globally and restores them on exit, so with N
    concurrent in-process rollouts the first to finish would tear the overlay
    down under the other N-1, and the last could put back an overlay already
    removed.  It is only ever entered in a dedicated worker, and it refuses a
    second entry in one process.
    """

    if request.python_env_isolation and in_rollout_worker():
        from platoon.utils.rollout_python_env import isolated_rollout_python_environment

        return isolated_rollout_python_environment()
    return contextlib.nullcontext()


def rollout_version(llm: Any, request: RolloutWorkerRequest) -> int:
    """The policy version a rollout's datums are stamped with (``checkpoint_version``, which staleness is measured
    against): the oldest one any sample of its tree used, in a worker that followed the trainer's weights while it
    ran; else the version its request was bound to.  Per-call versions are to replace it."""

    # Process workers follow current weights while waiting for the
    # environment. Anchor staleness to actual generation, retaining
    # the oldest root/child/judge sample if weights changed mid-tree.
    from platoon.train.workers.binding import FollowingSamplingClient

    sampling_client = getattr(llm, "sampling_client", None)
    checkpoint_version = request.model.checkpoint_version
    if isinstance(sampling_client, FollowingSamplingClient):
        sampled_version = sampling_client.oldest_sampled_version
        if sampled_version is not None:
            checkpoint_version = sampled_version
    return checkpoint_version


class _RolloutRun:
    """One rollout of :func:`run_rollout_request`: what its steps share, and the steps after the session opens.

    Its state: the request and the start of the run, the rollout's config, its trajectory collection and interactions
    once it has run, the conversion's seconds, and the timeline the proxy records into.
    """

    def __init__(
        self, request: RolloutWorkerRequest, started: float, *, startup_seconds: float, extra_allow: bool
    ) -> None:
        self.request = request
        self.started = started
        self.startup_seconds = startup_seconds
        self.extra_allow = extra_allow
        self.config = RolloutConfig(**dict(request.rollout_config))
        self.collection: Any = None  # the rollout's trajectory collection, once it has run
        self.interactions: dict = {}
        self.conversion_seconds = 0.0
        # ``TinkerLLMProxySession`` binds the timeline the proxy records into,
        # exactly as it binds the interaction map.  Binding a second one here would
        # be shadowed by the session's, and every timing would silently read zero;
        # so this starts as an empty timeline (well-formed if the session never
        # opens) and is replaced by the session's as soon as there is one.
        self.timeline = RolloutTimeline()

    def response(
        self,
        status: str,
        result,
        workload: RolloutWorkload,
        *,
        observed: bool,
        error: BaseException | None = None,
    ) -> RolloutWorkerResponse:
        """The rollout's response, with what the run measured and the process it ran in."""

        return RolloutWorkerResponse(
            protocol_version=PROTOCOL_VERSION,
            request_id=self.request.request_id,
            status=status,  # type: ignore[arg-type]
            result=result,
            workload=workload,
            observed=observed,
            error_type=type(error).__name__ if error is not None else None,
            error_repr=repr(error)[:2000] if error is not None else None,
            startup_seconds=float(self.startup_seconds),
            rollout_seconds=time.monotonic() - self.started,
            conversion_seconds=self.conversion_seconds,
            model_calls=len(self.interactions),
            routing_id_present=_routing_id_present(self.interactions),
            extra_allow_verified=bool(self.extra_allow),
            telemetry_disabled=os.environ.get("TINKER_TELEMETRY") == "0",
            torch_threads=_torch_threads(),
            peak_rss_bytes=_peak_rss_bytes(),
            cell_guard=cell_guard_cumulative(at_exit=True),
            worker_pid=os.getpid(),
            worker_pgid=os.getpgrp() if hasattr(os, "getpgrp") else 0,
            timing=self.timeline.summary(),
            outcome_status=_outcome_status(self.collection, error),
            reward_key=_reward_key(self.collection),
            success_reward=_success_reward(self.collection),
        )

    async def roll_out(self, session: Any, rollout_fn: Callable, on_first_call: Callable[[], None] | None) -> None:
        """Run the rollout in ``session``, whose timeline becomes the run's; its interactions are kept whatever the
        rollout does."""

        self.timeline = getattr(session, "timeline", self.timeline)
        if on_first_call is not None:
            self.timeline.on_first_call = on_first_call
        try:
            self.collection = await asyncio.create_task(rollout_fn(self.request.task, self.config))
        finally:
            # Copy before the context manager resets its ContextVar.
            self.interactions = dict(session.interactions)

    def convert(
        self, llm: Any, reward_processor: Callable[[dict], tuple[float, dict]], workload: RolloutWorkload
    ) -> tuple[Any, RolloutWorkload]:
        """The rollout's trajectories as datums, stamped with :func:`rollout_version`, and its workload with the
        conversion's funnel."""

        conversion_started = time.monotonic()
        checkpoint_version = rollout_version(llm, self.request)
        result = get_train_data_for_trajectory_collection(
            trajectory_collection=self.collection,
            interactions=self.interactions,
            task_id=self.request.task_id,
            checkpoint_version=checkpoint_version,
            filter_errors=self.request.conversion.filter_errors,
            filter_message_actions=self.request.conversion.filter_message_actions,
            reward_processor=reward_processor,
            include_traj_depth=self.request.conversion.include_traj_depth,
            include_traj_start=self.request.conversion.include_traj_start,
            subagent_datum_sampler=self.request.conversion.subagent_datum_sampler,
        )
        workload = add_datum_funnel(workload, result)
        self.conversion_seconds = time.monotonic() - conversion_started
        return result, workload

    def classify(self, result: Any, workload: RolloutWorkload) -> RolloutWorkerResponse:
        """The converted rollout's response: ``conversion_empty`` for one with no datum to train, else ``completed``."""

        task_id, rollout_number = self.request.task_id, self.request.rollout_number
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
            return self.response("conversion_empty", None, workload, observed=True)

        if not result.datums:
            logger.warning("No train data found for task %s and rollout %s", task_id, rollout_number)
            # Retain an empty result produced by Bernoulli sampling or
            # policy exclusion: its reward and trajectory stats still
            # belong in group baselines/metrics, and a sibling rollout
            # may provide the group's actual training datums.
            if self.request.conversion.subagent_datum_sampler is None and result.num_policy_excluded_datums == 0:
                return self.response("conversion_empty", None, workload, observed=True)

        return self.response("completed", result, workload, observed=True)

    def failed(self, exc: Exception) -> RolloutWorkerResponse:
        """The response of a rollout that raised: ``error``, observed when it had produced a trajectory."""

        task_id, rollout_number = self.request.task_id, self.request.rollout_number
        logger.exception("Error in tinker workflow for task %s and rollout %s: %s", task_id, rollout_number, exc)
        workload = workload_from_raw_rollout(self.collection, self.interactions)
        return self.response(
            "error",
            None,
            workload,
            observed=workload.trajectories > 0,
            error=exc,
        )
