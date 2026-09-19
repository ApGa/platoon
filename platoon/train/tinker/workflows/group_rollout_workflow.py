"""Group-centered rollout workflow for tinker RL training.

This module provides a workflow that runs multiple rollouts per task (group_size),
computes group-centered advantages, and produces training data in tinker.Datum format.
"""

import asyncio
import logging
import os
import uuid
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Callable

import tinker
import torch
from tinker import TensorData

from platoon.envs.base import Task
from platoon.train.tinker.config_defs import RolloutConfig, WorkflowConfig
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.tinker.workers.protocol import (
    PROTOCOL_VERSION,
    ConversionPolicy,
    ModelBinding,
    RolloutWorkerRequest,
    RolloutWorkerResponse,
    ipc_dir,
    validated_ref,
)
from platoon.train.tinker.workers.run import DEFAULT_REWARD_PROCESSOR, run_rollout_request
from platoon.train.tinker.workers.supervisor import RolloutWorkerSupervisor, WorkerOutcome
from platoon.train.tinker.workers.telemetry import record_worker_cell_guard
from platoon.train.tinker.workers.workload import add_datum_funnel, workload_from_raw_rollout
from platoon.utils.rollout_stats import RolloutStats, summarize_rollout_group
from platoon.utils.rollout_workload import (
    RolloutWorkload,
    record_workload_distribution,
    sum_rollout_workloads,
)
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import (
    SubagentDatumSamplingStats,
    TrajectoryCollectionResult,
)
from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY

logger = logging.getLogger(__name__)

# Set by ``_run_group_rollouts`` before it creates its member tasks, so each
# member can report readiness back to its group's admission permit without
# changing ``arun_episode_single``'s signature (custom workflows override it).
_group_ready_callback: ContextVar = ContextVar("platoon_group_ready_callback", default=None)


@dataclass(frozen=True)
class _SingleRolloutOutcome:
    """Internal result that keeps raw generation work even if conversion fails.

    ``disposition`` records *how* the member ended (see the taxonomy in
    ``workers.supervisor``).  It is defaulted so existing callers -- including
    tests that construct this positionally -- are unaffected, and it is never
    load-bearing for training: a worker death is a member failure, decided by
    ``min_successful_group_size`` like any other, never a group or trainer
    failure.
    """

    result: TrajectoryCollectionResult | None
    workload: RolloutWorkload
    observed: bool
    disposition: str = "completed"


class _TaskRolloutOutput(list[tinker.Datum]):
    """List-compatible task result carrying workload and rollout statistics.

    The trainer and legacy consumers can continue treating this as an ordinary
    datum list.  In particular, an empty instance behaves like an empty list,
    while still carrying the generation work for a task which produced no
    trainable policy data.
    """

    def __init__(
        self,
        datums: list[tinker.Datum],
        *,
        workload: RolloutWorkload,
        requested_rollouts: int,
        observed_rollouts: int,
        trainable_rollouts: int,
        task_retained_datums: int,
        rollout_stats: RolloutStats | None = None,
    ) -> None:
        super().__init__(datums)
        self.workload = workload
        self.requested_rollouts = requested_rollouts
        self.observed_rollouts = observed_rollouts
        self.trainable_rollouts = trainable_rollouts
        self.task_retained_datums = task_retained_datums
        self.rollout_stats = rollout_stats


def _summarize_outcomes(outcomes: list[_SingleRolloutOutcome]) -> RolloutStats:
    results = [outcome.result for outcome in outcomes if outcome.result is not None]
    trajectories = [stats for result in results for stats in result.trajectory_stats]
    data = {
        "task_reward": [result.task_reward for result in results],
        "task_reward_valid": [result.task_reward_valid for result in results],
        **{
            key: [getattr(stats, key) for stats in trajectories]
            for key in ("num_steps", "num_input_tokens", "num_output_tokens")
        },
    }
    root_keys = {key for result in results for key in result.root_rewards_dict}
    for key in root_keys:
        data[f"root_{key}"] = [
            result.root_rewards_dict[key] for result in results if key in result.root_rewards_dict
        ]
    reward_keys = {key for stats in trajectories for key in stats.rewards_dict if key.startswith("reward/")}
    for key in reward_keys:
        data[key] = [stats.rewards_dict[key] for stats in trajectories if key in stats.rewards_dict]
    return summarize_rollout_group(data)


# Both helpers now live in ``workers.workload`` so the rollout worker calls the
# same code the trainer does.  The module-level aliases keep the historical
# names importable.
_workload_from_raw_rollout = workload_from_raw_rollout
_add_datum_funnel = add_datum_funnel


def _filter_positive_centered_error_tokens(
    datum: tinker.Datum,
) -> tuple[bool, dict[str, float]]:
    """Consume the error side channel and mask only positive-advantage errors."""

    encoded_error_mask = datum.loss_fn_inputs.pop(ERROR_ACTION_MASK_KEY, None)
    if encoded_error_mask is None:
        # Agent-message-only datums have no policy tokens even when error
        # filtering is disabled. Drop them after their rewards were centered.
        return bool(datum.loss_fn_inputs["mask"].to_torch().any()), {}
    error_mask = encoded_error_mask.to_torch().bool()
    action_mask = datum.loss_fn_inputs["mask"].to_torch()
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    if error_mask.shape != action_mask.shape or advantages.shape != action_mask.shape:
        raise ValueError(
            "Deferred Tinker error filtering requires aligned error/mask/advantage tensors: "
            f"error={tuple(error_mask.shape)}, mask={tuple(action_mask.shape)}, "
            f"advantages={tuple(advantages.shape)}"
        )

    action_tokens = action_mask > 0
    error_actions = error_mask & action_tokens
    suppressed = error_actions & (advantages > 0)
    filtered_mask = torch.where(suppressed, torch.zeros_like(action_mask), action_mask)
    datum.loss_fn_inputs["mask"] = TensorData.from_torch(filtered_mask)
    # The trainer strips mask before protocol dispatch; CISPO and the other
    # policy losses use advantages to suppress these tokens on the server.
    datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
        torch.where(suppressed, torch.zeros_like(advantages), advantages)
    )
    keep = bool(torch.any(filtered_mask > 0))
    return keep, {
        "detected_action_tokens": float(error_actions.sum().item()),
        "suppressed_positive_action_tokens": float(suppressed.sum().item()),
        "retained_nonpositive_action_tokens": float((error_actions & ~suppressed).sum().item()),
        "emptied_datums": float(not keep and bool(action_tokens.any())),
    }


class _GroupAdmission:
    """One startup permit held for a whole group until its members are up.

    Released when every member has signalled readiness or exited, or after
    ``timeout_seconds`` -- whichever comes first -- and never released twice.
    """

    def __init__(self, semaphore: asyncio.Semaphore | None, expected: int, timeout_seconds: float):
        self._semaphore = semaphore
        self._pending = max(0, int(expected))
        self._released = semaphore is None
        self._timer: asyncio.TimerHandle | None = None
        if semaphore is not None and timeout_seconds > 0:
            # A group whose members never signal readiness (all of them failed
            # to launch, say) must not hold the permit forever.
            self._timer = asyncio.get_running_loop().call_later(timeout_seconds, self.release)

    def member_slot(self):
        """A one-shot callback for a single member."""

        fired = False

        def _slot() -> None:
            nonlocal fired
            if fired:
                return
            fired = True
            self._pending -= 1
            if self._pending <= 0:
                self.release()

        return _slot

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if self._semaphore is not None:
            self._semaphore.release()


class GroupRolloutWorkflow:
    """Workflow that runs multiple rollouts per task and computes group-centered advantages.

    1. Runs `group_size` rollouts for each task in parallel
    2. Collects training data from each rollout
    3. Computes group-centered advantages (reward - mean_reward)
    4. Returns training data in tinker.Datum format
    """

    def __init__(
        self,
        rollout_fn: Callable[[Task, RolloutConfig], dict],
        get_task_fn: Callable[[str], Task],
        config: WorkflowConfig,
        model_info: ModelInfo,
        log_path: str | None = None,
        stats_scope: str = "train",
        filter_errors: bool = False,
        reward_processor: Callable[[dict], tuple[float, dict]] = DEFAULT_REWARD_PROCESSOR,
    ):
        """Initialize the workflow.

        Args:
            rollout_fn: Async function that runs a rollout given a task and RolloutConfig.
            get_task_fn: Function that returns a Task given a task_id.
            config: Workflow configuration (contains group_size and rollout_config).
            model_info: Model information for the tinker LLM.
            log_path: Base log directory for the training run. If provided, rollout results
                      will be stored at {log_path}/rollouts/{stats_scope}/.
            stats_scope: Name for the stats tracker scope (e.g., "train" or "eval").
            filter_errors: Whether to suppress erroneous action tokens only when
                their final group-centered advantage is positive.
            reward_processor: Function to process trajectory rewards.
        """
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.config = config
        self.model_info = model_info
        self.log_path = log_path
        self.stats_scope = stats_scope
        self.filter_errors = filter_errors
        self.reward_processor = reward_processor
        self.tracker = get_tracker(stats_scope)
        self.subagent_datum_sampler = (
            DeterministicSubagentDatumSampler(
                keep_probability=config.subagent_datum_keep_probability,
                seed=config.subagent_datum_sampling_seed,
            )
            if config.subagent_datum_keep_probability < 1.0
            else None
        )
        self.rollout_isolation = getattr(config, "rollout_isolation", "inprocess")
        # Callable references are built ONCE, at construction, and proven to
        # round-trip.  A closure or a functools.partial cannot cross a process
        # boundary by name, and a ``ref or default`` fallback would silently
        # substitute the historical reward processor -- changing task_reward,
        # root_rewards_dict and every advantage with no error and entirely
        # plausible metrics.  In ``inprocess`` mode the callables are handed to
        # the shared rollout body directly, so an unnameable callable is only
        # recorded, not rejected: today's behaviour must not change.
        self._reward_processor_is_default = reward_processor is DEFAULT_REWARD_PROCESSOR
        self.rollout_fn_ref = self._build_ref(rollout_fn, what="rollout_fn")
        self.reward_processor_ref = (
            None
            if self._reward_processor_is_default
            else self._build_ref(reward_processor, what="reward_processor")
        )
        self._supervisor = None
        self._admission: asyncio.Semaphore | None = None
        self._admission_limit = 0

    def _build_ref(self, obj, *, what: str) -> str | None:
        """Name a callable for a worker, strictly when a worker will need it."""

        if obj is None:
            if self.rollout_isolation == "process":
                raise ValueError(f"rollout_isolation='process' requires a real {what}")
            return None
        try:
            return validated_ref(obj, what=what)
        except Exception:
            if self.rollout_isolation == "process":
                raise
            logger.debug("%s is not nameable across processes; inprocess mode only", what, exc_info=True)
            return None

    def _get_rollout_config(self) -> RolloutConfig:
        """Get a copy of the rollout config with model info populated."""
        config = deepcopy(self.config.rollout_config)
        config.model_name = self.model_info.model_name
        config.model_endpoint = self.model_info.base_url
        config.model_api_key = self.model_info.api_key
        config.return_dict = True
        config.train = True

        # If log_path is provided, use it as the base for rollout output
        if self.log_path is not None:
            config.output_dir = os.path.join(self.log_path, "rollouts", self.stats_scope)

        return config

    def _make_task_output(
        self,
        datums: list[tinker.Datum],
        *,
        outcomes: list[_SingleRolloutOutcome],
        trainable_rollouts: int,
    ) -> _TaskRolloutOutput:
        """Record unit distributions and attach exact totals to the queue item."""

        rollout_workloads = [outcome.workload for outcome in outcomes]
        task_workload = sum_rollout_workloads(rollout_workloads)
        postmerge_datums = task_workload.postmerge_datums
        policy_eligible_datums = task_workload.policy_eligible_datums
        post_sampling_datums = task_workload.post_sampling_datums
        task_retained_datums = len(datums)
        if not 0 <= task_retained_datums <= post_sampling_datums <= policy_eligible_datums <= postmerge_datums:
            raise ValueError(
                "Invalid task datum funnel: "
                f"postmerge={postmerge_datums}, policy_eligible={policy_eligible_datums}, "
                f"post_sampling={post_sampling_datums}, task_retained={task_retained_datums}"
            )
        record_workload_distribution(
            self.tracker,
            prefix="workload/rollout",
            workloads=rollout_workloads,
        )
        record_workload_distribution(
            self.tracker,
            prefix="workload/task",
            workloads=[task_workload],
        )

        # The workload/task/count denominator was registered immediately above,
        # so these are task-level distributions with the same sample boundary.
        self.tracker.stat(
            denominator="workload/task/count",
            **{
                "workload/task/requested_rollouts": torch.tensor(
                    [float(self.config.group_size)], dtype=torch.float32
                ),
                "workload/task/observed_rollouts": torch.tensor(
                    [float(sum(outcome.observed for outcome in outcomes))], dtype=torch.float32
                ),
                "workload/task/workflow_trainable_rollouts": torch.tensor(
                    [float(trainable_rollouts)], dtype=torch.float32
                ),
                "workload/task/total_task_retained_datums": torch.tensor(
                    [float(task_retained_datums)], dtype=torch.float32
                ),
                "workload/task/total_task_workflow_trainable_datums": torch.tensor(
                    [float(task_retained_datums)], dtype=torch.float32
                ),
                "workload/task/total_task_workflow_non_trainable_datums": torch.tensor(
                    [float(postmerge_datums - task_retained_datums)], dtype=torch.float32
                ),
                "workload/task/total_task_filter_dropped_datums": torch.tensor(
                    [float(post_sampling_datums - task_retained_datums)], dtype=torch.float32
                ),
            },
        )
        rollout_stats = _summarize_outcomes(outcomes)
        rollout_stats.record(self.tracker, prefix="rollout_completion")
        return _TaskRolloutOutput(
            datums,
            workload=task_workload,
            requested_rollouts=self.config.group_size,
            observed_rollouts=sum(outcome.observed for outcome in outcomes),
            trainable_rollouts=trainable_rollouts,
            task_retained_datums=task_retained_datums,
            rollout_stats=rollout_stats,
        )

    async def arun_episode(self, data: dict) -> list[tinker.Datum]:
        """Run multiple rollouts for a task and return training data.

        Args:
            data: Dictionary containing 'task_id' and optionally other task data.

        Returns:
            A list-compatible datum result. An empty result still carries exact
            task generation workload for trainer-side batch accounting.
        """
        with self._monitor().group_in_flight():
            raw_outcomes = await self._run_group_rollouts(data)
        # Accept legacy test/custom overrides that return the old result type,
        # while the canonical single-rollout path always returns an outcome.
        outcomes = [
            outcome
            if isinstance(outcome, _SingleRolloutOutcome)
            else _SingleRolloutOutcome(
                result=outcome,
                workload=_add_datum_funnel(RolloutWorkload(), outcome),
                observed=outcome is not None,
            )
            for outcome in raw_outcomes
        ]
        valid_results = [outcome.result for outcome in outcomes if outcome.result is not None]

        # Filter out None results and collect data
        all_data: list[tinker.Datum] = []
        subagent_sampling_stats = SubagentDatumSamplingStats()

        for result in valid_results:
            all_data.extend(result.datums)
            subagent_sampling_stats.merge(result.subagent_sampling_stats)

        if self.subagent_datum_sampler is not None:
            self.tracker.scalar(**subagent_sampling_stats.to_metrics())

        if not valid_results:
            logger.warning(f"No results found for task {data['task_id']}")
            return self._make_task_output([], outcomes=outcomes, trainable_rollouts=0)

        min_successful = getattr(self.config, "min_successful_group_size", 1)
        if len(valid_results) < min_successful:
            # Too few members for a meaningful within-task baseline. The
            # generation work is still accounted for through the outcomes.
            logger.warning(
                "Rejecting task %s group with %s usable rollout(s); minimum is %s",
                data["task_id"],
                len(valid_results),
                min_successful,
            )
            self.tracker.scalar(group_size_rejected=1.0)
            return self._make_task_output([], outcomes=outcomes, trainable_rollouts=0)

        # === Now compute advantages and filter ===

        # Interrupted roots carry partial task rewards. Keep those rewards in
        # rollout statistics, but exclude them from the control-variate estimate used
        # to center policy advantages.
        valid_task_rewards = [result.task_reward for result in valid_results if result.task_reward_valid]
        if not valid_task_rewards:
            logger.warning(
                "No completed root rewards available for task %s; retaining rollout stats but skipping training",
                data["task_id"],
            )
            return self._make_task_output([], outcomes=outcomes, trainable_rollouts=0)

        if not all_data:
            logger.warning(f"No train data found for task {data['task_id']}")
            return self._make_task_output([], outcomes=outcomes, trainable_rollouts=0)

        # Compute group-centered advantages from completed root rewards only.
        mean_task_reward = sum(valid_task_rewards) / len(valid_task_rewards)

        # Center advantages by rollout. The old_adv was set to trajectory_reward, so
        # this produces either reward - mean_reward or reward - loo_baseline.
        if self.config.leave_one_out_baseline:
            total_valid_task_reward = sum(valid_task_rewards)
            num_valid_task_rewards = len(valid_task_rewards)
            baselines = []
            for result in valid_results:
                if not result.task_reward_valid:
                    # A completed child from a partial rollout can still train,
                    # but the partial root must not affect its baseline.
                    baselines.append(mean_task_reward)
                elif num_valid_task_rewards > 1:
                    baselines.append(
                        (total_valid_task_reward - result.task_reward) / (num_valid_task_rewards - 1)
                    )
                else:
                    # A singleton completed root has no leave-one-out peer.
                    baselines.append(result.task_reward)
        else:
            baselines = [mean_task_reward] * len(valid_results)

        for result, baseline in zip(valid_results, baselines):
            for datum in result.datums:
                old_advantages = datum.loss_fn_inputs["advantages"].to_torch()
                mask = datum.loss_fn_inputs["mask"].to_torch()
                new_advantages = torch.where(mask > 0, old_advantages - baseline, old_advantages)
                datum.loss_fn_inputs["advantages"] = TensorData.from_torch(new_advantages)

        error_filter_totals: dict[str, float] = {}
        error_filter_active = False
        for result in valid_results:
            retained_datums: list[tinker.Datum] = []
            for datum in result.datums:
                keep, metrics = _filter_positive_centered_error_tokens(datum)
                if metrics:
                    error_filter_active = True
                    for key, value in metrics.items():
                        error_filter_totals[key] = error_filter_totals.get(key, 0.0) + value
                if keep:
                    retained_datums.append(datum)
            result.datums[:] = retained_datums
        if error_filter_active:
            self.tracker.scalar(
                **{f"error_filter/{key}": value for key, value in error_filter_totals.items()}
            )
        all_data = [datum for result in valid_results for datum in result.datums]

        # Full-tree rollout statistics intentionally include empty and
        # policy-ineligible trajectories. The train/no-train decision must not:
        # a sampled-out or invalid-verifier child with a different reward cannot
        # turn an otherwise zero-signal retained batch into an optimizer step.
        retained_action_advantages = [
            datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0]
            for datum in all_data
        ]
        retained_action_advantages = [values for values in retained_action_advantages if values.numel() > 0]
        if not retained_action_advantages:
            logger.debug("No retained policy action tokens for task %s", data["task_id"])
            return self._make_task_output([], outcomes=outcomes, trainable_rollouts=0)
        flattened_advantages = torch.cat(retained_action_advantages)
        if getattr(self.config, "filter_zero_advantage_datums", True) and bool(
            torch.all(flattened_advantages == 0)
        ):
            logger.debug(
                "Deferring all-zero policy datums for task %s to the post-transform "
                "trainer filter (mean task reward %.2f)",
                data["task_id"],
                mean_task_reward,
            )
            # Keep these candidates until the trainer has assembled the
            # cross-task microbatch. They must participate in depth-frequency
            # normalization and the original action-token denominator even
            # though they can be omitted from the expensive model pass.

        trainable_rollouts = sum(
            any(
                bool(
                    torch.any(
                        datum.loss_fn_inputs["advantages"].to_torch()[
                            datum.loss_fn_inputs["mask"].to_torch() > 0
                        ]
                        != 0
                    )
                )
                for datum in result.datums
            )
            for result in valid_results
        )
        return self._make_task_output(
            all_data,
            outcomes=outcomes,
            trainable_rollouts=trainable_rollouts,
        )

    def _admission_semaphore(self) -> asyncio.Semaphore:
        limit = max(1, int(getattr(self.config, "worker_launch_concurrent_groups", 2)))
        semaphore = getattr(self, "_admission", None)
        if semaphore is None or getattr(self, "_admission_limit", 0) != limit:
            semaphore = asyncio.Semaphore(limit)
            self._admission = semaphore
            self._admission_limit = limit
        return semaphore

    async def _admit_group(self, group_size: int) -> "_GroupAdmission":
        """Take one startup permit for the WHOLE group, not one per rollout.

        Two things make this the right granularity.  A per-rollout permit
        released at ``exec`` bounds nothing at all: ``create_subprocess_exec``
        returns in well under a millisecond while the child then spends 20-30 s
        importing, so the gate must be readiness, not launch.  But a
        *per-rollout* readiness gate would then inject start-time skew inside a
        group, and the straggler policy has no per-member reference point -- it
        starts its tail-grace clock when the quorum settles.  With FIFO
        admission and tasks created in group order, the same rollout numbers
        would deterministically start last and be cancelled having had a
        fraction of their budget, biasing the leave-one-out baseline toward
        whichever members launch early.  Admitting whole groups removes
        intra-group skew entirely; inter-group skew is harmless because each
        group has its own clock.  One acquire per group also cannot deadlock.
        """

        if self._isolation != "process":
            return _GroupAdmission(None, group_size, 0.0)
        semaphore = self._admission_semaphore()
        await semaphore.acquire()
        budget = float(getattr(self.config, "worker_startup_budget_seconds", 300.0)) + 60.0
        return _GroupAdmission(semaphore, group_size, budget)

    async def _run_group_member(
        self, data: dict, rollout_number: int, admission: "_GroupAdmission"
    ) -> _SingleRolloutOutcome:
        """Run one member, reporting its readiness to the group's permit."""

        slot = admission.member_slot()
        token = _group_ready_callback.set(slot)
        try:
            return await self.arun_episode_single(data, rollout_number)
        finally:
            _group_ready_callback.reset(token)
            slot()

    async def _run_group_rollouts(self, data: dict) -> list:
        """Run ``group_size`` rollouts concurrently with the straggler policy.

        Without ``straggler_timeout_seconds`` this is equivalent to
        ``asyncio.gather``. With it, once ``straggler_quorum`` members
        (default ``group_size - 1``) have settled, the remaining members get at
        most ``straggler_timeout_seconds`` more; leftovers are cancelled and
        reported as unobserved outcomes plus the ``straggler_cancelled`` scalar.
        """
        group_size = self.config.group_size
        straggler_timeout = getattr(self.config, "straggler_timeout_seconds", None)
        admission = await self._admit_group(group_size)
        tasks = [
            asyncio.ensure_future(self._run_group_member(data, i, admission))
            for i in range(group_size)
        ]
        if straggler_timeout is None:
            try:
                return list(await asyncio.gather(*tasks))
            finally:
                admission.release()

        quorum = getattr(self.config, "straggler_quorum", None)
        if quorum is None:
            quorum = max(group_size - 1, 1)
        pending: set[asyncio.Future] = set(tasks)
        try:
            while pending and len(pending) > group_size - quorum:
                _done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            if pending:
                _done, pending = await asyncio.wait(pending, timeout=straggler_timeout)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            admission.release()
            raise

        # A rollout can settle between the deadline and this cancellation;
        # only members that are still running count as cancelled stragglers.
        stragglers = [task for task in tasks if not task.done()]
        if stragglers:
            logger.warning(
                "Cancelling %s straggler rollout(s) for task %s after %ss grace (quorum %s/%s)",
                len(stragglers),
                data["task_id"],
                straggler_timeout,
                quorum,
                group_size,
            )
            for task in stragglers:
                task.cancel()
            # run_rollout's own cancellation cleanup marks trajectories as
            # interrupted; leaked environment sessions are reaped server-side.
            await asyncio.gather(*stragglers, return_exceptions=True)
        self.tracker.scalar(straggler_cancelled=float(len(stragglers)))
        admission.release()

        outcomes: list = []
        for task in tasks:
            if task.cancelled():
                outcomes.append(
                    _SingleRolloutOutcome(
                        result=None,
                        workload=RolloutWorkload(),
                        observed=False,
                        disposition="straggler_killed",
                    )
                )
            else:
                # Propagates a legacy override's exception like gather did.
                outcomes.append(task.result())
        return outcomes

    def _slug(self, value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value)).strip("-") or "task"

    def _model_binding(self, checkpoint_version: int) -> ModelBinding:
        """Describe the sampling model for a rollout host.

        Never pickles the ``Renderer``: ``Renderer.__reduce__`` carries only
        ``(renderer_name, model_name, has_image_processor)`` and drops
        ``renderer_kwargs``, which ``register_tinker_llm`` applies by
        ``setattr``.  Today's values happen to equal the class defaults, so
        shipping the object is a latent silent-drift bug that fires the day
        someone sets ``preserve_thinking: false``.
        """

        llm = getattr(self.model_info, "llm", None)
        bare_model_name = getattr(llm, "model_name", None) or str(
            getattr(self.model_info, "model_name", "") or ""
        ).removeprefix("platoon-tinker/")
        return ModelBinding(
            # Same reason as the scratch paths in ``_build_worker_request``:
            # only a worker follows this file, so only ``process`` mode resolves
            # (and creates) the directory that holds it.
            binding_path=self._binding_path if self._isolation == "process" else "",
            bootstrap_blob=b"",
            follow_poll_seconds=float(getattr(self.config, "worker_binding_poll_seconds", 5.0)),
            model_name=bare_model_name,
            renderer_name=getattr(self.model_info, "renderer_name", "") or "",
            renderer_kwargs=dict(getattr(self.model_info, "renderer_kwargs", {}) or {}),
            context_window_length=getattr(llm, "context_window_length", None),
            max_tokens=int(getattr(llm, "max_tokens", 4096)),
            temperature=float(getattr(llm, "temperature", 1.0)),
            top_k=int(getattr(llm, "top_k", -1)),
            top_p=float(getattr(llm, "top_p", 1.0)),
            seed=getattr(llm, "seed", None),
            checkpoint_version=checkpoint_version,
        )

    def _build_worker_request(self, data: dict, rollout_number: int) -> RolloutWorkerRequest:
        """Everything the parent must decide before a rollout can run.

        Identical to what ``arun_episode_single`` used to do inline: resolve
        the task parent-side (which also removes the shared-mutable-``Task``
        aliasing some plugin datasets have), apply ``max_steps``, snapshot the
        checkpoint version and join ``output_dir`` with it.
        """

        task_id = data["task_id"]
        task = self.get_task_fn(task_id)
        rollout_config = self._get_rollout_config()
        if rollout_config.max_steps is not None:
            task.max_steps = rollout_config.max_steps

        # Get checkpoint version from the LLM
        checkpoint_version = self.model_info.llm.version
        rollout_config.output_dir = os.path.join(rollout_config.output_dir, str(checkpoint_version))

        request_id = f"{self._slug(task_id)}-r{rollout_number}-{uuid.uuid4().hex[:8]}"
        # Only a worker needs node-local scratch, and only a worker ever writes
        # these files.  Resolving them in ``inprocess`` mode would make the
        # fallback -- the mode a run rolls BACK to -- create and depend on a
        # directory it never uses, so an unwritable $TMPDIR would fail every
        # rollout in the mode that is supposed to be unaffected by any of this.
        if self._isolation == "process":
            scratch = self._ipc_dir()
            response_path = os.path.join(scratch, f"{request_id}.res")
            ready_path = os.path.join(scratch, f"{request_id}.ready")
        else:
            response_path = ready_path = ""
        return RolloutWorkerRequest(
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            task_id=task_id,
            rollout_number=rollout_number,
            task=task,
            rollout_config=asdict(rollout_config),
            rollout_fn_ref=self.rollout_fn_ref or "",
            model=self._model_binding(checkpoint_version),
            conversion=ConversionPolicy(
                token_efficiency_reward=self.config.token_efficiency_reward,
                filter_errors=self.filter_errors,
                filter_message_actions=getattr(self.config, "filter_message_actions", False),
                include_traj_depth=self.config.depth_level_weighting,
                include_traj_start=self.config.depth_level_weighting,
                reward_processor_is_default=self._reward_processor_is_default,
                reward_processor_ref=self.reward_processor_ref,
                subagent_datum_sampler=self.subagent_datum_sampler,
            ),
            response_path=response_path,
            ready_path=ready_path,
            startup_budget_seconds=float(getattr(self.config, "worker_startup_budget_seconds", 300.0)),
            rollout_budget_seconds=self._rollout_budget_seconds(rollout_config),
            rss_soft_limit_bytes=int(getattr(self.config, "worker_rss_soft_limit_bytes", 24 * 1024**3)),
            python_env_isolation=bool(getattr(self.config, "worker_python_env_isolation", False)),
        )

    def _rollout_budget_seconds(self, rollout_config: RolloutConfig) -> float:
        """The child's own deadline: the rollout timeout plus finalize grace.

        The child also converts datums and writes a multi-MB response after the
        rollout proper, so its alarm must outlast ``rollout_config.timeout``.
        """

        timeout = rollout_config.timeout
        grace = float(getattr(self.config, "worker_finalize_grace_seconds", 120.0))
        if timeout is None:
            return float(getattr(self.config, "worker_unbounded_rollout_budget_seconds", 24 * 3600.0))
        return float(timeout) + grace

    def _ipc_dir(self) -> str:
        cached = getattr(self, "_ipc_dir_cache", None)
        if cached is None:
            cached = ipc_dir()
            self._ipc_dir_cache = cached
        return cached

    def _monitor(self):
        """Client throughput monitor for this workflow's stats scope.

        ``getattr`` because custom/legacy workflows construct this class with
        ``__new__`` and set only the attributes they use.
        """

        return client_monitor(getattr(self, "stats_scope", "train"))

    @property
    def _isolation(self) -> str:
        return getattr(self, "rollout_isolation", "inprocess")

    @property
    def supervisor(self) -> RolloutWorkerSupervisor:
        """Lazily built so ``inprocess`` runs never touch node-local scratch."""

        if getattr(self, "_supervisor", None) is None:
            self._supervisor = RolloutWorkerSupervisor(
                scratch_dir=self._ipc_dir(),
                termination_grace_seconds=float(
                    getattr(self.config, "worker_termination_grace_seconds", 10.0)
                ),
                parent_grace_seconds=float(getattr(self.config, "worker_parent_grace_seconds", 60.0)),
                mem_available_floor_bytes=int(
                    getattr(self.config, "worker_mem_available_floor_bytes", 0)
                ),
            )
        return self._supervisor

    def publish_model_binding(self, sampling_client, version: int) -> None:
        """Republish the trainer's sampling client for its rollout workers."""

        if self._isolation != "process":
            return
        self.supervisor.publish_binding(sampling_client, version)

    def _worker_log_dir(self, checkpoint_version: int) -> str | None:
        """Next to the rollout artifacts, so the smoke can grep both together."""

        if self.log_path is None:
            return None
        return os.path.join(
            self.log_path, "rollouts", self.stats_scope, str(checkpoint_version), "workers"
        )

    @property
    def _binding_path(self) -> str:
        return os.path.join(self._ipc_dir(), "model_binding.pkl")

    def _outcome_from_response(
        self,
        response: RolloutWorkerResponse | None,
        disposition: str,
    ) -> _SingleRolloutOutcome:
        """Map a worker response (or its absence) onto a group member outcome."""

        if response is None:
            return _SingleRolloutOutcome(
                result=None,
                workload=RolloutWorkload(),
                observed=False,
                disposition=disposition,
            )
        return _SingleRolloutOutcome(
            result=response.result,
            workload=response.workload,
            observed=response.observed,
            disposition=disposition,
        )

    def _record_worker_telemetry(self, outcome: WorkerOutcome) -> None:
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
                    if coverage < 1.0:
                        # A dropped routing_id is not an error anywhere in the
                        # stack: the datum simply loses routing_handle and the
                        # gradient is quietly worse.
                        logger.warning(
                            "rollout worker %s returned routing ids for only %s/%s model calls "
                            "(R3 routing_handle coverage %.3f)",
                            outcome.request_id,
                            response.routing_id_present,
                            response.model_calls,
                            coverage,
                        )
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
                record_worker_cell_guard(getattr(self, "stats_scope", "train"), response.cell_guard)
                self._monitor().record_rollout_timing(
                    response.timing, startup_offset=outcome.launch_seconds
                )
                self._monitor().record_worker_peak_rss(response.peak_rss_bytes)
            elif outcome.diag:
                # A SIGKILLed worker writes its cumulative cell-guard counters
                # into the marker precisely because it cannot write a response;
                # without this the metric would read zero exactly when a cell
                # was killed for hanging.
                record_worker_cell_guard(getattr(self, "stats_scope", "train"), outcome.diag)
            self.tracker.scalar(**scalars)
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
        except Exception:  # pragma: no cover - telemetry must never break a rollout
            logger.debug("failed to record rollout worker telemetry", exc_info=True)

    async def arun_episode_single(self, data: dict, rollout_number: int = 0) -> _SingleRolloutOutcome:
        """Run a single rollout and return training data.

        The blanket guard stays and now also covers request building: that adds
        ``pickle.dumps``, a write to node-local scratch and (in ``process``
        mode) opening a log file, any of which can raise on a full disk.  An
        escape here would propagate through ``_run_group_rollouts``'s
        ``task.result()`` and cost ALL group members, including the ones that
        completed and would have satisfied ``min_successful_group_size``.

        Args:
            data: Dictionary containing 'task_id' and optionally other task data.
            rollout_number: Index of this rollout within the group.

        Returns:
            Internal outcome containing converted data when usable and the raw
            workload even when conversion or rollout execution failed.
        """
        try:
            request = self._build_worker_request(data, rollout_number)
            monitor = self._monitor()
            if self._isolation != "process":
                with monitor.rollout_in_flight():
                    response = await run_rollout_request(
                        request,
                        self.model_info.llm,
                        rollout_fn=self.rollout_fn,
                        reward_processor=self.reward_processor,
                    )
                monitor.record_rollout_timing(response.timing)
                return self._outcome_from_response(response, response.status)

            ready = _group_ready_callback.get()
            with monitor.rollout_in_flight():
                outcome = await self.supervisor.run(
                    request,
                    log_dir=self._worker_log_dir(request.model.checkpoint_version),
                    on_ready=ready,
                    on_launch=monitor.register_worker,
                    on_reap=monitor.unregister_worker,
                )
            self._record_worker_telemetry(outcome)
            return self._outcome_from_response(outcome.response, outcome.disposition)
        except asyncio.CancelledError:
            # The straggler policy owns cancellation.
            raise
        except Exception as exc:
            logger.exception(
                "Error in tinker workflow for task %s and rollout %s: %s",
                data.get("task_id"),
                rollout_number,
                exc,
            )
            self.tracker.scalar(rollout_worker_launch_failed=1.0)
            return _SingleRolloutOutcome(
                result=None,
                workload=RolloutWorkload(),
                observed=False,
                disposition="worker_launch_failed",
            )
