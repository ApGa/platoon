"""Group-centered rollout workflow for tinker RL training.

This module provides a workflow that runs multiple rollouts per task (group_size),
computes group-centered advantages, and produces training data in tinker.Datum format.
"""

import asyncio
import logging
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Callable

import tinker
import torch
from tinker import TensorData

from platoon.envs.base import Task
from platoon.train.tinker.config_defs import RolloutConfig, WorkflowConfig
from platoon.train.tinker.proxy import ModelInfo
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
from platoon.train.tinker.workers.workload import add_datum_funnel, workload_from_raw_rollout
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
    TrajectoryStats,
)
from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY

logger = logging.getLogger(__name__)


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
    """List-compatible task result with an out-of-band workload payload.

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
    ) -> None:
        super().__init__(datums)
        self.workload = workload
        self.requested_rollouts = requested_rollouts
        self.observed_rollouts = observed_rollouts
        self.trainable_rollouts = trainable_rollouts
        self.task_retained_datums = task_retained_datums


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
        return True, {}
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
    keep = bool(torch.any(filtered_mask > 0))
    return keep, {
        "detected_action_tokens": float(error_actions.sum().item()),
        "suppressed_positive_action_tokens": float(suppressed.sum().item()),
        "retained_nonpositive_action_tokens": float((error_actions & ~suppressed).sum().item()),
        "emptied_datums": float(not keep and bool(action_tokens.any())),
    }


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
        return _TaskRolloutOutput(
            datums,
            workload=task_workload,
            requested_rollouts=self.config.group_size,
            observed_rollouts=sum(outcome.observed for outcome in outcomes),
            trainable_rollouts=trainable_rollouts,
            task_retained_datums=task_retained_datums,
        )

    async def arun_episode(self, data: dict) -> list[tinker.Datum]:
        """Run multiple rollouts for a task and return training data.

        Args:
            data: Dictionary containing 'task_id' and optionally other task data.

        Returns:
            A list-compatible datum result. An empty result still carries exact
            task generation workload for trainer-side batch accounting.
        """
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
        task_rewards: list[float] = []
        all_trajectory_stats: list[TrajectoryStats] = []
        all_root_rewards_dicts: list[dict[str, float]] = []
        subagent_sampling_stats = SubagentDatumSamplingStats()

        for result in valid_results:
            all_data.extend(result.datums)
            task_rewards.append(result.task_reward)
            all_trajectory_stats.extend(result.trajectory_stats)
            all_root_rewards_dicts.append(result.root_rewards_dict)
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

        # === Track stats BEFORE early returns ===
        # This ensures we track stats even for groups that get filtered out

        # Per-trajectory stats (num_steps, num_tokens are tracked per trajectory)
        num_steps_per_traj = torch.tensor([float(s.num_steps) for s in all_trajectory_stats])
        num_input_tokens_per_traj = torch.tensor([float(s.num_input_tokens) for s in all_trajectory_stats])
        num_output_tokens_per_traj = torch.tensor([float(s.num_output_tokens) for s in all_trajectory_stats])

        # Per-step averages (useful for understanding step-level characteristics)
        # Avoid division by zero for trajectories with no steps
        safe_num_steps = torch.clamp(num_steps_per_traj, min=1.0)
        avg_input_tokens_per_step = num_input_tokens_per_traj / safe_num_steps
        avg_output_tokens_per_step = num_output_tokens_per_traj / safe_num_steps

        # Masks for per-trajectory stats
        trajectory_mask = torch.ones(len(all_trajectory_stats), dtype=torch.bool)

        self.tracker.denominator(
            num_output_tokens_mask=trajectory_mask,
            num_input_tokens_mask=trajectory_mask,
            num_steps_mask=trajectory_mask,
            avg_input_tokens_per_step_mask=trajectory_mask,
            avg_output_tokens_per_step_mask=trajectory_mask,
        )
        self.tracker.stat(num_output_tokens=num_output_tokens_per_traj, denominator="num_output_tokens_mask")
        self.tracker.stat(num_input_tokens=num_input_tokens_per_traj, denominator="num_input_tokens_mask")
        self.tracker.stat(num_steps=num_steps_per_traj, denominator="num_steps_mask")
        self.tracker.stat(
            avg_input_tokens_per_step=avg_input_tokens_per_step, denominator="avg_input_tokens_per_step_mask"
        )
        self.tracker.stat(
            avg_output_tokens_per_step=avg_output_tokens_per_step, denominator="avg_output_tokens_per_step_mask"
        )

        # Per-rollout stats (task_reward is the root trajectory's reward, one per rollout)
        task_rewards_tensor = torch.tensor(task_rewards)
        rollout_mask = torch.ones(len(task_rewards), dtype=torch.bool)

        self.tracker.denominator(task_reward_mask=rollout_mask)
        self.tracker.stat(task_reward=task_rewards_tensor, denominator="task_reward_mask")

        # task_reward @ K metrics (computed per-task across K rollouts)
        task_reward_at_k_mask = torch.ones(1, dtype=torch.bool)
        self.tracker.denominator(task_reward_at_k_mask=task_reward_at_k_mask)
        self.tracker.stat(
            task_reward_at_k_mean=torch.mean(task_rewards_tensor).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        self.tracker.stat(
            task_reward_at_k_max=torch.max(task_rewards_tensor).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        self.tracker.stat(
            task_reward_at_k_min=torch.min(task_rewards_tensor).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )

        # root_* metrics from reward_processor (per-rollout, from root trajectory only)
        # Collect all reward component keys from the root trajectories
        all_reward_keys: set[str] = set()
        for rewards_dict in all_root_rewards_dicts:
            all_reward_keys.update(rewards_dict.keys())

        for key in all_reward_keys:
            # Optional reward components must be averaged only where present;
            # zero-filling absent judgments would underreport the metric.
            values = torch.tensor([rewards_dict[key] for rewards_dict in all_root_rewards_dicts if key in rewards_dict])
            root_reward_mask = torch.ones_like(values, dtype=torch.bool)
            root_mask_name = f"root_{key}_mask"
            self.tracker.denominator(**{root_mask_name: root_reward_mask})

            self.tracker.stat(**{f"root_{key}": values}, denominator=root_mask_name)
            self.tracker.stat(
                **{f"root_{key}_at_k_mean": torch.mean(values).unsqueeze(0)},
                denominator="task_reward_at_k_mask",
            )
            self.tracker.stat(
                **{f"root_{key}_at_k_max": torch.max(values).unsqueeze(0)},
                denominator="task_reward_at_k_mask",
            )
            self.tracker.stat(
                **{f"root_{key}_at_k_min": torch.min(values).unsqueeze(0)},
                denominator="task_reward_at_k_mask",
            )

        # reward/* metrics from per-trajectory rewards_dict (tracked per trajectory)
        all_per_traj_reward_keys: set[str] = set()
        for stats in all_trajectory_stats:
            for key in stats.rewards_dict:
                if key.startswith("reward/"):
                    all_per_traj_reward_keys.add(key)

        for key in all_per_traj_reward_keys:
            values = torch.tensor(
                [stats.rewards_dict[key] for stats in all_trajectory_stats if key in stats.rewards_dict]
            )
            reward_mask = torch.ones_like(values, dtype=torch.bool)
            self.tracker.denominator(**{f"{key}_mask": reward_mask})
            self.tracker.stat(**{key: values}, denominator=f"{key}_mask")

        # === Now compute advantages and filter ===

        # Interrupted roots carry partial task rewards. Keep those rewards in all
        # metrics above, but exclude them from the control-variate estimate used
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

        # Full-tree reward/LOO statistics above intentionally include empty and
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
        tasks = [
            asyncio.ensure_future(self.arun_episode_single(data, i))
            for i in range(group_size)
        ]
        if straggler_timeout is None:
            return list(await asyncio.gather(*tasks))

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
            binding_path=self._binding_path,
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
        scratch = self._ipc_dir()
        response_path = os.path.join(scratch, f"{request_id}.res")
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
                filter_errors=self.filter_errors,
                include_traj_depth=self.config.depth_level_weighting,
                include_traj_start=self.config.depth_level_weighting,
                reward_processor_is_default=self._reward_processor_is_default,
                reward_processor_ref=self.reward_processor_ref,
                subagent_datum_sampler=self.subagent_datum_sampler,
            ),
            response_path=response_path,
            ready_path=os.path.join(scratch, f"{request_id}.ready"),
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
        return ipc_dir()

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
            response = await run_rollout_request(
                request,
                self.model_info.llm,
                rollout_fn=self.rollout_fn,
                reward_processor=self.reward_processor,
            )
            return self._outcome_from_response(response, response.status)
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
