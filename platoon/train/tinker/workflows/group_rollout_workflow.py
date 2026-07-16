"""Group-centered rollout workflow for tinker RL training.

This module provides a workflow that runs multiple rollouts per task (group_size),
computes group-centered advantages, and produces training data in tinker.Datum format.
"""

import asyncio
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Callable

import tinker
import torch
from tinker import TensorData

from platoon.envs.base import Task
from platoon.train.tinker.config_defs import RolloutConfig, WorkflowConfig
from platoon.train.tinker.proxy import ModelInfo, TinkerLLMProxySession
from platoon.utils.rollout_workload import (
    RolloutWorkload,
    record_workload_distribution,
    sum_rollout_workloads,
    trajectory_collection_shape,
)
from platoon.utils.stats_tracker import get as get_tracker
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.tinker_data_processing import (
    SubagentDatumSamplingStats,
    TrajectoryCollectionResult,
    TrajectoryStats,
    get_train_data_for_trajectory_collection,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SingleRolloutOutcome:
    """Internal result that keeps raw generation work even if conversion fails."""

    result: TrajectoryCollectionResult | None
    workload: RolloutWorkload
    observed: bool


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


def _workload_from_raw_rollout(
    trajectory_collection: dict | None,
    interactions: dict,
) -> RolloutWorkload:
    """Count one raw recursive tree and its unique proxy interactions."""

    trajectories, environment_steps = trajectory_collection_shape(trajectory_collection)
    # ``interactions`` is keyed by completion ID, so walking its values counts
    # every logical model request exactly once even when a recursive tree later
    # excludes, interrupts, or fails to convert one of its trajectories.
    input_tokens = sum(int(interaction.obs.length) for interaction in interactions.values())
    output_tokens = sum(len(interaction.action.tokens) for interaction in interactions.values())
    return RolloutWorkload(
        environment_steps=environment_steps,
        model_calls=len(interactions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trajectories=trajectories,
    )


def _add_datum_funnel(
    workload: RolloutWorkload,
    result: TrajectoryCollectionResult | None,
) -> RolloutWorkload:
    """Attach post-merge, policy, and sampling counts to raw workload."""

    if result is None:
        return workload
    postmerge = sum(stats.num_datums for stats in result.trajectory_stats)
    policy_eligible = postmerge - result.num_policy_excluded_datums
    post_sampling = len(result.datums)
    # RolloutWorkload validates that the funnel is ordered and non-negative.
    return replace(
        workload,
        postmerge_datums=postmerge,
        policy_eligible_datums=policy_eligible,
        post_sampling_datums=post_sampling,
    )


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
        reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
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
            filter_errors: Whether to filter out error steps from successful trajectories.
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
        raw_outcomes = await asyncio.gather(
            *[self.arun_episode_single(data, i) for i in range(self.config.group_size)]
        )
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

    async def arun_episode_single(self, data: dict, rollout_number: int = 0) -> _SingleRolloutOutcome:
        """Run a single rollout and return training data.

        Args:
            data: Dictionary containing 'task_id' and optionally other task data.
            rollout_number: Index of this rollout within the group.

        Returns:
            Internal outcome containing converted data when usable and the raw
            workload even when conversion or rollout execution failed.
        """
        task_id = data["task_id"]
        trajectory_collection: dict | None = None
        interactions: dict = {}

        try:
            task = self.get_task_fn(task_id)
            rollout_config = self._get_rollout_config()

            if rollout_config.max_steps is not None:
                task.max_steps = rollout_config.max_steps

            # Get checkpoint version from the LLM
            checkpoint_version = self.model_info.llm.version

            rollout_config.output_dir = os.path.join(
                rollout_config.output_dir,
                str(checkpoint_version),
            )

            # Use the proxy session to track LLM interactions
            async with TinkerLLMProxySession() as session:
                try:
                    # Run the rollout with the proper config
                    trajectory_collection = await asyncio.create_task(self.rollout_fn(task, rollout_config))
                finally:
                    # Copy before the context manager resets its ContextVar.
                    interactions = dict(session.interactions)

                workload = _workload_from_raw_rollout(trajectory_collection, interactions)

                if not trajectory_collection.get("trajectories"):
                    logger.warning(f"No trajectories found for task {task_id} and rollout {rollout_number}")
                    return _SingleRolloutOutcome(result=None, workload=workload, observed=False)

                # Extract training data
                result = get_train_data_for_trajectory_collection(
                    trajectory_collection=trajectory_collection,
                    interactions=interactions,
                    task_id=task_id,
                    checkpoint_version=checkpoint_version,
                    filter_errors=self.filter_errors,
                    reward_processor=self.reward_processor,
                    include_traj_depth=self.config.depth_level_weighting,
                    include_traj_start=self.config.depth_level_weighting,
                    subagent_datum_sampler=self.subagent_datum_sampler,
                )
                workload = _add_datum_funnel(workload, result)

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
                    return _SingleRolloutOutcome(result=None, workload=workload, observed=True)

                if not result.datums:
                    logger.warning(f"No train data found for task {task_id} and rollout {rollout_number}")
                    # Retain an empty result produced by Bernoulli sampling or
                    # policy exclusion: its reward and trajectory stats still
                    # belong in group baselines/metrics, and a sibling rollout
                    # may provide the group's actual training datums.
                    if self.subagent_datum_sampler is None and result.num_policy_excluded_datums == 0:
                        return _SingleRolloutOutcome(result=None, workload=workload, observed=True)

                return _SingleRolloutOutcome(result=result, workload=workload, observed=True)

        except Exception as e:
            logger.exception(f"Error in tinker workflow for task {task_id} and rollout {rollout_number}: {e}")
            workload = _workload_from_raw_rollout(trajectory_collection, interactions)
            return _SingleRolloutOutcome(
                result=None,
                workload=workload,
                observed=workload.trajectories > 0,
            )
