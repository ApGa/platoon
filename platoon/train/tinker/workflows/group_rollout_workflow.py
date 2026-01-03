"""Group-centered rollout workflow for tinker RL training.

This module provides a workflow that runs multiple rollouts per task (group_size),
computes group-centered advantages, and produces training data in tinker.Datum format.
"""

import asyncio
import logging
import os
from copy import deepcopy
from typing import Callable

import tinker
import torch

logger = logging.getLogger(__name__)

from tinker import TensorData

from platoon.envs.base import Task
from platoon.train.tinker.config_defs import WorkflowConfig, RolloutConfig
from platoon.train.tinker.proxy import ModelInfo, TinkerLLMProxySession
from platoon.utils.tinker_data_processing import (
    get_train_data_for_trajectory_collection,
    TrajectoryCollectionResult,
    TrajectoryStats,
)
from platoon.utils.stats_tracker import get as get_tracker


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
        reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj['reward'], {}),
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

    async def arun_episode(self, data: dict) -> list[tinker.Datum] | None:
        """Run multiple rollouts for a task and return training data.
        
        Args:
            data: Dictionary containing 'task_id' and optionally other task data.
            
        Returns:
            List of tinker.Datum with group-centered advantages, or None if no data.
        """
        results = await asyncio.gather(
            *[self.arun_episode_single(data, i) for i in range(self.config.group_size)]
        )
        
        # Filter out None results and collect data
        all_data: list[tinker.Datum] = []
        task_rewards: list[float] = []
        all_trajectory_stats: list[TrajectoryStats] = []
        all_root_rewards_dicts: list[dict[str, float]] = []
        
        for result in results:
            if result is not None:
                all_data.extend(result.datums)
                task_rewards.append(result.task_reward)
                all_trajectory_stats.extend(result.trajectory_stats)
                all_root_rewards_dicts.append(result.root_rewards_dict)
        
        if not all_data:
            logger.warning(f"No results found for task {data['task_id']}")
            return None
        
        # === Track stats BEFORE early returns ===
        # This ensures we track stats even for groups that get filtered out
        
        # Per-trajectory stats (num_steps, num_tokens are tracked per trajectory)
        num_steps_per_traj = torch.tensor([float(s.num_steps) for s in all_trajectory_stats])
        num_input_tokens_per_traj = torch.tensor([float(s.num_input_tokens) for s in all_trajectory_stats])
        num_output_tokens_per_traj = torch.tensor([float(s.num_output_tokens) for s in all_trajectory_stats])
        
        # Masks for per-trajectory stats
        trajectory_mask = torch.ones(len(all_trajectory_stats), dtype=torch.bool)
        
        self.tracker.denominator(
            num_output_tokens_mask=trajectory_mask,
            num_input_tokens_mask=trajectory_mask,
            num_steps_mask=trajectory_mask,
        )
        self.tracker.stat(num_output_tokens=num_output_tokens_per_traj, denominator="num_output_tokens_mask")
        self.tracker.stat(num_input_tokens=num_input_tokens_per_traj, denominator="num_input_tokens_mask")
        self.tracker.stat(num_steps=num_steps_per_traj, denominator="num_steps_mask")
        
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
            # Collect values for this key from all root trajectories (one per rollout)
            values = torch.tensor([
                rewards_dict.get(key, 0.0) for rewards_dict in all_root_rewards_dicts
            ])
            
            # Track root_* metrics with rollout mask (same as task_reward)
            self.tracker.stat(**{f"root_{key}": values}, denominator="task_reward_mask")
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
                if key.startswith('reward/'):
                    all_per_traj_reward_keys.add(key)
        
        for key in all_per_traj_reward_keys:
            values = torch.tensor([
                stats.rewards_dict.get(key, 0.0) for stats in all_trajectory_stats
            ])
            reward_mask = torch.ones_like(values, dtype=torch.bool)
            self.tracker.denominator(**{f"{key}_mask": reward_mask})
            self.tracker.stat(**{key: values}, denominator=f"{key}_mask")
        
        # === Now compute advantages and filter ===
        
        # Compute group-centered advantages
        mean_task_reward = sum(task_rewards) / len(task_rewards) if task_rewards else 0.0
        
        # Check if all rewards are the same (no learning signal)
        if len(task_rewards) > 1 and max(task_rewards) == min(task_rewards):
            logger.debug(f"All rewards are the same for task {data['task_id']}: {mean_task_reward:.2f}")
            return None
        
        # Center advantages: new_adv = old_adv - mean_reward
        # The old_adv was set to trajectory_reward, so this gives us (reward - mean_reward)
        for datum in all_data:
            old_advantages = datum.loss_fn_inputs["advantages"].to_torch()
            # The mask tells us which tokens were action tokens (advantage != 0)
            mask = datum.loss_fn_inputs["mask"].to_torch()
            # Subtract mean_task_reward from non-zero advantages
            new_advantages = torch.where(
                mask > 0,
                old_advantages - mean_task_reward,
                old_advantages
            )
            datum.loss_fn_inputs["advantages"] = TensorData.from_torch(new_advantages)
        
        return all_data

    async def arun_episode_single(
        self, 
        data: dict, 
        rollout_number: int = 0
    ) -> TrajectoryCollectionResult | None:
        """Run a single rollout and return training data.
        
        Args:
            data: Dictionary containing 'task_id' and optionally other task data.
            rollout_number: Index of this rollout within the group.
            
        Returns:
            TrajectoryCollectionResult with datums and stats, or None if failed.
        """
        task_id = data['task_id']
        
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
                # Run the rollout with the proper config
                results = await asyncio.create_task(self.rollout_fn(task, rollout_config))
                
                if not results.get('trajectories'):
                    logger.warning(f"No trajectories found for task {task_id} and rollout {rollout_number}")
                    return None
                
                # Get the llm interactions recorded during this session
                interactions = session.interactions
                
                # Extract training data
                result = get_train_data_for_trajectory_collection(
                    trajectory_collection=results,
                    interactions=interactions,
                    task_id=task_id,
                    checkpoint_version=checkpoint_version,
                    filter_errors=self.filter_errors,
                    reward_processor=self.reward_processor,
                )
                
                if not result.datums:
                    logger.warning(f"No train data found for task {task_id} and rollout {rollout_number}")
                    return None
                
                return result
                
        except Exception as e:
            logger.exception(f"Error in tinker workflow for task {task_id} and rollout {rollout_number}: {e}")
            return None
