"""Step-wise AReaL workflow for RL training.

This module implements the StepWiseArealWorkflow which runs rollouts and extracts
step-wise training data with optional prefix-aware sequence aggregation.
"""

import asyncio
import logging
import os
from copy import deepcopy
from typing import Callable

import torch
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy import ProxyServer
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors

from platoon.envs.base import Task
from platoon.train.areal.config_defs import WorkflowConfig
from platoon.train.areal.proxy import ArealProxySession
from platoon.utils.areal_data_processing import get_train_data_for_trajectory_collection

logger = logging.getLogger(__name__)


class StepWiseArealWorkflow(RolloutWorkflow):
    """Workflow that runs rollouts and extracts step-wise training data.

    This workflow:
    1. Runs `group_size` rollouts for each task in parallel
    2. Collects training data from each step (with optional prefix merging)
    3. Computes group-centered advantages (reward - mean_reward)
    4. Returns training data for AReaL

    When merge_prefixes=True (default), consecutive steps whose observations
    are prefixes of subsequent observations are merged into single sequences.
    This reduces redundant computation during training by avoiding reprocessing
    the same prefix tokens multiple times.
    """

    def __init__(
        self,
        rollout_fn: Callable[[Task, dict], dict],
        get_task_fn: Callable[[str], Task],
        config: WorkflowConfig,
        proxy_server: ProxyServer,
        stats_scope: str,
        device: torch.device,
        filter_errors: bool = False,
        reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
        merge_prefixes: bool = True,
    ):
        self.config = config
        self.config.rollout_config.return_dict = True
        self.config.rollout_config.train = True
        self.proxy_server = proxy_server
        self.api_version = "v1"
        self.proxy_url = f"{self.proxy_server.public_addr}/{self.api_version}"
        self.stats_scope = stats_scope
        self.device = device
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.filter_errors = filter_errors
        self.reward_processor = reward_processor
        self.merge_prefixes = merge_prefixes

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        """Run multiple rollouts for a task and return training data."""
        results = await asyncio.gather(
            *[self.arun_episode_single(engine, data, i) for i in range(self.config.group_size)]
        )
        results = [result for result in results if result is not None]
        if not results:
            print(f"[StepWiseWorkflow] No results found for task {data['task_id']}")
            return None

        train_data = concat_padded_tensors(results)

        mean_unprocessed_reward = torch.mean(train_data["rewards"])

        # Center advantages
        train_data["rewards"] = train_data["rewards"] - torch.mean(train_data["task_reward"])

        tracker = stats_tracker.get(self.stats_scope)

        # Track per-trajectory stats
        task_reward_mask = torch.ones_like(train_data["task_reward"], dtype=torch.bool).to(self.device)
        output_token_mask = torch.ones_like(train_data["num_output_tokens"], dtype=torch.bool).to(self.device)
        input_token_mask = torch.ones_like(train_data["num_input_tokens"], dtype=torch.bool).to(self.device)
        num_steps_mask = torch.ones_like(train_data["num_steps"], dtype=torch.bool).to(self.device)

        # Per-step averages (useful for understanding step-level characteristics)
        num_steps = train_data["num_steps"].to(self.device)
        num_input_tokens = train_data["num_input_tokens"].to(self.device)
        num_output_tokens = train_data["num_output_tokens"].to(self.device)
        safe_num_steps = torch.clamp(num_steps, min=1.0)
        avg_input_tokens_per_step = num_input_tokens / safe_num_steps
        avg_output_tokens_per_step = num_output_tokens / safe_num_steps

        tracker.denominator(
            task_reward_mask=task_reward_mask,
            num_output_tokens_mask=output_token_mask,
            num_input_tokens_mask=input_token_mask,
            num_steps_mask=num_steps_mask,
            avg_input_tokens_per_step_mask=num_steps_mask,
            avg_output_tokens_per_step_mask=num_steps_mask,
        )
        tracker.stat(task_reward=train_data["task_reward"].to(self.device), denominator="task_reward_mask")
        tracker.stat(num_output_tokens=num_output_tokens, denominator="num_output_tokens_mask")
        tracker.stat(num_input_tokens=num_input_tokens, denominator="num_input_tokens_mask")
        tracker.stat(num_steps=num_steps, denominator="num_steps_mask")
        tracker.stat(avg_input_tokens_per_step=avg_input_tokens_per_step, denominator="avg_input_tokens_per_step_mask")
        tracker.stat(
            avg_output_tokens_per_step=avg_output_tokens_per_step, denominator="avg_output_tokens_per_step_mask"
        )

        # task_reward @ K metrics (computed per-task across K rollouts)
        task_rewards = train_data["task_reward"].to(self.device)
        task_reward_at_k_mask = torch.ones(1, dtype=torch.bool).to(self.device)
        tracker.denominator(task_reward_at_k_mask=task_reward_at_k_mask)
        tracker.stat(task_reward_at_k_mean=torch.mean(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")
        tracker.stat(task_reward_at_k_max=torch.max(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")
        tracker.stat(task_reward_at_k_min=torch.min(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")

        # Track root_* and reward/* metrics
        for key, value in train_data.items():
            if key.startswith("root_"):
                tracker.stat(**{key: value.to(self.device)}, denominator="task_reward_mask")
                tracker.stat(
                    **{f"{key}_at_k_mean": torch.mean(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
                tracker.stat(
                    **{f"{key}_at_k_max": torch.max(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
                tracker.stat(
                    **{f"{key}_at_k_min": torch.min(value).unsqueeze(0).to(self.device)},
                    denominator="task_reward_at_k_mask",
                )
            elif key.startswith("reward/"):
                reward_mask = torch.ones_like(value, dtype=torch.bool).to(self.device)
                tracker.denominator(**{f"{key}_mask": reward_mask})
                tracker.stat(**{key: value.to(self.device)}, denominator=f"{key}_mask")

        if train_data["rewards"].max() == train_data["rewards"].min() and len(results) > 1:
            print(
                f"[StepWiseWorkflow] All rewards same for task {data['task_id']}: {mean_unprocessed_reward.item():.2f}"
            )
            return None

        return train_data

    async def arun_episode_single(self, engine: InferenceEngine, data: dict, rollout_number: int) -> dict | None:
        """Run a single rollout and return training data."""
        config = deepcopy(self.config)
        try:
            task_id = data["task_id"]
            task = self.get_task_fn(task_id)
            if config.rollout_config.max_steps is not None:
                task.max_steps = config.rollout_config.max_steps

            async with ArealProxySession(base_url=self.proxy_url) as session:
                config.rollout_config.model_endpoint = session.session_base_url
                # Prepend 'openai/' to be compatible with LiteLLM
                config.rollout_config.model_name = "openai/" + config.rollout_config.model_name
                config.rollout_config.model_api_key = "None"

                config.rollout_config.output_dir = os.path.join(
                    config.rollout_config.output_dir,
                    str(engine.get_version()),
                )

                results = await asyncio.create_task(self.rollout_fn(task, config.rollout_config))

                if not results["trajectories"]:
                    print(f"[StepWiseWorkflow] No trajectories found for task {task_id} and rollout {rollout_number}")
                    return None

                # Get completions from proxy server session cache
                completions = self.proxy_server.session_cache[session.session_id].completions
                train_data = get_train_data_for_trajectory_collection(
                    results,
                    completions,
                    task_id,
                    self.filter_errors,
                    self.reward_processor,
                    self.merge_prefixes,
                    concat_fn=concat_padded_tensors,
                )

                if train_data is None:
                    print(f"[StepWiseWorkflow] No train data found for task {task_id} and rollout {rollout_number}")
                    return None

            return train_data

        except Exception as e:
            import traceback

            print(f"[StepWiseWorkflow] Error in areal workflow for task {task_id} and rollout {rollout_number}: {e}")
            traceback.print_exc()
            return None
