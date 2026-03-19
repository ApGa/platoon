"""Step-wise AReaL workflow for RL training.

This module implements the StepWiseArealWorkflow which runs rollouts and extracts
step-wise training data with optional prefix-aware sequence aggregation.
"""

import asyncio
import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

import torch
import multiprocessing as mp
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
        reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (
            traj["reward"],
            {},
        ),
        merge_prefixes: bool = True,
    ):
        self.config = deepcopy(config)
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
        
        self.config.rollout_config.output_dir = os.path.join(self.config.rollout_config.output_dir, self.stats_scope)

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        """Run multiple rollouts for a task and return training data."""
        if self.config.use_subprocesses:
            results = await self._arun_episode_with_subprocesses(engine, data)
        else:
            results = await asyncio.gather(
                *[self._arun_episode_single(engine, data, i) for i in range(self.config.group_size)]
            )
        results = [result for result in results if result is not None]
        if not results:
            print(f"[StepWiseWorkflow] No results found for task {data['task_id']}")
            return None

        train_data = concat_padded_tensors(results)

        mean_unprocessed_reward = torch.mean(train_data["rewards"])

        # Center advantages
        if self.config.leave_one_out_baseline and len(results) > 1:
            # Leave-one-out: each rollout's baseline is the mean of all other rollouts' rewards
            task_rewards = train_data["task_reward"]  # shape [N]
            N = len(task_rewards)
            total_reward = task_rewards.sum()
            loo_baselines = (total_reward - task_rewards) / (N - 1)  # shape [N]
            # Expand per-rollout baselines to per-datum baselines
            datum_counts = torch.tensor([r["rewards"].shape[0] for r in results])
            per_datum_baselines = torch.repeat_interleave(loo_baselines, datum_counts)
            train_data["rewards"] = train_data["rewards"] - per_datum_baselines
        else:
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
        tracker.stat(
            task_reward=train_data["task_reward"].to(self.device),
            denominator="task_reward_mask",
        )
        tracker.stat(num_output_tokens=num_output_tokens, denominator="num_output_tokens_mask")
        tracker.stat(num_input_tokens=num_input_tokens, denominator="num_input_tokens_mask")
        tracker.stat(num_steps=num_steps, denominator="num_steps_mask")
        tracker.stat(
            avg_input_tokens_per_step=avg_input_tokens_per_step,
            denominator="avg_input_tokens_per_step_mask",
        )
        tracker.stat(
            avg_output_tokens_per_step=avg_output_tokens_per_step,
            denominator="avg_output_tokens_per_step_mask",
        )

        # task_reward @ K metrics (computed per-task across K rollouts)
        task_rewards = train_data["task_reward"].to(self.device)
        task_reward_at_k_mask = torch.ones(1, dtype=torch.bool).to(self.device)
        tracker.denominator(task_reward_at_k_mask=task_reward_at_k_mask)
        tracker.stat(
            task_reward_at_k_mean=torch.mean(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        tracker.stat(
            task_reward_at_k_max=torch.max(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )
        tracker.stat(
            task_reward_at_k_min=torch.min(task_rewards).unsqueeze(0),
            denominator="task_reward_at_k_mask",
        )

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

        if not self.config.filter_zero_variance_groups:
            train_data["trainable_datums"] = torch.ones_like(
                train_data["rewards"], dtype=torch.bool
            )

        if train_data["rewards"].max() == train_data["rewards"].min() and len(results) > 1:
            tracker.scalar(zero_variance_reward_group=1.0)
            print(
                f"[StepWiseWorkflow] All rewards same for task {data['task_id']}: {mean_unprocessed_reward.item():.2f}"
            )
            if self.config.filter_zero_variance_groups:
                return None
            train_data["trainable_datums"] = torch.zeros_like(
                train_data["rewards"], dtype=torch.bool
            )
            return train_data

        return train_data

    def _process_trajectory_result(
        self,
        trajectory_data: dict | None,
        session: ArealProxySession,
        task_id: str,
        rollout_number: int,
    ) -> dict | None:
        """Process trajectory data into training data.

        Shared by both in-process and subprocess execution paths.
        Handles None checks, completion retrieval, and data processing.

        Args:
            trajectory_data: Raw trajectory data from rollout (or None if failed)
            session: Proxy session used for this rollout
            task_id: Task identifier
            rollout_number: Index of this rollout within the group

        Returns:
            Processed training data dict, or None if processing failed
        """
        if trajectory_data is None:
            print(f"[StepWiseWorkflow] Rollout {rollout_number} returned None for task {task_id}")
            return None

        if not trajectory_data.get("trajectories"):
            print(f"[StepWiseWorkflow] No trajectories for task {task_id}, rollout {rollout_number}")
            return None

        # Get completions from proxy server session cache
        completions = self.proxy_server.session_cache[session.session_id].completions
        use_depth_weighting = self.config.depth_level_weighting
        use_depth_discount = self.config.depth_level_discount_gamma is not None

        # Process data
        train_data = get_train_data_for_trajectory_collection(
            trajectory_data,
            completions,
            task_id,
            self.filter_errors,
            self.reward_processor,
            self.merge_prefixes,
            concat_fn=concat_padded_tensors,
            include_traj_depth=use_depth_weighting or use_depth_discount,
            include_traj_start=use_depth_weighting,
        )

        if train_data is None:
            print(f"[StepWiseWorkflow] No train data for task {task_id}, rollout {rollout_number}")
            return None

        return train_data

    async def _arun_episode_subprocess_single(
        self,
        executor: "ProcessPoolExecutor",
        engine: InferenceEngine,
        data: dict,
        session: ArealProxySession,
        rollout_number: int,
    ) -> dict | None:
        """Run a single rollout in a subprocess and process the result.

        Args:
            executor: ProcessPoolExecutor to submit the rollout task to
            engine: Inference engine (used for versioning)
            data: Task data dict with 'task_id' key
            session: Pre-created proxy session for this rollout
            rollout_number: Index of this rollout within the group

        Returns:
            Processed training data dict, or None if rollout/processing failed
        """
        from dataclasses import asdict

        from platoon.train.areal.subprocess_worker import run_rollout_subprocess

        task_id = data["task_id"]

        # Prepare config for subprocess
        config = deepcopy(self.config)
        config.rollout_config.model_endpoint = session.session_base_url
        # Prepend 'openai/' to be compatible with LiteLLM
        config.rollout_config.model_name = "openai/" + config.rollout_config.model_name
        config.rollout_config.model_api_key = "None"
        config.rollout_config.output_dir = os.path.join(
            config.rollout_config.output_dir,
            str(engine.get_version()),
        )

        # Hard timeout in the parent: rollout timeout + AppWorld init budget (120s)
        # + grace for cleanup (60s). AppWorldEnv.__init__ runs synchronously and
        # can take up to 120s; this budget ensures the subprocess has time to
        # complete init + run the full rollout before the parent gives up.
        # If the subprocess is still stuck after this, SIGALRM will force-kill it.
        hard_timeout = (self.config.rollout_config.timeout or 900) + 120 + 60

        # Run rollout in subprocess. If subprocess/executor-level failures happen
        # (e.g. BrokenProcessPool), treat them as a failed rollout and continue.
        try:
            loop = asyncio.get_running_loop()
            trajectory_data = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    run_rollout_subprocess,
                    self.rollout_fn.__module__,
                    self.rollout_fn.__name__,
                    self.get_task_fn.__module__,
                    self.get_task_fn.__name__,
                    task_id,
                    asdict(config.rollout_config),  # Serialize config to dict
                ),
                timeout=hard_timeout,
            )
        except asyncio.TimeoutError:
            print(
                f"[StepWiseWorkflow] Subprocess hard timeout ({hard_timeout}s) for task {task_id}, "
                f"rollout {rollout_number} — subprocess will self-terminate via SIGALRM"
            )
            return None
        except Exception as e:
            print(
                f"[StepWiseWorkflow] Subprocess execution failed for task {task_id}, "
                f"rollout {rollout_number}: {e}"
            )
            return None

        # Process result using shared helper
        return self._process_trajectory_result(trajectory_data, session, task_id, rollout_number)

    async def _arun_episode_with_subprocesses(self, engine: InferenceEngine, data: dict) -> list[dict | None]:
        """Run rollouts in subprocess pool for isolation.

        This method provides an alternative to the in-process asyncio.gather() approach
        by running each rollout in a separate subprocess.

        Args:
            engine: Inference engine (used for versioning)
            data: Task data dict with 'task_id' key

        Returns:
            List of processed training data dicts (or None for failed rollouts)
        """
        from concurrent.futures import ProcessPoolExecutor

        # Create sessions in parent process
        # Sessions are created here so that the proxy server can track them
        sessions = []
        for i in range(self.config.group_size):
            session = ArealProxySession(base_url=self.proxy_url)
            await session.__aenter__()
            sessions.append(session)

        # Create the executor manually (not via `with`) so we can call
        # shutdown(wait=False): the context-manager form uses wait=True, which
        # would block here if any subprocess is stuck past its asyncio timeout.
        # Orphaned subprocesses will self-terminate via the SIGALRM set in
        # run_rollout_subprocess.
        executor = ProcessPoolExecutor(max_workers=self.config.group_size, mp_context=mp.get_context("spawn"))
        try:
            results = await asyncio.gather(
                *[
                    self._arun_episode_subprocess_single(executor, engine, data, session, i)
                    for i, session in enumerate(sessions)
                ]
            )

            return results

        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            for session in sessions:
                await session.__aexit__(None, None, None)

    async def _arun_episode_single(self, engine: InferenceEngine, data: dict, rollout_number: int) -> dict | None:
        """Run a single rollout and return training data (in-process version)."""
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

                trajectory_data = await asyncio.create_task(self.rollout_fn(task, config.rollout_config))

                return self._process_trajectory_result(trajectory_data, session, task_id, rollout_number)

        except Exception as e:
            import traceback

            print(f"[StepWiseWorkflow] Error in areal workflow for task {task_id} and rollout {rollout_number}: {e}")
            traceback.print_exc()
            return None
