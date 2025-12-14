from omegaconf import DictConfig
from skyrl_train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.generators.utils import get_rollout_metrics
from typing import Callable, List, Dict, Any, Optional, Tuple
from platoon.envs.base import Task
import asyncio
from copy import deepcopy
from platoon.episode.config import RolloutConfig

def get_training_data_for_step(
    step: dict,
    step_idx: int,
    trajectory_reward: float,
    is_last_step: bool,
) -> Optional[Dict[str, Any]]:
    """
    Extract training data from a single step.
    
    Args:
        step: The step dictionary from the trajectory
        step_idx: Index of the step in the trajectory
        trajectory_reward: The total reward for the trajectory
        is_last_step: Whether this is the last step in the trajectory
        
    Returns:
        Dict with prompt_token_ids, response_ids, loss_mask, reward, or None if no valid data
    """
    # Check if this step has the completion data we need
    if 'misc' not in step:
        return None
        
    action_misc = step.get('misc', {}).get('action_misc', {})
    if not action_misc:
        return None
        
    completion_data = action_misc.get('completion_data', {})
    if not completion_data:
        return None
    
    input_token_ids = completion_data.get('input_token_ids', [])
    output_token_ids = completion_data.get('output_token_ids', [])
    
    if not input_token_ids or not output_token_ids:
        return None
    
    # Create loss mask: 0 for prompt, 1 for response
    loss_mask = [1] * len(output_token_ids)
    
    # For per-token rewards, assign the trajectory reward to the last token of the last step
    # For intermediate steps, reward is 0
    per_token_reward = [0.0] * len(output_token_ids)
    if is_last_step and len(per_token_reward) > 0:
        per_token_reward[-1] = float(trajectory_reward)
    
    return {
        'prompt_token_ids': list(input_token_ids),
        'response_ids': list(output_token_ids),
        'loss_mask': loss_mask,
        'reward': per_token_reward,
        'stop_reason': 'stop',  # TODO: Revisit how to filter out errored and truncated steps ("length")
    }


def get_training_data_for_trajectory(
    trajectory: dict,
    trajectory_id: TrajectoryID,
) -> List[Dict[str, Any]]:
    """
    Extract training data from a trajectory (all steps).
    
    Args:
        trajectory: The trajectory dictionary
        trajectory_id: The trajectory ID object
        
    Returns:
        List of training data dicts, one per step
    """
    steps = trajectory.get('steps', [])
    trajectory_reward = trajectory.get('reward', 0.0)
    
    step_data_list = []
    for step_idx, step in enumerate(steps):
        is_last_step = (step_idx == len(steps) - 1)
        
        step_data = get_training_data_for_step(
            step=step,
            step_idx=step_idx,
            trajectory_reward=trajectory_reward,
            is_last_step=is_last_step,
        )
        
        if step_data is not None:
            # Add trajectory ID with step information
            step_trajectory_id = TrajectoryID(
                instance_id=trajectory_id.instance_id,
                repetition_id=trajectory_id.repetition_id,
            )
            step_data['trajectory_id'] = step_trajectory_id
            step_data['step'] = step_idx
            step_data['is_last_step'] = is_last_step
            step_data_list.append(step_data)
    
    return step_data_list


def construct_training_data_from_rollout(
    rollout: dict,
    trajectory_id: TrajectoryID,
) -> List[Dict[str, Any]]:
    """
    Convert a rollout (trajectory collection dict) into training data format.
    
    Args:
        rollout: The trajectory collection dictionary from to_dict()
        trajectory_id: The base trajectory ID for this rollout
        
    Returns:
        List of training data dicts, one per step across all trajectories
    """
    all_step_data = []
    
    trajectories = rollout.get('trajectories', {})
    
    for traj_idx, (traj_id, trajectory) in enumerate(trajectories.items()):
        # Create a unique trajectory ID for this trajectory within the rollout
        traj_trajectory_id = TrajectoryID(
            instance_id=trajectory_id.instance_id,
            repetition_id=traj_idx,
        )
        
        step_data_list = get_training_data_for_trajectory(trajectory, traj_trajectory_id)
        all_step_data.extend(step_data_list)
    
    return all_step_data


class PlatoonSkyRLGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        model_name: str,
        rollout_fn: Callable[[Task, dict], dict],
        get_task_fn: Callable[[str], Task]
    ): 
        self.config = generator_cfg
        
        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8002
        )
        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name
        
        self.rollout_fn = rollout_fn
        self.get_task_fn = get_task_fn
        self.rollout_config = RolloutConfig(
            **self.config.rollout_config  # TODO: Use a structured config?
        )
        self.rollout_config.model_name = self.litellm_model_name
        self.rollout_config.model_endpoint = self.base_url
        self.rollout_config.model_api_key = self.rollout_config.model_api_key or "None"
        self.rollout_config.return_dict = True
    
    async def arun_episode(self, task: Task) -> dict:
        """Run a single episode and return the trajectory collection dict."""
        rollout_config = deepcopy(self.rollout_config)
        rollout_config.max_steps = rollout_config.max_steps or task.max_steps
        return await asyncio.create_task(self.rollout_fn(task, rollout_config))
        
    async def generate(
        self, 
        input_batch: GeneratorInput,
    ) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.
        
        Returns outputs in the same order as the input batch, with one entry per step
        in each trajectory (step-wise training format).
        
        Args:
            input_batch: GeneratorInput with task information
            
        Returns:
            GeneratorOutput with step-wise training data
        """
        # Get trajectory IDs from input if available
        input_trajectory_ids = input_batch.get('trajectory_ids', None)
        print("input_batch")
        print(input_batch)
        
        # Filter to tasks with valid task_id
        valid_indices = []
        tasks: List[Task] = []
        for i, task_info in enumerate(input_batch['env_extras']):
            task_id = task_info.get('task_id') if task_info else None
            print(f"Loading task: {task_id}...")
            if task_id is not None:
                valid_indices.append(i)
                tasks.append(self.get_task_fn(task_id))
        
        print(f"Launching {len(tasks)} rollouts...")
        # Run all episodes concurrently
        trajectory_collections: List[dict] = await asyncio.gather(
            *[self.arun_episode(task) for task in tasks]
        )
        print(f"Rollouts completed.")

        print(trajectory_collections)
        # Aggregate results across all rollouts
        all_prompt_token_ids: List[List[int]] = []
        all_response_ids: List[List[int]] = []
        all_rewards: List[List[float]] = []
        all_loss_masks: List[List[int]] = []
        all_stop_reasons: List[str] = []
        all_trajectory_ids: List[TrajectoryID] = []
        all_is_last_step: List[bool] = []
        
        for idx, (valid_idx, rollout) in enumerate(zip(valid_indices, trajectory_collections)):
            # Get or create trajectory ID for this rollout
            if input_trajectory_ids is not None and valid_idx < len(input_trajectory_ids):
                base_trajectory_id = input_trajectory_ids[valid_idx]
            else:
                task_info = input_batch['env_extras'][valid_idx]
                task_id = task_info.get('task_id', f'task_{valid_idx}')
                base_trajectory_id = TrajectoryID(instance_id=task_id, repetition_id=0)
            
            # Extract training data from this rollout
            step_data_list = construct_training_data_from_rollout(rollout, base_trajectory_id)
            
            for step_data in step_data_list:
                all_prompt_token_ids.append(step_data['prompt_token_ids'])
                all_response_ids.append(step_data['response_ids'])
                all_rewards.append(step_data['reward'])
                all_loss_masks.append(step_data['loss_mask'])
                all_stop_reasons.append(step_data['stop_reason'])
                all_trajectory_ids.append(step_data['trajectory_id'])
                all_is_last_step.append(step_data['is_last_step'])
        
        # Compute rollout metrics
        #rollout_metrics = get_rollout_metrics(all_response_ids, all_rewards)
        rollout_metrics = {}
        
        # Add custom metrics
        rollout_metrics['generate/num_trajectories'] = len(trajectory_collections)
        rollout_metrics['generate/num_steps'] = len(all_response_ids)
        if trajectory_collections:
            total_reward = sum(
                sum(traj.get('reward', 0.0) for traj in rollout.get('trajectories', {}).values())
                for rollout in trajectory_collections
            )
            num_trajectories = sum(
                len(rollout.get('trajectories', {}))
                for rollout in trajectory_collections
            )
            if num_trajectories > 0:
                rollout_metrics['generate/avg_trajectory_reward'] = total_reward / num_trajectories
        
        generator_output: GeneratorOutput = {
            "prompt_token_ids": all_prompt_token_ids,
            "response_ids": all_response_ids,
            "rewards": all_rewards,
            "loss_masks": all_loss_masks,
            "stop_reasons": all_stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,  # Not captured in this generator
            "trajectory_ids": all_trajectory_ids,
            "is_last_step": all_is_last_step,
        }
        
        return generator_output
