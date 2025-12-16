from areal.utils.data import concat_padded_tensors
from copy import deepcopy
from platoon.train.rl import WorkflowConfig
from platoon.train.proxy import ArealProxySession
from areal.experimental.openai.proxy import ProxyServer
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import stats_tracker  
import os
import torch
import traceback
from areal.experimental.openai.client import InteractionWithTokenLogpReward
import asyncio
from typing import Callable
from platoon.envs.base import Task
    
def get_train_data_for_step(
    step: dict,
    completions: dict[str, InteractionWithTokenLogpReward],
    task_id: str,
    filter_errors: bool = False,
    ) -> dict | None:
    
    if 'action_misc' not in step['misc'] or 'completion_id' not in step['misc']['action_misc']:
        return None
    
    if filter_errors and (('error' in step and step['error']) or ('output' in step and step['output'] and 'traceback' in step['output'].lower())):
        error_info = step.get('error') or step.get('output', 'Unknown error')
        print(f"Filtering Step: Error in step for task {task_id}: {error_info}")
        return None
    
    completion_id = step['misc']['action_misc']['completion_id']
    completion = completions[completion_id].model_response
    
    seq = completion.input_tokens + completion.output_tokens
    logprobs = [0.0] * completion.input_len + completion.output_logprobs
    loss_mask = [0] * completion.input_len + [1] * completion.output_len
    versions = [-1] * completion.input_len + completion.output_versions
    attention_mask = torch.ones(len(seq), dtype=torch.bool).unsqueeze(0)
    num_input_tokens = torch.tensor(completion.input_len, dtype=torch.float32).unsqueeze(0)
    num_output_tokens = torch.tensor(completion.output_len, dtype=torch.float32).unsqueeze(0)
    
    return dict(
        input_ids=torch.tensor(seq).unsqueeze(0),
        loss_mask=torch.tensor(loss_mask).unsqueeze(0),
        logprobs=torch.tensor(logprobs).unsqueeze(0),
        versions=torch.tensor(versions).unsqueeze(0),
        attention_mask=attention_mask,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
    )
    
def get_train_data_for_trajectory(
    trajectory: dict,
    completions: dict[str, InteractionWithTokenLogpReward],
    task_id: str,
    trajectory_id: str,
    filter_errors: bool = False,
    ) -> dict | None:
    train_data = []
    count_found_train_data = 0
    for i, step in enumerate(trajectory['steps']):
        step_train_data = get_train_data_for_step(step, completions, task_id, filter_errors)
        if step_train_data:
            count_found_train_data += 1
            step_train_data['rewards'] = torch.tensor([trajectory['reward']])
            # Make rewards 2D [1, seq_len] so it is split/packed per-token with the batch
            seq_len = step_train_data['attention_mask'].shape[1]
            step_train_data['token_rewards'] = torch.full(
                (1, seq_len), float(trajectory['reward']), dtype=torch.float32
            )
            train_data.append(step_train_data)
        else:
            print(f"No train data found for step {i} for task {task_id}")
    # TODO: Use logger instead of print
    print(f"Found {count_found_train_data} / {len(trajectory['steps'])} train data for task {task_id} and trajectory {trajectory_id}")
    
    if not train_data:
        print(f"No train data found for trajectory {trajectory_id} for task {task_id}")
        return None
    
    return concat_padded_tensors(train_data) | { 'num_steps': torch.tensor([float(len(trajectory['steps']))]) }

def get_train_data_for_trajectory_collection(
    trajectory_collection: dict, 
    completions: dict[str, InteractionWithTokenLogpReward],
    task_id: str,
    filter_errors: bool = False,
    ) -> dict | None:
    
    train_data = []
    for trajectory_id, trajectory in trajectory_collection['trajectories'].items():
        trajectory_data = get_train_data_for_trajectory(trajectory, completions, task_id, trajectory_id, filter_errors)
        if trajectory_data is not None:
            train_data.append(trajectory_data)
    
    if not train_data:
        print(f"No train data found for any trajectory for task {task_id}")
        return None
    
    return concat_padded_tensors(train_data) | {
        'task_reward': torch.tensor(list(trajectory_collection['trajectories'].values())[0]['reward']).unsqueeze(0)
    }

class StepWiseArealWorkflow(RolloutWorkflow):
    def __init__(self, rollout_fn: Callable[[Task, dict], dict], get_task_fn: Callable[[str], Task], config: WorkflowConfig, proxy_server: ProxyServer, stats_scope: str, device: torch.device, filter_errors: bool = False):
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

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:

        results = await asyncio.gather(
            *[self.arun_episode_single(engine, data, i) for i in range(self.config.rollout_config.group_size)]
        )
        results = [result for result in results if result is not None]
        if not results:
            print(f"No results found for task {data['task_id']}")
            return None
        
        train_data = concat_padded_tensors(results)

        mean_unprocessed_reward = torch.mean(train_data['rewards'])

        # TODO: Make this configurable.
        train_data['rewards'] = train_data['rewards'] - torch.mean(train_data['task_reward'])

        tracker = stats_tracker.get(self.stats_scope)
        
        reward_mask = torch.ones_like(train_data['task_reward'], dtype=torch.bool).to(self.device)
        output_token_mask = torch.ones_like(train_data['num_output_tokens'], dtype=torch.bool).to(self.device)
        input_token_mask = torch.ones_like(train_data['num_input_tokens'], dtype=torch.bool).to(self.device)
        num_steps_mask = torch.ones_like(train_data['num_steps'], dtype=torch.bool).to(self.device)
        
        tracker.denominator(task_reward_mask=reward_mask, num_output_tokens_mask=output_token_mask, num_input_tokens_mask=input_token_mask, num_steps_mask=num_steps_mask)
        tracker.stat(task_reward=train_data['task_reward'].to(self.device), denominator="task_reward_mask")
        tracker.stat(num_output_tokens=train_data['num_output_tokens'].to(self.device), denominator="num_output_tokens_mask")
        tracker.stat(num_input_tokens=train_data['num_input_tokens'].to(self.device), denominator="num_input_tokens_mask")
        tracker.stat(num_steps=train_data['num_steps'].to(self.device), denominator="num_steps_mask")
        
        # task_reward @ K metrics (computed per-task across K rollouts)
        task_rewards = train_data['task_reward'].to(self.device)
        task_reward_at_k_mask = torch.ones(1, dtype=torch.bool).to(self.device)
        tracker.denominator(task_reward_at_k_mask=task_reward_at_k_mask)
        tracker.stat(task_reward_at_k_mean=torch.mean(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")
        tracker.stat(task_reward_at_k_max=torch.max(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")
        tracker.stat(task_reward_at_k_min=torch.min(task_rewards).unsqueeze(0), denominator="task_reward_at_k_mask")

        if train_data['rewards'].max() == train_data['rewards'].min() and len(results) > 1:
            print(f"All rewards are the same for task {data['task_id']}: {mean_unprocessed_reward.item():.2f}")
            return None

        return train_data

    async def arun_episode_single(self, engine: InferenceEngine, data: dict, rollout_number: int) -> dict | None:
        
        config = deepcopy(self.config)
        try:
            task_id = data['task_id']
            task = self.get_task_fn(task_id)
            if config.rollout_config.max_steps is not None:
                task.max_steps = config.rollout_config.max_steps
            
            async with ArealProxySession(base_url=self.proxy_url) as session:
                
                config.rollout_config.model_endpoint = session.session_base_url
                
                config.rollout_config.output_dir = os.path.join(
                    config.rollout_config.output_dir,
                    str(engine.get_version()),
                )
                
                results = await asyncio.create_task(self.rollout_fn(task, config.rollout_config))
                
                if not results['trajectories']:
                    print(f"No trajectories found for task {task_id} and rollout {rollout_number}")
                    return None
                

                # TODO: This needs to only be done when training. Maybe we can make proxy server optional and only use it when training.
                completions = self.proxy_server.session_cache[session.session_id].completions
                train_data = get_train_data_for_trajectory_collection(results, completions, task_id, self.filter_errors)
                
                if train_data is None:
                    print(f"No train data found for task {task_id} and rollout {rollout_number}")
                    return None
            
            
            return train_data
            
        except Exception as e:
            print(f"Error in areal workflow for task {task_id} and rollout {rollout_number}: {e}\n{traceback.format_exc()}")
            return None
