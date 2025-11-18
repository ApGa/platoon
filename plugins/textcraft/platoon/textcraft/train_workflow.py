from areal.utils.data import concat_padded_tensors
from .rollout import run_rollout, run_recursive_rollout
from .tasks import get_task
from copy import deepcopy
from platoon.train.rl import WorkflowConfig
from platoon.train.proxy import ArealProxySession
from areal.experimental.openai.proxy import ProxyServer
from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import stats_tracker  
import os
import torch
from areal.experimental.openai.client import InteractionWithTokenLogpReward
import asyncio
    
def get_train_data_for_step(
    step: dict,
    completions: dict[str, InteractionWithTokenLogpReward],
    task_id: str,
    ) -> dict | None:
    
    if 'action_misc' in step['misc'] and 'areal_completion_data' in step['misc']['action_misc'] and 'completion_id' in step['misc']['action_misc']:
        print(f"No train data found for step {step['id']} for task {task_id}")
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
    ) -> dict:
    train_data = []
    const_found_train_data = 0
    for step in trajectory['steps']:
        step_train_data = get_train_data_for_step(step, completions, task_id)
        if step_train_data:
            const_found_train_data += 1
            step_train_data['rewards'] = torch.tensor([trajectory['reward']])
            # Make rewards 2D [1, seq_len] so it is split/packed per-token with the batch
            seq_len = step_train_data['attention_mask'].shape[1]
            step_train_data['token_rewards'] = torch.full(
                (1, seq_len), float(trajectory['reward']), dtype=torch.float32
            )
            train_data.append(step_train_data)
    # TODO: Use logger instead of print
    print(f"Found {const_found_train_data} / {len(trajectory['steps'])} train data for for task {task_id} and trajectory {trajectory['id']}")
    return concat_padded_tensors(train_data) | { 'num_steps': torch.tensor([float(len(trajectory['steps']))]) }

def get_train_data_for_trajectory_collection(
    trajectory_collection: dict, 
    completions: dict[str, InteractionWithTokenLogpReward],
    task_id: str,
    ) -> dict:
    
    train_data = []
    for trajectory in trajectory_collection['trajectories'].values():
        train_data.append(get_train_data_for_trajectory(trajectory, completions, task_id))
    return concat_padded_tensors(train_data) | {
        'task_reward': torch.tensor(list(trajectory_collection['trajectories'].values())[0]['reward']).unsqueeze(0)
    }

class TextCraftArealWorkflow(RolloutWorkflow):
    def __init__(self, config: WorkflowConfig, proxy_server: ProxyServer, stats_scope: str, device: torch.device):
        self.config = config
        self.config.rollout_config.return_dict = True
        self.config.rollout_config.train = True
        self.proxy_server = proxy_server
        self.api_version = "v1"
        self.proxy_url = f"{self.proxy_server.public_addr}/{self.api_version}"
        self.stats_scope = stats_scope
        self.device = device
        
    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        
        config = deepcopy(self.config)
        try:
            task_id = data['task_id']
            task = get_task(task_id)
            if config.rollout_config.max_steps is None:
                config.rollout_config.max_steps = task.max_steps
            
            async with ArealProxySession(base_url=self.proxy_url) as session:
                
                config.rollout_config.model_endpoint = session.session_base_url
                
                config.rollout_config.output_dir = os.path.join(
                    config.rollout_config.output_dir,
                    str(engine.get_version()),
                )
                
                results = await asyncio.create_task(run_recursive_rollout(task, config.rollout_config))
                
                completions = self.proxy_server.session_cache[session.session_id].completions
                train_data = get_train_data_for_trajectory_collection(results, completions, task_id)
               
            
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
            
            
            return train_data
            
        except Exception as e:
            print(f"Error in areal workflow: {e}")
            return None
