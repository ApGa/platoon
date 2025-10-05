from __future__ import annotations

from platoon.utils.llm_client import LLMClient
from areal.experimental.openai import ArealOpenAI
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.data import concat_padded_tensors
from openai.types.chat import ChatCompletion
from platoon.episode.trajectory import Trajectory, TrajectoryStep
from platoon.visualization.event_sinks import TrajectoryEventHandler
from contextvars import ContextVar
import torch
from typing import Any
from platoon.episode.context import current_trajectory

areal_llm_clients: ContextVar[dict[str, ArealLLMClient]] = ContextVar("areal_llm_clients", default={})

class ArealLLMClient(LLMClient): #TODO: Decide if we want to add this to create_llm_client or not.
    def __init__(self, model: str, engine: str):
        self.engine = engine
        self.tokenizer = load_hf_tokenizer(model)
        self.async_client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        
        curr_traj=current_trajectory.get(None)
        if curr_traj is not None:
            areal_llm_clients.get()[curr_traj.id] = self
        else:
            areal_llm_clients.get()[0] = self # TODO: Hack of using 0 for the first trajectory id. We need to fix this.
        
    async def async_chat_completion(self, *args: Any, **kwargs: Any) -> ChatCompletion:
        # TODO: We should think of an alternate solution. 
        # This is a hack to include stop tokens in the logged response tokens for training.
        if 'stop' in kwargs:
            kwargs.pop('stop')
        return await super().async_chat_completion(*args, **kwargs)
            
        
    def chat_completion(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ArealLLMClient does not support synchronous chat completion. Please use async_chat_completion instead.")


    def fork(self) -> ArealLLMClient:
        return ArealLLMClient(model=self.model, engine=self.engine)


class ArealEventSink(TrajectoryEventHandler):
    
    def on_trajectory_step_added(self, trajectory: Trajectory, step: TrajectoryStep) -> None:
        completion_id = step.misc['action_misc']['completion_id']
        client = areal_llm_clients.get()[trajectory.id] if trajectory.id in areal_llm_clients.get() else areal_llm_clients.get()[0] # TODO: Hack of using 0 for the first trajectory id. We need to fix this.
        if client is None:
            raise ValueError(f"ArealLLMClient not found for trajectory {trajectory.id}")
        
        areal_completion = client.async_client.get_completion(completion_id=completion_id)
        seq = areal_completion.input_tokens + areal_completion.output_tokens
        logprobs = [0.0] * areal_completion.input_len + areal_completion.output_logprobs
        loss_mask = [0] * areal_completion.input_len + [1] * areal_completion.output_len
        versions = [-1] * areal_completion.input_len + areal_completion.output_versions
        
        step.misc['action_misc']['areal_completion_data'] = dict(
            # unsqueeze to add an additional batch dimension
            input_ids=torch.tensor(seq).unsqueeze(0),
            loss_mask=torch.tensor(loss_mask).unsqueeze(0),
            logprobs=torch.tensor(logprobs).unsqueeze(0),
            versions=torch.tensor(versions).unsqueeze(0),
            attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
        )
        
        
    def on_trajectory_finished(self, trajectory: Trajectory) -> None:
        areal_completion_data_list = []
        for step in trajectory.steps:
            step_completion_data = step.misc['action_misc']['areal_completion_data']
            step_completion_data['rewards'] = torch.tensor([trajectory.reward])
            areal_completion_data_list.append(step_completion_data)
            
        trajectory.misc['areal_completion_data'] = concat_padded_tensors(areal_completion_data_list)
