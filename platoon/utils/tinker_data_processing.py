"""Data processing utilities for tinker RL training.

This module provides functions to convert agent trajectories to tinker.Datum format
for training, including sequence merging for efficiency.

Sequence accumulation implementation is inspired by tinker_cookbook.rl.data_processing.trajectory_to_data:
https://raw.githubusercontent.com/thinking-machines-lab/tinker-cookbook/refs/heads/main/tinker_cookbook/rl/data_processing.py
"""

import logging
from dataclasses import dataclass, field
from typing import Callable

import tinker
import torch

from tinker import TensorData

from platoon.train.tinker.proxy import TinkerLLMInteraction

logger = logging.getLogger(__name__)


def create_rightshifted_model_input_and_leftshifted_targets(
    chunks: list[tinker.ModelInputChunk],
) -> tuple[tinker.ModelInput, list[int]]:
    """
    Given a full sequence of model input chunks, create
     "inputs" (with last token removed); these are also list[ModelInputChunk] because text+images
     "targets" (with first token removed); these are list[int] text tokens

     Taken from https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/supervised/common.py
    """
    assert len(chunks) >= 1, "must have at least one chunk"

    last_chunk = chunks[-1]
    if not isinstance(last_chunk, tinker.types.EncodedTextChunk):
        raise ValueError(
            "The last chunk must be a text chunk. This is because images are 0-loss anyways, so we should remove them beforehand."
        )

    total_length = sum(c.length for c in chunks)
    if total_length < 2:
        raise ValueError("need at least 2 tokens for input/target split")

    # Build input chunks: all but last, then append truncated last chunk
    input_chunks: list[tinker.ModelInputChunk] = list(chunks[:-1])
    if last_chunk.length > 1:
        input_chunks.append(tinker.types.EncodedTextChunk(tokens=last_chunk.tokens[:-1]))

    # Build target tokens: collect all tokens, then slice off first
    all_tokens: list[int] = []
    for chunk in chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            all_tokens.extend(chunk.tokens)
        else:
            all_tokens.extend([0] * chunk.length)
    target_tokens = all_tokens[1:]

    return tinker.ModelInput(chunks=input_chunks), target_tokens


@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory."""
    trajectory_id: str
    reward: float
    num_steps: int
    num_input_tokens: int
    num_output_tokens: int
    num_datums: int
    rewards_dict: dict[str, float] = field(default_factory=dict)
    is_root: bool = False


@dataclass
class TrajectoryCollectionResult:
    """Result from processing a trajectory collection."""
    datums: list[tinker.Datum]
    task_reward: float  # Reward of the root trajectory
    trajectory_stats: list[TrajectoryStats]
    root_rewards_dict: dict[str, float]  # Reward components from root trajectory


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    """Flatten ModelInput chunks into a list of ints and special chunks."""
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    """Get the token length of a flattened observation."""
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    """Convert a flattened observation back to a ModelInput."""
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=list(current_text_chunk)))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


@dataclass
class SequenceAccumulator:
    """Accumulates tokens across steps to enable sequence merging."""
    full_sequence: FlatOb = field(default_factory=list)
    sampled_logprobs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)

    def clear(self):
        self.full_sequence = []
        self.sampled_logprobs = []
        self.advantages = []
        self.mask = []


def make_datum_from_accumulator(
    accumulator: SequenceAccumulator,
    checkpoint_version: int,
) -> tinker.Datum:
    """Create a tinker.Datum from the accumulated sequence data.
    
    Following the format from tinker_cookbook.rl.data_processing.trajectory_to_data.
    """
    all_tokens_T = _flat_ob_to_model_input(accumulator.full_sequence)
    input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
        list(all_tokens_T.chunks)
    )
    sampled_logprobs_T = accumulator.sampled_logprobs[1:]
    advantages_T = accumulator.advantages[1:]
    mask_T = accumulator.mask[1:]
    
    assert (
        input_tokens_T.length
        == len(target_tokens_T)
        == len(sampled_logprobs_T)
        == len(advantages_T)
        == len(mask_T)
    ), f"Length mismatch: {input_tokens_T.length} vs {len(target_tokens_T)} vs {len(sampled_logprobs_T)} vs {len(advantages_T)} vs {len(mask_T)}"
    
    return tinker.Datum(
        model_input=input_tokens_T,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
            "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
            "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
            "mask": TensorData.from_torch(torch.tensor(mask_T)),
            # Store checkpoint_version for staleness checking (will be stripped before forward_backward)
            "checkpoint_version": TensorData.from_torch(torch.tensor([checkpoint_version])),
        },
    )


@dataclass
class TrajectoryDataResult:
    """Result from processing a single trajectory."""
    datums: list[tinker.Datum]
    num_steps: int  # Number of valid steps found
    num_input_tokens: int
    num_output_tokens: int


def trajectory_to_data(
    trajectory: dict,
    interactions: dict[str, TinkerLLMInteraction],
    task_id: str,
    trajectory_id: str,
    trajectory_advantage: float,
    checkpoint_version: int,
    filter_errors: bool = False,
    trajectory_reward: float = 0.0,
) -> TrajectoryDataResult:
    """Convert a trajectory to training data, merging sequences when possible.
    
    If observations are prefixes of subsequent observations (i.e., the sequence 
    grows by appending), we can merge them into a single Datum for efficiency.
    
    Args:
        trajectory: Trajectory dictionary containing steps.
        interactions: Dict mapping completion_id to TinkerLLMInteraction.
        task_id: Task identifier for logging.
        trajectory_id: Trajectory identifier for logging.
        trajectory_advantage: The advantage for this trajectory (reward - mean_reward).
        checkpoint_version: The checkpoint version for staleness checking.
        filter_errors: Whether to filter out error steps.
        trajectory_reward: The reward for the trajectory (used for error filtering).
        
    Returns:
        TrajectoryDataResult with datums and statistics.
    """
    data: list[tinker.Datum] = []
    accumulator = SequenceAccumulator()
    count_found = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, step in enumerate(trajectory['steps']):
        # Check if we should skip this step
        if 'action_misc' not in step.get('misc', {}) or 'completion_id' not in step['misc']['action_misc']:
            continue
        
        # Filter error steps from successful trajectories
        if filter_errors and trajectory_reward >= 1:
            has_error = ('error' in step and step['error']) or \
                        ('output' in step and step['output'] and 'traceback' in step['output'].lower())
            if has_error:
                continue
        
        completion_id = step['misc']['action_misc']['completion_id']
        if completion_id not in interactions:
            logger.warning(f"Completion ID {completion_id} not found in interactions for task {task_id}")
            continue
        
        interaction = interactions[completion_id]
        count_found += 1
        
        # Get observation and action
        ob = interaction.obs
        ob_flat = _flatten_chunks(list(ob.chunks))
        ac_tokens = list(interaction.action.tokens)
        ac_logprobs = list(interaction.action.logprobs)
        
        # Track token counts per step (before merging)
        step_input_tokens = _flat_ob_token_len(ob_flat)
        step_output_tokens = len(ac_tokens)
        total_input_tokens += step_input_tokens
        total_output_tokens += step_output_tokens
        
        # Determine if we can extend the current sequence or need to start fresh
        if len(accumulator.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(accumulator.full_sequence, ob_flat):
            # Observation extends the current sequence - we can merge
            delta_ob_flat = ob_flat[len(accumulator.full_sequence):]
        else:
            # New sequence doesn't extend current - flush and start new
            data.append(make_datum_from_accumulator(accumulator, checkpoint_version))
            accumulator.clear()
            delta_ob_flat = ob_flat
        
        # Add observation tokens (with 0.0 logprobs and 0.0 mask - don't train on prompts)
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        accumulator.full_sequence.extend(delta_ob_flat)
        accumulator.sampled_logprobs.extend([0.0] * delta_ob_len)
        accumulator.advantages.extend([0.0] * delta_ob_len)
        accumulator.mask.extend([0.0] * delta_ob_len)
        
        # Add action tokens (with actual logprobs and advantages)
        accumulator.full_sequence.extend(ac_tokens)
        accumulator.sampled_logprobs.extend(ac_logprobs)
        accumulator.advantages.extend([trajectory_advantage] * len(ac_tokens))
        accumulator.mask.extend([1.0] * len(ac_tokens))
    
    # Flush remaining accumulated data
    if accumulator.full_sequence:
        data.append(make_datum_from_accumulator(accumulator, checkpoint_version))
    
    logger.debug(f"Found {count_found} steps, produced {len(data)} datums for task {task_id} trajectory {trajectory_id}")
    
    return TrajectoryDataResult(
        datums=data,
        num_steps=count_found,
        num_input_tokens=total_input_tokens,
        num_output_tokens=total_output_tokens,
    )


def get_train_data_for_trajectory_collection(
    trajectory_collection: dict,
    interactions: dict[str, TinkerLLMInteraction],
    task_id: str,
    checkpoint_version: int,
    filter_errors: bool = False,
    reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj['reward'], {})
) -> TrajectoryCollectionResult:
    """Extract training data from all trajectories in a collection.
    
    A trajectory collection may contain multiple trajectories when using multi-agent
    rollouts. The first trajectory is the "root" trajectory, and others are subagent
    trajectories.
    
    Args:
        trajectory_collection: Dictionary with 'trajectories' key mapping to trajectory dicts.
        interactions: Dict mapping completion_id to TinkerLLMInteraction.
        task_id: Task identifier for logging.
        checkpoint_version: The checkpoint version for staleness checking.
        filter_errors: Whether to filter out error steps.
        reward_processor: Function to process trajectory rewards.
        
    Returns:
        TrajectoryCollectionResult with datums and per-trajectory statistics.
    """
    train_data: list[tinker.Datum] = []
    trajectory_stats: list[TrajectoryStats] = []
    task_reward = 0.0
    root_rewards_dict: dict[str, float] = {}
    is_first = True
    
    for trajectory_id, trajectory in trajectory_collection['trajectories'].items():
        trajectory_reward, rewards_dict = reward_processor(trajectory)
        
        # Store the root (first) trajectory's reward as the task reward
        if is_first:
            task_reward = trajectory_reward
            root_rewards_dict = rewards_dict
        
        # Advantage will be set later after all rollouts complete
        result = trajectory_to_data(
            trajectory=trajectory,
            interactions=interactions,
            task_id=task_id,
            trajectory_id=trajectory_id,
            trajectory_advantage=trajectory_reward,  # Will be further processed later.
            checkpoint_version=checkpoint_version,
            filter_errors=filter_errors,
            trajectory_reward=trajectory_reward,
        )
        
        train_data.extend(result.datums)
        
        # Record per-trajectory stats
        trajectory_stats.append(TrajectoryStats(
            trajectory_id=trajectory_id,
            reward=trajectory_reward,
            num_steps=result.num_steps,
            num_input_tokens=result.num_input_tokens,
            num_output_tokens=result.num_output_tokens,
            num_datums=len(result.datums),
            rewards_dict=rewards_dict,
            is_root=is_first,
        ))
        
        is_first = False
    
    if not train_data:
        logger.warning(f"No train data found for any trajectory for task {task_id}")
    
    return TrajectoryCollectionResult(
        datums=train_data,
        task_reward=task_reward,
        trajectory_stats=trajectory_stats,
        root_rewards_dict=root_rewards_dict,
    )

