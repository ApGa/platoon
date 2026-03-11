"""Data processing utilities for AReaL RL training.

This module provides functions to convert agent trajectories to AReaL training format,
including sequence merging for efficiency.

The sequence accumulation implementation mirrors the approach in tinker_data_processing.py,
enabling prefix-aware merging where consecutive steps whose observations are prefixes
of subsequent observations are merged into single sequences.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

logger = logging.getLogger(__name__)


class CompletionResponse(Protocol):
    """Protocol for completion response objects."""

    input_tokens: list[int]
    output_tokens: list[int]
    input_len: int
    output_len: int
    output_logprobs: list[float]
    output_versions: list[int]


class CompletionWithResponse(Protocol):
    """Protocol for completion objects that contain a model_response."""

    model_response: CompletionResponse


@dataclass
class SequenceAccumulator:
    """Accumulates tokens across steps to enable sequence merging.

    When step N+1's observation is a prefix of step N's observation + action,
    we can merge them into a single sequence for more efficient training.

    Note: num_input_tokens and num_output_tokens track the ORIGINAL per-step
    counts (before merging), not the merged sequence lengths. This ensures
    metrics are consistent whether merging is enabled or not.
    """

    full_sequence: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    # Track original token counts (before merging) for consistent metrics
    num_input_tokens: int = 0  # Sum of full observation lengths per step
    num_output_tokens: int = 0  # Sum of action lengths per step

    def clear(self):
        self.full_sequence = []
        self.logprobs = []
        self.loss_mask = []
        self.versions = []
        self.num_input_tokens = 0
        self.num_output_tokens = 0

    def to_train_data(self, trajectory_reward: float) -> dict:
        """Convert accumulated data to training format."""
        seq_len = len(self.full_sequence)
        return dict(
            input_ids=torch.tensor(self.full_sequence).unsqueeze(0),
            loss_mask=torch.tensor(self.loss_mask).unsqueeze(0),
            logprobs=torch.tensor(self.logprobs).unsqueeze(0),
            versions=torch.tensor(self.versions).unsqueeze(0),
            attention_mask=torch.ones(seq_len, dtype=torch.bool).unsqueeze(0),
            num_input_tokens=torch.tensor(self.num_input_tokens, dtype=torch.float32).unsqueeze(0),
            num_output_tokens=torch.tensor(self.num_output_tokens, dtype=torch.float32).unsqueeze(0),
            rewards=torch.tensor([trajectory_reward]),
            token_rewards=torch.full((1, seq_len), float(trajectory_reward), dtype=torch.float32),
        )


def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def get_train_data_for_step(
    step: dict,
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    filter_errors: bool = False,
    trajectory_reward: float = 0.0,
) -> dict | None:
    """Extract training data from a single step (non-aggregated version).

    Args:
        step: Step dictionary from trajectory.
        completions: Dict mapping completion_id to completion data.
        task_id: Task identifier for logging.
        filter_errors: Whether to filter out error steps from successful trajectories.
        trajectory_reward: Reward for the trajectory (used for error filtering).

    Returns:
        Training data dict or None if step should be skipped.
    """
    if "action_misc" not in step.get("misc", {}) or "completion_id" not in step["misc"]["action_misc"]:
        return None

    # Only filter error steps from trajectories with reward >= 1 (successful trajectories)
    if (
        filter_errors
        and trajectory_reward >= 1
        and (
            ("error" in step and step["error"])
            or ("output" in step and step["output"] and "traceback" in step["output"].lower())
        )
    ):
        error_info = step.get("error") or step.get("output", "Unknown error")
        logger.debug(f"Filtering Step: Error in step for task {task_id}: {error_info}")
        return None

    completion_id = step["misc"]["action_misc"]["completion_id"]
    completion = completions[completion_id].model_response

    seq = list(completion.input_tokens) + list(completion.output_tokens)
    logprobs = [0.0] * completion.input_len + list(completion.output_logprobs)
    loss_mask = [0] * completion.input_len + [1] * completion.output_len
    versions = [-1] * completion.input_len + list(completion.output_versions)
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
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    trajectory_id: str,
    filter_errors: bool = False,
    reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
    merge_prefixes: bool = True,
    concat_fn: Callable[[list[dict]], dict] | None = None,
) -> dict | None:
    """Extract training data from a trajectory with optional prefix merging.

    When merge_prefixes=True (default), sequences are merged when step N+1's
    observation is a prefix of the accumulated sequence, reducing redundant
    computation during training.

    Args:
        trajectory: Trajectory dictionary containing steps.
        completions: Dict mapping completion_id to completion data.
        task_id: Task identifier for logging.
        trajectory_id: Trajectory identifier for logging.
        filter_errors: Whether to filter out error steps.
        reward_processor: Function to process trajectory rewards.
        merge_prefixes: Whether to merge prefix sequences (default True).
        concat_fn: Function to concatenate training data dicts (required).

    Returns:
        Training data dict or None if no valid data found.
    """
    if concat_fn is None:
        raise ValueError("concat_fn is required for get_train_data_for_trajectory")

    trajectory_reward, trajectory_rewards_dict = reward_processor(trajectory)

    if not merge_prefixes:
        # Fall back to non-aggregated version
        return _get_train_data_for_trajectory_no_merge(
            trajectory,
            completions,
            task_id,
            trajectory_id,
            filter_errors,
            trajectory_reward,
            trajectory_rewards_dict,
            concat_fn,
        )

    train_data = []
    accumulator = SequenceAccumulator()
    count_found = 0
    total_input_tokens = 0
    total_output_tokens = 0
    num_merged = 0

    for i, step in enumerate(trajectory["steps"]):
        # Check if we should skip this step
        if "action_misc" not in step.get("misc", {}) or "completion_id" not in step["misc"]["action_misc"]:
            continue

        # Filter error steps from successful trajectories
        if (
            filter_errors
            and trajectory_reward >= 1
            and (
                ("error" in step and step["error"])
                or ("output" in step and step["output"] and "traceback" in step["output"].lower())
            )
        ):
            continue

        completion_id = step["misc"]["action_misc"]["completion_id"]
        if completion_id not in completions:
            logger.warning(f"Completion ID {completion_id} not found for task {task_id}")
            continue

        completion = completions[completion_id].model_response
        count_found += 1

        # Get observation (input) and action (output) tokens
        ob_tokens = list(completion.input_tokens)
        ac_tokens = list(completion.output_tokens)
        ac_logprobs = list(completion.output_logprobs)
        ac_versions = list(completion.output_versions)

        # Track token counts (before merging) for overall trajectory stats
        total_input_tokens += len(ob_tokens)
        total_output_tokens += len(ac_tokens)

        # Determine if we can extend the current sequence or need to start fresh
        if len(accumulator.full_sequence) == 0:
            # First step - start new accumulator
            delta_ob_tokens = ob_tokens
        elif _is_prefix(accumulator.full_sequence, ob_tokens):
            # Observation extends the current sequence - we can merge!
            delta_ob_tokens = ob_tokens[len(accumulator.full_sequence) :]
            num_merged += 1
        else:
            # New sequence doesn't extend current - flush and start new
            # Debug: show why prefix check failed (only for first failure per trajectory)
            # if num_merged == 0 and len(train_data) == 0:
            #     acc_len = len(accumulator.full_sequence)
            #     ob_len = len(ob_tokens)
            #     # Check where they diverge
            #     diverge_idx = 0
            #     for idx in range(min(acc_len, ob_len)):
            #         if accumulator.full_sequence[idx] != ob_tokens[idx]:
            #             diverge_idx = idx
            #             break
            #     else:
            #         diverge_idx = min(acc_len, ob_len)
            #     print(f"[MergeDebug] Task {task_id}: Prefix check failed at step {i}")
            #     print(f"  accumulated_len={acc_len}, observation_len={ob_len}, diverge_at={diverge_idx}")
            #     # completion is already the ModelResponse object
            #     tokenizer = getattr(completion, 'tokenizer', None)
            #     if tokenizer is not None and diverge_idx < acc_len and diverge_idx < ob_len:
            #         # Show tokens and decoded text around divergence point
            #         start = max(0, diverge_idx - 10)
            #         end = min(diverge_idx + 10, min(acc_len, ob_len))
            #         acc_slice = accumulator.full_sequence[start:end]
            #         ob_slice = ob_tokens[start:end]
            #         print(f"  accumulated[{start}:{end}] tokens: {acc_slice}")
            #         print(f"  accumulated[{start}:{end}] decoded: {repr(tokenizer.decode(acc_slice))}")
            #         print(f"  observation[{start}:{end}] tokens: {ob_slice}")
            #         print(f"  observation[{start}:{end}] decoded: {repr(tokenizer.decode(ob_slice))}")
            train_data.append(accumulator.to_train_data(trajectory_reward))
            accumulator.clear()
            delta_ob_tokens = ob_tokens

        # Add observation tokens (with 0.0 logprobs and 0 loss_mask - don't train on prompts)
        accumulator.full_sequence.extend(delta_ob_tokens)
        accumulator.logprobs.extend([0.0] * len(delta_ob_tokens))
        accumulator.loss_mask.extend([0] * len(delta_ob_tokens))
        accumulator.versions.extend([-1] * len(delta_ob_tokens))
        # Track FULL observation length for consistent metrics (not delta)
        accumulator.num_input_tokens += len(ob_tokens)

        # Add action tokens (with actual logprobs, loss_mask=1, and versions)
        accumulator.full_sequence.extend(ac_tokens)
        accumulator.logprobs.extend(ac_logprobs)
        accumulator.loss_mask.extend([1] * len(ac_tokens))
        accumulator.versions.extend(ac_versions)
        accumulator.num_output_tokens += len(ac_tokens)

    # Flush remaining accumulated data
    if accumulator.full_sequence:
        train_data.append(accumulator.to_train_data(trajectory_reward))

    print(
        f"[DataProcessing] Task {task_id} trajectory {trajectory_id}: "
        f"Found {count_found} steps, merged {num_merged}, produced {len(train_data)} datums"
    )

    if not train_data:
        logger.debug(f"No train data found for trajectory {trajectory_id} for task {task_id}")
        return None

    concat_result = concat_fn(train_data)
    # Sum token counts across all sequences to get trajectory-level totals
    # This ensures num_input_tokens and num_output_tokens have shape [1] like num_steps
    trajectory_num_input_tokens = concat_result["num_input_tokens"].sum().unsqueeze(0)
    trajectory_num_output_tokens = concat_result["num_output_tokens"].sum().unsqueeze(0)

    return concat_result | {
        "num_steps": torch.tensor([float(len(trajectory["steps"]))]),
        "num_input_tokens": trajectory_num_input_tokens,
        "num_output_tokens": trajectory_num_output_tokens,
        **{key: torch.tensor(value).unsqueeze(0) for key, value in trajectory_rewards_dict.items()},
    }


def _get_train_data_for_trajectory_no_merge(
    trajectory: dict,
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    trajectory_id: str,
    filter_errors: bool,
    trajectory_reward: float,
    trajectory_rewards_dict: dict,
    concat_fn: Callable[[list[dict]], dict],
) -> dict | None:
    """Non-aggregated version for comparison/fallback."""
    train_data = []
    count_found_train_data = 0

    for i, step in enumerate(trajectory["steps"]):
        step_train_data = get_train_data_for_step(step, completions, task_id, filter_errors, trajectory_reward)
        if step_train_data:
            count_found_train_data += 1
            step_train_data["rewards"] = torch.tensor([trajectory_reward])
            seq_len = step_train_data["attention_mask"].shape[1]
            step_train_data["token_rewards"] = torch.full((1, seq_len), float(trajectory_reward), dtype=torch.float32)
            train_data.append(step_train_data)
        else:
            logger.debug(f"No train data found for step {i} for task {task_id}")

    logger.debug(
        f"Found {count_found_train_data} / {len(trajectory['steps'])} train data "
        f"for task {task_id} and trajectory {trajectory_id}"
    )

    if not train_data:
        logger.debug(f"No train data found for trajectory {trajectory_id} for task {task_id}")
        return None

    concat_result = concat_fn(train_data)
    # Sum token counts across all sequences to get trajectory-level totals
    # This ensures num_input_tokens and num_output_tokens have shape [1] like num_steps
    trajectory_num_input_tokens = concat_result["num_input_tokens"].sum().unsqueeze(0)
    trajectory_num_output_tokens = concat_result["num_output_tokens"].sum().unsqueeze(0)

    return concat_result | {
        "num_steps": torch.tensor([float(len(trajectory["steps"]))]),
        "num_input_tokens": trajectory_num_input_tokens,
        "num_output_tokens": trajectory_num_output_tokens,
        **{key: torch.tensor(value).unsqueeze(0) for key, value in trajectory_rewards_dict.items()},
    }


def _compute_trajectory_depths(trajectory_collection: dict) -> dict[str, int]:
    """Compute the depth of each trajectory in the rollout tree.

    Root trajectory is depth 0, its direct children are depth 1, etc.
    Depth is determined by following parent_info links.

    Args:
        trajectory_collection: Dict with 'trajectories' key mapping traj_id to traj dict.

    Returns:
        Dict mapping trajectory_id to its depth in the tree.
    """
    trajectories = trajectory_collection.get("trajectories", {})
    if not trajectories:
        return {}

    traj_ids = list(trajectories.keys())
    root_id = traj_ids[0]

    parents: dict[str, str | None] = {}
    for traj_id, traj in trajectories.items():
        parent_info = traj.get("parent_info")
        parent_id = parent_info.get("id") if isinstance(parent_info, dict) else None
        parents[traj_id] = parent_id

    depth_cache: dict[str, int] = {}

    def _depth_for(traj_id: str) -> int:
        if traj_id in depth_cache:
            return depth_cache[traj_id]
        if traj_id == root_id or parents.get(traj_id) is None or parents[traj_id] not in parents:
            depth_cache[traj_id] = 0
            return 0
        d = _depth_for(parents[traj_id]) + 1
        depth_cache[traj_id] = d
        return d

    return {tid: _depth_for(tid) for tid in traj_ids}


def get_train_data_for_trajectory_collection(
    trajectory_collection: dict,
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    filter_errors: bool = False,
    reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
    merge_prefixes: bool = True,
    concat_fn: Callable[[list[dict]], dict] | None = None,
    include_traj_depth: bool = False,
    include_traj_start: bool = False,
) -> dict | None:
    """Extract training data from all trajectories in a collection.

    Args:
        trajectory_collection: Collection of trajectories.
        completions: Dict mapping completion_id to completion data.
        task_id: Task identifier for logging.
        filter_errors: Whether to filter out error steps.
        reward_processor: Function to process trajectory rewards.
        merge_prefixes: Whether to merge prefix sequences for efficiency.
        concat_fn: Function to concatenate training data dicts (required).
        include_traj_depth: Whether to include per-datum trajectory depth labels.
        include_traj_start: Whether to mark the first datum of each trajectory.

    Returns:
        Training data dict or None if no valid data found.
    """
    if concat_fn is None:
        raise ValueError("concat_fn is required for get_train_data_for_trajectory_collection")

    depth_map = _compute_trajectory_depths(trajectory_collection) if include_traj_depth else {}

    train_data = []
    for trajectory_id, trajectory in trajectory_collection["trajectories"].items():
        trajectory_data = get_train_data_for_trajectory(
            trajectory, completions, task_id, trajectory_id, filter_errors, reward_processor, merge_prefixes, concat_fn
        )
        if trajectory_data is not None:
            if include_traj_depth and trajectory_id in depth_map:
                num_datums = trajectory_data["rewards"].shape[0]
                trajectory_data["traj_depth"] = torch.full(
                    (num_datums,), float(depth_map[trajectory_id]), dtype=torch.float32
                )
                if include_traj_start:
                    # Mark the first datum of this trajectory so we can count
                    # distinct trajectories (not datums) at each depth level.
                    traj_start = torch.zeros(num_datums, dtype=torch.float32)
                    traj_start[0] = 1.0
                    trajectory_data["traj_start"] = traj_start
            train_data.append(trajectory_data)

    if not train_data:
        logger.debug(f"No train data found for any trajectory for task {task_id}")
        return None

    root_reward, root_rewards_dict = reward_processor(list(trajectory_collection["trajectories"].values())[0])

    return concat_fn(train_data) | {
        "task_reward": torch.tensor(root_reward).unsqueeze(0),
        **{f"root_{key}": torch.tensor(value).unsqueeze(0) for key, value in root_rewards_dict.items()},
    }
