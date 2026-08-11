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

from platoon.agents.actions.subagent import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
)
from platoon.train.tinker.proxy import TinkerLLMInteraction
from platoon.utils.subagent_sampling import SubagentDatumSampler
from platoon.utils.trajectory_error_filtering import (
    ERROR_ACTION_MASK_KEY,
    completion_id_for_step,
    detected_error_completion_ids,
)
from platoon.utils.trajectory_status import trajectory_was_interrupted

logger = logging.getLogger(__name__)


def _exclude_from_training(trajectory: dict) -> bool:
    misc = trajectory.get("misc", {})
    if isinstance(misc, dict) and bool(misc.get(EXCLUDE_FROM_TRAINING_MISC_KEY)):
        return True
    task = trajectory.get("task")
    task_misc = task.get("misc", {}) if isinstance(task, dict) else {}
    return isinstance(task_misc, dict) and bool(task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY))


def _exclude_from_policy_training(trajectory: dict) -> bool:
    """Return whether policy datums should be dropped while retaining stats."""

    misc = trajectory.get("misc", {})
    return isinstance(misc, dict) and bool(misc.get(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY))


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
        raise ValueError("The last chunk must be a text chunk. Images are 0-loss anyways, so remove them beforehand.")

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


_SAMPLING_COUNT_FIELDS = (
    "eligible_datums",
    "retained_datums",
    "eligible_attention_tokens",
    "retained_attention_tokens",
    "eligible_loss_tokens",
    "retained_loss_tokens",
)


@dataclass
class DatumSamplingCounts:
    """Post-merge datum and token counts for one depth level."""

    eligible_datums: int = 0
    retained_datums: int = 0
    eligible_attention_tokens: int = 0
    retained_attention_tokens: int = 0
    eligible_loss_tokens: int = 0
    retained_loss_tokens: int = 0

    def merge(self, other: "DatumSamplingCounts") -> None:
        for field_name in _SAMPLING_COUNT_FIELDS:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass
class SubagentDatumSamplingStats:
    """Sampling counters retained with a Tinker rollout collection result."""

    by_depth: dict[int, DatumSamplingCounts] = field(default_factory=dict)

    def record(self, datum: tinker.Datum, *, depth: int, retained: bool) -> None:
        counts = self.by_depth.setdefault(depth, DatumSamplingCounts())
        attention_tokens = int(datum.model_input.length)
        loss_tokens = int(datum.loss_fn_inputs["mask"].to_torch().sum().item())
        counts.eligible_datums += 1
        counts.eligible_attention_tokens += attention_tokens
        counts.eligible_loss_tokens += loss_tokens
        if retained:
            counts.retained_datums += 1
            counts.retained_attention_tokens += attention_tokens
            counts.retained_loss_tokens += loss_tokens

    def merge(self, other: "SubagentDatumSamplingStats") -> None:
        for depth, other_counts in other.by_depth.items():
            self.by_depth.setdefault(depth, DatumSamplingCounts()).merge(other_counts)

    def to_metrics(self) -> dict[str, float]:
        """Render backend-aligned overall and per-depth sampling metrics."""

        total = DatumSamplingCounts()
        metrics: dict[str, float] = {}
        for depth, counts in sorted(self.by_depth.items()):
            total.merge(counts)
            prefix = f"subagent_sampling/depth_{depth}"
            for field_name in _SAMPLING_COUNT_FIELDS:
                metrics[f"{prefix}/{field_name}"] = float(getattr(counts, field_name))
        for field_name in _SAMPLING_COUNT_FIELDS:
            metrics[f"subagent_sampling/{field_name}"] = float(getattr(total, field_name))
        return metrics


@dataclass
class TrajectoryCollectionResult:
    """Result from processing a trajectory collection."""

    datums: list[tinker.Datum]
    task_reward: float  # Reward of the root trajectory
    trajectory_stats: list[TrajectoryStats]
    root_rewards_dict: dict[str, float]  # Reward components from root trajectory
    subagent_sampling_stats: SubagentDatumSamplingStats = field(default_factory=SubagentDatumSamplingStats)
    num_policy_excluded_datums: int = 0
    # Interrupted roots may carry a partial reward which is useful for metrics,
    # but must not participate in group advantage centering.
    task_reward_valid: bool = True


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
    error_action_mask: list[bool] | None = None

    def clear(self):
        self.full_sequence = []
        self.sampled_logprobs = []
        self.advantages = []
        self.mask = []
        if self.error_action_mask is not None:
            self.error_action_mask = []


def make_datum_from_accumulator(
    accumulator: SequenceAccumulator,
    checkpoint_version: int,
    traj_depth: int | None = None,
    traj_start: bool = False,
) -> tinker.Datum:
    """Create a tinker.Datum from the accumulated sequence data.

    Following the format from tinker_cookbook.rl.data_processing.trajectory_to_data.
    """
    all_tokens_T = _flat_ob_to_model_input(accumulator.full_sequence)
    input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(list(all_tokens_T.chunks))
    sampled_logprobs_T = accumulator.sampled_logprobs[1:]
    advantages_T = accumulator.advantages[1:]
    mask_T = accumulator.mask[1:]

    assert (
        input_tokens_T.length == len(target_tokens_T) == len(sampled_logprobs_T) == len(advantages_T) == len(mask_T)
    ), (
        f"Length mismatch: input={input_tokens_T.length} target={len(target_tokens_T)} logprobs={len(sampled_logprobs_T)}"  # noqa: E501
    )

    loss_fn_inputs = {
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
        "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
        "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
        "mask": TensorData.from_torch(torch.tensor(mask_T)),
        # Store checkpoint_version for staleness checking (will be stripped before forward_backward)
        "checkpoint_version": TensorData.from_torch(torch.tensor([checkpoint_version])),
    }
    if accumulator.error_action_mask is not None:
        if len(accumulator.error_action_mask) != len(accumulator.full_sequence):
            raise ValueError(
                "error-action mask/token alignment mismatch: "
                f"tokens={len(accumulator.full_sequence)}, mask={len(accumulator.error_action_mask)}"
            )
        loss_fn_inputs[ERROR_ACTION_MASK_KEY] = TensorData.from_torch(
            torch.tensor(accumulator.error_action_mask[1:], dtype=torch.bool)
        )
    if traj_depth is not None:
        loss_fn_inputs["traj_depth"] = TensorData.from_torch(torch.tensor([traj_depth], dtype=torch.long))
        loss_fn_inputs["traj_start"] = TensorData.from_torch(torch.tensor([1.0 if traj_start else 0.0]))

    return tinker.Datum(
        model_input=input_tokens_T,
        loss_fn_inputs=loss_fn_inputs,
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
    traj_depth: int | None = None,
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
        filter_errors: Whether to mark erroneous completion tokens for deferred filtering.
        trajectory_reward: Retained for call-site compatibility. Error
            filtering is deferred until centered advantages are available.

    Returns:
        TrajectoryDataResult with datums and statistics.
    """
    data: list[tinker.Datum] = []
    accumulator = SequenceAccumulator(error_action_mask=[] if filter_errors else None)
    count_found = 0
    total_input_tokens = 0
    total_output_tokens = 0
    seen_completion_ids: set[str] = set()
    deferred_error_completion_ids = (
        detected_error_completion_ids(trajectory.get("steps", ()))
        if filter_errors
        else set()
    )

    for i, step in enumerate(trajectory["steps"]):
        # Check if we should skip this step
        completion_id = completion_id_for_step(step)
        if completion_id is None:
            continue

        if completion_id in seen_completion_ids:
            continue

        if completion_id not in interactions:
            logger.warning(f"Completion ID {completion_id} not found in interactions for task {task_id}")
            continue

        interaction = interactions[completion_id]
        # OpenHands can serialize one parallel LLM response over several
        # environment steps. Its sampled tokens belong in the batch once.
        seen_completion_ids.add(completion_id)
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
            delta_ob_flat = ob_flat[len(accumulator.full_sequence) :]
        else:
            # New sequence doesn't extend current - flush and start new
            data.append(
                make_datum_from_accumulator(
                    accumulator,
                    checkpoint_version,
                    traj_depth=traj_depth,
                    traj_start=len(data) == 0,
                )
            )
            accumulator.clear()
            delta_ob_flat = ob_flat

        # Add observation tokens (with 0.0 logprobs and 0.0 mask - don't train on prompts)
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        accumulator.full_sequence.extend(delta_ob_flat)
        accumulator.sampled_logprobs.extend([0.0] * delta_ob_len)
        accumulator.advantages.extend([0.0] * delta_ob_len)
        accumulator.mask.extend([0.0] * delta_ob_len)
        if accumulator.error_action_mask is not None:
            accumulator.error_action_mask.extend([False] * delta_ob_len)

        # Add action tokens (with actual logprobs and advantages)
        accumulator.full_sequence.extend(ac_tokens)
        accumulator.sampled_logprobs.extend(ac_logprobs)
        accumulator.advantages.extend([trajectory_advantage] * len(ac_tokens))
        accumulator.mask.extend([1.0] * len(ac_tokens))
        if accumulator.error_action_mask is not None:
            accumulator.error_action_mask.extend(
                [completion_id in deferred_error_completion_ids] * len(ac_tokens)
            )

    # Flush remaining accumulated data
    if accumulator.full_sequence:
        data.append(
            make_datum_from_accumulator(
                accumulator,
                checkpoint_version,
                traj_depth=traj_depth,
                traj_start=len(data) == 0,
            )
        )

    logger.debug(
        f"Found {count_found} steps, produced {len(data)} datums for task {task_id} trajectory {trajectory_id}"
    )

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
    reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
    include_traj_depth: bool = False,
    include_traj_start: bool = False,
    subagent_datum_sampler: SubagentDatumSampler | None = None,
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
        filter_errors: Whether to mark erroneous completion tokens for deferred filtering.
        reward_processor: Function to process trajectory rewards.
        subagent_datum_sampler: Optional deterministic sampler applied to each
            non-root trajectory after its steps have been merged into datums.

    Returns:
        TrajectoryCollectionResult with datums and per-trajectory statistics.
    """
    train_data: list[tinker.Datum] = []
    trajectory_stats: list[TrajectoryStats] = []
    task_reward = 0.0
    root_rewards_dict: dict[str, float] = {}
    sampling_stats = SubagentDatumSamplingStats()
    num_policy_excluded_datums = 0
    trajectories = trajectory_collection["trajectories"]
    root_trajectory_id = next(iter(trajectories), None)
    task_reward_valid = True
    if root_trajectory_id is not None:
        task_reward_valid = not trajectory_was_interrupted(trajectories[root_trajectory_id])
    need_depths = include_traj_depth or subagent_datum_sampler is not None
    depth_map = _compute_trajectory_depths(trajectory_collection) if need_depths else {}

    # Process every trainable trajectory's reward before sampling any datums.
    # Recursive reward processors may depend on child outcomes attached to the
    # complete tree, and reward/stat records must not depend on which child
    # datums happen to survive the Bernoulli draw.
    processed_trajectories: list[tuple[str, dict, float, dict[str, float], bool, int, bool]] = []
    for trajectory_id, trajectory in trajectories.items():
        if _exclude_from_training(trajectory):
            continue
        trajectory_reward, rewards_dict = reward_processor(trajectory)
        is_root = trajectory_id == root_trajectory_id
        depth = depth_map.get(trajectory_id, 0)
        # Roots are mandatory policy data. The source marker is child-only,
        # but keep that invariant explicit at the converter boundary.
        policy_ineligible = trajectory_was_interrupted(trajectory) or (
            not is_root and _exclude_from_policy_training(trajectory)
        )
        processed_trajectories.append(
            (trajectory_id, trajectory, trajectory_reward, rewards_dict, is_root, depth, policy_ineligible)
        )

    if processed_trajectories:
        root_entry = next((entry for entry in processed_trajectories if entry[4]), processed_trajectories[0])
        task_reward = root_entry[2]
        root_rewards_dict = root_entry[3]

    for (
        trajectory_id,
        trajectory,
        trajectory_reward,
        rewards_dict,
        is_root,
        depth,
        policy_ineligible,
    ) in processed_trajectories:
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
            traj_depth=depth if include_traj_depth else None,
        )

        if policy_ineligible:
            # Keep reward processing and full TrajectoryStats above/below, but
            # never train policy tokens from a child whose verifier failed or
            # did not finish. These datums are outside Bernoulli eligibility.
            keep_mask = [False] * len(result.datums)
            num_policy_excluded_datums += len(result.datums)
        elif subagent_datum_sampler is None:
            keep_mask = [True] * len(result.datums)
        else:
            keep_mask = subagent_datum_sampler.sample_mask(
                task_id=task_id,
                trajectory_id=trajectory_id,
                depth=depth,
                num_datums=len(result.datums),
            )
            if len(keep_mask) != len(result.datums):
                raise ValueError(
                    "subagent datum sampler returned a mask with the wrong length: "
                    f"trajectory={trajectory_id}, datums={len(result.datums)}, mask={len(keep_mask)}"
                )

        retained_datums: list[tinker.Datum] = []
        for datum, retained in zip(result.datums, keep_mask, strict=True):
            if subagent_datum_sampler is not None and not policy_ineligible:
                sampling_stats.record(datum, depth=depth, retained=retained)
            if retained:
                retained_datums.append(datum)

        # Sampling may remove the datum which originally carried traj_start.
        # Mark the first *retained* datum so depth weighting counts only
        # trajectories represented in the actual training batch.
        for retained_index, datum in enumerate(retained_datums):
            if include_traj_start:
                datum.loss_fn_inputs["traj_start"] = TensorData.from_torch(
                    torch.tensor([1.0 if retained_index == 0 else 0.0])
                )
            else:
                datum.loss_fn_inputs.pop("traj_start", None)

        train_data.extend(retained_datums)

        # Record per-trajectory stats
        trajectory_stats.append(
            TrajectoryStats(
                trajectory_id=trajectory_id,
                reward=trajectory_reward,
                num_steps=result.num_steps,
                num_input_tokens=result.num_input_tokens,
                num_output_tokens=result.num_output_tokens,
                # Keep trajectory stats sampling-independent. Dedicated
                # sampling metrics expose the retained datum count.
                num_datums=len(result.datums),
                rewards_dict=rewards_dict,
                is_root=is_root,
            )
        )

    if not train_data:
        logger.warning(f"No train data found for any trajectory for task {task_id}")

    return TrajectoryCollectionResult(
        datums=train_data,
        task_reward=task_reward,
        trajectory_stats=trajectory_stats,
        root_rewards_dict=root_rewards_dict,
        subagent_sampling_stats=sampling_stats,
        num_policy_excluded_datums=num_policy_excluded_datums,
        task_reward_valid=task_reward_valid,
    )


def _compute_trajectory_depths(trajectory_collection: dict) -> dict[str, int]:
    """Compute trajectory depth from parent links in the rollout tree."""
    trajectories = trajectory_collection.get("trajectories", {})
    if not trajectories:
        return {}

    traj_ids = list(trajectories.keys())
    root_id = traj_ids[0]
    parents: dict[str, str | None] = {}
    for traj_id, traj in trajectories.items():
        parent_info = traj.get("parent_info")
        parents[traj_id] = parent_info.get("id") if isinstance(parent_info, dict) else None

    depth_cache: dict[str, int] = {}

    def _depth_for(traj_id: str) -> int:
        if traj_id in depth_cache:
            return depth_cache[traj_id]
        parent_id = parents.get(traj_id)
        if traj_id == root_id or parent_id is None or parent_id not in parents:
            depth_cache[traj_id] = 0
            return 0
        depth_cache[traj_id] = _depth_for(parent_id) + 1
        return depth_cache[traj_id]

    return {traj_id: _depth_for(traj_id) for traj_id in traj_ids}
