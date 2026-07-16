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

from platoon.agents.actions.subagent import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
)
from platoon.utils.subagent_sampling import SubagentDatumSampler
from platoon.utils.trajectory_status import trajectory_was_interrupted

logger = logging.getLogger(__name__)

OPTIONAL_REWARD_METRIC_MASK_PREFIX = "_platoon_reward_metric_present/"
SUBAGENT_DATUM_KEEP_MASK_KEY = "_platoon_subagent_datum_keep"
SUBAGENT_DATUM_DEPTH_KEY = "_platoon_subagent_datum_depth"
POLICY_TRAINING_ELIGIBILITY_MASK_KEY = "_platoon_policy_training_eligible"


def reward_metric_presence_key(metric_key: str) -> str:
    return f"{OPTIONAL_REWARD_METRIC_MASK_PREFIX}{metric_key}"


def _exclude_from_training(trajectory: dict) -> bool:
    misc = trajectory.get("misc", {})
    if isinstance(misc, dict) and bool(misc.get(EXCLUDE_FROM_TRAINING_MISC_KEY)):
        return True
    # A hard process kill can happen before the verifier's trajectory-level
    # exclusion marker is written. Its forked task is tagged before launch, so
    # this also makes partial/event-replayed verifier trajectories safe.
    task = trajectory.get("task")
    task_misc = task.get("misc", {}) if isinstance(task, dict) else {}
    return isinstance(task_misc, dict) and bool(task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY))


def _exclude_from_policy_training(trajectory: dict) -> bool:
    """Return the child-only policy exclusion without dropping reward data."""

    misc = trajectory.get("misc", {})
    return isinstance(misc, dict) and bool(misc.get(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY))


def harmonize_optional_reward_metrics(train_data: list[dict]) -> list[dict]:
    """Zero-fill optional reward metrics while preserving presence masks.

    Reward processors may emit trajectory-specific telemetry.  For example, a
    judged recursive child has ``reward/subagent_judgment`` while its root does
    not.  AReaL's concatenator intentionally rejects dictionaries with
    different keys, so make only the optional reward namespaces uniform.  A
    synthetic zero is accompanied by a false presence mask, allowing reporting
    to distinguish "not applicable" from a genuine score of zero.  Any
    non-reward key mismatch is left intact for the concatenator to reject.

    Each item supplies its own zero template.  This matters when harmonizing
    complete group rollouts: different rollouts can contain different numbers
    of trajectories, so copying the shape of the item that introduced a metric
    would create a wrongly sized reward vector.
    """

    reward_prefixes = ("reward/", "root_reward/")
    keys_by_prefix = {
        prefix: {
            key
            for item in train_data
            for key in item
            if key.startswith(prefix)
        }
        for prefix in reward_prefixes
    }

    masked_keys = {
        prefix: {
            key
            for key in reward_keys
            if any(key not in item for item in train_data)
            or any(reward_metric_presence_key(key) in item for item in train_data)
        }
        for prefix, reward_keys in keys_by_prefix.items()
    }

    harmonized = []
    for item in train_data:
        normalized = dict(item)
        for prefix, reward_keys in keys_by_prefix.items():
            template = next(
                (value for key, value in item.items() if key.startswith(prefix)),
                None,
            )
            for key in reward_keys:
                mask_key = reward_metric_presence_key(key)
                if key in item:
                    if key in masked_keys[prefix] and mask_key not in normalized:
                        normalized[mask_key] = torch.ones_like(item[key], dtype=torch.bool)
                    continue
                if template is None:
                    raise ValueError(
                        f"Cannot infer the local shape for missing {prefix} metric: {key}"
                    )
                normalized[key] = torch.zeros_like(template)
                normalized[mask_key] = torch.zeros_like(template, dtype=torch.bool)
        harmonized.append(normalized)
    return harmonized


class CompletionWithResponse(Protocol):
    """Protocol for exported AReaL completion records."""

    def to_tensor_dict(self) -> dict[str, torch.Tensor]: ...


@dataclass(frozen=True)
class RouterReplayConfig:
    """Shape and value contract for rollout-time MoE router replay data."""

    num_layers: int
    topk: int
    # Qwen3.5/3.6 route among 256 experts. Keeping this explicit prevents a
    # lossy uint8 conversion from silently wrapping an unexpected model.
    max_expert_id: int = 255

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("router replay num_layers must be positive")
        if self.topk <= 0:
            raise ValueError("router replay topk must be positive")
        if not 0 <= self.max_expert_id <= torch.iinfo(torch.uint8).max:
            raise ValueError("router replay max_expert_id must fit in uint8")

    @property
    def width(self) -> int:
        return self.num_layers * self.topk


@dataclass(frozen=True)
class _CompletionTokens:
    observation: list[int]
    action: list[int]
    action_logprobs: list[float]
    action_versions: list[int]
    routed_experts: torch.Tensor | None = None
    routed_experts_valid: torch.Tensor | None = None


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
    router_replay_config: RouterReplayConfig | None = None
    routed_expert_chunks: list[torch.Tensor] = field(default_factory=list)
    routed_expert_valid_chunks: list[torch.Tensor] = field(default_factory=list)

    def clear(self):
        self.full_sequence = []
        self.logprobs = []
        self.loss_mask = []
        self.versions = []
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.routed_expert_chunks = []
        self.routed_expert_valid_chunks = []

    def merge_completion_routes(
        self,
        routes: torch.Tensor | None,
        valid: torch.Tensor | None,
        prefix_len: int,
    ) -> None:
        """Merge one full-request route matrix into the accumulated token path.

        Existing valid rows belong to the completion that originally produced
        those tokens and are never overwritten. A later prefixed request may
        fill an invalid prefix row -- normally the former terminal token -- and
        contributes rows for only its newly appended observation/action suffix.
        """

        if self.router_replay_config is None:
            if routes is not None or valid is not None:
                raise ValueError("received routed experts while router replay is disabled")
            return
        if routes is None or valid is None:
            raise ValueError("router replay requires both routed_experts and routed_experts_valid")
        expected_shape = (
            routes.shape[0],
            self.router_replay_config.num_layers,
            self.router_replay_config.topk,
        )
        if tuple(routes.shape) != expected_shape or tuple(valid.shape) != (routes.shape[0],):
            raise ValueError(
                "invalid aligned routed-expert shape: "
                f"routes={tuple(routes.shape)}, valid={tuple(valid.shape)}, expected={expected_shape}"
            )
        if prefix_len != len(self.full_sequence) or prefix_len > routes.shape[0]:
            raise ValueError(
                f"invalid route prefix length {prefix_len} for accumulated length "
                f"{len(self.full_sequence)} and completion length {routes.shape[0]}"
            )

        # Fill only previously missing prefix routes. This handles the causal
        # S-1 convention: once another turn is appended, the old final token is
        # no longer terminal and its route appears in the new full request.
        offset = 0
        for old_routes, old_valid in zip(
            self.routed_expert_chunks,
            self.routed_expert_valid_chunks,
            strict=True,
        ):
            end = offset + old_valid.numel()
            new_valid = valid[offset:end]
            fill = ~old_valid & new_valid
            if fill.any():
                old_routes[fill] = routes[offset:end][fill]
                old_valid[fill] = True
            offset = end
        if offset != prefix_len:
            raise ValueError(f"route chunks cover {offset} tokens but accumulated prefix has {prefix_len}")

        self.routed_expert_chunks.append(routes[prefix_len:].clone())
        self.routed_expert_valid_chunks.append(valid[prefix_len:].clone())

    def to_train_data(self, trajectory_reward: float) -> dict:
        """Convert accumulated data to training format."""
        seq_len = len(self.full_sequence)
        result = dict(
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
        if self.router_replay_config is not None:
            routes = torch.cat(self.routed_expert_chunks, dim=0)
            valid = torch.cat(self.routed_expert_valid_chunks, dim=0)
            if routes.shape[0] != seq_len or valid.shape != (seq_len,):
                raise ValueError(
                    "routed-expert/token alignment mismatch: "
                    f"tokens={seq_len}, routes={tuple(routes.shape)}, valid={tuple(valid.shape)}"
                )
            expected_valid = torch.ones(seq_len, dtype=torch.bool)
            expected_valid[-1] = False
            if not torch.equal(valid, expected_valid):
                missing = torch.nonzero(expected_valid & ~valid, as_tuple=False).flatten().tolist()
                raise ValueError(
                    "router replay is missing routes for non-terminal token positions "
                    f"{missing[:16]}{'...' if len(missing) > 16 else ''}"
                )
            result["routed_experts"] = routes.unsqueeze(0)
            result["routed_experts_valid"] = valid.unsqueeze(0)
        return result


def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
    """Check if seq1 is a prefix of seq2."""
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _extract_completion_tokens(
    completion_record: CompletionWithResponse,
    router_replay_config: RouterReplayConfig | None = None,
) -> _CompletionTokens | None:
    """Return prompt/output token parts from an exported AReaL interaction."""

    tensor_dict = completion_record.to_tensor_dict()
    input_ids = tensor_dict["input_ids"].squeeze(0).tolist()
    loss_mask = tensor_dict["loss_mask"].squeeze(0).tolist()
    logprobs = tensor_dict["logprobs"].squeeze(0).tolist()
    versions = tensor_dict["versions"].squeeze(0).tolist()

    output_start = next((idx for idx, value in enumerate(loss_mask) if value), None)
    if output_start is None:
        return None

    ob_tokens = [int(token) for token in input_ids[:output_start]]
    ac_tokens = [int(token) for token, mask in zip(input_ids, loss_mask, strict=True) if mask]
    ac_logprobs = [float(logprob) for logprob, mask in zip(logprobs, loss_mask, strict=True) if mask]
    ac_versions = [int(version) for version, mask in zip(versions, loss_mask, strict=True) if mask]
    routes = None
    routes_valid = None
    if router_replay_config is not None:
        routes, routes_valid = _extract_aligned_routed_experts(
            tensor_dict,
            sequence_len=len(input_ids),
            config=router_replay_config,
        )
    return _CompletionTokens(
        observation=ob_tokens,
        action=ac_tokens,
        action_logprobs=ac_logprobs,
        action_versions=ac_versions,
        routed_experts=routes,
        routed_experts_valid=routes_valid,
    )


def _extract_aligned_routed_experts(
    tensor_dict: dict[str, torch.Tensor],
    *,
    sequence_len: int,
    config: RouterReplayConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and align proxy route rows to a completion's token sequence.

    SGLang returns one row for each causal model input, hence ``S - 1`` rows
    for an ``S``-token prompt+completion. The returned tensors always have an
    explicit row for every token; the terminal row is zero-filled and invalid.
    Expert zero is valid, so validity is never inferred from ID values.
    """

    aligned = torch.zeros(
        (sequence_len, config.num_layers, config.topk),
        dtype=torch.uint8,
    )
    aligned_valid = torch.zeros(sequence_len, dtype=torch.bool)

    raw_routes = tensor_dict.get("routed_experts")
    raw_valid = tensor_dict.get("routed_experts_valid")
    if raw_valid is None:
        raw_valid = tensor_dict.get("routed_experts_valid_mask")
    if raw_routes is None and raw_valid is None:
        raise ValueError(
            "router replay is enabled but an exported completion has no routed-expert data; "
            "verify rollout.return_routed_experts=true and the OpenAI proxy transport patch"
        )
    if raw_routes is None or raw_valid is None:
        raise ValueError("exported completion must contain both routed_experts and explicit routed_experts_valid")
    if raw_routes.ndim < 3 or raw_routes.shape[0] != 1:
        raise ValueError(f"exported routed_experts must have shape [1, rows, ...], got {tuple(raw_routes.shape)}")
    if raw_valid.ndim != 2 or tuple(raw_valid.shape[:1]) != (1,):
        raise ValueError(f"exported routed_experts_valid must have shape [1, rows], got {tuple(raw_valid.shape)}")

    routes = raw_routes.squeeze(0).flatten(start_dim=1)
    valid = raw_valid.squeeze(0).to(dtype=torch.bool)
    if routes.shape[0] != valid.numel():
        raise ValueError(f"routed expert row/mask mismatch: routes={routes.shape[0]}, valid={valid.numel()}")
    if routes.shape[1] != config.width:
        raise ValueError(
            "routed expert width mismatch: "
            f"got {routes.shape[1]}, expected {config.num_layers}*{config.topk}={config.width}"
        )
    if torch.is_floating_point(routes):
        if not torch.isfinite(routes).all() or not torch.equal(routes, routes.trunc()):
            raise ValueError("routed expert IDs must be finite integers")
    if routes.numel() and not (routes.dtype == torch.uint8 and config.max_expert_id == 255):
        minimum_tensor, maximum_tensor = torch.aminmax(routes)
        minimum = int(minimum_tensor.item())
        maximum = int(maximum_tensor.item())
        if minimum < 0 or maximum > config.max_expert_id:
            raise ValueError(
                f"routed expert IDs must be in [0, {config.max_expert_id}], observed [{minimum}, {maximum}]"
            )

    expected_rows = max(sequence_len - 1, 0)
    if routes.shape[0] != expected_rows:
        direction = "extra" if routes.shape[0] > expected_rows else "missing"
        raise ValueError(
            f"routed experts have {direction} rows: got {routes.shape[0]} for a "
            f"{sequence_len}-token completion, expected exactly S-1={expected_rows}. "
            "This indicates failed pause/resume stitching or a changed SGLang route contract."
        )
    if not valid.all():
        missing = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise ValueError(
            "exported completion has invalid routed-expert rows at positions "
            f"{missing[:16]}{'...' if len(missing) > 16 else ''}"
        )

    row_count = routes.shape[0]
    aligned[:row_count] = routes.reshape(
        row_count,
        config.num_layers,
        config.topk,
    ).to(dtype=torch.uint8)
    aligned_valid[:row_count] = valid
    # Even if an upstream backend ever includes a speculative final row, our
    # exact-row check above keeps terminal validity explicit and false.
    return aligned, aligned_valid


def get_train_data_for_step(
    step: dict,
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    filter_errors: bool = False,
    trajectory_reward: float = 0.0,
    router_replay_config: RouterReplayConfig | None = None,
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
    parts = _extract_completion_tokens(completions[completion_id], router_replay_config)
    if parts is None:
        logger.warning("Completion ID %s for task %s has no trainable tokens", completion_id, task_id)
        return None
    ob_tokens = parts.observation
    ac_tokens = parts.action
    ac_logprobs = parts.action_logprobs
    ac_versions = parts.action_versions

    seq = ob_tokens + ac_tokens
    logprobs = [0.0] * len(ob_tokens) + ac_logprobs
    loss_mask = [0] * len(ob_tokens) + [1] * len(ac_tokens)
    versions = [-1] * len(ob_tokens) + ac_versions
    attention_mask = torch.ones(len(seq), dtype=torch.bool).unsqueeze(0)
    num_input_tokens = torch.tensor(len(ob_tokens), dtype=torch.float32).unsqueeze(0)
    num_output_tokens = torch.tensor(len(ac_tokens), dtype=torch.float32).unsqueeze(0)

    result = dict(
        input_ids=torch.tensor(seq).unsqueeze(0),
        loss_mask=torch.tensor(loss_mask).unsqueeze(0),
        logprobs=torch.tensor(logprobs).unsqueeze(0),
        versions=torch.tensor(versions).unsqueeze(0),
        attention_mask=attention_mask,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
    )
    if router_replay_config is not None:
        assert parts.routed_experts is not None
        assert parts.routed_experts_valid is not None
        result["routed_experts"] = parts.routed_experts.unsqueeze(0)
        result["routed_experts_valid"] = parts.routed_experts_valid.unsqueeze(0)
    return result


def get_train_data_for_trajectory(
    trajectory: dict,
    completions: dict[str, CompletionWithResponse],
    task_id: str,
    trajectory_id: str,
    filter_errors: bool = False,
    reward_processor: Callable[[dict], tuple[float, dict]] = lambda traj: (traj["reward"], {}),
    merge_prefixes: bool = True,
    concat_fn: Callable[[list[dict]], dict] | None = None,
    router_replay_config: RouterReplayConfig | None = None,
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
            router_replay_config,
        )

    train_data = []
    accumulator = SequenceAccumulator(router_replay_config=router_replay_config)
    seen_completion_ids: set[str] = set()
    count_found = 0
    count_duplicates = 0
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
        # OpenHands may expose one parallel model response over multiple
        # environment steps as individual tool observations arrive. Every such
        # step carries the same completion ID, while the exported completion
        # record contains the *entire* model response. Training each occurrence
        # would therefore duplicate all loss-masked tokens from that response.
        if completion_id in seen_completion_ids:
            count_duplicates += 1
            continue
        if completion_id not in completions:
            logger.warning(f"Completion ID {completion_id} not found for task {task_id}")
            continue

        parts = _extract_completion_tokens(completions[completion_id], router_replay_config)
        if parts is None:
            logger.warning(
                "Completion ID %s for task %s has no trainable tokens; skipping step %s",
                completion_id,
                task_id,
                i,
            )
            continue
        # Mark the completion only after it has produced trainable data. This
        # preserves the existing fallback behavior if an earlier occurrence was
        # filtered or its exported record was temporarily unavailable.
        seen_completion_ids.add(completion_id)
        ob_tokens = parts.observation
        ac_tokens = parts.action
        ac_logprobs = parts.action_logprobs
        ac_versions = parts.action_versions
        count_found += 1

        # Track token counts (before merging) for overall trajectory stats
        total_input_tokens += len(ob_tokens)
        total_output_tokens += len(ac_tokens)

        # Determine if we can extend the current sequence or need to start fresh
        if len(accumulator.full_sequence) == 0:
            # First step - start new accumulator
            delta_ob_tokens = ob_tokens
            prefix_len = 0
        elif _is_prefix(accumulator.full_sequence, ob_tokens):
            # Observation extends the current sequence - we can merge!
            prefix_len = len(accumulator.full_sequence)
            delta_ob_tokens = ob_tokens[prefix_len:]
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
            prefix_len = 0

        accumulator.merge_completion_routes(
            parts.routed_experts,
            parts.routed_experts_valid,
            prefix_len,
        )

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
        f"Found {count_found} unique completions, skipped {count_duplicates} duplicate occurrences, "
        f"merged {num_merged}, produced {len(train_data)} datums"
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
    router_replay_config: RouterReplayConfig | None,
) -> dict | None:
    """Non-aggregated version for comparison/fallback."""
    train_data = []
    seen_completion_ids: set[str] = set()
    count_found_train_data = 0

    for i, step in enumerate(trajectory["steps"]):
        completion_id = step.get("misc", {}).get("action_misc", {}).get("completion_id")
        if completion_id is not None and completion_id in seen_completion_ids:
            continue

        step_train_data = get_train_data_for_step(
            step,
            completions,
            task_id,
            filter_errors,
            trajectory_reward,
            router_replay_config,
        )
        if step_train_data:
            # See the merged path above: one exported completion may back
            # several environment steps, but its model tokens must be trained
            # exactly once per trajectory.
            if completion_id is not None:
                seen_completion_ids.add(completion_id)
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
    router_replay_config: RouterReplayConfig | None = None,
    subagent_datum_sampler: SubagentDatumSampler | None = None,
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
        subagent_datum_sampler: Optional deterministic sampler.  It attaches a
            post-merge keep mask without removing data, allowing group reward
            centering and statistics to run over the complete rollout first.

    Returns:
        Training data dict or None if no valid data found.
    """
    if concat_fn is None:
        raise ValueError("concat_fn is required for get_train_data_for_trajectory_collection")

    trajectories = trajectory_collection["trajectories"]
    root_trajectory_id = next(iter(trajectories), None)
    depth_map = (
        _compute_trajectory_depths(trajectory_collection)
        if include_traj_depth or subagent_datum_sampler is not None
        else {}
    )

    train_data = []
    for trajectory_id, trajectory in trajectories.items():
        if _exclude_from_training(trajectory):
            continue
        trajectory_data = get_train_data_for_trajectory(
            trajectory,
            completions,
            task_id,
            trajectory_id,
            filter_errors,
            reward_processor,
            merge_prefixes,
            concat_fn,
            router_replay_config,
        )
        if trajectory_data is not None:
            num_datums = trajectory_data["rewards"].shape[0]
            depth = int(depth_map.get(trajectory_id, 0))
            policy_eligible = not trajectory_was_interrupted(trajectory) and (
                trajectory_id == root_trajectory_id or not _exclude_from_policy_training(trajectory)
            )
            trajectory_data[POLICY_TRAINING_ELIGIBILITY_MASK_KEY] = torch.full(
                (num_datums,),
                policy_eligible,
                dtype=torch.bool,
            )
            sampling_mask: torch.Tensor | None = None
            if subagent_datum_sampler is not None:
                if policy_eligible:
                    sampled = subagent_datum_sampler.sample_mask(
                        task_id=task_id,
                        trajectory_id=trajectory_id,
                        depth=depth,
                        num_datums=num_datums,
                    )
                else:
                    # Policy-ineligible verifier children do not participate in
                    # the Bernoulli population and must not consume a draw.
                    sampled = [True] * num_datums
                sampling_mask = torch.tensor(sampled, dtype=torch.bool)
                if sampling_mask.shape != (num_datums,):
                    raise ValueError(
                        "Subagent datum sampler returned the wrong mask shape: "
                        f"expected {(num_datums,)}, got {tuple(sampling_mask.shape)}"
                    )
                trajectory_data[SUBAGENT_DATUM_KEEP_MASK_KEY] = sampling_mask
                trajectory_data[SUBAGENT_DATUM_DEPTH_KEY] = torch.full(
                    (num_datums,), depth, dtype=torch.long
                )
            if include_traj_depth and trajectory_id in depth_map:
                trajectory_data["traj_depth"] = torch.full(
                    (num_datums,), depth, dtype=torch.long
                )
                if include_traj_start:
                    # Mark the first datum of this trajectory so we can count
                    # distinct retained trajectories (not datums) at each depth
                    # level. A trajectory with no retained datum has no start.
                    traj_start = torch.zeros(num_datums, dtype=torch.float32)
                    if sampling_mask is None:
                        traj_start[0] = 1.0
                    else:
                        retained = torch.nonzero(sampling_mask, as_tuple=False).reshape(-1)
                        if retained.numel() > 0:
                            traj_start[int(retained[0].item())] = 1.0
                    trajectory_data["traj_start"] = traj_start
            train_data.append(trajectory_data)

    if not train_data:
        logger.debug(f"No train data found for any trajectory for task {task_id}")
        return None

    train_data = harmonize_optional_reward_metrics(train_data)
    root_trajectory = next(iter(trajectories.values()))
    root_reward, root_rewards_dict = reward_processor(root_trajectory)

    return concat_fn(train_data) | {
        "task_reward": torch.tensor(root_reward).unsqueeze(0),
        "task_reward_valid": torch.tensor(
            [not trajectory_was_interrupted(root_trajectory)],
            dtype=torch.bool,
        ),
        **{f"root_{key}": torch.tensor(value).unsqueeze(0) for key, value in root_rewards_dict.items()},
    }
