"""Group-wise AReaL rollout workflow for Platoon training."""

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

import torch
from areal.api import InferenceEngine, RolloutWorkflow
from areal.infra import workflow_context
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string

from platoon.envs.base import Task
from platoon.train.areal.config_defs import WorkflowConfig
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.workflow_serialization import RemoteWorkflowSerializable, callable_import_path
from platoon.utils.areal_data_processing import (
    POLICY_TRAINING_ELIGIBILITY_MASK_KEY,
    SUBAGENT_DATUM_DEPTH_KEY,
    SUBAGENT_DATUM_KEEP_MASK_KEY,
    RouterReplayConfig,
    get_train_data_for_trajectory_collection,
    harmonize_optional_reward_metrics,
    reward_metric_presence_key,
)
from platoon.utils.rollout_workload import (
    RolloutWorkload,
    record_workload_distribution,
    sum_rollout_workloads,
    trajectory_collection_shape,
)
from platoon.utils.subagent_sampling import DeterministicSubagentDatumSampler
from platoon.utils.token_efficiency import (
    annotate_policy_subtree_token_efficiency,
)
from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_SUBPROCESS_INIT_BUDGET_SECONDS = 120
_SUBPROCESS_CLEANUP_GRACE_SECONDS = 60
_PARENT_FUTURE_GRACE_SECONDS = 30
_PROXY_SESSION_CLOSE_TIMEOUT_SECONDS = 30
_ROLLOUT_TASK_CANCEL_GRACE_SECONDS = 5


@dataclass(frozen=True)
class _SubprocessRolloutOutcome:
    result: dict | None
    elapsed_seconds: float
    force_pool_shutdown: bool = False


@dataclass(frozen=True)
class _ProcessedRolloutResult:
    """Training result plus inference telemetry for one requested rollout.

    Keeping workload separate from ``train_data`` is important: a rollout can
    consume substantial generation work and still yield no policy datum (for
    example, an interrupted root with only an unparseable verifier response).
    """

    train_data: dict | None
    workload: RolloutWorkload
    observed: bool


_WORKLOAD_SIDECAR_PREFIX = "_platoon_workload_"
_WORKLOAD_SIDECAR_FIELDS = {
    "environment_steps": f"{_WORKLOAD_SIDECAR_PREFIX}environment_steps",
    "model_calls": f"{_WORKLOAD_SIDECAR_PREFIX}model_calls",
    "input_tokens": f"{_WORKLOAD_SIDECAR_PREFIX}input_tokens",
    "output_tokens": f"{_WORKLOAD_SIDECAR_PREFIX}output_tokens",
    "trajectories": f"{_WORKLOAD_SIDECAR_PREFIX}trajectories",
}
_WORKLOAD_REQUESTED_ROLLOUTS_KEY = f"{_WORKLOAD_SIDECAR_PREFIX}requested_rollouts"
_WORKLOAD_OBSERVED_ROLLOUTS_KEY = f"{_WORKLOAD_SIDECAR_PREFIX}observed_rollouts"
_WORKLOAD_TRAINABLE_ROLLOUTS_KEY = f"{_WORKLOAD_SIDECAR_PREFIX}trainable_rollouts"
_WORKLOAD_DATUM_SIDECAR_FIELDS = {
    "postmerge_datums": f"{_WORKLOAD_SIDECAR_PREFIX}postmerge_datums",
    "policy_eligible_datums": f"{_WORKLOAD_SIDECAR_PREFIX}policy_eligible_datums",
    "post_sampling_datums": f"{_WORKLOAD_SIDECAR_PREFIX}post_sampling_datums",
}
_WORKLOAD_TASK_RETAINED_DATUMS_KEY = f"{_WORKLOAD_SIDECAR_PREFIX}task_retained_datums"


def _interaction_token_counts(interaction: Any) -> tuple[int, int]:
    """Return logical prompt/output tokens for one exported interaction."""

    tensor_dict = interaction.to_tensor_dict()
    input_ids = tensor_dict["input_ids"]
    loss_mask = tensor_dict["loss_mask"]
    if not torch.is_tensor(input_ids) or not torch.is_tensor(loss_mask):
        raise TypeError("exported interaction input_ids/loss_mask must be tensors")
    if input_ids.shape != loss_mask.shape:
        raise ValueError(
            "exported interaction token/mask shape mismatch: "
            f"input_ids={tuple(input_ids.shape)}, loss_mask={tuple(loss_mask.shape)}"
        )
    attention_mask = tensor_dict.get("attention_mask")
    if attention_mask is None:
        valid_tokens = torch.ones_like(loss_mask, dtype=torch.bool)
    elif not torch.is_tensor(attention_mask):
        raise TypeError("exported interaction attention_mask must be a tensor")
    elif attention_mask.shape != loss_mask.shape:
        raise ValueError(
            "exported interaction attention/loss-mask shape mismatch: "
            f"attention_mask={tuple(attention_mask.shape)}, loss_mask={tuple(loss_mask.shape)}"
        )
    else:
        valid_tokens = attention_mask.bool()
    output_tokens = int((loss_mask.bool() & valid_tokens).sum().item())
    total_tokens = int(valid_tokens.sum().item())
    return total_tokens - output_tokens, output_tokens


def _completion_token_counts(completions: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Measure every valid exported request exactly once."""

    counts: dict[str, tuple[int, int]] = {}
    for completion_id, completion in completions.items():
        try:
            counts[completion_id] = _interaction_token_counts(completion)
        except Exception:
            logger.warning(
                "Unable to account tokens for exported interaction %s; training-data processing remains independent",
                completion_id,
                exc_info=True,
            )
    return counts


def _rollout_workload(
    trajectory_data: dict | None,
    completions: dict[str, Any],
    completion_token_counts: dict[str, tuple[int, int]] | None = None,
) -> RolloutWorkload:
    """Account for raw recursive work without affecting training eligibility.

    ``style="individual"`` export gives one record per actual model request.
    Counting that mapping directly avoids both duplicate step references and
    prefix-merging effects.  A malformed telemetry record is logged and omitted
    from token totals, but never prevents otherwise valid train data from being
    constructed by the existing processing path.
    """

    trajectory_count, environment_steps = trajectory_collection_shape(trajectory_data)
    if completion_token_counts is None:
        completion_token_counts = _completion_token_counts(completions)
    input_tokens = sum(counts[0] for counts in completion_token_counts.values())
    output_tokens = sum(counts[1] for counts in completion_token_counts.values())
    return RolloutWorkload(
        environment_steps=environment_steps,
        model_calls=len(completions),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trajectories=trajectory_count,
    )


def _rollout_datum_funnel(train_data: dict | None) -> tuple[int, int, int]:
    """Return post-merge, policy-eligible, and post-sampling datum counts."""

    if train_data is None:
        return 0, 0, 0
    rewards = train_data["rewards"].reshape(-1)
    postmerge = int(rewards.numel())
    policy_mask = train_data.get(POLICY_TRAINING_ELIGIBILITY_MASK_KEY)
    if policy_mask is None:
        policy_mask = torch.ones(postmerge, dtype=torch.bool, device=rewards.device)
    else:
        policy_mask = policy_mask.detach().bool().reshape(-1)
    keep_mask = train_data.get(SUBAGENT_DATUM_KEEP_MASK_KEY)
    if keep_mask is None:
        keep_mask = torch.ones(postmerge, dtype=torch.bool, device=policy_mask.device)
    else:
        keep_mask = keep_mask.detach().bool().reshape(-1).to(policy_mask.device)
    if policy_mask.numel() != postmerge or keep_mask.numel() != postmerge:
        raise ValueError(
            "Datum funnel masks do not match post-merge rewards: "
            f"postmerge={postmerge}, policy={policy_mask.numel()}, sampling={keep_mask.numel()}"
        )
    policy_eligible = int(policy_mask.sum().item())
    post_sampling = int((policy_mask & keep_mask).sum().item())
    return postmerge, policy_eligible, post_sampling


def _measure_zero_centered_reward_candidates(train_data: dict) -> dict[str, float]:
    """Measure policy candidates whose group-centered scalar reward is zero.

    Physical filtering happens later, after global DP selection and depth
    weighting. Keeping candidates here preserves full-batch normalization and
    lets the trainer retain the minimum zero padding required for dispatch.
    """

    rewards = train_data["rewards"].detach().reshape(-1)
    num_datums = int(rewards.numel())
    existing = train_data.get("trainable_datums")
    if existing is None:
        existing = torch.ones(num_datums, dtype=torch.bool, device=rewards.device)
    else:
        existing = existing.detach().bool().reshape(-1).to(rewards.device)
    if existing.numel() != num_datums:
        raise ValueError("trainable_datums does not match centered rewards")

    loss_mask = train_data.get("loss_mask")
    if not torch.is_tensor(loss_mask) or loss_mask.ndim < 1 or loss_mask.shape[0] != num_datums:
        raise ValueError("Zero-reward filtering requires per-datum loss_mask")
    per_datum_action_tokens = loss_mask.detach().bool().reshape(num_datums, -1).sum(dim=1)
    attention_mask = train_data.get("attention_mask")
    if not torch.is_tensor(attention_mask) or attention_mask.ndim < 1 or attention_mask.shape[0] != num_datums:
        raise ValueError("Zero-reward filtering requires per-datum attention_mask")
    per_datum_attention_tokens = attention_mask.detach().bool().reshape(num_datums, -1).sum(dim=1)

    # Exact comparison preserves arbitrarily small real signal and keeps NaN/
    # Inf visible to downstream failure detection.
    zero_reward = existing & (rewards == 0)
    return {
        "workflow_zero_reward_candidate_population_datums": float(existing.sum().item()),
        "workflow_zero_reward_candidate_population_action_tokens": float(
            per_datum_action_tokens[existing.to(per_datum_action_tokens.device)].sum().item()
        ),
        "workflow_zero_reward_candidate_population_attention_tokens": float(
            per_datum_attention_tokens[existing.to(per_datum_attention_tokens.device)].sum().item()
        ),
        "workflow_zero_reward_candidate_datums": float(zero_reward.sum().item()),
        "workflow_zero_reward_candidate_action_tokens": float(
            per_datum_action_tokens[zero_reward.to(per_datum_action_tokens.device)].sum().item()
        ),
        "workflow_zero_reward_candidate_attention_tokens": float(
            per_datum_attention_tokens[zero_reward.to(per_datum_attention_tokens.device)].sum().item()
        ),
    }


def _filter_positive_centered_error_tokens(train_data: dict) -> dict[str, float]:
    """Mask erroneous action tokens only when their centered reward is positive.

    The token-aligned side channel is produced before group rewards exist.  It
    is always consumed here, before actor dispatch, and never becomes a model
    input.  A merged datum can therefore retain clean/negative-signal actions
    while suppressing only the positively reinforced erroneous completion.
    """

    error_mask = train_data.pop(ERROR_ACTION_MASK_KEY, None)
    if error_mask is None:
        return {}
    loss_mask = train_data.get("loss_mask")
    rewards = train_data.get("rewards")
    if not torch.is_tensor(error_mask) or not torch.is_tensor(loss_mask) or not torch.is_tensor(rewards):
        raise TypeError("Deferred error filtering requires tensor error, loss, and reward fields")
    if error_mask.shape != loss_mask.shape:
        raise ValueError(
            "Deferred error/loss masks must be token-aligned: "
            f"error={tuple(error_mask.shape)}, loss={tuple(loss_mask.shape)}"
        )
    batch_size = int(loss_mask.shape[0])
    centered_rewards = rewards.reshape(-1)
    if centered_rewards.numel() != batch_size:
        raise ValueError(
            "Deferred error filtering requires one centered reward per datum: "
            f"rewards={centered_rewards.numel()}, datums={batch_size}"
        )

    action_mask = loss_mask.bool()
    error_actions = error_mask.bool() & action_mask
    positive = centered_rewards > 0
    positive_shape = (batch_size,) + (1,) * (loss_mask.ndim - 1)
    suppressed = error_actions & positive.reshape(positive_shape).to(error_actions.device)
    train_data["loss_mask"] = torch.where(suppressed, torch.zeros_like(loss_mask), loss_mask)

    has_trainable_tokens = train_data["loss_mask"].bool().reshape(batch_size, -1).any(dim=1)
    existing = train_data.get("trainable_datums")
    if existing is not None:
        existing = existing.detach().bool().reshape(-1).to(has_trainable_tokens.device)
        if existing.numel() != batch_size:
            raise ValueError("trainable_datums does not match deferred error mask")
        train_data["trainable_datums"] = existing & has_trainable_tokens
    elif not bool(has_trainable_tokens.all()):
        train_data["trainable_datums"] = has_trainable_tokens

    emptied = action_mask.reshape(batch_size, -1).any(dim=1) & ~has_trainable_tokens
    return {
        "error_filter/detected_action_tokens": float(error_actions.sum().item()),
        "error_filter/suppressed_positive_action_tokens": float(suppressed.sum().item()),
        "error_filter/retained_nonpositive_action_tokens": float(
            (error_actions & ~suppressed).sum().item()
        ),
        "error_filter/emptied_datums": float(emptied.sum().item()),
    }


class GroupRolloutWorkflow(RolloutWorkflow, RemoteWorkflowSerializable):
    """Workflow that preserves Platoon's recursive group-rollout processing."""

    def __init__(
        self,
        rollout_fn: Callable[[Task, dict], dict] | str,
        get_task_fn: Callable[[str], Task] | str,
        config: WorkflowConfig | dict[str, Any],
        proxy_base_url: str | None,
        proxy_admin_api_key: str,
        output_subdir: str = "rollout",
        filter_errors: bool = False,
        reward_processor: Callable[[dict], tuple[float, dict]] | str = lambda traj: (traj["reward"], {}),
        merge_prefixes: bool = True,
    ):
        if isinstance(config, dict):
            config = WorkflowConfig(**config)
        self.config = deepcopy(config)
        self.config.rollout_config.return_dict = True
        self.config.rollout_config.train = True
        self.proxy_base_url = proxy_base_url
        self.proxy_admin_api_key = proxy_admin_api_key
        self.rollout_fn = self._resolve_callable(rollout_fn)
        self.get_task_fn = self._resolve_callable(get_task_fn)
        self.filter_errors = filter_errors
        self.reward_processor = self._resolve_reward_processor(reward_processor)
        self.merge_prefixes = merge_prefixes
        self.output_subdir = output_subdir
        self.router_replay_config = self._build_router_replay_config(self.config)
        self.subagent_datum_sampler = (
            DeterministicSubagentDatumSampler(
                keep_probability=self.config.subagent_datum_keep_probability,
                seed=self.config.subagent_datum_sampling_seed,
            )
            if self.config.subagent_datum_keep_probability < 1.0
            else None
        )

    @staticmethod
    def _build_router_replay_config(config: WorkflowConfig) -> RouterReplayConfig | None:
        if not config.enable_router_replay:
            return None
        if config.router_replay_num_layers is None or config.router_replay_topk is None:
            raise ValueError(
                "Router replay requires model-derived workflow dimensions: "
                "router_replay_num_layers and router_replay_topk"
            )
        return RouterReplayConfig(
            num_layers=config.router_replay_num_layers,
            topk=config.router_replay_topk,
        )

    @staticmethod
    def _resolve_callable(fn: Callable | str) -> Callable:
        if isinstance(fn, str):
            return import_from_string(fn)
        return fn

    @staticmethod
    def _resolve_reward_processor(
        reward_processor: Callable[[dict], tuple[float, dict]] | str,
    ) -> Callable[[dict], tuple[float, dict]]:
        if isinstance(reward_processor, str):
            return import_from_string(reward_processor)
        return reward_processor

    def to_workflow_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "rollout_fn": callable_import_path(self.rollout_fn),
            "get_task_fn": callable_import_path(self.get_task_fn),
            "config": asdict(self.config),
            "proxy_base_url": None,
            "proxy_admin_api_key": self.proxy_admin_api_key,
            "output_subdir": self.output_subdir,
            "filter_errors": self.filter_errors,
            "merge_prefixes": self.merge_prefixes,
        }
        reward_processor_path = callable_import_path(self.reward_processor)
        if kwargs["rollout_fn"] is None or kwargs["get_task_fn"] is None:
            raise ValueError("GroupRolloutWorkflow requires importable rollout_fn/get_task_fn")
        if reward_processor_path is not None:
            kwargs["reward_processor"] = reward_processor_path
        return kwargs

    def to_remote_workflow(self) -> tuple[type["GroupRolloutWorkflow"], dict[str, Any]]:
        """Describe how this workflow should be reconstructed on workers."""
        return self.__class__, self.to_workflow_kwargs()

    def set_proxy_base_url(self, proxy_base_url: str) -> None:
        """Bind the worker-local proxy URL before rollout execution."""
        self.proxy_base_url = proxy_base_url

    def _require_proxy_base_url(self) -> str:
        if not self.proxy_base_url:
            raise RuntimeError(
                "GroupRolloutWorkflow.proxy_base_url is not set. "
                "Expected AReaL worker dispatch to inject a proxy endpoint."
            )
        return self.proxy_base_url

    def _session_task_id(self, task_id: str, rollout_number: int) -> str:
        return f"{task_id}-rollout-{rollout_number}-{uuid.uuid4().hex[:8]}"

    def _build_rollout_config(self, engine: InferenceEngine, session: ArealProxySession) -> WorkflowConfig:
        config = deepcopy(self.config)
        config.rollout_config.output_dir = os.path.join(
            config.rollout_config.output_dir,
            self.output_subdir,
        )
        config.rollout_config.model_endpoint = self._require_proxy_base_url()
        model_name = config.rollout_config.model_name or ""
        if not model_name.startswith("openai/"):
            config.rollout_config.model_name = f"openai/{model_name}"
        config.rollout_config.model_api_key = session.session_api_key
        config.rollout_config.output_dir = os.path.join(
            config.rollout_config.output_dir,
            str(engine.get_version()),
        )
        return config

    def _record_stats(self, train_data: dict) -> None:
        tracker = stats_tracker.get(workflow_context.stat_scope())

        def scalar_series(name: str, values: torch.Tensor) -> None:
            for value in values.detach().float().reshape(-1):
                tracker.scalar(**{name: value.item()})

        def scalar_value(name: str, value: torch.Tensor) -> None:
            tracker.scalar(**{name: value.detach().float().item()})

        def present_values(name: str, values: torch.Tensor) -> torch.Tensor:
            values = values.detach().float().reshape(-1)
            mask = train_data.get(reward_metric_presence_key(name))
            if mask is None:
                return values
            mask = mask.detach().bool().reshape(-1)
            if mask.shape != values.shape:
                raise ValueError(
                    f"Reward metric presence mask shape mismatch for {name}: "
                    f"values={tuple(values.shape)}, mask={tuple(mask.shape)}"
                )
            return values[mask]

        num_steps = train_data["num_steps"]
        num_input_tokens = train_data["num_input_tokens"]
        num_output_tokens = train_data["num_output_tokens"]
        safe_num_steps = torch.clamp(num_steps, min=1.0)
        avg_input_tokens_per_step = num_input_tokens / safe_num_steps
        avg_output_tokens_per_step = num_output_tokens / safe_num_steps

        scalar_series("task_reward", train_data["task_reward"])
        scalar_series("num_output_tokens", num_output_tokens)
        scalar_series("num_input_tokens", num_input_tokens)
        scalar_series("num_steps", num_steps)
        scalar_series("avg_input_tokens_per_step", avg_input_tokens_per_step)
        scalar_series("avg_output_tokens_per_step", avg_output_tokens_per_step)

        task_rewards = train_data["task_reward"]
        scalar_value("task_reward_at_k_mean", torch.mean(task_rewards))
        scalar_value("task_reward_at_k_max", torch.max(task_rewards))
        scalar_value("task_reward_at_k_min", torch.min(task_rewards))

        for key, value in train_data.items():
            if key.startswith("root_"):
                values = present_values(key, value)
                if values.numel() == 0:
                    continue
                scalar_series(key, values)
                scalar_value(f"{key}_at_k_mean", torch.mean(values))
                scalar_value(f"{key}_at_k_max", torch.max(values))
                scalar_value(f"{key}_at_k_min", torch.min(values))
            elif key.startswith("reward/"):
                scalar_series(key, present_values(key, value))

    def _activate_subagent_datum_sampling(self, train_data: dict) -> None:
        """Expose post-metric policy-eligibility and sampling masks.

        All rollout tensors remain present here so leave-one-out reward math and
        workflow statistics observe the complete group. Policy-ineligible
        children are then intersected with the Bernoulli mask and any existing
        ``trainable_datums`` before trainer-side depth weighting or R3 dispatch.
        """

        policy_eligible = train_data.pop(POLICY_TRAINING_ELIGIBILITY_MASK_KEY, None)
        keep_mask = train_data.pop(SUBAGENT_DATUM_KEEP_MASK_KEY, None)
        depth = train_data.pop(SUBAGENT_DATUM_DEPTH_KEY, None)
        if policy_eligible is None and keep_mask is None:
            return
        if keep_mask is not None and depth is None:
            raise ValueError("Subagent datum sampling mask is missing depth metadata")

        num_datums = int(train_data["rewards"].shape[0])
        if policy_eligible is None:
            policy_eligible = torch.ones(num_datums, dtype=torch.bool)
        else:
            policy_eligible = policy_eligible.detach().bool().reshape(-1)
        if keep_mask is None:
            keep_mask = torch.ones(num_datums, dtype=torch.bool, device=policy_eligible.device)
        else:
            keep_mask = keep_mask.detach().bool().reshape(-1)
        if policy_eligible.numel() != num_datums or keep_mask.numel() != num_datums:
            raise ValueError(
                "Policy/sampling metadata does not match reward batch: "
                f"policy={policy_eligible.numel()}, keep={keep_mask.numel()}, rewards={num_datums}"
            )

        # Sampling telemetry is emitted only when Bernoulli sampling is active.
        # Policy-excluded verifier children are deliberately outside both the
        # eligible and retained populations.
        if depth is not None:
            depth = depth.detach().long().reshape(-1)
            if depth.numel() != num_datums:
                raise ValueError("Subagent sampling depth metadata does not match reward batch")

            def per_datum_tokens(key: str) -> torch.Tensor:
                value = train_data.get(key)
                if not torch.is_tensor(value) or value.ndim < 1 or value.shape[0] != num_datums:
                    raise ValueError(f"Subagent datum sampling requires per-datum {key}")
                return value.detach().reshape(num_datums, -1).sum(dim=1).float()

            attention_tokens = per_datum_tokens("attention_mask")
            loss_tokens = per_datum_tokens("loss_mask")
            tracker = stats_tracker.get(workflow_context.stat_scope())

            def record(prefix: str, eligible: torch.Tensor) -> None:
                token_eligible = eligible.to(attention_tokens.device)
                retained = token_eligible & keep_mask.to(attention_tokens.device)
                tracker.scalar(
                    **{
                        f"{prefix}eligible_datums": float(token_eligible.sum().item()),
                        f"{prefix}retained_datums": float(retained.sum().item()),
                        f"{prefix}eligible_attention_tokens": float(attention_tokens[token_eligible].sum().item()),
                        f"{prefix}retained_attention_tokens": float(attention_tokens[retained].sum().item()),
                        f"{prefix}eligible_loss_tokens": float(loss_tokens[token_eligible].sum().item()),
                        f"{prefix}retained_loss_tokens": float(loss_tokens[retained].sum().item()),
                    }
                )

            policy_for_depth = policy_eligible.to(depth.device)
            record("subagent_sampling/", policy_for_depth)
            for depth_value in torch.unique(depth, sorted=True).tolist():
                record(
                    f"subagent_sampling/depth_{int(depth_value)}/",
                    policy_for_depth & (depth == int(depth_value)),
                )

        existing_present = "trainable_datums" in train_data
        existing = train_data.get("trainable_datums")
        if existing is None:
            existing = torch.ones_like(keep_mask, dtype=torch.bool)
        else:
            existing = existing.detach().bool().reshape(-1)
            if existing.numel() != num_datums:
                raise ValueError("trainable_datums does not match subagent sampling mask")
        combined = policy_eligible.to(existing.device) & keep_mask.to(existing.device)
        # Keep the historical p=1/no-policy-exclusion path structurally exact.
        if existing_present or not bool(combined.all()):
            train_data["trainable_datums"] = existing & combined

        # Keep depth/start metadata until trainer-side filtering and divisibility
        # trimming.  The trainer uses it to preserve roots and repair a start
        # marker for every surviving trajectory, then removes it before actor
        # dispatch when no depth transform consumes it.

    @staticmethod
    def _record_workload_stats(
        processed_results: list[_ProcessedRolloutResult],
        *,
        retained_datums_per_rollout: list[int],
    ) -> None:
        """Record rollout and task distributions before policy-data filtering."""

        if len(retained_datums_per_rollout) != len(processed_results):
            raise ValueError("Retained-datum counts must align with requested rollouts")
        tracker = stats_tracker.get(workflow_context.stat_scope())
        rollout_workloads = [result.workload for result in processed_results]
        task_workload = sum_rollout_workloads(rollout_workloads)
        task_retained_count = sum(retained_datums_per_rollout)
        if not 0 <= task_retained_count <= task_workload.post_sampling_datums:
            raise ValueError(
                "Invalid task datum funnel: "
                f"postmerge={task_workload.postmerge_datums}, "
                f"policy_eligible={task_workload.policy_eligible_datums}, "
                f"post_sampling={task_workload.post_sampling_datums}, "
                f"task_retained={task_retained_count}"
            )
        record_workload_distribution(
            tracker,
            prefix="workload/rollout",
            workloads=rollout_workloads,
        )
        record_workload_distribution(
            tracker,
            prefix="workload/task",
            workloads=[task_workload],
        )
        task_retained = float(task_retained_count)
        trainable_rollouts = sum(count > 0 for count in retained_datums_per_rollout)
        # Reuse the one task denominator registered immediately above; adding
        # it twice would silently halve these distribution averages.
        tracker.stat(
            denominator="workload/task/count",
            **{
                "workload/task/requested_rollouts": torch.tensor(
                    [float(len(processed_results))], dtype=torch.float32
                ),
                "workload/task/observed_rollouts": torch.tensor(
                    [float(sum(result.observed for result in processed_results))],
                    dtype=torch.float32,
                ),
                "workload/task/trainable_rollouts": torch.tensor(
                    [float(trainable_rollouts)],
                    dtype=torch.float32,
                ),
                "workload/task/total_task_retained_datums": torch.tensor(
                    [task_retained], dtype=torch.float32
                ),
                "workload/task/total_task_workflow_trainable_datums": torch.tensor(
                    [task_retained], dtype=torch.float32
                ),
                "workload/task/total_task_workflow_non_trainable_datums": torch.tensor(
                    [float(task_workload.postmerge_datums) - task_retained],
                    dtype=torch.float32,
                ),
            },
        )

    @staticmethod
    def _attach_task_workload_sidecar(
        train_data: dict,
        processed_results: list[_ProcessedRolloutResult],
        *,
        retained_datums_per_rollout: list[int] | None = None,
    ) -> None:
        """Carry exact accepted-task totals to the controller-side trainer."""

        workload = sum_rollout_workloads(result.workload for result in processed_results)
        for field, key in _WORKLOAD_SIDECAR_FIELDS.items():
            train_data[key] = torch.tensor([getattr(workload, field)], dtype=torch.int64)
        train_data[_WORKLOAD_REQUESTED_ROLLOUTS_KEY] = torch.tensor(
            [len(processed_results)], dtype=torch.int64
        )
        train_data[_WORKLOAD_OBSERVED_ROLLOUTS_KEY] = torch.tensor(
            [sum(result.observed for result in processed_results)], dtype=torch.int64
        )
        if retained_datums_per_rollout is None:
            retained_datums_per_rollout = GroupRolloutWorkflow._retained_datums_per_rollout(
                train_data,
                processed_results,
            )
        for field, key in _WORKLOAD_DATUM_SIDECAR_FIELDS.items():
            train_data[key] = torch.tensor(
                [getattr(workload, field)],
                dtype=torch.int64,
            )
        train_data[_WORKLOAD_TASK_RETAINED_DATUMS_KEY] = torch.tensor(
            [sum(retained_datums_per_rollout)], dtype=torch.int64
        )
        train_data[_WORKLOAD_TRAINABLE_ROLLOUTS_KEY] = torch.tensor(
            [sum(count > 0 for count in retained_datums_per_rollout)],
            dtype=torch.int64,
        )

    @staticmethod
    def _retained_datums_per_rollout(
        train_data: dict,
        processed_results: list[_ProcessedRolloutResult],
    ) -> list[int]:
        """Return final retained policy datum counts for every requested rollout."""

        rewards = train_data["rewards"].reshape(-1)
        trainable = train_data.get("trainable_datums")
        if trainable is None:
            trainable = torch.ones_like(rewards, dtype=torch.bool)
        else:
            trainable = trainable.detach().bool().reshape(-1)
        if trainable.numel() != rewards.numel():
            raise ValueError("Final trainable mask does not match the task reward batch")

        offset = 0
        counts: list[int] = []
        for result in processed_results:
            if result.train_data is None:
                counts.append(0)
                continue
            datum_count = int(result.train_data["rewards"].shape[0])
            counts.append(int(trainable[offset : offset + datum_count].sum().item()))
            offset += datum_count
        if offset != trainable.numel():
            raise ValueError(
                "Processed rollout datum counts do not match final task batch: "
                f"processed={offset}, final={trainable.numel()}"
            )
        return counts

    async def arun_episode(self, engine: InferenceEngine, data: dict) -> dict | None:
        tracker = stats_tracker.get(workflow_context.stat_scope())
        tracker.scalar(group_size_requested=float(self.config.group_size))
        if self.config.use_subprocesses:
            raw_processed_results = await self._arun_episode_with_subprocesses(engine, data)
        else:
            raw_processed_results = await asyncio.gather(
                *[self._arun_episode_single(engine, data, i) for i in range(self.config.group_size)]
            )

        # Tests and external subclasses written before workload telemetry may
        # still return the historical dict/None shape. Preserve that API while
        # using the richer side channel for all native execution paths.
        has_workload_sidechannel = any(
            isinstance(result, _ProcessedRolloutResult) for result in raw_processed_results
        )
        processed_results = [
            result
            if isinstance(result, _ProcessedRolloutResult)
            else _ProcessedRolloutResult(
                train_data=result,
                workload=RolloutWorkload(),
                observed=result is not None,
            )
            for result in raw_processed_results
        ]
        def record_workload_stats(final_train_data: dict | None) -> list[int]:
            if final_train_data is None:
                retained_datums_per_rollout = [0] * len(processed_results)
            else:
                retained_datums_per_rollout = self._retained_datums_per_rollout(
                    final_train_data,
                    processed_results,
                )
            if has_workload_sidechannel:
                self._record_workload_stats(
                    processed_results,
                    retained_datums_per_rollout=retained_datums_per_rollout,
                )
            return retained_datums_per_rollout

        results = [result.train_data for result in processed_results if result.train_data is not None]
        tracker.scalar(group_size_effective=float(len(results)))
        if len(results) < self.config.min_successful_group_size:
            logger.warning(
                "Rejecting task %s group with %s returned members; minimum is %s",
                data["task_id"],
                len(results),
                self.config.min_successful_group_size,
            )
            tracker.scalar(group_size_rejected=1.0)
            record_workload_stats(None)
            return None
        if not results:
            logger.warning("No rollout results found for task %s", data["task_id"])
            record_workload_stats(None)
            return None

        results = harmonize_optional_reward_metrics(results)
        train_data = concat_padded_tensors(results)
        mean_unprocessed_reward = torch.mean(train_data["rewards"])

        task_rewards = train_data["task_reward"]
        task_reward_valid = train_data.get("task_reward_valid")
        if task_reward_valid is None:
            # Backward compatibility for previously serialized/custom workflow
            # results which predate explicit root-validity metadata.
            valid_roots = torch.ones_like(task_rewards, dtype=torch.bool)
        else:
            valid_roots = task_reward_valid.detach().bool().reshape(-1).to(task_rewards.device)
            if valid_roots.numel() != task_rewards.numel():
                raise ValueError(
                    "task_reward_valid does not match task_reward: "
                    f"valid={valid_roots.numel()}, rewards={task_rewards.numel()}"
                )

        completed_root_count = int(valid_roots.sum().item())
        tracker.scalar(group_size_completed_roots=float(completed_root_count))
        if not bool(valid_roots.any()):
            # Preserve complete rollout/reward telemetry, but never construct a
            # policy target from a group whose every root reward is partial.
            self._record_stats(train_data)
            tracker.scalar(no_valid_root_reward_group=1.0)
            logger.warning("Rejecting task %s group with no valid root rewards", data["task_id"])
            record_workload_stats(None)
            return None
        if completed_root_count < self.config.min_successful_group_size:
            # Partial members may still contain useful completed descendants,
            # but they cannot satisfy the configured baseline quorum.  Keep
            # their rollout telemetry while refusing a statistically weaker
            # group than the run requested.
            self._record_stats(train_data)
            tracker.scalar(group_completed_root_quorum_rejected=1.0)
            logger.warning(
                "Rejecting task %s group with %s completed roots; minimum is %s",
                data["task_id"],
                completed_root_count,
                self.config.min_successful_group_size,
            )
            record_workload_stats(None)
            return None

        if bool(valid_roots.all()):
            # Preserve the historical all-valid arithmetic bit-for-bit.
            if self.config.leave_one_out_baseline and len(results) > 1:
                total_reward = task_rewards.sum()
                loo_baselines = (total_reward - task_rewards) / (len(task_rewards) - 1)
                datum_counts = torch.tensor([r["rewards"].shape[0] for r in results])
                per_datum_baselines = torch.repeat_interleave(loo_baselines, datum_counts)
                train_data["rewards"] = train_data["rewards"] - per_datum_baselines
            else:
                train_data["rewards"] = train_data["rewards"] - torch.mean(task_rewards)
        elif self.config.leave_one_out_baseline:
            valid_rewards = task_rewards[valid_roots]
            valid_total = valid_rewards.sum()
            valid_count = int(valid_rewards.numel())
            member_baselines = torch.ones_like(task_rewards) * valid_rewards.mean()
            if valid_count > 1:
                member_baselines[valid_roots] = (valid_total - task_rewards[valid_roots]) / (valid_count - 1)
            else:
                # The sole valid member cannot leave itself out; subtracting its
                # own valid reward is the only non-contaminating fallback.
                member_baselines[valid_roots] = task_rewards[valid_roots]
            datum_counts = torch.tensor([r["rewards"].shape[0] for r in results])
            train_data["rewards"] = train_data["rewards"] - torch.repeat_interleave(
                member_baselines,
                datum_counts,
            )
        else:
            valid_mean = task_rewards[valid_roots].mean()
            train_data["rewards"] = train_data["rewards"] - valid_mean

        self._record_stats(train_data)
        self._activate_subagent_datum_sampling(train_data)
        error_filter_metrics = _filter_positive_centered_error_tokens(train_data)
        if error_filter_metrics:
            tracker.scalar(**error_filter_metrics)

        # Trainer-side full-batch transforms may still need batch metadata like
        # traj_depth, so the workflow only signals which datums are trainable.
        if not self.config.filter_zero_variance_groups and "trainable_datums" not in train_data:
            train_data["trainable_datums"] = torch.ones_like(train_data["rewards"], dtype=torch.bool)

        final_trainable = train_data.get("trainable_datums")
        if final_trainable is None:
            final_trainable = torch.ones_like(train_data["rewards"], dtype=torch.bool)
        final_rewards = train_data["rewards"].reshape(-1)[final_trainable.bool().reshape(-1)]
        zero_signal = final_rewards.numel() == 0 or final_rewards.max() == final_rewards.min()
        zero_filter_enabled = bool(
            getattr(self.config, "filter_zero_advantage_datums", False)
        )
        if zero_signal and len(results) > 1:
            stats_tracker.get(workflow_context.stat_scope()).scalar(zero_variance_reward_group=1.0)
            logger.info(
                "All retained rewards identical for task %s (unprocessed mean=%.2f)",
                data["task_id"],
                mean_unprocessed_reward.item(),
            )
            if self.config.filter_zero_variance_groups:
                record_workload_stats(None)
                return None

        if zero_filter_enabled:
            tracker.scalar(**_measure_zero_centered_reward_candidates(train_data))

        if has_workload_sidechannel:
            retained_datums_per_rollout = record_workload_stats(train_data)
            self._attach_task_workload_sidecar(
                train_data,
                processed_results,
                retained_datums_per_rollout=retained_datums_per_rollout,
            )

        return train_data

    async def _process_trajectory_result(
        self,
        trajectory_data: dict | None,
        session: ArealProxySession,
        task_id: str,
        rollout_number: int,
    ) -> _ProcessedRolloutResult:
        # Export every requested session, including a rollout whose raw result
        # is None. The proxy can still contain completed model interactions
        # from work performed before a timeout/cancellation.
        completions = await session.export_interactions()
        completion_token_counts = _completion_token_counts(completions)
        workload = _rollout_workload(trajectory_data, completions, completion_token_counts)
        observed = trajectory_data is not None
        if trajectory_data is None:
            logger.warning("Rollout %s returned None for task %s", rollout_number, task_id)
            return _ProcessedRolloutResult(None, workload, observed)
        if not trajectory_data.get("trajectories"):
            logger.warning("No trajectories found for task %s rollout %s", task_id, rollout_number)
            return _ProcessedRolloutResult(None, workload, observed)

        efficiency_config = getattr(self.config, "token_efficiency_reward", None)
        if efficiency_config is not None and efficiency_config.enabled:
            attribution = annotate_policy_subtree_token_efficiency(
                trajectory_data,
                completion_token_counts,
                coefficient=efficiency_config.coefficient,
                reference_tokens=efficiency_config.reference_tokens,
                max_penalty=efficiency_config.max_penalty,
                input_token_weight=efficiency_config.input_token_weight,
                output_token_weight=efficiency_config.output_token_weight,
            )
            logger.info(
                "Policy-subtree token attribution for task %s rollout %s: "
                "policy=%s verifier=%s attributed=%s verifier_calls=%s "
                "ambiguous=%s unattributed=%s missing=%s",
                task_id,
                rollout_number,
                attribution.policy_trajectories,
                attribution.verifier_trajectories,
                attribution.attributed_completions,
                attribution.verifier_completions,
                attribution.ambiguous_completions,
                attribution.unattributed_completions,
                attribution.malformed_or_missing_completions,
            )

        use_depth_weighting = self.config.depth_level_weighting
        use_depth_discount = self.config.depth_level_discount_gamma is not None
        use_subagent_sampling = self.subagent_datum_sampler is not None
        train_data = get_train_data_for_trajectory_collection(
            trajectory_data,
            completions,
            task_id,
            self.filter_errors,
            self.reward_processor,
            self.merge_prefixes,
            concat_fn=concat_padded_tensors,
            include_traj_depth=use_depth_weighting or use_depth_discount or use_subagent_sampling,
            include_traj_start=use_depth_weighting or use_subagent_sampling,
            router_replay_config=self.router_replay_config,
            subagent_datum_sampler=self.subagent_datum_sampler,
        )
        if train_data is None:
            logger.warning("No train data found for task %s rollout %s", task_id, rollout_number)
        postmerge, policy_eligible, post_sampling = _rollout_datum_funnel(train_data)
        workload = replace(
            workload,
            postmerge_datums=postmerge,
            policy_eligible_datums=policy_eligible,
            post_sampling_datums=post_sampling,
        )
        return _ProcessedRolloutResult(train_data, workload, observed)

    async def _run_rollout_subprocess(
        self,
        executor: "ProcessPoolExecutor",
        engine: InferenceEngine,
        task_id: str,
        rollout_number: int,
        session: ArealProxySession,
    ) -> _SubprocessRolloutOutcome:
        from dataclasses import asdict

        from platoon.train.areal.subprocess_worker import run_rollout_subprocess

        config = self._build_rollout_config(engine, session)
        hard_timeout = (
            (self.config.rollout_config.timeout or 900)
            + _SUBPROCESS_INIT_BUDGET_SECONDS
            + _SUBPROCESS_CLEANUP_GRACE_SECONDS
        )
        started = time.monotonic()

        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    run_rollout_subprocess,
                    self.rollout_fn.__module__,
                    self.rollout_fn.__name__,
                    self.get_task_fn.__module__,
                    self.get_task_fn.__name__,
                    task_id,
                    asdict(config.rollout_config),
                ),
                # Let the child SIGALRM own the process-tree deadline, then
                # allow the executor future a short interval to report death.
                timeout=hard_timeout + _PARENT_FUTURE_GRACE_SECONDS,
            )
            return _SubprocessRolloutOutcome(
                result=result,
                elapsed_seconds=time.monotonic() - started,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Subprocess hard timeout (%ss) for task %s rollout %s", hard_timeout, task_id, rollout_number
            )
            return _SubprocessRolloutOutcome(
                result=None,
                elapsed_seconds=time.monotonic() - started,
                force_pool_shutdown=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Subprocess rollout failed for task %s rollout %s", task_id, rollout_number)
            return _SubprocessRolloutOutcome(
                result=None,
                elapsed_seconds=time.monotonic() - started,
                force_pool_shutdown=True,
            )

    @staticmethod
    async def _terminate_executor_processes(
        executor: "ProcessPoolExecutor",
        grace_seconds: float,
    ) -> None:
        """Terminate every worker/process group in this group-owned executor."""

        # Python 3.12 has no public ProcessPoolExecutor.terminate_workers().
        # This executor is private to one rollout group, so snapshotting its
        # multiprocessing.Process values is contained and cannot affect other
        # groups.  Guard the private lookup for forward compatibility.
        processes = list(getattr(executor, "_processes", {}).values())

        def _signal_process(process, sig: int) -> None:
            pid = getattr(process, "pid", None)
            if not pid:
                return
            try:
                # Workers call setpgrp() before executing rollouts.  Never use
                # killpg unless that invariant is true, or we could signal the
                # controller's own process group during a startup race.
                if os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                logger.warning("Unable to signal rollout subprocess pid=%s", pid)

        for process in processes:
            _signal_process(process, signal.SIGTERM)

        deadline = time.monotonic() + grace_seconds
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            await asyncio.sleep(min(0.1, max(deadline - time.monotonic(), 0.0)))

        for process in processes:
            if process.is_alive():
                _signal_process(process, signal.SIGKILL)

    @staticmethod
    async def _close_proxy_session(session: ArealProxySession) -> None:
        try:
            await asyncio.wait_for(
                session.__aexit__(None, None, None),
                timeout=_PROXY_SESSION_CLOSE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out closing AReaL proxy session %s after %ss",
                session.session_id,
                _PROXY_SESSION_CLOSE_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Failed to close AReaL proxy session %s", session.session_id)

    async def _arun_episode_with_subprocesses(
        self,
        engine: InferenceEngine,
        data: dict,
    ) -> list[_ProcessedRolloutResult]:
        from concurrent.futures import ProcessPoolExecutor

        http_session = await workflow_context.get_aiohttp_session()
        sessions: list[ArealProxySession] = []
        outcomes: list[_SubprocessRolloutOutcome | None] = [None] * self.config.group_size
        rollout_tasks: dict[asyncio.Task[_SubprocessRolloutOutcome], int] = {}
        tail_cancelled = 0
        tail_cutoff_triggered = False
        settled_outcomes = 0
        force_pool_shutdown = False
        executor = ProcessPoolExecutor(max_workers=self.config.group_size, mp_context=mp.get_context("spawn"))
        proxy_base_url = self._require_proxy_base_url()
        try:
            for rollout_number in range(self.config.group_size):
                session = ArealProxySession(
                    session=http_session,
                    base_url=proxy_base_url,
                    task_id=self._session_task_id(data["task_id"], rollout_number),
                    admin_api_key=self.proxy_admin_api_key,
                )
                await session.__aenter__()
                sessions.append(session)

            rollout_tasks = {
                asyncio.create_task(
                    self._run_rollout_subprocess(
                        executor,
                        engine,
                        data["task_id"],
                        rollout_number,
                        session,
                    )
                ): rollout_number
                for rollout_number, session in enumerate(sessions)
            }
            pending = set(rollout_tasks)
            tail_deadline: float | None = None
            while pending:
                wait_timeout = None
                if tail_deadline is not None:
                    wait_timeout = max(tail_deadline - time.monotonic(), 0.0)
                done, pending = await asyncio.wait(
                    pending,
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    tail_cutoff_triggered = True
                    force_pool_shutdown = True
                    logger.warning(
                        "Cutting off %s tail rollout(s) for task %s after %ss straggler grace",
                        len(pending),
                        data["task_id"],
                        self.config.straggler_timeout_seconds,
                    )
                    break
                for task in done:
                    rollout_number = rollout_tasks[task]
                    outcome = task.result()
                    outcomes[rollout_number] = outcome
                    # Tail grace is relative to terminal peers, not to usable
                    # training results. An interrupted partial or failed-closed
                    # wrapper has still stopped making progress; excluding it
                    # can leave the final live member waiting until its much
                    # longer absolute rollout deadline. Training eligibility is
                    # checked separately via min_successful_group_size.
                    settled_outcomes += 1
                    force_pool_shutdown = force_pool_shutdown or outcome.force_pool_shutdown

                if (
                    pending
                    and tail_deadline is None
                    and self.config.straggler_timeout_seconds is not None
                    and settled_outcomes
                    >= (
                        self.config.straggler_quorum
                        if self.config.straggler_quorum is not None
                        else self.config.group_size - 1
                    )
                ):
                    tail_deadline = time.monotonic() + self.config.straggler_timeout_seconds
        finally:
            # A rollout can finish after ``asyncio.wait`` reports the tail
            # deadline but before this finally block runs.  Retain that result
            # rather than silently counting it as a cancelled straggler.
            for task, rollout_number in rollout_tasks.items():
                if outcomes[rollout_number] is not None or not task.done() or task.cancelled():
                    continue
                try:
                    outcome = task.result()
                except Exception:
                    logger.exception(
                        "Rollout wrapper failed for task %s rollout %s",
                        data["task_id"],
                        rollout_number,
                    )
                    force_pool_shutdown = True
                else:
                    outcomes[rollout_number] = outcome
                    force_pool_shutdown = force_pool_shutdown or outcome.force_pool_shutdown

            pending_tasks = [task for task in rollout_tasks if not task.done()]
            if tail_cutoff_triggered:
                # Exclude members that won the race and completed between the
                # deadline observation and this cleanup block.
                tail_cancelled = len(pending_tasks)
            if pending_tasks:
                force_pool_shutdown = True
                for task in pending_tasks:
                    task.cancel()

            if force_pool_shutdown:
                # Reap the process tree before awaiting wrappers.  Waiting
                # first can deadlock if third-party code below run_in_executor
                # suppresses cancellation.
                await self._terminate_executor_processes(
                    executor,
                    self.config.subprocess_shutdown_grace_seconds,
                )

            if pending_tasks:
                done_after_cancel, still_pending = await asyncio.wait(
                    pending_tasks,
                    timeout=_ROLLOUT_TASK_CANCEL_GRACE_SECONDS,
                )
                for task in done_after_cancel:
                    if not task.cancelled():
                        task.exception()
                if still_pending:
                    logger.warning(
                        "%s rollout wrapper task(s) remained pending after process-pool termination",
                        len(still_pending),
                    )

                    def _consume_late_task(task: asyncio.Task) -> None:
                        if not task.cancelled():
                            task.exception()

                    for task in still_pending:
                        task.add_done_callback(_consume_late_task)

            if force_pool_shutdown:
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)

            await asyncio.gather(
                *[self._close_proxy_session(session) for session in sessions],
            )

        tracker = stats_tracker.get(workflow_context.stat_scope())
        tracker.scalar(group_tail_cancelled=float(tail_cancelled))
        for outcome in outcomes:
            if outcome is not None:
                tracker.scalar(group_member_wall_time=outcome.elapsed_seconds)
        raw_results = [outcome.result if outcome is not None else None for outcome in outcomes]

        return await asyncio.gather(
            *[
                self._process_trajectory_result(raw_result, session, data["task_id"], rollout_number)
                for rollout_number, (raw_result, session) in enumerate(zip(raw_results, sessions, strict=True))
            ]
        )

    async def _arun_episode_single(
        self,
        engine: InferenceEngine,
        data: dict,
        rollout_number: int,
    ) -> _ProcessedRolloutResult:
        task_id = data["task_id"]
        proxy_base_url = self._require_proxy_base_url()
        session = ArealProxySession(
            session=await workflow_context.get_aiohttp_session(),
            base_url=proxy_base_url,
            task_id=self._session_task_id(task_id, rollout_number),
            admin_api_key=self.proxy_admin_api_key,
        )
        trajectory_data = None
        try:
            await session.__aenter__()
            config = self._build_rollout_config(engine, session)
            task = self.get_task_fn(task_id)
            if config.rollout_config.max_steps is not None:
                task.max_steps = config.rollout_config.max_steps
            trajectory_data = await asyncio.create_task(self.rollout_fn(task, config.rollout_config))
        except Exception:
            logger.exception("Error in AReaL workflow for task %s rollout %s", task_id, rollout_number)
        finally:
            await session.__aexit__(None, None, None)

        return await self._process_trajectory_result(trajectory_data, session, task_id, rollout_number)
