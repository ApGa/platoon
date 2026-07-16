"""Platoon AReaL Trainer for distributed training."""

from __future__ import annotations

import math
import os
import secrets
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from areal.api import WorkflowLike
from areal.infra import RolloutController, current_platform
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.environ import is_single_controller
from areal.utils.perf_tracer import Category
from datasets import Dataset

from platoon.train.areal.actor import PlatoonPPOActor
from platoon.train.areal.batch_transforms import (
    BatchTransform,
    BatchTransformContext,
    build_default_batch_transforms,
    get_batch_size,
    index_batch,
    localize_rtensors,
    run_batch_transforms,
    split_batch_to_trajectories,
)
from platoon.train.areal.config_defs import PlatoonArealRLTrainerConfig, PlatoonPPOActorConfig
from platoon.train.areal.preallocated_slurm import PreallocatedSlurmScheduler
from platoon.train.areal.workflow_serialization import normalize_remote_workflow
from platoon.utils.areal_data_processing import OPTIONAL_REWARD_METRIC_MASK_PREFIX
from platoon.utils.rollout_workload import RolloutWorkload, sum_rollout_workloads

if TYPE_CHECKING:
    # Imported only for typing; the runtime import is deferred to the Megatron
    # branch of _create_train_engine so FSDP-only runs never import Megatron /
    # Transformer Engine.
    from platoon.train.areal.actor import PlatoonMegatronPPOActor

logger = logging.getLogger("PlatoonArealRLTrainer")

# AReaL's publicly documented default proxy admin key. AReaL refuses to bind
# the proxy rollout server to a routable (non-loopback) host while the key is
# still this value, since anyone reachable could call admin endpoints.
_DEFAULT_AREAL_ADMIN_API_KEY = "areal-admin-key"
_ROUTER_REPLAY_FIELDS = ("routed_experts", "routed_experts_valid")
_ROUTER_REPLAY_ORDER_FIELD = "_platoon_router_replay_order"
_TRAJECTORY_SEGMENT_ID_FIELD = "_platoon_trajectory_segment_id"
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
_WORKFLOW_STAT_KEYS = (
    "task_reward",
    "task_reward_valid",
    "num_steps",
    "num_input_tokens",
    "num_output_tokens",
)


def _is_workflow_stat_key(key: str) -> bool:
    return (
        key in _WORKFLOW_STAT_KEYS
        or key.startswith(_WORKLOAD_SIDECAR_PREFIX)
        or key.startswith("root_")
        or key.startswith("reward/")
        or key.startswith(OPTIONAL_REWARD_METRIC_MASK_PREFIX)
    )


@dataclass(frozen=True)
class _AcceptedBatchWorkload:
    workload: RolloutWorkload
    tasks: int
    requested_rollouts: int
    observed_rollouts: int
    trainable_rollouts: int
    task_retained_datums: int


def _sidecar_nonnegative_int(item: dict[str, Any], key: str) -> int:
    value = localize_rtensors(item[key])
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"Workload sidecar {key} must contain exactly one value")
        scalar = value.item()
    else:
        scalar = value
    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        raise TypeError(f"Workload sidecar {key} must be numeric")
    result = int(scalar)
    if result < 0 or float(result) != float(scalar):
        raise ValueError(f"Workload sidecar {key} must be a non-negative integer")
    return result


def _extract_accepted_batch_workload(
    rollout_batch: list[dict[str, Any]] | None,
) -> _AcceptedBatchWorkload | None:
    """Sum exact inference work for accepted outer-batch task groups."""

    workloads: list[RolloutWorkload] = []
    requested_rollouts = 0
    observed_rollouts = 0
    trainable_rollouts = 0
    task_retained_datums = 0
    required = {
        *_WORKLOAD_SIDECAR_FIELDS.values(),
        *_WORKLOAD_DATUM_SIDECAR_FIELDS.values(),
        _WORKLOAD_REQUESTED_ROLLOUTS_KEY,
        _WORKLOAD_OBSERVED_ROLLOUTS_KEY,
        _WORKLOAD_TRAINABLE_ROLLOUTS_KEY,
        _WORKLOAD_TASK_RETAINED_DATUMS_KEY,
    }
    for item in rollout_batch or []:
        if not isinstance(item, dict):
            continue
        present = required.intersection(item)
        if not present:
            # Backward compatibility for arbitrary AReaL workflows that do not
            # implement Platoon's workload side channel.
            continue
        missing = required.difference(item)
        if missing:
            raise ValueError(f"Incomplete workload sidecar; missing {sorted(missing)}")
        fields = {
            field: _sidecar_nonnegative_int(item, key)
            for field, key in _WORKLOAD_SIDECAR_FIELDS.items()
        }
        datum_fields = {
            field: _sidecar_nonnegative_int(item, key)
            for field, key in _WORKLOAD_DATUM_SIDECAR_FIELDS.items()
        }
        workloads.append(RolloutWorkload(**fields, **datum_fields))
        requested_rollouts += _sidecar_nonnegative_int(item, _WORKLOAD_REQUESTED_ROLLOUTS_KEY)
        observed_rollouts += _sidecar_nonnegative_int(item, _WORKLOAD_OBSERVED_ROLLOUTS_KEY)
        trainable_rollouts += _sidecar_nonnegative_int(item, _WORKLOAD_TRAINABLE_ROLLOUTS_KEY)
        task_retained_datums += _sidecar_nonnegative_int(
            item,
            _WORKLOAD_TASK_RETAINED_DATUMS_KEY,
        )

    if not workloads:
        return None
    workload = sum_rollout_workloads(workloads)
    if task_retained_datums > workload.post_sampling_datums:
        raise ValueError(
            "Accepted task-retained datums exceed post-sampling candidates: "
            f"retained={task_retained_datums}, post_sampling={workload.post_sampling_datums}"
        )
    return _AcceptedBatchWorkload(
        workload=workload,
        tasks=len(workloads),
        requested_rollouts=requested_rollouts,
        observed_rollouts=observed_rollouts,
        trainable_rollouts=trainable_rollouts,
        task_retained_datums=task_retained_datums,
    )


def _training_batch_workload_metrics(
    trajectories: list[dict[str, Any]] | None,
    *,
    total_postmerge_datums: int | None = None,
) -> dict[str, float]:
    """Count the post-filter datums and tokens actually sent to PPO update."""

    attention_tokens = 0
    action_tokens = 0
    values = trajectories or []
    for index, trajectory in enumerate(values):
        attention_mask = localize_rtensors(trajectory.get("attention_mask"))
        loss_mask = localize_rtensors(trajectory.get("loss_mask"))
        if not torch.is_tensor(attention_mask) or not torch.is_tensor(loss_mask):
            raise TypeError(f"Training datum {index} requires tensor attention_mask/loss_mask")
        if attention_mask.shape != loss_mask.shape:
            raise ValueError(
                f"Training datum {index} mask shape mismatch: "
                f"attention={tuple(attention_mask.shape)}, loss={tuple(loss_mask.shape)}"
            )
        valid = attention_mask.bool()
        attention_tokens += int(valid.sum().item())
        action_tokens += int((loss_mask.bool() & valid).sum().item())
    submitted_datums = len(values)
    result = {
        "workload/training_batch/total_submitted_datums": float(submitted_datums),
        "workload/training_batch/total_attention_tokens": float(attention_tokens),
        "workload/training_batch/total_action_tokens": float(action_tokens),
    }
    if total_postmerge_datums is not None:
        if total_postmerge_datums < submitted_datums:
            raise ValueError(
                "Submitted training datums cannot exceed accepted post-merge datums: "
                f"submitted={submitted_datums}, postmerge={total_postmerge_datums}"
            )
        result["workload/training_batch/total_non_submitted_datums"] = float(
            total_postmerge_datums - submitted_datums
        )
    return result


def _detach_router_replay_sidechannels(
    trajectories: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]] | None]:
    """Keep large route tensors out of non-training AReaL batch utilities."""

    presence = [tuple(field in trajectory for field in _ROUTER_REPLAY_FIELDS) for trajectory in trajectories]
    if not any(any(item) for item in presence):
        return trajectories, None
    if any(item != (True, True) for item in presence):
        raise RuntimeError(
            "R3 sidechannel mismatch: every trajectory must contain both routed_experts and routed_experts_valid"
        )

    stripped: list[dict[str, Any]] = []
    sidechannels: list[tuple[Any, Any]] = []
    for trajectory in trajectories:
        if _ROUTER_REPLAY_ORDER_FIELD in trajectory:
            raise RuntimeError(f"Reserved R3 ordering key {_ROUTER_REPLAY_ORDER_FIELD!r} is already present")
        shallow = dict(trajectory)
        sidechannels.append(
            (
                shallow.pop(_ROUTER_REPLAY_FIELDS[0]),
                shallow.pop(_ROUTER_REPLAY_FIELDS[1]),
            )
        )
        stripped.append(shallow)
    return stripped, sidechannels


def _add_router_replay_order_markers(
    trajectories: list[dict[str, Any]],
    sidechannels: list[tuple[Any, Any]] | None,
) -> list[dict[str, Any]]:
    if sidechannels is None:
        return trajectories
    if len(trajectories) != len(sidechannels):
        raise RuntimeError("R3 sidechannel count changed before advantage computation")

    marked: list[dict[str, Any]] = []
    for index, (trajectory, (routes, valid)) in enumerate(zip(trajectories, sidechannels, strict=True)):
        if not torch.is_tensor(routes) or not torch.is_tensor(valid):
            raise RuntimeError("R3 sidechannels must be local tensors before advantage computation")
        if routes.ndim != 4 or valid.ndim != 2 or routes.shape[:2] != valid.shape:
            raise RuntimeError(
                f"R3 sidechannel shape mismatch: routes={tuple(routes.shape)}, valid={tuple(valid.shape)}"
            )
        shallow = dict(trajectory)
        if _ROUTER_REPLAY_ORDER_FIELD in shallow:
            raise RuntimeError(f"Reserved R3 ordering key {_ROUTER_REPLAY_ORDER_FIELD!r} is already present")
        shallow[_ROUTER_REPLAY_ORDER_FIELD] = torch.full(
            (routes.shape[0],),
            index,
            dtype=torch.int64,
            device=routes.device,
        )
        marked.append(shallow)
    return marked


def _reattach_router_replay_sidechannels(
    trajectories: list[dict[str, Any]],
    sidechannels: list[tuple[Any, Any]] | None,
) -> tuple[list[dict[str, Any]], list[Any] | None]:
    if sidechannels is None:
        return trajectories, None
    if len(trajectories) != len(sidechannels):
        raise RuntimeError(f"R3 advantage batch count changed: expected {len(sidechannels)}, got {len(trajectories)}")

    marker_cleanup: list[Any] = []
    for index, (trajectory, (routes, valid)) in enumerate(zip(trajectories, sidechannels, strict=True)):
        if not isinstance(trajectory, dict):
            raise RuntimeError(f"R3 advantage item {index} is not a dictionary")
        if any(field in trajectory for field in _ROUTER_REPLAY_FIELDS):
            raise RuntimeError(f"R3 advantage item {index} unexpectedly contains replay sidechannel keys")
        raw_marker = trajectory.pop(_ROUTER_REPLAY_ORDER_FIELD, None)
        marker_cleanup.append(raw_marker)
        marker = localize_rtensors(raw_marker)
        if not torch.is_tensor(marker) or marker.numel() != routes.shape[0] or not bool((marker == index).all()):
            raise RuntimeError(f"R3 advantage batch order changed at item {index}")
        # Reattach the exact compact CPU tensors retained before non-training
        # compute. No copy or generic concat/split round trip is involved.
        trajectory[_ROUTER_REPLAY_FIELDS[0]] = routes
        trajectory[_ROUTER_REPLAY_FIELDS[1]] = valid
    return trajectories, marker_cleanup


def _batch_cleanup_targets(
    raw_rollout_batch: Any,
    rollout_batch: Any,
    adv_batch: Any,
    router_replay_marker_cleanup: Any,
) -> tuple[Any, ...]:
    """Keep every RTensor-bearing container visible to clear_batches."""

    return tuple(
        target
        for target in (
            raw_rollout_batch,
            rollout_batch,
            adv_batch,
            router_replay_marker_cleanup,
        )
        if target is not None
    )


def _filter_zero_centered_reward_batch(
    batch: dict[str, Any],
    *,
    dispatch_dp_size: int,
    ensure_batch_divisible_by: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Remove exact scalar-zero candidates after global multiplicative transforms.

    All nonzero candidates are preserved whenever zero datums can provide the
    minimum structural padding required by actor dispatch. Removed zero action
    tokens remain represented in the policy-loss denominator by scaling the
    retained scalar rewards before advantage construction.
    """

    if dispatch_dp_size < 1 or ensure_batch_divisible_by < 1:
        raise ValueError("dispatch and configured divisors must be positive")
    batch_size = get_batch_size(batch)
    rewards = localize_rtensors(batch.get("rewards"))
    loss_mask = localize_rtensors(batch.get("loss_mask"))
    attention_mask = localize_rtensors(batch.get("attention_mask"))
    if not torch.is_tensor(rewards) or rewards.shape[0] != batch_size:
        raise ValueError("Zero-reward filtering requires one scalar reward per datum")
    flat_rewards = rewards.reshape(batch_size, -1)
    if flat_rewards.shape[1] != 1:
        raise ValueError("Zero-reward filtering requires scalar rewards")
    if not torch.is_tensor(loss_mask) or loss_mask.shape[0] != batch_size:
        raise ValueError("Zero-reward filtering requires per-datum loss_mask")
    if not torch.is_tensor(attention_mask) or attention_mask.shape[0] != batch_size:
        raise ValueError("Zero-reward filtering requires per-datum attention_mask")

    zero_mask = flat_rewards[:, 0] == 0
    zero_indices = torch.nonzero(zero_mask, as_tuple=False).reshape(-1)
    nonzero_indices = torch.nonzero(~zero_mask, as_tuple=False).reshape(-1)
    per_datum_loss_tokens = loss_mask.bool().reshape(batch_size, -1).sum(dim=1)
    per_datum_attention_tokens = attention_mask.bool().reshape(batch_size, -1).sum(dim=1)

    full_divisor = math.lcm(dispatch_dp_size, ensure_batch_divisible_by)
    input_divisibility_fallback = int(
        batch_size < full_divisor or batch_size % full_divisor != 0
    )
    divisor = dispatch_dp_size if input_divisibility_fallback else full_divisor
    padding_count = (-int(nonzero_indices.numel())) % divisor if nonzero_indices.numel() else 0
    can_zero_pad = (
        nonzero_indices.numel() > 0
        and padding_count <= zero_indices.numel()
        and nonzero_indices.numel() + padding_count >= dispatch_dp_size
    )
    if can_zero_pad and padding_count:
        order = torch.randperm(zero_indices.numel(), device=zero_indices.device)
        padding_indices = zero_indices.index_select(0, order[:padding_count])
    else:
        padding_indices = zero_indices[:0]

    retained_nonzero = nonzero_indices
    divisibility_trimmed = nonzero_indices[:0]
    if not can_zero_pad:
        trim_count = (
            int(nonzero_indices.numel())
            if nonzero_indices.numel() < dispatch_dp_size
            else int(nonzero_indices.numel()) % dispatch_dp_size
        )
        if trim_count:
            order = torch.randperm(nonzero_indices.numel(), device=nonzero_indices.device)
            divisibility_trimmed = nonzero_indices.index_select(0, order[:trim_count])
            retained_nonzero = nonzero_indices.index_select(0, order[trim_count:])

    retained_indices = torch.cat((retained_nonzero, padding_indices)).sort().values
    retained_zero_mask = torch.zeros(batch_size, dtype=torch.bool, device=zero_mask.device)
    if padding_indices.numel():
        retained_zero_mask[padding_indices] = True
    filtered_zero_indices = torch.nonzero(
        zero_mask & ~retained_zero_mask,
        as_tuple=False,
    ).reshape(-1)

    retained_loss_tokens = int(
        per_datum_loss_tokens.index_select(0, retained_indices).sum().item()
    ) if retained_indices.numel() else 0
    filtered_zero_loss_tokens = int(
        per_datum_loss_tokens.index_select(0, filtered_zero_indices).sum().item()
    ) if filtered_zero_indices.numel() else 0
    denominator_tokens = retained_loss_tokens + filtered_zero_loss_tokens
    denominator_scale = (
        float(retained_loss_tokens) / float(denominator_tokens)
        if denominator_tokens > 0
        else 1.0
    )

    metrics = {
        "input_datums": float(batch_size),
        "zero_advantage_datums": float(zero_indices.numel()),
        "zero_padding_datums": float(padding_indices.numel()),
        "filtered_zero_advantage_datums": float(filtered_zero_indices.numel()),
        "filtered_zero_advantage_loss_tokens": float(filtered_zero_loss_tokens),
        "filtered_zero_advantage_attention_tokens": float(
            per_datum_attention_tokens.index_select(0, filtered_zero_indices).sum().item()
            if filtered_zero_indices.numel()
            else 0
        ),
        "divisibility_trimmed_datums": float(divisibility_trimmed.numel()),
        "divisibility_trimmed_loss_tokens": float(
            per_datum_loss_tokens.index_select(0, divisibility_trimmed).sum().item()
            if divisibility_trimmed.numel()
            else 0
        ),
        "divisibility_trimmed_attention_tokens": float(
            per_datum_attention_tokens.index_select(0, divisibility_trimmed).sum().item()
            if divisibility_trimmed.numel()
            else 0
        ),
        "input_divisibility_fallback": float(input_divisibility_fallback),
        "dispatch_divisor": float(divisor),
        "retained_datums": float(retained_indices.numel()),
        "retained_loss_tokens": float(retained_loss_tokens),
        "policy_gradient_denominator_tokens": float(denominator_tokens),
        "policy_gradient_denominator_scale": denominator_scale,
    }
    if not retained_indices.numel():
        return None, metrics

    retained = index_batch(batch, retained_indices)
    if denominator_scale != 1.0:
        retained["rewards"] = localize_rtensors(retained["rewards"]) * denominator_scale
    return retained, metrics


def _normalization_is_active(config: Any) -> bool:
    if config is None:
        return False
    if isinstance(config, dict):
        return config.get("mean_level") is not None or config.get("std_level") is not None
    return getattr(config, "mean_level", None) is not None or getattr(config, "std_level", None) is not None


def _zero_reward_filter_incompatibilities(
    config: PlatoonArealRLTrainerConfig,
    *,
    custom_batch_transforms: list[BatchTransform] | None = None,
) -> list[str]:
    """Return features for which scalar-zero reward is not a safe proxy."""

    actor = config.actor
    reasons: list[str] = []
    if float(getattr(actor, "kl_ctl", 0.0)) != 0.0:
        reasons.append("actor.kl_ctl != 0")
    if float(getattr(actor, "reward_bias", 0.0)) != 0.0:
        reasons.append("actor.reward_bias != 0")
    if _normalization_is_active(getattr(actor, "reward_norm", None)):
        reasons.append("actor.reward_norm is active")
    if _normalization_is_active(getattr(actor, "adv_norm", None)):
        reasons.append("actor.adv_norm is active")
    if bool(getattr(actor, "overlong_reward_penalty", False)):
        reasons.append("actor.overlong_reward_penalty is enabled")
    if getattr(config, "critic", None) is not None:
        reasons.append("critic objective is present")
    if getattr(config, "teacher", None) is not None:
        reasons.append("teacher/distillation objective is present")
    model_path = str(getattr(actor, "path", "")).lower()
    megatron = getattr(actor, "megatron", None)
    bridge_type = (
        megatron.get("bridge_type")
        if isinstance(megatron, dict)
        else getattr(megatron, "bridge_type", None)
    )
    if (
        bridge_type == "megatron-bridge"
        and ("qwen3.5" in model_path or "qwen3.6" in model_path)
        and ("a3b" in model_path or "moe" in model_path)
    ):
        reasons.append(
            "Qwen3.5/3.6 MoE Megatron-Bridge has an independent global router auxiliary loss"
        )
    if custom_batch_transforms:
        reasons.append("custom batch transforms are present (additive transforms are incompatible)")
    return reasons


def _warn_for_zero_reward_filter_assumptions(
    config: PlatoonArealRLTrainerConfig,
    *,
    custom_batch_transforms: list[BatchTransform] | None = None,
) -> None:
    if not config.workflow_config.filter_zero_advantage_datums:
        return
    reasons = _zero_reward_filter_incompatibilities(
        config,
        custom_batch_transforms=custom_batch_transforms,
    )
    suffix = (
        " Detected incompatible settings: " + "; ".join(reasons) + "."
        if reasons
        else " Current actor settings satisfy the known reward-only constraints."
    )
    warnings.warn(
        "workflow_config.filter_zero_advantage_datums uses centered scalar reward as an early "
        "proxy for final policy advantage. Disable it when KL is nonzero, reward/advantage "
        "normalization or reward bias/overlong penalty is active, a critic or teacher objective "
        "is present, the model has an independent MoE/router objective, or a custom transform "
        "adds to rewards." + suffix,
        RuntimeWarning,
        stacklevel=2,
    )


def _evaluation_enabled(config: Any) -> bool:
    """Return whether AReaL can ever schedule evaluation for this run."""

    evaluator = getattr(config, "evaluator", None)
    if evaluator is None:
        # Preserve upstream behavior for custom/legacy configs whose evaluator
        # shape is unknown rather than silently suppressing evaluation.
        return True
    return bool(getattr(evaluator, "eval_before_train", False)) or any(
        getattr(evaluator, field, None) is not None
        for field in ("freq_epochs", "freq_steps", "freq_secs")
    )


def _normalize_proxy_admin_api_key(config: PlatoonArealRLTrainerConfig) -> None:
    """Ensure the AReaL proxy admin key is a unique secret, not the default.

    AReaL validates ``rollout.agent.admin_api_key`` on the proxy server (and
    adopts it as the server's accepted admin key), while Platoon's client
    authenticates management calls with ``rollout.admin_api_key``. The two must
    share a single secret. Operators can pin a value via the
    ``PLATOON_AREAL_ADMIN_API_KEY`` env var; otherwise a per-run random token is
    generated. Mutating the shared ``config.rollout`` here (before the trainer
    builds its train/eval rollout controllers) covers every plugin config
    without per-YAML edits.
    """
    rollout = config.rollout
    candidates = [rollout.admin_api_key]
    if rollout.agent is not None:
        candidates.append(rollout.agent.admin_api_key)

    configured = next(
        (key for key in candidates if key and key != _DEFAULT_AREAL_ADMIN_API_KEY),
        None,
    )
    if configured is not None:
        resolved = configured
    else:
        env_key = (os.environ.get("PLATOON_AREAL_ADMIN_API_KEY") or "").strip()
        resolved = env_key or f"platoon-{secrets.token_hex(16)}"

    rollout.admin_api_key = resolved
    if rollout.agent is not None:
        rollout.agent.admin_api_key = resolved


class PlatoonArealRLTrainer(PPOTrainer):
    """Platoon's AReaL-based RL trainer."""

    def __init__(
        self,
        config: PlatoonArealRLTrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None,
        batch_transforms: list[BatchTransform] | None = None,
    ):
        # Resolve a unique proxy admin key before super().__init__() builds the
        # rollout controllers, so both the proxy server (rollout.agent.admin_api_key)
        # and Platoon's client (rollout.admin_api_key) share one non-default secret.
        _warn_for_zero_reward_filter_assumptions(
            config,
            custom_batch_transforms=batch_transforms,
        )
        _normalize_proxy_admin_api_key(config)
        super().__init__(config=config, train_dataset=train_dataset, valid_dataset=val_dataset)
        self.proxy_admin_api_key = self.config.rollout.admin_api_key
        self.proxy_base_url: str | None = None
        self.eval_proxy_base_url: str | None = None
        self.batch_transforms = self._build_batch_transforms(batch_transforms)
        self._start_platoon_proxies()

    def _init_scheduler(self):
        if self.config.scheduler.type == "slurm_prealloc":
            return PreallocatedSlurmScheduler(exp_config=self.config)
        return super()._init_scheduler()

    def _init_rollout(
        self,
        rollout_config: Any,
        is_eval: bool = False,
        lora_path: str | None = None,
    ) -> Any:
        """Avoid constructing an unused colocated evaluation controller."""

        if is_eval and not _evaluation_enabled(self.config):
            return None
        return super()._init_rollout(
            rollout_config,
            is_eval=is_eval,
            lora_path=lora_path,
        )

    def _create_train_engine(self, actor_config, alloc):
        actor_cls: type[PlatoonPPOActor | PlatoonMegatronPPOActor] | None = None
        if isinstance(actor_config, PlatoonPPOActorConfig):
            if alloc.backend == "fsdp":
                actor_cls = PlatoonPPOActor
            elif alloc.backend == "megatron":
                # Deferred import: pulls in Megatron / Transformer Engine only
                # when the Megatron backend is actually selected.
                from platoon.train.areal.actor import PlatoonMegatronPPOActor

                actor_cls = PlatoonMegatronPPOActor
        if actor_cls is not None:
            if is_single_controller():
                actor = actor_cls.as_controller(actor_config, self.scheduler)
            else:
                actor = actor_cls(actor_config)
            actor.create_process_group(parallel_strategy=alloc.parallel)
            return actor
        return super()._create_train_engine(actor_config, alloc)

    def _proxy_mode(self) -> str:
        # OpenAIProxyConfig was folded into AgentConfig at AReaL HEAD; the proxy
        # mode now lives on rollout.agent (which always has a default factory).
        agent_cfg = self.config.rollout.agent
        return agent_cfg.mode if agent_cfg is not None else "inline"

    def _resolve_proxy_base_url(self, controller: RolloutController) -> str | None:
        mode = self._proxy_mode()
        if mode == "online":
            controller.start_proxy_gateway()
            return controller.proxy_gateway_addr
        return None

    def _start_platoon_proxies(self) -> None:
        if not is_single_controller():
            raise NotImplementedError("Platoon's updated AReaL integration requires single-controller mode")
        if not isinstance(self.rollout, RolloutController):
            raise TypeError("Expected rollout to be a RolloutController in single-controller mode")

        logger.info("Starting Platoon proxy workers for mode=%s", self._proxy_mode())
        self.rollout.start_proxy()
        self.proxy_base_url = self._resolve_proxy_base_url(self.rollout)

        if isinstance(self.eval_rollout, RolloutController):
            self.eval_rollout.start_proxy()
            self.eval_proxy_base_url = self._resolve_proxy_base_url(self.eval_rollout)
        else:
            self.eval_proxy_base_url = self.proxy_base_url

    def _build_batch_transforms(
        self,
        extra_batch_transforms: list[BatchTransform] | None = None,
    ) -> list[BatchTransform]:
        """Build the ordered full-batch transform pipeline.

        Ordering matters:
        1. The workflow has already applied per-group reward centering.
        2. The trainer performs canonical batch reduction/filtering.
        3. The trainer shuffles/trims to the final DP-divisible actor batch.
        4. These transforms run on that final retained batch, so transforms
           such as depth weighting normalize exactly the datums that train.
        5. Only then do ref/prox/teacher enrichment and advantage computation run.
        """

        transforms = build_default_batch_transforms(self.config)
        if extra_batch_transforms:
            transforms.extend(extra_batch_transforms)
        return transforms

    @staticmethod
    def _controller_dispatch_group_size() -> int:
        """Platoon workflows already own rollout multiplicity internally."""
        return 1

    def _actor_dispatch_dp_size(self) -> int:
        """Return the DP size used by AReaL controller tensor dispatch."""
        parallel_strategy = getattr(self.actor, "parallel_strategy", None)
        if parallel_strategy is not None and getattr(parallel_strategy, "dp_size", None) is not None:
            return int(parallel_strategy.dp_size)
        return int(self.actor.data_parallel_world_size)

    def _advance_logical_versions(self, new_version: int) -> None:
        """Keep engine/rollout versions aligned with the trainer's global step.

        This intentionally performs no optimizer, scheduler, or weight-broadcast
        operation. AReaL already uses this path for an empty rollout batch; an
        all-zero-advantage batch must follow the same invariant so checkpoints,
        staleness tracking, and the next rollout do not disagree about version.
        """

        self.actor.set_version(new_version)
        if self.critic is not None:
            self.critic.set_version(new_version)
        if self.ref is not None:
            self.ref.set_version(new_version)
        if self.teacher is not None:
            self.teacher.set_version(new_version)
        self.rollout.set_version(new_version)
        if self.eval_rollout is not None:
            self.eval_rollout.set_version(new_version)

    @staticmethod
    def _maybe_clear_device_cache(engine: Any) -> None:
        """Release cached CUDA blocks on an engine's GPU workers.

        The pre-migration SPMD trainer ran ``torch.cuda.empty_cache()`` on every
        rank between training phases. In single-controller mode this must be an
        RPC to the workers; engines without the RPC (e.g. stock AReaL critics)
        are skipped.
        """
        if engine is None:
            return
        clear = getattr(engine, "clear_device_cache", None)
        if clear is None:
            return
        with stats_tracker.record_timing("clear_device_cache"):
            clear()

    def _maybe_shuffle_and_trim_batch(self, batch: dict[str, Any]) -> dict[str, Any] | None:
        batch_size = get_batch_size(batch)
        if batch_size == 0:
            return None
        dispatch_dp_size = self._actor_dispatch_dp_size()

        index_device = None
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch_size:
                index_device = value.device
                break
        if index_device is None:
            # RTensors expose shape metadata but fetch CPU tensors when localized,
            # so CPU indices are the safest default until index_batch localizes.
            index_device = torch.device("cpu")

        indices = torch.arange(batch_size, device=index_device)

        # Match the pre-migration trimming semantics: enforce divisibility by
        # lcm(ensure_batch_divisible_by, dp_size) so a single trim preserves both
        # guarantees, and skip trimming entirely when the batch is smaller than
        # one full multiple (sequential %-trims could over-trim and break the
        # ensure_batch_divisible_by contract).
        ensure = math.lcm(
            max(int(self.config.rollout.ensure_batch_divisible_by), 1),
            dispatch_dp_size,
        )
        total = int(indices.numel())
        if total < dispatch_dp_size:
            return None
        remainder = total % ensure
        trim_count = remainder if remainder != 0 and total >= ensure else 0

        # Divisibility trimming always draws a random subset so later/deeper
        # trajectories are not systematically discarded.  The shuffle flag
        # controls only the order of the retained datums.
        if trim_count or self.config.rollout.shuffle_cross_task:
            selection_order = indices[torch.randperm(total, device=index_device)]
        else:
            selection_order = indices

        keep = torch.ones(total, dtype=torch.bool, device=index_device)
        if trim_count:
            depth = batch.get("traj_depth")
            if torch.is_tensor(depth) and depth.ndim >= 1 and depth.shape[0] == total:
                flat_depth = depth.detach().reshape(total).to(index_device)
                # Roots are mandatory sampling data.  Prefer trimming a random
                # subset of non-root datums, falling back to roots only when
                # there are not enough non-root candidates for divisibility.
                nonroot = selection_order[flat_depth.index_select(0, selection_order) != 0]
                root = selection_order[flat_depth.index_select(0, selection_order) == 0]
                trim_order = torch.cat((nonroot, root))
            else:
                trim_order = selection_order
            keep[trim_order[:trim_count]] = False

        if self.config.rollout.shuffle_cross_task:
            indices = selection_order[keep.index_select(0, selection_order)]
        else:
            indices = indices[keep]
        if int(indices.numel()) < dispatch_dp_size:
            return None

        selected = index_batch(batch, indices)
        segment_ids = selected.pop(_TRAJECTORY_SEGMENT_ID_FIELD, None)
        traj_start = selected.get("traj_start")
        if torch.is_tensor(segment_ids) and torch.is_tensor(traj_start):
            flat_segments = segment_ids.detach().reshape(-1)
            if flat_segments.numel() != indices.numel():
                raise ValueError("Trajectory segment IDs do not match the selected batch")
            repaired_start = torch.zeros_like(traj_start).reshape(-1)
            for segment_id in torch.unique(flat_segments):
                first = torch.nonzero(flat_segments == segment_id, as_tuple=False).reshape(-1)[0]
                repaired_start[int(first.item())] = 1
            selected["traj_start"] = repaired_start.reshape_as(traj_start)

        return selected

    def _reduce_rollout_batch(self, rollout_batch: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Reduce rollout items into the canonical trainer batch.

        This stage intentionally owns batch-cardinality changes such as consuming
        `trainable_datums`. User-extensible transforms only run after this
        reduction has produced a stable full batch.
        """
        rollout_batch = [item for item in rollout_batch if item]
        if not rollout_batch:
            return None

        # In single-controller mode prepare_batch returns remotized trajectories
        # whose values are RTensor handles, not torch.Tensors. AReaL's
        # concat_padded_tensors only concatenates tensor/list values and silently
        # keeps the *first* dict's value for anything else, which would drop every
        # rollout group but the first. Localize before concatenating.
        rollout_batch = [
            {
                key: value
                for key, value in localize_rtensors(item).items()
                if not _is_workflow_stat_key(key)
            }
            for item in rollout_batch
        ]
        batch = concat_padded_tensors(rollout_batch)

        # Workflow-level stat tensors were already consumed by rollout-side stats
        # recording and do not share the per-datum batch dim, so they cannot be
        # filtered/split consistently with the rest of the batch. Strip them
        # before concatenating task groups: optional metrics may legitimately be
        # absent from an entire group, and strict concatenation must not see that
        # structural mismatch.

        # Give every original trajectory a globally unique segment ID before
        # trainable-datum filtering.  A sampled/trimmed subset can then repair
        # exactly one traj_start marker for each surviving trajectory even if
        # the original first datum was removed.
        traj_start = batch.get("traj_start")
        if torch.is_tensor(traj_start):
            flat_start = traj_start.detach().reshape(-1)
            batch_size = get_batch_size(batch)
            if flat_start.numel() != batch_size:
                raise ValueError("traj_start does not match the rollout batch")
            normalized_start = flat_start != 0
            if batch_size and not bool(normalized_start[0]):
                # Defensive handling for an already-filtered leading segment.
                normalized_start = normalized_start.clone()
                normalized_start[0] = True
            batch[_TRAJECTORY_SEGMENT_ID_FIELD] = torch.cumsum(
                normalized_start.to(dtype=torch.int64),
                dim=0,
            )

        if "trainable_datums" in batch:
            trainable_mask = batch.pop("trainable_datums").bool()
            global_trainable = int(trainable_mask.sum().item())
            min_per_step = self._actor_dispatch_dp_size()
            if global_trainable < min_per_step:
                return None
            if not bool(trainable_mask.all()):
                indices = torch.nonzero(trainable_mask, as_tuple=False).squeeze(-1)
                batch = index_batch(batch, indices)

        return batch

    def _postprocess_rollout_batch(
        self,
        rollout_batch: list[dict[str, Any]],
        global_step: int,
        epoch: int,
        epoch_step: int,
    ) -> list[dict[str, Any]] | None:
        batch = self._reduce_rollout_batch(rollout_batch)
        if batch is None:
            return None

        # Trimming after depth weighting would normalize rewards using datums
        # that never reach the actor.  Establish the final retained batch first;
        # custom transforms likewise see exactly the batch that trains.
        batch = self._maybe_shuffle_and_trim_batch(batch)
        if batch is None:
            return None

        context = BatchTransformContext(
            config=self.config,
            actor_dp_world_size=self._actor_dispatch_dp_size(),
            global_step=global_step,
            epoch=epoch,
            epoch_step=epoch_step,
        )
        batch = run_batch_transforms(batch, self.batch_transforms, context)
        if batch is None:
            return None
        if bool(
            getattr(
                self.config.workflow_config,
                "filter_zero_advantage_datums",
                False,
            )
        ):
            batch, zero_filter_metrics = _filter_zero_centered_reward_batch(
                batch,
                dispatch_dp_size=self._actor_dispatch_dp_size(),
                ensure_batch_divisible_by=max(
                    int(self.config.rollout.ensure_batch_divisible_by),
                    1,
                ),
            )
            stats_tracker.scalar(
                **{
                    f"zero_advantage_filter/{key}": value
                    for key, value in zero_filter_metrics.items()
                },
                **{"zero_advantage_filter/enabled": 1.0},
            )
            if batch is None:
                return None
        # Depth/start are temporary trainer metadata.  The built-in depth
        # transform consumes them; this cleanup also covers sampling-only runs
        # and custom transform lists without leaking fields to the actor.
        batch.pop(_TRAJECTORY_SEGMENT_ID_FIELD, None)
        batch.pop("traj_depth", None)
        batch.pop("traj_start", None)
        # Restore AReaL's canonical per-trajectory representation so downstream
        # controller dispatch can rebalance work across DP groups.
        return split_batch_to_trajectories(batch)

    def train(
        self,
        workflow: WorkflowLike | None = None,
        eval_workflow: WorkflowLike | None = None,
        workflow_kwargs: dict[str, Any] | None = None,
        eval_workflow_kwargs: dict[str, Any] | None = None,
        dynamic_filter_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        total_epochs: int | None = None,
    ):
        config = self.config
        start_step = self.recover_info.last_step_info.next().global_step if self.recover_info is not None else 0
        workflow, workflow_kwargs = normalize_remote_workflow(
            workflow,
            workflow_kwargs,
        )
        eval_workflow, eval_workflow_kwargs = normalize_remote_workflow(
            eval_workflow,
            eval_workflow_kwargs,
        )

        if total_epochs is None:
            total_epochs = config.total_train_epochs
        if total_epochs <= 0:
            raise ValueError(f"Total epochs must be positive: {total_epochs}")
        steps_per_epoch = len(self.train_dataloader)
        max_steps = total_epochs * steps_per_epoch

        if workflow is None:
            agent_cfg = self.config.rollout.agent
            if agent_cfg is not None and agent_cfg.mode == "online":
                self._ensure_proxy_started()
            else:
                raise ValueError(
                    "workflow must be specified for train() unless "
                    "rollout.agent.mode='online' is configured. "
                    "Pass a RolloutWorkflow, AgentWorkflow, or callable."
                )
        elif self._requires_proxy_workflow(workflow):
            self._ensure_proxy_started()

        for global_step in range(start_step, max_steps):
            if config.total_train_steps is not None and global_step >= config.total_train_steps:
                break
            epoch = global_step // steps_per_epoch
            step = global_step % steps_per_epoch

            if self._should_offload_rollout:
                self._onload_rollout()
            with (
                stats_tracker.record_timing("rollout"),
                perf_tracer.trace_scope(
                    "train.rollout",
                    category=Category.COMPUTE,
                    args={"global_step": global_step, "epoch_step": step},
                ),
            ):
                raw_rollout_batch = self.actor.prepare_batch(
                    self.train_dataloader,
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    should_accept_fn=dynamic_filter_fn,
                    group_size=self._controller_dispatch_group_size(),
                    dynamic_bs=self.config.dynamic_bs,
                )
            accepted_workload = None
            try:
                accepted_workload = _extract_accepted_batch_workload(raw_rollout_batch)
            except Exception:
                # Workload reporting is observability-only. Never discard an
                # otherwise valid policy batch because telemetry was malformed.
                logger.warning("Unable to extract accepted-batch workload telemetry", exc_info=True)
            else:
                if accepted_workload is not None:
                    stats_tracker.scalar(
                        **accepted_workload.workload.to_metrics("workload/batch"),
                        **{
                            "workload/batch/total_tasks": float(accepted_workload.tasks),
                            "workload/batch/total_requested_rollouts": float(
                                accepted_workload.requested_rollouts
                            ),
                            "workload/batch/total_observed_rollouts": float(
                                accepted_workload.observed_rollouts
                            ),
                            "workload/batch/total_trainable_rollouts": float(
                                accepted_workload.trainable_rollouts
                            ),
                            "workload/batch/total_task_retained_datums": float(
                                accepted_workload.task_retained_datums
                            ),
                            "workload/batch/total_task_workflow_trainable_datums": float(
                                accepted_workload.task_retained_datums
                            ),
                            "workload/batch/total_task_workflow_non_trainable_datums": float(
                                accepted_workload.workload.postmerge_datums
                                - accepted_workload.task_retained_datums
                            ),
                        },
                    )
            if self._should_offload_rollout:
                self._offload_rollout()

            rollout_batch = self._postprocess_rollout_batch(
                raw_rollout_batch,
                global_step=global_step,
                epoch=epoch,
                epoch_step=step,
            )
            if rollout_batch is not None:
                compute_batch, router_replay_sidechannels = _detach_router_replay_sidechannels(rollout_batch)
            else:
                compute_batch, router_replay_sidechannels = None, None
            router_replay_marker_cleanup = None

            if compute_batch is not None and self.critic is not None:
                if self._should_offload_critic:
                    self._onload_model(self.critic, role="critic")
                with (
                    stats_tracker.record_timing("critic_values"),
                    perf_tracer.trace_scope(
                        "train.compute_values",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    values = self.critic.compute_values(compute_batch)
                    for traj, v in zip(compute_batch, values, strict=True):
                        traj["values"] = v
                    self.critic.get_device_stats().log("critic values")
                if self._should_offload_critic:
                    self._offload_model(self.critic, role="critic")

            if compute_batch is not None and self.ref is not None:
                if self._should_offload_ref:
                    self._onload_model(self.ref, role="ref")
                with (
                    stats_tracker.record_timing("ref_logp"),
                    perf_tracer.trace_scope(
                        "train.ref_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    ref_logps = self.ref.compute_logp(compute_batch)
                    for traj, logp in zip(compute_batch, ref_logps, strict=True):
                        traj["ref_logp"] = logp
                    self.ref.get_device_stats().log("ref logp")
                self._maybe_clear_device_cache(self.ref)
                if self._should_offload_ref:
                    self._offload_model(self.ref, role="ref")

            if compute_batch is not None and self.teacher is not None:
                if self._should_offload_teacher:
                    self._onload_model(self.teacher, role="teacher")
                with (
                    stats_tracker.record_timing("teacher_logp"),
                    perf_tracer.trace_scope(
                        "train.teacher_logp",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    teacher_logps = self.teacher.compute_logp(compute_batch)
                    for traj, logp in zip(compute_batch, teacher_logps, strict=True):
                        traj["teacher_logp"] = logp
                        traj["rl_loss_weight"] = self.config.teacher.rl_loss_weight
                        traj["distill_loss_weight"] = self.config.teacher.distill_loss_weight
                    self.teacher.get_device_stats().log("teacher logp")
                if self._should_offload_teacher:
                    self._offload_model(self.teacher, role="teacher")

            # Zero-reward candidates were already filtered before ref/prox and
            # advantage computation, so the actor result is the optimizer batch.
            adv_batch = None
            optimizer_batch = None
            if compute_batch is not None:
                if self._should_offload_actor:
                    self._onload_model(self.actor, role="actor")
                if config.actor.should_compute_prox_logp():
                    with (
                        stats_tracker.record_timing("recompute_logp"),
                        perf_tracer.trace_scope(
                            "train.recompute_logp",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        prox_logps = self.actor.compute_logp(compute_batch)
                        for traj, logp in zip(compute_batch, prox_logps, strict=True):
                            traj["prox_logp"] = logp
                        self.actor.get_device_stats().log("recompute logp")
                    self._maybe_clear_device_cache(self.actor)

                with (
                    stats_tracker.record_timing("compute_advantage"),
                    perf_tracer.trace_scope(
                        "train.compute_advantage",
                        category=Category.COMPUTE,
                        args={"global_step": global_step},
                    ),
                ):
                    advantage_input = _add_router_replay_order_markers(
                        compute_batch,
                        router_replay_sidechannels,
                    )
                    adv_batch = self.actor.compute_advantages(advantage_input)
                    adv_batch, router_replay_marker_cleanup = _reattach_router_replay_sidechannels(
                        adv_batch,
                        router_replay_sidechannels,
                    )
                    self.actor.get_device_stats().log("compute advantages")

                optimizer_batch = adv_batch

                try:
                    training_workload_metrics = _training_batch_workload_metrics(
                        optimizer_batch,
                        total_postmerge_datums=(
                            accepted_workload.workload.postmerge_datums
                            if accepted_workload is not None
                            else None
                        ),
                    )
                except Exception:
                    # This metric must never turn an otherwise valid update into
                    # a failed step. Canonical batches should always have both
                    # masks; custom actor paths may not.
                    logger.warning("Unable to compute submitted training-batch telemetry", exc_info=True)
                else:
                    stats_tracker.scalar(**training_workload_metrics)

                if optimizer_batch:
                    self.saver.maybe_wait_for_staging()

                    with (
                        stats_tracker.record_timing("train_step"),
                        perf_tracer.trace_scope(
                            "train.ppo_update",
                            category=Category.COMPUTE,
                            args={"global_step": global_step},
                        ),
                    ):
                        self.actor.ppo_update(optimizer_batch)
                        self.actor.step_lr_scheduler()
                        self.actor.get_device_stats().log("ppo update")
                    # Free the training-peak allocator cache before the NCCL-heavy
                    # weight-update broadcast and checkpoint phases below.
                    self._maybe_clear_device_cache(self.actor)
                    if self._should_offload_actor:
                        self._offload_model(self.actor, role="actor")

                    if self.critic is not None:
                        if self._should_offload_critic:
                            self._onload_model(self.critic, role="critic")
                        with (
                            stats_tracker.record_timing("critic_train_step"),
                            perf_tracer.trace_scope(
                                "train.critic_ppo_update",
                                category=Category.COMPUTE,
                                args={"global_step": global_step},
                            ),
                        ):
                            critic_adv_batch, _ = _detach_router_replay_sidechannels(optimizer_batch)
                            self.critic.ppo_update(critic_adv_batch)
                            self.critic.step_lr_scheduler()
                            self.critic.get_device_stats().log("ppo critic update")
                        if self._should_offload_critic:
                            self._offload_model(self.critic, role="critic")

                    self.rollout.pause()

                    with (
                        stats_tracker.record_timing("update_weights"),
                        perf_tracer.trace_scope(
                            "train.update_weights",
                            category=Category.COMM,
                            args={"global_step": global_step},
                        ),
                    ):
                        new_version = global_step + 1
                        versioned_meta = self.weight_update_meta.with_version(new_version)
                        self.actor.update_weights(versioned_meta)
                        self._advance_logical_versions(new_version)
                    # The bucketed broadcast leaves gathered full-parameter buckets
                    # in the cache; drop them before DCP save's NCCL collectives.
                    self._maybe_clear_device_cache(self.actor)
                else:
                    # Match AReaL's empty-rollout behavior: do not step the
                    # optimizer/LR or broadcast unchanged weights, but advance
                    # logical versions with the trainer's global step.
                    logger.info("Skipping optimizer update because advantage computation returned no batch")
                    if self._should_offload_actor:
                        self._offload_model(self.actor, role="actor")
                    self._advance_logical_versions(global_step + 1)
            else:
                # An accepted batch can be entirely removed by policy/sampling
                # eligibility or the exact-zero fast path. Keep the optimizer
                # payload accounting explicit in that case rather than making
                # dashboards infer a missing metric as zero.
                try:
                    empty_training_workload_metrics = _training_batch_workload_metrics(
                        [],
                        total_postmerge_datums=(
                            accepted_workload.workload.postmerge_datums
                            if accepted_workload is not None
                            else None
                        ),
                    )
                except Exception:
                    logger.warning(
                        "Unable to compute empty training-batch telemetry",
                        exc_info=True,
                    )
                else:
                    stats_tracker.scalar(**empty_training_workload_metrics)
                self._advance_logical_versions(global_step + 1)

            with (
                stats_tracker.record_timing("save"),
                perf_tracer.trace_scope(
                    "train.save",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_hf(epoch=epoch, epoch_step=step, global_step=global_step)

            with (
                stats_tracker.record_timing("checkpoint_for_recover"),
                perf_tracer.trace_scope(
                    "train.checkpoint",
                    category=Category.IO,
                    args={"global_step": global_step},
                ),
            ):
                self._save_recover_checkpoint(epoch=epoch, epoch_step=step, global_step=global_step)

            if self._should_offload_rollout:
                self._onload_rollout(is_eval=True)
            with (
                stats_tracker.record_timing("eval"),
                perf_tracer.trace_scope(
                    "train.eval",
                    category=Category.COMPUTE,
                    args={"global_step": global_step},
                ),
            ):
                self._evaluate(
                    eval_workflow=eval_workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                    epoch=epoch,
                    epoch_step=step,
                    global_step=global_step,
                )
            if self._should_offload_rollout:
                self._offload_rollout(is_eval=True)

            with (
                stats_tracker.record_timing("clear_batches"),
                perf_tracer.trace_scope(
                    "train.clear_batches",
                    category=Category.INSTR,
                    args={"global_step": global_step},
                ),
            ):
                cleanup_targets = _batch_cleanup_targets(
                    raw_rollout_batch,
                    rollout_batch,
                    adv_batch,
                    router_replay_marker_cleanup,
                )
                if cleanup_targets:
                    self.actor.clear_batches(*cleanup_targets)
                if self.data_controller is not None:
                    self.data_controller.clear_batches()

            with perf_tracer.trace_scope(
                "train.log_stats",
                category=Category.INSTR,
                args={"global_step": global_step},
            ):
                self._export_and_commit_stats(epoch=epoch, epoch_step=step, global_step=global_step)

            self.rollout.resume()
            current_platform.synchronize()
            self._save_perf_tracer(step=global_step)

    def _evaluate_fn(
        self,
        eval_workflow: WorkflowLike,
        eval_workflow_kwargs,
    ):
        if self.actor.is_data_parallel_head():
            cnt = 0
            for data in self.valid_dataloader:
                for item in data:
                    self.eval_rollout.submit(
                        item,
                        eval_workflow,
                        eval_workflow_kwargs,
                        group_size=self._controller_dispatch_group_size(),
                        is_eval=True,
                    )
                    cnt += 1
            self.eval_rollout.wait(cnt, timeout=None)

        dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
