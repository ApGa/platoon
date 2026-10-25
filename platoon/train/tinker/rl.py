"""Tinker RL Trainer with stats tracking and WandB logging."""

import asyncio
import contextlib
import logging
import os
import signal
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import tinker
import torch
from datasets import Dataset
from tinker import TensorData
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.rl.metrics import compute_kl_sample_train

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet, choose_admissions, rule_mode
from platoon.train.components import row_entry
from platoon.train.labels import dataset_labels, label_column, metric_labels, metric_prefix, task_label
from platoon.train.leases import holding
from platoon.train.sampler_runtime import SamplerSet, TaskPool, TaskSampling
from platoon.train.sampling import fingerprint_changes, planned_steps, sampling_fingerprint
from platoon.train.task_stats import DISPATCH_KEY, next_start
from platoon.train.tinker.batch_transforms import (
    LOSS_NORMALIZATION_TOKENS_KEY,
    BatchTransform,
    BatchTransformContext,
    build_default_batch_transforms,
    depth_credit_metrics,
    filter_zero_advantage_datums,
    get_datum_counts,
    get_loss_normalization_token_count,
    has_zero_objective,
    objective_is_zero_for_zero_advantages,
    run_batch_transforms,
    set_loss_normalization_token_counts,
    summarize_depths,
)
from platoon.train.tinker.censoring import Censoring
from platoon.train.tinker.checkpoint_pruning import (
    RECOVERY_CHECKPOINT_PREFIX,
    RecoveryCheckpointPruner,
    check_recovery_pruning,
    read_checkpoint_records,
)
from platoon.train.tinker.config_defs import (
    PlatoonTinkerRLTrainerConfig,
    TrainEventTriggerConfig,
    evaluates,
    event_interval,
)
from platoon.train.tinker.dataset_order import PRESERVE_DATASET_ORDER_COLUMN, SHUFFLE_SEED, EpochOrder, epoch_order
from platoon.train.tinker.dispatcher import (
    BoundDraw,
    Dispatcher,
    plan_resume,
    read_state,
    save_state,
    saved_draws,
    saved_fallback_slots,
    saved_token_positions,
)
from platoon.train.tinker.evaluation_record import EVALUATIONS_NAME, completed_evaluation_steps, record_evaluation
from platoon.train.tinker.kept_groups import decode_group, read_kept
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.permits import DrawLease, PermitPool
from platoon.train.tinker.proxy import ModelInfo, register_tinker_llm
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.tinker.rollout_metrics import step_metrics as client_step_metrics
from platoon.train.tinker.staleness import StaleWatch
from platoon.train.tinker.start_limit import StartLimit
from platoon.train.tinker.task_records import TaskRecorder, record_lost_state, record_sampling_changed, resume_record
from platoon.train.tinker.workflows.base import RolloutWorkflow
from platoon.train.tinker.workflows.group_rollout_workflow import GroupRolloutWorkflow
from platoon.utils.rollout_stats import REWARD_KEY_MISMATCH, REWARD_KEY_PREFIX, RolloutStats, outcome_counts
from platoon.utils.rollout_workload import RolloutWorkload
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.stats_tracker import get as get_tracker

logger = logging.getLogger(__name__)

# Global flag to track if we've already received a shutdown signal
_SHUTDOWN_REQUESTED = False

# How long a start asks the training service for its training client before it gives up, and the waits between two
# asks: the first, doubled after each refusal up to the longest.  A self-hosted service that trains one model at a
# time may refuse a new start until the session of a dead one expires (miles-tinker's full fine-tuning lease is ten
# minutes), and a service that restarts refuses every connection for a while.
TRAINING_CLIENT_OPEN_TIMEOUT_S = 20 * 60.0
TRAINING_CLIENT_FIRST_WAIT_S = 5.0
TRAINING_CLIENT_LONGEST_WAIT_S = 60.0
# The service's answers that asking again cannot change: a key it does not take, a permission it does not give, a
# checkpoint it does not have, a request it cannot read.
_FINAL_REFUSALS = frozenset({401, 403, 404, 422})


def _service_refusal(exc: BaseException) -> bool:
    """Whether ``exc``, raised while opening a training client, is the training service's refusal (or its silence)
    that asking again may change: any Tinker error, a lost connection or a timeout, but not a refusal in
    :data:`_FINAL_REFUSALS`.  The Tinker SDK raises a ``ValueError`` for a request whose result it could not
    retrieve, with the service's answer as its cause."""

    error: BaseException = exc
    if isinstance(error, ValueError) and isinstance(error.__cause__, tinker.TinkerError):
        error = error.__cause__
    if isinstance(error, tinker.APIStatusError):
        return error.status_code not in _FINAL_REFUSALS
    return isinstance(error, (tinker.TinkerError, ConnectionError, TimeoutError))


@dataclass(frozen=True)
class _TaskWorkloadPayload:
    """Validated workload sidechannel attached to a list-compatible result."""

    workload: RolloutWorkload
    requested_rollouts: int
    observed_rollouts: int
    trainable_rollouts: int
    task_retained_datums: int


def _get_task_workload_payload(rollout: Any) -> _TaskWorkloadPayload | None:
    """Read optional workload metadata without constraining custom workflows."""

    workload = getattr(rollout, "workload", None)
    if not isinstance(workload, RolloutWorkload):
        return None
    count_names = (
        "requested_rollouts",
        "observed_rollouts",
        "trainable_rollouts",
        "task_retained_datums",
    )
    counts = [getattr(rollout, name, None) for name in count_names]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        logger.warning("Ignoring malformed task workload rollout counts: %s", dict(zip(count_names, counts)))
        return None
    return _TaskWorkloadPayload(workload, *counts)


def _no_update_reason(zero_objective_microbatches: int, empty_microbatches: int, *, counted: int | None = None) -> str:
    """Why a step took no optimizer update, for its log line; ``counted``, of a step under a group admission rule that
    admitted no group, the groups it counted."""

    if counted is not None:
        return f"it admitted no group (every one of the {counted} groups it counted was rejected, stale or failed)"
    if zero_objective_microbatches and not empty_microbatches:
        # The usual case: every group scored alike, so every advantage is zero.  No setting forces a step then.
        return (
            "its objective is zero (every advantage is exactly zero: say every group's rollouts scored alike), and "
            "a step whose objective is zero never steps the optimizer, whatever filter_zero_advantage_datums says"
        )
    if empty_microbatches and not zero_objective_microbatches:
        return "it had no datums to train on"
    return f"{zero_objective_microbatches} microbatch(es) had a zero objective and {empty_microbatches} had no datums"


# What a label's own metrics (a step's ``batch/<column>/<label>/...``, an evaluation's ``<column>/<label>/...``) keep of
# its groups' rollout statistics, beside their group and outcome counts (platoon.train.labels): the reward, and the
# steps of each trajectory, which an entry's own step limit shapes.
LABEL_STATS = ("task_reward", "num_steps")
# The key of a full checkpoint row's loop_state that, under a group admission rule, records where the run's order
# stood (the dispatcher's next position), so that a resume that cannot read the dispatcher's state goes on from there:
# a step under a rule draws up to its cap's worth of positions, so the checkpoint's batch no longer says where.
ORDER_POSITION_KEY = "next_position"


@dataclass(frozen=True)
class _Settled:
    """A group's output as a rollout group hands it to the step that consumes it, with the environment entry and the
    label of the row it ran: the stale filter and the label metrics read them from the row, which the group holds,
    and not from the output, which a custom workflow's plain list of datums cannot carry.  ``record`` is the group's
    record in the run's per-task statistics, when the trainer keeps them, and ``key`` its draw's ``(start, seq)``,
    which the dispatcher keeps it by until a committed step consumes it.

    Under a group admission rule (RFC 0019, section 6), ``admitted`` is an admitted group's reason, and a group
    rejected or failed when it settled goes to the step as a marker: ``rejected`` its reason, ``window`` the step
    current when it settled, no ``key``, its output without its datums (None for a failed group's), and ``held``
    whether its group kept its slot of the horizon until a step dequeues it (a group that failed or carried no datum,
    :meth:`~platoon.train.tinker.dispatcher.Dispatcher.reject`)."""

    entry: str | None
    label: str | None
    rollout: Any
    record: Any = None
    key: tuple[int, int] | None = None
    admitted: str | None = None
    rejected: str | None = None
    window: int | None = None
    held: bool = False


def carries_datums(output: Any) -> bool:
    """Whether a settled group's output carries a datum to train: a list (or other sized output) with one, or an output
    whose size is unknown.  A group whose output carries none (it failed, or its verdict is ``no_results``,
    ``no_completed_root`` and the like) can have ended at once.  Never raises."""

    if output is None:
        return False
    try:
        return len(output) > 0
    except Exception:
        return True


def emptied(output: Any) -> Any:
    """A rejected group's output as its marker carries it: a list (a ``GroupRolloutWorkflow`` output, any list
    subclass) emptied of its datums in place, its other attributes kept; any other output, or a list that refuses to be
    emptied, replaced by an empty list.  Never raises."""

    if isinstance(output, list):
        try:
            del output[:]
            return output
        except Exception:
            return []
    return []


def _named_samplers(config: Any) -> list[str]:
    """The task samplers ``config`` names other than the ones a dataloader's order draws as (``auto``, ``weighted``,
    ``uniform``, and an entry's ``uniform``), one line each."""

    from platoon.train.samplers import STATIC_RUN_SAMPLERS

    named = []
    run = str(getattr(getattr(config, "train", None), "task_sampler", None) or "auto")
    if run not in STATIC_RUN_SAMPLERS:
        named.append(f"train.task_sampler: {run}")
    for environment in getattr(config, "environments", None) or []:
        own = getattr(environment, "task_sampler", None)
        if (own is not None and own != "uniform") or getattr(environment, "task_sampler_options", None):
            named.append(f"environments[{environment.label}].task_sampler: {own or 'uniform'}")
    return named


def _output_label(rollout: Any) -> str | None:
    """The label a group's output carries (``GroupRolloutWorkflow``'s ``label``), if any."""

    return getattr(rollout, "label", None)


def _settled(item: Any) -> _Settled:
    """A result queue item as a worker's :class:`_Settled`: itself, or a bare output (a test's or a custom loop's),
    whose entry and label are its own (``GroupRolloutWorkflow``'s ``entry`` and ``label``), when it has them."""

    if isinstance(item, _Settled):
        return item
    return _Settled(getattr(item, "entry", None), getattr(item, "label", None), item)


@dataclass(frozen=True)
class _Rows:
    """What the dispatcher, and an evaluation's rollout workers, read from each row they run: the row's environment
    entry (one of ``entries``), its label (its ``column`` value) and its entry's group size (``group_sizes``, else
    ``group_size``), whose rollout permits (``permits``) the group reserves before it starts."""

    entries: tuple[str, ...]
    column: str
    group_sizes: Mapping[str, int]
    group_size: int
    permits: PermitPool

    def of(self, data: Mapping[str, Any]) -> tuple[str | None, str | None, int]:
        entry = row_entry(data, self.entries)
        return entry, task_label(data, self.column), self.group_sizes.get(entry, self.group_size)


def _group_sizes(config: Any, phase: str) -> tuple[dict[str, int], int]:
    """The group size of each environment entry of a phase (``train`` or ``eval``) of a run that sets its own, by
    label, and the run's."""

    settings = entry_settings.run_settings(config)
    run = getattr(getattr(config, phase, None), "workflow_config", None)
    sizes = {
        label: getattr(own, phase).group_size for label, own in settings.items() if getattr(own, phase) is not None
    }
    return sizes, int(getattr(run, "group_size", 1))


def _groups_at_once(config: Any, phase: str, budget: int | None) -> tuple[int, int]:
    """The budget of rollouts of a phase of a run (``max_concurrent_rollouts``, else its group size) and how many of
    its groups run at once at most: as many as it holds groups of the smallest entry's size, each taking its group's
    size in permits, so the budget is never exceeded."""

    sizes, group_size = _group_sizes(config, phase)
    budget = budget if budget is not None else group_size
    return budget, max(1, budget // max(1, min([*sizes.values(), group_size])))


def _rows(config: Any, phase: str, budget: int | None) -> tuple[_Rows, int]:
    """The rows of a phase (``train`` or ``eval``) of a run, with the rollout permits of its budget, and how many of
    its groups run at once at most (:func:`_groups_at_once`): an evaluation runs as many rollout workers."""

    sizes, group_size = _group_sizes(config, phase)
    budget, groups = _groups_at_once(config, phase, budget)
    rows = _Rows(
        entries=tuple(entry_settings.run_settings(config)),
        column=label_column(config),
        group_sizes=sizes,
        group_size=group_size,
        permits=PermitPool(budget),
    )
    return rows, groups


def _rollouts_at_once(config: Any, phase: str, budget: int | None) -> float:
    """The most rollouts of a phase of a run that run at once, which the client metrics compare the rollouts in flight
    with: its budget, or as many groups of the largest entry's size as it holds groups of the smallest's, when that
    is fewer; the budget itself under ``train.permit_release: rollout``, whose rollouts return their permits as they
    end."""

    sizes, group_size = _group_sizes(config, phase)
    budget, groups = _groups_at_once(config, phase, budget)
    if phase == "train" and getattr(config.train, "permit_release", "group") == "rollout":
        return float(budget)
    return float(min(budget, groups * max([*sizes.values(), group_size])))


def _limit_starts(config: Any, train_workflow: Any, eval_workflow: Any | None) -> None:
    """Set this start's limit on the rollouts starting at once (``train.max_starting_rollouts``; None when it is off)
    as the ``start_limit`` of the training workflow and of ``eval_workflow``, the evaluation's that runs during
    training, where each is a ``GroupRolloutWorkflow``, so that both share it, first come, first served.  Off, their
    ``start_limit`` is None, so that a workflow an earlier start used keeps no limit."""

    limit = getattr(config.train, "max_starting_rollouts", None)
    gate = StartLimit(limit) if limit is not None else None
    for workflow in (train_workflow, eval_workflow):
        if isinstance(workflow, GroupRolloutWorkflow):
            workflow.start_limit = gate
    if gate is None:
        return
    phases = ("train",) if eval_workflow is None else ("train", "eval")
    largest = 0
    for phase in phases:
        sizes, size = _group_sizes(config, phase)
        largest = max(largest, size, *sizes.values())
    gate.announce(
        largest_group=largest,
        rollouts_at_once=sum(getattr(config, phase).max_concurrent_rollouts for phase in phases),
        gated=isinstance(train_workflow, GroupRolloutWorkflow),
    )


def _run_steps(config: Any, batches_per_epoch: int) -> int:
    """The training steps a run plans, with ``batches_per_epoch`` batches an epoch (:func:`planned_steps`)."""

    steps = planned_steps(config.train, batches_per_epoch)
    if steps is None:
        raise ValueError("Must specify at least one of num_epochs or max_training_steps")
    return steps


def _step_datums(rollout: Sequence[tinker.Datum]) -> list[tinker.Datum]:
    """A group's datums as a step trains them: each its own datum, with its own loss inputs, so that the step's batch
    transforms, which set loss inputs in place, leave the group as it settled; a checkpoint that keeps a group a dropped
    step had dequeued then keeps it as it settled, and the step after the resume transforms it once."""

    return [tinker.Datum(model_input=datum.model_input, loss_fn_inputs=dict(datum.loss_fn_inputs)) for datum in rollout]


def _group_done(rollout: Any, elapsed: float, task_id: Any, tracker: StatsTracker, watchdog: Any) -> None:
    """What a group that ended adds to the statistics: its time, with the group's own statistics when it has them (the
    step that consumes it records them, :func:`_record_group_stats`), else in ``tracker`` at once (a failed group, or
    a workflow's plain list); a log line for a slow one; and a watchdog heartbeat."""

    group_stats = getattr(rollout, "group_stats", None)
    if isinstance(group_stats, DeferredStats):
        group_stats.scalar(**{"rollout_completion/rollout_time": elapsed})
    else:
        tracker.scalar(**{"rollout_completion/rollout_time": elapsed})
    rollout_stats = getattr(rollout, "rollout_stats", None)
    if isinstance(rollout_stats, RolloutStats):
        rollout_stats.add("rollout_time", [elapsed])
    if elapsed > 120.0:
        logger.warning(f"Rollout for task {task_id} took {elapsed:.1f}s (very slow)")
    elif elapsed > 60.0:
        logger.info(f"Rollout for task {task_id} took {elapsed:.1f}s (slow)")
    if watchdog:
        watchdog.heartbeat()


def _no_groups_yet() -> RolloutStats:
    """A step's statistics, or a label's, before any group: no groups, no rollout of any outcome status and no group
    whose rollouts trained different reward keys, so that each count is logged every step, also at zero."""

    return RolloutStats(counts={**outcome_counts(()), REWARD_KEY_MISMATCH: 0})


def _count_every_reward_key(seen: set[str], stats: Iterable[RolloutStats]) -> None:
    """Add the ``reward_key/<key>`` counts of ``stats`` to ``seen``, the reward keys trained so far, and give each of
    ``stats`` every key seen, at zero when it trained none: each key is logged every step once it has trained."""

    stats = list(stats)
    seen.update(name for item in stats for name in item.counts if name.startswith(REWARD_KEY_PREFIX))
    for item in stats:
        for name in seen:
            item.counts.setdefault(name, 0)


def _record_group_stats(rollout: Any, tracker: StatsTracker) -> None:
    """Record what a group's workflow measured about the group (its ``group_stats``) in the consuming step.

    ``GroupRolloutWorkflow`` keeps a group's completion statistics, credit and workload distributions with the
    group instead of recording them when it completes: with prefetching, groups of the next step complete during
    this one, and their statistics would land in this step (a step counting 20 rollouts against the 16 it trained).
    """

    group_stats = getattr(rollout, "group_stats", None)
    if isinstance(group_stats, DeferredStats):
        group_stats.record(tracker)


def _ptc_cell_guard_stats(stats_scope: str = "train") -> dict[str, float]:
    """Counters for programmatic-tool-calling cells that have not returned.

    A PTC cell runs in-process on a rollout worker thread with no timeout, and a
    Python thread cannot be killed, so a never-terminating cell consumes a
    worker thread for the rest of the run without raising anything. The guard
    that counts them ships with the extension that owns programmatic tool
    calling and reports through the ``rollout_worker.telemetry`` hook; core does
    not depend on it, so a run without that extension -- or any failure inside
    the guard -- contributes no scalars instead of failing. Never raises: this
    is called inline in the train loop's stats path.

    With ``rollout_isolation: process`` the guard's registry lives in the
    *workers*, so this process's own snapshot is permanently empty.  Because
    this function swallows everything and returns ``{}``, that regression would
    be completely silent under unchanged key names -- so worker-reported
    counters take precedence when there are any.  They are deliberately the
    cumulative ones plus ``*_at_exit`` gauges: summing the instantaneous
    ``ptc_cells_stuck`` across short-lived workers would read zero precisely
    when cells are being killed for hanging.  The instantaneous names keep
    their original meaning in ``inprocess`` mode.
    """

    try:
        from platoon.train.tinker.workers.telemetry import worker_cell_guard_stats

        forwarded = worker_cell_guard_stats(stats_scope)
    except Exception:
        forwarded = {}
    if forwarded:
        return forwarded
    from platoon.train.tinker.workers.telemetry import local_telemetry

    return local_telemetry()


def compute_training_metrics(
    datums: list[tinker.Datum],
    forward_backward_result: tinker.ForwardBackwardOutput,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
) -> dict[str, float]:
    """Compute training metrics from forward_backward results.

    Extracts training logprobs from the forward pass and computes:
    - KL divergence between sampling and training policies
    - Entropy (approximated from sampling logprobs)
    - Importance weight statistics
    - Clipping statistics (for cispo/ppo loss functions)

    Args:
        datums: Original datums with mask and sampling logprobs in loss_fn_inputs
        forward_backward_result: Result from forward_backward containing training logprobs
        loss_fn: Loss function type ('cispo', 'ppo', 'importance_sampling', etc.)
        loss_fn_config: Loss function config with clip thresholds

    Returns:
        Dictionary of training metrics
    """
    # Extract training logprobs from forward_backward result
    training_logprobs_list: list[torch.Tensor] = []
    for output in forward_backward_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_list.append(training_logprobs)

    # Use tinker_cookbook's function for KL and entropy
    kl_metrics = compute_kl_sample_train(datums, training_logprobs_list)

    # Compute importance weight statistics
    all_importance_weights: list[torch.Tensor] = []
    all_clipped_low: list[torch.Tensor] = []
    all_clipped_high: list[torch.Tensor] = []

    for datum, training_logprobs in zip(datums, training_logprobs_list):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0

        # Extract action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            # Importance weight = exp(new_logprob - old_logprob) = π_new / π_old
            log_ratio = training_logprobs_actions - sampling_logprobs_actions
            importance_weights = torch.exp(log_ratio)
            all_importance_weights.append(importance_weights)

            # Compute clipping stats for cispo/ppo (both use clip_low_threshold/clip_high_threshold)
            if loss_fn in ("cispo", "ppo"):
                clip_low = loss_fn_config.get("clip_low_threshold", 0.0)
                clip_high = loss_fn_config.get("clip_high_threshold", 5.0)

                clipped_low = importance_weights < clip_low
                clipped_high = importance_weights > clip_high
                all_clipped_low.append(clipped_low)
                all_clipped_high.append(clipped_high)

    metrics = dict(kl_metrics)

    if all_importance_weights:
        flat_weights = torch.cat(all_importance_weights)
        metrics["optim/importance_weight_mean"] = flat_weights.mean().item()
        metrics["optim/importance_weight_std"] = flat_weights.std().item()
        metrics["optim/importance_weight_min"] = flat_weights.min().item()
        metrics["optim/importance_weight_max"] = flat_weights.max().item()

        # Clipping stats for cispo/ppo
        if loss_fn in ("cispo", "ppo") and all_clipped_low:
            flat_clipped_low = torch.cat(all_clipped_low)
            flat_clipped_high = torch.cat(all_clipped_high)
            total_tokens = len(flat_clipped_low)
            metrics["optim/clip_frac_low"] = flat_clipped_low.sum().item() / total_tokens
            metrics["optim/clip_frac_high"] = flat_clipped_high.sum().item() / total_tokens
            metrics["optim/clip_frac_total"] = (flat_clipped_low | flat_clipped_high).sum().item() / total_tokens

    return metrics


# Upper bound for one save_checkpoint_async (full-parameter save_state of a
# 35B model is ~490 GB and takes minutes at 1-2 GB/s).
SAVE_CHECKPOINT_TIMEOUT_S = 7200.0

# Watchdog heartbeat cadence while a rollout group is in flight (rollouts can
# outlast the watchdog timeout: rollout timeout + straggler grace + judge tail).
ROLLOUT_HEARTBEAT_INTERVAL_S = 60.0


def _resume_policy_counters(resume_info, num_minibatches: int) -> tuple[int, int]:
    """The ``policy_version`` and ``optimizer_steps`` a checkpoint saved.  A checkpoint without them comes from a
    trainer that published once per consumed batch: one policy version and ``num_minibatches`` optimizer steps."""

    batch = int(resume_info.batch or 0) if resume_info is not None else 0
    extra = resume_info.extra if resume_info is not None else {}
    policy_version = extra.get("policy_version", batch)
    optimizer_steps = extra.get("optimizer_steps", batch * num_minibatches)
    for name, value in (("policy_version", policy_version), ("optimizer_steps", optimizer_steps)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Checkpoint {name} must be a non-negative integer")
    return policy_version, optimizer_steps


class Watchdog:
    """Background thread that monitors for hangs and forcibly exits if no activity.
     If Tinker hangs, the only recovery is to forcibly exit the process.

    Usage:
        watchdog = Watchdog(timeout_seconds=1800)  # 30 min timeout
        watchdog.start()

        # In your main loop:
        watchdog.heartbeat()  # Call periodically to reset the timer

        # When done:
        watchdog.stop()
    """

    def __init__(
        self,
        timeout_seconds: float = 1800,  # 30 minutes default
        exit_code: int = 2,
        on_timeout_callback: Callable[[], None] | None = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.exit_code = exit_code
        self.on_timeout_callback = on_timeout_callback
        self._last_heartbeat = time.time()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def heartbeat(self):
        """Call this periodically to indicate the process is making progress."""
        with self._lock:
            self._last_heartbeat = time.time()

    def start(self):
        """Start the watchdog monitoring thread."""
        if self._running:
            return

        self._running = True
        self._last_heartbeat = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Watchdog started with {self.timeout_seconds}s timeout")

    def stop(self):
        """Stop the watchdog thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Watchdog stopped")

    def _monitor_loop(self):
        """Background thread that checks for hangs."""
        check_interval = min(60.0, self.timeout_seconds / 10)  # Check every minute or less

        while self._running:
            time.sleep(check_interval)

            with self._lock:
                elapsed = time.time() - self._last_heartbeat

            if elapsed > self.timeout_seconds:
                logger.error(
                    f"WATCHDOG: No heartbeat for {elapsed:.0f}s (timeout={self.timeout_seconds}s). "
                    f"Process appears hung. Forcing exit with code {self.exit_code}."
                )

                # Call optional callback (e.g., to save state)
                if self.on_timeout_callback:
                    try:
                        self.on_timeout_callback()
                    except Exception as e:
                        logger.error(f"Watchdog callback failed: {e}")

                # Force immediate exit - this bypasses all cleanup, so tell the rollout workers to stop first
                _stop_rollout_workers()
                os._exit(self.exit_code)

            elif elapsed > self.timeout_seconds * 0.75:
                # Warn when approaching timeout
                logger.warning(
                    f"WATCHDOG: No heartbeat for {elapsed:.0f}s (timeout in {self.timeout_seconds - elapsed:.0f}s)"
                )


def _stop_rollout_workers() -> None:
    """SIGTERM this trainer's live rollout workers before an exit that runs no cleanup; never raises.

    Workers lead their own sessions, so nothing else in the trainer's exit reaches them but PDEATHSIG; on SIGTERM
    each deletes its env sessions and exits within its termination deadline, and ``platoon train`` kills what is
    left before it starts training again (:mod:`platoon.train.reaping`).
    """

    try:
        from platoon.train.tinker.workers.supervisor import signal_live_workers

        stopped = signal_live_workers()
    except Exception:
        return
    if stopped:
        logger.warning(f"Sent SIGTERM to {stopped} rollout worker(s) before exiting")


class TerminateTrainLoop(Exception):
    """Exception raised to terminate the train loop."""


class TerminateEvalLoop(Exception):
    """Exception raised to terminate the eval loop for a single evaluation run."""


class TrainInterrupted(Exception):
    """A shutdown signal stopped the train loop where the last completed step is a consistent checkpoint."""


class PlatoonTinkerDataloader:
    """Batches of task rows, epoch after epoch.

    Batch ``b`` is the ``b % len(self)``-th batch of epoch ``b // len(self)`` of the dataset's order
    (:mod:`platoon.train.tinker.dataset_order`): ``order`` when given, such as the weighted order of a mixture's tasks,
    which ``platoon train`` hands the trainer; else the dataset's rows in seeded shuffled passes (``shuffle_seed``),
    or as they are.  With ``drop_last`` every batch is whole: an epoch holds the whole batches its rows fill, and a
    shuffled dataset's last partial batch starts the next epoch instead of being left out of every one.  Without it,
    an epoch is every row once, and its last batch may be partial.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle_seed: int | None = SHUFFLE_SEED,
        drop_last: bool = True,
        order: EpochOrder | None = None,
    ):
        if order is None:
            if drop_last and len(dataset) < batch_size:
                raise ValueError(
                    f"batch_size {batch_size} is more than the dataset's {len(dataset)} rows, so no full batch exists"
                )
            epoch_size = len(dataset) // batch_size * batch_size if drop_last else len(dataset)
            order = epoch_order(dataset, shuffle_seed=shuffle_seed, epoch_size=epoch_size)
        elif order.epoch_size < 1 or (drop_last and order.epoch_size % batch_size):
            raise ValueError(f"An epoch of {order.epoch_size} rows is not a whole number of batches of {batch_size}")
        self._order = order
        self._batch_size = batch_size
        self._batches_per_epoch = -(-order.epoch_size // batch_size)
        self._epoch: tuple[int, Sequence[Mapping[str, Any]]] | None = None

    def get_batch(self, batch_num: int) -> list[dict]:
        epoch, index = divmod(batch_num, self._batches_per_epoch)
        if self._epoch is None or self._epoch[0] != epoch:
            self._epoch = (epoch, self._order.epoch(epoch))
        rows = self._epoch[1][index * self._batch_size : (index + 1) * self._batch_size]
        return [dict(row) for row in rows]

    def __len__(self) -> int:
        return self._batches_per_epoch


@dataclass
class TrainLoopSharedState:
    config: PlatoonTinkerRLTrainerConfig
    shutdown_event: asyncio.Event
    train_step: int
    train_dataloader: PlatoonTinkerDataloader
    eval_dataloader: PlatoonTinkerDataloader | None
    optimizer_params: tinker.AdamParams
    service_client: tinker.ServiceClient
    training_client: tinker.TrainingClient
    sampling_client: tinker.SamplingClient
    sampling_client_step: int
    sampling_client_updated_event: asyncio.Event
    model_info: ModelInfo
    stats_logger: StatsLogger
    train_tracker: StatsTracker
    watchdog: Watchdog | None = None
    optimizer_steps: int = 0
    # Whether an evaluation loop runs beside the training loop.  The evaluation loop records each training step it
    # has handled (evaluated, when an evaluation was due) and sets ``eval_progress_event``, so that the training loop
    # can let it handle the last step before the loops stop.
    eval_loop_running: bool = False
    eval_handled_step: int | None = None
    eval_progress_event: asyncio.Event = field(default_factory=asyncio.Event)
    # The step this start of the run began at (where it resumed; 0 for a fresh run), and whether an earlier start
    # completed the evaluation of that step (the run directory's evaluations.jsonl lists it), which then does not run
    # again.  ``evaluations_path`` is the record each evaluation that completes is appended to (None: none is kept).
    start_step: int = 0
    start_step_evaluated: bool = False
    evaluations_path: str | None = None

    def evaluation_due(self, step: int) -> bool:
        """Whether the evaluation loop evaluates ``step`` when training reaches it: a multiple of ``eval_every``, step
        0 only with ``eval.at_start``, and not the step this start began at when an earlier start completed its
        evaluation."""

        every = self.eval_every
        if every is None or step % every:
            return False
        if step == 0 and not getattr(self.config.eval, "at_start", False):
            return False
        return not (step == self.start_step and self.start_step_evaluated)

    def _get_event_frequency(self, config: TrainEventTriggerConfig) -> int | None:
        """Training steps between the events ``config`` triggers (``every``), or ``None`` for ``every: 0``, which
        never triggers the event; ``strategy: epoch`` counts ``every`` in passes over the training tasks."""

        if getattr(config, "strategy", None) == "epoch":
            return event_interval(config, self.num_train_batches_per_epoch)
        return event_interval(config, 1)

    @property
    def eval_every(self) -> int | None:
        return self._get_event_frequency(self.config.eval)

    @property
    def save_every(self) -> int | None:
        return self._get_event_frequency(self.config.checkpoint)

    @property
    def num_train_batches_per_epoch(self) -> int:
        return len(self.train_dataloader)


def _evaluation_plan(state: TrainLoopSharedState, *, resumed: bool, end: int) -> tuple[int, str]:
    """The log line that says when a start of the run that evaluates evaluates first, and why, with its level: a
    warning when a fresh run evaluates nothing, as its first evaluation would come after its last step, ``end``."""

    every, start = state.eval_every or 1, state.start_step
    first = start if state.evaluation_due(start) else (start // every + 1) * every
    if not resumed:
        if first == 0:
            then = f"then every {every} steps" if every <= end else f"and no other by the run's last step, {end}"
            return logging.INFO, (
                f"Evaluating the starting policy at step 0, before the first update (eval.at_start), {then}"
            )
        if first > end:
            setting = f"eval.every is {every}"
            if getattr(state.config.eval, "strategy", None) == "epoch":  # every counts epochs: say both
                setting = f"eval.every is {state.config.eval.every} epoch(s) (strategy: epoch), {every} steps"
            return logging.WARNING, (
                f"No evaluation runs: {setting}, past the run's last step, {end}; eval.at_start: true would evaluate "
                "the starting policy at step 0"
            )
        return logging.INFO, (
            f"Evaluating every {every} steps, the first at step {every}; eval.at_start: true would evaluate the "
            "starting policy at step 0 too"
        )
    if first == start:
        return logging.INFO, (
            f"Resuming at step {start}, whose evaluation is due and no earlier start completed ({EVALUATIONS_NAME} "
            "lists none): evaluating it now"
        )
    if start % every == 0 and state.start_step_evaluated:
        then = f"the next is at step {first}" if first <= end else f"no other is due by the run's last step, {end}"
        return logging.INFO, (
            f"Resuming at step {start}, whose evaluation an earlier start completed ({EVALUATIONS_NAME}): it does not "
            f"run again, and {then}"
        )
    if first > end:
        return logging.INFO, f"Resuming at step {start}: no evaluation is due by the run's last step, {end}"
    return logging.INFO, f"Resuming at step {start}: the next evaluation is at step {first}"


class PlatoonTinkerRLTrainer:
    # Deletes the run's older recovery checkpoints while train() runs (checkpoint.keep_recovery); None without it.
    _recovery_pruner: RecoveryCheckpointPruner | None = None
    # Starts the training groups of the start train() runs (platoon.train.tinker.dispatcher); None before it.
    _dispatcher: Dispatcher | None = None
    # The order training draws its tasks in, epoch by epoch, such as a mixture's weighted order; None: the dataloader
    # shuffles train_dataset (platoon.train.tinker.dataset_order).  The trainer counts its epochs by it.
    train_order: EpochOrder | None = None
    # How the run draws its training tasks through task samplers, as platoon train hands it the run's
    # (AutoDataset.training); None: each position of the dataloader's order, as it is.
    train_sampling: TaskSampling | None = None

    def __init__(
        self,
        config: PlatoonTinkerRLTrainerConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        batch_transforms: list[BatchTransform] | None = None,
        train_order: EpochOrder | None = None,
        train_sampling: TaskSampling | None = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_order = train_order
        self.train_sampling = train_sampling
        self.batch_transforms = self._build_batch_transforms(batch_transforms)
        self._model_info = register_tinker_llm(
            self.config.train.model_name,
            self.config.train.renderer_name,
            context_window_length=self.config.train.context_window_length,
            renderer_kwargs=self.config.train.renderer_kwargs,
            base_url=self.config.tinker_base_url,  # the server train() opens its training client on
            # The bound on each sampling call, resolved here once (the key, else TINKER_SAMPLE_TIMEOUT_SECONDS), which
            # rollout workers take from the model binding.
            sample_timeout_seconds=self.config.train.sample_timeout_seconds,
        )
        self._stats_logger: StatsLogger | None = None
        self._train_tracker: StatsTracker | None = None
        self._sandbox_labels: ExitStack | None = None

    def _build_batch_transforms(
        self,
        extra_batch_transforms: list[BatchTransform] | None = None,
    ) -> list[BatchTransform]:
        """Build the ordered microbatch transform pipeline.

        Tinker trainer transforms intentionally operate at the current
        microbatch boundary (`task_rollout_results`) so the existing math is
        preserved while allowing trainer-side customization.
        """

        transforms = build_default_batch_transforms(self.config)
        if extra_batch_transforms:
            transforms.extend(extra_batch_transforms)
        return transforms

    @property
    def model_info(self) -> ModelInfo:
        """The model info containing the LLM proxy and renderer."""
        return self._model_info

    @property
    def run_log_path(self) -> str:
        """Get the full log path for this run: log_path/experiment_name/trial_name.

        This path is used for all run-specific artifacts including:
        - Checkpoints
        - Rollout results
        - WandB logs
        """
        return os.path.join(
            self.config.log_path,
            self.config.stats.experiment_name,
            self.config.stats.trial_name,
        )

    async def __aenter__(self) -> "PlatoonTinkerRLTrainer":
        """Initialize resources when entering the async context."""
        # Check for resume info to get wandb run ID
        # Use run_log_path which includes experiment_name/trial_name
        resume_info = checkpoint_utils.get_last_checkpoint(self.run_log_path)

        # Pass base log_path to StatsLogger - it will add experiment_name/trial_name internally
        stats_logger_config = self.config.stats.to_stats_logger_config(self.config.log_path)

        # If resuming, use the saved wandb run ID.
        resume_extra = resume_info.extra if resume_info is not None else {}
        wandb_run_id = resume_extra.get("wandb_run_id")
        if wandb_run_id:
            stats_logger_config.wandb.resume_run_id = wandb_run_id
            logger.info(f"Resuming WandB run: {wandb_run_id}")

        self._stats_logger = StatsLogger(stats_logger_config, exp_config=self.config)
        self._train_tracker = get_tracker("train")
        # Every sandbox the run starts, here or in a rollout worker (which carries this process's labels on), is
        # labelled with the run, so that the run's sandboxes can be reaped by it and no other run's are.
        run_label = sandbox_cleanup.run_label(self.run_log_path)
        self._sandbox_labels = ExitStack()
        self._sandbox_labels.enter_context(sandbox_cleanup.labelled(run_label))
        logger.info("This run labels its sandboxes %s", run_label)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Clean up resources when exiting the async context."""
        sandbox_labels, self._sandbox_labels = getattr(self, "_sandbox_labels", None), None
        if sandbox_labels is not None:
            sandbox_labels.close()  # this process's labels as they were before the run
        if self._stats_logger is not None:
            self._stats_logger.close()
        # Don't suppress exceptions
        return False

    async def _await_with_heartbeat(
        self,
        coro,
        name: str,
        heartbeat_interval: float = 60.0,
        watchdog: Watchdog | None = None,
        total_timeout: float | None = None,
    ):
        """Await a coroutine with periodic heartbeat logging.

        Useful for long-running async operations where we want visibility
        that we're still waiting. Also sends watchdog heartbeats to prevent
        the process from being killed during long operations.

        Args:
            coro: The coroutine to await
            name: Name for logging purposes
            heartbeat_interval: Seconds between heartbeat logs
            watchdog: Optional watchdog to send heartbeats to
            total_timeout: Optional total timeout in seconds (raises TimeoutError if exceeded)
        """
        start_time = time.perf_counter()
        task = asyncio.ensure_future(coro)

        while not task.done():
            wait = heartbeat_interval
            # Check total timeout
            if total_timeout is not None:
                elapsed = time.perf_counter() - start_time
                if elapsed >= total_timeout:
                    task.cancel()
                    raise TimeoutError(f"{name} timed out after {elapsed:.0f}s (limit: {total_timeout}s)")
                wait = min(wait, total_timeout - elapsed)  # the deadline fires on time, not at the next heartbeat

            try:
                # Wait for either the task to complete or the heartbeat interval
                await asyncio.wait_for(asyncio.shield(task), timeout=wait)
            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - start_time
                logger.info(f"Still waiting for {name}... ({elapsed:.0f}s elapsed)")
                # Send watchdog heartbeat to prevent timeout during long operations
                if watchdog is not None:
                    watchdog.heartbeat()

        return task.result()

    async def _await_rollout_with_heartbeat(
        self,
        coro,
        watchdog: Watchdog | None,
        heartbeat_interval: float | None = None,
    ):
        """Await an in-flight rollout group while feeding the watchdog.

        A group is bounded by the per-member rollout timeout plus the straggler
        grace and the judge/reward tail, which can exceed the watchdog timeout;
        without heartbeats during collection the watchdog kills the client at
        startup and after every restart before any group completes.  Unlike
        :meth:`_await_with_heartbeat` this does not shield the rollout, so a
        cancellation (interrupt checkpoint) still propagates to the group and
        its straggler members.
        """
        if heartbeat_interval is None:
            heartbeat_interval = ROLLOUT_HEARTBEAT_INTERVAL_S
        task = asyncio.ensure_future(coro)
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=heartbeat_interval)
                if watchdog is not None:
                    watchdog.heartbeat()
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        return task.result()

    async def _run_group(self, workflow: RolloutWorkflow, row: dict[str, Any], *, watchdog, tracker) -> Any:
        """Run the group of ``row`` and return its workflow's output; None for a workflow that raised, or whose own
        await was cancelled while this group was not (an inner task it awaited was cancelled): the group failed,
        which ``tracker`` records at once (``rollout_completion/failed_rollouts``), no group output carrying it to the
        step that consumes the group.  A cancellation of the group itself goes on."""

        task_id = row.get("task_id", "unknown")
        try:
            return await self._await_rollout_with_heartbeat(workflow.arun_episode(row), watchdog=watchdog)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is None or current.cancelling():
                raise
            logger.exception(f"The rollout group for task {task_id} was cancelled from within: it counts as failed")
        except Exception as e:
            logger.exception(f"Exception in rollout group for task {task_id}: {e}")
        tracker.scalar(**{"rollout_completion/failed_rollouts": 1.0})
        return None

    async def _rollout_workflow_worker_loop(
        self,
        shared_state: TrainLoopSharedState,
        workflow: RolloutWorkflow,
        task_data_queue: asyncio.Queue[dict],
        rollout_result_queue: asyncio.Queue[Any],
        *,
        tracker: StatsTracker,
        rows: _Rows | None = None,
    ):
        """An evaluation's rollout worker (training's groups start through the dispatcher): run the groups of the rows
        ``task_data_queue`` gives, one at a time, until a shutdown is requested, record their statistics in
        ``tracker``, and put each output on ``rollout_result_queue``.  With ``rows``, a group first takes its entry's
        group size in rollout permits, and its output goes with the entry and the label of the row it ran
        (:class:`_Settled`)."""

        while not shared_state.shutdown_event.is_set():
            data = await task_data_queue.get()
            entry, label, group_size = rows.of(data) if rows is not None else (None, None, 0)
            reservation = rows.permits.reserve(group_size) if rows is not None else None
            try:
                if reservation is not None:
                    await reservation.ready()
                start_time = time.perf_counter()
                rollout = await self._run_group(workflow, data, watchdog=shared_state.watchdog, tracker=tracker)
            finally:
                if reservation is not None:
                    reservation.withdraw()

            task_id = data.get("task_id", "unknown")
            _group_done(rollout, time.perf_counter() - start_time, task_id, tracker, shared_state.watchdog)
            rollout_result_queue.put_nowait(_Settled(entry, label, rollout) if rows is not None else rollout)

    async def _run_training_group(
        self,
        shared_state: TrainLoopSharedState,
        workflow: RolloutWorkflow,
        dispatcher: Dispatcher,
        draw: BoundDraw,
        lease: DrawLease,
    ) -> None:
        """Run the group of one of the dispatcher's draws, whose label slot and rollout permits are taken, with the
        draw's ``lease`` as its workflow's :func:`~platoon.train.leases.rollout_lease`: release them once its workflow
        returns (calling ``lease``), record the group in the run's per-task statistics, and hand its output to the step
        that dequeues it (:meth:`Dispatcher.settle`); a group that failed settles with no output.  Under a group
        admission rule, the group is decided first (:meth:`_settle_under_rule`)."""

        start_time = time.perf_counter()
        try:
            with holding(lease):
                rollout = await self._run_group(
                    workflow, draw.row, watchdog=shared_state.watchdog, tracker=shared_state.train_tracker
                )
        finally:
            lease()
        _group_done(
            rollout, time.perf_counter() - start_time, draw.task_id, shared_state.train_tracker, shared_state.watchdog
        )
        recorder = getattr(self, "_recorder", None)
        record = (
            recorder.settle(draw.row, entry=draw.entry, label=draw.label, output=rollout)
            if recorder is not None
            else None
        )
        if (admissions := getattr(self, "_admissions", None)) is not None:
            self._settle_under_rule(admissions, shared_state, dispatcher, draw, rollout, record)
            return
        dispatcher.settle(draw, _Settled(draw.entry, draw.label, rollout, record, draw.key))

    def _settle_under_rule(
        self,
        admissions: AdmissionSet,
        shared_state: TrainLoopSharedState,
        dispatcher: Dispatcher,
        draw: BoundDraw,
        rollout: Any,
        record: Any,
    ) -> None:
        """Decide a settled group by the run's group admission rule (RFC 0019, section 6): an admitted group settles
        as any group does, with its reason; a rejected or failed one gets its fate at once (with the rule and its
        reason), leaves the outstanding draws, and only its marker goes to the result queue (:meth:`Dispatcher.reject`).
        A rejected group with datums frees its slot at once; one that failed or carried no datum, which can have ended
        at once, keeps it until a step dequeues its marker.  Synchronous, and never raises: nothing runs between the
        fate and the reject, so a checkpoint finds the statistics' outstanding draws and the dispatcher's alike."""

        decision, rule = admissions.at_settle(draw, rollout, record, now=shared_state.model_info.llm.version)
        if decision.admit:
            dispatcher.settle(
                draw, _Settled(draw.entry, draw.label, rollout, record, draw.key, admitted=decision.reason)
            )
            return
        held = not carries_datums(rollout)  # before its datums are dropped
        marker = _Settled(
            draw.entry,
            draw.label,
            emptied(rollout) if rollout is not None else None,
            record,
            None,
            rejected=decision.reason,
            window=dispatcher.step,
            held=held,
        )
        fate = "failed" if rollout is None else "rejected"
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and record is not None:
            recorder.fate(record, fate, rule=rule, admission=decision.reason)
        elif recorder is not None:
            recorder.fate_of(draw.key, fate, rule=rule, admission=decision.reason)
        dispatcher.reject(draw, marker, hold=held)

    @staticmethod
    def _publish_worker_binding(workflow, sampling_client, version: int, *, required: bool = False) -> None:
        """Republish the sampling client for rollout workers.

        A no-op for ``inprocess`` mode and for custom workflows that do not
        run workers.  A *re*-publication failure is logged and swallowed: every
        worker keeps the client it already holds, and under the full-finetune
        gateway that client samples the newest weights anyway.

        The FIRST publication is different, and only under
        ``rollout_isolation: process``.  Each worker builds its sampling client
        by reading this file at startup and dies with ``FileNotFoundError``
        when it is missing, so swallowing that failure would leave every
        rollout of the run crashing and the trainer producing no training data
        at all -- for as long as the job is allowed to run.  That one is fatal,
        and fatal here, before the first rollout is dispatched.
        """

        publish = getattr(workflow, "publish_model_binding", None)
        if publish is None:
            return
        try:
            publish(sampling_client, version)
        except Exception:
            logger.exception("could not publish the rollout worker sampling-client binding")
            if required and getattr(workflow, "rollout_isolation", "inprocess") == "process":
                raise

    async def _train_loop(
        self,
        task_group: asyncio.TaskGroup,
        shared_state: TrainLoopSharedState,
        start_batch: int,
        end_batch: int,
        train_workflow: RolloutWorkflow,
    ):
        column = label_column(self.config)
        # Every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
        tasks = self._training_tasks()
        group_limits = getattr(self.config.train, "rollout_group_limits", {})
        # The groups settle into the result queue, which the steps dequeue from: the dispatcher starts each group when
        # the horizon, the rollout permits and the label limits allow (platoon.train.tinker.dispatcher).
        task_rollout_result_queue = asyncio.Queue[_Settled]()
        dispatcher = self._dispatch(
            task_group, shared_state, start_batch, end_batch, train_workflow, task_rollout_result_queue
        )

        # Publish the initial sampling-client binding before any group can
        # start, then declare the concurrency ceiling the metrics compare
        # against: max_concurrent_rollouts is a budget of rollouts, of which
        # a group takes its size in permits, and the monitor warns once when
        # more rollouts run at once.
        self._publish_worker_binding(
            train_workflow,
            shared_state.sampling_client,
            shared_state.model_info.llm.version,
            required=True,
        )
        budget = self.config.train.max_concurrent_rollouts
        client_monitor("train").set_configured_concurrency(
            _rollouts_at_once(self.config, "train", budget), budget=budget
        )

        # Start training loop.
        shared_state.train_step = start_batch
        logger.info(f"Starting training from batch {start_batch} to {end_batch}")
        # Each step's metrics break down by these labels of the training tasks (train.label_by), also at zero: those of
        # every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
        labels = metric_labels(tasks, column)
        reward_keys: set[str] = set()
        # The stale filter's bound for each entry's groups: its own max_staleness, else the run's.
        recorder = getattr(self, "_recorder", None)
        prefixes = {label: metric_prefix(column, label) for label in labels}
        run_bound = self.config.train.max_staleness
        settings = entry_settings.run_settings(self.config)
        bounds = {label: own.max_staleness for label, own in settings.items()}
        skipped_seen: Counter[str | None] = Counter()  # draws a capped label's full backlog skipped, logged so far
        skipped_slots_seen = 0  # slots owed to adaptive entries that were skipped, logged so far
        # A run under a group admission rule (RFC 0019, section 6): each step trains the first batch_size admitted
        # groups it dequeues that are not stale, and closes once it has counted its cap (platoon.train.admissions).
        admissions: AdmissionSet | None = getattr(self, "_admissions", None)
        # What the straggler policy censors of the groups each step dequeues, and the warning of stale groups.
        threshold = recorder.threshold if recorder is not None else None
        censoring = Censoring(settings, self.config.train.workflow_config, threshold=threshold)
        stale_watch = StaleWatch(run_bound)

        while shared_state.train_step < end_batch:
            batch_start_time = time.perf_counter()

            assert self.config.train.batch_size % self.config.train.num_minibatches == 0, (
                f"batch_size {self.config.train.batch_size} must be divisible by "
                f"num_minibatches {self.config.train.num_minibatches}"
            )
            tasks_per_minibatch = self.config.train.batch_size // self.config.train.num_minibatches

            assert tasks_per_minibatch % self.config.train.num_microbatches == 0, (
                f"tasks_per_minibatch {tasks_per_minibatch} must be divisible by "
                f"num_microbatches {self.config.train.num_microbatches}"
            )

            tasks_per_microbatch = tasks_per_minibatch // self.config.train.num_microbatches

            total_datums = 0
            batch_workload = RolloutWorkload()
            batch_total_tasks = 0
            batch_total_tasks_with_workload_metadata = 0
            batch_total_requested_rollouts = 0
            batch_total_observed_rollouts = 0
            batch_total_workflow_trainable_rollouts = 0
            batch_total_task_retained_datums = 0
            batch_submitted_training_datums = 0
            batch_submitted_training_attention_tokens = 0
            batch_submitted_training_loss_tokens = 0
            batch_rollout_stats = _no_groups_yet()
            # Each label's share of batch_rollout_stats: its groups, rewards and outcome counts.
            batch_label_stats = {label: _no_groups_yet() for label in labels}
            batch_stale_groups = 0
            batch_label_stale: Counter[str] = Counter()
            # The step's groups in the per-task statistics, each with its record and its draw, and its fate once the
            # step commits, and the kinds of the groups it trained on, by label ("" for the whole batch).
            step_fates: list[tuple[Any, tuple[int, int] | None, str]] = []
            # The draws of the groups the step dequeued: a checkpoint keeps them until the step commits.
            step_keys: list[tuple[int, int] | None] = []
            batch_kinds: dict[str, Counter[str]] = {"": Counter(), **{label: Counter() for label in labels}}
            # Of the consumed groups a task sampler drew with a score, by the sampler's metric prefix: the sum of the
            # scores (each the chance it predicted that the group trains), the groups, and those that trained.
            batch_predictions: dict[str, list[float]] = {}
            batch_groups_without_stats = 0
            batch_optimizer_steps = 0
            # Microbatches that took no forward/backward: their objective was zero, or they had no datums.
            batch_zero_objective_microbatches = 0
            batch_empty_microbatches = 0
            # Under a group admission rule: what the step counted toward its cap and toward its batch, and whether it
            # closed on its cap.
            step_counted = step_admitted = 0
            closed = False

            # We perform num_minibatches weight updates per batch.
            for minibatch_num in range(self.config.train.num_minibatches):
                forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
                # Keep original datums (with mask and logprobs) for metrics computation
                original_datums_per_microbatch: list[list[tinker.Datum]] = []

                # Microbatches are used for gradient accumulation. While gradient accumulation
                # may be less important with tinker, this can help performance.
                # Microbatches are processed as they become available, allowing us to overlap
                # rollout sampling with training even within the same batch.
                # This is a second-level of pipelining orthogonal to the async/off-policy
                # pipelining at the batch level.
                for microbatch_num in range(self.config.train.num_microbatches):
                    if closed:  # a step under a rule that counted its cap dequeues no more
                        break
                    task_rollout_results: list[tinker.Datum] = []
                    microbatch_groups = 0  # under a rule: its groups that count toward the batch

                    task_num = 0
                    while task_num < tasks_per_microbatch and not closed:
                        # In all mode each item takes a place of the microbatch, counted first so that nothing in the
                        # body can skip it; under a rule only an admitted group that is not stale does (at the end).
                        place = task_num
                        if admissions is None:
                            task_num += 1
                        queue_wait_start = time.perf_counter()
                        logger.debug(
                            f"Waiting for rollout {place + 1}/{tasks_per_microbatch} "
                            f"(minibatch {minibatch_num}, microbatch {microbatch_num})"
                        )
                        settled = _settled(
                            await self._next_rollout(
                                task_rollout_result_queue,
                                shared_state,
                                interruptible=batch_optimizer_steps == 0,
                                ready=admissions is not None,
                            )
                        )
                        if dispatcher is not None and admissions is None:
                            dispatcher.dequeued()
                        step_keys.append(settled.key)
                        entry, row_label, rollout, record = (
                            settled.entry,
                            settled.label,
                            settled.rollout,
                            settled.record,
                        )
                        failed = rollout is None
                        stale = False
                        batch_total_tasks += 1
                        # The group's own statistics belong to this step, which consumes it, stale or not.
                        _record_group_stats(rollout, shared_state.train_tracker)
                        rollout_stats = getattr(rollout, "rollout_stats", None)
                        if not isinstance(rollout_stats, RolloutStats):
                            batch_groups_without_stats += 1
                        workload_payload = _get_task_workload_payload(rollout)
                        if workload_payload is not None:
                            # Attribute generation work when this outer batch
                            # actually consumes the queue item.  Producer-side
                            # recording would leak prefetched future tasks into
                            # the current step.  Stale work is deliberately
                            # included so wasted inference remains visible.
                            batch_total_tasks_with_workload_metadata += 1
                            batch_workload += workload_payload.workload
                            batch_total_requested_rollouts += workload_payload.requested_rollouts
                            batch_total_observed_rollouts += workload_payload.observed_rollouts
                            batch_total_workflow_trainable_rollouts += workload_payload.trainable_rollouts
                            batch_total_task_retained_datums += workload_payload.task_retained_datums
                        queue_wait_elapsed = time.perf_counter() - queue_wait_start
                        # The trainer's own view of "rollout-bound": time spent
                        # blocked on the result queue during this step.
                        client_monitor("train").record_queue_wait(
                            queue_wait_elapsed, immediate=queue_wait_elapsed < 0.001
                        )
                        if queue_wait_elapsed > 60.0:
                            logger.warning(f"Waited {queue_wait_elapsed:.1f}s for rollout from queue (slow)")

                        # Filter out stale rollouts, by the bound of the group's entry. TODO: Consider requeuing.
                        max_staleness = bounds.get(entry, run_bound) if entry is not None else run_bound
                        if max_staleness is not None and rollout is not None and len(rollout) > 0:
                            # Group members can begin sampling at different
                            # versions after environment setup. Keep group
                            # advantages intact and reject by its oldest datum.
                            versions = [
                                int(version.to_torch().item())
                                for datum in rollout
                                if (version := datum.loss_fn_inputs.get("checkpoint_version")) is not None
                            ]
                            rollout_checkpoint_version = min(versions, default=shared_state.model_info.llm.version)

                            if shared_state.model_info.llm.version - rollout_checkpoint_version > max_staleness:
                                logger.info(
                                    f"Stale rollout detected in batch {shared_state.train_step}: "
                                    f"policy version {shared_state.model_info.llm.version}, "
                                    f"rollout version {rollout_checkpoint_version}, beyond the bound of "
                                    f"{max_staleness} versions"
                                    + (f" of environments[{entry}]" if entry is not None else "")
                                    + ". Filtering out."
                                )
                                shared_state.train_tracker.scalar(stale_rollouts=1.0)
                                batch_stale_groups += 1
                                batch_label_stale[row_label if row_label is not None else _output_label(rollout)] += 1
                                rollout = None
                                stale = True
                        # A marker of a group a rule rejected (or that failed) when it settled has its fate already.
                        marker = settled.rejected is not None
                        censoring.add(row_label if row_label is not None else _output_label(settled.rollout), record)
                        if not marker:
                            stale_watch.dequeued(stale)
                        if admissions is not None:
                            taken = "marker" if marker else "stale" if stale else "failed" if failed else "admitted"
                            counts = (
                                dispatcher.took(taken, settled.window, held=settled.held)
                                if dispatcher is not None
                                else True
                            )
                            step_counted += counts
                            step_admitted += taken == "admitted"
                            microbatch_groups += taken == "admitted"
                            admissions.dequeued(
                                row_label if row_label is not None else _output_label(settled.rollout),
                                taken,
                                settled,
                            )
                            if (
                                marker
                                and settled.rollout is not None
                                and record is not None
                                and record.score is not None
                                and dispatcher is not None
                                and (where := dispatcher.samplers.prefix(record.sampler)) is not None
                            ):  # a rejected group's prediction counts, as a consumed group's does (stale, failed not)
                                predicted = batch_predictions.setdefault(where, [0.0, 0.0, 0.0])
                                predicted[0] += record.score
                                predicted[1] += 1
                                predicted[2] += record.verdict == "trained"
                            if step_counted >= admissions.cap:
                                closed = True
                            elif counts:
                                admissions.waiting(
                                    shared_state.train_step, counted=step_counted, admitted=step_admitted
                                )
                        fate = "stale" if stale else "failed" if failed else "consumed"
                        if recorder is not None and not marker:
                            step_fates.append((record, settled.key, fate))
                        if recorder is not None and record is not None and fate == "consumed" and not marker:
                            if (kind := record.kind(recorder.threshold(record.entry))) is not None:
                                batch_kinds[""][kind] += 1
                                if row_label in batch_kinds:
                                    batch_kinds[row_label][kind] += 1
                            where = dispatcher.samplers.prefix(record.sampler) if dispatcher is not None else None
                            if record.score is not None and where is not None:
                                predicted = batch_predictions.setdefault(where, [0.0, 0.0, 0.0])
                                predicted[0] += record.score
                                predicted[1] += 1
                                predicted[2] += record.verdict == "trained"

                        if rollout is not None:
                            if isinstance(rollout_stats, RolloutStats):
                                batch_rollout_stats.merge(rollout_stats)
                                label = row_label if row_label is not None else _output_label(rollout)
                                if label in batch_label_stats:  # a label whose metrics are logged
                                    batch_label_stats[label].merge(rollout_stats.subset(LABEL_STATS))
                            task_rollout_results.extend(_step_datums(rollout))
                            total_datums += len(rollout)
                        if admissions is not None and not marker and not stale and not failed:
                            task_num += 1

                    if len(task_rollout_results) == 0 and admissions is not None and closed:
                        break  # a step under a rule closed on its cap with nothing more to send
                    if len(task_rollout_results) == 0:
                        logger.warning(f"No rollouts found for microbatch {microbatch_num} (minibatch {minibatch_num})")
                        batch_empty_microbatches += 1
                        continue

                    depths_before = summarize_depths(task_rollout_results)
                    task_rollout_results = run_batch_transforms(
                        task_rollout_results,
                        self.batch_transforms,
                        BatchTransformContext(
                            config=self.config,
                            train_step=shared_state.train_step,
                            minibatch_num=minibatch_num,
                            microbatch_num=microbatch_num,
                        ),
                    )
                    if depths_before is not None:
                        shared_state.train_tracker.scalar(
                            **depth_credit_metrics(depths_before, summarize_depths(task_rollout_results or []))
                        )
                    if task_rollout_results is None or len(task_rollout_results) == 0:
                        logger.warning(
                            f"All datums were filtered out for microbatch {microbatch_num} (minibatch {minibatch_num})"
                        )
                        batch_empty_microbatches += 1
                        continue

                    # Depth weighting is part of the objective, so zero-signal
                    # datums participate in that normalization before being
                    # removed from model compute. Moving this filter into the
                    # task workflow would change per-depth gradient weights.
                    loss_fn = getattr(self.config.train, "loss_fn", "cispo")
                    loss_fn_config = getattr(self.config.train, "loss_fn_config", {})
                    if (
                        self.config.train.workflow_config.filter_zero_advantage_datums
                        and objective_is_zero_for_zero_advantages(loss_fn, loss_fn_config)
                    ):
                        eligible_datums, eligible_attention_tokens, eligible_loss_tokens = get_datum_counts(
                            task_rollout_results
                        )
                        task_rollout_results = filter_zero_advantage_datums(task_rollout_results)
                        retained_datums, retained_attention_tokens, retained_loss_tokens = get_datum_counts(
                            task_rollout_results
                        )
                        shared_state.train_tracker.scalar(
                            **{
                                "zero_advantage_filter/eligible_datums": float(eligible_datums),
                                "zero_advantage_filter/retained_datums": float(retained_datums),
                                "zero_advantage_filter/filtered_datums": float(eligible_datums - retained_datums),
                                "zero_advantage_filter/eligible_attention_tokens": float(eligible_attention_tokens),
                                "zero_advantage_filter/retained_attention_tokens": float(retained_attention_tokens),
                                "zero_advantage_filter/filtered_attention_tokens": float(
                                    eligible_attention_tokens - retained_attention_tokens
                                ),
                                "zero_advantage_filter/eligible_loss_tokens": float(eligible_loss_tokens),
                                "zero_advantage_filter/retained_loss_tokens": float(retained_loss_tokens),
                                "zero_advantage_filter/filtered_loss_tokens": float(
                                    eligible_loss_tokens - retained_loss_tokens
                                ),
                            }
                        )
                        if task_rollout_results:
                            # Preserve the pre-filter action-token denominator;
                            # otherwise removing zero terms would amplify every
                            # remaining gradient in this microbatch.
                            set_loss_normalization_token_counts(
                                task_rollout_results,
                                represented_loss_tokens=eligible_loss_tokens,
                            )

                    if len(task_rollout_results) == 0:
                        logger.warning(
                            f"All datums had zero advantage for microbatch {microbatch_num} (minibatch "
                            f"{minibatch_num}): its objective is zero, so it takes no forward/backward"
                        )
                        batch_zero_objective_microbatches += 1
                        continue

                    # Kept datums or not (filter_zero_advantage_datums), a microbatch whose every advantage is
                    # exactly zero has a zero objective under these losses: it is never sent.
                    if has_zero_objective(task_rollout_results, loss_fn, loss_fn_config):
                        logger.info(
                            "Zero policy-gradient objective for microbatch %s (minibatch %s): every advantage is "
                            "exactly zero, so it takes no forward/backward and adds nothing to an optimizer step",
                            microbatch_num,
                            minibatch_num,
                        )
                        batch_zero_objective_microbatches += 1
                        continue

                    (
                        submitted_datums,
                        submitted_attention_tokens,
                        submitted_loss_tokens,
                    ) = get_datum_counts(task_rollout_results)
                    batch_submitted_training_datums += submitted_datums
                    batch_submitted_training_attention_tokens += submitted_attention_tokens
                    batch_submitted_training_loss_tokens += submitted_loss_tokens

                    # Filter out mask and checkpoint_version from loss_fn_inputs before forward_backward
                    # Neither is needed in forward_backward computation.
                    # mask is redundant since advantages are 0 for masked tokens.
                    # Keep original datums for metrics computation (they have mask and logprobs).
                    original_datums_per_microbatch.append(task_rollout_results)
                    filtered_datums = [
                        tinker.Datum(
                            model_input=datum.model_input,
                            loss_fn_inputs={
                                k: v
                                for k, v in datum.loss_fn_inputs.items()
                                if k
                                not in (
                                    "mask",
                                    "checkpoint_version",
                                    "traj_depth",
                                    "traj_start",
                                    LOSS_NORMALIZATION_TOKENS_KEY,
                                )
                            },
                        )
                        for datum in task_rollout_results
                    ]

                    # Normalize by represented action-token mass so Tinker's
                    # sum-reduced objective behaves like a mean reduction and
                    # grad_norm is not sensitive to batch size.
                    normalization_token_count = get_loss_normalization_token_count(task_rollout_results)
                    if (
                        admissions is not None
                        and closed
                        and microbatch_num > 0
                        and microbatch_groups < tasks_per_microbatch
                    ):
                        # A short step's partial last microbatch, after full ones: its groups weigh as theirs do.
                        normalization_token_count *= tasks_per_microbatch / max(1, microbatch_groups)
                    scale_factor = 1.0 / (normalization_token_count + 1e-8)

                    for datum in filtered_datums:
                        datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
                            datum.loss_fn_inputs["advantages"].to_torch() * scale_factor
                        )

                    logger.debug(
                        f"Submitting forward_backward_async with {len(filtered_datums)} datums "
                        f"(minibatch {minibatch_num}, microbatch {microbatch_num})"
                    )
                    fwd_bwd_submit_start = time.perf_counter()
                    try:
                        forward_backward_futures.append(
                            await shared_state.training_client.forward_backward_async(
                                filtered_datums,
                                loss_fn=self.config.train.loss_fn,
                                loss_fn_config=self.config.train.loss_fn_config,
                            )
                        )
                    except Exception as e:
                        elapsed = time.perf_counter() - fwd_bwd_submit_start
                        logger.exception(f"forward_backward_async submission failed after {elapsed:.1f}s: {e}")
                        raise
                    logger.debug(
                        f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s"
                    )

                if len(forward_backward_futures) == 0:
                    logger.warning(
                        f"Minibatch {minibatch_num} of batch {shared_state.train_step} takes no optimizer update: "
                        "none of its microbatches had a nonzero objective"
                    )
                    continue

                if start_batch == 0 and getattr(shared_state, "optimizer_steps", 0) == 0:
                    # The run's first update: an evaluation of its starting policy (eval.at_start) must sample that
                    # policy throughout.
                    await self._wait_for_starting_policy_evaluation(shared_state)
                optim_start = time.perf_counter()
                logger.debug("Submitting optim_step_async")
                try:
                    optim_future = await shared_state.training_client.optim_step_async(shared_state.optimizer_params)
                except Exception as e:
                    logger.exception(f"optim_step_async submission failed: {e}")
                    raise
                logger.debug(f"optim_step_async submitted in {time.perf_counter() - optim_start:.1f}s")

                # Consume all forward backward results and compute training metrics.
                all_training_metrics: list[dict[str, float]] = []
                for microbatch_num, (forward_backward_future, original_datums) in enumerate(
                    zip(forward_backward_futures, original_datums_per_microbatch)
                ):
                    result_start = time.perf_counter()
                    logger.debug(f"Waiting for forward_backward result (microbatch {microbatch_num})")
                    try:
                        # Use a wrapper to log periodic "still waiting" messages.
                        # NOTE: every microbatch is submitted before this loop and the service
                        # drains them in order, so the await on microbatch 0 spans the whole
                        # accumulated window; the deadline is sized for a full step.
                        forward_backward_result = await self._await_with_heartbeat(
                            forward_backward_future.result_async(),
                            name=f"forward_backward (microbatch {microbatch_num})",
                            heartbeat_interval=60.0,
                            watchdog=shared_state.watchdog,
                            total_timeout=self.config.train.forward_backward_timeout_seconds,
                        )
                    except Exception as e:
                        logger.exception(
                            f"forward_backward result_async failed after {time.perf_counter() - result_start:.1f}s: {e}"
                        )
                        raise
                    result_elapsed = time.perf_counter() - result_start
                    if result_elapsed > 60.0:
                        logger.warning(f"forward_backward result took {result_elapsed:.1f}s (slow)")
                    else:
                        logger.debug(f"forward_backward result received in {result_elapsed:.1f}s")

                    # Compute training metrics (KL, entropy, importance weights, clipping stats)
                    try:
                        microbatch_metrics = compute_training_metrics(
                            datums=original_datums,
                            forward_backward_result=forward_backward_result,
                            loss_fn=self.config.train.loss_fn,
                            loss_fn_config=self.config.train.loss_fn_config,
                        )
                        all_training_metrics.append(microbatch_metrics)
                    except Exception as e:
                        logger.warning(f"Failed to compute training metrics for microbatch {microbatch_num}: {e}")

                # Aggregate and log training metrics across microbatches
                if all_training_metrics:
                    aggregated_metrics: dict[str, float] = {}
                    for key in all_training_metrics[0].keys():
                        values = [m[key] for m in all_training_metrics if key in m]
                        if values:
                            aggregated_metrics[key] = sum(values) / len(values)
                    for key, value in aggregated_metrics.items():
                        shared_state.train_tracker.scalar(**{key: value})

                # Wait for optimizer step to complete.
                logger.debug("Waiting for optim_step result")
                optim_result_start = time.perf_counter()
                try:
                    # The optimizer step runs after the window is finalized and is fast.
                    optim_result = await self._await_with_heartbeat(
                        optim_future.result_async(),
                        name="optim_step",
                        heartbeat_interval=60.0,
                        watchdog=shared_state.watchdog,
                        total_timeout=self.config.train.optim_step_timeout_seconds,
                    )
                except Exception as e:
                    logger.exception(
                        f"optim_step result_async failed after {time.perf_counter() - optim_result_start:.1f}s: {e}"
                    )
                    raise
                optim_elapsed = time.perf_counter() - optim_result_start
                batch_optimizer_steps += 1
                shared_state.optimizer_steps = getattr(shared_state, "optimizer_steps", 0) + 1
                if optim_elapsed > 60.0:
                    logger.warning(f"optim_step result took {optim_elapsed:.1f}s (slow)")
                else:
                    logger.debug(f"optim_step result received in {optim_elapsed:.1f}s")

                # Extract optimizer metrics (e.g., grad_norm if grad_clip_norm is set)
                if optim_result and optim_result.metrics:
                    for key, value in optim_result.metrics.items():
                        shared_state.train_tracker.scalar(**{f"optim/{key}": value})

            exact_training_batch_metrics: dict[str, float] = {}
            # A legacy/custom plain-list workflow has no generation sidechannel.
            # Continue reporting the authoritative submitted payload, but do not
            # subtract it from a known-workload subtotal and claim a negative or
            # otherwise inexact non-submitted count.
            if batch_total_tasks_with_workload_metadata == batch_total_tasks:
                batch_non_submitted_datums = batch_workload.postmerge_datums - batch_submitted_training_datums
                if batch_non_submitted_datums < 0:
                    raise ValueError(
                        "Submitted training datums exceed the consumed post-merge datum count: "
                        f"submitted={batch_submitted_training_datums}, "
                        f"postmerge={batch_workload.postmerge_datums}"
                    )
                exact_training_batch_metrics["workload/training_batch/total_non_submitted_datums"] = float(
                    batch_non_submitted_datums
                )
            shared_state.train_tracker.scalar(
                total_datums_per_batch=total_datums,
                **batch_workload.to_metrics("workload/batch"),
                **{
                    "workload/batch/total_tasks": float(batch_total_tasks),
                    "workload/batch/total_tasks_with_workload_metadata": float(
                        batch_total_tasks_with_workload_metadata
                    ),
                    "workload/batch/total_requested_rollouts": float(batch_total_requested_rollouts),
                    "workload/batch/total_observed_rollouts": float(batch_total_observed_rollouts),
                    "workload/batch/total_workflow_trainable_rollouts": float(batch_total_workflow_trainable_rollouts),
                    "workload/batch/total_task_retained_datums": float(batch_total_task_retained_datums),
                    "workload/batch/total_task_workflow_trainable_datums": float(batch_total_task_retained_datums),
                    "workload/batch/total_task_workflow_non_trainable_datums": float(
                        batch_workload.postmerge_datums - batch_total_task_retained_datums
                    ),
                    "workload/batch/total_task_filter_dropped_datums": float(
                        batch_workload.post_sampling_datums - batch_total_task_retained_datums
                    ),
                    "workload/training_batch/total_submitted_datums": float(batch_submitted_training_datums),
                    "workload/training_batch/total_attention_tokens": float(batch_submitted_training_attention_tokens),
                    "workload/training_batch/total_action_tokens": float(batch_submitted_training_loss_tokens),
                    **exact_training_batch_metrics,
                },
            )

            if admissions is not None:  # a short step, or one that admitted nothing: warned of once each
                admissions.step_closed(shared_state.train_step, counted=step_counted, admitted=step_admitted)

            # The step commits: what became of each of its groups, before a full checkpoint saves the statistics and
            # keeps the groups that settled and no step consumed.
            if recorder is not None:
                for record, key, fate in step_fates:
                    if record is not None:
                        recorder.fate(record, fate)
                    elif key is not None:  # a group a start kept after its statistics stopped: no record of its own
                        recorder.fate_of(key, fate)
            if dispatcher is not None:
                dispatcher.committed(step_keys)

            # Consuming an all-zero batch does not change the policy. In
            # particular it must not age slower, useful rollouts as stale.
            update_start = time.perf_counter()
            if batch_optimizer_steps:
                logger.info(f"Calling _save_checkpoint_and_get_sampling_client for batch {shared_state.train_step}")
                sampling_client = await self._save_checkpoint_and_get_sampling_client(
                    training_client=shared_state.training_client,
                    i_batch=shared_state.train_step,
                    save_every=shared_state.save_every,
                    start_batch=start_batch,
                    watchdog=shared_state.watchdog,
                    recovery_every=self.config.checkpoint.recovery_every,
                    policy_version=shared_state.model_info.llm.version + 1,
                    optimizer_steps=shared_state.optimizer_steps,
                )
                shared_state.model_info.llm.update_sampling_client(sampling_client)
                self._publish_worker_binding(train_workflow, sampling_client, shared_state.model_info.llm.version)
                shared_state.sampling_client = sampling_client
                shared_state.sampling_client_step += 1
            else:
                logger.info(
                    "Batch %s (step %s) takes no optimizer update: %s; the policy stays at version %s",
                    shared_state.train_step,
                    shared_state.train_step + 1,
                    _no_update_reason(
                        batch_zero_objective_microbatches,
                        batch_empty_microbatches,
                        counted=step_counted if admissions is not None and not step_admitted else None,
                    ),
                    shared_state.model_info.llm.version,
                )
            update_elapsed = time.perf_counter() - update_start
            shared_state.train_tracker.scalar(update_weights_time=update_elapsed)

            shared_state.train_step += 1
            if dispatcher is not None:
                dispatcher.step_started()  # its policy published: the horizon moves one batch on

            # Evaluation cadence remains based on consumed batches, including
            # zero-signal ones; the sampler itself may be unchanged.
            shared_state.sampling_client_updated_event.set()

            # Heartbeat after completing a train step
            if shared_state.watchdog:
                shared_state.watchdog.heartbeat()

            # Track total batch time
            batch_elapsed = time.perf_counter() - batch_start_time

            # Log stats for this step
            _count_every_reward_key(reward_keys, [batch_rollout_stats, *batch_label_stats.values()])
            batch_rollout_stats.record(shared_state.train_tracker, prefix="batch")
            for label, stats in batch_label_stats.items():
                stats.record(shared_state.train_tracker, prefix=f"batch/{metric_prefix(column, label)}")
                shared_state.train_tracker.scalar(
                    **{f"batch/{metric_prefix(column, label)}/stale_groups": float(batch_label_stale[label])}
                )
            if recorder is not None:
                recorder.flush()  # every settled group taken in first: a store that fails to stops the statistics
            if recorder is not None and not recorder.stopped:
                # The kinds of the groups the step trained on, and what each label's tasks' groups say so far; nothing
                # once the statistics stopped, rather than zeros that read as no informative groups.
                for label, kinds in batch_kinds.items():
                    where = f"batch/{prefixes[label]}/" if label else "batch/"
                    for kind in ("informative", "all_pass", "all_fail", "uniform_mixed"):
                        shared_state.train_tracker.scalar(**{f"{where}{kind}_groups": float(kinds[kind])})
                shared_state.train_tracker.scalar(
                    **recorder.step_metrics(now=shared_state.model_info.llm.version, prefixes=prefixes)
                )
            if dispatcher is not None:
                # What the task samplers drew since the last step and why, how long their decisions took, their own
                # metrics, the slots waiting for an adaptive entry's sampler and those skipped; and, of the groups the
                # step consumed, the share each sampler predicted would train against the share that did.
                version = shared_state.model_info.llm.version
                sampler_metrics = dispatcher.samplers.step_metrics(version)
                shared_state.train_tracker.scalar(**sampler_metrics)
                for entry, count in dispatcher.tokens_by_entry().items():
                    where = dispatcher.samplers.prefix(dispatcher.samplers.entries[entry].name)
                    shared_state.train_tracker.scalar(**{f"{where}tokens": float(count)})
                skipped_slots = dispatcher.skipped_slots - skipped_slots_seen
                skipped_slots_seen = dispatcher.skipped_slots
                shared_state.train_tracker.scalar(**{"sampler/skipped_slots": float(skipped_slots)})
                for where, (scores, groups, trained) in batch_predictions.items():
                    shared_state.train_tracker.scalar(
                        **{
                            f"{where}predicted_informative": scores / groups,
                            f"{where}informative_share": trained / groups,
                        }
                    )
            if admissions is not None:
                # What the step admitted, rejected and dropped, by label and reason, and what it counted.
                draws = sampler_metrics.get("sampler/draws") if dispatcher is not None else None
                shared_state.train_tracker.scalar(
                    **admissions.step_metrics(
                        prefixes=prefixes, counted=step_counted, admitted=step_admitted, draws=draws
                    )
                )
            if group_limits and dispatcher is not None:
                # Draws a capped label's full backlog skipped since the last step: its share of the order asks for
                # more groups than its limit runs. And the draws that wait for their label's limit now (parked).
                skipped = dispatcher.skipped - skipped_seen
                skipped_seen.update(skipped)
                parked = dispatcher.parked_by_label()
                shared_state.train_tracker.scalar(
                    **{"batch/skipped_rows": float(skipped.total()), "batch/parked_rows": float(parked.total())}
                )
                for label in labels:
                    where = f"batch/{metric_prefix(column, label)}"
                    shared_state.train_tracker.scalar(
                        **{f"{where}/skipped_rows": float(skipped[label]), f"{where}/parked_rows": float(parked[label])}
                    )
            shared_state.train_tracker.scalar(
                **{
                    "batch/groups_dequeued": float(batch_total_tasks),
                    "batch/stale_groups": float(batch_stale_groups),
                    "batch/groups_without_stats": float(batch_groups_without_stats),
                    "batch/optimizer_updates": float(batch_optimizer_steps),
                    "batch/skipped_optimizer": float(batch_optimizer_steps == 0),
                    "policy/version": float(shared_state.model_info.llm.version),
                    "optim/updates_total": float(getattr(shared_state, "optimizer_steps", 0)),
                }
            )
            # What held the dispatcher back from starting groups, and what the straggler policy censored.
            blocked = dispatcher.blocked_shares() if dispatcher is not None else {}
            shared_state.train_tracker.scalar(**blocked, **censoring.step_metrics(prefixes))
            stale_watch.step_ended(blocked)
            step_stats = shared_state.train_tracker.export(reset=True)
            step_stats["train/step"] = shared_state.train_step
            step_stats["train/epoch"] = shared_state.train_step // shared_state.num_train_batches_per_epoch
            step_stats["progress/done_frac"] = shared_state.train_step / end_batch
            step_stats["optim/lr"] = self.config.train.optimizer.learning_rate
            step_stats["timing/batch_total"] = batch_elapsed
            # Non-zero ``ptc_threads_leaked`` means rollout worker threads are gone
            # for good; see ``_ptc_cell_guard_stats``.
            step_stats.update(_ptc_cell_guard_stats("train"))
            # Client-side throughput: concurrency actually achieved, the queue
            # wait the trainer experienced, where an agent step's wall clock
            # went, and whether this process (plus its rollout workers) is CPU
            # saturated.  See ``platoon.train.tinker.rollout_metrics``.
            step_stats.update(client_step_metrics("train"))
            if (pruner := getattr(self, "_recovery_pruner", None)) is not None:
                # Recovery checkpoint pruning: paths deleted and deletes failed since the last step, and what is kept.
                step_stats.update(pruner.step_metrics())
            shared_state.stats_logger.log(step=shared_state.train_step, stats=step_stats)

            logger.info(
                f"Step {shared_state.train_step}/{end_batch} | "
                f"Epoch {shared_state.train_step // shared_state.num_train_batches_per_epoch} | "
                f"Batch time: {batch_elapsed:.1f}s | "
                f"Datums: {total_datums}"
            )

        await self._wait_for_last_step_evaluation(shared_state)
        await self.shutdown_loops(shared_state)

    @staticmethod
    async def _next_rollout(
        queue: asyncio.Queue, shared_state: TrainLoopSharedState, *, interruptible: bool, ready: bool = False
    ):
        """The next rollout result; with ``interruptible`` (no update of the current batch applied yet),
        ``TrainInterrupted`` once a shutdown is requested instead, since the feeders stop on it and the batch
        may never complete.  The batch is then dropped whole: a resume from the last completed step redoes it.
        With ``ready`` (a step under a group admission rule), a result already queued is taken without letting
        other tasks run first: a step takes the queued markers in one go, so that groups that end at once cannot start
        again in the slots those markers free and queue more markers between two of them.
        """

        shutdown_event = getattr(shared_state, "shutdown_event", None)
        if ready and not queue.empty():
            if interruptible and shutdown_event is not None and shutdown_event.is_set():
                raise TrainInterrupted()
            return queue.get_nowait()
        if not interruptible or shutdown_event is None:
            return await queue.get()
        if shutdown_event.is_set():
            raise TrainInterrupted()
        get = asyncio.ensure_future(queue.get())
        stop = asyncio.ensure_future(shutdown_event.wait())
        try:
            await asyncio.wait({get, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.cancel()
            if not get.done():
                get.cancel()
        if get.done() and not get.cancelled():
            return get.result()
        raise TrainInterrupted()

    async def _eval_completion_monitor(
        self,
        total_eval_tasks: int,
        eval_rollout_result_queue: asyncio.Queue[Any],
        tracker: StatsTracker | None = None,
        label_stats: dict[str, RolloutStats] | None = None,
    ):
        """Monitor task that waits for all eval results and then terminates the eval TaskGroup; each group's rollout
        statistics go into its label's ``label_stats``, when its label's metrics are logged."""
        for _ in range(total_eval_tasks):
            settled = _settled(await eval_rollout_result_queue.get())
            row_label, rollout = settled.label, settled.rollout
            if tracker is not None:
                _record_group_stats(rollout, tracker)
            rollout_stats = getattr(rollout, "rollout_stats", None)
            label = row_label if row_label is not None else _output_label(rollout)
            if label_stats is not None and label in label_stats and isinstance(rollout_stats, RolloutStats):
                label_stats[label].merge(rollout_stats.subset(LABEL_STATS))
        raise TerminateEvalLoop()

    async def _eval_loop(self, shared_state: TrainLoopSharedState, workflow: RolloutWorkflow):
        eval_tracker = get_tracker("eval")
        # Each evaluation's metrics break down by these labels of the evaluation's tasks (train.label_by), as a
        # training step's do: eval/<column>/<label>/...
        column = label_column(self.config)
        labels = metric_labels(getattr(self, "eval_dataset", None), column, split="evaluation")
        reward_keys: set[str] = set()

        while not shared_state.shutdown_event.is_set():
            await shared_state.sampling_client_updated_event.wait()

            # Check shutdown again after waking up
            if shared_state.shutdown_event.is_set():
                break

            # Clear the event so we wait for the next update
            shared_state.sampling_client_updated_event.clear()

            step = shared_state.train_step
            # Every eval_every steps: step 0, the starting policy, only with eval.at_start, and a start that resumed
            # each multiple from the step it resumed at on, that one unless an earlier start completed its evaluation.
            if shared_state.evaluation_due(step):
                eval_start = time.perf_counter()
                logger.info(f"Starting evaluation at step {shared_state.train_step}")
                limited = getattr(workflow, "start_limit", None) is not None  # train.max_starting_rollouts
                if limited:  # its start-limit metrics cover this evaluation
                    client_monitor("eval").start_limit_metrics(reset=True)
                # Rollout workers read the sampling client from its published binding, which training publishes for
                # its own: an evaluation whose rollouts run in worker processes while training's do not needs it too.
                if getattr(workflow, "rollout_isolation", "inprocess") == "process":
                    version = shared_state.model_info.llm.version
                    self._publish_worker_binding(workflow, shared_state.sampling_client, version)

                eval_data_queue = asyncio.Queue[dict]()
                eval_rollout_result_queue = asyncio.Queue[Any]()  # each group's _Settled

                # Populate eval queue with all eval data upfront
                total_eval_tasks = 0
                for i_batch in range(len(shared_state.eval_dataloader)):
                    batch = shared_state.eval_dataloader.get_batch(i_batch)
                    for data in batch:
                        eval_data_queue.put_nowait(data)
                        total_eval_tasks += 1

                rows, num_workers = _rows(self.config, "eval", self.config.eval.max_concurrent_rollouts)
                label_stats = {label: _no_groups_yet() for label in labels}

                # Run eval with TaskGroup - completion monitor will raise TerminateEvalLoop when done
                try:
                    async with asyncio.TaskGroup() as tg:
                        # Spawn workers with eval tracker
                        for i in range(num_workers):
                            tg.create_task(
                                self._rollout_workflow_worker_loop(
                                    shared_state,
                                    workflow,
                                    eval_data_queue,
                                    eval_rollout_result_queue,
                                    tracker=eval_tracker,
                                    rows=rows,
                                ),
                                name=f"eval_rollout_workflow_worker_{i}",
                            )
                        # Spawn completion monitor
                        tg.create_task(
                            self._eval_completion_monitor(
                                total_eval_tasks,
                                eval_rollout_result_queue,
                                tracker=eval_tracker,
                                label_stats=label_stats,
                            ),
                            name="eval_completion_monitor",
                        )
                except* TerminateEvalLoop:
                    pass  # Expected - eval completed successfully

                eval_elapsed = time.perf_counter() - eval_start
                logger.info(f"Evaluation completed in {eval_elapsed:.1f}s for {total_eval_tasks} tasks")

                # Export and log eval stats, each label's among them
                _count_every_reward_key(reward_keys, label_stats.values())
                for label, stats in label_stats.items():
                    stats.record(eval_tracker, prefix=metric_prefix(column, label))
                eval_stats = eval_tracker.export(reset=True)
                if limited:
                    eval_stats.update(client_monitor("eval").start_limit_metrics(prefix="eval/rollout_client"))
                eval_stats["eval/time"] = eval_elapsed
                eval_stats["eval/num_tasks"] = total_eval_tasks
                # The step whose policy the evaluation started on; it is logged at the step training reached by its end.
                eval_stats["eval/step"] = step
                # Use force=True since train already logged at this step
                shared_state.stats_logger.log(step=shared_state.train_step, stats=eval_stats, force=True)
                if shared_state.evaluations_path is not None:  # so that a start that resumes at step does not rerun it
                    record_evaluation(
                        shared_state.evaluations_path,
                        step=step,
                        logged_at=shared_state.train_step,
                        tasks=total_eval_tasks,
                        seconds=eval_elapsed,
                    )

            shared_state.eval_handled_step = step
            shared_state.eval_progress_event.set()

    async def _wait_for_starting_policy_evaluation(self, shared_state: TrainLoopSharedState) -> None:
        """Hold the first update of a run launched at step 0 until the evaluation loop has handled the step it started
        on, when that is due: the evaluation of the starting policy (``eval.at_start``).

        An evaluation's rollouts sample the policy that is current when they run: rollout workers follow the sampling
        client the trainer publishes, and a full fine-tuning service serves its newest weights.  An update during the
        step-0 evaluation would mix later policies into the number the held-out curve starts from.  Training rollouts
        go on meanwhile.  There is nothing to wait for without an evaluation loop, without ``eval.at_start``, or when
        an earlier start of the run completed the evaluation of step 0.  A shutdown request ends the wait with
        ``TrainInterrupted``: no update of the batch has been applied, so it is dropped whole and a resume redoes it,
        as when a shutdown comes while the batch waits for its rollouts.
        """

        if not shared_state.eval_loop_running or not shared_state.evaluation_due(0):
            return
        while shared_state.eval_handled_step is None:
            if shared_state.shutdown_event.is_set():
                raise TrainInterrupted()
            shared_state.eval_progress_event.clear()
            progress = asyncio.ensure_future(shared_state.eval_progress_event.wait())
            shutdown = asyncio.ensure_future(shared_state.shutdown_event.wait())
            try:
                await asyncio.wait((progress, shutdown), return_when=asyncio.FIRST_COMPLETED)
            finally:
                progress.cancel()
                shutdown.cancel()

    async def _wait_for_last_step_evaluation(self, shared_state: TrainLoopSharedState) -> None:
        """Let the evaluation loop handle the last training step before the loops stop.

        Stopping the loops cancels the evaluation loop, so without this wait an evaluation due at the last step
        never ran, and one still running from an earlier step was cancelled.  A start that resumed a run at its last
        step trains none, and waits for the evaluation of that step if it is due: one that a crash cut short, which
        the run's evaluations.jsonl does not list.  There is nothing to wait for without an evaluation loop.  A
        shutdown request ends the wait at once: the loops then stop as before, cancelling the evaluation, and the
        final checkpoint is still saved.
        """

        if not shared_state.eval_loop_running:
            return
        while shared_state.eval_handled_step != shared_state.train_step and not shared_state.shutdown_event.is_set():
            shared_state.eval_progress_event.clear()
            progress = asyncio.ensure_future(shared_state.eval_progress_event.wait())
            shutdown = asyncio.ensure_future(shared_state.shutdown_event.wait())
            try:
                await asyncio.wait((progress, shutdown), return_when=asyncio.FIRST_COMPLETED)
            finally:
                progress.cancel()
                shutdown.cancel()

    def _evaluates_during_training(self, shared_state: TrainLoopSharedState) -> bool:
        """Whether an evaluation loop runs beside training: there is an eval split, and ``eval`` triggers (its
        ``every`` is positive)."""

        eval_every = shared_state.eval_every
        return self.eval_dataset is not None and eval_every is not None and eval_every > 0

    async def shutdown_loops(self, shared_state: TrainLoopSharedState):
        shared_state.shutdown_event.set()
        shared_state.sampling_client_updated_event.set()
        raise TerminateTrainLoop()

    async def _save_checkpoint_and_get_sampling_client(
        self,
        training_client: tinker.TrainingClient,
        i_batch: int,
        save_every: int | None,
        start_batch: int = 0,
        watchdog: Watchdog | None = None,
        recovery_every: int | None = None,
        policy_version: int | None = None,
        optimizer_steps: int | None = None,
    ) -> tinker.SamplingClient:
        """Save a checkpoint with wandb_run_id and return the new sampling client.

        Three cadences, in precedence order (``steps_completed`` counts steps
        since ``start_batch``; the name/batch are ``i_batch + 1`` because the
        weights are the ones *after* training on ``i_batch``, so a resume starts
        from ``i_batch + 1``):

        1. ``steps_completed % save_every == 0`` - a *permanent* full checkpoint
           named ``<batch>``, kept forever. ``save_every`` is ``None`` for
           ``checkpoint.every: 0``, which saves no permanent checkpoints.
        2. ``recovery_every`` set and ``steps_completed % recovery_every == 0``
           - an AReaL-style *recovery* full checkpoint named
           ``recovery-<batch>``. Same weights+optimizer payload, but
           ``checkpoint.keep_recovery`` deletes all but the newest few once the
           save is recorded (:mod:`platoon.train.tinker.checkpoint_pruning`),
           so the recovery cadence can run every step without growing the
           checkpoint store.
        3. otherwise - a weights-only update (no disk write in the gateway's
           full mode), which cannot be resumed from.

        Unlike tinker_cookbook's version this puts the wandb_run_id in the
        checkpoint loop_state so WandB run resumption works.
        """
        # Count steps completed since start_batch (1-indexed for human intuition)
        steps_completed = i_batch - start_batch + 1
        next_batch = i_batch + 1
        if save_every is not None and steps_completed % save_every == 0:
            checkpoint_kind, checkpoint_name = "permanent", f"{next_batch:06d}"
        elif recovery_every is not None and steps_completed % recovery_every == 0:
            checkpoint_kind = "recovery"
            checkpoint_name = f"{RECOVERY_CHECKPOINT_PREFIX}{next_batch:06d}"
        else:
            checkpoint_kind, checkpoint_name = None, None

        if checkpoint_name is None:
            logger.info(
                f"Saving weights only after completing batch {i_batch} "
                f"(steps_completed={steps_completed}, save_every={save_every}, "
                f"recovery_every={recovery_every})"
            )
            try:
                sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            except Exception as e:
                logger.exception(f"save_weights_and_get_sampling_client_async failed: {e}")
                raise
            logger.debug("Weights saved and sampling client created")
            return sampling_client

        # Save a full (weights + optimizer) checkpoint with loop state.
        logger.info(
            f"Saving {checkpoint_kind} checkpoint {checkpoint_name} after completing batch {i_batch} "
            f"(steps_completed={steps_completed}, save_every={save_every}, "
            f"recovery_every={recovery_every})"
        )
        try:
            # A full-parameter save_state takes minutes; keep the watchdog fed.
            path_dict = await self._await_with_heartbeat(
                checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=checkpoint_name,
                    log_path=self.run_log_path,
                    loop_state=await self._checkpoint_state(
                        checkpoint_name,
                        {
                            "batch": next_batch,
                            "wandb_run_id": self._stats_logger.wandb_run_id,
                            "policy_version": next_batch if policy_version is None else policy_version,
                            "optimizer_steps": next_batch if optimizer_steps is None else optimizer_steps,
                        },
                    ),
                    kind="both",
                ),
                name="save_checkpoint",
                watchdog=watchdog,
                total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
            )
        except Exception as e:
            logger.exception(f"save_checkpoint_async failed: {e}")
            raise
        if checkpoint_kind == "recovery":
            self._recovery_saved(path_dict, next_batch if policy_version is None else policy_version)
        logger.info(f"Checkpoint async completed. Creating sampling client from {path_dict['sampler_path']}")
        try:
            sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        except Exception as e:
            logger.exception(f"create_sampling_client failed: {e}")
            raise
        logger.info("Sampling client created successfully")
        return sampling_client

    def _training_tasks(self) -> Any:
        """Every task training draws from, once each, in pool order: a given order's tasks (a weighted order's, those
        its first epoch leaves out included), else the training dataset's rows."""

        tasks = getattr(getattr(self, "train_order", None), "items", None)
        if tasks is None:
            tasks = getattr(self, "train_dataset", None)
        return tasks

    def _begin_task_records(self, resume_info: Any, start_batch: int) -> None:
        """Start this start's records of the run's per-task statistics (:class:`TaskRecorder`) over every task training
        draws from, with those of the checkpoint ``resume_info`` resumes from, when it names a snapshot that reads back,
        else those of the newest older checkpoint whose snapshot does, else from empty (:meth:`_task_records_started`
        records why in ``lock.json``)."""

        tasks = self._training_tasks() or []
        self._recorder = None
        self._tasks_lost = None
        try:
            self._recorder, self._tasks_lost = TaskRecorder.begin(
                run_dir=self.run_log_path,
                config=self.config,
                tasks=tasks,
                records=read_checkpoint_records(self.run_log_path),
                checkpoint=resume_info.name if resume_info is not None else None,
                resume=resume_record(resume_info) if resume_info is not None else None,
                batch=start_batch,
            )
        except Exception:  # the statistics never fail training
            logger.warning("This start keeps no per-task statistics: they could not start", exc_info=True)

    def _task_records_started(self) -> None:
        """The launch save succeeded: the start's events go to the event log, and a start that could not load the
        statistics its checkpoint named says so in ``lock.json`` (``tasks_state_lost``).  Never raises."""

        recorder = getattr(self, "_recorder", None)
        if recorder is None:
            return
        try:
            recorder.started()
        except Exception:  # the statistics never fail training
            logger.warning("Could not log the start of this start's per-task statistics", exc_info=True)
        if (lost := getattr(self, "_tasks_lost", None)) is not None:
            record_lost_state(self.run_log_path, lost)

    async def _checkpoint_state(self, checkpoint: str, loop_state: dict[str, Any]) -> dict[str, Any]:
        """``loop_state`` of full checkpoint ``checkpoint``, with the records of what it saves first: the snapshot of
        the run's per-task statistics (``tasks``), and where the run's draws stand, with the groups that settled and no
        committed step consumed (``dispatch``, :func:`~platoon.train.tinker.dispatcher.save_state`).  Both are taken
        at the same point, before anything else runs; the dispatcher's files are written in a thread, since the groups
        they keep can take seconds to encode.  Each record is left out when it was not written: the statistics are not
        kept or stopped, or a file could not be written.  Under a group admission rule, the row also records where the
        order stands (``next_position``), which a resume that cannot read the dispatcher's state goes on from."""

        loop_state = dict(loop_state)
        recorder = getattr(self, "_recorder", None)
        if recorder is not None and (record := recorder.snapshot(checkpoint)) is not None:
            loop_state["tasks"] = record
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is not None:
            state, kept = dispatcher.state(), dispatcher.kept()
            if getattr(self, "_admissions", None) is not None:  # under a rule: where the order stands, in the row too
                loop_state[ORDER_POSITION_KEY] = state["next_position"]
            if (record := await asyncio.to_thread(save_state, self.run_log_path, checkpoint, state, kept)) is not None:
                loop_state[DISPATCH_KEY] = record
        return loop_state

    def _dispatch(
        self,
        task_group: asyncio.TaskGroup,
        shared_state: TrainLoopSharedState,
        start_batch: int,
        end_batch: int,
        train_workflow: RolloutWorkflow,
        results: asyncio.Queue,
    ) -> Dispatcher | None:
        """Start the dispatcher of this start's training groups (:mod:`platoon.train.tinker.dispatcher`) in
        ``task_group``: the one built at launch (:meth:`_new_dispatcher`), with what the checkpoint it resumes from
        saved, else a new one.  Its groups settle into ``results``, the groups a resume kept first.  The train loop's
        tests put their groups in ``results`` in its place, and return None: the loop then tells no dispatcher of its
        steps."""

        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is None:
            dispatcher = self._dispatcher = self._new_dispatcher(shared_state.train_dataloader, start_batch, end_batch)
        dispatcher.bind(results)
        dispatcher.rolling = bool(getattr(train_workflow, "runs_rollouts_on_lease", False))  # rolling starts
        if dispatcher.permit_release == "rollout":
            logger.info(
                "Each rollout returns its permit as it ends (train.permit_release: rollout); a group's rollouts "
                "start %s",
                "as permits free" if dispatcher.rolling else "together (its workflow sets no runs_rollouts_on_lease)",
            )

        async def run_group(draw: BoundDraw, lease: DrawLease) -> None:
            await self._run_training_group(shared_state, train_workflow, dispatcher, draw, lease)

        def spawn(coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
            return task_group.create_task(coroutine)

        task_group.create_task(
            dispatcher.run(spawn=spawn, run_group=run_group, stop=shared_state.shutdown_event),
            name="train_dispatcher",
        )
        return dispatcher

    def _new_dispatcher(self, train_dataloader: Any, start_batch: int, end_batch: int) -> Dispatcher:
        """The dispatcher of this start's training groups, from ``start_batch`` to ``end_batch``, over the run's order
        (:meth:`_samplers`), starting at the checkpoint's batch; and, for a run under a group admission rule, its rules
        beside it (``_admissions``, :class:`~platoon.train.admissions.AdmissionSet`), whose cap the dispatcher counts
        its steps by."""

        config = self.config
        batch_size = config.train.batch_size
        tasks = self._training_tasks()
        column = label_column(config)
        recorder = getattr(self, "_recorder", None)
        rows, _ = _rows(config, "train", config.train.max_concurrent_rollouts)
        rows.permits.on_held = client_monitor("train").set_permits_held  # rollout_client/permits_held_mean
        sampler = str(getattr(config.train, "task_sampler", None) or "auto")
        samplers = self._samplers(train_dataloader, sampler)
        if recorder is not None:
            samplers.attach(recorder)  # adaptive samplers read the statistics, and observe settled groups
        # The group admission rules (RFC 0019, section 6), resolved as platoon train resolved them, whose resolution
        # ran each rule's check with the run's workflow (run_components): none in a run whose every rule is all, which
        # trains as before.
        choices = choose_admissions(config, check=False) if rule_mode(config) else None
        self._admissions = (
            AdmissionSet(choices, samplers=samplers, batch_size=batch_size, label_column=column)
            if choices is not None and choices.active
            else None
        )
        try:
            fingerprint = sampling_fingerprint(
                config, tasks or [], getattr(self, "train_sampling", None), admission=choices
            )
        except Exception:  # what a resume compares never stops training
            logger.warning("Could not take the fingerprint of the run's sampling inputs", exc_info=True)
            fingerprint = {}
        if recorder is not None:
            start = recorder.start
        else:  # the statistics did not start: number this start's draws as they would have
            start = next_start(read_checkpoint_records(self.run_log_path))
        return Dispatcher(
            rows=rows,
            samplers=samplers,
            batch_size=batch_size,
            max_staleness=config.train.max_staleness,
            steps=max(0, end_batch - start_batch),
            limits=getattr(config.train, "rollout_group_limits", None) or {},
            labels=dataset_labels(tasks, column) if tasks is not None else (),
            start=start,
            next_position=start_batch * batch_size,
            recorder=recorder,
            version=lambda: self.model_info.llm.version,
            finish_batches=getattr(config.train, "num_minibatches", 1) > 1,
            sampler=sampler,
            fingerprint=fingerprint,
            step=start_batch,
            groups_per_step=self._admissions.cap if self._admissions is not None else None,
            permit_release=config.train.permit_release,
        )

    def _samplers(self, train_dataloader: Any, sampler: str) -> SamplerSet:
        """The task samplers the run draws through: those of ``train_sampling``, as platoon train hands them the
        trainer; else one that draws each position of the dataloader's order (its batch ``b`` is positions ``b ×
        batch_size`` on) as it is, over the training tasks (:meth:`_training_tasks`), whose first row of a task a
        resume draws again."""

        config = self.config
        batch_size = config.train.batch_size
        column = label_column(config)
        sampling = getattr(self, "train_sampling", None)
        if sampling is not None:
            return SamplerSet.build(sampling, batch_size=batch_size, label_column=column)
        if named := _named_samplers(config):
            raise ValueError(
                f"{'; '.join(named)}: a trainer given no train_sampling draws its dataloader's order, so it runs no "
                "other task sampler: entry and adaptive samplers need AutoDataset.training's sampling, which platoon "
                "train hands the trainer (train_sampling=AutoDataset.training(config).sampling)"
            )
        batches: dict[int, list[dict]] = {}

        def order(position: int) -> dict:
            batch, index = divmod(position, batch_size)
            if batch not in batches:
                batches.clear()  # the dispatcher reads the order forward: one batch at a time
                batches[batch] = train_dataloader.get_batch(batch)
            return batches[batch][index]

        tasks = self._training_tasks()
        entries = tuple(entry_settings.run_settings(config))
        drop = (PRESERVE_DATASET_ORDER_COLUMN,)
        pool = None
        try:
            if hasattr(tasks, "column_names"):  # a dataset: its rows are converted only when a resume reads them
                pool = TaskPool.from_dataset(tasks, entries=entries, column=column, drop_columns=drop)
            elif tasks is not None:
                pool = TaskPool.from_rows(tasks, entries=entries, column=column, drop_columns=drop)
        except Exception:  # a resume then cannot draw the outstanding draws again, as with no training tasks
            logger.warning("Could not list the training tasks that a resume draws again", exc_info=True)
        return SamplerSet.of_order(order, pool=pool, name=sampler, batch_size=batch_size, label_column=column)

    @staticmethod
    def _resume_position(dispatcher: Dispatcher, extra: Mapping[str, Any]) -> bool:
        """For a resume that cannot read the dispatcher's state: move the order to where the checkpoint's row says it
        stood (``next_position``, which a checkpoint saved under a group admission rule records), so that the resume
        draws no position again that the run had consumed (a step under a rule draws up to its cap's worth of
        positions, so the checkpoint's batch lags behind).  Whether the row said, and the order moved."""

        position = extra.get(ORDER_POSITION_KEY)
        if isinstance(position, bool) or not isinstance(position, int) or position < dispatcher.next_position:
            return False
        dispatcher.next_position = position
        return True

    def _restore_dispatch(self, dispatcher: Dispatcher, resume_info: Any, policy_version: int) -> None:
        """Take up the draws that the checkpoint this start resumes from (``resume_info``) left outstanding, from the
        dispatcher's state its row names (``dispatch``): the groups it kept go to the steps first, the draws that waited
        for their label back to their backlogs, and the rest are drawn again, each from the row it drew (the pool's
        first row of its task when that row no longer holds it, or a state of platoon 0.3.0 named none); the order goes
        on where it stopped, each task sampler at its saved count, else at the count the position implies.  A
        change of the sampling inputs is warned of and recorded in ``lock.json`` (``sampling_changed``), and the run
        goes on with it: the order goes on at the same position, and a draw whose task left the pool counts as lost.
        A checkpoint without that state (one saved before platoon 0.3.0, or one whose write failed), or a state that
        cannot be read, has the outstanding draws the per-task statistics hold counted as lost, and the order goes on
        from the checkpoint's batch, as the producer had it; or, from a checkpoint saved under a group admission rule,
        from where its row says the order stood (:meth:`_resume_position`)."""

        checkpoint = str(resume_info.name)
        recorder = getattr(self, "_recorder", None)
        extra = getattr(resume_info, "extra", None) or {}
        record = extra.get(DISPATCH_KEY)
        if not isinstance(record, Mapping):
            lost = recorder.lose_outstanding() if recorder is not None else 0
            if self._resume_position(dispatcher, extra):
                logger.info(
                    "Checkpoint %s saved no state of the run's draws (its write failed): the run draws from where its "
                    "order stood, position %d, which its row records, and %d draws that no committed step had trained "
                    "count as lost",
                    checkpoint,
                    dispatcher.next_position,
                    lost,
                )
                return
            logger.info(
                "Checkpoint %s saved no state of the run's draws (a platoon before 0.3.0 saved none, or its write "
                "failed): the run draws from the order where the checkpoint's batch begins, and %d draws that no "
                "committed step had trained count as lost",
                checkpoint,
                lost,
            )
            return
        pool = dispatcher.samplers.pool
        try:
            state = read_state(self.run_log_path, record)
            position = state.get("next_position")
            if isinstance(position, bool) or not isinstance(position, int) or position < 0:
                raise ValueError(f"next_position {position!r} is no position")
            kept = self._read_kept_groups(state.get("kept"))
            drawn, saved_tokens, fallback_slots = self._saved_samplings(state, checkpoint)
            plan = plan_resume(
                outstanding=state.get("outstanding") or [],
                waiting=state.get("waiting") or {},
                kept=kept,
                pool=pool,
            )
            # Each draw drawn again takes the row it drew: its pool row while that row still holds its task, else the
            # pool's first row of the task (the pool changed, or a state of format 1 saved no row).
            rows: dict[tuple[int, int], dict[str, Any]] = {}
            taken: dict[tuple[int, int], tuple[int | None, str | None]] = {}
            for key, task_id, _place in [*plan.reissue, *plan.waiting]:
                index, by = drawn.get(key, (None, None))
                found = pool.reissue_row(task_id, index)
                if found is None:
                    raise ValueError(f"no row of the training tasks holds task {task_id!r}")
                rows[key], taken[key] = found[0], (found[1], by)
            for key, _task_id, _place in plan.kept:
                taken[key] = drawn.get(key, (None, None))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            if recorder is not None:
                recorder.lose_outstanding()
            moved = self._resume_position(dispatcher, extra)
            logger.warning(
                "The state of the run's draws that checkpoint %s saved cannot be read (%s): the draws it left "
                "outstanding are lost, and the order goes on from %s",
                checkpoint,
                exc,
                f"position {dispatcher.next_position}, where its row says it stood"
                if moved
                else "the checkpoint's batch",
            )
            return
        if recorder is not None:
            # The draws an older snapshot of the statistics holds as outstanding that the checkpoint no longer does: the
            # statistics lost what became of them, when the checkpoint's own snapshot did not read back.
            recorder.lose_outstanding(keep=plan.keys())
            for key, task_id, _position in plan.lost:
                recorder.lose(key, task_id)
        if changes := fingerprint_changes(state.get("fingerprint"), dispatcher.fingerprint):
            logger.warning(
                "The run's sampling inputs changed since checkpoint %s: %s. The run goes on with them, from where the "
                "order stopped, and the draws of tasks that left the run's training tasks are lost",
                checkpoint,
                "; ".join(changes),
            )
            record_sampling_changed(
                self.run_log_path, {"start": dispatcher.start, "checkpoint": checkpoint, "changes": changes}
            )
        dispatcher.next_position = position
        dispatcher.samplers.restore(state.get("samplers"), next_position=position)
        dispatcher.restore(
            kept=[(key, task_id, place, kept[key]) for key, task_id, place in plan.kept],
            reissue=[(key, rows[key], place) for key, _task_id, place in plan.reissue],
            waiting=[(key, rows[key], place) for key, _task_id, place in plan.waiting],
            version=policy_version,
            drawn=taken,
        )
        tokens = dispatcher.restore_tokens(saved_tokens, version=policy_version, fallbacks=fallback_slots)
        if tokens["dropped"]:
            logger.warning(
                "Resuming from checkpoint %s: %d slots that waited for an entry's task sampler are dropped, since "
                "their entry no longer gets slots (each logged as slot_skipped in tasks/events.jsonl)",
                checkpoint,
                tokens["dropped"],
            )
        logger.info(
            "Resuming from checkpoint %s: %d groups that had settled and no step had trained train first, %d draws "
            "that had not trained are drawn again first%s, and the order goes on at position %d",
            checkpoint,
            len(plan.kept),
            len(plan.reissue) + len(plan.waiting),
            f" ({len(plan.lost)} lost: their tasks left the run)" if plan.lost else "",
            position,
        )

    @staticmethod
    def _saved_samplings(
        state: Mapping[str, Any], checkpoint: str
    ) -> tuple[dict[tuple[int, int], tuple[int | None, str | None]], dict[str, list[int]], frozenset[int]]:
        """What a dispatcher's state of format 2 adds to platoon 0.3.0's: the row each outstanding draw drew and the
        sampler that drew it, each entry's tokens, and those whose slot the run sampler's fallback handed.  A part that
        cannot be read is left out with a warning, the rest of the state taken up: without the rows, each draw drawn
        again takes the first row of its task, as from a state of format 1; without the tokens, their slots are
        lost."""

        try:
            drawn = saved_draws(state)
        except ValueError as exc:
            logger.warning(
                "Resuming from checkpoint %s: %s; each draw drawn again takes the first row of its task instead",
                checkpoint,
                exc,
            )
            drawn = {}
        try:
            tokens, fallbacks = saved_token_positions(state), saved_fallback_slots(state)
        except ValueError as exc:
            logger.warning(
                "Resuming from checkpoint %s: %s; the slots that waited for an entry's task sampler are lost",
                checkpoint,
                exc,
            )
            tokens, fallbacks = {}, frozenset()
        return drawn, tokens, fallbacks

    def _read_kept_groups(self, record: Any) -> dict[tuple[int, int], "_Settled"]:
        """The groups the checkpoint a start resumes from kept, as the steps dequeue them, by draw, in the order they
        settled; none, with a warning, when their file cannot be read, so that their tasks are rolled out again."""

        if not isinstance(record, Mapping):
            return {}
        try:
            groups = [decode_group(group) for group in read_kept(self.run_log_path, record)]
        except ValueError as exc:
            logger.warning(
                "The groups the checkpoint kept, which had settled and no step had trained yet, cannot be read, so "
                "their tasks are rolled out again: %s",
                exc,
            )
            return {}
        return {
            key: _Settled(entry, label, output, group_record, key) for key, entry, label, group_record, output in groups
        }

    def _start_recovery_pruning(self) -> RecoveryCheckpointPruner | None:
        """The pruner of this run's recovery checkpoints (``checkpoint.keep_recovery``), which deletes them on a Tinker
        session of its own; None without ``keep_recovery``."""

        keep = getattr(self.config.checkpoint, "keep_recovery", None)
        # The largest bound of the run and its entries: a slow entry's group may still sample an older checkpoint.
        max_staleness, _ = entry_settings.largest_staleness(self.config)
        if keep is None or max_staleness is None:  # loading refuses keep_recovery without max_staleness
            return None
        base_url = self.config.tinker_base_url
        return RecoveryCheckpointPruner(
            run_dir=self.run_log_path,
            keep=keep,
            max_staleness=max_staleness,
            service_client_factory=lambda: tinker.ServiceClient(base_url=base_url),
        )

    def _recovery_saved(self, paths: dict[str, str], policy_version: int) -> None:
        """Hand a recovery save that ``checkpoints.jsonl`` now records to the pruner, if any, which deletes the run's
        older recovery checkpoints beside training."""

        pruner = getattr(self, "_recovery_pruner", None)
        if pruner is not None:
            pruner.recovery_saved(paths, policy_version)

    async def _create_training_client(
        self, service_client: tinker.ServiceClient, resume_info: checkpoint_utils.CheckpointRecord | None
    ) -> tinker.TrainingClient:
        if resume_info:
            # Resuming interrupted training - load optimizer state for proper continuation
            training_client = await service_client.create_training_client_from_state_with_optimizer_async(
                resume_info.state_path
            )
            logger.info(f"Resumed training from {resume_info.state_path}")
        elif self.config.checkpoint.load_checkpoint_path:
            # Starting fresh from a checkpoint - load weights only (fresh optimizer)
            training_client = await service_client.create_training_client_from_state_async(
                self.config.checkpoint.load_checkpoint_path
            )
            logger.info(f"Loaded weights from {self.config.checkpoint.load_checkpoint_path}")
        else:
            training_client = await service_client.create_lora_training_client_async(
                self.config.train.model_name, rank=self.config.train.lora_rank
            )
            logger.info(
                f"Created new training client for {self.config.train.model_name} "
                f"with rank {self.config.train.lora_rank}"
            )

        return training_client

    async def _open_training_client(
        self,
        service_client: tinker.ServiceClient,
        resume_info: checkpoint_utils.CheckpointRecord | None,
        stop: asyncio.Event | None = None,
    ) -> tinker.TrainingClient | None:
        """:meth:`_create_training_client`, asked again after each refusal until the service accepts it, for up to
        :data:`TRAINING_CLIENT_OPEN_TIMEOUT_S`, each refusal logged; None once ``stop`` is set between two asks
        (training was asked to stop before it began).

        A service may refuse a start while it still holds the session of a start that died, until that session
        expires, or answer nothing while it restarts.  A refusal that asking again cannot change
        (:func:`_service_refusal`), and any error that is not the service's, is raised at once.
        """

        first = time.monotonic()
        wait = TRAINING_CLIENT_FIRST_WAIT_S
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._create_training_client(service_client, resume_info)
            except Exception as exc:
                waited = time.monotonic() - first
                left = TRAINING_CLIENT_OPEN_TIMEOUT_S - waited
                if not _service_refusal(exc):
                    raise
                if left <= 0:
                    exc.add_note(
                        f"The training service refused the training client {attempt} times in {waited:.0f}s; a start "
                        f"asks for {TRAINING_CLIENT_OPEN_TIMEOUT_S:.0f}s"
                    )
                    raise
                pause = min(wait, left)
                reason = " ".join(f"{type(exc).__name__}: {exc}".split())[:500].rstrip(".")
                logger.warning(
                    f"The training service refused the training client (attempt {attempt}, {waited:.0f}s after the "
                    f"first): {reason}. Asking again in {pause:.0f}s, for up to {left:.0f}s more"
                )
                if stop is None:
                    await asyncio.sleep(pause)
                else:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(stop.wait(), timeout=pause)
                    if stop.is_set():
                        logger.info("Stopped asking the training service for a training client: training was stopped")
                        return None
                wait = min(wait * 2, TRAINING_CLIENT_LONGEST_WAIT_S)

    def _setup_signal_handlers(self, shutdown_event: asyncio.Event) -> None:
        """Set up signal handlers for graceful shutdown.

        First Ctrl+C (or SIGTERM): Sets shutdown_event. The train loop stops at the next step boundary (a batch
        that has applied an update finishes first) and ``train`` saves ``<step>_interrupted``.
        Second Ctrl+C: Forces immediate exit.

        SIGUSR2 and SIGUSR1 are what ``platoon train`` passes on to a start it supervises
        (:mod:`platoon.train.supervise`): SIGUSR2 asks for the stop a first signal makes and never counts as one, so a
        start that got the signal itself as well does not take the two for a second signal; SIGUSR1 forces the exit
        a second signal forces.
        """
        global _SHUTDOWN_REQUESTED
        _SHUTDOWN_REQUESTED = False

        def force_exit():
            _stop_rollout_workers()
            # Force cleanup of wandb
            try:
                import wandb

                wandb.finish(quiet=True)
            except Exception:
                pass
            os._exit(1)

        def signal_handler(signum, frame):
            global _SHUTDOWN_REQUESTED
            sig_name = signal.Signals(signum).name

            if _SHUTDOWN_REQUESTED:
                # Second signal - force exit
                logger.warning(f"Received {sig_name} again. Forcing immediate exit.")
                force_exit()
            else:
                # First signal - request graceful shutdown
                _SHUTDOWN_REQUESTED = True
                logger.info(f"Received {sig_name}. Requesting graceful shutdown... (press Ctrl+C again to force exit)")
                shutdown_event.set()

        def supervisor_stop_handler(signum, frame):
            if not shutdown_event.is_set():
                logger.info("platoon train was interrupted: stopping at the next step boundary")
                shutdown_event.set()

        def supervisor_force_handler(signum, frame):
            logger.warning("platoon train was interrupted again. Forcing immediate exit.")
            force_exit()

        # Register handlers for SIGINT (Ctrl+C) and SIGTERM, and for what platoon train passes on of them
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGUSR2, supervisor_stop_handler)
        signal.signal(signal.SIGUSR1, supervisor_force_handler)

    async def train(self, train_workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow | None = None):
        if self._stats_logger is None or self._train_tracker is None:
            raise RuntimeError(
                "Trainer must be used as an async context manager. Use 'async with trainer:' before calling train()."
            )

        # keep_recovery's rules again, before training opens its service: a config class that overrides the training
        # config's own checks loads without them (platoon train checks them first, in training_problems).
        check_recovery_pruning(self.config)
        if getattr(self.config.eval, "at_start", False) and not evaluates(self.config):
            logger.warning(
                "eval.at_start is ignored: eval.every is 0, which turns evaluation during training off, the starting "
                "policy's too"
            )
        shutdown_event = asyncio.Event()

        # Set up signal handlers for graceful Ctrl+C shutdown
        self._setup_signal_handlers(shutdown_event)

        # A key the service refuses (401, 403) ends training with one line that names the key and the service.
        with key_refusals():
            service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)

        resume_info = checkpoint_utils.get_last_checkpoint(self.run_log_path)
        start_batch = (resume_info.batch or 0) if resume_info else 0
        policy_version, optimizer_steps = _resume_policy_counters(resume_info, self.config.train.num_minibatches)

        with key_refusals(asked="a training client"):
            training_client = await self._open_training_client(service_client, resume_info, stop=shutdown_event)
        if training_client is None:
            return  # stopped before training began: nothing to save

        # Ahead of the first rollout, while the trainer saves at launch: the toolchains the environments' harnesses run
        # from are staged on this node, and the rollout workers' fork servers start importing (platoon.train.startup).
        # No rollout starts before the staging is done (startup.ready() below).
        from platoon.train.components import normalize_environment_configs
        from platoon.train.startup import RolloutStartup

        startup = await RolloutStartup.begin(
            normalize_environment_configs(getattr(self.config, "environments", None)),
            [train_workflow, eval_workflow if self.eval_dataset is not None and evaluates(self.config) else None],
        )

        # The run's per-task statistics, those of the checkpoint it resumes from loaded before the launch save, which
        # saves them: a crash before the next full save loses none of them.
        self._begin_task_records(resume_info, start_batch)
        # The dispatcher of the training groups, which takes up the draws the checkpoint left outstanding before the
        # launch save, which saves them again: its kept groups, the draws to draw again, and the order's position.
        train_dataloader = PlatoonTinkerDataloader(
            self.train_dataset, self.config.train.batch_size, order=self.train_order
        )
        end_batch = _run_steps(self.config, len(train_dataloader))
        self._dispatcher = self._new_dispatcher(train_dataloader, start_batch, end_batch)
        if resume_info is not None:
            self._restore_dispatch(self._dispatcher, resume_info, policy_version)

        # Initial sampling client to use (the watchdog is not running yet). The
        # launch save is a *recovery* checkpoint: every (re)launch writes one, so
        # naming it permanently would accumulate a ~490 GB checkpoint per restart.
        launch_checkpoint_name = f"{RECOVERY_CHECKPOINT_PREFIX}{start_batch:06d}"
        logger.info(f"Saving recovery checkpoint {launch_checkpoint_name} at launch (batch {start_batch})")
        path_dict = await self._await_with_heartbeat(
            checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=launch_checkpoint_name,
                log_path=self.run_log_path,
                loop_state=await self._checkpoint_state(
                    launch_checkpoint_name,
                    {
                        "batch": start_batch,
                        "wandb_run_id": self._stats_logger.wandb_run_id,
                        "policy_version": policy_version,
                        "optimizer_steps": optimizer_steps,
                    },
                ),
                kind="both",
            ),
            name="save_checkpoint",
            total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
        )
        self._task_records_started()
        # Pruning starts with the launch save, so a resumed run first deletes what its earlier starts left.
        self._recovery_pruner = self._start_recovery_pruning()
        self._recovery_saved(path_dict, policy_version)
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
        sampling_client_step = policy_version
        sampling_client_updated_event = asyncio.Event()
        sampling_client_updated_event.set()

        # Set initial version and update sampling client (without auto-incrementing)
        self.model_info.llm.set_version(policy_version)
        self.model_info.llm.update_sampling_client(sampling_client, increment_version=False)

        optimizer_params = tinker.AdamParams(
            learning_rate=self.config.train.optimizer.learning_rate,
            beta1=self.config.train.optimizer.beta1,
            beta2=self.config.train.optimizer.beta2,
            eps=self.config.train.optimizer.eps,
            weight_decay=self.config.train.optimizer.weight_decay,
            grad_clip_norm=self.config.train.optimizer.grad_clip_norm,
        )

        eval_dataloader = None
        if self.eval_dataset is not None:
            eval_dataloader = PlatoonTinkerDataloader(
                self.eval_dataset, batch_size=1, shuffle_seed=None, drop_last=False
            )

        # The first rollouts, training's and an evaluation's, start only once the toolchains are staged, or once
        # startup.STAGING_WAIT_S passed or a shutdown was asked for (the rollouts then stage them themselves).
        await self._await_with_heartbeat(
            startup.ready(stop=shutdown_event), name="the staging of the harness toolchains"
        )

        # Initialize watchdog if enabled
        watchdog = None
        if self.config.watchdog.enabled:
            watchdog = Watchdog(
                timeout_seconds=self.config.watchdog.timeout_seconds,
                exit_code=self.config.watchdog.exit_code,
            )
            watchdog.start()

        shared_state = TrainLoopSharedState(
            config=self.config,
            shutdown_event=shutdown_event,
            train_step=start_batch,
            optimizer_params=optimizer_params,
            service_client=service_client,
            training_client=training_client,
            sampling_client=sampling_client,
            sampling_client_step=sampling_client_step,
            optimizer_steps=optimizer_steps,
            sampling_client_updated_event=sampling_client_updated_event,
            model_info=self._model_info,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            stats_logger=self._stats_logger,
            train_tracker=self._train_tracker,
            watchdog=watchdog,
            start_step=start_batch,
            # A start that resumes does not run again the evaluation of the step it resumes at when an earlier start
            # completed it; a fresh run has none to leave out.
            start_step_evaluated=resume_info is not None
            and start_batch in completed_evaluation_steps(self.run_log_path),
            evaluations_path=os.path.join(self.run_log_path, EVALUATIONS_NAME),
        )

        shared_state.eval_loop_running = self._evaluates_during_training(shared_state)
        if shared_state.eval_loop_running:
            logger.log(*_evaluation_plan(shared_state, resumed=resume_info is not None, end=end_batch))
        _limit_starts(self.config, train_workflow, eval_workflow if shared_state.eval_loop_running else None)

        interrupted = False
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    self._train_loop(tg, shared_state, start_batch, end_batch, train_workflow), name="train_loop"
                )
                if shared_state.eval_loop_running:
                    tg.create_task(self._eval_loop(shared_state, eval_workflow), name="eval_loop")
        except* TerminateTrainLoop:
            pass
        except* (KeyboardInterrupt, asyncio.CancelledError, TrainInterrupted):
            interrupted = True
            logger.info("Training interrupted. Saving checkpoint before exit...")
        finally:
            # Stop watchdog
            if watchdog:
                watchdog.stop()

        # Save checkpoint
        current_step = shared_state.train_step
        if interrupted:
            # Save interrupt checkpoint so we can resume
            if current_step > start_batch:
                logger.info(f"Saving permanent checkpoint {current_step:06d}_interrupted at step {current_step}...")
                await self._await_with_heartbeat(
                    checkpoint_utils.save_checkpoint_async(
                        training_client=shared_state.training_client,
                        name=f"{current_step:06d}_interrupted",
                        log_path=self.run_log_path,
                        kind="both",
                        loop_state=await self._checkpoint_state(
                            f"{current_step:06d}_interrupted",
                            {
                                "batch": current_step,
                                "wandb_run_id": self._stats_logger.wandb_run_id,
                                "policy_version": shared_state.model_info.llm.version,
                                "optimizer_steps": shared_state.optimizer_steps,
                            },
                        ),
                    ),
                    name="save_checkpoint",
                    total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
                )
                logger.info(f"Checkpoint saved. Resume training to continue from step {current_step}.")
        elif start_batch < end_batch:
            # Training completed normally - save final checkpoint
            logger.info(f"Saving permanent checkpoint final at step {end_batch}...")
            await self._await_with_heartbeat(
                checkpoint_utils.save_checkpoint_async(
                    training_client=shared_state.training_client,
                    name="final",
                    log_path=self.run_log_path,
                    kind="both",
                    loop_state=await self._checkpoint_state(
                        "final",
                        {
                            "batch": end_batch,
                            "wandb_run_id": self._stats_logger.wandb_run_id,
                            "policy_version": shared_state.model_info.llm.version,
                            "optimizer_steps": shared_state.optimizer_steps,
                        },
                    ),
                ),
                name="save_checkpoint",
                total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
            )
            logger.info("Training complete successfully!")
        elif shared_state.eval_loop_running and shared_state.evaluation_due(start_batch):
            # A start that resumed at the last step, whose evaluation no earlier start completed: it trained nothing.
            if shared_state.eval_handled_step == start_batch:
                logger.info(f"Training was already complete; this start ran the evaluation due at step {start_batch}")
            else:
                logger.info(
                    f"Training was already complete; the evaluation due at step {start_batch} stopped before it "
                    "ended, so the next start runs it"
                )
        else:
            logger.info("Training was already complete; nothing to do")
        if self._recovery_pruner is not None:
            await self._recovery_pruner.aclose()
        if (recorder := getattr(self, "_recorder", None)) is not None:
            recorder.close()
