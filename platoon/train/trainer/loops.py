"""The training loop of one start: each step dequeues settled groups, trains on them and publishes the new policy
(``_train_loop``); and stopping the loops (``shutdown_loops``).

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.
"""

import asyncio
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any

import tinker
from tinker import TensorData

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet
from platoon.train.dispatch.permits import _rollouts_at_once
from platoon.train.dispatch.settled import _output_label, _Settled, _settled
from platoon.train.labels import label_column, metric_labels, metric_prefix
from platoon.train.step.batch_transforms import (
    LOSS_NORMALIZATION_TOKENS_KEY,
    BatchTransformContext,
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
from platoon.train.step.collect import _get_task_workload_payload
from platoon.train.step.datums import _step_datums
from platoon.train.step.metrics import (
    LABEL_STATS,
    Censoring,
    _count_every_reward_key,
    _no_groups_yet,
    _ptc_cell_guard_stats,
    _record_group_stats,
)
from platoon.train.step.publish import _no_update_reason
from platoon.train.step.staleness import StaleWatch
from platoon.train.step.update import compute_training_metrics
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.tinker.rollout_metrics import step_metrics as client_step_metrics
from platoon.train.trainer.signals import TerminateTrainLoop
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.protocol import RolloutWorkflow
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.rollout_workload import RolloutWorkload

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


async def _train_loop(
    trainer: "PlatoonTinkerRLTrainer",
    task_group: asyncio.TaskGroup,
    shared_state: TrainLoopSharedState,
    start_batch: int,
    end_batch: int,
    train_workflow: RolloutWorkflow,
):
    column = label_column(trainer.config)
    # Every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
    tasks = trainer._training_tasks()
    group_limits = getattr(trainer.config.train, "rollout_group_limits", {})
    # The groups settle into the result queue, which the steps dequeue from: the dispatcher starts each group when
    # the horizon, the rollout permits and the label limits allow (platoon.train.dispatch.dispatcher).
    task_rollout_result_queue = asyncio.Queue[_Settled]()
    dispatcher = trainer._dispatch(
        task_group, shared_state, start_batch, end_batch, train_workflow, task_rollout_result_queue
    )

    # Publish the initial sampling-client binding before any group can
    # start, then declare the concurrency ceiling the metrics compare
    # against: max_concurrent_rollouts is a budget of rollouts, of which
    # a group takes its size in permits, and the monitor warns once when
    # more rollouts run at once.
    trainer._publish_worker_binding(
        train_workflow,
        shared_state.sampling_client,
        shared_state.model_info.llm.version,
        required=True,
    )
    budget = trainer.config.train.max_concurrent_rollouts
    client_monitor("train").set_configured_concurrency(
        _rollouts_at_once(trainer.config, "train", budget), budget=budget
    )

    # Start training loop.
    shared_state.train_step = start_batch
    logger.info(f"Starting training from batch {start_batch} to {end_batch}")
    # Each step's metrics break down by these labels of the training tasks (train.label_by), also at zero: those of
    # every task the run draws from, a weighted order's tasks that its first epoch leaves out included.
    labels = metric_labels(tasks, column)
    reward_keys: set[str] = set()
    # The stale filter's bound for each entry's groups: its own max_staleness, else the run's.
    recorder = getattr(trainer, "_recorder", None)
    prefixes = {label: metric_prefix(column, label) for label in labels}
    run_bound = trainer.config.train.max_staleness
    settings = entry_settings.run_settings(trainer.config)
    bounds = {label: own.max_staleness for label, own in settings.items()}
    skipped_seen: Counter[str | None] = Counter()  # draws a capped label's full backlog skipped, logged so far
    skipped_slots_seen = 0  # slots owed to adaptive entries that were skipped, logged so far
    # A run under a group admission rule (RFC 0019, section 6): each step trains the first batch_size admitted
    # groups it dequeues that are not stale, and closes once it has counted its cap (platoon.train.admissions).
    admissions: AdmissionSet | None = getattr(trainer, "_admissions", None)
    # What the straggler policy censors of the groups each step dequeues, and the warning of stale groups.
    threshold = recorder.threshold if recorder is not None else None
    censoring = Censoring(settings, trainer.config.train.workflow_config, threshold=threshold)
    stale_watch = StaleWatch(run_bound)

    while shared_state.train_step < end_batch:
        batch_start_time = time.perf_counter()

        assert trainer.config.train.batch_size % trainer.config.train.num_minibatches == 0, (
            f"batch_size {trainer.config.train.batch_size} must be divisible by "
            f"num_minibatches {trainer.config.train.num_minibatches}"
        )
        tasks_per_minibatch = trainer.config.train.batch_size // trainer.config.train.num_minibatches

        assert tasks_per_minibatch % trainer.config.train.num_microbatches == 0, (
            f"tasks_per_minibatch {tasks_per_minibatch} must be divisible by "
            f"num_microbatches {trainer.config.train.num_microbatches}"
        )

        tasks_per_microbatch = tasks_per_minibatch // trainer.config.train.num_microbatches

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
        for minibatch_num in range(trainer.config.train.num_minibatches):
            forward_backward_futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
            # Keep original datums (with mask and logprobs) for metrics computation
            original_datums_per_microbatch: list[list[tinker.Datum]] = []

            # Microbatches are used for gradient accumulation. While gradient accumulation
            # may be less important with tinker, this can help performance.
            # Microbatches are processed as they become available, allowing us to overlap
            # rollout sampling with training even within the same batch.
            # This is a second-level of pipelining orthogonal to the async/off-policy
            # pipelining at the batch level.
            for microbatch_num in range(trainer.config.train.num_microbatches):
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
                        await trainer._next_rollout(
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
                    client_monitor("train").record_queue_wait(queue_wait_elapsed, immediate=queue_wait_elapsed < 0.001)
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
                            admissions.waiting(shared_state.train_step, counted=step_counted, admitted=step_admitted)
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
                    trainer.batch_transforms,
                    BatchTransformContext(
                        config=trainer.config,
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
                loss_fn = getattr(trainer.config.train, "loss_fn", "cispo")
                loss_fn_config = getattr(trainer.config.train, "loss_fn_config", {})
                if (
                    trainer.config.train.workflow_config.filter_zero_advantage_datums
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
                            loss_fn=trainer.config.train.loss_fn,
                            loss_fn_config=trainer.config.train.loss_fn_config,
                        )
                    )
                except Exception as e:
                    elapsed = time.perf_counter() - fwd_bwd_submit_start
                    logger.exception(f"forward_backward_async submission failed after {elapsed:.1f}s: {e}")
                    raise
                logger.debug(f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s")

            if len(forward_backward_futures) == 0:
                logger.warning(
                    f"Minibatch {minibatch_num} of batch {shared_state.train_step} takes no optimizer update: "
                    "none of its microbatches had a nonzero objective"
                )
                continue

            if start_batch == 0 and getattr(shared_state, "optimizer_steps", 0) == 0:
                # The run's first update: an evaluation of its starting policy (eval.at_start) must sample that
                # policy throughout.
                await trainer._wait_for_starting_policy_evaluation(shared_state)
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
                    forward_backward_result = await trainer._await_with_heartbeat(
                        forward_backward_future.result_async(),
                        name=f"forward_backward (microbatch {microbatch_num})",
                        heartbeat_interval=60.0,
                        watchdog=shared_state.watchdog,
                        total_timeout=trainer.config.train.forward_backward_timeout_seconds,
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
                        loss_fn=trainer.config.train.loss_fn,
                        loss_fn_config=trainer.config.train.loss_fn_config,
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
                optim_result = await trainer._await_with_heartbeat(
                    optim_future.result_async(),
                    name="optim_step",
                    heartbeat_interval=60.0,
                    watchdog=shared_state.watchdog,
                    total_timeout=trainer.config.train.optim_step_timeout_seconds,
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
                "workload/batch/total_tasks_with_workload_metadata": float(batch_total_tasks_with_workload_metadata),
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
            sampling_client = await trainer._save_checkpoint_and_get_sampling_client(
                training_client=shared_state.training_client,
                i_batch=shared_state.train_step,
                save_every=shared_state.save_every,
                start_batch=start_batch,
                watchdog=shared_state.watchdog,
                recovery_every=trainer.config.checkpoint.recovery_every,
                policy_version=shared_state.model_info.llm.version + 1,
                optimizer_steps=shared_state.optimizer_steps,
            )
            shared_state.model_info.llm.update_sampling_client(sampling_client)
            trainer._publish_worker_binding(train_workflow, sampling_client, shared_state.model_info.llm.version)
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
                **admissions.step_metrics(prefixes=prefixes, counted=step_counted, admitted=step_admitted, draws=draws)
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
        step_stats["optim/lr"] = trainer.config.train.optimizer.learning_rate
        step_stats["timing/batch_total"] = batch_elapsed
        # Non-zero ``ptc_threads_leaked`` means rollout worker threads are gone
        # for good; see ``_ptc_cell_guard_stats``.
        step_stats.update(_ptc_cell_guard_stats("train"))
        # Client-side throughput: concurrency actually achieved, the queue
        # wait the trainer experienced, where an agent step's wall clock
        # went, and whether this process (plus its rollout workers) is CPU
        # saturated.  See ``platoon.train.tinker.rollout_metrics``.
        step_stats.update(client_step_metrics("train"))
        if (pruner := getattr(trainer, "_recovery_pruner", None)) is not None:
            # Recovery checkpoint pruning: paths deleted and deletes failed since the last step, and what is kept.
            step_stats.update(pruner.step_metrics())
        shared_state.stats_logger.log(step=shared_state.train_step, stats=step_stats)

        logger.info(
            f"Step {shared_state.train_step}/{end_batch} | "
            f"Epoch {shared_state.train_step // shared_state.num_train_batches_per_epoch} | "
            f"Batch time: {batch_elapsed:.1f}s | "
            f"Datums: {total_datums}"
        )

    await trainer._wait_for_last_step_evaluation(shared_state)
    await trainer.shutdown_loops(shared_state)


async def shutdown_loops(trainer: "PlatoonTinkerRLTrainer", shared_state: TrainLoopSharedState):
    shared_state.shutdown_event.set()
    shared_state.sampling_client_updated_event.set()
    raise TerminateTrainLoop()
