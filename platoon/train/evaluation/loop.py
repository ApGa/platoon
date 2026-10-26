"""The evaluation loop beside training (:class:`EvaluationLoop`): each due step evaluated on the eval split, the first
update held for step 0's evaluation, and the last step's evaluation let end before the loops stop.

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.  The evaluation loop calls
them through the trainer.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import tinker

from platoon.train.dispatch.permits import _Rows, _rows
from platoon.train.dispatch.runner import _group_done
from platoon.train.dispatch.settled import _output_label, _Settled, _settled
from platoon.train.evaluation.record import record_evaluation
from platoon.train.labels import label_column, metric_labels, metric_prefix
from platoon.train.step.metrics import LABEL_STATS, _count_every_reward_key, _no_groups_yet, _record_group_stats
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.trainer.signals import TerminateEvalLoop, TrainInterrupted
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.protocol import RolloutWorkflow, report_of
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.stats_tracker import get as get_tracker

if TYPE_CHECKING:
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

logger = logging.getLogger(__name__)


async def _rollout_workflow_worker_loop(
    trainer: "PlatoonTinkerRLTrainer",
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
            rollout = await trainer._run_group(workflow, data, watchdog=shared_state.watchdog, tracker=tracker)
        finally:
            if reservation is not None:
                reservation.withdraw()

        task_id = data.get("task_id", "unknown")
        _group_done(rollout, time.perf_counter() - start_time, task_id, tracker, shared_state.watchdog)
        rollout_result_queue.put_nowait(_Settled(entry, label, rollout) if rows is not None else rollout)


async def _eval_completion_monitor(
    trainer: "PlatoonTinkerRLTrainer",
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
        rollout_stats = report_of(rollout).rollout_stats
        label = row_label if row_label is not None else _output_label(rollout)
        if label_stats is not None and label in label_stats and rollout_stats is not None:
            label_stats[label].merge(rollout_stats.subset(LABEL_STATS))
    raise TerminateEvalLoop()


async def _eval_loop(trainer: "PlatoonTinkerRLTrainer", shared_state: TrainLoopSharedState, workflow: RolloutWorkflow):
    """The evaluation loop of a start that evaluates during training (:class:`EvaluationLoop`), until a shutdown."""

    await EvaluationLoop(trainer, shared_state, workflow, Publication.of(shared_state)).run()


@dataclass(frozen=True)
class Publication:
    """What the evaluation loop reads of the policy that training publishes: the event each publication sets
    (``updated``), and the policy's version and sampling client as they are when it reads them."""

    updated: asyncio.Event
    version: Callable[[], int]
    sampling_client: Callable[[], tinker.SamplingClient]

    @classmethod
    def of(cls, shared_state: TrainLoopSharedState) -> "Publication":
        """The publication of the policy that the train and evaluation loops share (``shared_state``)."""

        return cls(
            updated=shared_state.sampling_client_updated_event,
            version=lambda: shared_state.model_info.llm.version,
            sampling_client=lambda: shared_state.sampling_client,
        )


class EvaluationLoop:
    """The evaluation loop beside the train loop: :meth:`run` waits for each policy that training publishes
    (``publication``) and evaluates the step training reached when that step's evaluation is due (:meth:`evaluate`),
    then records the step as handled, which training waits for: it holds its first update until step 0 is handled,
    and lets the last step's evaluation end before the loops stop (:func:`_wait_for_starting_policy_evaluation`,
    :func:`_wait_for_last_step_evaluation`)."""

    def __init__(
        self,
        trainer: "PlatoonTinkerRLTrainer",
        shared_state: TrainLoopSharedState,
        workflow: RolloutWorkflow,
        publication: Publication,
    ) -> None:
        self.trainer, self.state, self.workflow, self.publication = trainer, shared_state, workflow, publication
        self.tracker = get_tracker("eval")
        # Each evaluation's metrics break down by these labels of the evaluation's tasks (train.label_by), as a
        # training step's do: eval/<column>/<label>/...
        self.column = label_column(trainer.config)
        self.labels = metric_labels(getattr(trainer, "eval_dataset", None), self.column, split="evaluation")
        self.reward_keys: set[str] = set()

    async def run(self) -> None:
        """Until a shutdown: wait for the next policy, then evaluate the step training reached when its evaluation is
        due, and record the step as handled."""

        shared_state, updated = self.state, self.publication.updated
        while not shared_state.shutdown_event.is_set():
            await updated.wait()

            # Check shutdown again after waking up
            if shared_state.shutdown_event.is_set():
                break

            # Clear the event so we wait for the next update
            updated.clear()

            step = shared_state.train_step
            # Every eval_every steps: step 0, the starting policy, only with eval.at_start, and a start that resumed
            # each multiple from the step it resumed at on, that one unless an earlier start completed its evaluation.
            if shared_state.evaluation_due(step):
                await self.evaluate(step)

            shared_state.eval_handled_step = step
            shared_state.eval_progress_event.set()

    async def evaluate(self, step: int) -> None:
        """Evaluate the policy of ``step`` on the eval split: every row queued at once, the groups run by the rollout
        workers until the collector has every one (:meth:`_run_groups`), then the evaluation's metrics, each label's
        among them, logged at the step training reached by its end, and the evaluation recorded."""

        shared_state, workflow = self.state, self.workflow
        eval_start = time.perf_counter()
        logger.info(f"Starting evaluation at step {shared_state.train_step}")
        limited = getattr(workflow, "start_limit", None) is not None  # train.max_starting_rollouts
        if limited:  # its start-limit metrics cover this evaluation
            client_monitor("eval").start_limit_metrics(reset=True)
        # Rollout workers read the sampling client from its published binding, which training publishes for
        # its own: an evaluation whose rollouts run in worker processes while training's do not needs it too.
        if getattr(workflow, "rollout_isolation", "inprocess") == "process":
            version = self.publication.version()
            self.trainer._publish_worker_binding(workflow, self.publication.sampling_client(), version)

        total_eval_tasks, label_stats = await self._run_groups()

        eval_elapsed = time.perf_counter() - eval_start
        logger.info(f"Evaluation completed in {eval_elapsed:.1f}s for {total_eval_tasks} tasks")

        # Export and log eval stats, each label's among them
        _count_every_reward_key(self.reward_keys, label_stats.values())
        for label, stats in label_stats.items():
            stats.record(self.tracker, prefix=metric_prefix(self.column, label))
        eval_stats = self.tracker.export(reset=True)
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

    async def _run_groups(self) -> tuple[int, dict[str, RolloutStats]]:
        """Queue every row of the eval split, then run its groups: a rollout worker for each group that the
        evaluation's rollout permits (``eval.max_concurrent_rollouts``) let run at once, and the collector, which ends
        the evaluation once it has every group's output.  The rows queued, and each label's rollout statistics."""

        shared_state, trainer = self.state, self.trainer
        eval_data_queue = asyncio.Queue[dict]()
        eval_rollout_result_queue = asyncio.Queue[Any]()  # each group's _Settled

        # Populate eval queue with all eval data upfront
        total_eval_tasks = 0
        for i_batch in range(len(shared_state.eval_dataloader)):
            batch = shared_state.eval_dataloader.get_batch(i_batch)
            for data in batch:
                eval_data_queue.put_nowait(data)
                total_eval_tasks += 1

        rows, num_workers = _rows(trainer.config, "eval", trainer.config.eval.max_concurrent_rollouts)
        label_stats = {label: _no_groups_yet() for label in self.labels}

        # Run eval with TaskGroup - completion monitor will raise TerminateEvalLoop when done
        try:
            async with asyncio.TaskGroup() as tg:
                # Spawn workers with eval tracker
                for i in range(num_workers):
                    tg.create_task(
                        trainer._rollout_workflow_worker_loop(
                            shared_state,
                            self.workflow,
                            eval_data_queue,
                            eval_rollout_result_queue,
                            tracker=self.tracker,
                            rows=rows,
                        ),
                        name=f"eval_rollout_workflow_worker_{i}",
                    )
                # Spawn completion monitor
                tg.create_task(
                    trainer._eval_completion_monitor(
                        total_eval_tasks,
                        eval_rollout_result_queue,
                        tracker=self.tracker,
                        label_stats=label_stats,
                    ),
                    name="eval_completion_monitor",
                )
        except* TerminateEvalLoop:
            pass  # Expected - eval completed successfully
        return total_eval_tasks, label_stats


async def _wait_for_starting_policy_evaluation(
    trainer: "PlatoonTinkerRLTrainer", shared_state: TrainLoopSharedState
) -> None:
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
        await _wait_for_event_or_stop(shared_state.eval_progress_event, shared_state.shutdown_event)


async def _wait_for_last_step_evaluation(trainer: "PlatoonTinkerRLTrainer", shared_state: TrainLoopSharedState) -> None:
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
        await _wait_for_event_or_stop(shared_state.eval_progress_event, shared_state.shutdown_event)


async def _wait_for_event_or_stop(event: asyncio.Event, stop: asyncio.Event) -> None:
    """Wait until ``event`` or ``stop`` is set: the evaluation loop's progress, or a shutdown request."""

    waiting = asyncio.ensure_future(event.wait())
    stopping = asyncio.ensure_future(stop.wait())
    try:
        await asyncio.wait((waiting, stopping), return_when=asyncio.FIRST_COMPLETED)
    finally:
        waiting.cancel()
        stopping.cancel()
