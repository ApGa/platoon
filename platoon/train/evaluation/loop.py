"""The evaluation loop beside training (:class:`EvaluationLoop`): each due step evaluated on the eval split by its
workers (:func:`eval_worker`, as many as the groups an evaluation runs at once) and its collector
(:func:`_eval_completion_monitor`); the first update held for step 0's evaluation, and the last step's evaluation let
end before the loops stop.  What they read of the trainer and of the loops' shared state is
:mod:`platoon.train.evaluation.protocol`'s.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tinker

from platoon.train.dispatch.permits import _Rows, _rows
from platoon.train.dispatch.runner import _group_done, run_group_guarded
from platoon.train.dispatch.settled import (
    LABEL_STATS,
    SettledGroup,
    _count_every_reward_key,
    _no_groups_yet,
    _output_label,
    _record_group_stats,
    as_settled_group,
)
from platoon.train.evaluation.protocol import EvaluatedTrainer, LoopState
from platoon.train.evaluation.record import record_evaluation
from platoon.train.labels import label_column, metric_labels, metric_prefix
from platoon.train.policy.binding import _publish_worker_binding, publish_other_bindings
from platoon.train.tinker.rollout_metrics import monitor as client_monitor
from platoon.train.trainer.signals import TerminateEvalLoop, TrainInterrupted
from platoon.train.workflow.protocol import RolloutWorkflow, report_of
from platoon.utils.rollout_stats import RolloutStats
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.stats_tracker import get as get_tracker

logger = logging.getLogger(__name__)


async def eval_worker(
    shared_state: LoopState,
    workflow: RolloutWorkflow,
    row_queue: asyncio.Queue[dict],
    output_queue: asyncio.Queue[Any],
    *,
    tracker: StatsTracker,
    rows: _Rows | None = None,
):
    """One of an evaluation's workers, the tasks that run its groups (training's groups start through the dispatcher;
    a rollout worker process is another thing): run the groups of the rows ``row_queue`` gives, one at a time, until a
    shutdown is requested, record their statistics in ``tracker``, and put each output on ``output_queue``.  With
    ``rows``, a group first takes its entry's group size in rollout permits, and its output goes with the entry and the
    label of the row it ran (:class:`~platoon.train.dispatch.settled.SettledGroup`)."""

    while not shared_state.shutdown_event.is_set():
        data = await row_queue.get()
        entry, label, group_size = rows.of(data) if rows is not None else (None, None, 0)
        reservation = rows.permits.reserve(group_size) if rows is not None else None
        try:
            if reservation is not None:
                await reservation.ready()
            start_time = time.perf_counter()
            output = await run_group_guarded(workflow, data, watchdog=shared_state.watchdog, tracker=tracker)
        finally:
            if reservation is not None:
                reservation.withdraw()

        task_id = data.get("task_id", "unknown")
        _group_done(output, time.perf_counter() - start_time, task_id, tracker, shared_state.watchdog)
        output_queue.put_nowait(SettledGroup(entry, label, output) if rows is not None else output)


async def _eval_completion_monitor(
    total_eval_tasks: int,
    output_queue: asyncio.Queue[Any],
    tracker: StatsTracker | None = None,
    label_stats: dict[str, RolloutStats] | None = None,
):
    """The evaluation's collector: wait for the output of each of its ``total_eval_tasks`` groups (``output_queue``),
    then end the evaluation's task group; each group's rollout statistics go into its label's ``label_stats``, when its
    label's metrics are logged."""
    for _ in range(total_eval_tasks):
        settled = as_settled_group(await output_queue.get())
        row_label, output = settled.label, settled.output
        if tracker is not None:
            _record_group_stats(output, tracker)
        rollout_stats = report_of(output).rollout_stats
        label = row_label if row_label is not None else _output_label(output)
        if label_stats is not None and label in label_stats and rollout_stats is not None:
            label_stats[label].merge(rollout_stats.subset(LABEL_STATS))
    raise TerminateEvalLoop()


async def _eval_loop(trainer: EvaluatedTrainer, shared_state: LoopState, workflow: RolloutWorkflow):
    """The evaluation loop of a start that evaluates during training (:class:`EvaluationLoop`), until a shutdown."""

    await EvaluationLoop(trainer, shared_state, workflow, Publication.of(shared_state)).run()


@dataclass(frozen=True)
class Publication:
    """What the evaluation loop reads of the steps and of the policy that training publishes for the agents' default
    (``agents.policy``): the event each step's end sets (``step_ended``, also a step's that published no policy, and the
    loops' stop), and the policy's version and sampling client as they are when it reads them."""

    step_ended: asyncio.Event
    version: Callable[[], int]
    sampling_client: Callable[[], tinker.SamplingClient]

    @classmethod
    def of(cls, shared_state: LoopState) -> "Publication":
        """The publication of the agents' default policy among the run's policies that the train and evaluation loops
        share (``shared_state``): the policy the loops train, unless ``agents.policy`` names another."""

        policies = shared_state.policies
        if policies.agents.default == policies.leader:
            return cls(
                step_ended=shared_state.step_ended,
                version=lambda: shared_state.model_info.llm.version,
                sampling_client=lambda: shared_state.sampling_client,
            )
        default = policies[policies.agents.default]
        return cls(
            step_ended=shared_state.step_ended,
            version=lambda: default.version,
            sampling_client=lambda: default.sampling_client,
        )


class EvaluationLoop:
    """The evaluation loop beside the train loop: :meth:`run` wakes at each step's end (``publication``) and evaluates
    the policy of the step training reached when that step's evaluation is due (:meth:`evaluate`),
    then records the step as handled, which training waits for: it holds its first update until step 0 is handled,
    and lets the last step's evaluation end before the loops stop (:func:`_wait_for_starting_policy_evaluation`,
    :func:`_wait_for_last_step_evaluation`)."""

    def __init__(
        self,
        trainer: EvaluatedTrainer,
        shared_state: LoopState,
        workflow: RolloutWorkflow,
        publication: Publication,
    ) -> None:
        self.trainer, self.state, self.workflow, self.publication = trainer, shared_state, workflow, publication
        self.tracker = get_tracker("eval")
        # Each evaluation's metrics break down by these labels of the evaluation's tasks (train.label_by), as a
        # training step's do: eval/<column>/<label>/...
        self.column = label_column(trainer.config)
        self.labels = metric_labels(trainer.eval_dataset, self.column, split="evaluation")
        self.reward_keys: set[str] = set()

    async def run(self) -> None:
        """Until a shutdown: wait for the next step's end, then evaluate the step training reached when its evaluation
        is due, and record the step as handled."""

        shared_state, step_ended = self.state, self.publication.step_ended
        while not shared_state.shutdown_event.is_set():
            await step_ended.wait()

            # Check shutdown again after waking up
            if shared_state.shutdown_event.is_set():
                break

            # Clear the event so we wait for the next step's end
            step_ended.clear()

            step = shared_state.train_step
            # Every eval_every steps: step 0, the starting policy, only with eval.at_start, and a start that resumed
            # each multiple from the step it resumed at on, that one unless an earlier start completed its evaluation.
            if shared_state.evaluation_due(step):
                await self.evaluate(step)

            shared_state.eval_handled_step = step
            shared_state.eval_progress_event.set()

    async def evaluate(self, step: int) -> None:
        """Evaluate the policy of ``step`` on the eval split: every row queued at once, the groups run by the
        evaluation's workers until the collector has every one (:meth:`_run_groups`), then the evaluation's metrics,
        each label's among them, logged at the step training reached by its end, and the evaluation recorded."""

        shared_state, workflow = self.state, self.workflow
        eval_start = time.perf_counter()
        versions = _versions(shared_state.policies)  # in a run of several trained policies
        logger.info(f"Starting evaluation at step {shared_state.train_step}")
        limited = getattr(workflow, "start_limit", None) is not None  # train.max_starting_rollouts
        if limited:  # its start-limit metrics cover this evaluation
            client_monitor("eval").start_limit_metrics(reset=True)
        # Rollout workers read the sampling client from its published binding, which training publishes for
        # its own: an evaluation whose rollouts run in worker processes while training's do not needs it too.
        if getattr(workflow, "rollout_isolation", "inprocess") == "process":
            version = self.publication.version()
            _publish_worker_binding(workflow, self.publication.sampling_client(), version)
            publish_other_bindings(shared_state.policies, workflow, _publish_worker_binding)  # the others they sample

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
        independent = getattr(shared_state, "clocks", None) is not None  # the policies step each on its own clock
        if independent:  # a row of its own, on the run's axis (the leader's steps), as the leader's rows are
            eval_stats["run/step"] = shared_state.train_step
            shared_state.stats_logger.log_row(eval_stats, axis="run/step")
        else:
            # Use force=True since train already logged at this step
            shared_state.stats_logger.log(step=shared_state.train_step, stats=eval_stats, force=True)
        # Stepping independently, the policies may publish while the evaluation runs: their versions at its end.
        moved = {"versions_end": end} if independent and (end := _versions(shared_state.policies)) != versions else {}
        if shared_state.evaluations_path is not None:  # so that a start that resumes at step does not rerun it
            record_evaluation(
                shared_state.evaluations_path,
                step=step,
                logged_at=shared_state.train_step,
                tasks=total_eval_tasks,
                seconds=eval_elapsed,
                **({"versions": versions} if versions else {}),
                **moved,
            )

    async def _run_groups(self) -> tuple[int, dict[str, RolloutStats]]:
        """Queue every row of the eval split, then run its groups: a worker (:func:`eval_worker`) for each group that
        the evaluation's rollout permits (``eval.max_concurrent_rollouts``) let run at once, and the collector, which
        ends the evaluation once it has every group's output.  The rows queued, and each label's rollout statistics."""

        shared_state, config = self.state, self.trainer.config
        row_queue = asyncio.Queue[dict]()
        output_queue = asyncio.Queue[Any]()  # each group's SettledGroup

        # Populate eval queue with all eval data upfront
        eval_dataloader = shared_state.eval_dataloader
        assert eval_dataloader is not None  # an evaluation runs only over the run's evaluation split
        total_eval_tasks = 0
        for i_batch in range(len(eval_dataloader)):
            batch = eval_dataloader.get_batch(i_batch)
            for data in batch:
                row_queue.put_nowait(data)
                total_eval_tasks += 1

        rows, num_workers = _rows(config, "eval", config.eval.max_concurrent_rollouts)
        label_stats = {label: _no_groups_yet() for label in self.labels}

        # Run eval with TaskGroup - completion monitor will raise TerminateEvalLoop when done
        try:
            async with asyncio.TaskGroup() as tg:
                # Spawn workers with eval tracker
                for i in range(num_workers):
                    tg.create_task(
                        eval_worker(
                            shared_state,
                            self.workflow,
                            row_queue,
                            output_queue,
                            tracker=self.tracker,
                            rows=rows,
                        ),
                        name=f"eval_worker_{i}",
                    )
                # Spawn completion monitor
                tg.create_task(
                    _eval_completion_monitor(
                        total_eval_tasks,
                        output_queue,
                        tracker=self.tracker,
                        label_stats=label_stats,
                    ),
                    name="eval_completion_monitor",
                )
        except* TerminateEvalLoop:
            pass  # Expected - eval completed successfully
        return total_eval_tasks, label_stats


def _versions(policies: Any) -> dict[str, int]:
    """Each policy the agents sample at its version now, by policy, in a run that trains several policies, whose
    evaluations record them (an evaluation samples each policy at its version then); none in any other run, whose
    evaluations record its one trained policy's step."""

    if len(policies.trainable()) < 2:
        return {}
    return {policy.id: policy.version for policy in policies.sampled()}


async def _wait_for_starting_policy_evaluation(shared_state: LoopState) -> None:
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


async def _wait_for_last_step_evaluation(shared_state: LoopState) -> None:
    """Let the evaluation loop handle the last training step before the loops stop.

    Stopping the loops cancels the evaluation loop, so without this wait an evaluation due at the last step
    never ran, and one still running from an earlier step was cancelled.  A start that resumed a run at its last
    step trains none, and waits for the evaluation of that step if it is due: one that a crash cut short, which
    the run's evaluations.jsonl does not list.  There is nothing to wait for without an evaluation loop.  A
    shutdown request ends the wait at once: the loops then stop, cancelling the evaluation, and the
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
