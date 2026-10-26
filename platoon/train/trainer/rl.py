"""The trainer (``PlatoonTinkerRLTrainer``): one start of training, from its training client and resume point
through the train and evaluation loops to the last save.  ``train()`` reads as the start's phases, each a method of the
trainer, in the order they run: the checks, the training client at the start point, the rollouts' startup, the draws
taken up, the launch save, what the loops share, the loops, the save at exit.

What it does at each stage lives in the stage packages (``platoon.train.trainer``, ``.step``, ``.dispatch``,
``.checkpoints``, ``.evaluation``): each of those methods is a module function there that takes the trainer first,
and stays a method of the trainer under its name, so that subclasses and tests that replace one keep working.
"""

import asyncio
import logging
import os
from contextlib import ExitStack
from typing import Any

import tinker
from datasets import Dataset
from tinker_cookbook import checkpoint_utils

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet
from platoon.train.checkpoints import resume, saving
from platoon.train.checkpoints.pruning import (
    RECOVERY_CHECKPOINT_PREFIX,
    RecoveryCheckpointPruner,
    check_recovery_pruning,
    read_checkpoint_records,
)
from platoon.train.checkpoints.resume import StartPoint
from platoon.train.components import normalize_environment_configs
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, evaluates
from platoon.train.dataloader import PlatoonTinkerDataloader, _run_steps
from platoon.train.dataset_order import EpochOrder
from platoon.train.dispatch import runner, setup
from platoon.train.dispatch.dispatcher import Dispatcher
from platoon.train.dispatch.permits import _group_sizes
from platoon.train.evaluation import loop as evaluation_loop
from platoon.train.evaluation import plan as evaluation_plan
from platoon.train.evaluation.plan import _evaluation_plan
from platoon.train.evaluation.record import EVALUATIONS_NAME, completed_evaluation_steps
from platoon.train.sampler_runtime import TaskSampling
from platoon.train.startup import RolloutStartup
from platoon.train.step import publish
from platoon.train.step.batch_transforms import BatchTransform, build_default_batch_transforms
from platoon.train.task_records import TaskRecorder, record_lost_state, resume_record
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.proxy import ModelInfo, register_tinker_llm
from platoon.train.trainer import loops, service, signals, waiting
from platoon.train.trainer.signals import TerminateTrainLoop, TrainInterrupted, Watchdog
from platoon.train.trainer.state import TrainLoopSharedState
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.protocol import RolloutWorkflow
from platoon.train.workflow.start_limit import StartLimit
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import StatsTracker
from platoon.utils.stats_tracker import get as get_tracker

logger = logging.getLogger(__name__)


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


class PlatoonTinkerRLTrainer:
    # Deletes the run's older recovery checkpoints while train() runs (checkpoint.keep_recovery); None without it.
    _recovery_pruner: RecoveryCheckpointPruner | None = None
    # Starts the training groups of the start train() runs (platoon.train.dispatch.dispatcher); None before it.
    _dispatcher: Dispatcher | None = None
    # The order training draws its tasks in, epoch by epoch, such as a mixture's weighted order; None: the dataloader
    # shuffles train_dataset (platoon.train.dataset_order).  The trainer counts its epochs by it.
    train_order: EpochOrder | None = None
    # How the run draws its training tasks through task samplers, as platoon train hands it the run's
    # (AutoDataset.training); None: each position of the dataloader's order, as it is.
    train_sampling: TaskSampling | None = None
    # The group admission rules of the start train() runs, beside its dispatcher (platoon.train.dispatch.setup); None
    # when every entry admits every group.  Unset before (read with getattr).
    _admissions: AdmissionSet | None

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

    # Methods whose code lives in the stage packages, as module functions that take the trainer first.  Each stays a
    # method under its name, so that replacing it on the class or on an instance (a subclass, a test) replaces what the
    # trainer calls; replacing the module function does not.  Here: waiting, running groups, the loops, evaluation and
    # the checkpoint cadence.
    _await_with_heartbeat = waiting._await_with_heartbeat
    _await_rollout_with_heartbeat = waiting._await_rollout_with_heartbeat
    _run_group = runner._run_group
    _rollout_workflow_worker_loop = evaluation_loop._rollout_workflow_worker_loop
    _run_training_group = runner._run_training_group
    _settle_under_rule = runner._settle_under_rule
    _publish_worker_binding = staticmethod(publish._publish_worker_binding)
    _train_loop = loops._train_loop
    _next_rollout = staticmethod(waiting._next_rollout)
    _eval_completion_monitor = evaluation_loop._eval_completion_monitor
    _eval_loop = evaluation_loop._eval_loop
    _wait_for_starting_policy_evaluation = evaluation_loop._wait_for_starting_policy_evaluation
    _wait_for_last_step_evaluation = evaluation_loop._wait_for_last_step_evaluation
    _evaluates_during_training = evaluation_plan._evaluates_during_training
    shutdown_loops = loops.shutdown_loops
    _save_checkpoint_and_get_sampling_client = saving._save_checkpoint_and_get_sampling_client

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

    # Moved methods, as above: what a full checkpoint holds, the dispatcher, and resuming.
    _checkpoint_state = saving._checkpoint_state
    _dispatch = setup._dispatch
    _new_dispatcher = setup._new_dispatcher
    _samplers = setup._samplers
    _resume_position = staticmethod(resume._resume_position)
    _restore_dispatch = resume._restore_dispatch
    _saved_samplings = staticmethod(resume._saved_samplings)
    _read_kept_groups = resume._read_kept_groups

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

    # Moved methods, as above: the training client and the signal handlers.
    _create_training_client = service._create_training_client
    _open_training_client = service._open_training_client
    _setup_signal_handlers = signals._setup_signal_handlers

    async def train(self, train_workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow | None = None):
        """Train this start of the run, from the checkpoint it resumes from (:class:`StartPoint`; none for a fresh run)
        to the run's last step: open the training service and the training client, take up the draws the checkpoint
        left outstanding, save the launch checkpoint, then run the train loop, and beside it the evaluation loop when
        the start evaluates, and save once they stop.  Each phase is a method below, in the order it runs."""

        self._require_entered()
        self._check_before_start()
        shutdown_event = asyncio.Event()

        # Set up signal handlers for graceful Ctrl+C shutdown
        self._setup_signal_handlers(shutdown_event)

        # A key the service refuses (401, 403) ends training with one line that names the key and the service.
        with key_refusals():
            service_client = tinker.ServiceClient(base_url=self.config.tinker_base_url)
        start = StartPoint.read(self.run_log_path, self.config.train.num_minibatches)
        with key_refusals(asked="a training client"):
            training_client = await self._open_training_client(service_client, start.resume_info, stop=shutdown_event)
        if training_client is None:
            return  # stopped before training began: nothing to save

        startup = await self._begin_rollout_startup(train_workflow, eval_workflow)
        train_dataloader, end_batch = self._prepare_draws(start)
        sampling_client = await self._launch(training_client, start)
        # Also waits for the staging of the harness toolchains (startup.STAGING_WAIT_S at most), then starts the
        # watchdog, which _run_loops stops.
        shared_state = await self._loop_state(
            start, shutdown_event, service_client, training_client, sampling_client, train_dataloader, startup
        )
        interrupted = await self._run_loops(shared_state, start, end_batch, train_workflow, eval_workflow)
        await self._save_at_exit(shared_state, start.batch, end_batch, interrupted)
        await self._close_after_training()

    def _require_entered(self) -> None:
        """train() runs inside ``async with trainer:``, which opens the stats logger and the train tracker."""

        if self._stats_logger is None or self._train_tracker is None:
            raise RuntimeError(
                "Trainer must be used as an async context manager. Use 'async with trainer:' before calling train()."
            )

    def _check_before_start(self) -> None:
        """The config's checks before training opens its service: ``checkpoint.keep_recovery``'s rules, and a warning
        that ``eval.at_start`` is ignored when evaluation during training is off."""

        # keep_recovery's rules again, before training opens its service: a config class that overrides the training
        # config's own checks loads without them (platoon train checks them first, in training_problems).
        check_recovery_pruning(self.config)
        if getattr(self.config.eval, "at_start", False) and not evaluates(self.config):
            logger.warning(
                "eval.at_start is ignored: eval.every is 0, which turns evaluation during training off, the starting "
                "policy's too"
            )

    async def _begin_rollout_startup(
        self, train_workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow | None
    ) -> RolloutStartup:
        """Begin readying the rollouts, ahead of the first one, while the trainer saves at launch: the toolchains the
        environments' harnesses run from are staged on this node, and the rollout workers' fork servers start
        importing (:mod:`platoon.train.startup`), the evaluation's only when the start evaluates.  No rollout starts
        before the staging is done (:meth:`_loop_state` waits for it)."""

        return await RolloutStartup.begin(
            normalize_environment_configs(getattr(self.config, "environments", None)),
            [train_workflow, eval_workflow if self.eval_dataset is not None and evaluates(self.config) else None],
        )

    def _prepare_draws(self, start: StartPoint) -> tuple[PlatoonTinkerDataloader, int]:
        """Start the run's per-task statistics and the dispatcher of the training groups, both before the launch
        save.  The training dataloader, and the end batch: the run's last step."""

        # The run's per-task statistics, those of the checkpoint it resumes from loaded before the launch save, which
        # saves them: a crash before the next full save loses none of them.
        self._begin_task_records(start.resume_info, start.batch)
        # The dispatcher of the training groups, which takes up the draws the checkpoint left outstanding before the
        # launch save, which saves them again: its kept groups, the draws to draw again, and the order's position.
        train_dataloader = PlatoonTinkerDataloader(
            self.train_dataset, self.config.train.batch_size, order=self.train_order
        )
        end_batch = _run_steps(self.config, len(train_dataloader))
        self._dispatcher = self._new_dispatcher(train_dataloader, start.batch, end_batch)
        if start.resume_info is not None:
            self._restore_dispatch(self._dispatcher, start.resume_info, start.policy_version)
        return train_dataloader, end_batch

    async def _launch(self, training_client: tinker.TrainingClient, start: StartPoint) -> tinker.SamplingClient:
        """Save the launch checkpoint, then start from it: the per-task statistics' events, the pruning of the run's
        recovery checkpoints, and the first sampling client, of the launch save's weights, which the policy samples at
        the checkpoint's policy version.  The sampling client."""

        # Initial sampling client to use (the watchdog is not running yet). The
        # launch save is a *recovery* checkpoint: every (re)launch writes one, so
        # naming it permanently would accumulate a ~490 GB checkpoint per restart.
        launch_checkpoint_name = f"{RECOVERY_CHECKPOINT_PREFIX}{start.batch:06d}"
        logger.info(f"Saving recovery checkpoint {launch_checkpoint_name} at launch (batch {start.batch})")
        path_dict = await saving.save_run_checkpoint(
            self,
            training_client,
            launch_checkpoint_name,
            batch=start.batch,
            policy_version=start.policy_version,
            optimizer_steps=start.optimizer_steps,
            wait=self._await_with_heartbeat,
        )
        self._task_records_started()
        # Pruning starts with the launch save, so a resumed run first deletes what its earlier starts left.
        self._recovery_pruner = self._start_recovery_pruning()
        self._recovery_saved(path_dict, start.policy_version)
        # Not saving.sampler_from, whose log lines the launch never wrote.
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])

        # Set initial version and update sampling client (without auto-incrementing)
        self.model_info.llm.set_version(start.policy_version)
        self.model_info.llm.update_sampling_client(sampling_client, increment_version=False)
        return sampling_client

    async def _loop_state(
        self,
        start: StartPoint,
        shutdown_event: asyncio.Event,
        service_client: tinker.ServiceClient,
        training_client: tinker.TrainingClient,
        sampling_client: tinker.SamplingClient,
        train_dataloader: PlatoonTinkerDataloader,
        startup: RolloutStartup,
    ) -> TrainLoopSharedState:
        """What the train and evaluation loops share (:class:`TrainLoopSharedState`), once the harness toolchains are
        staged (``startup``): the optimizer's settings, the evaluation's dataloader, the clients, the first sampling
        client (new: the evaluation loop's first wait passes at once), the watchdog, started when it is enabled, and
        where the start begins."""

        assert self._stats_logger is not None and self._train_tracker is not None  # train() checked (_require_entered)
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

        sampling_client_updated_event = asyncio.Event()
        sampling_client_updated_event.set()
        return TrainLoopSharedState(
            config=self.config,
            shutdown_event=shutdown_event,
            train_step=start.batch,
            optimizer_params=optimizer_params,
            service_client=service_client,
            training_client=training_client,
            sampling_client=sampling_client,
            sampling_client_step=start.policy_version,
            optimizer_steps=start.optimizer_steps,
            sampling_client_updated_event=sampling_client_updated_event,
            model_info=self._model_info,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            stats_logger=self._stats_logger,
            train_tracker=self._train_tracker,
            watchdog=watchdog,
            start_step=start.batch,
            # A start that resumes does not run again the evaluation of the step it resumes at when an earlier start
            # completed it; a fresh run has none to leave out.
            start_step_evaluated=start.resume_info is not None
            and start.batch in completed_evaluation_steps(self.run_log_path),
            evaluations_path=os.path.join(self.run_log_path, EVALUATIONS_NAME),
        )

    async def _run_loops(
        self,
        shared_state: TrainLoopSharedState,
        start: StartPoint,
        end_batch: int,
        train_workflow: RolloutWorkflow,
        eval_workflow: RolloutWorkflow | None,
    ) -> bool:
        """Run the train loop to ``end_batch``, and beside it the evaluation loop when the start evaluates (its plan
        logged), under this start's limit on the rollouts starting at once, until the loops stop; then stop the
        watchdog.  Whether training was interrupted (a signal, a cancellation, a stop before an update)."""

        shared_state.eval_loop_running = self._evaluates_during_training(shared_state)
        if shared_state.eval_loop_running:
            logger.log(*_evaluation_plan(shared_state, resumed=start.resume_info is not None, end=end_batch))
        _limit_starts(self.config, train_workflow, eval_workflow if shared_state.eval_loop_running else None)

        watchdog = shared_state.watchdog
        interrupted = False
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(
                    self._train_loop(tg, shared_state, start.batch, end_batch, train_workflow), name="train_loop"
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
        return interrupted

    async def _save_at_exit(
        self, shared_state: TrainLoopSharedState, start_batch: int, end_batch: int, interrupted: bool
    ) -> None:
        """Once the loops stop: an interrupted start that trained a step saves a permanent checkpoint to resume from
        (``<step>_interrupted``; one that trained none saves nothing), and a start that trained to ``end_batch`` the
        final one; a start that had no step left to train says why."""

        # Save checkpoint
        current_step = shared_state.train_step
        if interrupted:
            # Save interrupt checkpoint so we can resume
            if current_step > start_batch:
                logger.info(f"Saving permanent checkpoint {current_step:06d}_interrupted at step {current_step}...")
                await saving.save_run_checkpoint(
                    self,
                    shared_state.training_client,
                    f"{current_step:06d}_interrupted",
                    batch=current_step,
                    policy_version=shared_state.model_info.llm.version,
                    optimizer_steps=shared_state.optimizer_steps,
                    wait=self._await_with_heartbeat,
                )
                logger.info(f"Checkpoint saved. Resume training to continue from step {current_step}.")
        elif start_batch < end_batch:
            # Training completed normally - save final checkpoint
            logger.info(f"Saving permanent checkpoint final at step {end_batch}...")
            await saving.save_run_checkpoint(
                self,
                shared_state.training_client,
                "final",
                batch=end_batch,
                policy_version=shared_state.model_info.llm.version,
                optimizer_steps=shared_state.optimizer_steps,
                wait=self._await_with_heartbeat,
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

    async def _close_after_training(self) -> None:
        """Stop the pruning of the run's recovery checkpoints, and close the per-task statistics."""

        if self._recovery_pruner is not None:
            await self._recovery_pruner.aclose()
        if (recorder := getattr(self, "_recorder", None)) is not None:
            recorder.close()
