"""The rl trainer (``PlatoonTinkerRLTrainer``, ``trainer: rl``): GRPO-style training on the rollouts of the run's
tasks, with evaluation during training.

Each start of a run (its first launch, or a resume) goes through :class:`~platoon.train.trainer.run.TrainerRun`, the
lifecycle every trainer shares.  ``train()`` runs it with the workflows it is given; ``run()`` builds them from the
trainer's setup first.  The lifecycle calls this trainer's hooks in this order (the numbers are ``TrainerRun``'s):

1. ``check``: the trainer's checks;
4. ``_open_policies``: the policy's training client at the start point, through the trainer's seams;
5. ``prepare``: the rollouts' startup, and the start's draws (:class:`_TrainerDraws`, its rollout source);
6. ``started``, after the launch save: the recovery checkpoint pruning;
7. ``ready``: the state the loops share;
8. ``run_steps``: the training and evaluation loops;
10. ``close``.

The code of each stage lives in a stage package (``platoon.train.trainer``, ``.step``, ``.dispatch``,
``.checkpoints``, ``.evaluation``), as a module function that takes the trainer first.  Each stays a method of the
trainer under its name, so that subclasses and tests that replace one keep working.  The functions of the start's
draws take the rollout source instead of the trainer; the source calls them through the trainer's methods of the
same names.
"""

import asyncio
import logging
import os
from typing import Any

import tinker
from datasets import Dataset
from tinker_cookbook import checkpoint_utils

from platoon.train import entry_settings
from platoon.train.admissions import AdmissionSet
from platoon.train.checkpoints import resume, saving
from platoon.train.checkpoints.participants import RunCheckpoints
from platoon.train.checkpoints.pruning import RecoveryCheckpointPruner, check_recovery_pruning
from platoon.train.checkpoints.resume import StartPoint
from platoon.train.components import normalize_environment_configs
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, evaluates
from platoon.train.dataloader import PlatoonTinkerDataloader
from platoon.train.dataset_order import EpochOrder
from platoon.train.dispatch import runner, setup
from platoon.train.dispatch import source as rollout_source
from platoon.train.dispatch.dispatcher import Dispatcher
from platoon.train.dispatch.permits import _group_sizes
from platoon.train.dispatch.source import RolloutSource, training_tasks
from platoon.train.evaluation import loop as evaluation_loop
from platoon.train.evaluation import plan as evaluation_plan
from platoon.train.evaluation.plan import _evaluation_plan
from platoon.train.evaluation.record import EVALUATIONS_NAME, completed_evaluation_steps
from platoon.train.policy import service
from platoon.train.policy.policy import Policy, adam_params
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, policy_specs
from platoon.train.sampler_runtime import SamplerSet, TaskSampling
from platoon.train.startup import RolloutStartup
from platoon.train.step import publish
from platoon.train.step.batch_transforms import BatchTransform, build_default_batch_transforms
from platoon.train.task_records import TaskRecorder
from platoon.train.tinker.key_refusal import key_refusals
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.trainer import loops, waiting
from platoon.train.trainer.api import StepPlan, TaskSources, TrainerBase, TrainerNeeds, TrainerSetup, Workflows
from platoon.train.trainer.run import StepClock, TrainerRun
from platoon.train.trainer.signals import TerminateTrainLoop, TrainInterrupted, Watchdog
from platoon.train.trainer.state import TrainLoopSharedState, _main_holding
from platoon.train.workflow import output as _group_output  # noqa: F401 - registers GroupOutput's kept-group codec
from platoon.train.workflow.group import GroupRolloutWorkflow  # noqa: F401 - the platoon.train.tinker.rl shim reads it
from platoon.train.workflow.protocol import RolloutWorkflow, capabilities_of
from platoon.train.workflow.start_limit import StartLimit
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import DeferredStats  # noqa: F401 - tinker.rl's shim reads DeferredStats

logger = logging.getLogger(__name__)


def _limit_starts(config: Any, train_workflow: Any, eval_workflow: Any | None) -> None:
    """Set this start's limit on the rollouts starting at once (``train.max_starting_rollouts``; None when it is off)
    as the ``start_limit`` of the training workflow and of ``eval_workflow``, the evaluation's that runs during
    training, where each takes one (``takes_start_limit``, as a ``GroupRolloutWorkflow`` does), so that both share it,
    first come, first served.  Off, their ``start_limit`` is None, so that a workflow an earlier start used keeps no
    limit."""

    limit = config.train.max_starting_rollouts
    gate = StartLimit(limit) if limit is not None else None
    for workflow in (train_workflow, eval_workflow):
        if workflow is not None and capabilities_of(workflow).takes_start_limit:
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
        gated=capabilities_of(train_workflow).takes_start_limit,
    )


class PlatoonTinkerRLTrainer(TrainerBase):
    # Deletes the run's older recovery checkpoints while train() runs (checkpoint.keep_recovery); None without it.
    _recovery_pruner: RecoveryCheckpointPruner | None = None
    # The draws of the start train() runs, its rollout source (_TrainerDraws), which prepare() opens; None before it.
    # Its per-task statistics, dispatcher and admission rules are _recorder, _dispatcher and _admissions.
    _rollouts: "_TrainerDraws | None" = None
    # The order training draws its tasks in, epoch by epoch, such as a mixture's weighted order; None: the dataloader
    # shuffles train_dataset (platoon.train.dataset_order).  The trainer counts its epochs by it.
    train_order: EpochOrder | None = None
    # How the run draws its training tasks through task samplers, as platoon train hands it the run's
    # (AutoDataset.training); None: each position of the dataloader's order, as it is.
    train_sampling: TaskSampling | None = None
    # The run's policies (platoon.train.policy): main, as train: declares it, with its model gateway.
    policies: PolicySet
    # The start that train() runs (TrainerRun), which train() records before it begins, so that the steps save through
    # its checkpoints whatever a subclass's hooks do; None before the first.  As it runs: the rollouts' startup and the
    # training dataloader (prepare), and what its loops share (ready).
    _run: TrainerRun | None = None
    _startup: RolloutStartup
    _train_dataloader: PlatoonTinkerDataloader
    _shared_state: TrainLoopSharedState

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
        # The run's policies as the config declares them now (a later change of the config does not reach them), and
        # main's model gateway, this process's default, on the service train() opens main's training client on.
        self.policies = PolicySet.build(policy_specs(self.config))

    @classmethod
    def needs(cls, config: Any, options: Any) -> TrainerNeeds:
        """It trains on rollouts of the run's training tasks."""

        return TrainerNeeds(rollouts=True)

    @classmethod
    def from_setup(cls, setup: TrainerSetup) -> "PlatoonTinkerRLTrainer":
        """The trainer over the setup's tasks, the order they are drawn in and their task samplers, whose ``run()``
        builds its workflows with the setup.  Its constructor registers the policies its config declares, so a setup
        that declares others is refused.  (``platoon train`` builds this trainer with its constructor instead, and
        hands it the workflows it builds, ``train()``.)"""

        if dict(setup.policy_specs) != policy_specs(setup.config):
            raise ValueError("the rl trainer trains the policies its config declares, not the setup's other ones")
        tasks = setup.tasks
        assert tasks is not None  # platoon train lists the tasks of a trainer that rolls them out (needs)
        trainer = cls(
            config=setup.config,
            train_dataset=tasks.train_dataset,  # a weighted order's first epoch, or the listing the dataloader shuffles
            eval_dataset=tasks.eval_dataset,
            train_order=tasks.train_order,  # the epochs the trainer counts
            train_sampling=tasks.train_sampling,  # the task samplers it draws through, which draw the order's rows
        )
        trainer.setup = setup
        return trainer

    def _build_batch_transforms(
        self,
        extra_batch_transforms: list[BatchTransform] | None = None,
    ) -> list[BatchTransform]:
        """The batch transforms each microbatch of a step goes through, in order
        (:meth:`~platoon.train.trainer.loops.TrainStep.prepare`): those the config selects
        (:func:`build_default_batch_transforms`), then ``extra_batch_transforms``."""

        transforms = build_default_batch_transforms(self.config)
        if extra_batch_transforms:
            transforms.extend(extra_batch_transforms)
        return transforms

    @property
    def model_info(self) -> ModelInfo:
        """The model info containing the LLM proxy and renderer."""
        return self._model_info

    def _leader(self) -> Policy:
        """The policy this trainer trains, and its rollouts sample: the leader of the run's policies (``main``, the
        one policy of a run)."""

        return self.policies[self.policies.leader]

    @property
    def _model_info(self) -> ModelInfo:
        """The model info of the policy the trainer trains (:meth:`_leader`), which the policy holds."""

        return self._leader().info

    @_model_info.setter
    def _model_info(self, info: ModelInfo) -> None:
        # Tests set it on a trainer they make without its constructor, which has no policies then: the leader's is
        # made here, as the config declares it (the first policy), around the info they give.  To be removed once the
        # tests make their trainers with the constructor or with fakes.
        if getattr(self, "policies", None) is None:
            leader = next(iter(policy_specs(self.config).values()))
            self.policies = PolicySet([Policy(leader, info)])
        else:
            self._leader().info = info

    def _new_stats_logger(self, stats_logger_config: Any) -> StatsLogger:
        """The run's stats logger, of this module's ``StatsLogger``, which tests replace here."""

        return StatsLogger(stats_logger_config, exp_config=self.config)

    # Methods whose code lives in the stage packages, as module functions.  Each stays a method under its name, so
    # that replacing it on the class or on an instance (a subclass, a test) replaces what the trainer calls; replacing
    # the module function does not.  Those that take no trainer are static methods.  Here: running an evaluation's
    # groups, the loops, evaluation and the checkpoint cadence (the heartbeat and the signal handlers are
    # TrainerBase's).  A training group runs in the start's rollout source (platoon.train.dispatch.runner), not through
    # _run_group, and no group waits through _await_rollout_with_heartbeat, which stays for the tests that call it.
    _await_rollout_with_heartbeat = staticmethod(waiting._await_rollout_with_heartbeat)
    _run_group = staticmethod(runner._run_group)
    _rollout_workflow_worker_loop = evaluation_loop._rollout_workflow_worker_loop
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

    # The start's draws: its rollout source (_TrainerDraws), which prepare() opens and the run starts and closes.  Its
    # steps of opening, saving and taking up the draws, and its training tasks, go through the methods below, which
    # tests replace; each calls the part of the source of its name (platoon.train.dispatch.source, .setup,
    # platoon.train.checkpoints.saving, .resume) with the trainer's source.  The parts those parts call are module
    # functions, not the trainer's methods: the task samplers (setup._samplers, which _samplers stays for the tests
    # that call it), the kept groups and the saved samplings a resume reads, and a training group's run.

    def _task_sources(self) -> TaskSources:
        """The training tasks as the trainer holds them: its dataset, the order it draws them in and its task
        samplers."""

        return TaskSources(
            train_dataset=getattr(self, "train_dataset", None),
            train_order=self.train_order,
            train_sampling=self.train_sampling,
        )

    def _training_tasks(self) -> Any:
        """Every task training draws from, once each, in pool order
        (:func:`~platoon.train.dispatch.source.training_tasks`)."""

        return training_tasks(self._task_sources())

    def _draws(self) -> RolloutSource:
        """The start's draws, which prepare() opens.  A trainer that a test drives outside a start, made without its
        constructor, has none: each use gets a source over its fields that no start opened (to go once the tests make
        their trainers with the constructor)."""

        return self._rollouts if self._rollouts is not None else _TrainerDraws(self, self.run_log_path)

    @property
    def _recorder(self) -> TaskRecorder | None:
        return self._rollouts.recorder if self._rollouts is not None else None

    @_recorder.setter
    def _recorder(self, recorder: TaskRecorder | None) -> None:
        # A test's _begin_task_records sets it, as the start's draws open.
        assert self._rollouts is not None, "only the start's draws hold per-task statistics"
        self._rollouts.recorder = recorder

    @property
    def _dispatcher(self) -> Dispatcher | None:
        return self._rollouts.dispatcher if self._rollouts is not None else None

    @property
    def _admissions(self) -> AdmissionSet | None:
        return self._rollouts.admissions if self._rollouts is not None else None

    def _begin_task_records(self, resume_info: Any, start_batch: int) -> None:
        """Begin this start's per-task statistics (:func:`~platoon.train.dispatch.source._begin_task_records`)."""

        rollout_source._begin_task_records(self._draws(), resume_info, start_batch)

    def _task_records_started(self) -> None:
        """The launch save succeeded: the start's per-task statistics log their beginning
        (:func:`~platoon.train.dispatch.source._task_records_started`)."""

        rollout_source._task_records_started(self._draws())

    async def _checkpoint_state(self, checkpoint: str, loop_state: dict[str, Any]) -> dict[str, Any]:
        """``loop_state`` of full checkpoint ``checkpoint``, with what it saves of the draws first
        (:func:`~platoon.train.checkpoints.saving._checkpoint_state`)."""

        return await saving._checkpoint_state(self._draws(), checkpoint, loop_state)

    def _new_dispatcher(self, train_dataloader: Any, start_batch: int, end_batch: int) -> Dispatcher:
        """The dispatcher of this start's training groups (:func:`~platoon.train.dispatch.setup._new_dispatcher`)."""

        return setup._new_dispatcher(self._draws(), train_dataloader, start_batch, end_batch)

    def _samplers(self, train_dataloader: Any, sampler: str) -> SamplerSet:
        """The task samplers the run draws through (:func:`~platoon.train.dispatch.setup._samplers`)."""

        return setup._samplers(self._draws(), train_dataloader, sampler)

    def _dispatch(
        self,
        task_group: asyncio.TaskGroup,
        shared_state: TrainLoopSharedState,
        start_batch: int,
        end_batch: int,
        train_workflow: RolloutWorkflow,
        results: asyncio.Queue,
    ) -> Dispatcher | None:
        """Start the start's draws in the train loop's ``task_group`` (:meth:`RolloutSource.start`): the dispatcher
        prepare() built, with what the checkpoint the start resumes from saved, its groups settling into ``results``
        and feeding the loops' watchdog, until the loops' shutdown.  The draws hold the steps' range and the training
        workflow (``train_workflow``, the start's).  The train loop's tests replace this method, put their groups in
        ``results`` and return None: the loop then tells no dispatcher of its steps."""

        return self._draws().start(
            task_group,
            results=results,
            watchdog=shared_state.watchdog,
            tracker=shared_state.train_tracker,
            stop=shared_state.shutdown_event,
        )

    def _restore_dispatch(self, dispatcher: Dispatcher, resume_info: Any, policy_version: int) -> None:
        """Take up the draws the checkpoint ``resume_info`` left outstanding
        (:func:`~platoon.train.checkpoints.resume._restore_dispatch`)."""

        resume._restore_dispatch(self._draws(), dispatcher, resume_info, policy_version)

    def _start_recovery_pruning(self) -> RecoveryCheckpointPruner | None:
        """The pruner of this run's recovery checkpoints (``checkpoint.keep_recovery``), which deletes them on a Tinker
        session of its own; None without ``keep_recovery``."""

        keep = self.config.checkpoint.keep_recovery
        # The largest bound of the run and its entries: a slow entry's group may still sample an older checkpoint.
        max_staleness, _ = entry_settings.largest_staleness(self.config)
        if keep is None or max_staleness is None:  # loading refuses keep_recovery without max_staleness
            return None
        # The service the policy trains on, as its spec read it when the trainer was built.
        base_url = self._leader().spec.tinker_base_url
        return RecoveryCheckpointPruner(
            run_dir=self.run_log_path,
            keep=keep,
            max_staleness=max_staleness,
            service_client_factory=lambda: tinker.ServiceClient(base_url=base_url),
        )

    def _recovery_saved(self, paths: dict[str, str], policy_version: int) -> None:
        """Hand a recovery save that ``checkpoints.jsonl`` now records to the pruner, if any, which deletes the run's
        older recovery checkpoints beside training."""

        if self._recovery_pruner is not None:
            self._recovery_pruner.recovery_saved(paths, policy_version)

    def _checkpoints(self, training_client: tinker.TrainingClient) -> RunCheckpoints:
        """The full checkpoints that the start's steps save (``_save_checkpoint_and_get_sampling_client``): those of the
        start train() runs, whose leader policy trains with ``training_client``.  A trainer that a test drives outside
        a start, made without its constructor, saves through checkpoints of its run directory over a policy that holds
        ``training_client``, with its draws (to go once the tests make their trainers with the constructor).  A trainer
        whose start opened its draws, but which train() did not record, raises instead: saving outside that start would
        lose its other participants."""

        if self._run is not None:
            return self._run.checkpoints
        if self._rollouts is not None:
            raise RuntimeError("the rl trainer's steps save through the start train() runs, which it did not record")
        checkpoints = RunCheckpoints(
            self.run_log_path,
            PolicySet([_main_holding(training_client=training_client)]),
            wandb_run_id=lambda: self._stats_logger.wandb_run_id if self._stats_logger is not None else None,
        )
        checkpoints.register(self._draws())
        return checkpoints

    async def _create_training_client(
        self, service_client: tinker.ServiceClient, resume_info: checkpoint_utils.CheckpointRecord | None
    ) -> tinker.TrainingClient:
        """Ask the trained policy's service once for its training client: the one ``resume_info`` resumes, else a new
        run's (:meth:`~platoon.train.policy.policy.Policy.create_training_client`)."""

        return await self._leader().create_training_client(service_client, resume_info)

    async def _open_training_client(
        self,
        service_client: tinker.ServiceClient,
        resume_info: checkpoint_utils.CheckpointRecord | None,
        stop: asyncio.Event | None = None,
    ) -> tinker.TrainingClient | None:
        """:meth:`_create_training_client`, asked again after each refusal that asking again may change
        (:func:`~platoon.train.policy.service.open_training_client`); None once ``stop`` is set between two asks."""

        return await service.open_training_client(
            lambda: self._create_training_client(service_client, resume_info), stop=stop
        )

    async def run(self) -> None:
        """A start of a trainer built from its setup (:meth:`from_setup`): the lifecycle every trainer goes through,
        whose run builds the training and evaluation workflows from the setup, on the trainer's policies, first."""

        self._run = TrainerRun(self)
        await self._run.execute()

    async def train(self, train_workflow: RolloutWorkflow, eval_workflow: RolloutWorkflow | None = None):
        """Train this start of the run, from the checkpoint it resumes from (:class:`StartPoint`; none for a fresh run)
        to the run's last step, with these workflows: the lifecycle every trainer goes through
        (:class:`~platoon.train.trainer.run.TrainerRun`), whose hooks below are this trainer's part, in the order they
        run."""

        self._run = TrainerRun(self, workflows=Workflows(train_workflow, eval_workflow))
        await self._run.execute()

    # The lifecycle's hooks (TrainerBase), each a phase of the start or a step of one, in the order the run calls them.

    def check(self, run: TrainerRun) -> None:
        """1. Inside ``async with trainer:``, the config's checks before training opens its service."""

        self._require_entered()
        self._check_before_start()

    async def _open_policies(self, run: TrainerRun, start: StartPoint) -> bool:
        """4. Open the trained policy's training client on its service, at its start point: the one its checkpoint
        resumes, else a new run's, through the trainer's seams, which tests replace: :meth:`_open_training_client`,
        which asks through :meth:`_create_training_client`.  False when training was stopped before the service
        accepted it."""

        policy = self._leader()
        with key_refusals(asked="a training client"):
            training_client = await self._open_training_client(
                policy.service_client, start.policies[policy.id].record, stop=run.shutdown
            )
        if training_client is None:
            return False
        policy.training_client = training_client
        return True

    async def prepare(self, run: TrainerRun, start: StartPoint) -> StepPlan:
        """5. Begin the rollouts' startup (:meth:`_begin_rollout_startup`), then open the start's draws before the
        launch save, which saves them: its rollout source over the training workflow (:class:`_TrainerDraws`), which
        the run opens as it opens any trainer's (:meth:`TrainerRun._open_draws`): the per-task statistics, the order
        and the dispatcher of the training groups, which take up the draws the checkpoint left outstanding and take
        part in every full checkpoint.  The training dataloader is the draws' order, and the steps end at theirs."""

        assert run.workflows is not None  # train() runs the lifecycle with its workflows, run() with its setup's
        self._startup = await self._begin_rollout_startup(run.workflows.train, run.workflows.eval)
        draws = _TrainerDraws(self, run.run_dir, workflow=run.workflows.train, leader=run.policies.leader)
        self._rollouts = draws  # before it opens: a test's _begin_task_records sets its _recorder
        run._open_draws(draws)
        assert draws.order is not None  # opened
        self._train_dataloader = draws.order
        return StepPlan(end_step=draws.end_step)

    async def started(self, run: TrainerRun) -> None:
        """6. The launch save succeeded, and the start's draws logged their start (the run's, through
        :meth:`_task_records_started`): the pruning of the run's recovery checkpoints starts with the launch save, so
        that a resumed run first deletes what its earlier starts left."""

        self._recovery_pruner = self._start_recovery_pruning()
        self._recovery_saved(run.launch_save[self.policies.leader], run.start.policy_version)

    async def ready(self, run: TrainerRun) -> None:
        """7. What the train and evaluation loops share (:meth:`_loop_state`), once the harness toolchains are staged;
        its clock, which it starts where the start's steps begin, is the run's."""

        self._shared_state = await self._loop_state(
            run.start, run.shutdown, self._train_dataloader, self._startup, end_batch=run.plan.end_step
        )
        assert self._shared_state.clock is not None  # the loop state is given its clock
        run.clock = self._shared_state.clock

    async def run_steps(self, run: TrainerRun) -> None:
        """8. The train loop, and beside it the evaluation loop when the start evaluates (:meth:`_run_loops`)."""

        assert run.workflows is not None  # train() runs the lifecycle with its workflows
        run.interrupted = await self._run_loops(
            self._shared_state, run.start, run.plan.end_step, run.workflows.train, run.workflows.eval
        )

    def _nothing_to_train(self, run: TrainerRun) -> None:
        """9. A start that resumed at the last step trained nothing: it says whether it ran the evaluation due there,
        which no earlier start completed."""

        shared_state, start_batch = self._shared_state, run.start.batch
        if shared_state.eval_loop_running and shared_state.evaluation_due(start_batch):
            if shared_state.eval_handled_step == start_batch:
                logger.info(f"Training was already complete; this start ran the evaluation due at step {start_batch}")
            else:
                logger.info(
                    f"Training was already complete; the evaluation due at step {start_batch} stopped before it "
                    "ended, so the next start runs it"
                )
        else:
            logger.info("Training was already complete; nothing to do")

    async def close(self, run: TrainerRun) -> None:
        """10. Close what the start opened (:meth:`_close_after_training`)."""

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
        if self.config.eval.at_start and not evaluates(self.config):
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
            normalize_environment_configs(self.config.environments),
            [train_workflow, eval_workflow if self.eval_dataset is not None and evaluates(self.config) else None],
        )

    async def _loop_state(
        self,
        start: StartPoint,
        shutdown_event: asyncio.Event,
        train_dataloader: PlatoonTinkerDataloader,
        startup: RolloutStartup,
        *,
        end_batch: int | None = None,
    ) -> TrainLoopSharedState:
        """What the train and evaluation loops share (:class:`TrainLoopSharedState`), once the harness toolchains are
        staged (``startup``): the run's policies, the trained one's with its optimizer's settings, the evaluation's
        dataloader, the step clock (:meth:`StepClock.started`: where the start begins, towards ``end_batch``, with the
        watchdog, started when it is enabled, and the event of each new policy, set: the evaluation loop's first wait
        passes at once), and where the start begins."""

        assert self._stats_logger is not None and self._train_tracker is not None  # train() checked (_require_entered)
        policy = self._leader()
        assert policy.spec.optimizer is not None  # train.optimizer, as policy_specs took it when the trainer was built
        policy.optimizer_params = adam_params(policy.spec.optimizer)

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

        clock = StepClock.started(self.config, step=start.batch, end=end_batch)
        return TrainLoopSharedState(
            config=self.config,
            shutdown_event=shutdown_event,
            clock=clock,
            policies=self.policies,
            sampling_client_step=start.policy_version,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            stats_logger=self._stats_logger,
            train_tracker=self._train_tracker,
            watchdog=clock.watchdog,
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

        watchdog: Watchdog | None = shared_state.watchdog
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

    async def _close_after_training(self) -> None:
        """Stop the pruning of the run's recovery checkpoints (the run closes the start's draws after it)."""

        if self._recovery_pruner is not None:
            await self._recovery_pruner.aclose()


class _TrainerDraws(RolloutSource):
    """The rl trainer's draws: a rollout source over the trainer's config, training tasks and policies.

    Its steps of opening, saving and taking up the draws go through the trainer's methods of their names, which tests
    replace (on the class or an instance).  Those methods call the source's parts:

    - ``_begin_task_records`` and ``_task_records_started``: the per-task statistics' beginning and start;
    - ``_new_dispatcher``: the dispatcher;
    - ``_checkpoint_state`` and ``_restore_dispatch``: the draws' entries of a full checkpoint, and their resume.

    Its training tasks are the trainer's (``_training_tasks``)."""

    def __init__(
        self,
        trainer: PlatoonTinkerRLTrainer,
        run_dir: str,
        *,
        workflow: RolloutWorkflow | None = None,
        leader: str = MAIN,
    ) -> None:
        super().__init__(
            trainer.config,
            run_dir=run_dir,
            tasks=trainer._task_sources(),
            workflow=workflow,
            versions=lambda: trainer.policies.versions(),
            leader=leader,
        )
        self.trainer = trainer

    def begin_task_records(self, record: Any, step: int) -> None:
        self.trainer._begin_task_records(record, step)

    def new_dispatcher(self, order: PlatoonTinkerDataloader, start_step: int, end_step: int) -> Dispatcher:
        return self.trainer._new_dispatcher(order, start_step, end_step)

    def training_tasks(self) -> Any:
        return self.trainer._training_tasks()

    def started(self) -> None:
        self.trainer._task_records_started()

    async def checkpoint_state(self, name: str) -> dict[str, Any]:
        return await self.trainer._checkpoint_state(name, {})

    def restore(self, dispatcher: Dispatcher, record: Any, policy_version: int) -> None:
        self.trainer._restore_dispatch(dispatcher, record, policy_version)
