"""The trainer API (RFC 0021; experimental): what ``platoon train`` runs (:class:`Trainer`), what it builds a trainer
with (:class:`TrainerSetup`: the config, the policies' specs, the tasks and the workflows), what a trainer asks of the
run (:class:`TrainerNeeds`) and where its steps end (:class:`StepPlan`); and an optional base class
(:class:`TrainerBase`), whose ``run()`` takes the trainer through the lifecycle every trainer shares
(:class:`~platoon.train.trainer.run.TrainerRun`), which calls the base's hooks in order.

Imports no Tinker, torch or datasets, so that ``platoon.experimental`` re-exports it: the policies, which load Tinker,
are built when a trainer is.
"""

import logging
import os
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Self

from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.train.policy.spec import PolicyId, PolicySpec
from platoon.train.trainer import signals, waiting

if TYPE_CHECKING:
    from platoon.train.checkpoints.resume import StartPoint
    from platoon.train.policy.set import PolicySet
    from platoon.train.trainer.run import TrainerRun
    from platoon.utils.stats_logger import StatsLogger
    from platoon.utils.stats_tracker import StatsTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Workflows:
    """A start's rollout workflows: training's, and the evaluation's (None when the start evaluates nothing)."""

    train: Any
    eval: Any | None = None


@dataclass(frozen=True)
class TaskSources:
    """The run's tasks as ``platoon train`` lists them: the training tasks (``train_dataset``, a weighted order's first
    epoch or the listing a dataloader shuffles), the order they are drawn in epoch by epoch and the task samplers that
    draw them (None: the dataset as it is), and the evaluation's (None when the run evaluates nothing)."""

    train_dataset: Any
    eval_dataset: Any = None
    train_order: Any = None
    train_sampling: Any = None


@dataclass(frozen=True)
class TrainerNeeds:
    """What a trainer asks of the run, which ``platoon train`` reads before it lists any task (RFC 0021, rule 10 for
    the second and third)."""

    # Whether it trains on rollouts of the run's training tasks: then the run lists them, builds the training workflow,
    # and opens the start's draws when the trainer asks for them (run.rollouts()).
    rollouts: bool = False
    # Whether its steps reach a checkpoint whole or not at all: once a step sent training work, an interrupt saves no
    # <step>_interrupted checkpoint of it, and the resume starts from the last full one.  The rl trainer's are not.
    atomic_steps: bool = True
    # Whether an optimizer step that the service skipped, or whose metrics are not finite, raises (UpdateRejected)
    # before anything publishes, for every update, the step hooks' too.  The rl trainer only logs such metrics.
    fail_closed_updates: bool = True
    # Whether each training episode sees its draw (ctx.draw: platoon.programs.DrawInfo), the key that a step's end
    # names it by (StepEnd.consumed).  Off, the rollouts' requests are as they always were.
    draw_provenance: bool = False


@dataclass(frozen=True)
class TrainerSetup:
    """What ``platoon train`` prepared for the trainer it builds (:meth:`Trainer.from_setup`).  Building the setup
    does no I/O.

    - ``config``: the loaded and checked config, read-only.  Its ``log_path``, ``stats.experiment_name`` and
      ``stats.trial_name`` name the run directory (:attr:`TrainerBase.run_log_path`).
    - ``policy_specs``: the policies the config declares (:func:`~platoon.train.policy.spec.policy_specs`), which the
      trainer's constructor registers.
    - ``options``: the trainer's options (None: it takes none).
    - ``tasks``: the run's tasks, when the trainer rolls them out.
    - ``workflow_builder``: how the start's workflows are built on the trainer's policies, for a trainer that rolls
      out.  :meth:`build_workflows` calls it as the run begins."""

    config: Any
    policy_specs: Mapping[PolicyId, PolicySpec]
    options: Any = None
    tasks: TaskSources | None = None
    # Launch's builder of the start's workflows, on the trainer's policies; None: the setup builds none.
    workflow_builder: "Callable[[PolicySet], Workflows] | None" = None

    def build_workflows(self, policies: "PolicySet") -> Workflows:
        """The start's training and evaluation workflows, which roll out on ``policies``."""

        if self.workflow_builder is None:
            raise RuntimeError(
                "this trainer's setup builds no workflows: platoon train gives them to trainers that roll out"
            )
        return self.workflow_builder(policies)


@dataclass(frozen=True)
class StepPlan:
    """Where a start's steps end: ``end_step``, the step after the run's last one (the start trains up to it); and
    ``steps_per_epoch``, the steps of one pass over what the trainer trains on, by which ``checkpoint.strategy: epoch``
    counts (None: a pass is a step)."""

    end_step: int
    steps_per_epoch: int | None = None


class Trainer(Protocol):
    """What ``platoon train`` runs: a trainer is resolved from the config's ``trainer:``
    (:mod:`platoon.train.trainer.registry`), built from its setup, entered with ``async with``, and run once per start.
    Supervising the run, its restarts, the reaping of what a dead start left, the run's lock, its config snapshot and
    its identity check stay ``platoon train``'s."""

    @classmethod
    def needs(cls, config: Any, options: Any) -> TrainerNeeds: ...

    @classmethod
    def from_setup(cls, setup: TrainerSetup) -> Self: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None: ...

    async def run(self) -> None: ...


class TrainerBase:
    """An optional base of trainers.  ``run()`` takes the trainer through the lifecycle every trainer shares
    (:class:`~platoon.train.trainer.run.TrainerRun`), which calls the hooks below in order, each with the run; every
    hook has a default but ``prepare`` and ``run_steps``, which a subclass writes.  ``async with`` opens what every
    start logs to, the stats logger (which resumes the W&B run of the checkpoint the start resumes from) and the train
    tracker, and labels every sandbox the run starts with the run, so that the run's sandboxes can be reaped by it."""

    # What the trainer was built with; None for a trainer built with a constructor of its own, outside platoon train.
    setup: TrainerSetup | None = None
    config: Any
    options: Any = None
    # The run's policies, registered as the trainer is built.
    policies: "PolicySet"
    _stats_logger: "StatsLogger | None" = None
    _train_tracker: "StatsTracker | None" = None
    _sandbox_labels: ExitStack | None = None

    def __init__(self, setup: TrainerSetup) -> None:
        from platoon.train.policy.set import PolicySet  # Tinker loads with the policies, not with this module

        self.setup, self.config, self.options = setup, setup.config, setup.options
        # The policies the config declares, and the model gateway that samples them, this process's default.
        self.policies = PolicySet.build(setup.policy_specs)

    @classmethod
    def needs(cls, config: Any, options: Any) -> TrainerNeeds:
        """What the trainer asks of the run: by default, nothing to roll out."""

        return TrainerNeeds()

    @classmethod
    def from_setup(cls, setup: TrainerSetup) -> Self:
        return cls(setup)

    @property
    def run_log_path(self) -> str:
        """The run directory, ``log_path/<stats.experiment_name>/<stats.trial_name>``, where the run writes everything:
        its checkpoints, its rollouts and its logs."""

        return os.path.join(self.config.log_path, self.config.stats.experiment_name, self.config.stats.trial_name)

    async def __aenter__(self) -> Self:
        """Open the stats logger, with the W&B run of the checkpoint the start resumes from, and the train tracker, and
        label the run's sandboxes."""

        from tinker_cookbook import checkpoint_utils

        from platoon.utils.stats_tracker import get as get_tracker

        resume_info = checkpoint_utils.get_last_checkpoint(self.run_log_path)
        # The stats logger adds experiment_name/trial_name to the base log_path itself.
        stats_logger_config = self.config.stats.to_stats_logger_config(self.config.log_path)
        resume_extra = resume_info.extra if resume_info is not None else {}
        wandb_run_id = resume_extra.get("wandb_run_id")
        if wandb_run_id:
            stats_logger_config.wandb.resume_run_id = wandb_run_id
            logger.info(f"Resuming WandB run: {wandb_run_id}")

        self._stats_logger = self._new_stats_logger(stats_logger_config)
        self._train_tracker = get_tracker("train")
        # Every sandbox the run starts, here or in a rollout worker (which carries this process's labels on), is
        # labelled with the run, so that the run's sandboxes can be reaped by it and no other run's are.
        run_label = sandbox_cleanup.run_label(self.run_log_path)
        self._sandbox_labels = ExitStack()
        self._sandbox_labels.enter_context(sandbox_cleanup.labelled(run_label))
        logger.info("This run labels its sandboxes %s", run_label)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Put back this process's sandbox labels as they were before the run, and close the stats logger; an error
        is not suppressed."""

        sandbox_labels, self._sandbox_labels = self._sandbox_labels, None
        if sandbox_labels is not None:
            sandbox_labels.close()
        if self._stats_logger is not None:
            self._stats_logger.close()
        return False

    def _new_stats_logger(self, stats_logger_config: Any) -> "StatsLogger":
        """The run's stats logger, with the run's config as its W&B run's config."""

        from platoon.utils.stats_logger import StatsLogger

        return StatsLogger(stats_logger_config, exp_config=self.config)

    async def run(self) -> None:
        """Run this start of the run: the lifecycle every trainer shares, with this trainer's hooks."""

        from platoon.train.trainer.run import TrainerRun

        await TrainerRun(self).execute()

    # The lifecycle's hooks, in the order the run calls them (TrainerRun.execute numbers its phases).

    def check(self, run: "TrainerRun") -> None:
        """1. Before any service opens: raise ``ConfigError`` for what the trainer cannot run.  Participants of the
        run's checkpoints registered here (``run.checkpoints.register``) take up the start point before any training
        client opens; step hooks may register here too (``run.hooks.register``), or in ``prepare``."""

    async def prepare(self, run: "TrainerRun", start: "StartPoint") -> StepPlan:
        """5. The policies are open at ``start``: prepare the steps, and say where they end.  A trainer that rolls out
        opens the start's draws here (``run.rollouts()``: the run's training groups, which take part in every full
        checkpoint), and starts them as its steps begin."""

        raise NotImplementedError(f"{type(self).__name__} must say where its steps end (TrainerBase.prepare)")

    async def started(self, run: "TrainerRun") -> None:
        """6. The launch save is done (``run.launch_save``), before each policy's first sampler."""

    async def ready(self, run: "TrainerRun") -> None:
        """7. Just before the steps: the run's clock starts (:meth:`~platoon.train.trainer.run.TrainerRun.start_clock`),
        which a trainer that overrides this hook starts too."""

        run.start_clock()

    async def run_steps(self, run: "TrainerRun") -> None:
        """8. The steps, from ``run.clock.step`` to ``run.clock.end``, stopping at a step boundary once a stop is asked
        for (``run.shutdown``).  Each step ends with ``run.end_step``: the step hooks, the publication of the policies
        it updated, and the clock's advance.  Steps that the stop request stopped short of the end were interrupted:
        the run then saves ``<step>_interrupted``, also when they did not set ``run.interrupted``, which an interrupt
        of their own sets, and so does a ``TrainInterrupted`` they raise."""

        raise NotImplementedError(f"{type(self).__name__} must write its steps (TrainerBase.run_steps)")

    async def close(self, run: "TrainerRun") -> None:
        """10. After the save at exit, and after a failure too (``run.ended`` says which): close what the trainer
        opened.  The run's step hooks and participants close after it."""

    # What the lifecycle calls through the trainer, so that a subclass or a test may replace it on the class or an
    # instance: the waiter of the run's saves, the signal handlers, how the policies open, and what a start that had
    # no step left to train says.

    _await_with_heartbeat = waiting._await_with_heartbeat
    _setup_signal_handlers = signals._setup_signal_handlers

    async def _open_policies(self, run: "TrainerRun", start: "StartPoint") -> bool:
        """4. Open the policies' training clients at the start point; False when training was stopped first."""

        return await run.policies.open(start, stop=run.shutdown)

    def _nothing_to_train(self, run: "TrainerRun") -> None:
        """9. Say why a start that resumed at its last step trained nothing."""

        logger.info("Training was already complete; nothing to do")
