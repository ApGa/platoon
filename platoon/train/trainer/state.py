"""What the train and evaluation loops of one start share (``TrainLoopSharedState``).

The policy they train is the leader of the run's policies (``policies``): its model info, its Tinker clients, its
optimizer's parameters and the optimizer steps it took are the policy's, which the shared state reads through
properties of the same names.  The step the loops are at (``train_step``) and the event of each step's end
(``step_ended``) are the run's clock's (``clock``, :class:`~platoon.train.trainer.run.StepClock`).
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field

import tinker

from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, TrainEventTriggerConfig, event_interval
from platoon.train.dataloader import PlatoonTinkerDataloader
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.trainer.run import StepClock
from platoon.train.trainer.signals import Watchdog
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import StatsTracker


@dataclass(kw_only=True)
class TrainLoopSharedState:
    """What the train and evaluation loops of one start share, made with keyword arguments: the run's policies
    (``policies``), whose leader the loops train, and the run's step clock (``clock``), which each step's end advances.
    A copy (``copy.copy()``) shares the original's policies and clock, and so their state."""

    config: PlatoonTinkerRLTrainerConfig
    shutdown_event: asyncio.Event
    train_dataloader: PlatoonTinkerDataloader
    eval_dataloader: PlatoonTinkerDataloader | None
    policies: PolicySet
    clock: StepClock
    # Each trained policy's clock when they step independently, the leader's ``clock`` (None: in lockstep).
    clocks: Mapping[str, StepClock] | None = None
    stats_logger: StatsLogger
    train_tracker: StatsTracker
    # The run's watchdog, the one its clock feeds at each step's end (clock.watchdog), which the loops' waits feed too;
    # None when the config turns it off.
    watchdog: Watchdog | None = None
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

    @property
    def train_step(self) -> int:
        """The step the loops are at: the run's clock's."""

        return self.clock.step

    @property
    def step_ended(self) -> asyncio.Event:
        """The event each step's end sets, also a step's that published no policy, and the loops' stop: the run's
        clock's (``StepClock.updated``), which the evaluation loop wakes on."""

        return self.clock.updated

    # The state of the policy the loops train, the leader: the steps publish each new sampling client here.

    def _leader(self) -> Policy:
        return self.policies[self.policies.leader]

    @property
    def model_info(self) -> ModelInfo:
        return self._leader().info

    @property
    def training_client(self) -> tinker.TrainingClient:
        return self._leader().training_client

    @property
    def sampling_client(self) -> tinker.SamplingClient:
        return self._leader().sampling_client

    @sampling_client.setter
    def sampling_client(self, client: tinker.SamplingClient) -> None:
        self._leader().sampling_client = client

    @property
    def optimizer_params(self) -> tinker.AdamParams:
        return self._leader().optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, params: tinker.AdamParams) -> None:
        self._leader().optimizer_params = params

    @property
    def optimizer_steps(self) -> int:
        return self._leader().optimizer_steps

    @optimizer_steps.setter
    def optimizer_steps(self, steps: int) -> None:
        self._leader().optimizer_steps = steps

    def evaluation_due(self, step: int) -> bool:
        """Whether the evaluation loop evaluates ``step`` when training reaches it: a multiple of ``eval_every``, step
        0 only with ``eval.at_start``, and not the step this start began at when an earlier start completed its
        evaluation."""

        every = self.eval_every
        if every is None or step % every:
            return False
        if step == 0 and not self.config.eval.at_start:
            return False
        return not (step == self.start_step and self.start_step_evaluated)

    def _get_event_frequency(self, config: TrainEventTriggerConfig) -> int | None:
        """Training steps between the events ``config`` triggers (``every``), or ``None`` for ``every: 0``, which
        never triggers the event; ``strategy: epoch`` counts ``every`` in passes over the training tasks."""

        if config.strategy == "epoch":
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
