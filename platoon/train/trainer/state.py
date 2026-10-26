"""What the train and evaluation loops of one start share (``TrainLoopSharedState``)."""

import asyncio
from dataclasses import dataclass, field

import tinker

from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, TrainEventTriggerConfig, event_interval
from platoon.train.dataloader import PlatoonTinkerDataloader
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.trainer.signals import Watchdog
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import StatsTracker


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
