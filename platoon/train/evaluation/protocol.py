"""What the evaluation during training reads of the steps it runs beside: the loops' shared state (:class:`LoopState`,
the rl trainer's ``TrainLoopSharedState``) and the trainer it evaluates for (:class:`EvaluatedTrainer`, the rl
trainer).  The evaluation names them by these protocols, not by the rl trainer's classes, which it does not import
(RFC 0021's rule of layers): they are what it depends on.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import tinker

    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.train.dataloader import PlatoonTinkerDataloader
    from platoon.train.policy.set import PolicySet
    from platoon.train.tinker.proxy import ModelInfo
    from platoon.train.trainer.signals import Watchdog
    from platoon.utils.stats_logger import StatsLogger


class LoopState(Protocol):
    """The loops' shared state, as the evaluation reads it: the run's config, the step training is at, the shutdown
    request, the event of each step's end and the policy (its sampling client and model info) among the run's policies
    (``policies``, whose frozen ones an evaluation publishes the bindings of), the evaluation's split, the stats logger
    and the watchdog; where the start began and whether an earlier start evaluated that step; the evaluation's cadence
    and record.  It sets one field, the step it handled last (``eval_handled_step``), and sets the event of its
    progress."""

    eval_handled_step: int | None

    @property
    def config(self) -> "PlatoonTinkerRLTrainerConfig": ...
    @property
    def train_step(self) -> int: ...
    @property
    def shutdown_event(self) -> asyncio.Event: ...
    @property
    def step_ended(self) -> asyncio.Event: ...
    @property
    def sampling_client(self) -> "tinker.SamplingClient": ...
    @property
    def model_info(self) -> "ModelInfo": ...
    @property
    def policies(self) -> "PolicySet": ...
    @property
    def eval_dataloader(self) -> "PlatoonTinkerDataloader | None": ...
    @property
    def stats_logger(self) -> "StatsLogger": ...
    @property
    def watchdog(self) -> "Watchdog | None": ...
    @property
    def start_step(self) -> int: ...
    @property
    def start_step_evaluated(self) -> bool: ...
    @property
    def eval_every(self) -> int | None: ...
    @property
    def eval_loop_running(self) -> bool: ...
    @property
    def eval_progress_event(self) -> asyncio.Event: ...
    @property
    def evaluations_path(self) -> str | None: ...
    def evaluation_due(self, step: int) -> bool: ...


class EvaluatedTrainer(Protocol):
    """The trainer, as the evaluation reads it: its config and its evaluation split (None without one)."""

    @property
    def config(self) -> "PlatoonTinkerRLTrainerConfig": ...
    @property
    def eval_dataset(self) -> Any: ...
