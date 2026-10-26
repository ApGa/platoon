"""What the evaluation during training reads of the steps it runs beside: the loops' shared state (:class:`LoopState`,
the rl trainer's ``TrainLoopSharedState``) and the trainer whose seams it calls (:class:`EvaluatedTrainer`, the rl
trainer).  The evaluation names them by these protocols, not by the rl trainer's classes, which it does not import
(RFC 0021's rule of layers): they are what it depends on.
"""

import asyncio
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import tinker

    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.train.dataloader import PlatoonTinkerDataloader
    from platoon.train.dispatch.permits import _Rows
    from platoon.train.tinker.proxy import ModelInfo
    from platoon.train.trainer.signals import Watchdog
    from platoon.utils.rollout_stats import RolloutStats
    from platoon.utils.stats_logger import StatsLogger
    from platoon.utils.stats_tracker import StatsTracker


class LoopState(Protocol):
    """The loops' shared state, as the evaluation reads it: the run's config, the step training is at, the shutdown
    request, the event of each published policy and the policy (its sampling client and model info), the evaluation's
    split, the stats logger and the watchdog; where the start began and whether an earlier start evaluated that step;
    the evaluation's cadence and record.  It sets one field, the step it handled last (``eval_handled_step``), and sets
    the event of its progress."""

    eval_handled_step: int | None

    @property
    def config(self) -> "PlatoonTinkerRLTrainerConfig": ...
    @property
    def train_step(self) -> int: ...
    @property
    def shutdown_event(self) -> asyncio.Event: ...
    @property
    def sampling_client_updated_event(self) -> asyncio.Event: ...
    @property
    def sampling_client(self) -> "tinker.SamplingClient": ...
    @property
    def model_info(self) -> "ModelInfo": ...
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
    """The trainer, as the evaluation reads it: its config and evaluation split (None without one), and the seams the
    evaluation calls through it, which tests replace: a group's run, a rollout worker, the collector of an evaluation's
    groups, and the publication of the sampling client to rollout workers."""

    @property
    def config(self) -> "PlatoonTinkerRLTrainerConfig": ...
    @property
    def eval_dataset(self) -> Any: ...
    async def _run_group(self, workflow: Any, row: dict[str, Any], *, watchdog: Any, tracker: Any) -> Any: ...
    async def _rollout_workflow_worker_loop(
        self,
        shared_state: LoopState,
        workflow: Any,
        task_data_queue: asyncio.Queue[dict],
        rollout_result_queue: asyncio.Queue[Any],
        *,
        tracker: "StatsTracker",
        rows: "_Rows | None" = None,
    ) -> None: ...
    async def _eval_completion_monitor(
        self,
        total_eval_tasks: int,
        eval_rollout_result_queue: asyncio.Queue[Any],
        tracker: "StatsTracker | None" = None,
        label_stats: "dict[str, RolloutStats] | None" = None,
    ) -> None: ...
    def _publish_worker_binding(
        self, workflow: Any, sampling_client: Any, version: int, *, required: bool = False
    ) -> None: ...
