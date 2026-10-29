"""What the train and evaluation loops of one start share (``TrainLoopSharedState``).

The policy they train is the leader of the run's policies (``policies``): its model info, its Tinker clients, its
optimizer's parameters and the optimizer steps it took are the policy's, which the shared state reads and writes
through properties of the same names.  The step the loops are at and the event of each step's end are the run's clock's
(``clock``, :class:`~platoon.train.trainer.run.StepClock`), likewise ``train_step`` and
``sampling_client_updated_event``.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

import tinker

from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, TrainEventTriggerConfig, event_interval
from platoon.train.dataloader import PlatoonTinkerDataloader
from platoon.train.policy.policy import Policy
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, PolicySpec
from platoon.train.tinker.proxy import ModelInfo
from platoon.train.trainer.run import StepClock
from platoon.train.trainer.signals import Watchdog
from platoon.utils.stats_logger import StatsLogger
from platoon.utils.stats_tracker import StatsTracker

# The default of each part of the policy's state below: not given.
_NOT_GIVEN: Any = object()


@dataclass(kw_only=True)
class _GivenPolicyState:
    """The policy's state as a shared state was given it before the run's policies held it, and the loops' step and the
    event of each step's end before the run's clock held them: platoon 0.4.0's keyword arguments, which tests still
    give.  A shared state made with these, and without ``policies`` or ``clock``, keeps them in a policy or a clock of
    its own (:meth:`TrainLoopSharedState.__post_init__`).  To be removed with platoon 0.4.0's paths (0.6.0), once the
    tests make their states with policies and a clock."""

    optimizer_params: InitVar[Any] = _NOT_GIVEN
    service_client: InitVar[Any] = _NOT_GIVEN
    training_client: InitVar[Any] = _NOT_GIVEN
    sampling_client: InitVar[Any] = _NOT_GIVEN
    model_info: InitVar[Any] = _NOT_GIVEN
    optimizer_steps: InitVar[Any] = _NOT_GIVEN
    train_step: InitVar[Any] = _NOT_GIVEN
    sampling_client_updated_event: InitVar[Any] = _NOT_GIVEN


def _main_holding(
    *,
    optimizer_params: Any = None,
    service_client: Any = None,
    training_client: Any = None,
    sampling_client: Any = None,
    model_info: Any = None,
    optimizer_steps: int = 0,
) -> Policy:
    """Main's policy, the model ``model_info`` names, holding the state a shared state was given without the run's
    policies: each part not given is None, and the optimizer steps 0, as the shared state's defaults were."""

    model_name, renderer_name = getattr(model_info, "model_name", ""), getattr(model_info, "renderer_name", "")
    policy = Policy(PolicySpec(MAIN, model_name=model_name, renderer_name=renderer_name), model_info)
    policy.optimizer_params = optimizer_params
    policy.service_client = service_client
    policy.training_client = training_client
    policy.sampling_client = sampling_client
    policy.optimizer_steps = optimizer_steps
    return policy


@dataclass(kw_only=True)
class TrainLoopSharedState(_GivenPolicyState):
    """What the train and evaluation loops of one start share, made with keyword arguments: the trainer gives it the
    run's policies (``policies``) and clock (``clock``); a state made without them, given the policy's state or the
    step and the event instead (:class:`_GivenPolicyState`), makes its own.  Given both, it raises ``TypeError``, as
    ``dataclasses.replace()`` of it does, which gives both: the part given would be dropped.  A copy (``copy.copy()``)
    shares the original's policies and clock, and so their state."""

    config: PlatoonTinkerRLTrainerConfig
    shutdown_event: asyncio.Event
    train_dataloader: PlatoonTinkerDataloader
    eval_dataloader: PlatoonTinkerDataloader | None
    # The run's policies, whose leader the loops train; None only until __post_init__ makes the state's own.
    policies: PolicySet | None = None
    # The run's step clock, which each step's end advances; None only until __post_init__ makes the state's own.
    clock: StepClock | None = None
    # Each trained policy's clock when they step independently, the leader's ``clock`` (None: in lockstep).
    clocks: Mapping[str, StepClock] | None = None
    sampling_client_step: int
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

    def __post_init__(
        self,
        optimizer_params: Any,
        service_client: Any,
        training_client: Any,
        sampling_client: Any,
        model_info: Any,
        optimizer_steps: Any,
        train_step: Any,
        sampling_client_updated_event: Any,
    ) -> None:
        """Made without the run's policies: one of its own, ``main``, which holds the state the shared state was given
        (:func:`_main_holding`).  Made with them: given none of that state, which the policies hold.  Likewise the
        clock: made without one, one of its own at ``train_step``, with ``sampling_client_updated_event``."""

        self._own_clock(train_step, sampling_client_updated_event)

        parts = {
            "optimizer_params": optimizer_params,
            "service_client": service_client,
            "training_client": training_client,
            "sampling_client": sampling_client,
            "model_info": model_info,
            "optimizer_steps": optimizer_steps,
        }
        given = {name: part for name, part in parts.items() if part is not _NOT_GIVEN}
        if self.policies is None:
            self.policies = PolicySet([_main_holding(**given)])
        elif given:
            raise TypeError(
                "TrainLoopSharedState takes the run's policies or the policy's state, not both: given policies and "
                + ", ".join(given)
            )

    def _own_clock(self, train_step: Any, sampling_client_updated_event: Any) -> None:
        """The state's clock: the run's, when it was given one, else one of its own over the step and the event it was
        given (``TypeError`` when it was given both, or neither)."""

        given = [
            name
            for name, part in (
                ("train_step", train_step),
                ("sampling_client_updated_event", sampling_client_updated_event),
            )
            if part is not _NOT_GIVEN
        ]
        if self.clock is not None and given:
            raise TypeError(
                "TrainLoopSharedState takes the run's clock or the loops' step and event, not both: given clock and "
                + ", ".join(given)
            )
        if self.clock is None:
            if len(given) < 2:
                raise TypeError(
                    "TrainLoopSharedState takes the run's clock, or train_step and sampling_client_updated_event"
                )
            self.clock = StepClock(step=train_step, updated=sampling_client_updated_event, watchdog=self.watchdog)

    # The step the loops are at, and the event of each step's end, the run's clock's, under the names the shared state
    # gave them when it held them itself.

    def _clock(self) -> StepClock:
        assert self.clock is not None  # __post_init__ made it
        return self.clock

    @property
    def train_step(self) -> int:
        return self._clock().step

    @train_step.setter
    def train_step(self, step: int) -> None:
        self._clock().step = step

    @property
    def sampling_client_updated_event(self) -> asyncio.Event:
        return self._clock().updated

    @sampling_client_updated_event.setter
    def sampling_client_updated_event(self, event: asyncio.Event) -> None:
        self._clock().updated = event

    # The state of the policy the loops train, the leader, read and written under the names the shared state gave it
    # when it held that state itself.

    def _leader(self) -> Policy:
        assert self.policies is not None  # __post_init__ made them
        return self.policies[self.policies.leader]

    @property
    def model_info(self) -> ModelInfo:
        return self._leader().info

    @model_info.setter
    def model_info(self, info: ModelInfo) -> None:
        self._leader().info = info

    @property
    def service_client(self) -> tinker.ServiceClient:
        return self._leader().service_client

    @service_client.setter
    def service_client(self, client: tinker.ServiceClient) -> None:
        self._leader().service_client = client

    @property
    def training_client(self) -> tinker.TrainingClient:
        return self._leader().training_client

    @training_client.setter
    def training_client(self, client: tinker.TrainingClient) -> None:
        self._leader().training_client = client

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
