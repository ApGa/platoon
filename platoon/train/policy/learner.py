"""How a step updates a policy (:class:`Learner`): through the training client the policy holds, on its own Tinker
service, with its own optimizer, under its objective (:mod:`platoon.train.objectives`).

A step opens one update per minibatch (:class:`PolicyUpdate`): the forward_backward calls of its microbatches, as the
step sends each, then one optimizer step over them all, each result awaited under its deadline, and the update's
metrics: the objective's of each microbatch, averaged, and the optimizer's.  The rl trainer's step, ``TrainStep`` in
:mod:`platoon.train.trainer.loops`, builds the learner of the policy it trains and opens its updates; a step hook
updates the same policy through a learner of its own objective (:meth:`Learner.for_objective`).

Two of the run's rules hold here, for every update of any trainer (RFC 0021, rule 10): a step's first submission marks
the step open (its clock's ``step_open``), and a learner that fails closed refuses an optimizer step that the service
skipped or whose metrics are not finite (:class:`UpdateRejected`), before it is counted.  The clock also holds each
policy's open update, sent and not counted yet (``open_updates``): a policy has one at a time, since its optimizer
step would take in another's gradients, and a step's end refuses to start its hooks or to publish with one.  A step
hook's learner, and any learner made from it (:meth:`Learner.for_objective`), refuses to send or count anything outside
its hook's ``update()`` (``guard``).  In a run of several trained policies every learner is ``named``: an error of its
policy's service, in a step's own update or a step hook's, says whose service failed
(:func:`~platoon.train.policy.spec.failure_note`).
"""

import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

import tinker

from platoon.train.objectives.api import Objective
from platoon.train.policy.policy import Policy
from platoon.train.policy.spec import PolicyId, failure_note
from platoon.utils.stats_tracker import StatsTracker

if TYPE_CHECKING:
    from platoon.train.policy.checkpoints import Waiter

logger = logging.getLogger(__name__)


class UpdateTimeouts(Protocol):
    """The deadlines of an update's results, in seconds, under their ``train:`` names (a ``TrainConfig`` has them):
    each forward_backward result, and the optimizer step's."""

    forward_backward_timeout_seconds: float
    optim_step_timeout_seconds: float


class UpdateTally(Protocol):
    """What a step counts of its updates: the optimizer steps it took."""

    optimizer_steps: int


class OpenedSteps(Protocol):
    """The clock whose step a learner's first submission opens (``step_open``, rule 10), and which holds each policy's
    open update, from its first submission until it is counted (``open_updates``)."""

    step_open: bool
    open_updates: dict[PolicyId, Any]


class PrefixedTracker:
    """A stats tracker as the learner of a trained policy other than the leader records into it, in a run that trains
    several: every metric under ``prefix`` (``policies/<id>/``) in front of its name, in ``tracker``, the step's own."""

    def __init__(self, tracker: Any, prefix: str) -> None:
        self.tracker = tracker
        self.prefix = prefix

    def scalar(self, **values: float) -> None:
        self.tracker.scalar(**{f"{self.prefix}{name}": value for name, value in values.items()})


class UpdateRejected(RuntimeError):
    """An optimizer step that a learner which fails closed refused: the service skipped it, or a forward_backward or
    optimizer metric is not finite.  The update is not counted, and the step never publishes."""


class Learner:
    """Updates one trainable policy (``policy``): sends its rows through the policy's training client under
    ``objective``, and steps its optimizer with the policy's optimizer parameters.  Each result is awaited with ``wait``
    (the trainer's heartbeat) under its deadline in ``timeouts``, read as it is awaited, and the update's metrics go to
    ``tracker``, under ``metric_prefix``.  Before each optimizer step it awaits ``before_update`` (the rl trainer's
    barrier of the starting policy's evaluation), its first submission of a step sets ``clock.step_open``, an update
    is open on the clock until it is counted, and with ``fail_closed`` a skipped or nonfinite optimizer result raises
    :class:`UpdateRejected` before it is counted.  ``guard``, when set, is called before each submission, optimizer
    step and count, and raises when the learner may not make them (a step hook's, outside its ``update()``; the
    run's, as the step's end runs its step hooks).  ``named``: whether an error of the policy's service in an update
    gets a note that names the policy and its service (:func:`~platoon.train.policy.spec.failure_note`), as in a run
    of several trained policies; with one, its errors are raised as they are."""

    def __init__(
        self,
        policy: Policy,
        *,
        objective: Objective,
        timeouts: UpdateTimeouts,
        wait: "Waiter",
        tracker: StatsTracker | PrefixedTracker,
        metric_prefix: str = "",
        fail_closed: bool = False,
        before_update: Callable[[], Awaitable[None]] | None = None,
        clock: OpenedSteps | None = None,
        guard: Callable[[], None] | None = None,
        named: bool = False,
    ) -> None:
        self.policy = policy
        self.objective = objective
        self.timeouts = timeouts
        self.wait = wait
        self.tracker = tracker
        self.metric_prefix = metric_prefix
        self.fail_closed = fail_closed
        self.before_update = before_update
        self.clock = clock
        self.guard = guard
        self.named = named

    def update(self) -> "PolicyUpdate":
        """A minibatch's update of the policy, which the step sends each microbatch to and then applies."""

        return PolicyUpdate(self)

    def for_objective(
        self, objective: Objective, *, metric_prefix: str = "", guard: Callable[[], None] | None = None
    ) -> "Learner":
        """A learner of the same policy, its waits, deadlines, barrier, clock, checks and failure notes (``named``),
        under ``objective``, whose metrics go under ``metric_prefix`` and whose sends ``guard`` checks (a step hook's:
        ``<name>/``, and its ``update()`` only); None: this learner's own guard, so that a step hook's learner under
        another objective refuses what it refuses."""

        return Learner(
            self.policy,
            objective=objective,
            timeouts=self.timeouts,
            wait=self.wait,
            tracker=self.tracker,
            metric_prefix=metric_prefix,
            fail_closed=self.fail_closed,
            before_update=self.before_update,
            clock=self.clock,
            guard=guard if guard is not None else self.guard,
            named=self.named,
        )


class PolicyUpdate:
    """One minibatch's update of a learner's policy: the forward_backward calls of its microbatches, as the step sends
    each (:meth:`submit`), then the optimizer step and every result (:meth:`apply`)."""

    # The optimizer step's result, once it is submitted (submit_optim_step).
    optim_future: tinker.APIFuture[tinker.OptimStepResponse]

    def __init__(self, learner: Learner) -> None:
        self.learner = learner
        self.futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        # Keep original datums (with mask and logprobs) for metrics computation
        self.originals: list[list[tinker.Datum]] = []

    @property
    def submitted(self) -> bool:
        """Whether a microbatch of the minibatch was sent: when every one had a zero objective, or no datums, the
        minibatch takes no optimizer update."""

        return len(self.futures) > 0

    async def submit(
        self, datums: list[tinker.Datum], originals: list[tinker.Datum], *, minibatch: int, microbatch: int
    ) -> None:
        """Send a microbatch's ``datums`` to forward_backward, under the objective's loss; keep its ``originals``, the
        datums as they trained, side channels included, for the objective's metrics.  The step is open from now on
        (rule 10): an interrupt saves no checkpoint of it until it ends; and so is the update, until it is counted
        (:meth:`_open`)."""

        learner = self.learner
        objective = learner.objective
        self._check_guard()
        if learner.clock is not None:
            self._open(learner.clock)
        self.originals.append(originals)
        logger.debug(
            f"Submitting forward_backward_async with {len(datums)} datums "
            f"(minibatch {minibatch}, microbatch {microbatch})"
        )
        fwd_bwd_submit_start = time.perf_counter()
        try:
            self.futures.append(
                await learner.policy.training_client.forward_backward_async(
                    datums, loss_fn=objective.loss_fn, loss_fn_config=objective.loss_fn_config
                )
            )
        except Exception as e:
            elapsed = time.perf_counter() - fwd_bwd_submit_start
            logger.exception(f"forward_backward_async submission failed after {elapsed:.1f}s: {e}")
            self._note(e, "forward_backward")
            raise
        logger.debug(f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s")

    async def apply(self, tally: UpdateTally | None = None, *, adam_params: tinker.AdamParams | None = None) -> None:
        """Once the learner's barrier lets the update through (``before_update``), step the optimizer over the
        microbatches sent (:meth:`submit_optim_step`), with ``adam_params`` for this update alone (None: the policy's
        optimizer parameters), then await every result and count the update (:meth:`finish`), in ``tally`` too when
        given (a step's)."""

        if self.learner.before_update is not None:
            await self.learner.before_update()
        await self.submit_optim_step(adam_params)
        await self.finish(tally)

    async def submit_optim_step(self, adam_params: tinker.AdamParams | None = None) -> None:
        """Submit the optimizer step over the microbatches sent, with ``adam_params``, else the policy's optimizer
        parameters."""

        self._check_guard()
        policy = self.learner.policy
        optim_start = time.perf_counter()
        logger.debug("Submitting optim_step_async")
        try:
            self.optim_future = await policy.training_client.optim_step_async(
                policy.optimizer_params if adam_params is None else adam_params
            )
        except Exception as e:
            logger.exception(f"optim_step_async submission failed: {e}")
            self._note(e, "optim_step")
            raise
        logger.debug(f"optim_step_async submitted in {time.perf_counter() - optim_start:.1f}s")

    async def finish(self, tally: UpdateTally | None = None) -> None:
        """Await the microbatches' results and record the average of the objective's metrics of each, then await the
        optimizer step, count it (in the policy's ``optimizer_steps``, and in ``tally`` when given) and record the
        optimizer's metrics.  A learner that fails closed raises :class:`UpdateRejected` first for an optimizer step the
        service skipped, or a forward_backward or optimizer metric that is not finite.  Counted, the update is no
        longer open."""

        self._check_guard()
        try:
            all_training_metrics = await self._forward_backward_metrics()
            self._record_training_metrics(all_training_metrics)
            optim_result, optim_elapsed = await self._optim_step_result()
            if self.learner.fail_closed:
                _refuse_skipped_or_nonfinite(optim_result)
        except Exception as error:
            self._note(error, "the update's results")
            raise
        if tally is not None:
            tally.optimizer_steps += 1
        self.learner.policy.optimizer_steps += 1
        if self.learner.clock is not None:
            self.learner.clock.open_updates.pop(self.learner.policy.spec.id, None)
        if optim_elapsed > 60.0:
            logger.warning(f"optim_step result took {optim_elapsed:.1f}s (slow)")
        else:
            logger.debug(f"optim_step result received in {optim_elapsed:.1f}s")

        # Extract optimizer metrics (e.g., grad_norm if grad_clip_norm is set)
        if optim_result and optim_result.metrics:
            for key, value in optim_result.metrics.items():
                self.learner.tracker.scalar(**{f"{self.learner.metric_prefix}optim/{key}": value})

    def _note(self, error: Exception, doing: str) -> None:
        """A note on ``error`` of the policy's service that names the policy, its service and what failed (``doing``),
        when the learner is ``named`` (a run of several trained policies)."""

        if self.learner.named:
            error.add_note(failure_note(self.learner.policy.spec, doing))

    def _check_guard(self) -> None:
        """Raise when the learner may not send or count anything now (its ``guard``: a step hook's learner, outside its
        hook's ``update()``)."""

        if self.learner.guard is not None:
            self.learner.guard()

    def _open(self, clock: OpenedSteps) -> None:
        """The step is open on ``clock`` (rule 10), and so is the update, from its first submission until it is counted
        (:meth:`finish`): its policy's one open update, since the policy's optimizer step would take in the gradients
        of another sent meanwhile, which is refused."""

        clock.step_open = True
        if self.originals:
            return  # a later microbatch of the open update
        policy_id = self.learner.policy.spec.id  # its spec's: a policy that a test made as platoon 0.4.0 did has no id
        if policy_id in clock.open_updates:
            raise RuntimeError(
                f"an update of policy {policy_id!r} is open, sent and not applied yet: apply it before sending "
                "another, or one optimizer step would take in both"
            )
        clock.open_updates[policy_id] = self

    async def _forward_backward_metrics(self) -> list[dict[str, float]]:
        """Consume all forward backward results and compute training metrics: each microbatch's, in order."""

        learner = self.learner
        all_training_metrics: list[dict[str, float]] = []
        for microbatch_num, (forward_backward_future, original_datums) in enumerate(zip(self.futures, self.originals)):
            result_start = time.perf_counter()
            logger.debug(f"Waiting for forward_backward result (microbatch {microbatch_num})")
            try:
                # Use a wrapper to log periodic "still waiting" messages.
                # NOTE: every microbatch is submitted before this loop and the service
                # drains them in order, so the await on microbatch 0 spans the whole
                # accumulated window; the deadline is sized for a full step.
                forward_backward_result = await learner.wait(
                    forward_backward_future.result_async(),
                    name=f"forward_backward (microbatch {microbatch_num})",
                    total_timeout=learner.timeouts.forward_backward_timeout_seconds,
                )
            except Exception as e:
                logger.exception(
                    f"forward_backward result_async failed after {time.perf_counter() - result_start:.1f}s: {e}"
                )
                raise
            if learner.fail_closed:
                _refuse_nonfinite("forward_backward", getattr(forward_backward_result, "metrics", None))
            result_elapsed = time.perf_counter() - result_start
            if result_elapsed > 60.0:
                logger.warning(f"forward_backward result took {result_elapsed:.1f}s (slow)")
            else:
                logger.debug(f"forward_backward result received in {result_elapsed:.1f}s")

            # Compute training metrics (KL, entropy, importance weights, clipping stats)
            try:
                microbatch_metrics = learner.objective.metrics(original_datums, forward_backward_result)
                all_training_metrics.append(microbatch_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute training metrics for microbatch {microbatch_num}: {e}")
        return all_training_metrics

    def _record_training_metrics(self, all_training_metrics: list[dict[str, float]]) -> None:
        """Aggregate and log training metrics across microbatches: each metric's mean over those that have it."""

        if all_training_metrics:
            aggregated_metrics: dict[str, float] = {}
            for key in all_training_metrics[0].keys():
                values = [m[key] for m in all_training_metrics if key in m]
                if values:
                    aggregated_metrics[key] = sum(values) / len(values)
            for key, value in aggregated_metrics.items():
                self.learner.tracker.scalar(**{f"{self.learner.metric_prefix}{key}": value})

    async def _optim_step_result(self) -> tuple[Any, float]:
        """Wait for optimizer step to complete: its result, and the seconds it took."""

        learner = self.learner
        logger.debug("Waiting for optim_step result")
        optim_result_start = time.perf_counter()
        try:
            # The optimizer step runs after the window is finalized and is fast.
            optim_result = await learner.wait(
                self.optim_future.result_async(),
                name="optim_step",
                total_timeout=learner.timeouts.optim_step_timeout_seconds,
            )
        except Exception as e:
            logger.exception(
                f"optim_step result_async failed after {time.perf_counter() - optim_result_start:.1f}s: {e}"
            )
            raise
        return optim_result, time.perf_counter() - optim_result_start


def _refuse_skipped_or_nonfinite(optim_result: Any) -> None:
    """Raise :class:`UpdateRejected` for an optimizer result whose metrics say ``skipped``, or one not finite."""

    metrics = getattr(optim_result, "metrics", None) or {}
    if metrics.get("skipped"):
        raise UpdateRejected("the training service skipped the optimizer step: the update is refused before publishing")
    _refuse_nonfinite("optim_step", metrics)


def _refuse_nonfinite(call: str, metrics: Mapping[str, Any] | None) -> None:
    """Raise :class:`UpdateRejected` for a numeric metric of ``call``'s result that is not finite."""

    for key, value in (metrics or {}).items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise UpdateRejected(f"{call} returned {key} = {value}: the update is refused before publishing")


# Its earlier name, kept as an alias for one minor release.
OptimizerUpdate = PolicyUpdate
