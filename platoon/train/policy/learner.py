"""How a step updates a policy (:class:`Learner`): through the training client the policy holds, on its own Tinker
service, with its own optimizer, under its objective (:mod:`platoon.train.objectives`).

A step opens one update per minibatch (:class:`PolicyUpdate`): the forward_backward calls of its microbatches, as the
step sends each, then one optimizer step over them all, each result awaited under its deadline, and the update's
metrics: the objective's of each microbatch, averaged, and the optimizer's.  The rl trainer's step, ``TrainStep`` in
:mod:`platoon.train.trainer.loops`, builds the learner of the policy it trains and opens its updates.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

import tinker

from platoon.train.objectives.api import Objective
from platoon.train.policy.policy import Policy
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


class Learner:
    """Updates one trainable policy (``policy``): sends its rows through the policy's training client under
    ``objective``, and steps its optimizer with the policy's optimizer parameters.  Each result is awaited with ``wait``
    (the trainer's heartbeat) under its deadline in ``timeouts``, read as it is awaited, and the update's metrics go to
    ``tracker``."""

    def __init__(
        self,
        policy: Policy,
        *,
        objective: Objective,
        timeouts: UpdateTimeouts,
        wait: "Waiter",
        tracker: StatsTracker,
    ) -> None:
        self.policy = policy
        self.objective = objective
        self.timeouts = timeouts
        self.wait = wait
        self.tracker = tracker

    def update(self) -> "PolicyUpdate":
        """A minibatch's update of the policy, which the step sends each microbatch to and then applies."""

        return PolicyUpdate(self)


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
        datums as they trained, side channels included, for the objective's metrics."""

        objective = self.learner.objective
        self.originals.append(originals)
        logger.debug(
            f"Submitting forward_backward_async with {len(datums)} datums "
            f"(minibatch {minibatch}, microbatch {microbatch})"
        )
        fwd_bwd_submit_start = time.perf_counter()
        try:
            self.futures.append(
                await self.learner.policy.training_client.forward_backward_async(
                    datums, loss_fn=objective.loss_fn, loss_fn_config=objective.loss_fn_config
                )
            )
        except Exception as e:
            elapsed = time.perf_counter() - fwd_bwd_submit_start
            logger.exception(f"forward_backward_async submission failed after {elapsed:.1f}s: {e}")
            raise
        logger.debug(f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s")

    async def apply(self, tally: UpdateTally) -> None:
        """Step the optimizer over the microbatches sent (:meth:`submit_optim_step`), then await every result and count
        the update (:meth:`finish`)."""

        await self.submit_optim_step()
        await self.finish(tally)

    async def submit_optim_step(self) -> None:
        """Submit the optimizer step over the microbatches sent, with the policy's optimizer parameters."""

        policy = self.learner.policy
        optim_start = time.perf_counter()
        logger.debug("Submitting optim_step_async")
        try:
            self.optim_future = await policy.training_client.optim_step_async(policy.optimizer_params)
        except Exception as e:
            logger.exception(f"optim_step_async submission failed: {e}")
            raise
        logger.debug(f"optim_step_async submitted in {time.perf_counter() - optim_start:.1f}s")

    async def finish(self, tally: UpdateTally) -> None:
        """Await the microbatches' results and record the average of the objective's metrics of each, then await the
        optimizer step, count it (in ``tally``, and in the policy's ``optimizer_steps``) and record the optimizer's
        metrics."""

        all_training_metrics = await self._forward_backward_metrics()
        self._record_training_metrics(all_training_metrics)
        optim_result, optim_elapsed = await self._optim_step_result()
        tally.optimizer_steps += 1
        self.learner.policy.optimizer_steps += 1
        if optim_elapsed > 60.0:
            logger.warning(f"optim_step result took {optim_elapsed:.1f}s (slow)")
        else:
            logger.debug(f"optim_step result received in {optim_elapsed:.1f}s")

        # Extract optimizer metrics (e.g., grad_norm if grad_clip_norm is set)
        if optim_result and optim_result.metrics:
            for key, value in optim_result.metrics.items():
                self.learner.tracker.scalar(**{f"optim/{key}": value})

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
                self.learner.tracker.scalar(**{key: value})

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


# Its earlier name, kept as an alias for one minor release.
OptimizerUpdate = PolicyUpdate
