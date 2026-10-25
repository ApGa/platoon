"""A minibatch's optimizer update, the third stage of a step (:class:`OptimizerUpdate`): its microbatches'
forward_backward calls as the step sends them, then the optimizer step, their results under deadlines, and the
update's metrics: the training metrics of each microbatch (:func:`compute_training_metrics`: KL, entropy, importance
weights and clipping), averaged, and the optimizer's.

The step itself, ``TrainStep`` in :mod:`platoon.train.trainer.loops`, sends each microbatch and applies each update.
"""

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import tinker
import torch
from tinker_cookbook.rl.metrics import compute_kl_sample_train

if TYPE_CHECKING:
    from platoon.train.step.collect import StepTally
    from platoon.train.step.microbatch import Prepared
    from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
    from platoon.train.trainer.state import TrainLoopSharedState

logger = logging.getLogger(__name__)


def compute_training_metrics(
    datums: list[tinker.Datum],
    forward_backward_result: tinker.ForwardBackwardOutput,
    loss_fn: str,
    loss_fn_config: dict[str, Any],
) -> dict[str, float]:
    """Compute training metrics from forward_backward results.

    Extracts training logprobs from the forward pass and computes:
    - KL divergence between sampling and training policies
    - Entropy (approximated from sampling logprobs)
    - Importance weight statistics
    - Clipping statistics (for cispo/ppo loss functions)

    Args:
        datums: Original datums with mask and sampling logprobs in loss_fn_inputs
        forward_backward_result: Result from forward_backward containing training logprobs
        loss_fn: Loss function type ('cispo', 'ppo', 'importance_sampling', etc.)
        loss_fn_config: Loss function config with clip thresholds

    Returns:
        Dictionary of training metrics
    """
    # Extract training logprobs from forward_backward result
    training_logprobs_list: list[torch.Tensor] = []
    for output in forward_backward_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_list.append(training_logprobs)

    # Use tinker_cookbook's function for KL and entropy
    kl_metrics = compute_kl_sample_train(datums, training_logprobs_list)

    # Compute importance weight statistics
    all_importance_weights: list[torch.Tensor] = []
    all_clipped_low: list[torch.Tensor] = []
    all_clipped_high: list[torch.Tensor] = []

    for datum, training_logprobs in zip(datums, training_logprobs_list):
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        action_mask = datum.loss_fn_inputs["mask"].to_torch() > 0

        # Extract action token logprobs
        sampling_logprobs_actions = sampling_logprobs[action_mask]
        training_logprobs_actions = training_logprobs[action_mask]

        if len(sampling_logprobs_actions) > 0:
            # Importance weight = exp(new_logprob - old_logprob) = π_new / π_old
            log_ratio = training_logprobs_actions - sampling_logprobs_actions
            importance_weights = torch.exp(log_ratio)
            all_importance_weights.append(importance_weights)

            # Compute clipping stats for cispo/ppo (both use clip_low_threshold/clip_high_threshold)
            if loss_fn in ("cispo", "ppo"):
                clip_low = loss_fn_config.get("clip_low_threshold", 0.0)
                clip_high = loss_fn_config.get("clip_high_threshold", 5.0)

                clipped_low = importance_weights < clip_low
                clipped_high = importance_weights > clip_high
                all_clipped_low.append(clipped_low)
                all_clipped_high.append(clipped_high)

    metrics = dict(kl_metrics)

    if all_importance_weights:
        flat_weights = torch.cat(all_importance_weights)
        metrics["optim/importance_weight_mean"] = flat_weights.mean().item()
        metrics["optim/importance_weight_std"] = flat_weights.std().item()
        metrics["optim/importance_weight_min"] = flat_weights.min().item()
        metrics["optim/importance_weight_max"] = flat_weights.max().item()

        # Clipping stats for cispo/ppo
        if loss_fn in ("cispo", "ppo") and all_clipped_low:
            flat_clipped_low = torch.cat(all_clipped_low)
            flat_clipped_high = torch.cat(all_clipped_high)
            total_tokens = len(flat_clipped_low)
            metrics["optim/clip_frac_low"] = flat_clipped_low.sum().item() / total_tokens
            metrics["optim/clip_frac_high"] = flat_clipped_high.sum().item() / total_tokens
            metrics["optim/clip_frac_total"] = (flat_clipped_low | flat_clipped_high).sum().item() / total_tokens

    return metrics


class OptimizerUpdate:
    """One minibatch's optimizer update: the forward_backward calls of its microbatches, as the step sends each
    (:meth:`submit`), then the optimizer step and every result (:meth:`apply`), each result awaited with heartbeats
    under its deadline (``train.forward_backward_timeout_seconds``, ``train.optim_step_timeout_seconds``).

    It sends them through ``training_client``, under the loss ``loss_fn`` with ``loss_fn_config``, and steps the
    optimizer with ``optimizer_params``, as the steps' context holds them (``StepContext`` in
    :mod:`platoon.train.trainer.loops`)."""

    def __init__(
        self,
        trainer: "PlatoonTinkerRLTrainer",
        state: "TrainLoopSharedState",
        *,
        training_client: tinker.TrainingClient,
        loss_fn: str,
        loss_fn_config: dict[str, Any],
        optimizer_params: tinker.AdamParams,
    ) -> None:
        self.trainer = trainer
        self.state = state
        self.training_client = training_client
        self.loss_fn = loss_fn
        self.loss_fn_config = loss_fn_config
        self.optimizer_params = optimizer_params
        self.futures: list[tinker.APIFuture[tinker.ForwardBackwardOutput]] = []
        # Keep original datums (with mask and logprobs) for metrics computation
        self.originals: list[list[tinker.Datum]] = []

    @property
    def submitted(self) -> bool:
        """Whether a microbatch of the minibatch was sent: when every one had a zero objective, or no datums, the
        minibatch takes no optimizer update."""

        return len(self.futures) > 0

    async def submit(self, prepared: "Prepared", minibatch_num: int, microbatch_num: int) -> None:
        """Send a prepared microbatch to forward_backward; keep its original datums for the training metrics."""

        self.originals.append(prepared.originals)
        logger.debug(
            f"Submitting forward_backward_async with {len(prepared.datums)} datums "
            f"(minibatch {minibatch_num}, microbatch {microbatch_num})"
        )
        fwd_bwd_submit_start = time.perf_counter()
        try:
            self.futures.append(
                await self.training_client.forward_backward_async(
                    prepared.datums, loss_fn=self.loss_fn, loss_fn_config=self.loss_fn_config
                )
            )
        except Exception as e:
            elapsed = time.perf_counter() - fwd_bwd_submit_start
            logger.exception(f"forward_backward_async submission failed after {elapsed:.1f}s: {e}")
            raise
        logger.debug(f"forward_backward_async submitted in {time.perf_counter() - fwd_bwd_submit_start:.1f}s")

    async def apply(self, tally: "StepTally", *, training_metrics: Callable[..., dict[str, float]]) -> None:
        """Step the optimizer over the microbatches sent, await their results and record the average of their training
        metrics (each microbatch's from ``training_metrics``, as :func:`compute_training_metrics` computes them; the
        step passes the one its module names at each call), then await the optimizer step, count it (in ``tally``, and
        in the run's ``optimizer_steps``) and record the optimizer's metrics."""

        optim_future = await self._submit_optim_step()
        all_training_metrics = await self._forward_backward_metrics(training_metrics)
        self._record_training_metrics(all_training_metrics)
        optim_result, optim_elapsed = await self._optim_step_result(optim_future)
        tally.optimizer_steps += 1
        self.state.optimizer_steps = getattr(self.state, "optimizer_steps", 0) + 1
        if optim_elapsed > 60.0:
            logger.warning(f"optim_step result took {optim_elapsed:.1f}s (slow)")
        else:
            logger.debug(f"optim_step result received in {optim_elapsed:.1f}s")

        # Extract optimizer metrics (e.g., grad_norm if grad_clip_norm is set)
        if optim_result and optim_result.metrics:
            for key, value in optim_result.metrics.items():
                self.state.train_tracker.scalar(**{f"optim/{key}": value})

    async def _submit_optim_step(self) -> "tinker.APIFuture[tinker.OptimStepResponse]":
        """Submit the optimizer step over the microbatches sent: its future."""

        optim_start = time.perf_counter()
        logger.debug("Submitting optim_step_async")
        try:
            optim_future = await self.training_client.optim_step_async(self.optimizer_params)
        except Exception as e:
            logger.exception(f"optim_step_async submission failed: {e}")
            raise
        logger.debug(f"optim_step_async submitted in {time.perf_counter() - optim_start:.1f}s")
        return optim_future

    async def _forward_backward_metrics(
        self, training_metrics: Callable[..., dict[str, float]]
    ) -> list[dict[str, float]]:
        """Consume all forward backward results and compute training metrics: each microbatch's, in order."""

        trainer = self.trainer
        all_training_metrics: list[dict[str, float]] = []
        for microbatch_num, (forward_backward_future, original_datums) in enumerate(zip(self.futures, self.originals)):
            result_start = time.perf_counter()
            logger.debug(f"Waiting for forward_backward result (microbatch {microbatch_num})")
            try:
                # Use a wrapper to log periodic "still waiting" messages.
                # NOTE: every microbatch is submitted before this loop and the service
                # drains them in order, so the await on microbatch 0 spans the whole
                # accumulated window; the deadline is sized for a full step.
                forward_backward_result = await trainer._await_with_heartbeat(
                    forward_backward_future.result_async(),
                    name=f"forward_backward (microbatch {microbatch_num})",
                    heartbeat_interval=60.0,
                    watchdog=self.state.watchdog,
                    total_timeout=trainer.config.train.forward_backward_timeout_seconds,
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
                microbatch_metrics = training_metrics(
                    datums=original_datums,
                    forward_backward_result=forward_backward_result,
                    loss_fn=self.loss_fn,
                    loss_fn_config=self.loss_fn_config,
                )
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
                self.state.train_tracker.scalar(**{key: value})

    async def _optim_step_result(self, optim_future: "tinker.APIFuture[tinker.OptimStepResponse]") -> tuple[Any, float]:
        """Wait for optimizer step to complete: its result, and the seconds it took."""

        logger.debug("Waiting for optim_step result")
        optim_result_start = time.perf_counter()
        try:
            # The optimizer step runs after the window is finalized and is fast.
            optim_result = await self.trainer._await_with_heartbeat(
                optim_future.result_async(),
                name="optim_step",
                heartbeat_interval=60.0,
                watchdog=self.state.watchdog,
                total_timeout=self.trainer.config.train.optim_step_timeout_seconds,
            )
        except Exception as e:
            logger.exception(
                f"optim_step result_async failed after {time.perf_counter() - optim_result_start:.1f}s: {e}"
            )
            raise
        return optim_result, time.perf_counter() - optim_result_start
