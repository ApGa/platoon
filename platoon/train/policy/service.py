"""A policy's training service: the training client a start opens on it (new, from a checkpoint's weights, or from the
state a resume takes up: :func:`create_training_client`), asked again after each refusal that asking again may change
(:func:`open_training_client`).

``PlatoonTinkerRLTrainer`` opens main's through its methods ``_open_training_client`` and ``_create_training_client``,
which call these: to replace one, replace the method, on the class or an instance, not the function.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

import tinker
from tinker_cookbook import checkpoint_utils

from platoon.train.policy.spec import PolicySpec

logger = logging.getLogger(__name__)


# How long a start asks the training service for its training client before it gives up, and the waits between two
# asks: the first, doubled after each refusal up to the longest.  A self-hosted service that trains one model at a
# time may refuse a new start until the session of a dead one expires (miles-tinker's full fine-tuning lease is ten
# minutes), and a service that restarts refuses every connection for a while.
TRAINING_CLIENT_OPEN_TIMEOUT_S = 20 * 60.0
TRAINING_CLIENT_FIRST_WAIT_S = 5.0
TRAINING_CLIENT_LONGEST_WAIT_S = 60.0
# The service's answers that asking again cannot change: a key it does not take, a permission it does not give, a
# checkpoint it does not have, a request it cannot read.
_FINAL_REFUSALS = frozenset({401, 403, 404, 422})


def _service_refusal(exc: BaseException) -> bool:
    """Whether ``exc``, raised while opening a training client, is the training service's refusal (or its silence)
    that asking again may change: any Tinker error, a lost connection or a timeout, but not a refusal in
    :data:`_FINAL_REFUSALS`.  The Tinker SDK raises a ``ValueError`` for a request whose result it could not
    retrieve, with the service's answer as its cause."""

    error: BaseException = exc
    if isinstance(error, ValueError) and isinstance(error.__cause__, tinker.TinkerError):
        error = error.__cause__
    if isinstance(error, tinker.APIStatusError):
        return error.status_code not in _FINAL_REFUSALS
    return isinstance(error, (tinker.TinkerError, ConnectionError, TimeoutError))


async def create_training_client(
    service_client: tinker.ServiceClient,
    resume_info: checkpoint_utils.CheckpointRecord | None,
    spec: PolicySpec,
) -> tinker.TrainingClient:
    """Ask the policy's service (``service_client``) once for the training client of the policy ``spec`` declares: the
    one a resume takes up (``resume_info``: its weights and its optimizer's state), else a new run's, from the weights
    ``spec.load_checkpoint_path`` names (with a new optimizer), else a new LoRA adapter of its model."""

    if resume_info:
        # Resuming interrupted training - load optimizer state for proper continuation
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            resume_info.state_path
        )
        logger.info(f"Resumed training from {resume_info.state_path}")
    elif spec.load_checkpoint_path:
        # Starting fresh from a checkpoint - load weights only (fresh optimizer)
        training_client = await service_client.create_training_client_from_state_async(spec.load_checkpoint_path)
        logger.info(f"Loaded weights from {spec.load_checkpoint_path}")
    else:
        training_client = await service_client.create_lora_training_client_async(spec.model_name, rank=spec.lora_rank)
        logger.info(f"Created new training client for {spec.model_name} with rank {spec.lora_rank}")

    return training_client


async def open_training_client(
    create: Callable[[], Awaitable[tinker.TrainingClient]], stop: asyncio.Event | None = None
) -> tinker.TrainingClient | None:
    """The training client ``create`` asks the service for, asked again after each refusal until the service accepts
    it, for up to :data:`TRAINING_CLIENT_OPEN_TIMEOUT_S`, each refusal logged; None once ``stop`` is set between two
    asks (training was asked to stop before it began).

    A service may refuse a start while it still holds the session of a start that died, until that session
    expires, or answer nothing while it restarts.  A refusal that asking again cannot change
    (:func:`_service_refusal`), and any error that is not the service's, is raised at once.
    """

    first = time.monotonic()
    wait = TRAINING_CLIENT_FIRST_WAIT_S
    attempt = 0
    while True:
        attempt += 1
        try:
            return await create()
        except Exception as exc:
            waited = time.monotonic() - first
            left = TRAINING_CLIENT_OPEN_TIMEOUT_S - waited
            if not _service_refusal(exc):
                raise
            if left <= 0:
                exc.add_note(
                    f"The training service refused the training client {attempt} times in {waited:.0f}s; a start "
                    f"asks for {TRAINING_CLIENT_OPEN_TIMEOUT_S:.0f}s"
                )
                raise
            pause = min(wait, left)
            reason = " ".join(f"{type(exc).__name__}: {exc}".split())[:500].rstrip(".")
            logger.warning(
                f"The training service refused the training client (attempt {attempt}, {waited:.0f}s after the "
                f"first): {reason}. Asking again in {pause:.0f}s, for up to {left:.0f}s more"
            )
            if stop is None:
                await asyncio.sleep(pause)
            else:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=pause)
                if stop.is_set():
                    logger.info("Stopped asking the training service for a training client: training was stopped")
                    return None
            wait = min(wait * 2, TRAINING_CLIENT_LONGEST_WAIT_S)
