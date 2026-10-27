"""A policy's own saves, through its one training client: a full checkpoint (:func:`save_full_checkpoint`: the weights
and the optimizer's state, a row of its ``checkpoints.jsonl``), the weights alone for its sampler
(:func:`save_weights_only`), and the sampling client of the sampler weights a full checkpoint saved
(:func:`sampler_from`).

What a run's checkpoint holds besides, and when one is due, is :mod:`platoon.train.checkpoints.saving`.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import tinker
from tinker_cookbook import checkpoint_utils

logger = logging.getLogger(__name__)


# Upper bound for one save_checkpoint_async (full-parameter save_state of a
# 35B model is ~490 GB and takes minutes at 1-2 GB/s).
SAVE_CHECKPOINT_TIMEOUT_S = 7200.0


# How a save is awaited: ``wait(awaitable, *, name, total_timeout)``, under the trainer's heartbeat
# (``_await_with_heartbeat``), which logs while it waits and feeds the watchdog the waiter was given, if any.
Waiter = Callable[..., Awaitable[Any]]


async def save_full_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    *,
    log_path: str,
    loop_state: dict[str, Any],
    wait: Waiter,
) -> dict[str, str]:
    """Save full checkpoint ``name`` (the weights and the optimizer's state, ``kind="both"``), which
    ``checkpoints.jsonl`` in ``log_path`` records with ``loop_state``.  A full-parameter save takes minutes, so it runs
    under ``wait``, for at most :data:`SAVE_CHECKPOINT_TIMEOUT_S`.  The paths the save wrote (``state_path``,
    ``sampler_path``)."""

    return await wait(
        checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name=name,
            log_path=log_path,
            loop_state=loop_state,
            kind="both",
        ),
        name="save_checkpoint",
        total_timeout=SAVE_CHECKPOINT_TIMEOUT_S,
    )


async def save_weights_only(training_client: tinker.TrainingClient) -> tinker.SamplingClient:
    """The sampling client of the weights after a step that saves no full checkpoint, from a weights-only save."""

    try:
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    except Exception as e:
        logger.exception(f"save_weights_and_get_sampling_client_async failed: {e}")
        raise
    logger.debug("Weights saved and sampling client created")
    return sampling_client


def sampler_from(training_client: tinker.TrainingClient, path_dict: dict[str, str]) -> tinker.SamplingClient:
    """The sampling client of the sampler weights that a full checkpoint saved (``path_dict["sampler_path"]``)."""

    logger.info(f"Checkpoint async completed. Creating sampling client from {path_dict['sampler_path']}")
    try:
        sampling_client = training_client.create_sampling_client(path_dict["sampler_path"])
    except Exception as e:
        logger.exception(f"create_sampling_client failed: {e}")
        raise
    logger.info("Sampling client created successfully")
    return sampling_client
