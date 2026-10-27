"""A policy's saves through its training client (``platoon.train.policy.checkpoints``): a save for the sampler or a
sampling client that fails is logged with its error before it is raised, so the run's log names the call that failed.

What each step saves, and when, is ``tests/train/checkpoints``."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from platoon.train.policy.checkpoints import sampler_from, save_weights_only


class _Refusing:
    """A training client whose saves for the sampler and sampling clients fail."""

    async def save_weights_and_get_sampling_client_async(self):
        raise RuntimeError("the service went away")

    def create_sampling_client(self, path):
        raise RuntimeError("the service went away")


def test_a_failed_save_for_the_sampler_is_logged_before_it_is_raised(caplog):
    client: Any = _Refusing()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="went away"):
        asyncio.run(save_weights_only(client))

    assert "save_weights_and_get_sampling_client_async failed: the service went away" in caplog.text


def test_a_failed_sampling_client_of_a_full_checkpoint_is_logged_before_it_is_raised(caplog):
    client: Any = _Refusing()

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="went away"):
        sampler_from(client, {"sampler_path": "tinker://m/sampler_weights/000002"})

    assert "create_sampling_client failed: the service went away" in caplog.text
