"""A rollout worker told to stop refuses what its rollout would start next (``entrypoint._refuse_calls``): its
model calls, its remote env sessions, and its sandboxes, so a rollout that unwinds slowly cannot start a sandbox that
nothing would stop."""

from __future__ import annotations

import asyncio

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.envs.ors import cleanup as ors_cleanup
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.base import SandboxSpec
from platoon.sandbox.process import ProcessProvider
from platoon.train.workers import entrypoint


def test_a_stopping_worker_starts_no_sandbox(tmp_path, monkeypatch):
    def no_model_bound():
        raise RuntimeError("no model bound")

    # This process's switches, put back after the test.
    monkeypatch.setattr(sandbox_cleanup, "_shut_down", False)
    monkeypatch.setattr(ors_cleanup, "_shut_down", False)
    monkeypatch.setattr("platoon.gateway.core.default_gateway", no_model_bound)
    provider = ProcessProvider(tmp_path / "sandboxes")

    entrypoint._refuse_calls()

    with pytest.raises(sandbox_cleanup.SandboxesShutDown):
        asyncio.run(provider.start(SandboxSpec(image="local")))
    assert not (tmp_path / "sandboxes").exists()  # nothing was laid out
