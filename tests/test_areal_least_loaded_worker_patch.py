from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_patches_module():
    # Importing ``platoon.train.areal`` applies every runtime patch and pulls in
    # the full distributed training stack.  Load this standalone, stdlib-only
    # module directly so the unit test remains isolated and CPU-only.
    name = "platoon_areal_least_loaded_patch_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/train/areal/patches.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _PendingTask:
    task_id: int
    gate: asyncio.Event = field(default_factory=asyncio.Event)
    assigned_rank: int | None = None
    error: Exception | None = None


def _install_fake_rollout_controller(monkeypatch):
    class FakeRolloutController:
        def __init__(self) -> None:
            self.workers = ["worker-0", "worker-1"]
            self._current_worker_idx = 0

        def _choose_worker(self):
            rank = self._current_worker_idx
            self._current_worker_idx = (rank + 1) % len(self.workers)
            return self.workers[rank], rank

        def _create_submit_callback(self, pending_task):
            async def submit_then_wait():
                _, rank = self._choose_worker()
                pending_task.assigned_rank = rank
                await pending_task.gate.wait()
                if pending_task.error is not None:
                    raise pending_task.error
                return pending_task.task_id

            return submit_then_wait

    areal = types.ModuleType("areal")
    infra = types.ModuleType("areal.infra")
    controller = types.ModuleType("areal.infra.controller")
    rollout_controller = types.ModuleType("areal.infra.controller.rollout_controller")
    rollout_controller.RolloutController = FakeRolloutController
    monkeypatch.setitem(sys.modules, "areal", areal)
    monkeypatch.setitem(sys.modules, "areal.infra", infra)
    monkeypatch.setitem(sys.modules, "areal.infra.controller", controller)
    monkeypatch.setitem(sys.modules, "areal.infra.controller.rollout_controller", rollout_controller)
    return FakeRolloutController


@pytest.mark.asyncio
async def test_least_loaded_worker_reuses_worker_released_by_completed_callback(monkeypatch):
    controller_cls = _install_fake_rollout_controller(monkeypatch)
    patches = _load_patches_module()
    patches._patch_rollout_controller_least_loaded_workers()
    controller = controller_cls()

    first = _PendingTask(1)
    second = _PendingTask(2)
    first_task = asyncio.create_task(controller._create_submit_callback(first)())
    second_task = asyncio.create_task(controller._create_submit_callback(second)())
    await asyncio.sleep(0)
    assert [first.assigned_rank, second.assigned_rank] == [0, 1]

    second.gate.set()
    assert await second_task == 2

    third = _PendingTask(3)
    third_task = asyncio.create_task(controller._create_submit_callback(third)())
    await asyncio.sleep(0)
    assert third.assigned_rank == 1

    # Error and cancellation exits must both release their load slots.
    third.error = RuntimeError("submit failed")
    third.gate.set()
    with pytest.raises(RuntimeError, match="submit failed"):
        await third_task
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    state = controller._platoon_worker_load_state
    assert state["loads"] == [0, 0]
    assert state["assignments"] == {}
