"""Tests for definitive AReaL EngineCallError passthrough."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"


class EngineCallError(Exception):
    pass


class RequestTimeout(Exception):
    pass


class FixtureSlurmScheduler:
    def __init__(self):
        self.attempts = 0

    def call_engine(self, outcome, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                self.attempts += 1
                value = outcome(attempt)
                if isinstance(value, BaseException):
                    raise value
                if value != "service-unavailable":
                    return value
            except requests.exceptions.Timeout as e:
                del e
            except Exception as e:
                del e
            if attempt < max_retries:
                continue
        raise EngineCallError("max retries")

    async def async_call_engine(self, outcome, max_retries=3):
        for attempt in range(1, max_retries + 1):
            try:
                self.attempts += 1
                value = outcome(attempt)
                if isinstance(value, BaseException):
                    raise value
                if value != "service-unavailable":
                    return value
            except TimeoutError as e:
                del e
            except Exception as e:
                del e
            if attempt < max_retries:
                await asyncio.sleep(0)
        raise EngineCallError("max retries")


requests = ModuleType("requests")
requests.exceptions = ModuleType("requests.exceptions")
requests.exceptions.Timeout = RequestTimeout


def _load_patches_module():
    spec = importlib.util.spec_from_file_location(
        "platoon_areal_engine_call_retry_patch_test",
        PATCHES_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_scheduler_module():
    module = ModuleType("fake_areal_slurm_scheduler")
    scheduler_cls = type(
        "FixtureSlurmScheduler",
        (),
        {
            "__init__": FixtureSlurmScheduler.__init__,
            "call_engine": FixtureSlurmScheduler.call_engine,
            "async_call_engine": FixtureSlurmScheduler.async_call_engine,
        },
    )
    module.__dict__.update(
        SlurmScheduler=scheduler_cls,
        EngineCallError=EngineCallError,
        requests=requests,
        asyncio=asyncio,
    )
    return module


def test_patch_is_idempotent_and_preserves_signatures():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()

    assert patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    sync = scheduler_module.SlurmScheduler.call_engine
    asynchronous = scheduler_module.SlurmScheduler.async_call_engine
    assert str(patches.inspect.signature(sync)) == "(self, outcome, max_retries=3)"
    assert str(patches.inspect.signature(asynchronous)) == ("(self, outcome, max_retries=3)")
    assert not patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    assert scheduler_module.SlurmScheduler.call_engine is sync
    assert scheduler_module.SlurmScheduler.async_call_engine is asynchronous


def test_sync_definitive_engine_error_is_not_retried():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    with pytest.raises(EngineCallError, match="definitive"):
        scheduler.call_engine(lambda _attempt: EngineCallError("definitive"))
    assert scheduler.attempts == 1


def test_sync_transport_timeout_still_retries_and_recovers():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    result = scheduler.call_engine(lambda attempt: RequestTimeout() if attempt < 3 else "healthy")
    assert result == "healthy"
    assert scheduler.attempts == 3


def test_sync_service_unavailable_and_transport_error_still_retry():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    outcomes = {
        1: "service-unavailable",
        2: ConnectionError("transport"),
        3: "healthy",
    }
    assert scheduler.call_engine(outcomes.__getitem__) == "healthy"
    assert scheduler.attempts == 3


def test_async_definitive_engine_error_is_not_retried():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    with pytest.raises(EngineCallError, match="definitive"):
        asyncio.run(scheduler.async_call_engine(lambda _attempt: EngineCallError("definitive")))
    assert scheduler.attempts == 1


def test_async_request_timeout_still_retries_and_recovers():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    result = asyncio.run(scheduler.async_call_engine(lambda attempt: TimeoutError() if attempt < 3 else "healthy"))
    assert result == "healthy"
    assert scheduler.attempts == 3


def test_async_service_unavailable_and_transport_error_still_retry():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()
    patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
    scheduler = scheduler_module.SlurmScheduler()

    outcomes = {
        1: "service-unavailable",
        2: ConnectionError("transport"),
        3: "healthy",
    }
    assert asyncio.run(scheduler.async_call_engine(outcomes.__getitem__)) == "healthy"
    assert scheduler.attempts == 3


def test_source_layout_drift_fails_closed():
    patches = _load_patches_module()
    scheduler_module = _fake_scheduler_module()

    def incompatible(self):
        return None

    scheduler_module.SlurmScheduler.call_engine = incompatible
    with pytest.raises(RuntimeError, match="Unsupported AReaL engine RPC retry"):
        patches._patch_slurm_scheduler_engine_call_error_retry(scheduler_module)
