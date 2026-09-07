"""Regression tests for the programmatic-tool-calling cell guard.

OpenHands runs each PTC cell in-process on an ``InteractiveShellEmbed``, on a
worker thread borrowed from the event loop's default executor, with no timeout
anywhere on the path.  A Python thread cannot be killed, so a model that writes
a never-terminating cell removes one rollout worker thread from the process for
good -- silently, because the awaiting coroutine is merely abandoned and the
trajectory is recorded as interrupted.

``platoon.openreward.cell_guard`` does not fix that (only process-per-rollout
can).  It makes it *visible*, and these tests pin the two properties that makes
useful: that a cell which overstays is counted and reported, and that observing
a cell changes nothing about how it runs.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "plugins/openreward/platoon/openreward/cell_guard.py"
MODULE_NAME = "platoon_openreward_cell_guard_under_test"

PTC_PACKAGE = "openhands.tools.programmatic_tool_calling"

# Small enough to keep the suite fast, large enough that a loaded CI box cannot
# cross them by accident between two statements.
WARN_S = 0.25
LEAK_S = 0.60

ENV_KNOBS = (
    "OPENREWARD_PTC_CELL_GUARD",
    "OPENREWARD_PTC_CELL_WARN_SECONDS",
    "OPENREWARD_PTC_CELL_LEAK_SECONDS",
    "OPENREWARD_PTC_CELL_GUARD_REPORT_SECONDS",
    "OPENREWARD_PTC_CELL_SOURCE_CHARS",
)


def _load_guard():
    """Import the module standalone, with a registry private to one test.

    It must not need the ``platoon.openreward`` package (nor OpenHands) to
    import, and each test needs process-wide counters that start at zero.
    """

    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``dataclass`` resolves annotations through ``sys.modules[cls.__module__]``.
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(MODULE_NAME, None)
    return module


@pytest.fixture
def guard(monkeypatch):
    """A fresh guard with every knob at a documented default."""

    for name in ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    return _load_guard()


@pytest.fixture
def fast_guard(guard, monkeypatch):
    """A fresh guard whose thresholds are sub-second."""

    monkeypatch.setenv("OPENREWARD_PTC_CELL_WARN_SECONDS", str(WARN_S))
    monkeypatch.setenv("OPENREWARD_PTC_CELL_LEAK_SECONDS", str(LEAK_S))
    monkeypatch.setenv("OPENREWARD_PTC_CELL_GUARD_REPORT_SECONDS", "0")
    return guard


class _Action:
    """The bit of ``ProgrammaticToolCallingAction`` the guard reads."""

    def __init__(self, code: str, gate: threading.Event | None = None, boom: BaseException | None = None):
        self.code = code
        self.gate = gate
        self.boom = boom


class _FakeExecutor:
    """Stands in for ``ProgrammaticToolCallingExecutor`` with the same call shape."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def __call__(self, action, conversation=None):
        self.calls.append((action, conversation))
        if action.boom is not None:
            raise action.boom
        if action.gate is not None:
            assert action.gate.wait(timeout=30), "test cell was never released"
        return f"ran:{action.code}"


def _install_fake_ptc(monkeypatch) -> type[_FakeExecutor]:
    """Expose a fake PTC module so the guard patches it instead of the real one.

    ``from pkg import impl`` consults ``sys.modules[pkg]`` first, so seeding the
    package name is enough and the real OpenHands is never imported.
    """

    impl = types.ModuleType(f"{PTC_PACKAGE}.impl")
    executor_cls = type("ProgrammaticToolCallingExecutor", (_FakeExecutor,), {})
    impl.ProgrammaticToolCallingExecutor = executor_cls
    package = types.ModuleType(PTC_PACKAGE)
    package.impl = impl
    monkeypatch.setitem(sys.modules, PTC_PACKAGE, package)
    monkeypatch.setitem(sys.modules, f"{PTC_PACKAGE}.impl", impl)
    return executor_cls


def _run_in_thread(executor, action) -> threading.Thread:
    thread = threading.Thread(target=lambda: executor(action), daemon=True)
    thread.start()
    return thread


def _await_count(guard_module, key: str, expected: int, timeout: float = 10.0) -> int:
    """Poll until ``key`` reaches ``expected``; the guard's own clock is real time."""

    deadline = threading.Event()
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        counts = guard_module.registry().poll()
        if counts[key] == expected:
            return counts[key]
        deadline.wait(0.02)
    return guard_module.registry().poll()[key]


# -- knobs and the kill switch -------------------------------------------


def test_defaults_match_the_documented_values(guard):
    assert guard.enabled() is True
    assert guard.warn_seconds() == 300.0
    assert guard.leak_seconds() == 1800.0
    assert guard.report_seconds() == 60.0


def test_leak_threshold_is_never_below_the_warn_threshold(guard, monkeypatch):
    """A cell is always reported stuck before it is written off as leaked."""

    monkeypatch.setenv("OPENREWARD_PTC_CELL_WARN_SECONDS", "900")
    monkeypatch.setenv("OPENREWARD_PTC_CELL_LEAK_SECONDS", "60")
    assert guard.leak_seconds() == 900.0


def test_unparseable_knob_falls_back_to_the_default(guard, monkeypatch):
    monkeypatch.setenv("OPENREWARD_PTC_CELL_WARN_SECONDS", "not-a-number")
    assert guard.warn_seconds() == 300.0


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_kill_switch_disables_everything(guard, monkeypatch, value):
    executor_cls = _install_fake_ptc(monkeypatch)
    original = executor_cls.__call__
    monkeypatch.setenv("OPENREWARD_PTC_CELL_GUARD", value)

    assert guard.enabled() is False
    assert guard.install() == {"cell_guard": False, "monitor": False}
    assert executor_cls.__call__ is original

    executor = executor_cls()
    assert executor(_Action("print(1)")) == "ran:print(1)"
    assert guard.snapshot()["ptc_cells_started"] == 0


# -- installation --------------------------------------------------------


def test_install_is_idempotent(guard, monkeypatch):
    executor_cls = _install_fake_ptc(monkeypatch)
    original = executor_cls.__call__

    assert guard.install_programmatic_tool_calling_cell_guard() is True
    patched = executor_cls.__call__
    assert patched is not original

    assert guard.install_programmatic_tool_calling_cell_guard() is False
    assert executor_cls.__call__ is patched

    # ``install()`` wraps the same one-shot patch plus a single monitor thread.
    assert guard.install() == {"cell_guard": False, "monitor": True}
    assert guard.install() == {"cell_guard": False, "monitor": False}
    assert executor_cls.__call__ is patched


def test_install_is_a_no_op_without_openhands(guard, monkeypatch):
    monkeypatch.setitem(sys.modules, PTC_PACKAGE, None)
    assert guard.install_programmatic_tool_calling_cell_guard() is False


def test_real_openhands_executor_matches_the_wrapped_call_shape():
    """Pin the upstream contract the wrapper relies on: ``__call__(self, action, ...)``."""

    impl = pytest.importorskip(f"{PTC_PACKAGE}.impl")
    import inspect

    parameters = list(inspect.signature(impl.ProgrammaticToolCallingExecutor.__call__).parameters)
    assert parameters[:2] == ["self", "action"]
    action_module = pytest.importorskip(f"{PTC_PACKAGE}.definition")
    assert "code" in action_module.ProgrammaticToolCallingAction.model_fields


def test_the_wrapper_works_on_a_real_openhands_executor(guard):
    """Assigning ``__call__`` on the real class must still produce a working executor.

    The signature test above cannot catch a metaclass/pydantic surprise in the
    class this actually patches, and that is the failure that would take down a
    production run at import time.
    """

    impl = pytest.importorskip(f"{PTC_PACKAGE}.impl")
    definition = pytest.importorskip(f"{PTC_PACKAGE}.definition")

    original = impl.ProgrammaticToolCallingExecutor.__call__
    already_guarded = getattr(original, "_platoon_cell_guard", False)
    assert guard.install_programmatic_tool_calling_cell_guard() is not already_guarded
    try:
        executor = impl.ProgrammaticToolCallingExecutor(tool_name="ptc")
        try:
            # ``conversation=None`` returns an observation without running a
            # cell, which keeps this test off the IPython execution path.
            observation = executor(definition.ProgrammaticToolCallingAction(code="print(6 * 7)"), None)
            assert isinstance(observation, definition.ProgrammaticToolCallingObservation)
        finally:
            executor._tool_thread_pool.shutdown(wait=False)
            executor._loop.close()
    finally:
        if not already_guarded:
            impl.ProgrammaticToolCallingExecutor.__call__ = original

    counts = guard.snapshot()
    assert counts["ptc_cells_started"] == 1
    assert counts["ptc_cells_finished"] == 1
    assert counts["ptc_cells_running"] == 0


# -- the guard never changes what a cell does ----------------------------


def test_guard_preserves_return_value_and_arguments(guard, monkeypatch):
    executor_cls = _install_fake_ptc(monkeypatch)
    guard.install_programmatic_tool_calling_cell_guard()

    executor = executor_cls()
    action = _Action("1 + 1")
    sentinel = object()

    assert executor(action, sentinel) == "ran:1 + 1"
    assert executor.calls == [(action, sentinel)]
    counts = guard.snapshot()
    assert counts["ptc_cells_started"] == 1
    assert counts["ptc_cells_finished"] == 1
    assert counts["ptc_cells_running"] == 0
    assert counts["ptc_cells_stuck"] == 0


def test_guard_preserves_exception_propagation_and_deregisters(guard, monkeypatch):
    executor_cls = _install_fake_ptc(monkeypatch)
    guard.install_programmatic_tool_calling_cell_guard()

    boom = ValueError("cell blew up")
    with pytest.raises(ValueError) as excinfo:
        executor_cls()(_Action("raise ValueError", boom=boom))
    assert excinfo.value is boom

    counts = guard.snapshot()
    assert counts["ptc_cells_finished"] == 1
    assert counts["ptc_cells_running"] == 0


def test_a_failure_inside_the_guard_still_runs_the_cell(guard, monkeypatch, caplog):
    """Bookkeeping must never be able to stop a cell from running."""

    executor_cls = _install_fake_ptc(monkeypatch)
    guard.install_programmatic_tool_calling_cell_guard()

    def _broken_begin(*args, **kwargs):
        raise RuntimeError("guard is broken")

    monkeypatch.setattr(guard.registry(), "begin", _broken_begin)

    with caplog.at_level(logging.ERROR, logger=MODULE_NAME):
        assert executor_cls()(_Action("still_runs()")) == "ran:still_runs()"

    assert any("running it unobserved" in record.getMessage() for record in caplog.records)


# -- detection -----------------------------------------------------------


def test_a_normal_cell_is_never_reported_stuck(fast_guard, monkeypatch, caplog):
    executor_cls = _install_fake_ptc(monkeypatch)
    fast_guard.install_programmatic_tool_calling_cell_guard()

    with caplog.at_level(logging.WARNING, logger=MODULE_NAME):
        assert executor_cls()(_Action("2 + 2")) == "ran:2 + 2"
        counts = fast_guard.registry().poll()

    assert counts["ptc_cells_stuck"] == 0
    assert counts["ptc_threads_leaked"] == 0
    assert caplog.records == []


def test_a_slow_cell_is_reported_stuck_and_counted(fast_guard, monkeypatch, caplog):
    executor_cls = _install_fake_ptc(monkeypatch)
    fast_guard.install_programmatic_tool_calling_cell_guard()

    gate = threading.Event()
    executor = executor_cls()
    with caplog.at_level(logging.WARNING, logger=MODULE_NAME):
        thread = _run_in_thread(executor, _Action("while True: spin()  # runaway", gate=gate))
        try:
            assert _await_count(fast_guard, "ptc_cells_stuck", 1) == 1
        finally:
            gate.set()
            thread.join(timeout=30)

    stuck = [record for record in caplog.records if "still running after" in record.message]
    assert len(stuck) == 1, "exactly one WARNING per stuck cell"
    assert stuck[0].levelno == logging.WARNING
    message = stuck[0].getMessage()
    assert "cell #1" in message
    assert "while True: spin()" in message, "the operator must see what the model wrote"
    assert "cannot be interrupted" in message


def test_the_stuck_count_decrements_when_a_slow_cell_finishes(fast_guard, monkeypatch):
    executor_cls = _install_fake_ptc(monkeypatch)
    fast_guard.install_programmatic_tool_calling_cell_guard()

    gate = threading.Event()
    thread = _run_in_thread(executor_cls(), _Action("time.sleep(600)", gate=gate))
    try:
        assert _await_count(fast_guard, "ptc_cells_stuck", 1) == 1
    finally:
        gate.set()
        thread.join(timeout=30)

    counts = fast_guard.registry().poll()
    assert counts["ptc_cells_stuck"] == 0
    assert counts["ptc_threads_leaked"] == 0
    assert counts["ptc_cells_running"] == 0
    assert counts["ptc_cells_finished"] == 1


def test_leaked_threads_accumulate(fast_guard, monkeypatch, caplog):
    """Each cell past the leak threshold is written off once, and they add up."""

    executor_cls = _install_fake_ptc(monkeypatch)
    fast_guard.install_programmatic_tool_calling_cell_guard()

    gate = threading.Event()
    executor = executor_cls()
    threads = [_run_in_thread(executor, _Action(f"runaway_{index}()", gate=gate)) for index in range(3)]
    try:
        with caplog.at_level(logging.ERROR, logger=MODULE_NAME):
            assert _await_count(fast_guard, "ptc_threads_leaked", 3) == 3
            # Re-polling must not re-report or double count.
            counts = fast_guard.registry().poll()
    finally:
        gate.set()
        for thread in threads:
            thread.join(timeout=30)

    assert counts["ptc_threads_leaked"] == 3
    assert counts["ptc_cells_stuck"] == 3
    leaks = [record for record in caplog.records if "PRESUMED LEAKED" in record.message]
    assert len(leaks) == 3, "one ERROR per leaked thread, once"
    assert all(record.levelno == logging.ERROR for record in leaks)
    assert "permanently lost" in leaks[0].getMessage()


def test_the_guard_never_terminates_a_cell(fast_guard, monkeypatch):
    """The whole point: reporting a runaway does not stop, cancel or truncate it."""

    executor_cls = _install_fake_ptc(monkeypatch)
    fast_guard.install_programmatic_tool_calling_cell_guard()

    gate = threading.Event()
    results: list[object] = []
    executor = executor_cls()
    action = _Action("long_but_legitimate()", gate=gate)
    thread = threading.Thread(target=lambda: results.append(executor(action)), daemon=True)
    thread.start()
    try:
        assert _await_count(fast_guard, "ptc_threads_leaked", 1) == 1
        assert thread.is_alive(), "the guard must not have touched the cell's thread"
    finally:
        gate.set()
        thread.join(timeout=30)

    assert results == ["ran:long_but_legitimate()"], "the cell still returns its own value"


# -- what the trainer logs ------------------------------------------------


def test_snapshot_exposes_the_two_named_scalars(guard):
    counts = guard.snapshot()
    assert counts["ptc_cells_stuck"] == 0
    assert counts["ptc_threads_leaked"] == 0
    assert all(isinstance(value, int) for value in counts.values())


def test_source_prefix_is_one_bounded_line(guard, monkeypatch):
    monkeypatch.setenv("OPENREWARD_PTC_CELL_SOURCE_CHARS", "20")
    prefix = guard._source_prefix("import os\nwhile True:\n    os.getpid()  # " + "x" * 200)
    assert "\n" not in prefix
    assert "+" in prefix and "chars" in prefix
    assert guard._source_prefix("   ") == "<empty>"
