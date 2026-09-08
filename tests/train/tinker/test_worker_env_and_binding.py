"""Two things a rollout worker must not get wrong before it can run anything.

1. The pre-import startup deadline is armed from the environment, and that
   environment is inherited by the worker's whole descendant tree.  Its handler
   ``killpg``s the group, so a descendant that armed one would take the entire
   rollout tree down mid-rollout.  The variable is therefore consumed by the
   arming process.
2. With ``rollout_isolation: process`` every worker builds its sampling client
   by reading the published binding file.  If the FIRST publication fails and is
   swallowed, every rollout of the run dies with ``FileNotFoundError`` and the
   trainer produces no training data for as long as the job runs.

    HF_HUB_OFFLINE=1 plugins/openreward/.venv/bin/python -m pytest tests/train/tinker -q
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from platoon.train.tinker import worker_env  # noqa: E402
from platoon.train.tinker.rl import PlatoonTinkerRLTrainer  # noqa: E402


@pytest.fixture
def restore_alarm():
    previous = signal.getsignal(signal.SIGALRM)
    yield
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous)


def test_startup_budget_is_consumed_so_descendants_cannot_arm(monkeypatch, tmp_path, restore_alarm):
    monkeypatch.setenv(worker_env.STARTUP_BUDGET_ENV, "600")
    monkeypatch.setenv(worker_env.DIAG_PATH_ENV, str(tmp_path / "worker.diag"))

    assert worker_env.arm_startup_alarm_from_env() == 600.0
    # this process is on a deadline ...
    assert signal.getitimer(signal.ITIMER_REAL)[0] > 0
    # ... and no child of it will be, because the budget is gone from the
    # environment every descendant inherits.
    assert worker_env.STARTUP_BUDGET_ENV not in os.environ
    assert worker_env.DIAG_PATH_ENV in os.environ  # the handler still needs it

    # A descendant importing platoon.train.tinker runs exactly this code.
    assert worker_env.arm_startup_alarm_from_env() == 0.0
    # and did not disarm the worker that legitimately owns the deadline
    assert signal.getitimer(signal.ITIMER_REAL)[0] > 0


def test_worker_invariants_still_apply_after_the_budget_is_consumed(monkeypatch, tmp_path, restore_alarm):
    monkeypatch.setenv(worker_env.STARTUP_BUDGET_ENV, "600")
    monkeypatch.setenv(worker_env.DIAG_PATH_ENV, str(tmp_path / "worker.diag"))
    monkeypatch.setenv("TINKER_TELEMETRY", "1")
    monkeypatch.setenv("TINKER_SUBPROCESS_SAMPLING", "1")

    assert worker_env.apply_rollout_worker_env_invariants(force=True) is True
    assert worker_env.STARTUP_BUDGET_ENV not in os.environ
    # re-applying is idempotent and keeps every invariant, deadline or not
    assert worker_env.apply_rollout_worker_env_invariants(force=True) is True
    assert os.environ["TINKER_TELEMETRY"] == "0"
    assert "TINKER_SUBPROCESS_SAMPLING" not in os.environ
    assert os.environ["OMP_NUM_THREADS"] == "1"


class _Workflow:
    def __init__(self, isolation: str, *, fail: bool) -> None:
        self.rollout_isolation = isolation
        self._fail = fail
        self.published: list[tuple[object, int]] = []

    def publish_model_binding(self, sampling_client, version: int) -> None:
        if self._fail:
            raise OSError("no space left on device")
        self.published.append((sampling_client, version))


def test_initial_binding_failure_is_fatal_in_process_mode():
    workflow = _Workflow("process", fail=True)
    with pytest.raises(OSError):
        PlatoonTinkerRLTrainer._publish_worker_binding(workflow, object(), 3, required=True)


def test_republication_failure_is_survivable():
    """A worker keeps the client it holds; under full finetune that is the newest."""

    workflow = _Workflow("process", fail=True)
    PlatoonTinkerRLTrainer._publish_worker_binding(workflow, object(), 4)


def test_inprocess_mode_never_fails_on_a_binding():
    workflow = _Workflow("inprocess", fail=True)
    PlatoonTinkerRLTrainer._publish_worker_binding(workflow, object(), 5, required=True)


def test_a_workflow_without_workers_is_a_no_op():
    PlatoonTinkerRLTrainer._publish_worker_binding(SimpleNamespace(), object(), 6, required=True)


def test_a_successful_publish_reaches_the_workflow():
    workflow = _Workflow("process", fail=False)
    client = object()
    PlatoonTinkerRLTrainer._publish_worker_binding(workflow, client, 7, required=True)
    assert workflow.published == [(client, 7)]


def test_inprocess_mode_never_resolves_node_local_scratch(monkeypatch):
    """The fallback must not depend on the isolation design's scratch directory.

    ``inprocess`` is what a run rolls BACK to, and it writes no request, no
    response and no binding.  Resolving ``ipc_dir()`` for it would create (and
    require) ``$TMPDIR/platoon-rollout-ipc`` on a path the mode never uses, so an
    unwritable or missing TMPDIR would fail every rollout in the one mode that is
    supposed to be untouched by any of this.
    """

    from platoon.train.tinker.workflows import group_rollout_workflow as module

    def _explode() -> str:
        raise AssertionError("inprocess mode resolved the rollout worker scratch directory")

    workflow = object.__new__(module.GroupRolloutWorkflow)
    workflow.rollout_isolation = "inprocess"
    workflow.config = SimpleNamespace(
        rollout_config=SimpleNamespace(max_steps=None, timeout=None, output_dir="out"),
        depth_level_weighting=False,
        worker_binding_poll_seconds=5.0,
        worker_startup_budget_seconds=300.0,
        worker_rss_soft_limit_bytes=1,
        worker_python_env_isolation=False,
        worker_unbounded_rollout_budget_seconds=1.0,
    )
    monkeypatch.setattr(workflow, "_ipc_dir", _explode, raising=False)

    binding = module.GroupRolloutWorkflow._model_binding(
        SimpleNamespace(
            _isolation="inprocess",
            model_info=SimpleNamespace(llm=None, model_name="platoon-tinker/m"),
            config=workflow.config,
            _ipc_dir=_explode,
        ),
        7,
    )
    assert binding.binding_path == ""
    assert binding.checkpoint_version == 7
