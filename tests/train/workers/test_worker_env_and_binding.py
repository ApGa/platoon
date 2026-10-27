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

    HF_HUB_OFFLINE=1 python -m pytest tests/train/workers -q
"""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

from platoon.train.trainer.rl import PlatoonTinkerRLTrainer  # noqa: E402
from platoon.train.workers import env as worker_env  # noqa: E402


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
    # And every other variable the invariants set, away from its invariant.  Through monkeypatch, which puts each
    # back after the test: this process is no rollout worker, and the worker marker left set would make the
    # in-process rollouts of every later test in it enter a worker's per-process Python overlay.
    monkeypatch.setenv(worker_env.WORKER_ENV_MARKER, "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    for variable in worker_env.THREAD_PIN_VARIABLES:
        monkeypatch.setenv(variable, "8")
    for variable in worker_env.NO_CUDA_DRIVER:  # the GPUs a trainer sees, and torch asking the driver
        monkeypatch.setenv(variable, "0")

    assert worker_env.apply_rollout_worker_env_invariants(force=True) is True
    assert worker_env.STARTUP_BUDGET_ENV not in os.environ
    # re-applying is idempotent and keeps every invariant, deadline or not
    assert worker_env.apply_rollout_worker_env_invariants(force=True) is True
    assert os.environ["TINKER_TELEMETRY"] == "0"
    assert "TINKER_SUBPROCESS_SAMPLING" not in os.environ
    assert all(os.environ[variable] == "1" for variable in worker_env.THREAD_PIN_VARIABLES)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert all(os.environ[variable] == value for variable, value in worker_env.NO_CUDA_DRIVER.items())
    assert os.environ[worker_env.WORKER_ENV_MARKER] == "1"


def test_a_worker_sees_no_gpu_and_asks_nvml(monkeypatch):
    """Set before transformers asks torch at import, which would otherwise start the CUDA driver wherever one is
    installed, in every exec'd worker and in the fork server (which then could not fork)."""

    for name in (*worker_env.THREAD_PIN_VARIABLES, "TINKER_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, os.environ.get(name, ""))  # so that this test's process gets them back
    monkeypatch.setenv(worker_env.WORKER_ENV_MARKER, os.environ.get(worker_env.WORKER_ENV_MARKER, ""))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    assert worker_env.apply_rollout_worker_env_invariants(force=True) is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


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
    require) ``platoon-rollout-ipc-<uid>`` in node scratch, a path the mode never
    uses, so an unwritable or missing scratch would fail every rollout in the one
    mode that is supposed to be untouched by any of this.
    """

    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.workflow.group import GroupRolloutWorkflow

    def _explode() -> str:
        raise AssertionError("inprocess mode resolved the rollout worker scratch directory")

    workflow = GroupRolloutWorkflow(
        rollout_fn=None,
        get_task_fn=None,
        config=WorkflowConfig(rollout_isolation="inprocess"),
        model_info=SimpleNamespace(llm=None, model_name="m"),
    )
    monkeypatch.setattr(workflow, "_ipc_dir", _explode, raising=False)

    binding = workflow._model_binding(7)
    assert binding.binding_path == ""
    assert binding.checkpoint_version == 7
