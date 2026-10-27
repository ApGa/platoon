"""The training tests, which mirror the packages of ``platoon/train`` (``trainer/``, ``step/``, ``dispatch/``, ...;
the tests of a module at the top of ``platoon/train`` at the top), but for two: ``tasks/`` holds the tests of the task
samplers and admission rules, top-level modules (``sampler_api``, ``sampler_runtime``, ``samplers``, ``sampling``,
``admission_api``, ``admissions``, ``task_stats``), and the objectives (``platoon/train/objectives``) are tested with
the learner (``policy/test_learner.py``).

They share their fakes in the ``fakes`` package next to this file: pytest imports this conftest with ``tests/train``
first on the path, as it imports each conftest of a directory that is not a package, so every test below imports them
as ``fakes.<module>``, whichever directory it is in.  Every test here gets the signal handlers it started with back
(:func:`_restore_signal_handlers`), as ``train()`` installs its own.
"""

import signal

import pytest

# The signals whose handlers ``PlatoonTinkerRLTrainer.train`` installs (``platoon.train.trainer.signals``).
TRAIN_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, signal.SIGUSR2)


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    """Put back the test process's handlers of the signals ``train()`` installs handlers for, after each test."""

    saved = {sig: signal.getsignal(sig) for sig in TRAIN_SIGNALS}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
