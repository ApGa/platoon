"""Core must never import the extensions that plug into it."""

from __future__ import annotations

import subprocess
import sys

_LEAK_CHECK = """
import sys

import platoon
import platoon.cli
import platoon.hooks
import platoon.eval.run
import platoon.train.trainer.rl
import platoon.train.launch
import platoon.train.workers.entrypoint

leaked = sorted(
    name
    for name in sys.modules
    if name.split(".")[0] in {"areal", "openhands"}
    or name.startswith(("platoon.openhands", "platoon.openreward"))
)
print(leaked)
sys.exit(1 if leaked else 0)
"""


def test_importing_core_loads_no_extension_module():
    completed = subprocess.run(
        [sys.executable, "-c", _LEAK_CHECK],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_EXPERIMENTAL_CHECK = """
import sys

import platoon.experimental

heavy = sorted(name for name in ("tinker", "datasets", "torch") if name in sys.modules)
print(heavy)
sys.exit(1 if heavy else 0)
"""


def test_importing_the_experimental_api_stays_cheap():
    """``platoon.experimental`` (the task sampler API and the statistics' types) loads neither tinker nor datasets: a
    package that registers a sampler imports it in every process that imports the package."""

    completed = subprocess.run(
        [sys.executable, "-c", _EXPERIMENTAL_CHECK],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_TRAIN_CHECK = """
import sys

import platoon.train
import platoon.train.config_defs

heavy = sorted(name for name in ("tinker", "torch") if name in sys.modules)
print(heavy)
sys.exit(1 if heavy else 0)
"""


def test_importing_the_training_package_and_its_config_stays_cheap():
    """``platoon.train`` and ``platoon.train.config_defs`` (the training config's schema, which ``platoon config``
    reads) load neither tinker nor torch: ``platoon.train`` exports its names lazily."""

    completed = subprocess.run(
        [sys.executable, "-c", _TRAIN_CHECK],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


_OLD_PACKAGE_CHECK = """
import sys

import platoon.train.tinker

trainer = sorted(
    name for name in ("platoon.train.trainer.rl", "platoon.train.workflow.group") if name in sys.modules
)
print(trainer)
sys.exit(1 if trainer else 0)
"""


def test_importing_the_old_tinker_package_loads_no_trainer():
    """``platoon.train.tinker`` no longer imports the trainer: its old names resolve lazily
    (``tests/test_moved_modules.py``), so a module of the policy client (``platoon.train.tinker.proxy``) does not pay
    for the trainer."""

    completed = subprocess.run(
        [sys.executable, "-c", _OLD_PACKAGE_CHECK],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
