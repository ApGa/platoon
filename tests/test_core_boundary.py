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
import platoon.train.tinker.rl
import platoon.train.tinker.train
import platoon.train.tinker.workers.entrypoint

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
