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


_CREDIT_CONFIG_CHECK = """
import sys
import tempfile
from pathlib import Path

import yaml

from platoon.train import entry_settings
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import resolve_config

directory = Path(tempfile.mkdtemp())
single = {"group_size": 1}
entries = [
    {"name": "keyed", "credit": "keyed", "credit_options": {"key": ["task"]}, "workflow_config": single},
    {"name": "running", "credit": "running", "credit_options": {"strata": ["kind"]}, "workflow_config": single},
]
path = directory / "train.yaml"
path.write_text(
    yaml.safe_dump(
        {
            "train": {"model_name": "m", "renderer_name": "r", "max_staleness": 2},
            "eval": {"every": 0},
            "log_path": str(directory / "logs"),
            "environments": [{"dataset_loader": "x:y", "rollout": "x:y", **entry} for entry in entries],
        }
    )
)
config = resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config
warned = [line.split(".")[0] for line in entry_settings.warnings(config, entry_settings.entry_settings(config))]
estimators = {f"platoon.train.credit.{name}" for name in ("group", "keyed", "running", "reverse_kl")}
heavy = sorted(name for name in sys.modules if name.split(".")[0] in {"tinker", "torch"} or name in estimators)
print(warned, heavy)
sys.exit(0 if warned == ["environments[keyed]"] and not heavy else 1)
"""


def test_loading_a_config_that_names_core_s_credit_estimators_imports_none_of_them():
    """A training config whose entries name core's credit estimators, with options, loads and gives its entries'
    warnings (an entry in groups of one, unless its estimator credits them) without importing any estimator, nor
    Tinker or torch, which they import: ``platoon config`` runs without the ``tinker`` extra."""

    completed = subprocess.run(
        [sys.executable, "-c", _CREDIT_CONFIG_CHECK],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
