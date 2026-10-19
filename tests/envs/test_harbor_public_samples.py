"""Published Harbor task sets parse: every ``task.toml`` below ``$PLATOON_HARBOR_SAMPLES`` (a directory of sample tasks
from public datasets, such as the Harbor task sets on the Hugging Face Hub) loads, or is refused for a reason
platoon names.  Skipped unless the variable points at such a directory.

Sample downloads often hold a task's ``task.toml`` and little else, so each task is loaded from a copy that adds an
instruction and a test script where the sample has none; the files that change how it loads (``tests/Dockerfile``
and compose files) are copied as they are.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from platoon.envs.harbor.dataset import _scan
from platoon.envs.harbor.task import COMPOSE_FILENAMES, VERIFIER_BUILD_FILES, HarborTask, HarborTaskError

SAMPLES = os.environ.get("PLATOON_HARBOR_SAMPLES")
# Refusals that published tasks may meet on purpose: platoon cannot run them as written.
NAMED_REFUSALS = ("multi-container", "[[verifier.collect]]", "compose service", "only Linux", "multi-step")

pytestmark = pytest.mark.skipif(not SAMPLES, reason="set PLATOON_HARBOR_SAMPLES to a directory of Harbor sample tasks")


def _sample_dirs() -> list[Path]:
    return sorted(_scan(Path(SAMPLES))) if SAMPLES else []


def _loadable_copy(task_dir: Path, into: Path) -> Path:
    copy = into / task_dir.name
    (copy / "tests").mkdir(parents=True)
    (copy / "environment").mkdir()
    shutil.copy2(task_dir / "task.toml", copy / "task.toml")
    for name, default in (
        ("instruction.md", "Do the task.\n"),
        ("tests/test.sh", "echo 0 > /logs/verifier/reward.txt\n"),
    ):
        source = task_dir / name
        (copy / name).write_text(source.read_text() if source.is_file() else default)
    for name in VERIFIER_BUILD_FILES:
        if (task_dir / "tests" / name).is_file():
            shutil.copy2(task_dir / "tests" / name, copy / "tests" / name)
    for name in COMPOSE_FILENAMES:
        if (task_dir / "environment" / name).is_file():
            shutil.copy2(task_dir / "environment" / name, copy / "environment" / name)
    return copy


@pytest.mark.parametrize("task_dir", _sample_dirs(), ids=lambda path: path.name)
def test_published_tasks_load_or_are_refused_by_name(task_dir: Path, tmp_path: Path):
    copy = _loadable_copy(task_dir, tmp_path)
    try:
        task = HarborTask.load(copy, agent_user="ignore")
    except HarborTaskError as exc:
        assert any(reason in str(exc) for reason in NAMED_REFUSALS), str(exc)
        return
    assert (task.verifier_mode == "separate") == (task.verifier_environment is not None)
