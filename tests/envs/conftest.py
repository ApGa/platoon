"""Toy Harbor tasks run with the process provider.

The process provider maps sandbox paths in commands but not inside scripts, so
task scripts here write rewards below ``${PLATOON_SANDBOX_ROOT:-}``, which only
the process provider sets.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from platoon.envs.harbor import HarborEnvironmentService
from platoon.sandbox.process import ProcessProvider

HARBOR_FIXTURES = Path(__file__).parent / "fixtures" / "harbor"
# Fixture tasks are data, never platoon's tests: a task's own tests (Harbor's hello-mcp runs pytest in its sandbox) must
# not be collected here.
collect_ignore = ["fixtures"]
VERIFIER_DIR = '"${PLATOON_SANDBOX_ROOT:-}/logs/verifier"'

MakeTask = Callable[..., Path]


@pytest.fixture
def harbor_fixtures() -> Path:
    return HARBOR_FIXTURES


@pytest.fixture
def provider(tmp_path: Path) -> ProcessProvider:
    return ProcessProvider(root=tmp_path / "sandboxes")


@pytest.fixture
def service(provider: ProcessProvider) -> HarborEnvironmentService:
    return HarborEnvironmentService(HARBOR_FIXTURES, provider=provider, splits="splits.json")


@pytest.fixture
def make_task(tmp_path: Path) -> MakeTask:
    """Write a Harbor task directory below ``tmp_path/tasks``."""

    def make(
        name: str,
        *,
        test_sh: str = f"echo 1 > {VERIFIER_DIR}/reward.txt",
        task_toml: str = "",
        instruction: str = "Do the task.",
    ) -> Path:
        task_dir = tmp_path / "tasks" / name
        (task_dir / "tests").mkdir(parents=True)
        (task_dir / "environment").mkdir()
        (task_dir / "task.toml").write_text(task_toml)
        (task_dir / "instruction.md").write_text(instruction)
        (task_dir / "tests" / "test.sh").write_text(test_sh)
        return task_dir

    return make
