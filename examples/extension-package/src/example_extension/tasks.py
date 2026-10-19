"""The arithmetic tasks: a task table that ships inside the package.

A config builds it with ``service: example_extension.tasks:arithmetic``. The rows are read from the installed
package, so the config needs no path, and every process that runs rollouts finds the same file.
"""

from __future__ import annotations

from importlib.resources import files

from platoon.envs import TaskTableService


def arithmetic() -> TaskTableService:
    """The arithmetic questions: splits ``train`` and ``test``, each row with its ``answer``."""

    return TaskTableService(path=str(files(__package__) / "data" / "arithmetic.jsonl"), name="example-arithmetic")
