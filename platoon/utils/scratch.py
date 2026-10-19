"""Node-local scratch: the directory for files that belong to one node, such as staged copies of harness toolchains,
per-node markers and image build trees, and for platoon's temporary files.

A node's ``/tmp`` may be small, or shared with other jobs, so batch systems give a job a node-local directory of its
own and name it in the environment.  :func:`node_scratch` takes the first of ``$PLATOON_NODE_SCRATCH``,
``$PBS_LOCALDIR``, ``$SLURM_TMPDIR`` and ``$TMPDIR`` that is set and not empty, else ``/tmp``, and platoon makes its
temporary files and directories there (:func:`mkdtemp`, :func:`temporary_directory`, :func:`temporary_file`,
:func:`named_temporary_file`) rather than where :mod:`tempfile` would, ``$TMPDIR`` alone, which a job script would
otherwise have to move off a small ``/tmp`` by hand (RFC 0018, section 4).

Stdlib only, so any module can import it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import IO, Any

__all__ = [
    "NODE_SCRATCH_ENV",
    "NODE_SCRATCH_VARIABLES",
    "mkdtemp",
    "named_temporary_file",
    "node_scratch",
    "scratch_dir",
    "temporary_directory",
    "temporary_file",
]

NODE_SCRATCH_ENV = "PLATOON_NODE_SCRATCH"
# The variables that name node-local scratch, in the order node_scratch() reads them.
NODE_SCRATCH_VARIABLES = (NODE_SCRATCH_ENV, "PBS_LOCALDIR", "SLURM_TMPDIR", "TMPDIR")
DEFAULT_NODE_SCRATCH = "/tmp"


def node_scratch() -> Path:
    """Node-local scratch: ``$PLATOON_NODE_SCRATCH``, else ``$PBS_LOCALDIR``, ``$SLURM_TMPDIR`` or ``$TMPDIR``, else
    ``/tmp``.  A variable set to the empty string counts as unset."""

    for variable in NODE_SCRATCH_VARIABLES:
        if value := os.environ.get(variable):
            return Path(value)
    return Path(DEFAULT_NODE_SCRATCH)


def scratch_dir() -> Path:
    """:func:`node_scratch`, created when it does not exist yet: where platoon's temporary files go."""

    path = node_scratch()
    path.mkdir(parents=True, exist_ok=True)
    return path


def mkdtemp(prefix: str) -> str:
    """A new private directory in node-local scratch (:func:`tempfile.mkdtemp`); the caller removes it."""

    return tempfile.mkdtemp(prefix=prefix, dir=scratch_dir())


def temporary_directory(prefix: str = "platoon-") -> tempfile.TemporaryDirectory[str]:
    """A directory in node-local scratch that is removed on exit (:class:`tempfile.TemporaryDirectory`)."""

    return tempfile.TemporaryDirectory(prefix=prefix, dir=scratch_dir())


def temporary_file(**options: Any) -> IO[Any]:
    """An unnamed file in node-local scratch (:func:`tempfile.TemporaryFile`), with its options."""

    return tempfile.TemporaryFile(dir=scratch_dir(), **options)


def named_temporary_file(*args: Any, **options: Any) -> Any:
    """A named file in node-local scratch (:func:`tempfile.NamedTemporaryFile`), with its arguments."""

    return tempfile.NamedTemporaryFile(*args, dir=scratch_dir(), **options)
