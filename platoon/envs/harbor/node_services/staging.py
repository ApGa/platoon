"""Staged binds: a host directory that a node service binds read-only, copied once per node into node-local scratch
first, so that its service reads it from local disk rather than from a shared filesystem (a judge's test data, say).

:func:`stage` copies a bind's source to ``staged/<digest of its path>.versions/<digest of its files>`` below the node's
node-service directory (:func:`~platoon.envs.harbor.node_services.spec.staging_root`), under a lock of that source's
own, so that two services, or two runs, that stage one source on a node make one copy.  A copy is a **version** of its
source, named by its files, sizes and modification times, kept for the next start while its source has not changed.  A
source that changed gets a new version beside the old one, which a service that still runs keeps reading: each user of a
version holds it (a shared lock on ``<version>.lock``) until it releases it (:meth:`StagedCopy.release`, or its process
ends), and a version no one holds that is not the current one is removed as the source is staged again.  A copy cut
short leaves a ``.partial`` directory, which the next stage removes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from platoon.envs.harbor.node_services.spec import staging_root
from platoon.utils.scratch import private_dir


@dataclass
class StagedCopy:
    """A version of a staged source (``path``), held until :meth:`release`."""

    path: Path
    _hold: IO[Any] | None = field(default=None, repr=False)

    def release(self) -> None:
        """Stop holding the copy: a stage of a newer version may remove it; idempotent."""

        hold, self._hold = self._hold, None
        if hold is not None:
            hold.close()


def stage(source: str | Path) -> StagedCopy:
    """The current version of the directory (or file) ``source`` in node-local scratch, made now unless it is there
    (the module docstring says how), held until released; ``OSError`` when it cannot be read or copied."""

    source = Path(source).resolve(strict=True)
    private_dir(staging_root().parent)
    root = private_dir(staging_root())
    digest = hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    versions = private_dir(root / f"{digest}.versions")
    with open(root / f"{digest}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fingerprint = _fingerprint(source)
        version = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:16]
        target = versions / version  # named by its files: a copy there is complete (it is renamed there once made)
        if not os.path.lexists(target):
            partial = versions / f"{version}.partial"
            _remove(partial)
            if source.is_dir():
                shutil.copytree(source, partial, symlinks=True)
            else:
                shutil.copy2(source, partial)
            partial.rename(target)
        hold = open(versions / f"{version}.lock", "a")  # noqa: SIM115 - held until the copy is released
        fcntl.flock(hold, fcntl.LOCK_SH)
        _remove_unheld(versions, keep=version)
    return StagedCopy(path=target, _hold=hold)


def _remove_unheld(versions: Path, *, keep: str) -> None:
    """Remove the versions in ``versions`` other than ``keep`` that no one holds, and copies cut short."""

    for path in versions.iterdir():
        name = path.name
        if name.endswith(".partial"):  # a copy cut short: no other stage of the source runs while this one does
            _remove(path)
            continue
        if name == keep or name.endswith(".lock"):
            continue
        with open(versions / f"{name}.lock", "a") as hold:
            try:
                fcntl.flock(hold, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a service still reads it
            _remove(path)
            (versions / f"{name}.lock").unlink(missing_ok=True)


def _fingerprint(source: Path) -> dict[str, list[int]]:
    """Each file below ``source`` (or ``source`` itself), with its size and modification time."""

    if not source.is_dir():
        info = source.stat()
        return {".": [info.st_size, info.st_mtime_ns]}
    found: dict[str, list[int]] = {}
    for directory, _, files in os.walk(source):
        for name in files:
            path = Path(directory) / name
            info = path.lstat()
            found[str(path.relative_to(source))] = [info.st_size, info.st_mtime_ns]
    return found


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif os.path.lexists(path):
        path.unlink()


__all__ = ["StagedCopy", "stage"]
