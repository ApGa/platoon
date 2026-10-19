"""Artifacts: what a separate verifier receives from the agents' sandbox, as Harbor 0.24 hands them over
(``trial/artifact_handler.py``), as regular files only.

Each path a task declares in ``artifacts``, and the directory every agent may publish files in
(``/logs/artifacts``), is copied from ``main`` to the host when grading starts, and put back at the same path in
the verifier's sandbox.  A path that does not exist is skipped: the tests decide what its absence means.  A path
that exists but cannot be copied is an infrastructure failure (:class:`ArtifactError`), never a silent skip.

Only directories and regular files reach the verifier, never a symlink, which would read whatever it points to:
a host file wherever the host reads the copy, the verifier's own files (its tests, its reward files) in the
verifier's sandbox.  Grading kills every process the agents left in ``main`` first, so nothing changes under the
collection.  Each artifact is packed there with ``tar`` and unpacked on the host by :func:`_unpack`, which writes
the archive's directories and regular files and leaves out everything else: symlinks, devices, FIFOs, and hard
links to anything but a regular file the archive holds (those become copies).  A declared path that is itself a
symlink or a special file is left out the same way, so the verifier sees it missing, and so is one whose archive
would be larger than ``max_bytes`` (:data:`DEFAULT_MAX_ARTIFACT_BYTES`), measured in ``main`` before anything is
written.  What was left out is reported (``skipped``).  An artifact the agents took the owner's permissions from
(chmod 000) gets them back before it is packed, so that it reaches the verifier rather than fail the collection.

Each grade downloads into a directory of its own that holds nothing else (``into``, new or empty), since a download
can reach whatever its destination held before (hpc-sandbox keeps the symlinks that stay inside the copy).  The
archive is one regular file, so a download that brings back anything else, or reports links it left out
(:class:`~platoon.sandbox.base.DownloadReport`), means the sandbox replaced it: an :class:`ArtifactError`.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import shutil
import tarfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from platoon.envs.harbor.task import ARTIFACTS_DIR, Artifact
from platoon.envs.harbor.users import DOCKER_UMASK
from platoon.sandbox.base import DownloadReport, Sandbox

_TRANSFER_TIMEOUT_S = 600.0
DEFAULT_MAX_ARTIFACT_BYTES = 1 << 30  # an artifact whose archive is larger is left out
# sh -c _PACK_SCRIPT platoon-artifact PATH ARCHIVE MAX_BYTES [--exclude=PATTERN ...]: prints the kind of PATH, never
# following a symlink: "missing", "symlink" or "special" (left as they are), "too_large" (an archive of it would be
# larger than MAX_BYTES, measured from a stream it stops reading at the cap), or "dir" or "file" once ARCHIVE holds it
# (a directory's contents, a file by its name).
_PACK_SCRIPT = r"""
path=$1; archive=$2; max=$3; shift 3
if [ -L "$path" ]; then echo symlink; exit 0; fi
if [ ! -e "$path" ]; then echo missing; exit 0; fi
if [ -d "$path" ]; then kind=dir; elif [ -f "$path" ]; then kind=file; else echo special; exit 0; fi
if ! command -v tar >/dev/null 2>&1; then echo "the sandbox has no tar to pack $path with" >&2; exit 3; fi
if [ "$kind" = dir ]; then
  chmod -R u+rwX -- "$path" 2>/dev/null
  pack() { out=$1; shift; tar -cf "$out" "$@" -C "$path" .; }
else
  chmod u+r -- "$path" 2>/dev/null
  pack() { tar -cf "$1" -C "$(dirname -- "$path")" -- "$(basename -- "$path")"; }
fi
size=$( (pack - "$@" | head -c "$((max + 1))" | wc -c) 2>/dev/null )
if [ "$((size + 0))" -gt "$max" ]; then echo too_large; exit 0; fi
mkdir -p -- "${archive%/*}" || exit 3
pack "$archive" "$@" || exit 4
echo "$kind"
"""
_KINDS = frozenset({"missing", "symlink", "special", "too_large", "dir", "file"})
_COPY_CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """An artifact exists but could not be collected or handed to the verifier."""


@dataclass(frozen=True)
class CollectedArtifact:
    """A host copy of one artifact; ``source`` is the path as the task declares it (relative: from the workdir)."""

    source: str
    local: Path
    is_dir: bool
    convention: bool = False  # the /logs/artifacts directory, not a declared artifact

    def target(self, workdir: str) -> str:
        """Where the artifact goes in a sandbox whose workdir is ``workdir``."""

        return ARTIFACTS_DIR if self.convention else posixpath.join(workdir, self.source)


async def collect_artifacts(
    sandbox: Sandbox,
    artifacts: Sequence[Artifact],
    *,
    workdir: str,
    into: Path,
    skipped: dict[str, str] | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> list[CollectedArtifact]:
    """Copy the declared artifacts and ``/logs/artifacts`` from ``sandbox`` into ``into`` (a new or empty directory:
    ValueError otherwise), as directories and regular files only; ``skipped`` (when given) receives each sandbox path
    left out and why: ``symlink``, ``special`` (a device, FIFO or socket), ``link`` (a hard link to something else)
    or ``too_large`` (an archive of it would exceed ``max_bytes``)."""

    into.mkdir(parents=True, exist_ok=True)
    if held := sorted(entry.name for entry in into.iterdir()):
        raise ValueError(f"Artifacts are collected into a new or empty directory; {into} holds {held[:5]}")
    skipped = {} if skipped is None else skipped
    entries = [(artifact, False) for artifact in artifacts] + [(Artifact(source=ARTIFACTS_DIR), True)]
    stage = f"/tmp/platoon-artifacts-{uuid.uuid4().hex[:12]}"
    collected = []
    try:
        for number, (artifact, convention) in enumerate(entries):
            path = posixpath.join(workdir, artifact.source)
            archive = f"{stage}/{number}.tar"
            kind = await _pack(sandbox, path, archive, artifact.exclude, max_bytes)
            if kind in ("symlink", "special", "too_large"):
                skipped[path] = kind
            if kind not in ("dir", "file"):
                continue
            local, local_archive = into / str(number), into / f"{number}.tar"
            try:
                report = await sandbox.download(archive, local_archive)
            except Exception as exc:
                raise ArtifactError(f"cannot collect the artifact {path}: {type(exc).__name__}: {exc}") from exc
            _check_archive(path, local_archive, report)
            try:
                await asyncio.to_thread(_unpack, local_archive, local, path, skipped)
            finally:
                local_archive.unlink(missing_ok=True)
            if kind == "file":
                copies = list(local.iterdir())
                if not copies:
                    continue
                local = copies[0]
            elif convention and not any(local.iterdir()):
                continue  # nothing was published; the verifier's sandbox has the empty directory already
            collected.append(
                CollectedArtifact(source=artifact.source, local=local, is_dir=kind == "dir", convention=convention)
            )
    finally:
        try:
            await sandbox.exec(["rm", "-rf", stage], cwd="/", timeout_s=60)
        except Exception:
            pass  # leftover archives in a sandbox that is about to stop
    return collected


async def materialize_artifacts(
    sandbox: Sandbox, collected: Sequence[CollectedArtifact], *, workdir: str, readable: bool = False
) -> None:
    """Put each collected artifact at its path in ``sandbox`` (a directory replaces what is there).

    With ``readable`` (a verifier that runs as root, whose tests may run the agent's code as another user), the
    directories it makes get Docker's umask, and every artifact becomes readable by every user, its directories
    enterable (``chmod -R a+rX``), as an agent's files are under Docker's umask: the host's may have kept them from
    other users."""

    if not collected:
        return
    commands = [f"umask {DOCKER_UMASK}"] if readable else []
    for artifact in collected:
        target = artifact.target(workdir)
        if artifact.is_dir:
            commands.append(f"rm -rf -- {shlex.quote(target)} && mkdir -p -- {shlex.quote(target)}")
        else:
            commands.append(f"mkdir -p -- {shlex.quote(posixpath.dirname(target) or '/')}")
    result = await sandbox.exec(["/bin/sh", "-c", " && ".join(commands)], cwd="/", timeout_s=_TRANSFER_TIMEOUT_S)
    if result.exit_code != 0 or result.timed_out:
        raise ArtifactError(f"cannot prepare the artifacts' directories: {(result.stderr or result.stdout).strip()}")
    for artifact in collected:
        try:
            await sandbox.upload(artifact.local, artifact.target(workdir))
        except Exception as exc:
            raise ArtifactError(
                f"cannot hand the artifact {artifact.source} to the verifier: {type(exc).__name__}: {exc}"
            ) from exc
    if readable:
        targets = [artifact.target(workdir) for artifact in collected]
        result = await sandbox.exec(["chmod", "-R", "a+rX", "--", *targets], cwd="/", timeout_s=_TRANSFER_TIMEOUT_S)
        if result.exit_code != 0 or result.timed_out:
            raise ArtifactError(
                f"cannot make the artifacts readable to every user: {(result.stderr or result.stdout).strip()}"
            )


async def _pack(sandbox: Sandbox, path: str, archive: str, exclude: Sequence[str], max_bytes: int) -> str:
    """Pack ``path`` into ``archive`` in the sandbox (:data:`_PACK_SCRIPT`); its kind."""

    flags = [f"--exclude={pattern}" for pattern in exclude]
    result = await sandbox.exec(
        ["/bin/sh", "-c", _PACK_SCRIPT, "platoon-artifact", path, archive, str(max_bytes), *flags],
        cwd="/",
        timeout_s=_TRANSFER_TIMEOUT_S,
    )
    kind = result.stdout.strip().rsplit("\n", 1)[-1] if result.stdout.strip() else ""
    if result.exit_code != 0 or result.timed_out or kind not in _KINDS:
        detail = (result.stderr or result.stdout).strip()[-500:]
        raise ArtifactError(f"cannot pack the artifact {path} (exit {result.exit_code}): {detail}")
    return kind


def _check_archive(path: str, archive: Path, report: DownloadReport | None) -> None:
    """Raise :class:`ArtifactError` unless the download of ``path``'s archive brought back one regular file and left
    nothing out."""

    if report is not None and report.dropped_total:
        links = ", ".join(f"{link.path} -> {link.target} ({link.reason})" for link in report.dropped_links[:5])
        raise ArtifactError(
            f"cannot collect the artifact {path}: the download of its archive left out {report.dropped_total} "
            f"symlink(s) ({links}), so the sandbox replaced the archive"
        )
    if archive.is_symlink() or not archive.is_file():
        raise ArtifactError(f"cannot collect the artifact {path}: its archive did not come back as a regular file")


def _unpack(archive: Path, into: Path, origin: str, skipped: dict[str, str]) -> None:
    """Unpack the directories and regular files of ``archive`` (made from the sandbox path ``origin``) into ``into``,
    and record in ``skipped`` every other member it leaves out; a hard link to a regular file unpacked before it
    becomes a copy of that file.  Nothing is written through a symlink, as none is ever made."""

    into.mkdir(parents=True, exist_ok=True)
    unpacked: dict[tuple[str, ...], Path] = {}
    try:
        with tarfile.open(archive) as tar:
            for member in tar:
                parts = _parts(member.name)
                if parts is None:
                    skipped[f"{origin}/{member.name}"] = "unsafe_name"
                    continue
                target = into.joinpath(*parts)
                where = posixpath.join(origin, *parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isreg():
                    source = tar.extractfile(member)
                    assert source is not None
                    with source:
                        _write(source, target, member.mode)
                    unpacked[parts] = target
                elif member.islnk() and (linked := unpacked.get(_parts(member.linkname) or ())) is not None:
                    with linked.open("rb") as source:
                        _write(source, target, member.mode)
                    unpacked[parts] = target
                else:
                    skipped[where] = "symlink" if member.issym() else "link" if member.islnk() else "special"
    except (tarfile.TarError, OSError) as exc:
        raise ArtifactError(f"cannot unpack the artifact {origin}: {exc}") from exc


def _parts(name: str) -> tuple[str, ...] | None:
    """A member name's path below the archive's root (``()`` for the root); None for one that would leave it."""

    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    if name.startswith("/") or ".." in parts:
        return None
    return parts


def _write(source: IO[bytes], target: Path, mode: int) -> None:
    """Write ``source`` to the regular file ``target``, readable and writable by its owner, never through a link."""

    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "wb") as out:
        shutil.copyfileobj(source, out, _COPY_CHUNK)
        os.fchmod(out.fileno(), (mode & 0o777) | 0o600)
