"""A Harbor task's input files: the files its agents' sandbox takes from the task, which a hold-out keeps together.

A task's image takes files from its build context, ``environment/``: its Dockerfile's ``COPY`` and ``ADD`` steps,
whether a build of its image bakes a file in (the build prefix) or a session uploads it when it opens (the deferred
steps, :mod:`platoon.envs.harbor.dockerfile`); a task with a prebuilt image and no Dockerfile gets the directory
whole, as Harbor uploads it.  Those are its input files: the data its agents work on, the same however the image is
built (an overlay that moves a ``COPY`` behind a ``RUN`` changes nothing here).  A main-only compose file's binds of
the task's own files (``./data:/data:ro``, :mod:`platoon.envs.harbor.compose`) put them in the sandbox too: they are
inputs, but for what the bind's volumes mask.  Never inputs, bound or not:

- the build's own files at the context's root (``Dockerfile``, ``.dockerignore``, the compose file), and the task's
  own definition (``task.toml``, ``instruction.md``), which a templated dataset's tasks share byte for byte, so that a
  hold-out would join them all by those;
- a file of the context that no step takes (a leftover helper script, say), which no sandbox holds;
- the tests and the solution, which the agents never see (and which join no tasks even where a bind leaves them
  unmasked), and a separate verifier's own build context;
- remote content (``ADD <url>``, ``COPY --from``), which is not the task's.

A task whose image cannot be planned, or whose ``task.toml`` cannot be read, has no input files platoon knows of: a
fetch refuses it (``image``, ``task_config``), and it never runs.

**Text compares as text** (:func:`input_digest`).  A public table is uploaded under several names, and not always
byte for byte: one dataset's Titanic tables come with CRLF line endings and with LF ones, its IBM attrition table with
a byte-order mark and without.  So a text file's digest is the SHA-256 of its text with its line endings, a leading
byte-order mark, the spaces and tabs that end its lines, and its blank lines normalized away; a binary file's (one with
a NUL byte among its first 8000 bytes, as git tells them apart) is the SHA-256 of its bytes.  Text already in that
form (LF line endings, the last line's too, no byte-order mark, no blank lines, no blanks at a line's end) has the
file's own SHA-256 as its digest, and text that normalizes to nothing has an empty file's, which joins nothing.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from platoon.envs.harbor.dockerfile import DOCKERFILE, DockerfileError, plan_dockerfile
from platoon.envs.harbor.hashing import file_sha256, task_files
from platoon.envs.harbor.task import COMPOSE_FILENAMES, CONFIG_FILENAME, INSTRUCTION_FILENAME, TESTS_DIR

if TYPE_CHECKING:
    from platoon.envs.harbor.compose import ComposePlan

CONTEXT_DIR = "environment"
BUILD_FILES = frozenset({DOCKERFILE, ".dockerignore", *COMPOSE_FILENAMES})  # at the context's root
# What a bind of the task never makes an input (besides BUILD_FILES): its definition, its tests and its solution.
_DEFINITION = frozenset({CONFIG_FILENAME, INSTRUCTION_FILENAME})
_HIDDEN_DIRS = frozenset({TESTS_DIR, "solution"})
# The digest of an empty file, which holds no data (and of text that normalizes to nothing): a hold-out joins no
# groups by it.
EMPTY_FILE = "sha256:" + hashlib.sha256(b"").hexdigest()
_CHUNK = 1024 * 1024
_BINARY_PROBE = 8000  # a NUL byte among a file's first 8000 bytes makes it binary, as git decides
_BOM = b"\xef\xbb\xbf"
_LINE_END_BLANKS = re.compile(rb"[ \t]+\n")
_BLANK_LINES = re.compile(rb"\n\n+")
_LONG_LINE = 4 * _CHUNK  # an unfinished line this long is passed on, but for the blanks it may end with


def copied_paths(task_dir: str | Path) -> tuple[str, ...]:
    """The paths of ``environment/`` that the task's agents' image takes, relative to it (``.`` for all of it): the
    sources of its Dockerfile's ``COPY`` and ``ADD`` steps from the context, or ``.`` for a prebuilt image without a
    Dockerfile or compose file, whose directory Harbor uploads; none when the image takes nothing from the task or
    cannot be planned."""

    root = Path(task_dir)
    try:
        config = tomllib.loads((root / CONFIG_FILENAME).read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ()
    environment = config.get("environment")
    if not isinstance(environment, dict):
        environment = {}
    context = root / CONTEXT_DIR
    dockerfile = context / DOCKERFILE
    if environment.get("docker_image"):
        compose = any((context / name).is_file() for name in COMPOSE_FILENAMES)
        return (".",) if context.is_dir() and not dockerfile.is_file() and not compose else ()
    if not dockerfile.is_file():
        return ()
    try:
        return plan_dockerfile(dockerfile, context, with_key=False).copied
    except (DockerfileError, OSError, ValueError):
        return ()


def input_files(
    task_dir: str | Path, *, digest: Callable[[Path], str] | None = None, compose: ComposePlan | None = None
) -> dict[str, str]:
    """The task's input files (:func:`copied_paths`, and the files ``compose``'s binds of ``main`` take,
    :func:`bound_paths`, less the build's own files, the task's definition, its tests and its solution), by
    task-relative path (``environment/input/sales.csv``), each with ``"sha256:<hex>"`` of its digest
    (:func:`input_digest`), sorted by path.  Only files that Harbor's content hash covers count
    (:func:`~platoon.envs.harbor.hashing.task_files`), so a lock's task digest vouches for them.

    ``digest`` gives a file's digest (default: :func:`input_digest`, remembered for this process by device, inode,
    size, modification time and change time).
    """

    root = Path(task_dir)
    copied = copied_paths(root)
    bound = bound_paths(compose)
    if not copied and not bound:
        return {}
    digest = digest or _remembered_digest
    files: dict[str, str] = {}
    for relative in task_files(root):
        top, _, rest = relative.partition("/")
        built = top == CONTEXT_DIR and rest and rest not in BUILD_FILES
        if built and any(path == "." or rest == path or rest.startswith(f"{path}/") for path in copied):
            files[relative] = f"sha256:{digest(root / relative)}"
        elif (
            relative not in _DEFINITION
            and top not in _HIDDEN_DIRS
            and not (top == CONTEXT_DIR and rest in BUILD_FILES)
            and any(
                _within(relative, source) and not any(_within(relative, mask) for mask in masks)
                for source, masks in bound
            )
        ):
            files[relative] = f"sha256:{digest(root / relative)}"
    return dict(sorted(files.items()))


def bound_paths(compose: ComposePlan | None) -> list[tuple[str, tuple[str, ...]]]:
    """The task paths a main-only compose file binds into ``main`` (``.`` for the whole task), each with the paths
    below it that its volumes mask, relative to the task."""

    main = compose.main() if compose is not None else None
    if main is None:
        return []
    return [
        (bind.task, tuple(_join(bind.task, mask) for mask in bind.masks))
        for bind in main.binds
        if bind.task is not None and not bind.missing
    ]


def _within(relative: str, path: str) -> bool:
    return path == "." or relative == path or relative.startswith(f"{path}/")


def _join(base: str, relative: str) -> str:
    return relative if base == "." else f"{base}/{relative}"


def input_digest(path: Path, *, sha256: Callable[[Path], str] = file_sha256) -> str:
    """The hex digest by which a hold-out compares the input file at ``path``: a binary file's SHA-256 (``sha256``), or
    the SHA-256 of a text file's text as :func:`normalized_text` gives it, with a leading UTF-8 byte-order mark
    dropped.  One table saved with CRLF or LF line endings, with a byte-order mark or without, has one digest."""

    with path.open("rb") as handle:
        first = handle.read(_CHUNK)
        if b"\0" in first[:_BINARY_PROBE]:
            return sha256(path)
        digest = hashlib.sha256()
        for block in normalized_text(_chunks(first.removeprefix(_BOM), handle)):
            digest.update(block)
    return digest.hexdigest()


def normalized_text(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """Text, given in chunks that may cut it anywhere (a CRLF included), with CRLF and CR line endings as LF, the
    spaces and tabs that end each line dropped, and blank lines dropped; every line it yields ends with LF."""

    carry = b""  # an unfinished line, and a CR that may begin a CRLF
    in_line = False  # whether what was yielded so far ends inside a line
    for chunk in chunks:
        data = carry + chunk
        cr = data.endswith(b"\r")
        data = _lf(data[:-1] if cr else data)
        cut = data.rfind(b"\n") + 1
        if not cut and len(data) >= _LONG_LINE:
            cut = len(data.rstrip(b" \t"))
        block, carry = _tidy(data[:cut]), data[cut:] + (b"\r" if cr else b"")
        if not in_line:
            block = block.lstrip(b"\n")
        if block:
            yield block
            in_line = not block.endswith(b"\n")
    tail = _tidy(_lf(carry) + b"\n")  # the last line ends too
    if not in_line:
        tail = tail.lstrip(b"\n")
    if tail:
        yield tail


def _lf(data: bytes) -> bytes:
    """CRLF and CR as LF, each looked for first: most text has neither, or CRLF alone."""

    if b"\r" not in data:
        return data
    data = data.replace(b"\r\n", b"\n")
    return data.replace(b"\r", b"\n") if b"\r" in data else data


def _tidy(lines: bytes) -> bytes:
    """Lines ending with LF, without the spaces and tabs before each LF and without blank lines (all but a first LF,
    which the caller drops when no line is open)."""

    if b" \n" in lines or b"\t\n" in lines:
        lines = _LINE_END_BLANKS.sub(b"\n", lines)
    return _BLANK_LINES.sub(b"\n", lines) if b"\n\n" in lines else lines


def _chunks(first: bytes, handle: BinaryIO) -> Iterator[bytes]:
    yield first
    yield from iter(lambda: handle.read(_CHUNK), b"")


# (device, inode, size, modification time, change time) -> input digest.  The change time keeps a file created on the
# inode of a removed one, with its size and modification time (a copy keeps its source's), from taking its digest:
# creating a file sets its change time, which a copy cannot set back.
_DIGESTS: dict[tuple[int, int, int, int, int], str] = {}


def _remembered_digest(path: Path) -> str:
    info = path.stat()
    key = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    known = _DIGESTS.get(key)
    if known is None:
        known = _DIGESTS[key] = input_digest(path)
    return known
