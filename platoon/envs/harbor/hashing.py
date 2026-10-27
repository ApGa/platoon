"""Harbor's task content hash, computed without Harbor.

Harbor identifies a task package by a SHA-256 over the lines ``relpath\\0sha256(file)\\n``, sorted by POSIX
relative path, of ``task.toml``, ``instruction.md``, ``README.md`` and ``trajectory.json`` where present, plus
every file under ``environment/``, ``tests/``, ``solution/`` and ``steps/``. A ``.gitignore`` at the task root
filters that list; without one, Harbor's default ignores do (``__pycache__/``, ``*.pyc``, ...). The rule is
``Packager.compute_content_hash`` in Harbor 0.24 (``harbor/publisher/packager.py``); platoon records it in its
locks as ``harbor-packager-v1`` and computes it itself, so rollout workers never import Harbor.

The ``.gitignore`` patterns follow git's rules: ``#`` comments, ``!`` negation, a trailing ``/`` for directories,
patterns with an inner ``/`` anchored at the task root, ``*``, ``?``, ``[...]`` and ``**``. As in Harbor, every
pattern is matched against each file's relative path (a pattern that names a directory matches the files below
it) and the last matching pattern decides.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

HASH_RULE = "harbor-packager-v1"
DEFAULT_IGNORES = ("__pycache__/", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~")
GITIGNORE = ".gitignore"
TASK_FILES = ("task.toml", "instruction.md", "README.md", "trajectory.json")
TASK_DIRS = ("environment", "tests", "solution", "steps")
_CHUNK = 1024 * 1024


def file_sha256(path: Path) -> str:
    """The hex SHA-256 of a file's bytes."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_files(task_dir: str | Path) -> list[str]:
    """The POSIX relative paths of the files Harbor's hash covers, sorted.

    Symbolic links count as the files they point to, as in Harbor, whose ``rglob`` this is: a link to a directory
    below ``environment/``, ``tests/``, ``solution/`` or ``steps/`` is not followed, while one of those directories
    that is itself a link lists the directory it names.  A fetch refuses the links of a download or a checkout that
    leave their task (:mod:`platoon.envs.harbor.sources`).
    """

    root = Path(task_dir)
    found = [name for name in TASK_FILES if (root / name).is_file()]
    for directory in TASK_DIRS:
        base = root / directory
        if base.is_dir():
            found += [path.relative_to(root).as_posix() for path in base.rglob("*") if path.is_file()]
    ignore = root / GITIGNORE
    patterns = ignore.read_text().splitlines() if ignore.is_file() else DEFAULT_IGNORES
    matcher = IgnoreMatcher(patterns)
    return sorted(path for path in found if not matcher.ignored(path))


def content_hash(task_dir: str | Path, files: Sequence[str] | None = None) -> str:
    """Harbor's content hash of a task directory (hex), over ``files`` when given (as :func:`task_files` lists)."""

    root = Path(task_dir)
    outer = hashlib.sha256()
    for relative in task_files(root) if files is None else sorted(files):
        outer.update(f"{relative}\0{file_sha256(root / relative)}\n".encode())
    return outer.hexdigest()


class IgnoreMatcher:
    """``.gitignore`` patterns matched against relative file paths; the last matching pattern decides."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._rules = [rule for line in lines if (rule := _compile(line)) is not None]

    def ignored(self, relative: str) -> bool:
        ignored = False
        for regex, negated in self._rules:
            if regex.match(relative):
                ignored = not negated
        return ignored


def _compile(line: str) -> tuple[re.Pattern[str], bool] | None:
    pattern = _strip_trailing_spaces(line)
    if not pattern or pattern.startswith("#"):
        return None
    negated = pattern.startswith("!")
    if negated:
        pattern = pattern[1:]
    elif pattern.startswith("\\!") or pattern.startswith("\\#"):
        pattern = pattern[1:]
    directory_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    if not pattern:
        return None
    anchored = "/" in pattern  # a leading or inner slash anchors the pattern at the task root
    pattern = pattern.lstrip("/")
    if not pattern:
        return None
    segments = pattern.split("/")
    any_depth = "(?:.*/)?"
    body = ""
    for index, segment in enumerate(segments):
        first, last = index == 0, index == len(segments) - 1
        if segment == "**":
            body += ".*" if first and last else any_depth if first else "/.*" if last else "(?:/.*)?"
            continue
        if not first and not body.endswith(any_depth):
            body += "/"
        body += _glob(segment)
    prefix = "" if anchored else any_depth
    suffix = "/.*" if directory_only else "(?:/.*)?"
    return re.compile(f"^{prefix}{body}{suffix}$", re.DOTALL), negated


def _strip_trailing_spaces(line: str) -> str:
    line = line.rstrip("\n").rstrip("\r")
    end = len(line)
    while end > 0 and line[end - 1] == " ":
        if end >= 2 and line[end - 2] == "\\":
            break
        end -= 1
    return line[:end]


def _glob(segment: str) -> str:
    """One path segment's wildcards as a regular expression: ``*`` and ``?`` stop at ``/``."""

    out: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "\\" and index + 1 < len(segment):
            out.append(re.escape(segment[index + 1]))
            index += 2
            continue
        if char == "*":
            while index + 1 < len(segment) and segment[index + 1] == "*":
                index += 1
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            close = _class_end(segment, index)
            if close is None:
                out.append(re.escape(char))
            else:
                out.append(_char_class(segment[index + 1 : close]))
                index = close
        else:
            out.append(re.escape(char))
        index += 1
    return "".join(out)


def _class_end(segment: str, start: int) -> int | None:
    index = start + 1
    if index < len(segment) and segment[index] in "!^":
        index += 1
    if index < len(segment) and segment[index] == "]":
        index += 1
    while index < len(segment):
        if segment[index] == "\\":
            index += 2
            continue
        if segment[index] == "]":
            return index
        index += 1
    return None


def _char_class(content: str) -> str:
    negated = content[:1] in ("!", "^")
    if negated:
        content = content[1:]
    escaped = []
    index = 0
    while index < len(content):
        char = content[index]
        if char == "\\" and index + 1 < len(content):
            escaped.append(re.escape(content[index + 1]))
            index += 2
            continue
        escaped.append(char if char == "-" else re.escape(char))
        index += 1
    body = "".join(escaped)
    return f"[^/{body}]" if negated else f"[{body}]"
