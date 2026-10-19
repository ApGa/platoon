"""Nothing the repository holds names the private infrastructure or the internal work of the people who develop
platoon: no cluster, reservation, queue, host, path or account of theirs, no internal project, run or planning
document, and none of the people they work with.

The scan covers every text file git tracks, and every new file it does not ignore, outside hidden directories and
caches.  A file is text when its first 8000 bytes hold no NUL byte, as git judges it.  Without git, the scan walks
the repository.  A file may hold a banned word only in a text that ``ALLOWED`` lists, with the reason.

Banned words are stored as SHA-256 digests, so that this file does not publish them itself.  A line has its
camelCase words split with underscores (``FooBar`` reads as ``foo_bar``), is lower-cased and is split into words
(letters, digits and underscores), which hyphens may join into names.  A word matches when it or any run of its
underscore-separated parts is banned, also without its trailing digits (a numbered host); a hyphenated name also
matches when a run of two or more of its parts is (a document's file name); and each of these matches without a
trailing ``s`` too (a plural).  Three path patterns catch home directories, group storage and per-user data
directories, whoever they belong to.  File names are scanned like file contents.  To ban another word or name, add
``hashlib.sha256(word.lower().encode()).hexdigest()``.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BANNED_DIGESTS = frozenset(
    {
        "691c1224a07863335713b1c9f6b5d345529a6519636b7d52428234b0fef76b2e",  # a cluster
        "09a23692b3d972ae19593ac7d765c589f058f4ed0ad8c76d48b23bafd7d590e1",  # a reservation
        "ca2bdd5b6d4548191177406fbe92213341a4201b3512f1713716b9d63e7f81cd",  # a queue
        "092bf25babc151516ee10f1a279ee6ff18c985232ce50922039e6160af87bc20",  # a host name prefix
        "6d40d88d5fdd7975e3352fffdd19ee0488deeffd1f367853171c02c62897d8c0",  # a project group (and its paths)
        "c37b2536dc177e388d518a1e7fd3338870530827a916c2f5ce946f6899238b2d",  # a user account (and its paths)
        "a20f7c790c4e5b614044da5d481dcbf13aceee89f67f3cf0354095db43080134",  # the same account, shortened
        "73b4e0351271f9ba5fb13298de353999ac5572278f484fbfc53708fdbc169781",  # an account
        "a0c20e36c2b51ed79229c34df5f9952925691c8eff55e11409a327500ce2e0cc",  # the same account, shortened
        "f8197c2208c0b2ce0e67817c6bcd95ed48e1f6ad9ce5be4fededa96e95734fcc",  # an internal project
        "8b6b767c3df161e7401893c1f1056156911475cdc3bcbfe29c54865995c03dac",  # an internal run
        "a633b01aa452bb85d7c218221ad1ba659e5a060ff55a1a7113350b77290a29b4",  # an internal run
        "0e4a1b71a8e1df8f87a964dcc0b897b7ac458787a86aca82fac375369863e94d",  # an internal run
        "0b8fc9580cdd4e33bc5c14db43f6dbb412e76e2b843bfac07f91b16cd6d4f5d0",  # an internal run
        "6f568a7e009178d11fe5261f16b47ae3fee3a15f88e2fd7cf5ab6f2449579b3b",  # an internal run
        "c747f3646963f5be94a54bdbbbc15ee7e635abdaa97cd1d398a598a8281131d3",  # an internal run
        "f09116e554021da514197565eca30ad057a688726e52f1d5361afb3ab20330a4",  # an internal run
        "538a3e0aaf426a303cf73444f1a7d10d92b75b7b3d2e7bdb44d2aea148edcc9d",  # an internal planning document
        "80ba56186aef08eee6b033bcdbf8ee52ddeeb1ce3a8d680c4c334e2348b34f3d",  # an internal planning document
        "ee428421506b94fa7b46fde33e583f4e36d694c0495f3de6ac0bda608c11aa94",  # an internal planning document
        "b93828f5dfd3dc5862c7e28460109bd789c69a246cf96ce28ddcd7e536abf028",  # an internal planning document
        "1ea94ea96afcc25638e16e6847b6ebebab6db5fd92dd082fa88ead3e468b5522",  # an internal planning document
        "16b85a87c85489ace255fde8b857d356d943c7e022e5c1ac99173baa9081c6f5",  # an internal planning document
        "31bc8c97c89745668bed237e221d06393f70326c5ad4fa43877d65484da1f300",  # an internal planning document
        "3884cc1f562d03a88e6a3c44a804b01ccf8e5de1fe18dabf0407603283647de8",  # an internal planning document
        "0865d71d7489606010eb86c85aa635c98214c9c16f9d82a4ed6463806bf9d606",  # an internal planning document
        "dfbe949ddb8062e6b4a5fd6da6451a41fbc7a163de90e756ec27e8b43b78113b",  # an internal planning document
        "b5c14265f1dc8b1cb2c3e09aff65c5382b8bee8c0ca0e9f9d8e7e2569414578e",  # an internal planning document
        "b8c68359d21c00f9856c46b05024de616fc4551e635cf8a8a7948d7fb96648fb",  # an internal planning document
        "53adf83d9f7b4dd136fee848946a5ea6d28640406aa260d1bb6adb79dccb58ee",  # the people the developers work with
        "0e7166bcec7a1b2dbcb1f2ac975521307fe28e98a13985d2f7447e92a6f17318",  # the people the developers work with
    }
)

# An absolute path into a home directory or into group storage names the account and the machine a run used, and a
# per-user data directory names the account.  A URL path does not match, and neither does a ``<placeholder>`` for
# the account.
BANNED_PATHS = (
    re.compile(r"(?<![\w.~$-])/home/(?!<)[\w.-]+"),
    re.compile(r"(?<![\w.~$-])/groups(?=/)"),
    re.compile(r"(?<![\w.-])user_data/(?!<)[\w.-]+"),
)

# (file, text) -> why the file may hold that text although it names something banned.  Every entry must still be
# needed (see the last test).  Keep the list short: reword the file instead wherever its readers lose nothing.
ALLOWED: dict[tuple[str, str], str] = {}

_NAME = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*")
_CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Directories a checkout may hold untracked: hidden ones (environments, caches, git itself), caches and run outputs.
_SKIPPED_DIRECTORIES = {"__pycache__", "node_modules", "runs", "site-packages", "wandb"}


@cache
def _banned(name: str) -> bool:
    """Whether a lower-case word or hyphenated name is banned (see the module docstring)."""

    parts = name.split("-")
    candidates = {"-".join(parts[start:end]) for start in range(len(parts)) for end in range(start + 2, len(parts) + 1)}
    for part in parts:
        words = [word for word in part.split("_") if word]
        runs = {"_".join(words[start:end]) for start in range(len(words)) for end in range(start + 1, len(words) + 1)}
        candidates |= runs | {run.rstrip("0123456789") for run in runs}
    candidates |= {candidate.removesuffix("s") for candidate in candidates}
    return any(hashlib.sha256(candidate.encode()).hexdigest() in BANNED_DIGESTS for candidate in candidates)


def _hits(text: str, allowed: Iterable[str] = ()) -> list[str]:
    """``line N: 'match'`` for each banned word, name or path in ``text``, outside the ``allowed`` texts."""

    for exempt in allowed:
        text = text.replace(exempt, "")
    text = _CAMEL_CASE.sub("_", text).lower()
    names = {name for name in set(_NAME.findall(text)) if _banned(name)}
    if not names and not any(pattern.search(text) for pattern in BANNED_PATHS):
        return []
    hits = []
    for number, line in enumerate(text.split("\n"), 1):
        found = [name for name in _NAME.findall(line) if name in names]
        found += [match.group(0) for pattern in BANNED_PATHS for match in pattern.finditer(line)]
        hits += [f"line {number}: {item!r}" for item in found]
    return hits


def _skipped(relative: Path) -> bool:
    directories = relative.parts[:-1]
    return any((part.startswith(".") and part != ".github") or part in _SKIPPED_DIRECTORIES for part in directories)


def _is_text(path: Path) -> bool:
    with path.open("rb") as file:
        return b"\0" not in file.read(8000)


def _git_files(*options: str) -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", *options], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout
    return [Path(name) for name in listed.split("\0") if name]


def _scanned_files(*, text_only: bool = True) -> list[Path]:
    """Tracked files, and new files that git does not ignore; without git, a walk of the repository.  Only the text
    files, unless ``text_only`` is false."""

    try:
        tracked = set(_git_files("--cached"))
        new = _git_files("--others", "--exclude-standard")
    except (OSError, subprocess.CalledProcessError):
        tracked, new = set(), [path.relative_to(REPO) for path in REPO.rglob("*")]
    candidates = tracked | {path for path in new if not _skipped(path)}
    return sorted(path for path in candidates if (REPO / path).is_file() and (not text_only or _is_text(REPO / path)))


FILES = _scanned_files()


def test_the_scan_covers_every_text_file() -> None:
    names = {path.as_posix() for path in FILES}
    assert {"README.md", "CONTRIBUTING.md", "pyproject.toml", "uv.lock", "docs/README.md"} <= names
    assert {"platoon/__init__.py", "tests/docs/test_no_internal_references.py", ".pre-commit-config.yaml"} <= names
    for prefix in ("platoon/", "extensions/", "recipes/", "tests/", "examples/", "scripts/", ".github/"):
        assert any(name.startswith(prefix) for name in names), prefix
    assert not any(name.endswith(".png") for name in names)  # binary


# A project, a run, a planning document and a host name prefix that nothing real is called: the matcher test bans
# them, so that it needs no banned word.
MADE_UP_BANS = ("zorbix", "w99", "alpha-memo", "qnode")


@pytest.fixture
def made_up_bans(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ban ``MADE_UP_BANS`` as well, for one test."""

    digests = {hashlib.sha256(name.encode()).hexdigest() for name in MADE_UP_BANS}
    monkeypatch.setitem(globals(), "BANNED_DIGESTS", BANNED_DIGESTS | digests)
    _banned.cache_clear()
    yield
    _banned.cache_clear()


def test_the_matcher_finds_words_names_paths_and_numbered_hosts(made_up_bans: None) -> None:
    project, run, document, host = MADE_UP_BANS
    assert _banned(project) and _banned(f"{project}_harness") and _banned(f"{project}-openhands")
    assert not _banned(f"g{project}") and not _banned("export_hf") and not _banned("platoon")
    assert _banned(run) and not _banned("w1") and not _banned("w2") and not _banned("pydantic")
    # A planning document's name; its words alone are fine.
    assert _banned(document) and _banned(f"the-{document}") and not _banned("alpha") and not _banned("memo")
    assert _banned(f"{host}042") and _banned(f"{host}s") and not _banned(f"{host}x")
    assert _banned(f"{project}s") and _banned(f"{run}s") and not _banned(f"g{project}s")  # plurals
    assert _hits(f"class {project.capitalize()}Bridge:") == [f"line 1: '{project}_bridge'"]  # camelCase

    home, groups = "/".join(("", "home", "someone", "data")), "/".join(("", "groups", "team", "cache"))
    assert _hits(f"cache in {home}\nmore") == [f"line 1: {home.rsplit('/', 1)[0]!r}"]
    assert _hits(f'path = "{groups}"') == ["line 1: '/groups'"]
    assert not _hits(f"https://example.com{home} and https://example.com{groups} and ~{home}")
    assert not _hits("/".join(("", "home", "<user>", "data")))
    user_data = "/".join(("user_data", "someone"))
    assert _hits(f"log_path: ./data/{user_data}/runs") == [f"line 1: {user_data!r}"]
    assert not _hits("user_data/<user>/runs") and not _hits(f"my_{user_data}")
    assert _hits(f"x\n{run} and {project}") == [f"line 2: {run!r}", f"line 2: {project!r}"]
    assert not _hits(f"x {run} y", allowed=[f"{run} y"])


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.as_posix())
def test_no_internal_references(path: Path) -> None:
    text = (REPO / path).read_text(encoding="utf-8", errors="replace")
    hits = _hits(text, [exempt for (file, exempt) in ALLOWED if file == path.as_posix()])
    assert not hits, f"{path} names internal infrastructure, work or people:\n" + "\n".join(hits)


def test_no_file_name_names_internal_work() -> None:
    hits = {path.as_posix(): _hits(path.as_posix()) for path in _scanned_files(text_only=False)}
    assert not {path: found for path, found in hits.items() if found}


def test_every_allowed_text_is_still_needed() -> None:
    for (file, exempt), reason in ALLOWED.items():
        assert reason.strip(), f"say why {file} may hold {exempt!r}"
        assert exempt in (REPO / file).read_text(encoding="utf-8", errors="replace"), f"{file} no longer holds it"
        assert _hits(exempt), f"{exempt!r} names nothing banned: drop its ALLOWED entry"
