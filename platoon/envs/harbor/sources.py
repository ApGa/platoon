"""Where a fetch takes Harbor datasets from besides Hugging Face and local directories: the Harbor Hub, Harbor's legacy
registry, git repositories and adapters (:mod:`platoon.envs.harbor.fetch` does the rest, and
:mod:`platoon.envs.harbor.references` reads their spellings).

    dataset: harbor://terminal-bench/terminal-bench@4              # a Hub package at a revision, digest or tag
    dataset: harbor://kumo@1.0                                     # the legacy registry; registry: another one
    dataset: git+https://github.com/org/repo@<commit>#datasets/x   # a directory of a git repository
    adapter: {repo: https://github.com/org/repo, commit: <commit>, path: adapters/x, args: [--split, test]}

**The Harbor Hub** (``harbor://org/name@ref``, a dataset or a single task): platoon never imports Harbor.  The Harbor
CLI at platoon's pin (:data:`HARBOR_CLI_VERSION`, run with ``uvx`` in an environment of its own, with its dependencies
as they were when it was released; the command line in ``PLATOON_HARBOR_CLI`` replaces it) resolves the reference to the
package's digest (``harbor version show --json``), with each task's digest for a dataset, and downloads that digest
(``harbor download ... --export``).  The CLI has no task filter, so a dataset downloads whole; the fetch then selects,
overlays, hashes and locks as for a directory, and fails on a task whose files do not hash to the digest the Hub gives
it.  Task ids are the tasks' names.  A revision number (``@4``) or a digest (``@sha256:...``) is pinned; a tag
(``@latest``, ``@v4.0.0``, or none) moves.

**Harbor's legacy registry** (``harbor://name@version``): platoon reads the ``registry.json`` that ``registry`` names
(a file, its directory, or a URL; default Harbor's own, :data:`DEFAULT_REGISTRY`) and takes each selected task from its
``git_url`` at its ``git_commit_id`` and ``path``, with the sparse clone below, rather than the CLI's clone of whole
repositories; a task in a Hugging Face dataset repository comes through ``hf://``'s downloads instead, which need no
git-lfs, and one without a ``git_url`` is a directory next to the registry file, which a fetch copies again every time,
as it does a directory.  It is pinned when it names a version and the registry is a file, or a URL with a commit in its
path; a task at ``HEAD`` (or none) resolves only with ``update``, and the lock records the commit it resolved to.

**A git repository** (``git+<url>@<commit>[#<path>]``): a shallow, blobless fetch of the commit, whose tree lists the
tasks under ``path`` (its ``registry.json``, else ``task-index.json``, else every directory that holds a
``task.toml``, as for a directory); only the selected tasks' files are checked out (sparse).  A full 40-character
commit is pinned; a branch or a tag moves (an abbreviated commit cannot be fetched).  Files kept in Git LFS need
``git lfs`` on the fetching host.

**An adapter** (``adapter:`` in place of ``dataset``) generates the tasks: the repository at ``commit`` is checked out
(whole, or only what the ``sparse`` patterns name), and ``command`` runs in its directory ``path`` with ``args`` and
``env``, in which ``{checkout}`` is the checkout and ``{output}`` the directory the tasks go to (``--output-dir
{output}`` is added when none of them names it).  The default command is the adapter project's own, as Harbor's
adapters run: ``uv run --project . <script>``, ``--locked`` when it has a ``uv.lock``, with the one script of its
``pyproject.toml``.  It runs with the fetching process's environment and network, never in a rollout; its output is
then fetched as a local directory.  The lock records the repository, commit, command, arguments and the digest of the
adapter's ``uv.lock``.

**Symbolic links** in what a source downloads or checks out may point only into their own task, and are taken as the
files they point to, as Harbor's content hash takes them.  A link that leaves its task (an absolute one, one that climbs
out, a task directory such as ``solution/`` that links elsewhere, or a task that is a link or lies behind one) would
bring files of the fetching host into the cache, the task's digest and every sandbox, so the fetch fails on it.

Workers never fetch: each source runs only in :func:`~platoon.envs.harbor.fetch.fetch`, which ``platoon harbor
fetch``, ``platoon prepare`` and the trainer call; a source that is not pinned (:attr:`FetchConfig.pinned
<platoon.envs.harbor.fetch.FetchConfig.pinned>` decides) opens only with ``update``.  Whatever a source downloads or
checks out goes to a directory of its own under the cache's ``.tmp``, removed once the tasks are cached.  A network
step is tried again while it fails for a reason that may pass; what the Harbor CLI or git words as a missing package,
repository or reference, or as no access, is not.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from platoon.envs.harbor.fetch import (
    TASK_INDEX,
    Candidate,
    FetchConfig,
    IndexFile,
    _candidates_from_index,
    _hashed_paths,
    _HashMemo,
    _HubSource,
    _index_path,
    _json_rows,
    _LocalSource,
    _retry,
    _Source,
    _task_name,
    _TaskFiles,
)
from platoon.envs.harbor.references import (
    _COMMIT,
    _DIGEST,
    _HUB,
    _LEGACY,
    _PACKAGE_NAME,
    _URL,
    REGISTRY,
    FetchError,
    GitReference,
    check_source,
    check_url,
    parse_registry,
    registry_dataset,
    registry_file,
    relative_path,
)

logger = logging.getLogger(__name__)

# The Harbor CLI a harbor:// fetch runs: the newest release (0.24.0, which has `harbor version show --json` and
# `harbor download --export`), with its dependencies as PyPI had them when it was released (on 2026-10-05), so that
# every host and every new uv cache installs the same packages. PLATOON_HARBOR_CLI, a command line, runs another one.
HARBOR_CLI_VERSION = "0.24.0"
HARBOR_CLI_EXCLUDE_NEWER = "2026-10-06T00:00:00Z"
HARBOR_CLI_ENV = "PLATOON_HARBOR_CLI"
# Harbor's legacy registry: the registry.json on its repository's main branch. Harbor 0.24.0's JSON registry client
# defaults to the same file under the repository's former name (laude-institute/harbor), and its own `harbor download
# name@version` asks the Hub's database for the dataset rather than reading the file.
DEFAULT_REGISTRY = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json"

_ATTEMPTS = 4  # tries of a network step that fails for a reason that may pass (a reset stream, a 5xx)
_RETRY_DELAY_S = 5.0
_HEX = re.compile(r"^[0-9a-f]{4,40}$")
_HF_GIT = re.compile(r"^https://huggingface\.co/datasets/(?P<repo>[\w.-]+/[\w.-]+?)(?:\.git)?/?$")
# How the Harbor CLI and git word a failure that no retry changes: phrases, never single words, matched only where
# they stand apart from names, so that a dataset, a task or a repository named "invalid-tasks" or "not-found" is no
# error. The Harbor CLI's usage errors (exit 2) and git's (exit 129) are permanent too.
_PERMANENT = re.compile(
    r"(?<![\w/.-])(?:not found|do not have access|permission denied|authentication failed|could not read username"
    r"|could not read password|couldn't find remote ref|not our ref|does not exist|does not appear to be a git"
    r" repository|is not a valid|only valid for|host key verification failed|validation error)(?![\w/-])(?!\.\w)",
    re.IGNORECASE,
)
# The Harbor CLI's own error lines: "Error: ..." and an exception's last line, not its progress, which names tasks.
_ERROR_LINE = re.compile(r"^(?:error\b|[A-Za-z_][\w.]*(?:Error|Exception|ExceptionGroup)\b)", re.IGNORECASE)
_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
_OWN_FILES = (
    "platoon takes only a task's own files, so that no file of the fetching host gets into the cache or a sandbox"
)


# --- opening a source ------------------------------------------------------------------------------------------


def open_source(config: FetchConfig, *, update: bool, log: Callable[[str], None]) -> _Source:
    """The source of ``config``'s dataset (or adapter), resolved.  One that is not pinned (:attr:`FetchConfig.pinned`
    decides) opens only with ``update``, which resolves it to what it names now."""

    if config.adapter is None:
        check_source(config.dataset)
    if not (update or config.pinned):
        raise _moving(config)
    if config.adapter is not None:
        return _AdapterSource(config, log=log)
    dataset = config.dataset
    if dataset.startswith("git+"):
        return _GitSource(config, log=log)
    if _HUB.match(dataset):
        return _HarborHubSource(config, log=log)
    if _LEGACY.match(dataset):
        return _RegistrySource(config, update=update, log=log)
    raise FetchError(f"{dataset}: not a source platoon knows")


def _moving(config: FetchConfig, what: str | None = None) -> FetchError:
    """The error for a moving source fetched without ``update``, naming what moves (``what``, else its reference)."""

    if what is None:
        if config.adapter is not None:
            what = f"the branch or tag {config.adapter.commit!r}"
        elif match := _HUB.match(config.dataset):
            what = f"the tag {match.group('ref') or 'latest'!r}"
        elif match := _LEGACY.match(config.dataset):
            what = "a registry on a branch" if match.group("version") else "no version"
        else:
            what = f"the branch or tag {GitReference.parse(config.dataset).ref or 'HEAD'!r}"
    return FetchError(
        f"{config.dataset} is a moving reference ({what}): pin it, or resolve it with `platoon harbor fetch CONFIG "
        "--update`, which records what it resolved to in the lock"
    )


def _work_dir(config: FetchConfig, kind: str) -> Path:
    path = config.cache / ".tmp" / f"{kind}-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


class _StepFailed(FetchError):
    """A try of a network step that failed; ``permanent`` when another try cannot change that."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


def _retrying(call: Callable[[], Any], what: str, *, attempts: int = _ATTEMPTS) -> Any:
    """``call()``, again after a pause while it fails for a reason that may pass (:class:`_StepFailed`); a permanent
    failure, any other error and the last try's failure propagate."""

    delay = _RETRY_DELAY_S
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except _StepFailed as exc:
            if exc.permanent or attempt == attempts:
                raise
            logger.warning("%s failed (attempt %d of %d), retrying in %.0f s: %s", what, attempt, attempts, delay, exc)
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def _permanent(output: str, *, error_lines: bool = False) -> bool:
    """Whether a tool's ``output`` says that it failed for a reason no retry changes (:data:`_PERMANENT`); with
    ``error_lines``, only its error lines say so (:data:`_ERROR_LINE`; a traceback's box drawing set aside)."""

    lines = (line.strip().lstrip("|│+ ") for line in output.splitlines())
    return any(_PERMANENT.search(line) for line in lines if not error_lines or _ERROR_LINE.match(line))


def _tail(text: str, lines: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


# --- the Harbor CLI --------------------------------------------------------------------------------------------


def harbor_cli() -> list[str]:
    """The Harbor CLI's command line: ``PLATOON_HARBOR_CLI`` when set, else the pinned release through ``uvx`` (or
    ``uv tool run``), in an environment of its own made with this interpreter, of the packages PyPI had when the
    release came out (:data:`HARBOR_CLI_EXCLUDE_NEWER`)."""

    if override := os.environ.get(HARBOR_CLI_ENV, "").strip():
        return shlex.split(override)
    if uvx := shutil.which("uvx"):
        tool = [uvx]
    elif uv := shutil.which("uv"):
        tool = [uv, "tool", "run"]
    else:
        raise FetchError(
            "harbor:// datasets are downloaded by the Harbor CLI, which platoon runs with uv's uvx: install uv "
            f"(https://docs.astral.sh/uv/), or set {HARBOR_CLI_ENV} to a Harbor CLI's command line"
        )
    return [
        *tool,
        "--python",
        sys.executable,
        "--exclude-newer",
        HARBOR_CLI_EXCLUDE_NEWER,
        "--from",
        f"harbor=={HARBOR_CLI_VERSION}",
        "harbor",
    ]


class _HarborCli:
    """Runs the Harbor CLI (:func:`harbor_cli`), retrying what may pass, without its telemetry."""

    def __init__(self) -> None:
        self.argv = harbor_cli()
        self.env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"}
        self.env.setdefault("HARBOR_TELEMETRY", "0")

    def run(
        self, args: Sequence[str], *, timeout: float | None = None, before: Callable[[], None] | None = None
    ) -> str:
        """The command's standard output; ``FetchError`` with the end of its output when it fails.  ``before`` runs
        ahead of each attempt."""

        argv = [*self.argv, *args]

        def call() -> str:
            if before is not None:
                before()
            try:
                done = subprocess.run(
                    argv,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",  # its progress bars, whatever the locale
                    env=self.env,
                    timeout=timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise FetchError(f"cannot run the Harbor CLI ({shlex.join(self.argv)}): {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                message = f"harbor {shlex.join(args)} did not finish within {timeout:.0f} s"
                raise _StepFailed(message, permanent=False) from exc
            if done.returncode != 0:
                output = f"{done.stdout}\n{done.stderr}"
                raise _StepFailed(
                    f"harbor {shlex.join(args)} failed (exit {done.returncode}): {_tail(output)}",
                    permanent=done.returncode == 2 or _permanent(output, error_lines=True),
                )
            return done.stdout

        return _retrying(call, f"harbor {args[0]}")

    def version(self) -> str:
        return self.run(["--version"], timeout=900).strip().splitlines()[-1].strip()

    def show(self, reference: str, *, tasks: bool = False) -> dict[str, Any]:
        """``harbor version show reference --json`` (with ``--tasks``): the package version it resolves to."""

        output = self.run(["version", "show", reference, "--json", *(["--tasks"] if tasks else [])], timeout=900)
        try:
            data = json.loads(output[output.index("{") :])
        except ValueError as exc:
            raise FetchError(f"harbor version show {reference}: its output is not JSON: {_tail(output, 5)}") from exc
        if not isinstance(data, dict):
            raise FetchError(f"harbor version show {reference}: expected a JSON object, got {data!r}")
        return data


def _digest(value: Any) -> str:
    """A Hub content hash as ``sha256:<hex>`` (the Hub writes a dataset's with the prefix, a task's without)."""

    text = str(value or "")
    digest = text if text.startswith("sha256:") else f"sha256:{text}"
    if not _DIGEST.match(digest):
        raise FetchError(f"the Harbor Hub gave the content hash {value!r}, which is not a SHA-256")
    return digest


class _HarborHubSource(_Source):
    """A Harbor Hub package, a dataset or one task, at the digest its reference resolves to (see the module
    docstring)."""

    def __init__(self, config: FetchConfig, *, log: Callable[[str], None]) -> None:
        match = _HUB.match(config.dataset)
        assert match is not None  # open_source checked
        self.config = config
        self.log = log
        self.package = f"{match.group('org')}/{match.group('name')}"
        ref = match.group("ref") or "latest"
        self.cli = _HarborCli()
        cli_version = self.cli.version()
        version = self.cli.show(f"{self.package}@{ref}")
        self.kind = version.get("type")
        digest = _digest(version.get("content_hash"))
        if version.get("yanked_at"):
            logger.warning("%s@%s is yanked: %s", self.package, ref, version.get("yanked_reason") or "no reason given")
        if self.kind == "dataset":
            # By its digest, which the reference resolved to: a tag moved since cannot give other tasks.
            tasks = self.cli.show(f"{self.package}@{digest}", tasks=True).get("tasks") or []
            self.tasks = self._dataset_tasks(tasks)
        elif self.kind == "task":
            self.tasks = {match.group("name"): digest}
        else:
            raise FetchError(f"{config.dataset}: the Harbor Hub package is a {self.kind!r}, not a dataset or a task")
        self.resolved = digest
        self.fetched_with = f"harbor=={cli_version} (cli)"
        self.details = {
            "kind": "harbor-hub",
            "package": self.package,
            "type": self.kind,
            "ref": ref,
            "digest": digest,
            "revision": version.get("revision"),
            "tags": sorted(version.get("tags") or []),
        }
        self.work: Path | None = None
        # The export, and the directory in it that holds the package's tasks (a dataset's own, under its name).
        self.export: Path | None = None
        self.prefix = match.group("name") if self.kind == "dataset" else ""

    def _dataset_tasks(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """``{task name: digest}`` from the rows of ``harbor version show --tasks``."""

        tasks: dict[str, str] = {}
        orgs: dict[str, str] = {}
        unavailable = 0
        for row in rows:
            task = row.get("task_version")
            if not isinstance(task, Mapping) or row.get("available") is False:
                unavailable += 1
                continue
            package = task.get("package") or {}
            name, org = package.get("name"), (package.get("org") or {}).get("name")
            # A task's name is a directory of the export and a task id: a package name, as the Hub's own rule has it.
            if not isinstance(name, str) or not _PACKAGE_NAME.match(name) or ".." in name:
                raise FetchError(
                    f"{self.config.dataset}: the Harbor Hub listed a task named {name!r}, which is not a package name "
                    f"(letters, digits, '.', '_' and '-'): {row!r}"
                )
            if name in tasks:
                raise FetchError(
                    f"{self.config.dataset}: two of its tasks are named {name!r} ({orgs[name]}/{name} and "
                    f"{org}/{name}); platoon names a Hub dataset's tasks by their names"
                )
            tasks[name], orgs[name] = _digest(task.get("content_hash")), str(org)
        if unavailable:
            raise FetchError(f"{self.config.dataset}: {unavailable} of its tasks are not available to this account")
        if not tasks:
            raise FetchError(f"{self.config.dataset} holds no task")
        return tasks

    def candidates(self) -> dict[str, Candidate]:
        self.work = _work_dir(self.config, "harbor-hub")
        export = self.export = self.work / "export"
        started = time.monotonic()
        self.log(f"downloading {self.package}@{self.resolved} ({len(self.tasks)} task(s)) with the Harbor CLI")
        # Into an empty directory each time: the CLI takes a task directory that exists for one downloaded already,
        # so a retry after a partial download would keep its partial tasks.
        self.cli.run(
            ["download", f"{self.package}@{self.resolved}", "--export", "-o", str(export)],
            before=lambda: shutil.rmtree(export, ignore_errors=True),
        )
        self.log(f"downloaded {self.package} in {time.monotonic() - started:.0f}s")
        root = export / self.prefix
        missing = sorted(name for name in self.tasks if not (root / name / "task.toml").is_file())
        if missing:
            raise FetchError(
                f"{self.config.dataset}: the Harbor CLI's export lacks {len(missing)} of its tasks, e.g. {missing[:3]}"
            )
        candidates = {name: Candidate(name=name, path=name, digest=digest) for name, digest in self.tasks.items()}
        index = self._package_file(TASK_INDEX)
        if index is not None:  # a dataset file of the package: index fields of its tasks
            for name, row in json.loads(index.read_text()).items():
                if name in candidates and isinstance(row, Mapping):
                    candidates[name].index.update({key: value for key, value in row.items() if key != "task_path"})
        return candidates

    def _package_file(self, relative: str) -> Path | None:
        """A file of a dataset package beside its tasks in the export (``task-index.json``, an ``index`` file), else
        None; ``FetchError`` for one that is a symbolic link, or lies behind one, as for a task's own files."""

        if self.export is None or self.kind != "dataset":
            return None
        name = "/".join(part for part in (self.prefix, relative) if part)
        if not (self.export / name).is_file():
            return None
        if _Links(self.export).behind(name):
            raise FetchError(f"{self.config.dataset}: {relative} is a symbolic link, or lies behind one: {_OWN_FILES}")
        return self.export / name

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        path = self._package_file(relative_path(spec.path, "index"))
        if path is None:
            raise FetchError(f"index: {self.config.dataset} has no file {spec.path} (a Hub dataset's files are read)")
        return _json_rows(path.read_text(), spec)

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        assert self.export is not None  # candidates() downloaded the package
        return _directory_files(self.export, self.prefix, selected, link=True)

    def close(self) -> None:
        if self.work is not None:
            shutil.rmtree(self.work, ignore_errors=True)


class _Links:
    """Which paths below ``base``, a directory a fetch downloaded or checked out, are symbolic links or lie behind one;
    each directory is asked once."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self._known: dict[str, bool] = {"": False}

    def behind(self, relative: str) -> bool:
        """Whether ``relative`` (a POSIX path below ``base``) is a link, or a directory above it below ``base`` is."""

        known = self._known.get(relative)
        if known is None:
            known = self.behind(relative.rpartition("/")[0]) or os.path.islink(self.base / relative)
            self._known[relative] = known
        return known


def _directory_files(base: Path, prefix: str, selected: Sequence[Candidate], *, link: bool) -> dict[str, _TaskFiles]:
    """The files Harbor's hash covers (and the ``.gitignore``) of each selected task, at ``prefix / candidate.path``
    in ``base``, a directory the fetch downloaded or checked out: a link that stays in its task as the file it names,
    and ``FetchError`` for a link that leaves its task (see the module docstring)."""

    real_base = base.resolve()
    links = _Links(base)
    out = {}
    for candidate in selected:
        relative = "/".join(part for part in (prefix, candidate.path) if part and part != ".")
        where = f"Harbor task {candidate.name!r}"
        if links.behind(relative):
            raise FetchError(f"{where} is a symbolic link, or lies behind one ({relative}): {_OWN_FILES}")
        task_dir = base / relative
        if not (task_dir / "task.toml").is_file():
            raise FetchError(f"{where} has no task.toml ({candidate.path})")
        inside = real_base / relative
        files = {}
        for name in _hashed_paths(task_dir):
            path = task_dir / name
            if links.behind(f"{relative}/{name}" if relative else name):
                path = path.resolve()
                if not path.is_relative_to(inside):
                    raise FetchError(
                        f"{where}: {name} is a symbolic link, or lies behind one, that leaves the task (to {path}): "
                        f"{_OWN_FILES}"
                    )
            files[name] = path
        out[candidate.name] = _TaskFiles(files=files, link=link)
    return out


# --- git -------------------------------------------------------------------------------------------------------


class _Git:
    """git commands in one repository, with no prompts, no Git LFS downloads and no line-ending conversion."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory  # None: no repository (ls-remote)
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "GIT_ASKPASS": "true"}

    def run(
        self, *args: str, input: bytes | None = None, check: bool = True, network: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        argv = [
            "git",
            *(["-C", str(self.directory)] if self.directory is not None else []),
            "-c",
            "core.autocrlf=false",
            "-c",
            "advice.detachedHead=false",
            "--literal-pathspecs",
            *args,
        ]

        def call() -> subprocess.CompletedProcess[bytes]:
            try:
                done = subprocess.run(argv, input=input, capture_output=True, env=self.env, check=False)
            except FileNotFoundError as exc:
                raise FetchError(f"git sources need git: {exc}") from exc
            if check and done.returncode != 0:
                stderr = done.stderr.decode(errors="replace")
                message = f"git {' '.join(args[:2])} failed (exit {done.returncode}): "
                message += _tail(stderr or done.stdout.decode(errors="replace"))
                if not network:
                    raise FetchError(message)
                # git -q writes only its own messages, its server's and ssh's to the standard error.
                raise _StepFailed(message, permanent=done.returncode == 129 or _permanent(stderr))
            return done

        return _retrying(call, f"git {args[0]}") if network else call()


def git_version() -> str:
    try:
        output = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FetchError(f"git sources need git: {exc}") from exc
    return output.strip().removeprefix("git version ").strip()


def resolve_ref(url: str, ref: str, *, where: str) -> str:
    """The commit a branch, a tag or ``HEAD`` (``ref`` empty) of the repository at ``url`` names now."""

    ref = ref or "HEAD"
    if _COMMIT.match(ref):
        return ref
    listing = _Git(None).run("ls-remote", url, ref, network=True)
    names: dict[str, str] = {}
    for line in listing.stdout.decode().splitlines():
        sha, _, name = line.partition("\t")
        names[name] = sha
    for name in (
        "HEAD" if ref == "HEAD" else f"refs/heads/{ref}",
        f"refs/tags/{ref}^{{}}",  # an annotated tag's commit
        f"refs/tags/{ref}",
        ref,
    ):
        if name in names:
            return names[name]
    hint = " (an abbreviated commit cannot be fetched: give all 40 characters)" if _HEX.match(ref) else ""
    raise FetchError(f"{where}: {url} has no branch or tag {ref!r}{hint}")


class _Checkout:
    """One commit of a repository: fetched shallow and without blobs, so that its tree can be listed and read, and
    checked out sparsely, which downloads only the files it checks out."""

    def __init__(self, url: str, commit: str, directory: Path, *, where: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.url, self.commit, self.where = url, commit, where
        self.directory = directory.resolve()  # what links in the checkout are compared with
        self.git = _Git(self.directory)
        self.git.run("init", "-q")
        self.git.run("remote", "add", "origin", url)
        self.git.run("config", "remote.origin.promisor", "true")
        self.git.run("config", "remote.origin.partialclonefilter", "blob:none")
        # A server without partial clone (a local repository) ignores the filter and sends the blobs too.
        self.git.run("fetch", "-q", "--no-tags", "--depth=1", "--filter=blob:none", "origin", commit, network=True)
        found = self.git.run("rev-parse", "--verify", "-q", f"{commit}^{{commit}}", check=False)
        if found.returncode != 0:
            raise FetchError(f"{where}: {url} has no commit {commit}")

    def files(self, prefix: str = "") -> list[str]:
        """The paths of the commit's files (``prefix`` and below), relative to the repository's root."""

        listing = self.git.run("ls-tree", "-r", "-z", "--name-only", self.commit, *(["--", prefix] if prefix else []))
        return [os.fsdecode(path) for path in listing.stdout.split(b"\0") if path]

    def read(self, path: str) -> bytes:
        """One file of the commit (downloaded alone)."""

        return self.git.run("cat-file", "blob", f"{self.commit}:{path}", network=True).stdout

    def check_out(self, *, directories: Iterable[str] = (), patterns: Sequence[str] = ()) -> None:
        """Check the commit out: only ``directories`` (with the files directly in their parents), or only what the
        sparse-checkout ``patterns`` match, or (neither) everything."""

        dirs = sorted(set(directories))
        if "" in dirs:  # the repository's root is a task: all of it
            dirs, patterns = [], ()
        if dirs:
            lines = "\n".join(dirs).encode() + b"\n"
            done = self.git.run(
                "sparse-checkout", "set", "--cone", "--sparse-index", "--stdin", input=lines, check=False
            )
            if done.returncode != 0:  # a git without the sparse index
                self.git.run("sparse-checkout", "set", "--cone", "--stdin", input=lines)
        elif patterns:
            self.git.run("sparse-checkout", "set", "--no-cone", "--stdin", input="\n".join(patterns).encode() + b"\n")
        self.git.run("checkout", "-q", "--detach", self.commit, network=True)


def _lfs_pointers(paths: Iterable[Path]) -> list[Path]:
    """The files among ``paths`` that are Git LFS pointers rather than their content."""

    found = []
    for path in paths:
        try:
            if path.stat().st_size < 1024 and path.read_bytes().startswith(_LFS_POINTER):
                found.append(path)
        except OSError:
            continue
    return found


def _resolve_lfs(checkout: _Checkout, files: Mapping[str, _TaskFiles]) -> None:
    """Replace the Git LFS pointers among the checked-out ``files`` with their content (``git lfs pull``), or fail,
    naming them, where git-lfs is missing: a pointer would hash, and run, as something else."""

    pointers = _lfs_pointers(path for task in files.values() for path in task.files.values())
    if not pointers:
        return
    relative = [path.relative_to(checkout.directory).as_posix() for path in pointers]
    if shutil.which("git-lfs") is None:
        raise FetchError(
            f"{checkout.where}: {len(pointers)} file(s) of the selected tasks are kept in Git LFS, e.g. "
            f"{relative[:3]}, and git-lfs is not installed: install it (https://git-lfs.com), or leave those tasks out"
        )
    for start in range(0, len(relative), 100):
        checkout.git.run("lfs", "pull", "--include", ",".join(relative[start : start + 100]), network=True)
    if left := _lfs_pointers(pointers):
        raise FetchError(f"{checkout.where}: git lfs pull left {len(left)} pointer(s), e.g. {left[0]}")


def _tree_candidates(files: Sequence[str], read: Callable[[str], bytes], label: str) -> dict[str, Candidate]:
    """A dataset's tasks from the paths of its files (relative to its root), as a fetch of the same directory finds
    them: its ``registry.json`` or ``task-index.json``, else the root itself when it holds a ``task.toml``, else every
    directory below it that holds one, without searching a task's own directories, hidden ones or ``__pycache__``."""

    present = set(files)
    if REGISTRY in present or TASK_INDEX in present:
        registry = json.loads(read(REGISTRY)) if REGISTRY in present else None
        task_index = json.loads(read(TASK_INDEX)) if TASK_INDEX in present else None
        return _candidates_from_index(registry, task_index, label)
    if "task.toml" in present:
        return {}  # the root is one task: the caller names it
    roots = sorted(
        {path.rsplit("/", 1)[0] for path in files if path.endswith("/task.toml")},
        key=lambda path: (path.count("/"), path),
    )
    tasks: list[str] = []
    for root in roots:
        parts = root.split("/")
        if any(part.startswith(".") or part == "__pycache__" for part in parts):
            continue
        if any(root.startswith(f"{task}/") for task in tasks):
            continue  # a task.toml among a task's own files is not another task
        tasks.append(root)
    candidates = {name: Candidate(name=name, path=name) for name in sorted(tasks)}
    if not candidates:
        raise FetchError(f"{label} holds no Harbor task (no task.toml)")
    return candidates


class _GitSource(_Source):
    """A directory of a git repository at one commit (see the module docstring)."""

    def __init__(self, config: FetchConfig, *, log: Callable[[str], None]) -> None:
        reference = GitReference.parse(config.dataset)
        commit = resolve_ref(reference.url, reference.ref, where=config.dataset)
        self.config, self.reference, self.log = config, reference, log
        self.work = _work_dir(config, "git")
        started = time.monotonic()
        try:
            self.checkout = _Checkout(reference.url, commit, self.work / "repo", where=config.dataset)
        except BaseException:
            shutil.rmtree(self.work, ignore_errors=True)
            raise
        log(f"fetched the tree of {reference.url} at {commit} in {time.monotonic() - started:.0f}s")
        self.resolved = commit
        self.fetched_with = f"git {git_version()}"
        self.details = {"kind": "git", "url": reference.url, "commit": commit, "path": reference.path}
        if reference.ref and reference.ref != commit:
            self.details["ref"] = reference.ref
        self.label = f"{reference.url}@{commit}" + (f"#{reference.path}" if reference.path else "")

    def _repository_path(self, relative: str) -> str:
        return "/".join(part for part in (self.reference.path, relative) if part and part != ".")

    def candidates(self) -> dict[str, Candidate]:
        prefix = self.reference.path
        paths = [path.removeprefix(f"{prefix}/") if prefix else path for path in self.checkout.files(prefix)]
        candidates = _tree_candidates(
            paths, lambda relative: self.checkout.read(self._repository_path(relative)), self.label
        )
        if not candidates:  # the directory is one task: named by its [task].name, else its directory
            config = self.work / "task.toml"
            config.write_bytes(self.checkout.read(self._repository_path("task.toml")))
            name = _task_name(config) or PurePosixPath(prefix or self.reference.url.rstrip("/")).name.removesuffix(
                ".git"
            )
            candidates = {name: Candidate(name=name, path=".")}
        return candidates

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        path = self._repository_path(relative_path(spec.path, f"index: {spec.path}"))
        if path not in set(self.checkout.files(path)):
            raise FetchError(f"index: {self.label} has no file {spec.path}")
        return _json_rows(self.checkout.read(path).decode(), spec)

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        started = time.monotonic()
        self.checkout.check_out(directories=[self._repository_path(candidate.path) for candidate in selected])
        self.log(f"checked out {len(selected)} task(s) of {self.label} in {time.monotonic() - started:.0f}s")
        files = _directory_files(self.checkout.directory, self.reference.path, selected, link=True)
        _resolve_lfs(self.checkout, files)
        return files

    def close(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


# --- Harbor's legacy registry ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _RegistryTask:
    name: str
    url: str | None  # None: a directory next to the registry file
    commit: str | None
    path: str


class _RegistrySource(_Source):
    """A dataset of Harbor's legacy registry at one version (see the module docstring)."""

    def __init__(self, config: FetchConfig, *, update: bool, log: Callable[[str], None]) -> None:
        self.config, self.log, self.update = config, log, update
        registry = config.registry or DEFAULT_REGISTRY
        entry, version = registry_dataset(self._read_registry(registry), config.dataset, registry)
        name = str(entry.get("name"))
        self.tasks = self._tasks(entry.get("tasks") or [])
        commits = {task.url: task.commit for task in self.tasks.values() if task.url is not None}
        # The dataset as resolved: its tasks, each at its commit.
        rows = sorted([task.name, task.url, task.commit, task.path] for task in self.tasks.values())
        self.resolved = "sha256:" + hashlib.sha256(json.dumps([name, version, rows]).encode()).hexdigest()
        self.fetched_with = f"platoon (registry.json), git {git_version()}" if commits else "platoon (registry.json)"
        self.details = {
            "kind": "harbor-registry",
            "registry": registry,
            "registry_sha256": self.registry_sha256,
            "name": name,
            "version": version,
            "repositories": [{"url": url, "commit": commit} for url, commit in sorted(set(commits.items()))],
        }
        self.work: Path | None = None

    def _read_registry(self, registry: str) -> list[Any]:
        self.registry_dir: Path | None = None  # a registry file's directory, where its tasks without a git_url are
        if _URL.match(registry):

            def call() -> bytes:
                try:
                    with urllib.request.urlopen(registry, timeout=120) as response:  # noqa: S310 (a URL the config names)
                        return response.read()
                except urllib.error.HTTPError as exc:
                    permanent = 400 <= exc.code < 500 and exc.code not in (408, 429)  # not found, no access
                    raise _StepFailed(f"registry {registry}: HTTP {exc.code}", permanent=permanent) from exc
                except (urllib.error.URLError, OSError) as exc:
                    raise _StepFailed(f"registry {registry}: {exc}", permanent=False) from exc

            raw = _retrying(call, f"reading {registry}")
        else:
            path = registry_file(Path(registry))
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise FetchError(f"registry: cannot read {path}: {exc}") from exc
            self.registry_dir = path.parent
        self.registry_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
        return parse_registry(raw, registry)

    def _tasks(self, rows: Sequence[Any]) -> dict[str, _RegistryTask]:
        """The dataset's tasks by name, each at a commit (a ``HEAD`` one resolved, with ``update``); rows Harbor's
        registry repeats are taken once, as Harbor takes them."""

        tasks: dict[str, _RegistryTask] = {}
        resolved: dict[tuple[str, str], str] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("name"), str) or row.get("path") is None:
                raise FetchError(f"{self.config.dataset}: a registry task without a name or a path: {row!r}")
            name, url = row["name"], row.get("git_url") or None
            path = _index_path(row["path"], name, f"{self.config.dataset}: {REGISTRY}")
            commit = str(row.get("git_commit_id") or "HEAD") if url is not None else None
            if url is None and self.registry_dir is None:
                raise FetchError(f"{self.config.dataset}: task {name!r} names no git_url, in a registry at a URL")
            if url is not None:
                check_url(url, f"{self.config.dataset}: task {name!r}")
            if url is not None and commit is not None and not _COMMIT.match(commit):
                if not self.update:
                    raise _moving(self.config, f"task {name!r} is at {commit} of {url}")
                key = (url, commit)
                if key not in resolved:
                    resolved[key] = self._resolve(url, commit)
                commit = resolved[key]
            task = _RegistryTask(name=name, url=url, commit=commit, path=path)
            if name in tasks and tasks[name] != task:
                raise FetchError(f"{self.config.dataset}: two registry tasks are named {name!r}")
            tasks[name] = task
        if not tasks:
            raise FetchError(f"{self.config.dataset} lists no task")
        return tasks

    def _resolve(self, url: str, ref: str) -> str:
        if match := _HF_GIT.match(url):
            from huggingface_hub import HfApi

            revision = "main" if ref == "HEAD" else ref
            return _retry(lambda: HfApi().repo_info(match.group("repo"), repo_type="dataset", revision=revision)).sha
        return resolve_ref(url, "" if ref == "HEAD" else ref, where=self.config.dataset)

    def candidates(self) -> dict[str, Candidate]:
        return {
            task.name: Candidate(
                name=task.name, path=f"{task.url}@{task.commit}#{task.path}" if task.url else task.path
            )
            for task in self.tasks.values()
        }

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        raise FetchError(f"index: {self.config.dataset} is a registry's list of tasks, which has no index files")

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        groups: dict[tuple[str | None, str | None], list[_RegistryTask]] = defaultdict(list)
        for candidate in selected:
            task = self.tasks[candidate.name]
            groups[(task.url, task.commit)].append(task)
        self.work = _work_dir(self.config, "registry")
        out: dict[str, _TaskFiles] = {}
        for (url, commit), tasks in sorted(groups.items(), key=lambda item: str(item[0])):
            local = [Candidate(name=task.name, path=task.path) for task in tasks]
            if url is None:
                assert self.registry_dir is not None  # checked in _tasks
                # Directories of the registry file's owner, fetched as a directory is (its links followed).
                out.update(_LocalSource(self.registry_dir).materialize(local, hashes))
            elif match := _HF_GIT.match(url):
                hub = _HubSource(
                    self.config, update=False, log=self.log, dataset=f"hf://{match.group('repo')}@{commit}"
                )
                out.update(hub.materialize(local, hashes))
            else:
                assert commit is not None
                started = time.monotonic()
                directory = self.work / hashlib.sha256(f"{url}@{commit}".encode()).hexdigest()[:16]
                checkout = _Checkout(url, commit, directory, where=f"{self.config.dataset}: {url}@{commit}")
                checkout.check_out(directories=[task.path for task in tasks])
                self.log(f"checked out {len(tasks)} task(s) of {url} at {commit} in {time.monotonic() - started:.0f}s")
                files = _directory_files(checkout.directory, "", local, link=True)
                _resolve_lfs(checkout, files)
                out.update(files)
        return out

    def close(self) -> None:
        if self.work is not None:
            shutil.rmtree(self.work, ignore_errors=True)


# --- adapters --------------------------------------------------------------------------------------------------


class _AdapterSource(_Source):
    """The tasks an adapter generates at one commit of its repository (see the module docstring)."""

    def __init__(self, config: FetchConfig, *, log: Callable[[str], None]) -> None:
        spec = config.adapter
        assert spec is not None  # open_source checked
        commit = resolve_ref(spec.repo, spec.commit, where=config.dataset)
        self.config, self.spec, self.log = config, spec, log
        self.work = _work_dir(config, "adapter")
        try:
            self._generate(commit)
        except BaseException:
            shutil.rmtree(self.work, ignore_errors=True)
            raise
        self.resolved = commit
        self.fetched_with = f"git {git_version()} (adapter)"

    def _generate(self, commit: str) -> None:
        spec = self.spec
        started = time.monotonic()
        checkout = _Checkout(spec.repo, commit, self.work / "repo", where=self.config.dataset)
        checkout.check_out(patterns=spec.sparse)
        self.log(f"checked out {spec.repo} at {commit} in {time.monotonic() - started:.0f}s")
        directory = checkout.directory / spec.path if spec.path else checkout.directory
        if not directory.is_dir():
            raise FetchError(f"{self.config.dataset}: the checkout has no directory {spec.path!r}")
        lock_file = directory / "uv.lock"
        command = list(spec.command) or _project_command(directory, self.config.dataset)
        output = self.work / "output"
        output.mkdir()
        values = {"checkout": str(checkout.directory), "output": str(output)}

        def fill(text: str) -> str:
            return text.replace("{checkout}", values["checkout"]).replace("{output}", values["output"])

        args = list(spec.args)
        if not any("{output}" in text for text in [*command, *args, *spec.env.values()]):
            args += ["--output-dir", "{output}"]  # Harbor's adapters all take it
        argv = [fill(part) for part in [*command, *args]]
        env = {**os.environ, **{name: fill(value) for name, value in spec.env.items()}}
        log_path = self.work / "adapter.log"
        self.log(f"running the adapter: {shlex.join(argv)} (in {spec.path or 'the repository'})")
        started = time.monotonic()
        with log_path.open("wb") as handle:
            try:
                done = subprocess.run(
                    argv, cwd=directory, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            except OSError as exc:
                raise FetchError(f"{self.config.dataset}: cannot run {argv[0]}: {exc}") from exc
        if done.returncode != 0:
            raise FetchError(
                f"{self.config.dataset}: the adapter failed (exit {done.returncode}): "
                f"{_tail(log_path.read_text(errors='replace'))}"
            )
        self.log(f"the adapter ran in {time.monotonic() - started:.0f}s")
        self.local = _LocalSource(output, link=True)
        self.details = {
            "kind": "adapter",
            "repo": spec.repo,
            "commit": commit,
            "path": spec.path,
            "command": command,
            "args": list(spec.args),
            "env": dict(spec.env),
            "sparse": list(spec.sparse),
            "uv_lock": ("sha256:" + hashlib.sha256(lock_file.read_bytes()).hexdigest())
            if lock_file.is_file()
            else None,
        }
        if spec.commit != commit:
            self.details["ref"] = spec.commit

    def candidates(self) -> dict[str, Candidate]:
        return self.local.candidates()

    def read_json_rows(self, spec: IndexFile) -> Iterator[tuple[str, Mapping[str, Any]]]:
        return self.local.read_json_rows(spec)

    def materialize(self, selected: Sequence[Candidate], hashes: _HashMemo) -> dict[str, _TaskFiles]:
        return self.local.materialize(selected, hashes)

    def close(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)


def _project_command(directory: Path, where: str) -> list[str]:
    """The adapter project's own command: ``uv run --project . [--locked] <script>``, with the one script its
    ``pyproject.toml`` declares."""

    try:
        project = tomllib.loads((directory / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise FetchError(
            f"{where}: name the adapter's command (adapter.command): no pyproject.toml to run ({exc})"
        ) from exc
    scripts = list((project.get("project") or {}).get("scripts") or {})
    if len(scripts) != 1:
        raise FetchError(
            f"{where}: name the adapter's command (adapter.command): its pyproject.toml declares "
            f"{len(scripts)} scripts, {scripts}, not one"
        )
    locked = ["--locked"] if (directory / "uv.lock").is_file() else []
    return ["uv", "run", "--project", ".", *locked, scripts[0]]
