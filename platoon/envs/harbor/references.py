"""How a Harbor entry names where its tasks come from, read without fetching anything.

A Harbor entry's ``dataset`` is a reference: ``hf://org/name@revision[#subdir]`` (a Hugging Face dataset repository),
``harbor://org/name@ref`` (a package of the Harbor Hub), ``harbor://name@version`` (a dataset of Harbor's legacy
registry, which ``registry`` may name), ``git+<url>@<commit>[#<path>]`` (a directory of a git repository) or a local
directory; or its ``adapter`` (:class:`AdapterSpec`) generates the tasks instead.  This module parses and checks those
spellings, says which of them are pinned, and reads what of a ``registry`` a lock's inputs digest hashes;
:mod:`platoon.envs.harbor.fetch` fetches ``hf://`` datasets and directories, and :mod:`platoon.envs.harbor.sources` the
rest.  A URL that carries credentials (``https://user:token@...``) is refused wherever a source is named: the lock and
the fetch's log lines record every URL as written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.version import InvalidVersion, Version

REGISTRY = "registry.json"

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_SEGMENT = re.compile(r"/[0-9a-f]{40}(?:/|$)")
# A Harbor package name, or either half of org/name: Harbor's ORG_NAME_PATTERN, which also refuses "..".
_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_PACKAGE_NAME = re.compile(rf"^{_NAME}$")
_HUB = re.compile(rf"^harbor://(?P<org>{_NAME})/(?P<name>{_NAME})(?:@(?P<ref>[^\s@#/]+))?$")
_LEGACY = re.compile(rf"^harbor://(?P<name>{_NAME})(?:@(?P<version>[^\s@#/]+))?$")
_GIT_SCHEMES = ("https://", "http://", "ssh://", "git://", "file://")
_ADAPTER_KEYS = ("repo", "commit", "path", "command", "args", "env", "sparse")


class FetchError(ValueError):
    """A dataset that cannot be fetched as configured."""


def check_source(dataset: str) -> None:
    """Raise :class:`FetchError` unless ``dataset`` spells a ``hf://``, ``harbor://`` or ``git+`` source."""

    if dataset.startswith("hf://"):
        return
    if dataset.startswith("adapter:"):
        raise FetchError(f"{dataset}: name an adapter with adapter: {{repo, commit, path, ...}}, not as a dataset")
    if dataset.startswith("harbor://"):
        if not (_HUB.match(dataset) or _LEGACY.match(dataset)) or ".." in dataset.partition("@")[0]:
            raise FetchError(
                f"{dataset}: expected harbor://org/name@ref (a Hub package at a revision, a sha256: digest or a tag) "
                "or harbor://name@version (Harbor's legacy registry)"
            )
        return
    if dataset.startswith("git+"):
        GitReference.parse(dataset)
        return
    raise FetchError(f"{dataset}: not a source platoon knows")


def source_pinned(dataset: str) -> bool:
    """Whether a ``harbor://`` Hub package or a ``git+`` directory names immutable content: a Hub revision (``@4``)
    or digest (``@sha256:...``), a full git commit.  A dataset of the legacy registry is pinned only by its registry
    (:func:`legacy_registry_pinned`), so it counts as moving here."""

    if match := _HUB.match(dataset):
        return hub_ref_pinned(match.group("ref"))
    if dataset.startswith("git+"):
        return GitReference.parse(dataset).pinned
    return False


def hub_ref_pinned(ref: str | None) -> bool:
    """Whether a Hub reference is a revision or a digest; a tag, ``latest`` or none moves."""

    return bool(ref and (ref.isdigit() or _DIGEST.match(ref)))


def is_legacy_registry(dataset: str) -> bool:
    """Whether ``dataset`` names a dataset of Harbor's legacy registry (``harbor://name@version``)."""

    return bool(_LEGACY.match(dataset))


def legacy_registry_pinned(dataset: str, registry: str | None) -> bool:
    """Whether a ``harbor://name@version`` dataset is pinned: it names a version, and its registry is a local file,
    or a URL with a commit in its path (the default registry, None, is Harbor's on a branch, which moves)."""

    match = _LEGACY.match(dataset)
    if match is None or not match.group("version") or registry is None:
        return False
    return not _URL.match(registry) or bool(_COMMIT_SEGMENT.search(registry.split("?", 1)[0]))


def registry_inputs(registry: str) -> Any:
    """What the lock's inputs digest hashes of a ``registry`` setting: a URL as written; a file by its SHA-256 too,
    so that a lock no longer answers once the file changes."""

    if _URL.match(registry):
        return registry
    path = registry_file(Path(registry))
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise FetchError(f"registry: cannot read {path}: {exc}") from exc
    return {"path": str(path), "sha256": digest}


def registry_has_local_tasks(dataset: str, registry: str) -> bool:
    """Whether the ``harbor://name@version`` dataset that the registry file ``registry`` lists takes tasks without a
    ``git_url``: directories next to the file, which a fetch copies again every time, as it copies a directory.  False
    for a registry at a URL, which cannot name such tasks, and for a file that cannot be read or does not list the
    dataset: the fetch reports those."""

    if _URL.match(registry):
        return False
    try:
        data = parse_registry(registry_file(Path(registry)).read_bytes(), registry)
        entry, _ = registry_dataset(data, dataset, registry)
    except (OSError, FetchError):
        return False
    return any(isinstance(row, Mapping) and not row.get("git_url") for row in entry.get("tasks") or [])


def registry_file(path: Path) -> Path:
    """A ``registry`` that names a directory names its ``registry.json``."""

    return path / REGISTRY if path.is_dir() else path


def parse_registry(raw: bytes, registry: str) -> list[Any]:
    """The datasets of a ``registry.json``: a list of them, or an object whose ``datasets`` lists them."""

    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise FetchError(f"registry {registry} is not JSON: {exc}") from exc
    if isinstance(data, Mapping) and isinstance(data.get("datasets"), list):
        data = data["datasets"]
    if not isinstance(data, list):
        raise FetchError(f"registry {registry}: expected a list of datasets")
    return data


def registry_dataset(data: Sequence[Any], dataset: str, registry: str) -> tuple[Mapping[str, Any], str]:
    """The entry of the registry ``registry`` (its ``data``) for ``harbor://name@version``, and its version, as
    Harbor's JSON registry client takes it: with no version, Harbor's newest (``head``, else the highest that reads as
    a version); of two entries with one name and version, the last."""

    match = _LEGACY.match(dataset)
    if match is None:
        raise FetchError(f"{dataset}: expected harbor://name@version")
    name, version = match.group("name"), match.group("version")
    entries = [entry for entry in data if isinstance(entry, Mapping) and entry.get("name") == name]
    versions = [str(entry.get("version")) for entry in entries]
    if not entries:
        raise FetchError(f"{dataset}: the registry {registry} has no dataset {name!r}")
    if version is None:
        version = _newest_version(versions)
    if version not in versions:
        raise FetchError(f"{dataset}: the registry has no version {version!r} of {name} (it has {versions})")
    return entries[len(versions) - 1 - versions[::-1].index(version)], version


def _newest_version(versions: Sequence[str]) -> str:
    """The version Harbor takes when none is named: ``head``, else the highest that reads as a version, else the
    last in sort order."""

    if "head" in versions:
        return "head"
    parsed = []
    for version in versions:
        try:
            parsed.append((Version(version), version))
        except InvalidVersion:
            continue
    return max(parsed)[1] if parsed else sorted(versions)[-1]


def check_url(url: str, where: str) -> None:
    """Refuse a URL that carries credentials, a password or (over http and https) a token as its user name: the lock,
    its ``source`` and the fetch's log lines record the URL as written.  The error shows it with them blanked."""

    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError as exc:
        raise FetchError(f"{where}: not a URL ({exc})") from exc
    if parts.password is None and not (parts.username is not None and parts.scheme in ("http", "https")):
        return
    redacted = parts._replace(netloc="***@" + parts.netloc.rpartition("@")[2]).geturl()
    raise FetchError(
        f"{where}: {redacted} carries credentials, which the lock and the logs would record: leave them out of the URL "
        "(git takes them from a credential helper, ~/.netrc or an SSH key; a private registry.json can be downloaded "
        "and named as a file)"
    )


@dataclass(frozen=True)
class GitReference:
    """``git+<url>@<ref>[#<path>]``: a repository, a commit (or a branch or tag, which move) and a directory in it."""

    url: str
    ref: str  # "" for the remote's HEAD
    path: str  # relative to the repository's root; "" for the root

    @classmethod
    def parse(cls, dataset: str) -> GitReference:
        location, _, path = dataset.removeprefix("git+").partition("#")
        head, sep, ref = location.rpartition("@")
        # An @ before the host's path is a user (git+ssh://git@host/repo), not a reference.
        url, ref = (head, ref) if sep and "/" in head.partition("://")[2] else (location, "")
        if not url.startswith(_GIT_SCHEMES):
            raise FetchError(f"{dataset}: expected git+<https, ssh, git or file URL>@<commit>[#<path>]")
        check_url(url, "dataset")
        return cls(url=url, ref=ref, path=relative_path(path, dataset))

    @property
    def pinned(self) -> bool:
        return bool(_COMMIT.match(self.ref))


def relative_path(path: str, where: str) -> str:
    """``path`` as a relative POSIX path without ``.`` parts; ``FetchError`` for one that is absolute or has ``..``."""

    posix = PurePosixPath(path.strip())
    if posix.is_absolute() or ".." in posix.parts:
        raise FetchError(f"{where}: {path!r} must be a relative path inside the repository")
    return "/".join(part for part in posix.parts if part != ".")


@dataclass(frozen=True)
class AdapterSpec:
    """An ``adapter:`` source: the adapter in ``path`` of ``repo`` at ``commit``, run with ``command`` (default: its
    project's own script, through ``uv``), ``args`` and ``env``, on a checkout of the repository, whole or as the
    ``sparse`` patterns say (git's sparse-checkout patterns, which ``!`` negates)."""

    repo: str  # a git URL, or a local repository
    commit: str  # a full commit is pinned; a branch or a tag moves
    path: str = ""
    command: tuple[str, ...] = ()
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    sparse: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: Mapping[str, Any], *, resolve: Callable[[str], str] = os.path.abspath) -> AdapterSpec:
        """An adapter from its config mapping; a ``repo`` that is a local path resolves with ``resolve`` (a config
        file's is absolute already: ``adapter.repo`` is a path option, relative to the file)."""

        if not isinstance(value, Mapping):
            raise FetchError(f"adapter must be a mapping of {', '.join(_ADAPTER_KEYS)}, got {value!r}")
        if unknown := sorted(set(value) - set(_ADAPTER_KEYS)):
            raise FetchError(f"adapter: unknown keys {unknown}; known: {', '.join(_ADAPTER_KEYS)}")
        for key in ("repo", "commit"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise FetchError(
                    f"adapter.{key} must name the adapter's repository and its commit, got {value.get(key)!r}"
                )

        def strings(key: str) -> tuple[str, ...]:
            items = value.get(key) or ()
            if isinstance(items, str) or not all(isinstance(item, str) for item in items):
                raise FetchError(f"adapter.{key} must be a list of strings, got {items!r}")
            return tuple(items)

        env = value.get("env") or {}
        if not isinstance(env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise FetchError(f"adapter.env must map names to strings, got {env!r}")
        repo = value["repo"]
        if _URL.match(repo):
            check_url(repo, "adapter.repo")
        elif not repo.startswith("git@"):  # git@host:org/repo is a git address too
            repo = resolve(repo)
        return cls(
            repo=repo,
            commit=value["commit"],
            path=relative_path(str(value.get("path") or ""), "adapter.path"),
            command=strings("command"),
            args=strings("args"),
            env=dict(env),
            sparse=strings("sparse"),
        )

    @property
    def pinned(self) -> bool:
        return bool(_COMMIT.match(self.commit))

    @property
    def dataset(self) -> str:
        """Its name in messages and the lock."""

        return f"adapter:{self.repo}@{self.commit}#{self.path}"

    def inputs(self) -> dict[str, Any]:
        """What the lock's inputs digest hashes of it (its name is the dataset's)."""

        return {
            "command": list(self.command),
            "args": list(self.args),
            "env": dict(sorted(self.env.items())),
            "sparse": list(self.sparse),
        }
