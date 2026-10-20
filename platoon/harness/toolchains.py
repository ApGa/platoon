"""Harness toolchains: a CLI harness's interpreter and packages, built once into a directory and mounted read-only
into the sandboxes the harness runs in, so that nothing installs per rollout.

A spec names its toolchain in ``[install] toolchain``:

* an absolute path, used as it is: it must exist where the harness runs (the environment service's ``mounts`` can
  put it into a sandbox);
* a name, looked up under the toolchain root: ``toolchains`` under the cache root (:func:`platoon.site.cache_root`:
  the site profile's ``cache``, else ``~/.cache/platoon``), where ``platoon prepare`` builds it (``platoon toolchain
  build --root`` builds one elsewhere).

**Building** (:func:`build_toolchain`, ``platoon prepare`` or ``platoon toolchain build``) follows the spec's
``[install.recipe]``
(:class:`platoon.harness.spec.ToolchainRecipe`) once per site, at the directory the spec resolves its toolchain to
(``<root>/<name>``), so the interpreter and every script in it keep the paths they were installed with when the
directory is bound at the same path inside a sandbox:

* ``bin/python3`` and the rest of a uv-managed CPython (python-build-standalone, so task images need glibc), its
  standard library compiled to bytecode;
* the recipe's hash-locked package closure, installed into that interpreter without a virtual environment, also
  compiled to bytecode, since a read-only toolchain cannot cache bytecode at run time;
* ``share/tiktoken``, the tiktoken encodings, for recipes with the ``tiktoken`` cache;
* the files the recipe generates with the toolchain's own interpreter, such as a harness's settings;
* ``manifest.json`` (:class:`ToolchainManifest`), written last.

A directory with a manifest is a finished toolchain.  The manifest's ``recipe_sha256`` (:func:`recipe_sha256`)
identifies what the build installed from: the Python version, the lock, the caches and the generate scripts.
Building the toolchain again is a no-op while that matches the spec's recipe; otherwise, and for a manifest that an
older platoon wrote without it, or one that cannot be read, the build is an error until it is forced.  A directory
without a manifest is a build that failed or was killed, and the next build replaces it.  Builds of the same toolchain
take turns on a lock file next to it.  ``platoon prepare`` never forces one: a finished toolchain built from another
recipe may be the one a live run of another config, or of another platoon, stages its rollouts from (each rollout
stages the toolchain it finds under the root), so replacing it in place would change that run's next rollouts, and
break those that start while the build is half done.  Only ``platoon toolchain build --force`` replaces it, run while
no job uses it.

**A run never builds one.**  Building needs uv and network access to PyPI, takes minutes and writes a directory
that every sandbox of the site mounts, so it is a step of its own, once per site (``platoon prepare``); a spec's
options never render into ``[install]``, so no setting of a harness needs another build.  Before any rollout,
``platoon train``, ``eval`` and ``rollout`` refuse (exit 78) a config whose specs' named toolchains are not built under
the root they will use, with the ``platoon prepare`` command that builds them, or, for one built from another recipe,
the forced build that replaces it (:func:`toolchain_problem`, ``platoon.train.auto.check_harness_toolchains``).

**The manifest** of a named toolchain says where its files expect to live and what they are::

    {"name": "<name>", "build_path": "/where/it/was/built", "tree_hash": "sha256:<hex>", ...}

* ``build_path`` is where the toolchain's files expect to live, since interpreters and console scripts record
  absolute paths when they are installed.  A sandbox gets the toolchain bound read-only there, and ``{toolchain}``
  renders as that path, so nothing in it has to be relocatable.  Without a manifest, the toolchain directory's own
  resolved path is its build path.
* ``tree_hash`` is :func:`tree_hash` of the directory.  It names the staged copy, and the copy is checked against
  it.

**Staging.** Before a rollout uses a named toolchain, :func:`stage` copies it to node-local disk, once per node and
toolchain version: into a temporary directory under an exclusive ``flock``, checked against the tree hash, then
renamed into place, so no rollout ever sees a partial copy and concurrent rollouts copy it once.  Each staging that
takes the lock first removes the temporary directories that killed stagings of the same version left behind.  Staged
copies live under ``$PLATOON_TOOLCHAIN_CACHE``, else under ``platoon-toolchains-<uid>`` in node-local scratch
(:func:`platoon.utils.scratch.node_scratch`: ``$PLATOON_NODE_SCRATCH``, ``$PBS_LOCALDIR``, ``$SLURM_TMPDIR``,
``$TMPDIR``, else ``/tmp``).  :func:`prepare` does both and returns the paths a launch renders: the build path in a
task sandbox, the staged copy on the rollout host.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from platoon.sandbox.base import Mount
from platoon.site import cache_root as platoon_cache_root
from platoon.utils.scratch import node_scratch

if TYPE_CHECKING:
    from platoon.harness.spec import HarnessSpec, ToolchainRecipe

TOOLCHAIN_CACHE_ENV = "PLATOON_TOOLCHAIN_CACHE"
MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
HASH_PREFIX = "sha256:"
TIKTOKEN_DIR = "share/tiktoken"
# Derived files that Python writes beside its sources when it may: not part of a toolchain's identity.
BYTECODE_CACHE = "__pycache__"
# The encodings litellm and the OpenHands SDK count tokens with.
TIKTOKEN_ENCODINGS = ("cl100k_base", "o200k_base")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CHUNK_BYTES = 1024 * 1024

Log = Callable[[str], None]


class ToolchainError(RuntimeError):
    """A named toolchain cannot be found, read or staged (or, :class:`ToolchainBuildError`, built)."""


class ToolchainBuildError(ToolchainError):
    """A toolchain could not be built; nothing is left at its path."""


@dataclass(frozen=True)
class Toolchain:
    """A named toolchain under a toolchain root."""

    name: str
    path: Path  # the directory under the root
    build_path: str  # where its files expect to live: its mount target in a sandbox, and what {toolchain} renders
    tree_hash: str | None  # from the manifest; None without one


@dataclass(frozen=True)
class PreparedToolchain:
    """A staged toolchain: where a launch finds it, and the bind that puts it into a sandbox."""

    toolchain: Toolchain
    staged: Path  # the node-local copy

    @property
    def sandbox_path(self) -> str:
        """``{toolchain}`` in a task sandbox: the build path, where :meth:`mount` binds the copy."""
        return self.toolchain.build_path

    @property
    def host_path(self) -> str:
        """``{toolchain}`` on the rollout host: the build path when the toolchain is there, else the staged copy."""
        build = Path(self.toolchain.build_path)
        return str(build) if build.resolve() == self.toolchain.path.resolve() else str(self.staged)

    def mount(self) -> Mount:
        return Mount(source=str(self.staged), target=self.toolchain.build_path, read_only=True)


def is_named(toolchain: str) -> bool:
    """Whether ``[install] toolchain`` is a name under the toolchain root, rather than an absolute path."""

    return not toolchain.startswith("/")


def check_name(toolchain: str) -> None:
    """Raise ``ValueError`` unless ``toolchain`` is an absolute path or a valid name."""

    if is_named(toolchain) and not _NAME.match(toolchain):
        raise ValueError(f"toolchain {toolchain!r} must be an absolute path or a name matching {_NAME.pattern}")


def toolchain_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The toolchain root: ``root`` when given, else ``toolchains`` under the cache root
    (:func:`platoon.site.cache_root`: the site profile's ``cache``, else ``~/.cache/platoon``)."""

    return Path(root).expanduser() if root else platoon_cache_root() / "toolchains"


def resolve(name: str, *, root: str | os.PathLike[str] | None = None) -> Toolchain:
    """Find the toolchain ``name`` under the toolchain root (:func:`toolchain_root`) and read its manifest."""

    base = toolchain_root(root)
    path = base / name
    if not path.is_dir():
        raise ToolchainError(f"toolchain {name!r} is not under the toolchain root {base}: build it there first")
    build_path, digest = str(path.resolve()), None
    manifest = path / MANIFEST
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ToolchainError(f"cannot read {manifest}: {exc}") from exc
        if not isinstance(data, dict):
            raise ToolchainError(f"{manifest} must hold a JSON object")
        build_path = data.get("build_path") or build_path
        digest = data.get("tree_hash")
        if not isinstance(build_path, str) or not build_path.startswith("/"):
            raise ToolchainError(f"{manifest}: build_path must be an absolute path, not {build_path!r}")
        if digest is not None and (not isinstance(digest, str) or not digest.startswith(HASH_PREFIX)):
            raise ToolchainError(f"{manifest}: tree_hash must be 'sha256:<hex>', not {digest!r}")
    return Toolchain(name=name, path=path, build_path=build_path, tree_hash=digest)


def tree_digest(directory: str | os.PathLike[str], *, exclude: tuple[str, ...] = (MANIFEST,)) -> tuple[str, int, int]:
    """``(sha256:<hex>, files, bytes)`` of a directory tree, ``exclude`` (top-level names) left out.

    The digest is the SHA-256 of one line per entry, in order of POSIX relative path, each ending in a newline:
    ``d\\0<path>`` for a directory, ``l\\0<path>\\0<target>`` for a symbolic link, and
    ``f\\0<path>\\0<sha256 of the content>\\0<x or ->`` for a file (``x``: executable).  A copy that keeps symbolic
    links and executable bits (``cp -a``) hashes the same.  Bytecode caches (``__pycache__``) are left out: Python
    writes them whenever it imports from a writable tree (an import hook such as beartype's adds its own), and they
    derive from the sources.  ``files`` and ``bytes`` count the regular files the digest covers.
    """

    digest = hashlib.sha256()
    files = size = 0
    for relpath, path in _walk(Path(directory), exclude):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            line = f"l\0{relpath}\0{os.readlink(path)}"
        elif stat.S_ISDIR(info.st_mode):
            line = f"d\0{relpath}"
        else:
            executable = "x" if info.st_mode & stat.S_IXUSR else "-"
            line = f"f\0{relpath}\0{_file_sha256(path)}\0{executable}"
            files += 1
            size += info.st_size
        digest.update(line.encode("utf-8", "surrogateescape") + b"\n")
    return f"{HASH_PREFIX}{digest.hexdigest()}", files, size


def tree_hash(directory: str | os.PathLike[str]) -> str:
    """``sha256:<hex>`` over a directory tree, its top-level ``manifest.json`` left out: what a toolchain's manifest
    records (:func:`tree_digest`)."""

    return tree_digest(directory)[0]


def _walk(root: Path, exclude: tuple[str, ...]) -> list[tuple[str, Path]]:
    """Every entry under ``root`` but bytecode caches and the top-level ``exclude`` names, by relative path.
    Symbolic links to directories are listed, not entered."""

    entries: list[tuple[str, Path]] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        relative = base.relative_to(root)
        dirnames[:] = [name for name in dirnames if name != BYTECODE_CACHE]
        if relative == Path("."):
            dirnames[:] = [name for name in dirnames if name not in exclude]
            filenames = [name for name in filenames if name not in exclude]
        entries.extend(((relative / name).as_posix(), base / name) for name in [*dirnames, *filenames])
    return sorted(entries)


def _file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_root(cache: str | os.PathLike[str] | None = None) -> Path:
    """Where staged copies live: ``cache``, else ``$PLATOON_TOOLCHAIN_CACHE``, else ``platoon-toolchains-<uid>`` in
    node-local scratch (:func:`platoon.utils.scratch.node_scratch`), since a node's ``$TMPDIR`` may be small."""

    configured = cache if cache is not None else os.environ.get(TOOLCHAIN_CACHE_ENV)
    if configured:
        return Path(configured)
    return node_scratch() / f"platoon-toolchains-{os.getuid()}"


def stage(toolchain: Toolchain, *, cache: str | os.PathLike[str] | None = None) -> Path:
    """The node-local copy of ``toolchain``, made once per node and version (see the module docstring).

    Blocks while another process stages the same version; call it off the event loop.
    """

    root = cache_root(cache)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    version = (toolchain.tree_hash or _fingerprint(toolchain.path)).removeprefix(HASH_PREFIX)[:16]
    target = root / f"{toolchain.name}-{version}"
    if target.is_dir():
        return target
    with (root / f".{target.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _remove_partial_copies(root, target.name)
        if target.is_dir():  # another process staged it while this one waited
            return target
        partial = root / f".{target.name}.{uuid.uuid4().hex[:8]}.partial"
        try:
            shutil.copytree(toolchain.path, partial, symlinks=True)
            if toolchain.tree_hash is not None and (copied := tree_hash(partial)) != toolchain.tree_hash:
                raise ToolchainError(
                    f"toolchain {toolchain.name!r} at {toolchain.path} does not match its manifest: tree hash "
                    f"{copied}, manifest {toolchain.tree_hash}; rebuild it, or restore what changed"
                )
            os.rename(partial, target)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
    return target


def _remove_partial_copies(root: Path, name: str) -> None:
    """Remove the partial copies of the staged copy ``name`` (``.<name>.<hex>.partial``) that stagings killed while
    they copied left in ``root``.  Only a holder of that copy's lock makes them, so for the caller, which holds it,
    each one is stale.  The exact pattern leaves the partial copies of other toolchains and versions alone."""

    pattern = re.compile(rf"\.{re.escape(name)}\.[0-9a-f]+\.partial")
    for entry in root.iterdir():
        if pattern.fullmatch(entry.name):
            shutil.rmtree(entry, ignore_errors=True)


def _fingerprint(path: Path) -> str:
    """A version key for a toolchain without a tree hash: its files' paths, sizes and modification times."""

    digest = hashlib.sha256()
    for relative, entry in _walk(path, (MANIFEST,)):
        info = entry.lstat()
        digest.update(f"{relative}\0{info.st_mode}\0{info.st_size}\0{info.st_mtime_ns}\n".encode("utf-8", "replace"))
    return HASH_PREFIX + digest.hexdigest()


def prepare(
    name: str, *, root: str | os.PathLike[str] | None = None, cache: str | os.PathLike[str] | None = None
) -> PreparedToolchain:
    """Resolve the named toolchain and stage it (:func:`resolve`, :func:`stage`); blocks, so call it off the event
    loop.  Raises :class:`ToolchainError`."""

    toolchain = resolve(name, root=root)
    try:
        staged = stage(toolchain, cache=cache)
    except ToolchainError:
        raise
    except OSError as exc:
        raise ToolchainError(f"cannot stage toolchain {name!r} from {toolchain.path}: {exc}") from exc
    return PreparedToolchain(toolchain=toolchain, staged=staged)


@dataclass(frozen=True)
class ToolchainManifest:
    """What ``manifest.json`` records about a built toolchain."""

    name: str
    # The directory it was built in; sandboxes bind it at this same path.
    build_path: str
    # The harness spec it was built for (``name@version``).
    spec: str
    # The full CPython version, e.g. ``3.12.14``.
    python: str
    # sha256 of the recipe's lock file.
    lock_sha256: str
    # sha256 of what the recipe installs from (:func:`recipe_sha256`), which decides whether the toolchain is current;
    # None in a manifest written before platoon recorded it.  Keyword-only, so that it can default to None here.
    recipe_sha256: str | None = field(default=None, kw_only=True)
    # Installed distributions and their versions.
    packages: Mapping[str, str]
    caches: tuple[str, ...]
    generated: tuple[str, ...]
    # ``sha256:<hex>`` over every file but the manifest and bytecode caches (:func:`tree_hash`).
    tree_hash: str
    # The files and bytes the hash covers.
    files: int
    bytes: int
    built_at: str
    # Versions of what built it: ``platoon`` and ``uv``.
    builder: Mapping[str, str] = field(default_factory=dict)
    format: int = MANIFEST_FORMAT

    def to_json(self) -> str:
        data = asdict(self)
        data["packages"] = dict(sorted(self.packages.items()))
        return json.dumps(data, indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> ToolchainManifest:
        data = json.loads(text)
        data["caches"] = tuple(data.get("caches", ()))
        data["generated"] = tuple(data.get("generated", ()))
        return cls(**data)


def build_command(ref: str, root: str | os.PathLike[str] | None, *, force: bool = False) -> str:
    """The command that builds the toolchain of the spec ``ref`` (as a config names it) under ``root``."""

    words = ["platoon", "toolchain", "build", shlex.quote(ref), "--root", shlex.quote(os.fspath(root or "DIR"))]
    return " ".join([*words, "--force"] if force else words)


@dataclass(frozen=True)
class ToolchainState:
    """Why a harness spec's named toolchain cannot serve a run yet (:func:`toolchain_state`)."""

    spec: str  # the spec, name@version
    name: str  # the toolchain's name
    root: Path  # the toolchain root it is looked for under
    # not_built: no directory; unfinished: a directory without a manifest (a build that failed or was killed);
    # stale: built from another recipe, or by a platoon that recorded none; unreadable: its manifest cannot be read;
    # absent: not there, and the spec has no recipe to build it from.
    reason: Literal["not_built", "unfinished", "stale", "unreadable", "absent"]
    detail: str = ""  # what differs (stale), or why the manifest cannot be read (unreadable)

    @property
    def path(self) -> Path:
        return self.root / self.name

    @property
    def buildable(self) -> bool:
        """Whether building it from the spec's recipe makes it current: false when the spec has no recipe."""

        return self.reason != "absent"

    @property
    def needs_force(self) -> bool:
        """Whether making it current replaces a finished toolchain (one built from another recipe, or whose manifest
        cannot be read), which a live run of another config or platoon may be staging its rollouts from: only a forced
        build does that (``platoon toolchain build --force``), never ``platoon prepare``."""

        return self.reason in ("stale", "unreadable")

    def describe(self) -> str:
        """The problem, one line, without what fixes it."""

        what = f"harness {self.spec} runs from the toolchain {self.name}"
        if self.reason == "absent":
            return f"{what}, which is not under the toolchain root {self.root}: put it at {self.path}"
        if self.reason == "unreadable":
            return f"{what}, whose manifest at {self.path} cannot be read ({self.detail})"
        if self.reason == "stale":
            return f"{what}, but the toolchain at {self.path} was built from another recipe ({self.detail})"
        state = "an unfinished build is there" if self.reason == "unfinished" else "it is not built there"
        return f"{what}, which a run never builds, and {state} (toolchain root {self.root})"


def toolchain_state(spec: HarnessSpec, *, root: str | os.PathLike[str] | None = None) -> ToolchainState | None:
    """Why ``spec``'s named toolchain cannot serve a run, or None when it can (or when the spec has no named toolchain:
    an absolute path must exist where the harness runs, which only the rollout can see).

    The toolchain is found under the toolchain root (:func:`toolchain_root`: ``root``, else ``toolchains`` in the
    site profile's cache, else ``~/.cache/platoon/toolchains``).  It serves a run once it is built there: it has a
    manifest (:func:`read_manifest`), which a build writes last, and for a spec with a recipe, one that records that
    recipe (:func:`recipe_sha256`).  A spec without a recipe needs only the directory.
    """

    name = spec.install.toolchain
    if name is None or not is_named(name):
        return None
    base = toolchain_root(root)
    path = base / name
    recipe = spec.install.recipe
    if recipe is None:
        return None if path.is_dir() else ToolchainState(spec.ref, name, base, "absent")
    try:
        manifest = read_manifest(path)
    except (OSError, ValueError, TypeError) as exc:
        return ToolchainState(spec.ref, name, base, "unreadable", str(exc))
    if manifest is None:
        return ToolchainState(spec.ref, name, base, "unfinished" if path.exists() else "not_built")
    if manifest.recipe_sha256 != recipe_sha256(recipe):
        changes = _recipe_changes(manifest, recipe, _file_sha256(recipe.lock))
        return ToolchainState(spec.ref, name, base, "stale", changes)
    return None


def toolchain_problem(
    spec: HarnessSpec, *, ref: str | None = None, root: str | os.PathLike[str] | None = None
) -> str | None:
    """Why ``spec``'s named toolchain cannot serve a run (:func:`toolchain_state`), with what fixes it: ``platoon
    prepare`` on the run's config, or the ``platoon toolchain build`` command; for a toolchain that only a forced build
    replaces (:attr:`ToolchainState.needs_force`), that command alone.  None when it can serve.  ``ref`` is the spec as
    the config names it, for the command."""

    state = toolchain_state(spec, root=root)
    if state is None:
        return None
    if state.reason == "absent":
        return state.describe()
    if state.needs_force:
        command = build_command(ref or spec.ref, state.root, force=True)
        return (
            f"{state.describe()}: a run of another config or platoon may be running it, so platoon prepare leaves it "
            f"as it is; rebuild it in place, while no job uses it, with: {command}"
        )
    command = build_command(ref or spec.ref, state.root)
    return f"{state.describe()}: build it once per site with `platoon prepare` on the run's config, or with: {command}"


def toolchain_dir(spec: HarnessSpec, root: str | os.PathLike[str] | None = None) -> Path:
    """Where ``spec``'s toolchain lives: its absolute path, or its name under the toolchain root
    (:func:`toolchain_root`)."""

    from platoon.harness.spec import HarnessSpecError, resolve_toolchain

    if spec.install.toolchain is None:
        raise HarnessSpecError(f"harness {spec.ref} has no [install] toolchain")
    return Path(resolve_toolchain(spec.install.toolchain, None if root is None else os.fspath(root)))


def read_manifest(directory: str | os.PathLike[str]) -> ToolchainManifest | None:
    """The manifest of a finished toolchain; None when the directory has none (not built, or not finished)."""

    path = Path(directory) / MANIFEST
    try:
        return ToolchainManifest.from_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def recipe_sha256(recipe: ToolchainRecipe) -> str:
    """The identity of a toolchain recipe, which a manifest records and a build compares: the SHA-256 of everything
    the recipe installs from, as one JSON document with sorted keys.

    That is the Python version as the recipe writes it, the lock file's sha256, the caches, and each generated file's
    path inside the toolchain with its script's sha256.  A changed settings generator makes another recipe; a spec
    moved elsewhere with its files does not.
    """

    identity = {
        "python": recipe.python,
        "lock_sha256": _file_sha256(recipe.lock),
        "caches": sorted(set(recipe.cache)),
        "generate": {path: _file_sha256(script) for path, script in recipe.generate.items()},
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


def build_toolchain(
    spec: HarnessSpec,
    *,
    root: str | os.PathLike[str] | None = None,
    uv: str | None = None,
    force: bool = False,
    log: Log = print,
) -> ToolchainManifest:
    """Build ``spec``'s toolchain from its ``[install.recipe]`` and return its manifest (see the module docstring).

    A finished toolchain whose manifest records this recipe (:func:`recipe_sha256`) is returned as it is, unless
    ``force``; one built from another recipe, by an older platoon that recorded none, or whose manifest cannot be
    read, is an error that says what differs, unless ``force``, which rebuilds it.  ``uv`` is the uv executable
    (default: ``uv`` on the ``PATH``).  Raises :class:`ToolchainBuildError` when a step fails, after removing the
    partial directory, and ``HarnessSpecError`` for a spec without a recipe.
    """

    return _build_toolchain(spec, root=root, uv=uv, force=force, log=log)[0]


def ensure_toolchain(
    spec: HarnessSpec, *, root: str | os.PathLike[str] | None = None, uv: str | None = None, log: Log = print
) -> tuple[ToolchainManifest, bool]:
    """Make ``spec``'s toolchain current under the toolchain root, as ``platoon prepare`` does: a toolchain built from
    the spec's recipe is left as it is, and one that is not there, or whose build failed or was killed, is built.
    Returns the manifest and whether this call built it.

    A finished toolchain built from another recipe, by an older platoon, or whose manifest cannot be read is never
    replaced (:class:`ToolchainBuildError`): a live run of another config or platoon may be staging its rollouts from
    it (:attr:`ToolchainState.needs_force`).  Builds of one toolchain take turns on its lock file, so of two at once,
    on one node or on two that share the root's filesystem, the second finds the first's toolchain current and builds
    nothing.  Raises as :func:`build_toolchain` does.
    """

    return _build_toolchain(spec, root=root, uv=uv, force=False, log=log)


def _build_toolchain(
    spec: HarnessSpec, *, root: str | os.PathLike[str] | None, uv: str | None, force: bool, log: Log
) -> tuple[ToolchainManifest, bool]:
    """Build ``spec``'s toolchain unless one built from its recipe is there (``force``: in any case); one built from
    another recipe, or whose manifest cannot be read, is an error unless ``force``.  Returns the manifest and whether
    it was built here."""

    from platoon.harness.spec import HarnessSpecError

    recipe = spec.install.recipe
    if recipe is None:
        raise HarnessSpecError(f"harness {spec.ref} has no [install.recipe]; nothing to build")
    target = toolchain_dir(spec, root)
    uv = uv or shutil.which("uv")
    if uv is None:
        raise ToolchainBuildError("building a toolchain needs uv: install it, or pass its path with --uv")
    lock_sha256 = _file_sha256(recipe.lock)
    recipe_digest = recipe_sha256(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _build_lock(target, log):
        try:
            existing, unreadable = read_manifest(target), None
        except (OSError, ValueError, TypeError) as exc:
            existing, unreadable = None, f"has a manifest that cannot be read ({type(exc).__name__}: {exc})"
        if not force:
            if existing is not None and existing.recipe_sha256 == recipe_digest:
                log(f"{spec.ref}: {target} is already built ({existing.tree_hash})")
                return existing, False
            if existing is not None:
                stale: str | None = f"was built from another recipe ({_recipe_changes(existing, recipe, lock_sha256)})"
            else:
                stale = unreadable
            if stale is not None:
                raise ToolchainBuildError(f"{target} {stale}; pass --force to rebuild it")
        if target.exists() or target.is_symlink():
            why = "an unfinished build" if existing is None and unreadable is None else "a rebuild"
            log(f"{spec.ref}: removing {target} ({why})")
            _remove(target)
        started = time.monotonic()
        try:
            manifest = _build(spec, target, uv=uv, lock_sha256=lock_sha256, recipe_digest=recipe_digest, log=log)
        except BaseException as exc:
            _remove(target)
            if isinstance(exc, (ToolchainBuildError, KeyboardInterrupt, SystemExit)):
                raise
            raise ToolchainBuildError(f"building {target} failed: {type(exc).__name__}: {exc}") from exc
        log(
            f"{spec.ref}: built {target} in {time.monotonic() - started:.0f}s: Python {manifest.python}, "
            f"{len(manifest.packages)} packages, {manifest.files} files, {manifest.bytes / 1e6:.0f} MB, "
            f"{manifest.tree_hash}"
        )
        return manifest, True


def _recipe_changes(manifest: ToolchainManifest, recipe: ToolchainRecipe, lock_sha256: str) -> str:
    """How the recipe a toolchain was built from differs from ``recipe``, as far as the toolchain's manifest tells:
    it records the lock's hash, the installed Python, the caches and the generated paths, but not the generate
    scripts' contents or the Python version as the recipe wrote it."""

    changes = []
    if manifest.recipe_sha256 is None:
        changes.append("its manifest has no recipe_sha256: an older platoon built it")
    if manifest.lock_sha256 != lock_sha256:
        changes.append(f"lock sha256 {manifest.lock_sha256[:12]}, the recipe's {lock_sha256[:12]}")
    if not _same_python(manifest.python, recipe.python):
        changes.append(f"Python {manifest.python}, the recipe's {recipe.python}")
    if set(manifest.caches) != set(recipe.cache):
        changes.append(f"caches {sorted(set(manifest.caches))}, the recipe's {sorted(set(recipe.cache))}")
    if set(manifest.generated) != set(recipe.generate):
        changes.append(f"generated files {sorted(manifest.generated)}, the recipe's {sorted(recipe.generate)}")
    return "; ".join(changes) or "a generate script or the recipe's Python version string changed"


def _build(
    spec: HarnessSpec, target: Path, *, uv: str, lock_sha256: str, recipe_digest: str, log: Log
) -> ToolchainManifest:
    recipe = spec.install.recipe
    assert recipe is not None
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as scratch:
        installs, home = Path(scratch) / "python", Path(scratch) / "home"
        home.mkdir()
        log(f"{spec.ref}: installing CPython {recipe.python} with uv")
        _run([uv, "python", "install", recipe.python, "--install-dir", str(installs), "--no-bin",
              "--compile-bytecode"], log)  # fmt: skip
        found = [path for path in installs.glob("cpython-*") if path.is_dir() and not path.is_symlink()]
        if len(found) != 1 or not (found[0] / "bin" / "python3").exists():
            raise ToolchainBuildError(f"uv installed {[path.name for path in found]}, not one CPython with bin/python3")
        found[0].rename(target)  # the interpreter finds its standard library relative to itself
        python = target / "bin" / "python3"
        version = _run([str(python), "-I", "-c", "import platform; print(platform.python_version())"], log).strip()
        if not _same_python(version, recipe.python):
            raise ToolchainBuildError(f"uv installed CPython {version}, but the recipe asks for {recipe.python}")
        log(f"{spec.ref}: installing {recipe.lock.name} into CPython {version}")
        _run([uv, "pip", "install", "--python", str(python), "--break-system-packages", "--require-hashes",
              "--compile-bytecode", "--link-mode", "copy", "-r", str(recipe.lock)], log)  # fmt: skip
        env = _tool_env(target, home)
        if "tiktoken" in recipe.cache:
            log(f"{spec.ref}: caching the tiktoken encodings {list(TIKTOKEN_ENCODINGS)} in {TIKTOKEN_DIR}")
            load = f"import tiktoken\nfor name in {TIKTOKEN_ENCODINGS!r}:\n    tiktoken.get_encoding(name)"
            _run([str(python), "-I", "-c", load], log, env=env)
        for relpath, script in recipe.generate.items():
            log(f"{spec.ref}: generating {relpath} with {script.name}")
            _run([str(python), "-I", str(script), str(target / relpath)], log, env=env)
            if not (target / relpath).is_file():
                raise ToolchainBuildError(f"{script.name} did not write {relpath}")
    listing = (
        "import importlib.metadata as m, json; "
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}))"
    )
    packages = json.loads(_run([str(python), "-I", "-c", listing], log, quiet=True))
    digest, files, size = tree_digest(target)
    manifest = ToolchainManifest(
        name=target.name,
        build_path=str(target),
        spec=spec.ref,
        python=version,
        lock_sha256=lock_sha256,
        recipe_sha256=recipe_digest,
        packages=packages,
        caches=tuple(recipe.cache),
        generated=tuple(recipe.generate),
        tree_hash=digest,
        files=files,
        bytes=size,
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        builder={"platoon": _platoon_version(), "uv": _run([uv, "--version"], log, quiet=True).strip()},
    )
    partial = target / f"{MANIFEST}.partial"
    partial.write_text(manifest.to_json(), encoding="utf-8")
    partial.rename(target / MANIFEST)  # the manifest is the mark of a finished build
    return manifest


def _tool_env(target: Path, home: Path) -> dict[str, str]:
    """The environment of the build steps that run the toolchain's own interpreter: no user configuration, no
    network fetch of litellm's cost map, the tiktoken cache inside the toolchain and a throwaway home."""

    keep = ("PATH", "LANG", "LC_ALL", "TMPDIR", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY",
            "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")  # fmt: skip
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update(
        {
            "HOME": str(home),
            "TIKTOKEN_CACHE_DIR": str(target / TIKTOKEN_DIR),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "OPENHANDS_SUPPRESS_BANNER": "1",
        }
    )
    return env


def _run(argv: list[str], log: Log, *, env: Mapping[str, str] | None = None, quiet: bool = False) -> str:
    """Run one build step; its output goes to ``log`` (unless ``quiet``) and is returned.  Failure raises."""

    completed = subprocess.run(
        argv,
        env=None if env is None else dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if not quiet else subprocess.PIPE,
        text=True,
        check=False,
    )
    if not quiet and completed.stdout.strip():
        log(completed.stdout.rstrip())
    if completed.returncode != 0:
        detail = (completed.stdout + (completed.stderr or "")).strip()[-4000:]
        raise ToolchainBuildError(f"{' '.join(argv[:3])} ... exited with {completed.returncode}: {detail}")
    return completed.stdout


@contextlib.contextmanager
def _build_lock(target: Path, log: Log) -> Generator[None, None, None]:
    """Builds of one toolchain take turns on ``.<name>.lock`` next to it (an flock; skipped with a warning on a
    filesystem that refuses flock)."""

    with open(target.parent / f".{target.name}.lock", "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log(f"waiting for another build of {target} to finish")
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            log(f"warning: cannot lock {handle.name} ({exc}); do not build {target.name} twice at once")
        yield


def _same_python(version: str, wanted: str) -> bool:
    return version == wanted or version.startswith(wanted + ".")


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _platoon_version() -> str:
    from platoon.utils.config import distribution_version

    return distribution_version("platoon") or "unknown"


__all__ = [
    "MANIFEST",
    "TOOLCHAIN_CACHE_ENV",
    "PreparedToolchain",
    "Toolchain",
    "ToolchainBuildError",
    "ToolchainError",
    "ToolchainManifest",
    "ToolchainState",
    "build_command",
    "build_toolchain",
    "cache_root",
    "check_name",
    "ensure_toolchain",
    "is_named",
    "prepare",
    "read_manifest",
    "recipe_sha256",
    "resolve",
    "stage",
    "toolchain_dir",
    "toolchain_problem",
    "toolchain_root",
    "toolchain_state",
    "tree_digest",
    "tree_hash",
]
