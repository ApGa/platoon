"""Named harness toolchains: the tree hash, lookup under the toolchain root, node-local staging, and the spec keys
that name a toolchain and the harness's memory."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from platoon.harness import toolchains
from platoon.harness.spec import HarnessSpecError, parse_spec
from platoon.harness.toolchains import (
    MANIFEST,
    PreparedToolchain,
    Toolchain,
    ToolchainError,
    cache_root,
    prepare,
    resolve,
    stage,
    tree_hash,
)
from platoon.sandbox.base import Mount


def make_toolchain(root: Path, name: str = "tc-1.0", *, build_path: str | None = None, manifest: bool = True) -> Path:
    """A small toolchain: an executable, a module, a symlink and a byte-compiled file."""

    path = root / name
    (path / "bin").mkdir(parents=True)
    (path / "lib" / "__pycache__").mkdir(parents=True)
    (path / "bin" / "tool").write_text("#!/bin/sh\necho tool\n")
    (path / "bin" / "tool").chmod(0o755)
    (path / "lib" / "mod.py").write_text("VALUE = 1\n")
    (path / "lib" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00compiled")
    (path / "bin" / "alias").symlink_to("tool")
    if manifest:
        data = {"name": name, "build_path": build_path or str(path.resolve()), "tree_hash": tree_hash(path)}
        (path / MANIFEST).write_text(json.dumps(data))
    return path


# --------------------------------------------------------------------------- the tree hash


def test_the_tree_hash_covers_contents_modes_and_links_but_not_the_manifest_or_byte_code(tmp_path):
    path = make_toolchain(tmp_path, manifest=False)
    digest = tree_hash(path)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64

    (path / MANIFEST).write_text("{}")
    (path / "lib" / "__pycache__" / "other.pyc").write_bytes(b"more")
    assert tree_hash(path) == digest

    for change in (
        lambda: (path / "lib" / "mod.py").write_text("VALUE = 2\n"),
        lambda: (path / "bin" / "tool").chmod(0o644),
        lambda: ((path / "bin" / "alias").unlink(), (path / "bin" / "alias").symlink_to("mod")),
        lambda: (path / "lib" / "empty").mkdir(),
    ):
        change()
        assert tree_hash(path) != digest
        digest = tree_hash(path)


# --------------------------------------------------------------------------- lookup


def test_a_name_resolves_under_the_root_with_its_manifest(tmp_path, monkeypatch):
    path = make_toolchain(tmp_path / "root", build_path="/opt/toolchains/tc-1.0")

    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    assert toolchain == Toolchain(
        name="tc-1.0", path=path, build_path="/opt/toolchains/tc-1.0", tree_hash=tree_hash(path)
    )

    monkeypatch.setenv(toolchains.TOOLCHAIN_ROOT_ENV, str(tmp_path / "root"))
    assert resolve("tc-1.0") == toolchain  # the environment variable when the service sets no root


def test_a_toolchain_without_a_manifest_is_built_where_it_is(tmp_path):
    path = make_toolchain(tmp_path, manifest=False)

    assert resolve("tc-1.0", root=tmp_path) == Toolchain("tc-1.0", path, str(path.resolve()), None)


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ("{not json", "cannot read"),
        ("[]", "JSON object"),
        ('{"build_path": "relative/path"}', "absolute path"),
        ('{"tree_hash": "md5:abc"}', "sha256"),
    ],
)
def test_a_bad_manifest_is_refused(tmp_path, manifest, message):
    path = make_toolchain(tmp_path, manifest=False)
    (path / MANIFEST).write_text(manifest)

    with pytest.raises(ToolchainError, match=message):
        resolve("tc-1.0", root=tmp_path)


def test_a_name_is_looked_up_under_the_toolchain_root(tmp_path, monkeypatch):
    """Without a root or $PLATOON_TOOLCHAINS, a named toolchain is looked up under the cache root's toolchains,
    where platoon prepare builds it."""

    monkeypatch.delenv(toolchains.TOOLCHAIN_ROOT_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert toolchains.toolchain_root() == tmp_path / "home" / ".cache" / "platoon" / "toolchains"
    with pytest.raises(ToolchainError, match=f"is not under the toolchain root {tmp_path / 'home'}/.cache/platoon"):
        resolve("tc-1.0")
    with pytest.raises(ToolchainError, match="is not under the toolchain root"):
        resolve("tc-1.0", root=tmp_path)
    monkeypatch.setenv(toolchains.TOOLCHAIN_ROOT_ENV, str(tmp_path / "site"))
    assert toolchains.toolchain_root() == tmp_path / "site"
    assert toolchains.toolchain_root(tmp_path / "entry") == tmp_path / "entry"  # an entry's toolchain_root wins


# --------------------------------------------------------------------------- staging


def test_staging_copies_once_and_checks_the_tree_hash(tmp_path):
    path = make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    staged = stage(toolchain, cache=tmp_path / "cache")

    assert staged.parent == tmp_path / "cache" and staged.name.startswith("tc-1.0-")
    assert tree_hash(staged) == toolchain.tree_hash and os.access(staged / "bin" / "tool", os.X_OK)
    assert (staged / "bin" / "alias").is_symlink()
    marker = staged / "bin" / "tool"
    marker.write_text("#!/bin/sh\necho staged\n")
    assert stage(toolchain, cache=tmp_path / "cache") == staged  # already there: not copied again
    assert marker.read_text().endswith("staged\n")
    assert [entry.name for entry in (tmp_path / "cache").iterdir() if entry.name.endswith(".partial")] == []

    (path / "lib" / "mod.py").write_text("VALUE = 3\n")  # the toolchain changed after its manifest was written
    changed = Toolchain(toolchain.name, path, toolchain.build_path, toolchain.tree_hash)
    with pytest.raises(ToolchainError, match="does not match its manifest"):
        stage(changed, cache=tmp_path / "other-cache")
    assert [entry.name for entry in (tmp_path / "other-cache").iterdir() if not entry.name.endswith(".lock")] == []


def test_concurrent_stagings_copy_once(tmp_path, monkeypatch):
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    copies = []
    copytree = toolchains.shutil.copytree

    def counting_copytree(*args, **kwargs):
        if Path(args[0]) == toolchain.path:  # copytree calls itself for each subdirectory
            copies.append(args[0])
        return copytree(*args, **kwargs)

    monkeypatch.setattr(toolchains.shutil, "copytree", counting_copytree)
    results: list[Path] = []
    threads = [
        threading.Thread(target=lambda: results.append(stage(toolchain, cache=tmp_path / "cache"))) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(copies) == 1 and len(set(results)) == 1


def test_staging_removes_the_partial_copies_that_killed_stagings_left(tmp_path):
    make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")
    name = stage(toolchain, cache=tmp_path / "probe").name  # a staged copy has the same name in every cache
    cache = tmp_path / "cache"
    left = [cache / f".{name}.{suffix}.partial" for suffix in ("0badc0de", "5ca1ab1e")]
    for partial in left:
        (partial / "bin").mkdir(parents=True)
        (partial / "bin" / "tool").write_text("half copied")
    others = [
        cache / ".tc-2.0-0123456789abcdef.0badc0de.partial",  # another toolchain's, which its own stager may write
        cache / f".{name}.x-0123456789abcdef.0badc0de.partial",  # that of a toolchain named "<this copy's name>.x"
    ]
    for partial in others:
        partial.mkdir()

    staged = stage(toolchain, cache=cache)

    assert staged == cache / name and tree_hash(staged) == toolchain.tree_hash
    assert [partial for partial in left if partial.exists()] == []
    assert all(partial.is_dir() for partial in others)


def test_staged_copies_go_to_node_local_scratch_unless_a_cache_is_set(tmp_path, monkeypatch):
    for variable in (toolchains.TOOLCHAIN_CACHE_ENV, "PLATOON_NODE_SCRATCH", "PBS_LOCALDIR", "SLURM_TMPDIR"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("PBS_LOCALDIR", str(tmp_path / "job-local"))
    copies = f"platoon-toolchains-{os.getuid()}"

    assert cache_root() == tmp_path / "job-local" / copies  # the job's node-local directory before $TMPDIR

    monkeypatch.setenv("PLATOON_NODE_SCRATCH", str(tmp_path / "node"))
    make_toolchain(tmp_path / "root")
    assert prepare("tc-1.0", root=tmp_path / "root").staged.parent == tmp_path / "node" / copies

    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "cache"))
    assert cache_root() == tmp_path / "cache" and cache_root(tmp_path / "given") == tmp_path / "given"


def test_a_toolchain_without_a_tree_hash_is_versioned_by_its_files(tmp_path):
    path = make_toolchain(tmp_path / "root", manifest=False)
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    first = stage(toolchain, cache=tmp_path / "cache")
    (path / "lib" / "mod.py").write_text("VALUE = 22\n")
    second = stage(toolchain, cache=tmp_path / "cache")

    assert first != second and (second / "lib" / "mod.py").read_text() == "VALUE = 22\n"


def test_a_prepared_toolchain_mounts_its_copy_at_its_build_path(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "cache"))
    make_toolchain(tmp_path / "root", build_path="/opt/toolchains/tc-1.0")

    prepared = prepare("tc-1.0", root=tmp_path / "root")

    assert prepared.staged.parent == tmp_path / "cache"
    assert prepared.mount() == Mount(source=str(prepared.staged), target="/opt/toolchains/tc-1.0", read_only=True)
    assert prepared.sandbox_path == "/opt/toolchains/tc-1.0"
    assert prepared.host_path == str(prepared.staged)  # the build path is not on this host


def test_on_the_host_a_toolchain_runs_from_its_build_path_when_it_is_there(tmp_path):
    path = make_toolchain(tmp_path / "root")
    toolchain = resolve("tc-1.0", root=tmp_path / "root")

    prepared = PreparedToolchain(toolchain=toolchain, staged=tmp_path / "cache" / "tc-1.0-x")

    assert prepared.host_path == str(path.resolve())  # no relocation where nothing needs it


# --------------------------------------------------------------------------- spec keys


def _spec(**tables) -> dict:
    return {"name": "agent", "version": "1", "launch": {"argv": ["{toolchain}/bin/tool"]}, **tables}


def test_a_spec_names_its_toolchain_and_the_memory_its_processes_need():
    spec = parse_spec(_spec(install={"toolchain": "openhands-1.54.0"}, resources={"memory_mb": 2048}))

    assert spec.install.toolchain == "openhands-1.54.0" and spec.resources.memory_mb == 2048
    assert parse_spec(_spec(install={"toolchain": "/opt/tc"})).resources.memory_mb is None


@pytest.mark.parametrize(
    ("tables", "message"),
    [
        ({"install": {"toolchain": "../escape"}}, "absolute path or a name"),
        ({"install": {"toolchain": "has space"}}, "absolute path or a name"),
        ({"resources": {"memory_mb": 0}}, "positive number"),
        ({"resources": {"memory_mb": 1.5}}, "whole number"),
        ({"resources": {"cpus": 2}}, "unknown key"),
    ],
)
def test_bad_toolchain_names_and_resources_are_refused(tables, message):
    tables.setdefault("install", {"toolchain": "tc"})
    with pytest.raises(HarnessSpecError, match=message):
        parse_spec(_spec(**tables))
