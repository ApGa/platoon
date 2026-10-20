"""``platoon toolchain build``: a toolchain is built where its spec resolves it, from the spec's recipe, finished by its
manifest, rebuilt only when asked or when its recipe changed, and never left half built.  A fake uv stands in for
the real one: its CPython is a wrapper around this interpreter, and its ``pip install`` records the lock it was
given."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path

import pytest
from site_cache import use_site_cache

from platoon import cli
from platoon.harness.spec import HarnessSpecError, load_spec
from platoon.harness.toolchains import (
    MANIFEST,
    ToolchainBuildError,
    ToolchainManifest,
    build_toolchain,
    read_manifest,
    recipe_sha256,
    toolchain_dir,
    tree_digest,
)

PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"
# This interpreter's full version, which the fake uv's CPython reports.
PYTHON_PATCH = f"{PYTHON}.{sys.version_info.micro}"

FAKE_UV = """#!{python}
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args == ["--version"]:
    print("uv 0.0.0 (fake)")
elif args[:2] == ["python", "install"]:
    installs = pathlib.Path(args[args.index("--install-dir") + 1])
    root = installs / "cpython-{version}.99-linux-x86_64-gnu"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "lib" / "os.py").write_text("# the standard library")
    python3 = root / "bin" / "python3"
    python3.write_text('#!/bin/sh\\nexec {python} "$@"\\n')
    python3.chmod(0o755)
    (installs / "cpython-{version}-linux-x86_64-gnu").symlink_to(root)
elif args[:2] == ["pip", "install"]:
    if os.environ.get("FAKE_UV_PIP_FAILS"):
        print("error: no matching distribution")
        sys.exit(2)
    prefix = pathlib.Path(args[args.index("--python") + 1]).parent.parent
    (prefix / "lib" / "installed.txt").write_text(pathlib.Path(args[args.index("-r") + 1]).read_text())
else:
    sys.exit(f"fake uv: unexpected {{args}}")
"""

GENERATOR = """import pathlib, sys
target = pathlib.Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text('{"made_by": "the toolchain"}')
"""
NEW_GENERATOR = GENERATOR.replace("the toolchain", "the new generator")


@pytest.fixture
def fake_uv(tmp_path, monkeypatch):
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(FAKE_UV.format(python=sys.executable, version=PYTHON))
    uv.chmod(0o755)
    monkeypatch.setenv("FAKE_UV_LOG", str(tmp_path / "uv.log"))
    return uv


def uv_calls(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "uv.log"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def write_spec(
    tmp_path: Path,
    *,
    lock: str = "pkg==1.0 --hash=sha256:00\n",
    generate: bool = True,
    python: str = PYTHON,
    cache: tuple[str, ...] = (),
    generator: str = GENERATOR,
    generated: str = "share/agent/settings.json",
) -> Path:
    specs = tmp_path / "specs"
    specs.mkdir(exist_ok=True)
    (specs / "agent.txt").write_text(lock)
    (specs / "make.py").write_text(generator)
    generates = f'\ngenerate = {{ "{generated}" = "make.py" }}' if generate else ""
    caches = f"\ncache = {json.dumps(list(cache))}" if cache else ""
    path = specs / "agent.toml"
    path.write_text(
        textwrap.dedent(
            f"""\
            name = "agent"
            version = "1.0"
            [install]
            toolchain = "agent-1.0"
            [install.recipe]
            python = "{python}"
            lock = "agent.txt"
            """
        )
        + generates
        + caches
        + '\n[launch]\nargv = ["{toolchain}/bin/python3", "-m", "agent"]\n'
    )
    return path


def build(tmp_path: Path, fake_uv: Path, **kwargs) -> ToolchainManifest:
    lines: list[str] = []
    manifest = build_toolchain(
        load_spec(tmp_path / "specs" / "agent.toml"),
        root=tmp_path / "toolchains",
        uv=str(fake_uv),
        log=lines.append,
        **kwargs,
    )
    return manifest


def test_a_toolchain_builds_at_its_resolved_path_and_finishes_with_its_manifest(tmp_path, fake_uv):
    spec_path = write_spec(tmp_path)

    manifest = build(tmp_path, fake_uv)

    target = tmp_path / "toolchains" / "agent-1.0"
    assert manifest == read_manifest(target) and manifest.build_path == str(target)
    assert manifest.recipe_sha256 == recipe_sha256(load_spec(spec_path).install.recipe)
    assert (manifest.name, manifest.spec, manifest.generated) == (
        "agent-1.0",
        "agent@1.0",
        ("share/agent/settings.json",),
    )
    assert manifest.python.startswith(PYTHON + ".") and manifest.builder["uv"] == "uv 0.0.0 (fake)"
    assert (target / "bin" / "python3").exists() and (target / "lib" / "os.py").exists()
    assert (target / "lib" / "installed.txt").read_text() == "pkg==1.0 --hash=sha256:00\n"
    assert json.loads((target / "share" / "agent" / "settings.json").read_text()) == {"made_by": "the toolchain"}
    assert manifest.tree_hash == tree_digest(target)[0] and manifest.files == tree_digest(target)[1]
    assert sorted(path.name for path in (tmp_path / "toolchains").iterdir()) == [".agent-1.0.lock", "agent-1.0"]
    python_install, pip_install = uv_calls(tmp_path)[:2]
    assert python_install[:3] == ["python", "install", PYTHON] and "--no-bin" in python_install
    assert pip_install[:2] == ["pip", "install"] and {"--require-hashes", "--break-system-packages"} <= set(pip_install)
    assert pip_install[pip_install.index("--python") + 1] == str(target / "bin" / "python3")


def test_building_a_built_toolchain_again_does_nothing(tmp_path, fake_uv):
    write_spec(tmp_path)
    first = build(tmp_path, fake_uv)
    calls = len(uv_calls(tmp_path))

    assert build(tmp_path, fake_uv) == first
    assert len(uv_calls(tmp_path)) == calls


def test_another_recipe_needs_force_to_rebuild(tmp_path, fake_uv):
    write_spec(tmp_path)
    first = build(tmp_path, fake_uv)
    write_spec(tmp_path, lock="pkg==2.0 --hash=sha256:11\n")

    with pytest.raises(ToolchainBuildError, match="built from another recipe .* pass --force"):
        build(tmp_path, fake_uv)
    rebuilt = build(tmp_path, fake_uv, force=True)

    assert rebuilt.lock_sha256 != first.lock_sha256 and rebuilt.tree_hash != first.tree_hash
    assert (tmp_path / "toolchains" / "agent-1.0" / "lib" / "installed.txt").read_text().startswith("pkg==2.0")


def test_the_recipe_identity_covers_what_the_recipe_installs_from_but_not_where_the_spec_is(tmp_path):
    def identity(spec_path: Path) -> str:
        recipe = load_spec(spec_path).install.recipe
        assert recipe is not None
        return recipe_sha256(recipe)

    spec_path = write_spec(tmp_path)
    base = identity(spec_path)
    shutil.copytree(tmp_path / "specs", tmp_path / "moved")
    assert identity(tmp_path / "moved" / "agent.toml") == base

    for change in (
        {"lock": "pkg==2.0 --hash=sha256:11\n"},
        {"python": PYTHON_PATCH},
        {"cache": ("tiktoken",)},
        {"generator": NEW_GENERATOR},
        {"generated": "share/agent/other.json"},
        {"generate": False},
    ):
        write_spec(tmp_path, **change)
        assert identity(spec_path) != base, change
    write_spec(tmp_path)
    assert identity(spec_path) == base


def test_a_changed_generate_script_is_another_recipe_which_force_rebuilds(tmp_path, fake_uv):
    write_spec(tmp_path)
    first = build(tmp_path, fake_uv)
    write_spec(tmp_path, generator=NEW_GENERATOR)

    with pytest.raises(ToolchainBuildError, match=r"built from another recipe \(a generate script .*\); pass --force"):
        build(tmp_path, fake_uv)
    rebuilt = build(tmp_path, fake_uv, force=True)

    settings = tmp_path / "toolchains" / "agent-1.0" / "share" / "agent" / "settings.json"
    assert json.loads(settings.read_text()) == {"made_by": "the new generator"}
    assert rebuilt.lock_sha256 == first.lock_sha256 and rebuilt.recipe_sha256 != first.recipe_sha256
    assert build(tmp_path, fake_uv) == rebuilt  # current again


@pytest.mark.parametrize(
    ("change", "difference"),
    [
        ({"lock": "pkg==2.0 --hash=sha256:11\n"}, r"lock sha256 [0-9a-f]{12}, the recipe's [0-9a-f]{12}"),
        ({"python": "3.99"}, re.escape(f"Python {PYTHON_PATCH}, the recipe's 3.99")),
        ({"cache": ("tiktoken",)}, re.escape("caches [], the recipe's ['tiktoken']")),
        (
            {"generated": "share/agent/other.json"},
            re.escape("generated files ['share/agent/settings.json'], the recipe's ['share/agent/other.json']"),
        ),
        ({"generate": False}, re.escape("generated files ['share/agent/settings.json'], the recipe's []")),
        # The same interpreter, now pinned to its patch version: only the recipe's text changed.
        ({"python": PYTHON_PATCH}, re.escape("a generate script or the recipe's Python version string changed")),
    ],
    ids=["lock", "python", "cache", "generated path", "no generated file", "python string"],
)
def test_a_toolchain_built_from_another_recipe_is_an_error_that_says_what_differs(
    tmp_path, fake_uv, change, difference
):
    write_spec(tmp_path)
    build(tmp_path, fake_uv)
    write_spec(tmp_path, **change)

    with pytest.raises(ToolchainBuildError, match=rf"built from another recipe \({difference}\); pass --force"):
        build(tmp_path, fake_uv)


def test_a_manifest_without_a_recipe_identity_reads_but_needs_force_to_rebuild(tmp_path, fake_uv):
    """A manifest that an older platoon wrote has no ``recipe_sha256``, so nothing says its toolchain is current."""

    spec_path = write_spec(tmp_path)
    build(tmp_path, fake_uv)
    path = tmp_path / "toolchains" / "agent-1.0" / MANIFEST
    older = json.loads(path.read_text())
    del older["recipe_sha256"]
    path.write_text(json.dumps(older))
    assert read_manifest(path.parent).recipe_sha256 is None

    with pytest.raises(ToolchainBuildError, match=r"\(its manifest has no recipe_sha256: .*\); pass --force"):
        build(tmp_path, fake_uv)
    rebuilt = build(tmp_path, fake_uv, force=True)

    assert rebuilt.recipe_sha256 == recipe_sha256(load_spec(spec_path).install.recipe)
    assert read_manifest(path.parent) == rebuilt


def test_a_recipe_may_pin_the_patch_version_which_the_build_checks(tmp_path, fake_uv):
    write_spec(tmp_path, python=PYTHON_PATCH)

    assert build(tmp_path, fake_uv).python == PYTHON_PATCH
    assert uv_calls(tmp_path)[0][:3] == ["python", "install", PYTHON_PATCH]

    other = f"{PYTHON}.{sys.version_info.micro + 1}"
    write_spec(tmp_path, python=other)
    with pytest.raises(ToolchainBuildError, match=re.escape(f"installed CPython {PYTHON_PATCH}, but the recipe asks")):
        build(tmp_path, fake_uv, force=True)
    assert not (tmp_path / "toolchains" / "agent-1.0").exists()


def test_an_unfinished_build_is_replaced(tmp_path, fake_uv):
    write_spec(tmp_path, generate=False)
    leftover = tmp_path / "toolchains" / "agent-1.0"
    leftover.mkdir(parents=True)
    (leftover / "half-written").write_text("")

    manifest = build(tmp_path, fake_uv)

    assert not (leftover / "half-written").exists() and (leftover / MANIFEST).exists() and manifest.generated == ()


def test_a_failed_build_leaves_nothing_behind(tmp_path, fake_uv, monkeypatch):
    write_spec(tmp_path)
    monkeypatch.setenv("FAKE_UV_PIP_FAILS", "1")

    with pytest.raises(ToolchainBuildError, match="exited with 2: error: no matching distribution"):
        build(tmp_path, fake_uv)

    assert [path.name for path in (tmp_path / "toolchains").iterdir()] == [".agent-1.0.lock"]


def test_a_build_needs_a_recipe_and_goes_under_the_toolchain_root(tmp_path, fake_uv, monkeypatch):
    spec_path = write_spec(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    default = tmp_path / "home" / ".cache" / "platoon" / "toolchains" / "agent-1.0"
    assert toolchain_dir(load_spec(spec_path)) == default  # the cache root's, without a root or a site profile
    use_site_cache(monkeypatch, tmp_path / "site")
    assert toolchain_dir(load_spec(spec_path)) == tmp_path / "site" / "toolchains" / "agent-1.0"
    plain = tmp_path / "plain.toml"
    plain.write_text('name = "plain"\nversion = "1"\n[launch]\nargv = ["true"]\n')
    with pytest.raises(HarnessSpecError, match="has no \\[install.recipe\\]"):
        build_toolchain(load_spec(plain), root=tmp_path, uv=str(fake_uv), log=lambda line: None)


def test_the_tree_hash_follows_contents_modes_and_links_but_not_the_manifest(tmp_path):
    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "tool").write_text("#!/bin/sh\n")
    (tree / "lib").mkdir()
    (tree / "lib" / "data").write_text("data")
    (tree / "bin" / "alias").symlink_to("tool")
    (tree / "lib-link").symlink_to("lib", target_is_directory=True)
    base, files, size = tree_digest(tree)
    assert (files, size) == (2, len("#!/bin/sh\n") + len("data"))

    copy = tmp_path / "copy"
    shutil.copytree(tree, copy, symlinks=True)
    (copy / MANIFEST).write_text("{}")
    assert tree_digest(copy)[0] == base

    def changed(mutate) -> str:
        target = tmp_path / f"mutated-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(tree, target, symlinks=True)
        mutate(target)
        return tree_digest(target)[0]

    assert changed(lambda t: (t / "lib" / "data").write_text("other")) != base
    assert changed(lambda t: (t / "bin" / "tool").chmod(0o755)) != base
    assert changed(lambda t: ((t / "bin" / "alias").unlink(), (t / "bin" / "alias").symlink_to("other"))) != base
    assert changed(lambda t: (t / "lib" / "new").write_text("")) != base
    # Bytecode that Python (or an import hook) writes when it imports from a writable tree does not count.
    assert (
        changed(lambda t: ((t / "lib" / "__pycache__").mkdir(), (t / "lib" / "__pycache__" / "m.pyc").write_text("")))
        == base
    )


def test_the_tree_hash_has_the_line_format_staging_checks(tmp_path):
    """The digest staging compares a toolchain's copy with: one line per directory, link and file, in path order."""

    import hashlib

    tree = tmp_path / "tree"
    (tree / "bin").mkdir(parents=True)
    (tree / "bin" / "tool").write_text("#!/bin/sh\n")
    (tree / "bin" / "tool").chmod(0o755)
    (tree / "bin" / "alias").symlink_to("tool")
    (tree / "lib").mkdir()
    (tree / "lib" / "data").write_text("data")
    (tree / "lib-link").symlink_to("lib", target_is_directory=True)
    (tree / "share").mkdir()
    (tree / "lib" / "__pycache__").mkdir()
    (tree / "lib" / "__pycache__" / "m.pyc").write_text("")
    (tree / MANIFEST).write_text("{}")

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    lines = [
        "d\0bin",
        "l\0bin/alias\0tool",
        f"f\0bin/tool\0{sha('#!/bin/sh' + chr(10))}\0x",
        "d\0lib",
        "l\0lib-link\0lib",
        f"f\0lib/data\0{sha('data')}\0-",
        "d\0share",
    ]
    expected = "sha256:" + hashlib.sha256("".join(line + "\n" for line in lines).encode()).hexdigest()

    assert tree_digest(tree) == (expected, 2, len("#!/bin/sh\n") + len("data"))


def test_the_cli_builds_each_spec_and_exits_nonzero_when_one_fails(tmp_path, fake_uv, monkeypatch, capsys):
    spec_path = write_spec(tmp_path)
    root = tmp_path / "toolchains"

    assert cli.main(["toolchain", "build", str(spec_path), "--root", str(root), "--uv", str(fake_uv)]) is None
    out = capsys.readouterr().out
    assert f"agent@1.0: {root / 'agent-1.0'} (sha256:" in out

    monkeypatch.setenv("FAKE_UV_PIP_FAILS", "1")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["toolchain", "build", str(spec_path), "--root", str(root), "--uv", str(fake_uv), "--force"])
    assert exit_info.value.code == 1
    assert "platoon: agent@1.0: " in (err := capsys.readouterr().err) and "exited with 2" in err
    assert not (root / "agent-1.0").exists()

    plain = tmp_path / "plain.toml"
    plain.write_text('name = "plain"\nversion = "1"\n[launch]\nargv = ["true"]\n')
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["toolchain", "build", str(plain), "--root", str(root)])
    assert exit_info.value.code == cli.EXIT_CONFIG
    assert "has no [install.recipe]" in capsys.readouterr().err


def test_the_build_lock_is_next_to_the_toolchain(tmp_path, fake_uv):
    write_spec(tmp_path)
    build(tmp_path, fake_uv)
    assert (tmp_path / "toolchains" / ".agent-1.0.lock").exists()
    assert os.access(tmp_path / "toolchains" / "agent-1.0" / "bin" / "python3", os.X_OK)
