"""Harbor's task content hash, computed by platoon (``harbor-packager-v1``)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from platoon.envs.harbor.hashing import IgnoreMatcher, content_hash, task_files


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _reference(root: Path, relatives: list[str]) -> str:
    """The rule written out: sha256 over ``relpath\\0sha256(file)\\n`` in POSIX path order."""

    outer = hashlib.sha256()
    for relative in sorted(relatives):
        inner = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        outer.update(f"{relative}\0{inner}\n".encode())
    return outer.hexdigest()


TASK = {
    "task.toml": "[task]\nname = 'x'\n",
    "instruction.md": "Do it.\n",
    "README.md": "readme\n",
    "trajectory.json": "{}\n",
    "notes.txt": "not part of the package\n",
    "inputs-provenance.json": "{}\n",
    "environment/Dockerfile": "FROM python:3.12-slim\n",
    "environment/input/a.csv": "x,y\n1,2\n",
    "environment/__pycache__/m.cpython-312.pyc": "bytecode",
    "environment/tool.pyc": "bytecode",
    "environment/.DS_Store": "finder",
    "tests/test.sh": "#!/bin/bash\n",
    "tests/backup.txt~": "editor",
    "solution/solve.sh": "echo\n",
    "steps/one/instruction.md": "step\n",
}


def test_hash_covers_the_package_files_with_the_default_ignores(tmp_path: Path):
    root = _write(tmp_path / "task", TASK)
    expected = [
        "README.md",
        "environment/Dockerfile",
        "environment/input/a.csv",
        "instruction.md",
        "solution/solve.sh",
        "steps/one/instruction.md",
        "task.toml",
        "tests/test.sh",
        "trajectory.json",
    ]
    assert task_files(root) == expected
    assert content_hash(root) == _reference(root, expected)


def test_a_task_gitignore_replaces_the_default_ignores(tmp_path: Path):
    root = _write(tmp_path / "task", {**TASK, ".gitignore": "*.csv\n!keep.csv\n/solution/\n"})
    (root / "environment" / "input" / "keep.csv").write_text("kept\n")
    files = task_files(root)
    assert "environment/input/a.csv" not in files and "environment/input/keep.csv" in files
    assert "solution/solve.sh" not in files
    assert "environment/tool.pyc" in files and "environment/__pycache__/m.cpython-312.pyc" in files  # no defaults
    assert ".gitignore" not in files  # it filters; Harbor does not package it
    assert content_hash(root) == _reference(root, files)


def test_one_byte_changes_the_hash(tmp_path: Path):
    root = _write(tmp_path / "task", TASK)
    before = content_hash(root)
    (root / "environment" / "input" / "a.csv").write_text("x,y\n1,3\n")
    assert content_hash(root) != before


@pytest.mark.parametrize(
    ("patterns", "ignored", "kept"),
    [
        (["*.log"], ["a.log", "x/y/b.log"], ["a.log.txt", "x/log"]),
        (["build/"], ["build/a", "x/build/b/c"], ["build", "x/builder/a"]),
        (["/top.txt"], ["top.txt"], ["x/top.txt"]),
        (["docs/*.md"], ["docs/a.md"], ["docs/x/a.md", "a/docs/a.md"]),
        (["**/cache"], ["cache", "a/b/cache", "a/cache/file"], ["cachex"]),
        (["a/**/z"], ["a/z", "a/b/z", "a/b/c/z"], ["b/a/z"]),
        (["logs/**"], ["logs/a", "logs/b/c"], ["logs"]),
        (["data[0-9].csv"], ["data1.csv"], ["dataa.csv"]),
        (["data[!0-9].csv"], ["dataa.csv"], ["data1.csv"]),
        (["?.txt"], ["a.txt"], ["ab.txt"]),
        (["*.txt", "!keep.txt"], ["a.txt"], ["keep.txt", "sub/keep.txt"]),
        (["# comment", "", "\\#hash"], ["#hash"], ["comment"]),
        (["trailing  "], ["trailing"], []),
    ],
)
def test_gitignore_patterns(patterns: list[str], ignored: list[str], kept: list[str]):
    matcher = IgnoreMatcher(patterns)
    for path in ignored:
        assert matcher.ignored(path), path
    for path in kept:
        assert not matcher.ignored(path), path


HARBOR_PYTHON = os.environ.get("PLATOON_TEST_HARBOR_PYTHON")


@pytest.mark.skipif(not HARBOR_PYTHON, reason="set PLATOON_TEST_HARBOR_PYTHON to a Python with harbor installed")
def test_matches_harbors_own_packager(tmp_path: Path):
    """Cross-check with ``harbor.publisher.packager.Packager.compute_content_hash`` (Harbor 0.24)."""

    cases = {
        "plain": TASK,
        "gitignore": {**TASK, ".gitignore": "*.csv\n!keep.csv\n/solution/\nbuild/\n**/tmp\n"},
        "nested": {**TASK, ".gitignore": "environment/input/\n!environment/input/a.csv\n"},
    }
    for name, files in cases.items():
        root = _write(tmp_path / name, files)
        script = (
            "import sys; from pathlib import Path; from harbor.publisher.packager import Packager; "
            "print(Packager.compute_content_hash(Path(sys.argv[1]))[0])"
        )
        theirs = subprocess.run(
            [HARBOR_PYTHON, "-I", "-c", script, str(root)], check=True, capture_output=True, text=True
        ).stdout.strip()
        assert content_hash(root) == theirs, name
