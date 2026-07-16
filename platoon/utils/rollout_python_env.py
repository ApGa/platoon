"""Disposable Python-package overlay for untrusted rollout code.

Programmatic tool calling executes model-authored Python in the rollout worker.
Package-manager subprocesses launched by that code must never target the Python
environment that contains the trainer itself.  This module creates a very small
virtual-environment overlay: it has its own writable site-packages while a
``.pth`` file makes the read-only trainer packages available as a base layer.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import site
import stat
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

_MISSING = object()


def _restore_environment(name: str, value: str | object) -> None:
    if value is _MISSING:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)


def _write_uv_pip_shim(path: Path) -> None:
    path.write_text('#!/bin/sh\nexec uv pip "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _create_overlay(root: Path, original_sys_path: list[str]) -> tuple[Path, Path]:
    """Create the minimum files Python and uv require for a virtual environment."""
    bin_dir = root / "bin"
    site_packages = root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    bin_dir.mkdir(parents=True)
    site_packages.mkdir(parents=True)

    base_executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    overlay_python = bin_dir / "python"
    overlay_python.symlink_to(base_executable)
    (bin_dir / "python3").symlink_to("python")
    (bin_dir / f"python{sys.version_info.major}.{sys.version_info.minor}").symlink_to("python")

    (root / "pyvenv.cfg").write_text(
        f"home = {base_executable.parent}\n"
        "include-system-site-packages = false\n"
        f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n"
        f"executable = {base_executable}\n"
    )

    # A child Python (for example the OpenReward MCP bridge) sees the same
    # immutable packages as this worker. Empty sys.path entries mean cwd.
    base_paths: list[str] = []
    for entry in original_sys_path:
        resolved = os.getcwd() if entry == "" else entry
        if resolved and "\n" not in resolved and resolved not in base_paths:
            base_paths.append(resolved)
    (site_packages / "platoon_trainer_base.pth").write_text("".join(f"{entry}\n" for entry in base_paths))

    # These shims preserve the common `pip install ...` spelling while routing
    # installation through uv into UV_PROJECT_ENVIRONMENT below. There is no
    # pip package in the trainer environment to accidentally invoke.
    _write_uv_pip_shim(bin_dir / "pip")
    (bin_dir / "pip3").symlink_to("pip")
    (bin_dir / f"pip{sys.version_info.major}.{sys.version_info.minor}").symlink_to("pip")
    return overlay_python, site_packages


@contextlib.contextmanager
def isolated_rollout_python_environment() -> Iterator[Path]:
    """Route rollout-created packages to a disposable per-process overlay.

    The caller is expected to be a short-lived rollout subprocess. State is
    restored and the overlay is removed on normal exit; a killed worker leaves
    only a directory under the node-local temporary directory.
    """
    original_executable = sys.executable
    original_sys_path = list(sys.path)
    original_environment = {
        name: os.environ.get(name, _MISSING)
        for name in (
            "PATH",
            "PIP_REQUIRE_VIRTUALENV",
            "PYTHONNOUSERSITE",
            "UV_CACHE_DIR",
            "UV_PROJECT_ENVIRONMENT",
            "VIRTUAL_ENV",
        )
    }
    root = Path(tempfile.mkdtemp(prefix=f"platoon-rollout-{os.getpid()}-"))

    try:
        overlay_python, site_packages = _create_overlay(root, original_sys_path)
        original_bin = str(Path(original_executable).parent.resolve())
        path_entries = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry]
        filtered_path = [entry for entry in path_entries if str(Path(entry).resolve()) != original_bin]

        os.environ["VIRTUAL_ENV"] = str(root)
        os.environ["UV_PROJECT_ENVIRONMENT"] = str(root)
        if "UV_CACHE_DIR" not in os.environ:
            os.environ["UV_CACHE_DIR"] = str(root / "uv-cache")
        os.environ["PIP_REQUIRE_VIRTUALENV"] = "1"
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.environ["PATH"] = os.pathsep.join([str(root / "bin"), *filtered_path])
        sys.executable = str(overlay_python)

        # Put newly installed packages ahead of the immutable base in this
        # already-running interpreter. `site` also processes any package .pth
        # files that uv installs into the overlay.
        site.addsitedir(str(site_packages))
        sys.path.remove(str(site_packages))
        sys.path.insert(0, str(site_packages))
        yield root
    finally:
        sys.executable = original_executable
        sys.path[:] = original_sys_path
        for name, value in original_environment.items():
            _restore_environment(name, value)
        shutil.rmtree(root, ignore_errors=True)
