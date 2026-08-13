from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "pbs-scripts/openreward-controller-runtime-receipt.py"
COMMON = REPO_ROOT / "pbs-scripts/openreward-ta-swe-prealloc-common.sh"
MODULE = REPO_ROOT / "platoon/training_success_receipt.py"


def _load():
    spec = importlib.util.spec_from_file_location("controller_runtime_receipt_test", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def runtime(tmp_path: Path):
    cache = tmp_path / "cache"
    envs = cache / "envs"
    manifests = cache / "manifests"
    envs.mkdir(parents=True)
    manifests.mkdir()
    scratch = envs / "scratch"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(scratch)], check=True)
    source = "a" * 64
    base_python = Path(os.path.realpath(scratch / "bin/python"))
    python_root = base_python.parents[1]
    manifest_text = (
        "schema=test\n"
        f"source_sha256={source}\n"
        f"python_root={python_root}\n"
        f"python_executable_sha256={_sha(base_python)}\n"
    )
    key = hashlib.sha256(manifest_text.encode()).hexdigest()
    venv = envs / key
    scratch.rename(venv)
    (venv / ".platoon-controller-manifest").write_text(manifest_text)
    (venv / ".platoon-controller-ready").write_text("")
    (venv / ".build-complete").write_text("")
    (manifests / f"{key}.txt").write_text(manifest_text)
    site = venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/platoon"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    (site / "training_success_receipt.py").write_bytes(MODULE.read_bytes())
    for path in (
        venv / ".platoon-controller-manifest",
        venv / ".platoon-controller-ready",
        venv / ".build-complete",
        manifests / f"{key}.txt",
        site / "__init__.py",
        site / "training_success_receipt.py",
    ):
        path.chmod(0o400)
    (venv / "bin").chmod(0o500)
    venv.chmod(0o500)
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir(mode=0o700)
    yield cache.resolve(), key, source, venv.resolve(), receipt_dir.resolve()
    # The fixture deliberately models a non-writable published environment;
    # restore test-owned directories so pytest can remove its temporary root.
    for current, directories, _files in os.walk(cache):
        Path(current).chmod(0o700)
        for directory in directories:
            path = Path(current) / directory
            if not path.is_symlink():
                path.chmod(0o700)


def _args(cache: Path, key: str, source: str, receipt: Path, command: str) -> list[str]:
    result = [
        command,
        "--receipt",
        str(receipt),
        "--pbs-job-id",
        "123.pbs1",
        "--experiment",
        "experiment",
        "--trial",
        "trial",
        "--controller-cache-root",
        str(cache),
    ]
    if command == "publish":
        result += ["--source-sha256", source, "--controller-environment-key", key]
    return result


def test_publish_and_read_exact_controller_python(runtime, capsys: pytest.CaptureFixture[str]) -> None:
    helper = _load()
    cache, key, source, venv, root = runtime
    receipt = root / "controller-runtime.json"
    assert helper.main(_args(cache, key, source, receipt, "publish")) == 0
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    capsys.readouterr()
    assert helper.main(_args(cache, key, source, receipt, "read-controller-python")) == 0
    output = capsys.readouterr().out
    assert output == f"{venv / 'bin/python'}\n"
    value = json.loads(receipt.read_text())
    assert value["controller_environment_key"] == key
    assert value["training_success_receipt_sha256"] == _sha(Path(value["training_success_receipt_module"]))


def test_publish_requires_fresh_private_direct_receipt_parent(runtime, tmp_path: Path) -> None:
    helper = _load()
    cache, key, source, _venv, root = runtime
    stale = root / "stale.json"
    stale.write_text("stale")
    stale.chmod(0o600)
    with pytest.raises(RuntimeError, match="freshness"):
        helper.main(_args(cache, key, source, stale, "publish"))

    permissive = tmp_path / "permissive"
    permissive.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="parent is unsafe"):
        helper.main(_args(cache, key, source, permissive / "receipt.json", "publish"))

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    indirect = tmp_path / "indirect"
    indirect.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="path is not exact"):
        helper.main(_args(cache, key, source, indirect / "receipt.json", "publish"))


def test_runtime_rejects_indirect_or_out_of_keyed_cache_environment(runtime, tmp_path: Path) -> None:
    helper = _load()
    cache, key, source, venv, _root = runtime
    outside = tmp_path / "outside"
    venv.chmod(0o700)
    venv.rename(outside)
    (cache / "envs" / key).symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="directory is not direct"):
        helper._runtime_identity(cache, key, source)


@pytest.mark.parametrize("name", (".platoon-controller-ready", ".build-complete"))
def test_runtime_rejects_symlink_or_permissive_markers(runtime, name: str) -> None:
    helper = _load()
    cache, key, source, venv, _root = runtime
    marker = venv / name
    venv.chmod(0o700)
    marker.unlink()
    marker.symlink_to(venv / ".platoon-controller-manifest")
    venv.chmod(0o500)
    with pytest.raises(RuntimeError, match="not direct"):
        helper._runtime_identity(cache, key, source)

    venv.chmod(0o700)
    marker.unlink()
    marker.write_text("")
    marker.chmod(0o600)
    venv.chmod(0o500)
    with pytest.raises(RuntimeError, match="runtime file is writable"):
        helper._runtime_identity(cache, key, source)


def test_runtime_rejects_foreign_owned_metadata(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    cache, key, source, _venv, _root = runtime
    actual_uid = os.getuid()
    monkeypatch.setattr(helper.os, "getuid", lambda: actual_uid + 1)
    with pytest.raises(RuntimeError, match="user-owned"):
        helper._runtime_identity(cache, key, source)


def test_runtime_rejects_manifest_key_source_and_python_identity(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    cache, key, source, venv, _root = runtime
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        helper._runtime_identity(cache, key, "b" * 64)

    original_sha = helper._sha256
    expected_python = Path(os.path.realpath(venv / "bin/python"))

    def wrong_python_sha(path: Path) -> str:
        if path == expected_python:
            return "0" * 64
        return original_sha(path)

    monkeypatch.setattr(helper, "_sha256", wrong_python_sha)
    with pytest.raises(RuntimeError, match="Python executable digest mismatch"):
        helper._runtime_identity(cache, key, source)
    monkeypatch.setattr(helper, "_sha256", original_sha)

    venv.chmod(0o700)
    (venv / "bin").chmod(0o700)
    (venv / "bin/python").unlink()
    (venv / "bin/python").symlink_to("/bin/false")
    (venv / "bin").chmod(0o500)
    venv.chmod(0o500)
    with pytest.raises(RuntimeError, match="Python target mismatch"):
        helper._runtime_identity(cache, key, source)


def test_runtime_rejects_python_prefix_escape(runtime, monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _load()
    cache, key, source, venv, _root = runtime

    class Completed:
        stdout = json.dumps(
            {
                "prefix": "/tmp/escaped",
                "module": str(venv / "lib/python3.12/site-packages/platoon/training_success_receipt.py"),
            }
        )

    monkeypatch.setattr(helper.subprocess, "run", lambda *_args, **_kwargs: Completed())
    with pytest.raises(RuntimeError, match="prefix/module escaped"):
        helper._runtime_identity(cache, key, source)


@pytest.mark.parametrize("field", ("pbs_job_id", "experiment", "trial"))
def test_read_rejects_wrong_job_identity(runtime, field: str) -> None:
    helper = _load()
    cache, key, source, _venv, root = runtime
    receipt = root / "controller-runtime.json"
    helper.main(_args(cache, key, source, receipt, "publish"))
    arguments = _args(cache, key, source, receipt, "read-controller-python")
    flag = {"pbs_job_id": "--pbs-job-id", "experiment": "--experiment", "trial": "--trial"}[field]
    arguments[arguments.index(flag) + 1] = "wrong"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        helper.main(arguments)


@pytest.mark.parametrize("mutation", ("manifest", "source", "module", "ready", "venv_mode"))
def test_read_rejects_changed_runtime(runtime, mutation: str) -> None:
    helper = _load()
    cache, key, source, venv, root = runtime
    receipt = root / "controller-runtime.json"
    helper.main(_args(cache, key, source, receipt, "publish"))
    if mutation == "manifest":
        (venv / ".platoon-controller-manifest").chmod(0o600)
        (venv / ".platoon-controller-manifest").write_text("changed")
        (venv / ".platoon-controller-manifest").chmod(0o400)
    elif mutation == "source":
        value = json.loads(receipt.read_text())
        value["source_sha256"] = "b" * 64
        receipt.write_text(json.dumps(value))
        receipt.chmod(0o600)
    elif mutation == "module":
        module = Path(json.loads(receipt.read_text())["training_success_receipt_module"])
        module.chmod(0o600)
        module.write_text("changed")
        module.chmod(0o400)
    elif mutation == "ready":
        venv.chmod(0o700)
        (venv / ".platoon-controller-ready").unlink()
        venv.chmod(0o500)
    else:
        venv.chmod(0o700)
    with pytest.raises((OSError, RuntimeError, subprocess.CalledProcessError)):
        helper.main(_args(cache, key, source, receipt, "read-controller-python"))


def test_common_publishes_runtime_only_after_all_cleanup_passes() -> None:
    source = COMMON.read_text()
    cleanup = source[source.index("cleanup() {") : source.index("handle_signal()")]
    assert cleanup.index("request_all_service_shutdowns") < cleanup.index("service_cleanup_failed")
    assert cleanup.index("service_cleanup_failed") < cleanup.index('if [[ "${status}" -eq 0 ]]')
    assert cleanup.index('if [[ "${status}" -eq 0 ]]') < cleanup.index("controller runtime receipt")
    assert '--controller-cache-root "${OPENREWARD_CONTROLLER_ENV_CACHE_ROOT}"' in cleanup


def test_controller_runtime_helper_is_executable() -> None:
    assert stat.S_IMODE(HELPER.stat().st_mode) == 0o755
