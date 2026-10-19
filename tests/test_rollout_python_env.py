import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import pytest

from platoon.utils.rollout_python_env import isolated_rollout_python_environment


def _write_probe_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("platoon_overlay_probe/__init__.py", "VALUE = 42\n")
        wheel.writestr(
            "platoon_overlay_probe-0.0.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: platoon-overlay-probe\nVersion: 0.0.0\n",
        )
        wheel.writestr(
            "platoon_overlay_probe-0.0.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: platoon-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("platoon_overlay_probe-0.0.0.dist-info/RECORD", "")


def test_isolated_rollout_python_environment_routes_package_tools_to_overlay(tmp_path):
    original_executable = sys.executable
    original_path = os.environ.get("PATH")
    original_virtual_env = os.environ.get("VIRTUAL_ENV")
    original_uv_environment = os.environ.get("UV_PROJECT_ENVIRONMENT")

    wheel_path = tmp_path / "platoon_overlay_probe-0.0.0-py3-none-any.whl"
    _write_probe_wheel(wheel_path)

    with isolated_rollout_python_environment() as overlay:
        assert Path(sys.executable).parent == overlay / "bin"
        assert os.environ["VIRTUAL_ENV"] == str(overlay)
        assert os.environ["UV_PROJECT_ENVIRONMENT"] == str(overlay)
        assert Path(os.environ["PATH"].split(os.pathsep)[0]) == overlay / "bin"
        assert Path(shutil.which("pip", path=os.environ["PATH"]) or "") == overlay / "bin" / "pip"
        assert "uv pip" in (overlay / "bin" / "pip").read_text()
        subprocess.run(["uv", "pip", "install", str(wheel_path)], check=True, capture_output=True, text=True)
        importlib.invalidate_caches()
        probe = importlib.import_module("platoon_overlay_probe")
        assert probe.VALUE == 42
        assert Path(probe.__file__).is_relative_to(overlay)

    sys.modules.pop("platoon_overlay_probe", None)

    assert not overlay.exists()
    assert sys.executable == original_executable
    assert os.environ.get("PATH") == original_path
    assert os.environ.get("VIRTUAL_ENV") == original_virtual_env
    assert os.environ.get("UV_PROJECT_ENVIRONMENT") == original_uv_environment


def test_a_second_entry_in_one_process_is_refused_and_changes_nothing():
    """The overlay swaps sys.executable, sys.path and environment variables process-wide and puts back what it found,
    so it cannot nest: an inner entry would find the outer overlay, and exiting after the outer one would put back an
    overlay already removed."""

    original_executable = sys.executable
    with isolated_rollout_python_environment() as overlay:
        inside = (sys.executable, list(sys.path), dict(os.environ))
        overlays = set(overlay.parent.glob(f"platoon-rollout-{os.getpid()}-*"))
        with pytest.raises(RuntimeError, match="already in the rollout Python overlay"):
            with isolated_rollout_python_environment():
                pass
        assert (sys.executable, list(sys.path), dict(os.environ)) == inside
        assert set(overlay.parent.glob(f"platoon-rollout-{os.getpid()}-*")) == overlays  # the refused one left none

    assert sys.executable == original_executable and not overlay.exists()
    with isolated_rollout_python_environment() as again:  # one at a time: the process can enter another now
        assert Path(sys.executable).parent == again / "bin"
    assert sys.executable == original_executable


def test_of_two_rollouts_at_once_in_one_process_the_second_is_refused():
    """Two in-process rollouts at once, the first finishing first: had the second entered, it would have found the
    first's overlay, and exiting last would have put back that overlay, removed by then, leaving ``sys.executable``
    naming a missing interpreter for every later launch in the process."""

    original_executable = sys.executable
    events: list[str] = []

    async def two_at_once() -> None:
        first_inside = asyncio.Event()

        async def first() -> None:
            with isolated_rollout_python_environment():
                events.append("first entered")
                first_inside.set()
                await asyncio.sleep(0.05)
            events.append("first exited")

        async def second() -> None:
            await first_inside.wait()
            try:
                with isolated_rollout_python_environment():
                    events.append("second entered")
                    await asyncio.sleep(0.1)
            except RuntimeError:
                events.append("second refused")

        await asyncio.gather(first(), second())

    asyncio.run(two_at_once())
    assert events == ["first entered", "second refused", "first exited"]
    assert sys.executable == original_executable and Path(sys.executable).exists()


def test_an_entry_while_another_thread_is_in_the_overlay_is_refused_though_that_one_leaves_first(monkeypatch):
    """A second thread's entry is refused at once, before it reads what to put back: read while the first thread is in
    the overlay, that is the first one's overlay, which the first can remove before the second claims the process
    (here while the second makes its directory, as on a slow filesystem), and the second would put it back on exit."""

    original_executable = sys.executable
    make_directory = tempfile.mkdtemp
    first_in, second_waits, first_out = threading.Event(), threading.Event(), threading.Event()
    outcome: list[str] = []

    def mkdtemp(*args, **kwargs):
        if threading.current_thread().name == "second":  # the second makes its directory once the first has left
            second_waits.set()
            first_out.wait(30)
        return make_directory(*args, **kwargs)

    def first() -> None:
        with isolated_rollout_python_environment():
            first_in.set()
            second_waits.wait(30)
        first_out.set()

    def second() -> None:
        first_in.wait(30)
        try:
            with isolated_rollout_python_environment():
                outcome.append("entered")
        except RuntimeError as error:
            outcome.append(str(error))
        finally:
            second_waits.set()

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    threads = [threading.Thread(target=first, name="first"), threading.Thread(target=second, name="second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert not any(thread.is_alive() for thread in threads)
    assert len(outcome) == 1 and "already in the rollout Python overlay" in outcome[0]
    assert sys.executable == original_executable and Path(sys.executable).exists()
