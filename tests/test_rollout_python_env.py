import importlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

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
