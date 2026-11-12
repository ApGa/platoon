"""The teams recipes' checks, which need the ``platoon-harness`` extra, and read no platoon-harness config of the
developer's: a team file loads over the user config and ``PLATOON_HARNESS_CONFIG``."""

import pytest

pytest.importorskip(
    "platoon_harness",
    reason="platoon-harness is not installed: install the platoon-harness extra (uv sync --all-packages --dev --extra "
    "tinker --extra ors --extra platoon-harness --extra platoon)",
)


@pytest.fixture(autouse=True)
def _no_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("PLATOON_HARNESS_CONFIG", raising=False)
