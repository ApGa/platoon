"""A site cache of a test's own: a site profile (``platoon.site``) whose ``cache`` is a directory the test chooses,
named in ``PLATOON_SITE`` until the test ends.  The Harbor cache is ``harbor`` in it, with a fetched dataset's locks
and default image maps (``harbor/images/<provider>.json``), and the harness toolchains are ``toolchains``: since
platoon 0.3.0 removed a Harbor entry's ``cache``, ``image_map`` and ``toolchain_root`` and the variables
``PLATOON_HARBOR_CACHE`` and ``PLATOON_TOOLCHAINS``, the site profile is what names them::

    def test_something(tmp_path, monkeypatch):
        harbor_cache = use_site_cache(monkeypatch, tmp_path / "cache")
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def use_site_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> Path:
    """Name a site profile whose ``cache`` is ``cache`` in ``PLATOON_SITE`` until the test ends, and return the Harbor
    cache in it, ``cache / "harbor"`` (the toolchain root is ``cache / "toolchains"``)."""

    cache = Path(cache).absolute()
    profile = cache.with_name(f"{cache.name}.site.yaml")
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(yaml.safe_dump({"cache": str(cache)}))
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    return cache / "harbor"
