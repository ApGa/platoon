"""Behavior checks for the lightweight forked-proxy entrypoint."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "platoon/areal_proxy_rollout.py"


def _load_entrypoint(monkeypatch, *, upstream_main, reset):
    real_spec_from_file_location = importlib.util.spec_from_file_location
    target_spec = real_spec_from_file_location(
        "platoon_areal_proxy_rollout_entrypoint_test",
        ENTRYPOINT,
    )
    assert target_spec is not None and target_spec.loader is not None

    patch_calls: list[str] = []

    class PatchLoader:
        def create_module(self, _spec):
            return None

        def exec_module(self, module):
            module.apply_proxy_patches = lambda: patch_calls.append("patches")

    patch_spec = importlib.machinery.ModuleSpec(
        "platoon.train.areal.patches",
        PatchLoader(),
    )
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda _name, _path: patch_spec,
    )

    upstream = types.SimpleNamespace(main=upstream_main)
    name_resolve = types.SimpleNamespace(reset=reset)
    real_import_module = __import__("importlib").import_module

    def fake_import_module(name):
        if name == "areal.experimental.openai.proxy.proxy_rollout_server":
            return upstream
        if name == "areal.utils.name_resolve":
            return name_resolve
        return real_import_module(name)

    monkeypatch.setattr(__import__("importlib"), "import_module", fake_import_module)
    module = importlib.util.module_from_spec(target_spec)
    monkeypatch.setitem(sys.modules, target_spec.name, module)
    target_spec.loader.exec_module(module)
    assert patch_calls == ["patches"]
    return module


def test_proxy_entrypoint_resets_after_normal_return(monkeypatch):
    events: list[str] = []
    expected = object()

    def upstream_main():
        events.append("upstream")
        return expected

    module = _load_entrypoint(
        monkeypatch,
        upstream_main=upstream_main,
        reset=lambda: events.append("reset"),
    )

    assert module.main() is expected
    assert events == ["upstream", "reset"]


def test_proxy_entrypoint_resets_and_preserves_primary_failure(monkeypatch):
    events: list[str] = []
    primary = RuntimeError("upstream stopped")

    def upstream_main():
        events.append("upstream")
        raise primary

    module = _load_entrypoint(
        monkeypatch,
        upstream_main=upstream_main,
        reset=lambda: events.append("reset"),
    )

    with pytest.raises(RuntimeError) as caught:
        module.main()
    assert caught.value is primary
    assert events == ["upstream", "reset"]


def test_proxy_entrypoint_does_not_mask_primary_when_reset_also_fails(monkeypatch):
    primary = RuntimeError("upstream stopped")
    reset_error = OSError("record root unavailable")

    def upstream_main():
        raise primary

    def reset():
        raise reset_error

    module = _load_entrypoint(
        monkeypatch,
        upstream_main=upstream_main,
        reset=reset,
    )

    with pytest.raises(RuntimeError) as caught:
        module.main()
    assert caught.value is primary
    assert any("OSError: record root unavailable" in note for note in primary.__notes__)


def test_proxy_entrypoint_propagates_reset_failure_after_normal_return(monkeypatch):
    reset_error = OSError("record root unavailable")

    def reset():
        raise reset_error

    module = _load_entrypoint(
        monkeypatch,
        upstream_main=lambda: "complete",
        reset=reset,
    )

    with pytest.raises(OSError) as caught:
        module.main()
    assert caught.value is reset_error
