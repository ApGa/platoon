from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_patches_module():
    # Importing ``platoon.train.areal`` applies every runtime patch and pulls in
    # the distributed training stack. Load the standalone patch module so these
    # policy tests remain isolated and CPU-only.
    name = "platoon_areal_rollout_pause_rpc_patch_test"
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "platoon/train/areal/patches.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Dispatcher:
    def __init__(self) -> None:
        self.paused = False

    def pause(self) -> None:
        self.paused = True


def _make_fake_rollout_controller():
    # Return a fresh class for each test so method patching cannot leak between
    # test cases through class-level state.
    class FakeRolloutController:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple, dict]] = []
            self.dispatcher = _Dispatcher()
            self.rpc_error: BaseException | None = None

        def _collective_rpc(self, method: str, *args, **kwargs):
            self.calls.append((method, args, kwargs))
            if self.rpc_error is not None:
                raise self.rpc_error
            return f"result:{method}"

        def pause(self):
            # Match AReaL's ordering and hard-coded timeout at the patch boundary.
            self.dispatcher.pause()
            return self._collective_rpc("pause", http_timeout=60.0)

    return FakeRolloutController


def test_pause_rpc_uses_one_long_attempt(monkeypatch):
    monkeypatch.delenv("PLATOON_AREAL_ROLLOUT_CONTROL_TIMEOUT_SECS", raising=False)
    patches = _load_patches_module()
    controller_cls = _make_fake_rollout_controller()
    assert patches._patch_rollout_controller_pause_rpc_policy(controller_cls) is True
    controller = controller_cls()

    result = controller._collective_rpc(
        "pause",
        "positional",
        http_timeout=60.0,
        max_retries=3,
        caller_value="preserved",
    )

    assert result == "result:pause"
    assert controller.calls == [
        (
            "pause",
            ("positional",),
            {
                "http_timeout": 300.0,
                "max_retries": 1,
                "caller_value": "preserved",
            },
        )
    ]


def test_pause_rpc_timeout_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("PLATOON_AREAL_ROLLOUT_CONTROL_TIMEOUT_SECS", "425.5")
    patches = _load_patches_module()
    controller_cls = _make_fake_rollout_controller()
    patches._patch_rollout_controller_pause_rpc_policy(controller_cls)
    controller = controller_cls()

    controller.pause()

    assert controller.calls == [("pause", (), {"http_timeout": 425.5, "max_retries": 1})]


def test_pause_rpc_failure_propagates_and_dispatcher_stays_paused(monkeypatch):
    monkeypatch.delenv("PLATOON_AREAL_ROLLOUT_CONTROL_TIMEOUT_SECS", raising=False)
    patches = _load_patches_module()
    controller_cls = _make_fake_rollout_controller()
    patches._patch_rollout_controller_pause_rpc_policy(controller_cls)
    controller = controller_cls()
    failure = RuntimeError("pause acknowledgement failed")
    controller.rpc_error = failure

    with pytest.raises(RuntimeError, match="pause acknowledgement failed") as exc_info:
        controller.pause()

    assert exc_info.value is failure
    assert controller.dispatcher.paused is True
    assert controller.calls == [("pause", (), {"http_timeout": 300.0, "max_retries": 1})]


def test_pause_rpc_patch_is_idempotent_and_non_pause_calls_are_unchanged(monkeypatch):
    monkeypatch.delenv("PLATOON_AREAL_ROLLOUT_CONTROL_TIMEOUT_SECS", raising=False)
    patches = _load_patches_module()
    controller_cls = _make_fake_rollout_controller()
    assert patches._patch_rollout_controller_pause_rpc_policy(controller_cls) is True
    patched_method = controller_cls._collective_rpc
    assert patches._patch_rollout_controller_pause_rpc_policy(controller_cls) is False
    assert controller_cls._collective_rpc is patched_method
    controller = controller_cls()

    result = controller._collective_rpc(
        "set_version",
        9,
        http_timeout=60.0,
        max_retries=7,
        marker=object(),
    )

    assert result == "result:set_version"
    method, args, kwargs = controller.calls[0]
    assert method == "set_version"
    assert args == (9,)
    assert kwargs["http_timeout"] == 60.0
    assert kwargs["max_retries"] == 7
    assert "marker" in kwargs
