"""Routing tests for Platoon's lightweight rollout RPC-server entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from test_areal_preallocated_pbs import _scheduler
from test_areal_preallocated_slurm import FakeSchedulingSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER_PATH = REPO_ROOT / "platoon/areal_rpc_server.py"


def _load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "platoon_areal_rpc_server_worker_patch_test",
        WRAPPER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_applies_patch_before_delegating(monkeypatch):
    wrapper = _load_wrapper()
    events = []
    monkeypatch.setattr(
        wrapper,
        "_apply_platoon_areal_rollout_worker_patches",
        lambda: events.append("patch"),
    )
    monkeypatch.setattr(
        wrapper.importlib,
        "import_module",
        lambda name: events.append(("import", name)) or SimpleNamespace(main=lambda: events.append("upstream-main")),
    )

    wrapper.main()
    assert events == [
        "patch",
        ("import", "areal.infra.rpc.rpc_server"),
        "upstream-main",
    ]


def test_pbs_rollout_default_routes_through_worker_wrapper(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)

    rollout_script = scheduler._rpc_worker_script(
        "rollout",
        FakeSchedulingSpec(cmd="python -m areal.infra.rpc.rpc_server"),
    )
    actor_script = scheduler._rpc_worker_script(
        "actor",
        FakeSchedulingSpec(cmd="python -m areal.infra.rpc.rpc_server"),
    )

    assert "exec python -m platoon.areal_rpc_server " in rollout_script
    assert "areal.infra.rpc.rpc_server" not in rollout_script
    assert "exec python -m areal.infra.rpc.rpc_server " in actor_script
    assert "platoon.areal_rpc_server" not in actor_script


def test_pbs_rollout_preserves_explicit_custom_rpc_command(monkeypatch, tmp_path):
    _module, scheduler, _processes = _scheduler(monkeypatch, tmp_path, node_count=1)

    script = scheduler._rpc_worker_script(
        "rollout",
        FakeSchedulingSpec(cmd="python -m custom.rpc_server --custom-flag"),
    )

    assert "exec python -m custom.rpc_server --custom-flag " in script
    assert "platoon.areal_rpc_server" not in script
