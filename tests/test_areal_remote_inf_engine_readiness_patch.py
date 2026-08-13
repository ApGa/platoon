"""Focused tests for fail-fast inference-server child readiness polling."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PATCHES_PATH = REPO_ROOT / "platoon/train/areal/patches.py"
logger = logging.getLogger(__name__)


def gethostip():
    return "node"


def find_free_ports(_count):
    return [1234]


def format_hostport(host, port):
    return f"{host}:{port}"


class LocalInfServerInfo(SimpleNamespace):
    pass


ray = SimpleNamespace(is_initialized=lambda: False)


def _load_patches_module():
    spec = importlib.util.spec_from_file_location(
        "platoon_areal_server_readiness_patches_test",
        PATCHES_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_remote_module():
    module = ModuleType("fake_remote_inf_engine")
    module.split_hostport = lambda address: address.rsplit(":", 1)
    module.format_hostport = format_hostport

    class RequestException(Exception):
        pass

    requests = SimpleNamespace(
        exceptions=SimpleNamespace(RequestException=RequestException),
        request=lambda *_args, **_kwargs: SimpleNamespace(status_code=503),
    )
    module.requests = requests
    module.gethostip = gethostip
    module.find_free_ports = find_free_ports
    module.LocalInfServerInfo = LocalInfServerInfo
    module.ray = ray
    module.logger = logger

    class RemoteInfEngine:
        def __init__(self, *, setup_timeout=5.0, healthy=False):
            self.config = SimpleNamespace(setup_timeout=setup_timeout)
            self.healthy = healthy
            self.original_calls = []
            self.health_calls = []
            self.shutdown_calls = []
            self.local_server_processes = []
            self.backend = SimpleNamespace(
                get_health_check_request=lambda: SimpleNamespace(method="GET", endpoint="/health", payload=None),
                launch_server=lambda _server_args: SequenceProcess(None),
            )

        def check_health(self, base_url):
            self.health_calls.append(base_url)
            return self.healthy

        def _wait_for_server(self, address: str, process=None):
            self.original_calls.append((address, process))
            return "original"

        def launch_server(self, server_args):
            server_args["host"] = gethostip()
            server_args["port"] = find_free_ports(1)[0]
            process = self.backend.launch_server(server_args)
            address = format_hostport(server_args["host"], server_args["port"])
            server_info = LocalInfServerInfo(
                host=server_args["host"],
                port=server_args["port"],
                process=process,
            )
            try:
                self._wait_for_server(address, process=process)
                self.local_server_processes.append(server_info)
                if ray.is_initialized():
                    return LocalInfServerInfo(
                        host=server_args["host"],
                        port=server_args["port"],
                        process=None,
                    )
                return server_info
            except TimeoutError:
                logger.warning("launch timed out")
                self._shutdown_one_server(server_info)
                raise

        def _shutdown_one_server(self, server_info):
            self.shutdown_calls.append(server_info)

    module.RemoteInfEngine = RemoteInfEngine
    module.__dict__.update(
        gethostip=module.gethostip,
        find_free_ports=module.find_free_ports,
        format_hostport=module.format_hostport,
        LocalInfServerInfo=module.LocalInfServerInfo,
        ray=module.ray,
        logger=module.logger,
    )
    return module


class SequenceProcess:
    def __init__(self, *returncodes):
        self.returncodes = list(returncodes)
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        if len(self.returncodes) > 1:
            return self.returncodes.pop(0)
        return self.returncodes[0]


def test_patch_preserves_signature_and_is_idempotent():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    original = remote.RemoteInfEngine._wait_for_server
    original_signature = inspect.signature(original)

    assert patches._patch_remote_inf_engine_server_child_exit(remote) is True
    patched = remote.RemoteInfEngine._wait_for_server
    assert inspect.signature(patched) == original_signature
    assert patched.__wrapped__ is original
    assert patches._patch_remote_inf_engine_server_child_exit(remote) is False
    assert remote.RemoteInfEngine._wait_for_server is patched


def test_no_process_delegates_to_upstream_unchanged():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine()

    assert engine._wait_for_server("node:1234") == "original"
    assert engine.original_calls == [("node:1234", None)]
    assert engine.health_calls == []


def test_running_healthy_process_returns_normally_without_sleep():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine(healthy=True)
    process = SequenceProcess(None)
    remote.requests.request = lambda *_args, **_kwargs: SimpleNamespace(status_code=200)

    assert engine._wait_for_server("node:1234", process=process) is None
    assert process.poll_calls == 1
    assert engine.health_calls == []


def test_child_exit_is_reported_before_health_poll_with_actionable_code():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine()

    with pytest.raises(
        ChildProcessError,
        match=r"node:1234: exit code 17.*Inspect this rollout worker's stdout/stderr",
    ):
        engine._wait_for_server("node:1234", process=SequenceProcess(17))
    assert engine.health_calls == []


def test_child_exit_during_health_probe_avoids_poll_sleep(monkeypatch):
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine(healthy=False)
    process = SequenceProcess(None, -3)
    monkeypatch.setattr(
        patches.time,
        "sleep",
        lambda _seconds: pytest.fail("must not sleep after observing child exit"),
    )

    with pytest.raises(ChildProcessError, match=r"exit code -3 \(signal 3/SIGQUIT\)"):
        engine._wait_for_server("node:1234", process=process)
    assert engine.health_calls == []


def test_health_probe_timeout_is_bounded_by_remaining_monotonic_deadline(
    monkeypatch,
):
    patches = _load_patches_module()
    remote = _fake_remote_module()
    observed_timeouts = []
    remote.requests.request = lambda *_args, **kwargs: (
        observed_timeouts.append(kwargs["timeout"]) or SimpleNamespace(status_code=503)
    )
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine(setup_timeout=0.25)
    process = SequenceProcess(None)
    monotonic_values = iter((10.0, 10.0, 10.1, 10.25))
    monkeypatch.setattr(patches.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(patches.time, "sleep", lambda _seconds: None)

    with pytest.raises(TimeoutError, match=r"^server launch failed$"):
        engine._wait_for_server("node:1234", process=process)

    assert observed_timeouts == pytest.approx([0.25])


def test_launch_server_cleans_up_child_exit_exactly_once(caplog):
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine()
    process = SequenceProcess(17)
    engine.backend.launch_server = lambda _server_args: process

    with pytest.raises(ChildProcessError, match="exit code 17"):
        engine.launch_server({})

    assert len(engine.shutdown_calls) == 1
    assert engine.shutdown_calls[0].process is process
    assert engine.local_server_processes == []
    assert "launch timed out" not in caplog.text


def test_live_subprocess_exit_code_is_observed():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine()
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"])
    assert process.wait(timeout=10) == 23

    with pytest.raises(ChildProcessError, match=r"exit code 23"):
        engine._wait_for_server("node:1234", process=process)


def test_live_child_that_never_becomes_healthy_preserves_timeout_contract():
    patches = _load_patches_module()
    remote = _fake_remote_module()
    patches._patch_remote_inf_engine_server_child_exit(remote)
    engine = remote.RemoteInfEngine(setup_timeout=0.0, healthy=False)
    process = SequenceProcess(None)

    with pytest.raises(TimeoutError, match=r"^server launch failed$"):
        engine._wait_for_server("node:1234", process=process)


def test_incompatible_upstream_signature_fails_closed():
    patches = _load_patches_module()
    remote = _fake_remote_module()

    def incompatible(self, address):
        return None

    remote.RemoteInfEngine._wait_for_server = incompatible
    with pytest.raises(RuntimeError, match="Unsupported AReaL.*_wait_for_server API"):
        patches._patch_remote_inf_engine_server_child_exit(remote)
