"""A killed rollout worker's remote env sessions are deleted, not left to the env server's 15-minute reaper.

These launch the real worker entrypoint against a real SDK ``Server``, so they are slow (each worker imports the
tinker/torch stack).  Two kills: the supervisor cancelling a straggler (SIGTERM, which the worker handles by
deleting its sessions first) and the worker's own hard-timeout alarm (SIGKILL, after which the supervisor
deletes what the worker listed).
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from types import SimpleNamespace

import pytest

import platoon.train.tinker  # noqa: F401  (import order: package before data processing)
from platoon.config_defs import RolloutConfig
from platoon.task import Task
from platoon.train.workers.protocol import PROTOCOL_VERSION, ConversionPolicy, ModelBinding, RolloutWorkerRequest
from platoon.train.workers.supervisor import RolloutWorkerSupervisor

STUB_DIR = os.path.dirname(os.path.abspath(__file__))

environments = pytest.importorskip("openreward.environments")
uvicorn = pytest.importorskip("uvicorn")


class HeldEnv(environments.Environment):
    """A session to hold open."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": "held"}]

    def get_prompt(self) -> list:
        return [environments.TextBlock(text="Hold.")]

    @environments.tool
    async def wait(self) -> environments.ToolOutput:
        """Do nothing."""
        return environments.ToolOutput(blocks=[environments.TextBlock(text="waited")])


@pytest.fixture(scope="module")
def ors_server() -> Iterator[SimpleNamespace]:
    ors = environments.Server([HeldEnv])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(ors.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("The ORS test server did not start")
        time.sleep(0.05)
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{port}", ors=ors)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


def _request(tmp_path, request_id: str, **overrides) -> RolloutWorkerRequest:
    scratch = tmp_path / "ipc"
    scratch.mkdir(parents=True, exist_ok=True)
    base = dict(
        protocol_version=PROTOCOL_VERSION,
        request_id=request_id,
        task_id="task",
        rollout_number=0,
        task=Task(goal="g", id="task"),
        rollout_config=asdict(RolloutConfig(output_dir=str(tmp_path / "out"))),
        rollout_fn_ref="worker_stub_rollouts:hold_a_remote_session",
        model=ModelBinding(
            binding_path=str(scratch / "model_binding.pkl"),
            bootstrap_blob=b"",
            follow_poll_seconds=5.0,
            model_name="",  # no model bound: no gateway, no tokenizer
            renderer_name="",
        ),
        conversion=ConversionPolicy(),
        response_path=str(scratch / f"{request_id}.res"),
        ready_path=str(scratch / f"{request_id}.ready"),
        startup_budget_seconds=600.0,
        rollout_budget_seconds=600.0,
    )
    base.update(overrides)
    return RolloutWorkerRequest(**base)


def _supervisor(tmp_path, server: SimpleNamespace) -> RolloutWorkerSupervisor:
    return RolloutWorkerSupervisor(
        scratch_dir=str(tmp_path / "ipc"),
        log_dir=str(tmp_path / "logs"),
        env_overrides={
            "PYTHONPATH": os.pathsep.join([STUB_DIR, os.environ.get("PYTHONPATH", "")]),
            "STUB_ORS_URL": server.url,
            "STUB_ORS_ENV": "HeldEnv",
            "HF_HUB_OFFLINE": "1",
            "OPENHANDS_SUPPRESS_BANNER": "1",
        },
        parent_grace_seconds=30.0,
    )


async def _until(condition, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        await asyncio.sleep(0.1)


def test_a_straggler_worker_deletes_its_env_sessions_when_cancelled(tmp_path, ors_server, monkeypatch):
    async def leave_them(self, request) -> None:  # only the worker's own SIGTERM handler may delete them
        return None

    monkeypatch.setattr(RolloutWorkerSupervisor, "_delete_orphaned_sessions", leave_them)
    supervisor = _supervisor(tmp_path, ors_server)

    async def scenario() -> None:
        running = asyncio.create_task(supervisor.run(_request(tmp_path, "straggler-0")))
        await _until(lambda: ors_server.ors._active_envs, 300)  # the worker holds its session
        running.cancel()
        started = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert time.monotonic() - started < 10  # the worker exited inside the termination grace
        await _until(lambda: not ors_server.ors._active_envs, 5)

    asyncio.run(scenario())


def test_a_worker_killed_by_its_own_alarm_has_its_env_sessions_deleted(tmp_path, ors_server):
    supervisor = _supervisor(tmp_path, ors_server)
    request = _request(tmp_path, "hung-0", rollout_budget_seconds=8.0)

    outcome = asyncio.run(supervisor.run(request))

    assert outcome.disposition == "hard_timeout" and outcome.response is None
    assert not ors_server.ors._active_envs
    assert not os.path.exists(request.sessions_path)
