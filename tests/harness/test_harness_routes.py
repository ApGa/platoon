"""A CLI harness in a Harbor sandbox without a network reaches the gateway through the sandbox's route, and runs from
a named toolchain staged on the node and mounted at its build path.

The process provider stands in for a provider that enforces ``network: none`` (it declares the capability), and
runs the route's forwarder for real; the fake gateway listens on the rollout's Unix socket as the embedded one does.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pytest
from aiohttp import web
from harness_fakes import FakeChatServer, FakeGateway, answer, bash_call
from test_harness_rollout import rollout_config, runner

from platoon.harness import toolchains
from platoon.harness.rollout import HARNESS_MISC_KEY, run_harness_rollout
from platoon.harness.toolchains import MANIFEST, tree_hash
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

FIXTURE = Path(__file__).resolve().parent.parent / "envs" / "fixtures" / "harbor" / "hello-file"
PLATOON_BASH = Path(__file__).resolve().parents[2] / "platoon" / "harness" / "builtin" / "platoon_bash.py"
BUILD_PATH = "/opt/platoon-test-toolchains/tc-agent-1"


class RecordingProvider(ProcessProvider):
    """A process provider that declares it enforces ``network: none``, and records the specs it starts."""

    capabilities = frozenset({"snapshot", "network_none", "routes", "host_loopback"})
    started: ClassVar[list[Any]] = []

    async def start(self, spec):
        RecordingProvider.started.append(spec)
        return await super().start(spec)


class ForwarderKillingProvider(RecordingProvider):
    """Ends each sandbox's route forwarder once the sandbox has started, as a policy's ``pkill python`` in an earlier
    harness run would: the next harness reaches the gateway only if the runner starts the route again."""

    async def start(self, spec):
        sandbox = await super().start(spec)
        script = (
            'pid=$(cat /tmp/platoon-route/gateway.pid) && kill "$pid" && '
            'for i in $(seq 50); do kill -0 "$pid" 2>/dev/null || exit 0; sleep 0.1; done; exit 1'
        )
        killed = await sandbox.exec(["sh", "-c", script])
        assert killed.exit_code == 0, killed.stderr
        return sandbox


class UnixChatServer(FakeChatServer):
    """The fake model endpoint, also on a Unix socket; notes which transport each call came in on."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        super().__init__(replies)
        self.transports: list[str] = []

    async def serve_unix(self, path: str) -> None:
        assert self._runner is not None
        await web.UnixSite(self._runner, path).start()

    async def _chat(self, request: web.Request) -> web.Response:
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        self.transports.append("unix" if isinstance(sockname, str) else "tcp")
        return await super()._chat(request)


@dataclass
class UnixGateway(FakeGateway):
    sockets: list[tuple[str, str]] = field(default_factory=list)

    async def serve_unix(self, path: str, *, rollout_id: str) -> None:
        self.sockets.append((path, rollout_id))
        assert isinstance(self.server, UnixChatServer)
        await self.server.serve_unix(path)


@pytest.fixture(autouse=True)
def recorded_specs():
    RecordingProvider.started.clear()
    yield RecordingProvider.started
    RecordingProvider.started.clear()


def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "dataset" / "offline"
    shutil.copytree(FIXTURE, task_dir)
    toml = task_dir / "task.toml"
    toml.write_text(toml.read_text().replace("[environment]\n", '[environment]\nnetwork_mode = "no-network"\n'))
    return task_dir


def _toolchain_root(tmp_path: Path) -> Path:
    """A toolchain whose interpreter is this test's Python, built (as far as its manifest says) at BUILD_PATH."""

    root = tmp_path / "toolchains"
    toolchain = root / "tc-agent-1"
    (toolchain / "bin").mkdir(parents=True)
    (toolchain / "bin" / "python3").symlink_to(sys.executable)
    manifest = {"name": "tc-agent-1", "build_path": BUILD_PATH, "tree_hash": tree_hash(toolchain)}
    (toolchain / MANIFEST).write_text(json.dumps(manifest))
    return root


def _toolchain_spec(tmp_path: Path) -> str:
    """platoon-bash run with the toolchain's interpreter, needing 256 MB of its own."""

    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    shutil.copy(PLATOON_BASH, spec_dir / "platoon_bash.py")
    (spec_dir / "tc-agent.toml").write_text(
        """
name = "tc-agent"
version = "1"
max_steps = 5
[install]
toolchain = "tc-agent-1"
[resources]
memory_mb = 256
[launch]
argv = ["{toolchain}/bin/python3", "{state_dir}/platoon_bash.py", "--base-url", "{base_url}/v1",
        "--model", "{model}", "--instruction-file", "{state_dir}/instruction.md",
        "--final-message-file", "{state_dir}/final_message.txt", "--max-turns", "{max_steps}"]
env = { PLATOON_API_KEY = "{api_key}" }
files = { "instruction.md" = "{instruction}" }
assets = { "platoon_bash.py" = "platoon_bash.py" }
[final_message]
file = "{state_dir}/final_message.txt"
[exit_codes]
3 = "completed"
"""
    )
    return str(spec_dir / "tc-agent.toml")


async def _run(
    tmp_path: Path,
    gateway: FakeGateway,
    *,
    harness: str = "platoon-bash",
    placement: str = "auto",
    provider: type[ProcessProvider] = RecordingProvider,
    **service_kwargs,
) -> dict:
    task_dir = _task_dir(tmp_path)
    route = EnvironmentRoute(
        name="harbor",
        service="harbor",
        service_kwargs={
            "dataset_dir": str(task_dir.parent),
            "provider": f"{provider.__module__}:{provider.__qualname__}",
            "provider_options": {"root": str(tmp_path / "sandboxes")},
            **service_kwargs,
        },
        harness=harness,
        placement=placement,
        resolve_task=True,
    )
    collection = await run_harness_rollout(
        route.attach(Task(id=task_dir.name)), rollout_config(tmp_path), gateway=gateway, runner=runner(tmp_path)
    )
    assert isinstance(collection, dict)
    return collection


REPLIES = [bash_call("echo hello > hello.txt"), answer("done")]


@pytest.mark.asyncio
async def test_a_harness_in_a_sandbox_without_a_network_reaches_the_gateway_through_its_route(tmp_path):
    server = UnixChatServer(list(REPLIES))
    gateway = UnixGateway(server)

    collection = await _run(tmp_path, gateway)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and outcome.reward == 1.0
    assert server.transports == ["unix", "unix"]  # every call crossed the route to the rollout's socket
    ((socket_path, rollout_id),) = gateway.sockets
    (spec,) = RecordingProvider.started
    assert spec.network == "none" and spec.upstreams == {"gateway": f"unix://{socket_path}"}
    assert len(socket_path) < 108 and rollout_id not in socket_path  # short and opaque
    assert not Path(socket_path).parent.exists()  # the rollout removed its socket directory
    (root,) = collection["trajectories"].values()
    assert root["misc"][HARNESS_MISC_KEY]["placement"] == "task_sandbox"


@pytest.mark.asyncio
async def test_the_runner_starts_a_route_that_is_gone_again_before_it_launches_the_harness(tmp_path):
    server = UnixChatServer(list(REPLIES))
    gateway = UnixGateway(server)

    await _run(tmp_path, gateway, provider=ForwarderKillingProvider)

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and outcome.reward == 1.0, outcome  # not a crash after a dead route
    assert server.transports == ["unix", "unix"]


@pytest.mark.asyncio
async def test_a_named_toolchain_is_staged_and_mounted_at_its_build_path(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "toolchain-cache"))
    root = _toolchain_root(tmp_path)
    server = UnixChatServer([bash_call("echo hello > hello.txt"), answer("done")])
    gateway = UnixGateway(server)

    await _run(tmp_path, gateway, harness=_toolchain_spec(tmp_path), toolchain_root=str(root))

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored" and outcome.reward == 1.0, outcome
    (spec,) = RecordingProvider.started
    (staged,) = (tmp_path / "toolchain-cache").glob("tc-agent-1-*")
    assert [(m.source, m.target, m.read_only) for m in spec.mounts] == [(str(staged), BUILD_PATH, True)]
    assert spec.memory_mb == 512 + 256  # the task's limit plus the harness's own
    log = next((tmp_path / "logs").glob("*.log")).read_text()
    assert "Traceback" not in log


@pytest.mark.asyncio
async def test_on_the_rollout_host_a_named_toolchain_runs_from_its_staged_copy(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "toolchain-cache"))
    root = _toolchain_root(tmp_path)
    server = UnixChatServer([answer("done")])
    gateway = UnixGateway(server)

    await _run(tmp_path, gateway, harness=_toolchain_spec(tmp_path), placement="local", toolchain_root=str(root))

    ((_, outcome),) = gateway.collected
    assert outcome.status == "scored"  # graded: nothing was written, so a reward of 0
    assert server.transports == ["tcp"] and gateway.sockets == []  # on the host: loopback, no socket needed
    (spec,) = RecordingProvider.started
    assert spec.upstreams == {} and spec.mounts == [] and spec.memory_mb == 512


@pytest.mark.asyncio
async def test_a_toolchain_that_is_not_there_is_a_setup_error(tmp_path, monkeypatch):
    monkeypatch.setenv(toolchains.TOOLCHAIN_CACHE_ENV, str(tmp_path / "toolchain-cache"))
    server = UnixChatServer([answer("done")])
    gateway = UnixGateway(server)

    await _run(tmp_path, gateway, harness=_toolchain_spec(tmp_path), toolchain_root=str(tmp_path / "empty"))

    ((_, outcome),) = gateway.collected
    assert (outcome.status, outcome.failure_kind) == ("error", "harness_setup")
    assert "tc-agent-1" in outcome.reason and "not under the toolchain root" in outcome.reason
    assert server.requests == []
