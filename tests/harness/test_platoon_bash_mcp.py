"""platoon-bash with env tools over the gateway's MCP endpoint, against a scripted OpenAI-compatible model."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import pytest_asyncio
from harness_fakes import FakeChatServer, answer, bash_call

from platoon.config_defs import RolloutConfig
from platoon.envs.service import Outcome, ToolResult, ToolSpec
from platoon.gateway.core import AgentSession, ModelGateway
from platoon.gateway.mcp import bind_agent_tools, mcp_url, tool_call_records
from platoon.gateway.server import GatewayServer
from platoon.harness.registry import get_harness_spec
from platoon.harness.rollout import run_harness_rollout
from platoon.harness.runner import HarnessRunner
from platoon.task import Task
from platoon.train.components import EnvironmentRoute


class StubRenderer:
    """The model is the scripted chat server; the gateway here only serves tools."""

    def build_generation_prompt(self, messages, **kwargs):
        raise AssertionError("no model calls go through this gateway")

    def parse_response(self, tokens, **kwargs):
        raise AssertionError("no model calls go through this gateway")

    def get_stop_sequences(self):
        return []


def make_gateway() -> ModelGateway:
    return ModelGateway(
        sampler=cast(Any, None), renderer=cast(Any, StubRenderer()), tokenizer=cast(Any, None), model_name="policy"
    )


def tool_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": function}],
    }


@dataclass
class VaultSession:
    """No sandbox; the code is only reachable through the env's tools."""

    task: Task
    instruction: str = "Open the vault: read the code with lookup, then submit it."
    sandbox: None = None
    submitted: str | None = None
    closed: bool = False

    async def tools(self) -> list[ToolSpec]:
        key = {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}
        code = {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
        return [ToolSpec("lookup", "Read a value.", key), ToolSpec("submit", "Submit the code; ends the task.", code)]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name == "lookup":
            return ToolResult.text("1234" if arguments.get("key") == "vault" else "nothing here")
        self.submitted = arguments.get("code")
        correct = self.submitted == "1234"
        return ToolResult(
            content=[{"type": "text", "text": "the vault opens" if correct else "wrong code"}],
            finished=True,
            reward=1.0 if correct else 0.0,
        )

    async def grade(self) -> Outcome:
        return Outcome(status="scored", rewards={"reward": 1.0 if self.submitted == "1234" else 0.0})

    async def close(self) -> None:
        self.closed = True


class VaultService:
    name = "vault"
    sessions: ClassVar[list[VaultSession]] = []

    async def list_tasks(self, split: str) -> list[str]:
        return ["vault-0"]

    async def get_task(self, task_id: str) -> Task:
        return Task(id=task_id, goal="Open the vault.")

    async def open(self, task: Task, *, rollout_id: str) -> VaultSession:
        session = VaultSession(task=task)
        VaultService.sessions.append(session)
        return session


def make_runner(tmp_path: Path) -> HarnessRunner:
    return HarnessRunner(logs_dir=tmp_path / "logs", state_root=tmp_path / "state", kill_grace_s=0.2)


@pytest_asyncio.fixture
async def tool_gateway():
    gateway = make_gateway()
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    yield gateway, server
    await server.stop()


SCRIPT = [
    tool_call("lookup", {"key": "vault"}, "call-1"),
    bash_call("echo checked", "call-2"),
    tool_call("submit", {"code": "1234"}, "call-3"),
    answer("never reached: submit finished the task"),
]


@pytest.mark.asyncio
async def test_platoon_bash_calls_mcp_tools_next_to_bash(tmp_path, tool_gateway):
    gateway, server = tool_gateway
    agent = gateway.open_agent(rollout_id="rollout-1", trajectory_id="traj-1", interactions={})
    session = VaultSession(task=Task(id="vault-0"))
    bind_agent_tools(gateway, agent, session)

    async with FakeChatServer(list(SCRIPT), api_key=agent.api_key) as model:
        result = await make_runner(tmp_path).run(
            get_harness_spec("platoon-bash"),
            session=cast(Any, session),
            agent=agent,
            base_url=model.base_url,
            placement="local",
            mcp_url=mcp_url(server.base_url),
        )

    # The submit result said finished, so platoon-bash stopped with it as its final message.
    assert result.exit_code == 0 and result.final_message == "the vault opens"
    assert len(model.requests) == 3 and session.submitted == "1234"
    first, second, third = model.requests
    assert [tool["function"]["name"] for tool in first["tools"]] == ["bash", "lookup", "submit"]
    assert first["tools"][1]["function"]["parameters"]["required"] == ["key"]
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call-1", "content": "1234"}
    assert third["messages"][-1]["content"].startswith("exit code 0\nchecked")
    records = tool_call_records(agent)
    assert [(r["name"], r["arguments"]) for r in records] == [
        ("lookup", {"key": "vault"}),
        ("submit", {"code": "1234"}),
    ]
    assert records[1]["finished"] is True and records[1]["reward"] == 1.0


@pytest.mark.asyncio
async def test_platoon_bash_fails_as_infra_when_the_mcp_server_is_down(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    spec = get_harness_spec("platoon-bash")
    async with FakeChatServer([answer("never")], api_key="agent-secret") as model:
        result = await make_runner(tmp_path).run(
            spec,
            session=cast(Any, VaultSession(task=Task(id="vault-0"))),
            agent=cast(Any, type("Agent", (), {"agent_id": "n-1", "api_key": "agent-secret"})()),
            base_url=model.base_url,
            placement="local",
            mcp_url=f"http://127.0.0.1:{port}/mcp",
        )

    assert result.exit_code == 4 and spec.exit_status(result.exit_code) == "error"
    assert model.requests == []
    assert "MCP server unavailable" in Path(result.logs_path or "").read_text()


@dataclass
class ToolPlaneGateway:
    """The rollout's gateway port: the scripted model for chat, a real gateway server for the tool plane."""

    model: FakeChatServer
    gateway: ModelGateway = field(default_factory=make_gateway)
    server: GatewayServer = field(init=False)
    root: AgentSession | None = None

    def __post_init__(self) -> None:
        self.server = GatewayServer(self.gateway, keepalive_seconds=0.05)

    async def start(self) -> str:
        await self.server.start(host="127.0.0.1")
        return await self.model.start()

    def open_agent(self, *, rollout_id: str, trajectory_id: str, sampling: Any = None) -> AgentSession:
        self.root = self.gateway.open_agent(
            rollout_id=rollout_id, trajectory_id=trajectory_id, interactions={}, sampling_defaults=sampling
        )
        self.model.api_key = self.root.api_key
        return self.root

    def serve_tools(self, agent: AgentSession, session: Any) -> str:
        bind_agent_tools(self.gateway, agent, session)
        return mcp_url(self.server.base_url)

    def collect(self, root: AgentSession, *, task: Task, outcome: Outcome) -> Any:
        from platoon.trajectory import Trajectory, TrajectoryCollection

        trajectory = Trajectory(id=root.trajectory_id, task=task, reward=outcome.reward or 0.0)
        trajectory.misc["tool_calls"] = tool_call_records(root)
        self.gateway.close_rollout(root.rollout_id)
        return TrajectoryCollection(id=root.rollout_id, trajectories={trajectory.id: trajectory})

    async def stop(self) -> None:
        await self.model.stop()
        await self.server.stop()


@pytest.mark.asyncio
async def test_harness_rollout_serves_the_env_tools_to_an_mcp_spec(tmp_path):
    VaultService.sessions.clear()
    route = EnvironmentRoute(name="vault", service="test_platoon_bash_mcp:VaultService", harness="platoon-bash")
    gateway = ToolPlaneGateway(FakeChatServer(list(SCRIPT)))
    config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False)

    collection = await run_harness_rollout(
        route.attach(Task(id="vault-0", goal="Open the vault.")), config, gateway=gateway, runner=make_runner(tmp_path)
    )

    (session,) = VaultService.sessions
    assert session.submitted == "1234" and session.closed
    assert isinstance(collection, dict)
    (trajectory,) = collection["trajectories"].values()
    assert trajectory["reward"] == 1.0
    assert trajectory["misc"]["harness"]["placement"] == "local"
    assert [call["name"] for call in trajectory["misc"]["tool_calls"]] == ["lookup", "submit"]
    assert gateway.root is not None and gateway.root.closed
