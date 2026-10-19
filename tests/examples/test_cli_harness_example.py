"""examples/cli-harness: tiny-agent, a command-line agent wrapped by a TOML spec, plays number search.

Most tests run ``harness_rollout`` with the shared fakes: the agent's model calls go to a scripted
OpenAI-compatible server, and the environment's tools to the gateway's real MCP tool plane, so they need no
tokenizer. The last runs ``platoon rollout`` on the example's config with a scripted policy behind the real gateway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from harness_example_fakes import (
    EXAMPLES,
    VAL_0_GUESSES,
    bisect,
    load_tokenizer,
    serve_scripted_policy,
)
from harness_fakes import FakeChatServer, FakeGateway, FakeService, answer
from test_platoon_bash_mcp import ToolPlaneGateway, tool_call

from platoon import cli
from platoon.config_defs import RolloutConfig
from platoon.harness.rollout import run_harness_rollout
from platoon.harness.runner import HarnessRunner
from platoon.harness.spec import RenderContext, load_spec
from platoon.task import Task
from platoon.train.components import EnvironmentRoute

EXAMPLE = EXAMPLES / "cli-harness"
SPEC = EXAMPLE / "tiny-agent.toml"
CONFIG = EXAMPLE / "config.yaml"
NUMBER_SEARCH = "python://platoon.envs.number_search.env:NumberSearchEnvironment"


class RecordingGateway(ToolPlaneGateway):
    """The scripted model plus the real MCP tool plane; keeps the outcome the rollout graded."""

    outcome: Any = None

    def collect(self, root, *, task, outcome):
        self.outcome = outcome
        return super().collect(root, task=task, outcome=outcome)


class WrongKeyGateway(RecordingGateway):
    """A model server that refuses the agent's key, so tiny-agent exits 2 (model endpoint error)."""

    def open_agent(self, **kwargs):
        root = super().open_agent(**kwargs)
        self.model.api_key = "some other key"
        return root


def guesses(*numbers: int) -> list[dict[str, Any]]:
    return [tool_call("guess", {"number": number}, f"call-{index}") for index, number in enumerate(numbers)]


async def play(tmp_path: Path, gateway: Any, *, service: str = NUMBER_SEARCH, task_id: str = "val:0", **config):
    route = EnvironmentRoute(name="example", service=service, harness=str(SPEC), resolve_task=True)
    runner = HarnessRunner(logs_dir=tmp_path / "logs", state_root=tmp_path / "state", kill_grace_s=0.2)
    rollout_config = RolloutConfig(output_dir=str(tmp_path / "out"), return_dict=True, verbose=False, **config)
    collection = await run_harness_rollout(
        route.attach(Task(id=task_id, max_steps=config.get("max_steps"))),
        rollout_config,
        gateway=gateway,
        runner=runner,
    )
    (trajectory,) = collection["trajectories"].values()
    return trajectory


def test_the_spec_renders_the_launch_the_mcp_config_and_the_builtins_knob():
    spec = load_spec(SPEC)
    context = RenderContext(
        instruction="Guess the number.",
        base_url="http://127.0.0.1:8000",
        api_key="agent-key",
        model="policy",
        workdir="/run/work",
        state_dir="/run",
        mcp_url="http://127.0.0.1:8000/mcp",
    )

    launch = spec.render(context, disable_builtins=True)

    assert launch.argv == (
        *("python3", "/run/tiny_agent.py", "--model", "policy", "--task-file", "/run/task.md", "--max-turns", "20"),
        *("--no-shell", "--mcp-config", "/run/mcp.json"),
    )
    assert launch.env == {
        "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
        "OPENAI_API_KEY": "agent-key",
        "PLATOON_API_KEY": "agent-key",
    }
    assert launch.files["task.md"] == "Guess the number."
    assert json.loads(launch.files["mcp.json"]) == {
        "mcpServers": {
            "env": {"url": "http://127.0.0.1:8000/mcp", "headers": {"Authorization": "Bearer ${PLATOON_API_KEY}"}}
        }
    }
    assert set(launch.assets) == {"tiny_agent.py"}
    assert [spec.exit_status(code) for code in (0, 3, 2, 4, None)] == [
        "completed",
        "completed",
        "error",
        "error",
        "error",
    ]


@pytest.mark.asyncio
async def test_tiny_agent_plays_number_search_with_the_env_tools_over_mcp(tmp_path):
    gateway = RecordingGateway(FakeChatServer(guesses(*VAL_0_GUESSES)))

    trajectory = await play(tmp_path, gateway, max_steps=12)

    assert gateway.outcome.status == "scored" and trajectory["reward"] == 1.0
    harness = trajectory["misc"]["harness"]
    assert (harness["spec"], harness["placement"], harness["exit_code"]) == ("example-tiny-agent@0.1.0", "local", 0)
    assert trajectory["finish_message"] == "You guessed the number 447 correctly!"  # the FINAL ANSWER line
    assert [call["arguments"]["number"] for call in trajectory["misc"]["tool_calls"]] == VAL_0_GUESSES
    first = gateway.model.requests[0]
    # Run locally, the built-in shell is off (--no-shell): the model sees only the env's tools.
    assert [tool["function"]["name"] for tool in first["tools"]] == ["guess"]
    assert first["messages"] == [{"role": "user", "content": "Guess the correct number between 119 and 965."}]


@pytest.mark.asyncio
async def test_the_turn_limit_exits_3_which_the_spec_grades(tmp_path):
    gateway = RecordingGateway(FakeChatServer(guesses(1, 2, 3)))

    trajectory = await play(tmp_path, gateway, max_steps=2)

    assert len(gateway.model.requests) == 2  # --max-turns {max_steps}
    assert trajectory["misc"]["harness"]["exit_code"] == 3
    assert gateway.outcome.status == "scored" and trajectory["reward"] == 0.0


@pytest.mark.asyncio
async def test_a_model_endpoint_error_exits_2_an_untrainable_harness_crash(tmp_path):
    gateway = WrongKeyGateway(FakeChatServer(guesses(*VAL_0_GUESSES)))

    trajectory = await play(tmp_path, gateway, max_steps=12)

    assert trajectory["misc"]["harness"]["exit_code"] == 2
    assert (gateway.outcome.status, gateway.outcome.failure_kind) == ("error", "harness_crash")
    log = Path(trajectory["misc"]["harness"]["logs_path"]).read_text()
    assert "tiny-agent: model request failed" in log


@pytest.mark.asyncio
async def test_in_a_task_sandbox_tiny_agent_keeps_its_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(FakeService, "root", tmp_path / "sandboxes")
    monkeypatch.setattr(FakeService, "sessions", [])
    shell = tool_call("shell", {"command": "echo 42 > answer.txt"}, "call-0")
    gateway = FakeGateway(FakeChatServer([shell, answer("Wrote 42 to answer.txt.")]))

    trajectory = await play(tmp_path, gateway, service="harness_fakes:FakeService", task_id="train-0", max_steps=5)

    # The service's task has a sandbox, so placement auto runs tiny-agent in it, with its built-in shell.
    harness = trajectory["misc"]["harness"]
    assert (harness["placement"], harness["exit_code"]) == ("task_sandbox", 0)
    assert [tool["function"]["name"] for tool in gateway.server.requests[0]["tools"]] == ["shell"]
    assert trajectory["reward"] == 1.0 and trajectory["finish_message"] == "Wrote 42 to answer.txt."
    (session,) = FakeService.sessions
    assert (session.sandbox.workdir / "answer.txt").read_text().strip() == "42"


def test_harness_show_describes_the_spec(capsys):
    cli.main(["harness", "show", str(SPEC)])

    out = capsys.readouterr().out
    assert out.startswith("example-tiny-agent@0.1.0  (spec)\n")
    assert "built-in tools: ['shell']  (can be disabled)" in out
    assert "final message: stdout /^FINAL ANSWER: (.*)$/" in out
    assert "exit codes: {3: 'completed'}" in out


def test_platoon_rollout_runs_the_example_config(tmp_path, monkeypatch, capsys):
    environment = yaml.safe_load(CONFIG.read_text())["environments"][0]
    assert (CONFIG.parent / environment["harness"]).resolve() == SPEC
    monkeypatch.chdir(tmp_path)  # the config names the spec relative to itself: it runs from any directory
    sampler = serve_scripted_policy(monkeypatch, load_tokenizer(), bisect())
    run_dir = tmp_path / "run"

    cli.main(["rollout", str(CONFIG), "--task", "val:0", "--output-dir", str(run_dir)])

    out = capsys.readouterr().out
    assert "task val:0\nagents 1  steps 9  max depth 0  reward 1.0  outcome scored reward=1.0\n" in out
    collection = json.loads((run_dir / "trajectory_collection.json").read_text())
    (trajectory,) = collection["trajectories"].values()
    assert trajectory["misc"]["harness"]["placement"] == "local"
    assert trajectory["finish_message"] == "You guessed the number 447 correctly!"
    assert sampler.calls[0] == "Guess the correct number between 119 and 965."


def test_the_example_config_validates():
    assert cli.config_validate(CONFIG) == 0


def test_a_spec_that_puts_the_agent_key_in_argv_is_refused(tmp_path, capsys):
    bad = tmp_path / "leaky.toml"
    bad.write_text(SPEC.read_text().replace('"--model", "{model}",', '"--key", "{api_key}",'))

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["harness", "show", str(bad)])

    assert exit_info.value.code == cli.EXIT_CONFIG == 78
    assert "{api_key} may only be rendered into [launch.env]" in capsys.readouterr().err
