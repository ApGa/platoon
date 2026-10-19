"""The openhands runner (``run_openhands.py``) turns the spec's options into the SDK agent's settings: reasoning replay,
the request timeout and retries, and the condenser.  The SDK lives only in the toolchain, so these tests load the
runner over a stand-in SDK that records how it is called; ``test_openhands_live.py`` runs the real one."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from platoon.harness.registry import get_harness_spec

RUNNER = Path(get_harness_spec("openhands").launch.assets["run_openhands.py"])


class LLM:
    """The SDK's LLM: its settings, as the runner gives them."""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings

    def model_copy(self, *, update: dict[str, Any] | None = None) -> LLM:
        copy = type(self)(**self.settings)
        copy.settings.update(update or {})
        return copy


class Recorded:
    """An SDK class whose instances keep their arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs


class Conversation(Recorded):
    """A conversation that finishes at once, as an agent that called finish first would."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        CONVERSATIONS.append(self)
        self.state = types.SimpleNamespace(events=[], execution_status="finished")

    def send_message(self, text: str) -> None:
        self.message = text

    def run(self) -> None:
        pass


CONVERSATIONS: list[Conversation] = []


@pytest.fixture
def runner(monkeypatch):
    """``run_openhands`` imported over the stand-in SDK, with the gateway's endpoint in the environment."""

    sdk = types.ModuleType("openhands.sdk")
    vars(sdk).update(
        LLM=LLM,
        Agent=Recorded,
        AgentContext=Recorded,
        Tool=Recorded,
        Conversation=Conversation,
        TextContent=type("TextContent", (), {}),
    )
    condenser = types.ModuleType("openhands.sdk.context.condenser")
    vars(condenser).update(LLMSummarizingCondenser=type("LLMSummarizingCondenser", (Recorded,), {}))
    modules = {
        "openhands": types.ModuleType("openhands"),
        "openhands.sdk": sdk,
        "openhands.sdk.context": types.ModuleType("openhands.sdk.context"),
        "openhands.sdk.context.condenser": condenser,
        "openhands.tools": types.ModuleType("openhands.tools"),
    }
    for tool in ("file_editor", "task_tracker", "terminal"):
        module = types.ModuleType(f"openhands.tools.{tool}")
        name = "".join(part.title() for part in tool.split("_")) + "Tool"
        setattr(module, name, type(name, (), {"name": tool}))
        modules[module.__name__] = module
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("LLM_MODEL", "openai/policy")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_API_KEY", "agent-key")
    spec = importlib.util.spec_from_file_location("run_openhands_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    CONVERSATIONS.clear()
    return module


def required(tmp_path: Path) -> list[str]:
    (tmp_path / "instruction.md").write_text("Write 42 to answer.txt.")
    return [
        "--instruction-file",
        str(tmp_path / "instruction.md"),
        "--max-iterations",
        "30",
        "--final-message-file",
        str(tmp_path / "final.txt"),
    ]


def test_the_defaults_are_the_training_settings(runner, tmp_path):
    args = runner.parse_args(required(tmp_path))

    assert runner.settings(args) == {
        "max_iterations": 30,
        "reasoning_replay": True,
        "llm_timeout": 3600,
        "num_retries": 0,
        "condenser": "none",
    }


def test_the_flags_take_what_the_spec_renders(runner, tmp_path):
    flags = [
        "--reasoning-replay",
        "false",
        "--llm-timeout",
        "300",
        "--num-retries",
        "5",
        "--condenser",
        "llm_summarizing",
    ]

    args = runner.parse_args(required(tmp_path) + flags)

    assert (args.reasoning_replay, args.llm_timeout, args.num_retries, args.condenser) == (
        False,
        300,
        5,
        "llm_summarizing",
    )


@pytest.mark.parametrize(
    "flags",
    [
        ["--reasoning-replay", "0"],
        ["--reasoning-replay", "False"],
        ["--condenser", "summarize"],
        ["--num-retries", "-1"],
        ["--llm-timeout", "0"],
        ["--no-reasoning-replay"],  # the old spelling, gone with platoon-openhands-eval
    ],
)
def test_other_values_are_refused(runner, tmp_path, flags):
    with pytest.raises(SystemExit):
        runner.parse_args(required(tmp_path) + flags)


def test_training_settings_build_a_replaying_agent_without_retries_or_condenser(runner):
    agent = runner.build_agent(3600)

    llm = agent.kwargs["llm"]
    assert type(llm) is runner.ReplayLLM
    assert llm.settings | {"api_key": "-"} == {
        "model": "openai/policy",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "-",
        "usage_id": "agent",
        "timeout": 3600,
        "num_retries": 0,
        "force_string_serializer": True,
    }
    assert llm.settings["api_key"] == "agent-key" and "LLM_API_KEY" not in os.environ  # not the terminal's
    assert agent.kwargs["condenser"] is None
    assert [tool.kwargs["name"] for tool in agent.kwargs["tools"]] == ["terminal", "file_editor", "task_tracker"]


def test_the_adapters_settings_build_a_plain_agent(runner):
    agent = runner.build_agent(300, replay=False, num_retries=5)

    llm = agent.kwargs["llm"]
    assert type(llm) is LLM and (llm.settings["timeout"], llm.settings["num_retries"]) == (300, 5)
    # The SDK serializes the messages as it does for Harbor's adapter, which forces no string serializer: replay's
    # one-string messages would render the system prompt's two parts with a newline between them.
    assert llm.settings["force_string_serializer"] is None


def test_the_condenser_summarizes_with_a_copy_of_the_agents_llm_and_no_call_kind(runner):
    agent = runner.build_agent(3600, condenser="llm_summarizing")

    condenser = agent.kwargs["condenser"]
    assert type(condenser).__name__ == "LLMSummarizingCondenser" and condenser.kwargs.keys() == {"llm"}  # SDK defaults
    summarizer, llm = condenser.kwargs["llm"], agent.kwargs["llm"]
    assert summarizer is not llm and summarizer.settings == {**llm.settings, "usage_id": "condenser"}
    assert "extra_headers" not in summarizer.settings  # unmarked: the agent's own policy calls (module docstring)


def test_main_reports_its_settings_first_and_runs_the_conversation_with_them(runner, tmp_path, capsys):
    code = runner.main(required(tmp_path) + ["--reasoning-replay", "false", "--num-retries", "2"])

    lines = capsys.readouterr().out.splitlines()
    assert code == runner.EXIT_FINISHED
    assert lines[0] == "openhands: settings " + json.dumps(
        {"max_iterations": 30, "reasoning_replay": False, "llm_timeout": 3600, "num_retries": 2, "condenser": "none"}
    )
    assert json.loads(lines[-1].removeprefix("openhands: "))["status"] == "finished"
    (conversation,) = CONVERSATIONS
    assert conversation.kwargs["max_iteration_per_run"] == 30 and conversation.message == "Write 42 to answer.txt."
    assert conversation.kwargs["agent"].kwargs["llm"].settings["num_retries"] == 2
    assert (tmp_path / "final.txt").read_text() == ""
