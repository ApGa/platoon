"""The openhands runner (``run_openhands.py``) turns the spec's options into the SDK agent's settings: reasoning replay,
the request timeout and retries, and the condenser with its token trigger, which counts prompts at the gateway.  The
SDK lives only in the toolchain, so these tests load the runner over a stand-in SDK that records how it is called,
and serve the gateway's token count from a real gateway; ``test_openhands_live.py`` runs the real SDK, the
condenser's own steps (a condensation, a context reset, a summary without text) included."""

from __future__ import annotations

import asyncio
import copy
import enum
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from platoon.harness.registry import get_harness_spec

RUNNER = Path(get_harness_spec("openhands").launch.assets["run_openhands.py"])
# The SDK's own estimate of any prompt, in the stand-in: what a count falls back to.
ESTIMATE = 7


class LLM:
    """The SDK's LLM: its settings, as the runner gives them, and what the condenser's count needs of it."""

    def __init__(self, **settings: Any) -> None:
        self.settings = settings
        self.base_url, self.api_key = settings.get("base_url"), settings.get("api_key")

    def model_copy(self, *, update: dict[str, Any] | None = None) -> LLM:
        copy = type(self)(**self.settings)
        copy.settings.update(update or {})
        return copy

    def format_messages_for_llm(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return messages  # the stand-in's messages are chat dicts already

    def should_mock_tool_calls(self, tools: list[dict[str, Any]]) -> bool:
        return False

    def get_token_count(self, messages, tools=None, add_security_risk_prediction=False) -> int:
        return ESTIMATE


class ToolDefinition:
    """An SDK tool, as the count reads it: its OpenAI function tool."""

    def __init__(self, tool: dict[str, Any]) -> None:
        self.tool = tool

    def to_openai_tool(self, *, add_security_risk_prediction: bool = False) -> dict[str, Any]:
        return self.tool


class Recorded:
    """An SDK class whose instances keep their arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs


class TextContent:
    def __init__(self, text: str = "") -> None:
        self.text = text


class Event:
    """An SDK event as the condenser's summary call reads it: the message it puts in the agent's prompt."""

    def __init__(self, source: str, role: str, *texts: str, name: str | None = None, calls=()) -> None:
        self.source = source
        self.message = types.SimpleNamespace(
            role=role, name=name, content=[TextContent(text) for text in texts], tool_calls=list(calls)
        )

    def to_llm_message(self):
        return self.message


class ActionEvent(Event):
    pass


class ObservationEvent(Event):
    pass


class CondensationSummaryEvent(Event):
    pass


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
        TextContent=TextContent,
    )
    modules = {
        "openhands": types.ModuleType("openhands"),
        "openhands.sdk": sdk,
        "openhands.sdk.context": types.ModuleType("openhands.sdk.context"),
        "openhands.tools": types.ModuleType("openhands.tools"),
    }
    # What the runner's condenser takes from the SDK: the condenser it extends and the pieces of it that it reuses.
    for name, values in {
        "openhands.sdk.context.condenser": {
            "LLMSummarizingCondenser": type("LLMSummarizingCondenser", (Recorded,), {})
        },
        "openhands.sdk.context.condenser.base": {
            "NoCondensationAvailableException": type("NoCondensation", (Exception,), {})
        },
        "openhands.sdk.context.condenser.llm_summarizing_condenser": {
            "Reason": enum.Enum("Reason", "REQUEST TOKENS EVENTS")
        },
        "openhands.sdk.event": {},
        "openhands.sdk.event.condenser": {
            "Condensation": Recorded,
            "CondensationSummaryEvent": CondensationSummaryEvent,
        },
        "openhands.sdk.utils": {"maybe_truncate": lambda content, truncate_after=None: content[:truncate_after]},
    }.items():
        module = types.ModuleType(name)
        vars(module).update(values)
        modules[name] = module
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
        "condenser_max_tokens": 0,
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
        "--condenser-max-tokens",
        "104857",
    ]

    args = runner.parse_args(required(tmp_path) + flags)

    assert (args.reasoning_replay, args.llm_timeout, args.num_retries, args.condenser) == (
        False,
        300,
        5,
        "llm_summarizing",
    )
    assert args.condenser_max_tokens == 104857


@pytest.mark.parametrize(
    "flags",
    [
        ["--reasoning-replay", "0"],
        ["--reasoning-replay", "False"],
        ["--condenser", "summarize"],
        ["--num-retries", "-1"],
        ["--llm-timeout", "0"],
        ["--condenser-max-tokens", "-1"],
        ["--condenser-max-tokens", "0.8"],  # a token count, not a share of the context window
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


def test_the_condenser_marks_its_summary_calls_aux_on_a_copy_of_the_agents_llm(runner):
    from platoon.harness.agent import CALL_KIND_HEADER

    agent = runner.build_agent(3600, condenser="llm_summarizing")

    condenser = agent.kwargs["condenser"]
    assert type(condenser) is runner.SummarizingCondenser and condenser.kwargs.keys() == {"llm"}  # the SDK's defaults
    summarizer, llm = condenser.kwargs["llm"], agent.kwargs["llm"]
    assert runner.CALL_KIND_HEADER == CALL_KIND_HEADER  # the runner imports nothing of platoon: its own copy
    assert summarizer is not llm and summarizer.settings == {
        **llm.settings,
        "usage_id": "condenser",
        "extra_headers": {CALL_KIND_HEADER: "aux"},  # the SDK sends an LLM's extra_headers with each of its calls
    }
    assert "extra_headers" not in llm.settings  # the agent's own calls carry no kind: policy calls


def test_main_reports_its_settings_first_and_runs_the_conversation_with_them(runner, tmp_path, capsys):
    code = runner.main(required(tmp_path) + ["--reasoning-replay", "false", "--num-retries", "2"])

    lines = capsys.readouterr().out.splitlines()
    assert code == runner.EXIT_FINISHED
    assert lines[0] == "openhands: settings " + json.dumps(
        {
            "max_iterations": 30,
            "reasoning_replay": False,
            "llm_timeout": 3600,
            "num_retries": 2,
            "condenser": "none",
            "condenser_max_tokens": 0,
        }
    )
    assert json.loads(lines[-1].removeprefix("openhands: "))["status"] == "finished"
    (conversation,) = CONVERSATIONS
    assert conversation.kwargs["max_iteration_per_run"] == 30 and conversation.message == "Write 42 to answer.txt."
    assert conversation.kwargs["agent"].kwargs["llm"].settings["num_retries"] == 2
    assert (tmp_path / "final.txt").read_text() == ""


# --------------------------------------------------------------------------- the token trigger


@pytest.mark.parametrize("replay", [True, False])
@pytest.mark.parametrize("condenser", ["none", "llm_summarizing"])
def test_without_a_token_trigger_nothing_counts_tokens(runner, replay, condenser):
    agent = runner.build_agent(3600, replay=replay, condenser=condenser)

    assert type(agent.kwargs["llm"]) is (runner.ReplayLLM if replay else LLM)  # the agent's LLM of main
    if condenser == "none":
        assert agent.kwargs["condenser"] is None
    else:
        assert agent.kwargs["condenser"].kwargs.keys() == {"llm"}  # the SDK's defaults


@pytest.mark.parametrize("condenser", ["none", "llm_summarizing"])
def test_a_token_trigger_runs_the_summarizing_condenser_with_it(runner, condenser, monkeypatch):
    agent = runner.build_agent(3600, condenser=condenser, condenser_max_tokens=104857)

    summarizer, llm = agent.kwargs["condenser"], agent.kwargs["llm"]
    assert type(summarizer) is runner.SummarizingCondenser
    assert summarizer.kwargs.keys() == {"llm", "max_tokens"} and summarizer.kwargs["max_tokens"] == 104857
    # The agent's LLM counts its prompts at the gateway, and replays as before; the summaries go through a copy, as
    # aux calls.
    assert type(llm) is runner.CountedReplayLLM and isinstance(llm, runner.ReplayLLM)
    assert type(summarizer.kwargs["llm"]) is runner.CountedReplayLLM
    aux = {runner.CALL_KIND_HEADER: "aux"}
    assert summarizer.kwargs["llm"].settings == {**llm.settings, "usage_id": "condenser", "extra_headers": aux}
    assert "extra_headers" not in llm.settings
    monkeypatch.setenv("LLM_API_KEY", "agent-key")  # the first agent took it out of the environment
    assert type(runner.build_agent(300, replay=False, condenser_max_tokens=8192).kwargs["llm"]) is runner.CountedLLM


def test_main_runs_the_agent_with_the_token_trigger_it_reports(runner, tmp_path, capsys):
    code = runner.main(required(tmp_path) + ["--condenser-max-tokens", "104857"])

    reported = json.loads(capsys.readouterr().out.splitlines()[0].removeprefix("openhands: settings "))
    assert code == runner.EXIT_FINISHED and reported["condenser_max_tokens"] == 104857
    assert reported["condenser"] == "llm_summarizing"  # what runs, though --condenser stayed "none"
    (conversation,) = CONVERSATIONS
    assert conversation.kwargs["agent"].kwargs["condenser"].kwargs["max_tokens"] == 104857


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {"type": "function", "function": {"name": "finish", "parameters": {"type": "object", "properties": {}}}},
]


def tool_call(call_id: str, command: str) -> dict[str, Any]:
    arguments = json.dumps({"command": command})
    return {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": arguments}}


# A conversation the SDK sends with replay: the system prompt, the task, turns with reasoning, text and tool calls,
# their results, and a condensation's summary as a user turn after the task.
CHAT: list[dict[str, Any]] = [
    {"role": "system", "content": "You are an agent that works in a terminal."},
    {"role": "user", "content": "Write 42 to answer.txt."},
    {"role": "user", "content": "SUMMARY: nothing is written yet."},
    {
        "role": "assistant",
        "content": "Writing it.",
        "reasoning_content": "The task wants 42 in answer.txt.",
        "tool_calls": [tool_call("call_1", "echo 42 > answer.txt")],
    },
    {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": "[exit code 0]"},
    {"role": "assistant", "content": None, "reasoning_content": "", "tool_calls": [tool_call("call_2", "seq 1 30")]},
    {"role": "tool", "tool_call_id": "call_2", "name": "terminal", "content": "\n".join(map(str, range(1, 31)))},
]


def parts(*texts: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text} for text in texts]


# Without replay the SDK sends text parts as lists.
LISTED: list[dict[str, Any]] = [
    {"role": "system", "content": parts("You are an agent.", " Work in the terminal.")},
    {"role": "user", "content": parts("Write 42 to answer.txt.")},
    {"role": "assistant", "content": parts("Writing it."), "tool_calls": [tool_call("call_1", "echo 42 > answer.txt")]},
    {"role": "tool", "tool_call_id": "call_1", "name": "terminal", "content": parts("[exit code 0]")},
]


def rendered(gateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
    """The prompt the gateway samples for a chat call with ``messages`` and ``tools``."""

    from platoon.gateway.dialects.openai_chat import OpenAIChat

    request = OpenAIChat(gateway).chat_request({"messages": copy.deepcopy(messages), "tools": tools})
    return gateway.render(request.messages, request.render).to_ints()


def counted_as_messages(runner, gateway, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[int]:
    """The prompt the gateway renders for the Messages count body of the same call."""

    from platoon.gateway.dialects.anthropic import AnthropicMessages

    request = AnthropicMessages().chat_request(runner.messages_count_body(copy.deepcopy(messages), tools))
    return gateway.render(request.messages, request.render).to_ints()


@pytest.mark.parametrize("messages", [CHAT, CHAT[:2], CHAT[:4], CHAT[:5], LISTED], ids=["all", "2", "4", "5", "lists"])
def test_the_messages_count_body_renders_as_the_chat_call(runner, messages):
    from platoon.testing import make_gateway

    gateway = make_gateway()  # platoon.testing's toy tokenizer, under the qwen3_instruct renderer

    for tools in (TOOLS, []):
        assert counted_as_messages(runner, gateway, messages, tools) == rendered(gateway, messages, tools)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}],
        [*CHAT[:2], {"role": "system", "content": "Instructions that come late."}],
        [*CHAT[:2], {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "[1]"}}]}],
        [*CHAT[:2], {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{"}}]}],
        [{"role": "system", "content": "A system prompt and nothing else."}],
        [{"role": "developer", "content": "A role the count does not carry."}],
    ],
    ids=["image", "late-system", "list-arguments", "broken-arguments", "system-only", "developer"],
)
def test_a_call_the_messages_count_cannot_carry_is_left_to_the_estimate(runner, messages):
    assert runner.messages_count_body(messages, TOOLS) is None


@pytest_asyncio.fixture
async def served():
    """A real gateway (``platoon.testing``'s toy tokenizer) served over HTTP, and one agent of it."""

    from platoon.gateway.server import GatewayServer
    from platoon.testing import make_gateway

    gateway = make_gateway()
    server = GatewayServer(gateway)
    await server.start(host="127.0.0.1")
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    try:
        yield gateway, f"{server.base_url}/v1", agent
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_the_gateway_counts_a_chat_call_as_it_renders_it_and_records_nothing(runner, served):
    gateway, base_url, agent = served
    count = runner.GatewayTokenCount(base_url, agent.api_key)

    for messages in (CHAT, CHAT[:2], CHAT[:4], LISTED):
        assert await asyncio.to_thread(count.count, messages, TOOLS) == len(rendered(gateway, messages, TOOLS))
    assert count.unsupported is None and not agent.calls and not agent.interactions  # no model call


@pytest.mark.asyncio
async def test_a_gateway_that_counts_no_tokens_leaves_the_estimate_and_says_so_once(runner, served, capsys):
    _, base_url, agent = served
    count = runner.GatewayTokenCount(f"{base_url}/no-such-route", agent.api_key)

    assert await asyncio.to_thread(count.count, CHAT, TOOLS) is None
    assert await asyncio.to_thread(count.count, CHAT[:2], TOOLS) is None  # not asked again
    assert count.unsupported == "HTTP 404"
    assert capsys.readouterr().out.count("the gateway counts no tokens (HTTP 404)") == 1


@pytest.mark.asyncio
async def test_an_answer_that_is_no_count_leaves_that_call_to_the_estimate_and_the_first_is_reported(
    runner, served, capsys
):
    _, base_url, _ = served
    refused = runner.GatewayTokenCount(base_url, "a-key-of-no-agent")  # 401
    unreachable = runner.GatewayTokenCount("http://127.0.0.1:9/v1", "a-key")

    for count, why in ((refused, "HTTP 401"), (unreachable, "URLError")):
        assert await asyncio.to_thread(count.count, CHAT, TOOLS) is None
        assert await asyncio.to_thread(count.count, CHAT[:2], TOOLS) is None
        assert count.unsupported is None and count.uncounted == why  # the next call asks again
    out = capsys.readouterr().out
    assert out.count("the gateway gave no token count for a call (HTTP 401)") == 1
    assert out.count("the gateway gave no token count for a call (URLError)") == 1


def test_a_call_the_messages_count_cannot_carry_is_reported_once(runner, capsys):
    count = runner.GatewayTokenCount("http://127.0.0.1:9/v1", "a-key")
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]

    assert count.count(image, TOOLS) is None and count.count(image, TOOLS) is None
    assert count.uncounted == "the Messages count cannot carry the call"
    assert capsys.readouterr().out.count("the gateway gave no token count for a call") == 1


# --------------------------------------------------------------------------- what the summary call reads


def test_the_summary_call_reads_each_forgotten_event_whole_without_its_reasoning(runner):
    output = "\n".join(map(str, range(1, 1501)))  # the SDK would show its first 500 characters
    summary = "SUMMARY: " + "kept " * 200 + "TAIL"
    command = {"id": "call_1", "name": "terminal", "arguments": json.dumps({"command": "seq 1 1500"})}
    events = [
        ActionEvent("agent", "assistant", "Now look at many numbers.", calls=[types.SimpleNamespace(**command)]),
        ObservationEvent("environment", "tool", output, name="terminal"),
        CondensationSummaryEvent("environment", "user", summary),
    ]

    action, observation, earlier = (runner.summary_input(event) for event in events)

    assert (
        action
        == 'ActionEvent (agent)\n  assistant: Now look at many numbers.\n  Call: terminal {"command": "seq 1 1500"}'
    )
    assert observation == f"ObservationEvent (environment)\n  Tool: terminal\n  tool: {output}"
    assert earlier == f"CondensationSummaryEvent (environment)\n  user: {summary}"


def test_an_event_that_is_no_message_reads_as_the_sdk_prints_it(runner):
    class Request:
        def to_llm_message(self):
            raise NotImplementedError

        def __str__(self) -> str:
            return "CondensationRequest (environment)"

    assert runner.summary_input(Request()) == "CondensationRequest (environment)"


@pytest.mark.parametrize(
    ("lengths", "budget", "cap"),
    [
        ([100, 100, 100], 300, 100),  # all of it fits
        ([1000, 100, 100], 600, 400),  # the long event is cut first
        ([1000, 100, 100], 240, 80),  # then all of them
        ([50, 50], 1, 1),  # never below one character
    ],
)
def test_a_context_reset_cuts_the_events_to_the_length_that_fits_its_budget(runner, lengths, budget, cap):
    assert runner.event_length_cap(lengths, budget) == cap
    assert sum(min(length, cap) for length in lengths) <= budget or cap == 1


@pytest.mark.asyncio
async def test_the_counted_llm_counts_at_its_own_gateway_and_else_takes_the_sdks_estimate(runner, served, capsys):
    gateway, base_url, agent = served
    tools = [ToolDefinition(tool) for tool in TOOLS]
    llm = runner.CountedReplayLLM(model="openai/policy", base_url=base_url, api_key=agent.api_key)

    assert await asyncio.to_thread(llm.get_token_count, CHAT, tools, True) == len(rendered(gateway, CHAT, TOOLS))
    image = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]
    assert await asyncio.to_thread(llm.get_token_count, image, tools) == ESTIMATE
    offline = runner.CountedLLM(model="openai/policy", base_url="http://127.0.0.1:9/v1", api_key=agent.api_key)
    assert await asyncio.to_thread(offline.get_token_count, CHAT, tools) == ESTIMATE

    broken = [types.SimpleNamespace(to_openai_tool=lambda **_: 1 / 0)]
    assert llm.get_token_count(CHAT, broken) == ESTIMATE == llm.get_token_count(CHAT, broken)
    out = capsys.readouterr().out
    assert out.count("a token count at the gateway failed") == 1 and "ZeroDivisionError" in out


def test_the_messages_count_of_each_recorded_openhands_call_is_its_prompt(runner):
    """The calls the real SDK runner sent (``openhands_live.py``'s fixture), under the Qwen3.5-4B chat template the
    fast path trains with: the Messages count body renders to each call's prompt, token for token.  The fixture keeps
    its JSON with sorted keys, so its tools are put back in the order the SDK writes them (``type``, then
    ``function`` with ``name``, ``description`` and ``parameters``), which is the order the Messages dialect gives
    them: a chat template writes tools out as JSON, keys in order."""

    from openhands_live import SNAPSHOT

    if not (SNAPSHOT / "tokenizer.json").exists():
        pytest.skip(f"Qwen3.5-4B tokenizer not available at {SNAPSHOT}")
    from openhands_live import ScriptedPolicy, load_fixture, load_tokenizer, recording_gateway

    from platoon.gateway.core import RenderRequest
    from platoon.gateway.dialects.anthropic import AnthropicMessages

    tokenizer = load_tokenizer()
    gateway = recording_gateway(tokenizer, ScriptedPolicy(tokenizer, []))
    for call in load_fixture()["calls"]:
        keys = ("name", "description", "parameters")
        functions = [{key: tool["function"][key] for key in keys} for tool in call["tools"]]
        tools = [{"type": "function", "function": function} for function in functions]
        prompt = gateway.render(copy.deepcopy(call["messages"]), RenderRequest(tools=tools))
        request = AnthropicMessages().chat_request(runner.messages_count_body(call["messages"], tools))
        assert gateway.render(request.messages, request.render).to_ints() == prompt.to_ints()
    for messages in (CHAT, LISTED):  # also text parts as lists, as the SDK sends them without replay
        assert counted_as_messages(runner, gateway, messages, TOOLS) == rendered(gateway, messages, TOOLS)
