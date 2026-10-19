"""Conversations, renderer settings and a simulated episode for the saved renderer token fixtures.

``tokens.json.gz`` holds the prompt token ids that platoon's custom Qwen3.6/3.8 preserve-thinking renderers
produced for every setting in use, before the chat-template renderer replaced them, plus the datum boundaries
``trajectory_to_data`` cut on a simulated multi-call episode.  ``test_renderer_fixtures.py``
checks the current renderers reproduce both exactly.  Each renderer is built by name and configured the way
``platoon.train.tinker.proxy.build_tinker_llm`` does it, and renders through the real ``ModelGateway``.

Re-record only on purpose (a deliberate prompt change), from the repository root::

    HF_HUB_OFFLINE=1 python tests/renderers/renderer_cases.py
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from huggingface_hub.constants import HF_HUB_CACHE

# A Hugging Face hub cache: PLATOON_TEST_HF_HUB, else the standard one (HF_HUB_CACHE, else HF_HOME/hub, else
# ~/.cache/huggingface/hub).
HUB = Path(os.environ.get("PLATOON_TEST_HF_HUB") or HF_HUB_CACHE)
SNAPSHOTS = {
    "qwen3_6": HUB
    / "models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/ce815485d640a33461c25e3d5c505b5346e8c857",
    "qwen3_8": HUB
    / "models--apurvaga--Qwen3.8-27B-preserve-thinking/snapshots/97c725615b8aa025b800da4ceed38811358b30f1",
}
FIXTURE = Path(__file__).with_name("tokens.json.gz")

# (setting, snapshot, renderer_name, renderer_kwargs): every renderer configuration in use.
SETTINGS: list[tuple[str, str, str, dict[str, Any]]] = [
    ("3.6 preserve", "qwen3_6", "qwen3_6_preserve_thinking", {"preserve_thinking": True}),
    ("3.6 preserve off", "qwen3_6", "qwen3_6_preserve_thinking", {"preserve_thinking": False}),
    ("3.6 disable thinking", "qwen3_6", "qwen3_6_preserve_thinking_disable_thinking", {}),
    ("3.8 xhigh (default)", "qwen3_8", "qwen3_8_preserve_thinking", {"preserve_thinking": True}),
    ("3.8 xhigh", "qwen3_8", "qwen3_8_preserve_thinking", {"preserve_thinking": True, "reasoning_effort": "xhigh"}),
    ("3.8 medium", "qwen3_8", "qwen3_8_preserve_thinking", {"preserve_thinking": True, "reasoning_effort": "medium"}),
    ("3.8 low", "qwen3_8", "qwen3_8_preserve_thinking", {"preserve_thinking": True, "reasoning_effort": "low"}),
    ("3.8 disable thinking", "qwen3_8", "qwen3_8_preserve_thinking_disable_thinking", {}),
]

# Per-request chat_template_kwargs (the condenser sends enable_thinking/preserve_thinking False).
OVERRIDES: list[dict[str, bool] | None] = [
    None,
    {"enable_thinking": False},
    {"preserve_thinking": False},
    {"enable_thinking": False, "preserve_thinking": False},
]


# --------------------------------------------------------------------------- conversations

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast — supports ünïcode city names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Forecast horizon"},
                    "units": {"type": "string", "enum": ["c", "f"]},
                    "opts": {"type": "object"},
                    "verbose": {"type": "boolean"},
                    "threshold": {"type": "number"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"],
            },
        },
    },
]

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command; output <= 10KB & UTF-8 (é, 漢字).",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "background": {"type": "boolean"},
                    "env": {"type": ["object", "null"]},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file.write-v2",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "mode": {"type": "number"},
                    "tags": {"type": "array"},
                },
            },
        },
    },
]
FLAT_TOOLS = [
    {
        "name": "lookup",
        "description": "flat tool, no wrapper",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
]
LOOKUP_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]

SYSTEM = "You are OpenHands, an autonomous coding agent.\nBe concise."


def asst(reasoning: str | None, content: str, tool_calls=None, *, omit_reasoning: bool = False) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if not omit_reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return message


def tool(call_id: str, name: str, content: Any) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def tc(name: str, arguments: Any, i: int = 0) -> dict:
    return {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": arguments}}


C1 = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What's the weather in Paris?"}]
C2 = C1 + [
    asst(
        "The user wants the Paris forecast.\nI should call get_weather.",
        "",
        [
            (
                "get_weather",
                {"city": "Paris", "days": 3, "opts": {"unit": "c", "x": [1, 2]}, "verbose": True, "threshold": 1.5},
            )
        ],
    ),
    tool("call_0", "get_weather", "  Sunny, 21C for 3 days.\n"),
]
C3 = C2 + [
    asst("The tool says sunny.", "It will be sunny in Paris for the next 3 days."),
    {"role": "user", "content": "And Tokyo and Osaka? Run `date` too."},
]
C4 = C3 + [
    asst(
        "Two cities plus a shell command: three calls.",
        "Checking both cities and the date.",
        [
            ("get_weather", {"city": "Tokyo"}),
            ("get_weather", {"city": "Ōsaka", "units": "c"}),
            ("execute_bash", {"command": "date\necho done", "timeout": 30}),
        ],
    ),
    tool("call_0", "get_weather", "Rainy"),
    tool("call_1", "get_weather", "Cloudy"),
    tool("call_2", "execute_bash", "Mon Sep  7 10:00:00 UTC 2026\ndone"),
]
C5 = C4 + [
    asst("Summarize the three results.", "Tokyo: rainy. Osaka: cloudy. Date: Mon Sep 7 2026."),
    {"role": "user", "content": "Thanks!"},
]
C_PARTS = [
    {"role": "system", "content": [{"type": "text", "text": "  You are a helpful assistant. "}]},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this file: "},
            {"type": "text", "text": "a.py\n"},
            {"type": "text", "text": "and tell me what it does."},
        ],
    },
]
C_NO_REASONING = C1 + [
    asst(None, "Let me look that up.", [("get_weather", {"city": "Paris"})], omit_reasoning=True),
    tool("call_0", "get_weather", "Sunny"),
    asst(None, "Sunny in Paris.", omit_reasoning=True),
    {"role": "user", "content": "And with an embedded think block?"},
    asst(None, "<think>\nembedded reasoning\n</think>\n\nembedded answer", omit_reasoning=True),
    {"role": "user", "content": "And an explicit null reasoning_content?"},
    asst(None, "plain answer"),
    {"role": "user", "content": "ok"},
]
C_TOOL_RESPONSE_AS_USER = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Run ls."},
    asst("Calling execute_bash.", "", [("execute_bash", {"command": "ls"})]),
    {"role": "user", "content": "<tool_response>\na.py b.py\n</tool_response>"},
    asst("Done.", "Two files."),
    {"role": "user", "content": "<tool_response>\nack\n</tool_response>"},
]
C_EMPTY_REASONING_CALL = C1 + [
    asst("", "Sunny.", [("get_weather", {"city": "Paris"})]),
    tool("call_0", "get_weather", "Sunny"),
]
C_HISTORICAL_INLINE = [
    {"role": "user", "content": "Look up the result."},
    {"role": "assistant", "content": "<think>Keep this historical reasoning.</think>Found it."},
    {"role": "user", "content": "Check again."},
]

# A Toolathlon-shaped agent loop: tool calls, tool results, a mid-conversation user nudge, more calls.
C_AGENT_NUDGE = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Find the failing test and fix it."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Run the test suite first.",
        "tool_calls": [tc("run_shell", json.dumps({"cmd": "pytest -q", "timeout": 600}))],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "F.\nFAILED test_a.py::test_x - assert 1 == 2\n"},
    {
        "role": "assistant",
        "content": "One failure in test_a.py.",
        "reasoning_content": "Look at test_a.py.",
        "tool_calls": [tc("run_shell", json.dumps({"cmd": "cat test_a.py"}), 1)],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "def test_x():\n    assert 1 == 2\n"},
    {"role": "assistant", "content": "The assertion is wrong.", "reasoning_content": "The test itself is wrong."},
    {"role": "user", "content": "Your response was cut off. Continue, and keep the fix minimal."},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Rewrite the assertion.",
        "tool_calls": [
            tc(
                "file.write-v2",
                json.dumps({"path": "test_a.py", "content": "def test_x():\n    assert 1 == 1\n", "mode": 420}),
                2,
            ),
            tc("run_shell", json.dumps({"cmd": "pytest -q", "background": False, "env": None}), 3),
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "written"},
    {"role": "tool", "tool_call_id": "call_3", "content": ".. 2 passed"},
]

# Adversarial conversations.
C_AGENT_LOOP_TOOL_LAST = [
    {"role": "system", "content": "  You are terse.  \n"},
    {"role": "user", "content": "List files then write a summary."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "  Need ls first.\n",
        "tool_calls": [tc("run_shell", json.dumps({"cmd": "ls -la", "timeout": 30, "background": False, "env": None}))],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "a.txt\nb.txt\n"},
    {
        "role": "assistant",
        "content": "Writing now.",
        "reasoning_content": "Two files.",
        "tool_calls": [
            tc(
                "file.write-v2",
                json.dumps(
                    {"path": "/tmp/ñ.md", "content": "naïve <b>&amp;</b>", "mode": 0.5, "tags": ["x", 1, None, True]},
                    ensure_ascii=False,
                ),
                1,
            )
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [{"type": "text", "text": "ok "}, {"type": "text", "text": "written"}],
    },
]
C_PARALLEL_CONSECUTIVE_TOOLS = [
    {"role": "user", "content": [{"type": "text", "text": "Do two things"}]},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "parallel",
        "tool_calls": [
            tc("run_shell", {"cmd": "echo 1", "timeout": 1}, 0),
            tc("run_shell", {"cmd": "echo 2", "env": {"A": "ü", "B": [1.5, 1e-07, 12345678901234567890]}}, 1),
        ],
    },
    {"role": "tool", "tool_call_id": "call_0", "content": "1"},
    {"role": "tool", "tool_call_id": "call_1", "content": "2"},
    {"role": "tool", "tool_call_id": "call_x", "content": "late"},
    {"role": "user", "content": "<tool_response>\nfake wrapped\n</tool_response>"},
]
C_EMPTY_AND_NONE_ARGS = [
    {"role": "user", "content": "go"},
    {
        "role": "assistant",
        "content": "  \n ",
        "reasoning_content": "",
        "tool_calls": [tc("run_shell", "", 0), tc("run_shell", "{}", 1), tc("run_shell", None, 2)],
    },
    {"role": "tool", "content": "done"},
]
C_INLINE_THINK_AND_MIXED = [
    {"role": "system", "content": [{"type": "text", "text": "sys part1 "}, {"type": "text", "text": "part2"}]},
    {"role": "user", "content": "q1 with </think> literal"},
    {"role": "assistant", "content": "<think>\ninline r1\n</think>\n\nanswer 1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "<think>ignored inline</think>answer 2", "reasoning_content": "explicit r2"},
    {"role": "user", "content": "q3"},
    {"role": "assistant", "content": "<think>kept inline</think>answer 3", "reasoning_content": ""},
    {"role": "user", "content": "q4 tool output has <|im_end|> literal"},
]
C_FLAT_TOOL_CALL_MULTI_USER = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
    {"role": "user", "content": "lookup x"},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "r",
        "tool_calls": [{"name": "lookup", "arguments": {"q": "x"}}],
    },
    {"role": "tool", "content": "X!"},
    {"role": "assistant", "content": "X is X!", "reasoning_content": "found"},
    {"role": "user", "content": "thanks; now again"},
]
C_LONG_HISTORY = [{"role": "user", "content": "start"}] + sum(
    (
        [
            {
                "role": "assistant",
                "content": f"step {i}" if i % 3 else "",
                "reasoning_content": ("think " * (i % 7)).strip(),
                "tool_calls": [tc("run_shell", json.dumps({"cmd": f"echo {i}", "timeout": i}), i)],
            },
            {"role": "tool", "tool_call_id": f"call_{i}", "content": f"out {i}\n" * (i % 4 + 1)},
        ]
        + ([{"role": "user", "content": f"nudge {i}"}] if i % 10 == 9 else [])
        for i in range(60)
    ),
    [],
)

CONVERSATIONS: dict[str, tuple[list[dict] | None, list[dict]]] = {
    "system+user": (TOOLS, C1),
    "system+user no tools": (None, C1),
    "tool call and result": (TOOLS, C2),
    "tool call, result, answer, new user": (TOOLS, C3),
    "three tool calls and three results": (TOOLS, C4),
    "three assistant turns": (TOOLS, C5),
    "content parts": (None, C_PARTS),
    "content parts + tools": (TOOLS, C_PARTS),
    "no reasoning_content": (TOOLS, C_NO_REASONING),
    "tool_response wrapped as user": (TOOLS, C_TOOL_RESPONSE_AS_USER),
    "tools without system": (TOOLS, C1[1:]),
    "tools with blank system": (TOOLS, [{"role": "system", "content": "  \n"}] + C1[1:]),
    "empty tools list": ([], C1),
    "no system, no tools": (None, C1[1:]),
    "empty system, no tools": (None, [{"role": "system", "content": ""}, {"role": "user", "content": "short"}]),
    "empty reasoning on a tool call": (TOOLS, C_EMPTY_REASONING_CALL),
    "historical inline think": (None, C_HISTORICAL_INLINE),
    "historical inline think + system": (
        None,
        [{"role": "system", "content": "Follow the task."}] + C_HISTORICAL_INLINE,
    ),
    "historical inline think + blank system + tools": (
        LOOKUP_TOOLS,
        [{"role": "system", "content": ""}] + C_HISTORICAL_INLINE,
    ),
    "agent loop with a user nudge": (AGENT_TOOLS, C_AGENT_NUDGE),
    "agent loop ending in a tool result": (AGENT_TOOLS, C_AGENT_LOOP_TOOL_LAST),
    "parallel calls, consecutive tool results": (AGENT_TOOLS, C_PARALLEL_CONSECUTIVE_TOOLS),
    "empty, {} and null arguments": (AGENT_TOOLS, C_EMPTY_AND_NONE_ARGS),
    "inline think mixed with reasoning_content": (None, C_INLINE_THINK_AND_MIXED),
    "flat tools and a flat tool call": (FLAT_TOOLS, C_FLAT_TOOL_CALL_MULTI_USER),
    "60-step history with nudges": (AGENT_TOOLS, C_LONG_HISTORY),
}


def cookbook_conversations() -> dict[str, tuple[list[dict] | None, list[Any]]]:
    """Cookbook message shapes (ToolCall objects, thinking parts) that clients built on tinker_cookbook send."""
    from tinker_cookbook.renderers import ToolCall

    conv = C1 + [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "cookbook thinking"},
                {"type": "text", "text": "\n\ncookbook text"},
            ],
            "tool_calls": [
                ToolCall(id="call_0", function=ToolCall.FunctionBody(name="get_weather", arguments='{"city": "Paris"}'))
            ],
        },
        tool("call_0", "get_weather", "Sunny"),
    ]
    return {"cookbook ToolCall and thinking part": (TOOLS, conv)}


def all_conversations() -> dict[str, tuple[list[dict] | None, list[Any]]]:
    return {**CONVERSATIONS, **cookbook_conversations()}


# --------------------------------------------------------------------------- simulated episode

# Each call samples a completion (after the generation prompt) and the client answers with a tool result or,
# once, a user message.  The shapes include an empty think block, which re-renders differently and so cuts a
# new datum on every renderer.
EPISODE: list[tuple[str, str | None, str | None]] = [
    (
        "Run the tests first.\n</think>\n\n"
        "<tool_call>\n<function=run_shell>\n<parameter=cmd>\npytest -q\n</parameter>\n"
        "<parameter=timeout>\n600\n</parameter>\n</function>\n</tool_call>",
        "F.\nFAILED test_a.py::test_x\n",
        None,
    ),
    (
        "Look at the failing test.\n</think>\n\nReading it.\n\n"
        "<tool_call>\n<function=run_shell>\n<parameter=cmd>\ncat test_a.py\n</parameter>\n</function>\n</tool_call>",
        "def test_x():\n    assert 1 == 2\n",
        None,
    ),
    ("The assertion is wrong.\n</think>\n\nThe test asserts 1 == 2.", None, "Continue, and fix it."),
    (
        "Rewrite it.\n</think>\n\n"
        "<tool_call>\n<function=file.write-v2>\n<parameter=path>\ntest_a.py\n</parameter>\n"
        "<parameter=content>\ndef test_x():\n    assert 1 == 1\n</parameter>\n"
        '<parameter=tags>\n["fix", 1]\n</parameter>\n</function>\n</tool_call>\n'
        "<tool_call>\n<function=run_shell>\n<parameter=cmd>\npytest -q\n</parameter>\n"
        "<parameter=background>\nfalse\n</parameter>\n</function>\n</tool_call>",
        "written",
        None,
    ),
    (
        "\n</think>\n\n<tool_call>\n<function=run_shell>\n<parameter=cmd>\ngit diff\n</parameter>\n</function>\n"
        "</tool_call>",
        "-    assert 1 == 2\n+    assert 1 == 1\n",
        None,
    ),
    ("Done.\n</think>\n\nFixed test_a.py; the suite passes.", None, None),
]


def make_gateway(renderer: Any, tokenizer: Any):
    from platoon.gateway.core import ModelGateway

    return ModelGateway(sampler=None, renderer=renderer, tokenizer=tokenizer, model_name="fixture")  # type: ignore[arg-type]


def build_renderer(tokenizer: Any, renderer_name: str, renderer_kwargs: dict[str, Any]):
    """``platoon.train.tinker.proxy.build_tinker_llm``'s renderer construction."""
    from tinker_cookbook.renderers import get_renderer

    from platoon.train.tinker.renderers import register

    register()
    renderer = get_renderer(renderer_name, tokenizer, image_processor=None, model_name=tokenizer.name_or_path)
    for key, value in renderer_kwargs.items():
        if not hasattr(renderer, key):
            raise ValueError(f"Renderer '{renderer_name}' does not support renderer_kwargs['{key}']")
        setattr(renderer, key, value)
    return renderer


def render_ids(gateway: Any, tools: list[dict] | None, messages: list[Any], overrides: dict | None) -> list[int]:
    from platoon.gateway.core import RenderRequest

    request = RenderRequest(tools=tools or None, render_kwargs=dict(overrides) if overrides else None)
    return list(gateway.render(copy.deepcopy(messages), request).to_ints())


def run_episode(gateway: Any, tokenizer: Any) -> list[tuple[int, str]]:
    """Datum boundaries of the simulated episode: ``(model_input length, sha256 of its tokens)`` per datum."""
    import tinker

    import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
    from platoon.gateway.core import RenderRequest
    from platoon.utils import tinker_data_processing as tdp

    stop = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Fix the failing test."}]
    request = RenderRequest(tools=AGENT_TOOLS)
    interactions: dict[str, Any] = {}
    for i, (completion, tool_result, user_message) in enumerate(EPISODE):
        prompt = gateway.render(copy.deepcopy(messages), request)
        tokens = tokenizer.encode(completion, add_special_tokens=False) + stop
        sampled = tinker.types.SampleResponse(
            sequences=[tinker.types.SampledSequence(stop_reason="stop", tokens=tokens, logprobs=[-0.5] * len(tokens))]
        )
        response = gateway.parse(prompt, sampled, request)
        interactions[f"c{i}"] = SimpleNamespace(
            obs=prompt, action=SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens)), routing_id=None
        )
        reply: dict[str, Any] = {"role": "assistant", "content": response.content}
        if response.reasoning_content is not None:
            reply["reasoning_content"] = response.reasoning_content
        if response.tool_calls:
            reply["tool_calls"] = response.tool_calls
        messages.append(reply)
        for call in response.tool_calls or []:
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result})
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
    result = tdp.trajectory_to_data(
        trajectory={"steps": [{"misc": {"action_misc": {"completion_id": cid}}} for cid in interactions]},
        interactions=interactions,
        task_id="task",
        trajectory_id="episode",
        trajectory_advantage=1.0,
        checkpoint_version=0,
    )
    return [
        (datum.model_input.length, hashlib.sha256(json.dumps(datum.model_input.to_ints()).encode()).hexdigest())
        for datum in result.datums
    ]


def load_tokenizer(snapshot: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOTS[snapshot]))


def record() -> dict[str, Any]:
    tokenizers = {key: load_tokenizer(key) for key in SNAPSHOTS}
    conversations = all_conversations()
    out: dict[str, Any] = {"settings": {}, "episodes": {}}
    for setting, snapshot, renderer_name, renderer_kwargs in SETTINGS:
        tokenizer = tokenizers[snapshot]
        gateway = make_gateway(build_renderer(tokenizer, renderer_name, renderer_kwargs), tokenizer)
        cases: dict[str, Any] = {}
        for name, (tools, messages) in conversations.items():
            for overrides in OVERRIDES:
                key = f"{name} | {json.dumps(overrides, sort_keys=True)}"
                try:
                    cases[key] = render_ids(gateway, tools, messages, overrides)
                except Exception as e:  # the fixture then expects the renderer to refuse the conversation
                    cases[key] = {"error": type(e).__name__}
        out["settings"][setting] = cases
        if gateway.renderer.enable_thinking:
            out["episodes"][setting] = run_episode(gateway, tokenizer)
    return out


def load_fixture() -> dict[str, Any]:
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


if __name__ == "__main__":
    data = record()
    with gzip.GzipFile(FIXTURE, "wb", mtime=0) as raw:
        raw.write(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
    sizes = {s: len(c) for s, c in data["settings"].items()}
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes): {sizes}; episodes {data['episodes']}")
