"""Tool calls round-trip through every chat template in the local Hugging Face cache.

For each cached model with a chat template (PLATOON_TEST_HF_HUB, else the standard cache: HF_HUB_CACHE, else
HF_HOME/hub), the chat-template renderer must choose the parser of the model's format by itself, and only that
parser may read the template's probe turn back.  Assistant turns rendered by the template (multi-line code with a
literal ``</parameter>``, strings that read as numbers and booleans, a call with no arguments next to others,
nested and non-ASCII values) are encoded as a model would sample them and parsed back through
``ChatTemplateRenderer.parse_response`` with their prompt: the calls, content and reasoning must come back exactly.
Malformed variants of the same turns must parse exactly by a lenient rule or report the broken call, never lose it.
Models are skipped when no snapshot is cached.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HUB = Path(os.environ.get("PLATOON_TEST_HF_HUB") or HF_HUB_CACHE)


def _cached_templates() -> list[Path]:
    snapshots = []
    for snapshot in sorted(HUB.glob("models--*/snapshots/*")):
        if not (snapshot / "tokenizer_config.json").exists():
            continue
        has_template = (snapshot / "chat_template.jinja").exists() or "chat_template" in json.loads(
            (snapshot / "tokenizer_config.json").read_text()
        )
        if has_template:
            snapshots.append(snapshot)
    return snapshots


SNAPSHOTS = _cached_templates()
pytestmark = pytest.mark.skipif(not SNAPSHOTS, reason=f"no chat-template snapshots cached under {HUB}")


def _model(snapshot: Path) -> str:
    return snapshot.parent.parent.name.removeprefix("models--").replace("--", "/")


# The parser each known model family must get; other cached models only need some parser that round-trips.
EXPECTED = [
    (re.compile(r"Qwen3\.[5-9]|Qwen3-Coder"), "qwen3_coder"),
    (re.compile(r"Qwen2\.5|Qwen3-[0-9.]+B(-Instruct|-Thinking)?(-2507)?$"), "hermes"),
]


def _expected(snapshot: Path) -> str | None:
    model = _model(snapshot)
    return next((name for pattern, name in EXPECTED if pattern.search(model)), None)


@pytest.fixture(scope="module")
def renderers():
    from transformers import AutoTokenizer

    from platoon.train.tinker.renderers import ChatTemplateRenderer

    cache: dict[tuple[Path, bool], Any] = {}
    tokenizers: dict[Path, Any] = {}

    def get(snapshot: Path, enable_thinking: bool = True):
        if snapshot not in tokenizers:
            tokenizers[snapshot] = AutoTokenizer.from_pretrained(str(snapshot))
        if (snapshot, enable_thinking) not in cache:
            cache[snapshot, enable_thinking] = ChatTemplateRenderer(
                tokenizers[snapshot], enable_thinking=enable_thinking
            )
        return cache[snapshot, enable_thinking]

    return get


IDS = [_model(snapshot) for snapshot in SNAPSHOTS]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "opts": {"type": "object"},
                    "threshold": {"type": "number"},
                    "verbose": {"type": "boolean"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "overwrite": {"type": "boolean"},
                    "note": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {"name": "list-dir.v2", "description": "No arguments.", "parameters": {"type": "object"}},
    },
]
CODE = "    def foo():\n        return {\"a\": '</parameter>'}\n"
HARD_CALLS = [
    ("get_weather", {"location": "Tokyo, JP", "days": 3, "unit": "celsius"}),
    ("write_file", {"path": "123", "content": CODE, "tags": ["x", "y"], "overwrite": True, "note": "true"}),
    ("list-dir.v2", {}),
]
TURNS: dict[str, tuple[str | None, str, list[tuple[str, dict[str, Any]]]]] = {
    "three calls with hard values": ("I will call three tools.", "Sure.", HARD_CALLS),
    "nested and non-ASCII values": (
        "Ōsaka next.",
        "",
        [
            (
                "get_weather",
                {"location": "Ōsaka 大阪", "opts": {"unit": "c", "x": [1, 2]}, "threshold": 1.5, "verbose": False},
            )
        ],
    ),
    "a call with no arguments alone": ("", "", [("list-dir.v2", {})]),
    "a literal call and </think> in a value": (
        "Write a parser test.",
        "Writing it.",
        [
            (
                "write_file",
                {
                    "path": "test.py",
                    "content": "X = '</think>'\n<tool_call>\n<function=a>\n<parameter=b>\nc\n</parameter>\n"
                    "</function>\n</tool_call>\n",
                },
            ),
            ("list-dir.v2", {}),
        ],
    ),
    "calls written out in a value, as OpenHands' prompts write them": (
        "Document the call format.",
        "Writing the docs.",
        [
            (
                "write_file",
                {
                    "path": "docs/calls.md",
                    # No <tool_call>, and parameters on one line or on several.
                    "content": "Call tools like this:\n<function=get_weather>\n<parameter=location>Paris</parameter>\n"
                    "<parameter=days>\n3\n</parameter>\n</function>\n",
                    "note": "The last parameter.",
                },
            ),
            ("list-dir.v2", {}),
        ],
    ),
    "Hermes calls written out in a value": (
        "",
        "",
        [
            (
                "write_file",
                {
                    "path": "tests/hermes.txt",
                    "content": 'Examples:\n<tool_call>\n{"name": "get_weather", "arguments": {"location": "Oslo"}}\n'
                    '</tool_call>\n<tool_call>\n{"name": "list-dir.v2", "arguments": {}}\n</tool_call>\n',
                },
            )
        ],
    ),
    "a value with a line that opens a later parameter": (
        "",
        "",
        [("write_file", {"path": "a.md", "content": "Format:\n<parameter=note>\nvalue\n", "note": "x"})],
    ),
    "text only": ("Nothing to call.", "All done.", []),
}


def _assistant(reasoning: str | None, content: str, calls) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": arguments}}
            for i, (name, arguments) in enumerate(calls)
        ]
    return message


def sampled_turn(renderer, assistant: dict[str, Any]) -> tuple[list[int], str]:
    """The prompt's tokens and the text a model would sample for ``assistant``, cut at the stop token."""
    messages = [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Do the tasks."},
        assistant,
    ]
    prompt = renderer.render_text(messages[:-1], tools=TOOLS)
    full = renderer.render_text(messages, tools=TOOLS, add_generation_prompt=False)
    assert full.startswith(prompt), "the template is not prefix-stable for this turn"
    sampled = full[len(prompt) :]
    stops = [renderer.tokenizer.decode([token]) for token in renderer.get_stop_sequences()]
    cut = min((sampled.find(stop) for stop in stops if stop in sampled), default=len(sampled))
    return renderer.tokenizer.encode(prompt, add_special_tokens=False), sampled[:cut]


def parse_sampled(renderer, prompt: list[int], text: str):
    tokens = renderer.tokenizer.encode(text, add_special_tokens=False) + renderer.get_stop_sequences()[:1]
    message, success = renderer.parse_response(tokens, tools=TOOLS, prompt=prompt)
    assert success
    return message


def _calls(message) -> list[tuple[str, str]]:
    return [
        (call.function.name, json.dumps(json.loads(call.function.arguments), sort_keys=True))
        for call in message.get("tool_calls") or []
    ]


def _want(calls) -> list[tuple[str, str]]:
    return [(name, json.dumps(arguments, sort_keys=True)) for name, arguments in calls]


# --------------------------------------------------------------------------- choosing the parser


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=IDS)
@pytest.mark.parametrize("enable_thinking", [True, False])
def test_auto_chooses_the_models_format(renderers, snapshot, enable_thinking):
    _, choice = renderers(snapshot, enable_thinking).resolve_tool_call_parser()
    assert choice.setting == "auto" and choice.verified is True, choice
    expected = _expected(snapshot)
    if expected is not None:
        assert choice.parser == expected


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=IDS)
def test_only_the_chosen_parser_reads_the_template_back(renderers, snapshot):
    from platoon.train.tinker.renderers.tool_parsers import get_parser, parser_names
    from platoon.train.tinker.renderers.tool_parsers.probe import check, probe_template

    renderer = renderers(snapshot)
    stops = [renderer.tokenizer.decode([token]) for token in renderer.get_stop_sequences()]
    probe = probe_template(renderer._render_probe, stops)
    readers = [name for name in parser_names() if check(get_parser(name), probe) is None]
    assert readers == [renderer.tool_call_parser_choice.parser]


# --------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=IDS)
@pytest.mark.parametrize("turn", list(TURNS))
def test_rendered_turns_parse_back_exactly(renderers, snapshot, turn):
    renderer = renderers(snapshot)
    reasoning, content, calls = TURNS[turn]
    prompt, text = sampled_turn(renderer, _assistant(reasoning, content, calls))
    message = parse_sampled(renderer, prompt, text)
    assert _calls(message) == _want(calls)
    assert message["content"] == (content if content and content in text else "")
    assert message["reasoning_content"] == (reasoning if reasoning and reasoning in text else "")
    assert "unparsed_tool_calls" not in message and "recovered_tool_calls" not in message


# --------------------------------------------------------------------------- malformed output


def _xml(snapshot: Path) -> bool:
    return _expected(snapshot) == "qwen3_coder"


XML_SNAPSHOTS = [snapshot for snapshot in SNAPSHOTS if _xml(snapshot)]
HERMES_SNAPSHOTS = [snapshot for snapshot in SNAPSHOTS if _expected(snapshot) == "hermes"]


def _once(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


# variant -> (how the sampled text changes, how many calls parse exactly, recovered, unparsed)
XML_VARIANTS = {
    "missing inner </parameter>": (lambda t: _once(t, "3\n</parameter>\n", "3\n"), 3, 1, 0),
    "missing last </parameter>": (lambda t: _once(t, "celsius\n</parameter>\n", "celsius\n"), 3, 1, 0),
    "missing final </tool_call>": (lambda t: t[: t.rfind("</tool_call>")], 3, 1, 0),
    "calls inside an unclosed think block": (lambda t: _once(t, "\n</think>\n\n", "\n\n"), 3, 0, 0),
    "cut inside the second call": (lambda t: t[: t.find("<parameter=content>") + 40], 1, 0, 1),
    "Hermes JSON from an XML model": (
        lambda t: t[: t.find("<tool_call>")] + '<tool_call>\n{"name": "list-dir.v2", "arguments": {}}\n</tool_call>',
        0,
        0,
        1,
    ),
}


@pytest.mark.parametrize("snapshot", XML_SNAPSHOTS, ids=[_model(s) for s in XML_SNAPSHOTS])
@pytest.mark.parametrize("variant", list(XML_VARIANTS))
def test_malformed_xml_calls_parse_exactly_or_are_reported(renderers, snapshot, variant):
    renderer = renderers(snapshot)
    prompt, text = sampled_turn(renderer, _assistant("I will call three tools.", "Sure.", HARD_CALLS))
    change, exact, recovered, unparsed = XML_VARIANTS[variant]
    message = parse_sampled(renderer, prompt, change(text))
    assert _calls(message) == _want(HARD_CALLS[:exact])
    assert message.get("recovered_tool_calls", 0) == recovered
    assert len(message.get("unparsed_tool_calls", [])) == unparsed


@pytest.mark.parametrize("snapshot", HERMES_SNAPSHOTS, ids=[_model(s) for s in HERMES_SNAPSHOTS])
def test_malformed_hermes_calls_are_reported_one_by_one(renderers, snapshot):
    renderer = renderers(snapshot)
    prompt, text = sampled_turn(renderer, _assistant(None, "Sure.", HARD_CALLS))
    message = parse_sampled(renderer, prompt, _once(text, '"arguments": {"path"', '"arguments": {path"'))
    assert _calls(message) == _want([HARD_CALLS[0], HARD_CALLS[2]])
    assert message["unparsed_tool_calls"][0].raw_text.startswith('<tool_call>\n{"name": "write_file"')
    message = parse_sampled(renderer, prompt, text[: text.rfind("</tool_call>") - 5])
    assert _calls(message) == _want(HARD_CALLS[:2]) and len(message["unparsed_tool_calls"]) == 1
