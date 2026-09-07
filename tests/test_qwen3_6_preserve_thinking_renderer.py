"""Token-exactness tests for the Qwen3.6 preserve-thinking tinker renderer.

Every rendering is compared against ``tokenizer.apply_chat_template`` of the
local HF snapshot (skipped when the snapshot is not on disk).  Run with

    HF_HUB_OFFLINE=1 plugins/openreward/.venv/bin/python -m pytest \
        tests/test_qwen3_6_preserve_thinking_renderer.py -q
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SNAPSHOT = Path(
    os.environ.get(
        "QWEN36_PRESERVE_THINKING_SNAPSHOT",
        "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
        "models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/"
        "ce815485d640a33461c25e3d5c505b5346e8c857",
    )
)

pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "tokenizer.json").exists() or not (SNAPSHOT / "chat_template.jinja").exists(),
    reason=f"Qwen3.6 preserve-thinking tokenizer snapshot not available at {SNAPSHOT}",
)

# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


@pytest.fixture(scope="module")
def renderer(tokenizer):
    from platoon.train.tinker.renderers import Qwen3_6PreserveThinkingRenderer

    return Qwen3_6PreserveThinkingRenderer(tokenizer)


@pytest.fixture(scope="module")
def disable_thinking_renderer(tokenizer):
    from platoon.train.tinker.renderers import Qwen3_6PreserveThinkingDisableThinkingRenderer

    return Qwen3_6PreserveThinkingDisableThinkingRenderer(tokenizer)


# --------------------------------------------------------------------------- helpers

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
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
            },
        },
    },
]

SYSTEM = "You are OpenHands, an autonomous coding agent.\nBe concise."


def asst(reasoning: str | None, content: str, tool_calls=None, *, omit_reasoning=False):
    message = {"role": "assistant", "content": content}
    if not omit_reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return message


def tool(call_id: str, name: str, content: str):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def to_hf(messages):
    """What AReaL fed to apply_chat_template: JSON-string arguments -> dicts, None fields dropped."""
    out = []
    for message in copy.deepcopy(messages):
        message = {k: v for k, v in message.items() if v is not None}
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
        out.append(message)
    return out


def hf_ids(tokenizer, messages, tools=None, add_generation_prompt=True, **kwargs):
    out = tokenizer.apply_chat_template(
        to_hf(messages), tools=tools, tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs
    )
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def hf_text(tokenizer, messages, tools=None, **kwargs):
    return tokenizer.apply_chat_template(
        to_hf(messages), tools=tools, tokenize=False, add_generation_prompt=True, **kwargs
    )


def assert_exact(renderer, tokenizer, messages, tools=None, **kwargs):
    expected = hf_ids(tokenizer, messages, tools=tools, **kwargs)
    actual = renderer.build_generation_prompt(messages, tools=tools, **kwargs).to_ints()
    if actual != expected:
        # Show the string diff for a readable failure.
        assert renderer.render_conversation_text(messages, tools=tools, **kwargs) == hf_text(
            tokenizer, messages, tools=tools, **kwargs
        )
        assert actual == expected
    return actual


def assert_prefix(shorter: list[int], longer: list[int]):
    assert len(shorter) < len(longer)
    assert longer[: len(shorter)] == shorter


# --------------------------------------------------------------------------- conversations

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


# --------------------------------------------------------------------------- exactness


@pytest.mark.parametrize(
    "name,messages,tools",
    [
        ("system+tools+user", C1, TOOLS),
        ("system+user no tools", C1, None),
        ("assistant think+tool_call, tool result, assistant text", C3, TOOLS),
        ("three tool calls in one turn + three tool results", C4, TOOLS),
        ("three assistant turns keep thinking", C5, TOOLS),
        ("user content parts", C_PARTS, None),
        ("user content parts + tools", C_PARTS, TOOLS),
        ("no reasoning_content on assistant turns", C_NO_REASONING, TOOLS),
        ("tool_response wrapped as user turn", C_TOOL_RESPONSE_AS_USER, TOOLS),
        ("tools without system message", C1[1:], TOOLS),
        ("tools with empty system message", [{"role": "system", "content": "  \n"}] + C1[1:], TOOLS),
        ("empty tools list", C1, []),
        ("no system, no tools", C1[1:], None),
    ],
)
def test_matches_hf_chat_template(renderer, tokenizer, name, messages, tools):
    assert_exact(renderer, tokenizer, messages, tools=tools)


def test_rendered_text_matches_hf(renderer, tokenizer):
    assert renderer.render_conversation_text(C5, tools=TOOLS) == hf_text(tokenizer, C5, tools=TOOLS)
    assert renderer.render_conversation_text(
        C4, tools=TOOLS, add_generation_prompt=False
    ) == tokenizer.apply_chat_template(to_hf(C4), tools=TOOLS, tokenize=False, add_generation_prompt=False)


def test_enable_thinking_false_variant(disable_thinking_renderer, tokenizer):
    for messages, tools in ((C1, TOOLS), (C3, TOOLS), (C_PARTS, None), (C5, TOOLS)):
        assert_exact(disable_thinking_renderer, tokenizer, messages, tools=tools, enable_thinking=False)
    # Per-call override on the thinking renderer gives the same tokens.
    assert disable_thinking_renderer.build_generation_prompt(C3, tools=TOOLS).to_ints() == hf_ids(
        tokenizer, C3, tools=TOOLS, enable_thinking=False
    )
    tail = tokenizer.decode(disable_thinking_renderer.build_generation_prompt(C1).to_ints()[-12:])
    assert tail.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_per_call_overrides(renderer, tokenizer):
    assert renderer.build_generation_prompt(C3, tools=TOOLS, enable_thinking=False).to_ints() == hf_ids(
        tokenizer, C3, tools=TOOLS, enable_thinking=False
    )
    assert renderer.build_generation_prompt(C5, tools=TOOLS, preserve_thinking=False).to_ints() == hf_ids(
        tokenizer, C5, tools=TOOLS, preserve_thinking=False
    )


def test_preserve_thinking_false_attribute_matches_upstream_qwen_behavior(tokenizer):
    from platoon.train.tinker.renderers import Qwen3_6PreserveThinkingRenderer

    upstream = Qwen3_6PreserveThinkingRenderer(tokenizer, preserve_thinking=False)
    assert upstream.has_extension_property is False
    for messages in (C5, C_NO_REASONING, C_TOOL_RESPONSE_AS_USER):
        assert upstream.build_generation_prompt(messages, tools=TOOLS).to_ints() == hf_ids(
            tokenizer, messages, tools=TOOLS, preserve_thinking=False
        )
    # The thinking of turns before the last real user query is gone, later ones stay.
    text = upstream.render_conversation_text(C5, tools=TOOLS)
    assert "Summarize the three results." not in text
    assert text.count("<think>") == 1  # only the generation prompt


# --------------------------------------------------------------------------- prefix extension


def test_prefix_extension_across_turns(renderer, tokenizer):
    ids = [assert_exact(renderer, tokenizer, conv, tools=TOOLS) for conv in (C1, C2, C3, C4, C5)]
    for shorter, longer in zip(ids, ids[1:]):
        assert_prefix(shorter, longer)
    assert renderer.has_extension_property is True


def test_prefix_extension_with_sampled_tokens_roundtrip(renderer, tokenizer):
    """prompt(N) + sampled tokens is a prefix of prompt(N + parsed reply + tool result)."""
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    sampled_text = (
        "The user wants the Paris forecast.\nI should call get_weather.\n</think>\n\n"
        "Let me check the forecast.\n\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        '<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{"unit": "c", "x": [1, 2]}\n</parameter>\n'
        "<parameter=verbose>\ntrue\n</parameter>\n</function>\n</tool_call>"
    )
    sampled = tokenizer.encode(sampled_text, add_special_tokens=False) + im_end
    prompt = renderer.build_generation_prompt(C1, tools=TOOLS).to_ints()

    message, ok = renderer.parse_response(sampled, tools=TOOLS)
    assert ok
    assert message["reasoning_content"] == "The user wants the Paris forecast.\nI should call get_weather."
    assert message["content"] == "Let me check the forecast."
    assert [tc.function.name for tc in message["tool_calls"]] == ["get_weather"]
    assert json.loads(message["tool_calls"][0].function.arguments) == {
        "city": "Paris",
        "days": 3,
        "opts": {"unit": "c", "x": [1, 2]},
        "verbose": True,
    }

    echoed = renderer.to_openai_message(message)
    assert echoed["reasoning_content"] == message["reasoning_content"]
    assert echoed["tool_calls"][0]["function"]["name"] == "get_weather"
    assert isinstance(echoed["tool_calls"][0]["function"]["arguments"], str)
    next_conv = C1 + [echoed, tool(echoed["tool_calls"][0]["id"], "get_weather", "Sunny")]

    next_prompt = assert_exact(renderer, tokenizer, next_conv, tools=TOOLS)
    assert_prefix(prompt + sampled, next_prompt)


def test_empty_reasoning_breaks_token_prefix_but_matches_hf(renderer, tokenizer):
    """Template quirk: an assistant turn with empty reasoning renders "<think>\\n\\n</think>" whose
    "\\n\\n" is one token, so it cannot extend the "<think>\\n" generation prompt token-wise."""
    prompt = renderer.build_generation_prompt(C1, tools=TOOLS).to_ints()
    conv = C1 + [asst("", "Sunny.", [("get_weather", {"city": "Paris"})]), tool("call_0", "get_weather", "Sunny")]
    next_prompt = assert_exact(renderer, tokenizer, conv, tools=TOOLS)
    assert next_prompt[: len(prompt)] != prompt
    assert next_prompt[: len(prompt) - 1] == prompt[:-1]  # diverges exactly at the "\n" after <think>


# --------------------------------------------------------------------------- parse_response


def sample(tokenizer, text: str, stop: str | None = "<|im_end|>") -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if stop:
        ids += tokenizer.encode(stop, add_special_tokens=False)
    return ids


def test_parse_think_and_two_tool_calls(renderer, tokenizer):
    text = (
        "Need both cities.\n</think>\n\nChecking both.\n\n"
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=execute_bash>\n<parameter=command>\ndate\necho done\n</parameter>\n"
        "<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>"
    )
    message, ok = renderer.parse_response(sample(tokenizer, text), tools=TOOLS)
    assert ok
    assert message["role"] == "assistant"
    assert message["reasoning_content"] == "Need both cities."
    assert message["content"] == "Checking both."
    calls = message["tool_calls"]
    assert [tc.function.name for tc in calls] == ["get_weather", "execute_bash"]
    assert all(tc.type == "function" and tc.id and tc.id.startswith("call_") for tc in calls)
    assert json.loads(calls[0].function.arguments) == {"city": "Tokyo"}
    assert json.loads(calls[1].function.arguments) == {"command": "date\necho done", "timeout": 30}
    # Without a schema every value stays a string (AReaL/SGLang behaviour).
    untyped, _ = renderer.parse_response(sample(tokenizer, text))
    assert json.loads(untyped["tool_calls"][1].function.arguments) == {"command": "date\necho done", "timeout": "30"}
    # Re-rendering the parsed message reproduces the sampled text (modulo the prefilled <think>).
    rendered = renderer.render_conversation_text(
        C3 + [renderer.to_openai_message(message)], tools=TOOLS, add_generation_prompt=False
    )
    assert rendered.endswith("<|im_start|>assistant\n<think>\n" + text + "<|im_end|>\n")


def test_parse_plain_text(renderer, tokenizer):
    message, ok = renderer.parse_response(sample(tokenizer, "Sure, thinking...\n</think>\n\nHello there!"))
    assert ok
    assert message == {"role": "assistant", "content": "Hello there!", "reasoning_content": "Sure, thinking..."}
    # Model re-emitted the <think> tag itself.
    message, ok = renderer.parse_response(sample(tokenizer, "<think>\nagain\n</think>\n\nHi"))
    assert ok and message["reasoning_content"] == "again" and message["content"] == "Hi"


def test_parse_truncated_reasoning(renderer, tokenizer):
    message, ok = renderer.parse_response(sample(tokenizer, "still thinking about it", stop=None))
    assert not ok
    assert message["reasoning_content"] == "still thinking about it"
    assert message["content"] == ""
    assert "tool_calls" not in message


def test_parse_endoftext_stop(renderer, tokenizer):
    message, ok = renderer.parse_response(sample(tokenizer, "r\n</think>\n\nbye", stop="<|endoftext|>"))
    assert ok and message["content"] == "bye"
    assert renderer.get_stop_sequences() == tokenizer.encode("<|im_end|><|endoftext|>", add_special_tokens=False)


def test_parse_disable_thinking_variant(disable_thinking_renderer, tokenizer):
    message, ok = disable_thinking_renderer.parse_response(sample(tokenizer, "Direct answer."))
    assert ok
    assert message == {"role": "assistant", "content": "Direct answer.", "reasoning_content": ""}
    message, _ = disable_thinking_renderer.parse_response(sample(tokenizer, "<think>\nhmm\n</think>\n\nanswer"))
    assert message["reasoning_content"] == "hmm" and message["content"] == "answer"


def test_parse_malformed_tool_call_is_unparsed(renderer, tokenizer):
    message, ok = renderer.parse_response(
        sample(tokenizer, "r\n</think>\n\nText before\n\n<tool_call>\ngarbage\n</tool_call>")
    )
    assert ok
    assert "tool_calls" not in message
    assert message["content"] == "Text before"
    assert message["unparsed_tool_calls"][0].raw_text.startswith("<tool_call>")


def test_parse_zero_argument_tool_call_uses_fallback(renderer, tokenizer):
    message, ok = renderer.parse_response(
        sample(tokenizer, "r\n</think>\n\n<tool_call>\n<function=execute_bash>\n</function>\n</tool_call>"), tools=TOOLS
    )
    assert ok
    assert message["content"] == ""
    assert message["tool_calls"][0].function.name == "execute_bash"
    assert json.loads(message["tool_calls"][0].function.arguments) == {}


def test_parse_streaming_final_message(renderer, tokenizer):
    from tinker_cookbook.renderers import Message as CookbookMessage

    text = "plan\n</think>\n\nvisible"
    deltas = list(renderer.parse_response_streaming(sample(tokenizer, text)))
    final = deltas[-1]
    assert isinstance(final, dict) and final["reasoning_content"] == "plan" and final["content"] == "visible"
    assert CookbookMessage is not None


# --------------------------------------------------------------------------- supervised


def test_supervised_example_tokens_and_weights(renderer, tokenizer):
    from tinker_cookbook.renderers import TrainOnWhat

    conv = C5[:-1]  # ends with an assistant message
    model_input, weights = renderer.build_supervised_example(conv, TrainOnWhat.LAST_ASSISTANT_MESSAGE, tools=TOOLS)
    ids = model_input.to_ints()
    expected = hf_ids(tokenizer, conv, tools=TOOLS, add_generation_prompt=False)
    assert expected[-1] == tokenizer.encode("\n", add_special_tokens=False)[0]
    assert ids == expected[:-1]
    assert len(weights) == len(ids)
    last_output = tokenizer.encode(
        "Summarize the three results.\n</think>\n\nTokyo: rainy. Osaka: cloudy. Date: Mon Sep 7 2026.<|im_end|>",
        add_special_tokens=False,
    )
    assert ids[-len(last_output) :] == last_output
    assert weights[-len(last_output) :].tolist() == [1.0] * len(last_output)
    assert weights[: -len(last_output)].sum().item() == 0.0
    # A trailing tool message gets its <|im_end|> via stop_overlap.
    model_input, _ = renderer.build_supervised_example(C4, TrainOnWhat.ALL_ASSISTANT_MESSAGES, tools=TOOLS)
    assert model_input.to_ints() == hf_ids(tokenizer, C4, tools=TOOLS, add_generation_prompt=False)[:-1]


# --------------------------------------------------------------------------- registry & errors


def test_registration(tokenizer):
    from tinker_cookbook.renderers import get_registered_renderer_names, get_renderer

    from platoon.train.tinker.renderers import (
        RENDERER_NAME,
        RENDERER_NAME_DISABLE_THINKING,
        Qwen3_6PreserveThinkingDisableThinkingRenderer,
        Qwen3_6PreserveThinkingRenderer,
        register,
    )

    assert register() == [RENDERER_NAME, RENDERER_NAME_DISABLE_THINKING]
    assert register() == [RENDERER_NAME, RENDERER_NAME_DISABLE_THINKING]  # idempotent
    assert set(get_registered_renderer_names()) >= {RENDERER_NAME, RENDERER_NAME_DISABLE_THINKING}
    thinking = get_renderer(
        RENDERER_NAME, tokenizer, image_processor=None, model_name="apurvaga/Qwen3.6-35B-A3B-preserve-thinking"
    )
    assert (
        isinstance(thinking, Qwen3_6PreserveThinkingRenderer)
        and thinking.enable_thinking
        and thinking.preserve_thinking
    )
    assert thinking._renderer_name == RENDERER_NAME
    disabled = get_renderer(RENDERER_NAME_DISABLE_THINKING, tokenizer)
    assert isinstance(disabled, Qwen3_6PreserveThinkingDisableThinkingRenderer) and not disabled.enable_thinking
    # renderer_kwargs path used by register_tinker_llm: plain attributes.
    setattr(disabled, "preserve_thinking", False)
    assert disabled.build_generation_prompt(C5, tools=TOOLS).to_ints() == hf_ids(
        tokenizer, C5, tools=TOOLS, enable_thinking=False, preserve_thinking=False
    )


def test_template_errors_are_mirrored(renderer):
    from tinker_cookbook.exceptions import RendererError

    with pytest.raises(RendererError, match="No messages"):
        renderer.build_generation_prompt([])
    with pytest.raises(RendererError, match="No user query"):
        renderer.build_generation_prompt(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "<tool_response>\nx\n</tool_response>"}]
        )
    with pytest.raises(RendererError, match="System message must be at the beginning"):
        renderer.build_generation_prompt([{"role": "user", "content": "hi"}, {"role": "system", "content": "x"}])
    with pytest.raises(RendererError, match="Unexpected message role"):
        renderer.build_generation_prompt([{"role": "user", "content": "hi"}, {"role": "function", "content": "x"}])
    with pytest.raises(RendererError, match="[Ii]mage"):
        renderer.build_generation_prompt(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:..."}}]}]
        )
    with pytest.raises(RendererError, match="not valid JSON"):
        renderer.build_generation_prompt(
            C1
            + [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{oops"}}],
                }
            ]
        )


def test_cookbook_message_shapes_are_accepted(renderer, tokenizer):
    """Cookbook ToolCall objects and ThinkingPart content render like their OpenAI equivalents."""
    from tinker_cookbook.renderers import ToolCall

    cookbook_conv = C1 + [
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
    openai_conv = C1 + [
        asst("cookbook thinking", "cookbook text", [("get_weather", {"city": "Paris"})]),
        tool("call_0", "get_weather", "Sunny"),
    ]
    assert renderer.build_generation_prompt(cookbook_conv, tools=TOOLS).to_ints() == assert_exact(
        renderer, tokenizer, openai_conv, tools=TOOLS
    )
