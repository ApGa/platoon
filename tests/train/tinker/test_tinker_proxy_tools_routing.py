"""TinkerLLM proxy: tool pass-through, response splitting, chat_template_kwargs and routing_id capture.

CPU only.  The tool/parse/chat-template tests use the Qwen3.6 preserve-thinking snapshot tokenizer and are
skipped when it is not on disk; the routing_id / extra='allow' tests use fakes and always run.  Run with

    HF_HUB_OFFLINE=1 plugins/openreward/.venv/bin/python -m pytest tests/train/tinker -q
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

import tinker  # noqa: E402
import tinker.types  # noqa: E402

from platoon.train.tinker import proxy as proxy_module  # noqa: E402
from platoon.train.tinker.proxy import (  # noqa: E402
    RenderRequest,
    TinkerLLM,
    TinkerLLMInteraction,
    TinkerLLMProxySession,
    to_cookbook_tool_specs,
)

SNAPSHOT = Path(
    os.environ.get(
        "QWEN36_PRESERVE_THINKING_SNAPSHOT",
        "/groups/gai51740/aci18912it/rasp/cache/huggingface/hub/"
        "models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/"
        "ce815485d640a33461c25e3d5c505b5346e8c857",
    )
)
needs_snapshot = pytest.mark.skipif(
    not (SNAPSHOT / "tokenizer.json").exists() or not (SNAPSHOT / "chat_template.jinja").exists(),
    reason=f"Qwen3.6 preserve-thinking tokenizer snapshot not available at {SNAPSHOT}",
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Forecast horizon"},
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
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
]
SYSTEM = "You are OpenHands, an autonomous coding agent.\nBe concise."
IM_END = "<|im_end|>"

XML_TOOL_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
    "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
)


# --------------------------------------------------------------------------- helpers


def to_hf(messages):
    out = []
    for message in copy.deepcopy(messages):
        message = {k: v for k, v in message.items() if v is not None}
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
        out.append(message)
    return out


def hf_ids(tokenizer, messages, tools=None, **kwargs):
    out = tokenizer.apply_chat_template(
        to_hf(messages), tools=tools, tokenize=True, add_generation_prompt=True, **kwargs
    )
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def make_response(tokens: list[int], *, stop_reason: str = "stop", routing_id: int | None = None):
    seq: dict[str, Any] = {"stop_reason": stop_reason, "tokens": tokens, "logprobs": [-0.5] * len(tokens)}
    if routing_id is not None:
        seq["routing_id"] = routing_id
    return tinker.types.SampleResponse.model_validate({"sequences": [seq], "type": "sample"})


class FakeSamplingClient:
    """Records the prompt and returns a canned SampleResponse."""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        self.calls.append({"prompt": prompt, "sampling_params": sampling_params, "num_samples": num_samples})
        return self.response

    def sample(self, *, prompt, sampling_params, num_samples):
        self.calls.append({"prompt": prompt, "sampling_params": sampling_params, "num_samples": num_samples})
        return self.response


class FakeTokenizer:
    def batch_decode(self, token_lists):
        return [f"<{t[0]}>" for t in token_lists]


class FakeRenderer:
    """Minimal cookbook-style renderer (no tools/thinking kwargs): 1 token per message."""

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        return tinker.ModelInput.from_ints([len(m.get("content") or "") for m in messages])

    def parse_response(self, response):
        return {"role": "assistant", "content": "".join(f"<{t}>" for t in response)}, True

    def get_stop_sequences(self):
        return [0]

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        return [{"role": "system", "content": system_prompt + json.dumps(tools)}]


def make_llm(renderer, tokenizer, response, **kwargs) -> tuple[TinkerLLM, FakeSamplingClient]:
    client = FakeSamplingClient(response)
    llm = TinkerLLM(model_name="fake", renderer=renderer, tokenizer=tokenizer, sampling_client=client, **kwargs)
    return llm, client


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


@pytest.fixture(scope="module")
def renderer(tokenizer):
    from tinker_cookbook.renderers import get_renderer

    return get_renderer("qwen3_6_preserve_thinking", tokenizer)


@pytest.fixture
def im_end(tokenizer):
    (token,) = tokenizer.encode(IM_END, add_special_tokens=False)
    return token


# --------------------------------------------------------------------------- extra='allow' + routing_id


def test_sample_response_extra_allow_keeps_routing_id():
    response = tinker.types.SampleResponse.model_validate(
        {
            "sequences": [{"stop_reason": "stop", "tokens": [1, 2], "logprobs": [0.0, -1.0], "routing_id": 4242}],
            "type": "sample",
        }
    )
    assert getattr(response.sequences[0], "routing_id", None) == 4242
    assert response.sequences[0].tokens == [1, 2]
    # The rebinding is visible everywhere the SDK resolves the class at call time.
    assert tinker.types.sample_response.SampleResponse is tinker.types.SampleResponse
    assert tinker.SampleResponse is tinker.types.SampleResponse
    assert tinker.types.sampled_sequence.SampledSequence is tinker.types.SampledSequence
    from tinker.lib.public_interfaces import sampling_client as sc

    assert sc.types.SampleResponse is tinker.types.SampleResponse
    # Still frozen, still a subclass of the SDK's own model; sequences without the key parse as before.
    plain = tinker.types.SampleResponse.model_validate({"sequences": [{"stop_reason": "length", "tokens": [3]}]})
    assert getattr(plain.sequences[0], "routing_id", None) is None
    with pytest.raises(Exception):
        plain.sequences[0].tokens = [9]  # type: ignore[misc]


def test_patch_is_idempotent_and_refuses_subprocess_sampling(monkeypatch):
    before = tinker.types.SampleResponse
    proxy_module._patch_tinker_sample_response_extra_allow()
    assert tinker.types.SampleResponse is before
    monkeypatch.setenv("TINKER_SUBPROCESS_SAMPLING", "1")
    with pytest.raises(AssertionError, match="TINKER_SUBPROCESS_SAMPLING"):
        proxy_module._patch_tinker_sample_response_extra_allow()


def test_interaction_dataclass_has_routing_id_default_none():
    interaction = TinkerLLMInteraction(obs=tinker.ModelInput.from_ints([1]), action=None)  # type: ignore[arg-type]
    assert interaction.routing_id is None


@pytest.mark.parametrize("routing_id", [4242, None])
def test_acompletion_records_routing_id(routing_id):
    llm, client = make_llm(FakeRenderer(), FakeTokenizer(), make_response([5, 0], routing_id=routing_id))
    with TinkerLLMProxySession() as session:
        result = asyncio.run(llm.acompletion(messages=[{"role": "user", "content": "hi"}], optional_params={}))
        interaction = session.interactions[result.id]
    assert interaction.routing_id == routing_id
    assert interaction.action.tokens == [5, 0]
    assert interaction.action.maybe_logprobs == [-0.5, -0.5]
    assert interaction.obs.to_ints() == [2]
    assert client.calls[0]["num_samples"] == 1
    assert result.choices[0].finish_reason == "stop"


def test_completion_records_routing_id_sync():
    llm, _ = make_llm(FakeRenderer(), FakeTokenizer(), make_response([7], routing_id=99))
    with TinkerLLMProxySession() as session:
        result = llm.completion(messages=[{"role": "user", "content": "hi"}], optional_params={})
        assert session.interactions[result.id].routing_id == 99


def test_routing_id_of_ignores_garbage():
    assert TinkerLLM._routing_id_of(make_response([1], routing_id=12)) == 12
    assert TinkerLLM._routing_id_of(make_response([1])) is None
    weird = tinker.types.SampleResponse.model_validate(
        {"sequences": [{"stop_reason": "stop", "tokens": [1], "routing_id": "77"}], "type": "sample"}
    )
    assert TinkerLLM._routing_id_of(weird) == 77
    bad = tinker.types.SampleResponse.model_validate(
        {"sequences": [{"stop_reason": "stop", "tokens": [1], "routing_id": "x"}], "type": "sample"}
    )
    assert TinkerLLM._routing_id_of(bad) is None


# --------------------------------------------------------------------------- render request extraction


def test_extract_render_request_reads_extra_body_and_top_level():
    llm, _ = make_llm(FakeRenderer(), FakeTokenizer(), make_response([1]))
    req = llm._extract_render_request(
        {
            "optional_params": {
                "tools": TOOLS,
                "tool_choice": "auto",
                "extra_body": {
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
                },
            }
        }
    )
    assert req == RenderRequest(tools=TOOLS, render_kwargs={"enable_thinking": True, "preserve_thinking": False})
    # top-level chat_template_kwargs override extra_body's
    req = llm._extract_render_request(
        {
            "optional_params": {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
                "chat_template_kwargs": {"enable_thinking": False},
            }
        }
    )
    assert req.render_kwargs == {"enable_thinking": False}
    # reasoning_effort none -> enable_thinking False (extra_body or top-level)
    assert llm._extract_render_request(
        {"optional_params": {"extra_body": {"reasoning_effort": "none"}}}
    ).render_kwargs == {"enable_thinking": False}
    assert llm._extract_render_request({"optional_params": {"reasoning_effort": "none"}}).render_kwargs == {
        "enable_thinking": False
    }
    assert llm._extract_render_request({"optional_params": {}}) == RenderRequest(tools=None, render_kwargs=None)
    assert llm._extract_render_request({}) == RenderRequest()


@pytest.mark.parametrize("tool_choice", ["required", {"type": "function", "function": {"name": "get_weather"}}])
def test_tool_choice_required_or_named_rejected(tool_choice):
    llm, _ = make_llm(FakeRenderer(), FakeTokenizer(), make_response([1]))
    with pytest.raises(ValueError, match="tool_choice"):
        llm._prepare_model_input(
            messages=[{"role": "user", "content": "hi"}], optional_params={"tools": TOOLS, "tool_choice": tool_choice}
        )


def test_cookbook_renderer_gets_tools_via_conversation_prefix():
    """Renderers without a ``tools`` kwarg get the cookbook prefix folded into the system message."""
    llm, _ = make_llm(FakeRenderer(), FakeTokenizer(), make_response([1]))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    model_input = llm._prepare_model_input(messages=messages, optional_params={"tools": TOOLS})
    # cookbook ToolSpec is flat {name, description, parameters}; the system content became
    # "sys" + json.dumps(specs) and FakeRenderer emits len(content) per message
    specs = to_cookbook_tool_specs(TOOLS)
    assert [spec["name"] for spec in specs] == ["get_weather", "execute_bash"]
    assert set(specs[0]) == {"name", "description", "parameters"}
    assert specs[0]["parameters"] == TOOLS[0]["function"]["parameters"]
    assert model_input.to_ints() == [len("sys" + json.dumps(specs)), 2]
    without_system = llm._prepare_model_input(messages=messages[1:], optional_params={"tools": TOOLS})
    assert without_system.to_ints() == [len(json.dumps(specs)), 2]
    with pytest.raises(ValueError, match="Invalid tool definition"):
        to_cookbook_tool_specs([{"type": "function"}])


def test_condenser_optional_params_logged_once(caplog):
    llm, _ = make_llm(FakeRenderer(), FakeTokenizer(), make_response([1]))
    with caplog.at_level("INFO", logger="platoon.train.tinker.proxy"):
        llm._extract_render_request(
            {"optional_params": {"max_tokens": 26214, "extra_body": {"reasoning_effort": "high"}}}
        )
        llm._extract_render_request({"optional_params": {"max_tokens": 26214}})
    assert sum("First condenser-sized request" in r.message for r in caplog.records) == 1


# --------------------------------------------------------------------------- snapshot-backed: tools, parse, kwargs


@needs_snapshot
def test_renderer_registered_at_proxy_import(tokenizer, renderer):
    from tinker_cookbook.renderers import is_renderer_registered

    from platoon.train.tinker.renderers import Qwen3_6PreserveThinkingRenderer

    assert is_renderer_registered("qwen3_6_preserve_thinking")
    assert isinstance(renderer, Qwen3_6PreserveThinkingRenderer)


@needs_snapshot
def test_tools_reach_renderer_and_match_chat_template(tokenizer, renderer):
    llm, _ = make_llm(renderer, tokenizer, make_response([1]))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What's the weather in Paris?"}]
    model_input = llm._prepare_model_input(messages=messages, optional_params={"tools": TOOLS, "tool_choice": "auto"})
    ids = model_input.to_ints()
    assert ids == hf_ids(tokenizer, messages, tools=TOOLS)
    text = tokenizer.decode(ids)
    assert "# Tools" in text and "<tools>" in text
    assert json.dumps(TOOLS[0], ensure_ascii=False) in text
    assert "get_weather" in text and "execute_bash" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    # Without tools the schema block is absent (and equals the template without tools).
    plain = llm._prepare_model_input(messages=messages, optional_params={})
    assert plain.to_ints() == hf_ids(tokenizer, messages)
    assert "<tools>" not in tokenizer.decode(plain.to_ints())


@needs_snapshot
def test_chat_template_kwargs_from_extra_body_honoured(tokenizer, renderer):
    llm, _ = make_llm(renderer, tokenizer, make_response([1]))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Summarise."},
        {"role": "assistant", "content": "Working.", "reasoning_content": "thinking hard"},
        {"role": "user", "content": "Condense the history."},
    ]
    condenser_params = {
        "max_tokens": 26214,
        "extra_body": {
            "reasoning_effort": "high",
            "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
        },
    }
    ids = llm._prepare_model_input(messages=messages, optional_params=condenser_params).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=True, preserve_thinking=False)
    assert "thinking hard" not in tokenizer.decode(ids)  # preserve_thinking=False drops history reasoning
    # reasoning_effort none -> enable_thinking False -> empty think block in the generation prompt
    disabled = {"extra_body": {"reasoning_effort": "none", "chat_template_kwargs": {"preserve_thinking": False}}}
    ids = llm._prepare_model_input(messages=messages, optional_params=disabled).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=False, preserve_thinking=False)
    assert tokenizer.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # top-level chat_template_kwargs (no extra_body) works too
    ids = llm._prepare_model_input(
        messages=messages, optional_params={"chat_template_kwargs": {"enable_thinking": False}}
    ).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=False)
    # default request: preserve thinking, thinking enabled
    ids = llm._prepare_model_input(messages=messages, optional_params={}).to_ints()
    assert ids == hf_ids(tokenizer, messages)
    assert "thinking hard" in tokenizer.decode(ids)


@needs_snapshot
def test_parse_xml_tool_call_to_openai_tool_calls(tokenizer, renderer, im_end):
    completion = f"plan the call\n</think>\n\nI'll check the forecast.\n\n{XML_TOOL_CALL}"
    tokens = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
    llm, _ = make_llm(renderer, tokenizer, make_response(tokens, routing_id=7))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather in Paris?"}]
    with TinkerLLMProxySession() as session:
        result = asyncio.run(llm.acompletion(messages=messages, optional_params={"tools": TOOLS}))
        assert session.interactions[result.id].routing_id == 7
        assert session.interactions[result.id].action.tokens == tokens
    choice = result.choices[0]
    assert choice.finish_reason == "tool_calls"
    message = choice.message
    assert message.role == "assistant"
    assert message.content == "I'll check the forecast."
    assert message.reasoning_content == "plan the call"
    assert len(message.tool_calls) == 1
    call = message.tool_calls[0]
    assert call.type == "function"
    assert call.id
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris", "days": 3}  # schema-typed via tools
    assert choice.token_ids == tokens
    assert result.usage.completion_tokens == len(tokens)
    assert (
        result.prompt_token_ids
        == llm._prepare_model_input(messages=messages, optional_params={"tools": TOOLS}).to_ints()
    )


@needs_snapshot
def test_parse_without_tool_schema_keeps_string_arguments(tokenizer, renderer, im_end):
    tokens = tokenizer.encode(f"\n</think>\n\n{XML_TOOL_CALL}", add_special_tokens=False) + [im_end]
    llm, _ = make_llm(renderer, tokenizer, make_response(tokens))
    result = llm._parse_response(tinker.ModelInput.from_ints([1]), make_response(tokens), tools=None)
    call = result.choices[0].message.tool_calls[0]
    assert json.loads(call.function.arguments) == {"city": "Paris", "days": "3"}
    assert result.choices[0].message.content == ""
    assert getattr(result.choices[0].message, "reasoning_content", None) is None  # litellm drops the attr when None
    assert result.choices[0].finish_reason == "tool_calls"


@needs_snapshot
def test_reasoning_content_split_and_finish_reasons(tokenizer, renderer, im_end):
    llm, _ = make_llm(renderer, tokenizer, make_response([1]))
    prompt = tinker.ModelInput.from_ints([1])
    # thinking + visible text, clean stop
    tokens = tokenizer.encode("let me think\n</think>\n\nThe answer is 42.", add_special_tokens=False) + [im_end]
    choice = llm._parse_response(prompt, make_response(tokens)).choices[0]
    assert (choice.message.content, choice.message.reasoning_content) == ("The answer is 42.", "let me think")
    assert choice.message.tool_calls is None
    assert choice.finish_reason == "stop"
    # truncated inside the think block: reasoning only, finish_reason length
    tokens = tokenizer.encode("still thinking about", add_special_tokens=False)
    choice = llm._parse_response(prompt, make_response(tokens, stop_reason="length")).choices[0]
    assert choice.message.content == ""
    assert choice.message.reasoning_content == "still thinking about"
    assert choice.finish_reason == "length"
    # a tool call cut off by max_tokens still has stop_reason length: no parsed call -> length
    tokens = tokenizer.encode("\n</think>\n\n<tool_call>\n<function=get_we", add_special_tokens=False)
    choice = llm._parse_response(prompt, make_response(tokens, stop_reason="length"), tools=TOOLS).choices[0]
    assert choice.finish_reason in ("length", "tool_calls")


@needs_snapshot
def test_multi_turn_prompt_extends_with_reasoning_content(tokenizer, renderer, im_end):
    """Turn k's prompt + sampled tokens is a prefix of turn k+1's prompt when reasoning_content is re-sent."""
    llm, _ = make_llm(renderer, tokenizer, make_response([1]))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather in Paris?"}]
    turn1 = llm._prepare_model_input(messages=messages, optional_params={"tools": TOOLS}).to_ints()
    completion = f"plan\n</think>\n\n{XML_TOOL_CALL}"
    sampled = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
    parsed = (
        llm._parse_response(tinker.ModelInput.from_ints(turn1), make_response(sampled), tools=TOOLS).choices[0].message
    )
    assistant = {
        "role": "assistant",
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in parsed.tool_calls
        ],
    }
    tool_msg = {"role": "tool", "tool_call_id": parsed.tool_calls[0].id, "name": "get_weather", "content": "Sunny, 25C"}
    turn2 = llm._prepare_model_input(
        messages=messages + [assistant, tool_msg], optional_params={"tools": TOOLS}
    ).to_ints()
    assert turn2 == hf_ids(tokenizer, messages + [assistant, tool_msg], tools=TOOLS)
    assert turn2[: len(turn1) + len(sampled)] == turn1 + sampled
