"""Model gateway with the Qwen3.6 renderer: tool pass-through, response splitting, chat_template_kwargs, routing_id.

The gateway renders, samples and parses every policy call harnesses make (over HTTP, or in process through
``AgentContext.chat``), so these pin its OpenAI-facing behavior against the real chat template.

CPU only.  The tool/parse/chat-template tests use the Qwen3.6 preserve-thinking snapshot tokenizer and are
skipped when it is not on disk; the routing_id / extra='allow' tests use fakes and always run.  Run with

    HF_HUB_OFFLINE=1 python -m pytest tests/train/tinker -q
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
from huggingface_hub.constants import HF_HUB_CACHE  # noqa: E402

from platoon.gateway import core  # noqa: E402
from platoon.gateway.core import (  # noqa: E402
    ChatRequest,
    ChatResponse,
    ContextWindowExceededError,
    ModelGateway,
    RenderRequest,
    TinkerLLMInteraction,
    as_messages,
    routing_id_of,
    to_cookbook_tool_specs,
)

# In the standard Hugging Face cache (HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub).
SNAPSHOT = Path(
    os.environ.get(
        "QWEN36_PRESERVE_THINKING_SNAPSHOT",
        Path(HF_HUB_CACHE)
        / "models--apurvaga--Qwen3.6-35B-A3B-preserve-thinking/snapshots/ce815485d640a33461c25e3d5c505b5346e8c857",
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


class Client:
    """One policy agent on a gateway, called the way a harness calls it (Chat Completions parameters)."""

    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway
        self.agent = gateway.open_agent(rollout_id="rollout", trajectory_id="root", interactions={})

    def render(self, messages, params=None) -> tinker.ModelInput:
        return self.gateway.render(as_messages(messages), RenderRequest.from_params(params or {}))

    def parse(self, prompt, response, *, tools=None) -> ChatResponse:
        return self.gateway.parse(prompt, response, RenderRequest(tools=tools))

    async def achat(self, messages, params=None) -> ChatResponse:
        return await self.gateway.chat(self.agent, ChatRequest.from_params(messages, params or {}))

    def chat(self, messages, params=None) -> ChatResponse:
        return asyncio.run(self.achat(messages, params))


def make_client(renderer, tokenizer, response, **kwargs) -> tuple[Client, FakeSamplingClient]:
    sampler = FakeSamplingClient(response)
    gateway = ModelGateway(sampler=sampler, renderer=renderer, tokenizer=tokenizer, model_name="fake", **kwargs)
    return Client(gateway), sampler


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(SNAPSHOT))


@pytest.fixture(scope="module")
def renderer(tokenizer):
    from tinker_cookbook.renderers import get_renderer

    from platoon.train.tinker.renderers import register

    register()  # idempotent; nothing else in this module imports the proxy, which registers platoon's renderers
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
    core.patch_tinker_sample_response_extra_allow()
    assert tinker.types.SampleResponse is before
    monkeypatch.setenv("TINKER_SUBPROCESS_SAMPLING", "1")
    with pytest.raises(AssertionError, match="TINKER_SUBPROCESS_SAMPLING"):
        core.patch_tinker_sample_response_extra_allow()


def test_interaction_dataclass_has_routing_id_default_none():
    interaction = TinkerLLMInteraction(obs=tinker.ModelInput.from_ints([1]), action=None)  # type: ignore[arg-type]
    assert interaction.routing_id is None


@pytest.mark.parametrize("routing_id", [4242, None])
def test_chat_records_routing_id(routing_id):
    client, sampler = make_client(FakeRenderer(), FakeTokenizer(), make_response([5, 0], routing_id=routing_id))
    result = client.chat([{"role": "user", "content": "hi"}])
    interaction = client.agent.interactions[result.call_id]
    assert interaction.routing_id == routing_id
    assert interaction.action.tokens == [5, 0]
    assert interaction.action.maybe_logprobs == [-0.5, -0.5]
    assert interaction.obs.to_ints() == [2]
    assert sampler.calls[0]["num_samples"] == 1
    assert result.finish_reason == "stop"


def test_routing_id_of_ignores_garbage():
    assert routing_id_of(make_response([1], routing_id=12)) == 12
    assert routing_id_of(make_response([1])) is None
    weird = tinker.types.SampleResponse.model_validate(
        {"sequences": [{"stop_reason": "stop", "tokens": [1], "routing_id": "77"}], "type": "sample"}
    )
    assert routing_id_of(weird) == 77
    bad = tinker.types.SampleResponse.model_validate(
        {"sequences": [{"stop_reason": "stop", "tokens": [1], "routing_id": "x"}], "type": "sample"}
    )
    assert routing_id_of(bad) is None


# --------------------------------------------------------------------------- render request extraction


def test_render_request_reads_extra_body_and_top_level():
    req = RenderRequest.from_params(
        {
            "tools": TOOLS,
            "tool_choice": "auto",
            "extra_body": {
                "reasoning_effort": "high",
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
            },
        }
    )
    assert req == RenderRequest(tools=TOOLS, render_kwargs={"enable_thinking": True, "preserve_thinking": False})
    # top-level chat_template_kwargs override extra_body's
    req = RenderRequest.from_params(
        {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    assert req.render_kwargs == {"enable_thinking": False}
    # reasoning_effort none -> enable_thinking False (extra_body or top-level)
    assert RenderRequest.from_params({"extra_body": {"reasoning_effort": "none"}}).render_kwargs == {
        "enable_thinking": False
    }
    assert RenderRequest.from_params({"reasoning_effort": "none"}).render_kwargs == {"enable_thinking": False}
    assert RenderRequest.from_params({}) == RenderRequest(tools=None, render_kwargs=None)


@pytest.mark.parametrize("tool_choice", ["required", {"type": "function", "function": {"name": "get_weather"}}])
def test_tool_choice_required_or_named_rejected(tool_choice):
    client, _ = make_client(FakeRenderer(), FakeTokenizer(), make_response([1]))
    with pytest.raises(ValueError, match="tool_choice"):
        client.render([{"role": "user", "content": "hi"}], {"tools": TOOLS, "tool_choice": tool_choice})


def test_cookbook_renderer_gets_tools_via_conversation_prefix():
    """Renderers without a ``tools`` kwarg get the cookbook prefix folded into the system message."""
    client, _ = make_client(FakeRenderer(), FakeTokenizer(), make_response([1]))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    model_input = client.render(messages, {"tools": TOOLS})
    # cookbook ToolSpec is flat {name, description, parameters}; the system content became
    # "sys" + json.dumps(specs) and FakeRenderer emits len(content) per message
    specs = to_cookbook_tool_specs(TOOLS)
    assert [spec["name"] for spec in specs] == ["get_weather", "execute_bash"]
    assert set(specs[0]) == {"name", "description", "parameters"}
    assert specs[0]["parameters"] == TOOLS[0]["function"]["parameters"]
    assert model_input.to_ints() == [len("sys" + json.dumps(specs)), 2]
    without_system = client.render(messages[1:], {"tools": TOOLS})
    assert without_system.to_ints() == [len(json.dumps(specs)), 2]
    with pytest.raises(ValueError, match="Invalid tool definition"):
        to_cookbook_tool_specs([{"type": "function"}])


# --------------------------------------------------------------------------- snapshot-backed: tools, parse, kwargs


@needs_snapshot
def test_renderer_registered_at_policy_import(tokenizer, renderer):
    from tinker_cookbook.renderers import is_renderer_registered

    import platoon.train.tinker.proxy  # noqa: F401  (registers platoon's renderers on import)
    from platoon.train.tinker.renderers import ChatTemplateRenderer

    assert is_renderer_registered("qwen3_6_preserve_thinking")
    assert isinstance(renderer, ChatTemplateRenderer) and renderer.preserve_thinking is True


@needs_snapshot
def test_tools_reach_renderer_and_match_chat_template(tokenizer, renderer):
    client, _ = make_client(renderer, tokenizer, make_response([1]))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What's the weather in Paris?"}]
    model_input = client.render(messages, {"tools": TOOLS, "tool_choice": "auto"})
    ids = model_input.to_ints()
    assert ids == hf_ids(tokenizer, messages, tools=TOOLS)
    text = tokenizer.decode(ids)
    assert "# Tools" in text and "<tools>" in text
    assert json.dumps(TOOLS[0], ensure_ascii=False) in text
    assert "get_weather" in text and "execute_bash" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    # Without tools the schema block is absent (and equals the template without tools).
    plain = client.render(messages)
    assert plain.to_ints() == hf_ids(tokenizer, messages)
    assert "<tools>" not in tokenizer.decode(plain.to_ints())


@needs_snapshot
def test_chat_template_kwargs_from_extra_body_honoured(tokenizer, renderer):
    client, _ = make_client(renderer, tokenizer, make_response([1]))
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
    ids = client.render(messages, condenser_params).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=True, preserve_thinking=False)
    assert "thinking hard" not in tokenizer.decode(ids)  # preserve_thinking=False drops history reasoning
    # reasoning_effort none -> enable_thinking False -> empty think block in the generation prompt
    disabled = {"extra_body": {"reasoning_effort": "none", "chat_template_kwargs": {"preserve_thinking": False}}}
    ids = client.render(messages, disabled).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=False, preserve_thinking=False)
    assert tokenizer.decode(ids).endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # top-level chat_template_kwargs (no extra_body) works too
    ids = client.render(messages, {"chat_template_kwargs": {"enable_thinking": False}}).to_ints()
    assert ids == hf_ids(tokenizer, messages, enable_thinking=False)
    # default request: preserve thinking, thinking enabled
    ids = client.render(messages).to_ints()
    assert ids == hf_ids(tokenizer, messages)
    assert "thinking hard" in tokenizer.decode(ids)


@needs_snapshot
def test_nonthinking_summary_is_visible_content(tokenizer, renderer, im_end):
    tokens = tokenizer.encode("Summary of completed work.", add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens))
    result = client.chat(
        [{"role": "user", "content": "Summarize the completed work."}], {"extra_body": {"reasoning_effort": "none"}}
    )
    assert result.content == "Summary of completed work."
    assert result.reasoning_content is None
    assert result.finish_reason == "stop"
    assert result.tokens == tokens
    assert renderer.enable_thinking is True


@needs_snapshot
@pytest.mark.parametrize("utility_first", [False, True])
def test_concurrent_policy_and_summary_keep_request_local_thinking(tokenizer, renderer, im_end, utility_first):
    policy_tokens = tokenizer.encode("Unfinished policy reasoning", add_special_tokens=False)
    summary_tokens = tokenizer.encode("Completed work summary.", add_special_tokens=False) + [im_end]
    client, sampler = make_client(renderer, tokenizer, make_response(summary_tokens))

    async def run():
        both_started = asyncio.Event()
        calls = []

        async def sample_async(*, prompt, sampling_params, num_samples):
            assert renderer.enable_thinking is True
            utility = tokenizer.decode(prompt.to_ints()).endswith("<think>\n\n</think>\n\n")
            calls.append(utility)
            if len(calls) == 2:
                both_started.set()
            await both_started.wait()
            return make_response(
                summary_tokens if utility else policy_tokens, stop_reason="stop" if utility else "length"
            )

        sampler.sample_async = sample_async
        modes = [utility_first, not utility_first]
        results = await asyncio.gather(
            *(
                client.achat(
                    [{"role": "user", "content": "Continue."}],
                    {"chat_template_kwargs": {"enable_thinking": False}} if utility else {},
                )
                for utility in modes
            )
        )
        return dict(zip(modes, results, strict=True))

    results = asyncio.run(run())
    assert client.agent.interactions[results[True].call_id].action.tokens == summary_tokens
    assert client.agent.interactions[results[False].call_id].action.tokens == policy_tokens
    summary, policy = results[True], results[False]
    assert summary.content == "Completed work summary."
    assert summary.reasoning_content is None
    assert policy.content == ""
    assert policy.reasoning_content == "Unfinished policy reasoning"
    assert policy.finish_reason == "length"
    assert not policy.tool_calls
    assert renderer.enable_thinking is True


@needs_snapshot
def test_parse_xml_tool_call_to_openai_tool_calls(tokenizer, renderer, im_end):
    completion = f"plan the call\n</think>\n\nI'll check the forecast.\n\n{XML_TOOL_CALL}"
    tokens = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens, routing_id=7))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather in Paris?"}]
    result = client.chat(messages, {"tools": TOOLS})
    assert client.agent.interactions[result.call_id].routing_id == 7
    assert client.agent.interactions[result.call_id].action.tokens == tokens
    assert result.finish_reason == "tool_calls"
    message = result.message()
    assert message["role"] == "assistant"
    assert message["content"] == "I'll check the forecast."
    assert message["reasoning_content"] == "plan the call"
    assert len(message["tool_calls"]) == 1
    call = message["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": 3}  # schema-typed via tools
    assert result.tokens == tokens
    assert result.usage()["completion_tokens"] == len(tokens)
    assert result.prompt.to_ints() == client.render(messages, {"tools": TOOLS}).to_ints()


@needs_snapshot
def test_parse_without_tool_schema_keeps_string_arguments(tokenizer, renderer, im_end):
    tokens = tokenizer.encode(f"\n</think>\n\n{XML_TOOL_CALL}", add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens))
    result = client.parse(tinker.ModelInput.from_ints([1]), make_response(tokens), tools=None)
    call = result.tool_calls[0]
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "days": "3"}
    assert result.content == ""
    assert result.reasoning_content is None
    assert result.finish_reason == "tool_calls"


@needs_snapshot
def test_reasoning_content_split_and_finish_reasons(tokenizer, renderer, im_end):
    client, _ = make_client(renderer, tokenizer, make_response([1]))
    # A real generation prompt: it ends in <think>, so the response starts inside the reasoning block.
    prompt = client.render([{"role": "user", "content": "What is the answer?"}], {})
    # thinking + visible text, clean stop
    tokens = tokenizer.encode("let me think\n</think>\n\nThe answer is 42.", add_special_tokens=False) + [im_end]
    result = client.parse(prompt, make_response(tokens))
    assert (result.content, result.reasoning_content) == ("The answer is 42.", "let me think")
    assert result.tool_calls is None
    assert result.finish_reason == "stop"
    # truncated inside the think block: reasoning only, finish_reason length
    tokens = tokenizer.encode("still thinking about", add_special_tokens=False)
    result = client.parse(prompt, make_response(tokens, stop_reason="length"))
    assert result.content == ""
    assert result.reasoning_content == "still thinking about"
    assert result.finish_reason == "length"
    # a tool call cut off by max_tokens still has stop_reason length: no parsed call -> length
    tokens = tokenizer.encode("\n</think>\n\n<tool_call>\n<function=get_we", add_special_tokens=False)
    result = client.parse(prompt, make_response(tokens, stop_reason="length"), tools=TOOLS)
    assert result.finish_reason == "length"
    assert not result.tool_calls


@needs_snapshot
def test_length_never_executes_tools_and_keeps_sampling_attribution(tokenizer, renderer, im_end):
    # Even a parseable tool call is withheld when the sampling server reports
    # a cutoff. Never execute a complete prefix of an incomplete multi-call.
    tokens = tokenizer.encode("Ready\n</think>\n" + XML_TOOL_CALL, add_special_tokens=False) + [im_end]
    client, sampler = make_client(renderer, tokenizer, make_response(tokens, stop_reason="length", routing_id=123))
    response = client.chat([{"role": "user", "content": "Check weather"}], {"tools": TOOLS})
    interaction = client.agent.interactions[response.call_id]
    assert len(sampler.calls) == 1
    assert interaction.action.tokens == tokens
    assert interaction.action.maybe_logprobs == [-0.5] * len(tokens)
    assert interaction.routing_id == 123
    assert response.finish_reason == "length"
    assert not response.tool_calls
    assert response.generation == {
        "stop_reason": "length",
        "generated_tokens": len(tokens),
        "parse_success": True,
        "parsed_tool_calls": 1,
        "emitted_tool_calls": 0,
        "recovered_tool_calls": 0,
        "unparsed_tool_calls": 0,
    }


@needs_snapshot
def test_every_call_of_a_response_reaches_the_harness(tokenizer, renderer, im_end):
    """A call with no arguments next to others, and calls after a think block that never closed, are executed."""
    zero_arguments = "<tool_call>\n<function=execute_bash>\n</function>\n</tool_call>"
    for completion in (
        f"Two calls.\n</think>\n\n{XML_TOOL_CALL}\n{zero_arguments}",
        f"Two calls, no close tag.\n\n{XML_TOOL_CALL}\n{zero_arguments}",
    ):
        tokens = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
        client, _ = make_client(renderer, tokenizer, make_response(tokens))
        response = client.chat([{"role": "user", "content": "Check weather"}], {"tools": TOOLS})
        assert [call["function"]["name"] for call in response.tool_calls] == ["get_weather", "execute_bash"]
        assert json.loads(response.tool_calls[1]["function"]["arguments"]) == {}
        assert response.content == "" and response.reasoning_content.startswith("Two calls")
        assert response.generation["parsed_tool_calls"] == response.generation["emitted_tool_calls"] == 2


@needs_snapshot
def test_unparsed_calls_stay_in_the_content_and_are_counted(tokenizer, renderer, im_end):
    cut_off = "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls"
    completion = f"plan\n</think>\n\nChecking.\n\n{XML_TOOL_CALL}\n{cut_off}"
    tokens = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens))
    response = client.chat([{"role": "user", "content": "Check weather"}], {"tools": TOOLS})
    assert [call["function"]["name"] for call in response.tool_calls] == ["get_weather"]
    assert response.content == f"Checking.\n\n{cut_off}"
    assert (response.generation["unparsed_tool_calls"], response.generation["recovered_tool_calls"]) == (1, 0)
    # A missing close tag is read by a lenient rule, and counted.
    lenient = XML_TOOL_CALL.replace("Paris\n</parameter>\n", "Paris\n")
    tokens = tokenizer.encode(f"plan\n</think>\n\n{lenient}", add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens))
    response = client.chat([{"role": "user", "content": "Check weather"}], {"tools": TOOLS})
    assert json.loads(response.tool_calls[0]["function"]["arguments"]) == {"city": "Paris", "days": 3}
    assert (response.generation["unparsed_tool_calls"], response.generation["recovered_tool_calls"]) == (0, 1)


@needs_snapshot
def test_the_prompt_decides_whether_a_response_starts_in_reasoning(tokenizer, renderer, im_end):
    tokens = tokenizer.encode("The answer is 42.", add_special_tokens=False) + [im_end]
    client, _ = make_client(renderer, tokenizer, make_response(tokens))
    thinking = client.chat([{"role": "user", "content": "Answer?"}])
    assert (thinking.content, thinking.reasoning_content) == ("", "The answer is 42.")
    direct = client.chat([{"role": "user", "content": "Answer?"}], {"chat_template_kwargs": {"enable_thinking": False}})
    assert (direct.content, direct.reasoning_content) == ("The answer is 42.", None)


@needs_snapshot
def test_multi_turn_prompt_extends_with_reasoning_content(tokenizer, renderer, im_end):
    """Turn k's prompt + sampled tokens is a prefix of turn k+1's prompt when reasoning_content is re-sent."""
    client, _ = make_client(renderer, tokenizer, make_response([1]))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Weather in Paris?"}]
    turn1 = client.render(messages, {"tools": TOOLS}).to_ints()
    completion = f"plan\n</think>\n\n{XML_TOOL_CALL}"
    sampled = tokenizer.encode(completion, add_special_tokens=False) + [im_end]
    assistant = client.parse(tinker.ModelInput.from_ints(turn1), make_response(sampled), tools=TOOLS).message()
    tool_msg = {
        "role": "tool",
        "tool_call_id": assistant["tool_calls"][0]["id"],
        "name": "get_weather",
        "content": "Sunny, 25C",
    }
    turn2 = client.render(messages + [assistant, tool_msg], {"tools": TOOLS}).to_ints()
    assert turn2 == hf_ids(tokenizer, messages + [assistant, tool_msg], tools=TOOLS)
    assert turn2[: len(turn1) + len(sampled)] == turn1 + sampled


# --------------------------------------------------------------------------- context window fit


class _WideRenderer(FakeRenderer):
    """FakeRenderer whose prompt has one token per character of content (tests long prompts cheaply)."""

    def build_generation_prompt(self, messages, role="assistant", prefill=None):
        return tinker.ModelInput.from_ints([1] * sum(len(m.get("content") or "") for m in messages))


def _long_messages(n_tokens: int):
    return [{"role": "user", "content": "x" * n_tokens}]


def test_max_tokens_clamped_to_remaining_context():
    client, sampler = make_client(_WideRenderer(), FakeTokenizer(), make_response([1]), context_window_length=65536)
    client.chat(_long_messages(40000), {"max_tokens": 26214})
    client.chat(_long_messages(40000), {"max_completion_tokens": 26214})
    assert [call["sampling_params"].max_tokens for call in sampler.calls] == [65536 - 40000] * 2


def test_max_tokens_unchanged_when_it_fits():
    client, sampler = make_client(_WideRenderer(), FakeTokenizer(), make_response([1]), context_window_length=65536)
    client.chat(_long_messages(1000), {"max_tokens": 4096})
    assert sampler.calls[0]["sampling_params"].max_tokens == 4096


def test_no_context_window_means_no_clamp():
    client, sampler = make_client(_WideRenderer(), FakeTokenizer(), make_response([1]))
    assert client.gateway.context_window_length is None
    client.chat(_long_messages(70000), {"max_tokens": 26214})
    assert sampler.calls[0]["sampling_params"].max_tokens == 26214


@pytest.mark.parametrize("prompt_tokens", [65536, 70000])
def test_prompt_overflow_raises_context_window_exceeded(prompt_tokens):
    client, sampler = make_client(_WideRenderer(), FakeTokenizer(), make_response([1]), context_window_length=65536)
    with pytest.raises(ContextWindowExceededError):
        client.chat(_long_messages(prompt_tokens), {"max_tokens": 1})
    assert sampler.calls == []  # never sampled
    assert client.agent.context_overflows == 1


# --------------------------------------------------------------------------- seed


def test_seed_only_set_when_request_carries_one():
    client, sampler = make_client(FakeRenderer(), FakeTokenizer(), make_response([1]))
    assert client.gateway.seed is None
    client.chat([{"role": "user", "content": "hi"}])
    client.chat([{"role": "user", "content": "hi"}], {"seed": 7})
    assert sampler.calls[0]["sampling_params"].seed is None
    assert sampler.calls[1]["sampling_params"].seed == 7
