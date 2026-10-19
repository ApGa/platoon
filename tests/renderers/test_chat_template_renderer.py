"""The chat-template renderer's contract: names and presets, gateway signatures, stop tokens, cleanup and parsing.

Prompt tokens are checked against the saved fixtures in ``test_renderer_fixtures.py``.
"""

from __future__ import annotations

import inspect
import json
import os

import pytest
import renderer_cases as cases

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

pytestmark = pytest.mark.skipif(
    not all((path / "tokenizer.json").exists() for path in cases.SNAPSHOTS.values()),
    reason=f"Qwen3.6/3.8 preserve-thinking snapshots not available under {cases.HUB}",
)

TOOLS = cases.TOOLS


@pytest.fixture(scope="module")
def tokenizer():
    return cases.load_tokenizer("qwen3_6")


@pytest.fixture(scope="module")
def tokenizer38():
    return cases.load_tokenizer("qwen3_8")


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {})


@pytest.fixture(scope="module")
def disable_thinking_renderer(tokenizer):
    return cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking_disable_thinking", {})


def sample(tokenizer, text: str, stop: str | None = "<|im_end|>") -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if stop:
        ids += tokenizer.encode(stop, add_special_tokens=False)
    return ids


# --------------------------------------------------------------------------- names, presets and contract


def test_aliases_are_the_chat_template_renderer_with_presets(tokenizer, tokenizer38):
    from tinker_cookbook.renderers import get_registered_renderer_names

    from platoon.train.tinker.renderers import ALIASES, RENDERER_NAME, ChatTemplateRenderer, register

    names = register()
    assert register() == names  # idempotent
    assert set(names) == {RENDERER_NAME, *ALIASES} and set(get_registered_renderer_names()) >= set(names)
    expected = {
        "chat_template": (True, None, None),
        "qwen3_6_preserve_thinking": (True, True, None),
        "qwen3_6_preserve_thinking_disable_thinking": (False, True, None),
        "qwen3_8_preserve_thinking": (True, True, "xhigh"),
        "qwen3_8_preserve_thinking_disable_thinking": (False, True, "xhigh"),
    }
    for name, settings in expected.items():
        renderer = cases.build_renderer(tokenizer38 if "3_8" in name else tokenizer, name, {})
        assert isinstance(renderer, ChatTemplateRenderer)
        assert (renderer.enable_thinking, renderer.preserve_thinking, renderer.reasoning_effort) == settings
        assert renderer._renderer_name == name
    # renderer_kwargs are plain attributes (build_tinker_llm sets them).
    medium = cases.build_renderer(tokenizer38, "qwen3_8_preserve_thinking", {"reasoning_effort": "medium"})
    assert medium.template_kwargs() == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    }
    assert medium.template_kwargs(enable_thinking=False, preserve_thinking=False) == {
        "enable_thinking": False,
        "preserve_thinking": False,
        "reasoning_effort": "medium",
    }
    with pytest.raises(ValueError, match="does not support"):
        cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {"no_such_setting": 1})


def test_gateway_contract_signatures(renderer):
    """ModelGateway routes on these parameter names; **kwargs would send it down the cookbook branch."""
    build = inspect.signature(renderer.build_generation_prompt).parameters
    assert {"tools", "enable_thinking", "preserve_thinking"} <= set(build)
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in build.values())
    assert {"tools", "enable_thinking"} <= set(inspect.signature(renderer.parse_response).parameters)
    assert renderer.has_extension_property is False


def test_stop_tokens_come_from_generation_config(tokenizer, tokenizer38, renderer):
    expected = tokenizer.encode("<|im_end|><|endoftext|>", add_special_tokens=False)
    assert renderer.get_stop_sequences() == expected == [248046, 248044]
    assert cases.build_renderer(tokenizer38, "qwen3_8_preserve_thinking", {}).get_stop_sequences() == expected


def test_cleanup_refuses_what_the_template_cannot_render(renderer):
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
    with pytest.raises(RendererError, match="[Vv]ideo"):
        renderer.build_generation_prompt([{"role": "user", "content": [{"type": "video", "video": "v.mp4"}]}])
    bad_arguments = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{oops"}}],
    }
    with pytest.raises(RendererError, match="not valid JSON"):
        renderer.build_generation_prompt(cases.C1 + [bad_arguments])


# --------------------------------------------------------------------------- parse_response


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
    # Re-rendering the parsed message (cookbook ToolCall objects) reproduces the sampled text.
    rendered = renderer.render_text(cases.C3 + [message], tools=TOOLS)
    assert "<|im_start|>assistant\n<think>\n" + text + "<|im_end|>\n" in rendered


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


def test_parse_disable_thinking_variant(disable_thinking_renderer, tokenizer):
    message, ok = disable_thinking_renderer.parse_response(sample(tokenizer, "Direct answer."))
    assert ok
    assert message == {"role": "assistant", "content": "Direct answer.", "reasoning_content": ""}
    message, _ = disable_thinking_renderer.parse_response(sample(tokenizer, "<think>\nhmm\n</think>\n\nanswer"))
    assert message["reasoning_content"] == "hmm" and message["content"] == "answer"


@pytest.mark.parametrize("default_thinking, request_thinking", [(True, False), (False, True), (True, None)])
def test_parse_follows_the_request_thinking(tokenizer38, default_thinking, request_thinking):
    renderer = cases.build_renderer(tokenizer38, "qwen3_8_preserve_thinking", {"enable_thinking": default_thinking})
    tokens = tokenizer38.encode("Summary of completed work.<|im_end|>", add_special_tokens=False)
    message, success = renderer.parse_response(tokens, enable_thinking=request_thinking)
    effective = default_thinking if request_thinking is None else request_thinking
    assert success
    assert message["content"] == ("" if effective else "Summary of completed work.")
    assert message["reasoning_content"] == ("Summary of completed work." if effective else "")
    assert renderer.enable_thinking == default_thinking


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


# --------------------------------------------------------------------------- the tool-call parser


XML_CALL = "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"


def test_the_parser_is_a_renderer_setting(tokenizer, caplog):
    from platoon.train.tinker.renderers import ToolCallParserError

    auto = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {})
    assert (auto.tool_call_parser, auto.reasoning_tags) == ("auto", "auto")
    assert auto.tool_call_parser_choice.record() == {
        "setting": "auto",
        "parser": "qwen3_coder",
        "verified": True,
        "reason": "round trip (one turn with every call)",
    }
    with caplog.at_level("WARNING"):
        hermes = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {"tool_call_parser": "hermes"})
        assert hermes.tool_call_parser_choice.verified is False  # set by name, so it is used anyway
    assert "does not read the chat template's own tool calls back" in caplog.text
    none = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {"tool_call_parser": "none"})
    message, _ = none.parse_response(sample(tokenizer, f"r\n</think>\n\n{XML_CALL}"), tools=TOOLS)
    assert "tool_calls" not in message and message["content"] == XML_CALL and message["reasoning_content"] == "r"
    for settings, match in (
        ({"tool_call_parser": "qwen3"}, "Unknown tool_call_parser 'qwen3'"),
        ({"tool_call_parser": ["hermes"]}, "must be a parser's name"),
        ({"reasoning_tags": ["<think>"]}, "reasoning_tags must be"),
    ):
        with pytest.raises(ToolCallParserError, match=match):
            cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", settings).resolve_tool_call_parser()


def test_a_choice_is_made_once_per_tokenizer_and_settings(tokenizer):
    first = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {}).resolve_tool_call_parser()
    second = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {}).resolve_tool_call_parser()
    assert first[0] is second[0] and first[1] is second[1]
    renderer = cases.build_renderer(tokenizer, "qwen3_6_preserve_thinking", {})
    renderer.tool_call_parser = "none"  # renderer_kwargs are set after construction; the choice follows them
    assert renderer.tool_call_parser_choice.parser == "none"


def test_the_prompt_says_whether_a_response_starts_in_reasoning(renderer, tokenizer):
    thinking = renderer.build_generation_prompt([{"role": "user", "content": "Hi"}])
    direct = renderer.build_generation_prompt([{"role": "user", "content": "Hi"}], enable_thinking=False)
    tokens = sample(tokenizer, "Just an answer.")
    message, _ = renderer.parse_response(tokens, prompt=thinking)
    assert (message["reasoning_content"], message["content"]) == ("Just an answer.", "")
    message, _ = renderer.parse_response(tokens, prompt=direct)
    assert (message["reasoning_content"], message["content"]) == ("", "Just an answer.")
    message, _ = renderer.parse_response(tokens, prompt=direct.to_ints())  # token ids work too
    assert message["content"] == "Just an answer."


class RaisingParser:
    """A user parser with a bug."""

    def parse(self, text, *, tools=None, in_reasoning=False):
        raise KeyError("a bug")


class NotJsonParser:
    """A user parser that returns arguments JSON cannot hold."""

    def parse(self, text, *, tools=None, in_reasoning=False):
        from platoon.train.tinker.renderers.tool_parsers import ParsedToolCall, ParseResult

        return ParseResult(tool_calls=[ParsedToolCall(name="execute_bash", arguments={"command": float("nan")})])


@pytest.mark.parametrize("parser", ["RaisingParser", "NotJsonParser"])
def test_a_parser_that_fails_leaves_the_whole_response_unparsed(tokenizer, parser):
    renderer = cases.build_renderer(
        tokenizer, "qwen3_6_preserve_thinking", {"tool_call_parser": f"{__name__}:{parser}"}
    )
    text = f"r\n</think>\n\n{XML_CALL}"
    message, success = renderer.parse_response(sample(tokenizer, text), tools=TOOLS)
    assert success and "tool_calls" not in message and (message["content"], message["reasoning_content"]) == ("", "")
    (unparsed,) = message["unparsed_tool_calls"]
    assert unparsed.raw_text == text and unparsed.error.startswith("the tool-call parser failed")
