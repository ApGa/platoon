"""platoon.testing's scripted policy and the real model gateway over it (``make_gateway``)."""

from __future__ import annotations

import asyncio
import json

import pytest

from platoon.gateway.core import ChatRequest, ContextWindowExceededError
from platoon.testing import (
    TOY_MODEL_NAME,
    Completion,
    ScriptedPolicy,
    ScriptExhaustedError,
    last_user_message,
    make_gateway,
    rules,
    sample_response,
    tool_call,
    toy_renderer,
    toy_tokenizer,
)

TOOLS = [{"type": "function", "function": {"name": "guess", "parameters": {"type": "object", "properties": {}}}}]


def chat(gateway, messages, **params):
    agent = gateway.open_agent(rollout_id="r", trajectory_id="t", interactions={})
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params(messages, params)))


def user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


def test_a_list_script_answers_in_order_and_records_every_call():
    policy = ScriptedPolicy(["first", Completion("second", routing_id=7, logprob=-1.0)])
    gateway = make_gateway(policy, max_tokens=50)

    first, second = chat(gateway, user("one")), chat(gateway, user("two"))

    assert (first.content, second.content) == ("first", "second")
    assert [call.last_user_message for call in policy.calls] == ["one", "two"]
    assert policy.prompts == [call.text for call in policy.calls]
    assert policy.calls[0].text.endswith("<|im_start|>user\none<|im_end|>\n<|im_start|>assistant\n")
    assert policy.calls[0].sampling_params.max_tokens == 50
    assert policy.calls[1].tokens == second.tokens == toy_tokenizer().encode("second<|im_end|>")
    assert second.routing_id == 7 and second.logprobs == [-1.0] * len(second.tokens)
    assert first.logprobs == [-0.25] * len(first.tokens)
    assert policy.remaining == []


def test_an_exhausted_script_fails_with_the_prompt():
    gateway = make_gateway(["only"])
    chat(gateway, user("one"))

    with pytest.raises(ScriptExhaustedError, match="no completion left for call 2.*two"):
        chat(gateway, user("two"))


def test_a_function_script_reads_the_prompt():
    policy = ScriptedPolicy(lambda prompt: f"echo {last_user_message(prompt)}")

    assert chat(make_gateway(policy), user("ping")).content == "echo ping"
    assert policy.remaining is None


def test_a_completion_cut_off_by_length_has_no_stop_token_and_no_tool_calls():
    policy = ScriptedPolicy([Completion(tool_call("guess", number=1), stop_reason="length")])

    response = chat(make_gateway(policy), user("go"), tools=TOOLS)

    assert response.finish_reason == "length" and response.tool_calls is None
    assert response.tokens == toy_tokenizer().encode(tool_call("guess", number=1))


def test_tool_call_writes_what_the_qwen3_renderer_parses():
    response = chat(make_gateway([tool_call("guess", {"number": 5}, hint="low")]), user("go"), tools=TOOLS)

    (call,) = response.tool_calls
    assert call["function"]["name"] == "guess"
    assert json.loads(call["function"]["arguments"]) == {"number": 5, "hint": "low"}
    assert tool_call("guess", number=5) == '<tool_call>\n{"name": "guess", "arguments": {"number": 5}}\n</tool_call>'


def test_rules_answer_by_the_last_user_message():
    reply = rules({"obs:": "FINISH", "goal": "LOOK"}, default="SHRUG")
    prompt = "<|im_start|>user\ngoal: find it<|im_end|>\n<|im_start|>user\nobs: nothing<|im_end|>\n"

    assert reply(prompt) == "FINISH"
    assert reply("<|im_start|>user\ngoal: x<|im_end|>") == "LOOK"
    assert reply("<|im_start|>user\nsomething else<|im_end|>") == "SHRUG"
    with pytest.raises(AssertionError, match="No rule matches the last user message 'something else'"):
        rules({"goal": "LOOK"})("<|im_start|>user\nsomething else<|im_end|>")


def test_last_user_message_includes_tool_results():
    prompt = toy_tokenizer().decode(
        toy_renderer()
        .build_generation_prompt(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "calling"},
                {"role": "tool", "content": "lower"},
            ]
        )
        .to_ints()
    )

    assert last_user_message(prompt) == "<tool_response>\nlower\n</tool_response>"
    assert last_user_message("no chat markers") == ""


def test_make_gateway_takes_a_renderer_by_name_and_gateway_settings():
    policy = ScriptedPolicy(["<think>hm</think>ok"])
    gateway = make_gateway(policy, renderer="qwen3", renderer_kwargs={"strip_thinking_from_history": False})

    response = chat(gateway, user("go"))

    assert (response.reasoning_content, response.content) == ("hm", "ok")
    assert gateway.renderer.strip_thinking_from_history is False and gateway.model_name == TOY_MODEL_NAME
    with pytest.raises(ContextWindowExceededError):
        chat(make_gateway(["x"], context_window_length=16), user("too long for sixteen tokens"))
    with pytest.raises(ValueError, match="has no attribute"):
        make_gateway(renderer="qwen3", renderer_kwargs={"no_such_setting": 1})


def test_the_policy_and_the_gateway_share_one_tokenizer():
    other = object()
    policy = ScriptedPolicy(["x"], tokenizer=other)

    with pytest.raises(ValueError, match="another tokenizer"):
        make_gateway(policy, tokenizer=toy_tokenizer(), renderer=toy_renderer())
    assert make_gateway(ScriptedPolicy(["x"])).sampler.tokenizer is toy_tokenizer()


def test_sample_answers_synchronously():
    policy = ScriptedPolicy(["sync"])
    prompt = toy_renderer().build_generation_prompt(user("go"))

    response = policy.sample(prompt=prompt, sampling_params=None, num_samples=1)

    assert response.sequences[0].tokens == toy_tokenizer().encode("sync<|im_end|>")
    with pytest.raises(ValueError, match="one completion per call"):
        ScriptedPolicy(["a"]).sample(prompt=prompt, sampling_params=None, num_samples=2)


def test_a_script_is_a_sequence_or_a_function():
    with pytest.raises(TypeError, match="not one completion"):
        ScriptedPolicy("just one string")
    with pytest.raises(TypeError, match="a string or a Completion"):
        chat(make_gateway([42]), user("go"))


def test_sample_response_builds_a_tinker_response():
    response = sample_response([1, 2], stop_reason="length", routing_id=3, logprob=-0.5)

    (sequence,) = response.sequences
    assert (sequence.tokens, sequence.logprobs, sequence.stop_reason) == ([1, 2], [-0.5, -0.5], "length")
    assert sequence.routing_id == 3
