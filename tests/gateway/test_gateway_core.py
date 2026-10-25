"""ModelGateway: render -> sample -> parse -> record, agents, and the TinkerLLM adapter over the same core."""

from __future__ import annotations

import asyncio
import json
import pickle

import pytest
from gateway_fakes import MODEL_NAME, TOOLS, WEATHER_CALL, Completion, sample_response

from platoon.config_defs import InferenceParams
from platoon.gateway import core
from platoon.gateway.core import (
    AgentClosedError,
    ChatRequest,
    ContextWindowExceededError,
    ModelGateway,
    SamplingOverrides,
    UnknownAgentError,
)

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What's the weather in Paris?"}


def chat(gateway, agent, messages, **params):
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params(messages, params)))


def open_root(gateway, interactions=None, **kwargs):
    return gateway.open_agent(
        rollout_id="rollout-1",
        trajectory_id="traj-root",
        interactions={} if interactions is None else interactions,
        **kwargs,
    )


# --------------------------------------------------------------------------- render, sample, parse, record


def test_chat_records_exact_tokens_in_call_order(gateway, sampler, renderer, tokenizer):
    sampler.script = [Completion("Hello there.", routing_id=11), Completion("It is sunny.")]
    interactions: dict = {}
    agent = open_root(gateway, interactions)

    first = chat(gateway, agent, [SYSTEM, USER])
    history = [SYSTEM, USER, first.message(), {"role": "user", "content": "And tomorrow?"}]
    second = chat(gateway, agent, history)

    assert agent.call_ids == [first.call_id, second.call_id]
    assert list(interactions) == agent.call_ids
    assert first.call_id.startswith("tinker-sampling-")
    # The record is exactly what the sampler saw and produced.
    for response, call in zip((first, second), sampler.calls, strict=True):
        interaction = interactions[response.call_id]
        assert interaction.obs.to_ints() == call["prompt"].to_ints()
        assert interaction.action.tokens == response.tokens
        assert interaction.action.maybe_logprobs == [-0.25] * len(response.tokens)
        assert call["num_samples"] == 1
    assert interactions[first.call_id].routing_id == 11
    assert interactions[second.call_id].routing_id is None
    assert sampler.calls[0]["prompt"].to_ints() == renderer.build_generation_prompt([SYSTEM, USER]).to_ints()
    assert (first.content, first.finish_reason, first.reasoning_content) == ("Hello there.", "stop", None)
    # A plain-text turn re-renders token for token: the second prompt extends the first call.
    first_sequence = sampler.calls[0]["prompt"].to_ints() + first.tokens
    assert sampler.calls[1]["prompt"].to_ints()[: len(first_sequence)] == first_sequence
    assert first.usage() == {
        "prompt_tokens": first.prompt_tokens,
        "completion_tokens": len(first.tokens),
        "total_tokens": first.prompt_tokens + len(first.tokens),
    }


def test_tool_calls_parse_into_openai_shape(gateway, sampler, tokenizer):
    sampler.script = [Completion(WEATHER_CALL)]
    agent = open_root(gateway)
    response = chat(gateway, agent, [SYSTEM, USER], tools=TOOLS, tool_choice="auto")

    assert response.finish_reason == "tool_calls"
    assert response.content == ""
    (tool_call,) = response.tool_calls
    assert tool_call["id"].startswith("tinker-tool-call-")
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    assert response.generation == {
        "stop_reason": "stop",
        "generated_tokens": len(response.tokens),
        "parse_success": True,
        "parsed_tool_calls": 1,
        "emitted_tool_calls": 1,
        "recovered_tool_calls": 0,
        "unparsed_tool_calls": 0,
    }
    # The cookbook renderer received the tool specs folded into the system prompt.
    prompt = tokenizer.decode(sampler.calls[0]["prompt"].to_ints())
    assert prompt.startswith("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools")
    assert '"name": "get_weather"' in prompt


def test_an_unparsed_tool_call_stays_in_the_content(gateway, sampler):
    broken = '<tool_call>\n{"name": "get_weather", "arguments": {city"}\n</tool_call>'
    sampler.script = [Completion(f"Checking.\n{broken}")]
    response = chat(gateway, open_root(gateway), [SYSTEM, USER], tools=TOOLS)

    assert response.tool_calls is None and response.finish_reason == "stop"
    assert response.content == f"Checking.\n\n{broken}"
    assert response.generation["unparsed_tool_calls"] == 1 and response.generation["parsed_tool_calls"] == 0


def test_length_cutoff_never_emits_tool_calls(gateway, sampler):
    sampler.script = [Completion(WEATHER_CALL, stop_reason="length")]
    response = chat(gateway, open_root(gateway), [SYSTEM, USER], tools=TOOLS)
    assert response.finish_reason == "length"
    assert response.tool_calls is None
    assert response.generation["emitted_tool_calls"] == 0


def test_openai_tool_history_renders_for_cookbook_renderers(gateway, renderer):
    """Cookbook renderers need ToolCall objects and string content; the gateway converts OpenAI history."""
    from tinker_cookbook.renderers import ToolCall

    arguments = '{"city": "Paris"}'
    openai_history = [
        SYSTEM,
        USER,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "get_weather", "arguments": arguments}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
        {"role": "assistant", "content": "Sunny in Paris.", "tool_calls": None},
        {"role": "user", "content": "Thanks!"},
    ]
    cookbook_history = [
        SYSTEM,
        USER,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(id="call-1", function=ToolCall.FunctionBody(name="get_weather", arguments=arguments))
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
        {"role": "assistant", "content": "Sunny in Paris."},
        {"role": "user", "content": "Thanks!"},
    ]
    assert gateway.render(openai_history).to_ints() == renderer.build_generation_prompt(cookbook_history).to_ints()


def test_sampling_defaults_and_overrides(make_gateway, sampler, renderer):
    sampler.script = [Completion("a"), Completion("b")]
    gateway = make_gateway(max_tokens=100, temperature=0.7, top_k=5, top_p=0.9, seed=3)
    agent = open_root(gateway)
    chat(gateway, agent, [USER])
    chat(gateway, agent, [USER], max_completion_tokens=7, temperature=1, top_k=2, top_p=1, seed=9, stop=["\n"])
    defaults, overridden = (call["sampling_params"] for call in sampler.calls)
    assert (defaults.max_tokens, defaults.temperature, defaults.top_k, defaults.top_p, defaults.seed) == (
        100,
        0.7,
        5,
        0.9,
        3,
    )
    assert defaults.stop == renderer.get_stop_sequences()
    assert (overridden.max_tokens, overridden.temperature, overridden.top_k, overridden.top_p, overridden.seed) == (
        7,
        1.0,
        2,
        1.0,
        9,
    )
    assert overridden.stop == ["\n"]


def test_agent_sampling_defaults_fill_what_a_call_leaves_unset(make_gateway, sampler):
    sampler.script = [Completion("a"), Completion("b"), Completion("c")]
    gateway = make_gateway(max_tokens=100, temperature=0.7, top_k=5, top_p=0.9)
    rollout = InferenceParams(temperature=0.2, top_p=0.5, max_completion_tokens=33)
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(rollout))
    child = gateway.open_subagent(root, role="helper")  # inherits the root's defaults
    chat(gateway, root, [USER])
    chat(gateway, child, [USER])
    chat(gateway, child, [USER], temperature=1, max_completion_tokens=7)
    root_call, child_call, overridden = (call["sampling_params"] for call in sampler.calls)
    assert (root_call.max_tokens, root_call.temperature, root_call.top_p, root_call.top_k) == (33, 0.2, 0.5, 5)
    assert (child_call.max_tokens, child_call.temperature, child_call.top_p) == (33, 0.2, 0.5)
    assert (overridden.max_tokens, overridden.temperature, overridden.top_p) == (7, 1.0, 0.5)


def test_a_rollout_applies_only_the_sampling_its_config_sets(make_gateway, sampler):
    assert SamplingOverrides.from_inference_params(InferenceParams()) == SamplingOverrides()
    assert SamplingOverrides.from_inference_params(InferenceParams(top_p=0.5)) == SamplingOverrides(top_p=0.5)

    sampler.script = [Completion("a"), Completion("b")]
    gateway = make_gateway(max_tokens=100, temperature=0.7, top_p=0.9)
    for index, params in enumerate((InferenceParams(), InferenceParams(max_completion_tokens=33))):
        root = gateway.open_agent(
            rollout_id=f"rollout-{index}",
            trajectory_id=f"root-{index}",
            interactions={},
            sampling_defaults=SamplingOverrides.from_inference_params(params),
        )
        chat(gateway, root, [USER])
    unset, budget = (call["sampling_params"] for call in sampler.calls)
    assert (unset.max_tokens, unset.temperature, unset.top_p) == (100, 0.7, 0.9)
    assert (budget.max_tokens, budget.temperature, budget.top_p) == (33, 0.7, 0.9)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"temperature": 3.0}, "temperature"),
        ({"max_tokens": -1}, "max_tokens"),
        ({"top_p": "high"}, "top_p"),
        ({"stop": "\n"}, "stop"),
        ({"tool_choice": "required", "tools": TOOLS}, "tool_choice"),
        ({"tools": {"name": "x"}}, "tools"),
    ],
)
def test_invalid_request_params_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        ChatRequest.from_params([USER], params)


def test_chat_template_kwargs_and_reasoning_effort():
    request = ChatRequest.from_params(
        [USER],
        {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False}},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert request.render.render_kwargs == {"enable_thinking": False, "preserve_thinking": False}
    assert ChatRequest.from_params([USER], {"reasoning_effort": "none"}).render.render_kwargs == {
        "enable_thinking": False
    }
    assert ChatRequest.from_params([USER], {}).render.render_kwargs is None


def test_context_window_clamps_and_rejects_overflow(make_gateway, sampler):
    sampler.script = [Completion("ok")]
    gateway = make_gateway(context_window_length=64)
    agent = open_root(gateway)
    response = chat(gateway, agent, [USER], max_tokens=1000)
    assert sampler.calls[0]["sampling_params"].max_tokens == 64 - response.prompt_tokens

    with pytest.raises(ContextWindowExceededError, match="input exceeds the context window") as info:
        chat(gateway, agent, [{"role": "user", "content": "word " * 100}])
    assert info.value.context_window == 64 and info.value.prompt_tokens > 64
    assert len(sampler.calls) == 1  # never sampled
    assert agent.call_ids == [response.call_id]


def test_sample_timeout_cancels_the_stalled_request(make_gateway, sampler):
    gateway = make_gateway(sample_timeout_seconds=0.01)
    cancelled = []

    async def stalled(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    sampler.sample_async = stalled
    agent = open_root(gateway)
    with pytest.raises(TimeoutError, match="0.01s"):
        chat(gateway, agent, [USER])
    assert cancelled == [True]
    assert agent.calls == [] and agent.interactions == {}


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_sample_timeout_must_be_finite_and_positive(make_gateway, monkeypatch, value):
    monkeypatch.setenv("TINKER_SAMPLE_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match=r"\(TINKER_SAMPLE_TIMEOUT_SECONDS\) must be finite and positive"):
        make_gateway()


def test_the_sample_timeout_argument_wins_over_the_variable_which_wins_over_the_default(make_gateway, monkeypatch):
    """The argument is a config's ``sample_timeout_seconds``; the variable is what a gateway job exports."""

    monkeypatch.setenv("TINKER_SAMPLE_TIMEOUT_SECONDS", "1800")
    assert make_gateway(sample_timeout_seconds=90).sample_timeout_seconds == 90.0
    assert make_gateway(sample_timeout_seconds=None).sample_timeout_seconds == 1800.0
    monkeypatch.delenv("TINKER_SAMPLE_TIMEOUT_SECONDS")
    assert make_gateway().sample_timeout_seconds == core.DEFAULT_SAMPLE_TIMEOUT_SECONDS == 600.0


def test_an_argument_leaves_the_variable_unread(make_gateway, monkeypatch):
    monkeypatch.setenv("TINKER_SAMPLE_TIMEOUT_SECONDS", "half an hour")
    assert make_gateway(sample_timeout_seconds=1800).sample_timeout_seconds == 1800.0
    with pytest.raises(ValueError, match=r"must be a number of seconds, got 'half an hour'"):
        make_gateway()


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_a_sample_timeout_argument_must_be_finite_and_positive(make_gateway, monkeypatch, value):
    monkeypatch.setenv("TINKER_SAMPLE_TIMEOUT_SECONDS", "1800")  # a valid variable does not stand in for a bad key
    with pytest.raises(ValueError, match=r"\(sample_timeout_seconds\) must be finite and positive"):
        make_gateway(sample_timeout_seconds=value)


# --------------------------------------------------------------------------- agents


def test_agents_have_distinct_256_bit_secrets_and_authenticate(gateway):
    first, second = open_root(gateway), open_root(gateway)
    assert first.secret != second.secret
    assert len(first.secret) >= 43  # token_urlsafe(32): 256 bits
    assert first.api_key == first.secret
    assert gateway.authenticate(first.secret) is first
    with pytest.raises(UnknownAgentError):
        gateway.authenticate("not-a-secret")
    gateway.close_agent(first)
    gateway.close_agent(first)  # idempotent
    with pytest.raises(AgentClosedError):
        gateway.authenticate(first.secret)
    assert gateway.authenticate(second.secret) is second


def test_children_share_the_rollout_and_record_where_they_were_forked(gateway, sampler):
    sampler.script = [Completion("one"), Completion("two"), Completion("child")]
    interactions: dict = {}
    root = open_root(gateway, interactions)
    chat(gateway, root, [USER])
    chat(gateway, root, [USER])
    child = gateway.open_subagent(root, role="explorer", goal="Find the forecast")
    response = chat(gateway, child, [{"role": "user", "content": "Find the forecast"}])

    assert (child.rollout_id, child.parent_agent_id, child.role, child.goal) == (
        "rollout-1",
        root.agent_id,
        "explorer",
        "Find the forecast",
    )
    assert child.fork_step == 2
    assert child.interactions is interactions and response.call_id in interactions
    assert child.call_ids == [response.call_id] and len(root.calls) == 2
    assert gateway.agents("rollout-1") == [root, child]
    with pytest.raises(UnknownAgentError):
        gateway.open_agent(rollout_id="rollout-1", trajectory_id="t", parent_agent_id="missing", interactions={})
    with pytest.raises(ValueError, match="belongs to rollout"):
        gateway.open_agent(rollout_id="other", trajectory_id="t", parent_agent_id=root.agent_id, interactions={})

    assert gateway.close_rollout("rollout-1") == [root, child]
    assert gateway.agents("rollout-1") == [] and root.closed and child.closed
    with pytest.raises(AgentClosedError):
        gateway.open_subagent(root, role="late")


def test_a_agent_closed_while_sampling_records_nothing(gateway, sampler):
    sampler.script = [Completion("late", delay=0.05)]
    agent = open_root(gateway)

    async def run():
        call = asyncio.ensure_future(gateway.chat(agent, ChatRequest.from_params([USER], {})))
        await asyncio.sleep(0.01)
        gateway.close_agent(agent)
        return await call

    with pytest.raises(AgentClosedError):
        asyncio.run(run())
    assert agent.calls == [] and agent.interactions == {}


def test_agent_timeline_receives_phase_timings(gateway, sampler):
    from platoon.train.tinker.rollout_metrics import RolloutTimeline

    sampler.script = [Completion("one"), Completion("two")]
    timeline = RolloutTimeline()
    root = open_root(gateway, timeline=timeline)
    chat(gateway, root, [USER])
    chat(gateway, gateway.open_subagent(root, role="helper"), [USER])
    summary = timeline.summary()
    assert summary.model_calls == 2
    assert summary.sample_seconds_total >= 0.0 and summary.render_seconds_total > 0.0
    timing = root.calls[0].timing
    assert timing.started_at <= timing.parse_started_at <= timing.ended_at


def test_the_timeline_is_told_as_each_call_is_issued_before_it_samples_and_its_first_call_fires_once(gateway, sampler):
    """The start limit's in-process signal (``train.max_starting_rollouts``): the gateway tells the agent's timeline as
    it issues a call, before it samples, and ``RolloutTimeline`` calls its ``on_first_call`` once for the whole tree,
    root and subagent alike."""

    from platoon.train.tinker.rollout_metrics import RolloutTimeline

    sampler.script = [Completion("one", delay=0.05), Completion("two")]
    fired: list[int] = []
    timeline = RolloutTimeline(on_first_call=lambda: fired.append(len(sampler.calls)))
    root = open_root(gateway, timeline=timeline)

    async def run():
        first = asyncio.ensure_future(gateway.chat(root, ChatRequest.from_params([USER], {})))
        await asyncio.sleep(0.01)
        assert fired == [0] and not first.done()  # told before it sampled, and it still samples
        await first
        await gateway.chat(gateway.open_subagent(root, role="helper"), ChatRequest.from_params([USER], {}))

    asyncio.run(run())
    assert fired == [0] and timeline.summary().model_calls == 2


def test_a_failing_first_call_callback_breaks_no_call_and_a_timeline_without_the_hook_is_fine(gateway, sampler):
    from platoon.train.tinker.rollout_metrics import RolloutTimeline

    def fail() -> None:
        raise RuntimeError("a callback that fails")

    class Plain:
        """A call timeline with ``record`` only."""

        def record(self, **timings: float) -> None:
            self.timings = timings

    sampler.script = [Completion("one"), Completion("two")]
    assert chat(gateway, open_root(gateway, timeline=RolloutTimeline(on_first_call=fail)), [USER]).content == "one"
    plain = Plain()
    assert chat(gateway, open_root(gateway, timeline=plain), [USER]).content == "two" and plain.timings


# --------------------------------------------------------------------------- the sampler is followed, never pinned


class _VersionedClient:
    """A picklable sampling client that tags its responses with its weights version."""

    def __init__(self, version: int) -> None:
        self.version = version

    async def sample_async(self, *, prompt, sampling_params, num_samples):
        return sample_response([1000 + self.version], stop_reason="length")


def test_following_sampling_client_is_read_at_every_call(renderer, tokenizer, tmp_path):
    from platoon.train.tinker.workers.binding import FollowingSamplingClient, publish_binding

    path = str(tmp_path / "binding.pkl")
    publish_binding(_VersionedClient(3), 3, path)
    follower = FollowingSamplingClient(path, pickle.dumps(_VersionedClient(3)), poll_seconds=0.0, bootstrap_version=3)
    gateway = ModelGateway(sampler=follower, renderer=renderer, tokenizer=tokenizer, model_name=MODEL_NAME)
    agent = open_root(gateway)

    assert chat(gateway, agent, [USER]).tokens == [1003]
    publish_binding(_VersionedClient(4), 4, path)
    assert chat(gateway, agent, [USER]).tokens == [1004]
    assert follower.oldest_sampled_version == 3


# --------------------------------------------------------------------------- the trainer's policy handle


def test_tinker_llm_reads_through_to_its_gateway(renderer, tokenizer, sampler):
    from platoon.train.tinker.proxy import TinkerLLM

    llm = TinkerLLM(model_name=MODEL_NAME, renderer=renderer, tokenizer=tokenizer, sampling_client=sampler)
    assert llm.sampling_client is sampler and llm.renderer is renderer and llm.model_name == MODEL_NAME
    assert llm.gateway.sampler is sampler and llm.gateway.model_name == MODEL_NAME
    llm.update_sampling_client(None)
    assert llm.gateway.sampler is None and llm.version == 1


def test_set_as_default_makes_the_policy_gateway_the_process_default(renderer, tokenizer, sampler, monkeypatch):
    from platoon.train.tinker.proxy import TinkerLLM

    monkeypatch.setattr(core, "_default_gateway", None)
    with pytest.raises(RuntimeError, match="No model gateway"):
        core.default_gateway()
    llm = TinkerLLM(model_name=MODEL_NAME, renderer=renderer, tokenizer=tokenizer, sampling_client=sampler)
    assert llm.set_as_default() is llm
    assert core.default_gateway() is llm.gateway
