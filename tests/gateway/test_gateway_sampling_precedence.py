"""The rollout's sampling precedence (``RolloutConfig.sampling_precedence``): with ``config``, a policy call samples
with ``inference_params``' temperature, top_p and top_k, or the gateway's defaults, never its own, and with the
config's completion budget when it sets one, while its stop sequences and seed stay its own; in process and over
HTTP.  An aux call keeps its own parameters, but one of an agent whose key code the policy runs can read
(``policy_only``) samples as a policy call."""

from __future__ import annotations

import asyncio
from typing import Any

import openai
import pytest
from gateway_fakes import Completion

from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.gateway.core import ChatRequest, SamplingOverrides
from platoon.gateway.server import GatewayServer
from platoon.harness.agent import CALL_KIND_HEADER
from platoon.harness.rollout import rollout_sampling

USER: Any = {"role": "user", "content": "What's the weather in Paris?"}
CONFIG = InferenceParams(temperature=0.6, top_p=0.95, max_completion_tokens=40)


def open_root(gateway, **kwargs):
    return gateway.open_agent(rollout_id="rollout-1", trajectory_id="root", interactions={}, **kwargs)


def chat(gateway, agent, *, kind: str = "policy", **params):
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params([USER], params), kind=kind))


def test_the_rollout_config_carries_its_precedence_into_the_agents_defaults():
    assert rollout_sampling(RolloutConfig()).precedence == "request"
    defaults = rollout_sampling(RolloutConfig(inference_params=CONFIG, sampling_precedence="config"))
    assert defaults == SamplingOverrides(max_tokens=40, temperature=0.6, top_p=0.95, precedence="config")
    assert RolloutConfig(**{"sampling_precedence": "config", "inference_params": {"temperature": 1.0}}).inference_params
    assert rollout_sampling(RolloutConfig(**{"inference_params": {"top_k": 20}})).top_k == 20
    with pytest.raises(ValueError, match="sampling_precedence must be request or config"):
        RolloutConfig(sampling_precedence="harness")
    with pytest.raises(ValueError, match="sampling_precedence must be one of"):
        SamplingOverrides.from_inference_params(CONFIG, precedence="harness")  # type: ignore[arg-type]


def test_with_config_precedence_the_config_wins_over_a_policy_calls_own_sampling(make_gateway, sampler):
    sampler.script = [Completion("policy"), Completion("aux"), Completion("unset by the config")]
    gateway = make_gateway(max_tokens=100, temperature=1.0, top_p=1.0, top_k=5)
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(CONFIG, precedence="config"))
    child = gateway.open_subagent(root, role="helper")  # inherits the defaults and their precedence

    chat(gateway, root, temperature=0.0, top_p=0.5, max_completion_tokens=8)
    chat(gateway, child, kind="aux", temperature=0.0, max_completion_tokens=8)
    chat(gateway, child, top_k=2, seed=7, stop=["</answer>"])

    policy, aux, unset = (call["sampling_params"] for call in sampler.calls)
    assert (policy.temperature, policy.top_p, policy.max_tokens) == (0.6, 0.95, 40)  # the config's
    assert (aux.temperature, aux.top_p, aux.max_tokens) == (0.0, 0.95, 8)  # an aux call keeps its own
    # The config sets no top_k, so the gateway's applies, not the call's; the call's seed and stop sequences stay.
    assert (unset.top_k, unset.temperature, unset.seed, unset.stop) == (5, 0.6, 7, ["</answer>"])


def test_with_config_precedence_a_policy_call_never_samples_with_its_own_temperature_top_p_or_top_k(
    make_gateway, sampler
):
    sampler.script = [Completion("as configured")]
    gateway = make_gateway(max_tokens=100)  # the gateway's defaults: temperature 1, top_p 1, no top-k cutoff
    config = InferenceParams(temperature=1.0)
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(config, precedence="config"))

    chat(gateway, root, temperature=0.2, top_p=0.5, top_k=5, max_completion_tokens=8)

    (params,) = (call["sampling_params"] for call in sampler.calls)
    assert (params.temperature, params.top_p, params.top_k) == (1.0, 1.0, -1)
    assert params.max_tokens == 8  # the config sets no completion budget: the call's own


def test_a_rollout_config_sets_top_k(make_gateway, sampler):
    sampler.script = [Completion("default"), Completion("own"), Completion("configured")]
    gateway = make_gateway(top_k=50)
    config = InferenceParams(top_k=20)
    by_request = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(config))
    by_config = gateway.open_agent(
        rollout_id="rollout-2",
        trajectory_id="root",
        interactions={},
        sampling_defaults=SamplingOverrides.from_inference_params(config, precedence="config"),
    )

    chat(gateway, by_request)
    chat(gateway, by_request, top_k=5)
    chat(gateway, by_config, top_k=5)

    assert [call["sampling_params"].top_k for call in sampler.calls] == [20, 5, 20]


def test_by_default_a_calls_own_sampling_wins(make_gateway, sampler):
    sampler.script = [Completion("own")]
    gateway = make_gateway(max_tokens=100)
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(CONFIG))

    chat(gateway, root, temperature=0.0, max_completion_tokens=8)

    (params,) = (call["sampling_params"] for call in sampler.calls)
    assert (params.temperature, params.top_p, params.max_tokens) == (0.0, 0.95, 8)


@pytest.mark.asyncio
async def test_config_precedence_applies_to_http_policy_calls_only(gateway, sampler):
    sampler.script = [Completion("policy"), Completion("aux")]
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(CONFIG, precedence="config"))
    try:
        async with openai.AsyncOpenAI(base_url=f"{server.base_url}/v1", api_key=root.api_key, max_retries=0) as api:
            await api.chat.completions.create(
                model="p", messages=[USER], temperature=0.0, top_p=0.5, max_tokens=8, extra_body={"top_k": 3}
            )
            await api.chat.completions.create(
                model="p", messages=[USER], temperature=0.0, max_tokens=8, extra_headers={CALL_KIND_HEADER: "aux"}
            )
    finally:
        await server.stop()
    policy, aux = (call["sampling_params"] for call in sampler.calls)
    assert (policy.temperature, policy.top_p, policy.top_k, policy.max_tokens) == (0.6, 0.95, gateway.top_k, 40)
    assert (aux.temperature, aux.max_tokens) == (0.0, 8)


@pytest.mark.asyncio
async def test_a_policy_only_agent_s_aux_calls_sample_as_its_policy_calls(gateway, sampler):
    """Code the policy runs in a task sandbox can read its agent's key (``policy_only``): an aux call it makes trains,
    so it samples as a policy call does, with the config's parameters, never its own, in process and over HTTP, and
    so do those of an agent it mints.  Marking a call aux gets that code no sampling of its own."""

    sampler.script = [Completion("in process"), Completion("over HTTP"), Completion("minted")]
    server = GatewayServer(gateway, keepalive_seconds=0.05)
    await server.start(host="127.0.0.1")
    root = open_root(gateway, sampling_defaults=SamplingOverrides.from_inference_params(CONFIG, precedence="config"))
    root.policy_only = True
    child = gateway.mint_subagent(root, role="helper")  # as POST /v1/platoon/agents mints it: policy_only too
    own = {"temperature": 0.0, "top_p": 0.5, "max_tokens": 8, "top_k": 3}
    try:
        await gateway.chat(root, ChatRequest.from_params([USER], own), kind="aux")
        for agent in (root, child):
            async with openai.AsyncOpenAI(
                base_url=f"{server.base_url}/v1", api_key=agent.api_key, max_retries=0
            ) as api:
                await api.chat.completions.create(
                    model="p",
                    messages=[USER],
                    temperature=0.0,
                    top_p=0.5,
                    max_tokens=8,
                    extra_body={"top_k": 3},
                    extra_headers={CALL_KIND_HEADER: "aux"},
                )
    finally:
        await server.stop()

    assert len(sampler.calls) == 3 and child.policy_only and (root.policy_calls, child.policy_calls) == (0, 0)
    assert set(root.call_kinds.values()) == set(child.call_kinds.values()) == {"aux"}
    for call in sampler.calls:
        params = call["sampling_params"]
        assert (params.temperature, params.top_p, params.top_k, params.max_tokens) == (0.6, 0.95, gateway.top_k, 40)
