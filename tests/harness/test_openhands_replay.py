"""Reasoning replay of ``openhands`` (its default, training option): each request its SDK runner sends renders,
under the Qwen3.5-4B chat template the fast path trains with, to the previous prompt plus the previous completion
plus what came after, so the calls of an episode merge into one training sequence.  CPU only: the requests are the
fixture that ``openhands_live.py record`` saved from the real runner (see that module); only the tokenizer is
needed."""

from __future__ import annotations

import copy
import os

import pytest
from openhands_live import FIXTURE, SNAPSHOT, ScriptedPolicy, extends, first_difference, load_fixture

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "tokenizer.json").exists(), reason=f"Qwen3.5-4B tokenizer not available at {SNAPSHOT}"
)


@pytest.fixture(scope="module")
def replay():
    from openhands_live import load_tokenizer, recording_gateway

    tokenizer = load_tokenizer()
    policy = ScriptedPolicy(tokenizer, [])
    return tokenizer, policy, recording_gateway(tokenizer, policy), load_fixture()


def render(gateway, call: dict) -> list[int]:
    from platoon.gateway.core import RenderRequest

    request = RenderRequest(tools=call["tools"], render_kwargs=call.get("render_kwargs"))
    return gateway.render(copy.deepcopy(call["messages"]), request).to_ints()


def test_the_fixture_is_the_runners_own_three_call_episode(replay):
    _, _, _, data = replay
    calls = data["calls"]
    assert (data["model"], data["renderer"], data["harness"]) == (
        "Qwen/Qwen3.5-4B",
        "chat_template",
        "openhands@1.54.0",
    )
    assert len(calls) == 3 and FIXTURE.name == "replay.json.gz"
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == [
        "terminal",
        "file_editor",
        "task_tracker",
        "finish",
        "think",
    ]
    replayed = [message for message in calls[-1]["messages"] if message["role"] == "assistant"]
    # The runner's overrides: each assistant turn carries its reasoning, and its text as one string.
    assert [message.get("reasoning_content") for message in replayed] == [
        "The task wants 42 in answer.txt. I will write it.",
        "Let me check what I wrote.",
    ]
    assert replayed[1]["content"] == "Checking the file." and all(m["tool_calls"] for m in replayed)


def test_each_prompt_extends_the_previous_prompt_and_completion(replay):
    tokenizer, policy, gateway, data = replay
    prompts = [render(gateway, call) for call in data["calls"]]
    completions = [policy.tokens(call["completion"]) for call in data["calls"]]

    for index in range(1, len(prompts)):
        head = prompts[index - 1] + completions[index - 1]
        assert extends(prompts[index - 1], completions[index - 1], prompts[index]), first_difference(
            tokenizer, prompts[index][: len(head)], head
        )


def test_without_the_reasoning_the_prompts_would_not_extend(replay):
    """The control: the SDK's default serialization drops reasoning_content, and every call starts a new datum."""

    _, policy, gateway, data = replay
    stripped = copy.deepcopy(data["calls"])
    for call in stripped:
        for message in call["messages"]:
            message.pop("reasoning_content", None)
    prompts = [render(gateway, call) for call in stripped]
    completions = [policy.tokens(call["completion"]) for call in stripped]

    assert not any(
        extends(prompts[index - 1], completions[index - 1], prompts[index]) for index in range(1, len(prompts))
    )
