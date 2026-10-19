"""``openhands`` run for real from its built toolchain through ``harness_rollout`` and the real gateway, with each of
its settings: the training defaults, Harbor's adapter's evaluation options and the condenser, all from one
toolchain.  The episodes run in the harness fakes' process sandbox with a scripted policy: the Qwen3.5-4B chat
template a Harbor fast path trains with, and ``platoon.testing``'s toy model.

Opt in with ``PLATOON_TEST_TOOLCHAINS``: a directory where ``platoon toolchain build openhands`` built the toolchain
(about 600 MB).  The tests then check what the fixture-based and stand-in-SDK tests cannot: that the SDK runner
finishes an episode with calls that extend each other, stops at its own step limit, sends no reasoning back with the
adapter's options and then sends exactly the prompts of an agent built as Harbor's adapter builds it, takes its
retries, timeout and condenser from the options, each reported by the runner, and that a real condensation is
recorded as the agent's own call and takes one of its steps.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openhands_live import (
    HARNESS,
    SDK_SCRIPT,
    SNAPSHOT,
    TOOLCHAINS,
    extends,
    harbor_adapter_spec,
    run_episode,
    run_specific,
    settings,
    toolchain_ready,
    turn,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULTS = {"reasoning_replay": True, "llm_timeout": 3600, "num_retries": 0, "condenser": "none"}
ADAPTER = {"reasoning_replay": False, "request_timeout_s": 300, "num_retries": 5}  # Harbor's openhands-sdk adapter

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not toolchain_ready("openhands-1.54.0"), reason="PLATOON_TEST_TOOLCHAINS has no openhands toolchain"
    ),
]
needs_qwen = pytest.mark.skipif(not (SNAPSHOT / "tokenizer.json").exists(), reason="Qwen3.5-4B tokenizer not available")


def root_of(collection: dict) -> dict:
    (root,) = collection["trajectories"].values()
    return root


@needs_qwen
async def test_openhands_finishes_and_each_call_extends_the_last(tmp_path):
    collection, gateway, policy = await run_episode(HARNESS, SDK_SCRIPT, tmp_path)

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["exit_code"] == 0 and root["finish_message"] == "Wrote 42 to answer.txt."
    assert root["misc"]["harness"]["spec"] == HARNESS and settings(collection) == {"max_iterations": 30, **DEFAULTS}
    assert len(root["steps"]) == 3 and len(policy.prompts) == 3
    for index in (1, 2):
        assert extends(policy.prompts[index - 1], policy.completions[index - 1], policy.prompts[index])
    # Each call goes back with its reasoning, and no sampling parameter of the harness's own reaches the gateway.
    last = gateway.requests[-1]["messages"]
    assert [m.get("reasoning_content") for m in last if m["role"] == "assistant"][0].startswith("The task wants 42")


@needs_qwen
async def test_openhands_stops_at_its_own_step_limit_and_is_graded(tmp_path):
    collection, _, policy = await run_episode(HARNESS, SDK_SCRIPT, tmp_path, max_steps=1)

    root = root_of(collection)
    assert root["misc"]["harness"]["exit_code"] == 3 and len(policy.prompts) == 1
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0)  # the first call wrote 42


@needs_qwen
async def test_the_adapters_options_send_no_reasoning_back_and_are_graded(tmp_path):
    collection, gateway, policy = await run_episode(HARNESS, SDK_SCRIPT, tmp_path, options=ADAPTER)

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["spec"] == HARNESS and len(policy.prompts) == 3
    assert settings(collection) == {
        "max_iterations": 30,
        "reasoning_replay": False,
        "llm_timeout": 300,
        "num_retries": 5,
        "condenser": "none",
    }
    # Without replay the SDK drops each turn's reasoning, as Harbor's adapter sends it: no prompt extends the last.
    last = gateway.requests[-1]["messages"]
    assert not any(message.get("reasoning_content") for message in last if message["role"] == "assistant")
    assert not extends(policy.prompts[0], policy.completions[0], policy.prompts[1])


@needs_qwen
async def test_with_the_adapters_options_each_prompt_is_the_one_harbors_adapter_sends(tmp_path):
    """Through one gateway and one scripted policy, the adapter's options send what an agent built as Harbor's
    ``openhands-sdk`` adapter builds it sends: each prompt the same, token for token, but for the run's own date and
    sandbox directory.  The training defaults do not: replay sends each message as one string, so the system prompt's
    two parts render with a newline between them."""

    adapter_spec = harbor_adapter_spec(tmp_path / "adapter")
    prompts = {}
    for name, harness, options in (
        ("ours", HARNESS, ADAPTER),
        ("adapter", str(adapter_spec), None),
        ("training", HARNESS, None),
    ):
        collection, _, policy = await run_episode(harness, SDK_SCRIPT, tmp_path / name, options=options)
        assert (root_of(collection)["misc"]["outcome"]["status"], root_of(collection)["reward"]) == ("scored", 1.0)
        prompts[name] = [run_specific(policy.tokenizer.decode(prompt), tmp_path / name) for prompt in policy.prompts]

    assert len(prompts["ours"]) == 3 and prompts["ours"] == prompts["adapter"]
    assert "</PROCESS_MANAGEMENT><CURRENT_DATETIME>" in prompts["adapter"][0]
    assert "</PROCESS_MANAGEMENT>\n<CURRENT_DATETIME>" in prompts["training"][0]


@needs_qwen
async def test_the_condenser_runs_from_the_same_toolchain_and_its_agent_still_replays(tmp_path):
    collection, _, policy = await run_episode(HARNESS, SDK_SCRIPT, tmp_path, options={"condenser": "llm_summarizing"})

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert settings(collection) == {"max_iterations": 30, **DEFAULTS, "condenser": "llm_summarizing"}
    # A three-call episode is far below the condenser's thresholds: nothing is summarized, and replay still holds.
    assert len(policy.prompts) == 3 and all(
        extends(policy.prompts[index - 1], policy.completions[index - 1], policy.prompts[index]) for index in (1, 2)
    )


@needs_qwen
async def test_a_condensation_is_the_agents_own_call_and_takes_one_of_its_steps(tmp_path):
    """A prompt that overflows the gateway's context window sets off a real condensation.  The summary is a model call
    that the gateway records as one of the agent's own policy calls, unmarked, since a harness in a task sandbox may
    make no other kind; the prompt after it starts a new training sequence; and the refused call and the condensation
    each take one of the agent's steps (``--max-iterations``)."""

    options = {"condenser": "llm_summarizing"}
    _, _, sizing = await run_episode(HARNESS, SDK_SCRIPT[2:], tmp_path / "sizing", options=options)
    window = len(sizing.prompts[0]) + 3000  # room for two short turns, not for a long tool output
    script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("Now look at many numbers.", call=("terminal", {"command": "seq 1 1500"})),
        turn("I summarize.", text="SUMMARY: the agent wrote 42 to answer.txt and printed 1 to 1500."),
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]

    collection, gateway, policy = await run_episode(
        HARNESS, script, tmp_path / "run", options=options, context_window_length=window
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["exit_code"] == 0 and root["finish_message"] == "Wrote 42."
    # Four sampled calls: two turns, the summary and the finish; the gateway refused a fifth, which overflowed.
    assert len(policy.prompts) == 4 and len(gateway.requests) == 5
    assert [step["misc"]["action_misc"].get("kind", "policy") for step in root["steps"]] == ["policy"] * 4
    assert "SUMMARY" in root["steps"][2]["misc"]["message"]["content"]
    assert extends(policy.prompts[0], policy.completions[0], policy.prompts[1])
    assert not any(extends(policy.prompts[i - 1], policy.completions[i - 1], policy.prompts[i]) for i in (2, 3))

    collection, _, policy = await run_episode(
        HARNESS, script, tmp_path / "steps", options=options, context_window_length=window, max_steps=4
    )

    # Four steps: two turns, the refused call and the condensation, so the agent never gets to finish.
    root = root_of(collection)
    assert root["misc"]["harness"]["exit_code"] == 3 and len(policy.prompts) == 3
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0)  # it wrote 42 first


async def test_a_scripted_openhands_episode_through_the_gateway_with_platoon_testing(tmp_path, monkeypatch):
    """The ``platoon.testing`` way: a scripted policy over the toy model, one task, the path training takes."""

    from harness_fakes import FakeService

    from platoon.testing import ScriptedPolicy, tool_call
    from platoon.testing import run_episode as run_testing_episode

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    monkeypatch.setenv("PLATOON_TOOLCHAINS", str(TOOLCHAINS))  # the named toolchain's root, as a site sets it
    policy = ScriptedPolicy(
        [tool_call("terminal", command="echo 42 > answer.txt"), tool_call("finish", message="Wrote 42.")]
    )

    run = await run_testing_episode(
        "train-0",
        service="harness_fakes:FakeService",
        harness="openhands",
        policy=policy,
        harness_options={"num_retries": 1, "request_timeout_s": 600},
        output_dir=tmp_path / "out",
    )

    assert (run.outcome["status"], run.reward) == ("scored", 1.0), run.root.misc
    assert run.root.misc["harness"]["exit_code"] == 0 and run.root.finish_message == "Wrote 42."
    assert len(policy.calls) == 2 and len(run.interactions) == 2  # every call recorded, as training reads them
    log = Path(run.root.misc["harness"]["logs_path"]).read_text(errors="replace")
    assert '"llm_timeout": 600, "num_retries": 1' in log
