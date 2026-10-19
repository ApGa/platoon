"""``openhands`` run for real from its built toolchain through ``harness_rollout`` and the real gateway, with each of
its settings: the training defaults, Harbor's adapter's evaluation options and the condenser with and without its
token trigger, all from one toolchain.  The episodes run in the harness fakes' process sandbox with a scripted
policy: the Qwen3.5-4B chat template a Harbor fast path trains with, and ``platoon.testing``'s toy model.

Opt in with ``PLATOON_TEST_TOOLCHAINS``: a directory where ``platoon toolchain build openhands`` built the toolchain
(about 600 MB).  The tests then check what the fixture-based and stand-in-SDK tests cannot: that the SDK runner
finishes an episode with calls that extend each other, stops at its own step limit, sends no reasoning back with the
adapter's options and then sends exactly the prompts of an agent built as Harbor's adapter builds it, takes its
retries, timeout and condenser from the options, each reported by the runner, that a real condensation is recorded
as an aux call of the agent, which the gateway takes from a harness in a task sandbox without a gateway step budget,
and takes one of its steps, and that a token trigger condenses where the gateway's own count of the next prompt passes
it, with a summary that trains as the agent's turns do.  They also check the runner's condenser where it differs
from the SDK's: its summary call reads the forgotten turns and an earlier summary whole, a condensation keeps the
task (when early turns hold most tokens, and in a context reset whose summary prompt must be cut to fit the
window), a summary without text leaves a notice, and a trigger with nothing left to forget is set aside.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openhands_live import (
    HARNESS,
    SDK_SCRIPT,
    SNAPSHOT,
    extends,
    harbor_adapter_spec,
    run_episode,
    run_specific,
    settings,
    toolchain_ready,
    toolchain_site,
    turn,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULTS = {
    "reasoning_replay": True,
    "llm_timeout": 3600,
    "num_retries": 0,
    "condenser": "none",
    "condenser_max_tokens": 0,
}
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
    assert {kind for kind, _ in gateway.renders} == {"call"}  # without a token trigger, nothing counts tokens
    assert kinds(gateway) == [("policy", True, None)] * 3  # the agent's calls carry no call kind
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
        "condenser_max_tokens": 0,
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
    options = {"condenser": "llm_summarizing"}
    collection, gateway, policy = await run_episode(HARNESS, SDK_SCRIPT, tmp_path, options=options)

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert settings(collection) == {"max_iterations": 30, **DEFAULTS, "condenser": "llm_summarizing"}
    # A three-call episode is far below the condenser's thresholds: nothing is summarized, and replay still holds.
    assert len(policy.prompts) == 3 and all(
        extends(policy.prompts[index - 1], policy.completions[index - 1], policy.prompts[index]) for index in (1, 2)
    )
    assert {kind for kind, _ in gateway.renders} == {"call"}  # no token trigger: nothing counts tokens, as on main
    assert kinds(gateway) == [("policy", True, None)] * 3  # nothing condensed: no aux call


@needs_qwen
async def test_a_condensation_is_an_aux_call_of_the_agent_and_takes_one_of_its_steps(tmp_path):
    """A prompt that overflows the gateway's context window sets off a real condensation.  The summary is a model call
    that the runner marks ``aux`` (the call-kind header, which only the condenser's calls carry), and the gateway
    records it as one of the agent's aux calls: the harness runs in the task sandbox (``policy_only``) without a
    gateway step budget.  The prompt after it starts a new training sequence, and the refused call and the
    condensation each take one of the agent's steps (``--max-iterations``)."""

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
    assert kinds(gateway) == [("policy", True, None)] * 2 + [("aux", True, None), ("policy", True, None)]
    assert step_kinds(root) == ["policy", "policy", "aux", "policy"]
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


@needs_qwen
async def test_a_token_trigger_condenses_once_the_gateways_count_passes_it_and_the_summary_trains(tmp_path):
    """``condenser_max_tokens``: before each model call the condenser counts the prompt it would send at the gateway,
    which gives exactly the length of the prompt it renders, so the agent condenses at the first count past the
    trigger: here the prompt after a long tool output, which no context window refused.  The summary is an aux call
    of the agent, outside its policy calls; with the rollout's advantage it trains like the others, and the prompt
    after it starts a new sequence.  The condensation takes one of the agent's steps, and the step counts of training
    and evaluation count it as a step."""

    _, _, sizing = await run_episode(HARNESS, SDK_SCRIPT[:1], tmp_path / "sizing", max_steps=1)
    trigger = len(sizing.prompts[0]) + 3000  # past two short turns and the summary, short of a long tool output
    script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("Now look at many numbers.", call=("terminal", {"command": "seq 1 1500"})),
        turn("I summarize.", text="SUMMARY: the agent wrote 42 to answer.txt and printed 1 to 1500."),
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]
    options = {"condenser_max_tokens": trigger}  # the condenser option stays none: the trigger runs it

    collection, gateway, policy = await run_episode(HARNESS, script, tmp_path / "run", options=options)

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["exit_code"] == 0 and root["finish_message"] == "Wrote 42."
    assert settings(collection) == {"max_iterations": 30, **DEFAULTS, "condenser_max_tokens": trigger}
    # Four sampled calls: two turns, the summary and the finish, and nothing refused: no context window was set.
    assert len(policy.prompts) == 4 and len(gateway.requests) == 4
    # Each turn's prompt is what the condenser counted just before it: the gateway's count is the rendered prompt.
    calls = [index for index, (kind, _) in enumerate(gateway.renders) if kind == "call"]
    assert len(calls) == 4 and gateway.renders[calls[0] - 1] == ("count", len(policy.prompts[0]))
    for turn_call in (1, 3):
        assert gateway.renders[calls[turn_call] - 1] == ("count", len(policy.prompts[turn_call]))
    # The condensation came at the first count past the trigger: the third prompt, after the long output.
    counts = [(index, length) for index, (kind, length) in enumerate(gateway.renders) if kind == "count"]
    first_over = next(index for index, length in counts if length > trigger)
    assert calls[1] < first_over < calls[2] and len(policy.prompts[1]) <= trigger
    assert all(length <= trigger for index, length in counts if index > calls[2])  # the summary brought it under
    # The summary is an aux call, the only call with the call-kind header, from an agent in the task sandbox that no
    # gateway step budget caps; and the prompts after the long output start new sequences.
    assert kinds(gateway) == [("policy", True, None)] * 2 + [("aux", True, None), ("policy", True, None)]
    assert step_kinds(root) == ["policy", "policy", "aux", "policy"]
    assert "SUMMARY" in root["steps"][2]["misc"]["message"]["content"]
    assert "SUMMARY" in policy.tokenizer.decode(policy.prompts[3])
    assert extends(policy.prompts[0], policy.completions[0], policy.prompts[1])
    assert not any(extends(policy.prompts[i - 1], policy.completions[i - 1], policy.prompts[i]) for i in (2, 3))

    # It trains: every sampled token of the four calls, the summary's included, with the rollout's advantage, in
    # three sequences (the two turns merged, the summary, the finish).
    from platoon.utils.tinker_data_processing import trajectory_to_data

    result = trajectory_to_data(
        root,
        gateway.interactions,
        task_id=collection["id"],
        trajectory_id=root["id"],
        trajectory_advantage=0.5,
        checkpoint_version=0,
    )
    trained = [trained_tokens(datum) for datum in result.datums]
    assert len(trained) == 3 and trained[1] == policy.completions[2]  # the summary, alone
    assert trained[0] == policy.completions[0] + policy.completions[1] and trained[2] == policy.completions[3]
    assert {advantage for datum in result.datums for advantage in trained_values(datum, "advantages")} == {0.5}
    # It samples as an agent call does: the gateway samples an aux call of an agent in the task sandbox as a policy
    # call, also where the config's sampling takes precedence (``sampling_precedence``), and no call asks for sampling
    # of its own.
    from platoon.gateway.core import SamplingOverrides

    assert [chat["sampled_as"] for chat in gateway.chats] == ["policy"] * 4
    assert all(chat["sampling"] == SamplingOverrides() for chat in gateway.chats)
    # An evaluation counts it under aux, and the step counts of training and evaluation count it as a step.
    from platoon.cli import summarize_collection
    from platoon.eval.accounting import rollout_accounting

    usage = rollout_accounting(collection)["usage"]
    assert (usage["policy"]["calls"], usage["aux"]["calls"]) == (3, 1)
    assert result.num_steps == 4 and summarize_collection(collection)["total_steps"] == 4

    collection, _, policy = await run_episode(HARNESS, script, tmp_path / "steps", options=options, max_steps=3)

    # Three steps: two turns and the condensation, so the agent never gets to finish.
    root = root_of(collection)
    assert root["misc"]["harness"]["exit_code"] == 3 and len(policy.prompts) == 3
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0)  # it wrote 42 first


def kinds(gateway) -> list[tuple[str, bool, int | None]]:
    """For each call the gateway sampled: its kind, whether its agent was ``policy_only`` and the agent's call cap."""

    return [(chat["kind"], chat["policy_only"], chat["call_cap"]) for chat in gateway.chats]


def step_kinds(root: dict) -> list[str]:
    """The kind of each step's call, as the trajectory records it (``action_misc.kind``, absent for a policy call)."""

    return [step["misc"]["action_misc"].get("kind", "policy") for step in root["steps"]]


def trained_values(datum, key: str) -> list[float]:
    """The values of ``datum.loss_fn_inputs[key]`` at the positions the loss trains."""

    mask = datum.loss_fn_inputs["mask"].to_torch().tolist()
    return [value for value, trains in zip(datum.loss_fn_inputs[key].to_torch().tolist(), mask, strict=True) if trains]


def trained_tokens(datum) -> list[int]:
    return [int(token) for token in trained_values(datum, "target_tokens")]


def trained_sequences(root: dict, gateway) -> list[list[int]]:
    """The sampled tokens each datum of ``root`` trains, with the rollout's advantage (0.5) at each of them."""

    from platoon.utils.tinker_data_processing import trajectory_to_data

    result = trajectory_to_data(
        root,
        gateway.interactions,
        task_id="train-0",
        trajectory_id=root["id"],
        trajectory_advantage=0.5,
        checkpoint_version=0,
    )
    assert {value for datum in result.datums for value in trained_values(datum, "advantages")} == {0.5}
    return [trained_tokens(datum) for datum in result.datums]


# The SDK's instructions to the summary call (its system prompt), and the task of the harness fakes.
SUMMARIZER = "You are maintaining a context-aware state summary"
TASK = "Write 42 to answer.txt"


def decoded(policy, index: int) -> str:
    return policy.tokenizer.decode(policy.prompts[index])


def summary_calls(policy) -> list[int]:
    """The calls of ``policy`` that were summary calls: their prompt opens with the SDK's summarizing instructions."""

    return [index for index, prompt in enumerate(policy.prompts) if SUMMARIZER in policy.tokenizer.decode(prompt[:64])]


def harness_log(collection: dict) -> str:
    return Path(root_of(collection)["misc"]["harness"]["logs_path"]).read_text(errors="replace")


@pytest.fixture
def same_sandboxes(monkeypatch):
    """One rollout id for every episode, so that two episodes in directories of one length render the same prompts:
    the sandbox's path, which holds the rollout id, is in the system prompt and in the terminal's output."""

    import types
    import uuid

    import platoon.harness.rollout as rollout

    fixed = uuid.UUID("12345678-1234-5678-1234-567812345678")
    monkeypatch.setattr(rollout, "uuid", types.SimpleNamespace(uuid4=lambda: fixed))


@needs_qwen
async def test_the_summary_call_reads_the_forgotten_turns_whole_and_an_earlier_summary_entire(tmp_path):
    """The summary call is shown what the forgotten turns did: the command each ran, with its arguments, and its
    whole output.  A second condensation is shown the first summary entire, so what that kept is kept again.  The
    SDK's condenser shows 500 characters of each and names a tool call's action only."""

    _, _, sizing = await run_episode(HARNESS, SDK_SCRIPT[:1], tmp_path / "sizing", max_steps=1)
    trigger = len(sizing.prompts[0]) + 3000
    first_summary = "SUMMARY-ONE: wrote 42 and printed 1 to 1500. " + "More of it. " * 60 + "KEPT-BY-ONE: seed 1234."
    script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("Now look at many numbers.", call=("terminal", {"command": "seq 1 1500"})),
        turn("I summarize.", text=first_summary),
        turn("Look at more numbers.", call=("terminal", {"command": "seq 1501 3000"})),
        turn("I summarize again.", text="SUMMARY-TWO: seed 1234; printed 1 to 3000."),
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]

    collection, gateway, policy = await run_episode(
        HARNESS, script, tmp_path / "run", options={"condenser_max_tokens": trigger}
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert summary_calls(policy) == [2, 4]
    first, second = decoded(policy, 2), decoded(policy, 4)
    assert "Call: terminal" in first and "seq 1 1500" in first and "1\n2\n3\n" in first and "1499\n1500" in first
    assert first_summary in second and "seq 1501 3000" in second and "2999\n3000" in second
    assert "SUMMARY-TWO" in decoded(policy, 5) and TASK in decoded(policy, 5)
    # Five sequences train: the two first turns merged, each summary alone, the turn between them, the finish.
    completions = policy.completions
    assert trained_sequences(root, gateway) == [completions[0] + completions[1], *completions[2:]]


@needs_qwen
async def test_a_condensation_for_the_trigger_forgets_only_what_it_must_and_keeps_the_task(tmp_path, same_sandboxes):
    """One long output early, then many short turns: the prompt gets under half the trigger once the early turns are
    forgotten, fewer than the tenth of the events the SDK's condenser asks a condensation to forget
    (``minimum_progress``), where it would reset the whole context and summarize the task away with the rest."""

    short = 40
    sizing_script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("A big output.", call=("terminal", {"command": "seq 1 6000"})),
        *[turn(f"Small step {i}.", call=("terminal", {"command": f"echo step{i}"})) for i in range(short)],
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]
    _, _, sizing = await run_episode(HARNESS, sizing_script, tmp_path / "a" / "w", max_steps=60)
    lengths = [len(prompt) for prompt in sizing.prompts]
    last = 2 + short  # the finish call, after the last short step
    trigger = (lengths[last - 1] + lengths[last]) // 2  # first passed by the count before the finish call
    assert 4 < 0.1 * (2 + 2 * last)  # forgetting the 4 early events is less than the SDK's minimum progress
    script = [*sizing_script[:-1], turn("I summarize.", text="SUMMARY-EARLY: printed 1 to 6000."), sizing_script[-1]]

    collection, gateway, policy = await run_episode(
        HARNESS, script, tmp_path / "b" / "w", options={"condenser_max_tokens": trigger}, max_steps=60
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert summary_calls(policy) == [last] and len(policy.prompts) == last + 2
    summarized, after = decoded(policy, last), decoded(policy, last + 1)
    assert "seq 1 6000" in summarized and "5999\n6000" in summarized and "echo step0" not in summarized
    # The task, the summary and every short turn are in the next prompt; the long output is not.
    assert TASK in after and "SUMMARY-EARLY" in after and "5999\n6000" not in after
    assert all(f"echo step{i}" in after for i in range(short))
    assert "<|im_start|>user\n" + TASK in after  # the task is still the agent's first user message


@needs_qwen
async def test_a_context_reset_keeps_the_task_and_cuts_the_events_until_its_summary_fits(tmp_path):
    """With ``llm_summarizing``, an output that overflows the context window makes the gateway refuse the agent's
    call, and the condenser summarizes the turns, whole.  Here that summary prompt overflows the window too, so the
    condenser resets the context: it summarizes every turn after the task, cutting each one's middle (the total by a
    fifth at each try) until the prompt fits, and keeps the system prompt and the task."""

    _, _, sizing = await run_episode(
        HARNESS, SDK_SCRIPT[2:], tmp_path / "sizing", options={"condenser": "llm_summarizing"}
    )
    window = len(sizing.prompts[0]) + 3000
    script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("Now look at many numbers.", call=("terminal", {"command": "seq 1 2500"})),  # far past the window
        turn("I summarize.", text="SUMMARY-RESET: wrote 42 and printed 1 to 2500."),
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]

    collection, gateway, policy = await run_episode(
        HARNESS, script, tmp_path / "run", options={"condenser": "llm_summarizing"}, context_window_length=window
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["exit_code"] == 0 and summary_calls(policy) == [2] and len(policy.prompts) == 4
    # The gateway refused the agent's call, the first summary prompt and at least one cut that was still too long.
    assert len(gateway.requests) >= len(policy.prompts) + 3
    summarized, after = decoded(policy, 2), decoded(policy, 3)
    assert "1\n2\n3\n" in summarized and "2499\n2500" in summarized and "<response clipped>" in summarized
    assert "<|im_start|>user\n" + TASK in after and "SUMMARY-RESET" in after and "2499\n2500" not in after
    assert "the summary call of a context reset failed" in harness_log(collection)


@needs_qwen
async def test_a_summary_call_answered_without_text_leaves_a_notice_in_its_place(tmp_path):
    """The policy answers the summary call with a tool call, as an agent policy may.  The SDK's condenser then drops
    the forgotten turns and inserts nothing; the runner's leaves a notice that they were dropped, and says so."""

    _, _, sizing = await run_episode(HARNESS, SDK_SCRIPT[:1], tmp_path / "sizing", max_steps=1)
    script = [
        turn("Write the answer first.", call=("terminal", {"command": "echo 42 > answer.txt"})),
        turn("Now look at many numbers.", call=("terminal", {"command": "seq 1 1500"})),
        turn("I will just call a tool.", call=("terminal", {"command": "echo summary"})),  # the summary call
        turn("All done.", call=("finish", {"message": "Wrote 42."})),
    ]
    options = {"condenser_max_tokens": len(sizing.prompts[0]) + 3000}

    collection, gateway, policy = await run_episode(HARNESS, script, tmp_path / "run", options=options)

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert summary_calls(policy) == [2] and len(policy.prompts) == 4
    after = decoded(policy, 3)
    assert TASK in after and "no summary of them could be made" in after and "1499\n1500" not in after
    assert "the summary call answered without text" in harness_log(collection)
    # It trains as every summary does: an aux step, with the rollout's advantage.
    assert step_kinds(root)[2] == "aux" and policy.completions[2] in trained_sequences(root, gateway)


@needs_qwen
async def test_a_trigger_with_nothing_left_to_forget_is_set_aside_and_the_agent_works_on(tmp_path):
    """A trigger the first prompt alone passes (80, meant as 80%), and one that the first prompt with a summary
    passes: no condensation can meet either.  The condenser sets the trigger aside for the rest of the run, says so,
    and counts no more tokens; the agent works on, where the SDK's would summarize at every step."""

    collection, gateway, policy = await run_episode(
        HARNESS, SDK_SCRIPT, tmp_path / "tiny", options={"condenser_max_tokens": 80}
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert root["misc"]["harness"]["exit_code"] == 0 and len(policy.prompts) == 3 and not summary_calls(policy)
    assert [kind for kind, _ in gateway.renders] == ["count", "call", "call", "call"]  # one count, then none
    assert harness_log(collection).count("token trigger (80 tokens) is set aside") == 1

    _, _, sizing = await run_episode(HARNESS, SDK_SCRIPT[:1], tmp_path / "sizing", max_steps=1)
    trigger = len(sizing.prompts[0]) + 20  # passed by the second prompt, and by the first with a long summary
    script = [SDK_SCRIPT[0], turn("I summarize.", text="SUMMARY: " + "wrote 42 to answer.txt. " * 40), *SDK_SCRIPT[1:]]

    collection, gateway, policy = await run_episode(
        HARNESS, script, tmp_path / "short", options={"condenser_max_tokens": trigger}
    )

    root = root_of(collection)
    assert (root["misc"]["outcome"]["status"], root["reward"]) == ("scored", 1.0), root["misc"]
    assert summary_calls(policy) == [1] and len(policy.prompts) == 4 and root["misc"]["harness"]["exit_code"] == 0
    assert TASK in decoded(policy, 2) and "SUMMARY: wrote 42" in decoded(policy, 2)
    assert harness_log(collection).count(f"token trigger ({trigger} tokens) is set aside") == 1


async def test_a_scripted_openhands_episode_through_the_gateway_with_platoon_testing(tmp_path, monkeypatch):
    """The ``platoon.testing`` way: a scripted policy over the toy model, one task, the path training takes."""

    from harness_fakes import FakeService

    from platoon.testing import ScriptedPolicy, tool_call
    from platoon.testing import run_episode as run_testing_episode

    FakeService.root = tmp_path / "sandboxes"
    FakeService.sessions.clear()
    monkeypatch.setenv("PLATOON_SITE", str(toolchain_site(tmp_path)))  # the named toolchain's root, as a site's
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
