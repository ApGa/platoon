"""Cross-mode parity, the ORS half: number search played in process (``python://``) and on a local env server
(``ors+http://``) must train alike.

Each scenario runs as a task group of four through the trainer's own path (``TaskRouter``, ``GroupRolloutWorkflow``,
``harness_rollout``, the model gateway and the datum builder), with a scripted policy for a sampler
(``parity_policy``) and the mode-neutral ``parity-probe`` harness.  The served variant must give the same
trajectories, model calls (prompt and sampled tokens, logprobs, routing ids), sampling parameters, outcomes,
datums before and after centering, baselines and group statistics as the in-process one, up to the allowlist of
``parity_compare`` (identifiers, wall-clock times, where the task came from).  The scenarios cover a flat group and
two recursive ones: a subagent on a shared fork that is not graded on its own (whose winning guess ends its parent's
episode, so a fork that does not share the env scores otherwise), and one on a fresh fork graded by its own session.

CI renders with a byte-level tokenizer; where the Qwen3-4B-Instruct-2507 snapshot is on disk the scenarios also run
with its real token ids.  Another mode joins by adding its :class:`~parity_runs.Variant` to ``VARIANTS``, with no
change to the comparison.  A Harbor leg, for one, is a generated task directory on the ``process`` provider: the
env's prompt as ``instruction.md``, a ``guess`` script on a mount that answers as the env does and writes the
answer to a file, and a ``tests/test.sh`` that grades that file; the probe plays it with its ``bash`` adapter.
"""

from __future__ import annotations

import copy
import dataclasses
import os

import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENREWARD_DISABLE_UPDATE_CHECK", "1")

pytest.importorskip("uvicorn", reason="the ORS env server needs uvicorn (the ors extra)")

from parity_compare import ALLOWED, allowances, canonical, compare  # noqa: E402
from parity_policy import byte_tokenizer, qwen_tokenizer  # noqa: E402
from parity_runs import IN_PROCESS, SCENARIOS, SERVED, Endpoints, run_group, scenario_problems, serve_ors  # noqa: E402

from platoon.gateway.core import TinkerLLMInteraction  # noqa: E402

# The variants compared with IN_PROCESS, the reference.
VARIANTS = (SERVED,)


@pytest.fixture(scope="module")
def endpoints():
    from platoon.envs.number_search.env import NumberSearchEnvironment

    with serve_ors([NumberSearchEnvironment]) as server:
        yield Endpoints(ors_server=server)


@pytest.fixture(params=["bytes", "qwen3"], scope="module")
def tokenizer(request):
    if request.param == "bytes":
        return byte_tokenizer()
    tokenizer = qwen_tokenizer()
    if tokenizer is None:
        pytest.skip("the Qwen3-4B-Instruct-2507 tokenizer snapshot is not on disk")
    return tokenizer


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SCENARIOS, ids=str)
async def test_served_ors_trains_like_in_process_ors(scenario, tokenizer, endpoints, tmp_path):
    reference = await run_group(IN_PROCESS, scenario, tokenizer=tokenizer, endpoints=endpoints, log_path=str(tmp_path))
    assert not scenario_problems(reference), scenario_problems(reference)

    for variant in VARIANTS:
        run = await run_group(variant, scenario, tokenizer=tokenizer, endpoints=endpoints, log_path=str(tmp_path))
        assert not scenario_problems(run), scenario_problems(run)
        found = compare(reference, run)
        assert not found, f"{variant} trains differently from {IN_PROCESS}:\n" + "\n".join(found)


@pytest.mark.asyncio
async def test_the_comparison_names_ids_and_masks_only_the_allowlist(endpoints, tmp_path):
    """Two runs of one variant differ in every id and time, and nothing else; anything else is reported."""

    flat = SCENARIOS[0]
    first, second = [
        await run_group(IN_PROCESS, flat, tokenizer=byte_tokenizer(), endpoints=endpoints, log_path=str(tmp_path))
        for _ in range(2)
    ]
    assert compare(first, second) == []

    member = canonical(first, second)["members"][1]
    root = member["trajectories"][0]
    assert member["id"] == "<rollout>" and root["id"] == "<traj 0>"
    assert root["steps"][0]["misc"]["action_misc"]["completion_id"] == "<call 0.0>"
    assert root["misc"]["tool_calls"][0]["after_call_id"] == "<call 0.0>"
    assert root["misc"]["model_call_spans"] == root["task"]["id"] == ALLOWED
    assert root["misc"]["tool_calls"][0]["name"] == "guess"  # compared between variants with the same tools

    changed = copy.deepcopy(second)
    trajectory = next(iter(changed.members[1].collection["trajectories"].values()))
    trajectory["reward"] = 0.5
    call_id = trajectory["steps"][2]["misc"]["action_misc"]["completion_id"]
    call = changed.members[1].interactions[call_id]
    sampled = dataclasses.replace(call.action, tokens=[*call.action.tokens[:-1], 0])
    changed.members[1].interactions[call_id] = TinkerLLMInteraction(call.obs, sampled, call.routing_id)
    found = compare(first, changed)
    assert [line.partition(":")[0] for line in found] == [
        "members[1].trajectories[0].steps[2].call.sampled",
        "members[1].trajectories[0].reward",
    ]
    assert found[1] == "members[1].trajectories[0].reward: 1.0 != 0.5"


@pytest.mark.asyncio
async def test_the_shared_fork_scenario_fails_where_a_fork_does_not_share_the_env(endpoints, tmp_path):
    """A mode whose shared forks gave the child an env of its own must not pass: there the members whose child found
    the number end with nothing submitted on their own env and score 0, which the scenario's rewards refuse."""

    shared = next(scenario for scenario in SCENARIOS if scenario.name == "subagent-shared-fork")
    unshared = dataclasses.replace(shared, agents={**shared.agents, "fork": "fresh"})
    run = await run_group(IN_PROCESS, unshared, tokenizer=byte_tokenizer(), endpoints=endpoints, log_path=str(tmp_path))

    refused = [problem.partition(":")[0] for problem in scenario_problems(run) if "root scored" in problem]
    assert refused == [f"{IN_PROCESS} member 1", f"{IN_PROCESS} member 3"]


def test_mode_choices_are_allowed_only_between_modes_that_make_them():
    harbor = dataclasses.replace(IN_PROCESS, name="harbor", surface="bash", grading="harbor")
    always = [allowance.path for allowance in allowances(IN_PROCESS, SERVED)]
    across = [allowance.path for allowance in allowances(IN_PROCESS, harbor)]

    assert "misc.tool_calls.*.text" not in always and "misc.outcome.reason" not in always
    assert {"misc.tool_calls.*.text", "misc.outcome.reason"} <= set(across)
    assert set(always) < set(across)
