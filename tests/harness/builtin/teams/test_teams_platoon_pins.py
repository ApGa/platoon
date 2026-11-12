"""What of platoon the ``platoon-harness`` harness and program build on, each pinned by a test named for it, so that a
change to any of them fails here.  CPU only; the episodes run core's ``Lockbox`` toy
(``tests/testing/testing_toys.py``) on a scripted policy."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
import tinker
import torch
from fakes.config import make_config
from fakes.loop import loop_state, loop_trainer, run_train_loop
from testing_toys import LOCKBOX

import platoon as pt
from platoon.envs.service import Outcome
from platoon.gateway.server import GatewayServer
from platoon.harness.agent import CALL_KIND_HEADER
from platoon.programs import single
from platoon.registry import resolve_component
from platoon.rewards import zero_subagent_rewards
from platoon.testing import ScriptedPolicy, make_gateway, rules, run_episode
from platoon.train.workers.run import DEFAULT_REWARD_PROCESSOR
from platoon.utils.subagent_rewards import propagate_root_success
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import EXCLUDE_FROM_TRAINING_MISC_KEY

USER = {"role": "user", "content": "Say hello."}


@pt.harness("teamspin-ask")
async def ask(ctx: pt.AgentContext) -> pt.AgentResult:
    """Asks the model once, as a call of the kind its options name."""

    reply = await ctx.chat([{"role": "user", "content": ctx.instruction}], kind=ctx.options["kind"])
    return pt.AgentResult("completed", final_message=reply.text)


@pt.harness("teamspin-aux-over-http")
async def aux_over_http(ctx: pt.AgentContext) -> pt.AgentResult:
    """Makes one ``aux`` call over its HTTP endpoint, as a black-box harness compacts, and notes its budget after."""

    endpoint = ctx.endpoint
    headers = {"Authorization": f"Bearer {endpoint.api_key}", CALL_KIND_HEADER: "aux"}
    body = {"model": endpoint.model, "messages": [{"role": "user", "content": ctx.instruction}]}
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{endpoint.base_url}/chat/completions", json=body, headers=headers) as response:
            response.raise_for_status()
    ctx.annotate(budget={"used": ctx.budget.used(), "remaining": ctx.budget.remaining()})
    return pt.AgentResult("completed")


def episode(harness: str, policy, **settings):
    return asyncio.run(run_episode("test:1", service=LOCKBOX, harness=harness, policy=policy, **settings))


def delegated_episode():
    """A root that delegates the task to one child, rewarded 1.0 and 0.25."""

    run = episode(
        "testing-delegator",
        ScriptedPolicy(rules({"Open the box": "DELEGATE", "Find the code": "1"})),
        agents={"child_grading": "env", "max_depth": 1},
    )
    run.root.reward, run.children[0].reward = 1.0, 0.25
    return run


def converted(run, processor=DEFAULT_REWARD_PROCESSOR):
    return get_train_data_for_trajectory_collection(
        run.collection.to_dict(), run.interactions, task_id="pin", checkpoint_version=0, reward_processor=processor
    )


def test_step_exclusion_keeps_a_roots_task_reward():
    run = delegated_episode()
    for step in run.root.steps:
        step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True

    result = converted(run)

    assert (result.task_reward, result.task_reward_valid) == (1.0, True)
    assert [stats.num_datums for stats in result.trajectory_stats] == [0, 1]  # the root's steps train no tokens


def test_trajectory_exclusion_of_a_root_replaces_the_task_reward():
    run = delegated_episode()
    run.root.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True

    result = converted(run)

    assert result.task_reward == 0.25  # the child's
    assert [stats.trajectory_id for stats in result.trajectory_stats] == [run.children[0].id]


def _streamed_id(text: str, where: tuple[str, ...]) -> str:
    """The id a streamed reply's first event names, at ``where`` in its data."""

    value = json.loads(next(line for line in text.splitlines() if line.startswith("data: "))[len("data: ") :])
    for key in where:
        value = value[key]
    return value


# Each dialect's path, a request body, and where its first streamed event names the reply: a chunk's id, the created
# response's, the started message's.
DIALECTS = {
    "chat": ("/v1/chat/completions", {"messages": [USER]}, ("id",)),
    "responses": ("/v1/responses", {"input": "Say hello."}, ("response", "id")),
    "messages": ("/v1/messages", {"messages": [USER], "max_tokens": 16}, ("message", "id")),
}


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_a_streamed_reply_names_its_recorded_call(dialect):
    path, body, where = DIALECTS[dialect]

    async def stream() -> tuple[list[str], str]:
        gateway = make_gateway(ScriptedPolicy(["hello"]))
        server = GatewayServer(gateway)
        await server.start(host="127.0.0.1")
        try:
            agent = gateway.open_agent(rollout_id="pin", trajectory_id="pin-root", interactions={})
            headers = {"Authorization": f"Bearer {agent.api_key}"}
            async with aiohttp.ClientSession() as http:
                payload = {"model": "policy", "stream": True, **body}
                async with http.post(f"{server.base_url}{path}", json=payload, headers=headers) as response:
                    return agent.call_ids, await response.text()
        finally:
            await server.stop()

    call_ids, text = asyncio.run(stream())

    assert call_ids == [_streamed_id(text, where)]


def test_a_bearer_on_the_origin_reaches_the_keys_agent():
    async def calls() -> tuple[list[str], list[str]]:
        gateway = make_gateway(ScriptedPolicy(["one", "two"]))
        server = GatewayServer(gateway)
        await server.start(host="127.0.0.1")
        try:
            first = gateway.open_agent(rollout_id="pin", trajectory_id="pin-first", interactions={})
            second = gateway.open_agent(rollout_id="pin", trajectory_id="pin-second", interactions={})
            async with aiohttp.ClientSession() as http:
                for prefix, bearer in (("", first.api_key), (f"/s/{first.api_key}", second.api_key)):
                    url = f"{server.base_url}{prefix}/v1/chat/completions"
                    headers = {"Authorization": f"Bearer {bearer}"}
                    async with http.post(url, json={"model": "policy", "messages": [USER]}, headers=headers) as reply:
                        reply.raise_for_status()
            return first.call_ids, second.call_ids
        finally:
            await server.stop()

    first, second = asyncio.run(calls())

    assert (len(first), second) == (2, [])  # the bearer's agent, and then the path's, whose secret beats the bearer


def _datum(advantage: float) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            key: tinker.TensorData.from_torch(torch.tensor(values))
            for key, values in {
                "advantages": [advantage, advantage],
                "mask": [1.0, 1.0],
                "target_tokens": [2, 3],
                "logprobs": [-0.1, -0.2],
                "checkpoint_version": [0],
            }.items()
        },
    )


def test_an_all_zero_batch_publishes_no_version():
    config = make_config(
        train={
            "batch_size": 1,
            "num_minibatches": 1,
            "num_microbatches": 1,
            "max_concurrent_rollouts": 1,
            "workflow_config": {"group_size": 1},
            "loss_fn": "cispo",
            "optimizer": {"learning_rate": 1e-6},
            "forward_backward_timeout_seconds": 10,
            "optim_step_timeout_seconds": 10,
        }
    )
    trainer = loop_trainer(config)

    logs = run_train_loop(trainer, loop_state(trainer), [[_datum(0.0)], [_datum(1.0)]], steps=2)

    assert [record["train/policy/version"] for record in logs] == [0, 1]


def test_delegation_scores_reward_success_and_the_default_scores_the_reward():
    trajectory = next(
        iter(
            episode("teamspin-ask", ["1"], harness_options={"kind": "policy"})
            .collection.to_dict()["trajectories"]
            .values()
        )
    )
    trajectory["reward"] = 0.25
    trajectory["steps"][-1]["misc"].setdefault("reward_misc", {})["reward/success"] = 0.75

    assert DEFAULT_REWARD_PROCESSOR(trajectory)[0] == 0.25
    assert resolve_component("reward_processor", "platoon-delegation")(trajectory)[0] == 0.75


@pytest.mark.parametrize("flags", [(True, True), (False, False)])
def test_single_adds_the_two_finalizers_we_add(flags):
    added = []

    class Episode:
        config = SimpleNamespace(skip_subagent_reward_computation=flags[0], propagate_root_success=flags[1])

        async def run(self) -> None:
            pass

        async def grade(self) -> Outcome:
            return Outcome(status="scored", rewards={"reward": 1.0})

        def add_finalizer(self, fn) -> None:
            added.append(fn)

    asyncio.run(single(Episode()))  # type: ignore[arg-type]

    assert added == ([zero_subagent_rewards, propagate_root_success] if all(flags) else [])


def test_an_aux_step_trains_as_a_policy_step_does():
    trained = {}
    for kind in ("policy", "aux"):
        run = episode("teamspin-ask", ["1"], harness_options={"kind": kind})
        (step,) = run.root.steps
        assert step.misc["action_misc"].get("kind", "policy") == kind
        trained[kind] = [
            (
                datum.model_input.to_ints(),
                datum.loss_fn_inputs["mask"].tolist(),
                datum.loss_fn_inputs["advantages"].tolist(),
            )
            for datum in converted(run).datums
        ]

    assert trained["aux"] == trained["policy"] and trained["policy"]


def test_an_aux_call_of_a_function_harness_is_not_counted():
    run = episode("teamspin-aux-over-http", ["summary"], max_steps=1)

    (step,) = run.root.steps
    assert step.misc["action_misc"]["kind"] == "aux"
    assert run.root.misc["budget"] == {"used": 0, "remaining": 1}
