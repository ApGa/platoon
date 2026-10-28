"""A gateway that serves several policies (``ModelGateway.serve_policies``): each agent samples one policy for its whole
life, the default for a root agent and its role's for a child, and its calls render, sample and parse in that policy's
runtime, with that policy's tokenizer.  Converted with the rollout's policies, only the trained policy's agents make
datums, each stamped with its own policy's version, while the frozen policy's agents keep their rewards and tags."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from gateway_fakes import Completion

from platoon.gateway.core import ChatRequest, ModelGateway
from platoon.gateway.trace import build_trajectory_collection
from platoon.task import Task
from platoon.testing import ScriptedPolicy, make_gateway, toy_tokenizer
from platoon.utils.tinker_data_processing import RolloutPolicies, get_train_data_for_trajectory_collection

TASK = Task(goal="Count the keys", id="vault-1")
# A word the helper's tokenizer has one token for, and the main policy's spells out byte by byte.
WORD = "Counting"


@dataclass
class Outcome:
    """Mirrors ``platoon.envs.service.Outcome``."""

    status: str
    rewards: dict[str, float] = field(default_factory=dict)
    primary: str = "reward"
    reason: str | None = None

    @property
    def reward(self) -> float | None:
        return self.rewards.get(self.primary) if self.status == "scored" else None


def helper_tokenizer():
    """The toy tokenizer with one more token: another tokenizer, which encodes :data:`WORD` as one id."""

    from transformers import PreTrainedTokenizerFast

    from platoon.testing.tokenizer import EOS_TOKEN, PAD_TOKEN, TOY_TOKENIZER_DIR

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(TOY_TOKENIZER_DIR / "tokenizer.json"), eos_token=EOS_TOKEN, pad_token=PAD_TOKEN
    )
    tokenizer.add_tokens([WORD])
    return tokenizer


@pytest.fixture
def served() -> tuple[ModelGateway, ScriptedPolicy, ScriptedPolicy]:
    """A gateway of policy ``main`` that also serves ``helper``, of another tokenizer, to the agents of role
    ``counter``."""

    main_policy = ScriptedPolicy(lambda prompt: "main answers", tokenizer=toy_tokenizer())
    helper_policy = ScriptedPolicy(lambda prompt: f"{WORD} done", tokenizer=helper_tokenizer())
    gateway, helper = make_gateway(main_policy), make_gateway(helper_policy)
    gateway.model_name, helper.model_name = "main-model", "helper-model"
    gateway.serve_policies("main", runtimes={"helper": helper}, roles={"counter": "helper"})
    return gateway, main_policy, helper_policy


def chat(gateway: ModelGateway, agent, text: str):
    return asyncio.run(gateway.chat(agent, ChatRequest.from_params([{"role": "user", "content": text}], {})))


def test_each_agent_samples_its_role_s_policy_for_life(served):
    gateway, main_policy, helper_policy = served
    interactions: dict = {}
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions=interactions, role="counter")
    counter = gateway.open_subagent(root, role="counter", goal="count")
    other = gateway.open_subagent(root, role="subagent", goal="other")
    minted = gateway.mint_subagent(root, role="counter")

    # A root agent samples the default, whatever its role; a child its role's, else the default.
    assert [agent.policy for agent in (root, counter, other, minted)] == ["main", "helper", "main", "helper"]
    replies = [chat(gateway, agent, f"{WORD} please") for agent in (root, counter, other, minted)]

    assert [reply.model for reply in replies] == ["main-model", "helper-model", "main-model", "helper-model"]
    assert [reply.policy for reply in replies] == ["main", "helper", "main", "helper"]
    # The tree's one map of interactions, which a program trains on, names each call's policy.
    assert [interactions[reply.call_id].policy for reply in replies] == ["main", "helper", "main", "helper"]
    assert len(main_policy.calls) == 2 and len(helper_policy.calls) == 2
    # Each policy's calls are rendered with its own tokenizer: only the helper's has one token for the word.
    word = helper_policy.tokenizer.convert_tokens_to_ids(WORD)
    assert all(word in call.prompt.to_ints() for call in helper_policy.calls)
    assert all(word not in call.prompt.to_ints() for call in main_policy.calls)
    assert gateway.budget_stop(counter).model == "helper-model"


def test_a_gateway_of_one_policy_records_no_policy(make_gateway, sampler):
    gateway = make_gateway()
    sampler.script = [Completion("Hello.")]
    interactions: dict = {}
    agent = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions=interactions)
    child = gateway.open_subagent(agent, role="counter")

    assert (agent.policy, child.policy, gateway.runtime(None)) == (None, None, gateway)
    reply = chat(gateway, agent, "hello")
    assert interactions[reply.call_id].policy is None
    collection = build_trajectory_collection(gateway.agents("r"), task=TASK, outcome=Outcome("scored", {"reward": 1}))
    assert all("policy" not in trajectory.misc["agent"] for trajectory in collection.trajectories.values())


def test_only_the_trained_policy_s_agents_make_datums_each_stamped_with_its_version(served):
    gateway, _, _ = served
    interactions: dict = {}
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions=interactions)
    counter = gateway.open_subagent(root, role="counter", goal="count")
    chat(gateway, root, "start")
    chat(gateway, counter, "count")
    chat(gateway, root, "finish")
    collection = build_trajectory_collection(gateway.agents("r"), task=TASK, outcome=Outcome("scored", {"reward": 1}))
    trajectories = collection.to_dict()
    assert [trajectory["misc"]["agent"]["policy"] for trajectory in trajectories["trajectories"].values()] == [
        "main",
        "helper",
    ]

    result = get_train_data_for_trajectory_collection(
        trajectory_collection=trajectories,
        interactions=interactions,
        task_id="vault-1",
        checkpoint_version=99,  # not read: each policy's own version is
        tag_datums=True,
        policies=RolloutPolicies(default="main", versions={"main": 3, "helper": 0}, trained=frozenset({"main"})),
    )

    assert result.datums and set(result.datum_trajectories or ()) == {root.trajectory_id}
    assert {int(datum.loss_fn_inputs["checkpoint_version"].to_torch()[0]) for datum in result.datums} == {3}
    tags = {tag.trajectory_id: tag for tag in result.agents or ()}
    assert (tags[root.trajectory_id].policy, tags[counter.trajectory_id].policy) == ("main", "helper")
    assert tags[counter.trajectory_id].reward is not None  # its reward still reaches credit
    assert result.num_policy_excluded_datums > 0  # the helper's datums, excluded from training


def test_each_call_records_its_own_policy_s_version_and_only_the_trained_rows_are_masked(served):
    """The per-datum stale filter (RFC 0020's P5) in a rollout of several policies: each call records the version of
    the runtime that sampled it, not the default policy's; the frozen policy's agents make no datums, so they give no
    own versions either; and the trained policy's datums, stamped with their own, are the rows the filter masks."""

    import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
    from platoon.train.step.staleness import StaleFilter, datum_version

    gateway, _, _ = served
    helper = gateway.runtime("helper")
    gateway.version, helper.version = 4, 0  # main's published version; the frozen helper's never moves
    interactions: dict = {}
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions=interactions)
    counter = gateway.open_subagent(root, role="counter", goal="count")
    first = chat(gateway, root, "start")
    counted = chat(gateway, counter, "count")
    gateway.version = 6  # the trainer published twice while the rollout ran
    last = chat(gateway, root, "finish")
    assert [interactions[reply.call_id].version for reply in (first, counted, last)] == [4, 0, 6]

    collection = build_trajectory_collection(gateway.agents("r"), task=TASK, outcome=Outcome("scored", {"reward": 1}))
    result = get_train_data_for_trajectory_collection(
        trajectory_collection=collection.to_dict(),
        interactions=interactions,
        task_id="vault-1",
        checkpoint_version=99,  # not read: each policy's own version is
        tag_datums=True,
        policies=RolloutPolicies(default="main", versions={"main": 4, "helper": 0}, trained=frozenset({"main"})),
    )

    assert result.num_policy_excluded_datums > 0  # the helper's
    assert set(result.datum_trajectories or ()) == {root.trajectory_id}
    assert [datum_version(datum) for datum in result.datums] == [4, 4]  # main's version for the rollout
    assert result.own_versions == (4, 6)  # one per datum of main's, none of the helper's
    assert result.stamp_own_versions()
    assert [datum_version(datum) for datum in result.datums] == [4, 6]
    verdict = StaleFilter(1, {}, per_datum=True).check(None, result.datums, 6)
    assert not verdict.stale and verdict.masked == (0,)


def test_a_role_bound_to_an_unknown_runtime_fails_its_first_call_not_another_policy(served):
    gateway, main_policy, _ = served
    gateway.role_policies["stray"] = "missing"
    root = gateway.open_agent(rollout_id="r", trajectory_id="root", interactions={})
    stray = gateway.open_subagent(root, role="stray")

    with pytest.raises(KeyError):
        chat(gateway, stray, "hello")
    assert main_policy.calls == []
