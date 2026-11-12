"""Each span of an agent trains on its own reward (RFC 0021, section 6): a trajectory's ``misc["spans"]`` reaches its
agent's tag and run (``AgentTag.spans``, ``AgentRun.spans``); ``keyed`` pools agent rewards as before and trains each
call of each scored span on that span's reward less its agent's baseline shared equally by its scored spans, and every
other call on nothing; a training group whose agents name spans stops the run under any credit that does not read
them, and an evaluation's group, which trains nothing, runs, in the workflows ``platoon train`` builds too; kept groups
keep the span advantages; and a tag pickled before the field loads as naming no spans."""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import tinker
import torch
import yaml
from fakes.launch import built_workflows
from tinker import TensorData

import platoon.train.tinker  # noqa: F401  (resolves the package's import cycle first)
from platoon.task import Task
from platoon.testing.credit import fixture_group, run_checks
from platoon.train.checkpoints.kept_groups import decode_group, encode_group
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig, WorkflowConfig
from platoon.train.credit.api import (
    AgentRun,
    CreditEstimatorBase,
    CreditGroup,
    CreditSetup,
    MemberRollout,
    PolicyView,
    Span,
    TrainingSample,
)
from platoon.train.credit.group import GroupCredit
from platoon.train.credit.keyed import Keyed, KeyedOptions
from platoon.train.credit.runtime import CreditEstimatorError, assign_credit
from platoon.train.dispatch import runner
from platoon.train.policy.spec import MAIN
from platoon.train.workflow.crediting import member_rollout
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.output import PolicyGroupOutput
from platoon.train.workflow.protocol import report_of, rows_by_policy
from platoon.utils import tinker_data_processing as tdp
from platoon.utils.config import resolve_config
from platoon.utils.stats_tracker import DeferredStats, StatsTracker

_SETUP = CreditSetup(name="test", entry="default", settings=WorkflowConfig(), policies=(MAIN,))


# --- groups built by hand --------------------------------------------------------------------------------------------


def _row(reward: float, calls: Sequence[tuple[str, int]], member: int) -> TrainingSample:
    """A row as the converter makes it: an observation token, then each call's sampled tokens (``(call_id, n)``), each
    followed by an observation token, every action token holding the agent's ``reward``."""

    mask, named = [0.0], []
    for call, tokens in calls:
        named.append((call, len(mask), len(mask) + tokens))
        mask += [1.0] * tokens + [0.0]
    datum = tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(range(len(mask)))),
        loss_fn_inputs={
            "mask": TensorData.from_torch(torch.tensor(mask)),
            "advantages": TensorData.from_torch(torch.tensor([reward * action for action in mask])),
            "checkpoint_version": TensorData.from_torch(torch.tensor([0])),
        },
    )
    return TrainingSample(datum, MAIN, member, calls=tuple(named))


def _agent(
    name: str,
    reward: float | None,
    rows: Sequence[Sequence[tuple[str, int]]],
    spans: Sequence[tuple[Sequence[str], float | None]] | None = None,
    *,
    role: str = "peer",
    member: int = 0,
) -> AgentRun:
    agent = AgentRun(name, name, None, role, MAIN, 0, reward)
    agent.samples = [_row(reward or 0.0, calls, member) for calls in rows]
    for sample in agent.samples:
        sample.agent_id, sample.role, sample.trajectory_id, sample.depth = name, role, name, 0
    if spans is not None:
        agent.spans = tuple(Span(frozenset(calls), None if value is None else float(value)) for calls, value in spans)
    return agent


def _group(*agents: AgentRun) -> CreditGroup:
    """A group of ``agents``, each in the member its rows name (member 0 for an agent with none)."""

    by_member: dict[int, list[AgentRun]] = {}
    for agent in agents:
        by_member.setdefault(agent.samples[0].member if agent.samples else 0, []).append(agent)
    unused = tdp.TrajectoryCollectionResult(datums=[], task_reward=0.0, trajectory_stats=[], root_rewards_dict={})
    members = [
        MemberRollout(index, [sample for agent in group for sample in agent.samples], group, 0.0, True, unused)
        for index, group in sorted(by_member.items())
    ]
    return CreditGroup(
        task_id="task",
        label=None,
        entry="default",
        settings=WorkflowConfig(),
        members=members,
        policies={MAIN: PolicyView(MAIN)},
        versions={MAIN: 0},
        stats=DeferredStats(),
    )


def _credit(group: CreditGroup, options: KeyedOptions | None = None) -> CreditGroup:
    asyncio.run(Keyed(_SETUP, options).assign(group))
    return group


def _by_call(agent: AgentRun) -> dict[str, float]:
    """The advantage each call of ``agent`` trains on: the one value on its tokens."""

    values = {}
    for sample in agent.samples:
        for call, start, end in sample.calls:
            [value] = set(sample.advantages[start:end].tolist())
            values[call] = value
    return values


def _metrics(group: CreditGroup) -> dict[str, Any]:
    tracker, stats = StatsTracker(), group.stats
    assert isinstance(stats, DeferredStats)
    stats.record(tracker)
    return tracker.export()


# --- the conversion --------------------------------------------------------------------------------------------------


def _interaction(prompt: list[int], completion: list[int]) -> Any:
    """Stand-in for ``TinkerLLMInteraction``."""

    return SimpleNamespace(
        obs=tinker.ModelInput.from_ints(prompt),
        action=SimpleNamespace(tokens=list(completion), logprobs=[-0.5] * len(completion)),
        version=None,
    )


def _step(call: str) -> dict:
    return {"misc": {"action_misc": {"completion_id": call}}}


# A worker's three calls, the second's prompt extending the first's (one row of two calls), the third a row of its own;
# and its helper's one call.
_CALLS = {
    "w1": _interaction([1, 2], [10, 11]),
    "w2": _interaction([1, 2, 10, 11, 3], [12, 13, 14]),
    "w3": _interaction([7], [15]),
    "h1": _interaction([5], [16, 17]),
}


def _converted(
    worker: float, spans: list[Any] | None, *, helper_spans: list[Any] | None = None, tag_datums: bool = True
) -> tdp.TrajectoryCollectionResult:
    """A rollout of a worker (its reward ``worker``, the spans it names ``spans``, None for none) and a helper it
    delegated to, as the converter returns it."""

    def misc(agent: str, role: str, named: list[Any] | None) -> dict:
        return {"agent": {"agent_id": agent, "role": role}, **({} if named is None else {tdp.SPANS_MISC_KEY: named})}

    collection = {
        "trajectories": {
            "root": {
                "steps": [_step(c) for c in ("w1", "w2", "w3")],
                "reward": worker,
                "misc": misc("w", "worker", spans),
            },
            "root.1": {
                "steps": [_step("h1")],
                "reward": 0.5,
                "misc": misc("h", "helper", helper_spans),
                "parent_info": {"id": "root"},
            },
        }
    }
    return tdp.get_train_data_for_trajectory_collection(
        collection, _CALLS, task_id="task", checkpoint_version=0, tag_datums=tag_datums
    )


def test_a_span_names_calls_of_its_trajectory_and_reaches_its_agent_run():
    result = _converted(0.75, [{"calls": ["w1", "w3"], "reward": 0.5}, {"calls": ["w2"], "reward": None}])
    spans = (Span(frozenset({"w1", "w3"}), 0.5), Span(frozenset({"w2"}), None))

    assert result.agents is not None and result.agents[0].spans == spans
    worker, _ = member_rollout(0, result, MAIN, tagged=True).agents
    assert (worker.agent_id, worker.reward, worker.spans) == ("w", 0.75, spans)
    # Each span's calls are named on the rows that train them (TrainingSample.calls).
    assert [call for sample in worker.samples for call, _, _ in sample.calls] == ["w1", "w2", "w3"]


def test_a_trajectory_that_names_no_spans_differs_from_one_that_names_none():
    worker, helper = member_rollout(0, _converted(0.75, None, helper_spans=[]), MAIN, tagged=True).agents

    assert worker.spans is None  # it names no spans: every call trains on its reward
    assert helper.spans == ()  # it names spans, and has none: no call of it trains


def test_a_call_named_by_two_spans_refuses_the_conversion():
    with pytest.raises(ValueError, match=r"misc\['spans'\]\[1\] names calls an earlier span names.*\['w2'\]"):
        _converted(0.75, [{"calls": ["w1", "w2"], "reward": 0.5}, {"calls": ["w2", "w3"], "reward": 0.25}])


@pytest.mark.parametrize(
    "spans",
    [
        {"calls": ["w1"], "reward": 0.5},
        [{"calls": "w1", "reward": 0.5}],
        [{"calls": ["w1"]}],
        [{"calls": [], "reward": True}],
        [{"calls": ["w1"], "reward": float("nan")}],
        [{"calls": ["w1"], "reward": float("inf")}],
    ],
)
def test_spans_of_another_shape_refuse_the_conversion(spans):
    with pytest.raises(ValueError, match=r"trajectory root's misc\['spans'\]"):
        _converted(0.75, spans)


def test_a_call_that_no_step_of_the_trajectory_made_refuses_the_conversion():
    """A span names calls of its own trajectory: a stale or mistyped id would still count the span among its agent's
    scored spans, and move the baseline share of each of the others."""

    with pytest.raises(
        ValueError, match=r"misc\['spans'\]\[1\] names calls that no step of the trajectory made: \['h1'\]"
    ):
        _converted(0.75, [{"calls": ["w1"], "reward": 0.5}, {"calls": ["w2", "h1"], "reward": 0.25}])


def test_a_rollout_whose_trajectory_names_spans_is_tagged_unasked():
    """Only an estimator that reads spans may train them, so the trainer must see them whatever its credit asked."""

    assert _converted(0.75, None, tag_datums=False).agents is None
    tagged = _converted(0.75, [{"calls": ["w1"], "reward": 0.75}], tag_datums=False)
    assert tagged.agents is not None and tagged.datum_trajectories == ("root", "root", "root.1")


def test_a_result_pickled_before_spans_loads_as_naming_no_spans():
    result = _converted(0.75, [{"calls": ["w1"], "reward": 0.75}])
    assert result.agents is not None
    for tag in result.agents:
        del tag.__dict__["spans"]  # the state of a tag pickled before the field
    loaded = pickle.loads(pickle.dumps(result))

    assert loaded.agents is not None and [tag.spans for tag in loaded.agents] == [None, None]
    assert [agent.spans for agent in member_rollout(0, loaded, MAIN, tagged=True).agents] == [None, None]


# --- keyed -----------------------------------------------------------------------------------------------------------


def test_keyed_pools_agent_rewards_and_shares_each_agents_baseline_over_its_scored_spans():
    """Three peers of one key, whose spans hold all their credit (the agent rows 5/24, 7/24 and 6/24, their mean 1/4),
    each span on its own credit less a share of the baseline; and a helper, alone in its key, on its own."""

    def th(numerator: int) -> float:  # numerator/24
        return float(Fraction(numerator, 24))

    p1 = _agent("p1", th(5), [[("a", 2), ("b", 1)]], [(["a"], th(1)), (["b"], th(4))])
    p2 = _agent("p2", th(7), [[("c", 3)]], [(["c"], th(7))])
    p3 = _agent("p3", th(6), [[("d", 1)], [("e", 2)]], [(["d"], th(2)), (["e"], th(4))])
    helper = _agent("h", 0.5, [[("x", 2)]], [(["x"], 0.25), ([], 0.25)], role="helper")
    group = _credit(_group(p1, p2, p3, helper))

    assert {**_by_call(p1), **_by_call(p2), **_by_call(p3)} == pytest.approx(
        {"a": th(1) - th(3), "b": th(4) - th(3), "c": th(7) - th(6), "d": th(2) - th(3), "e": th(4) - th(3)}
    )
    for agent in (p1, p2, p3):
        assert agent.reward is not None
        assert sum(_by_call(agent).values()) == pytest.approx(agent.reward - 0.25)  # its own advantage, in parts
    # Alone in its key, the helper is its own baseline, shared by both its scored spans, one of which has no row.
    assert _by_call(helper) == pytest.approx({"x": 0.25 - 0.5 / 2})
    assert _metrics(group)["credit/keyed/baselines"] == 2.0


@pytest.mark.parametrize("leave_one_out", [False, True])
def test_an_agents_advantages_do_not_depend_on_how_many_spans_its_credit_was_cut_into(leave_one_out):
    options = KeyedOptions(leave_one_out=leave_one_out)

    def credited(cuts: int) -> tuple[dict[str, float], dict[str, float]]:
        calls = [(f"b{index}", 1) for index in range(cuts)]
        a = _agent("a", 1 / 3, [[("a0", 2)]], [(["a0"], 1 / 3)])
        b = _agent("b", 1 / 3, [calls], [([call], 1 / 3 / cuts) for call, _ in calls])
        c = _agent("c", 0.0, [[("c0", 1)]])
        _credit(_group(a, b, c), options)
        return _by_call(a), _by_call(b)

    a_whole, b_whole = credited(1)
    for cuts in (3, 4):
        a, b = credited(cuts)
        assert a == pytest.approx(a_whole)  # its peer's cuts move nothing of it
        assert sum(b.values()) == pytest.approx(sum(b_whole.values()))
        assert len(set(b.values())) == 1  # equal parts of its credit train alike


@pytest.mark.parametrize("leave_one_out", [False, True])
def test_an_agent_of_one_span_holding_all_its_credit_trains_as_keyed_would(leave_one_out):
    def agents(spans: bool) -> list[AgentRun]:
        rows = {"p": [[("p1", 2), ("p2", 1)], [("p3", 3)]], "q": [[("q1", 2)]], "r": [[("r1", 1), ("r2", 1)]]}
        rewards = {"p": 0.75, "q": 0.25, "r": 1.0}
        return [
            _agent(
                name,
                rewards[name],
                rows[name],
                [([call for row in rows[name] for call, _ in row], rewards[name])] if spans else None,
                member=index,
            )
            for index, name in enumerate(rows)
        ]

    options = KeyedOptions(leave_one_out=leave_one_out)
    plain, spanned = _credit(_group(*agents(False)), options), _credit(_group(*agents(True)), options)

    for left, right in zip(plain.samples(), spanned.samples(), strict=True):
        assert torch.equal(left.advantages, right.advantages)


def test_an_agent_alone_in_its_key_under_leave_one_out_trains_its_spans_against_each_other():
    """Its baseline is its own reward, so it gets 0 as an agent alone in its key does, in parts that sum to 0."""

    alone = _agent("a", 0.75, [[("a1", 2)], [("a2", 1)]], [(["a1"], 0.25), (["a2"], 0.5)], role="worker")
    peers = _agent("p", 1.0, [[("p1", 1)]]), _agent("q", 0.0, [[("q1", 1)]])
    _credit(_group(alone, *peers), KeyedOptions(leave_one_out=True))

    assert _by_call(alone) == pytest.approx({"a1": 0.25 - 0.75 / 2, "a2": 0.5 - 0.75 / 2})
    assert sum(_by_call(alone).values()) == pytest.approx(0.0)
    assert (_by_call(peers[0]), _by_call(peers[1])) == ({"p1": 1.0}, {"q1": -1.0})  # each on the other


def test_an_agent_without_spans_trains_as_before():
    spanned = _agent("s", 0.5, [[("s1", 2)]], [(["s1"], 0.5)])
    plain = _agent("p", 1.0, [[("p1", 2), ("p2", 1)]])
    group = _credit(_group(spanned, plain))

    [sample] = plain.samples
    assert sample.advantages[sample.mask > 0].tolist() == [0.25] * 3  # its reward less the mean, on every action token
    assert sample.advantages[sample.mask <= 0].tolist() == [0.0] * 3
    alone = _credit(_group(_agent("p", 1.0, [[("p1", 2)]]), _agent("q", 0.5, [[("q1", 1)]])))
    assert "credit/keyed/unscored_tokens" not in _metrics(alone)  # no agent names spans: the metrics are as they were
    assert _metrics(group)["credit/keyed/unscored_tokens"] == 0.0


def test_a_call_in_no_span_trains_nothing():
    agent = _agent("a", 0.75, [[("a1", 2), ("a2", 3)]], [(["a1"], 0.75)])
    group = _credit(_group(agent, _agent("b", 0.25, [[("b1", 1)]])))

    assert _by_call(agent) == pytest.approx({"a1": 0.75 - 0.5, "a2": 0.0})
    assert _metrics(group)["credit/keyed/unscored_tokens"] == 3.0


def test_an_unscored_span_trains_nothing():
    agent = _agent("a", 0.75, [[("a1", 2), ("a2", 3)]], [(["a1"], 0.75), (["a2"], None)])
    group = _credit(_group(agent, _agent("b", 0.25, [[("b1", 1)]])))

    # The baseline is shared by its scored spans alone: an unscored span is no span of 0.
    assert _by_call(agent) == pytest.approx({"a1": 0.75 - 0.5, "a2": 0.0})
    assert _metrics(group)["credit/keyed/unscored_tokens"] == 3.0


@pytest.mark.parametrize("spans", [[], [(["a1", "a2"], None)]])
def test_an_agent_whose_credit_lies_in_no_span_enters_the_baseline_and_trains_nothing(spans):
    agent = _agent("a", 0.6, [[("a1", 2), ("a2", 3)]], spans)
    peer = _agent("b", 0.2, [[("b1", 1)]])
    group = _credit(_group(agent, peer))

    assert _by_call(agent) == {"a1": 0.0, "a2": 0.0}
    assert _by_call(peer) == pytest.approx({"b1": 0.2 - 0.4})  # its baseline holds the agent's reward
    assert _metrics(group)["credit/keyed/unscored_tokens"] == 5.0


def test_an_unscored_agent_trains_nothing_whatever_its_spans():
    unscored = _agent("u", None, [[("u1", 2)]], [(["u1"], 1.0)])
    a, b = _agent("a", 1.0, [[("a1", 1)]]), _agent("b", 0.0, [[("b1", 1)]])
    _credit(_group(unscored, a, b))

    assert _by_call(unscored) == {"u1": 0.0}
    assert (_by_call(a), _by_call(b)) == ({"a1": 0.5}, {"b1": -0.5})  # it enters no baseline


def test_two_spans_merged_in_one_row_train_on_their_own_advantages():
    agent = _agent("a", 1.0, [[("a1", 2), ("a2", 2)]], [(["a1"], 0.125), (["a2"], 0.875)])
    _credit(_group(agent, _agent("b", 0.0, [[("b1", 1)]])))

    # The baseline 0.5, a quarter for each of its two spans; its observation tokens untouched.
    [sample] = agent.samples
    assert sample.advantages.tolist() == [0.0, -0.125, -0.125, 0.0, 0.625, 0.625, 0.0]


def test_the_token_advantage_is_the_mean_size_of_what_the_action_tokens_train_with():
    """a's two spans train on 1/8 - 1/4 and 7/8 - 1/4 over two tokens each, and b, which names no spans, on -1/2 over
    one: the tokens' mean absolute advantage is 2/5, where the agents' is 1/2."""

    agent = _agent("a", 1.0, [[("a1", 2), ("a2", 2)]], [(["a1"], 0.125), (["a2"], 0.875)])
    group = _credit(_group(agent, _agent("b", 0.0, [[("b1", 1)]])))

    metrics = _metrics(group)
    assert metrics["credit/token_advantage/abs_avg"] == pytest.approx(0.4)
    assert metrics["credit/agent_advantage/abs_avg"] == pytest.approx(0.5)


def test_a_row_without_calls_of_an_agent_with_spans_stops_the_credit():
    agent = _agent("a", 1.0, [[("a1", 2)]], [(["a1"], 1.0)])
    agent.samples[0].calls = ()  # its conversion recorded none: which span its tokens lie in is unknown
    group = _group(agent, _agent("b", 0.0, [[("b1", 1)]]))

    with pytest.raises(CreditEstimatorError, match="a row of agent a, whose trajectory names spans, names none of its"):
        asyncio.run(assign_credit(Keyed(_SETUP), group))


# --- the workflow ----------------------------------------------------------------------------------------------------


def _workflow(estimator: Any = None) -> GroupRolloutWorkflow:
    """The training workflow of one entry, ``default``, credited by ``estimator`` (None: the workflow's own credit)."""

    async def rollout(task: Any, rollout_config: Any) -> dict:
        raise AssertionError("not run")

    cls: Any = GroupRolloutWorkflow  # built with stand-ins for its task and its model
    workflow = cls(
        rollout_fn=rollout,
        get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
        config=WorkflowConfig(group_size=2),
        model_info=SimpleNamespace(
            model_name="m", renderer_name="qwen3_instruct", renderer_kwargs={}, llm=SimpleNamespace(version=4)
        ),
    )
    workflow.entry_configs = {"default": WorkflowConfig(group_size=2)}
    workflow.entry_credit = {"default": estimator} if estimator is not None else {}
    return workflow


def _run(workflow: GroupRolloutWorkflow, results: Sequence[tdp.TrajectoryCollectionResult]) -> Any:
    async def single(data: dict, rollout_number: int) -> tdp.TrajectoryCollectionResult:
        return results[rollout_number]

    setattr(workflow, "arun_episode_single", single)
    tracker: Any = SimpleNamespace(scalar=lambda **values: None)
    return asyncio.run(runner.run_group_guarded(workflow, {"task_id": "task"}, watchdog=None, tracker=tracker))


def _members(*, tag_datums: bool) -> list[tdp.TrajectoryCollectionResult]:
    """Two rollouts whose workers name spans (one scored span each, and a call in none), as the converter returns
    them."""

    return [
        _converted(1.0, [{"calls": ["w1", "w2"], "reward": 0.875}], tag_datums=tag_datums),
        _converted(0.5, [{"calls": ["w3"], "reward": 0.5}], tag_datums=tag_datums),
    ]


class _ReadsAgents(CreditEstimatorBase):
    """An estimator that reads agents and no spans."""

    needs_agents = True

    async def assign(self, group: CreditGroup) -> None:
        for agent in group.agents():
            for sample in agent.samples:
                sample.set_advantage(agent.reward or 0.0)


@pytest.mark.parametrize(
    ("estimator", "credit"),
    [
        (None, r"the workflow's own credit of environments\[default\]"),
        (GroupCredit(_SETUP), r"the credit estimator test of environments\[default\]"),
        (_ReadsAgents(_SETUP), r"the credit estimator test of environments\[default\]"),
    ],
)
def test_an_estimator_that_reads_no_spans_refuses_a_group_with_spans(estimator, credit):
    """Its credit would train every call of the worker on the worker's reward, which its spans say it did not earn
    whole: the run stops, at the group's credit, naming the credit, the entry, the agent and the task."""

    tagged = bool(getattr(estimator, "needs_agents", False))
    with pytest.raises(
        CreditEstimatorError,
        match=rf"^{credit} reads no spans, and agent w's trajectory names spans \(misc\['spans'\]\), on task task: "
        r"credit the entry with an estimator that reads them \(reads_spans\), such as keyed$",
    ):
        _run(_workflow(estimator), _members(tag_datums=tagged))


def test_an_evaluation_group_whose_agents_name_spans_runs_and_is_not_refused():
    """An evaluation's groups train nothing, so no call of theirs trains on a reward its spans say it did not earn
    whole: the group runs under the workflow's own credit, the one credit an evaluation has."""

    evaluation = _workflow()
    evaluation.phase = "eval"
    output = _run(evaluation, _members(tag_datums=False))

    assert output is not None and report_of(output).verdict == "trained"


def test_a_kept_group_resumes_with_its_span_advantages():
    output = _run(_workflow(Keyed(_SETUP, KeyedOptions(key=["role"]))), _members(tag_datums=True))
    assert isinstance(output, PolicyGroupOutput)

    def advantages(samples: Sequence[TrainingSample]) -> list[dict[str, float]]:
        return [
            {call: float(sample.advantages[start]) for call, start, _ in sample.calls}
            for sample in samples
            if sample.role == "worker"
        ]

    # The workers' rewards 1.0 and 0.5 (their mean 0.75), each one's span on its credit less the mean; the calls in no
    # span (the first's w3, the second's w1 and w2) train nothing.
    assert advantages(output.samples) == [
        {"w1": 0.125, "w2": 0.125},
        {"w3": 0.0},
        {"w1": 0.0, "w2": 0.0},
        {"w3": -0.25},
    ]
    encoded = encode_group((0, 3), entry="default", label=None, record=None, output=output)
    assert encoded is not None
    _, _, _, _, decoded = decode_group(pickle.loads(pickle.dumps(encoded)))
    assert advantages(decoded.samples) == advantages(output.samples)
    assert [datum.loss_fn_inputs["advantages"].tolist() for datum in rows_by_policy(decoded, MAIN)[MAIN]] == [
        datum.loss_fn_inputs["advantages"].tolist() for datum in rows_by_policy(output, MAIN)[MAIN]
    ]


# --- the workflows platoon train builds ------------------------------------------------------------------------------

HERE = __name__


def tasks(config: Any, split: str) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}"} for index in range(4)]


async def rollout(task: Any, rollout_config: Any) -> dict:
    raise AssertionError("not run")


def get_task(task_id: str) -> Task:
    return Task(id=task_id)


def _built(tmp_path: Path, credit: str | None) -> tuple[Any, Any]:
    """The training and the evaluation workflow ``platoon train`` builds for a run that evaluates, of one entry, ``a``,
    credited by ``credit`` (None: the workflow's own credit)."""

    entry = {
        "name": "a",
        "dataset_loader": f"{HERE}:tasks",
        "rollout": f"{HERE}:rollout",
        "task_loader": f"{HERE}:get_task",
    }
    groups = {"group_size": 2}  # the two members of _members
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "model_name": "m",
                    "renderer_name": "r",
                    "batch_size": 4,
                    "max_training_steps": 4,
                    "workflow_config": groups,
                },
                "eval": {"every": 2, "workflow_config": groups},
                "watchdog": {"enabled": False},
                "log_path": str(tmp_path / "logs"),
                "environments": [entry if credit is None else {**entry, "credit": credit}],
            }
        )
    )
    return built_workflows(resolve_config(path, [], PlatoonTinkerRLTrainerConfig).config)


@pytest.mark.parametrize("credit", [None, "group", "keyed"])
def test_platoon_train_s_evaluation_runs_a_group_whose_agents_name_spans(tmp_path, credit):
    """Whatever credits its training, the run's evaluation has no estimator, and its groups train nothing: a group
    whose agents name spans runs there."""

    _, evaluation = _built(tmp_path, credit)
    output = _run(evaluation, _members(tag_datums=False))

    assert output is not None and report_of(output).verdict == "trained"


@pytest.mark.parametrize(
    ("credit", "named"),
    [(None, "the workflow's own credit"), ("group", r"the credit estimator group of environments\[a\]")],
)
def test_platoon_train_s_training_still_refuses_a_group_with_spans_under_a_credit_that_reads_none(
    tmp_path, credit, named
):
    train, _ = _built(tmp_path, credit)
    with pytest.raises(CreditEstimatorError, match=rf"^{named} reads no spans, and agent w's trajectory names spans"):
        _run(train, _members(tag_datums=False))


# --- the conformance check -------------------------------------------------------------------------------------------


def test_the_check_credits_an_estimator_that_reads_spans_over_agents_that_name_them():
    group = fixture_group(0, tagged=True, spans=True)
    assert all(agent.spans is not None for agent in group.agents())
    assert {span.reward is None for agent in group.agents() for span in agent.spans or ()} == {True, False}
    assert not run_checks(Keyed)


class _SpansWithoutAgents(CreditEstimatorBase):
    reads_spans = True

    async def assign(self, group: CreditGroup) -> None:
        pass


def test_the_check_refuses_an_estimator_that_reads_spans_but_no_agents():
    assert run_checks(_SpansWithoutAgents) == {
        "protocol": [
            "'_SpansWithoutAgents' is not a credit estimator: it reads spans (reads_spans) but not agents "
            "(needs_agents), and spans come on its agents"
        ]
    }
