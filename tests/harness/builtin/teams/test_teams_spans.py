"""The ``platoon-harness`` program's finalizer names each span of every agent of a producer's role, on rollouts of the
lockbox through platoon's gateway: each span by the calls its agent reported once that have a step and the value of the
producer's row on its name, and each trajectory's ``misc["team"]`` each span's row; and platoon's ``credit: keyed`` then
trains each scored span's calls on its value less its agent's baseline shared equally by the agent's scored spans, and
every other call on nothing.  On rows a toy producer writes through ``Session.annotate``, and on ``propagate@4``'s, in a
tree and a mesh.  An evaluation's group of such rollouts, which trains nothing, runs, where a training group under a
credit that reads no spans stops the run."""

from __future__ import annotations

import asyncio
import copy
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from platoon_harness import Session, Team
from platoon_harness.credit import PROPAGATE
from platoon_harness.primitives import Span
from platoon_harness.testing import AskModel, Call, Say
from teams_toys import (
    LEDGER,
    RATING,
    SERVED,
    SPANNING,
    UNLOCK,
    episode,
    named,
    prove,
    rollout,
    scripted,
    session_name,
    team,
    wayward,
)

import platoon as pt
from platoon.rewards import delegation_capped
from platoon.testing import ScriptedPolicy, tool_call
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import CreditGroup, CreditSetup, PolicyView
from platoon.train.credit.keyed import Keyed
from platoon.train.credit.runtime import CreditEstimatorError
from platoon.train.policy.spec import MAIN
from platoon.train.workers.run import DEFAULT_REWARD_PROCESSOR
from platoon.train.workflow.crediting import member_rollout
from platoon.train.workflow.group import GroupRolloutWorkflow
from platoon.train.workflow.protocol import WorkflowSetup
from platoon.utils.stats_tracker import DeferredStats, StatsTracker
from platoon.utils.tinker_data_processing import get_train_data_for_trajectory_collection
from platoon.utils.trajectory_status import EXCLUDE_FROM_TRAINING_MISC_KEY, TRAJECTORY_INVALID_MISC_KEY

GOAL = "Open the box. Its code is 1."


def spawning(*names: str) -> list[Call]:
    return [Call("agent_spawn", {"name": name, "child_role": "worker", "goal": GOAL}) for name in names]


def board(lead: str, worker: str, **worker_role: Any) -> str:
    """A manager-worker team on the board ``notes``, whose lead runs on ``lead`` and each worker on ``worker``."""

    roles = {
        "lead": {"harness": lead, "may_spawn": ["worker"], "approvals": "allow", "stays": False},
        "worker": {"harness": worker, "approvals": "allow", **worker_role},
    }
    topology = {"manager-worker": {"manager": "lead", "workers": "worker"}}
    return named(Team.model_validate({"roles": roles, "topology": topology}))


LEAD = scripted(spawning("w1"), [AskModel()])
BOARD = board(LEAD, scripted([AskModel()]))


def posting(code: str = "1") -> Callable[[str], str]:
    """The model of a board team: each worker posts its try, tries ``code`` and finishes, a model call for each and one
    after, so a span for its post and one for the rest; and its lead, once told, reads ``w1``'s try and says so."""

    def model(prompt: str) -> str:
        if "You are @lead" in prompt:
            read = tool_call("board_get", board="notes", name="try-w1")
            return "the worker has it" if "<tool_response>" in prompt else read
        found = re.search(r"You are @(\w+)", prompt)
        assert found is not None
        name = found[1]
        steps = [
            tool_call("board_post", board="notes", name=f"try-{name}", gist=f"trying {code}"),
            tool_call(UNLOCK, code=code),
            tool_call("agent_finish", gist="tried"),
            "done",
        ]
        return steps[min(prompt.count("<tool_response>"), len(steps) - 1)]

    return model


def spanned(*values: float | None, **settings: Any) -> dict[str, Any]:
    """An entry's ``harness_options`` that train its workers on the toy producer's rows: ``values`` on each worker's
    spans, oldest first, and their sum on its start."""

    return {"rewards": {"worker": {"producer": SPANNING, "values": list(values), **settings}}}


@pytest.fixture
def proven(_sdk_home: Path) -> Path:
    prove(_sdk_home)
    return _sdk_home


def by_agent(run: Any) -> dict[str | None, pt.Trajectory]:
    return {trajectory.misc["team"]["agent"]: trajectory for trajectory in run.trajectories}


def completions(trajectory: pt.Trajectory) -> list[str]:
    return [step.misc["action_misc"]["completion_id"] for step in trajectory.steps]


def record(run: Any, agent: str) -> tuple[list[Span], list[Any]]:
    """An agent's spans in a rollout's session, and the session's rows."""

    async def read() -> tuple[list[Span], list[Any]]:
        async with Session.attach(session_name(run)) as ended:
            return list((await ended.spans(f"@{agent}")).spans), await ended.annotations()

    return asyncio.run(read())


def row_on(rows: Sequence[Any], target: str, producer: str = SPANNING) -> Any:
    """The row ``producer`` wrote on ``target``, or None."""

    return next((row for row in rows if (row.target, row.producer) == (target, producer)), None)


def start_row(rows: Sequence[Any], agent: str, producer: str = SPANNING) -> Any:
    """The row ``producer`` wrote on an agent's first start."""

    (row,) = [row for row in rows if row.producer == producer and row.target.startswith(f"agent/{agent}@")]
    return row


def credited(
    runs: Sequence[Any], processor: Callable[[dict], Any] = DEFAULT_REWARD_PROCESSOR, *, spans: bool = True
) -> CreditGroup:
    """``runs`` as one group, each converted as a worker converts it, tagged with its agents, under ``processor``, and
    without the spans its trajectories name unless ``spans``; and credited by ``credit: keyed`` with its default key,
    task, policy and role."""

    task = runs[0].collection.id.rpartition("-")[0]
    results = []
    for run in runs:
        collection = run.collection.to_dict()
        for trajectory in collection["trajectories"].values() if not spans else ():
            trajectory["misc"].pop("spans", None)
        results.append(
            get_train_data_for_trajectory_collection(
                collection, run.interactions, task, 0, reward_processor=processor, tag_datums=True
            )
        )
    group = CreditGroup(
        task_id=task,
        label=None,
        entry="default",
        settings=WorkflowConfig(),
        members=[member_rollout(index, result, MAIN, tagged=True) for index, result in enumerate(results)],
        policies={MAIN: PolicyView(MAIN)},
        versions={MAIN: 0},
        stats=DeferredStats(),
    )
    setup = CreditSetup(name="keyed", entry="default", settings=WorkflowConfig(), policies=(MAIN,))
    asyncio.run(Keyed(setup).assign(group))
    return group


def metrics(group: CreditGroup) -> dict[str, float]:
    """The metrics ``credit: keyed`` recorded on ``group``."""

    tracker, stats = StatsTracker(), group.stats
    assert isinstance(stats, DeferredStats)
    stats.record(tracker)
    return tracker.export()


def trained(group: CreditGroup, role: str) -> dict[str, float]:
    """The advantage each call of the agents of ``role`` trains on, over the tokens of it each row trains: one each."""

    found: dict[str, set[float]] = defaultdict(set)
    for agent in group.agents():
        for sample in agent.samples if agent.role == role else ():
            for call, start, end in sample.calls:
                found[call].update(round(value, 5) for value in sample.advantages[start:end].tolist())
    assert all(len(values) == 1 for values in found.values()), found
    return {call: values.pop() for call, values in found.items()}


def as_spans_say(group: CreditGroup, role: str) -> dict[str, float]:
    """What each call of the agents of ``role`` that a row trains should train on, by the spans its trajectory names:
    a scored span's value less its agent's baseline, the mean of the group's scored agents of its role, shared equally
    by its scored spans; and 0 for any other call, and for each call of an unscored agent."""

    agents = [agent for agent in group.agents() if agent.role == role]
    rewards = [agent.reward for agent in agents if agent.reward is not None]
    baseline = sum(rewards) / len(rewards)
    expected: dict[str, float] = {}
    for agent in agents:
        if agent.reward is None:
            continue
        scored = [(span.calls, span.reward) for span in agent.spans or () if span.reward is not None]
        for calls, value in scored:
            expected |= dict.fromkeys(calls, round(value - baseline / len(scored), 5))
    return {call: expected.get(call, 0.0) for call in trained(group, role)}


def test_every_agent_of_a_producer_role_names_all_its_spans_scored_or_not(proven):
    """Two workers of a producer's role, each cut at its post: each names both its spans, oldest first, the first with
    no row as None, and the root, whose role names no producer, names none."""

    crew = board(scripted(spawning("w1", "w2"), [AskModel()]), scripted([AskModel()]))

    run = episode(crew, posting(), harness_options=spanned(None, 0.25), agents={"budget": "depth_aware"})

    for name in ("w1", "w2"):
        worker = by_agent(run)[name]
        post, *rest = completions(worker)
        assert worker.misc["spans"] == [{"calls": [post], "reward": None}, {"calls": rest, "reward": 0.25}]
    assert "spans" not in run.root.misc and "spans" not in run.root.misc["team"]


def test_an_agent_of_a_role_that_names_no_producer_names_no_spans(proven):
    """A lead that spawns a worker, whose role names a producer, and a helper, whose role names none: the helper's
    trajectory names no spans and its ``misc["team"]`` no span rows, while the worker's names its one span."""

    finishing = scripted([AskModel(), Call("agent_finish", {"gist": "unlocked"})])
    spawns = [*spawning("w1"), Call("agent_spawn", {"name": "h1", "child_role": "helper", "goal": GOAL})]
    crew = named(
        team(
            lead={"harness": scripted(spawns, [Say("done")]), "may_spawn": ["worker", "helper"]},
            worker={"harness": finishing},
            helper={"harness": finishing},
        )
    )

    run = episode(crew, harness_options=spanned(0.5), agents={"budget": "depth_aware"})

    helper, worker = by_agent(run)["h1"], by_agent(run)["w1"]
    assert "spans" not in helper.misc and set(helper.misc["team"]) == {"session", "agent", "row"}
    assert [span["reward"] for span in worker.misc["spans"]] == [0.5] and len(worker.misc["team"]["spans"]) == 1


def test_each_span_names_the_calls_its_agent_reported_once_with_a_step(proven):
    """A worker whose harness reports its second call twice: that call lies in its span, which names only the calls
    reported once, so no span names a call whose purpose is unknown, and every other call with a step is named once."""

    reported: list[str] = []

    def second_twice(call: Any) -> Any:
        reported.append(call.call_id)
        return [call, call] if len(reported) == 2 else call

    crew = board(LEAD, wayward([AskModel()], rewrite=second_twice))

    run = episode(crew, posting(), harness_options=spanned(0.5, 0.25), agents={"budget": "depth_aware"})

    worker = by_agent(run)["w1"]
    post, twice, finish, last = completions(worker)
    assert worker.misc["spans"] == [{"calls": [post], "reward": 0.5}, {"calls": [finish, last], "reward": 0.25}]
    assert twice == reported[1] and worker.steps[1].misc[EXCLUDE_FROM_TRAINING_MISC_KEY]


def test_each_trajectory_names_its_span_rows(proven):
    """A worker's ``misc["team"]`` names the row its producer wrote on each of its spans, in its ``misc["spans"]``'s
    order, and None for a span with none."""

    run = episode(BOARD, posting(), harness_options=spanned(None, 0.25), agents={"budget": "depth_aware"})

    worker = by_agent(run)["w1"]
    spans, rows = record(run, "w1")
    assert worker.misc["team"]["spans"] == [None, row_on(rows, spans[1].ref).id]
    assert worker.misc["team"]["row"] == start_row(rows, "w1").id


def test_a_span_row_never_counts_as_its_agents_row(proven):
    """The worker trains on its row on its first start, the sum of its spans' here, and no span's row is ever its
    agent's: its reward and its ``misc["team"]``'s row are its start's."""

    run = episode(BOARD, posting(), harness_options=spanned(0.5, 0.25), agents={"budget": "depth_aware"})

    worker = by_agent(run)["w1"]
    spans, rows = record(run, "w1")
    start = start_row(rows, "w1")
    assert worker.reward == start.value == 0.75
    assert worker.misc["team"]["row"] == start.id not in worker.misc["team"]["spans"]
    assert [row_on(rows, span.ref).value for span in spans] == [0.5, 0.25]


def test_a_non_scored_agent_row_makes_its_trajectory_ineligible_whatever_its_spans(proven):
    """A producer that writes an invalid row on the worker's start and scored rows on its spans: the spans are named
    with their values, and the trajectory is marked invalid all the same, so ``keyed`` trains none of its calls."""

    run = episode(
        BOARD, posting(), harness_options=spanned(0.5, 0.25, status="invalid"), agents={"budget": "depth_aware"}
    )

    worker = by_agent(run)["w1"]
    group = credited([run])
    assert worker.misc[TRAJECTORY_INVALID_MISC_KEY]
    assert [span["reward"] for span in worker.misc["spans"]] == [0.5, 0.25]
    assert [agent.reward for agent in group.agents() if agent.role == "worker"] == [None]
    assert trained(group, "worker") == {}


def test_a_span_with_no_row_trains_nothing(proven):
    """Two rollouts, each worker's post with no row: ``keyed`` centres each worker on the mean of the two, and trains
    its finishing span's calls on its value less that, and its post's on nothing."""

    runs = [
        episode(BOARD, posting(), harness_options=spanned(None, value), agents={"budget": "depth_aware"})
        for value in (0.25, 0.75)
    ]

    group = credited(runs)
    trains = trained(group, "worker")
    for run, advantage in zip(runs, (-0.25, 0.25), strict=True):
        post, *rest = completions(by_agent(run)["w1"])
        assert trains[post] == 0.0 and {trains[call] for call in rest} == {advantage}
    assert trains == as_spans_say(group, "worker")


def test_a_producer_that_writes_rows_on_agents_starts_alone_trains_none_of_its_roles_calls(proven):
    """Two rollouts whose producer rates each worker's start, 0.25 and 0.75, and writes no row on any span: each
    worker names its spans, none scored, so ``keyed`` trains every one of its calls on nothing and counts each of its
    action tokens as lying in no scored span, where the same rows with the spans dropped train its calls on -0.25 and
    0.25."""

    runs = [
        episode(
            BOARD,
            posting(),
            harness_options={"rewards": {"worker": {"producer": RATING, "value": value}}},
            agents={"budget": "depth_aware"},
        )
        for value in (0.25, 0.75)
    ]

    group = credited(runs)
    assert [[span["reward"] for span in by_agent(run)["w1"].misc["spans"]] for run in runs] == [[None, None]] * 2
    assert set(trained(group, "worker").values()) == {0.0}
    assert set(trained(credited(runs, spans=False), "worker").values()) == {-0.25, 0.25}
    tokens = sum(
        int((sample.mask > 0).sum()) for agent in group.agents() if agent.role == "worker" for sample in agent.samples
    )
    assert tokens > 0 and metrics(group)["credit/keyed/unscored_tokens"] == tokens


def test_a_span_with_no_step_trains_nothing(proven):
    """A worker that stays, told to go on after its first turn, whose second turn the gateway answers with a budget stop
    alone: that turn is a span with no step, named with no call, so its value trains nothing, while it counts among
    the worker's scored spans, as each scored span does."""

    crew = board(
        scripted(
            [*spawning("w1"), Call("wait", {"from": ["@w1"]})], [Call("message_send", {"to": "@w1", "gist": "go on"})]
        ),
        scripted([AskModel()]),
        stays=True,
    )

    def model(prompt: str) -> str:
        return "done" if "You are @lead" in prompt else "tried"

    run = episode(
        crew,
        model,
        harness_options=spanned(0.5, 0.25),
        agents={"budget": "depth_aware", "child_max_steps": 1, "child_grading": "env"},
    )

    worker = by_agent(run)["w1"]
    (tried,) = completions(worker)
    spans, _ = record(run, "w1")
    assert len(spans) == 2 and spans[0].turn != spans[1].turn and worker.misc["budget"]["stops"] == 1
    assert worker.misc["spans"] == [{"calls": [tried], "reward": 0.5}, {"calls": [], "reward": 0.25}]
    assert trained(credited([run]), "worker") == {tried: 0.5 - 0.75 / 2}


@pytest.mark.parametrize("processor", [DEFAULT_REWARD_PROCESSOR, delegation_capped], ids=["default", "delegation"])
def test_an_agent_of_one_span_trains_on_its_agent_row_under_the_processor_gate(proven, processor):
    """A worker whose one span holds all its credit, under either reward processor the load check lets a producer's
    row through: ``keyed`` trains each of its calls as it would an agent that names no spans."""

    worker = scripted([AskModel(), Call("agent_finish", {"gist": "unlocked"})])
    crew = named(
        team(
            lead={"harness": scripted(spawning("w1"), [Say("done")]), "may_spawn": ["worker"]},
            worker={"harness": worker},
        )
    )
    runs = [episode(crew, harness_options=spanned(value), agents={"budget": "depth_aware"}) for value in (0.75, 0.25)]

    group = credited(runs, processor)

    assert [len(by_agent(run)["w1"].misc["spans"]) for run in runs] == [1, 1]
    assert trained(group, "worker") == trained(credited(runs, processor, spans=False), "worker")
    assert set(trained(group, "worker").values()) == {0.25, -0.25}


QUEUE = {"kind": "queue", "writers": "all", "readers": "all", "delivery": "pull", "max_inflight": 2}


def test_a_persistent_worker_trains_each_task_on_its_own_span(proven):
    """A worker that claims and completes three tasks in one turn, a model call for each, is cut at each result, so each
    task's calls train on that task's span alone."""

    adds = [Call("task_add", {"queue": "tasks", "gist": f"task {number}"}) for number in (1, 2, 3)]
    roles = {
        "lead": {"harness": scripted([*adds, *spawning("w1")], [AskModel()]), "may_spawn": ["worker"], "stays": False},
        "worker": {"harness": scripted([AskModel()])},
    }
    roles = {name: {"approvals": "allow", **role} for name, role in roles.items()}
    crew = named(Team.model_validate({"roles": roles, "topology": {"channels": {"tasks": QUEUE}}}))
    script = []
    for number in (1, 2, 3):
        done = {"queue": "tasks", "task": str(number), "gist": f"task {number} done"}
        script += [tool_call("task_claim", queue="tasks"), tool_call("task_complete", done)]

    run = episode(
        crew,
        ScriptedPolicy([*script, "all three are done", "the worker did them"]),
        harness_options=spanned(0.5, 0.25, 0.125),
        agents={"budget": "depth_aware"},
    )

    calls = completions(by_agent(run)["w1"])
    tasks = [calls[0:2], calls[2:4], calls[4:]]
    assert [span["calls"] for span in by_agent(run)["w1"].misc["spans"]] == tasks
    group = credited([run])
    assert trained(group, "worker") == as_spans_say(group, "worker")
    assert [{trained(group, "worker")[call] for call in task} for task in tasks] == [
        {round(value - 0.875 / 3, 5)} for value in (0.5, 0.25, 0.125)
    ]


def test_a_harness_that_gives_no_call_id_names_spans_with_no_calls_and_trains_nothing(proven):
    """A worker whose harness reports each call with no id, as Codex does: no write joins a call, so its turn is one
    span, which names no call, as no id joins one to a step, and none of its steps trains."""

    crew = board(LEAD, wayward([AskModel()], rewrite=lambda call: call.model_copy(update={"call_id": None})))

    run = episode(crew, posting(), harness_options=spanned(0.5), agents={"budget": "depth_aware"})

    worker = by_agent(run)["w1"]
    spans, _ = record(run, "w1")
    assert [span.writes for span in spans] == [()]
    assert worker.misc["spans"] == [{"calls": [], "reward": 0.5}]
    assert all(step.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] for step in worker.steps)
    assert trained(credited([run]), "worker") == {}


def test_a_harness_that_joins_no_tool_call_to_its_call_names_one_span_a_turn(proven):
    """A worker whose harness sends no tool call's id, so that the record joins none of its writes to a call, as on
    OpenHands: its turn is one span, holding every call of it, where a harness that joins them is cut at its post."""

    joined = episode(BOARD, posting(), harness_options=spanned(0.5, 0.25), agents={"budget": "depth_aware"})
    unjoined = episode(
        board(LEAD, wayward([AskModel()], ids=False)),
        posting(),
        harness_options=spanned(0.5, 0.25),
        agents={"budget": "depth_aware"},
    )

    worker = by_agent(unjoined)["w1"]
    assert [len(span["calls"]) for span in by_agent(joined)["w1"].misc["spans"]] == [1, 3]
    assert worker.misc["spans"] == [{"calls": completions(worker), "reward": 0.5}]


def propagating(*roles: str) -> dict[str, Any]:
    """An entry's ``harness_options`` that train ``roles`` on ``propagate@4``'s rows, retaining half."""

    return {"rewards": {role: {"producer": PROPAGATE, "retained": 0.5} for role in roles}}


def test_a_tree_trains_each_span_on_its_propagate_row_centred_by_keyed_credit(proven):
    """Two rollouts of the board tree under ``propagate@4``, the box opened and not: the lead's last words take the
    worker's report and post, so propagate reaches both its spans; each span trains on the row propagate wrote on its
    name, less the worker's baseline, the mean of the two workers' rows, shared by its two spans."""

    runs = [
        episode(BOARD, posting(code), harness_options=propagating("worker"), agents={"budget": "depth_aware"})
        for code in ("1", "2")
    ]

    for run in runs:
        worker = by_agent(run)["w1"]
        spans, rows = record(run, "w1")
        found = [row_on(rows, span.ref, PROPAGATE) for span in spans]
        assert [span["reward"] for span in worker.misc["spans"]] == [row.value for row in found]
        assert worker.misc["team"]["spans"] == [row.id for row in found]
        assert worker.reward == start_row(rows, "w1", PROPAGATE).value
    opened = by_agent(runs[0])["w1"].misc["spans"]
    assert [span["reward"] for span in opened] == pytest.approx([1 / 12, 1 / 6])
    group = credited(runs)
    assert [len(agent.spans or ()) for agent in group.agents() if agent.role == "worker"] == [2, 2]
    assert trained(group, "worker") == as_spans_say(group, "worker")


SCORE = f"{SERVED}score"
PEERS = named(
    team(
        alpha={"harness": scripted([AskModel()])},
        beta={"harness": scripted([Say("standing by")])},
    )
)


def test_a_mesh_trains_each_span_on_its_propagate_row_centred_by_keyed_credit(proven):
    """The mesh recipe's credit, end to end: three meshes of two peers, whose ledger scores 1, 0 and an invalid grade,
    each peer's span trained on its ``propagate@4`` row.  Each span's value is its row's, the row its ``misc["team"]``
    names; a grade that is not scored gives every row its status, so its peers and their spans are unscored, and the
    converter keeps no row of theirs; and ``credit: keyed`` trains each scored span's calls on its value less the mean
    of the group's scored peers of its role, shared by its scored spans."""

    scripts = [
        [tool_call(SCORE, value=value, invalid=invalid), "done"]
        for value, invalid in ((1.0, False), (0.0, False), (0.9, True))
    ]
    runs = [
        asyncio.run(
            rollout(
                PEERS,
                ScriptedPolicy(script),
                task="test:0",
                service=LEDGER,
                harness_options=propagating("alpha", "beta"),
            )
        )
        for script in scripts
    ]

    for run in runs:
        for peer in run.trajectories[1:]:
            spans, rows = record(run, peer.misc["team"]["agent"])
            found = [row_on(rows, span.ref, PROPAGATE) for span in spans]
            assert peer.misc["team"]["spans"] == [None if row is None else row.id for row in found]
            assert [span["reward"] for span in peer.misc["spans"]] == [
                row.value if row is not None and row.status == "scored" else None for row in found
            ]
            start = start_row(rows, peer.misc["team"]["agent"], PROPAGATE)
            assert peer.misc["team"]["row"] == start.id
            assert peer.reward == start.value if start.status == "scored" else peer.misc[TRAJECTORY_INVALID_MISC_KEY]
    invalid = by_agent(runs[2])["alpha"]
    assert invalid.misc[TRAJECTORY_INVALID_MISC_KEY] and [span["reward"] for span in invalid.misc["spans"]] == [None]
    group = credited(runs)
    alphas = [agent for agent in group.agents() if agent.role == "alpha"]
    assert [bool(agent.samples) for agent in alphas] == [True, True, False]  # the unscored peer: no row
    assert trained(group, "alpha") == as_spans_say(group, "alpha")


async def never(task: Any, rollout_config: Any) -> dict:
    raise AssertionError("not run")


def test_an_evaluation_runs_a_group_of_team_rollouts_whose_agents_name_spans(proven, tmp_path):
    """Two rollouts whose workers name their spans, each converted as an evaluation's worker converts it, asked for no
    tags: as a group of the workflow ``platoon train`` builds for its evaluation, which has no credit estimator and
    whose groups train nothing, they run; as a group of its training workflow, whose own credit reads no spans, they
    stop the run."""

    runs = [
        episode(BOARD, posting(), harness_options=spanned(0.5, value), agents={"budget": "depth_aware"})
        for value in (0.25, 0.75)
    ]
    task = runs[0].collection.id.rpartition("-")[0]
    results = [
        get_train_data_for_trajectory_collection(run.collection.to_dict(), run.interactions, task, 0) for run in runs
    ]

    def group(phase: Literal["train", "eval"]) -> Any:
        setup = WorkflowSetup(
            phase=phase,
            rollout_fn=never,
            get_task_fn=lambda task_id: SimpleNamespace(id=task_id, max_steps=None),
            config=WorkflowConfig(group_size=len(results)),
            entry_configs=None,
            policies={MAIN: SimpleNamespace(model_name="m", llm=SimpleNamespace(version=0))},
            log_path=str(tmp_path),
            stats_scope=phase,
            filter_errors=False,
            reward_processor=DEFAULT_REWARD_PROCESSOR,
        )
        workflow = GroupRolloutWorkflow.from_setup(setup)

        async def member(data: dict, rollout_number: int = 0) -> Any:
            return copy.deepcopy(results[rollout_number])

        workflow.arun_episode_single = member  # type: ignore[method-assign]
        return asyncio.run(workflow.arun_episode({"task_id": task}))

    assert all("spans" in by_agent(run)["w1"].misc for run in runs)
    assert group("eval").verdict == "trained"
    with pytest.raises(CreditEstimatorError, match=r"^the workflow's own credit reads no spans, and agent "):
        group("train")
