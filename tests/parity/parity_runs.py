"""Run a parity scenario in one execution mode, as training runs a task group, and keep everything it produced.

A :class:`Variant` is one way to run the number-search task: a service (``python://`` in process, ``ors+http://``
on an env server; a Harbor dataset or a gym adapter would be others), the probe's adapter for its tools, and the
task's id there.  A :class:`Scenario` is what the group's members do, the same in every variant: their
:class:`~parity_policy.Plan`, the environment's ``agents`` settings, and the rewards the scenario must produce.

:func:`run_group` runs one scenario on one variant through the trainer's own code: the environment entry is routed
by ``TaskRouter``, the group runs in ``GroupRolloutWorkflow.arun_episode`` (rollouts in process, as
``rollout_isolation: inprocess`` runs them), and each member runs ``harness_rollout`` through a ``TinkerLLM`` whose
sampling client is the scripted policy.  :class:`ParityWorkflow` only records: each member's trajectory collection
and model calls, its converted data before the group centers it, the baselines, and the group's output.
"""

from __future__ import annotations

import contextlib
import copy
import socket
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import parity_probe
from parity_policy import MEMBER, RENDERER_NAME, Plan, ScriptedSampler

from platoon.config_defs import InferenceParams, RolloutConfig
from platoon.gateway import set_default_gateway
from platoon.harness.rollout import harness_rollout
from platoon.train.auto import TaskRouter
from platoon.train.components import EnvironmentConfig
from platoon.train.tinker.config_defs import WorkflowConfig
from platoon.train.tinker.proxy import ModelInfo, TinkerLLM, proxy_interactions
from platoon.train.tinker.workflows.group_rollout_workflow import GroupRolloutWorkflow, _SingleRolloutOutcome

MODEL_NAME = "parity/policy"
CHECKPOINT_VERSION = 7
NUMBER_SEARCH_CLASS = "platoon.envs.number_search.env:NumberSearchEnvironment"
# Every variant's environment entry has this name, so the routes its trajectories record differ only in the service.
ENVIRONMENT_NAME = "number-search"
# The task: "Guess the correct number between 119 and 965.", target 447 (number-search's val:0).
TASK_ID = "val:0"


# --------------------------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    """One execution mode of the task.

    ``surface`` is the probe's adapter (``parity_probe.ADAPTERS``): variants with different surfaces call different
    tools.  ``grading`` says what grades an episode (``ors``: the result of the call that ends it; ``harbor``: the
    task's tests, after the harness ends; ``gym``: the return).  The comparison allows the differences these imply,
    and only between variants that differ in them (``parity_compare.ALLOWLIST``).  ``service`` gives the environment
    entry's ``service`` and ``service_kwargs``, given what the test set up (:class:`Endpoints`).
    """

    name: str
    surface: str
    grading: str
    task_id: str
    service: Callable[[Endpoints], tuple[str, dict[str, Any]]]

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Endpoints:
    """What the test set up for the variants: the local ORS env server (``host:port``)."""

    ors_server: str | None = None


IN_PROCESS = Variant(
    name="ors-in-process",
    surface="ors",
    grading="ors",
    task_id=TASK_ID,
    service=lambda endpoints: (f"python://{NUMBER_SEARCH_CLASS}", {}),
)
SERVED = Variant(
    name="ors-served",
    surface="ors",
    grading="ors",
    task_id=TASK_ID,
    service=lambda endpoints: (f"ors+http://{endpoints.ors_server}/NumberSearchEnvironment", {}),
)


@contextlib.contextmanager
def serve_ors(environments: Sequence[type]) -> Iterator[str]:
    """An openreward ``Server`` for ``environments`` on a free local port, in a uvicorn thread; yields ``host:port``."""

    import uvicorn
    from openreward.environments import Server

    from platoon.envs.ors import service as ors_service

    server = Server(list(environments))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    ors_service._SERVER_ENVIRONMENTS.pop(f"http://{host}:{port}", None)  # an earlier server's, on this port
    uv = uvicorn.Server(uvicorn.Config(server.app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=uv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uv.started:
        if time.monotonic() > deadline or not thread.is_alive():
            raise RuntimeError("The ORS env server did not start")
        time.sleep(0.05)
    try:
        yield f"{host}:{port}"
    finally:
        uv.should_exit = True
        thread.join(timeout=10)
        sock.close()


# --------------------------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    """What a group's members do, in every variant.

    ``plans`` has one plan per member.  ``root_rewards`` are the rewards the roots must score, and
    ``child_rewards`` those of each member's subagent (None: the member spawns none), so that two variants that
    agree have also done what the scenario is about.
    """

    name: str
    plans: tuple[Plan, ...]
    root_rewards: tuple[float, ...]
    child_rewards: tuple[float | None, ...] | None = None
    agents: Mapping[str, Any] = field(default_factory=dict)
    max_steps: int = 12

    def __str__(self) -> str:
        return self.name


_NARROW = "Narrow down the hidden number between {low} and {high} with at most %d guesses, then report the range."

SCENARIOS = (
    # Bisection needs 9 guesses for 447 in [119, 965]: patience 3 gives up (no_submit, 0), so the advantages are
    # not trivial.
    Scenario("flat", plans=tuple(Plan(patience) for patience in (3, 9, 9, 12)), root_rewards=(0.0, 1.0, 1.0, 1.0)),
    # The root first has a subagent work on the root's live env (a shared fork).  Members 0 and 2 have it narrow the
    # range with 4 guesses, then need 5 guesses of their own; members 1 and 3 have it find the number and end without a
    # guess of their own: the child's winning guess ended the shared episode, so they score 1 (had the child forked an
    # env of its own, 0).  The child is not graded on its own: it shares the episode's outcome.
    Scenario(
        "subagent-shared-fork",
        plans=(
            Plan(3, delegate=_NARROW % 4),
            Plan(9, delegate=_NARROW % 12, trust_child=True),
            Plan(9, delegate=_NARROW % 4),
            Plan(12, delegate=_NARROW % 12, trust_child=True),
        ),
        root_rewards=(0.0, 1.0, 1.0, 1.0),
        child_rewards=(0.0, 1.0, 1.0, 1.0),
        agents={"fork": "shared", "child_grading": "none", "child_max_steps": 13},
        max_steps=30,
    ),
    # The subagent plays the whole task in a fresh session of its own and is graded there (child_grading: env).  With
    # 12 guesses it finds the number and the root guesses it once; with 3 it reports a range and the root bisects on.
    Scenario(
        "subagent-fresh-fork",
        plans=(
            Plan(12, delegate=_NARROW % 12),
            Plan(3, delegate=_NARROW % 3),
            Plan(12, delegate=_NARROW % 3),
            Plan(9, delegate=_NARROW % 12),
        ),
        root_rewards=(1.0, 0.0, 1.0, 1.0),
        child_rewards=(1.0, 0.0, 0.0, 1.0),
        agents={"fork": "fresh", "child_grading": "env", "child_max_steps": 13},
        max_steps=30,
    ),
)


# --------------------------------------------------------------------------------------------- running a group


@dataclass
class MemberRun:
    """One group member: its plan, what its rollout recorded, and what the trainer made of it."""

    index: int
    plan: Plan
    sampling: list[dict[str, Any]] = field(default_factory=list)  # each model call's sampling parameters
    collection: dict[str, Any] | None = None  # the trajectory collection the rollout returned
    interactions: dict[str, Any] = field(default_factory=dict)  # call id -> TinkerLLMInteraction
    outcome: dict[str, Any] | None = None  # the member's converted data, before the group centered it


@dataclass
class GroupRun:
    """One scenario on one variant."""

    variant: Variant
    scenario: Scenario
    members: list[MemberRun]
    baselines: list[float] | None
    output: dict[str, Any]


async def _recorded_harness_rollout(task: Any, config: Any) -> Any:
    """``harness_rollout``, keeping the collection and the model calls of the member it runs for."""

    member = MEMBER.get()
    collection = await harness_rollout(task, config)
    member.collection = copy.deepcopy(collection)
    member.interactions = dict(proxy_interactions.get())
    return collection


class ParityWorkflow(GroupRolloutWorkflow):
    """``GroupRolloutWorkflow`` that records what each member and the group produced (see the module docstring)."""

    def __init__(self, *, plans: Sequence[Plan], **kwargs: Any) -> None:
        super().__init__(rollout_fn=_recorded_harness_rollout, **kwargs)
        self.members = [MemberRun(index, plan) for index, plan in enumerate(plans)]
        self.baselines: list[float] | None = None

    async def arun_episode_single(self, data: dict, rollout_number: int = 0) -> _SingleRolloutOutcome:
        member = self.members[rollout_number]
        token = MEMBER.set(member)
        try:
            outcome = await super().arun_episode_single(data, rollout_number)
        finally:
            MEMBER.reset(token)
        member.outcome = member_outcome(outcome)  # now: the group centers and filters these datums in place
        return outcome

    def compute_baselines(self, results: list[Any]) -> list[float]:
        self.baselines = super().compute_baselines(results)
        return list(self.baselines)


def environment(variant: Variant, scenario: Scenario, endpoints: Endpoints) -> EnvironmentConfig:
    """The environment entry of ``scenario`` on ``variant``: the same harness, options and agents everywhere."""

    service, service_kwargs = variant.service(endpoints)
    return EnvironmentConfig(
        name=ENVIRONMENT_NAME,
        service=service,
        service_kwargs=service_kwargs,
        harness=parity_probe.PROBE,
        harness_options={"adapter": variant.surface},
        agents=dict(scenario.agents),
    )


async def run_group(
    variant: Variant, scenario: Scenario, *, tokenizer: Any, endpoints: Endpoints, log_path: str
) -> GroupRun:
    """Run ``scenario``'s group on ``variant`` (see the module docstring)."""

    from tinker_cookbook.renderers import get_renderer

    llm = TinkerLLM(
        model_name=MODEL_NAME,
        renderer=get_renderer(RENDERER_NAME, tokenizer),
        tokenizer=tokenizer,
        sampling_client=ScriptedSampler(tokenizer),
    )
    llm.set_version(CHECKPOINT_VERSION)
    group_size = len(scenario.plans)
    config = WorkflowConfig(
        group_size=group_size,
        rollout_config=RolloutConfig(
            max_steps=scenario.max_steps,
            timeout=120,
            verbose=False,
            # Sampling that differs from the gateway's defaults, so a mode that dropped it would show.
            inference_params=InferenceParams(temperature=0.7, top_p=0.95, max_completion_tokens=256),
        ),
        leave_one_out_baseline=True,
        depth_level_weighting=True,  # datums carry traj_depth and traj_start
        min_successful_group_size=group_size,  # every member must be trainable, or the group trains nothing
    )
    workflow = ParityWorkflow(
        plans=scenario.plans,
        get_task_fn=TaskRouter([environment(variant, scenario, endpoints)]),
        config=config,
        model_info=ModelInfo(llm=llm, model_name=MODEL_NAME, renderer_name=RENDERER_NAME),
        log_path=log_path,
        stats_scope=f"parity/{scenario.name}/{variant.name}",
        filter_errors=True,
    )
    llm.set_as_default()
    try:
        output = await workflow.arun_episode({"task_id": variant.task_id})
    finally:
        set_default_gateway(None)
    return GroupRun(variant, scenario, workflow.members, workflow.baselines, group_output(output))


# --------------------------------------------------------------------------------------------- recording


def datum_record(datum: Any) -> dict[str, Any]:
    """A ``tinker.Datum`` as plain data: its input chunks' tokens and every loss input."""

    chunks = [list(getattr(chunk, "tokens", ())) or repr(chunk) for chunk in datum.model_input.chunks]
    inputs = {name: value.to_torch().tolist() for name, value in sorted(datum.loss_fn_inputs.items())}
    return {"model_input": chunks, "loss_fn_inputs": inputs}


def member_outcome(outcome: _SingleRolloutOutcome) -> dict[str, Any]:
    """A member's outcome as the group got it, before centering: its converted result and workload."""

    result = outcome.result
    return {
        "disposition": outcome.disposition,
        "observed": outcome.observed,
        "workload": vars(outcome.workload).copy(),
        "result": None
        if result is None
        else {
            "task_reward": result.task_reward,
            "task_reward_valid": result.task_reward_valid,
            "root_rewards_dict": dict(result.root_rewards_dict),
            "num_policy_excluded_datums": result.num_policy_excluded_datums,
            "trajectory_stats": [vars(stats).copy() for stats in result.trajectory_stats],
            "sampling": result.subagent_sampling_stats.to_metrics(),
            "datums": [datum_record(datum) for datum in result.datums],
        },
    }


class _RecordingTracker:
    """Takes the stats calls a group's ``DeferredStats`` replays, as plain data."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def _record(self, method: str, kwargs: dict[str, Any]) -> None:
        values = {key: value.tolist() if hasattr(value, "tolist") else value for key, value in kwargs.items()}
        self.calls.append([method, {key: values[key] for key in sorted(values)}])

    def scalar(self, **kwargs: Any) -> None:
        self._record("scalar", kwargs)

    def denominator(self, **kwargs: Any) -> None:
        self._record("denominator", kwargs)

    def stat(self, **kwargs: Any) -> None:
        kwargs["reduce_type"] = None if kwargs.get("reduce_type") is None else str(kwargs["reduce_type"])
        self._record("stat", kwargs)


def group_output(output: Any) -> dict[str, Any]:
    """The group's training data and accounting, as the trainer receives them."""

    stats = _RecordingTracker()
    output.group_stats.record(stats)
    return {
        "datums": [datum_record(datum) for datum in output],
        "requested_rollouts": output.requested_rollouts,
        "observed_rollouts": output.observed_rollouts,
        "trainable_rollouts": output.trainable_rollouts,
        "task_retained_datums": output.task_retained_datums,
        "workload": vars(output.workload).copy(),
        "stats": stats.calls,
    }


# ----------------------------------------------------------------------------------- the scenario's own checks


def scenario_problems(run: GroupRun) -> list[str]:
    """What in ``run`` the scenario did not do: wrong rewards, a missing or extra subagent, a group that did not train.

    Parity alone would pass two variants that fail alike, so every run is checked against its scenario first.
    """

    problems: list[str] = []
    scenario = run.scenario
    for member, root_reward in zip(run.members, scenario.root_rewards, strict=True):
        where = f"{run.variant} member {member.index}"
        if member.collection is None:
            problems.append(f"{where}: the rollout returned no trajectories")
            continue
        trajectories = list(member.collection["trajectories"].values())
        root, children = trajectories[0], trajectories[1:]
        if root["reward"] != root_reward or root["misc"]["outcome"]["status"] != "scored":
            problems.append(f"{where}: root scored {root['misc']['outcome']} instead of reward {root_reward}")
        child_reward = None if scenario.child_rewards is None else scenario.child_rewards[member.index]
        if child_reward is None:
            if children:
                problems.append(f"{where}: {len(children)} unexpected subagent trajectories")
        elif len(children) != 1:
            problems.append(f"{where}: {len(children)} subagent trajectories instead of 1")
        else:
            (child,) = children
            if child["reward"] != child_reward or child["parent_info"]["id"] != root["id"]:
                problems.append(f"{where}: subagent reward {child['reward']} instead of {child_reward}")
            parent_goals = [task["goal"] for task in child["task"]["parent_tasks"]]
            if parent_goals != [root["task"]["goal"]]:
                problems.append(f"{where}: the subagent's parent tasks have goals {parent_goals}")
    output = run.output
    if output["trainable_rollouts"] != len(scenario.plans) or not output["datums"]:
        problems.append(f"{run.variant}: the group trained {output['trainable_rollouts']} rollouts")
    return problems
