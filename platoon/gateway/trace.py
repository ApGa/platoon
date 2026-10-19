"""Agent call records -> the trajectories the Tinker workflows train on.

A rollout served through the gateway needs no harness-side trajectory: the agent tree is the trace.
:func:`build_trajectory_collection` turns it into the existing ``TrajectoryCollection``: one trajectory per
agent, one step per model call (``misc.action_misc.completion_id = call_id``), parent links from the agent tree
and rewards from the outcome.  ``platoon.utils.tinker_data_processing`` then merges calls whose prompt extends
the previous prompt and completion into one datum, and starts a new datum when that extension property
fails, exactly as for in-process harnesses.  Each trajectory's ``misc`` also carries what the in-process
provider records on its trajectory: per-call generation diagnostics and ``time.perf_counter`` spans.

Per agent:

* **Outcome.**  The root carries the episode outcome.  A child with its own outcome (``AgentSession.outcome``:
  its env grade or its verifier's verdict) takes its reward from it, and the untrainable marker of an
  ``invalid``, ``error``, ``timeout`` or ``cancelled`` outcome.  A child without one shares the episode outcome,
  the default for gateway children (and the original recursive rollouts' default once ``propagate_root_success``
  ran); it is also marked untrainable when its own run timed out or was cancelled.  A scored outcome's reward
  is written into the last step's ``reward_misc`` as ``reward/success``, where the legacy envs put it.  The
  trajectory of an agent with its own outcome records it in ``misc["outcome"]``: status, rewards and reason, the
  infrastructure ``failure_kind`` of an untrainable one (``harness_crash``, ``env_busy``, ...), the grader's own
  reason (``grade_reason``) when the rollout reports another, such as ``context_overflow``, and the ``reward_key`` a
  scored one trained: the name its grader reported the reward under, null for a score platoon gave itself.
* **Annotations.**  ``AgentSession.misc`` (``ctx.annotate``) is merged into the trajectory misc, and
  ``AgentSession.step_misc`` (``ctx.annotate_step``) into each call's step misc (``reward_misc`` merged by key).
  A step annotated ``exclude_from_training=True`` stays in the trajectory for viewers, but the datum builder
  (``platoon.utils.tinker_data_processing``) takes no tokens from its call, e.g. a rejected condensation.
* **Call kinds.**  ``policy`` and ``aux`` calls are steps (an ``aux`` step's ``action_misc`` says ``kind:
  aux``); ``judge`` calls are not trained and go to the trajectory misc under ``judge_calls`` for audit.  A hosted
  model's call (:mod:`platoon.gateway.hosted`) adds its provider record to its ``action_misc`` and holds no tokens,
  so the datum builder trains nothing from it.  Agents with role ``judge`` (and their descendants) get no
  trajectory: their calls are kept under ``judge_calls`` of the nearest ancestor that has one.
* **Tool calls** recorded on the agent (``AgentSession.tool_calls``, the gateway's tool plane) go to the misc
  under ``tool_calls``.
* **Judges.**  The root trajectory of a rollout whose judges are off (``AgentSession.judges``) records it under
  ``judges`` (:mod:`platoon.utils.judges`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from platoon.envs.service import GRADE_REASON_KEY, PLATOON_ZERO_KEY
from platoon.gateway.core import AgentSession, ChatResponse
from platoon.task import Task
from platoon.trajectory import ParentInfo, Trajectory, TrajectoryCollection, TrajectoryStep
from platoon.utils.critical_path import MODEL_CALL_SPANS_KEY
from platoon.utils.generation import MODEL_GENERATIONS_KEY
from platoon.utils.judges import JUDGE_ROLE, JUDGES_MISC_KEY
from platoon.utils.trajectory_status import (
    OUTCOME_MISC_KEY,
    OUTCOME_REWARD_KEY,
    TRAJECTORY_CANCELLED_MISC_KEY,
    TRAJECTORY_INVALID_MISC_KEY,
    TRAJECTORY_TIMED_OUT_MISC_KEY,
)

AGENT_MISC_KEY = "agent"
AGENT_RESULT_MISC_KEY = "agent_result"
JUDGE_CALLS_MISC_KEY = "judge_calls"
TOOL_CALLS_MISC_KEY = "tool_calls"
JUDGE_KIND = "judge"

# The trajectory marker that keeps each untrainable outcome out of policy training and out of the group
# baseline (``platoon.utils.trajectory_status.trajectory_was_interrupted``).  Infrastructure failures are
# never trained as reward 0.
_UNTRAINABLE_MARKERS = {
    "invalid": TRAJECTORY_INVALID_MISC_KEY,
    "error": TRAJECTORY_INVALID_MISC_KEY,
    "timeout": TRAJECTORY_TIMED_OUT_MISC_KEY,
    "cancelled": TRAJECTORY_CANCELLED_MISC_KEY,
}


class Outcome(Protocol):
    """What the trace reads of an environment outcome (``platoon.envs.service.Outcome``)."""

    @property
    def status(self) -> str: ...

    @property
    def rewards(self) -> Mapping[str, float]: ...

    @property
    def reason(self) -> str | None: ...

    @property
    def reward(self) -> float | None: ...


def build_trajectory_collection(
    agents: Iterable[AgentSession], *, task: Task, outcome: Outcome
) -> TrajectoryCollection:
    """One rollout's agent tree -> its ``TrajectoryCollection`` (the root trajectory first).

    ``outcome`` is the episode's: the root's, and that of every child without its own (see the module
    docstring).  Children without a recorded task get ``task`` forked with their goal.
    """
    ordered = _root_first(agents)
    episode = _OutcomeFields.of(outcome)
    excluded = _judge_subtrees(ordered)
    by_id = {agent.agent_id: agent for agent in ordered}
    trajectories: dict[str, Trajectory] = {}
    audit: dict[str, list[dict[str, Any]]] = {}
    for agent in ordered:
        if agent.agent_id in excluded:
            owner = _nearest_kept_ancestor(agent, by_id, excluded)
            audit.setdefault(owner, []).extend(_judge_call(agent, call) for call in agent.calls)
    for agent in ordered:
        if agent.agent_id in excluded:
            continue
        is_root = agent.parent_agent_id is None
        own = None if is_root or agent.outcome is None else _OutcomeFields.of(agent.outcome)
        fields = episode if own is None else own
        if is_root:
            agent_task, parent_info = task, None
        else:
            agent_task = agent.task if agent.task is not None else task.fork(goal=agent.goal or "")
            parent_info = ParentInfo(id=by_id[agent.parent_agent_id].trajectory_id, fork_step=agent.fork_step)
        trajectory = build_agent_trajectory(
            agent, task=agent_task, parent_info=parent_info, fields=fields, own_outcome=is_root or own is not None
        )
        judge_calls = audit.get(agent.agent_id, [])
        if judge_calls:
            trajectory.misc.setdefault(JUDGE_CALLS_MISC_KEY, []).extend(judge_calls)
        trajectories[agent.trajectory_id] = trajectory
    return TrajectoryCollection(id=ordered[0].rollout_id, trajectories=trajectories)


def attach_judge_audit(collection: TrajectoryCollection, agents: Iterable[AgentSession]) -> None:
    """Refresh each trajectory's ``judge_calls`` from the agents, e.g. after finalizers called a judge."""

    ordered = list(agents)
    by_id = {agent.agent_id: agent for agent in ordered}
    excluded = _judge_subtrees(ordered)
    audit: dict[str, list[dict[str, Any]]] = {}
    for agent in ordered:
        owner = _nearest_kept_ancestor(agent, by_id, excluded) if agent.agent_id in excluded else agent.agent_id
        calls = [
            _judge_call(agent, call)
            for call in agent.calls
            if agent.agent_id in excluded or agent.call_kind(call.call_id) == JUDGE_KIND
        ]
        if calls:
            audit.setdefault(owner, []).extend(calls)
    for agent_id, calls in audit.items():
        trajectory = collection.trajectories.get(by_id[agent_id].trajectory_id)
        if trajectory is not None:
            trajectory.misc[JUDGE_CALLS_MISC_KEY] = calls


def build_agent_trajectory(
    agent: AgentSession,
    *,
    task: Task | None,
    parent_info: ParentInfo | None = None,
    fields: _OutcomeFields | None = None,
    own_outcome: bool = False,
) -> Trajectory:
    """One agent's trajectory; without ``fields`` (e.g. for a judge looking at a child mid-episode) it carries
    the agent's own outcome when it has one, else reward 0 and no marker."""

    if fields is None:
        own_outcome = agent.outcome is not None
        fields = _OutcomeFields.of(agent.outcome) if own_outcome else _OutcomeFields(0.0, None, None, None)
    misc: dict[str, Any] = dict(agent.misc)
    misc.update(
        {
            AGENT_MISC_KEY: {"agent_id": agent.agent_id, "role": agent.role},
            MODEL_GENERATIONS_KEY: {
                call.call_id: {**call.generation, "finish_reason": call.finish_reason} for call in agent.calls
            },
            MODEL_CALL_SPANS_KEY: {
                call.call_id: [call.timing.started_at, call.timing.ended_at] for call in agent.calls if call.timing
            },
        }
    )
    error_message = fields.error_message
    if fields.marker is not None:
        misc[fields.marker] = True
    result = agent.result
    if result is not None:
        misc[AGENT_RESULT_MISC_KEY] = {"status": result.status, "error": result.error}
        if not own_outcome and result.status in _RUN_STATUS_MARKERS:
            misc[_RUN_STATUS_MARKERS[result.status]] = True
            error_message = error_message or result.error or f"Agent {result.status}"
    if own_outcome and fields.summary is not None:
        misc[OUTCOME_MISC_KEY] = fields.summary
    if agent.parent_agent_id is None and not getattr(agent, "judges", True):
        misc[JUDGES_MISC_KEY] = {"enabled": False}
    tool_calls = getattr(agent, "tool_calls", None)
    if tool_calls:
        misc[TOOL_CALLS_MISC_KEY] = [*misc.get(TOOL_CALLS_MISC_KEY, []), *(dict(record) for record in tool_calls)]
    steps: list[TrajectoryStep] = []
    judge_calls: list[dict[str, Any]] = []
    for call in agent.calls:
        if agent.call_kind(call.call_id) == JUDGE_KIND:
            judge_calls.append(_judge_call(agent, call))
        else:
            steps.append(_step(call, agent.step_misc.get(call.call_id), kind=agent.call_kind(call.call_id)))
    if judge_calls:
        misc.setdefault(JUDGE_CALLS_MISC_KEY, []).extend(judge_calls)
    if steps and fields.success is not None:
        steps[-1].misc.setdefault("reward_misc", {})["reward/success"] = fields.success
    return Trajectory(
        id=agent.trajectory_id,
        task=task,
        parent_info=parent_info,
        steps=steps,
        reward=fields.reward,
        finish_message=result.final_message if result is not None else None,
        error_message=error_message if error_message is not None else _run_error_message(result),
        misc=misc,
    )


def _run_error_message(result: Any) -> str | None:
    """The legacy ``error_message`` of a harness run (``platoon.harness.agent.AgentResult``)."""
    if result is None:
        return None
    if result.status == "limit" and not result.error:
        return "WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete."
    return result.error


# A child that shares the episode outcome is still untrainable when its own run was interrupted.
_RUN_STATUS_MARKERS = {"timeout": TRAJECTORY_TIMED_OUT_MISC_KEY, "cancelled": TRAJECTORY_CANCELLED_MISC_KEY}


class _OutcomeFields:
    """What a trajectory takes from an outcome: reward, success, untrainable marker and error text."""

    def __init__(
        self, reward: float, success: float | None, marker: str | None, error_message: str | None, summary=None
    ) -> None:
        self.reward = reward
        self.success = success
        self.marker = marker
        self.error_message = error_message
        self.summary: dict[str, Any] | None = summary

    @classmethod
    def of(cls, outcome: Outcome) -> _OutcomeFields:
        summary = {"status": outcome.status, "rewards": dict(outcome.rewards), "reason": outcome.reason}
        failure_kind = getattr(outcome, "failure_kind", None)
        if failure_kind is not None:
            summary["failure_kind"] = failure_kind  # the infrastructure taxonomy, for reports to count by
        metadata = getattr(outcome, "metadata", None) or {}
        grade_reason = metadata.get(GRADE_REASON_KEY)
        if grade_reason is not None:
            summary[GRADE_REASON_KEY] = grade_reason  # the grader's own reason, when the rollout reports another
        if outcome.status == "scored":
            if outcome.reward is None:
                raise ValueError("A scored outcome must carry a reward")
            reward = float(outcome.reward)
            # The reward key that trained: the name the grader reported the reward under, none for platoon's own zero.
            summary[OUTCOME_REWARD_KEY] = None if metadata.get(PLATOON_ZERO_KEY) else getattr(outcome, "primary", None)
            return cls(reward, reward, None, None, summary)
        if outcome.status in _UNTRAINABLE_MARKERS:
            error_message = f"Outcome {outcome.status}" + (f": {outcome.reason}" if outcome.reason else "")
            return cls(0.0, None, _UNTRAINABLE_MARKERS[outcome.status], error_message, summary)
        raise ValueError(f"Unknown outcome status {outcome.status!r}")


def _action_misc(call: ChatResponse, kind: str = "policy") -> dict[str, Any]:
    """A call's ``action_misc``: its id, model and usage; its kind when it is not ``policy``; and a hosted call's
    provider record (served model, ids, attempts, cost: ``ProviderRecord.action_misc``)."""

    misc: dict[str, Any] = {"completion_id": call.call_id, "model": call.model, "usage": call.usage()}
    if kind != "policy":
        misc["kind"] = kind
    if call.provider is not None:
        misc.update(call.provider.action_misc())
    return misc


def _step(call: ChatResponse, annotations: Mapping[str, Any] | None = None, *, kind: str = "policy") -> TrajectoryStep:
    misc: dict[str, Any] = {"action_misc": _action_misc(call, kind), "message": call.message()}
    for key, value in (annotations or {}).items():
        if key == "action_misc":
            continue  # the completion id links the step to its tokens; annotations cannot move it
        if key == "reward_misc" and isinstance(value, Mapping):
            misc.setdefault("reward_misc", {}).update(value)
        else:
            misc[key] = value
    return TrajectoryStep(misc=misc)


def _judge_call(agent: AgentSession, call: ChatResponse) -> dict[str, Any]:
    record = {
        "call_id": call.call_id,
        "agent_id": agent.agent_id,
        "role": agent.role,
        "message": call.message(),
        "finish_reason": call.finish_reason,
        "usage": call.usage(),
    }
    if call.provider is not None:
        record.update(call.provider.action_misc())
    return record


def _judge_subtrees(agents: list[AgentSession]) -> set[str]:
    """Agent ids of non-root ``judge``-role agents and their descendants."""
    by_id = {agent.agent_id: agent for agent in agents}
    excluded: set[str] = set()
    for agent in agents:
        current: AgentSession | None = agent
        while current is not None and current.parent_agent_id is not None:
            if current.role == JUDGE_ROLE:
                excluded.add(agent.agent_id)
                break
            current = by_id.get(current.parent_agent_id)
    return excluded


def _nearest_kept_ancestor(agent: AgentSession, by_id: Mapping[str, AgentSession], excluded: set[str]) -> str:
    current = agent
    while current.agent_id in excluded and current.parent_agent_id is not None:
        current = by_id[current.parent_agent_id]
    return current.agent_id


def _root_first(agents: Iterable[AgentSession]) -> list[AgentSession]:
    """The agents with the single root first, everything else in the given (opening) order."""
    agents = list(agents)
    roots = [agent for agent in agents if agent.parent_agent_id is None]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root agent, got {len(roots)}")
    agent_ids = {agent.agent_id for agent in agents}
    orphans = [agent.agent_id for agent in agents if agent.parent_agent_id not in agent_ids and agent is not roots[0]]
    if orphans:
        raise ValueError(f"Agents whose parent is not in the trace: {orphans}")
    return roots + [agent for agent in agents if agent is not roots[0]]
