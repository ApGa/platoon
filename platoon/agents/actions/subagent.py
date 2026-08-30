import asyncio
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, cast

from platoon.agents.base import ForkableAgent
from platoon.envs.base import ForkableEnv
from platoon.episode.context import (
    budget_tracker,
    current_agent,
    current_env,
    current_trajectory,
    current_trajectory_collection,
    episode_step_timeout,
    subagent_reward_judge_config,
)
from platoon.episode.loop import _close_episode_resource, run_episode
from platoon.episode.trajectory import (
    BudgetExceededError,
    SubagentDepthScope,
    Trajectory,
)
from platoon.utils.span_profile import profile_span

SUBAGENT_REWARD_JUDGMENT_MISC_KEY = "subagent_reward_judgment"
SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY = "subagent_outcome_judgment"
SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY = "subagent_behavior_judgment"
SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY = "subagent_reward_verifier_task"
SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY = "subagent_reward_verifies_trajectory_id"
EXCLUDE_FROM_TRAINING_MISC_KEY = "exclude_from_training"
# Unlike ``exclude_from_training`` (used for synthetic verifier trajectories),
# this marker suppresses only the trajectory's policy datums.  Reward and
# rollout-stat processing must still see the trajectory so a failed verifier
# cannot silently change group baselines or recursive delegation accounting.
EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY = "exclude_from_policy_training"
SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY = "training_eligible"

_VALID_JUDGMENT_STATUSES = frozenset({"verified", "partial", "failed", "insufficient_evidence"})
_VALID_BEHAVIOR_JUDGMENT_STATUSES = frozenset({"pass", "fail", "insufficient_evidence"})


class SubagentBehaviorJudge(Protocol):
    """Judge whether a subagent's own behavior deserves its outcome reward.

    Implementations return a dictionary with the strict schema
    ``{"status": "pass" | "fail" | "insufficient_evidence",
    "passed": bool | None, "reason": str, ...}``.  A pass must use
    ``passed=true``, a fail must use ``passed=false``, and insufficient
    evidence must use ``passed=null``.  Extra diagnostic fields such as
    ``evidence`` and ``violations`` are preserved.
    """

    async def judge(self, *, goal: str, trajectory: Trajectory) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SubagentRewardJudgeConfig:
    max_steps: int = 20
    behavior_judge: SubagentBehaviorJudge | None = None


def _subagent_return_message(traj: Trajectory) -> str:
    if traj.finish_message:
        return traj.finish_message
    if traj.error_message:
        return _subagent_error_message(traj.error_message)
    return traj.finish_message or ""


def _subagent_error_message(error: str) -> str:
    first_line = next((line.strip() for line in error.splitlines() if line.strip()), "")
    if first_line.startswith("WARNING: Exhausted budget"):
        return "Subagent did not finish before its step budget was exhausted."
    return "Subagent failed before finishing."


def _clip(value: str | None, *, limit: int = 6000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def _trajectory_task_misc(traj: Trajectory) -> dict[str, Any]:
    if traj.task is None:
        return {}
    return dict(traj.task.misc)


def _trajectory_task_goal(traj: Trajectory) -> str:
    if traj.task is None:
        return ""
    return traj.task.goal or ""


def _is_verifier_trajectory(traj: Trajectory) -> bool:
    return bool(_trajectory_task_misc(traj).get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY))


def _propagate_verifier_task_misc(
    parent_task_misc: dict[str, Any],
    child_task_misc: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep every descendant of a reward verifier in the verifier tree.

    ``Task.fork`` normally inherits all parent task metadata when callers do not
    supply ``task_misc``.  A launch path that does supply metadata, however,
    replaces it wholesale.  The verifier marker is an ancestry invariant, not
    caller-controlled child metadata: losing it would make the child eligible
    for policy training and would schedule another reward verifier for it.
    The direct verifier's ``verifies_trajectory_id`` is intentionally not
    copied: descendants are nested under the verifier through ``parent_info``
    and must not masquerade as direct judges of the solver trajectory.
    """

    if not parent_task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
        return child_task_misc

    propagated = dict(
        parent_task_misc if child_task_misc is None else child_task_misc
    )
    propagated[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] = True
    propagated.pop(SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY, None)
    return propagated


def _subagent_depth_scope(
    *,
    parent_task_misc: dict[str, Any],
    child_task_misc: dict[str, Any],
    synthetic_verifier_parent: Trajectory | None,
) -> SubagentDepthScope:
    if not child_task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY):
        return "policy"
    if (
        synthetic_verifier_parent is not None
        and child_task_misc.get(
            SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY
        )
        == synthetic_verifier_parent.id
    ):
        return "verifier_root"
    if (
        parent_task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY)
        and parent_task_misc.get(
            SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY
        )
    ):
        return "verifier_helper"
    return "verifier_descendant"


def _format_verifier_goal(*, child_goal: str, child_traj: Trajectory) -> str:
    child_message = _subagent_return_message(child_traj)
    child_task_goal = _trajectory_task_goal(child_traj) or child_goal
    return (
        "You are a verifier agent judging a completed child-agent task.\n\n"
        "Do not trust the child agent's summary. Use available tools to inspect "
        "the environment, files, and other externally visible state before giving "
        "a verdict. Avoid mutating state unless a read-only inspection path is "
        "not available.\n\n"
        f"Judged Child Trajectory ID:\n{child_traj.id}\n\n"
        f"Judged Child Goal:\n{_clip(child_task_goal)}\n\n"
        f"Child Final Message:\n{_clip(child_message)}\n\n"
        f"Child Error Message:\n{_clip(child_traj.error_message)}\n\n"
        "Return only a JSON object via `finish` with this schema:\n"
        "{\n"
        '  "status": "one of: verified, partial, failed, insufficient_evidence",\n'
        '  "score": 0.0,\n'
        '  "summary": "short verdict",\n'
        '  "passed_claims": ["claim that was verified"],\n'
        '  "failed_claims": ["claim that failed verification"],\n'
        '  "evidence": ["tool-backed evidence you inspected"]\n'
        "}\n\n"
        "Use score 1.0 only when the delegated goal is fully verified, 0.0 when "
        "it failed, and an intermediate score for partial completion."
    )


def _parse_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_score(value: Any, *, status: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = {
            "verified": 1.0,
            "partial": 0.5,
            "failed": 0.0,
            "insufficient_evidence": 0.0,
        }.get(status, 0.0)
    return min(1.0, max(0.0, score))


def _normalize_judgment(raw_message: str, verifier_traj: Trajectory) -> dict[str, Any]:
    parsed = _parse_json_object(raw_message)
    if parsed is None:
        status = "unparseable"
        return {
            "status": status,
            "score": 0.0,
            "summary": _clip(raw_message, limit=2000),
            "raw_response": raw_message,
            "verifier_trajectory_id": verifier_traj.id,
            "verifier_error_message": verifier_traj.error_message,
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
        }

    status = str(parsed.get("status") or "unknown").strip().lower()
    raw_score = parsed.get("score")
    if raw_score is None:
        score = _coerce_score(None, status=status)
        score_is_numeric = True
    else:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
            score_is_numeric = False
        else:
            score_is_numeric = bool(
                not isinstance(raw_score, bool)
                and math.isfinite(score)
                and 0.0 <= score <= 1.0
            )
            if not score_is_numeric:
                score = 0.0

    score_is_consistent = bool(
        score_is_numeric
        and (
            (status == "verified" and score > 0.0)
            or (status == "partial" and 0.0 < score < 1.0)
            or (status in {"failed", "insufficient_evidence"} and score == 0.0)
        )
    )
    normalized = dict(parsed)
    normalized["status"] = status
    normalized["score"] = score if score_is_consistent else 0.0
    normalized["raw_response"] = raw_message
    normalized["verifier_trajectory_id"] = verifier_traj.id
    normalized["verifier_error_message"] = verifier_traj.error_message
    # A syntactically valid object is not enough to create a trustworthy
    # policy target: the verifier must have emitted a finish result and
    # returned one of the statuses requested by the judgment schema. A valid ``failed`` or
    # ``insufficient_evidence`` judgment remains a legitimate zero-reward
    # target; only missing/malformed/non-finished judgments are suppressed.
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = bool(
        verifier_traj.finish_message
        and status in _VALID_JUDGMENT_STATUSES
        and score_is_consistent
    )
    if status in _VALID_JUDGMENT_STATUSES and not score_is_consistent:
        normalized["schema_error"] = (
            "Outcome verifier status and score are inconsistent."
        )
    return normalized


def _unparseable_behavior_judgment(
    raw_judgment: Any,
    *,
    reason: str,
) -> dict[str, Any]:
    """Return a fail-closed result for a malformed behavior-judge response."""

    return {
        "status": "unparseable",
        "passed": None,
        "gate": 0.0,
        "reason": reason,
        "raw_response": raw_judgment,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


def _normalize_behavior_judgment(raw_judgment: Any) -> dict[str, Any]:
    """Validate a behavior verdict and derive its strict binary reward gate."""

    if not isinstance(raw_judgment, dict):
        return _unparseable_behavior_judgment(
            raw_judgment,
            reason="Behavior judge did not return a JSON object.",
        )

    raw_status = raw_judgment.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    has_passed = "passed" in raw_judgment
    passed = raw_judgment.get("passed")
    reason = raw_judgment.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return _unparseable_behavior_judgment(
            raw_judgment,
            reason="Behavior judgment is missing a non-empty string `reason`.",
        )

    expected_passed = {
        "pass": True,
        "fail": False,
        "insufficient_evidence": None,
    }.get(status, object())
    if status not in _VALID_BEHAVIOR_JUDGMENT_STATUSES or not has_passed or passed is not expected_passed:
        return _unparseable_behavior_judgment(
            raw_judgment,
            reason=("Behavior judgment must pair status/pass, fail/false, or insufficient_evidence/null exactly."),
        )

    normalized = dict(raw_judgment)
    normalized["status"] = status
    normalized["passed"] = passed
    normalized["reason"] = reason.strip()
    normalized["gate"] = 1.0 if passed is True else 0.0
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = status in {
        "pass",
        "fail",
    }
    return normalized


def _behavior_judge_error(error: Exception) -> dict[str, Any]:
    return {
        "status": "judge_error",
        "passed": None,
        "gate": 0.0,
        "reason": f"Behavior judge raised {type(error).__name__}: {error}",
        "error_type": type(error).__name__,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


def _combine_outcome_and_behavior_judgments(
    outcome_judgment: dict[str, Any],
    behavior_judgment: dict[str, Any],
) -> dict[str, Any]:
    """Gate the environment verifier score with the behavioral verdict."""

    outcome_status = str(outcome_judgment.get("status") or "")
    outcome_score = _coerce_score(
        outcome_judgment.get("score"),
        status=outcome_status,
    )
    behavior_gate = 1.0 if behavior_judgment.get("gate") == 1.0 else 0.0
    behavior_status = str(behavior_judgment.get("status") or "")
    combined = dict(outcome_judgment)
    combined.update(
        {
            "score": outcome_score * behavior_gate,
            "outcome_status": outcome_status,
            "outcome_score": outcome_score,
            "behavior_gate": behavior_gate,
            "outcome_judgment": dict(outcome_judgment),
            "behavior_judgment": dict(behavior_judgment),
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: bool(
                outcome_judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)
                and behavior_judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)
            ),
        }
    )
    if behavior_status == "fail":
        combined["status"] = "behavior_rejected"
    elif behavior_status != "pass":
        combined["status"] = "behavior_judge_invalid"
    return combined


async def _run_behavior_judge(
    judge: SubagentBehaviorJudge,
    *,
    goal: str,
    trajectory: Trajectory,
) -> dict[str, Any]:
    try:
        raw_judgment = await judge.judge(goal=goal, trajectory=trajectory)
    except Exception as error:
        return _behavior_judge_error(error)
    return _normalize_behavior_judgment(raw_judgment)


def _behavior_judgment_not_run(*, outcome_eligible: bool) -> dict[str, Any]:
    if outcome_eligible:
        return {
            "status": "not_run_zero_outcome",
            "passed": None,
            "gate": 1.0,
            "reason": "Behavior judge was skipped because the outcome score was zero.",
            "judged": False,
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: True,
        }
    return {
        "status": "not_run_ineligible_outcome",
        "passed": None,
        "gate": None,
        "reason": "Behavior judge was skipped because the outcome verdict was ineligible.",
        "judged": False,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


def _record_judgment_reward(traj: Trajectory, judgment: dict[str, Any]) -> None:
    score = _coerce_score(judgment.get("score"), status=str(judgment.get("status") or ""))
    traj.reward = score
    if bool(judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)):
        traj.misc.pop(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY, None)
    else:
        traj.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] = True
    if traj.steps:
        step = traj.steps[-1]
        if isinstance(step, dict):
            reward_misc = step.setdefault("misc", {}).setdefault("reward_misc", {})
        else:
            reward_misc = step.misc.setdefault("reward_misc", {})
        reward_misc["reward/success"] = score
        reward_misc["reward/subagent_judgment"] = score
        outcome_judgment = traj.misc.get(SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY)
        if isinstance(outcome_judgment, dict):
            reward_misc["reward/subagent_outcome_judgment"] = _coerce_score(
                outcome_judgment.get("score"),
                status=str(outcome_judgment.get("status") or ""),
            )
        elif "outcome_score" in judgment:
            reward_misc["reward/subagent_outcome_judgment"] = _coerce_score(
                judgment.get("outcome_score"),
                status=str(judgment.get("outcome_status") or ""),
            )
        if "behavior_gate" in judgment:
            reward_misc["reward/subagent_behavior_gate"] = 1.0 if judgment.get("behavior_gate") == 1.0 else 0.0


def _emit_trajectory_finished_update(traj: Trajectory) -> None:
    traj_collection = current_trajectory_collection.get(None)
    if traj_collection is None or traj.id not in traj_collection.trajectories:
        return
    traj_collection.finish_trajectory(traj.id)


async def _run_owned_subagent_episode(
    agent: ForkableAgent,
    env: ForkableEnv,
    *,
    timeout: int,
    ownership_started: asyncio.Event,
) -> Trajectory:
    """Hand fork ownership to ``run_episode`` before its first suspension."""

    ownership_started.set()
    return await run_episode(agent, env, timeout=timeout)


async def _run_subagent_trajectory(
    *,
    goal: str,
    max_steps: int,
    task_misc: dict | None,
    verbose: bool,
    parent_traj: Trajectory | None = None,
) -> Trajectory | str:
    # Cast is safe here: launch_subagent only works in contexts with forkable agents/envs
    agent = cast(ForkableAgent, current_agent.get())
    env = cast(ForkableEnv, current_env.get())
    task = parent_traj.task if parent_traj is not None and parent_traj.task is not None else env.task
    task_misc = _propagate_verifier_task_misc(dict(task.misc), task_misc)

    subtask = task.fork(goal, max_steps, task_misc=task_misc)
    child_depth_scope = _subagent_depth_scope(
        parent_task_misc=dict(task.misc),
        child_task_misc=dict(subtask.misc),
        synthetic_verifier_parent=parent_traj,
    )
    tracker = budget_tracker.get()
    reserved_budget = max_steps + 1
    forked_agent: ForkableAgent | None = None
    forked_env: ForkableEnv | None = None
    episode_ownership_started = asyncio.Event()

    try:
        tracker.reserve_budget(
            reserved_budget,
            raise_on_failure=True,
            child_depth_scope=child_depth_scope,
        )
    except (BudgetExceededError, ValueError) as e:
        guidance = getattr(e, "guidance", "")
        msg = f"Not enough budget to launch subagent for goal {goal}. {e}"
        if guidance:
            msg += " " + guidance
        return msg

    try:
        forked_agent = await agent.fork(subtask)
        forked_env = await env.fork(subtask)
        _ = verbose
        parent_token = current_trajectory.set(parent_traj) if parent_traj is not None else None
        try:
            return await asyncio.create_task(
                _run_owned_subagent_episode(
                    forked_agent,
                    forked_env,
                    timeout=episode_step_timeout.get(),
                    ownership_started=episode_ownership_started,
                )
            )
        finally:
            if parent_token is not None:
                current_trajectory.reset(parent_token)
    finally:
        try:
            # Release synchronously, before cleanup awaits can be cancelled or
            # mutate the trajectory context used by StepBudgetTracker.
            tracker.release_budget(reserved_budget)
        finally:
            # Once the child task starts, run_episode is the sole owner and
            # closes both resources. Before that handoff, close only handles
            # that were successfully returned by their fork methods.
            if not episode_ownership_started.is_set():
                if forked_agent is not None:
                    await _close_episode_resource(forked_agent, "forked agent")
                if forked_env is not None:
                    await _close_episode_resource(forked_env, "forked environment")


async def _maybe_judge_subagent(*, goal: str, traj: Trajectory) -> None:
    config = subagent_reward_judge_config.get(None)
    if not isinstance(config, SubagentRewardJudgeConfig) or config.max_steps <= 0:
        return
    if _is_verifier_trajectory(traj):
        return

    # Fail closed while verification is in flight.  This update is persisted
    # before launching the verifier, so a rollout/process cancellation cannot
    # leave the already-completed child looking like a valid policy target.
    # A successfully parsed verdict clears the marker in
    # ``_record_judgment_reward`` below.
    pending_judgment = {
        "status": "pending",
        "score": 0.0,
        "summary": "Subagent reward verification is still in progress.",
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }
    traj.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = pending_judgment
    # Persist a fail-closed score as well as the policy marker.  If the rollout
    # is cooperatively returned while the verifier is in flight, delegation
    # accounting must not read the child's stale, unverified success reward.
    _record_judgment_reward(traj, pending_judgment)
    _emit_trajectory_finished_update(traj)

    verifier_misc = {
        **_trajectory_task_misc(traj),
        SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY: True,
        SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY: traj.id,
    }
    verifier_args = {
        "goal": _format_verifier_goal(child_goal=goal, child_traj=traj),
        "max_steps": config.max_steps,
        "task_misc": verifier_misc,
        "verbose": True,
        "parent_traj": traj,
    }
    behavior_judge = config.behavior_judge
    # Outcome verification is deliberately first: the behavior judge
    # is useful only for positive, trainable outcomes and can otherwise add
    # substantial latency and API load without changing the final score.
    verifier_result = await _run_subagent_trajectory(**verifier_args)

    if isinstance(verifier_result, str):
        outcome_judgment = {
            "status": "judge_error",
            "score": 0.0,
            "summary": verifier_result,
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
        }
    else:
        verifier_result.misc[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] = True
        verifier_result.misc[SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] = traj.id
        verifier_result.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
        _emit_trajectory_finished_update(verifier_result)
        raw_message = _subagent_return_message(verifier_result)
        outcome_judgment = _normalize_judgment(
            raw_message,
            verifier_result,
        )

    if behavior_judge is None:
        judgment = outcome_judgment
    else:
        # Keep the source verdicts independently queryable.  The historical
        # key remains the effective reward verdict consumed by trainers.
        traj.misc[SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY] = outcome_judgment
        outcome_eligible = bool(
            outcome_judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)
        )
        outcome_score = _coerce_score(
            outcome_judgment.get("score"),
            status=str(outcome_judgment.get("status") or ""),
        )
        if outcome_eligible and outcome_score > 0.0:
            # Own the task explicitly so cancellation of the rollout also
            # cancels and drains an in-flight policy-judge request.
            behavior_task = asyncio.create_task(
                _run_behavior_judge(behavior_judge, goal=goal, trajectory=traj)
            )
            try:
                behavior_judgment = await behavior_task
            finally:
                if not behavior_task.done():
                    behavior_task.cancel()
                await asyncio.gather(behavior_task, return_exceptions=True)
            traj.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY] = behavior_judgment
            judgment = _combine_outcome_and_behavior_judgments(
                outcome_judgment,
                behavior_judgment,
            )
        else:
            # A valid zero outcome must remain a trainable zero.  This
            # synthetic diagnostic is not folded into the effective verdict,
            # and therefore does not emit a misleading behavior-gate metric.
            traj.misc[SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY] = _behavior_judgment_not_run(
                outcome_eligible=outcome_eligible
            )
            judgment = outcome_judgment
    traj.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    _record_judgment_reward(traj, judgment)
    _emit_trajectory_finished_update(traj)


async def launch_subagent(goal: str, max_steps: int = 15, task_misc: dict | None = None, verbose: bool = True) -> Any:
    """Launch a subagent to solve a task.

    Args:
        goal: The goal of the subagent.
        max_steps: The maximum number of steps the subagent can take.

    Returns:
        Returns the result of the subagent's execution.
    """
    parent_traj = current_trajectory.get()
    async with profile_span(
        "launch_subagent",
        metadata={
            "goal_len": len(goal),
            "max_steps": max_steps,
            "parent_task_id": getattr(current_env.get().task, "id", None),
            "parent_trajectory_id": parent_traj.id,
        },
    ):
        result = await _run_subagent_trajectory(
            goal=goal,
            max_steps=max_steps,
            task_misc=task_misc,
            verbose=verbose,
        )
        if isinstance(result, str):
            return result
        traj = result
        await _maybe_judge_subagent(goal=goal, traj=traj)
        _ = verbose
        return _subagent_return_message(traj)
