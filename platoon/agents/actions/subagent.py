import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

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
from platoon.episode.trajectory import BudgetExceededError, Trajectory
from platoon.utils.span_profile import profile_span

SUBAGENT_REWARD_JUDGMENT_MISC_KEY = "subagent_reward_judgment"
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


@dataclass(frozen=True)
class SubagentRewardJudgeConfig:
    max_steps: int = 20


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
    normalized = dict(parsed)
    normalized["status"] = status
    normalized["score"] = _coerce_score(parsed.get("score"), status=status)
    normalized["raw_response"] = raw_message
    normalized["verifier_trajectory_id"] = verifier_traj.id
    normalized["verifier_error_message"] = verifier_traj.error_message
    # A syntactically valid object is not enough to create a trustworthy
    # policy target: the verifier must have emitted a finish result and
    # returned one of the statuses requested by the judgment schema. A valid ``failed`` or
    # ``insufficient_evidence`` judgment remains a legitimate zero-reward
    # target; only missing/malformed/non-finished judgments are suppressed.
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = bool(
        verifier_traj.finish_message and status in _VALID_JUDGMENT_STATUSES
    )
    return normalized


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

    subtask = task.fork(goal, max_steps, task_misc=task_misc)
    tracker = budget_tracker.get()
    reserved_budget = max_steps + 1
    forked_agent: ForkableAgent | None = None
    forked_env: ForkableEnv | None = None
    episode_ownership_started = asyncio.Event()

    try:
        tracker.reserve_budget(reserved_budget, raise_on_failure=True)
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
    verifier_result = await _run_subagent_trajectory(
        goal=_format_verifier_goal(child_goal=goal, child_traj=traj),
        max_steps=config.max_steps,
        task_misc=verifier_misc,
        verbose=True,
        parent_traj=traj,
    )

    if isinstance(verifier_result, str):
        judgment = {
            "status": "judge_error",
            "score": 0.0,
            "summary": verifier_result,
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
        }
        traj.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
        _record_judgment_reward(traj, judgment)
        _emit_trajectory_finished_update(traj)
        return

    verifier_result.misc[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] = True
    verifier_result.misc[SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] = traj.id
    verifier_result.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True
    _emit_trajectory_finished_update(verifier_result)
    raw_message = _subagent_return_message(verifier_result)
    judgment = _normalize_judgment(
        raw_message,
        verifier_result,
    )
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
