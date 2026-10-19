"""Verifier children: grade a subagent by running a read-only verifier agent on what it did.

With ``agents.child_grading: verifier``, every completed policy child is judged before its parent sees the
result.  A *verifier* agent runs the same harness as the child, on a fork of the child's session with
``agents.verifier.access`` (default ``read_only``; ``no_terminal`` lists every tool but the terminal ones), with a
goal that asks for a JSON verdict (:func:`format_verifier_goal`).  The verdict is parsed
and normalized (:func:`normalize_judgment`): only a finished verifier that returns a requested status with a
consistent score yields a *training-eligible* judgment.  An optional behavior judge
(:class:`SubagentBehaviorJudge`) audits the child in parallel, and its verdict gates the score
(:func:`combine_outcome_and_behavior_judgments`).  After the episode, an optional parent-usage judge
(:class:`SubagentParentUsageJudge`) drops the policy data of successful children their parent did not use
(:func:`apply_subagent_parent_usage_gates`, a program finalizer).

The judged child's trajectory gets the misc keys the trainer and credit code read (ported unchanged from the
legacy ``launch_subagent``): the effective verdict in ``subagent_reward_judgment``, the source verdicts, the
reward components in its last step's ``reward_misc``, and ``exclude_from_policy_training`` when the verdict is
not training-eligible.  Its outcome is ``scored`` with the verdict's score.  Verifier trajectories carry
``exclude_from_training`` and never train.

The rollout's judges switch (``RolloutConfig.judges``, which ``platoon eval`` sets from ``inference.judges``) turns
all of this off, parent-usage judges included; ``RolloutConfig.enable_subagent_verifiers`` (read by
``harness_rollout``) and ``agents.verifier.enabled`` turn off verifiers alone.  A child that is not verified then
shares the episode's outcome and records why (``child_grading``, :mod:`platoon.utils.judges`).  The
training-exclusion keys live in :mod:`platoon.utils.trajectory_status`, next to the datum builder's other markers;
they are re-exported here.

A verifier is not counted against ``agents.max_depth``, which bounds the policy tree: it runs one level below the
child it judges, a child at ``max_depth`` included, and may launch one level of helpers, which may not launch any.
With verifier grading the tree so reaches ``max_depth + 2``; ``agents.verifier.max_steps`` and
``agents.child_timeout_s`` bound those levels (by design: :mod:`platoon.harness.budget`).
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import TYPE_CHECKING, Any, Protocol

from platoon.utils.judges import JUDGES_OFF, VERIFIER_ROLE, judges_enabled
from platoon.utils.trajectory_status import (
    EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY,
    EXCLUDE_FROM_TRAINING_MISC_KEY,
    SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY,
    TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY,
    trajectory_has_forced_zero_reward,
)

if TYPE_CHECKING:
    from platoon.envs.service import Outcome
    from platoon.gateway.core import AgentSession
    from platoon.harness.agent import AgentResult
    from platoon.trajectory import Trajectory, TrajectoryCollection

SUBAGENT_REWARD_JUDGMENT_MISC_KEY = "subagent_reward_judgment"
SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY = "subagent_outcome_judgment"
SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY = "subagent_behavior_judgment"
SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY = "subagent_parent_usage_judgment"
SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY = "subagent_reward_verifies_trajectory_id"
SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY = "training_eligible"

DEFAULT_VERIFIER_MAX_STEPS = 20

_VALID_JUDGMENT_STATUSES = frozenset({"verified", "partial", "failed", "insufficient_evidence"})
_VALID_BEHAVIOR_JUDGMENT_STATUSES = frozenset({"pass", "fail", "insufficient_evidence"})
_VALID_PARENT_USAGE_JUDGMENT_STATUSES = frozenset({"used", "not_used", "insufficient_evidence"})

BUDGET_EXHAUSTED_MESSAGE = "Subagent did not finish before its step budget was exhausted."
FAILED_MESSAGE = "Subagent failed before finishing."


class SubagentBehaviorJudge(Protocol):
    """Judge whether a subagent's own behavior deserves its outcome reward.

    Implementations return a dictionary with the strict schema
    ``{"status": "pass" | "fail" | "insufficient_evidence",
    "passed": bool | None, "reason": str, ...}``.  A pass must use
    ``passed=true``, a fail must use ``passed=false``, and insufficient
    evidence must use ``passed=null``.  Extra diagnostic fields such as
    ``evidence`` and ``violations`` are preserved.  ``trajectory`` is the
    child's trajectory as the trace builds it (steps with their annotations).
    """

    async def judge(self, *, goal: str, trajectory: Trajectory) -> dict[str, Any]: ...


class SubagentParentUsageJudge(Protocol):
    """Judge whether a parent observably used a successful child's contribution."""

    async def judge_parent_usage(
        self,
        *,
        parent_goal: str,
        parent_trajectory: Trajectory,
        child_goal: str,
        child_trajectory: Trajectory,
    ) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- messages and goals


def return_message(result: AgentResult) -> str:
    """What a parent sees of a finished child: its final message, or a failure text (the legacy wording)."""

    if result.final_message:
        return result.final_message
    if result.status == "limit":
        return BUDGET_EXHAUSTED_MESSAGE
    if result.status != "completed":
        return FAILED_MESSAGE
    return ""


def error_message(result: AgentResult) -> str | None:
    """The legacy trajectory ``error_message`` of a run."""

    if result.status == "limit":
        return (
            result.error or "WARNING: Exhausted budget when running episode. Halting episode; task may be incomplete."
        )
    if result.status != "completed":
        return result.error or f"Subagent {result.status}"
    return result.error


def clip(value: str | None, *, limit: int = 6000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...[truncated]..."


def format_verifier_goal(
    *,
    child_goal: str,
    child_trajectory_id: str,
    child_message: str,
    child_error_message: str | None,
    child_task_goal: str | None = None,
) -> str:
    return (
        "You are a verifier agent judging a completed child-agent task.\n\n"
        "Do not trust the child agent's summary. Use available tools to inspect "
        "the environment, files, and other externally visible state before giving "
        "a verdict. Avoid mutating state unless a read-only inspection path is "
        "not available.\n\n"
        f"Judged Child Trajectory ID:\n{child_trajectory_id}\n\n"
        f"Judged Child Goal:\n{clip(child_task_goal or child_goal)}\n\n"
        f"Child Final Message:\n{clip(child_message)}\n\n"
        f"Child Error Message:\n{clip(child_error_message)}\n\n"
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


def propagate_verifier_task_misc(
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

    propagated = dict(parent_task_misc if child_task_misc is None else child_task_misc)
    propagated[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] = True
    propagated.pop(SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY, None)
    return propagated


def is_verifier_task_misc(task_misc: dict[str, Any]) -> bool:
    return bool(task_misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY))


# --------------------------------------------------------------------------- outcome verdicts


def parse_json_object(text: str) -> dict[str, Any] | None:
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


def coerce_score(value: Any, *, status: str) -> float:
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


def normalize_judgment(
    raw_message: str,
    *,
    verifier_trajectory_id: str,
    verifier_error_message: str | None,
    verifier_finished: bool,
    verifier_forced_zero: bool = False,
) -> dict[str, Any]:
    """Parse a verifier's final message into the legacy judgment record."""

    parsed = parse_json_object(raw_message)
    if parsed is None:
        return {
            "status": "unparseable",
            "score": 0.0,
            "summary": clip(raw_message, limit=2000),
            "raw_response": raw_message,
            "verifier_trajectory_id": verifier_trajectory_id,
            "verifier_error_message": verifier_error_message,
            SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
        }

    status = str(parsed.get("status") or "unknown").strip().lower()
    raw_score = parsed.get("score")
    if raw_score is None:
        score = coerce_score(None, status=status)
        score_is_numeric = True
    else:
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0
            score_is_numeric = False
        else:
            score_is_numeric = bool(not isinstance(raw_score, bool) and math.isfinite(score) and 0.0 <= score <= 1.0)
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
    normalized["verifier_trajectory_id"] = verifier_trajectory_id
    normalized["verifier_error_message"] = verifier_error_message
    # A syntactically valid object is not enough to create a trustworthy
    # policy target: the verifier must have emitted a finish result and
    # returned one of the statuses requested by the judgment schema. A valid ``failed`` or
    # ``insufficient_evidence`` judgment remains a legitimate zero-reward
    # target; only missing/malformed/non-finished judgments are suppressed.
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = bool(
        verifier_finished and not verifier_forced_zero and status in _VALID_JUDGMENT_STATUSES and score_is_consistent
    )
    if status in _VALID_JUDGMENT_STATUSES and not score_is_consistent:
        normalized["schema_error"] = "Outcome verifier status and score are inconsistent."
    return normalized


def judge_error_judgment(summary: str) -> dict[str, Any]:
    """The verdict of a verifier that could not run (e.g. its launch was rejected)."""

    return {
        "status": "judge_error",
        "score": 0.0,
        "summary": summary,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


PENDING_JUDGMENT: dict[str, Any] = {
    "status": "pending",
    "score": 0.0,
    "summary": "Subagent reward verification is still in progress.",
    SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
}


# --------------------------------------------------------------------------- behavior verdicts


def _unparseable_behavior_judgment(raw_judgment: Any, *, reason: str) -> dict[str, Any]:
    """Return a fail-closed result for a malformed behavior-judge response."""

    return {
        "status": "unparseable",
        "passed": None,
        "gate": 0.0,
        "reason": reason,
        "raw_response": raw_judgment,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


def normalize_behavior_judgment(raw_judgment: Any) -> dict[str, Any]:
    """Validate a behavior verdict and derive its strict binary reward gate."""

    if not isinstance(raw_judgment, dict):
        return _unparseable_behavior_judgment(raw_judgment, reason="Behavior judge did not return a JSON object.")

    raw_status = raw_judgment.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    has_passed = "passed" in raw_judgment
    passed = raw_judgment.get("passed")
    reason = raw_judgment.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return _unparseable_behavior_judgment(
            raw_judgment, reason="Behavior judgment is missing a non-empty string `reason`."
        )

    expected_passed = {"pass": True, "fail": False, "insufficient_evidence": None}.get(status, object())
    if status not in _VALID_BEHAVIOR_JUDGMENT_STATUSES or not has_passed or passed is not expected_passed:
        return _unparseable_behavior_judgment(
            raw_judgment,
            reason="Behavior judgment must pair status/pass, fail/false, or insufficient_evidence/null exactly.",
        )

    normalized = dict(raw_judgment)
    normalized["status"] = status
    normalized["passed"] = passed
    normalized["reason"] = reason.strip()
    normalized["gate"] = 1.0 if passed is True else 0.0
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = status in {"pass", "fail"}
    return normalized


def behavior_judge_error(error: Exception) -> dict[str, Any]:
    return {
        "status": "judge_error",
        "passed": None,
        "gate": 0.0,
        "reason": f"Behavior judge raised {type(error).__name__}: {error}",
        "error_type": type(error).__name__,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


async def run_behavior_judge(judge: SubagentBehaviorJudge, *, goal: str, trajectory: Trajectory) -> dict[str, Any]:
    try:
        raw_judgment = await judge.judge(goal=goal, trajectory=trajectory)
    except Exception as error:
        return behavior_judge_error(error)
    return normalize_behavior_judgment(raw_judgment)


def behavior_judgment_not_run(*, outcome_eligible: bool) -> dict[str, Any]:
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


def combine_outcome_and_behavior_judgments(
    outcome_judgment: dict[str, Any],
    behavior_judgment: dict[str, Any],
) -> dict[str, Any]:
    """Gate the environment verifier score with the behavioral verdict."""

    outcome_status = str(outcome_judgment.get("status") or "")
    outcome_score = coerce_score(outcome_judgment.get("score"), status=outcome_status)
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


# --------------------------------------------------------------------------- parent-usage verdicts


def _unparseable_parent_usage_judgment(raw_judgment: Any, *, reason: str) -> dict[str, Any]:
    """Return a fail-closed result for a malformed parent-usage verdict."""

    return {
        "status": "unparseable",
        "used": None,
        "gate": 0.0,
        "reason": reason,
        "raw_response": raw_judgment,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


def normalize_parent_usage_judgment(raw_judgment: Any) -> dict[str, Any]:
    """Validate a parent-usage verdict and derive its policy-data gate."""

    if not isinstance(raw_judgment, dict):
        return _unparseable_parent_usage_judgment(
            raw_judgment, reason="Parent-usage judge did not return a JSON object."
        )

    raw_status = raw_judgment.get("status")
    status = raw_status.strip().lower() if isinstance(raw_status, str) else ""
    has_used = "used" in raw_judgment
    used = raw_judgment.get("used")
    reason = raw_judgment.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return _unparseable_parent_usage_judgment(
            raw_judgment, reason="Parent-usage judgment is missing a non-empty string `reason`."
        )

    expected_used = {"used": True, "not_used": False, "insufficient_evidence": None}.get(status, object())
    if status not in _VALID_PARENT_USAGE_JUDGMENT_STATUSES or not has_used or used is not expected_used:
        return _unparseable_parent_usage_judgment(
            raw_judgment,
            reason="Parent-usage judgment must pair used/true, not_used/false, or insufficient_evidence/null exactly.",
        )

    normalized = dict(raw_judgment)
    normalized["status"] = status
    normalized["used"] = used
    normalized["reason"] = reason.strip()
    normalized["gate"] = 1.0 if used is True else 0.0
    normalized[SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY] = status in {"used", "not_used"}
    return normalized


def parent_usage_judge_error(error: Exception) -> dict[str, Any]:
    return {
        "status": "judge_error",
        "used": None,
        "gate": 0.0,
        "reason": f"Parent-usage judge raised {type(error).__name__}: {error}",
        "error_type": type(error).__name__,
        SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
    }


async def run_parent_usage_judge(
    judge: SubagentParentUsageJudge,
    *,
    parent_goal: str,
    parent_trajectory: Trajectory,
    child_goal: str,
    child_trajectory: Trajectory,
) -> dict[str, Any]:
    try:
        raw_judgment = await judge.judge_parent_usage(
            parent_goal=parent_goal,
            parent_trajectory=parent_trajectory,
            child_goal=child_goal,
            child_trajectory=child_trajectory,
        )
    except Exception as error:
        return parent_usage_judge_error(error)
    return normalize_parent_usage_judgment(raw_judgment)


# The parent-usage verdict of a candidate child in a rollout whose judges are off: no judge ran and nothing is gated.
PARENT_USAGE_SKIPPED: dict[str, Any] = {
    "status": "skipped",
    "reason": JUDGES_OFF,
    SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY: False,
}


def _last_step_reward_misc(trajectory: Trajectory) -> dict[str, Any] | None:
    if not trajectory.steps:
        return None
    step = trajectory.steps[-1]
    if isinstance(step, dict):
        return step.setdefault("misc", {}).setdefault("reward_misc", {})
    return step.misc.setdefault("reward_misc", {})


def _record_parent_usage_judgment(trajectory: Trajectory, judgment: dict[str, Any]) -> None:
    """Apply a usage verdict only to policy eligibility, never to reward."""

    trajectory.misc[SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY] = judgment
    usage_gate = 1.0 if judgment.get("gate") == 1.0 else 0.0
    usage_eligible = bool(judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY))
    # Suppress only a decisive not-used verdict. Ambiguous or unavailable
    # usage evidence must not penalize an otherwise verified, well-behaved
    # child. The verified reward remains intact for rollout statistics and
    # group baselines; only this child's policy datums are suppressed.
    if usage_eligible and usage_gate != 1.0:
        trajectory.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] = True
    reward_misc = _last_step_reward_misc(trajectory)
    if reward_misc is not None:
        reward_misc["reward/subagent_parent_usage_gate"] = usage_gate
        reward_misc["reward/subagent_parent_usage_judgment_valid"] = 1.0 if usage_eligible else 0.0


def _trajectory_goal(trajectory: Trajectory) -> str:
    return (trajectory.task.goal or "") if trajectory.task is not None else ""


def _is_verifier_trajectory(trajectory: Trajectory) -> bool:
    task_misc = dict(trajectory.task.misc) if trajectory.task is not None else {}
    return is_verifier_task_misc(task_misc) or bool(trajectory.misc.get(SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY))


async def apply_subagent_parent_usage_gates(
    collection: TrajectoryCollection,
    judge: SubagentParentUsageJudge,
    *,
    max_concurrency: int = 4,
) -> None:
    """Suppress positive child policy data that its completed parent did not use.

    Outcome/behavior-zero children remain trainable negative examples and are
    intentionally not audited here.  Parent usage is necessarily evaluated
    after the episode, when the parent's post-delegation behavior is complete,
    so programs register this as a finalizer.  In a rollout whose judges are
    off (:func:`platoon.utils.judges.judges_enabled`) the judge is never
    called: each candidate records :data:`PARENT_USAGE_SKIPPED` and keeps its
    policy data.
    """

    if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    judging = judges_enabled(collection)  # a rollout whose judges are off records each candidate as skipped

    candidates: list[tuple[Trajectory, Trajectory | None]] = []
    for trajectory in sorted(collection.trajectories.values(), key=lambda item: item.id):
        if trajectory.parent_info is None or _is_verifier_trajectory(trajectory):
            continue
        if trajectory.misc.get(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY):
            continue
        effective = trajectory.misc.get(SUBAGENT_REWARD_JUDGMENT_MISC_KEY)
        if not isinstance(effective, dict) or not bool(effective.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)):
            continue
        if coerce_score(effective.get("score"), status=str(effective.get("status") or "")) <= 0.0:
            continue
        candidates.append((trajectory, collection.trajectories.get(trajectory.parent_info.id)))
    if not judging:
        for child, _ in candidates:
            child.misc[SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY] = dict(PARENT_USAGE_SKIPPED)
        return

    semaphore = asyncio.Semaphore(max_concurrency)

    async def judge_candidate(child: Trajectory, parent: Trajectory | None) -> tuple[Trajectory, dict[str, Any]]:
        if parent is None:
            return child, _unparseable_parent_usage_judgment(
                None, reason="The successful child's parent trajectory is missing from the rollout collection."
            )
        async with semaphore:
            return child, await run_parent_usage_judge(
                judge,
                parent_goal=_trajectory_goal(parent),
                parent_trajectory=parent,
                child_goal=_trajectory_goal(child),
                child_trajectory=child,
            )

    results = await asyncio.gather(*(judge_candidate(child, parent) for child, parent in candidates))
    for child, judgment in results:
        _record_parent_usage_judgment(child, judgment)


# --------------------------------------------------------------------------- recording on the judged agent


def record_judgment(agent: AgentSession, judgment: dict[str, Any]) -> Outcome:
    """Write ``judgment`` onto the judged child agent and return the child's outcome.

    The agent's misc gets the effective verdict and, when the verdict is not training-eligible,
    ``exclude_from_policy_training``; its last recorded call's ``reward_misc`` gets the reward components.  A
    child the env already scored as a forced zero keeps reward 0.
    """

    from platoon.envs.service import Outcome

    score = coerce_score(judgment.get("score"), status=str(judgment.get("status") or ""))
    if trajectory_has_forced_zero_reward(agent):
        score = 0.0
    agent.misc[SUBAGENT_REWARD_JUDGMENT_MISC_KEY] = judgment
    if bool(judgment.get(SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY)):
        agent.misc.pop(EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY, None)
    else:
        agent.misc[EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY] = True
    if agent.calls:
        step_misc = agent.step_misc.setdefault(agent.calls[-1].call_id, {})
        reward_misc = step_misc.setdefault("reward_misc", {})
        reward_misc["reward/success"] = score
        reward_misc["reward/subagent_judgment"] = score
        outcome_judgment = agent.misc.get(SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY)
        if isinstance(outcome_judgment, dict):
            reward_misc["reward/subagent_outcome_judgment"] = coerce_score(
                outcome_judgment.get("score"), status=str(outcome_judgment.get("status") or "")
            )
        elif "outcome_score" in judgment:
            reward_misc["reward/subagent_outcome_judgment"] = coerce_score(
                judgment.get("outcome_score"), status=str(judgment.get("outcome_status") or "")
            )
        if "behavior_gate" in judgment:
            reward_misc["reward/subagent_behavior_gate"] = 1.0 if judgment.get("behavior_gate") == 1.0 else 0.0
    return Outcome(
        status="scored",
        rewards={"reward": score},
        reason=None if judgment.get("status") != "pending" else "verification_pending",
        metadata={SUBAGENT_REWARD_JUDGMENT_MISC_KEY: judgment},
    )


def verifier_task_misc(child_task_misc: dict[str, Any], child_trajectory_id: str) -> dict[str, Any]:
    return {
        **child_task_misc,
        SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY: True,
        SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY: child_trajectory_id,
    }


def mark_verifier_agent(verifier: AgentSession, child_trajectory_id: str) -> None:
    """A verifier trajectory never trains, and names the trajectory it judges."""

    verifier.misc[SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY] = True
    verifier.misc[SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY] = child_trajectory_id
    verifier.misc[EXCLUDE_FROM_TRAINING_MISC_KEY] = True


def verifier_judgment(verifier: AgentSession, result: AgentResult) -> dict[str, Any]:
    """Normalize a finished verifier agent's verdict."""

    return normalize_judgment(
        return_message(result),
        verifier_trajectory_id=verifier.trajectory_id,
        verifier_error_message=error_message(result),
        verifier_finished=bool(result.status == "completed" and result.final_message),
        verifier_forced_zero=bool(verifier.misc.get(TRAJECTORY_FORCED_ZERO_REWARD_MISC_KEY)),
    )


__all__ = [
    "BUDGET_EXHAUSTED_MESSAGE",
    "DEFAULT_VERIFIER_MAX_STEPS",
    "EXCLUDE_FROM_POLICY_TRAINING_MISC_KEY",
    "EXCLUDE_FROM_TRAINING_MISC_KEY",
    "FAILED_MESSAGE",
    "PARENT_USAGE_SKIPPED",
    "SUBAGENT_BEHAVIOR_JUDGMENT_MISC_KEY",
    "SUBAGENT_OUTCOME_JUDGMENT_MISC_KEY",
    "SUBAGENT_PARENT_USAGE_JUDGMENT_MISC_KEY",
    "SUBAGENT_REWARD_JUDGMENT_MISC_KEY",
    "SUBAGENT_REWARD_JUDGMENT_TRAINING_ELIGIBLE_KEY",
    "SUBAGENT_REWARD_VERIFIER_TASK_MISC_KEY",
    "SUBAGENT_REWARD_VERIFIES_TRAJECTORY_ID_MISC_KEY",
    "VERIFIER_ROLE",
    "SubagentBehaviorJudge",
    "SubagentParentUsageJudge",
    "apply_subagent_parent_usage_gates",
    "combine_outcome_and_behavior_judgments",
    "format_verifier_goal",
    "normalize_behavior_judgment",
    "normalize_judgment",
    "normalize_parent_usage_judgment",
    "record_judgment",
]
