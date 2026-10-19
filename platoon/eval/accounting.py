"""What an evaluation counts in each rollout's trajectories besides its outcome: model usage, cost, how the policy's
backend behaved, and what the judges did.

:func:`rollout_accounting` reads one rollout's trajectory collection (its dump), so a report built later from the
rollouts on disk (``stage: report``) counts exactly what the run counted.  :func:`report_blocks` sums the rollouts
into the report's ``usage``, ``cost``, ``backend`` and ``judges`` blocks.

Every model call is counted once, in one of four buckets: ``policy`` and ``aux`` (the calls of the policy's agents,
by their kind), ``judge`` (``judge`` calls, and every call of a ``judge``-role agent) and ``verifier`` (the calls of
verifier agents and their helpers, which judge children).  Usage is the call's own record: token counts for a
Tinker policy, the provider's reported usage for a hosted one (with cached-prompt, cache-write and reasoning tokens
when it reports them).  A hosted call is one whose record names its ``dialect``
(``platoon.gateway.core.ProviderRecord``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from platoon.utils.judges import (
    CHILD_GRADING_MISC_KEY,
    JUDGES_REFUSED_MISC_KEY,
    is_judge_trajectory,
)

BUCKETS = ("policy", "aux", "judge", "verifier")
# Usage keys a call may report, summed per bucket; a call without one counts 0 there.
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cache_write_tokens", "reasoning_tokens")
# The trajectory misc key of an agent's model calls that failed after retries, as the model gateway records them.
BACKEND_FAILURES_MISC_KEY = "backend_failures"
# The trajectory misc key of judge calls kept for audit (``platoon.gateway.trace.JUDGE_CALLS_MISC_KEY``).
_JUDGE_CALLS_MISC_KEY = "judge_calls"


def _empty_usage() -> dict[str, float]:
    return {"calls": 0, **dict.fromkeys(USAGE_KEYS, 0)}


def _add_usage(totals: dict[str, float], record: Mapping[str, Any]) -> None:
    totals["calls"] += 1
    usage = record.get("usage")
    if isinstance(usage, Mapping):
        for key in USAGE_KEYS:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += value


def _calls(trajectory: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """``(kind, record)`` of each model call a trajectory holds: its steps' ``action_misc`` and its audited judge
    calls."""

    for step in trajectory.get("steps") or []:
        misc = step.get("misc") if isinstance(step, Mapping) else None
        action = misc.get("action_misc") if isinstance(misc, Mapping) else None
        if isinstance(action, Mapping):
            yield str(action.get("kind") or "policy"), action
    misc = trajectory.get("misc")
    for record in (misc.get(_JUDGE_CALLS_MISC_KEY) if isinstance(misc, Mapping) else None) or []:
        if isinstance(record, Mapping):
            yield "judge", record


def rollout_accounting(collection: Mapping[str, Any]) -> dict[str, Any]:
    """One rollout's ``usage``, ``cost``, ``backend`` and ``judges`` records, from its trajectory collection dump."""

    usage = {bucket: _empty_usage() for bucket in BUCKETS}
    cost: dict[str, Any] = {"usd": 0.0, "priced_calls": 0, "unpriced_calls": 0}
    backend: dict[str, Any] = {
        "attempts": 0,
        "rate_limited": 0,
        "failed_calls": 0,
        "refusals": 0,
        "calls_without_usage": 0,
        "served_models": {},
        "fatal": None,
    }
    judges: dict[str, Any] = {
        "verifier_agents": 0,
        "verifier_steps": 0,
        "judge_calls": 0,
        "grading_fallbacks": {},
        "refused": 0,
        "refused_first": None,
    }
    trajectories = collection.get("trajectories")
    for trajectory in trajectories.values() if isinstance(trajectories, Mapping) else []:
        if not isinstance(trajectory, Mapping):
            continue
        judging = is_judge_trajectory(trajectory)
        if judging:
            judges["verifier_agents"] += 1
            judges["verifier_steps"] += len(trajectory.get("steps") or [])
        for kind, record in _calls(trajectory):
            bucket = "judge" if kind == "judge" else "verifier" if judging else kind if kind in BUCKETS else "policy"
            if bucket == "judge":
                judges["judge_calls"] += 1
            _add_usage(usage[bucket], record)
            _count_call(record, cost, backend)
        misc = trajectory.get("misc")
        misc = misc if isinstance(misc, Mapping) else {}
        fallback = misc.get(CHILD_GRADING_MISC_KEY)
        if isinstance(fallback, Mapping) and fallback.get("applied") == "none":
            reason = str(fallback.get("reason"))
            judges["grading_fallbacks"][reason] = judges["grading_fallbacks"].get(reason, 0) + 1
        refused = misc.get(JUDGES_REFUSED_MISC_KEY)
        if isinstance(refused, Mapping):
            judges["refused"] += int(refused.get("count") or 0)
            judges["refused_first"] = judges["refused_first"] or refused.get("first")
        failures = misc.get(BACKEND_FAILURES_MISC_KEY)
        if isinstance(failures, Mapping):
            backend["failed_calls"] += int(failures.get("count") or 0)
            backend["fatal"] = backend["fatal"] or failures.get("fatal")
    priced = cost.pop("priced_calls")
    cost["usd"] = cost["usd"] if priced else None  # a rollout none of whose calls was priced has no cost
    return {"usage": usage, "cost": cost, "backend": backend, "judges": judges}


def _count_call(record: Mapping[str, Any], cost: dict[str, Any], backend: dict[str, Any]) -> None:
    """Add one call's cost and provider record (a hosted call carries them in its ``action_misc``)."""

    if record.get("dialect") is None:
        return  # a Tinker call: no cost, no provider
    usd = record.get("cost_usd")
    if isinstance(usd, (int, float)) and not isinstance(usd, bool):
        cost["usd"] += float(usd)
        cost["priced_calls"] += 1
    else:
        cost["unpriced_calls"] += 1
    backend["attempts"] += int(record.get("attempts") or 1)
    backend["rate_limited"] += int(record.get("rate_limited") or 0)
    if record.get("refusal"):
        backend["refusals"] += 1
    if record.get("usage_reported") is False:  # its usage counts nothing and it has no cost
        backend["calls_without_usage"] += 1
    served = record.get("served_model")
    if isinstance(served, str) and served:
        backend["served_models"][served] = backend["served_models"].get(served, 0) + 1


def _merge_counts(into: dict[str, Any], counts: Mapping[str, Any]) -> None:
    for key, value in counts.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            into[key] = into.get(key, 0) + value


def report_blocks(records: Iterable[Any], *, judges_enabled: bool) -> dict[str, Any]:
    """The report's ``usage``, ``cost``, ``backend`` and ``judges`` blocks, summed over ``records``
    (``EvalRolloutRecord`` objects; errored rollouts included, since their calls were made and paid for)."""

    records = list(records)
    rollouts = len(records)
    usage = {bucket: _empty_usage() for bucket in BUCKETS}
    total_usd, priced, unpriced = 0.0, False, 0
    backend = {
        "attempts": 0,
        "rate_limited": 0,
        "failed_calls": 0,
        "refusals": 0,
        "calls_without_usage": 0,
        "rollouts_with_failed_calls": 0,
    }
    served_models: dict[str, int] = {}
    judges = {"verifier_agents": 0, "verifier_steps": 0, "judge_calls": 0, "refused": 0}
    fallbacks: dict[str, int] = {}
    for record in records:
        for bucket, totals in (getattr(record, "usage", None) or {}).items():
            if bucket in usage and isinstance(totals, Mapping):
                _merge_counts(usage[bucket], totals)
        cost = getattr(record, "cost", None) or {}
        if cost.get("usd") is not None:
            total_usd += float(cost["usd"])
            priced = True
        unpriced += int(cost.get("unpriced_calls") or 0)
        calls = getattr(record, "backend", None) or {}
        for key in ("attempts", "rate_limited", "failed_calls", "refusals", "calls_without_usage"):
            backend[key] += int(calls.get(key) or 0)
        backend["rollouts_with_failed_calls"] += 1 if calls.get("failed_calls") else 0
        _merge_counts(served_models, calls.get("served_models") or {})
        judged = getattr(record, "judges", None) or {}
        for key in ("verifier_agents", "verifier_steps", "judge_calls", "refused"):
            judges[key] += int(judged.get(key) or 0)
        _merge_counts(fallbacks, judged.get("grading_fallbacks") or {})
    usage_block: dict[str, Any] = {
        bucket: {
            "total": totals,
            "mean_per_rollout": {key: (value / rollouts if rollouts else 0.0) for key, value in totals.items()},
        }
        for bucket, totals in usage.items()
    }
    cost_block = {
        "total_usd": total_usd if priced else None,
        "mean_usd_per_rollout": (total_usd / rollouts) if priced and rollouts else None,
        "unpriced_calls": unpriced,
    }
    judges_block = {
        "enabled": judges_enabled,
        **judges,
        "children_sharing_episode_outcome": dict(sorted(fallbacks.items())),
    }
    return {
        "usage": usage_block,
        "cost": cost_block,
        "backend": {**backend, "served_models": dict(sorted(served_models.items()))},
        "judges": judges_block,
    }


__all__ = ["BACKEND_FAILURES_MISC_KEY", "BUCKETS", "USAGE_KEYS", "report_blocks", "rollout_accounting"]
