"""The scenarios of the golden trace, the one file a new scenario changes: each is a config laid over
:data:`BASE_CONFIG`, how its starts end, how its fake rollouts end, and a check that it still exercises what it is
there for.  A config with ``environments`` runs through ``platoon train``'s entry point; any other runs the trainer as
a train script does."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import trace_fakes
from trace_fakes import Rollouts, Start
from trace_format import Trace, series

BASE_CONFIG: dict[str, Any] = {
    "train": {
        "model_name": "test/model",
        "renderer_name": "test-renderer",
        "batch_size": 2,
        "max_training_steps": 2,
        "max_staleness": 0,
        "sample_timeout_seconds": 600,
        "workflow_config": {"group_size": 4},
    },
    "eval": {"every": 0},
    "watchdog": {"enabled": False},
    "stats": {"wandb": {"mode": "disabled"}},
}


@dataclass(frozen=True)
class Scenario:
    """A run to trace.

    ``config``: the sections laid over :data:`BASE_CONFIG`, mappings merged key by key.  Without ``environments``, its
    tasks are ``tasks`` training tasks (``train:<i>``) and ``eval_tasks`` evaluation tasks (``val:<i>``).
    ``check``: asserts on the trace that the scenario still exercises what it is there for; :func:`check_trains`
    applies to every scenario too.
    """

    description: str
    config: Mapping[str, Any]
    check: Callable[[Trace], None]
    tasks: int = 6
    eval_tasks: int = 0
    starts: tuple[Start, ...] = (Start(),)
    rollouts: Rollouts = Rollouts()


def entry(name: str, count: int = 4, **dataset_kwargs: Any) -> dict[str, Any]:
    """An environment entry, whose ``count`` tasks :func:`trace_fakes.task_rows` lists."""

    fakes = trace_fakes.__name__
    return {
        "name": name,
        "dataset_loader": f"{fakes}:task_rows",
        "dataset_kwargs": {"count": count, **dataset_kwargs},
        "rollout": f"{fakes}:unused_rollout",
        "task_loader": f"{fakes}:get_task",
    }


# --------------------------------------------------------------------------- what a check reads


def messages(trace: Trace) -> list[str]:
    return [message for start in trace["starts"] for message in start["messages"]]


def calls(trace: Trace, name: str, start: int = 0) -> list[Any]:
    return [call for call in trace["starts"][start]["tinker"] if call[0] == name]


def requests(trace: Trace, scope: str = "train") -> list[list[Any]]:
    return [row for start in trace["starts"] for row in start["requests"].get(scope, [])]


def resumed_from(trace: Trace, start: int) -> str:
    opened = calls(trace, "create_training_client_from_state_with_optimizer", start)
    assert opened, f"start {start} did not resume from a checkpoint"
    return opened[0][1]


def check_trains(trace: Trace) -> None:
    """That the run trains: each start that completes applies an update, and its rollouts sample two policy
    versions or more."""

    for index, start in enumerate(trace["starts"]):
        if start["outcome"] == "completed":
            assert calls(trace, "optim_step", index), f"start {index} applies no update"
    versions = sorted({row[2] for row in requests(trace)})
    assert len(versions) >= 2, f"every training rollout samples policy version {versions}"


# --------------------------------------------------------------------------- the checks


def _check_loo(trace: Trace) -> None:
    assert any("trainable rollout(s)" in message for message in messages(trace)), "no group below its minimum"
    assert any(value and value > 0 for value in series(trace, "train/credit/baseline/spread")), "no LOO spread"


def _check_entry(trace: Trace) -> None:
    assert requests(trace, "eval"), "no evaluation rollout"
    steps = {setting["task"]["max_steps"] for setting in trace["request_settings"]}
    assert steps == {7}, f"the entry's own max_steps reaches neither training nor evaluation alone: {steps}"


def _check_micro(trace: Trace) -> None:
    assert 2.0 in series(trace, "train/batch/optimizer_updates"), "no step applied two updates"
    assert len(calls(trace, "forward_backward")) > len(calls(trace, "optim_step")), "no gradient accumulation"
    assert 2.0 in series(trace, "rollout_client/groups_in_flight_max"), "never two groups at once"
    zero = "Zero policy-gradient objective for microbatch"
    assert any(zero in message for message in messages(trace)), "no zero-advantage microbatch kept by the filter off"


def _check_mix(trace: Trace) -> None:
    for name in ("quick", "slow"):
        assert any(series(trace, f"train/batch/environment/{name}/groups")), f"no group of {name} trained"


def _check_capped(trace: Trace) -> None:
    assert any(series(trace, "train/batch/parked_rows")), "the label limit never held a draw back"


def _check_label_by(trace: Trace) -> None:
    assert any("straggler" in message for message in messages(trace)), "no straggler cancelled"
    assert series(trace, "train/batch/kind/hard/groups"), "no metrics by the label column"


def _check_informative(trace: Trace) -> None:
    assert any(series(trace, "train/admission/rejected")), "the rule rejected no group"
    assert any(value > 1 for value in series(trace, "train/admission/draws_per_admitted")), "no place refilled"
    predicted = series(trace, "train/sampler/environment/pool/predicted_informative")
    assert predicted, "the scoring sampler's predictions of the groups the rule took are not counted"


def _check_pass_rate(trace: Trace) -> None:
    assert any(series(trace, "train/sampler/environment/pool/draws/exploit")), "pass_rate never exploited"


def _check_eval(trace: Trace) -> None:
    assert series(trace, "eval/step", kind="eval") == [0, 2, 4], "evaluations not at steps 0, 2 and 4"


def _check_resume(trace: Trace) -> None:
    first, second = trace["starts"]
    assert first["outcome"].startswith("crashed"), "the first start did not crash"
    name = resumed_from(trace, 1).rsplit("/", 1)[-1]
    assert name.startswith("recovery-"), f"resumed from {name}, not a recovery checkpoint"
    saved = [save for save in first["saves"] if save["name"] == name]
    assert saved and saved[-1]["dispatch"].get("kept"), "the checkpoint resumed from kept no group"
    kept = [re.search(r": (\d+) groups that had settled", message) for message in second["messages"]]
    assert any(found and int(found.group(1)) > 0 for found in kept), "the resumed start trained no kept group first"


def _check_depth(trace: Trace) -> None:
    assert any(series(trace, "train/zero_advantage_filter/filtered_datums")), "the filter dropped no datum"
    assert series(trace, "train/credit/depth1/weight"), "no depth weighting"


def _check_stale(trace: Trace) -> None:
    assert any(series(trace, "train/batch/stale_groups")), "no group dropped as stale"
    bound = "beyond the bound of 2 versions of environments[slow]"
    assert any(bound in message for message in messages(trace)), "the slow entry's own bound dropped no group"
    stamps: dict[tuple[str, int], set[int]] = {}
    for row in requests(trace):  # a group's rollouts: one task, drawn on one policy version
        if len(row) > 5:  # a rollout cancelled before its first sample stamped nothing
            stamps.setdefault((row[0], row[2]), set()).add(row[5])
    assert any(len(found) > 1 for found in stamps.values()), "no group whose rollouts sampled different policies"


def _rollout_starts(trace: Trace) -> list[set[tuple[int, int]]]:
    """For each training group in turn, the policy versions and updates applied that its rollouts' requests carry, as
    pairs: one pair when its rollouts started together.  The permits go to one draw at a time, so a group's rollouts
    are requested one after another, number 0 first."""

    groups: list[set[tuple[int, int]]] = []
    for row in requests(trace):
        if row[1] == 0 or not groups:
            groups.append(set())
        groups[-1].add((row[2], row[4]))
    return groups


def _check_permits_bind(trace: Trace) -> None:
    """That the rollout permits (8) held the dispatcher back, that each step's mean of the permits held stayed within
    them, and that no more rollouts than that ever ran at once."""

    assert any(series(trace, "train/dispatch/blocked_share/permits")), "the permits never held the dispatcher back"
    held = series(trace, "rollout_client/permits_held_mean")
    assert held and max(held) <= 8, f"the permits held are not logged, or their mean exceeds the 8 there are: {held}"
    running = series(trace, "rollout_client/rollouts_in_flight_max")
    assert max(running) <= 8, f"more rollouts at once than the 8 permits: {running}"


def _check_permits_group(trace: Trace) -> None:
    _check_permits_bind(trace)
    most = max(series(trace, "rollout_client/groups_in_flight_max"), default=0)
    assert most == 2, f"at most {most:g} groups at once, not the two whose permits the 8 hold whole"
    assert all(len(starts) == 1 for starts in _rollout_starts(trace)), "a group's rollouts did not start together"


def _check_permits_rollout(trace: Trace) -> None:
    _check_permits_bind(trace)
    assert any("rollouts start as permits free" in message for message in messages(trace)), "no rolling starts"
    most = max(series(trace, "rollout_client/groups_in_flight_max"), default=0)
    assert most > 2, f"at most {most:g} groups at once: none started while two others still ran"
    versions = [{version for version, _ in starts} for starts in _rollout_starts(trace)]
    assert any(len(found) > 1 for found in versions), "no group whose rollouts started on either side of a publication"


def _check_start_limit(trace: Trace) -> None:
    """That the start limit (2) bound: never more than 2 rollouts starting at once, from their request to their first
    model call, as the fake rollouts saw them, and 2 at times, while more ran; members waited for their slots; every
    request asked for the first-call signal; the run said so as it started; and every step logged the start-limit
    metrics, consistent with that: at most 2 slots held on average, slots that came back at their rollout's first call,
    none at its end or on its timer, and no more of those than first calls."""

    limit, name = 2, "rollout_client/start_limit"
    (start,) = trace["starts"]
    ups = start.get("start_ups", {})
    assert ups.get("most_at_once") == limit, f"not at most and at times {limit} rollouts starting at once: {ups}"
    running = series(trace, "rollout_client/rollouts_in_flight_max")
    assert max(running, default=0) > limit, f"never more rollouts running than starting: {running}"
    flags = {setting["signal_first_call"] for setting in trace["request_settings"]}
    assert flags == {True}, f"a request that does not ask for the first-call signal: {flags}"
    said = messages(trace)
    assert any(f"limited to {limit} at once" in message for message in said), "the run did not say its limit"
    assert any("below the largest group size (4)" in message for message in said), "no warning of starts in waves"
    steps = len(start["metrics"]["train"]["steps"])
    causes = ("first_call", "end", "timer", "cancelled")
    keys = ("starting_mean", "waiting_mean", *(f"releases/{cause}" for cause in causes))
    logged = {key: series(trace, f"{name}/{key}") for key in keys}
    for key, values in logged.items():
        assert len(values) == steps and None not in values, f"{name}/{key} is not logged at every step: {values}"
    held = logged["starting_mean"]
    assert any(held) and max(held) <= limit, f"the slots held on average, each step: {held}"
    assert any(logged["waiting_mean"]) and any(series(trace, f"{name}/wait_seconds_p90")), "no member waited"
    late = [*logged["releases/end"], *logged["releases/timer"]]
    assert not any(late), "a slot came back at its rollout's end or on its timer, not at its first call"
    released = sum(logged["releases/first_call"])
    calls, requested = ups.get("first_calls", 0), len(start["requests"]["train"])
    assert 0 < released <= calls <= requested, f"{released:g} first-call releases, {calls} calls, {requested} requests"


def _check_rule_short(trace: Trace) -> None:
    assert any(series(trace, "train/admission/short")), "the rule never closed a step short on its cap"


def _check_rule_held(trace: Trace) -> None:
    _check_rule_short(trace)
    rejected = series(trace, "train/admission/rejected/no_completed_root")
    assert any(rejected), "the rule rejected no group without datums, whose slot it holds"


def _check_rule_stale(trace: Trace) -> None:
    _check_rule_short(trace)
    assert any(series(trace, "train/admission/stale")), "no stale group under the rule"


def _check_cadence(trace: Trace) -> None:
    names = [save["name"] for save in trace["starts"][0]["saves"]]
    assert any(name.isdigit() for name in names), f"no permanent checkpoint: {names}"
    assert any(name.startswith("recovery-") and name != "recovery-000000" for name in names), f"no recovery: {names}"


def _check_eval_resume(trace: Trace) -> None:
    first, second = trace["starts"]
    assert first["outcome"].startswith("crashed"), "the first start did not crash"
    assert resumed_from(trace, 1).endswith("/000003"), "not resumed from the permanent checkpoint 000003"
    assert any("whose evaluation an earlier start completed" in m for m in second["messages"]), "evaluation redone"


def _check_sizes(trace: Trace) -> None:
    numbers = {(row[0].split("::")[0], row[1]) for row in requests(trace)}
    assert ("big", 3) in numbers and ("small", 2) not in numbers, "the entries' own group sizes were not used"


def _check_stop(trace: Trace) -> None:
    first, _ = trace["starts"]
    assert any("stopping at the next step boundary" in m for m in first["messages"]), "no stop request"
    assert any(save["name"].endswith("_interrupted") for save in first["saves"]), "no interrupted save"
    assert resumed_from(trace, 1).endswith("_interrupted"), "the second start did not resume from it"


def _check_stop_mid_step(trace: Trace) -> None:
    _check_stop(trace)
    assert calls(trace, "optim_step"), "the stop came before any update"


def _check_default_straggler(trace: Trace) -> None:
    assert any("straggler" in message and "(quorum 2/3)" in message for message in messages(trace)), "no straggler"


def _check_eval_long(trace: Trace) -> None:
    first_step = [row for row in requests(trace, "eval") if row[2] == 0]
    assert len(first_step) >= 12, "the starting policy's evaluation is not long"
    assert series(trace, "eval/step", kind="eval")[:1] == [0], "the starting policy was not evaluated"


SCENARIOS: dict[str, Scenario] = {
    # ------------------------------------------------------------------- one environment
    "single-loo": Scenario(
        "One environment, no environments list: leave-one-out baselines and a minimum group size",
        {"train": {"workflow_config": {"leave_one_out_baseline": True, "min_successful_group_size": 3}}},
        check=_check_loo,
    ),
    "single-entry": Scenario(
        "One named environment entry with settings of its own (rollout_config.max_steps), in training and evaluation",
        {
            "eval": {"every": 2},
            "environments": [
                {
                    **entry("only", 6),
                    "eval_dataset_kwargs": {"count": 2},
                    "workflow_config": {"rollout_config": {"max_steps": 7}},
                }
            ],
        },
        check=_check_entry,
    ),
    "single-micro": Scenario(
        "Two minibatches of two microbatches, two groups at once, the zero-advantage filter off",
        {
            "train": {
                "batch_size": 4,
                "num_minibatches": 2,
                "num_microbatches": 2,
                "max_concurrent_rollouts": 8,
                "workflow_config": {"filter_zero_advantage_datums": False},
            }
        },
        tasks=8,
        check=_check_micro,
    ),
    "straggler-default": Scenario(
        "A run-level straggler timeout with the default quorum (group_size - 1), in groups of three",
        {"train": {"workflow_config": {"group_size": 3, "straggler_timeout_seconds": 600}}},
        rollouts=Rollouts(stragglers=True),
        check=_check_default_straggler,
    ),
    "depth-zero-adv": Scenario(
        "Depth-level weighting of subagent datums, then the zero-advantage filter, in microbatches of two groups",
        {"train": {"batch_size": 4, "num_microbatches": 2, "workflow_config": {"depth_level_weighting": True}}},
        tasks=8,
        rollouts=Rollouts(subagents=True),
        check=_check_depth,
    ),
    # ------------------------------------------------------------------- mixtures
    "mix-default": Scenario(
        "A mixture of two entries that set nothing of their own",
        {"train": {"batch_size": 4}, "environments": [entry("quick"), entry("slow")]},
        rollouts=Rollouts(slowdown={"slow::": 3.0}),
        check=_check_mix,
    ),
    "mix-sizes": Scenario(
        "A mixture whose entries set their own group sizes (2 and 4), with rollout permits for 8 rollouts",
        {
            "train": {"batch_size": 4, "max_concurrent_rollouts": 8},
            "environments": [
                {**entry("small"), "workflow_config": {"group_size": 2}},
                {**entry("big"), "workflow_config": {"group_size": 4}},
            ],
        },
        check=_check_sizes,
    ),
    "mix-capped": Scenario(
        "The mixture with a limit of one running group on its slow entry (rollout_group_limits)",
        {
            "train": {"batch_size": 4, "max_training_steps": 3, "rollout_group_limits": {"slow": 1}},
            "environments": [entry("quick"), entry("slow")],
        },
        rollouts=Rollouts(slowdown={"slow::": 3.0}),
        check=_check_capped,
    ),
    "capped-k1": Scenario(
        "A limit of two running groups on the slow entry at max_staleness 1, where draws hold their label's slots",
        {
            "train": {
                "batch_size": 4,
                "max_training_steps": 5,
                "max_staleness": 1,
                "max_concurrent_rollouts": 16,
                "rollout_group_limits": {"slow": 2},
                "workflow_config": {"group_size": 2},
            },
            "environments": [entry("quick", 8), entry("slow", 8)],
        },
        rollouts=Rollouts(slowdown={"slow::": 3.0}),
        check=_check_capped,
    ),
    "mix-label-by": Scenario(
        "The mixture labelled by another column (train.label_by), with a straggler policy for one label",
        {
            "train": {
                "batch_size": 4,
                "label_by": "kind",
                "workflow_config": {"straggler_policies": {"hard": {"timeout_seconds": 600, "quorum": 1}}},
            },
            "environments": [entry("quick", kinds=["easy", "hard"]), entry("slow", kinds=["easy", "hard"])],
        },
        rollouts=Rollouts(stragglers=True),
        check=_check_label_by,
    ),
    "stale-follow": Scenario(
        "Groups dropped as stale at max_staleness 1, the slow entry by its own bound (2), with datums stamped by the "
        "policy they sampled",
        {
            "train": {"max_training_steps": 8, "max_staleness": 1, "max_concurrent_rollouts": 16},
            "environments": [entry("quick", 12), {**entry("slow", 4), "max_staleness": 2}],
        },
        rollouts=Rollouts(slowdown={"slow::": 5.0}, follow=True, first_share=0.5),
        check=_check_stale,
    ),
    # ------------------------------------------------------------------- rollout permits
    "permits-group": Scenario(
        "Rollout permits for two groups of four, at max_staleness 2 so that groups run ahead of the step, under "
        "permit_release group (the default): a group holds its permits until its slowest rollout ends",
        {
            "train": {
                "max_training_steps": 4,
                "max_staleness": 2,
                "max_concurrent_rollouts": 8,
                "permit_release": "group",
            }
        },
        tasks=8,
        check=_check_permits_group,
    ),
    "permits-rollout": Scenario(
        "Rollout permits for two groups of four, at max_staleness 2 so that groups run ahead of the step, under "
        "permit_release rollout: each rollout returns its permit as it ends, and a group's rollouts start as permits "
        "free, some after the next policy is published",
        {
            "train": {
                "max_training_steps": 4,
                "max_staleness": 2,
                "max_concurrent_rollouts": 8,
                "permit_release": "rollout",
            }
        },
        tasks=8,
        check=_check_permits_rollout,
    ),
    "start-limit": Scenario(
        "At most two rollouts starting at once (train.max_starting_rollouts: 2), with rollout permits for two groups "
        "of four under permit_release rollout at max_staleness 2: each member takes a start slot after its permit and "
        "gives it back at its rollout's first model call, so a group's members start in waves",
        {
            "train": {
                "max_training_steps": 4,
                "max_staleness": 2,
                "max_concurrent_rollouts": 8,
                "permit_release": "rollout",
                "max_starting_rollouts": 2,
            }
        },
        tasks=8,
        check=_check_start_limit,
    ),
    # ------------------------------------------------------------------- task selection
    "rule-informative": Scenario(
        "Group admission by the informative rule, whose rejected groups' places are refilled, under the pass_rate "
        "sampler, which scores its draws",
        {
            "train": {
                "group_admission": "informative",
                "refill": {"max_groups_per_step": 3},
                "task_sampler": "weighted",
            },
            "environments": [{**entry("pool", 6), "task_sampler": "pass_rate"}],
        },
        check=_check_informative,
    ),
    "rule-k1": Scenario(
        "The informative rule at max_staleness 1, closing steps short on its cap in their second microbatch",
        {
            "train": {
                "batch_size": 4,
                "num_microbatches": 2,
                "max_training_steps": 6,
                "max_staleness": 1,
                "max_concurrent_rollouts": 32,
                "group_admission": "informative",
                "refill": {"max_groups_per_step": 2.0},
                "workflow_config": {"group_size": 2},
            },
            "environments": [entry("quick", 12), {**entry("slow", 4), "max_staleness": 2}],
        },
        rollouts=Rollouts(invalid_one_in=4, slowdown={"slow::": 5.0}),
        check=_check_rule_short,
    ),
    "rule-k1-held": Scenario(
        "The informative rule at max_staleness 1 on one pool, whose failed groups hold their slots",
        {
            "train": {
                "batch_size": 4,
                "num_microbatches": 2,
                "max_training_steps": 6,
                "max_staleness": 1,
                "max_concurrent_rollouts": 32,
                "group_admission": "informative",
                "refill": {"max_groups_per_step": 2.0},
                "workflow_config": {"group_size": 2},
            },
            "environments": [entry("pool", 8)],
        },
        rollouts=Rollouts(invalid_one_in=4),
        check=_check_rule_held,
    ),
    "rule-k1-stale": Scenario(
        "The informative rule at max_staleness 1 with a small refill (1.5), where stale groups count toward the cap",
        {
            "train": {
                "batch_size": 4,
                "num_microbatches": 2,
                "max_training_steps": 6,
                "max_staleness": 1,
                "max_concurrent_rollouts": 32,
                "group_admission": "informative",
                "refill": {"max_groups_per_step": 1.5},
                "workflow_config": {"group_size": 2},
            },
            "environments": [entry("quick", 12), {**entry("slow", 4), "max_staleness": 2}],
        },
        rollouts=Rollouts(invalid_one_in=4, slowdown={"slow::": 5.0}),
        check=_check_rule_stale,
    ),
    "pass-rate": Scenario(
        "The pass_rate task sampler on the run's one entry, under the weighted run sampler",
        {
            "train": {"max_training_steps": 3, "task_sampler": "weighted"},
            "environments": [{**entry("pool", 6), "task_sampler": "pass_rate"}],
        },
        check=_check_pass_rate,
    ),
    # ------------------------------------------------------------------- evaluation
    "eval-at-start": Scenario(
        "Evaluation every second step beside training, and of the starting policy (eval.at_start)",
        {
            "train": {"max_training_steps": 4},
            "eval": {"every": 2, "at_start": True, "max_concurrent_rollouts": 4, "workflow_config": {"group_size": 2}},
        },
        eval_tasks=3,
        check=_check_eval,
    ),
    "eval-long": Scenario(
        "An evaluation of the starting policy that outlasts the first step, one evaluation group at a time",
        {
            "train": {"max_concurrent_rollouts": 4, "workflow_config": {"group_size": 2}},
            "eval": {"every": 1, "at_start": True, "max_concurrent_rollouts": 2, "workflow_config": {"group_size": 2}},
        },
        eval_tasks=6,
        check=_check_eval_long,
    ),
    # ------------------------------------------------------------------- checkpoints, resumes and stops
    "cadence": Scenario(
        "A permanent checkpoint every 2 steps and a recovery checkpoint every step, both due at once, over 6 steps",
        {
            "train": {"max_training_steps": 6, "workflow_config": {"group_size": 2}},
            "checkpoint": {"every": 2, "recovery_every": 1},
        },
        check=_check_cadence,
    ),
    "resume-kept": Scenario(
        "A crash after a recovery checkpoint that kept settled groups, then a resume from it (max_staleness 1)",
        {
            "train": {"max_training_steps": 3, "max_staleness": 1, "max_concurrent_rollouts": 16},
            "checkpoint": {"recovery_every": 1},
        },
        tasks=8,
        starts=(Start(crash_after_saves=2), Start()),
        check=_check_resume,
    ),
    "eval-resume": Scenario(
        "Evaluations every 3 steps, a crash after the permanent save at step 3, and a resume that skips the "
        "evaluation done",
        {
            "train": {"max_training_steps": 6, "max_concurrent_rollouts": 2, "workflow_config": {"group_size": 2}},
            "checkpoint": {"every": 3, "recovery_every": 2},
            "eval": {"every": 3, "max_concurrent_rollouts": 4, "workflow_config": {"group_size": 2}},
        },
        eval_tasks=2,
        starts=(Start(crash_after_saves=2), Start()),
        check=_check_eval_resume,
    ),
    "stop-mid-step": Scenario(
        "A stop request at the first update of a step of two minibatches, which finishes the step, then a resume",
        {
            "train": {"batch_size": 4, "num_minibatches": 2, "max_training_steps": 4, "max_concurrent_rollouts": 8},
            "environments": [entry("pool", 8, rate=0.5)],
        },
        starts=(Start(stop_at=("optim_step", 1)), Start()),
        check=_check_stop_mid_step,
    ),
    "stop-before-update": Scenario(
        "A stop request after a step's first microbatch was weighted by depth, before its update, then a resume",
        {
            "train": {
                "batch_size": 4,
                "num_microbatches": 2,
                "max_training_steps": 4,
                "max_concurrent_rollouts": 8,
                "workflow_config": {"depth_level_weighting": True},
            },
            "environments": [entry("pool", 8, rate=0.5)],
        },
        rollouts=Rollouts(subagents=True),
        starts=(Start(stop_at=("forward_backward", 3)), Start()),
        check=_check_stop,
    ),
}
