"""How a run's trained policies step (``train.stepping``, RFC 0021 section 2.11), as loading and the policy package read
it: the key and its values, a policy's own batch and checkpoint cadence and what loading refuses of them (an unknown key
of its cadence too, and a ``keep_recovery`` that its own cadence cannot keep) or warns of (a policy that saves no full
checkpoint before the run's end), the rl trainer's answer (``TrainerNeeds.stepping``), the stepping ``identity.json``
records and a resume that changes it. The shares, horizons and commits are tested beside those packages' own tests;
whole starts in ``trainer/test_independent_run.py``."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from platoon.train import identity
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.policy.spec import policy_specs, policy_warnings, steps_independently
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer
from platoon.utils.config import ConfigError, build_config, validate_config

SETTINGS: dict[str, Any] = {
    "train": {
        "model_name": "Qwen/Qwen3-4B-Instruct-2507",
        "renderer_name": "qwen3_instruct",
        "max_staleness": 1,
        "stepping": "independent",
    },
    "eval": {"every": 0},
    "log_path": "logs",
}
WORKER = {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"}
BOUND = {"agents": {"roles": {"subagent": {"policy": "worker"}}}}


def config(train: dict[str, Any] | None = None, **sections: Any) -> PlatoonTinkerRLTrainerConfig:
    data = {**copy.deepcopy(SETTINGS), **sections}
    data["train"] = {**data["train"], **(train or {})}
    return build_config(PlatoonTinkerRLTrainerConfig, data)


def problems(**sections: Any) -> list[str]:
    with pytest.raises(ConfigError) as raised:
        config(**sections)
    return raised.value.problems


def worker(**settings: Any) -> dict[str, Any]:
    """A trained worker policy bound to the subagents' role, with ``settings``."""

    return {"policies": {"worker": {**WORKER, **settings}}, **BOUND}


def test_the_policies_step_in_lockstep_unless_the_config_says_independent():
    data = copy.deepcopy(SETTINGS)
    del data["train"]["stepping"]
    assert build_config(PlatoonTinkerRLTrainerConfig, data).train.stepping == "lockstep"
    assert steps_independently(config(**worker()))
    # With one trained policy there is one clock: independent stepping is lockstep.
    assert not steps_independently(config())
    assert not steps_independently(config(train={"stepping": "lockstep"}, **worker()))
    with pytest.raises(ValueError, match="train.stepping must be 'lockstep' or 'independent', got 'leader'"):
        config(train={"stepping": "leader"})


def test_a_policy_s_own_batch_and_cadence_reach_its_spec():
    spec = policy_specs(config(**worker(batch_size=2, checkpoint={"every": 10, "recovery_every": 5})))["worker"]
    assert spec.batch_size == 2
    assert spec.checkpoint is not None and (spec.checkpoint.every, spec.checkpoint.recovery_every) == (10, 5)
    unset = policy_specs(config(**worker()))["worker"]
    assert unset.batch_size is None and unset.checkpoint is None


def test_keep_recovery_counts_each_policy_s_own_cadence_in_its_own_steps():
    """Each policy's pruner keeps its own bound at its own cadence, in its own steps (and so its own versions)."""

    # The worker commits every 3 of its steps, bound 7: keep 3 spans 9 of its versions, though 3 of the run's cadence
    # would not.
    own = worker(max_staleness=7, checkpoint={"recovery_every": 3})
    config(checkpoint={"recovery_every": 1, "keep_recovery": 3}, **own)
    # It commits every one of its steps, bound 3: keep 1 spans 1 of its versions, though 1 of the run's 4 would span 4.
    own = worker(max_staleness=3, checkpoint={"recovery_every": 1})
    found = problems(checkpoint={"recovery_every": 4, "keep_recovery": 1}, **own)
    assert len(found) == 1, found
    assert found[0].startswith(
        "checkpoint.keep_recovery: 1 × policies.worker.checkpoint.recovery_every: 1 = 1 must exceed "
        "policies.worker.max_staleness: 3: "
    ), found
    assert found[0].endswith(
        "raise checkpoint.keep_recovery to 4 or policies.worker.checkpoint.recovery_every to 4, or lower "
        "policies.worker.max_staleness"
    ), found
    # Without a bound of its own, its pruner keeps the run's largest (train.max_staleness: 1), at its own cadence.
    found = problems(checkpoint={"recovery_every": 2, "keep_recovery": 1}, **worker(checkpoint={"recovery_every": 1}))
    assert len(found) == 1 and found[0].startswith(
        "checkpoint.keep_recovery: 1 × policies.worker.checkpoint.recovery_every: 1 = 1 must exceed "
        "train.max_staleness: 1: "
    ), found
    # Stepping together, a policy has no cadence of its own: the run's counts, against its bound (as in lockstep).
    found = problems(
        train={"stepping": "lockstep"}, checkpoint={"recovery_every": 2, "keep_recovery": 1}, **worker(max_staleness=4)
    )
    assert any(
        problem.startswith("checkpoint.keep_recovery: 1 × checkpoint.recovery_every: 2 = 2 must exceed policies.worker")
        for problem in found
    ), found


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"batch_size": 0}, "policies.worker: batch_size must be a whole number, 1 or more, or null, got 0"),
        ({"checkpoint": {"recovery_every": 0}}, "checkpoint.recovery_every must be 1 or more, or null, got 0"),
        ({"checkpoint": {"every": -1}}, "checkpoint.every must be a number of the policy's steps, or 0 for never"),
    ],
)
def test_a_policy_s_own_batch_or_cadence_out_of_range_is_refused(settings, message):
    with pytest.raises(ValueError) as raised:
        config(**worker(**settings))
    assert message in str(raised.value)


def test_a_policy_s_own_batch_must_split_into_the_step_s_minibatches_and_microbatches():
    [problem] = problems(train={**SETTINGS["train"], "num_minibatches": 2}, **worker(batch_size=3))
    assert problem == (
        "policies.worker.batch_size: 3 does not split into train.num_minibatches × train.num_microbatches (2) parts: "
        "give a multiple of 2"
    )


def test_a_frozen_policy_s_own_batch_or_cadence_is_refused():
    judge = {**WORKER, "trainable": False, "batch_size": 4, "checkpoint": {"recovery_every": 2}}
    roles = {"judge": {"policy": "judge"}, "subagent": {"policy": "worker"}}
    found = problems(policies={"judge": judge, "worker": WORKER}, agents={"roles": roles})
    assert any("policies.judge: batch_size, checkpoint are for a policy the run trains" in problem for problem in found)


def test_a_group_admission_rule_is_refused_under_independent_stepping():
    found = problems(train={**SETTINGS["train"], "group_admission": "informative"}, **worker())
    assert any(
        problem.startswith("train.stepping: independent steps each policy on its own clock") for problem in found
    )
    config(train={**SETTINGS["train"], "group_admission": "informative", "stepping": "lockstep"}, **worker())


def test_the_rl_trainer_answers_with_train_stepping():
    assert PlatoonTinkerRLTrainer.needs(config(**worker()), None).stepping == "independent"
    assert PlatoonTinkerRLTrainer.needs(config(train={"stepping": "lockstep"}), None).stepping == "lockstep"
    assert PlatoonTinkerRLTrainer.needs(None, None).stepping == "lockstep"  # asked without a config


def test_a_policy_s_cadence_with_a_key_it_does_not_take_is_refused_by_name():
    data = {**copy.deepcopy(SETTINGS), **worker(checkpoint={"recovery": 2})}
    with pytest.raises((ConfigError, ValueError)) as raised:
        build_config(PlatoonTinkerRLTrainerConfig, data)
    assert "checkpoint takes every and recovery_every, not recovery" in str(raised.value)


def test_a_policy_that_saves_no_full_checkpoint_before_the_run_s_end_is_warned_of():
    recovers = {"checkpoint": {"recovery_every": 1}}
    lines = policy_warnings(config(**worker()))  # neither main nor the worker has a cadence
    assert [line.split(" saves")[0] for line in lines] == ["main", "policies.worker"]
    assert lines[1].endswith("set policies.worker.checkpoint.recovery_every")
    assert policy_warnings(config(**worker(checkpoint={"recovery_every": 2}), **recovers)) == []
    assert policy_warnings(config(train={"stepping": "lockstep"}, **worker())) == []  # lockstep keeps nothing


def test_identity_records_independent_stepping_and_a_resume_that_changes_it_is_refused():
    independent = identity.run_identity(config(**worker()), [])
    lockstep = identity.run_identity(config(train={"stepping": "lockstep"}, **worker()), [])
    assert independent["stepping"] == "independent" and "stepping" not in lockstep
    assert identity.policy_changes(independent, lockstep) == [
        "train.stepping: the run's is 'independent', and this start's 'lockstep': a resume keeps the stepping its run "
        "began with, whose checkpoints say what each policy trained"
    ]
    assert identity.policy_changes(lockstep, independent)[0].startswith("train.stepping: the run's is 'lockstep'")
    assert identity.policy_changes(independent, independent) == []
    # One trained policy steps on one clock: nothing recorded, so a config that says independent resumes a lockstep run.
    judged = config(
        policies={"judge": {**WORKER, "trainable": False}}, agents={"roles": {"judge": {"policy": "judge"}}}
    )
    assert "stepping" not in identity.run_identity(judged, [])


def test_a_policy_s_own_batch_is_refused_in_lockstep_and_validates_as_a_key():
    [problem] = problems(train={"stepping": "lockstep"}, **worker(batch_size=2))
    assert problem.startswith("policies.worker.batch_size: a policy's own batch size is for policies that step on")
    assert validate_config({**copy.deepcopy(SETTINGS), **worker(batch_size=2)}, PlatoonTinkerRLTrainerConfig) == []
