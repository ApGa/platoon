"""The credit of a run that trains several policies in lockstep (RFC 0021, D5): an entry without ``credit:`` credits
with ``keyed`` on ``[task, policy, role]``, which a workflow that credits with hooks of its own cannot apply; a
reverse-KL teacher must share the tokenizer of every policy the run trains, whose rows it scores; and an evaluation's
group, which trains nothing, runs without the estimator that a training group needs, also when the policies step
independently."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes.launch import built_workflows
from fakes.members import member_outcome

from platoon.task import Task
from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.train.credit.registry import choose_credit, workflow_problems
from platoon.train.credit.reverse_kl import ReverseKL, ReverseKLOptions
from platoon.train.policy.spec import MAIN
from platoon.train.workflow.output import GroupOutput
from platoon.train.workflow.protocol import report_of
from platoon.utils.config import build_config


def _credit_config(policies: dict[str, Any]) -> Any:
    entry = SimpleNamespace(label="vault", credit=None, credit_options={})
    return SimpleNamespace(environments=[entry], policies=policies, agents=None, train=SimpleNamespace())


def test_with_several_trained_policies_an_entry_without_credit_credits_by_policy_and_role():
    [choice] = choose_credit(_credit_config({"worker": SimpleNamespace(trainable=True)})).values()
    assert (choice.entry, choice.name, choice.default) == ("vault", "keyed", True)
    assert choice.options.key == ["task", "policy", "role"]
    assert choose_credit(_credit_config({"judge": SimpleNamespace(trainable=False)})) == {}

    class OwnHooks:
        applies_credit = False

    [problem] = workflow_problems({"vault": choice}, OwnHooks, "acme:Own")
    assert problem.startswith("environments[vault].credit: the run trains several policies, so its groups are credited")


@pytest.fixture
def tokenizers():
    """Three models' tokenizers as tinker-cookbook loads them by name: the student's and the teacher's are one, the
    worker's has one token more."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer, unregister_tokenizer
    from transformers import PreTrainedTokenizerFast

    from platoon.testing.tokenizer import EOS_TOKEN, TOY_TOKENIZER_DIR, toy_tokenizer

    other: Any = PreTrainedTokenizerFast(tokenizer_file=str(TOY_TOKENIZER_DIR / "tokenizer.json"), eos_token=EOS_TOKEN)
    other.add_tokens(["lockstep"])
    names = {"student-model": toy_tokenizer, "teacher-model": toy_tokenizer, "worker-model": lambda: other}
    for name, load in names.items():
        register_tokenizer(name, load)
    yield
    for name in names:
        unregister_tokenizer(name)


def test_a_teacher_must_share_the_tokenizer_of_every_policy_the_run_trains(tokenizers):
    teacher = SimpleNamespace(model_name="teacher-model", trainable=False)
    policies = {"teacher": teacher, "worker": SimpleNamespace(model_name="worker-model", trainable=True)}
    loaded = SimpleNamespace(train=SimpleNamespace(model_name="student-model"), policies=policies)
    context: Any = SimpleNamespace(config=loaded)
    with pytest.raises(ValueError, match=r"teacher: policies.teacher's tokenizer .* \(policies.worker, trained too\)"):
        ReverseKL.check(ReverseKLOptions(teacher="teacher"), context)
    loaded.policies = {"teacher": teacher}
    ReverseKL.check(ReverseKLOptions(teacher="teacher"), context)


# --- the workflows platoon train builds ------------------------------------------------------------------------------

HERE = __name__


def tasks(config: Any, split: str) -> list[dict[str, Any]]:
    return [{"task_id": f"t{index}"} for index in range(4)]


async def rollout(task: Any, rollout_config: Any) -> dict:
    raise AssertionError("not run")


def get_task(task_id: str) -> Task:
    return Task(id=task_id)


def _two_policies(tmp_path: Path, stepping: str) -> Any:
    """A run that trains main and its subagents' policy, worker, stepping as ``stepping`` says, and evaluates: one
    entry, ``a``, without ``credit:``, whose groups have two members."""

    groups = {"group_size": 2}
    return build_config(
        PlatoonTinkerRLTrainerConfig,
        {
            "train": {"model_name": "m", "renderer_name": "r", "stepping": stepping, "workflow_config": groups},
            "eval": {"every": 2, "workflow_config": groups},
            "policies": {"worker": {"model_name": "m", "renderer_name": "r"}},
            "agents": {"roles": {"subagent": {"policy": "worker"}}},
            "log_path": str(tmp_path / "logs"),
            "environments": [
                {
                    "name": "a",
                    "dataset_loader": f"{HERE}:tasks",
                    "rollout": f"{HERE}:rollout",
                    "task_loader": f"{HERE}:get_task",
                }
            ],
        },
    )


def _group(workflow: Any) -> Any:
    """The output of ``workflow``'s group of task t0, whose two members' roots completed with rewards 1 and 0."""

    async def member(data: dict, rollout_number: int) -> Any:
        return member_outcome(0, rollout_number, reward=1.0 - rollout_number, version=0)

    workflow.arun_episode_single = member
    return asyncio.run(workflow.arun_episode({"task_id": "t0"}))


@pytest.mark.parametrize("stepping", ["lockstep", "independent"])
def test_an_evaluation_group_runs_without_the_estimator_that_a_training_group_needs(tmp_path, stepping):
    """Training credits each group with an estimator, the one credit that tells each row's policy, and refuses a group
    without one; the evaluation has none, and its groups, which train nothing, run with the workflow's own credit."""

    train, evaluation = built_workflows(_two_policies(tmp_path, stepping))
    assert train.trained_policies == evaluation.trained_policies == {MAIN, "worker"}
    assert list(train.entry_credit) == ["a"] and dict(evaluation.entry_credit) == {}

    output = _group(evaluation)
    assert isinstance(output, GroupOutput) and report_of(output).verdict == "trained"

    evaluation.phase = "train"  # the same group in training
    with pytest.raises(RuntimeError, match=r"^a run that trains several policies credits every training group with an"):
        _group(evaluation)
