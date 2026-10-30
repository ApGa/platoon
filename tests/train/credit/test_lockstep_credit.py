"""The credit of a run that trains several policies in lockstep (RFC 0021, D5): an entry without ``credit:`` credits
with ``keyed`` on ``[task, policy, role]``, which a workflow that credits with hooks of its own cannot apply; and a
reverse-KL teacher must share the tokenizer of every policy the run trains, whose rows it scores."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from platoon.train.credit.registry import choose_credit, workflow_problems
from platoon.train.credit.reverse_kl import ReverseKL, ReverseKLOptions


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
