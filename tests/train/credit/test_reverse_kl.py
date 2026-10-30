"""``credit: reverse_kl`` (on-policy distillation): each action token's advantage is the negative reverse KL between the
student's sampled logprob and a frozen teacher's score of the same token, plus an optional centered reward term; a
teacher whose scoring fails, or whose scores are not one per target token, fails the group alone (``ScoringFailed``,
which ``assign_credit`` lets through); what loading refuses of its teacher (an undeclared policy, a trained one, the
student itself, another tokenizer, compared by model name) and what building it refuses (another tokenizer, as the run
loaded it); and that it is core's, resolved by name."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from platoon.testing import toy_tokenizer
from platoon.testing.credit import fixture_group
from platoon.train.config_defs import WorkflowConfig
from platoon.train.credit.api import CreditCheckContext, CreditSetup, PolicyView, ScoringFailed
from platoon.train.credit.group import group_baselines
from platoon.train.credit.registry import RESERVED_CREDIT_NAMES, resolve_credit_estimator
from platoon.train.credit.reverse_kl import ReverseKL, ReverseKLOptions, same_tokenizer
from platoon.train.credit.runtime import assign_credit
from platoon.train.policy.spec import MAIN
from platoon.utils.config import ConfigError
from platoon.utils.stats_tracker import StatsTracker

# The fixture rows' sampled logprob (platoon.testing.credit), and the score the fake teacher gives every token.
STUDENT, TEACHER = -0.5, -1.25


class _Teacher:
    def __init__(self) -> None:
        self.scored: list[Any] = []

    async def __call__(self, datums: Sequence[Any]) -> list[torch.Tensor]:
        self.scored.extend(datums)
        return [torch.full((len(datum.loss_fn_inputs["target_tokens"].to_torch()),), TEACHER) for datum in datums]


def _estimator(teacher: Any, **options: Any) -> ReverseKL:
    views = {MAIN: PolicyView(MAIN), "teacher": PolicyView("teacher", scorer=teacher)}
    setup = CreditSetup("reverse_kl", "default", WorkflowConfig(), (MAIN, "teacher"), policy_views=views)
    return ReverseKL(setup, ReverseKLOptions(teacher="teacher", **options))


def _actions(sample: Any) -> list[float]:
    return sample.advantages[sample.mask > 0].tolist()


def _observations_between_actions(sample: Any) -> int:
    positions = (sample.mask > 0).nonzero().flatten().tolist()
    return int((sample.mask[positions[0] : positions[-1]] == 0).sum()) if positions else 0


@pytest.mark.parametrize("kl_coef", [1.0, 0.5])
def test_each_action_token_s_advantage_is_the_negative_reverse_kl(kl_coef):
    group, teacher = fixture_group(1, tagged=False), _Teacher()
    asyncio.run(_estimator(teacher, kl_coef=kl_coef).assign(group))

    samples = group.samples()
    assert teacher.scored == [sample.datum for sample in samples]  # every row, once
    assert any(_observations_between_actions(sample) for sample in samples)  # which keep 0, as every observation
    for sample in samples:
        assert _actions(sample) == pytest.approx([-kl_coef * (STUDENT - TEACHER)] * int((sample.mask > 0).sum()))
        assert sample.advantages[sample.mask == 0].tolist() == [0.0] * int((sample.mask == 0).sum())
    tracker = StatsTracker()
    group.stats.record(tracker)
    assert tracker.export()["credit/reverse_kl/avg"] == pytest.approx(STUDENT - TEACHER)


def test_the_reward_term_adds_each_row_s_reward_centered_on_its_member_s_baseline():
    group, plain = fixture_group(2, tagged=False), fixture_group(2, tagged=False)
    rewards = {id(sample): _actions(sample) for sample in plain.samples()}  # as the converter loaded them
    asyncio.run(_estimator(_Teacher(), kl_coef=1.0, reward_coef=0.5).assign(group))

    members = group.members
    rewards_of, valid = [member.task_reward for member in members], [member.task_reward_valid for member in members]
    baselines = group_baselines(rewards_of, valid, leave_one_out=False)
    for member, baseline, plain_member in zip(members, baselines, plain.members, strict=True):
        for sample, original in zip(member.samples, plain_member.samples, strict=True):
            expected = [-(STUDENT - TEACHER) + 0.5 * (reward - baseline) for reward in rewards[id(original)]]
            assert _actions(sample) == pytest.approx(expected)


def test_a_group_without_rows_scores_nothing():
    group, teacher = fixture_group(0, tagged=False), _Teacher()
    for member in group.members:
        member.samples.clear()
    asyncio.run(_estimator(teacher).assign(group))
    assert teacher.scored == []


def test_a_teacher_whose_scoring_fails_fails_the_group_alone():
    async def unavailable(datums: Sequence[Any]) -> list[torch.Tensor]:
        raise RuntimeError("503 Service Unavailable")

    group, estimator = fixture_group(1, tagged=False), _estimator(unavailable)
    rows = len(group.samples())
    with pytest.raises(ScoringFailed, match=rf"^policy teacher could not score {rows} row\(s\): RuntimeError: 503 "):
        asyncio.run(estimator.assign(group))
    # The run's call lets it through, so that the workflow fails this group and the run goes on.
    with pytest.raises(ScoringFailed):
        asyncio.run(assign_credit(estimator, fixture_group(1, tagged=False)))
    # Each scoring call has its own bound (the teacher's sample_timeout_seconds): assign has none of its own.
    assert estimator.assign_timeout_seconds is None


@pytest.mark.parametrize("kind", ["one entry", "one short", "one long", "a row fewer"])
def test_scores_that_are_not_one_per_target_token_fail_the_group_and_set_no_advantage(kind):
    async def malformed(datums: Sequence[Any]) -> list[torch.Tensor]:
        scores = []
        for datum in datums:
            targets = len(datum.loss_fn_inputs["target_tokens"].to_torch())
            length = {"one entry": 1, "one short": targets - 1, "one long": targets + 1}.get(kind, targets)
            scores.append(torch.full((length,), TEACHER))
        return scores[:-1] if kind == "a row fewer" else scores

    group = fixture_group(1, tagged=False)
    before = [sample.advantages.clone() for sample in group.samples()]
    message = (
        r"^policy teacher gave scores of \d+ row\(s\) for \d+$"
        if kind == "a row fewer"
        else r"^policy teacher's scores of row 0 have shape \(\d+,\), and its sampled logprobs \(\d+,\): one score "
        r"per target token is needed$"
    )
    with pytest.raises(ScoringFailed, match=message):  # through the run's call: the group fails, the run goes on
        asyncio.run(assign_credit(_estimator(malformed), group))
    assert all(torch.equal(sample.advantages, old) for sample, old in zip(group.samples(), before, strict=True))


STUDENT_MODEL = "acme/student-4b"


def _frozen(model_name: str = "acme/teacher-32b") -> Any:
    return SimpleNamespace(trainable=False, model_name=model_name)


def _context(policies: dict[str, Any]) -> CreditCheckContext:
    config = SimpleNamespace(train=SimpleNamespace(model_name=STUDENT_MODEL), policies=policies)
    return CreditCheckContext(entry="default", policies=(MAIN, *policies), config=config)


@pytest.fixture
def tokenizers() -> Iterator[None]:
    """Tokenizers by model name, as tinker-cookbook loads them: the student's and its family's teacher share the toy
    tokenizer, another family's teacher has one token more, and one model's does not load here."""

    from tinker_cookbook.tokenizer_utils import register_tokenizer, unregister_tokenizer

    other = _other_tokenizer()

    def unloadable() -> Any:
        raise OSError("no network")

    factories = {
        STUDENT_MODEL: toy_tokenizer,
        "acme/teacher-32b": toy_tokenizer,
        "other/teacher": lambda: other,
        "acme/offline": unloadable,
    }
    for name, factory in factories.items():
        register_tokenizer(name, factory)
    yield
    for name in factories:
        unregister_tokenizer(name)


@pytest.mark.parametrize(
    ("teacher", "policies", "message"),
    [
        ("main", {}, "teacher: main is the student"),
        ("teacher", {}, "teacher: 'teacher' is not a policy of policies: (declared: none)"),
        ("teacher", {"teacher": SimpleNamespace(trainable=True)}, "teacher: policies.teacher trains"),
    ],
)
def test_loading_refuses_a_teacher_that_is_not_a_frozen_policy(teacher, policies, message, tokenizers):
    with pytest.raises(ValueError, match="^" + message.replace("(", r"\(").replace(")", r"\)")):
        ReverseKL.check(ReverseKLOptions(teacher=teacher), _context(policies))
    ReverseKL.check(ReverseKLOptions(teacher="t"), _context({"t": _frozen()}))


def _other_tokenizer() -> Any:
    from transformers import PreTrainedTokenizerFast

    from platoon.testing.tokenizer import EOS_TOKEN, TOY_TOKENIZER_DIR

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOY_TOKENIZER_DIR / "tokenizer.json"), eos_token=EOS_TOKEN)
    tokenizer.add_tokens(["distill"])
    return tokenizer


def test_loading_refuses_a_teacher_of_another_tokenizer_compared_by_model_name(tokenizers, caplog):
    toy, other = toy_tokenizer(), _other_tokenizer()
    assert same_tokenizer(toy, toy) and not same_tokenizer(toy, other)

    def check(teacher_model: str) -> None:
        ReverseKL.check(ReverseKLOptions(teacher="teacher"), _context({"teacher": _frozen(teacher_model)}))

    with pytest.raises(
        ValueError,
        match=r"^teacher: policies.teacher's tokenizer \(other/teacher\) differs from the student's "
        r"\(acme/student-4b\), so it cannot score the student's token ids",
    ):
        check("other/teacher")
    check("acme/teacher-32b")  # the student's family: the same tokenizer
    check(STUDENT_MODEL)  # the student's own model
    with caplog.at_level(logging.WARNING, logger="platoon.train.credit.reverse_kl"):
        check("acme/offline")  # a tokenizer that does not load here: not compared, with a warning
    assert "could not load the tokenizers of acme/student-4b and acme/offline to compare them" in caplog.text


def test_building_refuses_a_teacher_whose_tokenizer_as_the_run_loaded_it_differs():
    def build(student: Any, teacher: Any) -> ReverseKL:
        views = {MAIN: PolicyView(MAIN, tokenizer=student), "teacher": PolicyView("teacher", tokenizer=teacher)}
        setup = CreditSetup("reverse_kl", "guess", WorkflowConfig(), (MAIN, "teacher"), policy_views=views)
        return ReverseKL(setup, ReverseKLOptions(teacher="teacher"))

    with pytest.raises(ConfigError) as raised:  # one loading could not compare (a tokenizer did not load there)
        build(toy_tokenizer(), _other_tokenizer())
    assert raised.value.problems == [
        "guess: credit_options.teacher: policies.teacher's tokenizer, as the run loaded it, differs from the "
        "student's, so it cannot score the student's token ids (cross-tokenizer distillation is not built): use a "
        "teacher of the student's model family"
    ]
    build(toy_tokenizer(), toy_tokenizer())  # the same tokenizer, loaded for each policy
    build(toy_tokenizer(), None)  # a setup without both tokenizers compares nothing


def test_reverse_kl_is_core_s_estimator():
    assert resolve_credit_estimator("reverse_kl") is ReverseKL
    assert {"group", "keyed", "reverse_kl", "gae"} <= RESERVED_CREDIT_NAMES
    with pytest.raises(ValueError, match="teacher: the id of a frozen policy"):
        ReverseKLOptions(teacher="")
