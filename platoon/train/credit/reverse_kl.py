"""``credit: reverse_kl``: on-policy distillation, each sampled token's advantage the negative reverse KL between the
student that sampled it and a frozen teacher that scores it.

For every row of the group, the teacher (``credit_options.teacher``, a frozen policy of ``policies:``) scores the
student's sampled tokens (``PolicyView.score``), and each action token's advantage becomes

    -kl_coef * (student_logprob - teacher_logprob) + reward_coef * (reward - baseline)

where the student's logprob is the one it sampled the token with, and the reward term, off by default, is the row's
trajectory reward centered on its member's group baseline (the workflow's own baseline: the completed roots' mean,
or leave-one-out under ``leave_one_out_baseline``).  With ``reward_coef: 1`` this is tinker-cookbook's on-policy
distillation advantage, the centered reward plus the per-token negative reverse KL, without the cookbook's discounted
KL (``kl_discount_factor``), which is not built.  The teacher's calls run in the trainer, after the group's members
returned their rollout permits, so they hold no rollout slot; an evaluation runs no estimator, so it never calls the
teacher.  Each call is bounded by the teacher's ``sample_timeout_seconds``, so ``assign`` has no limit of its own: a
call that fails or runs out of time, or scores that are not one per target token of each row, fail the group alone
(``ScoringFailed``), which then trains nothing.

The teacher reads the student's token ids, so it must share the student's tokenizer (cross-tokenizer distillation is
not built): loading refuses one that does not, comparing the two models' tokenizers by model name, and the estimator,
as the run builds it, compares the tokenizers the run loaded for the two policies, so that a teacher whose tokenizer
could not be loaded when the config loaded is still refused.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from platoon.train.credit.api import (
    CreditCheckContext,
    CreditEstimatorBase,
    CreditGroup,
    CreditSetup,
    ScoringFailed,
)
from platoon.train.credit.group import group_baselines
from platoon.train.policy.spec import MAIN, trained_policies
from platoon.utils.config import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class ReverseKLOptions:
    """``credit_options`` under ``credit: reverse_kl``."""

    # The frozen policy that scores the student's tokens: the id of a policies: entry with trainable: false.
    teacher: str
    # The weight of the reverse KL term in each token's advantage.
    kl_coef: float = 1.0
    # The weight of the centered reward term (0: distillation alone).
    reward_coef: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.teacher, str) or not self.teacher:
            raise ValueError(f"teacher: the id of a frozen policy of policies:, got {self.teacher!r}")
        for name in ("kl_coef", "reward_coef"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: a number, got {value!r}")


class ReverseKL(CreditEstimatorBase):
    """On-policy distillation from a frozen teacher (the module's docstring)."""

    needs_agents = False
    options_schema = ReverseKLOptions
    options: ReverseKLOptions
    # Each scoring call is bounded by the teacher's sample_timeout_seconds, and one that fails fails its group.
    assign_timeout_seconds = None

    def __init__(self, setup: CreditSetup, options: Any = None) -> None:
        """Refuse (:class:`~platoon.utils.config.ConfigError`, exit 78) a teacher whose tokenizer, as the run loaded it
        for the policy, differs from the student's: loading compares them by model name, but leaves them uncompared
        when one does not load there (:func:`tokenizer_problem`).  A setup without both tokenizers (a test's) compares
        nothing."""

        super().__init__(setup, options)
        teacher = self.options.teacher
        student_view, teacher_view = setup.policy_views.get(MAIN), setup.policy_views.get(teacher)
        student, scorer = (getattr(view, "tokenizer", None) for view in (student_view, teacher_view))
        if student is not None and scorer is not None and not same_tokenizer(student, scorer):
            raise ConfigError(
                f"{setup.entry}: credit_options.teacher: policies.{teacher}'s tokenizer, as the run loaded it, differs "
                "from the student's, so it cannot score the student's token ids (cross-tokenizer distillation is not "
                "built): use a teacher of the student's model family"
            )

    @staticmethod
    def check(options: ReverseKLOptions, ctx: CreditCheckContext) -> None:
        """The teacher is a frozen policy the config declares, not the student (``main``, the policy the run
        trains), and has the student's tokenizer (:func:`tokenizer_problem`), and each other trained policy's, whose
        rows it scores too."""

        if options.teacher == MAIN:
            raise ValueError("teacher: main is the student, the policy the run trains; name a frozen policy")
        declared = getattr(ctx.config, "policies", None) or {}
        if options.teacher not in declared:
            known = ", ".join(sorted(declared)) or "none"
            raise ValueError(f"teacher: {options.teacher!r} is not a policy of policies: (declared: {known})")
        teacher = declared[options.teacher]
        if getattr(teacher, "trainable", True):
            raise ValueError(f"teacher: policies.{options.teacher} trains; a teacher is frozen (trainable: false)")
        if problem := tokenizer_problem(ctx.config.train.model_name, teacher.model_name):
            raise ValueError(f"teacher: policies.{options.teacher}'s {problem}")
        for student in trained_policies(ctx.config)[1:]:  # the run's other students: it scores their rows too
            if problem := tokenizer_problem(declared[student].model_name, teacher.model_name):
                raise ValueError(f"teacher: policies.{options.teacher}'s {problem} (policies.{student}, trained too)")

    async def assign(self, group: CreditGroup) -> None:
        samples = group.samples()
        if not samples:
            return
        teacher = self.setup.policy_views[self.options.teacher]
        scores = await teacher.score([sample.datum for sample in samples])  # ScoringFailed fails the group
        sampled = [sample.datum.loss_fn_inputs["logprobs"].to_torch() for sample in samples]
        _check_aligned(teacher.id, scores, sampled)
        baselines = self._baselines(group)
        kl_sum, kl_tokens = 0.0, 0
        for sample, teacher_logprobs, student_logprobs in zip(samples, scores, sampled, strict=True):
            reverse_kl = student_logprobs - teacher_logprobs
            advantages = -self.options.kl_coef * reverse_kl
            if self.options.reward_coef:
                advantages = advantages + self.options.reward_coef * (sample.advantages - baselines[sample.member])
            sample.set_advantages(advantages)
            actions = sample.mask > 0
            kl_sum += float(reverse_kl[actions].sum())
            kl_tokens += int(actions.sum())
        if kl_tokens:
            group.stats.scalar(**{"credit/reverse_kl/avg": kl_sum / kl_tokens})

    @staticmethod
    def _baselines(group: CreditGroup) -> dict[int, float]:
        """Each member's baseline, by its index: the workflow's own (the completed roots' mean, or leave-one-out)."""

        members = group.members
        baselines = group_baselines(
            [member.task_reward for member in members],
            [member.task_reward_valid for member in members],
            leave_one_out=group.settings.leave_one_out_baseline,
        )
        return {member.index: baseline for member, baseline in zip(members, baselines, strict=True)}


def _check_aligned(teacher: str, scores: Sequence[Any], sampled: Sequence[Any]) -> None:
    """Refuse the teacher's scores unless they are one per target token of each row, as the row's sampled logprobs are:
    :class:`~platoon.train.credit.api.ScoringFailed`, which fails the group, before any row's advantages are set."""

    if len(scores) != len(sampled):
        raise ScoringFailed(f"policy {teacher} gave scores of {len(scores)} row(s) for {len(sampled)}")
    for index, (score, logprobs) in enumerate(zip(scores, sampled, strict=True)):
        if tuple(score.shape) != tuple(logprobs.shape):
            raise ScoringFailed(
                f"policy {teacher}'s scores of row {index} have shape {tuple(score.shape)}, and its sampled logprobs "
                f"{tuple(logprobs.shape)}: one score per target token is needed"
            )


def tokenizer_problem(student_model: str, teacher_model: str) -> str | None:
    """Why a teacher of ``teacher_model`` cannot score the token ids of a student of ``student_model``, or None when it
    can: their tokenizers (tinker-cookbook's ``get_tokenizer``, which a run samples them with) must map the same tokens
    to the same ids.  When either tokenizer does not load here (no network, say), nothing is compared, with a
    warning: the estimator compares the tokenizers the run loads for the two policies as it is built
    (:class:`ReverseKL`)."""

    if student_model == teacher_model:
        return None
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    try:
        student, teacher = get_tokenizer(student_model), get_tokenizer(teacher_model)
    except Exception as exc:  # noqa: BLE001 - any failure to load leaves the check undone, said so
        logger.warning(
            "reverse_kl: could not load the tokenizers of %s and %s to compare them (%s: %s), so the teacher's is not "
            "checked against the student's",
            student_model,
            teacher_model,
            type(exc).__name__,
            exc,
        )
        return None
    if same_tokenizer(student, teacher):
        return None
    return (
        f"tokenizer ({teacher_model}) differs from the student's ({student_model}), so it cannot score the student's "
        "token ids (cross-tokenizer distillation is not built): use a teacher of the student's model family"
    )


def same_tokenizer(first: Any, second: Any) -> bool:
    """Whether two tokenizers map the same tokens to the same ids."""

    return first is second or _vocabulary(first) == _vocabulary(second)


def _vocabulary(tokenizer: Any) -> Sequence[Any]:
    return sorted(tokenizer.get_vocab().items())
