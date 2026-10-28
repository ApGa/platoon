"""Credit over a task group, which ``GroupRolloutWorkflow._run_group`` computes once the group's members have run, in
this order:

1. The converter (``platoon.utils.tinker_data_processing``) loads every action
   token of a trajectory with that trajectory's processed reward. Subagent
   trajectories carry their own reward (their env grade, their verifier's
   verdict, or the root's success with ``propagate_root_success``); datums of
   policy-ineligible children and Bernoulli-sampled-out child datums
   (``subagent_datum_keep_probability``) are already gone.
2. ``compute_baselines`` returns one baseline per usable rollout from the
   group's completed root rewards: the group mean, or the leave-one-out mean
   with ``leave_one_out_baseline``. Interrupted roots keep their statistics but
   never enter a baseline.
3. ``assign_advantages`` broadcasts each rollout's baseline over all of its
   datums, root and subagents alike, by subtracting it on action tokens.
4. Error tokens with a positive advantage are suppressed (``filter_errors``)
   and datums left without action tokens are dropped (:func:`filter_error_tokens`).
   Then :func:`group_verdict` says what the group trains: nothing when no datum
   kept an action token, else its datums (``uniform`` when all their advantages
   are zero).
5. Per microbatch, the trainer applies the depth transforms of
   ``platoon.train.step.batch_transforms`` and the zero-advantage filter.

A group whose members trained different reward keys does not train
(``reward_key_mismatch``): a reward contract may list the names a reward can
train under (``reward: [reward, correctness]``), and centering one name's
rewards on another's would let code of the policy's that the tests run choose
the name, and so the scale, a group compares on.  A member that trained no
key, such as one platoon scored 0 itself, takes no part in the check.

A group trains only when at least ``min_successful_group_size`` of its members
are *trainable*: their root completed with a scored outcome
(``TrajectoryCollectionResult.task_reward_valid``).  A member that ended
untrainable (an ``invalid``, ``error``, ``timeout`` or ``cancelled`` root, even
one that recorded model calls) counts as missing, like one that returned no
result (a setup failure, a rollout that made no model call, a cancelled
straggler, a worker that died).  With ``group_size: 8`` and
``min_successful_group_size: 4``, a group trains when at least four of its
members are trainable; with the default of 1, when one root completed.

``GroupRolloutWorkflow._untrainable_verdict`` checks these two rules, before step 2.

An environment entry may credit its groups with an estimator of its own instead of steps 2 and 3
(``environments[].credit``, :mod:`platoon.train.credit.api`); :class:`GroupCredit` is this module's, ``credit:
group``, which centers the rows as the workflow does.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import tinker
import torch
from tinker import TensorData

from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY

if TYPE_CHECKING:
    from platoon.train.config_defs import WorkflowConfig
    from platoon.train.credit.api import CreditGroup, CreditSetup
    from platoon.utils.tinker_data_processing import TrajectoryCollectionResult

logger = logging.getLogger(__name__)


def group_baselines(
    task_rewards: Sequence[float],
    task_rewards_valid: Sequence[bool],
    *,
    leave_one_out: bool,
) -> list[float]:
    """Return one advantage baseline per rollout from a group's root rewards.

    Only completed roots (``task_rewards_valid``) enter a baseline: an
    interrupted root's partial reward stays metric-only, while its completed
    children still train against the completed-root mean. With
    ``leave_one_out`` a completed root is centered on the mean of the *other*
    completed roots; a singleton completed root has no peer and is centered on
    itself (zero advantage).
    """

    completed = [reward for reward, valid in zip(task_rewards, task_rewards_valid, strict=True) if valid]
    if not completed:
        raise ValueError("a group baseline needs at least one completed root reward")
    total = sum(completed)
    mean = total / len(completed)
    if not leave_one_out:
        return [mean] * len(task_rewards)
    if len(completed) == 1:
        return [reward if valid else mean for reward, valid in zip(task_rewards, task_rewards_valid)]
    return [
        (total - reward) / (len(completed) - 1) if valid else mean
        for reward, valid in zip(task_rewards, task_rewards_valid)
    ]


def _filter_positive_centered_error_tokens(
    datum: tinker.Datum,
) -> tuple[bool, dict[str, float]]:
    """Consume the error side channel and mask only positive-advantage errors."""

    encoded_error_mask = datum.loss_fn_inputs.pop(ERROR_ACTION_MASK_KEY, None)
    if encoded_error_mask is None:
        # Agent-message-only datums have no policy tokens even when error
        # filtering is disabled. Drop them after their rewards were centered.
        return bool(datum.loss_fn_inputs["mask"].to_torch().any()), {}
    error_mask = encoded_error_mask.to_torch().bool()
    action_mask = datum.loss_fn_inputs["mask"].to_torch()
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    if error_mask.shape != action_mask.shape or advantages.shape != action_mask.shape:
        raise ValueError(
            "Deferred Tinker error filtering requires aligned error/mask/advantage tensors: "
            f"error={tuple(error_mask.shape)}, mask={tuple(action_mask.shape)}, "
            f"advantages={tuple(advantages.shape)}"
        )

    action_tokens = action_mask > 0
    error_actions = error_mask & action_tokens
    suppressed = error_actions & (advantages > 0)
    filtered_mask = torch.where(suppressed, torch.zeros_like(action_mask), action_mask)
    datum.loss_fn_inputs["mask"] = TensorData.from_torch(filtered_mask)
    # The trainer strips mask before protocol dispatch; CISPO and the other
    # policy losses use advantages to suppress these tokens on the server.
    datum.loss_fn_inputs["advantages"] = TensorData.from_torch(
        torch.where(suppressed, torch.zeros_like(advantages), advantages)
    )
    keep = bool(torch.any(filtered_mask > 0))
    return keep, {
        "detected_action_tokens": float(error_actions.sum().item()),
        "suppressed_positive_action_tokens": float(suppressed.sum().item()),
        "retained_nonpositive_action_tokens": float((error_actions & ~suppressed).sum().item()),
        "emptied_datums": float(not keep and bool(action_tokens.any())),
    }


def filter_error_tokens(
    results: Sequence["TrajectoryCollectionResult"],
) -> tuple[list[tinker.Datum], dict[str, float], list[int]]:
    """Step 4, once the group's advantages are assigned: run each member's datums through
    :func:`_filter_positive_centered_error_tokens`, keeping those it keeps in the member's ``datums`` (in place).
    Returns the datums kept, in member order; the filter's totals (``error_filter/*``), empty when no datum carried
    the error side channel; and the indices of the datums kept among the members' datums, in member order, as they were
    before the filter.  Without ``filter_errors`` no datum carries the side channel, and this only drops the datums left
    without an action token (agent-message-only ones)."""

    error_filter_totals: dict[str, float] = {}
    kept: list[int] = []
    index = 0
    for result in results:
        retained_datums: list[tinker.Datum] = []
        for datum in result.datums:
            keep, metrics = _filter_positive_centered_error_tokens(datum)
            for key, value in metrics.items():  # none for a datum without the error side channel
                error_filter_totals[key] = error_filter_totals.get(key, 0.0) + value
            if keep:
                retained_datums.append(datum)
                kept.append(index)
            index += 1
        result.datums[:] = retained_datums
    totals = {f"error_filter/{key}": value for key, value in error_filter_totals.items()}
    return [datum for result in results for datum in result.datums], totals, kept


def group_verdict(
    datums: Sequence[tinker.Datum],
    results: Sequence["TrajectoryCollectionResult"],
    completed_rewards: Sequence[float],
    config: "WorkflowConfig",
    data: dict,
) -> tuple[str, int]:
    """What a group trains once its error tokens are filtered (``datums``, its members' ``results``), with the number of
    its members that train (``trainable_rollouts``): ``no_datums`` when none of its datums has a policy action token
    left; else ``trained`` when some member's action tokens carry an advantage other than zero, or ``uniform`` when
    every one is zero.  A uniform group's datums still go to the trainer, whose zero-advantage filter drops them after
    the transforms (step 5).  ``datums`` are the ``results``' datums (:func:`filter_error_tokens`).  Only the DEBUG
    lines read the task (``data``, for its ``task_id``) and ``completed_rewards``: the completed roots' rewards from
    before the credit, at least one (``GroupRolloutWorkflow._untrainable_verdict`` checked)."""

    # Full-tree rollout statistics intentionally include empty and
    # policy-ineligible trajectories. The train/no-train decision must not:
    # a sampled-out or invalid-verifier child with a different reward cannot
    # turn an otherwise zero-signal retained batch into an optimizer step.
    retained_action_advantages = [
        datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0] for datum in datums
    ]
    retained_action_advantages = [values for values in retained_action_advantages if values.numel() > 0]
    if not retained_action_advantages:
        logger.debug("No retained policy action tokens for task %s", data["task_id"])
        return "no_datums", 0
    flattened_advantages = torch.cat(retained_action_advantages)
    if getattr(config, "filter_zero_advantage_datums", True) and bool(torch.all(flattened_advantages == 0)):
        logger.debug(
            "Deferring all-zero policy datums for task %s to the post-transform trainer filter (mean task reward %.2f)",
            data["task_id"],
            sum(completed_rewards) / len(completed_rewards),
        )
        # Keep these candidates until the trainer has assembled the
        # cross-task microbatch. They must participate in depth-frequency
        # normalization and the original action-token denominator even
        # though they can be omitted from the expensive model pass.

    trainable_rollouts = sum(
        any(
            bool(
                torch.any(
                    datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0] != 0
                )
            )
            for datum in result.datums
        )
        for result in results
    )
    # Every retained advantage zero: the group trains nothing (a uniform group).
    return ("trained" if trainable_rollouts else "uniform"), trainable_rollouts


class GroupCredit:
    """``credit: group``, the estimator of today's credit (steps 2 and 3 above): each member's rows centered on its
    baseline from the group's completed root rewards, the group mean, or the leave-one-out mean with the entry's
    ``leave_one_out_baseline``, as ``GroupRolloutWorkflow.assign_advantages`` centers them, value for value, with the
    same ``credit/*`` metrics.  An entry that sets no ``credit`` gets the same credit from the workflow itself.  It
    takes no options and reads no agents."""

    needs_agents = False

    def __init__(self, setup: "CreditSetup", options: None = None) -> None:
        self.setup = setup

    async def assign(self, group: "CreditGroup") -> None:
        members = group.members
        baselines = group_baselines(
            [member.task_reward for member in members],
            [member.task_reward_valid for member in members],
            leave_one_out=group.settings.leave_one_out_baseline,
        )
        for member, baseline in zip(members, baselines, strict=True):
            for sample in member.samples:
                sample.set_advantages(sample.advantages - baseline)
        completed = [
            (member.task_reward, baseline)
            for member, baseline in zip(members, baselines, strict=True)
            if member.task_reward_valid
        ]
        if completed:
            group.stats.scalar(
                **{
                    "credit/baseline/avg": sum(baseline for _, baseline in completed) / len(completed),
                    "credit/baseline/spread": max(b for _, b in completed) - min(b for _, b in completed),
                    "credit/root_advantage/abs_avg": sum(abs(reward - baseline) for reward, baseline in completed)
                    / len(completed),
                }
            )
