"""Credit over a task group, in this order:

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
   and datums left without action tokens are dropped.
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
"""

from collections.abc import Sequence

import tinker
import torch
from tinker import TensorData

from platoon.utils.trajectory_error_filtering import ERROR_ACTION_MASK_KEY


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
