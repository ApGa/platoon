"""When a start evaluates: whether an evaluation loop runs beside training, and the log line that says when it
evaluates first.

The functions here that take the trainer first are its methods still: ``PlatoonTinkerRLTrainer`` keeps each under its
name, so to replace one, replace the method, on the class or an instance, not the function.  What they read of the
trainer and of the loops' shared state is :mod:`platoon.train.evaluation.protocol`'s.
"""

import logging
from typing import TYPE_CHECKING

from platoon.train.evaluation.record import EVALUATIONS_NAME

if TYPE_CHECKING:
    from platoon.train.evaluation.protocol import EvaluatedTrainer, LoopState


def _evaluation_plan(state: "LoopState", *, resumed: bool, end: int) -> tuple[int, str]:
    """The log line that says when a start of the run that evaluates evaluates first, and why, with its level: a
    warning when a fresh run evaluates nothing, as its first evaluation would come after its last step, ``end``."""

    every, start = state.eval_every or 1, state.start_step
    first = start if state.evaluation_due(start) else (start // every + 1) * every
    if not resumed:
        if first == 0:
            then = f"then every {every} steps" if every <= end else f"and no other by the run's last step, {end}"
            return logging.INFO, (
                f"Evaluating the starting policy at step 0, before the first update (eval.at_start), {then}"
            )
        if first > end:
            setting = f"eval.every is {every}"
            if getattr(state.config.eval, "strategy", None) == "epoch":  # every counts epochs: say both
                setting = f"eval.every is {state.config.eval.every} epoch(s) (strategy: epoch), {every} steps"
            return logging.WARNING, (
                f"No evaluation runs: {setting}, past the run's last step, {end}; eval.at_start: true would evaluate "
                "the starting policy at step 0"
            )
        return logging.INFO, (
            f"Evaluating every {every} steps, the first at step {every}; eval.at_start: true would evaluate the "
            "starting policy at step 0 too"
        )
    if first == start:
        return logging.INFO, (
            f"Resuming at step {start}, whose evaluation is due and no earlier start completed ({EVALUATIONS_NAME} "
            "lists none): evaluating it now"
        )
    if start % every == 0 and state.start_step_evaluated:
        then = f"the next is at step {first}" if first <= end else f"no other is due by the run's last step, {end}"
        return logging.INFO, (
            f"Resuming at step {start}, whose evaluation an earlier start completed ({EVALUATIONS_NAME}): it does not "
            f"run again, and {then}"
        )
    if first > end:
        return logging.INFO, f"Resuming at step {start}: no evaluation is due by the run's last step, {end}"
    return logging.INFO, f"Resuming at step {start}: the next evaluation is at step {first}"


def _evaluates_during_training(trainer: "EvaluatedTrainer", shared_state: "LoopState") -> bool:
    """Whether an evaluation loop runs beside training: there is an eval split, and ``eval`` triggers (its
    ``every`` is positive)."""

    eval_every = shared_state.eval_every
    return trainer.eval_dataset is not None and eval_every is not None and eval_every > 0
