"""Numerical-safety helpers for AReaL/Megatron optimizer updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch

MINIBATCH_UPDATE_SUCCESSES_KEY = "minibatch_update_successes"


def _as_finite_scalar(value: Any) -> tuple[bool, float | None]:
    if value is None:
        return False, None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return False, None
        value = value.detach().item()
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return False, None
    return math.isfinite(scalar), scalar


def optimizer_update_succeeded(
    train_stat: dict[str, Any],
    *,
    require_finite_grad_norm: bool = True,
) -> bool:
    """Fail closed when an engine claims success with a non-finite norm."""

    successful = bool(train_stat.get("update_successful", True))
    if not require_finite_grad_norm:
        return successful
    finite, _ = _as_finite_scalar(train_stat.get("grad_norm"))
    return successful and finite


def make_optimizer_update_result(
    minibatch_successes: list[bool],
) -> dict[str, list[bool]]:
    """Build the small RPC payload describing every minibatch optimizer step."""

    return {
        MINIBATCH_UPDATE_SUCCESSES_KEY: [
            bool(successful) for successful in minibatch_successes
        ]
    }


def _collect_optimizer_update_patterns(result: Any) -> list[tuple[bool, ...] | None]:
    if result is None:
        return [None]
    if isinstance(result, dict):
        if set(result) != {MINIBATCH_UPDATE_SUCCESSES_KEY}:
            raise RuntimeError(
                "Malformed optimizer update result from actor worker: "
                f"expected only {MINIBATCH_UPDATE_SUCCESSES_KEY!r}, got "
                f"{sorted(result)}"
            )
        raw_pattern = result[MINIBATCH_UPDATE_SUCCESSES_KEY]
        if not isinstance(raw_pattern, (list, tuple)):
            raise RuntimeError(
                "Malformed optimizer update result from actor worker: "
                f"{MINIBATCH_UPDATE_SUCCESSES_KEY!r} must be a list"
            )
        return [tuple(bool(value) for value in raw_pattern)]
    if isinstance(result, (list, tuple)):
        patterns: list[tuple[bool, ...] | None] = []
        for item in result:
            patterns.extend(_collect_optimizer_update_patterns(item))
        return patterns
    # Compatibility with an older Platoon worker returning a single bool.
    return [(bool(result),)]


def aggregate_optimizer_update_results(result: Any) -> bool:
    """Validate replica agreement and report whether actor weights changed.

    Controller tensor dispatch replicates each DP worker's scalar/dict result
    once per trajectory in that shard. All workers participate in the same
    gradient-norm collectives and therefore must report the exact same
    per-minibatch success pattern. Silently reducing disagreement to ``False``
    would be unsafe: a worker that reported success may already have mutated its
    local weights. Fail hard instead so recovery uses the previous consistent
    checkpoint.

    ``True`` means one or more finite minibatches applied an optimizer step. A
    partial pattern such as ``[True, False]`` must still be broadcast and
    checkpointed; the failed minibatch itself remains non-mutating.
    """

    patterns = _collect_optimizer_update_patterns(result)
    if not patterns or all(pattern is None for pattern in patterns):
        # Backward compatibility for stock AReaL actor implementations, whose
        # ppo_update RPC returns None.
        return True
    if any(pattern is None for pattern in patterns):
        raise RuntimeError(
            "Actor workers disagreed on optimizer update reporting: some "
            "returned no result while others returned a success pattern."
        )

    reported_patterns = [pattern for pattern in patterns if pattern is not None]
    reference = reported_patterns[0]
    if any(pattern != reference for pattern in reported_patterns[1:]):
        raise RuntimeError(
            "Actor workers disagreed on per-minibatch optimizer update "
            f"success: {reported_patterns}"
        )
    return any(reference)


@dataclass
class _NonfiniteGradientState:
    nonfinite: bool = False
    grad_norm: float | None = None


def install_nonfinite_gradient_guard(
    optimizer: Any,
    *,
    logger: Any | None = None,
) -> bool:
    """Make Megatron's BF16 optimizer skip non-finite-gradient updates.

    Megatron's BF16 optimizer has no gradient scaler by default.  Its standard
    path clips an infinite norm, reports ``update_successful=True``, and executes
    Adam; multiplying an infinite gradient by a zero clip coefficient can turn
    the gradient into NaN and irreversibly poison weights and optimizer state.

    This narrowly wraps the optimizer's existing norm and ready-step methods.
    The original collectives and clipping behavior are retained, but
    ``step_with_ready_grads`` returns ``False`` when the norm observed in that
    step was non-finite.  That matches Megatron's existing overflow-skip
    contract and leaves parameters and optimizer state untouched.
    """

    if getattr(optimizer, "__platoon_nonfinite_gradient_guard__", False):
        return False

    required_methods = (
        "prepare_grads",
        "get_grad_norm",
        "clip_grad_norm",
        "step_with_ready_grads",
    )
    missing = [name for name in required_methods if not callable(getattr(optimizer, name, None))]
    if missing:
        raise TypeError(f"Optimizer does not expose the methods required for a non-finite gradient guard: {missing}")

    state = _NonfiniteGradientState()
    original_prepare_grads = optimizer.prepare_grads
    original_get_grad_norm = optimizer.get_grad_norm
    original_clip_grad_norm = optimizer.clip_grad_norm
    original_step_with_ready_grads = optimizer.step_with_ready_grads

    def _record_norm(value: Any) -> Any:
        finite, scalar = _as_finite_scalar(value)
        state.nonfinite = not finite
        state.grad_norm = scalar
        return value

    def guarded_prepare_grads(_self: Any, *args: Any, **kwargs: Any) -> Any:
        state.nonfinite = False
        state.grad_norm = None
        return original_prepare_grads(*args, **kwargs)

    def guarded_get_grad_norm(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _record_norm(original_get_grad_norm(*args, **kwargs))

    def guarded_clip_grad_norm(_self: Any, *args: Any, **kwargs: Any) -> Any:
        return _record_norm(original_clip_grad_norm(*args, **kwargs))

    def guarded_step_with_ready_grads(
        _self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        if state.nonfinite:
            if logger is not None:
                logger.error(
                    "Skipping optimizer update because gradient norm is non-finite: %r",
                    state.grad_norm,
                )
            return False
        return bool(original_step_with_ready_grads(*args, **kwargs))

    optimizer.prepare_grads = MethodType(guarded_prepare_grads, optimizer)
    optimizer.get_grad_norm = MethodType(guarded_get_grad_norm, optimizer)
    optimizer.clip_grad_norm = MethodType(guarded_clip_grad_norm, optimizer)
    optimizer.step_with_ready_grads = MethodType(
        guarded_step_with_ready_grads,
        optimizer,
    )
    optimizer.__platoon_nonfinite_gradient_guard__ = True
    optimizer.__platoon_nonfinite_gradient_state__ = state
    return True
