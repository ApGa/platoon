from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "platoon/train/areal/numerical_stability.py"

_SPEC = importlib.util.spec_from_file_location(
    "platoon_areal_numerical_stability_test",
    MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
aggregate_optimizer_update_results = _MODULE.aggregate_optimizer_update_results
install_nonfinite_gradient_guard = _MODULE.install_nonfinite_gradient_guard
make_optimizer_update_result = _MODULE.make_optimizer_update_result
optimizer_update_succeeded = _MODULE.optimizer_update_succeeded


class _FakeDirectOptimizer:
    def __init__(self, grad_norm: float):
        self.grad_norm = grad_norm
        self.parameter = 1.0
        self.optimizer_steps = 0

    def prepare_grads(self):
        return False

    def get_grad_norm(self):
        return self.grad_norm

    def clip_grad_norm(self, _max_norm):
        result = self.grad_norm
        if not math.isfinite(result):
            # Mirrors inf * 0 becoming NaN in a clipping kernel.
            self.grad_norm = float("nan")
        return result

    def step_with_ready_grads(self):
        self.parameter += self.grad_norm
        self.optimizer_steps += 1
        return True

    def step(self):
        if self.prepare_grads():
            return False, None, None
        grad_norm = self.clip_grad_norm(1.0)
        successful = self.step_with_ready_grads()
        return successful, grad_norm, 0


class _FakeChainedOptimizer(_FakeDirectOptimizer):
    def step(self):
        if self.prepare_grads():
            return False, None, None
        grad_norm = self.get_grad_norm()
        if not math.isfinite(grad_norm):
            self.grad_norm = float("nan")
        successful = self.step_with_ready_grads()
        return successful, grad_norm, 0


def test_direct_bf16_style_optimizer_skips_nonfinite_update_and_recovers():
    optimizer = _FakeDirectOptimizer(float("inf"))
    assert install_nonfinite_gradient_guard(optimizer) is True
    assert install_nonfinite_gradient_guard(optimizer) is False

    successful, grad_norm, _ = optimizer.step()
    assert successful is False
    assert math.isinf(grad_norm)
    assert optimizer.parameter == 1.0
    assert optimizer.optimizer_steps == 0

    optimizer.grad_norm = 0.25
    successful, grad_norm, _ = optimizer.step()
    assert successful is True
    assert grad_norm == 0.25
    assert optimizer.parameter == 1.25
    assert optimizer.optimizer_steps == 1


def test_chained_megatron_style_optimizer_skips_nonfinite_update():
    optimizer = _FakeChainedOptimizer(float("inf"))
    install_nonfinite_gradient_guard(optimizer)

    successful, grad_norm, _ = optimizer.step()

    assert successful is False
    assert math.isinf(grad_norm)
    assert optimizer.parameter == 1.0
    assert optimizer.optimizer_steps == 0


def test_update_result_fails_closed_on_nonfinite_reported_norm():
    assert optimizer_update_succeeded({"update_successful": 1.0, "grad_norm": 0.5})
    assert not optimizer_update_succeeded({"update_successful": 1.0, "grad_norm": float("inf")})
    assert not optimizer_update_succeeded({"update_successful": 1.0, "grad_norm": float("nan")})
    assert not optimizer_update_succeeded({"update_successful": 1.0})
    assert not optimizer_update_succeeded({"update_successful": 0.0, "grad_norm": 0.5})
    # A disabled clipping configuration intentionally has no norm to inspect.
    assert optimizer_update_succeeded(
        {"update_successful": 1.0, "grad_norm": float("nan")},
        require_finite_grad_norm=False,
    )


def test_partial_minibatch_update_is_published_and_remains_finite():
    optimizer = _FakeDirectOptimizer(0.25)
    install_nonfinite_gradient_guard(optimizer)
    successes = []

    for grad_norm in (0.25, float("inf"), 0.5):
        optimizer.grad_norm = grad_norm
        successful, _, _ = optimizer.step()
        successes.append(successful)

    result = make_optimizer_update_result(successes)
    assert successes == [True, False, True]
    assert optimizer.parameter == 1.75
    assert optimizer.optimizer_steps == 2
    assert math.isfinite(optimizer.parameter)
    # The two finite mutations must be broadcast/checkpointed even though the
    # middle minibatch was safely skipped.
    assert aggregate_optimizer_update_results([result, [result, result]])


def test_all_skipped_minibatches_report_no_model_mutation():
    result = make_optimizer_update_result([False, False])

    assert not aggregate_optimizer_update_results([result, result])


def test_controller_requires_exact_rank_agreement():
    result = make_optimizer_update_result([True, False])
    assert aggregate_optimizer_update_results([result, [result, result]])

    with pytest.raises(RuntimeError, match="disagreed"):
        aggregate_optimizer_update_results(
            [
                make_optimizer_update_result([True, False]),
                make_optimizer_update_result([False, True]),
            ]
        )
    with pytest.raises(RuntimeError, match="no result"):
        aggregate_optimizer_update_results([result, None])


def test_controller_supports_legacy_uniform_results():
    assert aggregate_optimizer_update_results([True, [True, True]])
    assert not aggregate_optimizer_update_results([False, [False, False]])
    with pytest.raises(RuntimeError, match="disagreed"):
        aggregate_optimizer_update_results([True, False])
    assert aggregate_optimizer_update_results(None)
