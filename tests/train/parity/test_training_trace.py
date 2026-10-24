"""The golden trace of the training loop: each scenario of ``trace_scenarios.SCENARIOS`` trains a few steps through the
trainer's own code on fakes, twice, and what the trainer did must match ``golden/<scenario>.json`` and
``golden/<scenario>.metrics.json`` byte for byte.  README.md says what a failure means and what to do about it."""

from __future__ import annotations

import pytest
from trace_format import GOLDEN_DIR
from trace_scenarios import SCENARIOS
from training_trace import trace_scenario


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_training_trace_matches_its_golden(name, tmp_path):
    result = trace_scenario(name, tmp_path)
    if not result.ok:
        pytest.fail(result.problems(result.write(tmp_path / "this-run")), pytrace=False)


def test_every_golden_has_a_scenario():
    names = {path.name.removesuffix(".json").removesuffix(".metrics") for path in GOLDEN_DIR.glob("*.json")}
    assert not names - set(SCENARIOS), f"golden files without a scenario: {sorted(names - set(SCENARIOS))}"
