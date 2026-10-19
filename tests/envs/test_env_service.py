from __future__ import annotations

import pytest

from platoon.envs.service import EnvSetupError, Outcome, ToolResult


def test_scored_outcomes_train_on_their_primary_reward():
    outcome = Outcome(status="scored", rewards={"reward": 0.25, "tests_passed": 3.0})

    assert outcome.trainable
    assert outcome.reward == 0.25
    assert Outcome(status="scored", rewards={"pass": 1.0}, primary="pass").reward == 1.0


@pytest.mark.parametrize("status", ["invalid", "error", "timeout", "cancelled"])
def test_unscored_outcomes_never_report_a_reward(status):
    outcome = Outcome(status=status, rewards={"reward": 0.0})

    assert not outcome.trainable
    assert outcome.reward is None


def test_scored_outcomes_require_the_primary_reward():
    with pytest.raises(ValueError, match="primary reward 'reward'"):
        Outcome(status="scored", rewards={"other": 1.0})


def test_infra_outcomes_carry_a_failure_kind():
    outcome = Outcome.infra("verifier_timeout", status="invalid", metadata={"timeout_s": 10})

    assert (outcome.status, outcome.failure_kind, outcome.reward) == ("invalid", "verifier_timeout", None)
    assert outcome.metadata == {"timeout_s": 10}
    assert Outcome.infra("sandbox_start").status == "error"
    with pytest.raises(ValueError, match="cannot be scored"):
        Outcome.infra("oops", status="scored")


def test_tool_result_text_and_setup_errors():
    expected = ToolResult(content=[{"type": "text", "text": "boom"}], is_error=True)
    assert ToolResult.text("boom", is_error=True) == expected
    error = EnvSetupError("sandbox_start", "no capacity")
    assert (error.failure_kind, str(error)) == ("sandbox_start", "no capacity")
