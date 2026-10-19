"""What an evaluation counts of each rollout's model calls (``platoon.eval.accounting``): usage by kind of call, cost
and a hosted model's record, from the trajectory collection's dump."""

from __future__ import annotations

from types import SimpleNamespace

from platoon.eval.accounting import report_blocks, rollout_accounting


def _step(**action_misc):
    return {"misc": {"action_misc": {"completion_id": "c", "model": "m", **action_misc}}}


def _collection(*steps):
    return {"trajectories": {"root": {"id": "root", "steps": list(steps), "misc": {}}}}


def test_a_call_without_reported_usage_is_counted_and_left_unpriced():
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    priced = _step(dialect="openai_chat", usage=usage, cost_usd=0.5, attempts=1, served_model="m-1")
    unreported = _step(
        dialect="openai_chat",
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        attempts=2,
        rate_limited=1,
        usage_reported=False,
    )

    accounting = rollout_accounting(_collection(priced, unreported))

    assert accounting["usage"]["policy"]["calls"] == 2 and accounting["usage"]["policy"]["prompt_tokens"] == 100
    assert accounting["cost"] == {"usd": 0.5, "unpriced_calls": 1}
    backend = accounting["backend"]
    assert (backend["calls_without_usage"], backend["attempts"], backend["rate_limited"]) == (1, 3, 1)
    assert backend["served_models"] == {"m-1": 1}

    report = report_blocks([SimpleNamespace(**accounting), SimpleNamespace(**accounting)], judges_enabled=True)
    assert report["backend"]["calls_without_usage"] == 2 and report["backend"]["served_models"] == {"m-1": 2}
    assert report["cost"] == {"total_usd": 1.0, "mean_usd_per_rollout": 0.5, "unpriced_calls": 2}


def test_a_hosted_run_without_prices_has_no_cost():
    unpriced = _step(dialect="anthropic_messages", usage={"prompt_tokens": 5, "completion_tokens": 1}, attempts=1)
    accounting = rollout_accounting(_collection(unpriced))

    assert accounting["cost"] == {"usd": None, "unpriced_calls": 1}
    report = report_blocks([SimpleNamespace(**accounting)], judges_enabled=True)
    assert report["cost"] == {"total_usd": None, "mean_usd_per_rollout": None, "unpriced_calls": 1}


def test_a_tinker_call_counts_its_tokens_and_nothing_hosted():
    usage = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    accounting = rollout_accounting(_collection(_step(usage=usage), _step(usage=usage, kind="aux")))

    assert accounting["usage"]["policy"]["calls"] == 1 and accounting["usage"]["aux"]["completion_tokens"] == 3
    assert accounting["cost"] == {"usd": None, "unpriced_calls": 0}
    assert accounting["backend"]["attempts"] == 0 and accounting["backend"]["calls_without_usage"] == 0
