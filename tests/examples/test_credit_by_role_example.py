"""The credit-by-role example (``examples/credit-by-role``) trains as its README says, on CPU: its config validates,
its estimator passes ``platoon.testing.check_credit_estimator``, and ``platoon train`` runs in process on
``platoon.testing``'s fake Tinker service, whose rollouts sample a scripted policy through the real gateway, the
subagents example's Vault environment and delegating harness.  The subagents always answer right, and the roots of
each group answer right in one rollout and wrong in the other.  Centered on its role, every subagent's advantage is 0,
so the step sends only the roots' rows, as many with a positive advantage as with a negative one; under the
workflow's own credit, centered on the roots' baseline, the subagents' rows train too.  And an estimator that raises
stops the run, which trains nothing and saves no final checkpoint."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_subagents_example import tool_call, vault_policy

import platoon.experimental as px
from platoon import cli
from platoon.testing import check_credit_estimator
from platoon.testing import trainers as checks
from platoon.train.credit.runtime import CreditEstimatorError

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "credit-by-role"
CONFIG = "examples/credit-by-role/train.yaml"
_FIRST_USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


@pytest.fixture(autouse=True)
def _as_the_readme_runs_it(monkeypatch):
    """With the example's directory on the path, beside the subagents example's (which the conftest adds)."""

    monkeypatch.syspath_prepend(str(EXAMPLE))


def one_right_root_a_group():
    """The subagents example's policy, except that each task's root answers wrong in every other rollout."""

    answers: Counter[str] = Counter()

    def policy(prompt: str) -> str:
        reply = vault_policy(prompt)
        goal = _FIRST_USER.search(prompt)
        if '"name": "answer"' in reply and "<tool_response>\nAnswered" in prompt and goal is not None:
            answers[goal.group(1)] += 1  # a root adding up its subagents' answers
            if answers[goal.group(1)] % 2:
                return tool_call("answer", total=json.loads(reply.split("\n")[1])["arguments"]["total"] + 1)
        return reply

    return policy


def train(serve_policy, log_path: Path, config: str = CONFIG) -> tuple[checks.FakePolicyService, Path]:
    """``platoon train`` on the example's config (or ``config``) on the fake service: 2 steps of 2 groups of 2
    rollouts."""

    serve_policy(one_right_root_a_group())
    service = checks.FakePolicyService()
    settings = (
        f"log_path={log_path}",
        "train.batch_size=2",
        "train.max_training_steps=2",
        "train.workflow_config.group_size=2",
        "eval.every=0",
        "watchdog.enabled=false",
    )
    with checks._faked(service), checks._restored_signals():
        cli.main(["train", config, *settings])
    return service, log_path / "subagents" / "delegator-by-role"


def _advantages(datum: Any) -> list[float]:
    """A sent datum's nonzero advantages, scaled as the step sends them (its mask stays in the trainer)."""

    return [value for value in datum.loss_fn_inputs["advantages"].to_torch().tolist() if value != 0]


def test_the_example_config_validates(capsys):
    cli.main(["config", "validate", CONFIG])
    assert capsys.readouterr().out.startswith("ok ")


def test_the_estimator_keeps_the_credit_contract():
    check_credit_estimator("by_role:ByRole", options={"leave_one_out": True})


def test_platoon_train_credits_each_role_on_its_own(serve_policy, tmp_path):
    service, run = train(serve_policy, tmp_path / "by-role")

    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [(row["name"], row["batch"]) for row in rows] == [("recovery-000000", 0), ("final", 2)]
    # The roots' rows only, each a right root's (positive) or a wrong one's (negative), as many of each.
    signs = [{value > 0 for value in _advantages(datum)} for datum in service.datums]
    assert all(len(sign) == 1 for sign in signs)
    assert signs.count({True}) == signs.count({False}) > 0
    lock = json.loads((run / "lock.json").read_text())
    [record] = [record for record in lock["components"] if record["kind"] == "credit"]
    assert (record["ref"], record["environments"], record["options"]) == (
        "by_role:ByRole",
        ["vault"],
        {"vault": {"leave_one_out": True}},
    )

    # The same config without the estimator: under the workflow's own credit the subagents' rows, each a right answer
    # above the roots' mean, train too.
    data = yaml.safe_load((EXAMPLE / "train.yaml").read_text())
    data["extends"] = str(EXAMPLE.parent / "subagents" / "subagents.yaml")
    del data["environments"][0]["credit"], data["environments"][0]["credit_options"]
    (tmp_path / "plain.yaml").write_text(yaml.safe_dump(data))
    plain, _ = train(serve_policy, tmp_path / "plain", str(tmp_path / "plain.yaml"))
    assert len(plain.datums) > len(service.datums)


class Broken(px.CreditEstimatorBase):
    """An estimator with a bug, for the run that it stops."""

    async def assign(self, group: px.CreditGroup) -> None:
        raise KeyError("a bug in the estimator")


def _leaves(error: BaseException) -> list[BaseException]:
    if isinstance(error, BaseExceptionGroup):
        return [leaf for inner in error.exceptions for leaf in _leaves(inner)]
    return [error]


def test_an_estimator_that_raises_stops_platoon_train(serve_policy, tmp_path):
    data = yaml.safe_load((EXAMPLE / "train.yaml").read_text())
    data["extends"] = str(EXAMPLE.parent / "subagents" / "subagents.yaml")
    data["environments"][0]["credit"] = f"{__name__}:Broken"
    del data["environments"][0]["credit_options"]
    (tmp_path / "broken.yaml").write_text(yaml.safe_dump(data))

    with pytest.raises(BaseException) as raised:
        train(serve_policy, tmp_path / "broken", str(tmp_path / "broken.yaml"))
    [error] = [leaf for leaf in _leaves(raised.value) if isinstance(leaf, CreditEstimatorError)]
    assert str(error).startswith(f"the credit estimator {__name__}:Broken of environments[vault]: assign(group) raised")
    run = tmp_path / "broken" / "subagents" / "delegator-by-role"
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [row["name"] for row in rows] == ["recovery-000000"]  # no step trained, no final checkpoint
