"""``credit: running`` in a run, on CPU: ``platoon train`` on the subagents example's Vault environment in groups of one
rollout, on ``platoon.testing``'s fake Tinker service, whose rollouts sample a scripted policy through the real gateway
(the credit-by-role example's harness: the subagents answer right, and each task's root answers wrong in every other
rollout).  Under the group baselines a group of one centers its root on its own reward, so no root ever trains, only
the subagents of a wrong root, above it; and ``platoon config validate`` warns of the entry.  Under ``running`` with
priors the roots train from the first step on, a wrong one's rows with a negative advantage, nothing is warned of,
``lock.json`` records the estimator with its priors, and each full checkpoint saves its statistics, the roots' and the
subagents' apart, with the groups it keeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_credit_by_role_example import EXAMPLE, _advantages, train

from platoon import cli

_PRIORS = {"mean": 0.5, "scale": 0.5}
_WARNING = "environments[vault].workflow_config.group_size is 1: a group of one rollout has a zero advantage"


@pytest.fixture(autouse=True)
def _as_the_readme_runs_it(monkeypatch):
    """With the credit-by-role example's directory on the path, beside the subagents example's."""

    monkeypatch.syspath_prepend(str(EXAMPLE))


def _config(tmp_path: Path, name: str, **credit: Any) -> str:
    """The credit-by-role example's config with groups of one rollout, crediting with ``credit`` (none: the
    workflow's own credit)."""

    data = yaml.safe_load((EXAMPLE / "train.yaml").read_text())
    data["extends"] = str(EXAMPLE.parent / "subagents" / "subagents.yaml")
    data["train"] = {"max_staleness": 1}  # how old a row may be, and so how far back running's statistics reach
    entry = data["environments"][0]
    del entry["credit"], entry["credit_options"]
    entry["workflow_config"] = {"group_size": 1}
    entry.update(credit)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_platoon_train_trains_groups_of_one_under_running(serve_policy, tmp_path, capsys):
    plain = _config(tmp_path, "plain")
    running = _config(tmp_path, "running", credit="running", credit_options={"priors": _PRIORS})
    cli.main(["config", "validate", plain, running])
    validated = capsys.readouterr().out
    assert validated.count(_WARNING) == 1 and validated.index(_WARNING) < validated.index(f"ok   {running}")

    service, run = train(serve_policy, tmp_path / "running", running)

    # The roots trained, a right one's rows above the prior's 0.5 and a wrong one's below it.
    signs = {value > 0 for datum in service.datums for value in _advantages(datum)}
    assert signs == {True, False}
    lock = json.loads((run / "lock.json").read_text())
    [record] = [record for record in lock["components"] if record["kind"] == "credit"]
    assert (record["ref"], record["resolved"]) == ("running", "platoon.train.credit.running:Running")
    assert record["options"]["vault"]["priors"] == {**_PRIORS, "weight": 16.0, "strata": []}
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    saved = rows[-1]["credit"]["vault"]
    assert rows[-1]["name"] == "final" and saved["class"] == "platoon.train.credit.running:Running"
    trackers = {(policy, role): tracker for policy, role, tracker in saved["state"]["trackers"]}
    assert ("main", "policy") in trackers and len(trackers) == 2  # the roots' statistics, and the subagents'
    assert all(tracker["settles"] for tracker in trackers.values())

    # The same groups under the workflow's own credit: no root trains, only the subagents above a wrong root.
    plain_run, _ = train(serve_policy, tmp_path / "plain", plain)
    assert {value > 0 for datum in plain_run.datums for value in _advantages(datum)} == {True}
