"""``credit: roles`` in a run, on CPU: ``platoon train`` on the subagents example's Vault environment in groups of one
rollout, on ``platoon.testing``'s fake Tinker service, whose rollouts sample a scripted policy through the real gateway
(the credit-by-role example's harness: the subagents answer right, and each task's root answers wrong in every other
rollout).  The roots (role ``policy``) are credited with ``running`` and a share of the mass, the subagents with
``running`` of their own: nothing is warned of, ``lock.json`` records each role's estimator with its options, each step
logs the roles' masses and metrics, and each full checkpoint saves each role's statistics with the masses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_credit_by_role_example import EXAMPLE, train

from platoon import cli
from platoon.utils.stats_logger import StatsLogger

_PRIORS = {"mean": 0.5, "scale": 0.5}
_ROLES = ["policy", "subagent"]
_BY_ROLE = {
    "policy": {"credit": "running", "options": {"priors": _PRIORS}, "share": 0.25},
    "subagent": {"credit": "running", "options": {"priors": {"mean": 0.25, "scale": 0.5}}},
}


@pytest.fixture(autouse=True)
def _as_the_readme_runs_it(monkeypatch):
    """With the credit-by-role example's directory on the path, beside the subagents example's."""

    monkeypatch.syspath_prepend(str(EXAMPLE))


def _config(tmp_path: Path) -> str:
    """The credit-by-role example's config with groups of one rollout, each role credited on its own."""

    data = yaml.safe_load((EXAMPLE / "train.yaml").read_text())
    data["extends"] = str(EXAMPLE.parent / "subagents" / "subagents.yaml")
    data["train"] = {"max_staleness": 1}  # how old a row may be, and so how far back running's statistics reach
    entry = data["environments"][0]
    entry.update(credit="roles", credit_options={"by_role": _BY_ROLE}, workflow_config={"group_size": 1})
    path = tmp_path / "roles.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def test_platoon_train_credits_each_role_with_its_own_estimator_and_holds_the_roots_share(
    serve_policy, tmp_path, capsys, monkeypatch
):
    config = _config(tmp_path)
    cli.main(["config", "validate", config])
    assert capsys.readouterr().out.startswith(f"ok   {config}")  # its groups of one are not warned of
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(StatsLogger, "log", lambda self, step, stats, *args, **kwargs: logged.append(dict(stats)))

    _, run = train(serve_policy, tmp_path / "roles", config)

    lock = json.loads((run / "lock.json").read_text())
    record, running = [record for record in lock["components"] if record["kind"] == "credit"]
    assert (record["ref"], record["resolved"]) == ("roles", "platoon.train.credit.roles:Roles")
    assert (running["ref"], running["level"], sorted(running["options"]["vault"])) == ("running", "role", _ROLES)
    by_role = record["options"]["vault"]["by_role"]
    assert by_role["policy"]["share"] == 0.25 and by_role["policy"]["options"]["priors"]["weight"] == 16.0
    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    saved = rows[-1]["credit"]["vault"]
    assert rows[-1]["name"] == "final" and saved["class"] == "platoon.train.credit.roles:Roles"
    assert sorted(saved["state"]["estimators"]) == ["policy", "subagent"]
    [[policy, by_version]] = saved["state"]["masses"]  # each version's masses, as far back as a read reaches
    *_, (_, masses) = by_version
    assert policy == "main" and [role for role, _ in masses] == ["policy", "subagent"]

    assert len(logged) == 2
    for stats in logged:
        shares = [stats[f"train/roles/{role}/mass_share"] for role in ("policy", "subagent")]
        assert all(0 <= share <= 1 for share in shares) and sum(shares) == pytest.approx(1.0)
        assert stats["train/roles/policy/log_ratio/abs_avg"] >= 0 and stats["train/roles/policy/entropy"] >= 0
        assert stats["train/credit/role_groups"] == 2.0  # the step's two groups
        assert "train/credit/roles/policy/running/advantage/avg" in stats
        assert "train/credit/role_mass/subagent" in stats
    # Once the masses of a group settled, the roots train at the weight that holds their share.
    assert "train/credit/role_weight/policy" in logged[-1]
