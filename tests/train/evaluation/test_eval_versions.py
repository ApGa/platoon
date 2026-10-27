"""The evaluations of a run that trains several policies in lockstep record each sampled policy's version
(``versions`` in ``evaluations.jsonl``); those of a run of one trained policy record none, as before."""

from __future__ import annotations

import json
from typing import Any

from fakes.trainer import named_policy

from platoon.train.evaluation.record import record_evaluation
from platoon.train.policy.set import PolicySet
from platoon.train.policy.spec import MAIN, AgentBindings, PolicySpec


def test_an_evaluation_of_a_run_of_several_trained_policies_records_their_versions(tmp_path):
    path = tmp_path / "evaluations.jsonl"
    record_evaluation(path, step=2, logged_at=3, tasks=4, seconds=1.0)
    record_evaluation(path, step=4, logged_at=4, tasks=4, seconds=1.0, versions={MAIN: 4, "worker": 3})
    first, second = (json.loads(line) for line in path.read_text().splitlines())
    assert "versions" not in first and second["versions"] == {MAIN: 4, "worker": 3}


def test_an_evaluation_records_versions_only_in_a_run_of_several_trained_policies():
    from platoon.train.evaluation.loop import _versions

    main, trained, frozen = named_policy(MAIN), named_policy("worker"), named_policy("judge")
    frozen.spec = PolicySpec("judge", model_name="m", renderer_name="r", trainable=False)
    llm: Any = trained.info.llm
    llm.version = 3
    agents = AgentBindings(default=MAIN, roles={"subagent": "worker", "judge": "judge"})
    assert _versions(PolicySet([main, trained, frozen], agents)) == {MAIN: 0, "worker": 3, "judge": 0}
    assert _versions(PolicySet([main, frozen], AgentBindings(MAIN, {"judge": "judge"}))) == {}
    assert _versions(None) == {}
