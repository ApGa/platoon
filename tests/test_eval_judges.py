"""``inference.judges`` in ``platoon eval``: what an evaluation with its judges off refuses, records and reports.

The rollouts run the toy function harness of ``tests/inline`` (``toy-agent``) on its forkable toy service, through
the real gateway with ``platoon.testing``'s scripted policy in place of Tinker.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
TESTS = Path(__file__).resolve().parent
for fakes in ("inline", "gateway"):
    if str(TESTS / fakes) not in sys.path:
        sys.path.insert(0, str(TESTS / fakes))

from inline_fakes import ToyService  # noqa: E402

import platoon as pt  # noqa: E402
from platoon import cli  # noqa: E402
from platoon.testing import TOY_MODEL_NAME, rules, serve_policy  # noqa: E402

VERIFIED = json.dumps({"status": "verified", "score": 1.0, "summary": "checked"})
POLICY = rules(
    {
        "goal:t0": "SPAWN goal:a",
        "goal:a": "SUBMIT ok-a",
        "You are a verifier agent": "LOOK",
        "obs: submissions": f"FINISH {VERIFIED}",
        "obs: [": "SUBMIT ok",
        "obs: submitted": "FINISH done",
    }
)


@pt.program("evaltest-judging")
async def judging(ep: pt.EpisodeContext) -> pt.Outcome:
    ep.judge_endpoint()  # refused while judges are off
    await ep.run()
    return await ep.grade()


def reward_from_child_verdicts(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    return float(trajectory.get("reward", 0.0)), {}


@pytest.fixture(autouse=True)
def _toy_service():
    ToyService.sessions.clear()
    yield
    ToyService.sessions.clear()


def _config(tmp_path: Path, *, inference: dict | None = None, environment: dict | None = None) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "model_name": TOY_MODEL_NAME,
                    "renderer_name": "qwen3_instruct",
                    "output_dir": str(tmp_path / "results"),
                    "workflow": {"num_concurrent_workers": 1, "rollout_config": {"verbose": False, "max_steps": 20}},
                    **(inference or {}),
                },
                "environments": [
                    {
                        "name": "toy",
                        "package": "inline_fakes",
                        "service": "inline_fakes:ToyService",
                        "harness": "toy-agent",
                        "agents": {"child_grading": "verifier", "verifier": {"max_steps": 4}},
                        **(environment or {}),
                    }
                ],
            }
        )
    )
    return path


def _eval(config: Path, *overrides: str, policy=POLICY) -> int:
    with serve_policy(policy):
        try:
            cli.main(["eval", str(config), *overrides])
        except SystemExit as exit:
            return int(exit.code or 0)
    return 0


def _report(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "results/reports/final_report.json").read_text())


def _lock(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "results/lock.json").read_text())


def _metadata(tmp_path: Path) -> dict[str, Any]:
    (path,) = (tmp_path / "results/rollouts").glob("*/rollout_0/metadata.json")
    return json.loads(path.read_text())


def test_judges_off_runs_no_verifier_and_says_so_everywhere(tmp_path, capsys):
    assert _eval(_config(tmp_path, inference={"judges": False})) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["success_rate"] == 1.0  # the root's success: judges never decide it
    report = _report(tmp_path)
    assert report["judges"] == {
        "enabled": False,
        "verifier_agents": 0,
        "verifier_steps": 0,
        "judge_calls": 0,
        "refused": 0,
        "children_sharing_episode_outcome": {"judges_off": 1},
    }
    assert report["stats"]["num_steps_subtrajectories"]["overall"]["mean"] > 0  # the child, a policy agent
    assert _lock(tmp_path)["judges"] == {
        "enabled": False,
        "child_grading": {"toy": {"configured": "verifier", "applied": "none", "reason": "judges_off"}},
    }
    metadata = _metadata(tmp_path)
    assert metadata["provenance"]["judges"] is False and metadata["judges"]["grading_fallbacks"] == {"judges_off": 1}
    (collection_path,) = (tmp_path / "results/rollouts").glob("*/rollout_0/trajectory_collection.json")
    root, child = json.loads(collection_path.read_text())["trajectories"].values()
    assert root["misc"]["judges"] == {"enabled": False}
    assert child["misc"]["child_grading"]["reason"] == "judges_off"


def test_judges_on_count_verifiers_apart_from_the_policy(tmp_path):
    assert _eval(_config(tmp_path)) == 0

    report = _report(tmp_path)
    judges = report["judges"]
    assert judges["enabled"] is True and judges["verifier_agents"] == 1 and judges["verifier_steps"] == 2
    assert judges["children_sharing_episode_outcome"] == {}
    assert _lock(tmp_path)["judges"] == {"enabled": True}
    (collection_path,) = (tmp_path / "results/rollouts").glob("*/rollout_0/trajectory_collection.json")
    trajectories = json.loads(collection_path.read_text())["trajectories"]
    policy_steps = sum(len(t["steps"]) for t in trajectories.values() if t["misc"]["agent"]["role"] != "verifier")
    # The step statistics count the policy's agents: the root and its child, not the verifier.
    assert report["stats"]["num_steps_total"]["overall"]["mean"] == policy_steps
    assert report["summary"]["valid_rollouts"] == 1
    assert report["usage"]["verifier"]["total"]["calls"] == 2
    assert report["usage"]["policy"]["total"]["calls"] == policy_steps


def test_an_entry_whose_own_code_may_read_child_verdicts_is_refused(tmp_path, capsys):
    environment = {"reward_processor": f"{__name__}:reward_from_child_verdicts"}
    assert _eval(_config(tmp_path, inference={"judges": False}, environment=environment)) == 78

    error = capsys.readouterr().err
    assert "environments[toy]: inference.judges is false" in error and "reward_processor" in error
    assert "agents.child_grading: none" in error
    assert not (tmp_path / "results/rollouts").exists()

    # Stating the fallback runs it.
    agents = {"child_grading": "none"}
    assert _eval(_config(tmp_path, inference={"judges": False}, environment={**environment, "agents": agents})) == 0


def test_a_refused_judge_stops_the_evaluation(tmp_path, capsys):
    workflow = {"num_rollouts_per_task": 3, "num_concurrent_workers": 1, "rollout_config": {"verbose": False}}
    config = _config(
        tmp_path,
        inference={"judges": False, "workflow": workflow},
        environment={"program": "evaltest-judging", "agents": {"child_grading": "none"}},
    )

    assert _eval(config) == 78

    error = capsys.readouterr().err
    assert "the evaluation stopped after 1 rollout(s), 2 not started" in error
    assert "judges_off: a judge agent was refused" in error and "program 'evaltest-judging'" in error
    (metadata_path,) = (tmp_path / "results/rollouts").glob("*/rollout_*/metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["status"] == "errored" and metadata["error"].startswith("judges_off:")
    assert not (tmp_path / "results/reports").exists()


def test_the_rollout_config_cannot_turn_judges_off_by_itself(tmp_path, capsys):
    config = _config(tmp_path)
    assert _eval(config, "inference.workflow.rollout_config.judges=false") == 78
    assert "set inference.judges: false instead" in capsys.readouterr().err


def test_resuming_with_judges_switched_is_refused(tmp_path, capsys):
    config = _config(tmp_path)
    assert _eval(config) == 0
    capsys.readouterr()

    assert _eval(config, "inference.judges=false") == 78
    error = capsys.readouterr().err
    assert "holds rollouts made with judges True (now False)" in error and "set a new inference.output_dir" in error


def test_a_report_of_the_rollouts_takes_the_judges_setting_that_made_them(tmp_path, capsys):
    """``stage: report`` labels the rollouts with the run's settings: a report with judges off of rollouts made with
    judges on would claim judges that ran were off."""

    config = _config(tmp_path)
    assert _eval(config) == 0
    capsys.readouterr()

    assert _eval(config, "inference.judges=false", "stage=report") == 78
    assert "holds rollouts made with judges True (now False)" in capsys.readouterr().err
    assert _eval(config, "stage=report") == 0
