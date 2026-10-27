"""The replay-SFT hook example (``examples/replay-sft-hook``) on CPU: its config validates as its README runs it, and
the rl trainer trains with the step hook the config attaches, on ``platoon.testing``'s fake Tinker service, with a
workflow whose every group is one datum in place of the quickstart's rollouts.  The hook replays its conversations at
the end of every other step, under cross entropy and its own learning rate; each step publishes once; and a resumed run
goes on with the next conversations."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from example_runs import load_config

import platoon.experimental as px
from platoon import cli
from platoon.testing import trainers as checks
from platoon.train.components import task_ids_to_dataset
from platoon.train.policy.spec import policy_specs
from platoon.train.trainer.rl import PlatoonTinkerRLTrainer

pytest.importorskip("openreward")  # the quickstart's environment, which the example's config extends

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE, QUICKSTART = EXAMPLES / "replay-sft-hook", EXAMPLES / "quickstart"


@pytest.fixture(autouse=True)
def _as_the_readme_runs_it(monkeypatch):
    """From the quickstart's directory, with it and the example's on the path (``PYTHONPATH=.:../replay-sft-hook``)."""

    monkeypatch.syspath_prepend(str(QUICKSTART))
    monkeypatch.syspath_prepend(str(EXAMPLE))
    monkeypatch.chdir(QUICKSTART)


def train(log_path: Path, steps: int) -> tuple[checks.FakePolicyService, list[dict[str, Any]]]:
    """A start of the example's training on the fake service, ``steps`` steps of two groups, without evaluation; the
    service, and every row of the run's checkpoints.jsonl."""

    overrides = (f"log_path={log_path}", "eval.every=0", "train.batch_size=2", f"train.max_training_steps={steps}")
    config = load_config(EXAMPLE / "train.yaml", *overrides, "train.num_epochs=10")
    setup = px.TrainerSetup(
        config=config,
        policy_specs=policy_specs(config),
        tasks=px.TaskSources(train_dataset=task_ids_to_dataset([f"train:{index}" for index in range(16)])),
        workflow_builder=lambda policies: px.Workflows(checks._OneDatumGroups(), None),
    )
    service = checks.FakePolicyService()
    with checks._faked(service), checks._restored_signals():
        asyncio.run(checks._run(PlatoonTinkerRLTrainer.from_setup(setup)))
    run = log_path / "quickstart" / "guess-qwen3-4b-replay-sft"
    return service, [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]


def test_the_example_config_validates(capsys):
    cli.main(["config", "validate", str(EXAMPLE / "train.yaml")])

    assert capsys.readouterr().out.startswith("ok ")


def test_the_hook_replays_at_every_other_step_s_end_with_its_own_learning_rate_and_the_step_publishes_once(tmp_path):
    service, rows = train(tmp_path, steps=4)

    # Each step's policy-gradient update; at the ends of steps 1 and 3, a cross-entropy update of 4 conversations.
    assert service.losses == ["cispo", "cispo", "cross_entropy", "cispo", "cispo", "cross_entropy"]
    assert service.calls.count("forward_backward 4") == 2
    rl_rate = load_config(EXAMPLE / "train.yaml", f"log_path={tmp_path}").train.optimizer.learning_rate
    assert [params.learning_rate for params in service.adam] == [rl_rate, rl_rate, 1e-5, rl_rate, rl_rate, 1e-5]
    # One publication a step, whatever its updates: 4 versions, 6 optimizer steps.
    assert service.calls.count("save_weights_and_get_sampling_client") == 4
    assert [(row["name"], row["policy_version"], row["optimizer_steps"]) for row in rows] == [
        ("recovery-000000", 0, 0),
        ("final", 4, 6),
    ]
    assert rows[-1]["replay_sft"] == {"position": 0}  # 8 conversations replayed, all of them


def test_a_resumed_run_goes_on_with_the_next_conversations(tmp_path):
    train(tmp_path, steps=2)
    service, rows = train(tmp_path, steps=4)

    # The second start resumes from the first's final row, whose cursor its launch checkpoint saves again.
    assert [(row["name"], row["batch"], row["replay_sft"]) for row in rows] == [
        ("recovery-000000", 0, {"position": 0}),
        ("final", 2, {"position": 4}),
        ("recovery-000002", 2, {"position": 4}),
        ("final", 4, {"position": 0}),
    ]
    assert service.losses == ["cispo", "cispo", "cross_entropy"]
