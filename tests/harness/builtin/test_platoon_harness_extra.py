"""The ``platoon-harness`` harness and program without the ``platoon-harness`` extra: both stay registered, and naming
them fails loudly with the install command, when the config loads and when a rollout runs.  The tests that run them
are in ``teams/``, which needs the extra."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml

from platoon import cli
from platoon.harness.builtin.platoon_harness import INSTALL
from platoon.harness.registry import available_harnesses
from platoon.programs import available_programs
from platoon.testing import ScriptedPolicy, run_episode

ROWS = {"rows": [{"id": "t1", "goal": "say hi"}], "splits": {"train": ["t1"]}}
ENTRY = {
    "service": "table",
    "service_kwargs": ROWS,
    "harness": "platoon-harness",
    "program": "platoon-harness",
    "harness_options": {"team": "crew.toml"},
}


@pytest.fixture(autouse=True)
def without_the_extra(monkeypatch):
    """platoon-harness as if it were not installed: importing it raises ImportError."""

    monkeypatch.setitem(sys.modules, "platoon_harness", None)


def test_both_are_listed_without_the_extra():
    assert "platoon-harness@1" in available_harnesses() and "platoon-harness@1" in available_programs()


def test_the_program_refuses_an_entry_naming_the_extra(tmp_path, capsys, monkeypatch):
    (tmp_path / "crew.toml").write_text("")
    config = tmp_path / "train.yaml"
    train = {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "batch_size": 1, "max_training_steps": 1}
    stats = {"experiment_name": "e", "trial_name": "t"}
    config.write_text(
        yaml.safe_dump({"train": train, "log_path": str(tmp_path / "logs"), "stats": stats, "environments": [ENTRY]})
    )
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")

    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", str(config)])

    captured = capsys.readouterr()
    assert exited.value.code == cli.EXIT_CONFIG
    assert f"environments.0.program (train): {INSTALL}" in captured.out + captured.err
    assert not Path(tmp_path / "logs").exists()


def test_the_harness_fails_its_rollout_naming_the_extra():
    run = asyncio.run(
        run_episode(
            "t1",
            service="table",
            service_kwargs=ROWS,
            harness="platoon-harness",
            harness_options={"team": "crew.toml"},
            policy=ScriptedPolicy(lambda prompt: "done"),
            max_steps=5,
        )
    )

    assert run.outcome["status"] == "error"
    assert INSTALL in run.root.misc["agent_result"]["error"]
