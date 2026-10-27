"""``platoon config show`` and ``platoon config validate`` say what each environment entry runs with: every per-group
key's effective value, the layer it came from and the value that layer replaced (``platoon.train.entry_settings``)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from platoon import cli

FIXTURE = Path(__file__).parent / "fixtures" / "entry_settings"
CONFIG = FIXTURE / "train.yaml"


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def test_config_show_prints_each_entry_s_settings_after_a_config_that_still_loads(capsys):
    """The blocks follow the config as comments, so the output still loads as the config, and they match the golden
    output, layers included."""

    cli.main(["config", "show", str(CONFIG)])

    out = capsys.readouterr().out
    config, _, blocks = out.partition("# The effective settings")
    assert yaml.safe_load(out) == yaml.safe_load(config)
    assert yaml.safe_load(out)["environments"][1]["max_staleness"] == 48
    assert "# The effective settings" + blocks == (FIXTURE / "show.txt").read_text()


def test_config_show_entry_prints_that_entry_s_blocks_alone(capsys):
    cli.main(["config", "show", str(CONFIG), "--entry", "slow-tasks"])

    assert capsys.readouterr().out == (FIXTURE / "show-entry.txt").read_text()


def test_config_show_entry_names_the_entries_when_none_has_that_name(capsys):
    assert _exit_code(["config", "show", str(CONFIG), "--entry", "nope"]) == cli.EXIT_CONFIG

    assert (
        "--entry nope: the config has no environment entry of that name; its entries are quick-tasks, slow-tasks"
        in (capsys.readouterr().err)
    )


def test_config_show_of_entries_that_set_nothing_prints_the_config_alone(tmp_path, capsys):
    data = yaml.safe_load(CONFIG.read_text())
    for entry in data["environments"]:
        for key in ("workflow_config", "max_staleness", "eval_overrides"):
            entry.pop(key, None)
    data.pop("checkpoint")
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))

    cli.main(["config", "show", str(path)])

    assert "# " not in capsys.readouterr().out


def test_config_validate_says_what_each_entry_changes_from_the_run(tmp_path, capsys):
    cli.main(["config", "validate", "--no-resolve", str(CONFIG)])

    assert capsys.readouterr().out.splitlines() == [
        f"ok   {CONFIG} (PlatoonTinkerRLTrainerConfig)",
        "  environments[quick-tasks]: rollout_config.max_steps 30, in evaluation: rollout_config.max_steps 30",
        "  environments[slow-tasks]: group_size 4, leave_one_out_baseline true, min_successful_group_size 3, "
        "rollout_config.max_steps null, rollout_config.timeout null, max_staleness 48, in evaluation: group_size 1, "
        "rollout_config.max_steps null, rollout_config.timeout null",
    ]

    data = yaml.safe_load(CONFIG.read_text())
    data["train"]["rollout_group_limits"] = {}
    data["environments"][1]["max_staleness"] = 2
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data))

    assert _exit_code(["config", "validate", "--no-resolve", str(broken)]) == cli.EXIT_CONFIG
    out = capsys.readouterr().out
    assert f"FAIL {broken}" in out
    assert "  environments[slow-tasks].max_staleness: 2 is below train.max_staleness: 3, which bounds how far" in out


def test_config_validate_warns_of_a_raised_bound_without_a_limit(tmp_path, capsys):
    data = yaml.safe_load(CONFIG.read_text())
    data["train"]["rollout_group_limits"] = {}
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))

    cli.main(["config", "validate", "--no-resolve", str(path)])

    assert "  warning: environments[slow-tasks].max_staleness: 48 raises the run's bound (3)" in capsys.readouterr().out


def test_platoon_rollout_runs_a_task_with_its_entry_s_rollout_settings():
    from platoon.train.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    config = resolve_config(CONFIG, [], PlatoonTinkerRLTrainerConfig).config

    quick = cli._entry_rollout_config(config, "train", "quick-tasks::a")
    slow = cli._entry_rollout_config(config, "train", "slow-tasks::b")
    assert (quick.max_steps, quick.timeout) == (30, 3600)
    assert (slow.max_steps, slow.timeout) == (None, None)
    assert cli._entry_rollout_config(config, "eval", "quick-tasks::a").max_steps == 30  # its evaluations inherit it
    assert cli._entry_rollout_config(config, "train", "unlabelled") is None  # a mixture's task names its entry


# --- config validate --tasks: what listing the tasks finds, as platoon train finds it ---------------------------------


def load_labelled(config, split, *, prefix, count, eval_label="easy"):
    """``count`` tasks labelled by difficulty: ``easy`` in training, ``eval_label`` in evaluation."""

    label = eval_label if split == "eval" else "easy"
    return [{"task_id": f"{prefix}{index}", "difficulty": label} for index in range(count)]


async def rollout(task, config) -> dict:
    return {}


def _labelled_config(tmp_path: Path) -> Path:
    entry = {"dataset_loader": f"{__name__}:load_labelled", "rollout": f"{__name__}:rollout"}
    data = {
        "train": {
            "model_name": "m",
            "renderer_name": "r",
            "batch_size": 4,
            "max_training_steps": 2,
            "label_by": "difficulty",
        },
        "eval": {
            "every": 10,
            "workflow_config": {"group_size": 4, "straggler_policies": {"hard": {"timeout_seconds": 60, "quorum": 3}}},
        },
        "log_path": str(tmp_path / "logs"),
        "environments": [
            {
                "name": "quick",
                **entry,
                "dataset_kwargs": {"prefix": "q", "count": 8},
                "eval_dataset_kwargs": {"prefix": "qe", "count": 2},
            },
            {
                "name": "slow",
                **entry,
                "dataset_kwargs": {"prefix": "s", "count": 8},
                "eval_dataset_kwargs": {"prefix": "se", "count": 2, "eval_label": "hard"},
                "eval_overrides": {"workflow_config": {"group_size": 2}},
            },
        ],
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_validate_tasks_checks_the_labels_of_the_evaluation_tasks_too(tmp_path, capsys):
    """A label's straggler quorum above the evaluation group size of an entry whose evaluation tasks alone carry the
    label: platoon train refuses it once it lists the tasks, and so does config validate --tasks, which lists the
    evaluation tasks when the run evaluates."""

    path = _labelled_config(tmp_path)

    assert cli.config_validate(path) == 0  # another label column's labels are known once the tasks are listed
    assert cli.config_validate(path, tasks=True) == cli.EXIT_CONFIG
    assert (
        "  eval.workflow_config.straggler_policies.hard.quorum: 3 is more than the 2 rollouts of a group of "
        "environments[slow], whose tasks carry the label" in capsys.readouterr().out
    )
    assert cli.config_validate(path, "eval.every=0", tasks=True) == 0  # no evaluation: its tasks are not listed


def test_config_validate_tasks_warns_of_rollout_permits_the_capped_labels_can_fill(tmp_path, capsys):
    path = _labelled_config(tmp_path)
    capped = ["eval.every=0", "train.rollout_group_limits={easy: 2}", "train.max_concurrent_rollouts=15"]

    assert cli.config_validate(path, *capped, tasks=True) == 0
    out = capsys.readouterr().out
    assert (
        "  warning: train.max_concurrent_rollouts: 15 rollouts, of which the groups of the labels with a limit" in out
    )
    assert "(easy: 2 × 8): a group of theirs can then wait for another of theirs to end" in out
    assert out.rstrip().endswith("raise it to 16 at least")
