"""examples/harbor-dataset: a configuration-only project trains on a Harbor dataset as its README and the README's
"Train on a Harbor dataset" say: a site profile, ``platoon prepare``, then ``platoon rollout`` and ``platoon train`` of
the one config.

CPU only.  The site profile gives the cache and the run directories, in the test's own directory.  The commands that
start sandboxes run them on the ``process`` provider, which a run config may name on the command line (a profile may
not): host directories, no container.  The policy is a script behind the real model gateway (``platoon.testing``'s
toy tokenizer under the config's renderer) that answers each task with its reference solution, through platoon-bash's
``bash`` tool, as a model would; ``platoon train`` stops where the trainer would open the Tinker service.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from platoon import cli  # noqa: E402
from platoon.testing import serve_policy, tool_call  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "examples" / "harbor-dataset"
CONFIG = EXAMPLE / "train.yaml"
DATASET = EXAMPLE / "dataset"
TASKS = sorted(path.name for path in DATASET.iterdir() if (path / "task.toml").is_file())
HELD_OUT = {"sales-best-month", "sales-revenue"}  # the topic seed 0 holds out, as the example's README says


def _command(task: str) -> str:
    """A task's reference solution as one shell command: what the scripted policy runs."""

    lines = (DATASET / task / "solution" / "solve.sh").read_text().splitlines()
    return "\n".join(line for line in lines if line and not line.startswith("#"))


def _answer(prompt: str) -> str:
    """The scripted policy: run the reference solution of the task the prompt asks about, then say it is done."""

    if "<tool_response>" in prompt:
        return "I wrote answer.txt."
    for task in TASKS:
        if (DATASET / task / "instruction.md").read_text().strip() in prompt:
            return tool_call("bash", command=_command(task))
    raise AssertionError("the prompt holds no task's instruction")


def _quoted(page: Path, after: str) -> list[str]:
    """The lines of the first ``text`` block that follows the line ``after`` on ``page``."""

    text = page.read_text()
    start = text.index("```text\n", text.index(after)) + len("```text\n")
    return text[start : text.index("```", start)].splitlines()


def _exit_code(argv: list[str]) -> int:
    try:
        cli.main(argv)
    except SystemExit as exit_info:
        return int(exit_info.code or 0)
    return 0


@pytest.fixture
def site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The site profile of the README's first step, with the cache and the runs in the test's directory; a Hugging
    Face cache of the test's own, into which a tokenizer load puts a file instead of downloading the model's."""

    from huggingface_hub import constants

    profile = tmp_path / "site.yaml"
    profile.write_text(f"cache: {tmp_path / 'cache'}\nruns: {tmp_path / 'runs'}\n")
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    for variable in ("PLATOON_HARBOR_CACHE", "PLATOON_TOOLCHAINS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    def build_renderer(model: str, *args: object) -> None:
        blob = tmp_path / "hub" / f"models--{model.replace('/', '--')}" / "blobs" / "tokenizer"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text("{}")

    monkeypatch.setattr("platoon.train.tinker.proxy.build_renderer", build_renderer)
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    return tmp_path


@pytest.fixture
def prepared(site: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """The site once the README's ``platoon prepare`` ran: the dataset fetched into the cache and locked."""

    assert _exit_code(["prepare", str(CONFIG)]) == 0
    capsys.readouterr()
    return site


def _process(site: Path) -> list[str]:
    """The overrides that run the sandboxes as host directories: the process provider."""

    return [
        "environments.0.service_kwargs.provider=process",
        f"environments.0.service_kwargs.provider_options={{root: {site / 'sandboxes'}}}",
    ]


def test_the_readme_shows_the_example_config_as_it_is() -> None:
    readme = (REPO / "README.md").read_text()
    marker = "# examples/harbor-dataset/train.yaml\n"
    block = readme[readme.index(marker) + len(marker) :]
    config = CONFIG.read_text()

    assert block[: block.index("```")] == config[config.index("environments:") :]


def test_the_config_names_no_site_path_and_validates_with_a_site_profile(
    site: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    cli.main(["config", "validate", str(CONFIG)])
    assert capsys.readouterr().out == f"ok   {CONFIG} (PlatoonTinkerRLTrainerConfig)\n"

    monkeypatch.delenv("PLATOON_SITE")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(site / "no-profile"))
    assert _exit_code(["config", "validate", str(CONFIG)]) == 78
    out = capsys.readouterr().out
    assert "log_path: required key is missing" in out and "or runs in the site profile" in out


@pytest.mark.parametrize("task", TASKS)
def test_each_task_scores_1_solved_and_0_untouched(task: str, tmp_path: Path) -> None:
    """The dataset is sound: each task's reference solution scores 1 and doing nothing 0 (checked on the host)."""

    def grade(solve: bool) -> str:
        root = tmp_path / ("solved" if solve else "untouched")
        (root / "logs" / "verifier").mkdir(parents=True)
        shutil.copytree(DATASET / task / "environment", root / "app")
        env = {**os.environ, "PLATOON_SANDBOX_ROOT": str(root)}
        if solve:
            subprocess.run(["bash", str(DATASET / task / "solution" / "solve.sh")], cwd=root / "app", check=True)
        tests = DATASET / task / "tests" / "test.sh"
        subprocess.run(["bash", str(tests)], cwd=root / "app", env=env, check=True, capture_output=True)
        return (root / "logs" / "verifier" / "reward.txt").read_text().strip()

    assert grade(solve=True) == "1"
    assert grade(solve=False) == "0"


def test_prepare_locks_the_dataset_and_holds_out_one_topic(site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from platoon.envs.harbor import HarborEnvironmentService

    assert _exit_code(["prepare", str(CONFIG)]) == 0

    out, err = capsys.readouterr()
    assert f"platoon: site profile {site / 'site.yaml'}: cache {site / 'cache'}, runs {site / 'runs'}" in err
    locks = sorted((site / "cache" / "harbor" / "locks").glob("*.lock.json"))
    assert len(locks) == 1
    shown = out.replace(str(DATASET), ".../examples/harbor-dataset/dataset")
    shown = shown.replace(str(locks[0]), "~/.cache/platoon/harbor/locks/<hash>.lock.json")
    assert shown.splitlines() == _quoted(EXAMPLE / "README.md", "### 2. Prepare the config")

    entry = yaml.safe_load(CONFIG.read_text())["environments"][0]
    service = HarborEnvironmentService(**{**entry["service_kwargs"], "dataset": str(DATASET), "provider": "process"})
    assert set(asyncio.run(service.list_tasks("eval"))) == HELD_OUT
    assert set(asyncio.run(service.list_tasks("train"))) == set(TASKS) - HELD_OUT

    assert _exit_code(["prepare", str(CONFIG)]) == 0  # a directory is fetched again: its files may have changed
    lines = capsys.readouterr().out.splitlines()
    assert "dataset: fetched" in lines[0] and "again (a local directory's files may have changed)" in lines[0]
    assert lines[-1] == "ready: 1 made now, 1 already current"


def test_a_rollout_solves_a_task_in_a_process_sandbox(prepared: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with serve_policy(_answer) as policy:
        cli.main(
            [
                "rollout",
                str(CONFIG),
                "--task",
                "towns-largest",
                "--output-dir",
                str(prepared / "r"),
                *_process(prepared),
            ]
        )

    tree = re.sub(r"(towns-largest-|policy )[0-9a-f]+", r"\1<id>", capsys.readouterr().out)
    assert tree.splitlines() == _quoted(EXAMPLE / "README.md", "### 3. Check one task")
    assert len(policy.calls) == 2


@pytest.mark.parametrize(("harness", "reward"), [("platoon-oracle", "1.0"), ("platoon-noop", "0.0")])
def test_the_oracle_scores_a_task_1_and_doing_nothing_0(
    prepared: Path, capsys: pytest.CaptureFixture[str], harness: str, reward: str
) -> None:
    with serve_policy(_answer) as policy:
        cli.main(
            [
                "rollout",
                str(CONFIG),
                "--task",
                "towns-largest",
                "--output-dir",
                str(prepared / harness),
                *_process(prepared),
                f"environments.0.harness={harness}",
            ]
        )

    assert f"reward {reward}  outcome scored reward={reward}" in capsys.readouterr().out
    assert policy.calls == []


class Reached(Exception):
    """Raised where the trainer would open the Tinker service."""


def test_train_runs_under_the_profiles_runs_with_six_tasks_and_two_held_out(
    site: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets: list[tuple[int, int]] = []

    def trainer(*, config, train_dataset, eval_dataset, train_order=None):
        datasets.append((len(train_dataset), len(eval_dataset)))
        raise Reached

    monkeypatch.setattr("platoon.train.tinker.train.PlatoonTinkerRLTrainer", trainer)

    with pytest.raises(Reached):
        cli.main(["train", str(CONFIG), *_process(site)])

    assert datasets == [(6, 2)]
    run_dir = site / "runs" / "harbor-dataset" / "qwen3-4b"
    snapshot = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    assert snapshot["log_path"] == str(site / "runs")
    assert snapshot["environments"][0]["service_kwargs"]["dataset"] == str(DATASET)
    assert f"platoon: from the site profile: log_path={site / 'runs'}" in capsys.readouterr().err


def test_a_harbor_dataset_needs_no_ors_extra(
    prepared: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """platoon[tinker] alone runs a Harbor dataset (docs/getting-started/install.md): nothing on the way imports the
    OpenReward SDK."""

    for name in [name for name in sys.modules if name == "openreward" or name.startswith("openreward.")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "openreward", None)

    with serve_policy(_answer):
        cli.main(
            ["rollout", str(CONFIG), "--task", "log-errors", "--output-dir", str(prepared / "r"), *_process(prepared)]
        )

    assert "reward 1.0  outcome scored reward=1.0" in capsys.readouterr().out
