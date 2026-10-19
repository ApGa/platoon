"""Where ``platoon eval`` writes, and whose rollouts it resumes.

An evaluation that names no ``inference.output_dir`` writes each start to a new directory,
``runs/eval/<policy>/<UTC time>``; one that names a directory resumes the rollouts of its own policy there, and
refuses, before anything in the directory changes, a directory whose rollouts another policy made.

The rollouts run the toy function harness of ``tests/inline`` (``toy-agent``) on its toy service, through the real
gateway with ``platoon.testing``'s scripted policy in place of Tinker.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
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

from platoon import cli  # noqa: E402
from platoon.eval.config_defs import EvalConfig, HostedConfig  # noqa: E402
from platoon.eval.run import new_output_dir, run_eval  # noqa: E402
from platoon.eval.runner import check_provenance  # noqa: E402
from platoon.testing import TOY_MODEL_NAME, rules, serve_policy  # noqa: E402
from platoon.utils.config import ConfigError  # noqa: E402

POLICY = rules({"goal:t0": "SUBMIT ok", "obs: submitted": "FINISH done"})
CHECKPOINT = "tinker://run-1/sampler_weights/000040"
TOY_DIR = "platoon-toy"  # TOY_MODEL_NAME as a directory name
STAMP = re.compile(r"\d{8}T\d{6}Z(-\d+)?")


@pytest.fixture(autouse=True)
def _toy_service():
    ToyService.sessions.clear()
    yield
    ToyService.sessions.clear()


def _config(tmp_path: Path, **inference: Any) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "inference": {
                    "model_name": TOY_MODEL_NAME,
                    "renderer_name": "qwen3_instruct",
                    "workflow": {"num_concurrent_workers": 1, "rollout_config": {"verbose": False, "max_steps": 8}},
                    **inference,
                },
                "environments": [
                    {
                        "name": "toy",
                        "package": "inline_fakes",
                        "service": "inline_fakes:ToyService",
                        "harness": "toy-agent",
                    }
                ],
            }
        )
    )
    return path


def _eval(config: Path, *overrides: str) -> tuple[int, int]:
    """Run ``platoon eval`` in this process: its exit code, and how many calls the policy answered."""

    with serve_policy(POLICY) as policy:
        try:
            cli.main(["eval", str(config), *overrides])
        except SystemExit as exit:
            return int(exit.code or 0), len(policy.calls)
    return 0, len(policy.calls)


def _files(directory: Path) -> dict[str, bytes]:
    """Every file under ``directory``, by its path there, with its bytes."""

    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def _metadata(run: Path) -> dict[str, Any]:
    (path,) = (run / "rollouts").glob("*/rollout_0/metadata.json")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------- the default directory


def test_without_an_output_dir_each_start_writes_a_new_directory_named_after_the_policy(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)

    code, calls = _eval(config)
    assert code == 0 and calls > 0
    assert _eval(config) == (0, calls)  # a start of its own: nothing to resume
    assert _eval(config, f"inference.model_path={CHECKPOINT}") == (0, calls)

    runs = sorted((tmp_path / "runs" / "eval").glob("*/*"))
    assert [run.parent.name for run in runs] == [TOY_DIR, TOY_DIR, "run-1-sampler_weights-000040"]
    assert runs[0] != runs[1] and all(STAMP.fullmatch(run.name) for run in runs)
    err = capsys.readouterr().err
    for run in runs:
        relative = str(run.relative_to(tmp_path))
        assert f"platoon: inference.output_dir is not set, so this evaluation writes to {relative}" in err
        # The record names the directory, absolute as a record's outputs are, so the run that loads it, and a report
        # from it, use the same one from any directory.
        resolved = yaml.safe_load((run / "config.resolved.yaml").read_text())
        assert resolved["inference"]["output_dir"] == str(run)
        assert json.loads((run / "reports" / "final_report.json").read_text())["summary"]["success_rate"] == 1.0
    lock = json.loads((runs[2] / "lock.json").read_text())
    assert lock["policy"]["model_path"] == CHECKPOINT
    assert _metadata(runs[2])["provenance"] == {
        "judges": True,
        "backend": "tinker",
        "model": TOY_MODEL_NAME,
        "model_path": CHECKPOINT,
        "renderer": "qwen3_instruct",
    }


def test_with_a_site_profile_each_start_writes_under_its_runs(tmp_path, monkeypatch, capsys):
    """The site profile's runs holds the new directories, <runs>/eval/<policy>/<UTC time>, which the record says the
    profile gave; the working directory gets none."""

    profile = tmp_path / "site.yaml"
    profile.write_text(yaml.safe_dump({"runs": str(tmp_path / "platoon-runs")}))
    monkeypatch.setenv("PLATOON_SITE", str(profile))
    monkeypatch.chdir(tmp_path)

    code, calls = _eval(_config(tmp_path))
    assert code == 0 and calls > 0
    (run,) = (tmp_path / "platoon-runs" / "eval" / TOY_DIR).iterdir()
    assert STAMP.fullmatch(run.name) and not (tmp_path / "runs").exists()
    assert f"so this evaluation writes to {run}, under the site profile's runs" in capsys.readouterr().err
    assert yaml.safe_load((run / "config.resolved.yaml").read_text())["inference"]["output_dir"] == str(run)
    site = json.loads((run / "lock.json").read_text())["site"]
    assert site["gave"] == {"inference.output_dir": str(run)} and "named_after" not in site
    assert json.loads((run / "reports" / "final_report.json").read_text())["summary"]["success_rate"] == 1.0


def test_a_default_directory_is_resumed_and_reported_by_naming_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    code, calls = _eval(config, "stage=rollouts")
    assert code == 0 and calls > 0
    (run,) = (tmp_path / "runs" / "eval" / TOY_DIR).iterdir()
    assert not (run / "reports").exists()

    assert _eval(config, f"inference.output_dir={run}") == (0, 0)  # every rollout is this policy's: resumed
    assert _eval(config, f"inference.output_dir={run}", "stage=report") == (0, 0)
    assert (run / "reports" / "final_report.json").is_file()
    assert [path.name for path in (tmp_path / "runs" / "eval" / TOY_DIR).iterdir()] == [run.name]


def test_a_report_must_name_the_directory_of_the_rollouts_it_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert _eval(_config(tmp_path), "stage=report") == (78, 0)

    assert (
        "stage: report builds the report of rollouts already on disk: set inference.output_dir to the directory of "
        "the evaluation that made them" in capsys.readouterr().err
    )
    assert not (tmp_path / "runs").exists()


def test_run_eval_chooses_and_checks_the_directory_as_the_command_does(tmp_path, monkeypatch):
    """Callers of :func:`platoon.eval.run.run_eval`, which ``platoon eval`` runs on its snapshot of the config, get a
    directory of their own when the config names none, and the refusal, before ``lock.json`` changes."""

    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path)
    with serve_policy(POLICY):
        report = run_eval(["--config", str(config)])
    assert report["summary"]["success_rate"] == 1.0
    (run,) = (tmp_path / "runs" / "eval" / TOY_DIR).iterdir()
    assert (run / "reports" / "final_report.json").is_file()

    (run / "lock.json").write_text('{"written": "before"}\n')
    with serve_policy(POLICY), pytest.raises(ConfigError, match=re.escape(f"model_path None (now {CHECKPOINT!r})")):
        run_eval(["--config", str(config), f"inference.output_dir={run}", f"inference.model_path={CHECKPOINT}"])
    assert (run / "lock.json").read_text() == '{"written": "before"}\n'


# --------------------------------------------------------------------------- a directory the config names


def test_a_named_directory_resumes_the_rollouts_of_its_own_policy(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = _config(tmp_path, output_dir=str(tmp_path / "results"))

    code, calls = _eval(config)
    assert code == 0 and calls > 0
    assert _eval(config) == (0, 0)

    assert "inference.output_dir is not set" not in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("override", "difference"),
    [
        (f"inference.model_path={CHECKPOINT}", f"model_path None (now {CHECKPOINT!r})"),
        ("inference.model_name=my-org/other-model", f"model {TOY_MODEL_NAME!r} (now 'my-org/other-model')"),
        ("inference.renderer_name=qwen3", "renderer 'qwen3_instruct' (now 'qwen3')"),
    ],
    ids=["checkpoint", "model", "renderer"],
)
def test_a_named_directory_with_another_policy_s_rollouts_is_refused_before_anything_in_it_changes(
    tmp_path, capsys, override, difference
):
    results = tmp_path / "results"
    config = _config(tmp_path, output_dir=str(results))
    assert _eval(config)[0] == 0
    before = _files(results)
    capsys.readouterr()

    assert _eval(config, override) == (78, 0)
    assert _eval(config, override, "inference.resume=false") == (78, 0)  # the report would still read them
    assert _eval(config, override, "stage=report") == (78, 0)  # it would label them as this policy's

    err = capsys.readouterr().err
    assert err.count(f"{results} holds rollouts made with {difference}") == 3
    assert "set a new inference.output_dir, or leave it unset for a new directory of this evaluation's own" in err
    # The record, the rollouts and the report are still those of the policy that made them.
    assert _files(results) == before


# --------------------------------------------------------------------------- the pieces


@pytest.mark.parametrize(
    ("inference", "name"),
    [
        ({"model_name": "Qwen/Qwen3-4B-Instruct-2507"}, "Qwen-Qwen3-4B-Instruct-2507"),
        (
            {
                "model_name": "Qwen/Qwen3-4B-Instruct-2507",
                "model_path": "tinker://model-55dee1f1d7df/sampler_weights/final",
            },
            "model-55dee1f1d7df-sampler_weights-final",
        ),
        ({"model_name": "gpt-5.1", "hosted": {"base_url": "https://api.openai.com/v1"}}, "gpt-5.1"),
        (
            {"model_name": "openai/aws/anthropic/bedrock-claude-opus-4-7"},
            "openai-aws-anthropic-bedrock-claude-opus-4-7",
        ),
        ({"model_name": "../.."}, "policy"),
        ({"model_name": "org/" + "m" * 120}, "m" * 100),
    ],
    ids=["model", "checkpoint", "hosted", "nested-name", "dots-only", "long"],
)
def test_the_policy_s_directory_name(inference, name):
    assert EvalConfig(**inference).policy_dir_name() == name


def test_two_starts_in_one_second_never_share_a_directory(tmp_path):
    inference = EvalConfig(model_name="Qwen/Qwen3-4B")
    now = datetime(2026, 10, 18, 1, 2, 3, tzinfo=timezone.utc)

    runs = [new_output_dir(inference, tmp_path, now=now) for _ in range(3)]

    assert [run.name for run in runs] == ["20261018T010203Z", "20261018T010203Z-2", "20261018T010203Z-3"]
    assert all(run.parent == tmp_path / "Qwen-Qwen3-4B" and run.is_dir() for run in runs)


TINKER = EvalConfig(model_name="m", renderer_name="qwen3").provenance()
NO_POLICY = EvalConfig(model_name="m").provenance()
HOSTED = EvalConfig(model_name="m", hosted=HostedConfig(base_url="https://api.example.com/v1")).provenance()


def _rollout(output_dir: Path, provenance: dict[str, Any] | None) -> Path:
    """An output directory with one rollout whose ``metadata.json`` records ``provenance`` (None: none)."""

    rollout = output_dir / "rollouts" / "t0" / "rollout_0"
    rollout.mkdir(parents=True)
    metadata: dict[str, Any] = {"task_id": "t0", "rollout_index": 0}
    if provenance is not None:
        metadata["provenance"] = provenance
    (rollout / "metadata.json").write_text(json.dumps(metadata))
    return output_dir


def test_rollouts_that_no_policy_made_and_a_policy_s_never_mix(tmp_path):
    assert NO_POLICY == {"judges": True, "backend": None}

    no_policy = _rollout(tmp_path / "no-policy", NO_POLICY)
    check_provenance(no_policy, NO_POLICY)
    with pytest.raises(ConfigError, match=re.escape("backend None (now 'tinker')")):
        check_provenance(no_policy, TINKER)

    policy = _rollout(tmp_path / "policy", TINKER)
    with pytest.raises(ConfigError, match=re.escape("backend 'tinker' (now None)")):
        check_provenance(policy, NO_POLICY)

    # Recorded before a run without a policy said so: it records only its judges, and a policy's run never resumes it.
    earlier = _rollout(tmp_path / "earlier-no-policy", {"judges": True})
    check_provenance(earlier, NO_POLICY)
    for current, now in ((TINKER, "'tinker'"), (HOSTED, "'hosted'")):
        with pytest.raises(ConfigError, match=re.escape(f"backend None (now {now})")):
            check_provenance(earlier, current)


def test_older_records_compare_what_they_recorded(tmp_path):
    """A rollout recorded before the renderer was is compared on the rest; one that records no provenance at all was
    made by a Tinker policy with judges on, the only kind there was then."""

    before_renderer = _rollout(tmp_path / "before-renderer", {key: TINKER[key] for key in TINKER if key != "renderer"})
    check_provenance(before_renderer, EvalConfig(model_name="m", renderer_name="qwen3_instruct").provenance())
    with pytest.raises(ConfigError, match=re.escape(f"model_path None (now {CHECKPOINT!r})")):
        check_provenance(
            before_renderer, EvalConfig(model_name="m", renderer_name="qwen3", model_path=CHECKPOINT).provenance()
        )

    legacy = _rollout(tmp_path / "legacy", None)
    check_provenance(legacy, TINKER)
    with pytest.raises(ConfigError, match=re.escape("backend 'tinker' (now 'hosted')")):
        check_provenance(legacy, HOSTED)
