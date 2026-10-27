"""The hosted-models guide (``docs/how-to/evaluate-hosted-models.md``) stays true: its configs validate, its
``lock.json`` example has the keys platoon writes, and the facts and messages it quotes are still platoon's."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from platoon import cli
from platoon.eval.config_defs import EvalConfig, EvalPolicy
from platoon.gateway import hosted

REPO = Path(__file__).resolve().parents[2]
GUIDE = (REPO / "docs" / "how-to" / "evaluate-hosted-models.md").read_text()
_YAML_BLOCK = re.compile(r"```yaml\n# (?P<name>[\w.-]+\.yaml)\n(?P<body>.*?)```", re.S)
BLOCKS = {match["name"]: match["body"] for match in _YAML_BLOCK.finditer(GUIDE)}


def test_the_guide_s_evaluation_config_validates(tmp_path, capsys):
    path = tmp_path / "eval-hosted.yaml"
    path.write_text(BLOCKS["eval-hosted.yaml"])

    cli.main(["config", "validate", str(path)])

    assert f"ok   {path} (PlatoonEvalConfig)" in capsys.readouterr().out


@pytest.mark.parametrize("server", sorted(yaml.safe_load(BLOCKS["hosted.yaml"])))
def test_each_server_s_settings_validate_in_the_evaluation_config(tmp_path, capsys, server):
    section = yaml.safe_load(BLOCKS["hosted.yaml"])[server]
    config = yaml.safe_load(BLOCKS["eval-hosted.yaml"])
    config["inference"].update(section)
    path = tmp_path / f"{server}.yaml"
    path.write_text(yaml.safe_dump(config))

    cli.main(["config", "validate", str(path)])

    assert "ok" in capsys.readouterr().out
    policy = EvalConfig(model_name=section["model_name"], hosted=section["hosted"]).policy()
    assert policy is not None and policy.is_hosted


def test_the_lock_example_has_the_keys_platoon_writes():
    example = re.search(r"```json\n(\"policy\": \{.*?\n\})\n```", GUIDE, re.S)
    assert example, "the guide shows lock.json's policy"
    shown = json.loads("{" + example[1] + "}")["policy"]
    section = yaml.safe_load(BLOCKS["eval-hosted.yaml"])["inference"]
    policy = EvalConfig(model_name=section["model_name"], hosted=section["hosted"]).policy()
    assert policy is not None and policy.lock_record(shown["probe"]) == shown


def test_the_guide_s_forwarding_table_is_the_gateway_s():
    for dialect, path in hosted.PATHS.items():
        assert f"`POST {{base_url}}{path}`" in GUIDE, dialect
    for headers in hosted.FORWARDED_HEADERS.values():
        assert ", ".join(f"`{header}`" for header in headers) in GUIDE


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        ("platoon/eval/config_defs.py", 'verb = "refused" if exc.status is not None else "did not answer"'),
        ("platoon/eval/config_defs.py", "configure a Tinker policy, but inference.hosted is"),
        ("platoon/eval/config_defs.py", "which api_key_env names"),
        ("platoon/eval/workflow.py", "backend_unavailable: {failed} model call(s) failed after retries"),
        ("platoon/eval/workflow.py", 'return f"backend_refused: {fatal}"'),
        ("platoon/eval/runner.py", "holds rollouts made with"),
        ("platoon/eval/runner.py", "the evaluation stopped after"),
        ("platoon/utils/judges.py", 'JUDGES_OFF = "judges_off"'),
        ("platoon/gateway/hosted.py", "RETRY_BACKOFF_S = (0.5, 8.0)"),
        ("platoon/gateway/hosted.py", "MAX_RETRY_AFTER_S = 60.0"),
        ("platoon/gateway/hosted.py", 'REDACTED = "[redacted]"'),
        ("platoon/gateway/server.py", "KEEPALIVE_SECONDS = 10.0"),
        ("platoon/train/config_defs.py", "serve evaluation only"),
    ],
)
def test_the_guide_s_facts_are_platoon_s(source, fragment):
    assert fragment in (REPO / source).read_text(), f"{source} no longer has {fragment!r}"


def test_a_tinker_policy_is_still_the_default():
    policy = EvalConfig(model_name="m", renderer_name="qwen3").policy()
    assert isinstance(policy, EvalPolicy) and not policy.is_hosted
    assert policy.lock_record()["backend"] == "tinker"
