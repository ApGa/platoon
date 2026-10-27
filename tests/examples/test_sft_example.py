"""The sft example (``examples/sft``) trains as its README says, on CPU: ``platoon train`` runs in process on each
config, on platoon.testing's fake Tinker service, whose policy renders with the toy tokenizer under the configs'
``qwen3_instruct`` renderer.  One config trains on an alpaca dataset, the other on a sharegpt one with tool calls,
both named by the example's ``dataset_info.json``, in LLaMA-Factory's conventions."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from platoon import cli
from platoon.testing import FakePolicyService, fake_tinker_service, toy_tokenizer


@pytest.fixture(autouse=True)
def _datasets_cache(tmp_path, monkeypatch):
    """``datasets.load_dataset`` caches what it loads: under the test's directory, not the user's cache."""

    import datasets.config

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "sft"


def train(config: str, root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakePolicyService, Path]:
    """``platoon train examples/sft/<config>.yaml``, run from ``root``, on a fake Tinker service; the service, and the
    run directory under ``root`` where the README says each run writes."""

    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    written = re.search(r"Each run writes to `([^`<]+)/<alpaca\|sharegpt>`", readme)
    assert written is not None
    monkeypatch.chdir(root)
    with fake_tinker_service() as service:
        cli.main(["train", str(EXAMPLE / f"{config}.yaml")])
    return service, root / written.group(1) / config


def test_the_example_configs_validate_and_list_their_datasets(capsys):
    cli.main(["config", "validate", "examples/sft/alpaca.yaml", "examples/sft/sharegpt.yaml"])
    assert capsys.readouterr().out.count("ok") == 2

    cli.main(["config", "validate", "--tasks", "examples/sft/alpaca.yaml", "examples/sft/sharegpt.yaml"])
    out = capsys.readouterr().out
    assert "tasks: trainer: sft rolls out none" in out
    assert "dataset alpaca_demo: 12 conversations" in out and "dataset sharegpt_demo: 8 conversations" in out


def test_config_validate_refuses_a_dataset_that_dataset_info_json_does_not_name(capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "validate", "examples/sft/alpaca.yaml", "trainer_options.dataset=alpaca_dmeo"])

    assert exited.value.code == 78
    assert "has no dataset 'alpaca_dmeo'" in capsys.readouterr().out


def test_platoon_train_fine_tunes_on_the_alpaca_dataset_every_conversation_once_an_epoch(tmp_path, monkeypatch):
    service, run = train("alpaca", tmp_path, monkeypatch)

    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    # 12 conversations, 4 a step: 3 steps an epoch, 2 epochs; a permanent checkpoint every 3 steps.
    assert [(row["name"], row["batch"], row["optimizer_steps"]) for row in rows] == [
        ("recovery-000000", 0, 0),
        ("000003", 3, 3),
        ("000006", 6, 6),
        ("final", 6, 6),
    ]
    texts = Counter(toy_tokenizer().decode(datum.model_input.to_ints()) for datum in service.datums)
    assert len(texts) == 12 and set(texts.values()) == {2}
    lock = json.loads((run / "lock.json").read_text())
    assert lock["trainer"]["name"] == "sft" and lock["trainer"]["options"]["dataset"] == "alpaca_demo"
    assert json.loads((run / "identity.json").read_text())["model"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_platoon_train_fine_tunes_on_the_sharegpt_dataset_and_prunes_its_recovery_checkpoints(tmp_path, monkeypatch):
    service, run = train("sharegpt", tmp_path, monkeypatch)

    rows = [json.loads(line) for line in (run / "checkpoints.jsonl").read_text().splitlines()]
    assert [row["name"] for row in rows] == [f"recovery-{step:06d}" for step in range(7)] + ["final"]
    # 8 conversations, 4 a step in two microbatches, 3 epochs: 6 optimizer steps of two forward_backward calls each.
    assert service.calls.count("optim_step") == 6 and service.calls.count("forward_backward 2") == 12
    # keep_recovery: 2 keeps the newest two recovery checkpoints; no sampler needs an older one.
    assert sorted({path.rsplit("/", 1)[1] for path in service.deleted}) == [f"recovery-{step:06d}" for step in range(5)]
    tool_call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    assert any(tool_call in toy_tokenizer().decode(datum.model_input.to_ints()) for datum in service.datums)
