"""A mixture refuses ORS entries that would run another entry's environment.

An env server answers requests that name no variant with its default environment, whatever the URI's env name
says, so two entries with different env names and no variant that reach one server would both run the same
environment.  That happens silently when an entry's own pool variable is missing and it falls back to a process-wide
pool (the legacy ``OPENREWARD_SESSION_URLS``, ``OPENREWARD_SESSION_URL`` or ``localhost:8080``) that serves another
entry.  ``platoon train``, ``eval`` and ``rollout`` check before they contact any env server.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest
import yaml

from platoon import cli
from platoon.eval.run import arun_eval
from platoon.train.auto import check_mixture_pools
from platoon.train.components import normalize_environment_configs
from platoon.train.tinker.train import arun_tinker_training
from platoon.utils.config import ConfigError

POOL = "http://10.0.0.1:8080"
POOL_VARIABLES = (
    "OPENREWARD_SESSION_URLS",
    "OPENREWARD_SESSION_URL",
    "OPENREWARD_SESSION_URLS_COUNTERENV",
    "OPENREWARD_SESSION_URLS_CRAFTENV",
)
# Two entries on one server that nobody serves: a check that ran after listing tasks would fail on the connection.
COLLIDING = [
    {"name": "counter", "service": "ors+http://127.0.0.1:9/CounterEnv", "rollout": f"{__name__}:rollout"},
    {"name": "craft", "service": "ors+http://127.0.0.1:9/CraftEnv", "rollout": f"{__name__}:rollout"},
]


async def rollout(task, config):  # the entries' rollout; never run here
    raise AssertionError("not called")


def _check(*entries: dict) -> None:
    check_mixture_pools(normalize_environment_configs(list(entries)))


@pytest.fixture(autouse=True)
def _no_pools(monkeypatch):
    for name in POOL_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("craft_first", [False, True])
def test_an_entry_that_falls_back_onto_another_entrys_servers_is_refused(monkeypatch, craft_first):
    monkeypatch.setenv("OPENREWARD_SESSION_URLS_COUNTERENV", POOL)
    monkeypatch.setenv("OPENREWARD_SESSION_URLS", POOL)  # the legacy process-wide pool: CounterEnv's servers
    entries = [
        {"name": "counter", "service": "ors+http:///CounterEnv"},
        {"name": "craft", "service": "ors+http:///CraftEnv"},  # its OPENREWARD_SESSION_URLS_CRAFTENV is missing
    ]

    with pytest.raises(ConfigError) as refused:
        _check(*(reversed(entries) if craft_first else entries))

    message = str(refused.value)
    assert "'counter' (CounterEnv)" in message and "'craft' (CraftEnv)" in message
    assert f"both reach the env server {POOL} without a variant" in message
    assert "'craft' took its pool from OPENREWARD_SESSION_URLS, since OPENREWARD_SESSION_URLS_CRAFTENV is unset" in (
        message
    )
    assert "'counter' took" not in message  # its pool is its own
    assert "legacy_global_pool: false" in message and "?variant=" in message


@pytest.mark.parametrize("fallback", ["OPENREWARD_SESSION_URL", None])
def test_entries_that_both_fall_back_to_the_single_url_are_refused(monkeypatch, fallback):
    if fallback is not None:
        monkeypatch.setenv(fallback, POOL)
    with pytest.raises(ConfigError, match="took its pool from " + (fallback or "http://localhost:8080")):
        _check(
            {"name": "counter", "service": "ors+http:///CounterEnv", "service_kwargs": {"legacy_global_pool": False}},
            {"name": "craft", "service": "ors+http:///CraftEnv", "service_kwargs": {"legacy_global_pool": False}},
        )


def test_two_env_names_on_one_server_without_a_variant_are_refused():
    with pytest.raises(ConfigError, match="both reach the env server http://10.0.0.1:8080 without a variant"):
        _check(
            {"name": "counter", "service": "ors+http://10.0.0.1:8080/CounterEnv"},
            {"name": "craft", "service": "ors+http://10.0.0.1:8080/CraftEnv"},
        )


def test_entries_that_name_their_environment_or_share_one_load(caplog):
    with caplog.at_level(logging.WARNING, logger="platoon.train.auto"):
        _check(  # one multi-environment server, each entry naming its environment
            {"name": "counter", "service": "ors+http://10.0.0.1:8080/srv?variant=CounterEnv"},
            {"name": "craft", "service": "ors+http://10.0.0.1:8080/srv?variant=CraftEnv"},
        )
        _check(  # the same environment twice, e.g. two task subsets
            {"name": "easy", "service": "ors+http://10.0.0.1:8080/CounterEnv", "service_kwargs": {"task_indices": [0]}},
            {"name": "hard", "service": "ors+http://10.0.0.1:8080/counterenv", "service_kwargs": {"task_indices": [1]}},
        )
        _check(  # hosted deployments, which the hosted service routes by their owner/name
            {"name": "a", "service": "ors+https://sessions.example.com/owner/CounterEnv"},
            {"name": "b", "service": "ors+https://sessions.example.com/owner/CraftEnv"},
        )
        _check(  # each entry with its own pool, and one in process
            {"name": "counter", "service": "ors+http://10.0.0.1:8080/CounterEnv"},
            {"name": "craft", "service": "ors+http://10.0.0.2:8080/CraftEnv"},
            {"name": "local", "service": "python://ors_toys:CraftEnv"},
        )
    assert caplog.text == ""


def test_an_entry_on_a_process_wide_pool_no_other_entry_shares_is_warned_about(monkeypatch, caplog):
    monkeypatch.setenv("OPENREWARD_SESSION_URLS_COUNTERENV", POOL)
    monkeypatch.setenv("OPENREWARD_SESSION_URLS", "http://10.0.0.2:8080")  # serves only CraftEnv
    with caplog.at_level(logging.WARNING, logger="platoon.train.auto"):
        _check(
            {"name": "counter", "service": "ors+http:///CounterEnv"},
            {"name": "craft", "service": "ors+http:///CraftEnv"},
        )
    assert (
        "'craft' (ORS CraftEnv) takes its env servers from OPENREWARD_SESSION_URLS, a process-wide pool, since "
        "OPENREWARD_SESSION_URLS_CRAFTENV is unset or empty" in caplog.text
    )
    assert "'counter'" not in caplog.text


def test_a_single_environment_keeps_the_process_wide_pool(monkeypatch, caplog):
    monkeypatch.setenv("OPENREWARD_SESSION_URLS", POOL)
    with caplog.at_level(logging.WARNING, logger="platoon.train.auto"):
        _check({"name": "craft", "service": "ors+http:///CraftEnv"})
    assert caplog.text == ""


# --- the commands check before they contact an env server ---------------------------------------------------------


def _write(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump({**config, "environments": COLLIDING}))
    return path


def test_training_refuses_the_mixture_before_listing_its_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # past the key check, which training runs first
    config = _write(
        tmp_path,
        {
            "train": {
                "model_name": "Qwen/Qwen3-4B",
                "renderer_name": "qwen3",
                "batch_size": 2,
                "max_training_steps": 1,
            },
            "eval": {"every": 0},
            "log_path": str(tmp_path / "logs"),
        },
    )
    with pytest.raises(ConfigError, match="both reach the env server http://127.0.0.1:9 without a variant"):
        asyncio.run(arun_tinker_training(["--config", str(config)]))


def test_evaluation_refuses_the_mixture_before_listing_its_tasks_but_a_report_needs_no_pools(tmp_path):
    inference = {"model_name": "m", "output_dir": str(tmp_path / "results")}
    config = _write(tmp_path, {"inference": inference})
    with pytest.raises(ConfigError, match="both reach the env server http://127.0.0.1:9 without a variant"):
        asyncio.run(arun_eval(["--config", str(config)]))
    assert not list((tmp_path / "results").rglob("metadata.json"))

    report = asyncio.run(arun_eval(["--config", str(_write(tmp_path, {"inference": inference, "stage": "report"}))]))
    assert report["summary"]["total_rollouts"] == 0


def test_a_rollout_refuses_the_mixture_before_loading_its_task(tmp_path, capsys):
    config = _write(tmp_path, {"inference": {"model_name": "m", "output_dir": str(tmp_path / "results")}})
    with pytest.raises(SystemExit) as exited:
        cli.main(["rollout", str(config), "--task", "craft::train:0", "--output-dir", str(tmp_path / "debug")])
    assert exited.value.code == cli.EXIT_CONFIG
    assert "both reach the env server http://127.0.0.1:9 without a variant" in capsys.readouterr().err
