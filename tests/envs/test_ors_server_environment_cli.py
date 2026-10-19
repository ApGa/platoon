"""``platoon train`` and ``platoon eval`` list their tasks at startup, which is where they meet an ORS env name that
its env server would answer with another environment: a configuration error, reported like the other refusals of
the run commands (exit 78 and one line naming the environment), not a traceback with exit 1."""

from __future__ import annotations

import logging

import ors_toys
import pytest
import yaml
from test_ors_remote import _serve

from platoon import cli

MODEL = {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "renderer_name": "qwen3_instruct"}


def _config(tmp_path, command: str) -> dict:
    if command == "train":
        return {
            "train": {**MODEL, "batch_size": 1, "max_training_steps": 1},
            "eval": {"every": 0},
            "log_path": str(tmp_path / "logs"),
        }
    return {"inference": {**MODEL, "output_dir": str(tmp_path / "results")}}


@pytest.mark.parametrize("command", ["train", "eval"])
def test_a_non_default_environment_without_a_variant_exits_78(tmp_path, capsys, monkeypatch, command):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")  # past the key check, to the task listing
    with _serve([ors_toys.CounterEnv, ors_toys.CraftEnv]) as served:
        environment = {"service": f"ors+http://{served.hostport}/CraftEnv", "harness": "platoon-react"}
        path = tmp_path / f"{command}.yaml"
        path.write_text(yaml.safe_dump({**_config(tmp_path, command), "environments": [environment]}))

        with pytest.raises(SystemExit) as exited:
            cli.main([command, str(path)])

    assert exited.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert f"environments[default].service: The env server http://{served.hostport} serves" in err
    assert "select it with ?variant=CraftEnv" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("command", ["train", "eval"])
def test_an_env_class_that_the_server_does_not_serve_exits_78(tmp_path, capsys, monkeypatch, command):
    monkeypatch.setenv("TINKER_API_KEY", "tml-test")
    # The test server's own report of a client that hung up mid-stream is not the command's output (uvicorn configures
    # its loggers when the server starts, so the method is silenced, not the logger).
    monkeypatch.setattr(logging.getLogger("uvicorn.error"), "error", lambda *args, **kwargs: None)
    with _serve([ors_toys.CounterEnv]) as served:
        environment = {
            "service": f"ors+http://{served.hostport}/CounterEnv",
            "service_kwargs": {"env_class": "ors_toys:CraftEnv"},
            "harness": "platoon-react",
        }
        path = tmp_path / f"{command}.yaml"
        path.write_text(yaml.safe_dump({**_config(tmp_path, command), "environments": [environment]}))

        with pytest.raises(SystemExit) as exited:
            cli.main([command, str(path)])

    assert exited.value.code == cli.EXIT_CONFIG
    err = capsys.readouterr().err
    assert f"environments[default].service: The env server http://{served.hostport} serves ORS env 'CounterEnv'" in err
    assert "with other tools than env_class ors_toys.CraftEnv" in err and "Traceback" not in err
