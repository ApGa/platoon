"""The Tinker key of ``platoon train``, ``eval`` and ``rollout``, and what they say about it.

Without a usable ``TINKER_API_KEY`` the commands that serve a policy through Tinker stop with one line (exit 78, no
traceback) before they prepare, write or reap anything: ``platoon eval`` used to finish with exit 0, every rollout
errored and a success rate of 0.0, ``platoon rollout`` and ``platoon train`` ended in the Tinker client's traceback,
and ``platoon train`` wrote its run directory and reaped the run's sandboxes first.  A hosted model needs the variable
its ``api_key_env`` names instead, and a command that serves no policy needs no key.

A key that the Tinker service refuses (HTTP 401, or 403) ends each of them with one line that names the key and the
service, instead of the client's traceback; an evaluation starts no rollout after it.  The service here is a local
fake that refuses every request, which the real Tinker client talks to over HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import pickle
import signal
import threading
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import yaml

from platoon import cli
from platoon.eval.config_defs import TinkerKeyError, TinkerKeyRefusedError, tinker_key_refused
from platoon.task import Task
from platoon.train import launch as train_module
from platoon.train import reaping, supervise
from platoon.utils.config import ConfigError

LISTED: list[str] = []
MISSING = "platoon: TINKER_API_KEY is not set: the policy is sampled through Tinker, which needs it"
# A key that the fake service refuses, and the answer it refuses it with, which repeats the key.
REFUSED_KEY = "tml-refused-0123456789"
ANSWER = f"no such key: {REFUSED_KEY}"


def load_tasks(config, split, **kwargs) -> list[str]:
    LISTED.append(split)
    return ["a", "b"]


def load_task(task_id: str) -> Task:
    return Task(id=task_id, goal="g")


async def rollout(task: Task, config) -> dict:
    raise AssertionError("no rollout runs without a policy")


async def scored_rollout(task: Task, config) -> dict:
    return {"id": "c", "trajectories": {"root": {"id": "root", "task": {"id": task.id}, "reward": 1.0, "steps": []}}}


ENVIRONMENT = {
    "dataset_loader": f"{__name__}:load_tasks",
    "task_loader": f"{__name__}:load_task",
    "rollout": f"{__name__}:rollout",
}
# Rollouts that sample nothing, for the commands that serve no Tinker policy.
SCORED = {**ENVIRONMENT, "rollout": f"{__name__}:scored_rollout"}
TINKER_POLICY = {"model_name": "m", "renderer_name": "qwen3_instruct"}
# A hosted model on a server nobody serves: a rollout that sampled it would fail.
HOSTED_POLICY = {"model_name": "m", "hosted": {"base_url": "http://127.0.0.1:9/v1"}}


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    LISTED.clear()


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def _eval_config(
    tmp_path, *, inference: dict | None = None, environment: dict = ENVIRONMENT, output_dir: str | None = "out", **top
) -> str:
    """An evaluation config, with ``inference.output_dir`` the directory ``output_dir`` names under ``tmp_path`` (None:
    unset, a new directory under the working directory's ``runs/eval``)."""

    path = tmp_path / "eval.yaml"
    section = {**(inference or TINKER_POLICY), **({"output_dir": str(tmp_path / output_dir)} if output_dir else {})}
    path.write_text(yaml.safe_dump({"inference": section, "environments": [environment], **top}))
    return str(path)


def _train_config(tmp_path) -> str:
    path = tmp_path / "train.yaml"
    settings = {
        "environments": [ENVIRONMENT],
        "train": {"model_name": "m", "renderer_name": "qwen3_instruct", "batch_size": 2, "max_training_steps": 1},
        "eval": {"every": 0},
        "log_path": str(tmp_path / "logs"),
    }
    path.write_text(yaml.safe_dump(settings))
    return str(path)


def _assert_says(err: str, *lines: str) -> None:
    """``err`` holds no traceback, and its lines that name the key are ``lines``."""

    assert "Traceback" not in err
    assert [line for line in err.splitlines() if "TINKER_API_KEY" in line] == list(lines), err


@pytest.fixture
def nothing_runs(monkeypatch):
    """Fail the test on a prepare or a reap, and on training, which the key check must come before."""

    def fail(what: str):
        def refuse(*args, **kwargs):
            pytest.fail(f"{what} before the key was checked")

        return refuse

    monkeypatch.setattr(cli, "_prepare_first", fail("prepared"))
    monkeypatch.setattr(reaping.RunReaper, "reap", fail("reaped"))
    monkeypatch.setattr("platoon.train.supervise.run_training", fail("trained"))


# --------------------------------------------------------------------------------------------- no usable key


def test_train_without_a_key_prepares_writes_and_reaps_nothing(tmp_path, capsys, nothing_runs):
    assert _exit_code(["train", _train_config(tmp_path), "--prepare"]) == cli.EXIT_CONFIG

    _assert_says(capsys.readouterr().err, MISSING)
    assert not (tmp_path / "logs").exists()
    assert LISTED == []


def test_eval_without_a_key_is_one_line_before_any_directory_not_a_zero_success_rate(
    tmp_path, capsys, monkeypatch, nothing_runs
):
    monkeypatch.chdir(tmp_path)

    assert _exit_code(["eval", _eval_config(tmp_path), "--prepare"]) == cli.EXIT_CONFIG
    assert _exit_code(["eval", _eval_config(tmp_path, output_dir=None)]) == cli.EXIT_CONFIG

    _assert_says(capsys.readouterr().err, MISSING, MISSING)
    assert LISTED == []
    assert not (tmp_path / "out").exists() and not (tmp_path / "runs").exists()


def test_rollout_without_a_key_is_one_line_before_any_directory(tmp_path, capsys, monkeypatch, nothing_runs):
    monkeypatch.chdir(tmp_path)

    assert _exit_code(["rollout", _train_config(tmp_path), "--task", "a", "--prepare"]) == cli.EXIT_CONFIG
    argv = ["rollout", _eval_config(tmp_path), "--task", "a", "--output-dir", str(tmp_path / "run")]
    assert _exit_code(argv) == cli.EXIT_CONFIG

    _assert_says(capsys.readouterr().err, MISSING, MISSING)
    assert not (tmp_path / "runs").exists() and not (tmp_path / "run").exists()


def test_a_key_without_the_tml_prefix_is_one_line(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "sk-not-tinker")

    assert _exit_code(["eval", _eval_config(tmp_path)]) == cli.EXIT_CONFIG
    _assert_says(capsys.readouterr().err, "platoon: TINKER_API_KEY must start with tml-, as the Tinker client requires")


def test_training_without_a_key_stops_before_listing_tasks(tmp_path):
    with pytest.raises(TinkerKeyError, match="TINKER_API_KEY is not set"):
        asyncio.run(train_module.arun_tinker_training(["--config", _train_config(tmp_path)]))
    assert LISTED == []


def test_a_hosted_model_without_its_key_variable_is_refused_before_any_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PLATOON_TEST_HOSTED_KEY", raising=False)
    hosted = {**HOSTED_POLICY, "hosted": {**HOSTED_POLICY["hosted"], "api_key_env": "PLATOON_TEST_HOSTED_KEY"}}
    config = _eval_config(tmp_path, inference=hosted, environment=SCORED, output_dir=None)

    assert _exit_code(["eval", config]) == cli.EXIT_CONFIG
    assert _exit_code(["rollout", config, "--task", "a"]) == cli.EXIT_CONFIG

    err = capsys.readouterr().err
    assert err.count("inference.hosted: set PLATOON_TEST_HOSTED_KEY in the environment, which api_key_env names") == 2
    assert "TINKER_API_KEY" not in err
    assert not (tmp_path / "runs").exists()


def test_what_serves_no_tinker_policy_needs_no_key(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    errs: list[str] = []

    def run(*argv: str) -> str:
        cli.main(list(argv))
        out, err = capsys.readouterr()
        errs.append(err)
        return out

    # A report reads finished rollouts.
    assert '"total_rollouts": 0' in run("eval", _eval_config(tmp_path, stage="report"))
    # A hosted model (here one without auth) needs its own key variable, if any, not Tinker's.
    hosted = _eval_config(tmp_path, inference=HOSTED_POLICY, environment=SCORED, output_dir=None)
    assert json.loads(run("eval", hosted))["success_rate"] == 1.0
    run("rollout", hosted, "--task", "a", "--output-dir", str(tmp_path / "hosted-rollout"))
    # Without a renderer no policy is served: custom rollouts that sample nothing run without one.
    unserved = _eval_config(tmp_path, inference={"model_name": "m"}, environment=SCORED, output_dir="unserved")
    assert json.loads(run("eval", unserved))["success_rate"] == 1.0
    run("rollout", unserved, "--task", "a", "--output-dir", str(tmp_path / "unserved-rollout"))
    # Checking a config runs nothing, a training config's included.
    assert run("config", "validate", _train_config(tmp_path)).startswith("ok ")

    assert not any("TINKER_API_KEY" in err for err in errs)


# --------------------------------------------------------------------------------------------- a refused key


@contextlib.contextmanager
def refusing_service(status: int) -> Iterator[SimpleNamespace]:
    """A Tinker service on a local port that refuses every request with ``status``, answering ``ANSWER``: its
    ``url``, and the ``paths`` it was asked for."""

    paths: list[str] = []

    class Refuse(http.server.BaseHTTPRequestHandler):
        def refuse(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            paths.append(self.path)
            body = json.dumps({"detail": ANSWER}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = refuse

        def log_message(self, format, *args) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Refuse)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}", paths=paths)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def refused_key(monkeypatch):
    """The key the service refuses, with no proxy between this process and the local service, and the Tinker SDK's
    telemetry off (as a policy that serves an evaluation sets it; it is put back after the test)."""

    monkeypatch.setenv("TINKER_API_KEY", REFUSED_KEY)
    monkeypatch.setenv("TINKER_TELEMETRY", "0")
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)


def _refusal(url: str, status: int) -> str:
    """The one line a command ends with when the service at ``url`` refuses the key with ``status``: the key's name and
    the service's answer, in which the key itself is replaced by its name."""

    return (
        f"platoon: the Tinker service at {url} refused TINKER_API_KEY (HTTP {status}: no such key: TINKER_API_KEY): "
        "set TINKER_API_KEY to a key that service accepts"
    )


def _assert_ends_with_the_refusal(err: str, url: str, status: int) -> None:
    _assert_says(err, _refusal(url, status))
    assert REFUSED_KEY not in err


@pytest.mark.parametrize("status", [401, 403])
def test_rollout_with_a_refused_key_ends_with_one_line(tmp_path, capsys, refused_key, status):
    with refusing_service(status) as service:
        override = f"tinker_base_url={service.url}"  # the server a training config names
        argv = ["rollout", _train_config(tmp_path), override, "--task", "a", "--output-dir", str(tmp_path / "run")]
        assert _exit_code(argv) == cli.EXIT_CONFIG

    _assert_ends_with_the_refusal(capsys.readouterr().err, service.url, status)
    assert service.paths == ["/api/v1/create_session"]


def test_eval_with_a_refused_key_ends_with_one_line_and_runs_no_rollout(tmp_path, capsys, monkeypatch, refused_key):
    with refusing_service(401) as service:
        monkeypatch.setenv("TINKER_BASE_URL", service.url)
        assert _exit_code(["eval", _eval_config(tmp_path)]) == cli.EXIT_CONFIG

    _assert_ends_with_the_refusal(capsys.readouterr().err, service.url, 401)
    assert service.paths == ["/api/v1/create_session"]  # one rollout asked for a session; none started after it
    assert not list((tmp_path / "out").rglob("metadata.json"))


def test_eval_in_worker_processes_with_a_refused_key_ends_with_one_line(tmp_path, capfd, monkeypatch, refused_key):
    # One worker process for two rollouts at once: the second waits for it, and starts no more once the first was
    # refused.
    workflow = {"use_subprocesses": True, "subprocess_max_workers": 1, "num_concurrent_workers": 2}
    with refusing_service(403) as service:
        monkeypatch.setenv("TINKER_BASE_URL", service.url)
        config = _eval_config(tmp_path, inference={**TINKER_POLICY, "workflow": workflow})
        assert _exit_code(["eval", config]) == cli.EXIT_CONFIG

    _assert_ends_with_the_refusal(capfd.readouterr().err, service.url, 403)  # the worker printed no traceback
    assert service.paths == ["/api/v1/create_session"]
    assert not list((tmp_path / "out").rglob("metadata.json"))


def test_train_with_a_refused_key_ends_with_one_line(tmp_path, capsys, monkeypatch, refused_key):
    with refusing_service(401) as service:
        monkeypatch.setenv("TINKER_BASE_URL", service.url)
        assert _exit_code(["train", _train_config(tmp_path)]) == cli.EXIT_CONFIG

    _assert_ends_with_the_refusal(capsys.readouterr().err, service.url, 401)
    assert service.paths == ["/api/v1/create_session"]  # the trainer's policy sampler: its first session


@pytest.fixture
def start_signals():
    """A supervised start's ``main`` handles the signals ``platoon train`` passes on; put the test process's back."""

    saved = {sig: signal.getsignal(sig) for sig in (supervise.STOP_SIGNAL, supervise.STOP_NOW_SIGNAL)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_a_supervised_start_ends_a_refused_key_with_one_line(tmp_path, capsys, monkeypatch, start_signals):
    def refused(args: list[str]) -> None:
        raise tinker_key_refused(401, ANSWER, service="https://tinker.example")

    monkeypatch.setenv("TINKER_API_KEY", REFUSED_KEY)
    monkeypatch.setattr(train_module, "run_tinker_training", refused)

    assert supervise.main([str(tmp_path / "config.resolved.yaml")]) == supervise.EXIT_CONFIG  # never restarted
    _assert_says(capsys.readouterr().err, _refusal("https://tinker.example", 401))


def test_a_refusal_is_one_line_that_never_holds_the_key_and_crosses_processes(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", REFUSED_KEY)
    error = tinker_key_refused(403, f"  {ANSWER}\n" + "x" * 400, asked="a training client")

    assert isinstance(error, ConfigError) and error.status == 403
    assert str(error).startswith(
        "the Tinker service refused TINKER_API_KEY when asked for a training client (HTTP 403: no such key: "
        "TINKER_API_KEY x"
    )
    assert str(error).endswith("...): set TINKER_API_KEY to a key that service accepts")
    assert REFUSED_KEY not in str(error) and "\n" not in str(error)
    copy = pickle.loads(pickle.dumps(error))  # as an evaluation's worker process sends it back
    assert type(copy) is TinkerKeyRefusedError and (str(copy), copy.status) == (str(error), 403)
