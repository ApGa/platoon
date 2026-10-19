"""``watchdog.max_restarts``: ``platoon train`` restarts training after a failure (``platoon.train.supervise``).

End to end, ``platoon train`` runs each start as a child process on a fake Tinker service
(``supervised_start_stub.py``): a service that stalls ends a start with a deadline error and one that dies with a
connection error, and both are restarted and resume from the newest checkpoint, each start asking for its training
client until the service accepts it, once the sandboxes the dead start left on the process provider and on the
singularity one (``fake_hpc_sandbox``, which keeps its sessions on disk) are reaped.  The restart rules and the signals
are checked on stand-in starts: small Python processes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import cast

import pytest
import yaml
from sandbox_leftovers import leave_sandbox

from platoon import cli
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.train import reaping, supervise
from platoon.train.reaping import HeldRun, read_process

HERE = Path(__file__).resolve().parent
STUB = HERE / "supervised_start_stub.py"
FAKE_HPC_SANDBOX = HERE / "fake_hpc_sandbox"


# --------------------------------------------------------------------------------------------- end to end


@pytest.fixture(autouse=True)
def tinker_key(monkeypatch):
    """``platoon train`` checks its Tinker key before anything else; the trainers here are stand-ins, past it."""

    monkeypatch.setenv("TINKER_API_KEY", "tml-test")


@pytest.fixture
def fake_hpc_sandbox(monkeypatch):
    """The file-backed ``hpc_sandbox`` stand-in, in this process and in the starts this process runs."""

    spec = importlib.util.spec_from_file_location("hpc_sandbox", FAKE_HPC_SANDBOX / "hpc_sandbox" / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", module)  # first: its dataclasses look their module up
    spec.loader.exec_module(module)
    paths = [str(FAKE_HPC_SANDBOX), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(path for path in paths if path))


def _supervised_config(tmp_path: Path, providers: list, **watchdog) -> Path:
    environments = [
        {
            "name": name,
            "service": "harbor",
            "service_kwargs": {"task_dirs": [], "provider": name, "provider_options": options},
        }
        for name, options in providers
    ]
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "model_name": "Qwen/Qwen3-4B",
                    "renderer_name": "qwen3",
                    "batch_size": 2,
                    "max_training_steps": 4,
                    "max_concurrent_rollouts": 16,  # two groups of the default 8
                    "forward_backward_timeout_seconds": 1.0,
                },
                "eval": {"every": 0},
                "checkpoint": {"every": 100, "recovery_every": 1},
                "watchdog": {"max_restarts": 5, **watchdog},
                "stats": {"wandb": {"mode": "disabled"}},
                "log_path": str(tmp_path / "logs"),
                "environments": environments,
            }
        )
    )
    return path


def _reaped(message: str, provider: str) -> int:
    match = re.search(rf"\b{provider}: (\d+)", message)
    assert match, message
    return int(match.group(1))


def test_a_stalled_and_a_dead_training_service_are_restarted_and_resume(
    tmp_path, monkeypatch, caplog, capfd, fake_hpc_sandbox
):
    providers = [
        ["process", {"root": str(tmp_path / "process-sandboxes")}],
        ["singularity", {"scratch_root": str(tmp_path / "hpc-sandbox")}],
    ]
    state = tmp_path / "service.json"
    # Start 1 stalls in its second step, and the service holds its session through the next start's first two asks;
    # start 2's service dies in its second step and answers no connection to the next start's first two asks.
    script = {
        "1": {"stall_at": 2, "then_refuse": [400, 400]},
        "2": {"die_at": 2, "then_refuse": [None, None]},
    }
    state.write_text(json.dumps({"script": script}))
    monkeypatch.setenv("FAKE_TINKER", str(state))
    monkeypatch.setenv("FAKE_SANDBOX_PROVIDERS", json.dumps(providers))
    monkeypatch.setattr(supervise, "start_command", lambda config: [sys.executable, str(STUB), str(config)])
    caplog.set_level(logging.INFO)

    cli.main(["train", str(_supervised_config(tmp_path, providers))])

    run_dir = tmp_path / "logs" / "platoon_tinker" / "run"
    starts = json.loads(state.read_text())["starts"]
    assert sorted(starts) == ["1", "2", "3"]
    assert starts["1"]["asks"] == ["opened None"]
    assert starts["2"]["asks"] == ["refused 400", "refused 400", "opened tinker://fake/weights/recovery-000001"]
    assert starts["3"]["asks"] == ["refused None", "refused None", "opened tinker://fake/weights/recovery-000002"]
    # Each step was trained once: the stalled and the dead step had applied no update, and the next start redid them.
    assert [starts[start]["steps"] for start in ("1", "2", "3")] == [1, 1, 2]
    rows = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()]
    assert (rows[-1]["name"], rows[-1]["batch"]) == ("final", 4)
    err = capfd.readouterr().err
    assert "forward_backward (microbatch 0) timed out after 1s (limit: 1.0s)" in err
    assert err.count("The training service refused the training client (attempt ") == 4
    # What a dead start left was reaped before the next began, and what the last left as platoon train exited.
    assert [starts[start]["sandboxes_at_start"] for start in ("1", "2", "3")] == [[0], [0], [0]]
    assert list((tmp_path / "process-sandboxes").glob("process-*")) == []
    assert list((tmp_path / "hpc-sandbox" / "sessions").glob("*.json")) == []
    reaps = [record.getMessage() for record in caplog.records if record.name == "platoon.train.reaping"]
    label = f"{sandbox_cleanup.RUN_LABEL}={run_dir}"
    assert all(label in message for message in reaps), reaps
    for when in ("before restart 1", "before restart 2", "as the run exits"):
        [message] = [message for message in reaps if when in message]
        # Two groups a step, each with a sandbox of each provider: the trained steps' and the failed step's.
        assert _reaped(message, "process") >= 4 and _reaped(message, "singularity") >= 4, message
    restarts = [record.getMessage() for record in caplog.records if "restarting it" in record.getMessage()]
    assert restarts == [
        "Training exited with code 1; restarting it (restart 1 of at most 5)",
        "Training exited with code 1; restarting it (restart 2 of at most 5)",
    ]
    record = read_process(run_dir)
    assert record is not None and (record.command, record.pid) == ("train", os.getpid())


# --------------------------------------------------------------------------------------------- the rules


class FakeRun:
    """What :func:`supervise.supervise` needs of the run it holds (a :class:`HeldRun`), recording the reaps and the
    training processes."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.trainers: list[int | None] = []
        self.reaps: list[str] = []
        self.reaper = self

    def record_trainer(self, pid: int | None) -> None:
        self.trainers.append(pid)

    def reap(self, when: str, **kwargs) -> int:
        self.reaps.append(when)
        return 0


def _held(run: FakeRun) -> HeldRun:
    return cast(HeldRun, run)


def _start(tmp_path: Path, body: str) -> list[str]:
    """A stand-in start: a Python process that counts its starts in ``starts`` and runs ``body`` (``n`` is its
    number, ``run`` the run directory)."""

    script = tmp_path / "start.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json, os, signal, sys, time
            from pathlib import Path
            run = Path({str(tmp_path)!r})
            counter = run / "starts"
            n = int(counter.read_text()) + 1 if counter.exists() else 1
            counter.write_text(str(n))
            """
        )
        + textwrap.dedent(body)
    )
    return [sys.executable, str(script)]


def _checkpoint(step: str) -> str:
    """Code for a stand-in start that writes a resumable checkpoint row at step ``step`` (an expression)."""

    return (
        'with open(run / "checkpoints.jsonl", "a") as f: '
        f'f.write(json.dumps({{"name": "c", "batch": {step}, "state_path": "tinker://x"}}) + "\\n")\n'
    )


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ("sys.exit(0)", 0),
        ("sys.exit(78)", 78),
        ("sys.exit(130)", 130),
        ("sys.exit(143)", 143),
        ("os.kill(os.getpid(), signal.SIGTERM)", 143),
        ("os.kill(os.getpid(), signal.SIGINT)", 130),
    ],
    ids=["success", "config-error", "130", "143", "killed-by-SIGTERM", "killed-by-SIGINT"],
)
def test_no_restart_after_success_a_configuration_error_or_an_interrupt(tmp_path, body, code):
    run = FakeRun(tmp_path)

    assert supervise.supervise(_start(tmp_path, body), run=_held(run), max_restarts=5) == code

    assert (tmp_path / "starts").read_text() == "1"
    assert run.reaps == []  # the run's own exit reaps (held_run), not supervise


@pytest.mark.parametrize(
    "failure",
    ["sys.exit(1)", "sys.exit(2)", "os._exit(3)", "os.kill(os.getpid(), signal.SIGKILL)", "sys.exit(128 + 35)"],
)
def test_any_other_exit_restarts_until_a_start_succeeds(tmp_path, caplog, failure):
    caplog.set_level(logging.INFO)
    run = FakeRun(tmp_path)
    command = _start(tmp_path, _checkpoint("n") + f"if n < 3:\n    {failure}\n")

    assert supervise.supervise(command, run=_held(run), max_restarts=5, watchdog_exit_code=2) == 0

    assert (tmp_path / "starts").read_text() == "3"
    assert run.reaps == ["before restart 1", "before restart 2"]
    assert len([pid for pid in run.trainers if pid is not None]) == 3  # each start's process, in the run's record
    if failure == "sys.exit(2)":
        assert "Training exited with code 2, the watchdog's" in caplog.text
    if "SIGKILL" in failure:
        assert "Training exited with code 137 (SIGKILL killed it)" in caplog.text
    if failure == "sys.exit(128 + 35)":  # above 128, but no signal that Python names (a real-time one)
        assert "Training exited with code 163; restarting it" in caplog.text


def test_three_starts_in_a_row_without_a_new_checkpoint_give_up(tmp_path, caplog):
    run = FakeRun(tmp_path)
    # The first start makes progress (step 1); every later one saves only what a launch saves, at the same step.
    command = _start(tmp_path, _checkpoint("1") + "sys.exit(1)\n")

    assert supervise.supervise(command, run=_held(run), max_restarts=20) == 1

    assert (tmp_path / "starts").read_text() == "4"
    assert "3 starts in a row saved no new checkpoint: not restarting" in caplog.text


def test_a_fresh_runs_launch_checkpoint_is_no_progress(tmp_path):
    run = FakeRun(tmp_path)

    assert (
        supervise.supervise(_start(tmp_path, _checkpoint("0") + "sys.exit(1)\n"), run=_held(run), max_restarts=20) == 1
    )

    assert (tmp_path / "starts").read_text() == "3"


def test_restarts_stop_at_max_restarts(tmp_path, caplog):
    run = FakeRun(tmp_path)
    command = _start(tmp_path, _checkpoint("n") + "sys.exit(1)\n")  # progress every time

    assert supervise.supervise(command, run=_held(run), max_restarts=2) == 1

    assert (tmp_path / "starts").read_text() == "3"
    assert "after 2 restart(s), which is watchdog.max_restarts: not restarting" in caplog.text


def test_resume_step_reads_the_last_resumable_row(tmp_path):
    assert supervise.resume_step(tmp_path) == 0
    rows = [{"name": "a", "batch": 4, "state_path": "s"}, {"name": "b", "batch": 6}, "not json"]
    (tmp_path / "checkpoints.jsonl").write_text(
        "\n".join(json.dumps(row) if isinstance(row, dict) else row for row in rows)
    )
    assert supervise.resume_step(tmp_path) == 4


def test_the_start_command_runs_this_module_on_the_snapshot(tmp_path):
    assert supervise.start_command(tmp_path / "config.resolved.yaml") == [
        sys.executable,
        "-m",
        "platoon.train.supervise",
        str(tmp_path / "config.resolved.yaml"),
    ]


# --------------------------------------------------------------------------------------------- signals

# A supervisor in a process of its own, so that the signals the tests send never reach pytest's.
SUPERVISOR = """
import sys
from pathlib import Path
from platoon.train import supervise

class Run:
    run_dir = Path(sys.argv[1])
    def record_trainer(self, pid): pass
    def reap(self, when, **kwargs): return 0
    reaper = property(lambda self: self)

sys.exit(supervise.supervise([sys.executable, sys.argv[2]], run=Run(), max_restarts=5))
"""

# A start that notes the signals it gets, as the trainer would take them, says it is ready, and exits once one came
# (and LINGER seconds more): 9 after SIGUSR1 (stop at once), else 0 (stopped at the next step boundary).
STOPPABLE = """
import os, signal, sys, time
from pathlib import Path
run = Path(sys.argv[1])
with open(run / "starts", "a") as f: f.write("x")
got = []
for sig in (signal.SIGUSR2, signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda s, f: got.append(signal.Signals(s).name))
(run / "ready").write_text(str(os.getpid()))
deadline = time.time() + 20
while time.time() < deadline and not got:
    time.sleep(0.01)
time.sleep(float(os.environ.get("LINGER", "0")))
(run / "got").write_text(",".join(got))
sys.exit(9 if "SIGUSR1" in got else 0)
"""


def _supervisor(tmp_path: Path, linger: float = 0.0) -> subprocess.Popen:
    start = tmp_path / "start.py"
    start.write_text(STOPPABLE.replace("sys.argv[1]", repr(str(tmp_path))))
    environment = {**os.environ, "LINGER": str(linger)}
    process = subprocess.Popen(
        [sys.executable, "-c", SUPERVISOR, str(tmp_path), str(start)], env=environment, start_new_session=True
    )
    deadline = time.monotonic() + 60
    while not (tmp_path / "ready").exists():
        assert process.poll() is None and time.monotonic() < deadline, "the start never got ready"
        time.sleep(0.02)
    return process


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_reaches_the_start_as_a_stop_and_nothing_restarts(tmp_path, sig):
    supervisor = _supervisor(tmp_path)

    supervisor.send_signal(sig)

    assert supervisor.wait(timeout=60) == 0  # the start's own exit code: it stopped as asked
    assert (tmp_path / "got").read_text() == "SIGUSR2"
    assert (tmp_path / "starts").read_text() == "x"


def test_a_second_signal_reaches_the_start_as_a_stop_at_once(tmp_path):
    supervisor = _supervisor(tmp_path, linger=2.0)

    supervisor.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    supervisor.send_signal(signal.SIGTERM)

    assert supervisor.wait(timeout=60) == 9
    assert (tmp_path / "got").read_text() == "SIGUSR2,SIGUSR1"
    assert (tmp_path / "starts").read_text() == "x"


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_a_signal_to_the_whole_group_reaches_the_start_as_itself_and_as_one_stop(tmp_path, sig):
    supervisor = _supervisor(tmp_path, linger=1.0)

    os.killpg(supervisor.pid, sig)  # a scheduler signals every process of its job, a terminal its whole group

    assert supervisor.wait(timeout=60) == 0
    assert sorted((tmp_path / "got").read_text().split(",")) == sorted([sig.name, "SIGUSR2"])
    assert (tmp_path / "starts").read_text() == "x"


def test_a_start_that_a_passed_on_stop_killed_before_it_took_it_was_interrupted(tmp_path):
    start = tmp_path / "start.py"
    start.write_text(f"import time; open({str(tmp_path / 'ready')!r}, 'w').close(); time.sleep(30)\n")
    supervisor = subprocess.Popen([sys.executable, "-c", SUPERVISOR, str(tmp_path), str(start)])
    deadline = time.monotonic() + 60
    while not (tmp_path / "ready").exists():
        assert supervisor.poll() is None and time.monotonic() < deadline
        time.sleep(0.02)

    supervisor.send_signal(signal.SIGTERM)  # its SIGUSR2 kills the start, which set no handler for it

    assert supervisor.wait(timeout=60) == 128 + signal.SIGTERM


# --------------------------------------------------------------------------------------------- the config


def _watchdog(tmp_path: Path, **watchdog):
    from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
    from platoon.utils.config import resolve_config

    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r"},
                "eval": {"every": 0},
                "log_path": str(tmp_path),
                "watchdog": watchdog,
            }
        )
    )
    return resolve_config(path, (), PlatoonTinkerRLTrainerConfig).config.watchdog


def test_max_restarts_defaults_to_none(tmp_path):
    assert _watchdog(tmp_path).max_restarts == 0
    assert _watchdog(tmp_path, max_restarts=20).max_restarts == 20


@pytest.mark.parametrize("value", [-1, 1.5, True, "3"])
def test_max_restarts_is_a_whole_number(tmp_path, value):
    from platoon.utils.config import ConfigError

    with pytest.raises(ConfigError, match="watchdog.max_restarts"):
        _watchdog(tmp_path, max_restarts=value)


@pytest.mark.parametrize("code", [0, 78, 130, 143])
def test_a_watchdog_exit_that_would_never_restart_is_refused_with_restarts(tmp_path, code):
    from platoon.utils.config import ConfigError

    with pytest.raises(ConfigError, match=rf"watchdog.exit_code: {code} is the exit code of a success"):
        _watchdog(tmp_path, max_restarts=3, exit_code=code)
    assert _watchdog(tmp_path, exit_code=code).exit_code == code  # without restarts, as before
    assert _watchdog(tmp_path, max_restarts=3, exit_code=code, enabled=False).max_restarts == 3


@pytest.mark.parametrize("code", [256, 78 + 256, -1])
def test_a_watchdog_exit_code_that_is_no_exit_code_is_refused_with_restarts(tmp_path, code):
    """A process's exit code is 8 bits: one that exits with 256 ends with 0, which looks like a success."""

    from platoon.utils.config import ConfigError

    with pytest.raises(ConfigError, match=rf"watchdog.exit_code: {code} is no exit code, which is 0 to 255"):
        _watchdog(tmp_path, max_restarts=3, exit_code=code)
    assert _watchdog(tmp_path, exit_code=code).exit_code == code  # without restarts, as before


# --------------------------------------------------------------------------------------------- platoon train


def _in_process_config(tmp_path: Path, root: Path) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "Qwen/Qwen3-4B", "renderer_name": "qwen3", "max_training_steps": 1},
                "eval": {"every": 0},
                "log_path": str(tmp_path / "logs"),
                "environments": [
                    {
                        "service": "harbor",
                        "service_kwargs": {
                            "task_dirs": [],
                            "provider": "process",
                            "provider_options": {"root": str(root)},
                        },
                    }
                ],
            }
        )
    )
    return path


def test_train_in_its_own_process_records_itself_and_reaps_before_and_after(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    root = tmp_path / "sandboxes"
    run_dir = tmp_path / "logs" / "platoon_tinker" / "run"
    label = sandbox_cleanup.run_label(run_dir)
    asyncio.run(leave_sandbox(root, label))  # what a killed start of the run left
    trained: list[list[str]] = []

    def train(args: list[str]) -> None:
        trained.append(args)
        assert list(root.glob("process-*")) == []
        record = read_process(run_dir)
        assert record is not None and record.pid == os.getpid()
        asyncio.run(leave_sandbox(root, label))  # and what this one leaves

    monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", train)

    cli.main(["train", str(_in_process_config(tmp_path, root))])

    assert trained == [["--config", str(run_dir / "config.resolved.yaml")]]
    assert list(root.glob("process-*")) == []
    reaps = [record.getMessage() for record in caplog.records if record.name == "platoon.train.reaping"]
    assert len(reaps) == 2
    assert "before the run starts (process: 1)" in reaps[0]
    assert "as the run exits (this process's own: 0, process: 1)" in reaps[1]


def test_train_refuses_a_run_directory_another_process_runs_before_writing_in_it(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "logs" / "platoon_tinker" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.resolved.yaml").write_text("# the running run's\n")
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        record = reaping.ProcessRecord("train", socket.gethostname(), 4_194_304, None)  # no such pid
        reaping.record_trainer(run_dir, record, sleeper.pid)
        monkeypatch.setattr("platoon.train.tinker.train.run_tinker_training", lambda args: pytest.fail("trained"))

        with pytest.raises(SystemExit) as exit_info:
            cli.main(["train", str(_in_process_config(tmp_path, tmp_path / "sandboxes"))])

        assert exit_info.value.code == cli.EXIT_CONFIG
        assert f"is in use: the training process of platoon train (pid {sleeper.pid}) still runs it" in (
            capsys.readouterr().err
        )
        assert (run_dir / "config.resolved.yaml").read_text() == "# the running run's\n"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_train_returns_the_supervised_runs_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(supervise, "start_command", lambda config: [sys.executable, "-c", "raise SystemExit(78)"])
    config = _in_process_config(tmp_path, tmp_path / "sandboxes")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["train", str(config), "watchdog.max_restarts=3"])

    assert exit_info.value.code == 78
