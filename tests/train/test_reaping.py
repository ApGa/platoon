"""What a run leaves behind, and reaping it (``platoon.train.reaping``): the providers a run's config names, the record
of the processes that run a run directory, the refusals that keep two runs from reaping each other's sandboxes, and
``platoon sandbox reap``."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sandbox_leftovers import alive, leave_sandbox

from platoon import cli
from platoon.sandbox import SandboxSpec
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.process import ProcessProvider
from platoon.train import reaping
from platoon.train.tinker.config_defs import PlatoonTinkerRLTrainerConfig
from platoon.utils.config import ConfigError, resolve_config

RUN = sandbox_cleanup.RUN_LABEL
FAKE_HPC_SANDBOX = Path(__file__).resolve().parent / "fake_hpc_sandbox" / "hpc_sandbox" / "__init__.py"


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(sandbox_cleanup, "_live", {})
    monkeypatch.setattr(sandbox_cleanup, "_labels", {})
    monkeypatch.setattr(sandbox_cleanup, "_mirror_path", None)
    monkeypatch.setattr(sandbox_cleanup, "_shut_down", False)


@pytest.fixture
def fake_hpc_sandbox(monkeypatch):
    spec = importlib.util.spec_from_file_location("hpc_sandbox", FAKE_HPC_SANDBOX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "hpc_sandbox", module)
    spec.loader.exec_module(module)
    return module


def _harbor(provider: str | None = None, **options) -> dict:
    kwargs: dict = {"task_dirs": []}
    if provider is not None:
        kwargs["provider"] = provider
    if options:
        kwargs["provider_options"] = options
    return {"service": "harbor", "service_kwargs": kwargs}


# --------------------------------------------------------------------------------------------- the providers


def test_the_providers_are_each_harbor_entrys_once_and_docker_by_default():
    environments = [
        _harbor(),
        _harbor("singularity", scratch_root="/scratch"),
        {**_harbor("singularity", scratch_root="/scratch"), "name": "again"},
        {"service": "platoon.envs.harbor:HarborEnvironmentService", "service_kwargs": {"provider": "process"}},
        {"service": "table", "service_kwargs": {"provider": "process"}},
        {"dataset_loader": "pkg:load"},
    ]

    assert reaping.sandbox_providers(environments) == [
        ("docker", {}),
        ("singularity", {"scratch_root": "/scratch"}),
        ("process", {}),
    ]


def test_the_providers_of_a_loaded_config_are_those_of_its_file(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {"model_name": "m", "renderer_name": "r"},
                "eval": {"every": 0},
                "log_path": str(tmp_path),
                "environments": [{"name": "a", **_harbor("enroot", allow_network_none=True)}, {"name": "b"}],
            }
        )
    )
    config = resolve_config(path, (), PlatoonTinkerRLTrainerConfig).config

    assert reaping.sandbox_providers(config.environments) == [("enroot", {"allow_network_none": True})]


def test_the_default_provider_is_the_harbor_services():
    from platoon.envs.harbor import HarborEnvironmentService

    default = inspect.signature(HarborEnvironmentService).parameters["provider"].default
    assert reaping.HARBOR_DEFAULT_PROVIDER == default


# --------------------------------------------------------------------------------------------- the record


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_the_record_names_this_process_its_host_and_the_runs_label(tmp_path):
    record = reaping.record_process(tmp_path / "run", "train")

    assert reaping.read_process(tmp_path / "run") == record
    assert (record.command, record.host, record.pid) == ("train", socket.gethostname(), os.getpid())
    assert record.label == {RUN: str(tmp_path / "run")}
    assert reaping.running_here(record) == f"platoon train (pid {os.getpid()}) still runs it on this host"
    assert reaping.running_here(record, ignore=[os.getpid()]) is None


def test_a_supervised_runs_record_names_its_training_process(tmp_path):
    record = reaping.record_process(tmp_path, "train")
    sleeper = _sleeper()
    try:
        record = reaping.record_trainer(tmp_path, record, sleeper.pid)
        assert reaping.read_process(tmp_path) == record
        assert reaping.running_here(record, ignore=[os.getpid()]) == (
            f"the training process of platoon train (pid {sleeper.pid}) still runs it on this host"
        )
    finally:
        sleeper.kill()
        sleeper.wait()
    assert reaping.running_here(record, ignore=[os.getpid()]) is None


def test_a_dead_or_recycled_pid_runs_nothing(tmp_path):
    sleeper = _sleeper()
    sleeper.kill()
    sleeper.wait()
    dead = reaping.ProcessRecord("train", socket.gethostname(), sleeper.pid, 1)
    recycled = reaping.ProcessRecord("eval", socket.gethostname(), os.getpid(), start_time=1)  # another start time

    assert reaping.running_here(dead) is None
    assert reaping.running_here(recycled) is None


def test_a_record_of_another_host_cannot_be_toldalive(tmp_path):
    reaping.record_process(tmp_path, "eval")
    data = json.loads((tmp_path / reaping.PROCESS_RECORD).read_text())
    (tmp_path / reaping.PROCESS_RECORD).write_text(json.dumps({**data, "host": "elsewhere"}))

    why = reaping.live_process(tmp_path)

    assert why is not None and why.startswith("its last process, platoon eval (pid") and "ran on host elsewhere" in why
    assert reaping.running_here(reaping.read_process(tmp_path)) is None  # another start of the run may begin here


def test_no_or_an_unreadable_record_names_no_process(tmp_path):
    assert reaping.read_process(tmp_path) is None and reaping.live_process(tmp_path) is None
    (tmp_path / reaping.PROCESS_RECORD).write_text("{not json")
    assert reaping.read_process(tmp_path) is None and reaping.live_process(tmp_path) is None


def test_a_start_refuses_a_run_directory_another_process_runs(tmp_path):
    sleeper = _sleeper()
    try:
        record = reaping.ProcessRecord("train", socket.gethostname(), sleeper.pid, None)
        (tmp_path / reaping.PROCESS_RECORD).write_text(json.dumps(record.__dict__))
        with pytest.raises(ConfigError, match=rf"is in use: platoon train \(pid {sleeper.pid}\) still runs it"):
            reaping.refuse_running(tmp_path, "train")
    finally:
        sleeper.kill()
        sleeper.wait()
    reaping.refuse_running(tmp_path, "train")


# --------------------------------------------------------------------------------------------- reaping


def test_a_reap_takes_the_runs_sandboxes_and_no_other_runs(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    root = tmp_path / "sandboxes"
    ours = asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "run")))
    theirs = asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "run-2")))

    reaper = reaping.RunReaper(tmp_path / "run", [("process", {"root": str(root)})])
    try:
        assert reaper.reap("for the test") == 1
        assert not alive(ours) and alive(theirs)
        assert len(list(root.glob("process-*"))) == 1
        assert f"Reaped 1 sandbox(es) labelled {RUN}={tmp_path / 'run'} for the test (process: 1)" in caplog.text
    finally:
        reaping.RunReaper(tmp_path / "run-2", [("process", {"root": str(root)})]).reap("after the test")
    assert not alive(theirs)


def test_a_reap_stops_the_sandboxes_this_process_still_holds_when_asked(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    root = tmp_path / "sandboxes"
    reaper = reaping.RunReaper(tmp_path / "run", [("process", {"root": str(root)})])

    async def run() -> None:
        await ProcessProvider(root=root).start(SandboxSpec(image="local"))  # unlabelled: no reap by label finds it
        assert await reaper.areap("without its own") == 0
        assert await reaper.areap("with its own", own=True) == 1

    asyncio.run(run())

    assert list(root.glob("process-*")) == []
    assert "with its own (this process's own: 1, process: 0)" in caplog.text


def test_a_reap_of_a_singularity_run_goes_through_hpc_sandbox_with_the_configs_options(
    tmp_path, fake_hpc_sandbox, caplog
):
    caplog.set_level(logging.INFO)
    label = sandbox_cleanup.run_label(tmp_path / "run")
    sessions = tmp_path / "scratch" / "sessions"
    sessions.mkdir(parents=True)
    for name, owner, labels in [("ours", 1 << 22, label), ("live", os.getpid(), label), ("other", 1 << 22, {RUN: "x"})]:
        (sessions / f"{name}.json").write_text(json.dumps({"owner": owner, "labels": labels}))
    environments = [_harbor("singularity", scratch_root=str(tmp_path / "scratch"))]

    reaped = reaping.RunReaper(tmp_path / "run", reaping.sandbox_providers(environments)).reap("now")

    assert reaped == 1
    assert sorted(path.stem for path in sessions.iterdir()) == ["live", "other"]
    assert "(singularity: 1)" in caplog.text


def test_a_reap_never_raises(tmp_path, caplog):
    reaper = reaping.RunReaper(tmp_path, [("no.such.module:Provider", {})])

    assert reaper.reap("anyway") == 0
    assert "Cannot reap the sandboxes labelled" in caplog.text


def test_held_run_records_the_process_and_reaps_before_and_after(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    root = tmp_path / "sandboxes"
    environments = [_harbor("process", root=str(root))]
    asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "run")))

    with reaping.held_run(tmp_path / "run", "train", environments, workers=False, reap_first=True) as run:
        assert run.record == reaping.read_process(tmp_path / "run")
        assert list(root.glob("process-*")) == []  # the leftover went before the run began
        asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "run")))

    assert list(root.glob("process-*")) == []
    assert "before the run starts (process: 1)" in caplog.text
    assert "as the run exits (this process's own: 0, process: 1)" in caplog.text


def test_held_run_reaps_when_the_run_fails(tmp_path):
    root = tmp_path / "sandboxes"
    with pytest.raises(RuntimeError, match="the run failed"):
        with reaping.held_run(
            tmp_path / "run", "eval", [_harbor("process", root=str(root))], workers=False, reap_first=False
        ):
            asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "run")))
            raise RuntimeError("the run failed")

    assert list(root.glob("process-*")) == []


# --------------------------------------------------------------------------------------------- platoon sandbox reap


def _killed_run(tmp_path: Path) -> tuple[Path, Path, int]:
    """A run directory as a killed platoon train left it, with a sandbox still running; and its sandbox root."""

    root = tmp_path / "sandboxes"
    run_dir = tmp_path / "logs" / "exp" / "trial"
    run_dir.mkdir(parents=True)
    config = {"environments": [{"name": "h", **_harbor("process", root=str(root))}]}
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(config))
    sleeper = _sleeper()
    record = reaping.ProcessRecord("train", socket.gethostname(), sleeper.pid, None, sandbox_cleanup.run_label(run_dir))
    (run_dir / reaping.PROCESS_RECORD).write_text(json.dumps(record.__dict__))
    sleeper.kill()
    sleeper.wait()
    pid = asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(run_dir)))
    return run_dir, root, pid


def _exit_code(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    return exit_info.value.code


def test_sandbox_reap_reaps_a_killed_runs_sandboxes(tmp_path, capsys):
    run_dir, root, pid = _killed_run(tmp_path)

    cli.main(["sandbox", "reap", str(run_dir)])

    assert capsys.readouterr().out == f"reaped 1 sandbox(es) of {run_dir}\n"
    assert not alive(pid) and list(root.glob("process-*")) == []


def test_sandbox_reap_refuses_a_run_that_still_runs(tmp_path, capsys):
    run_dir, root, pid = _killed_run(tmp_path)
    sleeper = _sleeper()
    try:
        record = reaping.read_process(run_dir)
        assert record is not None
        reaping.record_trainer(run_dir, record, sleeper.pid)

        assert _exit_code(["sandbox", "reap", str(run_dir)]) == 1

        assert "not reaping the sandboxes of" in capsys.readouterr().err
        assert alive(pid)
    finally:
        sleeper.kill()
        sleeper.wait()
    cli.main(["sandbox", "reap", str(run_dir)])
    assert not alive(pid)


def test_sandbox_reap_refuses_a_run_that_last_ran_on_another_host(tmp_path, capsys):
    run_dir, root, pid = _killed_run(tmp_path)
    data = json.loads((run_dir / reaping.PROCESS_RECORD).read_text())
    (run_dir / reaping.PROCESS_RECORD).write_text(json.dumps({**data, "host": "elsewhere"}))
    try:
        assert _exit_code(["sandbox", "reap", str(run_dir)]) == 1
        assert "ran on host elsewhere" in capsys.readouterr().err
        assert alive(pid)
    finally:
        reaping.RunReaper(run_dir, [("process", {"root": str(root)})]).reap("after the test")


def test_sandbox_reap_needs_a_run_directory(tmp_path, capsys):
    assert _exit_code(["sandbox", "reap", str(tmp_path)]) == cli.EXIT_USAGE
    assert "holds no config.resolved.yaml" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------- platoon eval


def load_tasks(config, split, **kwargs) -> list[str]:
    return ["one"]


def load_task(task_id: str):
    from platoon.task import Task

    return Task(goal=f"solve {task_id}", id=task_id)


async def answer(task, config) -> dict:
    return {
        "trajectories": {
            "root": {"id": "root", "task": {"id": task.id}, "parent_info": None, "reward": 1.0, "steps": [{}]}
        }
    }


def test_eval_records_itself_and_reaps_the_sandboxes_labelled_with_it_as_it_exits(tmp_path, capsys, caplog):
    caplog.set_level(logging.INFO)
    root, output = tmp_path / "sandboxes", tmp_path / "results"
    leftover = asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(output)))  # a killed evaluation's, here
    elsewhere = asyncio.run(leave_sandbox(root, sandbox_cleanup.run_label(tmp_path / "other")))
    entry = {
        "dataset_loader": f"{__name__}:load_tasks",
        "task_loader": f"{__name__}:load_task",
        "rollout": f"{__name__}:answer",
        "service": "harbor",
        "service_kwargs": {"task_dirs": [], "provider": "process", "provider_options": {"root": str(root)}},
    }
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump({"inference": {"model_name": "fake-model", "output_dir": str(output)}, "environments": [entry]})
    )
    try:
        cli.main(["eval", str(config)])

        assert json.loads(capsys.readouterr().out)["total_rollouts"] == 1
        record = reaping.read_process(output)
        assert record is not None and (record.command, record.pid) == ("eval", os.getpid())
        assert not alive(leftover) and alive(elsewhere)
        assert f"labelled {RUN}={output} as the run exits (this process's own: 0, process: 1)" in caplog.text
    finally:
        reaping.RunReaper(tmp_path / "other", [("process", {"root": str(root)})]).reap("after the test")


@pytest.mark.parametrize("stage", ["full", "report"])
def test_eval_refuses_a_directory_another_process_runs_before_writing_in_it(tmp_path, capsys, stage):
    output = tmp_path / "results"
    output.mkdir()
    # The running run's record, which a supervised training run loads again at each restart.
    (output / "config.resolved.yaml").write_text("# the running run's\n")
    (output / "lock.json").write_text("{}\n")
    sleeper = _sleeper()
    try:
        record = reaping.ProcessRecord("train", socket.gethostname(), sleeper.pid, None)
        (output / reaping.PROCESS_RECORD).write_text(json.dumps(record.__dict__))
        entry = {"dataset_loader": f"{__name__}:load_tasks", "task_loader": f"{__name__}:load_task"}
        entry["rollout"] = f"{__name__}:answer"
        config = tmp_path / "eval.yaml"
        config.write_text(
            yaml.safe_dump(
                {"inference": {"model_name": "m", "output_dir": str(output)}, "stage": stage, "environments": [entry]}
            )
        )

        assert _exit_code(["eval", str(config)]) == cli.EXIT_CONFIG

        assert f"is in use: platoon train (pid {sleeper.pid}) still runs it on this host" in capsys.readouterr().err
        assert (output / "config.resolved.yaml").read_text() == "# the running run's\n"
        assert (output / "lock.json").read_text() == "{}\n"
        assert reaping.read_process(output) == record
    finally:
        sleeper.kill()
        sleeper.wait()
