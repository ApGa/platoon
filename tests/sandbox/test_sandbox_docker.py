from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from platoon.sandbox import Mount, SandboxSpec, get_provider
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.docker import EXEC_MARKER_ENV, DockerProvider, image_reference, run_arguments

WORKER = sandbox_cleanup.WORKER_LABEL

# Records each CLI call as a JSON line; hangs on the command $FAKE_DOCKER_HANG names, fails the commands
# $FAKE_DOCKER_FAIL lists (comma-separated) with $FAKE_DOCKER_ERROR, answers the path-kind probe from
# $FAKE_DOCKER_KIND and the process listing with the container's own processes, and fails a kill of the others
# when $FAKE_DOCKER_KILL_FAILS is set.
FAKE_DOCKER = """\
import json, os, sys, time
args = sys.argv[1:]
with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
if args[0] == os.environ.get("FAKE_DOCKER_HANG"):
    time.sleep(300)
if args[0] in os.environ.get("FAKE_DOCKER_FAIL", "").split(","):
    sys.exit(os.environ.get("FAKE_DOCKER_ERROR", "simulated failure"))
if args[0] == "exec" and "kind" in args:
    print(os.environ.get("FAKE_DOCKER_KIND", ""))
if args[0] == "exec" and "platoon-list" in args:
    print("1:1000\\n7:1003")
if args[0] == "exec" and "platoon-kill" in args and os.environ.get("FAKE_DOCKER_KILL_FAILS"):
    sys.exit("platoon: processes still alive after SIGKILL: 12")
"""


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(sandbox_cleanup, "_live", {})
    monkeypatch.setattr(sandbox_cleanup, "_labels", {})
    monkeypatch.setattr(sandbox_cleanup, "_mirror_path", None)
    monkeypatch.setattr(sandbox_cleanup, "_shut_down", False)


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch) -> Path:
    binary = tmp_path / "bin" / "docker"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    binary.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("PATH", f"{binary.parent}:{Path(sys.executable).parent}:/usr/bin:/bin")
    return log


def _calls(log: Path, *, markers: bool = False) -> list[list[str]]:
    """The CLI calls since the last look; without ``markers``, leaving out the marker each exec carries."""

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    log.write_text("")
    if not markers:
        calls = [[arg for arg in call if not arg.startswith(f"--env={EXEC_MARKER_ENV}=")] for call in calls]
    return calls


def test_image_reference_accepts_docker_images_only():
    assert image_reference("docker://python:3.12-slim") == "python:3.12-slim"
    assert image_reference("ghcr.io/org/image@sha256:abc") == "ghcr.io/org/image@sha256:abc"
    for image in ["local", "/images/task.sif", "oras://registry/image:tag"]:
        with pytest.raises(ValueError, match="cannot run"):
            image_reference(image)


def test_run_arguments_reflect_the_spec():
    spec = SandboxSpec(
        image="docker://python:3.12",
        workdir="/work",
        env={"A": "1"},
        mounts=[Mount(source="/host/tools", target="/opt/tools"), Mount("/host/data", "/data", read_only=False)],
        network="none",
        cpus=1.5,
        memory_mb=2048,
        labels={"platoon.rollout_id": "r1"},
    )

    assert run_arguments(spec, name="box", image="python:3.12") == [
        "run",
        "--detach",
        "--name",
        "box",
        "--workdir",
        "/work",
        "--network",
        "none",
        "--env=A=1",
        "--label=platoon.rollout_id=r1",
        "--volume",
        "/host/tools:/opt/tools:ro",
        "--volume",
        "/host/data:/data",
        "--cpus",
        "1.5",
        "--memory",
        "2048m",
        "--entrypoint",
        "sleep",
        "python:3.12",
        "infinity",
    ]


@pytest.mark.asyncio
async def test_sandbox_lifecycle_drives_the_cli(fake_docker: Path, tmp_path: Path, monkeypatch):
    provider = get_provider("docker")
    assert isinstance(provider, DockerProvider)
    sandbox = await provider.start(SandboxSpec(image="docker://python:3.12"))
    container = sandbox.id
    run, listing = _calls(fake_docker)
    assert run[:4] == ["run", "--detach", "--name", container]
    # The container's own processes are listed before any command runs, from "/".
    assert listing[:6] == ["exec", "--workdir", "/", container, "/bin/sh", "-c"] and listing[-1] == "platoon-list"

    await sandbox.exec("echo hi", env={"K": "V"}, cwd="sub")
    await sandbox.exec(["cat"], stdin=b"data")
    assert _calls(fake_docker) == [
        ["exec", "--workdir", "/app/sub", "--env=K=V", container, "bash", "-c", "echo hi"],
        ["exec", "--workdir", "/app", "--interactive", container, "cat"],
    ]

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "x").write_text("x")
    await sandbox.upload(tests, "/tests")
    await sandbox.upload(tests / "x", "/logs/x.txt")
    calls = _calls(fake_docker)
    assert calls[0][-4:] == [container, "mkdir", "-p", "/tests"]
    assert calls[1] == ["cp", f"{tests}/.", f"{container}:/tests"]
    assert calls[2][-4:] == [container, "mkdir", "-p", "/logs"]
    assert calls[3] == ["cp", str(tests / "x"), f"{container}:/logs/x.txt"]

    monkeypatch.setenv("FAKE_DOCKER_KIND", "dir")
    await sandbox.download("/logs/verifier", tmp_path / "out")
    assert _calls(fake_docker)[1] == ["cp", f"{container}:/logs/verifier/.", str(tmp_path / "out")]
    monkeypatch.setenv("FAKE_DOCKER_KIND", "")
    with pytest.raises(FileNotFoundError, match="/logs/missing"):
        await sandbox.download("/logs/missing", tmp_path / "missing")
    _calls(fake_docker)

    await sandbox.kill_processes()
    (kill,) = _calls(fake_docker)
    assert kill[:6] == ["exec", "--workdir", "/", container, "/bin/sh", "-c"]
    assert kill[7:] == ["platoon-kill", "100", "0.05", "", "1:1000", "7:1003"]  # all but these die
    sandbox.keep_processes(["9:900", "20:2000"])  # a task's init, and what it started: they live on, with descendants
    await sandbox.kill_processes()
    (kill,) = _calls(fake_docker)
    assert kill[7:] == ["platoon-kill", "100", "0.05", "20:2000 9:900", "1:1000", "7:1003"]
    monkeypatch.setenv("FAKE_DOCKER_KILL_FAILS", "1")
    with pytest.raises(RuntimeError, match="still alive after SIGKILL: 12"):
        await sandbox.kill_processes()
    monkeypatch.delenv("FAKE_DOCKER_KILL_FAILS")
    _calls(fake_docker)

    snapshot = await sandbox.snapshot()
    restored = await provider.restore(snapshot, sandbox.spec)
    calls = _calls(fake_docker)
    assert calls[0] == ["commit", container, snapshot.path]
    assert calls[1][-3:] == ["sleep", snapshot.path, "infinity"]
    assert calls[2][-1] == "platoon-list"  # a restored container lists its own processes too

    await sandbox.stop()
    await sandbox.stop()
    assert _calls(fake_docker) == [["rm", "--force", "--volumes", container], ["rmi", snapshot.path]]
    with pytest.raises(RuntimeError, match="stopped"):
        await sandbox.exec("true")
    await restored.stop()


@pytest.mark.asyncio
async def test_containers_carry_the_labels_of_the_process_that_starts_them(fake_docker: Path):
    sandbox_cleanup.set_labels({WORKER: "req-7"})

    sandbox = await DockerProvider().start(SandboxSpec(image="docker://python:3.12", labels={"platoon.task": "t"}))

    (run,) = [call for call in _calls(fake_docker) if call[0] == "run"]
    assert {"--label=platoon.task=t", f"--label={WORKER}=req-7"} <= set(run)
    assert sandbox.spec.labels == {"platoon.task": "t", WORKER: "req-7"}
    await sandbox.stop()


@pytest.mark.asyncio
async def test_failed_starts_clean_up_and_raise(fake_docker: Path, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_FAIL", "run")
    provider = DockerProvider()

    with pytest.raises(RuntimeError, match="simulated failure"):
        await provider.start(SandboxSpec(image="docker://python:3.12"))
    run, remove = _calls(fake_docker)
    assert remove == ["rm", "--force", "--volumes", run[3]]


@pytest.mark.asyncio
async def test_a_start_cancelled_during_docker_run_removes_its_container(
    fake_docker: Path, tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("FAKE_DOCKER_HANG", "run")
    mirror = tmp_path / "worker.sandboxes"
    sandbox_cleanup.set_labels({WORKER: "req-8"})
    sandbox_cleanup.mirror_to(str(mirror))
    starting = asyncio.create_task(DockerProvider().start(SandboxSpec(image="docker://python:3.12")))
    deadline = time.monotonic() + 30
    while not (fake_docker.exists() and fake_docker.read_text()):  # `docker run` is running
        assert time.monotonic() < deadline, "docker run never ran"
        await asyncio.sleep(0.05)
    assert [record.labels for record in sandbox_cleanup.read_mirror(str(mirror))] == [{WORKER: "req-8"}]

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    run, remove = _calls(fake_docker)  # `docker run` may have created the container before it was killed
    assert remove == ["rm", "--force", "--volumes", run[3]]
    assert sandbox_cleanup.read_mirror(str(mirror)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "listed"),
    [("simulated failure", [{WORKER: "req-9"}]), ("Error response from daemon: No such container: box", [])],
    ids=["removal-fails", "no-container"],
)
async def test_a_failed_start_stays_listed_for_reaping_until_its_container_is_gone(
    fake_docker: Path, tmp_path: Path, monkeypatch, error, listed
):
    monkeypatch.setenv("FAKE_DOCKER_FAIL", "run,rm")
    monkeypatch.setenv("FAKE_DOCKER_ERROR", error)
    mirror = tmp_path / "worker.sandboxes"
    sandbox_cleanup.set_labels({WORKER: "req-9"})
    sandbox_cleanup.mirror_to(str(mirror))

    with pytest.raises(RuntimeError, match="Could not start a container"):
        await DockerProvider().start(SandboxSpec(image="docker://python:3.12"))

    run, remove = _calls(fake_docker)
    assert remove == ["rm", "--force", "--volumes", run[3]]
    assert [record.labels for record in sandbox_cleanup.read_mirror(str(mirror))] == listed


def test_the_provider_needs_the_cli(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")

    with pytest.raises(RuntimeError, match="'docker' CLI"):
        DockerProvider()


@pytest.mark.asyncio
async def test_every_command_carries_a_marker_of_its_own(fake_docker: Path):
    sandbox = await DockerProvider().start(SandboxSpec(image="docker://python:3.12"))
    _calls(fake_docker)
    await sandbox.exec("true")
    await sandbox.exec(["true"])
    first, second = _calls(fake_docker, markers=True)
    assert first[3].startswith(f"--env={EXEC_MARKER_ENV}=") and second[3].startswith(f"--env={EXEC_MARKER_ENV}=")
    assert first[3] != second[3]
    await sandbox.stop()


# Runs `docker exec` commands on this host, in a session of their own, as the docker daemon runs them in the
# container: killing the client leaves the command running.  Answers the process listing with nothing.
FAKE_DAEMON = """import os, subprocess, sys
args = sys.argv[1:]
if args[0] != "exec" or "platoon-list" in args:
    sys.exit(0)
env, index, interactive = dict(os.environ), 1, False
while args[index].startswith("--"):
    if args[index] == "--workdir":
        index += 2
        continue
    if args[index].startswith("--env="):
        key, _, value = args[index][len("--env="):].partition("=")
        env[key] = value
    interactive = interactive or args[index] == "--interactive"
    index += 1
command = args[index + 1:]
process = subprocess.Popen(
    command, env=env, start_new_session=True, stdin=None if interactive else subprocess.DEVNULL
)
sys.exit(process.wait())
"""


@pytest.fixture
def fake_daemon(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "daemon-bin" / "docker"
    binary.parent.mkdir()
    binary.write_text(f"#!{sys.executable}\n{FAKE_DAEMON}")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary.parent}:{Path(sys.executable).parent}:/usr/bin:/bin")


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except FileNotFoundError:
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["timeout", "cancelled"])
async def test_a_command_that_times_out_or_is_cancelled_is_killed_inside_the_container(
    fake_daemon, tmp_path: Path, ending: str
):
    """Docker leaves a command running when its client is killed; platoon kills it by its marker from inside the
    container, with what it started, so it cannot change the sandbox later (while the tests run, say)."""

    sandbox = await DockerProvider().start(SandboxSpec(image="docker://python:3.12"))
    pids, late = tmp_path / "pids", tmp_path / "late"
    command = f"(sleep 30; touch {late}) & echo $$ $! > {pids}; wait"
    try:
        if ending == "timeout":
            result = await sandbox.exec(command, timeout_s=0.5)
            assert result.timed_out
        else:
            running = asyncio.ensure_future(sandbox.exec(command))
            while not pids.exists():
                await asyncio.sleep(0.05)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
        shell, child = (int(pid) for pid in pids.read_text().split())
        assert not _running(shell) and not _running(child)
        assert not late.exists()
    finally:
        await sandbox.stop()
