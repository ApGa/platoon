"""The sandbox registry (``platoon.sandbox.cleanup``): labels, the mirror file, refusal after shutdown, and reaping
by label, in this process and from another one."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from platoon.sandbox import SandboxSpec, get_provider
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.process import RECORD_FILE, ProcessProvider

WORKER = sandbox_cleanup.WORKER_LABEL
RUN = sandbox_cleanup.RUN_LABEL


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(sandbox_cleanup, "_live", {})
    monkeypatch.setattr(sandbox_cleanup, "_labels", {})
    monkeypatch.setattr(sandbox_cleanup, "_mirror_path", None)
    monkeypatch.setattr(sandbox_cleanup, "_shut_down", False)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


async def _until_dead(pid: int, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while _alive(pid):
        assert time.monotonic() < deadline, f"process {pid} is still alive"
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_sandboxes_carry_this_process_labels_and_are_mirrored_until_they_stop(tmp_path):
    mirror = tmp_path / "worker.sandboxes"
    sandbox_cleanup.set_labels({WORKER: "req-1"})
    sandbox_cleanup.mirror_to(str(mirror))
    provider = ProcessProvider(root=tmp_path / "sandboxes")

    sandbox = await provider.start(SandboxSpec(image="local", labels={"platoon.task": "t"}))

    assert sandbox.spec.labels == {"platoon.task": "t", WORKER: "req-1"}
    assert stat.S_IMODE(mirror.stat().st_mode) == 0o600
    record = sandbox_cleanup.SandboxRecord(
        provider="platoon.sandbox.process:ProcessProvider",
        options={"root": str(tmp_path / "sandboxes")},
        labels={WORKER: "req-1"},
    )
    assert sandbox_cleanup.read_mirror(str(mirror)) == [record]
    second = await provider.start(SandboxSpec(image="local"))
    assert sandbox_cleanup.read_mirror(str(mirror)) == [record]  # one record per way to reap
    assert sandbox_cleanup.live_sandboxes() == [sandbox, second]

    await sandbox.stop()
    await second.stop()
    assert sandbox_cleanup.read_mirror(str(mirror)) == [] and sandbox_cleanup.live_sandboxes() == []


@pytest.mark.asyncio
async def test_a_sandbox_without_labels_is_not_mirrored(tmp_path):
    mirror = tmp_path / "worker.sandboxes"
    sandbox_cleanup.mirror_to(str(mirror))

    sandbox = await ProcessProvider(root=tmp_path / "sandboxes").start(SandboxSpec(image="local"))

    assert json.loads(mirror.read_text()) == []  # an empty label set would select every sandbox
    assert sandbox_cleanup.live_sandboxes() == [sandbox]
    await sandbox.stop()


def test_a_mirror_that_cannot_be_read_lists_nothing(tmp_path):
    assert sandbox_cleanup.read_mirror(str(tmp_path / "missing")) == []
    garbage = tmp_path / "garbage"
    garbage.write_text("{not json")
    assert sandbox_cleanup.read_mirror(str(garbage)) == []
    unlabelled = tmp_path / "unlabelled"
    unlabelled.write_text(json.dumps([{"provider": "process", "options": {}, "labels": {}}]))
    assert sandbox_cleanup.read_mirror(str(unlabelled)) == []


@pytest.mark.asyncio
async def test_no_sandbox_starts_after_shutdown(tmp_path):
    provider = ProcessProvider(root=tmp_path / "sandboxes")
    sandbox = await provider.start(SandboxSpec(image="local"))

    assert sandbox_cleanup.shut_down() == 1
    with pytest.raises(sandbox_cleanup.SandboxesShutDown):
        await provider.start(SandboxSpec(image="local"))
    with pytest.raises(sandbox_cleanup.SandboxesShutDown):
        await provider.restore(await sandbox.snapshot(), SandboxSpec(image="local"))
    await sandbox.stop()  # stopping still works
    assert [path.name for path in (tmp_path / "sandboxes").iterdir()] == []


@pytest.mark.asyncio
async def test_stop_live_stops_what_the_process_still_holds(tmp_path):
    provider = ProcessProvider(root=tmp_path / "sandboxes")
    sandboxes = [await provider.start(SandboxSpec(image="local")) for _ in range(2)]
    pid = int((await sandboxes[0].exec("sleep 300 > /dev/null 2>&1 & echo $!")).stdout)

    assert await sandbox_cleanup.stop_live(10.0) == 2

    assert sandbox_cleanup.live_sandboxes() == []
    assert not (tmp_path / "sandboxes").exists() or list((tmp_path / "sandboxes").iterdir()) == []
    await _until_dead(pid)
    assert await sandbox_cleanup.stop_live(10.0) == 0


@pytest.mark.asyncio
async def test_the_process_provider_reaps_the_sandboxes_of_a_process_that_is_gone(tmp_path):
    """Another process starts a labelled sandbox with a background command and dies without stopping it."""

    root = tmp_path / "sandboxes"
    script = f"""
import asyncio, sys
sys.path[:0] = {sys.path!r}
from platoon.sandbox import SandboxSpec
from platoon.sandbox import cleanup
from platoon.sandbox.process import ProcessProvider

async def main():
    cleanup.set_labels({{"{WORKER}": "dead-1"}})
    sandbox = await ProcessProvider(root={str(root)!r}).start(SandboxSpec(image="local"))
    result = await sandbox.exec("sleep 300 > /dev/null 2>&1 & echo $!")
    print(result.stdout.strip(), flush=True)

asyncio.run(main())
"""
    owner = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
    pid = int(owner.stdout.split()[-1])  # its process group outlived the owner
    assert _alive(pid)
    (directory,) = root.iterdir()
    survivor = await ProcessProvider(root=root).start(SandboxSpec(image="local", labels={WORKER: "dead-1"}))

    provider = ProcessProvider(root=root)
    assert await provider.reap({WORKER: "other"}) == 0
    assert await provider.reap({WORKER: "dead-1"}) == 2  # this process's own matching sandbox too

    await _until_dead(pid)
    assert not directory.exists()
    await survivor.stop()


def test_a_runs_label_is_its_directory_as_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert sandbox_cleanup.run_label("runs/exp/trial") == {RUN: str(tmp_path / "runs" / "exp" / "trial")}
    assert sandbox_cleanup.run_label(tmp_path / "out") == {RUN: str(tmp_path / "out")}


def test_labelled_adds_labels_inside_the_block_and_puts_back_the_old_ones():
    sandbox_cleanup.set_labels({"keep": "1"})

    with sandbox_cleanup.labelled({RUN: "/runs/a"}) as inside:
        assert inside == sandbox_cleanup.labels() == {"keep": "1", RUN: "/runs/a"}
    assert sandbox_cleanup.labels() == {"keep": "1"}

    with pytest.raises(RuntimeError), sandbox_cleanup.labelled({RUN: "/runs/b"}):
        raise RuntimeError("the run failed")
    assert sandbox_cleanup.labels() == {"keep": "1"}


_DEAD_RUN_OWNER = """
import asyncio, sys
sys.path[:0] = {path!r}
from platoon.sandbox import SandboxSpec
from platoon.sandbox import cleanup
from platoon.sandbox.process import ProcessProvider

async def main():
    cleanup.set_labels({{**cleanup.run_label({run!r}), cleanup.WORKER_LABEL: {worker!r}}})
    sandbox = await ProcessProvider(root={root!r}).start(SandboxSpec(image="local"))
    result = await sandbox.exec("sleep 300 > /dev/null 2>&1 & echo $!")
    print(result.stdout.strip(), sandbox.root.parent, flush=True)

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_a_runs_label_reaps_that_runs_sandboxes_and_no_other_runs(tmp_path):
    """Two runs share a node, and a worker of each died without stopping its sandbox: reaping one run's label stops
    that run's sandbox and leaves the other run's running."""

    root = tmp_path / "sandboxes"
    started = {}
    for run in ("run-a", "run-b"):
        script = _DEAD_RUN_OWNER.format(path=sys.path, run=str(tmp_path / run), worker=f"{run}-w0", root=str(root))
        owner = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=True)
        pid, directory = owner.stdout.split()[-2:]
        started[run] = (int(pid), Path(directory))
    provider = ProcessProvider(root=root)

    assert await provider.reap(sandbox_cleanup.run_label(tmp_path / "run-a")) == 1

    await _until_dead(started["run-a"][0])
    assert not started["run-a"][1].exists()
    assert _alive(started["run-b"][0]) and started["run-b"][1].exists()
    assert await provider.reap({RUN: str(tmp_path / "run-b")}) == 1
    await _until_dead(started["run-b"][0])


@pytest.mark.asyncio
async def test_the_process_provider_leaves_a_live_owners_sandboxes(tmp_path):
    provider = ProcessProvider(root=tmp_path / "sandboxes")
    sandbox = await provider.start(SandboxSpec(image="local", labels={WORKER: "live-1"}))
    record = sandbox._base_dir / RECORD_FILE
    data = json.loads(record.read_text())
    data["owner_pid"] = os.getppid()  # a live process other than this one
    data["owner_start_time"] = None
    record.write_text(json.dumps(data))

    assert await provider.reap({WORKER: "live-1"}) == 0
    assert record.exists()
    with pytest.raises(ValueError, match="needs labels"):
        await provider.reap({})
    await sandbox.stop()


@pytest.mark.asyncio
async def test_reap_records_rebuilds_each_provider_once_and_never_raises(tmp_path, caplog):
    root = tmp_path / "sandboxes"
    sandbox_cleanup.set_labels({WORKER: "req-9"})
    sandbox = await ProcessProvider(root=root).start(SandboxSpec(image="local"))
    sandbox_cleanup.set_labels({})
    ours = sandbox_cleanup.SandboxRecord(
        provider="platoon.sandbox.process:ProcessProvider", options={"root": str(root)}, labels={WORKER: "req-9"}
    )
    broken = sandbox_cleanup.SandboxRecord(provider="no.such.module:Provider", options={}, labels={WORKER: "req-9"})

    assert await sandbox_cleanup.reap_records([ours, ours, broken], timeout_s=30.0) == 1

    assert not sandbox._base_dir.exists()
    assert "Cannot reap the sandboxes labelled" in caplog.text


@pytest.mark.asyncio
async def test_the_docker_provider_reaps_containers_by_label(tmp_path, monkeypatch):
    binary = tmp_path / "bin" / "docker"
    binary.parent.mkdir()
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "open(os.environ['FAKE_DOCKER_LOG'], 'a').write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'ps' and '--quiet' in args and '--filter=label=platoon.stack' not in args:\n"
        "    print('c1\\nc2')\n"
        "if args[0] == 'images':\n"
        "    print('i1')\n"
    )
    binary.chmod(0o755)
    log = tmp_path / "docker.log"
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(log))
    monkeypatch.setenv("PATH", f"{binary.parent}:{Path(sys.executable).parent}:/usr/bin:/bin")

    provider = get_provider("docker", stack_root=tmp_path / "stacks")
    assert await provider.reap({WORKER: "req-2"}) == 2

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls == [
        # The stacks first (none here: platoon.sandbox.compose, tests/sandbox/test_sandbox_compose.py), then the rest.
        [
            "ps",
            "--all",
            f"--filter=label={WORKER}=req-2",
            "--filter=label=platoon.stack",
            "--format",
            '{{.Label "platoon.stack"}}',
        ],
        ["ps", "--all", "--quiet", f"--filter=label={WORKER}=req-2", "--filter=label=platoon.stack"],
        ["network", "ls", "--quiet", f"--filter=label={WORKER}=req-2", "--filter=label=platoon.stack"],
        ["ps", "--all", "--quiet", f"--filter=label={WORKER}=req-2"],
        ["rm", "--force", "--volumes", "c1", "c2"],
        ["images", "--quiet", "--filter=reference=platoon-snapshot", f"--filter=label={WORKER}=req-2"],
        ["rmi", "--force", "i1"],
    ]
    with pytest.raises(ValueError, match="needs labels"):
        await get_provider("docker").reap({})
