"""The sandboxes of ``platoon eval`` rollouts in worker processes.

Each slot's process labels its sandboxes and lists how to reap them; the slots reap them when they kill a process, when
one dies, when a rollout is cancelled and when they close.  A rollout past its hard timeout starts no sandbox, stops
those it holds and then kills its process group, or the backstop kills it when it holds its event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

from platoon.config_defs import RolloutConfig
from platoon.eval.workflow import _ProcessSlots
from platoon.sandbox import SandboxSpec
from platoon.sandbox import cleanup as sandbox_cleanup
from platoon.sandbox.process import ProcessProvider
from platoon.task import Task

WORKER = sandbox_cleanup.WORKER_LABEL


def _note(path: str, line: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


class RecordingSandbox:
    """A sandbox of :class:`RecordingProvider`; stopping it notes whether new sandboxes were refused by then."""

    def __init__(self, notes: str, token: int) -> None:
        self.id = f"recording-{token}"
        self._notes = notes
        self._token = token

    async def stop(self) -> None:
        try:
            sandbox_cleanup.check_running()
            refused = False
        except sandbox_cleanup.SandboxesShutDown:
            refused = True
        _note(self._notes, f"stopped refused={refused}")
        sandbox_cleanup.unregister(self._token)


class RecordingProvider:
    """A provider whose sandboxes run nothing: it registers them as the built-in providers do and notes their labels,
    and notes each reap, which takes ``reap_s``."""

    def __init__(self, notes: str, reap_s: float = 0.0) -> None:
        self.notes = notes
        self.reap_s = reap_s

    async def start(self, spec: SandboxSpec) -> RecordingSandbox:
        labels = {**spec.labels, **sandbox_cleanup.labels()}
        options = {"notes": self.notes, "reap_s": self.reap_s}
        token = sandbox_cleanup.register(provider=sandbox_cleanup.provider_ref(self), options=options, labels=labels)
        sandbox = RecordingSandbox(self.notes, token)
        sandbox_cleanup.attach(token, sandbox)
        _note(self.notes, f"started {json.dumps(labels)}")
        return sandbox

    async def reap(self, labels: Mapping[str, str]) -> int:
        _note(self.notes, f"reaping {json.dumps(dict(labels))}")
        await asyncio.sleep(self.reap_s)
        _note(self.notes, f"reaped {json.dumps(dict(labels))}")
        return 1


async def _start_sandboxes(notes: str, root: str, reap_s: float = 0.0) -> None:
    """Start a process sandbox with a background command and a recording sandbox; note the command's pid and the
    process sandbox's labels."""

    sandbox = await ProcessProvider(root=root).start(SandboxSpec(image="local"))
    pid = (await sandbox.exec("sleep 300 > /dev/null 2>&1 & echo $!")).stdout.strip()
    await RecordingProvider(notes, reap_s).start(SandboxSpec(image="local"))
    _note(notes, f"pid {pid} {json.dumps(sandbox.spec.labels)}")


# Run in slot processes.


def warm_up() -> int:
    return os.getpid()


def start_sandboxes_and_hang(notes: str, root: str, reap_s: float) -> None:
    async def run() -> None:
        await _start_sandboxes(notes, root, reap_s)
        await asyncio.sleep(3600)

    asyncio.run(run())


def start_a_sandbox(notes: str) -> None:
    asyncio.run(RecordingProvider(notes).start(SandboxSpec(image="local")))


def start_a_sandbox_and_die(notes: str) -> None:
    start_a_sandbox(notes)
    os._exit(3)


# Run by run_rollout_subprocess.


async def rollout_holding_sandboxes(task: Task, config: RolloutConfig) -> dict:
    """Starts its sandboxes, then waits past its hard timeout; with ``misc["block"]``, holding its event loop."""

    await _start_sandboxes(task.misc["notes"], task.misc["root"])
    if task.misc.get("block"):
        time.sleep(3600)
    await asyncio.sleep(3600)
    return {}


def _lines(notes: Path, prefix: str) -> list[str]:
    text = notes.read_text() if notes.exists() else ""
    return [line.removeprefix(prefix) for line in text.splitlines() if line.startswith(prefix)]


async def _until(condition: Callable[[], bool], timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def _until_dead(pid: int, timeout_s: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


def _started(notes: Path) -> tuple[int, dict[str, str]]:
    (line,) = _lines(notes, "pid ")
    pid, labels = line.split(" ", 1)
    return int(pid), json.loads(labels)


def test_a_slot_process_killed_past_its_timeout_has_its_sandboxes_reaped(tmp_path):
    notes, root = tmp_path / "notes", tmp_path / "sandboxes"

    async def run() -> str:
        slots = _ProcessSlots(1)
        try:
            await slots.run(300, warm_up)  # the process starts first, so the timeout below leaves its start out
            hung = asyncio.create_task(slots.run(10, start_sandboxes_and_hang, str(notes), str(root), 0.0))
            await _until(lambda: bool(_lines(notes, "pid ")))
            directory = slots._directory
            assert directory is not None and stat.S_IMODE(os.stat(directory).st_mode) == 0o700
            (listing,) = Path(directory).glob("*.sandboxes")
            assert stat.S_IMODE(listing.stat().st_mode) == 0o600
            _, labels = _started(notes)
            assert {record.provider: record.labels for record in sandbox_cleanup.read_mirror(str(listing))} == {
                "platoon.sandbox.process:ProcessProvider": {WORKER: labels[WORKER]},
                f"{__name__}:RecordingProvider": {WORKER: labels[WORKER]},
            }

            with pytest.raises(TimeoutError, match="hard timeout after 10s"):
                await hung
            assert not listing.exists()
        finally:
            await slots.aclose()
        assert not os.path.exists(directory)
        return labels[WORKER]

    worker = asyncio.run(run())

    pid, labels = _started(notes)
    assert worker.startswith("eval-") and labels == {WORKER: worker}
    assert _until_dead(pid)  # the command the killed process left running in its sandbox
    assert list(root.iterdir()) == []
    assert _lines(notes, "reaped ") == [json.dumps({WORKER: worker})]


def test_a_cancelled_rollout_has_its_sandboxes_reaped_also_when_it_is_cancelled_again(tmp_path):
    notes, root = tmp_path / "notes", tmp_path / "sandboxes"

    async def run() -> None:
        slots = _ProcessSlots(1)
        try:
            running = asyncio.create_task(slots.run(600, start_sandboxes_and_hang, str(notes), str(root), 2.0))
            await _until(lambda: bool(_lines(notes, "pid ")))
            running.cancel()
            await _until(lambda: bool(_lines(notes, "reaping ")), 60)
            running.cancel()  # while the reap runs
            with pytest.raises(asyncio.CancelledError):
                await running
        finally:
            await slots.aclose()  # waits for the reap

    asyncio.run(run())

    pid, labels = _started(notes)
    assert _lines(notes, "reaped ") == [json.dumps({WORKER: labels[WORKER]})]
    assert _until_dead(pid)
    assert list(root.iterdir()) == []


def test_the_sandboxes_of_a_slot_process_that_died_are_reaped_and_a_failing_reap_keeps_its_error(
    tmp_path, monkeypatch, caplog
):
    notes = tmp_path / "notes"
    reap_records = sandbox_cleanup.reap_records

    async def reap_then_fail(records, timeout_s):
        await reap_records(records, timeout_s)
        raise RuntimeError("the reap failed")

    monkeypatch.setattr(sandbox_cleanup, "reap_records", reap_then_fail)

    async def run() -> None:
        slots = _ProcessSlots(1)
        try:
            with pytest.raises(BrokenProcessPool):
                await slots.run(600, start_a_sandbox_and_die, str(notes))
        finally:
            await slots.aclose()  # tries again: the failed reap kept the file

    asyncio.run(run())

    assert _lines(notes, "started ") != [] and set(_lines(notes, "reaped ")) == set(_lines(notes, "started "))
    assert "Cannot reap the sandboxes listed in" in caplog.text


def test_closing_the_slots_reaps_the_sandboxes_their_processes_left(tmp_path):
    notes = tmp_path / "notes"

    async def run() -> str:
        slots = _ProcessSlots(1)
        await slots.run(600, start_a_sandbox, str(notes))
        directory = slots._directory
        assert _lines(notes, "reaped ") == []
        await slots.aclose()
        return directory

    directory = asyncio.run(run())

    assert _lines(notes, "reaped ") == _lines(notes, "started ") != []
    assert not os.path.exists(directory)


def test_a_slot_process_labels_its_sandboxes_with_the_evaluations_labels_too(tmp_path):
    """The labels in force when a slot starts its process (the evaluation's run label) go on its sandboxes beside its
    worker label, and are reaped with it."""

    notes = tmp_path / "notes"
    run = sandbox_cleanup.run_label(tmp_path / "results")

    async def run_slots() -> None:
        slots = _ProcessSlots(1)
        try:
            await slots.run(600, start_a_sandbox, str(notes))
        finally:
            await slots.aclose()

    with sandbox_cleanup.labelled(run):
        asyncio.run(run_slots())

    (started,) = _lines(notes, "started ")
    labels = json.loads(started)
    assert labels == {**run, WORKER: labels[WORKER]} and labels[WORKER].startswith("eval-")
    assert _lines(notes, "reaped ") == [started]


# The process that runs the rollout: a short hard timeout, stop window and backstop, and the modules the rollout
# imports already imported, since the alarm runs from the start.
_HARD_TIMEOUT_SCRIPT = """
import sys
sys.path[:0] = {path!r}
import platoon.config_defs, platoon.train.workers.protocol
from platoon.eval import subprocess_worker
from platoon.sandbox import cleanup
from platoon.task import Task

subprocess_worker._hard_timeout_seconds = lambda *_: 4
subprocess_worker.SANDBOX_STOP_SECONDS = 5.0
subprocess_worker.KILL_BACKSTOP_SECONDS = 8.0
cleanup.set_labels({{cleanup.WORKER_LABEL: "eval-test"}})
cleanup.mirror_to({listing!r})
subprocess_worker.run_rollout_subprocess(
    "{module}:rollout_holding_sandboxes", Task(goal="wait", id="t", misc={misc!r}), {{"timeout": 1}}
)
"""


def _run_past_the_hard_timeout(tmp_path: Path, *, block: bool) -> tuple[int, float, Path]:
    """Run ``rollout_holding_sandboxes`` past its hard timeout in a process of its own; returns its exit code, the
    seconds from its sandboxes' start to its end, and the file that lists its sandboxes."""

    notes, listing = tmp_path / "notes", tmp_path / "rollout.sandboxes"
    misc = {"notes": str(notes), "root": str(tmp_path / "sandboxes"), "block": block}
    script = _HARD_TIMEOUT_SCRIPT.format(path=sys.path, listing=str(listing), module=__name__, misc=misc)
    process = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 300
        while not _lines(notes, "pid ") and process.poll() is None:
            assert time.monotonic() < deadline, "the rollout never started its sandboxes"
            time.sleep(0.05)
        started = time.monotonic()
        returncode = process.wait(timeout=120)
        return returncode, time.monotonic() - started, listing
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_a_rollout_past_its_hard_timeout_stops_its_sandboxes_before_its_process_group_is_killed(tmp_path):
    returncode, _, listing = _run_past_the_hard_timeout(tmp_path, block=False)

    assert returncode == -signal.SIGKILL
    notes = tmp_path / "notes"
    assert _lines(notes, "stopped ") == ["refused=True"]  # stopped after no new sandbox could start
    pid, _ = _started(notes)
    assert _until_dead(pid)
    assert list((tmp_path / "sandboxes").iterdir()) == []
    assert sandbox_cleanup.read_mirror(str(listing)) == []


def test_a_rollout_holding_its_event_loop_past_its_hard_timeout_is_killed_by_the_backstop(tmp_path):
    returncode, seconds, listing = _run_past_the_hard_timeout(tmp_path, block=True)

    assert returncode == -signal.SIGKILL
    assert seconds > 5  # the backstop, 8 s after the 4 s hard timeout, not the hard timeout itself
    notes = tmp_path / "notes"
    assert _lines(notes, "stopped ") == []
    pid, _ = _started(notes)
    assert _until_dead(pid)  # the sandbox command's process group, killed with the rollout's
    records = sandbox_cleanup.read_mirror(str(listing))
    assert {record.provider for record in records} == {
        "platoon.sandbox.process:ProcessProvider",
        f"{__name__}:RecordingProvider",
    }

    # What it left running, its parent reaps by its label.
    assert asyncio.run(sandbox_cleanup.reap_records(records, 30.0)) == 2
    assert list((tmp_path / "sandboxes").iterdir()) == []
    assert _lines(notes, "reaped ") == [json.dumps({WORKER: "eval-test"})]
