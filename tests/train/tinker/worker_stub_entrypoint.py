"""A dependency-free stand-in for the rollout worker entrypoint.

Supervisor tests care about launch, deadlines, killing a process *tree* and
classifying what came back -- not about torch.  This stub therefore imports
nothing from platoon (or tinker) and is driven entirely by environment
variables, so a supervisor test costs milliseconds instead of a 20 s import per
child.  The real entrypoint is exercised by ``test_worker_entrypoint.py``.

Environment:
    STUB_MODE           what to do (see ``main``)
    STUB_RESPONSE_SRC   a pre-pickled RolloutWorkerResponse to publish
    STUB_RESPONSE_DST   where to publish it (``.diag`` is derived from it)
    STUB_READY_PATH     touched to signal readiness
    STUB_MARKER_DIR     where informational markers are dropped
    STUB_EVENTS_DIR     where ``events_then_killed`` leaves a temporary event file
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys
import time

EXIT_RSS_LIMIT = 87


def _touch_ready() -> None:
    path = os.environ.get("STUB_READY_PATH")
    if path:
        with open(path, "wb") as handle:
            handle.write(b"ready\n")


def _publish_response(*, truncate: bool = False) -> None:
    source = os.environ.get("STUB_RESPONSE_SRC")
    destination = os.environ.get("STUB_RESPONSE_DST")
    if not source or not destination:
        return
    with open(source, "rb") as handle:
        payload = handle.read()
    if truncate:
        payload = payload[: max(1, len(payload) // 2)]
    tmp = destination + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(payload)
    os.replace(tmp, destination)


def _write_diag(phase: str, **extra) -> None:
    destination = os.environ.get("STUB_RESPONSE_DST")
    if not destination:
        return
    fields = {"phase": phase, "elapsed": "1.0", "pid": str(os.getpid())}
    fields.update({key: str(value) for key, value in extra.items()})
    payload = ("\t".join(f"{key}={value}" for key, value in fields.items()) + "\n").encode()
    with open(destination + ".diag", "wb") as handle:
        handle.write(payload)


def _marker(name: str, content: str) -> None:
    directory = os.environ.get("STUB_MARKER_DIR")
    if not directory:
        return
    with open(os.path.join(directory, name), "w") as handle:
        handle.write(content)


def _forever() -> None:
    while True:
        time.sleep(3600)


def main() -> int:
    mode = os.environ.get("STUB_MODE", "ok")
    if len(sys.argv) > 1:
        # The real worker unlinks the request as soon as it has read it,
        # because it carries the gateway API key.
        try:
            os.unlink(sys.argv[1])
        except OSError:
            pass

    if mode == "launch_marker":
        _marker(f"launch-{os.getpid()}", str(os.getpid()))
    if mode == "parent":
        _marker(f"parent-{os.getpid()}", str(os.getppid()))
        _touch_ready()
        _publish_response()
        return 0

    _touch_ready()

    if mode == "ok":
        _publish_response()
        return 0
    if mode == "write_then_sleep":
        _publish_response()
        _forever()
    if mode == "truncated":
        _publish_response(truncate=True)
        return 0
    if mode == "clean_no_response":
        return 0
    if mode == "spin":
        while True:
            pass
    if mode == "sleep":
        _forever()
    if mode == "segfault":
        ctypes.string_at(0)
    if mode == "trap_sigterm":
        signal.signal(signal.SIGTERM, lambda *_: None)
        _marker(f"trapped-{os.getpid()}", "1")
        _forever()
    if mode == "diag_timeout":
        _write_diag("rollout", ptc_cells_started=4, ptc_cells_finished=3, ptc_cells_stuck_at_exit=1)
        os.killpg(os.getpgrp(), signal.SIGKILL)
    if mode == "diag_startup":
        _write_diag("startup")
        os.killpg(os.getpgrp(), signal.SIGKILL)
    if mode == "diag_rss":
        _write_diag("rss", rss_bytes=99, limit_bytes=1)
        os._exit(EXIT_RSS_LIMIT)
    if mode == "rss_exit_only":
        os._exit(EXIT_RSS_LIMIT)
    if mode == "check_stdin":
        try:
            target = os.readlink("/proc/self/fd/0")
        except OSError:
            target = "<none>"
        _marker("stdin", target)
        _publish_response()
        return 0
    if mode == "crash":
        return 3
    if mode == "events_then_killed":
        # Killed mid-write of its rollout's event file: the hidden temporary stays behind.
        directory = os.environ["STUB_EVENTS_DIR"]
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f".events_task_c1.jsonl.{os.getpid()}.tmp"), "w") as handle:
            handle.write("{}\n")
        os.kill(os.getpid(), signal.SIGKILL)
    if mode == "spawn_child_then_spin":
        # A grandchild in the same process group: the supervisor must reap the
        # whole tree, not just the direct child.
        pid = os.fork()
        if pid == 0:
            _marker(f"grandchild-{os.getpid()}", str(os.getpid()))
            _forever()
        _marker(f"grandchild-pid-{os.getpid()}", str(pid))
        while True:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
