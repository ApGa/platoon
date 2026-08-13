#!/usr/bin/env python3
"""Kill one Toolathlon worker and prove an unrelated live session survives."""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from openreward import OpenReward

ENVIRONMENT = "toolathlongym"
TASK_NAME = "canvas-announcement-summary"


def text_of(output: object) -> str:
    blocks = getattr(output, "blocks", [])
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


def selected_task(environment: object) -> object:
    for task in environment.list_tasks(split="train"):
        task_spec = getattr(task, "task_spec", {})
        if task_spec.get("task_name") == TASK_NAME:
            return task
    raise RuntimeError(f"Toolathlon recovery task not found: {TASK_NAME}")


def read_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return value if value > 1 else None


def live_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError):
        return False
    return state != "Z"


def endpoint_healthy(url: str) -> bool:
    try:
        with urlopen(url, timeout=5) as response:  # noqa: S310 - localhost only
            return 100 <= response.status < 500
    except HTTPError as error:
        return 100 <= error.code < 500
    except (OSError, URLError):
        return False


def main() -> None:
    pid_path_value = os.environ.get("OPENREWARD_KILL_WORKER_PID_FILE")
    killed_worker_url = os.environ.get("OPENREWARD_KILLED_WORKER_URL")
    if not pid_path_value or not killed_worker_url:
        raise RuntimeError(
            "OPENREWARD_KILL_WORKER_PID_FILE and OPENREWARD_KILLED_WORKER_URL are required"
        )
    pid_path = Path(pid_path_value)
    old_pid = read_pid(pid_path)
    if old_pid is None or not live_process(old_pid):
        raise RuntimeError(f"worker PID record is absent or stale: {pid_path}")
    if old_pid == os.getpid():
        raise RuntimeError("refusing to signal the recovery probe itself")

    # OPENREWARD_SESSION_URL points directly at worker 1.  Its in-flight state
    # must remain valid while the supervisor replaces worker 0.
    expected_marker = "unrelated-worker-session-survived"
    with OpenReward(api_key="local") as client:
        environment = client.environments.get(name=ENVIRONMENT)
        task = selected_task(environment)
        with environment.session(task=task) as session:
            write = session.call_tool(
                "python_execute",
                {
                    "code": (
                        "from pathlib import Path; "
                        f"Path('recovery-marker.txt').write_text({expected_marker!r})"
                    )
                },
            )
            if "error" in text_of(write).lower():
                raise RuntimeError(f"pre-kill marker write failed: {text_of(write)}")

            os.kill(old_pid, signal.SIGKILL)
            deadline = time.monotonic() + 90
            new_pid = None
            while time.monotonic() < deadline:
                candidate = read_pid(pid_path)
                if (
                    candidate is not None
                    and candidate != old_pid
                    and live_process(candidate)
                    and endpoint_healthy(killed_worker_url)
                ):
                    new_pid = candidate
                    break
                time.sleep(0.25)
            if new_pid is None:
                raise RuntimeError(
                    f"worker was not replaced within 90 seconds (old_pid={old_pid})"
                )

            read = session.call_tool(
                "python_execute",
                {
                    "code": (
                        "from pathlib import Path; "
                        "print(Path('recovery-marker.txt').read_text())"
                    )
                },
            )
            marker_output = text_of(read)
            if expected_marker not in marker_output:
                raise RuntimeError(
                    "session on the unrelated worker lost state after peer restart: "
                    f"{marker_output!r}"
                )
            terminal = session.call_tool("claim_done", {})
            if not bool(getattr(terminal, "finished", False)):
                raise RuntimeError("unrelated session could not finish after peer restart")

    print(
        json.dumps(
            {
                "toolathlon_worker_recovery": "ok",
                "killed_pid": old_pid,
                "replacement_pid": new_pid,
                "unrelated_session_preserved": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
