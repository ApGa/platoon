"""Sandboxes that a process which is gone left running, as a killed rollout worker or trainer leaves them, for the
reaping tests."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from pathlib import Path


async def leave_sandbox(root: Path, label: dict[str, str], command: str = "sleep 300") -> int:
    """Start a process sandbox under ``root``, labelled ``label``, from a process that then exits, leaving ``command``
    (a sleep) running in it in the background; returns its pid."""

    script = textwrap.dedent(
        f"""
        import asyncio
        from platoon.sandbox import SandboxSpec
        from platoon.sandbox import cleanup
        from platoon.sandbox.process import ProcessProvider

        async def main():
            cleanup.set_labels({label!r})
            sandbox = await ProcessProvider(root={str(root)!r}).start(SandboxSpec(image="local"))
            print((await sandbox.exec({command + " > /dev/null 2>&1 & echo $!"!r})).stdout.strip())

        asyncio.run(main())
        """
    )
    done = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return int(done.stdout.split()[-1])


def alive(pid: int) -> bool:
    """Whether ``pid`` runs (a zombie does not)."""

    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False
