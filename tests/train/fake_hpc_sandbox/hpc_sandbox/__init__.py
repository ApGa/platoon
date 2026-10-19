"""A stand-in for the ``hpc-sandbox`` package that keeps its sessions on disk, so that one process starts sandboxes
and another reaps them, as with the real one: put this directory's parent first on ``sys.path`` (or ``PYTHONPATH``).

A sandbox is a session file ``<scratch_root>/sessions/<id>.json`` naming the process that started it and its labels;
it runs nothing.  ``reap(labels)`` removes the sessions whose labels include ``labels`` and whose owner is gone, and
leaves a live owner's, as hpc-sandbox's janitor does.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
from pathlib import Path

Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
SandboxSpec = dataclasses.make_dataclass(
    "SandboxSpec", ["image", "workdir", "env", "mounts", "network", "cpus", "memory_mb", "labels"]
)
ExecResult = dataclasses.make_dataclass("ExecResult", ["exit_code", "stdout", "stderr", "timed_out"])


@dataclasses.dataclass
class JanitorReport:
    stopped: list = dataclasses.field(default_factory=list)
    removed: list = dataclasses.field(default_factory=list)
    live: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


class Sandbox:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.id = path.stem

    async def exec(self, cmd, **kwargs):
        return ExecResult(0, "", "", False)

    async def upload(self, src, dst) -> None:
        pass

    async def download(self, src, dst):
        return None

    async def stop(self) -> None:
        self.path.unlink(missing_ok=True)


class SingularityProvider:
    def __init__(self, scratch_root=None, **options) -> None:
        self.sessions = Path(scratch_root or os.environ.get("TMPDIR", "/tmp")) / "sessions"
        self.capabilities = frozenset({"isolation", "network_none", "snapshot"})

    async def start(self, spec) -> Sandbox:
        self.sessions.mkdir(parents=True, exist_ok=True)
        path = self.sessions / f"hpcsb-{uuid.uuid4().hex[:12]}.json"
        path.write_text(json.dumps({"owner": os.getpid(), "labels": dict(spec.labels)}))
        return Sandbox(path)

    async def reap(self, labels) -> JanitorReport:
        report = JanitorReport()
        for path in sorted(self.sessions.glob("*.json")):
            session = json.loads(path.read_text())
            if any(session["labels"].get(key) != value for key, value in labels.items()):
                continue
            if _alive(session["owner"]):
                report.live.append(path.stem)
                continue
            path.unlink()
            report.removed.append(path.stem)
        return report


EnrootProvider = SingularityProvider
