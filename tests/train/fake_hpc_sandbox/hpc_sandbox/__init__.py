"""A stand-in for the ``hpc-sandbox`` package (0.5's API) that keeps its sessions on disk, so that one process starts
sandboxes and pods and another reaps them, as with the real one: put this directory's parent first on ``sys.path`` (or
``PYTHONPATH``).

A sandbox is a session file ``<scratch_root>/sessions/<id>.json`` naming the process that started it and its labels;
it runs nothing.  A pod is such a file in ``<scratch_root>/pods/``, beside the pod's directory, and its services run
nothing either: a command in one exits 0 and prints nothing, but for the listing of a sandbox's own processes, which
prints one; a test that wants other answers replaces ``PodService.exec``, and ends a service's command by setting its
``exit_status``.  ``reap(labels)`` removes the sessions and pods whose labels include ``labels`` and whose owner is
gone, and leaves a live owner's, as hpc-sandbox's janitor does; it stops this process's own pods with those labels
first.  ``map_subordinate_ids()`` succeeds unless ``SUBORDINATE_IDS`` is False.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from pathlib import Path

Mount = dataclasses.make_dataclass("Mount", ["source", "target", "read_only"])
SandboxSpec = dataclasses.make_dataclass(
    "SandboxSpec", ["image", "workdir", "env", "mounts", "network", "cpus", "memory_mb", "labels"]
)
ExecResult = dataclasses.make_dataclass("ExecResult", ["exit_code", "stdout", "stderr", "timed_out"])
SUBORDINATE_IDS = True  # whether this user has subordinate ids to map


class SandboxError(RuntimeError):
    pass


class SandboxStartError(SandboxError):
    pass


class SandboxLostError(SandboxError):
    pass


@dataclasses.dataclass
class JanitorReport:
    stopped: list = dataclasses.field(default_factory=list)
    removed: list = dataclasses.field(default_factory=list)
    live: list = dataclasses.field(default_factory=list)
    errors: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class PodSpec:
    network: str = "none"
    upstreams: dict = dataclasses.field(default_factory=dict)
    upstream_ports: dict = dataclasses.field(default_factory=dict)
    labels: dict = dataclasses.field(default_factory=dict)
    users: str = "single"
    exposes: list = dataclasses.field(default_factory=list)
    hosts: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ServiceSpec:
    name: str
    sandbox: object
    command: list | None = None
    healthcheck: object = None
    shm_mb: int | None = None
    ulimits: dict = dataclasses.field(default_factory=dict)
    run_as: str = "root"


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")


def _session(path: Path, labels: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"owner": os.getpid(), "labels": dict(labels)}))


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


class PodService:
    """A service of a :class:`SessionPod`: its spec, and its command's exit status once a test ends it."""

    run_modes = ("root", "caller", "subids")

    def __init__(self, pod: SessionPod, spec: ServiceSpec) -> None:
        self.pod = pod
        self.service = spec
        self.name = spec.name
        self.id = f"{pod.id}-{spec.name}"
        self.exit_status: int | None = None
        self.stopped = False

    async def exec(self, cmd, **kwargs):
        if self.exit_status is not None or self.stopped:
            raise SandboxLostError(f"service {self.name!r} is gone")
        listing = isinstance(cmd, list) and cmd[-1:] == ["platoon-list"]
        return ExecResult(0, "1:10\n" if listing else "", "", False)

    async def upload(self, src, dst) -> None:
        pass

    async def download(self, src, dst):
        return None

    async def stop(self) -> None:
        self.stopped = True


class SessionPod:
    def __init__(self, provider: SingularityProvider, spec: PodSpec) -> None:
        self.id = f"hpcsb-{uuid.uuid4().hex[:12]}"
        self.spec = spec
        self.record = provider.pod_root / f"{self.id}.json"
        self.path = provider.pod_root / self.id
        self.started: list[ServiceSpec] = []  # what platoon asked to start, in order
        self.services: dict[str, PodService] = {}
        self.network = spec.network
        self.endpoints: dict[int, str] = {}
        self.stops = 0
        self._provider = provider
        ports = dict(spec.upstream_ports)
        for name in spec.upstreams:
            ports.setdefault(name, 18080 + len(ports))
        self.routes = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}

    async def start_service(self, spec: ServiceSpec, wait: str = "started") -> PodService:
        if spec.run_as == "subids" and self.spec.users != "subids":
            raise ValueError("run_as='subids' needs a pod started with PodSpec(users='subids')")
        if spec.run_as not in self._provider.service_type.run_modes:
            raise ValueError(f"the provider runs no services with run_as={spec.run_as!r}")
        self.started.append(spec)
        service = self.services[spec.name] = PodService(self, spec)
        return service

    async def health(self, name: str) -> str:
        if self.stops:
            raise SandboxError(f"pod {self.id} is not running")
        return "exited" if self.services[name].exit_status is not None else "healthy"

    async def logs(self, name: str) -> str:
        return f"the log of {name}\n"

    async def set_network(self, mode: str) -> None:
        self.network = mode

    async def expose(self, port: int) -> str:
        self.endpoints[port] = str(self.path / "exposed" / f"{port}.sock")
        return self.endpoints[port]

    async def stop(self) -> None:
        self.stops += 1
        for service in self.services.values():
            service.stopped = True
        shutil.rmtree(self.path, ignore_errors=True)
        self.record.unlink(missing_ok=True)
        self._provider.pods.discard(self)


class SingularityProvider:
    service_type = PodService

    def __init__(self, scratch_root=None, **options) -> None:
        root = Path(scratch_root or os.environ.get("TMPDIR", "/tmp"))
        self.sessions = root / "sessions"
        self.pod_root = root / "pods"
        self.capabilities = frozenset({"isolation", "network_none", "snapshot", "pods"})
        self.pods: set[SessionPod] = set()  # those started and not stopped
        self.started_pods: list[SessionPod] = []

    async def start(self, spec) -> Sandbox:
        path = self.sessions / f"hpcsb-{uuid.uuid4().hex[:12]}.json"
        _session(path, spec.labels)
        return Sandbox(path)

    async def start_pod(self, spec: PodSpec) -> SessionPod:
        pod = SessionPod(self, spec)
        pod.path.mkdir(parents=True)
        _session(pod.record, spec.labels)
        self.pods.add(pod)
        self.started_pods.append(pod)
        return pod

    async def map_subordinate_ids(self) -> None:
        if not SUBORDINATE_IDS:
            raise SandboxError("pods with users='subids' cannot start here: no range of subordinate ids")

    async def reap(self, labels) -> JanitorReport:
        report = JanitorReport()
        for pod in [pod for pod in self.pods if all(pod.spec.labels.get(k) == v for k, v in labels.items())]:
            await pod.stop()
            report.stopped.append(pod.id)
        for path in sorted([*self.sessions.glob("*.json"), *self.pod_root.glob("*.json")]):
            session = json.loads(path.read_text())
            if any(session["labels"].get(key) != value for key, value in labels.items()):
                continue
            if _alive(session["owner"]):
                report.live.append(path.stem)
                continue
            path.unlink()
            shutil.rmtree(path.with_suffix(""), ignore_errors=True)  # a pod's directory
            report.removed.append(path.stem)
        return report


EnrootProvider = SingularityProvider
