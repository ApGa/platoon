# Write a sandbox provider

**Goal.** Run task sandboxes on infrastructure platoon does not support out of the box: a cloud sandbox service, a
cluster's own container runtime, or an existing provider with behavior added. You will learn the provider
contract, write a provider that wraps another one and logs every command it runs, select it in a config, and check
it on CPU with a Harbor task.

**Prerequisites.**

- Platoon installed (`pip install 'platoon[tinker,ors]'`, or `uv sync --extra tinker --extra ors` in a checkout).
- [Sandboxes and Harbor tasks](../how-to/sandboxes-and-harbor.md): how Harbor tasks use sandboxes, and the built-in
  providers `process`, `docker`, `singularity` and `enroot`.

## What a provider does

A provider starts sandboxes from a `SandboxSpec` and hands back `Sandbox` objects that run commands, move files,
snapshot their filesystem and stop. The Harbor environment service starts one sandbox per session (the
`provider` and `provider_options` of its `service_kwargs`), runs the task's setup and tools in it, places CLI
harnesses inside it, and grades its final state with the task's tests. `platoon/sandbox/base.py` defines the
interface:

```python
@dataclass
class SandboxSpec:
    image: str  # "docker://repo:tag", a .sif or .sqsh path, or "local"
    workdir: str = "/app"
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[Mount] = field(default_factory=list)  # Mount(source=host path, target=sandbox path, read_only=True)
    network: Literal["host", "none"] = "host"
    cpus: float | None = None
    memory_mb: int | None = None
    labels: dict[str, str] = field(default_factory=dict)  # the rollout id and task, for cleanup tools
    upstreams: dict[str, str] = field(default_factory=dict)  # host services to reach: {"gateway": "unix:///…/gw.sock"}


class Sandbox(Protocol):
    id: str
    routes: Mapping[str, str]  # upstream name -> the URL a process inside calls ({"gateway": "http://127.0.0.1:18080"})

    async def exec(self, cmd: str | list[str], *, timeout_s=None, env=None, cwd=None, stdin=None) -> ExecResult: ...
    async def upload(self, src: str | Path, dst: str) -> None: ...
    async def download(self, src: str, dst: str | Path) -> DownloadReport | None: ...
    async def snapshot(self) -> SandboxSnapshot: ...
    async def kill_processes(self) -> None: ...
    async def stop(self) -> None: ...


class SandboxProvider(Protocol):
    name: str
    # a subset of {"snapshot", "network_none", "gpu", "isolation", "routes", "host_loopback"}
    capabilities: frozenset[str]

    async def start(self, spec: SandboxSpec) -> Sandbox: ...
    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> Sandbox: ...
```

| Method | Must |
|---|---|
| `start(spec)` | Start a sandbox from `spec.image`, with `spec.workdir` as the default working directory, `spec.env`, the mounts and, where the runtime can, the network mode and resource limits. Raise if it cannot: the session's setup fails (`EnvSetupError`, failure kind `sandbox_start`), and the rollout is dropped, never scored |
| `exec(cmd, ...)` | Run `cmd` to completion: a string through `bash -c`, a list as an argv. `cwd` and paths are paths inside the sandbox. Return `ExecResult(exit_code, stdout, stderr, timed_out)`; on `timeout_s`, kill the command and set `timed_out`. A command whose working directory is gone or cannot be entered fails as the command's own failure, with exit status 126 as `docker exec` gives it (`CWD_ERROR_EXIT_CODE`), not as an error of the sandbox; to say why, put only `CWD_ERROR_PREFIX` + `"<directory>: <reason>"` on stderr, as hpc-sandbox does (`platoon.sandbox.cwd_error` reads it) |
| `upload(src, dst)` | Copy a host file to `dst`, or a host directory's contents into `dst` |
| `download(src, dst)` | Copy a sandbox file, or a directory's contents, to the host; raise `FileNotFoundError` when `src` does not exist. Return a `DownloadReport` of the symlinks it left out (`dropped_links`, `dropped_total`), or `None` if the provider does not say |
| `snapshot()` | Capture the filesystem (not the processes) as a `SandboxSnapshot`; only when the provider declares `snapshot`. The snapshot stays valid until the sandbox it came from stops |
| `restore(snapshot, spec)` | Start a new sandbox from a snapshot this provider made; refuse another provider's |
| `kill_processes()` | Kill every process the commands left running, ones that left their process group or session included, and keep the sandbox running with its files. Raise `RuntimeError` when one survives, or when the sandbox cannot tell them from the processes that keep it alive. Harbor grading calls it before the tests and after them, and its grade is an untrainable `error` (`verifier_isolation`) when it raises; a sandbox without it is graded with a warning, while what the agents left running runs on |
| `stop()` | Kill everything running in the sandbox and release it. Idempotent |
| `routes` | For each of `spec.upstreams` that a process in the sandbox cannot reach as it is, the URL that reaches it from inside, such as a loopback port forwarded to the model gateway's Unix socket in a sandbox without a network (`platoon/sandbox/routes.py` implements them for the built-in providers). Optional: platoon reads it with `getattr(sandbox, "routes", {})`, and an `ensure_routes()` method, when there is one, starts a route's forwarder again after `kill_processes()` ended it |
| `reap(labels)` | Optional, on the provider: stop the sandboxes whose labels include `labels`, also those a process that died without stopping them left behind, and return how many. A training worker's supervisor and `platoon eval` call it for what a killed worker left. The built-in providers put this process's labels (`platoon.sandbox.cleanup.labels()`: a run's `platoon.run`, a rollout worker's `platoon.worker`) on every sandbox beside `spec.labels`; do the same, and a run's label finds your sandboxes too |

Capabilities tell platoon what it may ask for, instead of checking provider names:

| Capability | Platoon relies on it for |
|---|---|
| `snapshot` | `copy` forks of a Harbor session: a subagent gets a restored snapshot of its parent's sandbox. Without it, a `copy` fork fails |
| `network_none` | The provider starts a sandbox without network when the spec asks for none, as Harbor tasks with `network_mode = "no-network"` do; a CLI harness placed inside it then reaches the model gateway only through a route, and platoon refuses to place one there without |
| `isolation` | The sandbox is isolated from the host: say so only when that is true |
| `gpu` | The sandbox can reach GPUs |
| `routes` | The provider implements `spec.upstreams` and `Sandbox.routes` |
| `host_loopback` | A sandbox on the host network shares this host's network namespace, so its processes reach the model gateway on this host's loopback as they are. A provider whose sandboxes run elsewhere (another node, a cloud) must not declare it: platoon then places a harness in one of its sandboxes only through a route to the gateway |

Two optional class attributes concern a site's profile, which can choose a provider for every config run on the
site ([Write the site's profile](../how-to/clusters.md#4-write-the-sites-profile-once)):

| Class attribute | Platoon relies on it for |
|---|---|
| `isolates = True` | The provider isolates its sandboxes from the host, so a site profile may name it. A profile refuses a provider whose class does not say so, and `platoon train` refuses one that says `isolates = False`, as `process` does, unless the run config itself names it |
| `unavailable(options)` | A class method that says why this host cannot run the provider with `options` (no runtime executable, a missing package), or returns `None`, without building it. `platoon train`, `eval`, `rollout` and `prepare` refuse a provider from the profile that the node cannot run, quoting the profile, instead of failing every rollout |

Every rollout builds its own environment service and so its own provider, with `provider_options` as keyword
arguments. Keep the constructor cheap, and keep long-lived state (a client, a pool of warm sandboxes) at module
level.

## Steps

### 1. Write the provider

This provider wraps any other provider, built by name, and appends one JSON line per command to a log file:

```python
# my_sandboxes/audited.py
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from platoon.sandbox import DownloadReport, ExecResult, Sandbox, SandboxSnapshot, SandboxSpec, get_provider


class AuditedSandbox:
    """A sandbox that logs every command it runs, then behaves exactly like the sandbox it wraps."""

    def __init__(self, inner: Sandbox, log: Path) -> None:
        self.inner, self.log, self.id = inner, log, inner.id

    async def exec(self, cmd: str | list[str], **options: Any) -> ExecResult:
        result = await self.inner.exec(cmd, **options)
        record = {"sandbox": self.id, "cmd": cmd, "exit_code": result.exit_code, "at": time.time()}
        with self.log.open("a") as file:
            file.write(json.dumps(record) + "\n")
        return result

    async def upload(self, src: str | Path, dst: str) -> None:
        await self.inner.upload(src, dst)

    async def download(self, src: str, dst: str | Path) -> DownloadReport | None:
        return await self.inner.download(src, dst)

    async def snapshot(self) -> SandboxSnapshot:
        return await self.inner.snapshot()

    async def kill_processes(self) -> None:
        await self.inner.kill_processes()

    async def stop(self) -> None:
        await self.inner.stop()

    @property
    def routes(self) -> Mapping[str, str]:
        return getattr(self.inner, "routes", {})

    async def ensure_routes(self) -> Mapping[str, str]:
        ensure = getattr(self.inner, "ensure_routes", None)
        return await ensure() if ensure is not None else self.routes


class AuditedProvider:
    """The provider `inner` (a built-in name or module:Class, built with `inner_options`), logging to `log_path`."""

    def __init__(
        self, *, inner: str = "process", inner_options: dict[str, Any] | None = None, log_path: str = "commands.jsonl"
    ) -> None:
        self.inner = get_provider(inner, **(inner_options or {}))
        self.name = f"audited-{self.inner.name}"
        self.capabilities = frozenset(self.inner.capabilities)  # no more than the inner provider can do
        self.log = Path(log_path)

    async def start(self, spec: SandboxSpec) -> AuditedSandbox:
        return AuditedSandbox(await self.inner.start(spec), self.log)

    async def restore(self, snapshot: SandboxSnapshot, spec: SandboxSpec) -> AuditedSandbox:
        return AuditedSandbox(await self.inner.restore(snapshot, spec), self.log)
```

A provider for a remote sandbox service has the same shape: `start` creates a sandbox through the service's API
and returns an object whose methods call it, `exec` runs a command remotely and returns its output, `upload` and
`download` transfer files, and `stop` deletes the sandbox. Declare `snapshot` only if the service can copy a
sandbox's filesystem into a new one. Its sandboxes do not share this host's network, so it must not declare
`host_loopback`; give it `routes` to the model gateway, or keep harnesses out of its sandboxes (`placement: local`).

### 2. Check it on a Harbor task

[`examples/harbor-local`](../../examples/harbor-local/) has a task whose tests check one file. Open a session the
way a rollout does, act as the agent would, and grade, from the repository root:

```python
# check_provider.py
import asyncio
import json

from platoon.envs.harbor import HarborEnvironmentService


async def main() -> None:
    service = HarborEnvironmentService(
        dataset_dir="examples/harbor-local/tasks",
        provider="my_sandboxes.audited:AuditedProvider",
        provider_options={"inner": "process", "log_path": "commands.jsonl"},
    )
    session = await service.open(await service.get_task("write-greeting"), rollout_id="check")
    try:
        await session.call_tool("bash", {"command": "echo 'hello from platoon' > greeting.txt"})
        outcome = await session.grade()
    finally:
        await session.close()
    print(outcome.status, outcome.reward)
    with open("commands.jsonl") as log:
        for line in log:
            cmd = json.loads(line)["cmd"]
            if isinstance(cmd, list):  # an argv; grading's steps are shell scripts, shown by the name they run under
                cmd = f"{cmd[0]} -c <{cmd[3]} script>" if cmd[:2] == ["/bin/sh", "-c"] else " ".join(cmd)
            print(cmd)


asyncio.run(main())
```

```bash
PYTHONPATH=. uv run python check_provider.py
```

The log shows every command the session ran: it set up the sandbox and ran the agent's command; to grade, it checked
that the workdir is still there, made `/tests` and `/logs/verifier` afresh, put the task's tests in place, ran them,
read the reward files and removed the tests again. Grading's own steps are shell scripts, which the check shows by
the name they run under, and it also called `kill_processes()` before and after the tests, which the wrapper passes
on without logging:

```text
scored 1.0
mkdir -p /logs/agent /logs/verifier /logs/artifacts /app
echo 'hello from platoon' > greeting.txt
/bin/sh -c <platoon-workdir script>
/bin/sh -c <platoon-reset script>
bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1
/bin/sh -c <platoon-collect script>
rm -rf /tests
```

The `process` provider runs everything on your machine with no isolation, which is why the check needs no
container runtime; wrap `docker` or `singularity` to audit real sandboxes.

### 3. Select it in a config

`provider` takes a built-in name or a `module:Class` path, and `provider_options` are its constructor's keyword
arguments:

```yaml
environments:
  - name: harbor-local
    service: harbor
    service_kwargs:
      dataset_dir: ./tasks
      provider: my_sandboxes.audited:AuditedProvider
      provider_options:
        inner: docker
        log_path: /tmp/sandbox-commands.jsonl
    harness: platoon-bash
```

The module must be importable in every process that runs rollouts: install your package, or put its directory on
`PYTHONPATH`.

## Test it like the built-ins

Before you train with a new provider, check what the Harbor service and harnesses rely on, with a small task that
has a reference solution:

- A command that writes a file in `workdir`, then a `download` of it, round-trips the content; a string command
  sees shell syntax (pipes, `&&`); a list command does not.
- A command that sleeps past `timeout_s` comes back with `timed_out=True`, and nothing it started keeps running.
- `stop()` twice is fine, and after it the sandbox's processes are gone.
- `kill_processes()` ends a process a command left running in the background, also one that moved to a session of
  its own (`setsid`), and the sandbox runs commands and keeps its files afterwards.
- With `snapshot`: a file written before `snapshot()` is in a sandbox `restore`d from it, and a file written in that
  restored sandbox is not in the original.
- With `network_none`: a sandbox started with `network="none"` cannot open a connection.
- The task's reference solution scores 1 through `session.grade()`, and doing nothing scores 0.

## Next steps

- [Sandboxes and Harbor tasks](../how-to/sandboxes-and-harbor.md): the built-in providers and their options.
- [Write an environment service](environment-services.md): a service of your own whose sessions use a provider.
- [Extending platoon](index.md): every extension point and how platoon finds it.
