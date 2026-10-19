# RFC 0004: sandweave integration

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): sandbox providers, gateway reachability and fork modes |

## Summary

Add sandweave as one more sandbox provider behind platoon's sandbox interface, chosen per site by capability, never as a hard dependency. sandweave runs patched gVisor inside unprivileged Apptainer and adds what native providers lack: egress isolation without root, hard memory limits without cgroup delegation, verified filesystem and memory snapshots, warm pools, NVIDIA GPU sandboxes through nvproxy, and a multi-host scheduler. One blocker comes first: a gVisor guest cannot reach any address of its own host, so a harness inside sandweave cannot call the gateway. The fix to ask for is host Unix-socket mounts; until then a gateway on another host works for runner sandboxes. A one-day bake-off on a representative HPC site decides adoption. Snapshots then back `copy` forks, snapshot-then-verify and, later, rollouts that survive walltime.

## Motivation

Platoon ships native providers (`process`, `docker`, and `singularity` and `enroot` through `hpc-sandbox`). On sites with a setuid SingularityCE install, `--network none` is reserved for root, so native sandboxes there share the host network, and model-written code can reach every port on the host: the Tinker API, engines, env servers and databases. Only agent keys protect the gateway. Native providers also give only filesystem snapshots by copy, no memory snapshots, no GPU isolation and no warm pools.

sandweave closes those gaps, unprivileged, on the el9 5.14 kernel line common on HPC sites. It is also young (version 0.2.41, qualified to 64 tasks per pool where large runs need 400-1,000 sandboxes), cannot run OCI images in its native runtime, has no AMD GPU path and cannot reuse SIF caches. So it should be an option a site profile can prefer, with native providers as a permanent fallback.

## Proposal

### 1. Capabilities and routes

Today's capabilities (`snapshot`, `network_none`, `gpu`, `isolation`) grow into a vocabulary that providers declare and callers require:

```python
class Cap(StrEnum):
    FS_SNAPSHOT = "fs_snapshot"
    MEM_SNAPSHOT = "memory_snapshot"
    CHECKPOINT_RESUME = "checkpoint_resume"
    PAUSE = "pause_resume"
    NET_NONE = "network_none"
    NET_ALLOWLIST = "network_allowlist"
    HOST_ISOLATED = "host_isolated"  # model code cannot reach host or private addresses
    GPU_NVIDIA = "gpu_nvidia"
    GPU_AMD = "gpu_amd"
    POOLS = "pools"
    OCI_IMAGES = "oci_images"
    EXEC_STREAM = "exec_streaming"


class Sandbox(Protocol):  # today's methods, plus routes
    id: str
    routes: Mapping[str, str]  # {"gateway": "http://127.0.0.1:8742"}: what a harness should call

    async def exec(
        self,
        cmd: str | list[str],
        *,
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        stdin: bytes | None = None,
    ) -> ExecResult: ...
    async def upload(self, src: str | Path, dst: str) -> None: ...
    async def download(self, src: str, dst: str | Path) -> None: ...
    async def snapshot(self) -> "SandboxSnapshot": ...
    async def stop(self) -> None: ...


@runtime_checkable
class SupportsMemorySnapshot(Protocol):
    async def snapshot_memory(self) -> "SandboxSnapshot": ...


@runtime_checkable
class SupportsCheckpointResume(Protocol):
    async def checkpoint(self) -> "SandboxSnapshot": ...  # stop and keep the state
```

`Sandbox.routes` hides how a harness reaches the gateway. Harness specs use the route's URL and never change per provider. The resolver walks a site's provider order, pairs each provider with its configured route, computes the pair's effective capabilities, and takes the first pair that covers what the env, the harness and verification need, or fails at config time.

### 2. The adapter

```python
class SandweaveProvider:
    name = "sandweave"

    def __init__(
        self,
        *,
        mode: Literal["local", "cluster"] = "local",
        route: Literal["uds", "cidr"] = "cidr",
        gateway_socket_dir: str | None = None,
        gateway_cidr: str | None = None,
        network: Literal["offline", "allowlist", "internet"] = "offline",
    ) -> None: ...

    @property
    def capabilities(self) -> frozenset[str]: ...  # route-dependent: "cidr" removes NET_NONE

    async def start(self, spec: SandboxSpec) -> Sandbox: ...  # sandweave Sandbox.create.aio(...)
    async def restore(self, snapshot: "SandboxSnapshot", spec: SandboxSpec) -> Sandbox: ...
```

- **Imports.** `sandweave` is imported lazily, behind an extra with an exact version pin; core never imports it.
- **Mapping.** `SandboxSpec.image` maps to a digest-pinned `docker://` reference or an imported OCI archive; `cpus` and `memory_mb` to sandweave's CPU quota and guest memory; `mounts` to read-only mounts (toolchains); `labels` to the sandbox name until sandweave offers labels.
- **Placement.** `local` mode keeps one sandweave worker per cell host, inside the host's env CPU slice, with `connection="direct"` on loopback, so no controller sits on the critical path. `cluster` mode (Weave) is for remote or elastic capacity ([RFC 0002](0002-multi-cluster-placement.md)).
- **Errors.** `ResourceUnavailable` and launch failures become infra errors, which platoon's outcome rule keeps out of training; an operation sandweave reports as unknown is never replayed blindly.
- **Output.** sandweave caps command output at 64 MiB and spools it to files, so harness logs go to files, never to an exec stream.

### 3. The host-reachability ask

sandweave's default policy drops every packet to the host's own addresses, loopback and private ranges, so a guest cannot open a connection to a gateway on its host.

| Route | Mechanism | Isolation kept | Status |
|---|---|---|---|
| `uds` | mount a host socket directory into the guest; an in-guest forwarder listens on `127.0.0.1` and relays to the gateway's socket | `NET_NONE` and `HOST_ISOLATED` | needs host socket mounts in sandweave's public SDK; its engine already opens host sockets for CUDA MPS |
| `cidr` | the gateway runs on another host of the site; the sandbox allows that host by CIDR | loses `NET_NONE`: egress is allowed to the whole range | works today; interim |
| `relay` | an exec-stream relay over the provider's stdin and stdout | the provider's | a poor fit on sandweave (spooled output, output cap); meant for hosted providers |

**Asks to the sandweave maintainers, in priority order:** (1) host Unix-socket mounts in the public SDK, allowed in `offline` mode; (2) sandbox labels with label-scoped listing and termination, for exact janitors; (3) a PBS-friendly worker: the hostname in the agent stamp, a foreground join, explicit memory and CPU budgets where no cgroup exists; (4) scale qualification at 128-256 sandboxes per worker; (5) OCI images in the native runtime; (6) registry authentication and mirrors; (7) a Harbor pin aligned with platoon's; (8) controller restart and backup guidance on shared storage; (9) a license, for third parties who install platoon's sandweave extra.

### 4. The bake-off

One 6-hour batch job on two GPU nodes, with a fresh sandweave home and an existing SIF cache opened read-only:

| Step | Pass |
|---|---|
| facts: user namespaces, FUSE, cgroups, driver, local disk | recorded in the site's state directory |
| install and `doctor` | green; first setup within 5 minutes |
| latency and density against native Singularity at 1, 64, 128 and 256 sandboxes | p90 cold start ≤ 3 s at 128; exec p50 ≤ 30 ms; zero leftovers |
| isolation canaries: host addresses, loopback, engine, Tinker and database ports, a public site | `offline` blocks all; an allowlist reaches only what it lists |
| gateway routes `cidr` (and `uds` if a build has it) | 1,000 calls with 200 KB bodies; added p50 ≤ 5 ms; SSE keepalives arrive |
| Weave on PBS: worker kill, controller restart | no stuck reservations; no lost lease |
| parity: 20 tasks each of SWE-rebench, Frontier-CS and Harbor, oracle solutions | identical outcomes; wall time ≤ 1.5x native |
| GPU: CUDA under nvproxy against `singularity --nv` | throughput within 5% |

Adopt sandweave for a workload family only if its parity step passes; otherwise that family keeps its native route through capability matching, and runner sandboxes may still use sandweave.

### 5. GPU sandboxes

NVIDIA GPU sandboxes run under nvproxy with one whole GPU each (UNVERIFIED on each site's driver; the bake-off decides). The fallback is native `singularity --nv`. AMD clusters use native `apptainer --rocm`, which sandweave does not offer. CUDA live snapshots are experimental and tied to one GPU's UUID and driver, so they stay out of training paths.

### 6. Checkpoint and resume

| Capability | sandweave evidence | What it enables |
|---|---|---|
| filesystem snapshot and restore | verified; restore plus first command p50 1.07 s | `copy` forks; `snapshot_async` verification ([RFC 0001](0001-native-multi-agent.md)); isolated child workspaces |
| memory snapshot (CPU) | verified; about 2 s for a desktop restore | a verifier sees processes the child left running; mid-episode forks for tree rollouts and counterfactual credit |
| checkpoint, then resume elsewhere | `stop()` returns a snapshot; restore on a compatible worker | drain by migration; rollouts that survive a job's walltime |

Resume needs the harness's model connection to survive restore: a per-request UDS forwarder and addressing agents by fingerprint rather than by gateway address keep that possible. Snapshots and images are not evicted automatically, so the adapter deletes what it creates.

## Alternatives

- **Native providers only.** No egress isolation without root, no memory snapshots, no warm pools. It stays the fallback.
- **sandweave through Harbor's environment interface.** Harbor's base API has no snapshots, pause, pools or streaming exec, and sandweave pins an older Harbor than platoon.
- **Hosted sandboxes only** (Modal, Daytona, Sail, E2B). Strong isolation and snapshots, but roughly $4-15k per week at 384 concurrent sandboxes (EST) and a relay to reach an HPC gateway.
- **gVisor directly.** Needs setup sandweave already packages unprivileged.

## Rollout plan

| Milestone | Scope | Exit check |
|---|---|---|
| M0 adapter | `SandweaveProvider` behind an extra; capability and route model; contract tests on a fake | the provider contract suite passes on CPU |
| M1 bake-off | the §4 job on a representative HPC site | a written verdict per workload family |
| M2 runner sandboxes, `cidr` | sandboxed harnesses with gateway shards on another host | a GPU canary with sandboxed harnesses, no host ports reachable |
| M3 `uds` route | after ask 1 lands: `NET_NONE` with the socket route | isolation canaries pass; sandweave becomes the site default for runner sandboxes |
| M4 task sandboxes | Harbor and ORS task families that passed parity | outcomes identical to native on 20 oracle tasks per family |
| M5 snapshots in training | `copy` forks and `snapshot_async` verification on sandweave snapshots | an A/B of `sync` against `snapshot_async` |
| M6 GPU and resume | nvproxy sandboxes; checkpoint and resume across a walltime boundary | a rollout resumes in a new job and trains |

## Open questions

1. Until the socket route lands, is it acceptable to run gateway shards on the head host and allow it by CIDR?
2. Should runner sandboxes default to no egress, with prebuilt toolchains, or to public egress with host and private ranges still blocked?
3. Should SWE task sandboxes stay on native Singularity with host networking until their parity passes, or move first?
4. Which Slurm sites, NVIDIA and AMD, should a second bake-off cover?
5. Is a small public HTTPS VM acceptable for a multi-site Weave controller?
