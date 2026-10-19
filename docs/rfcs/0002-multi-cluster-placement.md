# RFC 0002: Multi-cluster placement and elasticity

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-13 |
| Builds on | [Architecture](../architecture.md): one training step, process-isolated rollout workers and the embedded gateway |

## Summary

Let one run spread over several clusters and hosted services, and grow or shrink while it trains. The unit that moves is the **cell**: one host's rollout workers with their gateway shard, env sessions and sandboxes. A rollout tree never leaves its cell. Every member **dials out** to a **hub**, a soft-state registry with leases, incarnations and mailboxes, so no site has to accept inbound connections and the orchestrator still opens no port. Placement is data: site profiles describe each place, deployment files bind pools of cells, env services and samplers to sites, and launchers (PBS, Slurm, local; SkyPilot optionally) start them. Members join, catch up, serve and drain without killing work in flight. Sampler pools can add engines near remote cells under version constraints, except where MoE routing replay forbids it.

## Motivation

Today a run is one batch job: the trainer and its rollout workers on one host, env servers and the Tinker service on other hosts of the same allocation. That shape stops scaling in three ways:

- **Capacity is fixed at submit time.** A run cannot add CPU hosts for env servers and sandboxes when rollouts are the bottleneck, or release them when the trainer is.
- **One site only.** Clusters differ: some run PBS with setuid Singularity and refuse inbound connections; others run Slurm with enroot or Apptainer; some env services and sandboxes are hosted. Nothing today lets a second site contribute rollouts.
- **Walltime ends everything.** A batch job's walltime or a preemption kills every rollout in flight; elastic members would drain and hand over instead.

Agentic RL suits cross-site placement because it moves little data and tolerates latency. At 384 concurrent rollouts, control traffic is 3-5 messages per second, sampling that crosses sites is 15-60 Mbit/s, and about 100 extra round trips of 150-600 ms add 1-3% to a 30-minute episode (EST). The hard parts are network policy, operations and RL correctness when samplers differ.

## Proposal

### 1. Cells

A **cell agent** runs on every rollout host. It starts rollout workers (today's worker stack, unchanged), one gateway shard shared by its workers instead of one server per worker, env servers it is asked to serve, and sandboxes through the site's provider. A rollout tree with its children, verifiers and restored sessions stays in one cell, so harness traffic, tool calls, snapshots and R3 payloads never cross sites. Inline harness loops run in the cell, never in the orchestrator, for the same reason.

### 2. The hub

Members (cells, env services, sandbox pools, samplers and the orchestrator itself) dial out to one reachable hub. Its registry is soft state: TTL leases, an incarnation per process start, an epoch per hub start, so a restarted hub rebuilds from re-joins. Orchestrator-to-cell commands keep typed stubs but travel through mailboxes that members long-poll.

```python
MemberKind = Literal["orchestrator", "cell", "env_service", "sandbox_pool", "sampler"]
MemberState = Literal["joining", "ready", "draining", "left", "lost"]


class MemberSpec(BaseModel, frozen=True):
    run_id: str
    member_id: str  # "pbs-a/cell-07", "slurm-a/cell-3"
    incarnation: str  # uuid per process start: the fencing token
    kind: MemberKind
    site: str
    release: str  # release digest; must equal the run's
    caps: frozenset[str] = frozenset()  # "snapshot", "network_none", "env:toolathlon", "gpus_free"
    capacity: Mapping[str, int] = {}  # {"rollouts": 96, "sandboxes": 128}
    endpoints: tuple["Endpoint", ...] = ()  # URLs with reachable_from (site ids; empty = dial-out only)


class LeaseRef(BaseModel, frozen=True):
    lease_id: str
    member_id: str
    incarnation: str
    orch_epoch: int


class Envelope(BaseModel):
    seq: int
    msg_id: str  # idempotency key
    sender: str
    to: str
    kind: Literal["cmd", "result", "event"]
    op: str  # "env.open", "agent.open", "harness.launch", "agents.spawn", ...
    lease: LeaseRef | None = None
    body: dict[str, Any]  # at most 1 MB; larger payloads travel as blob references


class Hub(Protocol):
    async def join(self, spec: MemberSpec, token: str) -> "JoinReply": ...
    async def beat(self, heartbeat: "Heartbeat") -> "HeartbeatReply": ...  # replies carry drain, activate_policy, ...
    async def watch(self, since_rev: int) -> "MemberEvents": ...
    async def post(self, to: str, messages: Sequence[Envelope]) -> None: ...
    async def mail(self, me: str, after: int, wait_s: float = 50) -> list[Envelope]: ...
```

Every command and result carries a `LeaseRef`; cells refuse commands for another incarnation and the orchestrator refuses results from a stale one. Defaults: 5 s heartbeats and a 30 s TTL inside a site; 10 s, 60 s and a 300 s partition grace across sites, during which a silent member's leases freeze instead of failing. For a single site the hub runs inside the head agent; across sites it runs on a small VM. JSON on the wire, never pickle; TLS on every link that leaves a site; every client honors `HTTPS_PROXY`.

### 3. Placement as data

Three things are standardized and everything else is swappable: **provider protocols** (launcher, container runtime, sandbox provider, blob store, exposure), **site profiles** with facts about one place, and **deployment files** that bind pools to sites. Experiments never name sites, and no code path tests for a site or scheduler name. [RFC 0008](0008-config-and-cli.md) defines the file kinds.

```python
PoolKind = Literal["orchestrator", "cell", "env_service", "sampler", "backend"]


class PoolSpec(BaseModel):
    kind: PoolKind
    site: str | None = None  # default: pool_defaults.site
    launcher: str | None = None  # pbs | slurm | local | ssh | skypilot
    count: "int | Count" = 1  # Count(min, max): elastic
    requires: frozenset[str] = frozenset()  # member facts: gpus_free, env:toolathlon, ...
    serves: tuple[str, ...] = ()  # env services this pool hosts
    policy: str | None = None  # sampler pools: the policy served (RFC 0003)


class Launcher(Protocol):
    def render(self, plan: "RolePlan", site: "SiteProfile") -> "LaunchArtifact": ...  # shown by --dry-run
    async def submit(self, artifact: "LaunchArtifact") -> "LaunchHandle": ...
    async def cancel(self, handle: "LaunchHandle") -> None: ...
    async def scale(self, pool: str, target: int) -> list["LaunchHandle"]: ...  # drains before cancel
    def join_command(self, plan: "RolePlan") -> str: ...  # for manual scale-up
```

A fan-out starts one cell agent per host: `pbsdsh` on PBS (no PBS cgroup, so budgets are explicit), `srun --overlap` on Slurm, SSH, or a pod. Inside a running PBS or Slurm job, `platoon train` fans out over that job's hosts instead of submitting.

### 4. Elastic join and leave

Members move `joining → ready → draining → left`, or to `lost` after TTL plus grace.

| Kind | Ready when | Drain | Lost |
|---|---|---|---|
| cell | release digest matches; env servers healthy (Frontier-CS judge pools take up to 20 minutes); gateway and sandbox probes pass | no new trees; trees in flight finish inside their reservation; at the deadline the rest end `cancelled/drain` | its trees end `error/member_lost` |
| env service | health, capacity and declared capabilities | no new sessions | sessions end `error/env_restart` |
| sampler | tokenizer digest matches; a probe sample returns finite logprobs; it serves a version inside the staleness window | calls in flight finish; agents re-pin on their next call | calls that delivered no tokens move to another sampler |

`cancelled/drain`, `error/member_lost` and the other exclusions never train as reward 0, which is platoon's outcome rule. Walltime guards, Slurm signals, scale-down decisions, operators and the memory guard all trigger drains. **Release identity** is a digest over the lock file, source revisions and image digests, never over built environments, so each site and CPU architecture builds its own release with the same digest.

### 5. Sampler pools

Two ways add inference capacity near remote cells:

| | Backend-managed engines | Platoon-managed engines |
|---|---|---|
| Run by | the Tinker backend (for example miles with external engines) | platoon's engine agent, launched as a `sampler` pool |
| Platoon sees | one Tinker endpoint | sampler members in the hub |
| MoE routing replay (R3) | preserved | impossible through the Tinker API: rows are `routing_uncovered` |
| Weights | the backend's job | exported by the backend, pulled from a `PolicyStore` |

```python
@dataclass(frozen=True)
class VersionConstraint:  # per sample request, after miles' disaggregated-rollout contract
    min_version: int | None = None  # default: learner version minus max_staleness
    exact_version: int | None = None  # evals and pinned verifiers


class PolicyStore(Protocol):  # immutable manifests: full, delta or LoRA adapter
    async def publish(self, ref: "PolicyRef", artifact: "PolicyArtifact") -> "PolicyManifest": ...
    async def head(self, policy_id: str) -> int: ...  # advances only after files are durable
    async def manifest(self, policy_id: str, version: int) -> "PolicyManifest": ...
```

Samplers advertise a version only after activating it, and every call records its sampler and its start and end versions. LoRA adapters hot-load as versions; full fine-tunes ship sparse deltas with a full snapshot every K versions. MoE runs that require R3 refuse platoon-managed engines at config time.

### 6. Connectivity

Members dial out to the hub; bulk data (weights, records, captures) moves by reference through a blob store (`file://`, `s3://`, `hub://`); synchronous paths stay inside a site. An **exposure** provider carries the few synchronous paths that must enter a site without ingress, such as remote cells sampling from a self-hosted Tinker service (for example [miles-tinker](https://github.com/neulab/miles-tinker)) on a cluster that refuses inbound connections. frp on the hub VM is the default because it works through HTTP proxies; every exposed route needs an edge token, because some Tinker backends accept any API key.

### 7. Autoscaling

The controller adopts AstraFlow's dead-band rule on the trainer's waiting fraction `w`: above 0.10, grow to `ceil(G / (1 - w))`; below 0.05, shrink toward consumed over produced, padded by 10%. Backpressure attributes the signal to a pool: long router waits mean more cells, long sampler queues mean more engines. HPC queues add minutes to hours, so the first version of the controller runs in **advise** mode: it prints ready-to-run commands for a person or an agent. `act` follows a churn soak.

### 8. SkyPilot as an optional launcher

SkyPilot covers about 20 clouds, Kubernetes, SSH pools and Slurm through a login node, with spot recovery. It has no PBS support, no Singularity or Apptainer images, no port opening on Slurm and no cross-cluster discovery, and its pins (`uvicorn<0.36`, `click<8.2`) force a separate environment. Platoon therefore keeps thin `pbs`, `slurm`, `local` and `ssh` launchers and offers `skypilot` as one more launcher, run as a separate tool, for cloud bursts, the hub VM and Slurm sites that already use it.

### 9. Lessons from AstraFlow

| Borrow | Platoon form |
|---|---|
| rollout execution next to the engine; only tasks, trajectories and weights cross | cells are the unit of placement |
| every token tagged with its weight version | `version_start` and `version_end` per call, plus the sampler stamp |
| suspect on data-path errors, confirm out of band | router suspicion plus hub TTLs |
| pull-triggered sparse bf16 deltas with periodic full resync | `PolicyStore` manifests |
| dead-band scaling on trainer wait; an agent as the maintainer | the advise-mode controller |

| Avoid | Platoon instead |
|---|---|
| cloudpickle over unauthenticated HTTP | versioned JSON, join tokens, member tokens, TLS |
| address discovery by hostname or a UDP probe of 8.8.8.8 | explicit endpoint URLs with `reachable_from` |
| trainer-initiated TCP and ZMQ into rollout hosts | consumers pull over HTTPS |
| admitting an engine before it caught up | ready only after catch-up |
| a cross-region result emulated with tc/netem only | probes on real sites before claims |

## Alternatives

- **One flat network** (AstraFlow, Ray across clusters). Needs inbound ports everywhere, which many HPC sites forbid.
- **An overlay VPN** (Tailscale or Headscale). Useful during development; needs site approval, and userspace mode routes all traffic through a local proxy.
- **SkyPilot as the portability layer.** No PBS and no Singularity, and its pins conflict with the gateway stack.
- **A file registry on shared storage.** Lustre is not shared across sites, and file times cannot fence a stale writer.
- **Kubernetes everywhere.** HPC sites do not offer it.

## Rollout plan

| Stage | Scope | Exit check |
|---|---|---|
| S0 protocol | cell agent over today's workers; hub embedded in the head agent; leases, incarnations, mailboxes; location-free addresses; `pbs`, `slurm` and `local` launchers; fault injection | CPU partition and restart tests on a fake clock; a dry run on a PBS site reproduces the host roles of a single-job run |
| S1 elastic within one site | satellite batch jobs add cells to a running run; drain at walltime | killing a satellite yields only excluded rows; join time measured |
| S2 cross-site cells | hub VM, frp exposure, cells on a Slurm cluster or in hosted containers | tc/netem emulation, then a real second site, under 5% slower per episode |
| S3 managed engines | `PolicyStore`, engine agent, version-constrained routing; LoRA first, then full fine-tune deltas | logprob mismatch per sampler kind within a preset bound; no R3 rows from engines |
| S4 autoscaler | `advise`, then `act` | fewer GPU-hours than a fixed pool at equal reward over days |
| S5 services per policy | policy-specific samplers on several sites ([RFC 0003](0003-multi-policy-training.md)) | a two-policy run trains both |

Calendar items no implementer can compress: an account on a second site, each site's policy on outbound connectors that expose compute-node services, a VM with DNS, TLS and an object-store budget, real WAN measurements, and a multi-day churn soak.

## Open questions

1. Do sites allow a long-lived outbound connector (frp, cloudflared) that exposes a compute-node service? This gates S2.
2. Can a node in one PBS job reach a port on a node in another job (S1)? Which outbound ports and proxies do compute nodes allow?
3. Should env services run inside cells or as separate pools by default? Frontier-CS judge pools take minutes to warm.
4. How large is the logprob mismatch between the Tinker service's sampler and a standalone engine of the same version and dtype?
5. Who owns the hub VM, its TLS certificate and its object-store bucket?
6. Does the straggler timer bias training toward short answers when one site's samplers are slower, and is one site per group enough to prevent it?
