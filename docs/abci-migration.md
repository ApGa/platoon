# ABCI PBS + Singularity migration

This file records the cluster-local facts and validation state for the OpenReward
migration. It intentionally contains no credentials.

## Legacy v7 migration inputs

The table below records the inputs used for the original v7 migration. The v8
release pins and refresh evidence are recorded separately in the v8 section
and `pbs-scripts/openreward-v8-release.json`.

| Component | Revision |
|---|---|
| Platoon | `d8c07cdbba423a06335e0fe469c03aa969045f36` |
| OpenHands Agent SDK | `d2677bf750ab202efc2808a2b05d4dcd8efa98b5` |
| SWE-rebench-V2 OpenReward | `538b5c5c9c529d3c78d5ef7fd3df7e54e7902ad7` (local `abci/singularity-backend`) |
| AReaL | `d99124ec15102ca2fcd4960cc8beaef3950c2672` (lockfile) |
| Starting model | `apurvaga/Qwen3.6-35B-A3B-preserve-thinking` at `ce815485d640a33461c25e3d5c505b5346e8c857` |
| SWE dataset | `PrimeIntellect/SWE-rebench-V2-Filtered-Verified` |

The migration checkout is `/home/aci18912it/rasp/source/platoon`. Large shared
state is rooted at `/groups/gai51740/aci18912it/rasp`.

## Confirmed login-node facts

- PBS Pro `2022.1.6.20240517133149`.
- Project `gai51740`; reserved execution queue `R9920261300`; resource type
  `rt_HF` is mandatory through `-v RTYPE=rt_HF`.
- `rt_HF` is one exclusive H200 node: 192 logical CPUs, eight H200 SXM GPUs,
  1,920 GB allocatable RAM, and approximately 14 TB local NVMe under `/local`.
- Queue range is 1--302 nodes. It defaults to one hour. The reservation queue
  exposes no walltime maximum; a job is killed at the reservation end.
- `/home` is Lustre with a 2 TiB user quota. `/groups/gai51740` is the shared
  project Lustre allocation and is used for model, dataset, image, checkpoint,
  and content-addressed caches. Per-session churn belongs in `$PBS_LOCALDIR`.
- The site runtime is Singularity, not a binary named `apptainer`.
  `singularity-ce/4.4.1` supports SIF/SQSH, instances, fakeroot, user binds,
  overlays, and `--cleanenv`. Its sessiondir limit is 64 MiB, so SWE sessions
  use node-local ext3 overlays rather than `--writable-tmpfs` alone.
- CUDA module: `cuda/12.9/12.9.1`. MPI module: `hpcx/2.20` (Open MPI 4.1.7a1,
  UCX 1.17, PBS TM support). HPC-X 2.20 is chosen over the CUDA-13-oriented
  2.26 module.
- PBS supplies `$PBS_NODEFILE` and `$PBS_LOCALDIR`. The ordered physical-node
  list must be obtained by order-preserving de-duplication of `$PBS_NODEFILE`.
- `pbsdsh` exists but only selects a vnode index; the PBS scheduler adapter must
  pass explicit worker and local ranks. The two-node probe confirmed Open MPI's
  `$OMPI_COMM_WORLD_RANK` and `$OMPI_COMM_WORLD_LOCAL_RANK` mappings for all 16
  ranks. The AReaL adapter uses explicit worker indexes instead of depending on
  these MPI variables.

## Actual PBS syntax

```bash
#PBS -P gai51740
#PBS -q R9920261300
#PBS -v RTYPE=rt_HF
#PBS -l select=2:mpiprocs=8
#PBS -l place=scatter
#PBS -l walltime=00:20:00
```

ABCI's resource-type hook supplies CPUs, GPUs, and memory; do not add generic
`ngpus` or `mem` chunks. Multi-node jobs require full-node `rt_HF`.

## Validation ledger

| Gate | State | Evidence |
|---|---|---|
| Phase-0 two-node probe | passed | Job `2130336.pbs1`: 8 H200s/node, MPI ranks, TCP 8082/8084, Lustre flock/atomic publish, outbound HTTPS, and OCI pull passed. Jobs `2130467`, `2130494`, and `2130507` characterized overlay modes. |
| Training SIF and two-node NCCL | passed | Job `2130658.pbs1` built and tested the digest-pinned CUDA 12.9 SIF. Job `2130754.pbs1` (`Exit_status=0`, 52 s; `hnode010` + `hnode020`) validated that exact SIF SHA-256 `53c770558ad86e4a8ec4d2532cd3ba1c60a10bd75fff6a51049dfb411b002b12`: 16/16 UUID-isolated H200 ranks saw exactly one GPU, CUDA SM90 kernels passed, PyTorch NCCL reduced across both nodes, NCCL selected Mlx5 `NET/IB` rather than socket fallback, and both nodes had clean teardown. |
| Qwen3.6 SGLang inference | passed | Job `2130939.pbs1` (`Exit_status=0`, 3m25s) ran the exact training SIF, exact 26-shard model snapshot, and all eight H200s as one offline TP8 SGLang server. A forced `add` call was parsed with the Qwen3-Coder tool parser, returned `finish_reason=tool_calls`, and preserved the expected tool name and arguments. |
| Model cache | passed | 38 files, 26/26 shards, 71,903,776,776 shard bytes at snapshot `ce815485...` |
| SWE dataset/index | passed | 6,272 total and 6,272 valid tasks |
| Immutable training runtime | passed; final-source replay pending | Job `2132611.pbs1` (`Exit_status=0`, 9m41s; `hnode010`) built the exact worker environment from a cold immutable cache, atomically promoted it, and proved an immediate exact cache hit. It separately built and replayed a host-ABI controller environment. Inside the exact training SIF, the real FSDP and Megatron actors, Transformer Engine, Apex, Megatron Bridge, and CUDA all imported on an H200. On the bare host, the controller entry point and PBS scheduler imported while a fail-closed assertion found no Megatron, Transformer Engine, Apex, or fused-extension modules loaded. Result keys: source `55b14a556f33...`, worker `4b6252ab33ca...`, controller `0f6af0d9c58f...`; SIF SHA-256 `53c770558ad86e4a8ec4d2532cd3ba1c60a10bd75fff6a51049dfb411b002b12`. The one-update receipt hook changes the content-addressed source, so this gate will be replayed once on that final source before Gate 5. |
| Toolathlon service | passed | Job `2130668.pbs1` (`Exit_status=0`, 24 s) ran two concurrent isolated sessions against the normal-user PostgreSQL/nginx service, observed terminal rewards `1.0` and `0.0`, and shut down cleanly. Extended job `2130903.pbs1` (`Exit_status=0`, 30m31s) killed worker 0 by its exact private PID, observed its replacement, preserved an unrelated live worker-1 session, repeated concurrent terminal rewards `1.0`/`0.0`, and then passed 1,802 seconds of continuous public/backend health checks. |
| SWE Singularity sandbox | passed | 39 focused backend/lifecycle/transport/outer-cleanup tests pass. Job `2132587.pbs1` (`Exit_status=0`, 1m32s, `hnode010`) lazily pulled a real task image, published the locked service adapter, ran two concurrent isolated ext3 sessions, edited the same tracked file that is UID 0/mode 0644 in the immutable lower SIF as UID 18912 in the upper layer, preserved foreground/background changes, restarted exact instances after timeout and cancellation, left zero sessions/processes, then proved strict parsed base reward `0` (one observed F2P failure) and gold reward `1` (zero wrong F2P/P2P). Raw SHA-256 `a4986e881222b07432100e4f195f526b7d9328e1a00975b62f24389bd216f78e`; adapted SHA-256 `155f65c47e5ea03dc7f8bcb97cb92589903b04ab8aa6ad85918de2ea7f253e70`; both unchanged after the gate. |
| PBS preallocated scheduler | passed | Job `2132603.pbs1` (`Exit_status=0`, 3m13s; `hnode010` + `hnode081`) exercised the real scheduler/launcher/Singularity supervisor path with one 8-rank actor role and one full-node rollout role. Normal teardown, delete/recreate while rollout stayed healthy, a competing controller constructor, an injected rank-3 failure, and controller `SIGKILL` all left zero worker processes, ENTRY/control records, or role directories. The competing constructor waited on the stable external lock, stale generation cleanup was rejected, rank 0 was stopped last, and every rank proved its exact H200 mapping and SIF digest. Focused PBS lifecycle tests pass 52/52; the combined PBS/Slurm scheduler set passes 72/72. Private result: `/groups/gai51740/aci18912it/rasp/job-state/pbs-scheduler-gate4/2132603/RESULT.json`. |
| Tiny optimizer update | pending | requires preceding gates |
| Fresh 16-node update | pending | requires preceding gates |
| 32-node recursive update | blocked by design | intentionally waits for 16-node parity |

## Post-parity H200 optimization experiments

The first fresh 16-node optimizer update intentionally retains the inherited
H100 allocation and concurrency settings. This provides a clean migration
parity result before performance tuning changes the numerical or rollout
schedule.

After that gate passes, test H200-specific changes as separate trial lineages:

1. Remove attention context parallelism (`c2` to `c1`) while retaining the
   actor's total GPU allocation. Rebalance another parallel axis rather than
   silently giving those GPUs to a different role; validate 40k-context peak
   memory, tokens/s, optimizer numerics, and checkpoint reload before adopting
   the new topology.
2. Sweep `rollout.max_concurrent_rollouts` independently of topology. The
   parity values are 8 for the 16-node job and 12 for the 32-node job; begin
   with bounded 12/16 (16-node) and 16/24 (32-node) variants. Compare accepted
   groups/hour, SGLang queue delay, KV-cache occupancy, prefix-cache hit rate,
   environment latency, and OOM/retry rate.
3. Change one dimension at a time. Do not combine the first CP-free run with a
   concurrency increase, so a memory or throughput regression remains
   attributable and the parity checkpoint stays a reproducible control.

## Known source-manifest gaps

The pinned Platoon commit calls `slurm-scripts/install_te.sh` and
`slurm-scripts/install_apex.sh`, but those files are absent from Git. The old
training SQSH was also absent and its OCI digest was not recorded in the
handoff. The PBS port replaces these with pinned build recipes. Their live CUDA
build/import gate passed in job `2132611.pbs1`; a final content-key replay is
required only because the Gate-5 instrumentation is being added afterward.

The v8 Toolathlon service comes from
`ghcr.io/apga/toolathlon-gym:18e62c0d0419da88209af911528f2f8a0b44a5ed`.
The published OCI index is
`sha256:494af57e8a0f158d668982bf985b5937e73ba58eb468715f884de03550bd677c`;
its linux/amd64 child is
`sha256:c1e60b3673c848b758bda1b32ec39898feb5d6ff4c7e55ead8661adb04b327bd`.
The ABCI SIF is
`openreward-toolathlon-18e62c0d-c1e60b36.sif`, with SHA-256
`e730e11d1abc63f5cac9eb1788e36da79aa9c7027ed89d91ee42a5b1e311e267`.
Its embedded labels bind the upstream repository, exact `18e62c0d` revision,
and amd64 child digest. The older v7 service pin
(`openreward-toolathlon-gym` child `2c1b624d`, SIF `814c361f`) remains only in
the archived v7 evidence and is not reused by v8.

The checked-in `pbs-scripts/openreward-v8-release.json` records the complete
non-secret v8 component and image provenance. The release overlay and common
launcher independently verify the SIF digest, source revision, and amd64 OCI
label before any service is started.

The named upstream refs were refreshed at `2026-08-13T07:54:55Z`. Platoon's
`apga/update-areal-with-registry`, OpenHands' `apga/ptc`, and Toolathlon's
`apga/session-db-shim` exactly matched the release revisions (zero commits
ahead or behind). The SWE fork publishes only `main` at `25b14c06`; the pinned
`538b5c5c` Singularity backend is exactly one local commit ahead and zero
behind that public ref. These observations are also fail-closed manifest data.

## Singularity overlay result

On ABCI, normal setuid Singularity CE with a manually seeded, uid-owned ext3
overlay supports persistent rootfs writes while retaining the submitting UID.
`--fakeroot --overlay` forces squashfuse, whose `allow_other` mount is rejected
by the site configuration; Singularity then extracts a read-only rootfs.
OCI-mode fakeroot can write an external overlay but does not support instances.
Consequently SWE uses normal-mode persistent instances and Toolathlon uses a
normal-mode service with node-local, user-owned PostgreSQL state. Fakeroot is
used only for the short Toolathlon seed-copy operation.

ABCI deviation: Singularity CE 4.4.1 writes unprivileged instance metadata in
the real passwd-database home even when `SINGULARITY_CONFIGDIR`,
`APPTAINER_CONFIGDIR`, or an alternate exported `HOME` is set. SWE therefore
uses exact UUID instance names, a mandatory per-session `--pid-file`, hostname,
Linux PID start time, owner record, and held lease. Both the in-process janitor
and PBS outer cleanup serialize only these prefix-scoped control operations,
require all identities to agree, and fail closed rather than using wildcard or
`--all` operations. Containers still use `--no-home` and do not bind host home.

The imported OCI task SIF has an executable but empty `%startscript`. A locked,
atomic adapter cache derives a new immutable SIF adding only a long-lived
startscript. The cache key covers the raw SIF SHA-256, exact adapter content,
and adapter version; a read-only provenance sidecar records raw/adapted digests
and builder version. Builds stage the raw image on node-local storage because
ABCI fakeroot cannot traverse the group path directly.

Normal setuid mode also cannot modify root-owned, mode-0644 lower source files
directly. Before formatting each ext3 overlay, the backend discovers the one
Git checkout in a contained read-only image view and copies only that checkout
through a session-private bind into the UID-owned upper tree. It rejects unsafe
paths, missing Git metadata, foreign ownership, and seeds consuming over 80%
of the overlay. Gate 3 compares the same tracked file as UID 0 in the immutable
lower SIF and as the submitting UID in the mounted session before editing it.

Secrets live only in `/home/aci18912it/rasp/config/secrets.env` (mode `0600`) or
standard protected client stores such as `.netrc`; they are never placed in PBS
scripts, Git, or command-line arguments.
