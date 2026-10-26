# RFC 0010: Harbor datasets, images, grading and multi-container tasks

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-14; revised 2026-10-16 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | The Harbor service (`platoon/envs/harbor/`), sandboxes (`platoon/sandbox/`), the harness runner (`platoon/harness/runner.py`) and the gateway (`platoon/gateway/server.py`); [Architecture](../architecture.md) (the Harbor backend, sandboxes, the native-tools rule); [RFC 0001](0001-native-multi-agent.md) (per-agent grading, `snapshot_async`), [RFC 0004](0004-sandweave-integration.md) (capabilities and routes), [RFC 0005](0005-gateway-dialects-and-tito.md) (dialects), [RFC 0008](0008-config-and-cli.md) (`platoon eval`), [RFC 0009](0009-gateway-step-budgets.md) (step budgets) |
| Discussion | The pull request that adds this revision |
| Implementation | The Harbor fast-path pull request: M0; M1 for `hf://` and local sources; M2 (no `platoon toolchain ls`, and the `docker` builder untested on a Docker host); M3; M4 (its kill is `Sandbox.kill_processes`). The compose-subset pull request: M6's compose parser and subset (§6), with stack plans in the lock, compose overlays, and compose files that name only `main` run as one sandbox on every provider. The stack-layer pull request: the provider-neutral stack layer (`platoon/sandbox/stacks.py`: the stack spec, `Stack`, the `stacks` capability and a driver that starts services in dependency order), sidecar stacks on the `process` provider, sidecar probes, collect hooks and sidecar artifacts, `stacks`, `network_policy` and `on_sidecar_down`. The sources pull request: the rest of M1, the Harbor Hub, Harbor's legacy registry, git repositories and adapters (no `ls`, and no `service: harbor://...` shorthand). The Docker-stacks pull request: stacks on the `docker` provider as `docker compose` projects, sidecars' build contexts built by `platoon prepare`, and Docker-only privileges for a fetch that keeps them (`refuse: {privileges: false}`, the `privileged` capability). The pods pull request: M7's stacks as hpc-sandbox pods on Singularity, and on Apptainer and enroot (experimental), with hpc-sandbox 0.5 (with 0.4 they are refused, naming it): a run mode per service, the gateway relayed by the pod, volumes in the pod's directory, egress per phase; the registry images that sidecars run pinned with their configuration by `platoon prepare` (`platoon harbor images pin`), which also fetches the pod tools. Not yet: M5, the rest of M6 (task MCP servers), enroot pods at a site, M8, M9, and trajectories recording the kill mode |

## Summary

platoon runs Harbor tasks only:
- from a local directory;
- as one container;
- when the task grades in its own sandbox;
- when the harness can reach the gateway over the host's loopback.

This RFC lets platoon take Harbor datasets as their authors publish them, and train any harness on them:
- **Datasets by name.** A config names a dataset (`hf://org/name@rev`, `harbor://org/name@ref`,
  `git+https://…@commit`), or the adapter that generates it. platoon fetches it once into a content-addressed cache and
  pins it with a lock that workers read offline.
- **Images.** They are built once per site from the tasks' Dockerfiles, with many tasks per image: trailing `COPY`,
  `WORKDIR` and `ENV` steps are applied when a session opens.
- **Toolchains.** A harness's toolchain is mounted read-only into the task sandbox.
- **Gateway.** The sandbox reaches the gateway through a mounted Unix socket, so CLI harnesses work in tasks without
  network.
- **Grading** gains:
  - Harbor's separate-verifier mode with artifacts;
  - reward contracts and log classifiers that a dataset supplies;
  - explicit rules for harness timeouts and crashes.
- **Multi-container tasks** run:
  - as one rootless network namespace per stack on Singularity, Apptainer and enroot, with fixed ports and real
    service names;
  - through `docker compose` on Docker.

Workers never import Harbor, and every outcome keeps platoon's rule that an infrastructure failure platoon detects
never trains as reward 0; the opt-in harness exit rules (§5.4) say which harness crashes are graded, and which
infrastructure failures they cannot tell from the policy's own.

## Motivation

### What exists

This table describes platoon when the revision was written, before the fast-path work that the `Implementation`
row lists.

| Harbor feature | platoon today (`platoon/envs/harbor`) |
|---|---|
| Dataset source | a local `dataset_dir` (only its immediate subdirectories: `dataset.py`) or explicit `task_dirs`; no registry, Hub, git or Hugging Face source; task ids are directory names, and the loaders use absolute paths |
| Image | `[environment].docker_image` as `docker://<image>`, else the service's `default_image`. A `.sif` or `.sqsh` path also gets the `docker://` prefix (`service.py`). `environment/Dockerfile` is never built, and `environment/` is never uploaded for prebuilt-image tasks, which Harbor does |
| Harness toolchains | `InstallSpec.toolchain` must exist where the harness runs, but nothing mounts it (`platoon/harness/spec.py`). Install commands run in every sandbox and need network |
| Gateway reachability | `{base_url}` is `http://127.0.0.1:<port>` (`platoon/harness/rollout.py`), so a harness in the task sandbox needs the host network. A `no-network` task cannot run a CLI harness at all |
| Verifier | `tests/` uploaded at grade time, `bash /tests/test.sh`, then `reward.json` or `reward.txt`. A `reward.json` without a `reward` key is an infrastructure error (`verifier.py`). Separate verifier environments are refused (`task.py`) |
| Agent timeout | `[agent].timeout_sec` is parsed but unused; a harness timeout or crash is excluded without grading, while Harbor grades after both |
| Multi-container tasks | refused at load, whatever their image source (the gap that let `main` run alone and train as reward 0 is closed) |
| Network | `host` or `none`; an allowlist runs as `host` |
| Resources | `cpus` and `memory_mb`; `gpus` is ignored, so a GPU task runs without one; `[agent].user` is ignored |
| Providers | `process`, `docker`, and `singularity` and `enroot` through hpc-sandbox |

Harbor support has been validated on 8 offline single-container toy tasks on Singularity: the oracle scored 1.0, a
no-op scored 0.0, and platoon-bash and mini-swe-agent ran.

### What published task sets need

- **Separate verifiers with artifacts.**
  - All 70 tasks of Terminal-Bench-Science on the Harbor Hub (digest `sha256:91531bf5…`) declare
    `[verifier] environment_mode = "separate"` and `artifacts`, and build their verifier image from
    `tests/Dockerfile`.
  - All 70 build their environment from `environment/Dockerfile` (11 are multi-stage), and none sets
    `docker_image`.
  - 5 are compose stacks.
  - Their agent phase is `public` and their verifier phase `no-network`, with agent timeouts of 28,800 s.
  - Training pools that adapt public sources for offline RL make the same move: they bake inputs into images and
    grade declared artifacts in a fresh verifier container, so the agent cannot touch the grader. platoon refuses
    every such task today.
- **Offline tasks with CLI harnesses.** Pools built for RL usually set `network_mode = "no-network"`: their
  instructions were scrubbed of pointers to the answers. A CLI harness, which acts with its own tools inside the task
  container, cannot reach platoon's gateway from such a sandbox.
- **Multi-key rewards.** Some sources write `reward.json` with several named rewards and no `reward` key. For
  example, `{correctness, submission, tool_efficiency}` in a public data-analysis task set, where `submission` is 1
  for any non-empty answer. Which key trains, and what makes a reading invalid, is the dataset's knowledge, not
  platoon's.
- **Graders that write 0 on an infrastructure failure.** Upstream verifiers often pre-write `reward=0`, then crash
  (a missing `pytest`, a `subprocess.TimeoutExpired`). Only the verifier's log tells the two apart.
- **Datasets on the Hugging Face Hub.** Large pools are published as dataset repositories with nested
  `tasks/<source>/<id>/` directories and their own `registry.json`. Harbor's git path clones such a repository
  whole, needs git-lfs and credentials, and finds only top-level task directories.
- **Images from Dockerfiles on HPC sites without Docker.** SingularityCE 4.1 and later builds Dockerfiles with
  BuildKit (`singularity build --oci`), but the native runtime cannot run the resulting OCI-SIF.
- **Multi-container tasks.** Frontier-CS's 201 Harbor tasks are compose stacks with judge services. Harbor's catalog
  has more: of its 85 adapters, 15 emit compose files.

Multi-harness training on an ORS environment needs none of this: the MCP tool plane serves an ORS session's tools to
any MCP-capable harness. This RFC is for harness-native training on Harbor tasks, where the harness keeps its own
tools inside the task container.

## Proposal

### 1. Dataset sources, cache and lock

A Harbor environment names its dataset in `service_kwargs.dataset`, or the adapter that generates it in
`service_kwargs.adapter`. `service: harbor://org/name@ref` is shorthand, like the `ors+http://` service strings.

| Source | Spelling | Resolved by | Index |
|---|---|---|---|
| Hugging Face dataset repository | `hf://org/name@revision[#subdir]` | `huggingface_hub`, in two passes: the index files, then only the selected task directories. The second pass lists those directories and downloads each distinct file once (`hf_hub_download`), hard-linking it into every task that has it; `snapshot_download` lists the whole repository and filters `allow_patterns` per file, which is slow for thousands of tasks | the repository's `registry.json` (Harbor's `DatasetSpec`), else a recursive `task.toml` scan |
| Harbor Hub package (a dataset, or a single task) | `harbor://org/name@ref` (a tag, a revision or a `sha256:` digest) | the pinned Harbor CLI (`harbor==0.24.0`) in an isolated environment: `uvx --from harbor==<pin> harbor version show org/name@ref --json` resolves the reference to the package's digest and its tasks' digests, then `harbor download org/name@<digest> --export` downloads that digest; each task's files must hash to the Hub's digest | the Hub's list of the version's tasks, named by their names |
| Legacy or JSON registry | `harbor://name@version`, plus `registry: <file, directory or URL>` (default: Harbor's own `registry.json`) | platoon reads the registry itself and takes each selected task from its `git_url` at its `git_commit_id` and `path` through the git path below (a Hugging Face repository's through the `hf://` path), rather than the CLI's clone of whole repositories | the registry's tasks |
| Git repository | `git+https://host/org/repo@<commit>#<path>` | platoon: a shallow, blobless fetch of the commit, then a sparse checkout of the selected tasks | `registry.json`, else `task-index.json`, else a recursive scan of the commit's tree |
| Upstream adapter | `adapter: {repo, commit, path, command, args, env, sparse}` | a checkout of the repository at the commit (whole, or the `sparse` patterns), where `command` (default: the adapter project's own script through `uv run --project`) runs with `args` (Frontier-CS, §8) | its output, as a local directory |
| Local directory | a path; `dataset_dir` and `task_dirs` keep working | platoon | `registry.json` when present, else a recursive scan |

```python
@dataclass(frozen=True)
class LockedTask:
    name: str  # the id: the registry name, else the task index's key, else the path relative to the dataset
    digest: str  # "sha256:<Harbor content hash>"
    path: str  # relative path in the cache entry
    index: Mapping[str, Any]  # metadata from the dataset's own index files, for selection and splits
    flags: Mapping[str, Any]  # verifier mode, stack kind, network modes, gpus, agent user
    image: BuildPlan  # §2
    refused: str | None  # why platoon cannot run it, recorded instead of skipping silently


@dataclass(frozen=True)
class DatasetLock:
    dataset: str  # as written in the config
    resolved: str  # commit, revision or Hub digest
    fetched_with: str  # e.g. "huggingface_hub==1.10.1" or "harbor==0.24.0 (cli)"
    source: Mapping[str, Any]  # how a Hub, registry, git or adapter source resolved: digests, commits, the command
    hash_rule: str  # "harbor-packager-v1"
    overlay_hash: str | None  # §2: dataset overlays, hashed with any host-data pins
    tasks: tuple[LockedTask, ...]


def fetch(config: FetchConfig, *, update: bool = False) -> DatasetLock: ...
def verify(lock: DatasetLock, *, cache: Path) -> None: ...  # offline; raises on any mismatch
```

**Fetching and pinning:**
- **Once, outside rollouts.** `platoon harbor fetch CONFIG` resolves a source on a host with network access.
  - It writes each task to `<cache>/tasks/<content hash>/`, building in a temporary directory and renaming under
    `flock`, so concurrent fetches never expose a partial task. An entry that exists is hashed again before it is
    reused; one that no longer matches is renamed aside, replaced and deleted, under the same `flock`.
  - It applies overlays (§2), computes build plans, records refusals (the `refuse` policy's, and every error of
    the task loader, as `task_config`) and writes the lock.
- **Automatically for pinned references.** `platoon train` and `platoon eval` fetch a reference pinned to a commit,
  revision or digest at startup, in the trainer process, if no lock lists it. Pointing a config at a dataset is then
  enough.
- **Moving references** (`@latest`, a branch, a Hub tag, a registry task at `HEAD`, the default registry itself)
  resolve only through `platoon harbor fetch --update`, which rewrites the lock with what they resolved to. A Hub
  revision or digest, a full git commit, an adapter's commit and a version in a registry file (or at a URL with a commit
  in its path) are pinned.
- **Workers read the lock and the cache, never the network.** They look a task up by its id in the lock, never by
  scanning a directory. Each cache entry's hash is checked once per node before its first session, not once per
  process, because task directories can carry large input data. A mismatch fails with `env_config`.
- **What a source downloads stays inside each task.** A symbolic link in a Hub export, a git checkout or a registry's
  checkout is taken as the file it names when that file is in the same task, as the content hash takes it, and fails
  the fetch when it leaves the task (absolute, climbing out, a linked `solution/`, a task behind a link): it would
  bring the fetching host's files into the cache and every sandbox. A URL with credentials is refused, since the lock
  and the logs record URLs as written.

**Ids and selection:**
- **Ids** never contain cache paths or content hashes, so a content update moves neither the harness draw
  (`sha256(seed:task_id)`) nor the GRPO group keys, and duplicates fail at fetch. They are the registry's names,
  else the keys of `task-index.json`, else, from a scan, each task's path relative to the dataset: the directory
  name for a task directly under it, so existing split manifests keep working, and `group/name` for one found
  deeper. A fetch of a local directory gives the ids `dataset_dir` gives it. Index paths that are absolute or contain
  `..` fail the fetch.
- **Selection.** `tasks:` names allowlist files of ids, and only those tasks are downloaded. `select:` filters on
  the dataset index's fields (`index.*`, or the bare field name) and on `name` globs. The lock's `flags` are
  computed by the fetch itself, so a selection on `flags.*` is refused; choose by flags with an allowlist instead.
- **A dataset's own integrity rule**, such as a bundle digest in its manifest, is checked by a hook from the
  dataset's package. platoon does not copy such rules.

**Harbor itself:**
- **The hash is platoon's code.** Harbor's task content hash (`Packager.compute_content_hash`) is sha256 over the
  lines `relpath\0sha256(file)\n`, sorted by POSIX relative path, of `task.toml`, `instruction.md`, `README.md` and
  `trajectory.json` where present, plus every file under `environment/`, `tests/`, `solution/` and `steps/`, filtered
  by the task's `.gitignore` or Harbor's default ignores. platoon computes it without Harbor.
- **The Harbor CLI, not the library.** Harbor 0.24.0 needs Python 3.12 or later, litellm 1.92 or later and
  supabase, and its core closure is about 92 packages. Its minor releases change agent, trial and environment APIs.
  So platoon runs its `version show` and `download` commands in a `uvx` environment and never imports it
  (`PLATOON_HARBOR_CLI` names another CLI's command line), with its dependencies as PyPI had them at its release
  (`--exclude-newer`), so that every host installs the same closure; its telemetry is off unless the fetching host
  turns it on.
  - The command has no task filter, so a `harbor://` source downloads the whole dataset; Terminal-Bench-Science is
    70 tasks and 727 MB, Terminal-Bench 4 66 tasks and 324 MB.
  - The download is by the digest the reference resolved to, so a tag that moves meanwhile cannot change the tasks.
  - Fetches retry what may pass (a transient HTTP/2 stream reset has been seen), and not what the CLI's or git's own
    error lines report as not found or no access; a dataset named like an error does not count as one.
- **Git without whole clones.** A git source (and each repository a legacy registry names) is fetched shallow and
  without blobs (harbor-datasets' tree at one commit is 41 MB of about 926,000 paths), and only the selected tasks'
  files are checked out, with a sparse index. git cannot fetch an abbreviated commit, so a pinned one is named in full.
  Files in Git LFS need `git lfs` where the fetch runs, and a pointer left in a task fails the fetch.
- **Adapters run code at fetch.** An adapter source runs its repository's code on the fetching host, with its network,
  never in a rollout. Adapters differ in how they run (Harbor's are `uv` projects with a script; Frontier-CS's expect
  Harbor's tree and run as modules), so the command is configurable; the lock records the repository, commit, command,
  arguments and the digest of the adapter's `uv.lock`.
- **Hugging Face sources do not go through Harbor's HF git path**, which needs git-lfs and credentials and clones
  the whole repository. The Hub's HTTP downloads keep no file modes, so the fetch restores the executable bit on
  `*.sh` files and files with a shebang, and records that in the lock. Only the files the content hash covers are
  downloaded (each task's `.gitignore` is read first, else Harbor's default ignores apply), and every download is
  checked against the listing: an LFS file's SHA-256, else its git blob id. A mismatch is downloaded once more with
  `force_download`, then fails the fetch.

### 2. Task images

A task names its images in four places: `[environment].docker_image`, `environment/Dockerfile`,
`tests/Dockerfile` (a separate verifier's image), and `image:` or `build:` per compose service.

**Build plans, computed at fetch:**

```python
@dataclass(frozen=True)
class BuildPlan:
    key: str  # sha256 of the normalized build prefix plus the hashes of the context files it references
    kind: Literal["prebuilt", "shared", "per_task"]
    deferred: tuple[Step, ...]  # trailing COPY/ADD of local files, WORKDIR, ENV, LABEL: applied at session open
```

- **The prefix and the suffix.**
  - The build prefix is the Dockerfile up to and including its last instruction that cannot be deferred (a `RUN`,
    a `FROM`, a `COPY --from`, …).
  - The deferred suffix is the instructions after it that copy task files or set metadata. Nothing executes after
    them at build time, so applying them when a session opens is equivalent.
  - `USER` in the suffix is refused unless the environment accepts running as one uid (§7).
- **Sharing.** Tasks whose prefixes match share one image. A data-analysis pool whose Dockerfiles differ only in a
  trailing `COPY input/` needs one image, not one per task.
- **Overlays.** A pool whose `RUN` comes after its `COPY` can be reordered by a dataset overlay. Overlays are task
  files replaced at fetch and hashed into the lock; cached tasks stay byte-identical to their upstream hashes.

**The image map is a site file.** The lock is site-independent; the map binds build keys to the site's images:

```python
@dataclass(frozen=True)
class ImageEntry:
    image: str  # a .sif or .sqsh path (never given a docker:// prefix), or docker://…@sha256:…
    sha256: str | None  # receipt of the file
    env: Mapping[str, str]  # from the image config; applied to every exec
    workdir: str | None
    user: str | None
    entrypoint: list[str] | None  # for sidecars (§6)
    cmd: list[str] | None
```

**At session open:**
1. The image comes from the map.
2. The image's env is merged with the task's.
3. Deferred `COPY` sources are uploaded from the task directory, and deferred `WORKDIR` directories created.
4. A prebuilt-image task without a Dockerfile gets `environment/` uploaded into its workdir, as Harbor does.

A task that builds from a Dockerfile and has no map entry is refused with `env_config`, never run on a default
image.

**Building.** `platoon harbor images plan LOCK` lists the keys and which are missing; a map entry whose image file is
gone counts as missing. `platoon harbor images build LOCK --image-map MAP [--builder oci|def|docker] [--shard i/n]`
builds the missing ones and writes receipts (builder, version, log, sha256, and `base_digests`: the digest each base
image resolved to). A build resolves each `FROM` image through the registry's manifest endpoint (with an anonymous
token where the registry asks for one) and builds from `reference@digest`. The build key is computed offline at
fetch and does not include the digest, so it does not change. A base that cannot be resolved is built as written,
with a warning and a null digest. `platoon harbor images verify LOCK --image-map MAP` checks that the map has every
key the lock's runnable tasks use and that each local image file hashes to its receipt's sha256 (exit 1 when one is
missing or mismatched; registry references are not checked). `platoon harbor images alias` points keys at an
existing image, such as `local` for the `process` provider in CPU tests.

| Builder | How | Status on a setuid SingularityCE 4.4.1 site |
|---|---|---|
| `oci` (SingularityCE 4.1+) | 1. `singularity build --oci` (rootless BuildKit) builds the Dockerfile as written. 2. `singularity sif dump` extracts the single squashfs layer and the image config. 3. `unsquashfs` unpacks the layer. 4. `singularity build native.sif <rootfs>` makes a native SIF. 5. `ENV`, `WORKDIR`, `USER`, `ENTRYPOINT` and `CMD` go into the map from the config | checked: pip, `apt-get` as root, `COPY` and `useradd` builds (60–73 s each), and the native SIF runs in an hpc-sandbox sandbox with its env re-applied. The OCI-SIF itself converts neither to a native SIF nor to a sandbox directory |
| `def` | `Bootstrap: docker` with `%post`, `%files` and `%environment`, built with `--fakeroot`. Each `RUN` runs in a subshell of its own (`/bin/sh -c` from the current `WORKDIR`) with the `ARG` values exported, and `ENV` values are escaped so that only their variable references expand. It refuses multi-stage builds, `ARG` without a default, `ADD <url>` and flags; it fails loudly, never falling back to the base image | checked for pip and `apt-get` |
| `docker` | Docker, Podman or BuildKit on another host, pushed by digest; enroot sites import with `enroot import docker://…@sha256` | — |

hpc-sandbox pulls registry references itself, caching them under `flock` and staging them to node-local disk.

### 3. Harness toolchains

A harness version is part of an experiment, so a CLI harness comes from a versioned, read-only toolchain, not from
whatever an image baked in or an install downloads.

```toml
# in a harness spec
[install]
toolchain = "openhands-1.54.0"          # a name under the service's toolchain_root (an absolute path still works)
[install.recipe]                        # what `platoon toolchain build` installs
python = "3.12.14"                      # a standalone CPython, pinned to its patch release: images need glibc
lock = "openhands-1.54.0.txt"           # the package closure with hashes, shipped next to the spec
cache = ["tiktoken"]
[resources]
memory_mb = 2048                        # the harness's own processes, added to the sandbox's memory limit
```

- **Build.** `platoon toolchain build <spec>` writes `<toolchain_root>/<name>/`: the interpreter, the locked
  packages, caches, generated files, and a manifest with the tree hash, the build path and the recipe's identity
  (`recipe_sha256`, over the Python version, the lock, the caches and the generate scripts). Building again is a
  no-op while the identity matches; another recipe, or a manifest without one, needs `--force`.
- **Mount.** `EnvironmentService.open` gains the drawn harness's needs (§4). The Harbor service stages each
  toolchain to node-local scratch (`$PLATOON_NODE_SCRATCH`, `$PBS_LOCALDIR`, `$SLURM_TMPDIR`, `$TMPDIR`, else `/tmp`;
  `$PLATOON_TOOLCHAIN_CACHE` overrides it) once per node (under `flock`, checked by tree hash; the next staging
  removes what a killed one left half done) and binds the staged copy read-only at the toolchain's build path. Interpreters and console scripts then find their files without being
  relocatable.
- **Nothing installs in the sandbox.** Install steps that need the network run once per toolchain instead of once
  per rollout.
- **Memory.** `[resources].memory_mb` is added to the task's limit. Rootless providers enforce memory with a watchdog
  that kills the largest process, which would otherwise be the harness.
- **Images without glibc** (musl) cannot run a toolchain built on python-build-standalone; the harness fails
  with `harness_setup` when it starts (an earlier check is not built yet).

### 4. Reaching the gateway from a task sandbox

The rollout says what the sandbox must reach, and the provider says how to reach it from inside. This is
RFC 0004's route abstraction, adopted for every provider.

```python
@dataclass(frozen=True)
class SessionNeeds:  # new: computed by the rollout before the sandbox starts
    upstreams: Mapping[str, str] = field(default_factory=dict)  # {"gateway": "unix:///…/gw.sock"}
    mounts: tuple[Mount, ...] = ()  # toolchains (§3)
    extra_memory_mb: int = 0


class EnvironmentService(Protocol):
    async def open(self, task: Task, *, rollout_id: str, needs: SessionNeeds | None = None) -> EnvSession: ...


@dataclass
class SandboxSpec:  # additions
    upstreams: dict[str, str] = field(default_factory=dict)
    network: Literal["host", "none", "egress"] = "host"  # "egress": outbound only, host loopback blocked (§6)


class Sandbox(Protocol):  # additions
    routes: Mapping[str, str]  # {"gateway": "http://127.0.0.1:18080"}: what a process inside should call
```

| Provider | `routes["gateway"]` |
|---|---|
| any provider with a host-network sandbox | the gateway's loopback TCP URL, as today |
| `singularity`, `enroot`, `docker` with `network: none` | the upstream's directory bound read-only at `/run/platoon/gateway`, plus a small in-sandbox forwarder from `127.0.0.1:18080` to the socket in it |
| stacks (§6) | the stack's namespace holder relays a fixed port to the upstream socket |
| remote sandbox servers | a relay in the server (a separate RFC) |

**How the route is wired:**
- **The gateway listens on a Unix socket** in a per-rollout 0700 directory, next to its loopback TCP port. The
  directory path is short and opaque, because a socket path holds at most 108 bytes and rollout ids approach 100
  characters. The directory, not the socket file, is bound into the sandbox, so the gateway can start before or
  after the sandbox. The bind is read-only: connecting to a socket needs no write access to its mount (checked on
  SingularityCE 4.4.1, with and without its own network namespace), and code in the sandbox cannot create or fill
  files in the host directory.
- **The runner renders `{base_url}`** from `session.routes["gateway"]`. `NO_PROXY` already covers loopback.
- **The forwarder** is stdlib Python run with the harness toolchain's interpreter. One per sandbox serves every
  agent in it, since agents are told apart by their keys. It carries an infrastructure marker, so agent cleanup
  spares it, and the runner restarts it if a harness launch finds it gone.
- **Checked on a setuid SingularityCE 4.4.1 site:**
  - rootless `--net --network none` sandboxes;
  - a Unix socket mounted into one, reached through such a forwarder;
  - three OpenHands front ends finishing an episode over it, with egress blocked.
- **Hardening.** The gateway keeps its loopback TCP port, which harnesses on the rollout host use, so the route does
  not take that port off a shared node. It comes with four guards:
  - a rollout's gateway takes only that rollout's agent keys, on its socket and on its TCP port alike, so a key read
    in one rollout's sandbox is refused by every other rollout's gateway;
  - each rollout has a cap on minted agents (`POST /v1/platoon/agents`, 64 by default);
  - specs without their own step limit get a call cap, which counts the policy calls of the capped agent and of
    every agent minted under it together;
  - a harness in the task sandbox, where code the policy runs can read its key, may make no `judge` calls, and under
    a call cap only the policy calls the cap counts: its other calls get a permission error (403) and nothing is
    sampled for them.

### 5. Grading

#### 5.1 Shared mode

Today's flow stays: wait for calls in flight, hold the sandbox exclusively, reset `/tests` and `/logs/verifier`,
upload `tests/`, run `bash /tests/test.sh`, read the reward files, and remove `/tests`. One step is added before it:
- **Kill every process the agents started.** The runner already kills the processes that carry an agent's marker.
  Background processes started by env-tool calls, and children that dropped the marker, survive until the sandbox
  stops, and could write `/logs/verifier/reward.json` while the tests run.
- **Sandbox method.** `Sandbox.kill_processes()` kills every process except the ones that keep the container
  alive, which each provider lists when the sandbox starts. Grading with a provider without it logs a warning
  and runs the tests with the agents' processes alive.
- **Not `kill -1`.** A bare `kill -1` from inside the sandbox is not a substitute. Inside a Singularity instance it
  also kills the keep-alive start script, and an instance whose start script exits is removed.

#### 5.2 Separate mode with artifacts

For tasks with `[verifier] environment_mode = "separate"` (or a `[verifier.environment]` table), following Harbor's
semantics:
1. Kill the agents' processes and hold `main`.
2. Download the declared `artifacts` and `/logs/artifacts`. A missing artifact is not uploaded, and the tests
   decide.
3. Release `main` unless forks or other agents still hold it.
4. Choose the verifier image:
   1. `[verifier.environment].docker_image`;
   2. else the image built from `tests/Dockerfile` (or a compose file there), through the image map;
   3. else the task's own image and deferred steps.
5. Start a verifier sandbox with `[verifier.environment]`'s resources and network and `[verifier].env`, without
   any harness route.
6. Upload each declared artifact to its original path and `/logs/artifacts` back to `/logs/artifacts`, as Harbor
   0.24 does, then `tests/` to `/tests` (unless the verifier's image holds its own).
7. Run `test.sh`, then read the reward files under the dataset's contract (§5.3).
8. Stop the verifier sandbox on every path.

**Refused:** tasks with `[[verifier.collect]]` hooks or with artifacts that name a compose service, with
`env_config`.

**Snapshot-at-finish verification.** On providers that cannot snapshot, the same mechanism (collect, then a fresh
sandbox) gives a verifier the state an agent left, as RFC 0001's `snapshot_async` needs.

#### 5.3 Reward contracts and log classifiers

```python
@dataclass(frozen=True)
class RewardContract:  # service_kwargs.reward
    primary: str = "reward"  # the key that trains; becomes Outcome.primary
    components: tuple[str, ...] | None = None  # allowed keys; others give invalid (verifier_unexpected_keys)
    range: tuple[float, float] | None = None  # out of range gives invalid (verifier_out_of_range)
    check: str | None = None  # "module:fn"; fn(rewards, verifier_dir, task) -> Outcome | None


@dataclass(frozen=True)
class LogClassifier:  # service_kwargs.verifier_classifiers
    pattern: str  # multiline regex over the verifier's stdout and stderr
    kind: str  # failure_kind: a match makes the outcome invalid, whatever the reward file says
```

- **`primary`** lets a dataset train on `correctness` instead of failing with `verifier_no_primary`.
- **`check`** is the dataset package's own rule, for example "`correctness` counts only when `submission` is 1",
  run on the reward files and the verifier's log directory.
- **Classifiers are opt-in.** A test that runs the agent's program under a timeout and lets the timeout escape
  measures a slow solution, not a broken grader, so no pattern is safe as a global default. platoon ships named
  sets (`harbor-common`: a timed-out grader subprocess, a missing test runner) that environments name explicitly.

#### 5.4 Harness exits and timeouts

```python
@dataclass(frozen=True)
class HarnessExitPolicy:  # environments[].on_harness_exit, next to on_context_overflow
    timeout: Literal["grade", "exclude"] = "exclude"
    crash: Literal["grade", "grade_if_gateway_clean", "exclude"] = "exclude"
```

- **`grade` is Harbor's rule.** A trial records an agent timeout or a non-zero exit and still verifies. `exclude`
  keeps platoon's current rule.
- **`grade_if_gateway_clean`** grades a crash unless the gateway failed one of that agent's calls (a sampling
  error or timeout, or another 5xx answer), counted per agent.
  - It sees only the gateway's own failures. A crash that other infrastructure caused is graded like one the
    policy caused: a harness client that gave up on a slow call and disconnected, a route that broke, the memory
    watchdog killing the harness (which nothing records yet). Harness clients therefore need a request timeout
    longer than the gateway's sampling timeout (`TINKER_SAMPLE_TIMEOUT_SECONDS`), so that a slow sample ends as a
    gateway error rather than as a client that hung up.
  - A harness that runs inside the task sandbox can be killed by the policy's own commands, or by the memory
    watchdog when the policy's processes grow.
  - Excluding every crash would hide such policy failures. Grading every crash would train transport failures as
    reward 0.
- **The harness's timeout** is `min(spec timeout, [agent].timeout_sec × agent_timeout_multiplier)`, capped by the
  rollout's `timeout`.
- **A gateway step budget** ([RFC 0009](0009-gateway-step-budgets.md), for specs that declare
  `[budget] enforce = "gateway"`) ends the harness with exit 0, and the episode is graded with `reason="budget"`.

**New failure kinds:** `verifier_start` and `artifact_collect` (`error`); `verifier_contract`,
`verifier_unexpected_keys`, `verifier_out_of_range` and classifier kinds (`invalid`); `harness_crash_gateway`
(`error`).

### 6. Multi-container tasks

A fetch reads `environment/docker-compose.yaml`, the only compose file Harbor reads (`compose.yaml` and compose's
other names are plain files of a Harbor task, which a fetch warns of), the way Harbor layers it: the task's file over
a base that defines `main`, platoon's own (the task's image, `sh -c "sleep infinity"`), never Harbor's egress
sidecar, since platoon has network phases of its own.  It records a **stack plan** in the lock (`compose`: each
service's image or build context with its build key, command, environment, volumes, healthcheck, dependencies,
ports and networks, and what was ignored, noted and edited), under the rule `compose_rule: harbor-c178c20`, so that a
change of Harbor's layering is a visible rule bump, and the kind in `flags.stack`: `single`, `trivial`, `main_only`
or `sidecars` (`platoon/envs/harbor/compose.py`).  Sessions read the plan from the lock; a dataset directory's
tasks are planned when they load.  Stacks run from it as a stack spec:

```python
@dataclass
class ServiceSpec:
    sandbox: SandboxSpec  # image, workdir, env, binds, resources; main's carries the harness's upstreams
    command: list[str] | None = None  # the final argv, by compose's rules over the image's ENTRYPOINT and CMD
    healthcheck: Healthcheck | None = None  # an argv, with Docker's interval, timeout, start period and retries
    depends_on: dict[str, Dependency] = field(default_factory=dict)  # started, healthy, completed_successfully
    volumes: list[Volume] = field(default_factory=list)  # named (shared by name), anonymous, tmpfs; a caller's fill
    ports: tuple[int, ...] = ()  # the TCP ports it declares
    aliases: tuple[str, ...] = ()
    network_of: str | None = None  # network_mode: service:<x>
    run_as: Literal["root", "caller", "subids"] | None = None
    shm_mb: int | None = None  # with ulimits, cap_add and stop_signal: what the provider applies where it can
    setup: Callable[[Sandbox], Awaitable[None]] | None = None  # once its sandbox runs, before its command
    timeout_s: float | None = None  # an init job's longest run


@dataclass
class StackSpec:
    services: dict[str, ServiceSpec]  # "main" is the agents' sandbox
    network: Literal["egress", "none"] = "egress"
    labels: dict[str, str] = field(default_factory=dict)


class Stack(Protocol):
    main: Sandbox  # carries routes["gateway"]
    services: Mapping[str, Sandbox]
    network: Literal["egress", "none"]
    run_modes: Mapping[str, str]

    async def health(self, services=None) -> dict[str, ServiceState]: ...
    async def logs(self, service: str, *, max_bytes: int = 2000) -> str: ...
    async def set_network(self, network) -> None: ...
    async def connect(self, service: str, port: int) -> str: ...  # a host endpoint (§6, task MCP servers)
    def rewrite(self, value: str, *, service: str | None = None) -> str: ...  # the identity but on the host network
    def volume_path(self, name: str) -> Path: ...  # a named volume's directory on this host (a sidecar's log directory)
    async def stop(self) -> None: ...  # every service, and what the stack holds; idempotent


class StackProvider(Protocol):  # a provider with the "stacks" capability
    async def start_stack(self, spec: StackSpec) -> Stack: ...
```

`platoon/sandbox/stacks.py` defines them, with a generic driver (`drive_stack`) over a backend that starts one service
at a time: it starts each service once its dependencies reach their conditions, waits as `docker compose up --wait`
does (every service running, each healthcheck passed, each init job completed within its bound), probes every service
once more before the agents act, and on a failure or a cancellation stops what it started before it raises
(`StackStartError`, naming the service, with its log), waiting for a service caught in the middle of its start to roll
back; each stop is a cleanup that a stopping worker lets finish, inside the rollout's cleanup scope.  The Harbor side
(`platoon/envs/harbor/sidecars.py`) builds the spec from the plan, starting nothing, so a run also checks before any
rollout that its stacks can start: `main` is the sandbox a main-only file would give the task, prepared (its
directories, its image's deferred files, its init) once its dependencies are up; each sidecar runs its image as
compose runs it (a registry reference, the image map's image of a build context's whole key, or `main`'s), with
compose's command over its image's: `platoon prepare` builds a fetched dataset's sidecar contexts with the provider's
builder, with their `build.args`, so no rollout builds one, and **pins** its registry images in the image map
(`pulled:<reference>`: the digest the registry serves, with the configuration it gives the image, `ENTRYPOINT`, `CMD`,
`ENV`, `WORKDIR`, `USER`, `HEALTHCHECK`, from which platoon resolves a sidecar's command on every provider, and which
a pod's runtimes do not apply by themselves), so a run whose map lacks one stops when its tasks are listed; a dataset
directory has no map, so its registry sidecars need a command of compose's and its build contexts cannot run.  Copies
of the task's files that services bind writable (or masked, or from a missing source) are volumes of the stack, filled
before any service starts, one per path and masks for every service that binds it, as compose shares the host
directory.

**The compose subset:**
- **Accepted:** `image`, `build` (`context` in `environment/`, `dockerfile`, `args`; a service's context is planned
  whole, with no deferred suffix, so that its key hashes every file it copies), `command` and `entrypoint` (a string
  is split into words as compose splits it), `environment` and `env_file`, `working_dir`, `volumes` and `tmpfs`,
  `depends_on` with its three conditions (`service_started`, `service_healthy`, `service_completed_successfully`) and
  `required`, `healthcheck` (compose's durations, `disable`, `test: NONE`; else the image's `HEALTHCHECK`), `expose`,
  `ports` (kept as `expose`: nothing is published on the host), `networks` (with aliases), `network_mode:
  service:<x>`, `hostname`, `shm_size`, `ulimits`, `cpus`, `mem_limit`, `deploy.resources`, `cap_add` of the
  capabilities a service's root holds over its own processes and files (`SYS_PTRACE`, `SYS_CHROOT`, Docker's
  defaults but `NET_RAW`), `stop_signal`, and `profiles` (such a service is not started, as Harbor starts none).
- **Ignored**, since they cannot change what a task measures: `container_name`, `restart`, `init`, `stdin_open`,
  `tty`, `labels`, `logging`, `stop_grace_period`, `pull_policy`, and `platform` when it is the host's.
- **Refused**, naming the service and key: every other key, security-relevant ones never ignored (`privileged`,
  `devices`, `pid`, `ipc`, `security_opt`, `sysctls`, `user`, `cap_drop`, `cap_add` of `NET_ADMIN`, `NET_RAW` or any
  capability over the stack's network, `DAC_READ_SEARCH` (which reads any of the host's files where a service's root
  is the host's), a `network_mode` other than another service's, GPU requests; a fetch with `refuse: {privileges:
  false}` keeps `privileged` and such a `cap_add` as the plan's `privileges`, which only a provider with the
  `privileged` capability, `docker`, grants, and every other refuses when the tasks load); binds of host paths,
  external volumes and networks, static addresses, networks that keep a service from `main`, a mix of internal and
  other networks (egress is the whole stack's or none), and two services that declare one port (a pod's services
  share one loopback).
- **Interpolated:** compose's forms (`${V}`, `${V:-d}`, `${V-d}`, `${V:?e}`, `${V:+r}`, `$$`), an unset variable
  blank, as compose makes it.  Harbor's variables take platoon's meaning: `CONTEXT_DIR` is the task's `environment/`
  in the cache, which only binds and build contexts may name; `HOST_{VERIFIER,AGENT}_LOGS_PATH` and
  `HOST_ARTIFACTS_PATH` are the session's log directories; `MAIN_IMAGE_NAME` is `main`'s image, `CPUS` and `MEMORY`
  the task's resources.  Then the task's `[environment].env`, and only the host variables an overlay lists
  (`compose_host_env`): those, and a task variable that is a host template, stay unresolved in the plan, with the
  form the file gives them (`${V:-d}`, ...), and are resolved when a session opens, so that no host value enters a
  lock.  Environment values may name both; a read-only bind, only the path a listed host variable names or one below
  it.  A file holding the private-use characters that mark them in a plan is refused, and so is one whose YAML
  aliases expand it beyond 100,000 values.  Env files are read as compose reads them (comments, quotes, escapes).
- **Volumes:** a bind of the task's own files binds the cached task (a writable one a copy of it, so that the
  content-addressed cache is never written; a missing source an empty directory, as Docker creates one); anonymous
  and named volumes below a bind mask it, and the bind is of a filtered copy in which they are empty directories, so
  that the masked files never enter a sandbox; binds of `${HOST_*_PATH}` are the session's log directories.
- **Edited** by a dataset's overlays (`[[compose]]`: `set`, `replace` and `unset` a service's keys, or mark it
  `external`, served by a node service), which the plan records.
- A refusal is recorded at fetch (`compose: <service>.<key>: <why>`, which a run that names the task meets as
  `env_config`), never a partial stack.

**A trivial compose file** that defines only `main` with nothing platoon applies (`services: main: {}`, or Harbor's
base restated) is a single-container task.  It is accepted from M0 on, because some conversion tools emit one for
every task.  **A main-only compose file**, which gives `main` settings and names no other service, runs as the
task's one sandbox on every provider: its environment, working directory, resources and binds go into the sandbox's
spec, its healthcheck is waited for, and its `command` (under the image's `ENTRYPOINT` where the image map knows it),
unless it only idles, is started once as `main`'s **init**.  The kill before grading spares the init and what it
started, daemons it detached included (`Sandbox.keep_processes`); an init that is gone by grading, which in Docker
would have ended the container, makes the outcome `Outcome.infra("main_exited")`; a `copy` fork of such a task is
refused, since a restore brings back files, not processes.  What one sandbox cannot apply yet (`cap_add`,
`shm_size`, `ulimits`, `tmpfs`, internal-only networks) refuses the task when it loads.

**Providers:**

| Provider | A stack is | Service names | Isolation between stacks |
|---|---|---|---|
| `docker` (implemented; `podman` with its compose untested) | a compose project of its own, whose file platoon writes from the stack's spec (each service's image, final command as its `entrypoint`, binds, volumes as the project's directories, resources), in a directory below the provider's `stack_root`; the driver creates and starts each service once its dependencies hold, and runs its healthchecks; at stop, the containers and networks that carry the stack's label, then the directory | compose DNS on the project's `internal` network | that network per stack, which has no address on the host, with a bridge network for egress connected only while the stack has egress |
| `singularity`, `enroot` on sites with unprivileged user namespaces (the default; implemented, with hpc-sandbox 0.5; Apptainer and enroot experimental) | a **pod**: one rootless user and network namespace per stack, with each service a container inside it, in a run mode of its own: `main` as `root`, a sidecar as `subids` (root with the caller's subordinate ids, which can switch users, as the official database images do) where the runtime and the caller give them, else as `caller` (the caller's own uid), as it is where its image's `USER` is another user | `/etc/hosts` in each service maps every service name and alias to `127.0.0.1`; ports stay as written | a private network namespace per stack: no stack reaches another stack's services or the host's loopback |
| `singularity`, `enroot` without unprivileged user namespaces (fallback) | one sandbox per service on the host network, with ports allocated and service URLs rewritten per rollout (`HostNetworkStackProvider`, below) | rewritten to `127.0.0.1:<port>` | none: sidecars must be authenticated by the overlay, or bound to node services |
| `process` (implemented) | each service a process sandbox on the host, its command in the background, for CPU tests; each declared port a port outside the kernel's ephemeral range that the stack holds for its life (a kernel-arbitrated claim, so no other stack is given it), which the service learns from its listen variable, `PLATOON_PORT_<declared port>` | `scheme://<service>:<port>` in environment values, healthchecks and collect hooks rewritten to `127.0.0.1:<its port>` | none |

**Pods** (`platoon/sandbox/hpc_stacks.py`) are hpc-sandbox's (its `docs/pods.md`): a holder process keeps the pod's
user and network namespaces, relays `main`'s upstreams (the gateway's socket) from ports of the pod's loopback that no
service declares, runs slirp4netns while the stack has egress, and dies with the process that started the pod; each
service is a long-lived `exec` container under a launcher that joins the pod, with an init of its own, in a nested user
namespace that its run mode maps; commands enter it with `nsenter`. A stack's volumes are directories in the pod's own
directory, which the pod's deletion takes, after hpc-sandbox gave back to the caller what a `subids` service wrote as
another user. A stack needs hpc-sandbox 0.5 (its run modes, host aliases and exposed ports) and a node that nests user
namespaces; a provider without either refuses stacks when the tasks are listed, naming it.

**What was checked on a setuid SingularityCE 4.4.1 site:**
- **Works:**
  - a pod holder starts in about 55 ms;
  - two `singularity exec` services in one pod share its loopback, each with its own PID and mount namespace;
  - the host cannot reach a pod's ports;
  - `nsenter` runs a command in about 1 ms, against about 40 ms for `singularity exec`;
  - `slirp4netns --disable-host-loopback` gives egress while host loopback stays unreachable.
- **Fails:** Singularity instances inside a pod ("while applying cgroups config: Interactive authentication
  required"), so pods use long-lived `exec` containers under anchors, as hpc-sandbox's enroot provider already
  does.
- **Since verified** (hpc-sandbox's pod suite, and the pods pull request's checks on SingularityCE 4.4.1): name
  resolution, the gateway relay and teardown in a pod; two stacks of one task on one port, each reaching only its own
  services; `postgres:16-alpine` and `redis:7-alpine` sidecars that switch users with subordinate ids; nothing left
  after 200 stacks, some cancelled while they started.
- **UNVERIFIED:** pods on Apptainer, and enroot pods under platoon (hpc-sandbox's ran at a Slurm site).

**Network per phase** (implemented on pods: the stack's network switches for the whole pod):
- `no-network` gives `lo` only.
- `public` gives slirp4netns with host loopback blocked.
- `allowlist` is refused on providers that cannot enforce it, unless the environment opts into running it as
  `public`.
- `[agent]` and `[verifier]` phase policies attach and detach egress around grading.
- Isolation is namespaces under one host uid: not a boundary against same-uid code outside the pods.
- Within a pod, every connection between services, `main`'s too, comes from `127.0.0.1`, where under compose `main` has
  an address of its own: a sidecar that trusts loopback (the official postgres image lets `127.0.0.1` in without its
  `POSTGRES_PASSWORD`) or listens on it alone is open to `main`'s agents. platoon documents this rather than refusing or
  changing such sidecars, since no list of them is complete and a refusal would also refuse tasks whose agents are meant
  to use them; a task that keeps a sidecar from its agents must not rely on its loopback rules (postgres:
  `POSTGRES_INITDB_ARGS: --auth-host=scram-sha-256`). An address per service, in hpc-sandbox's pods, is the follow-up
  that closes it.

**`HostNetworkStackProvider`** is the fallback for sites without unprivileged user namespaces. It allocates ports per
rollout and rewrites service names in env values, healthchecks and MCP URLs, including each service's references to
its own loopback ports. It enforces listen rules and scans task files for literal service names. It refuses
sidecars that the overlay does not authenticate or bind to a node service.

**Forks, probes and grading** (implemented, but for task MCP servers).
- `shared` forks share the stack; `copy` forks, and RFC 0001's `snapshot_async`, are refused on stacks, until stack
  snapshots exist.
- A sidecar that fails to start, exits or never turns healthy fails the session's start: `error` (`sidecar_start`),
  with its log.
- Before and after the tests (`shared`), or before and after a separate verifier's artifacts are collected, every
  sidecar must be up: an init job completed, any other running and passing its healthcheck.  Else the outcome is
  `invalid` (`sidecar_unhealthy`, with the sidecar's name, state and log), whatever the reward file says, unless the
  environment sets `on_sidecar_down: grade`, for datasets whose sidecars the policy can stop and should own.
- `[[verifier.collect]]` hooks run in their services once the agents are done, `main`'s first, best effort, as
  Harbor runs them, and a separate verifier receives the artifacts of the sidecars that `artifacts[].service` names,
  from their own files; a verifier that is itself a stack (`tests/docker-compose.yaml`) stays refused.
- A sidecar's bind of a session log directory (`${HOST_VERIFIER_LOGS_PATH}`, ...) is a volume of the stack, one per log
  directory for the sidecars that bind it, whose regular files grading copies (never through a link) into the tests'
  sandbox once it made that directory afresh, so the agents can neither erase nor forge what a sidecar records there;
  a sidecar's read-only bind of one, which would read `main`'s, is refused.
- A sidecar that is down when the start's last probe runs fails the start too (`sidecar_start`), and an init job that
  does not complete within `sandbox_start_budget_s`.
- A rollout's deadlines count `sandbox_start_budget_s` per service start and per init job's run, and each health
  wait's worst case (compose's healthcheck, else its image's), along the services' dependencies' longest path, three
  probes, and the collect hooks' timeouts.
- Task MCP servers (`[[environment.mcp_servers]]`) are proxied through the tool plane.

### 7. Resources, users and network policy

- **GPUs.** `gpus > 0` is refused on providers without the `gpu` capability. A provider with it leases whole devices
  per sandbox: one `flock` per device in a node-local directory, and the leased devices passed by UUID.
- **Users.** Rootless providers run the agents' processes as one uid, so `[agent].user` and `[verifier].user` cannot be
  honored there; a pod's sidecars run in their own run modes (§6), with the caller's subordinate ids where it has
  them, so that an image's own switch to its user works.
  - `agent_user: refuse`, the default, refuses such tasks at fetch.
  - `ignore` runs them as the single uid and records that in the trajectory.
- **Network policy** (implemented). `network_policy: task`, the default, honors the task's `network_mode` per phase
  (a stack graded in its own `main` gets the verifier's). `none` forces no network on every sandbox and stack, in
  every phase, for datasets whose answers can be found on the web.
- **Agent processes and memory.** A harness shares `main` with the policy's commands (§3, §5.4).

### 8. Frontier-CS on the Harbor path

Frontier-CS's algorithmic and 2.0 tracks are compose stacks (a judge proxy, a privileged go-judge engine, and a
2.0 judge per task). On the Harbor path:
- tasks come from the two adapters as `adapter:` sources at a pinned commit;
- an overlay carries the ORS server's verifier and judge patches, a pinned host checkout of the algorithmic tests,
  and the file-staging mounts for its generic images;
- on HPC, the algorithmic judge binds to an external, shared judge pool (go-judge needs privileges a rootless
  sandbox lacks, and one engine per rollout would swamp a node);
- the 2.0 judge runs as a per-rollout sidecar.

What changes is the default HPC provider:
- In a pod, the 2.0 judge keeps its fixed port and its `http://judge:8082` name, and only its own stack can reach
  it.
- The rewriting, listen rules and per-stack tokens apply only on the host-network fallback.
- The Harbor path remains a new baseline, not comparable with ORS-path numbers. The ORS path stays the choice for
  work that must match an ORS setup.

### 9. What changes for users

| Change | Compatible? |
|---|---|
| New `service_kwargs`: `dataset`, `adapter`, `registry`, `lock`, `cache`, `tasks`, `select`, `overlays`, `image_map`, `toolchain_root`, `reward`, `verifier_classifiers`, `agent_timeout_multiplier`, `network_policy` (`task`, `none`), `stacks` (`auto`, `refuse`; `host_network` with the fallback), `on_sidecar_down` (`invalid`, `grade`), `agent_user` | additive |
| New environment-entry key `on_harness_exit`; new rollout-config key `sampling_precedence` (`request`, today's behavior, or `config`, under which policy calls sample with the temperature, top_p and top_k of `inference_params`, never a harness's own) and `inference_params.top_k` | additive; defaults keep today's behavior |
| Spec keys `[install.recipe]`, `[resources]`, `[budget]`; `[install].toolchain` may be a name | additive |
| Overlay keys `[[compose]]` and `compose_host_env`; the lock's `compose` and `compose_rule`, and `flags.stack: main_only`; `Sandbox.keep_processes` (optional) | additive |
| The `stacks` capability and `StackProvider.start_stack` (`platoon.sandbox.stacks`); multi-container tasks run where a provider runs stacks (`docker`, `process`, and `singularity` and `enroot` with hpc-sandbox 0.5), `[[verifier.collect]]` hooks and sidecars' artifacts run | additive; tasks platoon refused now run |
| The `privileged` capability (`docker`) and the fetch's `refuse: {privileges: false}`; the `docker` provider's `stack_root` option | additive; a lock fetched without the switch answers its config as before |
| `platoon prepare` builds sidecars' contexts and pins their registry images (`pulled:<reference>` entries of the image map; `platoon harbor images pin`); the `hpc` extra takes hpc-sandbox 0.5 too | additive; a fetch refuses stacks unless `refuse: {sidecars: false}`, until a release requires hpc-sandbox 0.5 |
| Main-only compose files run; a task with `compose.yaml`, `compose.yml` or `docker-compose.yml` runs as one container, as in Harbor, which reads none of them | behavior change, intended |
| `EnvironmentService.open` takes `needs`; `Sandbox.routes`, `Sandbox.kill_processes`, `SandboxSpec.upstreams`, `SandboxProvider.reap` | services and providers outside platoon must accept the new keyword and fields (a public protocol change) |
| Commands `platoon harbor fetch\|verify\|images\|check\|ls`, `platoon toolchain build\|ls` | new |
| Task ids of fetched datasets come from `registry.json`, else `task-index.json`, else the task's path relative to the dataset, as `dataset_dir` names them; local directories keep directory names unless they ship a `registry.json` | existing configs unchanged |
| Tasks platoon refused (separate verifiers, trivial compose files) now run; `.sif` and `.sqsh` paths no longer get `docker://` | behavior change, intended |
| The loaders' absolute-path task ids (`platoon.envs.harbor.loaders`) | deprecated in favor of locked ids; kept through 1.x |
| Trajectories record the task's content digest, its image key, the grading mode (shared or separate) and how processes were killed before grading (`kill_processes` or none) in `misc` | additive |

No public name is removed.

## The hard parts

1. **Containers in a pod without instances.**
   - Singularity's per-user instance records and cgroup setup do not work inside a rootless user namespace, so each
     service is a long-lived `exec` under an anchor that owns its PID namespace.
   - Teardown relies on PID-namespace semantics: when the anchor's init dies, everything in its namespace dies.
   - Exec goes through `nsenter`, which must reproduce the payload's environment and working directory.
   - Inside the pod, processes are uid 0 mapped to the caller, so `chown` to other uids fails, unless the service
     runs with the caller's subordinate ids (its `subids` run mode, where the caller has them), as a sidecar does that
     switches users. apt drops privileges to `_apt`, which hpc-sandbox's drop-in keeps from failing.
2. **Images at scale.**
   - Each sandbox extracts its root filesystem from the image, and separate verifiers double the number of
     sandboxes. Large images make start time the bottleneck. A per-node template plus copy-on-write copies
     (`cp --reflink`) is the remedy where the filesystem supports it.
   - Base images come from public registries with pull limits.
   - Extracted OCI images lose their environment unless the map records it, which is why the map carries `env`.
3. **Graders that write 0 on an infrastructure failure.**
   - Contracts, opt-in classifiers, sidecar probes and overlays cover the cases that can be seen.
   - Separate verifiers stop agents from tampering with grading.
   - Per-source oracle and no-op controls before training catch the rest: the oracle must score 1 and a no-op 0.
4. **A harness that shares the policy's sandbox.**
   - The policy can kill its harness, exhaust memory, or start processes that outlive it.
   - Markers, `kill_processes`, memory allowances and `grade_if_gateway_clean` bound the damage.
   - A determined policy can still evade the marker kill by clearing its environment. That is why `kill_processes`
     exists, and why separate verifiers are preferred.
5. **Ports and names under host networking.** These are the fallback's hard parts, and its rules answer them:
   - fixed ports collide, so ports are allocated per rollout, reserved by binding `127.0.0.1:0` until the service
     starts;
   - programs choose their own ports, so a service needs a listen rule (the env var that sets its port, else an
     overlay entry) or it is refused;
   - names do not resolve, so `scheme://service:port` is rewritten in env values, `[verifier].env`, healthchecks and
     MCP URLs, including a service's references to its own loopback ports;
   - literals in task files cannot be rewritten, so the index-time scan refuses a task that names a service outside
     those places unless the overlay patches the file;
   - sidecars are reachable by every process on the node, so one the overlay does not authenticate (a per-stack
     token and a `127.0.0.1` bind) or bind to a node service is refused.
6. **Privileged judges without root.** go-judge and other sidecars that need privileges or devices are refused on
   rootless providers unless an overlay binds them to node services. On rlimits instead of cgroups, time and memory
   limits are enforced less exactly.

## Alternatives

- **Run Harbor's trial runner for training.** Launch `harbor run`, or its trial API, per rollout with platoon's
  gateway as the model endpoint.
  - It gives Harbor's compose, installed agents and verifier semantics, with nothing reimplemented.
  - But Harbor has no `invalid` outcome, grades every timeout, has no snapshots or `copy` forks, and its stock
    Singularity environment needs fakeroot and a writable tmpfs, which setuid sites cap or refuse.
  - It also moves harness choice, episode programs and per-agent credit out of platoon.
  - It stays an option for evaluation next to `platoon eval`.
- **Import Harbor as a library in workers.** Rejected: its dependency closure and API churn would enter every
  rollout process. The CLI at fetch time is the only Harbor platoon runs.
- **Host-network stacks as the default on HPC** (an earlier draft of this RFC). Kept as the fallback. Pods give
  fixed ports, real names and isolation between stacks without rewriting, and the checks above show they work
  rootless on a setuid site.
- **Definition-file builds only.** Rejected as the default: a translation covers a subset of Dockerfile
  instructions. BuildKit runs the Dockerfile as written, and the extraction step makes its output usable.
- **Remote servers that run whole episodes and return a reward.** Rejected: harness choice leaves the config, and
  per-agent grading, forks and verifier agents become impossible. Moving sandboxes, not episodes, to other nodes is
  the subject of a separate RFC.
- **Stay ORS-only.** Harnesses would act only through env tools, and every multi-container benchmark would need a
  hand-built server. It remains the path for work that must match an ORS setup.

## Rollout plan

Effort is EST, for one implementer plus review.

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M0 Fail closed | Compose tasks refused (merged). Accept trivial compose files. Parse separate-verifier tables and `artifacts`. Refuse `gpus > 0` without a `gpu` provider, and `[agent].user` unless `agent_user: ignore`. No `docker://` prefix on `.sif`, `.sqsh` and absolute paths. Recursive scans. 0.5 day | CPU tests: each refusal names its reason; the 8 toy tasks still score oracle 1.0 and no-op 0.0 | compose refusal: merged; the rest: the Harbor fast-path pull request |
| M1 Sources and lock | `hf://`, `harbor://` through the pinned CLI, `git+https://`, local; the cache; the lock with ids, index and flags; `fetch`, `verify`, `ls`; auto-fetch of pinned references; lock-based lookups. 3 days | CPU tests: a local Hub fixture, a JSON-registry fixture and a git fixture resolve, lock and verify offline; a one-byte change fails `verify`; platoon's hash equals `uvx harbor==0.24.0`'s on fixtures with a `README.md`, a `.gitignore` and `__pycache__` || the Harbor fast-path pull request: `hf://` and local sources, the cache, the lock, `fetch`, `verify`, auto-fetch, lock-based lookups; the sources pull request: `harbor://` (the Hub and the legacy registry), `git+https://` and adapters, checked offline against a fake Harbor CLI and local git repositories, and online on the Hub's and harbor-datasets' datasets and Frontier-CS's two adapters (`ls` not built) |
| M2 Images and toolchains | Build plans and deferred steps; the image map; `oci`, `def` and `docker` builders with receipts; named toolchains, recipes, per-node staging, binds at the build path; `[resources]`. 4 days | CPU tests on plans and deferred steps; a site job builds a fixture Dockerfile through `oci` and `def`, and a sandbox from each runs with the recorded env and workdir; a toolchain mounts read-only and its interpreter runs || the Harbor fast-path pull request |
| M3 Gateway routes | The gateway's Unix socket; `SessionNeeds`; `upstreams` and `routes` on every built-in provider; the forwarder; route-bound keys; mint and call caps. 2.5 days | CPU tests (process, and docker in CI): a harness in a no-network sandbox finishes an episode through the route; the host's loopback is unreachable from it; a key presented on another rollout's socket is refused || the Harbor fast-path pull request |
| M4 Grading | Separate verifiers with artifacts; reward contracts; opt-in classifiers; `on_harness_exit`; the task's agent timeout; `Sandbox.kill_processes` before grading. 3.5 days | CPU fixtures: separate-verifier tasks score oracle 1.0 and no-op 0.0; a background process cannot shadow `reward.txt`; a crash after a gateway error is `error`, a crash without one is graded || the Harbor fast-path pull request |
| M5 An offline training pool | An offline, separate-verifier pool through M0–M4 on Singularity, with a CLI harness in the task sandbox. 2 days plus the run | oracle at least 0.98 and no-op 0 on at least 100 tasks per source through the training path; `error` plus `invalid` under 5% with a base model; 5 training steps with no leaked sandboxes | |
| M6 Stacks on Docker | The compose parser and subset; the `docker compose` provider; sidecar probes; task MCP servers through the tool plane. 3.5 days | Harbor's `hello-mcp` oracle scores 1.0; a killed sidecar gives `invalid`, not 0; nothing is left after teardown | the compose parser and subset, with main-only compose files: the compose-subset pull request; the stack layer, sidecar probes and stacks on `process` (on CPU: a toy app with a key-value sidecar scores its oracle 1.0 and its no-op 0.0, a killed sidecar gives `invalid`, a sidecar that cannot start an `error`, nothing is left after 200 stacks, some cancelled while they start): the stack-layer pull request; stacks on Docker with `docker compose`, sidecars' images built by `platoon prepare`, checked by GitHub CI's `docker-stacks` job, which fails unless `hello-mcp`'s oracle scores 1.0 and its no-op 0.0 through `platoon-oracle` and `platoon-noop`, a killed sidecar gives `invalid`, and no container, network, volume or project directory is left after teardown or after a reap of a SIGKILLed worker's label: the Docker-stacks pull request |
| M7 Pods | hpc-sandbox pods (holder, anchors, exec services, relays, `/etc/hosts`, egress per phase, janitor); stacks on Singularity and Apptainer; enroot pods at a Slurm site. 6 days, plus site time | a two-service toy stack and `hello-mcp` score their oracles; name resolution, the relay and teardown pass; one stack cannot reach another's services; nothing leaks after 200 stacks; setup p50 and p99 at 16 and 64 stacks per node | the pods pull request, on SingularityCE (the stack fixtures, a Redis and a PostgreSQL sidecar that switch users, and hello-mcp score their oracles and no-ops; a kumo task's `main` reaches its judge sidecar; nothing leaks after 200 stacks; setup timed at 16 and 64 stacks per node); Apptainer and enroot pods experimental, until a site runs its integration tests |
| M8 Host-network fallback | `HostNetworkStackProvider` for sites without unprivileged user namespaces: port allocation, rewriting, listen rules, the literal scan, refusals. 4 days | CPU tests over `process` cover a healthcheck that names its own loopback port; with user namespaces disabled, a two-service toy stack scores its oracle, an unauthenticated sidecar is refused, and nothing leaks after 200 stacks | |
| M9 Frontier-CS | Adapter sources at the pinned commit; the overlay (verifier and judge patches, host-data pins, file-staging mounts); the shared judge-pool binding for the algorithmic track; the image map. 6 days plus compute | fetching both adapters yields the ORS server's 188 + 13 task ids; reference solutions and fixed submissions score as on the ORS server; a judge killed mid-grade gives `invalid`; a 4B run on the 2.0 tasks plus 40 algorithmic tasks takes 20 steps with no leaked sandboxes and `error` plus `invalid` under 5% | |

**Ordering:** M0 through M4 land first and in parallel, and M5 validates them together. M6 and M7 follow, with M8
only for sites that need it.

## Open questions

1. **Grade after a harness exits?** Should Harbor environments default to `{timeout: grade, crash:
   grade_if_gateway_clean}`, which matches Harbor and graders' best-submission fallbacks? Or keep `exclude` for
   both? Decide before M4.
2. **Training network policy.** Should `public` tasks run without network by default in training, so the policy
   cannot look answers up, at the cost of tasks that need the network? Decide before M5.
3. **Pods for single-container tasks.** Should they also run in pods once M7 passes, for host-loopback isolation and
   public egress without the host network? Decided with M7: no. A single sandbox keeps its own network namespace (or
   none), its routes and its snapshots, which pods lack; only stacks run as pods.
4. **Tool names.** Should a Harbor session's env tools (`bash`, `view`, `write`, `edit`) take the ORS names
   (`create_file`, `str_replace`), so a policy that uses env tools sees one surface? Decide before tool-plane
   training on Harbor tasks.
5. **Forks on stacks.** Should `copy` stay refused until stack snapshots exist, or be allowed with fresh sidecars?
   Decide before recursive programs run on stacks.
6. **Classifier sets.** Which named classifier sets should platoon ship, and who maintains them? Decide before M4.
7. **What sites allow.** Unprivileged user namespaces, slirp4netns, BuildKit, cgroup delegation and reflink-capable
   scratch differ per site. Answered with M7: `hpc-sandbox doctor` reports the runtime, the scratch and what pods
   need (egress, subordinate ids), and the providers check what a run needs before it starts (a node that nests user
   namespaces, hpc-sandbox's pods, the layouts and options a run asks for), refusing it with what is missing.
