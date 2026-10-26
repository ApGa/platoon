# Sandboxes and Harbor tasks

Harbor tasks are directories that describe a container, an instruction and a test script. platoon runs each
rollout of such a task in its own sandbox, runs the harness inside it, and grades the sandbox's final state with
the task's own tests. This guide shows how to choose a sandbox provider, point a config at a Harbor dataset, and
what platoon refuses to run. [How Harbor tasks are graded](harbor-grading.md) explains what happens between the
harness's last step and the reward that trains.

**Goal:** evaluate (or train) a policy on a directory of Harbor tasks with the harness running inside each task's
container.

**Prerequisites:**

- platoon installed with the `tinker` extra; add the `hpc` extra for the Singularity/Apptainer and enroot
  providers ([The hpc extra](../getting-started/install.md#the-hpc-extra)); in a checkout, name your other extras
  too (`uv sync --extra tinker --extra ors --extra hpc`; see [Install](../getting-started/install.md#extras)).
- One sandbox runtime: Docker (or Podman), SingularityCE or Apptainer, or enroot.
- A Tinker-compatible training service: hosted Tinker, or a self-hosted server such as
  [miles-tinker](https://github.com/neulab/miles-tinker) ([Train and evaluate](train-and-eval.md#the-training-service)).

To train on a dataset right away, the README's [Train on a Harbor dataset](../../README.md#train-on-a-harbor-dataset)
has the four steps, and [`examples/harbor-dataset`](../../examples/harbor-dataset/README.md) is a project to copy.
[`examples/harbor-local`](../../examples/harbor-local/README.md) evaluates a directory of tasks, as the steps below do.

## Sandbox providers

A provider starts, runs commands in, copies files to and from, snapshots and stops sandboxes. Name one with
`provider:` and pass its options in `provider_options:`, in an entry's `service_kwargs`, or once for a whole site in
its site profile, which every Harbor entry that names no provider runs with ([Write the site's
profile](clusters.md#4-write-the-sites-profile-once)). The profile names only a provider that isolates its sandboxes,
never `process`, and only the options that do not weaken that isolation.

| Provider | Isolation | Images | Needs | Options |
|---|---|---|---|---|
| `process` | None: a host directory, commands run as you | ignored | nothing | `root` (where sandbox directories go; default `<tmp>/platoon-sandboxes`) |
| `docker` | One long-lived container per sandbox, and a `docker compose` project per multi-container task ([Stacks on Docker](#stacks-on-docker)) | `docker://repo:tag` or a plain reference | the `docker` CLI (`binary: podman` for Podman), with its compose plugin for stacks | `binary`, `cli_timeout_s` (600; bounds `docker run` including the pull), `stack_root` (where stacks' project directories go; default `platoon-docker-stacks-<uid>` in node scratch) |
| `singularity` | Rootless SingularityCE or Apptainer instances: private PID and IPC namespaces, root filesystem, `/tmp` and environment | `docker://` references, also pinned by digest (`docker://repo@sha256:<digest>`), or local `.sif` files | the `hpc` extra | passed to hpc-sandbox: `scratch_root`, `image_cache`, `allow_network_none`, `gpu`, `rootfs_layout` ([below](#the-singularity-providers-root-filesystem)), `fakeroot` (since 0.3.0, with hpc-sandbox 0.4: [root and other users](#users-in-task-sandboxes)), ... |
| `enroot` | Rootless enroot containers in their own user, mount, PID and IPC namespaces | `docker://` references, also pinned by digest, or local `.sqsh` files | the `hpc` extra | passed to hpc-sandbox: `scratch_root`, `image_cache`, `gpus`, `remap_root`, ...; the singularity names also work (below) |

Both hpc-sandbox providers also run multi-container tasks, as rootless pods, with hpc-sandbox 0.5 ([Stacks as
pods](#stacks-as-pods-on-singularity-apptainer-and-enroot)).

Every provider runs a string command with `bash -c`, so a Harbor task's `bash` tool takes bash syntax (`[[ ]]`,
`source`, `set -o pipefail`) on all of them, and an image without bash runs only argv commands. A command that
times out is killed with what it started: `process` and the hpc-sandbox providers kill its process tree, and
`docker`, where killing the `docker exec` client leaves the command running, kills every process in the container
that carries the command's marker (`PLATOON_EXEC` in its environment).

Use `process` only for tests and for checking a task with a policy you trust: the agent's shell commands run on
your machine, and the image, network, resource limits and read-only mounts are not enforced. It maps the
sandbox paths `/app`, `/tests`, `/logs`, `/solution`, `/tmp`, the workdir and mount targets to directories below
its root, and translates them in commands, working directories, environment values and file transfers, but not
inside files. Scripts that must also run under `process` write `${PLATOON_SANDBOX_ROOT:-}/logs/verifier/...`;
real sandboxes leave that variable unset.

A `provider_options` block written for a Singularity site also serves an enroot site: the enroot provider takes
every singularity name with the same meaning (`gpu: nv` becomes `gpus: all`; `allow_network_none: false` turns off
`network: none`, which enroot otherwise always offers; `bind_resolv_conf` works on both; `stop_grace_s` is ignored,
as enroot sandboxes stop at once, and so is `rootfs_layout`, [below](#the-singularity-providers-root-filesystem))
and refuses what enroot cannot do (`gpu: rocm`, `cgroup_limits: true`, `fakeroot: true`). Two
options are exceptions: `binary` and `extra_start_args` are each runtime's own executable and container start
flags, passed through untranslated, so set them per site and leave them out of a block both providers read. The
enroot provider also takes `root` as another name for `remap_root` (uid 0 inside the container, your uid outside;
the default), but the singularity provider refuses `root`.

Both providers keep the host's environment out of their sandboxes, `TERM` and proxy variables included. If
commands in a sandbox need a proxy, set its variables in the Harbor service's `env`
([step 2](#2-point-a-config-at-the-dataset)).

### The singularity provider's root filesystem

`rootfs_layout` picks how each sandbox gets its root filesystem, and `HPC_SANDBOX_ROOTFS_LAYOUT` in the job's
environment sets it for every provider that does not. hpc-sandbox has the layouts from 0.3 on, and the `hpc` extra
installs 0.5.1 ([The hpc extra](../getting-started/install.md#the-hpc-extra)); with an older one, every sandbox gets a
copy, and a run that sets `rootfs_layout`, even to `copy`, is refused when its tasks are listed. The layouts:

| `rootfs_layout` | A sandbox's root filesystem | Start, alone / 64 at once on one node | Needs |
|---|---|---|---|
| `copy` (default) | A private extraction of the image (`build --sandbox`) | 2 s / 24-30 s, bound by the node's filesystem metadata (33k files for a 260 MB image) | nothing more |
| `fuse-overlay` | A writable layer of its own over one extraction of the image per node, mounted with `fuse-overlayfs`; the instance runs unprivileged (`--userns`) | 0.5 s / 1-1.5 s; 256 at once 1.4 s on 192 CPUs | `fuse-overlayfs` and `fusermount3` on `PATH`, and a Singularity that runs unprivileged instances (`allow user ns`) |

```yaml
      provider: singularity
      provider_options: {allow_network_none: true, rootfs_layout: fuse-overlay}
```

What the sandbox sees is the same: its files behave as on a copy, deletions, renames and replaced image directories
included. File operations go through FUSE, so metadata-heavy work is slower (`find /` over a 30k-file image 370 ms
instead of 26 ms; 2000 small writes 195 ms instead of 25 ms), while Python imports and large reads are not. A copy
fork (`agents.fork: copy`) snapshots only what the sandbox wrote and restores it over the same extraction: 0.8 s,
2.5 s with 64 at once, instead of 2 s and 59 s. A separate verifier's sandbox is a fresh start like any other, a new
empty layer over the image. GPUs, cgroup limits and `fakeroot` need the copy layout, and the fuse-overlay layout
needs the node's extraction of the image (hpc-sandbox's `rootfs_templates`, on by default for it): an explicit
`rootfs_layout: fuse-overlay` with `gpu`, `cgroup_limits`, `fakeroot` (or `--fakeroot` in `extra_start_args`) or
`rootfs_templates: false` is refused, and the job-wide default leaves such a provider on copies. The enroot provider
ignores `rootfs_layout`.

The copy layout and the enroot provider extract a root filesystem at each sandbox's start, so many starts at once on
a node, as at a run's start, contend for its filesystem. `train.max_starting_rollouts` bounds how many rollouts start
up at once, their sandboxes and harnesses included ([Rollout
start-ups](train-and-eval.md#rollout-start-ups-experimental)); a separate verifier's sandbox and a copy fork start
after their rollout's first model call, outside it.

## Steps

### 1. Lay out the tasks

A dataset is a directory of task directories ([at any depth](#task-ids)). platoon reads these parts of each task:

```text
tasks/write-greeting/
├── task.toml          # [environment] docker_image, workdir, env, cpus, memory_mb, gpus, network_mode, healthcheck
│                      # [agent] timeout_sec, network_mode, user; [verifier] timeout_sec, env, environment_mode,
│                      # network_mode, user; [verifier.environment]; artifacts
├── instruction.md     # the instruction the agent gets (leading canary comments are dropped)
├── tests/test.sh      # the verifier, uploaded only at grading time
└── solution/solve.sh  # optional reference solution, never shown to the agent
```

```toml
# tasks/write-greeting/task.toml
version = "1.0"

[task]
name = "examples/write-greeting"

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 60.0

[environment]
docker_image = "python:3.12-slim"
workdir = "/app"
cpus = 1
memory_mb = 1024
```

The verifier writes the outcome into `/logs/verifier/`: `reward.txt` with one number, or `reward.json` with named
numbers. Which of them trains is the [reward contract](#rewards-the-reward-contract-and-log-classifiers); by
default it is `reward`. A missing, empty or malformed reward file is an infrastructure error, never a reward of 0.
Write them as regular files: grading never reads a reward file that is a symlink (see
[how grading keeps the agent out](harbor-grading.md#how-grading-keeps-the-agent-out)).

#### A separate verifier

A task can grade in a sandbox of its own instead of the agent's, as Harbor 0.24 does. It says so with
`[verifier] environment_mode = "separate"`, or with a `[verifier.environment]` table, and it names the files the
verifier receives in `artifacts`:

```toml
# tasks/answer-the-question/task.toml
artifacts = ["/workdir/answer.txt"]   # also {source = "/workdir/out", exclude = ["*.pyc"]}

[verifier]
environment_mode = "separate"
timeout_sec = 120.0

[environment]
docker_image = "python:3.12-slim"
workdir = "/workdir"
network_mode = "no-network"
```

The verifier's sandbox takes the settings of `[verifier.environment]`, else a copy of `[environment]`
(`[verifier] network_mode` overrides its network). Its image is, in Harbor's order: `[verifier.environment]
docker_image`, an image built from `tests/Dockerfile`, else the task's own image. The first two hold their own
`/tests`, so the task's `tests/` is not uploaded to them. An image built from `tests/Dockerfile` needs a site image
map; until the service has one, such tasks are refused when their session opens.

#### Task ids

A dataset directory may hold its tasks at any depth: platoon looks for `task.toml` in every
directory below it (not in hidden directories, `__pycache__` or a task's own subdirectories). A task directly in
the dataset directory is named by its directory (`write-greeting`), one deeper by its path relative to the dataset
directory (`source-a/write-greeting`). A dataset with a Harbor `registry.json` at its root is the list of tasks the
registry names, with the registry's names as ids, so a partial download of a large dataset holds the tasks that
were downloaded; a `task-index.json` next to it (`{id: {"source_dataset": ...}}`) gives each task the row that
[reward checks](#rewards-the-reward-contract-and-log-classifiers) read as `task.index`. The id is what split
manifests and `platoon rollout --task` use, so no two tasks may share one, even when platoon refuses one of them;
two directories of a `task_dirs` list named `write-greeting` fail with `Two Harbor tasks are named
'write-greeting'`.

### 2. Point a config at the dataset

```yaml
environments:
  - name: harbor-local
    service: harbor
    service_kwargs:
      dataset_dir: ./tasks
    harness: platoon-bash
```

`service_kwargs` are the options of the Harbor service (`platoon/envs/harbor/service.py`). The ones that name files
platoon reads, `dataset`, `registry`, `adapter.repo`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`,
`overlays` and `mounts[].source`, are relative to the config file that names them, so `./tasks` is the `tasks`
directory beside the config, wherever platoon runs from ([Write and override
configs](configs.md#2-build-on-it-with-extends)):

| Option | Default | Meaning |
|---|---|---|
| `dataset_dir` or `task_dirs` | | A directory of tasks, or an explicit list of task directories (exactly one, unless a fetched dataset is named instead) |
| `dataset`, `lock` | none | A dataset fetched by name and pinned in a lock ([Fetch a dataset by name](#fetch-a-dataset-by-name)), in the Harbor cache, `harbor` in the site profile's cache; a fetched dataset's image map is `images/<provider>.json` there ([Build task images](#build-task-images)) |
| `splits` | every split holds every task | A mapping, or a JSON or TOML file (relative to `dataset_dir`), from split names to task names |
| `holdout` | none | `{fraction, group_by, stratify_by, seed, ignore_inputs}`: hold out whole groups of the tasks as the `eval` split, the rest being `train`, instead of `splits`; tasks that share an input file are always in one group ([Hold out whole groups for evaluation](#hold-out-whole-groups-for-evaluation)) |
| `provider`, `provider_options` | the site profile's, else `docker` | The sandbox provider and its options; an entry without a provider lays its `provider_options` over the profile's |
| `default_image` | none | The image for tasks without `docker_image`, used as written (a task's own `docker_image` gets a `docker://` prefix unless it names a scheme or is a `.sif` or `.sqsh` file or an absolute path) |
| `workdir` | `/app` | The working directory for tasks that do not set one |
| `env`, `mounts` | none | Extra environment variables, and `{source, target, read_only}` host mounts, for every sandbox |
| `tool_timeout_s` | 300 | The limit of one tool call (`bash`, `view`, `write`, `edit`) for harnesses that use the env's tools |
| `verifier_timeout_multiplier` | 1.0 | Scales each task's `[verifier] timeout_sec` (default 600) |
| `agent_timeout_multiplier` | 1.0 | Scales each task's `[agent] timeout_sec`, which bounds the harness ([below](#harness-timeouts-and-crashes)) and, without a rollout `timeout`, a rollout's deadline ([Run it](#4-run-it)) |
| `reward` | `{primary: reward}` | The [reward contract](#rewards-the-reward-contract-and-log-classifiers): `{primary, components, range, check}`, a reward name, or a list of the names that may train |
| `verifier_classifiers` | none | Verifier-log patterns that make an outcome `invalid`: a named set (`harbor-common`), `{pattern, kind}` tables, or a list of both |
| `agent_user` | `refuse` | A task that sets `[agent].user` or `[verifier].user`: `refuse` it, `ignore` the user and run as the sandbox's own user (recorded in the outcome's `ignored_users`), or, since 0.5.0, `honor` it, running each phase as its user as Harbor does ([Users in task sandboxes](#users-in-task-sandboxes)) |
| `verifier_user` | as `agent_user` | *Since 0.3.0.* What `[verifier].user` gets instead, when set: `refuse`, `ignore`, or `honor`, which runs the tests as that user, root among them ([Verifiers that run as root](#verifiers-that-run-as-root)) |
| `verifier_provider_options` | none: `provider_options` | *Since 0.3.0.* The provider options of every separate verifier's sandbox, laid over `provider_options`, such as `{rootfs_layout: copy, fakeroot: true}` for verifiers that run as root |
| `concurrent_tools` | none | Tools that may overlap other calls on the same sandbox (`bash`, `view`) |
| `queue_timeout_s` | 240 | How long a call or a grade waits for its turn before it is refused (`env_busy`) |
| `close_timeout_s` | 30 | How long closing waits for calls in flight before it stops the sandbox |
| `max_artifact_bytes` | 1 GiB | The largest artifact a separate verifier receives; a larger one is left out, and the tests decide what its absence means |
| `sandbox_start_budget_s` | 600 | What a rollout's deadline allows each sandbox start, the task's and a separate verifier's ([Run it](#4-run-it)) |
| `stacks`, `network_policy`, `on_sidecar_down` | `auto`, `task`, `invalid` | *Since 0.5.0.* Whether multi-container tasks run, the network of every sandbox, and what a sidecar that is down at grading makes the outcome ([Multi-container tasks](#multi-container-tasks)) |

Values like `${HF_TOKEN}` or `${REGION:-us}` in a task's `env` tables are read from the host: `[environment] env`
when a session opens (an unset variable without a default fails the session as a configuration error), and
`[verifier] env` when it is graded (an unset variable makes the outcome an error, not a score).

### 3. Pick where the harness runs

`placement` on the environment entry decides where the harness process runs:

| `placement` | Effect |
|---|---|
| `auto` (default) | In the task sandbox when the environment has one (Harbor always does), else on the rollout host |
| `task_sandbox` | Always in the task sandbox; an error for environments without one |
| `local` | On the rollout host |

A CLI harness in the task sandbox reaches the model gateway on the host's loopback when the sandbox shares the
host network: `docker` runs containers with `--network host`, and the other providers share the host's network,
unless the task asks for none. A provider says so with the `host_loopback` capability, which every built-in one
declares; a provider of your own whose sandboxes run elsewhere (on another node, in a cloud) leaves it out, and then
a harness is placed in its sandboxes only through a gateway route, else the rollout is a `harness_setup` error.

A task that sets `network_mode = "no-network"` (or `allow_internet = false`) gets a sandbox without a network on a
provider that enforces it (`docker`, `enroot` unless its `provider_options` set `allow_network_none: false`, and
`singularity` with `allow_network_none: true`; otherwise those two providers refuse such tasks, see
[What platoon refuses](#what-platoon-refuses)), and the harness reaches the gateway through the sandbox's route
instead:

1. Before the sandbox starts, the rollout makes a private directory (mode 0700) in node-local scratch and asks the
   session for the gateway as an upstream: `unix://<directory>/gw.sock`.
2. The provider binds that directory read-only at `/run/platoon/gateway` in the sandbox and starts a small
   forwarder inside it, which relays `127.0.0.1:18080` to the socket. Connecting to a socket writes nothing, so
   code in the sandbox cannot create or fill files in that host directory.
3. The rollout's gateway listens on the socket, next to its loopback port, and takes only that rollout's agent
   keys, on the socket and on the port alike.
4. The runner renders `{base_url}` (and `{mcp_url}`) as `http://127.0.0.1:18080`, so harness specs need no change.

The forwarder is a standard-library Python script. It runs with the harness toolchain's `bin/python3` when the spec
names a toolchain ([below](#harness-toolchains)), else with the image's `python3`. A sandbox with neither does not
start: the session fails to open with a `harness_setup` setup error, and the rollout fails without training. The
policy can kill the forwarder: before each harness launch the runner checks it and starts it again on the same
port. The sandbox still has no network: the socket is its only way out, and it leads only to its own rollout's
gateway.

`placement: local` still runs the harness on the rollout host, where it acts through the env's tools. An
environment service that does not set up routes (one written before them) keeps the earlier rule: platoon does not
start a harness in a sandbox without a network there, and the rollout is an `error` (`harness_setup`) whose
reason suggests `placement: local`.

`builtins` decides whether a CLI harness keeps its own tools: `default` keeps them in a task sandbox and turns them
off elsewhere where the harness's spec can; `keep` and `disable` force either. Harnesses that work well in a
Harbor sandbox are `platoon-bash` (a bash tool and nothing else, needs `python3` in the image),
`mini-swe-agent@2.4.6` (installed with uv on first use), and `openhands`, which runs from a prebuilt toolchain
([OpenHands in task sandboxes](#openhands-in-task-sandboxes)).

### 4. Run it

```bash
cd examples/harbor-local
uv run platoon eval eval.yaml                                                # docker
uv run platoon eval eval.yaml environments.0.service_kwargs.provider=singularity
uv run platoon eval eval.yaml 'environments.0.service_kwargs.provider_options={image_cache: /scratch/images}'
```

Training is the same with a training config: `platoon train` with the same `environments` entry.

Each rollout opens a sandbox from the task's image, runs the harness in it, and then grades what the agents left
there: in the same sandbox, or, for a task with a [separate verifier](#a-separate-verifier), in a fresh sandbox of
the verifier's own, which receives the task's artifacts. [How Harbor tasks are graded](harbor-grading.md) has each
step, and what keeps the agents from forging their reward.

`rollout_config.timeout` bounds the harness. A rollout takes longer: its sandbox starts first, and grading may start
a verifier's sandbox and runs the tests for up to the verifier's timeout. Deadlines on whole rollouts, the alarm of
a rollout worker (`rollout_isolation: process`) and the hard timeout of an evaluation slot, add what each task needs
beyond the harness: `sandbox_start_budget_s` per sandbox start, the healthcheck's longest wait, the verifier's
timeout (times `verifier_timeout_multiplier`), two minutes for grading's own commands, and `close_timeout_s`. So a
slow verifier no longer kills the worker grading it, and the timeout need not leave room for grading. Raise
`sandbox_start_budget_s` where sandboxes start slowly, such as when images are pulled at start.

A Harbor config needs no `timeout` for these deadlines. Without one, they start from the harness's own time limit:
the earliest of its spec's `timeout_s` and the task's `[agent] timeout_sec` times `agent_timeout_multiplier`
([Harness timeouts and crashes](#harness-timeouts-and-crashes)). So a task whose agent may take 30 minutes gets a
deadline of 30 minutes plus what it needs beyond the harness. Set `timeout` only to stop harnesses sooner than their
tasks allow. Where nothing limits the harness (a function harness on a task without `[agent] timeout_sec`), or the
environment runs a rollout function of its own (`rollout`, `eval_rollout`) or runs a function harness with an
episode program of its own (a `program` other than `single` and `gym`, which may run more after the root agent,
such as judges), whose limits platoon cannot know, a worker's deadline is `worker_unbounded_rollout_budget_seconds`
(a day), and an evaluation slot's is 900 s plus what the task needs beyond the harness.

## Prepare a config, once per site

A run never builds what it needs unasked: an image or a toolchain needs network access, a container builder and
minutes, and its build writes directories that every job of a site reads. One command makes all of it, on a node with
network access (a CPU node will do), before a run starts:

```bash
platoon prepare configs/train.yaml             # the run's own config, with the overrides the run gets
platoon prepare configs/train.yaml --dry-run   # what is missing, changing nothing
```

For each entry of the config (a training or evaluation config, or a file of environments that they extend), in order:

1. **The dataset.** A Harbor entry's `dataset` is fetched into its lock, as `platoon harbor fetch` fetches it
   ([Fetch a dataset by name](#fetch-a-dataset-by-name)), and an entry with a hold-out prints its report.
2. **The images** that its tasks, their separate verifiers and their stacks' sidecars build from Dockerfiles, and its
   image map lacks, are built with the builder of its sandbox provider and recorded in its image map ([Build task
   images](#build-task-images)): `oci` for `singularity` (`def` where its runtime is Apptainer), `docker` for
   `docker`. The `process` provider runs no image, so their keys are recorded as `local`. For any other provider,
   which pulls its images from a registry, the step fails and names the command that builds and pushes them. An
   entry that serves a directory builds nothing: its tasks run their own images. Then, on every provider but
   `process`, **the registry images** that its stacks' sidecars run are pinned in the map, by the digest each
   registry serves and with the configuration it gives the image ([Multi-container tasks](#multi-container-tasks)),
   and, where its stacks run as pods with egress, **the pod tools**: slirp4netns, fetched with `hpc-sandbox
   fetch-tools` where no slirp4netns on `PATH` does what pods need. Run prepare on a node like the run's: what it
   finds on `PATH` is its own node's.
3. **The toolchains** its harness specs run from, under the toolchain root ([Harness toolchains](#harness-toolchains)).
   One built from another recipe is left as it is, since a run of another config, or of another platoon, may be
   staging its rollouts from it: the step fails and names the forced build that replaces it, to run while no job uses
   the toolchain.

Then **the policy's tokenizer** (the model and renderer of the config's `train` or `inference` section) is loaded
once, into the Hugging Face cache, for runs on nodes that cannot reach the Hub.

Each step prints a line, and does nothing when what it makes is current, so prepare can run again at any time: after
any change to an entry's dataset, selection, refusals, index, overlays or harness, and right before a launch. Two at
once, also on two nodes of a site, take turns on file locks, and the second finds the first's work current. On a
site whose profile says `cache: /shared/platoon-cache` and `provider: singularity`, it first names the profile on
stderr, then prints:

```text
environments[science] dataset: fetched hf://my-org/science-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e into the lock /shared/platoon-cache/harbor/locks/0b2c41f1d09f6f6e.lock.json, 6,106 task(s), 5,865 runnable
environments[science] hold-out: 5% of each source_dataset, grouped by kaggle_dataset_name, else split_group, joined by shared input files, seed 0
  ...
environments[science] images: built 2 with the oci builder into the image map /shared/platoon-cache/harbor/images/singularity.json
environments[science] toolchain openhands-1.54.0: built at /shared/platoon-cache/toolchains/openhands-1.54.0 for openhands@1.54.0.post1 (sha256:...)
policy tokenizer: loaded Qwen/Qwen3.5-4B through the chat_template renderer, into the Hugging Face cache
ready: 4 made now, 0 already current
```

The exit status is 0 when the config is ready, 1 when a step failed (with `--dry-run`, when anything is missing),
and 78 on a configuration error. Only the config's environments are read, and checked as a run checks them, so a
training config that leaves `log_path` to its launcher prepares as it is. Where each thing goes, which is where runs
find it without naming it:

| What | Where |
|---|---|
| A dataset's tasks and lock | `harbor` in the cache root: the site profile's `cache`, else `~/.cache/platoon` |
| Its images | `images/<provider>.json` in that Harbor cache, with the image files in `images/<provider>/` |
| Toolchains | `toolchains` in the cache root |

The site profile's `cache` is the one place that names them. Platoon 0.3.0 removed an entry's `cache`, `image_map`
and `toolchain_root` and the variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, which named others: a config
that sets one of the options does not load, and a command that loads a config, or takes the cache from the site,
refuses to run while one of the variables is set, each with exit 78 and a message that names the profile's `cache`.

Runs check what prepare makes before any rollout: `platoon train`, `eval` and `rollout` stop with exit 78 and the
`platoon prepare` command for their config, overrides included, when a harness's toolchain is not built, or when a
fetched dataset's tasks build images that their image map lacks, or run registry images in sidecars that it has not
pinned (checked when the tasks are listed, which takes seconds once the lock exists).
A toolchain built from another recipe stops them too, naming the forced build instead. `platoon train` and `eval` still
fetch a dataset pinned to a commit that no lock answers, when they list its tasks; `platoon rollout`, which runs one
task, stops instead.

```text
platoon: what the run needs is not prepared on this site:
  environments[science].service: 5,584 of the 5,584 tasks of split 'train' build their images from Dockerfiles, which no image map holds (the default image map /shared/platoon-cache/harbor/images/singularity.json does not exist): 2 image(s), such as sha256:e879... for 4,439 task(s) (fineenvs--...) and 1 more; a run never builds them
  prepare it with: platoon prepare configs/train.yaml
```

The `platoon harbor` and `platoon toolchain` commands remain for what needs their flags: `--shard`, `--push`,
`--key`, `--builder`, `--repair`, `--update`, `images pin --reference` and `images alias`.

### Prepare as a run starts, on one machine

On a laptop, or on a single node that both builds and runs, a run can prepare first. `--prepare` on `platoon train`,
`eval` or `rollout` runs `platoon prepare` on the run's config and overrides, then the run, once the config is ready:

```bash
platoon train configs/train.yaml --prepare               # prepare, then train
platoon rollout configs/train.yaml --task ID --prepare   # prepares the whole config, then runs the one task
```

- **The same prepare.** It takes the same steps under the same file locks, so two runs with the flag take turns, and
  the second finds the first's work current. Once the config is ready, the flag makes nothing, except that a dataset
  in a local directory is fetched again each time, as prepare fetches it. The lines are prepare's, on stderr, so
  stdout holds what the run prints without the flag, such as `platoon eval`'s JSON summary.
- **What it prepares.** The whole config, whatever task `platoon rollout` runs. The tokenizer is the one the run
  loads: `platoon rollout`'s `--model` and `--renderer` replace the config's. An evaluation with `stage=report`, which
  runs no rollout, prepares nothing.
- **Only a ready config runs.** When a step fails, the run does not start, and the command exits with prepare's
  status: 1, or 78 for a configuration error.

Leave the flag out of jobs on a cluster: prepare once on a CPU node, then start the GPU jobs without it. A GPU job
that prepares holds its GPUs while it fetches and builds, which can take hours for a large dataset, and a second job
that waits for the first one's builds holds its GPUs too. Without the flag, a run that lacks something stops within
seconds and names the `platoon prepare` command.

## Fetch a dataset by name

Instead of a directory, a Harbor environment can name a dataset that platoon fetches once into a cache and pins with
a lock. It comes from one of these sources ([below](#the-harbor-hub-git-repositories-and-adapters) for all but the
first and the last):

| Source | `service_kwargs` | Pinned (fetched by `platoon train` and `platoon eval` themselves) by |
|---|---|---|
| A Hugging Face dataset repository | `dataset: hf://org/name@<commit>[#subdir]` | a commit; a branch moves |
| A package of the Harbor Hub: a dataset, or a single task | `dataset: harbor://org/name@<ref>` | a revision (`@4`) or a digest (`@sha256:...`); a tag (`@latest`, `@v4.0.0`, or none) moves |
| A dataset of Harbor's legacy registry | `dataset: harbor://name@<version>`, and `registry:` a `registry.json` (a file, its directory or a URL; default: Harbor's own) | a version in a registry file, or at a URL with a commit in its path; Harbor's own registry, on a branch, moves, and so does a task at `HEAD` |
| A directory of a git repository | `dataset: git+<url>@<commit>[#<path>]` | a full 40-character commit; a branch or a tag moves |
| An adapter that generates the tasks | `adapter: {repo, commit, path, command, args, env, sparse}`, instead of `dataset` | a full commit |
| A local directory | `dataset: <path>` | (copied again by every fetch, since its files may change) |

The entry's `service_kwargs` say what to fetch:

```yaml
# configs/tasks.yaml: an environment that training and evaluation configs extend
environments:
  - name: data-analysis
    service: harbor
    service_kwargs:
      dataset: hf://my-org/my-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e   # into harbor/ in the site's cache
      select: {source_dataset: [data-analysis]}  # filters on the dataset index's fields
      tasks: [splits/phase0.json]          # optional allowlists of task ids (a JSON list or split manifest)
      overlays: [overlays/install-first]   # optional edits of task files, hashed into the lock
      refuse: {sidecars: false, gpus: true, agent_user: true, privileges: true, per_task_build: true,
               uploads_over_mb: 100}
      index:                               # more index fields: from the dataset's files, and from task.toml
        - {path: data/tasks.jsonl, key: task_id, fields: [split_group]}
        - {metadata: [kaggle_dataset_name]}
    harness: platoon-bash
```

```bash
platoon harbor fetch configs/train.yaml    # resolve, download each Harbor entry's selected tasks, write its lock
platoon harbor verify configs/train.yaml   # every cached task must still hash to its digest (exit 1 if not)
```

Both commands take a training or evaluation config, with `key=value` overrides as `platoon train` takes them: they
act on each Harbor entry that names a `dataset` (`verify` also on one that names a `lock`), whose paths resolve as
when the run lists its tasks. So the lock they write is the one the run reads, by default
`<cache>/locks/<hash of the fetch settings>.lock.json`, and `fetch` prints the report of each entry's hold-out
([below](#one-dataset-one-environment)) before anything starts. `fetch` prints one JSON line per lock, with
`"fetched": false` when it had nothing to do. Fetches of one lock take turns on a file lock in the cache, also on
several nodes that share it, so of two at once the second finds the first's lock current. `verify` also takes a lock
file (with `--cache`).

**Fetch config files**, the input of these commands until the run config's entry said the same, held the keys
above at their top level. Platoon 0.3.0 removed them, with `fetch`'s `--lock` and `--cache`, which applied to them
alone: the commands refuse a fetch config file, and `fetch` the two options, each naming what replaces it (exit 78).
Move the file's keys into a Harbor entry of the run's config, and its `workers` into `fetch --workers`.

A fetch reads the dataset's index (`registry.json`, else `task-index.json`, else every directory with a `task.toml`),
adds the fields of the extra index files `index` names (JSON, or JSONL rows keyed by `key`), and downloads only the
selected tasks: their directories are listed, and each distinct file is downloaded once,
however many tasks share it (common graders, shared input data). Only the files Harbor's content hash covers are
downloaded: a task's `.gitignore` (else Harbor's default ignores, such as `__pycache__/`) leaves files out, as it does
when a local directory is fetched. Every download is checked against the repository's listing (an LFS file's
SHA-256, else its git blob id); a file that does not match is downloaded once more, and the fetch fails, naming it,
if it still does not. Each task lands in `<cache>/tasks/<digest>/`,
named by Harbor's content hash, which platoon computes itself; the Hub keeps no file modes, so `*.sh` files and files
that start with `#!` get their executable bit back. Task ids are the registry's names, else the keys of
`task-index.json`, else each task's path relative to the dataset (its directory name for a task directly in it):
the ids `dataset_dir` gives the same directory. A local directory that is itself one task is named by its
`[task].name`, else its directory name. Paths in `registry.json` and `task-index.json` must stay inside the dataset:
an absolute path or one with `..` fails the fetch. The lock records, for every task, its id, digest, the index's
metadata, the flags platoon reads from `task.toml` (verifier mode, network, GPUs, users, stacks) and the bytes a
session uploads into one of its sandboxes (`upload_bytes`), its image build plan (below), the digest of each of its
input files (`input_files`, which a [hold-out](#shared-input-files) joins groups by), and why platoon refuses it,
if it does: a compose file outside the subset platoon runs (`compose`, [Compose files](#compose-files)), what
`refuse` names (`stack`, `gpus`, `agent_user`, `per_task_build`, `uploads_over_mb`), and anything platoon's task
loader cannot run (`task_config`: a bad `network_mode`, a separate verifier built from `tests/docker-compose.yaml`, an
artifact path with `..` or of a compose service the task does not start, and the rest of [What platoon
refuses](#what-platoon-refuses)), which would fail every rollout. A stack is locked as runnable, and runs where the
provider runs stacks ([Multi-container tasks](#multi-container-tasks)), unless `sidecars: true` refuses it (`stack`,
the default before 0.5.0); with `privileges: false`, so is a stack whose compose services ask for `privileged` or for
a `cap_add` that a rootless service does not hold, which runs on the `docker` provider alone ([Stacks on
Docker](#stacks-on-docker)). A refused task stays in the lock with its reason: it is never listed, and a split
manifest that names it is an error. `refuse`'s switches take `true` or `false`; anything else is an error.

**Tasks that upload too much.** When a session opens, platoon copies a task's inputs into its sandbox: the sources of
its Dockerfile's deferred `COPY` steps ([Build task images](#build-task-images)), or, for a task with a prebuilt
image, its `environment/` directory. A separate verifier's sandbox gets its own copy. `upload_bytes` records the
larger of the two, and `refuse: {uploads_over_mb: 100}` refuses (as `uploads_over_mb`, with the size) a task whose
sessions would upload more than 100 MiB, which every rollout copies anew into each of its sandboxes. Left unset,
nothing is refused, and a lock written before the key existed still answers the config.

An entry may also name a lock alone, which it reads, instead of a dataset to fetch:

```yaml
environments:
  - name: data-analysis
    service: harbor
    service_kwargs:
      lock: locks/my-tasks.lock.json      # relative to this file, as every input path is
      splits: splits/phase0.json
```

| Option | Meaning |
|---|---|
| `dataset` | A source to fetch (`hf://org/name@<commit>`, `harbor://org/name@<ref>`, `harbor://name@<version>`, `git+<url>@<commit>#<path>` or a directory), with `tasks`, `select`, `overlays`, `refuse` and `index` choosing and editing what is fetched |
| `registry` | The legacy registry of a `harbor://name@<version>` dataset: a `registry.json` file, its directory or a URL (default: Harbor's own) |
| `adapter` | An adapter that generates the tasks, instead of a `dataset`: `{repo, commit, path, command, args, env, sparse}` ([above](#the-harbor-hub-git-repositories-and-adapters)) |
| `lock` | The lock to read (and, with `dataset`, to write; default `<Harbor cache>/locks/<hash of the fetch settings>.lock.json`) |

The Harbor cache, which holds the tasks' files, the locks and the image maps, is `harbor` in the site profile's
`cache`, else `~/.cache/platoon/harbor` ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)).

`platoon train` and `platoon eval` fetch a pinned `dataset` (or `adapter`; the table above says what pins each source)
when no lock lists it yet, in the process that lists the tasks; a moving reference (a branch, a tag, or no revision) is
fetched only by `platoon harbor fetch --update`. A local directory is copied into the cache when it is fetched: run
`platoon harbor fetch` again after changing it. For a pinned dataset, `platoon harbor fetch` does nothing when the lock
answers the config and the cache holds a directory for each of its tasks; `--repair` fetches anyway, and so does a
fetch of a lock written before platoon recorded `input_files`, once. A fetch hashes each cache entry it would reuse
again, and replaces one that no longer matches its digest, so `--repair` (and any fetch of a local directory) also
repairs a damaged cache; `platoon harbor verify` finds one without changing it.
Rollout workers never fetch and never scan: they look each task up by id in the lock, and check its cache entry
against its digest once per node (a marker under `$PLATOON_NODE_SCRATCH`, `$PBS_LOCALDIR`, `$SLURM_TMPDIR` or
`$TMPDIR`). A missing lock or a mismatched entry fails the rollout as a configuration error.

**Overlays** edit task files at fetch time, for the tasks they match. One that moves a dependency install in front of
a task-specific copy, so that every task of a source shares one image:

```toml
# overlays/install-first/overlay.toml
description = "Install the grader's dependency before copying the inputs"
[match]                               # index fields (one value or a list) and `name` globs; all must match
source_dataset = "data-analysis"
[[edit]]                              # a regular-expression replacement (Python syntax, multiline mode)
file = "environment/Dockerfile"
pattern = '^(COPY input/ /data/input/\n)(RUN pip install [^\n]*\n)'
replace = '\2\1'
count = 1                             # replacements expected; otherwise the task is refused
[[copy]]                              # a file of the overlay replaces (or adds) a task file
file = "tests/test.sh"
source = "files/test.sh"
```

An overlaid task's digest is that of its overlaid files; the lock keeps the upstream digest too, and a hash of the
overlays.

**Index fields from task metadata.** A dataset may keep a fact about each task only in its `task.toml`, such as
the Kaggle dataset a question's files come from. An `index` entry `{metadata: [field, ...]}` records those fields
of each selected task's `[metadata]` table among its index fields in the lock, where reward checks read them
(`task.index`). They are read after selection, so `select` cannot name them; only strings, numbers and booleans are
recorded, a task without a field records none for it, and a field that the dataset's own index gives the same task
with another value fails the fetch, naming both values. Like every `index` entry, it is one of the lock's inputs:
adding it fetches a new lock.

### The Harbor Hub, git repositories and adapters

A moving reference (a Hub tag, a branch, a registry task at `HEAD`) is fetched only by `platoon harbor fetch CONFIG
--update`, which records what it resolved to (a digest, a commit) in the lock; the runs then read that lock until the
next update. Rollout workers never fetch from any source: they read the lock and the cache. A source downloads or
checks out into a directory of its own in the Harbor cache, removed once its tasks are cached, and the lock records how
it resolved (`source`: the package and its digest, the repositories and their commits, an adapter's command). Its
tasks are cached and locked as any fetched task is, overlays, build plans and the stack plans of their compose files
([Compose files](#compose-files)) included: a main-only task of the Hub runs on every provider, and one with sidecars
runs as a stack where the provider runs stacks ([Multi-container tasks](#multi-container-tasks)), unless the fetch's
`refuse: {sidecars: true}` refuses it.

```yaml
environments:
  - name: terminal-bench
    service: harbor
    service_kwargs:
      dataset: harbor://terminal-bench/terminal-bench@4   # revision 4 of the dataset
      tasks: [splits/cpu-tasks.json]                      # optional: only these are cached and locked
  - name: kumo
    service: harbor
    service_kwargs:
      dataset: harbor://kumo@1.0                          # Harbor's legacy registry, which moves: fetch it with --update
  - name: gaia2
    service: harbor
    service_kwargs:
      dataset: git+https://github.com/laude-institute/harbor-datasets@37db108843a49bb31a592e37a75e2c40dc3f9749#datasets/gaia2-cli
```

**The Harbor Hub.** platoon runs the Harbor CLI (`harbor==0.24.0`, through `uvx` in an environment of its own, with
its dependencies as PyPI had them at its release, `--exclude-newer 2026-10-06T00:00:00Z`, so that every host installs
the same packages; set `PLATOON_HARBOR_CLI` to the command line of another Harbor CLI, such as an installed `harbor`),
never imports Harbor, and needs [uv](https://docs.astral.sh/uv/) on the fetching host; Harbor's telemetry is off
(`HARBOR_TELEMETRY=0`) unless the environment sets it. `harbor version show --json` resolves the reference to the
package's digest, and for a dataset to each task's digest; `harbor download` then downloads that digest. The CLI has no
task filter, so a dataset downloads whole and `tasks` and `select` choose among its tasks afterwards. Each task's files
must hash to the digest the Hub gives it, by Harbor's content hash, which platoon computes itself: a task that does not
fails the fetch. Task ids are the tasks' names, the directories of `harbor download --export`. A step that fails for
a reason that may pass (a reset stream, a timeout, a server error) is tried up to four times; one that the CLI reports
as a package or a version that is not found, or not yours to read, is not tried again.

**Harbor's legacy registry.** platoon reads the `registry.json` itself, and takes each selected task from its
`git_url` at its `git_commit_id` and `path`, through a sparse clone that downloads only the selected tasks' files,
rather than a clone of each whole repository; a task in a Hugging Face dataset repository comes through `hf://`'s
downloads, which need no git-lfs. Task ids are the registry's names. Of two entries with one name and version, the last
counts, as in Harbor. A task without a `git_url` is a directory next to a registry file, which a fetch copies again
every time, as it copies a local directory, so `platoon harbor fetch` takes its changes.

**A git repository.** The fetch takes the commit's tree without its files (a shallow, blobless fetch), finds the tasks
under `path` as it finds them in a directory (its `registry.json`, else `task-index.json`, else every directory that
holds a `task.toml`), and checks out only the selected tasks. A commit must be named in full: git cannot fetch an
abbreviated one. Files kept in Git LFS need `git lfs` on the fetching host; without it, a task that has one fails the
fetch, naming the file.

**An adapter** generates the tasks from an upstream repository at a pinned commit, as Harbor's adapters and
Frontier-CS's do. Frontier-CS's adapters, written for Harbor's tree, do not run as projects of their own in
Frontier-CS's repository, so this entry names the command Frontier-CS runs them with, a module on `PYTHONPATH`:

```yaml
environments:
  - name: frontier-cs
    service: harbor
    service_kwargs:
      adapter:
        repo: https://github.com/FrontierCS/Frontier-CS.git
        commit: ede1b44d2fe22a65320c5e36b1014b3defc03671
        path: adapters/frontier-cs-algorithm   # where the command runs
        sparse: [/adapters/, /algorithmic/, "!/algorithmic/problems/*/testdata/"]   # what to check out
        command: [uv, run, --no-project, --python, "3.12", --with, pyyaml==6.0.2, python, -m, frontier_cs_algorithm.main]
        env: {PYTHONPATH: src}
        args: [--source, "{checkout}"]         # --output-dir {output} is added
```

The fetch checks the repository out at the commit (whole, or what the `sparse` patterns name: git's sparse-checkout
patterns, where `!` excludes), runs `command` with `args` in the directory `path`, with `env` added to the fetching
process's environment, and fetches the directory it wrote as a local directory. In `command`, `args` and `env`,
`{checkout}` is the checkout and `{output}` the directory the tasks go to; `--output-dir {output}` is added when none of
them names `{output}`, since Harbor's adapters all take it. Without a `command`, the adapter runs as its own project
runs it: `uv run --project . <script>`, with the one script of its `pyproject.toml`, and `--locked` when it has a
`uv.lock`, whose digest the lock records. A `repo` may be a local repository, relative to the config file like every
input path. An adapter runs the repository's code, with the network and the credentials of the host that fetches: name
a commit you have read. The repository, commit, command, arguments, environment and patterns are the lock's inputs.

**Links and credentials.** What the Hub, git or a registry brings must stay inside each task. A symbolic link in a
task is taken as the file it names when that file is in the same task, as Harbor's content hash takes it; a link that
leaves its task (an absolute one, one that climbs out, a `solution/` or `tests/` directory that links elsewhere, or a
task that is itself a link or lies behind one) fails the fetch, naming the task and the path, since it would bring
files of the fetching host into the cache and into every sandbox. It fails the fetch also where it names a directory or
nothing that was downloaded or checked out (a file of the repository outside the selected tasks), which the hash would
skip, so that no task loses a file unnoticed, such as a compose file that links out of its task. A link that the
task's `.gitignore` (without one, Harbor's default ignores) leaves out of the hash is left out of the cache instead of
refused, wherever it leads, but for the `.gitignore` itself and the task's directories (`environment/`, `tests/`,
`solution/`), which are checked whatever it says. The lock and the fetch's log lines record each URL as written, so a
URL that carries credentials (`https://user:token@host/...`) is refused: give private repositories' credentials to git
(a credential helper, `~/.netrc` or an SSH key) instead.

## Build task images

Tasks that build from `environment/Dockerfile` run images built once per site. platoon splits each Dockerfile at its
last step that must run at build time (a `RUN`, a `FROM`, a `COPY --from`, ...):

- the **build prefix** is what gets built. Its **build key** is a SHA-256 over the prefix and the build-context files
  it reads, so tasks whose prefixes match share one image however their last steps differ;
- the **deferred suffix**, the trailing `COPY` and `ADD` of local files, `WORKDIR`, `ENV` and metadata, is applied
  when a session opens: the files are uploaded, the directories created and the variables set, with the result a
  build would have had. A `USER` there is the agents' user where users are honored ([Users in task
  sandboxes](#users-in-task-sandboxes)), and gives the directories that a later `WORKDIR` makes to that user, as
  BuildKit does; a fetch refuses it otherwise (`refuse.agent_user`).

A step that fetches remote content, an `ADD` of a URL or of a git repository, always stays in the build prefix,
wherever it is, also when an `ARG` or `ENV` value holds the URL (`ARG BASE=https://...`, then
`ADD --checksum=sha256:... ${BASE}/file /opt/file`). platoon tells a URL from a build-context path as Docker's
BuildKit does, once the step's variables are expanded, and plans without the network: the build key covers the step
as written, its URL and its `--checksum`. The `oci` and `docker` builders fetch it as a Docker build does, verifying
the checksum and never extracting a remote archive; the `def` builder refuses it. Give each fetched file a
`--checksum`: the key cannot see a file change behind its URL. A `COPY` of a URL is refused, as Docker refuses it.

BuildKit fetches with Go's TLS client, which offers a post-quantum key exchange (ML-KEM) by default. A network path
that drops TLS handshakes offering it fails the build with `TLS handshake timeout`, on a host that `curl` reaches.
Build with `GODEBUG=tlsmlkem=0` in the environment then: `platoon harbor images build` and `platoon prepare` pass it
on to BuildKit.

```bash
platoon harbor images plan configs/train.yaml             # one image per build key, and which the image map lacks
platoon harbor images build configs/train.yaml            # build the missing ones with the provider's builder
platoon harbor images build configs/train.yaml --shard 3/8   # every 8th missing image from the 4th, for job arrays
platoon harbor images pin configs/train.yaml              # pin the registry images that sidecars run
platoon harbor images verify configs/train.yaml
platoon harbor images plan locks/my-tasks.lock.json --image-map images/site.json   # a lock file and a map of yours
```

Each takes a training or evaluation config, with overrides, and acts on each Harbor entry's lock (the one that
answers its `dataset`, or the `lock` it names) with the entry's image map, `images/<provider>.json` in its Harbor
cache. `build` uses the builder of the entry's sandbox provider, `oci` for `singularity` (`def` where its runtime is
Apptainer) and `docker` for `docker` (`--builder` names another), and puts the image files in `images/<provider>/` next
to the entry's map, or, for a lock file, in `images/` next to the map `--image-map` names (`--out` names another
directory). Builds of one map at once, also on several nodes, take turns on each image and build it once. The commands
also take a lock file, with `--image-map` (and, for `build`, `--cache`).

`plan` lists one image per build key, with its task count, base images and whether the map has it, plus the
prebuilt images tasks name in `docker_image`; an entry whose image file is gone, or is not the size its build
recorded (a truncated copy), counts as missing. A stack's sidecar that builds from a context of the task is a key of
its own (role `service`), its whole Dockerfile built, and the registry images that sidecars run are listed apart,
with whether the map pinned each: `pin` pins those it has not (`--reference` pins one again), each under the key
`pulled:<reference>`, as an entry of the `pulled` builder whose image is the reference by its digest
(`docker://redis@sha256:...`), with the configuration its registry gives the image for this host's platform.
`build` builds the missing ones from a representative task and adds each to the image map as soon as it is built, with a
receipt: the builder and its version, the build time, the log, the image's SHA-256, the base images as the Dockerfile
names them (`base`) and the digest each one resolved to (`base_digests`). A build asks each base image's registry for
the digest its tag points to and builds from `name@digest`, so the receipt says exactly what the image was built from;
the build key stays the one the lock recorded. A base whose digest cannot be resolved (no network, a private registry)
is built as written, with a warning, and its digest in the receipt is `null`.

`verify` checks the map against the lock: every image the lock's runnable tasks use must be in the map, and each
local image file (`.sif`, `.sqsh`, ...) must exist and hash to the SHA-256 its entry records. It prints one line per
missing or mismatched image and exits 1 if there is any. Registry references cannot be hashed here; they are listed
as not checked and do not fail the check.

| Builder | How | Where |
|---|---|---|
| `oci` (default) | `singularity build --oci` runs the Dockerfile with rootless BuildKit, as a Docker build does: multi-stage builds, step flags, here-documents, and an `ADD` of a URL (verifying its `--checksum`) or of a git repository; its single squashfs layer is unpacked and rebuilt as a native SIF, and the image config's `ENV`, `WORKDIR`, `USER`, `ENTRYPOINT` and `CMD` go into the map. SingularityCE's BuildKit cannot load the frontend image a `# syntax=` directive names, and builds with its own Dockerfile frontend instead: a directive that names a 1.x release of Docker's (`# syntax=docker/dockerfile:1.7`) is left out, and one that names another frontend (labs, a third party's) is refused | SingularityCE 4.1 or later, `unsquashfs` (from squashfs-tools 4.4 or later to record the image's file owners, which users honored on hpc-sandbox need); the network, for base images and remote `ADD` sources |
| `def` | a translation to a `Bootstrap: docker` definition file built with `--fakeroot`, in which each `RUN` runs in a shell of its own that starts in the current `WORKDIR` and sees the `ARG` values, as in a Docker build; it refuses what it cannot translate faithfully (multi-stage builds, `ARG` without a default, `USER`, `SHELL`, step flags, here-documents, an `ADD` of a URL, a git repository or an archive), naming the `oci` builder. A definition file keeps none of the image's configuration, so the map records its base image's, from the registry, under what the Dockerfile sets (`CMD`, `ENTRYPOINT`, `EXPOSE`, ...) | Singularity or Apptainer with fakeroot; the network, for the base image's configuration |
| `docker` | `docker build`, tagged `platoon-harbor:<key>` for the `docker` provider; with `--push REPO`, pushed and recorded by digest (enroot sites import that digest). The only builder that passes a sidecar's build arguments (compose's `build.args`): the others refuse an image that has any | Docker or Podman (`PLATOON_DOCKER=podman`) |

The image map is a site file; the lock stays the same everywhere. An entry with a fetched dataset (`dataset` or
`lock`) uses its default image map, `images/<provider>.json` in its Harbor cache (with `provider: singularity` and no
site profile, `~/.cache/platoon/harbor/images/singularity.json`), when that file exists; a missing file means no map.
An entry that serves a directory (`dataset_dir`, `task_dirs`) has no image map: its tasks run their `docker_image` or
`default_image`, and a separate verifier built from `tests/Dockerfile` cannot start. To build a directory's images,
name it as a dataset to fetch (`dataset: ./tasks`), whose images `platoon prepare` builds.

```json
{"version": 1, "images": {"sha256:<key>": {"image": "images/<key>.sif", "sha256": "<file digest>", "builder": "oci",
  "env": {"PATH": "/usr/local/bin:/usr/bin:/bin"}, "workdir": "/workdir", "user": null, "receipt": {}}}}
```

When a session opens, its sandbox runs the map's image for the task's key, with the image's environment under the
deferred `ENV` steps, the service's `env` and the task's own, and the deferred `WORKDIR` (unless `task.toml` sets
one). A task that builds from a Dockerfile but whose key the map lacks is refused (`env_config`), never run on
`default_image`. So that no run loses its rollouts to that, `platoon train`, `eval` and `rollout` check the images
of a fetched dataset's tasks when they list them, before any rollout and right after a fetch, and stop with exit 78
and the `platoon prepare` command for their config when the map lacks one (or its image file is gone or
truncated). A dataset directory's tasks are not checked: they plan their images when a session opens. A task with a
`docker_image` runs that image (local `.sif` and `.sqsh` files as written), and gets its `environment/` directory
uploaded into its working directory when it has no Dockerfile, as Harbor does. For CPU
tests on the `process` provider, which runs no image, point the keys at `local`:
`platoon harbor images alias LOCK --image-map MAP --image local`.

A separate verifier gets its image the same way. One built from `tests/Dockerfile` runs the map's image for that
Dockerfile's key, with its own deferred steps (`platoon harbor images build` builds it too); one that runs the task's
own image gets the task's image and deferred steps, as Harbor's verifier would run the task's built image.

So does a stack's sidecar that builds from a context of the task (`build:` in its compose file,
[Multi-container tasks](#multi-container-tasks)): its Dockerfile is planned whole, deferring nothing, since a service's
command starts with its container, so its build key covers every file it copies, and the image is built with its
`build.args` (`--build-arg`, which only the `docker` builder passes yet: the others refuse such an image). `plan`
lists it with the role `service` (and, with `--json-output`, its context, such as `environment/mcp-server/Dockerfile`),
and one image serves every sidecar, of every task, whose key is the same.

## Compose files

*Since 0.5.0.* A Harbor task may ship `environment/docker-compose.yaml`, the only compose file Harbor reads: a
`compose.yaml`, `compose.yml` or `docker-compose.yml` is a plain file of the task, as in Harbor, and a fetch warns of
it (`flags.unread_compose` in the lock). platoon reads the file as Harbor layers it, over a `main` service that is the
task's own sandbox (its image from `environment/Dockerfile` or its `docker_image`, idling on `sh -c "sleep
infinity"`), and a fetch records what it found in the lock: the kind (`flags.stack`) and the file's stack plan
(`compose`).

| `flags.stack` | The compose file | platoon runs |
|---|---|---|
| `single` | none | one sandbox |
| `trivial` | names only `main`, with nothing platoon applies (`services: {main: {}}`, or Harbor's base restated) | one sandbox |
| `main_only` | names only `main`, and gives it settings | one sandbox, with what the file gives `main` (below) |
| `sidecars` | names services besides `main`: a database, a judge, an MCP server | a stack, on a provider that runs stacks ([Multi-container tasks](#multi-container-tasks)); elsewhere the task is refused, never started without its services. A fetch locks it as runnable (`refuse: {sidecars: true}` refuses it, as `stack`) |

**A main-only file** gives the task's sandbox:

- `environment` and `env_file`: its environment, over the image's and the service's `env`, under the task's
  `[environment].env`;
- `working_dir`: its working directory, unless `[environment].workdir` names one;
- `cpus`, `mem_limit` and `deploy.resources.limits`: its resources, over the task's;
- `volumes`: a bind of the task's own files (`./data:/data:ro`, `${CONTEXT_DIR}/..:/opt/task-src:ro`) binds the
  cached task, read-only as written, and a writable one binds a copy of its files, so that the cache is never
  written; a source the task does not have binds an empty directory, as Docker creates one; anonymous and named
  volumes below a bind mask what it binds there: the bind is of a filtered copy in which they are empty directories,
  so a task that hides its `solution/` and `tests/` that way never puts them in the sandbox; a bind of
  `${HOST_VERIFIER_LOGS_PATH}` (or of the agent's or the artifacts' log directory) at its own path changes nothing; a
  named volume elsewhere is a directory of the sandbox's own;
- `command` and `entrypoint`: main's **init**, which the session starts once the sandbox runs, as compose runs a
  service's command (under the image's `ENTRYPOINT` where the image map knows it), unless it only idles (`sleep
  infinity`, `tail -f /dev/null`). The kill before grading spares it and what it started, daemons it detached
  included; an init that is gone by grading, which in Docker would have ended the container, makes the attempt an
  `error` (`main_exited`), never scored; and a `copy` fork of such a task is refused, since a restore brings back
  files, not processes;
- `healthcheck`: waited for once the init has started (else the image's `HEALTHCHECK`), before
  `[environment.healthcheck]`; one that keeps failing is `env_healthcheck`, as that one is.

Published `ports` are kept as `expose`, since nothing is published on the host, and `container_name`, `restart`,
`init`, `stdin_open`, `tty`, `labels`, `logging`, `stop_grace_period`, `pull_policy` and a `platform` that is this
host's are ignored. What one sandbox cannot apply yet (`cap_add`, `shm_size`, `ulimits`, `tmpfs`, a log directory
bound at another path, a bind inside another, networks that are all `internal`) refuses the task when it loads, as
`task_config`.

**Variables.** `${NAME}`, `${NAME:-default}`, `${NAME-default}`, `${NAME:?error}`, `${NAME:+value}` and `$$` (a
literal `$`) work as in compose, and an unset variable without a default is blank, as compose makes it (the lock
notes it). The variables are Harbor's, with platoon's meaning: `CONTEXT_DIR` (the task's `environment/`, which only
binds and build contexts may name), `HOST_VERIFIER_LOGS_PATH` and `ENV_VERIFIER_LOGS_PATH` and their agent and
artifacts counterparts, `MAIN_IMAGE_NAME`, `PREBUILT_IMAGE_NAME`, `CPUS` and `MEMORY` (the task's resources); then
the task's `[environment].env`. The host's environment counts only for the variables an overlay lists in
`compose_host_env`, so a file that names an API key gets a blank one otherwise. A listed variable, and a task
variable that is a host template (`${TOKEN}`), are resolved when a session opens, never written into the lock, under
the form the file gives them (`${OPENAI_API_KEY:-}` is blank when the host has none); one that the file names without
a default must be set then, or the session fails as `env_config`. Environment values may name both, and a read-only
bind the path that a listed variable names, or a path below it
(`${SCICODE_TEST_DATA}/test_data.h5:/app/test_data.h5:ro`): never a task variable's, which the task, not you, chose.

**What the subset refuses**, when the dataset is fetched, as `compose: <service>.<key>: <why>`: keys outside the
subset; security-relevant ones (`privileged`, `devices`, `pid`, `ipc`, `security_opt`, `sysctls`, `user`, a
`cap_add` beyond the capabilities a service's root holds over its own processes and files, such as `NET_ADMIN`,
`network_mode: host`, ...); binds of host paths; static addresses, and networks that keep a service from `main`; two
services that declare one port, since a stack's services share one loopback where they run as a pod; GPU requests (a
task asks for GPUs with `[environment].gpus`); external volumes and networks; a file whose YAML aliases expand it
beyond 100,000 values; the private-use characters U+E000 to U+E007 anywhere a task writes (platoon marks the
variables a session resolves with them); and what compose itself would reject, such as an invalid interpolation, an
env file it cannot read or a dependency on a service that does not start. A dataset's overlay may edit the services:

```toml
# overlays/drop-net-admin/overlay.toml
compose_host_env = ["SCICODE_TEST_DATA"]  # host variables the compose file may name
[match]
source_dataset = "cybergym"
[[compose]]
service = "main"
unset = ["cap_add"]                    # dotted for a nested key: "deploy.resources.reservations"
set = {environment = {MODE = "eval"}}  # merged into the service's mappings; other values replace
[[compose]]
service = "task-server"
replace = {command = ["serve", "--port", "9111"]}  # keys set whole
```

The lock records the edits with the stack plan, and the overlay's hash with the lock's inputs, as for any overlay.
An entry may also mark a service `external = "<node service>"`: a service that a node service serves, which platoon
does not start.

## Multi-container tasks

*Since 0.5.0.* A task whose compose file names services besides `main` (`flags.stack: sidecars`) runs as a **stack**:
`main`, the agents' sandbox, and its sidecars, each in a sandbox of its own, started and stopped together. Only a
provider that declares the `stacks` capability runs stacks: `docker`, each stack a `docker compose` project of its
own ([below](#stacks-on-docker)); `singularity` and `enroot`, as rootless pods
([below](#stacks-as-pods-on-singularity-apptainer-and-enroot)); and `process`, for tests and for checking a dataset
with a policy you trust ([below](#stacks-on-the-process-provider)). Elsewhere such a task is refused, never started
without its services: a dataset directory skips it with a warning, and a fetched dataset's stacks stop a run when its
tasks are listed, naming them. A fetch locks stacks as runnable; where the provider runs none, leave them out of the
run (`exclude_tasks`), or fetch them refused (`stack`) with `refuse: {sidecars: true}`, as platoon before 0.5.0 did
by default:

```yaml
    service_kwargs:
      dataset: ./tasks
      provider: singularity       # runs the stacks as pods (docker: as compose projects)
      # refuse: {sidecars: true}  # on a provider that runs no stacks: lock them refused
```

A lock that platoon before 0.5.0 wrote for an entry that sets no `refuse.sidecars` still answers it where it refused no
task as a stack. Where it did, `platoon prepare`, `platoon train` and `platoon eval` fetch a pinned dataset again (its
cached tasks are reused), which locks its stacks as runnable, and a rollout worker that finds no lock says which lock
refused them; `refuse: {sidecars: true}` keeps that lock.

**What each service gets.** `main` is the sandbox a main-only file would give the task ([Compose files](#compose-files)):
its image, environment, working directory, resources, binds, command and healthchecks, with the harness's toolchain
and gateway route. Each sidecar gets, as compose gives it:

- its image: a registry reference (`postgres:16`, or pinned by digest) as the image map pins it, the image map's image
  of its build context (`build:`, planned whole, so that its build key covers every file it copies), or `main`'s image
  (`${MAIN_IMAGE_NAME}`). `platoon prepare` builds a fetched dataset's sidecar contexts with the provider's builder, as
  it builds the tasks' own images, and **pins** its registry images ([Prepare a
  config](#prepare-a-config-once-per-site)): the digest the registry serves for each, and the configuration it gives
  the image (`ENTRYPOINT`, `CMD`, `ENV`, `WORKDIR`, `USER`, `HEALTHCHECK`, ...), which the sidecar runs with, since
  platoon gives every runtime a sidecar's whole command, and a pod's runtimes apply none of an image's configuration by
  themselves; `platoon harbor images pin` does the same. A run whose images or pins the image map lacks stops when its
  tasks are listed (exit 78, naming `platoon prepare`). A dataset directory has no image map: its registry sidecars run
  as written, with compose's `command` (and `entrypoint`) alone, in compose's `working_dir` (else `/`), so one whose
  compose file gives it neither cannot run (an overlay's `[[compose]]` `replace` gives it one), nor can a `build:`
  sidecar. A dataset directory skips such stacks with a warning, as a fetched dataset's stop a run when its tasks are
  listed, as do stacks that no rollout could start for another reason (a `service_healthy` dependency on a service
  without a healthcheck, say), never failing rollout after rollout;
- its `command` and `entrypoint` over its image's, as compose combines them; its `environment` and `env_file` over its
  image's and the service's `env`; its `healthcheck`, else its image's;
- its `volumes`: read-only binds of the task's files from the cache; writable, masked and missing ones as copies that
  the stack owns, one for every service that binds the same path with the same masks, as compose shares the one host
  directory (so a sidecar sees what `main` writes there, and the cache is never written); named volumes as one
  directory for every service that names them, anonymous ones and `tmpfs` as directories of its own, empty when the
  stack starts. A bind of a log directory (`${HOST_VERIFIER_LOGS_PATH}`, the agent's or the artifacts') is a volume of
  the stack, one per log directory for the sidecars that bind it: grading copies its regular files (never through a
  link) into that directory of the sandbox the tests run in, once it made the directory afresh, so the agents can
  neither erase nor forge what a sidecar records there (a judge's log of the agent's actions, say). A sidecar does not
  see what `main` writes in its own log directories, so a sidecar's read-only bind of one is refused, as are binds of
  one path that services would share in Docker and platoon would copy apart (one inside another that a service
  writes, or with other masks). What the stack owns goes with it, and with a reap of a killed worker's sandboxes;
- its declared ports (`expose`, `ports`), names (`hostname`, network aliases), shared network (`network_mode:
  service:<x>`), resources, `shm_size`, `ulimits`, `cap_add` and the user its image runs as (`USER`), which each stack
  provider applies where its runtime can (`process` applies none).

**Task files that not every user may read.** The copies that a stack owns have the modes a Docker host's checkout has,
which every user may read, and a sidecar's read-only bind of the task's files that not every user may read binds such a
copy too, since a sidecar may read them as a user of its own (the official postgres image runs its init scripts as
`postgres`). A dataset fetched under a umask such as 027, as on many shared clusters, leaves its files so, where a
checkout under Docker's usual 022 leaves them 0644. Fetching under `umask 022` (before `platoon prepare`, say) lets
those binds take the cache, which every user who reaches it then reads, without a copy at each stack's start.

**Starting.** Each service starts once its `depends_on` conditions hold: `service_started`, `service_healthy` (its
healthcheck passed, by Docker's rules, where a duration of 0 is Docker's default) or `service_completed_successfully`
(an init job, such as a seeder, exited with status 0, within `sandbox_start_budget_s`); services that wait for none
start at once. The session opens once every service runs, each one with a healthcheck has passed it and each init job
has completed, as `docker compose up --wait` does, and a last probe, before any agent acts, finds every service up. A
sidecar that fails, exits, never turns healthy, or is down by that last probe fails the session's start
(`sidecar_start`, an `error`, with its log's tail), and nothing of the stack is left, also when the start is
cancelled. A rollout's deadlines count the start along the longest path of the services' dependencies:
`sandbox_start_budget_s` per service start and per init job's run, each healthcheck's longest wait (compose's, else
its image's), and three probes, besides the collect hooks below.

**Grading.** In `main` (`shared`), grading kills what the agents left in `main`, runs the task's `[[verifier.collect]]`
hooks (`main`'s first, each in its service, best effort: a failed one is noted in `collect_failed`), switches the stack
to the verifier's network (`[verifier].network_mode`), probes every sidecar, copies the sidecars' log directories
in, runs the tests, and probes again. With a separate verifier, it collects, after the kill, `main`'s hooks and
artifacts, then the sidecars' hooks and the artifacts that `artifacts[].service` names, each from that service's own
files (an absolute path there; a path another service's artifact already claims is left out), and the sidecars' log
directories, with a probe before and after, and the verifier's sandbox gets them all. A probe finds every sidecar up
(an init job completed, any other running, and healthy where it has a healthcheck), else the outcome is `invalid`
(`sidecar_unhealthy`: `metadata.sidecar`, its state and the tail of its log), whatever the reward files say, and
never trains. Where an agent may stop a sidecar on purpose and should own that, `on_sidecar_down: grade` grades
anyway and notes it (`sidecars_down`). Every stack outcome records its services' run modes and their last probed
states (`metadata.services`).

**Forks.** A `shared` fork shares the stack and a `fresh` one opens a new one; a `copy` fork is refused, since a
restore brings back `main`'s files, not its services.

| Option | Default | Meaning |
|---|---|---|
| `stacks` | `auto` | `auto` runs a stack on a provider that runs stacks, and refuses it elsewhere; `refuse` refuses every stack |
| `network_policy` | `task` | `task` gives each phase the network the task asks for (`network_mode`, `[agent]`'s and `[verifier]`'s); `none` gives every sandbox and stack none, in every phase, for datasets whose answers the web holds (a provider that cannot run a sandbox without a network refuses the tasks then) |
| `on_sidecar_down` | `invalid` | What a sidecar that is down at grading makes the outcome: `invalid` (`sidecar_unhealthy`), or `grade`, which runs the tests anyway |

### Stacks on Docker

The `docker` provider runs each stack as a `docker compose` project of its own, `platoon-stack-<id>`, whose directory
below the provider's `stack_root` holds the project's `compose.yaml`, written from the stack's spec, and its volumes.
The file is JSON, which compose reads as the YAML it also is, so that no value changes its type on the way (YAML reads a
plain `1e3` or `0o17` as a number). `stack_root` must be a directory of your own: a symlink, or a directory of another
user's, is refused, since on a node's shared scratch whoever made it could change a project's file before compose reads
it; and so is a path with a `:`, which ends a bind's source where Docker reads one. Stacks need the CLI's compose plugin
(`docker compose version`); without it, they are refused when a run lists its tasks. Each service is a container,
`<project>-<service>`, that compose creates once its dependencies hold, and that runs its command (compose's over its
image's) as its `entrypoint`, as its image's user; `main` idles, as a single sandbox does. platoon starts the services
in order, runs their healthchecks in them (`docker exec`) and probes them, as on every provider; the project's file
leaves dependencies and healthchecks out.

- **Networks.** The services share a network of their own, `<project>-internal`, which is `internal` and gives the
  host no address on it (`com.docker.network.bridge.inhibit_ipv4`): on it each answers to its name and aliases at the
  ports it declares, as in compose, and nothing beyond the stack answers, the host's own services included. A daemon
  before 25.0.5 passes an internal network's DNS queries on to the host's resolvers (CVE-2024-29018), through which a
  stack could send data out in a phase without egress: platoon warns of one when a run lists its tasks. While the
  stack has egress, each service is also connected to `<project>-egress`, a bridge network that reaches the web, and
  disconnected from it for a phase without. A bridge network also reaches the host's own addresses (not its
  loopback), so keep services that must stay out of the agents' reach, such as an inference engine, on `127.0.0.1`.
  `main` reaches the model gateway through its socket's directory and a forwarder on `127.0.0.1:18080`, as a
  container without a network does. Each stack takes a network of Docker's address pools, and one more while it has
  egress, and the default pools hold about 30, so about 15 stacks with egress at once: a host that runs more needs
  them widened (`default-address-pools` in Docker's `daemon.json`), and a start that finds none left fails
  (`sidecar_start`).
- **Volumes** are directories of the project (a named one shared by every service that names it), made and filled
  before any service starts, which any user of the services may write; `tmpfs` ones are Docker's. They start empty, as
  on every provider: Docker fills a new volume with what an image has at its path only where the image declares a
  `VOLUME` that compose leaves alone (`redis`' `/data`, say), and removes those volumes with the stack.
- **Privileges.** A `cap_add` of a capability that a rootless service's root holds (Docker's defaults, such as
  `SYS_CHROOT`, and `SYS_PTRACE` over the service's own processes) applies as written. `privileged`, and a `cap_add` of
  another (`NET_ADMIN`; `DAC_READ_SEARCH`, with which a root that is the host's, as on a rootful Docker, reads any of
  the host's files by its handle; ...), are refused when the dataset is fetched, unless the fetch keeps them with
  `refuse: {privileges: false}`: then such a stack runs on `docker` alone, the one provider with the `privileged`
  capability, and every other refuses it when a run lists its tasks. Grant them only to datasets you trust with them: a
  privileged container can take the host.
- **Teardown.** The containers that carry the stack's label go, with their anonymous volumes, then its networks: what
  the stack made, and also what a docker command that failed or timed out left the daemon to make. Since the daemon
  carries out a request whatever becomes of the CLI that sent it, a start that is given up waits for its docker commands
  rather than kill them, and removes what they made once they are done. Then the project's directory goes, what the
  services wrote there as root or as another user through a container of `main`'s image that runs `rm` as root, and its
  record last. A reap by label (a killed worker's,
  [Sandboxes left by stopped rollouts](#sandboxes-left-by-stopped-rollouts)) removes the stacks whose containers or
  directories carry the label, with their networks and directories, and a stack whose teardown failed stays registered
  for that reap.

### Stacks on the process provider

The `process` provider runs each service as a process sandbox on your machine, its command in the background there,
so it isolates nothing: images, networks, resources, users, capabilities and read-only mounts are not enforced (what
is written through a read-only bind reaches the task's files), and the services share the host's network. So that
stacks running at once never take one another's ports, each port a service declares is a port of the host that the
stack holds for its life: one outside the kernel's ephemeral range (`ip_local_port_range`, which neither a bind to
port 0 nor an outgoing connection takes), free when the stack starts, and claimed so that no other stack, in any
process, is given it. The service learns it from its **listen variable**, `PLATOON_PORT_<declared port>`
(`PLATOON_PORT_6379=21234`), and every `scheme://<service>:<port>` in the services' environment values, healthchecks
and collect hooks becomes `scheme://127.0.0.1:<that port>`. A service written to run
there too listens on `${PLATOON_PORT_6379:-6379}` and writes below `${PLATOON_SANDBOX_ROOT:-}`, which real stacks
leave unset; a reference elsewhere (a literal in a script, a service's own `localhost:<port>`) is not rewritten.
`tests/envs/fixtures/harbor-stacks` has tasks written that way.

### Stacks as pods on Singularity, Apptainer and enroot

The `singularity` and `enroot` providers run each stack as an hpc-sandbox **pod**: a user and network namespace of its
own, which a small process (the holder) keeps, with each service a container in it (hpc-sandbox's
[pods](https://github.com/ApGa/hpc-sandbox/blob/main/docs/pods.md)). The services share the pod's one loopback, so
each reaches the others by name at the ports it declares, as in a compose project, with nothing rewritten, and
nothing outside the pod reaches them: two stacks of one task on one node, their sidecars on one port, never see each
other's. Pods need hpc-sandbox 0.5, whose pod services run as your uid or with your subordinate ids and whose pods take
host aliases and expose ports to the host, and a node that lets unprivileged users nest user namespaces (the pod's,
each service's inside it, and its runtime's own); with hpc-sandbox 0.4, or on a node that does not nest them, every
stack is refused when the tasks are listed, naming what is missing. The `hpc` extra installs 0.5.1, with which a sidecar
that switches users (`subids`, below) also starts where the run's directory or the task's files are on storage that only
one of your groups opens, and reaches the mount points made for its binds under a umask such as 027, and with which
`main`, idle while the agents' commands run in it, outlives an agent's `kill -9 -1`, as a single sandbox does; with
0.5.0, such a sidecar fails to start or to enter those directories, and such a kill ends `main` and makes the attempt an
`error`. Pods on Apptainer and on enroot are *experimental*: `tests/integration/test_harbor_stacks_hpc.py` is what a
site runs to check them.

Each service runs in a **run mode**, which outcomes record in `metadata.services`:

| Service | Run mode | It runs as |
|---|---|---|
| `main` | `root` | root, which is your uid on the host, as a single sandbox's commands do |
| A sidecar whose image runs as root (no `USER`), on Singularity or Apptainer, where your user has subordinate ids | `subids` | root that can switch users, as Docker's does: the official database images give their data to a user of their own (`chown`) and switch to it (`gosu`, `su-exec`). What that user writes belongs to your subordinate ids on the host, which hpc-sandbox gives back to you before it deletes the pod |
| Any other sidecar: its image runs as another user, or enroot runs it (enroot maps one id into a container), or your user has no subordinate ids | `caller` | your own uid, without root, who owns the image's files; the database images skip their switch when they do not start as root |

The first pod of a process that runs a `subids` sidecar maps your subordinate ids (`newuidmap`), which can take tens of
seconds, or more than a minute, where `/etc/subuid` lists many users; the later pods of the process start without it.
Each rollout worker is such a process, so its first stack pays it, within that start's budget and while it holds a
start slot (`train.max_starting_rollouts`). Where the mapping is refused, each such start fails (`sandbox_start`), with
the tools' message; where your user has no subordinate ids, the sidecars run as `caller`, and the run warns once.

- **The gateway.** `main`'s upstreams, the model gateway's socket among them, are the pod's: the holder relays a port
  of the pod's loopback (18080, or the next that no service declares) to the socket on the host, so a harness in
  `main` reaches the gateway at `routes["gateway"]`, with no forwarder in `main` that what the agents run could stop.
  Every service of the pod reaches that port, as they share one loopback, but a model call needs an agent's key,
  which only the harness holds.
- **Network per phase.** A stack has the network its task asks for (`egress`, through slirp4netns, or `none`), and
  grading in `main` switches it to the verifier's. Egress reaches other hosts, the node's own addresses among them,
  but never the host's loopback. It needs slirp4netns on `PATH`, else the release that `hpc-sandbox fetch-tools`
  downloads once, which `platoon prepare` fetches where a stack needs egress and the node has none.
- **Volumes** are directories in the pod's own directory, in the provider's scratch, made with Docker's modes (0755,
  a `tmpfs` 1777) and filled before any service starts, so the pod's deletion takes them, with what a `subids` sidecar
  wrote there. A `tmpfs` is such a directory too, not memory. Grading reads a log directory that a `subids` sidecar
  binds through that sidecar, as its root, so what its own users keep from yours still reaches the tests; a file
  grading cannot read makes the outcome an `error` (`verifier_setup`; `artifact_collect` for a separate verifier),
  never a grade without it.
- **Images.** A sidecar runs its registry image as `platoon prepare` pinned it, and its build context's image as
  `platoon prepare` built it with the provider's builder: `oci`, or `def` where the runtime is Apptainer, which has no
  BuildKit (single-stage Dockerfiles only). enroot runs registry images only: build and push a build context's with
  `platoon harbor images build CONFIG --builder docker --push REGISTRY/NAME`.
- **What a pod does not give:** an address per service (two services on one port collide, so a fetch refuses two that
  declare one, and every connection between services comes from `127.0.0.1`: below), GPUs, a sidecar's image `USER` as
  its own uid (a `caller` sidecar runs as yours), snapshots (so `copy` forks stay refused), and capabilities over the
  stack's network (`NET_ADMIN`, `NET_RAW`, which a fetch refuses).
- **One loopback, so one client address.** Every connection between a pod's services comes from `127.0.0.1`, `main`'s
  too, where under compose `main` connects from an address of its own network. So a sidecar that trusts loopback, or
  listens on it alone, is open to what the agents run in `main`: the official postgres image, given a
  `POSTGRES_PASSWORD`, still lets `127.0.0.1` in without it, as a superuser (its `pg_hba.conf` trusts `127.0.0.1` before
  it asks other addresses for the password), so the agents can read or change a database that the task means to keep
  from them. platoon neither refuses nor changes such sidecars: no list of the images that trust loopback would be
  complete, and a refusal would also refuse tasks whose agents are meant to use the database. A task that keeps a
  sidecar from its agents must not rely on its loopback rules; for postgres,
  `POSTGRES_INITDB_ARGS: --auth-host=scram-sha-256` makes a connection from `127.0.0.1` need the password too. An
  address of its own for each service, which closes this, is a follow-up of hpc-sandbox's pods.

A pod's services stop with it, in milliseconds, and a pod dies with the process that started it; a reap by a run's
labels stops a live process's pods and sweeps those a dead one left, giving back what other users wrote first.

## Users in task sandboxes

*Since 0.5.0.* Harbor runs each phase of a task as the user the task names, as Docker does: the agents' commands (a
harness, the env tools, the reference solution) as `[agent].user`, else as the image's `USER`, else as root, and the
tests as `[verifier].user`, else as the image's `USER`, else as root. A rootless sandbox runs every process as you, so
platoon refuses a task that names a user by default (`agent_user: refuse`), and `ignore` runs it as the sandbox's own
user, whatever that changes. `honor` runs each phase as Harbor does, in sandboxes that give root and other users:

```yaml
environments:
  - name: as-their-users
    service: harbor
    service_kwargs:
      dataset: hf://my-org/my-harbor-dataset@<revision>
      refuse: {agent_user: false}       # fetch the tasks that name users, or whose Dockerfile ends with USER
      agent_user: honor                 # each phase as its user, as Harbor runs it
      provider: singularity
      provider_options: {allow_network_none: true, rootfs_layout: copy, fakeroot: true}
```

- **Who runs what.** With `agent_user: honor`, and `verifier_user` unset or `honor`, the agents' commands run as
  `[agent].user`, else the image's `USER`, else root; the tests run as `[verifier].user`, else the image's `USER`,
  else root; a `[[verifier.collect]]` hook runs as its own `user`; and platoon's own commands (the sandbox's setup,
  the kills before grading, reading the reward files) run as root. A user takes any of Docker's `--user` forms: a name
  the image's `/etc/passwd` lists or a uid, either with a group's name or a gid. An outcome records the users its
  task names (`users`; `null` for a phase that ran as the image's `USER`, else root). `verifier_user: honor` alone
  honors `[verifier].user` only, any user ([below](#verifiers-that-run-as-root) for root), and a verifier that names
  none grades as before.
- **Where users run.** A sandbox that runs a phase as its user must give root and other users:

  | Provider | Users | How |
  |---|---|---|
  | `singularity` | With `fakeroot: true` on the copy layout, with hpc-sandbox 0.4 (0.5 for an image that gives files to users other than root, below) | Root of a user namespace that maps your subordinate ids; platoon switches each command to its user with the image's `setpriv` (util-linux), else its `su`, else a Python's standard library (the image's, or a harness toolchain's), and sets `HOME`, `USER` and `LOGNAME` from the image's `/etc/passwd`. A session whose image has none of them fails to start (`env_config`) |
  | `docker` | Yes | `docker exec --user`; the sandbox's own commands run as root, whatever the image's `USER` |
  | `enroot`, `process` | No | Refused: they run every command as your own uid |
  | stacks | No, for now | Refused when they load, where users run in the stack itself: under `agent_user: honor` every stack, whose agents would run as `main`'s `USER`, and a stack graded in `main` that names a `[verifier].user`, or whose collect hook names a user. Docker's compose projects and the pods honor users in a later release; a separate verifier's sandbox runs its user already |

  `platoon train`, `eval` and `rollout` refuse, when they list their tasks (exit 78), those whose sandboxes cannot
  give their users, and a session refuses one before its agent runs; a user that the image lacks fails the session's
  start (`env_config`).
- **As Docker gives it.** In a sandbox that honors users, platoon's commands and the users' run with Docker's umask
  (022); `/logs/agent` and `/logs/artifacts` are every user's (777, as Harbor makes them); what platoon uploads for
  the agents (a harness's files, the reference solution) is theirs, and the files of a deferred `COPY` keep the host's
  modes, with every user given the read and search access that their group has, as a checkout under Docker's umask
  gives it (a file of mode 600 stays so), where a deferred `COPY --chown` gives its owner what it copies and the
  directories it makes, as BuildKit does. A deferred `WORKDIR` gives the directories it makes to the `USER` in effect
  there (the image's own before a deferred `USER`), as BuildKit does, while a directory the image has keeps its owner;
  a sandbox whose image defers such a `WORKDIR` starts in `/`, so that its start makes no working directory as root,
  and its commands run in the working directory. Tests that run as a user other than root get `/tests` as theirs and
  `/logs/verifier` every user's (777); tests that run as root get both as root's, `/tests` with the host's modes, as
  Harbor copies it, so that the agent's code they run as another user can neither read a test file the host keeps
  from it nor write a reward. Harness toolchains are staged readable by every user. The `edit` tool reads a file
  through the sandbox's shell, as the agents do, so it cannot read a file they may not, not even to count what it
  holds; it refuses at once what is not a regular file (a FIFO, a device), and gives up on a read after
  `tool_timeout_s`.
- **An image's file owners.** Docker gives an image's files the owners its layers give them. hpc-sandbox extracts every
  image as you, and Singularity's conversion of a registry image records every file as root's, so a user that the image
  gives a directory (its home, a workdir its Dockerfile `chown`s to it) would find it root's. An image's **owner
  manifest** lists what its layers give users other than root, and `platoon prepare` records it in the image map: the
  `oci` builder lists its layer's owners before it unpacks it, and a task's `docker_image` becomes a *pulled* entry, the
  image pinned by digest with its `USER` and the owners its registry's layers give, which the sessions run. hpc-sandbox
  0.5 applies a manifest in a `fakeroot` sandbox before anything runs there; with an older one, such as the 0.4.0 that
  platoon 0.4 installs, a task whose image gives files to users other than root is refused when it is listed, naming
  0.5. The image map's record also tells the image's `USER`, which a sandbox there cannot read. So a run refuses a task
  whose sandbox runs a user other than root, or anything as its image's `USER` (the agents' sandbox under `agent_user:
  honor`, tests whose `[verifier].user` is unset), where the image map records no manifest: a `docker_image` that
  `platoon prepare` has not recorded (exit 78, naming it), an image that platoon before 0.5, the `def` builder
  (Apptainer's), or the `oci` builder with a squashfs-tools before 4.4 built (build it again where `unsquashfs` is 4.4
  or later, `platoon harbor images build CONFIG --key KEY`), a local image file, and a dataset directory's task, which
  has no image map (serve the directory as a `dataset` to fetch). A sandbox in which only root runs, as the task names
  it (a root verifier), needs none, as root verifiers ran before. Docker keeps its images' owners itself, and runs an
  image's `USER` itself.
- **Forks.** A `copy` fork of a sandbox that honors users is refused on Singularity, whose snapshots keep no owners;
  on Docker it runs as its users too.
- **The cost.** `fakeroot` needs the copy layout, a start of 2 s alone and 24-30 s with 64 at once
  ([above](#the-singularity-providers-root-filesystem)); `rootfs_templates: true` copies a node's template of the
  image instead of extracting it. So `honor` is a choice per dataset.

### Verifiers that run as root

*Since 0.3.0.* Some Harbor tasks grade as root: `task.toml` says `[verifier] user = "root"`, and the tests
run the agent's code as another user, `nobody` (65534) as a rule, so that it cannot read the files the grader keeps
for itself, such as held-out answers. In Docker that works as written. In a rootless sandbox the tests run as your own
uid, which cannot switch to another user, and the files they keep for themselves are yours too: such tests either fail
closed and score every attempt 0, the reference solution's included, or run the agent's code where it can read the
held-out answers. platoon refuses these tasks by default, and `agent_user: ignore` runs them as the sandbox's own user
anyway, with those consequences. `verifier_user: honor` runs their tests as root, in a separate verifier's sandbox
that gives root and the other users:

```yaml
environments:
  - name: graded-as-root
    service: harbor
    service_kwargs:
      dataset: hf://my-org/my-harbor-dataset@<revision>
      refuse: {agent_user: false}       # let tasks that name users through the fetch
      agent_user: ignore                # [agent].user: the agents run as the sandbox's own user
      verifier_user: honor              # [verifier].user = "root": the tests run as root
      provider: singularity
      provider_options: {allow_network_none: true, rootfs_layout: fuse-overlay}   # the agents' sandboxes
      verifier_provider_options: {rootfs_layout: copy, fakeroot: true}           # every separate verifier's sandbox
```

On Singularity this needs an hpc-sandbox whose provider takes `fakeroot` (below): 0.4 or later (the `hpc` extra
installs 0.5.1). With an older one, such as 0.3.0, a run with this config stops when it lists its tasks, with a message
that says what is missing.

- **What `honor` takes.** A `[verifier].user` of root (`root`, `0`, `0:0` or `root:root`) in a task with a
  [separate verifier](#a-separate-verifier), and since 0.5.0 any other user too. A root verifier of a task graded in
  its agents' own sandbox is refused, whose agents would then run as root too, unless `agent_user: honor` runs them as
  their user ([above](#users-in-task-sandboxes)): set `verifier_user: ignore` for the others, or leave them out
  (`exclude_tasks`). `agent_user` still decides for `[agent].user`, and without `verifier_user` it decides for both
  users.
- **A fetched dataset.** The fetch's `refuse.agent_user`, on by default, leaves out every task that sets
  `[agent].user` or `[verifier].user`, before `honor` sees it: set `refuse: {agent_user: false}`, as above, and
  `agent_user: ignore` for the tasks that also set `[agent].user`. With `honor`, a run that lists its tasks warns of the
  root verifiers its fetch left out.
- **The verifiers' own sandboxes.** `verifier_provider_options` start every separate verifier's sandbox, laid over
  `provider_options`, so they name only what changes; the agents' sandboxes keep `provider_options`, and
  `platoon sandbox reap` and a run's reaping look for both. A separate verifier whose task names no user grades as
  root there too, as Docker would run it.
- **Which providers give root.** The tests need root that can switch to other users, which the providers give as
  follows:

  | Provider | Root that can switch users | What gives it |
  |---|---|---|
  | `singularity` | With `fakeroot: true` (Singularity's `--fakeroot`; `extra_start_args: [--fakeroot]` means the same) on the copy layout, with hpc-sandbox 0.4 | Root maps to your uid, and your subordinate ids (`getsubids $USER`, from `/etc/subuid` and `/etc/subgid`) to the users after it, so nothing gains a privilege on the host. What the other users write belongs to your subordinate ids, which you cannot delete: hpc-sandbox 0.4 gives those files back to you before it deletes the sandbox, with the image's `find` and `chown` (an image without them leaves the sandbox on the node, with a warning), also when a janitor sweeps one that a killed process left, and makes a working directory the image lacks with Docker's modes. With an older hpc-sandbox the run is refused |
  | `docker` | When the verifier's image sets no `USER`: Docker's root, or rootless Docker's and Podman's root over your subordinate ids | Checked when the verifier's sandbox starts |
  | `enroot` | No: it maps only your uid into a sandbox | Refused |
  | `process` | No: commands run on the host as you | Refused |

- **Refused before rollouts.** `platoon train`, `eval` and `rollout` refuse, when they list the tasks (exit 78,
  naming them and why), the tasks whose verifier runs as root where the verifiers' sandboxes cannot give it root, and
  a session refuses one before its agent runs ([What platoon refuses](#what-platoon-refuses)). They refuse the tasks
  of a fetched dataset whose `[verifier].user` `honor` cannot honor the same way, while a dataset directory
  (`dataset_dir`, `task_dirs`) skips those with a warning, as it skips other tasks it refuses.
- **The started sandbox is checked.** Whatever the provider, a root verifier's sandbox is checked when it starts: its
  commands must run as uid 0, with every uid and gid up to 65534 mapped, and `nobody` must be able to run the image's
  `/bin/sh` and load its libraries (through `setpriv`, else `su`, when the image has either). A sandbox that fails
  the check makes the outcome an `error` (`verifier_start`) that does not train, and its reason says why.
- **Graded as Docker would.** In a verifier that runs as root, platoon's commands and the tests run with Docker's
  umask, 022, rather than the host's (often 027 on shared machines), so what the tests write as root is readable by
  the users they switch to. What platoon makes there gets Docker's modes (755): `/logs` and its directories, and the
  image's deferred `WORKDIR` directories. A working directory that the image lacks is made when the sandbox starts:
  hpc-sandbox 0.4 makes it, and its missing parents, with these modes too, as Docker does; where a deferred `WORKDIR`
  follows a `USER`, platoon makes them after the sandbox starts, and gives them to that user, as BuildKit does. A
  directory the image has keeps its mode, as in Docker: a workdir such as `/root`, which images keep to root (700),
  keeps the files the tests write there from the users they switch to. The artifacts are made readable by every user
  (`chmod -R a+rX`), as an agent's files are under Docker's umask, and the outcome's metadata records `verifier_user:
  root`. `tests/`, when the verifier's image does not hold it, keeps its modes from the host, as Harbor's copy keeps
  them, and `/logs/verifier` stays root's (755), so that the agent's code the tests run as another user can neither
  read a test file the host keeps from it nor write a reward. Since 0.5.0, the files of an image's deferred `COPY`
  steps keep the host's modes with every user given the read and search access that their group has, as a checkout
  under Docker's umask gives it (a file of mode 600 stays so).
- **Images the other users can run.** The users the tests switch to must be able to read and run the image's
  programs. The `oci` builder of platoon before 0.3.0 unpacked the images it built under the host's umask, so that on
  a host whose umask withholds those bits (such as 027) their files keep other users out. Its receipts in the image
  map record no `builder_revision`, and with `honor`, `platoon train`, `eval` and `rollout` refuse the tasks whose
  verifier runs as root on such an image when they list them, and `platoon prepare` fails, naming the keys: build
  them again with `platoon harbor images build CONFIG --key KEY ...`. `images build` builds only the keys the map
  lacks unless `--key` names them (`platoon harbor images plan` lists the keys), and a fresh image map holds none.
  Build these verifiers' images with the `def` builder where you can (`--builder def`): Singularity's rootless
  BuildKit, which the `oci` builder runs, may fail to `chmod` files that a Dockerfile copied in, as such verifiers'
  Dockerfiles do to keep their files to root. Check an image with
  `singularity exec IMAGE stat -c '%a %n' /usr/local/bin/python3`: on an image whose files keep other users out,
  `singularity exec` itself fails, since it runs as you, one of those other users; and a program that other users may
  not run (750) fails the tests as surely as a missing one.

## Rewards: the reward contract and log classifiers

Which reported reward trains, and what makes a reading invalid, is the dataset's knowledge. The `reward` option
says it:

```yaml
service_kwargs:
  reward:
    primary: correctness              # the key that trains; a reward file without it is an error, never 0
    components: [correctness, submission, tool_efficiency]   # other keys make the outcome invalid
    range: [0, 1]                     # a primary outside it makes the outcome invalid
    check: my_dataset.contract:check  # the dataset's own rule, from its own package
  verifier_classifiers: [harbor-common]
```

**Tasks that name their score differently.** One dataset's tasks may report the reward that trains under different
names: some write `reward.txt` (its number is `reward`), others `reward.json` with `correctness` and `submission`.
List the names, and one environment trains them all:

```yaml
service_kwargs:
  reward: {primary: [reward, correctness], range: [0, 1]}   # or reward: [reward, correctness]
```

Each reading must report exactly one of the listed names, from one reward file. A reading that reports none of them
is an error, never a 0; one that reports two of them, or comes with both `reward.json` and `reward.txt`, is
`invalid`; and a group whose rollouts trained different names does not train. [Which reward
trains](harbor-grading.md#which-reward-trains) has the rules, and what each outcome records.

A `check` hook is `check(rewards, verifier_dir, task) -> Outcome | None`. It gets the reported rewards, a local copy of
what grading read from `/logs/verifier` (a directory named `verifier` that holds the reward files and the last MiB of
`test-stdout.txt`, `test-stderr.txt`, `stdout.txt` and `stderr.txt`, each where it was a regular file) and the task
(`task.index`, the dataset index row; `task.metadata`, its `[metadata]` table). It returns an outcome to use instead,
such as `Outcome.infra("verifier_contract", status="invalid")` for a reading its rules reject, or None to keep
platoon's. A hook that raises is an `error`. A module that cannot be imported fails when the service is built, so a run
stops at startup instead of at its first grade.

Classifiers read the verifier's logs (the last MiB of `test-stdout.txt`, where platoon sends `test.sh`'s output, and of
`test-stderr.txt`, `stdout.txt` and `stderr.txt` when the tests write them) before the reward files. A match makes the
outcome `invalid` with the classifier's kind, whatever the reward file says: upstream graders often write 0 before their
tests run, so a grader that crashed reads like a policy that failed. None is on by default, because a test that runs the
agent's program under a timeout and lets the timeout escape measures a slow solution, not a broken grader.
`harbor-common` holds three patterns: a grader subprocess that timed out (`verifier_subprocess_timeout`) and a missing
`pytest` or `uv` (`verifier_missing_test_runner`). Add your own as
`{pattern: "^judge unreachable$", kind: judge_unavailable}`; patterns are multiline regular expressions.

## One dataset, one environment

A Harbor dataset is one environment entry, however its tasks differ. Which reward file a task writes, which source
it comes from and which group of related tasks it belongs to are facts of its data, and platoon reads them from the
task, so a config never splits a dataset into entries to tell them apart:

```yaml
# configs/environment.yaml
environments:
  - name: science
    service: harbor
    service_kwargs:
      dataset: hf://my-org/science-tasks@3f1c0e9b6d2a4c8e9f0a1b2c3d4e5f6a7b8c9d0e
      select: {source_dataset: [data-analysis, constructions]}
      index:                                   # the fields the hold-out reads, recorded in the lock
        - {path: data/tasks.jsonl, key: task_id, fields: [split_group]}
        - {metadata: [kaggle_dataset_name]}
      # Evaluation tasks: whole groups, 5% of each source. A data-analysis question's group is the Kaggle dataset
      # its files come from, a construction's the family the dataset's split_group names; and tasks that share an
      # input file are always in one group.
      holdout: {fraction: 0.05, group_by: [kaggle_dataset_name, split_group], stratify_by: source_dataset}
      reward: {primary: [reward, correctness], range: [0, 1]}   # reward.txt, or reward.json's correctness
      refuse: {uploads_over_mb: 100}
      verifier_classifiers: harbor-common
      provider: singularity
      provider_options: {allow_network_none: true}
    harness: platoon-bash
```

- **Different reward files**: a list of primary names ([Rewards](#rewards-the-reward-contract-and-log-classifiers)).
- **A held-out split** by whole groups (`holdout`, below), from the index fields the fetch records, task metadata
  included ([Fetch a dataset by name](#fetch-a-dataset-by-name)), and from the files the tasks share.
- **Tasks too large to copy into every sandbox**: `refuse: {uploads_over_mb: ...}`.
- **Each source's numbers**, in training and on the held-out tasks: each index field is a column of the run's
  dataset, and `train.label_by` breaks the metrics down by one
  ([Index fields as dataset columns](#index-fields-as-dataset-columns)).

One entry draws every task once per epoch, in the dataset's own proportions. Several entries are for environments
that differ in what they run or how they grade (another harness, another sandbox provider, another reward contract),
or for a deliberate reweighting of the dataset's sources.

Prepare it with the training config itself, on a node with network access, before the run starts
([Prepare a config](#prepare-a-config-once-per-site)); the command prints the hold-out's report, and run again right
before a launch it does nothing but print it again:

```bash
platoon prepare configs/train.yaml    # configs/train.yaml extends configs/environment.yaml
```

### Hold out whole groups for evaluation

Without a split manifest, every split lists every task, so a run would evaluate the tasks it trains on. With
`holdout`, the service lists two splits, the ones `platoon train` and `platoon eval` ask for: `eval`, the held-out
tasks, and `train`, the rest.

| Key | Meaning |
|---|---|
| `fraction` | The share of each stratum's selected tasks to hold out, at most (between 0 and 1) |
| `group_by` | The index field that names a task's group, or a list of them: the first one a task has |
| `stratify_by` | An index field whose values each get their own share (default: the whole dataset is one stratum) |
| `seed` | Ranks the groups and orders the lists (default 0) |
| `ignore_inputs` | Input files that join no groups, as patterns of task-relative paths (`*` matches `/` too): files that every task holds by design, such as a game's code ([below](#shared-input-files); default: none) |

The rule runs over the dataset's selected tasks, refused ones included:

1. Tasks fall into strata by `stratify_by`, and into groups by the first `group_by` field they have. A runnable task
   without them is an error that names it; a refused one sits out.
2. Groups whose tasks share an input file are joined into one, also through other groups: two groups that each
   share a file with a third are one group with it. Files compare by their bytes, and text files by their text, so a
   table saved with other line endings or a byte-order mark is the same file ([below](#shared-input-files)). A joined
   group takes the name of its member with the most tasks.
3. Within each stratum, groups are ranked by the SHA-256 of `"<seed>\0<group>"`. Walking that order, a group is held
   out while the stratum's held-out tasks, with the group's, number at most ⌈`fraction` × the stratum's tasks⌉. A
   group that does not fit is passed over, and no group is split, so a stratum whose groups are all larger than its
   share holds out nothing. A group with tasks in several strata must fit the share of each, and is held out in all
   of them or in none.
4. Each split lists its runnable tasks, in the order of the SHA-256 of `"<seed>\0<task id>"`, so an
   `eval_dataset_kwargs.limit` takes a spread-out sample. Refused tasks leave both lists wherever they fell, so
   changing a refusal never moves a group; `exclude_tasks` apply afterwards, as always.

The same lock and rule give the same split anywhere. `holdout` and `splits` together are an error, and so is asking
for a split other than `train` and `eval`, or for an `eval` split that holds no runnable task, which an evaluation
would measure nothing on.

When the trainer lists the tasks, it logs the hold-out's report once: per stratum, the selected and runnable tasks,
the held-out tasks and their groups, the largest held-out group and its share, and refusals by reason; then the
groups the fields gave, how many of them shared input files joined, and the largest group; and a digest of the
evaluation's task ids:

```text
hold-out: 5% of each source_dataset, grouped by kaggle_dataset_name, else split_group, joined by shared input files, seed 0
  source_dataset constructions: 1,206 selected, 1,206 runnable; held out 61: eval 61 in 23 group(s) (largest graceful-graphs: 17, 28%); train 1,145
  source_dataset data-analysis: 4,900 selected, 4,659 runnable; held out 245: eval 220 in 21 group(s) (largest titanic: 61, 28%); train 4,439; refused: uploads_over_mb 241
  groups: 563 by kaggle_dataset_name, else split_group; 43 shared input file(s) joined 90 of them into 30, leaving 503; largest videogame-sales: 346 task(s)
  eval 281 task(s), train 5,584; eval digest b26d4f07df43
```

A seed that puts most of a stratum's evaluation into one large group shows here; pick another seed before the first
run, since a run's held-out tasks cannot change once it trains.

#### Shared input files

Tasks that hold the same data always land on the same side, whatever `group_by` names, so that no held-out number
measures data that training has seen. Group fields alone miss it when one table comes under several names: grouped
by dataset name alone, one data-analysis dataset left 2.5-46% of its held-out questions (by the seed) with a
byte-identical file in training, the same public tables uploaded under several names. Nor are the copies always byte
for byte: compared that way, the same dataset still left up to 37% of its held-out questions with a table that
training held too, saved with other line endings (CRLF or LF) or without its byte-order mark. So text files compare
as text: a file's digest is the SHA-256 of its text with CRLF and CR line endings as LF, every line ended, and a
leading byte-order mark, the spaces and tabs that end its lines, and its blank lines dropped. A binary file, one with
a NUL byte among its first 8000 bytes, compares by its bytes.

A task's input files are what its agents' sandbox takes from the task: the files of `environment/` that its
Dockerfile's `COPY` and `ADD` steps take, whether a build bakes them into the image or a session uploads them
([Build task images](#build-task-images)), or the whole `environment/` of a task with a prebuilt image and no
Dockerfile, and the task's files that a main-only compose file binds into its sandbox, but for what its volumes mask
([Compose files](#compose-files)). A file that no step takes, the `Dockerfile` and the compose file, `task.toml` and
`instruction.md`, the tests and the solution are not inputs, bound or not, and an empty file (or one of blank lines)
joins nothing. A fetch records each task's input files in the lock
(`input_files`), and a `dataset_dir`'s are hashed when its tasks are listed. A refused task's files join groups too,
so that a refusal never moves one.

Some datasets give every task the same files by design, such as a game's code and its rules: they would make the
whole dataset one group, which no share fits. The evaluation would then hold no task, and listing it stops with a
configuration error that names such a file. Name those files in `ignore_inputs`, and they join nothing:

```yaml
holdout: {fraction: 0.05, group_by: level, ignore_inputs: [environment/game.py, environment/rules.json]}
```

Platoon never decides by itself that a shared file is not data: `ignore_inputs` is recorded with the rule in the
run's `identity.json`, and the report counts the files it matched. A lock written before platoon recorded input
files still answers its config, and a run without a hold-out reads it as before; a hold-out on it stops with a
configuration error, `platoon harbor verify` exits 78 on it, and `platoon harbor fetch` on the config fetches it again
once to record them.

### Index fields as dataset columns

When a run lists a Harbor entry's tasks, each task's index fields become columns of the run's dataset, beside its
`task_id`: a locked task's fields from the dataset's own index (`task-index.json`) and from the `index` entries, its
`[metadata]` fields among them, or a dataset directory's `task-index.json` row. Each value is text (`true` or
`false` for a boolean, a number as Python writes it, such as `0` or `0.5`); a value that is not a scalar gives none.
`train.label_by` names one, to label the tasks by it ([Label tasks by a column](train-and-eval.md#label-tasks-by-a-column)).
Each training step's metrics, and each evaluation's, then break down by its values, such as each source's reward in
training and on the held-out tasks:

```yaml
train:
  label_by: source_dataset    # train/batch/source_dataset/<source>/task_reward/avg, eval/source_dataset/<source>/...
```

A field named like one of platoon's own columns, `task_id` or one that starts with `_platoon`, is left out of the
dataset, and a run whose `label_by` names it stops with exit 78: it would read platoon's column, not the field.

## Harness timeouts and crashes

A harness runs for at most the earliest of its spec's `timeout_s`, the rollout's `timeout`, and the task's own
`[agent] timeout_sec` times `agent_timeout_multiplier` (a function harness, for at most the last two). The
rollout's `timeout`, or else the other two, also set the deadline of the whole rollout ([4. Run it](#4-run-it)).
What a harness that ran out of time or crashed counts as is the environment's `on_harness_exit`:

```yaml
environments:
  - name: science
    service: harbor
    harness: platoon-bash
    on_harness_exit: {timeout: grade, crash: grade_if_gateway_clean}
```

| Rule | Effect |
|---|---|
| `timeout: exclude` (default) | A timeout is a `timeout` outcome, kept out of training |
| `timeout: grade` | The env is graded as the harness left it, with `reason: agent_timeout`, as Harbor grades after an agent timeout |
| `crash: exclude` (default) | A non-zero exit the spec does not map is an `error` (`harness_crash`) |
| `crash: grade` | Every crash is graded, with `reason: harness_crash` |
| `crash: grade_if_gateway_clean` | A crash is graded unless the model gateway failed one of the agent's calls (a sampling error or timeout, a 5xx): then it is an `error` (`harness_crash_gateway`) |

`grade` suits tasks whose agent works inside the task sandbox, where the policy's own commands (`pkill -f python`,
a process that fills memory) can end the harness: excluding every crash would hide such failures, while
`grade_if_gateway_clean` still keeps out of training the crashes that follow a failure of the gateway. It sees only
those. A crash that other infrastructure caused is graded like one the policy caused: a harness client that gave up
on a slow call and disconnected, a route to the gateway that broke, or the sandbox's memory watchdog killing the
harness. So give a harness's client a request timeout longer than the gateway's sampling timeout
(`train.sample_timeout_seconds` or `inference.sample_timeout_seconds`, else `TINKER_SAMPLE_TIMEOUT_SECONDS`, else
600 s; [the sampling timeout](train-and-eval.md#the-sampling-timeout)): a slow sample then ends as a gateway error. A
crash after the gateway refused an overflowing prompt keeps [`on_context_overflow`](troubleshooting.md#context_overflow).
Both settings apply to every environment kind.

Some harnesses send their own sampling parameters (`temperature`, `top_p`, `max_tokens`), which by default win over
the rollout config's `inference_params`. Set `rollout_config.sampling_precedence: config` to have every policy call
sample as configured: with the `temperature`, `top_p` and `top_k` of `inference_params`, or the gateway's defaults
for those it leaves unset, never the harness's own; and with its `max_completion_tokens` when it sets one. A call
keeps its own stop sequences and seed, by which its harness parses the reply, and `aux` and `judge` calls keep all
their own parameters, except the `aux` calls of a harness in a task sandbox, which sample as its policy calls do:
code the policy runs there can read the agent's key, and would otherwise choose how calls that train sample.

## Harness toolchains

A CLI harness that runs in task sandboxes should not install itself into every sandbox: that needs the network,
costs time per rollout, and lets the image decide what runs. Instead, its spec names a toolchain, a directory with
its interpreter and packages built once, which the rollout mounts read-only into each sandbox:

```toml
# in the harness spec
[install]
toolchain = "my-agent-1.0"   # a name under the toolchain root; an absolute path is used as it is
[resources]
memory_mb = 2048             # the harness's own processes: added to the sandbox's memory limit
```

The toolchain root is `toolchains` in the site profile's `cache`, else `~/.cache/platoon/toolchains`
(`platoon toolchain build --root` builds into another directory, where a run finds a toolchain only by its absolute
path). A toolchain is a directory `<root>/<name>/` with a `manifest.json`:

```json
{"name": "my-agent-1.0", "build_path": "/opt/toolchains/my-agent-1.0", "tree_hash": "sha256:..."}
```

- **`build_path`** is where the toolchain was built. Interpreters and console scripts record absolute paths when
  they are installed, so the sandbox gets the toolchain at exactly that path, read-only, and `{toolchain}` renders
  as it. Nothing in the toolchain has to be relocatable. Without a manifest, the toolchain's own path under the
  root is its build path. The build path cannot be under `/tmp` or `/var/tmp`, which sandboxes keep private.
- **`tree_hash`** is the hash of the directory's contents (`platoon.harness.toolchains.tree_hash`).

Before a rollout uses a toolchain, platoon copies it to node-local disk, once per node and toolchain version:
into a temporary directory under a file lock, checked against the tree hash, then renamed into place. Concurrent
rollouts, of one run or of several, copy it once, and none sees a partial copy; the next copy to take the lock removes
what a killed copy left half done. `platoon train` stages the toolchains of its environments' harnesses before its
first rollout, while it opens the training service and saves at launch, and `platoon eval` while it lists its tasks,
so that no rollout waits for the copy, and the first ones only for what of it outlasts the trainer's own start-up
([Rollouts are slow to start](troubleshooting.md#rollouts-are-slow-to-start)): each rollout's own staging then finds
it current and returns at once. Copies go to `$PLATOON_TOOLCHAIN_CACHE`, else
to `platoon-toolchains-<uid>` in node-local scratch: the first of `$PLATOON_NODE_SCRATCH`, `$PBS_LOCALDIR`,
`$SLURM_TMPDIR` and `$TMPDIR` that is set, else `/tmp`, where the Harbor service also keeps its per-node checks. A
toolchain holds tens of thousands of files and about 800 MB, so where none of these is node-local disk with room for
it, set `$PLATOON_NODE_SCRATCH` or `$PLATOON_TOOLCHAIN_CACHE`.

A copy in the job's scratch goes with the job, so every start and every resume in a new job copies the toolchain
again. To keep it across jobs, set `$PLATOON_TOOLCHAIN_CACHE` to a node-local directory that outlives them. The next
job on the node keeps the copy when its version and files still match: the copy's name carries the manifest's tree
hash, and the first staging of each job (each boot and scheduler job, `$PBS_JOBID`, `$SLURM_JOB_ID`, `$LSB_JOBID` or
`$JOB_ID`) hashes the copy again and compares it with the manifest's tree hash, under the lock, which takes seconds
instead of the copy's minute. A stamp beside the copy (`.<copy>.staged`) records the jobs that checked it, so that
jobs sharing the node and the copy check it once each. A copy that no longer matches, because a node's cleanup removed
some of its files, is copied again beside it, and the fresh copy takes its place; the old one is set aside whole
(`.<copy>.<hex>.stale`), since rollouts of another job may still run from it, and a warning names it. A cache platoon
cannot write to (read-only, another user's, a full disk) cannot record a check: a kept copy there is used as it is,
with a warning. Old versions and set-aside copies stay until you delete them, while no job uses them.
A run never builds a toolchain: `platoon prepare` does, or `platoon toolchain build`, once per site ([OpenHands in
task sandboxes](#1-build-the-toolchain)). Before any rollout, `platoon train`, `eval` and `rollout` refuse (exit 78) a
config whose spec's named toolchain is not built under the root the run uses, naming the `platoon prepare` command
that builds it. A
toolchain that is missing on a node all the same, or whose files do not match its manifest, makes the rollout an
`error` (`harness_setup`). With `placement: local`, `{toolchain}` is the toolchain's build path when that exists on
the host, else the local copy.

`[resources] memory_mb` matters on providers without cgroups, such as rootless Singularity: they enforce a task's
`memory_mb` with a watchdog that kills the largest process, which would otherwise be the harness.

## OpenHands in task sandboxes

The `openhands` harness runs one OpenHands SDK agent (openhands-sdk and openhands-tools 1.54.0) inside each task's
sandbox, with platoon's thin runner: the terminal, file editor and task tracker tools plus the SDK's finish and
think, and no skills or delegation, as Harbor's `openhands-sdk` adapter runs it. It installs nothing there: it runs
from a *toolchain*, a read-only directory with its own CPython and a locked set of packages, which you build once per
site. Every setting of the agent is an option of the one harness, so one toolchain serves training, evaluation and
any other setting (`platoon harness show openhands` lists them):

| Option | Default | What it does |
|---|---|---|
| `reasoning_replay` | `true` | Each assistant turn goes back with its reasoning, so the calls of an episode extend each other and merge into few training sequences; `false` sends turns back as the SDK serializes them, without it, as Harbor's adapter does |
| `request_timeout_s` | `3600` | Seconds one model call may take |
| `num_retries` | `0` | The SDK's `num_retries`, which counts attempts at a model call that fails transiently: 0 and 1 make one, 5 up to five. A retry may send a call the gateway already recorded, so training keeps 0 |
| `condenser` | `none` | `llm_summarizing` adds a summarizing condenser, the SDK's with the changes below, which condenses past 240 events, or when the gateway answers that a prompt overflows the context window |
| `condenser_max_tokens` | `0` | Above 0, a token trigger for the summarizing condenser, which it then runs whatever `condenser` says: it also condenses before a model call whose prompt would be longer than this many tokens, as the gateway counts them |

The defaults are the training settings. The step limit is the run's `max_steps` (default 30), the runner's
`--max-iterations`: one model call of the agent each, and with the condenser, one for each condensation (below).

The condenser's summaries are model calls through the gateway with the agent's key, which the runner marks as `aux`
calls (`X-Platoon-Call-Kind: aux`, on the summary calls only): the gateway records them as the agent's `aux` calls,
which train with the rollout's advantage as its turns do. A harness in a task sandbox may make `aux` calls only
without a gateway step budget ([step 3](#3-how-a-run-ends)), and `openhands` has none: its step limit is the runner's
`--max-iterations`, and each condensation takes one of those steps, since that step makes the summary call instead
of an agent call, and a call the gateway refused for overflowing the context window took one too. After a
condensation the next prompt no longer extends the last, so it starts a new training sequence.

The condenser is the SDK's `LLMSummarizingCondenser` with its defaults, changed by the runner in four ways. The
summary call reads each forgotten turn whole, with the commands it ran, their whole output and an earlier summary
entire, where the SDK's shows it 500 characters of each and no tool call's arguments. A condensation keeps the system
prompt and the task, even one that resets the whole context, which the SDK's would summarize away with the rest. A
condensation for the token trigger (below) forgets only what it takes to get under half the trigger, where the
SDK's also asks for a tenth of the turns and otherwise resets the whole context. And a summary call answered without
text, with a tool call or a completion cut short in its reasoning, still condenses, with a notice in the summary's
place that the earlier turns were dropped, where the SDK's drops them and says nothing; the runner logs it.

With `condenser_max_tokens`, the condenser counts each prompt before the agent sends it, and condenses once the count
passes the trigger (104857 condenses at 80% of a 131072-token window): it summarizes the oldest turns after the
system prompt and the task until the rest fits in half the trigger. Set the trigger well above the agent's first
prompt, its system prompt with the tools and the task: below twice that prompt, each condensation forgets every turn
after the task, and a trigger that a prompt passes with nothing left to forget but a summary can never be met, so the
condenser sets it aside for the rest of the run and says so in the harness log. The runner asks the gateway for the
count: it sends the messages and tools of the next call to the gateway's token count,
`POST /v1/messages/count_tokens`, as a Messages request that the gateway reads back into the same chat messages, so
the count is the length of the prompt the gateway renders, and nothing is recorded. That length is exact for a
renderer that writes tool results without their tool's name and serializes tool-call arguments itself, as the Qwen
chat templates do; one that writes either as the call sent it (`kimi_k2`, `gpt_oss`) counts a few tokens off for each
tool call. A hosted model's gateway forwards the count to its provider. Where the answer is no count (a provider
without that route answers 404, and the runner says so once in its log), the SDK's own estimate counts: litellm's
token counter, with the `cl100k_base` encoding for a model it does not know. That estimate runs below a Qwen chat
template's count by about a tenth on prose and code, and by nearly half on numbers, which Qwen's tokenizers split
into single digits, so a prompt the estimate undercounts can pass the trigger; the gateway's refusal of a prompt that
overflows the window still condenses it. Without the option nothing counts tokens.

### 1. Build the toolchain

On a node with network access to PyPI and [uv](https://docs.astral.sh/uv/) on the `PATH`, once per site,
`platoon prepare` builds the toolchain of every harness a config names, under the toolchain root
([Prepare a config](#prepare-a-config-once-per-site)), and `platoon toolchain build` builds one spec's:

```bash
platoon prepare configs/train.yaml
platoon toolchain build openhands --root /shared/toolchains
```

The spec's `[install.recipe]` says what to install, and the build writes `/shared/toolchains/openhands-1.54.0/`: a
uv-managed CPython 3.12.14 (the recipe pins the patch release), the packages of the lock file that ships next to the
spec, compiled to bytecode, the tiktoken encodings (`share/tiktoken`), and a `manifest.json` with the Python and
package versions, the recipe's identity and a tree hash. A build takes about a minute with a warm uv cache, and the
toolchain about 600 MB:

```text
openhands@1.54.0: built /shared/toolchains/openhands-1.54.0 in 51s: Python 3.12.14, 186 packages, 22188 files, ...
```

Building a toolchain again is a no-op while its recipe is the same: the manifest's `recipe_sha256` covers the
Python version, the lock file, the caches and the scripts that generate files. When any of them changed, or when an
older platoon built the toolchain and its manifest has no `recipe_sha256`, `platoon toolchain build` and `platoon
prepare` fail and say what differs: a run of another config, or of another platoon, may be running from that
toolchain, which each rollout looks up under the root when it starts, so a rebuild in place would change the run's
next rollouts and break those that start while the build is half done. `--force` rebuilds it in place: run it while
no job uses the toolchain. A build that fails leaves nothing behind. `--root` defaults to the toolchain root.

Options never change the toolchain: they render into the harness's launch, never into what it installs, so a new
setting needs no build. A run never builds a toolchain either, since a build needs uv, network access and minutes.
Before any rollout, `platoon train`, `eval` and `rollout` check that the toolchain is built under the root the run
uses, and otherwise stop with exit 78 and the `platoon prepare` command for their config:

```text
platoon: what the run needs is not prepared on this site:
  environments[harbor-openhands].harness: harness openhands@1.54.0.post1 runs from the toolchain openhands-1.54.0,
  which a run never builds, and it is not built there (toolchain root /shared/toolchains)
  prepare it with: platoon prepare train.yaml
```

### 2. Point the environment at the toolchain

The environment needs nothing for it: a run finds the toolchain under the site's toolchain root, where `platoon
prepare` built it. A rollout copies the toolchain to node-local disk once per node and mounts it read-only into each
task sandbox at the path it was built at, which `{toolchain}` renders ([Harness toolchains](#harness-toolchains)):

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs:
      dataset_dir: ./tasks       # the site profile names the provider: singularity, allow_network_none: true
    harness: openhands@1.54.0
    harness_options: {condenser: llm_summarizing}   # options the defaults do not cover; most configs set none

train:
  workflow_config:
    rollout_config:
      max_steps: 30          # --max-iterations of openhands
      timeout: 1800
```

An option set on the command line is checked like the config's own, so `platoon train train.yaml
environments.0.harness_options.num_retries=-1` stops with exit 78 before anything runs:

```bash
platoon train train.yaml environments.0.harness_options.reasoning_replay=false
platoon eval eval.yaml environments.0.harness_options.request_timeout_s=600
```

The harness reaches the gateway over the host network, or, in a task sandbox without a network, through the
sandbox's route ([step 3](#3-pick-where-the-harness-runs)), whose forwarder runs with the toolchain's own Python.
Task images need glibc (the toolchain's CPython is python-build-standalone), and nothing else: the toolchain brings
its interpreter. The spec asks for 2 GB of memory for its own processes (`[resources] memory_mb`), on top of the
task's.

### 3. How a run ends

| The run | Outcome |
|---|---|
| The agent calls `finish`, or answers without a tool call | graded |
| The agent reaches `--max-iterations` (exit 3), or the SDK finds it stuck in a loop (exit 4) | graded |
| The harness crashes, or its toolchain is missing on a node | `error` (`harness_crash`, `harness_setup`) |

The gateway also limits the agents a rollout may mint over its HTTP agent API (`POST /v1/platoon/agents`) to 64,
since code the policy runs in the sandbox can read the agent's key; `PLATOON_MAX_MINTED_AGENTS` sets another
limit. A minted agent shares its parent's step budget: the policy calls of the root and of every agent minted under
it count against one budget. For the same reason the gateway answers a harness in the task sandbox with a permission
error (HTTP 403), and samples nothing, for a call it marks `judge` (`X-Platoon-Call-Kind`), which never trains, and,
under a gateway step budget (`[budget] enforce = "gateway"`), for a call it marks `aux`, which that budget does not
count. Without a gateway step budget, as for `openhands`, its `aux` calls are sampled, recorded and trained as its
policy calls are, and the gateway counts neither. Harnesses on the rollout host keep their `aux` and `judge` calls.

### 4. Check a dataset first

Before training on a dataset, run its reference solutions and an agent that does nothing through the same
config, with two harnesses that make no model calls: `platoon-oracle` runs each task's `solution/solve.sh` in its
sandbox, and `platoon-noop` does nothing. A sound task scores 1 with the oracle and 0 with the no-op:

```bash
platoon eval eval.yaml environments.0.harness=platoon-oracle
platoon eval eval.yaml environments.0.harness=platoon-noop
```

As with Harbor's own oracle, the task is graded whatever the script's exit code; the exit code and the end of
its output are in the trajectory's `misc.oracle`. A script that runs past the task's agent timeout is a
`timeout`, which `on_harness_exit` decides about as for any harness ([Harness timeouts and
crashes](#harness-timeouts-and-crashes)).

### 5. Compare with an evaluation through Harbor

`openhands`'s defaults are its training settings. To compare its scores with an evaluation that runs the same agent
through Harbor's `openhands-sdk` adapter, evaluate it with the adapter's options, and sample and stop as the adapter
does. That is a config, not another harness; in an evaluation config (`platoon eval`):

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs:
      dataset_dir: ./tasks
    harness: openhands@1.54.0
    harness_options: {reasoning_replay: false, request_timeout_s: 300, num_retries: 5}

inference:
  workflow:
    rollout_config:
      max_steps: 500                     # the SDK's default step limit, which the adapter keeps
      inference_params: {temperature: 0.6}   # the adapter passes only the temperature to its model
```

In a training config, put the options under `eval_overrides`, so that the evaluations during training score as the
adapter does while training keeps the defaults. The step limit and the sampling are the evaluation phase's rollout
settings, not options:

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs: {dataset_dir: ./tasks}
    harness: openhands@1.54.0
    eval_overrides:
      harness_options: {reasoning_replay: false, request_timeout_s: 300, num_retries: 5}

eval:
  workflow_config:
    rollout_config: {max_steps: 500, inference_params: {temperature: 0.6}}
```

`eval_overrides` apply to the rollouts of every split but `train`, and `lock.json` records the options each phase
ran with, defaults included (`components[].options`).

| | The defaults, for training | The adapter's options |
|---|---|---|
| Messages | each as one string (the system prompt's two parts joined by a newline), earlier turns with their reasoning, so each call extends the last | as the SDK serializes them for the adapter (text parts as lists), earlier turns without their reasoning: each prompt is the adapter's |
| One model call | 3600 s, one attempt | 300 s, up to 5 attempts |
| Step limit | the run's `max_steps`, else 30 | `max_steps: 500` in the evaluation's rollout settings |

Both use the same tools, the SDK's default system prompt, no condenser and no skills, the task's working directory
as the workspace, and the same runner and toolchain. What still differs from an evaluation through Harbor:

- **Sampling.** The adapter sets only the temperature on its model; the model server's defaults then apply to the
  rest, where the gateway's are top_p 1 and no top_k. Set only `temperature` in `inference_params` to match.
- **Prompts and tool calls.** platoon renders each prompt with the policy's chat template and parses tool calls
  itself; a model server applies its own template and parser, which may differ in whitespace or reasoning
  handling.
- **The tests.** platoon runs `tests/test.sh` with `bash`; Harbor makes it executable and runs it, so its `#!` line
  picks the interpreter. The two differ for a test script written for another shell.

## Sandboxes left by stopped rollouts

Every sandbox a run starts on a built-in provider carries the label `platoon.run: <run directory>`, with the run
directory as an absolute path: `log_path/experiment_name/trial_name` for `platoon train`, and
`inference.output_dir` for `platoon eval`. The trainer and the evaluation set it, also for the evaluations during
training, and their rollout workers carry it beside their own label. So the label selects one run's sandboxes, and
no other run's, when several runs share a node, and a resumed run keeps it.

Each rollout worker (`rollout_isolation: process`) also labels the sandboxes it starts with `platoon.worker: <request
id>`, and keeps a file next to its other worker files that says how to reap them (the provider and its options).

- **A worker that is told to stop** (SIGTERM from its trainer or from the supervisor's straggler kill) starts no
  sandbox from then on, lets its sessions' teardowns run, and then stops whatever sandbox its rollout still holds.
- **A worker that is killed** (SIGKILL, its hard-timeout alarm, its memory limit), or that ran out of time while
  stopping, leaves its sandboxes behind. Its supervisor reaps them by the worker's label as soon as the worker is
  gone: hpc-sandbox's janitor stops the sessions whose owner process is gone, Docker removes the labelled stacks (their
  compose projects, networks and directories), then the labelled containers with their anonymous volumes (and snapshot
  images), and the `process` provider kills the labelled sandboxes' commands and deletes their directories.

`platoon eval` with `use_subprocesses: true` does the same for its worker processes. Each labels its sandboxes with
`platoon.worker: eval-<id>` and lists how to reap them in a private file.

- **A rollout past its hard timeout** starts no sandbox from then on and stops the sandboxes it holds, within 30
  seconds, before it kills its process group. A rollout whose event loop is blocked is killed 35 seconds after the
  timeout anyway.
- **A worker process that was killed or died, or whose rollout was cancelled,** leaves its sandboxes behind. The
  evaluation reaps them by that worker's label once the process is gone, and reaps whatever the worker processes
  left when it ends.

Every built-in provider has `reap(labels)`, which you can call yourself, for example at the end of a job, to reap
whatever its run left, after a crash too. Build the provider with the environment's `provider_options`, so that it
looks where the sandboxes were started, and give it the run's label:

```python
import asyncio
import os

from platoon.sandbox import get_provider

run_dir = os.path.abspath("runs/my-experiment/my-trial")  # log_path/experiment_name/trial_name
provider = get_provider("singularity", scratch_root="/scratch/hpc-sandbox", allow_network_none=True)
print(asyncio.run(provider.reap({"platoon.run": run_dir})), "sandboxes reaped")
```

The hpc-sandbox providers and `process` leave alone the sandboxes of processes that are still running, so a reap
while the run still runs takes only what its stopped workers left. Docker cannot tell, so reap a run's label there
once the run has stopped.

hpc-sandbox (0.3 on) deletes a stopped sandbox's files in a background process once the stop has returned, and platoon
does not wait for these deletions when it exits, so the end of the batch job kills those still running. What they
leave in the provider's `scratch_root` is deleted by the next sweep of that directory on that node: a provider's first
start there (which deletes at most 64 such entries), a reap, or `python -m hpc_sandbox janitor`. A sweep leaves alone a
deletion whose process still runs, so a janitor that the job script runs once platoon has exited does not finish them.
Nothing more is needed where the site clears that directory after each job, such as hpc-sandbox's default scratch root
in a job's own local directory. Where it outlives jobs, such as a `scratch_root` on the node's local disk
([Run on a cluster](clusters.md)), start the job script with the janitor, before platoon: it deletes what earlier jobs
left on that node, and leaves alone what other running jobs use.

```bash
python -m hpc_sandbox janitor --scratch-root /local/scratch/sandboxes  # the provider's scratch_root
```

Run it from the job script, not from an epilogue: an epilogue runs as root, and its janitor sweeps root's sessions,
not yours.

## What platoon refuses

platoon refuses a task it cannot run faithfully instead of running part of it and training on a wrong reward. A
dataset skips such a task with a warning; naming it directly (as a task id, in a split manifest or with
`platoon rollout --task`) fails with the reason.

| Task | Why |
|---|---|
| A multi-container environment (`environment/docker-compose.yaml` with sidecar services) on a provider that runs no stacks (`singularity` and `enroot` with an hpc-sandbox older than 0.5, or on a node that does not nest user namespaces), or with `stacks: refuse` (a fetched dataset's when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78; a dataset directory skips them with a warning) | The main container alone would lack its sidecar services (a judge, a database), and its tests would score 0 for work the policy may have done. Where a provider runs stacks, they run ([Multi-container tasks](#multi-container-tasks)); a compose file that names only `main` runs as one sandbox on every provider ([Compose files](#compose-files)); the other compose names (`compose.yaml`, ...) are plain files, as in Harbor |
| A stack with a service that a node service serves (an overlay's `external`), whose services depend on one another in a cycle, with a sidecar's read-only bind of a log directory, or with binds of one path of the task that services would share in Docker and platoon would copy apart (when the task loads) | platoon runs no node services yet, compose refuses a cycle, a sidecar does not see `main`'s log directories, and the services must see one another's writes as in Docker |
| A stack that no rollout could start: a dataset directory's sidecar of a registry image that its compose file gives no command, or that builds from a context; a fetched dataset's sidecar whose built image or pin the image map lacks; a `service_healthy` dependency on a service without a healthcheck (a dataset directory's when the task loads, for the first two; a fetched dataset's when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, naming `platoon prepare` for what it makes) | A registry image's own command is what `platoon prepare` pins, and a sidecar's build context what it builds, in a fetched dataset's image map, which a dataset directory has none of ([Multi-container tasks](#multi-container-tasks)) |
| A fetched stack whose services ask for privileges (`refuse: {privileges: false}` kept them) on a provider other than `docker` (when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, and when a session opens) | Only Docker gives a service every privilege, or a capability such as `NET_ADMIN` over its stack's network ([Stacks on Docker](#stacks-on-docker)) |
| A compose file outside the subset platoon runs (when the dataset is fetched, as `compose`; `privileged` and a `cap_add` of `NET_ADMIN` included, unless the fetch keeps them for Docker), or one that gives `main` what platoon cannot apply to it yet, such as `cap_add`, `privileged` or `shm_size` in a main-only file (when the task loads) | platoon runs a compose file as written or not at all ([Compose files](#compose-files)): a dataset's overlay may drop or replace what it refuses |
| A separate verifier built from `tests/docker-compose.yaml`; `[[verifier.collect]]` hooks or artifacts of a service the task does not start | A verifier that is itself a stack is not supported; the others name nothing that runs |
| `[agent].user` or `[verifier].user` (or a `[[verifier.collect]]` hook's `user`), unless the service sets `agent_user` (for `[verifier].user`, `verifier_user`) to `ignore` or `honor` | A rootless sandbox runs every process as you, unless it gives root and other users ([Users in task sandboxes](#users-in-task-sandboxes)) |
| With `honor`: a multi-container stack in which users would run (under `agent_user: honor` every stack; with `verifier_user: honor`, one graded in `main` that names a `[verifier].user`, or whose collect hook names a user), or a root verifier of a task graded in its agents' own sandbox whose agents are not honored (a fetched dataset's when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78; a dataset directory skips them with a warning) | A stack's `main` runs no command as a user that the task names yet, as a compose project or as a pod; a root verifier's agents would run as root too ([Verifiers that run as root](#verifiers-that-run-as-root)) |
| With `honor`: a task whose agents' sandbox, or separate verifier's, runs its users where the provider's options cannot give root and other users, such as `singularity` without `fakeroot` or on fuse-overlay, `enroot` or `process` (when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, and when a session opens) | Its commands could not run as their users, and tests that run the agent's code as another user would score every attempt 0 |
| With `honor` on Singularity: a task whose sandbox runs a user other than root, or anything as its image's `USER`, on an image whose owner manifest the image map lacks (a dataset directory's task, or a local image file, among them), or, with an hpc-sandbox before 0.5, on any image that gives files to users other than root (when the tasks are listed, with exit 78) | Those users would find root's files where Docker gives them their own, and the image's `USER` is known from that record alone: `platoon prepare` records the manifests, and hpc-sandbox 0.5 applies them ([Users in task sandboxes](#users-in-task-sandboxes)) |
| With `honor`: a fetched dataset's task whose sandbox honors users on an image that the `oci` builder of platoon before 0.3.0 built (when the tasks are listed, with exit 78; `platoon prepare` fails too) | On a host whose umask is 027, its files keep out the users other than root. Build the image again (`platoon harbor images build CONFIG --key KEY`) |
| `gpus` above 0 on a provider that cannot give a sandbox GPUs (when the session opens) | The tests would fail for want of a GPU and score 0 |
| `network_mode = "no-network"`, for the agents or a separate verifier, on `singularity` without `allow_network_none: true`, or on `enroot` with `allow_network_none: false` (when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, and when a session opens) | hpc-sandbox would refuse every such sandbox, so every rollout of the task would fail at sandbox start and the task would never train. Set `allow_network_none: true` where the site lets users create network namespaces (enroot needs no such permission: drop `allow_network_none: false` from its options), or leave the tasks out (`exclude_tasks`) |
| Any task, on `singularity` with a `rootfs_layout` that cannot run (an unknown name; any name, `copy` included, with an hpc-sandbox that has no layouts; `fuse-overlay` without `fuse-overlayfs` or `fusermount3` on `PATH`, or with `gpu`, `cgroup_limits` or `fakeroot`), or with `fakeroot` on an hpc-sandbox that does not take it, when the tasks are listed; `verifier_provider_options` are checked the same way for tasks with a separate verifier | Every sandbox start would fail. Fix the option or the `PATH`, or leave `rootfs_layout` out |
| Multi-step tasks (`steps` in `task.toml`) | Not supported |
| Non-Linux tasks | Not supported |

The warning names the reason:

```text
WARNING | platoon.envs.harbor.dataset | Skipping Harbor task .../tasks/needs-a-database: .../tasks/needs-a-database:
environment/docker-compose.yaml makes the environment a multi-container stack (sidecar services), and the singularity
provider runs no multi-container stacks; the task is refused, never started without them
```

Some task settings are read but not enforced: an allowlist network runs with the host network, and `storage_mb`
is ignored. `environment/Dockerfile` is built only through an image map ([Build task images](#build-task-images));
without one, a task runs its `docker_image` or `default_image`. A `docker_image` that is a `.sif` or `.sqsh` file, or
an absolute path, is used as written; any other reference without a scheme gets `docker://`.

## What comes next

[RFC 0010](../rfcs/0010-harbor-registry-and-multi-container-tasks.md) is the design behind this page. What it proposes
beyond what is described here: serving a task's own MCP servers through the tool plane, running stacks on the host
network on sites without unprivileged user namespaces, and stacks whose services run as the users a task names. Until
then, multi-container tasks run as `docker compose` projects, as rootless pods on Singularity, Apptainer and enroot,
and on the `process` provider ([Multi-container tasks](#multi-container-tasks)), main-only compose files on every
provider ([Compose files](#compose-files)), and a task's users in single sandboxes ([Users in task
sandboxes](#users-in-task-sandboxes)). The [RFC index](../rfcs/README.md) lists every proposal and explains how one
moves from draft to implemented; comment on the RFC, or open a new one, to shape what gets built.

## Next steps

- [Run on a cluster](clusters.md): Singularity/Apptainer and enroot on PBS and Slurm.
- [Troubleshooting](troubleshooting.md): `env_busy`, harness exit codes, and missing or rejected reward files.
- [Train and evaluate](train-and-eval.md): reports, checkpoints and metrics.
- [Write a sandbox provider](../extending/sandbox-providers.md): sandboxes on infrastructure of your own.
