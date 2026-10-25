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
| `docker` | One long-lived container per sandbox | `docker://repo:tag` or a plain reference | the `docker` CLI (`binary: podman` for Podman) | `binary`, `cli_timeout_s` (600; bounds `docker run` including the pull) |
| `singularity` | Rootless SingularityCE or Apptainer instances: private PID and IPC namespaces, root filesystem, `/tmp` and environment | `docker://` references, also pinned by digest (`docker://repo@sha256:<digest>`), or local `.sif` files | the `hpc` extra | passed to hpc-sandbox: `scratch_root`, `image_cache`, `allow_network_none`, `gpu`, `rootfs_layout` ([below](#the-singularity-providers-root-filesystem)), `fakeroot` (since 0.3.0, with hpc-sandbox 0.4: [root in a verifier](#verifiers-that-run-as-root)), ... |
| `enroot` | Rootless enroot containers in their own user, mount, PID and IPC namespaces | `docker://` references, also pinned by digest, or local `.sqsh` files | the `hpc` extra | passed to hpc-sandbox: `scratch_root`, `image_cache`, `gpus`, `remap_root`, ...; the singularity names also work (below) |

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
installs 0.4 ([The hpc extra](../getting-started/install.md#the-hpc-extra)); with an older one, every sandbox gets a
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
platoon reads, `dataset`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays` and `mounts[].source`,
are relative to the config file that names them, so `./tasks` is the `tasks` directory beside the config, wherever
platoon runs from ([Write and override configs](configs.md#2-build-on-it-with-extends)):

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
| `agent_user` | `refuse` | A task that sets `[agent].user` or `[verifier].user`: `refuse` it, or `ignore` the user and run as the sandbox's single uid (recorded in the outcome's `ignored_users`) |
| `verifier_user` | as `agent_user` | *Since 0.3.0.* What `[verifier].user` gets instead, when set: `refuse`, `ignore`, or `honor`, which runs a separate verifier whose user is root as root ([Verifiers that run as root](#verifiers-that-run-as-root)) |
| `verifier_provider_options` | none: `provider_options` | *Since 0.3.0.* The provider options of every separate verifier's sandbox, laid over `provider_options`, such as `{rootfs_layout: copy, fakeroot: true}` for verifiers that run as root |
| `concurrent_tools` | none | Tools that may overlap other calls on the same sandbox (`bash`, `view`) |
| `queue_timeout_s` | 240 | How long a call or a grade waits for its turn before it is refused (`env_busy`) |
| `close_timeout_s` | 30 | How long closing waits for calls in flight before it stops the sandbox |
| `max_artifact_bytes` | 1 GiB | The largest artifact a separate verifier receives; a larger one is left out, and the tests decide what its absence means |
| `sandbox_start_budget_s` | 600 | What a rollout's deadline allows each sandbox start, the task's and a separate verifier's ([Run it](#4-run-it)) |

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
2. **The images** that its tasks build from Dockerfiles, and its image map lacks, are built with the builder of its
   sandbox provider and recorded in its image map ([Build task images](#build-task-images)): `oci` for
   `singularity`, `docker` for `docker`. The `process` provider runs no image, so their keys are recorded as
   `local`. For any other provider, which pulls its images from a registry, the step fails and names the command
   that builds and pushes them. An entry that serves a directory builds nothing: its tasks run their own images.
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
fetched dataset's tasks build images that their image map lacks (checked when the tasks are listed, which takes
seconds once the lock exists). A toolchain built from another recipe stops them too, naming the forced build
instead. `platoon train` and `eval` still fetch a dataset pinned to a commit that no lock answers, when they list its
tasks; `platoon rollout`, which runs one task, stops instead.

```text
platoon: what the run needs is not prepared on this site:
  environments[science].service: 5,584 of the 5,584 tasks of split 'train' build their images from Dockerfiles, which no image map holds (the default image map /shared/platoon-cache/harbor/images/singularity.json does not exist): 2 image(s), such as sha256:e879... for 4,439 task(s) (fineenvs--...) and 1 more; a run never builds them
  prepare it with: platoon prepare configs/train.yaml
```

The `platoon harbor` and `platoon toolchain` commands remain for what needs their flags: `--shard`, `--push`,
`--key`, `--builder`, `--repair`, `--update` and `images alias`.

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
a lock: a Hugging Face dataset repository at a commit (`hf://org/name@<commit>[#subdir]`), or a local directory.
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
      refuse: {sidecars: true, gpus: true, agent_user: true, per_task_build: true, uploads_over_mb: 100}
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
if it does: what `refuse` names (`stack`, `gpus`, `agent_user`, `per_task_build`, `uploads_over_mb`), and anything
platoon's task loader cannot run (`task_config`: a bad `network_mode`, a separate
verifier built from `tests/docker-compose.yaml`, an artifact path with `..` or from another compose service, and the
rest of [What platoon refuses](#what-platoon-refuses)), which would fail every rollout. Sidecar stacks stay refused
with `sidecars: false` too, as `task_config`, until platoon runs stacks. A refused task stays in the lock with its
reason: it is never listed, and a split manifest that names it is an error. `refuse`'s switches take `true` or
`false`; anything else is an error.

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
| `dataset` | A source to fetch (`hf://org/name@<commit>` or a directory), with `tasks`, `select`, `overlays`, `refuse` and `index` choosing and editing what is fetched |
| `lock` | The lock to read (and, with `dataset`, to write; default `<Harbor cache>/locks/<hash of the fetch settings>.lock.json`) |

The Harbor cache, which holds the tasks' files, the locks and the image maps, is `harbor` in the site profile's
`cache`, else `~/.cache/platoon/harbor` ([Write the site's profile](clusters.md#4-write-the-sites-profile-once)).

`platoon train` and `platoon eval` fetch a `dataset` pinned to a commit when no lock lists it yet, in the process that
lists the tasks; a moving reference (a branch, or no revision) is fetched only by `platoon harbor fetch --update`. A
local directory is copied into the cache when it is fetched: run `platoon harbor fetch` again after changing it.
For a dataset pinned to a commit, `platoon harbor fetch` does nothing when the lock answers the config and the cache
holds a directory for each of its tasks; `--repair` fetches anyway, and so does a fetch of a lock written before
platoon recorded `input_files`, once. A fetch hashes each cache entry it would reuse
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

## Build task images

Tasks that build from `environment/Dockerfile` run images built once per site. platoon splits each Dockerfile at its
last step that must run at build time (a `RUN`, a `FROM`, a `COPY --from`, ...):

- the **build prefix** is what gets built. Its **build key** is a SHA-256 over the prefix and the build-context files
  it reads, so tasks whose prefixes match share one image however their last steps differ;
- the **deferred suffix**, the trailing `COPY` and `ADD` of local files, `WORKDIR`, `ENV` and metadata, is applied
  when a session opens: the files are uploaded, the directories created and the variables set, with the result a
  build would have had. A `USER` there is refused, since rootless sandboxes run as one uid.

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
platoon harbor images verify configs/train.yaml
platoon harbor images plan locks/my-tasks.lock.json --image-map images/site.json   # a lock file and a map of yours
```

Each takes a training or evaluation config, with overrides, and acts on each Harbor entry's lock (the one that
answers its `dataset`, or the `lock` it names) with the entry's image map, `images/<provider>.json` in its Harbor
cache. `build` uses the builder of the entry's sandbox provider, `oci` for `singularity` and `docker` for `docker`
(`--builder` names another), and puts the image files in `images/<provider>/` next to the entry's map, or, for a
lock file, in `images/` next to the map `--image-map` names (`--out` names another directory). Builds of one map at
once, also on several nodes, take turns on each image and build it once. The commands also take a lock file, with
`--image-map` (and, for `build`, `--cache`).

`plan` lists one image per build key, with its task count, base images and whether the map has it, plus the
prebuilt images tasks name in `docker_image`; an entry whose image file is gone, or is not the size its build
recorded (a truncated copy), counts as missing. `build` builds
the missing ones from a representative task and adds each to the image map as soon as it is built, with a receipt:
the builder and its version, the build time, the log, the image's SHA-256, the base images as the Dockerfile names
them (`base`) and the digest each one resolved to (`base_digests`). A build asks each base image's registry for the
digest its tag points to and builds from `name@digest`, so the receipt says exactly what the image was built from;
the build key stays the one the lock recorded. A base whose digest cannot be resolved (no network, a private
registry) is built as written, with a warning, and its digest in the receipt is `null`.

`verify` checks the map against the lock: every image the lock's runnable tasks use must be in the map, and each
local image file (`.sif`, `.sqsh`, ...) must exist and hash to the SHA-256 its entry records. It prints one line per
missing or mismatched image and exits 1 if there is any. Registry references cannot be hashed here; they are listed
as not checked and do not fail the check.

| Builder | How | Where |
|---|---|---|
| `oci` (default) | `singularity build --oci` runs the Dockerfile with rootless BuildKit, as a Docker build does: multi-stage builds, step flags, here-documents, and an `ADD` of a URL (verifying its `--checksum`) or of a git repository; its single squashfs layer is unpacked and rebuilt as a native SIF, and the image config's `ENV`, `WORKDIR`, `USER`, `ENTRYPOINT` and `CMD` go into the map. SingularityCE's BuildKit cannot load the frontend image a `# syntax=` directive names, and builds with its own Dockerfile frontend instead: a directive that names a 1.x release of Docker's (`# syntax=docker/dockerfile:1.7`) is left out, and one that names another frontend (labs, a third party's) is refused | SingularityCE 4.1 or later, `unsquashfs`; the network, for base images and remote `ADD` sources |
| `def` | a translation to a `Bootstrap: docker` definition file built with `--fakeroot`, in which each `RUN` runs in a shell of its own that starts in the current `WORKDIR` and sees the `ARG` values, as in a Docker build; it refuses what it cannot translate faithfully (multi-stage builds, `ARG` without a default, `USER`, `SHELL`, step flags, here-documents, an `ADD` of a URL, a git repository or an archive), naming the `oci` builder | Singularity or Apptainer with fakeroot |
| `docker` | `docker build`, tagged `platoon-harbor:<key>` for the `docker` provider; with `--push REPO`, pushed and recorded by digest (enroot sites import that digest) | Docker or Podman (`PLATOON_DOCKER=podman`) |

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

## Verifiers that run as root

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

On Singularity this needs an hpc-sandbox whose provider takes `fakeroot` (below): 0.4, which the `hpc` extra
installs. With an older one, such as 0.3.0, a run with this config stops when it lists its tasks, with a message
that says what is missing.

- **What `honor` takes.** A `[verifier].user` of root (`root`, `0`, `0:0` or `root:root`) in a task with a
  [separate verifier](#a-separate-verifier). Any other user is refused, and so is a root verifier of a task graded in
  its agents' own sandbox, whose agents would then run as root too: set `verifier_user: ignore` for them, or leave
  them out (`exclude_tasks`). `agent_user` still decides for `[agent].user`, and without `verifier_user` it decides
  for both users, as before.
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
  hpc-sandbox 0.4 makes it, and its missing parents, with these modes too, as Docker does. A directory the image has
  keeps its mode, as in Docker: a workdir such as `/root`, which images keep to root (700), keeps the files the tests
  write there from the users they switch to. The artifacts are made readable by every user (`chmod -R a+rX`), as an
  agent's files are under Docker's umask, and the outcome's metadata records `verifier_user: root`. Files that platoon
  uploads otherwise keep their modes from the host: `tests/`, when the verifier's image does not hold it, and the
  files of an image's deferred `COPY` steps, with the directories that the upload makes for them.
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
Dockerfile. A file that no step takes, the `Dockerfile` itself, the tests and the solution are not inputs, and an
empty file (or one of blank lines) joins nothing. A fetch records each task's input files in the lock
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
  gone: hpc-sandbox's janitor stops the sessions whose owner process is gone, Docker removes the labelled containers
  (and snapshot images), and the `process` provider kills the labelled sandboxes' commands and deletes their
  directories.

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
| A multi-container environment: `environment/docker-compose.yaml` (or `.yml`, `compose.yaml`, `compose.yml`) with sidecar services | The main container alone would lack its sidecar services (a judge, a database), and its tests would score 0 for work the policy may have done. A compose file that only names `main`, with no keys (`services: {main: {}}`), is one container and runs |
| A separate verifier built from `tests/docker-compose.yaml`, `[[verifier.collect]]` hooks, or artifacts of another compose service | They need multi-container stacks |
| `[agent].user` or `[verifier].user`, unless the service sets `agent_user: ignore` (or, for `[verifier].user`, `verifier_user: ignore` or `honor`) | Rootless sandboxes run every process as one user |
| With `verifier_user: honor`: a `[verifier].user` other than root, or a root one of a task graded in its agents' own sandbox (a fetched dataset's when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78; a dataset directory skips them with a warning) | platoon runs a verifier as root or as the sandbox's own user, and only in a separate verifier's sandbox, whose agents do not run as root ([Verifiers that run as root](#verifiers-that-run-as-root)) |
| With `verifier_user: honor`: a root verifier on verifier sandboxes that cannot give root, such as `singularity` without `fakeroot`, `enroot` or `process` (when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, and when a session opens) | The tests could not run the agent's code as another user, and would score every attempt 0 |
| With `verifier_user: honor`: a fetched dataset's root verifier on an image that the `oci` builder of platoon before 0.3.0 built (when the tasks are listed, with exit 78; `platoon prepare` fails too) | On a host whose umask is 027, its files keep out the users the tests switch to. Build the image again (`platoon harbor images build CONFIG --key KEY`) |
| `gpus` above 0 on a provider that cannot give a sandbox GPUs (when the session opens) | The tests would fail for want of a GPU and score 0 |
| `network_mode = "no-network"`, for the agents or a separate verifier, on `singularity` without `allow_network_none: true`, or on `enroot` with `allow_network_none: false` (when the tasks are listed, which stops `platoon train` and `platoon eval` at startup with exit 78, and when a session opens) | hpc-sandbox would refuse every such sandbox, so every rollout of the task would fail at sandbox start and the task would never train. Set `allow_network_none: true` where the site lets users create network namespaces (enroot needs no such permission: drop `allow_network_none: false` from its options), or leave the tasks out (`exclude_tasks`) |
| Any task, on `singularity` with a `rootfs_layout` that cannot run (an unknown name; any name, `copy` included, with an hpc-sandbox that has no layouts; `fuse-overlay` without `fuse-overlayfs` or `fusermount3` on `PATH`, or with `gpu`, `cgroup_limits` or `fakeroot`), or with `fakeroot` on an hpc-sandbox that does not take it, when the tasks are listed; `verifier_provider_options` are checked the same way for tasks with a separate verifier | Every sandbox start would fail. Fix the option or the `PATH`, or leave `rootfs_layout` out |
| Multi-step tasks (`steps` in `task.toml`) | Not supported |
| Non-Linux tasks | Not supported |

The warning names the reason:

```text
WARNING | platoon.envs.harbor.dataset | Skipping Harbor task .../tasks/needs-a-database: .../tasks/needs-a-database:
environment/docker-compose.yaml makes the environment a multi-container stack (sidecar services), which platoon
cannot run yet (RFC 0010); the task is refused, never started without them
```

Some task settings are read but not enforced: an allowlist network runs with the host network, and `storage_mb`
is ignored. `environment/Dockerfile` is built only through an image map ([Build task images](#build-task-images));
without one, a task runs its `docker_image` or `default_image`. A `docker_image` that is a `.sif` or `.sqsh` file, or
an absolute path, is used as written; any other reference without a scheme gets `docker://`.

## What comes next

[RFC 0010](../rfcs/0010-harbor-registry-and-multi-container-tasks.md) is the design behind this page. What it proposes
beyond what is described here: fetching datasets from the Harbor registry (`harbor://`) and from git repositories, and
running multi-container tasks, through `docker compose` where Docker exists and as rootless pods on Singularity,
Apptainer and enroot. Until then, multi-container tasks stay refused. The [RFC index](../rfcs/README.md) lists every
proposal and explains how one moves from draft to implemented; comment on the RFC, or open a new one, to shape what gets
built.

## Next steps

- [Run on a cluster](clusters.md): Singularity/Apptainer and enroot on PBS and Slurm.
- [Troubleshooting](troubleshooting.md): `env_busy`, harness exit codes, and missing or rejected reward files.
- [Train and evaluate](train-and-eval.md): reports, checkpoints and metrics.
- [Write a sandbox provider](../extending/sandbox-providers.md): sandboxes on infrastructure of your own.
