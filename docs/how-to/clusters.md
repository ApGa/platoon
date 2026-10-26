# Run on a cluster

This guide shows how to run platoon on a shared cluster with a batch scheduler, PBS or Slurm. Nothing in platoon
is tied to a scheduler or a site: a run is a few long-lived processes that find each other by URL, and you start
each one as an ordinary batch job.

**Goal:** a training run whose training service, environment servers and platoon client each run as scheduler
jobs, and that continues across walltime limits by chaining jobs.

**Prerequisites:**

- platoon installed in a virtual environment on shared storage that compute nodes can read
  (`pip install 'platoon[tinker,ors]'`, or `uv sync --extra tinker --extra ors` in a checkout, plus the `hpc` extra
  for container sandboxes: [The hpc extra](../getting-started/install.md#the-hpc-extra)).
- A config that works on one machine, checked with `platoon rollout` ([Train and evaluate](train-and-eval.md)).
- A Tinker-compatible training service you can start on GPU nodes (for example
  [miles-tinker](https://github.com/neulab/miles-tinker)), or access to a hosted one.

## The pieces of a run

| Piece | Runs on | Started by | Found through |
|---|---|---|---|
| Training service (Tinker API) | GPU nodes, or hosted | its own launcher, as a GPU job | `tinker_base_url` in the config, or `TINKER_BASE_URL` |
| Environment servers (ORS), optional | CPU nodes | your environment's own server entry point, as a CPU job or job array | `OPENREWARD_SESSION_URLS_<LABEL>`, or hosts in the service URI |
| platoon client (`platoon train` / `platoon eval`) | one CPU node | a CPU job | |

The client needs no GPU. It renders prompts, runs the rollouts (each rollout serves a model gateway on loopback and
runs its harness), builds the training data and calls the training service. Size its job for concurrency: CPU
cores and memory for `train.max_concurrent_rollouts` rollouts at once (`batch_size × group_size` by default), and
`eval.max_concurrent_rollouts` more while an evaluation runs beside training, plus their sandboxes. Environments that
run in process (`python://...`) and Harbor sandboxes run on the client's node; environment servers
(`ors+http://...`) move the environment's work to other nodes.

## Steps

### 1. Start the training service

Start the service on GPU nodes with its own launcher and wait until it listens. Then hand its URL to the client,
in the client's job script or as an override:

```bash
export TINKER_BASE_URL=http://gpu-node-17:8000        # or: platoon train ... tinker_base_url=http://gpu-node-17:8000
export TINKER_API_KEY="$(cat ~/.config/tinker/key)"   # keep secrets out of job scripts and config files
```

A training config's `tinker_base_url`, when set, comes first for every command that runs that config
(`platoon train`, and `platoon rollout` of the config); `platoon eval` reads only `TINKER_BASE_URL`, since
evaluation configs have no such key. With hosted Tinker, set only the API key.

### 2. Start the environment servers (if you use them)

An `environments` entry whose `service` is `ors+http:///EnvName` (no hosts) takes its server pool from a variable,
read each time a rollout opens a session:

| Variable | Used when |
|---|---|
| `OPENREWARD_SESSION_URLS_<LABEL>` | Always first. `<LABEL>` is the entry's label upper-cased, with every other character as `_`: the `label=` query or service option, else the environment name in the URI (`ors+http:///MyEnv` reads `OPENREWARD_SESSION_URLS_MYENV`). `session_urls_env_var` names another variable |
| `OPENREWARD_SESSION_URLS` | The labeled variable is unset, and `legacy_global_pool` is true (the default) |
| `OPENREWARD_SESSION_URL`, else `http://localhost:8080` | Neither pool is set |

Each value is a comma-separated list of `host:port` or URLs. Each rollout picks one server by a hash of its rollout
id, so load spreads across the pool and every call of one rollout goes to the same server.

A pattern that works with any scheduler: each server job writes its address to a file on shared storage, and the
client job reads the file into the variable before it starts platoon.

```bash
# in each environment-server job (one per array task)
PORT=8080
echo "$(hostname):${PORT}" >> "$RUN_DIR/env-servers.txt"
exec <your environment's server command> --port "$PORT"
```

```bash
# in the client job, before platoon starts
export OPENREWARD_SESSION_URLS_MYENV="$(paste -sd, "$RUN_DIR/env-servers.txt")"
```

In a mixture of several remote environments, give every entry its own variable and set
`legacy_global_pool: false` in its `service_kwargs`, so no entry falls back to a pool that serves another
environment; `platoon train`, `eval` and `rollout` refuse a mixture whose entries would reach one server under
different env names. You can also skip the variables and list the hosts in the URI:
`ors+http://cpu-node-3:8080,cpu-node-4:8080/MyEnv`.

### 3. Prepare the client's node

Compute nodes often have no internet access. The client loads the model's tokenizer files from Hugging Face
(and, with the `chat_template` renderer, its `generation_config.json`), so download them once on a node that has
access, into a cache the compute nodes share, and run offline:

```bash
# once, on a login or transfer node: the model's JSON and text files, no weights
HF_HOME=/shared/hf-cache uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-4B-Instruct-2507', allow_patterns=['*.json', '*.txt'])"
```

```bash
# in the client job
export HF_HOME=/shared/hf-cache HF_HUB_OFFLINE=1
```

platoon keeps its temporary files, such as rollout workers' files, gateway sockets, sandbox state and staged
toolchains, in node-local scratch: the first of `$PLATOON_NODE_SCRATCH`, `$PBS_LOCALDIR`, `$SLURM_TMPDIR` and
`$TMPDIR` that is set, else `/tmp`, in directories of each user's own. A job needs to move nothing off a small `/tmp`
by hand; set `PLATOON_NODE_SCRATCH` where none of these names node-local disk with room.

For metrics without network access, set `stats.wandb.mode=offline` (sync later with `wandb sync`) or `disabled`.

### 4. Write the site's profile, once

Where a site keeps platoon's files, and which container runtime starts its sandboxes, are facts of the site, not of
a project. They go in one file per site, the site profile, which every config run there takes them from:

```yaml
# ~/.config/platoon/site.yaml, or the file $PLATOON_SITE names: written once per site, never part of a project
cache: /shared/$USER/platoon-cache          # what platoon prepare makes: datasets, images, harness toolchains
runs: /shared/$USER/platoon-runs            # the run directories of configs that name no log_path
provider: singularity                       # or enroot, or docker: the sandbox runtime of Harbor entries
provider_options:
  image_cache: /shared/container-cache      # pull each image once for every job
  scratch_root: /local/scratch/$USER/sandboxes   # node-local disk for sandbox filesystems, one per user
  allow_network_none: true                  # where the site permits sandboxes without a network
```

- **Which file.** `$PLATOON_SITE`, else `$XDG_CONFIG_HOME/platoon/site.yaml` (`~/.config/platoon/site.yaml`), else
  none. Export `PLATOON_SITE` in a job script to use a group's file. `~` and `$VAR` expand in every value, so one file
  serves a whole group, and a relative path is relative to the file. Give each user directories of their own there,
  with `$USER`: the runs, and the sandboxes' `scratch_root`, which the HPC providers refuse when another user owns it.
- **`cache`** is the root under which `platoon prepare` fetches datasets (`<cache>/harbor`), builds their images and
  builds harness toolchains (`<cache>/toolchains`), and where runs find them. Without it, `~/.cache/platoon`.
- **`runs`** holds the run directory of a training config that names no `log_path`:
  `<runs>/<stats.experiment_name>/<stats.trial_name>`, its experiment named after the config file unless the config
  names one. A run directory named so belongs to that config file, which its `lock.json` records: another project's
  file of the same name is refused (exit 78) rather than resume its run, until the config names its experiment. An
  evaluation config without `inference.output_dir` writes each start to a new directory,
  `<runs>/eval/<policy>/<UTC time>`. A `log_path` or `inference.output_dir` the config names is never moved: a
  relative one stays relative to the working directory.
- **`provider` and `provider_options`** start the sandboxes of every Harbor entry that names no provider, and an entry's
  own `provider_options` are laid over the profile's. An entry that names a provider keeps its own options.

**What it may not say.** The profile chooses where files live and which isolating runtime starts sandboxes; it never
weakens isolation and never changes what trains:
- its provider must isolate its sandboxes: `docker`, `singularity` or `enroot`. `process`, which runs the agents'
  commands on the host, is refused there, and `platoon train` refuses an entry on it unless the run config itself
  names it;
- its `provider_options` are the ones that do not weaken isolation: `allow_network_none`, `rootfs_layout`,
  `scratch_root`, `image_cache`, `stage_images` and `gpu`. `binary`, `extra_start_args` and bind mounts (`mounts`)
  belong in a run config;
- it takes no other key: no dataset, model, reward or recipe setting.

`platoon train`, `eval`, `rollout` and `prepare` print the profile they used and what it gave the config, and refuse
(exit 78) a provider from it that the node cannot run, quoting the profile. The run's `lock.json` records the file's
path and SHA-256 and each Harbor entry's provider and options with where they came from (`site`,
`sandbox_providers`), and `config.resolved.yaml` holds the values and names each entry's provider, so the run
directory loads again without the profile, and a supervised restart keeps them however the profile changes. The
variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, which named the two cache directories before the profile's
`cache` did, are gone since platoon 0.3.0: a command that loads a config, or takes the cache from the site, refuses
to run while one of them is set (exit 78), so unset them in the job's environment and put the directory they named in
the profile's `cache`.

### 5. Choose a container runtime for sandboxes

Clusters rarely allow Docker. platoon's HPC providers run rootless containers through the `hpc` extra:

| Runtime on the cluster | Provider | Images |
|---|---|---|
| SingularityCE or Apptainer | `provider: singularity` | `docker://` references, or `.sif` files |
| enroot (often with the pyxis plugin on Slurm) | `provider: enroot` | `docker://` references, or `.sqsh` files |

Name it in the site profile (step 4), and leave it out of the configs, which then run on any site:

```yaml
environments:
  - name: harbor
    service: harbor
    service_kwargs:
      dataset_dir: /shared/datasets/my-harbor-tasks    # the provider and its options come from the site profile
    harness: platoon-bash
```

Point the image cache at shared storage so jobs do not pull the same image again, and the sandboxes' scratch at the
node's local disk. Pull large images ahead of the run when the nodes have no registry access. Both providers read
these options, and the enroot provider also takes the singularity provider's names, so a config moves between a
Singularity cluster and an enroot cluster with no change at all. `binary` and `extra_start_args` are the exceptions:
they are each runtime's own executable and start flags, so a config that needs them names them, with the provider, in
an entry's `service_kwargs`, never in a block both sites use. See [Sandboxes and Harbor
tasks](sandboxes-and-harbor.md#sandbox-providers).

Harbor tasks whose verifier runs as root ([Verifiers that run as
root](sandboxes-and-harbor.md#verifiers-that-run-as-root), since 0.3.0) need Singularity's `--fakeroot` for
their verifiers, and so do the sandboxes of tasks whose users are honored ([Users in task
sandboxes](sandboxes-and-harbor.md#users-in-task-sandboxes), since 0.5.0), which also need hpc-sandbox 0.5 for an image
that gives files to users other than root, and util-linux's `nsenter` on the node. Both work where the site gives
your user subordinate ids. Check on a compute node that
`getsubids $USER` lists a range, and that `singularity exec --fakeroot IMAGE cat /proc/self/uid_map` shows a range
that reaches uid 65534 (a line `1 <first subordinate id> 65536`, say), or that
`singularity exec --fakeroot IMAGE setpriv --reuid=65534 --regid=65534 --clear-groups id -u` prints 65534. `id -u`
alone prints 0 in a namespace that maps no other user too. enroot cannot run them.

Harbor tasks with sidecar services ([Multi-container
tasks](sandboxes-and-harbor.md#stacks-as-pods-on-singularity-apptainer-and-enroot), since 0.5.0) run as hpc-sandbox
pods, which need, on the compute nodes:

| What | For | Check |
|---|---|---|
| hpc-sandbox 0.5.1, the `hpc` extra's | every stack: its pods' run modes, host aliases and exposed ports (0.5); sidecars that switch users starting where the run's directory or the task's files are on storage that only one of your groups opens, and reaching their binds under a umask such as 027, and a `main` that outlives an agent's `kill -9 -1` (0.5.1) | `python -c 'import hpc_sandbox; print(hpc_sandbox.__version__)'` |
| unprivileged user namespaces that nest: the pod's (with a network namespace), each service's inside it, and the runtime's own | every stack | `unshare --user --map-root-user --net unshare --user --map-root-user unshare --user --map-root-user true` |
| util-linux's `nsenter` (and `prlimit` for services that set `ulimits`) | every stack | `hpc-sandbox doctor` |
| slirp4netns, on `PATH` or fetched once with `hpc-sandbox fetch-tools` (`platoon prepare` fetches it where a stack needs it) | stacks with egress | `hpc-sandbox doctor`'s `pods: egress` line |
| a range of subordinate ids for your user, with shadow-utils' `newuidmap` and `newgidmap` | sidecars that switch users, such as the official database images (Singularity and Apptainer only) | `hpc-sandbox doctor`'s `pods: subordinate ids` line, or `getsubids $USER` |

Without subordinate ids, sidecars run as your own uid, which most database images accept, as they accept `docker run
--user`. enroot runs every sidecar so, since it maps one id into a container. With them, each rollout worker maps them
once, at its first stack, which can take more than a minute where `/etc/subuid` lists many users. A fetch locks stacks
as runnable, and a run refuses them where pods cannot run, when it lists its tasks, naming what is missing: leave them
out (`exclude_tasks`), or fetch them refused with `refuse: {sidecars: true}`. Pods on Apptainer and enroot are
experimental. Where nodes do not nest user namespaces, or hpc-sandbox 0.5 is not installed, `stacks: host_network` runs
the stacks that a dataset's overlays make safe on the node's own network, each service a single sandbox, with any
hpc-sandbox ([Stacks on the host network](sandboxes-and-harbor.md#stacks-on-the-host-network)): their sidecars are then
reachable from the node, so each listens on the loopback and takes its stack's token, through its environment, never its
command line, which every user of the node can read. The message that refuses stacks under `stacks: auto` names the
fallback where it could run them.

Such sidecars may read the task's files as a user of their own (the official postgres image runs its init scripts as
`postgres`). Under a umask that keeps other users out, such as 027, a fetched dataset's files are 0640, so a
sidecar's read-only binds of them bind a copy of the stack's own that every user reads, made at each stack's start.
Fetch and prepare under `umask 022` (`(umask 022 && platoon prepare CONFIG)`) to let those binds take the cache itself,
at the cost of a cache that every user who can reach its directory reads.

Pull the images in the job, before the run, with the run's runtime and image cache, and name the record of what was
pulled in `PLATOON_IMAGE_PULLS`:

```bash
python -m hpc_sandbox pull --runtime singularity --image-cache /shared/container-cache \
  docker://ghcr.io/my-org/task@sha256:<digest> /shared/images/verifier.sif > "pulls-$PBS_JOBID.jsonl" || exit 1
export PLATOON_IMAGE_PULLS=$PWD/pulls-$PBS_JOBID.jsonl
```

`pull` prints one JSON line per image: its `reference`, the `path` of its file, that file's `sha256`, and for a
reference pinned by digest its `manifest_digest`. `lock.json` records them as `image_pulls`
([Configs](configs.md#6-find-what-a-run-used)), and the run refuses to start (exit 78) when the file cannot be read or
holds anything else.

hpc-sandbox keeps a tag's image as it was first pulled, for every job that shares the cache, until
`python -m hpc_sandbox pull --refresh IMAGE` pulls the tag again. A sandbox started after a refresh gets the new image,
in a run already going too, so refresh between runs, and pin training images by digest
(`docker://repo@sha256:<digest>`), which both providers take and a refresh never moves. Old image files are never
deleted for you, and with enroot every refresh writes a new one: remove the `sha256-*` files that no `.json` pointer
names once no job uses them. Jobs on hpc-sandbox releases before 0.2 can share the cache, but pin by digest while
they do: an older release pulls a tag again where only 0.2 or later pulled it, so for a tag that moved the two can run
different images.

### 6. Submit the client

A PBS job script (the resource line follows PBS Pro; adapt it to your site):

```bash
#!/bin/bash
#PBS -N platoon-train
#PBS -l select=1:ncpus=32:mem=128gb
#PBS -l walltime=24:00:00
#PBS -j oe
cd "$PBS_O_WORKDIR"
source .venv/bin/activate
export HF_HOME=/shared/hf-cache HF_HUB_OFFLINE=1
export TINKER_BASE_URL=http://gpu-node-17:8000
export TINKER_API_KEY="$(cat ~/.config/tinker/key)"
exec platoon train configs/train.yaml stats.trial_name=run-1
```

The same as a Slurm job script:

```bash
#!/bin/bash
#SBATCH --job-name=platoon-train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export HF_HOME=/shared/hf-cache HF_HUB_OFFLINE=1
export TINKER_BASE_URL=http://gpu-node-17:8000
export TINKER_API_KEY="$(cat ~/.config/tinker/key)"
exec platoon train configs/train.yaml stats.trial_name=run-1
```

`exec` makes platoon the job's main process, so signals the scheduler sends reach it directly. At the time limit a
scheduler sends SIGTERM and, after a grace period, SIGKILL; on the SIGTERM, `platoon train` stops at the next step
boundary and saves an interrupt checkpoint ([Checkpoints and
resume](train-and-eval.md#checkpoints-and-resume)). That save is a full one, which for a large model trained in
full can outlast a short grace. Have the signal come early enough, for example with Slurm's
`#SBATCH --signal=B:TERM@600` (ten minutes before the limit), or rely on `checkpoint.recovery_every` (step 7).

Check the config on the login node before you queue it: `platoon config validate configs/train.yaml` exits 78 on
a missing file, an unknown key, a badly typed value or a component that does not resolve, which it imports as the
run will. It contacts no service, so also run one task with `platoon rollout` before a long job.

### 7. Continue across walltime limits

A training run resumes by itself when the same command starts again with the same `log_path`,
`stats.experiment_name` and `stats.trial_name`: it restores the last full checkpoint, the optimizer and the W&B run
([Checkpoints and resume](train-and-eval.md#checkpoints-and-resume)). So chain jobs that run the same script:

```bash
# PBS
first=$(qsub train.pbs)
qsub -W depend=afterany:"$first" train.pbs

# Slurm
first=$(sbatch --parsable train.sbatch)
sbatch --dependency=afterany:"$first" train.sbatch
```

`afterany` starts the next job however the previous one ended, including at its walltime. Each job's `platoon train`
reads the site profile again, so change it between runs, not between the jobs of one: another `runs` starts the run
afresh somewhere else, and another `cache` or provider runs the rest of it from other files or on another runtime.
A job that is killed loses the steps since its last full checkpoint, so set `checkpoint.recovery_every` (for
example 5) to bound that loss; recovery checkpoints are disposable and every launch writes one, and
`checkpoint.keep_recovery` (for example 2) deletes all but the newest few, so that a long run does not fill the
training service's disk. A finished run's next job finds nothing to do and exits.

Inside one job, `watchdog.max_restarts` (for example 20) has `platoon train` start training again after a failure,
the watchdog's exit or a training service that stalled included, without waiting for a new allocation; before each
restart it reaps the sandboxes the failed start left ([Long runs](train-and-eval.md#long-runs)).

`platoon eval` resumes too: with `inference.resume: true` (the default) a rerun reuses the rollouts already in
`inference.output_dir` and runs only the missing and errored ones.

## Checklist

- [ ] The client node can reach the training service and every environment server (no firewall between them).
- [ ] `TINKER_API_KEY` and `TINKER_BASE_URL` are set in the client job, not in a config file.
- [ ] The tokenizer is in a shared Hugging Face cache, and the job sets `HF_HUB_OFFLINE=1`.
- [ ] Each remote environment has its `OPENREWARD_SESSION_URLS_<LABEL>` set, or its hosts in the URI.
- [ ] The site profile names the cache and the runs on shared storage and the sandbox runtime, and `platoon train`
  prints the one the job uses.
- [ ] Container images are cached on shared storage; sandbox scratch is on node-local disk.
- [ ] For multi-container tasks, `hpc-sandbox doctor` on a compute node shows what their pods need (step 5).
- [ ] `platoon config validate` passes on the exact config the job runs.
- [ ] `checkpoint.recovery_every` is set, with `checkpoint.keep_recovery` for a long run, and the next job is chained
  with `afterany`.

## Next steps

- [Troubleshooting](troubleshooting.md): connection errors, `env_busy`, harness exit codes and timeouts.
- [RFC 0002](../rfcs/0002-multi-cluster-placement.md) proposes placement across clusters with nodes joining and
  leaving a run; the [RFC index](../rfcs/README.md) lists every proposal.
