# Train on a Harbor dataset

A project that trains on a [Harbor](https://github.com/harbor-framework/harbor) dataset with configuration alone:
one config, [`train.yaml`](train.yaml), and the dataset it names. Copy this directory, point `dataset` at your own
dataset, and run the same commands.

```text
harbor-dataset/
├── train.yaml                       # the config: the dataset, the harness, the model and the run
└── dataset/                         # a small Harbor dataset: eight tasks about four files, two per file
    ├── towns-total-population/
    │   ├── task.toml                #   the image, time limits and [metadata] topic = "towns"
    │   ├── instruction.md           #   what the agent is asked
    │   ├── environment/towns.csv    #   the file it asks about, copied into the task's working directory
    │   ├── tests/test.sh            #   the verifier: writes /logs/verifier/reward.txt
    │   └── solution/solve.sh        #   a reference solution, never shown to the agent
    ├── towns-largest/               # the other towns task, then the sales, notes and log tasks
    └── ...
```

Each task asks a question about a table, a text or a log, which the agent answers with shell commands in the task's
own `python:3.12-slim` container, and its tests compare the answer it wrote with the expected one. The tasks are small
on purpose: the wiring is what matters, and it is the same for a dataset of thousands of tasks on the Hugging Face Hub.

## Prerequisites

- platoon with the `tinker` extra ([Install](../../docs/getting-started/install.md)): `pip install 'platoon[tinker]'`,
  and `hpc` too for Singularity/Apptainer or enroot, installed as
  [The hpc extra](../../docs/getting-started/install.md#the-hpc-extra) says.
- A container runtime for the task sandboxes: Docker, or on a cluster Singularity/Apptainer or enroot.
- A Tinker-compatible training service: `TINKER_API_KEY`, and `TINKER_BASE_URL` for a self-hosted one such as
  [miles-tinker](https://github.com/neulab/miles-tinker).

## Run it

The commands run from the repository root, but any directory will do: the paths in a config are relative to the
config file.

### 1. Write the site profile, once

The config names no run directory, no cache and no container runtime: those are the site's, in
`~/.config/platoon/site.yaml` (or the file `$PLATOON_SITE` names). On a machine with Docker, one line will do:

```yaml
runs: ~/platoon-runs   # where runs write; the cache defaults to ~/.cache/platoon and the runtime to docker
```

On a cluster, the profile also puts the cache on shared storage and names the runtime
([Write the site's profile](../../docs/how-to/clusters.md#4-write-the-sites-profile-once)):

```yaml
cache: /shared/$USER/platoon-cache
runs: /shared/$USER/platoon-runs
provider: singularity
provider_options: {allow_network_none: true}
```

### 2. Prepare the config

```bash
platoon prepare examples/harbor-dataset/train.yaml
```

Each step prints what it did, and the last line says whether the config is ready; stderr adds, first, the site
profile it read and the fetch's progress:

```text
environments[data-questions] dataset: fetched .../examples/harbor-dataset/dataset into the lock ~/.cache/platoon/harbor/locks/<hash>.lock.json, 8 task(s), 8 runnable
environments[data-questions] hold-out: 25% of the dataset, grouped by topic, joined by shared input files, seed 0
  all tasks: 8 selected, 8 runnable; held out 2: eval 2 in 1 group(s) (largest sales: 2, 100%); train 6
  groups: 4 by topic; no two of them share an input file; largest log: 2 task(s)
  eval 2 task(s), train 6; eval digest 9037690362d8
environments[data-questions] images: none to build: no runnable task builds its image
policy tokenizer: loaded Qwen/Qwen3-4B-Instruct-2507 through the qwen3_instruct renderer, into the Hugging Face cache
ready: 2 made now, 0 already current
```

- **The dataset** is copied into the cache and pinned by a lock, which every rollout reads. A dataset on the Hub
  (`hf://<org>/<name>@<commit>`) is downloaded once; a directory, whose files may change, is copied again by every
  prepare.
- **The hold-out** keeps whole topics out of training: the two sales tasks are the `eval` split, and the other six
  the `train` split. Tasks that share a file always land on the same side, so no held-out task asks about a file that
  training has seen.
- **The images.** These tasks name a prebuilt image (`docker_image`), so nothing is built. Prepare builds the images
  of tasks that build from a Dockerfile, and the toolchain of a harness that runs from one, such as `openhands`.
- **The tokenizer** is loaded once, for compute nodes that cannot reach the Hub.

Run prepare again after any change to the environment entry, and before a run on a cluster, where it runs on a node
with network access, not in the training job.

### 3. Check one task

```bash
platoon rollout examples/harbor-dataset/train.yaml --task towns-largest
```

It runs one task with the base model, from the lock that prepare wrote (a rollout never fetches), and prints the
agent tree. When the model answers with one shell command and then says it is done, that is two model calls, and the
task's tests score the answer 1:

```text
rollout towns-largest-<id>  task towns-largest
agents 1  steps 2  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy <id>  steps 2  reward 1.0  outcome scored reward=1.0
```

A wrong answer scores 0, which is still a measurement that trains. Before training on a dataset of your own, check
its tasks with the two harnesses that make no model call (the command still needs the training service, which it
serves the policy from): a sound task scores 1 with `platoon-oracle`, which runs the task's reference solution, and 0
with `platoon-noop`, which does nothing:

```bash
platoon rollout examples/harbor-dataset/train.yaml --task towns-largest environments.0.harness=platoon-oracle
platoon rollout examples/harbor-dataset/train.yaml --task towns-largest environments.0.harness=platoon-noop
```

### 4. Train

```bash
platoon train examples/harbor-dataset/train.yaml
```

Each of the ten steps rolls out the six training tasks four times, centers each rollout's reward on its group's,
and trains on the tokens the model sampled. The two held-out tasks are evaluated at step 0 and every five steps
(`eval/...` in the metrics). The run writes to `~/platoon-runs/harbor-dataset/qwen3-4b/`: the config it ran
(`config.resolved.yaml`), what it ran on (`lock.json`), its checkpoints (`checkpoints.jsonl`) and every rollout,
which `platoon viz` shows. The same command resumes it from its last checkpoint.

## Make it yours

| To | Change |
|---|---|
| Train on your dataset | `dataset: hf://<org>/<name>@<commit>`, a dataset repository on the Hugging Face Hub pinned to a commit (set `HF_TOKEN` for a private one), a package of the Harbor Hub (`harbor://<org>/<name>@<revision>`), or a directory of your tasks ([more sources](../../docs/how-to/sandboxes-and-harbor.md#fetch-a-dataset-by-name)) |
| Hold out your dataset's groups | `holdout.group_by` (and `index`): the field that says which tasks belong together, from each task's `[metadata]` or the dataset's `task-index.json`; `stratify_by` gives each source its own share |
| Use another agent | `harness: mini-swe-agent@2.4.6`, or `openhands@1.54.0`, whose toolchain `platoon prepare` builds |
| Train another model | `train.model_name`, and the `renderer_name` that matches it ([Models and tool calling](../../docs/how-to/models-and-tool-calling.md)) |
| Train longer or wider | `train.batch_size`, `train.max_training_steps`, `workflow_config.group_size` and the rollout's `max_steps` |
| Train tasks that report their reward under different names | `reward: [reward, correctness]` in `service_kwargs`, beside `dataset`: each task's tests must report exactly one of the names |
| See each source's reward | `train.label_by: <index field>` |

[Sandboxes and Harbor tasks](../../docs/how-to/sandboxes-and-harbor.md) covers each of these, and the README's
[vocabulary table](../../README.md#train-on-a-harbor-dataset) says what Harbor calls each part.

## How this example is tested

`tests/examples/test_harbor_dataset_example.py` runs these steps on CPU: `platoon config validate` and
`platoon prepare` with a site profile in a test directory, then `platoon rollout` and `platoon train` (up to the
training service) with the sandboxes on the `process` provider, host directories without a container, and a scripted
policy that answers with each task's reference solution. It checks the lines quoted here, and that every task's
reference solution scores 1 and doing nothing scores 0.
