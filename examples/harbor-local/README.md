# Harbor tasks in a local sandbox

Evaluate a policy on two Harbor task directories. The
`platoon-bash` harness runs inside each task's sandbox, and each task is graded by its own `tests/test.sh`.
The how-to guide [Sandboxes and Harbor tasks](../../docs/how-to/sandboxes-and-harbor.md) explains the
options used here.

```text
harbor-local/
├── eval.yaml                       # the evaluation config
└── tasks/
    ├── write-greeting/             # a runnable task
    │   ├── task.toml               #   image, workdir, resources, timeouts
    │   ├── instruction.md          #   what the agent is asked to do
    │   ├── tests/test.sh           #   the verifier: writes /logs/verifier/reward.txt
    │   └── solution/solve.sh       #   a reference solution (never shown to the agent)
    └── needs-a-database/           # a task refused here: its compose file adds a sidecar service
        └── environment/docker-compose.yaml
```

## Prerequisites

- platoon installed with the `tinker` extra (`uv sync --extra tinker`).
- A sandbox runtime: Docker (the default without a site profile), or Singularity/Apptainer or enroot through the
  `hpc` extra (`uv sync --extra tinker --extra ors --extra hpc`: `uv sync` uninstalls the extras it is not given).
- A Tinker API key in `TINKER_API_KEY` (and `TINKER_BASE_URL` for a self-hosted Tinker-compatible service, such as
  [miles-tinker](https://github.com/neulab/miles-tinker)).

## Run it

1. From this directory, evaluate the base model:

   ```bash
   cd examples/harbor-local
   uv run platoon eval eval.yaml
   ```

   On a machine without Docker, pick another provider with an override:

   ```bash
   uv run platoon eval eval.yaml environments.0.service_kwargs.provider=singularity
   uv run platoon eval eval.yaml environments.0.service_kwargs.provider=enroot
   ```

2. Read the log. `needs-a-database` is skipped before anything runs, since the provider runs no multi-container
   stacks ([Multi-container tasks](../../docs/how-to/sandboxes-and-harbor.md#multi-container-tasks)):

   ```text
   WARNING | platoon.envs.harbor.dataset | Skipping Harbor task .../tasks/needs-a-database: ...
   environment/docker-compose.yaml makes the environment a multi-container stack (sidecar services), and the
   'docker' sandbox provider runs no multi-container stacks; the task is refused, never started without them
   ```

3. Read the summary that `platoon eval` prints. A policy that solves the task prints:

   ```json
   {
     "total_tasks": 1,
     "valid_rollouts": 1,
     "successful_rollouts": 1,
     "errored_rollouts": 0,
     "outcome_status_counts": {"scored": 1},
     "success_rate": 1.0,
     ...
   }
   ```

   Everything else is under `runs/harbor-local/`: `config.resolved.yaml` and `lock.json`,
   `reports/final_report.json` and `reports/task_results.jsonl`, and one directory per rollout
   (`rollouts/write-greeting/rollout_0/`) with the trajectory, the harness log (`harness/`) and the viewer's
   event file (`events/`).

4. Look at what the agent did:

   ```bash
   uv run platoon viz show-dump runs/harbor-local/rollouts/write-greeting/rollout_0/trajectory_collection.json
   ```

## The `process` provider

`provider: process` runs the "sandbox" as a directory on this host: the agent's shell commands run as your user
with no container and no isolation. It exists for tests and for checking a task on a laptop with a policy you
trust, which is how `tests/examples/test_harbor_local_example.py` runs this example with a scripted policy.
Never point a model you are training at it.

Task scripts that must work under `process` write sandbox paths as `${PLATOON_SANDBOX_ROOT:-}/logs/...`:
the provider translates paths in commands, but not inside files, and real sandboxes leave the variable unset.
