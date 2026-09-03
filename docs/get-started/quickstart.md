# Quickstart

Two ways to a first result. Path 1 needs no GPU: it evaluates an agent against any OpenAI-compatible
endpoint. Path 2 starts a training run.

Install first — see [installation](installation.md).

## Path 1: evaluate against an endpoint

All you need is an OpenAI-compatible base URL: a local vLLM or SGLang server, a LiteLLM proxy, or a
hosted API. Nothing on the rollout path imports a training backend.

This uses the `textcraft` plugin, a crafting environment whose tasks ship as JSONL files next to the
code, so there is nothing to download and no service to stand up.

```bash
cd plugins/textcraft
uv sync

uv run python -m platoon.textcraft.inference_scripts.run_inference \
  --config platoon/textcraft/configs/inference/textcraft_inference.yaml \
  --inference.model_name Qwen/Qwen3-4B-Instruct-2507 \
  --inference.model_endpoint http://127.0.0.1:30000/v1 \
  --inference.model_api_key EMPTY \
  --inference.output_dir ./inference_results/smoke \
  --num_tasks 5
```

Three values to set for your setup:

- `--inference.model_endpoint` — your base URL, including `/v1`.
- `--inference.model_api_key` — your key. A local server started without one ignores the value, but
  it has to be present.
- `--inference.model_name` — a model your endpoint serves.

Overrides on this loader are `--dotted.key value`, with the dashes. Everything in the YAML can be
set that way; pass `--config` explicitly.

!!! tip "This config runs a multi-agent rollout"
    `textcraft_inference.yaml` sets `use_recursive_agent: true`, so the agent can delegate subtasks
    to further agents. Five tasks at `max_steps: 50` is a real amount of generation. Pass
    `--use_recursive_agent false` for a single flat agent.

**What success looks like.** The script prints the `summary` block of the final report —
`success_rate`, `success_at_k`, `reward_mean`, and counts of completed and errored rollouts — and
writes everything under `--inference.output_dir`:

```text
inference_results/smoke/
├── rollouts/<task-id>/rollout_0/   trajectory_collection.json, metadata.json, events/
└── reports/                        task_results.jsonl, final_report.json
```

Replay one rollout step by step:

```bash
uv run python -m platoon.visualization.cli replay --dir \
  ./inference_results/smoke/rollouts/<task-id>/rollout_0/events --delay 0.25
```

More in [evaluate a plugin](../guides/evaluate.md) and
[inspect rollouts](../guides/inspect-rollouts.md).

## Path 2: a first training run

The `number-search` plugin is the smallest trainable environment here: a task carries a range and a
hidden target, the only task action is `guess(number)`, and the reward is binary. No external
services, no dataset to build — a clean smoke test for the whole training stack.

Pick a backend. They are compared in [training backends](../architecture/backends.md).

=== "AReaL"

    AReaL trains locally on your own GPUs. `nv_number_search_cispo_areal.yaml` splits one 8-GPU node
    between a 4-GPU SGLang rollout engine and a 4-GPU FSDP actor; the configs in this repository are
    written for 8-GPU nodes.

    ```bash
    cd plugins/number-search
    uv sync --extra areal
    ```

    The shipped config carries cluster-specific paths, so point them at your own on the command
    line. AReaL overrides are bare `key=value` — no dashes.

    ```bash
    uv run python platoon/number_search/train.py \
      --config platoon/number_search/nv_number_search_cispo_areal.yaml \
      cluster.fileroot=/scratch/$USER/areal \
      cluster.name_resolve.nfs_record_root=/scratch/$USER/areal/name_resolve \
      workflow_config.rollout_config.output_dir=/scratch/$USER/areal/rollouts/number-search \
      trial_name=smoke-1 \
      saver.freq_steps=50 \
      evaluator.freq_steps=50 \
      stats_logger.wandb.mode=disabled
    ```

    | Key | Why |
    |---|---|
    | `cluster.fileroot` | A directory writable from every node. Logs, checkpoints and recovery state all hang off it. |
    | `cluster.name_resolve.nfs_record_root` | Shared directory the workers use to find each other. |
    | `workflow_config.rollout_config.output_dir` | Where rollout event logs go. It is a literal path, so set it per run alongside `trial_name`. |
    | `trial_name` | A fresh name per run; logs, checkpoints and the W&B run id are keyed on it. |
    | `saver.freq_steps`, `evaluator.freq_steps` | Write a Hugging Face checkpoint and run validation every 50 steps. Without them the run trains but produces no loadable model. |

    `actor.path` is `Qwen/Qwen3-4B-Instruct-2507`. Keep it if the nodes can reach Hugging Face,
    otherwise point it at a local snapshot; the tokenizer and the SGLang model path follow it.

    `train.py` is run by path, not with `-m`. It is the shortest complete example of the training
    entrypoint contract, and worth reading before you write your own.

=== "Tinker"

    The Tinker path needs no local GPUs. Training runs as a service on a Tinker-compatible backend
    reached over the network, so you need an account with such a service and whatever credentials
    its SDK reads from your environment; follow that backend's own documentation. Set
    `tinker_base_url` if it is not at the SDK's default.

    ```bash
    cd plugins/number-search
    uv sync --extra tinker
    ```

    `number_search_tinker.yaml` uses relative paths, so little needs changing. Overrides here are
    `--dotted.key value`, with dashes.

    ```bash
    uv run python -m platoon.number_search.train_tinker \
      --config platoon/number_search/number_search_tinker.yaml \
      --train.model_name <a model your backend can train> \
      --stats.trial_name smoke-1 \
      --stats.wandb.mode disabled
    ```

    | Key | Why |
    |---|---|
    | `train.model_name` | A model your backend offers. |
    | `train.renderer_name` | The chat rendering for that model family; it must match the model. |
    | `stats.trial_name` | A fresh name per run. `log_path/experiment_name/trial_name` is the run directory, and rollout events land under it. |

    The shipped config checkpoints every 5 epochs and evaluates every 10 steps.

### What success looks like

AReaL brings up its workers first — the rollout engine, the actor and the reference model — which
takes minutes rather than seconds. The Tinker path has no local workers and starts generating almost
immediately.

Then watch `task_reward`: the reward of a single rollout, `0.0` or `1.0` here, whose mean over a
batch is the success rate. `task_reward_at_k_mean` and `_max` aggregate over the rollouts in a group,
and `_max` rising while `_mean` stays flat means the policy can solve tasks but not yet reliably.
`num_steps` should trend down as the agent learns to use the "too low / too high" feedback.

Rollout event logs land under the rollout `output_dir`, split into `train_rollout/` and
`eval_rollout/` and then by engine version, so rollouts from before and after a weight update do not
overwrite each other. Read them with the same `replay` command as Path 1.

If a run produces no `task_reward` at all, the problem is in the rollout rather than the trainer —
read the event logs before touching hyperparameters.

## What just happened

Both paths run the same four steps; training adds a fifth.

1. A task id became a `Task` through the plugin's `get_task`.
2. The plugin's `run_rollout` built an LLM client, constructed the environment and agent, and
   attached a trajectory collection with a JSONL sink.
3. The episode loop alternated agent action and environment step until the agent finished or ran out
   of steps.
4. The environment's `evaluate` scored the trajectory.
5. In training, a group rollout workflow collected several trajectories per task, turned their
   rewards into advantages, and handed token batches to the trainer.

Steps 1-4 are identical either way. That is what lets you validate an environment against any
endpoint before allocating a GPU.

## Next

- [Core concepts](concepts.md) — tasks, agents, environments, trajectories, workflows.
- [Build your first plugin](../guides/first-plugin.md) — package your own task, in this repo or your
  own.
- [Configuration reference](../reference/configuration.md) — the keys behind both configs above.
- [CLI reference](../reference/cli.md) — entrypoints and override syntax.
