# Teams on TextCraft-Synth

These recipes train teams of agents that platoon-harness runs, through the `platoon-harness` harness and program
([Train a team of agents](../../docs/how-to/teams.md)), on the [RAO recipe](../rao/README.md)'s TextCraft-Synth
setting. They carry no published experiment over: they are two shapes of team on a setting whose single and recursive
agents the RAO recipe already runs.

## The arms

Each arm is one config in `textcraft-synth/`, beside the team file it names. Both extend `common.yaml`, which is the
RAO recipe's `common.yaml` with runs and logs of their own: the same model, the 852 medium training tasks, batches of 16
tasks with 8 rollouts each, sampling, loss and optimizer.

| Arm | Config | Team | Credit |
|---|---|---|---|
| tree | `tree.yaml` | `tree.toml`: one OpenHands crafter, which hands parts of its crafting to crafters of its own, each a platoon subagent graded on its own targets | the RAO recipe's recursive arm: 25 steps for every agent, at most 6 levels deep, leave-one-out baselines over the group's roots and depth levels weighted inversely to their trajectory counts |
| mesh | `mesh.yaml` | `mesh.toml`: three OpenHands peers that share one inventory and message each other, under a root that makes no model call | each span of each peer trains on its `propagate` row less its peer's baseline, which `credit: keyed` takes over the group's peers; the peers share 75 steps |

`tree.yaml` is the RAO recipe's `recursive.yaml` but for its harness, program and their options, and its run's names.

## Requirements

- What the RAO recipe requires: platoon with its `tinker` and `ors` extras, the TextCraft extension, a
  Tinker-compatible training service that serves `Qwen/Qwen3-4B-Instruct-2507`, and `WANDB_API_KEY` unless
  `stats.wandb.mode=disabled`.
- The `platoon-harness` extra, and on each host that runs rollouts what
  [Train a team of agents](../../docs/how-to/teams.md#install) lists for OpenHands, with
  [its checks](../../docs/how-to/teams.md#before-you-train) run first.

## Run

With the environment that [Train a team of agents](../../docs/how-to/teams.md#install) builds, which holds
platoon-harness's `openhands` extra and the TextCraft extension:

```bash
for arm in tree mesh; do
  platoon train recipes/teams/textcraft-synth/$arm.yaml log_path=/path/to/runs
done
```

## Check

```bash
uv sync --all-packages --dev --extra tinker --extra ors --extra platoon-harness --extra platoon
uv run --no-sync pytest recipes/teams -v
```

On CPU, without a training service, `test_teams_recipes.py` checks that `textcraft-synth/` holds exactly these arms and
their teams; that each arm loads from any directory, resolves every component it names and validates, the program's
load check included, which reads the team file beside the config; that the tree differs from the RAO recipe's
recursive arm only in its harness, program and names, and its team starts one crafter that spawns its own kind; and
that the mesh trains its three peers on `propagate` rows under `credit: keyed`, with one step budget for the mesh.

## Versions

The configs load and validate on CPU (`test_teams_recipes.py`) with platoon 0.6.0, the TextCraft
extension at the same commit and platoon-harness 0.1.0. Their OpenHands agents run on openhands-sdk 1.59.0, which
platoon-harness's `openhands` extra pins. No training run of them is published.
