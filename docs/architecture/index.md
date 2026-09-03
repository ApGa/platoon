# Architecture

Platoon trains agents on tasks you define. The shape of the system follows from one constraint: the
code that describes a task should not know how training works, and the code that computes gradients
should not know what a task is. Everything below is a consequence of keeping those two apart.

```mermaid
flowchart TB
  PL["Your plugin"] --> CO["Platoon core"]
  CO -->|"sampling"| IN["Inference"]
  CO -->|"steps"| ES["Env service"]
  CO -->|"records"| TJ["Trajectory trees"]
  TJ -->|"gradient steps"| BK["Training backend"]
  BK -->|"updated weights"| IN
  TJ --> OB["Visualization"]
```

## The layers

**Your plugin** supplies the task, the environment that scores it, the agent, the rollout program
that ties them together, and the YAML naming all of it — as an ordinary Python package that can live
in your own repository. **The core** runs episodes and holds the registry; it carries no training
dependencies, so an episode against a hosted model never loads a training framework. **A backend**
turns the resulting trajectories into gradient steps. **Observability** sits alongside rather than
downstream: rollout events stream to sinks as episodes run, so the TUIs read the same files a trainer
would.

## Composition by service

The expensive parts run as services addressed over the network, not as libraries linked into one
process, so each scales and is swapped independently.

- **Training engines behind the Tinker API.** That path targets an API, not a vendor, so any
  Tinker-compatible implementation works — a hosted service or your own.
- **Environments behind a server.** The [OpenReward](../plugins/openreward.md) integration talks to a
  server that owns the containers and the task catalog. Rollout workers stay thin.
- **Inference behind an endpoint.** Agents reach models through an OpenAI-compatible endpoint. In
  training it is managed for you; in [evaluation](../guides/evaluate.md) you point at any endpoint,
  which is why evaluation needs no GPU.

So the same plugin runs against a small hosted endpoint while you debug it and against a cluster when
you train, with no code change.

## Where the code lives

| Area | Package |
| --- | --- |
| Environments, tasks, observations | <span class="pl-src">platoon/envs</span> |
| Agents and actions | <span class="pl-src">platoon/agents</span> |
| Episode loop, trajectories, budgets | <span class="pl-src">platoon/episode</span> |
| Registry and `Auto` factories | <span class="pl-src">platoon/registry.py</span> |
| Training backends | <span class="pl-src">platoon/train</span> |
| Evaluation | <span class="pl-src">platoon/inference</span> |
| TUIs, event sinks, analysis | <span class="pl-src">platoon/visualization</span>, <span class="pl-src">platoon/analysis</span> |

## Pages

<div class="pl-cards pl-cards--tight" markdown>

<div class="pl-card" markdown>
### [Components](components.md)
The pieces you assemble — tasks, environments, agents, rollouts — and the registry that lets a
config name them.
</div>

<div class="pl-card" markdown>
### [Execution](execution.md)
What happens during a run: the episode loop, delegation to other agents, and the trajectory tree
that comes out.
</div>

<div class="pl-card" markdown>
### [Backends](backends.md)
AReaL and Tinker side by side — what each one runs, where the compute lives, and how to choose.
</div>

</div>

For the vocabulary these pages assume, read [Concepts](../get-started/concepts.md) first.
