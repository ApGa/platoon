# Architecture

This section explains *why* Platoon is shaped the way it is. It is the counterpart to the [code
walkthroughs](../walkthroughs/index.md): those follow control flow, these explain design.

## Layers

```mermaid
flowchart TB
  subgraph P["Plugins — one uv project per task suite"]
    PL["tasks · env · agent · rollout · configs · registry module"]
  end

  subgraph C["Core — no training dependencies"]
    EN["envs: Env, ForkableEnv, Task, SubTask, Observation"]
    AG["agents: Agent, CodeAct, actions incl. subagent"]
    EP["episode: run_episode, Trajectory, TrajectoryCollection, budgets"]
    RG["registry: named components + Auto factories"]
    UT["utils: config, llm_client, stats, data processing"]
  end

  subgraph T["Training backends"]
    AR["AReaL: trainer, actor, losses, workflows, batch transforms, patches"]
    TK["Tinker: trainer, proxy, workflows, batch transforms"]
  end

  subgraph O["Observability"]
    VZ["visualization: event sinks, TUIs, CLI"]
    AN["analysis: compare, error analysis, checkpoint acceptance"]
  end

  P --> C
  C --> T
  C --> O
  T --> O
```

## Pages

<div class="pl-cards pl-cards--tight" markdown>

<div class="pl-card" markdown>
### [Registry and Auto factories](registry.md)
How a string in a YAML file becomes a Python callable, and why the indirection is worth it.
</div>

<div class="pl-card" markdown>
### [Agents, environments, episodes](agents-envs.md)
The Protocol-based core, the context-variable design, and the episode loop's termination rules.
</div>

<div class="pl-card" markdown>
### [The fork and sub-agent model](subagents.md)
Trees of trajectories: forking, parent links, budget accounting, and reward propagation.
</div>

<div class="pl-card" markdown>
### [AReaL backend internals](areal.md)
Single-controller training, SGLang rollouts, the proxy, and what Platoon patches upstream.
</div>

<div class="pl-card" markdown>
### [Tinker backend internals](tinker.md)
The managed-service path, the sampling proxy, and where it diverges from AReaL.
</div>

<div class="pl-card" markdown>
### [Data pipeline](data-pipeline.md)
Trajectory tree to token tensors: grouping, advantages, masks, filtering, sampling.
</div>

<div class="pl-card" markdown>
### [Configuration system](config.md)
Two config loaders, two override syntaxes, and how the typed dataclasses are assembled.
</div>

</div>
