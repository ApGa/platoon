# Customization

Platoon is designed to be extended without being forked. Almost every piece you would want to
replace is either a **Protocol** you implement or a **registry entry** you name from a config —
usually both.

This section has one page per extension point. Each page follows the same shape: the contract, a
complete working example, the config wiring, and how to run it.

## The extension map

```mermaid
flowchart TB
  CFG["environments:<br/>config block"]

  subgraph Reg["Registry kinds"]
    DL["dataset_loader"]
    TL["task_loader"]
    RO["rollout"]
    RP["reward_processor"]
    WF["workflow"]
    TC["trainer_config"]
  end

  subgraph Yours["Code you write"]
    D["Dataset loader fn"]
    T["Task loader fn"]
    R["Rollout fn"]
    E["Env / Agent classes"]
    P["Reward processor fn"]
    W["Workflow subclass"]
  end

  CFG --> DL --> D
  CFG --> TL --> T
  CFG --> RO --> R
  CFG --> RP --> P
  CFG --> WF --> W
  CFG --> TC
  R --> E

  LOSS["loss_fn_config"] --> LR["loss registry"] --> L["Loss fn"]
```

## Pages

<div class="pl-cards pl-cards--tight" markdown>

<div class="pl-card" markdown>
### [Custom environment](environment.md)
Implement `Env` — or `ForkableEnv` if children should get their own copy of the world.
</div>

<div class="pl-card" markdown>
### [Custom agent](agent.md)
Implement `Agent`, or subclass `CodeActAgent` and change only the prompt.
</div>

<div class="pl-card" markdown>
### [Custom dataset and tasks](dataset.md)
Turn your data into task ids and `Task` objects the trainer can iterate.
</div>

<div class="pl-card" markdown>
### [Custom rollout](rollout.md)
The function that assembles an agent, an environment and an episode into one trajectory.
</div>

<div class="pl-card" markdown>
### [Custom rewards](rewards.md)
Environment-side `evaluate()`, trajectory-side reward processors, and judge-based rewards.
</div>

<div class="pl-card" markdown>
### [Custom workflow](workflow.md)
Replace the group rollout strategy itself when grouping, filtering or advantages need to differ.
</div>

<div class="pl-card" markdown>
### [Custom loss function](loss.md)
<span class="pl-tag pl-tag--areal">AReaL</span> Register a policy loss by name and select it from
`loss_fn_config`. The Tinker path has no equivalent; the page says what to reach for instead.
</div>

<div class="pl-card" markdown>
### [Custom batch transform](batch-transform.md)
Reshape what reaches the optimizer: masks, weights, filtering, extra tensors.
</div>

<div class="pl-card" markdown>
### [Packaging a plugin](packaging.md)
Namespace layout, `pyproject.toml`, entry points, and how your package gets discovered.
</div>

</div>

!!! tip "You do not have to register anything"

    Every registry field also accepts a dotted import path, so
    `rollout: my_package.rollouts.run_rollout` works without a decorator. Registration exists so
    configs can stay readable and so components can be listed and validated — not as a gate.
