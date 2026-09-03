# Tutorials

Guided, start-to-finish paths. Each one ends with something running on your machine. Work through
them in order the first time — later tutorials assume the vocabulary the earlier ones build.

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 1</span>
### [Train on TextCraft](textcraft.md)
A real task with a real reward signal. Run it on either backend, watch the reward curve move, and
see how a config maps onto the components it selects.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 2</span>
### [Evaluate a model endpoint](inference.md)
No training, no GPU. Point Platoon at any OpenAI-compatible endpoint, run a benchmark, and read
the report it writes.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 3</span>
### [Inspect rollouts in the TUI](visualization.md)
Replay an episode turn by turn, find the step where a reward went wrong, and diff two runs against
each other.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 4</span>
### [Build a task from scratch](build-a-plugin.md)
The long one. A new plugin from an empty directory: tasks, environment, agent, rollout,
registration, config, and a training run.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 5</span>
### [Train a system of agents](recursive-agents.md)
Give your agent a `subagent` action, let it delegate, and train on the whole tree — including how
credit reaches the child.
</div>

<div class="pl-card" markdown>
<span class="pl-card__kicker">Tutorial 6</span>
### [Scale to multiple nodes](multi-node.md)
From one node to many: Slurm submission, preallocated allocations, and the knobs that stop long
runs from dying at hour nine.
</div>

</div>

---

**Looking for something shorter?** [Customization](../customization/index.md) has one focused page
per extension point — a recipe rather than a journey. [Code walkthroughs](../walkthroughs/index.md)
explain the framework's own source instead of asking you to write any.
