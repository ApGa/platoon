# Recipes

Where [Customization](../customization/index.md) answers *"how do I plug my own thing in?"*, this
section answers *"which of the options already in the box should I use, and when?"*

Each page surveys one axis of the framework, states the trade-offs, and gives the exact config
block for each choice.

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
### [RL algorithms](algorithms.md)
GRPO/PPO and CISPO, token- versus sequence-level importance sampling, and the advantage-shaping
knobs that matter more than the loss you pick.
</div>

<div class="pl-card" markdown>
### [Recursive agent systems](recursive.md)
Delegation depth, step budgets across a tree, `fork_strategy`, root success propagation, and how
much of a child's trajectory should reach the optimizer.
</div>

<div class="pl-card" markdown>
### [Reward design](rewards.md)
Sparse task success, shaped intermediate rewards, LLM judges, rubric scoring, and penalties that
buy efficiency without collapsing behaviour.
</div>

<div class="pl-card" markdown>
### [Curriculum and task mixtures](curriculum.md)
Weighted mixtures over several task sources, staged introduction of harder environments, and
difficulty filtering.
</div>

<div class="pl-card" markdown>
### [LoRA, FSDP and Megatron](parallelism.md)
Which actor backend to run, how the backend string encodes parallelism, and what each choice costs
you in setup.
</div>

<div class="pl-card" markdown>
### [Long-running and preallocated jobs](scale.md)
Preallocated Slurm allocations, deadlines and stragglers, recovery, and the settings that keep a
multi-day run alive.
</div>

</div>
