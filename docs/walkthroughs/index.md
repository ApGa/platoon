# Code walkthroughs

These pages read the framework's real source with you, in execution order, naming the file and the
function at each step so you can open them alongside. They exist because the fastest way to trust a
framework — and to know where to cut into it — is to watch one request go all the way through.

Nothing here is pseudocode. Every excerpt is copied from the branch these docs are built from.

<div class="pl-cards" markdown>

<div class="pl-card" markdown>
### [A training run, end to end](training-run.md)
From `python -m platoon.train.areal.train --config ...` to an optimizer step: config parsing,
component resolution, engine placement, the rollout loop, batching, and the loss.
</div>

<div class="pl-card" markdown>
### [The group rollout workflow](group-rollout-workflow.md)
The busiest class in the codebase. How a group of `group_size` rollouts is launched, timed out,
rewarded, centered into advantages, and filtered before it becomes training data.
</div>

<div class="pl-card" markdown>
### [Anatomy of a plugin](plugin-anatomy.md)
Every file in `plugins/number-search`, and what would change if it were yours. Then the same tour
of a much richer plugin.
</div>

<div class="pl-card" markdown>
### [A sub-agent call](subagent-call.md)
What actually happens between "the agent emits a delegation action" and "the child's trajectory is
attached to the parent" — forking, budgets, cleanup, and reward flow.
</div>

<div class="pl-card" markdown>
### [Trajectory to training batch](trajectory-to-batch.md)
The data pipeline: a tree of trajectories becomes tokens, masks, advantages and a padded batch —
and which knobs drop data along the way.
</div>

</div>

!!! info "How to read these"

    Each step is labelled with its source location, like <span class="pl-src">platoon/episode/loop.py</span>.
    Open the file alongside the page; the excerpts are trimmed for readability but never rewritten.
