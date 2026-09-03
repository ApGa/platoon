# Engineering walkthroughs

Line-by-line traces through Platoon's internals, kept in the repository rather than published on
the documentation site. They are useful when you are modifying the framework itself; they are more
detail than most users need.

| Note | What it traces |
| --- | --- |
| [training-run.md](training-run.md) | A full AReaL training run, from the shell command to a weight update |
| [group-rollout-workflow.md](group-rollout-workflow.md) | How a group of rollouts is launched, rewarded, centered and filtered |
| [trajectory-to-batch.md](trajectory-to-batch.md) | How a trajectory tree becomes the tensors handed to the optimizer |
| [subagent-call.md](subagent-call.md) | What happens between a delegation call and the child's trajectory being attached |
| [plugin-anatomy.md](plugin-anatomy.md) | Every file in a plugin, and what you would change to make it yours |

These describe internals that move faster than the public API. Where one disagrees with the code,
the code is right — and the published docs at <https://apga.github.io/platoon/> are the supported
reference.
