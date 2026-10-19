# Credit assignment: from rewards to advantages

**Goal.** Know which reward each trajectory of an agent tree trains on, how a group of rollouts turns those rewards
into advantages, and which settings change that: per-agent outcomes, root-success propagation, baselines, depth
weighting and discounting, subagent sampling, and untrainable outcomes.

**Prerequisites.**

- [Subagents](subagents.md): the example below is its Vault environment and `example-delegator` harness.
- Basic policy-gradient RL: a baseline, an advantage, group-relative (GRPO-style) training.

[`docs/architecture.md`](../architecture.md) has the design behind the trainer and the trace.

Platoon trains every agent of a tree on its own trajectory. The recipes for recursive agent trees described here
were developed for [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639), the paper platoon was
originally designed for. If you use them, please cite it:

```bibtex
@article{gandhi2026rao,
  title   = {Recursive Agent Optimization},
  author  = {Gandhi, Apurva and Chakraborty, Satyaki and Wang, Xiangjun
             and Kumar, Aviral and Neubig, Graham},
  journal = {arXiv preprint arXiv:2605.06639},
  year    = {2026}
}
```

## How a reward reaches a token

One rollout is one task's agent tree. Training runs `group_size` rollouts of each task and, for each:

1. **Outcomes.** The root's trajectory carries the episode's outcome (the environment's grade, or a program's).
   A subagent graded on its own (`agents.child_grading: env` or `verifier`) carries its own grade; an ungraded one
   (`none`) carries the episode's outcome. A scored outcome's reward becomes the trajectory's `reward` and the
   `reward/success` of its last step.
2. **Finalizers.** The episode program's finalizers run on the built trajectories and may rewrite rewards, for
   example `propagate_root_success` or a reward-shaping finalizer ([Episode programs](programs.md)).
3. **Reward processor.** The environment's `reward_processor` (`trajectory dict -> (reward, {component: value})`)
   gives each trajectory's training reward; the default takes the trajectory's `reward` as it is.
4. **Datums.** Each trajectory's model calls merge into training datums (consecutive calls that extend each other
   share one), and every action token starts with its trajectory's reward.
5. **Baselines.** Each rollout gets a baseline from the group's completed **root** rewards, and every action token
   of the rollout, the root's and every subagent's, has that baseline subtracted. A subagent's advantage is therefore
   its own reward minus its rollout's root baseline.
6. **Filters and weights.** Error tokens that would get a positive advantage are suppressed (`filter_errors`), the
   depth transform runs per microbatch (`depth_level_weighting` or `depth_level_discount_gamma`), and datums whose
   advantages are all zero skip the model pass (`filter_zero_advantage_datums`). An error token belongs to a model
   call whose step reports an error: a `platoon-codeact` cell that raised (IPython prints the traceback to its
   `output`), was refused, timed out or wrote to stderr (its `error`); a tool observation with `is_error` (an
   OpenHands harness's `observation_events`); or what a `trajectory.error_completion_ids` hook reports. The detector
   reads these where the harness annotated them (`ctx.annotate_step`, in the step's misc) as well as on the step.
   Harnesses that annotate none of them, such as `platoon-react`, `pt.gym.loop` and TOML specs, get no error
   filtering.

## Steps

### 1. Read per-agent rewards

Run one Vault task (see [Subagents](subagents.md), step 4) and look at each agent's `reward` and `outcome`:

```bash
export PYTHONPATH=$PWD/examples/subagents
uv run platoon rollout examples/subagents/subagents.yaml --task train:0
```

With `child_grading: env`, each subagent's reward is its own answer's grade. When the subagents answer right and the
root then adds their answers up wrong (the policy of `tests/examples/test_credit_assignment_doc.py`), the tree
reads:

```text
rollout train-0-409bf5a41b30  task train:0
agents 3  steps 6  max depth 1  reward 0.0  outcome scored reward=0.0
└─ policy 7bf96f45  steps 2  reward 0.0  outcome scored reward=0.0
   ├─ subagent 3e42d817  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys amber,...'
   └─ subagent 4656d97d  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys fjord,...'
```

The subagents did their part and are rewarded for it; the root is not.

### 2. Propagate the root's success instead

`propagate_root_success` gives every trajectory the root's success, so the whole tree is rewarded for the task's
result only:

```bash
uv run platoon rollout examples/subagents/subagents.yaml --task train:0 \
  train.workflow_config.rollout_config.propagate_root_success=true
```

For the same rollout, both subagents now carry the root's reward:

```text
rollout train-0-8daa8d01b31f  task train:0
agents 3  steps 6  max depth 1  reward 0.0  outcome scored reward=0.0
└─ policy 476aafde  steps 2  reward 0.0  outcome scored reward=0.0
   ├─ subagent b2d664c2  steps 2  reward 0.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys amber,...'
   └─ subagent 8477bf0b  steps 2  reward 0.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys fjord,...'
```

Their own grades stay in each trajectory's outcome (`misc["outcome"]`, printed as `outcome`); only the reward they
train on changes. A trajectory the environment explicitly scored as a
forced zero keeps 0. The built-in `single` program adds this finalizer when the setting is on; a custom program
adds `platoon.utils.subagent_rewards.propagate_root_success` itself, where its other finalizers need it.

### 3. Choose a baseline

Take a group of four rollouts of one task whose roots score `[1, 0, 1, 1]`:

```python
from platoon.train.tinker.workflows.group_rollout_workflow import group_baselines

rewards, completed = [1.0, 0.0, 1.0, 1.0], [True, True, True, True]
print(group_baselines(rewards, completed, leave_one_out=False))
print(group_baselines(rewards, completed, leave_one_out=True))
```

```text
[0.75, 0.75, 0.75, 0.75]
[0.6666666666666666, 1.0, 0.6666666666666666, 0.6666666666666666]
```

- The **group mean** (default) gives the roots advantages `0.25, -0.75, 0.25, 0.25`.
- The **leave-one-out** mean (`leave_one_out_baseline: true`) centers each root on the other three, giving
  `0.33, -1.0, 0.33, 0.33`.

Subagent tokens use their rollout's baseline. A subagent of the second rollout that earned reward 1 while its root
failed gets advantage `1 - 0.75 = 0.25` with the group mean and `1 - 1.0 = 0` with leave-one-out; under
`propagate_root_success` it carries the root's 0 and gets `-0.75` or `-1.0`.

A root that was interrupted (timeout, cancellation, an infrastructure error) never enters a baseline. With the
fourth root interrupted, the baselines are `[0.67, 0.67, 0.67, 0.67]` (mean of the three completed roots) or
`[0.5, 1.0, 0.5, 0.67]` (leave-one-out; the interrupted rollout takes the completed mean). The interrupted root
trains no tokens, while its completed subagents still train against that baseline.

### 4. Weight subagent data

A tree can put many more subagent tokens than root tokens into a batch. Three settings rebalance that:

- `depth_level_weighting: true` scales each datum's advantages by the inverse number of trajectories at its depth
  in the microbatch, renormalized so the microbatch's total action-token weight is unchanged.
- `depth_level_discount_gamma: 0.5` scales advantages by `gamma ** depth` instead (renormalized the same way).
  Set one of the two, not both.
- `subagent_datum_keep_probability: 0.5` keeps every root datum and each subagent datum with that probability,
  drawn deterministically from `subagent_datum_sampling_seed`, the task id and the trajectory id. Rewards and
  statistics are computed before the draw, so they do not depend on it.

### 5. Put it in the config

```yaml
train:
  workflow_config:
    group_size: 8
    leave_one_out_baseline: true
    depth_level_weighting: true
    subagent_datum_keep_probability: 1.0
    rollout_config:
      propagate_root_success: false
```

Then train as usual (`uv run platoon train examples/subagents/subagents.yaml`) and watch the credit metrics:

| Metric | Meaning |
|---|---|
| `credit/baseline/avg` | Mean baseline over the completed roots of a group |
| `credit/baseline/spread` | Largest minus smallest baseline in a group: 0 for the group mean |
| `credit/root_advantage/abs_avg` | Mean absolute advantage of the completed roots |
| `credit/max_depth`, `credit/depth{d}/datums`, `credit/depth{d}/trajectories` | The depth mix of a microbatch |
| `credit/depth{d}/weight` | How much the depth transform scaled depth `d`'s advantages |
| `subagent_sampling/...` | Eligible and retained datums and tokens, per depth, when sampling is on |

### 6. Check the numbers on CPU

The rewards in steps 1 and 2 and the baselines in step 3 are checked by a test:

```bash
uv run pytest tests/examples/test_credit_assignment_doc.py -q
```

```text
...                                                                      [100%]
3 passed in ...s
```

## Untrainable outcomes

An outcome says whether a measurement is valid, not whether the policy succeeded. A failed attempt is `scored`
with a low reward and trains. Anything else trains nothing and never becomes a reward of 0:

| Outcome or marker | Comes from | Effect |
|---|---|---|
| `timeout`, `cancelled` | The agent ran out of time or was cancelled | No tokens train; a root leaves every baseline |
| `error` | A harness crash, a grader or program that raised, a finalizer that raised | Same |
| `invalid` | The environment declared the measurement invalid | Same |
| `exclude_from_policy_training` | A subagent whose verifier verdict is not training-eligible | Its tokens do not train; its reward stays in the statistics |
| `exclude_from_training` | Verifier and judge trajectories | Never trained, not converted at all |
| `exclude_from_training` on a step | `ctx.annotate_step(exclude_from_training=True)` | That call's tokens do not train |

A harness that crashed after the policy filled its context window is a policy failure: it is graded, with reason
`context_overflow`, unless the environment sets `on_context_overflow: exclude`.

A group trains only when at least `min_successful_group_size` of its rollouts are trainable (the default, 1,
needs one): rollouts whose root completed with a `scored` outcome. A rollout whose root ended `invalid`, `error`,
`timeout` or `cancelled` does not count, even when it recorded model calls, and neither does one that returned
nothing: a setup failure, a rollout that made no model call, a straggler cancelled by `straggler_timeout_seconds`,
or a rollout worker that died. With `group_size: 8` and `min_successful_group_size: 4`, for example, a group trains
when at least four of its rollouts are trainable: the others stay out of its baseline and train nothing, and the
group is kept.

## Reference: settings

| Setting | Default | Meaning |
|---|---|---|
| `environments[].agents.child_grading` | `none` | How subagents are graded: `env`, `verifier` or `none` |
| `environments[].program` | `single` | The episode program, whose finalizers can rewrite rewards |
| `environments[].reward_processor` | the trajectory's reward | `module:function` giving each trajectory's training reward and reward components |
| `environments[].on_context_overflow` | `grade` | `grade` or `exclude` a run that overflowed the context window |
| `train.workflow_config.rollout_config.propagate_root_success` | `false` | Every trajectory takes the root's success (with the `single` program) |
| `train.workflow_config.rollout_config.enable_subagent_verifiers` | `true` | Run verifier agents for `child_grading: verifier` |
| `train.workflow_config.group_size` | `8` | Rollouts per task: the group a baseline is computed over |
| `train.workflow_config.leave_one_out_baseline` | `false` | Leave-one-out mean instead of the group mean |
| `train.workflow_config.depth_level_weighting` | `false` | Inverse-frequency weighting by depth |
| `train.workflow_config.depth_level_discount_gamma` | `null` | `gamma ** depth` scaling (instead of depth weighting) |
| `train.workflow_config.subagent_datum_keep_probability` | `1.0` | Probability of keeping each subagent datum |
| `train.workflow_config.subagent_datum_sampling_seed` | `0` | Seed of the subagent datum draw |
| `train.workflow_config.filter_errors` | `true` | Suppress positive-advantage error tokens |
| `train.workflow_config.filter_zero_advantage_datums` | `true` | Skip the model pass for all-zero-advantage datums. A step whose every advantage is zero takes no optimizer update either way, under the `cispo`, `ppo` and `importance_sampling` losses with no `loss_fn_config` keys but their clip thresholds |
| `train.workflow_config.min_successful_group_size` | `1` | Fewest trainable rollouts (completed, `scored` roots) a group needs to train |

To estimate advantages another way, subclass
`platoon.train.tinker.workflows.group_rollout_workflow.GroupRolloutWorkflow`, override `compute_baselines` or
`assign_advantages`, and select the class with `environments[0].workflow: package.module.MyWorkflow`.

## Next steps

- [Episode programs](programs.md): write finalizers that shape rewards.
- [Subagents](subagents.md): grading subagents with environments and verifiers.
- [Reward processors and finalizers](../extending/rewards.md): change the reward each trajectory trains on.
