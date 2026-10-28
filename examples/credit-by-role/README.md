# Credit each role of a multi-agent rollout on its own

**Goal:** change how a group's rewards become advantages without subclassing the workflow: a credit estimator of about
40 lines that gives the root agents and the subagents of a delegating rollout baselines of their own, selected from
the config.

`ByRole` credits as platoon's own `credit: keyed` does with its default key (`[task, policy, role]`: a group holds one
task, and this run trains one policy), so to train this way, set `credit: keyed`. The example is a tour of the
credit API: what an estimator sees of a group, how it sets advantages and how it is selected, configured and tested,
in a class short enough to read at once, and a start for an estimator of your own.

**Prerequisites:** the [subagents example](../subagents/README.md), whose environment and harness this one trains;
platoon installed with the `tinker` and `ors` extras; a Tinker API key in `TINKER_API_KEY` to train.

| File | What it is |
|---|---|
| [`by_role.py`](by_role.py) | the estimator, `ByRole`, and its options |
| [`train.yaml`](train.yaml) | the subagents example's training config, with the estimator selected for its environment |

## How the estimator works

A group of the Vault environment is `group_size` rollouts of one task; in each, the root agent (role `policy`)
delegates halves of its keys to two subagents (role `subagent`), each graded by its own answer (`child_grading:
env`). Without an estimator, every agent of a rollout is centered on its rollout's root baseline, the group's mean
root reward, so in a group whose roots all failed, a subagent that answered right trains as if it beat the group.

`ByRole` is a `platoon.experimental.CreditEstimatorBase`. Its `needs_agents = True` has the rollout workers tag every
row with the agent that produced it, so that its `assign(group)` sees each member's agents (`AgentRun`: role, policy,
depth, parent, its own reward and its rows). It collects the group's scored agents by role and sets each row's
advantage, on its action tokens, to its agent's reward minus its role's baseline: the role's mean, or with
`leave_one_out` the mean of the other agents of its role. It records each role's mean reward under
`credit/by_role/<role>/reward`.

Platoon awaits `assign` once per training group, in the trainer, after every member has settled; the error-token
filter and the group's verdict run after it, as without an estimator. An evaluation never runs it. The `assign` calls
of different groups run concurrently, each switching to the others at its awaits, so `ByRole` keeps what it computes
for a group in the call's locals. An estimator that raises stops the run with an error that names it and its entry.

## Select it

```yaml
environments:
  - name: vault
    # ...
    credit: by_role:ByRole           # module:Class, or a name a package registers with @credit_estimator
    credit_options:
      leave_one_out: true            # checked against ByRoleOptions when the config loads
```

An estimator that does not import, or refuses its options, stops `platoon train` with exit code 78, and `lock.json`
records the one each entry runs with its options (`kind: credit`).

## Run it

```bash
export PYTHONPATH=$PWD/examples/subagents:$PWD/examples/credit-by-role
uv run platoon config validate examples/credit-by-role/train.yaml
uv run platoon train examples/credit-by-role/train.yaml
```

The run writes to `runs/subagents/delegator-by-role/`.

## How this example is tested

[`tests/examples/test_credit_by_role_example.py`](../../tests/examples/test_credit_by_role_example.py) validates the
config, checks the estimator with `platoon.testing.check_credit_estimator`, and runs `platoon train` with it on CPU:
on `platoon.testing`'s fake Tinker service, with a scripted policy whose subagents always answer right and whose
roots answer right in one rollout of each group. Every subagent's rows then get an advantage of 0, so the step sends
only the roots' rows, one right and one wrong root per group.

The guide to writing an estimator is [Write a credit estimator](../../docs/extending/credit-estimators.md).
