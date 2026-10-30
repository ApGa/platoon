# Write a credit estimator (experimental)

A credit estimator decides how a training group's rewards become the advantages its rows train with. Platoon's own are
`group` (the workflow's own credit: each rollout centered on the group's completed-root baseline), `keyed` (each
agent centered on the group's agents of its key: task, policy, role, depth) and `reverse_kl` (on-policy distillation:
each token's negative reverse KL against a frozen teacher's score of it); this page shows how to write your own,
register it and test it. What choosing one does to a run is in
[Credit each environment's groups](../how-to/train-and-eval.md#credit-each-environments-groups-experimental).

The credit API lives in `platoon.experimental` and is **experimental**: it is documented, and its names are in the
public API's snapshot, but a minor release may change it, and its CHANGELOG then says how. Pin the platoon version you
build against. The [experimental API reference](../reference/experimental.md) lists every name with its signature.

## Where an estimator sits

Each environment entry chooses its own, with its options beside it; an entry without one keeps the workflow's credit:

```yaml
environments:
  - name: solo
    package: acme_credit.running         # imported first, so its @credit_estimator names resolve
    credit: acme-running                 # this entry's estimator, with its own options
    credit_options: {rate: 0.05}
  - name: other                          # the workflow's own credit
```

## The estimator

An estimator is a class with `needs_agents`, a bool, and `async assign(group)`, awaited once per training group in the
trainer process, after every member has settled. `group` is a `CreditGroup`:

- `members`: the members that returned a result, in the group's order, each a `MemberRollout`: its `samples` (every
  row of its rollout), its root's `task_reward` and whether that is a completed root's scored outcome
  (`task_reward_valid`), and, when the estimator reads agents, its `agents`;
- `agents()` and `samples()`: every agent and every row of the group, member by member;
- `task_id`, `label`, `entry`, the entry's training `settings`, the run's `policies` (`PolicyView`: id, renderer),
  each policy's version when `assign` began (`versions`), and `stats`, where the group's own metrics go (put yours
  under `credit/`; the step that trains the group records them).

A row is a `TrainingSample`: its `datum`, the `policy` that sampled it, its `member` and, when the estimator reads
agents, its `agent_id`, `role`, `trajectory_id` and `depth`. `sample.advantages` and `sample.mask` read its
advantages and which of its target tokens are actions; `sample.set_advantage(value)` sets one value on its action
tokens, and `sample.set_advantages(values)` one value per target token, set on the action tokens only. A row the
estimator leaves alone keeps what the converter loaded: its trajectory's processed reward on every action token.

An agent is an `AgentRun`: its `agent_id`, `trajectory_id`, its parent's trajectory (`parent_id`, none for the root),
its `role` (the root's is `policy`; a subagent's is the role its harness minted it with, `subagent` for
`ctx.agents.spawn`), its `policy`, its `depth`, its `reward` and its rows (`samples`). Its reward is its trajectory's
processed reward, or none when it was not scored: the agent was interrupted, or it is a child whose verifier gave no
verdict (a judge error, an unparseable or a pending verdict), whose 0 is no score. An estimator never trains an
unscored agent's reward as 0, nor counts it in a baseline. An agent whose rows the converter left out has no
`samples`: an unscored one, and a scored child that was sampled out or that its parent did not use, whose reward still
counts.

It is built once per start of the process and entry, `Class(setup, options)`: `setup` is a `CreditSetup` (its name as
the config gives it, its entry, the entry's training settings, the run's policies by id, and `policy_views`, each
policy as the estimator may read it), `options` its options parsed. Subclass `CreditEstimatorBase` for the defaults.

An estimator that scores rows with another policy, a frozen teacher or judge the config declares under `policies:`,
reads it from `setup.policy_views[policy_id]`, a `PolicyView`: `await view.score(datums)` gives that policy's logprob
of each datum's target tokens, one per target token, aligned as the datum's own sampled logprobs
(`datum.loss_fn_inputs["logprobs"]`); its `tokenizer` and `renderer` say how it reads them. `reverse_kl` is such an
estimator: [`platoon/train/credit/reverse_kl.py`](../../platoon/train/credit/reverse_kl.py) is about 200 lines. The
scoring calls run in the trainer while the group's `assign` awaits, each bounded by that policy's
`sample_timeout_seconds`. A call that fails, runs out of time or gives a response of the wrong length raises
`ScoringFailed`: let it propagate, and the group fails alone (verdict `failed`, counted under `credit/failures`),
training nothing, while the run goes on. The group's own `policies` are the same views, and its `versions` each one's
version when `assign` began.

## The rules an estimator keeps

1. **Agents only when it reads them.** `needs_agents = True` has the rollout workers tag every row with the agent
   that produced it (worker protocol 3) and fills each member's `agents`; leave it False otherwise, and the rows and
   requests stay as they are without an estimator.
2. **Final advantages.** A group is credited once. A checkpoint that keeps a settled group keeps its advantages, and
   the resume trains it with them; no estimator sees it again.
3. **State only in `settle`.** An estimator whose credit depends on the groups before (a running baseline) changes
   its state in `settle(group)`, a synchronous method the run calls as it files the group with those a full
   checkpoint keeps, never in `assign`; it gives the state as JSON data with `state()` and takes it back with
   `load_state(state)`. Each full checkpoint takes the state where it takes the groups it keeps, so a checkpoint never
   holds an estimator's state for a group it does not hold, nor the reverse, and a resume credits the groups that
   settle after it as the run would have.
4. **No await-sensitive assumptions.** `assign` may await (a service's call, say), and the calls of different groups
   run concurrently, each switching to the others at its awaits: keep what you compute for one group in the call's
   locals or in the group, never in the estimator. It may not assume that the rows of different policies train in the
   same step, or at all: each policy's stale filter judges its own rows.
5. **Deterministic.** Never iterate a `set` of strings: its order changes from process to process.
6. **In time.** `assign` returns within `assign_timeout_seconds` (600 by default; `None`, no limit), an attribute an
   estimator that calls a slow service may set, from its options say, in `__init__`.

An estimator that raises, in `assign`, `settle`, `state` or `load_state`, or whose `assign` runs out of time (the call
is then cancelled), stops the run with a `CreditEstimatorError` that names it, its entry and the group's task: a
group it could not credit never trains otherwise, nor counts as a failed rollout. A resume from the last checkpoint
goes on once it is fixed. The one exception is a policy's failed scoring (`ScoringFailed`, above), which fails only
its group.

## Options and checks

An estimator's options are `credit_options`. Its class names their schema in `options_schema`, or the decorator's
`options=`: a dataclass (each unknown key refused with a did-you-mean, each wrong type with its path, a `ValueError`
its `__post_init__` raises reported under the field it names), a pydantic model, or a function of the options' mapping
that returns them parsed and raises `ValueError`; without one, any option is refused. `lock.json` records each
estimator with its entries and their options, defaults included.

A static `check(options, ctx)` (or the decorator's `check=`) refuses a config the estimator cannot serve: raise
`ValueError`, and `platoon config validate` and `platoon train` report it, with exit 78, before anything runs. Its
`ctx` (`CreditCheckContext`) holds the entry, the run's policies and the whole loaded config. A resume whose entry
names another estimator than the one that saved its state starts the new one afresh, with a warning.

## Register it, or name it

Register an estimator under a short name with `@credit_estimator(name, options=..., check=...)` in a module an entry's
`package` names, so that it is imported before the estimators resolve; or name it in the config as `module:Class`,
with no registration at all. Core's names (`group`, `keyed`, `reverse_kl`, and `gae`, which platoon adds later) are
reserved, whatever their case: registering one is an error, whatever was imported first, and so is registering a
name twice. An unknown name is refused before the run starts, with the nearest name.

## Test it

`platoon.testing.check_credit_estimator` checks an estimator against these rules, on fixture groups of four members,
each a root that delegated to one subagent (two roles), with an interrupted root and a subagent whose rows were left
out, and raises `AssertionError` naming each check it fails:

```python
from platoon.testing import check_credit_estimator


def test_the_running_baseline_keeps_the_contract():
    check_credit_estimator("acme_credit.running:RunningBaseline", options={"rate": 0.05})
```

Its checks are `protocol` (it is an estimator class, it builds, and its `check`, given a loaded config of one entry,
returns or raises `ValueError`), `options` (and a typed schema refuses an unknown key), `action_tokens` (advantages set
on action tokens only: every row keeps its length, its type and its values where its mask is 0, and every advantage is
finite) and `resumes` (an estimator that credited and settled three groups, and a new one that took up its state,
credit the next group alike; and an estimator with `state()` keeps it as it was through `assign` alone). Passing the
check is no proof: state kept outside `state()`, or that shows only after more groups or on other inputs, passes it.
See
[Test an environment, a harness or a program on CPU](../how-to/test-extensions.md#check-a-credit-estimator).

## A worked example

With one rollout per task (`group_size: 1`), a group has no peers to center on: the group mean is the rollout's own
reward, and every advantage is 0. This estimator centers each member on a running mean of the entry's completed root
rewards instead, which each settled group moves by `rate`; its state is that mean.

```python
# acme_credit/running.py
from dataclasses import dataclass

import platoon.experimental as px


@dataclass(frozen=True)
class RunningOptions:
    # How far each group's mean completed-root reward moves the running mean.
    rate: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValueError(f"rate: must be above 0 and at most 1, got {self.rate}")


@px.credit_estimator("acme-running", options=RunningOptions)
class RunningBaseline(px.CreditEstimatorBase):
    """Each member's rows centered on a running mean of the entry's completed root rewards."""

    def __init__(self, setup: px.CreditSetup, options: RunningOptions) -> None:
        super().__init__(setup, options)
        self.mean = 0.0

    async def assign(self, group: px.CreditGroup) -> None:
        for sample in group.samples():
            sample.set_advantages(sample.advantages - self.mean)
        group.stats.scalar(**{"credit/running/mean": self.mean})

    def settle(self, group: px.CreditGroup) -> None:
        rewards = [member.task_reward for member in group.members if member.task_reward_valid]
        if rewards:
            self.mean += self.options.rate * (sum(rewards) / len(rewards) - self.mean)

    def state(self) -> dict:
        return {"mean": self.mean}

    def load_state(self, state: dict) -> None:
        self.mean = float(state["mean"])
```

An entry selects it with `package: acme_credit.running` and `credit: acme-running`, or with
`credit: acme_credit.running:RunningBaseline` and no package. An estimator that reads agents, such as one that gives
each role a baseline of its own, sets `needs_agents = True` and walks `group.agents()`:
[`examples/credit-by-role`](../../examples/credit-by-role/README.md) is one, a tour of the API in about 40 lines that
credits as `keyed` does with its default key, with a CPU test that trains with it.
