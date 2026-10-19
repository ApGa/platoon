# Reward processors and finalizers

**Goal.** Change the reward a trajectory trains on without changing how the environment grades it: score each
trajectory with a **reward processor**, or post-process a whole episode with a **finalizer**. You will write one
of each, wire them into a config, and check them on CPU.

**Prerequisites.**

- Platoon installed with the `tinker` and `ors` extras (`pip install 'platoon[tinker,ors]'`, or
  `uv sync --extra tinker --extra ors` in a checkout).
- [Episode programs](../tutorials/programs.md): finalizers are added by a program.
- [Credit assignment](../tutorials/credit-assignment.md): how a trajectory's reward becomes an advantage.

## Where they run

One rollout's rewards pass through these stages, in order:

1. **Grade.** The environment (or the episode program) returns the episode's `pt.Outcome`. A scored outcome's
   primary reward becomes the root trajectory's `reward` and the `reward/success` of its last step; a subagent
   graded on its own carries its own grade.
2. **Finalizers.** The program's finalizers run on the rollout's `TrajectoryCollection`, in the order the program
   added them, before the rollout returns. They may rewrite any trajectory.
3. **Reward processor.** When a trajectory becomes training data, the environment's `reward_processor` gives the
   reward it trains on and its reward components. Without one, the trajectory's `reward` is used as it is.
4. **Baselines.** The group's completed root rewards set each rollout's baseline, and every token of the rollout
   is centered on it ([Credit assignment](../tutorials/credit-assignment.md#how-a-reward-reaches-a-token)).

Evaluation applies the same reward processor to each rollout's root trajectory for the report's reward statistics,
but `success_rate` counts the reward the environment graded, before any reward processor.

| | Reward processor | Finalizer |
|---|---|---|
| Is | `fn(trajectory: dict) -> (reward, {component: value})` | `fn(collection: pt.TrajectoryCollection) -> None`, sync or async |
| Sees | one trajectory, as a plain dict | every trajectory of the episode, as objects |
| Can | return the training reward and named components, which training reports as metrics | change any trajectory's `reward` and `misc`, call a judge model, look at the whole agent tree |
| Is chosen by | the environment entry: `reward_processor:` | the episode program: `ep.add_finalizer(fn)` |
| Runs in | the process that runs the rollout, when its trajectories become training data (a rollout worker under `rollout_isolation: process`), and in evaluation | the rollout, after its trajectories are built |
| Use it for | per-trajectory scoring rules: partial credit, combining graded components, reporting components | rules that need several trajectories, the run's records or an async call: propagating the root's success, efficiency penalties, judge critiques |

## Write a reward processor

A reward processor is a module-level function. It receives one finished trajectory in the dict form that
`trajectory_collection.json` stores, and returns the reward to train on with a mapping of components:

```python
# my_rewards/partial.py
from typing import Any


def partial_credit(trajectory: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Train on exact answers, with half of the graded `close` reward for a near miss."""

    graded = ((trajectory.get("misc") or {}).get("outcome") or {}).get("rewards") or {}
    exact = float(trajectory.get("reward") or 0.0)
    close = float(graded.get("close", 0.0))
    reward = exact if exact >= 1.0 else 0.5 * close
    return reward, {"reward/exact": exact, "reward/close": close}
```

What a trajectory dict holds:

| Key | Contents |
|---|---|
| `reward` | The trajectory's reward after the finalizers ran |
| `misc["outcome"]` | The trajectory's own grade: `status`, `rewards` (every named reward of the `pt.Outcome`) and `reason` |
| `steps` | One entry per recorded model call; the last one's `misc["reward_misc"]` holds `reward/success` and the components programs and subagents added |
| `misc` | The harness's annotations (`ctx.annotate`), `tool_calls`, `harness` and the subagent verdicts |
| `task` | The agent's task: `id`, `goal` and `misc` (a task table's extra columns, for example) |
| `parent_info` | `None` for the root, else `{"id": <the parent's trajectory id>, "fork_step": ...}` |
| `finish_message`, `error_message` | The agent's final message, and why it failed if it did |

Rules:

- **Return the reward you want trained.** A processor that returns `reward/success` instead of
  `trajectory["reward"]` undoes whatever a finalizer shaped.
- **Name components `reward/<name>`.** Training averages the components whose names start with `reward/` over the
  trajectories of each step and logs them with the step's batch metrics (for example
  `train/batch/reward/close/avg`); the root trajectory's components are also logged as `root_<name>`.
  Evaluation reports them per rollout.
- **Keep it a pure, module-level function.** Under `rollout_isolation: process` it runs in the rollout worker
  processes, which import it by reference: a closure or a lambda cannot cross that boundary.
- **Leave untrainable trajectories alone.** It is called for every trajectory, but timeouts, crashes, invalid
  grades and cancellations train nothing whatever it returns, and they stay out of the group baseline.

Name it on the environment entry. In a mixture, each entry's processor scores that entry's tasks:

```yaml
environments:
  - name: arithmetic
    service: example_extension.tasks:arithmetic
    harness: example-self-check
    program: example-exact-answer
    reward_processor: my_rewards.partial:partial_credit   # module:function
```

The example extension package ships this processor; [Publish an extension package](publish-a-package.md) runs it
end to end. Since the next release, an installed package can also name a processor with a
`platoon.reward_processors` entry point: the example package names it `example-partial-credit`, and its configs
write `reward_processor: example-partial-credit`. Platoon ships one for recursive agents, `platoon-delegation`
(`platoon.rewards:delegation_capped`): the trajectory's success, with the delegation counts reported as
components.

## Write a finalizer

A finalizer receives the rollout's `TrajectoryCollection` once it is built, with the root's trajectory first, and
changes it in place. This one charges every scored trajectory a cost per model call:

```python
# my_rewards/step_cost.py
from dataclasses import dataclass

import platoon as pt
from platoon.programs import single
from platoon.utils.trajectory_status import trajectory_was_interrupted


@dataclass
class StepCost:
    """A finalizer: charge every trainable trajectory `cost` per model call, never below 0."""

    cost: float = 0.01

    def __call__(self, collection: pt.TrajectoryCollection) -> None:
        for trajectory in collection.trajectories.values():
            if trajectory_was_interrupted(trajectory) or trajectory.misc.get("exclude_from_training"):
                continue  # untrainable, or a verifier's: leave it as it is
            charge = self.cost * len(trajectory.steps)
            trajectory.reward = max(0.0, trajectory.reward - charge)
            if trajectory.steps:  # next to reward/success, which evaluation keeps counting as success
                trajectory.steps[-1].misc.setdefault("reward_misc", {})["reward/step_cost"] = charge


@pt.program("my-step-cost")
async def step_cost(ep: pt.EpisodeContext) -> pt.Outcome:
    """Run and grade the episode as the default program does, then charge for its model calls."""

    outcome = await single(ep)
    ep.add_finalizer(StepCost(cost=float(ep.config.extra.get("step_cost", 0.01))))
    return outcome
```

The program composes with the built-in `single`: `single` runs and grades the root and adds the root-success
finalizer when `propagate_root_success` is on, and the step cost is added after it, so it runs after it. Select the
program and set its parameter in the config:

```yaml
environments:
  - service: python://guess:GuessTheNumber      # the quickstart's environment
    harness: platoon-react
    program: my-step-cost
    package: my_rewards.step_cost     # imported first, so @pt.program("my-step-cost") registers
train:
  workflow_config:
    rollout_config:
      extra: {step_cost: 0.02}        # read by the program as ep.config.extra
```

Rules:

- **Order matters.** Finalizers run in the order they were added; adding the same finalizer twice runs it once.
  `propagate_root_success` sets every trajectory's reward to the root's success, so a shaping finalizer added
  before it is undone.
- **Leave untrainable trajectories alone.** `trajectory_was_interrupted(trajectory)` is true for timeouts,
  cancellations, crashes and invalid grades; their reward trains nothing.
- **Change `reward`, keep `reward/success`.** Training reads `reward` (through the reward processor); evaluation
  counts success from `reward/success`, which the trace set from the grade.
- **Shaping the root moves the baseline.** Baselines come from root rewards, so a shaped root reward changes every
  advantage in its group.
- **A finalizer that raises fails the rollout.** The rollout gets an `error` outcome (failure kind
  `finalizer:<Type>`) and its trajectories are rebuilt untrainable.

Built-in finalizers that programs can add:

| Finalizer | Effect |
|---|---|
| `platoon.rewards.zero_subagent_rewards` | Every subagent scores 0 and no launch counts as a success; `single` adds it when `skip_subagent_reward_computation` is on, before root-success propagation |
| `platoon.utils.subagent_rewards.propagate_root_success` | Every trajectory takes the root's success, except explicit forced zeros; `single` adds it when `propagate_root_success` is on |
| `functools.partial(platoon.harness.verify.apply_subagent_parent_usage_gates, judge=...)` | Keeps successful subagents whose parent did not use their work out of policy training ([Judges and verifiers](judges-and-verifiers.md#parent-usage-judges)) |

A finalizer can also call a judge model through `ep.judge_endpoint()`, whose calls are recorded for audit and
never trained: [Episode programs](../tutorials/programs.md#2-write-the-finalizer) shapes number-search rewards
that way.

## Check them on CPU

Both are plain Python, so a unit test needs no model:

```python
# tests/test_my_rewards.py
from my_rewards.partial import partial_credit
from my_rewards.step_cost import StepCost
from platoon.trajectory import Trajectory, TrajectoryCollection, TrajectoryStep


def test_a_near_miss_earns_half_its_closeness():
    near_miss = {"reward": 0.0, "misc": {"outcome": {"status": "scored", "rewards": {"reward": 0.0, "close": 0.8}}}}
    assert partial_credit(near_miss) == (0.4, {"reward/exact": 0.0, "reward/close": 0.8})


def test_the_step_cost_charges_each_model_call():
    collection = TrajectoryCollection(
        trajectories={"root": Trajectory(id="root", reward=1.0, steps=[TrajectoryStep(), TrajectoryStep()])}
    )
    StepCost(cost=0.1)(collection)
    root = collection.trajectories["root"]
    assert root.reward == 0.8 and root.steps[-1].misc["reward_misc"]["reward/step_cost"] == 0.2
```

```bash
PYTHONPATH=. uv run pytest tests/test_my_rewards.py -q
```

```text
..                                                                       [100%]
2 passed in ...s
```

To check a program and its finalizers on a whole episode without a model, run it with `platoon.testing.run_episode`
and a scripted policy ([Test an episode program](../how-to/test-extensions.md#4-test-an-episode-program)), and
call the reward processor on the trajectories it returns, as
[Publish an extension package](publish-a-package.md#step-8-test-on-cpu) does.

## Next steps

- [Judges and verifiers](judges-and-verifiers.md): grade subagents with verifier agents and judges.
- [Episode programs](../tutorials/programs.md): what else a program can do around the root agent.
- [Extending platoon](index.md): every extension point and how platoon finds it.
