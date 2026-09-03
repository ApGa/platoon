# Core concepts

Platoon's vocabulary is small. An **environment** produces observations and consumes actions, an
**agent** turns an observation into an action, and an episode loop alternates the two. A **rollout**
wraps that into one unit of training data. Every other page on this site builds on these words.

## Task and SubTask

A `Task` is the unit of work handed to one episode. `misc` is free-form and carries whatever the
environment needs to construct itself; `max_steps` is the step budget, and `None` means unlimited.

```python title="platoon/envs/base.py"
@dataclass
class Task:
    goal: str | None = None
    id: str | None = None
    max_steps: int | None = None
    misc: dict[str, Any] = field(default_factory=dict)
    fork_strategy: Literal["task", "subtask"] = "subtask"
```

`Task.__str__` is what the model reads, so the goal and the budget go straight into the prompt.

When an agent delegates, `task.fork(goal, max_steps=...)` derives a child task. Under the default
`fork_strategy="subtask"` the child is a `SubTask` that renders its parent chain into its own prompt,
so the delegate knows why it was asked. Under `"task"` it sees only its own goal — the right choice
when the subgoal is self-contained and the ancestry would be noise.

## Env

An environment is a `Protocol`, not a base class. Implement five members and you have one: nothing
to subclass, nothing to register.

```python title="platoon/envs/base.py"
@runtime_checkable
class Env(Protocol):
    async def reset(self) -> Observation: ...
    async def step(self, action: Action) -> Observation: ...
    async def close(self) -> None: ...
    async def observe(self) -> Observation: ...
    @property
    def task(self) -> Task: ...
```

Two responsibilities the loop depends on: `reset()` registers its task on the current trajectory, and
`step()` appends its own `TrajectoryStep`. Step accounting and the budget read those records.
`ForkableEnv` adds `fork(task)`, returning an independently closeable child environment — that is
what makes an environment usable for delegation. An `Observation` carries `task`, `finished`,
`reward` and `misc`, and you subclass it; `Action` is a type alias for `Any`, since an action is
whatever a given agent and environment agree on and Platoon never inspects it.

## Agent

An agent is a protocol too, with three members.

```python title="platoon/agents/base.py"
@runtime_checkable
class Agent(Protocol):
    async def act(self, obs: Observation) -> Action: ...
    async def reset(self) -> None: ...
    async def close(self) -> None: ...
```

The reference implementation is `CodeActAgent`: it asks a model for a thought and a Python cell and
returns a `CodeActAction`, which `CodeActEnv` executes in an embedded IPython shell. `ForkableAgent`
adds `fork(task)` for delegation. A whole agent harness can also arrive as a capability plugin — the
[OpenHands integration](../plugins/openhands.md) ships one.

## The episode loop

`run_episode` is the whole of Platoon's control flow.

```python title="platoon/episode/loop.py"
obs = await env.reset()
while not halt_episode(obs):
    action = await asyncio.wait_for(agent.act(obs), timeout=timeout)
    obs = await asyncio.wait_for(env.step(action), timeout=timeout)
```

An episode halts when the environment sets `obs.finished`, when something called `finish()`, or when
the step budget runs out. `timeout` bounds `agent.act` and `env.step` individually; the deadline for
the whole rollout is separate. On a timeout, a cancellation or any exception the loop still closes
both sides and finalizes the trajectory, so a partial record always reaches your sinks.

Episode state lives in **context variables** rather than arguments, because the code that needs it is
often written by the model at runtime: a Python cell can call `finish("42")` or `launch_subagent(...)`
with nothing plumbed through to it. The rule that follows is to start every episode with
`asyncio.create_task(run_episode(agent, env))`. [Execution](../architecture/execution.md) explains
why.

## Trajectory and the trajectory tree

A `Trajectory` is one episode's record: its task, its list of `TrajectoryStep`s, a reward, and the
finish or error message. `CodeActStep` adds `code`, `thought`, `output` and `error`.

A rollout does not produce one trajectory. It produces a **tree** of them, and that tree is what
makes multi-agent workflows work: whenever an agent delegates, the nested episode becomes a child of
whichever trajectory was current when it started. Recursion — an agent delegating to more instances
of itself — is one shape the tree can take; a supervisor calling three different specialists is
another. No delegation code builds the tree by hand.

All of them live in one flat `TrajectoryCollection`, a `dict[id, Trajectory]`. The tree exists only
as back-pointers: each non-root trajectory's `parent_info` names its parent's id and the parent's
step index at the moment of the fork. `to_dict()` is the single hand-off artifact to reward
processing and to both training backends. [Execution](../architecture/execution.md) follows a tree
from delegation call to optimizer batch.

`TrajectoryCollection` also fires events — created, task set, step added, finished — to every
registered `TrajectoryEventHandler`; the JSONL stream behind
[rollout inspection](../guides/inspect-rollouts.md) is one such sink.

Budget is a pluggable policy installed as a context variable. The default `StepBudgetTracker` gives
a whole delegation subtree one shared budget, so a root with `max_steps=9` caps everything below it
at nine steps. `DepthAwareStepBudgetTracker` instead gives each trajectory its own budget and caps
tree depth: see [multi-agent workflows](../guides/multi-agent.md).

## Rollout function

A rollout is the smallest unit the training loop schedules: one task in, one trajectory tree out.

```python title="platoon/train/components.py"
@runtime_checkable
class RolloutFn(Protocol):
    def __call__(self, task: Task, config: Any) -> Any: ...
```

The second argument is a `RolloutConfig` — model endpoint and key, `max_steps`, `timeout`,
`step_timeout`, `output_dir`, `inference_params`. A rollout builds the agent and the environment,
creates a `TrajectoryCollection`, registers event sinks, runs the episode under the whole-rollout
deadline, and returns the collection. Every task plugin ships one: see
[your first plugin](../guides/first-plugin.md), and
[configuration](../reference/configuration.md) for the config keys.

## Workflow and group

A **workflow** is the backend-side object that turns one dataset row into training data. The default
on both backends is `GroupRolloutWorkflow`.

A **group** is `group_size` independent rollouts of the *same* task, run concurrently. The group
supplies a within-task baseline: instead of learning a value function, the workflow subtracts the
group's mean reward from every datum's reward. That baseline comes from root rewards only but
applies to every datum in the tree, which is how a delegate's tokens inherit credit from the root.

Defaults differ by backend — `group_size` is `1` on the AReaL path and `8` on the Tinker path. A
group of size 1 centers every advantage to zero, so set it above 1 for a real run. See
[backends](../architecture/backends.md).

## The registry

The registry is a process-local name-to-object map. Decorators such as `@register_rollout`,
`@register_task_loader` and `@register_reward_processor` put your components into it, and the
top-level `environments:` list in a training config wires them together:

```yaml
environments:
  - package: platoon.textcraft.registry
    dataset_loader: textcraft/synth
    task_loader: textcraft/synth
    rollout: textcraft/synth/depth_aware
    reward_processor: textcraft/synth/delegation_capped
    workflow: group_rollout
```

`package` is imported purely for its registration side effects, and a name that is not registered is
treated as a dotted import path — so a run needs no registrations at all.
[Components and the registry](../architecture/components.md) has the contracts, the discovery
options, and the OpenReward config key that shares the name `environments:` without being this list.

That is why **a plugin does not have to live in this repository**. A plugin is an ordinary Python
package — either a task plugin (a task or environment plus the rollout that runs it) or a capability
plugin (new framework functionality, such as an agent harness or an environment-server integration)
— installed alongside Platoon and named in a config. Keep yours in your own repo; see
[extending Platoon](../guides/extend.md).

## Glossary

| Term | What it is | Where it lives |
| --- | --- | --- |
| `Task` | Goal, id, `max_steps`, `misc`, `fork_strategy` | <span class="pl-src">platoon/envs/base.py</span> |
| `SubTask` | `Task` plus the parent chain, rendered into the prompt | <span class="pl-src">platoon/envs/base.py</span> |
| `Observation` | `task`, `finished`, `reward`, `misc`; subclassed per environment | <span class="pl-src">platoon/envs/base.py</span> |
| `Action` | Alias for `Any` — whatever the agent and env agree on | <span class="pl-src">platoon/envs/base.py</span> |
| `Env` / `ForkableEnv` | `reset`, `step`, `close`, `observe`, `task`, plus `fork` | <span class="pl-src">platoon/envs/base.py</span> |
| `Agent` / `ForkableAgent` | `act`, `reset`, `close`, plus `fork` | <span class="pl-src">platoon/agents/base.py</span> |
| `run_episode` | The agent/env loop, with a per-step timeout | <span class="pl-src">platoon/episode/loop.py</span> |
| Context variables | Current agent, env, trajectory, budget and messages | <span class="pl-src">platoon/episode/context.py</span> |
| `finish(message)` | Ends the episode from inside a model-written cell | <span class="pl-src">platoon/agents/actions/common.py</span> |
| `launch_subagent` | Forks agent and env, then runs a nested episode | <span class="pl-src">platoon/agents/actions/subagent.py</span> |
| `Trajectory` | One episode's record: task, steps, reward, messages | <span class="pl-src">platoon/episode/trajectory.py</span> |
| `TrajectoryCollection` | Flat `dict[id, Trajectory]` plus event handlers | <span class="pl-src">platoon/episode/trajectory.py</span> |
| `BudgetTracker` | Protocol behind the step-budget policy | <span class="pl-src">platoon/episode/trajectory.py</span> |
| `RolloutFn` | `(task, config)` to a trajectory tree | <span class="pl-src">platoon/train/components.py</span> |
| `EnvironmentConfig` | One entry of the top-level `environments:` list | <span class="pl-src">platoon/train/components.py</span> |
| `GroupRolloutWorkflow` | Runs `group_size` rollouts per task, centers advantages | <span class="pl-src">platoon/train/tinker/workflows/</span> |
| `Registry` | Name-to-object map with a dotted-import fallback | <span class="pl-src">platoon/registry.py</span> |

## Next

- [Components](../architecture/components.md) and [execution](../architecture/execution.md) — the
  protocols in depth, and how a rollout runs end to end.
- [Your first plugin](../guides/first-plugin.md) — put the vocabulary to work.
