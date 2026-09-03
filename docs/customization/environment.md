# Custom environment

An environment is the half of a Platoon rollout that owns the world: it decides what tools the
agent can call, what the agent sees after calling them, when the episode is over, and what reward
the trajectory earns. This page is the how-to — the contract, a complete worked example, the
wiring, and how to test it. For the design reasoning behind the protocols, read
[agents, environments, episodes](../architecture/agents-envs.md).

## The contract

`Env` is a `typing.Protocol`. There is no base class you must inherit and no decorator you must
apply — an object that has these five members is an environment.

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

The whole episode loop is five lines, and it tells you exactly which of those members it actually
uses:

```python title="platoon/episode/loop.py"
obs = await env.reset()
while not halt_episode(obs):
    action = await asyncio.wait_for(agent.act(obs), timeout=timeout)
    obs = await asyncio.wait_for(env.step(action), timeout=timeout)
    step_count += 1
```

| Member | Called by `run_episode`? | What it is for |
|---|---|---|
| `reset()` | Yes, exactly once | Build the initial observation and register the task |
| `step(action)` | Yes, once per turn | Apply the action, record a step, return the next observation |
| `close()` | Yes, in `finally` | Release resources; wrapped in a 10 s timeout |
| `observe()` | **No** | Snapshot state for forks and for external tooling |
| `task` | Indirectly | `launch_subagent` reads `current_env.get().task` to derive the child task |

Two asymmetries surprise people. `Agent.reset()` is never called by the loop, only `Env.reset()`
is. And `observe()` is never called by the loop either — `CodeActEnv.fork` uses it to snapshot
`parent_state` for a child, and that is its only production caller.

### Two invariants you must uphold

The loop records nothing on your behalf. Both of these are hard requirements, not conventions.

!!! warning "`reset()` must register the task, `step()` must record the step"
    **`reset()` must call `set_trajectory_task`.** `StepBudgetTracker._allocated_budget` reads
    `traj.task.max_steps or float("inf")`
    (<span class="pl-src">platoon/episode/trajectory.py</span>). If the trajectory has no
    task, the first call to `halt_episode` raises `AttributeError` on `None.max_steps`.

    **`step()` must call `add_trajectory_step`.** Budget accounting is `len(traj.steps)`. An
    environment that forgets never consumes budget, so the episode never halts on exhaustion and
    runs until the agent calls `finish()` or the whole-rollout deadline kills it.

From scratch, that is the entire skeleton:

```python
from platoon.envs.base import Observation, Task
from platoon.episode.context import current_trajectory, current_trajectory_collection
from platoon.episode.trajectory import TrajectoryStep


class MinimalEnv:
    def __init__(self, task: Task):
        self._task = task

    @property
    def task(self) -> Task:
        return self._task

    async def reset(self) -> Observation:
        collection = current_trajectory_collection.get()
        collection.set_trajectory_task(current_trajectory.get().id, self._task)
        return Observation(task=self._task, finished=False)

    async def step(self, action) -> Observation:
        step = TrajectoryStep(misc={"reward_misc": {"reward/success": 0.0}})
        current_trajectory_collection.get().add_trajectory_step(current_trajectory.get().id, step)
        return Observation(task=self._task, finished=False)

    async def observe(self) -> Observation:
        return Observation(task=self._task)

    async def close(self) -> None:
        return None
```

Write an environment this way when your world is not driven by model-authored Python — an SDK
session, a game server, a browser. The openhands plugin's environment is exactly this shape, with
its own `Observation` and `TrajectoryStep` subclasses.

For anything that looks like tool calling, do not write this by hand. Subclass `CodeActEnv`.

## Subclassing `CodeActEnv`

`CodeActEnv` already implements `reset`, `step`, `close`, `observe`, `fork` and `task`, upholds
both invariants, and delegates the two things only you know: **the action space** (a
`CodeExecutor`) and **the reward** (`evaluate`).

```python title="platoon/envs/codeact/env.py"
step = await self._code_executor.run(action.parsed_code)

if finish_message.get(None) is not None or error_message.get(None) is not None:
    self._state.finished = True
    self._state.misc["finish_message"] = finish_message.get()

step.thought = action.parsed_thought
step.reward, reward_info = await self.evaluate()
step.misc["action_misc"] = action.misc
step.misc["reward_misc"] = reward_info
```

That is `CodeActEnv.step` minus its profiling span and its trajectory bookkeeping. Read it as four
facts:

1. The code the agent wrote runs in a `CodeExecutor` and comes back as a `CodeActStep` carrying
   `code`, `output` and `error`.
2. The episode ends when the `finish_message` **or** the `error_message` contextvar is set. Any
   Python function you inject can end the episode by setting `finish_message` — that is all
   `finish()` does. Do not set `error_message` for warnings; it terminates the episode too.
3. `evaluate()` runs on **every** step, not just the last one.
4. The second element of `evaluate()`'s return lands in `step.misc["reward_misc"]`, which is what
   every downstream reward processor reads.

### Actions are plain Python callables

`IPythonCodeExecutor` builds an embedded IPython shell and injects your actions into its namespace
by name:

```python title="platoon/envs/codeact/env.py"
for action in self.actions:
    shell.user_ns[action.__name__] = action
```

The constructor is:

```python title="platoon/envs/codeact/env.py"
def __init__(
    self,
    task: Task,
    actions: tuple[Callable[..., object], ...] | Sequence[Callable[..., object]] = (finish, safe_asyncio),
    detect_unawaited_async_calls: bool = True,
    detect_while_loops: bool = False,
    detect_interactive_input: bool = False,
):
```

Because injection keys off `__name__`, anything with a `__name__` works: module-level functions,
closures returned by a factory, and **bound methods** — which is how an executor exposes tools that
read and write its own state. Async callables work too; the agent has to `await` them.

Three details that will cost you time if you learn them the hard way:

- `describe_action_space()` returns `""` on the base `IPythonCodeExecutor`. Override it. Its
  output is interpolated into the agent's first user turn under an `# Action Space` heading
  (<span class="pl-src">platoon/agents/codeact/prompts/user-initial.jinja</span>), so it is the
  only documentation the model gets for your tools.
- `IPythonCodeExecutor.reset()` rebuilds the shell from scratch and re-injects only
  `self.actions`. Anything you poked into `shell.user_ns` yourself is gone. The oolong plugin's
  executor overrides `reset` to re-inject its `context` binding for exactly this reason.
- `reset()` must **return an executor**. `CodeActEnv.reset` does
  `self._code_executor = await self._code_executor.reset()`; returning `None` silently replaces
  your executor with nothing.

!!! warning "`import asyncio` does not give you `asyncio`"
    The shell's `__import__` is replaced with a sandboxed version that returns `SafeAsyncio` for
    `asyncio` and its submodules. `gather`, `create_task`, `sleep`, `wait`, `wait_for`,
    `as_completed`, `shield` and the synchronization primitives are allowed; everything else —
    `run`, `get_event_loop`, `new_event_loop`, `set_event_loop_policy` — raises `RuntimeError`.
    Pass `safe_asyncio` in your `actions` tuple if you want the bare name `asyncio` available
    without an import; it carries `__name__ = "asyncio"` so that the injection loop above binds it
    correctly.

    The related AST guard, `UnawaitedAsyncCallDetector`, is on by default but its function-name
    set is hard-coded (`launch_subagent`, `search_web`, `view_webpage_content`, `search_emails`,
    `read_email`). Your own async tool is not covered unless you subclass the detector.

### `evaluate()` is the reward hook

```python title="platoon/envs/codeact/env.py"
async def evaluate(self) -> tuple[float, dict]:
    return 0.0, {}
```

Return `(score, reward_misc)`. The score is accumulated into the observation and, on the finishing
step, written to `Trajectory.reward`. The dict is stored verbatim under
`step.misc["reward_misc"]`.

What makes your reward legible to the rest of Platoon is the **`reward/` prefix**. The reference
reward processor sums every key starting with `reward/` across all steps of a trajectory, and
`reward/success` is the canonical scalar — `_get_base_success` in
<span class="pl-src">platoon/utils/subagent_rewards.py</span> reads
`steps[-1].misc["reward_misc"]["reward/success"]` and falls back to `Trajectory.reward`.

!!! warning "Reward keys are summed across steps"
    Because the reward processor accumulates `reward/*` over every step, emitting a non-zero
    `reward/success` on more than one step multiplies your reward. Emit `0.0` on intermediate
    steps and the real value only once the episode is finished — which is what every plugin
    environment does.

## A worked example

A small environment: the agent gets a bundle of configuration files, exactly one of which sets
`max_retries` above a threshold, and must report which file. It is tool-calling, it has a crisp
binary reward, and the world is read-only, which makes forking cheap.

Everything the environment needs travels in `Task.misc`:

```python title="my_plugin/tasks.py"
from platoon.envs.base import Task

Task(
    goal="Exactly one config file in this bundle sets max_retries above 5. Report its path.",
    id="config_audit.train.0",
    max_steps=8,
    misc={
        "files": {
            "services/api.yaml": "timeout: 30\nmax_retries: 9\n",
            "services/worker.yaml": "timeout: 60\nmax_retries: 3\n",
            "services/cron.yaml": "max_retries: 2\nschedule: '*/5 * * * *'\n",
        },
        "offending_file": "services/api.yaml",
    },
)
```

The executor owns the action space and the per-episode answer slot:

```python title="my_plugin/env.py"
from platoon.agents.actions.common import finish
from platoon.agents.actions.subagent import launch_subagent as _launch_subagent
from platoon.envs.base import Task
from platoon.envs.codeact import CodeActEnv, CodeExecutor, IPythonCodeExecutor, safe_asyncio
from platoon.episode.context import finish_message


class ConfigAuditExecutor(IPythonCodeExecutor):
    """Read-only file tools plus delegation, injected into the agent's IPython shell."""

    def __init__(self, task: Task, subagent_max_steps: int = 6):
        self.files: dict[str, str] = task.misc["files"]
        self.reported: str | None = None
        self.subagent_max_steps = subagent_max_steps
        super().__init__(
            task,
            actions=(finish, self.list_files, self.read_file, self.report, self.launch_subagent, safe_asyncio),
            detect_while_loops=True,
        )

    def list_files(self) -> list[str]:
        """Return every config file path in the bundle."""
        return sorted(self.files)

    def read_file(self, path: str) -> str:
        """Return the contents of one config file."""
        if path not in self.files:
            return f"No such file: {path}. Call list_files() to see what is available."
        return self.files[path]

    def report(self, path: str) -> str:
        """Record the offending file and end the episode."""
        self.reported = path
        finish_message.set(f"Reported {path}.")
        return f"Recorded {path} as the answer."

    async def launch_subagent(self, goal: str) -> str:
        """Delegate a slice of the bundle to a child agent."""
        return await _launch_subagent(goal=goal, max_steps=self.subagent_max_steps)

    async def describe_action_space(self) -> str:
        return (
            "1. def list_files() -> list[str]\n"
            "    List every config file path in the bundle.\n"
            "2. def read_file(path: str) -> str\n"
            "    Return the contents of one config file.\n"
            "3. def report(path: str) -> str\n"
            "    Record your answer. This ends the episode, so call it once and last.\n"
            "4. async def launch_subagent(goal: str) -> str\n"
            "    Delegate to a child agent and return its final message. Must be awaited.\n"
            "    Parallel: results = await asyncio.gather(launch_subagent('a'), launch_subagent('b'))\n"
            "5. def finish(message: str) -> str\n"
            "    Hand back a summary without reporting a file.\n"
        )

    async def reset(self) -> CodeExecutor:
        self.reported = None
        return await super().reset()

    async def close(self) -> None:
        self.files = {}
```

The environment binds that executor and scores the result:

```python title="my_plugin/env.py"
class ConfigAuditEnv(CodeActEnv):
    def __init__(self, task: Task, subagent_max_steps: int = 6):
        super().__init__(task, ConfigAuditExecutor(task, subagent_max_steps=subagent_max_steps))
        self._subagent_max_steps = subagent_max_steps

    async def evaluate(self) -> tuple[float, dict]:
        score = 0.0
        reward_misc: dict = {}
        if self._state.finished:
            reported = self._code_executor.reported
            expected = self._task.misc.get("offending_file")
            score = 1.0 if reported is not None and reported == expected else 0.0
            reward_misc["reported"] = reported
        reward_misc["reward/success"] = score
        return score, reward_misc

    async def fork(self, task: Task) -> "ConfigAuditEnv":
        # Task.fork hands the child the parent's misc dict *by reference*; copy it so the
        # child's bookkeeping cannot mutate the parent's task.
        task.misc = dict(task.misc)
        return ConfigAuditEnv(task, subagent_max_steps=self._subagent_max_steps)
```

Four choices in there worth naming.

**`report()` ends the episode by setting `finish_message` directly.** That is the same mechanism
`finish()` uses, and it means the terminal step's `evaluate()` already sees
`self._state.finished == True`. The alternative — asking the model to call `report(...)` and then
`finish(...)` — costs a step and buys nothing.

**`evaluate()` reads executor state through `self._code_executor`.** The environment and its
executor are a pair; `CodeActEnv` also exposes the executor as a `code_executor` property. Plugins
do the same thing: `TextCraftEnv.evaluate` reads `self._code_executor.inventory`.

**`evaluate()` scores only the finishing step, but always emits `reward/success`.** Intermediate
steps contribute `0.0` so the summing reward processor lands on the right number, and the last
step carries the value that `_get_base_success` and `propagate_root_success` look for.

**`close()` is defined on the executor even though it does almost nothing.**
`IPythonCodeExecutor` never defines `close()`, so it inherits the `CodeExecutor` protocol's
`async def close(self) -> None: ...` — awaiting it is a legal no-op. If your executor holds a
subprocess, a session or a socket, you must implement `close()` yourself, it must tolerate being
called under cancellation, and it must finish inside `EPISODE_CLOSE_TIMEOUT_SECONDS` (10 s) or the
loop logs a timeout and moves on.

## Making it forkable

`launch_subagent` treats the current environment as a `ForkableEnv` and calls `fork(subtask)`. The
`cast` in that code is a type-checker annotation with no runtime effect, so an environment without
a `fork` method does not fail politely — it raises `AttributeError` in the middle of the agent's
tool call.

```python title="platoon/envs/base.py"
@runtime_checkable
class ForkableEnv(Env, Protocol):
    async def fork(self, task: Task) -> ForkableEnv:
        """Return an independently closeable child environment.

        Implementations that allocate resources before returning must clean up
        partial allocations if the fork raises, including on cancellation.
        """
```

That docstring is a binding obligation, not a suggestion, because `launch_subagent` cleans up only
the handles a fork **successfully returned**:

```python title="platoon/agents/actions/subagent.py"
finally:
    # Once the child task starts, run_episode is the sole owner and
    # closes both resources. Before that handoff, close only handles
    # that were successfully returned by their fork methods.
    if not episode_ownership_started.is_set():
        if forked_agent is not None:
            await _close_episode_resource(forked_agent, "forked agent")
        if forked_env is not None:
            await _close_episode_resource(forked_env, "forked environment")
```

A fork that allocates a session, then raises or is cancelled while allocating a second one, has
leaked the first and nobody will ever close it. `ConfigAuditEnv.fork` above is safe by
construction: one constructor call that either returns a complete environment or raises before
returning anything. If your fork does more than that, wrap it and close what you already hold
before re-raising — and catch `BaseException`, not `Exception`, because cancellation arrives as
`CancelledError`.

Three more things to know about forking.

**Ownership.** The instant the child episode's task starts, `run_episode`'s `finally` block becomes
the sole owner of the forked agent and the forked env and will close both. Return something
independently closeable; never hand the child a handle the parent also closes.

**What the child's task looks like.** `Task.fork` branches on `fork_strategy`. The default
`"subtask"` returns a `SubTask` whose `parent_tasks` chain is rendered into the child's prompt;
`"task"` returns a plain `Task` with a fresh uuid and no ancestry. Set it in your `__init__` if you
want context-free children — `TextCraftEnv` does `task.fork_strategy = "task"` unconditionally.
Either way, when the caller passes no `task_misc` the child inherits the parent's `misc` dict **by
reference**, which is why the example copies it.

**Reward for children.** In the example the child inherits `offending_file` and is scored by the
same rule, which is correct here because a child solves the same question over a slice of the same
read-only bundle. If your children get genuinely different sub-goals, `fork` must rewrite
`task.misc` so `evaluate()` scores the child against *its own* goal — TextCraft parses the child's
goal string back into target items for exactly that reason. See
[the sub-agent call walkthrough](../walkthroughs/subagent-call.md) and
[recursive agent recipes](../recipes/recursive.md).

??? note "The inherited `CodeActEnv.fork`, and when it works"
    `CodeActEnv` ships a `fork` that forks the executor and rebuilds the environment:

    ```python title="platoon/envs/codeact/env.py"
    async def fork(self, task: Task) -> CodeActEnv:
        if isinstance(self._code_executor, ForkableCodeExecutor):
            return type(self)(
                task=task,
                code_executor=await self._code_executor.fork(task=task),
                return_obs_copy=self._return_obs_copy,
                parent_state=await self.observe(),
                **deepcopy(self._init_kwargs),
            )
        else:
            raise ValueError(
                "CodeExecutor is not forkable. "
                "Either implement fork() for your CodeExecutor or implement a new ForkableEnv for this task."
            )
    ```

    Using it needs two things. Your executor must implement `fork(task)` — that is all
    `ForkableCodeExecutor` adds. And your `__init__` must accept `task`, `code_executor`,
    `return_obs_copy` and `parent_state` as keyword arguments, passing your own construction knobs
    through as extra `**kwargs` so they are captured in `_init_kwargs` and replayed on the child.

    Every plugin in the repository writes its own env-level `fork` instead, because that is where
    world-state sharing decisions live: TextCraft passes the inventory dict by reference so a
    child's crafting is visible to the parent, and openreward's fork gives a verifier branch
    `shared` access while still stripping the environment-terminal tools.

**The agent must be forkable too.** `launch_subagent` forks the agent first, then the env. A
`CodeActAgent` subclass that follows the house `__init__` pattern already inherits a working
`fork`: it rebuilds `type(self)` with the same prompt builder and a forked LLM client. See
[custom agent](agent.md).

## Wiring it into training

**Environment classes are not registered.** There is no `register_env`. The registry has exactly
six kinds — `dataset_loader`, `task_loader`, `rollout`, `reward_processor`, `workflow` and
`trainer_config` — plus a separate `loss` registry used by the AReaL trainer. Your environment
reaches training because **your rollout function constructs it**, and the rollout function is what
a config can name.

```python title="my_plugin/rollout.py"
import asyncio
import os
from contextlib import suppress

from platoon.config_defs import RolloutConfig
from platoon.envs.base import Task
from platoon.episode.context import current_trajectory_collection
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import TrajectoryCollection
from platoon.utils.llm_client import LiteLLMClient
from platoon.visualization.event_sinks import JsonlFileSink

from .agent import ConfigAuditAgent
from .env import ConfigAuditEnv


async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LiteLLMClient(
            model=config.model_name,
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
        )
        env = ConfigAuditEnv(task)
        agent = ConfigAuditAgent(llm_client=llm_client, inference_params=config.inference_params)

        traj_collection = TrajectoryCollection()
        current_trajectory_collection.set(traj_collection)
        events_path = os.path.join(config.output_dir, "events", f"events_{task.id}_{traj_collection.id}.jsonl")
        traj_collection.register_event_handlers(
            JsonlFileSink(events_path, collection_id=traj_collection.id, process_id=os.getpid())
        )

        rollout_task = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        try:
            _ = await asyncio.wait_for(rollout_task, timeout=config.timeout)
        except asyncio.TimeoutError:
            rollout_task.cancel()
            with suppress(asyncio.CancelledError):
                await rollout_task
            raise

        collection = current_trajectory_collection.get()
        return collection.to_dict() if config.return_dict else collection
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
```

Four constraints on that function come from the environment side:

- Create a fresh `TrajectoryCollection` and set `current_trajectory_collection` **before**
  `run_episode`, because `CodeActEnv.reset` reads it to register the task.
- Launch `run_episode` inside `asyncio.create_task` so its contextvar writes do not leak into the
  caller's context.
- `run_episode(timeout=...)` is the **per-step** deadline; the outer `wait_for` is the
  whole-rollout deadline. They are different knobs (`step_timeout` and `timeout`).
- Close the agent and the env in `finally`. `run_episode` also closes them, so both `close`
  implementations have to tolerate running twice.

Anything you want installed for the whole episode tree — a different budget policy, a sub-agent
reward verifier — is a contextvar set just before `run_episode`:

```python
from platoon.episode.context import budget_tracker
from platoon.episode.trajectory import DepthAwareStepBudgetTracker

budget_tracker.set(DepthAwareStepBudgetTracker(max_depth=4))
```

`run_episode` installs a plain `StepBudgetTracker` only when the var is unset. Under that default,
a child's steps consume the *root's* budget, so the whole tree is capped by the root's
`max_steps`; under the depth-aware tracker every trajectory gets its own budget and only the tree
depth is capped. See [parallelism and budgets](../recipes/parallelism.md).

Register the rollout, then name it from a config:

```python title="my_plugin/registry.py"
from platoon.registry import register_rollout, register_task_loader

from .rollout import run_rollout
from .tasks import get_task

register_task_loader("config_audit/tasks", get_task)
register_rollout("config_audit/linear", run_rollout)
```

```yaml title="my_plugin/configs/config_audit_tinker.yaml"
environments:
  - package: platoon.config_audit.registry
    task_loader: config_audit/tasks
    dataset_loader: config_audit/tasks
    rollout: config_audit/linear
    workflow: group_rollout
```

With no `reward_processor` named, the default is `lambda traj: (traj["reward"], {})`, which is
enough for a binary environment like this one; [custom rewards](rewards.md) covers the processors
that read `reward_misc`. A registered name is not mandatory either — `Registry.resolve` accepts a
dotted import path, so `rollout: platoon.config_audit.rollout.run_rollout` works with no registry
module at all.

=== "AReaL"

    ```bash
    uv run python -m platoon.train.areal.train --config path/to/config_audit_areal.yaml
    ```

    AReaL configs go through `areal.api.cli_args.load_expr_config`. Overrides are OmegaConf style
    with **no** leading dashes:

    ```bash
    uv run python -m platoon.train.areal.train \
      --config path/to/config_audit_areal.yaml \
      trial_name=debug-run train_dataset.batch_size=8
    ```

=== "Tinker"

    ```bash
    uv run python -m platoon.train.tinker.train --config path/to/config_audit_tinker.yaml
    ```

    Tinker configs go through `platoon.utils.config.load_config`. Overrides **require** the
    leading dashes; a bare `key=value` token is silently dropped:

    ```bash
    uv run python -m platoon.train.tinker.train \
      --config path/to/config_audit_tinker.yaml \
      --train.batch_size 8
    ```

Most plugins on this branch still ship their own `train.py` / `train_tinker.py` that wires the
workflow explicitly rather than going through these shared entrypoints; today only textcraft has a
config wired for the registry route. The registry route is the recommended one for new work.
[Packaging a plugin](packaging.md) covers both.

!!! warning "`rollout_config.max_steps` overwrites `task.max_steps`"
    Both workflows do `task.max_steps = config.rollout_config.max_steps` before calling your
    rollout, whenever that key is set. The `max_steps` baked into your dataset is usually *not*
    what runs. And because task loaders conventionally cache `Task` objects in a module global,
    that mutation persists for the life of the process.

## Testing it in isolation

An environment is testable without a model, a trainer or a GPU: set the contextvars the loop
expects, hand `run_episode` a scripted agent, and inspect the returned `Trajectory`. This is how
the repository's own episode tests work
(<span class="pl-src">tests/test_episode_cancellation.py</span>,
<span class="pl-src">tests/test_step_budget_tracker.py</span>).

```python title="tests/test_config_audit_env.py"
import pytest

from platoon.config_audit.env import ConfigAuditEnv
from platoon.envs.base import Observation, Task
from platoon.envs.codeact import CodeActAction
from platoon.episode.context import budget_tracker, current_trajectory_collection, finish_message
from platoon.episode.loop import run_episode
from platoon.episode.trajectory import StepBudgetTracker, TrajectoryCollection


class ScriptedAgent:
    """Replays fixed code cells. Enough to satisfy the Agent protocol."""

    def __init__(self, cells: list[str]):
        self._cells = cells
        self._i = 0

    async def act(self, obs: Observation) -> CodeActAction:
        cell = self._cells[self._i] if self._i < len(self._cells) else "finish('out of script')"
        self._i += 1
        return CodeActAction(action=cell, parsed_code=cell, parsed_thought="")

    async def reset(self) -> None:
        self._i = 0

    async def close(self) -> None:
        return None


def make_task() -> Task:
    return Task(
        goal="Exactly one config file in this bundle sets max_retries above 5. Report its path.",
        id="config_audit.test.0",
        max_steps=4,
        misc={
            "files": {"api.yaml": "max_retries: 9\n", "worker.yaml": "max_retries: 3\n"},
            "offending_file": "api.yaml",
        },
    )


async def run_cells(cells: list[str]):
    collection = TrajectoryCollection()
    collection_token = current_trajectory_collection.set(collection)
    budget_token = budget_tracker.set(StepBudgetTracker())
    finish_token = finish_message.set(None)
    try:
        return await run_episode(ScriptedAgent(cells), ConfigAuditEnv(make_task()), timeout=30)
    finally:
        finish_message.reset(finish_token)
        budget_tracker.reset(budget_token)
        current_trajectory_collection.reset(collection_token)


@pytest.mark.asyncio
async def test_correct_report_scores_one():
    traj = await run_cells(["print(read_file('api.yaml'))", "report('api.yaml')"])

    assert traj.finish_message == "Reported api.yaml."
    assert traj.steps[-1].misc["reward_misc"]["reward/success"] == 1.0
    assert traj.reward == 1.0


@pytest.mark.asyncio
async def test_wrong_report_scores_zero():
    traj = await run_cells(["report('worker.yaml')"])

    assert traj.steps[-1].misc["reward_misc"]["reward/success"] == 0.0


@pytest.mark.asyncio
async def test_budget_exhaustion_halts_without_reward():
    traj = await run_cells(["print(list_files())"] * 10)

    assert len(traj.steps) == 4  # task.max_steps
    assert "Exhausted budget" in traj.error_message
    assert traj.reward == 0.0
```

Four notes on that harness:

- `run_episode` is awaited **directly**, not wrapped in `create_task`, so its contextvar writes
  land in the test's own context and `collection` is inspectable afterwards. Resetting the tokens
  you set keeps tests independent.
- The scripted agent returns a `CodeActAction` because `CodeActEnv.step` reads
  `action.parsed_code`. For a from-scratch environment, return whatever your `step` expects — the
  reference `MockEnv` in the test suite is driven by plain dicts.
- The third test is the one people skip and then debug in production. It asserts that budget
  accounting works, which is another way of asserting that your `step()` really does call
  `add_trajectory_step`.
- `pytest-asyncio` is a dev dependency and the repository has no `asyncio_mode` setting, so each
  coroutine test needs its own `@pytest.mark.asyncio` marker.

Worth adding beyond the three above: a test that your `fork` produces an independently closeable
child, and a test that the string your tools return to the model is the string you meant — the
model only ever sees captured stdout and the return value it prints.

Once the environment passes in isolation, run one real episode end to end through your rollout
function before wiring it into training. [Your first task](../get-started/first-task.md) walks
through that, and the JSONL events written under `{output_dir}/events/` can be replayed with the
[visualization tools](../tutorials/visualization.md).

## See also

- [Agents, environments, episodes](../architecture/agents-envs.md) — the protocols and the loop in
  depth
- [Custom agent](agent.md) — the other half of the pair
- [Custom rollout](rollout.md) — the function that constructs your environment
- [Custom rewards](rewards.md) — turning `reward_misc` into a training signal
- [The fork and sub-agent model](../architecture/subagents.md) — what `fork` is really for
- [Plugin anatomy](../walkthroughs/plugin-anatomy.md) — where `env.py` sits in a plugin
