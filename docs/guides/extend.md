# Extend the framework

Platoon is a set of small contracts. Each one is a plain function or a plain class — no base class
to inherit, no decorator you must apply — and a config field selects it by name. This page maps
every extension point: the contract, a minimal snippet, and the key that picks it. Two things hold
everywhere.

- **Your code lives in your own package.** A plugin is an ordinary Python distribution that installs
  into the `platoon.<name>` namespace. Keep it in your own repository; no fork, no upstreaming.
  [Build your first plugin](first-plugin.md) walks through the packaging.
- **Registration is a convenience, not a requirement.** Every registry field also accepts a dotted
  import path, so `rollout: my_pkg.rollout.run_rollout` works with no `registry.py` at all.
  Registering buys short names that survive a module rename and a helpful error when you typo one.
  The one exception is the loss registry, which takes registered names only.

## The map

| To change | Implement | Selected by |
| --- | --- | --- |
| Which tasks a split contains | `dataset_loader(config, split, **kwargs)` | `dataset_loader:`, `eval_dataset_loader:` |
| What one task id means | `task_loader(task_id) -> Task` | `task_loader:` |
| The world the agent acts in | an `Env` — usually a `CodeActEnv` subclass | your rollout constructs it |
| How the model picks actions | an `Agent` — usually a `CodeActAgent` subclass | your rollout constructs it |
| How one episode is assembled | `async rollout(task, config) -> dict` | `rollout:`, `eval_rollout:` |
| The task score | `evaluate() -> tuple[float, dict]` on your env | part of your env |
| The final scalar and its metrics | `reward_processor(traj) -> tuple[float, dict]` | `reward_processor:` |
| The policy objective | `@register_loss_fn`-decorated function | `loss_fn_config.loss_fn` (AReaL) |
| Group fan-out and batching | a `GroupRolloutWorkflow` subclass | `workflow:` |

All of the `key:` names above are fields of one entry in the top-level `environments:` list —
registry wiring, described in full in the [configuration reference](../reference/configuration.md).

## Dataset and task loaders

Two functions split one job. The **dataset loader** runs once per split in the trainer process and
decides which tasks that split contains. The **task loader** runs once per rollout, possibly in
another process, and turns one id into a `Task`. Keeping ids small and opaque is what makes that
split cheap.

```python title="my_plugin/registry.py"
from platoon.envs.base import Task
from platoon.registry import register_dataset_loader, register_task_loader

@register_dataset_loader("my_plugin/default")
def load_dataset(config, split: str, limit: int | None = None) -> list[str]:
    split_name = "val" if split == "eval" else split
    ids = [f"my_plugin.{split_name}.{i}" for i in range(1000)]
    return ids[:limit] if limit else ids

@register_task_loader("my_plugin/default")
def load_task(task_id: str) -> Task:
    return Task(goal=..., id=task_id, max_steps=8, misc={...})
```

The dataset loader is called with the literal split `"train"` or `"eval"` and gets
`dataset_kwargs` or `eval_dataset_kwargs` splatted in; translate your own split names inside it.
Return a `list` of id strings and Platoon builds the Hugging Face dataset for you, or return a
`datasets.Dataset` yourself when rows must carry extra columns — only `task_id` is read.

The task loader is **synchronous** and returns a `Task` dataclass: `goal`, `id`, `max_steps`,
`misc` and `fork_strategy`. Put everything your environment needs in `misc`. Set `id` to the id you
were given; event filenames and reward code assume it matches.

`eval_dataset_loader` falls back to `dataset_loader` when unset, but the two kwargs blocks are
independent — repeat every kwarg both splits need.

## Environment

An environment owns the world: the tools available, what the agent sees after using them, when the
episode ends, and what the trajectory scores. `Env` is a `Protocol` with five members —
`reset`, `step`, `close`, `observe`, and a `task` property — so any object carrying them qualifies.

For anything that looks like tool calling, subclass `CodeActEnv` instead of writing that from
scratch. It implements the whole protocol and delegates the two things only you know: the action
space (a `CodeExecutor`) and the reward (`evaluate`).

```python title="my_plugin/env.py"
from platoon.agents.actions.common import finish
from platoon.envs.codeact import CodeActEnv, IPythonCodeExecutor

class AuditExecutor(IPythonCodeExecutor):
    def __init__(self, task):
        self.files = task.misc["files"]
        super().__init__(task, actions=(finish, self.list_files, self.read_file))

    def list_files(self) -> list[str]:
        """Return every file path in the bundle."""
        return sorted(self.files)

    def read_file(self, path: str) -> str:
        """Return one file's contents."""
        return self.files.get(path, f"No such file: {path}")

    async def describe_action_space(self) -> str:
        return "1. list_files() -> list[str]\n2. read_file(path: str) -> str"

class AuditEnv(CodeActEnv):
    def __init__(self, task):
        super().__init__(task, AuditExecutor(task))
```

Actions are plain Python callables injected into the agent's IPython namespace by `__name__`, so
module-level functions, closures and bound methods all work. `describe_action_space()` is the only
documentation the model gets for them — the base implementation returns an empty string, so
override it or describe your tools in the system prompt.

Environments are not registered. Yours reaches training because your rollout constructs it. To let
an agent delegate, add `async def fork(self, task) -> Env` returning an independently closeable
child; see [multi-agent workflows](multi-agent.md).

## Agent

The agent turns an observation into an action: `act`, `reset` and `close`, plus `fork` if it
delegates. Almost every task subclasses `CodeActAgent` and changes only the prompt, which lives in
a `CodeActPromptBuilder`.

```python title="my_plugin/agent.py"
from platoon.agents.codeact import CodeActAgent, CodeActPromptBuilder

class AuditPromptBuilder(CodeActPromptBuilder):
    def build_system_prompt(self, obs, **context) -> str:
        context.setdefault("env_specific_system_context", "You audit configuration bundles.")
        return super().build_system_prompt(obs, **context)

class AuditAgent(CodeActAgent):
    def __init__(self, **kwargs):
        kwargs.setdefault("prompt_builder", AuditPromptBuilder())
        super().__init__(**kwargs)
```

`setdefault` rather than assignment matters: `CodeActAgent.fork` rebuilds the child with
`type(self)(prompt_builder=self.prompt_builder, ...)`, so the subclass must accept a builder from
outside. If you add constructor state of your own, override `fork` to carry it to children.

One requirement is load-bearing on both backends. Platoon never re-tokenizes prompts — training
data comes from the tokens inference actually sampled, joined to your step by a completion id. Send
requests through the `model_name`, `model_endpoint` and `model_api_key` handed to you in the
`RolloutConfig`, and put the response id into `action.misc["completion_id"]`. `CodeActEnv.step`
copies `action.misc` onto the step; a from-scratch environment must do the same. An agent that
builds its own client against a different endpoint produces trajectories and rewards, and zero
trainable tokens.

## Rollout function

The rollout function is the piece every part of Platoon calls: training workflows, the inference
harness, subprocess workers. It receives a `Task` and a `RolloutConfig`, builds the LLM client,
environment and agent, runs one episode, and returns the serialized trajectory tree.

```python title="my_plugin/rollout.py"
async def run_rollout(task: Task, config: RolloutConfig) -> dict | TrajectoryCollection:
    agent = env = None
    try:
        llm_client = LiteLLMClient(config.model_name, config.model_endpoint, config.model_api_key)
        env = AuditEnv(task)
        agent = AuditAgent(llm_client=llm_client, inference_params=config.inference_params)

        collection = TrajectoryCollection()
        current_trajectory_collection.set(collection)
        events = os.path.join(config.output_dir, "events", f"events_{task.id}_{collection.id}.jsonl")
        collection.register_event_handlers(JsonlFileSink(events, collection_id=collection.id))

        episode = asyncio.create_task(run_episode(agent, env, timeout=config.step_timeout))
        await asyncio.wait_for(episode, timeout=config.timeout)
        return collection.to_dict() if config.return_dict else collection
    finally:
        if agent is not None:
            await agent.close()
        if env is not None:
            await env.close()
```

Five points are load-bearing. Define it as a module-level `async def` — AReaL ships it to rollout
workers by import path, so a lambda or a closure cannot travel. Create the `TrajectoryCollection`
and set the contextvar *before* the episode; the environment reads it to register the task. Wrap
`run_episode` in `asyncio.create_task` so its contextvar writes stay inside the episode.
`step_timeout` is the per-step deadline, `timeout` the whole-rollout one. And return `to_dict()`
when `config.return_dict` is set — every harness forces it.

Keeping the `{output_dir}/events/events_{task.id}_{collection.id}.jsonl` convention is what makes
the replay tools work against your plugin with no extra wiring; see
[inspect rollouts](inspect-rollouts.md). `config.extra` is the escape hatch for plugin-specific
settings — store plain data only, since it may cross a process boundary.

## Reward

Reward enters in two places you will write. `evaluate()` on your environment answers "did the agent
do the task"; a registered `reward_processor` collapses a finished trajectory into the scalar that
trains it plus the metrics you want logged.

```python title="my_plugin/env.py"
    async def evaluate(self) -> tuple[float, dict]:
        score, reward_misc = 0.0, {}
        if self._state.finished:                       # gate: this runs on every step
            score = 1.0 if self._code_executor.reported == self._task.misc["expected"] else 0.0
        reward_misc["reward/success"] = score
        return score, reward_misc
```

`evaluate()` runs on **every** step, so gate anything expensive on `self._state.finished`. The
float accumulates into `Trajectory.reward`; the dict lands verbatim in `step.misc["reward_misc"]`.
Keys prefixed `reward/` are the ones downstream code reads — `reward/success` is the canonical
scalar — and their values must be numeric. Everything else is free-form diagnostics.

```python title="my_plugin/registry.py"
from platoon.registry import register_reward_processor

@register_reward_processor("my_plugin/success")
def reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    rewards: dict[str, float] = {}
    for step in traj["steps"]:
        for key, value in step.get("misc", {}).get("reward_misc", {}).items():
            if key.startswith("reward/"):
                rewards[key] = rewards.get(key, 0.0) + float(value)
    return rewards.get("reward/success", float(traj.get("reward", 0.0))), rewards
```

The argument is one serialized trajectory dict. Because this processor **sums** `reward/*` across
steps, emit a non-zero `reward/success` only on the terminal step. Keep the function pure and
defined at module level: AReaL calls it twice on the root trajectory, and ships it to workers by
import path. Leave `reward_processor:` unset and you get `lambda traj: (traj["reward"], {})`. The
returned dict is reporting-only — the trainer strips every metric key before the batch reaches the
optimizer. Judged rewards for delegated subtasks live in [multi-agent workflows](multi-agent.md).

## Loss function

<span class="pl-tag pl-tag--areal">AReaL</span> The policy loss is swappable by name. Write a
function, register it, and select it from YAML; the rollout, advantage computation and minibatch
loop are untouched.

```python title="my_plugin/losses.py"
from platoon.train.areal.loss_functions import register_loss_fn

@register_loss_fn("asymmetric_cispo", defaults={"positive_cap": 2.0, "negative_cap": 5.0})
def asymmetric_cispo(logprobs, entropy, input_data, positive_cap=2.0, negative_cap=5.0, **kwargs):
    old_logprobs = input_data["logprobs"]
    advantages = input_data["advantages"].detach()
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0.0)
    cap = torch.where(advantages > 0, positive_cap, negative_cap)
    coefficient = torch.minimum(ratio, cap).detach()
    pg_loss = -coefficient * advantages * logprobs
    return torch.where(loss_mask, pg_loss, 0.0).sum() / (loss_mask.count_nonzero() or 1)
```

```yaml
loss_fn_config:
  loss_fn: asymmetric_cispo
  loss_fn_kwargs:
    positive_cap: 1.5
```

Gradient must reach the objective only through the bare `logprobs` factor; detach everything you
multiply it by. Return a mean over this microbatch's own valid tokens — the engine rescales each
microbatch by its share of the total. Give the function a `**kwargs` sink: the actor offers every
loss the same bundle of PPO-style kwargs from `actor.*`, and those win on any shared key.

Name the module in `environments[0].package` so it is imported where the loss is built. `grpo`,
`ppo` and `cispo` ship registered. Tinker-compatible backends run the loss inside the service, so
`train.loss_fn` there names an objective the backend provides and `@register_loss_fn` does not
apply — shape the reward instead.

## Workflow

!!! warning "Advanced seam"
    The workflow decides how many rollouts a task gets, how their trajectories become datums, and
    how rewards are centered within the group. Most changes people reach for it to do are already
    `workflow_config` keys — `group_size`, `leave_one_out_baseline`, `filter_zero_advantage_datums`
    and the rest — or belong in a reward processor. Check those first.

What only a workflow sees is the group — per-member root rewards, before members from different
tasks are concatenated — so a different baseline or an adaptive group size lives here and nowhere
else.

The backends do not share a base class, so a workflow class works with exactly one of them.
Subclass the matching `GroupRolloutWorkflow`, override `arun_episode`, and call `super()` for the
real work. On AReaL the first five constructor arguments are positional and the class is rebuilt on
remote workers from `to_workflow_kwargs()` — override that too if you add constructor arguments, or
they silently revert to their defaults. On Tinker the object is used in process, and every argument
arrives by keyword.

```yaml
environments:
  - package: my_pkg.workflow
    workflow: my_pkg.workflow.TokenBudgetGroupRolloutWorkflow
    workflow_kwargs:
      max_group_tokens: 200000
```

`workflow: group_rollout` is the default and a sentinel: it means "whichever `GroupRolloutWorkflow`
matches the backend I am running". `workflow_kwargs` and `eval_workflow_kwargs` are independent —
set both when a parameter applies to both splits.

## Next

- [Build your first plugin](first-plugin.md) — these pieces assembled into a package
- [Multi-agent workflows](multi-agent.md) — forking, delegation budgets and judged subtasks
- [Components](../architecture/components.md) — the registry and the `Auto*` factories
- [Configuration reference](../reference/configuration.md) — every key named above
