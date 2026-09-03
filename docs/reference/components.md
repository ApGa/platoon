# Component contracts

Seven registry kinds exist. This page is the contract sheet for each: the exact signature, who
calls it, what it may assume, what it must return, and which config key selects it. For how the
registry resolves names at all, see [registry and Auto factories](../architecture/registry.md).

## Summary

| Kind | Registration helper | Selected by | Required | Contract |
| --- | --- | --- | --- | --- |
| `dataset_loader` | `register_dataset_loader` | `environments[0].dataset_loader` | yes | `(config, split, **kwargs) -> list[str] \| Dataset` |
| `task_loader` | `register_task_loader` | `environments[0].task_loader` | yes | `(task_id: str) -> Task` |
| `rollout` | `register_rollout` | `environments[0].rollout` | yes | `async (task: Task, config: RolloutConfig) -> dict` |
| `reward_processor` | `register_reward_processor` | `environments[0].reward_processor` | no | `(traj: dict) -> tuple[float, dict[str, float]]` |
| `workflow` | `register_workflow` | `environments[0].workflow` | no | a class matching the backend workflow constructor |
| `trainer_config` | `register_trainer_config` | `environments[0].trainer_config` | no | a config dataclass type — **read by nothing today** |
| `loss` | `register_loss_fn` | `loss_fn_config.loss_fn` <span class="pl-tag pl-tag--areal">AReaL</span> | no | `(logprobs, entropy, input_data, **kwargs) -> Tensor` |

The first six helpers live in <span class="pl-src">platoon/registry.py</span> and are thin wrappers
over `register_component(kind, name, value)`. `register_loss_fn` is separate — it lives in
<span class="pl-src">platoon/train/areal/loss_functions.py</span> and builds its registry directly
with `get_registry("loss")`.

`environments` here is the top-level `list[EnvironmentConfig]` on the trainer config. It is not
openreward's nested `environments:` mixture list, which is an unrelated task-source config; see
[the openreward integration](../integrations/openreward.md).

!!! warning "Two of these seven are declared but not wired up"

    `resolve_component("trainer_config", ...)` appears nowhere in the tree, so setting
    `environments[0].trainer_config` is inert. And nothing in the repository calls
    `register_workflow`, so the `workflow` registry is empty at runtime — a non-default value for
    that key resolves through the import-path fallback, not through a registered name. Details in
    each section below.

## Registration, in two forms

Every helper takes an optional `value`. Pass it and registration happens immediately; omit it and
you get a decorator back. Both forms appear in the one real registration module in the repo,
<span class="pl-src">plugins/textcraft/platoon/textcraft/registry.py</span>:

```python
@register_task_loader("textcraft/synth")
def load_synth_task(task_id: str):
    return get_synth_task(task_id)


register_rollout("textcraft/synth/linear", run_synth_rollout)
```

Names are arbitrary strings. `Registry.register` raises on a duplicate name unless you pass
`exist_ok=True`, so namespace yours (`"<plugin>/<variant>"`) — two installed plugins that both
claim `"default"` will crash at import time.

### The dotted-import-path escape hatch

You do not have to register anything. `Registry.resolve` checks the registry first and falls
through to `import_from_string` for any name it does not recognize, and that accepts both
`package.module.attr` and `package.module:attr`. A complete `environments` block with zero
registrations and no `package`:

```yaml
environments:
  - dataset_loader: my_pkg.components.load_dataset
    task_loader: my_pkg.components.load_task
    rollout: my_pkg.components.run_rollout
    reward_processor: my_pkg.components.score
```

Registering buys you three things: a short name that survives module reshuffling, a listing of valid
names in the `ValueError` when you typo one, and an `import_path` recorded on the `RegistryItem`.
Nothing else.

!!! danger "AReaL requires importable, module-level callables"

    On the AReaL path the workflow is shipped to worker processes as import paths, not as a pickle.
    `GroupRolloutWorkflow.to_workflow_kwargs` converts `rollout_fn` and `get_task_fn` back into
    dotted strings with `callable_import_path`
    (<span class="pl-src">platoon/train/areal/workflow_serialization.py</span>) and raises
    `ValueError("GroupRolloutWorkflow requires importable rollout_fn/get_task_fn")` when it cannot.

    So do not register a lambda, a `functools.partial`, a closure, or a function defined inside
    another function. `callable_import_path` reads `__name__` and `__module__`; a partial has
    neither, and a lambda's `__name__` is rejected outright. `infer_import_path` in
    <span class="pl-src">platoon/registry.py</span> has the matching restriction and returns `None`
    for anything with `<locals>` in its qualname or defined in `__main__`.

    A reward processor with no import path is not fatal — it is dropped from the worker kwargs and
    the worker falls back to the class default. Rollouts and task loaders are fatal.

## dataset_loader

```python
def __call__(self, config: Any, split: str, **kwargs: Any) -> Any: ...
```

The `DatasetLoader` protocol in <span class="pl-src">platoon/train/components.py</span>.

**Who calls it.** `AutoDataset.from_config` in <span class="pl-src">platoon/train/auto.py</span>,
twice per run — once with `split="train"` and once with `split="eval"`. Both entrypoints do this
before the trainer exists.

**What it may assume.** `config` is the fully parsed trainer config object. `split` is the literal
string `"train"` or `"eval"` and nothing else — if your data calls that split `"val"`, translate
inside the loader, as textcraft's `_get_filtered_synth_task_ids` does. `**kwargs` is
`dataset_kwargs` on the train call and `eval_dataset_kwargs` on the eval call.

**What it must guarantee.** Return either a `list` of task-id strings, which `AutoDataset` converts
via `task_ids_to_dataset` into `Dataset.from_list([{"task_id": task_id} for ...])`, or a dataset
object, which is passed straight through — in which case every row must carry a `task_id` key,
because that is what the workflow reads.

**Selected by** `environments[0].dataset_loader`; the eval call prefers `eval_dataset_loader` and
falls back to `dataset_loader`. Required — an unset spec raises
`ValueError("Config must set environments[0].dataset_loader")`.

```python
@register_dataset_loader("mytask/default")
def load_mytask_dataset(config, split: str, limit: int | None = None):
    task_ids = get_task_ids("val" if split == "eval" else split)
    return task_ids[:limit] if limit is not None else task_ids
```

!!! warning "`eval_dataset_kwargs` does not inherit from `dataset_kwargs`"

    The *loader* falls back for eval. The kwargs do not — `eval_dataset_kwargs` defaults to `{}` and
    is used as-is, so the eval call gets your function's own parameter defaults for anything you set
    only in `dataset_kwargs`. That is why the live textcraft config repeats `num_samples_train` and
    `num_samples_val` in both blocks.

More on shaping the dataset: [custom datasets](../customization/dataset.md).

## task_loader

```python
def __call__(self, task_id: str) -> Task: ...
```

**Who calls it.** Resolved by `AutoTaskLoader.from_config`, handed to the workflow as `get_task_fn`,
and called **synchronously** inside `arun_episode` for every dataset row. An `async` task loader
breaks it.

**What it may assume.** Nothing beyond the id string. It gets no config, so anything it needs must
come from module state or from what the id encodes.

**What it must guarantee.** Return a `Task` from <span class="pl-src">platoon/envs/base.py</span>
(`goal`, `id`, `max_steps`, `misc`, `fork_strategy`). A `SubTask` is fine — it subclasses `Task`.
The loader is called once per row per epoch, so memoize anything expensive.

**Selected by** `environments[0].task_loader`. Required:
`ValueError("Config must set environments[0].task_loader")` when unset.

```python
@register_task_loader("mytask/default")
def load_mytask_task(task_id: str) -> Task:
    return Task(id=task_id, goal=GOALS[task_id], max_steps=25)
```

## rollout

```python
async def __call__(self, task: Task, config: RolloutConfig) -> dict: ...
```

The `RolloutFn` protocol is not annotated `async`, but both workflows `await` the result, so a
synchronous function fails at rollout time.

**Who calls it.** `GroupRolloutWorkflow` on each backend, with **exactly two positional
arguments** — the task and the rollout config. There is no `rollout_kwargs` field on
`EnvironmentConfig`, so extra parameters are reachable only through their defaults. To vary them,
register a second name bound to a differently parameterized function; textcraft's three rollout
registrations are exactly that.

**What it may assume.** `config` is a `RolloutConfig` copy the workflow has already populated:
`model_name`, `model_endpoint` and `model_api_key` point at the live sampling endpoint, and both
workflows force `return_dict = True` and `train = True` regardless of what your YAML said.
`output_dir` has been rewritten to a per-run location. Field-by-field types and defaults are in the
[configuration reference](configuration.md).

**What it must guarantee.** Return the serialized trajectory collection —
`current_trajectory_collection.get().to_dict()`, whose shape is
`{"id": ..., "trajectories": {...}}`. Close the agent and the environment in a `finally` block; the
workflow will not do it for you.

**Selected by** `environments[0].rollout`; the eval workflow prefers `eval_rollout` and falls back
to `rollout`. Required.

```python
async def run_mytask_rollout(task: Task, config: RolloutConfig) -> dict:
    agent = MyTaskAgent(llm_client=LiteLLMClient(
        model=config.model_name, base_url=config.model_endpoint, api_key=config.model_api_key,
    ))
    env = MyTaskEnv(task)
    current_trajectory_collection.set(TrajectoryCollection())
    try:
        await run_episode(agent, env, timeout=config.step_timeout)
        return current_trajectory_collection.get().to_dict()
    finally:
        await agent.close()
        await env.close()


register_rollout("mytask/default", run_mytask_rollout)
```

The full shape of a real rollout, including event sinks and timeout handling, is in
[custom rollouts](../customization/rollout.md).

## reward_processor

```python
RewardProcessor = Callable[[dict[str, Any]], tuple[float, dict[str, Any]]]
```

**Who calls it.** Not the workflow directly — the data converters,
<span class="pl-src">platoon/utils/areal_data_processing.py</span> and
<span class="pl-src">platoon/utils/tinker_data_processing.py</span>, once per **trainable
trajectory** in the collection. A recursive rollout with subagents therefore invokes it several
times per episode: once for the root and once for each retained child. AReaL calls it again on the
root trajectory when computing root-level statistics.

**What it may assume.** `traj` is one `Trajectory` serialized through `_to_jsonable`: keys `id`,
`task`, `parent_info`, `steps`, `reward`, `finish_message`, `error_message`, `misc`. Each entry in
`steps` is a `TrajectoryStep` dict whose only guaranteed key is `misc` — everything else depends on
what your environment attached, so use `.get` rather than indexing. The rest of the shape is in the
[trajectory schema](schemas.md).

**What it must guarantee.** Return `(scalar_reward, metrics_dict)`. The float becomes the
trajectory's reward for advantage computation; the dict is logged as reward metrics. Keep it
deterministic — the converters call it before datum sampling specifically so that logged rewards do
not depend on which child datums survive the draw.

**Selected by** `environments[0].reward_processor`. **Optional.** When unset,
`AutoRewardProcessor.from_config` returns `lambda traj: (traj["reward"], {})`, which is identical to
the default parameter on both workflow classes — the accumulated per-step reward is used as-is and
no reward metrics are logged.

```python
@register_reward_processor("mytask/success")
def mytask_reward_processor(traj: dict) -> tuple[float, dict[str, float]]:
    metrics: dict[str, float] = {}
    for step in traj["steps"]:
        for key, value in step.get("misc", {}).get("reward_misc", {}).items():
            if key.startswith("reward/"):
                metrics[key] = metrics.get(key, 0.0) + float(value)
    if not metrics:
        return float(traj.get("reward", 0.0)), {}
    return metrics.get("reward/success", 0.0), metrics
```

Shaping rewards from these pieces is covered in [reward design](../customization/rewards.md).

## workflow

**Who calls it.** `AutoWorkflow.from_config(config, default=GroupRolloutWorkflow)`. The registered
value must be a **class**, and it is instantiated twice — once for train, once for eval — by the
entrypoint, so its `__init__` must accept exactly what that entrypoint passes.

=== "AReaL"

    <span class="pl-src">platoon/train/areal/train.py</span> passes five positionally
    (`rollout_fn`, `get_task_fn`, `config.workflow_config`, `trainer.proxy_base_url`,
    `trainer.proxy_admin_api_key`), then `output_subdir`, `filter_errors` and `reward_processor` by
    keyword, then splats the rest of `workflow_kwargs`. The class must also satisfy AReaL's
    `RolloutWorkflow` interface and, for multi-worker runs, the `RemoteWorkflowSerializable`
    protocol in <span class="pl-src">platoon/train/areal/workflow_serialization.py</span> by
    implementing `to_remote_workflow() -> tuple[type, dict]`.

=== "Tinker"

    <span class="pl-src">platoon/train/tinker/train.py</span> passes everything by keyword:
    `rollout_fn`, `get_task_fn`, `config` (from `config.train.workflow_config` or
    `config.eval.workflow_config`), `model_info`, `log_path`, `stats_scope`, `filter_errors`,
    `reward_processor`, then the rest of `workflow_kwargs`. Its entry point into training is
    `async def arun_episode(self, data: dict) -> list[tinker.Datum]`.

The realistic way to satisfy either is to subclass the backend's `GroupRolloutWorkflow` and override
one method; see [custom workflows](../customization/workflow.md).

**Selected by** `environments[0].workflow`, default `"group_rollout"`. That string is a **sentinel,
not a registry entry** — it selects the `default` class the entrypoint passed in, which is the
backend-appropriate `GroupRolloutWorkflow`. Any other value goes to the `workflow` registry.

!!! note "The workflow registry is empty"

    No `register_workflow` call exists in the repository. Since `Registry.resolve` falls through to
    `import_from_string` on a miss, `workflow: my_pkg.workflows.MyWorkflow` works today with no
    registration at all — but a bare name fails unless your `package` module registered it first.
    This is the least exercised extension point in the codebase; nothing tests a non-default
    workflow class.

**Extra constructor arguments** come from `workflow_kwargs` / `eval_workflow_kwargs`, splatted
verbatim after the entrypoint pops the keys it owns (`output_subdir` and `filter_errors` on AReaL,
`stats_scope` and `filter_errors` on Tinker). An unrecognized key becomes a `TypeError` from your
constructor. Note the split-dependent defaults: `filter_errors` is `True` for train and `False` for
eval on both backends.

## trainer_config

```python
TrainerConfigClass = type[Any]
```

`register_trainer_config(name, cls)` exists, `EnvironmentConfig.trainer_config` exists, and textcraft
registers `textcraft/synth/areal` and `textcraft/synth/tinker` under it. Nothing reads any of it:
there is no `resolve_component("trainer_config", ...)` and no `AutoTrainerConfig` anywhere in the
tree. Setting the key in YAML — as the live textcraft Tinker config does — has no effect.

The trainer config class is fixed by the entrypoint module you run: `PlatoonArealRLTrainerConfig` for
`python -m platoon.train.areal.train`, `PlatoonTinkerRLTrainerConfig` for
`python -m platoon.train.tinker.train`. Treat this kind as reserved for a future unified entrypoint
and do not depend on it.

Textcraft's two registrations are still worth copying for their shape — both sit inside
`try/except Exception: pass`, so importing the registration module does not require both training
backends to be installed.

## loss <span class="pl-tag pl-tag--areal">AReaL</span>

```python
def loss_fn(logprobs, entropy, input_data: dict, **kwargs) -> torch.Tensor: ...
```

**Who calls it.** `_make_loss_fn` on `PlatoonActorImpl` in
<span class="pl-src">platoon/train/areal/actor.py</span>, through `build_loss_fn`, once per actor
construction. `build_loss_fn` merges the registered `defaults`, then your `loss_fn_kwargs`, then the
actor's `common_kwargs` (clip epsilons, importance-sampling level, and the rest), drops any key your
signature does not accept, and returns a `functools.partial`.

**What it may assume.** `input_data` carries at least `advantages` and `loss_mask`, alongside the
packed-sequence tensors AReaL's actor supplies; the registered `cispo` implementation shows the full
set it touches. Anything else must arrive through `loss_fn_kwargs`.

**What it must guarantee.** Return a scalar `torch.Tensor` that AReaL can call `.backward()` on.

**Selected by** `loss_fn_config.loss_fn`, with `loss_fn_config.loss_fn_kwargs` for its arguments;
`__post_init__` on `PlatoonArealRLTrainerConfig` copies both onto `actor`. Optional — the default on
`LossFnConfig` is `"grpo"`. Registered names today: `cispo`, `grpo`, `ppo`.

```python
@register_loss_fn("my_loss", defaults={"beta": 0.1})
def my_loss_fn(logprobs, entropy, input_data, beta: float = 0.1, **kwargs):
    advantages = input_data["advantages"].detach()
    loss_mask = input_data.get("full_loss_mask", input_data["loss_mask"]).bool()
    pg_loss = -beta * advantages * logprobs
    return torch.where(loss_mask, pg_loss, 0.0).sum() / (loss_mask.count_nonzero() or 1)
```

Two things differ from the other six kinds. `register_loss_fn` passes `exist_ok=True`, so
re-registering a name overrides it instead of raising — that is how you replace `grpo` with your own.
And `signature_fn` lets a thin `**kwargs` wrapper borrow the upstream function's signature for the
kwarg filtering, which is why the `grpo` and `ppo` registrations pass
`signature_fn=upstream_grpo_loss_fn`.

!!! warning "Tinker's `train.loss_fn` is a different mechanism"

    It looks like the same idea and it is not. The Tinker trainer forwards the string straight to
    `training_client.forward_backward_async(loss_fn=...)`, so the name must be one Tinker's service
    understands. This registry is never consulted on that path, and registering a loss here does not
    make it available to Tinker. See [loss functions](../customization/loss.md).

## What is actually wired up today

The registry layer is new on this branch and the repository has not caught up with it.

| Kind | Entries in the repo | Consumed by code |
| --- | --- | --- |
| `dataset_loader` | `textcraft/synth` | yes |
| `task_loader` | `textcraft/synth` | yes |
| `rollout` | `textcraft/synth/linear`, `.../recursive`, `.../depth_aware` | yes |
| `reward_processor` | `textcraft/synth/delegation_capped` | yes |
| `workflow` | none | yes, but only via the sentinel or an import path |
| `trainer_config` | `textcraft/synth/areal`, `textcraft/synth/tinker` | no |
| `loss` | `cispo`, `grpo`, `ppo` | yes, AReaL only |

One plugin registers components (textcraft) and one live YAML uses an `environments:` block — its
Tinker config. The AReaL twin of that block exists but is commented out, so the only registry path
exercised end to end in the repository today is the Tinker one. Every other plugin still ships a
bespoke `train_*.py` script that wires these callables together by hand.

## See also

- [Registry and Auto factories](../architecture/registry.md) — how a name becomes an object.
- [Configuration reference](configuration.md) — every `EnvironmentConfig` and `RolloutConfig` field.
- [Plugin anatomy](../walkthroughs/plugin-anatomy.md) — where a registration module sits in a plugin.
- [Packaging a plugin](../customization/packaging.md) — `package` versus entry-point discovery.
