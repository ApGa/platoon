# Extending platoon

Platoon is built to be extended from your own code. A harness, an environment, an episode program, a reward rule,
a judge, a sandbox provider, a renderer or a tool-call parser can live in your own package: you name it in a config,
and platoon finds it when the run starts. None of this needs a fork of platoon or a change to it. This page lists
every extension point, how platoon finds your code, and the rules every extension follows; each row of the table
links the guide that walks through it.

## Extension points

| You add | You write | Platoon finds it through | A config selects it with | Guide |
|---|---|---|---|---|
| A function harness | `@pt.harness(name)` on `async def fn(ctx: pt.AgentContext) -> pt.AgentResult` | `package:`, `module:function`, or a `platoon.harnesses` entry point | `harness:` and `harness_options:`; code selects it with `ep.run(harness=...)` and `ctx.agents.spawn(harness=...)` | [Write a function harness](../tutorials/function-harness.md) |
| A command-line agent | a TOML harness spec, with its settings as `[options]` | a `.toml` path, or a `platoon.harnesses` entry point to the file or its directory | `harness:`, `harness_options:` (checked against its `[options]`), `placement:` and `builtins:` | [Wrap a command-line agent](../tutorials/cli-harness.md) |
| CodeAct prompts | a `CodeActPromptBuilder` subclass | `module:Class` | `harness_options.prompt_builder` and `prompt_builder_options` | [Built-in harnesses](../tutorials/builtin-harnesses.md#platoon-codeact) |
| An environment | an OpenReward (ORS) `Environment` class | `python://module:Class` | `service:` and `service_kwargs:` | [Write a custom environment](../tutorials/custom-environment.md) |
| An environment on its own servers | an ORS environment served by openreward's `Server` | `ors+http://HOST:PORT/EnvName` and a pool of server URLs | `service:` and `service_kwargs:` | [Run environments on env servers](../tutorials/remote-environments.md) |
| A step environment | an object with async `reset(task)` and `step(action)` | `pt.gym.loop` (a harness) or `pt.gym.as_environment` (an ORS class) | `harness:` with `program: gym`, or `service: python://...` | [Gym adapters and task tables](../tutorials/gym-and-task-tables.md) |
| Tasks without tools | JSONL rows, or a function that returns `task_table(rows)` | `service: table`, or `module:function` | `service_kwargs: {path: ...}`, and an episode program that grades | [Gym adapters and task tables](../tutorials/gym-and-task-tables.md#step-4-train-on-a-task-table) |
| Harbor tasks | Harbor task directories | `service: harbor` | `service_kwargs:` `dataset_dir` or `task_dirs`, `splits`, `provider` | [Sandboxes and Harbor tasks](../how-to/sandboxes-and-harbor.md) |
| A named environment (since 0.2.0) | `pt.EnvironmentEntry(name, version=..., **keys)`: an environment entry's keys, with the options each harness needs | a `platoon.environments` entry point, or `module:attr` | `env: name@version` | [Named environments](#named-environments) |
| An environment service | classes that follow `EnvironmentService` and `EnvSession` | `module:Class`, or a function that returns a service | `service:` and `service_kwargs:` | [Write an environment service](environment-services.md) |
| An episode program | `@pt.program(name)` on `async def fn(ep: pt.EpisodeContext) -> pt.Outcome` | `package:`, `module:function`, or a `platoon.programs` entry point (since 0.2.0) | `program:`; its settings in the rollout config's `extra` | [Episode programs](../tutorials/programs.md) |
| A finalizer | `fn(collection: pt.TrajectoryCollection)`, sync or async | added by an episode program: `ep.add_finalizer(fn)` | the program | [Reward processors and finalizers](rewards.md#write-a-finalizer) |
| A reward processor | `fn(trajectory: dict) -> (reward, {component: value})` | `module:function`, or a `platoon.reward_processors` entry point (since 0.2.0) | `reward_processor:` | [Reward processors and finalizers](rewards.md#write-a-reward-processor) |
| Verifier agents | nothing new: your harness runs with a verifier's goal | built in | `agents.child_grading: verifier` and `agents.verifier` | [Judges and verifiers](judges-and-verifiers.md#verifier-agents) |
| A behavior or parent-usage judge | an object with async `judge(...)` or `judge_parent_usage(...)` | set by an episode program: `ep.behavior_judge`, or the `apply_subagent_parent_usage_gates` finalizer | the program | [Judges and verifiers](judges-and-verifiers.md) |
| A sandbox provider | classes that follow `SandboxProvider` and `Sandbox` | a built-in name or `module:Class` | the Harbor service's `provider` and `provider_options` | [Write a sandbox provider](sandbox-providers.md) |
| A renderer | a tinker-cookbook `Renderer`, registered with `register_renderer(name, factory)` | registered when `package:` imports its module | `train.renderer_name` (`inference.renderer_name`) and `renderer_kwargs` | [Renderers and tool-call parsers](renderers.md) |
| A tool-call parser | an object with `parse(text, *, tools=None, in_reasoning=False)` that returns a `ParseResult` | `module:attr` | `renderer_kwargs.tool_call_parser`, with the `chat_template` renderer | [Models and tool calling](../how-to/models-and-tool-calling.md#write-your-own-parser) |
| Hooks | functions registered with `platoon.hooks.register(name, fn)` | registered when their module is imported | nothing: they run once registered | [Hooks, telemetry and metrics](hooks-and-metrics.md) |
| Metrics | reward components (`reward/<name>`) and telemetry counters | the reward processor and the telemetry hook | `stats:` (the console and Weights & Biases) | [Hooks, telemetry and metrics](hooks-and-metrics.md#metrics) |

### For advanced use

These seams change how training works. Their interfaces may still change between releases, so pin the platoon
version you build against.

| You add | You write | A config selects it with | Where it is described |
|---|---|---|---|
| A credit rule | a `GroupRolloutWorkflow` subclass that overrides `compute_baselines` or `assign_advantages` | `environments[].workflow: package.module.MyWorkflow` and `workflow_kwargs` | [Credit assignment](../tutorials/credit-assignment.md#reference-settings) |
| A batch transform | a `BatchTransform`, run at the trainer's microbatch boundary | not from a config: `PlatoonTinkerRLTrainer(..., batch_transforms=[...])` in a training script of your own | [`platoon/train/tinker/batch_transforms.py`](../../platoon/train/tinker/batch_transforms.py) |
| Config sections of your own | a subclass of the training config | `trainer_config: module:Class` | [Write and override configs](../how-to/configs.md#5-keep-keys-strict) |
| A rollout function or task loaders | `fn(task, rollout_config)`, `fn(config, split, **kwargs)`, `fn(task_id) -> pt.Task` | `rollout:`, `eval_rollout:`, `dataset_loader:`, `task_loader:` | [Config reference](../reference/config.md#environmentconfig); an entry with a `service` and a `harness` needs none of them |
| A task sampler (experimental, since 0.4.0) | a class with `next(ctx)`, `state_dict()` and `load_state_dict(state)`, registered with `@task_sampler(name)` from `platoon.experimental` | `train.task_sampler:` or an entry's `task_sampler:`, a registered name or `module:Class`, with `task_sampler_options:` | [Write a task sampler](task-samplers.md) |
| A group admission rule (experimental, since 0.4.0) | a class with `admit(group, stats)`, registered with `@group_admission(name)` from `platoon.experimental` | `train.group_admission:` or an entry's `group_admission:`, a registered name or `module:Class`, with `group_admission_options:` | [Write a group admission rule](group-admission.md) |
| Rollouts on the lease (experimental, since 0.4.0) | a workflow that runs each rollout in `async with rollout_lease().rollout(i):` from `platoon.experimental`, and says so with `runs_rollouts_on_lease = True` | `train.permit_release: rollout`, the default, beside `environments[].workflow` | [Run rollouts on the lease](rollout-leases.md) |

Some parts are not extension points: the gateway's wire dialects, how budgets are enforced, the training backend
(any service that implements the Tinker API) and the command line. Changing them, or opening a new extension
point such as a metrics sink, starts with an [RFC](../rfcs/README.md#how-to-propose-a-change).

## How platoon finds your code

A config names your code in one of three ways:

| Way | Looks like | Platoon imports the module | Use it when |
|---|---|---|---|
| Import string | `program: my_pkg.programs:best_of_two`, `service: python://my_pkg.env:MyEnv` | when it resolves the reference, in each process that needs it | trying something out; anything without a short name |
| `package:` | `package: my_pkg.components` on an environment entry, with `harness: my-agent` | in the trainer, in evaluation and in every rollout worker, before anything resolves | your decorators (`@pt.harness`, `@pt.program`), renderer registrations and hooks must run first, so that short names resolve |
| Entry point | `[project.entry-points."platoon.harnesses"]` in your package's `pyproject.toml` | when a config names the component | you ship a package: its components are found by name wherever it is installed |

An import string names a module and an object in it, `module:attr`, and the object may be nested:
`my_pkg.envs:Mazes.small` is the attribute `small` of the class `Mazes`.

Entry points are how an installed package provides components by name. There is one group per kind of component:
`platoon.harnesses`, and since 0.2.0 `platoon.programs`, `platoon.environments` and
`platoon.reward_processors`. Each entry point is named exactly as configs name its component. Platoon reads the
groups from the installed metadata, and imports an entry point only when a config (or code, through
`ctx.agents.spawn(harness=...)` or `ep.run(harness=...)`) names its component. [Publish an extension
package](publish-a-package.md) declares all four.

The module must be importable in every process that uses it: install your package (`uv pip install -e .` while
you work on it), or put its directory on `PYTHONPATH`. Rollouts can run in worker processes
(`rollout_isolation: process`, `inference.workflow.use_subprocesses`), and those import components by their
reference, so a component must be a module-level object: a closure or an object built at run time cannot cross
the process boundary.

## Names and versions

*Since 0.2.0.* Every component a config selects by name (a harness, a program, a named environment, a
reward processor) has a name of the form `<owner>-<name>`: lower-case words of letters and digits joined by `-`,
`_` or `.`, whose first word is its owner. A distribution keeps to one owner, and `platoon extensions check` warns
when it does not. Platoon's own components take the `platoon` owner (`platoon-codeact`, `platoon-number-search`),
plus a few names from before the rule and the product names of the third-party agents it ships specs for (`single`,
`gym`, `mini-swe-agent`, `openhands`); no extension can register them. The extension guides use the owners
`example`, `acme` and `my`, and the tutorials' examples the owner `example` (`example-guesser`, `example-tool-loop`,
`example-delegator`); a package of yours takes an owner of its own.

A component has a version, `1` unless it says otherwise (`@pt.harness("acme-agent", version="2")`), and a config may
pin one with `name@version`. Several versions of one name may be installed at once:

- `name@version` picks one, and a version prefix matches at dots: `mini-swe-agent@2` matches `2.4.6`.
- A bare `name` resolves only while exactly one version is registered; otherwise it is an error that lists them.
- A reference with `:` is an import string and skips the lookup.

A name resolves in this order: platoon's own components; the installed entry point of the component's group with
exactly that name; then the components that the environment entry's `package:` modules registered. An unknown name
is a configuration error (exit code 78) that lists what is registered, what is installed and the nearest names.

## Named environments

*Since 0.2.0.* A named environment says once how to run an environment: its service and options, the
options each harness needs on it, its program, its reward processor and its default splits. A package registers one
in the `platoon.environments` group:

```python
# acme_mazes/entries.py
import platoon as pt

MAZES = pt.EnvironmentEntry(
    "acme-mazes",
    version="1",
    description="Grid mazes: walk to the exit with move(direction).",
    service="python://acme_mazes.env:MazeEnvironment",
    harness_defaults={"platoon-react": {"system_prompt": "Find the exit."}},
    eval_dataset_kwargs={"split_name": "val"},
)
```

```toml
[project.entry-points."platoon.environments"]
acme-mazes = "acme_mazes.entries:MAZES"
```

A config selects it with `env:`, and its own keys merge over the named entry:

```yaml
environments:
  - env: acme-mazes@1
    harness: platoon-react
    dataset_kwargs: {limit: 500}
```

The named entry is the lowest layer: mappings merge key by key, and anything else, lists included, replaces it.
`harness_defaults` gives each harness the options this environment needs from it, so the entry's `harness_options`
start from the defaults of the harness it picks. A harness mixture (`harness: {platoon-react: 1, my-agent: 1}`)
shares one `harness_options` mapping, so when the named environment gives defaults for a harness of the mixture,
the entry sets `harness_options` itself and no defaults apply. The loaded config pins the version the name
resolved to, so `config.resolved.yaml` records it. A Harbor dataset that a package ships registers the same way,
with `service="harbor"` and its `dataset_dir`; site facts such as the sandbox provider stay in each site's config.

An entry that selects a named environment leaves tasks out with its own `exclude_tasks`
([Leave tasks out](../how-to/train-and-eval.md#leave-tasks-out)). A named environment cannot set that key, as it
cannot set `name`, `weight`, `harness_options`, `max_staleness` or the run-level settings, which each run chooses:
the entry's own list would replace the environment's instead of adding to it. An environment that should not offer a
task leaves it out of its own task lists, in a new version.

Of an entry's training settings ([Settings per environment](../how-to/train-and-eval.md#settings-per-environment)),
a named environment may ship the rollout settings its tasks were measured with, `workflow_config.rollout_config`,
such as a calibrated step budget (`workflow_config: {rollout_config: {max_steps: 30}}`): they are the lowest layer of
the entry's settings, beneath the run's evaluation settings in its evaluations. The rest of `workflow_config` (the
group size, the baseline, the minimum group size and the other training strategy) and `eval_overrides.workflow_config`
are the run's to choose, and building the entry refuses them. An environment that gives partial credit may ship its
`success_threshold`, the scored reward that counts as a pass in the run's per-task statistics
([Per-task statistics](../how-to/train-and-eval.md#per-task-statistics)).

`platoon-number-search@1` ships with platoon and needs nothing installed; `textcraft@1` and `textcraft-synth@1` come
from the first-party extension `platoon-textcraft` (`extensions/textcraft`).

## See what is installed

*Since 0.2.0.* Listing reads the installed metadata and imports nothing of an extension:

| Command | Shows |
|---|---|
| `platoon extensions list [--kind harness\|program\|environment\|reward_processor] [--json-output]` | every component the installation provides, core's first, with its distribution, version and whether this platoon can load it |
| `platoon extensions show NAME [--kind ...] [--load] [--json-output]` | one component, as the list shows it, with its entry point, its requirement on platoon and where configs use it; an extension's version shows once `--load` imports it. Exits 78 when nothing installed has the name, or when `--load` fails |
| `platoon extensions check [DIST ...] [--load]` | problems: names that break the rule, more than one owner, a missing or unbounded requirement on platoon, entries in retired groups; `--load` also imports every component. Exits 78 on an error |
| `platoon harness list`, `platoon program list`, `platoon env list` | the harnesses, programs and named environments, with the distribution of each |
| `platoon harness show NAME`, `platoon program show NAME`, `platoon env show NAME` | one component: its source and its documentation; a program's and a named environment's distribution; a named environment's entry and harness options |
| `platoon config validate CONFIG` | resolves every component a config names, as the start of a run does (`--no-resolve` checks the files and named environments only) |

## Compatibility and provenance

*Since 0.2.0.* An extension states the versions of platoon it works with in its requirements
(`platoon >= 0.3, < 0.4`: below 1.0, a minor release may break extensions). Before platoon imports any of its
components, it checks that range against the running platoon from the installed metadata, and refuses an extension
whose range excludes it, with an error that names both versions. Set `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` to load
it anyway, for example while you develop against a checkout of platoon.

`train`, `eval` and `rollout` resolve every component a config names before the run starts, and `lock.json` in the
run directory records each one: the reference as the config gives it, what it resolved to (`name@version`, or a
`module:attr` source), the distribution and version that provided it, and the entry point that named it, with the
extensions' versions and how they were installed. It also records the policy's tool-call parser
(`tool_call_parsing`), a parser of your own included ([Models and tool
calling](../how-to/models-and-tool-calling.md#set-the-parser)).

## Rules every extension follows

- **Fail closed.** A failure that says nothing about the policy (a crashed grader, a dead connection, a missing
  answer key) is an `error` or `invalid` outcome (`pt.Outcome.infra(...)`), never a reward of 0. Platoon keeps
  such rollouts out of training and out of the group baseline.
- **Never block the event loop.** Function harnesses, programs, finalizers, judges and environment sessions run in
  the rollout's event loop, next to the model gateway. Await what is slow, and run blocking work in a thread
  (`await asyncio.to_thread(fn, ...)`).
- **Respect the budget.** A step is one `policy` model call. Check `ctx.budget.remaining()` before each one and end
  with `pt.AgentResult("limit", ...)`: platoon does not stop a harness that ignores its budget.
- **Sample through the gateway.** Training uses the exact tokens of every call the gateway recorded. Calls to a
  model that bypass `ctx.chat` and `ctx.endpoint` are not recorded and do not train.
- **Keep imports cheap and side effects deliberate.** Platoon imports your module in many processes. Keep heavy
  imports out of your package's `__init__.py`, and build services and providers cheaply: every rollout builds its
  own.
- **Keep secrets out of configs.** Read them from environment variables; configs and run records are copied into
  every run directory.
- **Test on CPU.** `platoon.testing` runs your component through the path training takes, with a scripted policy
  in place of the model and no download ([Test an environment, a harness or a program on
  CPU](../how-to/test-extensions.md)).

## Ship it as a package

[Publish an extension package](publish-a-package.md) walks through
[`examples/extension-package/`](../../examples/extension-package/): a harness, an episode program, a reward
processor and a task table in one installable distribution, with entry points, a config that names them, and CPU
tests.

## Next steps

- [Publish an extension package](publish-a-package.md): package, test and ship your components.
- [Architecture](../architecture.md): how environments, harnesses, the gateway, programs and the trainer fit
  together.
- [CONTRIBUTING.md](../../CONTRIBUTING.md): adding a built-in harness, environment service or renderer to platoon
  itself.
