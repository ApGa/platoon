# RFC 0016: Component configuration

| | |
|---|---|
| Status | Accepted |
| Created | 2026-10-16; revised 2026-10-17 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0013](0013-extension-model.md) (the extension model: this RFC is its milestone M5 and the callbacks of its section 12), the config loader (`platoon/utils/config.py`), environment entries (`platoon/train/components.py`), named environments (`platoon/envs/named.py`), the TOML harness spec format (`platoon/harness/spec.py`), the hooks (`platoon/hooks.py`), [RFC 0008](0008-config-and-cli.md) (config and CLI) |
| Discussion | The pull request that added this RFC, and the pull request of M1, which revises it with the decisions of 2026-10-17 (open questions 1 to 4, 6 and 7) |
| Implementation | M1 (strict loading), in the pull request that revises this RFC; a `trainer_config` that does not resolve is already a configuration error (section 8), from RFC 0013's follow-ups |

## Summary

Every component a config selects by name (a harness, an episode program, a named environment, a reward processor)
declares its options as a typed schema, and a config sets them next to the key that selects the component:
`harness_options`, and the new `program_options`, `env_options` and `reward_processor_options`, with
`eval_overrides` for what differs in evaluation rollouts. A harness written as a TOML spec declares its options in
an `[options]` table and renders them into its launch, so one harness covers an agent's settings: one `openhands`
harness, configured by options, replaces three OpenHands specs. Platoon checks options when the config loads, in
`platoon config validate` and before `train`, `eval` or `rollout` runs anything, so an unknown or badly typed option
is a configuration error (exit 78) that names its key, not a failure in every rollout. A component can also check
how its options fit the rest of its entry. A config states the extensions it needs, and their versions, with
`requires:`, checked from installed metadata before anything is imported. The four process-wide hooks become
callbacks that components own: trace filters declared on the harness whose trajectories they read, and counters,
gauges and diagnostics registered through a new `platoon.telemetry` module. `trainer_config` subclasses, whose only
job is checking options, are deprecated, and the settings core keeps for a single extension (rollout fields, ORS
presentation options, a viewer mode) move to that extension. Everything this replaces keeps working, with a
deprecation warning, for one minor release, except the OpenHands specs, which no release has shipped (section 1).

## Motivation

- **Options are free-form until a rollout reads them.** `harness_options` is a `dict[str, Any]` on the environment
  entry (`EnvironmentConfig` in `platoon/train/components.py`), so the loader accepts any key. The built-in
  harnesses parse their options inside each rollout (`CodeActOptions.from_mapping` in
  `platoon/harness/builtin/codeact/harness.py`, `ReactOptions` in `platoon/harness/builtin/react.py`), so a
  misspelled option passes `platoon config validate` and then ends every rollout as an error while the trainer keeps
  stepping. The example extension's harness does the same and says so
  (`examples/extension-package/src/example_extension/harness.py`). A TOML harness spec ignores `harness_options`
  silently.
- **A TOML spec cannot be configured.** Without options, each setting of an agent is a spec of its own
  (`platoon/harness/builtin/specs/`). `platoon-openhands-eval` is `platoon-openhands` with three launch flags
  changed (no reasoning replay, a 300-second request timeout, five retries) and another default step limit, so that
  an evaluation can match Harbor's `openhands-sdk` adapter, and `openhands` runs the OpenHands CLI, a third way to
  run an OpenHands SDK agent, from a toolchain of its own. Users choose between names where they mean to choose
  settings, and each variant is one more spec to keep in step with the others.
- **Programs and reward processors have nowhere to put options.** A program reads its settings from
  `rollout_config.extra`, or from a rollout-config field that core itself never reads:
  `skip_subagent_reward_computation` (`platoon/config_defs.py`) exists for one program, as `step_timeout` exists for
  one extension's harness, which also takes it as an option of its own. A reward processor's knobs are constants in
  its code (`DELEGATION_REWARD_CAP` in `plugins/textcraft/platoon/textcraft/rewards.py`; the partial-credit weight in
  `examples/extension-package/src/example_extension/rewards.py`).
- **One mapping serves two components.** A research extension that trains an OpenHands harness keeps its episode
  program's settings in `harness_options`, next to the harness's own, and splits the mapping by field name. Settings
  that differ between training and evaluation live in the training rollout config's `extra`.
- **`trainer_config` is a workaround for checking options.** To check options when a config loads, that extension
  subclasses the training config and the evaluation config, and every environment entry of every config names the
  class (`trainer_config:`; `narrow_schema` in `platoon/utils/config.py`). A reference that does not resolve is only
  logged, and validation falls back to the base schema, so the check silently does not run. The extension also ships
  a pre-flight command that runs the same checks again before a job starts.
- **Named environments cannot be parametrized.** `textcraft-synth@1` serves two experiment arms, and a config picks
  one by reaching into one harness's internals, `harness_options.prompt_builder_options.variant`
  (`plugins/textcraft/platoon/textcraft/entries.py`), which no other harness reads.
- **A harness mixture shares one mapping.** Every harness of a mixture gets the same `harness_options`, so a named
  environment's per-harness defaults cannot apply there ([RFC 0013](0013-extension-model.md), open question 2), and
  a child that `ctx.agents.spawn(harness=...)` starts with another harness gets no options at all (`spawn` in
  `platoon/harness/agents.py`).
- **A config cannot say what it needs.** A recipe that names `env: textcraft-synth@1` on a machine without the
  extension fails with an unknown-name error that lists the installed entry points, not the distribution and the
  version range to install.
- **Hooks are process-wide.** `platoon.hooks` holds four seams that modules register into when they are imported.
  The two trajectory filters run on every trajectory, whatever harness produced it
  (`platoon/utils/trajectory_error_filtering.py`), so in a mixture an OpenHands error detector also reads CodeAct's
  trajectories; they exist only in processes that imported the registering module; and
  `filter_message_actions: true` with no message filter registered does nothing, silently. Of the telemetry
  counters, core forwards from rollout workers only the programmatic tool calling keys it knows by name
  (`CUMULATIVE_KEYS` in `platoon/train/tinker/workers/telemetry.py`, `cell_guard_cumulative` in
  `platoon/train/tinker/workers/run.py`). Tests reset `platoon.hooks._HOOKS` by hand.
- **Core keeps one extension's presentation.** The ORS service can render tool results as a stdio MCP bridge
  rendered them, with that bridge's extra tools and event log (`result_format`, `bridge_tools` and `log_dir` in
  `platoon/envs/ors/service.py` and `platoon/envs/ors/session.py`), and the trajectory viewer has an OpenHands mode
  (`platoon/visualization/tui.py`). Both exist for policies trained behind that bridge with that extension's
  OpenHands harness.

## Proposal

### 1. Components declare their options

```python
from typing import Literal

import pydantic

import platoon as pt


class BestOfNOptions(pydantic.BaseModel, extra="forbid"):
    n: int = pydantic.Field(4, ge=1)  # root attempts
    keep: Literal["best", "first_success"] = "best"


@pt.program("acme-best-of-n", options=BestOfNOptions)
async def best_of_n(ep: pt.EpisodeContext[BestOfNOptions]) -> pt.Outcome:
    for _ in range(ep.options.n):
        ...
```

| Kind | Declared with | Set in an environments entry with | The component reads them as |
|---|---|---|---|
| Function harness | `@pt.harness(name, options=...)` | `harness_options` | `ctx.options` |
| Episode program | `@pt.program(name, options=...)` | `program_options` (new) | `ep.options` |
| Named environment | `pt.EnvironmentEntry(name, options=..., configure=...)` | `env_options` (new) | `configure(options)`, when the config loads (section 4) |
| Reward processor | `@pt.reward_processor(name, options=...)` (new decorator) | `reward_processor_options` (new) | `fn(trajectory, options)` |
| TOML harness spec | an `[options]` table in the spec (below) | `harness_options` | `{options.NAME}` and `{options}` in its launch templates, rendered at run time |

- **What `options=` takes.** A pydantic model or a dataclass, checked by the code that checks the config itself
  (`validate_config` and `build_config` in `platoon/utils/config.py`): every problem at once, each with its full
  dotted path, and an unknown key with a did-you-mean. Or a callable `(Mapping[str, Any]) -> object` that raises
  `ValueError`, for a parser of the component's own; `CodeActOptions.from_mapping` already has that shape. pydantic
  is already a core dependency.
- **Without a schema.** A function harness that declares no `options=` still gets its `harness_options` unchecked,
  as today, and the loader warns that nothing checks them; when the deprecations end (section 10), such a harness
  takes no options. A program, a named environment or a reward processor without a schema takes none, since its
  option key is new, and a key set for it is an error; a TOML spec without options is treated the same way (below),
  since it never read `harness_options`. The defaults are the component's own, so a config sets only what it
  changes.
- **Typed contexts.** `pt.AgentContext` and `pt.EpisodeContext` become generic in their options type, so
  `ctx.options.cell_timeout_s` type-checks. A harness without a schema keeps today's read-only mapping.
- **The built-ins declare theirs.** `platoon-codeact` declares `CodeActOptions` and `platoon-react` `ReactOptions`,
  which they parse in every rollout today; `platoon-oracle` and `platoon-noop`, which read none, declare an empty
  schema; the `openhands` spec declares the options below, and the specs `platoon-bash` and `mini-swe-agent`
  declare none. The program `gym` takes none, and `single` takes none until RFC 0013's stage 2 folds the TextCraft
  RAO program into it, with `zero_subagent_rewards` (section 8).
- **Reward processors gain a decorator.** `@pt.reward_processor(name, version=, options=)` registers
  `fn(trajectory, options)`, versioned as programs are. A plain `fn(trajectory)` keeps working, without options, by
  import string or entry point.
- **Shown and recorded.** `platoon harness show`, `program show` and `env show` print a component's option schema
  (name, type, default, allowed values or range, description), `platoon config show --defaults` fills in each
  component's defaults, and `lock.json` records the options each component ran with, defaults included, beside the
  component ([RFC 0013, section 7](0013-extension-model.md#7-provenance)): per entry, and per phase where
  `eval_overrides` changes them (section 2).

**TOML harness specs.** A spec declares its options under `[options]`, one table per option, and its launch
templates render them. This is the `openhands` spec that M2 adds in place of the three OpenHands specs of the
motivation:

```toml
# platoon/harness/builtin/specs/openhands.toml
name = "openhands"
version = "1.54.0"  # the OpenHands SDK's
dialect = "openai_chat"
timeout_s = 3600
max_steps = 30  # used when the run sets no step budget; the run's max_steps wins

[install]
toolchain = "openhands-1.54.0"  # one toolchain, built once per site, serves every setting

[install.recipe]
python = "3.12.14"
lock = "openhands-1.54.0.txt"  # openhands-sdk and openhands-tools 1.54.0 and their closure
cache = ["tiktoken"]

[options.reasoning_replay]
type = "bool"
default = true
description = "Send each assistant turn back with its reasoning, so that each call extends the last"

[options.request_timeout_s]
type = "int"
default = 3600
min = 1
description = "Seconds one model call may take"

[options.num_retries]
type = "int"
default = 0
min = 0
max = 10
description = "Times the SDK retries a failed model call; a retry may record a call twice"

[options.condenser]
type = "str"
default = "none"
allowed = ["none", "llm_summarizing"]
description = "The SDK's context condenser; its summaries are the agent's model calls, recorded and trained on"

[launch]
argv = [
    "{toolchain}/bin/python3", "-I", "{state_dir}/run_openhands.py",
    "--instruction-file", "{state_dir}/instruction.md",
    "--max-iterations", "{max_steps}",
    "--final-message-file", "{state_dir}/final_message.txt",
    "--reasoning-replay", "{options.reasoning_replay}",
    "--llm-timeout", "{options.request_timeout_s}",
    "--num-retries", "{options.num_retries}",
    "--condenser", "{options.condenser}",
]
files = { "instruction.md" = "{instruction}" }
assets = { "run_openhands.py" = "../openhands/run_openhands.py" }
# [launch.env], [resources], [builtins], [final_message] and [exit_codes] as platoon-openhands has them
```

- **The table.** `[options.NAME]` declares one option, and `NAME` is a lower-case identifier, as a placeholder's
  name is. `type` is `bool`, `int`, `float` or `str`. `default` is required: TOML has no null, so an option always
  has a value, and an optional feature is an allowed value such as `"none"`. `description` is one line. `allowed`
  lists the values an `int` or `str` option may take, and `min` and `max` bound an `int` or a `float`. A default
  that breaks its own rules, an unknown key in the table, or a placeholder that names an option the spec does not
  declare fails when the spec loads (`HarnessSpecError`), as a malformed spec does today.
- **Checked as any schema is.** What a config gives the spec, from `harness_options` (flat, or keyed in a mixture),
  a named environment's `harness_defaults`, `eval_overrides` and command-line overrides, is checked against the
  table when the config loads, with section 6's errors: an unknown key with a did-you-mean, a value of the wrong
  type (a boolean is not an integer, while an integer is a valid `float`), a value outside `allowed` or the range.
  `harness show` prints the table, and `lock.json` records the options with their defaults.
- **Handed to the launch.** At run time the rollout parses the route's options with the same table, as a rollout
  worker parses a function harness's (section 6), and renders them where the run's own values render: in
  `[launch]` (argv, env and files), `[mcp]` and `[builtins.disable]`. `{options.NAME}` is one option's value (a
  boolean as `true` or `false`, a number in decimal, a string as it is), and `{options}` is all of them, defaults
  included, as one JSON object, for a program that reads them whole (`files = {"options.json" = "{options}"}`).
  Options never render into `[install]`: they change how a harness runs, not what is installed, so one toolchain
  serves every setting. They are recorded, and argv shows in a process listing, so options carry no secrets; a key
  stays in a variable, as `{api_key}` does.
- **Without a table.** A spec without `[options]` takes none. It never read `harness_options`, so from M2 a key set
  for it is a configuration error at once, not a deprecation: the config was configuring nothing.
- **No `check=`.** A spec is data. A rule that relates its options to the rest of the entry belongs to the
  program's `check=` (section 3).
- **Versions.** A new option whose default keeps the launch as it was keeps the spec's version; a changed default
  bumps it, as a change to the launch does today.

**One harness per agent.** With options, one spec covers an agent's settings, and a config chooses them, with
command-line overrides on top, instead of choosing between specs. M2 applies this to OpenHands first. `openhands`
runs the OpenHands SDK agent with platoon's runner, from one toolchain, and replaces `platoon-openhands` (its
defaults), `platoon-openhands-eval` (the options of Harbor's adapter, section 2) and the OpenHands CLI spec
(`openhands@1.16.0`, an older SDK's agent behind the CLI, with the CLI's tools). A harness that wraps a third-party
agent is named after the agent when platoon adds no behaviour of its own: the runner starts the SDK's agent with the
tools of Harbor's adapter and applies what the options choose, so the harness is `openhands`, not
`platoon-openhands`. No release has shipped the three names, so they leave without aliases; should a release ship
them before M2, they stay one minor release with a warning, as the deprecations of section 10 do.

### 2. Where a config sets them

```yaml
environments:
  - env: textcraft-synth@1
    env_options: {variant: recursive}
    harness: platoon-codeact
    harness_options: {cell_timeout_s: 7200}
    program: acme-best-of-n
    program_options: {n: 4}
    reward_processor: acme-length-penalty
    reward_processor_options: {per_token: 1.0e-4}
    eval_overrides:                      # evaluation rollouts only
      reward_processor_options: {per_token: 0.0}
```

- **Next to the selector.** Each `*_options` key belongs to the component its sibling key selects. Options for a
  component that takes none are an error (`program_options` with the program `gym`, say), so switching
  to another component never silently drops what the config set for the old one.
- **Harness mixtures.** With `harness: {platoon-codeact: 1, platoon-react: 1}`, `harness_options` may be keyed by
  harness name, each key holding one harness's options, and a named environment's `harness_defaults` then apply per
  harness too. An entry that sets no `harness_options` gets each harness's defaults, as the keyed form with nothing
  to add would, where RFC 0013 refuses such a mixture today. A flat mapping keeps working, shared as today, and must
  suit every harness of the mixture; it replaces the named environment's defaults, as RFC 0013 has it. Keys that all
  name harnesses of the mixture make the keyed form, keys that name none make the flat form, and a mix of the two is
  an error. This answers open question 2 of RFC 0013.
- **Children.** `ctx.agents.spawn(harness=H)` without `options` starts the child with the entry's options for `H` (the
  keyed form, else the named environment's defaults for `H`, else `H`'s own defaults) instead of none; a child of
  its parent's own harness inherits its parent's options, as today. Options that code passes (`ep.run(options=...)`,
  `spawn(options=...)`) are checked against the target harness's schema when the call is made. A bad one raises in
  the calling program or harness, which ends the episode as an `error` outcome, never a score.
- **Evaluation.** `eval_overrides` holds option keys (`harness_options`, `program_options`,
  `reward_processor_options`) that apply to evaluation rollouts only (every split but `train`, as for
  `eval_dataset_kwargs`): those of `platoon eval`, and those `platoon train` runs when it evaluates. They merge over
  the entry's own, key by key, and are checked as the entry's own are, for the evaluation phase (section 6). A
  training-only penalty then stops living in one phase's `rollout_config.extra`, and a harness needs no second spec
  for evaluation (decided 2026-10-17, open question 2). `env_options` cannot be overridden for evaluation
  (section 4).
- **Named environments.** A named environment may set `program_options` and `reward_processor_options`, also under
  `eval_overrides`, with the program and the reward processor it sets. Like `harness_defaults`, they apply only while
  the entry keeps that component. It cannot set `env_options`, which are the config's to choose.
- **On the command line.** An override sets an option as it sets any key, and the option is checked like the
  config's own: `platoon eval eval.yaml environments.0.harness_options.num_retries=0`.

**Evaluating as Harbor's adapter does.** `openhands` trains with its defaults: reasoning replay, a one-hour request
timeout and no retries. Harbor's `openhands-sdk` adapter runs the same agent without replay, with the SDK's
300-second timeout and five retries, and with the SDK's step limit of 500. An evaluation that should score as the
adapter does is a config, not another harness. In an evaluation config (`platoon eval`), the entry sets the options:

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs: {dataset_dir: ./tasks, toolchain_root: /shared/toolchains}
    harness: openhands
    harness_options: {reasoning_replay: false, request_timeout_s: 300, num_retries: 5}

inference:
  workflow:
    rollout_config:
      max_steps: 500                         # the adapter keeps the SDK's step limit
      inference_params: {temperature: 0.6}   # and sets only the temperature
```

In a training config, the same options go under `eval_overrides`, so the evaluations during training score as the
adapter does while training keeps the defaults. The step limit and the sampling are the phase's rollout settings,
not options:

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs: {dataset_dir: ./tasks, toolchain_root: /shared/toolchains}
    harness: openhands
    eval_overrides:
      harness_options: {reasoning_replay: false, request_timeout_s: 300, num_retries: 5}

eval:
  workflow_config:
    rollout_config: {max_steps: 500, inference_params: {temperature: 0.6}}
```

### 3. Checks across an entry

Some rules relate a component's options to the rest of its entry: a judge that needs verifier grading, a harness
that needs the service to render results in its format, a delegation reward that needs `propagate_root_success` off.
A component states them in `check=`:

```python
def check(options: EpisodeOptions, ctx: pt.CheckContext) -> None:
    if options.behavior_judging and ctx.environment.agents.child_grading != "verifier":
        raise ValueError("behavior_judging needs agents.child_grading: verifier")


@pt.program("acme-episode", options=EpisodeOptions, check=check)
async def episode(ep: pt.EpisodeContext[EpisodeOptions]) -> pt.Outcome: ...
```

- `check(options, ctx)` raises `ValueError` with what is wrong. `pt.CheckContext` holds the merged entry
  (`environment`), the phase (`train` or `eval`) and that phase's rollout config (`rollout`), all read-only: a check
  sees its entry and the phase's rollout settings, and nothing else of the config (decided 2026-10-17, open
  question 7).
- Platoon runs it when the config loads, for each entry that runs the component and each phase the command runs
  (`train` runs both when it evaluates), and reports a failure as a configuration error that names the entry and the
  phase.
- A check reads the config only. Whether a server answers, or a variable is set where the run executes, is for the
  component to find out when the run starts. The research extension's check that an env-server pool is set, for
  example, belongs in the ORS service, which would then refuse to fall back to `localhost`.

### 4. Options of named environments

```python
class TextCraftOptions(pydantic.BaseModel, extra="forbid"):
    variant: Literal["linear", "recursive"] = "linear"


def configure(options: TextCraftOptions) -> dict[str, Any]:
    return {"harness_defaults": {"platoon-codeact": {"prompt_builder_options": {"variant": options.variant}}}}


TEXTCRAFT_SYNTH = pt.EnvironmentEntry(
    "textcraft-synth",
    version="1",
    service="python://platoon.textcraft.env:TextCraftSynthEnvironment",
    options=TextCraftOptions,
    configure=configure,
)
```

- `configure(options)` returns entry keys, `harness_defaults` among them, laid over the entry's static keys. The
  result passes the checks a named environment's keys pass today (`FORBIDDEN_KEYS` and the schema, in
  `platoon/envs/named.py`), and the config's own keys stay on top.
- The loaded config keeps `env_options` as given, beside `env: name@version`, so `config.resolved.yaml` expands to the
  same entry again. An option whose default keeps today's behavior leaves the version as it is; changing what an
  option does bumps it.
- `env_options` without `env:` is an error. A service's `service_kwargs` are the service's own keyword arguments.
  When the config loads (section 6, step 7), platoon binds them to the service factory's signature without building
  it, so an unknown key fails with its path and a did-you-mean (`environments.0.service_kwargs.datset_dir: unknown
  key; did you mean environments.0.service_kwargs.dataset_dir?`), exit 78. The service checks their values when the
  run builds it to list the tasks, before any rollout, and an error it raises there is a configuration error that
  names the entry, not a traceback. `dataset_kwargs` and `eval_dataset_kwargs` of an entry whose service lists its
  tasks are checked against the keys a service listing takes (`split_name`, `shuffle_seed`, `limit`). `platoon
  config validate` takes `key=value` overrides, as `train` and `eval` do, and checks a file that holds only
  `environments` against the entry schema (amended with RFC 0018's milestone M4).
- **Options rather than a version per variant** (decided 2026-10-17, open question 6). One version serves its
  variants, so a config picks `textcraft-synth@1`'s recursive arm with `env_options: {variant: recursive}` instead
  of reaching into one harness's prompt-builder options.
- **The same in both phases.** `eval_overrides` cannot set `env_options`. They choose the environment, its tasks and
  its grading included, so an entry evaluates the environment it trains on, and `configure` runs once per entry, not
  once per phase.

### 5. `requires:`

```yaml
requires:
  platoon: ">=0.2,<0.3"
  platoon-textcraft: ">=1,<2"
```

- **The form.** A top-level mapping from distribution name to version specifier (PEP 440; `""` accepts any version).
  A mapping rather than a list of requirement strings, so that `extends:` merges it key by key: a leaf config adds a
  requirement, tightens one of its base's, or drops one with `null`, where a list would replace its base's list
  (decided 2026-10-17, open question 1).
- **Checked first.** Once the files and their `extends:` chains merge, with any override of `requires` itself
  applied as plain data, and before anything is imported, from installed metadata only. That is before the other
  overrides apply: a `trainer_config` class is imported first so that they are parsed against it (section 6), and
  named environments and `package:` modules come after it. The running platoon's version is read as RFC 0013's
  compatibility check reads it (`platoon_version` in `platoon/extensions.py`).
- **The error.** One line for each unmet requirement, exit 78, with the command that fixes it:
  `requires platoon-textcraft>=1,<2, which is not installed: pip install 'platoon-textcraft>=1,<2'`, or
  `requires platoon-textcraft>=1,<2, but 0.9.0 is installed`.
- **Recorded.** `lock.json` gains `requires`: each requirement and the version that met it.
- It adds to RFC 0013's check of each extension's own requirement on platoon; both run.

### 6. Validation when a config loads

```text
load(files, overrides)
  1. files and their extends: chains merged                                (today)
  2. requires: checked from installed metadata, with overrides of it;
     nothing imported                                                      (new)
  3. the trainer_config class imported, and the overrides applied,
     parsed against it                                                     (today; deprecated with the class)
  4. named environments expanded; env_options checked, configure applied   (expansion today; options new)
  5. the config checked against its schema                                 (today)
  6. every component the entries name resolved                             (at run start; in validate from M1)
  7. each component's options checked, per entry and phase                 (new)
  8. each component's check= run, per entry and phase                      (new)
  9. deprecations reported (section 10)                                    (new)
```

- **Which commands.** `train`, `eval` and `rollout` run every step before anything runs (`run_components` in
  `platoon/extensions.py` is step 6 today). `platoon config validate` runs every step too, by default from M1, so
  it needs the config's extensions installed, as its `requires:` says; `--no-resolve` stops after step 5, the old
  default, for a machine without them, and `--resolve` stays accepted as the default spelled out. The default
  changes with no release of notice: no release has shipped the command yet, so no user depends on the old default
  (decided 2026-10-17, open question 3). Steps 1 to 5 already import the `trainer_config` class and the named
  environments, so a machine without their extensions cannot check a config that names them even with
  `--no-resolve`. This answers open question 3 of RFC 0013.
- **Errors.** Every problem found is reported in one configuration error (exit 78), each with its path, for example
  `environments.0.harness_options.cell_timout_s: unknown key; did you mean cell_timeout_s?`. A component that does
  not resolve skips its options and checks, and an entry whose `package` does not import is reported for that
  alone, since the package registers the components it names (from M1).
- **In rollout workers.** Routes carry options as plain data (`EnvironmentRoute` in `platoon/train/components.py`),
  with `eval_overrides` already applied for an evaluation rollout; the worker parses them again with the component's
  schema before it calls the component, so `ctx.options` is typed there too, and a TOML spec's launch renders the
  parsed values (section 1). Options that failed at load never reach a worker.
- **Extensions check their own defaults.** `platoon extensions check --load` also checks each named environment's
  `harness_defaults`, `program_options` and `reward_processor_options` against the components they name, where
  those resolve, so a stale default is found before a user meets it.

### 7. Callbacks owned by components

| Hook today | Replacement | Applies to |
|---|---|---|
| `trajectory.error_completion_ids` | `@pt.harness(..., trace=pt.TraceFilters(errors=fn))` | the trajectories that harness produced |
| `trajectory.message_completion_ids` | `pt.TraceFilters(messages=fn)` | the same |
| `rollout_worker.telemetry` | `platoon.telemetry.register(namespace, counters=fn, gauges=fn)` | the process; metrics named `telemetry/<namespace>/<key>` |
| `rollout_worker.diagnostics` | `platoon.telemetry.register(namespace, diagnostics=fn)` | the process's diagnostic marker |

**Trace filters.**

```python
@pt.harness(
    "acme-openhands",
    options=OpenHandsOptions,
    trace=pt.TraceFilters(errors=failed_calls, messages=message_only_calls, summary=step_summary),
)
async def openhands(ctx: pt.AgentContext[OpenHandsOptions]) -> pt.AgentResult: ...
```

- `errors(steps)` and `messages(steps)` return completion ids, as the hooks do today; `summary(step)` returns the
  line the trajectory viewer shows for a step, or `None`.
- Each trajectory records the harness that produced it, as `name@version` in its agent record (`misc["agent"]`,
  written by `platoon/gateway/trace.py`, which records only the agent's id and role today). Where trajectories become
  training data (`trajectory_to_data` in `platoon/utils/tinker_data_processing.py`, in the trainer or in a rollout
  worker), platoon resolves that harness and applies its filters to that trajectory only, after the generic
  detection. A record from before this change names its harness only at the root, and the root's harness then
  applies to the whole rollout.
- `filter_message_actions: true` when no harness the config runs declares a message filter, and no deprecated hook
  registers one once the components have loaded, warns instead of doing nothing, as a deprecation does (section 10):
  `config validate --strict` fails on it, and it is a configuration error when the deprecations end.
- `platoon viz` shows each step with its harness's `summary` when that harness is installed, and the generic view
  otherwise. `--mode` is deprecated, the viewer's OpenHands code (`platoon/visualization/tui.py`) leaves core with it
  (section 9), and `platoon-codeact` declares the CodeAct summary that the viewer builds in today.

**Telemetry.**

```python
# platoon/telemetry.py
def register(
    namespace: str,
    *,
    counters: Callable[[], Mapping[str, float]] | None = None,  # cumulative in this process
    gauges: Callable[[], Mapping[str, float]] | None = None,  # current values
    diagnostics: Callable[[], Mapping[str, str]] | None = None,  # for a rollout worker's diagnostic marker
) -> None: ...
```

- A component's module registers when it is imported. Components load by exact name in each process that runs
  them (RFC 0013), so the trainer and every rollout worker that runs the component have its telemetry, and no other
  process does. Registering the same functions again does nothing; other functions under a registered namespace are
  an error.
- The namespace follows the naming rule and is usually the owner (`acme`). Each value becomes
  `telemetry/<namespace>/<key>` in the step's metrics.
- **Counters** are summed across processes: a rollout worker reports its counters when it exits, and through its
  diagnostic marker when it is killed. **Gauges** are reported live by in-process rollouts, and once by each worker,
  at exit, as `<key>_at_exit`, summed. That is the rule core applies today, by name, to the programmatic tool
  calling counters; it now applies to any registered gauge, and the names leave core.
- A callback that raises contributes nothing and never fails the step or the worker, as today.
- `platoon.testing` gets a fixture that gives a test fresh telemetry and component registrations, so tests stop
  resetting module globals.

**The hooks.** `platoon.hooks` is deprecated: `register` warns and names the replacement. Core keeps calling the
registered hooks while the deprecation lasts, unscoped as today, except that the trajectory hooks skip trajectories
whose harness declares trace filters, so a harness that has moved is not filtered twice.

### 8. Retiring `trainer_config` and the single-extension fields

- **Already.** A `trainer_config` that does not resolve, or that names something other than a class, is a
  configuration error (exit 78), not a warning followed by the base schema; so is an override that changes the class,
  which the files choose before the overrides are parsed against it (RFC 0013, section 12). The configs of the
  research extension, the plugins and the examples load as they did.
- **With options (M2).** `trainer_config` is deprecated: a config that sets it loads as today, with a warning that
  names the replacement. Its uses move: option checks to `options=` and `check=`, and settings of its own to the
  options of the component that reads them. No extension gets a section at the top of the config (RFC 0013,
  alternatives). `trainer_config` leaves the run-level settings (`RUN_LEVEL_FIELDS`) when its deprecation ends.
- **Rollout fields that one extension reads** become options of the component that reads them. The field stays for
  one minor release, warns when a config sets it, and keeps its effect meanwhile: the component reads the field when
  its own option is unset, as the research extension's OpenHands harness reads `step_timeout` today.

| Rollout config field | Becomes |
|---|---|
| `step_timeout` | an option of the research extension's OpenHands harness, which it already is: the harness reads the rollout field only when its option is unset. The TextCraft configs set the field too, where nothing reads it, and drop it |
| `skip_subagent_reward_computation` | `zero_subagent_rewards`, an option of the TextCraft RAO program, and of `single` once RFC 0013's stage 2 folds that program into it |
| `extra` | stays, for rollout functions (`rollout:`); components take options instead |

`propagate_root_success` and `enable_subagent_verifiers` stay where they are: core reads them.

### 9. What leaves core

The research extension that trains its own OpenHands harness moves first (M5), in its own repository: option schemas
for its harness and its program, with the program's settings in `program_options`; a `check=` for the rules its
config classes and pre-flight command check today; trace filters; telemetry; and its configs setting the per-step
timeout as the harness option it already is. Then what core keeps only for it leaves (M6):

| In core today | What it is for | Where it goes |
|---|---|---|
| ORS `result_format: bridge_json`, `bridge_tools`, `log_dir` | tool results, two extra tools and an event log as the stdio MCP bridge produced them | an ORS service of the extension's own, a subclass of `OrsEnvironmentService` through documented override points (rendering a result, offering extra tools, recording an event), which its named environments select |
| ORS `prompt_filter` | rewriting an environment's prompt, such as a stale line in an old image | the same subclass |
| ORS `openreward:v1:` task references | the task ids of the extension's earlier datasets | the same subclass (resolving a task id) |
| `platoon viz --mode openhands` and the OpenHands step summaries | reading OpenHands steps | the harness's `TraceFilters.summary` (section 7) |
| the `ptc_*` keys of the telemetry path (`platoon/train/tinker/rl.py`, `platoon/train/tinker/workers/`) | the programmatic tool calling guard's counters | the extension's `platoon.telemetry` registration |

The ORS service's `legacy_global_pool` is not in the table, because it is not one extension's: the docs teach the
process-wide `OPENREWARD_SESSION_URLS` pool, which that key turns on and which is on by default, for runs with one
remote environment, and `legacy_global_pool: false` for mixtures. Whether the pool should apply only when a config
runs one remote environment, so that the key can go, is open question 8.

Each keeps working, with a warning, for one minor release. Their tests move with them, and the extension's own
fixtures show that its subclass renders what core renders today.

### 10. Deprecations

- **How long.** A deprecated form warns for one minor release, the minimum that
  [RFC 0017](0017-public-surface-and-conformance.md)'s version policy sets before 1.0, and the next minor release
  removes it: a form that first warns in 0.3 is gone in 0.4 (decided 2026-10-17, open question 4). Each form counts
  from the first release that warns about it, so forms that different milestones deprecate leave in different
  releases.
- `pt.PlatoonDeprecationWarning` is a `FutureWarning` subclass, so users see it by default (Python hides a
  `DeprecationWarning` outside `__main__`). Each warning names the replacement and the release that removes the old
  form.
- The loader reports each deprecated config key or field once per command; `config validate` lists them, and
  `lock.json` records them (`deprecations`). `platoon config validate --strict` fails on them (exit 78), for an
  extension's or a recipe's CI; in Python, `warnings.simplefilter("error", pt.PlatoonDeprecationWarning)` does the
  same, and so does pytest's `-W error::platoon.PlatoonDeprecationWarning`. Python's own `-W` and `PYTHONWARNINGS`
  cannot name the class: they read their filters before `site-packages` is on the path, so a category from an
  installed package is ignored as an invalid option (found in M1).
- A config class names the keys that are going away in a class attribute, `DEPRECATED_KEYS` (`platoon/deprecations.py`),
  each with what replaces it and the release that removes it; the key keeps its field meanwhile. The loader finds
  each such key a config sets, wherever it sits, the config reference marks it, and `config show --defaults` leaves
  out a deprecated key that the config does not set, so that a config copied from its output sets no key that is
  going away.
- **What, and from when.** From M1: `propogate_root_success`, the misspelling of `propagate_root_success` that the
  rollout config read silently until M1 (`platoon/config_defs.py`). From M2, with the options that replace them:
  `trainer_config`; `step_timeout` and `skip_subagent_reward_computation`; `harness_options` for a function harness
  that declares no schema. From M4: `platoon.hooks`; and `filter_message_actions` with no message filter, a setting
  that does nothing, which warns as a deprecation does and becomes a configuration error when the deprecation ends.
  From M6: the ORS options of section 9 and `platoon viz --mode`. A TOML spec's `harness_options` are not on the
  list: specs take options (section 1).

### 11. What changes for users

- **What breaks:** configs that load today and fail at load after this RFC, each with a mistake that failed every
  rollout: an unknown option of a built-in harness, or of a third-party harness once it declares a schema, and a
  badly typed one (which failed the rollouts, or was used as something it does not mean). From M2, `harness_options`
  that a TOML spec does not declare fail at load too: the spec ignored them, so the config was configuring nothing.
  `filter_message_actions` with no message filter, which silently does nothing, warns instead, and fails only when
  the deprecations end. A `trainer_config` that does not resolve is already an error, and was logged as a warning
  before. Dashboards that read the `ptc_*` metrics need their new names once the extension that reports them moves
  (open question 5). The new keys (`program_options`, `env_options`, `reward_processor_options`, `eval_overrides`,
  `requires:`) are checked at load from the start. The harnesses `platoon-openhands`, `platoon-openhands-eval` and
  `openhands@1.16.0` (the OpenHands CLI) leave in M2, for `openhands@1.54.0` and its options (section 1).
- **Changed:** `config validate` resolves components by default from M1 (`--no-resolve` keeps today's check);
  `ctx.options` of a harness with a schema is the parsed object; a child spawned with another harness gets the
  entry's options for that harness; a harness mixture with no `harness_options` on a named environment gets each
  harness's defaults instead of an error; filters registered as hooks skip trajectories whose harness declares its
  own; telemetry is named `telemetry/<namespace>/<key>`, so the programmatic tool calling metrics are renamed when
  the extension moves; `openhands` names the OpenHands SDK agent that `platoon-openhands` ran, not the CLI.
- **New:** `options=` and `check=` on `@pt.harness`, `@pt.program`, `pt.EnvironmentEntry` (with `configure=`) and
  the new `@pt.reward_processor`; a TOML spec's `[options]` and its `{options.NAME}` and `{options}` placeholders;
  `program_options`, `env_options`, `reward_processor_options`, `eval_overrides` and keyed `harness_options`;
  `requires:`; `pt.TraceFilters`, `pt.CheckContext` and `pt.PlatoonDeprecationWarning`; `platoon.telemetry`;
  `config validate --no-resolve` and `--strict`.
- **Deprecated for one minor release:** the list of section 10.
- **Unchanged:** configs that set no options and name no OpenHands spec that M2 removes; harnesses without
  schemas, whose options stay unchecked, with a warning; import strings; `package:`; and the record formats,
  apart from the agent record's new harness.

## The hard parts

- **Options that depend on the run.** Some settings mix configured options with run-wide ones: the research
  extension's OpenHands harness completes its options with the rollout's sampling defaults and the policy's
  tokenizer. Schemas check what the config sets; the harness still reads the run's sampling defaults when it runs,
  and `check=` sees the phase's rollout config for the rules that span both.
- **The producing harness of each trajectory.** Scoped filters need it in every record, subagents spawned with
  another harness included. Records from before the change fall back to the root's harness, which is right for every
  single-harness episode and wrong only for mixed agent trees recorded before the change.
- **Keyed and flat mixture options.** Harness names are usually `<owner>-<name>` and option keys are Python
  identifiers, so the two forms rarely look alike; the rule (every key names a harness of the mixture, or none does)
  still makes a mix an error instead of a guess.
- **Hooks and callbacks side by side.** While the deprecations last, both run. The skip rule of section 7 keeps a
  harness that has moved from being filtered twice; telemetry registered both ways would count twice, so an
  extension moves its telemetry in one release.
- **Options rendered as text.** A TOML spec's options reach its process through argv, variables or files, so a
  boolean renders as `true` or `false`, which the program must accept. A third-party CLI whose flag takes no value
  (`--no-replay`) needs a shell wrapper in `argv`, as specs already use one to adapt a CLI (`mini-swe-agent` sets its
  `PATH` in `sh -c`). One placeholder rule keeps the spec format free of a template language (alternatives).
- **Options that change what trains.** `reasoning_replay: false` and a condenser change which of an episode's calls
  extend each other, and so how they merge into training sequences, and retries can record one call twice.
  `openhands`'s defaults are its training settings; the others serve evaluations and runs that accept the cost, and
  `lock.json` records which ran.
- **Code that runs at load.** `configure` and `check` run in `config validate` and in the trainer before anything
  starts, never in a rollout. They must be quick and do no I/O: a slow one delays every command that loads the
  config.

## Alternatives

- **A settings section per extension** (`extensions: {acme: {...}}` at the top of a config). Settings belong to
  components: a section per extension ties a config to a distribution instead of to the components it runs, and
  offers two places for one setting.
- **Keeping `trainer_config` subclasses.** The class must subclass both the training and the evaluation config,
  every entry of every config must name it, it cannot tell which component an option belongs to, and it fails open.
- **Option schemas in metadata** (JSON Schema next to the entry point, checked without importing). Listing would need
  no import, but every component would keep a second schema beside the Python types it parses with, and checks across
  an entry need code anyway.
- **Options nested under the selector** (`harness: {name: platoon-codeact, options: {...}}`). It collides with the
  `{harness: weight}` mixture form and rewrites every config.
- **`requires:` as a list of requirement strings**, as pytest's `required_plugins` and Sphinx's `needs_extensions`
  are (knowledge, H). Familiar, but lists replace under `extends:`, so a leaf that adds one requirement would drop its
  base's.
- **A phase flag instead of `eval_overrides`** (`ep.phase`). Each component would invent its own per-phase settings;
  the overlay keeps one rule for all of them.
- **A spec per setting**, as `platoon-openhands-eval` is today. Every setting multiplies the names users choose
  between, each copy of the launch must be kept in step, and a setting that differs only in evaluation needs a second
  harness instead of an override.
- **Argv fragments chosen by an option's value**
  (`[options.reasoning_replay.args] false = ["--no-reasoning-replay"]`). It would fit a CLI whose flags take no value
  without a shell wrapper, but it adds a second template mechanism to the spec format, and the placeholders cover
  platoon's own runners.
- **Scoping inside the hooks.** Every extension would reimplement the harness check, and processes that never
  imported the registering module would still miss the hooks.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M1 strict loading | `PlatoonDeprecationWarning`; deprecations in `config validate`'s output and in `lock.json`; `config validate --strict`; `propogate_root_success` reported as the first deprecation (section 10); `config validate` resolving every component by default, with `--no-resolve` (section 6). A `trainer_config` that does not resolve is a configuration error already (RFC 0013, section 12) | `rollout_config.propogate_root_success` warns once in `config validate` and in `platoon train`, naming its path and `propagate_root_success`, is listed in `lock.json`, and makes `config validate --strict` exit 78; `config validate` exits 78 on a config whose harness does not resolve, and passes it with `--no-resolve` | |
| M2 options | `options=` and `check=` on the four Python kinds, with the new `@pt.reward_processor`; a TOML spec's `[options]`, checked at load and rendered into its launch (section 1); the new option keys, keyed mixture options, children's options and `eval_overrides`; the built-ins' schemas; options in `lock.json`, in the `show` commands and in the testing kit's `run_episode`; the deprecations of section 10 that options replace; the example extension, the guides and the TextCraft configs updated. Then the OpenHands consolidation, the first consumer: `openhands@1.54.0`, with the options `reasoning_replay`, `request_timeout_s`, `num_retries` and `condenser`, and one toolchain replace `platoon-openhands`, `platoon-openhands-eval` and the OpenHands CLI spec, with the docs and the changelog that name them | a misspelled option of each kind, a TOML spec's included, fails `config validate` and `platoon train` with exit 78 and a did-you-mean before any rollout starts; the example extension's harness no longer parses its own options; a CPU test renders `openhands`'s launch for a training rollout with its defaults and for an evaluation rollout with section 2's `eval_overrides`, both from one toolchain, and `harness show openhands` lists the four options | |
| M3 `requires:` | the key, a mapping `{distribution: specifier}` that `extends:` merges key by key (section 5); its check before any import; `lock.json` | a config that requires a missing or too old distribution fails with the install command; a leaf config adds a requirement to its base's and drops one with `null`; a test that records imports shows that nothing of an extension was imported | |
| M4 callbacks | `pt.TraceFilters`, the producing harness in agent records, `platoon.telemetry`, `platoon.hooks` deprecated, the viewer's summaries, the `filter_message_actions` warning | in a mixture, an error filter declared by one harness changes only its own trajectories' datums; a registered counter sums across process-isolated rollout workers and a gauge is reported at exit; core names no `ptc_` key outside the deprecated path | |
| M5 the research extension moves | in its own repository: schemas, `check=`, `program_options`, trace filters and telemetry; its config classes and pre-flight command go | its configs load with no `trainer_config`, and its test suite passes against this platoon | |
| M6 core sheds the presentation | the ORS override points and the extension's service subclass; the options of section 9 and `viz --mode` deprecated | the extension's fixtures show that its subclass renders what core rendered; core's ORS tests pass on a branch with the deprecated options removed, as the release after their deprecation removes them | |

## Open questions

1. **`requires:` as a mapping.** This RFC proposes `{distribution: specifier}`, so that `extends:` merges it; the plan
   so far showed a list of requirement strings. Which form? Decide before M3. *Decided 2026-10-17:* the mapping
   `{distribution: specifier}`, as section 5 has it.
2. **Phase-specific options.** `eval_overrides` as proposed, or a phase flag that each component reads? Decide before
   M2. *Decided 2026-10-17:* `eval_overrides`, as proposed (section 2).
3. **`config validate` resolving by default.** It then needs the config's extensions installed wherever it runs, and
   `--no-resolve` keeps today's check; the default changes a minor release after M2 announces it. Agreed? Decide
   before M2. *Decided 2026-10-17:* it resolves by default from M1, with `--no-resolve`, and with no release of
   notice, since no release has shipped the command (section 6).
4. **How long deprecations last.** "Until 2.0" assumed a 1.x line, and the first PyPI release is 0.2.0. Keep 2.0,
   remove at 1.0, or remove after the minimum that [RFC 0017](0017-public-surface-and-conformance.md) sets? Decide
   before M1. *Decided 2026-10-17:* one minor release of warnings, RFC 0017's minimum; the next minor release
   removes the old form (section 10).
5. **Telemetry names.** `telemetry/<namespace>/<key>` renames the programmatic tool calling metrics when the extension
   moves. Accept the rename, or keep the old names as aliases until the deprecations end? Decide before M4.
6. **Environment options.** Should named environments take options (`configure`, as proposed), or stay static, with
   one version per variant? Decide before M2. *Decided 2026-10-17:* they take options, through `configure`, as
   proposed (section 4).
7. **What a check may read.** `check=` sees the phase's rollout config (sampling, root-success propagation, the
   verifier switch). Is that the right boundary, or should a check see only its entry? Decide before M2. *Decided
   2026-10-17:* the entry and the phase's rollout config, as proposed (section 3).
8. **The process-wide env-server pool.** `legacy_global_pool` (default on) lets an ORS entry fall back to
   `OPENREWARD_SESSION_URLS`, which the docs teach for runs with one remote environment. Keep the key, or apply the
   pool only when a config runs one remote environment and deprecate the key? Decide before M6.
