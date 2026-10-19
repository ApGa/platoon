# RFC 0013: The extension model

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-16 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | The harness registry (`platoon/harness/registry/`), episode programs (`platoon/programs.py`), the component registry (`platoon/registry.py`), the config loader (`platoon/utils/config.py`), [RFC 0008](0008-config-and-cli.md) (config and CLI), [RFC 0010](0010-harbor-registry-and-multi-container-tasks.md) (Harbor datasets by name) |
| Discussion | The pull request that adds this RFC. The number is assigned when that pull request opens; 0012 is tool-call parsing, so this draft takes 0013 |
| Implementation | In 0.2.0: milestones M1 (the model), M2 (the testing kit) and M3 (the docs), and M4 (stage 2, the layout of section 11), which retired `plugins/` |

## Summary

An **extension** is an installed distribution whose `pyproject.toml` lists the components it provides under
platoon's entry-point groups, one entry point per component, named exactly as configs name it. There are four
groups: `platoon.harnesses` (which exists), `platoon.programs`, `platoon.environments` (named environments: a
versioned, partial `environments` entry that a config selects with `env: <owner>-<name>@<version>`) and
`platoon.reward_processors`. A config names components; platoon checks from metadata that the extension supports the
running platoon, imports exactly the entry points the config names, in each process that needs them, and records
in `lock.json` the distribution and version behind every component a run resolved. Every component name follows
one rule, `<owner>-<name>[@version]`, with the `platoon` owner reserved for core. Listing reads metadata only.
The unused `platoon.plugins` group goes. A second stage, after the Harbor fast path merges, moves the in-repository
plugins to a uv workspace and retires the `platoon.<name>` namespace they install into.

## Motivation

Platoon has three ways for outside code to add components, none of them checked:

- **Import strings everywhere.** A config names programs, reward processors and environment classes as
  `module:attr` strings, and every recipe repeats the same module paths and a dozen keys per environment entry
  (the service, the harness's prompt builder, the program, the reward processor, the splits). A harness is the only
  kind an installed package can provide by name.
- **Every harness of every installed package, on the first lookup.** The first harness lookup in any process
  imported every installed `platoon.harnesses` entry point (`platoon/harness/registry/__init__.py`, before this
  RFC). A run that used only `platoon-codeact` imported a research extension and its whole agent SDK whenever that
  extension was installed, in the trainer and in every rollout worker, and installing a package could change what a
  run that never named it did.
- **A plugin group nothing used.** `platoon.plugins` entry points were imported only when an environment entry set
  `discover_entry_points: true`, and any failure aborted the run (`platoon/registry.py`), the opposite of the
  harness group's rule. No package declared one.
- **Two name rules, neither enough.** `@pt.harness` refused only empty names and `@`; TOML specs accepted upper
  case. `@pt.harness("acme:agent")` registered a name that `get_harness` read as an import string, so it could never
  resolve. Nothing stopped a package from registering a core name.
- **No compatibility statement.** Nothing compared an extension's requirement on platoon with the running version,
  so an extension built against renamed APIs failed at import, inside a rollout. The in-repository plugins required
  `platoon >= 0.1.0`, with no upper bound.
- **No provenance.** `lock.json` recorded the versions of four fixed distributions, not which distribution, at which
  version, provided a run's harness, program or environment.

The Harbor work needs the third group most: datasets registered by name, so that a recipe says
`env: acme-terminal-tasks@1` and site configs keep only site facts ([RFC 0010](0010-harbor-registry-and-multi-container-tasks.md)).

## Proposal

### 1. The model

| Term | Meaning |
|---|---|
| Component | A named thing a config selects: a harness, an episode program, a named environment or a reward processor. Services, sandbox providers, renderers and tool-call parsers stay import strings or core names for now |
| Extension | An installed distribution that declares components in `platoon.*` entry-point groups |
| Built-in | A component the `platoon` distribution registers itself. Its name is reserved |
| First-party extension | An extension this project releases from this repository, with the same API and checks as a third party's |
| Named environment | A versioned, partial `environments` entry registered under a name (`textcraft-synth@1`) |
| Recipe | Configs plus a README that reproduce one experiment. A recipe pins exact versions and is not an extension |

Installing an extension never changes a run that does not name one of its components, and nothing installs into
the `platoon` package. Code that is not installed is named by import string (`module:attr`), or imported with an
environment's `package:` when other code refers to it by its short name.

### 2. Extension points

Four kinds of component are found by name through an entry-point group:

| Kind | Declared with | Selected by | Entry-point group | An entry point loads to |
|---|---|---|---|---|
| Function harness, CLI harness spec | `@pt.harness(name, version=)`; a TOML spec | `harness:`, `ep.run(harness=)`, `ctx.agents.spawn(harness=)` | `platoon.harnesses` | a decorated function, a `HarnessSpec`, a spec path or directory, a module whose import registers harnesses, or a list of these |
| Episode program | `@pt.program(name, version=)` | `program:` | `platoon.programs` | a decorated function, a module whose import registers programs, or a list of decorated functions |
| Named environment | `pt.EnvironmentEntry(name, version=, ...)` | `env:` on an `environments` entry | `platoon.environments` | an `EnvironmentEntry`, or a list of them (several versions) |
| Reward processor | `fn(trajectory) -> (score, components)` | `reward_processor:` | `platoon.reward_processors` | the function |

A declaration in `pyproject.toml`:

```toml
[project]
name = "acme-platoon"
dependencies = ["platoon[ors]>=0.2,<0.3"]

[project.entry-points."platoon.harnesses"]
acme-agent = "acme_platoon.agent:acme_agent"

[project.entry-points."platoon.programs"]
acme-best-of-n = "acme_platoon.programs:best_of_n"

[project.entry-points."platoon.environments"]
acme-mazes = "acme_platoon.entries:MAZES"

[project.entry-points."platoon.reward_processors"]
acme-length-penalty = "acme_platoon.rewards:length_penalty"
```

The other extension points keep the references they have today:

| Kind | Declared with | Selected by | Found through |
|---|---|---|---|
| Environment service | an ORS `Environment` class, or an `EnvironmentService` | `service:` | a core name (`harbor`, `table`), a scheme (`python://`, `ors+http(s)://`) or `module:Class` |
| Sandbox provider | a `SandboxProvider` | the Harbor service's `provider` option | a core name (`process`, `docker`, `singularity`, `enroot`) or `module:Class` |
| Renderer | a tinker-cookbook renderer | `train.renderer_name` | tinker-cookbook's names, and core's `chat_template` with its aliases |
| Tool-call parser | a `ToolCallParser` subclass, or an object or function with its `parse` | `renderer_kwargs.tool_call_parser`, with `chat_template` | a core parser's name ([RFC 0012](0012-tool-call-parsing.md)) or `module:attr` |
| CodeAct prompt builder | a `CodeActPromptBuilder` subclass | `harness_options.prompt_builder` | `module:Class`; a named environment sets it as a harness default |
| Credit workflow, rollout, task and dataset loaders, trainer config | a `GroupRolloutWorkflow` subclass; callables; a config class | `workflow:`, `rollout:`, `task_loader:`, `dataset_loader:`, `trainer_config:` | `module:attr`, or a name a `package:` module registers |
| Process telemetry, error detection | functions | `platoon.hooks.register(name, fn)` | the import of the module that registers them |

A kind gets a group when a second independent user needs a short name for it: each group is public surface that
the version policy then covers. Named environments hide most of the remaining import strings (a service class, a
prompt builder) from configs anyway.

### 3. One discovery rule

```text
resolve(kind, ref)
  harness, ref ends in ".toml"   -> load that spec file
  ref contains ":"               -> import string module:attr[.attr...]
  otherwise ref is name[@version]:
    1. core's built-ins; their names are reserved (section 4), so no extension shadows them
    2. the entry point of group platoon.<kind> named exactly `name`, from installed metadata
       (scanned once per process), imported the first time a config or code names it
    3. names that modules imported through an environment's `package:` registered
    4. otherwise an error: what is registered, what is installed, the nearest names, what failed to
       load, and how to provide one
```

- **Versions.** Several versions of one name may be registered at once. `name@version` picks one, and a version
  prefix matches at dots (`mini-swe-agent@2` matches `2.4.6`). A bare name resolves only when exactly one version
  is registered; otherwise it is an error that lists them.
- **Loading one entry point**, in order: the entry point's name must follow the naming rule and may take a core
  name only from the project's own distributions; its distribution's requirement on platoon must admit the
  running platoon (section 6); then it is imported, the value must fit the kind, and a component it declares must
  carry the entry point's name. The result is cached, with the distribution and version recorded.
- **Failures are isolated.** An entry point that fails is recorded with why, and the error names it and its
  distribution. Only a lookup of that name fails; nothing else is imported.
- **Concurrency.** No lock is held while an extension imports. A thread that names an entry point another thread
  is importing waits for that import; lookups of every other name go on meanwhile.
- **Duplicates.** Two distributions that register the same `name@version` are an error that names both; a config
  picks one with an import string.
- **Harness entry points named after their module.** Before this RFC, an entry point could be named for its module
  (`my-harnesses = "my_harnesses.one_shot"`, registering `my-one-shot`). A harness name that matches nothing still
  loads every harness entry point not loaded yet, and a warning names each entry point to rename. The other groups
  are new and accept only exact names.
- **Import strings.** A `module:attr` reference may name a nested attribute (`pkg.mod:Outer.inner`) for the four
  kinds above and the registry's kinds (rollouts, loaders, workflows). The dotted form `module.attr`, and
  `pkg:mod.attr` read as `pkg.mod.attr`, keep working where they worked, and a dotted path that imports keeps its
  meaning when an installed reward processor entry point happens to have the same name.
- **Rollout workers** resolve names the same way, lazily, from the same installed metadata, read in one pass per
  process. On a network filesystem the pass took 0.03 s with 190 distributions installed and 0.13 s with 267,
  against a worker startup budget of 300 s (`worker_startup_budget_seconds`).

### 4. Names

```text
ref   := name ["@" version]
name  := [a-z0-9]+ ([._-][a-z0-9]+)*       lower case; never ":", "/", "@", or a ".toml" suffix
owner := the name's first "-"-separated word; one owner per distribution
```

- **Where the rule is enforced.** `@pt.harness`, `@pt.program`, `pt.EnvironmentEntry`, TOML specs, names that a
  `package:` module registers for reward processors, and every entry-point name when it loads.
- **The `platoon` owner is core's.** Names `platoon` and `platoon-*` may be registered only by code in the
  `platoon` package and by the project's own distributions (`platoon` and `platoon-*`), whether by entry point,
  decorator or `env: module:attr`.
- **Core's names without the prefix** are reserved the same way: the programs `single` and `gym`, and the specs
  `mini-swe-agent` and `openhands`, which keep their products' names. Core's other short names (the services `harbor`
  and `table`, the renderer `chat_template`, the workflow `group_rollout`, the sandbox providers) belong to kinds
  without a group yet, and stay core's when those kinds get one. Third-party CLI agents keep their product names.
- **One owner per distribution** is checked by `platoon extensions check`, from metadata, as a warning. There is no
  registry of owners, as with pytest's `pytest-` convention.

| Who | Distribution | Component names |
|---|---|---|
| core | `platoon` | `platoon-codeact`, `platoon-react`, `platoon-bash`; the core names `single`, `gym`, `mini-swe-agent`, `openhands` |
| first party | `platoon-number-search` | `platoon-number-search@1` (core's after stage 2) |
| first party | `platoon-textcraft` | `textcraft@1`, `textcraft-synth@1` |
| third party | `acme-platoon` | `acme-agent`, `acme-best-of-n`, `acme-mazes@1`, `acme-length-penalty` |

### 5. Named environments

A named environment says once how to run an environment: its service, the options each harness needs on it, its
program, reward processor and default splits.

```python
# acme_platoon/entries.py: imported only when a config names acme-mazes
import platoon as pt

MAZES = pt.EnvironmentEntry(
    "acme-mazes",
    version="1",
    description="Walk to the exit of a grid maze.",
    service="python://acme_platoon.env:Mazes",
    harness_defaults={
        "platoon-react": {"system_prompt": "Find the exit with move(direction)."},
        "platoon-codeact": {"prompt_builder": "acme_platoon.prompts:MazePrompts"},
    },
    eval_dataset_kwargs={"split_name": "val"},
)
```

```yaml
environments:
  - env: acme-mazes@1          # the lowest layer of this entry
    harness: platoon-react
    dataset_kwargs: {limit: 500}
```

- **The entry** takes any key of an `environments` entry except `env`, `name`, `weight`, `harness_options`,
  `exclude_tasks` and the run-level settings (`trainer_config`, `workflow`, `workflow_kwargs`,
  `eval_workflow_kwargs`), and is checked against the config schema when it is built, so a typo fails in the
  extension, not in the run.
- **Merging.** The named entry is the lowest layer, under the config's own keys after `extends:` and overrides:
  mappings merge key by key, and anything else, lists included, replaces; an explicit null overrides.
- **Tasks left out.** The entry that selects a named environment sets `exclude_tasks` like any entry: its items
  name the environment's tasks, and they leave each list before the named environment's `limit` takes its share. A
  named environment cannot set it: lists replace, so an entry that left out one more task would bring back every
  task the environment left out. An environment that should not offer a task leaves it out of its own task lists,
  in a new version (open question 8).
- **`harness_defaults`** maps a harness name to options. Those of the entry's harness become the lowest layer of
  its `harness_options`, so a config that picks another harness inherits nothing it cannot read. A harness mixture
  shares one `harness_options` mapping, so a mixture that includes a harness with defaults is a config error unless
  the entry sets that mapping itself, and then no defaults apply (open question 2).
- **Pinning.** The loaded config keeps `env: name@version` with the version the name resolved to, so
  `config.resolved.yaml` pins it, and loading the resolved config again changes nothing.
- **Versions.** A change to the tasks, the grading or the defaults bumps the version, and older versions stay
  registered, so a recipe pinned to `@1` keeps meaning the same thing.
- **Without installing.** `env: module:ATTR` names an `EnvironmentEntry` by import string.
- **Harbor datasets** register the same way: a package's `pt.EnvironmentEntry("acme-terminal-tasks", version="1",
  service="harbor", service_kwargs={...})`, which recipes name and site configs complete with site facts (image
  maps, toolchain roots, provider options).

TextCraft and number search stay ORS environments. In this milestone their plugins register `textcraft@1`,
`textcraft-synth@1` and `platoon-number-search@1`, which reproduce the entries their recipes spell out.

### 6. Compatibility

- **The requirement check.** Extensions declare a range, `platoon>=0.2,<0.3`. Before it imports an entry point,
  platoon reads the distribution's requirements on `platoon` from installed metadata (also those behind an extra,
  for an extension whose platoon glue is optional) and refuses one whose range excludes the running version, with
  an error that names both versions. The running version is the imported checkout's `pyproject.toml` when platoon
  runs from a source checkout, else its installed metadata.
- **For development**, `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` loads such an extension anyway, with a warning.
- **`platoon extensions check`** warns about a distribution that states no requirement on platoon, and about one
  with no upper bound.
- **No separate API version and no versioned group names.** Platoon's major version is the extension API version.

### 7. Provenance

`platoon train`, `eval` and `rollout` resolve every component the config names when they start, before anything
runs: named environments, harnesses (each in a mixture), the program, the reward processor, the service class, and
the import-string components (rollouts, loaders, a workflow). One that does not resolve is a configuration error,
exit 78, instead of a failure in every rollout. `lock.json` gains two sections:

```json
"components": [
  {"kind": "environment", "ref": "acme-mazes@1", "resolved": "acme-mazes@1", "source": "acme_platoon.entries:MAZES",
   "distribution": "acme-platoon", "distribution_version": "1.4.0",
   "entry_point": "platoon.environments:acme-mazes", "environments": ["default"]},
  {"kind": "harness", "ref": "acme-agent", "resolved": "acme-agent@2", "source": "acme_platoon.agent:acme_agent",
   "distribution": "acme-platoon", "distribution_version": "1.4.0",
   "entry_point": "platoon.harnesses:acme-agent", "environments": ["default"]},
  {"kind": "program", "ref": "single", "resolved": "single@1", "source": "platoon.programs:single",
   "distribution": "platoon", "distribution_version": "0.2.0", "entry_point": null, "environments": ["default"]}
],
"extensions": {
  "acme-platoon": {"version": "1.4.0", "requires": ["platoon[ors]>=0.2,<0.3"],
                   "direct_url": {"url": "file:///src/acme-platoon", "dir_info": {"editable": true}}}
}
```

- `ref` is the reference as the loaded config gives it (a named environment's is already pinned, section 5), and
  `resolved` is what it resolved to: `name@version` for a component found by name, `module:attr` for an import
  string.
- `direct_url` (PEP 610) shows an editable checkout, whose code may differ from its version.
- The distribution of import-string code is found from metadata, best effort: core's modules, then the distribution
  of a loaded entry point from the same package, then the package's installer, then the editable install whose
  directory holds the file; code that is not installed records none.
- Components that code names while a run is under way, such as a harness a program spawns by name, are not in the
  lock; the trajectories record them.
- The policy's tool-call parser is recorded next to these sections, in `tool_call_parsing`
  ([RFC 0012](0012-tool-call-parsing.md)): its name or `module:attr`, without the distribution behind it. Once
  `train` or `eval` has listed the tasks, `excluded_tasks` joins them: the ids each entry's `exclude_tasks` left out
  of each list, an entry that selects a named environment included.

### 8. Listing and checking

```text
platoon extensions list [--kind K] [--json-output]   every component: built-ins, then each extension's, from metadata
platoon extensions show NAME [--kind K] [--load]     one component, its entry point and where it is used; --load imports
platoon extensions check [DIST...] [--load]          names, owners, platoon requirements, retired groups; --load imports
platoon harness list | show NAME                     adds the distribution and version of each harness
platoon program list | show NAME                     new
platoon env list [CONFIG] | show NAME                adds the named environments; show prints one with its defaults
platoon config validate --resolve CONFIG...          also resolves every component, as a run's start does
```

Listing imports nothing of an extension; an extension's harness shows its kind once a command imports it, and an
extension's version once `extensions show --load` does. `extensions check` exits 78 on an error, and `extensions show`
on a name nothing installed has. `config validate` lists no tasks, with or without `--resolve`, so `train`
and `eval` check the `exclude_tasks` items when they list the tasks, before anything runs.

### 9. The testing kit

`platoon.testing` ships in the core wheel, and starts small:

- `toy_tokenizer()`, a byte-level toy tokenizer of about 7 KB that carries Qwen3's chat tokens, so tinker-cookbook's
  stock `qwen3_instruct` renderer works over it, and the gateway, harness and example tests run in CI without
  downloading a model's tokenizer; only tests of a model's own token ids (the CodeAct goldens) still need it;
- `ScriptedPolicy`, which stands in for the Tinker sampler behind the real model gateway (`make_gateway`);
  `run_episode`, which runs one task through `harness_rollout`, so an extension author tests a harness, a program or
  an environment end to end on a CPU; and `serve_policy`, which serves the policy to `platoon rollout` and
  `platoon eval` run in the test's process, so a test runs a config that names components by their entry points;
- the same as fixtures of a pytest plugin that is opt-in (`pytest -p platoon.testing`) and never loaded
  automatically.

[Test an environment, a harness or a program on CPU](../how-to/test-extensions.md) documents the kit, and the
example package of [Publish an extension package](../extending/publish-a-package.md) tests itself with it.

Conformance suites (`EnvironmentConformance`, `HarnessConformance`), golden-record helpers and `platoon new`
templates come later, once the API they exercise has settled.

### 10. What changes for users

- **Removed:** the `platoon.plugins` entry-point group, `platoon.registry.discover_entry_points`, the environment
  key `discover_entry_points`, and that argument of the evaluation and training workflows and of
  `import_environment_packages`; `AutoEnvironment.packages` returns the `package` modules only.
- **Changed:**
  - harness entry points load by exact name, the first time each is named, instead of all on the first lookup;
  - `@pt.program` takes `version=` (default `1`), and programs are `name@version`;
  - names follow the rule of section 4: TOML spec names are lower case, and `platoon-*`, `single`, `gym`,
    `mini-swe-agent` and `openhands` belong to core;
  - `platoon harness list` has a distribution column;
  - `train`, `eval` and `rollout` fail with exit 78 on any component that does not resolve, and their `lock.json`
    lists components and extensions;
  - `module:attr` import strings may name nested attributes (section 3).
- **New:** the `platoon.programs`, `platoon.environments` and `platoon.reward_processors` groups; `env:`;
  `pt.EnvironmentEntry`; the `platoon.extensions` module; the `platoon.testing` kit; `platoon extensions`,
  `platoon program`, `platoon env show` and `config validate --resolve`; `PLATOON_IGNORE_EXTENSION_REQUIREMENTS`.
- **Unchanged:** import strings, `package:`, `.toml` paths, and trajectory records, which keep the `program`
  string as configured.

### 11. Stage 2: the layout

The second stage changes the repository's layout, after the Harbor fast path merges: the Harbor work in flight and
the CodeAct goldens import `platoon.number_search`, so moving it first would break them. Nothing in stage 1 depends
on the layout, and the names stage 1 registers survive the move.

```text
pyproject.toml                    name "platoon"; [tool.uv.workspace] members = ["extensions/*"]
uv.lock                           one lock for core and every first-party extension
platoon/                          core, the only thing in core's wheel and sdist
├── envs/number_search/           platoon-number-search: the ORS environment and its CodeAct prompts
├── rewards.py                    delegation_capped as platoon-delegation, the subagent reward finalizers
└── testing/                      the testing kit
extensions/textcraft/             distribution platoon-textcraft, import platoon_textcraft, owner textcraft
recipes/rao/                      the RAO arms, a README with published numbers, a CPU check that each resolves
examples/                         tutorials, CPU-tested; not workspace members
tests/                            core tests, which never import an extension
```

- **Number search** moves into core with its prompt builder. Its environment keeps importing only openreward and
  pydantic, and its tasks are generated from their seed instead of committed files, after a test shows that the
  generated rows equal the committed ones. `platoon-number-search@1` becomes a built-in, so configs that name it keep
  working, and the first run works from a wheel with nothing else installed.
- **TextCraft** becomes `extensions/textcraft`: a module that imports only openreward, pydantic and its own recipes
  for the environment, and the platoon glue (prompt builders, entries) behind an optional `platoon` extra, whose
  requirement the compatibility check reads. `textcraft@1` and `textcraft-synth@1` keep their names and versions.
  Its task files keep their gold trajectories: the Harbor export of TextCraft-Synth, which came after this section
  was written, builds every task's reference solution from them, so dropping them would change the published tasks.
  They ship in the wheel (3.7 MB with them) under a SHA256 manifest, and the data nothing references goes. Its goldens
  move with it and still run on every core pull request.
- **Generic code moves to core.** `textcraft-rao` folds into `single`, which honors `skip_subagent_reward_computation`
  in the same order, and `delegation_capped` becomes core's reward processor `platoon-delegation`.
- **The namespace goes.** `extend_path` in `platoon/__init__.py`, the plugins' own lock files, the CI `PYTHONPATH`
  and the type checker's plugin overrides are removed, so no distribution installs into the `platoon` package.
- **CI** runs core's tests with no extension installed, one job per extension, the recipes' resolve check, and the
  built wheels in a clean environment.

| Before | After stage 2 |
|---|---|
| `python://platoon.number_search.env:NumberSearchEnvironment` | `env: platoon-number-search@1` |
| `python://platoon.textcraft.env:...`, `platoon.textcraft.prompts:...` | `env: textcraft@1` or `env: textcraft-synth@1`, or `platoon_textcraft.*` |
| `program: platoon.textcraft.programs:rao` | omitted (`single` behaves the same) |
| `reward_processor: platoon.textcraft.rewards:delegation_capped` | `platoon-delegation`, which `textcraft-synth@1` sets |
| `uv pip install -e plugins/textcraft` | `uv sync --package platoon-textcraft --extra platoon`, or `pip install platoon-textcraft` once it is published |

### 12. Left to later RFCs

- **Component configuration.** `options=` on `@pt.harness` and `@pt.program`, checked when a config loads; a
  `program_options:` key beside `harness_options`; `requires: ["platoon-textcraft>=1,<2"]` in configs, checked from
  metadata before validation; `config validate` resolving by default; and retiring `trainer_config` subclasses, whose
  only job today is option checking. A `trainer_config` that does not resolve is already a configuration error,
  instead of a warning after which the config validated against the base schema.
- **Callbacks owned by components** instead of process-wide hooks: trace filters declared on the harness that
  produces the trajectories, and telemetry a component registers when it starts.
- **The public surface.** Release metadata (`__version__`, a license field, `py.typed`; the PyPI-readiness work
  sets it, and numbers the first release 0.2.0 rather than 1.1.0), and public modules for what extensions import
  today from private ones (the CodeAct prompt-builder API, the reward finalizers).
  The committed signature snapshot of the public API (`tests/public_api.txt`) and the check that the bundled plugins
  and the examples import only it came first, as tests; the check lists every private import they still make.
- **More provenance.** Each environment server's version and each dataset's revision in `lock.json`, and the
  distribution behind a renderer or tool-call parser that a package provides; canonical `name@version` program
  identities in trajectory records (open question 1).
- **More commands.** A component's option schema and docs in `extensions show NAME`, which shows one component
  from its metadata already; one distribution's components with a flag of its own, since a distribution may be named
  like a component it provides (`extensions show --dist DIST`); and a `platoon doctor` that finds stale editable
  installs and leftover metadata of retired groups.

## The hard parts

- **Exact names against the old harness entry points.** An entry point named after its module is found only by
  loading every harness entry point on a miss. That keeps old packages working but costs a full load whenever a
  name is wrong, so the warning asks for a rename, and the fallback should go at the next major version.
- **Which distribution holds import-string code.** Editable installs record no module files, and the
  in-repository plugins share the `platoon` namespace with core, so the lookup works from the loaded entry points
  and the editable directories. It is best effort until stage 2 removes the namespace.
- **Which platoon is running.** A source checkout on `PYTHONPATH` can shadow an installed copy, so the check reads
  the imported checkout's version first.
- **Harness options in a mixture.** One `harness_options` mapping serves every harness of a mixture, so per-harness
  defaults cannot apply there yet.
- **Imports that name other components.** An extension's module may look components up while it imports. On the
  importing thread, a lookup of its own entry point's name returns at once, before the module has registered
  anything; and two threads whose imports each name the other's entry point would wait for each other, as Python's
  own imports can. Lookups at import time are best avoided.

## Alternatives

- **One group, with a registration module per owner.** Listing would import every extension, and loading by name
  prefix picks the wrong package when one owner's name prefixes another's (`acme` and `acme-tools`).
- **One group pointing at manifest objects.** The manifest repeats what the decorators already say.
- **Autoloading every installed extension**, pytest style. Loading only what a config names is faster, cannot
  change runs that do not name the extension, and records provenance per component.
- **pluggy hooks, versioned group names or a separate API number.** Platoon's seams each select one named
  component, so typed functions and one version number are enough.
- **A settings block per extension at the top of the config.** Settings attach to components (section 12).
- **A separate repository for first-party extensions.** It adds a second CI and release cadence before any outside
  contributor exists, and a contract change and its fix could no longer land in one pull request.
- **Keeping `platoon.<name>` as a redirect namespace.** A wheel that ships `platoon/__init__.py` overwrites core's
  on install and deletes it on uninstall, and the namespace presents outside code as platoon's own.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M1 the model | the four groups and exact-name loading; named environments and `env:`; the naming rule; the compatibility check; components and extensions in `lock.json`; the listing and checking commands; `platoon.plugins` and `discover_entry_points` removed; the plugins' named environments | `tests/test_extensions.py`, `tests/test_named_environments.py`: a name imports only its entry point, listing imports nothing, an incompatible extension is refused before import, `textcraft-synth@1` reproduces its recipe's entry, `lock.json` names every component's distribution; `tests/integration/test_installed_extensions.py`: a conflicting extension installed into a scratch venv changes nothing in a run that does not name it | |
| M2 the testing kit | the toy tokenizer, the scripted policy, the episode helper and the opt-in pytest plugin in `platoon.testing` | the gateway, inline, built-in harness and example tests run in CI without a downloaded tokenizer; the CodeAct goldens, which hold Qwen3's token ids, still need it | |
| M3 the docs | extension authoring guides and tutorials for each extension point, with the generated reference | the docs tests; every tutorial runs on a CPU | |
| M4 stage 2 | the layout of section 11, after the Harbor fast path | the moved goldens pass unchanged; core's tests pass with no extension installed; the wheels install in a clean environment | |
| M5 component configuration | section 12's first item, through its own RFC | a typo in an option fails at load with exit 78 | |
| M6 the public surface | section 12's third item, conformance suites, `platoon new` templates | CI fails on a public API change without a snapshot update | |

## Open questions

1. **Program identity in records.** Trajectory records keep the configured `program` string, which code outside
   core compares against. Should they store the canonical `name@version`, and in which release?
2. **Per-harness options in a mixture.** Should an entry carry options per harness, so named environments work
   with harness mixtures? Needed before a recipe mixes harnesses on a named environment.
3. **`config validate` resolving by default.** It checks files and named environments only unless given
   `--resolve`; resolving imports every component, so it needs the extensions installed where the config is
   checked. Decide with M5.
4. **The `platoon-*` distributions.** Any distribution named `platoon-*` may use the core owner. Is a convention
   enough, or should core keep a list of the project's distributions?
5. **Owners.** One owner per distribution is a warning. Should loading refuse a second owner?
6. **Terminology.** Should the documentation say "extension" throughout, and `plugins/` become `extensions/` in
   stage 2? *Answered by stage 2:* the docs say "extension", the first-party extensions live in `extensions/`, and
   the configs that reproduce experiments in `recipes/`.
7. **Version policy.** Release as 1.1.0 with semantic versioning from the first public release, with an
   experimental module for the areas still in flux? *Answered by the PyPI-readiness work:* the first PyPI release
   is 0.2.0, since the original platoon already called itself 0.1.0. Below 1.0 a minor release may break a stable
   name, so extensions bound their requirement below the next minor release (`platoon>=0.2,<0.3`). RFC 0017 sets
   the full policy.
8. **Tasks a named environment leaves out.** Should a named environment set `exclude_tasks`, for tasks it knows
   cannot be graded, with an entry's own items added to its list instead of replacing it? Until then it leaves
   such tasks out of its own task lists.
