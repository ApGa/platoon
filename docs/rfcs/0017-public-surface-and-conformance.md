# RFC 0017: Public surface and conformance

| | |
|---|---|
| Status | Draft |
| Created | 2026-10-16 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0013](0013-extension-model.md) (the extension model: this RFC is its milestone M6, with the public surface and the commands of its section 12; the extra provenance that section lists is left to a later RFC, section 11), [RFC 0016](0016-component-configuration.md) (component configuration), the package's exports (`platoon/__init__.py`), the testing kit (`platoon/testing/`), the generated reference (`scripts/gen_reference.py`), the checks that came with RFC 0013's follow-ups (the signature snapshot `scripts/public_api.py` writes to `tests/public_api.txt`, `tests/test_import_boundaries.py`, `platoon extensions show NAME`), the core boundary test (`tests/test_core_boundary.py`), [RFC 0008](0008-config-and-cli.md) (the command line and `doctor`) |
| Discussion | The pull request that adds this RFC. The number is assigned when that pull request opens; drafts on other branches hold 0014 and 0015, and 0016 is component configuration, so this draft takes 0017 |
| Implementation | None yet; the signature snapshot of three modules, the import-boundary test of the first-party extensions and the examples, and `extensions show NAME` came with RFC 0013's follow-ups |

## Summary

Platoon states which of its modules and names extensions may rely on. One table lists the public modules, each
stable, experimental or deprecated, and a public module's `__all__` lists its public names; everything else is
private: importable, but outside the version policy. New public modules give a stable home to what the docs and the
extensions import from private modules today: the CodeAct prompt-builder API, the reward finalizers, the judges, the
tool-call parser API and the keys of the record format. The committed snapshot of public signatures, which covers
`platoon`, `pt.gym` and `platoon.testing` today, grows to every public module and keeps failing CI on any change
until it is regenerated, and a pull request that removes or changes a stable name must say that it breaks; the
import-boundary check, which covers the first-party extensions and the examples today, grows to the docs and empties
its list of exceptions. `platoon.testing` gains conformance suites for environments, harnesses, programs, reward
processors and sandbox providers, which an extension subclasses in its own tests and which `platoon env check` and
`platoon harness check` run from the command line, and golden-record helpers. `platoon new` generates an extension
package that core's CI keeps working, a generated `docs/reference/extensions.md` documents the extension surface,
`platoon extensions show` grows from one component's metadata to its options and to a whole distribution
(`--dist`), and `platoon doctor` checks the installation.

## Motivation

- **Nothing says what is public.** `pt.*` holds 29 names (`platoon/__init__.py`). The guides and tutorials import
  from 17 modules besides `platoon` itself: `platoon.harness.builtin.codeact` for the CodeAct prompt builder
  (`docs/tutorials/builtin-harnesses.md`), `platoon.utils.trajectory_status` and `platoon.programs`
  (`docs/extending/rewards.md`), `platoon.harness.agent` for `LIMIT_MESSAGE` (`docs/extending/publish-a-package.md`),
  `platoon.train.tinker.renderers.tool_parsers` for tool-call parsers (`docs/how-to/models-and-tool-calling.md`), and
  more. Of those, only `platoon.testing` says that its exports are stable (`platoon/testing/__init__.py`). The
  in-repository plugins import six modules besides `platoon` (their prompt builders subclass CodeAct's), the examples
  six, and a research extension 16, ten of them with no counterpart in `pt.*`. Any refactor of those modules breaks
  them, and nothing says so until an extension fails.
- **Changes to most of the surface are invisible in review.** The API reference (`docs/reference/api.md`) is
  regenerated from the code and checked for staleness (`tests/docs/test_reference_up_to_date.py`), but it covers
  `pt.*` only and mixes signatures with docstrings, so that every documentation fix touches it. The signature
  snapshot (`tests/public_api.txt`) shows every change to `platoon`, `pt.gym` and `platoon.testing` in the diff, but
  no other module, and nothing makes a pull request that removes a name say that it breaks.
- **Every extension re-tests the contract by hand.** `platoon.testing` (RFC 0013, section 9) runs an episode with a
  scripted policy, and each extension then writes its own checks of what platoon expects of it: a terminal tool,
  grading that never raises, fork modes that work, closing twice, a crash that becomes `error/harness_crash`, a
  budget that holds. The rules are written down
  ([Rules every extension follows](../extending/index.md#rules-every-extension-follows)), but nothing checks them,
  and a misspelled ORS hint (`platoon_terminal_tool` for `platoon_terminal_tools`) is ignored silently.
- **A new extension starts as a copy.** `examples/extension-package/` is the only template: its layout, entry points
  and version bound are copied by hand.
- **The installation is hard to inspect.** `platoon extensions list`, `show NAME` and `check` read metadata (RFC
  0013, section 8), but nothing shows a component's options, or one distribution's components, and nothing finds an
  editable install
  whose metadata no longer matches its checkout (the research extension's test configuration re-registers its
  harness entry point by hand for that reason), metadata left in the retired `platoon.plugins` group, or two
  distributions that provide one `name@version`.

## Proposal

### 1. What is public, and how it is declared

```python
# platoon/_api.py
Stability = Literal["stable", "experimental", "deprecated"]

PUBLIC_MODULES: dict[str, Stability] = {
    "platoon": "stable",
    "platoon.codeact": "stable",
    # ...
    "platoon.experimental": "experimental",
    "platoon.hooks": "deprecated",
}
```

- **The rule.** A name is public when it is in the `__all__` of a module that `PUBLIC_MODULES` lists. Everything
  else is private, including the names a public module merely imports and every module the table does not list,
  with or without a leading underscore. Private code stays importable; it is only outside the version policy
  (section 2).
- **Stabilities.** Stable names follow the version policy. Experimental names are documented, but may change in any
  minor release, as the release notes say. Deprecated names keep working and warn (`pt.PlatoonDeprecationWarning`,
  RFC 0016, section 10) until they are removed.
- **One table.** The API snapshot (section 3), the import-boundary check (section 4) and the reference generator
  (section 8) read it; it replaces the `PUBLIC_MODULES` list of `scripts/public_api.py`, which the snapshot and the
  import-boundary test read today. Each listed module states its stability in its docstring, and a test checks that
  every listed module imports, has an `__all__`, and that every name in it resolves, as tinker-cookbook's downstream
  compatibility tests do for its package exports (`tinker-cookbook: tests/downstream_compat/test_package_exports.py`).
- **Types.** The wheel carries a `py.typed` marker (PEP 561), so that type checkers in an extension's repository read
  the annotations the snapshot records.

| Module | Stability | Holds |
|---|---|---|
| `platoon` (`pt`) | stable | the agent API, the decorators, programs, the environment protocols, records and named environments; also `TrajectoryStep`, the agent-contract helpers (`LIMIT_MESSAGE`, `out_of_steps`, `terminal_tool`, `terminal_arguments`) and RFC 0016's names |
| `platoon.envs` | stable | environment services: `get_service`, `task_table`, `TaskTableService`, `OrsEnvironmentService` with its override points (RFC 0016, section 9), `HarborEnvironmentService`, and the names of the env-server pool's variables |
| `platoon.sandbox` | stable | `Sandbox`, `SandboxProvider`, `SandboxSpec`, `ExecResult`, `SandboxSnapshot`, `Mount`, `get_provider` |
| `platoon.gym` | stable | `loop`, `as_environment`, `StepEnv`, the `gym` program and its return key, as its `__all__` lists them today |
| `platoon.programs` | stable | `single`, for programs that wrap it, beside `program`, `EpisodeContext` and `EpisodeProgram` |
| `platoon.trajectory` | stable | the record types (`Trajectory`, `TrajectoryCollection`, `TrajectoryStep`, `ParentInfo`) and the keys of the record format (the agent record, outcomes, spans, `completion_id_for_step`) |
| `platoon.codeact` | stable, new | the CodeAct prompt-builder API: `CodeActPromptBuilder`, `CodeActState`, `CodeActStep`, `PromptMode`, `CellFunction` (the functions a builder puts in the cell namespace), `CodeActOptions`, `Checks` |
| `platoon.rewards` | stable, new | the finalizers (`propagate_root_success`, `add_direct_subagent_delegation_rewards`), the trajectory status helpers (`trajectory_was_interrupted`, `trajectory_has_forced_zero_reward`) and the keys they read |
| `platoon.judges` | stable, new | the verifier and judge API: the behavior judge protocol, `apply_subagent_parent_usage_gates`, the judgment keys |
| `platoon.tool_calls` | stable, new | the tool-call parser API of [RFC 0012](0012-tool-call-parsing.md): `ToolCallParser`, `ParseResult`, `ParsedToolCall` |
| `platoon.telemetry` | stable, new | `register` (RFC 0016, section 7) |
| `platoon.extensions` | stable | the resolver and listing API for tools: `list_components`, `show_components`, `check_extensions`, `ComponentInfo`, `ComponentDetails`, `Problem`, `Origin`, `ExtensionError`, `GROUPS`, `NAME_PATTERN`, `check_name`, `platoon_version`, `suggested_requirement`; its other names leave `__all__` |
| `platoon.testing` | stable | the testing kit, the conformance suites (section 5), and `ChatRequest`, which callers of `make_gateway` need |
| `platoon.cli` | stable | `main`, the console entry point, which tests call to run commands in process |
| `platoon.experimental` | experimental, new | the seams for advanced use that the extension index lists: `GroupRolloutWorkflow`, `group_baselines`, `BatchTransform` |
| `platoon.hooks` | deprecated | `register` and `hooks` (RFC 0016, section 7) |

What is imported from private modules today, and where it goes:

| Imported today from | By | Becomes |
|---|---|---|
| `platoon.task`, `platoon.envs.service`, `platoon.harness.agent`, `platoon.harness.function`, `platoon.harness.registry`, for names `pt` exports already (`Task`, `Outcome`, `AgentContext`, `harness`, `get_harness`, ...) | plugins, research extension | `pt` |
| `platoon.harness.builtin.codeact` | plugins, docs | `platoon.codeact` |
| `platoon.harness.agent` (`LIMIT_MESSAGE`, ...) | docs, examples | `pt` |
| `platoon.harness.builtin.react` (`openai_tool`) | examples | a method of `pt.ToolSpec` |
| `platoon.utils.trajectory_status`, `platoon.utils.subagent_rewards` | docs, examples, plugins, research extension | `platoon.rewards` |
| `platoon.harness.verify` | docs, research extension | `platoon.judges` |
| `platoon.utils.trajectory_error_filtering`, `platoon.utils.critical_path` | docs, research extension | `platoon.trajectory` |
| `platoon.harness.inline` (`grade_session`, `with_reason`) | research extension | `EpisodeContext.grade_session()` and `Outcome.with_reason()` |
| `platoon.envs.ors`, `platoon.envs.ors.pool`, `platoon.envs.harbor` | docs, research extension | `platoon.envs` |
| `platoon.train.tinker.renderers.tool_parsers` | docs | `platoon.tool_calls` |
| `platoon.train.tinker.workflows.group_rollout_workflow` | docs | `platoon.experimental` |
| `platoon.gateway.core` (`ChatRequest`) | docs | `platoon.testing` |
| `platoon.hooks` | docs, research extension | `pt.TraceFilters` and `platoon.telemetry` (RFC 0016) |
| `platoon.utils.config`, `platoon.train.tinker.config_defs`, `platoon.eval.run` (config classes to subclass, `resolve_config`) | research extension | nothing: RFC 0016 retires the subclasses, and loading configs from Python waits for RFC 0008's schema |

- **Old paths keep working while deprecated.** `from platoon.harness.builtin.codeact import CodeActPromptBuilder`
  still works, with a warning, until the deprecations of RFC 0016 end.
- **The first-party extensions' and the examples' share of this table** is what `PRIVATE_IMPORTS` in
  `tests/test_import_boundaries.py` lists today, each import with why; M1 moves them to the public names and empties
  that list.
- **Formats and the command line are public too**, under the same policy: the config schema
  (`docs/reference/config.md`); the commands, their options and their exit codes (`docs/reference/cli.md`); the TOML
  harness spec format; the model gateway's HTTP endpoints; the record formats (trajectories, `config.resolved.yaml`,
  `lock.json`); the entry-point groups and the naming rule; and the conventions by which an environment or a harness
  meets platoon (`platoon_*` attributes on ORS environments, `_meta["platoon"]` on the MCP tool plane, and the
  environment variables of section 8). A TOML spec may state the version of its format (`schema = 1`); a spec that
  states a newer version than the running platoon reads fails to load with an upgrade hint.

### 2. The version policy

- **Patch releases** change no public name or format, except to fix a bug.
- **Before 1.0**, a minor release may remove or change a stable name: in a commit marked as breaking (`!` and a
  `BREAKING CHANGE:` footer), with a release note, and, where a replacement exists, only after the old form has
  warned for at least one minor release. Extensions therefore bound their requirement below the next minor release
  (`platoon>=0.2,<0.3` on 0.2.0, the first PyPI release). `platoon extensions check` suggests that bound below 1.0
  (`suggested_requirement` in `platoon/extensions.py`), and from M1 it also warns about a wider one. This replaces
  the range RFC 0013 showed (`platoon>=1.1,<2`), written when the first release was to be 1.1.0. The PyPI-readiness
  work already moves the suggestion, the example package, the plugins, the publishing guide and RFC 0013's examples
  to one minor release.
- **From 1.0**, semantic versioning: removals and changes only in a major release, after at least one minor release
  of warnings.
- **Experimental** names may change in any minor release, never in a patch release.
- These are minimums, and RFC 0016's deprecations keep to them: one minor release of warnings.
- CONTRIBUTING's [list of public contracts](../../CONTRIBUTING.md#when-a-change-needs-an-rfc) gains a pointer to
  section 1's table, which names the public modules.

### 3. The API snapshot

The snapshot exists: `scripts/public_api.py` writes `tests/public_api.txt`, and `tests/test_public_api.py` fails
while the file differs from the code, printing the lines added and removed and the command that rewrites it. It
has one line for each public name, and for each public member of a public class (methods, properties, dataclass and
model fields, protocol members, enum members): its dotted path and its signature, with string annotations as the
source writes them and the others as Python formats them, and a constant's value when it is a literal. Docstrings
are left out, so documentation never changes it, and the lines are sorted. This RFC changes what it covers and adds
a line per module:

```text
# tests/public_api.txt (its header left out)
platoon: module, stable
platoon.AgentResult: @dataclass(frozen=True) class AgentResult
platoon.AgentResult.error: field str | None = None
...
platoon.codeact: module, stable
platoon.codeact.CodeActPromptBuilder: class CodeActPromptBuilder
...
platoon.testing.run_episode: async def run_episode(task: Task | str, *, service: str, harness: str, ...) -> EpisodeRun
```

- **Every public module.** It covers `platoon`, `pt.gym` and `platoon.testing` today, from the list in
  `scripts/public_api.py`; it reads section 1's table instead, and each module gets a line with its stability.
- **Breaking changes say so.** A CI step compares the snapshot with the pull request's base. When a line of a stable
  module is removed or changed, the pull request's title must mark a breaking change (`feat!: <summary>`), which the
  title check already parses (`.github/workflows/pr-title.yml`). Lines of experimental modules are listed, but exempt.
- **The API reference** (`docs/reference/api.md`) grows from `pt.*` to every public module, read from the same
  table, with each module's stability.
- tinker-cookbook asserts the parameter names of the functions its downstream users call
  (`tinker-cookbook: tests/downstream_compat/sig_helpers.py`). A snapshot covers every public name without a test
  for each.

### 4. Import boundaries

`tests/test_import_boundaries.py` reads the source today: core never imports a first-party extension or an installed
extension's package, by an import statement or by an import string, and the extensions and the examples import only
the names the snapshot's modules export, apart from the private imports it lists with why (`PRIVATE_IMPORTS`). With
this RFC, three checks run in CI:

1. **The docs, the examples and the first-party extensions use the public surface.** The check reads every
   `import platoon...` and `from platoon... import ...`, and the attributes of names bound to public modules, from the
   syntax trees of the first-party extensions (`extensions/*/src`), of
   `examples/`, and, new, of the Python blocks and inline code of the user docs (`README.md`, `docs/getting-started/`,
   `docs/tutorials/`, `docs/how-to/`, `docs/extending/`), and fails on a private module or on a name outside a public
   module's `__all__`. `CONTRIBUTING.md`, the architecture page and the RFCs describe the internals and are exempt.
   The list of exceptions keeps each one's reason while it moves, and the test fails on an entry that is no longer
   needed, as it does today.
2. **Core imports no extension.** Beside the source check, `tests/test_core_boundary.py` looks for a fixed list of
   module prefixes after importing a few entry modules. It becomes: importing each core module loads nothing from any
   distribution other than platoon and its declared dependencies, computed from installed metadata.
3. **Public modules import cheaply.** `import platoon`, `platoon.extensions` and `platoon.testing` load no package of
   an optional extra (tinker, tinker-cookbook, openreward, hpc-sandbox), so that listing, checking and the command
   line stay fast, and an extension's metadata checks run without the extras.

For third parties, `platoon extensions check --imports [DIST...]` runs the first check over an installed
distribution's own files (from its record of installed files, or from its editable checkout) and reports each import
of a private module as a warning, with the public replacement when section 1's table names one.

### 5. Conformance suites

```python
# tests/test_conformance.py, in an extension
import platoon as pt
from platoon.testing.conformance import EnvironmentConformance, HarnessConformance


class TestMazes(EnvironmentConformance):
    environment = "acme-mazes@1"  # a named environment, python://module:Class or ors+http://...
    tasks = ["train:0", "val:3"]  # default: the first task of each split

    async def solve(self, session: pt.EnvSession) -> None:  # optional: play a task to its best score
        await session.call_tool("move", {"direction": "north"})
        ...


class TestAgent(HarnessConformance):
    harness = "acme-agent"
    driver = "native-tools"  # how the scripted policy writes replies that the harness reads
```

- Each suite is a set of named checks, each a test of its own. A check that does not apply (a fork mode the
  environment does not declare, a solution it does not provide, a capability a provider lacks) is skipped, with the
  reason. The same checks run without pytest (`platoon.testing.conformance.run(kind, ref, ...)`, which returns a
  report) for the commands of section 6.
- They check the documented contract and nothing more: the rules every extension follows, the protocols in `pt` and
  `platoon.sandbox`, and the options of RFC 0016.

| Suite | Checks |
|---|---|
| `EnvironmentConformance` | splits and tasks list, and a task id round-trips through `get_task`; a session opens with an instruction and with tools whose input schemas are valid JSON Schema; a terminal tool, or `platoon_grade`, exists; grading an untouched session returns an `Outcome` and never raises; closing twice is safe; each fork mode the environment declares works, and a `shared` fork with a goal of its own is graded on that goal; unknown `platoon_*` attributes are reported; with `solve`, the solution scores the best reward and an untouched session does not; outcomes are the same in process and over HTTP, through an env server the suite starts |
| `HarnessConformance` | the option schema rejects an unknown key and accepts the defaults; a final answer ends `completed`; the step budget holds (`limit` after at most `max_steps` policy calls); a tool error reaches the harness without ending the agent; an exception becomes `error` with `harness_crash`; cancellation ends the agent promptly and closes its children; every model call goes through the gateway; no call blocks the event loop for longer than a threshold; declared trace filters return completion ids of the trajectory |
| `ProgramConformance` | the program returns an `Outcome`; a grader that raises and a root that crashes end untrainable, never scored; finalizers run once, in order; the option schema rejects an unknown key |
| `RewardProcessorConformance` | on the kit's sample trajectories (a root, a subagent, an untrainable one, one with no steps) it returns a finite score and finite components, deterministically, and leaves the trajectory unchanged |
| `SandboxProviderConformance` | `exec` reports exit codes and output and honors the environment, the working directory and stdin; a timeout kills the command and sets `timed_out`; `upload` and `download` copy files and directories; read-only mounts stay read-only; `network: none` holds; `snapshot` and `restore` round-trip files when the provider declares the capability; `stop` kills what runs and is idempotent |

- **Drivers.** A harness check needs a policy that writes in the harness's format. `native-tools`, the default,
  writes the model's native tool calls with the kit's `tool_call` replies, which suits every harness that acts
  through tool calls (`platoon-react`, command-line agents over MCP, most agent SDKs). A harness with a format of its
  own provides a small driver: three replies, one that finishes with an answer, one that calls a tool with
  arguments, and one that does not parse. The kit ships the driver for `platoon-codeact`.
- **Toy environments.** The checks run on small environments that join `platoon.testing` (a submit-only environment,
  a forkable workspace, a flaky one), taken from core's own test toys (`tests/envs/ors_toys.py`,
  `tests/envs/concurrency_toys.py`, `tests/harness/builtin/submit_env.py`).
- **Core runs them first**, on its own components: the built-in harnesses and programs, the example environments, and
  the `process` and `docker` sandbox providers where CI has them. A suite becomes public only once core passes it.
- **Golden records**, which RFC 0013 left for later with the suites. `platoon.testing.golden` records an episode's
  trajectories and compares a later run with them, with normalizers for ids and times. It stores the tokenizer's
  fingerprint and compares token fields only when it matches, so a golden recorded with a model's tokenizer still
  checks messages, tool calls and outcomes under the toy tokenizer; core's CodeAct goldens are the first users.

### 6. `platoon env check` and `platoon harness check`

```text
platoon env check REF [--task ID ...] [--http] [--json-output]
platoon harness check REF [--driver NAME | --driver module:attr] [--json-output]
```

- `REF` is what a config names: a named environment, a `python://` class or an `ors+http://` URL for `env check`; a
  harness name, a `.toml` spec path or `module:function` for `harness check`.
- They run the suites of section 5 in process, on a CPU, with no model and no Tinker, and print one line per check
  (`ok`; `FAIL` and why; `skip` and why). The exit status is 0 when every check passes, 1 when one fails, and 78 when
  the reference does not resolve.
- `env check` of an `ors+http://` URL runs the checks a remote session supports (no attributes, no fork hooks);
  `--http` serves an in-process class over HTTP as well and compares the outcomes. `harness check` of a TOML spec
  checks the spec itself (its templates, placeholders, built-in tools and MCP settings): running a command-line
  agent needs it installed in a sandbox, which is what RFC 0008's `doctor harness` is for.
- `platoon doctor` (section 9) checks the installation, and RFC 0008's `doctor` subcommands check live
  infrastructure; these two commands check one component against platoon's contract.

### 7. `platoon new`

```text
platoon new extension NAME [--owner OWNER] [--dir DIR]       a package with one component of each kind
platoon new env|harness|program|reward-processor NAME [--owner OWNER] [--dir DIR]
```

It writes:

- `pyproject.toml`, with an entry point for each component and a requirement on platoon from
  `suggested_requirement()`;
- a package with an empty `__init__.py`, so that loading one component imports nothing of the others, and one module
  per component, with an options schema (RFC 0016) and docstrings that say what to change;
- for an environment, a module that imports only openreward and pydantic, so that any ORS server can serve it without
  platoon, and its named environment in a module of its own;
- tests on `platoon.testing`: an episode on a CPU with a scripted policy, and the conformance suite of each component;
- a `README.md`, which for an environment is an environment card: what it is, its tasks and splits, its tools, its
  reward, its data and the data's license, the compute it needs, the harnesses tested on it, its fork modes, and how
  to cite it;
- a CI workflow that runs the linter, the tests and `platoon extensions check --load --imports`.

The templates are Jinja2, already a core dependency, and ship in the wheel, so `platoon new` works from an installed
platoon and always matches it. It refuses a directory that is not empty. Core's CI generates each template into a
scratch directory, installs it beside the built platoon wheel, and runs its tests and `extensions check`, as
`tests/integration/test_installed_extensions.py` installs extensions into a scratch environment today.
`examples/extension-package/` stays the guided walkthrough
([Publish an extension package](../extending/publish-a-package.md)), with the layout the templates generate.

### 8. The generated extension reference

`docs/reference/extensions.md` is written by `scripts/gen_reference.py` with the other reference pages, and kept
current by the same staleness test. It documents the extension surface from the code:

- the entry-point groups, what an entry point of each may load to, and the naming rule (`GROUPS`, `NAME_PATTERN` and
  `CORE_NAMES` in `platoon/extensions.py`);
- core's components: each built-in harness, program and named environment, with its version, its description and
  its options, read from its schema (RFC 0016);
- the public modules and their stability (section 1);
- the conformance suites, with a description of each check;
- the environment variables that extensions and their users read or set (`PLATOON_IGNORE_EXTENSION_REQUIREMENTS`,
  the env-server pool's variables).

`platoon extensions show --dist DIST --markdown` writes the components part of the page for any installed
distribution, so that an extension generates its own reference, as the templates' README does.

### 9. `platoon extensions show` and `platoon doctor`

```text
platoon extensions show NAME [--kind KIND] [--load] [--json-output]
platoon extensions show --dist DIST [--json-output] [--markdown]
platoon doctor [--json-output]
```

`extensions show NAME` exists (RFC 0013, section 8): one component as `extensions list` lists it, with its entry
point, its distribution's requirement on platoon and where configs use it, from metadata only, and with `--load` the
versions it registers. With `--load` it now also shows what the component declares: the first paragraph of its
docstring, its options (name, type, default, description) and its trace filters and telemetry.

`extensions show --dist DIST` describes one distribution: its version; how it is installed (from PEP 610: an
editable checkout, or a URL); its owner; its requirements on platoon and whether this platoon meets them; and each
of its components as `show NAME --load` shows it. Like `--load`, it imports the distribution's entry points: a
component that fails to load shows why, and the command then exits 78. `platoon extensions show --dist platoon`
describes core's built-ins. A distribution takes a flag of its own because the two kinds of name collide: a
distribution may be named like a component it provides (a distribution `acme` with a program `acme`).

`doctor` checks the installation, reading metadata and each editable checkout's `pyproject.toml`, and imports no
extension:

| Finding | Severity |
|---|---|
| an editable install whose checkout declares entry points or a version that its installed metadata lacks (it prints the command that reinstalls it) | error |
| entry points in a retired group (`platoon.plugins`) | warning |
| one `name@version` provided by two distributions | error |
| an extension whose requirement on platoon excludes the running platoon | error |
| a platoon source checkout on `PYTHONPATH` that shadows an installed platoon of another version | warning |
| a distribution that installs into the `platoon` package (until RFC 0013's stage 2 retires that namespace) | warning |
| an extension that needs an extra of platoon whose packages are missing (`platoon[ors]` without openreward) | error |

It exits 78 when it finds an error, as `extensions check` does. RFC 0008's `doctor backend`, `sandbox`, `env`,
`harness` and `cluster` stay that RFC's: they test live infrastructure, and `doctor` with no subcommand is this
check.

### 10. What changes for users

- **Changed:** names move to public modules, and their old paths warn until RFC 0016's deprecations end;
  `platoon.extensions` exports fewer names; `extensions check` suggests a bound below the next minor release before
  1.0.
- **New:** the public modules of section 1; `pt.TrajectoryStep` and the agent-contract helpers;
  `EpisodeContext.grade_session()`, `Outcome.with_reason()` and `pt.ToolSpec`'s OpenAI tool schema;
  `platoon.testing`'s conformance suites, drivers, toy environments and golden records; `platoon env check`,
  `platoon harness check`, `platoon extensions show --dist` and what `show --load` adds, `platoon extensions check
  --imports`, `platoon doctor` and `platoon new`; `docs/reference/extensions.md`; `schema = 1` in TOML specs.
- **For contributors:** a change to a public signature regenerates the snapshot, as it does today for three modules,
  and a removal or a change of a stable one needs a breaking pull request title; the docs, like the examples and the
  first-party extensions today, may import public names only.
- **Unchanged:** every import that works today keeps working; private modules stay importable.

### 11. Not in this RFC

- **More provenance**, the last item of RFC 0013's section 12: each environment server's version and each dataset's
  revision in `lock.json`, the distribution behind a renderer or tool-call parser that a package provides, and
  canonical `name@version` program identities in trajectory records. It changes record formats, so it waits for an
  RFC of its own.
- **Release metadata** (the version, `__version__`, the license field and `py.typed`), which the same section lists
  with the public surface: the PyPI-readiness work for the first release, 0.2.0, sets it.

## The hard parts

- **The first list.** Promoting a name is easy; taking it back is a breaking change. The table starts small: what the
  docs, the examples, the first-party extensions and the research extension import, given a stable home, and nothing
  else.
- **Snapshot noise.** Lines come from signatures as written, so a reformatted annotation or a renamed type alias
  changes lines without changing behavior. The reviewer decides, and only lines of stable modules need a breaking
  title.
- **Docs as consumers.** Prose names modules in inline code, and some pages describe internals on purpose. The check
  reads inline code too, and exempts the pages that document internals.
- **Harness drivers.** A suite cannot know an arbitrary harness's format. The `native-tools` driver covers harnesses
  that act through tool calls; others write a driver, and a check that needs a behavior the driver cannot produce is
  skipped, with the reason.
- **Environments behind servers.** An `ors+http://` environment cannot run the in-process checks (attributes, fork
  hooks), and comparing outcomes over HTTP needs an env server that the suite starts in process.
- **Templates that rot.** Generated code goes stale silently when the API it uses changes; generating, installing and
  testing every template in CI is what prevents that.

## Alternatives

- **Underscores for private modules**, as SciPy did when it made modules such as `scipy.stats.morestats` private and
  kept deprecated shims (knowledge, H). It is self-describing, but it renames more than a hundred modules and every
  import of them in core and in the tests, and it breaks today's extensions before their replacements exist.
- **A marker in each module** (`__stability__ = "stable"`). There is no single list for tools and docs, and finding
  the public modules means importing every module.
- **Hand-written downstream compatibility tests** (tinker-cookbook's `tests/downstream_compat/`). They are precise
  for the names someone wrote a test for, and silent about the rest; the snapshot covers every public name, and the
  conformance suites cover behavior.
- **Conformance checks only on the command line.** An extension's own CI should run them in its test suite, where it
  can provide fixtures: a solution, a driver, an image.
- **A template repository** (cookiecutter or copier). It is one more tool and repository to keep in step with
  platoon; templates in the wheel always match the installed platoon.
- **Doing nothing.** The docs keep teaching private imports, and the first refactor of those modules breaks
  extensions with no warning.

## Rollout plan

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| M1 the declaration | `platoon/_api.py`; the new public modules, the moved names with their deprecated old paths, trimmed `__all__`s; `TrajectoryStep` and the agent-contract helpers in `pt`; the version policy in CONTRIBUTING and the warning about ranges wider than one minor release (the PyPI-readiness work already adds `py.typed` and moves the example package, the plugins and the publishing guide to one minor release) | every listed module imports and every `__all__` name resolves; `PRIVATE_IMPORTS` in `tests/test_import_boundaries.py` is empty; the docs import only public names (checked once by hand until M2) | |
| M2 the checks | the snapshot over every public module, with stabilities, and the breaking-change step in CI; the boundary test over the docs; the run-time core check and the cheap-import check; `extensions check --imports`; the API reference for every public module | removing a parameter of a public function fails CI until the snapshot is regenerated and the title says `!`; a guide that imports a private module fails the boundary check | |
| M3 conformance | `platoon.testing.conformance` with the five suites, the drivers and the toy environments; `platoon.testing.golden`; core runs them on its own components | core's built-in harnesses and programs, the example environments and the `process` provider pass; a toy broken on purpose fails the check that names its defect; the CodeAct goldens run through `platoon.testing.golden` | |
| M4 commands | `platoon env check`, `platoon harness check`, `extensions show --dist` and the details of `show --load`, and `doctor` | the CLI tests check each command's output on the example extension; `doctor` finds a stale editable install in a scratch environment | |
| M5 templates and the reference | `platoon new` and the template job in CI; `docs/reference/extensions.md` | every generated template installs and passes its own tests in CI; the reference page is current | |

## Open questions

1. **The list of public modules.** Approve the promotions and their names (section 1's table), in particular
   `platoon.programs` and `platoon.trajectory` as public modules and the agent-contract helpers in `pt`. Decide before
   M1.
2. **The version policy before 1.0.** May a minor release break a stable name after one minor release of warnings, as
   proposed, or should stable names freeze until 1.0? Decide before M1.
3. **Enforcing breaking titles.** Should CI require a `!` title when the snapshot loses or changes a stable line, as
   proposed, or should that be left to review? Decide before M2.
4. **The docs under the boundary check.** The proposal lets the user docs teach public names only, so the
   credit-assignment tutorial's `group_baselines` comes from `platoon.experimental`. Agreed? Decide before M2.
5. **Drivers for third-party harnesses.** Pass them with `--driver module:attr`, as proposed, or let an extension
   declare its harness's driver so that `platoon harness check NAME` finds it, at the cost of one more entry-point
   group? Decide before M4.
6. **Template defaults.** A `src/` layout, a GitHub Actions workflow, and which license, if any? Decide before M5.
7. **`doctor`'s exit status.** 78, as `extensions check` exits (proposed), or RFC 0008's 3 for a failed
   precondition, which no command uses yet? Decide before M4.
8. **One distribution's view.** `extensions show --dist DIST` (proposed), beside `show NAME` for one component, or
   `extensions list --dist DIST` for the rows and `show NAME --load` for each component's details? Decide before M4.
