# Wrap a command-line agent

**Goal.** Train or evaluate an existing command-line agent with platoon, without changing the agent: describe it
in a TOML *harness spec* that says how to install it, launch it against platoon's model gateway, give it the
environment's tools over MCP, turn its own tools off where they would act on the wrong machine, and read its
answer. You will wrap `tiny-agent`, a small agent that stands in for yours, and run it on number search. The
finished files are in [`examples/cli-harness/`](../../examples/cli-harness/).

For every rollout, platoon starts the model gateway, opens an *agent* with its own key, fills in the spec's
templates and runs the command, in the task's sandbox or as a local process. The agent's model calls go to the
gateway, which samples the policy being trained and records every call on that agent; that record is what trains,
so the agent needs no changes for RL. When the command exits, platoon maps its exit code to a status and the
environment grades the episode. [The architecture guide](../architecture.md) explains the layers.

## Prerequisites

- Python 3.12 or later and a checkout of platoon, installed from the repository root with the `tinker` and `ors`
  extras: `uv sync --extra tinker --extra ors`.
- Number search, the example environment, ships with platoon: the `ors` extra runs it.
- `python3` on the `PATH`, which `tiny-agent` runs with.
- A [Tinker](https://thinkingmachines.ai/tinker/) API key in `TINKER_API_KEY` to sample a real model in step 4.
  The example's test runs without one.

Run the commands below from the repository root.

## What the agent must support

| The agent needs | Because |
|---|---|
| An OpenAI-compatible base URL and API key it can be pointed at | Its model calls must reach platoon's gateway. The gateway speaks OpenAI Chat Completions, and experimentally OpenAI Responses and Anthropic Messages (the spec's `dialect`). |
| A headless mode: the task as an argument or a file, no prompts, exit when done | Platoon runs it unattended, once per rollout. |
| A turn or step limit | Platoon passes the step budget as `{max_steps}`. Without one, only the timeout stops the agent. |
| Optional: an MCP client, configured by a flag or a config file | The environment's tools reach the agent over MCP. |
| Optional: a way to turn its own tools off | Outside a task sandbox, a shell tool would act on the training host. |
| Optional: its final answer in a file or on stdout | Platoon records it as the trajectory's final message. |

`tiny-agent` ([`tiny_agent.py`](../../examples/cli-harness/tiny_agent.py), standard library only) has all of
them:

```text
python3 tiny_agent.py --model MODEL --task-file PATH [--max-turns N] [--mcp-config PATH] [--no-shell]
```

| Interface | |
|---|---|
| `OPENAI_BASE_URL`, `OPENAI_API_KEY` | the model endpoint, read from the environment as the OpenAI SDK reads them |
| `--max-turns N` | stop after `N` model calls (default 30) |
| `--mcp-config PATH` | a JSON file in the common `{"mcpServers": {"<name>": {"url": ..., "headers": {...}}}}` format; `${VAR}` in a header is read from the environment |
| `--no-shell` | turn the built-in `shell` tool off |
| stdout | `FINAL ANSWER: <text>` when it is done |
| exit codes | 0 answered, 2 model endpoint error, 3 turn limit reached, 4 MCP server error |

## Step 1: write the spec

[`tiny-agent.toml`](../../examples/cli-harness/tiny-agent.toml):

```toml
# tiny-agent as a platoon harness: how to install it, launch it against the gateway, turn its built-in tool
# off, hand it the environment's tools over MCP, and read its final answer.
#
#   platoon harness show examples/cli-harness/tiny-agent.toml

name = "example-tiny-agent"
version = "0.1.0"
dialect = "openai_chat"   # the API it speaks: openai_chat, openai_responses or anthropic_messages
timeout_s = 600           # wall clock for one run; the rollout's own timeout applies too, the earlier wins
max_steps = 20            # {max_steps} when the run sets no step budget; the run's max_steps wins

[install]
# Shell commands run where the agent runs (host or task sandbox) before each launch. A packaged agent would
# install itself here (see the mini-swe-agent spec); tiny-agent only needs python3.
commands = ["python3 --version"]

[launch]
# {state_dir} is a per-agent scratch directory; the agent runs in {workdir}.
argv = [
    "python3", "{state_dir}/tiny_agent.py",
    "--model", "{model}",
    "--task-file", "{state_dir}/task.md",
    "--max-turns", "{max_steps}",
]
files = { "task.md" = "{instruction}" }        # written into {state_dir} before the launch
assets = { "tiny_agent.py" = "tiny_agent.py" }  # copied from next to this spec into {state_dir}

[launch.env]
# The agent key may only appear here and in [mcp.env], never in argv or files.
OPENAI_BASE_URL = "{base_url}/v1"
OPENAI_API_KEY = "{api_key}"
PLATOON_API_KEY = "{api_key}"   # the MCP config below reads it as ${PLATOON_API_KEY}

[builtins]
tools = ["shell"]

[builtins.disable]
# Appended to argv when the built-ins are off: by default, whenever the agent runs outside a task sandbox.
args = ["--no-shell"]

[mcp]
# The environment's tools over MCP. tiny-agent reads MCP servers from a config file, so platoon writes one
# into {state_dir} and passes its path. {mcp_url} has no key in it; the client sends the agent key as a
# Bearer token.
config_template = '{"mcpServers": {"env": {"url": "{mcp_url}", "headers": {"Authorization": "Bearer ${PLATOON_API_KEY}"}}}}'
config_file = "mcp.json"
args = ["--mcp-config", "{mcp_config_path}"]

[final_message]
stdout_regex = "^FINAL ANSWER: (.*)$"   # the last match; group 1 when the regex has a group

[exit_codes]
# 0 is completed and any other code an error, unless mapped here.
3 = "completed"   # turn limit: the policy ran out of turns, a scored outcome rather than a crash
```

### Top-level keys

| Key | Default | Meaning |
|---|---|---|
| `name`, `version` | required | The harness is `name@version`. Configs may omit the version when only one is registered, or give a prefix: `mini-swe-agent@2` matches `2.4.6`. |
| `dialect` | `openai_chat` | The API the agent speaks: `openai_chat`, `openai_responses` or `anthropic_messages`. |
| `timeout_s` | none | Wall-clock limit of one run. The rollout's `timeout` also applies; the earlier wins. |
| `max_steps` | none | The value of `{max_steps}` when the run sets no step budget. Set it whenever a template uses `{max_steps}`: with neither, the launch fails. |

### Placeholders

String values in `[launch]`, `[builtins.disable]`, `[mcp]` and `[final_message]` are templates. Any other
`{lowercase_name}` is an error when the spec loads; other text, such as shell `${VAR}` or JSON braces, is kept as
written.

| Placeholder | Value | Allowed in |
|---|---|---|
| `{instruction}` | the task's instruction | everywhere |
| `{base_url}` | the gateway, `http://127.0.0.1:<port>`, without `/v1` (OpenAI clients add it; Anthropic clients take it as is); in a task sandbox without a network, the sandbox's route to it | everywhere |
| `{api_key}` | the agent's key: it authenticates the agent and attributes its calls | `[launch.env]` and `[mcp.env]` only, so it never shows in a process listing or a file |
| `{model}` | `policy`, the model name the agent should send; the gateway answers with the policy whatever the name | everywhere |
| `{workdir}` | the directory the agent runs in: the sandbox's working directory, or `work/` in the state directory when run locally | everywhere |
| `{state_dir}` | a per-agent scratch directory for files, assets and logs | everywhere |
| `{max_steps}` | the step budget: the task's or the rollout config's `max_steps`, else the spec's | everywhere |
| `{toolchain}` | `[install] toolchain`: the path as written, or for a named toolchain its build path (in a task sandbox) or its local copy | everywhere, when `toolchain` is set |
| `{task_mcp_servers}` | the MCP servers the task itself offers its agents, as JSON: a list of objects with each one's `name`, `transport` and `url` as the task's sandbox reaches it, or, for `stdio`, `command` and `args`; `[]` for a task without any and for a harness that runs locally, which reaches only the servers that the env's tools serve ([Task MCP servers](../how-to/sandboxes-and-harbor.md#task-mcp-servers)) | everywhere |
| `{mcp_url}` | the gateway's MCP endpoint, without the key in it | `[mcp]` only |
| `{mcp_config_path}` | the path of the written `config_file` | `[mcp]` only, when `config_template` is set |
| `{options.NAME}` | the value of the spec's option `NAME` (`[options]`): a boolean as `true` or `false`, a number in decimal, a string as it is | `[launch]` argv, env and files, `[builtins.disable]` and `[mcp]` |
| `{options}` | every option, defaults included, as one JSON object, for a program that reads them whole (`files = {"options.json" = "{options}"}`) | the same |

### Tables

| Table | Keys | Meaning |
|---|---|---|
| `[install]` | `commands`, `timeout_s` (1200), `toolchain` | Shell commands run where the agent runs, before its launch. Each rollout installs again, so make them cheap to repeat: check first, as `mini-swe-agent`'s do. A failure is a setup error, not a policy failure. `toolchain` is an absolute path that must exist where the agent runs (the Harbor service's `mounts` can put it into a sandbox), or the name of a toolchain under the toolchain root (`toolchains` in the site profile's cache, else `~/.cache/platoon/toolchains`), which platoon copies to node-local disk and mounts read-only into the task sandbox at its build path ([Harness toolchains](../how-to/sandboxes-and-harbor.md#harness-toolchains)). |
| `[install.recipe]` | `python`, `lock`, `cache`, `generate` | How `platoon toolchain build` builds the toolchain: a uv-managed CPython (`python = "3.12"`), the hash-locked requirements file `lock` (`uv pip compile --generate-hashes` output, next to the spec), caches filled at build time (`cache = ["tiktoken"]`), and files written by scripts next to the spec, run with the toolchain's own interpreter (`generate = { "share/x.json" = "make_x.py" }`). See [OpenHands in task sandboxes](../how-to/sandboxes-and-harbor.md#openhands-in-task-sandboxes). |
| `[resources]` | `memory_mb` | The memory the harness's own processes need. It is added to a task sandbox's memory limit, when the task sets one, so that the harness does not count against the task's limit. |
| `[budget]` | `enforce` (`harness`) | Who ends the agent at its step budget: `harness`, the harness itself from `{max_steps}`; or `gateway`, for a harness without a step limit: the gateway answers its policy calls past the budget with a final message, unsampled and unrecorded, and the episode is graded with reason `budget`. |
| `[launch]` | `argv` (required), `env`, `files`, `assets` | The command. `files` maps paths to content written before the launch; `assets` maps paths to files next to the spec, copied before the launch. Relative paths are in the state directory. |
| `[builtins]` | `tools`, `disable.args`, `disable.env` | The agent's own tools, and the arguments or environment that turn them off. Without `disable`, they cannot be turned off. |
| `[mcp]` | `enabled` (true), `args`, `env`, `config_template`, `config_file` (`mcp.json`) | Applied when the agent gets the environment's tools over MCP: `args` are appended to `argv` after any `disable.args`, `env` is added, and `config_template` is written to `config_file` in the state directory. |
| `[final_message]` | `file` and optional `json_key`, or `stdout_regex` | Where the answer is: a file (with `json_key`, a dotted key into its JSON), or the last match of a regex on the output (group 1 if the regex has a group). |
| `[exit_codes]` | `<code> = "completed"` or `"error"` | Exit 0 is `completed`, any other code `error`, unless mapped here. |
| `[options.NAME]` | `type` (`bool`, `int`, `float` or `str`), `default` (required), `description`, `allowed`, `min`, `max` | One option a config sets in `harness_options`, which the launch renders as `{options.NAME}`. `allowed` lists the values an `int` or `str` option may take; `min` and `max` bound an `int` or `float` one. A default that breaks its own rules, or a placeholder that names an undeclared option, fails when the spec loads; a value a config gives that breaks them stops the run before any rollout (exit 78). Options never render into `[install]`: they change how the agent runs, not what is installed, so one toolchain serves every setting. They are recorded, so they carry no secrets. |

The agent's key never needs to be in a file: `tiny-agent.toml` puts it in the environment as `PLATOON_API_KEY`,
and the MCP config refers to it as `${PLATOON_API_KEY}`, which the agent expands. An agent that takes the MCP
endpoint as a flag gets `--mcp-url {mcp_url}` in `[mcp] args` instead, as `platoon-bash` does; it sends the key
as a `Bearer` token.

## Step 2: check the spec

```bash
uv run platoon harness show examples/cli-harness/tiny-agent.toml
```

Expected output:

```text
example-tiny-agent@0.1.0  (spec)
source: examples/cli-harness/tiny-agent.toml
dialect: openai_chat
timeout_s: 600.0
max_steps: 20
install: ['python3 --version']
launch: python3 {state_dir}/tiny_agent.py --model {model} --task-file {state_dir}/task.md --max-turns {max_steps}
launch env: ['OPENAI_API_KEY', 'OPENAI_BASE_URL', 'PLATOON_API_KEY']
built-in tools: ['shell']  (can be disabled)
final message: stdout /^FINAL ANSWER: (.*)$/
exit codes: {3: 'completed'}
mcp: McpSpec(enabled=True, args=('--mcp-config', '{mcp_config_path}'), env={}, config_template='{"mcpServers": {"env": {"url": "{mcp_url}", "headers": {"Authorization": "Bearer ${PLATOON_API_KEY}"}}}}', config_file='mcp.json')
```

The spec is checked when it loads: unknown keys, unknown placeholders and misplaced secrets are errors. For
example, `{api_key}` in `argv` fails with `{api_key} may only be rendered into [launch.env]`, and the command
exits with code 78.

For a run with the built-ins off and the MCP endpoint served, the launch renders to:

```text
python3 <state_dir>/tiny_agent.py --model policy --task-file <state_dir>/task.md --max-turns 20 --no-shell --mcp-config <state_dir>/mcp.json
```

## Step 3: name the spec in a config

[`config.yaml`](../../examples/cli-harness/config.yaml) evaluates the wrapped agent on number search:

```yaml
inference:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  output_dir: ./runs/cli-harness
  workflow:
    num_concurrent_workers: 8
    rollout_config:
      max_steps: 12          # rendered as {max_steps}: tiny-agent's --max-turns
      timeout: 300           # seconds; the spec's timeout_s also applies, the earlier wins
      verbose: false
      inference_params: {max_completion_tokens: 512}

environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: tiny-agent.toml   # a spec path, relative to this file
    eval_dataset_kwargs: {split_name: val, limit: 20}
```

`harness` takes a path to a `.toml` file, relative to the config file that names it (an override's is relative to
the working directory), or the name of a registered spec. To register yours, publish it from a package through the
`platoon.harnesses` entry-point group: an entry point named after the spec, which names a module attribute that
holds the path of the spec file. A spec takes the `harness_options` its `[options]` table declares, and none without
one, so give a setting that configs should choose an option instead of a copy of the spec per setting:

```toml
[options.max_turns_per_task]
type = "int"
default = 20
min = 1
description = "Turns tiny-agent may take"

[launch]
argv = ["python3", "{state_dir}/tiny_agent.py", "--max-turns", "{options.max_turns_per_task}"]
```

```yaml
environments:
  - env: platoon-number-search@1
    harness: my-agent.toml
    harness_options: {max_turns_per_task: 8}
    eval_overrides:                      # evaluation rollouts only: every split but train
      harness_options: {max_turns_per_task: 30}
```

`platoon harness show my-agent.toml` lists the options, and `lock.json` records the ones each run used, defaults
included. On the command line, `environments.0.harness_options.max_turns_per_task=12` sets one like any key.

The rollout config's `inference_params` set the sampling of every call the agent makes, unless the agent sends
its own values. An agent that sends no `max_tokens` gets `max_completion_tokens`, or 4096 tokens without it.

## Step 4: run one task

```bash
uv run platoon rollout examples/cli-harness/config.yaml --task val:0
```

With a policy that plays well, the summary reads:

```text
rollout val-0-ba08d41bf3c0  task val:0
agents 1  steps 9  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy e2a9c546  steps 9  reward 1.0  outcome scored reward=1.0
```

Number search offers no sandbox, so the entry's `placement`, `auto` unless a config says otherwise, ran `tiny-agent`
locally, and its `builtins`, `default`, turned the agent's shell off: the model saw only the environment's `guess`
tool, over MCP. In `trajectory_collection.json` in the run directory, the root trajectory has one step per model call,
`finish_message` holds the `FINAL ANSWER` line, `misc.tool_calls` the MCP tool calls, and `misc.harness` the run
itself:

```json
{"spec": "example-tiny-agent@0.1.0", "placement": "local", "exit_code": 0, "timed_out": false, "logs_path": "<run dir>/rollouts/harness/<agent id>.log"}
```

The log holds the install output and everything the agent printed:

```text
[platoon] install example-tiny-agent@0.1.0: exit 0
<what python3 --version printed>

$ guess {"number": 542}
Too high, try again.
$ guess {"number": 330}
Too low, try again.
...
```

`platoon eval examples/cli-harness/config.yaml` evaluates on 20 validation tasks, and the same `environments`
entry trains in a training config, as in [the function-harness tutorial](function-harness.md#step-5-evaluate-and-train).

## Placement and built-in tools

`placement` on the environment entry says where the agent runs:

| `placement` | Where |
|---|---|
| `auto` (default) | in the task sandbox when the environment session has one (Harbor tasks), else locally |
| `task_sandbox` | in the task sandbox, through the sandbox's `exec`, in the sandbox's working directory, so the agent's own tools act on the task. Its state directory is under `/tmp/platoon-harness/` in the sandbox. An environment without sandboxes is a setup error. |
| `local` | as a subprocess of the rollout process, in its own process group and a fresh state directory that is removed afterwards. It gets a small allowlisted environment (`PATH`, `HOME`, locale, temp, proxy and CA settings, plus the spec's `env`), never the rollout process's secrets. |

The gateway listens on `127.0.0.1`, so a task sandbox must share the host's network, as platoon's sandboxes do by
default. On a timeout, a cancellation or after the agent exits, platoon terminates the whole process tree it
started.

`builtins` says what happens to the agent's own tools:

| `builtins` | Effect |
|---|---|
| `default` | keep them in a task sandbox; turn them off elsewhere, with `[builtins.disable]`. A spec without a `disable` knob keeps them, and platoon logs a warning: they then act on the rollout host. |
| `keep` | keep them everywhere |
| `disable` | turn them off everywhere; an error, before any rollout, for a spec without a `disable` knob |

A spec runs as one agent with no episode program, and its own tools act where it runs. `platoon train`, `eval` and
`rollout` therefore refuse an environment entry whose rollouts could never train, with exit code 78 before they
list any task:

- a spec on `service: table`, whose sessions have no grader: only a program sets a table's outcome, so every
  rollout would be `invalid` (`self_grade`);
- a spec with a `program` other than `single`, or with `agents` settings, which apply only to function harnesses
  (`agents.max_depth: 0`, which says that no agent spawns, is accepted);
- a spec without `[mcp]` on an environment without task sandboxes (`service: table`, `python://` and `ors+http://`
  ORS services), or placed `local` on Harbor: neither its own tools nor any env tool reach the environment, so every
  rollout would score 0 (`no_submit`), or on Harbor be graded on a sandbox it never acted in.

An entry whose `rollout` or `eval_rollout` names a rollout function of its own is not checked: that function decides
how its harness runs.

## Exit codes and outcomes

| How the run ended | Outcome | Trains |
|---|---|---|
| exit 0, or a code mapped to `"completed"` | the environment grades the session | yes |
| any other exit code | `error` (`harness_crash`) | no |
| any other exit code, after the gateway refused a prompt that overflowed the context window | graded, with `reason: context_overflow` (or `error` with `on_context_overflow: exclude`) | yes |
| the spec's `timeout_s` or the rollout's `timeout` passed | `timeout` | no |
| the install failed, the toolchain is missing, or the command could not start | `error` (`harness_setup`) | no |
| the task sandbox has no network, so the harness could not reach the gateway; nothing starts | `error` (`harness_setup`) | no |
| the launch renders `{max_steps}`, but the rollout, the task and the spec set no step budget | `error` (`harness_config`) | no |
| with `[budget] enforce = "gateway"`: the gateway stopped the agent at its step budget, and the command exited with a `completed` code | the environment grades the session, with `reason: budget` | yes |

Map the exit codes that mean "the policy ran out of turns" to `"completed"`, as `tiny-agent.toml` does with 3:
running out of turns is a policy failure, which should be graded, not an infrastructure error, which is dropped.
Leave real failures, such as a model endpoint or MCP error, unmapped.

The environment's tools tell an MCP client when the episode is over: the result's `_meta.platoon.finished` is
true. An agent that stops there, as `tiny-agent` and `platoon-bash` do, saves a model call; one that does not
simply goes on until its next answer.

## Reference specs

Two built-in specs show the common cases. `platoon harness show <name>` prints either, and both live in
[`platoon/harness/builtin/specs/`](../../platoon/harness/builtin/specs/).

**`platoon-bash`** ships with platoon: a standard-library agent with a `bash` tool, plus the environment's tools
when the agent gets them over MCP. It takes the MCP endpoint as a flag, writes its answer to a file and maps its
turn limit (exit 3) to `completed`:

```toml
# platoon-bash: the reference harness shipped in platoon/harness/builtin/platoon_bash.py.
# A bash tool, executed with the harness's working directory as cwd, plus the env's tools over MCP when the
# agent serves them. Needs only python3.
name = "platoon-bash"
version = "1"
dialect = "openai_chat"
timeout_s = 1800
max_steps = 50  # used when the run sets no step budget; the run's max_steps wins

[launch]
argv = [
    "python3", "{state_dir}/platoon_bash.py",
    "--base-url", "{base_url}/v1",
    "--model", "{model}",
    "--instruction-file", "{state_dir}/instruction.md",
    "--final-message-file", "{state_dir}/final_message.txt",
    "--max-turns", "{max_steps}",
]
env = { PLATOON_API_KEY = "{api_key}" }
files = { "instruction.md" = "{instruction}" }
assets = { "platoon_bash.py" = "../platoon_bash.py" }

[builtins]
tools = ["bash"]

[mcp]
enabled = true
args = ["--mcp-url", "{mcp_url}"]  # the Bearer token is PLATOON_API_KEY

[final_message]
file = "{state_dir}/final_message.txt"

[exit_codes]
3 = "completed"  # turn limit: the policy ran out of turns, which is a scored outcome
```

**`mini-swe-agent@2.4.6`** wraps a third-party agent. `[install]` installs it with `uv` unless it is already there,
the launch sets litellm's `OPENAI_*` variables to the gateway, `-c agent.step_limit={max_steps}` passes the
budget, and the answer is the `info.submission` key of the trajectory file it writes (`[final_message] file`
with `json_key`). Its only tool is bash and it has no MCP client, so it belongs in a task sandbox, such as a
Harbor task's.

## Next steps

- [Built-in harnesses](builtin-harnesses.md): when to use `platoon-bash`, `platoon-react` or `platoon-codeact`.
- [Write a function harness](function-harness.md) when the agent is Python code you control.
- [Publish an extension package](../extending/publish-a-package.md): ship specs and harnesses in a package
  through the `platoon.harnesses` entry points.
- [The architecture guide](../architecture.md): the gateway, the MCP tool plane and Harbor task sandboxes.
- The spec format in full: [`platoon/harness/spec.py`](../../platoon/harness/spec.py); the runner:
  [`platoon/harness/runner.py`](../../platoon/harness/runner.py).
