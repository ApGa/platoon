# Built-in harnesses

**Goal.** Pick the built-in harness that fits an environment, and configure it. Platoon ships three harnesses of
its own, `platoon-react`, `platoon-codeact` and `platoon-bash`, specs for two third-party agents, `mini-swe-agent`
and `openhands` (the OpenHands SDK agent, configured by options), and two harnesses that check a dataset rather than
a policy, `platoon-oracle` and `platoon-noop`. Each is a name you put in an environment's `harness`.

## Prerequisites

- Platoon installed from the repository root: `uv sync --extra tinker --extra ors`.
- An environment config. [Write a function harness](function-harness.md#step-2-name-the-harness-in-a-config)
  shows one; [`examples/number-search`](../../examples/number-search/README.md) has ready configs for
  `platoon-codeact` and `platoon-react` on number search, the environment that ships with platoon.

## At a glance

| Harness | Kind | How the model acts | Its tools | Where it runs | Use it when |
|---|---|---|---|---|---|
| `platoon-react` | function | native tool calls | the environment's tools | in the rollout process | the model calls tools natively and the environment has a handful of them; the simplest baseline |
| `platoon-codeact` | function, runs model code | writes Python cells that call the tools as functions | the environment's tools, `finish`, `launch_subagent` | cells run in an IPython shell in the rollout process: not a sandbox | acting in code pays off (loops, arithmetic, many calls per step), or agents should delegate to subagents |
| `platoon-bash` | TOML spec | native tool calls | `bash`, plus the environment's tools over MCP | a process in the task sandbox, or locally | the task is files in a sandbox (Harbor tasks); smoke tests; a template for your own spec |
| `mini-swe-agent@2.4.6` | TOML spec, third party | `bash` commands | `bash` only | the task sandbox | software-engineering tasks in Harbor sandboxes |
| `openhands@1.54.0.post1` | TOML spec, third party: the OpenHands SDK agent with platoon's thin runner | native tool calls | terminal, file editor, task tracker | the task sandbox, from a toolchain built once per site | training or evaluating an OpenHands agent on Harbor tasks; its options choose replay, timeout, retries and a condenser ([below](#openhands)) |
| `platoon-oracle`, `platoon-noop` | function, no model calls | runs the task's reference solution; does nothing | none | the rollout process | checking a dataset: a sound task scores 1 with the oracle and 0 with the no-op |

List them:

```bash
uv run platoon harness list
```

```text
harness                 kind      distribution   source
mini-swe-agent@2.4.6    spec      platoon 0.4.0  platoon/harness/builtin/specs/mini-swe-agent.toml
openhands@1.54.0.post1  spec      platoon 0.4.0  platoon/harness/builtin/specs/openhands.toml
platoon-bash@1          spec      platoon 0.4.0  platoon/harness/builtin/specs/platoon-bash.toml
platoon-codeact@1       function  platoon 0.4.0  platoon.harness.builtin.codeact.harness:codeact
platoon-noop@1          function  platoon 0.4.0  platoon.harness.builtin.oracle:noop
platoon-oracle@1        function  platoon 0.4.0  platoon.harness.builtin.oracle:oracle
platoon-react@1         function  platoon 0.4.0  platoon.harness.builtin.react:react
```

Harnesses that installed packages provide appear in the same list, with their package and its version; the list
reads their metadata without importing them. `platoon harness show <name>` prints one
harness's details.

Every harness takes its options from the environment's `harness_options`, and evaluation rollouts take
`eval_overrides.harness_options` merged over them. A TOML spec declares its options in an `[options]` table, which
`platoon harness show <name>` lists, and they are checked when the run starts: an unknown or badly typed option
stops `platoon train`, `eval` and `rollout` (and `platoon config validate`) with exit 78, naming its key, before
any rollout. A spec without options, such as `platoon-bash`, takes none. Function harnesses check their
options in each rollout, where an unknown key fails the rollout with an error instead of being ignored. The rollout
config's `max_steps` is every harness's step budget, and its `inference_params` (`temperature`, `top_p`, `top_k`,
`max_completion_tokens`) the default sampling of every model call.

## `platoon-react`

A minimal tool-calling loop over `ctx.chat` ([`react.py`](../../platoon/harness/builtin/react.py)):

1. The environment's tools go to the model as OpenAI function tools. A terminal tool (the environment's submit or
   finish tool) is kept out of the list.
2. Each turn is one model call. Every tool call in the reply is answered with the tool result's text; a call
   whose arguments are not a JSON object gets `Invalid arguments: ...` and never reaches the environment.
3. The agent ends `completed` when a tool result says the episode is finished, or when the model answers
   without a tool call. That answer goes to the terminal tool when the environment has one.
4. It ends `limit` when its step budget runs out.

| Option | Default | Meaning |
|---|---|---|
| `system_prompt` | none | a system message before the instruction |
| `sampling` | `{}` | Chat Completions parameters for every call, such as `temperature`; they override the rollout's `inference_params` |

```yaml
environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-react
    harness_options:
      system_prompt: >-
        Find the hidden number. Call the guess tool with one integer per turn; it tells you whether
        your guess is too low, too high or correct.
```

The model must support tool calling through the renderer you train it with; for models that do not, use
`platoon-codeact`, which needs only text.

## `platoon-codeact`

The model acts by writing Python ([`codeact/`](../../platoon/harness/builtin/codeact/)). Each step:

1. **Prompt.** The conversation has a system prompt, the task (`Your Goal: <instruction>`, plus the step budget),
   the functions the cells can call, and one assistant turn and one user turn per cell so far.
2. **Sample.** The model writes an optional `<thought>` and a `<python>` cell; sampling stops at `</python>`.
3. **Run.** The cell runs in the agent's IPython shell, where top-level `await` works. What it prints, or its
   error, is the next user turn. That text reaches the model, so it names no path that Python or platoon is
   installed at: tracebacks, warnings and module reprs name an installed file by its path below its import root
   (`File json/decoder.py:338`), and the cell loop's futures and tasks print without source locations
   (`<_GatheringFuture pending>`). Other paths, such as the cell's own files, stay as they are.

Cells can call:

| Name | What it does |
|---|---|
| each environment tool | A synchronous Python function, its signature and docstring taken from the tool's JSON schema. It returns the tool's structured content when the environment sends some, else its text. |
| `finish(message="")` | Hands the message to the environment's terminal tool when it has one, and ends the agent. |
| `await launch_subagent(targets, num_steps, context="")` | Runs a subagent on `targets` with a budget of `num_steps`, and returns its final message. Offered when `subagents` is on. |
| `asyncio` | `gather`, `create_task`, `wait_for` and the other operations that are safe in a cell. |

The agent ends `completed` when `finish` is called or a tool says the episode is finished, `limit` when its budget
runs out, and `timeout` when a cell runs longer than `cell_timeout_s`. When its last cells repeat a pattern, it
ends with the message "Stuck in a loop, terminating early.", which costs no model call.

| Option | Default | Meaning |
|---|---|---|
| `prompt_builder` | built-in | a `CodeActPromptBuilder` subclass, as `package.module:Class` |
| `prompt_builder_options` | `{}` | extra constructor arguments of the builder |
| `prompt_mode` | `sequence_extension` | `sequence_extension`: the conversation grows by one turn pair per cell, so each prompt extends the last and steps merge into few training sequences; `no_sequence_extension`: one user message with the whole history, rebuilt every step |
| `include_reasoning` | `true` | ask for a `<thought>` before each cell, and keep it in the history |
| `sampling` | `{}` | Chat Completions parameters for every call; they override the rollout's `inference_params` |
| `cell_timeout_s` | `300` | seconds a cell may run; a cell that takes longer ends the agent with `timeout` |
| `max_output_chars` | none | cut longer cell output |
| `subagents` | `true` | offer `launch_subagent` |
| `subagent_budget` | `argument` | `argument`: the call's `num_steps` is the subagent's budget; `fixed`: no `num_steps`, and the environment's `agents.child_max_steps` applies |
| `checks` | see below | what cells may not do |

| `checks` key | Default | Meaning |
|---|---|---|
| `unawaited_async_calls` | `true` | refuse a cell that calls `launch_subagent` without `await` |
| `while_loops` | `false` | refuse cells with `while` loops |
| `interactive_input` | `false` | refuse cells that call `input()` |
| `stuck_in_loop_threshold` | `4` | how often the last cells must repeat a pattern to count as stuck (at least 1) |
| `stuck_in_loop_window` | `3` | the longest repeated pattern, in cells, that the loop check looks for; 0 turns the check off |

A refused cell does not run; the model sees why as the cell's error. Syntax errors and empty cells are always
refused.

```yaml
environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: platoon-codeact
    harness_options:
      prompt_builder: platoon.envs.number_search.prompts:NumberSearchPromptBuilder
      include_reasoning: false
      subagents: false
    agents:
      max_depth: 0
```

**Prompts are a subclass.** To change the wording, subclass `CodeActPromptBuilder`
(`from platoon.harness.builtin.codeact import CodeActPromptBuilder`) and name it in `prompt_builder`. Override
`build_system_prompt` for the instructions, `describe_action_space` for how the functions are listed,
`subagent_goal` and `subagent_task_misc` for what `launch_subagent` hands a subagent, and `stop` and
`parse_reply` for another reply format. Number search's `NumberSearchPromptBuilder`
(`platoon.envs.number_search.prompts`) and the TextCraft extension's `TextCraftPromptBuilder`
(`platoon_textcraft.prompts`) are examples.

**Subagents.** `launch_subagent` starts a subagent with `ctx.agents.spawn`; it runs `platoon-codeact` again, with
the parent's options. The environment's `agents` section bounds the tree: `max_depth`, `budget` (`linear`, where
subagents draw on the parent's budget, or `depth_aware`), `child_max_steps` and `child_grading`. [The architecture
guide](../architecture.md) covers subagents and their grading.

**Model code runs on the rollout host.** Cells run in the process that runs the rollout, on a thread of their
own, so code that blocks never stalls the gateway or other agents. That thread is not a sandbox: a cell can do
whatever the process can, including reading files and the network. `platoon-codeact` is registered with
`executes_model_code=True`, which every rollout records in its trajectory's `misc.harness`. Use it where running
model-written Python on that host is acceptable.

## `platoon-bash`

A small agent that ships with platoon, run from a TOML spec
([`platoon-bash.toml`](../../platoon/harness/builtin/specs/platoon-bash.toml)) as a separate process. It needs
only `python3`:

- Its tools are `bash`, which runs a command with `bash -c` in its working directory (120 seconds per command,
  the last 16,000 characters of output), and, when the agent gets them over MCP, the environment's tools.
- It ends when the model answers without a tool call, when a tool result says the episode is finished (that
  result is its final message), or when it reaches its turn limit.
- The step budget is its turn limit: the rollout's `max_steps`, else 50. `timeout_s` is 1800.

| Exit code | Meaning | Outcome |
|---|---|---|
| 0 | finished | graded |
| 2 | model endpoint error | error, not trained; graded when the gateway had refused a prompt that overflowed the context window |
| 3 | turn limit reached | graded (`[exit_codes] 3 = "completed"`) |
| 4 | MCP server error | error, not trained |

Where it runs follows the environment's `placement`: in the task sandbox when the session has one, as in Harbor
tasks, else locally. `bash` cannot be turned off, so `builtins: disable` is an error for this spec, and a local
run keeps `bash` with a warning. Run locally, `bash` executes on the rollout host, in a temporary directory; prefer
task sandboxes.

```yaml
environments:
  - name: harbor
    service: harbor
    service_kwargs: {dataset_dir: /path/to/harbor/tasks, provider: docker}   # or singularity, enroot, process
    harness: platoon-bash
```

The spec has no options of its own. To change, say, the command timeout, copy `platoon-bash.toml`, point its
`platoon_bash.py` asset at the installed `platoon/harness/builtin/platoon_bash.py`, add an `[options]` table that
renders as `"--command-timeout", "{options.command_timeout_s}"` in `argv` (see
[Wrap a command-line agent](cli-harness.md)), and name your copy in `harness`.

## `mini-swe-agent`

[`mini-swe-agent.toml`](../../platoon/harness/builtin/specs/mini-swe-agent.toml) runs
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, headless, against the gateway. Its install
step fetches the agent with `uv` inside each task sandbox that lacks it, so the sandbox needs network access, or
an image with the agent already installed. Its only tool is bash and it has no MCP client, so
it belongs in a task sandbox. Its budget is `agent.step_limit`, set from `max_steps` (default 100); reaching it
exits 0, so the episode is graded. Name it `mini-swe-agent@2.4.6`, or `mini-swe-agent@2`.

## `openhands`

[`openhands.toml`](../../platoon/harness/builtin/specs/openhands.toml) runs one
[OpenHands](https://github.com/OpenHands/software-agent-sdk) SDK agent (openhands-sdk and openhands-tools 1.54.0)
with platoon's thin runner, [`run_openhands.py`](../../platoon/harness/builtin/openhands/run_openhands.py), inside
the task sandbox. It is the agent Harbor's `openhands-sdk` adapter runs: the terminal, file editor and task tracker
tools plus the SDK's finish and think, no skills and no delegation. It runs from a read-only toolchain that
`platoon prepare CONFIG` (or `platoon toolchain build openhands --root DIR`) builds once per site; a run never
builds it, and refuses to start while it is missing, with the `platoon prepare` command for its config
([how-to](../how-to/sandboxes-and-harbor.md#openhands-in-task-sandboxes)).
Its step budget is `max_steps` (default 30), one model call each (with the condenser, each condensation takes one
too); reaching it, or the SDK finding the agent stuck, ends a graded episode.

| Option | Type | Default | Meaning |
|---|---|---|---|
| `reasoning_replay` | bool | `true` | send each assistant turn back with its reasoning, so that each call extends the last and an episode trains as few sequences |
| `request_timeout_s` | int, at least 1 | `3600` | seconds one model call may take |
| `num_retries` | int, 0 to 10 | `0` | the SDK's `num_retries`, which counts attempts at a model call that fails transiently (0 and 1 make one); a retry may record a call twice |
| `condenser` | `none` or `llm_summarizing` | `none` | a summarizing condenser, the SDK's with the changes below (it condenses past 240 events, or when a prompt overflows the context window); its summary calls are the agent's `aux` calls, recorded and trained on |
| `condenser_max_tokens` | int, at least 0 | `0` | a token trigger for that condenser, which it then runs whatever `condenser` says: it condenses before a model call whose prompt would be longer than this many tokens, as the gateway counts them; `0` sets none |

The defaults are the training settings. Options change how the agent runs, never what is installed, so every
setting runs from the same toolchain. To evaluate as Harbor's adapter does (no replay, the SDK's 300-second timeout,
its `num_retries` of 5 and its step limit of 500), set the options in an evaluation config, or under
`eval_overrides` in a training config so that only its evaluations use them:

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs: {dataset_dir: ./tasks}
    harness: openhands
    eval_overrides:
      harness_options: {reasoning_replay: false, request_timeout_s: 300, num_retries: 5}

eval:
  workflow_config:
    rollout_config: {max_steps: 500}
```

On the command line, an override sets an option as it sets any key, and it is checked the same way:

```bash
platoon eval eval.yaml environments.0.harness_options.num_retries=0
platoon train train.yaml environments.0.harness_options.condenser=llm_summarizing
platoon train train.yaml environments.1.eval_overrides.harness_options.reasoning_replay=false
```

To condense before the context window fills, give the condenser a token trigger. With a 131072-token window
(`train.context_window_length`), this condenses at 80% of it:

```yaml
environments:
  - name: harbor-openhands
    service: harbor
    service_kwargs: {dataset_dir: ./tasks}
    harness: openhands
    harness_options: {condenser_max_tokens: 104857}  # 80% of 131072
```

Before each model call, the condenser counts the prompt the agent would send. Once it is longer than the trigger,
the agent condenses instead of calling: it summarizes its oldest turns, after the system prompt and the task, until
the rest fits in half the trigger, and goes on from the summary. The gateway does the counting, so the count is the
length of the prompt it renders with the policy's renderer: exactly, for a renderer that writes tool results without
their tool's name and serializes tool-call arguments itself, as the Qwen chat templates do, and a few tokens off for
each tool call with one that writes either as the call sent it (`kimi_k2`, `gpt_oss`). A hosted model's gateway
passes the count on to its provider, and where that gives none, the SDK's estimate counts instead (its `cl100k_base`
encoding runs about a tenth under a Qwen template on prose and code, and nearly half under on numbers); a prompt that
then overflows the context window still condenses, when the gateway refuses it. The trigger is a count of tokens,
not a share of the window, which the harness does not know. Set it well above the agent's first prompt, its system
prompt with the tools and the task (about 8k tokens with a short task under a Qwen template): below twice that
prompt, each condensation forgets every turn after the task, and a trigger that a prompt passes with nothing left to
forget but a summary can never be met, so the condenser sets it aside for the rest of the run and says so in the
harness log.

The summary calls are the agent's [`aux` calls](../getting-started/concepts.md#call-kinds): the runner marks them
with the `X-Platoon-Call-Kind` header, and the agent's own turns stay `policy` calls. A summary's step in the
trajectory says `kind: aux` in its `action_misc`, and an evaluation report counts the summaries under `usage.aux`.
They train with the rollout's advantage like the agent's turns, each in a training sequence of its own, and the
prompt after a summary starts another. The gateway takes `aux` calls from a harness in a task sandbox only where no
gateway step budget counts its calls, as here: the harness keeps its own step limit, `max_steps`, and each
condensation takes one of those steps. The step counts in training and evaluation metrics count a trajectory's
steps, so they count each summary call as a step, as they count the agent's turns. The summary call reads each
forgotten turn whole, with the commands it ran, their whole output and an earlier summary entire (the SDK's condenser
shows it 500 characters of each and no tool call's arguments), and a condensation keeps the task. A summary call
answered without text, with a tool call or a completion cut short in its reasoning, still condenses, with a notice
in the summary's place that the earlier turns were dropped.

Leave room for completions: the gateway clamps a completion to what the window has left after its prompt, so with a
completion budget of 49152 tokens (`max_completion_tokens`), a trigger of 81920 or less leaves every turn its whole
budget. Above it, a turn near the trigger can be cut short: cut in its reasoning, the SDK asks the agent again for an
answer and the episode goes on; cut in its visible text, that text becomes the agent's final message and the episode
ends.

## Which one?

1. The task lives in a sandbox (files, a repository, tests): `platoon-bash`, `mini-swe-agent` for software
   engineering, or `openhands` for the OpenHands agent. Your own command-line agent: [wrap it](cli-harness.md).
2. The environment's tools are the whole interface and the model calls tools well: `platoon-react`.
3. The model should compute between tool calls, batch many calls, or delegate to subagents, and running its code
   on the rollout host is acceptable: `platoon-codeact`.
4. None fits: [write a function harness](function-harness.md). A different CodeAct decision step can reuse the
   pieces in `platoon.harness.builtin.codeact`: its prompt builder, cell functions and cell executor.

Several harnesses can share one environment: `harness: {platoon-react: 1, platoon-codeact: 1}` draws one per task,
by weight.

## Changing a built-in

Start with options and prompt builders; then copy a spec or write your own harness. Changes to the built-ins
themselves go through a pull request with tests: the `platoon-codeact` tests in `tests/harness/builtin/` include
golden rollouts that record every prompt, step and training datum of scripted runs, so any change in behavior
shows up as a diff to explain. A change to a public contract (a new kind of harness, the spec format or
`AgentContext`) or to what trains starts as an RFC: see
[when a change needs an RFC](../../CONTRIBUTING.md#when-a-change-needs-an-rfc) and
[the RFC process](../rfcs/README.md#how-to-propose-a-change).

## Next steps

- [Write a function harness](function-harness.md).
- [Wrap a command-line agent](cli-harness.md) with a TOML spec.
- [The architecture guide](../architecture.md): the gateway, subagents and episode programs.
