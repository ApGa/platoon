# Write a function harness

**Goal.** Write an agent as a Python function, register it as a harness, run it on one task, and read what platoon
recorded. You will build `example-tool-loop`, a minimal agent that plays any environment through its tools, and run it
on number search, a small guessing game. The finished files are in
[`examples/function-harness/`](../../examples/function-harness/).

A *harness* is agent code. A *function harness* is an `async` Python function that runs one agent:

```python
async def my_agent(ctx: pt.AgentContext) -> pt.AgentResult: ...
```

Platoon owns everything around the loop: it opens the environment session, samples the model through its gateway
and records every call, keeps the step budget, grades the episode and turns the records into training data. The
harness owns only its loop. Function harnesses run in the rollout process's event loop, next to the gateway, so a
harness must not block: use `await` for anything slow. [The architecture guide](../architecture.md) explains the
layers.

## Prerequisites

- Python 3.12 or later and a checkout of platoon, installed from the repository root with the `tinker` and `ors`
  extras: `uv sync --extra tinker --extra ors`.
- Number search, the example environment, ships with platoon: the `ors` extra runs it.
- A [Tinker](https://thinkingmachines.ai/tinker/) API key in `TINKER_API_KEY` (and `TINKER_BASE_URL` for a
  self-hosted server, such as [miles-tinker](https://github.com/neulab/miles-tinker)) to sample a real model in
  steps 4 and 5. The example's test runs without one.

Run the commands below from the repository root.

## Step 1: write the harness

Create `tool_loop.py`:

```python
"""``example-tool-loop``: a minimal function harness that plays any environment through its tools.

The model sees the environment's tools as OpenAI function tools. Each turn is one model call; every tool call
in the reply is answered through the environment, and the agent ends when a tool result says the episode is
over, when the model answers without calling a tool, or when the step budget runs out.

Options (``harness_options`` in the environment config):

* ``system_prompt``: a system message to put before the instruction.
* ``sampling``: Chat Completions parameters (``temperature``, ``max_completion_tokens``, ...) for every call;
  they override the rollout's ``inference_params``.
"""

from __future__ import annotations

import json
from typing import Any

import platoon as pt
from platoon.harness.agent import LIMIT_MESSAGE


@pt.harness("example-tool-loop")
async def tool_loop(ctx: pt.AgentContext) -> pt.AgentResult:
    """Call the environment's tools until it ends the episode, the model answers, or the budget runs out."""

    # The env's tools, already filtered by this agent's tool access.
    specs = await ctx.list_tools()
    if not specs:
        return pt.AgentResult("error", error="example-tool-loop needs an environment with tools")
    tools = [
        {
            "type": "function",
            "function": {"name": spec.name, "description": spec.description, "parameters": spec.input_schema},
        }
        for spec in specs
    ]

    messages: list[dict[str, Any]] = []
    if ctx.options.get("system_prompt"):
        messages.append({"role": "system", "content": ctx.options["system_prompt"]})
    messages.append({"role": "user", "content": ctx.instruction})
    sampling = dict(ctx.options.get("sampling") or {})

    while True:
        # One step is one policy call. Check the budget before each one: platoon does not stop the loop for you.
        if ctx.budget.remaining() <= 0:
            return pt.AgentResult("limit", error=LIMIT_MESSAGE)
        reply = await ctx.chat(messages, tools=tools, **sampling)
        messages.append({key: value for key, value in reply.message.items() if value is not None})

        if not reply.tool_calls:  # an answer without a tool call ends the agent
            ctx.annotate_step(observations=[])
            return pt.AgentResult("completed", final_message=reply.text)

        observations = []
        for call in reply.tool_calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except ValueError as exc:  # json.JSONDecodeError is a ValueError too
                text, result = f"Invalid arguments: {exc}", None
            else:
                result = await ctx.call_tool(name, arguments)
                text = result.as_text()
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
            observations.append({"tool": name, "output": text})
            if result is not None and result.finished:  # the environment ended the episode
                ctx.annotate_step(observations=observations)
                return pt.AgentResult("completed", final_message=text)
        # Annotations land on the step of the last model call: this turn's reply.
        ctx.annotate_step(observations=observations)
```

What each part does:

| Code | What it does |
|---|---|
| `@pt.harness("example-tool-loop")` | Registers the function as `example-tool-loop@1`. `version="2"` registers another version; `executes_model_code=True` declares that the harness runs code the model wrote, which every rollout records. The function must be `async`. |
| `ctx.list_tools()` | The environment's tools as `ToolSpec`s (`name`, `description`, `input_schema`, and the hints `terminal` and `read_only`), filtered by the agent's tool access. |
| `ctx.options` | The environment's `harness_options` (step 2), read-only. |
| `ctx.instruction` | The task's instruction: the goal this agent works on. |
| `ctx.chat(messages, tools=..., **sampling)` | One model call through the gateway, which renders the messages with the configured renderer, samples, parses the reply and records the exact tokens. It returns a `ChatReply`: `text`, `tool_calls` (OpenAI format), `message` (the whole assistant message), `reasoning` and `finish_reason`. Keyword arguments are Chat Completions parameters, such as `temperature` or `max_completion_tokens`; the ones a call leaves out come from the rollout's `inference_params`. |
| `ctx.call_tool(name, arguments)` | Calls an environment tool and records the call. It returns a `ToolResult`: `content` (MCP-style blocks), `as_text()`, `is_error`, and the episode signals `finished` and `reward`. A tool outside the agent's access returns an error result without reaching the environment. |
| `ctx.annotate_step(**misc)` | Merges `misc` into the step of the last model call recorded on this agent. Viewers show it; training ignores it. |
| `ctx.budget.remaining()` | Steps this agent may still take: `inf` when it has no budget. |
| `pt.AgentResult(status, final_message=..., error=...)` | How the run ended (see [How a run ends](#how-a-run-ends)). |

Platoon's own built-in `platoon-react` is this loop plus terminal-tool handling; [Built-in
harnesses](builtin-harnesses.md) compares it with the others.

## Step 2: name the harness in a config

An `environments` entry pairs an environment service with a harness. This is the example's evaluation config,
[`config.yaml`](../../examples/function-harness/config.yaml):

```yaml
inference:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  output_dir: ./runs/function-harness
  workflow:
    num_concurrent_workers: 8
    rollout_config:
      max_steps: 12          # the root agent's step budget: 12 policy calls
      timeout: 300           # seconds for the whole rollout
      verbose: false
      inference_params: {max_completion_tokens: 512}

environments:
  - service: python://platoon.envs.number_search.env:NumberSearchEnvironment
    harness: example-tool-loop  # or tool_loop:tool_loop, without `package`
    package: tool_loop          # imported in every rollout process, so @pt.harness("example-tool-loop") registers
    harness_options:
      system_prompt: >-
        Find the hidden number. Call the guess tool with one integer per turn; it says whether
        your guess is too low, too high or correct.
    eval_dataset_kwargs: {split_name: val, limit: 20}
```

`harness_options` becomes `ctx.options`. Platoon finds a function harness in one of three ways:

| How | Config | Use it when |
|---|---|---|
| `package` | `harness: example-tool-loop` and `package: tool_loop` | The module is on `PYTHONPATH`. Platoon imports it before it resolves the name, in every process that runs rollouts. |
| `module:function` | `harness: tool_loop:tool_loop` | The same, with no `package` line: platoon imports the module when it resolves the reference. |
| Entry point | `harness: example-tool-loop` | You ship the harness as a package. Platoon finds it wherever the package is installed, and `platoon harness list` shows it. |

For the entry point, declare the module in the `platoon.harnesses` group of your package's `pyproject.toml`, as
the example's [`pyproject.toml`](../../examples/function-harness/pyproject.toml) does:

```toml
[project.entry-points."platoon.harnesses"]
example-tool-loop = "tool_loop"
```

The entry point's name is the harness's name: platoon imports it the first time a config names `example-tool-loop`, and
never otherwise. Importing the module registers its `@pt.harness` functions; an entry point may also name the
decorated function itself, or a TOML spec file. An entry point that fails to import fails only the lookups of its
own name, with an error that names its package, and `platoon harness list` lists the installed harnesses without
importing them. When a TOML spec and a function harness share a name and version, the spec wins; when their
versions differ, a reference without a version is ambiguous, as for any two versions of one name.

## Step 3: check that platoon finds it

```bash
PYTHONPATH=examples/function-harness uv run platoon harness show tool_loop:tool_loop
```

Expected output:

```text
example-tool-loop@1  (function)
source: tool_loop:tool_loop
runtime: inline
executes_model_code: False

Call the environment's tools until it ends the episode, the model answers, or the budget runs out.
```

An unknown name exits with code 78 and lists the harnesses platoon knows.

## Step 4: run one task

`platoon rollout` runs a single task through the configured route and prints the agent tree. Number-search task
ids are `<split>:<index>`:

```bash
PYTHONPATH=examples/function-harness uv run platoon rollout examples/function-harness/config.yaml --task val:0
```

The output ends with a summary like this one, from a policy that plays well; a real model's step count and reward
vary:

```text
rollout val-0-5b986d46c362  task val:0
agents 1  steps 9  max depth 0  reward 1.0  outcome scored reward=1.0
└─ policy e2daca7d  steps 9  reward 1.0  outcome scored reward=1.0
```

One agent took nine steps (nine model calls) and the environment scored the episode 1.0. The run directory,
`runs/adhoc/<time>-val-0/` unless you pass `--output-dir`, holds the resolved config (`config.resolved.yaml`),
`lock.json`, `trajectory_collection.json` and the rollout's event file under `rollouts/events/`. In the
collection, the root trajectory has one step per model call, and:

- each step's `misc` holds the sampled `message` and your annotations, here `observations`;
- `misc.agent_result` is the `AgentResult`'s status and error, `misc.outcome` the environment's grade;
- `misc.tool_calls` records every tool call with its arguments, result text, `finished` and `reward`;
- `misc.harness` names the harness (`example-tool-loop@1`), the episode program and `executes_model_code`.

Open it in the trajectory viewer:

```bash
uv run platoon viz show-dump runs/adhoc/<time>-val-0/trajectory_collection.json
```

## Step 5: evaluate and train

Evaluate on 20 validation tasks:

```bash
PYTHONPATH=examples/function-harness uv run platoon eval examples/function-harness/config.yaml
```

Training uses the same `environments` entry in a training config.
[`train.yaml`](../../examples/function-harness/train.yaml) trains a LoRA for five optimizer steps of 8 tasks x 4
rollouts:

```bash
PYTHONPATH=examples/function-harness uv run platoon train examples/function-harness/train.yaml
```

Training uses the exact tokens the model sampled in each recorded call; calls whose prompt extends the previous
call's prompt and reply merge into one training sequence.

## Budgets

A step is one `policy` call recorded on the agent. Each call to `ctx.chat` has a kind:

| `kind=` | Uses a step | Trains | Use it for |
|---|---|---|---|
| `policy` (default) | yes | yes | the agent's turns |
| `aux` | no | yes | calls the harness makes for itself, such as summarizing its history |
| `judge` | no | no; kept in the trajectory's `misc.judge_calls` | judging |

The root agent's budget is `max_steps` from the rollout config (or the task); without one, `ctx.budget.remaining()`
is `inf`. Platoon does not stop a harness that ignores its budget: check it before each call, and end with
`pt.AgentResult("limit", error=LIMIT_MESSAGE)`, the message the built-in harnesses use. The rollout's `timeout` is
the only backstop. `platoon.harness.agent.out_of_steps(ctx, steps_taken)` checks both the budget and
`ctx.max_steps`.

Subagents started with `ctx.agents.spawn` get budgets of their own; with `agents.budget: linear` (the default) a
parent's `ctx.budget` also counts its subagents' steps and reserves theirs while they run. [The architecture
guide](../architecture.md) covers subagents.

## How a run ends

The status says how the harness ended, not whether it solved the task: the environment's grade is the reward.

| Status | Who sets it | What platoon does |
|---|---|---|
| `completed` | the harness returns it | grades the session |
| `limit` | the harness returns it when the budget runs out | grades the session |
| `error` | the harness returns it, raises an exception, or returns something other than an `AgentResult` | records an `error` outcome (`harness_crash`), which never trains |
| `timeout` | platoon, when the rollout's `timeout` passes (or the harness returns it) | records a `timeout` outcome, which never trains |
| `cancelled` | platoon, when the rollout is cancelled | records a `cancelled` outcome, which never trains |

One exception: when the gateway refused a prompt because it overflowed the model's context window and the harness
then failed, the episode is graded as a policy failure with `reason: context_overflow`. Set
`on_context_overflow: exclude` on the environment to keep such episodes out of training instead.

## Annotations

- `ctx.annotate_step(**misc)` merges into the step of the last model call. Call it after `ctx.chat` and before
  the next one, as `example-tool-loop` does. It raises `RuntimeError` before the first call.
- `ctx.annotate_step(exclude_from_training=True)` keeps that call out of the training data; the step stays in
  the trajectory for viewers.
- `ctx.annotate(**misc)` merges into the agent's trajectory `misc`.

## Bring your own client

A harness with its own model client, such as the OpenAI SDK or an agent framework, can use the same gateway over
HTTP. `ctx.endpoint` has an OpenAI-compatible `base_url` (ending in `/v1`, already scoped to this agent), an
`api_key` and a `model`; calls through it are recorded exactly like `ctx.chat`. Send
`ctx.endpoint.headers("aux")` to mark a call's kind. `ctx.mcp`, when set, serves the environment's tools over MCP
(`url` and `headers`). A command-line agent goes further and runs as its own process: see
[Wrap a command-line agent](cli-harness.md).

## Next steps

- [Wrap a command-line agent](cli-harness.md) with a TOML harness spec.
- [Built-in harnesses](builtin-harnesses.md): `platoon-codeact`, `platoon-react` and `platoon-bash`, and when to
  use each.
- [The architecture guide](../architecture.md): subagents (`ctx.agents.spawn`), episode programs (`@pt.program`)
  and the gateway.
- [Publish an extension package](../extending/publish-a-package.md): ship the harness in a package of its own,
  found by name wherever it is installed.
- The agent contract in full: [`platoon/harness/agent.py`](../../platoon/harness/agent.py).
