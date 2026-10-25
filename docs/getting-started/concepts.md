# Concepts

This page defines the terms the rest of the documentation uses. Read it once before your first run; come back
to it when a config key or a log line names something you have not met yet. For the reasons behind the design,
see [Architecture](../architecture.md).

## The training loop in one paragraph

A run trains a model on a set of **tasks** drawn from one or more **environments**. For each task, platoon runs a
group of **episodes** (rollouts). In each episode a **harness** (agent code) works on the task through the
environment's **tools** and samples the model through platoon's **model gateway**, which records the exact tokens
of every call. The environment **grades** the episode. The trainer turns each episode's agent tree into training
data, gives every token of a trajectory the same advantage (its reward minus its task group's baseline), and
sends the batch to a Tinker-compatible training backend (hosted Tinker, or a self-hosted server such as
[miles-tinker](https://github.com/neulab/miles-tinker)), which updates the weights and serves the new sampler.

## Glossary

| Term | Meaning | In code |
|---|---|---|
| task | One problem to solve: an id, a goal and task-specific data. | `pt.Task` |
| environment service | A source of tasks that opens one session per episode. | `pt.EnvironmentService` |
| session | One episode's live view of a task: instruction, tools, grading, forking. | `pt.EnvSession` |
| harness | Agent code: a Python function, or a command-line agent described by a TOML spec. | `@pt.harness`, `HarnessSpec` |
| agent | One running harness in an episode, with its own model endpoint, tools, step budget and trajectory. | `pt.AgentContext` |
| subagent | An agent started by another agent. The root agent and its subagents form the **agent tree**. | `ctx.agents.spawn` |
| episode | One task's root agent plus its subagents: one rollout. It yields one trajectory per agent. | `pt.TrajectoryCollection` |
| program | How an episode runs its agents and grades them. The default, `single`, runs the harness and grades the session. | `@pt.program`, `pt.EpisodeContext` |
| model gateway | The component that renders, samples, parses and records every model call. | `platoon.gateway` |
| agent key | The secret that identifies an agent at the gateway, for model calls and tool calls alike. | `ctx.endpoint`, `ctx.mcp` |
| tool plane | The gateway's MCP endpoint that serves each agent the environment's tools. | `ctx.mcp` |
| step budget | How many `policy` model calls an agent may make. | `ctx.budget` |
| call kind | What a model call is for: `policy`, `aux` or `judge`. It decides whether the call costs a step and trains. | `pt.CallKind` |
| outcome | An episode's (or a subagent's) grade, with a status that says whether it can train. | `pt.Outcome` |
| site profile | Where a machine or cluster keeps platoon's files, and which container runtime starts its sandboxes: `~/.config/platoon/site.yaml`, written once per site, never part of a project. | `platoon.site` |

## Environments, sessions and tools

An **environment service** lists task ids by split, loads a task, and opens a **session** for each episode
(`list_tasks(split)`, `get_task(task_id)`, `open(task, rollout_id=...)`). A session holds:

- `instruction`: the text the root agent works on;
- `tools()` and `call_tool(name, arguments)`: the environment's tools;
- `grade()`: the episode's `Outcome`, which never raises (a grader failure is an `error` outcome);
- `fork(mode)`: a session for a subagent;
- `sandbox`: an optional container for harnesses that act with their own built-in tools (for example a shell);
- `close()`.

An environment entry in a config names its service with the `service` key:

| `service` | What it is |
|---|---|
| `python://pkg.module:EnvClass` | An OpenReward (ORS) `Environment` class, run in the rollout process. |
| `ors+http://HOST:PORT[,...]/EnvName` | ORS environment servers over HTTP (or `ors+https`); a pool is sharded by rollout. `?variant=<Env>` selects one environment of a server that serves several. |
| `ors+http:///EnvName` | ORS environment servers listed in `OPENREWARD_SESSION_URLS_<LABEL>`. |
| `harbor` | A Harbor dataset, fetched by name (`hf://<org>/<name>@<commit>`, `harbor://<org>/<name>@<ref>` from the Harbor Hub, a git repository, or an adapter) or read from a directory of task directories, each task graded by its own tests in a sandbox. |
| `table` | Rows of tasks with no tools and no grader, for harnesses that grade themselves. |
| `pkg.module:Class` | Any other `EnvironmentService` class, built with `service_kwargs`. |

`platoon env list` prints this table. ORS task ids have the form `<split>:<index>`, for example `val:0`.

**Tools.** A tool is a `pt.ToolSpec`: a name, a description and a JSON schema, plus two hints. A `terminal` tool
ends the episode (a submit or finish tool); a `read_only` tool never changes the environment's state. A call
returns a `pt.ToolResult` with MCP-style content blocks, `is_error`, and the episode signals `finished` and
`reward`.

**Forks.** A subagent works in a fork of its parent's session:

| Fork mode | The subagent's session |
|---|---|
| `shared` | The same live environment as the parent; their tool calls take turns on it. |
| `copy` | A copy of the parent's environment state, changed independently. |
| `fresh` | A new session of the same task. |

## Harnesses

A **harness** is the code that plays a task. Platoon runs two kinds.

**Function harnesses** are async Python functions registered with `@pt.harness`. They run in the rollout process,
next to the gateway, and receive a `pt.AgentContext`:

```python
import platoon as pt


@pt.harness("echo")
async def echo(ctx: pt.AgentContext) -> pt.AgentResult:
    reply = await ctx.chat([{"role": "user", "content": ctx.instruction}])
    return pt.AgentResult("completed", final_message=reply.text)
```

**TOML spec harnesses** describe a command-line agent: how to install and launch it, which wire format it speaks
(`dialect`), and how it reports its final message. They run as a subprocess, locally or inside the task's sandbox,
and reach the model and the tools over HTTP.

Configs refer to a harness as `name` or `name@version`. Platoon ships these:

| Harness | Kind | How the model acts |
|---|---|---|
| `platoon-codeact` | function | Writes Python cells in `<python>` tags; the environment's tools are functions the cell calls. Runs model-written code on the rollout host. |
| `platoon-react` | function | Calls the environment's tools as native (OpenAI-format) tool calls. |
| `platoon-bash` | TOML spec | A minimal agent with a `bash` tool, plus the environment's tools over MCP. Needs only `python3`. |
| `mini-swe-agent` | TOML spec | The third-party mini-swe-agent CLI, installed into the task sandbox. Its only tool is `bash`. |
| `openhands` | TOML spec | An OpenHands SDK agent in the task sandbox, run from a toolchain that `platoon prepare` builds once per site. |
| `platoon-oracle`, `platoon-noop` | function | No model calls: run the task's reference solution, or nothing, to check a dataset. |

`platoon harness list` lists every registered harness, and `platoon harness show NAME` describes one. Packages
can add harnesses through the `platoon.harnesses` entry point group; [Extending
platoon](../extending/index.md) lists everything a package can add.

## Agents and the agent contract

An **agent** is one running harness. A function harness sees its agent as a `pt.AgentContext`:

| Attribute or method | What it gives the harness |
|---|---|
| `instruction`, `task`, `session` | The goal this agent works on, the task, and its env session. |
| `options` | The environment's `harness_options` (or the `options` of the spawn call). |
| `chat(messages, *, tools=None, kind="policy", **sampling)` | One model call through the gateway, recorded on this agent. Returns a `pt.ChatReply` (`text`, `tool_calls`, `message`). |
| `list_tools()`, `call_tool(name, arguments)` | The environment's tools, filtered by this agent's tool access. |
| `endpoint` | A `pt.ModelEndpoint` (`base_url`, `api_key`, `model`) for harnesses with their own OpenAI-compatible client. |
| `mcp` | A `pt.McpEndpoint` (`url`, `headers`): the same tools over MCP. |
| `budget` | `remaining()` and `used()` steps. |
| `max_steps`, `depth`, `role`, `access` | The agent's step limit, depth in the tree (root is 0), role and tool access. |
| `agents` | Subagents: `spawn(...)`. |
| `annotate(**misc)`, `annotate_step(**misc)` | Add data to the agent's trajectory, or to its last recorded step. `annotate_step(exclude_from_training=True)` keeps that call out of training. |

A harness returns a `pt.AgentResult(status, final_message=None, error=None)`. The status says how the run ended,
not whether it succeeded; the environment decides that.

| Status | Meaning |
|---|---|
| `completed` | The harness finished. The episode is graded. |
| `limit` | The agent ran out of steps. The episode is still graded. |
| `timeout`, `cancelled` | The run was interrupted. The outcome is untrainable. |
| `error` | The harness failed; `error` says why. The outcome is untrainable, unless the model overflowed its context window first. |

A harness never needs to raise: an exception in it becomes an `error` result.

## Subagents

An agent starts a subagent with `ctx.agents.spawn(goal, ...)`, which runs the child to completion and returns a
`pt.SubagentResult` (`goal`, `result`, `outcome`, `message`, `ok`). A child's failure never raises into its
parent: a rejected launch (no budget left, or too deep) or a failed child comes back with a non-`completed`
result and a message the harness can show the model. Spawns can run concurrently:

```python
children = await asyncio.gather(*(ctx.agents.spawn(goal, max_steps=5) for goal in goals))
summary = "\n".join(f"{child.goal}: {child.message}" for child in children)
```

The child runs the parent's harness and options unless the call names others, in a fork of the parent's
session, with tool access no wider than its parent's. A child's harness must be a function harness. The
environment's `agents` section sets the defaults:

| Key | Default | Meaning |
|---|---|---|
| `max_depth` | `null` (unlimited) | Deepest level of the policy tree; the root is depth 0. Verifiers may run up to two levels below it. |
| `budget` | `linear` | `linear`: a child draws on its parent's step budget. `depth_aware`: every agent has its own budget. |
| `child_max_steps` | `15` | A child's step budget when the spawn names none. |
| `child_grading` | `none` | `env`: grade the child's forked session. `verifier`: a read-only verifier agent judges the child. `none`: the child shares the episode's outcome. |
| `child_access` | `full` | `full`, `no_terminal` (no submit tool) or `read_only`. |
| `fork` | `null` | Fork mode for children: `shared`, `copy`, `fresh`, or `null` for the environment's default. |
| `child_timeout_s` | `null` | Wall-clock limit for each child run. |

## Step budgets

A **step** is one `policy` call recorded on an agent. An agent's budget is its `max_steps`; an unset or zero
budget is unbounded. With `budget: linear`, a spawn reserves the child's `max_steps + 1` steps on the parent (the
extra step is the parent's turn to read the result), and a parent's used steps include its descendants'. With
`budget: depth_aware`, each agent counts only its own steps and `max_depth` bounds the tree. Harnesses check
`ctx.budget.remaining()` between turns.

## Call kinds

Every model call carries a **kind**: the `kind` argument of `ctx.chat`, or the `X-Platoon-Call-Kind` header over
HTTP (`ctx.endpoint.headers(kind)` builds it).

| Kind | Costs a step | Trains | Use it for |
|---|---|---|---|
| `policy` (default) | yes | yes | The agent's turns. |
| `aux` | no | yes | Calls a harness makes for itself, such as summarizing its history. |
| `judge` | no | no | Judges; kept in the trajectory for audit only. |

## Episodes and programs

An **episode** is one rollout of one task. It yields a `pt.TrajectoryCollection` with one `pt.Trajectory` per
agent, linked parent to child.

A **program** decides how an episode runs. An environment picks one with `program:`; the default, `single`, runs
the configured harness on the root agent and grades the session. `gym` scores a `pt.gym.loop` harness by the
return it recorded. You can write your own:

```python
import platoon as pt


@pt.program("solve-then-check")
async def solve_then_check(ep: pt.EpisodeContext) -> pt.Outcome:
    await ep.run()  # the root agent, with the environment's harness
    ep.add_finalizer(my_reward_shaping)  # runs on the finished TrajectoryCollection
    return await ep.grade()
```

| `EpisodeContext` member | What it does |
|---|---|
| `task`, `session`, `config`, `route`, `root` | The task, the root session, the rollout config, the environment entry, and the root agent's record. |
| `run(harness=None, *, goal=None, options=None)` | Runs the root agent once and returns its `AgentResult`. |
| `grade()` | Grades the session, after the failure rules in [Outcomes](#outcomes). |
| `judge_endpoint()` | A model endpoint on a `judge` agent under the root: recorded for audit, never trained. |
| `add_finalizer(fn)` | Runs `fn(collection)` on the built `TrajectoryCollection` before the episode returns, in the order added. |
| `snapshot()` | The agent tree recorded so far, as a read-only `TrajectoryCollection`; agents without their own outcome carry a placeholder (`scored`, reward 0, reason `snapshot`). |

Programs run as trusted code and fail closed: a program or finalizer that raises gives an `error` outcome, never
a score.

**Gym adapters.** `pt.gym` adapts a classic step environment (`reset(task)` and `step(action)` returning
`(observation, reward, done)`). `pt.gym.loop(env_factory)` turns it into a function harness (register it with
`pt.harness("name")(pt.gym.loop(MyEnv))`), and `pt.gym.as_environment(env_factory, name=...)` turns it into an ORS
environment with one `act` tool, which any harness can play.

## The model gateway and agent keys

Every model call goes through the **model gateway**, in process (`ctx.chat`) or over HTTP. The gateway:

1. **renders** the messages and tool definitions into prompt tokens with the run's renderer;
2. **samples** the policy through the training backend;
3. **parses** the sampled tokens back into an assistant message (text, reasoning, tool calls);
4. **records** the exact prompt and sampled tokens on the agent that made the call.

Training uses the recorded tokens, not a re-tokenization of the messages, so what trains is exactly what the
model sampled.

Over HTTP the gateway speaks OpenAI Chat Completions (`/v1/chat/completions`), and, experimentally, OpenAI
Responses and Anthropic Messages, so harnesses that bring their own client work unchanged. Each agent gets a
random **agent key**. The key authenticates the agent's calls (as a Bearer token, an `x-api-key` header, or in the
URL path) and attributes every call to that agent. `ctx.endpoint.base_url` already contains the key, so a client
pointed at it needs no other setup.

The **renderer** (`train.renderer_name`) decides how messages become tokens and must match the model. Platoon
accepts the tinker-cookbook renderers, such as `qwen3_instruct`, and its own `chat_template` renderer, which
renders prompts with the model's own chat template and parses tool calls with the parser of the model's format
([Models and tool calling](../how-to/models-and-tool-calling.md)). Its Qwen aliases
(`qwen3_6_preserve_thinking`, `qwen3_8_preserve_thinking` and their `_disable_thinking` variants) are
`chat_template` with preset settings; `train.renderer_kwargs` sets the others.

## The MCP tool plane

The gateway also serves each agent the environment's tools over MCP (streamable HTTP), at the URL in `ctx.mcp`.
A harness reaches its tools the way it reaches the model: as the agent its key names. Inline calls
(`ctx.call_tool`) and MCP calls share one filter, one call order and one record format.

**Tool access** filters what an agent sees, using the tools' hints:

| Access | Tools listed |
|---|---|
| `full` | Every tool. |
| `no_terminal` | Every tool except terminal (submit or finish) tools. |
| `read_only` | Only tools marked read-only. |

A tool the agent cannot see cannot be called. A tool result's `finished` and `reward` travel in the MCP result's
`_meta["platoon"]`, which clients do not show the model.

## Outcomes

`pt.Outcome.status` says whether a measurement is valid, not whether the policy succeeded. Only `scored`
outcomes train.

| Status | Meaning | Trains |
|---|---|---|
| `scored` | A valid measurement, with a reward. A failed attempt is `scored` with a low reward. | yes |
| `invalid` | The environment declared the measurement invalid. | no |
| `error` | An infrastructure failure: the harness crashed, the grader failed, the environment broke. `failure_kind` classifies it. | no |
| `timeout` | The episode ran out of wall-clock time. | no |
| `cancelled` | The episode was cancelled. | no |

A failed measurement never becomes a reward of 0: it is kept out of training and out of the group's baseline.
`reason` adds detail, for example `no_submit` (the agent never submitted, scored 0), `context_overflow` (the
model filled its context window; a policy failure, so it is graded), or `agent_timeout`. Common `failure_kind`
values are `harness_crash`, `harness_setup`, `harness_config`, `env_error` and `unexpected:<ExceptionType>`.

An outcome can carry several named rewards (`rewards`); `primary` names the one that trains (default `reward`).

## Credit assignment

The trainer runs `train.workflow_config.group_size` episodes of each task and assigns credit in four steps:

1. **Datums.** Each agent's trajectory becomes training data. Consecutive calls whose prompt extends the previous
   prompt and completion merge into one sequence, so a multi-turn agent trains as few long sequences.
2. **Rewards.** Every action token of a trajectory carries that trajectory's reward: the root's from the episode
   outcome, a subagent's from its own grade or, with `child_grading: none`, from the episode.
3. **Baseline.** Each episode's baseline is the mean root reward of its group, or with `leave_one_out_baseline:
   true`, the mean of the other members. Interrupted episodes never enter a baseline.
4. **Advantages.** Each episode's baseline is subtracted from the rewards of all its datums, root and subagents
   alike.

Optional transforms reweight subagent data by depth (`depth_level_weighting`, `depth_level_discount_gamma`), sample
subagent datums (`subagent_datum_keep_probability`) and suppress error tokens that would get positive credit
(`filter_errors`). See [Architecture](../architecture.md) for how these fit together.

## Rollouts in the trainer

Rollouts run as asyncio tasks in the trainer process (`rollout_isolation: inprocess`, the default) or one per
worker process (`rollout_isolation: process`), which can be killed on its own. Rollouts run ahead of training:
a group sampled more than `train.max_staleness` policy versions behind the trainer is dropped.

In training and evaluation, every finished rollout writes an event file that `platoon viz` can show; see
[Your first run](first-run.md).

## Next steps

- [Install platoon](install.md).
- [Your first run](first-run.md): one episode, a short training run and the trajectory viewer.
- [Architecture](../architecture.md): why the layers are split this way.
