# Run environments on env servers

**Goal.** Serve an ORS environment from one or more env servers and train against them with `ors+http://`,
instead of running it inside the rollout process with `python://`. Then set a per-session cap on tool calls,
pick the result format the model sees, and tune how concurrent tool calls share one environment.

Env servers are worth it when an environment is heavy (databases, browsers, MCP servers, large datasets), needs
its own machine or container, or should keep running while trainers come and go.

You will serve `OrderPackingEnv` from [Write a custom environment](custom-environment.md) and train on it with
[`examples/custom-environment/remote.yaml`](../../examples/custom-environment/remote.yaml).

## Prerequisites

- Platoon installed with the `ors` extra (`uv sync --extra tinker --extra ors` from the repository root).
- An ORS `Environment` class. This tutorial uses the one in
  [`examples/custom-environment`](../../examples/custom-environment/); run every command from that directory.
- To train: `TINKER_API_KEY`.

## How platoon talks to env servers

An environment service URI names the servers and the environment:

```text
ors+http://HOST:PORT[,HOST:PORT...]/EnvName[?variant=...&split=...&label=...]
```

`ors+https://` works the same way. With no hosts (`ors+http:///EnvName`, three slashes), the servers come from
an environment variable (step 3).

`EnvName` names the deployment; it does not pick one of a server's environments. A server that serves several
answers every request without a `variant` with its default one, the first it was given, so select another with
`?variant=<Env>`. `platoon train` and `platoon eval` ask each server once which environments it serves
(`GET /list_environments`, with a 5-second limit) and refuse an `EnvName` that is one of the server's other
environments, rather than train or evaluate the default one under its name, and a session that opens with one
fails with an `env_config` error. A name the server does not list, such as a deployment's own name, passes.

Each rollout opens one session on one server of the pool, chosen by hashing the rollout id: a rollout always
lands on the same server, and rollouts spread evenly across the pool. A subagent's forked session stays on its
parent's server. Closing the session deletes it on the server.

## Step 1: Start an env server

The openreward SDK's `Server` serves each environment class it gets at `/<ClassName>` (matched without regard
to case). [`serve.py`](../../examples/custom-environment/serve.py) serves `OrderPackingEnv`:

```python
from openreward.environments import Server
from order_packing import OrderPackingEnv

Server([OrderPackingEnv]).run(host=args.host, port=args.port)
```

```bash
cd examples/custom-environment
PYTHONPATH=. uv run python serve.py --port 8080
```

The server logs `Uvicorn running on http://127.0.0.1:8080`. In a second terminal, check that platoon reaches
it:

```bash
PYTHONPATH=. uv run python -c '
import asyncio
from platoon.envs import get_service
service = get_service("ors+http://127.0.0.1:8080/OrderPackingEnv")
print(asyncio.run(service.list_tasks("test")))'
```

```text
['test:0', 'test:1']
```

## Step 2: Point a config at the servers

[`remote.yaml`](../../examples/custom-environment/remote.yaml) replaces the `python://` service of
`train.yaml`:

```yaml
environments:
  - name: packing
    service: ors+http:///OrderPackingEnv
    service_kwargs:
      terminal_tools: [done]     # class attributes do not cross the wire: name the hints here
      read_only_tools: [look]
      max_tool_calls: 40         # per session; the 41st call ends the episode
    harness: platoon-react
```

An env server sends each tool's name, description and schema, and the SDK's `@terminal` tool by its name and
argument (rebuilt as one string parameter, without its descriptions). It does not send the class's `platoon_*`
attributes, so `platoon_terminal_tools`, `platoon_read_only_tools`, `platoon_concurrent_tools`,
`platoon_tool_meta`, `platoon_fork`, `platoon_copy` and `platoon_grade` have no effect remotely, and neither do a
tool's own `_meta` (`readOnlyHint` among them) and the SDK's `@tool(exclusive=True)` and `@tool(concurrent=True)`:
platoon keeps whatever meta a server's tool listing carries, and the openreward SDK's server sends none. Name the
hints with the `terminal_tools`, `read_only_tools` and `concurrent_tools` options, and the meta with `tool_meta`,
instead. An environment with no terminal tool at all gets a warning in the log, because subagents with
`no_terminal` access could then end the episode.

When the trainer can import the class the servers serve, name it instead, and the environment lists its tools
exactly as it does in process:

```yaml
    service: ors+http:///OrderPackingEnv
    service_kwargs:
      env_class: order_packing:OrderPackingEnv   # importable where platoon runs, as for python://
```

With `env_class`, platoon reads the `platoon_*` hint attributes, each tool's `_meta`, its exclusive and concurrent
flags and a `@terminal` tool's parameter schema from the class, as it does in process. Before the first session
and the task listing, it checks that the server lists the class's tools, with the same descriptions and schemas,
and refuses an env server that serves another version of the class (`env_config`; `platoon train` and
`platoon eval` exit 78). The hooks (`platoon_fork`, `platoon_copy`, `platoon_grade`) need the live environment,
which is on the server, so they still have no effect remotely; platoon logs a warning once when the class defines
any.

Remotely, subagents fork `shared` (the same remote session) or `fresh` (a new session for the same task); `copy`
is not supported.

## Step 3: Give each environment a URL pool

With no hosts in the URI, platoon looks for the servers in this order, when each session opens (so a launcher
can export the pool after the config was loaded):

1. The hosts in the URI, or the `urls` option (a list).
2. `OPENREWARD_SESSION_URLS_<LABEL>`: comma-separated URLs. `<LABEL>` is the environment's label upper-cased,
   with every character that is not a letter or digit replaced by `_`. The label is the `label` option (or
   `?label=` in the URI), else the environment name: `OrderPackingEnv` reads
   `OPENREWARD_SESSION_URLS_ORDERPACKINGENV`, and `label: order-packing` reads
   `OPENREWARD_SESSION_URLS_ORDER_PACKING`. The `session_urls_env_var` option names a different variable.
3. `OPENREWARD_SESSION_URLS`, a process-wide pool, unless the entry sets `legacy_global_pool: false`.
4. `OPENREWARD_SESSION_URL`, else `http://localhost:8080`.

A URL without a scheme gets `http://`. Train against the server from step 1:

```bash
export OPENREWARD_SESSION_URLS_ORDERPACKINGENV=http://127.0.0.1:8080
PYTHONPATH=. uv run platoon train remote.yaml
```

With more servers, list them all; each rollout picks one:

```bash
export OPENREWARD_SESSION_URLS_ORDERPACKINGENV=http://10.0.0.5:8080,http://10.0.0.5:8081,http://10.0.0.6:8080
```

**Mixtures.** When a config trains on several environments, give each entry its own pool and set
`legacy_global_pool: false`, so an entry whose variable is missing never falls back to another environment's
servers. A server answers every entry that names no `variant` with its default environment, so entries with
different env names on one server would all run the same one: `platoon train`, `eval` and `rollout` refuse such
a mixture before they contact any server, and warn about an entry that takes a process-wide pool
(`OPENREWARD_SESSION_URLS`, `OPENREWARD_SESSION_URL` or `localhost:8080`):

```yaml
environments:
  - name: packing
    weight: 2
    service: ors+http:///OrderPackingEnv
    service_kwargs: {terminal_tools: [done], legacy_global_pool: false}
    harness: platoon-react
  - name: search
    weight: 1                                                    # packing gets twice search's share of the slots
    service: ors+http:///SearchEnv
    service_kwargs: {label: search, legacy_global_pool: false}   # reads OPENREWARD_SESSION_URLS_SEARCH
    harness: platoon-react
```

**On a cluster**, run the env servers as their own jobs or containers (a Slurm or PBS job, a Singularity/Apptainer
or enroot container, a Kubernetes service), and export the pool variable in the training job once their
addresses are known. When a rollout worker process is killed, platoon deletes the remote sessions it held
(from the worker's signal handler or its supervisor); the SDK's server also frees a session that stays idle for
15 minutes.

## Step 4: Cap tool calls per session

`max_tool_calls: N` (default 0, no cap) bounds the env tool calls of one session:

- The call over the cap is not run. It ends the episode and returns an error result (a plain result under
  `bridge_json`) whose text is
  `{"bridge_error": "max_tool_calls_exceeded", "finished": true, "reward": <last reward or null>}`.
- Every later call, like any call after a tool ended the episode, gets
  `{"bridge_error": "environment_already_finished", "finished": true, "reward": <last reward or null>}`. So both
  refusals show the agent the last reward a tool sent, in either result format; under `blocks` they are the only
  results that do.
- The session grades `scored` with the last reward a tool sent (0 if none), reason `max_tool_calls`.

`shared` forks count against their parent's cap; `fresh` forks and `platoon_fork` children (in process) count
their own. The cap works the same for `python://` environments.

## Step 5: Choose the result format

`result_format` decides what the agent sees of a tool result:

| `result_format` | The agent sees |
|---|---|
| `blocks` (default) | The `ToolOutput`'s blocks as MCP content: text, and images as image blocks. An error is the error message. |
| `bridge_json` | One text block holding a JSON object: `finished`, `reward`, `text` (the text blocks joined), `blocks` (each block's text), and `data` and `metadata` when the tool sent them. An error reads `Error executing tool <name>: <message>`. |

For example, `pack` under `bridge_json`:

```json
{"finished": false, "reward": null, "text": "Packed 2 apple. The box holds 2 apple.", "blocks": ["Packed 2 apple. The box holds 2 apple."]}
```

Use `bridge_json` when a policy was trained on that format. Two related options:

- `bridge_tools: true` adds two read-only tools, `get_task` (the prompt and the tool catalog) and `get_status`
  (call count, finished flag and last reward), unless the environment has tools with those names.
- `log_dir: PATH` writes one directory per rollout with `bridge_events.jsonl` (every call and result) and
  `bridge_state.json`.

## Tool-call concurrency

Agents that share a live environment (a root and its `shared` or `platoon_fork` children, verifiers, agents
that a harness starts over HTTP) send their calls to one queue per live environment. Each call takes a turn of
one of three classes:

| Class | Tools | Runs |
|---|---|---|
| `serial` | every tool, unless declared otherwise (read-only tools included) | one at a time per live environment |
| `concurrent` | tools declared safe to overlap | beside serial calls and other concurrent calls |
| `exclusive` | declared terminal tools; also `platoon_grade`, `platoon_fork`, copy forks and the first tool listing | alone, after every call that arrived before it |

Declare a tool concurrent when it never changes state that another call reads, for example a search or a file
read:

| Declaration | Works with |
|---|---|
| `@tool(concurrent=True)` (openreward 0.1.163) | `python://` |
| `platoon_concurrent_tools = ("search",)` on the class | `python://` |
| `concurrent_tools: [search]` in `service_kwargs` | `python://` and `ors+http://` |

Turns are fair: calls from one agent run in the order it made them, and agents take turns, so one agent's burst
of parallel calls does not hold off another agent's single call. An exclusive call waits for everything that
arrived before it, and nothing that arrives after it starts before it ends.

`queue_timeout_s` (default 240 seconds; `null` for no limit) bounds how long a call waits for its turn. A call
that does not get its turn in time is not run and not counted; the agent gets an error result that begins
`Not run: env busy.` and can try again. A grade that cannot get its turn in time is an `error` outcome
(`env_busy`). The default stays below the 300-second per-call deadline of OpenHands' MCP client, so a call
the client has given up on never runs later.

One pattern stalls: a tool that waits for another call's effect (a `wait_for_event` that another agent's
`send_event` releases). It waits until its own timeout unless both tools are declared concurrent.

Sessions of other environment services (a custom `EnvironmentService` whose sessions do not set
`manages_concurrency = True`) take one call at a time per session.

## Reference: ORS service options

Options go in `service_kwargs`; `variant`, `split` and `label` may also go in the URI's query string.

| Option | Default | Meaning |
|---|---|---|
| `urls` | the URI's hosts | Env server URLs (step 3). |
| `label` | the environment name | Names the pool variable `OPENREWARD_SESSION_URLS_<LABEL>`. |
| `session_urls_env_var` | | A different pool variable. |
| `legacy_global_pool` | `true` | Fall back to `OPENREWARD_SESSION_URLS`. Set `false` in mixtures. |
| `variant` | | The environment to open on a server that serves several; unset, the server's default. Passed to the openreward client. |
| `api_key` | `$OPENREWARD_API_KEY`, else `local` | Credentials the client sends. |
| `api_url` | `$OPENREWARD_API_URL`, else the server's URL | The openreward API base URL. |
| `secrets` | | A mapping passed to the environment as `secrets`. |
| `split` | `train` | The split in which task names are looked up. |
| `task_names`, `task_indices` | | Narrow the listed tasks to these names or indices. To leave tasks out instead, use the entry's [`exclude_tasks`](../how-to/train-and-eval.md#leave-tasks-out). |
| `terminal_tools`, `read_only_tools`, `concurrent_tools` | | Tool hints (step 2, concurrency). |
| `tool_meta` | | `{tool: {...}}` added to the tool's MCP `_meta`. |
| `env_class` | | The class the env servers serve, as `module:Class`: its hints and tool descriptors apply as in process, and the servers' listing is checked against it (step 2). Remote only. |
| `max_tool_calls` | `0` (no cap) | Tool calls per session (step 4). |
| `queue_timeout_s` | `240` | How long a call waits for its turn (concurrency). |
| `result_format` | `blocks` | `blocks` or `bridge_json` (step 5). |
| `bridge_tools` | `false` | Add `get_task` and `get_status` (step 5). |
| `log_dir` | | Per-rollout call logs (step 5). |
| `close_timeout_s` | `30` | Bound on closing a session. |
| `prompt_filter` | | A callable or `module:attr` that rewrites the prompt text. |

All but `urls`, `variant`, `api_key`, `api_url`, `env_class` and the pool options also apply to `python://` services.

## Troubleshooting

- **Rollouts fail with `env_connect`.** No server answered at the chosen URL. Check the pool variable's name
  with `python -c 'from platoon.envs.ors import pool_env_var; print(pool_env_var("OrderPackingEnv"))'` and that
  the servers listen on an address the trainer can reach (`--host 0.0.0.0` for other machines).
- **The log warns that the env "declares no terminal tool".** Name the submit tool in `terminal_tools`, or the
  class in `env_class`.
- **`platoon train` or `platoon eval` exits 78 with `... serves ORS env '<Env>' with other tools than env_class
  ...`.** The server runs another version of the class than the trainer imports: deploy the same version, or drop
  `env_class` and name the hints with the options.
- **`platoon train` or `platoon eval` exits 78 with `environments[<name>].service: The env server ... serves
  [...]`.** The env name is one of the server's other environments, which the server would not run without a
  variant. Add `?variant=<Env>` to the URI.
- **Agents see `Not run: env busy.`** Calls queue behind slow serial tools. Declare tools that are safe to
  overlap as concurrent, or raise `queue_timeout_s`.
- **Rollouts end after a tool raised.** On openreward 0.1.163 and later, the client ends a remote session whose
  tool raised, so the agent's later calls are refused and the episode scores 0 (`no_submit`) unless it already
  submitted. Return an error message in the `ToolOutput` instead of raising, so the episode goes on. A tool that
  raises one of the SDK's session or transport errors is an infrastructure failure (`env_error`), in process and
  remotely ([When a tool raises](custom-environment.md#step-4-end-and-grade-the-episode)).

## Next steps

- [Write a custom environment](custom-environment.md): the environment class itself, grading and `platoon_fork`.
- [Gym adapters and task tables](gym-and-task-tables.md): environments from reset/step code and plain task
  lists.
- [The architecture overview](../architecture.md): the environment service contract and the agent tree.
