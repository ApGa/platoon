# Write a custom environment

**Goal.** Write an environment for platoon as an [OpenReward](https://openreward.ai) (ORS) `Environment`
class, check it on your laptop without a model, and train a policy on it. You will build `OrderPackingEnv`, the
environment in [`examples/custom-environment`](../../examples/custom-environment/): the agent moves items from a
shelf into a box until the box holds exactly the order.

Along the way you will use every part of an environment that platoon reads: tasks and splits, the prompt, tools
with typed parameters, a terminal tool that grades the episode, read-only hints, and `platoon_fork`, which lets
subagents work on the same state as their parent.

For the design behind environments, sessions and grading, see [the architecture overview](../architecture.md).

## Prerequisites

- Python 3.12 or later and [uv](https://docs.astral.sh/uv/).
- Platoon installed from the repository root with the `ors` extra (the openreward SDK) and, to train, the
  `tinker` extra:

  ```bash
  uv sync --extra tinker --extra ors
  ```

- To train or to run a rollout with a real model: a [Tinker](https://thinkingmachines.ai/tinker/) API key in
  `TINKER_API_KEY`. Steps 1 to 7 need no key and no GPU.

## How platoon runs an environment

A config entry names the environment with `service: python://<module>:<Class>`. For every rollout, platoon
creates one instance of the class in the process that runs the rollout and drives it the way the openreward
SDK's own server does:

1. `Class(task_spec=..., secrets=...)`, then `setup()`.
2. `get_prompt()`: its text becomes the agent's instruction.
3. The harness calls the `@tool` methods. Each returns a `ToolOutput`; the first one with `finished=True` ends
   the episode and its `reward` grades it.
4. `teardown()` when the rollout ends.

Task ids are `<split>:<index>`, so `train:0` is the first task of the `train` split.

## Step 1: Define the tasks and splits

`list_splits` names the splits and `list_tasks(split)` returns one task spec (a JSON-serializable dict) per task.
Platoon passes the spec to the constructor as `task_spec`.

```python
# examples/custom-environment/order_packing.py
SHELF = {"apple": 6, "bread": 2, "cheese": 3, "egg": 12, "milk": 2}
ORDERS = {
    "train": [{"apple": 2, "bread": 1}, {"egg": 6, "milk": 1}, ...],
    "test": [{"apple": 1, "cheese": 2}, {"egg": 4, "bread": 1, "apple": 2}],
}


class OrderPackingEnv(Environment):
    @classmethod
    def list_splits(cls) -> Sequence[Split | str]:
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> Sequence[dict[str, Any]]:
        if split not in ORDERS:
            raise ValueError(f"Unknown split {split!r}; known: {sorted(ORDERS)}")
        return [{"id": f"{split}-{index}", "order": order} for index, order in enumerate(ORDERS[split])]
```

A spec's `task_name`, `id`, `task_id` or `name` key (the first one set) is the task's display name. Platoon
also resolves a task by that name, so `train-0` and `train:0` name the same task.

## Step 2: Write the constructor and the prompt

The constructor takes `task_spec` and `secrets`, as the SDK's server calls it. Extra keyword-only arguments are
fine; `OrderPackingEnv` uses them in step 6.

```python
class OrderPackingEnv(Environment):
    ...

    def __init__(self, task_spec=None, secrets=None, *, shelf=None, box=None) -> None:
        super().__init__(dict(task_spec or {}), dict(secrets or {}))
        self.order: dict[str, int] = dict(self.task_spec.get("order") or {})
        self.subtask = shelf is not None  # a platoon_fork child, packing into its parent's box
        self.shelf = shelf if shelf is not None else dict(SHELF)
        self.box = box if box is not None else {}
        self.start = dict(self.box)  # what the box held when this agent started

    def get_prompt(self) -> list[TextBlock]:
        return [
            TextBlock(
                text=(
                    f"Pack this order into the box: {describe(self.order)}.\n"
                    f"The shelf holds {describe(self.shelf)}. Move items with the pack tool. "
                    "When the box holds exactly the order, reply with a one-line summary to finish."
                )
            )
        ]
```

Write the prompt for any harness. `platoon-react` and `platoon-codeact` send the agent's final answer to the
environment's terminal tool (step 4), so "reply with a summary to finish" works with both, and a harness that
shows the model every tool lets it call `done` itself.

## Step 3: Add tools with typed parameters

A tool is a method decorated with `@tool` that takes one pydantic model (or nothing) and returns a
`ToolOutput`. Platoon shows the model the method name, its docstring and the model's JSON schema, so use
`Field(description=...)` for every field and pydantic constraints for the valid range:

```python
class PackParams(BaseModel):
    item: str = Field(description="The item to move from the shelf into the box, e.g. 'apple'.")
    quantity: int = Field(default=1, ge=1, le=12, description="How many to move.")


class OrderPackingEnv(Environment):
    ...

    @tool
    async def pack(self, params: PackParams) -> ToolOutput:
        """Move items from the shelf into the box."""

        item, quantity = params.item.strip().lower(), params.quantity
        left = self.shelf.get(item, 0)
        if left < quantity:
            return ToolOutput(blocks=[TextBlock(text=f"Cannot pack {quantity} {item}: the shelf has {left}.")])
        self.shelf[item] = left - quantity
        self.box[item] = self.box.get(item, 0) + quantity
        return ToolOutput(blocks=[TextBlock(text=f"Packed {quantity} {item}. The box holds {describe(self.box)}.")])
```

Arguments that fail validation never reach your method: the agent gets an error result, and the episode goes
on.

**Report the policy's mistakes in the result; do not raise.** A tool that raises comes back to the agent as an
error result. But on an env server with openreward 0.1.163 or later, a tool that raises also ends the remote
session, so the agent cannot try again, and the episode scores 0 unless it already submitted. Return a
`ToolOutput` that says what went wrong, as `pack` does when the shelf runs out.

## Step 4: End and grade the episode

A tool ends the episode by returning `finished=True`; its `reward` is the episode's reward. Declare such a tool
*terminal* so platoon knows it ends the episode:

```python
class OrderPackingEnv(Environment):
    platoon_terminal_tools = ("done",)

    @tool
    async def done(self, params: DoneParams) -> ToolOutput:
        """Finish packing: the box is checked against the order, and the episode ends."""

        packed = {item: self.box.get(item, 0) - self.start.get(item, 0) for item in self.box}
        missing = {item: n - packed.get(item, 0) for item, n in self.order.items() if packed.get(item, 0) < n}
        extra = {item: n - self.order.get(item, 0) for item, n in packed.items() if n > self.order.get(item, 0)}
        if self.subtask:  # a subagent answers only for its own items: its siblings pack into the same box
            extra = {}
        reward = 0.0 if missing or extra else 1.0
        ...
        return ToolOutput(
            blocks=[TextBlock(text=verdict)],
            reward=reward,
            finished=True,
            metadata={"missing": missing, "extra": extra},
        )
```

What "terminal" changes:

- The call runs alone: it waits for the calls in flight on the environment, and later calls wait for it.
- Agents with `no_terminal` tool access (for example subagents configured with `agents.child_access:
  no_terminal`) do not see the tool.
- `platoon-react` and `platoon-codeact` route the agent's final answer into it. `platoon-react` keeps the
  terminal tool out of the model's tool list and calls it with the final plain-text reply; `platoon-codeact`'s
  `finish(message)` calls it. The answer goes to the tool's first parameter (`summary` here).

There are three ways to declare a terminal tool:

| Declaration | Works with | Notes |
|---|---|---|
| `platoon_terminal_tools = ("done",)` on the class | `python://`, and `ors+http://` with `env_class` | Any openreward version. |
| `@terminal` above `@tool` (openreward 0.1.157 and later) | `python://` and `ors+http://` | At most one per environment; it takes one field or none. |
| `terminal_tools: [done]` in `service_kwargs` | `python://` and `ors+http://` | The way to declare it for an env server that does not use `@terminal`. |

`@tool(exclusive=True)` (openreward 0.1.163) also marks a tool terminal, in process, or on an env server whose
service names the class (`env_class`, see [Run environments on env servers](remote-environments.md)).

**How platoon grades the session.** Only `scored` outcomes train; everything else is kept out of training and
out of the group baseline instead of counting as reward 0.

| What happened | Outcome |
|---|---|
| A tool returned `finished=True` with a numeric `reward` (the first such result decides) | `scored`, that reward |
| ... with `metadata={"invalid": True}` | `invalid`: the environment declared the measurement invalid |
| ... without a numeric `reward` | `invalid` (`no_reward`) |
| No finished result, and the session reached `max_tool_calls` | `scored`, the last reward a tool sent (else 0), reason `max_tool_calls` |
| No finished result, and the class defines `platoon_grade()` | what `platoon_grade()` returns (see below) |
| No finished result, and an infrastructure failure: the session died, a transport error, a terminal tool raised, or a tool raised one of the openreward SDK's own session or transport errors (below) | `error` (`env_error`) |
| No finished result otherwise | `scored`, reward 0, reason `no_submit` |

**When a tool raises.** Platoon classifies a tool that raises the same way in process (`python://`) and on an env
server (`ors+http://`), whichever openreward version the trainer runs:

- One of the openreward SDK's own session or transport errors, as a sandbox or API client the environment uses
  raises them, says that the environment's backing session or service failed: it is infrastructure, and an
  episode that never submits is `error` (`env_error`), never a 0. In process platoon recognizes the SDK's classes
  `SessionTerminatedError`, `TransportError`, `MaxRetriesError` and `HeartbeatTimeoutError` and their subclasses,
  but not another library's class of the same name: httpx's `ConnectError` is an httpx `TransportError`, and the
  policy's, as on an env server. An env server sends only an exception's text, so remotely platoon recognizes the
  texts the SDK gives a `SessionTerminatedError` and a `MaxRetriesError`: there a `HeartbeatTimeoutError` (which
  has no text) or a `TransportError` with a text of your own counts as the policy's, and in either mode so does an
  exception raised `from` one of them.
- Any other exception is the policy's failed call, whatever caused it (a `ValueError` for an argument the tool
  cannot use, say): the agent sees the error, and an episode that never submits scores 0 (`no_submit`). Arguments
  that do not match the tool's parameter model never reach the tool: the SDK refuses them, and the episode goes on.
- From openreward 0.1.163, the client ends a remote session whose tool raised (the SDK treats a raised tool as the
  end of the rollout), so an episode on an env server cannot go on after one, while in process and with 0.1.125 it
  can.

So return an error message in the `ToolOutput` for anything the agent can fix, and end the episode with
`finished=True` and `metadata={"invalid": True}` when your environment knows its measurement failed: both work the
same in process and on an env server.

`platoon_grade()` grades from state when nothing was submitted, for example when the agent ran out of steps.
It returns a number, a mapping such as `{"reward": 0.5, "metadata": {...}}` (add `"invalid": True` to declare the
grade invalid), or a `pt.Outcome`:

```python
    def platoon_grade(self) -> dict:
        return {"reward": 0.5 if self.box else 0.0, "metadata": {"box": dict(self.box)}}
```

A harness that crashes or times out gets an untrainable outcome without grading.

## Step 5: Mark read-only tools

A read-only tool never changes the environment. Declare it, and agents with `read_only` access (by default the
verifier agents that check a subagent's work) can still use it:

```python
class OrderPackingEnv(Environment):
    platoon_read_only_tools = ("look",)

    @tool
    async def look(self) -> ToolOutput:
        """Show what is left on the shelf and what is in the box."""

        return ToolOutput(blocks=[TextBlock(text=f"Shelf: {describe(self.shelf)}.\nBox: {describe(self.box)}.")])
```

Platoon also treats a tool as read-only when its MCP metadata says so (`readOnlyHint: true` in `annotations`, or
`platoon.dev/read-only: true`), and when the `read_only_tools` service option names it. Read-only decides what
an agent may see, not what may run at the same time; [Remote environments](remote-environments.md#tool-call-concurrency)
explains the concurrency policy.

## Step 6: Let subagents share the environment

A harness can split its task among subagents with `ctx.agents.spawn(goal)`. Each child runs on a fork of its
parent's session. By default the fork is `shared`: the child uses the same live environment, the same call
count and the same "episode finished" latch, so a child that calls `done` would end everyone's episode.

`platoon_fork(task_spec)` gives a child an environment object of its own, with its own goal, terminal call and
grade, on the parent's state. Platoon calls it with the parent's task spec and the child's `goal`:

```python
    def platoon_fork(self, task_spec: Mapping[str, Any]) -> OrderPackingEnv:
        goal = str(task_spec.get("goal") or "")
        order = parse_items(goal)  # "Pack 2 apple" -> {"apple": 2}
        if not order:
            raise ValueError(f"a subgoal names the items to pack, like 'Pack 2 apple'; got {goal!r}")
        return OrderPackingEnv({"id": task_spec.get("id"), "order": order}, shelf=self.shelf, box=self.box)
```

The child's environment holds the same `shelf` and `box` dicts, so what a child packs is in its parent's box.
Its `done` grades only the child's items; the parent's `done` grades the whole box.

- Raising refuses the subgoal: the parent's spawn call returns a failed launch whose message includes yours.
- `platoon_fork` runs alone on the environment, and the child's tool calls take turns with the parent's on the
  same queue, so no two calls change the shared state at once.
- It works in process (`python://`) only.

The other fork modes (set with `agents.fork` in the config or `fork=` on the spawn call):

| Mode | In process | On an env server |
|---|---|---|
| `shared` (default) | `platoon_fork(task_spec)` when the class defines it and the child has a goal; otherwise the same environment object | the same remote session |
| `fresh` | a new environment for the same task | a new session on the same server |
| `copy` | `platoon_copy()` when the class defines it, else a deep copy (refused for a class with its own `teardown`) | not supported |

## Step 7: Check the environment without a model

[`check_env.py`](../../examples/custom-environment/check_env.py) opens the session a rollout would open and makes
the calls an agent would make, one of them from a subagent's forked session:

```python
service = get_service("python://order_packing:OrderPackingEnv")
session = await service.open(await service.get_task("train:0"), rollout_id="check-1")
child = await session.fork("shared", task=pt.Task(id="child-1", goal="Pack 2 apple"))
await child.call_tool("pack", {"item": "apple", "quantity": 2})
await child.call_tool("done", {"summary": "2 apple"})
print("child grade:", show(await child.grade()))
```

Run it from the example directory. `PYTHONPATH=.` lets platoon import the example's modules:

```bash
cd examples/custom-environment
PYTHONPATH=. uv run python check_env.py
```

Expected output (the SDK also logs a `tool_input_validation_error` warning for the call with quantity 0):

```text
splits: ['train', 'test']
test tasks: ['test:0', 'test:1']
instruction: Pack this order into the box: 2 apple, 1 bread.
tool done(summary): terminal
tool look(): read_only
tool pack(item, quantity): -
pack 0 apple: error input_validation
child: Packed 2 apple. The box holds 2 apple.
child grade: scored reward=1.0 reason=None
root: Packed 1 bread. The box holds 2 apple, 1 bread.
root: Shelf: 4 apple, 1 bread, 3 cheese, 12 egg, 2 milk. Box: 2 apple, 1 bread.
root grade: scored reward=1.0 reason=None
```

With openreward 0.1.125 the validation error's reason reads `invalid_call`.

## Step 8: Train on it

[`train.yaml`](../../examples/custom-environment/train.yaml) pairs the environment with the built-in
`platoon-react` harness, which shows the model the tools as native tool calls:

```yaml
environments:
  - name: packing
    service: python://order_packing:OrderPackingEnv
    harness: platoon-react
    dataset_kwargs: {split_name: train}
    eval_dataset_kwargs: {split_name: test}

train:
  model_name: Qwen/Qwen3-4B-Instruct-2507
  renderer_name: qwen3_instruct
  ...
```

`dataset_kwargs` selects the training tasks and `eval_dataset_kwargs` the evaluation tasks: `split_name` (a
split or a list of splits), `limit` (keep the first N) and `shuffle_seed` (shuffle before `limit`). The entry's
`exclude_tasks` leaves tasks out of both, by id (`train:3`) or by name ([Leave tasks
out](../how-to/train-and-eval.md#leave-tasks-out)).

1. Check the config and see what it selects:

   ```bash
   PYTHONPATH=. uv run platoon config validate train.yaml
   PYTHONPATH=. uv run platoon env list train.yaml
   ```

   ```text
   ok   train.yaml (PlatoonTinkerRLTrainerConfig)
   ```

   `platoon env list` ends with the environment table:

   ```text
   Environments of train.yaml:
     name     weight  service                                 harness        program  rollout
     packing  tasks   python://order_packing:OrderPackingEnv  platoon-react  -        harness_rollout
   ```

2. Run one task with the base model to see the agent's behavior (needs `TINKER_API_KEY`):

   ```bash
   PYTHONPATH=. uv run platoon rollout train.yaml --task train:0
   ```

   It prints the agent tree; ids and rewards vary from run to run:

   ```text
   rollout train-0-0260f242190c  task train:0
   agents 1  steps 3  max depth 0  reward 1.0  outcome scored reward=1.0
   └─ policy bcc81245  steps 3  reward 1.0  outcome scored reward=1.0
   ```

   The run directory (`runs/adhoc/...`) holds `trajectory_collection.json`, which
   `uv run platoon viz show-dump <file>` opens.

3. Train:

   ```bash
   PYTHONPATH=. uv run platoon train train.yaml
   ```

   Override any config value on the command line, for example `train.max_training_steps=20`.

## Step 9: Train with subagents

[`train_team.yaml`](../../examples/custom-environment/train_team.yaml) runs
[`packing_team.py`](../../examples/custom-environment/packing_team.py), a small function harness that gives the
root agent a `delegate(goal)` tool. Each delegate call runs `ctx.agents.spawn(goal)`, so the child packs on a
`platoon_fork` of the root's environment:

```yaml
environments:
  - name: packing
    service: python://order_packing:OrderPackingEnv
    harness: example-packing-team
    package: packing_team        # imported first, so @pt.harness("example-packing-team") registers
    agents:
      max_depth: 1               # the root (depth 0) may spawn; its children may not
      budget: depth_aware        # every agent has its own step budget
      child_max_steps: 6
      child_grading: env         # grade each child by its own forked session (its `done`)
```

```bash
PYTHONPATH=. uv run platoon rollout train_team.yaml --task train:0
```

A run where the root delegates both items prints a tree like this:

```text
rollout train-0-39152c09904e  task train:0
agents 3  steps 6  max depth 1  reward 1.0  outcome scored reward=1.0
└─ policy 03439af5  steps 2  reward 1.0  outcome scored reward=1.0
   ├─ subagent b2d7440f  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Pack 2 apple'
   └─ subagent cdd105e2  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Pack 1 bread'
```

Every agent's trajectory trains with its own reward: each child with its forked session's grade, the root with
the whole box's.

## Reference: what platoon reads from an environment class

All attributes are optional. The `platoon_*` attributes and methods work in process (`python://`); an env
server sends tool names, descriptions, schemas and `@terminal`, nothing else. A remote service that names the
class (`env_class`) reads the hint attributes from it as in process, but the methods (`platoon_fork`,
`platoon_copy`, `platoon_grade`) need the live environment and run only in process.

| Attribute or method | Effect |
|---|---|
| `list_splits()`, `list_tasks(split)` | The splits and their task specs. |
| `get_prompt()` | The instruction: its text blocks, joined (an image block becomes an `[image: <type>]` placeholder). |
| `setup()`, `teardown()` | Run after construction and when the session closes. Sync or async. |
| `@tool` methods | The tools. Sync methods run on a thread of their own. |
| `platoon_terminal_tools` | Tool names that end the episode (step 4). |
| `platoon_read_only_tools` | Tool names that never change state (step 5). |
| `platoon_concurrent_tools` | Tool names that may run while other calls run ([concurrency](remote-environments.md#tool-call-concurrency)). |
| `platoon_tool_meta` | `{tool: {...}}`, passed to MCP clients as the tool's `_meta`. |
| `platoon_grade()` | A grade from state when no tool finished the episode (step 4). |
| `platoon_fork(task_spec)` | A child's environment on shared state (step 6). |
| `platoon_copy()` | The environment a `copy` fork gets. |

Tool results can carry these `metadata` keys:

| Key | Effect |
|---|---|
| `"invalid": true` (on the finishing result) | The grade is `invalid`, not trained. |
| `"openhands.dev/tool-error": {"message": "..."}` | The result is an error result (`is_error`), with that message. |
| `"structured_content": value` | `platoon-codeact` cells get `value` back instead of the text. |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'order_packing'`.** Run from the example directory with
  `PYTHONPATH=.`, or install your environment as a package. Rollout worker processes import it too.
- **`... has no split 'val'`.** The config's `split_name` must be one of `list_splits()`.
- **Subagents can call `done` and end the parent's episode.** Define `platoon_fork`, or give children
  `agents.child_access: no_terminal`.
- **Every rollout scores 0 with reason `no_submit`.** No tool returned `finished=True`. Check that the terminal
  tool is declared, so harnesses route the final answer into it.

## Next steps

- [Remote environments](remote-environments.md): serve this environment from env servers with `ors+http://`,
  and tune the tool-call concurrency, call caps and result formats.
- [Gym adapters and task tables](gym-and-task-tables.md): turn a reset/step environment into a harness or an ORS
  environment, and train on plain task lists.
- [The architecture overview](../architecture.md): sessions, outcomes and the agent tree in depth.
- [Write an environment service](../extending/environment-services.md): tasks from a source that is not an ORS
  environment.
- To propose a change to the environment contract itself, write an RFC: [the RFC process](../rfcs/README.md).
