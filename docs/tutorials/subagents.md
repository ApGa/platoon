# Subagents: agents that delegate

**Goal.** Build an agent that hands parts of its task to subagents with `ctx.agents.spawn`, run one task and read
the agent tree it produces, and learn the settings that control subagents: fork modes, tool access, step budgets,
how each subagent is graded, verifiers and judges.

**Prerequisites.**

- Platoon installed with the `tinker` and `ors` extras (`uv sync --extra tinker --extra ors`), and a function
  harness you understand (the README quickstart's `example-guesser` is enough).
- The example used throughout: [`examples/subagents/`](../../examples/subagents/).
- For the run in step 4, a Tinker API key (`TINKER_API_KEY`). Step 6 runs the same example on CPU without one.

[`docs/architecture.md`](../architecture.md) explains the design behind agents, episodes and the model gateway.

## The example

The Vault environment stores a number under each of twelve keys and asks for the sum under four to eight of them.
An agent can read every key itself, or delegate: split the keys, give each half to a subagent, and add up their
answers.

| File | What it is |
|---|---|
| `vault_env.py` | `VaultEnvironment`, an ORS environment: `read(key)`, the terminal `answer(total)`, and `platoon_fork` |
| `delegator.py` | `@pt.harness("example-delegator")`: a tool-calling loop with an extra `delegate(goal, max_steps)` tool |
| `subagents.yaml` | The environment entry with its `agents:` settings, and a training section |
| `tests/examples/test_subagents_example.py` | Runs the config end to end with a scripted policy (no GPU, no Tinker) |

## Steps

### 1. Spawn subagents from the harness

`ctx.agents.spawn(goal)` runs a child agent to completion and returns a `pt.SubagentResult`. The `example-delegator`
harness offers the model a `delegate` tool and turns each call into a spawn. The calls of one reply run with `asyncio.gather`, so
two `delegate` calls in one reply run two subagents at once:

```python
# examples/subagents/delegator.py (abridged)
@pt.harness("example-delegator")
async def delegator(ctx: pt.AgentContext) -> pt.AgentResult:
    tools = [openai_tool(tool) for tool in await ctx.list_tools()] + [DELEGATE_TOOL]
    messages = [{"role": "user", "content": ctx.instruction}]
    steps = 0
    while not out_of_steps(ctx, steps):
        reply = await ctx.chat(messages, tools=tools)
        steps += 1
        messages.append({key: value for key, value in reply.message.items() if value is not None})
        if not reply.tool_calls:
            return pt.AgentResult("completed", final_message=reply.text)
        results = await asyncio.gather(*(run_call(ctx, call) for call in reply.tool_calls))
        ...


async def run_call(ctx, call):
    ...
    if function.get("name") == "delegate":
        child = await ctx.agents.spawn(str(arguments.get("goal", "")), max_steps=arguments.get("max_steps"))
        if child.ok:
            return child.message, False
        return f"The subagent did not finish ({child.result.status}): {child.message}", False
    result = await ctx.call_tool(str(function.get("name", "")), arguments)
    return result.as_text(), result.finished
```

The child runs the same harness unless the spawn names another, with the same options, on a fork of the
parent's environment session. Every argument of `spawn` is optional except the goal:

| Argument | Default | Meaning |
|---|---|---|
| `goal` | (required) | The child's instruction (`ctx.instruction` in the child) |
| `harness` | the parent's | A registered function harness; TOML-spec harnesses cannot be spawned (`TypeError`) |
| `max_steps` | `agents.child_max_steps` | The child's step budget: a positive integer, else the launch is rejected |
| `fork` | `agents.fork`, then the session's default | `shared`, `copy` or `fresh` (see [Fork modes](#fork-modes)) |
| `access` | `agents.child_access` | `full`, `no_terminal` or `read_only`, never more than the parent's |
| `role` | `"subagent"` | The role recorded on the child's trajectory |
| `options` | the parent's options (same harness), else `{}` | The child's `ctx.options` |
| `task_misc` | the parent task's misc | The child task's `misc` |

A child never raises into its parent. What comes back:

| `SubagentResult` field | Meaning |
|---|---|
| `result` | The child's `AgentResult`: `status` is `completed`, `limit`, `timeout`, `error` or `cancelled` |
| `message` | The child's final message, or a failure text (a rejected launch says why) |
| `outcome` | The child's own grade (`pt.Outcome`), or `None` when it was not graded |
| `ok` | The child ran and completed |
| `launched` | False when the launch was rejected (depth, budget, fork failure); then no agent ran |
| `agent_id`, `trajectory_id`, `goal` | Identify the child's record in the trajectory tree |

The parent's step that spawned children records `reward/subagent_launched` and `reward/subagent_succeeded` in its
`reward_misc`; harnesses do not record them again.

### 2. Give each subagent its own goal on the shared environment

A `shared` fork gives the child the parent's live environment. On its own, that means the child plays the parent's
whole episode: an `answer` it submits ends the episode for both. An ORS environment can define
`platoon_fork(task_spec)` to return the child's own environment on shared state. The Vault does, so each subagent
answers, and is graded on, just its keys:

```python
# examples/subagents/vault_env.py
def platoon_fork(self, task_spec):
    """A subagent's environment: the same vault, and the keys its goal names."""
    return type(self)({"vault": self.vault, "keys": parse_keys(task_spec.get("goal"))})
```

`task_spec` is the parent's task spec with the child's `goal`, so read the subgoal from the goal, not from fields
that still describe the parent's task.

### 3. Configure the tree

Subagent settings live under `agents:` in the environment entry:

```yaml
# examples/subagents/subagents.yaml
environments:
  - name: vault
    service: python://vault_env:VaultEnvironment
    harness: example-delegator
    package: delegator            # imported first, so @pt.harness("example-delegator") registers
    agents:
      max_depth: 1                # the root (depth 0) may delegate; its subagents may not
      child_max_steps: 6          # a subagent's budget when delegate names none
      child_grading: env          # each subagent is graded by its own answer (VaultEnvironment.platoon_fork)
      fork: shared                # subagents work on the parent's live environment
```

| `agents.` key | Default | Meaning |
|---|---|---|
| `max_depth` | `null` (no limit) | Deepest level a child may sit at; the root is depth 0, so `0` allows no subagents. Verifiers do not count against it (see [Step budgets](#step-budgets)) |
| `budget` | `linear` | `linear` or `depth_aware` (see [Step budgets](#step-budgets)) |
| `child_max_steps` | `15` | Step budget of a child whose spawn names none |
| `child_grading` | `none` | `env`, `verifier` or `none` (see [How subagents are graded](#how-subagents-are-graded)) |
| `child_access` | `full` | Tool access of children (see [Tool access](#tool-access)) |
| `fork` | `null` | Fork mode of children; `null` uses the session's default (`shared` for ORS and Harbor) |
| `child_timeout_s` | `null` | Wall-clock limit of each child run; `null` leaves only the rollout's `timeout` |
| `verifier.enabled` | `true` | Run verifier agents when `child_grading: verifier` |
| `verifier.max_steps` | `20` | Step budget of each verifier agent |
| `verifier.access` | `read_only` | Tool access of verifier agents |

Check the file before you run it:

```bash
export PYTHONPATH=$PWD/examples/subagents
uv run platoon config validate examples/subagents/subagents.yaml
```

```text
ok   examples/subagents/subagents.yaml (PlatoonTinkerRLTrainerConfig)
```

### 4. Run one task and read the tree

`platoon rollout` runs one task through the configured route with the training model served through Tinker, and
prints the agent tree:

```bash
uv run platoon rollout examples/subagents/subagents.yaml --task train:0
```

With a policy that delegates the way the scripted test policy does, the tree looks like this (ids differ per run;
a base model may not delegate at all before training):

```text
rollout train-0-bb1cfc4f4079  task train:0
agents 3  steps 6  max depth 1  reward 1.0  outcome scored reward=1.0
└─ policy e6b4dc2d  steps 2  reward 1.0  outcome scored reward=1.0
   ├─ subagent f2976f3a  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys amber,...'
   └─ subagent 99a74c51  steps 2  reward 1.0  outcome scored reward=1.0  goal 'Report the sum of the values stored under the keys fjord,...'
```

Each line is one agent and one trajectory: the root (`policy`) and two subagents, each with its own grade. The
run directory (default `runs/adhoc/<time>-<task>`, or `--output-dir`) holds `trajectory_collection.json`, which
`uv run platoon viz show-dump <run dir>/trajectory_collection.json` opens in the trajectory viewer.

### 5. Try other settings

Each override below changes one setting of the same run (add it after the config path):

| Override | What changes |
|---|---|
| `environments.0.agents.child_grading=none environments.0.agents.child_access=no_terminal` | Subagents cannot see `answer`, report in text, and take the root's reward |
| `environments.0.agents.child_grading=verifier environments.0.agents.verifier.max_steps=4` | A read-only verifier agent judges each subagent; the tree shows a `verifier` under each |
| `environments.0.agents.fork=fresh` | Each subagent gets a new Vault of the whole task; a half-sum answer scores 0 there |
| `environments.0.agents.fork=copy` | Each subagent gets a copy of the parent's environment, also the whole task |
| `environments.0.agents.max_depth=0` | Every `delegate` call is rejected; the model reads the rejection and can do the work itself |
| `environments.0.agents.budget=depth_aware` | Subagents get their own budgets instead of drawing on the root's |

### 6. Run the example on CPU

The example's test runs `platoon rollout` on the config with a scripted policy in place of the Tinker model, once
per setting in step 5:

```bash
uv run pytest tests/examples/test_subagents_example.py -q
```

```text
......                                                                   [100%]
6 passed in ...s
```

The tests need no model download: the gateway renders with `platoon.testing`'s toy tokenizer
([Test an environment, a harness or a program on CPU](../how-to/test-extensions.md)).

### 7. Train

```bash
uv run platoon train examples/subagents/subagents.yaml
```

Every agent's trajectory trains: the root on the episode's reward, each subagent on its own grade.
[Credit assignment](credit-assignment.md) explains how those rewards become advantages, and how to change that.

## Reference

### Fork modes

A child's environment session is a fork of its parent's:

| Mode | The child gets | Grading the child's session grades |
|---|---|---|
| `shared` | The parent's live environment; tool calls of all agents take turns on it | The parent's episode, unless the env's `platoon_fork(task_spec)` gives the child its own goal |
| `copy` | A snapshot of the parent's environment, restored into a new session | The copy: the parent's task, from the parent's current state |
| `fresh` | A new session of the same task | The whole task, from the start |

`shared` suits decomposition (with `platoon_fork`, as in the Vault) and helpers that only look. `copy` suits trying
something out without touching the parent's state; ORS environments support it in process (`python://`), by deep
copy or an env's `platoon_copy()`, and Harbor tasks with a sandbox provider that can snapshot. `fresh` suits
independent attempts at the whole task.

### Tool access

| Access | The agent sees |
|---|---|
| `full` | Every tool |
| `no_terminal` | Every tool except terminal ones (the tool that submits and ends the episode) |
| `read_only` | Only tools that declare themselves read-only |

A child's access is the spawn's `access` or `agents.child_access`, clamped to its parent's. An ORS environment
declares its tools with class attributes, as the Vault does (`platoon_terminal_tools = ("answer",)`,
`platoon_read_only_tools = ("read",)`), with the SDK's `@terminal`, or with the service options `terminal_tools`
and `read_only_tools`. A call to a tool the agent cannot see returns an error result and never reaches the
environment.

### Step budgets

A step is one `policy` model call. `ctx.budget.remaining()` and `ctx.budget.used()` read the agent's budget.

- **`linear`** (default): a child draws on its parent's budget. Each spawn reserves the child's `max_steps + 1` on
  the parent until the child returns (the extra step is the parent's turn to read the result), and the parent's
  used steps include its descendants'. A spawn the parent cannot afford is rejected with a message that suggests a
  smaller budget. In the example, the root has 20 steps and each subagent 6, so two concurrent subagents reserve 14
  of the root's 19 remaining steps; a third would be rejected.
- **`depth_aware`**: every agent is bounded by its own `max_steps` only, and spawns reserve nothing.

`max_depth` caps the policy tree in both modes. Concurrent spawns are safe: each reservation is atomic.

Verifiers are not counted against `max_depth`: a verifier judges one child, so it must be able to run below a child
at `max_depth`, or the deepest children could never be verified and trained. A verifier runs one level below the
child it judges and may launch one level of helpers, so with `child_grading: verifier` the tree reaches
`max_depth + 2`. What bounds those levels is `agents.verifier.max_steps` (with `linear` budgets the helpers draw on
the verifier's steps), each helper's own `max_steps` with `depth_aware` budgets, `agents.child_timeout_s` and the
rollout's timeout; they can take much of a step's wall time.

Verifier agents count too: with `linear` budgets, a verifier's `agents.verifier.max_steps + 1` is reserved on the
parent that spawned the judged child, so keep it inside the parent's budget (the default of 20 never fits a
parent with 20 steps) or use `depth_aware`. A verifier that cannot launch leaves its child with a `judge_error`
verdict, which keeps the child out of policy training.

### How subagents are graded

`agents.child_grading` decides what reward a subagent's trajectory carries:

| Value | The child's reward |
|---|---|
| `env` | The grade of its own forked session (`session.grade()`) |
| `verifier` | The verdict of a verifier agent (below) |
| `none` | The episode's outcome, shared with the root |

`env` grades the child's own goal only where its forked session grades that goal. On a `shared` fork, the default
of every built-in environment, the child's session is its parent's live environment: an in-process ORS class's
`platoon_fork(task_spec)` gives it a view with its own goal and terminal call, and without one the child would be
graded by the episode's first submission, or score 0 (`no_submit`), whatever its goal. A child on a fork of its own
(`fresh`, or `copy`) is graded on the original task, not on its goal. So `platoon train`, `eval` and `rollout`
refuse `child_grading: env`, with exit code 78 before they list any task, where it cannot grade a child's own goal:
on a task table (no grader), and on shared forks of an env server (`ors+http://`, which cannot run the hook), of a
Harbor sandbox (graded by the task's whole `test.sh`), or of an in-process class without `platoon_fork`. An entry
with `agents.max_depth: 0` has no children to grade, so it is not refused.

Whatever the setting, a child whose harness crashed, timed out or was cancelled gets an untrainable outcome instead
of a grade: it stays in the tree and the statistics but trains no tokens, and never counts as reward 0. A child
that crashed after filling its context window is still graded, with reason `context_overflow`, unless the
environment sets `on_context_overflow: exclude`.

### Verifiers and judges

With `child_grading: verifier`, every completed child is judged before its parent sees the result. A verifier
agent runs the child's harness on a fork of the child's session, with `agents.verifier.access` (default
`read_only`) and `agents.verifier.max_steps`, and a goal that asks for a JSON verdict
(`{"status": "verified" | "partial" | "failed" | "insufficient_evidence", "score": 0.0-1.0, ...}`) as its final
message. Only a finished verifier with a known status and a consistent score gives a training-eligible verdict;
otherwise the child is kept out of policy training. Verifier trajectories never train.

With `read_only` access a verifier sees only the tools declared read-only: `platoon_read_only_tools` or a
`readOnlyHint` in process, the `read_only_tools` or `tool_meta` options for an env server, and none of Harbor's
generic tools. A verifier that sees no tools judges from the child's transcript alone; platoon logs a warning once
per environment when that happens. Set
`train.workflow_config.rollout_config.enable_subagent_verifiers: false` (or `agents.verifier.enabled: false`) to
switch verifiers off, for example in evaluation, where the task is scored at its root; children are then ungraded
and share the episode's outcome.

Two more kinds of judge exist:

- **Behavior judges** audit a verified child's behavior and gate its score. An episode program sets one before it
  runs the root: `ep.behavior_judge = MyJudge()`, where `MyJudge` has
  `async def judge(self, *, goal, trajectory) -> dict` returning
  `{"status": "pass" | "fail" | "insufficient_evidence", "passed": True | False | None, "reason": "..."}`
  (see [Episode programs](programs.md)).
- **Judge calls**: `await ctx.chat(messages, kind="judge")` makes a model call that is recorded under the
  trajectory's `judge_calls` for audit and never trained, and does not use the agent's budget. Programs get a
  whole judge endpoint with `ep.judge_endpoint()`.

### Concurrency and cancellation

- Spawns may run concurrently (`asyncio.gather`); tool calls of agents that share a live environment take turns on
  it.
- Cancelling a spawn call (for example with `asyncio.wait_for`) cancels the child and gives its budget back.
- A spawn still running when the harness returns (say an un-awaited `asyncio.gather`) keeps its child: the parent
  waits for it to finish and be graded.
- `ctx.agents.keep_children()` makes cancelled spawn calls keep their children too, for a harness that cancels its
  pending calls on the way out but wants the children to finish. The built-in `platoon-codeact` harness does this
  before it closes its cell loop.
- When an agent is cancelled or times out, its running children are cancelled; children that already finished
  keep their grades.

## Next steps

- [Credit assignment](credit-assignment.md): how each agent's reward becomes a training signal.
- [Episode programs](programs.md): run custom logic around the root agent, set a behavior judge, shape rewards.
- [Judges and verifiers](../extending/judges-and-verifiers.md): the verdicts verifiers return, and behavior
  and parent-usage judges of your own.
- The built-in `platoon-codeact` harness offers `launch_subagent` to the model's Python cells; its
  `subagents` and `subagent_budget` options control it.
