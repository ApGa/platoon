# Write an environment service

**Goal.** Serve tasks to platoon from a source that is not an ORS environment, a Harbor dataset or a task table: a
service of your own that lists tasks, opens one session per rollout, offers tools, grades and forks. You will write
a small service, select it in a config with `service: module:Class`, and check it on CPU.

Most environments do not need this. An [ORS `Environment` class](../tutorials/custom-environment.md) gives you
tasks, tools, grading and forks with less code and runs in process or on [env
servers](../tutorials/remote-environments.md); [task tables and gym adapters](../tutorials/gym-and-task-tables.md)
cover tasks without tools and reset/step code. Write a service when tasks come from a system with its own protocol
(an evaluation server, a simulator with its own API) or when sessions need control that an ORS class does not
give, such as a sandbox per session that CLI harnesses run in.

**Prerequisites.**

- Platoon installed (`pip install 'platoon[tinker,ors]'`, or `uv sync --extra tinker --extra ors` in a checkout).
- [Concepts](../getting-started/concepts.md): sessions, outcomes and forks.

## The contract

A service is any object with a `name` and three async methods; `platoon.envs.service` defines both protocols:

```python
class EnvironmentService(Protocol):
    name: str

    async def list_tasks(self, split: str) -> list[str]: ...  # the task ids of a split
    async def get_task(self, task_id: str) -> Task: ...  # one task, by id
    async def open(self, task: Task, *, rollout_id: str) -> EnvSession: ...  # a session for one rollout


class EnvSession(Protocol):
    task: Task
    instruction: str  # what the root agent works on
    sandbox: Sandbox | None  # a task sandbox that CLI harnesses can run in, or None

    async def tools(self) -> list[ToolSpec]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
    async def grade(self) -> Outcome: ...
    async def fork(self, mode: ForkMode = "shared", *, task: Task | None = None) -> EnvSession: ...
    async def close(self) -> None: ...
```

Two more async methods are optional: `list_splits()`, the service's split names, and `list_task_names(split)`, each
task's name by task id, for a service whose task names are not its task ids. An entry's `exclude_tasks` matches
task names through `list_task_names`, and looks for an item that the listed splits do not hold in the other splits
that `list_splits` names. Without them, each item must be the id of a task in the splits the run lists
([Leave tasks out](../how-to/train-and-eval.md#leave-tasks-out)). A service that can tell before any session opens
that it would refuse some tasks may also have `check_tasks(split, task_ids)`, async too: it raises an
`EnvSetupError` with failure kind `env_config` for the tasks of the split that the run keeps once `exclude_tasks`
left some out, which stops the run at its start with exit 78 (the Harbor service refuses tasks without a network on
a provider that would never start them).

Two synchronous methods, optional as well, inform the deadlines of whole rollouts (a rollout worker's with
`rollout_isolation: process`, an evaluation slot's with `use_subprocesses`): `rollout_overhead_s(task_id)`, the
seconds a rollout of the task may take beyond its harness's time limit, such as starting a sandbox and running a
verifier, which those deadlines add; and `agent_timeout_s(task_id)`, the task's own time limit for its agent, or
`None`, which they count in place of a `rollout_config.timeout` that the config leaves unset. A session that has
such a limit also gives it as its `agent_timeout_s` attribute, and harness rollouts stop the harness there. The
Harbor service has both. Without `agent_timeout_s`, a deadline counts only the time limit of the harness's spec
(`timeout_s`). Without `rollout_overhead_s`, platoon cannot know how long opening, grading and closing a session
take, so a deadline counts a harness's own limit in place of a missing `timeout` only with it. A rollout that
nothing limits, or whose service has no `rollout_overhead_s`, gets a fallback: a day for a worker
(`worker_unbounded_rollout_budget_seconds`), 900 s for an evaluation slot, or the harness's own limit when that is
longer.

How platoon uses it:

| When | Where | Calls |
|---|---|---|
| A run starts | the trainer or evaluator | `list_tasks(split)` for each split the config selects (`dataset_kwargs`, `eval_dataset_kwargs`); with `exclude_tasks`, also `list_task_names(split)` and `list_splits()` where the service has them; then `check_tasks(split, task_ids)` where it has one |
| Each rollout | the process that runs the rollout | builds the service from the config, then `get_task(task_id)` and `open(task, rollout_id=...)` |
| The episode | the same process | `tools()` and `call_tool(...)` for harnesses that act through the environment, `fork(...)` for each subagent, `grade()` once the root is done (and for each subagent graded with `child_grading: env`) |
| The end of the rollout, on every path | the same process | `close()` on every session, forks included |

Every rollout builds its own service object, with the entry's `service_kwargs` as keyword arguments, so keep the
constructor cheap and cache anything expensive (a loaded dataset, a client) at module level.

## Steps

### 1. Write the service

This service asks the agent to submit one word, and offers one terminal tool:

```python
# my_services/words.py
from typing import Any

import platoon as pt

WORDS = {"train": ["apple", "river", "stone"], "test": ["cloud"]}
SUBMIT = pt.ToolSpec(
    name="submit",
    description="Submit one word. This ends the episode.",
    input_schema={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
    terminal=True,
)


class WordService:
    """Tasks `<split>:<index>`, each asking for one word; sessions grade the submitted word."""

    def __init__(self, *, words: dict[str, list[str]] | None = None) -> None:
        self.name = "words"
        self.words = words or WORDS

    async def list_tasks(self, split: str) -> list[str]:
        if split not in self.words:
            raise ValueError(f"no split {split!r}; the splits are {sorted(self.words)}")
        return [f"{split}:{index}" for index in range(len(self.words[split]))]

    async def get_task(self, task_id: str) -> pt.Task:
        split, _, index = task_id.partition(":")
        word = self.words[split][int(index)]
        return pt.Task(id=task_id, goal=f"Call submit with the word {word!r}.", misc={"word": word})

    async def open(self, task: pt.Task, *, rollout_id: str) -> "WordSession":
        return WordSession(task)


class WordSession:
    sandbox = None  # no task sandbox: CLI harnesses run on the rollout host

    def __init__(self, task: pt.Task) -> None:
        self.task = task
        self.instruction = task.goal or ""
        self.submitted: str | None = None

    async def tools(self) -> list[pt.ToolSpec]:
        return [SUBMIT]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> pt.ToolResult:
        if name != "submit":
            return pt.ToolResult.text(f"Unknown tool {name!r}.", is_error=True)
        if self.submitted is not None:
            return pt.ToolResult.text("The episode is already over.", is_error=True)
        self.submitted = str(arguments.get("word", "")).strip()
        correct = self.submitted == self.task.misc["word"]
        result = pt.ToolResult.text("Correct." if correct else "Wrong.")
        result.finished, result.reward = True, float(correct)
        return result

    async def grade(self) -> pt.Outcome:
        if self.submitted is None:  # a policy failure: scored, with a reason
            return pt.Outcome(status="scored", rewards={"reward": 0.0}, reason="no_submit")
        return pt.Outcome(status="scored", rewards={"reward": float(self.submitted == self.task.misc["word"])})

    async def fork(self, mode: pt.ForkMode = "shared", *, task: pt.Task | None = None) -> "WordSession":
        if mode != "fresh":  # refuse what the service cannot do; never quietly give another mode
            raise ValueError(f"words sessions fork only 'fresh', not {mode!r}")
        return WordSession(task or self.task)

    async def close(self) -> None:
        pass  # nothing to release; must be safe to call twice
```

### 2. Check it without a model

Drive it the way a rollout does:

```python
# check_words.py
import asyncio

from platoon.envs import get_service


async def main() -> None:
    service = get_service("my_services.words:WordService")
    print(await service.list_tasks("train"))
    session = await service.open(await service.get_task("train:1"), rollout_id="check")
    print(session.instruction)
    result = await session.call_tool("submit", {"word": "river"})
    print(result.as_text(), result.finished, result.reward)
    print(await session.grade())
    await session.close()


asyncio.run(main())
```

```bash
PYTHONPATH=. uv run python check_words.py
```

```text
['train:0', 'train:1', 'train:2']
Call submit with the word 'river'.
Correct. True 1.0
Outcome(status='scored', rewards={'reward': 1.0}, primary='reward', reason=None, failure_kind=None, metadata={})
```

To play it with a harness and a scripted policy, end to end as a rollout plays it, pass
`service="my_services.words:WordService"` to `platoon.testing.run_episode` ([Test an environment, a harness or a
program on CPU](../how-to/test-extensions.md#2-test-an-environment)).

### 3. Select it in a config

`service:` takes the class as `module:Class`, and `service_kwargs:` are its constructor's keyword arguments:

```yaml
environments:
  - name: words
    service: my_services.words:WordService
    service_kwargs:
      words: {train: [apple, river, stone], test: [cloud]}
    harness: platoon-react
    agents: {fork: fresh}          # the only fork mode the service offers
    dataset_kwargs: {split_name: train}
    eval_dataset_kwargs: {split_name: test}
```

The module must be importable in every process that runs rollouts: install your package, or put its directory on
`PYTHONPATH`. `platoon env list CONFIG` shows the entry, and `platoon rollout CONFIG --task train:0` runs one task.
`service:` also takes a function that returns a service, called with `service_kwargs`; the task tables'
`task_table(...)` is used that way.

A service whose options name files lists them in a class attribute, `path_options`, so that a config may give them
relative to itself, as it gives the built-in services' (a Harbor `dataset_dir`, a table's `path`): the config loader
makes such a path absolute against the directory of the config file that names it, and refuses one that names
nothing there, before the service is built.

```python
from typing import ClassVar

from platoon.envs.service import PathOption


class FileWordService:
    # A tuple of option names, or a mapping to PathOption: relative_to names another option whose directory the path
    # is relative to when that one is set, and written=True lets the path name a file the run writes later.
    path_options: ClassVar[dict[str, PathOption]] = {"word_file": PathOption(), "shards[].file": PathOption()}
    ...
```

`shards[].file` names the `file` key of each mapping in the list `shards`. An option that `path_options` leaves out
reaches the constructor as the config wrote it.

## The rules

**Grade honestly.** `grade()` never raises. Its outcome says whether the measurement is valid, not whether the
policy succeeded:

| Situation | Return |
|---|---|
| The policy finished, well or badly | `pt.Outcome(status="scored", rewards={"reward": r})`, with a `reason` when it helps (`no_submit`) |
| The environment cannot measure this episode (a missing answer key, a grader that declared itself unusable) | `pt.Outcome.infra("my_failure", status="invalid", reason=...)` |
| Something broke: a dead connection, a grader that crashed | `pt.Outcome.infra("my_failure", reason=...)`: status `error` |

Only `scored` outcomes train; an `invalid` or `error` outcome never becomes a reward of 0. A setup failure in
`open()` raises `platoon.envs.EnvSetupError(failure_kind, message)`: the rollout fails, so training drops it and
evaluation counts it as errored. Raised by `list_tasks` with `failure_kind="env_config"`, it stops the run before it
starts, as a configuration error (exit code 78).

**Tools.** A `ToolSpec` carries the tool's name, description and JSON schema. Mark the tool that ends the episode
`terminal=True` and tools that never change state `read_only=True`: agents with `no_terminal` or `read_only` access
are shown tools by these flags, and `platoon-react` and `platoon-codeact` route the final answer into the terminal
tool. A `ToolResult` holds MCP-style content blocks; set `finished` and `reward` on the result that ends the episode.
Report the policy's mistakes as error results (`is_error=True`) instead of raising. A service whose sessions always
serve the same tools may name them in a class attribute (experimental), `declared_tools` (a tuple of names), so that
loading warns of an agent role's tool pattern that matches none of them ([Limit the tools each role may
call](../how-to/train-and-eval.md#limit-the-tools-each-role-may-call-experimental)); an ORS environment run in process
declares its class's tools by itself.

**Forks.** A subagent's session is `await session.fork(mode, task=child_task)`:

| Mode | The child gets |
|---|---|
| `shared` | a handle on the same live state, so the child's actions change the parent's state |
| `copy` | an independent copy of the current state |
| `fresh` | a new session of the task, from the start |

Implement the modes your environment can provide and raise for the others: the spawn then fails with your message,
which the parent agent sees. `agents.fork` on the entry sets the mode its subagents use.

**Concurrency.** Platoon gives a session one call at a time, from all the agents that share it, unless the session
sets `manages_concurrency = True`; it then takes calls from any number of agents at once and orders them itself.

**Sandboxes.** A session with a task sandbox (`sandbox` set to a `platoon.sandbox.Sandbox`) lets CLI harnesses run
inside it (`placement: auto`). If that sandbox has no network, set `network_isolated = True` on the session: a harness
inside it could not reach the model gateway, so platoon refuses to start one there instead of failing late.

**Closing.** `close()` is idempotent. Platoon calls it for the root's session and every fork, on every path,
including cancellation.

## Next steps

- [Write a sandbox provider](sandbox-providers.md): sandboxes for sessions that run harnesses inside them.
- [Write a custom environment](../tutorials/custom-environment.md): the ORS route, which most environments take.
- [Extending platoon](index.md): every extension point and how platoon finds it.
